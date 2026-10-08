# Average-Reward Reinforcement Learning for Multichain MDPs: A Hierarchical Decomposition Approach<sup>†</sup>

Huizhen Yu Isaiah Heidt<sup>\*</sup>

Department of Computing Science University of Alberta, Canada

## Abstract

We study learning optimal policies in average-reward multichain Markov decision processes (MDPs), where the optimal gain may depend on the initial state and recurrence structures vary across policies, creating challenges for reinforcement learning (RL) methods. We propose an asynchronous value-iteration-based RL algorithm that requires no model knowledge beyond the MDP’s transition graph and leverages Bather’s decomposition to hierarchically partition the state space into communicating subsystems and transient states. This decomposition induces a recasting of the global decision problem into structured subproblems, which our algorithm exploits. We show that the algorithm converges to the optimal gain and produces gain-optimal policies after finite time. Building on this base algorithm, we develop two further algorithms: one approximately solves the multichain average optimality equations to obtain near gainoptimal policies, and another targets near bias-optimality by approximating the optimal bias function and solving an induced average-reward multichain MDP using the base algorithm. We provide almost-sure convergence guarantees for all three algorithms and empirically compare their tradeofs, showing that the latter two also consistently improve transient performance relative to the base algorithm. To our knowledge, these are the first essentially model-free average-reward RL algorithms for general multichain MDPs without reductions to discounted problems.

Keywords: average-reward reinforcement learning; multichain Markov decision processes; hierarchical decomposition; asynchronous stochastic value iteration; gain and bias optimality; stability and convergence

## 1 Introduction

Reinforcement learning (RL) methods for Markov decision processes (MDPs) under the averagereward criterion are important for sequential decision problems where optimizing long-run performance is essential. In this paper, we study learning optimal policies in finite-space multichain MDPs and propose new asynchronous RL methods that do not require model knowledge beyond the MDP’s transition graph—the directed graph indicating which states can be reached from each state under each action.

A multichain MDP represents the most general type of MDP structure, arising when a stationary policy can induce more than one recurrent class: the process may settle into diferent regions of the state space, and some may be better than others. This is not only a matter of the decisions we make but a question of the fundamental structure of the problem we are solving. In a communicating or weakly communicating MDP, mistakes can always be undone: outside of transient behavior, every state remains reachable from every other, so even a poor decision can be corrected by a policy that steers the process back out. Multichain structure that is not weakly communicating breaks this guarantee: the state space can fragment into regions that once exited, cannot be revisited, or that once entered, cannot be exited, so a policy may commit the process to a suboptimal long-run regime irrevocably, with no path back. Such irreversibility is pervasive in real-world stochastic systems: an agent interacting with its environment may make decisions that permanently alter the options available to it in the future. Descending a steep slope may leave it unable to climb back or taking a risky route may cause irreversible damage to its components. In this sense, essentially any real-world system with efectively irrevocable decisions represents a multichain problem, showing the importance of algorithms specifically designed to tackle them.

The classical theory of average-reward multichain MDPs is well understood. Let $\mathcal { M } = ( \boldsymbol { S } , \boldsymbol { A } , \boldsymbol { p } , \boldsymbol { r } )$ be a finite-state, finite-action MDP with state space $s ,$ action constraint set $\mathcal { A } = \{ \mathcal { A } ( s ) \} _ { s \in \mathcal { S } } ,$ transition function $p ,$ and reward function r. A classical approach to computing gain-optimal (or average-reward optimal) policies is via the average-reward optimality equations (AOE), which characterize optimal solutions via a pair of nested equations, given below in vector notation (cf. [22, Chap. 9]):

$$
g = \operatorname* { m a x } _ { \pi \in \Pi _ { \mathcal { A } } } \{ P _ { \pi } g \} , \qquad g + h = \operatorname* { m a x } _ { \pi \in \Pi _ { \mathcal { A } _ { * } } } \{ r _ { \pi } + P _ { \pi } h \} .\tag{1.1}
$$

Here, we solve for $( g , h ) \in \mathbb { R } ^ { | S | } \times \mathbb { R } ^ { | S | } ; \Pi _ { A }$ denotes the set of nonrandomized stationary policies satisfying the action constraint $\mathcal { A } ;$ and $\Pi _ { \boldsymbol { A } _ { * } }$ is defined analogously for an action constraint $A _ { * } \subset A$ induced by the first optimization problem. Specifically, $\mathcal { A } _ { \ast } = \{ \mathcal { A } _ { \ast } ( s ) \} _ { s \in \mathcal { S } }$ , where $\mathcal { A } _ { * } ( s ) \subset \mathcal { A } ( s )$ consists of actions that attain the maximum in the first equation for state s. The AOE (1.1) has infinitely many solutions, all of the form $( g ^ { * } , h )$ , where $g ^ { * }$ is the optimal gain vector of the MDP. (For the solution structure of the h-component, see [26].) For any AOE solution, any $\pi ^ { * } \in \Pi _ { A _ { } }$ that attains the maximum in the second equation of (1.1) for all states is gain-optimal.

This nested optimization problem is dificult to solve with existing model-free RL methods. Much recent work reduces the average-reward problem to discounted problems via the vanishing discount factor approach; the resulting discounted problems can then be addressed by model-free RL or model-based approaches that leverage the classical MDP toolkit (see, e.g., [17, 18, 40, 41] and references therein). While [16] studies non-communicating MDPs without this reduction, it focuses on a single communicating class and is model-based.

In contrast, rather than approximating the average-reward problem through a sequence of discounted problems, we tackle the multichain problem directly by exploiting the structure of the original problem itself. Our approach is based on Bather’s unique hierarchical decomposition (UHD) of the state space [3], which we view as fundamental to solving multichain MDPs, independent of the particular algorithm built on top of it: it identifies which regions of the state space are capable of sustaining long-run behavior and are safe to explore with existing model-free methods. This decomposition partitions the state space into communicating subsystems (also introduced by Bather [2]) and transient states, inducing a structured reformulation of the global decision problem into two types of subproblems (Schweitzer [24], Ohno [21]): optimal decision problems within each subsystem, and a global decision problem formulated as an aggregated optimal stopping problem (OSP).

These subproblems have features that are amenable to model-free RL. The aggregated OSP is a transient total-reward problem, while each subsystem is communicating, so its associated AOE reduces to a single equation (the second equation in (1.1) with $A _ { * } = A$ for the subsystem).

We exploit this structure algorithmically. Our base algorithm is an asynchronous, essentially model-free RL analogue of Ohno’s value-iteration algorithm for multichain MDPs. For each subsystem, we solve the average-reward problem using a stochastic relative value iteration method, RVI Q-learning, introduced in Abounadi, Bertsekas, and Borkar [1] for unichain MDPs and recently shown to converge in (weakly) communicating MDPs [33, 38, 39]. For the aggregated OSP, we employ classical $\mathrm { Q } -$ -learning [29, 34].

We show that the base algorithm converges almost surely (a.s.) to the optimal gain vector $g ^ { * }$ and produces gain-optimal policies after finite time (Theorems 3.1 and 3.2).

A shortcoming of the base algorithm, inherited from Ohno’s approach, is that it can exhibit poor transient performance. Moreover, although $g ^ { * }$ is obtained asymptotically, the AOE (1.1) is not fully solved: we only obtain partial solutions from the subsystem-level AOEs.

We propose two further algorithms that build on the base algorithm and address these issues. They are inspired by the seminal work of Federgruen and Schweitzer [13, 14] and Schweitzer [24] on value iteration methods for multichain MDPs. The first algorithm approximately solves the full AOE to obtain near gain-optimal policies by leveraging the sub-MDP AOE solutions computed by RVI Q-learning. The second algorithm, aimed at obtaining near bias-optimal policies, approximates the optimal bias function via discounted approximations and solves an induced average-reward multichain MDP using the base algorithm. We establish almost-sure convergence guarantees for both algorithms, together with eventual performance guarantees (Theorems 4.1–4.2, 5.1–5.2), so that all three algorithms come with such guarantees. To our knowledge, these are the first essentially model-free average-reward RL algorithms for general multichain MDPs that do not rely on reductions to discounted problems.

Beyond these results, an important implication of this work is removing a key barrier to averagereward options algorithms, a class of hierarchical RL algorithms with temporal abstractions: the restrictive unichain or communicating assumptions required in [31, 33, 38] are no longer needed. More broadly, the sub-policies learned within each class of Bather’s UHD are closely analogous to options [28], suggesting a principled basis for option discovery and reuse. Moreover, since the decomposition depends only on transition structure, not rewards or transition probability values, it may also suit continual learning settings where preferences evolve or transition probabilities vary over time but the environment’s transition structure remains stable.

Another important application of our algorithms is to infinite-space MDPs, which may themselves be obtained as discrete-time approximations of continuous-time stochastic control problems through the Markov chain approximation method [20]. Although our algorithms are developed for finitespace MDPs, they can be combined with quantization-based approximation methods, such as those studied in [19, 23], to compute approximate solutions for infinite-space MDPs. Since our algorithms handle general multichain MDPs, applying them does not require strong ergodicity conditions on the original infinite-space MDP model, in contrast to most existing learning-based algorithms for the quantized MDPs.

Finally, although our algorithms assume knowledge of the MDP’s transition graph, this assumption can in principle be removed by learning the transition structure through exploration. In Section 9, we discuss this limitation and related directions for future work, including online discovery and maintenance of Bather’s UHD.

The paper is organized as follows. Section 2 provides background on average-reward MDPs and a class of total-reward MDPs, and briefly describes the RL setting and assumptions considered in this paper. Related background material on RVI Q-learning for weakly communicating MDPs and total-reward Q-learning used by our algorithms is provided in Appendices A and B, respectively. Section 3 presents the base algorithm, Section 4 the algorithm for approximate AOE solutions, and Section 5 the algorithm aimed at obtaining near bias-optimal policies. These sections also state the main convergence results and establish the performance guarantees, with proof outlines for the convergence of the base algorithm included in Section 3. Section 6 presents preliminary experimental results, with additional experiments and details provided in Appendix C. Detailed convergence proofs are given in Sections 7 and 8. Section 9 concludes the paper with a discussion of limitations and future work.

## 2 Preliminaries

We first recall some basic concepts for average-reward MDPs and introduce notation used throughout. We then briefly review a class of total-reward MDPs that we use in developing the average-reward RL algorithms proposed in this paper.

## 2.1 Average-reward MDPs: basic definitions and notation

We follow standard MDP and RL notation (see, e.g., [22, 27]). For an MDP $\mathcal { M } = ( \boldsymbol { S } , \boldsymbol { A } , \boldsymbol { p } , \boldsymbol { r } )$ , we use π or $\mu$ to denote a policy. We focus on stationary policies and, in most cases, nonrandomized stationary policies, since the latter class contains optimal policies for the various expected reward criteria considered in this paper [22]. This class is denoted by $\Pi _ { \mathcal { A } } : = \{ \pi \mid \pi ( s ) \in \mathcal { A } ( s ) , \forall s \in \mathcal { S } \}$ More generally, for any $B \subset A$ such that $B ( s ) \neq \emptyset$ for all $s \in \mathcal { S }$ , Π<sub>B</sub> denotes the subset of nonrandomized stationary policies satisfying the action constraints specified by $B .$ (Here and in what follows, the set-inclusion notation $\subset$ is used to mean $\subseteq . { } )$

A stationary policy $\pi$ induces a Markov chain on $s$ with transition probability matrix $P _ { \pi }$ and expected one-stage reward vector $r _ { \pi }$ . The set of recurrent states (resp. transient states) of this Markov chain is denoted by $\mathcal { R } _ { \pi } \mathrm { ~ } ( \mathrm { r e s p . ~ } \mathcal { T } _ { \pi } )$ . A set $D \subset S$ is closed under $\pi \ \mathrm { i f } ,$ starting from any $s \in D$ , the chain remains in D with probability one.

The gain (or average reward) of a stationary policy π from state s is

$$
g _ { \pi } ( s ) : = \operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \mathbb { E } _ { s } ^ { \pi } \left[ \sum _ { t = 0 } ^ { N - 1 } r ( s _ { t } , a _ { t } ) \right] ,
$$

where $\mathbb { E } _ { s } ^ { \pi }$ denotes expectation under $\pi$ with $s _ { 0 } = s$ . The optimal gain is $g ^ { * } ( s ) = \operatorname* { s u p } _ { \pi } g _ { \pi } ( s )$ . We say that π is gain-optimal for state s if $g _ { \pi } ( s ) = g ^ { * } ( s )$ , and gain-optimal if this holds for all $s \in S$

The optimal gain function $g ^ { \ast } ( \cdot )$ is constant on each recurrent class induced by a stationary policy, but need not be constant over S in the multichain setting. As noted earlier, $g ^ { * }$ is the unique solution to the $g \cdot$ -component of the multichain AOE (1.1). Its two nested equations take the following componentwise form: for all $s \in S$

$$
\begin{array} { c l } { { \displaystyle g ^ { * } ( s ) = \operatorname* { m a x } _ { a \in A ( s ) } \Big \{ \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) \Big \} , } } \\ { { \displaystyle g ^ { * } ( s ) + h ( s ) = \operatorname* { m a x } _ { a \in A _ { * } ( s ) } \Big \{ r ( s , a ) + \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) h ( s ^ { \prime } ) \Big \} , } } \end{array}
$$

where $\begin{array} { r } { \mathcal { A } _ { * } ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \{ \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) \} } \end{array}$ . When $g ^ { * }$ is constant, independent of the initial state $s ,$ the first equation becomes vacuous, $A _ { * } = A$ , and the scalar $g ^ { * }$ is uniquely determined by the second equation.

A general structural condition ensuring a constant $g ^ { * }$ is that M is communicating: for every $s , s ^ { \prime } \in S$ , there exists a stationary policy under which s is reachable from $s ^ { \prime }$ with positive probability. More generally, a subset $C \subset S$ is called a closed communicating class if every pair of states in $C$ communicate in this sense and, starting from any state in $C ,$ the system remains within $C$ regardless of the policy; i $. . . , C$ is closed under every stationary policy. On each such $C , g ^ { * } ( \cdot )$ is constant. A communicating MDP has $s$ as its single closed communicating class. Slightly more generally, an MDP is weakly communicating if S consists of a single closed communicating class together with a possibly empty set of states that are transient under every stationary policy. In such an MDP, $g ^ { * }$ is also constant. This state communication structure forms the foundation for Bather’s UHD for general, multichain MDPs (see Section 3).

The AOE (1.1) admits infinitely many solutions of the form $( g ^ { * } , h )$ . The solution structure for h depends on the recurrence structure under stationary optimal policies [26]. In particular, h need not be unique up to an additive constant, even in the communicating setting. A particular solution related to the bias-optimality criterion is the optimal bias function $h ^ { * }$ . We introduce the bias of a policy and discuss bias optimality later (see Section 5), when these concepts are needed.

The RL setting and additional notation. In RL, we deal with random state transitions and rewards generated according to $\mathcal { M }$ at feasible state-action pairs $( s , a ) \in A .$ . Throughout, the reward distribution associated with each feasible pair $( s , a )$ is assumed to have mean $r ( s , a )$ and finite variance. The transition and reward data may be obtained from a simulator or through interactions with the environment, by following one or multiple trajectories, possibly generated by multiple agents. In a multichain MDP, a single trajectory need not provide data from all relevant parts of the state space, so multiple trajectories may in general be needed. Our requirements on how such data are generated and used are formulated based on general asynchronous stochastic approximation theory; specific conditions will be stated later for the respective algorithms.

We use $\mathbb { 1 } \{ \ldots \}$ for indicator functions, $\mathbf { 1 } _ { B }$ for set indicators, and 1 for all-ones vectors of various dimensions.

## 2.2 Total-reward MDPs of SSP type

In general, solving a (undiscounted) total-reward MDP is more demanding than solving an averagereward MDP, as it amounts to solving the problem with respect to (w.r.t.) the bias-optimality criterion, which is stronger than gain-optimality [22, Sec. 10.4]. However, the average-reward framework need not be invoked for transient total-reward problems or for certain non-transient problems with termination under every policy or under optimal policies. We discuss a family of such problems, which has a well-established theory of optimality and solution methods, including Q-learning in the RL setting. We make use of this class of problems in both the formulation and analysis of our average-reward RL algorithms for multichain MDPs. To provide the background, we review here its model assumptions and optimality properties; for references, see Bertsekas and Tsitsiklis [4] and [5, Sec. 2.2]. Prior Q-learning results for these problems are reviewed in Appendix B.

Consider an MDP $\hat { \mathcal { M } } : = \{ \hat { S } \cup \{ \Delta \} , \hat { A } , \hat { p } , \hat { r } \}$ where $\Delta$ is a special termination state: it is absorbing and has zero reward. We call $\hat { S }$ the efective state space of $\hat { \mathcal { M } } .$ and let $\hat { A }$ denote the set of feasible state-action pairs for states in $\hat { S } .$ . We call a policy proper if $\Delta$ is reached with probability one from every initial state $\boldsymbol { s } \in \hat { \mathcal { S } } ,$ , and improper otherwise. For any policy $\pi ,$ define its total-reward function by $\begin{array} { r } { V _ { \pi } ( s ) : = \operatorname* { l i m } \operatorname* { i n f } _ { N \to \infty } \mathbb { E } _ { s } ^ { \pi } \left\lceil \sum _ { t = 0 } ^ { N } \hat { r } ( s _ { t } , a _ { t } ) \right\rceil , s \in \hat { \mathcal { S } } } \end{array}$ . Define the optimal value function of $\hat { \mathcal { M } }$ by $V ^ { * } ( s ) : = \operatorname* { s u p } _ { \pi } V _ { \pi } ( s ) , s \in \hat { \mathcal { S } }$

Assumption 2.1. Consider $\hat { \mathcal { M } }$ and its nonrandomized stationary policies.

(i) There exists at least one proper policy.

(ii) Every improper policy π satisfies $V _ { \pi } ( s ) = - \infty$ for at least one state $s \in { \hat { S } }$

We call $\hat { \mathcal { M } }$ a total-reward MDP of SSP type if $\hat { \mathcal { M } }$ satisfies Assumption 2.1. SSP stands for stochastic shortest path. In [4, 5], which study total-cost MDPs, such MDPs are referred to as SSP problems.

Theorem 2.1 ([4], [5, Prop. 2.1]). Let $\hat { \mathcal { M } }$ be a total-reward MDP of SSP type. Then the optimal value function $V ^ { * }$ is real-valued and is the unique real-valued function satisfying the optimality equation

$$
\begin{array} { r } { V ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal A ( s ) } \big \{ \hat { r } ( s , a ) + \sum _ { s ^ { \prime } \in \hat { \mathcal S } } \hat { p } ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) \big \} , \quad s \in \hat { \mathcal S } . } \end{array}
$$

Furthermore, any stationary policy attaining the maximum in this equation for every state s is proper and optimal.

Algorithm 1 Method for obtaining optimal gain and gain-optimal policy (Conceptual Description)   
1: Phase 1: Apply Bather’s UHD to M   
2: Obtain communicating classes $\{ C _ { \ell , k } \}$ with restricted action sets $\boldsymbol { \mathcal { A } } _ { \ell , k }$ and transient sets {T<sub>ℓ</sub>}   
3: Define communicating sub-MDPs $\mathcal { M } _ { \ell , k } = ( C _ { \ell , k } , \mathcal { A } _ { \ell , k } , p , r )$   
4: end Phase   
5: for $t = 0 , 1 , 2 , \ldots$ do ▷ Phases 2–4 run concurrently   
6: Phase 2: In parallel, run RVI Q-learning on each $\mathcal { M } _ { \ell , k }$   
7: Obtain estimates $r _ { \ell , k } ^ { ( t ) }$ and $\pi _ { \ell , k } ^ { ( t ) }$ for aggregated OSP $\bar { \mathcal { M } } ^ { ( t ) } = ( \bar { \mathcal { S } } , \bar { \mathcal { A } } , \bar { p } , \bar { r } ^ { ( t ) } )$   
8: end Phase   
9: Phase 3: Run total-reward Q-learning on the OSP induced by $\bar { \mathcal { M } } ^ { ( t ) }$   
10: Obtain iterates $\bar { Q } ^ { ( t ) }$ and policy $\bar { \pi } ^ { ( t ) }$ specifying stop or exit through $( s ^ { * } , a ^ { * } )$ for each $\bar { s } _ { \ell , k }$   
11: end Phase   
12: Phase 4: Construct policy $\pi ^ { ( t ) }$ for original MDP M   
13: set $\pi ^ { ( t ) } ( s )  \bar { \pi } ^ { ( t ) } ( \bar { s } )$ for all $s \in \mathcal T$   
14: for all $( \ell , k )$ in parallel do   
15: if $\bar { \pi } ^ { ( \dot { t } ) } ( \bar { s } _ { \ell , k } ) = a _ { \ell , k } ^ { \mathrm { s t o p } }$ then   
16: set $\pi ^ { ( t ) } ( s )  \pi _ { \ell , k } ^ { ( t ) } ( s )$ for all $s \in C _ { \ell , k }$   
17: else if $\bar { \pi } ^ { ( t ) } ( \bar { s } _ { \ell , k } ) = ( s ^ { * } , a ^ { * } )$ then   
18: $\pi _ { \ell , k } ^ { \mathrm { r e a c h } } \gets \mathrm { R E A C H P O L I C Y } ( \mathcal { M } _ { \ell , k } , s ^ { * } )$   
19: Set $\pi ^ { ( t ) } ( s )  \pi _ { \ell , k } ^ { \mathrm { r e a c h } } ( s ) \mathrm { ~ f o r ~ } s \in C _ { \ell , k } \setminus \{ s ^ { * } \}$   
20: Set $\pi ^ { ( t ) } ( s ^ { * } )  a ^ { * }$   
21: end if end for end Phase   
22: end for

The optimality equation can equivalently be expressed in terms of the optimal state-action value function of M<sup>ˆ</sup> (or optimal action-value function for short), defined by

$$
\begin{array} { r } { Q ^ { * } ( s , a ) : = \hat { r } ( s , a ) + \sum _ { s ^ { \prime } \in \hat { S } } \hat { p } ( s ^ { \prime } \mid s , a ) V ^ { * } ( s ^ { \prime } ) , \quad ( s , a ) \in \hat { \mathcal { A } } . } \end{array}
$$

Throughout the paper, we call a nonrandonmized stationary policy π greedy w.r.t. a Q-function Q if $\pi ( s ) \in \arg \operatorname* { m a x } _ { a } Q ( s , a )$ over the feasible actions at every state s. Theorem 2.1 then implies that any stationary policy greedy w.r.t. $Q ^ { * }$ is proper and optimal for $\hat { \mathcal { M } }$ . Under the standard stepsize and asynchrony conditions of [29], classical Q-learning applied to an SSP-type total-reward MDP converges a.s. to $Q ^ { * }$ [29, 36] (cf. Appendix B).

For average-reward RL in multichain MDPs, our base algorithm, after decomposing the problem, makes use of Q-learning for a transient total-reward MDP, which is an SSP-type MDP with all policies proper. A subsequent RL algorithm we propose involves Q-learning in a more general SSP-type total-reward MDP, which may have improper policies.

## 3 Base algorithm

We first present a largely model-free algorithm for computing the optimal gain vector and gainoptimal policies in a multichain MDP. The method builds on Bather’s UHD [3], and subsequent value-iteration approaches of Schweitzer [24] and Ohno [21].

The algorithm proceeds in four phases. A summary is provided in Algorithm 1. Phase 1 decomposes the MDP into communicating subsystems. Phase 2 estimates the optimal gain within each subsystem via RVI Q-learning. Phase 3 constructs an aggregated OSP to determine whether each class should be retained or made transient. Phase 4 lifts the aggregated decisions to a policy on the original MDP. Phases 2–4 operate concurrently.

The algorithm assumes knowledge of the transition graph: for each state–action pair, the set of possible successor states, but not their transition probabilities. This structural information is used to construct the decomposition in Phase 1; the learning procedures in Phases $2 { - } 4$ do not require any model knowledge. We discuss the practical relevance of this assumption and possible approaches to relaxing it in Section 9.

## 3.1 The four phases of the algorithm

## 3.1.1 Phase 1: Bather’s UHD

Bather’s UHD [3, 24] is a purely structural decomposition, depending only on which transitions are possible, not on their underlying probabilities. As such, it can be computed from the transition graph introduced above. We apply this decomposition to partition $s$ into communicating subsystems and transient states.

The decomposition proceeds hierarchically, level by level. At level 0, we identify all closed communicating classes of S under the full action constraint set A, forming $\mathcal { C } _ { 0 }$ , which is always non-empty [3]. These classes can be identified, for example, using the algorithm of Fox and Landi [15]. The transient set $\mathcal { T } _ { 0 }$ consists of the states outside $\mathcal { C } _ { 0 }$ from which $\mathcal { C } _ { 0 }$ is reached almost surely under every policy. The remaining states form $S _ { 1 } = S \setminus ( \mathcal { C } _ { 0 } \cup \mathcal { T } _ { 0 } )$ , on which the procedure repeats with a restricted action set that excludes transitions out of $S _ { 1 }$ , yielding $\mathcal { C } _ { 1 } , \mathcal { T } _ { 1 }$ , and so on. The process terminates in finitely many levels $\ell = 0 , 1 , \ldots , L$ , producing a partition

$$
\mathcal { S } = \mathcal { C } _ { 0 } \cup \mathcal { T } _ { 0 } \cup \mathcal { C } _ { 1 } \cup \mathcal { T } _ { 1 } \cup \dots \cup \mathcal { C } _ { L } \cup \mathcal { T } _ { L } ,
$$

where $\mathcal { C } _ { \ell } = \{ C _ { \ell , k } : k = 1 , \ldots , n ( \ell ) \}$ is a collection of disjoint communicating classes and $\tau _ { \ell }$ is a set of transient states at level ℓ. Let $\textstyle T = \bigcup _ { \ell } { \mathcal { T } } _ { \ell }$ denote the set of all transient states.

For each class $C _ { \ell , k }$ , define the class-preserving action set $\mathcal { A } _ { \ell , k } ( s ) \subseteq \mathcal { A } ( s ) , s \in C _ { \ell , k }$ , consisting of actions whose transitions remain within $C _ { \ell , k }$ . The corresponding exit actions $\mathcal { E } _ { \ell , k } ( s ) : = \mathcal { A } ( s ) \backslash \mathcal { A } _ { \ell , k } ( s )$ are those whose transitions can reach at least one state at a strictly lower level of the decomposition. We define the exit state–action pairs by $\mathcal { E } _ { \ell , k } : = \{ ( s , a ) : s \in C _ { \ell , k } , \ a \in \mathcal { E } _ { \ell , k } ( s ) \}$

Under $\textstyle A _ { \ell , k } ,$ , each $C _ { \ell , k }$ forms a communicating MDP, so the optimal gain within the class is constant across its states. Each class induces a sub-MDP

$$
\mathcal { M } _ { \ell , k } = ( C _ { \ell , k } , \mathcal { A } _ { \ell , k } , p , r ) .
$$

The decomposition satisfies two key properties, which ensure the UHD uniquely identifies all regions capable of sustaining long-run behavior.

Lemma 3.1 (Properties of Bather’s UHD [3, 24]).

(i) Every recurrent class induced by any stationary policy is contained in a unique $C _ { \ell , k }$

(ii) For each $C _ { \ell , k _ { \ell } }$ , there exists a stationary policy under which $C _ { \ell , k }$ is recurrent.

This decomposition reduces the global problem to choosing, for each $C _ { \ell , k }$ , whether to remain within it or exit toward other regions, forming the basis of the aggregated decision problem in Phases 2–4.

## 3.1.2 Phase 2: RVI Q-learning

Phase 2 runs RVI Q-learning [1, 38] on each communicating sub-MDP $\mathcal { M } _ { \ell , k }$ in parallel, using the asynchronous scheme described in Appendix A. For each learner, choose a stepsize sequence $\{ \alpha _ { \ell , k } ^ { ( n ) } \} _ { n \ge 0 }$ satisfying Assumption A.2 and a reference function $f _ { \ell , k }$ satisfying Assumption A.1. Let $Y _ { \ell , k } ^ { ( t ) }$ denote the set of class-preserving state–action pairs selected for update at iteration t, with each learner’s update schedule satisfying Assumption A.3.

For each $( s , a ) \in Y _ { \ell , k } ^ { ( t ) }$ , observe a successor and reward $( S _ { t + 1 } ^ { s a } , R _ { t + 1 } ^ { s a } )$ generated by taking action a in state s, and set

$$
Q _ { \ell , k } ^ { ( t + 1 ) } ( s , a ) = Q _ { \ell , k } ^ { ( t ) } ( s , a ) + \alpha _ { \ell , k } ^ { ( \nu ( t , ( s , a ) ) ) } \Big [ R _ { t + 1 } ^ { s a } - f _ { \ell , k } ( Q _ { \ell , k } ^ { ( t ) } ) + \operatorname* { m a x } _ { a ^ { \prime } \in A _ { \ell , k } ( S _ { t + 1 } ^ { s a } ) } Q _ { \ell , k } ^ { ( t ) } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } ) - Q _ { \ell , k } ^ { ( t ) } ( s , a ) \Big ] .
$$

Here, $\nu ( t , ( s , a ) )$ counts the updates of $( s , a )$ prior to iteration t in this learner. Components not selected for update retain their values. This is the classwise version of (A.1).

The reference function $f _ { \ell , k }$ supplies the gain estimates, which satisfy

$$
r _ { \ell , k } ^ { ( t ) } : = f _ { \ell , k } ( Q _ { \ell , k } ^ { ( t ) } ) \to r _ { \ell , k } ^ { * } \qquad \mathrm { a . s . } ,
$$

where $r _ { \ell , k } ^ { * }$ is the optimal gain of $\mathcal { M } _ { \ell , k }$ . Moreover, almost surely, for all suficiently large t, every policy $\pi _ { \ell , k } ^ { ( t ) }$ greedy w.r.t. $Q _ { \ell , k } ^ { ( t ) }$ is gain-optimal for $\mathcal { M } _ { \ell , k }$ . These guarantees follow from [38, Thm. 3.1] and are formalized for Phase 2 in Lemma $7 . 1$ of Section 7.

The associated relative-value iterates $v _ { \ell , k } ^ { ( t ) } ( s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } _ { \ell , k } ( s ) } Q _ { \ell , k } ^ { ( t ) } ( s , a )$ also converge a.s. to a compact subset of relative-value vectors that, together with $r _ { \ell , k } ^ { * }$ , solve the AOE for $\mathcal { M } _ { \ell , k } ;$ see Appendix A. Convergence to a single relative-value vector is not required for the base algorithm.

Classes at level zero cannot be exited, so their restricted optimal gains coincide with their optimal gains in the original MDP. For higher-level classes, the restricted gains describe what can be achieved by remaining within the class. Phase 3 incorporates the possibility of leaving these classes to determine the optimal gains throughout the entire MDP.

## 3.1.3 Phase 3: aggregated OSP

Phase 3 uses the sub-MDP gain estimates from Phase 2 to determine whether to remain within each class or exit it, and which actions to take at transient states. These decisions are formulated as an OSP over an aggregated MDP, where stopping represents remaining within a class and following an optimal policy for the restricted sub-MDP. We use the aggregated OSP formulation implicit in Schweitzer [24] and made explicit by Ohno [21], solving it here through model-free learning.

We first specify the limiting aggregated MDP $\bar { \mathcal { M } } = ( \bar { \mathcal { S } } , \bar { \mathcal { A } } , \bar { p } , \bar { r } )$ using the exact sub-MDP gains $r _ { \ell , k } ^ { * }$ . During learning, the gains are replaced by the current estimates $r _ { \ell , k } ^ { ( t ) }$ from Phase 2, allowing the two phases to operate concurrently. The transition probabilities $\bar { p }$ need not be computed or stored, since aggregated successors can be obtained from transition observations in the original $\mathrm { M D P } ,$ as explained in the learning scheme below.

States and actions. Let $\mathcal { C } = \{ C _ { \ell , k } : \ell = 0 , \ldots , L , \ k = 1 , \ldots , n ( \ell ) \}$ . Represent each class $C _ { \ell , k }$ by a single state $\bar { s } _ { \ell , k }$ , retain all transient states, and introduce an absorbing terminal state $\Delta$ . The aggregated state space is

$$
\begin{array} { r } { \bar { \mathcal { S } } = \{ \bar { s } _ { \ell , k } : C _ { \ell , k } \in \mathcal { C } \} \cup \mathcal { T } \cup \{ \Delta \} . } \end{array}
$$

Define the aggregation map $\phi : { \cal S }  \bar { \cal S } \setminus \{ \Delta \}$ by

$$
\phi ( s ) = \left\{ \begin{array} { l l } { \bar { s } _ { \ell , k } , } & { s \in C _ { \ell , k } ; } \\ { s , } & { s \in \mathcal { T } . } \end{array} \right.
$$

At each class state $\bar { s } _ { \ell , k } .$ , the available actions consist of a stopping action $a _ { \ell , k } ^ { \mathrm { s t o p } }$ and the exit state–action pairs in $\mathcal { E } _ { \ell , k }$ . Thus,

$$
\begin{array} { r } { \bar { \mathcal { A } } ( \bar { s } ) = \left\{ \begin{array} { l l } { \{ a _ { \ell , k } ^ { \mathrm { s t o p } } \} \cup \mathcal { E } _ { \ell , k } , } & { \bar { s } = \bar { s } _ { \ell , k } ; } \\ { \mathcal { A } ( s ) , } & { \bar { s } = s \in \mathcal { T } ; } \\ { \{ a _ { \Delta } \} , } & { \bar { s } = \Delta , } \end{array} \right. } \end{array}
$$

where $a _ { \Delta }$ is a dummy terminal action. An exit pair $( s , a ) \in \mathcal { E } _ { \ell , k }$ specifies both the underlying state at which to act and the original action to execute. At level zero, $\mathcal { E } _ { 0 , k } = \emptyset$ , so stopping is the only available action.

Transitions and rewards. For an exit action $\bar { a } = ( s , a ) \in \mathcal { E } _ { \ell , k }$ , define

$$
\bar { p } ( \bar { y } \mid \bar { s } _ { \ell , k } , \bar { a } ) = \sum _ { y \in S : \phi ( y ) = \bar { y } } p ( y \mid s , a ) , \qquad \bar { y } \in \bar { S } \backslash \{ \Delta \} .\tag{3.1}
$$

For $s \in \mathcal T$ and $a \in \mathcal { A } ( s )$ , define similarly

$$
\bar { p } ( \bar { y } \mid s , a ) = \sum _ { y \in S : \phi ( y ) = \bar { y } } p ( y \mid s , a ) , \qquad \bar { y } \in \bar { S } \backslash \{ \Delta \} .\tag{3.2}
$$

For stopping and terminal actions, set

$$
\bar { p } ( \Delta \mid \bar { s } _ { \ell , k } , a _ { \ell , k } ^ { \mathrm { s t o p } } ) = 1 , \qquad \bar { p } ( \Delta \mid \Delta , a _ { \Delta } ) = 1 .
$$

The rewards are

$$
\bar { r } ( \bar { s } , \bar { a } ) = \left. { r _ { \ell , k } ^ { \ast } , \bar { s } = \bar { s } _ { \ell , k } , \bar { a } = a _ { \ell , k } ^ { \mathrm { s t o p } } } ; \right.\tag{3.3}
$$

The stopping reward represents the optimal gain attainable by remaining in the corresponding class. Since all other rewards are zero, the total reward of the aggregated process is precisely the reward received upon stopping. Maximizing expected total reward therefore amounts to maximizing the expected gain of the selected class. Under this formulation [21, 24], the optimal OSP values coincide with the original optimal gains through the aggregation map $\phi$ (Lemma 7.5).

The aggregated $\mathrm { O S P }$ is a transient total-reward problem of the type described in Section 2.2: every stationary policy is proper (Lemma 7.2). We include proofs of these properties for completeness in Section 7.

Learning and gain estimates. During learning, Phase 3 uses the aggregated OSP $\bar { \mathcal { M } } ^ { ( t ) } =$ $( \bar { S } , \bar { A } , \bar { p } , \bar { r } ^ { ( t ) } )$ , where $\bar { r } ^ { ( t ) }$ is obtained from (3.3) by replacing $r _ { \ell , k } ^ { * }$ with $r _ { \ell , k } ^ { ( t ) }$ . We apply the asynchronous total-reward Q-learning scheme of Appendix B, using transition observations from the original MDP and the current sub-MDP gain estimates. The aggregated transition probabilities need not be computed or stored; instead, we can use transition data generated from the original MDP.

Recall that a retained transient state $\bar { s } = s$ has access to its original actions $a \in \mathcal { A } ( s )$ , whereas a class state $\bar { s } _ { \ell , k }$ has access to exit actions $\bar { a } = ( s , a ) \in \mathcal { E } _ { \ell , k }$ and a stopping action $a _ { \ell , k } ^ { \mathrm { s t o p } }$ . Thus, each non-stopping aggregated state–action pair specifies an original pair $( s , a )$ from which a transition can be observed.

At iteration $t ,$ select a nonempty set $\bar { Y } _ { t }$ of nonterminal aggregated state–action pairs. For each $( \bar { s } , \bar { a } ) \in \bar { Y } _ { t }$ , construct a successor and reward $( \bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } , \bar { R } _ { t + 1 } ^ { \bar { s } \bar { a } } )$ as follows:

• Non-stopping actions. For the corresponding original pair $( s , a )$ , observe a successor $s ^ { \prime }$ generated by taking action a in state s of the original MDP, and set

$$
\bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } = \phi ( s ^ { \prime } ) , \qquad \bar { R } _ { t + 1 } ^ { \bar { s } \bar { a } } = 0 .
$$

• Stopping actions. For a stopping action $\bar { a } = a _ { \ell , k } ^ { \mathrm { s t o p } }$ , no transition observation is needed. Set

$$
\bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } = \Delta , \qquad \bar { R } _ { t + 1 } ^ { \bar { s } \bar { a } } = r _ { \ell , k } ^ { ( t ) } .
$$

Using these successors and rewards, update

$$
\bar { Q } ^ { ( t + 1 ) } ( \bar { s } , \bar { a } ) = \bar { Q } ^ { ( t ) } ( \bar { s } , \bar { a } ) + \bar { \alpha } _ { t } ( \bar { s } , \bar { a } ) \left[ \bar { R } _ { t + 1 } ^ { \bar { s } \bar { a } } + 1 \{ \bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } \neq \Delta \} \cdot \operatorname* { m a x } _ { \bar { a } ^ { \prime } \in \mathcal { A } ( \bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } ) } \bar { Q } ^ { ( t ) } ( \bar { S } _ { t + 1 } ^ { \bar { s } \bar { a } } , \bar { a } ^ { \prime } ) - \bar { Q } ^ { ( t ) } ( \bar { s } , \bar { a } ) \right] .
$$

Here $\bar { \alpha } _ { t } ( \bar { s } , \bar { a } )$ is the stepsize for the selected component. Components not selected for update retain their values, and $\bar { Q } ^ { ( t ) } ( \Delta , a _ { \Delta } ) = 0$ for every t.

Under the sampling scheme above and the asynchrony and stepsize conditions in Assumption B.1, Lemma 7.3 establishes

$$
{ \bar { Q } } ^ { ( t ) } \to { \bar { Q } } ^ { * } \qquad \mathrm { a . s . , }
$$

where $\bar { Q } ^ { * }$ is the optimal action-value function of the limiting aggregated OSP. The lemma combines properness of all stationary policies with the Phase 2 convergence guarantees, which ensure that the errors in the stopping rewards vanish almost surely.

Define the gain estimates on the original state space by

$$
g ^ { ( t ) } ( s ) : = \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \phi ( s ) ) } \bar { Q } ^ { ( t ) } ( \phi ( s ) , \bar { a } ) , \qquad s \in \mathcal { S } .\tag{3.4}
$$

$\mathrm { B y }$ Lemmas 7.4 and 7.5, the corresponding maxima of $\bar { Q } ^ { * }$ equal $g ^ { * } ( s )$ . Consequently, $g ^ { ( t ) }  g ^ { * }$ a.s. Choose an aggregated policy greedily:

$$
\bar { \pi } ^ { ( t ) } ( \bar { s } ) \in \arg \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \bar { s } ) } \bar { Q } ^ { ( t ) } ( \bar { s } , \bar { a } ) , \qquad \bar { s } \neq \Delta .
$$

The aggregated Q-values incorporate the Phase 2 sub-MDP gain estimates $r _ { \ell , k } ^ { ( t ) }$ through the stopping rewards. The resulting greedy policy specifies whether to remain in each class $C _ { \ell , k }$ (stop) or exit through some $( s ^ { \ast } , a ^ { \ast } ) \in \mathcal { E } _ { \ell , k }$ , as well as which actions to take at retained transient states. Phase 4 lifts these aggregated decisions to a policy on the original state space.

## 3.1.4 Phase 4: deriving a policy for the original MDP

Phase 4 lifts the aggregated policy $\bar { \pi } ^ { ( t ) }$ to a stationary policy $\pi ^ { ( t ) }$ on $\mathcal { M }$ . Since transient states and their action sets are retained in the aggregated MDP, their decisions can be used directly:

$$
\pi ^ { ( t ) } ( s ) = \bar { \pi } ^ { ( t ) } ( s ) , \qquad s \in \mathcal { T } .
$$

For a class $C _ { \ell , k } , \mathrm { i f } \ \bar { \pi } ^ { ( t ) } ( \bar { s } _ { \ell , k } ) = a _ { \ell , k } ^ { \mathrm { s t o p } }$ , the aggregated policy indicates it is currently best to remain within the class, receiving reward $r _ { \ell , k } ^ { ( t ) }$ . We implement this decision using the current sub-MDP policy $\pi _ { \ell , k } ^ { ( t ) }$ from Phase 2, setting $\pi ^ { ( t ) } ( s ) = \pi _ { \ell , k } ^ { ( t ) } ( s )$ for all $s \in C _ { \ell , k }$

If instead $\bar { \pi } ^ { ( t ) } ( \bar { s } _ { \ell , k } ) = ( s ^ { * } , a ^ { * } )$ , the aggregated policy indicates it is currently best to exit the class through $( s ^ { * } , a ^ { * } )$ . Implementing this decision requires first reaching $s ^ { * }$ within the class and then executing $a ^ { * }$ . Let ReachPolicy $\cdot ( \mathcal { M } _ { \ell , k } , s ^ { * } )$ return a stationary nonrandomized policy $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ that uses only class-preserving actions and reaches $s ^ { * }$ a.s. from every state in $C _ { \ell , k }$ . We refer to this requirement as the reaching property. Such a policy exists because $\mathcal { M } _ { \ell , k }$ is communicating. Set

$$
\pi ^ { ( t ) } ( s ) = { \left\{ \begin{array} { l l } { \pi _ { \ell , k } ^ { \mathrm { r e a c h } } ( s ) , } & { s \in C _ { \ell , k } \setminus \{ s ^ { * } \} ; } \\ { a ^ { * } , } & { s = s ^ { * } . } \end{array} \right. }
$$

Thus, the process follows the reaching policy until it reaches $s ^ { * }$ and then executes $a ^ { * }$

The base algorithm requires only the reaching property; the following example describes two possible implementations.

Example 3.1 (Implementing ReachPolicy). We construct the reaching policy by solving an SSP problem on the restricted subsystem $\mathcal { M } _ { \ell , k }$ . Make $s ^ { * }$ absorbing and terminal, and assign reward −1 to every transition except those entering $s ^ { * }$ , which receive reward 0. Since $\mathcal { M } _ { \ell , k }$ is communicating, a proper policy exists. Moreover, every improper stationary policy incurs a total reward of −∞. The SSP convergence results in Appendix B therefore apply under the stepsize and sampling conditions stated there: total-reward Q-learning converges a.s., and its greedy policies eventually satisfy the reaching property while minimizing expected hitting time.

If stationary randomized policies are allowed in the lifting construction, another option is to select uniformly among the class-preserving actions at each state. Communication ensures that this policy also satisfies the reaching property, although its expected hitting time may be substantially larger. We use the SSP construction in our implementation.

The transition graph available in Phase 1 can be used to verify whether a computed policy satisfies the reaching property. Thus, for a selected target, total-reward Q-learning can be continued until its greedy policy is verified to satisfy this property.

Alternatively, eventual validity can be ensured without using the transition graph for verification by maintaining a separate SSP learner for each class–target pair. Each learner retains its iterates and stepsize history, allowing learning to resume whenever its target is selected again. If every learner used infinitely often receives updates satisfying the SSP convergence conditions, the resulting policies eventually satisfy the reaching property a.s., even if the targets continue to change. ⋄

Fix an iteration t at which all reaching policies used in the construction of $\pi ^ { ( t ) }$ satisfy the reaching property. For every initial state s and class $C _ { \ell , k }$ , Lemma 7.6 gives

$$
\mathbb { P } _ { \pi ^ { ( t ) } } ( \mathrm { a b s o r b e d ~ i n ~ } C _ { \ell , k } \mid s _ { 0 } = s ) = \mathbb { P } _ { \bar { \pi } ^ { ( t ) } } ( \mathrm { s t o p s ~ a t ~ } \bar { s } _ { \ell , k } \mid \bar { s } _ { 0 } = \phi ( s ) ) .
$$

Thus, the process under $\pi ^ { ( t ) }$ eventually remains in each $C _ { \ell , k }$ with the same probability that $\bar { \pi } ^ { ( t ) }$ stops at $\bar { s } _ { \ell , k }$ . Once the Phase 2 sub-MDP policies used in stopping classes are gain-optimal, the gain of $\pi ^ { ( t ) }$ equals the expected stopping reward of $\bar { \pi } ^ { ( t ) }$ in the limiting OSP. An optimal aggregated policy then yields a gain-optimal policy on the original MDP.

## 3.2 Convergence results of the base algorithm

We now present the main theoretical guarantees of the base algorithm. The first result establishes convergence of the gain estimates, while the second establishes eventual gain optimality of the constructed policy. Their full proofs and supporting lemmas are given in Section 7.

Theorem 3.1 (Convergence of $g ^ { ( t ) } \to g ^ { * } )$ . Let M be a finite MDP and run Algorithm 1. Then the iterates $\bar { Q } ^ { ( t ) }$ produced in Phase 3 converge almost surely to $\bar { Q } ^ { * }$ , the optimal action-value function of the limiting aggregated OSP M<sup>¯</sup> . Moreover, the limit $\bar { Q } ^ { * }$ is related to the optimal gain $g ^ { * }$ of M by

$$
g ^ { * } ( s ) = \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \phi ( s ) ) } \bar { Q } ^ { * } ( \phi ( s ) , \bar { a } ) , \qquad s \in \mathcal { S } .
$$

Thus, $g ^ { ( t ) }  g ^ { * }$ almost surely.

Proof sketch. Phase 2 RVI Q-learning yields $r _ { \ell , k } ^ { ( t ) } \to r _ { \ell , k } ^ { \ast }$ a.s. for every class (Lemma 7.1). Moreover, every stationary policy of the aggregated OSP is proper (Lemma 7.2), providing the SSP structure needed for total-reward Q-learning convergence. The stopping rewards used in Phase 3 difer from their limiting values by vanishing errors. Combining this properness property with the vanishing reward errors, Lemma 7.3 establishes $\bar { Q } ^ { ( t ) } \to \bar { Q } ^ { * } \mathrm { ~ a . s . }$ , where $\bar { Q } ^ { * }$ is the optimal action-value function of the limiting aggregated OSP.

The actionwise maxima of $\bar { Q } ^ { * }$ give the optimal expected stopping rewards of this OSP (Lemma 7.4), which coincide with the optimal gains of the corresponding states in the original MDP (Lemma 7.5). Since each action set is finite, taking actionwise maxima in the converging Q-value iterates therefore gives $g ^ { ( t ) }  g ^ { * }$ a.s. □

Theorem 3.2 (Finite-time attainment of gain-optimality). In the construction of the policy $\pi ^ { ( t ) }$ in Algorithm 1 (Phase 4), let Reac $\cdot I P O L I C Y ( \mathcal { M } _ { \ell , k } , s ^ { * } )$ be any method that produces a stationary nonrandomized policy $\pi _ { \ell , k } ^ { r e a c h }$ on $\mathcal { M } _ { \ell , k }$ such that $s ^ { * }$ is reached with probability 1 from every initial state in $C _ { \ell , k }$ . Then, almost surely for all suficiently large t, the constructed policy $\pi ^ { ( t ) }$ is gainoptimal, $i . e . , g _ { \pi ^ { ( t ) } } ( s ) = g ^ { * } ( s )$ for all $s \in S$

Proof sketch. Phase 2 eventually produces gain-optimal policies within every class, almost surely (Lemma 7.1). Meanwhile, the aggregated Q-values converge to $\bar { Q } ^ { * }$ (Lemma 7.3). Because there are finitely many state–action pairs, the optimality gaps of strictly suboptimal aggregated actions, if any exist, have a positive minimum. Once the Q-value errors are suficiently small, these actions cannot be selected greedily. Hence, $\bar { \pi } ^ { ( t ) }$ is eventually optimal for the limiting aggregated OSP.

The lifting construction preserves the distribution over eventual stopping classes (Lemma 7.6). For all suficiently large $t ,$ the policy used within each stopping class attains its optimal class gain $r _ { \ell , k } ^ { * }$ , while the rewards accumulated before settling in that class do not afect the long-run average reward. Thus, the gain of the lifted policy equals the optimal expected stopping reward of the aggregated OSP. Since this value coincides with $g ^ { * }$ on the original state space (Lemma 7.5), $\pi ^ { ( t ) }$ is gain-optimal almost surely for all suficiently large t. □

## 3.3 Limitations and motivation for the subsequent algorithms

The base algorithm identifies the optimal gain vector and eventually produces gain-optimal policies almost surely. However, it does not construct a full relative-value vector solving the AOE, and its policies may exhibit poor transient performance. We next discuss these limitations to motivate the two subsequent algorithms.

Partial relative-value solutions. Phase 2 produces relative-value solutions for the restricted sub-MDPs $\mathcal { M } _ { \ell , k } ,$ where only class-preserving actions are available. These sub-MDP solutions do not generally combine directly into a relative-value vector solving the full AOE with $g ^ { * } \colon$ the global equations must also account for gain-preserving exit actions and retained transient states.

Thus, although the base algorithm identifies the optimal gain vector, constructing an associated full relative-value solution requires additional work. Algorithm 2 addresses this by identifying sub-MDP solutions that can be extended and learning an approximate solution of the full AOE.

Transient performance. Gain optimality alone does not distinguish policies by the rewards accumulated before their long-run behavior is established. The base algorithm can therefore produce gain-optimal policies with poor transient performance. Figure 1 illustrates two distinct sources of this behavior.

Example 3.2 (Routing to a selected exit state). Consider the MDP in Figure $\mathrm { 1 ( a ) }$ , with $K > 0$ The optimal gain achievable while remaining in $C = \{ s _ { 1 } , s _ { 2 } , s _ { 3 } \}$ is $2 / 3$ , attained by the cycle $s _ { 1 } \to s _ { 2 } \to s _ { 3 } \to s _ { 1 }$ . Exiting to the absorbing state z yields gain 1, so Phase 3 selects the exit action at s<sub>1</sub>.

Suppose ReachPolicy returns a policy minimizing expected hitting time, as sought by our auxiliary SSP construction. Starting from $s _ { 2 }$ , this policy takes the direct transition to $s _ { 1 } .$ , incurring reward −K, before exiting to $z .$ The alternative route $s _ { 2 }  s _ { 3 }  s _ { 1 }$ collects reward 2 before taking the same exit. Both lifted policies achieve the optimal gain of 1, however, the longer route improves the cumulative reward obtained before reaching z by $K + 2$ . This transient advantage can be made arbitrarily large by increasing K. ⋄

The underlying limitation is that the reaching property constrains eventual arrival at the selected exit state, but not the rewards accumulated along the way. A reaching method may account for these rewards, but the base algorithm’s gain-optimality guarantee does not require it to do so.

![](images/dfc484a2f3c2f34eeed36149b3f381694f0a31afd068830a6ed2f4258894dd55.jpg)  
Figure 1: Two sources of poor transient performance in the base algorithm. Each arrow represents a distinct action with transition probability one, and its label gives the original one-stage reward. Dashed boundaries indicate communicating subsystems.

Consequently, valid reaching policies can incur large transient losses, which may accumulate when the process passes through several exiting classes.

A separate source of poor transient performance comes from the selection of the aggregated policy. The OSP is designed to optimize gain, with zero reward assigned to all non-stopping actions. Consequently, selecting actions greedily from its Q-values does not account for diferences in transient reward among tied gain-optimal actions. This afects stopping and exit decisions at class states, as well as actions at retained transient states.

Example 3.3 (Gain-equivalent stopping and exit decisions). Consider Figure 1(b). Both singleton classes have gain zero. At x, the process can remain indefinitely with reward zero or exit to z with reward $\rho \in \{ - 1 , + 1 \}$ , receiving reward zero thereafter. Both decisions achieve gain zero. For transient performance, however, exiting is preferable when $\rho = 1$ , whereas remaining is preferable when $\rho = - 1$

These two reward settings produce the same aggregated OSP: both stopping rewards and the exit reward are zero. Its action values therefore provide no basis for choosing exit in the first setting and stop in the second. A greedy selection can incur an avoidable penalty or forgo a positive reward while remaining gain-optimal. ⋄

In the second example, no internal reaching steps are needed, so improving ReachPolicy alone cannot resolve the issue. These simple deterministic examples show that transient losses can arise both from the routes used to implement exit decisions and from the aggregated decisions themselves. In larger MDPs, the same efects can interact along stochastic trajectories spanning many states and classes. This motivates Algorithm 3, which seeks (near) bias-optimal policies to improve transient performance.

## 4 Extending sub-MDP solutions to approximate AOE solutions

In this section and the next, when Algorithm 1 is used as a subroutine on an auxiliary MDP, we refer to it as the base algorithm; references to Algorithm 1 itself concern its application to the original MDP.

Given $g ^ { * }$ , the AOE (1.1) reduces to

$$
h = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } } } \left\{ r _ { \mu } - g ^ { * } + P _ { \mu } h \right\} .\tag{4.1}
$$

Solving the AOE can provide additional information about the structural optimality properties of the MDP [26], as well as a natural way to select policies with potentially improved transient behavior. We are therefore interested in finding an approximate solution to this equation using the outputs of Algorithm 1. Several high-level considerations guide our approach. We explain them first, before presenting the algorithm.

Observe first that the exact value of the action constraint set $\mathcal { A } _ { * }$ is unavailable, although the a.s. convergence of Algorithm 1 (Theorem 3.1) enables approximations of $\mathcal { A } _ { * }$ to arbitrary accuracy as time goes to infinity. Thus, rather than aiming to solve (4.1) exactly, we take a perturbation approach inspired by Federgruen and Schweitzer’s work [14] on solving nested AOE-like equations via value iteration. Specifically, we relax both the equality and the action constraints in (4.1). For some small constants $\epsilon > 0$ and $\delta > 0$ , we aim to find $x \in \mathbb { R } ^ { | S | }$ satisfying

$$
\left. x - \operatorname* { m a x } _ { \mu \in \Pi _ { \tilde { A } } } \left\{ r _ { \mu } - g ^ { * } + P _ { \mu } x \right\} \right. _ { \infty } \leq \epsilon ,
$$

where the action constraint set $\tilde { \cal A }$ satisfies

$$
\mathcal { A } _ { \ast } ( s ) \subset \tilde { \mathcal { A } } ( s ) \subset \mathcal { A } _ { \delta } ( s ) , \qquad \forall s \in \mathcal { S } ,
$$

with $\mathcal { A } _ { \delta } ( s )$ given by

$$
\mathcal { A } _ { \delta } ( s ) : = \Big \{ a \in \mathcal { A } ( s ) \ | \ \sum _ { s ^ { \prime } \in \mathcal { S } } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) \geq g ^ { * } ( s ) - \delta \Big \} .
$$

Recall that $\begin{array} { r } { \mathcal { A } _ { * } ( s ) = \left\{ a \in \mathcal { A } ( s ) \vert \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \vert s , a ) g ^ { * } ( s ^ { \prime } ) = g ^ { * } ( s ) \right\} } \end{array}$ . Thus, $\tilde { \mathcal { A } }$ is an outer-approximation of $\mathcal { A } _ { * }$ , with $\tilde { \mathcal { A } } ( s )$ consisting of a subset of actions that are δ-optimal for state s in the first equation of the multichain AOE (1.1). The outer-approximation property is essential: excluding even one action in $A _ { * } ( s )$ could eliminate all optimal policies (cf. [26, Thm. $3 . 1 ( \mathrm { e 2 } ) ] ,$ ).

Second, to compute an approximate AOE solution, we make use of the RVI Q-learning iterates on each sub-MDP $\mathcal { M } _ { \ell , k }$ in Phase 2 of Algorithm 1. Since each sub-MDP is communicating, the RVI Q-learning algorithm can be configured with appropriate stepsizes and asynchronous update schedules ([38, Thm. 3.2]; cf. [39, Thm. 2.3] and Appendix A) so that it converges a.s. to an AOE solution for each sub-MDP, rather than merely to a compact subset of such solutions. Convergence to such a compact subset suficed earlier for Algorithm 1 to produce optimal policies after finite time for each sub-MDP. However, it is inadequate here for computing a global approximate AOE solution with convergence guarantees. Thus, we exploit the stronger convergence guarantee of RVI Q-learning.

Third, however, not every sub-MDP AOE solution is compatible with a global AOE solution [26]. Two questions then arise:

(i) Which sub-MDP solutions are usable?

(ii) How can the corresponding sub-MDPs be determined from Algorithm 1’s stochastic outputs? Based on Section 4.1 below, the sub-MDPs fall naturally into three categories depending on whether remaining, exiting, or both are gain-optimal. Solutions in the first category can be incorporated into a global AOE solution.

The actual RL algorithm must handle stochastic data while accomplishing the various tasks outlined above simultaneously. We therefore first present an idealized algorithm, assuming that the limiting sub-MDP solutions and other limiting values of Algorithm 1 are available. It serves as a blueprint for Algorithm 2 in Section 4.2.

## 4.1 An idealized algorithm via limiting sub-MDP solutions

Consider the idealized case in which the limits of the iterates from Algorithm 1 are known. By Theorem 3.1 and the stronger convergence property of RVI Q-learning discussed earlier, these limits include:

• For each $C _ { \ell , k } \in \mathcal { C }$ , the optimal gain $r _ { \ell , k } ^ { * }$ of the sub-MDP $\mathcal { M } _ { \ell , k } = ( C _ { \ell , k } , \mathcal { A } _ { \ell , k } , p , r )$ , along with a solution $v _ { \ell , k } ^ { * }$ to its AOE:

$$
v _ { \ell , k } ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } _ { \ell , k } ( s ) } \Big \{ r ( s , a ) - r _ { \ell , k } ^ { * } + \sum _ { s ^ { \prime } \in C _ { \ell , k } } p ( s ^ { \prime } \mid s , a ) v _ { \ell , k } ^ { * } ( s ^ { \prime } ) \Big \} , \quad \forall s \in C _ { \ell , k } .\tag{4.2}
$$

We denote by $v _ { * }$ the vector obtained by stacking the vectors $v _ { \ell , k } ^ { * }$ over all $C _ { \ell , k } \in \mathcal { C }$

• The optimal action-value function $\bar { Q } ^ { * }$ of the aggregated OSP $\bar { \mathcal { M } }$ , henceforth denoted by $Q _ { * } ^ { o }$ where the subscript o stands for the OSP.

• The optimal gain function $g ^ { * }$ in the original MDP $\mathcal { M }$

To simplify notation, we write

$$
\mathcal { A } _ { \ell } : = \bigcup _ { k = 1 } ^ { n ( \ell ) } \mathcal { A } _ { \ell , k } , \qquad \mathcal { E } : = \bigcup _ { \ell , k } \mathcal { E } _ { \ell , k } .
$$

Thus, for $s \in C _ { \ell , k } , \mathcal { A } _ { \ell } ( s )$ and $\mathcal E ( s )$ consist of the class-preserving and exiting actions, respectively, w.r.t. the sub-MDP $\mathcal { M } _ { \ell , k }$ on $C _ { \ell , k }$

We first recall several useful relations between the aggregated OSP $\bar { \mathcal { M } }$ and the original MDP M. For each $C _ { \ell , k } \in \mathcal { C }$ and all $s \in C _ { \ell , k }$

$$
g ^ { \ast } ( s ) \geq r _ { \ell , k } ^ { \ast } ,\tag{4.3}
$$

and

$$
\sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) = \left\{ { Q _ { * } ^ { o } } ( { \bar { s } } _ { \ell , k } , ( s , a ) ) \begin{array} { l l } { \forall a \in \mathcal { E } ( s ) ; } \\ { g ^ { * } ( s ) } & { \forall a \in \mathcal { A } _ { \ell } ( s ) . } \end{array} \right.\tag{4.4}
$$

For transient states $s \in \mathcal T$

$$
\sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) = Q _ { * } ^ { o } ( s , a ) , \qquad \forall a \in { \mathcal A } ( s ) .\tag{4.5}
$$

By (4.4) and (4.5), the action constraint set $\mathcal { A } _ { * }$ can be computed by comparing $Q _ { * } ^ { o }$ with $g ^ { * }$ Before doing so, we exploit another piece of information from the aggregated OSP: whether it is gain-optimal to remain in a sub-MDP or to exit it.

In particular, consider a three-way partition $\mathcal { C } ^ { \mathrm { r } } , \mathcal { C } ^ { \mathrm { e } } , \mathcal { C } ^ { \mathrm { b } }$ of ${ \mathcal { C } } ,$ according to whether it is gainoptimal to remain in a sub-MDP, to exit it, or both, respectively. More precisely, for the aggregated OSP $\bar { \mathcal { M } } ,$ consider the maximum optimal Q-value over the continuation (i.e., non-stopping) actions at each aggregated state $\bar { s } _ { \ell , k }$ :

$$
Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) : = \operatorname* { m a x } _ { \bar { a } \neq a _ { \ell , k } ^ { \mathrm { s t o p } } } Q _ { * } ^ { o } ( \bar { s } _ { \ell , k } , \bar { a } ) ,\tag{4.6}
$$

where we set $Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) = - \infty$ if no such actions are available at $\bar { s } _ { \ell , k }$ . Recall that the continuation actions correspond to the exit state-action pairs in $\mathcal { E } _ { \ell , k }$ for the sub-MDP $\mathcal { M } _ { \ell , k }$ , and $Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } )$ is thus the best gain achievable by exiting the sub-MDP. Comparing this value with the optimal gain $r _ { \ell , k } ^ { * }$ achievable by remaining in $\mathcal { M } _ { \ell , k }$ gives the following three-way partition of C.

Three-Way Partition of $\mathcal { C } \mathrm { : }$

$$
\mathcal { C } ^ { \mathrm { r } } : = \{ C _ { \ell , k } \ | \ r _ { \ell , k } ^ { * } > Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) \} ,\tag{4.7}
$$

$$
\mathcal { C } ^ { \mathrm { e } } : = \{ C _ { \ell , k } \ | \ r _ { \ell , k } ^ { * } < Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) \} ,\tag{4.8}
$$

$$
\mathcal { C } ^ { \mathrm { b } } : = \{ C _ { \ell , k } \ | \ r _ { \ell , k } ^ { * } = Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) \} .\tag{4.9}
$$

Note that $\mathcal C ^ { \mathrm { \scriptsize { r } } } \neq \emptyset .$ , since it contains every level-0 set $C _ { 0 , k } .$ , for which no exit actions are available. Moreover,

$$
\begin{array} { r } { g ^ { * } ( s ) = r _ { \ell , k } ^ { * } \quad \mathrm { f o r } \ s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { r } } \cup \mathcal { C } ^ { \mathrm { b } } ; \qquad g ^ { * } ( s ) > r _ { \ell , k } ^ { * } \quad \mathrm { f o r } \ s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } . } \end{array}
$$

Under any stationary optimal policy in the original MDP $\mathcal { M } ,$ all $C \in { \mathcal { C } } ^ { \mathrm { r } }$ are closed, while all $C \in { \mathcal { C } } ^ { \mathrm { e } }$ are transient. However, a set $C \in { \mathcal { C } } ^ { \mathrm { b } }$ may be closed under some stationary optimal policies and transient under others. These properties determine whether the sub-MDP AOE solutions on $C$ can be used to form a global AOE solution, as shown below.

Compute $A _ { * } \colon$ Using the above partition of ${ \mathcal { C } } ,$ together with (4.4) and (4.5), we can express $A _ { * } ( s )$ for each $s \in S$ as follows:

$$
\begin{array} { r } { A _ { * } ( s ) = \left\{ \begin{array} { l l } { A _ { \ell } ( s ) } & { \mathrm { i f ~ } s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathbf { r } } ; } \\ { A _ { \ell } ( s ) \cup \{ a \in \mathcal { E } ( s ) \mid Q _ { * } ^ { o } \left( \bar { s } _ { \ell , k } , ( s , a ) \right) = g ^ { * } ( s ) \} } & { \mathrm { i f ~ } s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathbf { e } } \cup \mathcal { C } ^ { \mathbf { b } } ; } \\ { \{ a \in \mathcal { A } ( s ) \mid Q _ { * } ^ { o } ( s , a ) = g ^ { * } ( s ) \} } & { \mathrm { i f ~ } s \in \mathcal { T } . } \end{array} \right. } \end{array}\tag{4.10}
$$

Observe that, by the definitions of ${ \mathcal { C } } ^ { \mathrm { r } } , { \mathcal { C } } ^ { \mathrm { e } }$ , and ${ \mathcal { C } } ^ { \mathrm { b } }$ , no exiting action can belong to $A _ { * } ( s )$ for any $s \in C _ { \ell , k }$ with $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { r } }$ , while for every $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$

$$
\mathcal A _ { * } ( s ) \cap \mathcal E ( s ) \neq \emptyset \quad \mathrm { f o r ~ s o m e ~ } s \in C _ { \ell , k } .\tag{4.11}
$$

In other words, every sub-MDP in $\mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$ admits a gain-optimal exit.

Extend the sub-MDP AOE solutions on $\mathcal { C } ^ { \mathrm { r } }$ to $s { \mathrm { : } }$ Let $S ^ { \mathrm { r } } , S ^ { \mathrm { e } } , S ^ { \mathrm { b } }$ be the unions of the sub-MDP state spaces in the corresponding categories of the partition:

$$
\mathcal { S } ^ { \mathrm { r } } : = \bigcup _ { C \in \mathcal { C } ^ { \mathrm { r } } } C , \qquad \mathcal { S } ^ { \mathrm { e } } : = \bigcup _ { C \in \mathcal { C } ^ { \mathrm { e } } } C , \qquad \mathcal { S } ^ { \mathrm { b } } : = \bigcup _ { C \in \mathcal { C } ^ { \mathrm { b } } } C ,
$$

and denote $( S ^ { \mathrm { r } } ) ^ { c } : = S \setminus S ^ { \mathrm { r } }$ . Define $v _ { * , \mathrm { r } } : = v _ { * } | _ { S ^ { \mathrm { r } } }$ , i.e.,

$$
v _ { * , \mathrm { r } } ( s ) = v _ { \ell , k } ^ { * } ( s ) \quad \mathrm { f o r } s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { r } } .
$$

For $s \in C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { r } }$ , we have $r _ { \ell , k } ^ { * } = g ^ { * } ( s )$ and $\mathcal { A } _ { * } ( s ) = \mathcal { A } _ { \ell } ( s )$ by (4.10), and any action in $\mathcal { A } _ { \ell } ( s )$ keeps the system within $C _ { \ell , k }$ . Therefore, by (4.2), $v _ { * , 1 }$ satisfies the AOE (4.1) on $S ^ { \mathrm { r } }$

$$
v _ { * , \mathrm { r } } ( s ) = \operatorname* { m a x } _ { a \in A _ { * } ( s ) } \left\{ r ( s , a ) - g ^ { * } ( s ) + \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) v _ { * , \mathrm { r } } ( s ^ { \prime } ) \right\} , \quad \forall s \in \mathcal { S } ^ { \mathrm { r } } .\tag{4.12}
$$

Thus, extending $v _ { * , \mathrm { r } }$ to a full AOE solution amounts to solving the remaining equations in the AOE:

$$
y ( s ) = \operatorname* { m a x } _ { a \in A _ { \infty } ( s ) } \left\{ r ( s , a ) - g ^ { * } ( s ) + \sum _ { s ^ { \prime } \in S ^ { r } } p ( s ^ { \prime } \mid s , a ) v _ { * , \tau } ( s ^ { \prime } ) + \sum _ { s ^ { \prime } \in S \backslash S ^ { r } } p ( s ^ { \prime } \mid s , a ) y ( s ^ { \prime } ) \right\} , \forall s \in S \backslash \mathcal { S } ^ { r } .\tag{4.13}
$$

As Prop. 4.1 below shows, this system of equations have a solution ${ \bar { y } } .$ . Hence, $( v _ { * , \mathrm { r } } , \bar { y } )$ solves the AOE (4.1). This completes the description of the idealized algorithm.

Proposition 4.1. Equation (4.13) admits a solution, and a unique solution $i f { \mathcal { C } } ^ { \mathrm { b } } = \emptyset$ . For any solution ${ \bar { y } } ,$ the pair $( v _ { * , \mathrm { r } } , \bar { y } )$ solves the AOE (4.1).

In the remainder of this subsection, we prove Prop. 4.1. While this can also be shown using non-constructive analytical arguments (see Remark 4.1), we instead use a penalty-based proof: we penalize states in $\begin{array} { r } { S ^ { \mathrm { b } } = \bigcup _ { C \in \mathcal { C } ^ { \mathrm { b } } } C , } \end{array}$ , relate the penalized version of (4.13) to the optimality equation of an SSP-type total-reward MDP, and then let the penalty parameter tend to 0. These arguments underlie the Q-learning step in Algorithm 2, introduced next. There, analogous penalties are applied to outer approximations of $S ^ { \mathrm { b } }$ to ensure the stability of Q-learning in the undiscounted total-reward setting.

For each $\epsilon > 0 .$ , consider the equation obtained from (4.13) by subtracting an amount of ϵ from the rewards for states in $S ^ { \mathrm { b } } { } ;$ : for all $s \in { \mathcal { S } } \setminus { \mathcal { S } } ^ { \mathrm { r } }$ 2

$$
y ( s ) = \operatorname* { m a x } _ { a \in A _ { \infty } ( s ) } \Big \{ r ( s , a ) - g ^ { * } ( s ) - \epsilon { \bf 1 } _ { S ^ { \mathrm { b } } } ( s ) + \sum _ { s ^ { \prime } \in S ^ { \tau } } p ( s ^ { \prime } \mid s , a ) v _ { * , \tau } ( s ^ { \prime } ) + \sum _ { s ^ { \prime } \in S \backslash S ^ { \tau } } p ( s ^ { \prime } \mid s , a ) y ( s ^ { \prime } ) \Big \} .\tag{4.14}
$$

We show that (4.14) is the optimality equation of an SSP-type total-reward MDP $\mathcal { M } ^ { \epsilon }$ and therefore admits a unique solution $y _ { \epsilon }$ by Theorem 2.1. This also establishes the uniqueness assertion in Prop. 4.1: if $\mathcal { C } ^ { \mathrm { b } } = \emptyset$ , then $S ^ { \mathrm { b } } = \emptyset$ , and (4.14) reduces to (4.13).

We define the MDP $\mathcal { M } ^ { \epsilon } : = \{ \hat { S } \cup \{ \Delta \} , \hat { \mathcal { A } } , \hat { p } , \hat { r } \}$ as follows.

• Let ${ \mathcal { S } } \backslash S ^ { \mathrm { r } }$ be the efective state space ${ \hat { S } } ,$ , and aggregate the states in $S ^ { \mathrm { r } }$ into a special reward-free termination state $\Delta$

• Let the action constraints be $\hat { \boldsymbol A } ( s ) = A _ { * } ( s ) , s \in \hat { \mathcal { S } } .$

• For each feasible state-action pair $( s , a ) \in { \hat { A } }$ , define the transition probabilities $\hat { p } ( \cdot \mid s , a )$ by

$$
\hat { p } ( s ^ { \prime } \mid s , a ) = p ( s ^ { \prime } \mid s , a ) , \quad s ^ { \prime } \in \hat { S } , \qquad \hat { p } ( \Delta \mid s , a ) = \sum _ { s ^ { \prime } \in S ^ { \tau } } p ( s ^ { \prime } \mid s , a ) ,\tag{4.15}
$$

and define the expected one-stage reward ${ \hat { r } } ( s , a )$ by

$$
\hat { r } ( s , a ) = r ( s , a ) - g ^ { * } ( s ) - \epsilon \mathbf { 1 } _ { \mathcal { S } ^ { \mathrm { b } } } ( s ) + \sum _ { s ^ { \prime } \in \mathcal { S } ^ { \mathrm { r } } } p ( s ^ { \prime } \mid s , a ) v _ { * , \mathrm { r } } ( s ^ { \prime } ) .\tag{4.16}
$$

Thus, transitions into $S ^ { \mathrm { r } }$ in the original MDP are replaced by transitions to the termination state $\Delta ,$ , while the expected contribution from the corresponding values $v _ { * , \mathrm { r } } ( s ^ { \prime } )$ is incorporated into the one-stage reward rˆ. Equivalently, the last term in ${ \hat { r } } ( s , a )$ can be viewed as the expected terminal reward associated with a transition into $S ^ { \mathrm { r } }$ , with $v _ { * , \mathrm { r } } ( s ^ { \prime } )$ serving as the terminal reward associated with $s ^ { \prime } \in S ^ { \mathrm { r } }$

## Lemma 4.1. The MDP M<sup>ϵ</sup> defined above satisfies Assumption 2.1 and is therefore of $S S P$ type.

Proof. We verify the two conditions in Assumption 2.1 for $\mathcal { M } ^ { \epsilon }$ . The existence of a proper policy as required by Assumption 2.1(i) follows by combining the communicating property of each $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$ with the existence of an admissible exit action for each such sub-MDP. Specifically, by (4.11), for each $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$ , there exists a state $s _ { \ell , k } \in C _ { \ell , k }$ such that $\mathcal { A } _ { \ast } ( s _ { \ell , k } ) \cap \mathcal { E } ( s _ { \ell , k } ) \neq \emptyset$ Since the sub-MDP $\mathcal { M } _ { \ell , k } = ( C _ { \ell , k } , \mathcal { A } _ { \ell , k } , p , r )$ is communicating, it has a policy $\mu _ { \ell , k } \in \Pi _ { \mathcal { A } _ { \ell , k } }$ under which the state $s _ { \ell , k }$ is reached with probability one from every initial state in $C _ { \ell , k }$ . Define a policy $\mu \in \Pi _ { \hat { A } }$ for $\mathcal { M } ^ { \epsilon }$ as follows:

(i) on each $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$ , let $\mu$ coincide with $\mu _ { \ell , k }$ except at the state $s _ { \ell , k }$ , where $\mu ( s _ { \ell , k } ) \in$ $\mathcal { A } _ { \ast } ( s _ { \ell , k } ) \cap \mathcal { E } ( s _ { \ell , k } ) ;$ and

(ii) for all $s \in \mathcal T$ , let $\mu ( s ) \in \mathcal { A } _ { * } ( s )$

(That $\mu \in \Pi _ { \hat { A } }$ follows from the definition of $\hat { A }$ and the fact that, for all $s \in C _ { \ell , k } , \mu _ { \ell , k } ( s ) \in \mathcal { A } _ { \ell , k } ( s ) \subset$ $\mathcal { A } _ { * } ( s ) . )$ By the defining properties of $\mathcal { E }$ and $\tau$ in Bather’s decomposition, under $\mu$ the system eventually exits the efective state space $S \setminus S ^ { \mathrm { r } }$ with probability one, regardless of the initial state. Thus, $\mu$ is proper, verifying Assumption 2.1(i).

For Assumption $2 . 1 ( \mathrm { i i } )$ , we use the property of Bather’s decomposition given in Lemma $3 . 1 ( \mathrm { i } )$ to locate a recurrent class of an improper policy, and show that its gain in $\mathcal { M } ^ { \epsilon }$ on this recurrent class is strictly negative. Specifically, let $\mu \in \Pi _ { \hat { A } }$ be an improper policy for $\mathcal { M } ^ { \epsilon }$ . Then $\mu$ induces at least one recurrent class $D \subset { \hat { S } }$ . For every $s \in D , \hat { p } ( \Delta \mid s , \mu ( s ) ) = 0$ and hence, by (4.15), $p ( s ^ { \prime } \mid s , \mu ( s ) ) = 0$ for all $s ^ { \prime } \in S ^ { \mathrm { r } }$ . Therefore, by (4.16),

$$
\hat { r } ( s , \mu ( s ) ) = r ( s , \mu ( s ) ) - g ^ { * } ( s ) - \epsilon \mathbf { 1 } _ { \mathcal { S } ^ { \mathrm { b } } } ( s ) , \qquad s \in D .\tag{4.17}
$$

By Lemma 3.1(i), D is contained in some $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } } \cup \mathcal { C } ^ { \mathrm { b } }$ . Together with (4.17), this implies that the gain $\hat { g } _ { \mu }$ of $\mu$ in $\mathcal { M } ^ { \epsilon }$ is strictly negative on D. Indeed, if $D \subset C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { e } }$ , then $r _ { \ell , k } ^ { \ast } < g ^ { \ast } ( s )$ for $s \in C _ { \ell , k }$ and thus

$$
\hat { g } _ { \mu } ( s ) \leq r _ { \ell , k } ^ { * } - g ^ { * } ( s ) < 0 , \qquad s \in D .
$$

If $D \subset C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { b } }$ , then $r _ { \ell , k } ^ { * } = g ^ { * } ( s )$ for $s \in C _ { \ell , k }$ , and therefore

$$
\begin{array} { r } { \hat { g } _ { \mu } ( s ) \le r _ { \ell , k } ^ { * } - g ^ { * } ( s ) - \epsilon = - \epsilon < 0 , \qquad s \in D . } \end{array}
$$

Consequently, by the definition of the gain of a policy, starting from any $s \in D$ , the expected cumulative n-stage reward under $\mu$ tends to −∞ as $n \to \infty$ . Hence $V _ { \mu } ( s ) = - \infty$ for $s \in D$ , verifying Assumption 2.1(ii). □

Proof of Prop. 4.1. We prove the general statement in the proposition. As discussed earlier, the uniqueness assertion when $\mathcal { C } ^ { \mathrm { b } } = \emptyset$ follows directly from Lemma 4.1 and Theorem 2.1.

For each $\epsilon > 0 , \mathcal { M } ^ { \epsilon }$ is a total-reward MDP of SSP type (Lemma 4.1), and its optimality equation (4.14) therefore admits a unique solution $y _ { \epsilon }$ (Theorem 2.1). By the definition of $\mathcal { M } ^ { \epsilon }$ , the family $\{ y _ { \epsilon } \} _ { \epsilon > 0 }$ is nondecreasing as $\epsilon \downarrow 0$ . Moreover, the optimal value $y _ { \epsilon }$ is attained by a proper stationary policy in $\Pi _ { \hat { \mathcal { A } } }$ (Theorem 2.1). Hence $\{ y _ { \epsilon } \} _ { \epsilon > 0 }$ is uniformly bounded above by $\| v _ { * , \mathrm { r } } \| _ { \infty }$ plus the maximum total expected reward attainable in the original MDP before hitting $S ^ { \mathrm { r } }$ , starting from states outside $S ^ { \mathrm { r } }$ , under any nonrandomized stationary policy that renders all states outside $S ^ { \mathrm { r } }$ transient. Therefore, $y _ { \epsilon } \uparrow \bar { y } \mathrm { ~ a s ~ } \epsilon \downarrow 0$ for some $\bar { y } \in \mathbb { R } ^ { | \boldsymbol { S } \backslash \boldsymbol { S } ^ { \mathrm { r } } | }$ . Since $y _ { \epsilon }$ satisfies (4.14), letting $\epsilon \downarrow 0$ and using continuity shows that $\bar { y }$ satisfies (4.13).

Finally, as discussed earlier, combining (4.13) with the AOEs for the sub-MDPs $\mathcal { M } _ { \ell , k }$ corresponding to $C _ { \ell , k } \in \mathcal { C } ^ { \mathrm { r } }$ shows that any solution y¯ of (4.13), together with $v _ { * , \mathrm { r } }$ , solves the system of equations in (4.12)–(4.13) and therefore solves the AOE (4.1). □

Remark 4.1. A non-constructive alternative for proving the existence of a solution to (4.13) is to regard this equation as a possible AOE for an MDP on ${ \bar { \boldsymbol { s } } } \setminus { \boldsymbol { s } } ^ { \mathrm { r } }$ and show that the gain of this MDP is zero. The latter can be established using arguments similar to those in the proof of Lemma 4.1.

## 4.2 An RL algorithm

We now introduce an RL algorithm for approximately solving the AOE (4.1). The algorithm mimics the idealized algorithm of Section 4.1. Here, however, only the convergent iterates produced by Algorithm 1 are available, not their limits. Specifically, we have the following convergent sequences:

• For each $C _ { \ell , k } \ \in \ \mathcal { C }$ , Algorithm 1, with Phase 2 configured as described earlier, produces $( r _ { \ell , k } ^ { ( t ) } , v _ { \ell , k } ^ { ( t ) } ) _ { t \geq 0 }$ satisfying

$$
\begin{array} { r } { r _ { \ell , k } ^ { ( t ) } \to r _ { \ell , k } ^ { * } , \qquad v _ { \ell , k } ^ { ( t ) } \to v _ { \ell , k } ^ { * } , \quad a . s . , } \end{array}
$$

where $r _ { \ell , k } ^ { * }$ is the optimal gain and $v _ { \ell , k } ^ { * }$ a sample-path-dependent AOE solution for the sub-MDP $\mathcal { M } _ { \ell , k }$ . (As noted earlier, this follows from the convergence results for RVI Q-learning in [39, Thm. 2.3] and [38, Thm. 3.2].) Stacking these vectors $v _ { \ell , k } ^ { ( t ) }$ over all $C _ { \ell , k } \in \mathcal { C }$ to define $v _ { t } ,$ , we have

$$
v _ { t } : = \left( \boldsymbol { v } _ { \ell , k } ^ { ( t ) } \right) _ { \ell , k } \ \to \ v _ { * } = \left( \boldsymbol { v } _ { \ell , k } ^ { * } \right) _ { \ell , k } , \quad a . s . ,
$$

where $v _ { * }$ is the corresponding limiting vector in the idealized algorithm.

• For the aggregated OSP $\bar { \mathcal { M } } ,$ by Theorem 3.1, Algorithm 1 (Phase 3) produces sequences $Q _ { t } ^ { o } : = \bar { Q } ^ { ( t ) }$ and $g _ { t } : = g ^ { ( t ) }$ , satisfying

$$
Q _ { t } ^ { o } \to Q _ { * } ^ { o } , \qquad g _ { t } \to g ^ { * } , \quad a . s .
$$

• For each $C _ { \ell , k } \in \mathcal { C }$ and $t \geq 0$ , consider the aggregated state $\bar { s } _ { \ell , k }$ in the OSP and its estimated maximum $Q -$ -value over the continuation actions,

$$
Q _ { t , c } ^ { o } ( \bar { s } _ { \ell , k } ) : = \operatorname * { m a x } _ { \bar { a } \neq a _ { \ell , k } ^ { \mathrm { s t o p } } } Q _ { t } ^ { o } ( \bar { s } _ { \ell , k } , \bar { a } ) ,
$$

with $Q _ { t , c } ^ { o } ( \bar { s } _ { \ell , k } ) : = - \infty$ if stopping is the only action available. From the convergence of {Q<sup>o</sup><sub>t</sub> } it follows that

$$
Q _ { t , c } ^ { o } \to Q _ { c } ^ { o } , \quad a . s . ,
$$

where the limit $Q _ { c } ^ { o }$ is as in (4.6) and determines the three-way partition of C in the idealized algorithm.

We use these sequences to approximate the steps of the idealized algorithm.

The resulting RL algorithm is given in Algorithm 2. In a nutshell, it uses the outputs of Algorithm 1 to concurrently approximate the three main steps of the idealized algorithm:

(i) Construct action constraint sets $\tilde { \mathcal { A } } _ { t }$ that eventually outer-approximate $\mathcal { A } _ { * }$

(ii) Approximate the three-way partition of C by $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } , \tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } } , \tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ . In particular, eventually, $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } \subset \mathcal { C } ^ { \mathrm { r } }$ and $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } } \subset \mathcal { C } ^ { \mathrm { e } }$ , while $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } } \supset \mathcal { C } ^ { \mathrm { b } }$ , with $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ collecting sub-MDPs for which both remaining and exiting are nearly gain-optimal.

(iii) Eventually construct an approximate global AOE solution that extends the sub-MDP solutions on $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } }$ . In particular, total-reward Q-learning is employed to compute the remaining components of the approximate AOE solution, with rewards shifted by $- g _ { t }$ from Algorithm 1 and states associated with $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } }$ treated as terminal, with terminal rewards given by their estimated sub-MDP solutions from Algorithm 1. States associated with $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ are further penalized by ϵ to ensure stability of $\mathrm { Q } \mathrm { . }$ -learning [29, 36], analogously to the penalty argument used in the proof of Prop. 4.1.

A technical complication arises in constructing $\tilde { \mathcal { A } } _ { t }$ and $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } , \tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } } , \tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ . Since these sets are determined by comparisons involving convergent random iterates from Algorithm 1, using a single threshold can cause them to keep changing indefinitely when the limiting value being compared lies exactly on the threshold. To prevent this, Algorithm 2 uses a pair of thresholds $\underline { { \delta } } < \bar { \delta }$ and separate rules for adding and retaining elements. The separation between the two thresholds ensures that a convergent sequence cannot cross both infinitely often. This two-threshold mechanism ensures finite-time stabilization $( { \mathrm { i . e . } }$ , eventual constancy) of these sets (see Lemma 4.2 below). It also contributes to the apparently more elaborate form of the set-construction rules below, despite the simplicity of the underlying mechanism.

For Algorithms 2 and 3, we adopt the following convention:

Information Structure Convention: All variables are defined on a common probability space, and all updates are non-anticipatory $( { \mathrm { i . e . } }$ , depend only on past and current information). Dependencies across algorithms and phases are specified by the information-flow statements preceding each algorithm.

We now describe the construction of the relevant sets and the Q-learning step in Algorithm 2. Its starting time, parameters, and stepsizes may be chosen adaptively based on past information, consistent with the information-flow structure.

Algorithm 2 (RL algorithm for approximate AOE solutions).

Information flow: Algorithm $2  \mathrm { A }$ lgorithm 1.

Initialization: Choose starting time $\bar { t } ,$ parameters $\epsilon > 0 , \bar { \delta } > \underline { { { \delta } } } > 0$ , and $Q _ { \bar { t } } \in \mathbb { R } ^ { | \mathcal { A } | }$

For $t \geq \bar { t } ,$ perform the following operations:

Set construction:

1. Construct $\tilde { \mathcal { A } } _ { t }$ to approximate $\mathcal { A } _ { * }$ by comparing $Q _ { t } ^ { o }$ with $g _ { t }$ .

For $s \in C _ { \ell , k } \in \mathcal { C }$ , all class-preserving actions $\mathcal { A } _ { \ell } ( s )$ belong to $A _ { * } ( s ) { \mathrm { ; } }$ ; hence, only the exiting actions from ${ \mathcal { E } } ( s )$ need to be selected. For $s \in \mathcal T$ , all actions need to be considered. Define the selected sets $\boldsymbol { \mathcal { A } } _ { t } ^ { e } ( \boldsymbol { s } )$ , initialized by $\mathcal { A } _ { \bar { t } - 1 } ^ { e } ( s ) = \emptyset$ , using the parameters $( \underline { { \delta } } , \bar { \delta } )$ as follows:

$$
\forall s \in C _ { \ell , k } \in \mathcal { C } : \quad \mathcal { A } _ { t } ^ { e } ( s ) : = \big \{ a \in \mathcal { E } ( s ) \setminus \mathcal { A } _ { t - 1 } ^ { e } ( s ) \mid Q _ { t } ^ { o } ( \bar { s } _ { \ell , k } , ( s , a ) ) > g _ { t } ( s ) - \underline { { \delta } } \big \}
$$

$$
\cup \big \{ a \in \mathcal { A } _ { t - 1 } ^ { e } ( s ) \mid Q _ { t } ^ { o } ( \bar { s } _ { \ell , k } , ( s , a ) ) > g _ { t } ( s ) - \bar { \delta } \big \} ;\tag{4.18}
$$

$$
\forall s \in { \mathcal { T } } : \quad { \mathcal { A } } _ { t } ^ { e } ( s ) : = \big \{ a \in { \mathcal { A } } ( s ) \setminus { \mathcal { A } } _ { t - 1 } ^ { e } ( s ) \mid Q _ { t } ^ { o } ( s , a ) > g _ { t } ( s ) - \underline { { \delta } } \big \}
$$

$$
\cup \big \{ a \in \mathcal { A } _ { t - 1 } ^ { e } ( s ) \mid Q _ { t } ^ { o } ( s , a ) > g _ { t } ( s ) - \bar { \delta } \big \} .\tag{4.19}
$$

The first set in each union adds new elements using $\underline { { \delta } } ,$ while the second retains existing elements using ${ \bar { \delta } } .$ We then define

$$
\tilde { \mathcal { A } } _ { t } ( s ) : = \mathcal { A } _ { \ell } ( s ) \cup \mathcal { A } _ { t } ^ { e } ( s ) \quad \forall s \in C _ { \ell , k } \in \mathcal { C } ; \qquad \tilde { \mathcal { A } } _ { t } ( s ) : = \mathcal { A } _ { t } ^ { e } ( s ) \quad \forall s \in \mathcal { T } .\tag{4.20}
$$

2. Construct a three-way partition of $\mathcal { C }$ that approximates the idealized partition, by comparing $Q _ { t , c } ^ { o }$ with $\{ r _ { \ell , k } ^ { ( t ) } \} _ { \ell , k }$

Define a partition of C as follows, with $\tilde { \mathcal { C } } _ { \bar { t } - 1 } ^ { \mathrm { e } } = \emptyset$ initially:

$$
\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } : = \big \{ C _ { \ell , k } \in \mathcal { C } \ | \ A _ { t } ^ { e } ( s ) = \varnothing , \forall s \in C _ { \ell , k } \big \} ,\tag{4.21}
$$

$$
\begin{array} { r } { \tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } } : = \left\{ C _ { \ell , k } \in \mathcal { C } \setminus \tilde { \mathcal { C } } _ { t - 1 } ^ { \mathrm { e } } \mid r _ { \ell , k } ^ { ( t ) } < Q _ { t , c } ^ { o } ( \bar { s } _ { \ell , k } ) - \bar { \delta } \right\} \cup \left\{ C _ { \ell , k } \in \tilde { \mathcal { C } } _ { t - 1 } ^ { \mathrm { e } } \mid r _ { \ell , k } ^ { ( t ) } < Q _ { t , c } ^ { o } ( \bar { s } _ { \ell , k } ) - \underline { { \delta } } \right\} , } \end{array}\tag{4.22}
$$

$$
\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } } : = \mathcal { C } \setminus \big ( \tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } \cup \tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } } \big ) .\tag{4.23}
$$

The two-threshold rule for $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { e } }$ uses $\bar { \delta }$ for adding new elements and $\underline { { \delta } }$ for retaining existing ones;   
$\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ is then determined by complement.

Q-learning step: We apply total-reward Q-learning, shifting rewards by $- g _ { t }$ , treating $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } } { \mathrm { - s t a t e s } }$ as terminal states with terminal reward $v _ { t }$ , and further penalizing $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } }$ -states by ϵ. In particular:

1. Let $\begin{array} { r } { \tilde { S } _ { t } ^ { i } : = \bigcup _ { C \in \tilde { \mathcal { C } } _ { t } ^ { i } } } \end{array}$ C for $i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \}$

2. Sample a subset $Y _ { t } \subset \tilde { \Gamma } _ { t } : = \{ ( s , a ) \in \tilde { \mathcal { A } } _ { t } \mid s \not \in \tilde { \mathcal { S } } _ { t } ^ { \mathrm { r } } \}$

For each $( s , a ) \in Y _ { t }$ , sample a transition to obtain reward $R _ { t + 1 } ^ { s a }$ and next state $S _ { t + 1 } ^ { s a }$

3. Update $Q _ { t }$ for each $( s , a ) \in Y _ { t }$ by

$$
\begin{array} { r l r } & { } & { Q _ { t + 1 } ( s , a ) = Q _ { t } ( s , a ) + \alpha _ { t } ( s , a ) \Big ( R _ { t + 1 } ^ { s a } - g _ { t } ( s ) - \epsilon \mathbb { 1 } \big \lbrace s \in \tilde { \mathcal { S } } _ { t } ^ { \mathrm { b } } \big \rbrace + \mathbb { 1 } \big \lbrace S _ { t + 1 } ^ { s a } \in \tilde { \mathcal { S } } _ { t } ^ { \mathrm { r } } \big \rbrace \cdot v _ { t } ( S _ { t + 1 } ^ { s a } ) } \\ & { } & { \qquad + \mathbb { 1 } \big \lbrace S _ { t + 1 } ^ { s a } \notin \tilde { \mathcal { S } } _ { t } ^ { \mathrm { r } } \big \rbrace \cdot \displaystyle \operatorname* { m a x } _ { a ^ { \prime } \in \tilde { \mathcal { A } } _ { t } ( S _ { t + 1 } ^ { s a } ) } Q _ { t } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } ) - Q _ { t } ( s , a ) \Big ) , } \end{array}\tag{4.24}
$$

where $\alpha _ { t } ( s , a ) \geq 0$ is a possibly random stepsize that can depend on information available at time $t ,$ including $Y _ { t }$ (but not on the transition data indexed by t + 1).

This completes the update at iteration t.

## 4.3 Convergence and performance results

Let Algorithm 2 be run under the following conditions, which are assumed throughout this subsection unless stated otherwise:

(C1) Algorithm 1 satisfies its convergence conditions. Furthermore, its Phase 2 outputs converge a.s. to an AOE solution for each $\mathcal { M } _ { \ell , k }$ , as specified earlier.

(C2) The information-flow structure indicated in Algorithm 2 is respected. (An explicit formal statement of this condition in terms of stochastic kernels is given in Assumption 8.1, Section 8.)

(C3) Standard Q-learning data and stepsize conditions hold a.s.:

(i) Any state-action pair that occurs in $\tilde { \Gamma } _ { t }$ infinitely often also occurs in $Y _ { t }$ infinitely often.

(ii) For any $( s , a )$ that occurs infinitely often in $Y _ { t }$ , the stepsizes satisfy $\textstyle \sum _ { t } \alpha _ { t } ( s , a ) = \infty$ and $\textstyle \sum _ { t } \alpha _ { t } ^ { 2 } ( s , a ) < \infty$ , under the convention that $\alpha _ { t } ( s , a ) = 0$ whenever $( s , a ) \notin Y _ { t }$

The next lemma establishes finite-time stabilization of the sets constructed by Algorithm 2 and summarizes the properties of their limiting sets. In particular, the approximation properties stated below are precisely those sought in the algorithmic construction outlined earlier and will also be used in the convergence analysis. The lemma follows directly from the update rules (4.18)–(4.23), the convergences $Q _ { t } ^ { o } \to Q _ { * } ^ { o } , g _ { t } \to g ^ { * }$ , and $r _ { \ell , k } ^ { ( t ) } \to r _ { \ell , k } ^ { \ast }$ from Algorithm 1, and the relations (4.3)–(4.5).

Lemma 4.2 (Finite-time set stabilization). Under $( C 1 )$ , almost surely, the sets $\tilde { \mathcal { C } } _ { t } ^ { i } , \tilde { \mathcal { S } } _ { t } ^ { i } ~ ( i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \} ) .$ $\tilde { \mathcal { A } } _ { t }$ , and $\tilde { \Gamma } _ { t }$ remain unchanged after finitely many iterations. Denote the resulting sets $b y \tilde { \mathcal { C } } ^ { i } , \tilde { \mathcal { S } } ^ { i } , \tilde { \mathcal { A } }$ and ${ \tilde { \Gamma } } ;$ then $\tilde { \mathcal { C } } ^ { \mathrm { r } } , \tilde { \mathcal { C } } ^ { \mathrm { e } } , \tilde { \mathcal { C } } ^ { \mathrm { b } }$ form a partition of ${ \mathcal { C } } ,$ and

$$
\tilde { \mathcal { S } } ^ { i } = \bigcup _ { C \in \tilde { \mathcal { C } } ^ { i } } C , \quad i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \} ; \qquad \tilde { \Gamma } = \{ \left( s , a \right) \in \tilde { \mathcal { A } } \mid s \not \in \tilde { \mathcal { S } } ^ { \mathrm { r } } \} .
$$

Moreover, the following hold:

$$
\tilde { A } ( s ) = A _ { * } ( s ) \forall s \in \tilde { \mathcal { S } } ^ { \mathrm { r } } , \qquad A _ { * } ( s ) \subset \tilde { \mathcal { A } } ( s ) \subset A _ { \bar { \delta } } ( s ) \forall s \notin \tilde { \mathcal { S } } ^ { \mathrm { r } } ,\tag{4.25}
$$

$$
\mathcal { C } _ { 0 } \subset \tilde { \mathcal { C } } ^ { \mathrm { r } } \subset \mathcal { C } ^ { \mathrm { r } } , \qquad \tilde { \mathcal { C } } ^ { \mathrm { e } } \subset \mathcal { C } ^ { \mathrm { e } } , \qquad \tilde { \mathcal { C } } ^ { \mathrm { b } } \supset \mathcal { C } ^ { \mathrm { b } } .\tag{4.26}
$$

For each $C _ { \ell , k } \in \mathcal { C } _ { : }$

$$
r _ { \ell , k } ^ { * } - Q _ { c } ^ { o } ( \bar { s } _ { \ell , k } ) \in \left\{ \begin{array} { l l } { [ \underline { { \delta } } , \infty ) } & { i f \ C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { r } } ; } \\ { ( - \infty , - \underline { { \delta } } ] } & { i f \ C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { e } } ; } \\ { [ - \bar { \delta } , \bar { \delta } ] } & { i f \ C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { b } } . } \end{array} \right.\tag{4.27}
$$

Furthermore, $\tilde { \mathcal { A } } ( s ) = \mathcal { A } _ { \ell } ( s )$ for all $s \in C _ { \ell , k }$ with $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { r } }$ , and for every $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { e } } \cup \tilde { \mathcal { C } } ^ { \mathrm { b } }$

$$
\tilde { \mathcal { A } } ( s ) \cap \mathcal { E } ( s ) \neq \emptyset f o r s o m e s \in C _ { \ell , k } .\tag{4.28}
$$

The finite-time stabilization in Lemma 4.2 allows the asymptotic behavior of the Q-learning step to be characterized in terms of the resulting fixed sets and a sample-path-dependent, SSP-type total-reward MDP determined by these sets, $v _ { * }$ , and the penalty parameter ϵ. The resulting convergence theorem is stated below; its proof is given in Section 8.

Theorem 4.1 (Convergence of Algorithm 2). Let $\tilde { \mathcal { C } } ^ { i } , \tilde { \mathcal { S } } ^ { i } \ ( i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \} ) , \tilde { \mathcal { A } } ,$ and $\tilde { \Gamma }$ be the samplepath-dependent, eventually fixed sets guaranteed by Lemma 4.2. Then the sequence $\{ Q _ { t } \}$ generated by Algorithm 2 converges a.s. componentwise on Γ<sup>˜</sup> to the unique solution $\hat { Q } ^ { * }$ of

$$
\begin{array} { l } { { \displaystyle { \cal Q } ( s , a ) = r ( s , a ) - g ^ { * } ( s ) - \epsilon { \bf 1 } _ { \tilde { \mathcal { S } } ^ { \mathrm { b } } } ( s ) + \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } ^ { \mathrm { r } } } p ( s ^ { \prime } \mid s , a ) v _ { * } ( s ^ { \prime } ) } } \\ { { + \displaystyle \sum _ { s ^ { \prime } \notin \tilde { \mathcal { S } } ^ { \mathrm { r } } } p ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in \tilde { \mathcal { A } } ( s ^ { \prime } ) } { \cal Q } ( s ^ { \prime } , a ^ { \prime } ) , } } \end{array}\tag{4.29}
$$

Consequently, if $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { | \boldsymbol { S } | }$ is define by

$$
\begin{array} { r } { x _ { t } ( s ) : = \left\{ \begin{array} { l l } { v _ { t } ( s ) } & { i f s \in \tilde { \cal S } _ { t } ^ { \mathrm { r } } , } \\ { \operatorname* { m a x } _ { a \in \tilde { \cal A } _ { t } ( s ) } Q _ { t } ( s , a ) } & { i f s \notin \tilde { \cal S } _ { t } ^ { \mathrm { r } } , } \end{array} \right. } \end{array}\tag{4.30}
$$

then $\{ x _ { t } \}$ converges a.s. to a solution x¯ of the perturbed $A O E ,$

$$
x = \operatorname* { m a x } _ { \mu \in \Pi _ { \tilde { A } } } \left\{ r _ { \mu } - g ^ { * } + P _ { \mu } x \right\} - \epsilon \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } ,\tag{4.31}
$$

and $\bar { x }$ is the unique solution that coincides with $v _ { * }$ on ${ \tilde { S } } ^ { \mathrm { r } }$ . Finally, there exists a deterministic constant $\bar { \delta } _ { 0 } > 0$ such that, for every $\bar { \delta } < \bar { \delta } _ { 0 } , \tilde { \mathcal { A } } = \mathcal { A } ,$ <sub>∗</sub> and the corresponding solution x¯ converges, as $\epsilon  0$ , to a solution of the AOE (4.1).

Thus, the theorem shows that, as a consequence of the Q-learning convergence, Algorithm 2 asymptotically constructs a solution to a sample-path-dependent perturbed AOE that extends the sub-MDP AOE solutions on ${ \tilde { S } } ^ { \mathrm { r } }$ , and thereby approximately solves the AOE (4.1). The relaxation of the constraint set $\mathcal { A } _ { * }$ to $\tilde { \cal A }$ and the penalty ϵ on $\tilde { S } ^ { \mathrm { b } }$ both contribute to the approximation error. If $\mathcal { C } ^ { \mathrm { b } } = \mathcal { D } .$ , i.e., there is no sub-MDP in the third category where remaining and exiting are both gain-optimal, then for suficiently small $\bar { \delta } , \tilde { S } ^ { \mathrm { b } } = \emptyset$ by Lemma 4.2, $\tilde { \mathcal { A } } = \mathcal { A } _ { \ast }$ , and the algorithm asymptotically solves the AOE exactly.

The approximate AOE solutions targeted by Algorithm 2, as well as its iterates, yield policies whose suboptimality in gain can be bounded in terms of the algorithmic parameter $\bar { \delta } ,$ which controls the accuracy of the outer approximations of both $\mathcal { A } _ { * }$ and $\mathcal { C } ^ { \mathrm { b } }$ . We next quantify these performance guarantees for suficiently large t.

Define an induced greedy policy $\mu _ { t } \in \Pi _ { A }$ as follows:

• for $\begin{array} { r } { s \not \in \tilde { S } _ { t } ^ { \mathrm { r } } , \mathrm { l e t } \mu _ { t } ( s ) \in \mathrm { a r g } \mathrm { m a x } _ { a \in \tilde { A } _ { t } ( s ) } Q _ { t } ( s , a ) ; } \end{array}$

• for $s \in \tilde { S } _ { t } ^ { \mathrm { r } }$ , let $\mu _ { t }$ coincide on each $C _ { \ell , k } \in \tilde { \mathcal { C } } _ { t } ^ { \mathrm { r } }$ with $\pi _ { \ell , k } ^ { ( t ) }$ , the policy for the sub-MDP $\mathcal { M } _ { \ell , k }$ produced by RVI Q-learning in Phase 2 of Algorithm 1.

For $B \subset S$ , let $\tau _ { B }$ denote the hitting time of $B$ for the state process $\{ S _ { n } \} _ { n \geq 0 } ;$ let $\bar { \tau } _ { B } ^ { \mu } ( s )$ denote its expected value under a policy $\mu ,$ starting from $S _ { 0 } = s$ . That is,

$$
\tau _ { B } : = \mathrm { m i n } \{ n \geq 0 \mid S _ { n } \in B \} , \quad \quad \bar { \tau } _ { B } ^ { \mu } ( s ) : = \mathbb { E } _ { s } ^ { \mu } [ \tau _ { B } ] .
$$

We define $\phi _ { \mu } ( s , B )$ as the expected number of times, before hitting $B ,$ that the state lies in $\tau$ or $\mu$ applies an exit action of Bather’s decomposition:

$$
\phi _ { \mu } ( s , B ) : = \mathbb { E } _ { s } ^ { \mu } \Big [ \sum _ { n = 0 } ^ { \tau _ { B } - 1 } \mathbb { 1 } \big \lbrace S _ { n } \in \mathcal { T } \mathrm { ~ o r ~ } \mu ( S _ { n } ) \in \mathcal { E } ( S _ { n } ) \big \rbrace \Big ] .\tag{4.32}
$$

Theorem 4.2 (Eventual performance bounds for Algorithm 2). In the setting of Theorem $4 . 1 ,$ almost surely, for all suficiently large t, the induced policy $\mu : = \mu _ { t }$ satisfies $g _ { \mu } ( s ) = g ^ { * } ( s )$ for $s \in \bar { S } ^ { \mathrm { r } }$ while all states in ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ are transient under $\mu$ , with

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - \bar { \delta } \phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } ) , \qquad s \in S \backslash \tilde { S } ^ { \mathrm { r } } .
$$

Hence, for all suficiently small $\bar { \delta } , \mu _ { t }$ is eventually gain-optimal.

Before giving the proof, we make several remarks.

Remark 4.2 (Sharpness and estimability of the bounds). In general, $\phi _ { \mu } ( s , B )$ can be bounded as

$$
\phi _ { \mu } ^ { \ast } ( s , B ) : = \mathbb { E } _ { s } ^ { \mu } \Big [ \sum _ { n = 0 } ^ { \tau _ { B } - 1 } \mathbb { 1 } \big \lbrace \mu ( S _ { n } ) \notin \mathcal { A } _ { \ast } ( S _ { n } ) \big \rbrace \Big ] \leq \phi _ { \mu } ( s , B ) \leq \bar { \tau } _ { B } ^ { \mu } ( s ) .\tag{4.33}
$$

As the proof will show, the term $\phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } )$ in Theorem 4.2 can be replaced by $\phi _ { \mu } ^ { * } ( s , \tilde { S } ^ { \mathrm { r } } )$ , yielding a sharper performance bound. However, $\phi _ { \mu } ^ { * } ( s , \tilde { S } ^ { \mathrm { r } } )$ depends on the unknown set $\mathcal { A } _ { * }$ , whereas $\phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } )$ is, in principle, estimable from sample trajectories under $\mu ,$ using Bather’s decomposition.

Remark 4.3 (Efect of algorithmic parameters). Theorem 4.2 shows how the algorithmic parameters afect the induced policies $\mu _ { t }$ for suficiently large t. Smaller $\bar { \delta } ,$ which controls the accuracy of the outer approximations of both $\mathcal { A } _ { * }$ and $\mathcal { C } ^ { \mathrm { b } }$ , yields a tighter gain bound. In contrast, the penalty ϵ used to stabilize Q-learning does not afect this worst-case asymptotic bound, since it is applied only to $\tilde { S } ^ { \mathrm { b } }$ , where remaining and exiting are nearly equally good in average reward (Lemma 4.2). It does afect policy behavior: the penalty ensures exit from the sub-MDPs in $\tilde { S } ^ { \mathrm { b } }$ , with larger ϵ encouraging earlier exit.

Remark 4.4 (Alternative policies and algorithmic variants).

(a) Besides the greedy policies studied above, near-greedy policies induced by Algorithm 2 can be analyzed similarly. We illustrate this with a case in which $\tilde { S } ^ { \mathrm { b } }$ need not be entirely transient under the induced policy. Let $\bar { \epsilon } , \epsilon , \delta$ satisfy $\epsilon \le \bar { \epsilon } < \underline { { \delta } }$ . Define a policy $\mu _ { t } ^ { \prime }$ similarly to $\mu _ { t } .$ , except that, for $s \not \in \tilde { S } _ { t } ^ { \mathrm { r } }$ , let $\mu _ { t } ^ { \prime } ( s )$ be ϵ¯-greedy w.r.t. $Q _ { t } \mathrm { { : } }$

$$
\mu _ { t } ^ { \prime } ( s ) \in \Bigl \{ a \in \tilde { \mathcal { A } } _ { t } ( s ) \ | \ Q _ { t } ( s , a ) > \operatorname* { m a x } _ { a \in \tilde { \mathcal { A } } _ { t } ( s ) } Q _ { t } ( s , a ) - \bar { \epsilon } \Bigr \} , \qquad s \in \mathcal { S } \setminus \tilde { \mathcal { S } } _ { t } ^ { \mathrm { r } } .\tag{4.34}
$$

Then, for suficiently large $t , \mu : = \mu _ { t } ^ { \prime }$ is gain-optimal on ${ \tilde { S } } ^ { \mathrm { r } }$ , as before. On ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ , the following additional properties hold under $\mu \colon \tilde { S } ^ { \mathrm { e } }$ is transient, while $\tilde { S } ^ { \mathrm { b } }$ may contain recurrent states. For the latter states,

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - ( \bar { \epsilon } - \epsilon ) , \quad s \in \mathcal { R } _ { \mu } \cap \tilde { \mathcal { S } } ^ { \mathrm { b } } ;\tag{4.35}
$$

and, for all transient states $s \in \mathcal { S } \setminus \tilde { \mathcal { S } } ^ { \mathrm { r } }$ 2

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - \bar { \epsilon } - \bar { \delta } \phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } \cup \mathcal { R } _ { \mu } ) .\tag{4.36}
$$

Derivations of these performance bounds are given at the end of Section 4.4.

(b) Consider settings in which solving the AOE is not a primary goal and, for $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { b } }$ , it is preferable to use the eventually optimal policies $\pi _ { \ell , k } ^ { ( t ) }$ for $\mathcal { M } _ { \ell , k }$ , so as to remainwithin the sub-MDP rather than exit it. Algorithm 2 can then be modified as follows. In the Q-learning step, treat $\tilde { \mathcal { C } } _ { t } ^ { \mathrm { b } } .$ -states as terminal states as well, with terminal rewards specified by $v _ { t }$ , rather than penalizing them. The update rule (4.24) can be modified accordingly and carried out only for selected stateaction pairs in $\{ ( s , a ) \in \tilde { \mathcal { A } } _ { t } \ | \ { s } \notin \tilde { \mathcal { S } } _ { t } ^ { \mathrm { r } } \cup \tilde { \mathcal { S } } _ { t } ^ { \mathrm { b } } \}$ . On $\mathcal { S } \setminus ( { \tilde { S } } ^ { \mathrm { r } } \cup { \bar { S } } ^ { \mathrm { b } } )$ , eventual performance bounds for the resulting greedy policies (omitted here) can be derived by similar arguments.

(c) Finally, as an alternative to classical Q-learning, the Q-learning step can be implemented using a policy-iteration-like Q-learning variant [6, 37], which avoids maximization over all actions at each iteration while retaining the same convergence guarantee.

## 4.4 Proof of Theorem 4.2 and supporting derivations

For a stationary policy $\mu ,$ let $P _ { \mu } ^ { * }$ denote the limiting matrix and $Z _ { \mu }$ the fundamental matrix. Recall that $P _ { \mu } ^ { * } = \operatorname* { l i m } _ { n \to \infty } P _ { \mu } ^ { n }$ in the aperiodic case and in general equals the Cesàro limit of $P _ { \mu } ^ { n }$ . By definition, $Z _ { \mu } = ( I - \overset { \sim } { P } _ { \mu } + \overset { \ast } { P } _ { \mu } ^ { \ast } ) ^ { - 1 }$ . In terms of these matrices, the gain and bias of $\mu$ are given, respectively, by

$$
g _ { \mu } = P _ { \mu } ^ { \ast } r _ { \mu } , \qquad h _ { \mu } = Z _ { \mu } ( r _ { \mu } - g _ { \mu } ) .
$$

See, e.g., [22, Secs. 8.2 and A.5] for these standard facts. We will use the following properties of $Z _ { \mu }$ (see, e.g., [7, 14]):

$$
Z _ { \mu } ( P _ { \mu } - I ) = P _ { \mu } ^ { * } - I ,\tag{4.37}
$$

$$
Z _ { \mu } = I + \operatorname * { l i m } _ { \beta \uparrow 1 } \sum _ { n = 1 } ^ { \infty } \beta ^ { n } ( P _ { \mu } ^ { n } - P _ { \mu } ^ { * } ) .\tag{4.38}
$$

The following lemma is a direct consequence of (4.38).

Lemma 4.3. For any $B \subset \tau _ { \mu }$ such that $B ^ { c }$ is closed under $\mu ,$ the submatrix $[ Z _ { \mu } ] _ { B , B }$ is nonnegative, and its row sums equal $\bar { \tau } _ { B ^ { c } } ^ { \mu } ( s )$ for $s \in B$ . Moreover, for $D \subset B$ and $s \in B$

$$
\big ( [ Z _ { \mu } ] _ { B , B } \mathbf { 1 } _ { D } \big ) ( s ) = \mathbb { E } _ { s } ^ { \mu } \Big [ \sum _ { n = 0 } ^ { \tau _ { B ^ { c } } - 1 } \mathbb { 1 } \{ S _ { n } \in D \} \Big ] \leq \bar { \tau } _ { B ^ { c } } ^ { \mu } ( s ) .
$$

$\mathrm { B y }$ the definition of ${ \mathcal { A } } _ { * } , ( P _ { \mu } g ^ { * } ) ( s ) = g ^ { * } ( s )$ if and only if $\mu ( s ) \in \mathcal { A } _ { * } ( s )$ . If s is recurrent under $\mu ,$ then $( P _ { \mu } g ^ { * } ) ( s ) = ( P _ { \mu } ^ { * } g ^ { * } ) ( s ) \stackrel { . } { = } g ^ { * } ( s )$ . The next lemma bounds $( P _ { \mu } ^ { * } g ^ { * } ) ( s ) - g ^ { * } ( s )$ for transient states under $\mu$ and will be used repeatedly in the performance-bounds analysis.

Lemma 4.4. Let µ be a policy in $\Pi _ { \mathcal { A } _ { \delta } }$ with $\delta > 0$ . Let $B \subset \tau _ { \mu }$ be such that $B ^ { c }$ is closed under $\mu$ and $( P _ { \mu } g ^ { * } ) ( s ) = g ^ { * } ( s )$ for all $s \in B ^ { c }$ . Then, $f o r \ s \in B$ ，

$$
( P _ { \mu } ^ { * } g ^ { * } ) ( s ) - g ^ { * } ( s ) \geq - \delta \mathbb { E } _ { s } ^ { \mu } \Big [ \sum _ { n = 0 } ^ { \tau _ { B ^ { c } } - 1 } \mathbb { 1 } \big \lbrace \mu ( S _ { n } ) \notin \mathcal { A } _ { * } ( S _ { n } ) \big \rbrace \Big ]\tag{4.39}
$$

$$
\begin{array} { r l } & { \geq - \delta \mathbb { E } _ { s } ^ { \mu } \bigg [ \displaystyle \sum _ { n = 0 } ^ { \tau _ { B } c - 1 } \mathbb { 1 } \big \lbrace S _ { n } \in { \mathcal { T } } \ o r \ \mu ( S _ { n } ) \in { \mathcal { E } } ( S _ { n } ) \big \rbrace \bigg ] , } \end{array}\tag{4.40}
$$

or equivalently, $( P _ { \mu } ^ { * } g ^ { * } ) ( s ) - g ^ { * } ( s ) \ge - \delta \phi _ { \mu } ^ { * } ( s , B ^ { c } ) \ge - \delta \phi _ { \mu } ( s , B ^ { c } )$

Proof. Let $E : = \{ s \in \mathcal { S } \mid \mu ( s ) \not \in \mathcal { A } _ { * } ( s ) \}$ and $E ^ { \prime } : = \{ s \in \mathcal { S } \mid s \in \mathcal { T }$ or $\mu ( s ) \in \mathcal { E } ( s ) \}$ . Since each s lies either in T or in some $C _ { \ell , k } \in \mathcal { C } _ { \ell , k }$ , and in the latter case $\mathscr { A } ( s ) = \mathscr { A } _ { \ell } ( s ) \cup \mathcal { E } ( s )$ with $\mathcal { A } _ { \ell } ( s ) \subset \mathcal { A } _ { * } ( s )$ we have $E \subset E ^ { \prime }$

By (4.37),

$$
P _ { \mu } ^ { * } g ^ { * } - g ^ { * } = Z _ { \mu } ( P _ { \mu } - I ) g ^ { * } .
$$

From the definition of $\mathcal { A } _ { \delta }$ and $E ,$

$$
( P _ { \mu } - I ) g ^ { * } \geq - \delta \mathbf { 1 } _ { E } .
$$

Moreover, $[ ( P _ { \mu } - I ) g ^ { * } ] ( s ) = 0$ for $s \in B ^ { c }$ by assumption. Since $[ Z _ { \mu } ] _ { B , B }$ is nonnegative, it follows that, for $s \in B$

$$
[ Z _ { \mu } ( P _ { \mu } - I ) g ^ { * } ] ( s ) \ge - \delta \left[ Z _ { \mu } { \bf 1 } _ { E \cap B } \right] ( s ) .
$$

Lemma 4.3 identifies $[ Z _ { \mu } \mathbf { 1 } _ { E \cap B } ] ( s )$ with the expected number of visits to $E \cap B$ before hitting $B ^ { c }$ giving (4.39). The inclusion $E \subset E ^ { \prime }$ then yields (4.40). □

We now prove Theorem 4.2. For brevi $\operatorname { \mathrm { [ y , } }$ we omit the qualifier $\mathrm { ^ { 6 6 } a . s . } ^ { \mathrm { 5 } }$ throughout.

Proof of Theorem $4 . 2 .$ Let t be suficiently large so that the sets constructed in Algorithm 2 remain unchanged and identify with the respective limiting sets given by Lemma 4.2.

(i) Eventual gain-optimality of $\mu _ { t }$ on ${ \tilde { S } } ^ { \mathrm { r } }$ .

The claim follows from the convergence of RVI Q-learning. In particular, for each $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { r } }$ RVI Q-learning converges in the communicating sub-MDP $\mathcal { M } _ { \ell , k }$ , producing policies $\pi _ { \ell , k } ^ { ( t ) }$ that are gain-optimal for the sub-MDP for all suficiently large $t ;$ see [33, 38] and the asynchronous stochastic approximation results in [39].

Since $\tilde { \mathcal { C } } ^ { \mathrm { r } } \subset \mathcal { C } ^ { \mathrm { r } }$ (Lemma 4.2), the defining property for $\mathcal { C } ^ { \mathrm { r } }$ (cf. (4.7)) implies that these policies $\pi _ { \ell , k } ^ { ( t ) }$ are also eventually gain-optimal in the original MDP M for states in $C _ { \ell , k }$ . As $\mu _ { t }$ eventually coincides with these policies on each $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { r } }$ , it follows that $\mu _ { t }$ is eventually gain-optimal on ${ \tilde { S } } ^ { \mathrm { r } }$

(ii) Bounding $g _ { \mu _ { t } }$ on ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$

We now bound the eventual performance of $\mu _ { t }$ for initial states in ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ . By Theorem 4.1, $Q _ { t }$ converges to $\hat { Q } ^ { * }$ on $\tilde { \Gamma } _ { \ O }$ , where $\hat { Q } ^ { * }$ is the optimal action-value function of an associated SSP-type total-reward MDP M<sup>ˆ</sup> defined on $\tilde { \Gamma }$ (Lemma 8.1). Therefore, for all suficiently large $t ,$ the policy $\mu _ { t }$ is greedy w.r.t. $\hat { Q } ^ { * }$ on $\tilde { \Gamma }$ and satisfies the optimality equation of $\hat { \mathcal { M } }$ . Hence, writing $\mu : = \mu _ { t } .$ , we have

$$
\bar { x } = r _ { \mu } - g ^ { * } + P _ { \mu } \bar { x } - \epsilon \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } .\tag{4.41}
$$

Left-multiplying (4.41) by $P _ { \mu } ^ { * }$ , and noting that $P _ { \mu } ^ { * } \bar { x } = P _ { \mu } ^ { * } P _ { \mu } \dot  $ x¯ and $g _ { \mu } = P _ { \mu } ^ { * } r _ { \mu }$ , we obtain

$$
g _ { \mu } = P _ { \mu } ^ { * } g ^ { * } + \epsilon P _ { \mu } ^ { * } \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } .\tag{4.42}
$$

Under $\mu , \tilde { S } ^ { \mathrm { r } }$ is closed, while ${ \mathcal { S } } \backslash { \tilde { S } } ^ { \mathrm { r } } \subset { \mathcal { T } } _ { \mu }$ . Indeed, since $\mu$ restricted to $| S \rangle \langle \tilde { S } ^ { \mathrm { r } }$ solves the optimality equation of the SSP-type MDP M<sup>ˆ</sup> , it is optimal and hence proper for $\hat { \mathcal { M } }$ . Thus, the entire efective state space ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ of $\hat { \mathcal { M } }$ is transient under $\mu .$ In particular, $\bar { \tilde { S } ^ { \mathrm { b } } } \bar { C } \mathcal { T } _ { \mu }$ and hence $P _ { \mu } ^ { * } \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } = 0$ , reducing (4.42) to

$$
g _ { \mu } = P _ { \mu } ^ { * } g ^ { * } .\tag{4.43}
$$

We now use Lemma 4.4 to bound $P _ { \mu } ^ { * } g ^ { * } - g ^ { * }$ on ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ . By (4.25) from Lemma 4.2, we have $\mu \in \Pi _ { \mathcal { A } _ { i } }$ and $( P _ { \mu } g ^ { * } ) ( s ) = g ^ { * } ( s )$ for $s \in \tilde { S } ^ { \mathrm { r } }$ . Applying Lemma 4.4 with $B = { \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ then yields the desired bound

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) = ( P _ { \mu } ^ { * } g ^ { * } ) ( s ) - g ^ { * } ( s ) \geq - \bar { \delta } \phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } ) , \qquad \forall s \in \mathcal { S } \setminus \tilde { S } ^ { \mathrm { r } } .
$$

Since $\Pi _ { \cal A }$ and $s$ are finite, the non-zero gain gaps are bounded away from zero. Moreover, $\phi _ { \mu } ( s , \tilde { S } ^ { \mathrm { r } } ) \leq \bar { \tau } _ { \tilde { S } ^ { \mathrm { r } } } ( s )$ , and the latter is uniformly bounded over all possible $s , \mu ,$ and ${ \tilde { S } } ^ { \mathrm { r } }$ . Hence, if the parameter $\bar { \delta }$ is below a deterministic M-dependent threshold, the above bound implies $g _ { \mu } = g ^ { * }$ , so $\mu$ is gain-optimal. □

Derivations of (4.35)–(4.36) for the near-greedy policies in Remark ${ \bf 4 . 4 ( a ) }$

Consider now the near-greedy policies $\mu _ { t } ^ { \prime }$ in Remark $4 . 4 ( \mathrm { a } )$ . Their definition together with the convergence of Algorithm 2 (Theorem 4.2) implies that, for suficiently large t, the policy $\mu : = \mu _ { t } ^ { \prime }$ satisfies

$$
r _ { \mu } - g ^ { * } + P _ { \mu } \bar { x } - \epsilon \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } \geq \bar { x } - \bar { \epsilon } \mathbf { 1 } _ { S \backslash \tilde { S } ^ { \mathrm { r } } } ,\tag{4.44}
$$

where $\bar { x }$ is the solution of the perturbed AOE in Theorem 4.2. As in the preceding proof, leftmultiplying (4.44) by $P _ { \mu } ^ { * }$ and rearranging terms yields

$$
g _ { \mu } \geq P _ { \mu } ^ { * } g ^ { * } + \epsilon P _ { \mu } ^ { * } \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } - \bar { \epsilon } P _ { \mu } ^ { * } \mathbf { 1 } _ { S \backslash \tilde { S } ^ { \mathrm { r } } } .\tag{4.45}
$$

Suppose that $\mu$ induces a recurrent class $E$ outside ${ \tilde { S } } ^ { \mathrm { r } }$ . Then $g ^ { * }$ is constant on $E$ and hence $( P _ { \mu } ^ { * } g ^ { * } ) ( s ) = g ^ { * } ( s )$ for all $s \in E$ . Moreover, by the properties of Bather’s decomposition, $E \subset C _ { \ell , k }$ for some $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { e } } \cup \tilde { \mathcal { C } } ^ { \mathrm { b } }$ . It then follows from (4.45) that, for recurrent states $s \in \mathcal { S } \setminus \tilde { \mathcal { S } } ^ { \mathrm { r } }$

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq { \left\{ \begin{array} { l l } { - { \bar { \epsilon } } } & { { \mathrm { i f ~ } } s \in { \mathcal { R } } _ { \mu } \cap { \tilde { \mathcal { S } } } ^ { \mathrm { e } } ; } \\ { - ( { \bar { \epsilon } } - \epsilon ) } & { { \mathrm { i f ~ } } s \in { \mathcal { R } } _ { \mu } \cap { \tilde { \mathcal { S } } } ^ { \mathrm { b } } . } \end{array} \right. }\tag{4.46}
$$

The second bound establishes (4.35).

The first bound in (4.46) implies that $\mathcal { R } _ { \mu } \cap \tilde { S } ^ { \mathrm { e } } = \emptyset$ . Indeed, for $s \in \mathcal { R } _ { \mu } \cap \tilde { S } ^ { \mathrm { e } }$ , that bound gives $g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - { \bar { \epsilon } } .$ , whereas (4.27) in Lemma 4.2, applied to $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { e } }$ , gives $g _ { \mu } ( s ) \leq g ^ { * } ( s ) - \underline { { \delta } }$ contradicting $\bar { \epsilon } < \underline { { \delta } } .$

Finally, for transient states $s \in \mathcal { S } \setminus \tilde { \mathcal { S } } ^ { \mathrm { r } }$ , note that $\tilde { S } ^ { \mathrm { r } } \cup \mathcal { R } _ { \mu }$ is closed and $( P _ { \mu } g ^ { * } ) ( s ) = g ^ { * } ( s )$ on this set. We can therefore apply Lemma 4.4 with $B = \mathcal { S } \setminus ( \tilde { S } ^ { \mathrm { r } } \cup \mathcal { R } _ { \mu } ) = \mathcal { T } _ { \mu } \setminus \tilde { S } ^ { \mathrm { r } }$ to obtain

$$
\bigl ( P _ { \mu } ^ { * } g ^ { * } - g ^ { * } \bigr ) ( s ) \geq - \bar { \delta } \phi _ { \mu } ( s , \tilde { \mathcal { S } } ^ { \mathrm { r } } \cup \mathcal { R } _ { \mu } ) , \qquad s \in \mathcal { T } _ { \mu } \setminus \tilde { \mathcal { S } } ^ { \mathrm { r } } .
$$

Combining this bound with (4.45) establishes (4.36).

## 5 Toward bias-optimal policies via approximation

In this section, we consider the computation of (near) bias-optimal policies, building on the base algorithm. Whereas gain optimality concerns long-run average reward, bias provides a stronger criterion for comparing gain-optimal policies by accounting for their transient reward behavior. When multiple policies attain the same optimal gain, bias optimality can therefore provide a further criterion for distinguishing among them.

Let $\mu$ be a stationary policy. By standard MDP theory [22, Sec. 8.2], its bias $h _ { \mu } = Z _ { \mu } ( r _ { \mu } - g _ { \mu } )$ ， or equivalently,

$$
h _ { \mu } ( s ) = { \underset { n  \infty } { C \mathrm { - } \mathrm { l i m } } } \mathbb { E } _ { s } ^ { \mu } { \Big [ } \sum _ { t = 0 } ^ { n } { \big ( } r _ { \mu } ( s _ { t } ) - g _ { \mu } ( s _ { t } ) { \big ) } { \Big ] } , \qquad s \in \mathcal { S } ,
$$

where C-lim denotes the Cesàro limit and can be replaced by the ordinary limit in the aperiodic case. The terms $\frac { g _ { \mu } } { 1 - \beta }$ and $h _ { \mu }$ are the first two terms in the Laurent series expansion of the $\beta -$ discounted value function $v _ { \beta , \mu }$ of $\mu .$ We call $\mu$ bias-optimal if it is gain-optimal and $h _ { \mu } \geq h _ { \pi }$ for all stationary gain-optimal policies $\pi .$ . Such a policy $\mu$ exists in $\Pi _ { \mathcal { A } }$ . The optimal bias function $h ^ { \ast } : = \operatorname* { s u p } _ { \mu : g _ { \mu } = g ^ { \ast } } h _ { \mu }$ is a solution of the AOE (4.1).

The bias optimality criterion is closely related to two other equivalent criteria: the 0-discount optimality criterion and the average-overtaking optimality criterion. Any bias-optimal policy is also optimal w.r.t. these criteria, meaning that, for all policies π and $s \in S$

$$
\operatorname* { l i m i n f } _ { \beta \uparrow 1 } \big ( v _ { \beta , \mu } ( s ) - v _ { \beta , \pi } ( s ) \big ) \geq 0 , \qquad \operatorname* { l i m i n f } _ { n \to \infty } \frac { 1 } { n } \sum _ { k = 1 } ^ { n } ( v _ { k , \mu } ( s ) - v _ { k , \pi } ( s ) ) \geq 0 ,
$$

where $v _ { k , \pi }$ denotes the k-stage value function of $\pi .$ See [22, Chap. 10] for details.

MDP theory [22, Sec. 10.2] tells us that, given $g ^ { * }$ , a bias-optimal policy can be found by solving the following pair of nested equations, which uniquely determines the optimal bias function $h ^ { * }$ and involves an associated vector $\bar { \boldsymbol { y } } \in \mathbb { R } ^ { | \boldsymbol { S } | }$ :

$$
h ^ { * } = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } } } \{ r _ { \mu } - g ^ { * } + P _ { \mu } h ^ { * } \} , \qquad y = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } [ h ^ { * } ] } } \{ - h ^ { * } + P _ { \mu } y \} .\tag{5.1}
$$

Here, $\mathcal { A } _ { * } [ h ^ { * } ] = \left\{ \mathcal { A } _ { * } [ h ^ { * } ] ( s ) \right\} _ { s \in \mathcal { S } } ,$ where $\mathcal { A } _ { \ast } [ h ^ { \ast } ] ( s )$ consists of the actions in $A _ { * } ( s )$ that attain the maximum in the first equation for state s. Any policy in $\Pi _ { \mathcal { A } _ { \ast } [ h ^ { \ast } ] }$ attaining the maximum in the second equation is bias-optimal. Rather than tackling these nested equations directly, we shall again take an approximation approach inspired by Federgruen and Schweitzer [14]. The main idea is, roughly speaking, as follows.

We seek an approximation $\tilde { h }$ of the optimal bias function $h ^ { * }$ by leveraging the convergent estimates of $g ^ { * }$ and outer approximations of $\mathcal { A } _ { \ast }$ that can be obtained from the outputs of Algorithm 1. One option, pursued below, is to do this indirectly: we approximate $h ^ { * }$ by a function $\tilde { h } \approx \tilde { h } ^ { * }$ where $\tilde { h } ^ { * }$ is the optimal bias function of another MDP $\widetilde { \mathcal { M } } = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ with $\tilde { \cal A }$ an outer approximation of $\mathcal { A } _ { * }$ constructed as in Algorithm 2. We then apply the base algorithm to solve an average-reward MDP $\mathcal { M } ^ { \prime }$ whose one-stage reward is −h<sup>˜</sup> and whose action constraints approximate $A _ { * } [ h ^ { * } ]$ . When these approximations are suficiently accurate, we expect the gain-optimal policies of $\mathcal { M } ^ { \prime }$ to be near bias-optimal in the original MDP.

A crucial step in the approach just outlined is to approximate the optimal bias function $\tilde { h } ^ { * }$ of ${ \check { \mathcal { M } } }$ . In this section, we consider a straightforward way to do so: discounted approximation with large discount factors, applied to the gain-shifted MDP $\widetilde { \mathcal { M } }$ . (Alternative approaches are possible but appear more challenging from both computational and analytical perspectives.) The next lemma gives the properties of the discounted approximations used here and largely follows from the results of Blackwell [7].

Algorithm 3 Approximate Method toward Bias-Optimal Policies (Conceptual Description)   
Information flow: Algorithm 3 (Phase $1 )  \mathrm { A }$ lgorithm 1;   
Algorithm 3 (Phase 2) ← {Algorithm 1, Algorithm 3 (Phase 1)}.   
Initialize Phase-1 starting time t<sup>¯</sup> and discount factor $\beta \approx 1$   
Execute Phases 1 and 2 concurrently with the above dependency structure:   
1: Phase 1: Action selection and β-discounted Q-learning (rewards shifted by $- g _ { t } )$   
2: Initialize $Q _ { \beta , \bar { t } } ,$ parameters $\bar { \delta } _ { 1 } > \underline { { { \delta } } } _ { 1 } > 0 \mathrm { . }$ and action constraint set $\tilde { \mathcal { A } } _ { \bar { t } }$   
3: for $t \geq \bar { t }$ do   
4: Sample a random subset $Y _ { t } \subset \tilde { \mathcal { A } } _ { t }$   
5: Apply Q-learning update (5.4) to $Q _ { \beta , t }$ on $Y _ { t }$   
6: t ← t + 1 and update $\tilde { \mathcal { A } } _ { t }$ via Rules (4.18)–(4.20) with two thresholds $( \underline { { \delta } } _ { 1 } , \bar { \delta } _ { 1 } )$   
7: end For end Phase   
8: Phase 2: Average-reward optimization (reward $- v _ { \beta , t } ,$ action sets induced by $Q _ { \beta , t } )$   
9: Initialize starting time $\bar { t } ^ { \prime } ,$ , parameters $\bar { \delta } _ { 2 } > \underline { { { \delta } } } _ { 2 } > 0 .$ , and action constraint set $\tilde { B } _ { \bar { t } ^ { \prime } }$   
10: for $t \geq \bar { t } ^ { \prime }$ do   
11: Update Bather’s decomposition for $\tilde { B } _ { t }$ if $\tilde { B } _ { t } \neq \tilde { B } _ { t - 1 }$   
12: Run one iteration of Phases 2–4 of the base algorithm   
13: with rewards $\{ - v _ { \beta , t } ( s ) \} _ { s \in \mathcal { S } }$ and action sets $\{ \tilde { B } _ { t } ( s ) \} _ { s \in \mathcal { S } }$   
14: $\mu _ { t }  \pi ^ { ( t ) }$ (Phase 4 policy)   
15: t ← t + 1 and update $\tilde { B } _ { t }$ via Rule (5.5) with two thresholds $( \underline { { \delta } } _ { 2 } , \bar { \delta } _ { 2 } )$   
16: end For end Phase

Lemma 5.1. For any action constraint set $\tilde { \cal A }$ with $\mathcal { A } _ { \ast } \subset \tilde { \mathcal { A } } \subset \mathcal { A } ,$ , consider the MDP ${ \widetilde { \mathcal { M } } } =$ $( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ . Then the optimal gain in $\widetilde { \mathcal { M } }$ is 0, and its optimal bias function $\tilde { h } ^ { * }$ satisfies $\tilde { h } ^ { * } \geq h ^ { * }$ with $\tilde { h } ^ { * } = h ^ { * } \ i f \tilde { \mathcal { A } } = \mathcal { A } _ { * }$ . Moreover, for $\beta \in [ 0 , 1 )$ , let $v _ { \beta }$ denote the β-discounted optimal value function of $\widetilde { \mathcal { M } }$ . Then

$$
v _ { \beta }  \tilde { h } ^ { * } \quad a s \ \beta \uparrow 1 .\tag{5.2}
$$

Consequently, $\{ v _ { \beta } \} _ { \beta \in [ 0 , 1 ) }$ is uniformly bounded (indeed, uniformly over admissible A<sup>˜</sup> as well), and

$$
\eta _ { \beta } : = \| v _ { \beta } - \tilde { h } ^ { * } \| _ { \infty } \to 0 , \qquad \epsilon _ { \beta } : = ( 1 - \beta ) \| v _ { \beta } \| _ { \infty } \to 0 , \quad a s \beta \uparrow 1 .\tag{5.3}
$$

Remark 5.1. An important feature of the discounted approximation above is the uniform boundedness of $\{ v _ { \beta } \} _ { \beta \in [ 0 , 1 ) }$ , which stems from shifting the rewards by the exact optimal gain function $g ^ { * }$ . The ability to obtain $g ^ { * }$ in the limit, rather than merely an approximation of it, is therefore an important advantage of the base algorithm in the present approach. This boundedness is also favorable numerically, as it prevents the value estimates from growing in magnitude as $\beta \uparrow 1$ in contrast to discounted approximation applied directly to the original MDP, whose discounted optimal value functions usually diverge in magnitude.

## 5.1 Algorithm

Algorithm 3, whose pseudocode and details are presented in what follows, implements the two steps of the approach outlined above in two concurrent phases. It initializes and updates the action constraint sets $\tilde { \mathcal { A } } _ { t }$ according to the same set-construction rules (4.18)–(4.20) as in Algorithm 2. These sets eventually stabilize to a sample-path-dependent outer approximation $\tilde { \cal A }$ of $A _ { * } .$ The resulting A<sup>˜</sup> directly defines the action constraints of $\widetilde { \mathcal { M } }$ and indirectly those of $\mathcal { M } ^ { \prime } ,$ , so both MDPs are sample-path dependent.

(i) Phase 1 approximates the optimal bias function $h ^ { * }$ via discounted Q-learning with discount factor $\beta$ on selected state-action pairs in $\tilde { \mathcal { A } } _ { t }$ , using g from Algorithm 1 to shift the rewards. The resulting $\mathrm { Q } \mathrm { - }$ -function estimates are denoted by $Q _ { \beta , t }$

(ii) Phase 2 constructs action constraint sets $\tilde { B } _ { t } \subset \tilde { \mathcal { A } } _ { t }$ from near-greedy actions w.r.t. $Q _ { \beta , t }$ , and applies the base algorithm to the resulting approximation of $\mathcal { M } ^ { \prime }$

For each phase, the starting time, parameters, and stepsizes (if any) can be chosen based on past information while respecting the information-flow structure.

We now give the update details. Denote

$$
v _ { \beta , t } ( s ) : = \operatorname* { m a x } _ { a \in \tilde { \mathcal { A } } _ { t } ( s ) } Q _ { \beta , t } ( s , a ) , \qquad s \in \mathcal { S } .
$$

In Phase 1, Algorithm 3 samples a subset $Y _ { t } \subset \tilde { \mathcal { A } } _ { t }$ and updates $Q _ { \beta , t }$ on $Y _ { t }$ as follows:

1. For each $( s , a ) \in Y _ { t }$ , sample a transition to obtain a reward $R _ { t + } ^ { s a } .$ and next state $S _ { t + 1 } ^ { s a }$

2. Update $Q _ { \beta , t }$ for each $( s , a ) \in Y _ { t }$ by

$$
Q _ { \beta , t + 1 } ( s , a ) = Q _ { \beta , t } ( s , a ) + \alpha _ { t } ( s , a ) \big ( R _ { t + 1 } ^ { s a } - g _ { t } ( s ) + \beta v _ { \beta , t } ( S _ { t + 1 } ^ { s a } ) - Q _ { \beta , t } ( s , a ) \big ) ,\tag{5.4}
$$

where $\alpha _ { t } ( s , a ) \geq 0$ is a possibly random stepsize that can depend on information available at time $t ,$ including Y (but not on the transition data indexed by t + 1).

In Phase 2, define $\tilde { B } _ { t } \subset \tilde { A } _ { t } .$ , with $\tilde { B } _ { \bar { t } ^ { \prime } - 1 } ( s ) = \emptyset$ initially, by

$$
\begin{array} { r l } & { \tilde { \mathcal { B } } _ { t } ( s ) : = \bigr \{ a \in \tilde { \mathcal { A } } _ { t } ( s ) \setminus \tilde { \mathcal { B } } _ { t - 1 } ( s ) \mid Q _ { \beta , t } ( s , a ) > v _ { \beta , t } ( s ) - \underline { { \delta } } _ { 2 } \bigr \} } \\ & { \qquad \cup \ \bigl \{ a \in \tilde { \mathcal { A } } _ { t } ( s ) \cap \tilde { \mathcal { B } } _ { t - 1 } ( s ) \mid Q _ { \beta , t } ( s , a ) > v _ { \beta , t } ( s ) - \bar { \delta } _ { 2 } \bigr \} , \quad \forall s \in \mathcal { S } . } \end{array}\tag{5.5}
$$

Phase 2 then runs Phases 2–4 of the base algorithm with one-stage reward ${ - v _ { \beta , t } }$ and action constraint set $\tilde { B } _ { t }$ , as specified in the pseudocode. As in the construction of $\tilde { \mathcal { A } } _ { t } .$ two thresholds $( \underline { { \delta } } _ { 2 } , \bar { \delta } _ { 2 } )$ are used in (5.5) for adding new elements and retaining existing ones; finite-time stabilization of $\tilde { B } _ { t }$ is established in Lemma 5.2 below.

## 5.2 Convergence and performance results

Let Algorithm 3 be run under the following conditions, which are assumed throughout this subsection:

(D1) The base algorithm satisfies its required convergence conditions for each MDP instance to which it is applied and each initialization of its internal parameters (including its application within Algorithm 1 to the original MDP).

(D2) The information-flow structure specified in Algorithm 3 is respected.

(D3) The Q-learning updates in Phase 1 satisfy the standard sampling and stepsize conditions: specifically, condition (C3) holds with $Y _ { t }$ and $\tilde { \mathcal { A } } _ { t }$ here in place of $Y _ { t }$ and $\tilde { \Gamma } _ { t }$ , respectively, in Algorithm 2.

The next lemma establishes the desired finite-time stablization of the action sets that define the limiting MDPs $\widetilde { \mathcal { M } }$ and $\mathcal { M } ^ { \prime }$ . The subsequent theorem then establishes that the policies produced in Phase 2 are eventually gain-optimal for $\mathcal { M } ^ { \prime }$ . The proofs of these results are given in Section 8.

Lemma 5.2 (Finite-time stabilization of action sets and asymptotic convergence of Phase 1). Almost surely, $\tilde { \mathcal { A } } _ { t }$ and $\tilde { B } _ { t }$ stabilize after a finite time, yielding (random) limiting sets $\tilde { \cal A }$ and $\tilde { B } \subset \tilde { \mathcal { A } }$ such that for all $s \in S$

$$
\begin{array} { r l r } { \mathcal { A } _ { \ast } ( s ) \subset \tilde { \mathcal { A } } ( s ) \subset \mathcal { A } _ { \bar { \delta } _ { 1 } } ( s ) , } & { { } \quad \mathcal { B } _ { \ast } ( s ) \subset \tilde { \mathcal { B } } ( s ) \subset \mathcal { B } _ { \bar { \delta } _ { 2 } } ( s ) , } \end{array}\tag{5.6}
$$

where the sets $\boldsymbol { B } _ { * } ( s )$ and $B _ { \delta } ( s ) , \delta > 0$ , are defined as follows. For the sample-path-dependent MDP $\widetilde { \mathcal { M } } : = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ , let v (resp. $Q _ { \beta } )$ denote its β-discounted optimal value (resp. action-value) function. Then

$$
\mathcal { B } _ { \ast } ( s ) : = \{ a \in \tilde { \mathcal { A } } ( s ) | Q _ { \beta } ( s , a ) = v _ { \beta } ( s ) \} , \qquad \mathcal { B } _ { \delta } ( s ) : = \{ a \in \tilde { \mathcal { A } } ( s ) | Q _ { \beta } ( s , a ) \geq v _ { \beta } ( s ) - \delta \} .
$$

Moreover, on ${ \tilde { \cal A } } ,$ the sequence $\{ Q _ { \beta , t } \}$ produced in Phase 1 converges a.s. to $Q _ { \beta }$

Note that, for each state $s , B _ { * } ( s )$ consists of the greedy actions w.r.t. the β-discounted optimal action-value function $Q _ { \beta }$ for ${ \widetilde { \mathcal { M } } } .$ , while $\tilde { B } ( s )$ outer-approximates $\boldsymbol { B } _ { * } ( s )$ by near-greedy actions, with ${ \bar { \delta } _ { 2 } }$ controlling the approximation tolerance. Thus, $\Pi _ { B _ { * } }$ is the class of (nonrandomzied stationary) $\beta \mathrm { . }$ -discounted optimal policies for ${ \widetilde { \mathcal { M } } } _ { : }$ , whereas $\Pi _ { \tilde { B } }$ is the corresponding enlarged class induced by near-greedy actions.

Theorem 5.1. Almost surely, for all suficiently large t, the policy $\mu _ { t }$ generated by Algorithm 3 (Phase 2) is gain-optimal for the sample-path-dependent MDP $\mathcal { M } ^ { \prime } : = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ , where $\tilde { B }$ and $v _ { \beta }$ are as given in Lemma 5.2.

Next, we bound the performance of the policies generated in Algorithm 3 for suficiently large t. For the sample-path-dependent MDP $\widetilde { \mathcal { M } } = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ in Lemma 5.2, recall the quantities introduced in Lemma 5.1:

$$
\eta _ { \beta } = \| v _ { \beta } - \tilde { h } ^ { * } \| _ { \infty } , \qquad \epsilon _ { \beta } = ( 1 - \beta ) \| v _ { \beta } \| _ { \infty } .
$$

By that lemma, both vanish as $\beta \uparrow 1$ . These quantities capture the errors associated with the discounted approximation and enter the performance bounds below.

The next theorem quantifies how these approximations translate into gain-optimality guarantees in the original MDP M for policies admitted by the limiting action constraints ${ \tilde { B } } ,$ with stronger bias-optimality guarantees for gain-optimal policies of $\mathcal { M } ^ { \prime }$

Theorem 5.2 (Eventual performance bounds). Let $\tilde { \mathcal { A } } , \tilde { \mathcal { B } } , v _ { \beta }$ , and $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ be as above.

(i) Every $\mu \in \Pi _ { \tilde { B } }$ satisfies

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - \epsilon _ { \beta } - \bar { \delta } _ { 2 } , \qquad s \in \mathcal { R } _ { \mu } ,\tag{5.7}
$$

$$
g _ { \mu } ( s ) - g ^ { * } ( s ) \geq - \epsilon _ { \beta } - \bar { \delta } _ { 2 } - \bar { \delta } _ { 1 } \phi _ { \mu } ( s , \mathcal { R } _ { \mu } ) , \qquad s \in \mathcal { T } _ { \mu } .\tag{5.8}
$$

Consequently, every $\mu \in \Pi _ { \tilde { B } }$ is gain-optimal for M $i f \ { \bar { \delta } } _ { 1 } , { \bar { \delta } } _ { 2 }$ , and $1 - \beta$ are suficiently small.

Let $\bar { \mu } \in \Pi _ { \tilde { \cal B } }$ be any gain-optimal policy for M<sup>′</sup>. Then:

(ii) The policy µ¯ is bias-optimal within $\Pi _ { \tilde { B } }$ for a perturbed MDP $\mathcal { M } _ { \mathrm { p e r t } }$ whose reward function difers from that of M by the shift $- g ^ { * }$ and a perturbation uniformly bounded by $\epsilon _ { \beta } + \bar { \delta } _ { 2 }$

(iii) $I f \bar { \mu }$ is gain-optimal for M, then it is ϵ<sup>′</sup>-optimal for M w.r.t. bias-optimality, where

$$
\epsilon ^ { \prime } = \eta _ { \beta } + c ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } )
$$

for some M-dependent constant $c \geq 1$ . In particular, under the gain-optimality condition above, µ¯ is bias-optimal for M whenever $\epsilon ^ { \prime }$ falls below an $\mathcal { M }$ -dependent threshold, which is ensured by suficiently small $1 - \beta$ and ${ \bar { \delta } } _ { 2 }$

Consequently, combining Lemma 5.2 and Theorem 5.1, almost surely, for all suficiently large t, the following hold: $\Pi _ { \tilde { B } _ { t } } = \Pi _ { \tilde { B } } ;$ every policy in $\Pi _ { \tilde { B _ { t } } }$ satisfies $( i ) ;$ and the policy µ generated in Phase 2 of Algorithm 3 additionally satisfies $( i i )$ and $( i i i )$ . In particular, for suficiently small $\bar { \delta } _ { 1 } , \bar { \delta } _ { 2 } .$ , and $1 - \beta , \mu _ { t }$ is gain-optimal and $\epsilon ^ { \prime } .$ -optimal w.r.t. bias-optimality for $\mathcal { M } ,$ ; it is bias optimal whenever $\epsilon ^ { \prime }$ is below the threshold in (iii).

The proof of this theorem is given in Section 5.3.

The bias-optimality bound in part (iii) is a consequence of stronger statewise bounds developed below (see (5.10)). These bounds also characterize the performance of $\bar { \mu }$ when it is not gain-optimal from every initial state and will be useful in interpreting the numerical results.

Another subtlety is that, with suficiently accurate approximations, Phase 1 alone is suficient to produce near bias-optimal policies in the limit. This raises the question of what additional advantages Phase 2 can ofer. We investigate this question below.

Statewise bias bounds and the role of Phase 2

For any policy $\mu \in \Pi _ { \cal A }$ let $D _ { \mu } ^ { * } : = \{ s \in { \mathcal { S } } \mid g _ { \mu } ( s ) = g ^ { * } ( s ) \}$ , the set of initial states from which $\mu$ attains the optimal gain in the original MDP M, referred to below as the gain-optimal states for $\mu .$ Whenever $D _ { \mu } ^ { \ast }$ is nonempty, it is closed under µ. Otherwise, there would exist $s \in D _ { \mu } ^ { * }$ such that $g _ { \mu } ( s ) = ( P _ { \mu } g _ { \mu } ) ( s ) < ( P _ { \mu } g ^ { * } ) ( s ) \leq g ^ { * } ( s )$ , contradicting $g _ { \mu } ( s ) = g ^ { * } ( s )$

Define $\eta _ { \beta } ( s ) : = | v _ { \beta } ( s ) - { \tilde { h } } ^ { * } ( s ) |$ for $s \in S ;$ thus $\eta _ { \beta } = \operatorname* { m a x } _ { s \in \mathcal { S } } \eta _ { \beta } ( s )$ . For $\mu \in \Pi _ { \cal A }$ and $\hat { \Pi } \subset \Pi _ { A }$ define

$$
L _ { \mu } ( s ) : = ( \vert Z _ { \mu } \vert \mathbf 1 ) ( s ) \ge 1 , \qquad L _ { \hat { \Pi } } ( s ) : = \operatorname* { m a x } _ { \mu \in \hat { \Pi } } L _ { \mu } ( s ) .
$$

Proposition 5.1 (Bias bounds for gain-optimal states).

(i) For any policy $\mu \in \Pi _ { B _ { \ast } }$ and all $s \in D _ { \mu } ^ { * }$

$$
h _ { \mu } ( s ) \geq h ^ { \ast } ( s ) - \eta _ { \beta } ( s ) - L _ { \mu } ( s ) \epsilon _ { \beta } .\tag{5.9}
$$

(ii) For any policy $\bar { \mu } \in \Pi _ { \tilde { B } }$ that is gain-optimal for $\mathcal { M } ^ { \prime }$ and all $s \in D _ { \bar { \mu } } ^ { * }$

$$
h _ { \bar { \mu } } ( s ) \geq h ^ { * } ( s ) - \eta _ { \beta } ( s ) - L _ { \bar { \mu } } ( s ) ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } ) ,\tag{5.10}
$$

and also, with $\Pi _ { \tilde { \mathcal { B } } } ^ { s } : = \{ \mu \in \Pi _ { \tilde { \mathcal { B } } } : g _ { \mu } ( s ) = g ^ { * } ( s ) \}$

$$
h _ { \bar { \mu } } ( s ) \geq \operatorname* { m a x } _ { \mu \in \Pi _ { \tilde { s } } ^ { s } } h _ { \mu } ( s ) - 2 L _ { \Pi _ { \tilde { s } } ^ { s } } ( s ) ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } ) .\tag{5.11}
$$

The next proposition provides a necessary and suficient characterization of the condition $s \in D _ { \bar { \mu } } ^ { * }$ In particular, it makes explicit several necessary conditions that may not be immediately obvious.

Proposition 5.2. Let $\bar { \mu } \in \Pi _ { \tilde { B } }$ be a gain-optimal policy for $\mathcal { M } ^ { \prime }$ . Then $s \in D _ { \bar { \mu } } ^ { * }$ if and only if the following two conditions hold:

(i) $\bar { \mu } ( s ) \in \mathcal { A } _ { * } ( s )$ , and

$$
( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) = \operatorname* { i n f } _ { \mu \in \Pi _ { \tilde { B } } } ( P _ { \mu } ^ { * } v _ { \beta } ) ( s ) = 0 ,\tag{5.12}
$$

i.e., the optimal gain in M<sup>′</sup> starting from state s is 0.

(ii) ${ \mathit { I f s } } \in { \mathcal { R } } _ { \bar { \mu } }$ , then $\bar { \mu } ( s ) \in B _ { * } ( s )$ . Otherwise, $\bar { \mu } ( s ^ { \prime } ) \in B _ { * } ( s ^ { \prime } )$ for every recurrent state $s ^ { \prime } \in \mathcal { R } _ { \bar { \mu } }$ that is reachable from s under µ¯.

Furthermore, (5.12) implies the following necessary property:

(iii) Every policy $\mu \in \Pi _ { B _ { * } }$ is gain-optimal for state s in the MDP $\widetilde { \mathcal { M } } = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ , and therefore also in the original MDP whenever $\tilde { \mathcal { A } } ( s ) = \mathcal { A } _ { * } ( s )$

The optimal gain function in $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ is always nonnegative (see Lemma 5.3 below). A larger outer approximation $\tilde { B }$ of $\boldsymbol { B } _ { \ast }$ yields higher optimal gain values, which deviate further from the zero gain required by the necessary condition (5.12). The degree of departure from this condition translates directly into degradation in the gain of $\bar { \mu }$ for the original MDP M (cf. (5.24) below).

Thus, the preceding propositions appear to suggest that policies in $\Pi _ { B _ { * } }$ , being optimal for the β-discounted limiting MDP $\widetilde { \mathcal { M } }$ arising from Phase 1, may already sufice from the viewpoint of the preceding performance guarantees and can, in fact, have stronger theoretical guarantees than policies produced by Phase 2. In particular:

Corollary 5.1. For a state $s \in S _ { : }$ , suppose that $\tilde { \mathcal { A } } ( s ) = \mathcal { A } _ { * } ( s )$ and (5.12) holds. Then every policy $\mu \in \Pi _ { B _ { * } }$ is gain-optimal from s in $\mathcal { M } ^ { \prime }$ . Moreover, $\mu$ is also gain-optimal for s in the original $M D P$ so that its bias $h _ { \mu } ( s )$ can be bounded as in (5.9).

In the remainder of this subsection, we further compare the policies produced by the two phases and discuss what Phase 2 can ofer in practice, using an illustrative example.

Remark 5.2 (Practical role of Phase 2). Although all greedy policies w.r.t. $Q _ { \beta , t }$ eventually lie in $\Pi _ { B _ { * } }$ by the convergence of Phase 1 (Lemma 5.2), it could be numerically challenging to find a policy in $\Pi _ { B _ { 3 } }$ via Phase 1 alone in the situation described below. Such a situation can arise, for example, in an important class of total-reward MDPs known as positive bounded models [22, Sec. 7.2].

Consider the pair of nested equations in (5.1) for finding a bias-optimal policy:

$$
h ^ { * } = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } } } \{ r _ { \mu } - g ^ { * } + P _ { \mu } h ^ { * } \} , \qquad y = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } [ h ^ { * } ] } } \{ - h ^ { * } + P _ { \mu } y \} .
$$

Suppose that the first equation alone cannot identify a bias-optimal policy: $\Pi _ { \mathcal { A } _ { * } [ h ^ { * } ] }$ consists of multiple gain-optimal policies, not all of which are bias-optimal. For simplicity, suppose that $\tilde { \mathcal { A } } = \mathcal { A } _ { \ast }$ , as desired in Algorithm 3. Then, as we take $\beta$ close to 1 so that $v _ { \beta }$ approximates $h ^ { * }$ increasingly accurately, all policies $\mu$ in $\Pi _ { \mathcal { A } _ { \ast } [ h ^ { \ast } ] }$ , including those that are not bias-optimal, approximately attain the statewise maxima in the $\beta -$ discounted optimality equation for $\widetilde { \mathcal { M } }$

$$
v _ { \beta } \approx r _ { \mu } - g ^ { * } + \beta P _ { \mu } v _ { \beta } = Q _ { \beta } \bigl ( \cdot , \mu ( \cdot ) \bigr ) ,
$$

with the corresponding errors vanishing as $\beta \uparrow 1$

Consequently, when only the stochastic iterates $Q _ { \beta , t }$ computed in Phase 1 are used, distinguishing non-bias-optimal policies from bias-optimal ones based on greediness w.r.t. $Q _ { \beta , t }$ may require $Q _ { \beta , t }$ to approximate its limit $Q _ { \beta }$ extremely accurately. As $\beta \uparrow 1$ , the number of iterations needed to attain such accuracy may become prohibitively large. Moreover, the required accuracy may itself be dificult to achieve reliably in practice.

These observations suggest that, although one could in theory rely solely on the discounted approximation in Phase 1 to find (near-)bias-optimal policies for M by taking $\beta$ close to 1, Phase 2 may provide a less numerically demanding and more practical approach for computing such policies.

As noted earlier, positive bounded models provide examples in which the numerical challenges of using Phase 1 alone can arise. To illustrate this, we run Algorithm 3 on an example adapted from Puterman [22, Ex. 7.2.3], with zero-mean random noise added to the rewards. The results in Figure 2 support the observations above. ⋄

Remark 5.3. Finally, since the optimal gain in $\mathcal { M } ^ { \prime }$ is always nonnegative, Prop. 5.2 suggests the possibility of using the optimal gain estimates for M<sup>′</sup> produced by the base algorithm in Phase 2 of Algorithm 3 to adaptively tune the parameters $\bar { \delta } _ { 1 } , \bar { \delta } _ { 2 } .$ , and $\beta .$ The development and convergence analysis of such an adaptive algorithm are beyond the scope of the present work and remain a direction for future research.

## 5.3 Performance-bound analysis

## 5.3.1 Proof of Theorem 5.2

(i) Bounding $g _ { \mu }$ for $\mu \in \Pi _ { \tilde { B } } \colon$

Let $\Delta _ { \mu }$ be the nonnegative vector defined through the equality

$$
v _ { \beta } = r _ { \mu } - g ^ { * } + \beta P _ { \mu } v _ { \beta } + \Delta _ { \mu } .\tag{5.13}
$$

![](images/9c6d988159bcae6dd4d1a89f2c6355ce70b08ad0efd5552b5d8f7df88321315d.jpg)  
Figure 2: Comparison of the Phase 1 greedy policies and Phase 2 policies of Algorithm 3. Curves show the fraction of runs producing a bias-optimal policy across five random seeds. (a) Performance after $1 0 ^ { 6 }$ iterations for diferent discount factors $\beta .$ (b) Performance during learning for $\beta = 0 . 9 9 9$

Since $\mu ( s ) \in \tilde { \mathcal { B } } ( s ) \subset B _ { \bar { \delta } _ { 2 } } ( s )$ for all $s \in S$ (Lemma 5.2), we have $\| \Delta _ { \mu } \| _ { \infty } \leq \bar { \delta } _ { 2 }$ . Left-multiplying (5.13) by $P _ { \mu } ^ { * }$ and rearranging the terms, we obtain

$$
P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) = ( 1 - \beta ) P _ { \mu } ^ { * } v _ { \beta } - P _ { \mu } ^ { * } \Delta _ { \mu } \geq - ( 1 - \beta ) \| v _ { \beta } \| _ { \infty } { \bf 1 } - \| \Delta _ { \mu } \| _ { \infty } { \bf 1 } \geq - ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } ) { \bf 1 } .\tag{5.14}
$$

Next, we bound the term $P _ { \mu } ^ { * } g ^ { * } - g ^ { * }$ . For recurrent states under $\mu ,$

$$
( P _ { \mu } ^ { \ast } g ^ { \ast } ) ( s ) = g ^ { \ast } ( s ) , \qquad \forall s \in \mathcal { R } _ { \mu } .\tag{5.15}
$$

For transient states under $\mu ,$ Lemma 4.4 applies with $B = \mathcal { T } _ { \mu } , B ^ { c } = \mathcal { R } _ { \mu }$ , and $\delta = \bar { \delta } _ { 1 }$ , since $\mu \in \Pi _ { \mathcal { A } _ { \bar { \delta } _ { 1 } } }$ by $\tilde { B } \subset \tilde { \mathcal { A } } \subset \mathcal { A } _ { \bar { \delta } _ { 1 } }$ (Lemma 5.2). Hence

$$
( P _ { \mu } ^ { \ast } g ^ { \ast } ) ( s ) - g ^ { \ast } ( s ) \geq - \bar { \delta } _ { 1 } \phi _ { \mu } ( s , \mathcal { R } _ { \mu } ) , \quad \forall s \in \mathcal { T } _ { \mu } .\tag{5.16}
$$

Since $g _ { \mu } - g ^ { * } = P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) + P _ { \mu } ^ { * } g ^ { * } - g ^ { * }$ , it follows from (5.14)–(5.16) that

$$
\begin{array} { c c } { { g _ { \mu } ( s ) - g ^ { \ast } ( s ) \geq - \epsilon _ { \beta } - \bar { \delta } _ { 2 } , } } & { { \forall s \in { \mathcal R } _ { \mu } , } } \\ { { } } & { { } } \\ { { g _ { \mu } ( s ) - g ^ { \ast } ( s ) \geq - \epsilon _ { \beta } - \bar { \delta } _ { 2 } - \bar { \delta } _ { 1 } \phi _ { \mu } ( s , { \mathcal R } _ { \mu } ) , } } & { { \forall s \in { \mathcal T } _ { \mu } . } } \end{array}
$$

Since $\Pi _ { \cal A }$ and $s$ are finite, these bounds imply that for suficiently small $\bar { \delta } _ { 1 } , \bar { \delta } _ { 2 }$ , and $1 - \beta , \mu$ is gain-optimal. This proves part (i) of Theorem 5.2.

## (ii) Bias-optimality of $\bar { \mu }$ in a perturbed MDP:

For $\mu \in \Pi _ { \tilde { B } }$ , let $\Delta _ { \beta , \mu } : = - P _ { \mu } ( 1 - \beta ) v _ { \beta } + \Delta _ { \mu }$ . Note that $\| \Delta _ { \beta , \mu } \| _ { \infty } \leq \epsilon _ { \beta } + \bar { \delta } _ { 2 }$ . Left-multiplying (5.13) separately by $P _ { \mu } ^ { * }$ and $Z _ { \mu }$ , and rearranging, we obtain

$$
P _ { \mu } ^ { * } ( r _ { \mu } + \Delta _ { \beta , \mu } - g ^ { * } ) = 0 ,\tag{5.17}
$$

$$
Z _ { \mu } ( r _ { \mu } + \Delta _ { \beta , \mu } - g ^ { * } ) = Z _ { \mu } ( I - P _ { \mu } ) v _ { \beta } = v _ { \beta } - P _ { \mu } ^ { * } v _ { \beta } ,\tag{5.18}
$$

where we used (4.37) in (5.18). Equations (5.17)– (5.18) show that, under the reward perturbation $\Delta _ { \beta , \mu }$ and $\operatorname { s h i f t } - g ^ { * }$ , the gain of $\mu$ is 0 and its bias is $v _ { \beta } - P _ { \mu } ^ { * } v _ { \beta }$

Since $v _ { \beta }$ is independent of $\mu$ and $- P _ { \mu } ^ { \ast } v _ { \beta }$ is the gain of $\mu$ in the MDP $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ , the gain-optimality of $\bar { \mu }$ for $\mathcal { M } ^ { \prime }$ implies

$$
Z _ { \bar { \mu } } ( r _ { \bar { \mu } } + \Delta _ { \beta , \bar { \mu } } - g ^ { * } ) \geq Z _ { \mu } ( r _ { \mu } + \Delta _ { \beta , \mu } - g ^ { * } ) , \qquad \forall \mu \in \Pi _ { \tilde { \cal B } } .\tag{5.19}
$$

This shows that $\bar { \mu }$ is bias-optimal within $\Pi _ { \tilde { B } }$ for a perturbed MDP whose rewards difer from those of the original MDP by the shift $- g ^ { * }$ and a perturbation uniformly bounded by $\epsilon _ { \beta } + \bar { \delta } _ { 2 }$

## (iii) Near bias-optimality of $\bar { \mu }$ in the original MDP:

Under the assumption in part (iii) that $\bar { \mu }$ is gain-optimal for M, $D _ { \bar { \mu } } ^ { * } = S$ . The assertion then follows directly from the statewise bias bound (5.10) in Prop. 5.1(ii) (established below), upon noting that ma $\mathrm { x } _ { s \in \mathcal { S } } \eta _ { \beta } ( s ) = \eta _ { \beta }$ and

$$
\operatorname* { m a x } _ { s \in { \mathcal { S } } } L _ { \bar { \mu } } ( s ) \leq \operatorname* { m a x } _ { \mu \in \Pi _ { \mathcal { A } } } \left\| | Z _ { \mu } | \mathbf { 1 } \right\| _ { \infty } = : c ,
$$

where c is a finite M-dependent constant since $\Pi _ { \mathcal { A } }$ is finite.

This completes the proof of Theorem 5.2.

## 5.3.2 Proofs of Propositions 5.1 and 5.2

We begin with several useful relations. For any $\mu \in \Pi _ { A }$ , recall that $g _ { \mu } = P _ { \mu } ^ { * } r _ { \mu }$ and satisfies $g _ { \mu } = P _ { \mu } g _ { \mu } = P _ { \mu } ^ { * } g _ { \mu }$ , which implies $P _ { \mu } ^ { * } ( r _ { \mu } - g _ { \mu } ) = 0$ and

$$
P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) = P _ { \mu } ^ { * } ( g _ { \mu } - g ^ { * } ) \leq 0 .\tag{5.20}
$$

Suppose $D _ { \mu } ^ { \ast }$ is nonempty. Then, since $D _ { \mu } ^ { \ast }$ is closed under $\mu ,$ for all $s \in D _ { \mu } ^ { * }$

$$
[ P _ { \mu } ^ { * } ( g _ { \mu } - g ^ { * } ) ] ( s ) = [ P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) ] ( s ) = 0 ,\tag{5.21}
$$

$$
[ Z _ { \mu } ( g _ { \mu } - g ^ { * } ) ] ( s ) = 0 ,\tag{5.22}
$$

$$
h _ { \mu } ( s ) : = [ Z _ { \mu } ( r _ { \mu } - g _ { \mu } ) ] ( s ) = [ Z _ { \mu } ( r _ { \mu } - g ^ { \ast } ) ] ( s ) ,\tag{5.23}
$$

where (5.22) follows from the expression $Z _ { \mu } = I + \mathrm { l i m } _ { \beta \uparrow 1 } \sum _ { n = 1 } ^ { \infty } \beta ^ { n } ( P _ { \mu } ^ { n } - P _ { \mu } ^ { * } )$ and implies (5.23).

For $\mu \in \Pi _ { \tilde { B } }$ , left-multiplying (5.13) by $P _ { \mu } ^ { * }$ and $Z _ { \mu } ,$ respectively, yields

$$
( 1 - \beta ) P _ { \mu } ^ { * } v _ { \beta } = P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) + P _ { \mu } ^ { * } \Delta _ { \mu } ,\tag{5.24}
$$

$$
Z _ { \mu } ( r _ { \mu } - g ^ { * } ) = ( I - P _ { \mu } ^ { * } ) v _ { \beta } + ( 1 - \beta ) Z _ { \mu } P _ { \mu } v _ { \beta } - Z _ { \mu } \Delta _ { \mu } ,\tag{5.25}
$$

where we used $Z _ { \mu } ( I - P _ { \mu } ) = I - P _ { \mu } ^ { * }$ in deriving (5.25). If $\mu \in \Pi _ { B _ { * } }$ , then $\Delta _ { \mu } = 0$ by the definition of $\boldsymbol { B } _ { \ast }$ and we have

$$
v _ { \beta } = r _ { \mu } - g ^ { * } + \beta P _ { \mu } v _ { \beta } ,\tag{5.26}
$$

$$
( 1 - \beta ) P _ { \mu } ^ { * } v _ { \beta } = P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) ,\tag{5.27}
$$

$$
Z _ { \mu } ( r _ { \mu } - g ^ { * } ) = ( I - P _ { \mu } ^ { * } ) v _ { \beta } + ( 1 - \beta ) Z _ { \mu } P _ { \mu } v _ { \beta } .\tag{5.28}
$$

Lemma 5.3. The optimal gain function in $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ is nonnegative. Equivalently,

$$
\operatorname* { i n f } _ { \mu \in \Pi _ { \tilde { \mathcal { B } } } } P _ { \mu } ^ { \ast } v _ { \beta } \leq 0 .
$$

Proof. Since ${ \mathrm { I } } _ { B _ { * } } \subset { \Pi } _ { \tilde { B } }$ and, by (5.27),

$$
\begin{array} { r } { { P } _ { \mu } ^ { \ast } v _ { \beta } = \frac { 1 } { 1 - \beta } { P } _ { \mu } ^ { \ast } ( \boldsymbol { r } _ { \mu } - \boldsymbol { g } ^ { \ast } ) \leq 0 , \qquad \mu \in \Pi _ { \mathcal { B } _ { \ast } } , } \end{array}\tag{5.29}
$$

the result follows.

We now proceed to prove Props. 5.1 and 5.2.

Proof of Prop. 5.1. (i) Let $\mu \in \Pi _ { B _ { \ast } }$ <sub>∗</sub> and $s \in D _ { \mu } ^ { * }$ . Then, by (5.21) and (5.27),

$$
( P _ { \mu } ^ { * } v _ { \beta } ) ( s ) = 0 .\tag{5.30}
$$

Combining this with (5.23) and (5.28), we obtain

$$
\begin{array} { r l } & { h _ { \mu } ( s ) = v _ { \beta } ( s ) + [ Z _ { \mu } P _ { \mu } ( 1 - \beta ) v _ { \beta } ] ( s ) } \\ & { \qquad = \tilde { h } ^ { \ast } ( s ) + ( v _ { \beta } ( s ) - \tilde { h } ^ { \ast } ( s ) ) + [ Z _ { \mu } P _ { \mu } ( 1 - \beta ) v _ { \beta } ] ( s ) } \\ & { \qquad \geq h ^ { \ast } ( s ) - \eta _ { \beta } ( s ) - L _ { \mu } ( s ) \epsilon _ { \beta } , } \end{array}\tag{5.31}
$$

where, in deriving (5.31), we used the fact that $\tilde { h } ^ { * } \geq h ^ { * }$ , implied by $\tilde { \cal A } \supset { \cal A } _ { * }$ (cf. Lemma 5.1), and also the fact that $\| P _ { \mu } ( 1 - \beta ) v _ { \beta } \| _ { \infty } \leq \| ( 1 - \beta ) v _ { \beta } \| _ { \infty } = \epsilon _ { \beta }$ . This proves (5.9).

(ii) Let $\bar { \mu } \in \Pi _ { \tilde { \boldsymbol { B } } }$ be gain-optimal for the MDP $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ . Then

$$
P _ { \mu } ^ { * } v _ { \beta } \leq P _ { \mu } ^ { * } v _ { \beta } , \qquad \forall \mu \in \Pi _ { \tilde { b } } .\tag{5.32}
$$

We prove the bound (5.11) first. Let $s \in D _ { \bar { \mu } } ^ { * }$ , and consider any $\mu \in \Pi _ { \tilde { B } } ^ { s }$ so that $s \in D _ { \mu } ^ { * }$ . By (5.23) and (5.25),

$$
h _ { \bar { \mu } } ( s ) = v _ { \beta } ( s ) - ( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) + [ Z _ { \bar { \mu } } P _ { \bar { \mu } } ( 1 - \beta ) v _ { \beta } ] ( s ) - ( Z _ { \bar { \mu } } \Delta _ { \bar { \mu } } ) ( s ) ,
$$

$$
h _ { \mu } ( s ) = v _ { \beta } ( s ) - ( P _ { \mu } ^ { * } v _ { \beta } ) ( s ) + [ Z _ { \mu } P _ { \mu } ( 1 - \beta ) v _ { \beta } ] ( s ) - ( Z _ { \mu } \Delta _ { \mu } ) ( s ) .
$$

Since $- ( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) \geq - ( P _ { \mu } ^ { * } v _ { \beta } ) ( s )$ , this implies

$$
h _ { \bar { \mu } } ( s ) \geq h _ { \mu } ( s ) - [ Z _ { \mu } P _ { \mu } ( 1 - \beta ) v _ { \beta } ] ( s ) + ( Z _ { \mu } \Delta _ { \mu } ) ( s ) + [ Z _ { \bar { \mu } } P _ { \bar { \mu } } ( 1 - \beta ) v _ { \beta } ] ( s ) - ( Z _ { \bar { \mu } } \Delta _ { \bar { \mu } } ) ( s ) .
$$

The bound (5.11) then follows by taking maximum over $\mu \in \Pi _ { \tilde { B } } ^ { s }$

$$
h _ { \bar { \mu } } ( s ) \geq \operatorname* { m a x } _ { \mu \in \Pi _ { \tilde { s } } ^ { s } } h _ { \mu } ( s ) - 2 L _ { \Pi _ { \tilde { s } } ^ { s } } ( s ) ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } ) .
$$

Finally, we prove the bound (5.10). For $s \in D _ { \bar { \mu } } ^ { * }$ , by (5.21) and (5.24),

$$
\begin{array} { r } { ( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) = \frac { 1 } { 1 - \beta } ( P _ { \bar { \mu } } ^ { * } \Delta _ { \bar { \mu } } ) ( s ) \geq 0 , } \end{array}\tag{5.33}
$$

whereas $( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) \leq 0$ by Lemma 5.3. Therefore,

$$
( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) = 0 .\tag{5.34}
$$

Combining this with (5.23) and (5.25) for $\bar { \mu } ,$ it follows similarly to the derivation of (5.31) that

$$
h _ { \bar { \mu } } ( s ) = [ Z _ { \bar { \mu } } ( r _ { \bar { \mu } } - g ^ { * } ) ] ( s ) \geq h ^ { * } ( s ) - \eta _ { \beta } ( s ) - L _ { \bar { \mu } } ( s ) ( \epsilon _ { \beta } + \bar { \delta } _ { 2 } ) .
$$

This proves (5.10).

Proof of Prop. 5.2. Let $s \in D _ { \bar { \mu } } ^ { * }$ . Then $\bar { \mu } ( s ) \in \mathcal { A } _ { * } ( s )$ ; otherwise, $( P _ { \bar { \mu } } g ^ { * } ) ( s ) < g ^ { * } ( s )$ , whereas $( P _ { \bar { \mu } } g ^ { * } ) ( s ) \geq ( P _ { \bar { \mu } } g _ { \bar { \mu } } ) ( s ) = g _ { \bar { \mu } } ( \bar { s } ) = g ^ { * } ( s ) $ , a contradiction. Equality (5.12) has already been established (see (5.34)). This proves (i). Moreover, (5.33) and (5.34) yield $( P _ { \bar { \mu } } ^ { * } \Delta _ { \bar { \mu } } ) ( s ) = 0 $ which is equivalent to (ii).

Conversely, under conditions (i) and (ii), we have

$$
( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) = 0 , \quad ( P _ { \bar { \mu } } ^ { * } \Delta _ { \bar { \mu } } ) ( s ) = 0 , \quad \mathrm { a n d ~ } ( P _ { \bar { \mu } } g ^ { * } ) ( s ) = g ^ { * } ( s ) ,
$$

where the last equality follows from $\bar { \mu } ( s ) \in \mathcal { A } _ { * } ( s )$ . Combining the first two equalities with (5.24) yields $[ P _ { \bar { \mu } } ^ { * } ( r _ { \bar { \mu } } - g ^ { * } ) ] ( s ) = 0$ , which together with the third equality implies $g _ { \bar { \mu } } ( s ) = g ^ { * } ( s )$ . Hence, $s \in D _ { \bar { \mu } } ^ { * }$ , completing the proof of the equivalence.

For part (iii), suppose that (5.12) holds: $( P _ { \bar { u } } ^ { * } v _ { \beta } ) ( s ) = 0$ . Then, for every $\mu \in \Pi _ { B _ { * } } , ( P _ { \mu } ^ { * } v _ { \beta } ) ( s ) \geq$ $( P _ { \bar { \mu } } ^ { * } v _ { \beta } ) ( s ) = 0$ , whereas, by (5.27), $( 1 - \beta ) ( P _ { \mu } ^ { * } { \dot { v } } _ { \beta } ) ( s ) = [ P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) ] ( s ) \leq 0$ . Therefore,

$$
( P _ { \mu } ^ { * } v _ { \beta } ) ( s ) = 0 \quad \mathrm { a n d } \ : \ : \ : [ P _ { \mu } ^ { * } ( r _ { \mu } - g ^ { * } ) ] ( s ) = 0 ,
$$

which is equivalent to (iii). Thus, (5.12) implies (iii).

![](images/8b77213b5dee9dad2c8d6853d87454341c4e6c5cbf638934b348a34f8148ee76.jpg)  
(a) Fraction of gain-optimal states

![](images/916e3052a9d658ada88af66bddaea54d38a5660b3e06bbf3f051039797ff81ec.jpg)  
(b) Gain-transient tradeof  
Figure 3: Comparison of gain optimality and transient performance across algorithms. In panel (a), the shaded region denotes one standard deviation across the 100 MDPs.

## 6 Experimental results

The following results are preliminary; a more extensive empirical evaluation is in progress and will be included in a future version of this work. We evaluate our algorithms on 100 randomly generated multichain MDPs with hierarchical structure, constructed using Bather’s UHD framework. The number of levels, communicating classes, and transient states are randomized, and rewards and transitions are sampled to induce diverse gain structures and transient behavior. Full details are provided in Appendix C.1.

We compare the base algorithm with Algorithms 2 and 3, along with discounted Q-learning as a baseline. We run the 0-discount policy iteration algorithm [30] (see also [22, Sec. 10.3]) using the true MDP model to compute a bias-optimal policy, which serves as an exact benchmark. For reproducibility, all algorithms are updated using a round-robin scheme, in which each outer iteration advances every phase by one full, deterministic sweep over its state-action pairs (full details in Appendix C.2). We emphasize that our theoretical results hold under general asynchronous exploration schemes; round-robin updates were adopted here purely for implementation simplicity and ease of comparison across algorithms.

We say a state s is gain-optimal under a policy π $\mathrm { i f } \ g _ { \pi } ( s ) = g ^ { * } ( s )$ . Figure 3a shows, at each training iteration, the fraction of states that are gain-optimal under each algorithm’s current policy, within a tolerance of $1 0 ^ { - 2 } \ ( \mathrm { i . e . , } \ | g _ { \pi ^ { ( t ) } } ( s ) - g ^ { * } ( s ) | < 1 0 ^ { - 2 } )$ , averaged across the 100 MDPs, with the shaded band denoting one standard deviation across MDPs. This tolerance is chosen to account for numerical precision in our computation of $g _ { \pi }$ and $g ^ { * }$ rather than as a substantive design choice; results are similar under smaller tolerances. The base algorithm achieves the highest coverage, stabilizing near 96%. Algorithm 3 follows closely, while Algorithm 2 achieves slightly lower coverage. These results confirm that the base algorithm is most efective at finding gain-optimal policies, while extensions introduce a modest degradation in gain performance.

Comparing bias across policies with diferent gains requires care. For this preliminary evaluation, we compare transient performance using the expected cumulative reward obtained prior to entering a recurrent class. Figure 3b reports this quantity for each algorithm’s final policy after 750,000 iterations, averaged across all initial states within each MDP and then across the 100 MDPs. The x-axis gives the fraction of gain-optimal states, and the y-axis measures this transient reward. We leave a direct comparison of policy bias to future work.

The base algorithm achieves the highest gain-optimality but exhibits significantly worse transient performance, stemming from the policy $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ for states in classes where exiting occurs. Algorithm 2 achieves the best transient performance but with reduced gain-optimality. Algorithm 3 provides a strong compromise: although its aggregate gain-optimality in Figure 3a sits between the base algorithm and Algorithm 2, its transient performance remains close to Algorithm 2’s and to the bias-optimal benchmark. Since gain-optimality is the primary objective in our setting, and transient performance is only meaningfully compared among near-gain-optimal policies, we view this as evidence that Algorithm 3 achieves a favorable tradeof between the two objectives.

To validate these aggregated results, we examine per-MDP performance (Appendix C.3). The trends remain consistent: the base algorithm is strongest in terms of gain-optimality but exhibits large transient failures, while the extensions enhance transient performance while maintaining a high degree of gain-optimality.

## 7 Convergence proofs for the base algorithm

In this section, we prove the convergence results for Algorithm 1. The aggregated OSP is already known to be a transient total-reward problem, and its optimal values coincide with the optimal gains of the original MDP. These properties underlie the reduced formulation of Schweitzer [24] and the explicit OSP formulation of Ohno [21]. We include proofs for completeness in the notation of our formulation.

Section 7.1 contains the supporting results. We first establish convergence of the Phase 2 and Phase 3 learning procedures using existing results for RVI Q-learning and total-reward Q-learning. We then relate the optimal values and policy behavior in the original MDP and the aggregated OSP. Section 7.2 combines these results to prove convergence of the gain estimates and eventual gain optimality of the constructed policies.

For the comparison between the original and aggregated problems, we use the following componentwise form of the AOE:

$$
g ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ( s ^ { \prime } ) , \quad \forall s \in S ,\tag{7.1}
$$

$$
g ( s ) + h ( s ) = \operatorname* { m a x } _ { a \in A _ { + } ( s ) } \left\{ r ( s , a ) + \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) h ( s ^ { \prime } ) \right\} , \quad \forall s \in \mathcal { S } ,\tag{7.2}
$$

where

$$
\mathcal { A } _ { * } ( s ) : = \left\{ a \in \mathcal { A } ( s ) : g ( s ) = \sum _ { s ^ { \prime } \in \mathcal { S } } p ( s ^ { \prime } \mid s , a ) g ( s ^ { \prime } ) \right\} .
$$

This system of equations is equivalent to the vector form given in (1.1).

## 7.1 Supporting lemmas for base algorithm convergence

We first consider the learning procedures in Phases 2 and 3. Since each restricted sub-MDP obtained from Bather’s UHD is communicating, the convergence theory for RVI Q-learning [33, 38, 39] directly yields the Phase 2 result below.

Lemma 7.1 (Phase 2 convergence of RVI Q-learning). Fix a communicating sub-MDP $\mathcal { M } _ { \ell , k }$ obtained in Phase 1. Suppose its Phase 2 learner uses a reference function $f _ { \ell , k }$ satisfying Assumption A.1, a stepsize sequence satisfying Assumption A.2, and an update schedule satisfying Assumption A.3. Then, almost surely:

(i) $r _ { \ell , k } ^ { ( t ) } = f _ { \ell , k } ( Q _ { \ell , k } ^ { ( t ) } ) \to r _ { \ell , k } ^ { * }$ , where $r _ { \ell , k } ^ { * }$ is the optimal gain of $\mathcal { M } _ { \ell , k }$

(ii) for all suficiently large t, every policy greedy with respect to $Q _ { \ell , k } ^ { ( t ) }$ is gain-optimal for $\mathcal { M } _ { \ell , k }$

Proof. By Bather’s UHD, each $\mathcal { M } _ { \ell , k }$ is communicating. The result follows directly from Theorem 3.1 in [38]. □

We next consider Phase 3. Properness of every stationary policy makes the aggregated OSP a transient total-reward problem and provides the structural condition used in the Q-learning convergence argument.

Lemma 7.2 (Properness of all stationary policies on the aggregated OSP). Let $\bar { \mathcal { M } } = ( \bar { \mathcal { S } } , \bar { \mathcal { A } } , \bar { p } , \bar { r } )$ be the limiting aggregated OSP constructed in Phase 3 of Algorithm 1. Then every stationary policy π¯ on M<sup>¯</sup> is proper; that $i s ,$ it reaches the terminal state $\Delta$ with probability 1 from every initial state.

Proof. By Bather’s UHD, all states in $\tau$ are transient under every stationary policy. Hence, starting from any $s \in \mathcal T$ , the process reaches an aggregated state $\bar { s } _ { \ell , k }$ with probability 1. Therefore, it sufices to prove properness when the process is started from an aggregated state.

Fix a stationary policy π¯ on $\bar { \mathcal { M } }$ and suppose, for the sake of contradiction, that the Markov chain induced by π¯ has a recurrent class $\bar { R } \neq \{ \Delta \}$ . Since stopping actions send the system to $\Delta$ with probability 1, no state at level 0 can belong to ${ \bar { R } } ,$ , as level 0 states admit only the stopping action. Consequently, every state in R<sup>¯</sup> must have level $\ell \geq 1$ , and the actions selected by π¯ at such states must belong to the set of exit actions $\mathcal { E } _ { \ell , k }$

Let $\bar { s } _ { \ell , k } ^ { \mathrm { m i n } }$ be a state of minimal level among the states in R<sup>¯</sup>. By construction of the aggregated OSP, every action in $\mathcal { E } _ { \ell , k }$ has positive probability of transitioning to a state at a strictly lower level. In particular, the action $\bar { \pi } ( \bar { s } _ { \ell , k } ^ { \mathrm { m i n } } )$ leads to a lower level with positive probability. However, by the choice of $\bar { s } _ { \ell , k } ^ { \mathrm { m i n } }$ , no state of lower level belongs to ${ \bar { R } } .$ Hence, the process leaves $\bar { R }$ with positive probability, contradicting the assumption that $\bar { R }$ is a recurrent class.

Therefore, the induced Markov chain has no recurrent class other than {∆}. Equivalently, every stationary policy on $\bar { \mathcal { M } }$ is proper. □

The stopping rewards used in Phase 3 are the gain estimates produced by Phase 2 and therefore vary during learning. By Lemma 7.1, their deviations from the limiting stopping rewards vanish almost surely. Together with the properness of all stationary policies established in Lemma 7.2, this allows us to apply Q-learning convergence results with vanishing reward perturbations.

Lemma 7.3 (Phase 3 convergence of Q-learning). Let M<sup>¯</sup> be the limiting aggregated $O S P$ constructed in Phase 3 of Algorithm 1. Under the stepsize and asynchronous update conditions of Assumption $B . 1 ,$ the total-reward Q-learning iterates $\bar { Q } ^ { ( t ) }$ generated $b y$ the sampling and update scheme in Section 3.1.3 satisfy

$$
{ \bar { Q } } ^ { ( t ) } \to { \bar { Q } } ^ { * } \qquad a l m o s t \ s u r e l y ,
$$

where $\bar { Q } ^ { * }$ is the optimal action-value function of $\bar { \mathcal { M } }$

Proof. First note that the rewards $\bar { r } ^ { ( t ) }$ used in the aggregated OSP are time-varying because the stopping rewards are given by the estimates $r _ { \ell , k } ^ { ( t ) }$ produced by RVI Q-learning in Phase 2 of Algorithm 1. By Lemma 7.1, these estimates converge almost surely to the optimal gains $r _ { \ell , k } ^ { * } .$ Hence, we may write

$$
\bar { r } ^ { ( t ) } = \bar { r } + \varepsilon ^ { ( t ) } ,
$$

where $\bar { r }$ is the limiting reward function obtained by replacing each $r _ { \ell , k } ^ { ( t ) }$ with $r _ { \ell , k } ^ { * }$ and $\varepsilon ^ { ( t ) } \to 0$ almost surely. Thus, the $\mathrm { Q } \mathrm { . }$ -learning recursion is a vanishing perturbation of standard Q-learning. By Lemma 7.2, all stationary policies on $\bar { \mathcal { M } }$ are proper. By robustness results for stochastic approximation (see Proposition 4.5 of [5]), the convergence guarantees of Proposition 5.5 part (a) of [5] hold. Thus, $\bar { Q } ^ { ( t ) } \to \bar { Q } ^ { * }$ almost surely. □

We next relate the optimal values of the aggregated OSP to the optimal gains of the original MDP, beginning with the aggregated optimality equations.

Lemma 7.4 (Optimality equations of the aggregated OSP). Let $\bar { Q } ^ { * }$ denote the optimal action-value function of the limiting aggregated OSP $\bar { \mathcal { M } } ,$ and define its optimal value function by

$$
v ^ { * } ( \bar { s } ) = \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \bar { s } ) } \bar { Q } ^ { * } ( \bar { s } , \bar { a } ) , \qquad \forall \bar { s } \in \bar { \mathcal { S } } .
$$

Then $\bar { g } ^ { * } : = v ^ { * }$ satisfies $\bar { g } ^ { * } ( \Delta ) = 0$ and the following optimality equations. For each class state $\bar { s } _ { \ell , k }$

$$
\bar { g } ^ { * } ( \bar { s } _ { \ell , k } ) = \operatorname* { m a x } \left\{ r _ { \ell , k } ^ { * } , \ \underset { a \in \mathcal { E } _ { \ell , k } } { \operatorname* { m a x } } \left[ \sum _ { \ell ^ { \prime } , k ^ { \prime } } \bar { p } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } \mid \bar { s } _ { \ell , k } , \bar { a } ) \bar { g } ^ { * } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } ) + \sum _ { s \in \mathcal { T } } \bar { p } ( s \mid \bar { s } _ { \ell , k } , \bar { a } ) \bar { g } ^ { * } ( s ) \right] \right\} .\tag{7.3}
$$

When $\mathcal { E } _ { \ell , k } = \emptyset$ , the exit term in (7.3) is omitted, so $\bar { g } ^ { * } ( \bar { s } _ { \ell , k } ) = r _ { \ell , k } ^ { * }$ . For retained transient states,

$$
\bar { g } ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \Bigg \{ \sum _ { \ell , k } \bar { p } ( \bar { s } _ { \ell , k } \mid s , a ) \bar { g } ^ { * } ( \bar { s } _ { \ell , k } ) + \sum _ { s ^ { \prime } \in \mathcal { T } } \bar { p } ( s ^ { \prime } \mid s , a ) \bar { g } ^ { * } ( s ^ { \prime } ) \Bigg \} , \qquad s \in \mathcal { T } .\tag{7.4}
$$

Proof. For all $\bar { s } \in \bar { S }$ , let

$$
\begin{array} { r l } & { v ^ { * } ( \bar { s } ) = \displaystyle \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \bar { s } ) } \bar { Q } ^ { * } ( \bar { s } , \bar { a } ) } \\ & { \quad \quad = \displaystyle \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \bar { s } ) } \sum _ { \bar { s } ^ { \prime } } \bar { p } ( \bar { s } ^ { \prime } \mid \bar { s } , \bar { a } ) \big ( \bar { r } ( \bar { s } , \bar { a } ) + v ^ { * } ( \bar { s } ^ { \prime } ) \big ) . } \end{array}\tag{7.5}
$$

We now show that setting $\bar { g } ^ { * } ( \bar { s } ) = v ^ { * } ( \bar { s } )$ solves the optimality equations (7.3) and (7.4) by specializing to the two types of non-terminal states in $\bar { s } { : }$ aggregated and transient states. Note that $v ^ { * } ( \Delta ) = 0$ by definition.

(i) Aggregated states $\bar { s } _ { \ell , k }$ . In these states, there are two types of admissible actions: the stopping action $a _ { \ell , k } ^ { s t o p }$ and exit actions $\bar { a } \in \mathcal { E } _ { \ell , k } . \mathrm { ~ H ~ } a _ { \ell , k } ^ { s t o p }$ is selected, then $\bar { p } ( \Delta \mid \bar { s } _ { \ell , k } , a _ { \ell , k } ^ { s t o p } ) = 1$ and $\bar { r } ( \bar { s } _ { \ell , k } , a _ { \ell , k } ^ { s t o p } ) = r _ { \ell , k } ^ { * } ,$ thus,

$$
\sum _ { \bar { s } ^ { \prime } } \bar { p } ( \bar { s } ^ { \prime } \mid \bar { s } _ { \ell , k } , a _ { \ell , k } ^ { s t o p } ) \big ( \bar { r } \big ( \bar { s } _ { \ell , k } , a _ { \ell , k } ^ { s t o p } \big ) + v ^ { * } ( \bar { s } ^ { \prime } ) \big ) = r _ { \ell , k } ^ { * } + v ^ { * } ( \Delta ) = r _ { \ell , k } ^ { * } .
$$

If instead $\bar { \boldsymbol { a } } \in \mathcal { E } _ { \ell , k }$ is selected, then $\bar { r } ( \bar { s } _ { \ell , k } , \bar { a } ) = 0$ for all s¯ and transitions can only lead to aggregated states $\bar { s } _ { \ell ^ { \prime } , k ^ { \prime } }$ or transient states $s \in \mathcal T$ . Hence,

$$
\sum _ { \vec { s } ^ { \prime } } \bar { p } ( \vec { s } ^ { \prime } \mid \bar { s } _ { \ell , k } , \bar { a } ) \big ( \bar { r } ( \bar { s } _ { \ell , k } , \bar { a } ) + v ^ { * } ( \bar { s } ^ { \prime } ) \big ) = \sum _ { \ell ^ { \prime } , k ^ { \prime } } \bar { p } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } \mid \bar { s } _ { \ell , k } , \bar { a } ) v ^ { * } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } ) + \sum _ { s \in \mathcal { T } } \bar { p } ( s \mid \bar { s } _ { \ell , k } , \bar { a } ) v ^ { * } ( s ) .
$$

Thus, maximizing over these two scenarios yields (7.3).

(ii) Transient states $s \in \mathcal T$ . Here, $\bar { \mathcal { A } } ( s ) = \mathcal { A } ( s )$ and all one-step rewards are zero by definition. Hence, equation (7.5) reduces to

$$
v ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \sum _ { \bar { s } \in \bar { \mathcal { S } } } \bar { p } ( \bar { s } \mid s , a ) v ^ { * } ( \bar { s } ) ,
$$

and splitting the sum over $\bar { s }$ into aggregated and transient states yields (7.4).

We now relate the optimal OSP values to the original optimal gains. Recall the aggregation map $\phi$ from Section 3.1.3:

$$
\phi ( s ) = \left\{ \begin{array} { l l } { \bar { s } _ { \ell , k } , } & { s \in C _ { \ell , k } , } \\ { s , } & { s \in \mathcal { T } . } \end{array} \right.
$$

The following lemma shows that the optimal gain at each original state equals the optimal OSP value at its corresponding aggregated state.

Lemma 7.5 (Equivalence of optimal gains and OSP values). Let $g ^ { * }$ be the optimal gain vector of $\mathcal { M } _ { : }$ and let $\bar { g } ^ { * }$ be the optimal value function of the limiting aggregated OSP M<sup>¯</sup> . Then

$$
g ^ { * } ( s ) = \bar { g } ^ { * } ( \phi ( s ) ) , \qquad \forall s \in \mathcal { S } .
$$

Proof. First note that for every recurrent class R induced by a policy in the original MDP, there exists a unique class $C _ { \ell , k }$ such that $R \subseteq C _ { \ell , k }$ . Moreover, because every $C _ { \ell , k }$ is communicating, from any state $s \in C _ { \ell , k }$ , there exists a policy that can reach any other state $s ^ { \prime } \in C _ { \ell , k }$ . This implies that the optimal gain from s must equal the optimal gain from $s ^ { \prime } .$ . Otherwise, a higher gain from one state could be leveraged to improve the other, leading to a contradiction. Thus, we have that $g ^ { \ast } ( s ) = g ^ { \ast } ( s ^ { \prime } ) \forall s , s ^ { \prime } \in C _ { \ell , k } \ [ 2 4 ]$

We therefore define a lifted function $\tilde { g } ^ { * }$ on $\bar { \boldsymbol { S } }$ such that

$$
\begin{array} { r l r } & { \tilde { g } ^ { \ast } ( \bar { s } _ { \ell , k } ) : = g ^ { \ast } ( s ) , \quad } & { s \in C _ { \ell , k } , } \\ & { \tilde { g } ^ { \ast } ( s ) : = g ^ { \ast } ( s ) , \quad } & { s \in \mathcal { T } . } \end{array}
$$

Let $\pi ^ { * }$ be an optimal policy of M. Fix a class $C _ { \ell , k }$ and consider two cases based on the behavior of $\pi ^ { * }$ within it.

Case 1: $\pi ^ { * }$ never exits $C _ { \ell , k }$

Then $\pi ^ { \ast } ( s ) \in \mathcal { A } _ { \ell , k } ( s )$ for all $s \in C _ { \ell , k }$ . By Lemma $7 . 1 , r _ { \ell , k } ^ { \ast }$ is the maximal gain attainable under such policies. Therefore,

$$
g ^ { * } ( s ) = r _ { \ell , k } ^ { * } , \qquad \forall s \in C _ { \ell , k } .
$$

Case $\mathbf { 2 } \colon \pi ^ { * }$ exits $C _ { \ell , k }$

Then there exists $s _ { 0 } \in C _ { \ell , k }$ such that $\pi ^ { \ast } ( s _ { 0 } ) \in \mathcal { E } _ { \ell , k } ( s _ { 0 } )$ . Since $\pi ^ { * }$ is optimal, its gain vector satisfies the $\mathrm { A O E \ ( 7 . 1 ) - ( 7 . 2 ) }$ . Hence, we must have

$$
g ^ { * } ( s _ { 0 } ) = \operatorname* { m a x } _ { a \in { \mathcal { E } } _ { \ell , k } ( s _ { 0 } ) } \sum _ { s \in { \mathcal { S } } } p ( s \mid s _ { 0 } , a ) g ^ { * } ( s ) .
$$

Given that optimal gains are constant within $C _ { \ell , k }$ , this maximum extends uniformly across all states in the class. Thus, for any $s \in C _ { \ell , k }$ 2

$$
g ^ { * } ( s ) = g ^ { * } ( s _ { 0 } ) = \operatorname* { m a x } _ { ( s , a ) \in { \mathscr { E } _ { \ell , k } } } \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) .
$$

Combining cases 1 and 2, we have that for all $s \in C _ { \ell , k }$

$$
g ^ { * } ( s ) = \operatorname* { m a x } \left\{ r _ { \ell , k } ^ { * } , \operatorname* { m a x } _ { ( s , a ) \in { \mathcal E } _ { \ell , k } } \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) \right\} .
$$

Now, splitting the inner sum into classes and transient states we have

$$
\begin{array} { l } { { \displaystyle \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) = \sum _ { \ell ^ { \prime } , k ^ { \prime } } \sum _ { s ^ { \prime } \in C _ { \ell ^ { \prime } , k ^ { \prime } } } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) + \sum _ { s ^ { \prime } \in T } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) } } \\ { { = \displaystyle \sum _ { \ell ^ { \prime } , k ^ { \prime } } \bar { p } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } \mid \bar { s } _ { \ell , k } , \bar { a } ) \tilde { g } ^ { * } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } ) + \sum _ { s ^ { \prime } \in T } \bar { p } ( s ^ { \prime } \mid \bar { s } _ { \ell , k } , \bar { a } ) \tilde { g } ^ { * } ( s ^ { \prime } ) , } } \end{array}
$$

where the last equality follows from the definition of $\bar { p }$ and $\tilde { g } ^ { * }$

Thus, for every aggregated state $\bar { s } _ { \ell , k }$

$$
\tilde { g } ^ { * } ( \bar { s } _ { \ell , k } ) = \operatorname* { m a x } \left\{ r _ { \ell , k } ^ { * } , \operatorname* { m a x } _ { a \in \mathcal { E } _ { \ell , k } } \left[ \sum _ { \ell ^ { \prime } , k ^ { \prime } } \bar { p } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } \mid \bar { s } _ { \ell , k } , \bar { a } ) \tilde { g } ^ { * } ( \bar { s } _ { \ell ^ { \prime } , k ^ { \prime } } ) + \sum _ { s \in \mathcal { T } } \bar { p } ( s \mid \bar { s } _ { \ell , k } , \bar { a } ) \tilde { g } ^ { * } ( s ) \right] \right\} .\tag{7.6}
$$

This is precisely equation (7.3).

For any transient state $s \in \mathcal T$ , the AOE (7.1) - (7.2) in the original MDP gives

$$
g ^ { * } ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) g ^ { * } ( s ^ { \prime } ) .
$$

Splitting the sum and using the definitions of $\bar { p }$ and $\tilde { g } ^ { * }$ we have

$$
\tilde { g } ^ { * } ( s ) = \operatorname* { m a x } _ { a \in A ( s ) } \Big [ \sum _ { \ell , k } \bar { p } ( \bar { s } _ { \ell , k } \mid s , a ) \tilde { g } ^ { * } ( \bar { s } _ { \ell , k } ) + \sum _ { s ^ { \prime } \in \mathcal { T } } \bar { p } ( s ^ { \prime } \mid s , a ) \tilde { g } ^ { * } ( s ^ { \prime } ) \Big ] ,\tag{7.7}
$$

which is equation (7.4).

Equations (7.6) and (7.7) show that $\tilde { g } ^ { * }$ satisfies the optimality equations (7.3)–(7.4) of $\bar { \mathcal { M } }$ By Lemma 7.4, the optimal value function $\bar { g } ^ { * }$ of the aggregated OSP M<sup>¯</sup> also solves this system. Therefore,

$$
\tilde { g } ^ { * } = \bar { g } ^ { * } .
$$

By the definition of $\tilde { g } ^ { * }$ , this implies

$$
\begin{array} { r l } & { g ^ { \ast } ( \boldsymbol { s } ) = \bar { g } ^ { \ast } ( \bar { s } _ { \ell , k } ) , \quad \forall \boldsymbol { s } \in C _ { \ell , k } , } \\ & { } \\ & { g ^ { \ast } ( \boldsymbol { s } ) = \bar { g } ^ { \ast } ( \boldsymbol { s } ) , \quad \forall \boldsymbol { s } \in \mathcal { T } . } \end{array}
$$

It remains to relate an aggregated policy to its Phase 4 lifted policy. The following lemma shows that the probability of eventually remaining in a class under the lifted policy equals the probability of stopping at the corresponding aggregated state. When the Phase 2 sub-MDP policies are optimal, this correspondence equates the gain of the lifted policy with the total-reward value of the aggregated policy.

Lemma 7.6 (Preservation of stopping-class distributions). Fix an iteration $t ,$ and let $\pi ^ { ( t ) }$ be the Phase 4 lifted policy corresponding to $\bar { \pi } ^ { ( t ) }$ . Suppose every reaching policy used in this construction is class-preserving and reaches its selected exit state almost surely from every state in the corresponding class. Then, for every initial state $s \in S$ and every class $C _ { \ell , k }$

$$
\mathbb { P } _ { \pi ^ { ( t ) } } \big ( a b s o r b e d ~ i n ~ C _ { \ell , k } ~ | ~ s _ { 0 } = s \big ) = \mathbb { P } _ { \bar { \pi } ^ { ( t ) } } \big ( s t o p s ~ a t ~ \bar { s } _ { \ell , k } ~ | ~ \bar { s } _ { 0 } = \phi ( s ) \big ) .
$$

Proof. Let $( s _ { q } ) _ { q \geq 0 }$ and $( \bar { s } _ { q } ) _ { q \geq 0 }$ be the Markov chains under the fixed policies $\pi ^ { ( t ) }$ and $\bar { \pi } ^ { ( t ) }$ in the original MDP and the aggregated OSP, respectively, with $\bar { s } _ { 0 } = \phi ( s _ { 0 } )$ . We construct a sub-process of $( s _ { q } ) _ { q \geq 0 }$ by observing it only at times when the aggregated state $\phi ( s _ { q } )$ can change. Formally, define $\tau _ { 0 } = 0$ and $\tau _ { q + 1 }$ for $q \geq 0$ recursively as follows:

• If $s _ { \tau _ { q } } \in \mathcal { T } .$ , set $\tau _ { q + 1 } = \tau _ { q } + 1$

• If $s _ { \tau _ { q } } \in C _ { \ell , k }$ and $\bar { \pi } ^ { ( t ) } ( \bar { s } _ { \ell , k } ) = ( s ^ { * } , a ^ { * } )$ , let $\sigma _ { q } > \tau _ { q }$ be the first time after $\tau _ { q }$ at which $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ hits ${ \bar { s } } ^ { * }$ starting from $s _ { \tau _ { q } }$ , and set $\tau _ { q + 1 } = \sigma _ { q } + 1$ , where action $a ^ { * }$ is taken at time σa $\sigma _ { q }$

• If $s _ { \tau _ { q } } \in C _ { \ell , k }$ and $\bar { \pi } ^ { ( t ) } ( \bar { s } _ { \ell , k } ) = a _ { \ell , k } ^ { \mathrm { s t o p } }$ , the sub-process stops and the chain is absorbed in $C _ { \ell , k }$ By construction, the sub-process $( \phi ( s _ { \tau _ { q } } ) ) _ { q \geq 0 }$ evolves only when the aggregated state can change. A direct calculation (using the definition of $\bar { p }$ in Section 3.1.3) shows that for every aggregated state s¯ and action ${ \bar { a } } ,$ the one-step transition probabilities of $\phi ( s _ { \tau _ { q + 1 } } )$ under $\pi ^ { ( t ) }$ coincide with those of $\bar { s } _ { q + 1 }$ under $\bar { \pi } ^ { ( t ) }$ , conditional on $\bar { s } _ { q } = \phi ( s _ { \tau _ { q } } )$

Since $\bar { s } _ { 0 } = \phi ( s _ { 0 } )$ , the one-step transition distributions coincide, and the evolution depends only on the current aggregated state, it follows that $( \phi ( s _ { \tau _ { q } } ) ) _ { q \geq 0 }$ and $( \bar { s } _ { q } ) _ { q \geq 0 }$ have the same law. In particular, the probability that the sub-process stops in $C _ { \ell , k }$ equals the probability that the aggregated process stops at $\bar { s } _ { \ell , k }$

Finally, by construction, absorption of the original chain $( s _ { q } ) _ { q \geq 0 }$ in $C _ { \ell , k }$ is equivalent to the sub-process stopping in $C _ { \ell , k }$ . This yields

$$
\mathbb { P } _ { \pi ^ { ( t ) } } ( \mathrm { a b s o r b e d ~ i n ~ } C _ { \ell , k } \mid s _ { 0 } = s ) = \mathbb { P } _ { \bar { \pi } ^ { ( t ) } } ( \mathrm { s t o p s ~ a t ~ } \bar { s } _ { \ell , k } \mid \bar { s } _ { 0 } = \phi ( s ) ) ,
$$

as claimed.

## 7.2 Proofs of the main convergence results for the base algorithm

We now combine the supporting lemmas to prove Theorems 3.1 and 3.2. Theorem 3.1 asserts that the Phase 3 Q-values converge to the optimal action-value function of the limiting aggregated OSP and that the resulting gain estimates converge to $g ^ { * }$

Proof of Theorem 3.1. By Lemma 7.1, for every class $C _ { \ell , k }$ , Phase 2 RVI Q-learning converges almost surely and produces the optimal gain $r _ { \ell , k } ^ { * }$ when actions are restricted to $\boldsymbol { \mathcal { A } } _ { \ell , k }$

Consider the limiting aggregated OSP M<sup>¯</sup> constructed in Phase 3. By Lemma 7.3, Q-learning on $\bar { \mathcal { M } }$ converges almost surely to the optimal action-value function $\bar { Q } ^ { * }$ . Define

$$
\bar { g } ^ { * } ( \bar { s } ) = \operatorname* { m a x } _ { \bar { a } \in \bar { \mathcal { A } } ( \bar { s } ) } \bar { Q } ^ { * } ( \bar { s } , \bar { a } ) , \qquad \bar { s } \in \bar { \mathcal { S } } .
$$

$\mathrm { B y }$ Lemma 7.4, the function $\bar { g } ^ { * }$ satisfies the optimality equations $( 7 . 3 ) \mathrm { - } ( 7 . 4 )$ of $\bar { \mathcal { M } }$ . Now define $g : S $ R by lifting $\bar { g } ^ { * }$ back to the original state space: $g ( s ) = { \bar { g } } ^ { * } ( s )$ for $s \in \mathcal T$ and $g ( s ) = \bar { g } ^ { * } ( \bar { s } _ { \ell , k } )$ for $s \in C _ { \ell , k }$ . Then by Lemma 7.5, we have $g = g ^ { * }$ . Consequently, by the definition of $g ^ { ( t ) }$ and the convergence $\bar { Q } ^ { ( t ) } \to \bar { Q } ^ { * }$ , we obtain $g ^ { ( t ) }  g ^ { * }$ almost surely. □

We next prove Theorem 3.2, which establishes that, almost surely, every policy constructed in Phase 4 is gain-optimal after some finite time.

Proof of Theorem 3.2. By Bather’s UHD, each $C _ { \ell , k }$ is communicating under the restricted action set $\boldsymbol { \mathcal { A } } _ { \ell , k }$ . Hence, each restricted MDP $\mathcal { M } _ { \ell , k }$ is communicating. By Lemma 7.1, for each $( \ell , k )$ , there exists a finite time $T _ { \ell , k }$ such that for all $t \geq T _ { \ell , k }$ the policy $\pi _ { \ell , k } ^ { ( t ) }$ is optimal for $\mathcal { M } _ { \ell , k }$ . In particular,

$$
g _ { \pi _ { \ell , k } ^ { ( t ) } } ( s ) = r _ { \ell , k } ^ { * } , \qquad \forall s \in C _ { \ell , k } , t \geq T _ { \ell , k } .
$$

By Lemma 7.3, total-reward Q-learning on the aggregated OSP M<sup>¯</sup> converges almost surely. As Q-learning converges, the estimates $\bar { Q } ^ { ( t ) }$ approach the optimal action-values $\bar { Q } ^ { * }$ . Because the state and action spaces are finite, the diference between optimal and suboptimal action values at each state is bounded away from zero. Once the estimation error becomes smaller than this separation, the action that maximizes $\bar { Q } ^ { ( t ) }$ must coincide with an action that maximizes $\bar { Q } ^ { * }$ . Thus, after some finite time $\bar { T } _ { \ d { \cdot } }$ , we have that for all $t \geq \bar { T }$ , the policy $\bar { \pi } ^ { ( t ) }$ is optimal for $\bar { \mathcal { M } } .$ Hence,

$$
\bar { g } _ { \bar { \pi } ^ { ( t ) } } ( \bar { s } ) = \bar { g } ^ { * } ( \bar { s } ) , \qquad \forall \bar { s } \in \bar { \mathcal { S } } , t \geq \bar { T } .
$$

Thus, by Lemma 7.5, the optimal gain vector of the original MDP M satisfies

$$
g ^ { \ast } \big ( s \big ) = \left\{ \begin{array} { l l } { \bar { g } _ { \bar { \pi } ^ { ( t ) } } \big ( s \big ) , } & { s \in \mathcal { T } , } \\ { \bar { g } _ { \bar { \pi } ^ { ( t ) } } \big ( \bar { s } _ { \ell , k } \big ) , } & { s \in C _ { \ell , k } , } \end{array} \right. \quad \forall t \geq \bar { T } .\tag{7.8}
$$

Define

$$
T : = \operatorname* { m a x } \left( \bar { T } , \operatorname* { m a x } _ { \ell , k } T _ { \ell , k } \right) .
$$

Since this is the maximum of finitely many almost surely finite random variables, $T < \infty$ almost surely.

Fix any $t \geq T$ . We show that the policy $\pi ^ { ( t ) }$ constructed by Algorithm 1 attains $g ^ { * }$ in (7.8).

In a finite MDP, the gain of a stationary policy is determined by the distribution of possible recurrent classes eventually entered under that policy; transient states contribute only finitely many rewards almost surely and therefore do not afect the long run gain.

All states in $\tau$ are transient under every stationary policy by Bather’s UHD. Since $\bar { \pi } ^ { ( t ) }$ is proper on $\bar { \mathcal { M } }$ by Lemma 7.2, all policies reach a state $\bar { s } _ { \ell , k }$ where the system stops with probability 1. Since the absorption and stopping distributions coincide by Lemma 7.6, the original chain under $\pi ^ { ( t ) }$ is also absorbed with probability 1. Therefore, all non-stopping class states are transient under $\pi ^ { ( t ) }$ . At stopping classes $C _ { \ell , k } , \pi ^ { ( t ) }$ coincides with $\pi _ { \ell , k } ^ { ( t ) }$ , which achieves gain $r _ { \ell , k } ^ { * }$ for all $s \in C _ { \ell , k }$ by Lemma 7.1. Thus, for any $s \in S$

$$
g _ { \pi ^ { ( t ) } } { \bigl ( } s { \bigr ) } = \sum _ { ( \ell , k ) } \mathbb { P } _ { \pi ^ { ( t ) } } { \bigl ( } \mathrm { a b s o r b e d } \ \mathrm { i n } \ C _ { \ell , k } \ | \ s _ { 0 } = s { \bigr ) } r _ { \ell , k } ^ { * } ,\tag{7.9}
$$

where P <sub>(t)</sub>(absorbed in $C _ { \ell , k } \mid s _ { 0 } = s )$ is the probability of being absorbed into $C _ { \ell , k }$ when starting from $s _ { 0 }$ under $\pi ^ { ( t ) }$

In the aggregated OSP, all rewards are zero except for stopping actions in states $\bar { s } _ { \ell , k }$ which receive reward $r _ { \ell , k } ^ { * }$ . Thus, for any $\bar { s } \in \bar { \cal S }$

$$
\bar { g } _ { \bar { \pi } ^ { ( t ) } } ( \bar { s } ) = \sum _ { ( \ell , k ) } \mathbb { P } _ { \bar { \pi } ^ { ( t ) } } ( \operatorname { s t o p } \mathrm { ~ i n ~ } \bar { s } _ { \ell , k } \mid \bar { s } _ { 0 } = \bar { s } ) r _ { \ell , k } ^ { * } ,\tag{7.10}
$$

where $\mathbb { P } _ { \bar { \pi } ^ { ( t ) } } \left( \mathrm { s t o p } \right.$ in $\bar { s } _ { \ell , k } \mid \bar { s } _ { 0 } = \bar { s } )$ is the probability of stopping in $\bar { s } _ { \ell , k }$ when starting from $\bar { s } _ { 0 }$ under $\bar { \pi } ^ { ( t ) }$

By Lemma 7.6, for any $s _ { 0 } \in S$ and all $( \ell , k )$ ，

$$
\mathbb { P } _ { \pi ^ { ( t ) } } ( \mathrm { a b s o r b e d ~ i n ~ } C _ { \ell , k } \mid s _ { 0 } ) = \mathbb { P } _ { \bar { \pi } ^ { ( t ) } } ( \mathrm { s t o p ~ i n ~ } \bar { s } _ { \ell , k } \mid \bar { s } _ { 0 } = \phi ( s _ { 0 } ) ) .
$$

Substituting into equations (7.9) – (7.10), we have that for all $s \in S$

$$
g _ { \pi ^ { ( t ) } } ( s ) = \left\{ \begin{array} { l l } { \bar { g } _ { \bar { \pi } ^ { ( t ) } } ( s ) , } & { s \in \mathcal { T } , } \\ { \bar { g } _ { \bar { \pi } ^ { ( t ) } } ( \bar { s } _ { \ell , k } ) , } & { s \in C _ { \ell , k } , } \end{array} \right. \quad \forall t \geq T .
$$

Hence, $\pi ^ { ( t ) }$ attains the optimal gain vector $g ^ { * }$ for all $t \geq T$

The theorem does not require the policy sequence $\{ \pi ^ { ( t ) } \}$ to stabilize. It asserts only that, after an almost surely finite time, every policy produced by the algorithm is gain-optimal.

## 8 Convergence proofs for Algorithms 2 and 3

In this section, we prove the convergence theorems for Algorithms 2 and 3. The proof of Theorem 4.1 for Algorithm 2 is given in Section 8.2. The proofs of Lemma 5.2 and Theorem 5.1 for Algorithm 3 are given in Section 8.3.

To begin with, we restate the information structure condition (C2) of Algorithm 2 in a more formal manner and introduce several conventions to simplify the presentation of the main convergence proof in the next subsection.

## 8.1 Formal statement of assumptions and conventions

We first fix a few conventions that impose no restriction on the algorithm and are adopted without loss of generality. Although the algorithm starts at time $\bar { t } ,$ we extend the definitions of $Q _ { t }$ and $Y _ { t }$ to all $t \geq 0$ by setting $Q _ { t }$ to a fixed value and $Y _ { t } = \theta$ for $t < \bar { t } .$ We define $\alpha _ { t } ( s , a )$ for all state-action pairs and all $t \geq 0$ , with $\alpha _ { t } ( s , a ) = 0$ whenever $( s , a ) \notin Y _ { t }$ . Under this convention, $Y _ { t } = \{ ( s , a ) : \alpha _ { t } ( s , a ) \neq 0 \}$ , so specifying α<sub>t</sub> uniquely determines $Y _ { t } .$ . Finally, we define the transition data $( R _ { t } ^ { s a } , S _ { t } ^ { s a } )$ for all state-action pairs $( s , a )$ and all $t \geq 0$ , even though some of them are not used in the algorithm.

Assumption 8.1 below summarizes a set of non-restrictive regularity conditions. These conditions set up a basic measurability structure for treating the randomized algorithm probabilistically and reafirm its non-anticipatory nature.

Throughout, all spaces are assumed to be standard Borel spaces $( \mathrm { e . g . , [ 0 , 1 ] ) }$ with their Borel σ-algebras, and all measurability is understood in the Borel sense. While the conditions could be formulated more generally, we restrict attention to Borel spaces and Borel measurability for clarity and to avoid unnecessary technicalities.

Assumption 8.1 (Measurability, non-anticipation, and correct conditional distributions).

• Initialization: The starting time t<sup>¯</sup> of Algorithm 2 is an a.s. finite stopping time relative to the histories of Algorithm 1. The parameters $( \epsilon , \underline { { \delta } } , \bar { \delta } )$ and the initial $Q _ { \bar { t } }$ are measurable functions of the history of Algorithm 1 up to time t<sup>¯</sup>.

• Stochastic evolution: All other random elements in Algorithm 2, namely $\alpha _ { t }$ and $( R _ { t } ^ { s a } , S _ { t } ^ { s a } ) _ { ( s , a ) \in \mathcal { A } } ,$ admit measurable and non-anticipatory realizations driven by exogenous noises. Specifically, there exist i.i.d. exogenous noise sequences $\{ w _ { t } \} _ { t \ge 0 }$ and $\{ w _ { t } ^ { \prime } \} _ { t \ge 0 }$ , taking values in standard Borel spaces, such that:

(i) For each $t \geq 0 , \alpha _ { t } = \Psi _ { t } \big ( w _ { t } , ( g _ { n } , v _ { n } , Q _ { n } ) _ { n \leq t } , ( \alpha _ { n } ) _ { n < t } , ( R _ { n } ^ { s a } , S _ { n } ^ { s a } ) _ { ( s , a ) \in A , n \leq t } \big )$ , for some Borelmeasurable map $\Psi _ { t }$

(ii) For each $t \geq 0 , ( R _ { t } ^ { s a } , S _ { t } ^ { s a } ) _ { ( s , a ) \in A } = \Psi _ { t } ^ { \prime } \big ( w _ { t } ^ { \prime } , ( g _ { n } , v _ { n } , Q _ { n } , \alpha _ { n } ) _ { n < t } , ( R _ { n } ^ { s a } , S _ { n } ^ { s a } ) _ { ( s , a ) \in A , n < t } \big )$ , for some Borel-measurable map $\Psi _ { t } ^ { \prime }$

Moreover, the following conditional distribution requirement holds:

(iii) For each $t \geq 0$ and each $( s , a ) \in A _ { \cdot }$ , for any given realization of the past outcomes, the distribution of $( R _ { t } ^ { s a } , S _ { t } ^ { s a } )$ induced by the fresh noise $w _ { t } ^ { \prime }$ through the map $\Psi _ { t } ^ { \prime }$ coincides with the reward and transition distribution of the MDP at state s under action a.

Remark 8.1. In Assumption $8 . 1 ( \mathrm { i i } , \mathrm { i i i } )$ , we do not require the transition data $( R _ { t } ^ { s a } , S _ { t } ^ { s a } ) _ { ( s , a ) \in \mathcal { A } }$ to be mutually independent across diferent state-action pairs $( s , a )$ . These conditions accommodate situations in which one exploits the MDP structure to improve the eficiency of $\mathrm { Q }$ -learning by using a shared random outcome to generate transition data for multiple state-action pairs involved in the Q-learning updates.

We also restate condition (C3) of Algorithm 2 under the conventions introduced above:

Assumption 8.2 (Adequate transition data and stepsize conditions). The following hold $a . s . \cdot$

(i) Any state-action pair that occurs in $\tilde { \Gamma } _ { t }$ infinitely often also occurs in $Y _ { t }$ infinitely often.

(ii) For any (s, a) that appears infinitely often in $Y _ { t } ,$ the stepsizes satisfy

$$
\sum _ { t } \alpha _ { t } ( s , a ) = \infty , \quad \quad \sum _ { t } \alpha _ { t } ^ { 2 } ( s , a ) < \infty .
$$

## 8.2 Algorithm 2: proof of Theorem 4.1

Lemma 4.2 establishes finite-time stabilization of the sets constructed in Algorithm 2. Thus, the limiting sets $\tilde { S } ^ { i } \left( i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \} \right)$ , A<sup>˜</sup>, and $\tilde { \Gamma }$ appearing in Theorem 4.1, as well as the associated partition

$\mathcal { C } = \tilde { \mathcal { C } } ^ { \mathrm { r } } \cup \tilde { \mathcal { C } } ^ { \mathrm { e } } \cup \tilde { \mathcal { C } } ^ { \mathrm { b } }$ , are all well-defined.

Theorem 4.1 concerns the convergence of the Q-learning iterates in Algorithm 2 and properties of their limits. We prove it in three steps:

(i) First, we extend existing stability and convergence results for Q-learning in SSP-type totalreward MDPs ([29, 36] and [5, Sec. 4.3]).

(ii) Using this extension, we establish the convergence of a Q-learning algorithm corresponding to a limiting form of Algorithm 2 that solves a given SSP-type MDP.

(iii) We then tackle Algorithm 2 in its original form, establish the a.s. convergence of its Q-learning iterates to solutions of random SSP-type MDPs induced by the outputs of Algorithm 1.

## 8.2.1 Step 1: extending Q-learning results for SSP problems

We now begin the proof by presenting an extended convergence theorem for Q-learning in SSP problems, building on prior work. Let $\hat { \mathcal { M } } : = \{ \hat { S } \cup \{ \Delta \} , \hat { A } , \hat { p } , \hat { r } \}$ by an ${ \mathrm { S S P - t y p e } }$ total-reward MDP as defined in Section 2.2. We consider Q-learning iterates for solving $\hat { \mathcal { M } }$ , generated from a given initial condition $\hat { Q } _ { 0 }$ and for all $( s , a ) \in { \hat { A } }$ , by

$$
\begin{array} { r l } & { \hat { Q } _ { t + 1 } ( s , a ) = \hat { Q } _ { t } ( s , a ) + \hat { \alpha } _ { t } ( s , a ) \Big ( \hat { r } ( s , a ) + \omega _ { t + 1 } ( s , a ) + \delta _ { t + 1 } ( s , a ) } \\ & { \qquad + \mathbb { 1 } \{ \hat { S } _ { t + 1 } ^ { s a } \neq \Delta \} \cdot \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } ( \hat { S } _ { t + 1 } ^ { s a } ) } \hat { Q } _ { t } ( \hat { S } _ { t + 1 } ^ { s a } , a ^ { \prime } ) - \hat { Q } _ { t } ( s , a ) \Big ) . } \end{array}\tag{8.1}
$$

Here, the term $\hat { r } ( s , a ) + \omega _ { t + 1 } ( s , a ) + \delta _ { t + 1 } ( s , a )$ represents a random reward deviating from the expected reward ${ \hat { r } } ( s , a )$ , and the stepsizes $\hat { \alpha } _ { t } ( s , a )$ at iteration t implicitly determine which components of $\hat { Q } _ { t }$ are updated: if $\hat { \alpha } _ { t } ( s , a ) = 0$ , then $\hat { Q } _ { t + 1 } ( s , a ) = \hat { Q } _ { t } ( s , a )$

Moreover, there is an associated increasing sequence of σ-algebras $\mathcal { F } _ { t }$ , together with which, the random variables involved in (8.1) satisfy the properties listed below:

(A1) $\mathcal { F } _ { t } \supset \sigma \big ( \hat { Q } _ { 0 } , \hat { \alpha } _ { k } ( s , a ) , \omega _ { k } ( s , a ) , \delta _ { k } ( s , a ) , \hat { S } _ { k } ^ { s a } ; k \le t , ( s , a ) \in \hat { \mathcal { A } } \big )$

For each $( s , a ) \in { \hat { A } } \colon$

(A2) $\textstyle \sum _ { t } { \hat { \alpha } } _ { t } ( s , a ) = \infty$ and $\textstyle \sum _ { t } { \hat { \alpha } } _ { t } ^ { 2 } ( s , a ) < \infty { \mathrm { ~ a . s . } }$

(A3) For all $t \geq 0 , \mathbb { E } [ \omega _ { t + 1 } ( s , a ) \mid \mathcal { F } _ { t } ] = 0$ and $\mathbb { E } \left[ \omega _ { t + 1 } ^ { 2 } ( s , a ) \mid \mathcal { F } _ { t } \right] < c \mathrm { ~ a . s ~ }$ ., where c is a deterministic constant.

(A4) As $t \to \infty , \delta _ { t + 1 } ( s , a ) \to 0 \mathrm { ~ a . s ~ }$

(A5) Conditional on $\mathcal { F } _ { t } , \hat { S } _ { t + 1 } ^ { s a }$ follows the transition probability distribution $\hat { p } ( \cdot \mid s , a )$ of $\hat { \mathcal { M } } .$

Iteration (8.1) difers from the standard Q-learning iteration for $\hat { \mathcal { M } }$ only in that the reward includes an additional term $\delta _ { t + 1 } ( s , a )$ , which vanishes by (A4) but need not have conditional zero mean given $\mathcal { F } _ { t } .$ . Because of these biased diminishing noise terms $\delta _ { t + 1 }$ , we extend prior results in [29, 36] on the boundedness and convergence of Q-learning in SSP-type MDPs to establish convergence of (8.1) (see also [5, Sec. 4.3] for a presentation of the results in [29] in a more general formulation, albeit without communication delays).

We give the extended theorem below, along with an outline of the proof highlighting the minor modifications needed to make the prior results applicable here. We omit the full proof to avoid repeating the lengthy and involved arguments in those works.

Proposition 8.1. For the SSP-type MDP $\hat { \mathcal { M } }$ , the sequence $\{ \hat { Q } _ { t } \}$ generated $b y$ (8.1) under (A1)– (A5) converges a.s. to $\hat { Q } ^ { * }$ , the optimal action-value function of M<sup>ˆ</sup> .

Proof (outline). We first show how to modify prior results to establish convergence of $\{ \hat { Q } _ { t } \}$ , assuming its a.s. boundedness. We apply the convergence proof of [5, Prop. 4.6] (which assumes a.s.

boundedness), together with the following extension to account for the additional terms $\delta _ { t + 1 }$ not considered in the original proof. Define

$$
Z _ { t + 1 } ( s , a ) = ( 1 - \hat { \alpha } _ { t } ( s , a ) ) Z _ { t } ( s , a ) + \hat { \alpha } _ { t } ( s , a ) \delta _ { t + 1 } ( s , a ) ,\tag{8.2}
$$

with $Z _ { 0 } ( \cdot ) \equiv 0$ . Since $\delta _ { t }  0 \mathrm { \ a . s . \ b y \ ( A 4 ) }$ and $\left\{ \hat { \alpha } _ { t } \right\}$ satisfies (A3), it follows that $Z _ { t } \to 0 \ \mathrm { a . s }$ . We follow the same proof steps as in [5, Prop. 4.6, pp. 168–171] until near the end. There, we modify [5, Lem. 4.7, p. 170] by adding the term $Z _ { t } ( s , a )$ to its upper bound on $\hat { Q } _ { t + 1 } ( s , a )$ , using the same reasoning as in the original proof. Using this modified lemma and the fact that $Z _ { t } \to 0 \mathrm { a . s . }$ ., the final step [5, Lem. 4.8, pp. 170–171] continues to hold for $\{ \hat { Q } _ { t } \}$ , completing the convergence proof under the boundedness condition.

For the boundedness of $\{ \hat { Q } _ { t } \}$ , we explain how the boundedness proof in [36] extends to our setting. Following the analysis in [36], we show separately that $\{ \hat { Q } _ { t } \}$ is bounded above a.s. and bounded below a.s. The two parts rely on substantially diferent arguments.

For boundedness from above, we apply the reasoning in the proof of [36, Prop. 3.1], where we bound $\{ \hat { Q } _ { t } \}$ from above by an auxiliary sequence of Q-learning iterates constructed using a proper policy and argue that the auxiliary sequence is bounded a.s. The proof of [36, Prop. 3.1] carries over to the present setting once we note that, under a proper policy, the mapping underlying Q-learning is a contraction w.r.t. a weighted sup-norm. Consequently, by [5, Prop. 4.5, pp. 157–158], even in the presence of the additional diminishing biased noise terms $\{ \delta _ { t } \}$ , the auxiliary iterates remain bounded a.s. Therefore, $\{ \hat { Q } _ { t } \}$ is bounded above a.s.

For boundedness from below, the proof of [36, Prop. 3.3, Section 3.3] carries over to the present setting. While the proof involves the construction of auxiliary processes and other arguments, which we do not repeat here due to their complexity and length, it does not depend critically on the particular form of the random one-stage rewards appearing in Q-learning. Instead, it only requires the following estimates $\textstyle \left\{ { \hat { r } } _ { t } ( s , a ) \right\}$ , defined by the random rewards, to converge to the expected one-stage reward ${ \hat { r } } ( s , a )$ for each (s, a):

$$
\hat { r } _ { t + 1 } ( s , a ) = ( 1 - \hat { \alpha } _ { t } ( s , a ) ) \hat { r } _ { t } ( s , a ) + \hat { \alpha } _ { t } ( s , a ) ( \hat { r } ( s , a ) + \omega _ { t + 1 } ( s , a ) + \delta _ { t + 1 } ( s , a ) ) .
$$

By standard arguments under $( \mathrm { A 2 } ) – ( \mathrm { A 4 } ) , ^ { 1 } \ \hat { r } _ { t + 1 } ( s , a )  \hat { r } ( s , a )$ holds a.s. despite the additional diminishing noise terms $\delta _ { t + 1 }$ . Consequently, [36, Prop. 3.3] extends to our setting and implies that $\{ \hat { Q } _ { t } \}$ is bounded below a.s. Algorithm 2 Combining the two parts yields the a.s. boundedness of {Q<sup>ˆ</sup><sub>t</sub>}. □

## 8.2.2 Step 2: convergence of a limiting form of Algorithm 2

We now return to the original MDP M and consider a limiting form of Algorithm 2. In particular, fix arbitrary sets satisfying the limiting properties (4.25)–(4.28) in Lemma 4.2: a partition of C into $\tilde { \mathcal { C } } ^ { \mathrm { r } } , \tilde { \mathcal { C } } ^ { \mathrm { e } } , \tilde { \mathcal { C } } ^ { \mathrm { b } }$ , the associated state sets $\begin{array} { r } { \tilde { S } ^ { i } : = \bigcup _ { C \in \tilde { \mathcal { C } } ^ { i } } C . } \end{array}$ , and action constraint set A<sup>˜</sup>. Define

$$
\tilde { \Gamma } : = \{ ( s , a ) \in \tilde { \mathcal { A } } \mid s \notin \tilde { S } ^ { \mathrm { r } } \} .
$$

Let $\{ g _ { t } \} \subset \mathbb { R } ^ { | S | }$ and $\{ v _ { t } \} \subset \mathbb { R } ^ { | \tilde { S } ^ { \mathrm { r } } | }$ be two convergent sequences with $g _ { t } \to g ^ { * }$ and $v _ { t } \to v _ { * }$ . All the objects just introduced are treated as given and nonrandom—an assumption that we will relax later.

Given initial $\hat { Q } _ { 0 }$ , consider Q-learning iterates defined for all $( s , a ) \in \tilde { \Gamma }$ by

$$
\begin{array} { r l r } {  { \hat { Q } _ { t + 1 } ( s , a ) = \hat { Q } _ { t } ( s , a ) + \hat { \alpha } _ { t } ( s , a ) \Big ( R _ { t + 1 } ^ { s a } - g _ { t } ( s ) - \epsilon \mathbb { 1 } \big \{ s \in \tilde { \mathcal { S } } ^ { \mathrm { b } } \big \} + \mathbb { 1 } \big \{ S _ { t + 1 } ^ { s a } \in \tilde { \mathcal { S } } ^ { \mathrm { r } } \big \} \cdot v _ { t } \big ( S _ { t + 1 } ^ { s a } \big ) } } \\ & { } & { + \mathbb { 1 } \big \{ S _ { t + 1 } ^ { s a } \notin \tilde { \mathcal { S } } ^ { \mathrm { r } } \big \} \cdot \operatorname* { m a x } _ { a ^ { \prime } \in \tilde { \mathcal { A } } ( S _ { t + 1 } ^ { s a } ) } \hat { Q } _ { t } \big ( S _ { t + 1 } ^ { s a } , a ^ { \prime } \big ) - \hat { Q } _ { t } ( s , a ) \Big ) , } \end{array}\tag{8.3}
$$

together with these associated conditions:

(i) There is an increasing sequence of σ-algebras $\mathcal { F } _ { t }$ with

$$
\mathcal { F } _ { t } \supset \sigma \big ( \hat { Q } _ { 0 } , \hat { \alpha } _ { k } ( s , a ) , R _ { k } ^ { s a } , S _ { k } ^ { s a } ; k \le t , ( s , a ) \in \tilde { \Gamma } \big ) .
$$

(ii) The stepsizes $\hat { \alpha } _ { t } ( s , a )$ satisfy (A2).

(iii) Conditional on $\mathcal { F } _ { t } , ~ R _ { t + 1 } ^ { s a }$ and $S _ { t + 1 } ^ { s a }$ are distributed according to the reward and transition distributions of M at state s under action a.

The next lemma establishes the convergence of $\{ \hat { Q } _ { t } \}$ to a limiting vector $\hat { Q } ^ { * }$ satisfying the same equation as (4.29) in Theorem 4.1, which concerns the limits of the Q-learning iterates of Algorithm 2.

Lemma 8.1. In the above setting, the sequence $\{ \hat { Q } _ { t } \}$ generated by (8.3) converges a.s. to the unique solution $\hat { Q } ^ { * }$ of the following equation: for all $( s , a ) \in \ddot { \Gamma }$

$$
\hat { Q } ^ { * } ( s , a ) = r ( s , a ) - g ^ { * } ( s ) - \epsilon { \bf 1 } _ { \mathcal { S } ^ { \bf b } } ( s ) + \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } ^ { \tau } } p ( s ^ { \prime } \mid s , a ) v _ { * } ( s ^ { \prime } ) + \sum _ { s ^ { \prime } \in \mathcal { S } \backslash \tilde { \mathcal { S } } ^ { \tau } } p ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in \tilde { \mathcal { A } } ( s ^ { \prime } ) } \hat { Q } ^ { * } ( s ^ { \prime } , a ^ { \prime } ) .\tag{8.4}
$$

Moreover, this equation is the optimality equation of an SSP-type total-reward MDP defined on $\tilde { \Gamma } _ { i }$ for which $\hat { Q } ^ { * }$ is the optimal action-value function.

Proof. We view $\{ \hat { Q } _ { t } \}$ as Q-learning iterates for an SSP-type MDP $\hat { \mathcal { M } } = \{ \hat { \mathcal { S } } \cup \{ \Delta \} , \hat { \mathcal { A } } , \hat { p } , \hat { r } \}$ defined below. This MDP $\hat { \mathcal { M } }$ is similar to the MDP $\mathcal { M } ^ { \epsilon }$ introduced before Lemma 4.1 for the idealized algorithm in Section $4 . 1 ,$ , except that $\hat { \mathcal { M } }$ is defined using $\tilde { S } ^ { \mathrm { r } } , \tilde { S } ^ { \mathrm { b } } , \tilde { \mathcal { C } } ^ { \mathrm { e } } , \tilde { \mathcal { C } } ^ { \mathrm { b } }$ , and A<sup>˜</sup> in place of ${ \mathcal { S } } ^ { \mathrm { r } } , { \mathcal { S } } ^ { \mathrm { b } }$ ${ \mathcal { C } } ^ { \mathrm { { e } } } , { \mathcal { C } } ^ { \mathrm { { b } } }$ , and $A _ { * } ,$ respectively. In particular, we aggregate the states in ${ \tilde { S } } ^ { \mathrm { r } }$ into a special termination state $\Delta ,$ , and take $\hat { \hat { S } } = \mathcal { S } \setminus \widetilde { S } ^ { \mathrm { r } }$ as the efective state space. Let $\hat { \mathcal { A } } ( s ) = \tilde { \mathcal { A } } ( s )$ for $\boldsymbol { s } \in \hat { { \mathcal { S } } } ;$ thus, $\hat { A } = \tilde { \Gamma }$ For each feasible state-action pair $( s , a ) \in \tilde { \Gamma }$ , the transition probabilities $\hat { p } ( \cdot \mid s , a )$ are given by

$$
\hat { p } ( s ^ { \prime } \mid s , a ) = p ( s ^ { \prime } \mid s , a ) , \quad s ^ { \prime } \in \hat { \mathcal { S } } , \qquad \hat { p } ( \Delta \mid s , a ) = \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } ^ { \mathrm { r } } } p ( s ^ { \prime } \mid s , a ) ,\tag{8.5}
$$

and the expected one-stage reward ${ \hat { r } } ( s , a )$ is given by

$$
\hat { r } ( s , a ) = r ( s , a ) - g ^ { * } ( s ) - \epsilon \mathbf { 1 } _ { \tilde { \mathcal { S } } ^ { \mathrm { b } } } ( s ) + \sum _ { s ^ { \prime } \in \tilde { \mathcal { S } } ^ { \mathrm { r } } } p ( s ^ { \prime } \mid s , a ) v _ { * } ( s ^ { \prime } ) .\tag{8.6}
$$

(Compare (4.15)–(4.16) for M<sup>ϵ</sup>.)

That M<sup>ˆ</sup> is of SSP type follows from the same arguments as in the proof of Lemma 4.1, with the set replacements noted above. In particular, by (4.28), for each $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { e } } \cup \tilde { \mathcal { C } } ^ { \mathrm { b } }$ , there exists at least one state $s _ { \ell , k } \in C _ { \ell , k }$ with $\tilde { \mathcal { A } } ( s _ { \ell , k } ) \cap \mathcal { E } ( s _ { \ell , k } ) \neq \emptyset$ . Hence a proper policy for $\hat { \mathcal { M } }$ can be constructed as in the proof of Lemma 4.1. Since $\tilde { \mathcal { C } } ^ { \mathrm { e } } \subset \mathcal { C } ^ { \mathrm { e } }$ and $\tilde { \mathcal { C } } ^ { \mathrm { b } } \supset \mathcal { C } ^ { \mathrm { b } }$ by (4.26), the same reasoning as in the second half of the proof of Lemma 4.1 shows that any improper stationary policy in $\Pi _ { \hat { \mathcal { A } } }$ has expected total reward $- \infty$ in $\hat { \mathcal { M } }$ on each recurrent class it induces within the efective state space.

We can express the iterates $\hat { Q } _ { t }$ from (8.3) in the general form (8.1) of Q-learning for $\hat { \mathcal { M } }$ as follows. Take the σ-algebras associated with (8.1) to be the σ-algebras $\mathcal { F } _ { t }$ associated with iterates from (8.3). Define the noise terms $\omega _ { t + 1 } ( s , a )$ and $\delta _ { t + 1 } ( s , a )$ in (8.1) as

$$
\omega _ { t + 1 } ( s , a ) = R _ { t + 1 } ^ { s a } - r ( s , a ) + \mathbb { 1 } \{ S _ { t + 1 } ^ { s a } \in \tilde { S } ^ { \mathfrak { r } } \} \cdot v _ { * } ( S _ { t + 1 } ^ { s a } ) - \sum _ { s ^ { \prime } \in \tilde { S } ^ { \mathfrak { r } } } p ( s ^ { \prime } \mid s , a ) v _ { * } ( s ^ { \prime } ) ,\tag{8.7}
$$

$$
\delta _ { t + 1 } ( s , a ) = g ^ { * } ( s ) - g _ { t } ( s ) + \mathbb { 1 } \{ S _ { t + 1 } ^ { s a } \in \tilde { \mathcal { S } } ^ { \mathrm { r } } \} \cdot \bigl ( v _ { t } ( S _ { t + 1 } ^ { s a } ) - v _ { * } ( S _ { t + 1 } ^ { s a } ) \bigr ) .\tag{8.8}
$$

Recall that $v _ { * }$ lies in a given bounded set, $g _ { t } \to g ^ { * }$ , and $v _ { t }  v _ { * }$ . Consequently, $\omega _ { t + 1 }$ and $\delta _ { t + 1 }$ satisfy (A3) and (A4). Define the random states $\hat { S } _ { t + 1 } ^ { s a }$ in (8.1) as

$$
\hat { S } _ { t + 1 } ^ { s a } = \left\{ \begin{array} { l l } { S _ { t + 1 } ^ { s a } } & { \mathrm { i f ~ } S _ { t + 1 } ^ { s a } \notin \tilde { S } ^ { \mathrm { r } } ; } \\ { \Delta } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8.9}
$$

With this definition, (A5) holds. The conditions in Prop. 8.1 are therefore all satisfied.

Then, by Prop. 8.1, {Q<sup>ˆ</sup><sub>t</sub>} converges a.s. to the optimal action-value vector $\hat { Q } ^ { * }$ of $\hat { \mathcal { M } }$ . Finally, by Theorem 2.1 for SSP-type MDPs, $\hat { Q } ^ { * }$ is the unique solution to the action-value optimality equation (8.4) for $\hat { \mathcal { M } } .$ □

## 8.2.3 Step 3: convergence of Algorithm 2 in original form

We apply Lemma 8.1 to establish Theorem 4.1. However, because the outputs of Algorithm 1 are random, the parameter ϵ and the limiting sets $\tilde { \mathcal { C } } ^ { i } , \tilde { S } ^ { i }$ , and A<sup>˜</sup> in Theorem 4.1 are also random, preventing a direct application of Lemma 8.1. Instead, we start with the joint probability measure for Algorithm 1 and Algorithm 2, and then work with the associated conditional distributions in explicit form, as shown below.

First, we define the probability space $\left( \Omega , \Sigma _ { \Omega } , \mathcal { P } \right)$ for analyzing Algorithm 2. Let $( \Xi , \Sigma _ { \Xi } , \mathcal { P } ^ { b } )$ be the probability space associated with Algorithm 1. By Assumption 8.1, at initialization of Algorithm 2, the starting time $\bar { t } ,$ the parameters $( \epsilon , \underline { { \delta } } , \bar { \delta } )$ , and the initial $Q _ { \bar { t } }$ are measurable functions of $\xi \in \Xi$ . We take the sample space for Algorithm 2 to be $\Omega : = \Xi \times \mathcal { Z }$ , where $\mathcal { Z }$ consists of sequences $\zeta : = ( \zeta _ { t } ) _ { t \geq 0 }$ , with each $\zeta _ { t }$ representing a realization of the collection of random variables

$$
\{ Q _ { t } ( s , a ) , \alpha _ { t } ( s , a ) , ( R _ { t } ^ { s a } , S _ { t } ^ { s a } ) \mid ( s , a ) \in \mathcal { A } \} .
$$

We write $\boldsymbol { \omega } = ( \xi , \zeta )$ for a generic element of Ω. We equip $\mathcal { Z }$ with its Borel σ-algebra and take the σ-algebra $\Sigma _ { \Omega }$ on Ω to be the corresponding product σ-algebra. By construction, under the regularity conditions in Assumption 8.1(i,ii), Algorithm 2 induces a stochastic kernel $\kappa ( \xi , d \zeta )$ on $\mathcal { Z }$ ${ \mathrm { g i v e n } } \Sigma$ . Together with Algorithm 1, this defines a unique probability measure $\mathcal { P }$ on $\left( \Omega , \Sigma _ { \Omega } \right)$ by

$$
\mathcal { P } ( d \omega ) = \mathcal { P } ^ { b } ( d \xi ) \mathcal { K } ( \xi , d \zeta ) .\tag{8.10}
$$

By the sequential and non-anticipatory nature of Algorithm 2, under the regularity conditions in Assumption $8 . 1 ( \mathrm { i , i i } ) , \mathcal { K } ( \xi , d \zeta )$ admits an explicit representation as a composition of an infinite sequence of stochastic kernels, corresponding to the successive randomized updates of the algorithm based on the history up to each iteration. We now introduce notation to express this representation, for each $t \geq 0 ,$ , as a composition of two stochastic kernels corresponding to the iterations up to time t and after time $t ,$ respectively. These representations will be used in the proof below.

For each $t \geq 0 .$ , let $\mathcal { Z } _ { t }$ denote the space of $\zeta _ { t }$ , and let $\mathcal { Z } _ { 0 : t }$ and $\mathcal { Z } _ { t + 1 : \infty }$ denote the spaces of $\zeta _ { 0 : t } : = ( \zeta _ { n } ) _ { n \leq t }$ and $\zeta _ { t + 1 : \infty } : = ( \zeta _ { n } ) _ { n \geq t + 1 }$ , respectively (each equipped with its Borel σ-algebra). Under the regularity conditions stated in Assumption 8.1(i,ii), Algorithm 2 induces, for each $t \geq 0$ a stochastic kernel $\mathcal { K } _ { t } \big ( ( \xi , \zeta _ { 0 : t - 1 } ) , d \zeta _ { t } \big )$ on $\mathcal { Z } _ { t }$ given $\Xi \times \mathcal { Z } _ { 0 : t - 1 }$ . Consequently, for each $t \geq 0 , \mathcal { K } ( \xi , d \zeta )$ can be written in the product form

$$
\begin{array} { r } { \mathcal { K } ( \xi , d \zeta ) = K _ { 0 : t } \left( \xi , d \zeta _ { 0 : t } \right) \mathcal { K } _ { t + 1 : \infty } \big ( ( \xi , \zeta _ { 0 : t } ) , d \zeta _ { t + 1 : \infty } \big ) , } \end{array}\tag{8.11}
$$

where $\kappa _ { 0 : t }$ <sub>t</sub> is the stochastic kernel on $\mathcal { Z } _ { 0 : t }$ given Ξ obtained by composing $\mathcal { K } _ { 0 } , \mathcal { K } _ { 1 } , \ldots \mathcal { K } _ { t }$ , and $\scriptstyle { \boldsymbol { \mathcal { K } } } _ { t + 1 : \infty }$ is the stochastic kernel on $\mathcal { Z } _ { t + 1 : \infty }$ given $\Xi \times \mathcal { Z } _ { 0 : t }$ obtained by composing $\kappa _ { t + 1 } , \kappa _ { t + 2 } , . . . .$

Furthermore, for a given state-action pair $( s , a )$ , by Assumption 8.1(iii), for each $t \geq 0$ and each $( \xi , \zeta _ { 0 : t } ) \in \Xi \times \mathcal { Z } _ { 0 : t }$ , the marginal distribution of $( R _ { t + 1 } ^ { s a } , S _ { t + 1 } ^ { s a } )$ under the probability measure

$\mathcal { K } _ { t + 1 } \big ( ( \xi , \zeta _ { 0 : t } ) , d \zeta _ { t + 1 } \big )$ on $\mathcal { Z } _ { t + 1 }$ coincides with the joint distribution of the reward and next state in the MDP at state s under action $^ { a , }$ denoted by

$$
\mathcal { P } _ { o } ^ { s a } \big ( d ( r _ { t + 1 } ^ { s a } , s _ { t + 1 } ^ { s a } ) \big ) .\tag{8.12}
$$

With these preparations, we are now ready to prove Theorem 4.1.

Proof of Theorem 4.1. There are finitely many distinct configurations for the collection of sets $\{ \tilde { \mathcal { A } } , \tilde { \mathcal { C } } ^ { i } , \tilde { \mathcal { S } } ^ { i } | i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \} \}$ satisfying (4.25)–(4.26). Denote the set of all such configurations by ${ \mathcal { I } } .$ The smallest random time τ after which the sets $\tilde { \mathcal { A } } _ { t } , \tilde { \mathcal { C } } _ { i , t }$ and $\tilde { \cal S } _ { i , t }$ remain unchanged for all $i \in \{ \mathrm { r } , \mathrm { e } , \mathrm { b } \}$ may be ∞ or a finite integer $k \geq 0$ . These random outputs, as well as the other random outputs of Algorithm 1 are measurable functions on the sample space $\Xi ,$ , and this dependence on ξ will be made explicit below by writing, for example, $\tilde { \mathcal { A } }$ as $\tilde { \mathcal { A } } ( \xi )$ and $\tilde { \mathcal { A } } ( s )$ as $\tilde { \cal A } ( s ; \xi )$

For each $j \in \mathcal I$ and $k \geq 0 .$ , let $\Xi _ { j , k }$ denote the subset of all sample points $\xi \in \Xi$ for which:

(i) $\tau ( \xi ) = k$ and the limiting sets $\tilde { \mathcal { C } } ^ { i } ( \xi ) , \tilde { \mathcal { S } } ^ { i } ( \xi )$ , and $\tilde { \mathcal { A } } ( \xi )$ match the jth configuration in $\mathcal { T } ;$

(ii) as $t \to \infty , g _ { t } ( \xi ) \to g ^ { * }$ , and $v _ { t } ( \xi )$ converges to a limit $v _ { * } ( \xi )$ such that the restriction of $v _ { * } ( \xi )$ to each $C _ { \ell , k } \in \mathcal { C }$ solves the AOE of the sub-MDP $\mathcal { M } _ { \ell , k }$

Our conditions on Algorithm 1, together with Lemma 4.2, ensure that

$$
\begin{array} { r } { \sum _ { j \in \mathcal { I } } \sum _ { k \geq 0 } \mathcal { P } ^ { b } ( \Xi _ { j , k } ) = 1 . } \end{array}\tag{8.13}
$$

$\operatorname { L e t } \hat { \Xi } : = \cup _ { j \in \mathcal { T } } \cup _ { k \geq 0 } \Xi _ { j , k }$ . For $\xi \in \hat { \Xi } .$ , with $\tilde { \Gamma } ( \xi ) : = \big \{ ( s , a ) \in \tilde { \mathcal { A } } ( \xi ) \mid s \notin \tilde { \mathcal { S } } ^ { \mathrm { r } } ( \xi ) \big \}$ , define the event

$$
\begin{array} { r } { E _ { \alpha } : = \left\{ \omega = ( \xi , \zeta ) \in \hat { \Xi } \times \mathcal { Z } \Big | \sum _ { t } \alpha _ { t } ( s , a ; \omega ) = \infty , \sum _ { t } \alpha _ { t } ^ { 2 } ( s , a ; \omega ) < \infty , \forall ( s , a ) \in \tilde { \Gamma } ( \xi ) \right\} } \end{array}
$$

(where we make explicit the dependence of $\alpha _ { t }$ on the sample point ω). By (8.13), Lemma 4.2, and Assumption $8 . 2$ on Algorithm 2, we have $\mathcal { P } ( E _ { \alpha } ) = 1$ , or equivalently, using the stochastic kernel representations for $\mathcal { P }$ discussed earlier [see (8.10), (8.11)],

$$
\sum _ { j \in \mathcal { T } } \sum _ { k \geq 0 } \int _ { \Xi _ { j , k } } \mathcal { P } ^ { b } ( d \xi ) \int _ { \mathcal { Z } _ { 0 , k } } K _ { 0 : k } \left( \xi , d \zeta _ { 0 : k } \right) \int _ { \mathcal { Z } _ { k + 1 : \infty } } \mathbf { 1 } _ { E _ { \alpha } } \left( \xi , \zeta _ { 0 : k } , \zeta _ { k + 1 : \infty } \right) \mathcal { K } _ { k + 1 : \infty } \left( \left( \xi , \zeta _ { 0 : k } \right) , d \zeta _ { k + 1 : \infty } \right) = 1 .
$$

Together with (8.13), this implies that there exist measurable sets $D _ { j , k } \subset \Xi _ { j , k } \times \mathcal { Z } _ { 0 : k }$ for $j \in \mathcal I$ and $k \geq 0$ such that

$$
\begin{array} { r } { \sum _ { j \in \mathcal { I } } \sum _ { k \geq 0 } \mathcal { P } \big \{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } \big \} = 1 , } \end{array}\tag{8.14}
$$

and, for all (j, k) with $\mathcal { P } ^ { b } ( \Xi _ { j , k } ) > 0 $

$$
\int \mathbf { 1 } _ { E _ { \alpha } } \left( \xi , \zeta _ { 0 : k } , \zeta _ { k + 1 : \infty } \right) \mathcal { K } _ { k + 1 : \infty } \left( \left( \xi , \zeta _ { 0 : k } \right) , d \zeta _ { k + 1 : \infty } \right) = 1 , \quad \forall \left( \xi , \zeta _ { 0 : k } \right) \in D _ { j , k } .\tag{8.15}
$$

Note that since the sets $\Xi _ { j , k }$ are disjoint, the events $\{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } \} , j \in \mathcal { I } , k \ge 0$ , are disjoint. For each set $D _ { j , k }$ such that $\mathcal { P } ^ { b } ( \Xi _ { j , k } ) > 0$ , consider the stochastic kernel $\mathscr { K } _ { k + 1 : \infty } ( ( \xi , \zeta _ { 0 : k } ) , d \zeta _ { k + 1 : \infty } )$ Specifically, fix an arbitrary pair $( \xi , \zeta _ { 0 : k } ) \in D _ { j , k }$ , and consider the probability measure $\hat { \mathcal { P } } ( \cdot ) : =$ $\mathcal { K } _ { k + 1 : \infty } \big ( ( \xi , \zeta _ { 0 : k } ) , \cdot \big )$ on $\mathcal { Z } _ { k + 1 : \infty }$ . For $n \geq 0 .$ denote

$$
\begin{array} { r } { \hat { Q } _ { n } = Q _ { k + n } , \quad \hat { \alpha } _ { n } = \alpha _ { k + n } , \quad ( \hat { R } _ { n } ^ { s a } , \hat { S } _ { n } ^ { s a } ) = ( R _ { k + n } ^ { s a } , S _ { k + n } ^ { s a } ) , } \end{array}
$$

and note that, for $n = 0$ , the values of these variables are fully determined by $\zeta _ { k }$ . By (8.15) and the definition of $E _ { \alpha }$ , for all $( s , a ) \in \tilde { \Gamma } ( \xi )$ ,

$$
\begin{array} { r } { \sum _ { n } \hat { \alpha } _ { n } ( s , a ) = \infty , \quad \sum _ { n } \hat { \alpha } _ { n } ^ { 2 } ( s , a ) < \infty , \quad \hat { \mathcal { P } } \mathrm { - a . s . } } \end{array}\tag{8.16}
$$

Define an increasing sequence of σ-algebras $\{ \hat { \mathcal { F } } _ { n } \} _ { n \geq 0 }$ on $\mathcal { Z } _ { k + 1 : \infty }$ by

$$
\hat { \mathcal { F } } _ { n } : = \sigma \big ( \hat { Q } _ { m } , \hat { \alpha } _ { m } , \hat { R } _ { m } ^ { s a } , \hat { S } _ { m } ^ { s a } ; m \le n , ( s , a ) \in \mathcal { A } \big ) .
$$

in particular, $\hat { \mathcal { F } } _ { 0 }$ is the trivial σ-algebra $\{ \emptyset , \mathcal { Z } _ { k + 1 : \infty } \}$

Define a probability measure κ<sub>0</sub> and stochastic kernels $\kappa _ { n } , n \geq 1$ , by

$$
\begin{array} { r l } & { \kappa _ { 0 } ( d \zeta _ { k + 1 } ) : = K _ { k + 1 } \big ( ( \xi , \zeta _ { 0 : k } ) , d \zeta _ { k + 1 } \big ) , } \\ & { \kappa _ { n } \big ( ( \zeta _ { k + 1 } , \dots , \zeta _ { k + n } ) , d \zeta _ { k + n + 1 } \big ) : = K _ { k + n + 1 } \big ( ( \xi , \zeta _ { 0 : k + n } ) , d \zeta _ { k + n + 1 } \big ) , \quad n \geq 1 . } \end{array}
$$

For each $n \geq 0 , \kappa _ { n }$ is a version of the conditional distribution of $\big ( \hat { Q } _ { n + 1 } , \hat { \alpha } _ { n + 1 } , ( \hat { R } _ { n + 1 } ^ { s a } , \hat { S } _ { n + 1 } ^ { s a } ) _ { ( s , a ) \in \mathcal { A } } \big )$ under $\hat { \mathcal { P } }$ given ${ \hat { \mathcal { F } } } _ { n }$ . The discussion leading to (8.12) then shows that, under Assumption 8.1(iii), for all $n \geq 0$ and each $( s , a ) \in A _ { \ O }$ , the conditional distribution of $( \hat { R } _ { n + 1 } ^ { s a } , \hat { S } _ { n + 1 } ^ { s a } )$ given ${ \hat { \mathcal { F } } } _ { n }$ coincides the reward and transition distribution of the MDP at state s under action a. Consequently, Lemma 8.1 is applicable to the iterates $\{ { \hat { Q } } _ { n } \}$ on $\tilde { \Gamma } ( \xi )$ under $\hat { \mathcal { P } } _ { }$ , and hence

$$
{ \hat { Q } } _ { n } | _ { { \tilde { \Gamma } } ( \xi ) }  { \hat { Q } } ^ { * } ( \xi ) \mathrm { a s } n  \infty , { \hat { \mathcal { P } } } \mathrm { - a . s . , }\tag{8.17}
$$

where $\hat { Q } ^ { \ast } ( \xi )$ is the unique solution to (4.29) associated with $\left\{ v _ { * } ( \xi ) , \epsilon ( \xi ) , \tilde { S } ^ { \mathrm { r } } ( \xi ) , \tilde { S } ^ { \mathrm { b } } ( \xi ) , \tilde { \Gamma } ( \xi ) \right\}$ Since this holds for all $( \xi , \zeta _ { 0 : k } ) \in D _ { j , k }$ , using the stochastic kernel representation of ${ \mathcal { P } } ,$ we obtain

$$
\begin{array} { r } { \mathcal { P } \{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } , Q _ { t } | _ { \tilde { \Gamma } ( \xi ) } \to \hat { Q } ^ { * } ( \xi ) \} = \mathcal { P } \{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } \} . } \end{array}\tag{8.18}
$$

Finally, consider the probability of the event $\{ Q _ { t } | _ { \tilde { \Gamma } ( \xi ) }  \hat { Q } ^ { * } ( \xi ) \}$ under ${ \mathcal { P } } _ { : }$ , where we interpret the event as false whenever $\tilde { \Gamma } ( \boldsymbol { \xi } )$ or $\hat { Q } ^ { \ast } ( \xi )$ is not well-defined. $\mathrm { B y }$ (8.14) and (8.18), we have

$$
\begin{array} { r l } & { \mathcal { P } \big \{ Q _ { t } \big | _ { \tilde { \Gamma } ( \xi ) }  \hat { Q } ^ { * } ( \xi ) \big \} = \displaystyle \sum _ { j \in \mathcal { T } , k \geq 0 } \mathcal { P } \big \{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } , Q _ { t } \big | _ { \tilde { \Gamma } ( \xi ) }  \hat { Q } ^ { * } ( \xi ) \big \} } \\ & { \quad \quad \quad = \displaystyle \sum _ { j \in \mathcal { T } , k \geq 0 } \mathcal { P } \big \{ ( \xi , \zeta _ { 0 : k } ) \in D _ { j , k } \big \} = 1 . } \end{array}
$$

Hence, $Q _ { t } \big | _ { \tilde { \Gamma } ( \xi ) }  \hat { Q } ^ { * } ( \xi )$ a.s.

Consider now the sequence $\{ x _ { t } \}$ defined from $v _ { t }$ and $Q _ { t }$ via (4.30). By Lemma 4.2 and the a.s. convergences of $v _ { t }  v ,$ <sub>∗</sub> and $Q _ { t } | _ { \tilde { \Gamma } }  \hat { Q } ^ { * }$ , we have $x _ { t } \to \bar { x } \mathrm { a . s . }$ , where

$$
\begin{array} { r } { \bar { x } ( s ) : = \left\{ \begin{array} { l l } { v _ { * } ( s ) , \quad } & { s \in \tilde { S } ^ { \mathrm { r } } , } \\ { \operatorname* { m a x } _ { a \in \tilde { A } ( s ) } \hat { Q } ^ { * } ( s , a ) , } & { s \in {  \mathcal { S } } \setminus \tilde { S } ^ { \mathrm { r } } . } \end{array} \right. } \end{array}
$$

Since $\hat { Q } ^ { * }$ is the unique solution to (4.29), it follows that x¯ restricted to ${ \mathcal { S } } \setminus { \tilde { S } } ^ { \mathrm { r } }$ satisfies

$$
\bar { x } ( s ) = \operatorname* { m a x } _ { a \in \bar { A } ( s ) } \left\{ r ( s , a ) - g ^ { * } ( s ) - \epsilon { \bf 1 } _ { \bar { S } ^ { \mathrm { b } } } ( s ) + \sum _ { s ^ { \prime } \in \bar { S } ^ { \prime } } p ( s ^ { \prime } \mid s , a ) v _ { * } ( s ^ { \prime } ) + \sum _ { s ^ { \prime } \notin \bar { S } ^ { \prime } } p ( s ^ { \prime } \mid s , a ) \bar { x } ( s ^ { \prime } ) \right\} , \forall s \in \mathcal { S } \backslash \bar { \mathcal { S } } ^ { \mathrm { r } } ,
$$

and is the unique solution to this equation. Combining this with the AOEs for the sub-MDPs $\mathcal { M } _ { \ell , k }$ corresponding to $C _ { \ell , k } \in \tilde { \mathcal { C } } ^ { \mathrm { r } }$ , which are satisfied by the corresponding components of $v _ { * }$ , yields the assertion of Theorem 4.1 for (4.31) and x¯ (for fixed $\epsilon )$

Finally, view x¯ as a function of ϵ and write $\bar { x } _ { \epsilon }$ . If <sup>¯</sup>δ is suficiently small, then $\tilde { \mathcal { A } } = \mathcal { A } _ { \ast }$ by (4.25), and hence $\bar { x } _ { \epsilon }$ solves

$$
x = \operatorname* { m a x } _ { \mu \in \Pi _ { A _ { * } } } \left\{ r _ { \mu } - g ^ { * } + P _ { \mu } x \right\} - \epsilon \mathbf { 1 } _ { \tilde { S } ^ { \mathrm { b } } } .
$$

By the final step of the proof of Prop. 4.1, as $\epsilon  0 , \bar { x } _ { \epsilon }$ converges to a solution $x ^ { * }$ of the AOE (4.1), with $x ^ { * } = v _ { * }$ on ${ \tilde { S } } ^ { \mathrm { r } }$ □

## 8.3 Algorithm 3: proofs of Lemma 5.2 and Theorem 5.1

We outline the proofs of Lemma 5.2 and Theorem 5.1, omitting lengthy arguments analogous to those in Step 3 of the proof of Theorem 4.1 (Section 8.2.3), arguments that involve explicit representations of (conditional) probability measures as products of stochastic kernels.

## 8.3.1 Proof of Lemma 5.2 (outline)

The finite-time stabilization of $\tilde { \mathcal { A } } _ { t }$ and property (5.6) of its limit $\tilde { \mathcal { A } }$ were established in Lemma 4.2, following from the convergence of Algorithm 1 and the update rules (4.18)–(4.20) for $\tilde { \mathcal { A } } _ { t }$

For the remaining assertions of Lemma 5.2 concerning the convergence of $\{ Q _ { \beta , t } \}$ and the finite-time stabilization of $\tilde { B } _ { t } .$ we first prove the following lemma.

Lemma 8.2. The sequence $\{ Q _ { \beta , t } \}$ and $\{ v _ { \beta , t } \}$ generated in Phase 1 of Algorithm 3 converge $a . s .$ satisfying

$$
Q _ { \beta , t } \to Q _ { \beta } , \qquad v _ { \beta , t } \to v _ { \beta } ,\tag{8.19}
$$

where $Q _ { \beta }$ is the β-discounted optimal action-value function, and $v _ { \beta }$ is the β-discounted optimal value function, of the sample path-dependent MDP $\widetilde { \mathcal { M } } = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$

Proof. Since $\tilde { \mathcal { A } } _ { t } = \tilde { \mathcal { A } }$ for all suficiently large $t ,$ the Q-learning updates (5.4) in Phase 1 of Algorithm 3 eventually correspond to β-discounted Q-learning iterates for the MDP $\widetilde { \mathcal { M } } = ( S , \tilde { \mathcal { A } } , p , r - g ^ { * } )$ with rewards involving a vanishing biased noise term $g ^ { * } ( s ) - g _ { t } ( s )$ . We proceed as in the proof of Theorem 4.1 and first consider the limiting form of Phase 1, where $\{ g _ { t } \}$ and $\tilde { \cal A }$ are fixed (and thus so is the MDP $\widetilde { \mathcal { M } } )$

By [5, Prop. 4.5], β-discounted Q-learning converges a.s. in the presence of vanishing biased reward noise. Therefore, the limiting form produces iterates $Q _ { \beta , t }$ and $v _ { \beta , t }$ satisfying (8.19).

By arguments analogous to Step 3 of the proof of Theorem 4.1 (see Section 8.2.3), this convergence extends to the original form of Phase 1 of Algorithm 3, yielding the desired conclusion. □

From Lemma 8.2 and the update rule (5.5) for the sets $\tilde { B } _ { t }$ , we conclude that $\tilde { B } _ { t }$ eventually becomes constant, and that the resulting set B<sup>˜</sup> satisfies property (5.6). This establishes Lemma 5.2.

## 8.3.2 Proof of Theorem 5.1 (outline)

By Lemmas 8.2 and 5.2, almost surely, $v _ { \beta , t }  v _ { \beta }$ and $\tilde { B } _ { t } = \tilde { B }$ for all suficiently large t. Thus, in Phase 2 of Algorithm 3, the base algorithm is eventually applied to the (sample path-dependent) MDP $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ , but with RVI Q-learning in Phase 2 using rewards $- v _ { \beta , t } ( s )$ , which satisfy $v _ { \beta , t } ( s ) \to v _ { \beta } ( s )$ for all $s \in S$

As in the proof of Theorem 4.1, we first consider the limiting form of Phase 2, where $\tilde { B }$ and $v _ { \beta }$ as well as the convergent sequence $\{ v _ { \beta , t } \}$ , are treated as given.

Lemma 8.3. Suppose $\tilde { B } , \ v _ { \beta }$ , and $\{ v _ { \beta , t } \}$ are given, with $v _ { \beta , t }  v _ { \beta }$ . Run the base algorithm on the MDP $\mathcal { M } ^ { \prime } = ( \mathcal { S } , \tilde { \mathcal { B } } , p , - v _ { \beta } )$ , except that in Phase ${ \it 2 } ,$ RVI Q-learning uses rewards $- v _ { \beta , t } ( s )$ instead of $- v _ { \beta } ( s )$ . Then, almost surely for all suficiently large $t ,$ the policy $\pi ^ { ( t ) }$ produced in Phase $\it 4$ of the base algorithm is gain-optimal for M<sup>′</sup>.

Proof (outline). The only diference from the earlier setting in Section 3 is that, on each communicating sub-MDP $\mathcal { M } _ { \ell , k }$ in Bather’s decomposition of $\mathcal { M } ^ { \prime }$ , the RVI Q-learning iterates involve additional biased reward noise bounded by a deterministic diminishing sequence. This scenario is covered by extended analyses of RVI Q-learning under more general noise conditions [38], using asynchronous stochastic approximation results from [39]. In particular, the desired convergence of RVI Q-learning in this setting follows from [39, Thm. 2.2 and Cor. 2.1] and [38, Proof of Thm. 3.1].

(The latter applies straightforwardly after specializing from SMDPs to MDPs and redefining the noise terms $M _ { n + 1 }$ and $\epsilon _ { n + 1 }$ [38, Lem. 4.2] to match the present setting.)

This convergence of RVI Q-learning in Phase 2 ensures that Phases 3 and 4 of the base algorithm retain the same convergence properties as in the previous setting. In particular, the policy $\pi ^ { ( t ) }$ from Phase 4 is eventually gain-optimal for M<sup>′</sup> a.s. □

Finally, by arguments analogous to Step 3 of the proof of Theorem 4.1 (see Section 8.2.3), the conclusion of Lemma 8.3 extends from the limiting form to the original form of Phase 2 of Algorithm 3. This establishes Theorem 5.1.

## 9 Concluding remarks

In this work, we developed three asynchronous value-iteration-based RL algorithms for solving general multichain MDPs under the average-reward criterion. The algorithms leverage Bather’s UHD to recast the global decision problem into structured subproblems and require no model knowledge beyond the MDP’s transition graph. They address, respectively, optimal gain estimation and gain-optimal policy learning, approximate solution of the multichain AOE with near gain-optimality guarantees, and the computation of policies approaching bias optimality. We provided almost-sure convergence and eventual performance guarantees for all three algorithms, and our preliminary experiments illustrate their diferent tradeofs in gain optimality and transient performance. The ideas underlying these algorithms can also be combined in diferent ways to yield a broader family of essentially model-free RL algorithms, as illustrated by several variants discussed in the paper.

The performance guarantees established here are eventual: although the policies produced by our algorithms attain the stated (near) optimality properties after finite time a.s., the algorithms do not certify when this has occurred. This difers from high-probability policy identification results in the RL literature. We mention two such results for comparison. For communicating MDPs, the model-based policy iteration algorithm of [8] identifies a bias-optimal policy with high probability at a finite stopping time under strong policy uniqueness conditions that ensure robustness to model perturbations. For multichain MDPs, high-probability identification of near gain-optimal policies, along with sample complexity analysis, has been obtained in [40] through a discounted reduction and a model-based approach. These results provide a diferent type of guarantee from the almost-sure eventual guarantees studied here.

An important direction for future work is to remove the transition-graph assumption. This assumption is considerably weaker than full model knowledge and is natural in many simulationbased settings, where possible successor states can be enumerated from a simulator’s code or interface even when their transition probabilities are dificult to compute or depend on unobserved factors, such as other agents’ behavior or physical interactions. Nevertheless, it remains a genuine restriction. Within the almost-sure convergence framework considered here, the transition graph could in principle be learned in parallel with our algorithms through suficiently persistent exploration.

Rather than first learning the entire transition graph, it would be more interesting to discover and maintain the relevant structure online. Since Bather’s UHD depends only on which transitions are possible, an estimated decomposition could be dynamically refined as new transitions are observed. This also raises broader questions about how qualitative structural information about complex stochastic dynamics can be learned through exploration and exploited for control, separately from learning quantitative transition probabilities. We plan to investigate these questions in future work.

There are several other directions we plan to explore. The subproblem structure induced by Bather’s UHD suggests connections to the options framework and, in particular, the possibility of reusing learned sub-policies as options. We also plan more extensive experiments to better understand the relative strengths of the algorithms and their variants, as well as more flexible and adaptive choices of algorithmic parameters in place of the fixed thresholds used in our theoretical analysis.

## Acknowledgments

This research was supported by the Alberta Machine Intelligence Institute (Amii) and the Natural Sciences and Engineering Research Council of Canada (NSERC) under grants RGPIN-2024-04939 and DGECR-2024-00312. H.Y. thanks Professor Eugene Feinberg for helpful discussions on averagereward MDPs and for drawing her attention to Bather’s work. Both authors thank Professors Csaba Szepesvari and Rich Sutton for valuable feedback. We thank anonymous reviewers of an earlier version of this paper for helpful comments and suggestions that led to improvements in the presentation of this work. In preparing this paper, we used OpenAI’s ChatGPT (GPT-4 and GPT-5) to help refine the writing style.

Both authors contributed equally to this work.

## A Background: RVI Q-learning

RVI Q-learning, introduced by Abounadi, Bertsekas, and Borkar [1], is a stochastic analogue of the classical RVI algorithm for average-reward MDPs [25, 35]. Its development and analysis build on a “self-regulating” algorithmic design (reflected in the function f and its conditions discussed below), stochastic approximation theory [9–11, 39], and Borkar and Soumyanath’s general results concerning continuous-time analogs of fixed-point iteration involving nonexpansive mappings [12]. We use the asynchronous version of RVI Q-learning in our base algorithm for solving communicating sub-MDPs. Below we briefly explain the algorithm and the conditions and convergence guarantees needed for our purpose.

Consider a weakly communicating MDP $\mathcal { M } = ( \boldsymbol { S } , \boldsymbol { A } , \boldsymbol { p } , \boldsymbol { r } )$ . Let $\left\{ \alpha _ { n } \right\}$ be a given setpsize sequence and $f : \mathbb { R } ^ { | \mathcal { A } | }  \mathbb { R } \mathrm { ~ a ~ }$ Lipschitz continuous function. Further conditions on $\left\{ \alpha _ { n } \right\}$ and f will be given shortly. RVI Q-learning generates a sequence $\{ Q _ { t } \}$ by iteratively updating selected components of $Q _ { t }$ as follows. At iteration $t \geq 0 ;$

• Select a nonempty subset $Y _ { t } \subset A .$ . For each $( s , a ) \in Y _ { t }$ , sample a random state transition and reward $( S _ { t + 1 } ^ { s a } , R _ { t + 1 } ^ { s a } )$ from $( s , a )$ according to M.

• For $( s , a ) \in Y _ { t }$ , let

$$
Q _ { t + 1 } ( s , a ) = Q _ { t } ( s , a ) + \alpha _ { \nu ( t , ( s , a ) ) } \Big ( R _ { t + 1 } ^ { s a } - f ( Q _ { t } ) + \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } ( S _ { t + 1 } ^ { s a } ) } Q _ { t } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } ) - Q _ { t } ( s , a ) \Big ) .\tag{A.1}
$$

For $( s , a ) \notin Y _ { t }$ , let $Q _ { t + 1 } ( s , a ) = Q _ { t } ( s , a )$ . In (A.1),

$$
\nu ( t , ( s , a ) ) : = \sum _ { k = 0 } ^ { t - 1 } \mathbb { 1 } \{ ( s , a ) \in Y _ { k } \}
$$

counts how many times the state-action pair $( s , a )$ has been chosen prior to iteration t. Thus, if a pair has previously been selected n times, the update at iteration t uses the stepsize $\alpha _ { n }$ We next explain what the algorithm aims to compute and state the conditions needed for its convergence.

The goal is to solve an equivalent form of the AOE for M in terms of Q-factors:

$$
Q ( s , a ) = r ( s , a ) - \bar { r } + \sum _ { s ^ { \prime } \in S } p ( s ^ { \prime } \mid s , a ) \operatorname* { m a x } _ { a ^ { \prime } \in A ( s ^ { \prime } ) } Q ( s ^ { \prime } , a ^ { \prime } ) , \qquad ( s , a ) \in \mathcal { A } ,\tag{A.2}
$$

where r¯ is a scalar variable playing the role of the unknown optimal gain $g ^ { * }$ . (This equation can also be interpreted as the AOE of an MDP on enlarged state and action spaces, so average-reward MDP theory directly applies to it.) For any solution $( \bar { r } , Q )$ of this equation, necessarily $\bar { r } = g ^ { * }$ , and the function $h ( s ) = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } Q ( s , a ) , s \in \mathcal { S }$ , solves the AOE (4.1) for $\mathcal { M }$

The function f employed in RVI Q-learning serves to “self-regulate” and normalize the iterates, with $f ( Q _ { t } )$ intended to converge to $g ^ { * }$ . This normalization is what makes convergence of the $Q _ { t }$ iterates to the AOE solution set possible. Recently, Yu et al. [38] introduced the following general monotonicity conditions on $f$ and its scaling limit $f _ { \infty }$

Let $d = | A |$ . We call a function $\phi : \mathbb { R } ^ { d } $ R strictly increasing under scalar translation (SISTr) if, for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , the function $c \in \mathbb { R } \mapsto \phi ( x + c \mathbf { 1 } )$ is strictly increasing and maps R onto R. If this condition holds at a specific point x, we say that $\phi$ is SISTr at x.

Assumption A.1 (Conditions on function $f ; [ 3 8 ] )$

(i) f is Lipschitz continuous and SISTr.

(ii) As $c \uparrow \infty$ , the functions $x \mapsto f ( c x ) / c$ converge pointwise to a real-valued function $f _ { \infty } ( x )$ , and $f _ { \infty }$ is $S I S T r$ at the origin.

Functions satisfying this assumption include the following family of Lipschitz continuous functions considered in [1, 32]:

• For some scalar $u > 0 , f ( x + c \mathbf { 1 } ) = f ( x ) +$ cu for all $x \in \mathbb { R } ^ { d }$ and $c \in \mathbb { R } ;$ and

• for $c \geq 0 , f ( c x ) = f ( 0 ) + c ( f ( x ) - f ( 0 ) )$

For these functions, the scaling limit is $f _ { \infty } ( x ) = f ( x ) - f ( 0 ) , \mathrm { i . e . , } \ f _ { \infty }$ coincides with $f$ after subtracting $f ( 0 )$ . Examples include $\begin{array} { r } { f ( x ) = b + { \theta } ^ { \top } { x } . } \end{array}$ where $b \in \mathbb { R } , \theta \in \mathbb { R } ^ { d }$ , and $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \theta _ { i } > 0 ; } \end{array}$ and $f ( x ) = b + \beta \operatorname* { m a x } _ { i \in D } x _ { i }$ or $\begin{array} { r } { f ( x ) = b + \beta \operatorname* { m i n } _ { i \in D } x _ { i } . } \end{array}$ , where $b \in \mathbb { R } , \beta > 0 .$ , and $D \subset \{ 1 , 2 , \dotsc , d \}$ Examples beyond this family but admissible under the more general Assumption A.1 include $f ( x ) =$ max $\{ g _ { 1 } ( x ) , \ldots , g _ { m } ( x ) \}$ or min $\{ g _ { 1 } ( x ) , \ldots , g _ { m } ( x ) \}$ , where each function $g _ { k }$ satisfies Assumption $\mathrm { A . 1 }$ e.g., by fitting into one of the aforementioned types. See [38, Ex. 3.2] for more general examples involving a compositional mapping in place of max or min.

We next state the remaining conditions, concerning the stepsizes and the asynchronous update schedule.

Assumption A.2 (Stepsize conditions).

(i) $\begin{array} { r } { \sum _ { n } \alpha _ { n } = \infty , \sum _ { n } \alpha _ { n } ^ { 2 } < \infty } \end{array}$ , and $\alpha _ { n + 1 } \leq \alpha _ { n }$ for all n suficiently large.

(ii) $\begin{array} { r } { F o r x \in ( 0 , 1 ) , \operatorname* { s u p } _ { n } \frac { \alpha _ { [ x n ] } } { \alpha _ { n } } < \infty _ { } } \end{array}$ , where [xn] denotes the integral part of xn.

(iii) For $x \in ( 0 , 1 )$ , as n → ∞, $\frac { \sum _ { k = 0 } ^ { [ y n ] } \alpha _ { k } } { \sum _ { k = 0 } ^ { n } \alpha _ { k } }  1$ uniformly in $y \in [ x , 1 ]$

For $x > 0$ and $t \geq 0 .$ , define $N ( t , x ) : =$ min $\begin{array} { r } { \{ m > t : \sum _ { k = t } ^ { m } \alpha _ { k } \geq x \} } \end{array}$

Assumption A.3 (Asynchronous update conditions).

(i) For some deterministic $\eta > 0$ , lim inf $_ { t  \infty } \nu ( t , i ) / t \geq \eta \ a . s .$ , for all $i = ( s , a ) \in \mathcal { A }$

(ii) For each $x > 0$ , the limit lim $t  \infty \frac { \sum _ { k = \nu ( t , i ) } ^ { \nu ( N ( t , x ) , i ) } \alpha _ { k } } { \sum _ { k = \nu ( t , j ) } ^ { \nu ( N ( t , x ) , j ) } \alpha _ { k } }$ exists a.s., for all $i = ( s , a ) , j = ( s ^ { \prime } , a ^ { \prime } ) \in \mathcal { A }$

These conditions are based on those introduced in Borkar [9] for asynchronous stochastic approximation and adopted in [1] for RVI Q-learning. Admissible stepsize examples include stepsizes asymptotic to $1 / n { \mathrm { ~ o r ~ } } 1 / ( n \ln n )$ . For asynchronous updates, Assumption A.3(i) requires a strictly positive frequency for updating each Q-factor component, while Assumption $\mathrm { A . 3 ( i i ) }$ requires the components to receive comparable amounts of updates on the stepsize time scale. Both conditions are met, in particular, if the subset $Y _ { t }$ of state-action pairs is eventually chosen based on an irreducible Markov chain on A (see [33, Ex. 3]).

While these conditions are more restrictive than the stepsize and asynchony conditions required for classical Q-learning [29], they are needed here because the mappings underlying RVI Q-learning are generally neither contractive nor nonexpansive. They impose partial asynchrony, allowing the asymptotic behavior of the asynchronous algorithm to align, on average, with that of a synchronous counterpart and thereby facilitating analysis [9, 39].

Under Assumptions $\mathrm { { A . 1 { \mathrm { - A . 3 } } } , }$ it is shown in [38, Thm. 3.1] that RVI Q-learning (A.1) converges a.s., with

$$
f ( Q _ { t } ) \to g ^ { * } , \qquad Q _ { t } \to \mathrm { a ~ c o m p a c t ~ s u b s e t ~ o f ~ s o l u t i o n s ~ o f ~ t h e ~ A O E ~ ( A . 2 ) , }\tag{A.3}
$$

where the subset is generally sample-path-dependent. Moreover, this convergence of $Q _ { t }$ implies that, $\mathrm { a . s . }$ , any greedy policy w.r.t. $Q _ { t }$ is gain-optimal after finite time, and that the state-value functions $\begin{array} { r } { v _ { t } ( s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } Q _ { t } ( s , a ) , s \in \mathcal { S } _ { } } \end{array}$ , converge to a compact subset of solutions of the AOE (4.1).

Furthermore, under additional stepsize and asynchrony conditions [38, Assump. 3.4(i)], [38, Thm. 3.2] establishes the a.s. convergence of RVI Q-learning to a unique sample-path-dependent AOE solution. (This result itself is an application of the more general result for asynchronous stochastic approximation algorithms given in [39, Thm. 2.3].) Specialized to the context here, the following gives a particular choice of stepsize sequence and asynchronous update scheme satisfying the required conditions (cf. [38, Rem. 3.4]).

• Let the selection of the components to be updated eventually be governed by an irreducible Markov chain on ${ \mathcal { A } } .$

• Let $\textstyle \alpha _ { n } = { \frac { 1 } { A n } }$ for $n \geq 1$ , where $A > 0$ is a scaling parameter satisfying

$$
A > 4 + 2 L _ { f } ,\tag{A.4}
$$

where $L _ { f }$ is the Lipschitz constant of $f { \mathrm { ~ w . r . t . ~ } } \| \cdot \| _ { \infty }$

With these choices, [38, Thm. 3.2] implies that $Q _ { t }$ converges a.s. to a unique sample-path-dependent solution of the AOE (A.2), and its associated state-value function $v _ { t }$ converges to the corresponding solution of the AOE (4.1).

## B Background: Q-learning for SSP-type total-reward MDPs

Let $\hat { \mathcal { M } } = ( \hat { S } \cup \Delta , \hat { \mathcal { A } } , \hat { p } , \hat { r } )$ be a total-reward MDP of SSP type (as defined in Section 2.2). Q-learning for $\hat { \mathcal { M } }$ is similar to classical Q-learning for discounted problems [34], except that the discount factor is replaced by 1 and the Q-value at the terminal state $\Delta$ is fixed at zero. Specifically, at iteration $t \geq 0$ , the algorithm performs the following operations:

• Select a nonempty subset $Y _ { t } \subset \hat { \mathcal { A } }$ . For each $( s , a ) \in Y _ { t }$ , sample a random state transition and reward $( S _ { t + 1 } ^ { s a } , R _ { t + 1 } ^ { s a } )$ from $( s , a )$ according to $\hat { \mathcal { M } }$

• For $( s , a ) \in Y _ { t }$ , let

$$
Q _ { t + 1 } ( s , a ) = Q _ { t } ( s , a ) + \alpha _ { t } ( s , a ) \biggl ( R _ { t + 1 } ^ { s a } + \mathbb { 1 } \{ S _ { t + 1 } ^ { s a } \neq \Delta \} \cdot \operatorname* { m a x } _ { a ^ { \prime } \in \dot { A } ( S _ { t + 1 } ^ { s a } ) } Q _ { t } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } ) - Q _ { t } ( s , a ) \biggr ) ,\tag{B.1}
$$

where $\alpha _ { t } ( s , a ) \geq 0$ is a possibly random stepsize that can depend on information available at time t, including $Y _ { t }$ (but not on the transition data indexed by $t + 1 )$ . For $( s , a ) \notin Y _ { t }$ $Q _ { t + 1 } ( s , a ) = Q _ { t } ( s , a )$

We mention that more generally, when implemented using parallel distributed computation, instead of $Q _ { t } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } )$ , the update (B.1) can use delayed Q-factors of the form $Q _ { t - \tau } ( S _ { t + 1 } ^ { s a } , a ^ { \prime } ) , \tau \geq 0$ , to allow communication delays between diferent processors (see [29, 36] for details). For simplicity of exposition, we do not consider communication delays here.

Almost-sure convergence of $Q _ { t }$ to the optimal action-value function $Q ^ { * }$ of $\hat { \mathcal { M } }$ follows from the convergence results of Tsitsiklis [29, Thm. 2] given stability, together with the stability results of Yu and Bertsekas [36, Thm. 3.1], under the same general stepsize and asynchrony conditions as in discounted $\mathrm { Q } \mathrm { . }$ -learning:

Assumption B.1 (Asynchrony and stepsize conditions). The following hold $a . s . .$

(i) Every Q-factor component is updated infinitely often.

(ii) For each $( s , a ) \in { \hat { A } }$ , under the convention that $\alpha _ { t } ( s , a ) = 0$ whenever $( s , a ) \notin Y _ { t }$ , the stepsizes satisfy $\textstyle \sum _ { t } \alpha _ { t } ( s , a ) = \infty$ and $\textstyle \sum _ { t } \alpha _ { t } ^ { 2 } ( s , a ) < \infty$

The stability proof in [36] is quite involved for a general SSP-type MDP $\hat { \mathcal { M } }$ with improper policies. It need not be invoked for several special cases of $\hat { \mathcal { M } }$ , where almost-sure convergence of $Q _ { t }$ to $Q ^ { * }$ follows readily from alternative arguments that exploit stronger properties of $\hat { \mathcal { M } }$ . These cases include:

• When every stationary policy in $\hat { \mathcal { M } }$ is proper, the mapping underlying the Q-learning iteration (B.1) is contractive w.r.t. a weighted sup-norm [5, Prop. 2.2]. Consequently, $Q _ { t } \to Q ^ { * }$ a.s. by [29, Thm. 3] (see also [5, Secs. 4.3, 5.6]).

• When improper policies may exist in $\hat { \mathcal { M } }$ but the one-stage reward function ${ \hat { r } } ( s , a )$ is nonpositive, the a.s. convergence $Q _ { t } \to Q ^ { * }$ was proved in [29, Lem. 9 and Thm. 4] for the case where all random rewards are also nonpositive (see also [5, Prop. 5.6]), and later in [36, Prop. 3.2] with most of the sign restrictions on the random rewards removed.

Our base algorithm applies the two special cases above: its Phase 3 uses the first case in solving the aggregated OSP, while its Phase 4 uses the second case in a realization of the ReachPolicy subroutine. Algorithm 2 for computing an approximate AOE solution involves solving a general SSP-type MDP, and its convergence analysis required a slight extension of prior stability and convergence results from [29, 36], which we provided in Section 8.2.1.

## C Experimental methodology and additional results

This section provides further details on the experimental methodology and results. Section C.1 describes the MDP generation procedure and the resulting benchmark. Section C.2 specifies the implementation, update schedules, and parameter choices. Section C.3 presents per-MDP results and discusses the observed diferences in gain and transient performance.

## C.1 MDP generation

MDPs are generated using a structured procedure that directly leverages Bather’s UHD framework, ensuring that every generated instance has a known, valid hierarchical decomposition by construction. For each MDP, the number of levels L is sampled uniformly in the range [3, 6], the number of classes per level in the range [1, 4], the number of transient states per level in the range [0, 4], and the number of states per class in the range [3, 8].

Each communicating class $C _ { \ell , k }$ is constructed by first laying a random directed cycle over its states (guaranteeing communication), then adding additional random internal transitions (possibly stochastic) with probability sampled uniformly in the range [0.2, 0.6] for each MDP. For each class at a level $\ell \geq 1$ , a single exit action transitioning deterministically to a randomly selected class at level ℓ − 1 is added to ensure the desired hierarchical structure. Additional exit actions are added with probability sampled from [0.4, 0.8], these exit actions can be stochastic, with each stochastic transition pointing to a random subset of between 1 and 3 states, at least one of which is at a strictly lower level. Transient states receive random actions pointing to classes or earlier transients at the same or lower levels, with probability sampled from [0.4, 0.8]. Critically, to preserve the desired structure, we ensure no cycles are created between transient states.

To induce genuine multichain structure within classes, each class is independently made multichain with probability sampled per-MDP from [0.2, 0.9]. Two methods are used with equal probability: the first selects a random subset of $2 - 4$ states and equips each with a self-absorbing action; the second partitions the class into $2 - 3$ random subgroups, each of which is given its own directed subcycle, so multiple distinct recurrent behaviors are possible within the class.

Table 1: Summary statistics over 100 generated MDPs.
<table><tr><td>Statistic</td><td>Min</td><td>Mean</td><td>Max</td></tr><tr><td>Structural parameters (controlled)</td><td></td><td></td><td></td></tr><tr><td>Number of levels</td><td>3</td><td>4.53</td><td>6</td></tr><tr><td>Number of communicating classes</td><td>3</td><td>11.25</td><td>23</td></tr><tr><td>Number of transient states</td><td>1</td><td>9.36</td><td>20</td></tr><tr><td>Number of states</td><td>20</td><td>71.67</td><td>147</td></tr><tr><td>Number of actions</td><td>29</td><td>149.8</td><td>359</td></tr><tr><td>States per class (mean)</td><td>3.88</td><td>5.53</td><td>6.57</td></tr><tr><td>Actions per state (mean)</td><td>1.33</td><td>2.04</td><td>3.42</td></tr><tr><td>Per-MDP reward span</td><td>14.93</td><td>20.35</td><td>27.57</td></tr><tr><td>Number of preferred (high-gain) classes</td><td>3</td><td>6.49</td><td>12</td></tr><tr><td>Number of classes made multichain</td><td>0</td><td>6.52</td><td>16</td></tr><tr><td>Emergent gain and class structure</td><td></td><td></td><td></td></tr><tr><td>Cr classes per MDP</td><td>1</td><td>2.97</td><td>7</td></tr><tr><td>Ce classes per MDP</td><td>0</td><td>3.90</td><td>14</td></tr><tr><td> $\mathcal { C } ^ { \mathrm { b } }$  classes per MDP</td><td>0</td><td>4.38</td><td>12</td></tr><tr><td>Optimal gain span</td><td>0</td><td>7.51</td><td>16.78</td></tr><tr><td>MDPs with nonconstant optimal gain</td><td></td><td>77 / 100</td><td></td></tr></table>

Rewards are assigned through a three layered mechanism applied in sequence. First, a base random reward drawn uniformly from [−3, 3] is assigned to every action. Second, a subset of classes at randomly selected preferred levels are designated as high-gain targets and given an additional reward bonus sampled from [2, 6]; non-preferred classes receive a penalty sampled from [−6, −2]. For multichain classes at preferred levels, subcycle and absorbing actions additionally receive rewards sampled from [2, 6], while ordinary internal cycling actions are penalized by amounts sampled from [−6, −2]. This creates diverse gain and bias structure within classes, where some recurrent behaviors within the same class are more desirable than others.

To explicitly construct ${ \mathcal { C } } ^ { \mathrm { { b } } } { \mathrm { - t y p e } }$ classes (those for which staying and exiting are gain-tied, and optimal decisions come down to optimizing bias), a dedicated construction is applied. Between 2 and 10 classes per-MDP are selected for this construction. For each such class, a base gain level is sampled from [1, 10]. With equal probability, we assign each such class as exit-good or stay-good, where we construct rewards in order induce optimal behavior which depends on the bias. For exit-good classes, exit actions are designed to ofer a reward bonus sampled from [5, 15], over the base gain. For stay-good classes, a value x ∈ [5, 10] is sampled, −x is applied to exit actions and +x is applied to an internal action leading to a cycle, creating a better bias from remaining within the class. This aims to ensure that a nontrivial fraction of classes present a genuine gain-bias tradeof with hopes of distinguishing behavior between our algorithms.

Table 1 summarizes the structural and emergent properties of the 100 generated MDPs. The controlled parameters reflect a deliberately diverse benchmark: MDPs range from small instances with 20 states and 29 actions to larger ones with 147 states and 359 actions, with varying numbers of levels, classes, and transient states. The per-MDP reward span, which averages 20.35 and ranges from 14.93 to 27.57, confirms that the layered reward construction — combining base random rewards with level-preference bonuses and multichain subcycle penalties — creates substantial reward variation within each MDP, giving algorithms meaningful signals to diferentiate between good and bad behavior.

The emergent class and gain structure is more revealing. As defined in equations (4.7)-(4.9), $\mathcal { C } ^ { \mathrm { r } }$ is the set of classes where it is gain-optimal to stay, $\mathcal { C } ^ { \mathrm { e } }$ is the set where it is gain-optimal to exit, and $\mathcal { C } ^ { \mathrm { b } }$ is the set of gain-tied regions, where staying and exiting is equal in terms of gain. The presence of $\mathcal { C } ^ { \mathrm { e } }$ and $\mathcal { C } ^ { \mathrm { b } }$ classes across the benchmark confirms that the construction has largely succeeded in inducing the gain-tied and gain-dominated structure that motivates the augmented algorithms. Notably, 98 of the 100 MDPs contain at least one $\mathcal { C } ^ { \mathrm { b } }$ class providing a strong basis for evaluating the performance diferences between Algorithms 2 and 3. A moderate limitation of the construction is that 24 MDPs ended up with no $\mathcal { C } ^ { \mathrm { e } }$ classes, meaning in terms of gain there is no class where it is strictly better to exit. The absence of $\mathcal { C } ^ { \mathrm { e } }$ classes primarily afects the base algorithm’s transient performance: in classes where exiting is optimal, the base algorithm uses $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ to navigate to an exit state $s ^ { * }$ via an SSP formulation, which minimizes hitting time without accounting for intermediate rewards, leading to poor transient behavior. Algorithms 2 and 3 bypass this issue through their action constraint construction, so this limitation does not materially afect the comparison between them. Since the base algorithm’s transient performance was already demonstrated to be poor in the main results and in Subsection C.3, we believe these 24 MDPs remain representative of the general behavior of all algorithms. Additionally, 76 of the MDPs still had a variety of classes which fall into the $\mathcal { C } ^ { \mathrm { e } }$ classification. Nevertheless, future benchmark construction should enforce a minimum number of $\mathcal { C } ^ { \mathrm { e } }$ classes per MDP to ensure fuller coverage of this case.

Turning to the optimal gain structure, the 77 MDPs with non-constant optimal gain confirm that the majority of the benchmark consists of genuinely multichain problems where the long-run performance depends on the initial state. The remaining 23 MDPs, despite being structurally multichain, happen to have a constant optimal gain vector, reducing to a more weakly communicating problem and representing the easier end of the spectrum. Among the 77 truly multichain instances, the span of the optimal gain varies considerably: 39 MDPs had gain span exceeding 10.0 — cases where the optimal gain varies substantially across states and where correctly identifying the optimal classes to stop at are critical to performance. The remaining 38 MDPs fall in between, exhibiting moderate gain variation. Together, the benchmark spans a wide dificulty spectrum, providing a comprehensive testbed for evaluating average-reward RL algorithms in the multichain setting.

## C.2 Experimental implementation details

All experiments were run on a 2021 Apple MacBook Pro with an Apple M1 chip and 16 GB of memory. The algorithms were implemented in Python 3.13.2. Each of the 100 MDP experiments was run for 750,000 iterations, requiring approximately 1 hour per MDP and 100 hours in total. Experiments were run sequentially across MDPs. At the beginning of each run, the parameters needed to recreate the MDP were saved, along with the bias-optimal benchmark policy. Every 1,000 iterations a light history of the run was saved consisting of the current policies of the algorithms. Every 50,000 iterations a full checkpoint was created saving all of the learner objects. Each MDP experiment took up approximately 5 MB of storage for a total of approximately 500 MB.

The implementation uses a round-robin update scheme for reproducibility and ease of comparison. In each outer iteration, every phase advances by exactly one full sweep over its state-action pairs in a fixed deterministic order (randomized only at initialization). Phases are not actually run in parallel; instead, they are frozen and advanced sequentially within each outer iteration, which approximates the concurrent operation described in the paper. Since the iterative phases of our algorithms are specified to operate concurrently, a distributed or parallel implementation is a natural next step, which we leave for future work. The augmented algorithms begin advancing only after $\bar { t } = 3 0 { , } 0 0 0$ outer iterations, allowing the base algorithm suficient time to produce stable gain and Q-function estimates before the augmented phases begin using them.

All Q-learning components use a polynomial stepsize schedule $\alpha _ { t } ( s , a ) = c / ( \mathrm { o f f s e t } + n _ { t } ( s , a ) ) ^ { \omega }$ where $n _ { t } ( s , a )$ is the number of updates to pair $( s , a )$ up to time t. The parameters used are $c = 1 . 0 \AA$ $\omega = 0 . 7$ and ofset = 10.0 for all components. A warm-start phase is applied for the base algorithm in Phase 2 RVI Q-learning: for the first 5,000 outer iterations, the more aggressive polynomial stepsize schedule above is used, after which the schedule switches to $\alpha _ { t } ( s , a ) = 1 / ( A n _ { t } ( s , a ) )$ with $A = 8 . 0$ . The f function for RVI Q-learning is taken to be the maximum of the Q-function over all state-action pairs (and over all admissible state-action pairs for Algorithm 3 Phase 2).

![](images/c25b3758e7092503b1ba18be3dec6bac981fff68399b14031ec088b3c09db3cb.jpg)  
(a) Mean transient reward prior to entering a recurrent class, averaged over all initial states, per-MDP. Higher values indicate better transient performance.

![](images/34b87b9fcbaa744234880f2422a90df258662f583c48fbb465b75bd380c50bad.jpg)  
(b) Negative mean gain diference from bias-optimal baseline, averaged over all initial states, per-MDP. Higher values indicate better gain performance.  
Figure 4: Per-MDP performance across all 100 MDPs.

The thresholds for the set constructions in Algorithms 2 and 3 are set to $\underline { { \delta } } = \underline { { \delta } } _ { 1 } = 0 . 0 3$ $\bar { \delta } = \bar { \delta } _ { 1 } = 0 . 0 6$ . The additional thresholds for Algorithm 3 are set to $\underline { { \delta } } _ { 2 } = 0 . 0 1$ and $\bar { \delta } _ { 2 } = 0 . 0 3$ The penalty parameter for Algorithm 2 is $\epsilon = 0 . 0 5$ , held constant throughout for simplicity. The discount factor for Phase 1 of Algorithm 3 is $\beta = 0 . 9 9$ , and Algorithm 3’s Phase 2 begins at iteration $\bar { t } ^ { \prime } = \bar { t } + 1 0 \mathrm { , 0 0 0 }$ . The discounted Q-learning baseline uses $\beta = 0 . 9 9 9 5$

Phase 4 of the base algorithm requires constructing a policy that reaches a target state $s ^ { * }$ from all states within a class $C _ { \ell , k }$ . This is implemented by solving an SSP problem on $C _ { \ell , k }$ using 1,000 round-robin Q-learning sweeps, with results cached across outer iterations to avoid redundant recomputation.

The 0-discount policy iteration benchmark is computed once for each MDP using the true MDP model and the exact 0-discount policy iteration algorithm [30] (see also [22, Sec. 10.3]).

## C.3 Additional experimental results and discussion

Figures 4a and 4b display per-MDP performance across all 100 MDPs. Because the aggregated plots in Figure 3 do not include error bars — which proved too visually cluttered to be legible — these per-MDP plots serve as evidence further supporting the general claims of Section 6.

Figure 4a shows the mean transient reward prior to entering a recurrent class, averaged over all initial states per-MDP. Algorithms 2 and 3 both cluster near the bias-optimal benchmark across virtually all 100 MDPs. The base algorithm, by contrast, exhibits large transient failures on a subset of MDPs. The most striking is the base algorithm’s catastrophic transient failure on MDP 43, with mean transient reward of approximately −1,380. To partially understand why this failure occurs, recall that in Phase 4 of the base algorithm, for any class $C _ { \ell , k }$ where Phase 3 decides to exit via some $( s ^ { * } , a ^ { * } )$ , the within-class policy is set to $\pi _ { \ell , k } ^ { \mathrm { r e a c h } } -$ the solution to an SSP problem that navigates to $s ^ { * }$ in minimum expected time, with no regard for rewards accumulated along the way.

The failures arise in MDPs where the shortest path to $s ^ { * }$ within one or more exited classes passes through states with large negative per-step rewards. Since $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ minimizes hitting time rather than cumulative reward, it selects these paths without penalty, potentially accumulating large negative transient costs. When several such classes appear across multiple levels of the hierarchy, these costs compound: the system traverses each exited class via a reward-blind shortest path before eventually reaching its recurrent destination, and the cumulative transient reward can become extremely poor. In MDP 43, for instance, inspection of the base algorithm’s policy reveals that $\pi _ { \ell , k } ^ { \mathrm { r e a c h } }$ routes through several states with per-step rewards in the range $[ - 8 , - 5 ]$ , whereas the bias-optimal policy takes alternative paths through the same classes that achieve per-step rewards as high as +10, precisely because it accounts for transient reward through its bias function.

Figure 4b shows the mean gain sub-optimality per-MDP. The base algorithm is largely gainoptimal across all MDPs. Extension algorithms 2 and 3 remain near gain-optimal across the majority of MDPs, though both exhibit occasional failures: Algorithm 2 shows more frequent smaller deviations, while Algorithm 3 has fewer, but occasionally larger ones. These results are consistent with Figure 3 in Section 6. As augmented algorithms are approximation-based, we expect these failures to stem from approximation error. Future work will verify why these large failures occasionally occur and will investigate the sensitivity of the algorithms to parameter choices.

Together, Figures 4a and 4b validate the aggregated claims of Section 6: the base algorithm achieves the strongest gain performance but exhibits large transient failures on a subset of MDPs, while the extensions substantially improve transient performance while remaining near gain-optimal on the vast majority of instances.

## References

[1] Abounadi, J., Bertsekas, D. P., and Borkar, V. S. (2001). Learning algorithms for Markov decision processes with average cost. SIAM J. Control Optim., 40(3):681–698.

[2] Bather, J. (1973a). Optimal decision procedures for finite Markov chains. Part II: Communicating systems. Adv. Appl. Prob., 5:521–540.

[3] Bather, J. (1973b). Optimal decision procedures for finite Markov chains. Part III: General convex systems. Adv. Appl. Prob., 5:541–553.

[4] Bertsekas, D. P. and Tsitsiklis, J. N. (1991). An analysis of stochastic shortest path problems. Math. Oper. Res., 16:185–202.

[5] Bertsekas, D. P. and Tsitsiklis, J. N. (1996). Neuro-Dynamic Programming. Athena Scientific, Belmont.

[6] Bertsekas, D. P. and Yu, H. (2012). Q-learning and enhanced policy iteration in discounted dynamic programming. Math. Oper. Res., 37:66–94.

[7] Blackwell, D. (1962). Discrete dynamic programming. Ann. Math. Stat., 33:719–726.

[8] Boone, V. and Tuynman, A. (2025). Towards Blackwell optimality: Bellman optimality is all you can get. arXiv:2510.13476.

[9] Borkar, V. S. (1998). Asynchronous stochastic approximations. SIAM J. Control Optim., 36(3):840–851. Erratum, ibid. 38 (2000), 662–663.

[10] Borkar, V. S. (2023). Stochastic Approximations: A Dynamical Systems Viewpoint. Springer and Hindustan Book Agency, Singapore and New Delhi, 2nd edition.

[11] Borkar, V. S. and Meyn, S. (2000). The o.d.e. method for convergence of stochastic approximation and reinforcement learning. SIAM J. Control Optim., 38(2):447–469.

[12] Borkar, V. S. and Soumyanath, K. (1997). A new analog parallel scheme for fixed point computation, Part I: Theory. IEEE Trans. Circuits Systems—I Fund. Theory Appl., 44(4):351–355.

[13] Federgruen, A. and Schweitzer, P. J. (1981). Nonstationary Markov decision problems with converging parameters. J. Optim. Theory Appl., 34:207–241.

[14] Federgruen, A. and Schweitzer, P. J. (1984). Successive approximation methods for solving nested functional equations in Markov decision problems. Math. Oper. Res., 9:319–344.

[15] Fox, B. L. and Landi, D. M. (1968). An algorithm for identifying the ergodic subchains and transient states of a stochastic matrix. Communications of the ACM, 11(9):619–621.

[16] Fruit, R., Pirotta, M., and Lazaric, A. (2018). Near optimal exploration-exploitation in noncommunicating Markov decision processes. In Advances in Neural Information Processing Systems.

[17] Grand-Clément, J. and Petrik, M. (2023). Reducing Blackwell and average optimality to discounted MDPs via the Blackwell discount factor. In Advances in Neural Information Processing Systems.

[18] Jin, Y., Blanchet, J., Gummadi, R., and Zhou, Z. (2024). Feasible Q-learning for average reward reinforcement learning. In International Conference on Artificial Intelligence and Statistics.

[19] Kara, A. D. and Yüksel, S. (2026). Approximations and learning for continuous state and action mdps under average cost criteria. J. Mach. Learn. Res., 27(68):1–50.

[20] Kushner, H. J. and Dupuis, P. G. (2001). Numerical Methods for Stochastic Control Problems in Continuous Time. Springer, New York.

[21] Ohno, K. (1988). A value iteration method for undiscounted multichain Markov decision processes. Z. Oper. Res, 32:71–93.

[22] Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons.

[23] Saldi, N., Linder, T., and Yüksel, S. (2018). Finite Approximations in Discrete-Time Stochastic Control: Quantized Models and Asymptotic Optimality. Springer Nature, Cham.

[24] Schweitzer, P. J. (1984). A value-iteration scheme for undiscounted multichain Markov renewal programs. Z. Oper. Res., 28(4):143–152.

[25] Schweitzer, P. J. and Federgruen, A. (1977). The asymptotic behavior of undiscounted value iteration in Markov decision problems. Math. Oper. Res., 2(4):360–381.

[26] Schweitzer, P. J. and Federgruen, A. (1978). The functional equations of undiscounted Markov renewal programming. Math. Oper. Res., 3(4):308–321.

[27] Sutton, R. and Barto, A. (2018). Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2nd edition.

[28] Sutton, R. S., Precup, D., and Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112:181–211.

[29] Tsitsiklis, J. (1994). Asynchronous stochastic approximation and Q-learning. Mach. Learn., 16:185–202.

[30] Veinott, A. F. (1966). On finding optimal policies in discrete dynamic programming with no discounting. Ann. Math. Stat., 37:1284–1294.

[31] Wan, Y., Naik, A., and Sutton, R. S. (2021a). Average-reward learning and planning with options. In Advances in Neural Information Processing Systems.

[32] Wan, Y., Naik, A., and Sutton, R. S. (2021b). Learning and planning in average-reward Markov decision processes. In International Conference on Machine Learning.

[33] Wan, Y., Yu, H., and Sutton, R. S. (2024). On convergence of average-reward Q-learning in weakly communicating Markov decision processes. arXiv:2408.16262.

[34] Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. PhD thesis, Cambridge University.

[35] White, D. J. (1963). Dynamic programming, Markov chains, and the method of successive approximations. J. Math. Anal. Appl., 6(3):373–376.

[36] Yu, H. and Bertsekas, D. P. (2013a). On boundedness of Q-learning iterates for stochastic shortest path problems. Math. Oper. Res., 38:209–227.

[37] Yu, H. and Bertsekas, D. P. (2013b). Q-learning and policy iteration algorithms for stochastic shortest path problems. Ann. Oper. Res., 208:95–132.

[38] Yu, H., Wan, Y., and Sutton, R. S. (2025). Average-reward reinforcement learning in semi-Markov decision processes via relative value iteration. arXiv:2512.06218.

[39] Yu, H., Wan, Y., and Sutton, R. S. (2026). Asynchronous stochastic approximation with applications to average-reward reinforcement learning. SIAM J. Control Optim., 64:1456–1481. (Extended arXiv version: arXiv:2409.03915).

[40] Zurek, M. and Chen, Y. (2024). Span-based optimal sample complexity for weakly communicating and general average reward MDPs. In Advances in Neural Information Processing Systems.

[41] Zurek, M. and Chen, Y. (2025). The plug-in approach for average-reward and discounted MDPs: Optimal sample complexity analysis. In International Conference on Algorithmic Learning Theory. arXiv:2410.07616.