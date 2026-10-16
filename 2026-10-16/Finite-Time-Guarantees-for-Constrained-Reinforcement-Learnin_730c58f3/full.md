# Finite-Time Guarantees for Constrained Reinforcement Learning under Partial Observability

Valentina Zaccaria<sup>1</sup>, Philip Jordan<sup>∗2</sup>, Tingting Ni<sup>∗2</sup>, Gian Antonio Susto<sup>1</sup>, and Maryam Kargampour<sup>2</sup>

<sup>1</sup>University of Padova <sup>2</sup>SYCAMORE, EPFL

## Abstract

We study reinforcement learning in constrained partially observable Markov decision processes (CPOMDPs), where an agent maximizes reward subject to a cost constraint using only incomplete information about the underlying state. The lack of state observability may require optimal policies to depend on arbitrarily long observation-action histories, making planning and learning challenging. We leverage recent advances in finite-memory approximations of POMDPs and represent finite history windows as states of a constrained MDP. We then develop a primal–dual algorithm that provides the first non-asymptotic guarantees that jointly bound average reward suboptimality and constraint violation in the original CPOMDP. A key challenge is that this finite-window approximation afects both reward values and the set of feasible policies. Under a filter-stability assumption, we derive bounds relating optimality and feasibility in the finite-state model to those in the original CPOMDP, with approximation errors that decay geometrically with the window length. Experiments in partially observable gridworlds demonstrate the efectiveness of our approach.

## 1 Introduction

Partially Observable Markov Decision Processes (POMDPs) model sequential decision-making prob lems in which an agent seeks to maximize its expected cumulative reward without directly observing the state of the environment. Instead, the agent acts on observations that provide only partial information about the underlying state. For example, a robot may not know the exact positions of surrounding objects and must instead rely on noisy camera observations (Lauri et al., 2022). Despite the success of POMDPs in domains such as robotics and healthcare (Ding et al., 2018; Yu et al., 2021), maximizing reward alone does not ensure safe behavior, such as collision avoidance in robotics.

Incorporating safety constraints into POMDPs leads to Constrained Partially Observable Markov Decision Processes (CPOMDPs), where agents maximize expected cumulative reward subject to constraints under partial observability. Despite their practical relevance and successful empirical applications (Hogewind et al., 2022; Carr et al., 2023; Huang et al., 2025), theoretical guarantees for learning in CPOMDPs remain limited.

Existing theoretical work has focused primarily on planning with explicit knowledge of the model, providing guarantees on feasibility (Undurti and How, 2010), approximate optimality and feasibility in time polynomial in the model size for any fixed horizon (Khonji et al., 2019), and finite-iteration bounds on reward optimality and constraint satisfaction (Kalagarla et al., 2022). Other work proposes an online Monte Carlo planning method that uses a black-box generative simulator and provides an asymptotic convergence result under additional assumptions (Lee et al., 2018). However, these methods do not provide finite-time guarantees for both reward optimality and constraint satisfaction when the model is unknown, and the agent learns solely through interaction with the environment.

Addressing this gap requires overcoming a fundamental dificulty inherited from POMDPs. Since the agent receives only partial observations of the underlying state, an optimal policy need not be Markovian in the current observation and may depend on the entire history of observations and actions (Kaelbling et al., 1998). This history dependence makes both planning and learning challenging. Even with a known model, planning is computationally intractable in general (Pa padimitriou and Tsitsiklis, 1987); with an unknown model, learning a near-optimal policy requires exponentially many samples in the worst case (Krishnamurthy et al., 2016).

To obtain tractable solutions, recent work on POMDPs approximates the full interaction history using a finite window of recent observations and actions (Kara and Yuksel, 2022; Kara and Y¨uksel, 2023; Cayci et al., 2024; Anjarlekar et al., 2025; Jordan and Kamgarpour, 2026). In particular, Anjarlekar et al. (2025) represent truncated histories as states of a superstate Markov Decision Process $( M D P )$ , and, under a uniform filter-stability assumption, show that the superstate MDP approximates the original POMDP, with the approximation error decreasing with the history length. They also develop an algorithm with finite-time guarantees for learning a near-optimal policy. Building on the same finite-window approximation, Jordan and Kamgarpour (2026) study model-based learning in the induced superstate MDP for the tabular setting and establish improved sample complexity bounds. These results provide a foundation for learning with finite histories, but address only unconstrained reward maximization.

We extend finite-history methods to CPOMDPs by representing finite history windows as states of a superstate Constrained Markov Decision Process (CMDP). Our goal is to learn policies in this model while guaranteeing reward optimality and constraint satisfaction in the original CPOMDP. This extension requires addressing three technical challenges. First, history truncation afects both reward and cost values. Thus, a policy that is feasible in the superstate CMDP may violate a constraint in the original CPOMDP. Second, comparing the optimal values of the two constrained problems requires more than bounding the error in individual policy values, since history truncation changes the set of feasible policies. Third, values in the superstate CMDP must be estimated from trajectories generated by the original CPOMDP, which introduces a mismatch that the learning analysis must account for.

Contributions. Our contributions are as follows.

1. Under a uniform filter stability condition, we establish bounds relating policy values and constrained optima in the superstate CMDP to those in the original CPOMDP, quantifying the efects of history truncation on optimality and feasibility (Propositions 1 and 2).

2. We develop a primal-dual algorithm (Algorithm 1) and establish finite-time bounds on its average optimality gap and average constraint violation in the original CPOMDP, accounting for the estimation error arising when learning from CPOMDP trajectories (Theorem 1).

3. We evaluate our algorithm on a partially observable Gridworld and observe convergence across diferent history lengths, in accordance with our analysis (Section 6).

Notation. Let R be the set of real numbers, $\mathbb { N } : = \{ 1 , 2 , \dots \}$ the set of natural numbers, and $ { \mathbb { N } } _ { 0 } : =  { \mathbb { N } } \cup \{ 0 \}$ . For a finite set $x ,$ , let |X| denote its cardinality and $\Delta ( \mathcal { X } )$ the probability simplex over $\mathcal { X } .$ For $p , q \in \Delta ( \mathcal { X } )$ , the total variation (TV) distance is defined as $\begin{array} { r } { \| p - q \| _ { \mathrm { T V } } : = \frac { 1 } { 2 } \sum _ { x \in \mathcal { X } } | p ( x ) - q ( x ) | } \end{array}$ The indicator $\mathbf { 1 } \{ x \in \mathcal { X } \}$ equals 1 if $x \in \mathcal { X }$ and 0 otherwise. We denote the concatenation of two sequences $x = ( x _ { 1 } , \ldots , x _ { n } ) $ and $y = ( y _ { 1 } , \dots , y _ { m } )$ by $x \| y : = ( x _ { 1 } , \ldots , x _ { n } , y _ { 1 } , \ldots , y _ { m } )$ . Finally, for $x \in \mathbb { R }$ , we write $[ x ] _ { + } : = \operatorname* { m a x } ( 0 , x )$

## 2 CPOMDP

We consider an infinite-horizon discounted CPOMDP, defined by the tuple

$$
\mathcal { P } = ( \mathcal { S } , \mathcal { A } , \mathcal { O } , \mathbb { P } ^ { o } , \mathbb { O } ^ { o } , r ^ { o } , c ^ { o } , \gamma , d ^ { o } , \rho ^ { o } ) ,
$$

where $s , A$ and O denote finite state, action, and observation spaces, respectively. The transition kernel is denoted by $\mathbb { P } ^ { o } : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ , where $\mathbb { P } ^ { o } ( \cdot \mid s , a )$ specifies the distribution of the next state given the current state and action. Similarly, the observation kernel is denoted by $\mathbb { O } ^ { o } : \mathcal { S }  \Delta ( \mathcal { O } )$ where $\mathbb { O } ^ { o } ( \cdot \mid s )$ defines the distribution over observations given the current state. The functions $r ^ { o } , c ^ { o } : \mathcal { S } \times \mathcal { A }  [ 0 , 1 ]$ are the reward and cost functions, respectively. Moreover, $\gamma \in [ 0 , 1 )$ is the discount factor, $d ^ { o } \in [ 0 , 1 / ( 1 - \gamma ) ]$ is the constraint threshold on the expected discounted cumulative cost, and $\rho ^ { o } \in \Delta ( \mathcal { S } )$ is the initial state distribution.

![](images/1e89ecff9ed6a72c061a8699328df4cdb8105d37b7701552f17c5fe9bc3097ad.jpg)  
Figure 1: Comparison between the CPOMDP P (top) and the superstate CMDP M<sup>g</sup> (bottom). The truncation operator $\mathbb { G } _ { l }$ maps the full history $h _ { t }$ to the superstate $g _ { t }$ by retaining the length-l sufix, which is then treated as the observed state of $\mathcal { M } ^ { g }$

The interaction begins by sampling an initial state $s _ { 0 } \sim \rho ^ { o }$ and an initial observation $o _ { 0 } \sim \mathbb { O } ^ { o } ( \cdot \mid s _ { 0 } )$ At each timestep $t \in  { \mathbb { N } } _ { 0 }$ , the agent selects an action $a _ { t } \in { \mathcal { A } }$ and receives the reward $r ^ { o } ( s _ { t } , a _ { t } )$ and cost $c ^ { o } ( s _ { t } , a _ { t } )$ . The latent state then transitions to $s _ { t + 1 } \sim \mathbb { P } ^ { o } ( \cdot \mid s _ { t } , a _ { t } )$ , and the agent receives a new observation $o _ { t + 1 } \sim \mathbb { O } ^ { o } ( \cdot \mid s _ { t + 1 } )$

Histories and policies. Because the latent states are not directly observable, the agent selects actions based on the observation-action history. For each $t \in  { \mathbb { N } } _ { 0 }$ , let $\mathcal { H } _ { t } = \mathcal { O } \times ( \mathcal { A } \times \mathcal { O } ) ^ { t }$ denote the set of possible histories at timestep t. An element of $\mathcal { H } _ { t }$ is written as $h _ { t } : = ( o _ { 0 } , a _ { 0 } , o _ { 1 } , \ldots , a _ { t - 1 } , o _ { t } )$ for $t \geq 1$ , and $h _ { 0 } : = o _ { 0 }$ . We let $\textstyle { \mathcal { H } } : = \bigcup _ { t > 0 } { \mathcal { H } } _ { t }$ . A history-dependent policy is a sequence of mappings

$$
\pi = \{ \pi _ { t } \} _ { t \geq 0 } , \quad \pi _ { t } : \mathcal { H } _ { t } \to \Delta ( \mathcal { A } ) ,
$$

and we denote the class of all such policies by $\Pi ^ { \mathrm { h i s t } }$

Given a policy $\pi \in \Pi ^ { \mathrm { h i s t } }$ , for any $u \in \{ r ^ { o } , c ^ { o } \}$ , we define the corresponding value function as the expected discounted sum of the stage-wise u,

$$
V _ { u } ^ { \pi } ( \rho ^ { o } ) : = \mathbb { E } _ { \pi , \mathbb { P } ^ { o } , \mathbb { Q } ^ { o } , \rho ^ { o } } \left[ \sum _ { { t = 0 } } ^ { \infty } \gamma ^ { t } u ( s _ { t } , a _ { t } ) \right] .
$$

Objective. Our goal is to find a policy $\pi \in \Pi ^ { \mathrm { h i s t } }$ that maximizes $V _ { r ^ { o } } ^ { \pi } ( \rho ^ { o } )$ subject to a constraint on $V _ { c ^ { o } } ^ { \pi } ( \rho ^ { o } )$

$$
\operatorname* { m a x } _ { \pi \in \Pi ^ { \mathrm { h i s t } } } ~ V _ { r ^ { o } } ^ { \pi } ( \rho ^ { o } ) \quad \mathrm { s . t . } ~ V _ { c ^ { o } } ^ { \pi } ( \rho ^ { o } ) \leq d ^ { o } .\tag{1}
$$

If Problem (1) is feasible (e.g., under Assumption 2 below), there exists an optimal history-dependent policy (see Appendix B). We denote such an optimal solution to Problem (1) by $\pi ^ { \star , h }$

Learning an optimal policy for Problem (1) is challenging, as, even without constraints, optimal policies may depend on the full observation-action history, whose length grows unbounded over time Kaelbling et al. (1998).

## 3 Finite-window Approximation

Our goal is to construct a finite-state CMDP that approximates the original CPOMDP P. Since near-optimal policies of a finite-state CMDP can be found tractably, we then show that, under a suitable condition on the transition and observation kernels of P, such policies are approximately optimal and approximately feasible in P. To this end, in Subsection 3.1, we define the finite-state CMDP, whose states are finite windows of recent observations and actions (Kara and Yuksel, 2022; Anjarlekar et al., 2025). In Subsection 3.2, we present a suficient condition ensuring that policies in this CMDP have reward and cost values close to those in $\mathcal { P } _ { \cdot }$ , and we quantify how the resulting approximation error depends on the window length l.

## 3.1 Superstate CMDP

Fix $l \in  { \mathbb { N } } _ { 0 }$ . For any history $h _ { t } \in \mathcal { H } _ { t }$ , its superstate is defined as the sufix of $h _ { t }$ containing the $l + 1$ most recent observations and l most recent actions,

$$
g _ { t } : = \mathbb { G } _ { l } ( h _ { t } ) = \left\{ { \begin{array} { l l } { ( o _ { t - l } , a _ { t - l } , \ldots , a _ { t - 1 } , o _ { t } ) , } & { t \geq l , } \\ { h _ { t } , } & { t < l , } \end{array} } \right.
$$

where $\begin{array} { r } { \mathbb G _ { l } : \mathcal { H } \to \bigcup _ { k < l } \mathcal { H } _ { k } } \end{array}$ is the truncation operator.

The superstate CMDP has state space $\textstyle { \mathcal { G } } : = \bigcup _ { k < l } { \mathcal { H } } _ { k }$ and action space ${ \mathcal { A } } ,$ , both of which are finite. To define its transition, reward, and cost functions, we associate with each superstate $g \in { \mathcal { G } }$ the finite-history belief $b ^ { g } \in \Delta ( { \mathcal { S } } )$ , given by $b ^ { g } ( s ) : = \operatorname* { P r } ( s \mid g , \rho ^ { o } )$ . We give its precise definition and computation in Appendix A. Note that b<sup>g</sup> is in general only an approximation of the exact posterior over the current latent state, which depends on the full history preceding the window, and we discuss the resulting approximation error in Subsection 3.2.

Transition dynamics. For $g , g ^ { \prime } \in \mathcal { G }$ and $a \in { \mathcal { A } } .$

$$
\mathbb { P } ^ { g } ( { \boldsymbol { g } } ^ { \prime } \mid { \boldsymbol { g } } , { \boldsymbol { a } } ) : = \sum _ { o \in \mathcal { O } } \mathbf { 1 } \big ( { \boldsymbol { g } } ^ { \prime } = \mathbb { G } _ { l } ( { \boldsymbol { g } } \parallel ( { \boldsymbol { a } } , { \boldsymbol { o } } ) ) \big ) \operatorname* { P r } ( { \boldsymbol { o } } \mid { \boldsymbol { b } } ^ { g } , { \boldsymbol { a } } ) ,\tag{2}
$$

where $\begin{array} { r } { \operatorname* { P r } ( o \mid b ^ { g } , a ) : = \sum _ { s . s ^ { \prime } \in S } \mathbb { O } ^ { o } ( o \mid s ^ { \prime } ) \mathbb { P } ^ { o } ( s ^ { \prime } \mid s , a ) b ^ { g } ( s ) } \end{array}$ . The initial superstate distribution is $\begin{array} { r } { \rho ^ { g } ( g ) : = \mathbf { 1 } ( g \in \mathcal { H } _ { 0 } ) \sum _ { s \in { \mathcal { S } } } \mathbb { O } ^ { o } ( g \mid s ) \rho ^ { o } ( s ) } \end{array}$ , where we identify $\mathcal { H } _ { \mathrm { 0 } }$ with $\mathcal { O }$

Reward and cost. The superstate reward and cost are given by $\begin{array} { r } { r ^ { g } ( g , a ) : = \sum _ { s \in \mathcal { S } } r ^ { o } ( s , a ) b ^ { g } ( s ) } \end{array}$ and $\begin{array} { r } { c ^ { g } ( g , a ) : = \sum _ { s \in \mathcal { S } } c ^ { o } ( s , a ) b ^ { g } ( s ) } \end{array}$ , respectively. The resulting superstate CMDP is $\begin{array} { l l } { { \mathcal { M } } ^ { g } } & { : = } \end{array}$ $( \mathcal { G } , \mathcal { A } , \mathbb { P } ^ { g } , r ^ { g } , c ^ { g } , \gamma , \rho ^ { g } , \overset { . } { d } ^ { g } )$ , and Figure 1 illustrates how it difers from $\mathcal { P }$

Policies and value functions. A stationary superstate policy is a mapping $\pi : \mathcal { G }  \Delta ( \mathcal { A } )$ . Let $\Pi ^ { g } : = \{ \pi : { \mathcal { G } } \to \Delta ( A ) \}$ denote the set of all stationary Markov superstate policies. Given $\pi \in \Pi ^ { g }$ for any $u \in \{ r ^ { g } , c ^ { g } \}$ , we define the value function

$$
V _ { u } ^ { \pi } ( \rho ^ { g } ) : = \mathbb { E } _ { \pi , \mathbb { P } ^ { g } , \rho ^ { g } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } u ( g _ { t } , a _ { t } ) \right] .
$$

The superstate CMDP problem is given by

$$
\operatorname* { m a x } _ { \pi \in \Pi ^ { g } } V _ { r ^ { g } } ^ { \pi } \big ( \rho ^ { g } \big ) \quad \mathrm { s . t . ~ } V _ { c ^ { g } } ^ { \pi } \big ( \rho ^ { g } \big ) \leq d ^ { g } .\tag{3}
$$

Let $\pi ^ { * }$ denote an optimal policy for Problem (3). The threshold $d ^ { g } \in [ 0 , 1 / ( 1 - \gamma ) ]$ is a design parameter. Since truncating the history perturbs cost values, choosing $d ^ { g }$ below $d ^ { o }$ can help ensure that policies feasible in the superstate CMDP remain approximately feasible in the original CPOMDP. We next quantify this approximation error and derive a condition on $d ^ { g }$ that ensures this feasibility guarantee.

## 3.2 Approximating a CPOMDP with a Superstate CMDP

The finite-history approach replaces the original CPOMDP P with a superstate CMDP M<sup>g</sup>. This section examines how accurately the resulting model captures the original problem under a filterstability condition. First, we bound the diferences in reward and cost values between the two models (Proposition 1). We then use this bound to choose the constraint threshold $d ^ { g }$ for M<sup>g</sup> and compare the optimal reward values of M<sup>g</sup> and P under their respective constraints (Proposition 2).

Both results rely on controlling how much information about the current latent state is lost when the history is truncated. For any history $h \in \mathcal H$ , let $b ^ { h } \in \Delta ( S )$ denote the posterior over current states given $h , \mathrm { i . e . , } b ^ { h } ( s ) : = \operatorname* { P r } ( s \mid h )$ for all $s \in S$

The accuracy of the superstate approximation depends on how close the belief $b ^ { h }$ induced by the full history is to the belief b<sup>g</sup> induced by its superstate $g = \mathbb { G } _ { l } ( h )$ , defined in Section 3.1. Without additional assumptions, observations from the distant past may afect beliefs over the current state, so that the approximation error cannot be controlled in general. We therefore adopt a uniform filter stability assumption, commonly used in the literature on POMDPs (Kara and Y¨uksel, 2023; Anjarlekar et al., 2025; Jordan and Kamgarpour, 2026). Under this assumption, as stated below, the influence of the past on the current belief decays as new actions and observations are appended to the history.

Assumption 1 (Uniform Filter Stability). Given a CPOMDP P, there exists $\kappa \in ( 0 , 1 )$ such that, for all $h , h ^ { \prime } \in \mathcal { H } , a \in \mathcal { A }$ and $o \in \mathcal { O }$

$$
\left\| b ^ { h } \right\| ( a , o )  - b ^ { h ^ { \prime } } \| ( a , o )  \left\| _ { \mathrm { T V } } \leq ( 1 - \kappa ) \right\| b ^ { h } - b ^ { h ^ { \prime } } \left\| _ { \mathrm { T V } } \right.
$$

where $b ^ { h \parallel ( a , o ) }$ and $b ^ { h ^ { \prime } \parallel ( a , o ) }$ are the beliefs obtained by appending the same action–observation pair $( a , o )$ to h and $h ^ { \prime } { } _ { \bot }$ , respectively<sup>1</sup>.

Assumption 1 can be verified for a given CPOMDP via suficient conditions from the literature. E.g., McDonald and Y¨uksel (2020) give a condition in terms of the Dobrushin coeficients of the transition and observation kernels, which holds when state transitions are suficiently mixing relative to the information carried by observations.

Policy Transfer Gap. Our first result is a two-way policy transfer bound between $\mathcal { M } ^ { g }$ and P. In one direction, it bounds the value-function gap for any superstate policy $\pi \in \Pi ^ { g }$ . In the other, it maps any history-dependent policy $\pi \in \Pi ^ { \mathrm { h i s t } }$ to a stationary superstate policy π with approximately matching reward and cost values.

Proposition 1 (Policy Transfer Gap). Let $\begin{array} { r } { \delta ( l ) : = \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } } \end{array}$ denote the approximation error. Under Assumption 1, the following holds.

(a) For any superstate policy $\pi \in \Pi ^ { g }$ and $u \in \{ r , c \}$ 2

$$
| V _ { u ^ { o } } ^ { \pi } ( \rho ^ { o } ) - V _ { u ^ { g } } ^ { \pi } ( \rho ^ { g } ) | \leq \delta ( l ) .
$$

(b) For any policy $\pi \in \Pi ^ { \mathrm { h i s t } }$ , there exists a superstate policy $\pi ^ { g } \in \Pi ^ { g }$ such that, for both $u \in \{ r , c \}$

$$
| V _ { u ^ { o } } ^ { \pi } ( \rho ^ { o } ) - V _ { u ^ { g } } ^ { \pi ^ { g } } ( \rho ^ { g } ) | \leq \delta ( l ) .
$$

We defer the proof to Appendix D.1 and give a brief overview. Part (a) is a Simulation Lemma argument (Kearns and Singh, 2002): under Assumption 1, the rewards, costs and next-observation distributions of the two models are close, which yields a recursion for the worst-case value gap over histories. For Part (b), the same argument bounds the value gap between $\mathcal { P }$ and an auxiliary model in which π still acts on full histories, but rewards, costs, and transitions depend only on the current superstate. In this model, the superstate–action occupancy measure of π is induced by a stationary superstate policy $\pi ^ { g }$ , which therefore attains the same values in $\mathcal { M } ^ { g }$

Constrained Optimal Value Gap. We next bound the loss in constrained optimal reward from solving Problem (3) instead of Problem (1). Specifically, we compare the optimal value $V _ { r ^ { o } } ^ { \pi ^ { \star , h } } ( \rho ^ { o } )$ of P, attained by the optimal history-dependent policy $\pi ^ { \star , h }$ (Section 2), with the optimal value $V _ { r ^ { g } } ^ { \pi ^ { * } } ( \rho ^ { g } )$ of $\mathcal { M } ^ { g }$ , attained by the optimal superstate policy $\pi ^ { * }$

By Proposition 1(b), the optimal history-dependent policy $\pi ^ { \star , h }$ can be mapped to a superstate policy with nearly the same reward and cost values. If the mapped policy is feasible in $\mathcal { M } ^ { g }$ , its reward cannot exceed $V _ { r ^ { g } } ^ { \pi ^ { * } } \left( \rho ^ { g } \right)$ , giving the desired bound. If it is infeasible, its constraint violation is bounded by the approximation error $\delta ( l )$ . To control how much its reward may then exceed $V _ { r ^ { g } } ^ { \pi ^ { * } } ( \rho ^ { g } )$ , we use a standard sensitivity result for constrained optimization. This requires the strict feasibility condition stated below, which is common in the CMDP literature (Ding et al., 2025; Efroni et al., 2020).

Assumption 2 (Slater Condition on P). There exist $\tilde { \pi } \in \Pi ^ { \mathrm { h i s t } }$ and $\xi ^ { o } > 0$ such that $V _ { c ^ { o } } ^ { \tilde { \pi } } ( \rho ^ { o } ) \leq$ $d ^ { o } - \xi ^ { o }$

The assumption ensures that $\mathcal { P }$ has a policy whose cost is at least $\xi ^ { o }$ below the threshold $d ^ { o }$ . By Proposition $1 ( \mathrm { b } )$ , there exists a policy in Π<sup>g</sup> whose cost in $\mathcal { M } ^ { g }$ is at most $d ^ { o } - \xi ^ { o } + \delta ( l )$ . Therefore, if $d ^ { g }$ exceeds this bound, M<sup>g</sup> also satisfies the Slater condition, as formalized below.

Lemma 1 (Slater Condition on $\mathcal { M } ^ { g } )$ . Under Assumptions 1 and 2, if $\dot { \cdot } d ^ { g } > d ^ { o } - \xi ^ { o } + \delta ( l )$ , then there exists $\bar { \pi } \in \Pi ^ { g }$ such that

$$
V _ { c ^ { g } } ^ { \bar { \pi } } ( \rho ^ { g } ) \leq d ^ { g } - \xi ^ { g } ,
$$

where $\xi ^ { g } : = d ^ { g } - d ^ { o } + \xi ^ { o } - \delta ( l ) > 0 .$

The margin $\xi ^ { g } > 0$ bounds the optimal dual multiplier of $\mathcal { M } ^ { g }$ , which in turn controls how much the optimal value of $\mathcal { M } ^ { g }$ can increase when its constraint is relaxed. This yields the following bound.

Proposition 2 (Constrained Optimal Value Gap). Suppose that Assumptions 1 and 2 hold and that $d ^ { g }$ satisfies the condition of Lemma 1. Then,

$$
{ V _ { r ^ { o } } ^ { \pi ^ { \star , h } } } ( \rho ^ { o } ) - { V _ { r ^ { g } } ^ { \pi ^ { * } } } ( \rho ^ { g } ) \le \delta ( l ) + \frac { [ d ^ { o } - d ^ { g } + \delta ( l ) ] _ { + } } { ( 1 - \gamma ) \xi ^ { g } } .\tag{4}
$$

The proof is given in Appendix D.3. The proposition bounds how much the optimal reward in $\mathcal { M } ^ { g }$ can fall below that in $\mathcal { P }$ . In particular, if $d ^ { g } = d ^ { o } - \delta ( l )$ and the history length l is large enough that $\xi ^ { o } > 2 \delta ( l )$ to ensure $\xi ^ { g } > 0$ , this bound converges to zero as l grows.

Choice of $d ^ { g }$ . We seek to learn a policy that is near-optimal and (nearly) feasible in $\mathcal { P } .$ By Proposition $\mathrm { 1 ( a ) }$ , choosing $d ^ { g } = d ^ { o } - \delta ( l )$ ensures that every policy feasible in $\mathcal { M } ^ { g }$ is also feasible in $\mathcal { P }$ . For suficiently large l, Lemma 1 ensures that $\mathcal { M } ^ { g }$ remains strictly feasible. Our results hold more generally for any $d ^ { g }$ satisfying the condition of Lemma 1, which reveals a trade-of between optimality and feasibility. Increasing $d ^ { g }$ reduces the optimality-loss bound in Proposition 2, but may allow a policy feasible in $\mathcal { M } ^ { g }$ to violate the constraint in $\mathcal { P }$ by up to $[ d ^ { g } - d ^ { o } + \delta ( l ) ] _ { + }$ , as follows from Proposition 1(a). Theorem 1 makes this trade-of explicit.

## 4 Algorithm

## 4.1 Lagrangian Formulation

To solve Problem (3), we incorporate the cost constraint into the objective using a nonnegative Lagrange multiplier λ. This yields the following max–min formulation:

$$
\operatorname* { m a x } _ { \pi \in \Pi ^ { g } } \operatorname* { m i n } _ { \lambda \in \mathbb { R } _ { \geq 0 } } L ( \pi , \lambda ) : = V _ { r ^ { g } } ^ { \pi } ( \rho ^ { g } ) - \lambda \big ( V _ { c ^ { g } } ^ { \pi } ( \rho ^ { g } ) - d ^ { g } \big ) .\tag{5}
$$

For a fixed $\lambda ,$ the associated dual function is $D ^ { \lambda } ( \rho ^ { g } ) : = \operatorname* { m a x } _ { \pi \in \Pi ^ { g } } L ( \pi , \lambda )$ , and we denote an optimal dual variable by $\begin{array} { r } { \lambda ^ { * } \in \arg \operatorname* { m i n } _ { \lambda \in \mathbb { R } > 0 } D ^ { \lambda } ( \rho ^ { g } ) } \end{array}$

The following lemma establishes strong duality, thereby justifying the Lagrangian formulation, and provides an explicit upper bound on $\lambda ^ { * }$ . Both statements follow from standard CMDP arguments (Altman, 2021; Ding et al., 2025), which apply to $\mathcal { M } ^ { g }$ since it satisfies the Slater condition by Lemma 1; for completeness, we include a proof in Appendix D.2.

Lemma 2. Suppose that Assumptions 1 and 2 hold and that $d ^ { g }$ satisfies the condition of Lemma 1. Then,

$$
( \mathrm { i } ) V _ { r ^ { g } } ^ { \pi ^ { * } } ( \rho ^ { g } ) = D ^ { \lambda ^ { * } } ( \rho ^ { g } ) , \quad ( \mathrm { i i } ) \lambda ^ { * } \le \lambda ^ { m a x } ,
$$

where $\lambda ^ { m a x } : = 2 ( \xi ^ { g } ( 1 - \gamma ) ) ^ { - 1 } \mathopen { } \mathclose \bgroup ($

Part (i) shows that the Lagrangian formulation in (5) and Problem (3) have the same optimal value. Part (ii) shows that the search for the optimal dual variable $\lambda ^ { * }$ can be restricted to a compact interval $[ 0 , \lambda ^ { \mathrm { m a x } } ]$ . Together, these results motivate a primal–dual method that alternates between improving the policy and performing a projected update of the Lagrange multiplier. We first present these two updates assuming access to the exact value functions of the current policy $\pi ^ { k }$ in $\mathcal { M } ^ { g }$ namely $Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ and $V _ { c ^ { g } } ^ { \pi ^ { k } } \left( \rho ^ { g } \right)$ (defined below), and then introduce the estimation oracle used to implement them.

Policy update. For a fixed multiplier $\lambda ^ { k }$ , maximizing the Lagrangian over policies is equivalent to maximizing $V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi } \left( \rho ^ { g } \right)$ under the modified reward $r ^ { g } - \lambda ^ { k } c ^ { g }$ . We define the state-action value function of $\pi ^ { k }$ for this reward as

$$
Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) : = \mathbb { E } _ { \pi ^ { k } , \mathbb { P } ^ { g } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } ( r ^ { g } - \lambda ^ { k } c ^ { g } ) ( g _ { t } , a _ { t } ) \Bigg | g _ { 0 } = g , \ a _ { 0 } = a \right] .
$$

We then update the policy using exponential weights:

$$
\pi ^ { k + 1 } ( a \mid g ) \propto \pi ^ { k } ( a \mid g ) \exp \Bigl ( \eta _ { 1 } Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) \Bigr ) ,\tag{6}
$$

where the stepsize $\eta _ { 1 } > 0$ will be specified in Theorem 1. This is called Politex update (Abbasi-Yadkori et al., 2019).<sup>2</sup> When the multiplier is fixed, repeatedly applying this update yields an average optimality gap that decreases to zero at a sublinear rate for the corresponding modified reward.

Dual update. Given the current policy $\pi ^ { k } .$ we update the multiplier to penalize violations of the cost constraint. Since the dual problem minimizes the Lagrangian with respect to λ, we use the projected gradient descent step

$$
\lambda ^ { k + 1 } = \operatorname { P r o j } _ { [ 0 , \lambda ^ { \operatorname { m a x } } ] } \bigl ( \lambda ^ { k } - \eta _ { 2 } ( d ^ { g } - V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) ) \bigr ) ,\tag{7}
$$

where the stepsize $\eta _ { 2 } > 0$ will be specified in Theorem 1. When the current policy violates the constraint, $\mathrm { i . e . , ~ } V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) > d ^ { g }$ , the multiplier increases and imposes a larger cost penalty in the next policy update; otherwise, it decreases.

## 4.2 Estimation Oracle

The policy update in (6) requires the state-action values $Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } \left( g , a \right)$ , whereas the dual update in (7) requires the cost value $V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } )$ . Since neither quantity is directly available, we assume access to an oracle that estimates both using trajectories generated by CPOMDP P.

Estimation Oracle B. There exists $\delta _ { \mathrm { { e s t } } } \geq 0$ such that, at every iteration k, for any policy $\pi ^ { k } \in \Pi ^ { g }$ $\lambda ^ { k } \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ , and $( g , a ) \in \mathcal { G } \times \mathcal { A }$ , the oracle B returns estimates $\hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } \left( g , a \right)$ and $\hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } )$ satisfying

$$
\begin{array} { r l r } & { } & { \mathbb { E } \Big [ \Big | Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) - \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) \Big | \Big ] \leq \delta _ { \mathrm { e s t } } , } \\ & { } & { \mathbb { E } \Big [ \Big | V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) - \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) \Big | \Big ] \leq \delta _ { \mathrm { e s t } } , } \end{array}\tag{8}
$$

where the expectations are taken over the randomness $o f$ the estimation procedure.

The oracle abstraction allows us to accommodate diferent policy evaluation procedures. It can be instantiated using either the model-based method Jordan and Kamgarpour (2026) or the modelfree TD policy evaluation method Anjarlekar et al. (2025) (details in Appendix E). For these two instantiations, the estimation error $\delta _ { \mathrm { e s t } }$ has two sources: estimating values from finite-length trajectories and the mismatch between the full-history dynamics and the finite-history superstate model. Under Assumption 1, the latter error decreases as the window length l increases for both methods.

Note that exact values $Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ and $V _ { c ^ { g } } ^ { \pi ^ { k } }$ lie in the intervals

$$
\left[ - \frac { \lambda ^ { \operatorname* { m a x } } } { 1 - \gamma } , \frac { 1 } { 1 - \gamma } \right] \quad \mathrm { a n d } \quad \left[ 0 , \frac { 1 } { 1 - \gamma } \right] ,
$$

respectively. Before using their estimates in the policy and dual updates, we therefore clip $\hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ and $\hat { V } _ { c ^ { g } } ^ { \pi ^ { k } }$ to the corresponding intervals. This keeps estimates bounded without increasing estimation errors.

Combining the preceding steps, we obtain Algorithm 1 for learning an optimal policy for the original CPOMDP. At each iteration, the algorithm updates the policy using Politex and the dual variable using projected gradient descent, based on the value estimates provided by Estimation Oracle B.

Algorithm 1 Primal–Dual Politex   
1: Input: stepsizes $\eta _ { 1 } , \eta _ { 2 } > 0 ;$ number of iterations K, constraint threshold $d ^ { g } .$ , projection set   
$[ 0 , \lambda ^ { \mathrm { m a x } } ]$ , and Estimation Oracle B   
2: Initialize: $\pi ^ { 0 } ( a \mid g ) = 1 / \vert { \mathcal A } \vert$ for all $( g , a ) \in \mathcal { G } \times \mathcal { A } ,$ , and $\lambda ^ { 0 } = 0$   
3: for $k = 0 , \ldots , K - 1$ do   
4: Get ${ \hat { Q } } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a )$ , ∀g, a and $\hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } )$ from B   
5: for $( g , a ) \stackrel { \cdot \cdot } { \in } \mathcal { G } \times \mathcal { A }$ do   
6: $\pi ^ { k + 1 } ( a | g ) \propto \pi ^ { k } ( a | g ) \exp \Bigl ( \eta _ { 1 } \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) \Bigr )$   
7: end for   
8: $\lambda ^ { k + 1 } = \operatorname { P r o j } _ { [ 0 , \lambda ^ { \mathrm { m a x } } ] } \Bigl ( \lambda ^ { k } + \eta _ { 2 } ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) - d ^ { g } ) \Bigr )$   
9: end for

Remark. Since $| \mathcal { G } | = O ( | \mathcal { O } | ^ { l + 1 } | A | ^ { l } )$ , in each iteration, the tabular policy update requires time and memory exponential in the window length $l ;$ this reflects the trade-of between approximation accuracy and computational cost. A similar trade-of is encountered in prior work on finite-window methods Cayci et al. (2024); Anjarlekar et al. (2025); Jordan and Kamgarpour (2026).

## 5 Theoretical Guarantees

In this section, we evaluate the performance of Algorithm 1 in the original CPOMDP using the average optimality gap and average constraint violation, defined as follows:

$$
\begin{array} { r l } & { \mathrm { O p G a p } _ { K } : = \displaystyle \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } \Big [ { V } _ { r ^ { o } } ^ { \pi ^ { \star , h } } ( { \rho ^ { o } } ) - { V } _ { r ^ { o } } ^ { \pi ^ { k } } ( { \rho ^ { o } } ) \Big ] , } \\ & { \mathrm { C V i o l } _ { K } : = \displaystyle \Bigg [ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } \Big [ { V } _ { c ^ { o } } ^ { \pi ^ { k } } ( { \rho ^ { o } } ) - d ^ { o } \Big ] \Bigg ] _ { + } , } \end{array}
$$

where the expectations are over the randomness of the estimation oracle, and $\{ \pi ^ { k } \} _ { k = 0 } ^ { K - 1 }$ denotes the policies generated by Algorithm 1 over K iterations.

Theorem 1. Given Estimation Oracle B and under Assumptions 1 and ${ \it 2 } ,$ we run Algorithm 1 for K iterations with constraint threshold $d ^ { g } > d ^ { o } - \xi ^ { o } + \delta ( l )$ and stepsizes

$$
\begin{array} { l } { \eta _ { 1 } : = \displaystyle \frac { ( 1 - \gamma ) ^ { 2 } \xi ^ { g } } { ( 1 - \gamma ) \xi ^ { g } + 2 } \sqrt { \displaystyle \frac { 2 \log | { \cal A } | } { K } } , } \\ { \eta _ { 2 } : = \displaystyle \frac { 2 } { \xi ^ { g } ( 1 + ( 1 - \gamma ) d ^ { g } ) \sqrt { K } } . } \end{array}
$$

Then,

$$
\begin{array} { r l } & { \mathrm { O p G a p } _ { K } \leq \mathcal { O } \biggl ( \delta ( l ) + \frac { \delta _ { \mathrm { e s t } } + [ d ^ { o } - d ^ { g } + \delta ( l ) ] _ { + } } { \xi ^ { g } } \biggr ) + \mathcal { O } \biggl ( \frac { \sqrt { \log | A | } } { \xi ^ { g } \sqrt { K } } \biggr ) , } \\ & { \mathrm { C V i o l } _ { K } \leq \mathcal { O } ( [ d ^ { g } - d ^ { o } + \delta ( l ) ] _ { + } + \delta _ { \mathrm { e s t } } ) + \mathcal { O } \biggl ( \frac { \sqrt { \log | A | } } { \sqrt { K } } \biggr ) . } \end{array}
$$

The complete bounds are given in Appendix C and full proof in Appendix D.

The bounds combine three sources of error. First, the approximation error $\delta ( l )$ decays geometrically with l, consistent with finite-history analyses of unconstrained POMDPs (Anjarlekar et al., 2025; Jordan and Kamgarpour, 2026). Second, the bounds scale linearly with the estimation error $\delta _ { \mathrm { e s t } }$ specified in (8). Third, the optimization error from running finitely many primal–dual iterations decays as $\mathcal { O } ( \sqrt { \log | \mathcal { A } | / K } )$ , matching rates for primal-dual methods in CMDPs (Ding et al., 2025).

The threshold $d ^ { g }$ trades optimality for feasibility. Lowering $d ^ { g }$ reduces $[ d ^ { g } - d ^ { o } + \delta ( l ) ] _ { - }$ in the bound on $\operatorname { C V i o l } _ { K }$ , but increases $[ d ^ { o } - d ^ { g } + \delta ( l ) ] .$ <sub>+</sub> and reduces the Slater margin $\xi ^ { g }$ (Lemma 1); both efects loosen the bound on OpGap .

To our knowledge, this is the first work to provide non-asymptotic learning guarantees that jointly bound average reward suboptimality and constraint violation in CPOMDPs. Since these guarantees hold on average across iterates, they do not ensure per-iteration feasibility or final-policy performance. Establishing last-iterate guarantees with safe exploration is a key future direction Ni and Kamgarpour (2025); Ding et al. (2023).

## 5.1 Proof Sketch

We split $\mathrm { O p G a p } _ { K }$ and $\operatorname { C V i o l } _ { K }$ into model approximation terms, controlled by the results of Section 3.2, and optimization terms in ${ \mathcal { M } } ^ { g } ;$ , controlled by a primal–dual analysis based on (Ding et al., 2025), combined with an exponential weights regret bound. Throughout, we omit the dependence on the initial distributions $\rho ^ { o } , \rho ^ { g }$ for simplicity.

We decompose the average optimality gap as

$$
\mathrm { O p G a p } _ { K } = \underbrace { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { k } } - V _ { r ^ { o } } ^ { \pi ^ { k } } ] } _ { ( i ) } + \underbrace { V _ { r ^ { o } } ^ { \pi ^ { \star , h } } - V _ { r ^ { g } } ^ { \pi ^ { * } } } _ { ( i i ) } + \underbrace { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } ] } _ { ( i i i ) } ,
$$

and the constraint violation as

$$
\mathrm { C V i o l } _ { K } = \Bigg [ d ^ { g } - d ^ { o } + \underbrace { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { o } } ^ { \pi ^ { k } } - V _ { c ^ { g } } ^ { \pi ^ { k } } ] } _ { ( i v ) } + \underbrace { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] } _ { ( v ) } \Bigg ] _ { + } .
$$

Terms (i) and $( i v )$ are policy evaluation gaps between $\mathcal { P }$ and $\mathcal { M } ^ { g }$ , each bounded by $\delta ( l )$ by Proposition 1(a). Term (ii) is the gap between the optimal reward values of the two problems, bounded by Proposition 2. Terms (iii) and (v) are the average reward gap and the average constraint violation of the iterates in $\mathcal { M } ^ { g }$ . To control them, we first prove an average Lagrangian performance bound.

Lemma 3 (Bounded Average Performance). Let $\pi ^ { * }$ be an optimal policy for the superstate $\mathcal { M } ^ { g }$ $I f$ the oracle B satisfies inequalities in (8), after running Algorithm 1 for K iterations, it holds

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( V _ { r ^ { g } } ^ { \pi ^ { * } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { * } } ) - ( V _ { r ^ { g } } ^ { \pi ^ { k } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { k } } ) ] \le \delta _ { \mathrm { l a g } } ,
$$

$$
\begin{array} { r } { w h e r e ~ \delta _ { \mathrm { l a g } } : = \frac { 2 \delta _ { \mathrm { e s t } } } { 1 - \gamma } + \frac { 1 + \lambda ^ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \frac { 2 \log | \mathcal { A } | } { K } } . } \end{array}
$$

The proof uses the Performance Diference Lemma (Kakade and Langford, 2002) to express the Lagrangian value gap through $Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ . Adding and subtracting estimates $\hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ decomposes it into two estimation error terms, bounded by (8), and a statewise term comparing cumulative estimated Lagrangian Q-values under $\pi ^ { * }$ and the iterates $\pi ^ { k }$ . The latter is bounded by the exponential weights regret bound (Cesa-Bianchi and Lugosi, 2006), and choosing η<sub>1</sub> appropriately yields $\delta _ { \mathrm { l a g } }$ . Unlike the telescoping KL-divergence argument of Ding et al. (2025) and the highprobability analysis of Politex (Abbasi-Yadkori et al., 2019), this argument applies the regret bound pathwise and bounds the estimation errors directly in expectation. The full proof is in Appendix D.4.

Optimality. Adding and subtracting $\lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { * } } - V _ { c ^ { g } } ^ { \pi ^ { k } } )$ gives

$$
( i i i ) = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { * } } - V _ { c ^ { g } } ^ { \pi ^ { k } } ) ] + \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( V _ { r ^ { g } } ^ { \pi ^ { * } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { * } } ) - ( V _ { r ^ { g } } ^ { \pi ^ { k } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { k } } ) ] .
$$

![](images/091510a66e4c78199e5798b4a5476547520094aea914fa86671b1f2d276bb2c3.jpg)  
Figure 2: Gridworld: Green cell is the goal, orange cells are unsafe, and $\times$ is initial position. Arrows are an approximately optimal policy computed using SARSOP Kurniawati et al. (2008), with lengths proportional to action probabilities.

By telescoping the projected dual update, using non-expansiveness of the projection, feasibility of $\pi ^ { * }$ , and (8), the first sum is at most $\begin{array} { r } { \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } } \end{array}$ . The second sum is at most $\delta _ { \mathrm { l a g } }$ by Lemma 3. Hence

$$
( i i i ) \le \delta _ { \mathrm { l a g } } + \frac { \eta _ { 2 } } { 2 } \biggl ( \frac { 1 } { 1 - \gamma } + d ^ { g } \biggr ) ^ { 2 } + \lambda ^ { \mathrm { m a x } } \delta _ { \mathrm { e s t } } ,\tag{9}
$$

and substituting η<sub>2</sub> gives the bound in Theorem 1. The detailed proof is given in Appendix D.5.

Constraint. By Proposition $\mathrm { 1 ( a ) }$ , combined with the fact that $[ x + y ] + \leq [ x ] + [ y ] +$ , we have

$$
\mathrm { C V i o l } _ { K } \leq [ d ^ { g } - d ^ { o } + \delta ( l ) ] _ { + } + [ ( v ) ] _ { + } .\tag{10}
$$

For any fixed $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ , telescoping the projected dual update from $\lambda ^ { 0 } = 0$ and using (8) gives

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( { V _ { c } ^ { \pi ^ { k } } } - d ^ { g } ) ] \le \frac { \lambda ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \bigg ( \frac { 1 } { 1 - \gamma } + d ^ { g } \bigg ) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } ,
$$

while Lemma 3 and feasibility of $\pi ^ { * }$ give $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } + \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] \leq \delta _ { \mathrm { l a g } } } \end{array}$ . Adding the two cancels the $\lambda ^ { k }$ terms and, using $\lambda \le \lambda ^ { \mathrm { m a x } }$ , yields

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } ] + \lambda \cdot ( v ) \le \varepsilon _ { c } ,
$$

where $\varepsilon _ { c }$ collects $\delta _ { \mathrm { l a g } }$ and the right-hand side above with $\lambda = \lambda ^ { \mathrm { m a x } }$ . Maximizing over $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ turns $\lambda \cdot ( v )$ into $\lambda ^ { \operatorname* { m a x } } [ ( v ) ] _ { + }$ . Since $\lambda ^ { \mathrm { m a x } } \ge 2 \lambda ^ { * }$ by Lemma 2, Lemma 6 (adapted from Ding et al. (2025, Lemma 5)) gives $[ ( v ) ] _ { + } \leq ( 1 - \gamma ) \xi ^ { g } \varepsilon _ { c }$ . Substituting this and $\eta _ { 2 }$ into (10) yields the stated result. Full proof is in Appendix D.6.

## 6 Simulations

To illustrate how the algorithm’s performance depends on K and l in a controlled setting, we evaluate it on a $7 \times 7$ gridworld adapted from Sutton et al. (1998) and shown in Figure 2. The latent state is the agent’s position in the grid, and the action set is {up, right, down, left}. Both the transition and observation dynamics are stochastic. Instead of observing its position directly, the agent receives one of five observations: $O = \{ \mathsf { s a f e } _ { \mathsf { i } }$ , edge, near-hazard, unsafe, goal}. Because multiple grid cells can generate the same observation, the observation-action history does not generally identify the latent state. The agent receives a reward of 1 at the absorbing goal state and 0 elsewhere, and incurs a cost of 1 in unsafe states and 0 elsewhere. Accordingly, the agent seeks to reach the goal as quickly as possible while limiting the expected discounted number of visits to unsafe cells, despite observing the environment only partially.

Figure 3 shows the performance of our algorithm across diferent history lengths l and iterations K. It uses an exact policy-evaluation oracle in the superstate CMDP to isolate the efects of l and

![](images/f036c407a60d95f4d644b282693ba1b832af655c888f7325969d48e09792bb31.jpg)  
Figure 3: Algorithm performance for diferent history lengths l. Solid curves show the mean, shaded regions show the evaluation uncertainty.

K from the estimation error. As K increases, both the average optimality gap and the average constraint violation approach a neighborhood of zero. Consistent with our theoretical results, the size of this neighborhood is governed by the approximation error $\delta ( l )$ : increasing l reduces the approximation error and improves performance.

In Appendix $\mathrm { F } ,$ we replace the exact oracle with a sample-based TD(0) oracle, we examine sensitivity to step sizes and the constraint threshold $d ^ { g }$ , and the performance loss due to partial observability. We provide complete environment and implementation details.

## 7 Conclusion & Future Work

We addressed RL in CPOMDPs by representing finite observation-action windows as states of a superstate CMDP. We quantified how this approximation afects both reward values and the set of feasible policies, showing that the resulting errors decay geometrically with the window length under filter stability. Building on these bounds, we developed a primal–dual algorithm with non-asymptotic guarantees on average reward suboptimality and constraint violation in the original CPOMDP. Experiments on a partially observable constrained Gridworld supported our theoretical results.

Our analysis is limited to tabular superstate CMDPs and our experiments to Gridworld. Extensions to continuous spaces, realistic benchmarks, and a weaker uniform filter-stability assumption remain for future work.

## References

Abbasi-Yadkori, Y., Bartlett, P., Bhatia, K., Lazic, N., Szepesvari, C., and Weisz, G. (2019). Politex: Regret bounds for policy iteration using expert prediction. In International conference on machine learning, pages 3692–3702. PMLR.

Altman, E. (2021). Constrained Markov decision processes. Routledge.

Anjarlekar, A., Etesami, R., and Srikant, R. (2025). Scalable policy-based rl algorithms for POMDPs. arXiv preprint arXiv:2510.06540.

Carr, S., Jansen, N., Junges, S., and Topcu, U. (2023). Safe reinforcement learning via shielding under partial observability. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 14748–14756.

Cayci, S., He, N., and Srikant, R. (2024). Finite-time analysis of natural actor-critic for POMDPs. SIAM Journal on Mathematics of Data Science, 6(4):869–896.

Cesa-Bianchi, N. and Lugosi, G. (2006). Prediction, learning, and games. Cambridge university press.

Ding, D., Wei, C.-Y., Zhang, K., and Ribeiro, A. (2023). Last-iterate convergent policy gradient primal-dual methods for constrained MDPs. Advances in Neural Information Processing Systems, 36:66138–66200.

Ding, D., Zhang, K., Duan, J., Basar, T., and Jovanovic, M. R. (2025). Convergence and sample complexity of natural policy gradient primal-dual methods for constrained MDPs. Journal of Machine Learning Research, 26(256):1–76.

Ding, L., Li, S., Gao, H., Chen, C., and Deng, Z. (2018). Adaptive partial reinforcement learning neural network-based tracking control for wheeled mobile robotic systems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 50(7):2512–2523.

Efroni, Y., Mannor, S., and Pirotta, M. (2020). Exploration-exploitation in constrained MDPs. arXiv preprint arXiv:2003.02189.

Hern´andez-Lerma, O. and Gonz´alez-Hern´andez, J. (2000). Constrained markov control processes in borel spaces: the discounted case. Mathematical Methods of Operations Research, 52(2):271–285.

Hogewind, Y., Simao, T. D., Kachman, T., and Jansen, N. (2022). Safe reinforcement learning from pixels using a stochastic latent representation. arXiv preprint arXiv:2210.01801.

Huang, D., Wang, J., Li, Y., Xia, C., Zhang, T., and Zhang, K. (2025). PIGDreamer: Privileged information guided world models for safe partially observable reinforcement learning. arXiv preprint arXiv:2508.02159.

Jordan, P. and Kamgarpour, M. (2026). Model-based learning of near-optimal finite-window policies in POMDPs. arXiv preprint arXiv:2604.01024.

Kaelbling, L. P., Littman, M. L., and Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134.

Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pages 267–274.

Kalagarla, K. C., Dhruva, K., Shen, D., Jain, R., Nayyar, A., and Nuzzo, P. (2022). Optimal control of partially observable markov decision processes with finite linear temporal logic constraints. In Uncertainty in Artificial Intelligence, pages 949–958. PMLR.

Kara, A. and Yuksel, S. (2022). Near optimality of finite memory feedback policies in partially observed markov decision processes. Journal of Machine Learning Research, 23(11):1–46.

Kara, A. D. and Y¨uksel, S. (2023). Convergence of finite memory q learning for POMDPs and near optimality of learned policies under filter stability. Mathematics of Operations Research, 48(4):2066–2093.

Kearns, M. and Singh, S. (2002). Near-optimal reinforcement learning in polynomial time. Machine learning, 49(2):209–232.

Khonji, M., Jasour, A., and Williams, B. C. (2019). Approximability of constant-horizon constrained pomdp. In IJCAI, pages 5583–5590.

Krishnamurthy, A., Agarwal, A., and Langford, J. (2016). PAC reinforcement learning with rich observations. Advances in Neural Information Processing Systems, 29.

Kurniawati, H., Hsu, D., Lee, W. S., et al. (2008). Sarsop: Eficient point-based POMDP planning by approximating optimally reachable belief spaces. In Robotics: Science and systems, volume 2008. Zurich, Switzerland.

Lauri, M., Hsu, D., and Pajarinen, J. (2022). Partially observable markov decision processes in robotics: A survey. IEEE Transactions on Robotics, 39(1):21–40.

Lee, J., Kim, G.-H., Poupart, P., and Kim, K.-E. (2018). Monte-carlo tree search for constrained POMDPs. Advances in Neural Information Processing Systems, 31.

McDonald, C. and Y¨uksel, S. (2020). Exponential filter stability via dobrushin’s coeficient.

Ni, T. and Kamgarpour, M. (2025). A safe exploration approach to constrained markov decision processes. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3592–3600. PMLR.

Papadimitriou, C. H. and Tsitsiklis, J. N. (1987). The complexity of markov decision processes. Mathematics of operations research, 12(3):441–450.

Paternain, S., Chamon, L., Calvo-Fullana, M., and Ribeiro, A. (2019). Constrained reinforcement learning has zero duality gap. Advances in Neural Information Processing Systems, 32.

Puterman, M. L. (2014). Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons.

Sutton, R. S., Barto, A. G., et al. (1998). Reinforcement learning: An introduction, volume 1. MIT press Cambridge.

Tian, T., Yang, L. F., and Szepesv´ari, C. (2024). Confident natural policy gradient for local planning in q pi -realizable constrained MDPs. Advances in Neural Information Processing Systems, 37:76139–76176.

Undurti, A. and How, J. P. (2010). An online algorithm for constrained POMDPs. In 2010 IEEE International Conference on Robotics and Automation, pages 3966–3973. IEEE.

Yu, C., Liu, J., Nemati, S., and Yin, G. (2021). Reinforcement learning in healthcare: A survey. ACM Computing Surveys (CSUR), 55(1):1–36.

# Supplementary Material

## A Superstate CMDP posterior belief

Let $g = \left( o _ { 0 } , a _ { 0 } , o _ { 1 } , \ldots , a _ { k - 1 } , o _ { k } \right) \in \mathcal { H } _ { k } , k \le l$ be a superstate. We associate to it a finite-history belief $b ^ { g } \in \Delta ( \mathcal { S } )$ defined as follows. Starting from the prior on states $\rho ^ { o }$ , define

$$
b _ { 0 } ^ { g } ( s ) = \frac { \rho ^ { o } ( s ) \mathbb { O } ^ { o } ( o _ { 0 } | s ) } { \sum _ { \tilde { s } \in { \cal S } } \rho ^ { o } ( \tilde { s } ) \mathbb { O } ^ { o } ( o _ { 0 } | \tilde { s } ) } .
$$

For $i = 0 , \ldots , k - 1$ , recursively define

$$
b _ { i + 1 } ^ { g } ( s ^ { \prime } ) = \frac { \mathbb { O } ^ { o } ( o _ { i + 1 } \vert s ^ { \prime } ) \sum _ { s \in \mathcal { S } } \mathbb { P } ^ { o } ( s ^ { \prime } \vert s , a _ { i } ) b _ { i } ^ { g } ( s ) } { \sum _ { \tilde { s } \in \mathcal { S } } \mathbb { O } ^ { o } ( o _ { i + 1 } \vert \tilde { s } ) \sum _ { s \in \mathcal { S } } \mathbb { P } ^ { o } ( \tilde { s } \vert s , a _ { i } ) b _ { i } ^ { g } ( s ) } .
$$

We then set $b ^ { g } : = b _ { k } ^ { g }$ . Equivalently, $b ^ { g }$ is obtained by marginalizing over all latent-state trajectories compatible with the observation-action sequence encoded by $^ { g , }$ starting from the prior $\rho ^ { o }$

## B History-state CMDP

A classic result (Kaelbling et al., 1998) is that any POMDP can be equivalently formulated as an infinite-state observable MDP whose states are full observation-action histories. The same construction easily extends to CPOMDPs by also accounting for costs.

Specifically, define the infinite history-state CMDP the tuple

$$
\mathcal { M } ^ { h } = ( \mathcal { H } , \mathcal { A } , \mathbb { P } ^ { h } , r ^ { h } , c ^ { h } , \gamma , d ^ { h } , \rho ^ { h } )
$$

where $\mathcal { H }$ is the state space, $\mathcal { A }$ is the same action space of the $\mathrm { C P O M D P } \mathcal { P } .$ and $d ^ { h } = d ^ { o }$ . The transition kernel $\mathbb { P } ^ { h }$ , the reward and cost functions ${ \bar { r } } ^ { h } , c ^ { h }$ , and the initial history distribution $\rho ^ { h }$ are defined below.

The transition kernel $\mathbb { P } ^ { h }$ specifies the probability of the next history h<sup>′</sup> given the current history h and action $a .$ The next history is obtained by appending a and the next observation o to $h .$ Therefore, conditional on $( h , a )$ , the only random component of the next history is $^ { O , }$ and

$$
\mathbb { P } ^ { h } ( h ^ { \prime } | h , a ) = \sum _ { o \in \mathcal { O } } \mathbf { 1 } ( h ^ { \prime } = h \| ( a , o ) ) \operatorname* { P r } ( o | h , a )
$$

Since the latent state is not observed, the conditional distribution of the next observation given $h , a , { \mathrm { i . e . , P r } } ( o | h , a )$ is obtained by marginalizing over the current and next latent state. Let

$$
b ^ { h } ( s ) : = \operatorname* { P r } ( s _ { t } = s \mid h _ { t } = h ) , \qquad b ^ { h } \in \Delta ( \mathcal { S } ) ,
$$

denote the posterior belief induced by the history $h ^ { 3 }$ . Then, using the structure of the underlying CPOMDP

$$
\operatorname* { P r } ( o | h , a ) = \sum _ { s , s ^ { \prime } \in S } \mathbb { O } ^ { o } ( o | s ^ { \prime } ) \mathbb { P } ^ { o } ( s ^ { \prime } | s , a ) b ^ { h } ( s ) .
$$

The reward and cost functions are obtained analogously by taking their conditional expectations over the latent state under $b ^ { h }$

$$
r ^ { h } ( h , a ) = \sum _ { s \in \mathcal { S } } b ^ { h } ( s ) r ^ { o } ( s , a ) , \qquad c ^ { h } ( h , a ) = \sum _ { s \in \mathcal { S } } b ^ { h } ( s ) c ^ { o } ( s , a ) .
$$

The initial history consists only of the initial observation. Its initial distribution is therefore:

$$
\rho ^ { h } ( h ) : = \mathbf { 1 } ( h \in \mathcal { H } _ { 0 } ) \sum _ { s \in \mathcal { S } } \mathbb { O } ^ { o } ( h | s ) \rho ^ { o } ( s ) .
$$

For $\pi \in \Pi ^ { \mathrm { h i s t } }$ , define the reward value function:

$$
\begin{array} { r } { V _ { r ^ { h } } ^ { \pi } ( h ) = \mathbb { E } _ { \pi , \mathbb P ^ { h } } \Big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ^ { h } ( h _ { t } , a _ { t } ) \Big | h _ { 0 } = h \Big ] , } \end{array}
$$

and its expected reward value with respect to the initial history distribution $\rho ^ { h }$ as

$$
V _ { r ^ { h } } ^ { \pi } ( \rho ^ { h } ) : = \mathbb { E } _ { h \sim \rho ^ { h } } \left[ V _ { r ^ { h } } ^ { \pi } ( h ) \right]
$$

Then, $V _ { c ^ { h } } ^ { \pi } ( h ) , V _ { c ^ { h } } ^ { \pi } ( \rho ^ { h } )$ are defined analogously by replacing the reward $r ^ { h }$ with the cost $c ^ { h }$

By construction, $\rho ^ { h }$ reproduces the initial observation distribution of the original CPOMDP, while $\mathbb { P } ^ { \Breve { h } }$ reproduces the conditional distribution of the next observation given each history-action pair. Hence, under the same history-dependent policy, the two models induce the same distribution over observation-action histories. Moreover, $r ^ { h }$ and $c ^ { h }$ are the conditional expectations of the original reward and cost. Thus, the two formulations have the same expected reward and cost values when evaluated from their respective initial distributions (Kaelbling et al., 1998), namely $V _ { r ^ { h } } ^ { \pi } ( \rho ^ { h } ) = V _ { r ^ { o } } ^ { \pi } ( \rho ^ { o } ) , V _ { c ^ { h } } ^ { \pi } ( \rho ^ { h } ) = V _ { c ^ { o } } ^ { \pi } ( \rho ^ { o } )$ , and they can be used interchangeably.

Existence of an optimal policy. As shown above, Problem (1) is equivalent to the constrained problem in the history-state CMDP $\mathcal { M } ^ { h }$ , which has a countable state space H, a finite action space ${ \mathcal { A } } ,$ and rewards and costs in [0, 1]. For such discounted CMDPs, an optimal policy exists whenever the constrained problem is feasible (Altman, 2021; Hern´andez-Lerma and Gonz´alez-Hern´andez, 2000).

## C Full Theorem Statement

Let Assumptions 1, 2 hold. Let $d ^ { g }$ be such that $\begin{array} { r } { \xi ^ { g } : = d ^ { g } - d ^ { o } + \xi ^ { o } - \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } > 0 } \end{array}$ , and set $\begin{array} { r } { \lambda ^ { \operatorname* { m a x } } = \frac { 2 } { ( 1 - \gamma ) \xi ^ { g } } } \end{array}$ . Run Algorithm 1 for $K \in \mathbb N$ iterations, using an oracle $\boldsymbol { B }$ that satisfies inequalities in (8) and with

$$
\begin{array} { l } { \eta _ { 1 } = \displaystyle \frac { ( 1 - \gamma ) ^ { 2 } \xi ^ { g } } { ( 1 - \gamma ) \xi ^ { g } + 2 } \sqrt { \displaystyle \frac { 2 \log | A | } { K } } , } \\ { \eta _ { 2 } = \displaystyle \frac { 2 } { \xi ^ { g } ( 1 + ( 1 - \gamma ) d ^ { g } ) \sqrt { K } } } \end{array}
$$

Then, the average optimality gap satisfies

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { o } } ^ { \pi ^ { \star , h } } ( \rho ^ { o } ) - V _ { r ^ { o } } ^ { \pi ^ { k } } ( \rho ^ { o } ) ] \leq \frac { 4 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } + \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } \left[ d ^ { o } - d ^ { g } + \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } \right] _ { + } } & { { } \medskip } \\ { \displaystyle } & { { } \quad + \frac { 2 ( \xi ^ { g } + 1 ) } { ( 1 - \gamma ) \xi ^ { g } } \delta _ { \mathrm { e s t } } } \\ { \displaystyle } & { { } \quad + \frac { ( 1 - \gamma ) \xi ^ { g } + 2 } { ( 1 - \gamma ) ^ { 3 } \xi ^ { g } } \sqrt { \frac { 2 \log | A | } { K } } + \frac { 1 + ( 1 - \gamma ) d ^ { g } } { ( 1 - \gamma ) ^ { 2 } \xi ^ { g } \sqrt { K } } \displaystyle \frac { \xi ^ { g } } { K } \ln ( \xi ) , } \end{array}
$$

Moreover, the average constraint violation satisfies

$$
\begin{array} { r l } { \displaystyle \left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { o } } ^ { \pi ^ { k } } ( \rho ^ { o } ) - d ^ { o } ] \right] _ { + } \leq \left[ d ^ { g } - d ^ { o } + \displaystyle \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } \right] _ { + } } & { } \\ { + \displaystyle 2 ( \xi ^ { g } + 1 ) \delta _ { \mathrm { e s t } } } & { } \\ { + \displaystyle \frac { ( 1 - \gamma ) \xi ^ { g } + 2 } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \displaystyle \frac { 2 \log | A | } { K } } + \displaystyle \frac { 2 ( 1 + ( 1 - \gamma ) d ^ { g } ) } { ( 1 - \gamma ) \sqrt { K } } . } \end{array}
$$

## D Omitted Proofs

## D.1 Proof of Proposition 1

We use the following consequence of Assumption 1, obtained by applying its contraction over the l steps of a superstate.

Lemma 4 (Anjarlekar et al. (2025, Lemma 1)). Under Assumption 1, for any history $h \in \mathcal H$ and corresponding superstate $g = \mathbb { G } _ { l } ( h )$ , it holds $\| b ^ { h } - b ^ { g } \| _ { \mathrm { T V } } \leq ( 1 - \kappa ) ^ { l }$

For convenience, we prove the proposition in the history-state CMDP formulation rather than in the CPOMDP, since the former provides an equivalent fully observable formulation, as described in Appendix B. We then exploit the equivalence $V _ { u ^ { h } } ^ { \pi } ( \rho ^ { h } ) = V _ { u ^ { o } } ^ { \pi } ( \rho ^ { o } )$ to transfer the result back to the original formulation.

Proof of a). Fix a stationary superstate policy $\pi \in \Pi ^ { g }$ . When deployed in the history-state CMDP, it selects actions according to $a _ { t } \sim \pi ( \cdot | \mathbb { G } _ { l } ( h _ { t } ) )$ ). Fix $h \in \mathcal H$ and let $g = \mathbb { G } _ { l } ( h )$ . We first show that

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } | V _ { u ^ { h } } ^ { \pi } ( h ) - V _ { u ^ { g } } ^ { \pi } ( g ) | \leq \delta ( l ) .
$$

where $g = \mathbb { G } _ { l } ( h )$ and where $V _ { \boldsymbol { u } ^ { h } } ^ { \pi } ( h )$ is the reward/cost value function given a history, defined in Appendix B, and $\begin{array} { r } { V _ { u ^ { g } } ^ { \pi } ( g ) : = \mathbb { E } _ { \pi , \mathbb { P } ^ { g } } \big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } u ^ { g } ( g _ { t } , a _ { t } ) \big | g _ { 0 } = g \big ] } \end{array}$ is the corresponding superstate value function given an initial superstate g. In turn this implies

$$
\left| V _ { u ^ { h } } ^ { \pi } ( \rho ^ { h } ) - V _ { u ^ { g } } ^ { \pi } ( \rho ^ { g } ) \right| \leq \delta ( l ) .
$$

Finally, because $V _ { u ^ { h } } ^ { \pi } ( \rho ^ { h } ) = V _ { u ^ { o } } ^ { \pi } ( \rho ^ { o } )$ , this will imply statement a).

For $u \in \{ r , c \}$ recall that:

$$
u ^ { h } ( h , a ) : = \sum _ { s \in \cal S } u ^ { o } ( s , a ) b ^ { h } ( s ) , \qquad u ^ { g } ( g , a ) : = \sum _ { s \in \cal S } u ^ { o } ( s , a ) b ^ { g } ( s ) ,
$$

and

$$
\begin{array} { r l } & { \mathrm { P r } ( o | h , a ) = \displaystyle \sum _ { s , s ^ { \prime } \in S } \mathbb { O } ^ { o } ( o | s ^ { \prime } ) \mathbb { P } ^ { o } ( s ^ { \prime } | s , a ) b ^ { h } ( s ) } \\ & { \mathrm { P r } ( o | b ^ { g } , a ) = \displaystyle \sum _ { s , s ^ { \prime } \in S } \mathbb { O } ^ { o } ( o | s ^ { \prime } ) \mathbb { P } ^ { o } ( s ^ { \prime } | s , a ) b ^ { g } ( s ) . } \end{array}
$$

Since $u ^ { o } ( s , a ) \in [ 0 , 1 ]$ , Lemma 4 (which follows from Assumption 1) allows us to bound the reward and cost diferences:

$$
| u ^ { h } ( h , a ) - u ^ { g } ( g , a ) | = \left| \sum _ { s \in \mathcal { S } } u ^ { o } ( s , a ) ( b ^ { h } ( s ) - b ^ { g } ( s ) ) \right| \leq \sum _ { s \in \mathcal { S } } | b ^ { h } ( s ) - b ^ { g } ( s ) | = 2 \| b ^ { h } - b ^ { g } \| _ { \mathrm { T V } } \leq 2 ( 1 - \kappa ) ^ { l } .\tag{11}
$$

Similarly

$$
\sum _ { \sigma \in \mathcal { O } } \vert \operatorname* { P r } ( \sigma \vert h , a ) - \operatorname* { P r } ( \sigma \vert b ^ { \sigma } , a ) \vert = \sum _ { \sigma \in \mathcal { O } } \left. \sum _ { s , s ^ { \prime } \in S } \mathbb { O } ^ { \sigma } ( \sigma \vert s ^ { \prime } ) \mathbb { P } ^ { \sigma } ( s ^ { \prime } \vert s , a ) ( b ^ { h } ( s ) - b ^ { \sigma } ( s ) ) \right. \leq \sum _ { s \in S } \vert b ^ { h } ( s ) - b ^ { \sigma } ( s ) \vert \leq 2 ( 1 - \kappa ) ^ { l } .\tag{12}
$$

We now consider the Bellman expectation equations in the history-state and superstate CMDPs, which are, respectively:

$$
V _ { u ^ { h } } ^ { \pi } ( h ) = \mathbb { E } _ { a \sim \pi ( \cdot | g ) } \left[ u ^ { h } ( h , a ) + \gamma \sum _ { o \in \mathcal { O } } \operatorname* { P r } ( o | h , a ) V _ { u ^ { h } } ^ { \pi } ( h | | ( a , o ) ) \right]
$$

$$
V _ { u ^ { g } } ^ { \pi } ( g ) = \mathbb { E } _ { a \sim \pi ( \cdot | g ) } \left[ u ^ { g } ( g , a ) + \gamma \sum _ { o \in \mathcal { O } } \operatorname* { P r } ( o | b ^ { g } , a ) V _ { u ^ { g } } ^ { \pi } ( \mathbb { G } _ { l } ( h \| ( a , o ) ) ) \right]
$$

Subtracting the two equations, adding and subtracting $\operatorname* { P r } ( o | h , a ) V _ { u ^ { g } } ^ { \pi } ( \mathbb { G } _ { l } ( h \| ( a , o ) ) )$ , and applying the triangle inequality gives

$$
\begin{array} { r l } & { \bigl | V _ { u ^ { h } } ^ { \pi } ( h ) - V _ { u ^ { \sigma } } ^ { \pi } ( g ) \bigr | \leq \mathbb { E } _ { a \sim \pi ( \cdot | g ) } \bigl | u ^ { h } ( h , a ) - u ^ { \sigma } ( g , a ) \bigr | + \gamma \mathbb { E } _ { a \sim \pi ( \cdot | g ) } \displaystyle \sum _ { o \in \mathcal { O } } \bigl | \operatorname* { P r } ( o | h , a ) - \operatorname* { P r } ( o | b ^ { g } , a ) \bigr | \cdot \bigl | V _ { u ^ { \sigma } } ^ { \pi } ( \mathbb { G } _ { l } ( h \| ( a , o ) ) ) \bigr | } \\ & { \qquad + \gamma \mathbb { E } _ { a \sim \pi ( \cdot | g ) } \displaystyle \sum _ { o \in \mathcal { O } } \operatorname* { P r } ( o | h , a ) \cdot \bigl | V _ { u ^ { h } } ^ { \pi } ( h \| ( a , o ) ) - V _ { u ^ { \sigma } } ^ { \pi } ( \mathbb { G } _ { l } ( h \| ( a , o ) ) ) \bigr | } \end{array}\tag{13}
$$

Because $u ^ { g } \in [ 0 , 1 ]$ , it holds $| V _ { u ^ { g } } ^ { \pi } ( g ) | \leq 1 / ( 1 - \gamma )$ . Moreover, using (11)-(12) in (13), we obtain

$$
| V _ { u ^ { h } } ^ { \pi } ( h ) - V _ { u ^ { 0 } } ^ { \pi } ( g ) | \leq 2 ( 1 - \kappa ) ^ { l } + \frac { 2 \gamma ( 1 - \kappa ) ^ { l } } { 1 - \gamma } + \gamma \mathbb { E } _ { a \sim \pi ( \cdot \vert g ) } \sum _ { o \in \mathcal { O } } \operatorname* { P r } ( o | h , a ) \cdot | V _ { u ^ { h } } ^ { \pi } ( h | | ( a , o ) ) - V _ { u ^ { 0 } } ^ { \pi } ( \mathbb { G } _ { l } ( h | | ( a , o ) ) ) |
$$

Define

$$
\Delta : = \operatorname* { s u p } _ { h \in \mathcal { H } } \left| V _ { u ^ { h } } ^ { \pi } ( h ) - V _ { u ^ { g } } ^ { \pi } ( \mathbb { G } _ { l } ( h ) ) \right| ,
$$

which is finite because the value functions are finite. It follows that

$$
| V _ { u ^ { h } } ^ { \pi } ( h ) - V _ { u ^ { g } } ^ { \pi } ( g ) | \leq \frac { 2 ( 1 - \kappa ) ^ { l } } { 1 - \gamma } + \gamma \Delta .
$$

Taking the supremum over $h \in \mathcal H$ gives

$$
\Delta \le \frac { 2 ( 1 - \kappa ) ^ { l } } { 1 - \gamma } + \gamma \Delta \qquad \Rightarrow \qquad \Delta \le \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } = \delta ( l ) .
$$

Finally, since $\rho ^ { h }$ and $\rho ^ { g }$ agree

$$
\left| V _ { u ^ { h } } ^ { \pi } ( \rho ^ { h } ) - V _ { u ^ { g } } ^ { \pi } ( \rho ^ { g } ) \right| \leq \delta ( l ) .
$$

Proof of b). Fix an arbitrary history-dependent policy $\pi \in \Pi ^ { \mathrm { h i s t } }$ . The argument in part a) cannot be applied directly because the action distribution $\pi ( \cdot | h )$ does not generally define a unique policy on the superstate space. In particular, two histories with the same superstate may be associated to diferent action distributions.

To address this issue, we first construct an auxiliary history-state CMDP. Its state is the full history $h \in \mathcal H$ , so that the original policy $\pi ( \cdot | h )$ can be applied without modification. Its reward and cost functions and transition kernel are defined to depend only on the corresponding superstate:

$$
\begin{array} { r l } & { u ^ { \mathrm { a u x } } ( h , a ) : = u ^ { g } ( \mathbb { G } _ { l } ( h ) , a ) , } \\ & { \mathbb { P } ^ { \mathrm { a u x } } ( h ^ { \prime } | h , a ) : = \displaystyle \sum _ { o \in \mathcal { O } } \mathbf { 1 } ( h ^ { \prime } = h \| ( a , o ) ) \operatorname* { P r } ( o | b ^ { \mathbb { G } _ { l } ( h ) } , a ) . } \end{array}
$$

The same Bellman argument as in part a) can now be applied to $V _ { u ^ { h } } ^ { \pi } ( h )$ and $V _ { u ^ { \mathrm { a u x } } } ^ { \pi } ( h )$ , giving

$$
\left| V _ { u ^ { h } } ^ { \pi } ( \rho ^ { h } ) - V _ { u ^ { \operatorname { a u x } } } ^ { \pi } ( \rho ^ { h } ) \right| \leq \delta ( l ) .\tag{14}
$$

We next show that, although π is history-dependent, its behavior in the auxiliary CMDP, in terms of expected reward value and expected cost value, can be reproduced by a stationary policy that only observes the superstate. The key observation is that, for expected discounted values, the behavior of a policy is completely characterized by its discounted state-action occupancy measure. Define:

$$
\mu ^ { \pi } ( g , a ) : = ( 1 - \gamma ) \mathbb { E } _ { \pi , \mathbb { P } ^ { \mathrm { a u x } } , \rho ^ { h } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbf { 1 } ( \mathbb { G } _ { l } ( h _ { t } ) = g , a _ { t } = a ) \right] \in \Delta ( \mathcal { G } \times \mathcal { A } )
$$

Thus, $\mu ^ { \pi } ( g , a )$ is the discounted frequency with which the auxiliary process visits superstate $g$ and takes action a under $\pi .$ Let:

$$
\mu ^ { \pi } ( g ) : = \sum _ { a \in \mathcal { A } } \mu ^ { \pi } ( g , a ) \in \Delta ( \mathcal { G } )
$$

denote the corresponding state occupancy measure.

By construction of the auxiliary model, once the current superstate-action pair $( g , a )$ is fixed, the distribution of the next superstate is given by $\mathbb { P } ^ { g } ( \cdot | g , a )$ , independently of the particular history that produced g. Hence, the state occupancy measure $\mu ^ { \pi } ( g )$ must satisfy the following Bellman flow constraint (Puterman, 2014):

$$
\mu ^ { \pi } ( g ) = ( 1 - \gamma ) \rho ^ { g } ( g ) + \gamma \sum _ { g ^ { \prime } \in { \mathcal { G } } } \sum _ { a ^ { \prime } \in A } \mu ^ { \pi } ( g ^ { \prime } , a ^ { \prime } ) \mathbb { P } ^ { g } ( g | g ^ { \prime } , a ^ { \prime } ) .\tag{15}
$$

We now define a stationary randomized superstate policy $\pi ^ { g } \in \Pi ^ { g }$ by

$$
\pi ^ { g } ( a | g ) : = \frac { \mu ^ { \pi } ( g , a ) } { \mu ^ { \pi } ( g ) } ,
$$

whenever $\mu ^ { \pi } ( g ) > 0 .$ , and arbitrarily otherwise. Since by construction $\mu ^ { \pi } ( g , a ) = \mu ^ { \pi } ( g ) \pi ^ { g } ( a | g )$ , we can substitute this identity into the flow constraint in (15):

$$
\mu ^ { \pi } ( g ) = ( 1 - \gamma ) \rho ^ { g } ( g ) + \gamma \sum _ { g ^ { \prime } \in { \mathcal { G } } } \sum _ { a ^ { \prime } \in A } \mu ^ { \pi } ( g ^ { \prime } ) \pi ^ { g } ( a ^ { \prime } | g ^ { \prime } ) \mathbb { P } ^ { g } ( g | g ^ { \prime } , a ^ { \prime } ) .
$$

But, by definition, also the state occupancy measure induced by $\pi ^ { g } , \mu ^ { \pi ^ { g } }$ satisfies this constraint:

$$
\mu ^ { \pi ^ { g } } ( g ) = ( 1 - \gamma ) \rho ^ { g } ( g ) + \gamma \sum _ { g ^ { \prime } \in { \mathcal { G } } } \sum _ { a ^ { \prime } \in A } \mu ^ { \pi ^ { g } } ( g ^ { \prime } ) \pi ^ { g } ( a ^ { \prime } | g ^ { \prime } ) \mathbb { P } ^ { g } ( g | g ^ { \prime } , a ^ { \prime } ) .
$$

Because $\gamma < 1$ , this equation has a unique solution. Hence:

$$
\mu ^ { \pi ^ { g } } ( g ) = \mu ^ { \pi } ( g )
$$

and so

$$
\begin{array} { r } { \mu ^ { \pi ^ { g } } ( g , a ) = \mu ^ { \pi } ( g , a ) . } \end{array}
$$

Finally, since the auxiliary and superstate models have reward and cost as functions of the superstateaction pair $( g , a )$ , the expected discounted reward and cost values can be expressed in terms of the corresponding state-action occupancy measure as Puterman (2014):

$$
V _ { u ^ { \mathrm { a u x } } } ^ { \pi } ( \rho ^ { h } ) = \frac { 1 } { 1 - \gamma } \sum _ { g \in \mathcal { G } } \sum _ { a \in \mathcal { A } } \mu ^ { \pi } ( g , a ) u ^ { g } ( g , a ) .
$$

Since the state-action occupancy measures coincide, it follows that

$$
V _ { u ^ { g } } ^ { \pi ^ { g } } ( \rho ^ { g } ) = \frac { 1 } { 1 - \gamma } \sum _ { g \in \mathcal { G } } \sum _ { a \in \mathcal { A } } \mu ^ { \pi } ( g , a ) u ^ { g } ( g , a ) = V _ { u ^ { \mathrm { a u x } } } ^ { \pi } ( \rho ^ { h } )
$$

which concludes the proof.

## D.2 Proof of Lemma 2

Both parts are standard for CMDPs (Altman, 2021; Paternain et al., 2019), (Ding et al., 2025, Lemma 3); we include the proof formulated in our superstate setting for completeness. By Lemma 1, there exists $\bar { \pi } \in \Pi ^ { g }$ with $V _ { c ^ { g } } ^ { \bar { \pi } } ( \rho ^ { g } ) \leq d ^ { g } - \xi ^ { g }$ and $\xi ^ { g } > 0$

(i) Strong duality. For $\pi \in \Pi ^ { g }$ , let $\begin{array} { r } { \mu ^ { \pi } ( g , a ) : = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \operatorname* { P r } _ { \pi , \mathbb { P } ^ { g } } ( g _ { t } = g , a _ { t } = a \mid g _ { 0 } \sim \rho ^ { g } ) } \end{array}$ denote its discounted occupancy measure, so that $V _ { r ^ { g } } ^ { \pi } ( \rho ^ { g } ) = \langle \bar { \mu } ^ { \pi } , r ^ { g } \rangle$ and $V _ { c ^ { g } } ^ { \pi } ( \rho ^ { g } ) = \langle \mu ^ { \pi } , c ^ { g } \rangle$ . Since G and A are finite, the set $\mathcal { M } : = \{ \mu ^ { \pi } : \pi \in \Pi ^ { g } \}$ is a polytope described by the Bellman flow constraints, and every $\mu \in \mathcal { M }$ is induced by the stationary policy π $( a \mid g ) \propto \mu ( g , a )$ (Altman, 2021). Hence Problem (3) is equivalent to the linear program ma $\mathrm { x } _ { \mu \in { \mathcal { M } } } \{ \langle \mu , r ^ { g } \rangle : \langle \mu , c ^ { g } \rangle \leq d ^ { g } \}$ , and

$$
D ^ { \lambda } ( \rho ^ { g } ) = \operatorname* { m a x } _ { \mu \in \mathcal { M } } \langle \mu , r ^ { g } - \lambda c ^ { g } \rangle + \lambda d ^ { g } .
$$

This linear program is feasible and bounded, and $\mu ^ { \bar { \pi } }$ is strictly feasible. By linear programming duality, the dual problem min $\lambda { \geq } 0 D ^ { \lambda } ( \rho ^ { g } )$ attains its minimum at some $\lambda ^ { * }$ , and $V _ { r ^ { g } } ^ { \pi ^ { * } } ( \rho ^ { g } ) \stackrel { - } { = } D ^ { \lambda ^ { * } } ( \rho ^ { g } )$

(ii) Bound on $\lambda ^ { * }$ . By (i) and the definition of $D ^ { \lambda ^ { * } }$ ，

$$
V _ { r ^ { g } } ^ { \pi ^ { * } } ( \rho ^ { g } ) = D ^ { \lambda ^ { * } } ( \rho ^ { g } ) \geq L ( \bar { \pi } , \lambda ^ { * } ) = V _ { r ^ { g } } ^ { \bar { \pi } } ( \rho ^ { g } ) + \lambda ^ { * } \big ( d ^ { g } - V _ { c ^ { g } } ^ { \bar { \pi } } ( \rho ^ { g } ) \big ) \geq V _ { r ^ { g } } ^ { \bar { \pi } } ( \rho ^ { g } ) + \lambda ^ { * } \xi ^ { g } .
$$

Since $r ^ { g } ( g , a )$ is a convex combination of values of $r ^ { o } ~ \in ~ [ 0 , 1 ]$ , we have $r ^ { g } ~ \in ~ [ 0 , 1 ]$ and thus $V _ { r ^ { g } } ^ { \pi } ( \rho ^ { g } ) \in [ 0 , 1 / ( 1 - \gamma ) ]$ for all $\pi \in \Pi ^ { g }$ . Rearranging gives

$$
\lambda ^ { \ast } \leq \frac { V _ { r ^ { g } } ^ { \pi ^ { \ast } } ( \rho ^ { g } ) - V _ { r ^ { g } } ^ { \pi } ( \rho ^ { g } ) } { \xi ^ { g } } \leq \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } = \frac { \lambda ^ { \operatorname* { m a x } } } { 2 } \leq \lambda ^ { \operatorname* { m a x } } . \qquad \sqcap
$$

## D.3 Proof of Proposition 2

Let $\pi ^ { \star , h } \in \Pi ^ { \mathrm { h i s t } }$ be an optimal CPOMDP policy. By Proposition 1b), there exists a superstate policy $( \pi ^ { \star , h } ) ^ { g }$ such that

$$
\left| V _ { r ^ { o } } ^ { \pi ^ { \star , h } } - V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \right| \leq \delta ( l ) , \qquad \left| V _ { c ^ { o } } ^ { \pi ^ { \star , h } } - V _ { c ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \right| \leq \delta ( l ) .
$$

We have:

$$
\begin{array} { r l } & { { V _ { r ^ { o } } ^ { \pi ^ { \star , h } } } - { V _ { r ^ { g } } ^ { \pi ^ { \star } } } = \underbrace { \big ( V _ { r ^ { o } } ^ { \pi ^ { \star , h } } - V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \big ) } _ { \le \delta ( l ) } + \big ( { V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } } - { V _ { r ^ { g } } ^ { \pi ^ { \star } } } \big ) \le \delta ( l ) + \big ( { V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } } - { V _ { r ^ { g } } ^ { \pi ^ { \star } } } \big ) . } \end{array}\tag{16}
$$

We proceed by bounding $( V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } - V _ { r ^ { g } } ^ { \pi ^ { \ast } } )$ . Since $\pi ^ { \star , h }$ is feasible in the original CPOMDP ${ \mathcal { P } } _ { : }$ $V _ { c ^ { o } } ^ { \pi ^ { \star , h } } \leq d ^ { o }$

By Proposition 1b),

$$
V _ { c ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \le V _ { c ^ { o } } ^ { \pi ^ { \star , h } } + \delta ( l ) \le d ^ { o } + \delta ( l ) .
$$

We note that $( \pi ^ { \star , h } ) ^ { g }$ is not necessarily feasible for the superstate $\mathcal { M } ^ { g }$ . If we define

$$
\Delta _ { c } : = [ d ^ { o } + \delta ( l ) - d ^ { g } ] _ { + } ,
$$

then $\Delta _ { c } \geq 0$ and $V _ { c ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \leq d ^ { g } + \Delta _ { c }$ . So, $( \pi ^ { \star , h } ) ^ { g }$ is feasible for a relaxed version of the superstate problem

$$
\operatorname* { m a x } _ { \pi \in \Pi ^ { g } : V _ { c ^ { g } } ^ { \pi } \leq d ^ { g } + \Delta _ { c } } V _ { r ^ { g } } ^ { \pi }\tag{17}
$$

Now, $( \pi ^ { \star , h } ) ^ { g }$ is feasible for this relaxed problem, but it is not necessarily optimal. Therefore:

$$
V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } \le \operatorname* { m a x } _ { \pi \in \Pi ^ { g } : V _ { c ^ { g } } ^ { \pi } \le d ^ { g } + \Delta _ { c } } V _ { r ^ { g } } ^ { \pi } .
$$

We now use the following constraint-sensitivity result from Tian et al. (2024, Lemma 14). We restate it under our convention $V _ { c ^ { g } } ^ { \pi } \leq d$ and include the adapted proof for completeness.

Lemma 5. Assume that the superstate CMDP satisfies the Slater condition with margin $\xi ^ { g } > 0$ and let $\lambda ^ { * }$ be an optimal multiplier for the problem with threshold $d ^ { g }$ . For any $\Delta _ { c } \geq 0$ , we have:

$$
0 \leq \operatorname* { m a x } _ { V _ { c } ^ { \pi } \leq d ^ { g } + \Delta _ { c } } V _ { r ^ { g } } ^ { \pi } - V _ { r ^ { g } } ^ { \pi ^ { * } } \leq \lambda ^ { * } \Delta _ { c } \leq \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } \Delta _ { c } .
$$

Proof. Since $\Delta _ { c } \geq 0$ , increasing the threshold enlarges the feasible set, proving the first inequality.

Now, fix any policy $\pi \in \Pi ^ { g }$ feasible for the relaxed problem. Since $V _ { c ^ { g } } ^ { \pi } \leq d ^ { g } + \Delta _ { c }$ and $\lambda ^ { * } \geq 0$ , we have:

$$
\begin{array} { r l } & { V _ { r ^ { g } } ^ { \pi } = V _ { r ^ { g } } ^ { \pi } - \lambda ^ { * } ( V _ { c ^ { g } } ^ { \pi } - d ^ { g } ) + \lambda ^ { * } ( V _ { c ^ { g } } ^ { \pi } - d ^ { g } ) } \\ & { \quad \quad \leq V _ { r ^ { g } } ^ { \pi } - \lambda ^ { * } ( V _ { c ^ { g } } ^ { \pi } - d ^ { g } ) + \lambda ^ { * } \Delta _ { c } } \\ & { \quad \quad \leq \displaystyle \operatorname* { m a x } _ { \pi ^ { \prime } \in \Pi ^ { g } } \Big \{ V _ { r ^ { g } } ^ { \pi ^ { \prime } } - \lambda ^ { * } ( V _ { c ^ { g } } ^ { \pi ^ { \prime } } - d ^ { g } ) \Big \} + \lambda ^ { * } \Delta _ { c } } \\ & { \quad \quad = V _ { r ^ { g } } ^ { \pi ^ { * } } + \lambda ^ { * } \Delta _ { c } , } \end{array}
$$

where the last equality follows from strong duality and the optimality of $\lambda ^ { * } .$ Taking the max over all policies feasible for the relaxed problem gives the result. Finally, by the proof of Lemma $2 ( \mathrm { i i } )$ in Appendix D.2,

$$
\lambda ^ { * } \leq \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } ,
$$

which proves the last inequality □.

Applying Lemma 5 gives:

$$
V _ { r ^ { g } } ^ { ( \pi ^ { \star , h } ) ^ { g } } - V _ { r ^ { g } } ^ { \pi ^ { * } } \leq \frac { \Delta _ { c } } { ( 1 - \gamma ) \xi ^ { g } }
$$

Combining this inequality with (16) yields

$$
{ V _ { r ^ { o } } ^ { { \pi ^ { * , h } } } - V _ { r ^ { g } } ^ { \pi ^ { * } } } \le \delta ( l ) + \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } \Delta _ { c } = \delta ( l ) + \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } [ d ^ { o } + \delta ( l ) - d ^ { g } ] _ { + }
$$

which proves the lemma.

## D.4 Proof of Lemma 3

Define

$$
V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } : = V _ { r ^ { g } } ^ { \pi ^ { * } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { * } } , \qquad V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } : = V _ { r ^ { g } } ^ { \pi ^ { k } } - \lambda ^ { k } V _ { c ^ { g } } ^ { \pi ^ { k } } .
$$

Let $\mu ^ { * } \in \Delta ( \mathcal G )$ be the state-normalized discounted occupancy distribution induced by $\pi ^ { * }$ from $\rho ^ { g } \colon$

$$
\mu ^ { * } ( g ) : = ( 1 - \gamma ) \mathbb { E } _ { \pi ^ { * } , \mathbb { P } ^ { g } , \rho ^ { g } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \mathbf { 1 } ( g _ { t } = g ) \right] .
$$

By the Performance Diference Lemma (Kakade and Langford, 2002):

$$
V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } = \frac { 1 } { 1 - \gamma } \left. \mu ^ { * } , \mathbb { E } _ { a \sim \pi ^ { * } } [ Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] - \mathbb { E } _ { a \sim \pi ^ { k } } [ Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] \right. .
$$

By adding and subtracting the estimated Lagrangian $\mathrm { Q } \mathrm { - }$ -function $\hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$

$$
\begin{array} { r l } { V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } = \displaystyle \frac { 1 } { 1 - \gamma } \left. \mu ^ { * } , \mathbb { E } _ { a \sim \pi ^ { * } } [ Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) - \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] \right. } & { } \\ { + \displaystyle \frac { 1 } { 1 - \gamma } \left. \mu ^ { * } , \mathbb { E } _ { a \sim \pi ^ { k } } [ \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) - Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] \right. } & { } \\ { + \displaystyle \frac { 1 } { 1 - \gamma } \left. \mu ^ { * } , \mathbb { E } _ { a \sim \pi ^ { * } } [ \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] - \mathbb { E } _ { a \sim \pi ^ { k } } [ \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( \cdot , a ) ] \right. . } & { } \end{array}
$$

By Estimation Oracle B, after taking the expectation over the estimation-oracle randomness, the first two terms are bounded by $\frac { \delta _ { \mathrm { e s t } } } { 1 - \gamma }$ each. Hence:

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ] \leq \frac { 2 K \delta _ { \mathrm { e s t } } } { 1 - \gamma } + \frac { 1 } { 1 - \gamma } \mathbb { E } [ \langle \mu ^ { * } , \mathcal { R } _ { K } \rangle ] ,\tag{18}
$$

where, for each $g \in { \mathcal { G } }$

$$
\mathcal { R } _ { K } ( g ) : = \sum _ { k = 0 } ^ { K - 1 } ( \mathbb { E } _ { a \sim \pi ^ { * } } [ \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) ] - \mathbb { E } _ { a \sim \pi ^ { k } } [ \hat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) ] ) .
$$

The remaining task is to bound $\mathcal { R } _ { K } ( g )$

Fix $g \in { \mathcal { G } }$ and define:

$$
q _ { k } ( a ) : = \hat { Q } _ { r g - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) .
$$

By construction, the oracle outputs are clipped so that:

$$
q _ { k } ( a ) \in \left[ - { \frac { \lambda ^ { \operatorname* { m a x } } } { 1 - \gamma } } , { \frac { 1 } { 1 - \gamma } } \right] \subseteq [ - R , R ] , \qquad R : = { \frac { 1 + \lambda ^ { \operatorname* { m a x } } } { 1 - \gamma } } .
$$

For fixed $^ { g , }$ the policy update is an exponential weights update with payofs $q _ { k } . { \mathrm { S o } } .$ , by the standard exponential weights regret bound (Cesa-Bianchi and Lugosi, 2006, Theorem 2.2), applied to payofs in an interval of length 2R (without rescaling):

$$
\displaystyle \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { k = 0 } ^ { K - 1 } q _ { k } ( a ) - \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { a \sim \pi ^ { k } } [ q _ { k } ( a ) ] \le \frac { \log | \mathcal { A } | } { \eta _ { 1 } } + \frac { \eta _ { 1 } K R ^ { 2 } } { 2 } .
$$

Since

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } _ { a \sim \pi ^ { * } } [ q _ { k } ( a ) ] \leq \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { k = 0 } ^ { K - 1 } q _ { k } ( a ) ,
$$

it follows that

$$
\mathcal { R } _ { K } ( g ) \leq \frac { \log | \mathcal { A } | } { \eta _ { 1 } } + \frac { \eta _ { 1 } K R ^ { 2 } } { 2 } .
$$

The bound holds for any realization of the sequence $\{ q _ { k } \} _ { k = 0 } ^ { K - 1 }$ . Thus, it also holds after taking expectation over the algorithmic randomness. Moreover, since $\mu ^ { * }$ is a probability distribution:

$$
\mathbb { E } [ \langle \mu ^ { * } , \mathcal { R } _ { K } \rangle ] \leq \frac { \log | \mathcal { A } | } { \eta _ { 1 } } + \frac { \eta _ { 1 } K R ^ { 2 } } { 2 } .
$$

Choosing

$$
\eta _ { 1 } = \frac { 1 } { R } \sqrt { \frac { 2 \log | \cal { A } | } { K } }
$$

gives

$$
\mathbb { E } [ \langle \mu ^ { * } , \mathcal { R } _ { K } \rangle ] \leq R \sqrt { 2 K \log | \cal A | } .
$$

Substituting this into (18) yields

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ] \le \frac { 2 K \delta _ { \mathrm { e s t } } } { 1 - \gamma } + \frac { R } { 1 - \gamma } \sqrt { 2 K \log | A | } = \frac { 2 K \delta _ { \mathrm { e s t } } } { 1 - \gamma } + \frac { 1 + \lambda ^ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { 2 K \log | A | }
$$

Finally, dividing both sides by K gives

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ] \overset { } { \mathop = } \frac { 2 \delta _ { \mathrm { e s t } } } { 1 - \gamma } + \frac { 1 + \lambda ^ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \frac { 2 \log | A | } { K } } ,
$$

which concludes the proof.

## D.5 Average Optimality gap

We use telescoping, the non-expansiveness of the projection of the dual update and the boundedness of the clipped estimates (Section 4), as follows:

$$
\begin{array} { r l } {  { \mathcal { E } _ { \zeta \leq } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } = \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } - ( \lambda _ { j } ^ { k } ) ^ { \ell } | } } \\ & { = \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta - \tau } ^ { ( k , \ell ) } = \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } - \alpha _ { j } ^ { k } ) - ( \lambda _ { j } ^ { k } ) ^ { \ell } | } \\ & { \le \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta - \tau } ^ { ( k , \ell ) } = \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } - \alpha _ { j } ^ { k } ) ^ { \ell } | } \\ & { \le \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } = \lambda _ { j } ^ { k } ( \zeta _ { j } ^ { k , \ell } - \tau ) ^ { k } = 1 , \quad | | \zeta _ { \tau \in \tau _ { \zeta } \in \tau _ { \zeta } \in \tau _ { \zeta } \in \tau _ { \zeta } \in \tau _ { \zeta } \in \tau _ { \zeta } \in \tau _ { \zeta } \cap \zeta } + \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } - \alpha _ { j } ^ { k } | } \\ & { \le \sum _ { j = 1 } ^ { N } \mathbb { E } _ { | \zeta | } \lambda _ { \zeta } ^ { ( k , \ell ) } = \lambda _ { j } ^ { k } ( \zeta _ { j } ^ { k , \ell } - \tau ) ^ { k } = \lambda _ { j } ^ { k } ( \zeta _ { j } ^ { k , \ell } - \lambda _ { j } ^ { k } ) }  \end{array}
$$

Arranging terms:

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \lambda ^ { k } ( V _ { c s } ^ { \pi ^ { * } } - V _ { c s } ^ { \pi ^ { k } } ) ] \leq \frac { K \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \lambda ^ { k } ( \hat { V } _ { c s } ^ { \pi ^ { k } } - V _ { c s } ^ { \pi ^ { k } } ) ] .
$$

Using Estimation Oracle B and $\lambda ^ { k } \le \lambda ^ { \operatorname* { m a x } }$ , we can conclude:

$$
\sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { * } } - V _ { c ^ { g } } ^ { \pi ^ { k } } ) ] \leq \frac { K \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + K \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } .
$$

Finally, dividing both sides by K gives

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { * } } - V _ { c ^ { g } } ^ { \pi ^ { k } } ) ] \leq \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } .
$$

## D.6 Average constraint violation

To bound

$$
\left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] \right] _ { + } ,
$$

we use the dual update to compare the dual iterates with a fixed multiplier $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ . Combined with the Lagrangian average performance bound, in Lemma 3, this yields a bound on λ times the average constraint residual. Choosing $\lambda = \lambda ^ { \mathrm { m a x } }$ when the residual is positive and $\lambda = 0$ otherwise, turns this term into the positive-part average constraint violation.

For any fixed $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ , by non-expansiveness of the projection and the dual update, we have

$$
\begin{array} { r l } & { | \lambda ^ { k + 1 } - \lambda | ^ { 2 } \leq \big | \lambda ^ { k } - \lambda + \eta _ { 2 } ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) \big | ^ { 2 } } \\ & { \qquad = | \lambda ^ { k } - \lambda | ^ { 2 } + \eta _ { 2 } ^ { 2 } ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ^ { 2 } + 2 \eta _ { 2 } ( \lambda ^ { k } - \lambda ) ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) . } \end{array}
$$

Rearranging

$$
( \lambda - \lambda ^ { k } ) ( { \hat { V } } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) \leq \frac { | \lambda ^ { k } - \lambda | ^ { 2 } - | \lambda ^ { k + 1 } - \lambda | ^ { 2 } } { 2 \eta _ { 2 } } + \frac { \eta _ { 2 } } { 2 } ( { \hat { V } } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ^ { 2 } .
$$

Averaging over $k = 0 , \ldots , K - 1$ and taking expectations gives

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] \leq \frac { | \lambda ^ { 0 } - \lambda | ^ { 2 } - | \lambda ^ { K } - \lambda | ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ^ { 2 } ]
$$

Since $\lambda ^ { 0 } = 0$ , dropping the nonpositive term $- \mathbb { E } | \lambda ^ { K } - \lambda | ^ { 2 }$ and using $| \hat { V } _ { c g } ^ { \pi ^ { k } } | \leq 1 / ( 1 - \gamma )$ from the clipping in Section 4, we obtain

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] \leq \frac { \lambda ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } .
$$

We can now add and subtract the true superstate cost value $V _ { c ^ { g } } ^ { \pi ^ { k } }$ to the left-hand side, obtaining

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] - \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( V _ { c ^ { g } } ^ { \pi ^ { k } } - \hat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ) ] \leq \frac { \lambda ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 }
$$

By Estimation Oracle B and because $\lambda , \lambda ^ { k } \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ , we have

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ ( \lambda - \lambda ^ { k } ) ( { V _ { c } } ^ { \pi ^ { k } } - d ^ { g } ) ] \le \frac { \lambda ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } .\tag{19}
$$

Equation (19) bounds the term involving $( \lambda - \lambda ^ { k } ) ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } )$ . To obtain a bound involving the fixed multiplier $\lambda ,$ we are going to bound the other term $\lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } )$ . To do so, we use Lemma 3 that gives

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } + \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - V _ { c ^ { g } } ^ { \pi ^ { * } } ) ] \le \delta _ { \mathrm { l a g } } .
$$

Since $\pi ^ { * }$ is feasible, i.e., $V _ { c ^ { g } } ^ { \pi ^ { * } } \leq d ^ { g }$ , and $\lambda ^ { k } \geq 0$ , we have $\lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) \le \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - V _ { c ^ { g } } ^ { \pi ^ { * } } )$ and

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } + \lambda ^ { k } ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] \le \delta _ { \mathrm { l a g } } .
$$

Adding this inequality to (19) cancels the terms involving $\lambda ^ { k }$ , yielding, for any $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { k } } + \lambda ( V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ) ] \le \delta _ { \mathrm { l a g } } + \frac { \lambda ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \mathrm { e s t } } .
$$

Since this inequality holds for any $\lambda \in [ 0 , \lambda ^ { \operatorname* { m a x } } ]$ , we choose:

$$
\lambda = \left\{ \begin{array} { l l } { \lambda ^ { \operatorname* { m a x } } , } & { \mathrm { i f } \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

With this choice:

$$
\lambda \left( \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] \right) = \lambda ^ { \operatorname* { m a x } } \left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] \right] _ { + } ,
$$

and then:

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r s } ^ { \pi ^ { * } } - V _ { r s } ^ { \pi ^ { k } } ] + \lambda ^ { \operatorname* { m a x } } \left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c s } ^ { \pi ^ { k } } - d ^ { g } ] \right] _ { + } \leq \delta _ { \operatorname* { l a g } } + \frac { ( \lambda ^ { \operatorname* { m a x } } ) ^ { 2 } } { 2 \eta _ { 2 } K } + \frac { \eta _ { 2 } } { 2 } \left( \frac { 1 } { 1 - \gamma } + d ^ { g } \right) ^ { 2 } + \lambda ^ { \operatorname* { m a x } } \delta _ { \operatorname* { e x t } } .
$$

By the convexity of the discounted occupancy measure set (Ding et al., 2025), there exists a stationary randomized policy $\pi ^ { \prime } \in \Pi ^ { g }$ such that

$$
V _ { r ^ { g } } ^ { \pi ^ { \prime } } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { r ^ { g } } ^ { \pi ^ { k } } ] , \qquad V _ { c ^ { g } } ^ { \pi ^ { \prime } } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } ] .
$$

and

$$
V _ { r ^ { g } } ^ { \pi ^ { * } } - V _ { r ^ { g } } ^ { \pi ^ { \prime } } + \lambda ^ { \operatorname* { m a x } } [ V _ { c ^ { g } } ^ { \pi ^ { \prime } } - d ^ { g } ] _ { + } \leq \varepsilon _ { c } .
$$

We now use the upper-bound constraint counterpart of Ding et al. (2025, Lemma 5), which we report here.

Lemma 6 (Adapted from Ding et al. (2025, Lemma 5)). Assume superstate CMDP satisfies Slater condition and let $\lambda ^ { * }$ be an optimal dual multiplier. For any $B \geq 2 \lambda ^ { * } ,$ if there exists a policy $\pi ^ { \prime } \in \Pi ^ { g }$ and $\delta > 0$ such that $V _ { r ^ { g } } ^ { \pi ^ { * } } \stackrel { \textstyle \sim } { - } V _ { r ^ { g } } ^ { \pi ^ { \prime } } + B [ V _ { c ^ { g } } ^ { \pi ^ { \prime } } - \stackrel { \textstyle \sim } { d ^ { g } } ] _ { + } \leq \delta$ then $\begin{array} { r } { [ V _ { c g } ^ { \pi ^ { \prime } } - d ^ { g } ] _ { + } \leq \frac { 2 \delta } { B } } \end{array}$ . □

By the proof of Lemma 2(ii), Appendix D.2,

$$
\lambda ^ { * } \leq \frac { 1 } { ( 1 - \gamma ) \xi ^ { g } } ,
$$

and choosing $\lambda ^ { \operatorname* { m a x } } = 2 / ( ( 1 - \gamma ) \xi ^ { g } )$ ensures that $\lambda ^ { \mathrm { m a x } } \ge 2 \lambda ^ { * }$

Applying Lemma 6 with $B = \lambda ^ { \mathrm { m a x } } , \delta = \varepsilon _ { c }$ gives

$$
[ V _ { c ^ { g } } ^ { \pi ^ { \prime } } - d ^ { g } ] _ { + } \leq \frac { 2 \varepsilon _ { c } } { \lambda ^ { \mathrm { m a x } } } = ( 1 - \gamma ) \xi ^ { g } \varepsilon _ { c } .
$$

Using the definition of $\pi ^ { \prime } .$ , we conclude that

$$
\left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \mathbb { E } [ V _ { c ^ { g } } ^ { \pi ^ { k } } - d ^ { g } ] \right] _ { + } \leq ( 1 - \gamma ) \xi ^ { g } \varepsilon _ { c } .
$$

## E Conditional Sample Complexity for a Fixed Length

We derive the sample complexity of Algorithm 1 for a fixed history-window length l. Given a target accuracy $\varepsilon > 0 .$ , we seek a suficient number $N _ { \varepsilon }$ of observation-action pairs collected from the original $\operatorname { C P O M D P } \mathcal { P }$ such that

$$
{ \mathrm { O p G a p } } _ { K } \leq \varepsilon , \qquad { \mathrm { C V i o l } } _ { K } \leq \varepsilon .
$$

We consider an implementation of Estimation Oracle B based on the projected linear TD procedure of Anjarlekar et al. (2025). We first describe the estimation procedure, then state the finite-time property required from the reward and cost critics, and finally combine it with Theorem 1.

## E.1 Projected Linear TD Learning Estimation Oracle

Let $\phi : \mathcal { G } \times \mathcal { A }  \mathbb { R } ^ { d }$ be a fixed feature map satisfying

$$
\| \phi ( g , a ) \| _ { 2 } \leq 1 , \qquad \forall ( g , a ) \in \mathcal { G } \times \mathcal { A } .
$$

Fix an ordering of the superstate-action pairs and let $\Phi \in \mathbb { R } ^ { | \mathcal { G } | | \mathcal { A } | \times d }$ be the feature matrix whose row indexed by $( g , a )$ is $\bar { \phi ( g , a ) } ^ { \top }$ . We identify each action-value function with its vector in $\mathbb { R } ^ { | \mathcal { G } | | \mathcal { A } | }$

For each policy iterate $\pi ^ { k }$ and each $u \in \{ r , c \}$ , the oracle approximates the corresponding action-value function as

$$
\widehat { Q } _ { u ^ { g } } ^ { \pi ^ { k } } ( g , a ) : = \phi ( g , a ) ^ { \top } \theta _ { u , k } , \qquad \theta _ { u , k } \in \mathcal { B } ( R ) : = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \le R \} .
$$

Algorithm 2 gives the complete estimation procedure. The same trajectory is used to update both critics, so collecting reward and cost estimates does not double the number of interactions.

Algorithm 2 Projected Linear TD Estimation Oracle   
1: Input: fixed policy $\pi ^ { k }$ , feature map $\phi : \mathcal { G } \times \mathcal { A }  \mathbb { R } ^ { d } .$ , discount factor $\gamma ,$ projection radius $R ,$   
stepsizes $\left\{ \epsilon _ { t } \right\}$ , initial phase length $\tau _ { \textrm { \scriptsize m } } ^ { \prime }$ and number of TD updates $\tau$   
2: Initialize $\dot { \theta } _ { r } ^ { k } = \theta _ { c } ^ { k } = 0 \in \mathbb { R } ^ { d }$   
3: Initialize the original ${ \mathrm { C P O M D P } }$ from its initial distribution   
4: Interact with the original CPOMDP under $\pi ^ { k }$ for $\tau ^ { \prime }$ steps without updating the critics, obtaining   
$g _ { \tau ^ { \prime } }$   
5: Sample $a _ { \tau ^ { \prime } } \sim \pi ^ { k } ( \cdot \mid g _ { \tau ^ { \prime } } )$   
6: for $t = \tau ^ { \prime } , \ldots , \tau ^ { \prime } + \tau - 1$ do   
7: Execute $a _ { t }$ and observe $r _ { t } , \ c _ { t } ,$ and $g _ { t + 1 }$   
8: Sample $a _ { t + 1 } \sim \pi ^ { k } ( \cdot \mid g _ { t + 1 } )$   
9: for $u \in \{ r , c \}$ do   
10: Set $u _ { t } = r _ { t } { \mathrm { ~ i f ~ } } u = r ,$ and $u _ { t } = c _ { t } { \mathrm { ~ i f ~ } } u = c$   
11: Compute $\delta _ { u , t } ^ { k } = u _ { t } + \gamma \phi ( g _ { t + 1 } , a _ { t + 1 } ) ^ { \top } \theta _ { u } ^ { k } - \phi ( g _ { t } , a _ { t } ) ^ { \top } \theta _ { u } ^ { k }$   
12: Update $\theta _ { u } ^ { k } \gets \mathrm { P r o j } _ { { B } ( \mathit { R } ) } \big ( \theta _ { u } ^ { k } + \epsilon _ { t } \delta _ { u , t } ^ { k } \phi ( g _ { t } , a _ { t } ) \big )$   
13: end for   
14: end for   
15: Define $\widehat { Q } _ { u ^ { g } } ^ { \pi ^ { k } } ( g , a ) = \phi ( g , a ) ^ { \top } \theta _ { u } ^ { k }$ for every $( g , a ) \in \mathcal { G } \times \mathcal { A }$ and $u \in \{ r , c \}$   
16: Compute $\widehat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } ( g , a ) = \widehat { Q } _ { r ^ { g } } ^ { \pi ^ { k } } ( g , a ) - \lambda ^ { k } \widehat { Q } _ { c ^ { g } } ^ { \pi ^ { k } } ( g , a )$   
17: Compute $\widehat V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) = \sum \sum \rho ^ { g } ( g ) \pi ^ { k } ( a \mid g ) \widehat Q _ { c ^ { g } } ^ { \pi ^ { k } } ( g , a )$   
g∈G a∈A   
18: Output: $\widehat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } }$ and $\widehat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } )$

Algorithm 2 describes a possible implementation of the estimation oracle. Its finite-sample accuracy depends on the feature representation and on the coverage and mixing properties of the induced sampling process. We isolate the property required by the outer-loop analysis in the following assumption.

For each critic, define a best parameter in the projection set by

$$
\widehat { \theta } _ { u , k } \in \arg \operatorname* { m i n } _ { \| \theta \| _ { 2 } \leq R } \left\| Q _ { u ^ { g } } ^ { \pi ^ { k } } - \Phi \theta \right\| _ { 2 } ,
$$

and define the uniform function-approximation error as

$$
\varepsilon _ { \mathrm { F A } , u } : = \operatorname* { s u p } _ { k } \left\| Q _ { u ^ { g } } ^ { \pi ^ { k } } - \Phi \widehat { \theta } _ { u , k } \right\| _ { \infty } .
$$

Then we assume the following:

Assumption 3 (Linear TD critic rate). For the fixed feature representation and history length l, there exist nonnegative constants $C _ { u , l } ^ { \mathrm { { h i s t } } } , C _ { u , l } ^ { \mathrm { { i n i t } } } , C _ { u , l } ^ { \mathrm { { s t a t } } }$ , and $c _ { u , l } > 0$ , independent of $k , K ,$ , and $\tau _ { \mathrm { : } }$ such that, for every $u \in \{ r , c \}$ , Algorithm 2 satisfies

$$
\delta _ { u ^ { g } } ( \tau , l ) : = \operatorname* { s u p } _ { K \geq 1 } \operatorname* { s u p } _ { 0 \leq k < K } \mathbb { E } \Big [ \Big \| \widehat { Q } _ { u ^ { g } } ^ { \pi ^ { k } } - Q _ { u ^ { g } } ^ { \pi ^ { k } } \Big \| _ { \infty } \Big ]
$$

$$
\leq \varepsilon _ { \mathrm { F A } , u } + C _ { u , l } ^ { \mathrm { h i s t } } \frac { ( 1 - \kappa ) ^ { l } } { 1 - \gamma } + C _ { u , l } ^ { \mathrm { i n i t } } e ^ { - c _ { u , l } \sqrt { \tau } } + C _ { u , l } ^ { \mathrm { s t a t } } \frac { \log \tau } { \sqrt { \tau } } .
$$

Moreover, the required initial phase satisfies

$$
\tau ^ { \prime } = O _ { l } ( \log ( \tau ) ) .
$$

where the subscript l in $O _ { l }$ means it depend on constants that depend on fixed l.

Assumption 3 isolates the finite-sample property required by the outer-loop analysis. The first term is the linear function-approximation error. The second accounts for the mismatch between the full-history CPOMDP dynamics generating the samples and the finite-history superstate model whose values are estimated. The last two terms are respectively the initialization and finite-sample TD errors. Because rewards and costs have the same bounded range, the same form of the TD argument can be applied separately to the two critics.

The constants may depend on the fixed feature representation, projection radius, discount factor, and the relevant mixing and coverage properties, but not on K or τ. The form of the bound in this assumption follows the finite-time TD analysis of Anjarlekar et al. (2025): under persistent exploration and a corresponding mixing condition, their TD bound has the same structure as the one in Assumption 3. In our setting, we additionally require the same constants to apply uniformly over all policy generated by Algorithm 1 and all iteration budgets K. The exponential-weights update does not by itself guarantee the persistent exploration condition, so the resulting sample-complexity bound is conditional on these uniform coverage conditions. Establishing suficient conditions under which the policy sequence generated by Algorithm 1 satisfies the required uniform mixing and coverage properties is left for future work.

Since

$$
Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } = Q _ { r ^ { g } } ^ { \pi ^ { k } } - \lambda ^ { k } Q _ { c ^ { g } } ^ { \pi ^ { k } } , \qquad \widehat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } = \widehat { Q } _ { r ^ { g } } ^ { \pi ^ { k } } - \lambda ^ { k } \widehat { Q } _ { c ^ { g } } ^ { \pi ^ { k } } ,
$$

the triangle inequality and $\lambda ^ { k } \leq \lambda ^ { \operatorname* { m a x } }$ give

$$
\begin{array} { r } { \mathbb { E } \Big [ \Big \lVert \widehat { Q } _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } - Q _ { r ^ { g } - \lambda ^ { k } c ^ { g } } ^ { \pi ^ { k } } \Big \rVert _ { \infty } \Big ] \leq \delta _ { r ^ { g } } ( \tau , l ) + \lambda ^ { \operatorname* { m a x } } \delta _ { c ^ { g } } ( \tau , l ) . } \end{array}
$$

Moreover, by the definition of the estimated cost value,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left| \widehat { V } _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) - V _ { c ^ { g } } ^ { \pi ^ { k } } ( \rho ^ { g } ) \right| \right] \leq \mathbb { E } \left[ \left\| \widehat { Q } _ { c ^ { g } } ^ { \pi ^ { k } } - Q _ { c ^ { g } } ^ { \pi ^ { k } } \right\| _ { \infty } \right] } \\ & { \qquad \leq \delta _ { c ^ { g } } ( \tau , l ) . } \end{array}
$$

Consequently, the conditions of Estimation Oracle B are satisfied with

$$
\delta _ { \mathrm { e s t } } ( \tau , l ) : = \operatorname* { m a x } \left\{ \delta _ { r ^ { g } } ( \tau , l ) + \lambda ^ { \operatorname* { m a x } } \delta _ { c ^ { g } } ( \tau , l ) , \ : \ : \delta _ { c ^ { g } } ( \tau , l ) \right\} .
$$

A valid upper bound is

$$
\delta _ { \mathrm { e s t } } ( \tau , l ) \leq ( 1 + \lambda ^ { \operatorname* { m a x } } ) \operatorname* { m a x } \left\{ \delta _ { r ^ { g } } ( \tau , l ) , \delta _ { c ^ { g } } ( \tau , l ) \right\} .
$$

## E.2 Sample Complexity Bound for Fixed Window Length

For any fixed l and admissible superstate threshold $d ^ { g }$ , recall

$$
\delta ( l ) = \frac { 2 ( 1 - \kappa ) ^ { l } } { ( 1 - \gamma ) ^ { 2 } } , \qquad \xi ^ { g } = d ^ { g } - d ^ { o } + \xi ^ { o } - \delta ( l ) , \qquad \lambda ^ { \mathrm { m a x } } = \frac { 2 } { ( 1 - \gamma ) \xi ^ { g } } .
$$

Since the history length and feature representation are fixed, the superstate space, feature dimension, projection radius, and the relevant mixing and coverage constants do not depend on K or τ. For a general linear representation, R is a fixed design parameter defining the critic class.

The terms in Assumption 3 that do not vanish as τ increases induce the estimation-error floor

$$
\delta _ { \mathrm { e s t } } ^ { \mathrm { f a o r } } ( l , \lambda ^ { \mathrm { m a x } } ) : = \operatorname* { m a x } \Bigg \{ \varepsilon _ { \mathrm { F A } , r } + \lambda ^ { \mathrm { m a x } } \varepsilon _ { \mathrm { F A } , c } + \big ( C _ { r , l } ^ { \mathrm { h i s t } } + \lambda ^ { \mathrm { m a x } } C _ { c , l } ^ { \mathrm { h i s t } } \big ) \frac { ( 1 - \kappa ) ^ { l } } { 1 - \gamma } , \varepsilon _ { \mathrm { F A } , c } + C _ { c , l } ^ { \mathrm { h i s t } } \frac { ( 1 - \kappa ) ^ { l } } { 1 - \gamma } \Bigg \} .
$$

It follows from Assumption 3 that, for some fixed $c _ { l } > 0 .$

$$
\delta _ { \mathrm { e s t } } ( \tau , l ) \leq \delta _ { \mathrm { e s t } } ^ { \mathrm { f l o o r } } ( l , \lambda ^ { \mathrm { m a x } } ) + O _ { l } \left( e ^ { - c _ { l } \sqrt { \tau } } + \frac { \log \tau } { \sqrt { \tau } } \right) .
$$

Combining this inequality with Theorem 1, the average optimality gap satisfies

$$
\mathrm { O p } \mathrm { G a p } _ { K } \leq \underbracket { 2 \delta ( l ) + \frac { [ d ^ { o } - d ^ { g } + \delta ( l ) ] _ { + } } { ( 1 - \gamma ) \xi ^ { g } } + \frac { 2 ( \xi ^ { g } + 1 ) } { ( 1 - \gamma ) \xi ^ { g } } \delta ^ { \mathrm { f o o r } } ( l , \lambda ^ { \mathrm { m a x } } ) } _ { = : F _ { r } ( l , d ^ { o } ) } + O _ { l } \bigg ( e ^ { - c _ { l } \sqrt { \tau } } + \frac { \log \tau } { \sqrt { \tau } } \bigg ) + O _ { l } \big ( K ^ { - 1 / 2 } \big ) .
$$

Similarly, the average constraint violation satisfies

$$
\mathrm { C V i o l } _ { K } \le \underbrace { [ d ^ { g } - d ^ { o } + \delta ( l ) ] _ { + } + 2 ( \xi ^ { g } + 1 ) \delta _ { \mathrm { e s t } } ^ { \mathrm { f o o r } } ( l , \lambda ^ { \mathrm { m a x } } ) } _ { = : F _ { c } ( l , d ^ { o } ) } + O _ { l } \Big ( e ^ { - c _ { l } \sqrt { \tau } } + \frac { \log ( \tau ) } { \sqrt { \tau } } \Big ) + O _ { l } ( K ^ { - 1 / 2 } ) .
$$

The quantities $F _ { r } ( l , d ^ { g } )$ and $F _ { c } ( l , d ^ { g } )$ are respectively the fixed-window reward and constraint floors. One can consider, for example, the case in which $d ^ { g } = d ^ { o } - \delta ( l )$ or $d ^ { g } = d ^ { o }$ . They collect the CPOMDP-to-superstate approximation error, the linear function-approximation error, and the sampling-mismatch error of the TD oracle. The remaining terms are the finite-sample TD error and the finite-K optimization error. The notation $O _ { l }$ indicates that the hidden constants may depend on fixed quantities such as the feature representation, $R ,$ the mixing and coverage constants, $\xi ^ { g }$ $\lambda ^ { \mathrm { m a x } }$ , and $\gamma ,$ but not on K or τ.

## Define

$$
F _ { l } : = \operatorname* { m a x } \{ F _ { r } ( l , d ^ { g } ) , F _ { c } ( l , d ^ { g } ) \} .
$$

Because l is fixed, $F _ { l }$ does not decrease when more samples are collected. Therefore, the preceding bounds certify the target accuracy only when $\varepsilon > F _ { l }$ . For $\varepsilon \leq F _ { l }$ , the guarantee does not hold.

For any $\varepsilon > F _ { l }$ , define the remaining error budget

$$
\varepsilon _ { \mathrm { r e m } } : = \varepsilon - F _ { l } > 0 .
$$

It is suficient to choose K and $\tau \ \mathrm { s o }$ that the finite-sample TD and optimization errors are each of order $\varepsilon _ { \mathrm { r e m } }$ . For $0 < \varepsilon _ { \mathrm { r e m } } < 1$ , suficient choices are

$$
\tau = O _ { l } \left( \varepsilon _ { \mathrm { r e m } } ^ { - 2 } \log ^ { 2 } \frac { e } { \varepsilon _ { \mathrm { r e m } } } \right) , \qquad K = O _ { l } ( \varepsilon _ { \mathrm { r e m } } ^ { - 2 } ) .
$$

Each of the K calls to the oracle uses $\tau ^ { \prime }$ initial interactions followed by τ TD interactions. Since the same interactions update both critics, the total number of observation-action pairs is

$$
N _ { \varepsilon } = K \big ( \tau + \tau ^ { \prime } \big ) .
$$

Using $\tau ^ { \prime } = { \cal O } _ { l } ( \log ( \tau ) ) = o ( \tau )$ gives

$$
N _ { \varepsilon } = O _ { l } \left( \varepsilon _ { \mathrm { r e m } } ^ { - 4 } \log ^ { 2 } \frac { 1 } { \varepsilon _ { \mathrm { r e m } } } \right) .
$$

## F Experiments

## F.1 Experimental Setup

Hardware Specification Experiments are conducted on $\mathrm { A p p }$ le Mac Studio equipped with Apple M4 Max chip (16-core CPU, 50-core GPU), 64 GB unified memory.

Environment Details The state space consists of the grid cell coordinates. The action space is A = {up, right, down, left}. The observation space is O = {safe, edge, near-hazard, unsafe, goal}. The agent starts from the middle cell of the left boundary. The absorbing goal is located at the opposite of the grid. The agent receives reward +1 whenever the current state is the absorbing goal state, 0 otherwise. The cost is 1 when the current state is unsafe, 0 otherwise. The intended action is executed with probability 0.7, while with probability 0.3 one of the four action is sampled uniformly. Boundary-crossing actions leave the agent in place. The nominal observation is emitted with probability 0.9, while with probability 0.1 one of the five observations is sampled uniformly. We use the discount factor $\gamma = 0 . 9$

Algorithm Configuration We consider history lengths $l \in \{ 0 , 1 , 2 , 3 , 4 \}$ and iteration budgets $K \in \{ 1 0 0 , 5 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \}$ . The original and superstate constraint threshold are set to $d ^ { o } = d ^ { g } = 0 . 6 9 2 \colon$ this specific value is chosen so that, for every tested history length l, the corresponding superstate CMDP admits a strictly feasible policy, satisfying Slater condition. For every $l ,$ we verify strict feasibility of the corresponding superstate CMDP by computing $\begin{array} { r } { \hat { \xi } _ { l } ^ { g } = d ^ { g } - \operatorname* { m i n } _ { \pi \in \Pi ^ { g } } V _ { c ^ { g } } ^ { \pi } ( \rho ^ { g } ) > 0 . } \end{array}$ . Since the choice of $d ^ { g } = d ^ { o }$ is motivated by the fact the the filter stability coeficient κ, which is needed for the choice of $d ^ { g } = d ^ { o } - \delta ( l )$ , is not assumed to be known in the experiments. We still perform some experiments on the constraint tightening.

We initialize the policy uniformly over actions and set $\lambda ^ { 0 } = 0$ . The dual variable is projected onto $\begin{array} { r } { \left[ 0 , \frac { 2 } { ( 1 - \gamma ) \hat { \xi } _ { l } ^ { g } } \right] } \end{array}$ . All results in the main experiment use exact policy evaluation in the superstate CMDP.

Regarding the learning rates, in most experiments, we use $\eta _ { 1 } = \sqrt { \frac { 2 \log | \mathcal { A } | } { K } }$ and $\begin{array} { r } { \eta _ { 2 } = \frac { 1 } { ( ( 1 - \gamma ) ^ { - 1 } - d ^ { g } ) \sqrt { K } } . } \end{array}$ These have the same $K ^ { - 1 / 2 }$ dependence as those in Theorem 1, but use diferent multiplicative constants, alleviating the need of knowing critical parameters such as $\xi ^ { g }$ or $\xi ^ { o } , \kappa$ . Later in the appendix we will also analyze the case with exact constants. Our objective in these experiments is to study the practical behavior of the algorithm.

The primal rate is an exponential weights calibration based on the size of the action space. For the dual update, since $V _ { c ^ { g } } ^ { \pi } ( \rho ^ { g } ) - d ^ { g } \leq 1 / ( 1 - \gamma ) - d ^ { g }$ , our choice guarantees that the largest possible positive dual-update increment is scaled as

$$
\eta _ { 2 } \left( \frac { 1 } { 1 - \gamma } - d ^ { g } \right) = \frac { 1 } { \sqrt { K } } .
$$

This normalization makes the scale of the dual updates comparable across iteration budgets. Both rules are fixed before running and are used for all the history lengths l. For completeness, we also conduct additional experiments using exact learning rates from Theorem 1. The results are reported below.

For every pair $( l , K )$ , we evaluate the average reward value and average constraint violation of the K policy iterates in the original CPOMDP. At the beginning of each episode, one of the $K$ iterates is sampled uniformly and used throughout that episode. We use four independent random seeds, with 5,000 episodes per seed, for a total of 20,000 episodes per $( l , K )$ configuration. Since the policy-evaluation oracle is exact, the policy sequence is deterministic; the seeds afect only the Monte Carlo evaluation in the original CPOMDP. Each rollout has horizon $H = \textstyle \left\lceil { \frac { \log ( 1 0 ^ { - 7 } ) } { \log \gamma } } \right\rceil = 1 5 3$ The omitted discounted tail is therefore controlled by $\gamma ^ { H } \leq 1 0 ^ { - 7 }$

Construction of the SARSOP Reference Value and Visualization The purpose of this baseline is to obtain an approximation of the optimal constrained value $V _ { r ^ { o } } ^ { \pi ^ { \star , h } } ( \rho ^ { o } )$ in the original CPOMDP. We use SARSOP (Kurniawati et al., 2008), a point-based POMDP solver that approximates the belief space reachable under optimal policies. Unlike our algorithm, this baseline does not truncate histories and the complete belief over all 49 latent Gridworld states is updated using the exact transition and observation models.

Since SARSOP does not directly support constrained POMDPs, we consider the Lagrangian. For each $\lambda \geq 0$ , we solve the unconstrained problem with reward $r - \lambda c$ . We first solve it for $\lambda = 0$ and then evaluate $\lambda = 1 , 2 , 4 , . . .$ . until finding a policy satisfying

$$
\bar { V _ { c ^ { o } } ^ { \pi } } ( \rho ^ { o } ) + 2 \mathrm { S E } \big ( \bar { V _ { c ^ { o } } ^ { \pi } } ( \rho ^ { o } ) \big ) \leq d ^ { o } .
$$

where $\bar { V } _ { c ^ { o } } ^ { \pi } ( \rho ^ { o } )$ denote the empirical mean cost value computed from the evaluation rollouts, and $\mathrm { S E } ( \bar { V } _ { c ^ { o } } ^ { \pi } ( \rho ^ { o } ) )$ denote its standard error. The resulting multiplier interval is refined by at most 12 bisection steps, with relative tolerance $1 0 ^ { - 3 }$ . Each scalar problem is solved with target precision $1 0 ^ { - 2 }$ and a timeout of 1200 seconds. We then construct an episode-level mixture of one feasible and one infeasible candidate, choosing the mixture with the largest estimated reward while satisfying the constraint with a 1.96-standard-error margin. A final evaluation over 50,000 independent rollouts gives

$$
\bar { V } _ { r ^ { o } } ^ { \pi _ { \mathrm { m i x } } } ( \rho ) = 3 . 4 2 6 2 1 9 \pm 0 . 0 0 6 8 4 7 , \qquad \bar { V } _ { c ^ { o } } ^ { \pi _ { \mathrm { m i x } } } ( \rho ) = 0 . 6 8 4 4 1 7 \pm 0 . 0 0 1 8 2 7 ,
$$

where uncertainties denote standard errors. Moreover,

$$
0 . 6 8 4 4 1 7 + 1 . 9 6 ( 0 . 0 0 1 8 2 7 ) = 0 . 6 8 7 9 9 8 < d ^ { o } ,
$$

confirming feasibility at the adopted statistical margin. We use the resulting reward as a feasible SARSOP-based approximation of $V _ { r ^ { o } } ^ { \pi ^ { \star , h } } ( \rho ^ { o } )$

SARSOP returns a policy over belief states. We estimate its behavior at each latent cell from rollout frequencies. The arrow shows the frequency of each action.

![](images/07c43c69e42d436d7575529213ee68f989e00937b4497bb0c82b05a186e3863c.jpg)  
Figure 4: Performance of Algorithm 1 in the finite-history superstate model for diferent values of K and l.

## F.2 Other Results

Performance in the Superstate CMDP Fig. 4 reports the signed average optimality gap and average constraint violation in the superstate CMDP under exact policy evaluation. The constraint violation decreases with K for all tested history lengths. For some $l ,$ the positive average optimality gap approximately follows the displayed $K ^ { - 1 / 2 }$ reference slope. For other l values, it becomes negative. A negative signed gap can occur because the finite-K mixture remains infeasible, attaining a reward larger than the constrained optimum. Average optimality gap and average constraint violation curves must consequently be interpreted jointly. Overall, these curves are qualitatively compatible with the $O ( K ^ { - 1 / 2 } )$ optimization terms.

Efect of Implementing an Estimation Oracle We use persistent tabular TD as an empirical implementation of the estimation oracle. The experiment evaluates the practical efect of approximate policy evaluation, it is not intended as an empirical verification of an end-to-end TD sample-complexity bound. At each primal-dual iteration, the TD(0) oracle collects 64 trajectories of horizon 100 under the current policy. An additional 2000 warm-up trajectories are collected during the first iteration. The action-value tables persist across the primal-dual iterations, and, therefore, are not re-initialized. For the nth visit to a superstate-action pair we use as stepsize:

$$
\alpha _ { n } = \frac { 0 . 1 } { ( 1 + 0 . 0 1 n ) ^ { 1 / 2 } } .
$$

Results for $l \in \{ 0 , 1 , 2 , 3 , 4 \}$ and $K \in \{ 1 0 0 , 5 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \}$ are reported in Fig. 5. Compared with exact policy evaluation, the TD oracle introduces in general an additional bias in both reward and constraint performance. This is qualitatively consistent with the estimation-error terms ∝ $\delta _ { \mathrm { e s t } }$ appearing in the bounds. Since the TD estimates are retained across iterations, their actual errors vary with K. This illustrates the viability and the efect of approximate policy evaluation.

Efect of Superstate Constraint Tightening Fig. 6 illustrates the efect of tightening the superstate constraint for $l = 3 .$ , using smaller $d ^ { g }$ . Let

$$
\Delta _ { \mathrm { s l a c k } } : = d ^ { o } - \operatorname* { m i n } _ { \pi \in \Pi ^ { g } } V _ { c ^ { g } } ^ { \pi } ( \rho ^ { g } ) \simeq 0 . 4 7 5 3
$$

denote the available strict-feasibility slack. We set

$$
d _ { \alpha } ^ { g } = d ^ { o } - \alpha \Delta _ { \mathrm { s l a c k } } , \qquad \alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 \} ,
$$

obtaining

$$
d _ { \alpha } ^ { g } \in \{ 0 . 6 9 2 , 0 . 5 7 3 , 0 . 4 5 4 , 0 . 3 3 6 \} .
$$

![](images/18289c80591a9b485513e96095994442bc42708b9b9055726d956e8bf9097ca1.jpg)

![](images/90f6259fa110690abf34d3e923d5296609c73bb7819fe5fb1cb2bebda996fb72.jpg)  
Figure 5: Performance of using exact oracle vs TD(0) learning oracle in Algorithm 1. Average optimality gap (left) and average constraint violation (right). The iterates are evaluated in the CPOMDP.

![](images/d58f920012519f7b1d76af84a9bffa7fdc045aa58cfb386f48e732493a8635b8.jpg)

![](images/989fee9231450149038dc47f5d67b6f89a4f1c1217076bc8bff1899d11054d32.jpg)  
Figure 6: Efect of superstate constraint tightening. Average optimality gap (left) and average constraint violation (right). The iterates are evaluated in the CPOMDP. History length l = 3.

Hence, $d _ { 0 } ^ { g } = 0 . 6 9 2$ corresponds to the untightened problem, while smaller values impose increasingly conservative superstate constraints. The results show the expected reward-safety tradeof: stronger tightening reduces the cost incurred in the original CPOMDP, at the price of a lower reward.

## Fully Observable and Unconstrained Partially Observable Baselines

1. Model-Based Planning References. Table 1 compares three reference policies. The approximate constrained CPOMDP optimum achieves an expected discounted cumulative reward of 3.426 and an expected cumulative cost of 0.684. Removing the constraint, but keeping partial observability, increases the SARSOP reward to 3.898, but also raises the cost to 1.302, above the threshold $d ^ { o } \simeq 0 . 6 9 2$ . Under full observability, the exact constrained optimum achieves a reward of 4.162 with a cost of 0.692, showing the price of partial observability.

<table><tr><td>Policy</td><td>Reward</td><td>Cost</td></tr><tr><td>Approx. constrained CPOMDP</td><td> $3 . 4 2 6 \pm 0 . 0 1 3$ </td><td> $0 . 6 8 4 \pm 0 . 0 0 4$ </td></tr><tr><td>Approx. unconstrained POMDP</td><td> $3 . 8 9 8 \pm 0 . 0 2 0$ </td><td> $1 . 3 0 2 \pm 0 . 0 1 3$ </td></tr><tr><td>Exact constrained CMDP</td><td>4.162</td><td>0.692</td></tr></table>

Table 1: Model-based planning references. SARSOP-based values are reported with 95% confidence intervals. The constraint threshold is $d ^ { o } \simeq 0 . 6 9 2$

Fig. 7 illustrates the diference between the constrained partially observable policy and constrained fully observable policies. The former appears more conservative, tending to move toward the boundary of the grid rather than through the central corridor when the agent is in some unsafe cells. This behavior is consistent with the state uncertainty induced by noisy and aliased observations.

## 2. Unconstrained and Fully Observable Learning Baselines. Table 2 compares the

![](images/89897d9383e7f4ca22f131425cdd07ea97c29d6d44e84b1d2ff47c52e53f2299.jpg)

![](images/a6cd6ee9e85f0d56d5f4c641252d23e8936013d1f9d716b4c5153679df22cbd0.jpg)

![](images/1a8d632acb4a3df3323f27eb65875d793d252f93c4d3b2a3e7c521ef2344ff11.jpg)  
Figure 7: Approximate constrained CPOMDP policy obtained using SARSOP (left) and exact fully observable policy (right). Their respective expected cumulative reward values are 3.426 and 4.162, while their cost values are 0.684 and 0.692. Arrow lengths are proportional to action probabilities.

constrained and unconstrained variants of Algorithm 1 after $K = 1 0 ^ { 4 }$ iterations. The constrained variant yields lower constraint violation for every history length. The unconstrained values are not uniformly higher because the table reports finite-iteration average policies learned in the superstate model and evaluated in the original CPOMDP, rather than exact optimal policies. Hence, the usual ordering between constrained and unconstrained optimal values need not hold.

<table><tr><td>l</td><td>Average reward Constr. Unconstr.</td><td>Constr.</td><td>Constraint violation Unconstr.</td></tr><tr><td>0</td><td>1.599</td><td>0.685</td><td>0.682 1.158</td></tr><tr><td>1</td><td>1.885</td><td>1.889 0.541</td><td>0.695</td></tr><tr><td>2</td><td>2.750</td><td>2.901 0.331</td><td>0.840</td></tr><tr><td>3</td><td>3.175</td><td>2.958 0.299</td><td>1.504</td></tr><tr><td>4</td><td>3.329</td><td>3.534</td><td>0.237 0.922</td></tr></table>

Table 2: Constrained and unconstrained policies evaluated in the original CPOMDP after $K = 1 0 ^ { 4 }$ iterations. Reported values are averages over four random seeds. Constraint violations are computed relative to $d ^ { o } \simeq 0 . 6 9 2$

Finally, Fig. 8 compares Algorithm 1 performances under partial and full observability, showing that the performance gap decreases as l increases.

Matching the Theoretical stepsizes Figure 9 reports the performance of Algorithm 1 using the exact step-size constants from Theorem 1,

$$
\eta _ { 1 } = \frac { ( 1 - \gamma ) ^ { 2 } \xi ^ { g } } { ( 1 - \gamma ) \xi ^ { g } + 2 } \sqrt { \frac { 2 \log | { \cal A } | } { K } } , \qquad \eta _ { 2 } = \frac { 2 } { \xi ^ { g } ( 1 + ( 1 - \gamma ) d ^ { g } ) \sqrt { K } } .
$$

Note that here we report the average reward value rather than the average optimality gap for better visualization. Compared with the practical values used in the main experiments, the constraint violation decreases further, whereas the reward performances improve much more slowly. This is explained by the conservative theorem constants. For example, at $K = 1 0 0 0$ , the practical calibration gives $\eta _ { 1 } = 0 . 0 5 2 6$ and $\eta _ { 2 } ~ = ~ 0 . 0 0 3 4$ . With the theorem constants, $\eta _ { 1 }$ ranges from $1 . 7 4 \cdot 1 0 ^ { - 5 } \mathrm { ~ t o ~ } 1 . 3 8 \cdot 1 0 ^ { - 4 }$ , while $\eta _ { 2 }$ ranges from 0.110 to 0.894, depending on l through $\xi _ { l } ^ { g } .$ Thus, the theorem calibration makes the primal updates substantially smaller and the dual updates larger. This explains the stronger reduction in constraint violation and the considerably slower reward improvement and the diferences between curves associated with diferent l.

![](images/55fc605fda72f9c9a641a3bc4395ba8fd423c8dfbb7494dba6e8c1d6db374507.jpg)  
Figure 8: Performance of Algorithm 1 under partial observability for diferent values of K and l, compared with its fully observable performance (black line).

![](images/4a1c3b5e4482be9ca17d3146893175a7521b87fefeb2bf8a37a5ce6d845b1206.jpg)

![](images/d84a181b4bbadd3547ad3c16d0aaff577c944c3f4b21cca94c41d17a46a4d553.jpg)  
Figure 9: Performance of Algorithm 1 in the CPOMDP for diferent values of K and l using the stepsizes required by Theorem 1. Left: $\begin{array} { r } { \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } V _ { r ^ { o } } ^ { \pi ^ { k } } ( \rho ^ { o } ) } \end{array}$ . Right: $\begin{array} { r } { \left[ \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } V _ { c ^ { o } } ^ { \pi ^ { k } } ( \rho ^ { o } ) - d ^ { o } \right] _ { + } } \end{array}$