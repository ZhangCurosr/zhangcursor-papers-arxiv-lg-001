# Achieving an O(1/N) Optimality Gap in Average-Reward Weakly-Coupled MDPs

Yige Hong<sup>∗∥</sup>, Xiangcheng Zhang<sup>†∥</sup>, Qiaomin Xie<sup>‡</sup>, Yudong Chen<sup>§</sup>, and Weina Wang<sup>¶</sup> <sup>∗</sup>H. Milton Stewart School of Industrial & Systems Engineering, Georgia Institute of Technology yhong320@gatech.edu

<sup>†</sup>John A. Paulson School of Engineering and Applied Sciences, Harvard University xiangchengzhang@fas.harvard.edu

<sup>‡</sup>Department of Industrial and Systems Engineering, University of Wisconsin–Madison qiaomin.xie@wisc.edu

<sup>§</sup>Department of Computer Sciences, University of Wisconsin–Madison yudongchen@cs.wisc.edu

<sup>¶</sup>Computer Science Department, Carnegie Mellon University weinaw@cs.cmu.edu

Abstract—We study average-reward weakly-coupled Markov decision processes (WCMDPs), where a WCMDP consists of N smaller MDPs, called arms, that share multiple per-step budget constraints. We consider the setting where the arms have identical model parameters, multiple actions, and state- and action-dependent costs. For restless bandits (RBs), a well-studied special case of WCMDPs, prior work has developed policies that achieve an $O ( 1 / \sqrt { N } )$ optimality gap under general conditions, and has further identified conditions under which policies can achieve a better-than-1 $. / \sqrt { N }$ optimality gap. However, for general WCMDPs, no prior result achieves an optimality gap better than $1 / { \sqrt { N } } .$ . In this paper, we identify conditions analogous to those for RBs under which a better-than- $1 / \sqrt { N }$ optimality gap is achievable, and design a policy that attains an ${ \cal O } ( \bar { 1 } / N )$ optimality gap. Notably, unlike prior approaches based on generalizing priority orderings, our policy is not priority-based but rather is designed to induce locally linear mean-field dynamics.

## I. INTRODUCTION

## A. Motivation

Weakly-coupled Markov decision processes (WCMDPs) [1] model sequential resource allocation among many interacting components. A WCMDP consists of N smaller Markov decision processes, called arms, whose state transitions are independent conditional on their current states and chosen actions. The coupling arises through shared budget constraints: each action incurs costs of one or more types, and the total cost of each type across all arms must remain within its budget at every time step. This structure appears in applications such as online advertising [2], healthcare resource allocation [3], surveillance [4], and machine maintenance [5]. With known model parameters, we maximize long-run average reward per arm. The optimality gap is the difference between the optimal reward per arm and that achieved by a policy; we study its order as N grows.

A widely studied special case is the restless bandit (RB) problem [6]. Each arm has two actions, active and passive, and a single budget limits the number of arms that can be activated at each time step. An arm’s state can evolve under either action. For a broad review of RBs, their applications, and index policies, see Niño-Mora [7]. A substantial literature establishes asymptotically optimal policies, whose optimality gaps vanish as $N \to \infty$ . Results include o(1) gaps [8]–[10] and $O ( 1 / \sqrt { N } )$ gaps [11], [12], under different structural assumptions.

For RBs, gaps of smaller order than $1 / \sqrt { N }$ are already possible under suitable assumptions. The papers [13], [14] establish $O ( \exp ( - C N ) )$ gaps for suitable priority policies derived from a linear programming (LP) relaxation, where $C > 0$ is a constant independent of N. The key assumptions are the aperiodic unichain condition, non-degeneracy, and a uniform global attractor property (UGAP). Our prior work [15] achieves the same exponential order using a two-set policy, replacing UGAP with a weaker and easier-to-verify local stability condition near an optimal stationary distribution.

These RB results motivate seeking similarly small optimality gaps for general WCMDPs, which allow more than two actions, multiple budget constraints, and costs that depend on both state and action. Prior work on average-reward WCMDPs establishes o(1) gaps for the special case with multiple actions but a single budget [9], [16], [17], and for WCMDPs with multiple actions and budget constraints [18]. More recent work [19] establishes an $O ( 1 / \sqrt { N } )$ gap for WCMDPs with multiple actions, multiple budget constraints, and fully heterogeneous arms, whose model parameters may differ across arms. This guarantee also applies to homogeneous systems, where all arms share the same model parameters. However, to our knowledge, an optimality gap of smaller order than $1 / \sqrt { N }$ has not been established for average-reward WCMDPs with multiple budget constraints and state- and action-dependent costs.

This paper establishes an $O ( 1 / N )$ optimality gap for average-reward WCMDPs with homogeneous arms, multiple actions, multiple budget constraints, and state-dependent costs. We assume the aperiodic unichain condition, non-degeneracy, and local stability; the last two are additional assumptions relative to the ${ \cal O } ( 1 / \sqrt { N } )$ guarantee in [19]. We achieve this bound by extending the two-set policy of [15] from RBs to WCMDPs. We do not require a global attractor assumption.

## B. Technical insight

When generalizing from RBs to WCMDPs, much of the prior work has focused on adapting the notion of state priorities to define priority policies [9], [16], [17]. These policies assign priorities to states and favor arms in higher-priority states when choosing actions that incur positive costs. This approach is natural when there is a single budget constraint and the costs are state independent. However, with multiple budget constraints and state-dependent costs, a single priority order no longer directly specifies how to balance the competing budget requirements.

In this paper, our main technical insight is that the priority structure itself is not what enables an optimality gap smaller than $1 / \sqrt { N }$ . Rather, the key contributor is the local linearity of the mean-field dynamics induced by the policy. Therefore, to generalize this approach to WCMDPs and design a policy that achieves an optimality gap smaller than $1 / \sqrt { N }$ , we seek a policy whose mean-field dynamics is locally linear. We elaborate on this below.

For an RB, consider the empirical state distribution that records the fraction of arms in each state, and its mean-field approximation where the random transitions are replaced by their conditional means. Let $\mu ^ { * }$ denote the stationary state distribution prescribed by an optimal LP solution; it is a fixed point of the mean-field dynamics under these priority policies. We can bound the optimality gap by the expected distance between the empirical state distribution and $\mu ^ { * }$ The empirical state distribution typically fluctuates on the $1 / \sqrt { N }$ scale suggested by the Central Limit Theorem (CLT) for conditionally independent arm transitions. However, prior work [13], [14] shows that, near $\mu ^ { * }$ , the reward and mean transition depend linearly on the empirical state distribution. This observation allows the analysis to compare the expected empirical state distribution with $\mu ^ { * }$ . Since random deviations can cancel in expectation, this difference can be of smaller order than $1 / \sqrt { N }$ . Together with concentration near $\mu ^ { * }$ , local linearity therefore allows optimality gaps smaller than the CLT fluctuation scale.

In this paper, we extend the local linearity structure directly, retaining the two-set approach of [15] without forcing a priority policy. The challenge is to preserve local linearity while satisfying multiple budget constraints with state-dependent costs. We construct an auxiliary system of linear equations (Equation (6)) to determine target state-action frequencies (the desired fractions of all arms in each state-action pair). When the empirical state distribution is sufficiently close to $\mu ^ { * }$ , this system yields feasible target frequencies that depend linearly on that distribution. Rounding the target frequencies to feasible integer action counts introduces an $O ( 1 / N )$ residual in the reward and mean transition under our construction. The rounding residual contributes the $O ( 1 / N )$ term in our gap bound, which is smaller than the $O ( 1 / \sqrt { N } )$ gap previously established for general average-reward WCMDPs.

## C. Additional related work

Model predictive control (MPC) provides another approach to the average-reward setting by repeatedly solving a finitehorizon LP and implementing only the first action allocation. Under a mixing assumption, Gast and Narasimha [20] bound the optimality gap for homogeneous RBs by the sum of a planning-error term that can be reduced by increasing the horizon and an $O ( 1 / \sqrt { N } )$ term. The latter term becomes exponentially small under additional non-degeneracy, LP uniqueness, and local stability conditions, assuming an integer activation budget. They also discuss a general WCMDP extension, without proving the sharper bound in that setting. Narasimha and Gast [21] study fully heterogeneous systems; their WCMDP guarantee assumes a conjecture that the horizon needed for a fixed planning-error tolerance can be bounded independently of N.

So far we have discussed prior work under the longrun average-reward criterion. For a fixed finite horizon, with performance measured by expected total reward per arm over that horizon, optimality gaps of smaller order than $1 / \sqrt { N }$ are available under suitable non-degeneracy conditions. For homogeneous RBs, Zhang and Frazier [22] obtain an $O ( 1 / N )$ gap, which Gast, Gaujal, and Yan [14] improve to $O ( \exp ( - C N ) )$ ) with an integer activation budget. For general WCMDPs, $O ( 1 / N )$ guarantees cover both homogeneous systems [23] and systems with typed heterogeneity, which permits a fixed number of distinct arm types as N grows [24]. Zhang [25] further develops bounds for fully heterogeneous systems that improve with the degree of non-degeneracy. Recently, Yan, Wang, and Ying [26] also establish a ${ \widetilde O } ( 1 \dot { / } N )$ gap for RBs without requiring non-degeneracy conditions, where $\widetilde O$ suppresses logarithmic factors in N. These fixed-horizon guarantees can not be directly used to establish an average-reward guarantee: the bounds can grow faster than linearly with the horizon, and the policies themselves depend on the horizon. Even when additional assumptions give bounds with only linear horizon dependence, the policies still solve optimization problems whose sizes grow with the horizon [23], [24].

The LP-update policy of [23] inspires our design. Both approaches use LP solutions to construct target state-action frequencies and exploit local linearity in the current state distribution. Their LP plans the remaining finite-horizon meanfield evolution; we solve for stationary state-action frequencies and then use Equation (6), without planning a future trajectory.

## II. PROBLEM FORMULATION

We consider a WCMDP that consists of N homogeneous Markov decision processes (MDPs), indexed by $\begin{array} { r l } { [ N ] } & { { } = } \end{array}$ $\{ 1 , \ldots , N \}$ , each specified by $( \mathbb { S } , \mathbb { A } , P , r )$ . The state space S and action space $\mathbb { A } = \{ 0 , 1 , \ldots , | \mathbb { A } | - 1 \}$ are finite. The transition kernel is $P : \mathbb { S } \times \mathbb { A } \times \mathbb { S }  [ 0 , 1 ]$ , and $r : \mathbb { S } \times \mathbb { A } \to \mathbb { R }$ is the reward function. We call each MDP an arm and set $r _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { s , a } | r ( s , a ) |$ . Model parameters are known. Conditional on the current states and chosen actions, arms transition independently according to P. A policy π may be randomized and history-dependent. We write $S _ { t } ^ { \pi } ~ = ~ ( S _ { t } ^ { \pi } ( i ) ) _ { i \in [ N ] }$ and $A _ { t } ^ { \pi } = ( A _ { t } ^ { \pi } ( i ) ) _ { i \in [ N ] }$ for the state and action vectors.

The WCMDP system has K budget constraints, specified by cost functions $c _ { k } \colon \mathbb { S } \times \mathbb { A } \to [ 0 , 1 ]$ and budget levels $\alpha _ { k } \in ( 0 , 1 )$ for $k \in [ K ]$ as follows. We assume that the action 0 incurs zero cost of any type, i.e., $c _ { k } ( s , 0 ) = 0$ for all $s \in \mathbb { S }$ and $k \in [ K ]$ . A policy π for the N-armed system is feasible if it satisfies, for all $t \geq 0$ and all $k \in [ K ]$

$$
\frac { 1 } { N } \sum _ { i \in [ N ] } c _ { k } ( S _ { t } ^ { \pi } ( i ) , A _ { t } ^ { \pi } ( i ) ) \leq \alpha _ { k } ,\tag{1}
$$

where $S _ { t } ^ { \pi } ( i ) , A _ { t } ^ { \pi } ( i )$ are the state and action of arm i at time t under the policy π.

For a fixed initial state vector $S _ { \mathrm { 0 } }$ , we define the limsup and liminf average rewards per arm by

$$
\begin{array} { r l } & { R ^ { + } ( \pi , \pmb { S } _ { 0 } ) = \underset { T  \infty } { \operatorname* { l i m } \operatorname* { s u p } } \frac { 1 } { N T } \overset { T - 1 } { \underset { t = 0 } { \sum } } \underset { i \in [ N ] } { \sum } \mathbb { E } [ r \big ( \boldsymbol { S } _ { t } ^ { \pi } ( i ) , \boldsymbol { A } _ { t } ^ { \pi } ( i ) \big ) ] , } \\ & { R ^ { - } ( \pi , \pmb { S } _ { 0 } ) = \underset { T  \infty } { \operatorname* { l i m } \operatorname* { i n f } } \frac { 1 } { N T } \overset { T - 1 } { \underset { t = 0 } { \sum } } \underset { i \in [ N ] } { \sum } \mathbb { E } [ r \big ( \boldsymbol { S } _ { t } ^ { \pi } ( i ) , \boldsymbol { A } _ { t } ^ { \pi } ( i ) \big ) ] . } \end{array}
$$

When these agree, their common value is the long-run average reward $R ( \pi , S _ { 0 } )$ . Our objective is to maximize $R ^ { - } ( \pi , S _ { 0 } )$ over feasible policies; its optimal value is $R ^ { * } ( N , S _ { 0 } )$ . Stationary policies on a finite augmented state space have a well-defined long-run average reward, and finite-state MDPs admit an optimal stationary policy [27, Chapter 8 and Theorem 9.1.8]. For policies with a well-defined reward, the optimality gap is $R ^ { * } ( N , S _ { 0 } ) - R ( \pi , S _ { 0 } )$

a) Scaled counts and notation.: For $D \subseteq [ N ]$ , we let $m ( D ) = | D | / N$ and

$$
\begin{array} { r l } & { X _ { t } ^ { \pi } ( D , s ) = \displaystyle \frac { 1 } { N } \sum _ { i \in D } \mathbb { 1 } \{ S _ { t } ^ { \pi } ( i ) = s \} , } \\ & { Y _ { t } ^ { \pi } ( D , s , a ) = \displaystyle \frac { 1 } { N } \sum _ { i \in D } \mathbb { 1 } \{ S _ { t } ^ { \pi } ( i ) = s , \ A _ { t } ^ { \pi } ( i ) = a \} . } \end{array}
$$

We write $X _ { t } ^ { \pi } ( D )$ and $Y _ { t } ^ { \pi } ( D )$ for the corresponding row vectors, and $Y _ { t } ^ { \pi } = Y _ { t } ^ { \pi } ( [ N ] )$ . The set function $X _ { t } ^ { \pi }$ contains the same information as $S _ { t } ^ { \pi }$ . For nonempty D, $X _ { t } ^ { \pi } ( D ) / m ( D )$ is its empirical state distribution. Counts for an empty subset are zero. We write $\Delta ( \mathbb { S } )$ for the probability simplex, $I _ { k }$ for the k-by-k identity matrix, and 1 for the all-one row vector indexed by states. Distributions are row vectors, and policy superscripts are omitted when clear.

Input: A subset D of arms with state counts   
$( | D | x ( s ) ) _ { s \in \mathbb { S } }$ , optimal single-armed policy $\bar { \pi } ^ { * }$   
1: for $s \in \mathbb { S }$ do   
Independently assign each arm in state s   
an action $a \sim \bar { \pi } ^ { * } ( \cdot \mid s )$  
Algorithm 1. π¯<sup>∗</sup>-Fixed-Ratio Control

## A. LP relaxation and assumptions

We relax the per-step budget constraints (1) to their timeaverage counterparts, obtaining the following LP relaxation:

$$
\operatorname* { m a x i m i z e } _ { \{ y ( s , a ) \} _ { s \in \mathbb { S } , a \in \mathbb { A } } } \ \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } r ( s , a ) y ( s , a )\tag{LP-W}
$$

$$
\mathrm { s u b j e c t ~ t o } \quad \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y ( s , a ) \leq \alpha _ { k } , \quad \forall k \in [ K ] ,\tag{2}
$$

$$
\sum _ { s ^ { \prime } \in \mathbb { S } , a \in \mathbb { A } } y ( s ^ { \prime } , a ) P ( s ^ { \prime } , a , s ) = \sum _ { a \in \mathbb { A } } y ( s , a ) , \forall s \in \mathbb { S } ,\tag{3}
$$

$$
\sum _ { s ^ { \prime } \in \mathbb { S } , a ^ { \prime } \in \mathbb { A } } y ( s ^ { \prime } , a ^ { \prime } ) = 1 , \quad y ( s , a ) \geq 0 , \forall ( s , a ) .\tag{4}
$$

We let $R ^ { \mathrm { r e l } }$ denote the optimal value of the LP relaxation. A standard argument shows that $R ^ { \mathrm { r e l } } ~ \ge ~ R ^ { * } ( N , S _ { 0 } )$ [19]. Fixing an optimal solution of (LP-W), $y ^ { * }$ , we define $\mu ^ { \ast } ( s ) \triangleq$ $\textstyle \sum _ { a \in { \mathbb { A } } } y ^ { * } ( s , a )$ as the induced optimal stationary distribution over S. We let $S ^ { \varnothing } \ \triangleq \ \{ s \in \mathbb { S } \colon \mu ^ { * } ( s ) = 0 \}$ be the set of transient states under $\bar { \pi } ^ { * }$ . We define the optimal single-armed policy associated with $y ^ { * }$ by

$$
\begin{array} { r } { \bar { \pi } ^ { * } ( a \mid s ) = \left\{ \begin{array} { l l } { y ^ { * } ( s , a ) / \mu ^ { * } ( s ) , } & { \mu ^ { * } ( s ) > 0 , } \\ { 1 / | \mathbb { A } | , } & { s \in S ^ { \varnothing } , } \end{array} \right. } \end{array}\tag{5}
$$

with transition matrix $\begin{array} { r } { P _ { \bar { \pi } ^ { * } } ( s , s ^ { \prime } ) = \sum _ { a \in \mathbb { A } } \bar { \pi } ^ { * } ( a \mid s ) P ( s , a , s ^ { \prime } ) } \end{array}$

Assumption 1 (Aperiodic unichain). The transition matrix $P _ { \bar { \pi } }$ ∗ induced by $\bar { \pi } ^ { * }$ has a simple eigenvalue 1; all other eigenvalues have modulus strictly less than 1; i.e., $P _ { \bar { \pi } }$ ∗ defines an aperiodic unichain on S.

Assumption 1 depends only on the single-armed MDP. The two remaining assumptions, non-degeneracy (Assumption 2) and local stability (Assumption 3), are stated in Section III-B, where they arise naturally alongside the construction of the subroutine LP fixed-point. When there are multiple optimal solutions for (LP-W), we choose a fixed one that satisfies all assumptions.

## III. TWO-SET POLICY

The two-set policy combines two subroutines with complementary roles: $\bar { \pi } ^ { * } .$ -Fixed-Ratio Control steers a subset’s empirical state distribution toward $\mu ^ { * }$ , while LP fixed-point provides near-optimal control for a subset whose empirical state distribution is already close to $\mu ^ { * }$ . By combining these subroutines, the policy seeks to enlarge the subset following LP fixed-point. We first describe the two subroutines and then explain how the policy selects their respective subsets.

A. $\bar { \pi } ^ { * }$ -Fixed-Ratio Control

The subroutine $\bar { \pi } ^ { * }$ -Fixed-Ratio Control (Algorithm 1) assigns each arm an action by independent sampling from $\bar { \pi } ^ { * } ( \cdot \mid s )$

By Assumption 1, if all N arms follow Algorithm 1, their expected empirical state distribution evolves according to the transition matrix $P _ { \bar { \pi } ^ { * } }$ and converges to $\mu ^ { * }$ . In the two-set policy, this subroutine is applied to a subset of arms for this purpose.

## B. LP fixed-point

Next, we define the second subroutine, LP fixed-point. It is defined to be a locally linear and near-optimal control when the system’s empirical state distribution $x ( [ N ] )$ is close to $\mu ^ { * }$ . To construct this control, we compute a target stateaction frequency y as the solution of a linear system and use it to guide the actions in the next time step. Because the construction of the subroutine relies on the optimal solution to the LP relaxation, $y ^ { * }$ , which can be interpreted as an optimal fixed-point state-action frequency under the mean transitions, we refer to this subroutine as $L P$ fixed-point. LP fixed-point is inspired by the “LP-update” policy in the finite-horizon WCMDP literature [23].

Section III-B1 states the non-degeneracy and local stability assumptions (Assumptions 2 and 3). Section III-B2 gives the subroutine, Section III-B3 its dynamics, and Section III-B4 its feasibility conditions.

1) Two assumptions: non-degeneracy and local stability: We introduce two assumptions under which LP fixed-point is well-defined and induces locally stable dynamics.

For the fixed optimal LP solution $y ^ { * }$ , let $S ^ { \emptyset }$ be the set of transient states, $\kappa ^ { * }$ the set of tight budget constraints, and $\ b { \mathcal { U } } ^ { * }$ the set of pairs $( s , a )$ with $s \not \in { \bar { S } } ^ { \varnothing }$ and $y ^ { * } ( s , a ) = 0 :$

$$
\begin{array} { r l } & { S ^ { \varnothing } = \big \{ s \in \mathbb { S } \colon \sum _ { a \in \mathbb { A } } y ^ { * } ( s , a ) = 0 \big \} , } \\ & { { \mathcal K } ^ { * } \triangleq \big \{ k \in [ K ] \colon \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y ^ { * } ( s , a ) = \alpha _ { k } \big \} , } \\ & { { \mathcal U } ^ { * } \triangleq \big \{ ( s , a ) \in \mathbb { S } \times \mathbb { A } \colon s \notin S ^ { \varnothing } , \ y ^ { * } ( s , a ) = 0 \big \} . } \end{array}
$$

For the rest of this subsection, we apply LP fixed-point to all N arms, writing $x = X _ { t } ( [ N ] ) \in \Delta ( \mathbb { S } )$ for their empirical state distribution and $y \in \Delta ( \mathbb { S } \times \mathbb { A } )$ for a state-action distribution. Section III-C extends the subroutine to arbitrary subsets of arms.

We consider the following linear system in $y \in \mathbb { R } ^ { | \mathbb { S } | \times | \mathbb { A } | }$ ， parameterized by the empirical state distribution x:

$$
\begin{array} { r l r } & { y ( s , a ) = 0 , } & { ( s , a ) \in \mathcal { U } ^ { * } , } \\ & { y ( s , 0 ) = y ( s , 1 ) = \cdots = y ( s , \vert \mathbb { A } \vert - 1 ) , } & { s \in S ^ { \emptyset } , } \\ & { \displaystyle \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y ( s , a ) = \alpha _ { k } , } & { k \in K ^ { * } , } \\ & { \displaystyle \sum _ { a \in \mathbb { A } } y ( s , a ) = x ( s ) , } & { s \in \mathbb { S } . } \end{array}\tag{6}
$$

A solution y to (6) gives the target state-action frequency for LP fixed-point at this time step. On a high level, the purpose of these constraints is to find a budget-feasible state-action frequency that recovers $y ^ { * }$ when $x = \mu ^ { * }$ . Specifically, the first row of equations requires $y$ to share the support of $y ^ { * }$ The second row fixes the action distribution to be uniform conditional on any state in $S ^ { \emptyset }$ . The third row requires the same budgets to be tight under y and $y ^ { * }$ . The fourth row enforces consistency with the instantaneous state count x.

The equations in (6) are fundamentally different from the constraints of the LP relaxation (LP-W). In particular, the equations here require y to be an instantaneous state-action frequency consistent with the current state distribution $x ,$ whereas $y ^ { * }$ is a stationary state-action frequency satisfying the flow-balance equation (3). The first and second rows of (6) also have no counterparts in $( \mathrm { L P - W } )$ . Despite these differences, (6) recovers $y ^ { * }$ when $x = \mu ^ { * }$

We write this linear system in vector form as

$$
y C ^ { \ast } = ( 0 , 0 , ( \alpha _ { k } ) _ { k \in \mathcal { K } ^ { \ast } } , x ) ,\tag{7}
$$

where y and x are regarded as row vectors, $C ^ { * } \in \mathbb { R } ^ { | \mathbb { S } | | \mathbb { A } | \times d }$ denotes the coefficient matrix of this linear system, and $d =$ $| \mathcal { U } ^ { * } | + ( | \mathbb { A } | - 1 ) | S ^ { \varnothing } | + | K ^ { * } | + | \mathbb { S } |$ . A sufficient condition for (7) to have a solution $y$ for every $x \in \Delta ( \mathbb { S } )$ is that $C ^ { * }$ has full column rank.

Assumption 2 (Non-degeneracy). The matrix $C ^ { * }$ has full column rank.

A necessary condition for Assumption 2 is $d \leq | \mathbb { S } | | \mathbb { A } | \colon C ^ { * }$ must be square or tall. Under Assumption 1, this dimensional condition holds when $| S ^ { \varnothing } | = 0$ and $y ^ { * }$ is a non-degenerate basic feasible solution of (LP-W) (see Proposition 1 in $\mathsf { A p - }$ pendix A-A).

Assumption 2 can be viewed as a strengthened version of $d \leq | \mathbb { S } | | \mathbb { A } |$ , as it additionally requires the columns of $C ^ { * }$ to be linearly independent.

Remark 1 (Restless-bandit special case). To gain intuition for (6) and Assumption 2, we consider the RB specialization $| \mathbb { A } | = 2 , \mathcal { K } ^ { * } = \{ 1 \} , c _ { 1 } ( s , a ) = a .$ with $S ^ { 0 } = \emptyset$ . The state space partition as $\mathbb { S } = \bar { S } ^ { + } \cup \bar { S } ^ { - } \cup S ^ { 0 }$ , where $S ^ { + } = \{ s \colon y ^ { * } ( s , 1 ) >$ $0 , y ^ { * } ( s , 0 ) = 0 \} , S ^ { - } = \{ s \colon y ^ { * } ( s , 0 ) > 0 , y ^ { * } ( s , 1 ) = 0 \}$ , and the neutral states $S ^ { 0 } = \{ s \colon y ^ { * } ( s , 0 ) > 0 , y ^ { * } ( s , 1 ) > 0 \}$ . Then $\mathcal { U } ^ { * } = \{ ( s , 0 ) \colon s \in S ^ { + } \} \cup \{ ( s , 1 ) \colon s \in S ^ { - } \}$ , and the linear system (6) specializes into

$$
\begin{array} { r l r } & { } & { y ( s , 0 ) = 0 ( s \in S ^ { + } ) , ~ y ( s , 1 ) = 0 ( s \in S ^ { - } ) , } \\ & { } & { \sum _ { s } y ( s , 1 ) = \alpha , ~ y ( s , 0 ) + y ( s , 1 ) = x ( s ) ( s \in \mathbb { S } ) . } \end{array}
$$

The first two set of equtions pin down $y ( s , 1 ) = x ( s )$ for $s \in S ^ { + }$ and $y ( s , 0 ) = x ( s )$ for $s \in S ^ { - }$ . To find $y ( s , a )$ for $s \in S ^ { 0 }$ , we discuss based on the number of neutral states.

• When there is exactly one neutral state s˜ (the usual nondegenerate RB case), we can use the budget constraint to obtain $\begin{array} { r } { y ( \tilde { s } , 1 ) = \alpha - \sum _ { s \in S ^ { + } } x ( s ) } \end{array}$ and use the marginal constraint to obtain $y ( \tilde { s } , \bar { 0 } ) = x ( \tilde { s } ) - y ( \tilde { s } , 1 )$ ). In this case, the solution $y$ is uniquely determined for any x, implying that $C ^ { * }$ is an invertible matrix and thus has full column rank (i.e., Assumption 2 holds). This is consistent with the dimensionality of $C ^ { * } \colon$ one can verify that $d = 2 | \mathbb { S } | =$ $| \mathbb { S } | | \mathbb { A } |$ , so $C ^ { * }$ is a square matrix.

• When there is no neutral state, the linear system is overdetermined with $| S ^ { + } | + | S ^ { - } | + 1 + | \mathbb { S } | = 2 | \mathbb { S } | + 1 > | \mathbb { S } | | \mathbb { A } |$ equations, so it does not have solutions for all $x ; C ^ { * }$ is wide and cannot have full column rank.

• When there is more than one neutral state for RBs, the solution y always exists but is not unique, so $C ^ { * }$ still has full column rank and Assumption 2 still holds. In this case, we will select a fixed solution y, as we discuss immediately below while setting up for the next assumption.

When there is exactly one neutral state, the affine target allocates action 1 to the states in $S ^ { + }$ and uses the neutral state to meet the budget constraint.

Next, we prepare to state the local stability assumption. To this end, we need to fix a solution of $( 7 )$ . First, we note that at $x = \mu ^ { * }$ , one such solution is $y ^ { * }$ by definition, giving

$$
y ^ { * } C ^ { * } = ( 0 , 0 , ( \alpha _ { k } ) _ { k \in \mathcal { K } ^ { * } } , \mu ^ { * } ) .\tag{8}
$$

For general x, we need to invert the matrix $C ^ { * }$ . Since $C ^ { * }$ has full column rank, it admits a left inverse $C ^ { + } \in \mathbb { R } ^ { d \times | \mathbb { S } | | \mathbb { A } | }$ satisfying $C ^ { + } C ^ { * } = I _ { d } . { } ^ { 1 }$ We denote by $C _ { \mathbb { S } } ^ { + } ~ \in ~ \mathbb { R } ^ { | \mathbb { S } | \times | \mathbb { S } | | \mathbb { A } | }$ the |S| rows of $C ^ { + }$ corresponding to the marginal-distribution block (the last row of equations in (6)); then

$$
C _ { \otimes } ^ { + } C ^ { * } = ( 0 , 0 , 0 , I _ { | \mathbb { S } | } ) ,\tag{9}
$$

Therefore, the following y is a solution of (7):

$$
y \triangleq y ^ { \ast } + \left( x - \mu ^ { \ast } \right) C _ { \mathbb { S } } ^ { + } .\tag{10}
$$

To verify this, we right-multiply (10) by $C ^ { * }$ and apply (8) together with (9), obtaining

$$
\begin{array} { l } { y C ^ { * } = y ^ { * } C ^ { * } + \left( x - \mu ^ { * } \right) C _ { \mathfrak { S } } ^ { + } C ^ { * } } \\ { \ = \left( 0 , 0 , \left( \alpha _ { k } \right) _ { k \in \mathcal { K } ^ { * } } , \mu ^ { * } \right) + \left( 0 , 0 , 0 , x - \mu ^ { * } \right) } \\ { \ = \left( 0 , 0 , \left( \alpha _ { k } \right) _ { k \in \mathcal { K } ^ { * } } , x \right) , } \end{array}
$$

as required. We will select this particular y as the solution of (7) in the rest of the subsection and use it to construct the LP fixed-point subroutine.

Our next assumption concerns the dynamics induced by the state-action frequency y in (10). We let $\mathcal { P } \in \mathbb { R } ^ { ( | \mathbb { S } | | \mathbb { A } | ) \times | \overline { { \mathbb { S } } } | }$ be the transition matrix with entries $\mathcal { P } _ { ( s , a ) , s ^ { \prime } } = P ( s , a , s ^ { \prime } )$ , and define

$$
\Phi \triangleq C _ { \mathbb { S } } ^ { + } \mathcal { P } - \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } ,\tag{11}
$$

where $\mathbb { 1 } ^ { \top } \in \mathbb { R } ^ { \left. \mathbb { S } \right. }$ is the all-ones column vector. We will show in Section III-B3 that Φ captures the one-step transition of $X _ { t } ( [ N ] ) - \mu ^ { * }$ under the subroutine. We assume the following:

Assumption 3 (Local stability). The spectral radius of Φ defined in (11) is strictly less than 1.

Intuitively, local stability gives LP fixed-point a restoring effect near $\mu ^ { * }$ . Under the locally linear mean dynamics, an

Input: A set of n arms with number of arms   
in each state $( z ( s ) ) _ { s \in \mathbb { S } } ,$ , LP solution $y ^ { * } ,$   
left-inverse submatrix $C _ { \mathbb { S } } ^ { + }$ , feasibility radius η   
If $n = 0 ,$ return without assigning actions.   
Assert Assumption 2 and $\Vert z / n - \mu ^ { * } \Vert _ { U } \leq \eta$   
1: Compute $y  y ^ { * } + ( z / n - \mu ^ { * } ) C _ { \mathbb { S } } ^ { + }$   
2: for each $s \in \mathbb { S }$ and each $a \in \mathbb { A } \setminus \tilde { \{ 0 \} }$ do   
3: Uniformly choose $\lfloor n y ( s , a ) \rfloor$ unassigned   
arms in state s and assign them action a   
4: Assign action 0 to all remaining arms

## Algorithm 2. LP fixed-point

initial deviation v from $\mu ^ { * }$ evolves as $v \Phi ^ { t } ;$ ; the spectral-radius condition ensures that these deviations decay geometrically over time.

The purpose of centering term $- \mathbf { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P }$ in Φ is to replace the trivial eigenvalue 1 of $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ with 0 while preserving the effect of the operator on the probability simplex, as formalized in Proposition 2 in Appendix A-B.

2) The subroutine: The LP fixed-point subroutine is formally stated as the pseudocode in Algorithm 2: for a nonempty input, it computes y from (10) and then assigns arms by floorrounding the target counts $n y ( s , a )$ . Within each state, the resulting allocation is uniform over all assignments realizing those counts. The threshold η and the weighted norm $\left\| \cdot \right\| _ { U }$ that appear in the assertion are derived in Section III-B4.

3) Linear dynamics under LP fixed-point: Algorithm 2 induces a linear transition dynamics in expectation, up to an $O ( 1 / N )$ rounding error. The following lemma makes this precise.

Lemma 1 (LP fixed-point transition dynamics). Under Assumptions 2 and 3, suppose $\| X _ { t } ( [ N ] ) - \mu ^ { * } \| _ { U } \leq \eta$ at time t and all arms follow Algorithm 2. Let Y be the realized stateaction distribution, $y _ { t } = y ^ { * } + \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ as in (10), and $\mathcal { P }$ and Φ be as in (11). Then

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ( [ N ] ) - \mu ^ { * } \mid X _ { t } \right] } \\ & { \quad = \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \Phi ~ + ~ \left( Y _ { t } - y _ { t } \right) \mathcal { P } , } \end{array}\tag{12}
$$

where $\begin{array} { r } { \| ( Y _ { t } - y _ { t } ) \mathcal { P } \| _ { 1 } \leq \| Y _ { t } - y _ { t } \| _ { 1 } \leq 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N . } \end{array}$

The proof of Lemma 1 is given in Appendix B-C.

4) When is the subroutine feasible?: Next, we derive the assertion on Algorithm 2, the sufficient condition for the LP fixed-point subroutine to be applicable. Specifically, we find some weighted norm $\lVert \cdot \rVert _ { U }$ and a feasibility radius $\eta > 0$ , such that whenever $\| z / n - \mu ^ { * } \| _ { U } \leq \eta ,$ , we have

(i) Non-negativity: $y ( s , a ) \geq 0$ for all $( s , a ) ;$

(ii) Budget constraints: $\begin{array} { r } { \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } \big ( s , a \big ) y ( s , a ) \ \le \ \alpha _ { k } } \end{array}$ for all $k \in [ K ]$

which guarantees that $y$ is a valid probability distribution and that the resulting actions of LP fixed-point satisfy the budget constraints. Neither (i) nor (ii) is encoded in the linear system (10), and both hold only when x lies sufficiently close to $\mu ^ { * }$

Definition 1. Assume Assumptions 1 to 3. Let W and $U$ be the $\left| \mathbb { S } \right| { - } \mathbf { b } \mathbf { y } { - } \left| \mathbb { S } \right|$ matrices given by

$$
W = \sum _ { k = 0 } ^ { \infty } ( P _ { \bar { \pi } ^ { * } } - \mathbb { 1 } ^ { \top } \mu ^ { * } ) ^ { k } ( ( P _ { \bar { \pi } ^ { * } } - \mathbb { 1 } ^ { \top } \mu ^ { * } ) ^ { \top } ) ^ { k } ,\tag{13}
$$

$$
U = \sum _ { k = 0 } ^ { \infty } \Phi ^ { k } ( \Phi ^ { \top } ) ^ { k } ,\tag{14}
$$

where $\Phi$ is given by (11).

We write $\lambda _ { W } ~ \triangleq ~ \| W \| _ { 2 }$ and $\lambda _ { U } \ \triangleq \ \lVert U \rVert _ { 2 } ,$ , and let $\lVert \cdot \rVert _ { W }$ and $\lVert \cdot \rVert _ { U }$ denote the $W \mathrm { - }$ and U-weighted $L _ { 2 }$ norms on $\mathbb { R } ^ { | \mathbb { S } | } .$ defined by $\| u \| _ { W } = \sqrt { u W u ^ { \top } }$ and $\| u \| _ { U } = \sqrt { u U u ^ { \top } }$ for any row vector u.

Under Assumption $3 ~ ( \rho ( \Phi ) ~ < ~ 1 )$ , U is well-defined and satisfies the contraction property in Appendix B-A. The feasibility radius $\eta$ is then

$$
\begin{array} { r } { \eta \triangleq \frac { \epsilon } { \left. C _ { \mathbb { S } } ^ { + } \right. _ { U ^ { - 1 } } } , \qquad \mathrm { w h e r e } } \\ { \left. C _ { \mathbb { S } } ^ { + } \right. _ { U ^ { - 1 } } \triangleq \underset { ( s , a ) } { \operatorname* { m a x } } \left. ( C _ { \mathbb { S } } ^ { + } e _ { ( s , a ) } ) ^ { \top } \right. _ { U ^ { - 1 } } , } \end{array}\tag{15}
$$

$e _ { ( s , a ) } \in \mathbb { R } ^ { | \mathbb { S } | | \mathbb { A } | }$ is the one-hot column vector with 1 at entry $( s , a )$ , and the constant ϵ is given by

$$
\begin{array} { r l } & { \epsilon \triangleq \operatorname* { m i n } \biggr \{ \underset { ( s , a ) : \ : y ^ { * } ( s , a ) > 0 } { \operatorname* { m i n } } y ^ { * } ( s , a ) , } \\ & { \underset { k \not \in K ^ { * } } { \operatorname* { m i n } } \frac { \alpha _ { k } - \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y ^ { * } ( s , a ) } { \vert \mathbb { S } \vert \vert \mathbb { A } \vert } \biggr \} . } \end{array}\tag{16}
$$

The inner minimum over $k \notin \ K ^ { * }$ is interpreted as $+ \infty$ when all budgets are tight. The denominator in (15) is positive because (9) implies $C _ { \mathbb { S } } ^ { \bar { + } } \neq 0$

We now argue that whenever $\| x - \mu ^ { * } \| _ { U } \leq \eta$ , the target $y = y ^ { * } + \left( x - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ satisfies (i) and (ii) above. For every $( s , a ) \in \mathbb { S } \times \mathbb { A }$ , the Cauchy–Schwarz inequality gives

$$
| y ( s , a ) - y ^ { * } ( s , a ) | \leq \left\| C _ { \mathbb { S } } ^ { + } \right\| _ { U ^ { - 1 } } \left\| x - \mu ^ { * } \right\| _ { U } \leq \epsilon .\tag{17}
$$

Since $y ^ { * } ( s , a ) \geq \epsilon$ for all $( s , a ) \notin \mathcal { U } ^ { * } \cup ( S ^ { \varnothing } \times \mathbb { A } )$ (by definition of $\epsilon ) ,$ it follows that $y ( s , a ) \geq 0$ , verifying (i). The budget constraints (ii) is verified similarly. The formal statement is given in Lemma 9.

The feasibility condition $\| x - \mu ^ { * } \| _ { U } \leq \eta$ uses floor rounding for every nonzero action, with remaining arms assigned action 0.

## C. Two-set policy and feasibility

The two-set policy (Algorithm 3) is defined as follows: at each time step, it selects two subsets, $D _ { t } ^ { \mathrm { L P } }$ and $D _ { t } ^ { \bar { \pi } ^ { * } }$ , letting them follow the two subroutines, LP fixed-point and $\bar { \pi } ^ { * }$ -Fixed-Ratio Control, respectively. The rest of the arms can take arbitrary actions as long as the budget constraints are satisfied, and we simply assign them action ${ \mathrm { { \bar { 0 . } } } } ^ { 2 }$

To select the subset $D _ { t } ^ { \mathrm { L P } }$ , we define the slack function

$$
\delta ( X _ { t } , D ) \triangleq \eta m ( D ) \ - \ \| X _ { t } ( D ) - m ( D ) \mu ^ { * } \| _ { U } ,\tag{18}
$$

```latex
Input: number of arms N, budgets $( \alpha _ { k } N ) _ { k \in [ K ] } ,$
an optimal solution of LP-relaxation $y ^ { * }$
left-inverse submatrix $C _ { \mathbb { S } } ^ { + }$ , feasibility radius $\eta ,$
$\begin{array} { r } { \alpha _ { \operatorname* { m i n } } \triangleq \operatorname* { m i n } _ { k \in [ K ] . } \alpha _ { k } , } \end{array}$
error tolerance $\dot { \epsilon } _ { N } ^ { \mathrm { r d } } \geq 0$ with $\epsilon _ { N } ^ { \mathrm { r d } } = { \cal O } ( 1 / N ) .$
initial system state $X _ { 0 } .$ , initial state vector $S _ { 0 } ,$
initial subsets ${ \cal D } _ { - 1 } ^ { \mathrm { L P } } = { \cal D } _ { - 1 } ^ { \bar { \pi } ^ { * } } = \emptyset$
1: for $t = 0 , 1 , \ldots$ do
2: if $\delta ( X _ { t } , [ N ] ) \geq 0$ then
3: Let $\dot { D } _ { t } ^ { \mathrm { L P } } = [ N ]$
4: else if $\delta ( { \check { X } } _ { t } , D _ { t - 1 } ^ { \mathrm { { \check { L } } P } } ) \geq 0$ then
5: Let $\dot { D } _ { t } ^ { \mathrm { L P } }$ be any $\epsilon _ { N } ^ { \mathrm { r d } }$ -maximal
feasible subset such that $D _ { t } ^ { \mathrm { L P } } \supseteq D _ { t - 1 } ^ { \mathrm { L P } }$
6: else
7: Let $D _ { t } ^ { \mathrm { L P } }$ be any $\epsilon _ { N } ^ { \mathrm { r d } }$ -maximal
feasible subset
8: Let $D _ { t } ^ { \bar { \pi } ^ { * } } \subseteq [ N ] \setminus D _ { t } ^ { \mathrm { L P } }$ with
$| D _ { t } ^ { \bar { \pi } ^ { \ast } } | = \mathsf { \bar { L } } \alpha _ { \operatorname* { m i n } } ( N - | D _ { t } ^ { \mathrm { L P } } | ) \rfloor ,$
s.t. either $\overline { { { D _ { t } ^ { \bar { \pi } } } } } ^ { * } \supseteq D _ { t - 1 } ^ { \bar { \pi } ^ { * } } \backslash \bar { D } _ { t } ^ { \mathrm { L P } }$
or $D _ { t } ^ { \bar { \pi } ^ { * } } \subseteq \bar { D _ { t - 1 } ^ { \bar { \pi } ^ { * } } } \backslash \bar { D _ { t } ^ { \mathrm { L P } } }$
9: Set $A _ { t } ( i )$ for $i \in D _ { t } ^ { \mathrm { L P } }$ using Algorithm 2
10: Set $A _ { t } ( i )$ for $i \in D _ { t } ^ { \bar { \pi } ^ { * } }$ using Algorithm 1
11: Set $A _ { t } ( i ) = 0$ for all $i \notin \bar { D _ { t } ^ { \mathrm { L P } } } \cup \bar { D _ { t } ^ { \bar { \pi } ^ { * } } }$
12: Apply $( A _ { t } ( i ) ) _ { i \in [ N ] } ;$ observe $S _ { t + 1 }$
```

For ${ \cal D } \ \ne \ \emptyset ,$ this slack is non-negative if and only if $\| X _ { t } ( D ) / m ( D ) - \mu ^ { * } \| _ { U } \leq \eta ,$ i.e., the empirical state distribution of the arms in $D$ lies within the feasibility radius η of $\mu ^ { * }$ . For $D = \emptyset$ , the slack is zero and the subroutine performs no assignments. Consequently, whenever $\delta ( X _ { t } , D ) \geq 0$ , LP fixed-point is applicable to the arms in D.

Definition 2 $( \epsilon _ { N } ^ { \mathrm { r d } }$ -maximal feasible set). Given the current system state x and $\epsilon _ { N } ^ { \mathrm { r d } } \geq 0 .$ , a set of arms $D \subseteq [ N ]$ is $\epsilon _ { N } ^ { \mathrm { r d } } - $ maximal feasible if the two conditions hold: $( 1 ) \delta ( x , D ) \geq 0$ or $D = \varnothing ; ( 2 )$ for any $D ^ { \prime }$ such that $D \subseteq D ^ { \prime } \subseteq [ N ]$ and $\delta ( x , D ^ { \prime } ) \geq \epsilon _ { N } ^ { \mathrm { r d } }$ , we have $m ( D ^ { \prime } ) \leq m ( D ) + \epsilon _ { N } ^ { \mathrm { r d } }$

We choose $D _ { t } ^ { \mathrm { L P } }$ to be $\mathbf { \boldsymbol { \mathbf { 1 } } } \epsilon _ { N } ^ { \mathrm { r d } }$ -maximal feasible set in the sense of Definition 2 for some predetermined $\epsilon _ { N } ^ { \mathrm { r d } } \geq 0$ with $\epsilon _ { N } ^ { \mathrm { r d } } = $ $O ( 1 / N )$ , with some additional requirements specified in Lines 2–7 of Algorithm 3.

The subset $D _ { t } ^ { \bar { \pi } ^ { * } } \subseteq [ N ] \backslash D _ { t } ^ { \mathrm { L P } }$ is chosen with size $\lfloor \alpha _ { \mathrm { m i n } } ( N -$ $| D _ { t } ^ { \mathrm { L P } } | ) \rfloor$ and the nesting condition on Line 8, where $\alpha _ { \operatorname* { m i n } } \triangleq$ $\mathrm { m i n } _ { k \in [ K ] } \alpha _ { k }$ ensures that the arms in $D _ { t } ^ { \bar { \pi } ^ { * } }$ can follow $\bar { \pi } ^ { * } .$ Fixed-Ratio Control while satisfying the budget constraints. Both set selections admit at least one admissible choice, as shown in Lemma 12 in Appendix C-A.

Feasibility under the global budget constraint.: By construction, whenever $\delta ( \bar { X _ { t } } , \bar { D _ { t } ^ { \mathrm { L P } } } ) \ \geq \ 0 ,$ , Algorithm 2 applied to $D _ { t } ^ { \mathrm { L P } }$ is well-defined, and the type-k cost of arms in $D _ { t } ^ { \mathrm { L P } }$ satisfies $\begin{array} { r } { \frac { 1 } { N } \sum _ { i \in D _ { \epsilon } ^ { \mathrm { L P } } } c _ { k } \big ( S _ { t } ( i ) , A _ { t } ( i ) \big ) \ \le \ m ( D _ { t } ^ { \mathrm { L P } } ) \alpha _ { k } } \end{array}$ for each $k \in [ K ]$ . The π¯<sup>∗</sup>-Fixed-Ratio Control on $D _ { t } ^ { \bar { \pi } ^ { * } }$ incurs at most $| D _ { t } ^ { \bar { \pi } ^ { * } } | / \bar { N } \le \alpha _ { \mathrm { m i n } } ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) \le \alpha _ { k } ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) )$ type-k cost for every k (we recall that $c _ { k } ( s , a ) \leq 1$ for all $s \in \mathbb { S }$ and $a \in \mathbb { A } )$ . Since the remaining arms take the zero-cost action 0, the full system satisfies

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } c _ { k } ( S _ { t } ( i ) , A _ { t } ( i ) ) } & { } \\ { \displaystyle } & { \leq \alpha _ { k } m ( D _ { t } ^ { \mathrm { L P } } ) + \alpha _ { k } \big ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) \big ) = \alpha _ { k } , \quad \forall k \in [ K ] . } \end{array}
$$

The full subroutine conformity argument appears in Lemma 12 in Appendix C-A.

## IV. OPTIMALITY GAP

Theorem 1 (Optimality gap for the WCMDP two-set policy). Suppose Assumption 1, Assumption 2, and Assumption 3 hold. Let π denote the two-set policy given in Algorithm 3 with error tolerance $\epsilon _ { N } ^ { r d } = O ( 1 / N )$ . Then π satisfies

$$
R ^ { \mathrm { r e l } } - R ( \pi , S _ { 0 } ) = O \bigg ( \frac { 1 } { N } \bigg ) .\tag{19}
$$

The proof uses Lemmas 1 and 3, which characterize the mean transitions and instantaneous reward under LP fixedpoint. Both include an $O ( 1 / N )$ floor-rounding residual that contributes the $O ( 1 / N )$ term in our bound. The proof outline appears in Section V, with details in Appendix C.

## V. ANALYSIS OF THE TWO-SET POLICY

This section outlines the proof of Theorem 1; detailed arguments appear in Appendix C.

Throughout the analysis we assume Assumptions 1 to 3 and set $\overline { { \eta } } \triangleq \eta . \mathrm { I f } \ r _ { \mathrm { m a x } } = 0$ , the optimality gap is zero, so we assume $r _ { \operatorname* { m a x } } > 0$ below.

Under the two-set policy, $\Sigma _ { t } \triangleq ( X _ { t } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { \ast } } )$ is a timehomogeneous finite-state Markov chain. We let X denote its state space and write $\sigma = ( x , D ^ { \mathrm { L P } } , D ^ { \bar { \pi } ^ { * } } )$ for a generic element of X. For ease of presentation, we suppose that $\Sigma _ { t }$ is aperiodic. For any fixed initial state $( X _ { 0 } , D _ { 0 } ^ { \mathrm { L P } } , D _ { 0 } ^ { \bar { \pi } ^ { * } } )$ , we let $\dot { \Sigma _ { \infty } } = ( X _ { \infty } , D _ { \infty } ^ { \mathrm { L P } } , D _ { \infty } ^ { \bar { \pi } ^ { \ast } } )$ have its limiting distribution. For a general finite-state chain, we instead let $\Sigma _ { \infty }$ have the limit of the time-averaged state distributions for the chosen initial state. This distribution is stationary and satisfies the same long-run reward identity below, so the argument applies unchanged.

We define the one-step drift operator

$$
\Delta f ( \sigma ) \triangleq \mathbb { E } \left[ f ( \Sigma _ { t + 1 } ) | \Sigma _ { t } = \sigma \right] - f ( \sigma )
$$

for any function $f \colon \mathbb { X } \to \mathbb { R }$ . Stationarity gives $\mathbb { E } \left[ \Delta f ( \Sigma _ { \infty } ) \right] =$ 0.

We define the expected instantaneous reward under the twoset policy as

$$
r ^ { \pi } ( \sigma ) \triangleq \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } r ( s , a ) \mathbb { E } \big [ Y _ { t } ( s , a ) \big | \Sigma _ { t } = \sigma \big ] .
$$

The long-run average reward of the policy then satisfies $R ( \pi , S _ { 0 } ) = \mathbb { E } \left[ r ^ { \pi } ( \Sigma _ { \infty } ) \right]$

## A. Proof outline

1) Lyapunov framework: To show $R ^ { \mathrm { r e l } } - R ( \pi , S _ { 0 } ) \leq O _ { N }$ for some low-order term $O _ { N } \geq 0 .$ , it suffices to find $V , u \colon \mathbb { X } \to$ R satisfying the drift condition $\Delta V ( \sigma ) \le - u ( \sigma ) + O _ { N }$ and the dominance condition $u ( \sigma ) \geq R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma )$ for all $\sigma \in \mathbb { X }$

2) Proving $O ( 1 / \sqrt { N } )$ optimality gap: As a warm-up, we bound the reward gap by the distance from $\mu ^ { * }$ , allowing for rounding:

$$
\begin{array} { r l } & { R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) \leq K _ { 0 } \left. \mu ^ { * } - x ( [ N ] ) \right. _ { U } } \\ & { \qquad + \left. \frac { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } . \right. } \end{array}\tag{20}
$$

Here $K _ { 0 } > 0$ is a constant independent of N. We control this distance using two subset Lyapunov functions:

$$
h _ { W } ( x , D ) = \| x ( D ) - m ( D ) \mu ^ { * } \| _ { W }\tag{21}
$$

$$
h _ { U } ( x , D ) = \| x ( D ) - m ( D ) \mu ^ { * } \| _ { U } ,\tag{22}
$$

where matrices $W , U$ are defined via $P _ { \bar { \pi } ^ { * } , \Phi }$ as in Definition 1. We then set

$$
V _ { 1 } ( \sigma ) \triangleq h _ { U } ( x , D ^ { \mathrm { L P } } ) + h _ { W } ( x , D ^ { \bar { \pi } ^ { * } } ) + L _ { 1 } ( 1 - m ( D ^ { \mathrm { L P } } ) ) ,\tag{23}
$$

where $L _ { 1 } = 2 \lambda _ { U } ^ { 1 / 2 } + 4 \lambda _ { W } ^ { 1 / 2 } - 2 \lambda _ { W } ^ { 1 / 2 } \beta$ with $\beta \ { \stackrel { \triangle } { = } } \ \alpha _ { \mathrm { m i n } }$ and $\lambda _ { W } , \lambda _ { U }$ being the spectral radii of W and U.

Lemma 2 (Properties of $V _ { 1 } )$ . Assume Assumptions 1 to 3 hold, and let $\overline { { { \eta } } } = \eta .$ . There exist $\rho _ { 1 } \in ( 0 , 1 )$ and $K _ { 1 } , K _ { 2 } , C >$ 0 independent of N such that, for all $t \geq 0$ and all $\sigma =$ $( x , D ^ { \mathtt { L P } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X }$ , we have

$$
\mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) \right) ^ { + } \Big | \Sigma _ { t } = \sigma \right] \leq \frac { K _ { 1 } } { \sqrt { N } } ,\tag{24}
$$

$$
\begin{array} { r } { \mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) - \frac { ( 1 - \rho _ { 1 } ) \overline { { \eta } } } { 2 } \right) ^ { + } \bigg | \Sigma _ { t } = \sigma \right] } \end{array}
$$

$$
\leq K _ { 2 } \exp ( - C N ) ,\tag{25}
$$

and

$$
V _ { 1 } ( \sigma ) \geq \| x ( [ N ] ) - \mu ^ { * } \| _ { U } .\tag{26}
$$

The proof of Lemma 2, including the transition residual from Lemma 1, is given in Appendix C-B.

Combining (24) with (26) gives $\Delta V _ { 1 } ( \sigma ) \leq - ( 1 -$ $\rho _ { 1 } ) \| x ( [ N ] ) - \mu ^ { * } \| _ { U } + K _ { 1 } / \sqrt { N }$ . Taking stationary expectations and applying (20) yields

$$
\begin{array} { c l l } { R ^ { \mathrm { r e l } } - R ( \pi , S _ { 0 } ) \leq \displaystyle \frac { K _ { 0 } K _ { 1 } } { ( 1 - \rho _ { 1 } ) \sqrt { N } } + \frac { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } } \\ { \displaystyle = O ( 1 / \sqrt { N } ) . } \end{array}
$$

3) Proving $O ( 1 / N )$ optimality gap: To sharpen the optimality gap, we use a second Lyapunov function:

$$
\begin{array} { r } { V _ { 2 } ( \sigma ) \triangleq \Big ( V _ { 1 } ( \sigma ) - \frac { \overline { { \eta } } } { 2 } \Big ) ^ { + } + L _ { 2 } \Psi ( \sigma ) , } \end{array}\tag{27}
$$

with $\Psi ( \sigma ) \triangleq ( \mu ^ { * } - x ( [ N ] ) ) Q g , Q \triangleq ( I - \Phi ) ^ { - 1 }$ (welldefined by Assumption 3), λ<sub>Q</sub> as in (28), and $L _ { 2 } \ { \stackrel { \Delta } { = } } \ ( 1 \ -$ $\rho _ { 1 } ) \overline { { \eta } } / ( 4 \lambda _ { Q } K _ { g } + 4 r _ { \operatorname* { m a x } } + 4 K _ { g } )$ , where the vector g and the constant $K _ { g }$ are defined in Lemma 3 below. Here

$$
\lambda _ { Q } \triangleq \operatorname* { m a x } _ { v \in \Delta ( \mathbb { S } ) } \operatorname* { m a x } \{ \| ( v - \mu ^ { * } ) Q \| _ { 1 } , \| ( v - \mu ^ { * } ) \Phi Q \| _ { 1 } \} .\tag{28}
$$

Near $\mu ^ { * }$ , the local-approximation term Ψ produces a drift equal to the negative instantaneous reward gap up to an $O ( 1 / N )$ rounding residual. The truncated $V _ { 1 }$ term offsets the additional error outside this local region. The following three lemmas establish the resulting drift bound for $V _ { 2 }$

The first lemma characterizes the instantaneous expected reward near $\mu ^ { * }$ . Its proof requires new arguments specific to LP fixed-point; see Appendix C-C.

Lemma 3 (Instantaneous reward under LP fixed-point). Under the WCMDP two-set policy, for any $\sigma = ( x , D ^ { \bar { \mathrm { L P } } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X }$ with $\| x ( [ N ] ) - \mu ^ { * } \| _ { U } \leq \overline { { \eta } } \triangleq \eta ,$ , we have $D ^ { \mathrm { L P } } = [ N ]$ , and

$$
r ^ { \pi } ( \sigma ) = \hat { r } ( x ( [ N ] ) ) \ + \ \epsilon _ { N } ^ { r e w } ( \sigma ) ,
$$

where ${ \hat { r } } ( v ) \ { \stackrel { \triangle } { = } } \ R ^ { \mathrm { r e l } } + ( v - \mu ^ { * } ) g$ with $g \triangleq C _ { \mathbb { S } } ^ { + } r ^ { \top } -$ $( \mu ^ { * } C _ { \mathbb { S } } ^ { + } r ^ { \top } ) \mathbb { 1 } ^ { \top } \in \mathbb { R } ^ { | \mathbb { S } | }$ , r is the row vector $( r ( s , a ) )$ <sub>s∈S,a∈A</sub>, and $\epsilon _ { N } ^ { \bar { r } e w } \colon \mathbb { X } $ R satisfies $\lvert \epsilon _ { N } ^ { r e w } ( \sigma ) \rvert \le 2 r _ { \mathrm { m a x } } \lvert \mathbb { S } \rvert ( \lvert \mathbb { A } \rvert - 1 ) / N$ Moreover, $\| g \| _ { \infty } \le K _ { g } \triangleq 2 r _ { \operatorname* { m a x } } \bigl ( 1 + \sqrt { 2 } \lambda _ { U } ^ { 1 / 2 } / \eta \bigr )$

The next two lemmas control the truncated and localapproximation terms; their proofs appear in Appendices C-D and C-E.

Lemma 4 (Drift of truncated $V _ { 1 } )$ . For any $\sigma , \sigma ^ { \prime } \in \mathbb { X } ,$

$$
\begin{array} { r l r } {  { \big ( V _ { 1 } ( \sigma ^ { \prime } ) - \frac { \overline { \eta } } { 2 } \big ) ^ { + } - \big ( V _ { 1 } ( \sigma ) - \frac { \overline { \eta } } { 2 } \big ) ^ { + } } } \\ & { } & { \leq - \frac { ( 1 - \rho _ { 1 } ) \overline { \eta } } { 2 } \Im \{ V _ { 1 } ( \sigma ) > \overline { \eta } \} } \\ & { } & { + ( V _ { 1 } ( \sigma ^ { \prime } ) - \rho _ { 1 } V _ { 1 } ( \sigma ) - \frac { ( 1 - \rho _ { 1 } ) \overline { \eta } } { 2 } ) ^ { + } . } \end{array}
$$

Lemma 5 (Drift of Ψ). For any $\sigma = ( x , D ^ { \mathrm { L P } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X } ,$

$$
\begin{array} { r l } {  { \Delta \Psi ( \sigma ) } } \\ & { \le - ( R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) ) + \frac { K _ { \Psi } } { N } } \\ & { \quad + ( 2 \lambda _ { Q } K _ { g } + 2 r _ { \operatorname* { m a x } } + 2 K _ { g } ) \mathbb { 1 } \{ \| x ( [ N ] ) - \mu ^ { * } \| _ { U } > \overline { { \eta } } \} , } \end{array}\tag{29}
$$

where $K _ { \Psi } \ge 0$ is a constant independent of N and σ.

4) Proof of Theorem 1: Combining Lemma 4 with (25) of Lemma 2 and adding $L _ { 2 }$ times (29), we get

$$
\Delta V _ { 2 } ( \sigma ) \leq - L _ { 2 } ( R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) ) + \frac { L _ { 2 } K _ { \Psi } } { N } + K _ { 2 } \exp ( - C N ) .
$$

Taking the expectation over $\Sigma _ { \infty }$ and using $\mathbb { E } \left[ \Delta V _ { 2 } ( \Sigma _ { \infty } ) \right] = 0$ gives $R ^ { \mathrm { r e l } } - \bar { R } ( \pi , S _ { 0 } ) \leq K _ { \Psi } / N + ( { K _ { 2 } } \bar { / } L _ { 2 } ) \exp ( - C \bar { N } ) =$ $O ( 1 / N )$ . The dominant term $K _ { \Psi } / N$ comes from Lemma 5.

## REFERENCES

[1] J. T. Hawkins, “A lagrangian decomposition approach to weakly coupled dynamic optimization problems and its applications,” Ph.D. dissertation, Massachusetts Institute of Technology, 2003.

[2] C. Boutilier and T. Lu, “Budget allocation using weakly coupled, constrained Markov decision processes,” in Conf. Uncertainty in Artificial Intelligence (UAI), 2016, pp. 52–61.

[3] A. Biswas, G. Aggarwal, P. Varakantham, and M. Tambe, “Learning index policies for restless bandits with application to maternal healthcare,” in Proc. Int. Conf. Autonomous Agents and Multiagent Syst., 2021, pp. 1467–1468.

[4] S. S. Villar, “Indexability and optimal index policies for a class of reinitialising restless bandits,” Probab. Eng. Inf. Sci., vol. 30, no. 1, pp. 1–23, 2016.

[5] K. D. Glazebrook, H. M. Mitchell, and P. S. Ansell, “Index policies for the maintenance of a collection of machines by a set of repairmen,” Eur. J. Oper. Res., vol. 165, no. 1, pp. 267–284, 2005.

[6] P. Whittle, “Restless bandits: activity allocation in a changing world,” J. Appl. Probab., vol. 25, pp. 287 – 298, 1988.

[7] J. Niño-Mora, “Markovian restless bandits and index policies: A review,” Mathematics, vol. 11, no. 7, p. 1639, 2023.

[8] R. R. Weber and G. Weiss, “On an index policy for restless bandits,” J. Appl. Probab., vol. 27, no. 3, pp. 637–648, 1990.

[9] I. M. Verloop, “Asymptotically optimal priority policies for indexable and nonindexable restless bandits,” Ann. Appl. Probab., vol. 26, no. 4, pp. 1947–1995, 2016.

[10] C. Yan, “An optimal-control approach to infinite-horizon restless bandits: Achieving asymptotic optimality with minimal assumptions,” in Proc. IEEE Conf. Decision and Control (CDC), 2024, pp. 6665–6672.

[11] Y. Hong, Q. Xie, Y. Chen, and W. Wang, “Restless bandits with average reward: Breaking the uniform global attractor assumption,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 12 810– 12 844.

[12] ——, “Unichain and aperiodicity are sufficient for asymptotic optimality of average-reward restless bandits,” Math. Oper. Res., 2025, ahead of Print, http://dx.doi.org/10.1287/moor.2024.0678.

[13] N. Gast, B. Gaujal, and C. Yan, “Exponential asymptotic optimality of Whittle index policy,” Queueing Syst., vol. 104, no. 1, pp. 107–150, 2023.

[14] ——, “Linear program-based policies for restless bandits: Necessary and sufficient conditions for (exponentially fast) asymptotic optimality,” Math. Oper. Res., vol. 49, no. 4, pp. 2468–2491, 2024.

[15] Y. Hong, Q. Xie, Y. Chen, and W. Wang, “Achieving exponential asymptotic optimality in average-reward restless bandits without global attractor assumption,” arXiv:2405.17882 [cs.LG], 2024.

[16] D. J. Hodge and K. D. Glazebrook, “On the asymptotic optimality of greedy index heuristics for multi-action restless bandits,” Adv. Appl. Probab., vol. 47, no. 3, p. 652–667, 2015.

[17] G. Xiong, S. Wang, and J. Li, “Learning infinite-horizon average-reward restless multi-action bandits via index awareness,” in Conf. Neural Information Processing Systems (NeurIPS), 2022, pp. 17 911–17 925.

[18] D. Goldsztajn and K. Avrachenkov, “Asymptotically optimal policies for weakly coupled Markov decision processes,” arXiv:2406.04751 [math.OC], 2024.

[19] X. Zhang, Y. Hong, and W. Wang, “Projection-based lyapunov method for fully heterogeneous weakly-coupled mdps,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 91 054–91 076.

[20] N. Gast and D. Narasimha, “Model predictive control is almost optimal for restless bandits,” in Proc. Conf. Learning Theory (COLT), ser. Proceedings of Machine Learning Research, vol. 291, 2025, pp. 2326–2361. [Online]. Available: https://proceedings.mlr.press/v291/ gast25a.html

[21] D. Narasimha and N. Gast, “Model predictive control is almost optimal for heterogeneous restless multi-armed bandits,” arXiv:2511.08097v2 [math.OC], 2026. [Online]. Available: https://arxiv.org/abs/2511.08097v2

[22] X. Zhang and P. I. Frazier, “Restless bandits with many arms: Beating the central limit theorem,” arXiv:2107.11911 [math.OC], Jul. 2021.

[23] N. Gast, B. Gaujal, and C. Yan, “Reoptimization nearly solves weakly coupled Markov decision processes,” arXiv:2211.01961 [math.OC], 2024.

[24] D. B. Brown and J. Zhang, “Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation,” Oper. Res., vol. 73, no. 2, pp. 1029–1045, 2025.

[25] J. Zhang, “Leveraging nondegeneracy in dynamic resource allocation,” Available at SSRN 4920428, 2024.

[26] C. Yan, W. Wang, and L. Ying, “Achieving $\widetilde { \mathcal { O } } ( 1 / N )$ optimality gap in restless bandits through gaussian approximation,” in Advances in Neural Information Processing Systems, vol. 38. Curran Associates, Inc., 2025, pp. 51 190–51 212.

[27] M. L. Puterman, Markov decision processes: Discrete stochastic dynamic programming. John Wiley & Sons, 2005.

[28] A. Brauer, “Limits for the characteristic roots of a matrix. IV: Applications to stochastic matrices,” Duke Math. J., vol. 19, pp. 75–91, 1952.

## APPENDIX A AUXILIARY FACTS ABOUT THE ASSUMPTIONS

This appendix shows some auxiliary facts about Assumptions 2 and 3. In Appendix A-A, we compute the dimension of $C ^ { * }$ under the additional condition $| S ^ { \varnothing } | = 0$ , establishing a sufficient condition for $d \leq | \mathbb { S } | | \mathbb { A } |$ . This dimensional condition is necessary for Assumption 2. Then, in Appendix A-B, we investigate properties of the matrix Φ (defined in (11)) used in the local stability assumption (Assumption 3). These properties explain the centering in the definition of Φ and the role of the local stability assumption.

We use the notation $\rho ( M )$ to refer to the spectral radius of a square matrix M. We continue using $\mathbb { 1 } \in \mathbb { R } ^ { | \mathbb { S } | }$ to denote the all-one row vector indexed by states, consistent with other parts of the paper. We write $\bar { \mathbf { 1 } } _ { | \mathbb { S } | | \mathbb { A } | } \ \in \ \mathbb { R } ^ { | \mathbb { S } | | \mathbb { A } | }$ with explicit subscript to denote the all-one row vector indexed by stateaction pairs.

## A. Dimension of $C ^ { * }$ and validity of Assumption 2

Assumption 2 requires $C ^ { * } \in \mathbb { R } ^ { | \mathbb { S } | | \mathbb { A } | \times d }$ to have full column rank, which requires d to be no larger than $| \mathbb { S } | | \mathbb { A } |$ . Under ${ \mathrm { A s } } -$ sumption 1, the proposition below establishes this dimensional condition when $| \hat { S ^ { \varnothing } } | = 0$ (no transient states) and $y ^ { * }$ is a non-degenerate basic feasible solution (BFS) of $( \mathrm { L P - W } )$ . We account for redundant equality constraints by calling a BFS $y ^ { * }$ non-degenerate if the rank of the equality-constraint coefficient matrix plus the number of tight inequality constraints at $y ^ { * }$ equals the number of variables, $| \mathbb { S } | \left| \mathbb { A } \right|$

Proposition 1 (Dimension of $C ^ { * } )$ . Suppose Assumption 1 holds, $| S ^ { \varnothing } | = 0 ,$ , and $y ^ { * }$ is a non-degenerate basic feasible solution of (LP-W). Then $d = | \mathbb { S } | | \mathbb { A } |$ , so $C ^ { * }$ is square.

Proof. We write the flow-balance equations (3) as $y B = 0$ where $B \in \mathbb { R } ^ { | \mathbb { S } | | \mathbb { A } | \times | \mathbb { S } | }$ has entries

$$
B _ { ( s , a ) , s ^ { \prime } } = 1 \{ s = s ^ { \prime } \} - P ( s , a , s ^ { \prime } ) .
$$

Since $B \mathbf { 1 } ^ { \top } = 0$ , its rank is at most $| \mathbb { S } | - 1$ . Conversely, if a column vector v satisfies $B v = 0$ , then

$$
v ( s ) = \sum _ { s ^ { \prime } \in \mathbb { S } } P ( s , a , s ^ { \prime } ) v ( s ^ { \prime } ) \quad \forall ( s , a ) \in \mathbb { S } \times \mathbb { A } .
$$

Averaging over $\bar { \pi } ^ { * } ( a \mid s )$ gives $P _ { \bar { \pi } ^ { * } } v ~ = ~ v$ . By Assumption 1, the eigenspace at 1 is spanned by $\mathbb { 1 } ^ { \top }$ . Thus ker $B =$ $\operatorname { s p a n } \{ \mathbb { 1 } ^ { \top } \}$ and rank $B = | \mathbb { S } | - 1$

The equation $\begin{array} { r } { \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } y ( s , a ) = 1 } \end{array}$ in (4) adds one independent equality: its coefficient vector $\mathbb { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top }$ cannot lie in the column space of B, because $y ^ { * } B = 0$ whereas $y ^ { * } \mathbb { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top } = 1$ Thus the LP has |S| independent equality constraints.

By the definition of non-degenerate BFS, the number of tight inequality constraints at $y ^ { * }$ equals $| \mathbb { S } | | \mathbb { A } |$ minus the rank of the equality constraints, |S|, so there are $| \dot { \mathbb { S } } | | \mathbb { A } | - | \mathbb { S } |$ tight inequality constraints. With $| S ^ { \varnothing } | = 0$ , the binding inequalities at $y ^ { * }$ are the $| K ^ { * } |$ tight budget constraints and the $\lvert \mathcal { U } ^ { * } \rvert$ nonnegativity constraints on $\mathcal { U } ^ { * }$ , giving $| \mathcal { K } ^ { * } | + | \mathcal { U } ^ { * } | = | \mathbb { S } | | \mathbb { A } \bar { | } - | \mathbb { S } |$

Consequently, we have $| \mathcal { U } ^ { * } | = | \mathbb { S } | ( | \mathbb { A } | - 1 ) - | \mathcal { K } ^ { * } |$ . Substituting into the definition of d, we get

$$
\begin{array} { r l } & { d = | \mathcal { U } ^ { * } | + | \mathcal { K } ^ { * } | + | \mathbb { S } | } \\ & { \quad = \big ( | \mathbb { S } | ( | \mathbb { A } | - 1 ) - | \mathcal { K } ^ { * } | \big ) + | \mathcal { K } ^ { * } | + | \mathbb { S } | } \\ & { \quad = | \mathbb { S } | | \mathbb { A } | . } \end{array}
$$

Remark 2 (Transient states and degeneracy). Under Assumption 1, a non-degenerate BFS cannot have transient states. To prove this claim, we consider a BFS $y ^ { * }$ with $S ^ { \varnothing } \neq \varnothing$ and define the column vector u by $u ( s ) = 1$ for $s \in S ^ { \varnothing }$ and $u ( s ) = 0$ otherwise. Summing flow balance over $S ^ { \emptyset }$ gives $y B u = 0$ , with B as defined in the preceding proof. Since $S ^ { \emptyset }$ is a nonempty proper subset of $\mathbb { S } ,$ , u is nonconstant, so the identity ker $B { \stackrel { - } { = } } \mathrm { s p a n } \{ { \bf 1 } ^ { \top } \}$ from that proof implies $B u \ne 0 .$

The equation $y B u = 0$ involves only variables $y ( s , a )$ such that $y ^ { * } ( s , a ) = 0$ . Indeed, zero stationary mass in ${ \dot { S } } ^ { \varnothing }$ and the flow-balance equations (3) evaluated at $y ^ { \ast }$ give

$$
\begin{array} { l } { \displaystyle 0 = \sum _ { s ^ { \prime } \in S ^ { 0 } } \sum _ { a \in \mathbb { A } } y ^ { * } ( s ^ { \prime } , a ) } \\ { \displaystyle = \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } y ^ { * } ( s , a ) \sum _ { s ^ { \prime } \in S ^ { 0 } } P ( s , a , s ^ { \prime } ) . } \end{array}
$$

Nonnegativity of the terms in the last sum implies $\begin{array} { r } { \sum _ { s ^ { \prime } \in S ^ { \varnothing } } P ( s , a , s ^ { \prime } ) = 0 } \end{array}$ whenever $y ^ { * } ( s , a ) \ > \ 0$ . For such a pair $( s , a )$ , we also have $u ( s ) = 0 ;$ , so $( B u ) _ { ( s , a ) } = 0$

To conclude degeneracy, we replace the LP equalities by an independent spanning set of |S| equations, using the equality rank established in the preceding proof. The retained equalities have the same linear consequences as the original LP equalities, including $y B u = 0$ . The equation $y B u = 0$ is also a linear combination of the tight non-negativity constraints, written as $y ( s , a ) = 0$ at pairs $( s , a )$ where $y ^ { * } ( s , a ) \ = \ 0$ The two expressions for the nonzero vector $B u .$ one using the retained equalities and the other using the tight non-negativity constraints, give a nontrivial linear dependence among the constraint coefficient vectors. Since the retained LP equalities are independent and $y ^ { * }$ is a BFS, this linear dependence proves that $y ^ { * }$ is degenerate.

The dimensional condition $\begin{array} { r } { d \ \leq \ \vert \mathbb { S } \vert \vert \mathbb { A } \vert } \end{array}$ is a necessary dimension check for Assumption 2. Full column rank requires the columns of $C ^ { * }$ (i.e., the constraints in (6)) to be linearly independent in $\mathbb { R } ^ { | \mathbb { S } | | \hat { \mathbb { A } } | }$ ; the dimension calculation alone does not establish this independence.

## B. Properties of matrix Φ and validity of Assumption 3

In this subsection, we investigate properties of the matrix Φ given by

$$
\Phi \triangleq C _ { \mathbb { S } } ^ { + } \mathcal { P } - \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } .\tag{11}
$$

We show that the first term, $C _ { \mathbb { S } } ^ { + } \mathcal { P } ,$ , has an eigenvalue 1, while the second term $1 ^ { \top } \mu ^ { \ast } C _ { \mathbb { S } } ^ { + } \mathcal { P }$ replaces this eigenvalue with 0, without changing the right-multiplication on the centered probability simplex $\{ v \mathrm { ~ - ~ } \mu ^ { \ast } \colon v \mathrm { ~ \in ~ } \Delta ( \mathbb { S } ) \}$ . Therefore, the local stability assumption $\rho ( \Phi ) < 1$ is fully determined by the behavior of the matrix $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ restricted to the centered probability simplex. The lemma and proposition that formalize these facts are as follows.

Lemma 6. The matrix $C _ { \mathbb { S } } ^ { + }$ has row sums equal to 1, and the square matrix $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ has eigenvalue 1 with right eigenvector $\bar { 1 ^ { \top } } .$

$$
C _ { \mathbb { S } } ^ { + } { \mathbb { 1 } } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top } = { \mathbb { 1 } } ^ { \top }\tag{30}
$$

$$
C _ { \mathbb { S } } ^ { + } \mathcal { P } \mathbb { 1 } ^ { \top } = \mathbb { 1 } ^ { \top } .\tag{31}
$$

Proof. We first show (30). We recall that

$$
C _ { 8 } ^ { + } C ^ { * } = \big ( 0 , 0 , 0 , I _ { | 8 | } \big ) .\tag{9}
$$

We let $M \in \mathbb { R } ^ { ( | \mathbb { S } | | \mathbb { A } | ) \times | \mathbb { S } | }$ be the last |S| columns of $C ^ { * }$ . Then

$$
C _ { \mathbb { S } } ^ { + } M = I _ { | \mathbb { S } | } .
$$

Since $C ^ { * }$ is the coefficient matrix of the linear system (6), one can read off from the linear system that $M _ { ( s , a ) , s ^ { \prime } } = \mathbb { 1 } \{ s = s ^ { \prime } \}$ for $s , s ^ { \prime } \in \mathbb { S }$ and $a \in \mathbb { A }$ . Since $M 1 ^ { \top } \overset { \cdot } { = } \mathbf { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top } ,$ we have $C _ { \mathfrak { S } } ^ { + } \mathbb { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top } = C _ { \mathfrak { S } } ^ { + } M \mathbb { 1 } ^ { \top } = \mathbb { 1 } ^ { \top }$

Next, we show (31). Since $\mathcal { P } _ { ( s , a ) , s ^ { \prime } } = P ( s , a , s ^ { \prime } )$ , we have $\begin{array} { r } { \sum _ { s ^ { \prime } } \mathcal { P } _ { ( s , a ) , s ^ { \prime } } = 1 } \end{array}$ for each row $( s , \dot { a } ) \in \mathbb { S } \times \mathbb { A }$ . Consequently, we have $\mathop { \mathcal { P } } \mathbb { 1 } ^ { \top } = \mathbb { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top }$ and thus $\begin{array} { r } { C _ { \mathbb { S } } ^ { + } \mathcal { P } \mathbb { 1 } ^ { \top } = C _ { \mathbb { S } } ^ { + } \mathbb { 1 } _ { | \mathbb { S } | | \mathbb { A } | } ^ { \top } = } \end{array}$ $\mathbb { 1 } ^ { \top }$ □

Proposition 2. Let Φ be defined as in (11). Then:

(i) Φ and $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ agree on the hyperplane $H \triangleq \{ v \in \mathbb { R } ^ { | \mathbb { S } | }$ v $\mid ^ { \top } = 0 \stackrel { \sim }  \} , i . e . , v \Phi = v C _ { \mathbb { S } } ^ { + } \mathcal { P }$ for all $v \in H$

(ii) The eigenvalues of Φ are exactly the eigenvalues of $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ with one occurrence of the eigenvalue 1 (associated with the right eigenvector $\mathbb { 1 } ^ { \top } )$ replaced by 0, leaving all other eigenvalues unchanged.

Proof. For (i): For any $\begin{array} { r l r } { v } & { { } \in } & { H , v ( \Phi \ - \ C _ { \ S } ^ { + } \mathcal { P } ) = } \end{array}$ $- v \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } = - ( v \mathbb { 1 } ^ { \top } ) \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } = 0 .$

For (ii), we apply Brauer's rank-one perturbation theorem [28]. Since $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ has an eigenvalue 1 with right eigenvector $\mathbb { 1 } ^ { \top }$ , Brauer’s theorem implies that the spectrum of $\Phi \_ { } = $ $C _ { \mathbb { S } } ^ { + } \mathcal { P } - \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P }$ is identical to the spectrum of $C _ { \mathbb { S } } ^ { + } \mathcal { P }$ , with one occurrence of 1 replaced by $1 - \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } \mathbb { 1 } ^ { \top } = \tilde { 0 }$ □

## APPENDIX BGENERAL LEMMAS ON SUBROUTINE DYNAMICS ANDWEIGHTED NORMS

This appendix collects general-purpose lemmas used throughout the proofs in the WCMDP setting. Appendix B-A establishes well-definedness of the weight matrices W and U and records the pseudo-contraction properties of $P _ { \bar { \pi } } { : }$ ∗ and Φ under the corresponding weighted $L _ { 2 }$ norms. Appendix B-B establishes feasibility of the target state-action distribution $y = y ^ { * } + \left( x - \mu ^ { * } \right) C _ { \mathrm { { s } } } ^ { + }$ used by Algorithm 2. Building on these, Appendix B-C states the transition dynamics under the subroutines $\bar { \pi } ^ { * }$ -Fixed-Ratio Control (Algorithm 1) and LP fixed-point (Algorithm 2), and establishes an $O ( 1 / N )$ bound on the rounding error incurred by LP fixed-point.

A. Weighted $L _ { 2 }$ norms for quantifying the convergence $o f$ distributions

The matrices W, U and their weighted norms are defined in Definition 1. We establish their well-definedness and contraction properties below.

Lemma 7. Assume Assumptions 1 to 3. The matrices W and U in Definition 1 are well-defined and positive definite, and their eigenvalues are lower bounded by 1.

Proof. The well-definedness and eigenvalue lower bound for W has been proved in [12]. The main fact used in the proof is that all eigenvalues of $P _ { \bar { \pi } ^ { * } } - \mathbb { 1 } ^ { \top } \mu ^ { * }$ have moduli strictly less than 1 under Assumption 1. Since we have assumed the same things for Φ in Assumption 3, the proof for U is analogous with $P _ { \bar { \pi } ^ { * } } - \mathbb { 1 } ^ { \top } \mu ^ { * }$ replaced by Φ. □

Lemma 8 (Pseudo-contraction under the weighted $L _ { 2 }$ norms). Assume Assumptions 1 to 3. For any distribution $v \in \Delta ( \mathbb { S } )$

$$
\begin{array} { r } { \| ( v - \mu ^ { * } ) P _ { \bar { \pi } ^ { * } } \| _ { W } \leq \rho _ { w } \| v - \mu ^ { * } \| _ { W } , } \end{array}\tag{32}
$$

$$
\| ( v - \mu ^ { * } ) \Phi \| _ { U } \leq \rho _ { u } \| v - \mu ^ { * } \| _ { U } ,\tag{33}
$$

where $\rho _ { w } = 1 - 1 / ( 2 \lambda _ { W } )$ and $\rho _ { u } = 1 - 1 / ( 2 \lambda _ { U } )$

Proof. The inequality (32) follows from the weighted-norm argument in [12]; the same argument applies to (33) with $\bar { P _ { \bar { \pi } ^ { * } } } - \mathbb { 1 } ^ { \top } \mu ^ { * }$ replaced by Φ. □

## B. Feasibility of LP fixed-point

The next lemma shows that the feasibility radius condition $\| x - \mu ^ { * } \| _ { U } \leq \eta$ implies the feasibility of LP fixed-point.

Lemma 9 (Inequality feasibility of the affine target). With ϵ as in (16) and η as in (15),for every $x \in \Delta ( \mathbb { S } )$ with $\| x - \mu ^ { * } \| _ { U } \leq$ η, the vector $y _ { t } = y ^ { * } + \left( x - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ defined in (10) satisfies

(i) Non-negativity: $y _ { t } ( s , a ) \geq 0 f o r a l l ( s , a ) \in \mathbb { S } \times \mathbb { A } ,$

(ii) Budget constraints: $\begin{array} { r } { \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } \bigl ( s , a \bigr ) y _ { t } \bigl ( s , a \bigr ) \le \alpha _ { k } } \end{array}$ for every $k \in [ K ]$

Therefore, LP fixed-point is feasible for all arms whenever $\| X _ { t } ( [ N ] ) - \mu ^ { * } \| _ { U } \leq \eta .$

Proof. We first invoke Cauchy–Schwarz to get the following bound that is used for proving both (i) and (ii): for every $( s , a ) \in \mathbb { S } \times \mathbb { A }$ , we have

$$
\begin{array} { r l } & { \left| y _ { t } ( s , a ) - y ^ { * } ( s , a ) \right| } \\ & { \quad = \left| \left( x - \mu ^ { * } \right) C _ { \mathfrak { S } } ^ { + } e _ { ( s , a ) } \right| } \\ & { \quad \le \left\| C _ { \mathfrak { S } } ^ { + } \right\| _ { U ^ { - 1 } } \left\| x - \mu ^ { * } \right\| _ { U } } \\ & { \quad \le \left\| C _ { \mathfrak { S } } ^ { + } \right\| _ { U ^ { - 1 } } \eta = \epsilon . } \end{array}\tag{34}
$$

Now we prove (i) and (ii) separately.

(i) Non-negativity. For $( s , a ) \notin \mathcal { U } ^ { * } \cup ( S ^ { \emptyset } \times \mathbb { A } ) , y ^ { * } ( s , a ) \geq \epsilon$ by (16), so (34) gives $y _ { t } ( s , a ) \geq 0$ . For $( s , a ) \in \mathcal { U } ^ { * } , y _ { t } ( s , a ) =$ 0 by the U<sup>∗</sup>-block of (9). For $s \in S ^ { \varnothing }$ , the $S ^ { \emptyset }$ -block of (6) forces $y _ { t } ( s , 0 ) = \cdots = y _ { t } ( s , | \mathbb { A } | - 1 ) = x ( s ) / | \mathbb { A } | \geq 0 .$

(ii) Budget constraints. For $k \in { \cal K } ^ { * }$ , the linear equations (6) force $\begin{array} { r } { \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y _ { t } ( s , a ) = \alpha _ { k } } \end{array}$ . For $k \notin \mathcal { K } ^ { \ast }$ , by the definition of ϵ in (16), we have

$$
\sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } \mathopen { } \mathclose \bgroup \left( s , a \aftergroup \egroup \right) y ^ { * } \mathopen { } \mathclose \bgroup \left( s , a \aftergroup \egroup \right) \leq \alpha _ { k } - \lvert \mathbb { S } \rvert \mathopen { } \mathclose \bgroup \left. \mathbb { A } \aftergroup \egroup \right. \epsilon \textnormal { }
$$

Combining this bound with (34) and using the fact that $c _ { k } ( s , a ) \leq 1$ for all $( s , a ) \in \mathbb { S } \times \mathbb { A }$ , we have

$$
\begin{array} { r l } {  { \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y _ { t } ( s , a ) } } \\ & { \leq \displaystyle \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } c _ { k } ( s , a ) y ^ { * } ( s , a ) + | \mathbb { S } | | \mathbb { A } | \epsilon } \\ & { \leq \alpha _ { k } . } \end{array}\tag{□}
$$

## C. Transition dynamics under subroutines

The lemma below characterizes the dynamics of the empirical distribution of any subset of arms following $\bar { \pi } ^ { * }$ -Fixed-Ratio Control (Algorithm 1).

Lemma 10 (π¯<sup>∗</sup>-Fixed-Ratio Control transition dynamics). For any time step t and any $D \subseteq [ N ]$

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ( D ) \vert X _ { t } , \mathrm { a r m s \ i n \ } D \mathrm { \ f o l l o w \ A l g o r i t h m \ 1 } \right] } \\ & { \quad = X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \quad a . s . , } \end{array}\tag{35}
$$

where the expectation is taken entry-wise and $P _ { \bar { \pi } ^ { * } } ( s , s ^ { \prime } ) =$ $\begin{array} { r } { \sum _ { a \in \mathbb { A } } \bar { \pi } ^ { * } ( a | \bar { s ) } P ( s , a , s ^ { \prime } ) f o r s , s ^ { \prime } \in \mathbb { S } . } \end{array}$

Proof. For each $s ^ { \prime } \in \mathbb { S } ,$ , independent action sampling and the transition kernel give

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ( D , s ^ { \prime } ) \vert X _ { t } , \mathrm { ~ a r m s ~ i n ~ } D \mathrm { ~ f o l l o w ~ A l g o r i t h m ~ } 1 \right] } \\ & { \quad = \displaystyle \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } X _ { t } ( D , s ) \bar { \pi } ^ { * } ( a \vert s ) P ( s , a , s ^ { \prime } ) } \\ & { \quad = ( X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } ) ( s ^ { \prime } ) . } \end{array}\tag{□}
$$

Next, we state two lemmas for LP fixed-point (Algorithm 2). For simplicity, we state and prove them for the case where all N arms follow LP fixed-point. For a nonempty subset $D ~ \subseteq ~ [ N ]$ satisfying $\| X _ { t } ( D ) / m ( D ) - \mu ^ { * } \| _ { U } ~ \leq ~ \eta _ { \tau }$ , these lemmas apply verbatim to the standalone system formed by the arms in D, with $X _ { t } ( [ N ] )$ replaced by $X _ { t } ( D ) / m ( D )$ and N replaced by |D|. For $D = \emptyset$ , the subroutine makes no assignments and its state and state-action counts remain zero.

We first prove the following lemma that bounds the rounding error of LP fixed-point.

Lemma 11 (Rounding error of LP fixed-point). Suppose all arms follow LP fixed-point (Algorithm 2) at time t, with target $y _ { t } = y ^ { * } + \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ as in (10), and let $Y _ { t }$ denote the realized state-action distribution. Then

$$
\| Y _ { t } - y _ { t } \| _ { 1 } \leq \frac { 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } .
$$

Proof. By floor rounding on Line 3 of Algorithm 2, $| N Y _ { t } ( s , a ) - N y _ { t } ( s , a ) | \ \leq \ 1$ for each (s, a) with $a \ne 0 ,$ so

$$
\sum _ { a \neq 0 } | Y _ { t } ( s , a ) - y _ { t } ( s , a ) | \leq { \frac { | \mathbb { A } | - 1 } { N } } .
$$

For each state $s ,$ the arms not assigned actions $a \ne 0$ take action 0, so $Y _ { t } ( s , 0 ) = X _ { t } ( [ N ] , s ) - \sum _ { a \neq 0 } Y _ { t } ( s , a ) ;$ ; since $\begin{array} { r } { y _ { t } ( s , 0 ) = X _ { t } ( [ N ] , s ) - \sum _ { a \neq 0 } y _ { t } ( s , a ) } \end{array}$ as well, we have

$$
\begin{array} { r l } {  { \vert Y _ { t } ( s , 0 ) - y _ { t } ( s , 0 ) \vert } \quad } & { } \\ & { \le \displaystyle \sum _ { a \neq 0 } \vert Y _ { t } ( s , a ) - y _ { t } ( s , a ) \vert } \\ & { \le \frac { \vert \mathbb { A } \vert - 1 } { N } . } \end{array}
$$

Summing the first and second display over $s \in \ \mathbb { S }$ gives $\| Y _ { t } - y _ { t } \| _ { 1 } \leq 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N$ □

Next, we restate and prove Lemma 1 on the transition dynamics of LP fixed-point.

Lemma 1 (LP fixed-point transition dynamics). Under $A s \mathrm { - }$ sumptions 2 and 3, suppose $\| X _ { t } ( [ N ] ) - \mu ^ { * } \| _ { U } \le \eta$ at time t and all arms follow Algorithm 2. Let Y be the realized stateaction distribution, $y _ { t } = y ^ { * } + \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ as in (10), and $\mathcal { P }$ and Φ be as in (11). Then

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ( [ N ] ) - \mu ^ { * } \mid X _ { t } \right] } \\ & { \quad = \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \Phi ~ + ~ \left( Y _ { t } - y _ { t } \right) \mathcal { P } , } \end{array}\tag{12}
$$

where $\begin{array} { r } { \| ( Y _ { t } - y _ { t } ) \mathcal { P } \| _ { 1 } \leq \| Y _ { t } - y _ { t } \| _ { 1 } \leq 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N . } \end{array}$

Proof. Since $Y _ { t }$ is deterministic given $X _ { t }$ (floor rounding is a deterministic function of $X _ { t } )$ , the expected next state counts at each $s ^ { \prime } \in \mathbb { S }$ are

$$
\mathbb { E } \left[ X _ { t + 1 } ( [ N ] , s ^ { \prime } ) | X _ { t } \right] = \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } Y _ { t } ( s , a ) P ( s , a , s ^ { \prime } ) .
$$

Aggregating in vector form and separating out $y _ { t }$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ( [ N ] ) \mid X _ { t } \right] } \\ & { \quad = Y _ { t } \mathcal { P } } \\ & { \quad = y _ { t } \mathcal { P } + \left( Y _ { t } - y _ { t } \right) \mathcal { P } . } \end{array}\tag{36}
$$

It remains to calculate $y _ { t } \mathcal { P }$ . Substituting the definition of $y _ { t }$ from (10), we obtain

$$
y _ { t } \mathcal { P } = y ^ { * } \mathcal { P } + ( X _ { t } ( [ N ] ) - \mu ^ { * } ) C _ { \& } ^ { + } \mathcal { P } .
$$

By the flow constraint of $\begin{array} { r } { ( \mathbf { L P - W } ) , \sum _ { s , a } y ^ { * } ( s , a ) P ( s , a , s ^ { \prime } ) = } \end{array}$ $\mu ^ { * } ( s ^ { \prime } )$ for all $s ^ { \prime } \in \mathbb { S } .$ , so $\boldsymbol { y } ^ { * } \mathcal { P } = \boldsymbol { \mu } ^ { * }$ . We recall that $\Phi =$ $C _ { \mathbb { S } } ^ { + } \mathcal { P } - \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P }$ from (11), so

$$
\begin{array} { r l } & { \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + } \mathcal { P } } \\ & { \quad = \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \Phi } \\ & { \quad \quad + \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \mathbb { 1 } ^ { \top } \mu ^ { * } C _ { \mathbb { S } } ^ { + } \mathcal { P } } \\ & { \quad = \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \Phi , } \end{array}
$$

where the last equality uses $\left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \mathbb { 1 } ^ { \top } = 0$ . Hence $y _ { t } \mathcal { P } = \mu ^ { * } + \left( X _ { t } ( [ N ] ) - \mu ^ { * } \right) \Phi $ , which combined with (36)

gives (12). The norm bound $\| ( Y _ { t } - y _ { t } ) \mathcal { P } \| _ { 1 } \leq \| Y _ { t } - y _ { t } \| _ { 1 } \leq$ $2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N$ follows from Lemma 11 and row-stochasticity of $\mathcal { P }$ . □

## APPENDIX C DETAILED UPPER-BOUND PROOFS

## A. Subroutine conformity

The following lemma establishes that the two-set policy is well-defined and feasible.

Lemma 12 (Subroutine conformity). Assume Assumptions 1 to 3 hold. Consider the two-set policy described in Algorithm 3. Its set-selection steps always admit an admissible choice. For any time t, we have the following:

(i) The arms in $D _ { t } ^ { \mathrm { L P } }$ can follow LP fixed-point.

(ii) The budget constraints are satisfied, i.e., for every k ∈ [K], P<sub>i∈[N]</sub> c<sub>k</sub>(S<sub>t</sub>(i), A<sub>t</sub>(i)) ≤ α<sub>k</sub>N.

Proof of Lemma 12. We first consider the selection of $D _ { t } ^ { \mathrm { L P } }$ If the full set is feasible, it is $\epsilon _ { N } ^ { \mathrm { r d } }$ -maximal. Otherwise, when $D _ { t - 1 } ^ { \mathrm { L P } }$ is feasible, the family of feasible supersets of $D _ { t - 1 } ^ { \mathrm { L P } }$ is nonempty; when it is infeasible, the family of all feasible sets contains ∅. In either case, a set of maximum cardinality within that family is $\epsilon _ { N } ^ { \mathrm { r d } }$ -maximal, since $\epsilon _ { N } ^ { \mathrm { r d } } \geq 0$ and every superset with slack at least $\epsilon _ { N } ^ { \mathrm { r d } }$ is also feasible. Thus an admissible $D _ { t } ^ { \mathrm { L P } }$ exists.

For the selection of $D _ { t } ^ { \bar { \pi } ^ { * } }$ , we let $B _ { t } = D _ { t - 1 } ^ { \bar { \pi } ^ { * } } \setminus D _ { t } ^ { \mathrm { L P } }$ and $q _ { t } \ = \ \lfloor \alpha _ { \mathrm { m i n } } ( N - | D _ { t } ^ { \mathrm { L P } } | ) \rfloor$ . Both $B _ { t } \subseteq [ N ] \setminus D _ { t } ^ { \mathrm { L P } }$ and $0 ~ \leq$ $q _ { t } \leq N - | D _ { t } ^ { \mathrm { L P } } |$ hold. If $| B _ { t } | \geq q _ { t }$ , we take a subset of $B _ { t }$ of size $q _ { t } ;$ otherwise, we extend $B _ { t }$ to that size within $[ N ] \backslash D _ { t } ^ { \mathrm { L P } }$ These choices satisfy the required nesting condition. Each setselection step therefore samples from a finite, nonempty family of admissible choices.

For (i), if $D _ { t } ^ { \mathrm { L P } } = \varnothing$ , LP fixed-point returns without assigning actions. If $D _ { t } ^ { \mathsf { L P } } \neq \varnothing$ , because the two-set policy chooses $D _ { t } ^ { \mathrm { L } \overline { { \mathrm { P } } } }$ to satisfy $\delta ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \geq 0$ , we have

$$
\left\| { X } _ { t } ( D _ { t } ^ { \mathrm { L P } } ) / m ( D _ { t } ^ { \mathrm { L P } } ) - \mu ^ { * } \right\| _ { U } \leq \eta .
$$

Since $X _ { t } ( D _ { t } ^ { \mathrm { L P } } ) / m ( D _ { t } ^ { \mathrm { L P } } )$ is the empirical distribution of arms in $D _ { t } ^ { \mathrm { L P } }$ , by Lemma 9, the arms in $\overline { { D _ { t } ^ { \mathrm { L P } } } }$ can follow LP fixedpoint.

For (ii), the arms in $D _ { t } ^ { \mathrm { L P } }$ satisfy the scaled budget constraints

$$
\sum _ { i \in D _ { t } ^ { \mathrm { L P } } } c _ { k } \big ( S _ { t } ( i ) , A _ { t } ( i ) \big ) \leq N m ( D _ { t } ^ { \mathrm { L P } } ) \alpha _ { k } .
$$

This follows from Lemma 9 when $D _ { t } ^ { \mathrm { L P } } \neq \varnothing ;$ for an empty set, both sides are zero. The cost in $D _ { t } ^ { \bar { \pi } ^ { * } }$ is at most $| D _ { t } ^ { \bar { \pi } ^ { * } } | \leq$ $N \alpha _ { \mathrm { m i n } } ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) \leq N \alpha _ { k } ( 1 - m ( \bar { D } _ { t } ^ { \mathrm { L P } } ) )$ since $c _ { k } ( s , a ) \in$ [0, 1] for all $s \in \mathbb { S } , a \in \mathbb { A }$ , and $k \in [ K ]$ . Since the remaining arms take action 0, by the assumption that $c _ { k } ( s , 0 ) = 0$ , they do not contribute to the total cost. Therefore, summing the cost of the three sets of arms, we get

$$
\begin{array} { r l } {  { \sum _ { i \in [ N ] } c _ { k } \big ( S _ { t } ( i ) , A _ { t } ( i ) \big ) } \quad } & { } \\ & { \leq N m ( D _ { t } ^ { \mathrm { L P } } ) \alpha _ { k } + N \alpha _ { k } ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) + 0 } \\ & { = N \alpha _ { k } . } \end{array}
$$

## B. Proof of Lemma 2

Lemma 2 (Properties of $V _ { 1 } ) .$ . Assume Assumptions 1 to 3 hold, and let $\overline { { { \eta } } } = \eta .$ . There exist $\rho _ { 1 } \in ( 0 , 1 )$ and $K _ { 1 } , K _ { 2 } , C >$ 0 independent $o f \ N$ such that, for all $t \geq 0$ and all $\sigma =$ $( x , D ^ { \bar { \mathrm { L P } } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X }$ , we have

$$
\mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) \right) ^ { + } \Big | \Sigma _ { t } = \sigma \right] \leq \frac { K _ { 1 } } { \sqrt { N } } ,\tag{24}
$$

$$
\begin{array} { r } { \mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) - \frac { ( 1 - \rho _ { 1 } ) \overline { { \eta } } } { 2 } \right) ^ { + } \bigg | \Sigma _ { t } = \sigma \right] } \end{array}
$$

$$
\leq K _ { 2 } \exp ( - C N ) ,\tag{25}
$$

and

$$
V _ { 1 } ( \sigma ) \geq \| x ( [ N ] ) - \mu ^ { * } \| _ { U } .\tag{26}
$$

The proof uses three supporting lemmas: subset concentration, almost non-shrinking, and sufficient coverage. We state them first, then give the composite drift argument and supporting proofs.

Lemma 13 (Properties of $h _ { W }$ and $h _ { U } )$ . Assume Assumptions 1 to 3 hold. For any t and $D \subseteq [ N ]$ , let $X _ { t + 1 } ^ { \prime }$ be the system state at time t + 1 if the arms in D follow Algorithm 1. Then

$$
\mathbb { E } \left[ ( h _ { W } ( X _ { t + 1 } ^ { \prime } , D ) - \rho _ { w } h _ { W } ( X _ { t } , D ) ) ^ { + } \middle | X _ { t } \right]
$$

$$
\leq { \frac { K _ { W , 1 } } { \sqrt { N } } } \quad a . s . ,
$$

$$
\mathbb { P } \big [ h _ { W } ( X _ { t + 1 } ^ { \prime } , D ) > \rho _ { w } h _ { W } ( X _ { t } , D ) + r \big | X _ { t } \big ]\tag{37}
$$

$$
\leq K _ { W , 2 } \exp ( - C _ { W } N r ^ { 2 } ) \quad a . s . , \forall r \geq 0 ,\tag{38}
$$

where $\rho _ { w } = 1 - 1 / ( 2 \lambda _ { W } )$ and $K _ { W , 1 } , K _ { W , 2 } , C _ { W }$ are positive constants independent of N. Moreover, for any $D , D ^ { \prime } \subseteq [ N ]$

$$
h _ { W } ( x , D ) \geq \frac { 1 } { | \mathbb { S } | ^ { 1 / 2 } } \left. x ( D ) - m ( D ) \mu ^ { * } \right. _ { 1 } ,\tag{39}
$$

$$
\begin{array} { r l } & { | h _ { W } ( x , D ) - h _ { W } ( x , D ^ { \prime } ) | } \\ & { \quad \le 2 \lambda _ { W } ^ { 1 / 2 } \big ( m ( D ^ { \prime } \setminus D ) + m ( D \setminus D ^ { \prime } ) \big ) . } \end{array}\tag{40}
$$

For $h _ { U }$ , take a nonempty subset $D \subseteq [ N ]$ and suppose that

$$
\| X _ { t } ( D ) / m ( D ) - \mu ^ { * } \| _ { U } \leq \eta .
$$

If the arms in D follow Algorithm 2, then h satisfies the analogous drift bound (37) and tail bound (38), with constants $K _ { U , 1 } , K _ { U , 2 } , C _ { U }$ and contraction rate $\rho _ { u } = 1 - 1 / ( 2 \lambda _ { U } )$ . Here $X _ { t + 1 } ^ { \prime }$ denotes the resulting next state. For $D = \emptyset$ , both subset functions vanish and these drift and tail bounds hold trivially. The strength and Lipschitz inequalities (39)–(40) hold for $h _ { U }$ for all $D , D ^ { \prime } \subseteq [ N ]$ , with $\lambda _ { W }$ replaced by $\lambda _ { U }$

Lemma 14 (Almost non-shrinking). Under Assumptions 1 to 3, we have that for any $t \geq 0 _ { i }$

$$
\begin{array} { r l } & { \mathbb { E } \big [ m ( D _ { t } ^ { \mathrm { L P } } \setminus D _ { t + 1 } ^ { \mathrm { L P } } ) \big | \Sigma _ { t } \big ] } \\ & { \quad \le K _ { D , 1 } m ( D _ { t } ^ { \mathrm { L P } } ) \exp ( - C _ { D } N m ( D _ { t } ^ { \mathrm { L P } } ) ^ { 2 } ) \quad a . s . , } \\ & { \quad \mathbb { P } \big [ m ( D _ { t } ^ { \mathrm { L P } } \setminus D _ { t + 1 } ^ { \mathrm { L P } } ) > 0 \big | \Sigma _ { t } \big ] } \\ & { \quad \le K _ { D , 2 } \exp ( - C _ { D } N m ( D _ { t } ^ { \mathrm { L P } } ) ^ { 2 } ) \quad a . s . , } \end{array}
$$

where $K _ { D , 1 } , K _ { D , 2 } , C _ { D } > 0$ are constants independent of N.

Lemma 15 (Sufficient coverage). Assuming Assumptions 1 to 3 and $\epsilon _ { N } ^ { r d } = O ( 1 / N )$ , we have that for any $t \geq 0$

$$
\begin{array} { r l } & { 1 - m ( D _ { t } ^ { \mathrm { L P } } ) \leq K _ { C , 1 } h _ { W } \big ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } \big ) + \displaystyle \frac { K _ { C , 2 } } { N } \quad a . s . , } \\ & { ~ m ( D _ { t } ^ { \mathrm { L P } } ) = 1 \quad i f h _ { U } \big ( X _ { t } , [ N ] \big ) \leq \eta , } \end{array}
$$

where $K _ { C , 1 } , K _ { C , 2 } > 0$ are constants independent of $N .$

To prove Lemma 2, we apply the drift and tail bounds $( 3 7 ) –$ (38) and their $h _ { U }$ analogues conditional on $\Sigma _ { t }$ , with $D = D _ { t } ^ { \bar { \pi } ^ { * } }$ for h and $D = D _ { t } ^ { \mathrm { L P } }$ for $h _ { U }$

Proof of Lemma 2. The domination and preliminary drift calculation in Appendix C-F gives the following bound, with $\rho _ { 1 } \in ( 0 , 1 )$ and $K _ { \mathrm { d r i f t } } > 0$ as defined there:

$$
\begin{array} { r l } & { V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) } \\ & { \quad \le \big ( h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - \rho _ { u } h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \big ) } \\ & { \quad \quad + \big ( h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \bar { \pi } ^ { * } } ) - \rho _ { w } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \big ) } \\ & { \quad \quad + 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \big ) m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) } \\ & { \quad \quad + \frac { K _ { \mathrm { d r i f t } } } { N } . } \end{array}\tag{41}
$$

(42)

It remains to prove (24) and (25) using (41)–(42).

Proof of (24). To prove (24), we take the conditional expectation of $\Big ( V _ { 1 } \big ( \Sigma _ { t + 1 } \big ) - \rho _ { 1 } V _ { 1 } \big ( \Sigma _ { t } \big ) \Big ) ^ { - }$ + and apply (41)–(42), which yields

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) \right) ^ { + } \Bigg | \Sigma _ { t } = \sigma \right] } \\ & { \quad \leq \mathbb { E } \Big [ \big ( h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - \rho _ { u } h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \big ) ^ { + } } \\ & { \quad \quad + \left( h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \pi } ) - \rho _ { w } h _ { W } ( X _ { t } , D _ { t } ^ { \pi } ) \right) ^ { + } } \\ & { \quad \quad + 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \big ) m ( D _ { t } ^ { \mathrm { L P } } ) D _ { t + 1 } ^ { \mathrm { L P } } \Big ) \Big | \Sigma _ { t } = \sigma \Big ] + \frac { K _ { \mathrm { d r i f t } } } { N } } \\ & { \quad \quad \leq \frac { K _ { U , 1 } + K _ { W , 1 } } { \sqrt { N } } + \frac { 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \big ) K _ { D , 1 } } { \sqrt { 2 } e C _ { D } N } + \frac { K _ { \mathrm { d r i f t } } } { N } , } \end{array}
$$

where the last step uses Lemmas 13 and 14 and

$$
\operatorname * { s u p } _ { m \ge 0 } m \exp ( - C _ { D } N m ^ { 2 } ) = \frac { 1 } { \sqrt { 2 e C _ { D } N } } .
$$

Since $1 / N \le 1 / \sqrt { N }$ , we obtain (24) by choosing

$$
\begin{array} { l } { { \displaystyle K _ { 1 } = K _ { U , 1 } + K _ { W , 1 } } } \\ { { \displaystyle ~ + \frac { 4 \bigl ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \bigr ) K _ { D , 1 } } { \sqrt { 2 e C _ { D } } } + K _ { \mathrm { d r i f t } } . } } \end{array}
$$

Proof of (25). Let $r _ { 0 } \triangleq ( 1 - \rho _ { 1 } ) \overline { { \eta } } / 2$ and $M \triangleq 2 \lambda _ { \scriptscriptstyle { I J } } ^ { 1 / 2 } +$ $2 \lambda _ { W } ^ { 1 / 2 } + L _ { 1 }$ . The bounds $\| x ( D ) - m ( D ) \mu ^ { * } \| _ { 1 } \leq 2 m ( D ) \leq 2$ imply $V _ { 1 } ( \sigma ) \leq M$ for every N and $\sigma \in \mathbb { X } .$ . Since $\left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \right.$ $\rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) - r _ { 0 } ) ^ { + } \leq M$ a.s., we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left( V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) - r _ { 0 } \right) ^ { + } \bigg | \Sigma _ { t } \right] } \\ & { \quad \leq M \cdot \mathbb { P } \Big [ V _ { 1 } ( \Sigma _ { t + 1 } ) > \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) + r _ { 0 } \Big | \Sigma _ { t } \Big ] \quad a . s . , } \end{array}\tag{43}
$$

so it suffices to show the probability on the right decays as $\exp ( - C N )$ for some $C > 0$ independent of N and $\sigma .$

By (41)–(42) and the union bound,

$$
\begin{array} { r l } & { \mathbb { P } \Big [ V _ { 1 } ( \Sigma _ { t + 1 } ) > \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) + r _ { 0 } \Bigm | \Sigma _ { t } \Bigm ] } \\ & { \quad \leq \mathbb { P } \big [ h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) > \rho _ { u } h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) + r _ { 0 } / 4 \bigm | \Sigma _ { t } \bigm ] } \\ & { \quad \quad + \mathbb { P } \big [ h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \bar { \pi } ^ { * } } ) > \rho _ { w } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) + r _ { 0 } / 4 \bigm | \Sigma _ { t } \bigm ] } \\ & { \quad \quad \quad + \mathbb { P } \big [ 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \bigm ) m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) > r _ { 0 } / 4 \bigm | \Sigma _ { t } \bigm ] } \\ & { \quad \quad \quad + \mathbb { 1 } \left\{ K _ { \mathrm { d r i f t } } / N > r _ { 0 } / 4 \right\} . } \end{array}
$$

The first two probabilities are bounded by Lemma 13 at the fixed threshold $r _ { 0 } / 4 .$ . For the third probability, the event is impossible if $m ( \tilde { D } _ { t } ^ { \mathrm { L P } } ) \leq r _ { 0 } / [ 1 6 ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } ) ]$ ]. Otherwise, the event implies that at least one old arm is lost, so Lemma 14 bounds its probability by

$$
K _ { D , 2 } \exp \left( - C _ { D } N \left[ \frac { r _ { 0 } } { 1 6 ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } ) } \right] ^ { 2 } \right) .
$$

The last indicator vanishes for $N > 4 K _ { \mathrm { d r i f t } } / r _ { 0 }$ . Combining these bounds gives constants $\tilde { K } , \tilde { C } > 0$ , independent of N and $\sigma ,$ , such that for $N > 4 K _ { \mathrm { d r i f t } } / r _ { 0 }$

$$
\begin{array} { r l } & { \mathbb { P } \Big [ V _ { 1 } ( \Sigma _ { t + 1 } ) > \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) + r _ { 0 } \Big | \Sigma _ { t } \Big ] } \\ & { \quad \leq \tilde { K } \exp \big ( - \tilde { C } N \big ) \quad a . s . } \end{array}\tag{45}
$$

For $1 ~ \le ~ N ~ \le ~ 4 K _ { \mathrm { d r i f t } } / r _ { 0 } .$ , the probability is at most 1. Increasing $\tilde { K }$ to be at least $\exp ( 4 \tilde { C } K _ { \mathrm { d r i f t } } / r _ { 0 } )$ therefore makes (45) valid for every $N \geq 1$ . Substituting this bound into (43) and setting $K _ { 2 } = M \tilde { K }$ and $C = \tilde { C }$ establishes (25). □

Proof of Lemma 13. The drift bound (37) and tail bound (38), together with their $h _ { U }$ analogues, are immediate for $D = \emptyset$ For the rest of the proof, we assume $D \neq \varnothing ;$ for $h _ { U }$ , we also assume the feasibility condition stated in the lemma.

Proving (37) and (38). The subroutine $\bar { \pi } ^ { * }$ -Fixed-Ratio Control (Algorithm 1) samples each arm's action independently, so there is no rounding error to control.

By the pseudo-contraction property of $P _ { \bar { \pi } }$ ∗ under the $W -$ weighted norm given in Lemma 8,

$$
\begin{array} { r l } & { \rho _ { w } \ : h _ { W } ( X _ { t } , D ) } \\ & { \quad = \rho _ { w } \ : \| X _ { t } ( D ) - m ( D ) \mu ^ { * } \| _ { W } } \\ & { \quad \geq \| ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) P _ { \bar { \pi } ^ { * } } \| _ { W } } \\ & { \quad = \| X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } - m ( D ) \mu ^ { * } \| _ { W } , } \end{array}
$$

so by the triangle inequality and the relation between the $W \mathrm { . }$ weighted norm and the $L _ { 1 }$ norm,

$$
\begin{array} { r l } & { \displaystyle h _ { W } ( X _ { t + 1 } ^ { \prime } , D ) - \rho _ { w } h _ { W } ( X _ { t } , D ) } \\ & { \quad \leq \left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { W } } \\ & { \quad \leq \lambda _ { W } ^ { 1 / 2 } \left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { 1 } . } \end{array}
$$

Therefore, it suffices to upper bound

$$
\left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { 1 }
$$

in expectation and in probability.

Under Algorithm 1, each arm $i \in D$ independently samples an action $A _ { t } ( i ) \sim \bar { \pi } ^ { * } ( \cdot | S _ { t } ( i ) )$ and then transitions according to $P .$ Hence, given $X _ { t } ,$ , the indicators $\mathbb { 1 } \left\{ S _ { t + 1 } ^ { \prime } ( i ) = s \right\}$ for $i \in D$ are independent Bernoulli random variables with means $P _ { \bar { \pi } ^ { * } } ( S _ { t } ( i ) , s )$ , where $S _ { t + 1 } ^ { \prime } ( i )$ denotes the state of arm i at time $t + 1$ if the arms in D follow Algorithm 1. In particular, by Lemma 10, E $: \left[ X _ { t + 1 } ^ { \prime } ( D ) \right. \left| \left. X _ { t } \right] = \bar { X } _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right.$

For each $s \in \mathbb { S } .$

$$
X _ { t + 1 } ^ { \prime } ( D , s ) = ( 1 / N ) \sum _ { i \in D } \mathbb { 1 } \big \{ S _ { t + 1 } ^ { \prime } ( i ) = s \big \}
$$

is a sum of independent random variables taking values in $[ 0 , 1 / N ]$ . By Cauchy–Schwarz,

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \Big | X _ { t + 1 } ^ { \prime } ( D , s ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D , s ) \bigm | X _ { t } \right] \Big | X _ { t } \Big ] } \\ & { \quad \le \operatorname { V a r } \Big [ X _ { t + 1 } ^ { \prime } ( D , s ) \Bigm | X _ { t } \Big ] ^ { 1 / 2 } } \\ & { \quad \le \frac { | D | ^ { 1 / 2 } } { N } \le \frac { 1 } { \sqrt { N } } , } \end{array}
$$

and by Hoeffding’s inequality, for all $r \geq 0$

$$
\begin{array} { r l } & { \mathbb { P } \Big [ \Big | X _ { t + 1 } ^ { \prime } ( D , s ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D , s ) \vert X _ { t } \right] \Big | > r \Big | X _ { t } \Big ] } \\ & { \quad \leq 2 \exp \big ( - 2 N r ^ { 2 } \big ) . } \end{array}
$$

Summing the expectation bound over $s \in \mathbb { S } .$ , and applying the union bound over $s \in \mathbb { S }$ (with r replaced by $r / | \mathbb { S } | )$ to the probability bound, we get

$$
\begin{array} { r l } & { \mathbb { E } \big [ \left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { 1 } \big | X _ { t } \big ] } \\ & { \mathrm { ~ \ } \leq \frac { | \mathbb { S } | } { \sqrt { N } } \mathrm { \quad } a . s . , } \\ & { \mathbb { P } \big [ \left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { 1 } > r \big | X _ { t } \big ] } \\ & { \mathrm { \ ~ \ } \leq 2 | \mathbb { S } | \exp \big ( - \frac { 2 N r ^ { 2 } } { | \mathbb { S } | ^ { 2 } } \big ) a . s . , \ \forall r \geq 0 . } \end{array}
$$

Combining the two displays with the bound $h _ { W } ( X _ { t + 1 } ^ { \prime } , D ) -$ $\begin{array} { r l r } { \rho _ { w } h _ { W } ( X _ { t } , D ) } & { \leq } & { \lambda _ { W } ^ { 1 / 2 } \left\| X _ { t + 1 } ^ { \prime } ( D ) - X _ { t } ( D ) P _ { \bar { \pi } ^ { * } } \right\| _ { 1 } } \end{array}$ established earlier, we obtain (37) and (38) with $K _ { W , 1 } = \lambda _ { W } ^ { 1 / 2 } | \mathbb { S } |$ $K _ { W , 2 } = 2 | \mathbb { S } |$ , and $C _ { W } = 2 / ( \lambda _ { W } | \mathbb { S } | ^ { 2 } )$

Proving the analogs of (37) and (38) for $h _ { U }$ . For $h _ { U }$ , we must account for the rounding residual in Lemma 1. We write $K _ { \mathrm { r n d } } \triangleq 2 | \mathbb { S } | ( | \mathbb { A } | - 1 )$ for the constant in the rounding bound of Lemma 11.

By the pseudo-contraction property of $\Phi$ under the $U _ { - }$ weighted norm given in Lemma 8,

$$
\begin{array} { r l } & { \rho _ { u } h _ { U } ( X _ { t } , D ) = \rho _ { u } \left\| X _ { t } ( D ) - m ( D ) \mu ^ { * } \right\| _ { U } } \\ & { \qquad \geq \left\| \left( X _ { t } ( D ) - m ( D ) \mu ^ { * } \right) \Phi \right\| _ { U } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } & { h _ { U } ( X _ { t + 1 } ^ { \prime } , D ) - \rho _ { u } h _ { U } ( X _ { t } , D ) } \\ & { \quad \le \left\| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } \right\| _ { U } } \\ & { \quad \quad - \left\| ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi \| _ { U } \right. } \\ & { \quad \le \left\| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } - ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi \| _ { U } \right. } \\ & { \quad \le \lambda _ { U } ^ { 1 / 2 } \Big \| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } } \\ & { \quad \quad \left. - \left( X _ { t } ( D ) - m ( D ) \mu ^ { * } \right) \Phi \right\| _ { 1 } . } \end{array}
$$

Therefore, it suffices to show that, with $C ^ { \prime } = 1 / ( 2 | \mathbb { S } | ^ { 2 } )$ ), we have

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \big \| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } - \big ( X _ { t } ( D ) - m ( D ) \mu ^ { * } \big ) \Phi \big \| _ { 1 } } \\ & { \qquad \mid X _ { t } \Big ] } \\ & { \quad = O ( 1 / \sqrt { N } ) \quad a . s . , } \\ & { \mathbb { P } \Big [ \Big \| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } } \\ & { \qquad - \big ( X _ { t } ( D ) - m ( D ) \mu ^ { * } \big ) \Phi \Big \| _ { 1 } > r \mid X _ { t } \Big ] } \\ & { \quad = O \big ( \exp ( - C ^ { \prime } N r ^ { 2 } ) \big ) \quad a . s . , \ \forall r \geq 0 . } \end{array}\tag{46}
$$

(47)

To control the rounding residual, we define the LP target for the arms in D by

$$
y _ { t } ^ { D } \triangleq y ^ { * } + \big ( X _ { t } ( D ) / m ( D ) - \mu ^ { * } \big ) C _ { \mathbb { S } } ^ { + } .
$$

Applying Lemma 1 to the arms in D and multiplying by $m ( D )$ gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } \mid X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] } \\ & { \quad = ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi } \\ & { \quad \quad + ( Y _ { t } ( D ) - m ( D ) y _ { t } ^ { D } ) \mathcal { P } . } \end{array}\tag{48}
$$

We decompose $X _ { t + 1 } ^ { \prime } ( D ) \mathrm { ~  ~ { ~ - ~ } ~ } m ( D ) \mu ^ { * } \mathrm { ~  ~ { ~ - ~ } ~ } ( X _ { t } ( D )  \mathrm { ~  ~ { ~ - ~ } ~ }$ $m ( D ) \mu ^ { * } )$ Φ by adding and subtracting the conditional mean $\mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right]$ :

$$
\begin{array} { r l } & { X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } - ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi } \\ & { \quad = \Big ( X _ { t + 1 } ^ { \prime } ( D ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] \Big ) } \\ & { \quad \quad + \Big ( \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] } \\ & { \quad \quad - m ( D ) \mu ^ { * } - ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi \Big ) } \\ & { \quad = \left( X _ { t + 1 } ^ { \prime } ( D ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] \right) } \\ & { \quad \quad + \left( Y _ { t } ( D ) - m ( D ) y _ { t } ^ { D } \right) \mathcal { P } . } \end{array}
$$

The rounding bound for $| D |$ arms, scaled by $m ( D ) = | D | / N$ is $K _ { \mathrm { r n d } } / N$ . Since $\mathcal { P }$ is row-stochastic,

$$
\begin{array} { r l } & { \left\| ( Y _ { t } ( D ) - m ( D ) y _ { t } ^ { D } ) \mathcal { P } \right\| _ { 1 } } \\ & { \quad \leq \left\| Y _ { t } ( D ) - m ( D ) y _ { t } ^ { D } \right\| _ { 1 } \leq \frac { K _ { \operatorname { r n d } } } { N } . } \end{array}
$$

The triangle inequality therefore gives

$$
\begin{array} { r l } & { \left\| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } - ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi \right\| _ { 1 } } \\ & { \quad \leq \left\| X _ { t + 1 } ^ { \prime } ( D ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) \vert X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] \right\| _ { 1 } } \\ & { \quad ~ + \frac { K _ { \mathrm { r n d } } } { N } . } \end{array}\tag{49}
$$

Given $X _ { t }$ and $( A _ { t } ( i ) ) _ { i \in { \cal D } }$ , the next-state indicators are independent across arms. For each state, the centered coordinate on the right-hand side of (49) is a sum of $| D |$ centered Bernoulli variables divided by N. The same variance estimate used for $h _ { W }$ bounds the expected $L _ { 1 }$ norm by $| \mathbb { S } | \sqrt { | D | } / N \le | \mathbb { S } | / \sqrt { N }$ Averaging over the actions and adding the rounding bound gives an upper bound of $( \left. \mathbb { S } \right. + K _ { \mathrm { r n d } } ) / \sqrt { N }$ in (46).

To prove the tail bound, first suppose $r \geq 2 K _ { \mathrm { r n d } } / N$ . By (49), a deviation larger than r requires the centered $L _ { 1 }$ norm to exceed $r - K _ { \mathrm { r n d } } / N \ge r / 2$ . Hoeffding’s inequality and a union bound over states give

$$
\begin{array} { r l } & { \mathbb { P } \Big [ \Big \| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } - ( X _ { t } ( D ) - m ( D ) \mu ^ { * } ) \Phi \Big \| _ { 1 } > r } \\ & { \qquad \Big | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \Big ] } \\ & { \quad \le \mathbb { P } \Big [ \Big \| X _ { t + 1 } ^ { \prime } ( D ) - \mathbb { E } \left[ X _ { t + 1 } ^ { \prime } ( D ) \vert X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \right] \Big \| _ { 1 } > r / 2 } \\ & { \qquad \Big | X _ { t } , ( A _ { t } ( i ) ) _ { i \in D } \Big ] } \\ & { \quad \le 2 | \mathbb { S } | \exp \Big ( - \frac { N ^ { 2 } r ^ { 2 } } { 2 | D | | \mathbb { S } | ^ { 2 } } \Big ) } \\ & { \quad \le 2 | \mathbb { S } | \exp ( - C ^ { \prime } N r ^ { 2 } ) . } \end{array}
$$

For $0 \leq r < 2 K _ { \mathrm { r n d } } / N$ , we instead use the probability bound 1. Since $N r ^ { 2 } \ \leq \ 4 K _ { \mathrm { r n d } } ^ { 2 }$ for $N \geq 1$ , this bound is at most $\mathrm { e x p } ( 4 C ^ { \prime } K _ { \mathrm { r n d } } ^ { 2 } ) \mathrm { e x p } ( - \overleftrightarrow { C ^ { \prime } } N r ^ { 2 } )$ . Thus, setting

$$
K _ { U , 2 } = \operatorname* { m a x } \{ 2 | \mathbb { S } | , \exp ( 4 C ^ { \prime } K _ { \mathrm { r n d } } ^ { 2 } ) \}
$$

and averaging over the actions gives, for every $r \geq 0$

$$
\begin{array} { r l } & { \mathbb { P } \bigg [ \bigg \| X _ { t + 1 } ^ { \prime } ( D ) - m ( D ) \mu ^ { * } } \\ & { \qquad - \left( X _ { t } ( D ) - m ( D ) \mu ^ { * } \right) \Phi \bigg \| _ { 1 } > r \Big | X _ { t } \Big ] } \\ & { \leq K _ { U , 2 } \exp ( - C ^ { \prime } N r ^ { 2 } ) \quad a . s . } \end{array}\tag{50}
$$

This proves (47). Using the factor $\lambda _ { U } ^ { 1 / 2 }$ in the earlier bound for $h _ { U } ( X _ { t + 1 } ^ { \prime } , D ) - \rho _ { u } h _ { U } ( X _ { t } , D )$ establishes the claimed drift and tail bounds with

$$
K _ { U , 1 } = \lambda _ { U } ^ { 1 / 2 } ( | \mathbb { S } | + K _ { \mathrm { r n d } } ) , \qquad C _ { U } = \frac { C ^ { \prime } } { \lambda _ { U } } .
$$

All constants are independent of All constants are independent of $N , D ,$ and the current state. and the current state.

The inequalities (39)–(40) and their $h _ { U }$ analogs follow from the definitions of the weighted norms; the corresponding norm estimates are also discussed in [12]. □

Proof of Lemma 14. We condition throughout on $\Sigma _ { t }$ , which fixes $X _ { t }$ and $D _ { t } ^ { \mathrm { L P } } . \mathrm { H } D _ { t } ^ { \mathrm { L P } } = \varnothing$ , both quantities in the lemma are zero. We therefore assume $D _ { t } ^ { \mathrm { L P } } \neq \varnothing$ . By the definition of $D _ { t + 1 } ^ { \mathrm { L P } }$ in Algorithm 3, if $\delta ( X _ { t + 1 } , D _ { t } ^ { \mathrm { { i P } } } ) \geq 0$ , then $D _ { t + 1 } ^ { \mathrm { L P } }$ contains $D _ { t } ^ { \mathrm { L P } }$ Thus $\mathsf { \bar { D } } _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } \neq \dot { \varnothing }$ can occur only when $\delta ( { \bar { X } } _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) < \mathsf { 0 }$ Since $\tilde { m } ( \tilde { D } _ { t } ^ { \mathrm { L P } } \backslash \tilde { D } _ { t + 1 } ^ { \mathrm { L P } } ) \leq m ( D _ { t } ^ { \mathrm { L P } } )$ ), we have

$$
\mathbb { E } \big [ m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) \big | \Sigma _ { t } \big ]
$$

$$
\leq m ( D _ { t } ^ { \mathrm { { L P } } } ) \mathbb { P } \big [ \delta ( X _ { t + 1 } , D _ { t } ^ { \mathrm { { L P } } } ) < 0 \big | \Sigma _ { t } \big ]
$$

$$
\mathbb { P } \big [ m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) > 0 \big | \Sigma _ { t } \big ]\tag{51}
$$

$$
\leq \mathbb { P } \big [ \delta ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) < 0 \big | \Sigma _ { t } \big ] .\tag{52}
$$

To bound the probability that the old set becomes infeasible, we recall from (18) that $\delta ( x , D ) = \eta m ( D ) - h _ { U } ( x , D )$ . The policy chooses $D _ { t } ^ { \mathrm { L P } }$ to be feasible at time $t ,$ so $h _ { U } ( \bar { X } _ { t } , \bar { D } _ { t } ^ { \mathrm { L P } } ) \leq$ $\bar { \eta } m ( \bar { D } _ { t } ^ { \mathrm { L P } } )$ . Thus

$$
\begin{array} { r l } & { \mathbb { P } \left[ \delta ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) < 0 \middle | \Sigma _ { t } \right] } \\ & { \quad = \mathbb { P } \left[ h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) > \eta m ( D _ { t } ^ { \mathrm { L P } } ) \middle | \Sigma _ { t } \right] } \\ & { \quad \le \mathbb { P } \Big [ h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - \rho _ { u } h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) } \\ & { \qquad > ( 1 - \rho _ { u } ) \eta m ( D _ { t } ^ { \mathrm { L P } } ) \Big | \Sigma _ { t } \Big ] } \\ & { \quad \le K _ { U , 2 } \exp \big ( - C _ { U } N ( 1 - \rho _ { u } ) ^ { 2 } \eta ^ { 2 } m ( D _ { t } ^ { \mathrm { L P } } ) ^ { 2 } \big ) . } \end{array}\tag{53}
$$

The last inequality applies the $h _ { U }$ tail bound of Lemma 13 at $r = ( 1 - \rho _ { u } ) \eta m ( D _ { t } ^ { \mathrm { L P } } )$ . Conditional on $\Sigma _ { t }$ , the arms in the fixed set $D _ { t } ^ { \mathrm { L P } }$ follow LP fixed-point with fresh randomness, so the lemma’s conditional calculation applies. Combining (53) with (51)–(52) proves both claims with

$$
K _ { D , 1 } = K _ { D , 2 } = K _ { U , 2 } , \qquad C _ { D } = C _ { U } ( 1 - \rho _ { u } ) ^ { 2 } \eta ^ { 2 } .
$$

These constants are positive and independent of N and the current state. □

Finally, we restate and prove Lemma 15.

Lemma 15 (Sufficient coverage). Assuming Assumptions 1 to 3 and $\epsilon _ { N } ^ { r d } = O ( 1 / N )$ , we have that for any $t \geq 0$

$$
\begin{array} { r l } & { 1 - m ( D _ { t } ^ { \mathrm { L P } } ) \leq K _ { C , 1 } h _ { W } \big ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } \big ) + \displaystyle \frac { K _ { C , 2 } } { N } \quad a . s . , } \\ & { ~ m ( D _ { t } ^ { \mathrm { L P } } ) = 1 \quad i f h _ { U } \big ( X _ { t } , [ N ] \big ) \leq \eta , } \end{array}
$$

where $K _ { C , 1 } , K _ { C , 2 } > 0$ are constants independent of N.

Proof of Lemma 15. The second claim of Lemma 15 follows directly from the definition of the policy, so we only need to prove its first claim. We claim that either $m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \leq \epsilon _ { N } ^ { \mathrm { r d } }$ , or

$$
\lambda _ { U } ^ { 1 / 2 } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) > \eta m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - \epsilon _ { N } ^ { \mathrm { r d } } - 0 .\tag{54}
$$

We first show Lemma 15 assuming this claim. We recall that $m ( D _ { t } ^ { \bar { \pi } ^ { \ast } } ) = \big \lfloor \beta ( N - | D _ { t } ^ { \mathrm { L P } } | ) \big \rfloor / N$ . If $m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \leq \epsilon _ { N } ^ { \mathrm { r d } }$ ,

$$
\beta ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) - \frac { 1 } { N } \leq m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \leq \epsilon _ { N } ^ { \mathrm { r d } } ,
$$

so $1 - m ( D _ { t } ^ { \mathrm { L P } } ) \leq 1 / ( \beta N ) + \epsilon _ { N } ^ { \mathrm { r d } } / \beta ,$ which implies Lemma 15 by the non-negativity of $h _ { W } ( \bar { X _ { t } } , D _ { t } ^ { \bar { \pi } ^ { * } } )$ . If $m ( \bar { D } _ { t } ^ { \bar { \pi } ^ { * } } ) > \epsilon _ { N } ^ { \mathrm { r d } } , ( 5 4 )$ holds. Because m $( D _ { t } ^ { \bar { \pi } ^ { * } } ) = \lfloor \beta ( N - | D _ { t } ^ { \mathrm { L P } } | ) \rfloor / N \ge ( \ddot { \beta } ( N - $ $| D _ { t } ^ { \mathrm { L P } } | ) - 1 ) / N$ , we have

$$
\begin{array} { r l r } {  { \lambda _ { U } ^ { 1 / 2 } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) } } \\ & { } & { > \eta \big ( \beta ( N - | D _ { t } ^ { \mathrm { L P } } | ) - 1 \big ) \frac { 1 } { N } - \epsilon _ { N } ^ { \mathrm { r d } } - 0 } \\ & { } & { = \eta \beta ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) - \frac { \eta } { N } - \epsilon _ { N } ^ { \mathrm { r d } } - 0 . } \end{array}
$$

Rearranging gives

$$
1 - m ( D _ { t } ^ { \mathrm { L P } } ) < \frac { \lambda _ { U } ^ { 1 / 2 } } { \eta \beta } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) + \frac { 1 } { \beta N } + \frac { \epsilon _ { N } ^ { \mathrm { r d } } + 0 } { \eta \beta } .
$$

Combining the two cases, we have

$$
\begin{array} { r l r } {  { 1 - m ( D _ { t } ^ { \mathrm { L P } } ) \leq \frac { \lambda _ { U } ^ { 1 / 2 } } { \eta \beta } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) + \frac { 1 } { \beta N } } } \\ & { } & { \quad + \frac { \epsilon _ { N } ^ { \mathrm { r d } } + 0 } { \operatorname* { m i n } ( \eta , 1 ) \beta } . } \end{array}
$$

Since $\epsilon _ { N } ^ { \mathrm { r d } } , 0 \ = \ O ( 1 / N )$ , the additive term is ${ \cal O } ( 1 / N )$ , and we can write the bound as $K _ { C , 1 } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) + K _ { C , 2 } / N$ for positive constants $K _ { C , 1 } , K _ { C , 2 }$ independent of $N ,$ establishing Lemma 15.

Now we prove the claim by contradiction. We suppose that, at a certain time t, we have $\begin{array} { r l r } { m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) } & { { } > } & { \epsilon _ { N } ^ { \mathrm { r d } } } \end{array}$ and $\lambda _ { U } ^ { 1 / 2 } h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \leq \eta m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - \epsilon _ { N } ^ { \mathrm { r d } } - 0$ . Because $\| v \| _ { U } \leq$ $\lambda _ { U } ^ { \breve { 1 } / 2 } \left. v \right. _ { 2 } \leq \lambda _ { U } ^ { \breve { 1 } / 2 } \left. v \right. _ { W }$ for any $v \in \mathbb { R } ^ { | \mathbb { S } | }$

$$
\begin{array} { r l } & { \left\| X _ { t } ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \mu ^ { * } \right\| _ { U } } \\ & { \quad \leq \lambda _ { U } ^ { 1 / 2 } \| X _ { t } ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \mu ^ { * } \| _ { W } } \\ & { \quad \leq \eta m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - \epsilon _ { N } ^ { \mathrm { r d } } - 0 . } \end{array}
$$

Combined with the triangular inequality and the definition of $D _ { t } ^ { \mathrm { L P } }$ , we have

$$
\begin{array} { r l } & { \left\| X _ { t } ( D _ { t } ^ { \mathrm { L P } } \cup D _ { t } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t } ^ { \mathrm { L P } } \cup D _ { t } ^ { \bar { \pi } ^ { * } } ) \mu ^ { * } \right\| _ { U } } \\ & { \quad \leq \left\| X _ { t } ( D _ { t } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } ) \mu ^ { * } \right\| _ { U } } \\ & { \quad \quad + \left\| X _ { t } ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \mu ^ { * } \right\| _ { U } } \\ & { \quad \leq \eta m ( D _ { t } ^ { \mathrm { L P } } ) + \eta m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - \epsilon _ { N } ^ { \mathrm { r d } } - 0 . } \end{array}
$$

Consequently, $D _ { t } ^ { \mathrm { L P } } \cup D _ { t } ^ { \bar { \pi } ^ { * } }$ is a superset of $D _ { t } ^ { \mathrm { L P } }$ such that $\delta ( X _ { t } , \bar { D } _ { t } ^ { \mathrm { L P } } \cup \bar { D } _ { t } ^ { * ^ { * } } ) \geq \epsilon _ { N } ^ { \mathrm { r d } ^ { * } }$ and $m ( D _ { t } ^ { \mathsf { \bar { L } P } } \cup D _ { t } ^ { \bar { \pi } ^ { * } } ) \stackrel { \cdot } { = } m ( D _ { t } ^ { \mathsf { L P } } ) +$ $m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) > m ( D _ { t } ^ { \mathrm { L P } } ) + \bar { \epsilon } _ { N } ^ { \mathrm { r d } }$ , contradicting the ϵ<sup>rd</sup><sub>N</sub>-maximality of $D _ { t } ^ { \mathrm { L P } }$ . We have thus proved the claim that implies Lemma 15. □

## C. Proof of Lemma 3

Lemma 3 (Instantaneous reward under LP fixed-point). Under the WCMDP two-set policy, for any $\sigma = ( x , D ^ { \mathrm { \bar { L } P } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X }$ with $\| x ( [ N ] ) - \mu ^ { * } \| _ { U } \leq \overline { { \eta } } \triangleq \eta ,$ , we have $D ^ { \mathrm { L P } } = [ N ]$ , and

$$
r ^ { \pi } ( \sigma ) = \hat { r } ( x ( [ N ] ) ) \ + \ \epsilon _ { N } ^ { r e w } ( \sigma ) ,
$$

where ${ \hat { r } } ( v ) \ { \stackrel { \triangle } { = } } \ R ^ { \mathrm { r e l } } + ( v - \mu ^ { * } ) g$ with $g \triangleq C _ { \mathbb { S } } ^ { + } r ^ { \top } -$ $( \mu ^ { * } C _ { \mathbb { S } } ^ { + } r ^ { \top } ) \mathbb { 1 } ^ { \top } \in \mathbb { R } ^ { | \mathbb { S } | }$ , r is the row vector $( r ( s , a ) ) _ { s \in \mathbb { S } , a \in \mathbb { A } } ,$ and $\epsilon _ { N } ^ { \tilde { r } e w } \colon \mathbb { X }  \mathbb { R }$ satisfies $\lvert \epsilon _ { N } ^ { r e w } ( \sigma ) \rvert \le 2 r _ { \mathrm { m a x } } \lvert \mathbb { S } \rvert ( \lvert \mathbb { A } \rvert - 1 ) / N$ Moreover, $\| g \| _ { \infty } \le K _ { g } \triangleq 2 r _ { \operatorname* { m a x } } \bigl ( 1 + \sqrt { 2 } \lambda _ { U } ^ { 1 / 2 } / \eta \bigr )$

Proof of Lemma 3. When $\| x ( [ N ] ) - \mu ^ { * } \| _ { U } ~ \le ~ \overline { { \eta } } .$ , LP fixedpoint is feasible for all arms. Since the two-set policy chooses $\mathbf { \dot { \phi } } _ { D _ { t } ^ { \mathrm { L P } } } ^ { \mathrm { L P } }$ to be a maximal set that can follow LP fixed-point, we have $D _ { t } ^ { \mathrm { L P } } ~ = ~ [ N ]$ and $D _ { t } ^ { \bar { \pi } ^ { * } } = \varnothing$ . Letting $y _ { t } ~ = ~ y ^ { * } +$ $( \boldsymbol { x } ( [ N ] ) - \mu ^ { * } ) \boldsymbol { C } _ { \mathbb { S } } ^ { + }$ , we perform the following decomposition of the instantaneous expected reward $r ^ { \pi } ( \sigma )$

$$
\begin{array} { l } { \displaystyle r ^ { \pi } ( \sigma ) = \mathbb { E } \left[ \left. \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } r ( s , a ) Y _ { t } ( s , a ) \right| \mathbb { \Sigma } _ { t } = \sigma \right] } \\ { = \displaystyle \sum _ { s \in \mathbb { S } , a \in \mathbb { A } } r ( s , a ) Y _ { t } ( s , a ) } \\ { = y ^ { * } r ^ { \top } + ( y _ { t } - y ^ { * } ) r ^ { \top } + ( Y _ { t } - y _ { t } ) r ^ { \top } } \\ { = y ^ { * } r ^ { \top } + ( x ( [ N ] ) - \mu ^ { * } ) C _ { s } ^ { + } r ^ { \top } + \epsilon _ { N } ^ { \mathrm { r e w } } ( \sigma ) , } \end{array}
$$

Since $y ^ { * } r ^ { \top } = R ^ { \mathrm { r e l } }$ and $\left( x ( [ N ] ) - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + } \pmb { r } ^ { \top } = \left( x ( [ N ] ) - \right.$ $\mu ^ { * } ) g$ (the shift term in the definition of g does not contribute because $( x ( [ N ] ) - \mu ^ { * } ) \mathbb { 1 } ^ { \top } = 0 )$ , this gives $r ^ { \pi } ( \sigma ) =$ $\hat { r } ( x ( [ N ] ) ) + \epsilon _ { N } ^ { \mathrm { r e w } } ( \sigma )$ . To bound $\epsilon _ { N } ^ { \mathrm { r e w } } ( \sigma ) = ( Y _ { t } - y _ { t } ) { \boldsymbol { r } } ^ { \top }$ , we note that by Lemma 11, we have $\| \dot { Y } _ { t } - y _ { t } \| _ { 1 } \leq 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N$ Therefore,

$$
| \epsilon _ { N } ^ { \mathrm { r e w } } ( \sigma ) | \leq r _ { \operatorname* { m a x } } \| Y _ { t } - y _ { t } \| _ { 1 } \leq \frac { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } .
$$

Finally, we show $\| \pmb { g } \| _ { \infty } \le K _ { g }$ . By construction, $\mu ^ { \ast } g =$ $\mu ^ { * } C _ { \mathbb { S } } ^ { + } \pmb { r } ^ { \top } - ( \mu ^ { * } C _ { \mathbb { S } } ^ { + } \pmb { r } ^ { \top } ) ( \pmb { \mu } ^ { * } \pmb { \mathrm { 1 } } ^ { \top } ) = 0$ . Hence, for each $s \in \mathbb { S } ,$

$$
\begin{array} { r } { \pmb { g } ( s ) = ( e _ { s } - \mu ^ { * } ) \pmb { g } , } \end{array}\tag{55}
$$

where $e _ { s }$ denotes the point mass at state s. If $e _ { s } = \mu ^ { * }$ , then (55) gives $\begin{array} { r } { \pmb { g } ( s ) = 0 . } \end{array}$ , so the desired bound holds. We therefore assume $e _ { s } \neq \mu ^ { * }$ below. To bound the right-hand side of (55), we let $\theta \triangleq \operatorname* { m i n } \ ( 1 , \eta / \left. e _ { s } - \mu ^ { * } \right. _ { U } )$ and $v \triangleq \left( 1 - \theta \right) \mu ^ { * } + \theta e _ { s } ,$ so that $v ~ \in ~ \dot { \Delta } ( \mathbb { S } )$ and $\Vert v - \mu ^ { * } \Vert _ { U } ~ \leq ~ \eta .$ By Lemma 9, $y ( v ) \triangleq y ^ { * } + \left( v - \mu ^ { * } \right) C _ { \mathbb { S } } ^ { + }$ is entrywise non-negative; since its state-marginals equal v by $( 6 ) , y ( v )$ is a probability distribution over $\mathbb { S } \times \mathbb { A }$ . Therefore, $\hat { r } ( v ) = y ( v ) r ^ { \top } \in [ - r _ { \operatorname* { m a x } } , r _ { \operatorname* { m a x } } ]$ . Combining this with $\hat { r } ( v ) - R ^ { \mathrm { r e l } } = \theta \left( e _ { s } - \mu ^ { * } \right) g$ and $\begin{array} { r } { | R ^ { \mathrm { r e l } } | \leq r _ { \operatorname* { m a x } } , } \end{array}$ we get

$$
\begin{array} { r l r } {  { \vert \boldsymbol { g } ( \boldsymbol { s } ) \vert = \frac {  \hat { r } ( \boldsymbol { v } ) - R ^ { \mathrm { r e l } }  } { \theta } \le \frac { 2 r _ { \mathrm { m a x } } } { \theta } } } \\ & { } & { \le 2 r _ { \mathrm { m a x } } \operatorname* { m a x } \Bigl ( 1 , \frac { \Vert \boldsymbol { e } _ { s } - \boldsymbol { \mu } ^ { * } \Vert _ { U } } { \eta } \Bigr ) } \\ & { } & { \le 2 r _ { \mathrm { m a x } } \Bigl ( 1 + \frac { \sqrt { 2 } \lambda _ { U } ^ { 1 / 2 } } { \eta } \Bigr ) = K _ { g } , } \end{array}
$$

where in the last inequality, we use the facts that $\begin{array} { r l r } { \| e _ { s } - \mu ^ { * } \| _ { U } } & { { } \le } & { \lambda _ { U } ^ { 1 / 2 } \ \| e _ { s } - \mu ^ { * } \| _ { 2 } } \end{array}$ and $\begin{array} { r l } { \| e _ { s } - \mu ^ { * } \| _ { 2 } ^ { 2 } } & { { } \leq } \end{array}$ $\left\| e _ { s } - \mu ^ { * } \right\| _ { 1 } \left\| e _ { s } - \mu ^ { * } \right\| _ { \infty } \leq 2 .$ □

To prove the global reward bound (20), we first consider $\| x ( [ N ] ) - \mu ^ { * } \| _ { U } \leq \eta$ . Since $R ^ { \mathrm { r e l } } - \hat { r } ( v ) = ( \mu ^ { * } - v ) g$ , Lemma 3 gives

$$
R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) \leq ( \mu ^ { * } - x ( [ N ] ) ) \pmb { \mathscr { g } } + \frac { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } .
$$

The first term is at most $\begin{array} { r l } { K _ { g } \left\| \mu ^ { * } - x ( [ N ] ) \right\| _ { 1 } } & { { } ~ \leq } \end{array}$ $\sqrt { | \mathbb { S } | } K _ { g } \| \mu ^ { * } - x ( [ N ] ) \| _ { U } .$ , because $U \succeq I .$ . Outside this neighborhood, $R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) \leq 2 r _ { \operatorname* { m a x } } \leq ( 2 r _ { \operatorname* { m a x } } / \eta ) \lVert \mu ^ { * } - x ( [ N ] ) \rVert _ { U } .$ Choosing $K _ { 0 } \ \geq$ max $\{ \sqrt { | \mathbb { S } | } K _ { g } , 2 r _ { \operatorname* { m a x } } / \eta \}$ , we therefore get (20) in both cases.

## D. Proof of Lemma 4

Lemma 4 (Drift of truncated $V _ { 1 } )$ . For any $\sigma , \sigma ^ { \prime } \in \mathbb { X } ,$

$$
\begin{array} { r l r } {  { \big ( V _ { 1 } ( \sigma ^ { \prime } ) - \frac { \overline { \eta } } { 2 } \big ) ^ { + } - \big ( V _ { 1 } ( \sigma ) - \frac { \overline { \eta } } { 2 } \big ) ^ { + } } } \\ & { } & { \leq - \frac { ( 1 - \rho _ { 1 } ) \overline { \eta } } { 2 } \Im \{ V _ { 1 } ( \sigma ) > \overline { \eta } \} } \\ & { } & { + ( V _ { 1 } ( \sigma ^ { \prime } ) - \rho _ { 1 } V _ { 1 } ( \sigma ) - \frac { ( 1 - \rho _ { 1 } ) \overline { \eta } } { 2 } ) ^ { + } . } \end{array}
$$

Proof. We use the abbreviation that $a \triangleq V _ { 1 } ( \sigma ) , b \triangleq V _ { 1 } ( \sigma ^ { \prime } )$ and $r _ { 0 } \triangleq ( 1 - \rho _ { 1 } ) \overline { { \eta } } / 2$ . The key algebraic identity

$$
b - \frac { \overline { { { \eta } } } } { 2 } = \Big ( b - \rho _ { 1 } a - r _ { 0 } \Big ) + \rho _ { 1 } \Big ( a - \frac { \overline { { { \eta } } } } { 2 } \Big ) ,\tag{56}
$$

together with subadditivity $( x + y ) ^ { + } \leq x ^ { + } + y ^ { + }$ , yields

$$
\begin{array} { l } { \displaystyle { \left( b - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } - \left( a - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } } } \\ { \displaystyle { \quad \le \left( b - \rho _ { 1 } a - r _ { 0 } \right) ^ { + } } } \\ { \displaystyle { \quad \quad - ( 1 - \rho _ { 1 } ) \left( a - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } } . } \end{array}\tag{57}
$$

We derive the claim from (57) by cases on a.

Case 1: $a \leq { \overline { { \eta } } } .$ . Since $( 1 - \rho _ { 1 } ) ( a - \overline { { \eta } } / 2 ) ^ { + } \geq 0 ,$ (57) gives

$$
\left( b - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } - \left( a - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } \leq \left( b - \rho _ { 1 } a - r _ { 0 } \right) ^ { + } ,
$$

which matches the RHS of Lemma 4 because $\mathbb { 1 } \{ a > \overline { { \eta } } \} = 0 .$ Case $2 \colon a > \overline { { \eta } } .$ Then $( a - \overline { { { \eta } } } / 2 ) ^ { + } = a - \overline { { { \eta } } } / 2 > \overline { { { \eta } } } / 2 .$ , so $( 1 - \rho _ { 1 } ) ( a - \overline { { \eta } } / 2 ) ^ { + } > ( 1 - \rho _ { 1 } ) \overline { { \eta } } / 2 = r _ { 0 }$ . Hence (57) gives

$$
\left( b - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } - \left( a - \frac { \overline { { \eta } } } { 2 } \right) ^ { + } \leq \left( b - \rho _ { 1 } a - r _ { 0 } \right) ^ { + } - r _ { 0 } ,
$$

which matches the RHS of Lemma 4 because $\mathbb { 1 } \{ a > \overline { { \eta } } \} =$ 1. □

## E. Proof of Lemma 5

Lemma 5 (Drift of Ψ). For any $\sigma = ( x , D ^ { \mathrm { L P } } , D ^ { \bar { \pi } ^ { * } } ) \in \mathbb { X } ,$

$$
\begin{array} { r l } & { \Delta \Psi ( \sigma ) } \\ & { \leq - ( R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) ) + \displaystyle \frac { K _ { \Psi } } { N } } \\ & { \quad + \left( 2 \lambda _ { Q } K _ { g } + 2 r _ { \operatorname* { m a x } } + 2 K _ { g } \right) \mathbb { 1 } \{ \left\| x ( \left[ N \right] ) - \mu ^ { * } \right\| _ { U } > \overline { { \eta } } \} , } \\ & { \quad \quad ( 2 9 ) } \end{array}
$$

where $K _ { \Psi } \ge 0$ is a constant independent of N and σ.

Proof of Lemma 5. Using $( I - \Phi ) Q \ = \ I , \ \Delta \Psi$ admits the decomposition

$$
\begin{array} { r l } & { \Delta \Psi ( \sigma ) } \\ & { \quad = \Big ( \mathbb { E } \big [ ( \mu ^ { * } - X _ { t + 1 } ( [ N ] ) ) Q g \big | \Sigma _ { t } = \sigma \big ] } \\ & { \quad \quad - ( \mu ^ { * } - x ( [ N ] ) ) \Phi Q g \Big ) } \\ & { \quad \quad - \big ( R ^ { \mathrm { r e l } } - \hat { r } ( x ( [ N ] ) ) \big ) . } \end{array}\tag{58}
$$

The first term on the right-hand side of (58) can be bounded as

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( \mu ^ { * } - X _ { t + 1 } ( [ N ] ) ) Q g \big | \Sigma _ { t } = \sigma \big ] } \\ & { \quad - ( \mu ^ { * } - x ( [ N ] ) ) \Phi Q g } \\ & { \quad \le 2 \lambda _ { Q } K _ { g } \Im \big \{ D _ { t } ^ { \mathrm { L P } } \ne [ N ] \big \} + \frac { \tilde { K } } { N } . } \end{array}\tag{59}
$$

where $\tilde { K } = 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) \ \| Q \| _ { \infty } \ K _ { g }$ . Here the term $2 \lambda _ { Q } K _ { g }$ comes from bounding the left-hand side by $2 \lambda _ { Q } \left\| \pmb { g } \right\| _ { \infty }$ on the event $\begin{array} { r l r } { D _ { t } ^ { \mathrm { L P } } } & { { } \neq } & { [ N ] } \end{array}$ , and the term $\tilde { K } / N$ comes from bounding the rounding error term in Lemma 1 by $\left\| Y _ { t } - y _ { t } \right\| _ { 1 } \left\| Q \right\| _ { \infty } \left\| g \right\| _ { \infty }$ on the event $\begin{array} { l l l } { { D _ { t } ^ { \mathrm { L P } } } } & { { = } } & { { [ N ] } } \end{array}$ ; both bounds use the property $\| \pmb { g } \| _ { \infty } \le K _ { g }$ from Lemma 3.

For the second term on the right-hand side of (58), we have

$$
\begin{array} { r l } {  { - \big ( R ^ { \mathrm { r e l } } - \hat { r } \big ( x ( [ N ] ) \big ) \big ) } } \\ & { \leq - ( R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) ) } \\ & { \phantom { = } + \big ( 2 r _ { \operatorname* { m a x } } + 2 K _ { g } \big ) \mathbb { 1 } \{ \| x ( [ N ] ) - \mu ^ { * } \| _ { U } > \overline { { \eta } } \} } \\ & { \phantom { = } + \frac { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } { N } } \end{array}\tag{60}
$$

where the term $2 r _ { \mathrm { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N$ arises from the application of Lemma 3, which states that $| { \hat { r } } ( x ( [ N ] ) ) - r ^ { \pi } ( { \bar { \sigma ) | } } \leq$ $2 r _ { \mathrm { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) / N$ when $D _ { t } ^ { \mathrm { L P } } = [ N ] ;$ ; the factor $( 2 r _ { \mathrm { m a x } } +$ $2 K _ { g } )$ on the event $\| x ( [ N ] ) - \mu ^ { * } \| _ { U } > \overline { { \eta } }$ comes from $| \hat { r } ( v ) | \leq$ $| R ^ { \mathrm { r e l } } | + \left\| v - \mu ^ { * } \right\| _ { 1 } \left\| g \right\| _ { \infty } \leq r _ { \operatorname* { m a x } } + 2 K _ { g }$ for all $v \in \Delta ( \mathbb { S } )$ together with $\left. r ^ { \pi } ( \sigma ) \right. \leq r _ { \operatorname* { m a x } } .$

Substituting (59) and (60) into (58) and using 1 $\{ D _ { t } ^ { \mathrm { L P } } \neq [ N ] \} = \mathbb { 1 } \{ \| x ( [ N ] ) - \mu ^ { * } \| _ { U } > \overline { { \eta } } \}$ , we obtain

∆Ψ(σ)

$$
\begin{array} { r l } {  { \le - ( R ^ { \mathrm { r e l } } - r ^ { \pi } ( \sigma ) ) + \frac { K _ { \Psi } } { N } } } \\ & { + ( 2 \lambda _ { Q } K _ { g } + 2 r _ { \operatorname* { m a x } } + 2 K _ { g } ) \mathbb { 1 } \{ \| x ( [ N ] ) - \mu ^ { * } \| _ { U } > \overline { { \eta } } \} , } \end{array}
$$

with $\begin{array} { r c l c r c l } { K _ { \Psi } } & { \triangleq } & { \tilde { K } } & { + } & { 2 r _ { \operatorname* { m a x } } | \mathbb { S } | ( | \mathbb { A } | - 1 ) } & { = } & { 2 | \mathbb { S } | ( | \mathbb { A } | - 1 ) } \end{array}$ $1 ) \left( \left\| Q \right\| _ { \infty } K _ { g } + r _ { \operatorname* { m a x } } \right)$ □

## F. Supporting Lyapunov calculations

We recall that $V _ { 1 } ( \sigma )$ is defined as

$$
\begin{array} { r l } & { V _ { 1 } ( \sigma ) = h _ { U } ( x , D ^ { \mathrm { L P } } ) + h _ { W } ( x , D ^ { \bar { \pi } ^ { * } } ) } \\ & { ~ + L _ { 1 } ( 1 - m ( D ^ { \mathrm { L P } } ) ) . } \end{array}\tag{61}
$$

Proof of (26). By Lemma 13, $h _ { U } ( x , D )$ is Lipschitz continuous in $D ,$ so $h _ { U } ( x , D ^ { \mathrm { L P } } )$ changes by at most $2 \lambda _ { U } ^ { 1 / 2 } ( 1 -$ $m ( D ^ { \mathrm { L P } } ) )$ when we replace $\dot { D } ^ { \mathrm { L P } }$ with $[ N ]$ . Consequently,

$$
\begin{array} { r l } & { V _ { 1 } ( \sigma ) \geq h _ { U } ( x , D ^ { \mathrm { L P } } ) + 2 \lambda _ { U } ^ { 1 / 2 } ( 1 - m ( D ^ { \mathrm { L P } } ) ) } \\ & { \qquad \geq h _ { U } ( x , [ N ] ) } \\ & { \qquad = \| x ( [ N ] ) - \mu ^ { * } \| _ { U } . } \end{array}
$$

A preliminary bound for proving (24) and (25). We prove the following bound for constants $\rho _ { 1 } \in ( 0 , 1 )$ and $K _ { \mathrm { d r i f t } } > 0$ both independent of N and specified below:

$$
\begin{array} { r l } & { V _ { 1 } ( \Sigma _ { t + 1 } ) - \rho _ { 1 } V _ { 1 } ( \Sigma _ { t } ) } \\ & { \quad \le \big ( h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - \rho _ { u } h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \big ) } \\ & { \quad \quad + \big ( h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \overline { { \pi } } ^ { * } } ) - \rho _ { w } h _ { W } ( X _ { t } , D _ { t } ^ { \overline { { \pi } } ^ { * } } ) \big ) } \\ & { \quad \quad + 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \big ) m ( D _ { t } ^ { \mathrm { L P } } \setminus D _ { t + 1 } ^ { \mathrm { L P } } ) } \\ & { \quad \quad + \frac { K _ { \mathrm { d r i f t } } } { N } . } \end{array}\tag{62}
$$

(63)

To show this, we start with the decomposition:

$$
\begin{array} { r l } & { V _ { 1 } ( \Sigma _ { t + 1 } ) - V _ { 1 } ( \Sigma _ { t } ) } \\ & { \quad = \big ( V _ { 1 } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) - V _ { 1 } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \big ) } \\ & { \quad \quad + \left( V _ { 1 } ( X _ { t + 1 } , D _ { t + 1 } ^ { \mathrm { L P } } , D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) - V _ { 1 } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \right) } \end{array} (\tag{64}
$$

(65)

We refer to the difference term in right-hand side of (65) the set-update term, and the difference term in (64) the statetransition term. We calculate these terms separately.

For the state-transition term in (64), it follows directly from the definition of $V _ { 1 }$ that

$$
\begin{array} { r l } & { V _ { 1 } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) - V _ { 1 } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) } \\ & { \quad \leq \left( h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \right) } \\ & { \quad \quad + \left( h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \bar { \pi } ^ { * } } ) - h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \right) } \end{array}\tag{66}
$$

For the set-update term in (65), we can derive a bound fully in terms of the sizes of sets, using the Lipschitz continuity of $h _ { W } ( x , D )$ and $h _ { U } ( x , D )$ with respect to $D$ given in (40) of Lemma 13:

$$
\begin{array} { r l } & { V _ { 1 } ( X _ { t + 1 } , D _ { t + 1 } ^ { \mathrm { L P } } , D _ { t + 1 } ^ { \pi ^ { * } } ) - V _ { 1 } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \pi ^ { * } } ) } \\ & { \quad = h _ { U } ( X _ { t + 1 } , D _ { t + 1 } ^ { \mathrm { L P } } ) - h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) } \\ & { \quad \quad + h _ { W } ( X _ { t + 1 } , D _ { t + 1 } ^ { \pi ^ { * } } ) - h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \pi ^ { * } } ) } \\ & { \quad \quad - L _ { 1 } \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } ) \big ) } \\ & { \quad \le 2 \lambda _ { U } ^ { 1 / 2 } \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) + m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) \big ) } \\ & { \quad \quad + 2 \lambda _ { W } ^ { 1 / 2 } \big ( m ( D _ { t + 1 } ^ { \pi ^ { * } } \backslash D _ { t } ^ { \pi ^ { * } } ) + m ( D _ { t } ^ { \pi ^ { * } } \backslash D _ { t + 1 } ^ { \pi ^ { * } } ) \big ) } \\ & { \quad \quad - L _ { 1 } \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) \big ) . } \end{array}\tag{67}
$$

(68)

Next, we show that

$$
\begin{array} { r l } & { m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \backslash D _ { t } ^ { \bar { \pi } ^ { * } } ) + m ( D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) } \\ & { \quad \le 2 m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) } \\ & { \quad \quad - \beta \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } ) \big ) + \frac { 1 } { N } . } \end{array}\tag{69}
$$

We discuss based on whether $D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \supseteq D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \mathrm { L P } }$ or $D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \subseteq$ $D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \mathrm { L P } }$

• If $D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \supseteq D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \mathrm { L P } }$ , because $D _ { t } ^ { \bar { \pi } ^ { * } }$ is disjoint from $D _ { t } ^ { \mathrm { L P } }$ it is not hard to see that

$$
\begin{array} { r l } & { \qquad D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \subseteq D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } , } \\ & { m ( D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) \leq m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) . } \end{array}\tag{70}
$$

Moreover, by the definition of $D _ { t } ^ { \bar { \pi } ^ { * } } , m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) \geq \beta ( 1 -$ $m ( D _ { t } ^ { \mathrm { L P } } ) ) - \mathrm { \dot { 1 } } / N$ and $m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) \leq \bar { \beta } ( 1 - m ( D _ { t + 1 } ^ { \mathrm { L P } } ) )$ , so

$$
\begin{array} { r l } & { m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \backslash D _ { t } ^ { \bar { \pi } ^ { * } } ) } \\ & { \quad = m ( D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) + m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) } \\ & { \quad \le m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) } \\ & { \quad \quad - \beta \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } ) \big ) + \frac { 1 } { N } . } \end{array}\tag{71}
$$

Combining (70) and (71), we get (69).

• If $D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \subseteq D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \mathrm { L P } }$ , we have $m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } \backslash D _ { t } ^ { \bar { \pi } ^ { * } } ) = 0$ and

$$
\begin{array} { r l } & { m ( D _ { t } ^ { \bar { \pi } ^ { * } } \backslash D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) = m ( D _ { t } ^ { \bar { \pi } ^ { * } } ) - m ( D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) } \\ & { \qquad = \displaystyle \frac { 1 } { N } \Big ( \lfloor \beta N ( 1 - m ( D _ { t } ^ { \mathrm { L P } } ) ) \rfloor } \\ & { \qquad - \lfloor \beta N ( 1 - m ( D _ { t + 1 } ^ { \mathrm { L P } } ) ) \rfloor \Big ) } \\ & { \qquad \leq \beta \big ( m ( D _ { t + 1 } ^ { \mathrm { L P } } ) - m ( D _ { t } ^ { \mathrm { L P } } ) \big ) + \displaystyle \frac { 1 } { N } , } \end{array}
$$

which implies (69) because m $( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } ) \geq m ( D _ { t + 1 } ^ { \mathrm { L P } } ) -$ $m ( D _ { t } ^ { \mathrm { L P } } )$ and $\beta < 1$

Plugging (69) into (67) and rearranging the terms, we get an upper bound for the set-update term:

$$
\begin{array} { r l } & { V _ { 1 } ( X _ { t + 1 } , D _ { t + 1 } ^ { \mathrm { L P } } , D _ { t + 1 } ^ { \bar { \pi } ^ { * } } ) - V _ { 1 } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } , D _ { t } ^ { \bar { \pi } ^ { * } } ) } \\ & { \quad \leq 4 \bigl ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \bigr ) m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) } \\ & { \quad \quad + \frac { 2 \lambda _ { W } ^ { 1 / 2 } } { N } , } \end{array}\tag{72}
$$

where the terms involving to $m ( D _ { t + 1 } ^ { \mathrm { L P } } \backslash D _ { t } ^ { \mathrm { L P } } )$ have been canceled out.

Substituting the above calculation into (64) and (65), we get

$$
\begin{array} { r l } & { V _ { 1 } ( \Sigma _ { t + 1 } ) - V _ { 1 } ( \Sigma _ { t } ) } \\ & { \quad \le \big ( h _ { U } ( X _ { t + 1 } , D _ { t } ^ { \mathrm { L P } } ) - h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) \big ) } \\ & { \quad \quad + \big ( h _ { W } ( X _ { t + 1 } , D _ { t } ^ { \bar { \pi } ^ { * } } ) - h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { * } } ) \big ) } \\ & { \quad \quad + 4 \big ( \lambda _ { U } ^ { 1 / 2 } + \lambda _ { W } ^ { 1 / 2 } \big ) m ( D _ { t } ^ { \mathrm { L P } } \backslash D _ { t + 1 } ^ { \mathrm { L P } } ) } \\ & { \quad \quad + \frac { 2 \lambda _ { W } ^ { 1 / 2 } } { N } . } \end{array}\tag{73}
$$

(74)

By Lemma 15, we have

$$
\begin{array} { r l } & { V _ { 1 } ( \Sigma _ { t } ) \leq h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) + h _ { W } ( X _ { t } , D _ { t } ^ { \overline { { \pi } } ^ { * } } ) } \\ & { \qquad + L _ { 1 } \bigg ( K _ { C , 1 } h _ { W } ( X _ { t } , D _ { t } ^ { \overline { { \pi } } ^ { * } } ) + \frac { K _ { C , 2 } } { N } \bigg ) } \\ & { \qquad \leq K _ { V h } \cdot \big ( ( 1 - \rho _ { u } ) h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) } \\ & { \qquad + ( 1 - \rho _ { w } ) h _ { W } ( X _ { t } , D _ { t } ^ { \overline { { \pi } } ^ { * } } ) \big ) + \frac { L _ { 1 } K _ { C , 2 } } { N } , } \end{array}
$$

where we choose $K _ { V h } \ > \ 1$ , independent of $N ,$ so that $K _ { V h } ( 1 - \rho _ { u } ) \geq 1$ and $K _ { V h } ( 1 - \rho _ { w } ) \geq 1 + L _ { 1 } K _ { C , 1 }$ . We set $\rho _ { 1 } = 1 - 1 / K _ { V h } \in ( 0 , 1 )$ . Multiplying the preceding bound by $1 - \rho _ { 1 } = 1 / K _ { V h }$ gives

$$
\begin{array} { r l } & { ( 1 - \rho _ { 1 } ) V _ { 1 } ( \Sigma _ { t } ) } \\ & { \quad \leq ( 1 - \rho _ { u } ) h _ { U } ( X _ { t } , D _ { t } ^ { \mathrm { L P } } ) + ( 1 - \rho _ { w } ) h _ { W } ( X _ { t } , D _ { t } ^ { \bar { \pi } ^ { \ast } } ) } \\ & { \quad ~ + \frac { ( 1 - \rho _ { 1 } ) L _ { 1 } K _ { C , 2 } } { N } . } \end{array}\tag{75}
$$

Adding (75) to (73)–(74) proves (62)–(63) with

$$
K _ { \mathrm { d r i f t } } \triangleq 2 \lambda _ { W } ^ { 1 / 2 } + ( 1 - \rho _ { 1 } ) L _ { 1 } K _ { C , 2 } .
$$