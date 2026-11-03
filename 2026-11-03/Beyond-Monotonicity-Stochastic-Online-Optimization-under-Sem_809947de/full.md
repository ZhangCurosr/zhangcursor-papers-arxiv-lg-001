# Beyond Monotonicity: Stochastic Online Optimization under Semi-Bandit Feedback

Tsubasa Harada Institute of Science Tokyo

Yasushi Kawase Chuo University

## Abstract

Hanna Sumita Institute of Science Tokyo

We study stochastic online optimization (SOO) under semi-bandit feedback, in which a policy adaptively observes n independent random variables with unknown distributions in each of T rounds, and only the observed values are available for learning. Monotonicity with respect to first-order stochastic dominance enables general reductions for learning SOO, but it fails in natural problems where observations change future feasibility, resource use, or reference points. We provide an algorithm that runs the dynamic program for the optimal policy with each expectation replaced by its most optimistic value over a confidence set of distributions. Its regret is governed by the level-set complexity $s _ { i }$ of the values computed by this dynamic program as functions of the ith variable, that is, the number of intervals needed to describe their superlevel sets. For rewards in [0, 1], the regret is $\widetilde { O } ( \sqrt { S T } )$ without this monotonicity, where $S = \textstyle \sum _ { i = 1 } ^ { n }$ min $\{ s _ { i } , k _ { i } \}$ and $k _ { i }$ is the support size of the ith distribution, which may be infinite when $s _ { i }$ is finite. In particular, when the level-set complexities are bounded by a constant, the regret is $\widetilde { O } ( \sqrt { n T } )$ , even for continuous distributions. The algorithm requires no knowledge of $s _ { i } ,$ $k _ { i } ,$ , or the supports, and it also achieves regret $\widetilde { O } ( \sqrt { n k T } )$ for supports of size at most k without any level-set condition, which improves the previous bound $\widetilde { O } ( n ( \sqrt { k T } + k ) )$ for FSD-monotone problems. We also prove a lower bound of $\Omega ( \operatorname* { m i n } \{ T , \sqrt { S T } \} )$ for all positive integers $s _ { 1 } , \ldots , s _ { n }$ with $k _ { i } = 2 s _ { i } + 1$ , which holds even under full feedback and matches the upper bound up to logarithmic factors. As applications, we obtain regret $\widetilde { O } ( \sqrt { n T } )$ for the following problems, none of which is monotone under stochastic dominance: stochastic Boolean function evaluation, the full-information secretary problem, optimal stopping with loss aversion, scheduling with deadlines and correlated stochastic knapsack with unimodal rewards, and overbooking with group reservations.

Keywords: stochastic online optimization, semi-bandit feedback, regret, level sets

## 1 Introduction

Stochastic online optimization (SOO) is a framework for sequential decision-making under uncertainty. The input consists of n independent random variables, and a policy adaptively chooses the next action based on the values observed so far. This framework includes optimal stopping problems such as Pandora’s box [Wei79] and prophet inequalities [SC84, KW12], as well as stochastic probing [GN13], stochastic submodular optimization [AN16], and other stochastic combinatorial optimization problems.

Ghuge, Zhuo, Agarwal, and Nagarajan [GZAN26] (a preliminary version appeared at FOCS 2024) introduced a general framework for learning SOO when the distributions are unknown, in which the same instance is played for T rounds. In each round, the learner chooses a policy and receives semi-bandit feedback, which consists of the values of the variables observed during the execution of the policy. The goal is to minimize the regret, which is the cumulative loss in expected reward compared with the optimal policy that knows the distributions. Their framework assumes monotonicity with respect to first-order stochastic dominance (FSD), called FSD monotonicity, which means that replacing each distribution by a stochastically dominating one never decreases (or never increases) the optimal expected reward. Under this assumption, running an algorithm for known distributions on optimistic distributions that dominate the true ones (or are dominated by them in the latter case) achieves regret $\widetilde { O } ( n ( \sqrt { k T } + k ) )$ ) for rewards in [0, 1] and supports of size at most k [GZAN26], where $\widetilde O$ suppresses logarithmic factors. They also extend their framework to continuous distributions under additional assumptions on the threshold structure of policies and the regularity of the reward functions.

These results leave the following question. Can the regret grow as $\sqrt { T }$ without FSD monotonicity, with a polynomial dependence on n that does not involve the number of policies or observation histories, even for continuous distributions? Treating each deterministic policy as an arm already yields $\sqrt { T }$ dependence when there are finitely many policies, but the regret then grows with their number. We answer this question afirmatively, and our results extend the framework of Ghuge, Zhuo, Agarwal, and Nagarajan beyond FSD monotonicity and improve their regret bound.

FSD monotonicity fails in natural problems, and Section 4 studies five of them. In stochastic Boolean function evaluation (SBFE) [DHK14], binary variables are tested at a cost until a known Boolean function is determined, and the expected cost of evaluating the majority function is largest when successes and failures are balanced. In the full-information secretary problem [GM66, EHLM20], a stochastically larger later value makes accepting an earlier candidate less likely to succeed but waiting more likely to succeed. In optimal stopping with loss aversion [KKO22], a large early value raises the reference point and lowers the utility of later candidates. In scheduling with deadlines [Dea05] and its special case, correlated stochastic knapsack [GKMR11], a larger processing time or size can increase the reward of the current job or item but leaves less room for the others. In overbooking with group reservations [SSL99], more arrivals reduce refunds but increase the cost of exceeding capacity.

Under FSD monotonicity, optimistic distributions can be constructed by shifting probability mass in the empirical distributions upward or downward according to the direction of monotonicity. Without this property, the optimistic distribution can depend on the observation history and need not be confined to a single direction. The dependence on history already appears in SBFE for the majority of three variables (Figure 1): optimism favors increasing the success probability of the next test after observing 1, but decreasing it after observing 0, since agreement determines the majority. The DP for an optimal policy makes this dependence explicit through expectations of optimal continuation values, the maximum expected rewards obtainable from each decision point given the observations so far. Replacing each expectation by its supremum over the corresponding confidence set yields optimistic continuation values by backward induction, provided the confidence sets contain the true distributions.

The main challenge is to control the estimation errors in the expectations of these datadependent values uniformly over all observation histories, with no dependence on the number of histories. Uniform control over all bounded functions introduces a dependence on the support size and can be vacuous for continuous distributions, so the analysis must exploit the structure of these values. Moreover, summing variance-insensitive bounds on the local expectation errors yields a regret bound linear in n, even when the continuation values are monotone in the newly observed variable. We address these issues through the level-set structure of the optimistic continuation values and a variance-sensitive analysis of the accumulated expectation errors.

## 1.1 Our Contributions

We refer to the ith random variable as coordinate i. For a function of one observed coordinate, its level-set complexity is the smallest integer s such that every superlevel set is the intersection of the coordinate’s domain with a union of at most s intervals. Our algorithm replaces each expectation in the DP by its supremum over a confidence set of distributions, i.e., optimistic continuation values. Our main assumption, which we call the level-set condition, bounds by $s _ { i }$ the level-set complexity of the optimistic continuation values after coordinate i is revealed, for any possible confidence sets (Assumption 3.1).

Regret Upper Bound. With rewards in [0, 1], let $k _ { i }$ be the number of support points of the distribution $D _ { i }$ of coordinate i, which may be infinite, and let $S = \textstyle \sum _ { i = 1 } ^ { n }$ min $\{ s _ { i } , k _ { i } \}$ . The algorithm’s regret satisfies $R _ { T } = \widetilde { \cal O } ( \sqrt { S T } )$ against the optimal policy (Theorem 3.2). This guarantee does not require FSD monotonicity or finite supports, and the algorithm requires no knowledge of $s _ { i }$ or $k _ { i }$

In particular, when the level-set complexities are bounded by a constant, the regret is $\widetilde { O } ( \sqrt { n T } )$ , even for continuous distributions and without FSD monotonicity. If each distribution has at most k support points, the regret is $\widetilde { O } ( \sqrt { n k T } )$ even when the supports are unknown and without any condition on the level-set complexity (Corollary 3.3). In the setting of Ghuge, Zhuo, Agarwal, and Nagarajan, this improves their bound $\widetilde { O } ( n ( \sqrt { k T } + k ) )$ without requiring FSD monotonicity. For FSD-monotone problems with finite domains, as in their framework, it also sufices to run an exact or α-approximation algorithm for known distributions once per round on a single product distribution chosen from the confidence sets (Proposition 3.4), with α-regret $\widetilde { O } ( \sqrt { n k T } )$ for supports of size at most k. If the problem is realization-monotone, that is, all rewards and all sets of feasible actions are monotone in the observed values in a common direction, then $s _ { i } = 1$ (Section 3.2), and the regret is $\widetilde { O } ( \sqrt { n T } )$ . This applies to the prophet inequality, Pandora’s box, and stochastic knapsack with fixed rewards after rescaling the rewards to [0, 1], and for the first two with finite domains, each round runs in polynomial time by Proposition 3.4. The bound $\widetilde { O } ( \sqrt { n T } )$ does not depend on the support sizes, whereas the finite-support bound in [GZAN26] grows with k. Table 1 compares our general bounds with previous ones.

Regret Lower Bound. We complement our upper bounds with matching lower bounds, which show that their dependence on the level-set complexities and the support sizes is necessary. For any integers $T , s _ { 1 } , \ldots , s _ { n } \geq 1$ , we construct a problem satisfying Assumption 3.1 with these $s _ { i }$ such that, for every learning algorithm, some product distribution with $2 s _ { i } + 1$ support points for coordinate i forces regret Ω(min $\{ T , { \sqrt { S T } } \} )$ , even under full feedback (Theorem 3.6). Here, full feedback means that all n coordinates are observed in each round. Hence, the worst-case regret over the problems with given $s _ { 1 } , \ldots , s _ { n }$ is $\widetilde { \Theta } ( \operatorname* { m i n } \{ T , \sqrt { S T } \} )$ with $S = \textstyle \sum _ { i } s _ { i }$ . Since coordinate i of this instance has $2 s _ { i } + 1$ support points, the dependence on the support sizes in Corollary 3.3 cannot be improved in the worst case either. For realization-monotone problems, our bound $\widetilde { O } ( \sqrt { n T } )$ is also tight up to logarithmic factors, since Ghuge, Zhuo, Agarwal, and Nagarajan [GZAN26] observed a lower bound of $\Omega ( { \sqrt { n T } } )$ under semi-bandit feedback by a reduction from multi-armed bandits, and their instance with binary coordinates is realization-monotone.

Applications. For the five applications, we verify the level-set condition directly for SBFE, where $s _ { i } = 1$ since the variables are binary, and for the full-information secretary problem and optimal stopping with loss aversion, where $s _ { i } = 2$ . Scheduling with deadlines, including correlated stochastic knapsack, and overbooking with group reservations require an extension, since a sum of functions with small level-set complexity need not have small complexity itself. Theorem 3.7 handles continuation values decomposed into an immediate reward collected when a coordinate is revealed and a residual reward collected later. It assumes that the immediate rewards have bounded level-set complexity, are nonnegative, and sum to at most 1 in each round, and that larger revealed values cannot increase the residual rewards or enlarge future sets of feasible actions. Together, the two theorems yield regret $\widetilde { O } ( \sqrt { n T } )$ for all five applications after rescaling the rewards to [0, 1], provided that, in scheduling, the reward functions have bounded level-set complexity and the observed processing time and reward identify the outcome. For the full-information secretary problem, the bound holds for every known distribution over arrival orders.

A refinement of Theorem 3.2 yields $\widetilde { O } ( \sqrt { S T p ( 1 - p ) } + S )$ , where p is the optimal value (Proposition 3.5), and Proposition 3.8 provides an analogous refinement of Theorem 3.7. These refinements yield sharper regret bounds for all five applications in their original units.

Computation Time. For computation, we reduce each optimistic update to a linear program (LP) over the masses of the sample points and the intervals between them, which is the dual of a minimum-cost flow problem. Its coeficients are the suprema of the optimistic continuation value on these parts. Given these suprema, each round runs in time polynomial in $n , t ,$ and the size of the decision tree (Proposition A.3), which can be exponential in $n ,$ and the suprema themselves can be hard to compute in general. Each supremum reduces to evaluations of the optimistic continuation value at finitely many points when the domains are finite, or when the domains are [0, 1] and the values are continuous and either quasi-convex or monotone on each of finitely many known intervals. When the values are Lipschitz, approximate evaluations on a grid increase the regret by at most 1 and take time polynomial in $n , T$ , and the Lipschitz constant. As in a DP for known distributions, these evaluations can be organized by states when the rewards and the sets of feasible actions depend on the observations only through a state, apart from rewards collected along the $\mathrm { w a y } ,$ and each round then runs in time polynomial in $n , t ,$ the size of the tree, and the number of pairs of a node and a state reachable through the evaluation points (Proposition $\mathrm { A . 4 } )$ . For overbooking with group reservations, this yields time polynomial in n, t, the number of allowed sets of accepted groups, and the total group size (Appendix A.3). Each round also runs in polynomial time for optimal stopping with loss aversion when each domain is [0, 1] or finite (Appendix A.3) and for the full-information secretary problem with finite domains and a fixed arrival order (Section 4.2), and it runs in time $O ( n ^ { 2 } )$ for SBFE with threshold functions (Section 4.1).

Table 1: Regret bounds for learning SOO under semi-bandit feedback with rewards in [0, 1]. Here, $k _ { i }$ is the support size of $D _ { i } , k = \operatorname* { m a x } _ { i } k _ { i } ,$ and $\begin{array} { r } { S = \sum _ { i } \operatorname* { m i n } \{ s _ { i } , k _ { i } \} } \end{array}$ . We assume $T \geq S .$
<table><tr><td>Result</td><td>Condition</td><td>Supports</td><td>Regret</td></tr><tr><td>[GZAN26]</td><td>FSD monotonicity</td><td>known k points</td><td> $\widetilde { O } \big ( n ( \sqrt { k T } + k ) \big )$ </td></tr><tr><td>[GHTZ21, GZAN26]</td><td>none</td><td>known k points</td><td> $\widetilde { O } \big ( ( n k T ) ^ { 2 / 3 } \big )$ </td></tr><tr><td>[GZAN26]</td><td>FSD monotonicity</td><td>2 points</td><td> $\Omega ( { \sqrt { n T } } )$ </td></tr><tr><td>Theorem 3.2</td><td>level-set complexity  $s _ { i }$ </td><td>arbitrary</td><td> $\widetilde { O } ( \sqrt { S T } )$ </td></tr><tr><td>Theorem 3.2</td><td>realization monotonicity</td><td>arbitrary</td><td> $\widetilde { O } ( \sqrt { n T } )$ </td></tr><tr><td>Corollary 3.3</td><td>none</td><td>unknown k points</td><td> $\widetilde { O } ( \sqrt { n k T } )$ </td></tr><tr><td>Theorem 3.6</td><td>level-set complexity  $s _ { i }$ </td><td> $2 s _ { i } + 1 \ \mathrm { p o i n t s }$ </td><td> $\Omega ( { \sqrt { S T } } )$ </td></tr></table>

## 1.2 Technical Overview

Our algorithm follows the principle of optimism in the face of uncertainty. In our formulation (Section 2), the decision points are the nodes of a finite tree whose edges either reveal a coordinate or stop, and the observed values are carried along the path. For each coordinate, we construct from the samples collected so far a confidence set that contains the true distribution with high probability, and we replace each expectation in the DP by its supremum over this set. If the true distributions belong to the confidence sets, backward induction shows that the resulting values are at least the true optimal values. Each DP update takes the supremum separately, and diferent nodes and observations may use diferent distributions from the same confidence set.

The main technical step translates this local optimism into a global regret bound. The confidence sets impose Bernstein-type constraints on the probabilities of intervals. For two distributions $P , Q$ in the same confidence set built from $m \geq 1$ samples and a [0, 1]-valued function $\phi$ with level-set complexity at most $s ,$ Lemma 5.4(b) yields

$$
\begin{array} { r } { | \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] | \le \widetilde O \left( \sqrt { s / m } \sqrt { \mathrm { V a r } _ { X \sim P } ( \phi ( X ) ) } + s / m \right) . } \end{array}
$$

We prove this bound by integrating the probability errors over the superlevel sets of $\phi ,$ each of which is a union of at most s intervals. A variance-sensitive inequality for this integral yields the square-root dependence on $\operatorname { V a r } _ { X \sim P } ( \phi ( X ) )$ (Lemma 5.3).

Fix the history before round t and suppose that the confidence sets contain the true distributions. Let $Y _ { t , h }$ be the optimistic continuation value at the node visited at stage $h ,$ and let $d _ { t , h } = Y _ { t , h }$ $\mathbb { E } [ Y _ { t , h + 1 } \ | \ { \mathcal { F } } _ { t , h } ]$ and $w _ { t , h } = \operatorname { V a r } ( Y _ { t , h + 1 } \mid { \mathcal { F } } _ { t , h } )$ , where $\mathcal { F } _ { t , h }$ is the history through stage h. Since $\mathbb { E } [ Y _ { t , 0 } ]$ is at least the optimal value and the last $Y _ { t , h }$ is the reward, the expected regret in this round is at most $\mathbb { E } [ \sum _ { h } d _ { t , h } ]$ . Let $N _ { i , t }$ be the number of rounds before round t in which coordinate i was revealed, and let $I _ { i , t }$ indicate whether coordinate i is revealed on an edge in round t. Up to logarithmic factors, the proof combines three bounds:

$$
d _ { t , h } \lesssim \sqrt { \frac { s _ { i } w _ { t , h } } { N _ { i , t } + 1 } } + \frac { s _ { i } } { N _ { i , t } + 1 } , \qquad \mathbb { E } \biggl [ \sum _ { h } w _ { t , h } \biggr ] \leq 1 , \qquad \sum _ { t = 1 } ^ { T } \frac { I _ { i , t } } { N _ { i , t } + 1 } \leq 1 + \log T .
$$

In the first bound, i is the coordinate revealed at stage h. The first bound follows from the expectation bound above, and it is the only place where level-set complexity is used. If $D _ { i }$ has $k _ { i } < s _ { i }$ support points, a one-sided version of the expectation bound (Lemma 5.5) yields the first bound with $k _ { i }$ in place of $s _ { i } ,$ and hence the regret depends on $\begin{array} { r } { S = \sum _ { i } \operatorname* { m i n } \{ s _ { i } , k _ { i } \} } \end{array}$ . The second bound holds because $( Y _ { t , h } ) _ { h }$ is a supermartingale with values in [0, 1], and telescoping the squares $Y _ { t , h } ^ { 2 }$ bounds the sum of the conditional variances. The third bound holds because $N _ { i , t }$ increases by one in each round in which coordinate i is revealed. The Cauchy–Schwarz inequality then yields the leading term $\sqrt { S T }$

For the lower bound, we construct an instance with S unknown signs $\theta _ { i j }$ , where $j \in [ s _ { i } ]$ for coordinate i. Each coordinate i equals a special value ⊥ with probability $1 - s _ { i } / ( 2 S )$ . Otherwise, it takes a value $( j , y )$ , where the index $j \in [ s _ { i } ]$ is uniform and the sign $y \in \{ - 1 , + 1 \}$ equals $\theta _ { i j }$ with probability $( 1 + \varepsilon ) / 2$ . Before receiving observations in a round, the learner chooses a prediction $\widehat { y } _ { i j }$ of the observed sign y for each possible pair $( i , j )$ . The reward is 1 if exactly one coordinate difers from ⊥ and its sign is predicted correctly, and 0 otherwise. Each prediction $\widehat { y } _ { i j }$ that difers from $\theta _ { i j }$ reduces the expected reward by $\Theta ( \varepsilon / S )$ relative to predicting $\theta _ { i j }$

Two environments difering in one sign have one-sample Kullback–Leibler (KL) divergence $O ( \varepsilon ^ { 2 } / S )$ . Choosing $\varepsilon = \Theta ( \operatorname* { m i n } \{ 1 , \sqrt { S / T } \} )$ with a suficiently small constant keeps the probability that $\widehat { y } _ { i j } \neq \theta _ { i j }$ bounded below by a constant in every round, averaged over a uniformly random sign vector θ. Summing the resulting losses over all S signs and $T$ rounds yields regret $\Omega ( \operatorname* { m i n } \{ T , \sqrt { S T } \} )$ for some fixed sign vector. In the SOO instance, the policy chooses the prediction vector at the root and then observes all coordinates in order, and we order the $2 s _ { i } + 1$ values of coordinate i such that every optimistic continuation value has level-set complexity at most $s _ { i }$

## 1.3 Related Work

For known distributions, prophet inequalities provide guarantees against the expected ofline optimum for a single selection [KS77], multiple selections [Ala14], and matroid constraints [KW12]. Approximation algorithms are also known for many stochastic combinatorial optimization problems [DGV08, Ma18, AN16, GN13].

In combinatorial bandits [CWY13], the learner selects a set of arms and observes each of them, and algorithms based on upper confidence bounds achieve $\widetilde { O } ( \sqrt { m n T } )$ regret for linear rewards when at most m arms are selected [KWAS15]. Extensions allow rewards that depend on the whole distributions of the arms [CHL<sup>+</sup>16] and additional arms that are revealed depending on the outcomes [CWYW16]. In our setting, the policy also adapts its actions to the values observed within a round.

SOO can also be cast as an episodic Markov decision process (MDP) whose states are observation histories. Our algorithm resembles optimistic planning over a set of plausible transition models, as in extended value iteration [JOA10], and our variance analysis resembles the use of the law of total variance in the upper confidence bound value iteration algorithm [AOM17]. However, regret bounds for tabular MDPs depend on the numbers of states and actions, and observation histories can be exponentially many in n, or infinitely many for continuous observations. Factored MDPs [TQS20] exploit a product structure of the transitions, but their bounds still depend on the sizes of the factored state spaces. In SOO, all histories share the same coordinate distributions, and the level-set condition on the optimistic continuation values removes the dependence on the numbers of histories and support points.

Other work on learning in SOO considers specific problems. Gatmiry, Kesselheim, Singla, and Wang [GKSW24] proved an $\widetilde { \cal O } ( n ^ { 3 } \sqrt { T } )$ upper bound and an $\Omega ( { \sqrt { T } } )$ lower bound for the prophet inequality under bandit feedback. Liu, Chen, Wang, Luo, and Ratlif $[ \mathrm { L C W ^ { + } 2 5 } ]$ obtained $\widetilde { O } ( \sqrt { n T } )$ for the prophet inequality and Pandora’s box under semi-bandit feedback by using problem-specific structure. For Pandora’s box with inspection costs $c _ { i }$ and values in [0, 1], their bound avoids the factor $1 + \textstyle \sum _ { i } c _ { i }$ introduced by normalizing the net reward in our general bound.

A related line of work studies the sample complexity of stochastic optimization, with bounds for single-parameter revenue maximization [GHZ19], the prophet inequality [CDFS19], and general problems over product distributions [GHTZ21]. For the last class, Guo, Huang, Tang, and Zhang [GHTZ21] show that $\widetilde { O } ( n k / \varepsilon ^ { 2 } )$ samples sufice for supports of size at most k. Under a stronger monotonicity condition, $\widetilde { O } ( n / \varepsilon ^ { 2 } )$ samples sufice, and both bounds are optimal up to logarithmic factors. Under full feedback, such bounds lead to regret with $\sqrt { T }$ dependence, whereas under semi-bandit feedback, an explore-then-commit approach typically leads to $T ^ { 2 / 3 }$ dependence [GZAN26].

In concurrent and independent work that appeared on arXiv shortly before this submission, Zhuo and Nagarajan [ZN26] and Hu and Ghuge [HG26] improved the regret bounds for FSDmonotone problems.

## 2 Preliminaries

## 2.1 Model

Let $\mathbb { Z } _ { > 0 }$ be the set of positive integers. For $m \in \mathbb { Z } _ { > 0 }$ , write $[ m ] = \{ 1 , \dots , m \}$ , and for $a \in \mathbb { R }$ , write $a _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . Let $X _ { 1 } , \ldots , X _ { n }$ be independent random variables with unknown distributions $D _ { 1 } , \ldots , D _ { n }$ , and write $\begin{array} { r } { D = \prod _ { i = 1 } ^ { n } D _ { i } } \end{array}$ . Each $X _ { i }$ takes values in a known subset of [0, 1], called its domain (e.g., {0, 1} or [0, 1]). Our algorithm and analysis use only the order of the values, and hence real-valued coordinates can be mapped into [0, 1] by a known strictly increasing function. We refer to $X _ { i }$ as coordinate i and say that coordinate i is revealed when its realized value is observed.

A problem is specified by a finite rooted directed tree $( V , A )$ , called the decision tree,<sup>1</sup> with root r and children ch(u) at node u, together with the observation sets, feasible actions, and leaf rewards defined below. The set obs $( u ) \subseteq [ n ]$ contains the coordinates already observed upon reaching $u ,$ and $x _ { \mathrm { o b s } ( u ) } = ( x _ { i } ) _ { i \in \mathrm { o b s } ( u ) }$ denotes their values. The coordinates in obs(r) are observed before the first action. Each edge $( u , v )$ is either a revelation edge, which reveals one new coordinate i, with ob $\mathsf { s } ( v ) = \mathrm { o b s } ( u ) \cup \{ i \}$ and $i \not \in \mathrm { o b s } ( u )$ , or a stopping edge, which leads to a leaf v without a new observation, with ob ${ \mathfrak { s } } ( v ) = \mathrm { o b s } ( u )$ . Thus, a coordinate is observed at most once per round. We remark that an action that reveals several coordinates at once can be represented by a chain of revelation edges with no intermediate choices. A revelation edge may also lead to a leaf.

We regard each child of a node as an action at that node, and at a non-leaf node u with observations $x _ { \mathrm { o b s } ( u ) }$ , the feasible actions form a nonempty set $\mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } ) \subseteq \mathrm { c h } ( u )$ . A leaf ℓ has reward $f _ { \ell } ( x _ { \mathrm { o b s } ( \ell ) } ) \stackrel { \cdot } { \in } [ 0 , 1 ]$ . The tree encodes only the actions and the order of observations, and the observed values are carried along as $x _ { \mathrm { o b s } ( u ) }$ . In the example of Figure 1, a single node $u _ { 1 2 }$ serves all pairs $( x _ { 1 } , x _ { 2 } )$ , and its feasible actions depend on whether $x _ { 1 } = x _ { 2 }$ . Reward functions and functions specifying feasibility are assumed to be measurable.

![](images/d40322e12485f47f90c88fc2f1e490d6a09f69261bd2a016001f8f17f322a34b.jpg)  
Figure 1: SBFE (Section 4.1) for the majority function of three variables, $F ( x ) = \mathbf { 1 } \{ x _ { 1 } + x _ { 2 } + x _ { 3 } \geq$ 2}, with unit testing costs. Here, $\mathrm { o b s } ( r ) = \emptyset , \mathrm { o b s } ( u _ { i } ) = \{ i \}$ , and obs $( u _ { i j } ) = \{ i , j \}$ . At $u _ { i j }$ , the dashed stopping edge is feasible if $x _ { i } = x _ { j }$ , and the solid edge revealing the remaining coordinate is feasible if $x _ { i } \neq x _ { j }$ . Each leaf shows its reward, 1 minus the number of tests divided by 3.

A deterministic policy $\sigma$ chooses a feasible action $\sigma _ { u } ( x _ { \mathrm { o b s } ( u ) } ) \in \mathcal A _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ at each non-leaf node u. Deterministic policies are assumed to be measurable, and a randomized policy is a distribution over them. For a policy $\sigma$ and a realization x, write $f ( \sigma , x )$ for the reward, averaged over the randomness of $\sigma ,$ and let

$$
f ( \sigma \mid D ) = \mathbb { E } _ { X \sim D } [ f ( \sigma , X ) ] , \qquad { \mathrm { O P T } } ( D ) = \operatorname* { s u p } _ { \sigma } f ( \sigma \mid D ) .
$$

The learner faces the same problem in each of T rounds. Before the first round, the learner knows the problem and $T _ { i }$ but not $D .$ . In each round $t \in \left\lceil T \right\rceil$ , a vector $X ^ { t } \sim D$ is drawn independently of the previous rounds. The learner chooses a policy $\sigma ^ { t }$ based on the observations from the previous rounds and possibly internal randomness, and $\sigma ^ { t }$ adapts to the observations within round t. If $\ell ^ { t }$ is the leaf reached, the learner observes $X _ { \mathrm { o b s } ( \ell ^ { t } ) } ^ { t }$ , which we call semi-bandit feedback. Let $N _ { i , t }$ be the number of rounds before round t in which coordinate i was revealed, and let $Z _ { i , 1 } , \dots , Z _ { i , N _ { i , t } }$ be the revealed values in order.

The expected regret of the policy $\sigma ^ { t }$ chosen in round t is $\mathrm { R e g } _ { t } = \mathrm { O P T } ( D ) - f ( \sigma ^ { t } \mid D )$ . Taking expectation over the past observations and the learner’s randomness, we define the cumulative expected regret as

$$
R _ { T } = \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathrm { R e g } _ { t } ] = \sum _ { t = 1 } ^ { T } \bigl ( \mathrm { O P T } ( D ) - \mathbb { E } [ f ( \sigma ^ { t } , X ^ { t } ) ] \bigr ) .
$$

We write $R _ { T } ( D )$ when the dependence on $D$ matters. The benchmark $\mathrm { O P T } ( D )$ is attained by a policy that knows D but, like the learner, observes the coordinates only along its execution path. For $\alpha \in ( 0 , 1 ]$ , we also define the cumulative expected α-regret as

$$
R _ { T } ^ { ( \alpha ) } = \sum _ { t = 1 } ^ { T } \bigl ( \alpha \operatorname { O P T } ( D ) - \mathbb { E } [ f ( \sigma ^ { t } , X ^ { t } ) ] \bigr ) .
$$

We define functions $g _ { u }$ and $h _ { ( u , v ) }$ by backward induction. At a leaf, $g _ { \ell } = f _ { \ell }$ . At a non-leaf node,

$$
\begin{array} { r } { g _ { u } ( x _ { \mathrm { o b s } ( u ) } ) = \operatorname* { m a x } _ { v \in A _ { u } ( x _ { \mathrm { o b s } ( u ) } ) } h _ { ( u , v ) } ( x _ { \mathrm { o b s } ( u ) } ) , } \end{array}\tag{2.1}
$$

$$
h _ { ( u , v ) } ( x _ { \mathrm { o b s } ( u ) } ) = \left\{ \begin{array} { l l } { \mathbb { E } _ { X _ { i } \sim D _ { i } } [ g _ { v } ( X _ { i } , x _ { \mathrm { o b s } ( u ) } ) ] , } & { \mathrm { i f ~ } \mathrm { o b s } ( v ) \backslash \mathrm { o b s } ( u ) = \{ i \} , } \\ { f _ { v } ( x _ { \mathrm { o b s } ( u ) } ) , } & { \mathrm { i f ~ } \mathrm { o b s } ( v ) = \mathrm { o b s } ( u ) . } \end{array} \right.\tag{2.2}
$$

Here, $( x _ { i } , x _ { \mathrm { o b s } ( u ) } )$ adds the value of coordinate i to the observations, with coordinates identified by their indices. We call $g _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ the optimal continuation value at u. It is the maximum expected reward obtainable after reaching u with observations $x _ { \mathrm { o b s } ( u ) }$ , and $h _ { ( u , v ) } ( x _ { \mathrm { o b s } ( u ) } )$ is the corresponding maximum when the next move is to v. By backward induction, choosing a maximizing child at every node is optimal, and

$$
\begin{array} { r } { \mathrm { O P T } ( D ) = \mathbb { E } _ { X _ { \mathrm { o b s } ( r ) } \sim \prod _ { i \in \mathrm { o b s } ( r ) } D _ { i } } \big [ g _ { r } \big ( X _ { \mathrm { o b s } ( r ) } \big ) \big ] . } \end{array}\tag{2.3}
$$

We now relate our model to that of Ghuge, Zhuo, Agarwal, and Nagarajan [GZAN26]. They describe a policy as a tree whose internal nodes are labeled by the coordinates to observe next and whose branches correspond to the observed values. When the constraints on policies are imposed on each path, such a problem fits our model, with the nodes of $( V , A )$ corresponding to the sequences of observed coordinates. For continuous distributions, they consider policies with finitely many threshold branches at each node, since an arbitrary policy can have uncountably many branches. In our applications, an optimal policy can depend on the exact observed values, for example, through the maximum of the past values in optimal stopping with loss aversion, and it may then need uncountably many branches in their model. Our model carries the observed values along a finite tree, and a policy chooses a child at each node by a measurable function of these values. Carrying the values also makes each continuation value, such as $g _ { v } ( \cdot , x _ { \mathrm { o b s } ( u ) } )$ , a function of the coordinate revealed on the edge $( u , v )$ , and our regret bounds depend on the level-set complexity of such functions (Section 3).

## 2.2 Monotonicity and Level-Set Complexity

For distributions $P , Q$ , we say that P first-order stochastically dominates (FSD) Q if $P ( [ 0 , x ] ) \leq$ $Q ( [ 0 , x ] )$ for all $x \in [ 0 , 1 ]$ A problem is FSD-monotone [GZAN26] if either $\mathrm { O P T } ( D ) \leq \mathrm { O P T } ( E )$ for all product distributions $\begin{array} { r } { D = \prod _ { i } D _ { i } } \end{array}$ and $\begin{array} { r } { E = \prod _ { i } E _ { i } } \end{array}$ such that $E _ { i }$ dominates $D _ { i }$ for every $i ,$ or $\mathrm { O P T } ( D ) \geq \mathrm { O P T } ( E )$ for all such D and E. A problem is realization-monotone if the leaf rewards and the sets of feasible actions are all nondecreasing in every observed value, or all nonincreasing in every observed value, with the other values fixed and sets ordered by inclusion. If, for each coordinate i, the leaf rewards and the sets of feasible actions are monotone in $x _ { i }$ in a direction that depends on i, reversing the order of the values of the coordinates with the nonincreasing direction, which afects neither the algorithm nor level-set complexity, makes the problem realization-monotone. Realization monotonicity implies FSD monotonicity, since expectations and maxima over sets of feasible actions that are monotone in the same direction preserve monotonicity, and hence backward induction shows that every continuation value is monotone in each observed value and that the optimal value is monotone in each distribution. The converse fails, since FSD monotonicity concerns only the optimal value. For example, if the root has a stopping edge to a leaf with reward 1 and an edge revealing $X _ { 1 }$ to a leaf with reward $\psi ( x _ { 1 } ) \in [ 0 , 1 ]$ , the optimal value is always 1, whereas ψ need not be monotone and can have any level-set complexity.

For an integer $s \geq 1$ , let $\mathcal { T } _ { s }$ be the family of unions of at most s intervals in $[ 0 , 1 ]$ , including the empty set. Intervals may be open, closed, half-open, or singletons. A function ϕ on a subset of [0, 1] has level-set complexity at most s if, for every $y \in \mathbb { R }$ , there exists $B \in \mathcal { I } _ { s }$ such that the superlevel set of $\phi$ at y satisfies

$$
\{ z \in \mathrm { d o m } ( \phi ) : \phi ( z ) > y \} = \mathrm { d o m } ( \phi ) \cap B .
$$

By convention, every function has level-set complexity at most $\infty$

A function has level-set complexity at most 1 if it is monotone or unimodal, that is, nondecreasing and then nonincreasing. A function has level-set complexity at most 2 if it is nonincreasing and then nondecreasing, for example, the maximum of a nondecreasing function and a nonincreasing function, since each superlevel set is then the intersection of the domain with the complement of an interval. When the domain is an interval, the same bounds hold for quasi-concave and quasi-convex functions, respectively. A function on a domain of at most k points has level-set complexity at most $\lceil k / 2 \rceil$ , since each superlevel set consists of at most that many runs of consecutive domain points.

## 3 Main Results

We present an optimistic DP algorithm and state regret bounds in terms of level-set complexity, together with a worst-case lower bound that matches the main upper bound up to logarithmic factors. We then extend the upper bound to problems whose reward decomposes into the reward earned so far and the future reward, such as scheduling and overbooking.

## 3.1 Optimistic Dynamic Programming

At the beginning of each round, the algorithm forms a confidence set of distributions for each coordinate from the samples collected so far. It then evaluates the DP defined by (2.1) and (2.2) with each expectation replaced by its supremum over the corresponding confidence set, and during the round it chooses, at each node, a child that maximizes the resulting optimistic value.

For each $m \geq 1 $ , write $\widehat { D } _ { i , m }$ for the empirical distribution of the first m revealed values $Z _ { i , 1 } , \ldots , Z _ { i , m }$ of coordinate $i ,$ that is, $\begin{array} { r } { \widehat { D } _ { i , m } ( B ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { 1 } \{ Z _ { i , j } \in B \} } \end{array}$ for $B \subseteq [ 0 , 1 ]$ . For $\delta \in ( 0 , 1 )$ , let

$$
\Lambda = \log \left( 8 0 n ( T + 1 ) ^ { 3 } / \delta \right) .
$$

For $m \geq 1$ and $z \in [ 0 , 1 ]$ , define

$$
b _ { m } ( z ) = \operatorname* { m i n } \left\{ 1 , 6 \sqrt { z ( 1 - z ) \Lambda / m } + 3 2 \Lambda / m \right\} .\tag{3.1}
$$

The confidence set for $m \geq 1$ observations of coordinate i is

$$
\mathcal { C } _ { m } ( \widehat { D } _ { i , m } ) = \left\{ Q : | Q ( I ) - \widehat { D } _ { i , m } ( I ) | \leq b _ { m } ( \widehat { D } _ { i , m } ( I ) ) \quad \forall I \in \mathbb { Z } _ { 1 } \right\} ,\tag{3.2}
$$

where $Q$ ranges over distributions on the coordinate’s domain. For $m = 0$ , let $\widehat { D } _ { i , 0 }$ be an arbitrary fixed distribution on the coordinate’s domain and let $\mathcal { C } _ { 0 } ( \widehat { D } _ { i , 0 } )$ be the set of all distributions on that domain. For $m \geq 1$ , the empirical distribution belongs to the confidence set, and hence the set is

always nonempty. For $m \leq 3 2 \Lambda$ , we have $b _ { m } \equiv 1$ , and the confidence set consists of all distributions on the coordinate’s domain.

At the beginning of round t, set $\mathcal { C } _ { i , t } = \mathcal { C } _ { N _ { i , t } } ( \widehat { D } _ { i , N _ { i , t } } )$ . These sets stay fixed throughout the round. Define $\bar { g } _ { \ell } ^ { t } = f _ { \ell }$ at leaves and

$$
\begin{array} { r } { \bar { g } _ { u } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) = \operatorname* { m a x } _ { v \in A _ { u } ( x _ { \mathrm { o b s } ( u ) } ) } \bar { h } _ { ( u , v ) } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) , } \end{array}\tag{3.3}
$$

$$
\begin{array} { r } { \bar { h } _ { ( u , v ) } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) = \left\{ \begin{array} { l l } { \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \mathbb { E } _ { X _ { i } \sim Q } [ \bar { g } _ { v } ^ { t } ( X _ { i } , x _ { \mathrm { o b s } ( u ) } ) ] , } & { \mathrm { i f ~ } \mathrm { o b s } ( v ) \setminus \mathrm { o b s } ( u ) = \{ i \} , } \\ { f _ { v } ( x _ { \mathrm { o b s } ( u ) } ) , } & { \mathrm { i f ~ } \mathrm { o b s } ( v ) = \mathrm { o b s } ( u ) . } \end{array} \right. } \end{array}\tag{3.4}
$$

We call $\bar { g } _ { u } ^ { t } ( x _ { \mathrm { o b s } ( u ) } )$ the optimistic continuation value at u. The suprema at diferent edges may be attained by diferent distributions, and no single product distribution needs to attain all optimistic values. Upon reaching node u, the algorithm chooses

$$
\begin{array} { r } { \sigma _ { u } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) \in \arg \operatorname* { m a x } _ { v \in \mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } ) } \bar { h } _ { ( u , v ) } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) , } \end{array}\tag{3.5}
$$

breaking ties in a fixed order. Appendix A shows that each supremum in (3.4) is the optimal value of an LP whose size is polynomial in the number of samples, given the suprema of $\bar { g } _ { v } ^ { t }$ over the cells formed by the sample points and the intervals between them. For infinite domains, we assume that the optimistic values are measurable in the samples and observations, which makes this policy measurable. This holds automatically for finite domains. Algorithm 1 summarizes the procedure.

Algorithm 1: Learning with optimistic dynamic programming   
Input: parameter $\overline { { \delta \in ( 0 , 1 ) } }$   
1 for $t = 1 , \dots , T$ do   
2 Set $\mathcal { C } _ { i , t } = \mathcal { C } _ { N _ { i , t } } ( \widehat { D } _ { i , N _ { i , t } } )$ for every coordinate i, as defined above;   
3 Define $\bar { g } ^ { t }$ and ${ \bar { h } } ^ { t }$ by (3.3) and (3.4);   
4 Observe $x _ { \mathrm { o b s } ( r ) }$ and set $u  r ;$   
5 while u is not a leaf do   
6 Choose a child v of u by (3.5);   
7 Observe any coordinate revealed by $( u , v )$ and set $u  v ;$   
8 Add the values revealed in round t to the samples;

## 3.2 Regret Bounds

The regret depends on how the optimistic continuation values vary with the coordinate revealed next. On an edge $( u , v )$ that reveals coordinate i, the optimistic value $\bar { h } _ { ( u , v ) } ^ { t }$ is the supremum of the expectation of $z \mapsto \bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } )$ over $\mathcal { C } _ { i , t }$ , as defined in (3.4). We bound the diference between this optimistic value and the expectation of the same continuation-value function under $D _ { i }$ . Since expectations can be expressed as integrals of superlevel-set probabilities, the interval constraints control this diference whenever $D _ { i } \in \mathcal { C } _ { i , t }$ and each superlevel set is a union of a few intervals. We assume the following bound on the level-set complexity of this function for any confidence sets.

Assumption 3.1. There are $s _ { 1 } , \ldots , s _ { n } \in \mathbb { Z } _ { > 0 } \cup \{ \infty \}$ with the following property. Let $\mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { n }$ be any nonempty sets of distributions on the domains of the coordinates, and let g¯ be defined by (3.3) and (3.4) with $\mathcal { P } _ { i }$ in place of $\mathcal { C } _ { i , t }$ . Then, for every edge $( u , v )$ revealing coordinate i and all observations $x _ { \mathrm { o b s } ( u ) }$ for which $v \in \mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ , the function $z \mapsto \bar { g } _ { v } ( z , x _ { \mathrm { o b s } ( u ) } )$ has level-set complexity at most $s _ { i }$

We can take $s _ { i } ~ = ~ 1$ for every coordinate if the problem is realization-monotone, since the backward induction in Section 2.2, which also applies to suprema, shows that every optimistic continuation value is then monotone. These parameters are small constants in many problems, including SBFE, the full-information secretary problem, and optimal stopping with loss aversion in Section 4. If the domain of coordinate i is finite, the assumption holds with $s _ { i }$ equal to half its size rounded up, which is 1 for binary coordinates. The assumption always holds with $s _ { i } = \infty$ , for which it imposes no condition on coordinate i.

The following theorem bounds the regret in terms of these complexities and the support sizes.

Theorem 3.2. Under Assumption 3.1, let $k _ { i } \in \mathbb { Z } _ { > 0 } \cup \{ \infty \}$ be the number of support points of $D _ { i }$ , and suppose that $S = \textstyle \sum _ { i = 1 } ^ { n }$ min $\{ s _ { i } , k _ { i } \}$ is finite. There is a universal constant c such that, for every $\delta \in ( 0 , 1 )$ , Algorithm 1 satisfies

$$
R _ { T } \leq c \big ( \sqrt { S T \Lambda \log T ( 1 + \log T ) } + S \Lambda ( 1 + \log T ) \big ) + \delta T .\tag{3.6}
$$

In particular, for $\delta = 1 / ( T + 1 )$ , we have $R _ { T } = \widetilde { O } ( \sqrt { S T } )$

The last statement holds because $\delta T < 1$ and $S \le \sqrt { S T }$ when $T \geq S$ , whereas $R _ { T } \leq T \leq \sqrt { S T }$ when $T < S$ . Algorithm 1 requires no knowledge of $s _ { i }$ or $k _ { i }$ , adapts to the smaller of the two for each coordinate, and uses T only through Λ. For unknown $T$ , the doubling trick, restarting the algorithm in phases of lengths $H = 2 ^ { j }$ for $j = 0 , 1 , . . .$ . with H in place of $T$ and $\delta = 1 / ( H + 1 )$ , yields $R _ { T } = \widetilde { \cal O } ( \sqrt { S T } )$ for every T.

Since Assumption 3.1 holds with $s _ { i } = \infty$ , Theorem 3.2 yields the following corollary, which bounds the regret in terms of the support sizes even when the supports are unknown.

Corollary 3.3. Suppose that each distribution $D _ { i }$ has at most $k _ { i }$ support points, and let $K =$ $k _ { 1 } + \cdots + k _ { n }$ . Then, for $\delta = 1 / ( T + 1 )$ , Algorithm 1 satisfies $R _ { T } = \widetilde { O } ( \sqrt { K T } )$ , without any condition on the level-set complexity.

If the supports were known, we could restrict the coordinate domains to them and apply Theorem 3.2 with $s _ { i } = \lceil k _ { i } / 2 \rceil$ . Corollary 3.3 requires neither the supports nor their sizes as input and allows infinite coordinate domains. Restricting the continuation-value functions to the true supports does not sufice, since distributions in the confidence sets may place mass outside these supports, and the proof of Theorem 3.2 accounts for this mass through Lemma 5.5.

The analysis also applies to algorithms that choose a single product distribution from the confidence sets, provided that its optimal value is at least the true one up to $1 / T$ . For a product distribution $\textstyle P = \prod _ { i } P _ { i }$ , let $g ^ { P }$ and $h ^ { P }$ be defined by (2.1) and (2.2) with $P _ { i }$ in place of $D _ { i }$ . A policy is optimal for P at every node if, at every non-leaf node and for all observations, it chooses a child that maximizes $h ^ { P }$

Proposition 3.4. Suppose obs $( r ) = \emptyset$ and let $\delta = 1 / ( T + 1 )$ . For each round t, let $\begin{array} { r } { Q ^ { t } = \prod _ { i } Q _ { i } ^ { t } } \end{array}$ be a product distribution determined by the observations before round t, with $Q _ { i } ^ { t } \in \mathcal { C } _ { i , t }$ for every i, and satisfying $\mathrm { O P T } ( Q ^ { t } ) \geq \mathrm { O P T } ( D ) - 1 / T$ whenever $D _ { i } \in \mathcal { C } _ { i , t }$ for every i.

(a) Suppose that, for every product distribution $P ,$ every edge $( u , v )$ revealing a coordinate $i ,$ and all observations $x _ { \mathrm { o b s } ( u ) }$ for which $v \in \mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ , the function $z \mapsto g _ { v } ^ { P } ( z , x _ { \mathrm { o b s } ( u ) } )$ has level-set complexity at most $s _ { i }$ , and let $S$ be as in Theorem 3.2. If the algorithm runs a policy that is optimal for $Q ^ { t }$ at every node, then $R _ { T } = \widetilde { O } ( \sqrt { S T } )$

(b) Let $\alpha \in ( 0 , 1 ]$ , suppose that each $D _ { i }$ has at most k<sub>i</sub> support points, and let $K = k _ { 1 } + \cdot \cdot \cdot + k _ { n }$ If the algorithm runs a policy $\sigma ^ { t }$ with $f ( \sigma ^ { t } \mid Q ^ { t } ) \geq \alpha \mathrm { O P T } ( Q ^ { t } )$ , then $R _ { T } ^ { ( \alpha ) } \le \widetilde O ( \sqrt { K T } )$

The condition obs $( r ) = \emptyset$ can be satisfied by adding dummy nodes before the original root and revealing the initial observations along the resulting chain of edges. Thus, it does not restrict the SOO problems that can be represented.

A distribution $Q ^ { t }$ that satisfies the conditions in the proposition always exists, although finding it can be dificult in general. For example, it can be computed when every domain is finite and the optimal value is nondecreasing under FSD. For each i, we can solve an LP to find a distribution $Q _ { i } ^ { t } \in \mathcal { C } _ { i , t }$ that stochastically dominates every distribution in $\mathcal { C } _ { i , t }$ (Lemma A.2). Then $\begin{array} { r } { Q ^ { t } = \prod _ { i } Q _ { i } ^ { t } } \end{array}$ satisfies $\mathrm { O P T } ( Q ^ { t } ) \geq \mathrm { O P T } ( D )$ whenever $D _ { i } \in \mathcal { C } _ { i , t }$ for every i.

Part (b) shows that, if an α-approximation policy is known for the underlying SOO problem with known distributions, we can incorporate it into this approach, as in the algorithm of Ghuge, Zhuo, Agarwal, and Nagarajan [GZAN26]. The resulting α-regret is $\widetilde { O } ( \sqrt { K T } )$

The variance bound in the proof of Theorem 3.2 can be refined to depend on the optimal value.

Proposition 3.5. Under Assumption 3.1, let S be as in Theorem 3.2 and $p = \mathrm { O P T } ( D )$ . For $\delta = 1 / ( T + 1 )$ , Algorithm 1 satisfies $R _ { T } = \widetilde { O } ( \sqrt { S T p ( 1 - p ) } + S )$

This bound is smaller than that of Theorem 3.2 when the optimal value is close to 0 or 1. Section 5.2 proves Theorem 3.2 and Propositions 3.4 and 3.5, using the confidence-set bounds from Section 5.1.

The next theorem shows that, up to logarithmic factors, the dependence on $S$ and $T$ in Theorem 3.2 is necessary, even under full feedback.

Theorem 3.6. For any integers $T \geq 1$ and $s _ { 1 } , \ldots , s _ { n } \geq 1$ , there is a problem with n coordinates that has the following properties.

(i) Every coordinate has a finite domain, and every leaf reward is 0 or 1. The learner receives full feedback, that is, all n coordinates are observed in every round under every policy.

(ii) Assumption 3.1 holds with $s _ { 1 } , \ldots , s _ { n }$

(iii) For every randomized learning algorithm, there is a product distribution D for which $R _ { T } ( D ) \geq$ $\textstyle { \frac { 1 } { 6 4 } }$ min $\{ T , { \sqrt { S T } } \}$ , where $S = s _ { 1 } + \cdot \cdot \cdot + s _ { n }$ and each $D _ { i }$ has $2 s _ { i } + 1$ support points.

The proof is in Section 6. Together with Theorem 3.2, it shows that the worst-case regret over the problems satisfying Assumption 3.1 with $s _ { 1 } , \ldots , s _ { n }$ is $\widetilde { \Theta } ( \operatorname* { m i n } \{ T , \sqrt { S T } \} )$ for $S = \textstyle \sum _ { i } s _ { i }$ . Since coordinate $i$ of this instance has $2 s _ { i } + 1$ support points, the bound of Corollary 3.3 is also tight up to logarithmic factors in the worst case over supports of these sizes. The lower-bound instance has $2 ^ { S }$ root actions, and the rate $\sqrt { S T }$ coincides with the rate $\sqrt { T \log 2 ^ { S } }$ of prediction with expert advice over $2 ^ { S }$ actions under full feedback [CBL06]. The number of actions does not afect the upper bound, since Theorem 3.2 does not depend on the size of the tree.

## 3.3 Extension to Immediate Rewards

In some applications, the continuation value decomposes into an immediate reward and a residual value. Their sum can have large level-set complexity even when each term has small complexity. For example, in overbooking with group reservations, the refund savings of a group are nondecreasing and the remaining net revenue is nonincreasing in its show-up count, but the optimistic continuation value can alternate between two values as this count increases (Section 4.5). The following theorem bounds the regret by controlling the expectation errors of the two terms separately. We assume that the immediate rewards are nonnegative and sum to at most 1 along each root-to-leaf path consisting of feasible actions. We also require that increasing the revealed value, with all other values fixed, cannot increase the residual reward or enlarge the set of feasible actions at any subsequent node.

Theorem 3.7. Let $s _ { 1 } , \ldots , s _ { n } \geq 1$ be integers and $S = \textstyle \sum _ { i } s _ { i }$ . Suppose that each revelation edge $\boldsymbol { e } = ( u , v )$ has a measurable immediate reward $\chi _ { e } ( x _ { \mathrm { o b s } ( v ) } ) \in [ 0 , 1 ]$ satisfying the following conditions.

(a) For every edge $\boldsymbol { e } = \left( u , v \right)$ revealing $i ,$ fix any $x _ { \mathrm { o b s } ( u ) }$ for which $v \in \mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ . The function $\chi _ { e } ( \cdot , x _ { \mathrm { o b s } ( u ) } )$ has level-set complexity at most $s _ { i }$ , and every leaf ℓ in the subtree rooted at v admits the decomposition

$$
f _ { \ell } \bigl ( x _ { \mathrm { o b s } ( \ell ) } \bigr ) = \kappa _ { e } \bigl ( x _ { \mathrm { o b s } ( u ) } \bigr ) + \chi _ { e } \bigl ( x _ { \mathrm { o b s } ( v ) } \bigr ) + F _ { e , \ell } \bigl ( x _ { \mathrm { o b s } ( \ell ) } \bigr ) , \qquad 0 \le F _ { e , \ell } \le 1 ,
$$

where $\kappa _ { e }$ is common to all such leaves and the identity holds for all remaining coordinate values.

(b) For each edge and fixed observations in (a), the residual rewards $F _ { e , \ell }$ and the sets of feasible actions in the subtree rooted at v are nonincreasing in $x _ { i } ,$ , with other coordinates fixed and sets ordered by inclusion.

(c) The immediate rewards sum to at most 1 along every root-to-leaf path consisting of feasible actions.

Then, for every $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , Algorithm 1 satisfies (3.6) with a larger universal constant c. In particular, for $\delta = 1 / ( T + 1 )$ , we have $R _ { T } = \widetilde { O } ( \sqrt { S T } )$ .

Since the supremum of a sum is at most the sum of the suprema, the proof bounds the expectation errors separately for the immediate reward and the residual continuation value, and hence no level-set bound on their sum is needed. As in Theorem 3.2, the bound also holds with $\begin{array} { r } { S = \sum _ { i } \operatorname* { m i n } \{ s _ { i } , k _ { i } \} } \end{array}$ , since Lemma 5.5 applies to the immediate rewards if $k _ { i } < s _ { i }$

When the total immediate reward is bounded by the leaf reward, the regret bound can be refined in terms of the optimal value.

Proposition 3.8. In addition to the conditions of Theorem 3.7, suppose that the sum in (c) is at most the reward of the leaf at the end of the path, and let $p = \mathrm { O P T } ( D )$ . For $\delta = 1 / ( T + 1 )$ , Algorithm 1 satisfies $R _ { T } = \widetilde { O } ( \sqrt { S T p } + S )$

Section 5.3 proves Theorem 3.7 and Proposition 3.8.

## 4 Applications

We apply Theorems 3.2 and 3.7 to five problems, none of which is FSD-monotone in general. In each problem, the objective lies in a known interval of length $\Gamma > 0$ . We rescale it to a reward in [0, 1] by an afine map. For cost minimization, the reward is 1 minus the incurred cost divided by Γ. Regret in the original objective is then Γ times the normalized regret. Throughout this section, we take $\delta = 1 / ( T + 1 )$ ). When the scheduling reward functions have bounded level-set complexity, all five problems admit regret $\widetilde { O } ( \Gamma \sqrt { n T } )$ in the original objective, and Table 2 lists sharper bounds that depend on the optimal values.

Table 2: Applications in Section 4, with regret in the original objective up to logarithmic factors. Here, Γ is the normalizing constant, $C ^ { * } , p ^ { * } , U ^ { * } , \mathrm { O P T ^ { * } }$ , and $W ^ { * }$ are the optimal values, C is the largest overflow cost, and the scheduling reward functions $r _ { i }$ are unimodal. For scheduling and overbooking, $s _ { i }$ is the parameter of Theorem 3.7.
<table><tr><td>Problem</td><td>Why FSD monotonicity fails</td><td> $s _ { i }$ </td><td>Reference</td><td>Regret</td></tr><tr><td>SBFE</td><td>balanced outcomes need the most tests</td><td>1</td><td>Prop. 3.5</td><td> $\sqrt { n T C ^ { * } ( \Gamma - C ^ { * } ) } + \Gamma n$ </td></tr><tr><td>Secretary</td><td>larger later values favor waiting</td><td>2</td><td>Prop. 3.5</td><td> $\sqrt { n T p ^ { * } ( 1 - p ^ { * } ) } + n$ </td></tr><tr><td>Loss aversion</td><td>early values raise the reference point</td><td>2</td><td>Prop. 3.5</td><td> $\sqrt { n T ( U ^ { * } + \lambda ) ( 1 - U ^ { * } ) } + \Gamma n$ </td></tr><tr><td>Scheduling</td><td>longer jobs delay later jobs</td><td>1</td><td>Prop. 3.8</td><td> $\sqrt { n T \Gamma \mathrm { O P T } ^ { * } } + \Gamma n$ </td></tr><tr><td>Overbooking</td><td>arrivals save refunds but cause overflow</td><td>1</td><td>Prop. 3.8</td><td> $\sqrt { n T \Gamma ( W ^ { * } + C ) } + \Gamma n$ </td></tr></table>

## 4.1 Stochastic Boolean Function Evaluation

In SBFE, a known Boolean function $F : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ and nonnegative testing costs $c _ { 1 } , \ldots , c _ { n }$ are given, and the success probabilities of the independent binary variables $X _ { i }$ are unknown. A policy tests coordinates one at a time until the observed values determine $F ( X )$ , that is, until all completions of the unobserved coordinates yield the same function value. The objective is to minimize the expected testing cost. SBFE is a basic model of sequential testing, and approximation algorithms for known probabilities have been studied [DHK14].

Each node of the tree corresponds to the sequence of coordinates tested so far. If the observed values do not determine F, the feasible actions are the tests of the untested coordinates. Otherwise, the policy stops. With $\textstyle \Gamma = \sum _ { i } c _ { i } > 0$ , the reward at a leaf ℓ is $\begin{array} { r } { f _ { \ell } = 1 - ( \sum _ { i \in \mathrm { o b s } ( \ell ) } c _ { i } ) / \Gamma } \end{array}$

To see that SBFE need not be FSD-monotone, consider the majority function of three variables with unit testing costs shown in Figure 1, and suppose that all three variables have a common success probability $p \in ( 0 , 1 )$ . At least two tests are required, and the third is needed exactly when the first two outcomes difer. By symmetry, regardless of the order, the optimal expected cost is $2 + 2 p ( 1 - p )$ . For $p = 1 / 4 , 1 / 2 , 3 / 4$ , the Bernoulli distribution increases in FSD, while the cost is $1 9 / 8 , 5 / 2$ , and $1 9 / 8$ . Thus, the problem is not FSD-monotone, although the majority function is monotone as a Boolean function.

Proposition 4.1. For SBFE with any Boolean function and nonnegative testing costs, the expected regret in the original testing cost is $\widetilde { O } ( \Gamma \sqrt { n T } )$

Proof. Every coordinate is binary, and hence Assumption 3.1 holds with $s _ { i } = 1$ for every $i ,$ as noted after the assumption. Theorem 3.2 with $S = n$ implies the claim. □

Let $C ^ { * }$ be the optimal expected testing cost. Since the optimal normalized reward is $1 - C ^ { * } / \Gamma$ Proposition 3.5 bounds the regret in the original testing cost also by $\widetilde { O } ( \sqrt { n T C ^ { * } ( \Gamma - C ^ { * } ) } + \dot { \Gamma n } )$

which is smaller when $C ^ { * } \ll \Gamma$ . For threshold functions $F ( x ) = \mathbf { 1 } \{ x _ { 1 } + \cdot \cdot \cdot + x _ { n } \geq \theta \}$ , which include the majority function, the reward splits into two parts that are FSD-monotone in opposite directions, and a variant of the classical optimal policy for known distributions [BD81, SB84] is optimal for both parts under diferent optimistic distributions. Applying Proposition 3.4 to each part yields an algorithm with $O ( n ^ { 2 } )$ time per round and regret $\widetilde { O } ( \Gamma \sqrt { n T } )$ (Proposition B.1 in Appendix B).

## 4.2 Full-Information Secretary Problem

In the full-information secretary problem, candidates with independent values $X _ { 1 } , \ldots , X _ { n } \in [ 0 , 1 ]$ arrive in this order, the decision maker decides immediately whether to accept the current candidate, and the goal is to maximize the probability of accepting a candidate with the largest value among all n candidates. Unlike in the classical secretary problem, the decision maker observes the values themselves rather than only their relative ranks. Gilbert and Mosteller [GM66] study identical known distributions, and Esfandiari, Hajiaghayi, Lucier, and Mitzenmacher [EHLM20] study nonidentical known distributions. Harada, Kawase, and Sumita [HKS25] study the repeated version with unknown distributions under full feedback, together with per-round guarantees relative to a baseline. Since leaf rewards depend only on revealed values and the success of a policy depends on the values after the accepted candidate, these values must be revealed after acceptance, and hence all values are observed in every round.

Let $u _ { i }$ be the node at time $i ,$ with $X _ { 1 }$ initially observed at the root $r = u _ { 1 }$ . For $i < n$ , the node $u _ { i }$ has a child $u _ { i + 1 }$ revealing $X _ { i + 1 }$ and another child that starts a chain of revelation edges revealing $X _ { i + 1 } , \ldots , X _ { n }$ in order with no intermediate choices and ending at a leaf $\ell _ { i } ,$ and $u _ { n } = \ell _ { n }$ is the final leaf. The reward at $\ell _ { i }$ is $f _ { \ell _ { i } } ( x ) = \mathbf { 1 } \{ x _ { i } \geq x _ { j }$ for all $j \in [ n ] \}$ , and hence $\Gamma = 1$

The problem is not FSD-monotone even with two candidates. Let $n = 2 , X _ { 1 } = 1 / 2$ , and $X _ { 2 } \in \{ 0 , 1 \}$ with $\mathbb { P } ( X _ { 2 } = 1 ) = q .$ . Accepting the first candidate succeeds with probability $1 - q ,$ and accepting the second succeeds with probability $q .$ Hence, the optimal success probability is max $\{ q , 1 - q \}$ , which first decreases and then increases as q increases.

Proposition 4.2. For the full-information secretary problem, Assumption 3.1 holds with $s _ { 1 } ~ =$ $\dots = s _ { n } = 2$ , and the expected regret is $\widetilde { O } ( \sqrt { n T } )$

Proof. Fix arbitrary nonempty sets of distributions as in Assumption 3.1. Every leaf reward is nondecreasing in the value of the accepted candidate and nonincreasing in the values of the other candidates, and the feasible actions do not depend on the values. Expectations, suprema, and maxima preserve these monotonicities. Hence, backward induction shows that every optimistic continuation value on the chain after accepting candidate i is nondecreasing in $x _ { i }$ and nonincreasing in the values of the other candidates, and that the optimistic continuation value at $u _ { k }$ is nonincreasing in each past value $x _ { j }$ with $j < k$ . The optimistic continuation value after a value is revealed on a chain is therefore monotone in that value. At $u _ { i }$ with $i < n$ , the value of accepting is nondecreasing in $x _ { i }$ , and the value of continuing is nonincreasing in $x _ { i }$ . Their maximum has level-set complexity at most 2, and the value at $u _ { n }$ is nondecreasing in $x _ { n }$ . Thus, Assumption 3.1 holds with $s _ { i } = 2$ , and Theorem 3.2 with $S \leq 2 n$ implies the claim. □

Proposition 3.5 refines the bound to $\widetilde O ( \sqrt { n T p ^ { * } ( 1 - p ^ { * } ) } + n )$ , where $p ^ { * }$ is the optimal success probability, and this bound is smaller when $p ^ { * }$ is close to 0 or 1. If every domain is finite, Proposition A.4 applies with the following states. At $u _ { i }$ , the state is the pair of $\operatorname* { m a x } _ { j < i } x _ { j }$ , which is 0 for $i = 1$ , and $x _ { i }$ . On the chain after accepting candidate $i ,$ the state is the pair of $x _ { i }$ and the indicator that $x _ { i }$ is at least every value revealed so far. Each round then runs in time polynomial in $n , t ,$ and the sizes of the domains.

The model also captures random arrival orders. For $k \in [ n - 1 ]$ , let $Y _ { k }$ be an additional coordinate with the uniform distribution on [0, 1]. Before the kth arrival, that is, at the root if $k = 1$ and after the rejection of the $( k - 1 )$ th arriving candidate otherwise, the next edge reveals $Y _ { k }$ The only feasible action then reveals the candidate selected by $Y _ { k }$ through the conditional distribution of the kth arrival given the previous arrivals, for example, by listing the remaining candidates in a fixed order and selecting the first one whose cumulative conditional probability is at least $Y _ { k }$ . When one candidate remains, rejecting reveals this candidate. In this way, every known distribution over the n! arrival orders arises, and for uniformly random orders, $Y _ { k }$ can instead be uniform on $n - k + 1$ values. Since the distributions of the coordinates $Y _ { k }$ are known, the algorithm can use the set consisting only of the distribution of $Y _ { k }$ as its confidence set. The gaps at the stages that reveal $Y _ { k }$ then vanish in the proof of Theorem 3.2, and this proof, together with that of Proposition 4.2, which applies to every arrival order, yields regret $\widetilde { O } ( \sqrt { n T } )$ with S summing only over the candidates. The decision tree has a node for each possible order of the candidates seen so far, but the regret bound does not depend on its size.

## 4.3 Optimal Stopping with Loss Aversion

Kleinberg, Kleinberg, and Oren [KKO22] study a variant of optimal stopping with loss aversion, in which the best value seen so far serves as a reference point and accepting a worse candidate incurs a penalty. In our setting, independent candidate values $X _ { 1 } , \ldots , X _ { n } \in [ 0 , 1 ]$ with unknown distributions are observed in this order. After observing each value, the policy either accepts the current candidate and stops or observes the next one, and the last candidate must be accepted. Accepting candidate i yields the utility $x _ { i } - \lambda ( \operatorname* { m a x } _ { j < i } x _ { j } - x _ { i } ) _ { + }$ , where $\lambda \geq 0$ is a known loss-aversion coeficient and the maximum over the empty set is 0.

Let $u _ { i }$ be the node at time $i ,$ with $X _ { 1 }$ initially observed at the root $r = u _ { 1 }$ . For $i < n ,$ the node $u _ { i }$ has a stopping leaf $\ell _ { i }$ and a child $u _ { i + 1 }$ revealing $X _ { i + 1 }$ , and $u _ { n } = \ell _ { n }$ is the final leaf. Since utilities lie in $[ - \lambda , 1 ]$ , we set $\Gamma = 1 + \lambda$ and define the leaf reward as

$$
f _ { \ell _ { i } } ( x _ { 1 } , \dots , x _ { i } ) = { \frac { x _ { i } - \lambda ( \operatorname* { m a x } _ { j < i } x _ { j } - x _ { i } ) _ { + } + \lambda } { 1 + \lambda } } .
$$

Both directions of FSD monotonicity can fail even with two candidates. Let $\lambda > 0$ and $n = 2$ First let $X _ { 1 } = 0 , X _ { 2 } \sim$ Bernoulli(q), and $q \in ( 0 , 1 )$ . The optimal expected utility is $q .$ If only the first distribution is made larger in FSD by setting $X _ { 1 } = a , 0 < a < q$ , then the optimal expected utility becomes max $\{ a , q - \lambda a ( 1 - q ) \} < q$ . Thus, making a distribution larger can decrease the optimal value. On the other hand, with $X _ { 1 } = 0$ , increasing a deterministic second candidate from $1 / 2$ to 1 increases the optimal value, and hence the problem is not FSD-monotone.

Proposition 4.3. For optimal stopping with loss aversion and every $\lambda \geq 0$ , Assumption 3.1 holds with $s _ { 1 } = \cdot \cdot \cdot = s _ { n } = 2$ , and the expected regret in the original utility is $\widetilde { O } ( \sqrt { ( 1 + \lambda ) n T } + ( 1 + \lambda ) n )$

Proof. Fix arbitrary nonempty sets of distributions as in Assumption 3.1. The reward for accepting candidate i is nondecreasing in $x _ { i }$ and nonincreasing in every past value, since increasing a past value cannot lower the reference point. As in the proof of Proposition 4.2, the value of accepting the current candidate is nondecreasing in its value, the value of continuing is nonincreasing in ${ \mathrm { i t } } ,$ and Assumption 3.1 holds with $s _ { i } = 2$

Let $U ^ { * }$ be the optimal expected utility, which lies in [0, 1] since accepting the first candidate yields nonnegative utility. Proposition 3.5 with $S \le 2 n$ and $p = ( U ^ { * } + \lambda ) / ( 1 + \lambda )$ bounds the normalized regret by $\widetilde O ( \sqrt { n T p ( 1 - p ) } + n )$ . Multiplying by $\Gamma = 1 + \lambda$ and using $( 1 + \lambda ) ^ { 2 } p ( 1 - p ) =$ $( U ^ { * } + \lambda ) ( 1 - U ^ { * } )$ yields regret $\widetilde { O } ( \sqrt { n T \left( U ^ { * } + \lambda \right) ( 1 - U ^ { * } ) } + ( 1 + \lambda ) n )$ in the original utility, and $( U ^ { * } + \lambda ) ( 1 - U ^ { * } ) \leq 1 + \lambda$ proves the claim. □

The dependence on λ is necessary already for two candidates.

Proposition 4.4. For every $\lambda \geq 0$ and $T \geq 1$ , there are two instances of optimal stopping with loss aversion with $n = 2$ candidates such that every learning algorithm incurs expected regret at least $\frac { 1 } { 4 8 }$ min $\{ T , \sqrt { ( \lambda + 2 ) T } \}$ in the original utility on one of them, even under full feedback, in which both values are revealed at the end of every round.

For $n = 2$ and $T \geq \lambda + 2$ , this lower bound is at least $\textstyle { \frac { 1 } { 4 8 } } { \sqrt { ( 1 + \lambda ) T } }$ , and hence the dependence of Proposition 4.3 on λ and T is tight up to logarithmic factors. Appendix C proves Proposition 4.4. Appendix A.3 shows that each round runs in polynomial time when each domain is [0, 1] or finite.

## 4.4 Stochastic Scheduling with Deadlines

In stochastic scheduling with deadlines, a single machine processes n jobs one at a time without preemption. Job i has a known release time $e _ { i } \geq 0$ , a known deadline $d _ { i } .$ , and a random outcome $X _ { i } ~ \in ~ [ 0 , 1 ]$ The outcomes are independent across jobs and have unknown distributions. The processing time is $p _ { i } ( X _ { i } )$ , and completing the job by its deadline earns reward $r _ { i } ( X _ { i } )$ , where $p _ { i }$ and $r _ { i }$ are known nonnegative functions. We assume that $p _ { i }$ is nondecreasing and $r _ { i }$ has level-set complexity at most an integer $s _ { i } \geq 1$ . We can take $s _ { i } = 1$ for fixed, monotone, or unimodal reward functions. Examples include rewards proportional to processing time, $r _ { i } ( x ) = a _ { i } p _ { i } ( x )$ , and capped rewards $r _ { i } ( x ) = \mathrm { m i n } \{ a _ { i } p _ { i } ( x ) , b _ { i } \}$ with $a _ { i } , b _ { i } \geq 0$ . If job i has finitely many possible processing-time– reward pairs, ordering the pairs by processing time and assigning distinct outcome labels allows us to take $s _ { i }$ equal to half their number rounded up.

A policy starts at time 0 and repeatedly selects an unprocessed job or stops. If job i is selected when the machine becomes free at time $\tau _ { \ast }$ , it starts at time max $\{ \tau , e _ { i } \}$ and completes at time $\operatorname* { m a x } \{ \tau , e _ { i } \} + p _ { i } ( X _ { i } )$ . The job runs to completion even if it misses its deadline, in which case it earns no reward. Upon completion, the learner observes $( p _ { i } ( X _ { i } ) , r _ { i } ( X _ { i } ) )$ and may continue with another job. We assume that the observed pair uniquely identifies $X _ { i }$ , including when the deadline is missed.

Let $\Gamma > 0$ be a known upper bound on the total earned reward, for example, $\textstyle \sum _ { i } \operatorname* { s u p } _ { x } r _ { i } ( x )$ In the tree, each node records the sequence of jobs processed so far, and their observed processing times determine the current time. The normalized leaf reward is the total earned reward divided by Γ. Dean [Dea05] studies stochastic scheduling with deadlines, including release times. Here, processing time and reward may depend on a common random outcome.

Correlated stochastic knapsack [GKMR11, Ma18], under the conditions above, is a special case with $e _ { i } = 0$ and $d _ { i } = B$ for every i, where $B > 0$ is the capacity. Each job corresponds to an item with size $p _ { i } ( X _ { i } )$ and reward $r _ { i } ( X _ { i } )$ , earned if the cumulative size including that item is at most B. As in the standard stochastic knapsack model [DGV08], the process stops at the first overflow, and the overflowing item earns no reward. Our analysis also covers the soft-constraint model [YK25], in which the overflowing item earns its reward before the process stops. If $r _ { i } ( x ) \leq L p _ { i } ( x )$ for all i and x, we can take $\Gamma = L B$ under the standard convention and $\Gamma = L B + \operatorname* { m a x } _ { i } \operatorname* { s u p } _ { x } r _ { i } ( x )$ under the soft-constraint convention.

To see that FSD monotonicity can fail even under the standard knapsack convention, consider capacity $B = 1$ and two items. For item 1, take $p _ { 1 } ( z ) = r _ { 1 } ( z ) = z$ and let $X _ { 1 } = x$ deterministically, while item 2 has fixed size $1 / 2$ and reward 1. For $x = 1 / 4 , 1 / 2 , 3 / 4$ , the known-distribution optimal total rewards are, respectively, $5 / 4 , 3 / 2$ , and 1. In the first two cases both items fit, while in the last case selecting only item 2 is optimal. The distribution of $X _ { 1 }$ increases in FSD, but the optimal value first increases and then decreases. Thus, the problem is not FSD-monotone.

We apply Theorem 3.7 by separating the reward of the current job from the value of the remaining jobs. The latter is nonincreasing in the current outcome because $p _ { i }$ is nondecreasing. Their sum need not have small level-set complexity, even when the immediate reward is monotone. For example, consider knapsack with $B = 1$ , item 1 with $p _ { 1 } ( z ) = r _ { 1 } ( z ) = z$ , and item 2 with reward 1 and size uniform on $\{ 1 / m , 2 / m , \ldots , 1 \}$ . After item 1 is inserted, the optimal continuation value is proportional to $x _ { 1 } + \lfloor m ( 1 - x _ { 1 } ) \rfloor / m$ , which equals 1 at $x _ { 1 } \in \{ 0 , 1 / m , \ldots , 1 \}$ and lies in $( 1 - 1 / m , 1 )$ elsewhere, and hence its superlevel set at level $1 - 1 / ( 2 m )$ consists of $m + 1$ separated parts.

Proposition 4.5. For the scheduling model above with nondecreasing processing-time functions $p _ { i }$ and reward functions $r _ { i }$ of level-set complexity at most $s _ { i ; }$ , let $S = \textstyle \sum _ { i } s _ { i }$ . The expected regret in the original total reward is $\widetilde { O } ( \Gamma \sqrt { S T } )$

Proof. Fix an edge revealing job i, its start time $\tau _ { i } .$ , and all preceding observations. For each leaf below this edge, write $f _ { \ell } = \kappa + \chi ( x _ { i } ) + F _ { \ell } .$ , where κ and $F _ { \ell }$ are the normalized rewards earned before and after job i, and $\chi ( z ) = \mathbf { 1 } \{ \tau _ { i } + p _ { i } ( z ) \leq d _ { i } \} r _ { i } ( z ) / \Gamma$ . The reward bound ensures that $\chi , F _ { \ell } \in [ 0 , 1 ]$ Since $p _ { i }$ is nondecreasing, the set $\{ z : \tau _ { i } + p _ { i } ( z ) \leq d _ { i } \}$ is an initial interval of the outcome domain. At a nonnegative level, a superlevel set of χ is the intersection of this set with a superlevel set of $r _ { i }$ , and hence has at most $s _ { i }$ interval components.

Increasing $x _ { i }$ cannot decrease $p _ { i } ( x _ { i } )$ . Thus, no subsequent job along a fixed path completes earlier, and $F _ { \ell }$ cannot increase. The sets of feasible actions remain unchanged, except that an overflow in the knapsack case allows only stopping. The immediate rewards sum to the normalized leaf reward and hence to at most 1. Theorem 3.7 and rescaling by Γ imply the claim. For the soft-constraint knapsack model, the same proof applies with $\chi ( z ) = r _ { i } ( z ) / \Gamma$ □

Let ${ \mathrm { O P T ^ { * } } }$ be the optimal expected total reward. Since the immediate rewards sum to the leaf reward, Proposition 3.8 with $p = \mathrm { O P T ^ { * } / \Gamma }$ improves the bound to $\widetilde { O } ( \sqrt { S T \Gamma { \mathrm { O P T } } ^ { * } } + \Gamma S )$ . When the $s _ { i }$ are bounded by a constant, both bounds hold with n in place of S, up to constant factors.

## 4.5 Overbooking with Group Reservations

In overbooking with group reservations, a service provider receives reservation requests from n groups. Group i has $b _ { i } \geq 1$ members, pays a fee $a _ { i } \geq 0$ , and receives a refund $\rho _ { i } \geq 0$ for each member who does not show up, where $\rho _ { i } b _ { i } \leq a _ { i }$ . The number of members of group i who show up is $Y _ { i } \in \{ 0 , \ldots , b _ { i } \}$ . These counts are independent across groups and have unknown distributions.

We make no assumption on the dependence between the members of a group. The capacity is an integer $B \geq 0$ , and each customer beyond the capacity incurs a cost $c \geq 0$ . The parameters $b _ { i } ,$ $a _ { i } , \rho _ { i } , B _ { : }$ and c are known. We take $X _ { i } = Y _ { i } / b _ { i }$ as the coordinate of group i, whose domain is $\{ 0 , 1 / b _ { i } , 2 / b _ { i } , \ldots , 1 \}$ . The case $b _ { 1 } = \cdot \cdot \cdot = b _ { n } = 1$ is the model of individual reservations with binary show-up outcomes. Reservation management with cancellations, refunds, and no-shows is studied, for example, by Subramanian, Stidham, and Lautenbacher [SSL99].

Before any show-up outcome is observed, the policy chooses a set $U$ of accepted reservations from a known family $\mathcal { U } \subseteq 2 ^ { [ n ] }$ with $\varnothing \in \mathcal { U }$ . At the time of service, it observes $Y _ { i }$ for $i \in U$ and receives the net revenue

$$
W ( U , Y _ { U } ) = \sum _ { i \in U } \bigl ( a _ { i } - \rho _ { i } ( b _ { i } - Y _ { i } ) \bigr ) - c \Bigl ( \sum _ { i \in U } Y _ { i } - B \Bigr ) _ { + } .
$$

Since the policy chooses U before observing any outcome, this is a combinatorial semi-bandit problem. The monotonicity condition for combinatorial bandits with general reward functions [CHL<sup>+</sup>16] can fail here because of the cost of exceeding capacity.

In the tree, choosing a nonempty set U reveals its coordinates along a chain in a fixed order, while choosing $U = \emptyset$ leads directly to a stopping leaf. Let $\bar { b } = \textstyle \sum _ { i } b _ { i }$ and $C = c ( \bar { b } - B ) _ { + }$ . Since $\begin{array} { r } { - C \le W ( U , Y _ { U } ) \le \sum _ { i } a _ { i } } \end{array}$ , we take $\textstyle \Gamma = \sum _ { i } a _ { i } + C$ , assume $\Gamma > 0$ , and normalize the leaf reward as $f _ { \ell } = ( W ( U , y _ { U } ) + C ) / \Gamma$

The optimal value can also be non-monotone in the show-up distribution. For example, consider a single group with $b _ { 1 } = 3 , a _ { 1 } = 3$ , and $\rho _ { 1 } = 1$ , and let $B = 2$ and $c = 2$ . If $Y _ { 1 }$ is deterministic and equal to 1, 2, or 3, accepting the group yields the net revenue 1, 2, or 1, respectively, while rejecting it yields 0. Thus, the optimal value first increases and then decreases, and the problem is not FSD-monotone.

We apply Theorem 3.7 by decomposing the net revenue into the refund savings for the current group and the remaining net revenue. The former is nondecreasing in the current show-up count, while the latter is nonincreasing. Their sum, the full optimistic continuation value, can have levelset complexity that grows with the group sizes. Let $b _ { 1 } = b _ { 2 } = B = a _ { 1 } = a _ { 2 } = 2 m , \rho _ { 1 } = \rho _ { 2 } = 1$ and $c = 2 .$ , accept both groups, and let the set of distributions of $Y _ { 2 }$ consist of the point masses at $0 , 2 , \ldots , 2 m$ . Then, the optimistic net revenue after $Y _ { 1 } ~ = ~ y$ is observed is the maximum of $y + y ^ { \prime } - 2 ( y + y ^ { \prime } - 2 m ) _ { + }$ over these values $y ^ { \prime }$ , which equals 2m for even $y$ and $2 m - 1$ for odd $y .$ Hence, its superlevel set at level 2m $- 1 / 2$ consists of $m + 1$ separated points, and Assumption 3.1 would require $s _ { 1 } \geq m + 1$ , which grows with the group sizes.

Proposition 4.6. For overbooking with group reservations, the expected regret in the original net revenue is $\widetilde { O } ( \Gamma \sqrt { n T } )$ , where n is the number of groups.

Appendix C proves Proposition 4.6.

Let $W ^ { * }$ be the optimal expected net revenue. The total immediate reward is at most the normalized leaf reward because $a _ { i } - \rho _ { i } b _ { i } \geq 0$ and the overflow cost is at most C. Proposition 3.8 with $p = ( W ^ { \ast } + C ) / \Gamma$ therefore improves the bound to $\widetilde { O } ( \sqrt { n T \Gamma ( W ^ { * } + C ) } + \Gamma n )$ . Apart from $\Gamma _ { \mathrm { {  } } }$ the basic bound does not depend on the group sizes $b _ { i }$ . Appendix $\mathrm { A . 3 }$ shows that each round runs in time polynomial in $n , t , | \mathcal { U } |$ , and <sup>¯</sup>b.

## 5 Analysis of the Upper Bound

In this section, we prove the regret upper bounds of Section 3. Section 5.1 establishes confidenceset coverage and bounds the resulting expectation errors. Section 5.2 combines these bounds with a variance budget to prove Theorem 3.2 and Propositions 3.4 and 3.5. Section 5.3 extends the analysis to immediate rewards.

## 5.1 Properties of Confidence Sets

To analyze adaptively collected samples, we represent the observations of each coordinate by an independent sequence of i.i.d. samples. For each coordinate i, fix an infinite sequence $Z _ { i , 1 } , Z _ { i , 2 } , . . .$ of i.i.d. random variables with distribution $D _ { i } .$ , independently across coordinates, and interpret the j-th revealed value of coordinate i as $Z _ { i , j }$ . This preserves the observation law because coordinates are independent and each revelation decision precedes observing that coordinate’s value.

The following interval bound is a consequence of the empirical Bernstein inequality of Maurer and Pontil [MP09, Theorem 6], and Appendix C proves it.

Lemma 5.1. Let $m \geq 1 6$ and $\eta \in ( 0 , 1 )$ , and let $\widehat { P } _ { m }$ be the empirical distribution of i.i.d. samples $Z _ { 1 } , \ldots , Z _ { m }$ from a distribution P on [0, 1]. With probability at least $1 - \eta$ , we have simultaneously for all $I \in \mathcal { T } _ { 1 }$

$$
| P ( I ) - \widehat { P } _ { m } ( I ) | \leq \sqrt { \frac { 1 8 \widehat { P } _ { m } ( I ) ( 1 - \widehat { P } _ { m } ( I ) ) \beta } { m - 1 } } + \frac { 1 5 \beta } { m - 1 } , \qquad w h e r e \beta = \log \frac { 2 0 ( 2 m + 1 ) ^ { 2 } } \eta .\tag{5.1}
$$

We now apply this bound to show that the confidence sets contain the true distributions with high probability.

Lemma 5.2. With probability at least $1 - \delta$ , we have $D _ { i } \in \mathcal { C } _ { m } ( \widehat { D } _ { i , m } )$ simultaneously for every $i \in [ n ]$ and every $0 \leq m \leq T$

Proof. For $m = 0$ , the inclusion holds by definition. For $1 \leq m < 1 6$ , we have $b _ { m } ( z ) = 1$ for every $z \in [ 0 , 1 ]$ since $\Lambda > 1$ , and hence the constraints are vacuous.

Fix i and $1 6 \leq m \leq T$ , and apply (5.1) to $D _ { i }$ and its samples $Z _ { i , 1 } , \ldots , Z _ { i , m }$ with $\begin{array} { r } { \eta = \frac { \delta } { n ( T + 1 ) } } \end{array}$ Since $( 2 m + 1 ) ^ { 2 } \leq 4 ( T + 1 ) ^ { 2 }$ , we have $\beta = \log ( 2 0 n ( T + 1 ) ( 2 m + 1 ) ^ { 2 } / \delta ) \leq \log ( 8 0 n ( T + 1 ) ^ { 3 } / \delta ) = \Lambda$

For $I \in \mathcal { T } _ { 1 }$ , define $z = \widehat { D } _ { i , m } ( I )$ . By (5.1), we have

$$
\begin{array} { c } { \displaystyle { | D _ { i } ( I ) - \widehat { D } _ { i , m } ( I ) | \leq \sqrt { \displaystyle { \frac { 1 8 z ( 1 - z ) \beta } { m - 1 } } } + \frac { 1 5 \beta } { m - 1 } \leq \sqrt { \displaystyle { \frac { 1 8 z ( 1 - z ) \Lambda } { m - 1 } } } + \frac { 1 5 \Lambda } { m - 1 } } } \\ { \displaystyle { = \sqrt { \displaystyle { \frac { 1 8 m } { m - 1 } } } \sqrt { z ( 1 - z ) \displaystyle { \frac { \Lambda } { m } } } + \frac { 1 5 m } { m - 1 } \displaystyle { \frac { \Lambda } { m } } \leq 6 \sqrt { z ( 1 - z ) \displaystyle { \frac { \Lambda } { m } } } + 3 2 \displaystyle { \frac { \Lambda } { m } } . } } \end{array}
$$

The last inequality uses $m / ( m - 1 ) \leq 2$ , which implies $\sqrt { { 1 8 m / ( m - 1 ) } } \le 6$ and $1 5 m / ( m - 1 ) \leq 3 0 \leq$ 32. Since the diference is also at most 1, $D _ { i }$ satisfies the constraints with the radius (3.1). For each pair $( i , m )$ with $i \in [ n ]$ and $1 6 \leq m \leq T$ , the inclusion fails with probability at most $\delta / [ n ( T + 1 ) ]$ . A union bound over at most $n ( T + 1 )$ such pairs, together with the deterministic inclusion for $m < 1 6$ 2 proves that all inclusions hold simultaneously with probability at least $1 - \delta$ □

The next lemma bounds the integral that arises when a function is written using its superlevel sets.

Lemma 5.3. Let $Z \in [ 0 , 1 ]$ be a random variable, and define $p _ { y } = \mathbb { P } ( Z > y )$ . For every $\eta \in ( 0 , 1 )$

$$
\int _ { 0 } ^ { 1 } \sqrt { p _ { y } ( 1 - p _ { y } ) } d y \le \eta + \sqrt { \mathrm { V a r } ( Z ) \log ( 1 / \eta ) } .\tag{5.2}
$$

Proof. Let $\mu = \mathbb { E } [ Z ]$ . We split the integral in (5.2) into the regions $| y - \mu | < \eta$ and $| y - \mu | \geq \eta$ . The first part is at most $\eta ,$ since its integration range has length at most $2 \eta$ and $\sqrt { p _ { y } ( 1 - p _ { y } ) } \le 1 / 2$ for all $p _ { y } \in [ 0 , 1 ]$ . For the second part, the Cauchy–Schwarz inequality implies

$$
\int _ { \{ y \in [ 0 , 1 ] : | y - \mu | \geq \eta \} } \sqrt { p _ { y } ( 1 - p _ { y } ) } d y \leq \left( \int _ { 0 } ^ { 1 } | y - \mu | p _ { y } ( 1 - p _ { y } ) d y \right) ^ { 1 / 2 } \left( \int _ { \{ y \in [ 0 , 1 ] : | y - \mu | \geq \eta \} } \frac { d y } { | y - \mu | } \right) ^ { 1 / 2 } .
$$

For the first integral, we use $p _ { y } ( 1 - p _ { y } ) \leq p _ { y } = \mathbb { P } ( Z > y )$ for $y \geq \mu$ and $p _ { y } ( 1 - p _ { y } ) \leq ( 1 - p _ { y } ) =$ $\mathbb { P } ( Z \le y )$ for $y < \mu$ . Then, we have

$$
\begin{array} { r l r } {  { \int _ { 0 } ^ { 1 } | y - \mu | p _ { y } ( 1 - p _ { y } ) d y \le \int _ { \mu } ^ { 1 } ( y - \mu ) \mathbb { P } ( Z > y ) d y + \int _ { 0 } ^ { \mu } ( \mu - y ) \mathbb { P } ( Z \le y ) d y } } \\ & { } & { = \frac { \mathbb { E } [ ( Z - \mu ) _ { + } ^ { 2 } ] + \mathbb { E } [ ( \mu - Z ) _ { + } ^ { 2 } ] } { 2 } = \frac { \mathbb { E } [ ( Z - \mu ) ^ { 2 } ] } { 2 } = \frac { \mathrm { V a r } ( Z ) } { 2 } . } \end{array}
$$

The first equality follows from the tail-integral formula $\begin{array} { r } { \mathbb { E } [ U ^ { 2 } ] = 2 \int _ { 0 } ^ { \infty } t \mathbb { P } ( U > t ) } \end{array}$ dt for nonnegative U, applied to $( Z - \mu ) _ { + }$ and $( \mu - Z ) _ { + }$

For the second integral, substitute $u = | y - \mu |$ separately on each side of $\mu .$ . The integration range on either side is contained in $[ \eta , 1 ]$ , and hence

$$
\int _ { \{ y \in [ 0 , 1 ] : | y - \mu | \geq \eta \} } \frac { d y } { | y - \mu | } \leq 2 \int _ { \eta } ^ { 1 } \frac { d u } { u } = 2 \log ( 1 / \eta ) .
$$

The product of the two square roots is therefore at most $\sqrt { \operatorname { V a r } ( Z ) \log ( 1 / \eta ) }$ . Adding the contribution of at most η from $| y - \mu | < \eta$ proves the claim. □

The following lemma compares two distributions in the same confidence set, such as $D _ { i }$ and a distribution used in the optimistic update, first on unions of intervals and then on functions with small level-set complexity.

Lemma 5.4. Let $s \geq 1$ and $m \geq 0$ be integers and $P , Q \in \mathcal { C } _ { m } ( \widehat { D } _ { i , m } )$

(a) If $m \geq 1$ , then for every $\boldsymbol { B } \in \mathcal { I } _ { s }$

$$
| Q ( B ) - P ( B ) | \leq 2 4 \sqrt { P ( B ) ( 1 - P ( B ) ) \frac { s \Lambda } { m } } + 3 6 8 \frac { s \Lambda } { m } .\tag{5.3}
$$

(b) For every measurable [0, 1]-valued function $\phi$ with level-set complexity at most $s _ { ; }$

$$
| \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] | \leq 2 4 \sqrt { \frac { s \Lambda \log ( m + 1 ) } { m + 1 } \operatorname { V a r } _ { X \sim P } ( \phi ( X ) ) } + \frac { 4 1 7 s \Lambda } { m + 1 } .
$$

Proof. For $m = 0$ , part (b) holds because its last term exceeds 1. For $1 \leq m < 1 6$ , both bounds hold because their left-hand sides are at most 1 and their last terms exceed 1.

Let $m \geq 1 6$ . For (a), we first compare the empirical variance with the variance under P on an interval $I \in \mathcal { T } _ { 1 }$ . Define $z = \widehat { D } _ { i , m } ( I ) , p = P ( I )$ , and $w = \sqrt { z ( 1 - z ) }$ . Since $z , p \in [ 0 , 1 ]$ , we have

$$
z ( 1 - z ) - p ( 1 - p ) \leq | z ( 1 - z ) - p ( 1 - p ) | = | z - p | | 1 - z - p | \leq | z - p | .
$$

Since P belongs to the confidence set, we obtain

$$
w ^ { 2 } = z ( 1 - z ) \leq p ( 1 - p ) + | z - p | \leq p ( 1 - p ) + 6 \sqrt { z ( 1 - z ) \frac { \Lambda } { m } } + 3 2 \frac { \Lambda } { m } = p ( 1 - p ) + 6 w \sqrt { \frac { \Lambda } { m } } + 3 2 \frac { \Lambda } { m } .
$$

Completing the square yields $( w - 3 \sqrt { \Lambda / m } ) ^ { 2 } \leq p ( 1 - p ) + 4 1 \Lambda / m$ , and hence

$$
w \leq 3 { \sqrt { \frac { \Lambda } { m } } } + { \sqrt { p ( 1 - p ) + { \frac { 4 1 \Lambda } { m } } } } \leq { \sqrt { p ( 1 - p ) } } + ( 3 + { \sqrt { 4 1 } } ) { \sqrt { \frac { \Lambda } { m } } } \leq { \sqrt { p ( 1 - p ) } } + 1 0 { \sqrt { \frac { \Lambda } { m } } } ,\tag{5.4}
$$

where the last two inequalities use ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ for $x , y \geq 0$ and $3 + { \sqrt { 4 1 } } \leq 1 0$ , respectively. By the definition of our confidence set for $P$ and Q and (5.4), we now have

$$
\begin{array} { l } { \displaystyle | Q ( I ) - P ( I ) | \le | Q ( I ) - \widehat { D } _ { i , m } ( I ) | + | P ( I ) - \widehat { D } _ { i , m } ( I ) | \ ~ } \\ { \displaystyle \le 1 2 \sqrt { z ( 1 - z ) \frac { \Lambda } m } + 6 4 \frac { \Lambda } m \le 1 2 \sqrt { p ( 1 - p ) \frac { \Lambda } m } + 1 8 4 \frac { \Lambda } m . } \end{array}\tag{5.5}
$$

We next extend (5.5) to a union $B \in \mathcal { T } _ { s }$ . Write B as a disjoint union of intervals $I _ { 1 } , \ldots , I _ { k }$ with $k \leq s .$ , and define $p _ { r } = P ( I _ { r } )$ and $\begin{array} { r } { p = P ( B ) = \sum _ { r = 1 } ^ { k } p _ { r } } \end{array}$ . Applying (5.5) to each interval and using $p _ { r } ( 1 - p _ { r } ) \leq p _ { r }$ , we obtain

$$
| Q ( B ) - P ( B ) | \leq \sum _ { r = 1 } ^ { k } | Q ( I _ { r } ) - P ( I _ { r } ) | \leq 1 2 \sqrt { \frac { \Lambda } { m } } \sum _ { r = 1 } ^ { k } \sqrt { p _ { r } } + 1 8 4 \frac { k \Lambda } { m } \leq 1 2 \sqrt { \frac { s p \Lambda } { m } } + 1 8 4 \frac { s \Lambda } { m } .\tag{5.6}
$$

The last inequality follows from the arithmetic-quadratic mean inequality, since $\begin{array} { r } { \sum _ { r = 1 } ^ { k } \sqrt { p _ { r } } \ \leq } \end{array}$ $\begin{array} { r } { \sqrt { k \sum _ { r = 1 } ^ { k } p _ { r } } \ : = \ : \sqrt { k p } \ : \le \ : \sqrt { s p } . } \end{array}$ The complement $B ^ { c } \ = \ [ 0 , 1 ] \ \backslash \ B$ is a disjoint union of at most $s + 1$ intervals and satisfies $P ( B ^ { c } ) ~ = ~ 1 - p$ . Applying the same argument to $B ^ { c }$ and using $| Q ( B ^ { c } ) - P ( B ^ { c } ) | = | Q ( B ) - P ( B ) |$ yields

$$
| Q ( B ) - P ( B ) | \leq 1 2 \sqrt { \frac { ( s + 1 ) ( 1 - p ) \Lambda } { m } } + 1 8 4 \frac { ( s + 1 ) \Lambda } { m } .\tag{5.7}
$$

Since $p \in [ 0 , 1 ]$ , we have min $\{ p , 1 - p \} \le 2 p ( 1 - p )$ . Together with $s + 1 \leq 2 s$ , this shows that the smaller of the bounds in (5.6) and (5.7) is at most the right-hand side of (5.3).

For (b), define $B _ { y } = \{ z \in \mathrm { d o m } ( \phi ) : \phi ( z ) > y \}$ and $p _ { y } = P ( B _ { y } )$ . By assumption, each $B _ { y }$ is the intersection of dom(ϕ) with a set in $\mathcal { T } _ { s }$ . Using $\begin{array} { r } { \phi ( x ) = \int _ { 0 } ^ { 1 } \mathbf { 1 } \{ \phi ( x ) > y \} } \end{array}$ dy and applying (5.3) to the set in $\mathcal { T } _ { s }$ whose intersection with dom(ϕ) is $B _ { y }$ , which has the same probabilities under P and $Q$ since both are supported on dom(ϕ), we obtain

$$
\begin{array} { r l } {  { | \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] | = | \int _ { 0 } ^ { 1 } \bigl ( Q ( B _ { y } ) - P ( B _ { y } ) \bigr ) d y | \le \int _ { 0 } ^ { 1 } | Q ( B _ { y } ) - P ( B _ { y } ) | d y } } \\ & { \le 2 4 \sqrt { \frac { s \Lambda } { m } } \int _ { 0 } ^ { 1 } \sqrt { p _ { y } ( 1 - p _ { y } ) } d y + \frac { 3 6 8 s \Lambda } { m } . } \end{array}
$$

Applying Lemma 5.3 to $Z = \phi ( X )$ for $X \sim P$ with $\eta = ( m + 1 ) ^ { - 1 / 2 }$ , we obtain

$$
\int _ { 0 } ^ { 1 } { \sqrt { p _ { y } ( 1 - p _ { y } ) } } d y \leq { \frac { 1 } { \sqrt { m + 1 } } } + { \sqrt { \frac { \operatorname { V a r } _ { X \sim P } ( \phi ( X ) ) \log ( m + 1 ) } { 2 } } } .
$$

Substituting this estimate into the preceding bound yields

$$
\begin{array} { r l } { \displaystyle \left| \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] \right| \leq 2 4 \sqrt { \frac { s \Lambda \log ( m + 1 ) } { 2 m } \mathrm { V a r } _ { X \sim P } ( \phi ( X ) ) } + \frac { 2 4 \sqrt { s \Lambda } } { \sqrt { m ( m + 1 ) } } + \frac { 3 6 8 s \Lambda } { m } } & { } \\ { \displaystyle \leq 2 4 \sqrt { \frac { s \Lambda \log ( m + 1 ) } { m + 1 } \mathrm { V a r } _ { X \sim P } ( \phi ( X ) ) } + \frac { 3 9 2 s \Lambda } { m } } & { } \\ { \displaystyle \leq 2 4 \sqrt { \frac { s \Lambda \log ( m + 1 ) } { m + 1 } \mathrm { V a r } _ { X \sim P } ( \phi ( X ) ) } + \frac { 4 1 7 s \Lambda } { m + 1 } . } & { } \end{array}
$$

The second inequality uses 2m $\geq m + 1 , \sqrt { m ( m + 1 ) } \geq m$ , and $\sqrt { s \Lambda } \le s \Lambda$ . The last inequality follows from $( m + 1 ) / m \leq 1 7 / 1 6$ and $3 9 2 \cdot 1 7 / 1 6 = 4 1 6 . 5 \leq 4 1 7$ □

When P has few support points, the following one-sided version of Lemma $5 . 4 ( \mathrm { b } )$ holds for every function, which sufices for the optimistic updates.

Lemma 5.5. Let $k \geq 1$ and $m \geq 0$ be integers, and let $P , Q \in \mathcal { C } _ { m } ( \widehat { D } _ { i , m } )$ , where P has at most k support points. Then, for every measurable [0, 1]-valued function ϕ,

$$
\mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] \le 2 4 \sqrt { \frac { k \Lambda \log ( m + 1 ) } { m + 1 } \operatorname { V a r } _ { X \sim P } ( \phi ( X ) ) } + \frac { 4 1 7 k \Lambda } { m + 1 } .
$$

Proof. Let A be the support of P, and let $\psi ( z ) = ( 1 - \phi ( z ) ) \mathbf { 1 } \{ z \in A \}$ . Since $\psi \geq 0$ , every superlevel set of $\psi$ at a negative level is the whole domain, and every superlevel set at a level $y \geq 0$ is a subset of A and hence the intersection of the domain with a union of at most k singletons. Thus, $\psi$ has level-set complexity at most k. Since $\phi \leq 1 - \psi$ , with equality on A, and $P ( A ) = 1$ , we have

$$
\mathbb { E } _ { X \sim Q } [ \phi ( X ) ] - \mathbb { E } _ { X \sim P } [ \phi ( X ) ] \le \mathbb { E } _ { X \sim P } [ \psi ( X ) ] - \mathbb { E } _ { X \sim Q } [ \psi ( X ) ]
$$

and $\operatorname { V a r } _ { X \sim P } ( \psi ( X ) ) = \operatorname { V a r } _ { X \sim P } ( 1 - \phi ( X ) ) = \operatorname { V a r } _ { X \sim P } ( \phi ( X ) )$ . Lemma $5 . 4 ( \mathrm { b } )$ applied to $\psi$ with $s = k$ proves the claim. □

## 5.2 Proof of the Main Upper Bound

Before proving Theorem 3.2, we bound the expected sum of the conditional variances of a process with values in [0, 1] in terms of the positive parts of its drifts.

Lemma 5.6. Let $( Y _ { h } ) _ { h = 0 } ^ { H }$ be a process adapted to a filtration $( \mathcal { F } _ { h } ) _ { h = 0 } ^ { H }$ and taking values in [0, 1], and let $d _ { h } = Y _ { h } - \mathbb { E } [ Y _ { h + 1 } \mid { \mathcal { F } } _ { h } ] ~ f o r \ 0 \leq h < H$ . Then, we have

$$
\mathbb { E } \left[ \sum _ { h = 0 } ^ { H - 1 } \operatorname { V a r } ( Y _ { h + 1 } \mid \mathcal { F } _ { h } ) \right] \leq \mathbb { E } [ Y _ { H } ^ { 2 } ] - \mathbb { E } [ Y _ { 0 } ^ { 2 } ] + 2 \mathbb { E } \left[ \sum _ { h = 0 } ^ { H - 1 } ( d _ { h } ) _ { + } \right] .
$$

$H ,$ in addition, $( Y _ { h } )$ is a supermartingale, that is, $d _ { h } \ge 0$ for every $h _ { ; }$ , then the right-hand side is at most 1.

Proof. Since $\mathbb { E } [ Y _ { h + 1 } \mid { \mathcal { F } } _ { h } ] = Y _ { h } - d _ { h }$ , we have

$$
\begin{array} { r l } & { \operatorname { V a r } ( Y _ { h + 1 } \mid \mathcal { F } _ { h } ) = { \mathbb { E } } [ Y _ { h + 1 } ^ { 2 } \mid \mathcal { F } _ { h } ] - ( Y _ { h } - d _ { h } ) ^ { 2 } = { \mathbb { E } } [ Y _ { h + 1 } ^ { 2 } \mid \mathcal { F } _ { h } ] - Y _ { h } ^ { 2 } + d _ { h } ( 2 Y _ { h } - d _ { h } ) } \\ & { \qquad \le { \mathbb { E } } [ Y _ { h + 1 } ^ { 2 } \mid \mathcal { F } _ { h } ] - Y _ { h } ^ { 2 } + 2 ( d _ { h } ) _ { + } , } \end{array}
$$

where the inequality follows from $d _ { h } ( 2 Y _ { h } - d _ { h } ) \leq 2 d _ { h }$ if $d _ { h } \ge 0$ , since $Y _ { h } \le 1$ , and from $d _ { h } ( 2 Y _ { h } - d _ { h } ) \leq$ 0 if $d _ { h } < 0$ , since $Y _ { h } \ge 0$ . By the tower property, $\mathbb { E } [ \mathbb { E } [ Y _ { h + 1 } ^ { 2 } \mid \mathcal { F } _ { h } ] ] = \mathbb { E } [ Y _ { h + 1 } ^ { 2 } ]$ . Taking expectations and summing over $0 \leq h < H$ , we obtain the first claim, since the first sum telescopes:

$$
\mathbb { E } \left[ \sum _ { h = 0 } ^ { H - 1 } \operatorname { V a r } ( Y _ { h + 1 } \mid \mathcal { F } _ { h } ) \right] \leq \sum _ { h = 0 } ^ { H - 1 } \left( \mathbb { E } [ Y _ { h + 1 } ^ { 2 } ] - \mathbb { E } [ Y _ { h } ^ { 2 } ] \right) + 2 \mathbb { E } \left[ \sum _ { h = 0 } ^ { H - 1 } ( d _ { h } ) _ { + } \right] = \mathbb { E } [ Y _ { H } ^ { 2 } ] - \mathbb { E } [ Y _ { 0 } ^ { 2 } ] + 2 \mathbb { E } \left[ \sum _ { h = 0 } ^ { H - 1 } ( d _ { h } ) _ { + } \right] .
$$

If $d _ { h } \ge 0$ for every $h ,$ the tower property yields $\mathbb { E } [ ( d _ { h } ) _ { + } ] = \mathbb { E } [ d _ { h } ] = \mathbb { E } [ Y _ { h } ] - \mathbb { E } [ Y _ { h + 1 } ]$ , and hence the last sum is $\mathbb { E } [ Y _ { 0 } - Y _ { H } ]$ . Let $\mu = \mathbb { E } [ Y _ { 0 } ]$ . Since $Y _ { H } \in [ 0 , 1 ]$ , we have $Y _ { H } ^ { 2 } \le Y _ { H }$ , and $\mathbb { E } [ Y _ { 0 } ^ { 2 } ] \ge \mu ^ { 2 }$ by Jensen’s inequality. Hence,

$$
\begin{array} { r } { \mathbb { E } [ Y _ { H } ^ { 2 } ] - \mathbb { E } [ Y _ { 0 } ^ { 2 } ] + 2 \mathbb { E } [ Y _ { 0 } - Y _ { H } ] \le \mathbb { E } [ Y _ { H } ] - \mu ^ { 2 } + 2 \mu - 2 \mathbb { E } [ Y _ { H } ] = 1 - ( 1 - \mu ) ^ { 2 } - \mathbb { E } [ Y _ { H } ] \le 1 . } \end{array}
$$

We are now ready to prove Theorem 3.2.

Proof of Theorem 3.2. Let $G _ { t }$ be the event that $D _ { i } \in \mathcal { C } _ { i , t }$ for every $i \in [ n ]$ . This event is determined by the samples collected before round t. We call round t good if $G _ { t }$ occurs. By Lemma 5.2, we have $\mathbb { P } ( G _ { t } ^ { c } ) \leq \delta$ for every t. Since $0 \leq \mathrm { R e g } _ { t } \leq 1$ , we have

$$
R _ { T } = \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathrm { R e g } _ { t } ] \leq \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \mathrm { R e g } _ { t } ] + \delta T .\tag{5.8}
$$

We first bound the expected regret in a single good round. Fix the history before round t for which $G _ { t }$ holds. In this part of the proof, expectations are conditional on this history. Since the value vector $X ^ { t } = ( X _ { 1 } ^ { t } , \ldots , X _ { n } ^ { t } ) \sim D$ drawn in round t is independent of the history, these conditional expectations are taken over $X ^ { t } \sim D$ . At a leaf, the optimistic and true values are equal. Since $D _ { i } \in \mathcal { C } _ { i , t }$ at every revelation edge, backward induction shows tha

$$
\bar { g } _ { u } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) \geq g _ { u } ( x _ { \mathrm { o b s } ( u ) } )\tag{5.9}
$$

at every node. The supremum over $Q$ is at least the expectation under $D _ { i }$ , and taking the maximum over the same feasible actions preserves the inequality.

Execute the policy $\sigma ^ { t }$ selected by Algorithm 1 in round t under $D ,$ and let $u _ { t , 0 } , u _ { t , 1 } , \dotsc , u _ { t , h } .$ ∗ be the visited nodes, where $u _ { t , 0 } = r$ and $u _ { t , h ^ { * } } = \ell ^ { t }$ . Since there are at most n revelation edges and at most one stopping edge on the path, we have $h ^ { * } \leq n + 1$ . For $0 \leq h \leq h ^ { * }$ , let $\mathcal { F } _ { t , h }$ be the σ-algebra generated by the visited nodes and observed values up to arrival at $u _ { t , h }$ . For $h ^ { * } < h \leq n + 1$ , set $u _ { t , h } = \ell ^ { t }$ and $\mathcal { F } _ { t , h } = \mathcal { F } _ { t , h ^ { * } }$ . Then, $\mathcal { F } _ { t , 0 } \subseteq \cdots \subseteq \mathcal { F } _ { t , n + 1 }$ is a filtration. Define

$$
Y _ { t , h } = \bar { g } _ { u _ { t , h } } ^ { t } ( X _ { \mathrm { o b s } ( u _ { t , h } ) } ^ { t } ) , \qquad 0 \leq h \leq n + 1 .
$$

In particular, $Y _ { t , 0 } = \bar { g } _ { r } ^ { t } ( X _ { \mathrm { o b s } ( r ) } ^ { t } )$ and $Y _ { t , h } = f ( \sigma ^ { t } , X ^ { t } )$ for $h ^ { * } \leq h \leq n + 1$ . For $0 \leq h \leq n$ , define

$$
d _ { t , h } = Y _ { t , h } - \mathbb { E } [ Y _ { t , h + 1 } \mid { \mathcal F } _ { t , h } ] , \qquad w _ { t , h } = \operatorname { V a r } ( Y _ { t , h + 1 } \mid { \mathcal F } _ { t , h } ) .
$$

Suppose $h < h ^ { * }$ and the chosen edge $( u _ { t , h } , u _ { t , h + 1 } ) = ( u , v )$ reveals coordinate $i ,$ with preceding observations $x _ { \mathrm { o b s } ( u ) }$ . The edge is chosen by (3.5), while the unobserved coordinate still follows $D _ { i }$ conditional on $\mathcal { F } _ { t , h }$ . Therefore, since $D _ { i } \in \mathcal { C } _ { i , t }$

$$
\begin{array} { r l } & { d _ { t , h } = \underset { Q \in \mathscr { C } _ { i , t } } { \operatorname* { s u p } } ~ \mathbb { E } _ { X _ { i } \sim Q } [ \bar { g } _ { v } ^ { t } ( X _ { i } , x _ { \mathrm { o b s } ( u ) } ) ] - \mathbb { E } _ { X _ { i } \sim D _ { i } } [ \bar { g } _ { v } ^ { t } ( X _ { i } , x _ { \mathrm { o b s } ( u ) } ) ] \geq 0 , } \\ & { w _ { t , h } = \mathrm { V a r } _ { X _ { i } \sim D _ { i } } ( \bar { g } _ { v } ^ { t } ( X _ { i } , x _ { \mathrm { o b s } ( u ) } ) ) . } \end{array}
$$

If $h < h ^ { * }$ and the chosen edge is a stopping edge, its reward equals $Y _ { t , h }$ , and hence $Y _ { t , h + 1 } = Y _ { t , h }$ This equality also holds for $h ^ { * } \leq h \leq n$ by definition. Thus, ${ d _ { t , h } } = { w _ { t , h } } = 0$ in both cases. Since the optimistic DP preserves the range [0, 1], every $Y _ { t , h }$ lies in [0, 1]. Since $d _ { t , h } \geq 0$ for every $h ,$ the sequence $Y _ { t , 0 } , \ldots , Y _ { t , n + 1 }$ is a supermartingale with respect to $( \mathcal { F } _ { t , h } ) _ { h } ,$ and Lemma 5.6 yields

$$
\mathbb { E } \left[ \sum _ { h = 0 } ^ { n } w _ { t , h } \right] \leq 1 .\tag{5.10}
$$

By the definition of $d _ { t , h }$ and the tower property, we have $\begin{array} { r } { \mathbb { E } [ \sum _ { h = 0 } ^ { n } d _ { t , h } ] = \mathbb { E } [ Y _ { t , 0 } ] - f ( \sigma ^ { t } \mid D ) } \end{array}$ By (5.9) and (2.3), $\mathbb { E } [ Y _ { t , 0 } ] \ge \mathrm { O P T } ( D )$ , and hence

$$
\mathrm { R e g } _ { t } \leq \mathbb { E } \left[ \sum _ { h = 0 } ^ { n } d _ { t , h } \right] .\tag{5.11}
$$

Let ${ \bar { s } } _ { i } = \operatorname* { m i n } \{ s _ { i } , k _ { i } \}$ , and consider a stage $h \ < \ h ^ { * }$ at which the chosen edge $( u , v )$ reveals coordinate i, where $D _ { i } \in \mathcal { C } _ { i , t }$ since $G _ { t }$ holds. If $s _ { i } ~ \leq ~ k _ { i }$ , then $z \mapsto \bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } )$ has level-set complexity at most $s _ { i }$ by Assumption 3.1, and we apply Lemma $5 . 4 ( \mathrm { b } )$ to this function. Otherwise, we apply Lemma 5.5 with $k = k _ { i }$ . In both cases, we take $m = N _ { i , t } , P = D _ { i }$ , and every $Q \in \mathcal { C } _ { i , t } .$ and using log $( N _ { i , t } + 1 ) \leq \log T$ , we obtain

$$
0 \leq d _ { t , h } \leq 2 4 \sqrt { \frac { \bar { s } _ { i } \Lambda \log T } { N _ { i , t } + 1 } } w _ { t , h } + \frac { 4 1 7 \bar { s } _ { i } \Lambda } { N _ { i , t } + 1 } .\tag{5.12}
$$

We then bound the cumulative expected regret. From now on, expectations also average over the histories before the rounds. Let $I _ { i , t }$ indicate whether coordinate i is revealed on an edge in round t. If $I _ { i , t } = 1$ and coordinate i is revealed at stage $h ,$ define $\widetilde { w } _ { i , t } = w _ { t , h }$ . If $I _ { i , t } = 0$ , set $\widetilde { w } _ { i , t } = 0$ . Since each coordinate is revealed at most once per round and $w _ { t , h } = 0$ at stages without a revelation, we have $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \widetilde { w } _ { i , t } I _ { i , t } = \sum _ { h = 0 } ^ { n } w _ { t , h } } \end{array}$ . For a coordinate $i \not \in$ obs(r), every revelation occurs on an edge, and $N _ { i , t + 1 } = N _ { i , t } + I _ { i , t }$ for every realization. Hence, at the k-th round with $I _ { i , t } = 1$ we have $N _ { i , t } + 1 = k$ . For $i \in \mathrm { o b s } ( r )$ , we have $I _ { i , t } = 0$ . Thus, for every coordinate i and every realization,

$$
\sum _ { t = 1 } ^ { T } \frac { I _ { i , t } } { N _ { i , t } + 1 } \leq \sum _ { k = 1 } ^ { T } \frac { 1 } { k } \leq 1 + \log T .\tag{5.13}
$$

Let $\Delta$ and W be the cumulative gaps and conditional variances in good rounds,

$$
\Delta = \sum _ { t = 1 } ^ { T } \mathbb { E } \bigg [ \mathbf { 1 } _ { G _ { t } } \sum _ { h = 0 } ^ { n } d _ { t , h } \bigg ] , \qquad W = \sum _ { t = 1 } ^ { T } \mathbb { E } \bigg [ \mathbf { 1 } _ { G _ { t } } \sum _ { h = 0 } ^ { n } w _ { t , h } \bigg ] .
$$

By (5.11), we have $\begin{array} { r } { \sum _ { t } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \mathrm { R e g } _ { t } ] \leq \Delta } \end{array}$ . Since (5.10) holds for every history before round t for which $G _ { t }$ occurs, we have

$$
W = \sum _ { t = 1 } ^ { T } \mathbb { E } \bigg [ \mathbf { 1 } _ { G _ { t } } \sum _ { h = 0 } ^ { n } w _ { t , h } \bigg ] \leq \sum _ { t = 1 } ^ { T } \mathbb { P } ( G _ { t } ) \leq T .\tag{5.14}
$$

We next bound the cumulative contribution of the square-root term in (5.12). We have

$$
\begin{array} { r l r } {  { \mathbb { E } \biggl [ \sum _ { t , i } { \mathbf { 1 } } _ { G _ { t } } I _ { i , t } \sqrt { \frac { \bar { s } _ { i } \widetilde w _ { i , t } } { N _ { i , t } + 1 } } \biggr ] \leq \mathbb { E } \biggl [ \sqrt { \sum _ { t , i } \frac { \bar { s } _ { i } \mathbf { 1 } _ { G _ { t } } I _ { i , t } } { N _ { i , t } + 1 } } \sqrt { \sum _ { t , i } { \mathbf { 1 } } _ { G _ { t } } I _ { i , t } \widetilde w _ { i , t } } \biggr ] } } \\ & { } & { \leq \sqrt { \mathbb { E } \biggl [ \sum _ { t , i } \frac { \bar { s } _ { i } { \mathbf { 1 } } _ { G _ { t } } I _ { i , t } } { N _ { i , t } + 1 } \biggr ] \mathbb { E } \biggl [ \sum _ { t , i } { \mathbf { 1 } } _ { G _ { t } } I _ { i , t } \widetilde w _ { i , t } \biggr ] } \leq \sqrt { S ( 1 + \log T ) W } , } \end{array}\tag{5.15}
$$

where the first inequality follows from the Cauchy–Schwarz inequality applied to the sum over $( t , i )$ for each realization, the second follows from the Cauchy–Schwarz inequality applied to the expectation of the product, and the third follows from (5.13), $\begin{array} { r } { \mathbf { 1 } _ { G _ { t } } \leq 1 , S = \sum _ { i } \bar { s } _ { i } } \end{array}$ , and $\begin{array} { r } { \sum _ { i } \widetilde { w } _ { i , t } I _ { i , t } = } \end{array}$ $\sum _ { h } w _ { t , h }$ . Let $\Lambda ^ { \prime } = \Lambda \log T ( 1 + \log T )$ . Combining (5.12), (5.13), and (5.15), we obtain

$$
\begin{array} { r l } & { \displaystyle \Delta \leq 2 4 \sqrt { \Lambda \log T } \mathbb { E } \left[ \sum _ { t , i } { \mathbf 1 } _ { G t } I _ { i , t } \sqrt { \frac { \bar { s } _ { i } \widetilde w _ { i , t } } { N _ { i , t } + 1 } } \right] + 4 1 7 \Lambda \mathbb { E } \left[ \sum _ { t , i } \frac { \bar { s } _ { i } { \mathbf 1 } _ { G t } I _ { i , t } } { N _ { i , t } + 1 } \right] } \\ & { \quad \leq 2 4 \sqrt { S \Lambda ^ { \prime } W } + 4 1 7 S \Lambda ( 1 + \log T ) . } \end{array}\tag{5.16}
$$

By (5.14), we have $\begin{array} { r } { \sum _ { t } \mathbb { E } [ { \bf 1 } _ { G _ { t } } \mathrm { R e g } _ { t } ] \leq \Delta \leq 2 4 \sqrt { S T \Lambda ^ { \prime } } + 4 1 7 S \Lambda ( 1 + \log T ) } \end{array}$ . Adding δT from (5.8) proves (3.6) with $c = 4 1 7$ □

Proposition 3.5 follows from (5.16) with a sharper bound on W than (5.14).

Proof of Proposition 3.5. Fix a history before round t for which $G _ { t }$ holds, and take expectations conditional on it, as in the proof of Theorem 3.2. Let $\textstyle \Delta _ { t } = \operatorname { \mathbb { E } } [ \sum _ { h = 0 } ^ { n } d _ { t , h } ]$ , which equals $\mathbb { E } [ Y _ { t , 0 } - Y _ { t , n + 1 } ]$ by the tower property. By optimism, $\mathbb { E } [ Y _ { t , 0 } ^ { 2 } ] \ge ( \mathbb { E } [ Y _ { t , 0 } ] ) ^ { 2 } \ge p ^ { 2 }$ , and since $Y _ { t , n + 1 } = f ( \sigma ^ { t } , X ^ { t } )$ , we have $\mathbb { E } [ Y _ { t , n + 1 } ^ { 2 } ] \leq \mathbb { E } [ Y _ { t , n + 1 } ] = f ( \sigma ^ { t } \mid D ) \leq p .$ Since $d _ { t , h } \geq 0$ , we have $( d _ { t , h } ) _ { + } = d _ { t , h }$ , and hence the first claim of Lemma 5.6 yields

$$
\mathbb { E } \left[ \sum _ { h = 0 } ^ { n } w _ { t , h } \right] \leq p - p ^ { 2 } + 2 \Delta _ { t } = p ( 1 - p ) + 2 \Delta _ { t } .
$$

Since $\begin{array} { r } { \Delta \mathbf { \Psi } = \sum _ { t } \mathbb { E } \big [ \mathbf { 1 } _ { G _ { t } } \Delta _ { t } \big ] } \end{array}$ , averaging this bound over good histories and summing over t yields $W \le T p ( 1 - p ) + 2 \Delta$ . Substituting this bound into (5.16) yields

$$
\Delta \le 2 4 \sqrt { S \Lambda ^ { \prime } ( T p ( 1 - p ) + 2 \Delta ) } + 4 1 7 S \Lambda ( 1 + \log T ) .
$$

Using ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ for $x , y \geq 0$ and $2 4 \sqrt { 2 S \Lambda ^ { \prime } \Delta } \leq \Delta / 2 + 5 7 6 S \Lambda ^ { \prime }$ , we obtain

$$
\begin{array} { l } { \Delta \le 2 4 \sqrt { S T p ( 1 - p ) \Lambda ^ { \prime } } + 2 4 \sqrt { 2 S \Lambda ^ { \prime } \Delta } + 4 1 7 S \Lambda ( 1 + \log T ) } \\ { \le 2 4 \sqrt { S T p ( 1 - p ) \Lambda ^ { \prime } } + \displaystyle \frac { \Delta } { 2 } + 5 7 6 S \Lambda ^ { \prime } + 4 1 7 S \Lambda ( 1 + \log T ) . } \end{array}
$$

Subtracting $\Delta / 2$ from both sides and multiplying by 2 yields

$$
\Delta \leq 4 8 \sqrt { S T p ( 1 - p ) \Lambda ^ { \prime } } + 1 1 5 2 S \Lambda ^ { \prime } + 8 3 4 S \Lambda ( 1 + \log T ) .
$$

Finally, $R _ { T } \le \Delta + \delta T$ by (5.8) and $\begin{array} { r } { \sum _ { t } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \mathrm { R e g } _ { t } ] \leq \Delta } \end{array}$ , and $\delta T < 1$ for δ = 1/(T + 1).

Proposition 3.4 follows from the same calculation, with the first claim of Lemma 5.6 in place of the second.

Proof of Proposition $\ 3 . 4 \cdot$ . We treat (a) as the case $\alpha = 1$ and bound $R _ { T } ^ { ( \alpha ) }$ . Let $G _ { t }$ be the event that $D _ { i } \in \mathcal { C } _ { i , t }$ for every $i ,$ as in the proof of Theorem 3.2.

Fix a history before round t for which $G _ { t }$ holds, take expectations conditional on it, and let $Q = Q ^ { t }$ . If $\sigma ^ { t }$ is randomized, we also condition on the deterministic policy drawn from it. Define the visited nodes $u _ { t , h }$ , the last stage $h ^ { * }$ , and the filtration $( \mathcal { F } _ { t , h } ) _ { h }$ as in the proof of Theorem 3.2. In (a), let $Y _ { t , h } = g _ { u _ { t , h } } ^ { Q } ( X _ { \mathrm { o b s } ( u _ { t , h } ) } ^ { t } )$ . In (b), let $Y _ { t , h }$ be the expected reward of continuing the deterministic policy from $u _ { t , h }$ when the unobserved coordinates follow $Q .$ . In both cases, $Y _ { t , h } \in [ 0 , 1 ]$ and $Y _ { t , h } =$ $f _ { \ell ^ { t } } \bigl ( X _ { \mathrm { o b s } ( \ell ^ { t } ) } ^ { t } \bigr )$ for $h ^ { * } \leq h \leq n + 1$ . Define $d _ { t , h }$ and $w _ { t , h }$ as in the proof of Theorem 3.2.

Let $h < h ^ { * }$ . If the chosen edge is a stopping edge, then $Y _ { t , h + 1 } = Y _ { t , h }$ , in (a) since the chosen child maximizes $h ^ { Q }$ and in (b) by the definition of $Y _ { t , h }$ . This equality also holds for $h ^ { * } \leq h \leq n ,$ and hence ${ d _ { t , h } } = { w _ { t , h } } = 0$ in these cases. If the chosen edge $( u , v )$ reveals coordinate $i ,$ let $\phi ( z )$ be the value at v for $x _ { i } = z$ , which is $g _ { v } ^ { Q } ( z , x _ { \mathrm { o b s } ( u ) } )$ in (a). Then, $Y _ { t , h } = \mathbb { E } _ { X \sim Q _ { i } } [ \phi ( X ) ]$ for the same reasons, whereas $Y _ { t , h + 1 } = \phi ( X _ { i } ^ { t } )$ and $X _ { i } ^ { t }$ follows $D _ { i }$ conditional on $\mathcal { F } _ { t , h }$ . Hence,

$$
d _ { t , h } = \mathbb { E } _ { X \sim Q _ { i } } [ \phi ( X ) ] - \mathbb { E } _ { X \sim D _ { i } } [ \phi ( X ) ] , \qquad w _ { t , h } = \mathrm { V a r } _ { X \sim D _ { i } } ( \phi ( X ) ) .
$$

Unlike in the proof of Theorem 3.2, $d _ { t , h }$ can be negative, since $Q _ { i }$ need not maximize the expectation of $\phi$ over $\mathcal { C } _ { i , t } .$ and hence $( Y _ { t , h } ) _ { h }$ need not be a supermartingale.

We bound the regret in this round by the positive parts of the gaps. Since obs $( r ) = \emptyset$ , the expectation of $Y _ { t , 0 }$ over the randomness of $\sigma ^ { t }$ is $f ( \sigma ^ { t } \mid Q )$ , which is at least $\alpha { \mathrm { O P T } } ( Q ) \geq \alpha { \mathrm { O P T } } ( D ) -$ $1 / T$ since $D _ { i } \in \mathcal { C } _ { i , t }$ for every i. Taking expectations also over this randomness, the tower property yields $\begin{array} { r } { \mathbb { E } [ Y _ { t , 0 } ] - \mathbb { E } [ f ( \sigma ^ { t } , X ^ { t } ) ] = \mathbb { E } [ \sum _ { h = 0 } ^ { n } d _ { t , h } ] } \end{array}$ , and hence

$$
\alpha \operatorname { O P T } ( D ) - \mathbb { E } [ f ( \sigma ^ { t } , X ^ { t } ) ] \leq { \frac { 1 } { T } } + \mathbb { E } \left[ \sum _ { h = 0 } ^ { n } ( d _ { t , h } ) _ { + } \right] .\tag{5.17}
$$

Since $Y _ { t , h } \in [ 0 , 1 ]$ , the first claim of Lemma 5.6 yields

$$
\mathbb { E } \left[ \sum _ { h = 0 } ^ { n } w _ { t , h } \right] \leq \mathbb { E } [ Y _ { t , n + 1 } ^ { 2 } ] - \mathbb { E } [ Y _ { t , 0 } ^ { 2 } ] + 2 \mathbb { E } \left[ \sum _ { h = 0 } ^ { n } ( d _ { t , h } ) _ { + } \right] \leq 1 + 2 \mathbb { E } \left[ \sum _ { h = 0 } ^ { n } ( d _ { t , h } ) _ { + } \right] .\tag{5.18}
$$

We next bound each positive part, using $Q _ { i } , D _ { i } \in \mathcal { C } _ { i , t }$ and $m = N _ { i , t }$ . In (a), if $s _ { i } \leq k _ { i }$ , then $\phi$ has level-set complexity at most $s _ { i }$ by the condition in (a), and Lemma $5 . 4 ( \mathrm { b } )$ with $P = D _ { i }$ applies. Otherwise, Lemma 5.5 with $k = k _ { i }$ and $P = D _ { i }$ applies. In (b), Lemma 5.5 with $k = k _ { i }$ and $P = D _ { i }$ applies to every $\phi$ . Using log $( N _ { i , t } + 1 ) \leq$ log $T$ , we obtain (5.12) with $( d _ { t , h } ) _ { + }$ in place of $d _ { t , h } ,$ , where $\bar { s } _ { i } = \operatorname* { m i n } \{ s _ { i } , k _ { i } \}$ in (a) and $\bar { s } _ { i } = k _ { i }$ in (b).

Finally, we sum over the rounds, with expectations also averaging over the histories. Let $\Delta$ and W be the sums over t of $\mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \sum _ { h } ( d _ { t , h } ) _ { + } ]$ and $\mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \sum _ { h } w _ { t , h } ]$ , respectively. By (5.18), we have $W \leq T + 2 \Delta$ . The calculation in (5.15) and $( 5 . 1 6 )$ , with $( d _ { t , h } ) _ { + }$ <sub>+</sub> in place of $d _ { t , h }$ , yields $\Delta \le 2 4 \sqrt { S \Lambda ^ { \prime } ( T + 2 \Delta ) } + 4 1 7 S \Lambda ( 1 + \log T )$ , with K in place of S in (b). Using ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ and $2 4 \sqrt { 2 S \Lambda ^ { \prime } \Delta } \leq \Delta / 2 + 5 7 6 S \Lambda ^ { \prime }$ as in the proof of Proposition 3.5, we obtain $\Delta \leq 4 8 \sqrt { S T \Lambda ^ { \prime } } +$ $1 1 5 2 S \Lambda ^ { \prime } + 8 3 4 S \Lambda ( 1 + \log T )$ . By (5.17), the good rounds contribute at most $1 + \Delta$ to $R _ { T } ^ { ( \alpha ) }$ . Each other round contributes at most 1 and occurs with probability at most δ by Lemma 5.2. Hence, $R _ { T } ^ { ( \alpha ) } \leq 1 + \Delta + \delta T$ □

## 5.3 Extension to Immediate Rewards

The proof of Theorem 3.2 in Section 5.2 uses Assumption 3.1 only in the local error bound (5.12), and we prove Theorem 3.7 by replacing this bound with one that treats the immediate reward and the residual value separately.

Proof of Theorem 3.7. Consider a round in which $G _ { t }$ holds and a stage h at which the chosen edge $\boldsymbol { e } = \left( u , v \right)$ reveals coordinate i after observations $x _ { \mathrm { o b s } ( u ) }$ Let $\kappa = \kappa _ { e } \big ( x _ { \mathrm { o b s } ( u ) } \big )$ and $F _ { \ell } = F _ { e , \ell }$ be as in condition (a) of Theorem 3.7, and write $\chi ( z ) = \chi _ { e } ( z , x _ { \mathrm { o b s } ( u ) } )$ . Let $\phi ( z )$ be the optimistic continuation value at v for $x _ { i } = z$ , defined by (3.3) and (3.4) on the subtree of v with $F _ { \ell }$ in place of $f _ { \ell }$ . Since $\kappa + \chi ( x _ { i } )$ is determined by the observations at v, and expectations, suprema, and maxima commute with adding a constant, we have $\bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } ) = \kappa + \chi ( z ) + \phi ( z )$ . The function $\phi$ takes values in [0, 1] since every $F _ { \ell }$ does, and it is nonincreasing by condition (b), since expectations, suprema, and maxima preserve monotonicity. Hence, $\phi$ has level-set complexity at most $1 \leq s _ { i }$

$w _ { t , h }$ be as in Section 5.2, let $\sigma _ { \phi }$ and $\sigma _ { \chi }$ be the standard deviations of $\phi ( X _ { i } )$ and $\chi ( X _ { i } )$ for $X _ { i } \sim D _ { i }$ , and let $\nu _ { t , h } = \mathbb { E } _ { X _ { i } \sim D _ { i } } [ \chi ( X _ { i } ) ]$ . Since $\phi = \bar { g } _ { v } ^ { t } - \kappa - \chi$ , the triangle inequality for standard deviations implies $\sigma _ { \phi } \leq \sqrt { w _ { t , h } } + \sigma _ { \chi }$ . Moreover, $\sigma _ { \chi } ^ { 2 } \leq \nu _ { t , h }$ since $\chi$ takes values in $[ 0 , 1 ]$ . Combining these bounds with the arithmetic-quadratic mean inequality applied to $\sqrt { w _ { t , h } } , \sqrt { \nu _ { t , h } }$ , and $\sqrt { \nu _ { t , h } }$ , we obtain

$$
\sigma _ { \phi } + \sigma _ { \chi } \leq \sqrt { w _ { t , h } } + 2 \sqrt { \nu _ { t , h } } \leq \sqrt { 3 ( w _ { t , h } + 2 \nu _ { t , h } ) } .
$$

Since κ does not depend on $X _ { i } ,$ it cancels in $d _ { t , h } ,$ , and hence

$$
d _ { t , h } \leq \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \left( \mathbb { E } _ { X _ { i } \sim Q } [ \chi ( X _ { i } ) ] - \mathbb { E } _ { X _ { i } \sim D _ { i } } [ \chi ( X _ { i } ) ] \right) + \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \left( \mathbb { E } _ { X _ { i } \sim Q } [ \phi ( X _ { i } ) ] - \mathbb { E } _ { X _ { i } \sim D _ { i } } [ \phi ( X _ { i } ) ] \right) .
$$

Applying Lemma $5 . 4 ( \mathrm { b } )$ to $\chi$ and $\phi$ with $m = N _ { i , t }$ and $s = s _ { i }$ , and then the preceding bound on $\sigma _ { \chi } + \sigma _ { \phi }$ , we obtain

$$
0 \leq d _ { t , h } \leq 2 4 \sqrt { \frac { s _ { i } \Lambda \log T } { N _ { i , t } + 1 } } \left( \sigma _ { \chi } + \sigma _ { \phi } \right) + \frac { 8 3 4 s _ { i } \Lambda } { N _ { i , t } + 1 } \leq 2 4 \sqrt { \frac { 3 s _ { i } \Lambda \log T } { N _ { i , t } + 1 } } \left( w _ { t , h } + 2 \nu _ { t , h } \right) + \frac { 8 3 4 s _ { i } \Lambda } { N _ { i , t } + 1 } .\tag{5.19}
$$

If $D _ { i }$ has $k _ { i } < s _ { i }$ support points, then Lemma 5.5 for $\chi$ with $k = k _ { i }$ and Lemma $5 . 4 ( \mathrm { b } )$ for $\phi$ with $s = 1 \leq k _ { i }$ yield (5.19) with $k _ { i }$ in place of $s _ { i } .$ , which proves the remark after Theorem 3.7.

At a stage that reveals no coordinate, set $\nu _ { t , h } = 0$ . Then, $\nu _ { t , h }$ is the conditional expectation, given $\mathcal { F } _ { t , h }$ , of the immediate reward collected at stage h. By condition (c), these immediate rewards sum to at most 1 along the executed path, and hence $\mathbb { E } [ \sum _ { h } \nu _ { t , h } ] \le 1$ . Together with (5.10), we obtain $\mathbb { E } [ \sum _ { h } ( w _ { t , h } + 2 \nu _ { t , h } ) ] \le 3$ . Define $\widetilde { \nu } _ { i , t }$ from $\nu _ { t , h }$ in the same way as $\widetilde { w } _ { i , t }$ from $w _ { t , h }$ . Let $\Lambda ^ { \prime } = \Lambda$ log T(1 + log T), $\begin{array} { r } { \Delta = \sum _ { t } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \sum _ { h } d _ { t , h } ] } \end{array}$ , and $\begin{array} { r } { W ^ { \prime } = \sum _ { t } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \sum _ { h } ( w _ { t , h } + 2 \nu _ { t , h } ) ] } \end{array}$ , which is at most $3 T$ by the preceding bound. The calculation in (5.15) and (5.16), with (5.19) in place of (5.12), $s _ { i }$ in place of $\bar { s } _ { i } ,$ and $\widetilde { w } _ { i , t } + 2 \widetilde { \nu } _ { i , t }$ in place of $\widetilde { w } _ { i , t }$ , yields

$$
\Delta \leq 2 4 \sqrt { 3 S \Lambda ^ { \prime } W ^ { \prime } } + 8 3 4 S \Lambda ( 1 + \log T ) .\tag{5.20}
$$

Since $W ^ { \prime } \leq 3 T$ , we have $\begin{array} { r } { \sum _ { t } \mathbb { E } [ \mathbf { 1 } _ { G _ { t } } \mathrm { R e g } _ { t } ] \leq \Delta \leq 7 2 \sqrt { S T \Lambda ^ { \prime } } + 8 3 4 S \Lambda ( 1 + \log T ) } \end{array}$ , and adding δT from (5.8) proves the claim with $c = 8 3 4$ □

Proposition 3.8 follows from (5.20) with a sharper bound on $W ^ { \prime }$ than $W ^ { \prime } \leq 3 T$

Proof of Proposition 3.8. Fix a history before round t for which $G _ { t }$ holds, and let $\Delta _ { t }$ be as in the proof of Proposition 3.5, which shows $\begin{array} { r } { \mathbb { E } [ \sum _ { h } w _ { t , h } ] \le p ( 1 - p ) + 2 \Delta _ { t } } \end{array}$ without using Assumption 3.1. By the additional condition, the immediate rewards along the executed path sum to at most $f ( \sigma ^ { t } , X ^ { t } )$ ， and hence $\begin{array} { r } { \mathbb E [ \sum _ { h } \nu _ { t , h } ] \le f ( \sigma ^ { t } \mid D ) \le p . } \end{array}$ . Thus, $\begin{array} { r } { \mathbb E [ \sum _ { h } ( w _ { t , h } + 2 \nu _ { t , h } ) ] \le 3 p + 2 \Delta _ { t } } \end{array}$ . Averaging over good histories and summing over t yields $W ^ { \prime } \leq 3 T p + 2 \Delta$ , and substituting this bound into (5.20) yields

$$
\Delta \leq 2 4 \sqrt { 3 S \Lambda ^ { \prime } ( 3 T p + 2 \Delta ) } + 8 3 4 S \Lambda ( 1 + \log T ) .
$$

Absorbing the term proportional to $\sqrt { \Delta }$ as in the proof of Proposition 3.5 and using $R _ { T } \le \Delta + \delta T$ yields $R _ { T } = \widetilde { O } ( \sqrt { S T p } + S )$ □

## 6 Lower Bound

We prove Theorem 3.6 with an instance in which each coordinate i difers from a special label with probability $s _ { i } / ( 2 S )$ and the learner must predict the signs carried by these rare values. The use of rare values is inspired by a lower-bound construction of Guo, Huang, Tang, and Zhang [GHTZ21, Theorem $7 ]$ for the sample complexity of learning over product distributions. We embed the instance into a decision tree and order the labels such that the level-set complexity of coordinate i is at most $s _ { i }$

## 6.1 Construction

Fix integers $T , s _ { 1 } , \ldots , s _ { n } \geq 1$ , let $S = s _ { 1 } + \cdot \cdot \cdot + s _ { n }$ , and set

$$
\varepsilon = { \frac { 1 } { 4 } } \operatorname* { m i n } \left\{ 1 , \ { \sqrt { \frac { S } { T } } } \right\} .\tag{6.1}
$$

Let $J = \{ ( i , j ) : i \in [ n ] , j \in [ s _ { i } ] \}$ , which has S elements. An unknown sign vector $\pmb \theta = ( \theta _ { i j } ) _ { ( i , j ) \in J } \in$ $\{ - 1 , + 1 \} ^ { J }$ is fixed throughout all T rounds.

Coordinates and distributions. Coordinate i takes values in a set of $2 s _ { i } + 1$ labels, which consists of a special label ⊥ and the pairs $( j , y ) \in [ s _ { i } ] \times \{ - 1 , + 1 \}$ . Its distribution is

$$
D _ { i } ^ { \theta } ( \perp ) = 1 - \frac { s _ { i } } { 2 S } , \qquad D _ { i } ^ { \theta } ( j , y ) = \frac { 1 + y \varepsilon \theta _ { i j } } { 4 S } .
$$

Equivalently, $X _ { i } = \perp$ with probability $1 - s _ { i } / ( 2 S )$ , and otherwise $X _ { i } = ( j , y )$ , where the column $j$ is drawn uniformly from $[ s _ { i } ]$ and then the sign y is $\theta _ { i j }$ with probability $( 1 + \varepsilon ) / 2$ and $- \theta _ { i j }$ otherwise. Since $\varepsilon \le 1 / 4$ , all $2 s _ { i } + 1$ labels have positive probability. Let $\begin{array} { r } { D ^ { \theta } = \prod _ { i = 1 } ^ { n } D _ { i } ^ { \theta } } \end{array}$ . We identify the labels

$$
( 1 , - 1 ) , ~ \bot , ~ ( 1 , + 1 ) , ~ ( 2 , - 1 ) , ~ ( 2 , + 1 ) , ~ . ~ . ~ , ~ ( s _ { i } , - 1 ) , ~ ( s _ { i } , + 1 )
$$

with the points $0 , 1 / ( 2 s _ { i } ) , 2 / ( 2 s _ { i } ) , \ldots , 1$ of $[ 0 , 1 ]$ , respectively. The label ⊥ thus corresponds to the point $1 / ( 2 s _ { i } )$ , and only the order of the points matters for level-set complexity. Placing ⊥ between the two labels of column 1 keeps the level-set complexity of coordinate i at most $s _ { i } ,$ , as we verify after describing the rewards.

Decision tree and rewards. There are no initial observations. At the root, the learner chooses a prediction vector $\begin{array} { r } { \widehat { \pmb { y } } = ( \widehat { y } _ { i j } ) _ { ( i , j ) \in J } \in \{ - 1 , + 1 \} ^ { J } } \end{array}$ , where $\widehat { y } _ { i j }$ predicts the sign y when $X _ { i } = ( j , y )$ The edge chosen at the root reveals $X _ { 1 }$ , and a chain of edges then reveals $X _ { 2 } , \ldots , X _ { n }$ in order, where each intermediate node has a single child and the edge revealing $X _ { n }$ leads to a leaf (for $n = 1$ this is the edge chosen at the root). For a realization $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { n } )$ , the reward $f _ { \widehat { \pmb { y } } } ( \boldsymbol { x } )$ at the leaf corresponding to $\widehat { \pmb { y } }$ is 1 if exactly one coordinate i satisfies $x _ { i } \neq \bot$ and this coordinate satisfies $x _ { i } = ( j , { \widehat { y } } _ { i j } )$ for some $j \in [ s _ { i } ]$ , and 0 otherwise. Thus, every round reveals all coordinates regardless of the prediction, and only the distributions depend on $\pmb \theta .$

Level-set complexity. We verify Assumption 3.1 with $s _ { 1 } , \ldots , s _ { n }$ . Fix nonempty sets of distributions, a prediction vector ${ \widehat { \mathbf { y } } } ,$ a coordinate $i ,$ and values $x _ { 1 } , \ldots , x _ { i - 1 }$ , and let $\phi ( z )$ be the optimistic continuation value after $X _ { i } = z$ is revealed. $\operatorname { I f } z = ( j , - { \widehat { y } } _ { i j } )$ for some $j \in [ s _ { i } ]$ , then coordinate i differs from ⊥ and its sign is predicted incorrectly, and hence $\phi ( z ) = 0$ . Since $\phi \geq 0$ , every superlevel set of $\phi$ at a negative level is the whole domain, and every superlevel set at a nonnegative level is a subset of $\{ \bot \} \cup \{ ( j , \widehat { y } _ { i j } ) : j \in [ s _ { i } ] \}$ . Such a subset consists of at most $s _ { i }$ runs of consecutive labels, since it contains at most one label from each column and ⊥ is adjacent to both labels of column 1. Since a run of consecutive labels is the intersection of the domain with an interval, $\phi$ has level-set complexity at most $s _ { i }$

The next lemma shows that each incorrect sign in $\widehat { \pmb { y } }$ reduces the expected reward by at least a fixed amount.

Lemma 6.1. For every θ and every prediction vector $\widehat { \pmb { y } }$

$$
\operatorname { O P T } ( D ^ { \pm } ) - \mathbb { E } _ { X \sim D ^ { \theta } } [ f _ { \widehat { \mathfrak { Y } } } ( X ) ] \geq \frac { \varepsilon } { 4 S } \left| \left\{ ( i , j ) \in J : \widehat { y } _ { i j } \neq \theta _ { i j } \right\} \right| .
$$

Proof. Let $X \sim D ^ { \theta }$ , and let $c _ { i }$ be the probability that $X _ { k } = \perp$ for every $k \neq i$ . By the union bound, $c _ { i } \geq 1 - \sum _ { k \neq i } s _ { k } / ( 2 S ) \geq 1 / 2$ . For $( i , j ) \in J$ , the event that $X _ { i } = ( j , \widehat { y } _ { i j } )$ and $X _ { k } = \perp$ for every $k \neq i$ has probability $c _ { i } ( 1 + \varepsilon { \widehat { y } } _ { i j } \theta _ { i j } ) / ( 4 S )$ . These events are disjoint, and $f _ { \widehat { \mathbf { y } } } ( X ) = 1$ exactly when one of them occurs. Hence, $\begin{array} { r } { \mathbb { E } [ f _ { \widehat { \pmb { y } } } ( \bar { X } ) ] = \sum _ { ( i , j ) \in J } c _ { i } \big ( 1 + \varepsilon \widehat { y } _ { i j } \theta _ { i j } \big ) / ( 4 S ) } \end{array}$ , and each incorrect sign decreases this value by $c _ { i } \varepsilon / ( 2 S ) \geq \varepsilon / ( 4 S )$ relative to ${ \widehat { \pmb y } } = { \pmb \theta }$ . Since the root is the only decision and randomizing over root actions averages their values, $\mathrm { O P T } ( D ^ { \pmb \theta } )$ is the expected reward for $\widehat { \mathbf { y } } = \pmb { \theta }$ 2 which proves the claim. □

## 6.2 Proof of the Lower Bound

Before proving Theorem 3.6, we show that every learning algorithm predicts each sign $\theta _ { i j }$ incorrectly with constant probability when θ is chosen at random. For distributions $P , Q$ on a finite set $\mathcal { Z } ,$ we write $\begin{array} { r } { \mathrm { K L } ( P \| Q ) = \sum _ { z \in \mathcal { Z } } P ( z ) \log ( P ( z ) / Q ( z ) ) } \end{array}$ for the KL divergence and $d _ { \mathrm { T V } } ( P , Q ) =$ max $\ b { \cdot } _ { A \subseteq { \mathcal { Z } } } | P ( A ) - Q ( A ) |$ for the total variation distance. Fix any learning algorithm, and let $\widehat { \boldsymbol { y } } ^ { t }$ be its prediction vector in round $t \in [ T ]$ . Choose θ uniformly from $\{ - 1 , + 1 \} ^ { J }$ once at the beginning, and let $\overline { { \mathbb { P } } }$ denote the probability over θ, the observations, and the internal randomness of the learning algorithm.

Lemma 6.2. For every $t \in [ T ]$ and $( i , j ) \in J$ , we have $\overline { { \mathbb { P } } } ( \widehat { y } _ { i j } ^ { t } \neq \theta _ { i j } ) \geq 3 / 8$

Proof. Fix $t , ( i , j )$ , and the signs $\theta _ { i ^ { \prime } j ^ { \prime } }$ for all $( i ^ { \prime } , j ^ { \prime } ) \in J \backslash \{ ( i , j ) \}$ . Let Z be the finite set of possible values of the past observations $Z = ( X _ { k } ^ { \tau } ) _ { \tau < t , k \in [ n ] }$ , and let $P _ { \mathrm { + } } , P _ { \mathrm { - } }$ be the distributions of $Z$ when $\theta _ { i j } = + 1$ and $\theta _ { i j } = - 1$ , respectively. Since every round reveals all coordinates regardless of the predictions, $P _ { + }$ and $P _ { - }$ <sub>−</sub> do not depend on the learning algorithm. For one observation of coordinate $i ,$ only the probabilities of the labels $( j , - 1 )$ and $( j , + 1 )$ difer, and their contributions to the KL divergence sum to $\varepsilon \log ( ( 1 + \varepsilon ) / ( 1 - \varepsilon ) ) / ( 2 S )$ . By independence across coordinates and rounds, we have

$$
\mathrm { K L } ( P _ { + } \| P _ { - } ) = \frac { ( t - 1 ) \varepsilon } { 2 S } \log \frac { 1 + \varepsilon } { 1 - \varepsilon } \leq \frac { 3 T \varepsilon ^ { 2 } } { 2 S } \leq \frac { 3 } { 3 2 } ,
$$

where we used $\log ( ( 1 + \varepsilon ) / ( 1 - \varepsilon ) ) \leq 3 \varepsilon$ , which holds for $\varepsilon \le 1 / 4$ because the left-hand side vanishes at $\varepsilon = 0$ and its derivative $2 / ( 1 - \varepsilon ^ { 2 } )$ is at most 3, and then $\varepsilon \le \sqrt { S / T } / 4$

By Pinsker’s inequality [LS20], $d _ { \mathrm { T V } } ( P _ { + } , P _ { - } ) \leq \sqrt { \mathrm { K L } ( P _ { + } \| P _ { - } ) / 2 } \leq \sqrt { 3 / 6 4 } < 1 / 4$ . Since the learner’s internal randomness is independent of the observations, the conditional probability $a ( Z )$ that $\widehat { y } _ { i j } ^ { t } = + 1$ given $Z$ is the same [0, 1]-valued function of $Z$ in both environments. Let $A = \{ z \in$ $\mathcal { Z } : P _ { + } \overset { \cdot } { ( z ) } > P _ { - } ( z ) \}$ . Since a takes values in [0, 1], we have

$$
\mathbb { E } _ { Z \sim P _ { + } } [ a ( Z ) ] - \mathbb { E } _ { Z \sim P _ { - } } [ a ( Z ) ] = \sum _ { z \in \mathcal { Z } } a ( z ) \big ( P _ { + } ( z ) - P _ { - } ( z ) \big ) \le P _ { + } ( A ) - P _ { - } ( A ) \le d _ { \mathrm { T V } } ( P _ { + } , P _ { - } ) .
$$

Therefore, the error probability averaged over the two values of $\theta _ { i j }$ is

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { 2 } \big ( \mathbb { E } _ { Z \sim P _ { + } } [ 1 - a ( Z ) ] + \mathbb { E } _ { Z \sim P _ { - } } [ a ( Z ) ] \big ) = \frac { 1 } { 2 } \big ( 1 - \mathbb { E } _ { Z \sim P _ { + } } [ a ( Z ) ] + \mathbb { E } _ { Z \sim P _ { - } } [ a ( Z ) ] \big ) } \\ & { \displaystyle \geq \frac { 1 } { 2 } \big ( 1 - d _ { \mathrm { T V } } ( P _ { + } , P _ { - } ) \big ) \geq \frac { 3 } { 8 } . } \end{array}
$$

This bound holds for every choice of the fixed signs $\theta _ { i ^ { \prime } j ^ { \prime } }$ , and averaging over them yields $\overline { { \mathbb { P } } } ( \widehat { y } _ { i j } ^ { t } \neq$ $\theta _ { i j } ) \geq 3 / 8$ □

We are now ready to prove Theorem 3.6.

Proof of Theorem 3.6. Consider the problem in Section 6.1, which satisfies Assumption 3.1 with $s _ { 1 } , \ldots , s _ { n }$ . Conditional on $\theta ,$ the past observations, and the learner’s randomness, the prediction $\widehat { \boldsymbol { y } } ^ { t }$ is fixed and independent of $X ^ { t } \sim D ^ { \theta }$ . Hence, $\mathrm { R e g } _ { t } = \mathrm { O P T } ( D ^ { \theta } ) - \mathbb { E } _ { X \sim D ^ { \theta } } [ f _ { \widehat { \pmb { y } } ^ { t } } ( X ) ]$ ], which is at least

$\textstyle { \frac { \varepsilon } { 4 S } } | \{ ( i , j ) \in J : { \widehat { y } } _ { i j } ^ { t } \neq \theta _ { i j } \} |$ by Lemma 6.1. Taking expectations, summing over $t ,$ and averaging over $\theta ,$ we obtain from Lemma 6.2

$$
2 ^ { - S } \sum _ { \theta \in \{ - 1 , + 1 \} ^ { J } } R _ { T } ( D ^ { \theta } ) \geq \frac { \varepsilon } { 4 S } \sum _ { t = 1 } ^ { T } \sum _ { ( i , j ) \in J } \overline { { \mathbb { P } } } ( \partial _ { i j } ^ { t } \ne \theta _ { i j } ) \geq \frac { \varepsilon } { 4 S } \cdot T S \cdot \frac { 3 } { 8 } = \frac { 3 } { 3 2 } T \varepsilon \geq \frac { 1 } { 6 4 } \operatorname* { m i n } \{ T , ~ \sqrt { S T } \} ,
$$

where the last inequality uses (6.1). Hence, $R _ { T } ( D ^ { \pmb \theta } )$ is at least this value for some $\pmb \theta \in \{ - 1 , + 1 \} ^ { J }$ Since all $n$ coordinates are observed in every round and these observations determine the reward of every action, the lower bound holds even under full feedback. □

## 7 Conclusion

We studied SOO under semi-bandit feedback without FSD monotonicity and proposed a learning algorithm based on optimistic DP. Its regret is $\widetilde { O } ( \sqrt { S T } )$ with $\begin{array} { r } { S = \sum _ { i } \operatorname* { m i n } \{ s _ { i } , k _ { i } \} } \end{array}$ , where $s _ { i }$ bounds the level-set complexity of the optimistic continuation values for coordinate i and $k _ { i }$ is the number of support points of $D _ { i }$ , neither of which the algorithm needs to know. We also showed that this bound is tight up to logarithmic factors in the worst case for all positive integers $s _ { 1 } , \ldots , s _ { n }$ with $k _ { i } = 2 s _ { i } + 1$ , even under full feedback. With the extension in Theorem 3.7, the analysis yields regret $\widetilde { O } ( \sqrt { n T } )$ for all five applications in Section 4 after rescaling the rewards to [0, 1], provided that, in scheduling, the reward functions have bounded level-set complexity and the observed processing time and reward identify the outcome.

Three directions remain open. First, the running time of each round is polynomial in the size of the decision tree given the suprema of the optimistic continuation values over intervals (Proposition A.3), but the tree can be exponentially large in n, as in SBFE with general Boolean functions, and computing the suprema requires structure such as finite domains, quasi-convexity, or Lipschitz continuity (Appendix A.3). Proposition A.4 covers problems with few reachable states, and FSDmonotone problems in a known direction with finite domains reduce to exact or α-approximation algorithms for known distributions by Proposition 3.4, but it is open which other problems admit eficient learning algorithms. Second, level-set complexity is defined for real-valued coordinates, and it is open how to extend the analysis to multi-dimensional observations. Third, our algorithm builds confidence sets from the observed values, and it is open whether regret growing as $\sqrt { T }$ and polynomially in n is possible without FSD monotonicity under bandit feedback, where only the reward is observed, as studied for the prophet inequality and Pandora’s box [GKSW24].

## Acknowledgments

This work was partially supported by the joint project of Kyoto University and Toyota Motor Corporation, titled “Advanced Mathematical Science for Mobility Society”, JST ERATO Grant Number JPMJER2301, JST ASPIRE Grant Number JPMJAP2302, and JSPS KAKENHI Grant Numbers JP21K17708, JP21H03397, JP25K00137, JP26K14718, JP26K02867, and JP26KJ1106.

## AI Methodology

We wrote earlier versions of this paper without AI assistance. These versions used the total variation of the functions $\phi ( z ) = \bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } )$ on intervals I for revelation edges $( u , v )$ as a parameter. Here, the total variation of $\phi$ on I is the supremum of $\textstyle \sum _ { j = 0 } ^ { k - 1 } | \phi ( z _ { j + 1 } ) - \phi ( z _ { j } ) |$ over all $k \geq 0$ and $z _ { 0 } < \dots < z _ { k }$ in $I \cap \operatorname { d o m } ( \phi )$ . An analysis based on this parameter yields regret $\widetilde { O } ( n ^ { 3 / 4 } \sqrt { \nu T } )$ when the total variation is at most ν. In the current version, the idea of using level-set complexity and the proof of the lower bound came from OpenAI ChatGPT/Codex and Anthropic Claude. The tools materially afected Sections 2.2, 3, 4, 5, and 6 and Appendix $\mathrm { A }$ . The authors verified the correctness and originality of all content, including references. We also used these tools to assist with strengthening the content and editing the language throughout the paper.

## A Computational Aspects

We reduce each optimistic update to an LP, bound the running time of one round given the suprema of optimistic continuation values over intervals, and compute these suprema when the observations can be summarized by few reachable states.

## A.1 Finite Reduction to Interval Masses

Fix a coordinate i, an edge $( u , v )$ , and observations $x _ { \mathrm { o b s } ( u ) }$ , and let $\phi ( z ) = \bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } )$ . The optimistic update $\bar { h } _ { ( u , v ) } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) = \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \mathbb { E } _ { X \sim Q } [ \phi ( X ) ]$ in (3.4) maximizes over all distributions $Q$ on the domain of coordinate i that satisfy the constraint in (3.2) for every interval $I \in \mathcal { T } _ { 1 }$ , and hence, in general, over an infinite-dimensional set with infinitely many constraints. We reduce it to an $\mathrm { L P }$ over the masses of finitely many cells, which are formed by the sample points and the intervals between them, given the suprema of $\phi$ on the cells. If $N _ { i , t } < 1 6$ , the confidence set consists of all distributions on that coordinate’s domain, by definition if $N _ { i , t } = 0$ and since $b _ { N _ { i , t } } \equiv 1$ as in the proof of Lemma 5.2 otherwise, and the optimistic update equals $\operatorname { s u p } _ { z \in \mathrm { d o m } ( \phi ) } \phi ( z )$ . Otherwise, write $N = N _ { i , t } \geq 1 6$ , and let $\widehat { D } _ { i , N }$ be the empirical distribution of coordinate $i ,$ as in Section 3.1.

Let $y _ { 1 } < \dots < y _ { m ^ { \prime } }$ be the distinct sample points of the empirical distribution. Partition $[ 0 , 1 ]$ into the singletons $\{ y _ { j } \}$ , the open intervals between them, and the two end intervals $[ 0 , y _ { 1 } )$ and $( y _ { m ^ { \prime } } , 1 ]$ , and intersect each set with dom(ϕ). List the nonempty sets from left to right as $C _ { 1 } , \ldots , C _ { M }$ which we call cells. We have $M \leq 2 m ^ { \prime } + 1 \leq 2 N + 1$ . Let $N _ { j } = N \cdot \widehat { D } _ { i , N } ( C _ { j } )$ be the number of samples in $C _ { j }$ , and let $\bar { \phi } _ { j } = \operatorname* { s u p } _ { z \in C _ { i } } \phi ( z )$ . Each $N _ { j }$ is a nonnegative integer, and $\textstyle \sum _ { j = 1 } ^ { M } N _ { j } = N$ . A cell containing a sample point is the singleton consisting of that point.

Writing $q _ { j }$ for the mass assigned to cell $C _ { j }$ , define

$$
\mathcal { Q } = \left\{ q \in \mathbb { R } _ { \geq 0 } ^ { M } : \sum _ { j = 1 } ^ { M } q _ { j } = 1 , \left| \sum _ { j = a } ^ { c } q _ { j } - \frac { 1 } { N } \sum _ { j = a } ^ { c } N _ { j } \right| \leq b _ { N } \left( \frac { 1 } { N } \sum _ { j = a } ^ { c } N _ { j } \right) \mathrm { ~ f o r ~ a l l ~ } 1 \leq a \leq c \leq M \right\}\tag{A.1}
$$

We write $q _ { \leq c } = q _ { 1 } + \cdot \cdot \cdot + q _ { c }$ for the cumulative masses, with $q _ { \leq 0 } = 0$

Proposition A.1. The set Q is nonempty and compact, and the optimistic update satisfies

$$
\operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] = \operatorname* { m a x } _ { q \in \mathcal { Q } } \sum _ { j = 1 } ^ { M } q _ { j } \bar { \phi } _ { j } .\tag{A.2}
$$

Proof. For any $Q \in \mathcal { C } _ { i , t }$ , let $q _ { j } = Q ( C _ { j } )$ . For $1 \leq a \leq c \leq M$ , there is an interval $I \in \mathcal { T } _ { 1 }$ with $I \cap \operatorname { d o m } ( \phi ) = C _ { a } \cup \cdot \cdot \cdot \cup C _ { c } ,$ and hence $\begin{array} { r } { Q ( I ) = \sum _ { j = a } ^ { c } q _ { j } } \end{array}$ and $\begin{array} { r } { \widehat { D } _ { i , N } ( I ) = \frac { 1 } { N } \sum _ { i = a } ^ { c } N _ { j } } \end{array}$ . Thus, the constraint in (3.2) for this interval I, which Q satisfies, is the constraint in (A.1) for $( a , c )$ , and hence $q \in \mathcal { Q }$ . In addition, $\begin{array} { r } { \mathbb { E } _ { X \sim Q } [ \phi ( X ) ] \le \sum _ { j = 1 } ^ { M } q _ { j } \bar { \phi } _ { j } } \end{array}$ . Since the empirical distribution $\widehat { D } _ { i , N }$ belongs to $\mathcal { C } _ { i , t } .$ , this argument shows that the vector with $q _ { j } = \widehat { D } _ { i , N } ( C _ { j } )$ lies in Q. As a closed subset of the simplex, Q is therefore nonempty and compact, and the linear objective $\textstyle \sum _ { j = 1 } ^ { M } q _ { j } { \bar { \phi } } _ { j }$ with coeficients $\bar { \phi } _ { j } \in [ 0 , 1 ]$ attains its maximum on $\mathcal { Q } .$ . Hence, the left-hand side of (A.2) is at most the right-hand side.

Conversely, fix $q \in \mathcal { Q }$ . Choose any point $z _ { j } \in C _ { j }$ from each cell, and let $Q _ { q }$ place mass $q _ { j }$ at $z _ { j }$ For any $I \in \mathcal { T } _ { 1 }$ , the set $\{ j : z _ { j } \in I \}$ consists of consecutive indices $a , \ldots , c ,$ since the $z _ { j }$ are ordered. Since sample points are preserved as singleton cells, we also have $\begin{array} { r } { \widehat { D } _ { i , N } ( I ) = \frac { 1 } { N } \sum _ { j = a } ^ { c } N _ { j } } \end{array}$ , and the constraints in (A.1) imply $| Q _ { q } ( I ) - \widehat { D } _ { i , N } ( I ) | \leq b _ { N } ( \widehat { D } _ { i , N } ( I ) )$ . If no $z _ { j }$ lies in I, both probabilities are 0. Thus, $Q _ { q } \in \mathcal { C } _ { i , t }$

Let $q$ be a maximizing vector on the right-hand side. For any $\zeta > 0$ , choose a point $z _ { j }$ in each cell $C _ { j }$ such that $\phi ( z _ { j } ) \geq \bar { \phi } _ { j } - \zeta$ , which exists by the definition of supremum. Let $Q _ { q }$ place mass $q _ { j }$ at these points $z _ { j }$ . Since the previous paragraph applies to any choice of points in the cells, $Q _ { q } \in \mathcal { C } _ { i , t }$ , and hence the left-hand side is at least $\begin{array} { r } { \mathbb { E } _ { X \sim Q _ { q } } [ \phi ( X ) ] = \sum _ { j = 1 } ^ { M } q _ { j } \phi ( z _ { j } ) \ge \sum _ { j = 1 } ^ { M } q _ { j } \bar { \phi } _ { j } - \zeta . } \end{array}$ Equality follows by letting $\zeta \downarrow 0$ □

Section 3.2 uses the following consequence of Proposition A.1 for FSD-monotone problems.

Lemma A.2. If the domain of coordinate i is finite, then $\mathcal { C } _ { i , t }$ contains a distribution that dominates all distributions in $\mathcal { C } _ { i , t }$ and a distribution that is dominated by all distributions in $\mathcal { C } _ { i , t }$ , and both can be computed by solving an LP.

Proof. If $N _ { i , t } < 1 6$ , then $\mathcal { C } _ { i , t }$ consists of all distributions on the domain, and the point masses at the largest and the smallest points of the domain sufice. Otherwise, let the cells $C _ { 1 } , \ldots , C _ { M }$ and the set Q be as above, formed in the domain of coordinate i. Every $q \in \mathcal { Q }$ satisfies $q _ { \leq M } = 1$ and each constraint in (A.1) and each constraint $q _ { j } ~ \geq ~ 0$ bounds a diference $q \leq c - q \leq a - 1$ with $1 \leq a \leq c \leq M$ from above or below by a constant. Such constraints are preserved under componentwise minima of the cumulative masses, and hence, for $q , q ^ { \prime } \in \mathcal { Q }$ , the vector whose cumulative masses are min $\{ q _ { \mathrm { \leq } c } , q _ { \mathrm { < } c } ^ { \prime } \}$ also belongs to $\mathcal { Q } .$ . Since Q is compact, each $q { \leq } c$ attains its minimum on Q, and applying this closure to the minimizers for $c = 1 , \ldots , M$ yields $q \in \mathcal { Q }$ with $\begin{array} { r } { \underline { { q } } _ { \le c } = \operatorname* { m i n } _ { q \in \mathcal { Q } } q _ { \le c } } \end{array}$ for every $c .$ This vector is the unique optimal solution of the LP that minimizes $\textstyle \sum _ { c = 1 } ^ { M } q _ { \leq c }$ over Q. Let $Q ^ { + }$ place mass ${ \underline { { q } } } _ { j }$ at the largest point of $C _ { j }$ , which belongs to $\mathcal { C } _ { i , t }$ by the proof of Proposition A.1. For every $Q \in \mathcal { C } _ { i , t }$ , the vector $( Q ( C _ { j } ) ) _ { j }$ belongs to $\mathcal { Q }$ by the same proof. For every $x \in [ 0 , 1 ]$ ], the cells contained in $[ 0 , x ]$ are $C _ { 1 } , \ldots , C _ { c }$ for some $c ,$ and hence $\begin{array} { r } { Q ^ { + } ( [ 0 , x ] ) = \underline { { q } } _ { < c } \leq \sum _ { j \leq c } Q ( C _ { j } ) \leq Q ( [ 0 , x ] ) } \end{array}$ , that is, $Q ^ { + }$ dominates Q. Symmetrically, the vector whose cumulative sums are all the largest, with its masses placed at the smallest points of the cells, yields a distribution in $\mathcal { C } _ { i , t }$ that is dominated by all distributions in $\mathcal { C } _ { i , t }$ □

## A.2 Running Time with the Interval-Supremum Oracle

By Proposition A.1, each optimistic update reduces to the LP on the right-hand side of (A.2), and it remains to solve this LP and to obtain its coeficients $\bar { \phi } _ { j }$ . We bound the running time of one round of Algorithm 1 on the decision tree $( V , A )$ under the following assumptions. First, we use the realarithmetic model, in which arithmetic operations, comparisons, square roots, and logarithms have unit cost. Second, the algorithm can evaluate leaf rewards $f _ { \ell } \big ( x _ { \mathrm { o b s } ( \ell ) } \big )$ , test feasibility $v \in \mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ and call an interval-supremum oracle, and each of these operations has unit cost. Given an edge $( u , v )$ revealing a coordinate i, observations $x _ { \mathrm { o b s } ( u ) }$ , and an interval $I \in \mathcal { T } _ { 1 }$ , the interval-supremum oracle returns $\mathrm { s u p } _ { z \in I \cap \mathrm { d o m } ( \phi ) } \phi ( z )$ for $\phi ( z ) = \bar { g } _ { v } ^ { t } ( \dot { z } , \dot { x } _ { \mathrm { o b s } ( u ) } )$ , or reports that I ∩ dom(ϕ) is empty.

Proposition A.3. Assume the real-arithmetic model and unit-cost access to leaf rewards, feasibility tests, and the interval-supremum oracle. Then, the policy of Algorithm 1 in round t can be executed in time polynomial in $n , t , | V |$ , and $| A |$

Proof. Since $N _ { i , t } \leq t - 1$ for every $i ,$ preprocessing each sample sequence by sorting and finding distinct values and multiplicities takes time polynomial in n and t. Upon reaching node u, the algorithm computes $\bar { h } _ { ( u , v ) } ^ { t }$ for each feasible action v at the current observations. For an edge with no observation, this is a leaf-reward evaluation. For a revelation edge with fewer than 16 samples, this is one oracle query with $I = [ 0 , 1 ]$ . Otherwise, the algorithm obtains $\bar { \phi } _ { j }$ for the at most $2 N _ { i , t } + 1$ cells from the oracle and computes the optimal value of the LP of Proposition $\mathrm { A . 1 }$ . In terms of the cumulative masses $q { \le } j$ , the objective and every constraint of this LP involve only diferences $q \leq c - q \leq a - 1$ with $1 \leq a \leq c \leq M$ . Hence, the LP maximizes a linear function of the node potentials $q _ { \leq 0 } , \ldots , q _ { \leq M }$ subject to upper bounds on their diferences, which is the dual of an uncapacitated minimum-cost flow problem with $M { + 1 }$ nodes and $O ( M ^ { 2 } )$ arcs. By strong duality, the optimal value of the LP equals the minimum cost of this flow problem. The potentials given by the empirical distribution are feasible, and replacing the arc costs by the reduced costs with respect to them makes the costs nonnegative without changing the optimal flows. Then, the strongly polynomial algorithm of Orlin [Orl93] computes the minimum cost with $O ( M ^ { 3 } \log M )$ arithmetic operations and comparisons. Since $M \leq 2 N _ { i , t } + 1 \leq 2 t - 1$ , the computation for each edge, including the $O ( t ^ { 2 } )$ arc costs, takes $O ( t ^ { 3 } \log { t } )$ time. The algorithm then chooses a maximizing feasible action by comparing these values.

Since the nodes visited in a round lie on a single root-to-leaf path, each of the at most $| V |$ nodes is visited at most once, and at most $| A |$ outgoing edges are examined in total, which proves the claim. □

## A.3 Implementing the Interval-Supremum Oracle

The optimistic DP cannot simply be computed by backward induction, since each optimistic continuation value $\bar { g } _ { v } ^ { t }$ is a function of the observations $x _ { \mathrm { o b s } ( v ) }$ , which can take exponentially or infinitely many values. The interval-supremum oracle hides this dificulty, and each query takes the supremum of the function $\phi ( z ) = \bar { g } _ { v } ^ { t } ( z , x _ { \mathrm { o b s } ( u ) } )$ of one variable over I ∩ dom(ϕ). We first reduce a query to finitely many evaluations of $\phi$ and then show how to compute these evaluations.

A query asks for the supremum of $\phi$ over $I \cap \operatorname { d o m } ( \phi )$ , which can be an infinite set, and in the following cases, finitely many evaluations of ϕ sufice. If the domain of coordinate i is finite, a query reduces to evaluating $\phi$ at the points of the domain in I. If the domain is [0, 1], ϕ is continuous, and [0, 1] can be divided into finitely many known intervals on each of which $\phi$ is monotone, a query reduces to evaluating $\phi$ at the endpoints of the closure of I and at the division points in I. If the domain is [0, 1] and $\phi$ is continuous and quasi-convex, for example, monotone, the two endpoints of the closure of I sufice, even if the minimizer of $\phi$ is unknown.

If the domain is [0, 1] and the optimistic continuation values are L-Lipschitz in the revealed value, a query can be approximated within $\eta$ by the maximum of $\phi$ over a grid of the closure of I that contains its endpoints and has spacing at most $\eta / L$ . Each of the at most n revelation edges below a node then adds an error of at most $\eta _ { \mathrm { : } }$ , and changing every coeficient of the LP of Proposition A.1 by at most the same amount changes its optimal value by at most this amount since $\textstyle \sum _ { j } q _ { j } = 1$ . Hence, the computed values are within $n \eta$ of the optimistic continuation values, and choosing children by them loses at most 2nη in the optimistic continuation value at each of the at most $n + 1$ stages. For $\eta = 1 / ( 4 n ( n + 1 ) T )$ , these losses add at most $1 / 2$ to the cumulative gap $\Delta$ in the proof of Theorem 3.2, and the regret bounds increase by at most $^ { 1 , }$ also in the refinements in which $\Delta$ enters the variance budget.

It remains to compute the evaluations. Each value of $\phi$ is an optimistic continuation value at v after an additional observation, and computing it requires the optimistic continuation values below $v$ for all further observations, whose number can grow exponentially in $n .$ As in a DP for known distributions, this growth disappears when the past observations matter only through a small state. For example, in stochastic knapsack (in Section 4.4) with integer sizes and an integer capacity $B ,$ the past afects the future only through the set of inserted items and the remaining capacity, and the rewards of the inserted items are added to every subsequent leaf. With nodes representing sets of inserted items, as in the directed acyclic graph of the footnote in Section 2, the optimistic continuation value at a node is the reward collected so far plus a function of the remaining capacity, and each evaluation of $\phi$ needs this function only at the at most $B + 1$ capacities.

Formally, a state representation consists of sets $\Pi _ { u }$ for $u \in V$ and maps $\pi _ { u }$ from the observations $x _ { \mathrm { o b s } ( u ) }$ to $\Pi _ { u }$ with the following properties.

(a) The state is updated by the revealed value, that is, each revelation edge $\boldsymbol { e } = \left( u , v \right)$ revealing a coordinate i has a map $\pi _ { e }$ with $\pi _ { v } ( z , x _ { \mathrm { o b s } ( u ) } ) = \pi _ { e } ( \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } ) , z )$ , and each stopping edge $( u , v )$ satisfies $\pi _ { v } = \pi _ { u }$

(b) The feasible actions depend only on the state, that is, $\mathcal { A } _ { u } ( x _ { \mathrm { o b s } ( u ) } ) = \widetilde { \mathcal { A } } _ { u } ( \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } ) )$ for some sets $\widetilde { \mathcal { A } } _ { u } ( \pi ) \subseteq \mathrm { c h } ( u )$

(c) Each leaf reward is a reward of the final state plus rewards collected on the $\operatorname { w a y } .$ , that is, $f _ { \ell } \big ( x _ { \mathrm { o b s } ( \ell ) } \big )$ is $\beta _ { \ell } \big ( \pi _ { \ell } ( x _ { \mathrm { o b s } ( \ell ) } ) \big )$ plus the sum of $\gamma _ { e } ( \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } ) , x _ { i } )$ over the revelation edges $e =$ $( u , v )$ on the path to ℓ, where i is the coordinate revealed by e.

For knapsack, the state is the remaining capacity, or a symbol for an overflow, $\gamma _ { e }$ is the normalized reward of the inserted item if it fits, and $\beta _ { \ell } = 0$

The optimistic DP then runs on pairs of a node and a state. Let $\Phi _ { \ell } = \beta _ { \ell }$ at leaves, and at a non-leaf node $u ,$ let $\Phi _ { u } ( \pi )$ be the maximum of $\Psi _ { ( u , v ) } ( \pi )$ over $v \in \tilde { A } _ { u } ( \pi )$ , where $\Psi _ { ( u , v ) } ( \pi ) = \Phi _ { v } ( \pi )$ for a stopping edge and

$$
\Psi _ { e } ( \pi ) = \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i , t } } \mathbb { E } _ { X \sim Q } [ \psi _ { e , \pi } ( X ) ] , \quad \quad \psi _ { e , \pi } ( z ) = \gamma _ { e } ( \pi , z ) + \Phi _ { v } ( \pi _ { e } ( \pi , z ) ) ,
$$

for a revelation edge $\boldsymbol { e } = \left( u , v \right)$ revealing i. Let $\kappa _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ be the sum of the rewards $\gamma _ { e }$ collected on the path to $u .$ Since expectations, suprema, and maxima commute with adding $\kappa _ { u }$ , backward induction shows that $\bar { g } _ { u } ^ { t } ( x _ { \mathrm { o b s } ( u ) } ) = \kappa _ { u } ( x _ { \mathrm { o b s } ( u ) } ) + \Phi _ { u } ( \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } ) )$

By Proposition A.1, each $\Psi _ { e } ( \pi )$ is determined by the suprema of $\psi _ { e , \pi }$ over the cells and over the domain. We call a finite set $\mathcal { Z } _ { e } ( \pi )$ of points of the domain of coordinate i suficient if each of these suprema is the maximum of $\psi _ { e , \pi }$ over the points of $\mathcal { Z } _ { e } ( \pi )$ in the closure of the corresponding set. By the reductions above, the domain is suficient if it is finite, and {0, 1} together with the sample points of coordinate i is suficient if the domain is [0, 1] and $\psi _ { e , \pi }$ is continuous and quasi-convex. Computing $\Psi _ { e } ( \pi )$ then requires $\Phi _ { v }$ only at the states $\pi _ { e } ( \pi , z )$ with $z \in \mathcal { Z } _ { e } ( \pi )$ . Repeating this argument, the recursion evaluates Φ only at the pairs reachable from $( u , \pi )$ , which are obtained by moving along feasible actions, keeping the state on stopping edges and applying $\pi _ { e } ( \cdot , z )$ with $z$ in the suficient sets on revelation edges. Their states are of the form $\pi _ { v } ( x _ { \mathrm { o b s } ( v ) } )$ , since the suficient sets consist of points of the domains. Let $K _ { \Pi }$ be the largest number of pairs reachable from a pair $( u , \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } ) )$ over all nodes u and observations $x _ { \mathrm { o b s } ( u ) }$

For the following proposition, we work in the real-arithmetic model and assume that evaluating $\pi _ { r } , \pi _ { e } , \gamma _ { e } ,$ , and $\beta _ { \ell }$ , testing whether $v \in \widetilde { A } _ { u } ( \pi )$ , listing the points of $\mathcal { Z } _ { e } ( \pi )$ , and testing whether two states are equal all take unit time.

Proposition A.4. Suppose that the problem has a state representation with known suficient sets $\mathcal { Z } _ { e } ( \pi )$ for every state $\pi = \pi _ { u } \big ( x _ { \mathrm { o b s } ( u ) } \big )$ . Then, the policy of Algorithm 1 in round t can be executed in time polynomial in $n , t , | A | , K _ { \Pi }$ , and the sizes of the sets $\mathcal { Z } _ { e } ( \pi )$

Proof. At each of the at most $n + 1$ visited non-leaf nodes u with state $\pi ,$ the policy compares the values $\Psi _ { ( u , v ) } ( \pi )$ , which by the recursion above need Φ only at the pairs reachable from $( u , \pi )$ We compute Φ at each of these at most $K _ { \Pi }$ pairs once, by backward induction along the edges, and we detect repeated pairs by testing states for equality. For a revelation edge e and a state $\pi = \pi _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ , the value $\Psi _ { e } ( \pi )$ is the optimal value of the LP of Proposition A.1 for $\phi$ minus $\kappa _ { u } ( x _ { \mathrm { o b s } ( u ) } )$ , which is the optimal value of the same LP with coeficients for $\psi _ { e , \pi }$ . These coeficients are the maxima of $\psi _ { e , \pi }$ over the points of $\mathcal { Z } _ { e } ( \pi )$ in the closures of the cells, or one maximum over the domain if $N _ { i , t } < 1 6$ , and placing these points into the cells takes time polynomial in t and $| \mathcal { Z } _ { e } ( \pi ) |$ . Since each LP takes $O ( t ^ { 3 } \log { t } )$ time as in the proof of Proposition A.3, each reachable pair takes time polynomial in $t , | A |$ , and the sizes of the sets $\mathcal { Z } _ { e } ( \pi )$ □

We now apply Proposition A.4 to examples. If every domain is finite, the domains are suficient, and Proposition A.4 shows that each round runs in time polynomial in $n , t , | A | , K _ { \Pi }$ , and the sizes of the domains. If every domain is [0, 1] and the optimistic continuation values are L-Lipschitz in the revealed value, the grids of spacing $\eta / L$ in the closures of the cells with $\eta = 1 / ( 4 n ( n + 1 ) T )$ replace the suficient sets with an error of at most η in each supremum. As shown above, the regret bounds then increase by at most 1, and the proof of Proposition A.4 shows that each round runs in time polynomial in $n , T , | A | , K _ { \Pi }$ , and L, where $K _ { \Pi }$ counts the pairs reachable through the grid points.

For overbooking with group reservations in Section 4.5, let the state at a node be the total number of arrivals in the groups revealed so far, which is 0 at the root and at the leaf for $U = \emptyset$ With $\gamma _ { e } ( \pi , z ) = ( a _ { i } - \rho _ { i } b _ { i } + \rho _ { i } b _ { i } z ) / \Gamma$ for the edge revealing group i, $\pi _ { e } ( \pi , z ) ~ = ~ \pi + b _ { i } z$ , and $\beta _ { \ell } ( \pi ) = ( C - c ( \pi - B ) _ { + } ) / \Gamma$ , the domains are suficient, and every reachable state is an integer from 0 to <sup>¯</sup>b. Since $| V | \leq n | \mathcal { U } | + 2$ , we have $K _ { \Pi } \le ( n | \mathcal { U } | + 2 ) ( \bar { b } + 1 )$ , and Proposition A.4 shows that each round runs in time polynomial in $n , t , | \mathcal { U } |$ , and <sup>¯</sup>b.

For optimal stopping with loss aversion in Section 4.3, let $\omega ( \xi , z ) = ( z - \lambda ( \xi - z ) _ { + } + \lambda ) / ( 1 + \lambda )$ be the normalized reward for stopping at value z when $\xi$ is the maximum of the past values. The state at $u _ { i }$ and $\ell _ { i }$ is the pair $( \xi , x _ { i } )$ with $\xi = \operatorname* { m a x } _ { j < i } x _ { j } ,$ , the transition is $\pi _ { e } ( ( \xi , x ) , z ) = ( \operatorname* { m a x } \{ \xi , x \} , z )$ $\gamma _ { e } = 0$ , and $\beta _ { \ell _ { i } } = \omega$ . Then, $\Phi _ { u _ { n } } = \omega$ and, for $i < n , \Phi _ { u _ { i } } ( \xi , z ) = \operatorname* { m a x } \{ \omega ( \xi , z ) , H _ { i + 1 } ( \operatorname* { m a x } \{ \xi , z \} ) \}$ with $H _ { i + 1 } ( \xi ) = \operatorname* { s u p } _ { Q \in \mathcal { C } _ { i + 1 , t } } \mathbb { E } _ { X \sim Q } [ \Phi _ { u _ { i + 1 } } ( \xi , X ) ]$ . Since $\omega ( \cdot , z )$ is nonincreasing and λ/(1 + λ)-Lipschitz for every z, and composition with $\xi \mapsto \operatorname* { m a x } \{ \xi , z \}$ , maxima, expectations, and suprema preserve these properties with the same constant, every $H _ { i }$ is nonincreasing and continuous, and every $\Phi _ { u _ { i } } ( \xi , \cdot )$ is continuous and quasi-convex. Hence, if each domain is $[ 0 , 1 ]$ or finite, the set E consisting of 0, 1, the sample values of all coordinates, and the points of the finite domains has the property that its points in the domain of the revealed coordinate form a suficient set for every edge and state. For a state $( \xi , z _ { 0 } )$ , let $w _ { 0 } = \operatorname* { m a x } \{ \xi , z _ { 0 } \}$ . Since max $\{ w , z \} \in \mathcal { E } \cup \{ w _ { 0 } \}$ for $w \in \mathcal { E } \cup \{ w _ { 0 } \}$ and $z \in { \mathcal { E } }$ , every state reachable from $( \xi , z _ { 0 } )$ other than $( \xi , z _ { 0 } )$ itself lies in $( \mathcal { E } \cup \{ w _ { 0 } \} ) \times \mathcal { E }$ . Hence, $\bar { K } _ { \Pi } \leq 2 n ( \vert \mathcal { E } \vert + 1 ) ^ { 2 }$ since the tree has $2 n - 1$ nodes, and each round runs in polynomial time, since |E| is polynomial in $n , t ,$ and the sizes of the finite domains.

## B Eficient Learning for SBFE with Threshold Functions

In this section, we give a polynomial-time learning algorithm for SBFE with threshold functions. Consider

$$
F ( x ) = \mathbf { 1 } \{ x _ { 1 } + \cdots + x _ { n } \geq \theta \} , \qquad \theta \in [ n ] ,
$$

with positive testing costs $c _ { 1 } , \ldots , c _ { n }$ , and let $\textstyle \Gamma = \sum _ { i = 1 } ^ { n } c _ { i }$

At the beginning of each round t, form the confidence sets $\mathcal { C } _ { i , t }$ as in Section 3.1, and let

$$
\ell _ { i } = \operatorname* { m i n } _ { Q \in \mathcal { C } _ { i , t } } Q ( \{ 1 \} ) , \qquad u _ { i } = \operatorname* { m a x } _ { Q \in \mathcal { C } _ { i , t } } Q ( \{ 1 \} ) .
$$

Order the coordinates by increasing $c _ { i } / u _ { i }$ and, separately, by increasing $c _ { i } / ( 1 - \ell _ { i } )$ . We use the convention $c _ { i } / 0 = \infty$ and break ties by index in both orders. These orders remain fixed throughout the round.

While $F$ is not yet determined, let m be the number of untested coordinates and let $\theta ^ { \prime }$ be the number of additional 1’s needed to establish $F ( x ) = 1$ . Among the untested coordinates, consider the first $\theta ^ { \prime }$ in the $c _ { i } / u _ { i }$ order and the first $m - \theta ^ { \prime } + 1$ in the $c _ { i } / ( 1 - \ell _ { i } )$ order. These two sets intersect because their sizes sum to $m + 1 > m$ . The algorithm tests any coordinate in their intersection and repeats until F is determined. At the end of the round, it adds the observed values to the samples.

Proposition B.1. For $\delta = 1 / ( T + 1 )$ , the algorithm described above can be implemented in $O ( n ^ { 2 } )$ time per round and has expected regret $\widetilde { O } ( \Gamma \sqrt { n T } )$ in the original testing cost.

Proof. The endpoints $\ell _ { i }$ and $u _ { i }$ for all coordinates can be computed in $O ( n )$ time from the sample counts using (3.1) and (3.2). Sorting the two orders once per round and scanning them at each of the at most n steps takes $O ( n ^ { 2 } )$ time.

We next bound the regret. Write the normalized reward from Section 4.1 as $f = f _ { 0 } + f _ { 1 }$ , where $f _ { b } ( \sigma , x ) = f ( \sigma , x ) \mathbf { 1 } \{ F ( x ) = b \}$ for $b \in \{ 0 , 1 \}$ . For $q \in [ 0 , 1 ] ^ { n }$ , write Bernoul $\begin{array} { r } { \operatorname { l i } ( q ) = \prod _ { i } } \end{array}$ Bernoulli(q<sub>i</sub>), and define

$$
V _ { b } ( q ) = \operatorname* { m a x } _ { \sigma } \mathbb { E } _ { X \sim \mathrm { B e r n o u l l i } ( q ) } [ f _ { b } ( \sigma , X ) ] .
$$

Then $\mathrm { O P T } ( \mathrm { B e r n o u l l i } ( q ) ) \leq V _ { 0 } ( q ) + V _ { 1 } ( q )$ . The increasing orders of $c _ { i } / q _ { i }$ and $c _ { i } / ( 1 - q _ { i } )$ attain $V _ { 1 } ( q )$ and $V _ { 0 } ( q )$ , respectively [BD81, SB84]. This also holds when some probabilities are 0 or 1 by continuity, since there are finitely many policies.

Fix a round t, and write $\ell = ( \ell _ { 1 } , \dots , \ell _ { n } )$ and $u = ( u _ { 1 } , \ldots , u _ { n } )$ . Consider a stage with m untested coordinates and remaining threshold $\theta ^ { \prime }$ . On every completion with $F ( x ) = 1$ , a fixed order of the remaining coordinates tests all of its first $\theta ^ { \prime }$ coordinates. Moving any one of these coordinates to the front therefore does not change the testing cost on such completions. The same holds for completions with $F ( x ) = 0$ and the first $m - \theta ^ { \prime } + 1$ coordinates. By induction on $m ,$ the policy $\sigma ^ { t }$ of the algorithm has the same testing cost as the $c _ { i } / u _ { i }$ order on inputs with $F ( x ) = 1$ and as the $c _ { i } / ( 1 - \ell _ { i } )$ order on inputs with $F ( x ) = 0$ . Hence, it attains $V _ { 1 } ( u )$ under Bernoulli(u) and $V _ { 0 } ( \ell )$ under Bernoulli(ℓ).

For a fixed testing order, $f _ { 1 }$ is nondecreasing and $f _ { 0 }$ is nonincreasing in each $x _ { i }$ . Thus, $V _ { 1 }$ is nondecreasing and $V _ { 0 }$ is nonincreasing in each $q _ { i }$ . Let $p _ { i } = \mathbb { P } ( X _ { i } = 1 )$ and $p = ( p _ { 1 } , \dotsc , p _ { n } )$ If every $D _ { i }$ belongs to $\mathcal { C } _ { i , t }$ , then $\ell _ { i } \leq p _ { i } \leq u _ { i }$ for every $i ,$ so $V _ { 1 } ( u ) \geq V _ { 1 } ( p )$ and $V _ { 0 } ( \ell ) \geq V _ { 0 } ( p )$ Proposition $3 . 4 ( \mathrm { b } )$ with $\alpha = 1$ and $K = 2 n$ therefore applies to $f _ { 1 }$ with $Q ^ { t } = \mathrm { B e r n o u l l i } ( u )$ and to $f _ { 0 }$ with $Q ^ { t } = \mathrm { B e r n o u l l i } ( \ell )$ . Adding the two bounds yields

$$
\sum _ { t = 1 } ^ { T } \bigl ( V _ { 0 } ( p ) + V _ { 1 } ( p ) - \mathbb { E } [ f ( \sigma ^ { t } , X ^ { t } ) ] \bigr ) = \widetilde { O } ( \sqrt { n T } ) .
$$

This bounds the normalized regret since $\mathrm { O P T } ( \mathrm { B e r n o u l l i } ( p ) ) \leq V _ { 0 } ( p ) + V _ { 1 } ( p )$ . Multiplying by Γ gives the regret bound in the original testing cost. □

## C Omitted Proofs

Proof of Proposition $4 . 4$ . Let $n = 2$ , let $X _ { 1 } = 1 / 2$ with probability 1, and let $X _ { 2 } ~ \in ~ \{ 0 , 1 \}$ with $\mathbb { P } ( X _ { 2 } = 0 ) = q$ . Stopping at the first candidate yields utility $1 / 2$ , and continuing yields expected utility $( 1 - q ) - q \lambda / 2 = 1 - ( \lambda + 2 ) q / 2$ . Let $q _ { 0 } = 1 / ( \lambda + 2 ) , \varepsilon = \textstyle { \frac { 1 } { 8 } } \operatorname* { m i n } \{ q _ { 0 } , \sqrt { q _ { 0 } / T } \}$ , and $q _ { \pm } = q _ { 0 } \pm \varepsilon$ Continuing is optimal for $q = q _ { - }$ <sub>−</sub>, and stopping is optimal for $q = q _ { + }$ . In both instances, the other action loses $( \lambda + 2 ) \varepsilon / 2 = { \textstyle { \frac { 1 } { 1 6 } } } \operatorname* { m i n } \{ 1 , \sqrt { ( \lambda + 2 ) / T } \}$ in expected utility.

Under full feedback, $X _ { 2 }$ is revealed at the end of every round regardless of the action, which provides more information than semi-bandit feedback. Fix a round t, and let $P _ { + }$ and $P _ { - }$ be the distributions of the $t - 1$ previous samples of $X _ { 2 }$ for $q = q _ { + }$ and $q = q _ { - } ,$ , respectively. The action in round t depends only on these samples and the internal randomness of the learning algorithm. The KL divergence between the distributions of one sample is at most their chi-squared divergence $( q _ { + } - q _ { - } ) ^ { 2 } / ( q _ { - } ( 1 - q _ { - } ) )$ . Since $q _ { - } \ge 7 q _ { 0 } / 8$ and $1 - q _ { - } \ge 1 / 2$ , we have

$$
\mathrm { K L } ( P _ { + } \| P _ { - } ) \leq \frac { ( t - 1 ) ( 2 \varepsilon ) ^ { 2 } } { q _ { - } ( 1 - q _ { - } ) } \leq \frac { 6 4 T \varepsilon ^ { 2 } } { 7 q _ { 0 } } \leq \frac { 1 } { 7 } .
$$

By Pinsker’s inequality, $d _ { \mathrm { T V } } ( P _ { + } , P _ { - } ) \leq 1 / \sqrt { 1 4 } < 1 / 3$ . As in the proof of Lemma 6.2, the probability of the wrong action in round $t ,$ averaged over the two instances, is at least $( 1 - d _ { \mathrm { T V } } ( P _ { + } , P _ { - } ) ) / 2 \geq$ $1 / 3$ . Summing over the rounds, the average of the regrets on the two instances is at least $\textstyle { \frac { T } { 3 } }$ $\frac { 1 } { 1 6 }$ min $\{ 1 , \sqrt { ( \lambda + 2 ) / T } \} = \textstyle { \frac { 1 } { 4 8 } } \operatorname* { m i n } \{ T , \sqrt { ( \lambda + 2 ) T } \}$ , and hence the regret on one of them is at least this value. □

Proof of Proposition $ 4 . 6 .$ We verify the conditions of Theorem 3.7 with $s _ { i } = 1$ , for which $S = n$ Consider the edge revealing group i in the chain for an accepted set U. Let $\varrho$ be the sum of $a _ { j } - \rho _ { j } ( b _ { j } - Y _ { j } )$ over the groups $j$ of U revealed before group $i ,$ and let J be the set of groups of U revealed after group i. The only leaf below this edge is the leaf for U, and with $Y _ { i } = b _ { i } x _ { i } . $ , its reward is $\kappa + \chi ( x _ { i } ) + F$ , where $\kappa = ( \varrho + a _ { i } - \rho _ { i } b _ { i } ) / \Gamma , \chi ( z ) = \rho _ { i } b _ { i } z / \Gamma$ , and

$$
F = \frac { 1 } { \Gamma } \Bigl ( \sum _ { j \in J } \bigl ( a _ { j } - \rho _ { j } ( b _ { j } - Y _ { j } ) \bigr ) - c \Bigl ( \sum _ { j \in U } Y _ { j } - B \Bigr ) _ { + } + C \Bigr ) .
$$

The function $\chi$ has level-set complexity at most 1 because it is nondecreasing. Moreover, $\chi , F \in$ $[ 0 , 1 ]$ by $\rho _ { j } b _ { j } \leq a _ { j }$ and $\textstyle \sum _ { j \in U } Y _ { j } \leq { \bar { b } }$ . The residual reward $F$ is nonincreasing in $x _ { i } ,$ , and the sets of feasible actions do not depend on $x _ { i }$ . Finally, the immediate rewards sum to $\begin{array} { r } { \sum _ { j \in U } \rho _ { j } Y _ { j } / \Gamma \le 1 } \end{array}$ Theorem 3.7 and rescaling by Γ imply the claim. □

Proof of Lemma 5.1. We apply Maurer and Pontil’s theorem [MP09, Theorem 6] to the function class $\{ \mathbf { 1 } _ { I } : I \in \mathcal { T } _ { 1 } \} \cup \left\{ 1 - \mathbf { 1 } _ { I } : I \in \mathcal { T } _ { 1 } \right\}$ . Its uniform covering number at scale $1 / m$ on 2m points is bounded by the number of distinct value patterns. On ordered points $x _ { 1 } \leq \dots \leq x _ { 2 m }$ , an interval selects a consecutive block, yielding at most $( 2 m + 1 ) ^ { 2 }$ patterns. Including the complementary indicators yields at most $2 ( 2 m + 1 ) ^ { 2 }$ patterns, which combines with the factor 10 in their theorem to produce the numerator in $\beta .$ . The sample variance of either indicator is m $\widehat { P } _ { m } ( I ) ( 1 - \widehat { P } _ { m } ( I ) ) / ( m - 1 )$ , yielding the variance term in (5.1). □

## References

[Ala14] Saeed Alaei. Bayesian combinatorial auctions: Expanding single buyer mechanisms to many buyers. SIAM Journal on Computing, 43(2):930–972, 2014.

[AN16] Arash Asadpour and Hamid Nazerzadeh. Maximizing stochastic monotone submodular functions. Management Science, 62(8):2374–2391, 2016.

[AOM17] Mohammad Gheshlaghi Azar, Ian Osband, and R´emi Munos. Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning (ICML), pages 263–272, 2017.

[BD81] Yosi Ben-Dov. Optimal testing procedures for special structures of coherent systems. Management Science, 27(12):1410–1420, 1981.

[CBL06] Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

[CDFS19] Jos´e Correa, Paul D¨utting, Felix Fischer, and Kevin Schewior. Prophet inequalities for i.i.d. random variables from an unknown distribution. In Proceedings of the 20th ACM Conference on Economics and Computation (EC), pages 3–17, 2019.

[CHL<sup>+</sup>16] Wei Chen, Wei Hu, Fu Li, Jian Li, Yu Liu, and Pinyan Lu. Combinatorial multi-armed bandit with general reward functions. In Proceedings of the 30th Conference on Neural Information Processing System (NIPS), pages 1659–1667, 2016.

[CWY13] Wei Chen, Yajun Wang, and Yang Yuan. Combinatorial multi-armed bandit: General framework and applications. In Proceedings of the 30th International Conference on Machine Learning (ICML), pages 151–159, 2013.

[CWYW16] Wei Chen, Yajun Wang, Yang Yuan, and Qinshi Wang. Combinatorial multi-armed bandit and its extension to probabilistically triggered arms. Journal of Machine Learning Research, 17(50):1–33, 2016.

[Dea05] Brian C. Dean. Approximation Algorithms for Stochastic Scheduling Problems. PhD thesis, Massachusetts Institute of Technology, 2005.

[DGV08] Brian C. Dean, Michel X. Goemans, and Jan Vondr´ak. Approximating the stochastic knapsack problem: The benefit of adaptivity. Mathematics of Operations Research, 33(4):945–964, 2008.

[DHK14] Amol Deshpande, Lisa Hellerstein, and Devorah Kletenik. Approximation algorithms for stochastic Boolean function evaluation and stochastic submodular set cover. In Proceedings of the 25th ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1453–1466, 2014.

[EHLM20] Hossein Esfandiari, MohammadTaghi Hajiaghayi, Brendan Lucier, and Michael Mitzenmacher. Prophets, secretaries, and maximizing the probability of choosing the best. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), pages 3717–3727, 2020.

[GHTZ21] Chenghao Guo, Zhiyi Huang, Zhihao Gavin Tang, and Xinzhi Zhang. Generalizing complex hypotheses on product distributions: Auctions, prophet inequalities, and Pandora’s problem. In Proceedings of the 34th Conference on Learning Theory (COLT), pages 2248–2288, 2021.

[GHZ19] Chenghao Guo, Zhiyi Huang, and Xinzhi Zhang. Settling the sample complexity of single-parameter revenue maximization. In Proceedings of the 51st ACM Symposium on Theory of Computing (STOC), pages 662–673, 2019.

[GKMR11] Anupam Gupta, Ravishankar Krishnaswamy, Marco Molinaro, and R. Ravi. Approximation algorithms for correlated knapsacks and non-martingale bandits. In Proceedings of the 52nd IEEE Symposium on Foundations of Computer Science (FOCS), pages 827–836, 2011.

[GKSW24] Khashayar Gatmiry, Thomas Kesselheim, Sahil Singla, and Yifan Wang. Bandit algorithms for prophet inequality and Pandora’s box. In Proceedings of the 35th ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 462–500, 2024.

[GM66] John P. Gilbert and Frederick Mosteller. Recognizing the maximum of a sequence. Journal of the American Statistical Association, 61(313):35–73, 1966.

[GN13] Anupam Gupta and Viswanath Nagarajan. A stochastic probing problem with applications. In Proceedings of the 16th International Conference on Integer Programming and Combinatorial Optimization (IPCO), pages 205–216, 2013.

[GZAN26] Rohan Ghuge, Zhengjia Zhuo, Arpit Agarwal, and Viswanath Nagarajan. Semibandit learning for monotone stochastic optimization. Operations Research, 2026.

[HG26] Tao Hu and Rohan Ghuge. Near-optimal regret bounds for monotone stochastic optimization under semi-bandit feedback. arXiv:2610.17320v1, 2026.

[HKS25] Tsubasa Harada, Yasushi Kawase, and Hanna Sumita. Online algorithms for repeated optimal stopping: Balancing baseline guarantees and regret. arXiv:2511.04484v2, 2025.

[JOA10] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(51):1563–1600, 2010.

[KKO22] Jon Kleinberg, Robert Kleinberg, and Sigal Oren. Optimal stopping with behaviorally biased agents: The role of loss aversion and changing reference points. Games and Economic Behavior, 133:282–299, 2022.

[KS77] Ulrich Krengel and Louis Sucheston. Semiamarts and finite values. Bulletin of the American Mathematical Society, 83(4):745–747, 1977.

[KW12] Robert Kleinberg and S. Matthew Weinberg. Matroid prophet inequalities. In Proceedings of the 44th ACM Symposium on Theory of Computing (STOC), pages 123–136, 2012.

[KWAS15] Branislav Kveton, Zheng Wen, Azin Ashkan, and Csaba Szepesv´ari. Tight regret bounds for stochastic combinatorial semi-bandits. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 535–543, 2015.

[LCW<sup>+</sup>25] Junyan Liu, Ziyun Chen, Kun Wang, Haipeng Luo, and Lillian J. Ratlif. Improved regret and contextual linear extension for Pandora’s box and prophet inequality. In Proceedings of the 39th International Conference on Neural Information Processing Systems (NeurIPS), pages 46877–46914, 2025.

[LS20] Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, 2020.

[Ma18] Will Ma. Improvements and generalizations of stochastic knapsack and Markovian bandits approximation algorithms. Mathematics of Operations Research, 43(3):789– 812, 2018.

[MP09] Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample variance penalization. In Proceedings of the 22nd Conference on Learning Theory (COLT), 2009.

[Orl93] James B. Orlin. A faster strongly polynomial minimum cost flow algorithm. Operations Research, 41(2):338–350, 1993.

[SB84] Salam Salloum and Melvin A. Breuer. An optimum testing algorithm for some symmetric coherent systems. Journal of Mathematical Analysis and Applications, 101(1):170– 194, 1984.

[SC84] Ester Samuel-Cahn. Comparison of threshold stop rules and maximum for independent nonnegative random variables. The Annals of Probability, 12(4):1213–1216, 1984.

[SSL99] Janakiram Subramanian, Shaler Stidham, Jr., and Conrad J. Lautenbacher. Airline yield management with overbooking, cancellations, and no-shows. Transportation Science, 33(2):147–167, 1999.

[TQS20] Yi Tian, Jian Qian, and Suvrit Sra. Towards minimax optimal reinforcement learning in factored Markov decision processes. In Proceedings of the 34th Conference on Neural Information Processing System (NeurIPS), pages 19896–19907, 2020.

[Wei79] Martin L. Weitzman. Optimal search for the best alternative. Econometrica, 47(3):641–654, 1979.

[YK25] Toru Yoshinaga and Yasushi Kawase. Online contention resolution schemes for sizestochastic knapsacks. In Proceedings of the 19th International Conference and Workshops on Algorithms and Computation (WALCOM), pages 393–408, 2025.

[ZN26] Zhengjia Zhuo and Viswanath Nagarajan. Near-optimal regret for semi-bandit learning of monotone stochastic problems. arXiv:2610.24446v1, 2026.