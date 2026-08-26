# Bandit Submodular Maximization under Matroid Constraints: Learning Compressed Exchange Policy

Zongqi Wan Great Bay University zqwan@gbu.edu.cn

Zhijie Zhang Fuzhou University zzhang@fzu.edu.cn

## Abstract

We study adversarial bandit maximization of monotone submodular functions under a matroid constraint. For a rank-k matroid on n elements, we give a randomized oraclepolynomial algorithm that makes one feasible value query per round and has expected (1 − 1/e)-regret $\widetilde { \cal O } ( n ^ { 1 / 3 } k ^ { 2 / 3 } T ^ { 2 / 3 } )$ . This is the first sublinear-regret algorithm for adversarial bandit submodular maximization under general matroid constraints.

Technically, we view the problem as learning an exchange policy for the Poisson base walk. This connects the problem to contextual bandits and gives an information-theoretic sublinearregret guarantee, but directly learning the exponentially many policies requires exponential time and space. We therefore introduce balanced fractional exchanges, which compress the policy mixture into a single fractional base while retaining the exchange information needed by the Poisson analysis. This leads to an polynomial time algorithm with the same regret guarantee.

## Contents

1 Introduction 2   
1.1 Technical Overview 3   
1.2 Related Work . 5   
1.3 Paper Organization 6   
2 Preliminaries 6   
3 Poisson Base Walk with General Exchange Distribution 8   
4 Balanced Fractional Exchanges 12   
5 The Balanced-Exchange Bandit Algorithm 14   
5.1 Algorithm Overview 15   
5.2 Properties of the Bandit Estimator 19   
5.3 Regret Analysis . 20   
A Sampling the Nonhomogeneous Poisson Process 26

## 1 Introduction

Selecting a collection of items whose joint value exhibits diminishing returns is a basic problem in combinatorial optimization. The value of adding a new item is often largest when few related items have already been selected: an additional seed user reaches fewer new individuals after many influential users have been chosen, and an additional sensor supplies less information once nearby locations are already monitored. Monotone submodular functions formalize precisely this phenomenon. They underlie, among many other examples, influence maximization in social networks (Kempe et al., 2003) and information-based sensor placement (Krause et al., 2008). Matroid constraints complement the objective by expressing structured feasibility. Uniform matroids encode cardinality budgets, partition matroids encode group or position quotas, and graphic and linear matroids impose global independence requirements. Thus, monotone submodular maximization over a matroid provides a common abstraction for selecting a valuable yet nonredundant feasible collection.

The ofline submodular maximization problem has a long history. For a cardinality constraint, the classical greedy algorithm achieves the factor $1 - 1 / e$ (Nemhauser et al., 1978). Under a general matroid constraint, the elementary greedy algorithm no longer attains this factor; the optimal polynomial-time guarantee was recovered by continuous greedy, together with the multilinear extension and dependent rounding (Vondrák, 2008; Călinescu et al., 2011; Chekuri et al., 2010). Non-oblivious local search provides a discrete alternative with the same approximation factor (Filmus and Ward, 2014; Buchbinder and Feldman, 2024). Most recently, Ganz Rozenman et al. (2026) obtained $1 - 1 / e$ through a diferent nonhomogeneous Poisson walk approach on matroid bases. Unlike continuous-relaxation algorithms, this walk moves only by feasible single-element exchanges, a feature that will be very helpful in this paper’s setting.

In many applications, however, the objective is neither fixed nor known when the decision is made. A platform repeatedly chooses a slate of recommendations or advertisements before observing user response; a campaign selects seed users before the resulting difusion is realized; and a monitoring system deploys a feasible sensor configuration before its utility is revealed. These examples motivate an online model in which, on each round $t ,$ the learner selects an independent set $S _ { t }$ before seeing the current monotone submodular reward function $f _ { t }$ . Under full-information feedback the learner subsequently receives access to the reward function, whereas under full-bandit feedback it observes only the single scalar $f _ { t } ( S _ { t } )$ . The natural benchmark is the best fixed feasible set in hindsight, weakened by the approximation factor that is unavoidable even ofline.

Online and bandit submodular maximization were first studied through online-greedy and meta-action reductions. For cardinality constraints, Streeter and Golovin (2008) obtained $O ( T ^ { 2 / 3 } )$ $( 1 - 1 / e ) – \mathrm { r e g r e t } ;$ ; Streeter et al. (2009) developed the assignment setting, which corresponds to a structured partition-matroid constraint, with $O ( T ^ { 4 / 5 } ) ~ ( 1 - 1 / e )$ -regret. Ofline-to-online reductions based on Blackwell approachability later gave a general explanation of how robust greedy algorithms can be converted into online algorithms and bandit analogues (Niazadeh et al., 2023). Stochastic full-bandit variants have also been studied under cardinality, knapsack, and general matroid constraints (Nie et al., 2022, 2023; Fourati et al., 2024; Nie et al., 2025).

Turning an ofline (1 − 1/e)-approximation into a bandit algorithm faces a basic obstacle: most existing ofline methods evaluate auxiliary subsets that need not satisfy the matroid constraint. Such queries are harmless in the ofline value-oracle model, where the algorithm may inspect the objective on any subset. In the bandit model, however, the queried set is also the action played by the learner. Under the original, or strict feasible-query, model, every such set—including an exploratory set used only to construct an estimator—must be independent.

Zhang et al. (2019) made this obstruction precise for the multilinear-extension approach. They showed that, for some matroid, there is no function-independent randomized rounding scheme supported on feasible sets whose single observed value is an unbiased estimate of the multilinear extension for every submodular objective. Motivated by this impossibility, they introduced responsive bandit submodular maximization: the learner may query an infeasible set and observe its function value, but receives zero reward on that round. Under this relaxed feedback model, they obtained $\begin{array} { r } { O ( T ^ { 8 / 9 } ) \ ( 1 - 1 / e ) – \mathrm { r e g r e t } } \end{array}$ . However, in the original adversarial bandit model, where every queried set must be feasible, no sublinear $( 1 - 1 / e )$ -regret guarantee was known for arbitrary monotone submodular rewards under general matroid constraints. Indeed, after establishing such a guarantee for partition matroids, Wan et al. (2023) conjectured that sublinear regret might fail for general matroid constraints.

We revisit this problem under the original strict feasible-query model. Our work is organized around the following question:

Can we design a bandit submodular maximization algorithm that achieves sublinear

$( 1 - 1 / e )$ -regret under general matroid constraints in polynomial time?

We answer this question afirmatively, thereby disproving the conjecture of Wan et al. (2023). Our approach builds on the recent Poisson-process method for ofline submodular maximization (Ganz Rozenman et al., 2026). We first formulate its online counterpart as a policy learning problem: the current base of the Poisson walk is the context, a feasible exchange is an action, and each comparator base induces a state dependent exchange policy. This contextual-bandit viewpoint, which has not appeared in prior work on this problem, gives a possible route to sublinear regret. However, this approach only produce a information theoretic guarantee, because the number of policies is exponentially large and require exponential time and space to excute. We therefore introduce a technique to compress the resulting policy mixture into a fractional base and couple it with each current base of the Poisson walk while preserving the required exchange marginals. This new technique yields an polynomial time algorithm with sublinear $( 1 - 1 / e )$ -regret under general matroid constraints.

Main result. Our main result gives the optimal polynomial-time approximation factor $1 - 1 / e$ in expected approximate regret, formally defined in Section 2. Concretely, we obtain the following theorem, which is an informal version of Theorem 5.11.

Theorem 1.1. Let $T \geq 2$ , and let M be a rank-k matroid on n elements. There exists a randomized oracle-polynomial algorithm that makes exactly one feasible value query per round and satisfies

$$
\left( 1 - \frac { 1 } { e } \right) \operatorname* { m a x } _ { O \in { \mathcal { B } } } \sum _ { t = 1 } ^ { T } f _ { t } ( O ) - { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( S _ { t } ) \right] = \widetilde { O } \left( n ^ { 1 / 3 } k ^ { 2 / 3 } T ^ { 2 / 3 } \right) .
$$

To our knowledge, this is the first polynomial time sublinear $( 1 - 1 / e )$ -regret guarantee for bandit monotone submodular maximization in the strict feasible-query bandit problem under general matroid constraint. The dependence on $T$ matches the $T ^ { 2 / 3 }$ scale attained for previous results on uniform matroid and partition matroid (Streeter and Golovin, 2008; Wan et al., 2023).

## 1.1 Technical Overview

The starting point is the Poisson process approximation algorithm of Ganz Rozenman et al. (2026). It evolves a feasible base through a continuous-time Poisson process, performing at each arrival a single-element exchange that satisfies a suitable local improvement condition. These local conditions imply a diferential inequality whose integration yields the $1 - 1 / e$ approximation guarantee. This form is particularly well suited to bandit feedback because the subset maintained by the algorithm can remain feasible throughout. The dificulty is that an online learner must choose each exchange before observing the current reward function $f _ { t }$ and hence cannot directly enforce the required local condition. The Poisson analysis localizes the resulting loss: the gap in the final approximation is controlled by the accumulated deficits of the exchanges actually taken. Our task is therefore to learn these exchanges from one feasible query per round.

Why the standard meta-framework does not apply. A standard way to onlineize a sequential approximation algorithm is to associate a no-regret linear optimizer with each update step: on every round, the optimizers generate the updates in sequence, and the optimizer at a given step is trained on the marginal objective induced by the preceding updates (Streeter and Golovin, 2008; Streeter et al., 2009; Niazadeh et al., 2023). This template presumes that each step has a fixed action domain and that the ofline compar ator can be represented by a fixed action for the corresponding optimizer. The random number of Poisson arrivals can be handled by truncation, but the state dependence of an arrival cannot. Indeed, at a given arrival, both the set of feasible exchanges and their gains depend on the current random base A. Even for a fixed comparator base O, the exchange that certifies the Poisson improvement condition is selected through a Brualdi matching between A and O, and therefore varies with A rather than being a fixed pair $( i , j )$ . Thus, an online optimizer whose actions are individual exchanges has no fixed comparator valid across rounds.

First try: an information-theoretic policy learner. To address the above barrier, an action must instead specify how to exchange from every possible current base, that is, it must be a state-dependent exchange policy. For every base O, fix a complete exchange policy π<sub>O</sub>: whenever the Poisson walk is at a base A, the policy chooses a uniformly random element $i \in A$ and inserts the element matched to i by a fixed Brualdi bijection from A to O. The policy class thus contains, for every possible hindsight base, a rule that is locally correct at every state that the walk may visit. Treating the current base as the context and a feasible exchange as the action, one may run the Exp4 algorithm over this finite policy class (Auer et al., 2002). With standard exploration, its finite-policy regret bound controls the accumulated Poisson deficits and yields sublinear approximation regret. Although the policy class is exponentially large, the regret depends only logarithmically on its cardinality. Computationally, however, Exp4 maintains one weight per policy and therefore requires exponential time and memory.

Using balanced fractional exchange to compress the policy mixture. To resolve the computational issue, a key observation is that the Poisson process analysis does not require the identity of the mixed comparator. Let $W _ { t }$ be the distribution over comparator policies in the above exponential time algorithm and set

$$
x _ { t } \triangleq \sum _ { O \in B } W _ { t } ( O ) \mathbf { 1 } _ { O } \in P ( \mathcal { M } ) .
$$

When the Brualdi exchanges are averaged according to $W _ { t }$ , the resulting exchange distribution has two simple marginals: the element removed from the current base is uniform, and the element inserted has distribution $x _ { t } / k$ . The Poisson analysis uses only these marginals, not the identity of the policy from which an exchange originated. Thus the exponentially supported distribution $W _ { t }$ can be replaced by the single fractional base $x _ { t }$ . We can therefore learn $x _ { t }$ directly, without maintaining any distribution over bases.

The remaining question is whether these two marginals can be realized at every current base using only feasible exchanges. This is precisely the role of a balanced fractional exchange. Given $A$ and $x _ { t }$ , we construct a nonnegative matrix $Q _ { A } ^ { x _ { t } }$ supported on pairs $( i , j )$ for which $A - i + j$ is a base and satisfying

$$
\sum _ { j \in \mathcal { U } } Q _ { A } ^ { x _ { t } } ( i , j ) = 1 \quad ( i \in A ) , \qquad \sum _ { i \in A } Q _ { A } ^ { x _ { t } } ( i , j ) = x _ { t , j } \quad ( j \in \mathcal { U } ) .
$$

Sampling an exchange from $Q _ { A } ^ { x _ { t } } / k$ therefore reproduces exactly the deletion and insertion marginals of the policy mixture. We prove that, for every current base A and every $x _ { t } \in P ( \mathcal { M } )$ such a matrix exists and can be constructed in polynomial time as a transportation flow on the bipartite exchange graph of A.

From balanced exchanges to a linear bandit. The marginal identities of a balanced exchange allow us to aggregate the state-dependent deficits along the Poisson trajectory without tracking the individual comparator-specific exchange rules. Combining these identities with a local linearization, we upper-bound the cumulative Poisson deficit by the regret of an online linear optimization problem over the matroid base polytope: the learner chooses a fractional base $x _ { t } .$ , and every hindsight base O is represented by the fixed vertex $\mathbf { 1 } _ { O }$ . The original bandit submodular problem is thereby reduced to a specialized linear bandit problem. Its linear loss is not observed directly, but must be estimated from one feasible value query, which motivates the separate estimator construction below.

A feasible leave-one-out estimator. Constructing the estimator requires a further idea independent of the balanced-exchange representation. The local improvement condition in the original Poisson algorithm is expressed through the gradient gain $\partial _ { j } F ( \tau { \bf 1 } _ { A } ) - \partial _ { i } F ( \tau { \bf 1 } _ { A } )$ Directly sampling its insertion term may require evaluating $f ( S + j )$ for $S \sim \tau A$ , which need not be feasible when S contains the element i that must be removed before inserting j. The leave-one-out gain avoids this obstruction by sampling $R \sim \tau ( A - i )$ : both $R + j \subseteq A - i + j$ and $R + i \subseteq A$ are feasible. Moreover, it equals the normalized multilinear jump caused by the exchange and dominates the original gradient gain, so it preserves the Poisson analysis. After linearization leaves only an insertion loss, a fair-coin randomization between querying $R + j$ and R gives a nonnegative unbiased estimate of this loss. Thus the required feedback is obtained from one feasible value query.

The remaining online-learning step is standard: importance weighting turns the one-query statistic into an unbiased estimate of the linear loss, and entropic mirror descent controls its cumulative regret. Together with a capped Poisson simulation and polynomial-time implementations of the balanced exchange flow and the entropy projection, we obtain the mentioned result.

## 1.2 Related Work

Ofline submodular maximization. The cardinality-constrained greedy algorithm and its extensions initiated the approximation theory of monotone submodular maximization (Nemhauser et al., 1978). Under a matroid constraint, continuous greedy and pipage or swap rounding achieve $1 - 1 / e$ (Vondrák, 2008; Călinescu et al., 2011; Chekuri et al., 2010). Non-oblivious local-search methods provide alternative proofs of the same factor (Filmus and Ward, 2014; Buchbinder and Feldman, 2024). The recent Poisson-process method of Ganz Rozenman et al. (2026) realizes the approximation through a continuous-time base walk using few single-element exchanges.

Adversarial online and bandit submodular maximization. Early work developed online greedy and assignment algorithms for monotone submodular rewards (Streeter and Golovin, 2008; Streeter et al., 2009). General ofline-to-online frameworks transform robust greedy procedures through Blackwell approachability and its bandit analogue (Niazadeh et al., 2023). Continuous DR-submodular methods obtain bandit guarantees through zeroth-order gradient estimation; their discrete responsive-feedback reduction may evaluate sets outside the original feasible family (Zhang et al., 2019). For adversarial rewards under strict feasible feedback, Wan et al. (2023) obtain $( 1 - 1 / e )$ -regret for partition matroids by exploiting a product-form feasible extension. Si Salem et al. (2024) handle general matroids for the restricted class of weighted thresholdpotential objectives through a concave relaxation, including a bandit variant. The present work treats arbitrary monotone submodular objectives and arbitrary matroids without a product decomposition or a concave-relaxation assumption.

Stochastic submodular bandits. Several stochastic models exploit additional structure or richer observations. Linear submodular bandits assume that the unknown utility is a linear combination of known submodular components (Yue and Guestrin, 2011), while interactive submodular bandits impose an RKHS model and observe noisy marginal utilities (Chen et al., 2017). Closer to our feedback protocol is the stochastic full-bandit model, in which only the noisy total reward of the selected set is observed. Under a cardinality constraint, Nie et al. (2022) give an explore-then-commit algorithm for monotone rewards, and Fourati et al. (2024) improve the dependence on the cardinality bound through stochastic-greedy exploration; Tajdini et al. (2024) subsequently derive the first minimax lower bounds together with a nearly matching upper bound for a restricted algorithm class. Nie et al. (2023) give a black-box reduction from robust ofline approximation algorithms to stochastic full-bandit algorithms, with applications to monotone submodular maximization under cardinality and knapsack constraints. For general matroids, specializing the monotone $k _ { \mathrm { s u b } }$ -submodular result of Nie et al. (2025) to $k _ { \mathrm { s u b } } = 1$ yields ordinary set-submodular maximization with approximation factor $1 / 2$ and $\widetilde { O } ( n ^ { 1 / 3 } k T ^ { 2 / 3 } )$ expected $1 / 2 \AA$ regret. For unconstrained nonmonotone objectives that are submodular in expectation, Fourati et al. (2023) obtain a stochastic full-bandit guarantee. These results rely on a fixed mean reward function or other stochastic structure, whereas our reward sequence may be chosen adversarially.

Bandit submodular minimization. For unconstrained submodular minimization, the convex Lovász extension permits a substantially diferent reduction to online convex optimization. Hazan and Kale (2012) give eficient full-information and bandit algorithms, including an $O ( n T ^ { 2 / 3 } )$ adversarial bandit regret bound. Cardoso and Cummings (2019) develop diferentially private variants under both feedback models, and Ito (2022) obtain best-of-both-worlds guarantees with gap-dependent logarithmic regret in stochastic environments and robustness to adversarial corruptions. These convex-extension techniques do not directly apply to maximizing a submodular reward under a matroid constraint, where the objective is nonconcave and the relevant benchmark is approximate.

## 1.3 Paper Organization

Section 2 introduces the model and notation. Section 3 extends the Poisson-walk algorithm of Ganz Rozenman et al. (2026) to general exchange distributions and quantifies the resulting terminal approximation error. Section 4 establishes balanced fractional exchanges. Section 5 presents the bandit algorithm and its regret analysis. Supporting arguments appear in the appendices.

## 2 Preliminaries

Submodular functions. For a finite ground set U and a set $S \subseteq { \mathcal { U } } ,$ , let $\mathbf { 1 } _ { S } \in \{ 0 , 1 \} ^ { \mathcal { U } }$ denote the incidence vector of S. For an element $u \in \mathcal { U } .$ , write $S + u \triangleq S \cup \{ u \}$ and $S - u \triangleq S \setminus \{ u \}$ A function $f : 2 ^ { \mathcal { U } } \to [ 0 , 1 ]$ is normalized if $f ( \varnothing ) = 0 .$ , monotone if for any $A \subseteq B$ we have $f ( A ) \leq f ( B )$ . We call a set function is submodular if it satisfies following property for any $A \subseteq B , \ u \notin B$ :

$$
f ( A + u ) - f ( A ) \geq f ( B + u ) - f ( B ) .
$$

Matroids. Let $\mathcal { M } = ( \mathcal { U } , \mathcal { T } )$ be a matroid of rank $k \geq 1$ . The family of bases is denoted by B. For a base A, define its feasible single-exchange set

$$
{ \mathcal { E } } ( A ) \triangleq \left\{ ( i , j ) : i \in A , \ j \in { \mathcal { U } } , \ A - i + j \in { \mathcal { B } } \right\} .\tag{2.1}
$$

Self-loops (i, i) are included. If $j \in A$ and $( i , j ) \in { \mathcal { E } } ( A )$ , then necessarily $i = j$

The base polytope is

$$
P ( \mathcal { M } ) \triangleq \operatorname { c o n v } \left\{ \mathbf { 1 } _ { B } : B \in \mathcal { B } \right\} .\tag{2.2}
$$

Every $x \in P ( \mathcal { M } )$ satisfies $0 \leq x _ { u } \leq 1$ and $\textstyle \sum _ { u \in { \mathcal { U } } } x _ { u } = k$ . We need Brualdi’s strong basis-exchange theorem (Brualdi, 1969) in our analysis.

Lemma 2.1 ((Brualdi, 1969)). For every pair of bases $A , B \in B$ , there is a bijection $h _ { A , B }$ $A  B$ such that

$$
A - i + h _ { A , B } ( i ) \in \mathcal { B } \qquad f o r \ e v e r y \ i \in A .
$$

The bijection may be chosen so that $h _ { A , B } ( i ) = i$ for all $i \in A \cap B$

Bandit protocol. An oblivious adversary fixes a sequence $f _ { 1 } , \dots , f _ { T } : 2 ^ { \mathcal { U } } \to [ 0 , 1 ]$ of normalized, monotone submodular functions before the interaction. In each round $t = 1 , \dots , T$ , the learner chooses $S _ { t } \in \mathcal { T }$ without observing $f _ { t } .$ , then observes only the scalar $f _ { t } ( S _ { t } )$ and receives this value. Because every independent set extends to a base and every $f _ { t }$ is monotone,

$$
\operatorname* { m a x } _ { S \in \mathcal { T } } \sum _ { t = 1 } ^ { T } f _ { t } ( S ) = \operatorname* { m a x } _ { O \in \mathcal { B } } \sum _ { t = 1 } ^ { T } f _ { t } ( O ) .
$$

Accordingly, for $\alpha \in [ 0 , 1 ]$ , the expected α-regret is

$$
R _ { \alpha } ( T ) \triangleq \alpha \operatorname* { m a x } _ { O \in { \cal B } } \sum _ { t = 1 } ^ { T } f _ { t } ( O ) - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( S _ { t } ) \right] .
$$

Multilinear extension. Fix a monotone submodular function $f : 2 ^ { \mathcal { U } } \to [ 0 , 1 ]$ . Its multilinear extension is

$$
F ( x ) \triangleq \mathbb { E } _ { R \sim x } [ f ( R ) ] ,\tag{2.3}
$$

where every element u is included independently with probability $x _ { u }$ . For $S \subseteq { \mathcal { U } }$ and $\tau \in [ 0 , 1 ]$ ， the notation $R \sim \tau S$ means that every element of $S$ is included independently with probability $\tau$ and no element outside S is included.

The following two standard properties of the multilinear extension are basic ingredients in the analysis of continuous greedy (Vondrák, 2008; Călinescu et al., 2011). We include their short proofs for completeness.

Lemma 2.2. For every $u \in \mathcal { U }$

$$
\partial _ { u } F ( x ) = \mathbb { E } _ { R \sim x \mid _ { M - u } } [ f ( R + u ) - f ( R ) ] .\tag{2.4}
$$

The derivative is independent of $x _ { u }$ and is nonincreasing in every other coordinate. Consequently, $i f x \le y$ coordinatewise, then

$$
\partial _ { u } F ( x ) \geq \partial _ { u } F ( y ) .
$$

Proof. Because F is multilinear, conditioning on all coordinates except u gives

$$
F ( x ) = x _ { u } \mathbb { E } [ f ( R + u ) ] + ( 1 - x _ { u } ) \mathbb { E } [ f ( R ) ] ,
$$

where $R \sim x | \boldsymbol { u } _ { - u }$ . Diferentiating in $x _ { u }$ proves (2.4) and shows independence from $x _ { u }$

Now let x and y agree on coordinate u and satisfy $x _ { v } ~ \leq ~ y _ { v }$ for every $v \neq u .$ . Couple $R _ { x } \sim x | _ { \mathcal { U } - u }$ and $R _ { y } \sim y | _ { \mathcal { U } - u }$ by using common uniform random variables, so that $R _ { x } \subseteq R _ { y }$ almost surely. Submodularity gives

$$
f ( R _ { x } + u ) - f ( R _ { x } ) \geq f ( R _ { y } + u ) - f ( R _ { y } ) .
$$

Taking expectations proves monotonicity in every coordinate other than u. Independence from $x _ { u }$ completes the proof for arbitrary $x \leq y$ □

Algorithm 1: Poisson Base Walk with General Exchange Distributions   
Input: Matroid $\mathcal { M } ;$ starting base $A _ { 0 } ;$ starting time $\varepsilon ;$ exchange distributions $p _ { s } ( \cdot \mid A )$ supported   
on ${ \mathcal { E } } ( A )$   
1 Sample the event times $\varepsilon < \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { M } \leq 1$ of a nonhomogeneous Poisson process with   
intensity $\lambda ( s ) = k / s . ;$   
2 for $J = 1 , \ldots , M$ do   
3 Draw $e _ { J } = ( i _ { J } , j _ { J } ) \sim p _ { \tau _ { J } } ( \cdot \mid A _ { J - 1 } ) .$ ;   
4 Set $A _ { J } = A _ { J - 1 } - i _ { J } + j _ { J } . ;$   
5 return $A _ { M } . ;$

Lemma 2.3. For every $x \in [ 0 , 1 ] ^ { \mathcal { U } }$ and every set $O \subseteq { \mathcal { U } } ,$

$$
\sum _ { o \in O } \partial _ { o } F ( x ) \geq f ( O ) - F ( x ) .\tag{2.5}
$$

$P r o o f .$ Let $x \vee { \bf 1 } _ { O }$ denote the coordinatewise maximum. Monotonicity gives $F ( x \vee \mathbf { 1 } _ { O } ) \geq F ( \mathbf { 1 } _ { O } ) =$ $f ( O )$ . Increase the coordinates in O one at a time from their values in x to one. By Lemma 2.2, every marginal encountered along this monotone path is at most its value at x. Therefore

$$
F ( x \vee \mathbf { 1 } _ { O } ) - F ( x ) \leq \sum _ { o \in O } ( 1 - x _ { o } ) \partial _ { o } F ( x ) \leq \sum _ { o \in O } \partial _ { o } F ( x ) .
$$

Combining the two displays proves the claim.

## 3 Poisson Base Walk with General Exchange Distribution

The Poisson-process algorithm of Ganz Rozenman et al. (2026) is an important component of our bandit algorithm. We formulate the base walk with a general distribution over feasible exchanges, which is the form used throughout this paper, and then recover their ofline algorithm as a particular exchange rule. For simplicity, we call the algorithm of Ganz Rozenman et al. (2026) the GKSS algorithm, after its authors. The distinction lies in the exchange rule: the GKSS algorithm can choose an exchange using the entire objective $f ,$ whereas the bandit learner must choose one before observing $f _ { t }$

Poisson base walk. Fix a monotone submodular function f with multilinear extension $F _ { ; }$ a rank-k matroid ${ \mathcal { M } } ,$ a starting time $\varepsilon \in ( 0 , 1 )$ , and an arbitrary starting base $A _ { 0 }$ . The walk runs a nonhomogeneous Poisson process on [ε, 1] with intensity $\lambda ( s ) \triangleq k / s$ . For every time s and base A, let $p _ { s } ( \cdot \mid A )$ be a measurable distribution supported on the feasible exchanges $\mathcal { E } ( A )$ We call each arrival of this Poisson process an event: event $J$ occurs at time $\tau _ { J }$ , samples one feasible exchange $e _ { J } = ( i _ { J } , j _ { J } )$ , and updates the base from $A _ { J - 1 }$ to $A _ { J }$ . Thus an event is an internal jump of the simulated base trajectory, not a value query.

Let $A ( s - )$ denote the base immediately before time s, and let $A ( s )$ denote the right-continuous post-event state. In particular, $A ( \tau _ { J } - ) = A _ { J - 1 }$ and $A ( \tau _ { J } ) = A _ { J }$ . Every state in Algorithm 1 is a base, so the process requires no rounding. Moreover, the number M of exchange events satisfies

$$
M \sim { \mathrm { P o i s s o n } } ( \mu ) , \qquad \mu \triangleq \int _ { \varepsilon } ^ { 1 } \frac k s d s = k \log ( 1 / \varepsilon ) ,\tag{3.1}
$$

and hence the expected number of exchanges is $k \log ( 1 / \varepsilon )$ . Conditional on $M ,$ the event times can be sampled directly as described in Section A.

The GKSS algorithm is the objective-dependent specialization of Algorithm 1 obtained as follows. At an event time s with current base $A .$ it computes a maximum-weight base

$$
B _ { s } ( A ) \in \underset { B \in \cal B } { \arg \operatorname* { m a x } } \sum _ { j \in { \cal B } } \partial _ { j } F ( s { \bf 1 } _ { A } ) .\tag{3.2}
$$

By Brualdi’s basis-exchange theorem, there is a bijection ${ h _ { A , B _ { s } ( A ) } : A  B _ { s } ( A ) }$ such that $A - i + h _ { A , B _ { s } ( A ) } ( i )$ is a base for every $i \in A .$ . The GKSS exchange distribution chooses i uniformly from A and sets $j = h _ { A , B _ { s } ( A ) } ( i )$ . Conditional on A and $s ,$ this exchange satisfies, for every comparator base $O _ { i }$

$$
\begin{array} { r l } & { \mathbb { E } \left[ \partial _ { j } F ( s { \mathbf { 1 } } _ { A } ) - \partial _ { i } F ( s { \mathbf { 1 } } _ { A } ) \mathbf { \Phi } \middle | A , s \right] = \displaystyle \frac { 1 } { k } \left( \sum _ { j \in B _ { s } ( A ) } \partial _ { j } F ( s { \mathbf { 1 } } _ { A } ) - \sum _ { i \in A } \partial _ { i } F ( s { \mathbf { 1 } } _ { A } ) \right) } \\ & { \quad \geq \displaystyle \frac { 1 } { k } \left( \sum _ { o \in O } \partial _ { o } F ( s { \mathbf { 1 } } _ { A } ) - \sum _ { i \in A } \partial _ { i } F ( s { \mathbf { 1 } } _ { A } ) \right) . } \end{array}\tag{3.3}
$$

The equality follows because both the deletion element and its image under the Brualdi bijection are uniform on their respective bases; the inequality follows from the definition of $B _ { s } ( A )$ . The GKSS rule satisfies this condition at every event time, which is crucial for achieving the $1 - 1 / e$ approximation factor.

The bandit algorithm in Section 5 instantiates Algorithm 1 with a diferent exchange distribution. Because that distribution is chosen without observing the current reward function, it cannot in general satisfy (3.3) at every event. The remainder of this section bounds the terminal value when the expected exchange gain falls below the right-hand side of (3.3).

Leave-One-Out exchange gain. For a feasible exchange $( i , j ) \in { \mathcal { E } } ( A )$ , define the leave-oneout exchange gain for a given submodular function $f$ and its multilinear extension $F _ { ; }$ a time $\tau \in [ 0 , 1 ]$ as

$$
G _ { f } ( A , \tau ; i , j ) \triangleq \mathbb { E } _ { R \sim \tau ( A - i ) } [ f ( R + j ) - f ( R + i ) ] .\tag{3.4}
$$

The two sets in this diference are feasible: $R + i \subseteq A$ and $R + j \subseteq A - i + j .$

We have following two properties of $G _ { f }$ that are used in the analysis of the Poisson process algorithm.

Lemma 3.1. For every $( i , j ) \in { \mathcal { E } } ( A )$ 9

$$
F ( \tau { \bf 1 } _ { A - i + j } ) - F ( \tau { \bf 1 } _ { A } ) = \tau G _ { f } ( A , \tau ; i , j ) .\tag{3.5}
$$

Proof. Both vectors in (3.5) agree outside coordinates i and $j .$ . Insert the intermediate vector $\tau { \bf 1 } _ { A - i }$ . Multilinearity in coordinate j gives

$$
F ( \tau { \bf 1 } _ { A - i + j } ) - F ( \tau { \bf 1 } _ { A - i } ) = \tau \partial _ { j } F ( \tau { \bf 1 } _ { A - i } ) ,
$$

while multilinearity in coordinate i gives

$$
F ( \tau { \bf 1 } _ { A } ) - F ( \tau { \bf 1 } _ { A - i } ) = \tau \partial _ { i } F ( \tau { \bf 1 } _ { A - i } ) .
$$

Subtracting the second identity from the first and using (3.4) proves the claim.

Lemma 3.2. For every base A, time $\tau \in [ 0 , 1 ]$ , and feasible exchange $( i , j ) \in { \mathcal { E } } ( A )$ ，

$$
G _ { f } ( A , \tau ; i , j ) \geq \partial _ { j } F ( \tau { \bf 1 } _ { A } ) - \partial _ { i } F ( \tau { \bf 1 } _ { A } ) .\tag{3.6}
$$

Proof. The definition of $G _ { f }$ and Lemma 2.2 give, for every feasible exchange $( i , j )$

$$
\begin{array} { r l } & { G _ { f } ( A , \tau ; i , j ) = \partial _ { j } F ( \tau { \bf 1 } _ { A - i } ) - \partial _ { i } F ( \tau { \bf 1 } _ { A - i } ) } \\ & { \qquad \quad \geq \partial _ { j } F ( \tau { \bf 1 } _ { A } ) - \partial _ { i } F ( \tau { \bf 1 } _ { A } ) , } \end{array}
$$

where the derivative in coordinate i is unchanged when coordinate i itself is varied.

We first give the event-sum identity needed below. Its proof is included because the discount factor in the regret argument comes directly from the cancellation between the intensity $k / s$ and a factor s in the event statistic.

Lemma 3.3. Let H be any bounded event statistic, where $H ( A , s , e )$ assigns a real value to a base A, a time $s \in [ \varepsilon , 1 ]$ , and a feasible exchange $e \in { \mathcal { E } } ( A )$ . Assume that $H ( A , \cdot , e )$ is measurable and continuous except at finitely many points. Then

$$
\mathbb { E } \left[ \sum _ { J = 1 } ^ { M } H ( A ( \tau _ { J } - ) , \tau _ { J } , e _ { J } ) \right] = \int _ { \varepsilon } ^ { 1 } \frac { k } { s } \mathbb { E } \left[ \sum _ { e \in \mathcal { E } ( A ( s - ) ) } p _ { s } ( e \mid A ( s - ) ) H ( A ( s - ) , s , e ) \right] d s ,\tag{3.7}
$$

where $A ( s - )$ is the base held immediately before time $s ,$ so $A ( \tau _ { J } - )$ is the state immediately before the Jth exchange.

Proof. Let K bound $| H |$ , and note that $\lambda ( s ) \leq k / \varepsilon$ on $[ \varepsilon , 1 ]$ . Consider a partition $\varepsilon = t _ { 0 } < t _ { 1 } <$ $\cdots < t _ { m } = 1$ with mesh $\Delta ,$ and a cell $I _ { r } = ( t _ { r - 1 } , t _ { r } ]$ . Conditional on the history at $t _ { r - 1 }$ , the number of underlying Poisson events in $I _ { r }$ has mean

$$
\Lambda _ { r } = \int _ { t _ { r - 1 } } ^ { t _ { r } } \lambda ( s ) d s = { \cal O } ( \Delta ) .
$$

The probability of two or more events in this cell is

$$
1 - e ^ { - \Lambda _ { r } } ( 1 + \Lambda _ { r } ) = O ( \Delta ^ { 2 } ) ,
$$

uniformly in $^ r .$ . Thus the total expected contribution from cells containing at least two events is at most

$$
K \sum _ { r = 1 } ^ { m } \mathbb { E } [ N ( I _ { r } ) { \bf 1 } \{ N ( I _ { r } ) \geq 2 \} ] = O ( m \Delta ^ { 2 } ) = O ( \Delta ) ,
$$

where the implicit constant depends only on $K , k ,$ and $\varepsilon .$

On a cell containing exactly one event at time s, the pre-event base equals the base at the left endpoint, and the conditional mark distribution is $p _ { s } ( \cdot \ | \ A ( t _ { r - 1 } ) )$ . The standard density of the unique event in a nonhomogeneous Poisson process gives, conditionally on the left-endpoint history,

$$
\begin{array} { r l } & { \mathbb { E } [ \mathbf { 1 } \{ N ( I _ { r } ) = 1 \} H ( A ( s - ) , s , e )  \mathcal { F } _ { t _ { r - 1 } } ] } \\ & { \quad = \displaystyle \int _ { I _ { r } } \exp ( - \int _ { t _ { r - 1 } } ^ { t _ { r } } \lambda ( u ) d u ) \lambda ( s ) \sum _ { e } p _ { s } ( e \mid A ( t _ { r - 1 } ) ) H ( A ( t _ { r - 1 } ) , s , e ) d s . } \end{array}
$$

The exponential factor is $1 + O ( \Delta )$ uniformly. Replacing $A ( t _ { r - 1 } )$ by the actual pre-event state inside the integral changes only paths with an earlier event in the same cell, whose total contribution is already $O ( \Delta )$ after summing over cells. Therefore, summing over r gives

$$
\mathbb { E } \left[ \sum _ { J = 1 } ^ { M } H ( A ( \tau _ { J } - ) , \tau _ { J } , e _ { J } ) \right] = \sum _ { r = 1 } ^ { m } \int _ { I _ { r } } \lambda ( s ) \mathbb { E } \left[ \sum _ { e } p _ { s } ( e \mid A ( s - ) ) H ( A ( s - ) , s , e ) \right] d s + O ( \Delta ) .
$$

The integrand is bounded by $K \lambda ( s )$ and is measurable. Letting the mesh tend to zero and applying dominated convergence proves (3.7). A bounded measurable H can be obtained by the usual simple-function approximation; the stated regularity already covers every function used in this paper. □

For a comparator base $O ,$ , define the local benchmark

$$
\zeta _ { f , O } ( A , s ) \triangleq \frac { 1 } { k } \left( \sum _ { o \in O } \partial _ { o } { \cal F } ( s { \bf 1 } _ { A } ) - \sum _ { a \in A } \partial _ { a } { \cal F } ( s { \bf 1 } _ { A } ) \right) ,\tag{3.8}
$$

and the discount weight

$$
w ( s ) \triangleq s \exp ( - ( 1 - s ) ) .\tag{3.9}
$$

## Lemma 3.4. Let

$$
q ( s ) \triangleq \mathbb { E } [ F ( s \mathbf { 1 } _ { A ( s ) } ) ] .\tag{3.10}
$$

Then q is absolutely continuous and, for almost every $s \in [ \varepsilon , 1 ]$ ，

$$
q ^ { \prime } ( s ) = { \mathbb E } \left[ \sum _ { a \in A ( s - ) } \partial _ { a } F ( s { \mathbf { 1 } } _ { A ( s - ) } ) + k \sum _ { e \in { \mathcal E } ( A ( s - ) ) } p _ { s } ( e \mid A ( s - ) ) G _ { f } ( A ( s - ) , s ; e ) \right] .\tag{3.11}
$$

Proof. Fix $r \in [ \varepsilon , 1 ]$ . Along every sample path, the map $s \mapsto F { \bigl ( } s \mathbf { 1 } _ { A ( s ) } { \bigr ) }$ changes continuously between event times and has finitely many jumps. Holding the base A fixed,

$$
\frac { d } { d s } F ( s { \bf 1 } _ { A } ) = \sum _ { a \in A } \partial _ { a } F ( s { \bf 1 } _ { A } ) .\tag{3.12}
$$

At an event with pre-event base A and mark $e = ( i , j )$ , Lemma 3.1 gives the jump

$$
F ( s { \mathbf { 1 } } _ { A - i + j } ) - F ( s { \mathbf { 1 } } _ { A } ) = s G _ { f } ( A , s ; e ) .\tag{3.13}
$$

The pathwise decomposition is therefore

$$
\begin{array} { r l } { \displaystyle F ( \mathbf { r 1 } _ { A ( r ) } ) - F ( \varepsilon \mathbf { 1 } _ { A _ { 0 } } ) = \int _ { \varepsilon } ^ { r } \sum _ { a \in A ( s - ) } \partial _ { a } F ( s \mathbf { 1 } _ { A ( s - ) } ) d s } & { } \\ { \displaystyle + \sum _ { J : \tau _ { J } \leq r } \tau _ { J } G _ { f } ( A ( \tau _ { J } - ) , \tau _ { J } ; e _ { J } ) . } \end{array}\tag{3.14}
$$

All terms are integrable: derivatives and gains lie in $[ - 1 , 1 ]$ , and the expected event count is finite. Taking expectations and applying Lemma 3.3 to $H ( A , s , e ) \triangleq s G _ { f } ( A , s ; e )$ transforms the expected jump sum into

$$
\int _ { \varepsilon } ^ { r } \frac { k } { s } \mathbb { E } \left[ \sum _ { e } p _ { s } ( e \mid A ( s - ) ) s G _ { f } ( A ( s - ) , s ; e ) \right] d s .
$$

Substituting into (3.14) expresses $q ( r ) - q ( \varepsilon )$ as the integral of the right-hand side of (3.11). Hence $q$ is absolutely continuous and has that derivative almost everywhere. □

Theorem 3.5. Fix a comparator base O. For each event J, define the discounted deficit

$$
\begin{array} { r } { D _ { J } ( O ) \triangleq w ( \tau _ { J } ) \left( \zeta _ { f , O } ( A ( \tau _ { J } - ) , \tau _ { J } ) - G _ { f } ( A ( \tau _ { J } - ) , \tau _ { J } ; e _ { J } ) \right) . } \end{array}\tag{3.15}
$$

Then

$$
\mathbb { E } [ f ( A ( 1 ) ) ] \ge \left( 1 - e ^ { - ( 1 - \varepsilon ) } \right) f ( O ) - \mathbb { E } \left[ \sum _ { J = 1 } ^ { M } D _ { J } ( O ) \right]\tag{3.16}
$$

$$
\geq ( 1 - \varepsilon ) ( 1 - 1 / e ) f ( O ) - \mathbb { E } \left[ \sum _ { J = 1 } ^ { M } D _ { J } ( O ) \right] .\tag{3.17}
$$

Proof. Define the instantaneous expected deficit

$$
\delta ( s ) \triangleq \mathbb { E } \left[ \zeta _ { f , O } ( A ( s - ) , s ) - \sum _ { e \in \mathcal { E } ( A ( s - ) ) } p _ { s } ( e \mid A ( s - ) ) G _ { f } ( A ( s - ) , s ; e ) \right] .\tag{3.18}
$$

Use (3.8) to rewrite the generator (3.11):

$$
q ^ { \prime } ( s ) = \mathbb { E } \left[ \sum _ { a \in A ( s - ) } \partial _ { a } F ( s { \mathbf { 1 } } _ { A ( s - ) } ) + k \zeta _ { f , O } ( A ( s - ) , s ) \right] - k \delta ( s )
$$

$$
= \mathbb { E } \left[ \sum _ { o \in O } \partial _ { o } F ( s { \bf 1 } _ { A ( s - ) } ) \right] - k \delta ( s ) .
$$

Applying Lemma 2.3 at $s \mathbf { 1 } _ { A ( s - ) }$ and taking expectations gives

$$
\mathbb { E } \left[ \sum _ { o \in O } \partial _ { o } F ( s \mathbf { 1 } _ { A ( s - ) } ) \right] \geq f ( O ) - q ( s ) .
$$

Therefore, for almost every $s ,$

$$
q ^ { \prime } ( s ) + q ( s ) \geq f ( O ) - k \delta ( s ) .\tag{3.19}
$$

Multiply by $e ^ { s }$ and integrate from $\varepsilon$ to 1:

$$
e q ( 1 ) - e ^ { \varepsilon } q ( \varepsilon ) \geq f ( O ) ( e - e ^ { \varepsilon } ) - \int _ { \varepsilon } ^ { 1 } e ^ { s } k \delta ( s ) d s .
$$

Since $q ( \varepsilon ) \geq 0$ , division by e yields

$$
q ( 1 ) \geq \big ( 1 - e ^ { - ( 1 - \varepsilon ) } \big ) f ( O ) - \int _ { \varepsilon } ^ { 1 } e ^ { - ( 1 - s ) } k \delta ( s ) d s .\tag{3.20}
$$

Apply Lemma 3.3 with

$$
H ( A , s , e ) \triangleq s e ^ { - ( 1 - s ) } \big ( \zeta _ { f , O } ( A , s ) - { G } _ { f } ( A , s ; e ) \big ) .
$$

The factor s cancels the intensity $k / s$ , so the expected event sum equals the integral in (3.20).   
Also $q ( 1 ) = \mathbb { E } [ F ( \mathbf { 1 } _ { A ( 1 ) } ) ] = \mathbb { E } [ f ( A ( 1 ) ) ]$ . This proves (3.16).

Finally, the function $a \mapsto 1 - e ^ { - a }$ is concave on [0, 1] and vanishes at zero. Hence

$$
1 - e ^ { - ( 1 - \varepsilon ) } \geq ( 1 - \varepsilon ) ( 1 - 1 / e ) ,
$$

which proves (3.17).

The role of the deficit term distinguishes Theorem 3.5 from the analysis of Ganz Rozenman et al. (2026). Under their objective-dependent exchange rule, the conditional expected deficit at every event is nonpositive, so the expected cumulative deficit is nonpositive and can be discarded from the lower bound. This gives the coeficient $1 - e ^ { - ( 1 - \varepsilon ) }$ , which tends to $1 - 1 / e { \mathrm { ~ a s ~ } } \varepsilon \downarrow 0$ . In contrast, Algorithm 1 permits a general exchange distribution $p _ { s } ( \cdot \mid A )$ , which need not satisfy the exchange gain condition of GKSS. Our theorem therefore retains the cumulative deficit explicitly and quantifies exactly how its expected value degrades the approximation guarantee. In the online setting, bounding this term converts the learner’s cumulative local error into approximation regret.

## 4 Balanced Fractional Exchanges

The discussion in Section 1.1 suggests that an exponential mixture of comparator-specific Brualdi rules contains more information than the Poisson analysis needs. Its uniform deletion marginal and fractional insertion marginal can instead be represented by a single point $x \in P ( \mathcal { M } )$ . This section constructs a distribution over feasible exchanges with exactly these two marginals and shows that it can be computed by a polynomial-size transportation problem. The next section uses the marginal identities to derive the linear loss needed by the bandit algorithm.

Definition 4.1 (Balanced fractional exchange). Fix a base $A \in B$ and a point $x \in P ( \mathcal { M } )$ . A balanced fractional exchange from A with insertion marginal x is a nonnegative matrix

$$
Q _ { A } ^ { x } ( i , j ) , \qquad i \in A , \ j \in \mathcal { U } ,
$$

satisfying

$$
Q _ { A } ^ { x } ( i , j ) > 0 \Longrightarrow A - i + j \in B ,\tag{4.1}
$$

$$
\sum _ { j \in \mathcal { U } } Q _ { A } ^ { x } ( i , j ) = 1
$$

$$
\mathrm { f o r ~ e v e r y ~ } i \in A ,\tag{4.2}
$$

$$
\sum _ { i \in A } Q _ { A } ^ { x } ( i , j ) = x _ { j }
$$

$$
\mathrm { f o r ~ e v e r y ~ } j \in { \mathcal { U } } .\tag{4.3}
$$

The corresponding probability distribution on feasible exchanges is

$$
q _ { A } ^ { x } ( i , j ) \triangleq \frac { 1 } { k } Q _ { A } ^ { x } ( i , j ) .\tag{4.4}
$$

The row and column sums both have total mass k, so (4.4) is a probability distribution. Its two marginals are

$$
\mathbb { P } _ { ( i , j ) \sim q _ { A } ^ { x } } ( i = a ) = \frac { 1 } { k } \quad ( a \in A ) , \qquad \mathbb { P } _ { ( i , j ) \sim q _ { A } ^ { x } } ( j = u ) = \frac { x _ { u } } { k } \quad ( u \in \mathcal { U } ) .\tag{4.5}
$$

The following theorem shows that balanced fractional exchanges exist for every base and every fractional base.

Theorem 4.2. For every base $A \in B$ and every fractional base $x \in P ( \mathcal { M } )$ , a balanced fractional exchange $Q _ { A } ^ { x }$ exists.

Proof. By the definition of the base polytope, there are coeficients $\lambda _ { B } \geq 0$ with $\textstyle \sum _ { B \in B } \lambda _ { B } = 1$ such that

$$
x = \sum _ { B \in B } \lambda _ { B } \mathbf { 1 } _ { B } .\tag{4.6}
$$

For every B with $\lambda _ { B } > 0$ , choose a Brualdi bijection $h _ { A , B } : A \to B$ as in Lemma 2.1, and define

$$
Q _ { A } ^ { x } ( i , j ) \triangleq \sum _ { B \in \mathcal { B } } \lambda _ { B } \mathbf { 1 } \{ h _ { A , B } ( i ) = j \} .\tag{4.7}
$$

Every positive summand in (4.7) is supported on a feasible exchange. For a fixed $i \in A$ , each bijection assigns exactly one image to i, so

$$
\sum _ { j \in \mathcal { U } } Q _ { A } ^ { x } ( i , j ) = \sum _ { B \in \mathcal { B } } \lambda _ { B } = 1 .
$$

For a fixed $j \in \mathcal { U }$ , bijectivity implies that exactly one $i \in A$ satisfies $h _ { A , B } ( i ) = j$ when $j \in B$ and none does when $j \notin B$ . Hence

$$
\sum _ { i \in A } Q _ { A } ^ { x } ( i , j ) = \sum _ { B \in B } \lambda _ { B } \mathbf { 1 } \{ j \in B \} = x _ { j } ,
$$

where the last equality follows from (4.6). Thus all conditions in Definition 4.1 hold.

The preceding proof is existential because it begins with a base decomposition. A direct transportation formulation gives the desired oracle-polynomial construction.

Theorem 4.3. Fix a base A and a fractional base $x \in P ( \mathcal { M } )$ . Given x explicitly and access to an independence oracle for M, a balanced fractional exchange $Q _ { A } ^ { x }$ can be computed by a transportation flow on a bipartite graph with $k + n$ nonterminal vertices and at most kn edges.

Proof. Construct a directed network with a source s, a sink $t ,$ a left vertex for every $i \in A$ , and a right vertex for every $j \in \mathcal { U }$ . Add the following arcs:

• an arc $s \to i$ of capacity exactly 1 for each $i \in A ;$

• an arc $i  j$ of unbounded capacity whenever $A - i + j \in B ;$

• an arc $j  t$ of capacity exactly $x _ { j }$ for each $j \in \mathcal { U } .$

Here “capacity exactly” means an equal lower and upper bound. The support graph is found with at most kn independence-oracle calls: since A is a base, $A - i + j$ is a base exactly when it is independent.

A feasible flow of value k defines $Q ( i , j )$ as the flow on arc $i  j$ . Flow conservation at the left vertices gives the row equalities, flow conservation at the right vertices gives the column equalities, and the support arcs enforce feasibility. Conversely, every balanced exchange matrix defines such a feasible flow.

Feasibility of the network follows from Theorem 4.2. Standard feasible-circulation or transportation algorithms therefore return a balanced exchange in time polynomial in the network size. In the real-arithmetic oracle model used in this paper, the demands $x _ { j }$ may be arbitrary reals. □

Remark 4.4. When an explicit decomposition $\begin{array} { r } { x = \sum _ { r = 1 } ^ { m } \lambda _ { r } \mathbf { 1 } _ { B _ { r } } } \end{array}$ is available, one may sample from $q _ { A } ^ { x }$ without forming the transportation matrix. Draw $B _ { r }$ with probability $\lambda _ { r } .$ , compute a Brualdi bijection $h _ { A , B _ { r } }$ , draw i uniformly from A, and set $j = h _ { A , B _ { r } } ( i )$ . The resulting exchange has probability

$$
\frac { 1 } { k } \sum _ { r } \lambda _ { r } { \bf 1 } \{ h _ { A , B _ { r } } ( i ) = j \} = q _ { A } ^ { x } ( i , j ) .
$$

The transportation-flow implementation used by the main algorithm does not require such a decomposition.

Comparator vertices as a special case. I $\dot { \mathbf { \eta } } _ { \dot { \mathbf { \eta } } } x = \mathbf { 1 } _ { O }$ is the incidence vector of a comparator base, choose the one-term base decomposition supported on $O$ in the proof of Theorem 4.2. The resulting balanced exchange matrix is a single Brualdi bijection from A to $O .$ . In that case the deletion and insertion marginals are both uniform on their respective bases. For the multilinear extension $F$ of any monotone submodular function and every $\tau \in [ 0 , 1 ]$ ，

$$
\mathbb { E } _ { ( i , j ) \sim q _ { A } ^ { 1 _ { O } } } \left[ \partial _ { j } F ( \tau \mathbf { 1 } _ { A } ) - \partial _ { i } F ( \tau \mathbf { 1 } _ { A } ) \right] = \frac { 1 } { k } \left( \sum _ { o \in O } \partial _ { o } F ( \tau \mathbf { 1 } _ { A } ) - \sum _ { a \in A } \partial _ { a } F ( \tau \mathbf { 1 } _ { A } ) \right) .\tag{4.8}
$$

Thus every comparator vertex $x = \mathbf { 1 } _ { O }$ recovers the exact Brualdi identity. Allowing x to range over the convex hull of these vertices exposes a polynomial-dimensional decision variable while preserving the same local benchmark.

Balanced fractional exchanges therefore give an oracle-polynomial way to realize the two marginals of a distribution over comparator-specific Brualdi rules. In particular, comparator vertices recover the exact Brualdi identity, while arbitrary fractional bases interpolate these rules without representing a distribution over bases.

## 5 The Balanced-Exchange Bandit Algorithm

The first subsection presents the bandit algorithm and its linear-loss formulation. Section 5.2 establishes the guarantees of the bandit estimator. Section 5.3 proves the resulting regret bounds.

## 5.1 Algorithm Overview

Each round t, our algorithm run a Poisson base walk of Section 3 from some initial base. The exchange distribution used by algorithm is induced by a fractional base $x _ { t }$ , which is maintained and updated by our algorithm reach round. Given any current base A at the Poisson process, the balanced matrix $Q _ { A } ^ { x _ { t } }$ converts this single point into a distribution over feasible exchanges: its row constraints make the deleted element uniform on $A _ { i }$ , while its column constraints make the inserted element have marginal $x _ { t } / k$ . Thus the same $x _ { t }$ can be used at every state visited by the Poisson trajectory, and the state itself always remains a base.

The deficit term of the Poisson base walk is expressed through the signed exchange gain $G _ { f }$ from (3.4). To use this quantity in online learning over $P ( \mathcal { M } )$ , we express its expectation under a balanced exchange distribution as a linear function of the fractional decision x. We first separate $G _ { f }$ into insertion and deletion contributions. Fix a monotone submodular function $f ,$ its multilinear extension $F _ { ; }$ , a base $A ,$ , and a time $\tau \in [ 0 , 1 ]$ . For a feasible exchange $( i , j ) \in { \mathcal { E } } ( A )$ define the insertion marginal

$$
I _ { f } ( A , \tau ; i , j ) \triangleq \mathbb { E } _ { R \sim \tau ( A - i ) } [ f ( R + j ) - f ( R ) ] .\tag{5.1}
$$

Lemma 5.1. For every feasible exchange $( i , j ) \in { \mathcal { E } } ( A )$ 2

$$
0 \leq I _ { f } ( A , \tau ; i , j ) \leq 1 ,\tag{5.2}
$$

$$
I _ { f } ( A , \tau ; i , j ) = \partial _ { j } F ( \tau { \bf 1 } _ { A - i } ) ,\tag{5.3}
$$

$$
I _ { f } ( A , \tau ; i , j ) \geq \partial _ { j } F ( \tau { \bf 1 } _ { A } ) ,\tag{5.4}
$$

$$
G _ { f } ( A , \tau ; i , j ) = I _ { f } ( A , \tau ; i , j ) - \partial _ { i } F ( \tau { \bf 1 } _ { A } ) .\tag{5.5}
$$

Proof. Monotonicity gives $f ( R + j ) - f ( R ) \geq 0$ , and the range $f \in [ 0 , 1 ]$ gives the upper bound in (5.2). Identity (5.3) is the derivative formula (2.4) evaluated at $\tau { \bf 1 } _ { A - i }$

The vectors $\tau { \bf 1 } _ { A - i }$ and $\tau \mathbf { 1 } _ { A }$ difer only in coordinate i, with the former no larger than the latter. By diminishing multilinear marginals,

$$
I _ { f } ( A , \tau ; i , j ) = \partial _ { j } F ( \tau { \bf 1 } _ { A - i } ) \geq \partial _ { j } F ( \tau { \bf 1 } _ { A } ) ,
$$

which proves (5.4).

For $R \sim \tau ( A - i )$

$$
\begin{array} { r } { \mathbb { E } [ f ( R + i ) - f ( R ) ] = \partial _ { i } F ( \tau { \bf 1 } _ { A - i } ) . } \end{array}
$$

The derivative $\partial _ { i } F$ is independent of coordinate $i ,$ so this equals $\partial _ { i } F ( \tau { \bf 1 } _ { A } )$ . Subtracting this marginal from (5.1) proves (5.5). □

Fix $x \in P ( \mathcal { M } )$ with strictly positive coordinates and a balanced matrix $Q = Q _ { A } ^ { x }$ . Define the average insertion marginal and the corresponding insertion loss by

$$
\overline { { I } } _ { f , A , \tau , x } ( j ) \triangleq \frac { 1 } { x _ { j } } \sum _ { i \in A } Q ( i , j ) I _ { f } ( A , \tau ; i , j ) , \qquad \ell _ { f , A , \tau , x } ( j ) \triangleq 1 - \overline { { I } } _ { f , A , \tau , x } ( j ) .\tag{5.6}
$$

Lemma 5.2. For every $j \in \mathcal { U }$

$$
0 \le \ell _ { f , A , \tau , x } ( j ) \le 1 , \qquad \overline { { { I } } } _ { f , A , \tau , x } ( j ) \ge \partial _ { j } { \cal F } ( \tau { \bf 1 } _ { A } ) .\tag{5.7}
$$

Proof. By the column equality, the nonnegative weights $Q ( i , j ) / x _ { j }$ sum to one. Thus $\overline { { I } } ( j )$ is a convex combination of insertion marginals, each of which belongs to $[ 0 , 1 ]$ by (5.2). This proves the range of ℓ. Every insertion marginal in column $j$ is at least $\partial _ { j } F ( \tau { \bf 1 } _ { A } )$ by (5.4), so their convex combination has the same lower bound. □

The following theorem bound Poisson deficit with a linear term with the insertion loss being the coeficient.

Theorem 5.3. Let $q _ { A } ^ { x } \triangleq Q _ { A } ^ { x } / k$ be a balanced exchange distribution. For every comparator base $O _ { i }$

$$
\zeta _ { f , O } ( A , \tau ) - \mathbb { E } _ { ( i , j ) \sim q _ { A } ^ { x } } [ G _ { f } ( A , \tau ; i , j ) ] \leq \frac { 1 } { k } \left. x - \mathbf { 1 } _ { O } , \ell _ { f , A , \tau , x } \right. .\tag{5.8}
$$

Proof. To simplify notation, set $I ( i , j ) \triangleq I _ { f } ( A , \tau ; i , j )$ and $\overline { { I } } _ { j } \triangleq \overline { { I } } _ { f , A , \tau , x } ( j )$ . By (5.5),

$$
\begin{array} { l } { \mathbb { E } _ { ( i , j ) \sim q _ { A } ^ { \tau } } [ G _ { f } ( A , \tau ; i , j ) ] = \displaystyle \frac { 1 } { k } \sum _ { i \in A } \sum _ { j \in \mathcal { U } } Q ( i , j ) \big ( I ( i , j ) - \partial _ { i } F ( \tau \mathbf { 1 } _ { A } ) \big ) } \\ { = \displaystyle \frac { 1 } { k } \left( \sum _ { j \in \mathcal { U } } \sum _ { i \in A } Q ( i , j ) I ( i , j ) - \sum _ { i \in A } \partial _ { i } F ( \tau \mathbf { 1 } _ { A } ) \sum _ { j \in \mathcal { U } } Q ( i , j ) \right) } \\ { = \displaystyle \frac { 1 } { k } \left( \sum _ { j \in \mathcal { U } } x _ { j } \overline { { I } } _ { j } - \sum _ { i \in A } \partial _ { i } F ( \tau \mathbf { 1 } _ { A } ) \right) . } \end{array}\tag{5.9}
$$

The last equality uses both balance conditions: the column sum produces $x _ { j } \bar { I } _ { j }$ , and the row sum is exactly one. Subtracting (5.9) from (3.8) gives

$$
\begin{array} { r l } & { \zeta _ { f , O } ( A , \tau ) - \mathbb { E } [ G _ { f } ] = \displaystyle \frac { 1 } { k } \left( \sum _ { o \in O } \partial _ { o } F ( \tau { \mathbf { 1 } } _ { A } ) - \sum _ { j \in \mathcal { U } } x _ { j } \overline { { I } } _ { j } \right) } \\ & { \qquad \leq \displaystyle \frac { 1 } { k } \left( \sum _ { o \in O } \overline { { I } } _ { o } - \sum _ { j \in \mathcal { U } } x _ { j } \overline { { I } } _ { j } \right) } \\ & { \qquad = \displaystyle \frac { 1 } { k } \left. { \mathbf { 1 } } _ { O } - x , \overline { { I } } \right. . } \end{array}
$$

The inequality is Lemma 5.2. Since both x and ${ \bf 1 } _ { O }$ have coordinate sum $k$

$$
\left. \mathbf { 1 } _ { O } - x , \overline { { I } } \right. = \left. \mathbf { 1 } _ { O } - x , \mathbf { 1 } - \ell \right. = \left. x - \mathbf { 1 } _ { O } , \ell \right. .
$$

This proves (5.8).

Equation (5.8) therefore replaces the comparator-directed exchange rule by a linear loss over $P ( \mathcal { M } )$ . The bandit update estimates a coordinate of this loss indexed by the inserted element j. Since the estimate contains the importance weight $1 / x _ { t , j }$ , we need a uniform positive lower bound on every coordinate. Following the shrinkage technique of Flaxman et al. (2005), we obtain this bound by contracting the polytope toward a point whose coordinates are all positive.

For every $u \in \mathcal { U }$ , choose a base $B ^ { ( u ) }$ containing u, and define $\begin{array} { r } { x ^ { \circ } \triangleq \frac { 1 } { n } \sum _ { u \in \mathcal { U } } \mathbf { 1 } _ { B ^ { ( u ) } } } \end{array}$ . Then $x ^ { \circ } \in P ( { \mathcal { M } } )$ and $\textstyle x _ { u } ^ { \circ } \geq { \frac { 1 } { n } }$ for every $u \in \mathcal { U }$ . For $\delta \in ( 0 , 1 )$ , let

$$
K _ { \delta } \triangleq ( 1 - \delta ) P ( \mathcal { M } ) + \delta x ^ { \circ } = \{ ( 1 - \delta ) y + \delta x ^ { \circ } : y \in P ( \mathcal { M } ) \} .\tag{5.10}
$$

Every $x \in K _ { \delta }$ belongs to $P ( \mathcal { M } )$ and satisfies $x _ { u } \ge \delta / n$ for every $u \in \mathcal { U }$ . For a comparator base $O _ { i }$ , define its shrunken representative $u _ { O } ^ { \delta } \triangleq ( 1 - \delta ) \mathbf { 1 } _ { O } + \delta x ^ { \circ } \in K _ { \delta }$

The shrinkage has two roles. It guarantees $1 / x _ { t , j } \le n / \delta$ on every exploration round, and it supplies a feasible comparator $u _ { O } ^ { \delta }$ for mirror descent. The latter difers from the base vertex $\mathbf { 1 } _ { O }$ by only a δ-mixture, so the resulting error is handled as an explicit shrinkage term in the regret proof. We use the negative-entropy potential $\begin{array} { r } { \Phi ( x ) \triangleq \sum _ { u \in \mathcal { U } } ( x _ { u } \log x _ { u } - x _ { u } ) } \end{array}$ , whose Bregman divergence on the positive orthant is

$$
D _ { \Phi } ( x , y ) \triangleq \sum _ { u \in \mathcal { U } } \left( x _ { u } \log \frac { x _ { u } } { y _ { u } } - x _ { u } + y _ { u } \right) .\tag{5.11}
$$

Entropy is well matched to the one-sparse nonnegative estimates produced below: the mirror-descent stability term is the weighted second moment $\textstyle \sum _ { u } x _ { t , u } \widehat { L } _ { t , u } ^ { 2 }$ , for which the insertion marginal $x _ { t , j } / k$ cancels the inverse probability in the estimator. This cancellation is the reason the variance depends on $n / k$ , rather than on the number of feasible exchanges.

Each round instantiates the Poisson base walk in Algorithm 1 with the balanced exchange distribution $p _ { s } ( \cdot \mid A ) = Q _ { A } ^ { x _ { t } } / k .$ followed by either exploitation or a single exploration query. Let $\varepsilon , \gamma , \delta \in ( 0 , 1 )$ denote the Poisson starting time, exploration probability, and shrinkage parameter, and let $\eta > 0$ be the learning rate. The nonhomogeneous process has intensity $k / s$ on $[ \varepsilon , 1 ]$ and hence mean event count $\mu \triangleq k \log ( 1 / \varepsilon )$ . We cap the realized count at an integer $L .$ This does not alter the Poisson analysis; it only makes the per-round number of transportation flows worst-case polynomial. For the analysis, the probability space is extended on overflow rounds to include the corresponding uncapped Poisson trajectory, and the overflow probability is bounded separately.

Conditional on $M = m$ , the event times of the process with rate $k / s$ are the order statistics of m independent samples with density

$$
\varphi _ { \varepsilon } ( s ) \triangleq { \frac { 1 } { s \log ( 1 / \varepsilon ) } } , \qquad s \in [ \varepsilon , 1 ] .\tag{5.12}
$$

Equivalently, if $U \sim \mathrm { U n i f } [ 0 , 1 ]$ , then $\varepsilon ^ { 1 - U }$ has density (5.12).

Given $x _ { t } ,$ a nonoverflow simulation starts from a fixed base $A _ { 0 }$ and, at every event, samples from the balanced distribution associated with the current base and $x _ { t }$ . Here an event again means one Poisson arrival and its associated feasible base exchange. The simulated trajectory uses only the independence oracle and internal randomness, and makes no value query. With probability $1 - \gamma$ , the learner queries the terminal base. With probability $\gamma _ { ; }$ it selects one event uniformly and uses one feasible query to estimate the cumulative deficit. The complete procedure is given in Algorithm 2. We next describe the one-query estimator $\widehat { L } _ { t }$ used in its update.

Bandit estimator. To analyze an exploration round, condition on the selected pre-event base $A ,$ time $\tau ,$ , and exchange $( i , j )$ . The leave-one-out set $R \sim \tau ( A - i )$ makes both R and $R + j$ feasible. A fair coin chooses which of these two sets is queried, and the observation is transformed into $2 \big ( 1 - f _ { t } ( R + j ) \big )$ or $2 f _ { t } ( R )$ . The conditional mean of this statistic is $1 - I _ { f _ { t } } ( A , \tau ; i , j )$ , the insertion loss produced by the local linearization. The estimate is nonnegative and bounded by two, which gives the multiplicative entropy update a clean local-norm analysis.

To state the loss being estimated, let

$$
\ell _ { t , J } \triangleq \ell _ { f _ { t } , A _ { t , J - 1 } , \tau _ { t , J } , x _ { t } } \in [ 0 , 1 ] ^ { \mathcal { U } }\tag{5.13}
$$

be the local loss vector at event J. By the balanced-exchange linearization, the inner product of $\ell _ { t , J }$ with $x _ { t } - \mathbf { 1 } _ { O }$ controls the local deficit relative to every comparator base O. We aggregate these vectors with the discount $w ( s ) \triangleq s e ^ { - ( 1 - s ) }$ and define

$$
L _ { t } \triangleq \frac { \mathbf { 1 } \{ M _ { t } \leq L \} } { k } \sum _ { J = 1 } ^ { M _ { t } } w ( \tau _ { t , J } ) \ell _ { t , J } .\tag{5.14}
$$

On an exploration round with $1 \leq M _ { t } \leq L$ , we use the following estimator

$$
\widehat { L } _ { t } = \frac { M _ { t } } { \gamma } \frac { w ( \tau _ { t , J _ { t } } ) z _ { t } } { x _ { t , j _ { t , J _ { t } } } } e _ { j _ { t , J _ { t } } } ,\tag{5.15}
$$

for $L _ { t } ,$ and set $\widehat { L } _ { t } = 0$ when $M _ { t } > L$ . This is a useful bandit estimator in the precise senses needed by the regret analysis: it uses one feasible query, is conditionally unbiased for $L _ { t }$ , and has controlled weighted second moment. These properties are proved in Section 5.2.

Algorithm 2: Poisson Balanced-Exchange Bandit   
Input: Matroid M; start base $\overline { { A _ { 0 } ; } }$ point $x ^ { \circ } ;$ parameters $\varepsilon , \gamma , \delta , \eta , L .$   
1 Set $x _ { 1 } = x ^ { \circ }$ and $\mu = k \log ( 1 / \varepsilon )$ .;   
2 for $t = 1 , \dots , T$ do   
3 Sample $M _ { t } \sim$ Poisson $\mathfrak { i } ( \mu )$   
4 if $M _ { t } > L$ then   
5 Play $S _ { t } = A _ { 0 } ,$ observe $f _ { t } ( S _ { t } )$ , and set $\widehat { L } _ { t } = 0 .$   
6 else   
7 Sample and sort event times $\varepsilon < \tau _ { t , 1 } < \cdot \cdot \cdot < \tau _ { t , M _ { t } } \leq 1$ according to (5.12).;   
8 Set $A _ { t , 0 } = A _ { 0 } . ;$   
9 for $J = 1 , \ldots , M _ { t }$ do   
10 Compute a balanced exchange $Q _ { A _ { t , J - 1 } } ^ { x _ { t } } . ;$   
11 Draw $( i _ { t , J } , j _ { t , J } ) \sim Q _ { A _ { t , J - 1 } } ^ { x _ { t } } / k . ;$   
12 Set $A _ { t , J } = A _ { t , J - 1 } - i _ { t , J } + j _ { t , J } . ;$   
13 Draw $Z _ { t } \sim$ Bernoulli(γ).;   
14 if $Z _ { t } = 0$ then   
15 Play the final base $S _ { t } = A _ { t , M _ { t } }$ , observe $f _ { t } ( S _ { t } )$ , and set $\widehat { L } _ { t } = 0 .$   
16 else   
17 if $M _ { t } = 0$ then   
18 Play $S _ { t } = A _ { 0 } .$ , observe $f _ { t } ( S _ { t } )$ , and set $\widehat { L } _ { t } = 0 .$ ;   
19 else   
20 Draw $J _ { t } \sim \operatorname { U n i f } \{ 1 , \dots , M _ { t } \} .$ ;   
21 Put $A \triangleq A _ { t , J _ { t } - 1 } , \tau \triangleq \tau _ { t , J _ { t } }$ , and $( i , j ) \triangleq ( i _ { t , J _ { t } } , j _ { t , J _ { t } } ) .$   
22 Sample $R \sim \tau ( A - i )$ and $C _ { t } \sim$ Bernoulli $( 1 / 2 ) .$   
23 if $C _ { t } = 1$ then   
24 Play $S _ { t } = R + j $ observe $y _ { t } = f _ { t } ( S _ { t } )$ , and set $z _ { t } = 2 ( 1 - y _ { t } )$ .;   
25 else   
26 Play $S _ { t } = R ,$ observe $y _ { t } = f _ { t } ( S _ { t } )$ , and set $z _ { t } = 2 y _ { t } . ;$   
27 Set $\widehat { L } _ { t } = \frac { M _ { t } } { \mathfrak { a } } \frac { w ( \tau ) z _ { t } } { \mathfrak { a } } e _ { j } ,$ , where $e _ { j }$ is the jth coordinate vector and   
γ x<sub>t,j</sub>   
$w ( s ) \triangleq s e ^ { - ( 1 - s ) } . ;$   
28 Set $\widetilde { x } _ { t + 1 , u } = x _ { t , u } \exp ( - \eta \widehat { L } _ { t , u } )$ for every $u \in \mathcal { U } .$ ;   
29 Compute the entropy projection $x _ { t + 1 } \in \arg$ min $D _ { \Phi } ( x , \widetilde { x } _ { t + 1 } ) .$ .;   
x∈K<sub>δ</sub>

After the query, the learner performs the standard exponential weights update and returns to $K _ { \delta }$ by the exact entropy projection displayed in Algorithm 2. All vector divisions and logarithms in the algorithm are coordinatewise. A coordinate with a large estimated insertion loss is downweighted, while the projection restores the matroid base constraints and the uniform positivity condition. The resulting Bregman projection inequality is the projection property used by the online-learning analysis.

Computational complexity. We will set the event cap to be $L = O ( k \log ( 1 / \varepsilon ) + \log T )$ in the following sections, and each balanced exchange is computed by a polynomial-size transportation problem. The exact entropy projection is a separable strictly convex minimization problem over a shrunken matroid base polytope and is oracle-polynomial in the real arithmetic model by Suehiro et al. (2012). Thus the per-round computation is oracle-polynomial and no projection tolerance is needed.

## 5.2 Properties of the Bandit Estimator

This subsection establishes the estimator guarantees used in the regret analysis. The estimator uses one feasible value query, is conditionally unbiased for $L _ { t } .$ , and has controlled weighted second moment.

Proposition 5.4. Every query made by Algorithm 2 is an independent set, and each round makes exactly one value query.

Proof. On an overflow round the algorithm plays the base $A _ { 0 }$ . On a nonoverflow exploitation round it plays the final state of a sequence of feasible base exchanges, hence a base. On an exploration round, $R \subseteq A - i \subseteq A$ , so R is independent. Also $R + j \subseteq A - i + j .$ , and $A - i + j$ is a base because the sampled exchange lies in the support of a balanced matrix. Thus both possible queried sets are independent. Every branch plays one set and observes one scalar value. □

Let $\mathcal { H } _ { t }$ be the sigma-field generated by the interaction history through round $t - 1$ , the predictable state $x _ { t } ,$ , and $f _ { t } .$ , before the fresh round-t randomization. Thus $f _ { t }$ is fixed under conditioning on $\mathcal { H } _ { t }$ but remains unobserved by the learner.

Lemma 5.5. Fix a base A, a time τ , and a feasible exchange $( i , j )$ . Independently draw $R \sim \tau ( A - i )$ and $C \sim$ Bernoulli(1/2), and define

$$
Z \triangleq { \left\{ \begin{array} { l l } { 2 ( 1 - f ( R + j ) ) , } & { C = 1 , } \\ { 2 f ( R ) , } & { C = 0 . } \end{array} \right. }\tag{5.16}
$$

Then $0 \leq Z \leq 2 \ a n d \ \mathbb { E } [ Z \mid i , j , A , \tau ] = 1 - I _ { f } ( A , \tau ; i , j )$

Proof. The range follows from $f \in [ 0 , 1 ]$ . Averaging over the fair coin and then over R gives $\mathbb { E } [ Z \mid i , j , A , \tau ] = \mathbb { E } _ { R \sim \tau ( A - i ) } [ 1 - f ( R + j ) + f ( R ) ] = 1 - I _ { f } ( A , \tau ; i , j )$ □

Lemma 5.6. For every round t of Algorithm 2, it holds that $\mathbb { E } [ \widehat { L } _ { t } \mid \mathcal { H } _ { t } ] = \mathbb { E } [ L _ { t } \mid \mathcal { H } _ { t } ]$ , where $L _ { t }$ is defined in (5.14) and $\widehat { L } _ { t }$ is defined in (5.15).

Proof. Fix a realized nonoverflow pre-event base A and time τ before sampling the exchange, and let $Q \triangleq Q _ { A } ^ { x _ { t } }$ . Conditional on $( i , j )$ , Lemma 5.5 gives $\mathbb { E } [ z _ { t } \mid i , j , A , \tau ] = 1 - I _ { f _ { t } } ( A , \tau ; i , j )$ . The jth coordinate of the expected importance-weighted vector is $\begin{array} { r } { \frac { w ( \tau ) } { k } \big ( 1 - \frac { 1 } { x _ { t , j } } \sum _ { i \in A } Q ( i , j ) I _ { f _ { t } } ( A , \tau ; i , j ) \big ) } \end{array}$ where we used $\Sigma _ { i } Q ( i , j ) = x _ { t , j }$ . Hence $\mathbb { E } [ ( w ( \tau ) z _ { t } / x _ { t , j } ) e _ { j } \mid A , \tau ] = ( w ( \tau ) / k ) \ell _ { f _ { t } , A , \tau , x _ { t } } .$

Now condition on the nonoverflow trajectory. For ${ M _ { t } } ~ > ~ 0$ , uniform sampling of $J _ { t }$ and multiplication by M recover the sum over events, while $\mathbb { E } [ Z _ { t } / \gamma ] = 1$ corrects for exploration. Therefore $\begin{array} { r } { \mathbb { E } [ \widehat { L } _ { t } ] = ( 1 / k ) \sum _ { J = 1 } ^ { M _ { t } } w ( \tau _ { t , J } ) \ell _ { t , J } } \end{array}$ conditional on the trajectory. When $M _ { t } = 0$ or $M _ { t } > L$ both sides are zero. Taking the tower expectation proves the lemma. □

Lemma 5.7. For every round t, $\begin{array} { r } { \mathbb { E } [ \sum _ { u \in \mathcal { U } } x _ { t , u } \widehat { L } _ { t , u } ^ { 2 } \mid \mathcal { H } _ { t } ] \leq 4 n ( \mu ^ { 2 } + \mu ) / ( k \gamma ) } \end{array}$

Proof. The estimator is zero unless the round is a nonoverflow exploration round with at least one event. On such a round, one-sparsity gives $\begin{array} { r } { \sum _ { u \in \mathcal { U } } x _ { t , u } \widehat { L } _ { t , u } ^ { 2 } = ( Z _ { t } M _ { t } ^ { 2 } / \gamma ^ { 2 } ) w ( \tau ) ^ { 2 } z _ { t } ^ { 2 } / x _ { t , j } } \end{array}$ , where $Z _ { t }$ is the exploration indicator.

Fix the selected pre-event base A and time τ before sampling its exchange. Since $x _ { t } \in K _ { \delta }$ every coordinate $x _ { t , u }$ is positive. Moreover, $w ( \tau ) \leq 1$ for $\tau \in [ \varepsilon , 1 ]$ , and Lemma 5.5 gives $z _ { t } ^ { 2 } \leq 4$ . Expanding the conditional expectation over the entering element and using its marginal distribution from (4.5), we obtain

$$
\mathbb { E } \left[ \left. \frac { w ( \tau ) ^ { 2 } z _ { t } ^ { 2 } } { x _ { t , j } } \right| A , \tau \right] = \sum _ { u \in \mathcal { U } } \mathbb { P } ( j = u \mid A , \tau ) \mathbb { E } \left[ \left. \frac { w ( \tau ) ^ { 2 } z _ { t } ^ { 2 } } { x _ { t , u } } \right| A , \tau , j = u \right]
$$

$$
\leq \sum _ { u \in \mathcal { U } } \frac { x _ { t , u } } { k } \frac { 4 } { x _ { t , u } } = \frac { 4 n } { k } .
$$

Thus the sampling probability $x _ { t , u } / k$ cancels the remaining inverse weight $1 / x _ { t , u }$ for every possible entering element u. Since the bound is uniform over the realized A and τ, it also holds after conditioning on the preceding trajectory.

Conditional on $M _ { t } = m \in \{ 1 , \ldots , L \}$ , the uniform event index averages this bound over the m event states and times, while $\mathbb { E } [ Z _ { t } ] / \gamma ^ { 2 } = 1 / \gamma$ . Thus $\begin{array} { r } { \mathbb { E } \big [ \sum _ { u } x _ { t , u } \widehat { L } _ { t , u } ^ { 2 } \mid \mathcal { \bar { H } } _ { t } , M _ { t } = m \big ] \leq 4 n m ^ { 2 } / ( k \gamma ) } \end{array}$ Averaging over the Poisson count and using $\mathbb { E } [ M _ { t } ^ { 2 } ] = \mu ^ { 2 } + \mu$ proves the claim; the case $m = 0$ contributes zero. □

## 5.3 Regret Analysis

Now, we analyze the regret of the algorithm. We use the following standard negative-entropy Online Mirror Descent framework, see Orabona (2026, Chapter 6). We record the short specialization needed here because it applies pathwise to the adaptive loss sequence.

Lemma 5.8. Let $K \subset ( 0 , \infty ) ^ { n }$ be a convex compact set. Let $x _ { 1 } , \dotsc , x _ { T + 1 } \in K$ , let $g _ { 1 } , \dotsc , g _ { T } \in$ $[ 0 , \infty ) ^ { n }$ , and define $\widetilde { x } _ { t + 1 , u } \triangleq x _ { t , u } e ^ { - \eta g _ { t , u } }$ . Suppose that, for every $t = 1 , . . . , T , x _ { t + 1 }$ is the exact entropy projection

$$
x _ { t + 1 } \in \arg \operatorname* { m i n } _ { x \in K } D _ { \Phi } ( x , \widetilde { x } _ { t + 1 } ) .\tag{5.17}
$$

Then, for every comparator $u \in K$

$$
\sum _ { t = 1 } ^ { T } \langle x _ { t } - u , g _ { t } \rangle \leq \frac { D _ { \Phi } ( u , x _ { 1 } ) } { \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { n } x _ { t , j } g _ { t , j } ^ { 2 } .\tag{5.18}
$$

Proof. Fix a round t. The Bregman Pythagorean inequality for the exact projection in (5.17) and the negative-entropy dual update give

$$
\begin{array} { r l } { \displaystyle D _ { \Phi } ( u , x _ { t + 1 } ) \leq D _ { \Phi } ( u , \widetilde { x } _ { t + 1 } ) } \\ { \displaystyle } & { = \sum _ { j = 1 } ^ { n } \left( u _ { j } \log \frac { u _ { j } } { x _ { t , j } e ^ { - \eta g _ { t , j } } } - u _ { j } + x _ { t , j } e ^ { - \eta g _ { t , j } } \right) } \\ { \displaystyle } & { = \sum _ { j = 1 } ^ { n } \left( u _ { j } \log \frac { u _ { j } } { x _ { t , j } } - u _ { j } + x _ { t , j } \right) + \eta \sum _ { j = 1 } ^ { n } u _ { j } g _ { t , j } + \sum _ { j = 1 } ^ { n } x _ { t , j } \left( e ^ { - \eta g _ { t , j } } - 1 \right) } \\ { \displaystyle } & { = D _ { \Phi } ( u , x _ { t } ) + \eta \left. u , g _ { t } \right. + \sum _ { j = 1 } ^ { n } x _ { t , j } \left( e ^ { - \eta g _ { t , j } } - 1 \right) . } \end{array}
$$

The second equality substitutes $\boldsymbol { \widetilde { x } } _ { t + 1 , j } = \boldsymbol { x } _ { t , j } e ^ { - \eta g _ { t , j } }$ into (5.11). The third uses the coordinatewise identity

$$
\log \frac { u _ { j } } { x _ { t , j } e ^ { - \eta g _ { t , j } } } = \log \frac { u _ { j } } { x _ { t , j } } + \eta g _ { t , j } ,
$$

and adds and subtracts $x _ { t , j }$ , so that the first sum is exactly $D _ { \Phi } { \big ( } u , x _ { t } { \big ) }$ . Since $g _ { t } \geq 0$ and $e ^ { - a } \leq 1 - a + a ^ { 2 } / 2$ for $a \geq 0$ , it follows that

$$
\langle x _ { t } - u , g _ { t } \rangle \leq \frac { D _ { \Phi } ( u , x _ { t } ) - D _ { \Phi } ( u , x _ { t + 1 } ) } { \eta } + \frac { \eta } { 2 } \sum _ { j } x _ { t , j } g _ { t , j } ^ { 2 } .
$$

Summing over t telescopes the divergence terms. Dropping the final nonnegative divergence proves (5.18). □

Lemma 5.9. For every base O, it holds that $D _ { \Phi } ( u _ { O } ^ { \delta } , x ^ { \circ } ) \leq k \log n \leq k \log ( e n )$

Proof. Both $u _ { O } ^ { \delta }$ and $x ^ { \circ }$ belong to the base polytope, so their coordinate sums are k. The linear terms in (5.11) cancel:

$$
D _ { \Phi } ( u _ { O } ^ { \delta } , x ^ { \circ } ) = \sum _ { j } u _ { O , j } ^ { \delta } \log \frac { u _ { O , j } ^ { \delta } } { x _ { j } ^ { \circ } } .
$$

Every coordinate of a base-polytope point is at most one, while $x _ { j } ^ { \circ } \geq 1 / n$ . Hence

$$
\log \frac { u _ { O , j } ^ { \delta } } { x _ { j } ^ { \circ } } \leq \log \frac { 1 } { x _ { j } ^ { \circ } } \leq \log n
$$

whenever $u _ { O , j } ^ { \delta } > 0$ . Summing and using $\begin{array} { r } { \sum _ { j } u _ { O , j } ^ { \delta } = k } \end{array}$ proves the first inequality. The second is immediate. □

Proposition 5.10. Let $L _ { t }$ be the aggregate loss in (5.14). For every fixed base $O _ { i }$

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \Big \langle x _ { t } - u _ { O } ^ { \delta } , L _ { t } \Big \rangle \right] \leq \frac { k \log ( e n ) } { \eta } + \frac { 2 \eta n ( \mu ^ { 2 } + \mu ) T } { k \gamma } .\tag{5.19}
$$

Proof. Apply Lemma 5.8 pathwise with $K = K _ { \delta } , g _ { t } = { \widehat { L } } _ { t } , x _ { 1 } = x ^ { \circ }$ , and comparator $u = u _ { O } ^ { \delta }$ This gives

$$
\sum _ { t = 1 } ^ { T } \left. x _ { t } - u , \widehat L _ { t } \right. \leq \frac { D _ { \Phi } ( u , x ^ { \circ } ) } { \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \sum _ { j } x _ { t , j } \widehat L _ { t , j } ^ { 2 } .\tag{5.20}
$$

Because $x _ { t }$ and u are $\mathcal { H } _ { t } .$ -measurable, Lemma 5.6 implies

$$
\mathbb { E } \left. x _ { t } - u , \widehat { L } _ { t } \right. = \mathbb { E } \left. x _ { t } - u , L _ { t } \right. .
$$

Take expectations in (5.20), use Lemma 5.9 for the initial divergence, and use Lemma 5.7 for each quadratic term. The coeficient is

$$
\frac { \eta } { 2 } \cdot \frac { 4 n ( \mu ^ { 2 } + \mu ) } { k \gamma } = \frac { 2 \eta n ( \mu ^ { 2 } + \mu ) } { k \gamma } ,
$$

which proves (5.19) after summing over $T$ rounds.

The final regret analysis combines three inequalities: the Poisson theorem bounds approximation regret by event deficits, balanced exchanges bound those deficits by linear regret, and entropic mirror descent controls the resulting adaptive losses. On overflow rounds, we extend the probability space to include the uncapped Poisson trajectory and bound the discrepancy caused by the implementation’s event cap.

Theorem 5.11. Let $T \geq 2$ , set $\varepsilon \triangleq 1 / T$ and $\mu \triangleq$ k log T. If $n ( \mu ^ { 2 } + \mu ) \log ( e n ) < T .$ , run Algorithm $\mathcal { Q }$ with

$$
L \triangleq \lceil 2 e ( \mu + \log T ) \rceil , \qquad \delta \triangleq \operatorname* { m i n } \left\{ { \frac { 1 } { 2 } } , { \frac { 1 } { \mu T } } \right\} ,
$$

and

$$
\gamma \triangleq \left( \frac { n ( \mu ^ { 2 } + \mu ) \log ( e n ) } { T } \right) ^ { 1 / 3 } , \qquad \eta \triangleq \sqrt { \frac { k ^ { 2 } \gamma \log ( e n ) } { 2 n ( \mu ^ { 2 } + \mu ) T } } .
$$

Then

$$
\begin{array} { r l } & { R _ { 1 - 1 / e } ( T ) = O \bigg ( \Big [ n \big ( ( k \log T ) ^ { 2 } + k \log T \big ) \log ( e n ) \Big ] ^ { 1 / 3 } T ^ { 2 / 3 } + 1 \bigg ) } \\ & { \qquad = \widetilde O \Big ( n ^ { 1 / 3 } k ^ { 2 / 3 } T ^ { 2 / 3 } \Big ) . } \end{array}\tag{5.21}
$$

$I f n ( \mu ^ { 2 } + \mu ) \log ( e n ) \geq T$ , the same bound is achieved by playing any fixed base.

Proof. Let

$$
\alpha \triangleq 1 - 1 / e , \qquad \alpha _ { \varepsilon } \triangleq ( 1 - \varepsilon ) \alpha .
$$

Let

$$
O ^ { \star } \in \underset { O \in \mathcal { B } } { \arg \operatorname* { m a x } } \sum _ { t = 1 } ^ { T } f _ { t } ( O ) .\tag{5.22}
$$

Because the adversary is oblivious, $O ^ { \star }$ is fixed before the learner’s random choices.

Step 1: restore the uncapped Poisson process. At the beginning of round t, the algorithm samples the full Poisson count $M _ { t }$ . If $M _ { t } \le L$ , it also samples all event times and exchanges. If $M _ { t } > L$ , the implementation stops immediately. For analysis, extend the probability space on such a round by additionally sampling the event times and the balanced-exchange trajectory that the uncapped process would have generated using the same predictable point $x _ { t }$ . On a nonoverflow round this trajectory is identical to the one generated by the algorithm. Let $A _ { t } ^ { \mathrm { f i n } }$ be its final base, and let

$$
( A _ { t , J - 1 } , \tau _ { t , J } , e _ { t , J } ) \qquad ( J = 1 , \ldots , M _ { t } )
$$

denote its pre-event states, event times, and exchanges.

Condition on the history before round t. The function $f _ { t }$ and point $x _ { t }$ are fixed under this conditioning, and the extended trajectory is exactly Algorithm 1 with $p _ { s } ( \cdot \mid A ) = Q _ { A } ^ { x _ { t } } / k$ . Apply Theorem 3.5 with comparator $O ^ { \star }$ , and then sum over rounds. We obtain

$$
\alpha _ {  { \varepsilon } } \sum _ { t = 1 } ^ { T } f _ { t } ( O ^ { \star } ) -  { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( A _ { t } ^ { \mathrm { f n } } ) \right] \leq  { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { T } \sum _ { J = 1 } ^ { M _ { t } } D _ { t , J } \right] ,\tag{5.23}
$$

where

$$
D _ { t , J } \triangleq w ( \tau _ { t , J } ) \Bigl ( \zeta _ { f _ { t } , O ^ { \star } } ( A _ { t , J - 1 } , \tau _ { t , J } ) - G _ { f _ { t } } ( A _ { t , J - 1 } , \tau _ { t , J } ; e _ { t , J } ) \Bigr ) .\tag{5.24}
$$

Step 2: charge event deficits to linear regret. Fix an event of the extended trajectory and condition on its pre-event history. The current base, time, and learner point are fixed, and the exchange is drawn from the balanced distribution $q _ { A } ^ { x _ { t } }$ . By Theorem 5.3,

$$
\mathbb { E } [ D _ { t , J } | \mathrm { p r e - e v e n t h i s t o r y } ] \leq \frac { w ( \tau _ { t , J } ) } { k } \langle x _ { t } - \mathbf { 1 } _ { O ^ { \star } } , \ell _ { t , J } \rangle .\tag{5.25}
$$

Here $\ell _ { t , J }$ is also defined on overflow rounds through the extended trajectory, even though the implementation does not generate it.

On a nonoverflow round, use $x _ { t } - \mathbf { 1 } _ { O ^ { \star } } = \left( x _ { t } - u _ { O ^ { \star } } ^ { \delta } \right) + \delta ( x ^ { \circ } - \mathbf { 1 } _ { O ^ { \star } } )$ and the definition (5.14):

$$
\frac { 1 } { k } \sum _ { J = 1 } ^ { M _ { t } } w ( \tau _ { t , J } ) \left. x _ { t } - \mathbf { 1 } _ { O ^ { \star } } , \ell _ { t , J } \right. = \left. x _ { t } - u _ { O ^ { \star } } ^ { \delta } , L _ { t } \right. + \frac { \delta } { k } \sum _ { J = 1 } ^ { M _ { t } } w ( \tau _ { t , J } ) \left. x ^ { \delta } - \mathbf { 1 } _ { O ^ { \star } } , \ell _ { t , J } \right. .\tag{5.26}
$$

Because $\ell _ { t , J } \geq 0 , w \leq 1$ , and $\Sigma _ { j } x _ { j } ^ { \circ } = k$

$$
\langle x ^ { \circ } - \mathbf { 1 } _ { O ^ { \star } } , \ell _ { t , J } \rangle \leq \langle x ^ { \circ } , \ell _ { t , J } \rangle \leq k .
$$

Thus the second term in (5.26) is at most $\delta M _ { t }$

On an overflow round, we do not charge the event losses to online learning. Instead, for each event,

$$
\frac { w ( \tau ) } { k } \left. x _ { t } - \mathbf { 1 } _ { O ^ { \star } } , \ell \right. \leq \frac { 1 } { k } \left. x _ { t } , \ell \right. \leq \frac { 1 } { k } \sum _ { j } x _ { t , j } = 1 .\tag{5.27}
$$

The overflow moment has the exact Poisson representation

$$
\mathbb { E } [ M \mathbf { 1 } \{ M > L \} ] = \sum _ { m = L + 1 } ^ { \infty } m e ^ { - \mu } \frac { \mu ^ { m } } { m ! } = \mu \sum _ { r = L } ^ { \infty } e ^ { - \mu } \frac { \mu ^ { r } } { r ! } = \mu \mathbb { P } ( M \geq L ) ,
$$

where the middle equality changes variables to $r = m - 1$ . Combining (5.25)–(5.27), taking expectations, and using $\mathbb { E } [ M _ { t } \mathbf { 1 } \{ M _ { t } \leq L \} ] \leq \mu$ therefore gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { J = 1 } ^ { M _ { t } } D _ { t , J } \right] \leq \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left. x _ { t } - u _ { O ^ { \star } } ^ { \delta } , L _ { t } \right. \right] + \delta \mu T + \mu T \mathbb { P } ( M \geq L ) .\tag{5.28}
$$

Step 3: complete the comparison with the implemented rewards. Apply Proposition 5.10 with $O = O ^ { \star }$ to the first term on the right-hand side of (5.28). This controls the cumulative event deficits by the online linear-regret bound, together with the shrinkage and overflow terms already displayed there. It remains to replace the terminal reward $f _ { t } ( A _ { t } ^ { \mathrm { f i n } } )$ of the extended uncapped trajectory in (5.23) by the reward $f _ { t } ( S _ { t } )$ of the set actually played. On a nonoverflow exploitation round, $S _ { t } = A _ { t } ^ { \mathrm { f i n } }$ . On a nonoverflow exploration round, $f _ { t } ( S _ { t } ) \geq 0$ while $f _ { t } ( A _ { t } ^ { \mathrm { f i n } } ) \leq 1$ , so the discrepancy is at most one. On an overflow round, the algorithm plays $A _ { 0 }$ while $A _ { t } ^ { \mathrm { f i n } }$ exists only in the extended trajectory, and the same range bound again gives a discrepancy of at most one. These three cases yield, pathwise,

$$
f _ { t } ( S _ { t } ) \geq f _ { t } ( A _ { t } ^ { \mathrm { f i n } } ) - \mathbf { 1 } \{ M _ { t } > L \} - \mathbf { 1 } \{ M _ { t } \leq L \mathrm { a n d ~ r o u n d ~ } t \mathrm { ~ e x p l o r e s } \} .\tag{5.29}
$$

The second indicator has expectation at most $\gamma _ { ; }$ , since a round explores with probability γ. Taking expectations and summing gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( S _ { t } ) \right] \geq \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } f _ { t } ( A _ { t } ^ { \mathrm { f i n } } ) \right] - T \mathbb { P } ( M > L ) - \gamma T .\tag{5.30}
$$

Combining (5.23), (5.28), Proposition 5.10, and (5.30), and using $\mathbb { P } ( M > L ) \le \mathbb { P } ( M \ge L )$ ， yields

$$
R _ { \alpha _ { \varepsilon } } ( T ) \leq \left[ \gamma + \delta \mu + ( 1 + \mu ) \mathbb { P } ( M \geq L ) \right] T + \frac { k \log ( e n ) } { \eta } + \frac { 2 \eta n ( \mu ^ { 2 } + \mu ) T } { k \gamma } .\tag{5.31}
$$

It remains to substitute the parameters in the theorem statement. For every $\theta > 0$ , the exponential-moment bound gives

$$
\mathbb { P } ( M \geq L ) \leq \exp ( \mu ( e ^ { \theta } - 1 ) - \theta L ) .
$$

Since $L \geq 2 e \mu$ , choosing $e ^ { \theta } = L / \mu$ yields

$$
\mathbb { P } ( M \geq L ) \leq \left( { \frac { e \mu } { L } } \right) ^ { L } \leq 2 ^ { - L } .
$$

With $c \triangleq 2 e$ log $2 > 1$ and the prescribed value of $L ,$ it follows that

$$
( 1 + \mu ) T \mathbb { P } ( M \geq L ) \leq ( 1 + \mu ) e ^ { - c \mu } T ^ { 1 - c } = O ( 1 ) ,
$$

where the implicit constant is universal because $( 1 + \mu ) e ^ { - c \mu }$ is bounded on $[ 0 , \infty )$ and $T ^ { 1 - c } \leq 2 ^ { 1 - c }$ Moreover, $\delta \mu T \leq 1$ . Finally, the prescribed learning rate satisfies

$$
\frac { k \log ( e n ) } { \eta } + \frac { 2 \eta n ( \mu ^ { 2 } + \mu ) T } { k \gamma } = 2 \sqrt { \frac { 2 n ( \mu ^ { 2 } + \mu ) T \log ( e n ) } { \gamma } } = 2 \sqrt { 2 } \left[ n ( \mu ^ { 2 } + \mu ) \log ( e n ) \right] ^ { 1 / 3 } T ^ { 2 / 3 } ,
$$

whereas $\gamma T = \left[ n ( \mu ^ { 2 } + \mu ) \log ( e n ) \right] ^ { 1 / 3 } T ^ { 2 / 3 }$ . For every reward sequence,

$$
R _ { \alpha } ( T ) = R _ { \alpha _ { \varepsilon } } ( T ) + \varepsilon \alpha \operatorname* { m a x } _ { O \in { \cal { B } } } \sum _ { t = 1 } ^ { T } f _ { t } ( O )
$$

Since $\varepsilon = 1 / T$ and $\mu = k \log T$ , the conversion cost is at most one. Combining this inequality with (5.31) and the preceding estimates proves (5.21) in the nontrivial regime.

If $n ( \mu ^ { 2 } + \mu ) \log ( e n ) \geq T$ , then $\left[ n ( \mu ^ { 2 } + \mu ) \log ( e n ) \right] ^ { 1 / 3 } T ^ { 2 / 3 } \ge T$ . Since the comparator reward is at most T and the reward of a fixed base is nonnegative, a fixed-base strategy satisfies ${ \cal R } _ { \alpha } ( T ) \leq T$ , proving (5.21) in the remaining case. □

## Acknowledgement

Generative LLM was used as an interactive aid in the developing of the proofs. It also helps with the organization and drafting of the proofs, which are verified by the authors. The idea and technical framework are also proposed by the human authors.

## References

Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002. doi: 10.1137/ S0097539701398375.

Richard A. Brualdi. Comments on bases in dependence structures. Bulletin of the Australian Mathematical Society, 1(2):161–167, 1969.

Niv Buchbinder and Moran Feldman. Deterministic algorithm and faster algorithm for submodular maximization subject to a matroid constraint. In Proceedings of the 65th IEEE Annual Symposium on Foundations of Computer Science, pages 700–712, 2024.

Gruia Călinescu, Chandra Chekuri, Martin Pál, and Jan Vondrák. Maximizing a monotone submodular function subject to a matroid constraint. SIAM Journal on Computing, 40(6): 1740–1766, 2011.

Adrian Rivera Cardoso and Rachel Cummings. Diferentially private online submodular minimization. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1650–1658, 2019.

Chandra Chekuri, Jan Vondrák, and Rico Zenklusen. Dependent randomized rounding via exchange properties of combinatorial structures. In Proceedings of the 51st IEEE Annual Symposium on Foundations of Computer Science, pages 575–584, 2010.

Lin Chen, Andreas Krause, and Amin Karbasi. Interactive submodular bandit. In Advances in Neural Information Processing Systems 30, pages 141–152, 2017.

Yuval Filmus and Justin Ward. Monotone submodular maximization over a matroid via nonoblivious local search. SIAM Journal on Computing, 43(2):514–542, 2014.

Abraham D. Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. Online convex optimization in the bandit setting: Gradient descent without a gradient. In Proceedings of the Sixteenth Annual ACM–SIAM Symposium on Discrete Algorithms, pages 385–394, 2005.

Fares Fourati, Vaneet Aggarwal, Christopher John Quinn, and Mohamed-Slim Alouini. Randomized greedy learning for non-monotone stochastic submodular maximization under full-bandit feedback. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 7455–7471, 2023.

Fares Fourati, Christopher John Quinn, Mohamed-Slim Alouini, and Vaneet Aggarwal. Combinatorial stochastic-greedy bandit. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12052–12060, 2024.

Amit Ganz Rozenman, Ariel Kulik, Roy Schwartz, and Mohit Singh. A Poisson process for submodular maximization. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, pages 2152–2163, 2026.

Elad Hazan and Satyen Kale. Online submodular minimization. Journal of Machine Learning Research, 13:2903–2922, 2012.

Shinji Ito. Revisiting online submodular minimization: Gap-dependent regret bounds, best of both worlds and adversarial robustness. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9678–9694, 2022.

David Kempe, Jon Kleinberg, and Éva Tardos. Maximizing the spread of influence through a social network. In Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 137–146, 2003. doi: 10.1145/956750.956769.

Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-optimal sensor placements in Gaussian processes: Theory, eficient algorithms and empirical studies. Journal of Machine Learning Research, 9:235–284, 2008.

George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming, 14:265–294, 1978.

Rad Niazadeh, Negin Golrezaei, Joshua R. Wang, Fransisca Susan, and Ashwinkumar Badanidiyuru. Online learning via ofline greedy algorithms: Applications in market design and optimization. Management Science, 69(7):3797–3817, 2023.

Guanyu Nie, Mridul Agarwal, Abhishek Kumar Umrawal, Vaneet Aggarwal, and Christopher John Quinn. An explore-then-commit algorithm for submodular maximization under full-bandit feedback. In Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 1541–1551, 2022.

Guanyu Nie, Yididiya Y. Nadew, Yanhui Zhu, Vaneet Aggarwal, and Christopher John Quinn. A framework for adapting ofline algorithms to solve combinatorial multi-armed bandit problems with bandit feedback. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 26166–26198, 2023.

Guanyu Nie, Vaneet Aggarwal, and Christopher John Quinn. Stochastic k-submodular bandits with full bandit feedback. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, pages 2696–2698, 2025.

Francesco Orabona. Online learning: A modern introduction using convex optimization, 2026. arXiv:1912.13213, version 10.

Tareq Si Salem, Gözde Özcan, Iasonas Nikolaou, Evimaria Terzi, and Stratis Ioannidis. Online submodular maximization via online convex optimization. In Proceedings of the 38th AAAI Conference on Artificial Intelligence, volume 38, pages 15038–15046, 2024.

Matthew Streeter and Daniel Golovin. An online algorithm for maximizing submodular functions. In Advances in Neural Information Processing Systems 21, pages 1577–1584, 2008.

Matthew Streeter, Daniel Golovin, and Andreas Krause. Online learning of assignments. In Advances in Neural Information Processing Systems 22, pages 1794–1802, 2009.

Daiki Suehiro, Kohei Hatano, Shuji Kijima, Eiji Takimoto, and Kiyohito Nagano. Online prediction under submodular constraints. In Proceedings of the 23rd International Conference on Algorithmic Learning Theory, volume 7568 of Lecture Notes in Computer Science, pages 260–274. Springer, 2012. doi: 10.1007/978-3-642-34106-9\_22.

Artin Tajdini, Lalit Jain, and Kevin Jamieson. Nearly minimax optimal submodular maximization with bandit feedback. In Advances in Neural Information Processing Systems 37, 2024. doi: 10.52202/079017-3051.

Jan Vondrák. Optimal approximation for the submodular welfare problem in the value oracle model. In Proceedings of the 40th Annual ACM Symposium on Theory of Computing, pages 67–74, 2008.

Zongqi Wan, Jialin Zhang, Wei Chen, Xiaoming Sun, and Zhijie Zhang. Bandit multi-linear DR-submodular maximization and its applications on adversarial submodular bandits. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 35491–35524, 2023.

Yisong Yue and Carlos Guestrin. Linear submodular bandits and their application to diversified retrieval. In Advances in Neural Information Processing Systems 24, pages 2483–2491, 2011.

Mingrui Zhang, Lin Chen, Hamed Hassani, and Amin Karbasi. Online continuous submodular maximization: From full-information to bandit feedback. In Advances in Neural Information Processing Systems 32, pages 9210–9220, 2019.

## A Sampling the Nonhomogeneous Poisson Process

We record the count and conditional event-time distribution used in Algorithm 2.

Proposition A.1. Let N be a nonhomogeneous Poisson process on $[ \varepsilon , 1 ]$ with intensity $\lambda ( s ) \triangleq$ $k / s$ . Then

$$
M \triangleq N ( [ \varepsilon , 1 ] ) \sim \operatorname { P o i s s o n } ( \mu ) , \qquad \mu \triangleq k \log ( 1 / \varepsilon ) .\tag{A.1}
$$

Conditional on $M = m$ , the unordered event times are independent with density

$$
{ \frac { \lambda ( s ) } { \mu } } = { \frac { 1 } { s \log ( 1 / \varepsilon ) } } \qquad ( s \in [ \varepsilon , 1 ] ) ,\tag{A.2}
$$

and the ordered event times are their order statistics. If $U \sim \mathrm { U n i f } [ 0 , 1 ]$ , then $S \ { \stackrel { \triangle } { = } } \ \varepsilon ^ { 1 - U }$ has density (A.2).

Proof. The integrated intensity is

$$
\Lambda ( a , b ) \triangleq \int _ { a } ^ { b } { \frac { k } { s } } d s = k \log ( b / a ) .
$$

$\mathrm { B y }$ the defining increment law of a nonhomogeneous Poisson process, $N ( [ a , b ] )$ is Poisson with mean $\Lambda ( { \boldsymbol { a } } , { \boldsymbol { b } } )$ . Taking $a = \varepsilon$ and $b = 1$ gives (A.1).

The joint density of exactly m ordered events at $\varepsilon < t _ { 1 } < \cdots < t _ { m } \leq 1$ is

$$
e ^ { - \mu } \prod _ { r = 1 } ^ { m } \lambda ( t _ { r } ) .\tag{A.3}
$$

Indeed, partition the interval into small cells around the $t _ { r } \mathrm { ^ { * } s }$ and their complement. The probability of one event in each selected cell and none elsewhere is the product of the corresponding first-order Poisson probabilities; dividing by the cell widths and taking the limit gives (A.3). Integrating over the ordered simplex yields

$$
e ^ { - \mu } { \frac { 1 } { m ! } } \left( \int _ { \varepsilon } ^ { 1 } \lambda ( s ) d s \right) ^ { m } = e ^ { - \mu } { \frac { \mu ^ { m } } { m ! } } ,
$$

which is the probability that $M = m$ . Dividing (A.3) by this probability gives the conditional ordered density

$$
m ! \prod _ { r = 1 } ^ { m } { \frac { \lambda ( t _ { r } ) } { \mu } } .
$$

This is exactly the density of the order statistics of m independent samples from $\lambda / \mu ,$ proving (A.2).

Finally, for $s \in [ \varepsilon , 1 ]$ 2

$$
\begin{array} { c } { \displaystyle \mathbb { P } ( \varepsilon ^ { 1 - U } \leq s ) = \mathbb { P } \left( U \leq 1 - \displaystyle \frac { \log s } { \log \varepsilon } \right) } \\ { \displaystyle = \displaystyle \frac { \log ( s / \varepsilon ) } { \log ( 1 / \varepsilon ) } . } \end{array}
$$

Diferentiating gives $1 / ( s \log ( 1 / \varepsilon ) )$ .