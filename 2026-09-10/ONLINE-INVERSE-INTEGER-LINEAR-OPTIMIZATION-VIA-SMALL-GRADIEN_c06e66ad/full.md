# ONLINE INVERSE INTEGER LINEAR OPTIMIZATION VIA SMALL-GRADIENT SKIPPING: CONSTANT REGRET AND FINITE MISTAKES

AKIRA KITAOKA

Abstract. In online inverse linear optimization, the learner predicts a weight at each round, observes the optimal action of the agent, and updates its prediction. In the general setting, the gap of log T between the regret upper bound O(d log T) and the lower bound Ω(d) is unresolved (here T is the total number of rounds and d is the dimension). When the action set is M-convex, the regret is known to be bounded by O(d log d), but the method attaining it computes a center of gravity at every round. This paper therefore proposes Small-Gradient Skipping (SGS), a mechanism that skips the update at rounds without a mistake in the case where the correct action is uniformly separated from the other candidates, and applies it to online gradient descent, the online Newton step, and MetaGrad. The number of mistakes is then bounded, for all three, by a quantity independent of T; and for the online Newton step and for MetaGrad with SGS, the dimension dependence of the regret becomes O(d<sup>2</sup>) when the forward problem is an integer linear program, that is, the factor log T is removed. Moreover, when the action set is M-convex, the regret is bounded eficiently without computing a center of gravity.

## 1. Introduction

The problem of estimating, from observed actions, the criterion by which a decision maker chooses its actions has been studied as imitation learning and inverse reinforcement learning (Ng et al., 2000) and as inverse optimization (Ahuja and Orlin, 2001; Heuberger, 2004; Chan et al., 2023). The estimated weight can be interpreted as an objective function expressing the reason for the decision, and it has applications to the estimation of undisclosed objective functions in electricity markets (Birge et al., 2017; Liang and Dvorkin, 2023) and in healthcare (Chan et al., 2022). This paper treats the case where the objective function of the forward problem (the optimization problem solved by the agent) is linear, in the online setting where states arrive sequentially: at each round the learner predicts a weight, observes the optimal action of the agent, and updates its prediction. Methods from online learning are efective in this setting, and it is standard to measure the performance by the cumulative gap, measured by the true weight, between the action induced by the learner’s prediction and the correct action (the cumulative decision regret R<sup>est</sup>) (Bärmann et al., 2018; Besbes et al., 2021; Gollapudi et al., 2021; Sakaue et al., 2025b; Oki and Sakaue, 2026).

However, most of the known upper bounds grow with the total number of rounds <sub>T: O(</sub>√<sub>T) for online gradient descent (Bärmann et al., 2018), and even for methods</sub> attaining logarithmic regret the bound is O(d log T) (where d is the dimension of the weight) (Gollapudi et al., 2021; Sakaue et al., 2025b). Upper bounds independent of T do exist, but each has a limitation: that of Gollapudi et al. (2021, Theorem 4.2), exp(O(d log d)), assumes neither a margin nor a gap but is exponential in the dimension; the bound under a gap condition (Sakaue et al., 2025a) is proportional to the inverse square of the gap; and the bound O(d log d) under the M-convexity of the action set (Oki and Sakaue, 2026) requires computing a center of gravity at every round. Whether a guarantee independent of T and polynomial in the dimension can be obtained with light computation for general forward problems, including integer programs, was an open question.

Our contributions are the following.

• Proposal of small-gradient skipping under a uniform margin: we assume a uniform margin, namely that under the true weight the diference in objective value between the correct action and any other candidate action is at least some $\gamma > 0$ , uniformly over all states (Assumption 3.1). If the forward problem is an integer linear program, this margin is automatically positive, and its value can be bounded from below in terms of the combinatorial structure of the feasible set. Under this assumption we propose to incorporate into an online algorithm the mechanism—small-gradient skipping (SGS)—that updates neither the iterate nor the internal state at rounds where the proposal is correct, that is, at rounds where 0 can be chosen as a subgradient. The guarantees are stated in terms of the number of mistakes rather than the total number of rounds, and the margin bounds that number of mistakes finitely.

• Guarantees independent of T and explicit upper bounds by problem class: we apply SGS to online gradient descent (OGD), the online Newton step (ONS), and MetaGrad (van Erven and Koolen, 2016; van Erven et al., 2021), and show that the number of mistakes and the regret are both bounded by quantities independent of the total number of rounds T (Table 2). Furthermore, we bound the margin from below in terms of the combinatorial structure of the feasible set (Table 3) and substitute it into the upper bounds to obtain explicit upper bounds by problem class (Table 4). In particular, ONS and growing-grid SGS-MetaGrad, whose bounds depend on the margin only logarithmically, attain, when the forward problem is an ILP and the weight space is the probability simplex, $R _ { T } ^ { \mathrm { e s t } } = O ( d ^ { 2 } \Vert M \Vert _ { 2 } \log ( 2 \Vert M \Vert _ { 2 } ) )$ (where $\| M \| _ { 2 }$ is the norm of the vector M in Equation (6.3)), a bound polynomial in the dimension, with the light computation of $O ( d ^ { 2 } )$ plus one generalized projection per mistake round. This means that, under a uniform margin, the gap of log T between the upper bound O(d log T) and the lower bound $\Omega ( d )$ , raised as an open problem by Sakaue et al. (2025b), disappears from the upper bound, so that the gap no longer depends on T. A comparison with existing methods is summarized in Table 1.

## All detailed proofs are deferred to the appendices.

Organization. §2 describes related work. §3 gives the problem setting and §4 defines SGS. §5 gives the guarantees for SGS-OGD, ONS, and SGS-MetaGrad. §6 gives lower bounds on the margin by structure, and §7 gives explicit upper bounds by problem class.

Table 1. Comparison of the decision regret Equation (3.6) and the computational cost in the case where the forward problem is an integer linear program (ILP) and the weight space is the probability simplex $\Theta = \Delta ^ { d - 1 }$ Here $\| M \| _ { 2 }$ is the norm of the vector M of coordinatewise ranges Equation (6.3), K $( \leq T )$ is the number of mistakes of each method, $\tau _ { \mathrm { s o l v e } }$ is the time for one linear optimization that computes the proposal $\hat { x } ^ { t }$ , and $\tau _ { \mathrm { E - p r o j } }$ τ<sub>G-proj</sub> is the time for one Euclidean $/$ generalized projection onto Θ. The lower part of the table lists the guarantees obtained in this paper, with the explicit lower bounds on the uniform margin (§6) substituted in; these values do not depend on the total number of rounds T. Of the algorithms in that part, SGS-OGD (Algorithm 1) and growing-grid SGS-MetaGrad (Algorithm 6) are proposed in this paper, whereas ONS (Algorithm 3) is the existing method of Hazan et al. (2007) (applied to online inverse linear optimization by Sakaue et al., 2025b). They are the ILP / probability simplex entries of Table 4, and the details of the substitution are in Appendix N. $^ { \ S } _ { \therefore }$ a result under the gap condition $\Delta > 0$ rather than a uniform margin. The value rewrites Sakaue et al. (2025a, Theorem 5.2) in the notation of this paper; the derivation is in Appendix O. The dash “—” in the total computational cost indicates that it is not compared in this table.

<table><tr><td>Method</td><td> $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td>Total computational cost</td></tr><tr><td>OGD (Bärmann et al., 2018)</td><td> $O ( \| M \| _ { 2 } \sqrt { T } )$ </td><td> $O ( T ( \tau _ { \mathrm { s o l v e } } + \tau _ { \mathrm { E - p r o j } } + d ) )$ </td></tr><tr><td>Sakaue et al. (2025a)§ ONS, MetaGrad</td><td> $O ( \| M \| _ { \infty } ( \log d ) ^ { 3 / 2 } / \Delta ^ { 2 } )$ </td><td></td></tr><tr><td>(Sakaue et al., 2025b)</td><td> $O ( \| M \| _ { 2 } d \log { \frac { T } { d } } )$ </td><td> $O ( T ( \tau _ { \mathrm { s o l v e } } + d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } ) )$ </td></tr><tr><td>SGS-OGD</td><td> $O ( d 2 ^ { d } \| M \| _ { 2 } ^ { d + 1 } )$ </td><td> $O ( T \tau _ { \mathrm { s o l v e } } + K ( d + \tau _ { \mathrm { E - p r o j } } ) )$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td>ONS</td><td> $O ( d ^ { 2 } \| M \| _ { 2 } \bar { \log } ( 2 \| M \| _ { 2 } ) )$ </td><td> $O ( T \tau _ { \mathrm { s o l v e } } + K ( d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } ) )$ </td></tr><tr><td>Growing-grid SGS-MetaGrad</td><td> $O ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) )$ </td><td> $O ( T \tau _ { \mathrm { s o l v e } } + K ( d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } ) \log K )$ </td></tr></table>

## 2. Related work

Finitely many updates under a margin condition: the classical line. That a margin keeps the number of updates finite is a classical theme. It begins with the Perceptron convergence theorem for linearly separable data, continues with ALMA (Gentile, 2001), which approximates the maximum-margin classifier without being given the value of the margin explicitly, and reaches inverse optimization with Sun et al. (2023), who gives, by a Perceptron-type method, the skeleton that leads from separability through finitely many mistakes to exact recovery. These are, however, results for binary classification, and they do not apply directly to the suboptimality loss treated in this paper. The mechanism of not advancing the internal state in rounds without a mistake also has precedents: Gollapudi et al. (2021) skip the update at correct rounds in their reduction to a cutting-plane algorithm, and Besbes et al. (2021, 2025) use a threshold-type skip that leaves the ellipsoidal cone unchanged in periods where the decision is nearly optimal. What this paper does anew is to formulate this mechanism for first- and second-order online convex optimization methods, whose internal state would otherwise advance even in rounds without a mistake, and to derive from it, under a uniform margin, guarantees independent of the total number of rounds together with explicit constants by problem class.

Finite regret in online inverse optimization. This paper is not the first to bound the regret in online inverse linear optimization by a constant independent of the total number of rounds $T \colon$ there are the bound under a gap condition (Sakaue et al., 2025a), the bound under the M-convexity of the action set (Oki and Sakaue, 2026), and the bound of Gollapudi et al. (2021, Theorem 4.2), which assumes neither a margin nor a gap. The diference from the existing work is twofold. First, neither our algorithms nor their T-independent guarantees (Table 2) require any structure beyond the uniform margin, and each mistake round costs only $O ( d ^ { 2 } )$ plus one generalized projection onto Θ. Second, we bound the uniform margin explicitly from below in terms of the combinatorial structure of the feasible set (§6) and reduce it to explicit upper bounds by problem class $( \ S 7 )$

For the classical line, for a detailed comparison with each of these results, for a precedent of the uniform margin in ofline inverse optimization, and for how the generality of the weight space Θ difers from that in the existing work, see Appendix A.

## 3. Problem setting

We consider an online learning setting with two players, the learner and the agent.<sup>1</sup> Let d be a positive integer and let $\mathbb { R } ^ { d }$ be the space on which the forward optimization is defined. We call a nonempty set $s$ the set of states, and for each state $s \in S$ we write $X ( s ) \subseteq \mathbb { R } ^ { d }$ for the set of feasible actions. For a weight $\theta \in \mathbb { R } ^ { d }$ and a state $s \in S$ , we write the forward problem (a linear optimization) and its optimal solution as

$$
x ^ { * } ( \theta , s ) : \in \arg \operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x \rangle\tag{3.1}
$$

(the attainment of the maximum follows from the compactness of $X ( s )$ assumed in Assumption $3 . 1 ( 2 ) )$ . The agent has an unknown objective vector $\theta ^ { * } \in \mathbb { R } ^ { d }$ , and for $t = 1 , \dots , T$ , when a state $s ^ { t } \in S$ is given, it chooses $x ^ { t } = x ^ { * } ( \theta ^ { * } , s ^ { t } ) \in X ( s ^ { t } )$ as its action. From the observations $\{ ( s ^ { t } , x ^ { t } ) \} _ { t = 1 } ^ { T }$ we want to find a weight θ satisfying $x ^ { \ast } ( \theta ^ { \ast } , s ) \in \arg \operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x \rangle$ at each state $s ^ { t }$ (the inverse linear optimization problem).

Note that the set $X ( s )$ is not necessarily convex. If $X ( s )$ is a polyhedron, then the solution returned by any solver for linear programming (LP) can serve as an oracle for $x ^ { * } ( \theta , s )$ . Also when $X ( s )$ is defined by integer linear constraints, an optimal solution can be obtained with an empirically eficient solver such as Gurobi.

The learner predicts $\theta ^ { * }$ sequentially for $t = 1 , \dots , T$ . Let $\Theta \subseteq \mathbb { R } ^ { d }$ be the set of linear objective vectors from which the learner chooses its predictions (the conditions imposed on $\Theta$ and $\theta ^ { * }$ are collected in Assumption 3.1). Below, $\| \cdot \|$ denotes the $\ell _ { 2 }$ norm and log the natural logarithm (we write, for instance, log only when the base is made explicit). We set the diameter of the weight space and the constant expressing the spread of the actions to be

$$
D : = \mathrm { d i a m } ( \Theta ) , \qquad L : = \operatorname* { s u p } _ { s \in { \mathcal { S } } } \operatorname* { s u p } _ { x \in X ( s ) } \| x - x ^ { * } ( \theta ^ { * } , s ) \|\tag{3.2}
$$

respectively. For $t = 1 , \dots , T$ , the learner outputs a prediction $\hat { \theta } ^ { t } \in \Theta$ of $\theta ^ { * }$ based on the past observations $\{ ( s ^ { t ^ { \prime } } , x ^ { t ^ { \prime } } ) \} _ { t ^ { \prime } = 1 } ^ { t - 1 }$ , and receives $( s ^ { t } , x ^ { t } )$ as feedback from the agent. Let $Y ( s )$ be the set of extreme points of the convex set Conv $X ( s )$ (since, for a compact set $X ( s )$ , the extreme points of Conv $X ( s )$ belong to $X ( s )$ , we have $Y ( s ) \subseteq X ( s ) )$ ). The proposal $\hat { x } ^ { t }$ induced by the learner’s t-th prediction $\hat { \theta } ^ { t }$ is defined as an extreme optimal solution

$$
\hat { x } ^ { t } \in \underset { x \in Y ( s ^ { t } ) } { \arg \operatorname* { m a x } } \langle \hat { \theta } ^ { t } , x \rangle .\tag{3.3}
$$

Since the maximum of a linear function on Conv $X ( s ^ { t } )$ is attained at an extreme point, the optimal value of Equation (3.3) coincides with the maximum on $X ( s ^ { t } )$ (the value of Equation (3.1)). Moreover, the simplex method returns an extreme point of the feasible polyhedron; for an ILP, returning an extreme optimal solution when several optimal solutions exist is imposed as a requirement on the oracle Equation (3.3).<sup>2</sup>

In inverse linear optimization, the suboptimality loss (Mohajerin Esfahani et al., 2018) is a useful criterion. For a weight θ and a state s, the suboptimality loss is defined, using Equation (3.1), by

$$
\ell _ { \mathrm { s u b } } ( \theta , s ) : = \langle \theta , x ^ { * } ( \theta , s ) - x ^ { * } ( \theta ^ { * } , s ) \rangle = \operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x - x ^ { * } ( \theta ^ { * } , s ) \rangle \geq 0 .\tag{3.4}
$$

If the suboptimality loss is 0, then $x ^ { \ast } ( \theta ^ { \ast } , s ) \in$ arg $\operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x \rangle$ holds, which means that the inverse optimization problem is solved at that state.

Performance criteria. This paper measures the quality of the learner’s sequence of predictions $\hat { \theta } ^ { 1 } , \dots , \hat { \theta } ^ { T }$ by the following three quantities. The first is the cumulative suboptimality regret

$$
R _ { T } ^ { \mathrm { s u b } } : = \sum _ { t = 1 } ^ { T } \ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t } , s ^ { t } ) = \sum _ { t = 1 } ^ { T } \langle \hat { \theta } ^ { t } , \hat { x } ^ { t } - x ^ { t } \rangle ,\tag{3.5}
$$

the cumulative suboptimality loss Equation (3.4), which expresses how poorly the agent’s action $x ^ { t }$ is explained from the viewpoint of the learner’s weight $\hat { \theta } ^ { t }$ . The second is the cumulative decision regret

$$
R _ { T } ^ { \mathrm { e s t } } : = \sum _ { t = 1 } ^ { T } \langle \theta ^ { * } , x ^ { t } - \hat { x } ^ { t } \rangle ,\tag{3.6}
$$

which expresses how suboptimal the learner’s proposal $\hat { x } ^ { t }$ is from the viewpoint of the true weight $\theta ^ { * }$ . It is this $R _ { T } ^ { \mathrm { { e s t } } }$ that Besbes et al. (2021); Gollapudi et al. (2021); Besbes et al. (2025); Sakaue et al. (2025b); Oki and Sakaue (2026) simply call the regret. Which criterion is bounded in which reference is summarized in Table 6. The third is the sum of the two,

$$
\widetilde { R } _ { T } : = R _ { T } ^ { \mathrm { s u b } } + R _ { T } ^ { \mathrm { e s t } } = \sum _ { t = 1 } ^ { T } \langle \hat { \theta } ^ { t } - \theta ^ { * } , \hat { x } ^ { t } - x ^ { t } \rangle ,\tag{3.7}
$$

which amounts to the quantity $\smash { \widetilde { R } _ { T } ^ { c ^ { * } } }$ introduced by Sakaue et al. (2025b). Since $x ^ { t }$ and $\hat { x } ^ { t }$ are optimal for $\theta ^ { * }$ and $\hat { \theta } ^ { t }$ respectively, we have $R _ { T } ^ { \mathrm { s u b } } , R _ { T } ^ { \mathrm { e s t } } \geq 0$ , and hence

$$
\operatorname* { m a x } ( R _ { T } ^ { \mathrm { s u b } } , ~ R _ { T } ^ { \mathrm { e s t } } ) \leq \widetilde { R } _ { T }\tag{3.8}
$$

holds. An upper bound on one of the components does not give an upper bound on the other, but each theorem of this paper bounds the sum $\bar { R } _ { T }$ itself independently of $T .$ , so that $R _ { T } ^ { \mathrm { { s u b } } }$ and $R _ { T } ^ { \mathrm { { e s t } } }$ are bounded simultaneously.

To solve online inverse linear optimization, we introduce the following assumption.

Assumption 3.1. $( \mathbf { 1 } ) \colon \Theta \subset \mathbb { R } ^ { d }$ is a nonempty bounded closed convex set with $D > 0$ (boundedness gives $D < \infty )$

(2): For each state $s \in S$ , the set $X ( s ) \subset \mathbb { R } ^ { d }$ is nonempty and compact, and the set of extreme points $Y ( s )$ (defined just before Equation (3.3)) is a finite set.

(3): $\theta ^ { \ast } \in \Theta$ , and for each state $s \in S$ the set arg $\mathrm { m a x } _ { x \in X ( s ) } \langle \theta ^ { * } , x \rangle$ is a singleton. We write its unique element as $x ^ { * } ( \theta ^ { * } , s )$ (since the unique maximizer of a linear function is an extreme point of Conv $X ( s )$ , we have $x ^ { \ast } ( \theta ^ { \ast } , s ) \in Y ( s ) )$

(4): (Uniform margin) There exist $\bar { \theta } \in \Theta$ and $\gamma > 0$ such that, for every $s \in S$ and every $x \in Y ( s ) \setminus \{ x ^ { * } ( \theta ^ { * } , s ) \}$

$$
\langle \bar { \theta } , x ^ { \ast } ( \theta ^ { \ast } , s ) - x \rangle \geq \gamma .\tag{3.9}
$$

(5): The constant L in Equation (3.2) satisfies $0 < L < \infty$

Remark 3.2. For the constant $\gamma$ of the uniform margin in Assumption $3 . 1 ( 4 )$ ), a concrete lower bound can be obtained, for instance, when the constraint set $X ( s )$ is given by integer linear constraints. Concrete examples for each structure of the feasible set are given in $\ S 6 .$ . The lower bounds by structure are summarized in Table 3.

## 4. Proposed method: small-gradient skipping (SGS)

Definition 4.1 (Small-gradient skipping). For an online learning method with an iterate $\hat { \theta } ^ { t }$ and an internal state (a learning-rate index, an information matrix, the weights of experts, and so on), small-gradient skipping (Small-Gradient Skipping; SGS) refers to the following mechanism: at a round where no mistake occurred $( \hat { x } ^ { t } = x ^ { t } )$ , neither the iterate nor the internal state is updated at all $( \hat { \theta } ^ { t + 1 } = \hat { \theta } ^ { t } )$ and the index k of the internal state is not advanced either. Only at mistake rounds is the update performed with the subgradient $g = \hat { x } ^ { t } - x ^ { t }$ , and k advanced by one.

The name comes from the following observation: at a round where no mistake occurs, the value of the loss $\ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t } , s ^ { t } ) = \langle \hat { \theta } ^ { t } , \hat { x } ^ { t } - x ^ { t } \rangle = 0$ is the minimum value of $\ell _ { \mathrm { s u b } } ( \cdot , s ^ { t } ) \geq 0$ , and then $0 \in \partial _ { \theta } \ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t } , s ^ { t } )$ , that is, the learner can choose 0 (a suficiently small gradient) as a subgradient. SGS skips exactly these rounds. The oracle-based subgradient $\boldsymbol { g } ^ { t } = \hat { \boldsymbol { x } } ^ { t } - \boldsymbol { x } ^ { t }$ used in this paper is, as a vector, $g ^ { t } = 0$ at rounds where no mistake occurs.

Remark 4.2 (When SGS changes the algorithm and when it changes only the analysis). For an online learning method whose update rule depends only on $g ^ { t }$ , it follows that, even without incorporating SGS, the weight of the objective function does not move at rounds with $g ^ { t } = 0$ . That is, incorporating SGS does not change the algorithm. Examples are ONS and fixed-grid MetaGrad (Algorithm 5). On the other hand, for an online learning method whose update rule depends, in addition to $g ^ { t }$ , on the round index, incorporating SGS does change the algorithm. Examples are first-order methods whose learning rate decays with the round number (the $\dot { \alpha } k ^ { - 1 / 2 }$ of SGS-OGD difers from $\alpha t ^ { - 1 / 2 } )$ and growing-grid SGS-MetaGrad (Algorithm 6), whose learning-rate grid is refined according to the number of mistakes instead of the round index (§5).

We call a round t with $\hat { x } ^ { t } \neq x ^ { t }$ a mistake, and write K for the total number of such rounds. By Definition 4.1, the iterate is updated only at mistake rounds.

In this paper we treat the following three instances of SGS: SGS-OGD (Algorithm 1), which incorporates SGS into projected online gradient descent; ONS (Algorithm 3), a second-order method; and SGS-MetaGrad (Algorithm 5; including Algorithm 6, which grows the learning-rate grid with the number of mistakes), a universal method. The pseudocode and all the accompanying remarks are collected in Appendix B, and the guarantees are given in $\ S 5$

## 5. Main results: finitely many mistakes and regret independent of T

SGS-OGD (a first-order method). As the basic form of a first-order method, we consider SGS-OGD (Algorithm 1), which incorporates SGS into projected online gradient descent: only at mistake rounds does it perform a subgradient step with step size $\alpha k ^ { - 1 / 2 }$ (where k is the index counting the mistakes) followed by a Euclidean projection. Its only parameters are L and D.

ONS (a second-order method). To improve the dependence of SGS-OGD on γ, we use the second-order method ONS (cf. Hazan et al., 2007) (Algorithm 3). Even when SGS is incorporated into ONS, it holds automatically that neither the iterate nor the information matrix moves at rounds without a mistake, so the SGS version generates the same sequence of iterates as plain ONS (Remarks B.1 and 4.2). This paper therefore incorporates the SGS viewpoint of counting only mistake rounds into the analysis of ONS, and shows that under a margin the guarantee of ONS improves to one independent of T. Its only parameters are $L$ and $D .$

SGS-MetaGrad. Even if ONS is replaced by MetaGrad (van Erven and Koolen, 2016; van Erven et al., 2021) (Algorithm 5), the update at rounds without a mistake is automatically the identity, so SGS works on the side of the analysis and a guarantee of the same order is obtained (Remark $_ { \mathrm { B . 4 ; } }$ the statement and the proof of the fixed-grid version are in $\operatorname { A p p e n d i x }$ E). However, since MetaGrad constructs its learning-rate grid before execution, an upper bound $\bar { K }$ on the number of mistakes is needed to determine its size, and the factor $c _ { 0 } ( \bar { K } ) = 2 \log ( \textstyle { \frac { 1 } { 2 } } \log _ { 2 } { \bar { K } } + 3 )$ remains in the guarantee. Since in general only $\bar { K } = T$ can be taken as an a priori upper bound, we get $c _ { 0 } ( \bar { K } ) = O ( \log \log T )$ , and strict independence from the total number of rounds fails to that extent. This $\bar { K }$ originates solely from fixing the grid before execution, so if we do not fix the grid but keep adding smaller learning rates as the number of mistakes k progresses, the input K<sup>¯</sup> itself becomes unnecessary, $c _ { 0 }$ is replaced by $c _ { 0 } ( K )$ with the realized number of mistakes $K .$ , and the guarantee becomes completely independent of $T ,$ This growing-grid SGS-MetaGrad (Algorithm 6) is an algorithm of this paper that difers from ordinary MetaGrad in that the grid is refined according to the number of mistakes (Remark B.5).

Main results. The guarantees of the three methods are summarized below.

Theorem 5.1 (Summary of the main results). Under Assumption 3.1, if SGS-OGD (with $\alpha = D / ( L \sqrt { 2 } ) \}$ ), ONS, and growing-grid SGS-MetaGrad are run on an arbitrary (possibly adaptive) sequence of states $\{ s ^ { t } \} _ { t = 1 } ^ { T }$ , then the number of mistakes K, the cumulative suboptimality regret $R _ { T } ^ { \mathrm { { s u b } } }$ , and the sum $\widetilde { R } _ { T }$ satisfy the upper bounds in Table 2. All of them hold deterministically for every T, and the right-hand sides do not depend on the total number of rounds T. Moreover, by Equation (3.8), the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ has the same upper bound as the regret of the sum.

The complete statements with explicit constants, together with their proofs, are Theorem C.2 (Appendix C) for SGS-OGD, Theorem D.6 (Appendix D) for ONS, and Theorem F.3 (Appendix F) for growing-grid SGS-MetaGrad.

Table 2. The guarantees of the proposed methods (Theorem 5.1). None of the right-hand sides depends on the total number of rounds T. The decision regret has the same upper bound as the regret of the sum (Equation (3.8)). For the explicit form for growing-grid SGS-MetaGrad see Theorem F.3 (the coeficient of the logarithmic term becomes 52 against the 2 of ONS, and an additional term log log d enters).
<table><tr><td></td><td>SGS-OGD (Theorem C.2) (Theorem D.6)</td><td>ONS</td><td>Growing-grid SGS-MetaGrad (Theorem F.3)</td></tr><tr><td> $K$ </td><td> $\scriptstyle { \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma ^ { 2 } } }$ </td><td> $\begin{array} { r l } { d + \frac { 2 L D } { \gamma } ( 1 + d \log \operatorname* { m a x } ( \frac { 2 L D } { \gamma } , 1 ) ) } & { { } O ( \frac { d L D } { \gamma } \log \operatorname* { m a x } ( \frac { 2 L D } { \gamma } , 2 ) ) } \end{array}$ </td><td></td></tr><tr><td> $R _ { T } ^ { \mathrm { { s u b } } }$ </td><td> $\frac { L ^ { 2 } D ^ { 2 } } { 2 \gamma }$ </td><td> $\begin{array} { r } { L D ( 1 + d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 1 ) ) } \end{array}$ </td><td> $O ( L D d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 1 ) )$ </td></tr><tr><td> $\widetilde { R } _ { T }$ </td><td> $\frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma }$ </td><td> $\begin{array} { r } { L D ( 1 + 2 d \log ( 2 + \frac { 2 L D } { \gamma } ) ) } \end{array}$ </td><td> $O ( L D d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 2 ) )$ </td></tr></table>

Comparison of the methods. Compared with SGS-OGD, ONS reduces the dependence on the margin $\gamma$ from $\gamma ^ { - 2 } ~ \mathrm { t o } ~ \gamma ^ { - 1 }$ in the number of mistakes, and from $\gamma ^ { - 1 }$ to log $\gamma ^ { - 1 }$ in the cumulative suboptimality regret. Since lo $\underline { { { \bf y } } } \gamma ^ { - 1 } \le \gamma ^ { - 1 }$ holds for every $\gamma > 0$ , and moreover the margin is often small, as we shall see in §6 (for a general ILP the lower bound on $\gamma$ can be exponentially small in the dimension), this replacement is a substantial improvement. The price is that ONS incurs a linear dependence on the dimension $d ,$ and the per-round computational cost also increases from $O ( d + \tau _ { \mathrm { E - p r o j } } )$ to $O ( d ^ { 2 } + \tau _ { \mathrm { { G - p r o j } } } )$ (Table 5). Therefore, on problems with a small margin $( \gamma \lesssim L D / ( d \log ( L D / \gamma ) ) )$ ONS is superior, whereas on problems with a large margin the bound $2 L ^ { 2 } D ^ { 2 } / \gamma ^ { 2 } = O ( 1 )$ of SGS-OGD is superior. Growing-grid SGS-MetaGrad attains a guarantee of the same order as ONS at the price of worse constants, and in addition the Lipschitz-adaptive version does not even require knowledge of $L$ (Remark B.6). Moreover, compared with the guarantees of Sakaue et al. (2025b) for ONS and MetaGrad, under a margin the log T in the cumulative suboptimality regret is replaced by log $\operatorname* { m a x } ( L D / \gamma , 1 )$ and an upper bound on the number of mistakes independent of $T$ is added: it is the degree of separation of the problem, not the total number of rounds, that determines the logarithmic term.

## 6. Integer programming and lower bounds on the uniform margin

The upper bounds of Theorems C.2, D.6, E.1 and F.3 are given in closed form in the margin γ. In this section we quantify γ from below according to the combinatorial structure of the forward problem of data-driven inverse optimization (DDIOP), and by substituting the result into the upper bounds we make the number of mistakes and the cumulative regret explicit by problem class. The main target is the general integer linear program (ILP), where an explicit finite upper bound is obtained unconditionally from an explicit lower bound on $\gamma$ (which is exponentially small in the dimension, but positive).<sup>3</sup> Under a discrete convex structure such as M-convexity, the lower bound improves to a polynomial and the upper bounds become polynomial in the dimension.

6.1. The largest attainable margin. In this section we assume $( 1 ) , ( 2 )$ and (3) of Assumption 3.1 and ask how large the margin $\gamma$ in (4) can be taken. Integrality is imposed only from §6.2 on, at the stage where the lower bound is evaluated from the combinatorial structure. We define the largest attainable margin and the constant expressing the spread of the actions by

$$
\gamma _ { \mathrm { s u b } } : = \operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { i n f } _ { s \in \mathcal { S } } \operatorname* { m i n } _ { \substack { x \in Y ( s ) \backslash \left\{ x ^ { * } ( \theta ^ { * } , s ) \right\} } } \langle \theta , x ^ { * } ( \theta ^ { * } , s ) - x \rangle ,\tag{6.1}
$$

$$
L _ { \mathrm { s u b } } : = \operatorname* { s u p } _ { s \in { \mathcal { S } } } \operatorname* { s u p } _ { x \in X ( s ) } \| x - x ^ { * } ( \theta ^ { * } , s ) \|\tag{6.2}
$$

respectively (states s with $Y ( s ) \setminus \{ x ^ { * } ( \theta ^ { * } , s ) \} = \emptyset$ are excluded from $\operatorname { i n f } _ { s } )$ . The inner min is a minimum over a finite set by Assumption 3.1(2), and hence is attained. The quantity $\gamma _ { \mathrm { s u b } }$ is the value, at the weight that makes it largest, of “the diference in objective value between the correct action and the other candidate actions”.

Lemma 6.1 (Attainment of $\gamma _ { \mathrm { s u b } }$ and validity of the margin). Assume (1), (2) and (3) of Assumption 3.1 and $0 < L _ { \mathrm { s u b } } < \infty$ . Then the max<sub>θ∈Θ</sub> in Equation (6.1) is attained. Furthermore, $\mathrm { i f \ \gamma _ { \mathrm { s u b } } > 0 }$ , then Assumption 3.1 holds with $\gamma = \gamma _ { \mathrm { s u b } }$ for a maximizer $\bar { \theta } \in \Theta$ (and, since the definitions Equation (3.2) and Equation (6.2) are identical, $L = L _ { \mathrm { s u b } } )$

See Appendix H for the proof. All the lower-bound theorems below are for $\gamma _ { \mathrm { s u b } }$ and they can be substituted into the upper bounds through Lemma 6.1.

Remark 6.2 (Generalization to a general feature map). In data-driven inverse optimization, the forward problem is often written as max $_ { \cdot u } \langle \theta , f ( u , s ) \rangle$ with a decision variable u and a feature map $f .$ . Also in this case, the results below apply as they are once the set $X ( s )$ of this section is read as the image $f ( \mathcal { U } ( s ) , s )$ of the features (where $\mathcal { U } ( s )$ is the feasible set of the decision variable). Indeed, if $\mathcal { U } ( s )$ is a finite union of bounded closed convex polyhedra and each component of $f ( \cdot , s )$ is Lipschitz piecewise linear, then $f ( \mathcal { U } ( s ) , s )$ is also a finite union of polyhedra (cf. Kitaoka, 2024), and the arguments below apply to its set of integer points. Below, to keep the notation simple, we regard the feature map as the identity and argue on $X ( s )$

## 6.2. Explicit lower bounds for general ILPs.

Assumption 6.3 (Integer programming). For every $s \in { \mathcal { S } }$ we have $X ( s ) \subset \mathbb { Z } ^ { d }$ Furthermore, we set the coordinatewise ranges to be

$$
M _ { i } : = \operatorname* { s u p } _ { s \in \mathcal { S } } \left( \operatorname* { m a x } _ { x \in X ( s ) } x _ { i } - \operatorname* { m i n } _ { x \in X ( s ) } x _ { i } \right) \quad ( i = 1 , \ldots , d ) , \qquad M : = ( M _ { 1 } , \ldots , M _ { d } ) .\tag{6.3}
$$

An explicit lower bound on $\gamma _ { \mathrm { s u b } }$ is obtained for each combinatorial structure of the feasible set. The results in the case where the weight space Θ is the probability simplex $\Delta ^ { d - 1 } : = \{ \theta \in \mathbb { R } _ { > 0 } ^ { d } : \sum _ { i = 1 } ^ { d } \theta _ { i } = 1 \}$ and in the case where it is the unit ball $B ^ { d } : = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \leq \overline { { 1 } } \}$ are summarized in Table 3. Both the statements and the proofs are placed in Appendix G.

Table 3. Lower bounds on $\gamma _ { \mathrm { s u b } }$ by structure $( d \geq 2 )$ . The structure in each row is defined by the assumption in parentheses, and the theorem in parentheses below each value is the statement in which that bound is proved. Here $\| M \| _ { 2 }$ is the norm of the vector M in Equation (6.3), and $C _ { g } = g _ { \infty } ( \widetilde { A } )$ is the $\ell _ { \infty }$ norm of the Graver basis of the slack-augmented matrix $\widetilde { A } = [ A \vert \operatorname { I d } _ { N } ]$ (where Id is the identity matrix of order $N )$ ; if A is totally unimodular then $C _ { g } = 1$ The unit-ball entries marked with <sup>†</sup> require a full-dimensionality assumption; see Proposition G.2 and Remark G.3 for the details.
<table><tr><td rowspan="2">Structure of the feasible set</td><td colspan="2">Lower bound on  $\gamma _ { \mathrm { s u b } }$ </td></tr><tr><td> $\Theta = \Delta ^ { d - 1 }$ </td><td> $\Theta = B ^ { d }$ </td></tr><tr><td rowspan="3">General ILP (Assumption 6.3)</td><td>1</td><td>1 †  $\overline { { { 2 ^ { d - 1 } \sqrt { d - 1 } } \parallel M \parallel _ { 2 } ^ { d - 1 } } }$ </td></tr><tr><td> $\overline { { 2 ^ { d - 1 } \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) \Vert M \Vert _ { 2 } ^ { d - 1 } } }$  (Theorem G.4) 1</td><td>(Theorem G.1) 1 †</td></tr><tr><td> $\operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 }$  (Theorem G.20)</td><td> $\sqrt { d - 1 } ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 }$  (Theorem G.19)  $2 { \sqrt { 3 } }$ </td></tr><tr><td>M-convex set (Assumption G.11)</td><td> $\frac { 2 } { d ( d - 1 ) }$  (Theorem G.13) 2</td><td> $\overline { { \sqrt { d ( d ^ { 2 } - 1 ) } } }$  (Theorem G.13)</td></tr><tr><td>M-convex set (Assumption G.14)</td><td>d(d + 1) (Theorem Ġ.16)</td><td> $\sqrt { \frac { 6 } { d ( d + 1 ) ( 2 d + 1 ) } }$  &#x27;(Theorem G.16)</td></tr></table>

## 7. Explicit upper bounds on the number of mistakes and the regret by problem class

The upper bounds of Theorems C.2, D.6 and F.3 are nonincreasing in $\gamma ,$ so substituting the lower bounds of Table 3 yields explicit upper bounds by problem class. The orders of the results are summarized in Table 4 (for the statements including constants, see Appendix N).

From Table 4 we read of the following two points, each a comparison with the existing methods.

First, even for a general ILP, the regret of ONS and of growing-grid SGS-MetaGrad is $O ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) )$ ), which is independent of the total number of rounds T and polynomial in the dimension. Compared with the bound $O ( \| M \| _ { 2 } d \log { \frac { T } { d } } )$ of Sakaue et al. (2025b) for the same setting, the factor log $T$ disappears while the power of the dimension increases by one. Among the bounds independent of $T ,$ it turns the bound $\exp ( O ( d \log d ) )$ of Gollapudi et al. (2021, Theorem 4.2), which assumes neither a margin nor a gap, into one polynomial in the dimension; this paper does assume a uniform margin, and its bound depends linearly on $\| M \| _ { 2 }$

Table 4. Orders of the explicit upper bounds by problem class. The rows are indexed by the structure of the feasible set, the weight space Θ, and the criterion, and the columns by the method. The rows for linear inequalities refer to constraints of the form $A x \leq b ( s )$ , with $C _ { g } : = g _ { \infty } ( \widetilde { A } )$ (the same as in Table 3); $\| M \| _ { 2 }$ is the norm of the vector M in Equation (6.3), L is the constant in Equation (3.2), and in the ILP rows the bound $L \ \leq \ \| M \| _ { 2 }$ (Equation (G.2)) has been substituted. The values in the rows for $R _ { T } ^ { \mathrm { { e s t } } }$ are obtained by substituting the lower bounds into the upper bounds on the sum $\widetilde { R } _ { T }$ (item (iii) of each of Theorems C.2, D.6 and F.3), and the cumulative suboptimality regret $R _ { T } ^ { \mathrm { { s u b } } }$ has the same upper bound (Equation (3.8)). The M-convex and M<sup>♮</sup>-convex cases are combined since they have the same order. ONS and growing-grid SGS-MetaGrad are combined into one column since their orders coincide. The details of the substitution are given in Appendix N.
<table><tr><td>Structure</td><td>Θ</td><td>Criterion</td><td>SGS-OGD</td><td>ONS Growing-grid SGS-MetaGrad</td></tr><tr><td rowspan="3">General ILP (Assumption 6.3)</td><td rowspan="2"> $\Delta ^ { d - 1 }$ </td><td>K</td><td> $O ( d ^ { 2 } 4 ^ { d } \Vert M \Vert _ { 2 } ^ { 2 d } )$ </td><td> $O ( d ^ { 3 } 2 ^ { d } \| M \| _ { 2 } ^ { d } \log ( 2 \| M \| _ { 2 } ) )$ </td></tr><tr><td> $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> $O ( d 2 ^ { d } \| M \| _ { 2 } ^ { d + 1 } )$ </td><td> $O ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) )$ </td></tr><tr><td> $B ^ { d }$ </td><td>K  $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> $O ( d 4 ^ { d } \Vert M \Vert _ { 2 } ^ { 2 d } )$   $O ( \sqrt { d } 2 ^ { d } \| M \| _ { 2 } ^ { d + 1 } )$ </td><td> $O ( d ^ { 5 / 2 } 2 ^ { d } \| M \| _ { 2 } ^ { d } \log ( 2 \| M \| _ { 2 } ) )$   $O ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) )$ </td></tr><tr><td rowspan="2">Linear inequalities (Assumption G.17)</td><td> $\Delta ^ { d - 1 }$ </td><td>K  $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> $O ( L ^ { 2 } d ^ { 2 } ( 2 C _ { g } \sqrt { d } ) ^ { 2 d } )$   $O ( L ^ { 2 } d ( 2 C _ { g } \sqrt { d } ) ^ { d } )$ </td><td> $O ( L d ^ { 3 } ( 2 C _ { g } \sqrt { d } ) ^ { d } \log ( 2 C _ { g } d L ) )$   $O ( L d ^ { 2 } \log ( 2 C _ { g } d L ) )$ </td></tr><tr><td> $B ^ { d }$ </td><td> $K$   $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> $O ( L ^ { 2 } d ( 2 C _ { g } \sqrt { d } ) ^ { 2 d } )$   $O ( L ^ { 2 } { \sqrt { d } } ( 2 C _ { g } { \sqrt { d } } ) ^ { d } )$ </td><td> $O ( L d ^ { 5 / 2 } ( 2 C _ { g } \sqrt { d } ) ^ { d } \log ( 2 C _ { g } d L ) )$   $O ( L d ^ { 2 } \log ( 2 C _ { g } d L ) )$ </td></tr><tr><td>M-convex (Assumption G.11), M-convex</td><td> $\Delta ^ { d - 1 }$ </td><td> $K$   $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> $O ( L ^ { 2 } d ^ { 4 } )$   $O ( L ^ { 2 } d ^ { 2 } )$ </td><td> $O ( L d ^ { 3 } \log ( 2 d L ) )$   $O ( L d \log ( 2 d L ) )$ </td></tr><tr><td>(Assumption G.14)</td><td> $B ^ { d }$ </td><td> $K$   $R _ { T } ^ { \mathrm { { e s t } } }$ </td><td> ${ \cal O } ( L ^ { 2 } d ^ { 3 } )$   $O ( L ^ { 2 } d ^ { 3 / 2 } )$ </td><td> $O ( L d ^ { 5 / 2 } \log ( 2 d L ) )$   $O ( L d \log ( 2 d L ) )$ </td></tr></table>

Second, the regret O(Ld log(2dL)) for M-convex and $\mathrm { M } ^ { \natural } .$ -convex structures admits a direct comparison with the bound O(d log d) obtained by Oki and Sakaue (2026) under the same structure. The latter is a value under the normalization that makes the per-round regret $O ( 1 )$ (Oki and Sakaue, 2026, Assumption 2.2), which in the notation of this paper amounts to $\| \theta ^ { * } \| L = O ( 1 )$ . Its bound on the number of mistake rounds does not depend on that normalization, so without it their bound becomes $O ( \| \theta ^ { * } \| L d \log d )$ , and the only diference from our bound is the argument of the logarithm. The diference lies in the computational cost: whereas Oki and Sakaue (2026) computes a center of gravity at every round, our updates need only $O ( d ^ { 2 } )$ plus one generalized projection onto Θ per mistake round.

## 8. Conclusion

For online inverse linear optimization, this paper has proposed a mechanism— small-gradient skipping (SGS)—that skips both the update of the iterate and the advancement of the index of the internal state at rounds without a mistake. Under a uniform margin $\gamma > 0$ , we have shown that the three methods obtained by applying SGS to OGD, ONS and MetaGrad bound the number of mistakes $K$ , the cumulative suboptimality regret $R _ { T } ^ { \mathrm { { s u b } } }$ , and the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ all by quantities independent of the total number of rounds $T$ (Table 2). Furthermore, by substituting the lower bounds on the margin by structure (Table 3), we have given explicit upper bounds for situations in which the forward problem comes from an integer linear program. In particular, for an ILP with the probability simplex, ONS and growing-grid SGS-MetaGrad attain $R _ { T } ^ { \mathrm { e s t } } = O ( d ^ { 2 } \Vert M \Vert _ { 2 } \log ( 2 \Vert M \Vert _ { 2 } ) )$ (Table 1). The problem raised in $\ S 1$ was that every known bound independent of the total number of rounds has a limitation: the bound that assumes neither a margin nor a gap is exponential in the dimension (Gollapudi et al., 2021); the bound under a gap condition is proportional to the inverse square of the gap (Sakaue et al., 2025a); and the bound under M-convexity, O(d log d), is smaller than ours but requires computing a center of gravity at every round (Oki and Sakaue, 2026). Our bound assumes only a uniform margin, is polynomial in the dimension, and is obtained with a deterministic and light update— $\mathcal { O } ( d ^ { 2 } )$ plus one generalized projection per mistake round. It also removes the log $T$ dependence from the bound $\begin{array} { r } { { \dot { O } } ( \| M \| _ { 2 } d \log \frac { T } { d } ) } \end{array}$ of Sakaue et al. (2025b).

We list the remaining issues.

• The gap between the upper and lower bounds: in the case of an ILP, there is a gap of a factor d in the dimension, as well as a factor involving the coordinatewise ranges $\| M \| _ { 2 }$ , between our $R _ { T } ^ { \mathrm { e s t } } = O ( d ^ { 2 } \Vert M \Vert _ { 2 } \log ( 2 \Vert M \Vert _ { 2 } ) )$ and the known lower bound $\Omega ( d )$ (Sakaue et al., 2025b; Oki and Sakaue, 2026). Which of the two should be improved is an open question.

• Extension to noise and corruption: this paper is restricted to the noiseless setting. Frameworks that handle suboptimal feedback (Sakaue et al., 2025b) or corruption (Oki and Sakaue, 2026) have already been studied, and incorporating the SGS viewpoint into those analyses is an important direction for future work. Under corruption the per-mistakeround progress guaranteed by the uniform margin (Lemma C.1) is weakened, so the treatment of the quadratic term in the analysis of ONS has to be replaced by a per-round inequality involving the amount of corruption.

## References

Ahuja, R. K. and Orlin, J. B. (2001). Inverse optimization. Operations Research, 49(5):771–783.

Bärmann, A., Martin, A., Pokutta, S., and Schneider, O. (2018). An online-learning approach to inverse optimization. Available at arXiv:1810.12997.

Besbes, O., Fonseca, Y., and Lobel, I. (2021). Online learning from optimal actions. In The 34th Conference on Learning Theory, pages 586–586. PMLR.

Besbes, O., Fonseca, Y., and Lobel, I. (2025). Contextual inverse optimization: Ofline and online learning. Operations Research, 73(1):424–443.

Birge, J. R., Hortaçsu, A., and Pavlin, J. M. (2017). Inverse optimization for the recovery of market structure from market outcomes: An application to the miso electricity market. Operations Research, 65(4):837–855.

Chan, T. C., Eberg, M., Forster, K., Holloway, C., Ieraci, L., Shalaby, Y., and Yousefi, N. (2022). An inverse optimization approach to measuring clinical pathway concordance. Management Science, 68(3):1882–1903.

Chan, T. C., Mahmood, R., and Zhu, I. Y. (2023). Inverse optimization: Theory and applications. Operations Research.

Gentile, C. (2001). A new approximate maximal margin classification algorithm. Journal of Machine Learning Research, 2(Dec):213–242.

Gollapudi, S., Guruganesh, G., Kollias, K., Manurangsi, P., Leme, R., and Schneider, J. (2021). Contextual recommendations and low-regret cutting-plane algorithms. Advances in Neural Information Processing Systems, 34:22498–22508.

Hazan, E. (2019). Introduction to online convex optimization. Available at arXiv:1909.05207.

Hazan, E., Agarwal, A., and Kale, S. (2007). Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2):169–192.

Heuberger, C. (2004). Inverse combinatorial optimization: A survey on problems, methods, and results. Journal of Combinatorial Optimization, 8:329–361.

Kitaoka, A. (2024). Exact solution to data-driven inverse optimization of MILPs in finite time via gradient-based methods. Available at https://arxiv.org/abs/ 2405.14273v8.

Kitaoka, A. (2026). Explicit Iteration Complexity of Exact Data-Driven Inverse Optimization for Integer Linear Programs. Available at https://arxiv.org/ abs/2607.22263v1.

Liang, Z. and Dvorkin, Y. (2023). Data-driven inverse optimization for marginal ofer price recovery in electricity markets. In Proceedings of the 14th ACM International Conference on Future Energy Systems, pages 497–509.

Mohajerin Esfahani, P., Shafieezadeh-Abadeh, S., Hanasusanto, G. A., and Kuhn, D. (2018). Data-driven inverse optimization with imperfect information. Mathematical Programming, 167:191–234.

Murota, K. (1996). Convexity and Steinitz’s exchange property. Advances in Mathematics, 124(2):272–311.

Murota, K. (1998). Discrete convex analysis. Mathematical Programming, 83:313– 371.

Murota, K. (2003). Discrete Convex Analysis. Society for Industrial and Applied Mathematics, Philadelphia, Pennsylvania.

Murota, K. and Shioura, A. (1999). M-convex function on generalized polymatroid. Mathematics of Operations Research, 24(1):95–105.

Ng, A. Y., Russell, S., et al. (2000). Algorithms for inverse reinforcement learning. In 7th International Conference on Machine Learning, volume 1, page 2.

Oki, T. and Sakaue, S. (2026). Finite and corruption-robust regret bounds in online inverse linear optimization under M-convex action sets. Available at arXiv:2602.01682v2.

Onn, S. (2010). Nonlinear discrete optimization. Zurich Lectures in Advanced Mathematics. European Mathematical Society, Berlin.

Orabona, F. (2019). A Modern Introduction to Online Learning. arXiv preprint. arXiv:1912.13213.

Sakaue, S. (2026). Simple projection-free algorithm for contextual recommendation with logarithmic regret and robustness. Available at arXiv:2603.20826v2.

Sakaue, S., Bao, H., and Tsuchiya, T. (2025a). Revisiting online learning approach to inverse linear optimization: A Fenchel–Young loss perspective and gap-dependent regret analysis. In The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 46–54. PMLR. arXiv:2501.13648.

Sakaue, S., Tsuchiya, T., Bao, H., and Oki, T. (2025b). Online inverse linear optimization: Improved regret bound, robustness to suboptimality, and toward tight regret analysis. Available at arXiv:2501.14349v6, and to appear in The Thirty-Ninth Annual Conference on Neural Information Processing Systems.

Schrijver, A. (1986). Theory of Linear and Integer Programming. John Wiley & Sons, Chichester.

Sturmfels, B. (1996). Gröbner bases and convex polytopes, volume 8 of University Lecture Series. American Mathematical Society, Providence, Rhode Island.

Sun, C., Liu, S., and Li, X. (2023). Maximum optimality margin: A unified approach for contextual linear programming and inverse linear programming. In The 40th International Conference on Machine Learning, volume 202, pages 32886–32912. PMLR.

van Erven, T. and Koolen, W. M. (2016). Metagrad: Multiple learning rates in online learning. Advances in Neural Information Processing Systems, 29.

van Erven, T., Koolen, W. M., and van der Hoeven, D. (2021). Metagrad: Adaptation using multiple learning rates in online learning. Journal of Machine Learning Research, 22(161):1–61.

## Appendix A. Related work in detail

Finitely many updates under a margin condition: the classical line. We describe in detail the classical line and the precedents of the skipping mechanism mentioned in $\ S 2 .$ The Perceptron convergence theorem for linearly separable data is a classical result showing that the number of mistakes (updates) is bounded by $( R / \gamma ) ^ { 2 }$ in terms of the radius R of the data and the margin $\gamma ;$ it shares the $\gamma ^ { - 2 } \mathrm { - t y p e }$ structure of the upper bound with our bound $2 L ^ { 2 } D ^ { 2 } / \gamma ^ { 2 }$ on the number of mistakes of SGS-OGD (Theorem C.2). ALMA (Gentile, 2001) is a method that approximates the maximum-margin classifier without being given the margin γ explicitly, by means of the decaying step size $\eta _ { \boldsymbol { k } } \propto k ^ { - 1 / 2 }$ ; it shares its idea with the parameter-free step size design of this paper. These are, however, results for binary classification (the $0 / 1$ loss and linear surrogate losses), and they do not apply directly to the suboptimali $\mathrm { \ t y }$ loss treated in this paper (for an overview of the relation between online convex optimization and the Perceptron, see Orabona (2019)). In the context of inverse optimization, Sun et al. (2023) (Maximum Optimality Margin) gives the skeleton “separability → finitely many mistakes → exact recovery” by a Perceptrontype method, and is the prior work closest to the framework of this paper. The mechanism of not advancing the internal state in rounds without a mistake also has precedents. In the reduction from contextual recommendation to a cutting-plane algorithm of Gollapudi et al. (2021, Theorem 3.1), a round in which the proposal is correct is skipped: the state of the cutting-plane algorithm is reset to its state at the beginning of that round. Besbes et al. (2021, 2025) also use a threshold-type skip, leaving the ellipsoidal cone unchanged in periods where the decision is nearly optimal, which they introduce in order to keep the ellipsoid method from becoming ill-conditioned. The methods for which this paper formulates the mechanism are those whose internal state—the step-size index, the matrix $\Sigma _ { t }$ of ONS, and the grid of MetaGrad—would otherwise advance even in rounds without a mistake.

Finite regret in online inverse optimization. We now describe in detail how this paper relates to the three T-independent results listed in $\ S 2 .$ . Sakaue et al. (2025a) gives a finite regret of ${ \cal O } ( 1 / \Delta ^ { 2 } )$ under the gap condition $\Delta > 0$ , which is the closest to the uniform margin assumption of this paper. (For the definition of $\Delta$ and its rewriting in the notation of this paper, see Appendix O.) Oki and Sakaue (2026) gives $R _ { T } ^ { \mathrm { e s t } } =$ O(d log d) under the M-convexity of the action set. Their method, however, computes a center of gravity at every round. The exact computation is $\# \mathrm { P } .$ -hard, and although Oki and Sakaue (2026) also give a polynomial-time randomized implementation with approximate centers of gravity, that implementation guarantees the bound only in expectation and costs $O ( d ^ { 6 } \log d \log T )$ per round up to polylogarithmic factors arising from the random-walk implementation. M-convexity appears in this paper not as a requirement of the method but as a structural condition that makes the lower bound on the margin polynomial in the dimension (Table 3). Consequently, our upper bound in the M-convex case is $O ( L d \log ( 2 d L ) )$ (§7). The bound exp(O(d log d)) of Gollapudi et al. (2021, Theorem 4.2) is obtained through Gollapudi et al. (2021, Theorem 3.1), which reduces contextual recommendation to a cutting-plane algorithm. Its assumptions are weaker than ours in that it requires neither a margin nor a gap condition, but it is exponential in the dimension, and the authors themselves leave the true regret of that algorithm—in particular whether a polynomial dependence on the dimension is attainable—as an open question.

Uniform margins in ofline inverse optimization. The uniform margin assumption (Assumption 3.1(4)) has a precedent in ofline (batch) inverse optimization. Kitaoka (2024) introduced a geometric constant $\gamma ( \ell _ { \mathrm { s u b } } ) ~ > ~ 0$ of the same kind for the suboptimality loss on a finite sample, and showed that the projected subgradient method reaches the minimum value 0 of the loss in $O ( 1 / \gamma ( \ell _ { \mathrm { s u b } } ) ^ { 2 } )$ iterations. Our upper bound $2 L ^ { 2 } D ^ { 2 } / \gamma ^ { 2 }$ on the number of mistakes of SGS-OGD (Theorem C.2) amounts to transferring this $\gamma ^ { - 2 }$ -type dependence to the online setting. Kitaoka (2026) gives explicit lower bounds on this constant by test sets and Graver bases in the case where the forward problem is an ILP, and §6 applies that technique to the uniform margin. On the other hand, neither bounding the regret by a constant independent of the total number of rounds $T$ nor bounding the number of mistakes K (the number of rounds with $\hat { x } ^ { t } \neq x ^ { t } )$ is new in itself. Bärmann et al. (2018, Corollary 10) bound the number of rounds with $\hat { x } ^ { t } \neq x ^ { t }$ by $O ( \sqrt { T } )$ under a ∆-stability condition; the optimality-driven perceptron of Sun et al. (2023) bounds it by a quantity independent of $T$ under a separability condition; and Oki and Sakaue (2026) bound the number of rounds with nonzero regret by $O ( d \log d )$ under M-convexity. What this paper adds is that a single condition—the existence of a witness $\bar { \theta } \in \Theta$ with a uniform margin—bounds $K , R _ { T } ^ { \mathrm { s u b } }$ and $R _ { T } ^ { \mathrm { { e s t } } }$ simultaneously and independently of $T$ , without assuming integrality of $\theta ^ { * }$ or M-convexity of the action set, and that the margin itself admits explicit lower bounds in terms of combinatorial structure (§6).

Generality of the weight space. The weight space Θ from which the learner chooses its predictions (§3) is also treated diferently in the existing work and in this paper. Many of the existing studies state their guarantees for a set Θ specific to the method: the unit sphere in Besbes et al. (2021, 2025), the unit ball in Gollapudi et al. (2021), and the whole of $\mathbb { R } ^ { d }$ in Oki and Sakaue (2026) and Sakaue (2026) (Table 6). By contrast, Bärmann et al. (2018), Sakaue et al. (2025a) and Sakaue et al. (2025b) allow a general $\Theta .$ . As with the latter, the only condition we impose on Θ is that it be nonempty, bounded, closed and convex (Assumption 3.1(1)), which covers both the probability simplex and the unit ball.

This generality is essential for our results. When the forward problem is a general ILP, the explicit lower bound on the margin holds unconditionally if Θ is the probability simplex (Theorem G.4), whereas for the unit ball it requires that the convex hull of the diference vectors between the correct action and the other candidate actions be full-dimensional (Theorem G.1). Hence the unconditional explicit upper bounds of Table 1 rely on our being able to choose the probability simplex as Θ.

## Appendix B. Details of the algorithms

In this appendix we collect the pseudocode of the algorithms treated in §5 together with the accompanying remarks.

B.1. SGS-OGD. Here $\begin{array} { r } { \Pi _ { \Theta } ( y ) : = \arg \operatorname* { m i n } _ { \theta \in \Theta } \| \theta - y \| } \end{array}$ is the Euclidean projection, which is uniquely determined since Θ is nonempty, closed and convex.

B.2. ONS. The method obtained by applying ONS (Hazan et al., 2007) to online inverse linear optimization is shown in Algorithm 2 (cf. Sakaue et al., 2025b). It runs ONS on the exp-concave surrogate loss

$$
\ell _ { t } ^ { \eta } ( \theta ) : = - \eta \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle + \eta ^ { 2 } \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle ^ { 2 } , \qquad g ^ { t } : = \hat { x } ^ { t } - x ^ { t }\tag{B.1}
$$

Algorithm 1 Small-gradient skipping online gradient descent (SGS-OGD)   
Require: step size coeficient $\alpha > 0$ (default $\alpha = D / ( L { \sqrt { 2 } } ) )$ , initial point $\hat { \theta } ^ { 1 } \in \Theta$   
1: $k \gets 1$   
2: for $t = 1 , \dots , T$ do   
3: receive $s ^ { t } ,$ compute and present with the oracle the proposal $\hat { x } ^ { t } \in$   
arg max $\ _ { \cdot x \in Y ( s ^ { t } ) } \langle \hat { \theta } ^ { t } , x \rangle$ (Equation (3.3)), and observe $x ^ { t }$   
4: if $\hat { x } ^ { t } \neq x ^ { t }$ then   
5: $g ^ { t } \gets \hat { x } ^ { t } - x ^ { t } , \hat { \theta } ^ { t + 1 } \gets \Pi _ { \Theta } \Big ( \hat { \theta } ^ { t } - \alpha k ^ { - 1 / 2 } g ^ { t } \Big ) , k \gets k + 1$   
6: else   
7: $\hat { \theta } ^ { t + 1 } \gets \hat { \theta } ^ { t }$   
8: end if   
9: end for

for a learning rate $\eta > 0 ;$ using the fact that its gradient at $\theta = \hat { \theta } ^ { t }$ is $\nabla \ell _ { t } ^ { \eta } ( \hat { \theta } ^ { t } ) = \eta g ^ { t }$ Below, for symmetric matrices $\Sigma , \Sigma ^ { \prime }$ of order $d , \Sigma \succeq \Sigma ^ { \prime }$ means that $\Sigma - \Sigma ^ { \prime }$ is positive semidefinite and $\Sigma \succ \Sigma ^ { \prime }$ that $\Sigma - \Sigma ^ { \prime }$ is positive definite (the Loewner order); in particular $\Sigma \succ 0$ means that $\Sigma$ is positive definite. We also write $\operatorname { I d } _ { d }$ for the identity matrix of order d (and likewise $\operatorname { I d } _ { n } .$ Id<sub>N</sub> for other orders).

```latex
Algorithm 2 Online Newton Step (ONS) for online inverse linear optimization
Require: weight space Θ (Assumption 3.1), initial point $\hat { \theta } ^ { 1 } \in \Theta$ , learning rate
$\eta > 0 ,$ parameter $\kappa > 0 .$ , positive definite matrix $\Sigma _ { 0 } \succ 0$
1: for $t = 1 , \dots , T$ do
2: receive $s ^ { t } ,$ compute and present with the oracle the proposal $\hat { x } ^ { t } \in$
arg ma $\operatorname { \dot { } } _ { x \in Y ( s ^ { t } ) } \langle \hat { \theta } ^ { t } , x \rangle$ (Equation (3.3)), and observe $x ^ { t }$
3: $g ^ { t } \gets \hat { x } ^ { t } - x ^ { \hat { t } } , \nabla ^ { t } \gets \eta g ^ { t }$ ▷ gradient of the surrogate loss Equation (B.1)
4: $\Sigma _ { t } \gets \Sigma _ { t - 1 } + \nabla ^ { t } ( \nabla ^ { t } ) ^ { \top }$
5: $\begin{array} { r } { \small \hat { \theta } ^ { t + 1 }  \Pi _ { \Theta } ^ { \Sigma _ { t } } ( \hat { \theta } ^ { t } - \frac { 1 } { \kappa } \Sigma _ { t } ^ { - 1 } \nabla ^ { t } ) } \end{array}$ ▷ generalized projection
6: end for
```

Algorithm 3 Online Newton Step (ONS)   
Require: geometric constants $L , D$ (Equation (3.2)), initial point $\hat { \theta } ^ { 1 } \in \Theta$   
1: $ { k } \gets 1 , \eta \gets \frac { 1 } { L D } ,  { \Sigma } _ { 0 } \gets D ^ { - 2 }  { \mathrm { I d } } _ { d }$   
2: for $t = 1 , \ldots , T$ do   
3: receive $s ^ { t } .$ , compute and present with the oracle the proposal $\hat { x } ^ { t } \in$   
arg ma $\mathfrak { c } _ { x \in Y ( s ^ { t } ) } \langle \hat { \theta } ^ { t } , x \rangle$ (Equation (3.3)), and observe $x ^ { t }$   
4: if $\hat { x } ^ { t } \neq x ^ { t }$ (a mistake) then   
5: $\begin{array} { r l } & { g ^ { t } \dot { }  \hat { x } ^ { t } - x ^ { t } , \nabla ^ { t }  \eta g ^ { t } , \Sigma _ { k }  \Sigma _ { k - 1 } + \nabla ^ { t } ( \nabla ^ { t } ) ^ { \top } } \end{array}$   
6: $\hat { \theta } ^ { t + 1 } \gets \Pi _ { \Theta } ^ { \Sigma _ { k } } \left( \hat { \theta } ^ { t } - \Sigma _ { k } ^ { - 1 } \nabla ^ { t } \right) , k \gets k + 1$   
7: else   
8: $\hat { \theta } ^ { t + 1 } \gets \hat { \theta } ^ { t }$   
9: end if   
10: end for

Here $\begin{array} { r } { \Pi _ { \Theta } ^ { \Sigma } ( y ) : = \arg \operatorname* { m i n } _ { \theta \in \Theta } \| \theta - y \| _ { \Sigma } ^ { 2 } } \end{array}$ (where $\| \boldsymbol { x } \| _ { \Sigma } ^ { 2 } : = \boldsymbol { x } ^ { \top } \Sigma \boldsymbol { x } )$ is the generalized projection with respect to the Σ-norm (a convex quadratic program over Θ), which is uniquely determined since $\Sigma \succ 0$ and Θ is nonempty, closed and convex. The matrix $\Sigma _ { k } \succeq D ^ { - 2 } \operatorname { I d } _ { d } \succ 0$ is always positive definite. The inverse $\Sigma _ { k } ^ { - 1 }$ can be updated by a rank-one update via the Sherman–Morrison formula, and one update costs $O ( d ^ { 2 } )$ . The parameters of the algorithm are only L and $D ;$ no knowledge of the margin γ or of the total number of rounds T is required.

When the loss is α-exp-concave and satisfies ma $\begin{array} { r } { \mathrm { x } _ { \theta \in \Theta } | \langle \nabla \ell _ { t } ^ { \eta } ( \hat { \theta } ^ { t } ) , \theta - \hat { \theta } ^ { t } \rangle | \leq \beta } \end{array}$ the standard choice for ONS is $\begin{array} { r } { \kappa = \frac { 1 } { 2 } \operatorname* { m i n } \lbrace \frac { 1 } { \beta } , \alpha \rbrace , \Sigma _ { 0 } = \frac { d } { D ^ { 2 } \kappa ^ { 2 } } \operatorname { I d } _ { d } } \end{array}$ . The η-experts of MetaGrad (Definition B.3) follow this choice (there, since the center $\hat { \theta } ^ { t }$ of the surrogate loss difers from the point of the expert, the gradient acquires a factor $1 - 2 \eta \langle g ^ { t } , \hat { \theta } ^ { t } - \theta \rangle ;$ . By contrast, Algorithm 3 uses the same update formula with the diferent choice $\begin{array} { r } { \eta = \frac { 1 } { L D } , \kappa = 1 , \Sigma _ { 0 } = D ^ { - 2 } \mathrm { I d } _ { d } } \end{array}$ (and in addition omits the computation at rounds without a mistake). The validity of this choice is shown directly by the potential inequality of $\mathrm { A }$ ppendix D, without going through the standard regret bound.

Remark B.1 (Algorithm 3 is ONS itself). The branching in Algorithm 3 is there to make explicit that the computation at rounds without a mistake is omitted; it does not change the sequence of iterates. Indeed, at a round without a mistake we have $g ^ { t } = \hat { x } ^ { t } - x ^ { t } = 0$ , that is, $\nabla ^ { t } = 0$ , so the information matrix is unchanged, $\Sigma \gets \Sigma + \nabla ^ { t } ( \nabla ^ { t } ) ^ { \top } = \Sigma$ , and the update becomes $\Pi _ { \Theta } ^ { \Sigma } ( \hat { \theta } ^ { t } - \Sigma ^ { - 1 } \cdot 0 ) = \Pi _ { \Theta } ^ { \Sigma } ( \hat { \theta } ^ { t } ) \stackrel { - } { = } \hat { \theta } ^ { t }$ (the projection is the identity since $\hat { \theta } ^ { t } \in \Theta )$ . That is, Algorithm 3 generates the same sequence of iterates as plain ONS (Hazan et al., 2007) using the subgradient $\boldsymbol { g } ^ { t } = \hat { \boldsymbol { x } } ^ { t } - \boldsymbol { x } ^ { t }$ at every round. Consequently the contribution of this subsection is not the proposal of an algorithm, but the improvement of the existing guarantee for ONS by incorporating the SGS viewpoint (Remark 4.2) into the analysis of ONS: evaluating the log-det potential by the number of mistakes $K$ rather than by the total number of rounds $T _ { \cdot }$ , and balancing it against the per-mistake progress guaranteed by the uniform margin, replaces the regret upper bound $O ( d \log T )$ by the T-independent Theorem D.6. In implementation, the advantage remains that the $O ( d ^ { 2 } )$ matrix update and the generalized projection can be omitted at rounds without a mistake (Table 5).

Remark B.2 (Relation to Sakaue et al. (2025b)). Algorithm 3 coincides with the construction of Sakaue et al. (2025b, Theorem 3.1) applying ONS to the exp-concave surrogate loss $\ell _ { k } ^ { \eta } ( \theta ) = - \eta \langle \hat { \theta } ^ { t _ { k } } - \theta , g ^ { t _ { k } } \rangle + \eta ^ { 2 } \langle \hat { \theta } ^ { t _ { k } } - \theta , g ^ { t _ { k } } \rangle ^ { 2 }$ , once the parameters are fixed as $\eta = 1 / ( L D ) , \kappa = 1 , \Sigma _ { 0 } = D ^ { - 2 } \mathrm { I d } _ { d }$ . As stated in Remark B.1 there is no diference in the algorithm; the diference is on the side of the guarantee: that paper shows $O ( L D d \log { \frac { T } { d } } )$ without assuming a margin, whereas this paper shows a T-independent number of mistakes and cumulative suboptimality regret under a uniform margin. Our analysis does not use the surrogate loss explicitly, but proves the same content directly as a quadratic potential inequality (Appendix D).

B.3. MetaGrad. MetaGrad is a universal online learning method that runs in parallel the η-experts (which apply ONS to the surrogate loss $\ell _ { k } ^ { \eta }$ of Remark B.2) for each η in a learning-rate grid $\mathcal { E } ;$ it consists of two layers, these η-experts and a master. Here the master is the algorithm that, at each round $j ,$ updates the weight $p _ { j } ^ { \eta }$ attached to the η-expert by exponential weighting with respect to the surrogate losses and outputs the weighted average of the points $\boldsymbol { w _ { j } ^ { \eta } }$ of the η-experts, weighted

by the learning rates,

$$
w _ { j } : = \frac { \sum _ { \eta \in \varepsilon } \eta p _ { j } ^ { \eta } w _ { j } ^ { \eta } } { \sum _ { \eta \in \varepsilon } \eta p _ { j } ^ { \eta } } ;\tag{B.2}
$$

we call $w _ { j }$ the point of the master. The concrete forms of the grid $\mathcal { E } ,$ the weights $p _ { j } ^ { \eta }$ and the points $\boldsymbol { w _ { j } ^ { \eta } }$ are given in Algorithm 4. Algorithm 4 shows MetaGrad for a general sequence of convex losses (a restatement of Algorithm 2 of Sakaue et al., 2025b; the grid is constructed from $\bar { m } )$ , and Algorithm 5 shows its SGS version. The ONS used by the η-experts is made concrete for the surrogate loss as follows.

Definition B.3 (The ONS of an η-expert; following Appendix C of Sakaue et al., 2025b). Let $\mathcal { W } \subset \mathbb { R } ^ { n }$ be a nonempty closed convex set whose $\ell _ { 2 }$ diameter is at most $W > 0$ , let $G , H > 0$ , and take a learning rate $\eta \in ( 0 , \frac { 1 } { 5 H } ]$ . For the surrogate loss $\ell _ { j } ^ { \eta } ( w ) = - \eta \langle w _ { j } - w , g _ { j } \rangle + \eta ^ { 2 } \langle w _ { j } - w , g _ { j } \rangle ^ { 2 }$ associated with the point $w _ { j } \in \mathcal W$ of the master and a subgradient $g _ { j }$ (with $\| g _ { j } \| \leq G$ and sup $\{ \langle w ^ { \prime } - w , g _ { j } \rangle \ : | \ : w , w ^ { \prime } \in \mathcal { W } \} \leq H )$ ， the η-expert is the ONS that starts from an initial point $w _ { 1 } ^ { \eta } \in \mathcal { W }$ and updates

$$
\begin{array} { r } { \nabla _ { j } ^ { \eta } : = \nabla \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) = \eta \left( 1 - 2 \eta \langle g _ { j } , w _ { j } - w _ { j } ^ { \eta } \rangle \right) g _ { j } , } \end{array}\tag{B.3}
$$

$$
\Sigma _ { j } ^ { \eta } : = \Sigma _ { j - 1 } ^ { \eta } + \nabla _ { j } ^ { \eta } ( \nabla _ { j } ^ { \eta } ) ^ { \top } , \qquad \Sigma _ { 0 } ^ { \eta } : = \frac { n } { W ^ { 2 } \kappa _ { \eta } ^ { 2 } } \mathrm { I d } _ { n } ,\tag{B.4}
$$

$$
w _ { j + 1 } ^ { \eta } : = \Pi _ { \mathcal { W } } ^ { \Sigma _ { j } ^ { \eta } } \left( w _ { j } ^ { \eta } - \frac { 1 } { \kappa _ { \eta } } \big ( \Sigma _ { j } ^ { \eta } \big ) ^ { - 1 } \nabla _ { j } ^ { \eta } \right) , \qquad \kappa _ { \eta } : = \frac { 1 } { ( 1 + 2 \eta H ) ^ { 2 } } .\tag{B.5}
$$

The only diference from Algorithm 2 is that, since the center $w _ { j }$ of the surrogate loss difers from the point $w _ { i } ^ { \bar { \eta } }$ being updated, the gradient Equation (B.3) acquires the factor $1 - 2 \eta \langle g _ { j } , w _ { j } - w _ { j } ^ { \eta } \rangle$ ; the parameters follow the standard choice $\kappa = \kappa _ { \eta } ,$ $\begin{array} { r } { \Sigma _ { 0 } ^ { \eta } = \frac { n } { W ^ { 2 } \kappa _ { n } ^ { 2 } } \operatorname { I d } _ { n } } \end{array}$ . Here $\begin{array} { r } { \Pi _ { \mathcal { W } } ^ { \Sigma } ( y ) : = \arg \operatorname* { m i n } _ { w \in \mathcal { W } } \| w - y \| _ { \Sigma } ^ { 2 } } \end{array}$ is the same generalized projection with respect to the Σ-norm as in Algorithm 3.

The origin of the parameter $\kappa _ { \eta }$ is as follows. The standard form of ONS sets $\begin{array} { r } { \kappa = \frac { 1 } { 2 } \operatorname* { m i n } \bar { \lbrace } \frac { 1 } { \beta } , \alpha \rbrace } \end{array}$ and $\begin{array} { r } { \Sigma _ { 0 } = \frac { n } { W ^ { 2 } \kappa ^ { 2 } } \dot { \mathrm { I d } } _ { n } } \end{array}$ from the exp-concavity constant α and an upper bound $\beta$ on the inner product with the gradient (cf. Hazan et al., 2007). For the surrogate loss, from $\mathsf { \bar { V } } ^ { 2 } \ell _ { j } ^ { \eta } ( w ) = 2 \eta ^ { 2 } g _ { j } \bar { g } _ { j } ^ { \top }$ and Equation (B.3) we have $\begin{array} { r } { \nabla \ell _ { j } ^ { \eta } ( w ) \nabla \ell _ { j } ^ { \eta } ( w ) ^ { \top } \preceq \eta ^ { 2 } ( 1 + 2 \eta H ) ^ { 2 } g _ { j } g _ { j } ^ { \top } = \frac { ( 1 + 2 \eta H ) ^ { 2 } } { 2 } \nabla ^ { 2 } \ell _ { j } ^ { \eta } ( w ) } \end{array}$ , so it is $\begin{array} { r } { \alpha = \frac { 2 } { ( 1 + 2 \eta H ) ^ { 2 } } - } \end{array}$ exp-concave, and ma ${ { \mathfrak { c } } _ { w \in \mathcal { W } } | \langle \nabla \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) , w - w _ { j } ^ { \eta } \rangle | \leq \beta : = \eta H + 2 \eta ^ { 2 } H ^ { 2 } , \| \nabla \ell _ { j } ^ { \eta } ( w ) \| } \le$ $\lambda : = \eta ( 1 + 2 \eta H ) G$ hold. Since $\begin{array} { r } { \frac { 1 } { \alpha } = \frac { 1 } { 2 } + 2 \eta H + 2 \eta ^ { 2 } H ^ { 2 } \geq \beta } \end{array}$ , we get $\begin{array} { r } { \kappa = \frac { \alpha } { 2 } = \kappa _ { \eta } , } \end{array}$ and under $\begin{array} { r } { \eta \le \frac { 1 } { 5 H } } \end{array}$ we have $\kappa _ { \eta } \in [ \frac { 2 5 } { 4 9 } , 1 )$ and $\begin{array} { r } { \kappa _ { \eta } \lambda = \frac { \eta G } { 1 + 2 \eta H } \leq \frac { G } { 7 H } } \end{array}$ (this $\textstyle { \frac { 1 } { 4 9 } }$ is the origin of the denominator 49 in Equation (E.5)). In the SGS version, as in Algorithm 3, the update of the experts, the update of the weights, and the advancement of the index k are restricted to mistake rounds only. Here too the skipping holds automatically, and the substantial diference from plain MetaGrad is limited to the construction of the learning-rate grid (Remark B.4): building the grid from the side of the number of mistakes rather than the total number of rounds is what makes a T-independent guarantee possible. The construction of the grid uses an upper bound $\bar { K } \ge K$ on the number of mistakes (since there are at most $T$ mistakes, $\bar { K } = T$ is admissible; that the dependence stays at log log $\bar { K }$ is stated in Remark E.2). This dependence on $\bar { K }$ can be removed by growing the grid according to the number of mistakes (Theorem F.3).

```latex
Algorithm 4 MetaGrad (the version whose learning-rate grid is constructed from
m¯ )
Require: nonempty closed convex set $\mathcal { W } \subset \mathbb { R } ^ { n }$ , constants $W , H > 0 ,$ an upper
bound m¯ on the number of rounds m; the convex losses $h _ { 1 } , \hdots , h _ { m } \colon \mathcal { W } \to \mathbb { R }$ are
given online (the ONS of an η-expert is Definition B.3)
1: learning-rate grid $\begin{array} { r } { \mathcal { E }  \{ \eta _ { i } : = \frac { 2 ^ { - i } } { 5 H } \Big | i = 0 , 1 , \dotsc , \lceil \frac { 1 } { 2 } \log _ { 2 } \bar { m } \rceil \} } \end{array}$
2: for each $\eta _ { i } \in { \mathcal { E } } ,$ , prepare an initial weight $\begin{array} { r } { p _ { 1 } ^ { \eta _ { i } }  \frac { C } { ( i + 1 ) ( i + 2 ) } } \end{array}$ (where C is the
normalizing constant making $\begin{array} { r } { \sum _ { \eta \in \mathcal { E } } p _ { 1 } ^ { \eta } = 1 ) } \end{array}$ and an initial point $w _ { 1 } ^ { \eta _ { i } } \in \mathcal { W }$ of the
η<sub>i</sub>-expert
3: for $j = 1 , \ldots , m$ do
4: output $\begin{array} { r } { w _ { j }  \sum _ { \eta \in \mathcal { E } } \eta p _ { j } ^ { \eta } w _ { j } ^ { \eta } / \sum _ { \eta \in \mathcal { E } } \eta p _ { j } ^ { \eta } } \end{array}$ and observe a subgradient $g _ { j } \in$
$\partial h _ { j } ( w _ { j } )$
5: define the surrogate losses $\ell _ { j } ^ { \eta } ( w ) : = - \eta \langle w _ { j } - w , g _ { j } \rangle + \eta ^ { 2 } \langle w _ { j } - w , g _ { j } \rangle ^ { 2 } \left( \eta \in \mathcal { E } \right)$
6: for each $\begin{array} { r l r l r l r l } { \eta } & { { } \in } & { \mathcal { E } \colon } & { p _ { j + 1 } ^ { \eta } } & { { }  } & { p _ { j } ^ { \eta } \exp ( - \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) ) / Z _ { j } } \end{array}$ (where $Z _ { j } : =$
$\begin{array} { r } { \sum _ { \eta ^ { \prime } \in \mathcal { E } } p _ { j } ^ { \eta ^ { \prime } } \exp ( { - \ell _ { j } ^ { \eta ^ { \prime } } ( w _ { j } ^ { \eta ^ { \prime } } ) } ) ) } \end{array}$ , and compute $\boldsymbol { w _ { j + 1 } ^ { \eta } }$ by the ONS update of the
η-expert on $\ell _ { j } ^ { \eta }$
7: end for
Algorithm 5 Small-gradient skipping MetaGrad (SGS-MetaGrad)
Require: geometric constants $L , D$ (Equation (3.2)), an upper bound $\bar { K }$ on the
number of mistakes (for instance ${ \bar { K } } = T )$
1: learning-rate grid $\begin{array} { r } { \mathcal { E }  \{ \eta _ { i } : = \frac { 2 ^ { - i } } { 5 L D } \Big | i = 0 , 1 , \dotsc , \lceil \frac { 1 } { 2 } \log _ { 2 } \bar { K } \rceil \} } \end{array}$
2: for each $\eta _ { i } \in { \mathcal { E } } ,$ , prepare an initial weight $\begin{array} { r } { p _ { 1 } ^ { \eta _ { i } }  \frac { C } { ( i + 1 ) ( i + 2 ) } } \end{array}$ (where C is the
normalizing constant making $\begin{array} { r } { \sum _ { \eta \in \mathcal { E } } p _ { 1 } ^ { \eta } = 1 ) } \end{array}$ and an initial point $\theta _ { 1 } ^ { \eta _ { i } } \in \Theta$ of the
η -expert (Definition B.3)
3: $\begin{array} { r } { k  1 , \hat { \theta } ^ { 1 }  \sum _ { \eta \in { \mathcal { E } } } \eta p _ { 1 } ^ { \eta } \theta _ { 1 } ^ { \eta } / \sum _ { \eta \in { \mathcal { E } } } \eta p _ { 1 } ^ { \eta } } \end{array}$
4: for $t = 1 , \dots , T$ do
5: receive $s ^ { t } .$ compute and present with the oracle the proposal $\hat { x } ^ { t } \in$
arg $\mathrm { m a x } _ { x \in Y ( s ^ { t } ) } \langle \hat { \theta } ^ { t } , x \rangle$ (Equation (3.3)), and observe $x ^ { t }$
6: if $\hat { x } ^ { t } \neq x ^ { t }$ (a mistake) then
7: $g ^ { t } \gets \hat { x } ^ { t } - x ^ { t } , \ell _ { k } ^ { \eta } ( \theta ) : = - \eta \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle + \eta ^ { 2 } \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle ^ { 2 } ~ ( \eta \in \mathcal { E } )$
8: for each $\begin{array} { r l r l r l } { \eta } & { { } \stackrel {  } { \in } } & { \mathcal { E } : } & { p _ { k + 1 } ^ { \eta } } & {  } & { p _ { k } ^ { \eta } \exp ( - \ell _ { k } ^ { \eta } ( \theta _ { k } ^ { \eta } ) ) / Z _ { k } } \end{array}$ (where $Z _ { k } : = \begin{array} { r l } \end{array}$
$\begin{array} { r l } { ~ } & { { } \sum _ { \eta ^ { \prime } \in \mathcal { E } } p _ { k } ^ { \eta ^ { \prime } } \exp ( { - \ell _ { k } ^ { \eta ^ { \prime } } ( \theta _ { k } ^ { \eta ^ { \prime } } ) } ) ) } \end{array}$ , and compute $\theta _ { k + 1 } ^ { \eta }$ by the ONS update of the
η-expert on $\ell _ { k } ^ { \eta }$
9: $\begin{array} { r } { \hat { \theta } ^ { t + 1 } \gets \sum _ { \eta \in \mathcal { E } } \eta p _ { k + 1 } ^ { \eta } \theta _ { k + 1 } ^ { \eta } / \sum _ { \eta \in \mathcal { E } } \eta p _ { k + 1 } ^ { \eta } , k \gets k + 1 } \end{array}$
10: else
11: $\hat { \theta } ^ { t + 1 } \gets \hat { \theta } ^ { t }$
12: end if
13: end for
```

Every iterate of Algorithms 5 and 6 stays in Θ. Indeed, each η-expert starts at a point of Θ and is updated by Equation (B.5), whose generalized projection $\Pi _ { \mathcal { W } } ^ { \Sigma _ { j } ^ { \eta } }$ maps into $\mathcal { W } = \Theta$ , so that $\theta _ { k } ^ { \eta } \in \Theta ;$ and the output of the master is the weighted average with coeficients $\eta p ^ { \eta } / \sum _ { \eta ^ { \prime } } \eta ^ { \prime } p ^ { \eta ^ { \prime } }$ , which are nonnegative and sum to one, so $\hat { \theta } ^ { t }$ is a convex combination of points of Θ and hence $\hat { \theta } ^ { t } \in \Theta$ by Assumption 3.1(1). In particular the hypothesis $w _ { 1 } , \dots , w _ { m } \in \mathcal { W }$ of Proposition E.3 is satisfied.

Remark B.4 (Algorithm 5 is MetaGrad itself). The branching in Algorithm 5 does not change the sequence of iterates either, for the same reason as in Remark B.1. At a round without a mistake we have $\boldsymbol { g } ^ { t } = \hat { x } ^ { t } - \boldsymbol { x } ^ { t } = 0$ , so the surrogate losses become $\ell _ { k } ^ { \eta } \equiv 0$ , the weights are unchanged, $p ^ { \eta } \exp ( 0 ) = p ^ { \eta }$ , and the gradient Equation (B.3) of an expert is also $\nabla ^ { \eta } = 0$ , so neither $\Sigma ^ { \eta }$ nor the point of the expert moves. Hence the output of the master does not change either. That is, Algorithm 5 with the grid upper bound taken as $\bar { K } = T$ generates the same sequence of iterates as plain MetaGrad using the subgradient $\boldsymbol { g } ^ { t } = \hat { x } ^ { t } - \boldsymbol { x } ^ { t }$ at every round. The claim of this subsection is likewise not the proposal of a new algorithm, but the improvement of the guarantee of MetaGrad under a margin by incorporating the SGS viewpoint into the analysis of MetaGrad: evaluating the variance term appearing in the regret upper bound of MetaGrad only at mistake rounds and balancing it, by self-bounding, against the per-mistake progress guaranteed by the uniform margin replaces the $\begin{array} { r } { { \check { O } } ( L D d \log { \frac { \tilde { T } } { d } } ) } \end{array}$ of Sakaue et al. (2025b) by the $T \cdot$ -independent Theorem E.1. The diference between Algorithm 5 and plain MetaGrad is limited to two points: (i) the implementation advantage that the expert updates (the grid size times $( O ( d ^ { 2 } )$ plus a generalized projection)) can be omitted at rounds without a mistake, and (ii) that if an upper bound $\bar { K } < T$ on the number of mistakes is known then the grid (and hence the number of experts) can be taken smaller. This identity, however, concerns only the fixed-grid version Algorithm 5 and does not extend to the growing-grid version Algorithm 6 (Remark B.5).

## B.4. Growing-grid SGS-MetaGrad.

Removing the dependence on K<sup>¯</sup> by a growing grid. The factor $c _ { 0 }$ in Remark E.2(b) originates from fixing the grid E in advance by an upper bound K<sup>¯</sup> on the number of mistakes. If we do not fix the grid but keep adding smaller learning rates as the number of mistakes k progresses, then the input K<sup>¯</sup> itself becomes unnecessary and $c _ { 0 }$ is replaced by $c _ { 0 } ( K ) : = 2 \log ( \textstyle { \frac { 1 } { 2 } } \log _ { 2 } K + 3 )$ with the realized number of mistakes $K .$ The self-bounding of the number of mistakes closes without any dependence on $T$ under this replacement as well, and the number of mistakes and the cumulative suboptimality regret become constants that are completely independent of $T$ (Theorem F.3). Algorithm 6 shows the growing-grid version. It difers from Algorithm 5 in the following three points: (i) the prior weights are taken as $\begin{array} { r } { p _ { i } = \frac { 1 } { ( i + 1 ) ( i + 2 ) } } \end{array}$ on the countable grid $\begin{array} { r } { \{ \eta _ { i } = \frac { 2 ^ { - i } } { 5 L D } \ : | \ : i \in \mathbb { Z } _ { \geq 0 } \} } \end{array}$ (since $\begin{array} { r } { p _ { i } = \frac { 1 } { i + 1 } - \frac { 1 } { i + 2 } } \end{array}$ gives $\textstyle \sum _ { i > 0 } p _ { i } = 1$ , no normalizing constant is needed); (ii) the η<sub>i</sub>-expert is created from the $( 4 ^ { i - 1 } + 1 ) - \mathrm { s t }$ update on (since $\begin{array} { r } { \lceil \frac { 1 } { 2 } \log _ { 2 } k \rceil \geq i \iff k \geq 4 ^ { i - 1 } + 1 } \end{array}$ , the grid created so far always coincides with the grid of Algorithm 5 with ${ \bar { K } } = k )$ ; (iii) the weights are kept unnormalized (the output of the master is determined by the ratios of the weights alone). The freezing property of Definition 4.1 is preserved: at a round where no mistake occurs, neither the grid, nor the weights, nor any expert changes at all.

The key to the analysis is the reduction that regards a not-yet-created expert as a “virtual expert that outputs the point of the master” (a reduction to sleeping experts): its surrogate loss is identically 0, so the potential inequality of exponential-weight aggregation holds as it is, and the price of the delay in creation is limited to the additional term $\begin{array} { r } { H 4 ^ { i - 1 } = \frac { 1 } { 1 0 0 H \eta _ { i } ^ { 2 } } } \end{array}$ coming from the rounds before the creation of the grid point $\eta _ { i }$ used for comparison. This additional term is of a size that can be absorbed by self-bounding.

Algorithm 6 Growing-grid SGS-MetaGrad   
Require: geometric constants $L , D$ (Equation (3.2))   
1: $\begin{array} { r } { \bar { k }  1 , \bar { I }  0 , \eta _ { 0 }  \frac { 1 } { 5 L D } , } \end{array}$ , unnormalized weight $\tilde { p } ^ { \eta _ { 0 } } \gets \frac { 1 } { 2 }$ , an initial point $\theta ^ { \eta _ { 0 } } \in \Theta$   
of the η<sub>0</sub>-expert (Definition B.3), $\hat { \theta } ^ { 1 }  \theta ^ { \eta _ { 0 } }$   
2: for $t = 1 , \dots , T$ do   
3: receive $s ^ { t } ,$ compute and present with the oracle the proposal $\hat { x } ^ { t } \in$   
arg $\mathrm { n a x } _ { x \in Y ( s ^ { t } ) } \langle \hat { \theta } ^ { t } , x \rangle$ (Equation (3.3)), and observe $x ^ { t }$   
4: if $\hat { x } ^ { t } \neq x ^ { t }$ (a mistake) then   
5: $g ^ { t } \gets \hat { x } ^ { t } - x ^ { t } , \ell _ { k } ^ { \eta } ( \theta ) : = - \eta \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle + \eta ^ { 2 } \langle \hat { \theta } ^ { t } - \theta , g ^ { t } \rangle ^ { 2 } \left( \eta \in \{ \eta _ { 0 } , \dots , \eta _ { I } \} \right)$   
6: for each $\eta \in \{ \eta _ { 0 } , \dotsc , \eta _ { I } \} \colon \tilde { p } ^ { \eta }  \tilde { p } ^ { \eta } \exp ( - \ell _ { k } ^ { \eta } ( \theta ^ { \eta } ) )$ , and update $\theta ^ { \eta }$ by the   
ONS update of the η-expert on $\ell _ { k } ^ { \eta }$   
7: $k \gets k + 1$   
8: if $\left\lceil { \frac { 1 } { 2 } } \log _ { 2 } k \right\rceil > I$ then   
9: $\begin{array} { r } { I  I + 1 , \eta _ { I }  \frac { 2 ^ { - I } } { 5 L D } , \tilde { p } ^ { \eta I }  \frac { 1 } { ( I + 1 ) ( I + 2 ) } } \end{array}$ , prepare an initial point $\theta ^ { \eta _ { I } } \in \Theta$   
of the $\eta _ { I } .$ -expert (arbitrary; for instance ${ \hat { \theta } } ^ { t } )$   
10: end if   
11: $\begin{array} { r } { \hat { \theta } ^ { t + 1 } \gets \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } ^ { \eta _ { i } } \theta ^ { \eta _ { i } } / \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } ^ { \eta _ { i } } } \end{array}$   
12: else   
13: $\hat { \theta } ^ { t + 1 } \gets \hat { \theta } ^ { t }$   
14: end if   
15: end for

Remark B.5 (The place of the growing-grid version). Unlike Algorithm 5, Algorithm 6 does not coincide with the plain anytime version of MetaGrad (van Erven et al., 2021). The mechanism of growing the grid during execution is itself of the same kind, but this paper refines the grid by the number of mistakes k rather than by the round $t { : }$ if it were refined by the round $t ,$ a new expert would be created even at rounds without a mistake and the internal state would change, so the freezing property of Definition 4.1 would break, and the grid size would swell to ${ \cal O } ( \log T )$ leaving $c _ { 0 } = O ( \log \log T )$ . Moreover, since an expert created earlier receives updates and weightings at the subsequent mistake rounds, the sequences of outputs of the master themselves generally difer between refinement by t and refinement by $k .$ Therefore the growing-grid version of this subsection is not an “improvement of the analysis of an existing algorithm” in the sense of Remark B.4, but falls under the case where SGS actually changes the algorithm (Remark 4.2). The point of Proposition F.1 and Theorem F.3 lies in this refinement by k and in making its constants explicit. Compared with Theorem E.1, the argument of $c _ { 0 }$ changes from $\bar { K }$ to the realized value $\bar { K }$ , which makes the guarantee completely independent of $T ,$ while the coeficient of the logarithmic term changes hardly at all, from $\frac { 1 5 2 } { 3 } \approx 5 0 . 7$ to 52. The computational cost per mistake round is proportional to the number $1 + \textstyle { \left\lceil { \frac { 1 } { 2 } } \log _ { 2 } K \right\rceil }$ of created experts, which is also independent of $T$ (Table 5). The parameters of the algorithm are only L and D. To dispense even with the knowledge of $L _ { : }$ , the following Lipschitz-adaptive version is needed.

Remark B.6 (Parameter adaptation by the refined version of MetaGrad). The Lipschitz-adaptive anytime version of MetaGrad of van Erven et al. (2021, Algorithms 1 and 2) requires no prior knowledge of G, H or of the number of rounds and operates using only (a guess of) the value of $W ;$ according to Sakaue et al. (2025b, Appendix C.4) it attains, in the setting of Proposition E.3,

$$
\sum _ { j = 1 } ^ { m } \langle w _ { j } - u , g _ { j } \rangle = O \left( { \sqrt { n \log \left( { \frac { W G m } { n } } \right) \cdot V _ { m } ^ { u } } } + H n \log \left( { \frac { W G m } { n } } \right) \right)
$$

(note that the argument of this logarithm is not scale invariant; this comes from the normalization of the source). Applying this to the subsequence of mistake rounds, the self-bounding and the application of the transcendental inequality (Lemma D.3) in the proof of Theorem E.1 go through as they are, and, allowing constants in $O ( \cdot )$ form, Theorem F.3 holds in the same order up to the argument of the logarithm changing from $K / d$ to $D L K / d$ (an addition of the order of $\log ( 1 + L D ) \rangle$ ). In that case the only prior knowledge needed is $D _ { ; }$ , and $L$ and $\bar { K }$ are unnecessary. Since Θ is a set designed by the learner itself, $D = \mathrm { d i a m } \Theta$ is known, so the only substantially unknown parameter was $L .$ . The explicit constants are not tracked, since the upper bound of the source is in $O ( \cdot )$ form.

Finally, Table 5 summarizes the per-round computational cost of each method. Here $\tau _ { \mathrm { s o l v e } }$ is the time for one linear optimization that computes the proposal $\hat { x } ^ { t }$ $\tau _ { \mathrm { E - p r o j } } ~ / ~ \tau _ { \mathrm { G - p r o j } }$ is the time for one Euclidean $/$ generalized projection onto Θ, and K is the number of mistakes (the value for growing-grid SGS-MetaGrad is that of Algorithm 6). The column Θ records the weight space assumed in each reference; “any” means an arbitrary nonempty bounded closed convex set (Assumption 3.1(1)), so that the probability simplex is admissible.

The upper part is based on Sakaue et al. (2025b, Table 1): Besbes et al. (2021, 2025) and Gollapudi et al. (2021) only claim that the total computational cost is poly $( d , T )$ , and the scrutiny of Sakaue et al. (2025b) estimates the per-round cost of Gollapudi et al. (2021) to be at least $O ( \tau _ { \mathrm { s o l v e } } + d ^ { 5 } T ^ { 3 } )$ . CoRectron (Sakaue, 2026) (marked <sup>∗</sup> in the table) imposes no constraint on the iterate $\hat { \theta } ^ { t }$ ; it applies to $\Theta = \mathbb { R } ^ { d }$ rather than to a general Θ, and is listed for reference. The values in the lower part are the costs at mistake rounds; rounds without a mistake need only the computation of the proposal $\left( O ( \tau _ { \mathrm { s o l v e } } ) \right)$ ). This is the origin of the dependence on K in the total computational cost of Table 1.

B.5. Comparison of the performance criteria with existing methods. The existing work on online inverse linear optimization all takes the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ Equation (3.6) as the main object of evaluation, calling it the “regret”, and does not necessarily claim an upper bound on the cumulative suboptimality regret $R _ { T } ^ { \mathrm { { s u b } } }$ Equation (3.5). While the two are bounded simultaneously by the sum $\widetilde { R } _ { T }$ as in Equation (3.8), an upper bound on one of the components does not imply an upper bound on the other. We therefore organize in Table 6 which reference bounds which criterion.

The breakdown is as follows. Bärmann et al. (2018) takes as its direct object the total error $\textstyle \sum _ { t } \langle { \hat { \theta } } ^ { t } - \theta ^ { * } , { \hat { x } } ^ { t } - x ^ { t } \rangle$ , which amounts to the sum $\widetilde { R } _ { T }$ , makes explicit that it decomposes into the sum of the objective-function error $R _ { T } ^ { \mathrm { { s u b } } }$ and the solution error $R _ { T } ^ { \mathrm { { e s t } } }$ (both components being nonnegative), and then shows $O ( \sqrt { T } )$ . Sakaue et al. (2025b, Theorem 3.1) (ONS) and Sakaue et al. (2025b, Theorem 4.1) (MetaGrad)

Table 5. Comparison of the per-round computational cost (for the total cost see Table 1). The values in the upper part are the costs at every round, whereas those in the lower part are the costs at mistake rounds; a round without a mistake costs only ${ \cal O } ( \tau _ { \mathrm { s o l v e } } )$
<table><tr><td></td><td>Θ</td><td>Per-round computational cost</td></tr><tr><td>Bärmann et al. (2018)</td><td>any</td><td> $O ( \tau _ { \mathrm { s o l v e } } + \tau _ { \mathrm { E - p r o j } } + d )$ </td></tr><tr><td>Besbes et al. (2021, 2025)</td><td> $\| \theta \| = 1$ </td><td>not claimed</td></tr><tr><td>Gollapudi et al. (2021)</td><td> $B ^ { d }$ </td><td>not claimed</td></tr><tr><td>ONS (Sakaue et al., 2025b)</td><td>any</td><td> $O ( \tau _ { \mathrm { s o l v e } } + d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } )$ </td></tr><tr><td>MetaGrad (Sakaue et al., 2025b) any</td><td></td><td> $O ( \tau _ { \mathrm { s o l v e } } + ( d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } ) \log T )$ </td></tr><tr><td>CoRectron* (Sakaue, 2026)</td><td> $\mathbb { R } ^ { i }$ </td><td> $O ( \tau _ { \mathrm { s o l v e } } + d ^ { 2 } )$ </td></tr><tr><td>SGS-OGD</td><td>any</td><td> $O ( \tau _ { \mathrm { s o l v e } } + d + \tau _ { \mathrm { E - p r o j } } )$ </td></tr><tr><td>ONS</td><td>any</td><td> $O ( \tau _ { \mathrm { s o l v e } } + d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } )$ </td></tr><tr><td>Growing-grid SGS-MetaGrad</td><td>any</td><td> $O ( \tau _ { \mathrm { s o l v e } } + ( d ^ { 2 } + \tau _ { \mathrm { G - p r o j } } ) \log K )$ </td></tr></table>

likewise give upper bounds on $\widetilde { R } _ { T } ^ { c ^ { * } }$ (the $\widetilde { R } _ { T }$ of this paper), and both bound the two criteria simultaneously. By contrast, the logarithmic regret of Besbes et al. (2021, 2025) and Gollapudi et al. (2021), and the T-independent regret of Oki and Sakaue (2026), are claims about $R _ { T } ^ { \mathrm { { e s t } } }$ , and no upper bound on $R _ { T } ^ { \mathrm { { s u b } } }$ is claimed. The three methods of this paper bound the sum $\widetilde { R } _ { T }$ independently of $T$ under a uniform margin (Assumption 3.1), and hence bound the two criteria simultaneously and independently of the total number of rounds.

We add a few words on how to read the table. The upper part lists existing methods that do not assume a uniform margin and the lower part the proposed methods of this paper (under Assumption 3.1); “not claimed” indicates that the reference in question does not claim an upper bound on that criterion. The values marked with <sup>‡</sup> come from an upper bound on the sum $\widetilde { R } _ { T }$ and bound the two criteria simultaneously, and the mark <sup>†</sup> indicates that the value holds only when the feasible set is M-convex. The column Θ records the weight space assumed in each reference; “any” means an arbitrary nonempty bounded closed convex set (Assumption 3.1(1)), so that the probability simplex is admissible. Here $L , D$ are the constants in Equation (3.2).

The entries come from the following sources. Those of Bärmann et al. (2018) and Sakaue et al. (2025b) rewrite the upper bounds of the original papers in the notation of this paper (for the former, $\scriptstyle { \frac { 3 } { 2 } } L D { \sqrt { T } }$ ; for the latter, the upper bound on the per-round linearized regret is evaluated as $L D$ and the upper bound on the norm of a subgradient as L). The entry of Sakaue et al. (2025a) (marked <sup>§</sup>) is a result under the gap condition $\Delta > 0$ rather than a uniform margin ((Sakaue et al., 2025a, Theorem 5.2); for the definition of $\Delta$ and the derivation see Appendix O), and its constant depends on the regularizer of the FTRL and on the sizes of Θ and $X ( s )$ The entry of CoRectron (marked <sup>∗</sup>) is listed for reference, since that method imposes no constraint on the iterate $\hat { \theta } ^ { t }$ and applies to $\Theta = \mathbb { R } ^ { d }$ rather than to a general Θ; its value is Sakaue (2026, Theorem 3.1) with its weight space specialized to $\mathbb { R } ^ { d }$ with the range of $\langle { \theta ^ { * } , \cdot } \rangle$ on each $X ( s )$ bounded by $L \| \theta ^ { * } \|$ and its regularization parameter treated as a constant. The entries of Besbes et al. (2021, 2025), Gollapudi et al. (2021) and Oki and Sakaue (2026) are the values of the original papers, whose normalizations difer from one another. The two values in the entry of Gollapudi et al. (2021) correspond to two diferent algorithms of that paper: $O ( d \log T )$ is Gollapudi et al. (2021, Theorem 4.4) and $\exp ( O ( d \log d ) )$ is Gollapudi et al. (2021, Theorem 4.2) (both through the reduction of Gollapudi et al. (2021, Theorem 3.1)), and the latter does not depend on the total number of rounds T. The entries for the proposed methods restate Table 2.

Table 6. Comparison of the performance criteria under a general margin γ.
<table><tr><td>Method</td><td>Θ</td><td> $R _ { T } ^ { \mathrm { { s u b } } }$ </td><td> $R _ { T } ^ { \mathrm { { e s t } } }$ </td></tr><tr><td>OGD (Bärmann et al., 2018)</td><td>any</td><td> $O ( L D \sqrt { T } ) ^ { \ddagger }$ </td><td> $O ( L D \sqrt { T } ) ^ { \ddagger }$ </td></tr><tr><td>Besbes et al. (2021, 2025)</td><td></td><td>∥|θ∥ = 1 not claimed</td><td> $O ( d ^ { 4 } \log T )$ </td></tr><tr><td>Gollapudi et al. (2021)</td><td> $B ^ { d }$ </td><td>not claimed</td><td> $O ( d \log T ) , \exp ( O ( d \log d ) )$ </td></tr><tr><td>Sakaue et al. (2025a)§</td><td>any</td><td> ${ \cal O } ( 1 / \Delta ^ { 2 } ) ^ { \ddagger }$ </td><td> ${ \cal O } ( 1 / \Delta ^ { 2 } ) ^ { \ddagger }$ </td></tr><tr><td>ONS, MetaGrad (Sakaue et al., 2025b)</td><td>any</td><td> $O ( L D d \log { \frac { T } { d } } ) ^ { \ddagger }$ </td><td> $O ( L D d \log { \frac { T } { d } } ) ^ { \ddagger }$ </td></tr><tr><td>CoRectron* (Sakaue, 2026)</td><td> $\mathbb { R } ^ { d }$ </td><td>not claimed</td><td> $O ( L \| \theta ^ { * } \| d \log T )$ </td></tr><tr><td>Center-of-gravity method† (Oki and Sakaue, 2026)</td><td> $\mathbb { R } ^ { d }$ </td><td>not claimed</td><td>O(d log d)</td></tr><tr><td>SGS-OGD (Theorem C.2)</td><td>any</td><td> $\frac { L ^ { 2 } D ^ { 2 } } { 2 \gamma }$ </td><td> $\frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma } ^ { \ddagger }$ </td></tr><tr><td>ONS (Theorem D.6)</td><td>any</td><td> $\begin{array} { r } { L D ( 1 + d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 1 ) ) } \end{array}$ </td><td> $\begin{array} { r } { L D ( 1 + 2 d \log ( 2 + \frac { 2 L D } { \gamma } ) ) ^ { \ddagger } } \end{array}$ </td></tr><tr><td>Growing-grid SGS-MetaGrad (Theorem F.3)</td><td>any</td><td> $O ( L D d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 1 ) )$ </td><td> $\begin{array} { r } { O ( L D d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 2 ) ) ^ { \ddag } } \end{array}$ </td></tr></table>

## Appendix C. Analysis of SGS-OGD (proof of Theorem C.2)

In the analyses from this appendix on, we reindex by mistake rounds. We write the mistake rounds, in order of occurrence, as $t _ { 1 } < t _ { 2 } < \dots < t _ { K }$ (with the convention $t _ { K + 1 } : = T + 1 )$ . Since $\hat { \theta } ^ { t + 1 } = \hat { \theta } ^ { t }$ at rounds where no mistake occurs, the iteration can be described by the sequence $\hat { \theta } ^ { t _ { 1 } } , \ldots , \hat { \theta } ^ { t _ { K } }$ of mistake rounds alone, and at the k-th mistake round the subgradient is $g ^ { t _ { k } } = \hat { x } ^ { t _ { k } } - x ^ { t _ { k } }$ and the loss is $\ell ^ { t _ { k } } : = \ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t _ { k } } , s ^ { t _ { k } } )$ . We further use the quantity

$$
r _ { t _ { k } } : = \langle \hat { \theta } ^ { t _ { k } } - \bar { \theta } , g ^ { t _ { k } } \rangle\tag{C.1}
$$

relative to $\bar { \theta }$ (Assumption 3.1(4)). The following lemma is the core of the analysis, common to the first-order and second-order methods, and states that the uniform margin Equation (3.9) makes $r _ { t _ { k } }$ exceed $\ell ^ { t _ { k } }$ by at least $\gamma$ at every mistake round. Moreover, in the analysis of the regret of the sum we use the quantity

$$
\widetilde { r } _ { t _ { k } } : = \langle \hat { \theta } ^ { t _ { k } } - \theta ^ { * } , g ^ { t _ { k } } \rangle\tag{C.2}
$$

obtained by replacing $\bar { \theta }$ with $\theta ^ { * }$ . Since at rounds without a mistake the summand is 0 because $\hat { x } ^ { t } = x ^ { t }$ , we have $\begin{array} { r } { \widetilde { R } _ { T } = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } } \end{array}$ , and by the nonnegativity of both components of Equation (3.7) and the Cauchy–Schwarz inequality $( \hat { \theta } ^ { t _ { k } } , \theta ^ { * } \in \Theta$ Equation (3.2)),

$$
0 \leq \widetilde { r } _ { t _ { k } } \leq \Vert  { \hat { \theta } } ^ { t _ { k } } - \theta ^ { * } \Vert \Vert g ^ { t _ { k } } \Vert \leq L D\tag{C.3}
$$

holds. This is the counterpart of $0 \le r _ { t _ { k } } \le L D$ in Lemma C.1 with $\bar { \theta }$ replaced by $\theta ^ { * }$ , the only diference being that it does not have the lower bound $\gamma$ coming from the margin.

Lemma C.1 (Lower and upper bounds on $r _ { t _ { k } } )$ . Under Assumption 3.1, for every mistake round $t _ { k }$

$$
\ell ^ { t _ { k } } + \gamma \leq r _ { t _ { k } } \leq L D .\tag{C.4}
$$

In particular $0 < \gamma \leq r _ { t _ { k } }$

Proof of Lemma $C . 1 .$ Lower bound: from $\hat { x } ^ { t _ { k } } \in \arg \operatorname* { m a x } _ { x \in Y ( s ^ { t _ { k } } ) } \langle \hat { \theta } ^ { t _ { k } } , x \rangle$ we have $\begin{array} { r } { \langle \hat { \theta } ^ { t _ { k } } , g ^ { t _ { k } } \rangle = \operatorname* { m a x } _ { x } \langle \hat { \theta } ^ { t _ { k } } , x \rangle - \langle \hat { \theta } ^ { t _ { k } } , x ^ { t _ { k } } \rangle = \ell ^ { t _ { k } } } \end{array}$ , and at a mistake round we have $\hat { x } ^ { t _ { k } } \neq x ^ { t _ { k } }$ and $\hat { x } ^ { t _ { k } } \in Y ( s ^ { t _ { k } } )$ , so Equation $\left( 3 . 9 \right)$ gives $- \langle \bar { \theta } , g ^ { t _ { k } } \rangle = \langle \bar { \theta } , x ^ { t _ { k } } - \hat { x } ^ { t _ { k } } \rangle \geq \gamma$ . Adding the two, we obtain $r _ { t _ { k } } = \langle \hat { \theta } ^ { t _ { k } } , g ^ { t _ { k } } \rangle - \langle \bar { \theta } , g ^ { t _ { k } } \rangle \geq \ell ^ { t _ { k } } + \gamma$ . Upper bound: by the Cauchy–Schwarz inequality together with $\| g ^ { t _ { k } } \| \leq L$ and $\lVert \hat { { \boldsymbol { \theta } } } ^ { t _ { k } } - \bar { { \boldsymbol { \theta } } } \rVert \leq D$ (both from Equation (3.2)). □

Theorem C.2. Under Assumption 3.1, run SGS-OGD (Algorithm 1) on an arbitrary sequence of states $\{ s ^ { t } \} _ { t = 1 } ^ { T }$ , and set $C _ { \alpha } : = D ^ { 2 } / ( 2 \alpha ) + L ^ { 2 } \alpha$ . Then, for every $T _ { \cdot }$ , the following hold.

(i): (K)

$$
K \leq \frac { C _ { \alpha } ^ { 2 } } { \gamma ^ { 2 } } .\tag{C.5}
$$

(ii): $( R _ { T } ^ { \mathrm { s u b } } )$

$$
R _ { T } ^ { \mathrm { { s u b } } } \leq \frac { C _ { \alpha } ^ { 2 } } { 4 \gamma } .\tag{C.6}
$$

(iii): $( \widetilde { R } _ { T } )$

$$
\widetilde { R } _ { T } \leq \frac { C _ { \alpha } ^ { 2 } } { \gamma } .\tag{C.7}
$$

In particular, choosing $\alpha = D / ( L \sqrt { 2 } )$ gives $C _ { \alpha } ^ { 2 } = 2 L ^ { 2 } D ^ { 2 }$ and

$$
K \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma ^ { 2 } } , \qquad R _ { T } ^ { \mathrm { s u b } } \leq \frac { L ^ { 2 } D ^ { 2 } } { 2 \gamma } , \qquad \widetilde R _ { T } \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma } .
$$

By Equation (3.8), the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ also has the same upper bound as Equation (C.7).

Proof of Theorem C.2. Step 1. (Reindexing by SGS) At rounds where no mistake occurs we have $\hat { x } ^ { t } = x ^ { t }$ , so $\ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t } , s ^ { t } ) = \langle \hat { \theta } ^ { t } , \hat { x } ^ { t } - x ^ { t } \rangle = 0$ , and the update rule gives $\hat { \theta } ^ { t + 1 } = \hat { \theta } ^ { t }$ . Hence rounds without a mistake contribute neither to the cumulative suboptimality regret nor to the trajectory of $\theta ,$ and

$$
R _ { T } ^ { \mathrm { s u b } } = \sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } , \qquad \hat { \theta } ^ { t _ { k + 1 } } = \Pi _ { \Theta } \big ( \hat { \theta } ^ { t _ { k } } - \eta _ { k } g ^ { t _ { k } } \big ) , \quad \eta _ { k } : = \alpha k ^ { - 1 / 2 }
$$

(the reindexing of $\ S 4 )$ . Moreover, Equation (3.2) gives $\left\| g ^ { t _ { k } } \right\| = \left\| \hat { x } ^ { t _ { k } } - x ^ { t _ { k } } \right\| \leq L$

Step 2. (Potential estimate for an arbitrary $u \in \Theta ;$ one step) Below let $u \in \Theta$ be an arbitrary point and set $a _ { k } ( u ) : = \| \hat { \theta } ^ { t _ { k } } - u \| ^ { 2 }$ . By the nonexpansiveness of the Euclidean projection $( u \in \Theta )$ and the update formula of Step 1,

$$
\begin{array} { r l } & { a _ { k + 1 } ( u ) \leq \| \hat { \theta } ^ { t _ { k } } - \eta _ { k } g ^ { t _ { k } } - u \| ^ { 2 } = a _ { k } ( u ) - 2 \eta _ { k } \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle + \eta _ { k } ^ { 2 } \| g ^ { t _ { k } } \| ^ { 2 } } \\ & { \qquad \leq a _ { k } ( u ) - 2 \eta _ { k } \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle + \eta _ { k } ^ { 2 } L ^ { 2 } . } \end{array}
$$

Dividing both sides by $2 \eta _ { k }$ and rearranging,

$$
\langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \le \frac { 1 } { 2 \eta _ { k } } \left( a _ { k } ( u ) - a _ { k + 1 } ( u ) \right) + \frac { \eta _ { k } L ^ { 2 } } { 2 } .
$$

Step 3. (Estimate by Abel summation) Summing over $k = 1 , \ldots , K$ and setting $\zeta _ { k } : = 1 / ( 2 \eta _ { k } )$ 2

$$
\sum _ { k = 1 } ^ { K } \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \leq \sum _ { k = 1 } ^ { K } \zeta _ { k } ( a _ { k } ( u ) - a _ { k + 1 } ( u ) ) + \frac { L ^ { 2 } } { 2 } \sum _ { k = 1 } ^ { K } \eta _ { k } .
$$

By Abel summation,

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 1 } ^ { K } \zeta _ { k } ( a _ { k } ( u ) - a _ { k + 1 } ( u ) ) = \zeta _ { 1 } a _ { 1 } ( u ) + \displaystyle \sum _ { k = 2 } ^ { K } ( \zeta _ { k } - \zeta _ { k - 1 } ) a _ { k } ( u ) - \zeta _ { K } a _ { K + 1 } ( u ) } & { } \\ { \le \zeta _ { 1 } a _ { 1 } ( u ) + \displaystyle \sum _ { k = 2 } ^ { K } ( \zeta _ { k } - \zeta _ { k - 1 } ) a _ { k } ( u ) . } & { } \end{array}
$$

Since $\eta _ { k } = \alpha k ^ { - 1 / 2 }$ is monotonically decreasing in $k ,$ the sequence $\zeta _ { k } = k ^ { 1 / 2 } / ( 2 \alpha )$ is monotonically increasing and $\zeta _ { k } - \zeta _ { k - 1 } \geq 0$ . Moreover, $\hat { \theta } ^ { t _ { k } } , u \in \Theta$ and $D = \dim ( \Theta )$ give $a _ { k } ( u ) \leq D ^ { 2 }$ , so

$$
\zeta _ { 1 } a _ { 1 } ( u ) + \sum _ { k = 2 } ^ { K } ( \zeta _ { k } - \zeta _ { k - 1 } ) a _ { k } ( u ) \leq D ^ { 2 } \left( \zeta _ { 1 } + \sum _ { k = 2 } ^ { K } ( \zeta _ { k } - \zeta _ { k - 1 } ) \right) = D ^ { 2 } \zeta _ { K } = \frac { D ^ { 2 } \sqrt { K } } { 2 \alpha } .
$$

On the other hand, comparison with an integral gives $\begin{array} { r } { \sum _ { k = 1 } ^ { K } k ^ { - 1 / 2 } \leq \int _ { 0 } ^ { K } x ^ { - 1 / 2 } \mathrm { d } x = } \end{array}$ $2 \sqrt { K }$ , whence $\begin{array} { r } { \frac { L ^ { 2 } } { 2 } \sum _ { k = 1 } ^ { K } \eta _ { k } \leq L ^ { 2 } \alpha \sqrt { K } } \end{array}$ . Combining the above, we obtain, for every $u \in \Theta$

$$
\sum _ { k = 1 } ^ { K } \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \leq \left( \frac { D ^ { 2 } } { 2 \alpha } + L ^ { 2 } \alpha \right) \sqrt { K } = C _ { \alpha } \sqrt { K } .\tag{C.8}
$$

The only property of u used here is that $u \in \Theta$ (through the nonexpansiveness of the projection and $a _ { k } ( u ) \leq D ^ { 2 } )$

Step 4. ((i) and $( \operatorname { i i } ) \colon u = { \bar { \theta } } )$ Since $\bar { \theta } \in \Theta$ by Assumption 3.1(4), Equation (C.8) can be used with $u = \bar { \theta }$ , and each term on the left-hand side is $r _ { t _ { k } }$ . Substituting the lower bound $r _ { t _ { k } } \geq \ell ^ { t _ { k } } + \gamma$ of Lemma C.1, we obtain

$$
\sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } + \gamma K \leq C _ { \alpha } \sqrt { K } .\tag{C.9}
$$

From $\ell ^ { t _ { k } } \geq 0$ and Equation (C.9) we get $\gamma K \leq C _ { \alpha } \sqrt { K }$ , that is, $K \leq C _ { \alpha } ^ { 2 } / \gamma ^ { 2 }$ , which is (i). Moreover, Equation (C.9) gives $\begin{array} { r } { \sum _ { k } \ell ^ { t _ { k } } \leq C _ { \alpha } \sqrt { K } - \gamma K \leq } \end{array}$ max ${ \ L } _ { \mathrm { { s } \geq 0 } } ( C _ { \alpha } \sqrt { x } -$ $\gamma x ) = C _ { \alpha } ^ { 2 } / ( 4 \gamma )$ , which is (ii) (the right-hand side is maximized at $x ^ { * } = \overset { - } { C _ { \alpha } ^ { 2 } } / ( 4 \gamma ^ { 2 } ) )$ . Moreover, since $\hat { \theta } ^ { t + 1 } = \hat { \theta } ^ { t }$ at rounds without a mistake, the distinct iterates are only $\hat { \theta } ^ { 1 }$ and the points immediately after each mistake round, so the total number of iterates satisfies $| \{ \hat { \theta } ^ { t } \mid t = 1 , \dots , T \} | \leq K + 1$

Step 5. $( ( \mathrm { i i i } ) \colon u = \theta ^ { * } )$ Since $\theta ^ { \ast } \in \Theta$ by Assumption 3.1(3), Equation (C.8) can be used with $u \ = \ \theta ^ { * }$ as well, and each term on the left-hand side is $\widetilde { r } _ { t _ { k } }$ (Equation (C.2)), so $\begin{array} { r } { \widetilde { R } _ { T } = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } \le C _ { \alpha } \sqrt { K } } \end{array}$ . Since $\theta ^ { * }$ need not satisfy the margin Equation (3.9), the lower bound on $r _ { t _ { k } }$ cannot be substituted, unlike in Step 4. Instead, using the monotonicity of $\sqrt { \cdot }$ and the bound $K \leq C _ { \alpha } ^ { 2 } / \gamma ^ { 2 }$ of (i), we obtain Equation (C.7). In particular, when $\alpha = D / ( L \sqrt { 2 } )$ we have $\begin{array} { r } { C _ { \alpha } = D ^ { 2 } \cdot \frac { L \sqrt { 2 } } { 2 D } + L ^ { 2 } \cdot \frac { D } { L \sqrt { 2 } } = \frac { L D } { \sqrt { 2 } } + \frac { L D } { \sqrt { 2 } } = \sqrt { 2 } L D , \mathrm { s o } C _ { \alpha } ^ { 2 } = 2 L ^ { 2 } D ^ { 2 } } \end{array}$ □

## Appendix D. Analysis of ONS (proof of Theorem D.6)

Lemma D.1 (cf. Orabona, 2019, Proposition 2.11). Let $\Sigma \succ 0 .$ , let Θ be a nonempty closed convex set and let $u \in \Theta$ . Then, for every $y \in \mathbb { R } ^ { d } , \| \Pi _ { \Theta } ^ { \Sigma } ( y ) - u \| _ { \Sigma } \leq \| y - u \| _ { \Sigma }$ Lemma D.2 (cf. Hazan, 2019, §4). Let $\Sigma _ { 0 } : = D ^ { - 2 } { \mathrm { I d } } _ { d }$ , let $\nabla _ { t _ { k } } \in \mathbb { R } ^ { d }$ with $\| \nabla _ { t _ { k } } \| \leq$ $1 / D \ ( k = 1 , \ldots , K )$ , and let $\Sigma _ { k } : = \Sigma _ { k - 1 } + \nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top }$ . Then

$$
\sum _ { k = 1 } ^ { K } \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \leq \log \frac { \operatorname* { d e t } \Sigma _ { K } } { \operatorname* { d e t } \Sigma _ { 0 } } \leq d \log \left( 1 + \frac { K } { d } \right) .\tag{D.1}
$$

Proof. The first inequality: for each k we have $\Sigma _ { k } \succeq \Sigma _ { 0 } + \nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top } = D ^ { - 2 } I + \nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top }$ 2 so by the order reversal of the inverse and the Sherman–Morrison formula,

$$
\nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \leq \nabla _ { t _ { k } } ^ { \top } \left( D ^ { - 2 } I + \nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top } \right) ^ { - 1 } \nabla _ { t _ { k } } = \frac { \| \nabla _ { t _ { k } } \| ^ { 2 } } { D ^ { - 2 } + \| \nabla _ { t _ { k } } \| ^ { 2 } } < 1 .
$$

By the matrix determinant lemma,

$$
\begin{array} { r } { \operatorname* { d e t } \Sigma _ { k - 1 } = \operatorname* { d e t } \bigl ( \Sigma _ { k } - \nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top } \bigr ) = \operatorname* { d e t } \Sigma _ { k } \left( 1 - \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \right) , } \end{array}
$$

and applying $u \leq - \log ( 1 - u ) { \mathrm { ~ ( f o r ~ } } u < 1 )$ with $\boldsymbol { u } = \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } }$ gives $\nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \leq$ $\log { \frac { \operatorname* { d e t } \Sigma _ { k } } { \operatorname* { d e t } \Sigma _ { k - 1 } } }$ . Summing over $k = 1 , \ldots , K$ , the terms log det $\Sigma _ { k }$ of adjacent summands cancel on the right-hand side, so the sum equals log det $\Sigma _ { K } - \log$ det $\Sigma _ { 0 }$ , and we obtain the first inequality.

The second inequality: writing the eigenvalues of $\Sigma _ { K }$ as $\lambda _ { 1 } , . . . , \lambda _ { d } > 0$ , the arithmetic–geometric mean inequality gives det $\begin{array} { r } { \Sigma _ { K } = \prod _ { i } \lambda _ { i } \le ( \mathrm { t r } \Sigma _ { K } / d ) ^ { d } } \end{array}$ , and since tr $\begin{array} { r } { \Sigma _ { K } = d \bar { D } ^ { - 2 } + \sum _ { k } \| \nabla _ { t _ { k } } \| ^ { 2 } \leq d { D } ^ { - 2 } + K { D } ^ { - 2 } } \end{array}$

$$
\log \frac { \operatorname * { d e t } \Sigma _ { K } } { \operatorname * { d e t } \Sigma _ { 0 } } \leq d \log \frac { ( d + K ) D ^ { - 2 } / d } { D ^ { - 2 } } = d \log \left( 1 + \frac K d \right) .
$$

Lemma D.3. Let $c _ { 1 } , c _ { 2 } > 0$ and suppose that $y \geq 0$ satisfies $y \leq c _ { 1 } + c _ { 2 } \log ( 1 + y )$ Then

$$
y \leq 2 c _ { 1 } + 1 + 2 c _ { 2 } \log \operatorname* { m a x } ( 2 c _ { 2 } , 1 ) .
$$

Proof. Set $\mu : = \operatorname* { m a x } ( 2 c _ { 2 } , 1 ) > 0$ . By the concavity of log, the tangent-line inequality log $\begin{array} { r } { w \leq \log \mu + \frac { w } { \mu } - 1 } \end{array}$ holds for every $w > 0$ , so taking $w = 1 + y$

$$
y \leq c _ { 1 } + c _ { 2 } \left( \log \mu + \frac { 1 + y } { \mu } - 1 \right) \leq c _ { 1 } + c _ { 2 } \log \mu + \frac { 1 + y } { 2 } - c _ { 2 } ,
$$

where we used $c _ { 2 } / \mu \leq 1 / 2$ . Rearranging, $\begin{array} { r } { \frac { y } { 2 } \leq c _ { 1 } + \frac { 1 } { 2 } + c _ { 2 } \log \mu - c _ { 2 } \leq c _ { 1 } + \frac { 1 } { 2 } + c _ { 2 } \log \mu _ { \mathrm { : } } } \end{array}$ that is, $y \le 2 c _ { 1 } + 1 + 2 c _ { 2 } \log \mu$ □

Lemma D.4. Under Assumption 3.1, run Algorithm 3 on an arbitrary sequence of states. For the quantities $\eta = 1 / ( L D ) , \nabla _ { t _ { k } } = \eta g ^ { t _ { k } }$ and $\Sigma _ { k }$ (with $\Sigma _ { 0 } = D ^ { - 2 } \operatorname { I d } _ { d } )$ of Algorithm 3 and an arbitrary $u \in \Theta$ , setting $z _ { t _ { k } } : = \langle \nabla _ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle$ , we have

$$
\sum _ { k = 1 } ^ { K } z _ { t _ { k } } - \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } z _ { t _ { k } } ^ { 2 } \leq \frac { 1 } { 2 } \| \hat { { \boldsymbol \theta } } ^ { 1 } - { \boldsymbol u } \| _ { { \Sigma } _ { 0 } } ^ { 2 } + \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } .\tag{D.2}
$$

Proof. The update of Algorithm 3 is $\hat { \theta } ^ { t _ { k + 1 } } = \Pi _ { \Theta } ^ { \Sigma _ { k } } \bigl ( \hat { \theta } ^ { t _ { k } } - \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \bigr ) , \Sigma _ { k } = \Sigma _ { k - 1 } +$ $\nabla _ { t _ { k } } \nabla _ { t _ { k } } ^ { \top }$ <sup>⊤</sup><sub>t</sub> , and $\| \nabla _ { t _ { k } } \| = \| g ^ { t _ { k } } \| / ( L D ) \leq 1 / D$ (Equation (3.2)).

By Lemma D.1 (with $\Sigma = \Sigma _ { k }$ and $u \in \Theta )$

$$
\begin{array} { r l } & { \| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { \Sigma _ { k } } ^ { 2 } \leq \| \hat { \theta } ^ { t _ { k } } - \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } - u \| _ { \Sigma _ { k } } ^ { 2 } } \\ & { \qquad = \| \hat { \theta } ^ { t _ { k } } - u \| _ { \Sigma _ { k } } ^ { 2 } - 2 \langle \nabla _ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle + \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } , } \end{array}
$$

where the cross term is $2 \big ( \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } \big ) ^ { \top } \Sigma _ { k } \big ( \hat { \theta } ^ { t _ { k } } - u \big ) = 2 \big \langle \nabla _ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \big \rangle$ and the quadratic term is $( \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } } ) ^ { \top } \Sigma _ { k } ( \Sigma _ { k } ^ { - 1 } \ddot { \nabla } _ { t _ { k } } ) = \nabla _ { t _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { t _ { k } }$ . Rearranging,

$$
z _ { t _ { k } } \leq \frac { 1 } { 2 } \left( \| \hat { \theta } ^ { t _ { k } } - u \| _ { { \Sigma } _ { k } } ^ { 2 } - \| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { { \Sigma } _ { k } } ^ { 2 } \right) + \frac { 1 } { 2 } \nabla _ { { t } _ { k } } ^ { \top } \Sigma _ { k } ^ { - 1 } \nabla _ { { t } _ { k } } .\tag{D.3}
$$

We sum over $k = 1 , \ldots , K$ . For the sum of the first term, using $\| \hat { \theta } ^ { t _ { k } } - u \| _ { \Sigma _ { k } } ^ { 2 } =$ $\| \hat { \theta } ^ { t _ { k } } - u \| _ { \Sigma _ { k - 1 } } ^ { 2 } + z _ { t _ { k } } ^ { 2 }$ , which follows from $\boldsymbol { \Sigma } _ { k } = \boldsymbol { \Sigma } _ { k - 1 } + \boldsymbol { \nabla } _ { t _ { k } } \boldsymbol { \nabla } _ { t _ { k } } ^ { \top }$ , we obtain

$$
\sum _ { k = 1 } ^ { K } \Big ( \| \hat { \theta } ^ { t _ { k } } - u \| _ { \Sigma _ { k } } ^ { 2 } - \| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { \Sigma _ { k } } ^ { 2 } \Big ) = \sum _ { k = 1 } ^ { K } \Big ( \| \hat { \theta } ^ { t _ { k } } - u \| _ { \Sigma _ { k - 1 } } ^ { 2 } + z _ { t _ { k } } ^ { 2 } - \| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { \Sigma _ { k } } ^ { 2 } \Big )
$$

$$
= \| \hat { \theta } ^ { 1 } - u \| _ { \Sigma _ { 0 } } ^ { 2 } - \| \hat { \theta } ^ { t _ { K + 1 } } - u \| _ { \Sigma _ { K } } ^ { 2 } + \sum _ { k = 1 } ^ { K } z _ { t _ { k } } ^ { 2 }
$$

$$
\leq \| \hat { \theta } ^ { 1 } - u \| _ { \Sigma _ { 0 } } ^ { 2 } + \sum _ { k = 1 } ^ { K } z _ { t _ { k } } ^ { 2 } ,
$$

where the second equality holds because the term $- \| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { \Sigma _ { k } } ^ { 2 }$ of the k-th summand and the term $\| \hat { \theta } ^ { t _ { k + 1 } } - u \| _ { \Sigma _ { k } } ^ { 2 }$ of the $( k + 1 ) { \mathrm { - s t } }$ summand cancel, leaving only $\| \hat { \theta } ^ { 1 } - u \| _ { \Sigma _ { 0 } } ^ { 2 }$ from $k = 1$ and $- \| \hat { \theta } ^ { t _ { K + 1 } } - u \| _ { \Sigma _ { K } } ^ { 2 }$ from $k = K$ . Substituting this into the sum of Equation (D.3) gives Equation (D.2). □

Proposition D.5 (Logarithmic upper bound on $\sum _ { k } \boldsymbol { r } _ { t _ { k } } )$ . Under Assumption 3.1, run Algorithm 3 on an arbitrary sequence of states. If $u \in \Theta$ satisfies $\langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \geq 0$ at every mistake round $t _ { k } .$ , then

$$
\sum _ { k = 1 } ^ { K } \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \leq L D \left( 1 + d \log \left( 1 + \frac { K } { d } \right) \right) .\tag{D.4}
$$

Both $u = \bar { \theta }$ and $u = \theta ^ { * }$ satisfy this condition, and then the left-hand side of Equation (D.4) is $\textstyle \sum _ { k = 1 } ^ { K } r _ { t _ { k } }$ and $\widetilde { R } _ { T }$ respectively.

Proof of Proposition D.5. We use Lemma D.4 with this u. We have $\begin{array} { r l } { z _ { t _ { k } } } & { { } = } \end{array}$ $\eta \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } \mathrm { ~ - ~ } u \rangle$ , and from the assumption and the Cauchy–Schwarz inequality $( \lVert g ^ { t _ { k } } \rVert \leq L$ , and $\| \hat { \theta } ^ { t _ { k } } - u \| \leq D$ since $\hat { \theta } ^ { t _ { k } } , u \in \Theta )$ we get $0 \leq \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle \leq L D$ so $z _ { t _ { k } } \in [ 0 , 1 ]$ and hence $z _ { t _ { k } } ^ { 2 } \le z _ { t _ { k } }$ . The left-hand side of Lemma D.4 is bounded from below by $\begin{array} { r } { \sum _ { k } z _ { t _ { k } } - \frac { 1 } { 2 } \sum _ { k } z _ { t _ { k } } ^ { 2 } \geq \frac { 1 } { 2 } \sum _ { k } z _ { t _ { k } } } \end{array}$ , and the right-hand side is bounded, by $\| \hat { \theta } ^ { 1 } - u \| _ { \Sigma _ { 0 } } ^ { 2 } = D ^ { - 2 } \| \hat { \theta } ^ { 1 } - u \| ^ { 2 } \leq 1$ and Lemma D.2 (the condition $\| \nabla _ { t _ { k } } \| \leq 1 / D$ having been checked), as $\textstyle { \frac { 1 } { 2 } } + { \frac { d } { 2 } } \log ( 1 + K / d )$ from above (neither of the two terms on the right-hand side depends on u). Hence $\begin{array} { r } { \sum _ { k } z _ { t _ { k } } \le 1 + d \log ( 1 + K / d ) } \end{array}$ , and multiplying back by $\langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - u \rangle = L D z _ { t _ { k } }$ gives Equation (D.4).

Verification of the condition: we have $\bar { \theta } \in \Theta$ (Assumption $3 . 1 ( 4 ) )$ ), and Lemma C.1 gives $r _ { t _ { k } } = \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } - \bar { \theta } \rangle \geq \ell ^ { t _ { k } } + \gamma > 0$ . We have $\theta ^ { * } \in \Theta \ ( A$ ssumption $3 . 1 ( 3 ) )$ , and Equation (C.3) gives $\widetilde { r } _ { t _ { k } } ~ = ~ \langle g ^ { t _ { k } } , \hat { \theta } ^ { t _ { k } } ~ - \theta ^ { * } \rangle ~ \geq ~ 0 .$ In the latter case, since the contribution of the rounds without a mistake is 0, the left-hand side equals $\begin{array} { r } { \sum _ { k = 1 } ^ { K } { \widetilde { r } _ { t _ { k } } } = \widetilde { R } _ { T } } \end{array}$ □

Theorem D.6. Under Assumption 3.1, run ONS (Algorithm 3) on an arbitrary sequence of states $\{ s ^ { t } \} _ { t = 1 } ^ { T }$ . Then, for every $T _ { \mathrm { : } }$ , the following hold.

(i): (K)

$$
K \leq d + \frac { 2 L D } { \gamma } \left( 1 + d \log \operatorname* { m a x } \left( \frac { 2 L D } { \gamma } , 1 \right) \right) .\tag{D.5}
$$

(ii): $( R _ { T } ^ { \mathrm { s u b } } )$

$$
R _ { T } ^ { \mathrm { s u b } } \leq L D \left( 1 + d \log \operatorname* { m a x } \left( \frac { L D } { \gamma } , 1 \right) \right) .\tag{D.6}
$$

(iii): $( \widetilde { R } _ { T } )$

$$
\widetilde { R } _ { T } \leq L D \left( 1 + 2 d \log \left( 2 + \frac { 2 L D } { \gamma } \right) \right) .\tag{D.7}
$$

By Equation (3.8), the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ also has the same upper bound as Equation (D.7).

Proof of Theorem D.6. (i) Since Lemma C.1 gives $\begin{array} { r } { \sum _ { k } r _ { t _ { k } } \ge \gamma K } \end{array}$ , combining it with Proposition D.5 (with $u = \bar { \theta } )$ yields

$$
\gamma K \leq L D \left( 1 + d \log \left( 1 + { \frac { K } { d } } \right) \right) .\tag{D.8}
$$

Setting $y : = K / d$ , we have $\begin{array} { r } { y \le \frac { L D } { \gamma d } + \frac { L D } { \gamma } \log ( 1 + y ) } \end{array}$ , and Lemma D.3 (with $\begin{array} { r } { c _ { 1 } = \frac { L D } { \gamma d } } \end{array}$ and $\begin{array} { r } { c _ { 2 } = \frac { L D } { \gamma } ) } \end{array}$ gives

$$
y \leq \frac { 2 L D } { \gamma d } + 1 + \frac { 2 L D } { \gamma } \log \operatorname* { m a x } \left( \frac { 2 L D } { \gamma } , 1 \right) ;
$$

multiplying both sides by d gives Equation (D.5).

(ii) At rounds where no mistake occurs we have $\hat { x } ^ { t } = x ^ { t } \in \arg \operatorname* { m a x } _ { x } \langle \hat { \theta } ^ { t } , x \rangle$ , so $\ell _ { \mathrm { s u b } } ( \hat { \theta } ^ { t } , s ^ { t } ) = 0$ , and hence $\begin{array} { r } { R _ { T } ^ { \mathrm { s u b } } = \sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } } \end{array}$ . From $\ell ^ { t _ { k } } \leq r _ { t _ { k } } - \gamma$ of Lemma C.1 and Proposition D.5 (with $u = \bar { \theta } )$ we obtain

$$
\sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } \leq \sum _ { k = 1 } ^ { K } r _ { t _ { k } } - \gamma K \leq L D \left( 1 + d \log \left( 1 + { \frac { K } { d } } \right) \right) - \gamma K \leq \operatorname* { m a x } _ { x \geq 0 } \phi ( x ) ,
$$

where we have set $\begin{array} { r } { \phi ( x ) : = L D ( 1 + d \log ( 1 + \frac { x } { d } ) ) - \gamma x } \end{array}$ . The function ϕ is diferentiable with $\begin{array} { r } { \phi ^ { \prime } ( x ) = \frac { L D d } { d + x } - \gamma } \end{array}$ . Case $1 \ ( \gamma \geq L D )$ : for every $x \geq 0$ we have $\phi ^ { \prime } ( x ) \leq L D - \gamma \leq$ 0, so ϕ is nonincreasing and m $\operatorname { a x } _ { x \geq 0 } \phi = \phi ( 0 ) = L D$ . Case $\mathcal { Z } \left( \gamma < L D \right)$ : the solution

of $\phi ^ { \prime } = 0$ is $\begin{array} { r } { x ^ { * } = \frac { L D d } { \gamma } - d > 0 } \end{array}$ , and $\phi$ is increasing on $[ 0 , x ^ { * } ]$ and decreasing on $[ x ^ { * } , \infty )$ , so

$$
\displaystyle { \operatorname* { m a x } _ { x \ge 0 } \phi = \phi ( x ^ { * } ) = L D + L D d \log \frac { L D } { \gamma } - L D d + \gamma d \le L D + L D d \log \frac { L D } { \gamma } } ,
$$

where we used $\gamma d \leq L D d$ . In either case $\begin{array} { r } { \operatorname* { m a x } _ { x \geq 0 } \phi \leq L D ( 1 + d \log \operatorname* { m a x } ( L D / \gamma , 1 ) ) } \end{array}$ and we obtain Equation (D.6).

(iii) Using Proposition D.5 with $u = \theta ^ { * }$ we obtain

$$
\widetilde { R } _ { T } \leq L D \left( 1 + d \log \left( 1 + \frac { K } { d } \right) \right) .\tag{D.9}
$$

Setting $\begin{array} { r } { \beta : = \frac { 2 L D } { \gamma } } \end{array}$ , item (i) gives $K \leq K _ { \mathrm { O N S } } : = d + \beta ( 1 + d \log \operatorname* { m a x } ( \beta , 1 ) )$ . Here

$$
\begin{array} { c l l } { \displaystyle 1 + \frac { K _ { \mathrm { O N S } } } { d } = 2 + \frac { \beta } { d } + \beta \log \operatorname* { m a x } ( \beta , 1 ) \leq 2 + \beta + \beta \log \operatorname* { m a x } ( \beta , 1 ) } \\ { \leq ( 2 + \beta ) \left( 1 + \log \operatorname* { m a x } ( \beta , 1 ) \right) } \end{array}
$$

(the last inequality holds because expanding the right-hand side produces 2 log max $( \beta , 1 ) \ge 0 )$ . Furthermore, since log max $\cdot ( \beta , 1 ) \leq \beta$ gives $1 + \log \operatorname* { m a x } ( \beta , 1 ) \leq$ $2 + \beta ,$ we obtain $\log ( 1 + K _ { \mathrm { O N S } } / d ) \leq 2 \log ( 2 + \beta )$ . Since the right-hand side of Equation (D.9) is monotonically increasing in $K$ , substituting $K \le K _ { \mathrm { O N S } }$ yields Equation (D.7). □

## Appendix E. Analysis of MetaGrad (proof of Theorem E.1)

## E.1. The guarantee of the fixed-grid version.

Theorem E.1. Under Assumption 3.1, run MetaGrad (Algorithm 5) on an arbitrary sequence of states $\{ s ^ { t } \} _ { t = 1 } ^ { T }$ , and set $c _ { 0 } ( \bar { K } ) : = 2 \log ( \textstyle { \frac { 1 } { 2 } } \log _ { 2 } { \bar { K } } + 3 )$ . Then the following hold.

(i): (K)

$$
K \leq d + \frac { 1 5 2 L D } { 3 \gamma } ( c _ { 0 } ( \bar { K } ) + d ) + \frac { 1 5 2 L D d } { 3 \gamma } \log \operatorname* { m a x } \left( \frac { 1 5 2 L D } { 3 \gamma } , 1 \right) .\tag{E.1}
$$

(ii): (R<sup>sub</sup><sub>T</sub> )

$$
R _ { T } ^ { \mathrm { s u b } } \leq \frac { 7 6 } { 3 } L D \left( c _ { 0 } ( \bar { K } ) + d + d \log \operatorname* { m a x } \left( \frac { 7 6 L D } { 3 \gamma } , 1 \right) \right) .\tag{E.2}
$$

(iii): $( \widetilde { R } _ { T } )$ Writing $K _ { \mathrm { m a x } }$ for the right-hand side of Equation (E.1),

$$
\widetilde { R } _ { T } \leq \frac { 7 6 } { 3 } L D \left( c _ { 0 } ( \bar { K } ) + d \left( \log \left( 1 + \frac { K _ { \operatorname* { m a x } } } { 4 9 d } \right) + 1 \right) \right) .\tag{E.3}
$$

The proof is given in the next subsection.

Remark E.2 (Comparison with Theorem D.6). Theorem E.1 has the same dependence on $\gamma$ and d as Equations (D.5) and (D.6): the number of mistakes is $O ( \textstyle { \frac { d L D } { \gamma } }$ log max $\textstyle ( { \frac { 2 L D } { \gamma } } , 2 ) { \bar { ) } }$ , and the cumulative suboptimality regret is $O ( L D$ d log max $\left( L D / \gamma , 1 \right) )$ . The price is (a) worse constants (the coeficient of the logarithmic term is $\frac { 1 5 2 } { 3 }$ against 2, and there is an additive term $\textstyle { \frac { 7 6 } { 3 } } L D ( c _ { 0 } ( { \bar { K } } ) + d ) )$ ; (b) a doubly logarithmic dependence on the grid upper bound $\bar { K }$ (taking $\bar { K } = T$ breaks strict independence from $T$ to the extent of $c _ { 0 } ( \bar { K } ) = O ( \log \log T )$ ; this dependence is removed by considering the growing-grid version of MetaGrad,

Theorem F.3); and (c) a computational cost per mistake round multiplied by the grid size $\begin{array} { r } { ( 1 + \lceil \frac { 1 } { 2 } \log _ { 2 } \bar { K } \rceil } \end{array}$ experts each perform $O ( d ^ { 2 } )$ plus a generalized projection; see the end of $\bar { \ S } 4$ of Sakaue et al., 2025b). The parameters of the algorithm are $L , D , { \bar { K } }$ ; as with ONS, no knowledge of $\gamma$ is required.

## E.2. Proof. In the analysis we cite the following regret upper bound.

Proposition E.3 (Upper bound on the linearized regret of MetaGrad; cf. Proposition 2.6 of Sakaue et al., 2025b). Let n be a positive integer, let $\mathcal { W } \subset \mathbb { R } ^ { n }$ be a nonempty closed convex set whose $\ell _ { 2 }$ diameter is at most $W > 0$ , and take $G , H > 0$ and positive integers $m \leq \bar { m }$ . Let $h _ { 1 } , \hdots , h _ { m } \colon \mathcal { W } \to \mathbb { R }$ be a sequence of convex loss functions and let $w _ { 1 } , \dots , w _ { m } \in \mathcal { W }$ be the outputs of MetaGrad (Algorithm 4) applied to $h _ { 1 } , \ldots , h _ { m }$ . If, for each $j = 1 , \dots , m$ , the subgradient $g _ { j } ~ \in \partial h _ { j } ( w _ { j } )$ observed in Algorithm 4 satisfies $\| g _ { j } \| \leq G$ and $\operatorname* { s u p } \{ \langle w ^ { \prime } - w , g _ { j } \rangle \mid w , w ^ { \prime } \in \mathcal { W } \} \leq \bar { H }$ then, for every $u \in \mathcal W$

$$
\sum _ { j = 1 } ^ { m } \bigl < w _ { j } - u , g _ { j } \bigr > \leq 3 \sqrt { \Lambda _ { m } V _ { m } ^ { u } } + 1 0 H \Lambda _ { m } , \qquad V _ { m } ^ { u } : = \sum _ { j = 1 } ^ { m } \bigl < w _ { j } - u , g _ { j } \bigr > ^ { 2 }\tag{E.4}
$$

holds, where

$$
\Lambda _ { m } : = 2 \log \left( \frac { 1 } { 2 } \log _ { 2 } \bar { m } + 3 \right) + n \left( \log \left( \frac { W ^ { 2 } G ^ { 2 } m } { 4 9 n H ^ { 2 } } + 1 \right) + 1 \right) .\tag{E.5}
$$

Below we write the index of the grid as $I : = \lceil \frac { 1 } { 2 } \log _ { 2 } \bar { m } \rceil$ and $\begin{array} { r } { \mathcal { E } = \{ \eta _ { i } = \frac { 2 ^ { - i } } { 5 H } \ | } \end{array}$ $i = 0 , 1 , \ldots , I \}$ (Algorithm 4). Moreover, for the $n , W , G , H$ of Proposition E.3, we define the function

$$
B _ { q } : = n \left( \log \left( \frac { W ^ { 2 } G ^ { 2 } q } { 4 9 n H ^ { 2 } } + 1 \right) + 1 \right)\tag{E.6}
$$

of a positive integer q. The map $q \mapsto B _ { q }$ is monotonically nondecreasing, and $B _ { m }$ equals the second term of Equation (E.5).

The only external result cited in the proof is the following regret upper bound for a single η-expert.

Proposition E.4 (Regret upper bound for an η-expert; Appendix C.3 of Sakaue et al., 2025b). Let n be a positive integer, let $\mathcal { W } \subset \mathbb { R } ^ { n }$ be a nonempty closed convex set whose $\ell _ { 2 }$ diameter is at most $W > 0$ , and take $G , H > 0 , \eta \in ( 0 , \frac { 1 } { 5 H } ]$ and a positive integer $q .$ For a sequence of points $v _ { 1 } , \dots , v _ { q } \in \mathcal { W }$ and a sequence of vectors $g _ { 1 } , \ldots , g _ { q } \in \mathbb { R } ^ { n }$ (with $\| g _ { j } \| \leq G$ and sup $\{ \langle w ^ { \prime } - w , \dot { g } _ { j } \rangle \mid w , w ^ { \prime } \in \mathcal { W } \} \leq H )$ , define the surrogate losses by $\mathring { \ell _ { j } ^ { \eta } } ( w ) : = - \eta \langle v _ { j } - w , g _ { j } \rangle + \eta ^ { 2 } \langle v _ { j } - w , g _ { j } \rangle ^ { 2 }$ , and suppose that running the η-expert (Definition B.3) on $\ell _ { 1 } ^ { \eta } , \ldots , \ell _ { q } ^ { \eta }$ yields $w _ { 1 } ^ { \eta } , \ldots , w _ { q } ^ { \eta } \in \mathcal { W }$ . Then, for every $u \in \mathcal W$

$$
\sum _ { j = 1 } ^ { q } \left( \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) - \ell _ { j } ^ { \eta } ( u ) \right) \leq B _ { q }
$$

holds (where $B _ { q }$ is as in Equation (E.6)).

Lemma E.5 (Monotonicity of the potential for a fixed grid). In the setting of Proposition E.3, set $\begin{array} { r } { \mathcal { L } _ { j } ^ { i } : = \sum _ { j ^ { \prime } = 1 } ^ { j } \ell _ { j ^ { \prime } } ^ { \eta _ { i } } ( w _ { j ^ { \prime } } ^ { \eta _ { i } } ) } \end{array}$ (with $\mathcal { L } _ { 0 } ^ { i } : = 0 )$ and $\begin{array} { r } { \Phi _ { j } : = \sum _ { i = 0 } ^ { I } p _ { 1 } ^ { \eta _ { i } } \exp ( - \mathcal { L } _ { j } ^ { i } ) } \end{array}$ . Then $\Phi _ { m } \leq \Phi _ { 0 } = 1$ , and in particular, for every $i \in \{ 0 , 1 , \ldots , I \} , - \mathcal { L } _ { m } ^ { i } \leq 2 \log ( i + 2 )$

Proof. Setting $\tilde { p } _ { j } ^ { i } : = p _ { 1 } ^ { \eta _ { i } } \exp ( - \mathcal { L } _ { j - 1 } ^ { i } )$ , the weight update of Algorithm 4 gives $p _ { j } ^ { \eta _ { i } } =$ $\begin{array} { r } { \tilde { p } _ { j } ^ { i } / \prod _ { j ^ { \prime } < j } Z _ { j ^ { \prime } } } \end{array}$ , and since the point of the master is determined by the ratios of the weights alone,

$$
w _ { j } = \frac { \sum _ { i = 0 } ^ { I } \eta _ { i } p _ { j } ^ { \eta _ { i } } w _ { j } ^ { \eta _ { i } } } { \sum _ { i = 0 } ^ { I } \eta _ { i } p _ { j } ^ { \eta _ { i } } } = \frac { \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } } { \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } _ { j } ^ { i } }\tag{E.7}
$$

holds. We show $\Phi _ { j } \leq \Phi _ { j - 1 }$ for each $j .$ . Setting $x _ { i } : = \eta _ { i } \langle w _ { j } - w _ { j } ^ { \eta _ { i } } , g _ { j } \rangle$ , we have $\ell _ { j } ^ { \eta _ { i } } ( w _ { j } ^ { \eta _ { i } } ) = - x _ { i } + x _ { i } ^ { 2 }$ and $\begin{array} { r } { | x _ { i } | \le \eta _ { i } H \le \frac { 1 } { 5 } } \end{array}$ . From the elementary inequality $e ^ { x - x ^ { 2 } } \leq$ $1 + x .$ , valid for $x \geq - { \frac { 1 } { 2 } }$ (because $f ( x ) : = \log ( 1 + x ) - x + x ^ { 2 }$ has $\textstyle f ^ { \prime } ( x ) = { \frac { x ( 2 x + 1 ) } { 1 + x } }$ and hence attains its minimum value 0 at $x = 0 )$ , we obtain

$$
\Phi _ { j } = \sum _ { i = 0 } ^ { I } \hat { p } _ { j } ^ { i } e ^ { x _ { i } - x _ { i } ^ { 2 } } \leq \sum _ { i = 0 } ^ { I } \hat { p } _ { j } ^ { i } ( 1 + x _ { i } ) = \Phi _ { j - 1 } + \left. w _ { j } \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } _ { j } ^ { i } - \sum _ { i = 0 } ^ { I } \eta _ { i } \tilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } , g _ { j } \right. = \Phi _ { j - 1 }
$$

(using $\begin{array} { r } { \sum _ { i } \tilde { p } _ { j } ^ { i } = \Phi _ { j - 1 } } \end{array}$ , the last equality being Equation (E.7)). Since $\begin{array} { r } { \Phi _ { 0 } = \sum _ { i = 0 } ^ { I } p _ { 1 } ^ { \eta _ { i } } = } \end{array}$ 1, we have $p _ { 1 } ^ { \eta _ { i } } e ^ { - \mathcal { L } _ { m } ^ { i } } \leq \Phi _ { m } \leq 1$ . Since $\begin{array} { r } { \sum _ { i = 0 } ^ { I } \frac { 1 } { ( i + 1 ) ( i + 2 ) } = 1 - \frac { 1 } { I + 2 } \le 1 } \end{array}$ implies that the normalizing constant satisfies $C \geq 1$ , we get $\begin{array} { r } { - \mathscr { L } _ { m } ^ { i } \leq \log \frac { 1 } { p _ { 1 } ^ { \eta _ { i } } } = \log \frac { ( i + 1 ) ( i + 2 ) } { C } \leq } \end{array}$ $\log ( ( i + 1 ) ( i + 2 ) ) \leq 2 \log ( i + 2 )$ □

Proof of Proposition E.3. Take $u \in \mathcal W$ and $i \in \{ 0 , 1 , \ldots , I \}$ , and set $\eta : = \eta _ { i }$ and $a _ { j } : = \langle w _ { j } - u , g _ { j } \rangle$ (so that $| a _ { j } | \leq H$ by assumption). From the definition of the surrogate loss we have $- \ell _ { j } ^ { \eta } ( u ) \dot { = } \eta a _ { j } - \eta ^ { 2 } a _ { j } ^ { 2 }$ , so summing over $j = 1 , \ldots , m$ gives

$$
\sum _ { j = 1 } ^ { m } a _ { j } = \frac { 1 } { \eta } \sum _ { j = 1 } ^ { m } \left( - \ell _ { j } ^ { \eta } ( u ) \right) + \eta V _ { m } ^ { u } .\tag{E.8}
$$

Decomposing the sum in the first term on the right-hand side as

$$
\sum _ { j = 1 } ^ { m } ( - \ell _ { j } ^ { \eta } ( u ) ) = ( - \mathcal { L } _ { m } ^ { i } ) + \sum _ { j = 1 } ^ { m } \left( \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) - \ell _ { j } ^ { \eta } ( u ) \right) ,
$$

the first term is at most $2 \log ( i + 2 )$ by Lemma E.5, and the second is at most $B _ { m }$ by Proposition E.4 (with $q = m )$ , since the η-expert of Algorithm 4 is run on $\ell _ { 1 } ^ { \eta } , \ldots , \ell _ { m } ^ { \eta }$ with $v _ { j } = w _ { j }$ . Substituting into Equation (E.8), we obtain, for every $i \in \{ 0 , 1 , \ldots , I \}$ ，

$$
\sum _ { j = 1 } ^ { m } a _ { j } \le \frac { 2 \log ( i + 2 ) + B _ { m } } { \eta _ { i } } + \eta _ { i } V _ { m } ^ { u } .\tag{E.9}
$$

We distinguish cases according to $\eta ^ { * } : = \sqrt { \Lambda _ { m } / V _ { m } ^ { u } }$ (with $\eta ^ { * } : = + \infty$ when $V _ { m } ^ { u } = 0 )$ Below we repeatedly use the fact that the first term of Equation (E.5) is at least $2 \log ( \frac { 1 } { 2 } \log _ { 2 } \bar { m } + 3 ) \geq 2 \log 3$

<sup>2</sup>Case $\textit { 1 } ( \eta ^ { * } \geq \frac { 1 } { 5 H }$ , that is, $V _ { m } ^ { u } \leq 2 5 H ^ { 2 } \Lambda _ { m } ) \colon$ : we use Equation (E.9) with $i = 0$ $\begin{array} { r } { ( \eta _ { 0 } = \frac { 1 } { 5 H } ) } \end{array}$ . From 2 log $2 + B _ { m } \le \Lambda _ { m }$ and $\frac { V _ { m } ^ { u } } { 5 H } \leq 5 H \Lambda _ { m }$

$$
\sum _ { j = 1 } ^ { m } a _ { j } \le 5 H ( 2 \log 2 + B _ { m } ) + \frac { V _ { m } ^ { u } } { 5 H } \le 1 0 H \Lambda _ { m } .
$$

Case 2 $\begin{array} { r } { ( \eta ^ { * } < \frac { 1 } { 5 H } ) : } \end{array}$ we first show $\eta ^ { * } >$ min ${ \mathcal { E } } = \eta _ { I }$ . From $| a _ { j } | \le H$ we have $V _ { m } ^ { u } \leq H ^ { 2 } m \leq H ^ { 2 } { \bar { m } }$ , and Equation (E.5) gives $\Lambda _ { m } \geq 2 \log 3 + n \geq 1$ , so

$$
\eta ^ { * } = \sqrt { \frac { \Lambda _ { m } } { V _ { m } ^ { u } } } \ge \frac { 1 } { H \sqrt { \bar { m } } } > \frac { 1 } { 5 H \sqrt { \bar { m } } } = \frac { 2 ^ { - \frac { 1 } { 2 } \log _ { 2 } \bar { m } } } { 5 H } \ge \frac { 2 ^ { - I } } { 5 H } = \eta _ { I }
$$

(the last inequality holding because $I \geq { \textstyle \frac { 1 } { 2 } } \log _ { 2 } \bar { m } )$ . Letting $i ^ { * }$ be the largest i with $\eta _ { i } \geq \eta ^ { * }$ , such an $i ^ { * }$ exists since $\begin{array} { r } { \eta _ { 0 } = \frac { 1 } { 5 H } > \eta ^ { * } } \end{array}$ , and $i ^ { * } \leq I - 1$ since $\eta _ { I } < \eta ^ { * }$ . By the maximality of $i ^ { * }$ we have $\eta _ { i ^ { * } + 1 } = \eta _ { i ^ { * } } / 2 < \eta ^ { * }$ , hence $\eta ^ { \ast } \leq \eta _ { i ^ { \ast } } < 2 \eta ^ { \ast }$ . Moreover, $I \leq$ $\textstyle { \frac { 1 } { 2 } } \log _ { 2 } { \bar { m } } + 1$ gives $i ^ { * } \leq \frac { 1 } { 2 } \log _ { 2 } \bar { m }$ , so $2 \log ( i ^ { * } + 2 ) + B _ { m } \leq 2 \log ( { \frac { 1 } { 2 } } \log _ { 2 } { \bar { m } } + 3 ) + B _ { m } \leq \Lambda _ { m }$ Using Equation (E.9) with $i = i ^ { * }$ , we obtain

$$
\sum _ { j = 1 } ^ { m } a _ { j } \le \frac { \Lambda _ { m } } { \eta _ { i ^ { * } } } + \eta _ { i ^ { * } } V _ { m } ^ { u } \le \frac { \Lambda _ { m } } { \eta ^ { * } } + 2 \eta ^ { * } V _ { m } ^ { u } = 3 \sqrt { \Lambda _ { m } V _ { m } ^ { u } } .
$$

In either case the right-hand side is at most $3 \sqrt { \Lambda _ { m } V _ { m } ^ { u } } + 1 0 H \Lambda _ { m } ,$ so Equation (E.4) holds. □

Remark E.6. Proposition E.3 corresponds to Proposition 2.6 of Sakaue et al. (2025b), but there it is stated in $O ( \cdot )$ notation for the case where the grid is constructed from the actual number of rounds $( { \bar { m } } = m )$ . The proof above makes the constants explicit and treats the case where the grid is constructed from an upper bound m¯ on m: in our application $m = K$ (the realized number of mistakes) is unknown before execution, so the grid has to be fixed in advance by $\bar { K } \ge K$ which is why this generalization is needed. The quantity m¯ enters only in the first term of Equation (E.5) and at the place in Case 2 where the lower end $\eta _ { I }$ of the grid is estimated.

Proof of Theorem E.1. If $K = 0$ everything is trivial, so assume $K \geq 1$ . Since the internal state of Algorithm 5 (the index k, the weights, the experts, and the prediction ${ \hat { \theta } } ^ { t } )$ does not change at rounds where no mistake occurs, Algorithm 5 is nothing but MetaGrad of Algorithm 4 (with $n = d ,$ W = Θ, $W = D$ $H = L D$ ， ${ \bar { m } } = { \bar { K } } )$ applied to the sequence of convex losses $h _ { k } : = \ell _ { \mathrm { s u b } } ( \cdot , s ^ { t _ { k } } ) ( k = 1 , \dots , K )$ of length $m = K$ (under the reindexing of $\ S 4 , w _ { k } = \hat { \theta } ^ { t _ { k } }$ and $g _ { k } = g ^ { t _ { k } } ;$ ; the surrogate loss of Algorithm 5 coincides with that of Algorithm 4 since $w _ { k } = \hat { \theta } ^ { t _ { k } }$ at the k-th mistake round). To apply Proposition E.3 with $G = L$ , we verify its assumptions. That $g ^ { t _ { k } } \in \partial h _ { k } ( \hat { \theta } ^ { t _ { k } } )$ follows from

$$
h _ { k } ( \theta ) \geq \langle \theta , \hat { x } ^ { t _ { k } } - x ^ { t _ { k } } \rangle = h _ { k } ( \hat { \theta } ^ { t _ { k } } ) + \langle \theta - \hat { \theta } ^ { t _ { k } } , g ^ { t _ { k } } \rangle
$$

for every $\theta \in \Theta$ (the inequality by $\hat { x } ^ { t _ { k } } \in Y ( s ^ { t _ { k } } )$ , the equality by $\langle \hat { \theta } ^ { t _ { k } } , g ^ { t _ { k } } \rangle = \ell ^ { t _ { k } }$ from the proof of Lemma C.1). Moreover $\| g ^ { t _ { k } } \| \le L = G$ (Equation (3.2)), the Cauchy–Schwarz inequality gives sup $\{ \langle \theta ^ { \prime } - \theta , g ^ { t _ { k } } \rangle \mid \theta , \theta ^ { \prime } \in \Theta \} \le D L = H$ , and furthermore $K \leq { \bar { K } }$

Step 1. (Estimate of $\textstyle \sum _ { k } r _ { t _ { k } } )$ For $u = \bar { \theta }$ we have $\langle w _ { k } - u , g _ { k } \rangle = r _ { t _ { k } }$ , so Equations (E.4) and (E.5) (note that $W ^ { 2 } G ^ { 2 } / H ^ { 2 } = D ^ { 2 } L ^ { 2 } / ( L D ) ^ { 2 } = 1 )$ give

$$
R : = \sum _ { k = 1 } ^ { K } r _ { t _ { k } } \leq 3 \sqrt { \Lambda V } + 1 0 L D \Lambda ,
$$

$$
V : = \sum _ { k = 1 } ^ { K } r _ { t _ { k } } ^ { 2 } , \quad \Lambda : = c _ { 0 } + d \left( \log \left( 1 + \frac { K } { 4 9 d } \right) + 1 \right) .\tag{E.10}
$$

Since Lemma C.1 gives $0 < r _ { t _ { k } } \ \le \ L D$ , we have $V \leq L D \cdot R ,$ and hence $R \leq$ $3 \sqrt { \Lambda L D R } + 1 0 L D \Lambda = \sqrt { a R } + \zeta$ (with $a : = 9 L D \Lambda$ and $\zeta : = 1 0 L D \Lambda )$ . Here, for all $a , \zeta , R \geq 0$

$$
R \leq \sqrt { a R } + \zeta \quad \Longrightarrow \quad R \leq \frac { 4 } { 3 } ( a + \zeta )\tag{E.11}
$$

holds. Indeed, assuming $R \leq \sqrt { a R } + \zeta .$

$$
R = \frac { 4 } { 3 } R - \frac { 1 } { 3 } R \leq \frac { 4 } { 3 } \left( \sqrt { a R } + \zeta \right) - \frac { 1 } { 3 } R = - \frac { 1 } { 3 } \left( \sqrt { R } - 2 \sqrt { a } \right) ^ { 2 } + \frac { 4 } { 3 } ( a + \zeta ) \leq \frac { 4 } { 3 } ( a + \zeta ) .
$$

Hence Equation (E.11) yields

$$
\sum _ { k = 1 } ^ { K } r _ { t _ { k } } = R \leq \frac { 4 } { 3 } \cdot 1 9 L D \Lambda = \frac { 7 6 } { 3 } L D \left( c _ { 0 } + d \left( \log \left( 1 + \frac { K } { 4 9 d } \right) + 1 \right) \right) .\tag{E.12}
$$

(i) Since Lemma C.1 gives $\gamma K \leq R$ , from Equation (E.12) and $\begin{array} { r } { \log ( 1 + \frac { K } { 4 9 d } ) \leq } \end{array}$ $\textstyle \log ( 1 + { \frac { K } { d } } )$ we get

$$
\gamma K \leq \frac { 7 6 } { 3 } L D ( c _ { 0 } + d ) + \frac { 7 6 } { 3 } L D d \log \left( 1 + \frac { K } { d } \right) .
$$

Setting $y : = K / d ,$ we have $y \leq c _ { 1 } + c _ { 2 } \log ( 1 + y )$ (with $\begin{array} { r } { c _ { 1 } : = \frac { 7 6 L D ( c _ { 0 } + d ) } { 3 \gamma d } } \end{array}$ and $\begin{array} { r } { c _ { 2 } : = \frac { 7 6 L D } { 3 \gamma } ) } \end{array}$ , and Lemma D.3 gives $y \le 2 c _ { 1 } + 1 + 2 c _ { 2 }$ log max(2c<sub>2</sub>, 1). Multiplying both sides by d gives Equation (E.1).

(ii) As in the proof of Theorem $\mathrm { D } . 6 ( \mathrm { i i } )$ we have $\begin{array} { r } { R _ { T } ^ { \mathrm { s u b } } = \sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } \leq R - \gamma K } \end{array}$ , and setting $\begin{array} { r } { \rho : = \frac { 7 6 } { 3 } L D } \end{array}$ , Equation (E.12) gives

$$
\sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } \leq \rho ( c _ { 0 } + d ) + \operatorname* { m a x } _ { x \geq 0 } \psi ( x ) , \qquad \psi ( x ) : = \rho d \log \left( 1 + { \frac { x } { d } } \right) - \gamma x .
$$

Since $\begin{array} { r } { \psi ^ { \prime } ( x ) = \frac { \rho d } { d + x } - \gamma , \operatorname { i f } \gamma \geq \rho } \end{array}$ then $\psi$ is nonincreasing and $\operatorname* { m a x } _ { x \geq 0 } \psi = \psi ( 0 ) = 0$ whereas if $\mathit { \Pi } \cdot \mathit { \Pi } \gamma < \rho$ then at the stationary point $\begin{array} { r } { x ^ { * } = \frac { \rho d } { \gamma } - d > 0 } \end{array}$ we have $\psi ( x ^ { * } ) =$ ρd log $\begin{array} { r } { \frac { \rho } { \gamma } - \rho d + \gamma d \leq \rho d \log \frac { \rho } { \gamma } } \end{array}$ . In either case ma $x _ { x \geq 0 } \psi \leq \rho d$ log $\operatorname* { m a x } ( \rho / \gamma , 1 )$ , so we obtain Equation (E.2).

(iii) Since Proposition E.3 holds for every $u \in \mathcal W$ , we apply it with $u = \theta ^ { * }$ (where $\theta ^ { * } \in \Theta = \mathcal { W }$ by Assumption $3 . 1 ( 3 ) )$ . We have $\left. w _ { k } - \theta ^ { * } , g _ { k } \right. = \widetilde { r } _ { t _ { k } }$ , and the contribution of the rounds without a mistake is $0 ,$ so Equations (E.4) and (E.5) give the version of Equation (E.10) with $r _ { t _ { k } }$ replaced by $\widetilde { r } _ { t _ { k } }$

$$
\widetilde { R } _ { T } = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } \le 3 \sqrt { \Lambda \widetilde { V } } + 1 0 L D \Lambda , \qquad \widetilde { V } : = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } ^ { 2 }
$$

(where Λ is that of Equation (E.10)). Since $0 \le \widetilde { r } _ { t _ { k } } \le L D$ from Equation (C.3) gives $ { \widetilde { V } } \leq L D  { \widetilde { R } } _ { T }$ , we have $\widetilde { R } _ { T } \leq \sqrt { a \widetilde { R } _ { T } } + \zeta$ (with $a = 9 L D \Lambda$ and $\zeta = 1 0 L D \Lambda )$ , and Equation (E.11) yields

$$
\widetilde { R } _ { T } \leq \frac { 4 } { 3 } \cdot 1 9 L D \Lambda = \frac { 7 6 } { 3 } L D \left( c _ { 0 } ( \bar { K } ) + d \left( \log \left( 1 + \frac { K } { 4 9 d } \right) + 1 \right) \right) .
$$

Since the right-hand side is monotonically increasing in $K$ , substituting $K \leq K _ { \operatorname* { m a x } }$ from (i) gives Equation (E.3). □

## Appendix F. Analysis of growing-grid MetaGrad (proof of Theorem F.3)

In this appendix we analyze growing-grid MetaGrad (Algorithm 6) and prove Theorem F.3. The key is the reduction that regards a not-yet-created η<sub>i</sub>-expert as a “virtual expert that outputs the point of the master”.

Proposition F.1 (Upper bound on the linearized regret of growing-grid MetaGrad). In the setting of Proposition E.3, modify MetaGrad (Algorithm 4) as follows: place the prior weights $\begin{array} { r } { p _ { i } : = \frac { 1 } { ( i + 1 ) ( i + 2 ) } } \end{array}$ on the countable grid $\begin{array} { r } { \eta _ { i } : = \frac { 2 ^ { - i } } { 5 H } ( i \in \mathbb { Z } _ { \geq 0 } ) } \end{array}$ , create the η<sub>i</sub>-expert at round $j _ { i } : = 4 ^ { i - 1 } + 1 \ ( \mathrm { f o r } \ i \geq 1 ; \ j _ { 0 } : = 1 )$ and run ONS from then on, take as the point of the master the sum in Equation (B.2) restricted to the already created experts (those i with $j \geq j _ { i } )$ , and use the weights without normalizing them. Then, for every $u \in \mathcal W$

$$
\sum _ { j = 1 } ^ { m } \bigl < w _ { j } - u , g _ { j } \bigr > \leq 3 \sqrt { \Lambda _ { m } ^ { \prime } V _ { m } ^ { u } } + 1 0 H \Lambda _ { m } ^ { \prime } + \frac { V _ { m } ^ { u } } { 1 0 0 H \Lambda _ { m } ^ { \prime } }\tag{F.1}
$$

holds deterministically. Here $V _ { m } ^ { u }$ is the same as in Equation (E.4), and

$$
\Lambda _ { m } ^ { \prime } : = 2 \log \left( \frac { 1 } { 2 } \log _ { 2 } m + 3 \right) + n \left( \log \left( \frac { W ^ { 2 } G ^ { 2 } m } { 4 9 n H ^ { 2 } } + 1 \right) + 1 \right)\tag{F.2}
$$

is the quantity obtained from Equation (E.5) by replacing m¯ with the actual number of rounds m.

Lemma F.2 (Monotonicity of the potential including the not-yet-created experts). In the setting of Proposition F.1, adopt the convention that a not-yet-created $( j < j _ { i } )$ η -expert outputs the point of the master $( w _ { j } ^ { \eta _ { i } } : = w _ { j } )$ , and set $\begin{array} { r } { \mathcal { L } _ { j } ^ { i } : = \sum _ { j ^ { \prime } = 1 } ^ { j } \ell _ { j ^ { \prime } } ^ { \eta _ { i } } ( w _ { j ^ { \prime } } ^ { \eta _ { i } } ) } \end{array}$ (with $\mathcal { L } _ { 0 } ^ { i } : = 0 )$ and $\begin{array} { r } { \Phi _ { j } : = \sum _ { i > 0 } p _ { i } \exp ( - \mathcal { L } _ { j } ^ { i } ) } \end{array}$ . Then $\Phi _ { m } \leq \Phi _ { 0 } = 1$ , and in particular, for every $i \in \mathbb { Z } _ { \geq 0 } , - \mathcal { L } _ { m } ^ { i } \leq \log ( ( i + 1 ) ( i + 2 ) )$

Proof. Set $\tilde { p } _ { j } ^ { i } : = p _ { i } \exp ( - \mathcal { L } _ { j - 1 } ^ { i } )$ . The point of the master is defined as $w _ { j } =$ $\big ( \textstyle \sum _ { i : j \ge j _ { i } } \eta _ { i } \tilde { p _ { j } ^ { i } } w _ { j } ^ { \eta _ { i } } \big ) / \big ( \textstyle \sum _ { i : j \ge j _ { i } } \overline { { \eta _ { i } } } \tilde { p } _ { j } ^ { i } \big )$ , a sum ranging over the already created $\eta _ { i ^ { - } }$ experts only; we first show that this equals the sum over the whole grid,

$$
w _ { j } = \frac { \sum _ { i \geq 0 } \eta _ { i } \tilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } } { \sum _ { i \geq 0 } \eta _ { i } \tilde { p } _ { j } ^ { i } } .\tag{F.3}
$$

For a not-yet-created i (that is, $j < j _ { i } )$ , from $w _ { i ^ { \prime } } ^ { \eta _ { i } } = w _ { j ^ { \prime } } \ ( j ^ { \prime } < j _ { i } )$ and the definition of the surrogate loss we have $\ell _ { j ^ { \prime } } ^ { \eta _ { i } } ( w _ { j ^ { \prime } } ^ { \eta _ { i } } ) = 0$ , so $\tilde { p } _ { j } ^ { i } = p _ { i }$ and $w _ { j } ^ { \eta _ { i } } = w _ { j }$ . Hence

$$
\sum _ { i \geq 0 } \eta _ { i } \widetilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } = \sum _ { i : j \geq j _ { i } } \eta _ { i } \widetilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } + w _ { j } \sum _ { i : j < j _ { i } } \eta _ { i } \widetilde { p } _ { j } ^ { i } = w _ { j } \sum _ { i \geq 0 } \eta _ { i } \widetilde { p } _ { j } ^ { i }
$$

(the second equality by the definition of $w _ { j } )$ , which gives Equation (F.3). Here each series converges absolutely by $\begin{array} { r } { \eta _ { i } \le \frac { 1 } { 5 H } , \tilde { p } _ { j } ^ { i } \le p _ { i } , \sum _ { i \ge 0 } p _ { i } = 1 } \end{array}$ and the boundedness of W. Next we show $\Phi _ { j } \leq \Phi _ { j - 1 }$ for each $j .$ Setting $x _ { i } : = \eta _ { i } \langle w _ { j } - w _ { j } ^ { \eta _ { i } } , g _ { j } \rangle$ , we have $\ell _ { j } ^ { \eta _ { i } } ( w _ { j } ^ { \eta _ { i } } ) = - x _ { i } + x _ { i } ^ { 2 }$ and $\begin{array} { r } { | x _ { i } | \le \eta _ { i } H \le \frac { 1 } { 5 } } \end{array}$ . From the elementary inequality $e ^ { x - x ^ { 2 } } \leq 1 + x$ , valid for $x \ \geq \ - { \frac { 1 } { 2 } }$ (because $f ( x ) : = \log ( 1 + x ) - x + x ^ { 2 }$ has $\textstyle f ^ { \prime } ( x ) = { \frac { x ( 2 x + 1 ) } { 1 + x } }$ and hence attains its minimum value 0 at $x = 0 )$ , we obtain

$$
\Phi _ { j } = \sum _ { i \geq 0 } \tilde { p } _ { j } ^ { i } e ^ { x _ { i } - x _ { i } ^ { 2 } } \leq \sum _ { i \geq 0 } \tilde { p } _ { j } ^ { i } ( 1 + x _ { i } ) = \Phi _ { j - 1 } + \left. w _ { j } \sum _ { i \geq 0 } \eta _ { i } \tilde { p } _ { j } ^ { i } - \sum _ { i \geq 0 } \eta _ { i } \tilde { p } _ { j } ^ { i } w _ { j } ^ { \eta _ { i } } , g _ { j } \right. = \Phi _ { j - 1 } ,
$$

the last equality by Equation (F.3). The final claim follows from $\begin{array} { r } { \Phi _ { 0 } = \sum _ { i \geq 0 } p _ { i } = } \end{array}$ $\begin{array} { r } { \sum _ { i \geq 0 } ( \frac { 1 } { i + 1 } - \frac { 1 } { i + 2 } ) = 1 } \end{array}$ and $p _ { i } e ^ { - \mathscr { L } _ { m } ^ { i } } \leq \Phi _ { m } \leq 1$ □

Proof of Proposition F.1. We check the consistency of the creation schedule: for integers $j , i$ we have $\begin{array} { r } { \left[ \frac 1 2 \log _ { 2 } j \right] \geq i \iff \frac 1 2 \log _ { 2 } j > i - 1 \iff j > 4 ^ { i - 1 } \iff j \geq 1 } \end{array}$ $4 ^ { i - 1 } + 1 = j _ { i } \ ( \mathrm { f o r \ } i = \bar { 0 }$ this is always true, corresponding to $j _ { 0 } = 1 )$ ).

Take any $i \in \mathbb { Z } _ { > 0 }$ and $u \in \mathcal W$ , and set $\eta : = \eta _ { i }$ and $a _ { j } : = \left. w _ { j } - u , g _ { j } \right. \left( \mathrm { s o } \left| a _ { j } \right| \leq H \right)$ Summing $- \ell _ { j } ^ { \eta } ( u ) \bar { = } \eta a _ { j } - \eta ^ { 2 } a _ { j } ^ { 2 }$ over $j = 1 , \ldots , m$

$$
\sum _ { j = 1 } ^ { m } a _ { j } = \frac { 1 } { \eta } \sum _ { j = 1 } ^ { m } \left( - \ell _ { j } ^ { \eta } ( u ) \right) + \eta V _ { m } ^ { u } .\tag{F.4}
$$

We decompose the sum in the first term on the right-hand side under the convention of Lemma F.2:

$$
\sum _ { j = 1 } ^ { m } ( - \ell _ { j } ^ { \eta } ( u ) ) = ( - \mathcal { L } _ { m } ^ { i } ) + \sum _ { j < j _ { i } } \left( \ell _ { j } ^ { \eta } ( w _ { j } ) - \ell _ { j } ^ { \eta } ( u ) \right) + \sum _ { j = j _ { i } } ^ { m } \left( \ell _ { j } ^ { \eta } ( w _ { j } ^ { \eta } ) - \ell _ { j } ^ { \eta } ( u ) \right) .
$$

The first term is at most $\log ( ( i + 1 ) ( i + 2 ) ) \leq 2 \log ( i + 2 )$ by Lemma F.2. The second term is at most $\eta H ( j _ { i } - 1 ) = \eta H 4 ^ { i - 1 } ~ ( \mathrm { a n d } ~ 0$ for $i = 0 )$ , by $\ell _ { j } ^ { \eta } ( w _ { j } ) = 0$ and $- \ell _ { i } ^ { \eta } ( u ) \leq \eta a _ { j } \leq \eta H$ . The third term is the regret of the η -expert created at round $j _ { i }$ when run on the surrogate losses $\ell _ { j _ { i } } ^ { \eta } , \ldots , \ell _ { m } ^ { \eta }$ . Since this is a run of length $q : = m - j _ { i } + 1$ with shifted indices, it is at most $B _ { q }$ by Proposition $\mathrm { E . 4 , }$ and hence at most $B _ { m }$ by $q \leq m$ and the monotonicity of $q \mapsto B _ { q }$ . Substituting into Equation $\left( \mathrm { F . 4 } \right)$ and using the identity $\begin{array} { r } { H 4 ^ { i - 1 } = \frac { 1 } { 1 0 0 H \eta _ { i } ^ { 2 } } } \end{array}$ , which follows from $\begin{array} { r } { \eta _ { i } = \frac { 2 ^ { - i } } { 5 H } } \end{array}$ , we obtain

$$
\sum _ { j = 1 } ^ { m } a _ { j } \le \frac { 2 \log ( i + 2 ) + B _ { m } } { \eta _ { i } } + \frac { 1 } { 1 0 0 H \eta _ { i } ^ { 2 } } + \eta _ { i } V _ { m } ^ { u }\tag{F.5}
$$

(for $i = 0$ the middle term may be replaced by 0). We distinguish cases according to $\eta ^ { * } : = \sqrt { \Lambda _ { m } ^ { \prime } / V _ { m } ^ { u } }$

Case 1 $( \eta ^ { * } \ge \frac { 1 } { 5 H }$ , that is, $V _ { m } ^ { u } \leq 2 5 H ^ { 2 } \Lambda _ { m } ^ { \prime } )$ : we use Equation (F.5) with $i = 0$ From 2 log $2 + B _ { m } \leq \Lambda _ { m } ^ { \prime }$ (because the first term of Equation (F.2) is at least 2 log $3 \geq 2 \log 2 )$ ,

$$
\sum _ { j = 1 } ^ { m } a _ { j } \le 5 H \Lambda _ { m } ^ { \prime } + \frac { V _ { m } ^ { u } } { 5 H } \le 1 0 H \Lambda _ { m } ^ { \prime } .
$$

Case $\begin{array} { r } { 2 \ ( \eta ^ { * } < \frac { 1 } { 5 H } ) : } \end{array}$ taking the largest i<sup>∗</sup> with $\eta _ { i ^ { * } } \geq \eta ^ { * }$ , we have $\eta ^ { \ast } \leq \eta _ { i ^ { \ast } } < 2 \eta ^ { \ast }$ since the ratio of the grid is $\frac { 1 } { 2 }$ . From $V _ { m } ^ { u } \le H ^ { 2 } m$ and $\Lambda _ { m } ^ { \prime } \geq 1$ we get $\begin{array} { r } { \eta ^ { * } \ge \frac { 1 } { H \sqrt { m } } } \end{array}$ so $\begin{array} { r } { \frac { 2 ^ { - i ^ { * } } } { 5 H } \geq \frac { 1 } { H \sqrt { m } } } \end{array}$ gives $\begin{array} { r } { i ^ { * } \leq \frac { 1 } { 2 } \log _ { 2 } } \end{array}$ m and hence $\begin{array} { r } { 2 \log ( i ^ { * } + 2 ) + B _ { m } \leq 2 \log ( \frac { 1 } { 2 } \log _ { 2 } m + } \end{array}$ $2 ) + B _ { m } \leq \Lambda _ { m } ^ { \prime }$ . Moreover $\begin{array} { r } { \frac { 1 } { H \eta _ { i ^ { * } } ^ { 2 } } \leq \frac { 1 } { 1 0 0 H ( \eta ^ { * } ) ^ { 2 } } = \frac { V _ { m } ^ { u } } { 1 0 0 H \Lambda _ { m } ^ { \prime } } } \end{array}$ . Thus Equation (F.5) 100 gives

$$
\begin{array} { c l c r } { \displaystyle \sum _ { j = 1 } ^ { m } a _ { j } \le \displaystyle \frac { \Lambda _ { m } ^ { \prime } } { \eta _ { i ^ { * } } } + \eta _ { i ^ { * } } V _ { m } ^ { u } + \displaystyle \frac { V _ { m } ^ { u } } { 1 0 0 H \Lambda _ { m } ^ { \prime } } \le \displaystyle \frac { \Lambda _ { m } ^ { \prime } } { \eta ^ { * } } + 2 \eta ^ { * } V _ { m } ^ { u } + \displaystyle \frac { V _ { m } ^ { u } } { 1 0 0 H \Lambda _ { m } ^ { \prime } } } \\ { \displaystyle } & { = 3 \sqrt { \Lambda _ { m } ^ { \prime } V _ { m } ^ { u } } + \displaystyle \frac { V _ { m } ^ { u } } { 1 0 0 H \Lambda _ { m } ^ { \prime } } . } \end{array}
$$

In either case Equation (F.1) holds.

Theorem F.3. Under Assumption 3.1, run growing-grid SGS-MetaGrad $\mathrm { ( A l g o \mathrm { - } }$ rithm 6) on an arbitrary sequence of states $\{ s ^ { t } \} _ { t = 1 } ^ { T }$ . Then the following hold.

(i): (K)

$$
K \leq d + \frac { 5 2 L D } { \gamma } \left( 2 \log ( \log d + 3 ) + d \right) + \frac { 5 2 L D ( d + 2 ) } { \gamma } \log \operatorname* { m a x } \left( \frac { 5 2 L D ( d + 2 ) } { \gamma d } , 1 \right) .\tag{F.6}
$$

$$
R _ { T } ^ { \mathrm { s u b } } \leq 2 6 L D \left( 2 \log ( \log d + 3 ) + d + ( d + 2 ) \log \operatorname* { m a x } \left( \frac { 2 6 L D ( d + 2 ) } { \gamma d } , 1 \right) \right)\tag{F.7}
$$

(iii): (Re<sub>T</sub>) With c<sub>0</sub>(K) := 2 log( <sup>1</sup> log<sub>2</sub> K + 3) and $K _ { \mathrm { m a x } }$ the right-hand side of Equation (F.6),

$$
\begin{array} { l } { \displaystyle { \widetilde { R } _ { T } \leq 2 6 L D \left( c _ { 0 } ( K _ { \operatorname* { m a x } } ) + d \left( \log \left( 1 + \frac { K _ { \operatorname* { m a x } } } { 4 9 d } \right) + 1 \right) \right) } } \\ { \displaystyle { \qquad = O \left( L D d \log \operatorname* { m a x } \left( \frac { L D } { \gamma } , 2 \right) \right) . } } \end{array}\tag{F.8}
$$

In particular, none of the right-hand sides depends on the total number of rounds T at all, and the doubly logarithmic factor involves only the dimension $d ,$ not $T .$ Moreover, by Equation (3.8), the cumulative decision regret $R _ { T } ^ { \mathrm { { e s t } } }$ also has the same upper bound as Equation (F.8).

Proof of Theorem F.3. If $K = 0$ everything is trivial, so assume $K \geq 1$ . As in the proof of Theorem E.1, Algorithm 6 is nothing but growing-grid MetaGrad (with $n = d , \mathcal { W } = \Theta , W = D , G = L , H = L D )$ applied to the sequence of convex losses of the mistake rounds (of length $m = K )$ , and the verification of the subgradients and of G, H is identical (the agreement of the creation schedule being the content of the beginning of the proof of Proposition F.1). Noting that $W ^ { 2 } G ^ { 2 } / H ^ { 2 } = 1$ and applying Proposition F.1 with $u = { \bar { \theta } } _ { : }$ , we obtain

$$
R : = \sum _ { k = 1 } ^ { K } r _ { t _ { k } } \leq 3 \sqrt { \Lambda _ { K } V } + 1 0 L D \Lambda _ { K } + \frac { V } { 1 0 0 L D \Lambda _ { K } } , \qquad V : = \sum _ { k = 1 } ^ { K } r _ { t _ { k } } ^ { 2 } ,
$$

where we have set $c _ { 0 } ( K ) : = 2 \log ( \textstyle { \frac { 1 } { 2 } } \log _ { 2 } K + 3 )$ and $\Lambda _ { K } : = c _ { 0 } ( K ) + d ( \log ( 1 + { \frac { K } { 4 9 d } } ) + 1 )$ (the quantity obtained from Equation (F.2) with $n = d , W = D , G = L , \tilde { H } = L D$ and $m = K )$

Step 1. (Estimate of $\textstyle \sum _ { k } r _ { t _ { k } } )$ Since Lemma C.1 gives $0 < r _ { t _ { k } } \le L D$ , we have $V \leq L D \cdot R .$ , and from $\Lambda _ { K } \geq 2 \log 3 \geq 1$ the third term is bounded by $\begin{array} { r } { \frac { V } { 1 0 0 L D \Lambda _ { K } } \leq \frac { R } { 1 0 0 } . } \end{array}$ Rearranging,

$$
R \leq \frac { 1 0 0 } { 9 9 } \left( 3 \sqrt { \Lambda _ { K } L D R } + 1 0 L D \Lambda _ { K } \right) = \sqrt { a R } + \zeta
$$

(with $a : = ( \textstyle { \frac { 1 0 0 } { 3 3 } } ) ^ { 2 } \Lambda _ { K } L D$ and $\begin{array} { r } { \zeta : = \frac { 1 0 0 0 } { 9 9 } \Lambda _ { K } L D ) } \end{array}$ , and Equation (E.11) gives

$$
\sum _ { k = 1 } ^ { K } r _ { t _ { k } } = R \le \frac { 4 } { 3 } ( a + \zeta ) = \frac { 4 } { 3 } \left( \frac { 1 0 0 0 0 } { 1 0 8 9 } + \frac { 1 0 0 0 } { 9 9 } \right) L D \Lambda _ { K } \le 2 6 L D \Lambda _ { K } .
$$

(i) We first estimate $c _ { 0 } ( K )$ . Using $\begin{array} { r } { \frac { 1 } { 2 } \log _ { 2 } K \leq } \end{array}$ log K (for $K \geq 1$ , since $\textstyle { \frac { 1 } { 2 \log 2 } } \leq 1 )$ 2 log $\begin{array} { r } { K \leq \log d + \log ( 1 + \frac { K } { d } ) } \end{array}$ , and $a ^ { \prime } + \zeta ^ { \prime } \leq a ^ { \prime } ( 1 + \zeta ^ { \prime } ) ~ ( \mathrm { f o r } ~ a ^ { \prime } \geq 1 , \zeta ^ { \prime } \geq 0 )$ with

a<sup>′</sup> = log d + 3 and $\begin{array} { r } { \zeta ^ { \prime } = \log ( 1 + \frac { K } { d } ) } \end{array}$ , we obtain

$$
\begin{array} { r l r } {  { c _ { 0 } ( K ) \leq 2 \log \big ( \log d + 3 + \log \big ( 1 + \frac { K } { d } \big ) \big ) \leq 2 \log ( \log d + 3 ) + 2 \log \big ( 1 + \log \big ( 1 + \frac { K } { d } \big ) \big ) } } \\ & { } & { \leq 2 \log ( \log d + 3 ) + 2 \log \big ( 1 + \frac { K } { d } \big ) ~ ( \mathrm { F . 9 } ) } \end{array}
$$

(the last inequality by $\log ( 1 + x ) \leq x )$ . From $\gamma K \leq R$ of Lemma C.1 and Step 1, together with $\begin{array} { r } { \log ( 1 + \frac { K } { 4 9 d } ) \leq \log ( 1 + \frac { K } { d } ) } \end{array}$ and Equation (F.9), the quantity $y : = K / d$ satisfies

$$
y \leq c _ { 1 } + c _ { 2 } \log ( 1 + y ) , \qquad c _ { 1 } : = \frac { 2 6 L D ( 2 \log ( \log d + 3 ) + d ) } { \gamma d } , \quad c _ { 2 } : = \frac { 2 6 L D ( d + 2 ) } { \gamma d } .
$$

Lemma D.3 gives $y \le 2 c _ { 1 } + 1 + 2 c _ { 2 }$ log $\operatorname* { m a x } ( 2 c _ { 2 } , 1 )$ , and multiplying both sides by d gives Equation (F.6).

(ii) As in the proof of Theorem D.6(ii) we have $\begin{array} { r } { R _ { T } ^ { \mathrm { s u b } } = \sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } \leq R - \gamma K } \end{array}$ , and by (i) and Equation (F.9),

$$
\begin{array} { c l c r } { { \displaystyle \sum _ { k = 1 } ^ { K } \ell ^ { t _ { k } } \le 2 6 L D ( 2 \log ( \log d + 3 ) + d ) + \displaystyle \operatorname* { m a x } _ { x \ge 0 } \psi ( x ) , } } \\ { { \displaystyle \psi ( x ) : = 2 6 L D ( d + 2 ) \log \left( 1 + \displaystyle \frac { x } { d } \right) - \gamma x . } } \end{array}
$$

Since $\begin{array} { r } { \psi ^ { \prime } ( x ) = \frac { 2 6 L D ( d + 2 ) } { d + x } - \gamma , \mathrm { ~ i f ~ } \gamma d \ge 2 6 L D ( d + 2 ) } \end{array}$ then ψ is nonincreasing and $\begin{array} { r } { \operatorname* { m a x } _ { x \geq 0 } \psi = \psi ( 0 ) = 0 ; } \end{array}$ otherwise, at the stationary point $\begin{array} { r } { x ^ { * } = { \frac { 2 6 L D ( d + 2 ) } { \gamma } } - d > 0 } \end{array}$

$$
\begin{array} { l } { \displaystyle \psi ( \boldsymbol { x } ^ { * } ) = 2 6 L D ( d + 2 ) \log \frac { 2 6 L D ( d + 2 ) } { \gamma d } - 2 6 L D ( d + 2 ) + \gamma d } \\ { \displaystyle \qquad \leq 2 6 L D ( d + 2 ) \log \frac { 2 6 L D ( d + 2 ) } { \gamma d } . } \end{array}
$$

In either case ma $\begin{array} { r } { \mathfrak { c } _ { x \ge 0 } \psi \le 2 6 L D ( d + 2 ) \log \operatorname* { m a x } ( \frac { 2 6 L D ( d + 2 ) } { \gamma d } , 1 ) } \end{array}$ , so we obtain Equation (F.7).

(iii) Since Proposition F.1 holds for every $u \in \mathcal W ,$ , we apply it with $u = \theta ^ { * }$ (where $\theta ^ { * } \in \Theta = \mathcal { W }$ by Assumption $3 . 1 ( 3 ) $ . We have $\left. w _ { k } - \theta ^ { * } , g _ { k } \right. = \widetilde { r } _ { t _ { k } }$ , and the contribution of the rounds without a mistake is 0, so the inequality at the beginning of the proof with $r _ { t _ { k } }$ replaced by $\widetilde { r } _ { t _ { k } }$

$$
\widetilde { R } _ { T } = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } \le 3 \sqrt { \Lambda _ { K } \widetilde { V } } + 1 0 L D \Lambda _ { K } + \frac { \widetilde { V } } { 1 0 0 L D \Lambda _ { K } } , \qquad \widetilde { V } : = \sum _ { k = 1 } ^ { K } \widetilde { r } _ { t _ { k } } ^ { 2 } ,
$$

holds. Since $0 \le \widetilde { r } _ { t _ { k } } \le L D$ from Equation (C.3) gives $ { \widetilde { V } } \leq L D  { \widetilde { R } } _ { T }$ , and $\Lambda _ { K } \geq 1$ bounds the third term by $\frac { \widetilde { R } _ { T } } { 1 0 0 }$ , rearranging gives

$$
\widetilde { R } _ { T } \leq \frac { 1 0 0 } { 9 9 } \left( 3 \sqrt { \Lambda _ { K } L D \widetilde { R } _ { T } } + 1 0 L D \Lambda _ { K } \right) = \sqrt { a \widetilde { R } _ { T } } + \zeta
$$

(with $a = ( \textstyle { \frac { 1 0 0 } { 3 3 } } ) ^ { 2 } \Lambda _ { K } L D$ and $\begin{array} { r } { \zeta = \frac { 1 0 0 0 } { 9 9 } \Lambda _ { K } L D ) } \end{array}$ . By Equation (E.11) we get $\tilde { R } _ { T } \leq$ ${ \textstyle \frac { 4 } { 3 } } ( a + \zeta ) \leq 2 6 L D \Lambda _ { \cal K }$ . Since $\Lambda _ { K }$ is monotonically increasing in K, substituting $\breve { K } \leq K _ { \mathrm { m a x } }$ from (i) gives the explicit form of Equation (F.8). The order expression follows from Equation (F.9) and $\begin{array} { r } { K _ { \operatorname* { m a x } } = O ( \frac { d L \hat { D } } { \gamma } \log \operatorname* { m a x } ( \frac { 2 \acute { L } D } { \gamma } , 2 ) ) } \end{array}$ . □

## Appendix G. Lower bounds on the margin by structure

In this appendix we collect the statements of the lower bounds on $\gamma _ { \mathrm { s u b } }$ summarized in §6 (including the case where Θ is the unit ball).

We set the sets of points

$$
Z ( s ) : = \{ x ^ { * } ( \theta ^ { * } , s ) - x \mid x \in X ( s ) \setminus \{ x ^ { * } ( \theta ^ { * } , s ) \} \} , \qquad Z ^ { * } : = \bigcup _ { s \in S } Z ( s )\tag{G.1}
$$

(the diferences being taken over all of $X ( s )$ , not over $Y ( s ) )$ . By the minimax theorem (Proposition H.2 in Appendix H),

$$
\operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { i n f } _ { s \in S } \operatorname* { m i n } _ { z \in Z ( s ) } \langle \theta , z \rangle = \operatorname* { m i n } _ { z \in \mathrm { C o n v } } \operatorname* { m a x } _ { Z ^ { * } } \langle \theta , z \rangle
$$

holds. Since $Y ( s ) \subseteq X ( s )$ by Assumption $3 . 1 ( 2 )$ , the left-hand side is at most the quantity $\gamma _ { \mathrm { s u b } }$ of Equation (6.1), and hence a lower bound on the right-hand side gives a lower bound on $\gamma _ { \mathrm { s u b } }$ directly. Moreover, under Assumption 6.3 every $z \in Z ^ { * }$ has components satisfying $| z _ { i } | \leq M _ { i }$ , so

$$
L _ { \mathrm { s u b } } \leq \| M \| _ { 2 }\tag{G.2}
$$

(used in the corollaries of §7). Below we quantify the separation of the polyhedron Conv $Z ^ { \ast }$ from the origin.

Theorem G.1 (Explicit lower bound for a general ILP with the unit ball). Assume (1), (2) and (3) of Assumption 3.1 and Assumption 6.3, and let $\Theta = \{ \theta \in \mathbb { R } ^ { d }$ $\| \theta \| \le 1 \}$ (the unit ball) and $d \ge 2$ . Let M be the vector in Equation (6.3). Assume furthermore that Conv $Z ^ { * }$ is full-dimensional (dim Conv $Z ^ { * } = d )$ (for the low-dimensional case see Proposition G.2 and Remark G.3). Then

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 1 } { 2 ^ { d - 1 } \sqrt { d - 1 } \| M \| _ { 2 } ^ { d - 1 } } .
$$

Proposition G.2 (The low-dimensional case: when the afine hull does not contain the origin). Assume (1), (2) and (3) of Assumption 3.1 and Assumption 6.3, let Θ be the unit ball and let $Z ^ { * } \neq \emptyset$ . Set k := dim af $Z ^ { \ast }$ and assume $0 \not \in$ af $Z ^ { \ast }$ . Then $\gamma _ { \mathrm { s u b } } \geq ( 2 \Vert M \Vert _ { 2 } ) ^ { - k }$

Remark G.3 (Summary of the low-dimensional cases). The case dim Conv $Z ^ { * } < \mathfrak { c }$ d is treated as follows. (i) If 0 ∈/ af $Z ^ { * }$ , then Proposition G.2 gives the lower bound $( 2 \| M \| _ { 2 } ) ^ { - k }$ (for $k \leq d - 1$ this has the same dependence on $\| M \| _ { 2 }$ as the fulldimensional lower bound of Theorem G.1 and is stronger by the absence of the factor $\sqrt { d - 1 } )$ . (ii) In the case $0 \in$ af $Z ^ { \ast }$ with $k < d ,$ an isomorphic argument within the lattice induced on af $Z ^ { \ast }$ is required, but since the construction of an integral normal vector and the estimate of its norm depend on the norms of the (dual) basis of the induced lattice, a uniform constant of the type $( 2 \| M \| _ { 2 } ) ^ { k }$ does not follow immediately from our method. We leave the quantitative lower bound in this case as unresolved (status: unknown). (iii) In the case of the probability simplex (Theorem G.4), the assumption of full-dimensionality is unnecessary, since the proof goes through the polyhedron Conv $Z ^ { * } + \mathbb { R } _ { \geq 0 } ^ { d }$ , which is always full-dimensional. (iv) The same summary as in (i) and (ii) holds for the full-dimensionality assumption dim Conv $( S ^ { + } ) = d$ of Theorem G.19 (replacing $2 \| M \| _ { 2 }$ by twice the upper bound on the norms of the vertices, that is, by $2 C _ { g } \sqrt { d } )$

Theorem G.4 (Explicit lower bound for a general ILP with the probability simplex). Assume (1), (2) and (3) of Assumption 3.1 and Assumption 6.3, and let $\Theta = \Delta ^ { d - 1 } =$ $\{ \theta \in \mathbb { R } _ { > 0 } ^ { d } : \textstyle \sum _ { i = 1 } ^ { d } \theta _ { i } = 1 \}$ (the probability simplex). Let M be the vector in Equation (6.3). Then

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 1 } { 2 ^ { d - 1 } \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) \Vert M \Vert _ { 2 } ^ { d - 1 } } .
$$

No assumption of full-dimensionality is needed.

See Appendix J and Appendix K respectively for the proofs.

G.1. General theory of lower bounds via test sets. The lower bounds for discrete convex structures are obtained uniformly through test sets (defined below).

Definition G.5 (Test set). For a bounded discrete set $X \subset \mathbb { Z } ^ { d }$ (which is finite by boundedness), a finite set ${ \mathcal { T } } \subset \mathbb { Z } ^ { d } \setminus \{ 0 \}$ is a test set of X if the following holds. For every $x ^ { 1 } \in X$ and every $\theta \in \Theta$ , if $\langle \theta , x ^ { 1 } \rangle < \operatorname* { m a x } _ { x \in X } \langle \theta , x \rangle$ , then there exists $g \in { \mathcal { T } }$ with $x ^ { 1 } + g \in X$ and $\langle \theta , g \rangle > 0$

Proposition G.6 (Decomposition towards an optimal point via a test set; Kitaoka, 2026). Take a bounded discrete set $X \subset \mathbb { Z } ^ { d }$ and a test set T of it. Then, for every $x ^ { 1 } \in X$ and every $\theta \in \Theta$ , there exist $x ^ { * } \in \arg \operatorname* { m a x } _ { x \in X } \langle \theta , x \rangle$ and $g ^ { 1 } , \dotsc , g ^ { r } \in { \mathcal { T } }$ $( r \in \mathbb { Z } _ { \ge 0 } )$ such that

$$
x ^ { * } - x ^ { 1 } = \sum _ { i = 1 } ^ { r } g ^ { i } , \qquad \langle \theta , g ^ { i } \rangle > 0 \quad ( i = 1 , \ldots , r ) .
$$

See Appendix H for the proof.

Below we assume (1), (2) and (3) of Assumption 3.1 and let $\theta ^ { \ast } \in \Theta$ be the true weight. We define the set of weights whose signs are consistent with those of $\theta ^ { * }$ on the test set T by

$$
\begin{array} { r } { \Theta _ { \mathcal { T } } ( \theta ^ { * } ) : = \{ \theta \in \Theta \mid \mathrm { f o r ~ e v e r y ~ } g \in \mathcal { T } , \mathrm { ~ } \langle \theta ^ { * } , g \rangle > 0 \Rightarrow \langle \theta , g \rangle > 0 \} . } \end{array}\tag{G.3}
$$

By definition $\theta ^ { * } \in \Theta _ { T } ( \theta ^ { * } )$

Proposition G.7 (Lower bound on the margin via a test set). Assume that the set $\tau$ is a test set of the discrete set $X ( s )$ for every $s \in S$ . Then

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { s u p } _ { \theta \in \Theta \tau ( \theta ^ { * } ) } \operatorname* { m i n } _ { g \in \mathcal { T } , \langle \theta ^ { * } , g \rangle > 0 } \langle \theta , g \rangle .\tag{G.4}
$$

See Appendix H for the proof.

G.2. Definitions from discrete convex analysis and Graver bases. In this subsection we collect the definitions, taken from the cited references, that were used in §6. For $x \in \mathbb { Z } ^ { d }$ we set $\operatorname { s u p p } ^ { + } ( x ) : = \{ i \mid x _ { i } > 0 \}$ and $\operatorname { s u p p } ^ { - } ( x ) : = \{ i \mid x _ { i } < 0 \}$

Definition G.8 (M-convex set (Murota, 2003)). A set $X \subseteq \mathbb { Z } ^ { d }$ is an M-convex set if, for every $x , y \in X$ and every $i \in \operatorname { s u p p } ^ { + } ( x - y )$ , there exists $j \in \operatorname { s u p p } ^ { - } ( x - y )$ such that $x - e _ { i } + e _ { j } \in X$ and $y + e _ { i } - e _ { j } \in X$ (the exchange axiom). All elements of an M-convex set have the same coordinate sum $\sum _ { i } x _ { i }$ , and M-convex sets coincide with the sets of integer points of integral base polyhedra.

Definition G.9 (M<sup>♮</sup>-convex set (cf. Murota, 2003)). We adopt the convention $e _ { 0 } : = 0 \in \mathbb { Z } ^ { d }$ . A set $X \subseteq \mathbb { Z } ^ { d }$ is an M<sup>♮</sup>-convex set if, for every $x , y \in X$ and every $i \in \operatorname { s u p p } ^ { + } ( x - y )$ , there exists $j \in \mathrm { s u p p } ^ { - } ( x - y ) \cup \{ 0 \}$ such that $x - e _ { i } + e _ { j } \in X$ and $y + e _ { i } - e _ { j } \in X$ $\mathrm { M } ^ { \natural } .$ -convex sets are obtained as coordinate projections of M-convex sets, and coincide with the sets of integer points of generalized integral base polyhedra.

Definition G.10 (Graver basis (cf. Onn, 2010)). For a matrix $ { \widetilde { A } } \in  { \mathbb { Z } } ^ { N \times n }$ we set $\ker _ { \mathbb { Z } } ( { \widetilde { A } } ) : = \{ g \in \mathbb { Z } ^ { n } \mid { \widetilde { A } } g = 0 \}$ . Two vectors $g , h \in \mathbb { Z } ^ { n }$ are sign consistent $( g \subseteq h )$ if $g _ { i } h _ { i } \geq 0$ and $| g _ { i } | \leq | h _ { i } |$ hold componentwise. The Graver basis $\mathcal { G } ( \widetilde { A } )$ is the set of all ⊑-minimal elements of $\ker _ { \mathbb { Z } } ( { \widetilde { A } } ) \setminus \{ 0 \}$ (a finite set).

## G.3. Polynomial lower bounds for M-convex and $\mathbf { M } ^ { \natural } .$ -convex structures.

Assumption G.11 (M-convex feasible set). For every $s \in S$ , the set $X ( s ) \subseteq \mathbb { Z } ^ { d }$ is an M-convex set (defined in Appendix G.2).

In this case $X ( s )$ is a set of finitely many integer points, so Assumption 6.3 is satisfied as well.

Proposition G.12 (Test set of an M-convex set; Murota, 1996, 1998, 2003). A test set of an M-convex set can be taken to be the set of single exchange vectors $\mathcal { T } = \{ e _ { i } - e _ { j } \ | \ i \neq j \}$

Theorem G.13 (Polynomial lower bound for M-convex sets). Assume $( 1 ) , ( 2 )$ and (3) of Assumption 3.1 and Assumption G.11. Then, in the case $\Theta = \{ \theta \in \mathbb { R } ^ { d }$ $\| \theta \| _ { 2 } \leq 1 \}$ with $d \geq 2$

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 2 \sqrt { 3 } } { \sqrt { d ( d ^ { 2 } - 1 ) } } = \Omega \biggl ( \frac { 1 } { d ^ { 3 / 2 } } \biggr ) ,
$$

and in the case $\Theta = \Delta ^ { d - 1 }$ with $d \geq 2 .$

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 2 } { d ( d - 1 ) } = \Omega \biggl ( \frac { 1 } { d ^ { 2 } } \biggr ) .
$$

Assumption G.14 (M<sup>♮</sup>-convex feasible set). For every $s \in S$ , the set $X ( s ) \subseteq \mathbb { Z } ^ { d }$ is an M<sup>♮</sup>-convex set (defined in Appendix G.2).

Proposition G.15 (Test set of an $\mathrm { M } ^ { \natural } .$ -convex set; Murota and Shioura, 1999; Murota, 2003). A test set of an $\mathrm { M ^ { \natural } - c o n v e x }$ set can be taken to be $\mathcal { T } = \{ e _ { i } - e _ { j } , \pm e _ { i } | i \neq j \}$

Theorem G.16 (Polynomial lower bound for $\mathrm { M } ^ { \natural } .$ -convex sets). Assume $( 1 ) , ( 2 )$ and (3) of Assumption 3.1 and Assumption G.14. Then, in the case $\Theta = \{ \theta \in \mathbb { R } ^ { d }$ $\| \theta \| _ { 2 } \leq 1 \}$ with $d \geq 2$

$$
\gamma _ { \mathrm { s u b } } \geq \sqrt { \frac { 6 } { d ( d + 1 ) ( 2 d + 1 ) } } \geq \frac { 1 } { d ^ { 3 / 2 } } = \Omega \left( \frac { 1 } { d ^ { 3 / 2 } } \right) ,
$$

and in the case $\Theta = \Delta ^ { d - 1 }$ with $d \geq 2 .$

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 2 } { d ( d + 1 ) } = \Omega \biggl ( \frac { 1 } { d ^ { 2 } } \biggr ) .
$$

See Appendix L for both proofs.

G.4. General lower bounds for linear inequality constraints: independence of $\| M \| _ { 2 }$ . In this subsection we show, for general linear inequality constraints with integer coeficients, a lower bound determined solely by the $\ell _ { \infty }$ norm of the Graver basis (defined in Appendix G.2) of the coeficient matrix. In particular, the lower bound depends neither on the right-hand side b(s) nor on the range $\| M \| _ { 2 }$ of the features. This is an essential improvement over Theorems G.1 and G.4.

Assumption G.17 (Linear inequality constraints). For a coeficient matrix $A \in$ $\mathbb { Z } ^ { N \times d }$ and right-hand sides $b ( s ) \in \mathbb { Z } ^ { N }$ , let $X ( s ) = \{ x \in \mathbb { Z } ^ { d } \mid A x \leq b ( s ) \}$ (which is bounded by Assumption 3.1(2)).

By introducing slack variables we convert this into the system of equalities $\widetilde { A } \widetilde { x } = b ( s )$ (with $\widetilde { A } = [ A | \mathrm { I d } _ { N } ] \in \mathbb { Z } ^ { N \times ( d + N ) } , \widetilde { x } = ( x , y )$ and $y \geq 0 )$ . We define the $\ell _ { \infty }$ norm of the Graver basis $\mathcal { G } ( \widetilde { A } )$ by $g _ { \infty } ( \widetilde { A } ) : = \operatorname* { m a x } \{ \| g \| _ { \infty } \ | \ g \in \mathcal { G } ( \widetilde { A } ) \} \in \mathbb { Z } _ { \ge 1 }$ This is determined by $\widetilde { A }$ (and hence by A) alone, and depends neither on $b ( s )$ nor on $\| M \| _ { 2 }$

Proposition G.18 (Test set from a Graver basis; Kitaoka, 2026). Under Assumption G.17, setting ${ \mathcal { T } } _ { x } : = \pi _ { x } ( { \mathcal { G } } ( { \widetilde { A } } ) )$ by means of the projection $\pi _ { x } \colon  { \mathbb { Z } } ^ { d + N } \to  { \mathbb { Z } } ^ { d }$ , the set $\mathcal { T } _ { x }$ is a test set of $X ( s )$ for every $s \in S$ and satisfies $\| g \| _ { \infty } \leq g _ { \infty } ( \widetilde { A } )$ for all $g \in \mathcal { T } _ { x }$

Theorem G.19 (Lower bound independent of $\| M \| _ { 2 }$ for linear inequalities with the unit ball). Assume (1), (2) and (3) of Assumption 3.1 and Assumption G.17, let Θ be the unit ball and let $d \geq 2 .$ . Set $C _ { g } : = g _ { \infty } ( \widetilde { A } )$ and assume that the convex hull of $S ^ { + } : = \{ g \in \mathcal { T } _ { x } \mid \langle \theta ^ { * } , g \rangle > 0 \}$ is full-dimensional (dim $\operatorname { C o n v } ( S ^ { + } ) = d )$ . Then

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 1 } { \sqrt { d - 1 } ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } } .
$$

In particular, the lower bound depends neither on $b ( s )$ nor on $\| M \| _ { 2 }$ . If A is a totally unimodular matrix then $C _ { g } = 1$

Theorem G.20 (Lower bound independent of $\| M \| _ { 2 }$ for linear inequalities with the probability simplex). Assume (1), (2) and (3) of Assumption 3.1 and Assumption G.17, and let $\Theta = \Delta ^ { d - 1 }$ and $d \geq 2$ . Set $C _ { g } : = g _ { \infty } ( \widetilde { A } )$ . Then

$$
\gamma _ { \mathrm { s u b } } \geq \frac { 1 } { \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } } .
$$

In particular, the lower bound depends neither on $b ( s )$ nor on $\| M \| _ { 2 }$ . If A is a totally unimodular matrix then $C _ { g } = 1$

See Appendix M for both proofs. Whereas the existing ILP lower bounds (Theorems G.1 and G.4) depend strongly on the range m of the features, the lower bounds of this subsection depend only on $g _ { \infty } ( \widetilde { A } )$ , a quantity determined by the coeficient matrix alone. Comparing the denominators of the two lower bounds (in the case of the unit ball), the ratio is $( C _ { g } \sqrt { d } / \| M \| _ { 2 } ) ^ { d - 1 }$ , so for problems with large transportation amounts, demands or capacities but a simple structure of the coeficient matrix (a small $g _ { \infty } ( \widetilde { A } ) )$ ), the improvement factor is roughly $( \| M \| _ { 2 } / ( C _ { g } \sqrt { d } ) ) ^ { d - 1 }$

Remark G.21 (The M-convex and $\mathrm { M } ^ { \natural }$ -convex lower bounds are not corollaries of this subsection). An M<sup>♮</sup>-convex set is the set of integer points of a generalized integral base polyhedron, and that polyhedron is described by submodular and supermodular inequalities over a family of subsets. The corresponding coeficient matrix (the matrix whose rows are the indicator vectors of the subsets) is in general not totally unimodular (for instance, the rows $( 1 , 1 , 0 ) , ( 0 , 1 , 1 ) , ( 1 , 0 , 1 )$ form a square submatrix of determinant ±2). Therefore Theorems G.13 and G.16 cannot in general be derived as corollaries of the results of this subsection with $C _ { g } = 1$ They are obtained directly from the fact that the test sets can be taken explicitly as $\{ e _ { i } - e _ { j } \}$ and $\{ e _ { i } - e _ { j } , \pm e _ { i } \}$ , and as lower bounds they are, at $\Omega ( d ^ { - 3 / 2 } )$ and $\Omega ( d ^ { - 2 } )$ far better than the $( 2 C _ { g } \sqrt { d } ) ^ { - ( d - 1 ) } \mathrm { - t y p e }$ bounds of this subsection.

## Appendix H. Proofs for §6, I: general theory

Proof of Proposition G.6. Since $X \subset \mathbb { Z } ^ { d }$ is bounded, it is a finite set. We construct a sequence of points $\{ x ^ { ( k ) } \} _ { k \geq 0 }$ inductively as follows: set $x ^ { ( 0 ) } : = x ^ { 1 }$ , and as long as $x ^ { ( k ) } \in X$ satisfies $\langle \theta , x ^ { ( k ) } \rangle < \operatorname* { m a x } _ { x \in X } \langle \theta , x \rangle$ , take, by Definition G.5, some $g ^ { ( k + 1 ) } \in \mathcal { T }$ with $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + g ^ { ( k + 1 ) } \in X$ and $\langle \theta , g ^ { ( k + 1 ) } \rangle > 0$ . Then $\langle \theta , x ^ { ( k ) } \rangle$ is strictly increasing, so $x ^ { ( 0 ) } , x ^ { ( 1 ) } , \dotsc$ . are pairwise distinct, and by the finiteness of X the construction terminates after finitely many steps $r \geq 0$ . The terminal point does not satisfy the continuation condition, that is, ${ \bf \langle \theta , } \boldsymbol { x } ^ { ( r ) } { \rangle } = \operatorname* { m a x } _ { x \in X } \langle \theta , \boldsymbol { x } \rangle$ 2 so $x ^ { ( r ) } \in \arg \operatorname* { m a x } _ { x \in X } \langle \theta , x \rangle$ . Setting $x ^ { * } : = x ^ { ( r ) }$ and $g ^ { i } : = g ^ { ( i ) } \ ( i = 1 , \ldots , r )$ , the construction gives $\bar { \boldsymbol { x } } ^ { ( k + 1 ) } - \boldsymbol { x } ^ { ( k ) } = \boldsymbol { g } ^ { ( k + 1 ) }$ , so summing over $k = 0 , \ldots , r - 1$ yields $\begin{array} { r } { x ^ { * } - x ^ { 1 } = \sum _ { i = 1 } ^ { r } g ^ { i } \left( \mathrm { i f } x ^ { 1 } \right. } \end{array}$ is already a maximizer then $r = 0$ and the sum is empty). □

Proof of Lemma 6.1. Consider the function

$$
\varphi ( \theta ) : = \operatorname* { i n f } _ { s \in S } \operatorname* { m i n } _ { x \in Y ( s ) \backslash \{ x ^ { * } ( \theta ^ { * } , s ) \} } \langle \theta , x ^ { * } ( \theta ^ { * } , s ) - x \rangle .
$$

Since $Y ( s )$ is a finite set for each s by Assumption $3 . 1 ( 2 )$ , the inner min is the minimum of finitely many linear functions, and hence concave and $L _ { \mathrm { s u b } } \mathrm { - L i p s c h i t z }$ Here, for a compact set $X ( s )$ the extreme points of Conv $X ( s )$ belong to $X ( s )$ , so $Y ( s ) \subseteq X ( s )$ , and the norms of the gradients are bounded by $\| x ^ { * } ( \theta ^ { * } , s ) - x \| \leq L _ { \mathrm { s u b } }$ (Equation (6.2)). Therefore $\varphi _ { ; }$ , being the infimum of those functions over $s ,$ is concave and $L _ { \mathrm { s u b } } \mathrm { - L i p s c h i t z }$ (note that $\varphi ( \theta ) \geq - L _ { \mathrm { { s u b } } } \Vert \theta \Vert > - \infty$ at every point), and it attains its maximum value $\gamma _ { \mathrm { s u b } }$ on the bounded closed set Θ. Let <sup>¯</sup>θ be a maximizer.

Assume $\gamma _ { \mathrm { s u b } } > 0$ and let us verify (4) and (5) of Assumption 3.1 ((1), (2) and (3) being assumptions). (4): the identity $\varphi ( \bar { \theta } ) = \gamma _ { \mathrm { s u b } }$ is exactly Equation (3.9). (5): since the definitions Equation (3.2) and Equation (6.2) are identical we have $L = L _ { \mathrm { s u b } }$ , and by assumption $0 < L = L _ { \mathrm { s u b } } < \infty$ □

The following two propositions are used for the minimax expression of Equation (6.1).

Proposition H.1. Let $A \subset \mathbb { R } ^ { n }$ be a bounded set. Then Conv $\overline { { A } } = \mathrm { C o n v } \overline { { A } }$

Proof. (⊃) From $A \subset { \overline { { A } } }$ we get Conv $A \subset \operatorname { C o n v } \overline { { A } }$ . Taking the closures of both sides gives ${ \overline { { \operatorname { C o n v } A } } } \subset \operatorname { C o n v } { \overline { { A } } }$ . Since A is bounded, A is compact, and in a finitedimensional space the convex hull of a compact set is compact (by Carathéodory’s theorem, Conv A is the image of the compact set $\Delta _ { n } \times \overline { { A } } ^ { n + 1 }$ under the continuous map $\textstyle ( \lambda , x _ { 0 } , \dotsc , x _ { n } ) \mapsto \sum _ { i = 0 } ^ { n } \lambda _ { i } x _ { i } )$ . Hence Conv $\overline { { A } }$ is closed, so $\overline { { \mathrm { C o n v } \overline { { A } } } } = \mathrm { C o n v } \overline { { A } }$ and therefore Conv ${ \overline { { A } } } \subset$ Conv A.

(⊂) The set Conv A is a closed convex set containing A. Since A is the smallest closed set containing A, we have ${ \overline { { A } } } \subset { \overline { { \operatorname { C o n v } { A } } } }$ , and since $\overline { { \mathrm { C o n v } A } }$ is convex, Conv ${ \overline { { A } } } \subset$ Conv A. □

Proposition H.2. Let Θ be a bounded closed convex set and let $A \subset  { \mathbb { R } } ^ { d }$ be a bounded closed set. Then

$$
\operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { m i n } _ { a \in A } \langle \theta , a \rangle = \operatorname* { m i n } _ { a \in \operatorname { C o n v } A } \operatorname* { m a x } _ { \theta \in \Theta } \langle \theta , a \rangle .
$$

Proof. Since A is compact, so is Conv A (see the proof of Proposition H.1). By the maximum principle (the minimum of a linear function on Conv A is attained at an extreme point, and the extreme points of Conv $A$ are contained in $A )$ $\begin{array} { r } { \operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { m i n } _ { a \in A } \langle \theta , a \rangle = \operatorname* { m a x } _ { \theta \in \Theta } } \end{array}$ <sub>Θ</sub> mi $\Omega _ { a \in \operatorname { C o n v } A } \left. \theta , a \right.$ . Since Θ and Conv A are both compact convex sets and $( \theta , a ) \mapsto \langle \theta , a \rangle$ is bilinear, the minimax theorem gives $\begin{array} { r } { \operatorname* { n a x } _ { \theta \in \Theta } \operatorname* { m i n } _ { a \in \operatorname { C o n v } \cal A } \langle \theta , a \rangle = \operatorname* { m i n } _ { a \in \operatorname { C o n v } \cal A } \operatorname* { m a x } _ { \theta \in \Theta } \langle \theta , a \rangle } \end{array}$ □

Using the set $Z ^ { \ast }$ of Equation (G.1),

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { i n f } _ { z \in Z ^ { * } } \langle \theta , z \rangle = \operatorname* { m a x } _ { \theta \in \Theta } \operatorname* { m i n } _ { z \in \overline { { Z ^ { * } } } } \langle \theta , z \rangle = \operatorname* { m i n } _ { z \in \mathrm { C o n v } } \operatorname* { m a x } _ { \overline { { Z ^ { * } } } } \langle \theta , z \rangle\tag{H.1}
$$

holds. The first inequality is due to the diferences of $Z ^ { \ast }$ being taken over all of $X ( s )$ rather than over $Y ( s )$ (Equation $\left( \mathrm { G . 1 } \right) )$ : since $Y ( s ) \subseteq X ( s )$ , for each θ the quantity ${ \operatorname* { i n f } } _ { z \in Z ^ { * } } \left. \theta , z \right.$ is at most the quantity inside the max of Equation (6.1) (this direction sufices for the argument yielding a lower bound). For the first equality we used that $\overline { { Z ^ { * } } }$ is compact, since $Z ^ { * }$ is bounded (every $z \in Z ^ { * }$ satisfies $\| z \| \leq L _ { \mathrm { s u b } } < \infty )$ and that the infimum of the continuous function $\langle \theta , z \rangle$ on $Z ^ { \ast }$ coincides with its minimum on ${ \overline { { Z ^ { * } } } } ;$ ; for the second equality we used Proposition H.2 (with $A = { \overline { { Z ^ { * } } } } )$ .

Proof of Proposition G.7. Take an arbitrary $\theta \in \Theta _ { T } ( \theta ^ { * } )$ . For every $s \in \mathcal { S }$ and $y \in X ( s ) \setminus \{ x ^ { * } ( \theta ^ { * } , s ) \}$ , apply Proposition G.6 with $X = X ( s ) , \theta = \theta ^ { * }$ and $x ^ { 1 } = y$ Since $x ^ { * } ( \theta ^ { * } , s )$ is the unique maximizer of $\theta ^ { * }$ on $X ( s )$ by Assumption $3 . 1 ( 3 )$ , we have $x ^ { * } = x ^ { * } ( \theta ^ { * } , s )$ , and there exist $g ^ { 1 } , \ldots , g ^ { r } \in \mathcal { T } \left( r \geq 1 \right)$ with

$$
x ^ { * } ( \theta ^ { * } , s ) - y = \sum _ { i = 1 } ^ { r } g ^ { i } , \qquad \langle \theta ^ { * } , g ^ { i } \rangle > 0 \quad ( i = 1 , \ldots , r )
$$

(where $r \geq 1$ because $x ^ { * } ( \theta ^ { * } , s ) \neq y )$ . Since each $g ^ { i }$ satisfies $\langle \theta , g ^ { i } \rangle > 0$

$$
\langle \theta , x ^ { * } ( \theta ^ { * } , s ) - y \rangle = \sum _ { i = 1 } ^ { r } \langle \theta , g ^ { i } \rangle \geq \operatorname* { m i n } _ { g \in { \mathcal { T } } , \langle \theta ^ { * } , g \rangle > 0 } \langle \theta , g \rangle .
$$

Taking the infimum over y and s, inf mi $\begin{array} { r } { { 1 } _ { \boldsymbol { y } } \langle \boldsymbol { \theta } , \boldsymbol { x } ^ { * } ( \boldsymbol { \theta } ^ { * } , \boldsymbol { s } ) - \boldsymbol { y } \rangle \ge \operatorname* { m i n } _ { \boldsymbol { g } \in \mathcal { T } , \langle \boldsymbol { \theta } ^ { * } , \boldsymbol { g } \rangle > 0 } \langle \boldsymbol { \theta } , \boldsymbol { g } \rangle } \end{array}$ Since $Y ( s ) \subseteq X ( s )$ by Assumption $3 . 1 ( 2 )$ , the left-hand side is at most the quantity inside the max of Equation (6.1), and hence at most $\gamma _ { \mathrm { s u b } }$ since $\theta \in \Theta _ { T } ( \theta ^ { * } ) \subset \Theta$ Taking the supremum over $\theta \in \Theta _ { T } ( \theta ^ { * } )$ on the right-hand side gives Equation (G.4).

## Appendix I. The separating hyperplane theorem and a lemma on the norm of the normal vector

Definition I.1. For a vector $\textbf { b } \in \mathbb { R } ^ { d }$ with $\mathbf b \neq \mathbf 0$ and a scalar $c \in \mathbb { R }$ , we call $H = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \left. \mathbf { b } , \mathbf { x } \right. \leq c \right\}$ a closed halfspace. In this case we call $\partial H = \{ \mathbf { x } \in$ $\mathbb { R } ^ { d } \mid \langle \bar { \mathbf { b } } , \mathbf { x } \rangle = c \dot  \}$ the boundary hyperplane of H.

Proposition I.2. In the Euclidean space $\mathbb { R } ^ { d }$ , suppose we are given a d-dimensional bounded convex polytope Q and a point $p \in \mathbb { R } ^ { d } \backslash Q$ not belonging to $Q .$ Then there exist a closed halfspace $H = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \left. \mathbf { b } , \mathbf { x } \right. \leq c \right\}$ and afinely independent vertices $y ^ { 1 } , \ldots , y ^ { d }$ of Q such that the following hold:

(1) $Q \subset H ;$

(2) $p \notin H$ (that is, $\langle \mathbf { b } , p \rangle > c ) ;$

(3) $\partial H = \operatorname { a f f } ( y ^ { 1 } , \ldots , y ^ { d } ) .$

Proof. Since Q is a d-dimensional bounded convex polytope, it has an irredundant facet representation $\begin{array} { r } { Q = \bigcap _ { i = 1 } ^ { | \mathcal { F } ( Q ) | } \{ \mathbf { x } \in \mathbb { R } ^ { d } \mid \langle \mathbf { n } _ { i } , \mathbf { x } \rangle \leq c _ { i } \} } \end{array}$ , where $\mathcal { F } ( Q )$ is the set of facets of $Q { \mathrm { . } }$ and each hyperplane $\{ \mathbf { x } \ | \ \langle \mathbf { n } _ { i } , \mathbf { x } \rangle = c _ { i } \}$ determines a facet $G _ { i } : = Q \cap \{ \mathbf { x } \mid \langle \mathbf { n } _ { i } , \mathbf { x } \rangle = c _ { i } \}$ of Q (cf. Schrijver, 1986). From $p \notin Q$ there is some $i _ { 0 }$ with $\langle { \bf n } _ { i _ { 0 } } , p \rangle > c _ { i _ { 0 } }$ . Since the facet $G _ { i _ { 0 } }$ is a face of dimension $d - 1$ , it has d afinely independent vertices $y ^ { 1 } , \ldots , y ^ { d } \in G _ { i _ { 0 } }$ , and these are vertices of Q (being vertices of a face of $Q )$ . Furthermore a $\operatorname { f } ( y ^ { 1 } , \dots , y ^ { d } ) = \operatorname { a f f } G _ { i _ { 0 } } = \{ \mathbf { x } \mid \langle \mathbf { n } _ { i _ { 0 } } , \mathbf { x } \rangle = c _ { i _ { 0 } } \}$ $\mathrm { S e t t i n g ~ } \mathbf { b } : = { \mathbf { n } } _ { i _ { 0 } } , \ c : = c _ { i _ { 0 } }$ and $H : = \{ \mathbf { x } \mid \langle \mathbf { b } , \mathbf { x } \rangle \leq c \}$ , we have $Q \subset H , \langle \mathbf { b } , p \rangle > c$ and $\partial H = \operatorname { a f f } ( y ^ { 1 } , \dots , y ^ { d } )$ □

Proposition I.3. Let $\xi > 0$ and set $\begin{array} { r } { F _ { \xi } ( z ) : = \sum _ { i = 1 } ^ { d } ( 1 - z _ { i } ) ^ { \xi } \mathrm { ~ f o r ~ } z \in \Delta ^ { d - 1 } } \end{array}$ . Then:

(1) if $\xi \ge 1$ then $F _ { \xi }$ is convex on $\Delta ^ { d - 1 }$ and its maximum is attained at a vertex; in particular $F _ { \xi } ( z ) \leq d - 1 ;$

(2) if $0 < \xi < 1$ then $F _ { \xi }$ is strictly concave on $\Delta ^ { d - 1 }$ and its maximum is attained at the barycenter $z = ( 1 / d , \dots , 1 / d )$ ; in particular $F _ { \xi } ( z ) \leq d ( 1 - 1 / d ) ^ { \xi }$

Proof. The second derivative of each term $\begin{array} { r } { g ( t ) = ( 1 - t ) ^ { \xi } } \end{array}$ is $g ^ { \prime \prime } ( t ) = \xi ( \xi - 1 ) ( 1 - t ) ^ { \xi - 2 }$ For $t \in [ 0 , 1 )$ : when $\xi \ge 1 , g ^ { \prime \prime } \ge 0$ so g is convex. Hence $F _ { \xi }$ is convex on $\Delta ^ { d - 1 }$ and its maximum is attained at a vertex $\mathbf { e } _ { i }$ . From $F _ { \xi } ( { \bf e } _ { i } ) = ( 1 - \bar { 1 } ) ^ { \xi } + ( d - 1 ) ( 1 - 0 ) ^ { \xi } = d - 1$ the claim follows. When $0 < \xi < 1 , g ^ { \prime \prime } < \bar { 0 }$ so g is strictly concave. Hence $F _ { \xi }$ is strictly concave on $\Delta ^ { d - 1 }$ , and by Jensen’s inequality the maximizer is the barycenter $z = ( 1 / d , \dots , 1 / d )$ , with $F _ { \xi } ( 1 / d , \dots , 1 / d ) = d ( 1 - 1 / d ) ^ { \xi }$ □

## Proposition I.4.

$$
\sum _ { i = 1 } ^ { d } \left( \sum _ { j \neq i } { \cal M } _ { j } ^ { 2 } \right) ^ { d - 1 } \leq ( d - 1 ) \left\| { \cal M } \right\| _ { 2 } ^ { 2 ( d - 1 ) } .
$$

Proof. The case $m = 0$ is trivial, so we may assume m $\neq 0$

$$
\sum _ { i = 1 } ^ { d } \left( \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { d - 1 } = \sum _ { i = 1 } ^ { d } \left( \| M \| _ { 2 } ^ { 2 } - M _ { i } ^ { 2 } \right) ^ { d - 1 } = \| M \| _ { 2 } ^ { 2 ( d - 1 ) } \sum _ { i = 1 } ^ { d } \left( 1 - { \frac { M _ { i } ^ { 2 } } { \| M \| _ { 2 } ^ { 2 } } } \right) ^ { d - 1 } .
$$

Setting $z _ { i } : = M _ { i } ^ { 2 } / \| M \| _ { 2 } ^ { 2 }$ we have $z \in \Delta ^ { d - 1 }$ . Applying Proposition I.3 with $\xi =$ $d - 1 \geq 1$ gives the claim. □

## Appendix J. Proofs of Theorem G.1 and Proposition G.2

Proof of Theorem $G . 1 .$ Step 1 (reduction to a distance problem). By Assumption 6.3 we have $X ( s ) \subset \mathbb { Z } ^ { d }$ and $x ^ { \ast } ( \theta ^ { \ast } , s ) \in \mathbb { Z } ^ { d }$ for each s, so $Z ( s ) \subset \mathbb { Z } ^ { d }$ and hence

$Z ^ { * } \subset \mathbb { Z } ^ { d }$ . Furthermore, for every $z = x ^ { * } ( \theta ^ { * } , s ) - x \in Z ( s )$ , since ${ } x ^ { * } ( \theta ^ { * } , s ) , x \in X ( s )$ each component satisfies

$$
| z _ { i } | \leq \operatorname* { m a x } _ { x \in X ( s ) } x _ { i } - \operatorname* { m i n } _ { x \in X ( s ) } x _ { i } \leq M _ { i } .
$$

Hence $\begin{array} { r } { Z ^ { * } \subset \Lambda : = \prod _ { i = 1 } ^ { d } \{ - M _ { i } , - M _ { i } + 1 , \dotsc , M _ { i } \} } \end{array}$ , so $Z ^ { * }$ is a finite set. Therefore $\overline { { Z ^ { * } } } = Z ^ { * }$ , and Conv ${ \overline { { Z ^ { * } } } } =$ Conv $Z ^ { \ast }$ is a bounded closed convex polytope. From Equation (H.1) and the fact that max $\| \theta \| { \leq } 1  \langle \theta , z \rangle = \| z \|$ when Θ is the unit ball,

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { m i n } _ { z \in \mathrm { C o n v } Z ^ { * } } \| z \| .\tag{J.1}
$$

Step $\textbf { 2 } ( 0 \not \in \mathrm { C o n v } Z ^ { * } )$ . By Assumption $3 . 1 ( 3 )$ , for every $s \in \mathcal { S }$ and every $x \in X ( s ) \setminus \{ x ^ { * } ( \theta ^ { * } , s ) \}$ we have $\langle \theta ^ { * } , x ^ { * } ( \theta ^ { * } , s ) - x \rangle > 0$ , that is, $\langle \theta ^ { * } , z \rangle > 0$ for all $z \in Z ^ { * }$ . Since $Z ^ { \ast }$ is a finite set, $\delta : = \mathrm { m i n } _ { z \in Z ^ { * } } \langle \theta ^ { * } , z \rangle > 0 $ , and by the linearity of convex combinations, $\langle \theta ^ { * } , z \rangle \ge \delta > 0$ for every $z \in$ Conv $Z ^ { \ast }$ . In particular $0 \not \in$ Conv $Z ^ { \ast }$

Step 3 (a separating hyperplane through lattice points). Since Conv $Z ^ { \ast }$ is the convex hull of the finite set $Z ^ { \ast }$ , it is a convex polytope, and $0 \not \in$ Conv $Z ^ { * }$ . We apply Proposition I.2 with $Q = \operatorname { C o n v } Z ^ { * }$ and $p = 0$ (here we use the assumption dim Conv $Z ^ { * } = d$ of the theorem; for the low-dimensional case see Proposition G.2 and Remark G.3). This yields a closed halfspace $H ^ { - } = \{ x : a \cdot x \leq c \}$ and d afinely independent points $y ^ { 1 } , \ldots , y ^ { d } \in Z ^ { * }$ that are vertices of Conv $Z ^ { \ast }$ (hence elements of $Z ^ { * } \subset \Lambda$ , and therefore lattice points), such that (1) Conv $Z ^ { \ast } \subset H ^ { - } , ( 2 ) \ 0 \not \in H ^ { - }$ (that is, $0 > c )$ , and (3) $\partial H ^ { - } = \operatorname { a f f } ( y ^ { 1 } , \ldots , y ^ { d } )$

Let U be the (d − 1) × d integer matrix whose rows are the diference vectors $v _ { k } : = y ^ { k + 1 } - y ^ { 1 } \in \mathbb { Z } ^ { d } \ ( k = 1 , \ldots , d - 1 )$ , and let $U ^ { ( i ) }$ be the matrix obtained by deleting its i-th column; then the normal vector a can be constructed as $a _ { i } =$ $( - 1 ) ^ { i } \operatorname* { d e t } ( U ^ { ( i ) } ) \in \mathbb { Z } ~ ( i = 1 , \dots , d )$ . The sign of a is chosen so that Conv $Z ^ { * } \subset H ^ { - }$ that is, so that condition (1) holds. We have $c = a \cdot y ^ { 1 } \in \mathbb { Z }$ . From condition (2) and $- c \in \mathbb { Z }$

$$
- c \geq 1 .\tag{J.2}
$$

For every $z \in$ Conv $Z ^ { * } \subset H ^ { - }$ we have $a \cdot z \leq c < 0$ . By Cauchy–Schwarz,

$$
\left\| z \right\| \geq { \frac { | a \cdot z | } { \left\| a \right\| } } = { \frac { - a \cdot z } { \left\| a \right\| } } \geq { \frac { - c } { \left\| a \right\| } } \geq { \frac { 1 } { \left\| a \right\| } } .
$$

Taking the minimum over z,

$$
\operatorname* { m i n } _ { z \in \operatorname { C o n v } Z ^ { * } } \| z \| \geq { \frac { 1 } { \| a \| } } .\tag{J.3}
$$

Step 4 (estimate of the norm of the normal vector). Since $y ^ { 1 } , \ldots , y ^ { d } \in \Lambda$ each component of a diference vector satisfies $| ( v _ { k } ) _ { j } | \le 2 M _ { j }$ . By Hadamard’s inequality,

$$
| a _ { i } | = | \operatorname * { d e t } ( U ^ { ( i ) } ) | \leq \prod _ { k = 1 } ^ { d - 1 } \| v _ { k } ^ { ( i ) } \| \leq \prod _ { k = 1 } ^ { d - 1 } \sqrt { \sum _ { j \neq i } ( 2 M _ { j } ) ^ { 2 } } = 2 ^ { d - 1 } \left( \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { ( d - 1 ) / 2 } .
$$

Therefore $\begin{array} { r } { \| a \| ^ { 2 } = \sum _ { i = 1 } ^ { d } a _ { i } ^ { 2 } \le 4 ^ { d - 1 } \sum _ { i = 1 } ^ { d } ( \sum _ { j \neq i } M _ { j } ^ { 2 } ) ^ { d - 1 } } \end{array}$ , and Proposition I.4 gives

$$
\| a \| \leq 2 ^ { d - 1 } { \sqrt { d - 1 } } \| M \| _ { 2 } ^ { d - 1 } .\tag{J.4}
$$

Combining Equations (J.1), (J.3) and (J.4) gives the claim.

Proof of Proposition G.2. Steps 1–2 of the proof of Theorem G.1 use no assumption on the dimension, so they hold as they are, and

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { m i n } _ { z \in \mathrm { C o n v } } z ^ { * } \left\| z \right\| \geq \mathrm { d i s t } ( 0 , \mathrm { a f f } Z ^ { * } ) , \qquad Z ^ { * } \subset \Lambda \subset \mathbb { Z } ^ { d } .
$$

Take afinely independent $y ^ { 1 } , \ldots , y ^ { k + 1 } \in Z ^ { * }$ (spanning af $Z ^ { * } )$ and set $v _ { i } : = y ^ { i + 1 } -$ $y ^ { 1 } \in \mathbb { Z } ^ { d } ( i = 1 , \ldots , k )$ . Since $y ^ { i } \in \Lambda$ , each component satisfies $| v _ { i , j } | \leq 2 M _ { j }$ and hence $\| v _ { i } \| \leq 2 \| M \| _ { 2 }$ . Since af $Z ^ { * } = y ^ { 1 } + \operatorname { s p a n } ( v _ { 1 } , \dots , v _ { k } )$ , the condition $0 ~ \notin$ af $Z ^ { \ast }$ is equivalent to $y ^ { 1 } \notin \operatorname { s p a n } ( v _ { 1 } , . . . , v _ { k } )$ , and then $( v _ { 1 } , \ldots , v _ { k } , y ^ { 1 } )$ is linearly independent.

The distance from the point 0 to the afine subspace $y ^ { 1 } + \operatorname { s p a n } ( v _ { 1 } , \ldots , v _ { k } )$ can be expressed, using Gram determinants (written $G )$ , as

$$
\operatorname { d i s t } ( 0 , \operatorname { a f f } Z ^ { * } ) ^ { 2 } = { \frac { \operatorname * { d e t } G ( v _ { 1 } , \ldots , v _ { k } , y ^ { 1 } ) } { \operatorname * { d e t } G ( v _ { 1 } , \ldots , v _ { k } ) } }
$$

(decomposing orthogonally as $- y ^ { 1 } = w _ { U } + w _ { \perp }$ with $w _ { U } \in U : = \operatorname { s p a n } ( v _ { 1 } , \dots , v _ { k } )$ and $w _ { \bot } \bot U$ , the multilinearity of Gram determinants and elementary column operations give det $G ( v , - y ^ { 1 } ) = \operatorname* { d e t } G ( v ) \cdot \| w _ { \bot } \| ^ { 2 }$ , together with det $G ( v , - y ^ { 1 } ) = \operatorname* { d e t } G ( v , y ^ { 1 } ) )$ The numerator is the Gram determinant of linearly independent integer vectors, hence a positive integer, in particular $\geq 1$ . The denominator is bounded, by Hadamard’s inequality for Gram determinants, as det $\begin{array} { r } { G ( v _ { 1 } , \dots , v _ { k } ) \leq \prod _ { i = 1 } ^ { k } \| v _ { i } \| ^ { 2 } \leq } \end{array}$ $( 2 \| M \| _ { 2 } ) ^ { 2 k }$ . Altogether we obtain dist(0, af $Z ^ { * } ) \geq ( 2 \Vert M \Vert _ { 2 } ) ^ { - k }$ □

## Appendix K. Proof of Theorem G.4

Proof of Theorem $G . 4 .$ Step 1 (reduction to the lattice structure). As in Step 1 of the proof of Theorem G.1, the set $Z ^ { * } \subset \Lambda : = \prod _ { i = 1 } ^ { d } \{ - M _ { i } , . . . , M _ { i } \} \subset \mathbb { Z } ^ { d }$ is finite and Conv $Z ^ { \ast }$ is a bounded closed convex polytope. When $\Theta = \Delta ^ { d - 1 }$ we have $\begin{array} { r } { \operatorname* { m a x } _ { \theta \in \Theta } \langle \theta , z \rangle = \operatorname* { m a x } _ { i } z _ { i } , } \end{array}$ so Equation (H.1) gives

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { m i n } _ { z \in \mathrm { C o n v } \ : Z ^ { * } } \operatorname* { m a x } _ { i = 1 , \ldots , d } z _ { i } .\tag{K.1}
$$

Step 2 (Conv $Z ^ { * } \cap \mathbb { R } _ { < 0 } ^ { d } = \emptyset )$ . By Assumption 3.1(3) we have $\langle \theta ^ { * } , z \rangle > 0$ for every $z \in Z ^ { * }$ . Since $\theta ^ { * } \in \Delta ^ { d - 1 }$ has nonnegative components summing to 1, the inequality $\textstyle \sum _ { i } \theta _ { i } ^ { * } z _ { i } > 0$ implies that at least one $z _ { i } > 0$ , and in particular max ${ \mathrm { ; ~ } } z _ { i } > 0$ By the linearity of convex combinations, $\langle \theta ^ { * } , z \rangle > 0$ for every $z \in$ Conv $Z ^ { * }$ , and hence max $z _ { i } > 0$ . Rewritten in the language of sets,

$$
\begin{array} { r } { \mathrm { C o n v } Z ^ { * } \cap \mathbb { R } _ { \leq 0 } ^ { d } = \varnothing . } \end{array}\tag{K.2}
$$

Below we assume $Z ^ { * } \neq \varnothing \ ( \mathrm { i f } \ Z ^ { * } = \varnothing$ then the min in Equation (K.1) is $+ \infty$ as an infimum over the empty set and the claim is trivial). In this case some $z \in Z ^ { * }$ satisfies max $_ i z _ { i } > 0$ , and since z is an integer vector, $z _ { i } \geq 1$ for some $i ,$ hence $M _ { i } \geq 1$ and in particular $\| M \| _ { 2 } \geq 1$

Step 3 (the polyhedron and the nonnegativity of its facet normals). Consider the Minkowski sum $\widetilde { P } : = \mathrm { C o n v } Z ^ { * } + \mathbb { R } _ { > 0 } ^ { d }$ . The set $\widetilde { P }$ is a polyhedron (the Minkowski sum of a bounded polytope and a polyhedral cone; the Minkowski–Weyl decomposition, cf. Schrijver, 1986), and the following hold.

(1) (Full-dimensionality, pointedness, and integrality of the vertices) Since $\widetilde { P }$ contains $z + \mathbb { R } _ { > 0 } ^ { d }$ for $z \in$ Conv $Z ^ { * } ,$ , it is d-dimensional. Its characteristic cone is $\mathbb { R } _ { \geq 0 } ^ { d } .$ , which is pointed, so $\widetilde { P }$ has vertices. Furthermore the vertices of $\widetilde { P }$ are elements of $Z ^ { * }$ : a point $x = z + w \ ( { \mathrm { w i t h ~ } } z \in { \mathrm { C o n v } } Z ^ { * } , w \geq 0$ $w \neq 0 )$ can be written as $\begin{array} { r } { x = \frac { 1 } { 2 } z + \frac { 1 } { 2 } ( z + 2 w ) } \end{array}$ , the midpoint of two distinct points of ${ \widetilde { P } } ,$ , so it is not a vertex, and hence the vertices belong to Conv $Z ^ { * } ;$ since Conv $Z ^ { * } \subseteq { \tilde { P } }$ , the vertices of $\widetilde { P }$ are extreme points of Conv $Z ^ { \ast }$ , and as extreme points of the convex hull of the finite set $Z ^ { \ast }$ they belong to $Z ^ { * } \subset \Lambda$

(2) (Nonnegativity of the facet normals) Writing a valid inequality defining an arbitrary facet $F$ of $\widetilde { P }$ as $a _ { F } \cdot x \geq c _ { F }$ (with $\widetilde { P } \subseteq \{ x : a _ { F } \cdot x \geq c _ { F } \}$ and ${ \cal F } = \widetilde { P } \cap \{ x : a _ { \cal F } \cdot x = c _ { \cal F } \} )$ , for every $x \in \widetilde { P }$ and $t \geq 0$ we have $x + t e _ { i } \in \widetilde { P }$ so $a _ { F } \cdot ( x + t e _ { i } ) \geq c _ { F }$ holds for all $t \geq 0$ , which $\mathrm { g i v e s } ~ ( a _ { F } ) _ { i } \geq 0$ , that is, $a _ { F } \geq 0 .$

(3) (Separation of the origin) By Equation (K.2), for every $z \in$ Conv $Z ^ { * }$ and $w \geq 0$ we have max<sub>i</sub> $( z + w ) _ { i } \geq \operatorname* { m a x } _ { i } z _ { i } > 0$ , so $\widetilde { P } \cap \mathbb { R } _ { \leq 0 } ^ { d } \ : = \ : \varnothing$ and in particular $0 \not \in \tilde { P }$ . Since $\widetilde { P }$ is a d-dimensional polyhedron, it coincides with the intersection of the valid inequalities defined by its facets (a standard fact of polyhedral theory, cf. Schrijver, 1986). Therefore there exists a facet $F$ with $a _ { F } \cdot 0 = 0 < c _ { F }$ , that is, whose valid inequality separates the origin.

Step 4 (construction of an integral nonnegative normal vector from lattice points and directions in $\mathbb { R } _ { \geq 0 } ^ { d } )$ . Take the facet $F$ of Step 3. Since $F$ is a face of the pointed polyhedron ${ \widetilde { P } } ,$ , it is a pointed polyhedron, and its vertices are vertices of ${ \widetilde { P } } .$ , hence elements of $Z ^ { * } \subset \Lambda$ . Moreover, by $a _ { F } \geq 0$ , the characteristic cone of F is

$$
\begin{array} { r } { \operatorname { r e c } ( F ) = \mathbb { R } _ { \geq 0 } ^ { d } \cap \{ w : a _ { F } \cdot w = 0 \} = \operatorname { c o n e } \{ e _ { i } : i \in I _ { 0 } \} , \qquad I _ { 0 } : = \{ i : ( a _ { F } ) _ { i } = 0 \} } \end{array}
$$

(since $w \geq 0$ together with $\boldsymbol { a } _ { F } \cdot \boldsymbol { w } = 0$ forces $w _ { i } = 0$ for every i with $( a _ { F } ) _ { i } > 0 )$ Taking, among the vertices of $F ,$ afinely independent points $y ^ { 1 } , \ldots , y ^ { p } \in Z ^ { * } \subset \Lambda$ $( 1 \leq p \leq d )$ spanning the afine hull of the vertex set, we have

$$
{ \mathrm { a f f } } \ F = { \mathrm { a f f } } { \bigl ( } y ^ { 1 } , \ldots , y ^ { p } { \bigr ) } + { \mathrm { s p a n } } \{ e _ { i } : i \in I _ { 0 } \} , \qquad { \mathrm { d i m ~ a f f } } \ F = d - 1 ,
$$

so we can choose $I ^ { \prime } \subseteq I _ { 0 }$ with $| I ^ { \prime } | = d - p$ such that $y ^ { 1 } , \ldots , y ^ { p }$ and $y ^ { 1 } + e _ { i } \ ( i \in I ^ { \prime } )$ are d afinely independent points spanning af $F$

Define the $( d - 1 ) \times d$ integer matrix $U$ whose rows are the diference vectors, namely the rows $v _ { k } : = y ^ { k + 1 } - y ^ { 1 } ( k = 1 , \ldots , p - 1$ ; each component satisfying $| v _ { k , j } | \le 2 M _ { j }$ since $y ^ { k } \in \Lambda )$ and the rows $e _ { i } \ ( i \in I ^ { \prime } )$ , and construct the integer vector a from the cofactors $a _ { i } : = ( - 1 ) ^ { i } \operatorname* { d e t } ( U ^ { ( i ) } ) \in \mathbb { Z } ~ ( i = 1 , \dots , d )$ , where $U ^ { ( i ) }$ is the matrix with the i-th column deleted. The rows of $U$ are linearly independent (being diference vectors of $d$ afinely independent points), so $a \neq 0$ , and since a is orthogonal to all the rows of $U _ { : }$ it is a normal direction of af $F ,$ that is, parallel to $a _ { F }$ . Choosing the sign in the same direction as $a _ { F }$ , we have $a = \lambda a _ { F }$ for some $\lambda > 0$ , so

$$
a \in \mathbb { Z } _ { \geq 0 } ^ { d } \setminus \{ 0 \} , \qquad \mathrm { C o n v } Z ^ { * } \subseteq \widetilde { P } \subseteq \{ x : a \cdot x \geq c \} , \qquad c : = a \cdot y ^ { 1 } = \lambda c _ { F } > 0
$$

holds, and $c \in { \mathbb { Z } } _ { > 0 } { \mathrm { ~ g i v e s ~ } } c \geq 1$

Step 5 (the estimate). First the case $p = 1$ : the rows of $U$ are the $d - 1$ unit vectors $\{ e _ { i } \} _ { i \in I ^ { \prime } } , \mathrm { s o } a = \pm e _ { i _ { 0 } }$ (with $i _ { 0 } \notin I ^ { \prime } )$ , and nonnegativity gives $a = e _ { i _ { 0 } }$ and $c = y _ { i _ { 0 } } ^ { 1 } \geq 1$ . Hence for every $z \in$ Conv $Z ^ { \ast }$ we have max $z _ { i } \ge z _ { i _ { 0 } } = a \cdot z \ge c \ge 1$ and since $\| M \| _ { 2 } \geq 1$ the right-hand side of the claim is at most 1, so the claim follows. Below we assume $p \geq 2$

For every z ∈ Conv $Z ^ { * }$ , setting $\begin{array} { r } { \lambda _ { i } : = a _ { i } / \sum _ { j = 1 } ^ { d } a _ { j } } \end{array}$ under $\textstyle \sum _ { j } a _ { j } > 0$ (which holds since $a \neq 0$ and $a _ { j } \geq 0 )$ , we have $\lambda \in \Delta ^ { d - 1 }$ , so

$$
\operatorname* { m a x } _ { i = 1 , \ldots , d } z _ { i } \geq \sum _ { i = 1 } ^ { d } \lambda _ { i } z _ { i } = { \frac { a \cdot z } { \sum _ { j } a _ { j } } } \geq { \frac { c } { \sum _ { j } a _ { j } } } \geq { \frac { 1 } { \sum _ { j = 1 } ^ { d } a _ { j } } } .\tag{K.3}
$$

We now find an upper bound on $\textstyle \sum _ { j } a _ { j }$ . By Hadamard’s inequality, the contribution of the diference-vector rows of $U ^ { ( i ) }$ is $\begin{array} { r } { \prod _ { k = 1 } ^ { p - 1 } \| v _ { k } ^ { ( i ) } \| \le \prod _ { k = 1 } ^ { p - 1 } 2 ( \sum _ { j \neq i } M _ { j } ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ and the contribution of the unit-vector rows is at most 1, so

$$
\begin{array} { c } { { a _ { i } \leq \left| a _ { i } \right| \leq \left( 2 \left( \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { 1 / 2 } \right) ^ { p - 1 } \leq \left( 2 \left( \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { 1 / 2 } \right) ^ { d - 1 } } } \\ { { = 2 ^ { d - 1 } \left( \displaystyle \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { ( d - 1 ) / 2 } } } \end{array}
$$

holds (for the second inequality: if $\begin{array} { r } { \sum _ { j \neq i } M _ { j } ^ { 2 } \geq 1 } \end{array}$ it follows since the base is at least 2 and $p - 1 \leq d - 1$ , whereas $\begin{array} { r } { \mathrm { i f } \sum _ { j \neq i } M _ { j } ^ { 2 } = 0 } \end{array}$ then all the diference-vector rows $v _ { k } ^ { ( i ) }$ are zero vectors, so $a _ { i } = 0$ since $p \geq 2 ;$ in either case it holds). Since $a _ { i } \geq 0$

$$
\sum _ { i = 1 } ^ { d } a _ { i } \leq 2 ^ { d - 1 } \sum _ { i = 1 } ^ { d } \left( \sum _ { j \neq i } M _ { j } ^ { 2 } \right) ^ { ( d - 1 ) / 2 } = 2 ^ { d - 1 } \left\| M \right\| _ { 2 } ^ { d - 1 } \sum _ { i = 1 } ^ { d } \left( 1 - { \frac { M _ { i } ^ { 2 } } { \| M \| _ { 2 } ^ { 2 } } } \right) ^ { ( d - 1 ) / 2 }
$$

(noting that $\| M \| _ { 2 } \ge 1 > 0 )$ . Applying Proposition I.3 with $\xi = ( d - 1 ) / 2 \colon$ for $d \geq 3$ we have $\xi \ge 1$ , so $\textstyle \sum _ { i } ( 1 - z _ { i } ) ^ { ( d - 1 ) / 2 } \leq d - 1 ;$ ; for $d = 2$ we have $\xi = 1 / 2 < 1$ , so $\begin{array} { r } { \sum _ { i } ( 1 - z _ { i } ) ^ { 1 / 2 } \leq d ( 1 - 1 / d ) ^ { 1 / 2 } = \sqrt { 2 } . } \end{array}$ Hence

$$
\sum _ { i = 1 } ^ { d } a _ { i } \leq 2 ^ { d - 1 } \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) \| M \| _ { 2 } ^ { d - 1 } .\tag{K.4}
$$

Combining Equations (K.1), (K.3) and (K.4) gives the claim.

## Appendix L. Proofs of the lower bounds for M-convex and M<sup>♮</sup>-convex structures

Proof of Theorem G.13. By Proposition G.12, the test set can be taken to be $\mathcal { T } = \{ e _ { i } - e _ { j } \ | \ i \neq j \}$ . Take a permutation σ so that $\theta _ { \sigma ( 1 ) } ^ { * } \geq \theta _ { \sigma ( 2 ) } ^ { * } \geq \cdot \cdot \cdot \geq \theta _ { \sigma ( d ) } ^ { * }$ (fixing an arbitrary order among components of equal value).

The case $\Theta = \{ \theta : \| \theta \| _ { 2 } \leq 1 \}$ : setting the arithmetic arrangement ${ \theta } _ { \sigma ( i ) } ^ { \dagger } = \qquad $ $\delta ( ( d + 1 ) / 2 - i ) ~ ( i = 1 , \dots , d )$ with $\delta = { 2 \sqrt { 3 } } / { \sqrt { d ( d ^ { 2 } - 1 ) } }$ , we have $\lVert \theta ^ { \dagger } \rVert _ { 2 } = 1$ and hence $\theta ^ { \dag } \in \Theta$ . If $\theta _ { i } ^ { * } > \theta _ { j } ^ { * }$ then i ranks above $j$ in the order of $\sigma ,$ , so $\theta _ { i } ^ { \dagger } > \theta _ { j } ^ { \dagger }$ that is, $\langle \theta ^ { \dagger } , e _ { i } - e _ { j } \rangle > 0 ;$ hence $\theta ^ { \dag } \in \Theta _ { T } ( \theta ^ { * } )$ . Furthermore, for every $( i , j )$ with $\langle \theta ^ { \ast } , e _ { i } - e _ { j } \rangle > 0$ , the diference $\boldsymbol { \theta } _ { i } ^ { \dagger } - \boldsymbol { \theta } _ { j } ^ { \dagger }$ is at least the diference $\delta$ between adjacent ranks of $\sigma ,$ so Proposition G.7 gives

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { m i n } _ { g \in \mathcal { T } , \langle \theta ^ { * } , g \rangle > 0 } \langle \theta ^ { \dagger } , g \rangle \geq \delta = \frac { 2 \sqrt { 3 } } { \sqrt { d ( d ^ { 2 } - 1 ) } } .
$$

The case $\Theta = \Delta ^ { d - 1 }$ : setting $\theta _ { \sigma ( i ) } ^ { \dagger } = ( d - i ) \cdot 2 / ( d ( d - 1 ) ) { \ ( i = 1 , \dots , d ) }$ , we have $\theta ^ { \dagger } \in \Delta ^ { d - 1 }$ , and as above $\theta ^ { \dag } \in \Theta _ { T } ( \theta ^ { * } )$ . The diference between adjacent ranks is $2 / ( d ( d - 1 ) )$ ), so Proposition G.7 gives $\gamma _ { \mathrm { s u b } } \geq 2 / ( d ( d - 1 ) )$ □

Proof of Theorem G.16. By Proposition G.15, the set $\mathcal { T } = \{ e _ { i } - e _ { j } , \pm e _ { i } | i \neq j \}$ is a test set. The elements $g \in { \mathcal { T } }$ with $\langle \theta ^ { * } , g \rangle > 0$ are of three kinds: $g = e _ { i } - e _ { j }$ (with $\theta _ { i } ^ { * } > \theta _ { i } ^ { * } ) , g = + e _ { i }$ (with $\theta _ { i } ^ { * } > 0 )$ , and $g = - e _ { i }$ (with $\theta _ { i } ^ { * } < 0 )$ . Below we construct, for a fixed $\theta ^ { * }$ , a sign-consistent $\theta ^ { \dag } \in \Theta _ { T } ( \theta ^ { * } )$ and apply Proposition G.7.

The case $\Theta = \{ \theta \in \mathbb { R } ^ { d } : \| \theta \| _ { 2 } \leq 1 \}$ : reorder the coordinates so that $\theta _ { 1 } ^ { * } \geq \cdots \geq \theta _ { d } ^ { * }$ and let $d _ { + } , d _ { 0 } , d _ { - } \ ( d _ { + } + d _ { 0 } + d _ { - } = d )$ be the numbers of positive, zero and negative components (if $\theta ^ { * } = 0$ then $S ^ { + } = \{ g \in \mathcal { T } \mid \langle \theta ^ { * } , g \rangle > 0 \} = \emptyset$ and the lower bound of Proposition G.7 holds trivially, so below we may assume $\theta ^ { * } \neq 0$ , that $\mathrm { i s } , s \neq 0 )$ Define the integer vector

$$
\boldsymbol { s } = ( d _ { + } , d _ { + } - 1 , \dots , 1 , \underbrace { 0 , \dots , 0 } _ { d _ { 0 } } , - 1 , \dots , - d _ { - } )
$$

and set $\theta ^ { \dagger } : = c s$ with $\begin{array} { r } { c : = ( \sum _ { i } s _ { i } ^ { 2 } ) ^ { - 1 / 2 } \ ( \operatorname { s o } \ \| \theta ^ { \dagger } \| _ { 2 } = 1 } \end{array}$ and hence $\theta ^ { \dag } \in \Theta )$ . Each of the three kinds of g above satisfies $\langle \theta ^ { \dagger } , g \rangle \geq c > 0 \colon$ for $\boldsymbol { g } = \boldsymbol { e } _ { i } - \boldsymbol { e } _ { j }$ we have $s _ { i } - s _ { j } \geq 1$ and hence $\langle \theta ^ { \dagger } , g \rangle = c ( s _ { i } - s _ { j } ) \geq c \mathrm { ; }$ for $g = + e _ { i }$ we have $s _ { i } \geq 1$ and hence $\langle \theta ^ { \dagger } , g \rangle = c s _ { i } \geq c ;$ for $g = - e _ { i }$ we have $- s _ { i } \geq 1$ and hence $\langle \theta ^ { \dagger } , g \rangle = c ( - s _ { i } ) \geq c .$ Hence $\theta ^ { \dag } \in \Theta _ { T } ( \theta ^ { * } )$ , and Proposition G.7 gives $\gamma _ { \mathrm { s u b } } \geq c$ . Finally, from $\textstyle \sum _ { i } s _ { i } ^ { 2 } =$ $\begin{array} { r } { \sum _ { k = 1 } ^ { d _ { + } } k ^ { 2 } + \sum _ { k = 1 } ^ { d _ { - } } k ^ { 2 } \le \sum _ { k = 1 } ^ { d } k ^ { 2 } = \frac { d ( d + 1 ) ( 2 d + 1 ) } { 6 } } \end{array}$ (since $d _ { + } + d _ { - } \leq d )$ , together with $d ( d + 1 ) ( 2 d + 1 ) \leq 6 d ^ { 3 }$ , we get $c \geq \sqrt { 6 / ( d ( d + 1 ) ( 2 d + 1 ) ) } \geq d ^ { - 3 / 2 }$

The case $\Theta = \Delta ^ { d - 1 }$ : since $\theta ^ { * } \geq 0 .$ , an element $g = - e _ { i }$ has $\langle { \theta } ^ { \ast } , g \rangle = - { \theta } _ { i } ^ { \ast } \leq 0$ and hence does not satisfy $\langle \theta ^ { * } , g \rangle > 0$ . Reorder the coordinates so that $\theta _ { 1 } ^ { * } \geq \dots \geq \theta _ { d } ^ { * }$ and set $\begin{array} { r } { \theta _ { i } ^ { \dagger } : = \frac { 2 ( d - i + 1 ) } { d ( d + 1 ) } ~ ( i = 1 , \dots , d ) } \end{array}$ ; then, being a decreasing sequence with all components positive and $\textstyle \sum _ { i } \theta _ { i } ^ { \dagger } = 1$ , we have $\theta ^ { \dagger } \in \Delta ^ { d - 1 }$ . The elements g with $\langle \theta ^ { * } , g \rangle > 0$ are limited to the two kinds $\boldsymbol { g } = \boldsymbol { e } _ { i } - \boldsymbol { e } _ { j }$ (with $\theta _ { i } ^ { * } > \theta _ { j } ^ { * }$ , hence $i < j )$ and $g = + e _ { i }$ (with $\theta _ { i } ^ { * } > 0 )$ , and both satisfy $\begin{array} { r } { \langle \theta ^ { \dagger } , g \rangle \geq \frac { 2 } { d ( d + 1 ) } > 0 } \end{array}$ (for the former $\begin{array} { r } { \langle \theta ^ { \dagger } , g \rangle = ( j - i ) \frac { 2 } { d ( d + 1 ) } } \end{array}$ , and for the latter $\langle \theta ^ { \dagger } , g \rangle = \theta _ { i } ^ { \dagger } \geq \theta _ { d } ^ { \dagger } = \textstyle { \frac { 2 } { d ( d + 1 ) } } )$ . Hence $\theta ^ { \dag } \in \Theta _ { T } ( \theta ^ { * } )$ , and Proposition G.7 gives $\gamma _ { \mathrm { s u b } } \geq \frac { 2 } { d ( d { + } 1 ) }$ □

## Appendix M. Proofs of the lower bounds via Graver bases

Proof of Proposition G.18. Take any $s \in \mathcal S , x ^ { 1 } \in X ( s )$ and $\theta \in \Theta$ , and suppose $\langle \theta , x ^ { 1 } \rangle < \operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x \rangle$ . Since $X ( s )$ is bounded by Assumption $3 . 1 ( 2 )$ and lies on the integer lattice, it is a finite set, so a maximizer $x ^ { 2 } \in \arg \operatorname* { m a x } _ { x \in X ( s ) } \langle \theta , x \rangle$ exists. Adding slacks and setting $\widetilde { x } ^ { j } : = ( x ^ { j } , b ( s ) - A x ^ { j } ) \in \mathbb { Z } ^ { d + N } \ ( j = 1 , 2 )$ , we have ${ \widetilde { A } } { \widetilde { x } } ^ { j } = b ( s )$ , and the last N components (the slack components) are nonnegative. The diference $\widetilde { z } : = \widetilde { x } ^ { 2 } - \widetilde { x } ^ { 1 } \in \ker _ { \mathbb { Z } } ( \widetilde { A } ) \setminus \{ 0 \}$ can be written, by the sign-consistent decomposition property of Graver bases (every $0 \neq z \in$ ker<sub>Z</sub>(B) decomposes into a sum of elements of $\mathcal { G } ( B )$ that are sign consistent with z; cf. Sturmfels, 1996; Onn, 2010), as $\begin{array} { r } { \widetilde { z } = \sum _ { k = 1 } ^ { r } \widetilde { g } ^ { k } } \end{array}$ with $\widetilde { g } ^ { k } \in \mathcal G ( \widetilde { A } )$ and $\widetilde { g } ^ { k } \subseteq \widetilde { z }$ . By sign consistency, for every $K \subseteq \{ 1 , \ldots , r \}$ each component of the partial sum $\begin{array} { r } { \widetilde { x } ^ { 1 } + \sum _ { k \in K } \widetilde { g } ^ { k } } \end{array}$ takes a value between the corresponding components of $\widetilde { x } ^ { 1 }$ and $\widetilde { x } ^ { 2 }$ . In particular the slack components stay nonnegative, and the equality with respect to Ae is preserved, so the first d components of the partial sum belong to $X ( s )$ . Setting $\widetilde { \theta } : = ( \theta , 0 ) \in \mathbb { R } ^ { d + p }$ , we have $\langle \widetilde { \theta } , \widetilde { z } \rangle = \langle \theta , x ^ { 2 } - x ^ { 1 } \rangle > 0 \mathrm { , }$ , so $\langle \widetilde { \theta } , \widetilde { g } ^ { k _ { 0 } } \rangle > 0$ for some $k _ { 0 }$ . Setting $g : = \pi _ { x } ( \widetilde { g } ^ { k _ { 0 } } ) \in \mathcal { T } _ { x }$ the partial sum with $K = \{ k _ { 0 } \}$ gives $x ^ { 1 } + g \in X ( s )$ and $\langle \theta , g \rangle > 0$ . Finally, since a projection does not increase the $\ell _ { \infty }$ norm, $\| \pi _ { x } ( \widetilde { g } ) \| _ { \infty } \le \| \widetilde { g } \| _ { \infty } \le g _ { \infty } ( \widetilde { A } )$ □

Proof of Theorem G.19. By Proposition G.18, the set $\mathcal { T } _ { x }$ is a test set with $\| g \| _ { \infty } \leq$ $C _ { g }$ for all $g \in \mathcal { T } _ { x }$ . Set $S ^ { + } : = \{ g \in \mathcal { T } _ { x } ~ | ~ \langle \theta ^ { * } , g \rangle > 0 \} \subset \mathbb { Z } ^ { d }$

Step 1 (reduction to a distance via minimax). Since $\langle \theta ^ { * } , g \rangle > 0$ for each $g \in S ^ { + }$ , we have $\langle \theta ^ { * } , q \rangle > 0$ for every $q \in \operatorname { C o n v } ( S ^ { + } )$ , and in particular $0 \not \in \mathrm { C o n v } ( S ^ { + } )$ $\mathrm { B y }$ Proposition H.2 $( B ^ { d } = \{ \| \theta \| _ { 2 } \leq 1 \}$ being bounded, closed and convex, and $S ^ { + }$ finite) together with ma $\operatorname { x } _ { \| \theta \| _ { 2 } } _ { \leq 1 } \langle \theta , q \rangle = \| q \| _ { 2 }$

$$
\operatorname* { m a x } _ { \theta \in B ^ { d } } \operatorname* { m i n } _ { g \in S ^ { + } } \langle \theta , g \rangle = \operatorname* { m i n } _ { \substack { q \in \mathrm { C o n v } ( S ^ { + } ) } } \| q \| _ { 2 } = \mathrm { d i s t } ( 0 , \mathrm { C o n v } ( S ^ { + } ) ) > 0 .
$$

On the other hand, Proposition G.7 gives $\begin{array} { r } { \gamma _ { \mathrm { s u b } } \geq \operatorname* { s u p } _ { \theta \in \Theta _ { T _ { x } } ( \theta ^ { * } ) } \operatorname* { m i n } _ { g \in S ^ { + } } \langle \theta , g \rangle } \end{array}$ . A $\theta \in$ $B ^ { d }$ attaining the maximum above satisfies mi ${ \mathrm { 1 } } _ { g \in S ^ { + } } \langle \theta , g \rangle = \operatorname { d i s t } ( 0 , \operatorname { C o n v } ( S ^ { + } ) ) > 0 .$ that is, $\langle \theta , g \rangle > 0$ for all $g \in S ^ { + }$ , so $\theta \in \Theta _ { T _ { x } } ( \bar { \theta } ^ { * } )$ (since $S ^ { + } = \{ g \mid \langle \theta ^ { * } , g \rangle > 0 \} )$ . Hence $\gamma _ { \mathrm { s u b } } \geq \mathrm { d i s t } ( 0 , \operatorname { C o n v } ( S ^ { + } ) )$

Step 2 (an integral separating hyperplane). The set $\operatorname { C o n v } ( S ^ { + } )$ is a lattice polytope whose vertices lie in $S ^ { + } \subset \{ g \in \mathbb { Z } ^ { d } : \| g \| _ { \infty } \leq C _ { g } \}$ , and it is full-dimensional with $0 \not \in \mathrm { C o n v } ( S ^ { + } )$ by assumption. As in Step 3 of the proof of Theorem G.1, applying Proposition I.2 with $A = \operatorname { C o n v } ( S ^ { + } )$ and $p = 0$ , we can take d afinely independent vertices $y ^ { 1 } , \ldots , y ^ { d } \in S ^ { + }$ (lattice points) and, from the diference vectors $v _ { k } : = y ^ { k + 1 } - y ^ { 1 }$ , the cofactors $a _ { i } = ( - 1 ) ^ { i } \operatorname* { d e t } ( U ^ { ( i ) } ) \in \mathbb { Z }$ and $c = a \cdot y ^ { 1 } \in \mathbb { Z } .$ so that Conv $( S ^ { + } ) \subseteq \{ x : a \cdot x \leq c \}$ and $- c \geq 1$ hold.

Step 3 (the estimate). For every $q ~ \in \mathrm { ~ C o n v } ( S ^ { + } )$ , Cauchy–Schwarz gives $\| q \| _ { 2 } \geq - a \cdot q / \| a \| \geq - c / \| a \| \geq 1 / \| a \|$ , so dist $( 0 , \operatorname { C o n v } ( S ^ { + } ) ) \ge 1 / \| a \|$ . Since $y ^ { k } \in S ^ { + }$ gives $\| y ^ { k } \| _ { \infty } \leq C _ { g } ,$ , we have $| ( v _ { k } ) _ { j } | \le 2 C _ { g }$ . Taking $M _ { j } = C _ { g }$ in the Hadamard estimate of Step 4 of the proof of Theorem G.1 (so that $| ( v _ { k } ) _ { j } | \leq 2 M _ { j } = 2 C _ { g }$ and $\| ( C _ { g } , \dots , C _ { g } ) \| _ { 2 } = C _ { g } \sqrt { d } )$ , Equation (J.4) gives

$$
\| a \| \leq 2 ^ { d - 1 } { \sqrt { d - 1 } } ( C _ { g } { \sqrt { d } } ) ^ { d - 1 } = { \sqrt { d - 1 } } ( 2 C _ { g } { \sqrt { d } } ) ^ { d - 1 } .
$$

Combining the above gives $\gamma _ { \mathrm { s u b } } \geq 1 / \| a \| \geq 1 / ( \sqrt { d - 1 } ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } )$

Proof of Theorem G.20. By Proposition G.18, the set $\mathcal { T } _ { x }$ is a test set with $\| g \| _ { \infty } \leq$ $C _ { g } . \mathrm { ~ S e t ~ } S ^ { + } : = \{ g \in \mathcal { T } _ { x } \mid \langle \theta ^ { * } , g \rangle > 0 \} \subset \mathbb { Z } ^ { d } .$

Step 1 (reduction via minimax). Proposition G.7 gives

$$
\gamma _ { \mathrm { s u b } } \geq \operatorname* { s u p } _ { \theta \in \Theta _ { T _ { x } } ( \theta ^ { * } ) } \operatorname* { m i n } _ { g \in S ^ { + } } \langle \theta , g \rangle .
$$

By Proposition H.2 $( \Delta ^ { d - 1 }$ being bounded, closed and convex, and $S ^ { + }$ finite) together with ma $\mathrm { x } _ { \theta \in \Delta ^ { d - 1 } } \langle \theta , q \rangle \ = \ \operatorname* { m a x } _ { i } q _ { i }$ , we have ma $\mathrm { X } _ { \theta \in \Delta ^ { d - 1 } }$ 1 min $ \quad \operatorname { i } _ { g \in S ^ { + } } \langle \theta , g \rangle \ =$ $\begin{array} { r } { \operatorname* { m i n } _ { q \in \mathrm { C o n v } ( S ^ { + } ) } \operatorname* { m a x } _ { i } q _ { i } } \end{array}$ From $\theta ^ { * } \geq 0$ and $\langle \theta ^ { * } , g \rangle \ > \ 0$ for each $g ~ \in ~ S ^ { + }$ we get $\operatorname* { m a x } _ { i } q _ { i } > 0$ for every $q ~ \in \mathrm { ~ C o n v } ( S ^ { + } )$ , that is, $\mathrm { C o n v } ( S ^ { + } ) \cap \mathbb { R } _ { < 0 } ^ { d } \ : = \ : \emptyset$ , so $\mathrm { m i n } _ { q \in \mathrm { C o n v } ( S ^ { + } ) }$ <sub>)</sub> max<sub>i</sub> $q _ { i } > 0 . \mathrm { ~ A ~ } \theta \in \Delta ^ { d - 1 }$ attaining this maximum satisfies $\langle \theta , g \rangle > 0$ for all $g \in S ^ { + }$ , so $\theta \in \Theta _ { T _ { x } } ( \theta ^ { * } )$ . Hence γ<sub>sub</sub> $\geq { \mathrm { m i n } } _ { q \in \mathrm { C o n v } ( S ^ { + } ) }$ max<sub>i</sub> $q _ { i } > 0 .$

Steps 2–4 (an integral nonnegative normal vector via the polyhedron). As in Steps 3–4 of the proof of Theorem G.4 (replacing $Z ^ { \ast }$ by $S ^ { + }$ and the lattice box $\Lambda$ by $\{ g \in \mathbb { Z } ^ { d } : \| g \| _ { \infty } \leq C _ { g } \} )$ , consider the polyhedron $\widetilde P : = \mathrm { C o n v } ( S ^ { + } ) + \mathbb R _ { \ge 0 } ^ { d } .$ By Step 1 we have Conv $( S ^ { + } ) \cap \mathbb { R } _ { < 0 } ^ { d } \ : = \ : \emptyset$ , so $0 \notin \widetilde P ;$ the facet normals of $\widetilde { P }$ can be taken nonnegative, and there is a facet separating the origin. From the vertices $y ^ { 1 } , \ldots , y ^ { p } \in S ^ { + } \ ( 1 \leq p \leq d _ { \cdot }$ , lattice points) and the points $y ^ { 1 } + e _ { i } \ ( i \in I ^ { \prime }$ $| I ^ { \prime } | = d \mathrm { - } p )$ in unit-vector directions spanning the afine hull of that facet, the cofactor construction yields an integral normal vector $a \in \mathbb { Z } _ { \geq 0 } ^ { d } \setminus \{ 0 \}$ and $c : = a \cdot y ^ { 1 } \in \mathbb { Z } _ { \geq 1 }$ with Conv $( S ^ { + } ) \subseteq \{ x : a \cdot x \geq c \}$ . In the case $p = 1$ the vector a is a unit vector $e _ { i _ { 0 } } .$ so max<sub>i</sub> $q _ { i } \ge q _ { i _ { 0 } } \ge c \ge 1$ for every $q \in \operatorname { C o n v } ( S ^ { + } )$ , whereas the right-hand side of the claim is at most 1, so the claim follows immediately. Below we assume $p \geq 2$

Step 5 (the estimate). For every $q \in \operatorname { C o n v } ( S ^ { + } )$ , setting $\lambda _ { i } : = a _ { i } / \sum _ { j } a _ { j } \in$ $\Delta ^ { d - 1 }$ gives max $\begin{array} { r } { q _ { i } \ge \sum _ { i } \lambda _ { i } q _ { i } = ( a \cdot q ) / \sum _ { j } a _ { j } \ge c / \sum _ { j } a _ { j } \ge 1 / \sum _ { i } a _ { j } } \end{array}$ . Since $y ^ { k } \in S ^ { + }$ gives $| ( v _ { k } ) _ { j } | \leq 2 C _ { g }$ , taking $M _ { j } = C _ { g }$ in the Hadamard estimate of Step 5 of the proof of Theorem G.4, Equation (K.4) gives

$$
\sum _ { j = 1 } ^ { d } a _ { j } \leq 2 ^ { d - 1 } \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) ( C _ { g } \sqrt { d } ) ^ { d - 1 } = \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } .
$$

Combining the above gives $\begin{array} { r } { \gamma _ { \mathrm { s u b } } \geq 1 / \sum _ { i } a _ { j } \geq 1 / ( \operatorname* { m a x } ( d - 1 , \sqrt { 2 } ) ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } ) } \end{array}$ . □

Appendix N. Derivation of the explicit upper bounds by problem class

In this appendix we derive each entry of Table 4. We first give the general form of the substitution.

Corollary N.1 (The three guarantees in terms of $\gamma )$ . Under Assumption 3.1, the following hold (in the order expressions we may assume $\gamma \leq L D _ { ; }$ , since $K = 0$ otherwise by Lemma C.1).

(i): SGS-OGD (Algorithm 1 with $\alpha = D / ( L { \sqrt { 2 } } ) )$ satisfies

$$
K \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma ^ { 2 } } , \qquad \widetilde { R } _ { T } \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma } .
$$

(ii): ONS (Algorithm 3) satisfies

$$
K = O \left( \frac { d L D } { \gamma } \log \operatorname* { m a x } \left( \frac { 2 L D } { \gamma } , 2 \right) \right) ,
$$

$$
\widetilde { R } _ { T } \leq L D \left( 1 + 2 d \log \left( 2 + \frac { 2 L D } { \gamma } \right) \right) = O \left( L D d \log \operatorname* { m a x } \left( \frac { L D } { \gamma } , 2 \right) \right) .
$$

(iii): Growing-grid SGS-MetaGrad (Algorithm 6) satisfies

$$
K = O \left( \frac { d L D } { \gamma } \log \operatorname* { m a x } \left( \frac { 2 L D } { \gamma } , 2 \right) \right) ,
$$

$$
\widetilde { R } _ { T } = O \left( L D d \log \operatorname* { m a x } \left( \frac { L D } { \gamma } , 2 \right) \right) .
$$

Proof. The bounds are those in items (i) and (iii) of Theorems C.2, D.6 and $\mathrm { F . 3 }$ The order estimates hold because, in both (ii) and (iii), the leading terms are $\begin{array} { r } { \frac { L D } { \gamma } \cdot d \log \operatorname* { m a x } ( \frac { L D } { \gamma } , 1 ) } \end{array}$ (for K) and LD d log max $\scriptstyle ( { \frac { L D } { \gamma } } , 2 )$ (for $\widetilde { R } _ { T } )$ . □

By Lemma 6.1, Assumption 3.1 holds with $\gamma = \gamma _ { \mathrm { s u b } }$ and $L = L _ { \mathrm { s u b } }$ under each structure. The bounds of Corollary N.1 are monotonically nonincreasing in $\gamma ,$ since the factors $1 / \gamma$ , log $\operatorname* { m a x } ( \cdot / \gamma , 1 )$ and $\log ( 2 + \cdot / \gamma )$ are nonincreasing in $\gamma ;$ hence substituting the lower bounds of Table 3 for γ gives upper bounds. Below, M is the vector in Equation (6.3), and in the ILP and linear-inequality entries we also use $L \leq \| M \| _ { 2 }$ (Equation (G.2)). Moreover, in reducing the logarithmic factors to the forms of the entries of Table 4 $( \log ( 2 \lVert M \rVert _ { 2 } )$ for $\mathrm { I L P s , \log ( 2 C _ { \it g } d L ) }$ for linear inequalities, and log(2dL) for the M-convex and $\mathrm { M } ^ { \natural . }$ -convex cases), we assume $L \geq 1$ and $C _ { g } \geq 1$ (whence $\| M \| _ { 2 } \geq 1$ as well, since $L \leq \| M \| _ { 2 } )$ . The diameter $D = \dim ( \Theta )$ is 2 for the unit ball and $\sqrt { 2 }$ for the probability simplex.

General ILPs (Theorems G.1 and G.4). For the unit ball,

$$
\frac { 1 } { \gamma } \leq 2 ^ { d - 1 } \sqrt { d - 1 } \| M \| _ { 2 } ^ { d - 1 } = O ( 2 ^ { d } \sqrt { d } \| M \| _ { 2 } ^ { d - 1 } ) .
$$

By Corollary N.1(i), SGS-OGD satisfies

$$
\begin{array} { l } { { \displaystyle K \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma ^ { 2 } } = \frac { 8 L ^ { 2 } } { \gamma ^ { 2 } } \leq 2 ( d - 1 ) 4 ^ { d } \| M \| _ { 2 } ^ { 2 d - 2 } L ^ { 2 } = O ( d 4 ^ { d } \| M \| _ { 2 } ^ { 2 d } ) , } } \\ { { \displaystyle \widetilde { R } _ { T } \leq \frac { 2 L ^ { 2 } D ^ { 2 } } { \gamma } = \frac { 8 L ^ { 2 } } { \gamma } = O ( \sqrt { d } 2 ^ { d } \| M \| _ { 2 } ^ { d + 1 } ) , } } \end{array}
$$

and by (ii) and (iii), ONS and growing-grid SGS-MetaGrad satisfy

$$
\begin{array} { l } { { \displaystyle K = { \cal O } \left( \frac { d L D } { \gamma } \log \operatorname* { m a x } \left( \frac { 2 L D } { \gamma } , 2 \right) \right) = { \cal O } ( d ^ { 5 / 2 } 2 ^ { d } \| M \| _ { 2 } ^ { d } \log ( 2 \| M \| _ { 2 } ) ) } , } \\ { { \displaystyle \widetilde { R } _ { T } = { \cal O } \left( L D d \log \operatorname* { m a x } \left( \frac { L D } { \gamma } , 2 \right) \right) } } \\ { { \displaystyle \quad \quad = { \cal O } ( \| M \| _ { 2 } d \cdot d \log ( 2 \| M \| _ { 2 } ) ) = { \cal O } ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) ) } } \end{array}
$$

(where we used $\mathrm { o g } ( 1 / \gamma ) = O ( d \log \| M \| _ { 2 } + d ) )$ . For the probability simplex we have $1 / \gamma = O ( 2 ^ { d } d \| M \| _ { 2 } ^ { d - 1 } )$ , so the same substitution gives, for SGS-OGD,

$$
\begin{array} { c } { { K = O ( d ^ { 2 } 4 ^ { d } \| M \| _ { 2 } ^ { 2 d } ) , } } \\ { { \widetilde { R } _ { T } = O ( d 2 ^ { d } \| M \| _ { 2 } ^ { d + 1 } ) , } } \end{array}
$$

and, for ONS and growing-grid SGS-MetaGrad,

$$
\begin{array} { c } { { K = O ( d ^ { 3 } 2 ^ { d } \| M \| _ { 2 } ^ { d } \log ( 2 \| M \| _ { 2 } ) ) , } } \\ { { \widetilde { R } _ { T } = O ( d ^ { 2 } \| M \| _ { 2 } \log ( 2 \| M \| _ { 2 } ) ) . } } \end{array}
$$

Linear inequalities (Theorems G.19 and G.20). For the unit ball,

$$
\frac { 1 } { \gamma } \leq \sqrt { d - 1 } ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } = O ( \sqrt { d } ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } ) .
$$

SGS-OGD satisfies

$$
\begin{array} { r } { K = O ( L ^ { 2 } / \gamma ^ { 2 } ) = O ( L ^ { 2 } d ( 2 C _ { g } \sqrt { d } ) ^ { 2 d } ) , } \\ { \widetilde { R } _ { T } = O ( L ^ { 2 } / \gamma ) = O ( L ^ { 2 } \sqrt { d } ( 2 C _ { g } \sqrt { d } ) ^ { d } ) , } \end{array}
$$

and ONS and growing-grid SGS-MetaGrad satisfy

$$
\begin{array} { l } { { \displaystyle K = O \left( \frac { d L } \gamma \log \operatorname* { m a x } \left( \frac { 2 L } \gamma , 2 \right) \right) = O ( L d ^ { 5 / 2 } ( 2 C _ { g } \sqrt { d } ) ^ { d } \log ( 2 C _ { g } d L ) ) , } } \\ { { \displaystyle \widetilde R _ { T } = O \left( L d \log \operatorname* { m a x } \left( \frac { 2 L } \gamma , 2 \right) \right) = O ( L d ^ { 2 } \log ( 2 C _ { g } d L ) ) } } \end{array}
$$

(where we used $\log ( 1 / \gamma ) = O ( d \log ( C _ { g } d ) ) )$ . For the probability simplex we have $1 / \gamma = O ( d ( 2 C _ { g } \sqrt { d } ) ^ { d - 1 } )$ , so the power of d goes up by one and we obtain the probability-simplex entries of Table 4.

M-convex and M<sup>♮</sup>-convex (Theorems G.13 and G.16). For the unit ball we have $1 / \gamma \le \sqrt { d ( d ^ { 2 } - 1 ) } / ( 2 \sqrt { 3 } ) = O ( d ^ { 3 / 2 } )$ , so SGS-OGD satisfies

$$
K \leq \frac { 8 L ^ { 2 } } { \gamma ^ { 2 } } \leq \frac { 2 } { 3 } L ^ { 2 } d ( d ^ { 2 } - 1 ) = O ( L ^ { 2 } d ^ { 3 } ) ,
$$

$$
\widetilde { R } _ { T } \le \frac { 8 L ^ { 2 } } { \gamma } \le \frac { 4 L ^ { 2 } \sqrt { d ( d ^ { 2 } - 1 ) } } { \sqrt { 3 } } = O ( L ^ { 2 } d ^ { 3 / 2 } ) ,
$$

and ONS and growing-grid SGS-MetaGrad satisfy

$$
K = O \left( \frac { d L } { \gamma } \log \operatorname* { m a x } \left( \frac { 2 L } { \gamma } , 2 \right) \right) = O ( L d ^ { 5 / 2 } \log ( 2 d L ) ) ,
$$

$$
\widetilde { R } _ { T } = O \left( L d \log \operatorname* { m a x } \left( \frac { 2 L } { \gamma } , 2 \right) \right) = O ( L d \log ( 2 d L ) ) .
$$

For the probability simplex we have $1 / \gamma \le d ( d - 1 ) / 2 = O ( d ^ { 2 } )$ and $D = { \sqrt { 2 } } .$ , so similarly SGS-OGD satisfies

$$
K \leq \frac { 4 L ^ { 2 } } { \gamma ^ { 2 } } \leq L ^ { 2 } d ^ { 2 } ( d - 1 ) ^ { 2 } = O ( L ^ { 2 } d ^ { 4 } ) ,
$$

$$
\widetilde { R } _ { T } \le \frac { 4 L ^ { 2 } } { \gamma } \le 2 L ^ { 2 } d ( d - 1 ) = O ( L ^ { 2 } d ^ { 2 } ) ,
$$

and ONS and growing-grid SGS-MetaGrad satisfy

$$
K = O ( L d ^ { 3 } \log ( 2 d L ) ) ,
$$

$$
\widetilde { R } _ { T } = O ( L d \log ( 2 d L ) ) .
$$

In the $\mathrm { M } ^ { \natural } .$ -convex case only the constants of the bound on $1 / \gamma$ change $( d ^ { 3 / 2 }$ for the unit ball and $d ( d + 1 ) / 2$ for the probability simplex), and the orders coincide. In particular, when $L = O ( { \sqrt { d } } )$ , the number of mistakes is $O ( d ^ { 4 } )$ to $O ( d ^ { 5 } )$ for SGS-OGD and $O ( d ^ { 3 }$ log d) to $O ( d ^ { 7 / 2 } \log d )$ for ONS.

## Appendix O. Derivation of the entry of Sakaue et al. (2025a) in Table 1

Sakaue et al. (2025a, Theorem 5.2) assumes the $\Delta { \mathrm { - g a p } }$ condition, namely that $\langle \theta ^ { * } , x - \hat { x } \rangle \geq \Delta \Vert x - \hat { x } \Vert$ holds for the agent’s optimal action x and every xˆ induced by some prediction, and bounds the sum $\widetilde { R } _ { T }$ by

$$
\widetilde { R } _ { T } \leq \frac { 2 ^ { 5 / 4 } L _ { \infty } B ^ { 3 } } { \lambda ^ { 3 / 2 } \Delta ^ { 2 } } ,
$$

where $L _ { \infty }$ (denoted K in that paper, renamed here because K is the number of mistakes in this paper) is an upper bound on $\| \hat { x } ^ { t } - x ^ { t } \|$ , the regularizer $\psi : \Theta \to$ R of their FTRL is λ-strongly convex with respect to the dual norm, and B is any constant with

$$
B ^ { 2 } \geq \operatorname* { m a x } \Big \{ 2 ^ { 5 / 2 } \lambda \operatorname* { m a x } _ { \theta , \theta ^ { \prime } \in \Theta } \| \theta - \theta ^ { \prime } \| _ { \star } ^ { 2 } , \operatorname* { m a x } _ { \theta , \theta ^ { \prime } \in \Theta } ( \psi ( \theta ) - \psi ( \theta ^ { \prime } ) ) \Big \} .
$$

The right-hand side is increasing in $B _ { ; }$ , so the smallest admissible B is taken.

For $\Theta = \Delta ^ { d - 1 }$ the authors take $\| \cdot \| = \| \cdot \| _ { \infty }$ on the actions, $\| \cdot \| _ { \star } = \| \cdot \| _ { 1 }$ on the weights, and the entropic regularizer $\psi ( \theta ) = \left. \theta , \log \theta \right.$ , which is 1-strongly convex with respect to $\| \cdot \| _ { 1 }$ by Pinsker’s inequality, so that $\lambda = 1$ . The two quantities in the definition of $B$ are then constants and log $d ,$ respectively: the $\ell _ { 1 }$ diameter of

$\Delta ^ { d - 1 }$ is $2 ,$ so the first is $2 ^ { 9 / 2 } ;$ and $\psi$ ranges over $[ - \log d , 0 ]$ on $\Delta ^ { d - 1 }$ , attaining 0 at a vertex $\mathbf { e } _ { i }$ and $- \log d$ at the barycenter $( 1 / d , \ldots , 1 / d )$ , so the second is log d. Hence $B ^ { 2 } = \operatorname* { m a x } \{ 2 ^ { 9 / 2 }$ , log $d \}$ , which is $\Theta ( \log d )$ as $d \to \infty$ . Finally, $L _ { \infty }$ is the $\ell _ { \infty }$ diameter of the feasible sets, which is at most $\| M \| _ { \infty }$ by Equation (6.3). Substituting these gives

$$
\widetilde { R } _ { T } = O \left( \frac { | | M | | _ { \infty } ( \log d ) ^ { 3 / 2 } } { \Delta ^ { 2 } } \right) ,
$$

which is the entry of Table $1 ;$ the same bound applies to $R _ { T } ^ { \mathrm { { e s t } } }$ by Equation (3.8). Two remarks are in order. First, the numerical constant is loose: Sakaue et al. (2025a) set $B = 2 ^ { 1 1 / 4 } { \sqrt { \log d } } .$ , which is admissible for $d \geq 2$ but larger than the smallest admissible value, although the order in $d$ is unafected. Second, the form depends on the choice of the regularizer and of the pair of norms; the entropic choice above is the one with which Sakaue et al. (2025a) recover the guarantee of Bärmann et al. (2018) on the probability simplex.

NEC Corporation<sub>,</sub> 1753 Shimonumabe<sub>,</sub> Nakahara-ku<sub>,</sub> Kawasaki<sub>,</sub> Kanagawa<sub>,</sub> Japan Email address: akira-kitaoka@nec.com