# Approximating Optimal Policies in Heterogeneous Asynchronous Multi-player Bandits

Sacha Arroues-Paykin Ecole Polytechnique

Etienne Boursier INRIA, LMO, Universite Paris-Saclay´

Vianney Perchet CREST, ENSAE

SACHA.ARROUES@POLYTECHNIQUE.EDU

ETIENNE.BOURSIER@INRIA.FR

VIANNEY.PERCHET@NORMALESUP.ORG

## Abstract

Motivated by applications to cognitive radio networks, multi-player multi-armed bandits (MMAB) have attracted considerable attention in recent years. The asynchronous variant, in which players are active only at a fraction of timesteps, better reflects real-world decentralized networks, where devices are not permanently active. Despite its practical relevance, this setting remains relatively understudied. Existing work on asynchronous MMAB has focused on homogeneous players, who share the same activation probability, a setting in which the optimal centralized allocation can be easily characterized. In contrast, efficiently computing the optimal allocation in the heterogeneous setting, where players have different activation probabilities, has remained an open problem. In this work, we investigate this problem and first establish its NP-hardness. We then turn to the greedy algorithm, which runs in polynomial time, and prove that it achieves a <sup>1</sup> -approximation of the optimal allocation. Furthermore, when players are processed in decreasing order of their activation probabilities, the approximation guarantee improves to $\scriptstyle { \frac { 1 } { \sqrt { e } } } .$ . Overall, our results establish the greedy algorithm as an efficient approximation to the optimal allocation, providing a practical oracle baseline for the design of online algorithms for heterogeneous asynchronous multi-player bandits.

Keywords: Multi-player multi-armed bandits, asynchronous multi-player bandits, resource allocation, approximation algorithms

## 1. Introduction

Multi-player multi-armed bandits (MMAB) provide a natural framework for modeling decentralized resource allocation in cognitive radio networks, where multiple devices compete for access to a shared set of communication channels. This problem extends the classical multi-armed bandit setting, in which a single player repeatedly selects an arm (e.g., a communication channel or frequency) from a finite set of options and receives a reward indicating whether a packet has been successfully transmitted. In the multi-player setting, each player independently selects an arm at every timestep. A player receives a reward only if they are the sole player to select that arm; otherwise, a collision occurs due to interference between players, and the reward is zero.

Originally introduced by Liu and Zhao (2010); Anandkumar et al. (2010), the standard MMAB problem has since attracted substantial interest in the literature (Rosenski et al., 2016; Avner and Mannor, 2014, 2019; Besson and Kaufmann, 2018). We refer the reader to Boursier and Perchet (2024) for a survey of MMAB. As first observed by Boursier and Perchet (2019) and further developed by Wang et al. (2020), collisions can also serve as a communication mechanism between players. By exploiting collision information, players can share information about their rewards and implement nearly decentralized strategies that mimic optimal centralized ones.

A key feature enabling—or at least facilitating—such communication through collisions is synchronization: all players are assumed to act simultaneously at every timestep. This assumption, however, is not representative of many real-world communication networks, where devices are not only decentralized but also asynchronous and heterogeneous. This has motivated growing interest in asynchronous multi-player multi-armed bandits (AMMAB), starting with Bonnefoi et al. (2017) and followed by, among others, Dakdouk (2022) and Richard et al. (2024).

In the asynchronous setting, only a subset of players is active at each timestep, with each player’s activity governed by an independent Bernoulli random variable. Active players can then select an arm of their choice. This additional source of uncertainty makes AMMAB substantially more challenging, both in terms of communication between players and in terms of the underlying learning objective, even for a centralized learner. Indeed, whereas the standard MMAB problem focuses on identifying the best N arms, where N is the number of players, and assigning one arm to each player, AMMAB involves a more complex resource-allocation problem. Players must be assigned to arms so as to balance the occupancy across them, where the occupancy of an arm is determined by the probability that exactly one player assigned to it is active at the next timestep.

Even the optimal allocation for a centralized agent with full knowledge of the problem parameters is therefore not straightforward to characterize. Dakdouk (2022) showed that, when players are homogeneous—that is, when they all have the same activation probability—a simple greedy algorithm yields an optimal allocation. This result subsequently paved the way for the development of no-regret algorithms for AMMAB with homogeneous players (Dakdouk, 2022; Richard et al., 2024). In practical decentralized networks, such as the Internet of Things (IoT), devices can have widely varying characteristics and, consequently, substantially different activation probabilities (Bonnefoi et al., 2017). Yet, the optimal offline allocation in this heterogeneous setting has remained unknown. In particular, Dakdouk (2022) conjectured that the corresponding optimization problem might be NP-hard. In this work, we investigate this open problem, motivated by its relevance to AMMAB with heterogeneous players.

Contributions. We first confirm the conjecture of Dakdouk (2022) by showing in Section 3 that the heterogeneous AMMAB allocation problem is NP-hard in full generality. Our proof relies on a reduction from the 3-partition problem.

Motivated by this hardness result, we then turn to polynomial-time algorithms that provide provable approximation guarantees for the optimal allocation. Focusing on the greedy algorithm, which is already known to be optimal in the homogeneous case (Dakdouk, 2022), we show in Section 4 that it achieves a 1 -approximation in the general heterogeneous setting, and that this guarantee is tight.

Interestingly, this approximation ratio holds regardless of the order in which players are processed. When players are instead processed in decreasing order of their activation probabilities, we obtain an improved approximation guarantee of $\textstyle { \frac { 1 } { \sqrt { e } } }$

Finally, in Section 5, we investigate the impact of decentralized decision-making on allocation efficiency through the price of anarchy. Without additional assumptions on the game, strategic equilibria can lead to an arbitrarily large loss in social welfare compared with the optimal centralized allocation. However, we show that this degradation can be bounded under reasonable assumptions on the arm means and the total activation probability of the players.

## 2. Preliminaries and notations

We are interested in the following instance of a centralized agent playing an $\mathbf { A M M A B } ^ { 1 }$ , that we will refer to as allocation problem: given N players with non-zero activation probabilities $p _ { 1 } , \dotsc , p _ { N } <$ 1 and K arms with means $\theta _ { 1 } , \ldots , \theta _ { K }$ , the goal is to allocate players to arms in order to maximize the expected payout.

The set of possible allocations is given by

$$
{ \mathcal { S } } = \{ ( S _ { 1 } , \ldots , S _ { K } ) \mid S _ { k } \subseteq [ N ] { \mathrm { ~ a n d ~ } } S _ { k } \cap S _ { k ^ { \prime } } = \emptyset { \mathrm { ~ i f ~ } } k \neq k ^ { \prime } \} .
$$

For an allocation in $s , S _ { k }$ represents the set of players assigned to arm k. Note that some players may be assigned to no arm, or equivalently, be assigned to a “dummy” arm of mean 0.

When choosing an allocation, following AMMAB convention on a single timestep:

• each player i is active with probability $p _ { i }$ , independently of other players;

• the payout of arm k is then drawn according to some random variable of expected value $\theta _ { k } ;$

• the payout of arm k is only collected when a single player allocated to arm k is active, since multiple active players would lead to a collision event.

Consequently, the expected payout of an allocation $S \in S$ is given by

$$
\begin{array} { r l } & { \mu _ { \boldsymbol { \theta } } ( S ) = \displaystyle \sum _ { k \in [ K ] } \theta _ { k } f ( S _ { k } ) } \\ { \mathrm { w h e r e } \quad } & { f ( S _ { k } ) = \displaystyle \sum _ { i \in S _ { k } } \frac { p _ { i } } { 1 - p _ { i } } \displaystyle \prod _ { i \in S _ { k } } ( 1 - p _ { i } ) . } \end{array}
$$

The total expected payout is the sum of the expected payout of each arm k: $\theta _ { k }$ is the average payout of the arm, while $f ( S _ { k } )$ is the probability that exactly one player is active on arm $k .$ . For $k \in [ K ]$ we will throughout refer to $\textstyle \sum _ { i \in S _ { k } } { \frac { p _ { i } } { 1 - p _ { i } } }$ as the load on arm k. An arm with load $l \geq 1$ is referred to as saturated. In the offline problem, we seek to find, or approximate, the optimal allocation:

$$
\operatorname* { m a x } _ { S \in S } \mu _ { \theta } ( S ) .\tag{1}
$$

We assume without loss of generality<sup>2</sup> that $1 \geq \theta _ { 1 } \geq \cdot \cdot \cdot \geq \theta _ { K } > 0$ . Given an allocation $S ,$ the marginal gain from adding a player j to arm k is exactly

$$
\begin{array} { l } { \displaystyle { \mu _ { \theta } ( S _ { 1 } , \dots , S _ { k } \cup \{ j \} , \dots , S _ { K } ) - \mu _ { \theta } ( S ) = \theta _ { k } ( f ( S _ { k } \cup \{ j \} ) - f ( S _ { k } ) ) } } \\ { \displaystyle { \qquad = \theta _ { k } p _ { j } \prod _ { i \in S _ { k } } ( 1 - p _ { i } ) \left( 1 - \sum _ { i \in S _ { k } } \frac { p _ { i } } { 1 - p _ { i } } \right) . } } \end{array}\tag{2}
$$

This leads to the natural definition of the greedy algorithm, given by Algorithm 1 below (Dakdouk, 2022). It processes players one by one according to some fixed order—or, equivalently, permutation σ—of the players and allocates each incoming player to the arm that improves the current allocation the most. We assume without loss of generality that the players are processed in the natural order in Algorithm 1, i.e., $\sigma = \mathrm { I d }$

```latex
Algorithm 1: Greedy algorithm
Initialize $S _ { k } \gets \emptyset$ for any $k \in [ K ]$
for $n = 1 , \ldots , N$ do
if ma $\begin{array} { r } { \mathfrak { c } _ { \ell \in [ K ] } \theta _ { \ell } \prod _ { i \in S _ { \ell } } ( 1 - p _ { i } ) \left( 1 - \sum _ { i \in S _ { \ell } } \frac { p _ { i } } { 1 - p _ { i } } \right) > 0 } \end{array}$ then
$\begin{array} { r } { k \gets \arg \operatorname* { m a x } _ { \ell \in [ K ] } \theta _ { \ell } \prod _ { i \in S _ { \ell } } ( 1 - p _ { i } ) \left( 1 - \sum _ { i \in S _ { \ell } } \frac { p _ { i } } { 1 - p _ { i } } \right) } \end{array}$
$S _ { k } \gets S _ { k } \cup \{ n \}$ // allocates player n to arm k
end
end
return $( S _ { 1 } , \ldots , S _ { K } )$
```

A natural variant of the standard greedy algorithm is to process players in decreasing order of their activation probabilities, which Dakdouk (2022) introduces as descending-order reward greedy (DORG). Equivalently, it assumes that the players are indexed such that $p _ { 1 } \geq . . . \geq p _ { N }$ . DORG can also be viewed as a greedy algorithm that, rather than selecting for the upcoming player the arm that yields the largest improvement in the current allocation, selects at each step the most improving (player, arm) pair to add to the current allocation. The greedy algorithm runs in time $\mathcal { O } ( ( K + N )$ log K) by maintaining the arms in a max-heap keyed on their current marginal<sup>3</sup>. DORG has additional runtime O(N log N) from ordering players beforehand.

Dakdouk (2022) proves that, for some (unknown) ordering of the players, the greedy Algorithm 1 is optimal. In particular, this means that for homogeneous players, i.e., in the case where $p _ { i } = p$ for all i, greedy yields an optimal solution to the allocation problem (1). Indeed, in that case, the players’ order has no influence on the resulting allocation, by symmetry of the players. In the heterogeneous case however, finding this precise order can be difficult; and Dakdouk (2022) conjectures the allocation problem (1) is NP-hard in the general case.

## 3. NP-hardness of the allocation problem

In this section, we prove that the allocation problem (1) is NP-hard in general. This reveals a structural difference between homogeneous instances, for which a simple greedy algorithm yields an optimal allocation, and heterogeneous ones. It also motivates the search for polynomial-time approximation oracles carried out in Section 4.

We proceed by reduction from the well-known NP-complete 3-partition problem (Garey and Johnson, 1979) defined as follows.

Definition 1 (3-PARTITION PROBLEM). Let $m \geq 1$ and $B \geq 1$ be integers, and let $a _ { 1 } , \ldots , a _ { 3 m }$ be positive integers satisfying $\begin{array} { r } { \frac { B } { 4 } \ < \ a _ { i } \ < \ \frac { B } { 2 } } \end{array}$ for all i and $\textstyle \sum _ { i \leq 3 m } a _ { i } = m B$ . The 3-partition problem consists in deciding whether [3m] can be partitioned into m sets $T _ { 1 } , \ldots , T _ { m }$ such that $\textstyle \sum _ { i \in T _ { k } } a _ { i } = B$ for all k $\in [ m ]$

The constraint $\begin{array} { r } { \frac { B } { 4 } < a _ { i } < \frac { B } { 2 } } \end{array}$ forces each $T _ { k }$ to contain exactly three elements, hence the 3-partition name.

Theorem 1. 3-partition reduces in polynomial time to the following decision version of the allocation problem (1): given $( p _ { i } ) _ { i \le N } , ( \theta _ { k } ) _ { k \le K }$ and a threshold $\tau ^ { \star }$ , decide whether max $S \in S \mu _ { \theta } ( S ) \geq \tau ^ { \star }$ In particular, the allocation problem (1) is NP-hard.

Proof. Let $( B , a _ { 1 } , \dots , a _ { 3 m } )$ be an instance of 3-partition, and let $\begin{array} { r } { Q = \sum _ { i \leq 3 m } a _ { i } ^ { 2 } } \end{array}$ . We set

$$
\varepsilon = \frac { 1 } { 2 m ^ { 3 } B ^ { 3 } }
$$

and construct the following instance of the allocation problem:

$K = m$ arms, all of mean $\theta _ { k } = 1$

$N = 3 m$ players, with activation probabilities $p _ { i } = a _ { i } \varepsilon$ for $i \in [ 3 m ] ;$

• a threshold $\tau ^ { \star } = m B \varepsilon + ( Q - m B ^ { 2 } - 1 ) \varepsilon ^ { 2 } .$

All these quantities are rationals whose numerators and denominators are polynomial in m and $B ,$ so the construction is computable in time polynomial in the size of the 3-partition instance. Moreover $\begin{array} { r } { p _ { i } < \frac { 1 } { 4 m ^ { 3 } B ^ { 2 } } \le \frac { 1 } { 4 } } \end{array}$ , so all $p _ { i }$ are valid probabilities. We show that

$$
\operatorname* { m a x } _ { S \in S } \mu _ { \theta } ( S ) \geq \tau ^ { \star } \iff ( B , a _ { 1 } , \ldots , a _ { 3 m } ) { \mathrm { ~ a d m i t s ~ a ~ } } 3 \mathrm { - p a r t i t i o n } .\tag{3}
$$

Let $S \in S$ be any allocation. Since $\begin{array} { r } { p _ { i } \le \frac { 1 } { 4 } } \end{array}$ , we have $\frac { p _ { i } } { 1 - p _ { i } } \leq 2 p _ { i }$ , and the load of any arm satisfies

$$
\begin{array} { c } { { \displaystyle \sum _ { i \in S _ { k } } \frac { p _ { i } } { 1 - p _ { i } } \le 2 \varepsilon \sum _ { i \le 3 m } a _ { i } } } \\ { { = 2 \varepsilon m B < 1 } } \end{array}
$$

By Equation (2), the marginal gain of adding an unassigned player to any arm is then strictly positive. Consequently, any optimal allocation uses all players. Let $S \in S$ be an allocation using all players, and denote $\textstyle { \beta _ { k } = \sum _ { i \in S _ { k } } a _ { i } }$ . For any $i \in S _ { k }$

$$
1 - ( \varepsilon \beta _ { k } - p _ { i } ) \le \prod _ { j \in S _ { k } , j \neq i } ( 1 - p _ { j } )
$$

by union bound, and since $\begin{array} { r } { ( 1 - x ) \le e ^ { - x } \le 1 - x + \frac { x ^ { 2 } } { 2 } } \end{array}$ for all $x \geq 0$

$$
\prod _ { \substack { j \in S _ { k } , j \neq i } } ( 1 - p _ { j } ) \leq e ^ { - ( \varepsilon \beta _ { k } - p _ { i } ) } \leq 1 - ( \varepsilon \beta _ { k } - p _ { i } ) + \frac { ( \varepsilon \beta _ { k } - p _ { i } ) ^ { 2 } } { 2 }
$$

Multiplying by $p _ { i }$ and summing over $i \in S _ { k }$ , we obtain,

$$
0 \leq f ( S _ { k } ) - \Big ( \varepsilon \beta _ { k } - ( \varepsilon \beta _ { k } ) ^ { 2 } + \sum _ { i \in S _ { k } } p _ { i } ^ { 2 } \Big ) \leq \sum _ { i \in S _ { k } } p _ { i } \frac { ( \varepsilon \beta _ { k } - p _ { i } ) ^ { 2 } } { 2 } \leq \frac { ( \varepsilon \beta _ { k } ) ^ { 3 } } { 2 } .
$$

Finally,

$$
f ( S _ { k } ) = \varepsilon \beta _ { k } + \varepsilon ^ { 2 } \Big ( \sum _ { i \in S _ { k } } a _ { i } ^ { 2 } - \beta _ { k } ^ { 2 } \Big ) + R _ { k } , \qquad \mathrm { w i t h ~ } 0 \le R _ { k } \le \frac { \varepsilon ^ { 3 } \beta _ { k } ^ { 3 } } { 2 } .\tag{4}
$$

Since $\Sigma _ { k } \beta _ { k } = m B$ , summing Equation (4) over the m arms yields

$$
\mu _ { \theta } ( S ) = m B \varepsilon + \Big ( Q - \sum _ { k \leq m } \beta _ { k } ^ { 2 } \Big ) \varepsilon ^ { 2 } + R ,\tag{5}
$$

where

$$
\begin{array} { c } { { 0 \leq R \leq \displaystyle \frac { \varepsilon ^ { 3 } } { 2 } \sum _ { k \leq m } \beta _ { k } ^ { 3 } } } \\ { { \leq \displaystyle \frac { \varepsilon ^ { 3 } } { 2 } m ^ { 3 } B ^ { 3 } = \displaystyle \frac { \varepsilon ^ { 2 } } { 4 } . } } \end{array}
$$

Moreover, since $\begin{array} { r } { \sum _ { k } ( \beta _ { k } - B ) = 0 } \end{array}$

$$
\sum _ { k \leq m } \beta _ { k } ^ { 2 } = m B ^ { 2 } + \sum _ { k \leq m } ( \beta _ { k } - B ) ^ { 2 } .
$$

As such,

• If $\beta _ { k } = B$ for all k, then Equation (5) gives

$$
\mu _ { \theta } ( S ) \geq m B \varepsilon + ( Q - m B ^ { 2 } ) \varepsilon ^ { 2 } > \tau ^ { \star } .
$$

• Otherwise, the integers $( \beta _ { k } - B ) _ { k \le m }$ sum to 0 and are not all zero, so at least two of them are non-zero and $\begin{array} { r } { \sum _ { k } ( \beta _ { k } - B ) ^ { 2 } \ge \overline { { 2 } } } \end{array}$ . Then Equation (5) gives

$$
\mu _ { \theta } ( S ) \leq m B \varepsilon + ( Q - m B ^ { 2 } - 2 ) \varepsilon ^ { 2 } + \frac { \varepsilon ^ { 2 } } { 4 } < \tau ^ { \star } .
$$

If the instance admits a 3-partition $T _ { 1 } , \ldots , T _ { m }$ , then $S = ( T _ { 1 } , \dots , T _ { m } )$ is an allocation using all players with $\beta _ { k } = B$ for all $k ,$ , and therefore $\mu _ { \theta } ( S ) > \tau ^ { \star }$

Conversely, assume max $S \in S \mu _ { \theta } ( S ) \geq \tau ^ { \star }$ . By the above, an optimal allocation must satisfy $\beta _ { k } = B$ for all k. The sets $S _ { 1 } , \ldots , S _ { m }$ then partition [3m] with $\textstyle \sum _ { i \in S _ { k } } a _ { i } = B$ for all k, i.e., they form a 3-partition.

This proves Equation (3) and concludes the reduction.

## 4. Approximation oracles for the allocation problem

Given the NP-hardness of the general allocation problem (1), it is reasonable to look for a computationally efficient oracle providing an approximation of the optimal expected payout. In particular, we aim in this section to provide polynomial time α-approximation oracles for some $\alpha \in [ 0 , 1 ]$ i.e., algorithms that output $\widehat { S } = ( \widehat { S } _ { 1 } , \ldots , \widehat { S } _ { K } )$ such that for any problem instance $( p _ { 1 } , \ldots , p _ { N } )$ and $\theta _ { 1 } , \ldots , \theta _ { K }$

$$
\mu _ { \theta } ( \widehat { S } ) \geq \alpha \operatorname* { m a x } _ { S \in { \mathcal { S } } } \mu _ { \theta } ( S ) .
$$

The greedy Algorithm 1 described in Section 2 is a natural candidate, and is already known to be optimal for homogeneous instances.

## 4.1. Maximization over matroids

The allocation problem (1) can be framed as maximizing a set function over player-arm pairs. Indeed, for $A \ \subseteq \ [ N ] \times [ K ]$ , we can define $A _ { k } \ = \ \{ i \ \in \ [ N ] \ | \ ( i , k ) \ \in \ A \}$ and $\mu _ { \theta } ( A ) \ =$ $\textstyle \sum _ { k \in [ K ] } \theta _ { k } f ( A _ { k } )$

Problem (1) can then be seen as a maximization problem over subsets of $E = [ N ] \times [ K ]$ that assign at most one arm per player. Let us denote $\mathcal { I } \subseteq 2 ^ { E }$ the nonempty collection of such possible allocations. The collection $\mathcal { T }$ has a specific, useful structure: it is a matroid,<sup>4</sup> i.e., it satisfies the following three properties

1. it contains the empty set: $\varnothing \in { \mathcal { Z } } ;$

2. hereditary property: for any $A , A ^ { \prime }$ with $A ^ { \prime } \subset A , A \in { \mathcal { Z } } \implies A ^ { \prime } \in { \mathcal { Z } } ;$

3. augmentation property: for any $A , B \in { \mathcal { T } }$ with $| B | > | A |$ , there exists $x \in B \backslash$ A such that $A \cup \{ x \} \in { \mathcal { T } } .$

Maximizing set functions under a matroid constraint has attracted quite some interest in the optimization literature, with a particular focus on the approximation ratio of greedy algorithms. However, existing works typically rely on two key assumptions on the objective function: monotonicity and submodularity. A set function $F : 2 ^ { E } \to \mathbb { R }$ is said to be monotone if

$$
A \subseteq B \implies F ( A ) \leq F ( B ) ;
$$

and submodular if

$$
A \subseteq B { \mathrm { ~ a n d ~ } } e \in E \setminus B \implies F ( A \cup \{ e \} ) - F ( A ) \geq F ( B \cup \{ e \} ) - F ( B ) .
$$

Monotone submodular maximization. When the objective function is both monotone and submodular, the greedy algorithm is known to be a <sup>1</sup>-approximation under any matroid constraint (Fisher et al., 1978). In the case of a stronger, partition matroid constraint, its local greedy variant is also a ${ \frac { 1 } { 2 } } \mathrm { - a p p r o x i m a t i o n }$ (Goundan and Schulz, 2007). Finally, Calinescu et al. (2011) showed that a continuous relaxation of the greedy algorithm reaches a $( 1 - { \frac { 1 } { e } } )$ -approximation ratio under matroid constraints.

Non-monotone submodular maximization. When the objective function is only submodular, several polynomial-time algorithms have been developed in the literature. In the unconstrained setting, these include local search (Feige et al., 2011) and the double-greedy algorithm (Buchbinder et al., 2015). Under matroid constraints, randomized algorithms using multilinear relaxations have led to approximation ratio guarantees as high as 0.401 (Buchbinder and Feldman, 2019, 2024).

On the other hand, Gharan and Vondrak´ (2011) established an upper bound for the matroidconstrained setting. More precisely, they showed that no algorithm making only a polynomial number of value queries can guarantee an approximation ratio greater than 0.478 in full generality.

Allocation problem (1). By Equation (2), the marginal gain of adding a player to arm k has the sign of $\textstyle 1 - \sum _ { i \in A _ { k } } { \frac { p _ { i } } { 1 - p _ { i } } }$ , and is thus negative as soon as the arm is past saturation. This observation makes our objective function non-monotone, whenever loads larger than 1 are possible.

Moreover, adding a player j to arm k changes the marginal gain of any other player i on that arm by

$$
- \theta _ { k } p _ { i } p _ { j } \prod _ { l \in A _ { k } } ( 1 - p _ { l } ) \bigg ( 2 - \sum _ { l \in A _ { k } } \frac { p _ { l } } { 1 - p _ { l } } \bigg ) ,
$$

which is positive once the load exceeds 2. This observation makes our objective function nonsubmodular whenever loads larger than 2 are possible.

Consequently, the problem considered here does not satisfy the usual assumptions of monotonicity and submodularity; see Lemma 3 in Appendix A for further details. We therefore rely, in the remainder of this section, on custom arguments that exploit the specific structure of the allocation problem to derive approximation guarantees for the greedy Algorithm 1.

## 4.2. Approximation bound of greedy

Let $p _ { 1 } , \ldots , p _ { N }$ and $\theta = ( \theta _ { 1 } , \ldots , \theta _ { K } )$ be fixed throughout. In this subsection, we provide a lower bound on the performance of the greedy Algorithm 1, depending on the maximum activation probability among all players.

Theorem 2. For $q = \mathrm { m a x } _ { i \leq N } p _ { i }$ , the greedy Algorithm 1 is a e<sup>−q</sup>-approximation oracle for the allocation problem (1), regardless of the order in which players are processed. In particular, greedy is an instance-independent <sup>1</sup><sub>e</sub> -approximation.

Sketch of proof. We distinguish two cases: one in which the greedy Algorithm 1 saturates all arms (and potentially has some unassigned players), and one where it does not. The former is tackled in Appendix B, in which we show that said instances yield an even larger approximation ratio, given by $\begin{array} { r } { \frac { 1 } { \lceil \frac { 1 - q } { q } \rceil e \frac { q } { 1 - q } \cdot ( 1 - q ) ^ { \lceil \frac { 1 - q } { q } \rceil } } \geq e ^ { - q } } \end{array}$

We now assume greedy does not saturate all arms, and therefore in particular assigns all players. The proof hinges on two main results:

• For any $S _ { k } \subseteq [ N ]$ such that $\begin{array} { r } { \sum _ { i \in S _ { k } } p _ { i } = m _ { k } } \end{array}$ for a fixed mass $m _ { k }$ , the expected payout of arm k has closed-form upper and lower bounds that solely depend on q and $m _ { k }$ . If all players have activation probability at most $q ,$ , then for such $S _ { k }$

$$
f ( S _ { k } ) \leq m _ { k } e ^ { q - m _ { k } } .
$$

If moreover the mass $m _ { k }$ is at most 1,

$$
f ( S _ { k } ) \geq m _ { k } e ^ { - m _ { k } }
$$

A proof can be found in Appendix C. Informally, the bounds establish that the best way to reach mass $m _ { k }$ is to have few players with large activation probabilities, and the worst when $m _ { k } \le 1$ is to have a very large number of players with small activation probabilities.

• For any per-arm mass profile $c _ { 1 } , \dotsc , c _ { K } \in [ 0 , 1 ]$ such that $\textstyle \sum _ { k } c _ { k } \leq \sum _ { i } p _ { i }$ , there exists an allocation ${ \bar { S } } ,$ on a set of players potentially different from $p _ { 1 } , \ldots , p _ { N }$ , that puts on each arm k an activation mass of exactly $c _ { k }$ , and has a worse expected payout than the allocation given by the greedy Algorithm 1. See Appendix D for a full statement and proof.

Putting both of the above statements together, we can construct an allocation $\bar { S }$ that has worse payout than greedy and matches the capped per-arm mass profile of the optimal allocation $c _ { k } =$ $\mathrm { m i n } \{ m _ { k } ^ { \mathrm { o p t } } , 1 \}$ . Then, using the first point, we can bound the payouts of S<sup>¯</sup> on each arm. Specifically, if we denote $q = \operatorname* { m a x } _ { i } p _ { i }$ , and $\bar { \mu } _ { \theta }$ the payout function for the potentially distinct set of players used in ${ \bar { S } } ,$ we obtain the following bound

$$
\mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) \geq \bar { \mu } _ { \theta } ( \bar { S } ) \geq e ^ { - q } \mu _ { \theta } ( S _ { \mathrm { o p t } } ) .\tag{6}
$$

We refer to Appendix E for a complete proof.

Theorem 2 establishes an instance-dependent approximation ratio of $e ^ { - q }$ that depends on $q \ =$ $\operatorname* { m a x } _ { i \in [ N ] } p _ { i }$ . In full generality, we only have $q < 1$ , hence the general <sup>1</sup>-approximation bound. e

However, this approximation ratio becomes closer to 1 as q gets smaller. In particular, Theorem 2 establishes that greedy is almost optimal as the maximal activation probability goes to $0 ,$ which is close to practice in an IoT context (Bonnefoi et al., 2017).

Having now obtained a first result regarding greedy’s payout compared to the optimum, one may wonder whether the instance-independent $\frac { 1 } { e }$ approximation lower bound is tight.

Lemma 1. Let $\varepsilon > 0$ . There exists an instance of the allocation problem, with $K = 2$ arms and $N _ { \varepsilon }$ players such that $\begin{array} { r } { \frac { \mu _ { \theta } ( S _ { G r e e d y } ) } { \mu _ { \theta } ( S _ { o p t } ) } \leq \frac { 1 } { e } + \varepsilon } \end{array}$

Sketch of proof. Let $N \in  { \mathbb { N } } _ { \geq 3 }$ . We construct a two arm instance with $\theta _ { 1 } = 1 , \theta _ { 2 } = \textstyle \frac { 1 } { 3 N }$ and activation probabilities: $\begin{array} { r } { p _ { 1 } = \cdot \cdot \cdot = p _ { N - 1 } = \frac { 1 } { N } } \end{array}$ and $\begin{array} { r } { p _ { N } = 1 - \frac { 1 } { N } } \end{array}$ . We show in Appendix F that, on these instances, the greedy algorithm assigns the first $N - 1$ players to the first arm and the last player to the second, whereas the optimal allocation makes the opposite assignment. This leads to an exact ratio of

$$
\frac { \mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) } { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) } = \frac { ( 1 - \frac { 1 } { N } ) ^ { N - 1 } + \frac { 1 - \frac { 1 } { N } } { 3 N } } { 1 - \frac { 1 } { N } + \frac { ( 1 - \frac { 1 } { N } ) ^ { N - 1 } } { 3 N } } \xrightarrow [ N  \infty ] { } \frac { 1 } { e } .
$$

One therefore cannot hope for a better approximation with the greedy algorithm in all generality. However, as seen in the proof of Lemma 1, pathological cases in which the greedy Algorithm 1 performs the worst seem to be those in which the most active players are seen last, and therefore cannot be assigned to the best arms.

This shortcoming is in particular addressed if greedy processes players in decreasing order of activation probability, i.e., DORG.

## 4.3. Approximation bound for DORG

As a particular instance of the greedy Algorithm 1, DORG is already known to be a $\textstyle { \frac { 1 } { e } }$ -approximation algorithm, thanks to Theorem 2. Particularly motivated by the upper bound examples of Lemma 1, we study in this section whether DORG can lead to an improved approximation ratio, with respect to proceeding in a worst case order of the players.

Theorem 3. DORG is a 1 -approximation oracle for the allocation problem (1). √<sub>e</sub>

Sketch of proof. We show that DORG and an optimal allocation assign players of activation probability $p > \frac { 1 } { 2 }$ the same way. Furthermore, both DORG and optimal allocations have such players as sole occupants of their respective arms. From there, DORG can only differ from said optimal allocation on players with activation probabilities smaller than ${ \frac { 1 } { 2 } } ;$ , and on a subset of arms. The instance-dependent bound of Theorem 2 then yields a $\textstyle { \frac { 1 } { \sqrt { e } } }$ approximation ratio on those players. See Appendix G for a full proof.

As the proof sketch above illustrates, the key advantage of DORG lies in how it processes players with activation probabilities larger than $\frac { 1 } { 2 }$ . Indeed, the $\textstyle { \frac { 1 } { \sqrt { e } } }$ approximation guarantee holds for any greedy algorithm that processes players with activation probabilities greater than $\frac { 1 } { 2 }$ in decreasing order, regardless of the order in which players with smaller activation probabilities are then processed. In other words, the guarantee does not rely on DORG processing players with small activation probabilities in decreasing order. We conjecture that exploiting this additional structure in the analysis could lead to an improved approximation guarantee for DORG.

## 5. Online and decentralized considerations

The section above proves that we have at hand a deterministic α-oracle for some $\alpha \geq { \frac { 1 } { \sqrt { e } } }$ , that might depend on the considered instance. Fixing such an α, if player activation probabilities are known and arm means are not, and given the payout structure of the allocation problem, we are in the scope of combinatorial bandits with probabilistically triggered arms. The seminal work of Wang and Chen (2017) then bounds the (α, 1)-approximation regret of the CUCB algorithm by

$$
\mathrm { R e g } _ { \theta , \alpha , 1 } ( T ) = \mathcal { O } \left( \sqrt { K \operatorname* { m i n } \{ K , N \} T \ln T } \right) .
$$

We refer to Appendix H for further details, and additional distribution dependent bounds.

The direct use of CUCB mentioned above relies on a central learning agent, that can unilaterally assign players to different arms at each round. The AMMAB problem is however mostly motivated by a decentralized application, where each player corresponds to a single device. In the cooperative decentralized case, it is still possible to adapt centralized learning strategies, either via the direct use of communication channels (Richard et al., 2024), or via the use of collision information and sophisticated protocols (Boursier and Perchet, 2019).

Consequently, typical MMAB works can be adapted to the baseline of the greedy algorithm stated above for deriving decentralized cooperative learning algorithms. We instead focus in the next subsection on the non-cooperative decentralized instance, where players aim to maximize their own expected payout. Here again, the offline problem already provides a lot of interesting insights on this case.

## 5.1. Decentralized play and price of anarchy

In this section, we establish how the absence of a central controller affects the total expected payout in an offline case, via the study of the price of anarchy.

We therefore now let the players choose their arms themselves. Throughout this subsection, the arm means $\theta _ { 1 } \ge \dots \ge \theta _ { K } > 0$ and the activation probabilities $p _ { 1 } , \ldots , p _ { N }$ are fixed and known to

all players. We wish to know by how much overall payout is reduced when the allocation emerges from selfish behavior rather than from centralized policies.

We define the decentralized allocation problem as follows: each player $j \leq N$ chooses an arm $s _ { j } \in [ K ]$ . With choices $\boldsymbol s = ( s _ { 1 } , \ldots , s _ { N } )$ , the allocation is $S ( s ) = ( S _ { 1 } , \ldots , S _ { K } )$ with $S _ { k } = \{ j \ \vert$ $s _ { j } = k \}$ . The expected payout of player j is therefore

$$
\begin{array} { l } { { u _ { j } ( s ) = p _ { j } \theta _ { s _ { j } } \displaystyle \prod _ { i \neq j , s _ { j } = s _ { i } } ( 1 - p _ { i } ) } } \\ { { = p _ { j } \theta _ { s _ { j } } \displaystyle \frac { z _ { s _ { j } } } { 1 - p _ { j } } } } \end{array}
$$

where $\begin{array} { r } { z _ { k } = \prod _ { i , s _ { i } = k } ( 1 - p _ { i } ) } \end{array}$

A profile s is a pure Nash equilibrium if no player gains by switching arms, i.e., if for every player j and every arm $k \neq s _ { j }$

$$
\theta _ { s _ { j } } \frac { z _ { s _ { j } } } { 1 - p _ { j } } \geq \theta _ { k } z _ { k } .
$$

Let $\mathcal { N }$ be the set of pure Nash equilibria (see Appendix I for a proof that ${ \mathcal { N } } \neq \emptyset )$ and $S _ { \mathrm { o p t } }$ a centralized optimal allocation. The price of anarchy and the price of stability (Koutsoupias and Papadimitriou, 1999; Anshelevich et al., 2008) are defined as

$$
\mathrm { P o A } = \frac { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) } { \operatorname* { m i n } _ { s \in \mathcal { N } } \mu _ { \theta } ( S ( s ) ) } , \qquad \mathrm { P o S } = \frac { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) } { \operatorname* { m a x } _ { s \in \mathcal { N } } \mu _ { \theta } ( S ( s ) ) } .
$$

Both the price of anarchy (PoA) and the price of stability (PoS) quantify the cost of decentralization, or lack of cooperation, among agents. Specifically, the PoA is the ratio between the social welfare achieved by an optimal allocation and that achieved at the worst Nash equilibrium, while the PoS uses the best Nash equilibrium instead.

Since all players opt to play (their expected payout is always strictly positive), situations with many players and few arms are problematic due to arms populating past saturation. It leads to the following unboundedness of both PoA and PoS.

Lemma 2. Both the price of anarchy and price of stability are unbounded for unconstrained instances ofthe decentralized allocation problem.

Sketch of proof. For $N \geq 2 .$ , the instance with two arms of means $\theta _ { 1 } = 1 , \theta _ { 2 } = 4 ^ { - N }$ and N players of activation probability $\frac { 1 } { 2 }$ satisfies Po $\textstyle \mathrm { { A } = \mathrm { { P o S } } } \geq { \frac { 2 ^ { N - 1 } } { N } }$ . See Appendix J for a full proof.

The proof of Lemma 2 constructs decentralized scenarios in which removing the central controller can lead to an arbitrarily large loss in social welfare. However, these scenarios rely on extreme settings in which one arm is much better than all others, and there is enough total player load to push said arm well past saturation.

Instead, we now investigate a constrained but realistic case in which all arm means are in some fixed interval, and therefore players distribute more evenly over available arms. As such, if the total capacity on arms is roughly equal to the total load of players, we can reasonably hope for a bounded PoA.

Theorem 4. Assume that max<sub>i</sub> $p _ { i } = q < 1$ and $\theta _ { k } \in [ \bar { \theta } - \varepsilon _ { 1 } , \bar { \theta } + \varepsilon _ { 2 } ]$ with $0 \leq \varepsilon _ { 1 } , \varepsilon _ { 2 } < \bar { \theta }$ and some $\bar { \theta } > 0 .$ . Then every pure Nash equilibrium s ofthe allocation game satisfies

$$
\mu _ { \theta } ( S ( s ) ) \geq ( 1 - q ) ( \bar { \theta } - \varepsilon _ { 1 } ) L e ^ { - x } , \qquad \mu _ { \theta } ( S _ { o p t } ) \leq \frac { \bar { \theta } + \varepsilon _ { 2 } } { 1 - q } K h \bigl ( ( 1 - q ) x \bigr )
$$

and

$$
\mathrm { P o A } \leq \frac { \bar { \theta } + \varepsilon _ { 2 } } { \bar { \theta } - \varepsilon _ { 1 } } \cdot \left\{ \begin{array} { l l } { \frac { e ^ { q x } } { 1 - q } } & { i f x \leq \frac { 1 } { 1 - q } , } \\ { \frac { 1 } { ( 1 - q ) ^ { 2 } } \cdot \frac { e ^ { x - 1 } } { x } } & { i f x \geq \frac { 1 } { 1 - q } , } \end{array} \right.
$$

where $\begin{array} { r } { L = \sum _ { i \leq N } \frac { p _ { i } } { 1 - p _ { i } } } \end{array}$ is the total load, $\begin{array} { r } { x = \frac { L } { K } } \end{array}$ is the average load and $h ( t ) = g ( \operatorname* { m i n } ( t , 1 ) )$ , for $g : t \mapsto t e ^ { - t }$

In particular, Theorem 4 implies that decentralized selfish policies are asymptotically optimal in mean-field unsaturated regimes with equal arm means; and more generally, they incur a low price of anarchy (and price of stability) when the arms have relatively similar means. The complete proof is given in Appendix K.

## 6. Conclusion

In this work, we established the NP-hardness of the heterogeneous allocation problem and showed that the allocations produced by the greedy Algorithm 1 and DORG achieve expected payouts within multiplicative factors of the optimum. While the approximation guarantee for the greedy algorithm is tight, we believe that a more refined analysis could yield improved guarantees for DORG.

Our approximation guarantees provide a foundation for adapting centralized learning algorithms, such as CUCB and potentially the Cautious Greedy algorithm of Richard et al. (2024), to the heterogeneous asynchronous setting.

Extending these results to the decentralized setting remains a major open challenge. Without a central controller, communication is no longer free, and adapting communication protocols such as those introduced by Boursier and Perchet (2019) becomes more challenging in an asynchronous environment. Heterogeneous activation probabilities introduce an additional difficulty: players no longer play symmetric roles in the communication process. Efficient protocols must therefore balance rapid information exchange between frequently active players with the ability to conduct multiple exchanges in parallel. Addressing these challenges is an important direction for future work toward designing efficient learning policies for asynchronous multi-player bandits with heterogeneous players.

## AI Disclosure

On top of routine assistance with spelling, and more general proofreading of the present paper, LLMs (Claude Opus, used through Claude Code) were used to identify the 3-partition problem as a good candidate for the reduction argument of Section 3. The LLM’s attempt at proving said reduction was however incorrect, and the proof found in Section 3 was obtained by hand.

## References

Animashree Anandkumar, Nithin Michael, and Ao Tang. Opportunistic spectrum access with multiple users: Learning under competition. In 2010 Proceedings IEEE INFOCOM, pages 1–9. IEEE, 2010.

Elliot Anshelevich, Anirban Dasgupta, Jon Kleinberg, Eva Tardos, Tom Wexler, and Tim Rough-<sup>´</sup> garden. The price of stability for network design with fair cost allocation. SIAM journal on computing, 38(4):1602–1623, 2008.

Orly Avner and Shie Mannor. Concurrent bandits and cognitive radio networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 66–81. Springer, 2014.

Orly Avner and Shie Mannor. Multi-user communication networks: A coordinated multi-armed bandit approach. IEEE/ACM Transactions on Networking, 27(6):2192–2207, 2019.

Lilian Besson and Emilie Kaufmann. Multi-player bandits revisited. In Algorithmic Learning Theory, pages 56–92. PMLR, 2018.

Remi Bonnefoi, Lilian Besson, Christophe Moy, Emilie Kaufmann, and Jacques Palicot. Multi- ´ armed bandit learning in iot networks: Learning helps even in non-stationary settings. In International Conference on Cognitive Radio Oriented Wireless Networks, pages 173–185. Springer, 2017.

Etienne Boursier and Vianney Perchet. Sic-mmab: Synchronisation involves communication in multiplayer multi-armed bandits. Advances in Neural Information Processing Systems, 32, 2019.

Etienne Boursier and Vianney Perchet. A survey on multi-player bandits. Journal of Machine Learning Research, 25(137):1–45, 2024.

Niv Buchbinder and Moran Feldman. Constrained submodular maximization via a nonsymmetric technique. Mathematics ofOperations Research, 44(3):988–1005, 2019.

Niv Buchbinder and Moran Feldman. Constrained submodular maximization via new bounds for dr-submodular functions. In Proceedings ofthe 56th annual ACM symposium on theory ofcomputing, pages 1820–1831, 2024.

Niv Buchbinder, Moran Feldman, Joseph Seffi, and Roy Schwartz. A tight linear time (1/2)- approximation for unconstrained submodular maximization. SIAM Journal on Computing, 44 (5):1384–1402, 2015.

Gruia Calinescu, Chandra Chekuri, Martin Pal, and Jan Vondrak. Maximizing a monotone submod-´ ular function subject to a matroid constraint. SIAM Journal on Computing, 40(6):1740–1766, 2011.

Hiba Dakdouk. Massive multi-player multi-armed bandits for internet of things networks. PhD thesis, Ecole nationale superieure Mines-T´ el´ ecom Atlantique, 2022.´

Uriel Feige, Vahab S Mirrokni, and Jan Vondrak. Maximizing non-monotone submodular functions.´ SIAM Journal on Computing, 40(4):1133–1153, 2011.

M. L. Fisher, G. L. Nemhauser, and L. A. Wolsey. An analysis of approximations for maximizing submodular set functions—II, pages 73–87. Springer Berlin Heidelberg, 1978.

M.R. Garey and D.S. Johnson. Computers and Intractability: A Guide to the Theory of NPcompleteness. Mathematical Sciences Series. W. H. Freeman, 1979.

Shayan Oveis Gharan and Jan Vondrak. Submodular maximization by simulated annealing. In ´ Proceedings of the twenty-second annual ACM-SIAM symposium on Discrete Algorithms, pages 1098–1116. SIAM, 2011.

Pranava R Goundan and Andreas S Schulz. Revisiting the greedy approach to submodular set function maximization. Optimization online, 1, 2007.

Elias Koutsoupias and Christos Papadimitriou. Worst-case equilibria. In Annual symposium on theoretical aspects of computer science, pages 404–413. Springer, 1999.

Keqin Liu and Qing Zhao. Distributed learning in multi-armed bandit with multiple players. IEEE transactions on signal processing, 58(11):5667–5681, 2010.

Dov Monderer and Lloyd S Shapley. Potential games. Games and economic behavior, 14(1): 124–143, 1996.

Hugo Richard, Etienne Boursier, and Vianney Perchet. Constant or logarithmic regret in asynchronous multiplayer bandits with limited communication. In International Conference on Artificial Intelligence and Statistics, pages 388–396. PMLR, 2024.

Jonathan Rosenski, Ohad Shamir, and Liran Szlak. Multi-player bandits–a musical chairs approach. In International Conference on Machine Learning, pages 155–163. PMLR, 2016.

Po-An Wang, Alexandre Proutiere, Kaito Ariu, Yassir Jedra, and Alessio Russo. Optimal algorithms for multiplayer multi-armed bandits. In International Conference on Artificial Intelligence and Statistics, pages 4120–4129. PMLR, 2020.

Qinshi Wang and Wei Chen. Improving regret bounds for combinatorial semi-bandits with probabilistically triggered arms and its applications. Advances in neural information processing systems, 30, 2017.

## Appendix

## Table of Contents

A Restricted monotonicity and submodularity 15   
B Considerations in case of full saturation by greedy 16   
C Per-arm payout inequality for fixed mass 17   
D Existence of an allocation matching optimal masses 18   
E Calculations giving the lower bound on $\frac { \mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) } { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) }$ 20   
F Proof of tightness of the $\frac { 1 } { e }$ ratio 21   
G Bounds on DORG 21   
H Magnitudes of constants in CUCB’s distribution dependent bounds 22   
I Proof of the existence of Nash equilibria 23   
J Unbounded PoA and PoS 24   
K Proof of PoA bound in Theorem 4 24

## Appendix A. Restricted monotonicity and submodularity

We now view $\mu _ { \theta }$ as a set function on $[ N ] \times [ K ]$ , with elements $( i , k )$ being the allocation of player i to arm k. If S and $T$ are two allocations, we say $S \subseteq T$ if $S _ { k } \subseteq T _ { k }$ for all k.

Lemma 3. Let U (resp. V) be the allocations whose arms all have load at most 1 (resp. 2). Then $\mu _ { \theta }$ is monotone on U and submodular on V. Both ofthese properties break down iftheir respective conditions are lifted.

Proof. Both families are closed under subsets. For monotonicity, assume $S \subseteq T$ . We add the players of $T \setminus S$ to S sequentially. Each intermediate allocation lies in $u ,$ so all arms have load at most 1. As such, the marginal gain of each addition is positive, which concludes.

For $S _ { k } \subseteq [ N ]$ , adding player j to $S _ { k }$ changes the marginal gain of adding player i by

$$
\begin{array} { l } { { \theta _ { k } p _ { i } \displaystyle \prod _ { l \in S _ { k } } \bigl ( 1 - p _ { l } \bigr ) \biggl [ \bigl ( 1 - p _ { j } \bigr ) \biggl ( 1 - \sum _ { l \in S _ { k } } \frac { p _ { l } } { 1 - p _ { l } } - \frac { p _ { j } } { 1 - p _ { j } } \biggr ) - \bigl ( 1 - \sum _ { l \in S _ { k } } \frac { p _ { l } } { 1 - p _ { l } } \bigr ) \biggr ] } } \\ { { = - \theta _ { k } p _ { i } p _ { j } \displaystyle \prod _ { l \in S _ { k } } \bigl ( 1 - p _ { l } \bigr ) \bigl ( 2 - \sum _ { l \in S _ { k } } \frac { p _ { l } } { 1 - p _ { l } } \bigr ) , } } \end{array}
$$

which is negative when $\begin{array} { r } { \sum _ { l \in S _ { k } } \frac { p _ { l } } { 1 - p _ { l } } \leq 2 } \end{array}$ . As such, while loads are all below 2, we are in a submodular regime.

Reciprocals follow the exact same reasoning: for monotonicity, an intermediate load greater than 1 means a decrease in payout. For submodularity, the addition of a player past load 2 increases the marginal of adding another player, losing submodularity.

## Appendix B. Considerations in case of full saturation by greedy

Let $p _ { 1 } , \ldots , p _ { N }$ and $\theta _ { 1 } , \ldots , \theta _ { K }$ be an instance of the allocation problem (1) for which greedy saturates all arms, i.e., $\begin{array} { r } { \sum _ { i \in S _ { \mathrm { G r e e d y } , k } } \frac { p _ { i } } { 1 - p _ { i } } \geq 1 } \end{array}$ for all $k \in [ K ]$

Lemma 4. In instances where greedy saturates all arms,

$$
\mu _ { \boldsymbol { \theta } } ( S _ { G r e e d y } ) \geq \frac { 1 } { e } \sum _ { k \leq K } \theta _ { k }
$$

Proof. The greedy Algorithm 1 assigns a player to an arm only if the marginal of said addition is positive. As such, for $k \in [ K ]$ , the last player added to arm $k ,$ say player $j ,$ verifies that $\begin{array} { r } { \theta _ { k } p _ { j } \prod _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } ( 1 - p _ { i } ) \left( 1 - \sum _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } \frac { p _ { i } } { 1 - p _ { i } } \right) > 0 . } \end{array}$

In particular, this means that $\begin{array} { r } { \sum _ { i \in S _ { \mathrm { G r e e d y } , k } , i \ne j } \frac { p _ { i } } { 1 - p _ { i } } < 1 . \mathrm { L e t } \bar { p } _ { j } } \end{array}$ be such that $\begin{array} { r } { \sum _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } \frac { p _ { i } } { 1 - p _ { i } } + } \end{array}$ $\begin{array} { r } { \frac { \bar { p } _ { j } } { 1 - \bar { p } _ { j } } = 1 } \end{array}$ . We create a fictitious $( N + 1 )$ -th player of activation probability $\bar { p } _ { j }$ . Since $\bar { p } _ { j } \le p _ { j }$ , and given the linearity of the marginal, this in particular means that $f ( S _ { \mathrm { G r e e d y } , k } ) \geq f \big ( ( S _ { \mathrm { G r e e d y } , k } \setminus \{ j \} ) \cup$ $\{ N + 1 \} )$

Finally,

$$
\begin{array} { r l } & { f \big ( ( S _ { \mathrm { G r e e d y } , k } \setminus \{ j \} ) \cup \{ N + 1 \} \big ) = ( 1 - \hat { p } _ { j } ) \displaystyle \prod _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } ( 1 - p _ { i } ) \left( \displaystyle \sum _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } \frac { p _ { i } } { 1 - p _ { i } } + \frac { \hat { p } _ { j } } { 1 - \hat { p } _ { j } } \right) } \\ & { \qquad \ge ( 1 - \hat { p } _ { j } ) \displaystyle \prod _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } ( 1 - p _ { i } ) = \displaystyle \frac { 1 } { 1 + \frac { \hat { p } _ { j } } { 1 - \hat { p } _ { j } } } \displaystyle \prod _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } \frac { 1 } { 1 + \frac { p _ { i } } { 1 - p _ { i } } } } \\ & { \qquad \ge e ^ { ( - \sum _ { i \in S _ { \mathrm { G r e e d y } , k } , i \neq j } \frac { p _ { i } } { 1 - p _ { i } } ) - \frac { \hat { p } _ { j } } { 1 - \hat { p } _ { j } } } = \displaystyle \frac { 1 } { e } } \end{array}
$$

The result finally comes from summing expected payouts for each arm.

We now seek an upper bound on $\mu _ { \theta } ( S _ { \mathrm { o p t } } )$ to conclude.

Lemma 5. With $q < 1$ the maximal activation probability of players, $\begin{array} { r } { \mu _ { \theta } ( S _ { o p t } ) \le \lceil \frac { 1 - q } { q } \rceil \frac { q } { 1 - q } \cdot ( 1 - } \end{array}$ $\begin{array} { r } { q ) ^ { \lceil \frac { 1 - q } { q } \rceil } \cdot \sum _ { k \in [ K ] } \theta _ { k } } \end{array}$

Proof. Let S be a set of player indices, and $i \in S$ . Conditioning on the activity of player $i ,$

$$
f ( S ) = ( 1 - p _ { i } ) f ( S \setminus \{ i \} ) + p _ { i } \prod _ { \substack { j \in S , j \neq i } } ( 1 - p _ { j } ) ,
$$

so $f ( S )$ is an affine function of $p _ { i }$ when the other probabilities are fixed. Its maximum over $p _ { i } \in$ $[ 0 , q ]$ is therefore attained at $p _ { i } = 0 \mathrm { o r } p _ { i } = q$ . Replacing the activation probabilities one by one in this way never decreases $f ( S )$ , and leads to a set in which every player has activation probability 0 or $q .$ . In particular, an allocation where each arm gets the maximum amount until adding more players is detrimental, i.e., given (2), exactly $\textstyle \left\lceil { \frac { 1 - q } { q } } \right\rceil$ players, has higher expected payout than all allocations on $p _ { 1 } , \ldots , p _ { N }$ , including optimal ones.

The overall formula follows by adding per-arm payouts.

Taking the quotient of the two bounds we just obtained finally gives that

$$
\frac { \mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) } { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) } \geq \frac { 1 } { \lceil \frac { 1 - q } { q } \rceil e \frac { q } { 1 - q } \cdot ( 1 - q ) ^ { \lceil \frac { 1 - q } { q } \rceil } }
$$

This ratio is always at least $e ^ { - q }$ . Indeed, writing $\textstyle n = \lceil { \frac { 1 - q } { q } } \rceil \geq 1$ and using $1 - q \leq e ^ { - q }$

$$
\begin{array} { r l r } & { } & { n \displaystyle \frac { q } { 1 - q } ( 1 - q ) ^ { n } = n q ( 1 - q ) ^ { n - 1 } \leq n q e ^ { - q ( n - 1 ) } } \\ & { } & { = e ^ { q } \cdot n q e ^ { - n q } } \\ & { } & { \leq e ^ { q - 1 } } \end{array}
$$

## Appendix C. Per-arm payout inequality for fixed mass

We wish to formalize that for a fixed arm mass $m = \textstyle \sum _ { i } p _ { i }$ , the expected payout is lowest if all players have infinitesimal activation probability when $m \leq 1$ , and is largest when the mass is concentrated on few players of maximal activation probability, yielding closed-form lower and upper bounds for an arm’s expected payout.

Throughout this section, for a set $S$ of players, we write

$$
f ( S ) = \sum _ { i \in S } p _ { i } \prod _ { \substack { j \in S , j \neq i } } ( 1 - p _ { j } ) ,
$$

which coincides with the definition of Section 2 and remains well defined if some player has activation probability 1.

Lemma 6. Let $q < 1$ be a fixed real. For every integer $n \geq 1$ and every n-tuple $( p _ { 1 } , \ldots , p _ { n } )$ of reals in $[ 0 , q ]$ ofmass $\begin{array} { r } { m = \sum _ { i = 1 } ^ { n } p _ { i } , } \end{array}$ we have, with $S = [ n ]$

$$
f ( S ) \leq m e ^ { q - m }
$$

and, if furthermore $m \leq 1$

$$
f ( S ) \geq m e ^ { - m }
$$

Proof. For each i, since $1 - p _ { j } \le e ^ { - p _ { j } }$ and $p _ { i } \leq q ,$

$$
\prod _ { j \in S , j \neq i } ( 1 - p _ { j } ) \leq e ^ { - ( m - p _ { i } ) } \leq e ^ { q - m }
$$

Summing over i with weights $p _ { i }$ gives the upper bound $f ( S ) \leq m e ^ { q - m }$

For the lower bound, assume $0 < m \leq 1$ . The $\frac { p _ { i } } { m }$ sum to 1, so the weighted arithmetic-geometric inequality gives

$$
\begin{array} { r } { \frac { f ( S ) } { m } = \displaystyle \sum _ { i = 1 } ^ { n } \frac { p _ { i } } { m } \prod _ { j \neq i } ( 1 - p _ { j } ) \geq \prod _ { i = 1 } ^ { n } \Bigl [ \prod _ { j \neq i } ( 1 - p _ { j } ) \Bigr ] ^ { \frac { p _ { i } } { m } } } \\ { = \displaystyle \prod _ { j = 1 } ^ { n } ( 1 - p _ { j } ) ^ { 1 - \frac { p _ { j } } { m } } . } \end{array}
$$

Taking logarithms, it remains to show that

$$
\sum _ { j = 1 } ^ { n } \Bigl ( 1 - \frac { p _ { j } } { m } \Bigr ) \ln ( 1 - p _ { j } ) \geq - m = - \sum _ { j = 1 } ^ { n } p _ { j }
$$

Since $m \leq 1$ , we have $\begin{array} { r } { 1 - \frac { p _ { j } } { m } \le 1 - p _ { j } } \end{array}$ , and since ln $( 1 - p _ { j } ) \leq 0 \quad$

$$
\begin{array} { c } { { \left( 1 - \displaystyle \frac { p _ { j } } { m } \right) \ln ( 1 - p _ { j } ) \ge ( 1 - p _ { j } ) \ln ( 1 - p _ { j } ) } } \\ { { \ge - p _ { j } } } \end{array}
$$

We conclude by summing all terms.

## Appendix D. Existence of an allocation matching optimal masses

Lemma 7. Let $( c _ { 1 } , \ldots , c _ { K } ) \in \mathbb { R } _ { + } ^ { K }$ , arm means $\theta _ { 1 } , \ldots , \theta _ { K }$ and players $p _ { 1 } , \ldots , p _ { N }$ such that

$$
\sum _ { k \le K } c _ { k } \le \sum _ { i \le N } p _ { i }
$$

and let $S _ { G r e e d y }$ be the allocation given by the greedy Algorithm 1 on some order ofplayers. Assume that greedy assigns every player to an arm. There exists an allocation ${ \bar { S } } ,$ , on a set of players potentially different from the initial one, where the activation mass on arm k is exactly $c _ { k } ,$ , and that satisfies $\bar { \mu } _ { \boldsymbol { \theta } } ( \bar { S } ) \leq \mu _ { \boldsymbol { \theta } } ( S _ { G r e e d y } )$ , with $\bar { \mu } _ { \theta }$ the payout function for the set of players used in S<sup>¯</sup>.

We approach the proof by showing we can shrink players and potentially reassign the removed mass to other arms, while reducing the expected payout.

For a set S of players, we denote $\begin{array} { r } { z ( S ) = \prod _ { i \in S } ( 1 - p _ { i } ) } \end{array}$ and

$$
\Delta ( S ) = z ( S ) - f ( S )
$$

Notice that $p \theta _ { k } \Delta ( S _ { k } )$ is exactly the marginal of adding a player of mass $p$ to arm $k ,$ as specified in (2).

Lemma 8. Let S be a set of players and $p < 1$ the activation probability of an additional player ofindex i. Then

$$
\begin{array} { r l } & { f ( S \cup \{ i \} ) = f ( S ) + p \Delta ( S ) } \\ & { \Delta ( S \cup \{ i \} ) = ( 1 - p ) \Delta ( S ) - p z ( S ) \leq \operatorname* { m a x } \{ \Delta ( S ) , 0 \} . } \end{array}
$$

In particular, $i f S \subseteq T$ , then $\Delta ( T ) \le \operatorname* { m a x } \{ \Delta ( S ) , 0 \}$

Proof. By conditioning on the activity of the additional player, $f ( S \cup \{ i \} ) = ( 1 - p ) f ( S ) + p z ( S )$ Since $z ( S \cup \{ i \} ) = ( 1 - p ) z ( S )$ , we then have

$$
\begin{array} { c } { \Delta ( S \cup \{ i \} ) = ( 1 - p ) z ( S ) - ( 1 - p ) f ( S ) - p z ( S ) } \\ { = ( 1 - p ) \Delta ( S ) - p z ( S ) } \end{array}
$$

Since $p z ( S )$ is always strictly positive, we conclude.

The last point is obtained by adding the players of $T \setminus S$ one by one.

Lemma 9. Let $S _ { G r e e d y }$ be an allocation obtained by the greedy algorithm on an arbitrary order of the players. We assume that we know the order in which players on a given arm have been allocated. Let A, B be a partition of[K]. Let S be an allocation such that

• for $k \in A , S _ { k }$ is made of the first players assigned to arm k by greedy, the last of which may have a lower activation probability than initially;

• for $k ^ { \prime } \in B , S _ { k ^ { \prime } }$ contains $S _ { G r e e d y , k ^ { \prime } } .$

Let $k \in A ,$ , let p be the activation probability in S of the last player of $S _ { k } ,$ , and let $0 < b \leq p .$ Then thefollowing two actions have the effect ofreducing $\mu _ { \boldsymbol { \theta } } ( S )$

• Lowering the activation probability of the last player of $S _ { k }$ from p to $p - b$

• Lowering the activation probability of the last player of $S _ { k }$ from p to $\begin{array} { r } { p - b , } \end{array}$ and adding a new player ofactivation probability b to an arm $k ^ { \prime } \in B$

Furthermore, the resulting allocation still satisfies the two conditions above. In particular, it is possible to iterate these operations in any order.

Proof. Let i be the last player of $S _ { k }$ , and R the players of $S _ { k }$ other than i. By assumption, R is exactly the set of players assigned to arm k at the moment greedy processed player i. Since greedy assigned i to arm k, this means that

$$
\theta _ { k } \Delta ( R ) = \operatorname* { m a x } _ { \ell \in [ K ] } \theta _ { \ell } \Delta ( S _ { \ell } ^ { ( i ) } ) > 0 ,
$$

where $S _ { \ell } ^ { ( i ) }$ denotes the set of players assigned to arm ℓ at that moment.

By Lemma 8, $f ( R \cup \{ i \} ) = f ( R ) + p \Delta ( R )$ , so lowering the activation probability of player i by b adds exactly $- b \theta _ { k } \Delta ( R ) \leq 0 \mathrm { t o } \mu _ { \theta } ( S )$

We now wish to show that reassigning the removed mass to an arm in B keeps the effect on $\mu _ { \theta }$ negative. Adding a player of activation probability b to arm $k ^ { \prime }$ adds $b \theta _ { k ^ { \prime } } \Delta ( S _ { k ^ { \prime } } )$ to $\mu _ { \boldsymbol { \theta } } ( S )$ , by

Lemma 8. Arm $k ^ { \prime }$ may have received other players since greedy processed player i, but lost none, meaning $S _ { k ^ { \prime } } ^ { ( i ) } \subseteq S _ { \mathrm { G r e e d y } , k ^ { \prime } } \subseteq S _ { k ^ { \prime } }$ and, by Lemma $8 ,$

$$
\theta _ { k ^ { \prime } } \Delta ( S _ { k ^ { \prime } } ) \leq \operatorname* { m a x } \{ \theta _ { k ^ { \prime } } \Delta ( S _ { k ^ { \prime } } ^ { ( i ) } ) , 0 \} \leq \theta _ { k } \Delta ( R )
$$

The second action therefore also reduces expected payout.

Finally, arm k still contains the first players assigned to it by greedy, with the last one shrunk or removed, and arm $k ^ { \prime }$ only gained a player, so both conditions are preserved. ■

Proof of Lemma 7. We are now in a position to prove the main result initially mentioned. Let us start with greedy’s allocation, which populates each arm with mass $m _ { k }$ , and iterate operations from the above lemma. Let $A$ be the set of arms where $m _ { k } \ > \ c _ { k }$ , and $B$ its complement. The conditions of Lemma 9 are satisfied by $S _ { \mathrm { G r e e d y } }$ itself.

Since greedy assigns every player, $\begin{array} { r } { \sum _ { k } m _ { k } = \sum _ { i } p _ { i } \ge \sum _ { k } c _ { k } } \end{array}$

While there exists an arm k in A such that $m _ { k } > c _ { k }$ , we consider its last player, of activation probability $p \mathrm { : }$

• if there exists an arm $k ^ { \prime }$ in $B$ such that $c _ { k ^ { \prime } } > m _ { k ^ { \prime } }$ , we move a mass $b = \operatorname* { m i n } \{ p , \ m _ { k } \ -$ $c _ { k } , \ : c _ { k ^ { \prime } } - m _ { k ^ { \prime } } \}$ from the last player of arm k to a new player on arm $k ^ { \prime } ;$

• otherwise, we remove a mass $b = \operatorname* { m i n } \{ p , \ m _ { k } - c _ { k } \}$ from the last player of arm k.

The masses of the arms in A stay above their associated target, and those of the arms in B stay below.

Each of these “rounds” either removes a player entirely from an arm in A, or makes an arm exactly match its target mass. As such, there are at most $N + 2 K$ rounds. Moving mass lowers the excess and the missing mass by the same amount, and mass is only removed when no mass is missing, so the excess mass remains at least the missing mass throughout. When the process stops, there is no excess mass left, hence no missing mass either. We therefore reach the wanted state, all while lowering $\mu _ { \theta }$ in the process.

## Appendix E. Calculations giving the lower bound on $\frac { \mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) } { \mu _ { \theta } ( S _ { \bf o p t } ) }$

Throughout this section, we assume that the greedy Algorithm 1 assigns every player to an arm, and denote $q = \operatorname* { m a x } _ { i } p _ { i }$

Let $\begin{array} { r } { m _ { k } ^ { \mathrm { o p t } } = \sum _ { i \in S _ { \mathrm { o p t } , k } } p _ { i } } \end{array}$ be the per-arm masses of an optimal allocation, and $( c _ { 1 } , \ldots , c _ { K } ) =$ $( \operatorname* { m i n } \{ m _ { 1 } ^ { \mathrm { o p t } } , 1 \} , \dots , \operatorname* { m i n } \{ m _ { K } ^ { \mathrm { o p t } } , 1 \} )$ . Since

$$
\sum _ { k \le K } c _ { k } \le \sum _ { k \le K } m _ { k } ^ { \mathrm { o p t } } \le \sum _ { i \le N } p _ { i } ,
$$

we are in the scope of Lemma 7, and therefore have access to an allocation $\bar { S }$ which shares its masses with an optimal allocation on arms of mass at most 1, has mass exactly 1 on the others, and has worse payout than $S _ { \mathrm { G r e e d y } }$

The lower bound of Lemma 6 gives

$$
f ( \bar { S } _ { k } ) \geq c _ { k } e ^ { - c _ { k } }
$$

while its upper bound, applied to the optimal allocation, gives

$$
f ( S _ { \mathrm { o p t } , k } ) \leq m _ { k } ^ { \mathrm { o p t } } e ^ { q - m _ { k } ^ { \mathrm { o p t } } }
$$

It remains to compare the two. If $m _ { k } ^ { \mathrm { o p t } } \leq 1$ , then $c _ { k } \ = \ m _ { k } ^ { \mathrm { o p t } }$ and $m _ { k } ^ { \mathrm { o p t } } e ^ { - m _ { k } ^ { \mathrm { o p t } } } = c _ { k } e ^ { - c _ { k } }$ Otherwise, $c _ { k } = 1$ , and since $t \mapsto t e ^ { - t }$ is upper-bounded by $e ^ { - 1 }$ on $\mathbb { R } _ { + }$ , we have $m _ { k } ^ { \mathrm { o p t } } e ^ { - m _ { k } ^ { \mathrm { o p t } } } \leq$ $e ^ { - 1 } = c _ { k } e ^ { - c _ { k } }$ . In both cases,

$$
f ( S _ { \mathrm { o p t } , k } ) \leq e ^ { q } c _ { k } e ^ { - c _ { k } } \leq e ^ { q } f ( \bar { S } _ { k } ) .
$$

Summing across all arms, we finally have

$$
\begin{array} { r l } { \displaystyle \mu _ { \boldsymbol { \theta } } \big ( S _ { \mathrm { G r e e d y } } \big ) \geq \bar { \mu } _ { \boldsymbol { \theta } } \big ( \bar { S } \big ) = \sum _ { k \in [ K ] } \theta _ { k } f \big ( \bar { S } _ { k } \big ) } & { } \\ { \geq e ^ { - q } \displaystyle \sum _ { k \in [ K ] } \theta _ { k } f \big ( S _ { \mathrm { o p t } , k } \big ) } & { } \\ { = e ^ { - q } \mu _ { \boldsymbol { \theta } } \big ( S _ { \mathrm { o p t } } \big ) , } \end{array}
$$

## Appendix F. Proof of tightness of the $\frac { 1 } { e }$ ratio

Given the instance described in the proof sketch, i.e., $\theta _ { 1 } = 1 , \theta _ { 2 } = \frac { 1 } { 3 N } , p _ { 1 } = \cdot \cdot \cdot = p _ { N - 1 } = \frac { 1 } { N }$ and $\begin{array} { r } { p _ { N } = 1 - \frac { 1 } { N } } \end{array}$ for some $N \in \mathbb { N } _ { \geq 3 }$ , let us establish what optimal and greedy allocations are:

• Optimal allocations assign player $N$ to arm 1 and the rest to arm 2, yielding a total expected payout of $\begin{array} { r } { 1 - \frac { 1 } { N } + \frac { 1 \cdot ( 1 - \frac { 1 } { N } ) ^ { \mathbf { \hat { N } } - \mathbf { \tilde { 1 } } } } { 3 N } } \end{array}$

• Greedy processes players in order. The first player is assigned to the better of the two arms, arm 1. When $k < N - 1$ players of activation probability $\textstyle { \frac { 1 } { N } }$ are assigned to arm 1, the marginal of adding a player of activation probability p is $\begin{array} { r } { p ( 1 - \frac { 1 } { N } ) ^ { k } \frac { N - 1 - k } { N - 1 } \geq \frac { p } { 3 N } } \end{array}$ , with the latter being the marginal of adding that same player to the empty arm 2. As such, all of the $N - 1$ first players get assigned to arm 1. Finally, when $k = N - 1$ , the marginal of adding a player to arm 1 becomes 0, and player N of activation probability $\textstyle 1 - { \frac { 1 } { N } }$ gets assigned to arm 2. The final expected payout is $\begin{array} { r } { ( 1 - \frac { 1 } { N } ) ^ { N - 1 } + \frac { 1 - \frac { 1 } { N } } { 3 N } } \end{array}$

Taking the ratio of the two payouts gives

$$
{ \frac { \mu _ { \theta } ( S _ { \mathrm { G r e e d y } } ) } { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) } } = { \frac { ( 1 - { \frac { 1 } { N } } ) ^ { N - 1 } + { \frac { 1 - { \frac { 1 } { N } } } { 3 N } } } { 1 - { \frac { 1 } { N } } + { \frac { ( 1 - { \frac { 1 } { N } } ) ^ { N - 1 } } { 3 N } } } } { \xrightarrow { 1 } } { \frac { 1 } { e } }
$$

This concludes.

## Appendix G. Bounds on DORG

We first establish that DORG coincides with at least one optimal allocation on all players with activation probability above $\frac { 1 } { 2 }$

Lemma 10. Let $\kappa = \operatorname* { m a x } \{ i \mid p _ { i } > \frac { 1 } { 2 } \}$ , with $\kappa = 0$ if the set is empty. Then, there is an optimal allocation that coincides with DORG for the first κ players.

Proof. We first note that for any $q > \frac { 1 } { 2 }$ , and a set A of players with activation probabilities $p _ { j } \leq q .$ we have $f ( A ) \leq q$ . This is a direct consequence of Lemma 5.

Assuming player 1, with $p _ { 1 } > { \frac { 1 } { 2 } }$ , is assigned to some arm $k ,$ , this arm will not receive any supplementary players, as the marginal gain of doing so would be negative.

Let $S _ { \mathrm { o p t } }$ be an optimal allocation, and k be the arm player 1 is assigned to. Then $S _ { \mathrm { o p t } , k } = \{ 1 \}$ and therefore $f ( S _ { \mathrm { o p t } , k } ) = p _ { 1 }$ . Since ma $\mathrm { x } _ { i \in S _ { 1 } } p _ { i } \le p _ { 1 } , f ( S _ { 1 } ) \le f ( S _ { k } )$ . Swapping $S _ { k }$ and $S _ { 1 }$ does not decrease the total payout, and therefore an optimal allocation assigning player 1 to arm 1 exists. We follow by induction for all other used players with $\begin{array} { r } { p _ { i } > \frac { 1 } { 2 } } \end{array}$

Finally, let us prove that player 1 is indeed assigned to an arm w.l.o.g. Arm 1 contains players in $S _ { 1 }$ , which are all of activation probability $p _ { i } \leq p _ { 1 }$ . As such, replacing $S _ { 1 }$ with {1} (i.e., assigning player 1 instead of the current allocation) while keeping the rest of the allocation unchanged does not decrease the payout. We follow for the next arms by induction, while arms remain. If $\kappa > K$ the remaining players are left unassigned by both DORG and optimal allocations.

We therefore know DORG and some optimal allocation $S _ { \mathrm { o p t } }$ agree on players with activation probability above $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ . Since said players are left as sole occupants of their respective arms, S and $S _ { \mathrm { o p t } }$ can disagree only on players with activation probability less than ${ \frac { 1 } { 2 } } ,$ , and on a subset of arms. As such,

$$
\displaystyle \mu _ { \boldsymbol { \theta } } ( S _ { \mathrm { o p t } } ) = \sum _ { k \le K } \theta _ { k } f ( S _ { \mathrm { o p t } , k } )  \\  = \sum _ { k \le \operatorname* { m i n } \{ \kappa , K \} } \theta _ { k } p _ { k } + \sum _ { \operatorname* { m i n } \{ \kappa , K \} < k \le K } \theta _ { k } f ( S _ { \mathrm { o p t } , k } )
$$

while

$$
\displaystyle \mu _ { \boldsymbol \theta } ( S _ { \mathrm { D O R G } } ) = \sum _ { k \le K } \boldsymbol \theta _ { k } f ( S _ { \mathrm { D O R G } , k } )  \\  = \sum _ { k \le \operatorname* { m i n } \{ \kappa , K \} } \boldsymbol \theta _ { k } \boldsymbol p _ { k } + \sum _ { \operatorname* { m i n } \{ \kappa , K \} < k \le K } \boldsymbol \theta _ { k } f ( S _ { \mathrm { D O R G } , k } )
$$

The second term of the sum corresponds to DORG dealing with all players of activation probability below $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ , and as such we have bounds on the ratio of the second terms of both expressions<sup>5</sup>, by Theorem 2 and Lemma $5 ,$ as either $\textstyle { \frac { 1 } { \sqrt { e } } }$ in unsaturated instances, $\mathrm { o r ~ } \frac { 2 } { e }$ in saturated ones.

As such, we have an instance-independent $\scriptstyle { \frac { 1 } { \sqrt { e } } }$ lower bound on $\frac { \mu _ { \theta } ( S _ { \mathrm { D O R G } } ) } { \mu _ { \theta } ( S _ { \mathrm { o p t } } ) }$ , i.e., DORG is a $\scriptstyle { \frac { 1 } { \sqrt { e } } } -$ oracle for the allocation problem (1).

## Appendix H. Magnitudes of constants in CUCB’s distribution dependent bounds

Wang and Chen (2017) also provide logarithmic bounds that depend on instance dependent constant quantities $\Delta _ { \mathrm { m i n } }$ and $\Delta _ { \mathrm { m a x } }$ :

$$
\mathrm { R e g } _ { \theta , \alpha , 1 } ( T ) \le \sum _ { k = 1 } ^ { K } \frac { 5 7 6 \tilde { K } \ln { T } } { \Delta _ { \operatorname* { m i n } } ^ { k } } + \sum _ { k = 1 } ^ { K } \left( \left\lceil \log _ { 2 } \frac { 2 \tilde { K } } { \Delta _ { \operatorname* { m i n } } ^ { k } } \right\rceil _ { 0 } + 2 \right) \frac { \pi ^ { 2 } } { 6 } \Delta _ { \operatorname* { m a x } } + 4 K
$$

with $\tilde { K } = \operatorname* { m i n } \{ K , N \}$ the number of pullable arms, for each arm k

$$
\Delta _ { \operatorname* { m i n } } ^ { k } = \operatorname* { i n f } _ { \substack { S | f ( S _ { k } ) > 0 , \Delta _ { S } > 0 } } \Delta _ { S } , \qquad \Delta _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { \substack { k \leq K } } \operatorname* { s u p } _ { \substack { S | f ( S _ { k } ) > 0 , \Delta _ { S } > 0 } } \Delta _ { S }
$$

and

$$
\Delta _ { S } = \mathrm { m a x } \left( 0 , \alpha \mu _ { \theta } ( S _ { \mathrm { o p t } } ) - \mu _ { \theta } ( S ) \right)
$$

By definition, $\Delta _ { \mathrm { m a x } } = \alpha \mu _ { \boldsymbol { \theta } } ( S _ { \mathrm { o p t } } ) - \operatorname* { m i n } \{ \mu _ { \boldsymbol { \theta } } ( S ) \mid S \neq \emptyset \}$ . As such, we have the immediate bound

$$
\Delta _ { \mathrm { m a x } } \leq \alpha K
$$

For $\Delta _ { \operatorname* { m i n } } ^ { k }$ , we seek to find

$$
r _ { k } = \operatorname* { m a x } \left\{ v \in V _ { k } \mid v < \alpha \mu _ { \theta } ( S _ { \mathrm { o p t } } ) \right\}
$$

where $V _ { k } = \{ \mu _ { \theta } ( S ) \mid S _ { k } \neq \emptyset \} \subset [ 0 , \mu _ { \theta } ( S _ { \mathrm { o p t } } ) ]$ is the set of payouts for allocations that assign at least one player to arm k, and

$$
\Delta _ { \mathrm { m i n } } ^ { k } = \alpha \mu _ { \theta } ( S _ { \mathrm { o p t } } ) - r _ { k }
$$

This quantity is instance-dependent in magnitude, and can be arbitrarily small even in nondegenerate scenarios: for example, for $\alpha = \textstyle { \frac { 1 } { \sqrt { e } } } , { \mathrm { ~ i f ~ } } \theta _ { 1 } = 1 , \theta _ { 2 } = 1$ , and players have activation probabilities ${ \begin{array} { l } { { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } , { \frac { 1 } { \sqrt { e } } } \mathrm { ~ - ~ } { \frac { 1 } { 2 } } \mathrm { ~ - ~ } \varepsilon . } \end{array} }$ , then the optimal expected payout is 1 by splitting the two good players across arms, and having the third player replace one of them on either the first or second arm gives $\Delta _ { \mathrm { m i n } } ^ { 1 } = \Delta _ { \mathrm { m i n } } ^ { 2 } = \varepsilon$

## Appendix I. Proof of the existence of Nash equilibria

We show that a variant of the allocation game is a potential game.

Theorem 5. Let $\begin{array} { r } { w _ { i } = \ln \frac { 1 } { 1 - p _ { i } } } \end{array}$ and $\begin{array} { r } { W _ { k } ( s ) = \sum _ { i \in S _ { k } } w _ { i } , } \end{array}$ , so that $z _ { k } = e ^ { - W _ { k } ( s ) }$ . Then

$$
\Phi ( s ) = \sum _ { k = 1 } ^ { K } \left[ \sum _ { \{ i , j \} \subseteq S _ { k } } w _ { i } w _ { j } \ - \ \ln \theta _ { k } W _ { k } ( s ) \ \right]
$$

satisfies, for every unilateral deviation $s  s ^ { \prime }$ of a player j,

$$
\Phi ( s ^ { \prime } ) - \Phi ( s ) = - w _ { j } \left( \ln u _ { j } ( s ^ { \prime } ) - \ln u _ { j } ( s ) \right) :
$$

the “log-transformed” game is a weighted potential game.

Proof. Since $z _ { k } = e ^ { - W _ { k } ( s ) }$ , we have ln $u _ { j } ( s ) = \ln p _ { j } + w _ { j } + \ln \theta _ { s _ { j } } - W _ { s _ { j } } ( s )$ . If j moves from arm a to arm b, the pair terms of Φ change by $w _ { j } W _ { b } - w _ { j } ( W _ { a } - w _ { j } )$ and the linear terms by $w _ { j } ( \ln \theta _ { a } - \ln \theta _ { b } )$ , so that

$$
\Phi ( s ^ { \prime } ) - \Phi ( s ) = w _ { j } \left[ ( W _ { b } - \ln \theta _ { b } ) - \left( W _ { a } - w _ { j } - \ln \theta _ { a } \right) \right] = - w _ { j } \left( \ln u _ { j } ( s ^ { \prime } ) - \ln u _ { j } ( s ) \right) .
$$

Consequently, pure equilibria exist, and every better-response path is finite (Monderer and Shapley, 1996).

## Appendix J. Unbounded PoA and PoS

Let $N \geq 2 .$ . We wish to prove that the instance with two arms of means $\theta _ { 1 } = 1 , \theta _ { 2 } = 4 ^ { - N }$ and $N$ players of activation probability $\frac { 1 } { 2 }$ satisfies $\textstyle \mathrm { P o A } = \mathrm { P o S } \geq { \frac { 2 ^ { N - 1 } } { N } }$

Assigning every player to arm 1 is an equilibrium, since $\begin{array} { r } { \frac { z _ { 1 } } { 1 - p _ { j } } = 2 ^ { - ( N - 1 ) } \ge \theta _ { 2 } } \end{array}$ . Its expected payout is $N { \cdot } 2 ^ { - N }$ . This is the only pure equilibrium, as any player assigned to arm 2 could profitably deviate, and all players want to play.

Meanwhile, $\textstyle \mu _ { \theta } ( S _ { \mathrm { o p t } } ) = { \frac { 1 } { 2 } } ( 1 + 4 ^ { - N } )$ , assigning a single player to each of the two arms. The ratio of the two concludes.

## Appendix K. Proof of PoA bound in Theorem 4

We aim to prove the lower bound on $\mu _ { \boldsymbol { \theta } } ( S ( \boldsymbol { s } ) )$ and the upper bound on $\mu _ { \boldsymbol { \theta } } ( S _ { \mathrm { o p t } } )$ independently, then take their ratio to reach the final result. We use throughout $\begin{array} { r } { l _ { k } = \sum _ { i \in S _ { k } } \frac { p _ { i } } { 1 - p _ { i } } . } \end{array}$

• Let $\lambda _ { k } = \theta _ { k } e ^ { - W _ { k } }$ , with $W _ { k }$ the “log-load”

$$
W _ { k } = - \sum _ { i \in S _ { k } } \ln ( 1 - p _ { i } )
$$

For $p \in ( 0 , q ]$ , we have

$$
( 1 - q ) { \frac { p } { 1 - p } } \leq p \leq \ln { \frac { 1 } { 1 - p } } \leq { \frac { p } { 1 - p } }
$$

and therefore

$$
( 1 - q ) l _ { k } \leq W _ { k } \leq l _ { k }
$$

If arm k is occupied by some player j, the equilibrium condition imposes $\begin{array} { r } { \frac { \lambda _ { k } } { 1 - p _ { j } } \geq \lambda _ { k ^ { \prime } } } \end{array}$ for all $k ^ { \prime } ,$ , which gives $\lambda _ { k } \geq ( 1 - q )$ max<sub>k</sub>′ λ<sub>k</sub>′.

Since all players opt to play, $\textstyle \sum _ { k } l _ { k } = L \geq \sum _ { k } W _ { k }$ , so some arm $k ^ { \prime }$ has $W _ { k ^ { \prime } } \leq x$ . As such, since $\lambda _ { k } = \theta _ { k } e ^ { - W _ { k } }$ , we have by bounding each term

$$
\operatorname* { m a x } _ { k ^ { \prime } } \lambda _ { k ^ { \prime } } \geq ( \bar { \theta } - \varepsilon _ { 1 } ) e ^ { - x }
$$

Finally, since $\begin{array} { r } { \mu _ { \theta } ( S ) = \sum _ { k } \lambda _ { k } l _ { k } } \end{array}$ , the equilibrium payout is at least

$$
( 1 - q ) ( { \bar { \theta } } - \varepsilon _ { 1 } ) e ^ { - x } L
$$

• For any allocation S, by the inequalities above

$$
f ( S _ { k } ) = l _ { k } e ^ { - W _ { k } } \leq l _ { k } e ^ { - ( 1 - q ) l _ { k } }
$$

and since $h \geq g$

$$
\begin{array} { r } { l _ { k } e ^ { - ( 1 - q ) l _ { k } } = \displaystyle \frac { 1 } { 1 - q } g \big ( ( 1 - q ) l _ { k } \big ) } \\ { \leq \displaystyle \frac { 1 } { 1 - q } h \big ( ( 1 - q ) l _ { k } \big ) } \end{array}
$$

By Jensen

$$
\begin{array} { l } { \displaystyle \sum _ { k } \theta _ { k } f ( S _ { k } ) \leq K \sum _ { k } \frac { 1 } { K ( 1 - q ) } ( \bar { \theta } + \varepsilon _ { 2 } ) h \big ( ( 1 - q ) l _ { k } \big ) } \\ { \leq \displaystyle \frac { ( \bar { \theta } + \varepsilon _ { 2 } ) K } { 1 - q } h \big ( ( 1 - q ) \frac { L } { K } \big ) } \end{array}
$$

The final result is the ratio of the two bounds, in cases where $h = g$ and not.