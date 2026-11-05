# Improved approximation algorithms for capacitated cluster editing

Cong Ma<sup>∗</sup> Guangting Chen<sup>†</sup> Yong Chen<sup>∗‡§</sup> Guohui Lin<sup>¶‡</sup> Jiaxuan Ma<sup>∗</sup>

Lusheng Wang<sup>‖</sup>

## Abstract

We study the minimum-disagreement cluster editing problem that, from a complete edge-signed graph, where a positive edge represents two vertices agreeing to each other while a negative represents a disagreement, constructs disjoint cliques of upper bounded size such that the total number of negative edges inside cliques and positive edges outside cliques is minimized. When the sizes of the candidate cliques are unbounded, the problem is NP-hard and APX-hard, and many approximation algorithms with improving ratios have been proposed in the literature, with the most recent ones appearing in JACM 2008, STOC 2015, FOCS 2022, FOCS 2023, STOC 2024 and arXiv 2026 and the state-of-the-art approximation ratio of α = 1.3865 + ϵ for any ϵ > 0. The capacitated variant, in which the candidate cliques have sizes no greater than a given k, is less studied (SIAM J. Optim 2015, IJFCS 2024) and the best approximation ratio is 5.37. In this paper, we study the latter problem from the perspective of approximation algorithms. We first present a downsizing procedure that achieves an approximation ratio of $\alpha _ { k } = 2 \alpha + 1$ , based on any α-approximation algorithm for the uncapacitated problem. Secondly, we propose a hybrid of cluster-LP sampling and pivoting, followed by the downsizing procedure, and show that for every fractional capacitated cluster-LP solution of cost L, the hybrid using ordinary pivoting has an expected cost at most $\begin{array} { r } { { \frac { 9 } { 5 } } L + { \frac { 1 } { 1 0 } } \mathrm { O P T } _ { k } . } \end{array}$ , and the hybrid using conditional pivoting has an expected cost at most $\begin{array} { r } { \frac { 3 0 6 } { 2 0 3 } L + \frac { 5 0 } { 2 0 3 } \mathrm { O P T } _ { k } . } \end{array}$ A cardinality-constrained adaptation of the approximate dual-separation framework of Garc´ıa-Soriano and Schohn supplies the required fractional solution and yields a randomized (1.9 + ϵ)-approximation algorithm and a randomized (1.7537 + ϵ)-approximation algorithm, respectively, for any $\epsilon > 0$ . All these three algorithms have running time polynomial in the size of the graph, the number of positive values in the fractional solution, and the encoding length, and the exponents are independent of k.

Keywords: Cluster editing; cluster linear program; randomized sampling; pivoting; approximation algorithm

## Acknowledgments

The research is supported by the NNSF of China (Grants 12471301, 61972329), the China Scholarship Council (Grant No. 202508330173), the GRF grants for Hong Kong Special Administrative Region, China (Grant Nos. CityU 11218423, CityU 11219825), and the NSERC Canada.

## 1 Introduction

Correlation clustering, introduced by Bansal et al. [2], constructs a partition from pairwise similarity and dissimilarity information. In the complete unweighted form, every pair of vertices is labelled positive or negative. The objective is to form a partition of the vertices, in which a part is called a cluster, to minimize the number of positive edges between clusters and negative edges within clusters. Equivalently, one seeks to transform the graph of positive edges only into a disjoint union of cliques using as few edge insertions and deletions as possible. Such a formulation is also known as Cluster Editing, which is an NP-hard [25] and APX-hard [10] problem.

We study Capacitated Cluster Editing, in which each candidate cluster contains no more than k vertices, where k is part of the input and is referred to as the cluster capacity. The problem remains NP-hard for every fixed $k \geq 3 ~ [ 2 0 ]$ . On the approximation algorithms, Puleo and Milenkovic [24] developed an algorithm based on a linear program and combinatorial methods, including two proven 7- and 6-approximation algorithms. Il’ev et al. [19] established the APX-completeness for each fixed capacity, with an approximation ratio depending on that capacity. Later, Ji et al. [21] improved the approximation ratio to 5.37, independent on the capacity, based on a linear program formulation and rounding to construct size-bounded clusters. Kononov and Il’ev [23] further obtained better ratios of $\frac { 3 } { 2 }$ for $k = 3$ and $\frac { 1 3 } { 7 }$ for $k = 4$ . In this paper, we aim to design approximation algorithms with improved ratios when k is part of the input.

The problem introduced by Bansal et al. [2] is uncapacitated, i.e., a candidate cluster can have arbitrary size. Bansal et al. [2] had presented a constant ratio approximation algorithm, with the ratio being at most $9 \cdot 4 4 ^ { 2 } + 5 = 1 7 \phantom { . 0 }$ , 429. Recent progress for the uncapacitated problem has been driven by the Cluster linear program, which represents a fractional partition by assigning weights to all candidate clusters [7, 17]. The same representation can employed for capacitated variant, and in fact it makes an upper bound on cluster size easy to encode, but subsequently rounding poses a further dificulty. Basically, from a fractional solution, sampling the clusters preserves the size bound, whereas pivoting can create oversized clusters; for the latter oversized clusters, the feasibility can be restored by downsizing which however may increase the disagreement cost. Obtaining strong guarantees therefore requires incorporating downsizing into the rounding analysis.

## 1.1 Our results

We first propose a simple downsizing procedure that converts any uncapacitated partition P into a feasible capacitated partition denoted as $\mathrm { S p l i t } ( { \mathcal { P } } )$ having a disagreement cost at most twice of P plus an optimum. This gives rise to a (3.7730 + ϵ)-approximation algorithm based on the state-of-the-art (1.3865 + ϵ)-approximation for the uncapacitated Cluster Editing problem.

Next, realizing that we can extend the uncapacitated Cluster-LP to capacitated Cluster-LP to achieve a fractional solution of cost arbitrarily close to the optimum, we propose to combine the cluster sampling approach of Cao et al. [7] to obtain a feasible solution $\mathcal { R } _ { : }$ , and the ordinary pivot method introduced by Ailon et al. [1] to obtain another uncapacitated solution P, so that the better solution between R and Split(P) is a (1.9 + ϵ)-approximation.

Lastly, we propose to employ the conditional-afine pivoting framework of Garc´ıa-Soriano and Schohn [17] with diferent vertex admission probabilities for positive neighbors inside the seed, positive neighbors outside the seed, and negative neighbors. Coupling with a weighted disagreement cost analysis, which calculates more accurately the edge errors and thus estimates slightly better than the downsizing argument, we are able to show that the new pivoting method performs better than the ordinary pivoting, and the better solution between R and Split(P) is a (1.7537 + ϵ)-approximation.

Theorem 1 For every fixed rational $\epsilon > 0$ , the Capacitated Cluster Editing problem admits a randomized expected $\left( \frac { 3 5 6 } { 2 0 3 } + \epsilon \right)$ -approximation algorithm that runs in an expected polynomial time in the input size $n ,$ bounded by $\tilde { n } ^ { \mathrm { p o l y } ( 1 / \epsilon ) }$ in which the exponent does not depend on the capacity k, times a polynomial in the precision encoding length.

Note that ${ \begin{array} { l } { { \frac { 3 5 6 } { 2 0 3 } } = 1 . 7 5 3 6 9 4 5 8 1 2 8 0 \dots } \end{array} }$ . Our main contribution is to combine the rounding methods proposed in the literature with the downsizing procedure and analyze their costs jointly. The improvement from the first pivoting method to the second comes from a better bound on the edge errors together with a weighted disagreement cost analysis.

## 1.2 Related work

Approximation algorithms for uncapacitated correlation clustering have evolved through increasingly expressive linear program (LP) relaxations and rounding procedures. Improving upon Bansal et al. [2]’s large constant ratio, Charikar et al. [10] gave a 4-approximation using LP rounding and region growing. Ailon et al. [1] introduced randomized pivoting method, which repeatedly groups a random unclustered vertex with its positive neighbors, obtaining a 3-approximation. Furthermore, their LP-dependent rounding achieves a ratio of 2.5. Van Zuylen and Williamson [26] also presented deterministic pivoting algorithms with the same two guarantees. Afterwards, Chawla et al. [11] refined the LP-based rounding probabilities in [1] to obtain an improved ratio slightly below 2.06. It is proven that the underlying metric LP using pairwise separation variables constrained by triangle inequalities has an integrality gap lower bound of 2 [10, 11].

In the past few years, uncapacitated correlation clustering has experienced another wave of advancement, mostly stronger relaxations retaining correlations among several clustering decisions. Cohen-Addad et al. [14] combined the Sherali-Adams hierarchy with correlated rounding and a global charging argument to obtain a (1.994 + ϵ)-approximation algorithm. Cohen-Addad et al. [13] introduced pre-clustering, which identifies groups of vertices to be kept together and restricts those pairs to share a cluster; the proposed structure controls correlated-rounding errors and supports a combination of set-based and pivot-based rounding with a guarantee of 1.73 + ϵ.

The cluster LP of Cao et al. [7] brings the above ideas into a single relaxation over all candidate clusters, by setting up whole-cluster variables to retain joint membership information beyond pairwise distances. Their fractional partition construction uses pre-clustering and sampling to obtain an explicit feasible solution to the cluster LP of value no more than a fraction ϵ of the integral optimum. Subsequently, correlated rounding errors are handled so that the final rounding can work directly with the resulting cluster weights. Combining cluster sampling with LP-dependent pivoting gives a (1.485 + ϵ)-approximation algorithm in their extended full paper [8].

Soon after, Garc´ıa-Soriano and Schohn [17] improved both the computation and the rounding of the cluster LP. Their approximate dual-separation algorithm uses localization and weak regularity to search for violated dual constraints, yielding approximate primal and dual solutions relative to the fractional cluster-LP optimum. Their rounding couples cluster sampling with a conditional afine pivot rule, which uses membership in a sampled cluster to coordinate inclusion around the selected pivot. A variance-based analysis and derandomization yield a deterministic (1.3865 + ϵ)-approximation. This second wave of developments supply the cluster LP optimization framework and rounding procedures that we adapt to for our capacitated Cluster Editing.

Besides the above mentioned constant-ratio approximation algorithms for the capacitated Cluster Editing [24, 19, 21, 23], Ji et al. [22] consider a variant in which vertex-specific bounds are imposed on the size of the cluster containing each vertex, i.e., non-uniform capacity constraints, and they presented a bicriteria algorithm achieving a cost at most 4 times the optimum while allowing a factor-2 violation of these bounds. In this paper, we study the problem with prescribed uniform capacity.

Cluster Editing has also been studied from the perspectives of exact and parameterized algorithms. For example, the problem can be parameterized by the edit-cost budget, which has received substantial studies [4, 18, 15, 6, 9, 12], and by jointly the edit-cost budget and the number of output clusters [16]. Practical approaches for uncapacitated Cluster Editing include those that combine data reduction with exact branching and integer programming [5], and that integrate exact optimization with greedy randomized adaptive search and iterated-local-search heuristics [3].

## 1.3 Organization of the paper

In the next section, we introduce a set of terminologies and notations to be used in the rest of the paper, as well as the capacitated Cluster LP formulation. The downsizing procedure is presented in Section 3, followed by what it implies — our first (3.7730 + ϵ)-approximation algorithm building on the state-of-the-art algorithm for uncapacitated problem. In Section 4, we employ the near optimal fractional solution to the Cluster LP, which can be computed in polynomial time, and combine the rounding through sampling by Cao et al. [7] and the ordinary pivoting method by Ailon et al. [1], together with the downsizing procedure, to design an improved approximation algorithm with ratio of $1 9 / 1 0 + \epsilon$ We then replace the ordinary pivoting method by the conditional afine pivoting rule of Garc´ıa-Soriano and Schohn [17] to construct another uncapacitated partition dependent on the fractional solution to the Cluster LP, again together with the downsizing procedure, leading to a further improved (356/203 + ϵ)-approximation algorithm, where 356/203 = 1.753694581280 . . . . These results are presented in Section 5. We conclude the paper in Section 6.

## 2 Preliminaries

In this paper, we study the Cluster Editing problem. An instance of the problem consists of a complete edge-signed graph $G = ( V , E ^ { + } \cup E ^ { - } )$ , where an edge $u v \in E ^ { + }$ is called a positive edge or otherwise $u v \in E ^ { - }$ a negative edge, and a positive integer k when the problem is capacitated or otherwise $k = \infty$ when the problem is uncapacitated. A partition $\mathcal { P }$ of the vertex set V is a collection of disjoint nonempty subsets (called parts or clusters, used interchangeably) whose union is V. For the uncapacitated variant there is no constraint on the size of a part in the partition; for the capacitated variant, every part must have size at most k for the partition to be feasible (called k-feasible). This parameter k is referred to as the (uniform) capacity of the clusters. Note that there is no requirement on the number of parts in a partition.

For a feasible partition ${ \mathcal { P } } _ { : }$ , the edge set of P is denoted as $I ( \mathcal P )$ , which contains all the (positive and negative) edges in the complete subgraphs (called cliques) induced by the parts of ${ \mathcal { P } } ,$ i.e., those edges uv such that $u , v$ lie in the same part of P. The disagreement cost of P is

$$
\begin{array} { r } { \mathrm { c o s t } ( \mathcal { P } ) = | E ^ { + } \setminus I ( \mathcal { P } ) | + | E ^ { - } \cap I ( \mathcal { P } ) | = | E ^ { + } \setminus I ( \mathcal { P } ) | + | I ( \mathcal { P } ) \setminus E ^ { + } | = \mathrm { d i s t } ( E ^ { + } , I ( \mathcal { P } ) ) , } \end{array}\tag{1}
$$

which is the edit distance from the positive edge set $E ^ { + }$ to $I ( \mathcal P )$ , by deleting certain edges of $E ^ { + }$ and inserting certain edges of $E ^ { - }$ . The goal of the Cluster Editing problem is to find a feasible partition that minimizes the cost. We use $\mathrm { O P T } _ { k }$ and $\mathrm { O P T _ { \infty } }$ to denote the optimal (i.e., minimum) cost for the k-capacitated and the uncapacitated variants, respectively. One sees that $\mathrm { O P T } _ { \infty } \leq \mathrm { O P T } _ { k }$

In general, for two edge subsets $E _ { 1 }$ and $E _ { 2 }$ , their symmetric diference is $E _ { 1 } \triangle E _ { 2 } = ( E _ { 1 } \setminus E _ { 2 } ) \cup ( E _ { 2 } \setminus E _ { 1 } )$ and their edit distance is dist $\left( E _ { 1 } , E _ { 2 } \right) = \left| E _ { 1 } \triangle E _ { 2 } \right|$ . One sees that dist() is a metric.

Below we give another view for the disagreement cost of a partition $\mathcal { P } .$ . For two disjoint vertex subsets $P _ { 1 }$ and $P _ { 2 } .$ let $e ^ { + } ( P _ { 1 } , P _ { 2 } )$ count the number of positive edges with one endpoint in each set; let $e ^ { + } ( P _ { 1 } )$ and $e ^ { - } ( P _ { 1 } )$ count the number of positive and negative edges in the clique induced on $P _ { 1 }$ , respectively. The cost of using a vertex subset S as a part in a partition is

$$
c ( S ) = e ^ { - } ( S ) + { \textstyle \frac { 1 } { 2 } } e ^ { + } ( S , V \setminus S ) ,\tag{2}
$$

in which $e ^ { - } ( S )$ represents the number of insertions due to S and ${ \textstyle \frac { 1 } { 2 } } e ^ { + } ( S , V \setminus S )$ the number of deletions. Note that the other half of the deletions is attributed to the parts containing the other endpoints of the edges of $e ^ { + } ( S , V \setminus S )$ . This way, we have cost $\begin{array} { r } { ( \mathcal { P } ) = \sum _ { P \in \mathcal { P } } c ( P ) } \end{array}$

For a vertex $v \in V$ , if $u v \in E ^ { + }$ then u is a positive neighbor of $v ;$ the positive neighborhood of v is $N ^ { + } ( v )$ containing all its positive neighbors, the closed positive neighborhood of $v$ is $N ^ { + } [ v ] = \{ v \} \cup N ^ { + } ( v )$ , and the positive degree of v in a vertex subset $S \subseteq V$ is $\bar { d } _ { S } ^ { + } ( v ) = | N ^ { \bar { + } } ( v ) \cap S |$ . We write $d ^ { + } ( v ) = d _ { V } ^ { + } ( v )$

## 2.1 The capacitated cluster linear program

Let ${ \mathcal { F } } _ { k } = \{ S \subseteq V : 1 \leq | S | \leq k \}$ , i.e., all the non-empty vertex subsets of size up to $k .$ For any pair of distinct vertices $u , v ,$ we denote ${ \mathcal { C } } ( u , v ) = \{ S \in { \mathcal { F } } _ { k } \mid \{ u , v \} \subseteq S \}$ , i.e., the collection of all subsets in $\mathcal { F } _ { k }$ containing both vertices. We also denote ${ \mathcal { C } } ( v ) = \{ S \in { \mathcal { F } } _ { k } \mid v \in S \}$ for any vertex v. Using the definition of cluster cost in $\operatorname { E q . } \ ( 2 )$ , we set up a binary variable z for including S as a part in the partition, followed by the capacitated cluster relaxed linear program (abbreviated as Cluster-LP):

$$
\begin{array} { r l } { L _ { k } ^ { * } = \operatorname* { m i n } } & { \displaystyle \sum _ { S \in \mathcal { F } _ { k } } c ( S ) z _ { S } } \\ { \mathrm { s . t . ~ } } & { \displaystyle \sum _ { S \in \mathcal { C } ( v ) } z _ { S } = 1 , \ \forall v \in V ; } \\ & { z _ { S } \geq 0 , \ \forall S \in \mathcal { F } _ { k } . } \end{array}\tag{3}
$$

Here, $\begin{array} { r } { \sum _ { S \in \mathcal { C } ( v ) } z _ { S } = 1 } \end{array}$ is called the coverage equality for the vertex v. As $L _ { k } ^ { * }$ denotes the optimal disagreement for the Cluster-LP, we have $L _ { k } ^ { * } \le \mathrm { O P T } _ { k }$

Given a feasible solution z to the Cluster-LP, we define for each edge uv,

$$
y _ { u v } = \sum _ { S \in \mathcal { C } ( u , v ) } z _ { S } , \ x _ { u v } = 1 - y _ { u v } .\tag{4}
$$

The vertex coverage equalities imply $0 \leq y _ { u v } \leq 1$ , and hence $0 \leq x _ { u v } \leq 1$ too. For a positive edge uv, the total disagreement across all the clusters containing exactly one endpoint is $\begin{array} { r } { \sum _ { S \in \mathcal { C } ( u ) } z _ { S } + \sum _ { S \in \mathcal { C } ( v ) } z _ { S } - 2 y _ { u v } = } \end{array}$ $2 - 2 y _ { u v } = 2 x _ { u v } ;$ therefore, by Eq. (2) this edge contributes $x _ { u v }$ to the objective. Similarly, for a negative edge uv, the total disagreement across all the clusters containing both endpoints is $y _ { u v } = 1 - x _ { u v }$ . It follows that the objective function value of the solution z can be written in three ways as

$$
L ( z ) = \sum _ { S \in \mathcal { F } _ { k } } c ( S ) z _ { S } = \sum _ { u v \in E ^ { + } } x _ { u v } + \sum _ { u v \in E ^ { - } } ( 1 - x _ { u v } ) = \sum _ { u v \in E ^ { + } } ( 1 - y _ { u v } ) + \sum _ { u v \in E ^ { - } } y _ { u v } .
$$

For any three distinct vertices $u , v$ and $w ,$ since $\mathcal { C } ( u , w ) \cup \mathcal { C } ( v , w ) \setminus \mathcal { C } ( u , v ) \subseteq \mathcal { C } ( w )$ and $z _ { S } \geq 0$ for all $S \in \mathcal { F } _ { k }$ , we have $y _ { u w } + y _ { v w } - y _ { u v } \le 1$ , and consequently the triangle inequality holds for x

$$
x _ { u v } \le x _ { u w } + x _ { v w } , \ \forall \mathrm { d i s t i n c t } \ u , v , w .\tag{5}
$$

## 3 A downsizing theorem

Given a partition ${ \mathcal { P } } _ { : }$ , we want to downsize every part of size greater than k to convert it into a k-feasible solution to the capacitated problem. We do this by splitting each part of size $q k + r$ , where $q \geq 1$ and $0 \leq r < k$ , into q subsets of size exactly k and a subset of size r if $r > 0$ . Note that there is no requirement on co-memberships for any vertices during the splitting. The achieved partition through such a procedure is denoted as $\operatorname { S p l i t } _ { k } ( \mathcal { P } )$ .

Lemma 1 Given a partition P, dist $( I ( \mathcal { P } ) , I ( \mathrm { S p l i t } _ { k } ( \mathcal { P } ) ) ) \leq \mathrm { d i s t } ( I ( \mathcal { P } ) , I ( \mathcal { A } ) )$ for any k-feasible partition A.

Proof. We use the k-feasible partition A to intersect the parts of $\mathcal { P }$ to obtain a common refinement denoted as ${ \mathcal { A } } \wedge { \mathcal { P } }$ . In more details, let A be a part of A and $P$ be a part of ${ \mathcal { P } } ;$ then $A \cap P$ becomes a part in the refinement if it is non-empty. ${ \mathcal { A } } \wedge { \mathcal { P } }$ is k-feasible and its edge set $I ( A \land { \mathcal { P } } ) = I ( A ) \cap I ( { \mathcal { P } } )$ . It follows that

$$
\mathrm { d i s t } ( I ( { \mathcal { P } } ) , I ( { \cal A } \wedge { \mathcal { P } } ) ) = | I ( { \mathcal { P } } ) \setminus I ( { \cal A } ) | \leq \mathrm { d i s t } ( I ( { \mathcal { P } } ) , I ( { \cal A } ) ) .
$$

It thus sufices to prove the lemma for only the refinements of $\mathcal { P }$ , and furthermore independently for each part of ${ \mathcal P } .$

Let $P$ be a part of $\mathcal { P } .$ . If a k-feasible refinement of P contains two subsets of sizes $0 < s _ { 1 } \leq s _ { 2 } < k .$ then transferring them into a subset of size $s _ { 2 } + 1$ and, when $s _ { 1 } > 1$ , another of size $s _ { 1 } - 1$ , gives rise to a better refinement reducing the distance by $s _ { 2 } - s _ { 1 } + 1 > 0$ . Repeating such a process leads to the k-feasible partition $\mathrm { S p l i t } _ { k } ( \mathcal { P } )$ . This proves the lemma. ✷

Theorem 2 (Downsizing) For every partition P of the complete edge-signed graph $G = ( V , E ^ { + } \cup E ^ { - } )$ $\mathrm { c o s t } ( \mathrm { S p l i t } _ { k } ( \mathcal { P } ) ) \leq 2 \cdot \mathrm { c o s t } ( \mathcal { P } ) + \mathrm { O P T } _ { k }$ . Hence a polynomial-time α-approximation algorithm for the uncapacitated cluster editing problem yields a polynomial-time $( 2 \alpha + 1 )$ -approximation algorithm for the capacitated problem. The same implication holds for performance guarantees in expectation.

Proof. Let O be an optimal k-feasible partition for the cluster editing problem with capacity k and $\mathcal { Q } = \mathrm { S p l i t } _ { k } ( \mathcal { P } )$ . By $\mathrm { E q . ~ ( 1 ) }$ , the triangle inequality for the edge set edit distance dist() and Lemma 1, we have

$$
\begin{array} { r } { \mathrm { c o s t } ( \mathcal { Q } ) \leq \mathrm { c o s t } ( \mathcal { P } ) + \mathrm { d i s t } ( I ( \mathcal { P } ) , I ( \mathcal { Q } ) ) \leq \mathrm { c o s t } ( \mathcal { P } ) + \mathrm { d i s t } ( I ( \mathcal { P } ) , I ( \mathcal { O } ) ) \leq 2 \cdot \mathrm { c o s t } ( \mathcal { P } ) + \mathrm { c o s t } ( \mathcal { O } ) . } \end{array}
$$

If the performance guarantees are expected, i. $\begin{array} { r } { \mathrm { { s . , } } \operatorname { \mathbb { E } } [ \mathrm { c o s t } ( \mathcal { P } ) ] \leq \alpha \cdot \mathrm { { O P T } } _ { \infty } } \end{array}$ , then taking expectations and using $\mathrm { O P T } _ { \infty } \le \mathrm { O P T } _ { k }$ proves the theorem.

For the running time, one sees that obtaining $\mathcal { Q } = \mathrm { S p l i t } _ { k } ( \mathcal { P } )$ from P can be done in $O ( | V | ^ { 2 } )$ time. ✷

Corollary 1 For every fixed $\epsilon > 0$ , the approximation algorithm of Garc´ıa-Soriano and Schohn $[ 1 \gamma ]$ for the uncapacitated cluster editing problem together with the downsizing procedure gives a $\left( 3 . 7 7 3 0 + \epsilon \right)$ -approximation algorithm for the capacitated problem.

Proof. The algorithm of Garc´ıa-Soriano and Schohn [17, Corollary 1.4] can be set to have an approximation ratio of $1 . 3 8 6 5 + \epsilon / 2$ , and runs in $O ( 2 ^ { \mathrm { p o l y } ( 1 / \epsilon ) } | V | ^ { O ( 1 ) } )$ time. The resulting algorithm, by Theorem 2, has an approximation ratio of $2 \cdot ( 1 . 3 8 6 5 + \epsilon / 2 ) + 1 = 3 . 7 7 3 0 + \epsilon .$ ✷

## 4 An improved approximation algorithm

Let F be the collection of all non-empty subsets of V . Replacing $\mathcal { F } _ { k }$ in the Cluster-LP in Eq. (3) by $\mathcal { F }$ becomes the Cluster-LP for the uncapacitated cluster editing problem, which is employed in the works by Cao et al. [7] and Garc´ıa-Soriano and Schohn [17] that contain the state-of-the-art approximation algorithms. Most notably, it is shown that for any rational $0 < \delta \leq \frac { 1 } { 2 }$ , one can deterministically compute a feasible solution z to the Cluster-LP in time $| V | ^ { \mathrm { p o l y } ( 1 / \delta ) }$ times a polynomial in the precision encoding length, such that z has at most |V| positive values all of polynomial encoding length and $L ( z ) \leq ( 1 + \delta ) L ^ { * }$

One sees that by setting $z _ { S } = 0$ for each $S \in { \mathcal { F } } \backslash { \mathcal { F } } _ { k }$ in the Cluster-LP for the uncapacitated cluster editing problem, it reduces to the Cluster-LP in Eq. (3) for the capacitated problem. Therefore, we can use the conclusion for the uncapacitated problem and summarize it in the below theorem.

Theorem 3 (Approximate fractional solution) Given a complete edge-signed graph $G = ( V , E ^ { + } \cup E ^ { - } )$ an input capacity k, and a rational $\begin{array} { r } { 0 < \delta \le \frac { 1 } { 2 } } \end{array}$ , one can deterministically compute a feasible solution z to the Cluster-LP in Eq. (3) in time $| V | ^ { \mathrm { p o l y } ( 1 / \delta ) }$ times a polynomial in the precision encoding length, such that z has at most |V| positive values all of polynomial encoding length and $L ( z ) \leq ( 1 + \delta ) L _ { k } ^ { * }$

In the rest of the section, we present a hybrid rounding algorithm for z, in Algorithm 1, and prove the rounding guarantee independently of the LP solver for obtaining z. Inside the algorithm z is used for constructing two partitions.

A high-level description of the algorithm is depicted in Algorithm 1, which constructs two candidate partitions and returns the better one as the final solution. Lines 1–6 constitute the first sampling branch that generates a candidate k-feasible partition $\mathcal { R }$ , which is initialized empty at the beginning. In each iteration, let U denote the un-clustered vertices, and $\mathcal { T } _ { U }$ denote the collection of subsets S such that $z _ { S } > 0$ and S contains at least one un-clustered vertex; then each S of $\mathcal { T } _ { U }$ is selected with probability proportional to its $z _ { S }$ value and then $S \cap U$ is added to R. The branch repeats until every vertex is in a cluster of R. Lines 7–11 constitute the second branch that is the ordinary pivot algorithm by Ailon et al. [1]. It generates an uncapacitated partition ${ \mathcal { P } } _ { : }$ which is initialized empty at the beginning. In each iteration, let U denote the un-clustered vertices; each vertex $p \in U$ is selected uniformly at random as the pivot to form a cluster $N ^ { + } [ p ] \cap U , { \mathrm { i . e . , } }$ containing $p$ and all its positive un-clustered neighbors. The branch repeats until every vertex is in a cluster of ${ \mathcal { P } } ,$ of which $\operatorname { S p l i t } _ { k } ( \mathcal { P } )$ is the generated candidate k-feasible partition.

Algorithm 1 Hybrid rounding a fractional solution.   
Require: $G = ( V , E ^ { + } \cup E ^ { - } )$ , k, a feasible solution z to Cluster-LP in Eq. (3);   
Ensure: a partition for V.   
1: $U  V , \mathcal { R }  \emptyset ;$ ▷ Sampling branch;   
2: while $( U \neq \emptyset )$ do   
3: $\mathcal { I } _ { U } = \{ S : z _ { S } > 0 , S \cap U \neq \emptyset \}$ and $\begin{array} { r } { Z _ { U } = \sum _ { S \in \mathcal { I } _ { U } } z _ { S } ; } \end{array}$   
4: choose $S \in { \mathcal { T } } _ { U }$ with probability $z _ { S } / Z _ { U } ;$   
5: add $S \cap U$ to R and update $U  U \backslash S ;$   
6: end while   
$\begin{array} { r } { \begin{array} { r } { 7 \colon U  V , \mathcal { P }  \emptyset ; } \end{array} } \end{array}$ ▷ Ordinary pivoting branch;   
8: while $( U \neq \emptyset )$ do   
9: choose $p \in U$ with probability $1 / | U | ;$   
10: add $C \gets N ^ { + } [ p ] \cap U$ to $\mathcal { P }$ and update $U  U \backslash C ;$   
11: end while   
12: return H ← arg min<sub>Q∈{R,Split (P)}</sub> cost(Q).

One sees that either branch has at most |V| iterations, and the k-feasibility of the partition is guaranteed. The overall running time of Algorithm 1 is in $O ( | V | ^ { 3 } \log | V | )$ , using independent random bits in the iterations.

## 4.1 The performance analysis

Recall in Eq. (4) that we define $\begin{array} { r } { y _ { u v } = \sum _ { S \in \mathcal { C } ( u , v ) } z _ { S } } \end{array}$ for each edge uv, where $\mathcal { C } ( u , v )$ is the collection of subsets of $\mathcal { F } _ { k }$ each containing both end-vertices $u , v$ . With respect to the fractional solution $z ,$ we call $y _ { u v }$ the total weight of the clusters containing both end-vertices, and call $x _ { u v } = 1 - y _ { u v }$ the total weight of the clusters containing exactly one of the two end-vertices. When we fix the edge uv for discussion below, we denote $x : = x _ { u v }$ . In a k-feasible partition, we say u, v are together if they are in the same cluster of the partition, or otherwise separated. The next lemma on the partition R produced in the first branch of Algorithm 1 is analogous to the pairwise sampling principle of Cao et al. [7].

Lemma 2 In the partition R produced in the first branch of Algorithm 1, for every pair of vertices $u , v ,$

$$
\operatorname* { P r } [ u , v \ t o g e t h e r \ i n \ \mathcal { R } ] = \frac { 1 - x } { 1 + x } , \qquad \operatorname* { P r } [ u , v \ s e p a r a t e d \ i n \ \mathcal { R } ] = \frac { 2 x } { 1 + x } .
$$

Proof. Note that $u , v$ are together or separated in R is independent of the cluster construction history before any of u or v is clustered, at that time point every subset of $\mathcal { F } _ { k }$ containing either of them still intersects the current un-clustered vertex set $U _ { \textrm { \scriptsize i } }$ and thus x has not changed. That ${ \mathrm { i s } } ,$ for any history the probabilities that the next selected cluster contains both u and $v ,$ or only u, or only v are ${ \textstyle { \frac { 1 - x } { Z _ { U } } } } , \ { \frac { x } { Z _ { U } } } , \ { \frac { x } { Z _ { U } } }$ , respectively. It follows that the probabilities condition on one of them being clustered are ${ \frac { 1 - x } { 1 + x } } , \ { \frac { x } { 1 + x } } .$ , and $\frac { x } { 1 + x }$ , respectively, independent of the history. If the selected subset S contains both u and $v ,$ then they are in the same cluster $S \cap U$ . If S contains exactly one of u and $v ,$ then that vertex is in $S \cap U$ while the other vertex appears in a diferent cluster, i.e., u and v are separated. This proves the lemma. ✷

Recall that for each positive edge uv, it contributes x towards the objective in the fractional solution z. By Lemma 2, we define the slack of the edge uv, denoted as $d _ { u v }$ , as the diference between twice of its contribution towards the objective in the fractional solution $z , 2 x _ { u v }$ , and its expected contribution towards the objective in the partition $\mathcal { R } , \stackrel { 2 x } { 1 + x }$ . That is, $\begin{array} { r } { d _ { u v } = 2 x _ { u v } - \frac { 2 x _ { u v } } { 1 + x _ { u v } } = \frac { 2 x _ { u v } ^ { 2 } } { 1 + x _ { u v } } } \end{array}$ . Analogously, we can define the

slack for a negative edge. We have

$$
d _ { u v } = \left\{ \begin{array} { l l } { \displaystyle \frac { 2 x _ { u v } ^ { 2 } } { 1 + x _ { u v } } , } & { u v \in E ^ { + } ; } \\ { \displaystyle \frac { ( 1 - x _ { u v } ) ( 1 + 2 x _ { u v } ) } { 1 + x _ { u v } } , } & { u v \in E ^ { - } ; } \end{array} \right.\tag{6}
$$

and $\begin{array} { r } { D = \sum _ { u v \in E ^ { + } \cup E ^ { - } } d _ { u v } } \end{array}$ is called the total slack, which measures exactly the total improvement of the partition R over the bound $2 L ( z )$ and is summarized in the next lemma. One sees that every edge slack $d _ { u v }$ is nonnegative.

Lemma 3 For the partition R produced in the first branch of Algorithm $\begin{array} { r } { \boldsymbol { \mathcal { I } } , \mathbb { E } [ \mathrm { c o s t } ( \mathcal { R } ) ] = 2 L ( \boldsymbol { z } ) - \boldsymbol { D } } \end{array}$

Proof. For each edge uv, its slack is defined as the diference between twice of its contribution towards the objective in the fractional solution z and its expected contribution towards the objective in the partition R. Summing over all edges gives $D = 2 L ( z ) - \mathbb { E } [ \mathrm { c o s t } ( \mathcal { R } ) ]$ , and thus proves the lemma. ✷

We next use the total slack D to bound the disagreement cost of P produced in the second branch of Algorithm 1. Noting that for the selected pivot p in an iteration and a positive neighbor $u \in N ^ { + } [ p ]$ , bringing in another positive neighbor $v \in N ^ { + } [ p ]$ into the cluster $N ^ { + } [ p ] \cap U$ has the risk of introducing an error negative edge $u v ;$ while not bringing in a negative neighbor $v \not \in N ^ { + } [ p ]$ into the cluster $N ^ { + } [ p ] \cap \bar { U }$ also has the risk of introducing an error positive edge uv. Both scenarios involve a bad triangle, which has exactly two positive edges and one negative edge, and plays an important role in the performance analysis.

Lemma 4 For every bad triangle $\begin{array} { r } { T , \sum _ { e \in I ( T ) } d _ { e } \geq 2 / 3 } \end{array}$ , where $I ( T )$ is the edge set of $T .$

Proof. Assume the three vertices in $T$ are $u , v , w$ , and uv is a negative edge. Let $a , b$ denote the two values $x _ { u w }$ and $x _ { v w }$ for the positive edges and $c = x _ { u v }$ . By the triangle inequality for x in Eq. $( 5 ) , a + b \geq c .$

The slack function $f ( t ) = 2 t ^ { 2 } / ( 1 + t )$ for positive edges is nondecreasing and convex in [0, 1], since

$$
f ^ { \prime } ( t ) = 2 - { \frac { 2 } { ( 1 + t ) ^ { 2 } } } \geq 0 , { \mathrm { ~ a n d ~ } } f ^ { \prime \prime } ( t ) = { \frac { 4 } { ( 1 + t ) ^ { 3 } } } > 0 .
$$

Therefore, $f ( a ) + f ( b ) \geq 2 \cdot f ( ( a + b ) / 2 ) \geq 2 \cdot f ( c / 2 )$ . It follows that

$$
\sum _ { e \in I ( T ) } d _ { e } \geq 2 \cdot f ( c / 2 ) + \frac { ( 1 - c ) ( 1 + 2 c ) } { 1 + c } = \frac { 2 c ^ { 2 } } { 2 + c } + \frac { ( 1 - c ) ( 1 + 2 c ) } { 1 + c } = \frac { 2 } { 3 } + \frac { ( 1 - c ) ( 5 c + 2 ) } { 3 ( c + 1 ) ( c + 2 ) } \geq \frac { 2 } { 3 } ,
$$

where the last inequality holds due to $0 \leq c \leq 1$

Lemma 5 For the uncapacitated partition P produced in the second branch of Algorithm $\begin{array} { r } { 1 , \mathbb { E } [ \mathrm { c o s t } ( \mathcal { P } ) ] \leq \frac { 9 } { 2 } D } \end{array}$

Proof. Consider a cluster C added to P at the end of the iteration, in which U is the remaining unclustered vertex set at the beginning. Recall that each vertex of U is selected as the pivot $p$ with an equal probability. We write $C _ { p } = N ^ { + } [ p ] \cap U$ the cluster formed using p. Denote

$$
\begin{array} { l } { { a _ { p } = e ^ { - } ( C _ { p } ) + e ^ { + } ( C _ { p } , U \setminus C _ { p } ) , } } \\ { { b _ { p } = \displaystyle \sum _ { \left\{ u , v \right\} \subseteq U , \left\{ u , v \right\} \cap C _ { p } \neq \emptyset } d _ { u v } . } } \end{array}
$$

One sees that $a _ { p }$ counts the number of edge errors decided in this iteration if p is selected, and $b _ { p }$ is the total edge slack removed from the remaining instance and thus attributed to $C _ { p }$

Consider a negative edge error uv introduced by $C _ { p } { \mathrm { : } }$ We have $p \neq u , v$ , and thus $p u$ and $p v$ are positive edges and then these three vertices form a bad triangle.

Consider a positive edge error uv introduced by $C _ { p } { \mathrm { : } }$ We assume without loss of generality that pu is positive and thus pv is negative, and then these three vertices form a bad triangle too.

We conclude that for each edge error uv counted towards $a _ { p } .$ , it determines the unique bad triangle on $\{ p , u , v \}$ . Conversely, for every bad triangle containing positive edges $p u , p v$ and negative edge uv, it will be discovered when $p \ ( u , v ,$ respectively) is the pivot and the edge uv (pv, pu, respectively) is the edge error. In other words, every bad triangle contained in the subgraph induced by U contributes exactly 3 edge errors when its three vertices are selected as the pivot.

Let $T _ { U }$ be the number of bad triangles contained in the subgraph induced by U. We have

$$
\sum _ { p \in U } a _ { p } = 3 T _ { U } .\tag{7}
$$

Also for every bad triangle on $\{ p , u , v \}$ , the slack $d _ { u v }$ of the edge uv is counted towards $b _ { p }$ since at least one of $u , v$ is in $C _ { p }$ . Therefore, the total slack of the edges of the triangle is included in $\textstyle \sum _ { p \in U } b _ { p } , { \mathrm { i . e . } }$ .,

$$
\sum _ { p \in U } b _ { p } \geq \sum _ { T \subseteq U , T { \mathrm { ~ i s ~ b a d } } e \in I ( T ) } d _ { e } \geq { \frac { 2 } { 3 } } T _ { U } ,\tag{8}
$$

where the last inequality is by Lemma 4.

Let $\mathbb { E } [ \cos ( \mathcal { P } | _ { U } ) ]$ denote the expected cost of the partition P restricted on $U _ { : }$ and $D _ { U }$ denote the total slack of the edges in the subgraph induced by U. We prove by induction that $\begin{array} { r } { \mathbb { E } [ \mathrm { c o s t } ( \mathcal { P } | \boldsymbol { v } ) ] \leq \frac { 9 } { 2 } D _ { U } } \end{array}$ , for which the equality holds trivially when $| U | \le 1$ . When $| U | \geq 2$ , we have

$$
\begin{array} { r l } { \displaystyle \mathbb { E } [ \mathrm { c o s t } ( \mathcal { P } | \boldsymbol { v } ) ] = \frac { 1 } { | U | } \sum _ { p \in U } \left( a _ { p } + \mathbb { E } [ \mathrm { c o s t } ( \mathcal { P } | \boldsymbol { v } _ { \lfloor X \mathcal { C } _ { P } } ) ] \right) } & { } \\ { \displaystyle } & { \le \frac { 1 } { | U | } \sum _ { p \in U } \left( a _ { p } + \frac { 9 } { 2 } D _ { U \setminus \mathcal { C } _ { P } } \right) } \\ & { = \frac { 1 } { | U | } \sum _ { p \in U } \left( a _ { p } + \frac { 9 } { 2 } \left( D _ { U } - b _ { p } \right) \right) } \\ & { = \frac { 9 } { 2 } D _ { U } + \frac { 1 } { | U | } \sum _ { p \in U } \left( a _ { p } - \frac { 9 } { 2 } b _ { p } \right) } \\ & { \le \frac { 9 } { 2 } D _ { U } . } \end{array}
$$

(by inductive hypothesis)

(by Eqs. (7, 8))

We then take $U = V ,$ and thus prove the lemma.

Theorem 4 Given a fractional solution z to the Cluster-LP in Eq. (3), the hybrid rounding Algorithm 1 runs in expected polynomial time in $| V |$ , the number of positive variables in z, and the encoding length $o f z ,$ and returns a k-feasible partition H satisfying

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \frac { 9 } { 5 } L ( z ) + \frac { 1 } { 1 0 } \mathrm { O P T } _ { k } .
$$

Proof. For the running time, suppose z has m positive variables and each of them has at most b bits. Each iteration of the first and the second branches takes an expected polynomial time in $| V | , m , b$ for probability generation followed by cluster formation. There are at most $| V |$ iterations in each branch, and thus the overall running time is in expected polynomial time.

For the quality of the returned partition H, by Theorem 2 and Lemma 5 we have

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { Q } ) ] \leq 2 \cdot \frac { 9 } { 2 } D + \mathrm { O P T } _ { k } = 9 D + \mathrm { O P T } _ { k } .
$$

Hence, using Lemma 3, we have

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \frac { 9 } { 1 0 } \mathbb { E } [ \mathrm { c o s t } ( \mathcal { R } ) ] + \frac { 1 } { 1 0 } \mathbb { E } [ \mathrm { c o s t } ( \mathcal { Q } ) ] \leq \frac { 9 } { 1 0 } ( 2 L ( z ) - D ) + \frac { 1 } { 1 0 } ( 9 D + \mathrm { O P T } _ { k } ) = \frac { 9 } { 5 } L ( z ) + \frac { 1 } { 1 0 } \mathrm { O P T } _ { k } .
$$

This proves the theorem.

We are ready to present the complete algorithm in Algorithm 2 for the capacitated cluster editing problem, of which an instance consists of a complete edge-signed graph $G = ( V , E ^ { + } \cup E ^ { - } )$ , a capacity k, and an approximation tolerance ϵ.

Algorithm 2 The complete approximation algorithm.   
Require: $G = ( V , E ^ { + } \cup E ^ { - } ) , k \geq 1$ , a positive rational ϵ;   
Ensure: a k-feasible partition for V.   
1: if (k = 1) then   
2: return the partition of all singletons;   
3: else if (all positive components are cliques of size at most k) then   
4: return the partition of all the components;   
5: end if   
6: δ ← min $\{ { \textstyle \frac { 5 } { 9 } } \epsilon , { \textstyle \frac { 1 } { 2 } } \}$   
7: use Theorem 3 to compute a solution z;   
8: run Algorithm 1 on z to compute a solution H;   
9: return H.

Theorem 5 For every fixed rational $\epsilon > 0$ , the Capacitated Cluster Editing problem admits a randomized expected $( 1 . 9 + \epsilon )$ -approximation algorithm that runs in an expected polynomial time in the input size $n ,$ bounded by $n ^ { \mathrm { p o l y } ( 1 / \bar { \epsilon } ) }$ in which the exponent does not depend on the capacity k, times a polynomial in the precision encoding length.

Proof. The first conditional statement (Lines 1–5) of Algorithm 2 is exact, that gives an optimal solution. Otherwise, Theorems 3 and 4, together with $L _ { k } ^ { * } \le \mathrm { O P T } _ { k }$ , give

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \left( \frac { 9 } { 5 } \left( 1 + \delta \right) + \frac { 1 } { 1 0 } \right) \mathrm { O P T } _ { k } \leq \left( \frac { 1 9 } { 1 0 } + \epsilon \right) \mathrm { O P T } _ { k } .
$$

The overall running time of the algorithm follows from the running time stated in Theorems 3 and 4, for computing z and H, respectively. ✷

## 5 An improved pivoting algorithm

In the last section, we adopt the ordinary pivot algorithm of Ailon et al. [1] to first generate an uncapacitated partition P, independently from the fractional solution z stated in Theorem 3, followed by the splitting procedure to downsize large parts so that a candidate k-feasible partition is achieved. Both the algorithmic steps and their performance analysis appear straight, with the most important guarantee $\begin{array} { r } { \mathbb { E } [ \cos ( \mathcal { P } ) ] \le \frac { 9 } { 2 } D } \end{array}$ where $D = 2 L ( z ) - \mathbb { E } [ \mathrm { c o s t } ( \mathcal { R } ) ]$ is the total edge slack and R is the k-feasible partition produced by the first sampling branch of Cao et al. [7].

In this section, we propose an improved pivoting algorithm to generate directly a k-feasible partition using the fractional solution z stated in Theorem 3, that replaces the ordinary pivot algorithm in Algorithm 1. We continue to use the notations defined in the previous sections, such as the quantities $\begin{array} { r } { y _ { u v } = \sum _ { S \in \mathcal { C } ( u , v ) } z _ { S } , } \end{array}$ $x _ { u v } = 1 - y _ { u v } , d _ { u v }$ associated with an edge uv, etc. Nevertheless, the upcoming algorithmic description and the analysis use mostly $y _ { u v } \mathrm { ^ { \circ } s }$ . We therefore write the edge slack $d _ { u v }$ in $\operatorname { E q } .$ . (6) as an equivalent function of $y _ { u v }$ below.

Let

$$
f _ { + } ( y ) = \frac { 2 ( 1 - y ) ^ { 2 } } { 2 - y } , \qquad f _ { - } ( y ) = \frac { y ( 3 - 2 y ) } { 2 - y } ;
$$

the slack of edge uv in Eq. (6) can be re-written as

$$
d _ { u v } = \left\{ \begin{array} { l l } { f _ { + } ( y _ { u v } ) , } & { u v \in E ^ { + } ; } \\ { f _ { - } ( y _ { u v } ) , } & { u v \in E ^ { - } . } \end{array} \right.\tag{9}
$$

Given a partition ${ \mathcal { P } } ,$ when it is not k-feasible, we have proven in Theorem 2 that downsizing every part of $\mathcal { P }$ of size greater than k by splitting it into a number of parts of size exactly k and a possible residual part of size less than $k ,$ leads to a k-feasible partition $\mathrm { S p l i t } _ { k } ( \mathcal { P } )$ satisfying cost $( \mathrm { S p l i t } _ { k } ( \mathcal { P } ) ) \leq 2 \cdot \mathrm { c o s t } ( \mathcal { P } ) + \mathrm { O P T } _ { k }$ . The proof of this inequality uses mostly the triangle inequality of the edge set edit distance. Below we separate the disagreement cost of $\mathcal { P }$ into two parts and prove a more precise upper bound.

We denote $p ( \mathcal P ) = | E ^ { + } \setminus I ( \mathcal P )$ | the number of positive edge errors and $n ( \mathcal P ) = | I ( \mathcal P ) \setminus E ^ { + } | = | E ^ { - } \cap I ( \mathcal P ) |$ the number of negative edge errors. Clearly, cost $( \mathcal { P } ) = p ( \mathcal { P } ) + n ( \mathcal { P } )$

Lemma 6 For every partition ${ \mathcal { P } } _ { z }$ , the downsized k-feasible partition $\mathrm { S p l i t } _ { k } ( \mathcal { P } )$ satisfies

$$
\mathrm { c o s t } ( \mathrm { S p l i t } _ { k } ( \mathcal { P } ) ) \leq p ( \mathcal { P } ) + 2 n ( \mathcal { P } ) + \mathrm { O P T } _ { k } .
$$

Proof. Let $F = | I ( \mathcal P ) \setminus I ( \operatorname { S p l i t } _ { k } ( \mathcal P ) )$ | denote the number of edges deleted during the downsizing process, and let $t ^ { - }$ be the number of negative edges among these $F$ edges.

By Lemma 1, for an optimal k-feasible partition ${ \mathcal { O } } ,$ , we have $F \leq | I ( \mathcal P ) \setminus I ( \mathcal O ) | \leq \mathrm { O P T } _ { k } + n ( \mathcal P )$ , since every positive edge of $I ( \mathcal P ) \backslash I ( \mathcal O )$ is an error of $\mathcal { O }$ and every negative edge is an error counted towards $n ( \mathcal { P } )$ Since the downsizing process changes the cost by $( F - t ^ { - } ) - t ^ { - }$ , and $t ^ { - } \geq 0$ , we have

$$
\begin{array} { r } { \mathrm { c o s t } ( \mathrm { S p l i t } _ { k } ( \mathcal { P } ) ) = \mathrm { c o s t } ( \mathcal { P } ) + F - 2 t ^ { - } \leq ( p ( \mathcal { P } ) + n ( \mathcal { P } ) ) + ( \mathrm { O P T } _ { k } + n ( \mathcal { P } ) ) = p ( \mathcal { P } ) + 2 n ( \mathcal { P } ) + \mathrm { O P T } _ { k - 1 } . } \end{array}
$$

This proves the lemma.

✷

Lemma 6 together with the edge slacks in $\operatorname { E q . } \ ( 9 )$ suggest that the second pivoting branch can have better trade-of between positive and negative neighbors when forming a cluster, and that the positive and negative edge errors can be analyzed with unequal weights.

## 5.1 The modified pivoting branch

Our modified pivot rounding algorithm follows the conditional seed-based architecture of Garc´ıa-Soriano and Schohn [17], where a seed is the selected subset containing the chosen pivot and, while positive neighbors inside the seed are surely admitted into the cluster, a diferent admission rule is set for positive neighbors outside the seed. To this purpose, we set up

$$
\beta = \frac { 1 5 3 } { 5 0 } , \qquad r ( y ) = \operatorname* { m a x } \left\{ 0 , 1 - \beta \frac { 1 - y } { 2 - y } \right\} , \qquad m ( y ) = \operatorname* { m i n } \left\{ \frac { 3 } { 2 } y , 2 y - y ^ { 2 } \right\} .\tag{10}
$$

One sees that both $r ( y )$ and $m ( y )$ are increasing in $y .$ The modified pivot rounding algorithm takes in a fractional solution z to Cluster-LP in Eq. (3) and returns an uncapacitated partition $\mathcal { P } _ { \cdot }$ . The final produced candidate k-feasible partition is $\operatorname { S p l i t } _ { k } ( \mathcal { P } )$

In the algorithm, assume $U$ is the current set of un-clustered vertices. It chooses a pivot $p$ from $U$ uniformly at random, and then select the seed S from all those subsets containing $p , \mathrm { i . e . , } \mathcal { C } ( p )$ , with probability $z _ { S }$ (recall the vertex coverage equality in Eq. (3) that $\begin{array} { r } { \sum _ { S \in \mathcal { C } ( p ) } z _ { S } = 1 ) } \end{array}$ . The to-be constructed cluster $C$ initially contains $p$ and all its positive un-clustered neighbors in S, i.e., $N ^ { + } [ p ] \cap ( U \cap S )$ ; then for each positive neighbor $v \in U \backslash S$ , it is added to $C$ only i $\begin{array} { r } { { \dot { y } } _ { p v } \geq \frac { 5 3 } { 1 0 3 } } \end{array}$ and with probability $r ( y _ { p v } )$ . Every un-clustered negative neighbor v is treated the same, that it is added to $\bar { C }$ with probability $m ( y _ { p v } )$ . This forms a new cluster $C$ in $\mathcal { P }$ and the algorithm repeats until all vertices are clustered. $\mathrm { A }$ pseudocode of this modified pivot rounding is depicted in Algorithm 3.

Algorithm 3 Modified pivot rounding a fractional solution.   
Require: $G = ( V , E ^ { + } \cup E ^ { - } )$ , k, a feasible solution z to Cluster-LP in Eq. (3);   
Ensure: an uncapacitated partition $\mathcal { P }$ for V .   
1: $U  V , \mathcal { P }  \bar { \emptyset } ;$   
2: while $( U \neq \emptyset )$ do   
3: choose $p \in U$ with probability $1 / | U | ;$ ▷ p the pivot;   
4: choose $S \in { \mathcal { C } } ( p )$ with probability z<sub>S</sub>;   
5: $C \gets \{ p \}$   
6: for (each $v \in U \setminus \{ p \} )$ do   
7: if $( p v \in E ^ { + }$ and $v \in S )$ then   
8: $C \gets C \cup \{ v \} ;$ ▷ v positive neighbor inside $S ;$   
9: else if $( p v \in E ^ { + } )$ then   
10: $C  C \cup \{ v \}$ with probability $r ( y _ { p v } ) ;$ ▷ v positive neighbor outside $S ;$   
11: else   
12: $C  C \cup \{ v \}$ with probability $m ( y _ { p v } ) $ ▷ v negative neighbor, inside or outside $S ;$   
13: end if   
14: end for   
15: add C to P and update $U  U \backslash C ;$   
16: end while   
17: return P.

## 5.2 A weighted performance analysis

Instead of the disagreement cost of $\mathcal { P } , \mathrm { c o s t } ( \mathcal { P } ) = p ( \mathcal { P } ) + n ( \mathcal { P } )$ as in Lemma 5, in this section we estimate the quantity $p ( \mathcal { P } ) + 2 n ( \mathcal { P } )$ , which can be deemed as weighted cost of $\mathcal { P }$ by assigning twice weight to a negative edge of P. We prove the following lemma.

Lemma 7 The partition P produced by Algorithm 3 satisfies $\mathbb { E } [ p ( \mathcal { P } ) + 2 n ( \mathcal { P } ) ] \le \beta D$

Proof. Consider a cluster C added to P at the end of the current iteration, which is a subset of $U$ the remaining un-clustered vertices in the iteration. Note that each vertex of U is selected as the pivot p with an equal probability; we write $C _ { p }$ for the random cluster for p and denote

$$
a _ { p } = 2 e ^ { - } ( C _ { p } ) + e ^ { + } ( C _ { p } , U \setminus C _ { p } ) , \qquad b _ { p } = \sum _ { \{ u , v \} \subseteq U , \ \{ u , v \} \cap C _ { p } \neq \emptyset } d _ { u v } .
$$

Similar to the proof of Lemma $5 , e ^ { - } ( C _ { p } )$ counts negative edge errors, $e ^ { + } ( C _ { p } , U \setminus C _ { p } )$ counts positive edge errors, but this time $a _ { p }$ is the weighted total error decided in this iteration. Note that $b _ { p }$ is kept the same, which is the total edge slack removed from the remaining instance and thus attributed to the cluster $C _ { p }$

We prove the following inequality

$$
\sum _ { p \in U } \mathbb { E } [ a _ { p } ] \leq \beta \sum _ { p \in U } \mathbb { E } [ b _ { p } ] ,\tag{11}
$$

where the expectations are conditional on the fixed history leading to the beginning of this current iteration.

From how a positive neighbor is admitted into the cluster, we see that a positive edge pv is included in I(P) with probability $\mu _ { p v } = y _ { p v } + ( 1 - y _ { p v } ) r ( y _ { p v } ) \triangleq M ( y _ { p v } )$ , and thus excluded with probability

$$
1 - \mu _ { p v } = 1 - y _ { p v } - ( 1 - y _ { p v } ) \cdot \operatorname* { m a x } \left\{ 0 , 1 - \beta { \frac { 1 - y _ { p v } } { 2 - y _ { p v } } } \right\} = \operatorname* { m i n } \left\{ 1 - y _ { p v } , { \frac { \beta } { 2 } } f _ { + } ( y _ { p v } ) \right\} .\tag{12}
$$

In particular, when $y _ { p v } \geq \frac { 5 3 } { 1 0 3 }$ , the edge exclusion probability is exactly $\frac { \beta } { 2 } f _ { + } ( y _ { p v } )$

On the other hand, a negative edge pv is included in I(P) with probability $\mu _ { p v } = m ( y _ { p v } )$

Similar to the proof of Lemma 5, we use triangles to prove the joint probabilities of the edges of $I ( \mathcal P )$ We denote $\gamma _ { p v } = 1 - r ( y _ { p v } )$ if the edge pv is positive, or otherwise $\gamma _ { p v } = 0$ . Consider a triangle $T$ on three distinct vertices $p , u , v ,$ and let ${ \mathcal { C } } ( p , u , v ) = \{ S \in { \mathcal { F } } _ { k } : \{ p , u , v \} \subseteq S \}$ , followed by $\begin{array} { r } { t _ { p u v } = \sum _ { S \in \mathcal { C } ( p , u , v ) } z _ { S } } \end{array}$ . With the selected pivot $p ,$ the seed $S \in \mathcal { C } ( p , u , v ) \ ( \mathcal { C } ( p , u ) \setminus \mathcal { C } ( p , u , v ) , \mathcal { C } ( p , v ) \setminus \mathcal { C } ( p , u , v ) , \mathcal { C } ( p ) \setminus ( \mathcal { C } ( \vec { p } , u ) \cup \mathcal { C } ( p , v ) )$ respectively) has probability $t \ ( y _ { p u } - t , y _ { p v } - t , 1 - y _ { p u } - y _ { p v } + t ,$ respectively), where $t \triangleq t _ { p u v }$ . Therefore, the joint probability that both $u , v$ are included in $C _ { p } ^ { \phantom { - } } [ 1 7 ]$ is

$$
{ \begin{array} { r l } { { \boldsymbol { J } } _ { p ; u v } = { \left\{ \begin{array} { l l } { t + ( y _ { p u } - t ) r ( y _ { p v } ) + ( y _ { p v } - t ) r ( y _ { p u } ) + ( 1 - y _ { p u } - y _ { p v } + t ) r ( y _ { p u } ) r ( y _ { p v } ) , } & { u , v { \mathrm { ~ b o t h ~ p o s i t i v e ; } } } \\ { \mu _ { p u } \mu _ { p v } , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. } } \\ { = \mu _ { p u } \mu _ { p v } + \gamma _ { p u } \gamma _ { p v } ( t _ { p u v } - y _ { p u } y _ { p v } ) . } & { { \mathrm { ( f o r ~ } } t { \mathrm { ) } } } \end{array} }\tag{13}
$$

It follows that the weighted error made by the edge uv is

$$
A _ { p ; u v } = \left\{ \begin{array} { l l } { \mu _ { p u } + \mu _ { p v } - 2 J _ { p ; u v } , } & { u v \in E ^ { + } , } \\ { 2 J _ { p ; u v } , } & { u v \in E ^ { - } . } \end{array} \right.\tag{14}
$$

For $u v ,$ as long as one endpoint is included in the cluster $C _ { p } .$ its slack is attributed to $C _ { p } .$ and one sees that such probability is $\mu _ { p u } + \mu _ { p v } - J _ { p ; u v } ;$ the slack contributed by uv is thus

$$
B _ { p ; u v } = d _ { u v } ( \mu _ { p u } + \mu _ { p v } - J _ { p ; u v } ) .\tag{15}
$$

Since every vertex of the triangle T can be chosen as the pivot in the iteration, we denote

$$
A _ { T } = A _ { p ; u v } + A _ { u ; v p } + A _ { v ; p u } , \qquad B _ { T } = B _ { p ; u v } + B _ { u ; v p } + B _ { v ; p u } ,
$$

and we will validate that $A _ { T } \le \beta B _ { T }$ for every triangle T.

Triangle validation. We use Eqs. (13)–(15) to validate $A _ { T } \le \beta B _ { T }$ for all four kinds of triangles T, which have 3, 2, 1 and 0 positive edges (denoted as $+ ^ { 3 } - , + ^ { 2 } - , + ^ { 1 } -$ - and $+ ^ { 0 } .$ -triangle), respectively. Let $a , b , c$ denote the three unordered values $y _ { p u } , y _ { p v } , y _ { u v }$ . Recall that $\begin{array} { r } { t = t _ { p u v } = \sum _ { S \in \mathcal { C } ( p , u , v ) } z _ { S } } \end{array}$ . From the vertex coverage equalities of Eq. (3), we have

$$
\operatorname* { m a x } \{ 0 , a + b - 1 , a + c - 1 , b + c - 1 \} \leq t \leq \operatorname* { m i n } \{ a , b , c \} .\tag{16}
$$

For a +<sup>3</sup>-triangle T, order $a \leq b \leq c ;$ Eq. (16) becomes $b + c - 1 \leq t \leq a .$ and $\beta B _ { T } - A _ { T }$ is a linear function in t. It sufices to check that $\beta B _ { T } - A _ { T } \ge 0$ at $t = a$ and $t = \operatorname* { m a x } \{ 0 , b + c - 1 \}$

For a +<sup>2</sup>-triangle T, let $a \leq b$ be the values for two positive edges and c the value for the negative edge; $\mathrm { E q . ~ ( 1 6 ) }$ becomes $a + b - 1 \leq t \leq c$ and $b + c - 1 \leq t \leq a$ . The derivative of $\beta B _ { T } - A _ { T }$ with respect to t is $- ( \beta f _ { - } ( c ) + 2 ) ( 1 - r ( a ) ) ( 1 - r ( b ) ) \leq 0 ;$ hence it sufices to check that $\beta B _ { T } - A _ { T } \geq 0 \mathrm { ~ a t ~ } t = \operatorname* { m i n } \{ a , c \}$

For a +<sup>1</sup>-triangle, let a be the value for the positive edges and $b \leq c$ be the values for the two negative edges; Eq. (16) becomes $a + c - 1 \leq b$ and $b + c - 1 \leq a .$ , since one sees that t does not appear in $\beta B _ { T } - A _ { T }$

For each of the above three kinds of triangles, the function $\Psi ( a , b , c ) \triangleq ( 2 - a ) ( 2 - b ) ( 2 - c ) ( \beta B _ { T } - A _ { T } )$ is a polynomial and $( 2 - a ) ( 2 - b ) ( 2 - c ) > 0$ on $[ 0 , 1 ] ^ { 3 }$ . The non-negativity of the function can be verified by the exact rational certificate, and thus $A _ { T } \le \beta B _ { T }$ is proved.

Lastly, for $\mathrm { ~ a ~ } + ^ { 0 } .$ -triangle, i.e., all three edges of the triangle are negative, we first have $m ( y ) \leq { \frac { 9 } { 8 } } f _ { - } ( y )$ because

$$
{ \frac { 9 } { 8 } } f _ { - } ( y ) - m ( y ) = { \left\{ \begin{array} { l l } { { \frac { 3 y ( 1 - 2 y ) } { 8 ( 2 - y ) } } , } & { 0 \leq y \leq { \frac { 1 } { 2 } } , } \\ { { \frac { y ( 2 y - 1 ) ( 5 - 4 y ) } { 8 ( 2 - y ) } } , } & { { \frac { 1 } { 2 } } \leq y \leq 1 . } \end{array} \right. }
$$

Then, $A _ { T } = 2 ( m ( a ) m ( b ) + m ( b ) m ( c ) + m ( c ) m ( a ) )$ and

$$
B _ { T } \geq \frac { 8 } { 9 } ( 2 ( m ( a ) m ( b ) + m ( b ) m ( c ) + m ( c ) m ( a ) ) - 3 m ( a ) m ( b ) m ( c ) ) \geq \frac { 8 } { 9 } ( m ( a ) m ( b ) + m ( b ) m ( c ) + m ( c ) m ( a ) ) .
$$

Consequently, $\begin{array} { r } { A _ { T } \leq \frac { 9 } { 4 } B _ { T } \leq \beta B _ { T } } \end{array}$ . This proves $A _ { T } \le \beta B _ { T }$ for the last kind of triangles.

It remains to consider the last case where $| U | = 2 ,$ , in which the pivot is an endpoint of the last remaining edge uv. One sees that either way the slack $d _ { u v }$ is attributed to the formed cluster. By Eq. (12), if uv is positive then it contributes a cost of $\begin{array} { r } { 1 - \mu _ { u v } \le \frac \beta 2 f _ { + } ( y _ { u v } ) ; } \end{array}$ ; or otherwise it contributes a weighted cost of $\begin{array} { r } { 2 m ( y _ { u v } ) \leq \frac { 9 } { 4 } f _ { - } ( y _ { u v } ) \leq \beta f _ { - } ( y _ { u v } ) } \end{array}$ . We have thus proved that, when $| U | = 2 ,$ , for the last edge uv its weighted cost is no greater than $\beta d _ { u v }$

Adding $A _ { T } \le \beta B _ { T }$ over all triangles and possibly the above inequality for the last edge proves Eq. (11), which gives the conditional one-iteration inequality for the uniformly selected pivot p by dividing |U|. Summing the conditional inequalities over at most |V| iterations proves the lemma. ✷

## 5.3 The resulting approximation guarantee

Theorem 6 Given a fractional solution z to the Cluster-LP in Eq. (3), the cheaper partition H between the k-feasible solution R by the first branch of Algorithm 1 and the $\mathrm { S p l i t } _ { k } ( \mathcal { P } )$ where P produced by Algorithm 3 satisfies

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \frac { 3 0 6 } { 2 0 3 } L ( z ) + \frac { 5 0 } { 2 0 3 } \mathrm { O P T } _ { k } .
$$

The overall running time is in expected polynomial time in |V|, the number of positive values in z, and the encoding length $o f z .$

Proof. Let ${ \mathcal { Q } } = \operatorname { S p l i t } _ { k } ( { \mathcal { P } } )$ . Lemmas 6 and 7 give $\mathbb { E } [ \mathrm { c o s t } ( \mathcal { Q } ) ] \leq \beta D + \mathrm { O P T } _ { k }$ . Recall that Lemma 3 guarantees $\mathbb { E } [ \mathrm { c o s t } ( \mathcal { R } ) ] \le 2 L ( z ) - D$ . The cheaper cost is no greater than any linear combination of the two costs, and therefore,

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \frac { 1 5 3 } { 2 0 3 } \left( 2 L ( z ) - D \right) + \frac { 5 0 } { 2 0 3 } \left( \frac { 1 5 3 } { 5 0 } D + \mathrm { O P T } _ { k } \right) = \frac { 3 0 6 } { 2 0 3 } L ( z ) + \frac { 5 0 } { 2 0 3 } \mathrm { O P T } _ { k } .
$$

The running time of the entire algorithm is similar to the proof of Theorem 4.

Given that a fractional solution z can be computed in polynomial time so that $L ( z ) \leq ( 1 + \delta ) L _ { k } ^ { * }$ , for any positive δ (Theorem 3), we choose $\delta = \operatorname* { m i n } \{ 1 / 2 , \epsilon / 2 \}$ for any positive ϵ. Using $L _ { k } ^ { * } \le \mathrm { O P T } _ { k }$ , Theorem 6 gives

$$
\mathbb { E } [ \mathrm { c o s t } ( \mathcal { H } ) ] \leq \left( \frac { 3 5 6 } { 2 0 3 } + \epsilon \right) \mathrm { O P T } _ { k } \mathrm { , ~ w h e r e ~ } \frac { 3 5 6 } { 2 0 3 } = 1 . 7 5 3 6 9 4 5 8 1 2 8 0 \ldots . . .
$$

## 6 Conclusion

We studied the capacitated Cluster Editing problem where a graph is given, and the goal is to convert it into a collection of cliques of upper bounded size via a minimum number of edge insertions and deletions. The problem has been studied in the literature but not much, with the best approximation algorithm of ratio 5.37. Nevertheless, the uncapacitated variant has received at least two waves of extensive studies with several algorithmic frameworks, and the state-of-the-art approximation ratio of 1.3865.

In this paper, we presented three improved approximation algorithms, each having a distinct role. Our first (3.7730 + ϵ)-approximation algorithm builds on the proposed downsizing procedure, that converts an uncapacitated partition into a feasible one without increasing the cost too much, and it is achieved by calling the state-of-the-art 1.3865-approximation algorithm for uncapacitated problem. The second algorithm employs the near optimal fractional solution to the Cluster LP formulation of the problem, and combines the rounding through sampling by Cao et al. [7] and the ordinary pivoting method by Ailon et al. [1]. Using the downsizing procedure again, the second algorithm is proven to have an approximation ratio of $1 . 9 + \epsilon .$ Replacing the ordinary pivoting method in the second algorithm by the conditional afine pivoting rule of Garc´ıa-Soriano and Schohn [17], again together with the downsizing procedure, the third algorithm is a further improved (1.7537 + ϵ)-approximation algorithm. We remark that the downsizing procedure, despite simple, has its important role in all three algorithms. It is also important to note that we extended the algorithm design and analysis techniques developed for the uncapacitated problem, though not novel but non-trivially. It remains interesting to see further development for either uncapacitated or capacitated cluster editing, and improved lower bounds on the approximation ratios.

## References

[1] N. Ailon, M. Charikar, and A. Newman. Aggregating inconsistent information: Ranking and clustering. Journal of the ACM, 55(5):23, 2008.

[2] N. Bansal, A. Blum, and S. Chawla. Correlation clustering. Machine Learning, 56:89–113, 2004.

[3] L. Bastos, L. S. Ochi, F. Protti, A. Subramanian, I. C. Martins, and R. G. S. Pinheiro. Eficient algorithms for cluster editing. Journal of Combinatorial Optimization, 31(1):347–371, 2016.

[4] S. B¨ocker, S. Briesemeister, Q. B. A. Bui, and A. Truss. Going weighted: Parameterized algorithms for cluster editing. Theoretical Computer Science, 410(52):5467–5480, 2009.

[5] S. B¨ocker, S. Briesemeister, and G. W. Klau. Exact algorithms for cluster editing: Evaluation and experiments. Algorithmica, 60(2):316–334, 2011.

[6] S. B¨ocker and P. Damaschke. Even faster parameterized cluster deletion and cluster editing. Information Processing Letters, 111(14):717–721, 2011.

[7] N. Cao, V. Cohen-Addad, E. Lee, S. Li, A. Newman, and L. Vogl. Understanding the cluster linear program for correlation clustering. In Proceedings of STOC 2024, pages 1605–1616, 2024.

[8] N. Cao, V. Cohen-Addad, E. Lee, S. Li, A. Newman, and L. Vogl. Understanding the cluster LP for correlation clustering. arXiv:2404.17509v3, 2025.

[9] Y. Cao and J. Chen. Cluster editing: Kernelization based on edge cuts. Algorithmica, 64(1):152–169, 2012.

[10] M. Charikar, V. Guruswami, and A. Wirth. Clustering with qualitative information. Journal of Computer and System Sciences, 71(3):360–383, 2005.

[11] S. Chawla, K. Makarychev, T. Schramm, and G. Yaroslavtsev. Near optimal LP rounding algorithm for correlation clustering on complete and complete k-partite graphs. In Proceedings of STOC 2015, pages 219–228, 2015.

[12] J. Chen and J. Meng. A 2k kernel for the cluster editing problem. Journal of Computer and System Sciences, 78(1):211–220, 2012.

[13] V. Cohen-Addad, E. Lee, S. Li, and A. Newman. Handling correlated rounding error via preclustering: A 1.73-approximation for correlation clustering. In Proceedings of FOCS 2023, pages 1082–1104, 2023.

[14] V. Cohen-Addad, E. Lee, and A. Newman. Correlation clustering with Sherali–Adams. In Proceedings of FOCS 2022, pages 651–661, 2022.

[15] P. Damaschke. Fixed-parameter enumerability of cluster editing and related problems. Theory of Computing Systems, 46(2):261–283, 2010.

[16] F. V. Fomin, S. Kratsch, M. Pilipczuk, M. Pilipczuk, and Y. Villanger. Tight bounds for parameterized complexity of cluster editing with a small number of clusters. Journal of Computer and System Sciences, 80(7):1430–1447, 2014.

[17] D. Garc´ıa-Soriano and A. Schohn. Approximate dual separation for the cluster LP: A 1.387 approximation for correlation clustering. arXiv:2607.27829v2, 2026.

[18] J. Guo. A more efective linear kernelization for cluster editing. Theoretical Computer Science, 410(8– 10):718–726, 2009.

[19] V. P. Il’ev, S. D. Il’eva, and A. A. Navrotskaya. Graph clustering with a constraint on cluster sizes. Journal of Applied and Industrial Mathematics, 10(3):341–348, 2016.

[20] V. P. Il’ev and A. A. Navrotskaya. Computational complexity of the problem of approximation by graphs with connected components of bounded size. Prikladnaya Diskretnaya Matematika, 3(13):80–84, 2011.

[21] S. Ji, Y. Cheng, J. Tan, and Z. Zhao. An improved approximation algorithm for the capacitated correlation clustering problem. International Journal of Foundations of Computer Science, 35(6):757– 774, 2024.

[22] S. Ji, D. Xu, M. Li, and Y. Wang. Approximation algorithm for the correlation clustering problem with non-uniform hard constrained cluster sizes. In Proceedings of AAIM 2019, volume 11640 of Lecture Notes in Computer Science, pages 159–168, 2019.

[23] A. V. Kononov and V. P. Il’ev. Approximation algorithms for the cluster editing problem with small clusters. Discrete Optimization, 57:100900, 2025.

[24] G. J. Puleo and O. Milenkovic. Correlation clustering with constrained cluster sizes and extended weights bounds. SIAM Journal on Optimization, 25(3):1857–1872, 2015.

[25] R. Shamir, R. Sharan, and D. Tsur. Cluster graph modification problems. Discrete Applied Mathematics, 144(1–2):173–182, 2004.

[26] A. van Zuylen and D. P. Williamson. Deterministic pivoting algorithms for constrained ranking and clustering problems. Mathematics of Operations Research, 34(3):594–620, 2009.