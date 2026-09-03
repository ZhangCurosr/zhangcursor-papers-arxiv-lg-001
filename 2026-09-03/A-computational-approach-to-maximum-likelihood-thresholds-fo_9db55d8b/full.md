# A computational approach to maximum likelihood thresholds for colored Gaussian graphical models

Roser Homs<sup>∗</sup> Olga Kuznetsova<sup>†</sup> Bernadette J. Stolz<sup>‡</sup>

September 3, 2026

## Abstract

Gaussian graphical models (GGMs) are essential tools for interpretable structure learning. However, in high-dimensional, small-sample regimes, the available data is often insufficient for the maximum likelihood estimator to exist. Colored Gaussian graphical models (CGGMs) mitigate this limitation by imposing symmetry constraints through graph coloring, which reduces the required sample size. This minimal number of observations needed to guarantee that the estimator exists almost surely is defined as the maximum likelihood threshold (MLT). Here, we address the computation of the MLT for CGGMs by focusing on its geometric formulation: finding the minimum rank of a sample covariance matrix such that its projection lies almost surely within the interior of the cone of suficient statistics. We establish a unified theoretical framework, extending results from uncolored to colored models and introducing new symbolic algorithms. Furthermore, we present a computational study integrating sampling with topological data analysis (TDA) to investigate the local geometry of the cone of suficient statistics. Our results demonstrate the potential of TDA to overcome the computational bottlenecks of traditional symbolic algebraic methods–particularly Gr¨obner basis computations–in analyzing the likelihood geometry of CGGMs.

## 1 Introduction

Gaussian graphical models (GGMs) describe multivariate normal distributions whose conditional independence structure is encoded by a graph: each node represents a random variable, and missing edges correspond to conditional independences. A fundamental methodological question underlying the application of GGMs concerns sample size: what is the minimal number of observations guaranteeing that the maximum likelihood estimator (MLE) exists almost surely? This number is called the maximum likelihood threshold (MLT) and is relevant in the fields characterized by the high-dimensional, small-sample regime, such as the study of biological networks, e.g. gene regulatory networks [35, 34], biochemical pathways [21], protein interactions [33], multi-omics [26].

In colored Gaussian graphical models (CGGMs), introduced in [20], vertex or edge colors impose equality among the corresponding entries in the inverse of the covariance matrix. These symmetry constraints often lead to a reduction in the MLT.

For uncolored chordal graphs, the MLT coincides with the size of the maximal clique [8]. In the general uncolored setting, ML thresholds have been studied using algebraic and convex geometry [6, 30], algebraic matroids [19], rigidity theory [5, 4], and score matching estimators [15]. Nevertheless, for non-chordal or colored graphs, the behavior of ML thresholds remains poorly understood, and eficient computational approaches are lacking.

Here, we address the geometric formulation of the MLT problem for CGGMs: what is the minimum rank of a sample covariance matrix such that its suficient statistics lie almost surely in the interior of the cone of suficient statistics? The contributions of this paper are:

• A unified theoretic framework for ML thresholds of CGGMs and related quantities (weak MLT and generic completion rank), extending existing results from uncolored to colored models, introducing new symbolic algorithms that help resolve open cases, and standardizing notation.

• A computational study that introduces topological data analysis (TDA) computed locally as a method to study (weak) MLT. Our work demonstrates that TDA-based approaches have the potential to overcome the computational limitations of symbolic algebraic methods, particularly those relying on Gr¨obner basis computations, when investigating the likelihood geometry of colored Gaussian graphical models. At the same time, it highlights the challenge of obtaining representative suficient statistics for computational studies, as those arising from small-sample observations tend to concentrate near the boundary of the cone of suficient statistics rather than exploring its interior. We address this challenge by developing an algorithm that emulates uniform sampling by tracing the map between positive semidefinite matrices and their projection into the cone of suficient statistics.

We investigate the performance of our algorithms in the case of the colored 4-cycle, which has the benefit of having been studied extensively in [30], while also presenting challenges which have not been overcome by existing methods. Together, these results strengthen the largely underexplored connection between algebraic statistics and topological data analysis.

## 2 Geometry of MLE existence for Colored Gaussian Graphical Models

In this section, we introduce the theoretical background underlying CGGMs. Our goal is to provide a formal description of the geometry that governs the existence of MLE. For background on statistical graphical models and their algebraic treatment, see [22, Chapter 5].

## 2.1 Colored Gaussian graphical models

Let $\operatorname { S y m } ( m )$ denote the vector space of real symmetric $m \times m$ matrices and let $\mathrm { P D } ( m )$ (resp. $\mathrm { P S D } ( m ) )$ be the full-dimensional open (resp. closed) convex cone of $m \times m$ positive definite (resp. semidefinite) matrices. We denote by $\mathrm { S y m } ( m , n )$ and $\mathrm { P S D } ( m , n )$ , with $n < m$ , the set of m × m symmetric and positive semidefinite matrices, resp., of rank $\leq n$ . Consider a random vector $X \in \mathbb { R } ^ { m }$ following a centered multivariate Gaussian distribution with covariance matrix $\Sigma \in \mathrm { P D } ( m )$ . Let K denote the concentration matrix $K = \Sigma ^ { - 1 }$

Definition 2.1. A colored graph $G = ( V , E , \lambda )$ is an undirected graph on vertex set $V = [ m ]$ and edge set E together with a coloring function $\lambda : V \sqcup E \to [ d ]$ such that $\lambda ( V ) \cap \lambda ( E ) = \emptyset$

Colored graphs encode equalities among the entries of the concentration matrix $K = \Sigma ^ { - 1 }$ Note that the coloring function in Theorem 2.1 is not allowed to assign the same color to both vertices and edges. See [20, Section 3.1] for more details on the statistical interpretation of the vertex and edge symmetry constraints imposed by the coloring.

Definition 2.2. Consider the d-dimensional linear subspace

$$
\begin{array} { r l } & { \mathcal { L } _ { G } = \big \{ K = ( k _ { i j } ) \in \mathrm { S y m } ( m ) \big | k _ { i j } = 0 \mathrm { ~ f o r ~ } ( i , j ) \notin E , } \\ & { \qquad k _ { i i } = k _ { j j } \mathrm { ~ i f ~ } \lambda ( i ) = \lambda ( j ) \mathrm { ~ f o r ~ } i , j \in V , } \\ & { \qquad \mathrm { ~ a n d ~ } k _ { i j } = k _ { l m } \mathrm { ~ i f ~ } \lambda ( i , j ) = \lambda ( l , m ) \mathrm { ~ f o r ~ } ( i , j ) , ( l , m ) \in E \big \} . } \end{array}\tag{1}
$$

The cone $o f$ concentration matrices associated to the colored graph G is ${ \cal K } _ { G } = \mathcal { L } _ { G } \cap \mathrm { P D } ( m )$

Remark 2.3. The cone of concentration matrices ${ \cal K } _ { G } = \mathcal { L } _ { G } \cap \mathrm { P D } ( m )$ is a relatively open convex cone in $\mathcal { L } _ { G }$ . By considering a basis $K _ { 1 } , \ldots , K _ { d }$ of $\mathcal { L } _ { G }$ , one can identify $\kappa _ { G }$ with the open convex cone $\{ ( \lambda _ { 1 } , \dots , \lambda _ { d } ) \in \mathbb { R } ^ { d } : \lambda _ { 1 } K _ { 1 } + \cdot \cdot \cdot + \lambda _ { d } K _ { d } \in \mathrm { P D } ( m ) \}$ of $\mathbb { R } ^ { d }$ . For more details on the underlying convex (algebraic) geometry, see [28, Section 2].

Definition 2.4. A colored Gaussian graphical model is the set of covariance matrices $\mathcal { M } _ { G } =$ $\{ \Sigma \in \mathrm { P D } ( m ) \mid \Sigma ^ { - 1 } \in { \cal K } _ { G } \}$

Remark 2.5. Note that the model $\mathcal { M } _ { G }$ is the intersection of the algebraic variety $ { \mathcal { L } } _ { G } ^ { - 1 }$ with the cone of positive definite matrices $\mathrm { P D } ( m )$ , where $\mathcal { L } _ { G } ^ { - 1 }$ denotes the Zariski closure of $\{ \Sigma \in$ Sym(m) : det $\Sigma \neq 0$ and $\Sigma ^ { - 1 } \in \mathcal { L } _ { \mathrm { G } } \}$

![](images/9249fe09966e388256fef03275814a360938a2027a9fd2f01ef08a617e3170f1.jpg)

![](images/7c2bdfb61e1b4730b5bdae751c72072e6d0771d173061c81bc3dd151e414036b.jpg)

Figure 1: 2-cycle and 4-cycle with equally colored vertices and all edges colored diferently. See Table 2 and Table 1 for examples of other colorings on 3 and 4-cycles.

Example 2.6. Consider the colored 2-cycle $C _ { 2 }$ and 4-cycle $C _ { 4 }$ in Figure 1. Then

$$
\begin{array}{c} \mathcal { L } _ { C _ { 2 } } = \left\{ \left( \begin{array} { l l } { \lambda _ { 1 } } & { \lambda _ { 2 } } \\ { \lambda _ { 2 } } & { \lambda _ { 1 } } \end{array} \right) : \lambda _ { i } \in \mathbb { R } , i \in [ 2 ] \right\} \mathrm { ~ a n d ~ } \mathcal { L } _ { C _ { 4 } } = \left\{ \begin{array} { l l l } { \left( \lambda _ { 1 } \right)} & { \lambda _ { 2 } } & { 0 } & { \lambda _ { 5 } } \\ { \lambda _ { 2 } } & { \lambda _ { 1 } } & { \lambda _ { 3 } } & { 0 } \\ { 0 } & { \lambda _ { 3 } } & { \lambda _ { 1 } } & { \lambda _ { 4 } } \\ { \lambda _ { 5 } } & { 0 } & { \lambda _ { 4 } } & { \lambda _ { 1 } } \end{array}  : \lambda _ { i } \in \mathbb { R } , i \in [ 5 ]  \end{array} \right\} \mathrm { . ~ }
$$

## 2.2 The cone of suficient statistics

Given n independent and identically distributed observations $X _ { 1 } , \ldots , X _ { n }$ drawn from a centered multivariate Gaussian distribution, we consider their sample covariance matrix $S \ =$ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } X _ { i } X _ { i } ^ { \mathsf { T } }$ . Up to additive constants, its log-likelihood function is

$$
\ell _ { S } ( K ) = \log \operatorname* { d e t } K - \langle S , K \rangle ,\tag{2}
$$

where $\langle S , K \rangle = \operatorname { t r } ( S K )$ is the trace inner product on $\operatorname { S y m } ( m )$ . We describe the likelihood as a function of the concentration matrix because $\ell _ { S } ( K )$ is concave on $\mathrm { P D } ( m )$ as opposed to $\ell _ { S } ( \Sigma )$ see [31, Proposition 3.1]. Moreover, since $\kappa _ { G }$ is a convex cone, then for $S \in \operatorname { P D } ( m )$ ,

$$
\begin{array} { l l } { \mathrm { m a x i m i z e } } & { \log \operatorname* { d e t } K - \langle S , K \rangle , } \\ { \mathrm { s u b j e c t ~ t o } } & { K \in { \mathcal K } _ { G } . } \end{array}\tag{3}
$$

is a convex optimization problem with a unique maximum $\hat { K }$ , the MLE for the concentration matrix. The MLE for the covariance matrix is $\hat { \Sigma } = \hat { K } ^ { - 1 }$ . See [31, Section 3] for an introduction to Gaussian likelihood and convex optimization.

Given a basis $K _ { 1 } , \ldots , K _ { d }$ of $\mathcal { L } _ { G }$ , one can decompose $\begin{array} { r } { \langle S , K \rangle = \sum _ { i = 1 } ^ { d } \lambda _ { i } \langle S , K _ { i } \rangle } \end{array}$ , where $K =$ $\Sigma _ { i = 1 } ^ { d } \lambda _ { i } K _ { i }$ and $\lambda _ { i } \in \mathbb { R }$ . Therefore, all information from the data required to perform the convex optimization problem (3) is encoded in the projection map

$$
\pi _ { G } : \operatorname { S y m } ( m ) \longrightarrow \mathbb { R } ^ { d } , \qquad S \longmapsto ( \langle S , K _ { 1 } \rangle , \ldots , \langle S , K _ { d } \rangle ) .\tag{4}
$$

Example 2.7. Consider the 4-cycle $C _ { 4 }$ with equally colored vertices in Example 2.6. Choose the basis $K _ { 1 } , \ldots , K _ { 5 }$ of $\mathcal { L } _ { C _ { 4 } }$ where $K _ { i }$ is given by $\lambda _ { j } = \delta _ { i j }$ . Then

$$
\pi _ { G } : { \mathrm { S y m } } ( 4 ) { \longrightarrow } { \mathbb R } ^ { 5 } , \qquad S = ( s _ { i j } ) \longmapsto ( \operatorname { t r } ( S ) , 2 s _ { 1 2 } , 2 s _ { 1 4 } , 2 s _ { 2 3 } , 2 s _ { 3 4 } ) .
$$

Definition 2.8. The fiber of S with respect to G is

$$
\mathrm { f i b e r } _ { G } ( S ) = \{ S ^ { \prime } \in \mathrm { S y m } ( m ) \mid \pi _ { G } ( S ^ { \prime } ) = \pi _ { G } ( S ) \} .
$$

For any $S ^ { \prime } \in \mathrm { f i b e r } _ { G } ( S ) , \langle S ^ { \prime } , K \rangle = \langle S , K \rangle$ for any $K \in { \mathcal { L } } _ { G }$ or, equivalently, $\ell _ { S ^ { \prime } } = \ell _ { S }$ on $\mathcal { L } _ { G }$

Definition 2.9. The cone of suficient statistics is the open convex cone ${ \mathcal { C } } _ { G } : = \pi _ { G } ( { \mathrm { P D } } ( m ) )$ in $\mathbb { R } ^ { d }$

As any sample covariance matrix is a positive semi-definite matrix by construction, we seek to study the projection of the PSD cone. As established in [28], the euclidean closure of the cone of suficient statistics satisfies $\operatorname { c l } ( { \mathcal { C } } _ { G } ) = \pi _ { G } ( \operatorname { P S D } ( m ) )$ . Projections of PD matrices are interior points of $\mathcal { C } _ { G }$ . To precisely characterize when the projections of PSD matrices yield interior versus boundary points in this cone, we include the following result derived from fundamental principles of convex geometry.

Lemma 2.10. Let $S \in \mathrm { P S D } ( m )$ . Then $\pi _ { G } ( S )$ is on the boundary of $\mathcal { C } _ { G } ~ i f f \left. S , K \right. = 0$ for some non-zero $K \in { \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m )$ . Otherwise $\pi _ { G } ( S )$ is in the interior.

Proof. The cone of suficient statistics $\mathcal { C } _ { G }$ can be viewed inside $\mathrm { S y m } ( m ) / \mathcal { L } _ { G } ^ { \perp } \simeq \mathbb { R } ^ { d }$ as the open convex dual of the cone of concentration matrices ${ \cal K } _ { G } = \mathcal { L } _ { G } \cap \mathrm { P D } ( m )$ , namely

$$
\mathcal { C } _ { G } = \{ S \in \mathrm { S y m } ( m ) / \mathcal { L } _ { G } ^ { \perp } : \langle S , K \rangle > 0 \mathrm { ~ f o r ~ a l l ~ } K \in \mathcal { L } _ { G } \cap \mathrm { P D } ( m ) \} = K _ { G } ^ { * }
$$

and $\mathrm { c l } ( { \mathcal { C } } _ { G } ) = \{ S \in \mathrm { S y m } ( m ) / { \mathcal { L } _ { G } ^ { \perp } } : \langle S , K \rangle \ge 0$ for all $K \in { \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m ) \} = \operatorname { c l } ( \mathcal { K } _ { G } ) ^ { * }$ . Therefore, the boundary of the convex cone of suficient statistics can be described as $\partial \mathcal { C } _ { G } = \{ S \in$ $\mathrm { S y m } ( m ) / \mathcal { L } _ { G } ^ { \perp } : \langle S , K \rangle = 0$ for some non-zero $K \in { \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m ) \}$ □

Remark 2.11. Note that in some references [7, Chapter 2.6], the convex dual ${ \boldsymbol { \kappa } } _ { G } ^ { * }$ is defined as $\operatorname { c l } ( { \mathcal { C } } _ { G } )$ and $\mathcal { C } _ { G }$ is referred to as int $( { \cal K } _ { G } ^ { * } )$ . For more details, see [28, Sections 1,2] and [29, Chapter 8.3].

## 2.3 Existence of the MLE

The following classical result describes existence of the MLE geometrically, see [31, Theorem 5.2]:

Theorem 2.12. The $M L E \hat { \Sigma }$ exists if and only if fiber $_ G ( S ) \cap \operatorname { P D } ( m ) \neq \emptyset$ . If it exists, $\hat { \Sigma }$ is the unique matrix in the fiber satisfying $\hat { \Sigma } ^ { - 1 } \in \mathcal { K } _ { G }$

An immediate corollary is that the MLE exists for a sample covariance matrix $S$ if and only if $\pi _ { G } ( S )$ lies in the interior of the cone of suficient statistics $\mathcal { C } _ { G }$ . In particular, the MLE always exists for any $S \in \operatorname { P D } ( m )$ . By construction, a sample covariance matrix S is always positive semidefinite, hence $\pi _ { G } ( S ) \in \mathrm { c l } ( { \mathcal { C } } _ { G } )$

## MLE existence test via the cone of suficient statistics

Given a colored GGM $\mathcal { M } _ { G }$ and a sample covariance matrix $S \in \mathrm { P S D } ( m )$

• $\pi _ { G } ( S ) \in { \mathcal { C } } _ { G } \Leftrightarrow$ the MLE exists for S

• $\pi _ { G } ( S ) \in \partial \mathcal { C } _ { G } \Leftrightarrow$ the MLE does not exist for $S$

Definition 2.13. The maximum likelihood threshold of $G ,$ mlt(G), is the smallest n such that the MLE exists with probability one for sample covariance matrices of rank $n .$ The weak maximum likelihood threshold, wmlt(G), is the smallest n such that the MLE exists with positive probability for samples of rank n.

Remark 2.14. Generically, the sample covariance matrix of n observations has rank n. Throughout the paper, we refer interchangeably to “rank-n sample covariance matrices” and to “samples of n generic observations” to connect the geometric constraint $S \in P S D ( m , n )$ with its statistical sampling interpretation. Moreover, we call a statement generic when it holds on a Zariski-open dense set (stronger) to distinguish it from probabilistic almost sure statements that hold everywhere except for a positive measure set (weaker). Check [5, Appendix A] for a discussion on the relation among the two notions.

Geometric characterization of ML thresholds

• mlt(G) = minimum n for which $\pi _ { G } ( S ) \in { \mathcal { C } } _ { G }$ for almost every $S \in \mathrm { P S D } ( m , n )$

• wmlt(G) = minimum n for which $\pi _ { G } ( S ) \in { \mathcal { C } } _ { G }$ for some $S \in \mathrm { P S D } ( m , n )$

Remark 2.15. Note that existence of the MLE for some S immediately implies the existence of an open set of positive measure that contains a PD matrix $\Sigma \in \operatorname { f i b e r } _ { G } ( S )$ for which MLE exists, see [19, Definition 5.1].

Example 2.16. Consider the colored 2-clique $C _ { 2 }$ in Example 2.6. In this case, $\mathcal { L } _ { C _ { 2 } } ^ { - 1 } = \mathcal { L } _ { C _ { 2 } }$ and hence the model $\mathcal { M } _ { C _ { 2 } }$ is a convex cone as shown in Figure 2. The projection map $\pi _ { G } :$ $\mathrm { S y m } ( 2 )  \mathbb { R } ^ { 2 }$ is defined by $\pi _ { C _ { 2 } } ( S ) = ( s _ { 1 1 } + s _ { 2 2 } , 2 s _ { 1 2 } )$ , thus the cone of suficient statistics is $\mathcal { C } _ { C _ { 2 } } = \{ ( x , y ) \in \mathbb { R } ^ { 2 } : 0 < | y | < x \}$ . Since $\pi _ { C _ { 2 } } ( \mathrm { P S D } ( 2 , 1 ) ) = \pi _ { C _ { 2 } } ( \mathrm { P S D } ( 2 ) ) = \mathrm { c l } ( \mathcal { C } _ { C _ { 2 } } )$ , then the projection of a rank 1 matrix lies in the interior of $\mathcal { C } _ { C _ { 2 } }$ almost surely and mlt $( C _ { 2 } ) = 1$

![](images/07d6fad8ad5c5bfb51634f183802b2501c2fefde544c696562dfc60e4bb8438c.jpg)  
Figure 2: Diagram representing the likelihood geometry of the colored 2-cycle $C _ { 2 }$

## 3 Computational algebraic geometry approaches to ML thresholds

Given a colored Gaussian graphical model $\mathcal { M } _ { G }$ , a natural way to study existence of the MLE for n observations in $\mathbb { R } ^ { m }$ is to study $\pi _ { G } ( \mathrm { P S D } ( m , n ) )$ , as in the colored 2-cycle from Theorem 2.16. However, providing an explicit description of this set becomes infeasible already for very small graphs, such as the colored 4-cycle from Example 2.6. To take advantage of the machinery from algebraic geometry, instead of studying the projection of the convex cone $\mathrm { P S D } ( m , n )$ , one can study the projection of the algebraic variety $\mathrm { S y m } ( m , n )$

## 3.1 Algebraic relaxation: from convex cones to algebraic varieties

Definition 3.1. The generic completion rank of $G , \ \mathrm { g c r } ( G )$ , is the minimum n for which dim $\pi _ { G } ( \mathrm { S y m } ( m , n ) ) = d .$

Remark 3.2. Although the algebraic relaxation to tackle the MLE existence problem was first introduced by Uhler in [30], the previous definition appeared for the first time in [19], under the name of rank of G. The terminology gcr(G) was later adopted in [6], given the connection to low rank matrix completion problems in the uncolored setting. For more details on the equivalence of MLE existence and completion problems for colored graphs, see Section 3.3.

The following necessary condition relating the number of color classes d to the generic completion rank follows immediately from the definition:

Theorem 3.3. If $\operatorname { g c r } ( G ) = n$ , then $\begin{array} { r } { d \leq n m - \binom { n } { 2 } } \end{array}$

Proof. By definition of $g c r ( G ) , d = \dim \pi _ { G } ( \mathrm { S y m } ( m , n ) ) \leq \dim \mathrm { S y m } ( m , n ) = n m - n ( n - 1 ) / 2$ In the uncolored case $d = m + | E |$ , recovering [6, Theorem 1.4]. □

As is commonplace in algebraic geometry, we study the complexification of the projection map, namely $\pi _ { G } ^ { \mathbb { C } } : \mathrm { S y m } _ { \mathbb { C } } ( m )  \mathbb { C } ^ { d }$ . The definition of $\mathrm { g c r } ( G )$ is independent of the underlying field: dim<sub>R</sub> $\pi _ { \boldsymbol G } \left( \mathrm { S y m } _ { \mathbb { R } } ( \boldsymbol m , \boldsymbol n ) \right) =$ dim<sub>C</sub> π<sub>G</sub> $\left( \mathrm { S y m } _ { \mathbb { C } } ( m , n ) \right)$ . For simplicity, we omit R $\mathrm { o r } \mathbb { C }$ from our notation. Working over C provides two features that do not hold over R:

• For any $S \in \mathrm { S y m } ( m , n )$ , there exists $\ b X \in \mathbb { C } ^ { m \times n }$ such that $S = X X ^ { t }$

• The condition dim $\pi _ { G } \left( \mathrm { S y m } ( m , n ) \right) = d$ is equivalent to $\pi _ { G } \left( \mathrm { S y m } ( m , n ) \right)$ being dense in the target space with respect to Euclidean topology.

Therefore, the restriction of $\pi _ { G }$ to $\mathrm { S y m } ( m , n )$ is dominant (the image of the projection has maximal dimension) if and only if its diferential $d _ { S } \pi _ { G }$ is surjective when evaluated at a generic point $S = X X ^ { t }$ . Given that surjectivity of $d _ { S } \pi _ { G }$ is a rank condition, it is enough that the corresponding Jacobian matrix is full rank at a single point to ensure that it holds generically.

Geometric characterization of all thresholds

• $\mathrm { g c r } ( G )$ = minimum n for which $d _ { S } \pi _ { G }$ is surjective for some $S \in \mathrm { P S D } ( m , n )$

• mlt(G) = minimum n for which $\pi _ { G } ( S ) \in { \mathcal { C } } _ { G }$ for almost every $S \in \mathrm { P S D } ( m , n )$

• wmlt(G) = minimum n for which $\pi _ { G } ( S ) \in { \mathcal { C } } _ { G }$ for some $S \in \mathrm { P S D } ( m , n )$

Let us now characterize the three previous thresholds using a framework that enables direct comparison among them. The following result adapts and generalizes $[ 6 ,$ Proposition 1.2 (2), Proposition 1.6 (2)] of Blekherman-Sinn. For the sake of completeness, we include a variant of the original proof.

Theorem 3.4. Let G be a colored graph with vertex set $V = [ m ]$ . Then

1. $\mathrm { g c r } ( G )$ is the smallest n for which there exists an $m \times n$ matrix X with the property that $\operatorname { i m } ( X ) \not \subset \ker ( K )$ for any $K \in { \mathcal { L } } _ { G } \backslash \{ 0 \}$

2. mlt(G) is the smallest n for which a generic m × n matrix X satisfies $\operatorname { i m } ( X ) \not \subset \ker ( K )$ for any $K \in \mathcal { L } _ { G } \cap \mathrm { P S D } ( m ) \backslash \{ 0 \}$

3. wmlt(G) is the smallest n for which there exists an $m \times n$ matrix X with the property that im $( X ) \not \subset \ker ( K )$ for any $K \in \mathcal { L } _ { G } \cap \mathrm { P S D } ( m ) \backslash \{ 0 \}$

Proof. Consider the characterization of $\mathrm { g c r } ( G )$ in (3.1). Take $S = X X ^ { t }$ for some $\ b X \in \mathbb { C } ^ { m \times n }$ and let us denote by $T _ { S } V _ { n }$ the tangent space to $V _ { n } = \mathrm { S y m } ( m , n )$ at S. The diferential $d _ { S } \pi _ { G } :$ $T _ { S } V _ { n }  \mathbb { C } ^ { d }$ is surjective if the dual map $d _ { S } \pi _ { G } ^ { * } : \left( \mathbb { C } ^ { d } \right) ^ { * } \to ( T _ { S } V _ { n } ) ^ { * }$ is injective. The dual space $\left( \mathbb { C } ^ { d } \right) ^ { * }$ is identified with $\mathcal { L } _ { G }$ via the trace pairing, hence $d _ { S } \pi _ { G } ^ { * }$ fails to be injective if there is a non zero-matrix $K \in { \mathcal { L } } _ { G }$ such that $\langle K , M \rangle = 0$ for any $M \in T _ { S } V _ { n }$ . One can check that $T _ { S } V _ { n } = \{ X A ^ { t } + A X ^ { t } : A \in \mathbb { C } ^ { m \times n } \}$ , and $\langle K , X A ^ { t } + A X ^ { t } \rangle = 0$ for any $A \in \mathbb { C } ^ { m \times n } { \mathrm { ~ i f f ~ } } K X = 0$

Finally, the statements about mlt(G) and wmlt(G) follows from the description of boundary and interior points in Theorem 2.10. Indeed, any $S \in \mathrm { P S D } ( m , n )$ can be written as $S = X X ^ { t }$ for some $\ b X \in \mathbb { R } ^ { m \times n }$ . Because of positive semi-definiteness of both S and K, the condition $\langle S , K \rangle = 0$ is equivalent to $S K = 0$ . One can check that $S K = 0$ is equivalent to $\operatorname { i m } ( X ) \subset$ $\ker ( K )$ . Therefore, $\pi _ { G } ( S )$ is in the interior of $\mathcal { C } _ { G }$ if and only if $\operatorname { i m } ( X ) \not \subset \ker ( K )$ holds for any non-zero $K \in { \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m )$ □

Remark 3.5. Note that the matrix X in the statements of Theorem 3.4 can be regarded as a matrix whose column vectors are the observations $X _ { 1 } , \ldots , X _ { n }$ . The matrix $S = X X ^ { t }$ is the sample covariance matrix up to scaling.

Instead of giving conditions on the kernels of the covariance matrices of our model as in Theorem 3.4, one can alternatively characterize the thresholds by giving conditions on the concentration matrices themselves:

Theorem 3.6. Let G be a colored graph with vertex set $V = [ m ]$ . Then

1. gcr(G) is the smallest n such that there exist an $m \times ( m - n )$ matrix V that satisfies $V A V ^ { t } \notin { \mathcal { L } } _ { G }$ for any non-zero $A \in \mathrm { S y m } ( m - n )$

2. mlt(G) is the smallest n such that a generic $m \times ( m - n )$ matrix V satisfies $V A V ^ { t } \notin { \mathcal { L } } _ { G }$ for any non-zero $A \in \mathrm { P S D } ( m - n )$

3. wmlt(G) is the smallest n such that there exists an $m \times ( m - n )$ matrix V that satisfies $V A V ^ { t } \notin { \mathcal { L } } _ { G }$ for any non-zero $A \in \mathrm { P S D } ( m - n )$

Proof. We will prove that the existence of a rank n matrix X such that im $X ~ \mathcal { L }$ ker K for any $K \in { \mathcal { L } } _ { G } \backslash \{ 0 \}$ is equivalent to the existence of an $m \times ( m - n )$ matrix V such that $V A V ^ { t } \notin { \mathcal { L } } _ { G }$ for any non-zero $A \in \mathrm { S y m } ( m - n )$ . The proof is presented only for $\mathrm { g c r } ( G )$ , since the cases of wmlt(G) and mlt(G) are analogous after restriction to PSD matrices.

Let X be a matrix that satisfies the conditions of Theorem $3 . 4 ( 1 )$ , that is, if im $X \subset$ ker K, then $K \not \in { \mathcal { L } } _ { G } \backslash \{ 0 \}$ . Equivalently, the inclusion can be formulated as (ker $K ) ^ { \perp } \subset ( \mathrm { i m } X ) ^ { \perp }$ . Since K is symmetric, ker $K = ( \mathrm { i m } K ) ^ { \perp }$ . Taking V such that im $V = ( \operatorname { i m } X ) ^ { \perp }$ yields im $K \subset$ im V. Note that for any $K \in \mathrm { S y m } ( m )$ , im $K \subset \operatorname { i m } V$ is equivalent to $K \ : = \ : V A V ^ { t }$ for some $A \ \in$ $\operatorname { S y m } ( m - n )$ . By assumption, $K \not \in { \mathcal { L } } _ { G } \backslash \{ 0 \}$ , hence $V A V ^ { t } \notin { \mathcal { L } } _ { G }$ for any $A \neq 0$

Conversely, let V satisfy the conditions of Theorem $3 . 6 ( 1 )$ . It can be similarly checked that any matrix X such that im $X = ( \operatorname { i m } V ) ^ { \perp }$ satisfies Theorem $3 . 4 ( 1 )$ □

An immediate consequence of Theorem 3.4 is that gcr(G) gives an upper bound to ${ \mathrm { m l t } } ( G )$ This result was originally stated in [30] as the Elimination Criterion (Theorem 3.3) and proved with diferent techniques (see Section 3.2).

Corollary 3.7. mlt $( G ) \leq \operatorname { g c r } ( G )$

A direct consequence of the gcr(G) characterization of (3.1) is that it can be computed in random polynomial time by checking the rank of the Jacobian of $\pi _ { G }$ at a random point. For more details, see [3, Algorithm 3.2], which was originally designed for uncolored bipartite graphs. However, $\mathrm { g c r } ( G )$ is not always a sharp bound for mlt(G). The smallest uncolored graph where the maximum likelihood threshold and the generic completion rank do not coincide is the bipartite graph $K _ { 5 , 5 }$ has mlt $( K _ { 5 , 5 } )$ = 4 but $\mathrm { g c r } ( K _ { 5 , 5 } ) = 5$ , see [6, 4]. In the colored case, the 4-cycle with equally colored vertices in Figure 1 already displays this behavior, as we prove in Theorem 3.26.

## 3.2 Elimination ideals of a graph

The rank approach to compute the generic completion rank provides an upper bound for the maximum likelihood threshold, but whenever this bound is not sharp, it does not provide further insight into its exact value. However, the original proof of this upper bound [30, Theorem 3.3] provides a constructive definition of the generic completion rank via the computation of elimination ideals.

Definition 3.8. Let the $n { - } t h$ elimination ideal of $G , I _ { G , n }$ , be the vanishing ideal of the projection of $\mathrm { S y m } ( m , n )$ via $\pi _ { G }$ , that is,

$$
I _ { G , n } : = \Big ( I _ { n + 1 } ( S ) + \Big \langle \{ t _ { i } - \langle S , K _ { i } \rangle \} _ { 1 \leq i \leq d } \Big \rangle \Big ) \cap \mathbb { R } [ t _ { 1 } , \dots , t _ { d } ] ,
$$

where $I _ { n + 1 } ( S )$ denotes the ideal of $( n + 1 )$ -minors of the $m \times m$ symmetric matrix ${ \cal { S } } = ( s _ { i j } )$ and $K _ { 1 } , \ldots , K _ { d }$ is a basis of $\mathcal { L } _ { G }$

The n-th elimination ideal of G consists of all polynomials in $\mathbb { R } [ t _ { 1 } , \ldots , t _ { d } ]$ that vanish for all suficient statistics $t _ { i } = \langle S , K _ { i } \rangle$ obtained from a rank n sample covariance matrix $S = \left( s _ { i j } \right)$ . More precisely, $I _ { G , n }$ is the vanishing ideal of $\pi _ { G } ( \mathrm { S y m } ( m , n ) )$ . Thus, while Theorem 3.1 characterizes the generic completion rank geometrically via the dimension of the image under the projection map $\pi _ { G }$ , this is the algebraic characterization originally proposed in [30]:

Proposition 3.9. gcr(G) is the minimum n for which $I _ { G , n } = ( 0 )$

Example 3.10. Consider the colored 2-cycle $C _ { 2 }$ from Example 2.6 and theorem 2.16. Then

$$
I _ { C _ { 2 } , 1 } = \langle s _ { 1 1 } s _ { 2 2 } - s _ { 1 2 } ^ { 2 } , t _ { 1 } - ( s _ { 1 1 } + s _ { 2 } ) , t _ { 2 } - 2 s _ { 1 2 } \rangle \cap \mathbb { R } [ t _ { 1 } , t _ { 2 } ] = ( 0 ) ,
$$

thus mlt $( C _ { 2 } ) \leq \operatorname { g c r } ( C _ { 2 } ) = 1$ and the MLE exists for rank one matrices $S$ with probability 1.

If $I _ { G , n } \neq ( 0 )$ , the MLE can either not exist, exist with probability strictly between 0 and 1, or exist with probability 1, as illustrated in Figure 3. A significant advantage of knowing the elimination ideal - beyond mere knowledge of $\mathrm { g c r } ( G )$ - is that now we can study the generators of $I _ { G , n }$ to obtain bounds for maximum likelihood thresholds.

Lemma 3.11. If there exists $f ( t _ { 1 } , \dots , t _ { n } ) \in I _ { G , n }$ such that $f ( \langle \Sigma , K _ { 1 } \rangle , \dots , \langle \Sigma , K _ { d } \rangle ) \neq 0$ for any $\Sigma = \left( \sigma _ { i j } \right)$ in $\mathrm { P D } ( m )$ , then wmlt $( G ) \geq n + 1$

Proof. By construction, $f ( \langle S , K _ { 1 } \rangle , \dots , \langle S , K _ { d } \rangle ) = 0$ for all $S \in \mathrm { S y m } ( m , n )$ . But then $\pi _ { G } ( \Sigma ) \neq$ $\pi _ { G } ( S )$ for any $\Sigma \in \mathrm { P D } ( m )$ , hence the MLE never exists for rank n. □

![](images/a5b1eb413467664cb37b6c8a1e0663d3ecef86fd061b5bc443466a33c936e3ad.jpg)  
Figure 3: In orange, the interior of the cone of suficient statistics $\mathcal { C } _ { G }$ . In blue, the generator f of a principal elimination ideal $I _ { G , n } = \langle f \rangle$ . From left to right, the MLE does not exist, it exists with probability strictly between 0 and 1, and it exists with probability 1.

We design a positivity certificate for (at least one of) the generators of $I _ { G , n }$ when evaluated at positive definite matrices. In particular, we consider the Cholesky decomposition $\Sigma = L L ^ { t }$ where $L = \left( l _ { i j } \right)$ is a lower-triangular matrix with strictly positive diagonal entries, and compute a sum of squares (SOS) decomposition of f in $\mathbb { R } [ l _ { i j } ]$

Algorithm 1: Non-negativity certificate for generators $I _ { G , n }$ via sums-of-squares (SOS)   
Data: m - number of vertices of the graph; $K _ { 1 } , \ldots , K _ { d }$ standard basis of ${ \mathcal { L } } _ { G } ;$   
$f _ { 1 } , \ldots , f _ { k } \in \mathbb { R } [ t _ { 1 } , \ldots , t _ { d } ]$ - generators of ${ \cal I } _ { G , n } .$   
Result: for each $f _ { i } ,$ whether it changes sign or not for PD matrices.   
begin   
Generate random m × m positive definite matrices Σ;   
For each i, compute $f _ { i } ( \Sigma ) : = f _ { i } ( \langle \Sigma , K _ { 1 } \rangle , \dots , \langle \Sigma , K _ { d } \rangle )$ ;   
if Σ and $\Sigma ^ { \prime }$ are found such that sign $( f _ { i } ( \Sigma ) f _ { i } ( \Sigma ^ { \prime } ) ) < 0$ then   
return $f _ { i }$ changes signs for PD matrices;   
else   
Define an $m \times m$ lower triangular matrix $L = ( l _ { i j } ) ;$   
Compute a sum-of-squares decomposition of $f _ { i } ( L L ^ { t } )$   
return output of the SOS problem for $f _ { i } ( L L ^ { t } )$   
Remark 3.12. Note that an SOS decomposition only provides a non-negativity certificate.   
Therefore, Algorithm 1 requires further analysis of its output to ensure that the resulting SOS   
is strictly positive by using the fact that the diagonal entries $l _ { i i }$ of the lower triangular matrix   
$L = ( l _ { i j } )$ are strictly positive.   
Example 3.13. Let G be the colored 4-cycle $G _ { 9 }$ in [30, Table 2] displayed in Table 1. Consider   
the suficient statistics $t _ { 1 } = s _ { 1 , 1 } , t _ { 2 } = s _ { 2 , 2 } + s _ { 4 , 4 } , t _ { 3 } = s _ { 3 , 3 } , t _ { 4 } = 2 s _ { 1 , 2 } + 2 s _ { 1 , 4 } , t _ { 5 } = 2 s _ { 2 , 3 } ,$   
$t _ { 6 } = 2 s _ { 3 , 4 }$ . The elimination ideals are $I _ { G , 2 } = I _ { G , 3 } = 0 .$ and   
$I _ { G , 1 } = \langle t _ { 3 } t _ { 4 } ^ { 2 } - t _ { 1 } t _ { 5 } ^ { 2 } - 2 t _ { 1 } t _ { 5 } t _ { 6 } - t _ { 1 } t _ { 6 } ^ { 2 } , 4 t _ { 2 } t _ { 3 } - t _ { 5 } ^ { 2 } - t _ { 6 } ^ { 2 } \rangle .$ (5)   
Let $f _ { 1 }$ and $f _ { 2 }$ be the generators of $I _ { G , 1 }$ . An empirical test shows the following signs:   
n 1 2 3 4   
f<sub>1</sub> 0 ± ± ±   
f<sub>2</sub> 0 + + +   
The generator $f _ { 2 }$ evaluated at $\Sigma = L L ^ { t }$ , where $L = ( l _ { i j } )$ is lower-triangular with $l _ { i i } > 0 .$ is   
f<sub>2</sub>(Σ) = 4(−l<sub>13</sub>l<sub>22</sub> + l<sub>12</sub>l<sub>23</sub>)<sup>2</sup> + 4(l<sub>12</sub>l<sub>33</sub>)<sup>2</sup> + 4(−l<sub>14</sub>l<sub>23</sub> + l<sub>13</sub>l<sub>24</sub>)<sup>2</sup>   
+ 4(−l<sub>14</sub>l<sub>33</sub> + l<sub>13</sub>l<sub>34</sub>)<sup>2</sup> + 4(l<sub>13</sub>l<sub>44</sub>)<sup>2</sup> + 4(l<sub>22</sub>l<sub>33</sub>)<sup>2</sup>   
$+ 4 ( - l _ { 2 4 } l _ { 3 3 } + l _ { 2 3 } l _ { 3 4 } ) ^ { 2 } + 4 ( l _ { 2 3 } l _ { 4 4 } ) ^ { 2 } + 4 ( l _ { 3 3 } l _ { 4 4 } ) ^ { 2 } > 0 .$   
Thus, the MLE does not exist for 1 observation and wmlt $( G ) = \mathrm { m l t } ( G ) = \mathrm { g c r } ( G ) = 2 .$

Example 3.14. Consider the colored 4-cycle $G _ { 1 1 }$ in [30, Table 2] displayed in Table 1. Consider the suficient statistics $t _ { 1 } = s _ { 1 , 1 } + s _ { 3 , 3 } , t _ { 2 } = s _ { 2 , 2 } + s _ { 4 , 4 } , t _ { 3 } = 2 s _ { 1 , 2 } , t _ { 4 } = 2 s _ { 1 , 4 } + 2 s _ { 2 , 3 } , t _ { 5 } = 2 s _ { 3 , 4 }$ The elimination ideals are $I _ { G , 2 } = I _ { G , 3 } = 0$ , and $I _ { G , 1 } = \langle 4 t _ { 1 } t _ { 2 } - t _ { 3 } ^ { 2 } - t _ { 4 } ^ { 2 } + 2 t _ { 3 } t _ { 4 } - t _ { 5 } ^ { 2 } \rangle$ . For any positive definite matrix $\Sigma = L L ^ { t }$ , the single generator of $I _ { G , 1 }$ is

$$
f ( \Sigma ) = l _ { 3 3 } ^ { 2 } l _ { 4 4 } ^ { 2 } + l _ { 2 2 } ^ { 2 } l _ { 3 3 } ^ { 2 } + l _ { 1 1 } ^ { 2 } l _ { 4 4 } ^ { 2 } + g ,
$$

where g is an SOS. Therefore, wmlt $( G ) = \mathrm { m l t } ( G ) = \mathrm { g c r } ( G ) = 2 .$

We use Algorithm 1 to provide detailed computations for all colored 3-cycles in Table 2 in Section A. Moreover, we are now able to complete the gaps in [30, Table 2] for colored 4-cycles.

Proposition 3.15. For graphs 9, 11, 14, 17 in [30, Table 2], $\operatorname { g c r } ( G ) = \operatorname { m l t } ( G ) = \operatorname { w m l t } ( G ) = 2$

Proof. The statement is proven for graphs 9 and 11 in Theorem 3.13 and Theorem 3.14, respectively, using Algorithm 1. Note that if $\mathcal { L } _ { G } \subset \mathcal { L } _ { G ^ { \prime } }$ , then mlt $( G ) \leq \mathrm { m l t } ( G ^ { \prime } )$ and wmlt $( G ) \leq$ wmlt(G<sup>′</sup>). Since $\mathcal { L } _ { G _ { 9 } } \subset \mathcal { L } _ { G _ { 1 7 } }$ and $\mathcal { L } _ { G _ { 1 1 } } \subset \mathcal { L } _ { G _ { 1 4 } }$ , the statement holds. □

## 3.3 Generalized matrix completion problems

The study of the existence of maximum likelihood estimators for uncolored Gaussian graphical models was initiated by Dempster [13] through the lens of PD matrix completion. In an uncolored graph, missing edges correspond to unknown entries in the sample covariance matrix. The MLE exists if and only if there is a PD completion for this partial matrix. Consequently, the MLT represents the minimum rank of a sample covariance matrix such that a PD completion exists almost surely. Similarly, the GCR can be interpreted as a low-rank completion problem, that is, the lowest rank of a completion with (possibly) complex entries of a generic partial matrix.

Remark 3.16. As in Section 3.1, not all properties hold over the reals. When we only consider real completions, the GCR is the lowest completion rank for a full-dimensional semi-algebraic set of matrices with a given pattern of unknown entries. However, this is not necessarily true for a generic partial matrix: there can be other full-dimensional semi-algebraic sets with higher lowest completion rank (so-called typical ranks). See [3] for more details on generic and typical ranks.

Introducing a coloring function λ generalizes the notion of matrix completion. Rather than merely filling in gaps, a coloring imposes symmetry constraints that allow entries belonging to the same color class to be modified, provided their sum —– the suficient statistic —– remains preserved. In this framework, we can think of elements in ${ \mathrm { f i b e r } } _ { G } ( S )$ as generalized matrix completions of the suficient statistics $\pi _ { G } ( S )$

Example 3.17. A matrix completion of ${ \cal { S } } = ( s _ { i j } )$ associated to the uncolored 4-cycle G is of the form displayed below for some $x , y \in \mathbb { R } .$ . If we color all vertices equally as in $C _ { 4 }$ in Theorem 2.7, then the generalized matrix completion of ${ \cal { S } } = ( s _ { i j } )$ is required to preserve the trace but not the individual diagonal entries.

$$
\operatorname { f i b e r } _ { G } ( S ) = \left\{ { \left( \begin{array} { l l l l } { s _ { 1 1 } } & { s _ { 1 2 } } & { x } & { s _ { 1 4 } } \\ { s _ { 1 2 } } & { s _ { 2 2 } } & { s _ { 2 3 } } & { y } \\ { x } & { s _ { 2 3 } } & { s _ { 3 3 } } & { s _ { 3 4 } } \\ { s _ { 1 4 } } & { y } & { s _ { 3 4 } } & { s _ { 4 4 } } \end{array} \right) } : x , y \in \mathbb { R } \right\}
$$

$$
\operatorname { f h e r } _ { C _ { 4 } } ( S ) = \left\{ \left( \begin{array} { c c c c } { \mathscr { s } _ { 1 1 } + \varepsilon _ { 1 } } & { \mathscr { s } _ { 1 2 } } & { x } & { \mathscr { s } _ { 1 4 } } \\ { \mathscr { s } _ { 1 2 } } & { \mathscr { s } _ { 2 2 } + \varepsilon _ { 2 } } & { \mathscr { s } _ { 2 3 } } & { y } \\ { x } & { \mathscr { s } _ { 2 3 } } & { \mathscr { s } _ { 3 3 } + \varepsilon _ { 3 } } & { \mathscr { s } _ { 3 4 } } \\ { \mathscr { s } _ { 1 4 } } & { y } & { \mathscr { s } _ { 3 4 } } & { \mathscr { s } _ { 4 4 } + \varepsilon _ { 4 } } \end{array} \right) : x , y , \varepsilon _ { i } \in \mathbb { R } , \sum _ { i = 1 } ^ { 4 } \varepsilon _ { i } = 0 \right\}
$$

In [6], Blekherman and Sinn rephrase the characterizations of MLE and GCR of uncolored Gaussian graphical models as matrix completion problems through the lens of algebraic geometry [6, Proposition 1.2 (1), Proposition 1.6 (1)]. Their framework can be translated into the colored setting with minor variations. Let $\mathcal { L } _ { G } ^ { \perp }$ be the orthogonal space to $\mathcal { L } _ { G }$ in $\operatorname { S y m } ( m )$ . Note that ker $\pi _ { G } = \mathcal { L } _ { G } ^ { \bot }$ , hence matrices in $S + { \mathcal { L } } _ { G } ^ { \perp }$ are (generalized) matrix completions of S.

Definition 3.18. Consider a basis $K _ { d + 1 } , \ldots , K _ { r }$ of $\mathcal { L } _ { G }$ , with $r = m ( m + 1 ) / 2$ . We call $I _ { G }$ the ideal in $\mathbb { R } [ y _ { 1 } , \dots , y _ { m } ]$ generated by quadratic forms $( y _ { 1 } \dots y _ { m } ) K _ { i } ( y _ { 1 } \dots y _ { m } ) ^ { t } $ , for $d < i \leq r$

Recalling the one-to-one correspondence between symmetric matrices and quadratic forms, $\mathcal { L } _ { G } ^ { \perp }$ can be identified with the linear space of quadratic forms $\langle I _ { G } \rangle _ { \mathrm { 2 } } , \mathrm { i . e }$ . the degree 2 part of the homogeneous ideal $I _ { G }$ . Now consider the quotient ring $R = \mathbb { R } [ y _ { 1 } , \dots , y _ { m } ] / I _ { G }$ . Let $R _ { i }$ denote the homogeneous degree i part of R. Note that $R _ { 1 } = \mathbb { R } [ y _ { 1 } , \dots , y _ { m } ] _ { 1 }$ , so we can view an observation $X = ( x _ { 1 } , \dots , x _ { m } ) ^ { t }$ $\mathbb { R } ^ { m }$ as a polynomial (or a linear form) $x _ { 1 } y _ { 1 } + \cdot \cdot \cdot + x _ { m } y _ { m }$ in $R _ { 1 }$ . Moreover, $R _ { 2 }$ can be identified with $\mathbb { S } ^ { m } / \mathcal { L } _ { G } ^ { \perp }$ . That is, $R _ { 2 }$ is the d-dimensional vector space of suficient statistics of all symmetric matrices and the cone of suficient statistics $\mathcal { C } _ { G }$ can be regarded as the subspace of $R _ { 2 }$ consisting of the suficient statistics of all positive definite matrices.

Remark 3.19. For an uncolored graph $G , I _ { G }$ is a squarefree monomial ideal. Adding a coloring breaks both properties, but we still have an ideal generated in degree 2. Even when $I _ { G }$ is not monomial, one can take a monomial basis of $R _ { 2 }$ by choosing certain vertices and edges to represent each color class. See Theorem 3.20 for details on this construction.

Example 3.20. Consider the 4-cycle with equally colored vertices and a basis of $\mathcal { L } _ { G } ^ { \perp }$ given by

$$
\begin{array} { r } { \left[ \begin{array} { l l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { - 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] , \left[ \begin{array} { l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { - 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] , \left[ \begin{array} { l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { - 1 } \end{array} \right] , \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } \end{array} \right] , \left[ \begin{array} { l l l l } { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } \end{array} \right] . } \end{array}
$$

Then $I _ { G } = \langle y _ { 1 } y _ { 3 } , y _ { 2 } y _ { 4 } , y _ { 1 } ^ { 2 } - y _ { 2 } ^ { 2 } , y _ { 1 } ^ { 2 } - y _ { 3 } ^ { 2 } , y _ { 1 } ^ { 2 } - y _ { 4 } ^ { 2 } \rangle \subset \mathbb { R } [ y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } ]$ , and the first and second degree parts of $R = \mathbb { R } [ y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } ] / I _ { G }$ are $R _ { 1 } = \langle { y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } } \rangle _ { \mathbb { R } }$ and $R _ { 2 } = \langle y _ { 1 } ^ { 2 } , y _ { 1 } y _ { 2 } , y _ { 2 } y _ { 3 } , y _ { 3 } y _ { 4 } , y _ { 1 } y _ { 4 } \rangle _ { \mathbb { R } }$ 2 considering the monomial basis obtained from representatives of each color class.

We now adapt and extend the characterization of thresholds in terms of matrix completion problems given by Blekherman-Sinn in [6, Proposition $1 . 2 \ : ( 1 )$ , Proposition $1 . 6 \left( 1 \right) ]$ . We interpret their linear series as a collection of observations $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$ and denote by $\langle X _ { 1 } , \ldots , X _ { n } \rangle _ { 2 }$ their degree two span when considered as elements of $R _ { 1 }$

Theorem 3.21. Let G be a colored graph with vertex set $V = [ m ]$ . Then

1. $\mathrm { g c r } ( G )$ is the smallest n for which the vector space $\langle X _ { 1 } , \ldots , X _ { n } \rangle _ { 2 } + { \mathcal { L } } _ { G } ^ { \perp } = \mathrm { S y m } ( m )$ for some observations $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$

2. mlt(G) is the smallest n for which the vector space $\langle X _ { 1 } , \ldots , X _ { n } \rangle _ { 2 } + { \mathcal { L } } _ { G } ^ { \perp }$ contains a positive definite matrix for generic observations $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$

3. wmlt(G) is the smallest n for which the vector space $\langle X _ { 1 } , \ldots , X _ { n } \rangle _ { 2 } + { \mathcal { L } } _ { G } ^ { \perp }$ contains a positive definite matrix for some observations $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$

Proof. The proof follows exactly as in the uncolored setting in [6]. The key fact is the equality $\langle X _ { 1 } , \ldots , X _ { n } \rangle _ { 2 } = T _ { S } V _ { n } $ , where X is an $m \times n$ matrix with $X _ { i }$ as column i and $S = X X ^ { t }$ . Note that fiber ${ \bf \chi } _ { G } ( S ) = S + \mathcal { L } _ { G } ^ { \perp }$ . One can check that the existence of a PD matrix in the fiber is equivalent to the existence of such a matrix in $T _ { S } V _ { n } + { \mathcal { L } } _ { G } ^ { \bot }$ □

Remark 3.22. Although the proof of Theorem 3.21 does generalize directly from [6, Proposition 1.2, 1.6], results by Blekherman-Sinn on clique-sums of uncolored graphs [6, Theorems 1.13, 1.15] do not extend to the colored setting. $\operatorname { I f } G _ { 1 }$ and $G _ { 2 }$ are uncolored graphs and $G = G _ { 1 } \boxtimes G _ { 2 }$ is their clique-sum, then mlt $( G ) = \operatorname* { m a x } \{ \operatorname* { m l t } ( G _ { 1 } )$ , mlt $\left( G _ { 2 } \right) \}$ and $\operatorname { g c r } ( G ) = \operatorname* { m a x } \{ \operatorname { g c r } ( G _ { 1 } ) , \operatorname* { g c r } ( G _ { 2 } ) \}$ . One can naturally extend the clique-sum definition [6, Definition 1.9] to colored graphs by requiring that the gluing is done along cliques with compatible coloring. However, in the colored case, ideals $I _ { G }$ are no longer monomial and squarefree, see Theorem 3.19, which results in the failure to interpret $\mathcal { V } ( I _ { G } )$ as a subspace arrangement and consequently the failure to interpret clique sums as linear joins.

Example 3.23. Consider colored 3-cycles $G _ { 1 }$ and $G _ { 2 }$ below and let ${ \cal G } = { \cal G } _ { 1 } \boxtimes { \cal G } _ { 2 }$ be their clique-sum along the 2-clique corresponding to the red vertices.

![](images/7457aafb0dc89a78d7951f8f6fa34e629cea49c7843a32575a1a1b428c1ab3dc.jpg)  
One can check that mlt $( G _ { i } ) = \operatorname { g c r } ( G _ { i } ) = 2 .$ for i = 1, 2, but mlt $( G ) = \operatorname { g c r } ( G ) = 3$

In the remaining part of this section we showcase how Theorem 3.21 can be applied to compute maximum likelihood thresholds for specific families of graphs.

Example 3.24. Let us apply the previous result to determine thresholds for the 4-cycle with equally colored vertices following the construction from Theorem 3.20. Observations $X _ { 1 } =$ $( 1 , 0 , 0 , 0 ) , X _ { 2 } = ( 0 , 0 , 1 , 0 ) \in \mathbb { R } ^ { 4 }$ are identified with polynomials $y _ { 1 } , y _ { 3 }$ in $R _ { 1 }$ and $\langle y _ { 1 } , y _ { 3 } \rangle _ { 2 } =$ $\langle y _ { 1 } ^ { 2 } , y _ { 1 } y _ { 2 } , y _ { 1 } y _ { 3 } , y _ { 1 } y _ { 4 } , y _ { 2 } y _ { 3 } , y _ { 3 } ^ { 2 } , y _ { 3 } y _ { 4 } \rangle$ ⟩. Since matrix

$$
{ \left[ \begin{array} { l } { R _ { 2 } } \\ { y _ { 1 } ^ { 2 } } \\ { y _ { 1 } y _ { 2 } } \\ { y _ { 2 } y _ { 3 } } \\ { y _ { 3 } y _ { 4 } } \\ { y _ { 1 } y _ { 4 } } \end{array} \right] } { \left[ \begin{array} { l l l l l l l l } { y _ { 1 } ^ { 2 } } & { y _ { 1 } y _ { 2 } } & { y _ { 1 } y _ { 3 } } & { y _ { 1 } y _ { 4 } } & { y _ { 2 } y _ { 3 } } & { y _ { 3 } ^ { 2 } } & { y _ { 3 } y _ { 4 } } \\ { 1 } & { 0 } & { 0 } & { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right] }
$$

is of rank 5, then $\langle y _ { 1 } , y _ { 3 } \rangle _ { 2 } = R _ { 2 }$ . Note that $\langle l \rangle _ { 2 } \neq R _ { 2 }$ for any $l \in R _ { 1 }$ because $4 = \dim \langle l \rangle _ { 2 } < 5$ Therefore, $\operatorname { g c r } ( G ) = 2$ . Moreover, matrices in $\langle X _ { 1 } \rangle _ { 2 } + \mathcal { L } _ { G } ^ { \perp }$ are of the form

$$
\left[ \begin{array} { c c c c } { { \lambda + \mu _ { 1 } + \mu _ { 2 } + \mu _ { 3 } } } & { { 0 } } & { { \mu _ { 4 } } } & { { 0 } } \\ { { 0 } } & { { - \mu _ { 1 } } } & { { 0 } } & { { \mu _ { 5 } } } \\ { { \mu _ { 4 } } } & { { 0 } } & { { - \mu _ { 2 } } } & { { 0 } } \\ { { 0 } } & { { \mu _ { 5 } } } & { { 0 } } & { { - \mu _ { 3 } } } \end{array} \right] .
$$

Note that for any $\mu _ { 5 } = \mu 4 = 0 , \mu _ { 1 } , \mu _ { 2 } , \mu _ { 3 } < 0$ and $\lambda + \mu _ { 1 } + \mu _ { 2 } + \mu _ { 3 } > 0$ , the resulting matrix is positive definite. Therefore, wmlt $( G ) = 1$

Proposition 3.25. If G is a colored graph with equally colored vertices, then wmlt(G) = 1.

Proof. Consider observation $X _ { 1 } = ( 1 , 0 , \ldots , 0 ) \in \mathbb { R } ^ { m }$ and describe a positive definite matrix in $\langle X _ { 1 } \rangle _ { 2 } + \mathcal { L } _ { G } ^ { \perp }$ , generalizing the construction in Theorem 3.24. □

We finish the section by providing a family of graphs with mlt $( G ) < \operatorname { g c r } ( G )$

Proposition 3.26. If G is a colored m-cycle with equally colored vertices. If all edges are colored diferently, then $\operatorname { g c r } ( G ) = 2$ . If m is even, then mlt $( G ) = 1$

Proof. Let G be a colored graph with all vertices equally colored and all edges diferently colored. Note that $d = m + 1$ and hence $m = \dim \langle x \rangle _ { 2 } <$ dim $R _ { 2 } = m + 1$ for any $x \in R _ { 1 }$ . Since $\langle x _ { 1 } , x _ { 2 } \rangle _ { 2 } = R _ { 2 }$ for $\begin{array} { r } { x _ { 1 } = \sum _ { i = 1 } ^ { \lfloor m / 2 \rfloor } y _ { i } } \end{array}$ and $\begin{array} { r } { x _ { 2 } = y _ { 1 } + \sum _ { \lfloor m / 2 \rfloor } ^ { m - 1 } y _ { i } } \end{array}$ , then $\operatorname { g c r } ( G ) = 2$ by Theorem 3.21. By Theorem 3.4, proving ml ${ \mathrm { \Omega } } _ { \mathrm { { ' } } } ( G ) = 1$ is equivalent to proving that there is no nonzero $K \in { \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m )$ that contains a generic column vector $( a _ { 1 } \ldots a _ { m } ) ^ { t }$ in its kernel. Any matrix K in ${ \mathcal { L } } _ { G } \cap \operatorname { P S D } ( m )$ satisfies $\lambda _ { 1 } \geq | \lambda _ { i } |$ . Rewrite the system of equations $K ( a _ { 1 } \dots a _ { m } ) ^ { t } = 0$ as $A \left( \lambda _ { 1 } \ldots \lambda _ { m + 1 } \right) ^ { t } = 0$ . If $a _ { 1 } \neq 0 , \left( \textstyle \sum _ { i } ^ { m } ( - 1 ) ^ { i + 1 } a _ { i } ^ { 2 } \right) \lambda _ { 1 } = 0$ for m even. Thus $\lambda _ { 1 } = 0$ and $K = 0$ is the only positive semidefinite matrix in $\mathcal { L } _ { G }$ □

If m is odd, then bringing the system $A \left( \lambda _ { 1 } \ldots \lambda _ { m + 1 } \right) ^ { t } = 0$ in the previous proof into row echelon form yields the equation $\begin{array} { r } { \left( a _ { 1 } ^ { 2 } + \sum _ { i = 2 } ^ { m } ( - 1 ) ^ { i } a _ { i } ^ { 2 } \right) \lambda _ { 1 } + 2 a _ { 1 } a _ { 2 } \lambda _ { 2 } = 0 } \end{array}$ . However, computational experiments suggest that the statement still holds for m odd.

Conjecture 3.27. Let G be a colored m-cycle with equally colored vertices. If m is odd, then $\operatorname { m l t } ( G ) = 1$

## 4 Computational topology approaches to ML thresholds

While symbolic approaches such as elimination ideals or Gr¨obner bases discussed in Section 3 give exact criteria, they scale poorly and are computationally infeasible beyond small graphs. Addressing these limitations requires numerical methods. Typically, ML thresholds would be obtained numerically via optimization algorithms, e.g. by computing the MLE of randomly generated sample covariance matrices of a fixed rank or by maximizing random linear functions over $\mathcal { K } _ { G } \cap \{ \mathrm { t r a c e } ( K ) = 1 \}$ [28]. Here, we provide an alternative computational approach and explore $\mathcal { C } _ { G }$ empirically using ideas from TDA on samples from the cone of suficient statistics.

Specifically, we develop code for sampling within the cone of suficient statistics and propose to analyze the resulting samples via TDA, with the goal of identifying rank patterns and local connectivity features related to likelihood feasibility. We perform a case study on small graphs, for which symbolic methods still remain feasible and for which the ground truth is known to test the feasibility of our approach. Our experiments, while promising, expose the geometric complexity of colored models and the obstacles that future computational approaches must address.

## 4.1 Topological data analysis

TDA is a novel and active field of research combining concepts from pure and applied mathematics to quantify ’shape’ in data [9, 17, 14, 36, 25]. The most prominent method in TDA is persistent homology [9, 17, 14, 36, 25], an automatic, robust, and interpretable technique that studies topological invariants across multiple scales. Persistent homology applied to point cloud data yields persistence barcodes that capture connectivity patterns in dimension k such as connected components $( k = 0 )$ , loops $( k = 1 )$ , and voids $\left( k = 2 \right)$ . When computed on annular neighbourhoods, persistent homology can detect points near intersections or boundaries in non-manifold data [27]. The resulting local persistence barcodes can be interpreted to reflect qualitative changes in the local geometry across rank strata. Here, we leverage this property to investigate whether projections of positive semi-definite matrices onto their suficient statistics lie close to the boundary of the cone of suficient statistics of positive definite matrices.

## 4.1.1 Persistent homology

Typical input data to persistent homology is a collection of points in $\mathbb { R } ^ { n }$ . This data is then used to construct a filtered simplicial complex that includes data points as its vertices. The most common choice of filtration is the Vietoris-Rips filtration [32].

Definition 4.1 (Vietoris-Rips filtration $\nu \mathcal { R } _ { P } ^ { \bullet } )$ . Let $P$ be a point cloud, d be a distance function on $P _ { \mathrm { : } }$ and $\varepsilon \in \mathbb { R } _ { 0 } ^ { + }$ . The Vietoris-Rips complex at parameter ε, denoted $\mathcal { V R } _ { { P } } ^ { \varepsilon }$ , is a simplicial complex that has the points in P as its vertex set and includes the n-simplex ${ \boldsymbol { \tau } } = ( p _ { 0 } , \ldots , p _ { n } )$ if $d ( p _ { i } , p _ { j } ) \leq \varepsilon$ for all $i , j \in \{ 1 , \ldots , n \}$ with $i \neq j$ . A Vietoris-Rips filtration $\nu \mathcal { R } _ { P } ^ { \bullet }$ is a nested sequence of simplicial complexes $\mathcal { V R } _ { { P } } ^ { \varepsilon }$ obtained by considering a sequence of increasing ε.

For a given filtration and a fixed field F, the k-dimensional persistence module is the sequence of F-vector spaces together with linear maps induced by inclusions in the filtration:

Definition 4.2 (k-dimensional persistence module $P H _ { k } ( X ^ { \bullet } ) )$ . For a filtered simplicial complex

$$
T ^ { \bullet } = T ^ { 1 } \stackrel { \iota ^ { 1 } } { \longleftrightarrow } T ^ { 2 } \stackrel { \iota ^ { 2 } } { \longleftrightarrow } \cdots \stackrel { \iota ^ { N - 2 } } { \longleftrightarrow } T ^ { N - 1 } \stackrel { \iota ^ { N - 1 } } { \longleftrightarrow } T ^ { N } ,
$$

the k-dimensional persistence module is defined by

$$
P H _ { k } ( T ^ { \bullet } ) = H _ { k } ( T ^ { 1 } ; \mathbb { F } ) { \xrightarrow { \phi ^ { 1 } } } H _ { k } ( T ^ { 2 } ; \mathbb { F } ) { \xrightarrow { \phi ^ { 2 } } } \cdots { \xrightarrow { \phi ^ { N - 1 } } } H _ { k } ( T ^ { N } ; \mathbb { F } ) ,
$$

where $\phi ^ { \varepsilon }$ are the maps induced by $\iota ^ { \varepsilon }$ and $H _ { k } ( X ^ { i } ; \mathbb { F } )$ is the k-th homology group of $T ^ { i }$ over $\mathbb { F } .$

Here, we work with $\mathbb { F } = \mathbb { Z } / 2 \mathbb { Z }$ . The structure theorem of persistent homology provides a decomposition of the k-dimensional persistence module into a unique persistence barcode [36] which is stable to small perturbations of the points in P [10, 12]. Persistent homology refers to the process of constructing a persistence module from data, whose structure is then typically summarised by its persistence barcode.

## 4.1.2 Understanding existence of MLE using persistent homology of annular neighbourhoods

For finite samples from intersecting manifolds, persistent homology computed on annular neighbourhoods can detect singular regions even when none of the data points were sampled precisely from these singularities [27]. The strategy relies on the fact that for n-dimensional spheres, the dimension of the i-th homology group is equal to 1 if $i = n$ and 0 otherwise. For fixed real parameters $0 < r < s$ , we compute the annular neighbourhood $A _ { x }$ of each point $x \in P$ , i.e. all points $y \in P$ such that $r \leq d ( x , y ) \leq s$ . We then compute the $k { \mathrm { - t h } }$ persistent homology of the Vietoris-Rips filtration $\nu \mathcal { R } _ { A _ { x } } ^ { \bullet }$ . If the barcode does not approximate a single k-sphere, the point x lies close to a singular region.

We propose Algorithm 2 for MLT estimation using persistent homology on annular neighbourhoods and perform a case study on $G _ { 3 } , G _ { 6 }$ and $G _ { 1 8 }$ from Table 1. For a fixed sample size $t ,$ we set $R _ { 2 } = 1 . 3 \times$ median distance among points in D and $R _ { 1 } = 7 R _ { 2 } / 8$ , which have shown the best performance in our experiments. We exclude points that have less than 4 neighbours in their annular neighbourhoods. For $G _ { 3 }$ we restricted ourselves to computing persistent homology in dimension 1 since after the compactification with the trace constraint, the ambient space has dimension 2. In $G _ { 6 } .$ , we used both dimension 1 and 2 persistent homology as the cone of suficient statistics is 4-dimensional, but can be further reduced to $3$ dimensions as for $G _ { 3 }$ . For $G _ { 1 8 }$ , where the cone of suficient statistics is high-dimensional, we restrict computations to 2-dimensional homology due to the sampling limitations discussed below. To retain computational eficiency, we do not compute persistent homology beyond dimension 2.

Intuitively, to check whether MLE exists for a matrix $S \in \mathrm { P S D } ( m , n )$ , with $n < m$ , we approximate the interior of the cone of suficient statistics $\pi _ { G } ( \mathcal { C } _ { G } )$ by uniformly sampling a

$$
S _ { \mathrm { P D } } \subseteq \mathrm { P D } ( m )
$$

set (following [24]). We then compute the persistent homology of an annular   
neighbourhood of $\pi _ { G } ( S )$ in $\pi _ { G } ( S _ { \mathrm { P D } } )$ . The annular neighbourhood thereby gives us a discrete   
proxy of the boundary of the neighbourhood of $\pi _ { G } ( S )$ in the cone of suficient statistics. If the   
barcode resulting from the persistent homology calculation reveals an annular neighbourhood   
that resembling k-sphere, then we conclude that $\pi _ { G } ( S )$ lies in the interior of the cone of suficient   
statistics (see Algorithm 3).   
Algorithm 2: MLT estimation using persistent homology   
Data: G - undirected graph, n - candidate threshold, $\overline { { N _ { n } } }$ - number of n-tuples of   
observations, M - number of samples from $\mathrm { P D } ( m )$ with trace 1   
Result: $N _ { n , \mathrm { i n t } } -$ the number of n-tuples of observations whose suficient statistics lie in   
the interior of $\mathcal { C } _ { G }$   
begin   
$N _ { n , \mathrm { i n t } } \longleftarrow 0 ;$   
$D  \bigcup ;$   
Compute $S _ { \mathrm { P D } } \subseteq \mathrm { P D } ( m )$ - the i.i.d. set of positive definite matrices with trace 1   
sampled uniformly [24];   
Uniformly sample $\mathcal { X } = \{ X ^ { ( 1 ) } , \ldots , X ^ { ( N _ { n } ) } \}$ , where   
$X ^ { ( i ) } = \{ X ^ { ( i , 1 ) } , \ldots , X ^ { ( i , \overset { } { n } ) } \} \subseteq [ - 1 , 1 ] ^ { m } ;$   
for $X ^ { ( i ) } \in \mathcal { X }$ do   
$S ^ { ( i ) } = X ^ { ( i ) } ( X ^ { ( i ) } ) ^ { T } ;$   
$S ^ { ( i ) } = \frac { 1 } { \operatorname { t r a c e } ( S ^ { ( i ) } ) } S ^ { ( i ) } ;$   
$D  D \cup \{ \pi _ { G } ( S ^ { ( i ) } ) \}$   
Compute $D _ { \mathrm { i n t } }$ with Algorithm $s ;$   
$N _ { n , \mathrm { i n t } } \longleftarrow | D _ { \mathrm { i n t } } | ;$   
return $N _ { t , \mathrm { i n t } }$   
Algorithm 3: Persistent homology of annular regions of suficient statistics   
Data: Finite point set $\overline { { D \subset \mathbb { R } ^ { n } } }$ , real parameters $\overline { { 0 < R _ { 1 } < R _ { 2 } } }$   
Result: Partition of D into boundary points $D _ { \mathrm { b d r y } }$ and interior points $D _ { \mathrm { i n t } }$   
begin   
$D _ { \mathrm { i n t } } \longleftarrow \infty , D _ { \mathrm { b d r y } } \longleftarrow \infty ;$   
for $y \in D$ do   
Find $A _ { y } = \{ x \in D$ satisfying $R _ { 1 } \leq \| x - y \| \leq R _ { 2 } \}$   
if $| A _ { y } | \geq 1 0 0$ then   
Choose 100 points from $A _ { y }$ using the sequential maxmin algorithm;   
Compute $P H _ { 1 } ( A _ { y } )$ , the 1-dim barcode of $A _ { y }$ and $P H _ { 2 } ( A _ { y } )$ , the 2-dim   
barcode of $A _ { y } ;$   
Calculate $N _ { 1 , y } = \# \{ [ a , b ) \in P H _ { 1 } ( A _ { y } )$ with $( b - a ) > ( R _ { 2 } - R _ { 1 } ) \}$ and   
$N _ { 2 , y } = \# \{ [ a , b ) \in P H _ { 2 } ( A _ { y } )$ with $( b - a ) > ( R _ { 2 } - R _ { 1 } ) \}$ ;   
if $N _ { 2 , y } \geq 1$ then   
$D _ { \mathrm { i n t } }  D _ { \mathrm { i n t } } \cup \{ y \}$   
else if $N _ { 1 , y } \ge 1$ then   
$D _ { \mathrm { i n t } } \longleftarrow D _ { \mathrm { i n t } } \cup \{ y \}$   
else   
$D _ { \mathrm { b d r y } } \longleftarrow D _ { \mathrm { b d r y } } \cup \{ y \}$   
return $( D _ { i n t } , D _ { b d r y } )$   
For $G _ { 1 8 }$ (see Figure 4), we find that our approach reports the most accurate results with   
the percentage of boundary points being almost 1 for one observation (MLE does not exist),   
below 0.3 for two observations (MLE exists with probability $p \in ( 0 , 1 ) )$ , and below 0.1 for 3

observations (MLE exists with $p = 1 )$ . This corresponds to wmlt $( G _ { 1 8 } ) = 2 , \mathrm { m l t } ( G _ { 1 8 } ) = 3$ . For $G _ { 3 }$ and $G _ { 6 }$ , where the MLE exists with $p = 1$ for 1,2, or 3 observations, we find that the reported percentage of boundary points remains below 0.2 and 0.4, respectively, across all sample sizes, corresponding to wmlt $( G _ { 3 } ) = \mathrm { m l t } ( G _ { 3 } ) = 1$ and wmlt $( G _ { 6 } ) = \mathrm { m l t } ( G _ { 6 } ) = 1$ respectively. Overall, our approach appears to successfully capture relative trends but does not provide suficiently sharp decision boundaries to draw definitive conclusions.

![](images/f71f14f1261892a0f6410c482f075d3b9b74f2c84963a53e2ed0be881dbd8cc5.jpg)

![](images/ebfc915c902dcc29a9ba62b8145bf2673b9655d0af5b0d2f2a68b161436d1701.jpg)

![](images/9e67d55377d251614da73910577e9e4826572dde313628a809ba3777d6c07153.jpg)

![](images/8fd5e2ebffdf59b9b88b591c40022c998ddb71d668480b93e6a26512cc296844.jpg)

![](images/923975a17b0aa6069c3530f597bf69dfed3e8a42556d33ecc2dd9b4f7d300788.jpg)

![](images/9757779729bff7a088922cd41ece595d21bacd45e04eb1b2dc4a1cfdaf1a9058.jpg)  
Figure 4: From left to right, percentage of boundary points in $G _ { 3 } , \ G _ { 6 }$ and $G _ { 1 8 }$ (uncolored 4-cycle). We determine the boundary points based on persistent homology computations in dimension 1 for $G _ { 3 } ,$ , dimensions 1 and 2 for $G _ { 6 } .$ , and dim 2 for $G _ { 1 8 }$

When $A _ { X X ^ { T } } = \pi _ { G } ( { \cal S } _ { \mathrm { P D } } ) \cap N ( \pi _ { G } ( X X ^ { T } ) , R _ { 1 } , R _ { 2 } )$ is not suficiently dense, Algorithm 2 can produce false results. On the one hand, if there are insuficiently many points, Algorithm 3 may be unable to build a k-sphere around an interior point, which results in misclassification of an interior point as a boundary point. On the other hand, when the sampling is not dense enough to create a contractible disk, it can classify a boundary point as an interior point. We show potential misclassifications due to sparsity in Fig. 5.

## 4.2 Sampling the cone of suficient statistics

Local persistence homology methods are prone to misclassification of boundary and interior points if data is not dense enough in each annular region, as shown in Figure 5. Therefore, it is fundamental to understand how to sample points in the cone of suficient statistics $\mathcal { C } _ { G }$

Given a sample $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$ with $n < m$ , we call the suficient statistic of its sample covariance matrix $S$ the target statistic $\pi _ { G } ( S )$ . We investigated global sampling (across $[ - 1 , 1 ] ^ { n \times m }$ , the positive definite cone $\mathrm { P D } ( m )$ and Cholesky space) and local sampling around the target statistic via perturbations of $X _ { 1 } , \ldots , X _ { n } \in \mathbb { R } ^ { m }$ , S or Cholesky decompositions of S. Our global experiments reveal a dichotomy depending on the sampling strategy: generating observations uniformly causes sample covariance projections to concentrate along the boundary $\partial \mathcal { C } _ { G }$ , while uniform sampling across the positive definite cone $\mathrm { P D } ( m )$ does the exact opposite, clustering near the center and leaving the boundary sparse, see Figures 10–15. At the same time, local sampling via perturbations, such as adding $m - n$ observations of small magnitude to the data sample, also sufers from the clustering bias towards the center.

The main challenge of the naive sampling methods is that even in small graphs, the pre-image of suficient statistics lies in a high-dimensional space and for a given topologically representative neighborhood of $\pi _ { G } ( S )$ , its pre-image lies in a small portion of the space. We illustrate this for 4-cycle $G _ { 3 }$ and parametrization via Cholesky decomposition in Figure 6. Here, the interior of the cone $\mathcal { C } _ { G }$ was obtained by first sampling uniformly from $\mathrm { P D } ( 4 )$ with trace 1. We choose a $\Sigma _ { 0 }$ from this sample such that its projection is centrally positioned in $\mathcal { C } _ { G } .$ We study the Cholesky decompositions of the sample points whose projections are in the neighborhood of the target statistic and put them on the $\{ - \infty , 0 , \infty \} ^ { 1 0 }$ grid of deviations from the Cholesky decomposition of $\Sigma _ { 0 }$ . The histogram of counts of pre-images of sample points in the grid cells (Figure 16 in Section A.3) shows that the majority of cells have close to 0 observations and there are individual cells in the tail of the distribution with large number of pre-images of sample points. In this case, sampling only from the Cholesky grid cell with the maximum number of pre-images already provides a topologically representative neighborhood, as seen in Figure 7.

![](images/f93af201116e1345f296ecbd67965e86a232a10f270077ae327ec59ab5ccdc58.jpg)  
Figure 5: Possible classifications of interior points and boundary points. We show an example of possible classification of points sampled from a surface. Sparsity is a source of misclassifications for both interior and boundary points.

![](images/4f91be538a3194b631f32ade867d855ac6561e89539731daf92347309618c34d.jpg)  
Figure 6: Neighborhoods of the reference, intermediate and target statistics in the cone of suficient statistics for $G _ { 3 }$

![](images/3d2cc3b1887e5b60ae84322f1e6546bdc3de80c9c085afac3d55f82142b52de4.jpg)  
Figure 7: Synthetic neighborhoods of the target and intermediate statistics fixed in Fig. 6 for $G _ { 3 }$

This section introduces a method based on finding a relevant region in the Cholesky space to generate synthetic points that represent the local topology near the boundary of the cone. To ensure the uniqueness of Cholesky decomposition, we approximate S by suficiently close $\widetilde { \Sigma } \in \mathrm { P D } ( m )$ . Let $\mathcal { S } \subset \mathrm { P D } ( m )$ be a uniform sample from $\mathrm { P D } ( m )$ with trace 1 acquired using the algorithm in [24]. We choose a reference statistic $\pi _ { G } ( \Sigma _ { 0 } )$ with $\Sigma _ { 0 } \in S$ and $\pi _ { G } ( \Sigma _ { 0 } )$ centrally positioned in $\pi _ { G } ( \cal S )$ . This ensures the existence of dense sample neighborhood of $\pi _ { G } ( \Sigma _ { 0 } )$ though the local geometry of its Cholesky decomposition does not reflect the local geometry near the Cholesky decomposition of $\widetilde { \Sigma }$

Consider the segment $p ( \lambda ) = ( 1 - \lambda ) \pi _ { G } ( \Sigma _ { 0 } ) + \lambda \pi _ { G } ( \widetilde { \Sigma } )$ with $\lambda \in [ 0 , 1 ]$ . Along this path we fix $0 < \lambda ^ { * } < 1$ and use the corresponding intermediate statistic $\pi _ { G } ( \Sigma _ { \mathrm { m i d } } )$ with $\Sigma _ { \mathrm { m i d } } \in { S }$ such that $d ( p ( \lambda ^ { * } ) , \pi _ { G } ( \Sigma _ { \mathrm { m i d } } ) ) = \mathrm { m i n } _ { \Sigma \in { \cal S } } d ( p ( \lambda ^ { * } ) , \pi _ { G } ( \Sigma ) )$ . The goal is to identify the intermediate statistic whose projected neighborhood has enough points in $\pi _ { G } ( \cal S )$ to estimate local Cholesky behavior, and lies suficiently close to $\pi _ { G } ( \widetilde { \Sigma } )$ that it resembles the local behavior of the Cholesky decomposition of $\widetilde { \Sigma }$ , as illustrated in Figure 6.

We start by identifying the most populated Cholesky grid cells of the intermediate statistic following Algorithm 4. Let $\ell _ { \mathrm { m i d } }$ be the flattened vector of the Cholesky factor of $\Sigma _ { \mathrm { m i d } }$ . For each PD matrix $\Sigma _ { j } \in S$ whose suficient statistic $\pi _ { G } ( \Sigma _ { j } )$ is in the neighborhood of $\pi _ { G } ( \Sigma _ { \mathrm { m i d } } )$ we flatten its Cholesky factor to a vector $\ell _ { j }$ and compute deviations $d _ { j } = \ell _ { j } - \ell _ { \mathrm { m i d } }$ . Binning these deviations into the grid with edges $\{ - \infty , 0 , \infty \} ^ { m ( m + 1 ) / 2 }$ identifies a small set of relevant Cholesky coordinates.

Once the populated Cholesky grid cells have been identified around $\Sigma _ { \mathrm { m i d } }$ , we construct a synthetic neighborhood by sampling Cholesky coordinates uniformly within the corresponding grid cells with Algorithm 5. Each synthetic Cholesky factor is converted into a positive-definite matrix $\widehat { \Sigma }$ , with positivity ensured by truncating the lower bounds of the diagonal coordinates, and mapped to the corresponding suficient statistic $\widehat { \sigma } = \pi _ { G } ( \widehat { \Sigma } )$

Since the intermediate statistic is selected along the line segment connecting the reference statistic and the positive-definite approximation $\widetilde { \Sigma }$ of the target matrix, the resulting synthetic statistics provide an approximation of the local neighborhood of the target statistic. This synthetic neighborhood captures the local topological structure more faithfully than the neighborhood obtained directly from empirical covariance matrices, see Figure 7.

To evaluate the algorithms, we sampled 1000 PSD matrices derived from 1, 2, and 3 observations for the graphs in [30, Table 2]. As shown in Figure $^ { 8 , }$ synthetic neighborhoods are better at representing the local geometry of suficient statistics whenever the graph has more than a single vertex color and the performance generally improves in the number of observations. At the same time, Figure 9 shows that the performance of Algorithms 4 and 5 is sensitive to the number of color classes and the size of the data sample. Since the quality of the output of Algorithm 4 depends largely on the ability to sample a dense representation of the PD cone, an approach to address this limitation in applications is to perform the computationally intense generation of the representation of the PD cone in a centralized environment with high computational power and make it available for distributed environments with limited computational power as input to local computation with Algorithm 5.

Algorithm 4: Construction of Cholesky grid cells around an intermediate statistic   
Data: $G \mathrm { ~ - ~ }$ colored graph on $[ m ] ;$   
$( X _ { 1 } , \ldots , X _ { n } )$ , with $X _ { i } \in \mathbb { R } ^ { m }$ and $n < m -$ observations whose suficient statistic $\pi _ { G } ( S )$   
is the target statistic, where $S$ is the sample covariance matrix;   
$\mathcal { S } \subset \mathrm { P D } ( m ) \textrm { - a }$ uniform sample of positive-definite matrices with trace 1;   
$\pi _ { G } ( \Sigma _ { 0 } ) - \mathrm { a }$ reference statistic that is centrally positioned in $\pi _ { G } ( \cal S )$ , with $\Sigma _ { 0 } \in S ;$   
$\varepsilon > 0 -$ tolerance for the positive-definite approximation of the target matrix;   
Result: The intermediate Cholesky vector $\ell _ { \mathrm { m i d } }$ and a collection $\mathcal { C }$ of populated   
Cholesky grid cells   
begin   
Choose $\widetilde { \Sigma } \in \mathrm { P D } ( m )$ such that $\| \widetilde { \Sigma } - S \| < \varepsilon ,$ , where $\widetilde { \Sigma }$ is a PD approximation of the   
target PSD matrix $S ;$   
Construct the line segment   
$p ( \lambda ) = ( 1 - \lambda ) \pi _ { G } ( \Sigma _ { 0 } ) + \lambda \pi _ { G } ( \widetilde { \Sigma } ) , \qquad \lambda \in [ 0 , 1 ] ;$   
Choose $\lambda ^ { * } \in [ 0 , 1 ]$ such that $p ( \lambda ^ { * } )$ has a suficiently dense neighborhood $\mathcal { N } ^ { * }$ in   
$\pi _ { G } ( \cal { S } )$ );   
Identify the intermediate matrix $\Sigma _ { \mathrm { m i d } } \in S$ by   
$\Sigma _ { \mathrm { m i d } } \in \arg \operatorname* { m i n } _ { \Sigma \in S } d ( p ( \lambda ^ { * } ) , \pi _ { G } ( \Sigma ) )$   
Extract all matrices $\Sigma _ { j } \in S$ such that   
$\pi _ { G } ( \Sigma _ { j } ) \in \mathcal N ^ { * } ;$   
Compute the upper-triangular Cholesky factor $U _ { \mathrm { m i d } }$ of $\Sigma _ { \mathrm { m i d } }$ and flatten its   
upper-triangular entries into   
$\ell _ { \mathrm { m i d } } \in \mathbb { R } ^ { q } , \qquad q = \frac { m ( m + 1 ) } { 2 } ;$   
foreach extracted matrix $\Sigma _ { j }$ do   
Compute its upper-triangular Cholesky factor $U _ { j } { \mathrm { : } }$   
Flatten the upper-triangular entries of $U _ { j }$ into $\dot { \ell _ { j } } \in \mathbb { R } ^ { q }$   
Compute the deviation   
$d _ { j } = \ell _ { j } - \ell _ { \mathrm { m i d } } ;$   
Assign $d _ { j }$ to a grid cell $g _ { j } = ( g _ { j , 1 } , . . . , g _ { j , q } ) \in \{ - , + \} ^ { q } .$ , where   
$g _ { j , k } = \left\{ { \displaystyle - , } \atop { \displaystyle d _ { j , k } \geq 0 }  \right\}$   
Identify a collection C of suficiently populated Cholesky grid cells;   
return $\ell _ { \mathrm { m i d } } , \mathcal { C }$

Algorithm 5: Synthetic neighborhood generation from Cholesky grid cells   
Data: Flattened Cholesky factor $\overline { { \ell _ { \mathrm { m i d } } \in \mathbb { R } ^ { q } } }$ of the intermediate matrix, where   
$q = m ( m + 1 ) / 2 ;$   
Selected Cholesky grid cells ${ \mathcal { C } } \subseteq \{ - , + \} ^ { q } ;$   
Perturbation scale $\delta ;$   
Number of synthetic samples per grid cell $N _ { \mathrm { c e l l } } ;$   
Result: Synthetic neighborhood $\widehat { \mathcal { N } }$ approximating the local topology of the target   
statistic   
begin   
Initialize $\begin{array} { r } { \widehat { N }  \emptyset ; } \end{array}$   
foreach $g \in { \mathcal { C } }$ do   
for $r = 1$ to $N _ { \mathrm { c e l l } }$ do   
for $k = 1$ to $q$ do   
Sample $\widehat { \ell } _ { k } ^ { \left( g , r \right) }$ uniformly from the half-interval determined by $g _ { k } ;$ if k   
corresponds to a diagonal Cholesky entry, truncate the lower endpoint   
to $1 0 ^ { - 6 } ;$   
Form the upper-triangular Cholesky factor $\widehat { U } ^ { ( g , r ) }$ from $\widehat { \ell } ^ { ( g , r ) }$ ;   
Compute $\widehat { \Sigma } ^ { ( g , r ) } = ( \widehat { U } ^ { ( g , r ) } ) ^ { \mathsf { T } } \widehat { U } ^ { ( g , r ) }$ and rescale it to have trace 1;   
Compute the synthetic statistic ${ \widehat { \sigma } } ^ { ( g , r ) } = \pi _ { G } ( { \widehat { \Sigma } } ^ { ( g , r ) } )$ ;   
Add ${ \widehat { \boldsymbol { \sigma } } } ^ { ( g , r ) }$ to ${ \widehat { N } } ;$   
return $\widehat { \mathcal { N } }$

![](images/6622b44b8957438f7b09081f681164dfceb1eec70553cd2d5f5dc73048942d93.jpg)  
Figure 8: Comparison of original and synthetic neighbourhoods of suficient statistics of PSD matrices for the graphs in [30, Table 2].

![](images/51e729c010bef40c29a0561d47955c661de627d5cc5f655aaf886d9cd4f0164d.jpg)  
Figure 9: Empirical probability $p$ of the existence of MLE for the 4-cycles in [30, Table 2], ranging from 3 to 8 color classes. The sample used to compute $p$ has been obtained with Algorithm 5. MLE existence is determined using convex hull containment of the target statistic in the synthetic sample.

## 5 Discussion

Eficient computation of the maximum likelihood threshold $( { \mathrm { m l t } } ( G ) )$ for a given CGGM currently remains an open challenge. However, the generic completion rank $( \mathrm { g c r } ( G ) )$ can be computed in random polynomial time with an adapted version of [3, Algorithm 3.2]. While Bernstein et al. [4] conjecture that $\operatorname { g c r } ( G ) = \operatorname { m l t } ( G )$ for almost all uncolored G, we are still far from understanding the broader landscape in the colored setting.

A first analysis shows that for colored graphs with three vertices, the equality $\operatorname { g c r } ( G ) =$ mlt(G) always holds, and mlt $( G ) = \operatorname { w m l t } ( G )$ in 19 out of the 25 graphs. This raises fundamental questions:

• How often do we have $\operatorname { g c r } ( G ) = \operatorname { m l t } ( G ) ?$

• Can these relationships be characterized in terms of the coloring of the graph?

Regardless of these considerations, there exist suficient instances with mlt $( G ) < \operatorname { g c r } ( G )$ to justify pursuing methods for computing the MLT directly. Here, we explored computational ideas from algebraic geometry and TDA to address these questions.

From the perspective of algebraic geometry, we focused on two approaches:

1. Computing the vanishing ideal of the projection map. This approach is conceptually powerful: obtaining the vanishing ideal provides complete algebraic information about the model. However, its scalability is limited. The main bottlenecks are (1) Gr¨obner basis computations required to determine the ideal, and (2) the implementation of positive certificates via sums of squares (SOS).

• Direction for future work 1. Investigate numerical methods for computing the vanishing ideal using witness sets.

• Direction for future work 2. Explore alternative positivity certificates beyond SOS and evaluate their computational feasibility.

2. Solving matrix completion problems. This method can be pursued either by directly finding positive definite completions (Theorem 3.21), or by using the equivalent characterization based on kernels (Theorem 3.4). This perspective provides powerful tools for computing mlt(G) in specific graph families and is often more tractable when computing wmlt(G). Nonetheless, general-purpose algorithms for this approach have yet to be developed.

• Direction for future work 3. Exploit Theorem 3.21 and Theorem 3.4 to design algorithms for computing mlt(G) and wmlt(G).

From the TDA perspective, we investigated whether persistent homology of annular neighbourhoods can be used to detect points close to the boundary of the cone of suficient statistics. We find that this approach is promising, especially as a way to mitigate the computational challenges faced by algebraic-geometry methods, but the results depend strongly on the sampling method, which must be suficiently uniform. We propose an algorithm for local uniformization of the near-boundary suficient statistics.

However, the method only captures relative trends but does not provide suficiently sharp decision boundaries.

• Direction for future work 4. Investigate alternative methods for neighborhood construction that allow for variable density in samples, e.g. dynamic choice of radius, k-NN algorithms, methods from manifold learning theory [23].

• Direction for future work 5. Investigate the theoretical properties of the distribution of active Cholesky coordinates to enable more eficient sampling.

• Direction for future work 6. Generalize algorithm in [24] to uniformly sample from PSD(m) with trace constraint and understand theoretical properties of projections onto the cone of suficient statistics $\mathcal { C } _ { G }$ of uniform distributions in trace-constrained PD(m) and PSD(m).

## 6 Code and data availability

All TDA analyses were implemented in Matlab using ripser [2] for persistent homology computation. All algebraic geometry computations and algorithms have been performed and implemented in Macaulay2 [18], and are build on the packages GraphicalModels [16], GraphicalModelsMLE [1] and SumsOfSquares [11]. Source code and experimental data are made available on https:// www.dropbox.com/scl/fo/5n4kiff0xv63uw49n3jni/ADR5R9AoOGPUVMazqB9qpb4?rlkey=074sr1m93kdjk6p6 st=an67mvgo&dl=0.

## 7 Acknowledgments

The authors express their gratitude to the organizers of Biomat 2023: Multiscale Methods at the Frontier Between Data and Mathematical Models, held at the Centre de Recerca Matem\`atica (CRM). In particular, the course on TDA delivered by the third author led to expanding the initial work of the first two authors into this joint collaboration.

R.H. was supported by the postdoctoral fellowship programme Beatriu de Pin´os (ref. 2021BP00119), funded by the Secretary of Universities and Research (Government of Catalonia) and partially supported by projects with references PID2019-103849G-I00 and PID2023-146936NB-I00, financed by the Spanish State Agency, MCIN/AEI/10.13039/501100011033/ FEDER,UE, and by the AGAUR project 2021 SGR 00603 Geometry of Manifolds and Applications, GEOMVAP.

## References

[1] Carlos Am´endola, Luis David Garc´ıa-Puente, Roser Homs, Olga Kuznetsova, and Harshit J. Motwani. Computing maximum likelihood estimates for gaussian graphical models with macaulay2. Journal of Software for Algebra and Geometry, 12:1–10, 2022. doi:10.2140/ jsag.2022.12.1.

[2] Ulrich Bauer. Ripser: eficient computation of Vietoris–Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3):391–423, 2021. doi:10.1007/ s41468-021-00071-5.

[3] Daniel Irving Bernstein, Grigoriy Blekherman, and Rainer Sinn. Typical and generic ranks in matrix completion. Linear Algebra and its Applications, 585:71–104, 2020. doi:10. 1016/j.laa.2019.09.001.

[4] Daniel Irving Bernstein, Sean Dewar, Steven J. Gortler, Anthony Nixon, Meera Sitharam, and Louis Theran. Computing maximum likelihood thresholds using graph rigidity. Algebraic Statistics, 14(2):287–305, 2022. URL: 10.2140/astat.2023.14.287, doi:10.2140/ astat.2023.14.287.

[5] Daniel Irving Bernstein, Sean Dewar, Steven J. Gortler, Anthony Nixon, Meera Sitharam, and Louis Theran. Maximum likelihood thresholds via graph rigidity. Annals of Applied Probability, 3(34):3288–3319, 2024. doi:10.1214/23-AAP2039.

[6] Grigoriy Blekherman and Rainer Sinn. Maximum likelihood threshold and generic completion rank of graphs. Discrete & Computational Geometry, 61:303–324, 2019. doi: 10.1007/s00454-018-9990-3.

[7] Stephen Boyd, Stephen P Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004. doi:10.1017/CBO9780511804441.

[8] Søren L. Buhl. On the existence of maximum likelihood estimators for graphical gaussian models. Scandinavian Journal of Statistics, 20(3):263–270, 1993.

[9] Gunnar E. Carlsson. Topology and data. Bulletin of the American Mathematical Society, 46:255–308, 2009. doi:10.1090/S0273-0979-09-01249-X.

[10] Fr´ed´eric Chazal, David Cohen-Steiner, Marc Glisse, Leonidas J. Guibas, and Steve Y. Oudot. Proximity of persistence modules and their diagrams. In Proceedings of the Twenty-Fifth Annual Symposium on Computational Geometry, SCG ’09, page 237–246, New York, NY, USA, 2009. Association for Computing Machinery. doi:10.1145/1542362.1542407.

[11] Diego Cifuentes, Thomas Kahle Kahle, and Pablo Parrilo. Sums of squares in macaulay2. Journal of Software for Algebra and Geometry, 10:17–24, 2020. doi:10.2140/jsag.2020. 10.17.

[12] David Cohen-Steiner, Herbert Edelsbrunner, and John Harer. Stability of persistence diagrams. In Proceedings of the twenty-first annual symposium on Computational geometry, pages 263–271, 2005. doi:10.1145/1064092.1064133.

[13] Arthur P Dempster. Covariance selection. Biometrics, pages 157–175, 1972. doi:10.2307/ 2528966.

[14] Edelsbrunner, Letscher, and Zomorodian. Topological persistence and simplification. Discrete & computational geometry, 28(4):511–533, 2002. doi:10.1007/s00454-002-2885-2.

[15] Peter GM Forbes and Stefen Lauritzen. Linear estimating equations for exponential families with application to gaussian linear concentration models. Linear Algebra and its Applications, 473:261–283, 2015. doi:10.1016/j.laa.2014.08.015.

[16] Luis Garc´ıa-Puente, Sonja Petrovi´c, and Seth Sullivant. Graphical models. Journal of Software for Algebra and Geometry, 5(1):1–7, 2013. doi:10.2140/jsag.2013.5.1.

[17] Robert Ghrist. Barcodes: The persistent topology of data. Bulletin of The American Mathematical Society, 45, 02 2008. doi:10.1090/S0273-0979-07-01191-3.

[18] Daniel R. Grayson and Michael E. Stillman. Macaulay2, a software system for research in algebraic geometry. Available at http://www2.macaulay2.com.

[19] Elizabeth Gross and Seth Sullivant. The maximum likelihood threshold of a graph. Bernoulli, 24(1):386–407, 2018. doi:10.3150/16-BEJ881.

[20] Søren Højsgaard and Stefen L. Lauritzen. Graphical gaussian models with edge and vertex symmetries. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 70(5):1005–1027, 2008. doi:10.1111/j.1467-9868.2008.00666.x.

[21] Jan Krumsiek, Karsten Suhre, Thomas Illig, Jerzy Adamski, and Fabian J Theis. Gaussian graphical modeling reconstructs pathway reactions from high-throughput metabolomics data. BMC systems biology, 5(1):21, 2011. doi:10.1186/1752-0509-5-21.

[22] Marloes Maathuis, Mathias Drton, Stefen Lauritzen, and Martin Wainwright. Handbook of graphical models. CRC Press, 2018. doi:10.1201/9780429463976.

[23] Marina Meil˘a and Hanyu Zhang. Manifold learning: What, how, and why. Annual Review of Statistics and Its Application, 11(Volume 11, 2024):393–417, 2024. doi: 10.1146/annurev-statistics-040522-115238.

[24] Martin Mittelbach, Bho Matthiesen, and Eduard A Jorswieck. Sampling uniformly from the set of positive definite matrices with trace constraint. IEEE Transactions on Signal Processing, 60(5):2167–2179, 2012. doi:10.1109/TSP.2012.2186447.

[25] Vanessa Robins. Towards computing homology from finite approximations. In Topology proceedings, volume 24, pages 503–532, 1999.

[26] Katherine H Shutta, Deborah Weighill, Rebekka Burkholz, Marouen Ben Guebila, Dawn L DeMeo, Helena U Zacharias, John Quackenbush, and Michael Altenbuchinger. Dragon: determining regulatory associations using graphical models on multi-omic networks. Nucleic Acids Research, 51(3):e15–e15, 2023. doi:10.1093/nar/gkac1157.

[27] Bernadette J Stolz, Jared Tanner, Heather A Harrington, and Vidit Nanda. Geometric anomaly detection in data. Proceedings of the national academy of sciences, 117(33):19664– 19669, 2020. doi:10.1073/pnas.2001741117.

[28] Bernd Sturmfels and Caroline Uhler. Multivariate gaussians, semidefinite matrix completion, and convex algebraic geometry. Ann Inst Stat Math, 62:603–638, 2010. doi: 10.1007/s10463-010-0295-4.

[29] Seth Sullivant. Algebraic statistics, volume 194. American Mathematical Soc., 2018. doi: 10.1090/gsm/194.

[30] Caroline Uhler. Geometry of maximum likelihood estimation in Gaussian graphical models. Ann. Statist., 40(1):238–261, 02 2012. doi:10.1214/11-AOS957.

[31] Caroline Uhler. Gaussian graphical models. In Marloes Maathuis, Mathias Drton, Stefen Lauritzen, and Martin Wainwright, editors, Handbook of Graphical Models, pages 311–330. CRC Press, Boca Raton, 2018. doi:10.1201/9780429463976-9.

[32] L. Vietoris. Uber den h¨oheren Zusammenhang kompakter R¨aume und eine Klasse von<sup>¨</sup> zusammenhangstreuen Abbildungen. Mathematische Annalen, 97:454–472, 1927. doi: 10.1007/BF01447877.

[33] Ting Wang, Zhao Ren, Ying Ding, Zhou Fang, Zhe Sun, Matthew L MacDonald, Robert A Sweet, Jieru Wang, and Wei Chen. Fastggm: an eficient algorithm for the inference of gaussian graphical model in biological networks. PLoS computational biology, 12(2):e1004755, 2016. doi:10.1371/journal.pcbi.1004755.

[34] Anja Wille, Philip Zimmermann, Eva Vranov´a, Andreas F¨urholz, Oliver Laule, Stefan Bleuler, Lars Hennig, Amela Prelic, Peter von Rohr, Lothar Thiele, et al. Sparse graphical gaussian modeling for genetic regulatory network inference. Genome Biol, 5(11):R92, 2004. doi:10.1186/gb-2004-5-11-r92.

[35] Haitao Zhao and Zhong-Hui Duan. Cancer genetic network inference using gaussian graphical models. Bioinformatics and biology insights, 13:1177932219839402, 2019. doi: 10.1177/1177932219839402.

[36] Afra Zomorodian and Gunnar Carlsson. Computing persistent homology. Discrete Comput. Geom, 33(2):249–274, 2005. doi:10.1007/s00454-004-1146-y.

## A Supplementary Material

## A.1 Colored 3 and 4-cycles

Table 1: Colored 4-cycles from [30, Table 2] studied in this paper
<table><tr><td>Graph</td><td>K</td><td>1 obs.</td><td>2 obs.</td><td> $\geq 3$  obs.</td></tr><tr><td><img src="images/c2bf64a3985feb19d9e6637ce14144b4b3c76f330e83f16ed565805612abf422.jpg"/></td><td> $\left( \begin{array} { c c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 2 } } } & { { 0 } } & { { \lambda _ { 2 } } } \\ { { \lambda _ { 2 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 3 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } \\ { { \lambda _ { 2 } } } & { { 0 } } & { { \lambda _ { 3 } } } & { { \lambda _ { 1 } } } \end{array} \right)$ </td><td> $p = 1$ </td><td> $p = 1$ </td><td> $p = 1$ </td></tr><tr><td><img src="images/793162ed60c296e4658f74ddd25d5acd344afaf8e76b6ae8c374adeee3fe00a4.jpg"/></td><td> $\left( \begin{array} { c c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 2 } } } & { { 0 } } & { { \lambda _ { 2 } } } \\ { { \lambda _ { 2 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 3 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 4 } } } \\ { { \lambda _ { 2 } } } & { { 0 } } & { { \lambda _ { 4 } } } & { { \lambda _ { 1 } } } \end{array} \right)$ </td><td> $p = 1$ </td><td> $p = 1$ </td><td> $p = 1$ </td></tr><tr><td><img src="images/035dc4e538e2d021208751b280477bb2e9d933e895af7c19c0c5e3d0da136762.jpg"/></td><td> $\left( \begin{array} { c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 4 } } } & { { 0 } } & { { \lambda _ { 4 } } } \\ { { \lambda _ { 4 } } } & { { \lambda _ { 2 } } } & { { \lambda _ { 5 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 5 } } } & { { \lambda _ { 3 } } } & { { \lambda _ { 6 } } } \\ { { \lambda _ { 4 } } } & { { 0 } } & { { \lambda _ { 6 } } } & { { \lambda _ { 2 } } } \end{array} \right)$ </td><td> $\mathrm { { N o } \ ( P r o p . 3 . 1 5 ) }$ </td><td> $p = 1$ </td><td> $p = 1$ </td></tr><tr><td><img src="images/48817c74237d596797a4fad824f00cf75dc45b1d136d6cd2405e4d4deacfbdf1.jpg"/></td><td> $\left( \begin{array} { c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } & { { 0 } } & { { \lambda _ { 4 } } } \\ { { \lambda _ { 3 } } } & { { \lambda _ { 2 } } } & { { \lambda _ { 4 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 4 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 5 } } } \\ { { \lambda _ { 4 } } } & { { 0 } } & { { \lambda _ { 5 } } } & { { \lambda _ { 2 } } } \end{array} \right)$ </td><td> $\mathrm { { N o } \ ( P r o p . 3 . 1 5 ) }$ </td><td> $p = 1$ </td><td> $p = 1$ </td></tr><tr><td><img src="images/b00dc763f55c3b75d905edc2b9e72683a70bd7d896cdd521508711a5c18151ac.jpg"/></td><td> $\left( \begin{array} { c c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 2 } } } & { { 0 } } & { { \lambda _ { 5 } } } \\ { { \lambda _ { 2 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 3 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 4 } } } \\ { { \lambda _ { 5 } } } & { { 0 } } & { { \lambda _ { 4 } } } & { { \lambda _ { 1 } } } \end{array} \right)$ </td><td> $p = 1 ~ ( \mathrm { P r o p . ~ 3 . 2 6 } )$ </td><td> $p = 1$ </td><td>p = 1</td></tr><tr><td><img src="images/89ed1f102ef84ee94e3395d12ce50d8cf88836c8d3543660f902776fc6f0a409.jpg"/></td><td> $\left( \begin{array} { c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 3 } } } & { { 0 } } & { { \lambda _ { 6 } } } \\ { { \lambda _ { 3 } } } & { { \lambda _ { 2 } } } & { { \lambda _ { 4 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 4 } } } & { { \lambda _ { 1 } } } & { { \lambda _ { 5 } } } \\ { { \lambda _ { 6 } } } & { { 0 } } & { { \lambda _ { 5 } } } & { { \lambda _ { 2 } } } \end{array} \right)$ </td><td> $\mathrm { { N o } \ ( P r o p . 3 . 1 5 ) }$ </td><td> $p = 1$ </td><td> $p = 1$ </td></tr><tr><td><img src="images/c274e89ec449194be1dd3f87f185b0bd402ec38dc32c9b6140d1f4a858ccd9c0.jpg"/></td><td> $\left( \begin{array} { c c c } { { \lambda _ { 1 } } } & { { \lambda _ { 4 } } } & { { 0 } } & { { \lambda _ { 7 } } } \\ { { \lambda _ { 4 } } } & { { \lambda _ { 2 } } } & { { \lambda _ { 5 } } } & { { 0 } } \\ { { 0 } } & { { \lambda _ { 5 } } } & { { \lambda _ { 3 } } } & { { \lambda _ { 6 } } } \\ { { \lambda _ { 7 } } } & { { 0 } } & { { \lambda _ { 6 } } } & { { \lambda _ { 2 } } } \end{array} \right)$ </td><td> $\mathrm { { N o } \ ( P r o p . 3 . 1 5 ) }$ </td><td> $p = 1$ </td><td>p = 1</td></tr><tr><td><img src="images/569a1e21c85fd3ef0394a94077ddce0fe3538bb4083c62981fae4fed1033f93d.jpg"/></td><td> $\begin{array} { r } { \left( \lambda _ { 1 } \ \lambda _ { 5 } 0 \lambda _ { 8 } \right) } \\ { \lambda _ { 5 } \ \lambda _ { 2 } \ \lambda _ { 6 } \ 0 } \\ { 0 \lambda _ { 6 } \ \lambda _ { 3 } \ \lambda _ { 7 } } \\ { \lambda _ { 8 } \ 0 \lambda _ { 7 } \ \lambda _ { 4 } } \end{array}$ </td><td>No</td><td> $p \in ( 0 , 1 )$ </td><td>p = 1</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 2 <sub>:</sub> ML t h<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> f<sub>or</sub> <sub>a</sub>ll <sub>co</sub>l<sub>ore</sub>d 3- <sub>cyc</sub>l<sub>es</sub> t<sub>oge</sub>t h<sub>er</sub> <sub>w</sub>it h t h<sub>e</sub>i<sub>r</sub> <sub>e</sub>li<sub>m</sub>i<sub>na</sub>t i<sub>on</sub> id<sub>ea</sub>l<sub>s .</sub>
<table><tr><td rowspan=1 colspan=1>Graph</td><td rowspan=1 colspan=1>Conc. matrix</td><td rowspan=1 colspan=1>dim</td><td rowspan=1 colspan=1>WMLT</td><td rowspan=1 colspan=1>MLT</td><td rowspan=1 colspan=1> $I _ { G , 1 }$ </td><td rowspan=1 colspan=1> $I _ { G , 2 }$ </td><td rowspan=1 colspan=1>Sufficient statistics</td></tr><tr><td rowspan=1 colspan=1>uncolored</td><td rowspan=1 colspan=1>λ1 λ4 λ5λ4 λ2 λ6λ5 λ6 λ3</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $\begin{array} { l } { { t _ { 4 } ^ { 2 } \ - \ 4 t _ { 2 } t _ { 1 } , t _ { 4 } ^ { 2 } \ - \ 4 t _ { 2 } t _ { 1 } , t _ { 5 } t _ { 4 } \ - \ 2 t _ { 6 } t _ { 1 } , t _ { 6 } t _ { 4 } \ - \ 2 t _ { 5 } t _ { 2 } , t _ { 5 } ^ { 2 } \ - } } \\ { { 4 t _ { 3 } t _ { 1 } , t _ { 6 } t _ { 5 } \ - 2 t _ { 4 } t _ { 3 } , t _ { 6 } ^ { 2 } \ - 4 t _ { 3 } t _ { 2 } \ } } \end{array}$ </td><td rowspan=1 colspan=1> $t _ { 6 } t _ { 5 } t _ { 4 } - t _ { 4 } ^ { 2 } t _ { 3 } - t _ { 5 } ^ { 2 } t _ { 2 } -$  $t _ { 6 } ^ { 2 } t _ { 1 } + 4 t _ { 3 } \hat { t } _ { 2 } t _ { 1 }$ </td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } , t _ { 2 } = s _ { 2 2 } , t _ { 3 } = s _ { 3 3 } ,$  $t _ { 4 } = 2 s _ { 1 2 } , t _ { 5 } = 2 s _ { 1 3 } , t _ { 6 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>2 vertices</td><td rowspan=1 colspan=1>λ1 λ3 λ4λ3 λ1 λ5λ4 λ5 λ2</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } t _ { 3 } - 2 t _ { 5 } t _ { 2 } , t _ { 4 } ^ { 2 } + t _ { 3 } ^ { 2 } - 4 t _ { 5 } t _ { 1 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } , t _ { 2 } = s _ { 3 3 } ,$  $t _ { 3 } = 2 s _ { 1 2 } , t _ { 4 } = 2 s _ { 1 3 } , t _ { 5 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>2 edges</td><td rowspan=1 colspan=1>λ1 λ4 λ4λ4 λ2λ5λ4 λ5 λ3</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } ^ { 2 } - 4 t _ { 5 } t _ { 1 } - 4 t _ { 3 } t _ { 1 } - 4 t _ { 2 } t _ { 1 } , t _ { 5 } ^ { 2 } - 4 t _ { 3 } t _ { 2 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } , t _ { 2 } = s _ { 2 2 } , t _ { 3 } = s _ { 3 3 } ,$  $t _ { 4 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } , t _ { 5 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>2 vert. 2 edges(excl. edge)</td><td rowspan=1 colspan=1>λ1 λ3 λ4λ3 λ1 λ4λ4 λ4 λ2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } ^ { 2 } - 4 t _ { 3 } t _ { 2 } - 4 t _ { 2 } t _ { 1 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } , t _ { 2 } = s _ { 3 3 } ,$  $t _ { 3 } = 2 s _ { 1 2 } , t _ { 4 } = 2 s _ { 1 3 } + 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>2 vert. 2 edges(incl. edge)</td><td rowspan=1 colspan=1>λ1 λ3 λ3λ3 λ1 λ4λ3 λ4 λ2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } ^ { 4 } + 4 t _ { 4 } ^ { 3 } t _ { 2 } + 4 t _ { 4 } ^ { 2 } t _ { 2 } ^ { 2 } + 4 t _ { 3 } ^ { 2 } t _ { 2 } ^ { 2 } - 4 t _ { 4 } ^ { 2 } t _ { 2 } t _ { 1 } - 1 6 t _ { 4 } t _ { 2 } ^ { 2 } t _ { 1 } - 1 6 t _ { 2 } ^ { 3 } t _ { 1 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } , t _ { 2 } = s _ { 3 3 } ,$  $t _ { 3 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } , t _ { 4 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>3 vertices</td><td rowspan=1 colspan=1>λ1 λ2 λ3λ2 λ1 λ4λ3 λ4 λ1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } ^ { 2 } t _ { 3 } ^ { 2 } + t _ { 4 } ^ { 2 } t _ { 2 } ^ { 2 } + t _ { 3 } ^ { 2 } t _ { 2 } ^ { 2 } - 2 t _ { 4 } t _ { 3 } t _ { 2 } t _ { 1 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } + s _ { 3 3 } ,$  $t _ { 4 } = 2 s _ { 1 2 } , t _ { 5 } = 2 s _ { 1 3 } , t _ { 6 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>3 edges</td><td rowspan=1 colspan=1>λ1 λ4 λ4λ4 λ2 λ4λ4 λ4 λ3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1> $t _ { 4 } ^ { 4 } - 8 t _ { 4 } ^ { 2 } t _ { 3 } t _ { 2 } + 1 6 t _ { 3 } ^ { 2 } t _ { 2 } ^ { 2 } - 8 t _ { 4 } ^ { 2 } t _ { 3 } t _ { 1 } - 8 t _ { 4 } ^ { 2 } t _ { 2 } t _ { 1 } - 6 4 t _ { 4 } t _ { 3 } t _ { 2 } t _ { 1 } -$  $3 2 t _ { 3 } ^ { 2 } t _ { 2 } t _ { 1 } - 3 2 t _ { 3 } t _ { 2 } ^ { 2 } t _ { 1 } + 1 6 t _ { 3 } ^ { 2 } t _ { 1 } ^ { 2 } - 3 2 t _ { 3 } t _ { 2 } t _ { 1 } ^ { 2 } + 1 6 t _ { 2 } ^ { 2 } t _ { 1 } ^ { 2 }$ </td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } , t _ { 2 } = s _ { 2 2 } , t _ { 3 } = s _ { 3 3 } ,$  $t _ { 4 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } + 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>3 vert. 2 edges</td><td rowspan=1 colspan=1>λ1 λ2 λ2λ2 λ1 λ3λ2 λ3 λ1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } + s _ { 3 3 } ,$  $t _ { 2 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } , t _ { 3 } = 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>2 vert. 3 edges</td><td rowspan=1 colspan=1>λ1 λ3 λ3λ3 λ1 λ3λ3 λ3 λ2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } , t _ { 2 } = s _ { 3 3 } ,$  $t _ { 3 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } + 2 s _ { 2 3 }$ </td></tr><tr><td rowspan=1 colspan=1>3 vert. 3 edges</td><td rowspan=1 colspan=1>λ1 λ2 λ2λ2 λ1 λ2λ2 λ2 λ1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $t _ { 1 } = s _ { 1 1 } + s _ { 2 2 } + s _ { 3 3 } ,$  $t _ { 2 } = 2 s _ { 1 2 } + 2 s _ { 1 3 } + 2 s _ { 2 3 }$ </td></tr></table>

## A.2 Sampling the cone of suficient statistics

![](images/1e3c492854338006c3814edcaddb8b710fdbf755106a60e283d0e1543ed5da4c.jpg)  
Figure 10: Cross-section of the cone of suficient statistics for Graph 1 (all vertices in one color class), based on [30, Table 2].

## A.3 Histogram of Cholesky parameters

![](images/423cf508cd672418afa805d22b5a3f9a2c035c261963145273f86071055e5d96.jpg)

<table><tr><td>Graph 2 sufficient statistics</td></tr><tr><td>◆ 1 obs.</td></tr><tr><td>2 obs.</td></tr><tr><td>1 3 obs.</td></tr><tr><td>O 4 obs.</td></tr></table>

Figure 11: Cross-section of the cone of suficient statistics for Graph 2, based on [30, Table 2].

![](images/08c37d2389e91bb994b88f871bad41a5eb08299242b70ebcf8b1b181e4b64ac3.jpg)  
Figure 12: Cross-section of the cone of suficient statistics for Graph 3 (all vertices in one color class), based on [30, Table 2].

![](images/4808c080c7f21ffe0199649625b9db99c29e37c2e8979ac5bdfdccf804a0467f.jpg)

<table><tr><td>Graph 4 sufficient statistics</td></tr><tr><td>◆ 1 obs.</td></tr><tr><td>2 obs.</td></tr><tr><td>4 3 obs.</td></tr><tr><td>O 4 obs.</td></tr></table>

Figure 13: Cross-section of the cone of suficient statistics for Graph 4, based on [30, Table 2].

![](images/8c087286b4a9c508682bea24703ed3ed8514039e33ab62d714aeaac7a52d3b7d.jpg)

<table><tr><td>Graph 5 sufficient statistics</td></tr><tr><td>◆ 1 obs.</td></tr><tr><td>2 obs.</td></tr><tr><td>1 3 obs.</td></tr><tr><td>O 4 obs.</td></tr></table>

Figure 14: Cross-section of the cone of suficient statistics for Graph 5, based on [30, Table 2].

![](images/3cdd9268941dbec7884b2f2162764940ef9b9251eee67a34ac1e136ec53a08b4.jpg)

<table><tr><td>Graph 6 sufficient statistics</td></tr><tr><td>◆ 1 obs.</td></tr><tr><td>2 obs.</td></tr><tr><td>1 3 obs.</td></tr><tr><td>O 4 obs.</td></tr></table>

Figure 15: Cross-section of the cone of suficient statistics for Graph 6 (all vertices in one color class), based on [30, Table 2].

![](images/2a9bd0532bda6bc4b2969dcff453810a775a7d4904fd2b939ce6c7e6b2cfc159.jpg)  
Figure 16: Distribution of deviations of the Cholesky pre-images of neighborhood points versus target, intermediate and reference stats for Graph 3 from Figure 6.