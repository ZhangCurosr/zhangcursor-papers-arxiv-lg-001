# Feature-Aware Edge Sampling for Approximate One-Step Laplacian Propagation

Jonathan Z. Sun Charleston Southern University jsun@csuniv.edu

Zheng Wang University of Miami zheng.wang@miami.edu

Kebin Xu   
College of Charleston   
xuk@cofc.edu

## Abstract

This paper studies how to approximate one-step combinatorial Laplacian messages in graph neural networks using edge sampling. We propose a spanning tree based hybrid sampler that combines feature-conditioned message scores and efective-resistance leverage scores, derive the expected one-step message error and obtain the optimal sampling probabilities and error frontier for a given expected edge budget. Preliminary experiments on synthetic community graphs show that message error decreases, while classification accuracy increases slightly, as the edge budget grows, and performance of the hybrid sampler is in between of optimal selection uniform random sampling. We leave multi-step propagation and evaluation on larger and real graphs for future work.

Keywords: graph sparsification, graph neural networks, randomized algorithms, importance sampling, feature conditioned, efective resistance, message passing, Krylov subspace

## 1 Introduction

Graph neural networks have become a standard framework for learning from relational data. GCN, GraphSAGE, and GAT exemplify the message passing paradigm in which node representations are repeatedly updated by aggregating information from neighbors [11, 15, 22]. For a sparse propagation matrix $P .$ , a typical layer has the form

$$
H ^ { ( \ell + 1 ) } = \sigma _ { \ell } \Bigl ( P H ^ { ( \ell ) } W _ { \ell } \Bigr ) , \qquad H ^ { ( 0 ) } = X ,\tag{1}
$$

where X is the initial node feature matrix, P is the graph propagation matrix, and $H ^ { \ell } , \ W ^ { \ell } , \ \sigma _ { \ell }$ are node representation matrix, learnable weight matrix, and activation function at layer ℓ.

## 1.1 Motivation

Although the cost of one sparse propagation is linear in the number of retained edges, repeated aggregation on very large graphs is expensive in time and memory. This scalability problem has motivated a variety of sampling schemes such as neighborhood sampling, layer-wise importance sampling, subgraph sampling, clustering, graph reduction, and model simplification [3, 5, 11–13, 23, 24, 26].

Graph sparsification replaces a graph by a substantially smaller weighted subgraph while preserving selected properties of the original graph. Classical cut and spectral sparsifiers provide strong guarantees for cuts, Laplacian quadratic forms, and related structural quantities [1, 9, 14, 18, 19]. However, these guarantees are task agnostic. In general, for a graph Laplacian L, a spectral sparsifier $L _ { H }$ seeks to satisfy

$$
( 1 - \epsilon ) x ^ { \top } L x \leq x ^ { \top } L _ { H } x \leq ( 1 + \epsilon ) x ^ { \top } L x \quad { \mathrm { f o r ~ e v e r y ~ } } x \in \mathbb { R } ^ { n } .\tag{2}
$$

However, a graph neural network (GNN) does not ordinarily explore all directions in $\mathbb { R } ^ { n }$ . Instead of querying the graph through arbitrary vectors, a GNN repeatedly propagates a structured and evolving collection of node features [11, 15, 22]. Starting from a feature matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ , its linear propagation stages repeatedly generate signals related to X, $P X , P ^ { 2 } X , \dots , P ^ { n } X$ . Moreover, analysis of simplified GCN already illustrates the close relationship between graph propagation and polynomial graph filtering [23]; while recent work on spectral GNN further reveals the relationship more explicitly [7].

This motivates our main problem of study: How to design heuristic sparsifiers for GNN to preserve most feature signals used for message passing without paying the cost of preserving the original graph on every spectral direction?

## 1.2 Idea

In this paper, we study the above problem for one-step combinatorial Laplacian propagation on fixed input features. The idea is to retain a spanning tree and then independently sample the remaining edges, combining feature-conditioned message scores with efective-resistance scores. We derive the exact expected squared message error, the minimum error for each expected edge budget, and the hybrid proposal’s excess over that minimum. In a controlled synthetic community graph experiment (with preset graph structures and feature conditions), message error decreases as the edge budget grows, while node classification accuracy remains similar across the tested sampling rules. Schemes and experiments for standard multi-layer GNNs and real world graphs remain in future work.

Recent work already demonstrated the potential of GNN sparsification. FastGAT applies efective-resistancebased spectral sparsification to attention networks [4, 6, 7, 17, 20, 25]. Therefore we are not making “yet another edge score”, but instead, interested in designing a message-preserving sparsifier.

We study how the graph Laplacian L acts on the feature matrix X and consider message preservation, i.e., the propagation result on the sparsified graph should be close to that on the original graph. Our goal is to construct a reusable sparse graph of given edge budget (amount of edges retained) as a preprocessing before downstream model training. Our method retains a spanning tree and independently samples the remaining edges, and reweights each sampled edge by the inverse of its inclusion probability. It then derives the expected one-step message error and obtains the optimal sampling probabilities and error frontier for a given expected edge budget. By combining a spanning tree backbone with the proposed feature-conditioned message score and the classic efective-resistance leverage score, we provide the algorithm of a hybrid sampler and quantify its error relative to optimal message allocation.

## 1.3 Contributions

The goal of sampling here is computational cost in time and memory: to reduce the number of edges used in repeated propagation, while limiting a specified approximation error and not sacrificing predictive accuracy. Here “message approximation” refers to approximating the one-step action LX for the fixed input feature matrix X with $L _ { H } X$

Major contributions in this paper are as follows.

1. We define one-step message approximation through the error in applying combinatorial Laplacian to the input features. For independent edge sampling, we derive an unbiased estimator, its exact expected squared error, and the budget dependent optimal allocation with a fixed spanning tree backbone.

2. We design a hybrid scheme that considers both feature-conditioned message importance and efectiveresistance importance and also retains a spanning-tree backbone to guarantee sampled graph’s connectivity, and give exact decomposition of its message-error excess relative to the optimal message allocation.

3. We perform preliminary experiments to support the soundness of our algorithm and analysis. Experimental results meet expectation, although not significant due to the quantity and quality of synthetic data we generated.

4. We also present the feature-generated Krylov space as a direction for future work on multi-step propagation.

## 1.4 Paper Organization

The paper is organized as follows. Section 2 reviews three lines of related work: classical graph sparsification as background, GNN computation sampling as a strategy of saving training time and memory, and GNN-oriented graph reduction as the closest context of reference. Section 3 introduces the terminologies and formulations to define a weighted graph and its Laplacian, edge message score, and approximation objectives; it also introduces Krylov-space preservation as foundation for future analysis. Section 4 specifies the tree-based hybrid sampler, derives its expected one-step message error and budget-optimal allocation, and quantifies the hybrid algorithm’s excess error. Section 5 performs a controlled experiments with preset conditions on modest size synthetic data, which shows message error decreases with the edge budget grows, while node classification accuracy remains similar with no loss. Section 6 summarizes these preliminary findings and proposes future work for multi-layer propagation models and real graphs.

## 2 Related Work

In this section we summarize the following three lines of work as they represent the methodological background, the complementary strategy for computational eficiency, and the closest competing approaches to reference to.

## 2.1 Classical Graph Sparsification

Graph sparsification provides combinatorial algorithmic approaches to tackling the problem. Instead of sampling a temporary computation graph during a training iteration, it constructs a persistent sparse graph that approximates selected properties of the original graph.

Eppstein et al. introduced algorithmic sparsification using sparse certificates as a technique for accelerating dynamic graph algorithms [9]. Karger developed random edge sampling methods that preserve graph cuts with high probability and used sparse sampled skeletons in cut, flow, and network-design algorithms [14]. Spielman and Teng developed spectral sparsification to strengthened the preservation objective to Laplacian quadratic forms, resulting in nearly-linear-time graph algorithms [19]. Spielman and Srivastava then showed that sampling edges according to efective resistance yields high-quality spectral sparsifiers with $O ( n \log n / \epsilon ^ { 2 } )$ edges [18]. Batson, Spielman, and Srivastava subsequently presented twice-Ramanujan sparsifier to prove the existence of spectral sparsifiers with number of edges linear in n [1]. Fung et al. presented a broader framework that encloses multiple sparsification measures, including edge strength, connectivity, and efective resistance [10].

The objective of these classical combinatorial sparsifiers is to support arbitrary graph algorithms independent of any particular downstream machine learning models. However, ideas and results can be borrowed to inspire heuristics, particularly when the downstream model works primarily on low dimensional node signals.

## 2.2 Sampling for Scalable GNN Training

This line of work direct to reducing the cost of GNN training by sampling the computation graph. GraphSAGE samples and aggregates local neighborhoods to enable inductive representation learning [11]. FastGCN interprets graph convolution as an integral transform and applies importance sampling to obtain eficient mini-batch estimators [3]. Adaptive sampling methods explicitly seek variance reduction [13]. LADIES uses layer-dependent importance sampling conditioned on nodes sampled in the upper layer [26]. Cluster-GCN exploits graph clustering to train on densely connected blocks [5]. GraphSAINT samples subgraphs and applies normalization to reduce sampling bias [24]. DropEdge randomly removes edges during training and was proposed primarily as a regularization mechanism against overfitting and over-smoothing [16].

These methods are relevant to our work because they demonstrate that sampling distributions can be designed to reduce estimator variance and training cost. However, our approach is diferent: our proposed method produces a single persistent sparse graph as a preprocessing, so its cost can be amortized across epochs and iterations using the same graph. Reusing the sparsified graph across architectures or downstream tasks is possible only when the propagation requests are compatible with the feature-conditioned sampling target; it is not guaranteed by the one-step analysis. Diferent from this line of work, our focus is the graph-level preservation objective and its feature-generated subspace interpretation.

## 2.3 Graph Reduction and GNN-Oriented Sparsification

Graph reduction includes sparsification, coarsening, condensation, and related summarization techniques [12]. Empirical evaluations demonstrated that preservation of conventional graph metrics does not automatically imply preservation of GNN behavior [4]. Vatter et al. studied graph sparsification jointly with GNN sampling and showed that substantial graph reduction can coexist with competitive learning performance [21].The following methods were particularly designed for sparsification with GNNs.

FastGAT uses efective-resistance-based spectral sparsification to reduce the cost of graph attention networks while providing a theoretical connection between spectral approximation and learned features [20]. GPER uses efective resistance to identify edges for rewiring and preprocessing, targeting over-smoothing and over-squashing [17]. Spectral Greedy Graph Coresets select representative ego-graphs using spectral embeddings, addressing a related but node/ego-graph selection problem rather than persistent edge sparsification [8]. Mixture-of-Graphs dynamically selects locally appropriate pruning experts and mixes sparse graphs [25]. SGS-GNN learns supervised edge-sampling distributions and uses label-related objectives to improve sparse-graph prediction [6]. Ding et al. use Laplacian sparsification to approximate polynomial filters in large-scale spectral GNNs and provide corresponding theoretical guarantees [7].

Comparing to the above methods, the scheme proposed in the next is unique in 1) its message score is derived from the one-step action of the combinatorial Laplacian on the input features rather than learned from labels or a downstream supervised loss; 2) it retains a spanning tree backbone and also take into consideration efective-resistance to yield a hybrid algorithm; and 3) its analysis concerns message error on the fixed feature matrix, while also consider Krylov-space preservation as a future direction to study multi-layer propagation.

## 3 Terminologies and Problem Formulation

Let $G = ( V , E , w , X )$ be a connected, undirected, weighted attributed graph with $n = | V | , m = | E |$ , positive edge weights $w _ { e }$ , and feature matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ . Let $B \in \mathbb { R } ^ { m \times n }$ be an oriented incidence matrix and $W = \mathrm { d i a g } ( w _ { e } )$ The combinatorial Laplacian is

$$
L = B ^ { \top } W B = \sum _ { e \in E } w _ { e } b _ { e } b _ { e } ^ { \top } ,\tag{3}
$$

where $b _ { e . } ^ { \top }$ is the incidence row associated with edge e.

Applying L to X gives

$$
L X = \sum _ { e \in E } M _ { e } ( X ) , \qquad M _ { e } ( X ) = w _ { e } b _ { e } b _ { e } ^ { \top } X .\tag{4}
$$

For $\boldsymbol { e } = ( u , v )$

$$
\| M _ { e } ( X ) \| _ { F } = \| b _ { e } \| _ { 2 } w _ { e } \| b _ { e } ^ { \top } X \| _ { 2 } = \sqrt { 2 } w _ { e } \| x _ { u } - x _ { v } \| _ { 2 } .\tag{5}
$$

Here $M _ { e } ( X )$ the contribution of edge e to the matrix LX. That is, an edge has high one-step message importance when it carries a large feature diference under Laplacian propagation.

Definition 1 (Message-preserving sparsifier). For feature matrix X, budget q, and tolerance $\epsilon > 0$ , a weighted subgraph $H = ( V , E _ { H } , \widetilde { w } )$ with $| E _ { H } | \le q$ is an $( X , \epsilon )$ message-preserving sparsifier if

$$
\| L X - L _ { H } X \| _ { F } \leq \epsilon \| L X \| _ { F } .\tag{6}
$$

Note that this definition is weaker than the full spectral sparsification in Eq. (2). It asks for accurate action only on the columns of X, not on every vector in R<sup>n</sup>.

## 3.1 Subspace and Krylov Preservation

Let $S \subseteq \mathbb { R } ^ { n }$ be a subspace of R<sup>n</sup>.

Definition 2 (Subspace spectral sparsifier). H is an $( S , \epsilon )$ subspace spectral sparsifier of G if

$$
( 1 - \epsilon ) x ^ { \top } L x \leq x ^ { \top } L _ { H } x \leq ( 1 + \epsilon ) x ^ { \top } L x\tag{7}
$$

for every $x \in S .$

When ${ \mathcal { S } } = \mathbb { R } ^ { n }$ , this becomes the classical definition of the spectral sparsification. For attributed graphs, the straightforward choice is let ${ \cal S } = \mathrm { s p a n } ( X )$ . Therefore for a k-step propagation process, we propose the feature-generated block Krylov space denoted by

$$
{ \mathcal { K } } _ { k } ( P , X ) = \operatorname { s p a n } \{ X , P X , P ^ { 2 } X , \ldots , P ^ { k } X \} .\tag{8}
$$

This construction is motivated by the observation that polynomial graph filters and linear message-passing models act inside spaces generated by repeated applications of the graph operator [7, 23]. The efective dimension of $\mathcal { K } _ { k } ( P , X )$ could be much smaller than $n ,$ particularly when $d \ll n$ and the propagated signals are highly correlated.

Therefore the intrinsic question behind sparsification is sharper than a GNN stability statement. The dimension of the span alone may not determine the required sample size; the geometry and concentration of the edge-message vectors within that span matters.

A valid guarantee would address whether the operator is the combinatorial Laplacian or a normalized propagation matrix, whether the target is a quadratic form or the operator action on feature columns, and whether the result holds in expectation or with high probability. These distinctions of sampling objectives would lead to diferent heuristics. The general problem, for which the current paper addresses a first step, is as follows.

How many sampled edges are suficient to approximate L or P on $\displaystyle \mathcal { K } _ { k } ( P , X )$ , and under what conditions can this amount be substantially smaller than the edge budget needed for a full spectral sparsifier?

## 4 Tree-Based Feature-Aware Sampling and Its Analysis

This section presents the proposed main method: a hybrid sampling algorithm for one-step propagation. Given a graph and its node-feature matrix, we keep a spanning tree and sample the remaining edges with the goal of approximating the Laplacian message LX using fewer edges. We first show that inverse-probability reweighting gives an unbiased estimate and derive its exact expected squared error. We then use that error to characterize the best allocation of a fixed sampling budget and assess a hybrid proposal that, in addition to message importance, also takes efective resistance into consideration.

## 4.1 Unbiased Message Sampling

Let T be a fixed spanning tree and $J = E \setminus T$ be the remaining optional edges excluding the tree. The tree edges are retained with their original weights; for $e \in J .$ , let $\xi _ { e } \sim \mathrm { B e r n o u l l i } ( p _ { e } )$ be independent and assigned weight $w _ { e } / p _ { \epsilon }$ <sub>e</sub> when retained. Denote

$$
M _ { e } ( X ) = w _ { e } b _ { e } b _ { e } ^ { \top } X , \qquad s _ { e } = \| M _ { e } ( X ) \| _ { F } .
$$

Then we have

$$
L _ { H } X - L X = \sum _ { e \in J } \left( \frac { \xi _ { e } } { p _ { e } } - 1 \right) M _ { e } ( X ) .\tag{9}
$$

Since $\mathbb { E } \left[ \xi _ { e } \right] = p _ { e }$ , each summand in Equation (9) is centered and therefore $\mathbb { E } [ L _ { H } X ] = L X$ . Since the summands are centered and independent, the cross inner products are zero in expectation, yielding the exact identity

$$
V ( p ) : = \mathbb { E } \| L _ { H } X - L X \| _ { F } ^ { 2 } = \sum _ { e \in J } s _ { e } ^ { 2 } \left( { \frac { 1 } { p _ { e } } } - 1 \right) .\tag{10}
$$

That is, the deterministic spanning tree contributes no sampling variance; its cost is the $n - 1$ edges removed from the randomized budget. Equation (10) is not a spectral or prediction guarantee but the exact message error objective.

## 4.2 Exact Budget Error Frontier

Assume $s _ { e } > 0$ for the optional edges and fix an expected optional edge budget B with $0 \textless B \textless | J |$ . The constrained problem is

$$
V ^ { * } ( B ) = \operatorname* { m i n } _ { \begin{array} { c } { { 0 < p _ { e } \leq 1 } } \\ { { \sum _ { e \in J } p _ { e } = B } } \end{array} } V ( p ) .\tag{11}
$$

The next theorem gives the minimum error for every budget, including the transition points at which high score edges are selected with probability 1.

Theorem 1 (Fixed-backbone message-error frontier). There is a unique $\lambda > 0$ such that

$$
p _ { e } ^ { * } ( B ) = \operatorname* { m i n } \left\{ 1 , \frac { s _ { e } } { \lambda } \right\} , \qquad \sum _ { e \in J } p _ { e } ^ { * } ( B ) = B .\tag{12}
$$

Let $\mathcal { A } = \{ e : s _ { e } \geq \lambda \}$ be the saturated edges and ${ \mathcal { F } } = J \setminus A$ . Then

$$
V ^ { * } ( B ) = \lambda \sum _ { e \in \mathcal { F } } s _ { e } - \sum _ { e \in \mathcal { F } } s _ { e } ^ { 2 } , \qquad \frac { d V ^ { * } ( B ) } { d B } = - \lambda ^ { 2 }\tag{13}
$$

within each interval on which A is fixed. In particular, the marginal decrease in minimum expected squared error per additional expected edge is $\lambda ^ { 2 }$ ; the frontier is continuous and piecewise smooth, with breakpoints when an edge enters or leaves A.

Proof. For each $e \in J .$ , the function $f _ { e } ( p _ { e } ) = s _ { e } ^ { 2 } ( 1 / p _ { e } - 1 )$ is strictly convex on $( 0 , 1 ] .$ , since $f _ { e } ^ { \prime \prime } ( p _ { e } ) = 2 s _ { e } ^ { 2 } / p _ { e } ^ { 3 } > 0$ Thus, the objective is strictly convex, and any feasible point satisfying the Karush-Kuhn-Tucker (KKT) conditions $[ 2 ,$ Sec. 5.5.3] is the unique global minimizer.

Introduce a multiplier $\eta$ for the equality constraint $\textstyle \sum _ { e \in J } p _ { e } \ = \ B$ and multipliers $\mu _ { e } \geq 0$ for the upper constraints $p _ { e } \leq 1$ . The Lagrangian is

$$
\mathcal { L } ( p , \eta , \mu ) = \sum _ { e \in J } s _ { e } ^ { 2 } \left( \frac { 1 } { p _ { e } } - 1 \right) + \eta \left( \sum _ { e \in J } p _ { e } - B \right) + \sum _ { e \in J } \mu _ { e } ( p _ { e } - 1 ) .
$$

The stationarity and complementary-slackness conditions are

$$
- \frac { s _ { e } ^ { 2 } } { p _ { e } ^ { 2 } } + \eta + \mu _ { e } = 0 , \qquad \mu _ { e } ( p _ { e } - 1 ) = 0 , \qquad \mu _ { e } \geq 0 .
$$

Because $B < | J |$ , at least one edge has $p _ { e } < 1$ . For such an edge, complementary slackness gives $\mu _ { e } = 0 ,$ , and stationarity yields $\eta = s _ { e } ^ { 2 } / p _ { e } ^ { 2 } > 0 .$ . If $p _ { e } < 1$ , then $\mu _ { e } = 0 ,$ , hence $p _ { e } = s _ { e } / \sqrt { \eta }$ . Otherwise if $p _ { e } = 1$ , stationarity gives $\mu _ { e } = s _ { e } ^ { 2 } - \eta \ge 0$ , hence $s _ { e } \geq \sqrt { \eta }$ . Let $\lambda = \sqrt { \eta }$ and combine these two cases as

$$
p _ { e } = \operatorname* { m i n } \left\{ 1 , \frac { s _ { e } } { \lambda } \right\} .
$$

The budget constraint then requires

$$
\sum _ { e \in J } \operatorname* { m i n } { \Big \{ } 1 , { \frac { s _ { e } } { \lambda } } { \Big \} } = B .
$$

The left-hand side is continuous in $\lambda ,$ decreases from $| J |$ to 0, and is strictly decreasing whenever its value is below |J|. Since $0 < B < | J |$ , this equation has a unique solution $\lambda > 0$ . This proves Eq. (12) and, by strict convexity, the optimal solution is unique.

Let ${ \mathcal { A } } = \{ e \in J : s _ { e } \geq \lambda \}$ and ${ \mathcal { F } } = J \setminus A .$ . For $e \in \mathcal { F } , p _ { e } ^ { * } = s _ { e } / \lambda$ , so its contribution to $V ^ { * } ( B )$ is

$$
s _ { e } ^ { 2 } \left( \frac { 1 } { p _ { e } ^ { * } } - 1 \right) = \lambda s _ { e } - s _ { e } ^ { 2 } .
$$

For $e \in \mathcal { A } , p _ { e } ^ { * } = 1$ , so its contribution is zero. Summing over $\mathcal { F }$ gives the first identity in Eq. (13).

When A is fixed, let $\begin{array} { r } { S _ { \mathcal { F } } = \sum _ { e \in \mathcal { F } } s _ { e } } \end{array}$ . The budget equation gives

$$
B = | A | + \frac { S _ { \mathcal { F } } } { \lambda } , \qquad \lambda = \frac { S _ { \mathcal { F } } } { B - | A | } .
$$

Consequently,

$$
V ^ { * } ( B ) = \frac { S _ { \mathcal { F } } ^ { 2 } } { B - | \boldsymbol { A } | } - \sum _ { e \in \mathcal { F } } { s _ { e } ^ { 2 } } ,
$$

and diferentiating within an interval on which A is fixed yields

$$
{ \frac { d V ^ { * } ( B ) } { d B } } = - { \frac { S _ { \mathcal { F } } ^ { 2 } } { ( B - | { \cal { A } } | ) ^ { 2 } } } = - \lambda ^ { 2 } .
$$

At a breakpoint where an edge becomes saturated, the formulas for $V ^ { * } ( B )$ on the two adjacent budget intervals agree because that edge contributes zero error at $p _ { e } ^ { * } = 1$ □

This theorem characterizes the solution to a convex program. Its role here is to provide the exact benchmark for a message estimator that reserves a prescribed connectivity backbone. The proof also shows why the uncapped proportional-allocation rule alone is insuficient: the variance curve changes whenever an edge reaches an inclusion probability of one.

## 4.3 Structural- and Message-Preserving Sampling

The message-optimal allocation in Eq. (12) uses only the feature-conditioned scores $s _ { e }$ . This subsection defines a feasible proposal that also uses efective-resistance leverage scores $\tau _ { e } ,$ which encode structural information from the graph Laplacian. The goal is to compare the two allocations at the same expected edge budget and to quantify the one-step message-error cost of incorporating structural scores, which is given in the following proposition.

For $e \in J ,$ , let $s _ { e }$ be the feature-conditioned message score from Eq. (5) and let $\tau _ { e } = w _ { e } b _ { e } ^ { \top } L ^ { \dag } b _ { e }$ be the efective-resistance leverage score [18]. Normalize the two score families:

$$
r _ { e } = \frac { \tau _ { e } } { \sum _ { f \in J } \tau _ { f } } , \qquad m _ { e } = \frac { s _ { e } } { \sum _ { f \in J } s _ { f } } , \qquad \pi _ { e } ( \alpha ) = \alpha r _ { e } + ( 1 - \alpha ) m _ { e } ,\tag{14}
$$

where $0 \leq \alpha \leq 1$ . For a target expected optional-edge budget B, choose $c _ { \alpha }$ so that $p _ { e } ^ { \alpha } = \operatorname* { m i n } \{ 1 , c _ { \alpha } \pi _ { e } ( \alpha ) \}$ satisfies $\textstyle \sum _ { e \in J } p _ { e } ^ { \alpha } = B$ . This mixture is a feasible proposal, not the message-optimal allocation in Eq. (12).

Proposition 1 (Exact excess error of a feasible proposal). Let p be any feasible inclusion-probability vector with $0 < p _ { e } \leq 1$ and $\textstyle \sum _ { e \in J } p _ { e } = B$ , and let $p ^ { * } , \lambda , A _ { }$ , and $\mathcal { F }$ be as defined above. Then

$$
V ( p ) - V ^ { * } ( B ) \geq 0 .\tag{15}
$$

Proof. We will establish the following equation:

$$
V ( p ) - V ^ { * } ( B ) = \sum _ { e \in J } \frac { s _ { e } ^ { 2 } ( p _ { e } - p _ { e } ^ { * } ) ^ { 2 } } { p _ { e } ( p _ { e } ^ { * } ) ^ { 2 } } + \sum _ { e \in A } ( s _ { e } ^ { 2 } - \lambda ^ { 2 } ) ( 1 - p _ { e } ) .\tag{16}
$$

For each edge, we know the following algebraic identity holds:

$$
\frac { s _ { e } ^ { 2 } } { p _ { e } } - \frac { s _ { e } ^ { 2 } } { p _ { e } ^ { * } } = \frac { s _ { e } ^ { 2 } ( p _ { e } - p _ { e } ^ { * } ) ^ { 2 } } { p _ { e } ( p _ { e } ^ { * } ) ^ { 2 } } - \frac { s _ { e } ^ { 2 } } { ( p _ { e } ^ { * } ) ^ { 2 } } ( p _ { e } - p _ { e } ^ { * } ) .\tag{17}
$$

Therefore we only need to show that summing the 2nd term on RHS of (17) over all edges results in the 2nd term on RHS of (16). For $e \in \mathcal { F } , s _ { e } ^ { 2 } / ( p _ { e } ^ { * } ) ^ { 2 } = \lambda ^ { 2 }$ , whereas for $e \in \mathcal { A } , p _ { e } ^ { * } = 1$ . Thus we sum the 2nd term on RHS of (17) for all edges and get

$$
- \lambda ^ { 2 } \sum _ { e \in \mathcal { F } } ( p _ { e } - p _ { e } ^ { * } ) - \sum _ { e \in \mathcal { A } } s _ { e } ^ { 2 } ( p _ { e } - 1 )\tag{18}
$$

Because $p$ and $p ^ { * }$ have the same budget, we have

$$
\sum _ { e \in \mathcal { F } } ( p _ { e } - p _ { e } ^ { * } ) = \sum _ { e \in \mathcal { A } } ( 1 - p _ { e } )\tag{19}
$$

Substituting (19) into the first term of (18) gives

$$
\sum _ { e \in \mathcal { A } } ( s _ { e } ^ { 2 } - \lambda ^ { 2 } ) ( 1 - p _ { e } ) ,
$$

establishing Eq. (16). It’s obvious that each term on RHS of (16) is non-negative. The position is proved.

Eq. (15) applies directly to the calibrated hybrid probabilities $p ^ { \alpha }$ and quantifies their message error cost relative to the best allocation at the same expected edge count. The proposition is a comparison identity that separates the proposal’s excess message error into two costs: the deviation from the message-optimal probabilities, and the undersampling edges that the message-optimal rule would retain with probability one. Applied to the calibrated probabilities $p ^ { \alpha }$ , it lets the algorithm and experiments compare the structure- and message-preserving proposal with the message-only benchmark at a matched budget.

## 4.4 Sampling Algorithm

The sampling algorithm is given in Algorithm 1. The spanning tree backbone alone gives no cut, spectral, message, or prediction guarantee, but only provides connectivity. The algorithm then computes scores for the remaining edges, calibrates their inclusion probabilities to meet the expected edge budget, and samples and reweights those edges. Computing the feature-conditioned message scores costs $O ( m d )$ arithmetic. Once the mixture probabilities are calibrated, forming them and sampling the optional edges costs $O ( m )$ ; the cost of calibration depends on the implementation. Exact efective-resistance scores can be expensive to compute, motivating approximate scores for large graphs [18]. If the sampled graph is reused for propagation, each subsequent step costs $O ( Q d )$ in expectation, compared with $O ( m d )$ on the original graph.

Algorithm 1 Tree-based hybrid message sampling   
Require: Connected weighted graph $\overline { { G = \left( V , E , w \right) } }$ , feature matrix X, total expected edge budget $Q \geq n - 1$   
mixture parameter α   
Ensure: Sparse weighted graph H   
1: Construct a spanning tree T and retain its edges at original weights   
2: For each $e \in E \setminus T ,$ compute $s _ { e } = \| w _ { e } b _ { e } b _ { e } ^ { \top } X \| _ { F }$ and $\tau _ { e } = w _ { e } b _ { e } ^ { \top } L ^ { \dag } b _ { e }$   
3: Normalize the scores and form $\pi _ { e } ( \alpha )$ using Eq. (14)   
4: Set $p _ { e } = \operatorname* { m i n } \{ 1 , c _ { \alpha } \pi _ { e } ( \alpha ) \}$ with $\begin{array} { r } { \sum _ { e \notin { T } } p _ { e } = Q - ( n - 1 ) } \end{array}$   
5: Independently sample each optional edge; reweight each retained edge by $1 / p _ { e }$   
6: return H containing T and the sampled optional edges

The analysis only applies to the one-step action of the combinatorial Laplacian on the fixed input feature matrix X. It does not establish guarantees for normalized propagation matrices, nonlinear hidden representations, multilayer networks, or feature-generated Krylov spaces. Extending the analysis to these more realistic circumstances is left for future work.

## 5 Experiments

The small synthetic pilot in this section only evaluates one-step message reconstruction and a simple downstream prediction probe. Purpose of the pilot is to find supportive evidences for the soundness of our heuristic sampling scheme and mathematical analysis.

## 5.1 Setup

Ten connected undirected stochastic block model graphs were generated with seeds 20261106–20261115. Each graph has 120 vertices in three balanced classes, with within-class and between-class edge probabilities 0.18 and 0.02, respectively; disconnected draws were rejected. Unit edge weights were used. Each node has 12 features: four class-specific coordinates have mean 0.7, and independent Gaussian noise with standard deviation 1.2 is added.

Each graph uses a randomized minimum spanning tree shared by all samplers. For the non-tree edges, we compare the uniform random sampling, message score only sampling, and 50/50 efective resistance and message score sampling. We set target edge ratios (budgets) to be 30%, 40%, . . . , 90%, 95%, and 99%. The inclusion probabilities are calibrated to the expected total edge budget, and sampled edges are reweighted by their inverse probabilities. We perform each test on 10 graph instances, and run five independent sampling draws per graph, method, and budget. Each reported data point is the average with standard deviation. The random seed is $1 0 ^ { 6 } s + 1 0 ^ { 4 } j + r _ { \mathrm { c o d e } } + 1 7 h$ , where s is the graph seed, j the draw index, $r _ { \mathrm { c o d e } }$ the ratio in thousandths, and h the method index.

We measure relative message error $\| L _ { H } X - L X \| _ { F } / \| L X \| _ { F }$ . For prediction, we apply one residual step, $X ^ { \prime } = X - \eta L _ { H } X$ , with $\eta = 0 . 5 / \lambda _ { \mathrm { m a x } } ( L )$ , then train logistic regression $( C = 1 )$ on the first 20 vertices per class and test on the remaining. Finally, this linear probe is very preliminary and not comparable to a standard GCN, GraphSAGE, or GAT.

## 5.2 Results

Tables 1 and 2 report test accuracy and relative message error, respectively. Figure 1 plots accuracy against the target edge ratio. These results from the small synthetic pilot show consistent evidences to support the soundness of our sampling method and mathematical analysis.

1. While sampling ratio grows from 30% to 99%, relative message errors in all three sampling methods fall consistently and significantly from about 1.2 to about 0.3. This illustrates the message error and edge budget trade-of and supports our prioritized goal of preserving message importance.

Table 1: Test accuracy of node classification on the synthetic pilot.
<table><tr><td>Target edge ratio</td><td>Uniform + tree</td><td> $\mathrm { M e s s a g e + t r e e }$ </td><td>Hybrid + tree</td></tr><tr><td>30%</td><td> $0 . 6 5 1 \pm 0 . 0 6 7$ </td><td> $0 . 6 5 4 \pm 0 . 0 7 0$ </td><td> $0 . 6 6 3 \pm 0 . 0 5 6$ </td></tr><tr><td>40%</td><td> $0 . 6 7 8 \pm 0 . 0 5 8$ </td><td> $0 . 6 7 4 \pm 0 . 0 5 3$ </td><td> $0 . 6 7 1 \pm 0 . 0 5 8$ </td></tr><tr><td>50%</td><td> $0 . 6 6 9 \pm 0 . 0 5 3$ </td><td> $0 . 6 7 0 \pm 0 . 0 5 3$ </td><td> $0 . 6 7 6 \pm 0 . 0 5 2$ </td></tr><tr><td>60%</td><td> $0 . 6 7 4 \pm 0 . 0 6 1$ </td><td> $0 . 6 7 8 \pm 0 . 0 5 3$ </td><td> $0 . 6 6 8 \pm 0 . 0 6 0$ </td></tr><tr><td>70%</td><td> $0 . 6 7 9 \pm 0 . 0 5 4$ </td><td> $0 . 6 7 3 \pm 0 . 0 5 6$ </td><td> $0 . 6 7 8 \pm 0 . 0 6 3$ </td></tr><tr><td>80%</td><td> $0 . 6 7 9 \pm 0 . 0 5 5$ </td><td> $0 . 6 7 8 \pm 0 . 0 5 3$ </td><td> $0 . 6 8 0 \pm 0 . 0 5 2$ </td></tr><tr><td>90%</td><td> $0 . 6 7 8 \pm 0 . 0 5 3$ </td><td> $0 . 6 7 9 \pm 0 . 0 5 5$ </td><td> $0 . 6 8 0 \pm 0 . 0 5 1$ </td></tr><tr><td>95%</td><td> $0 . 6 8 3 \pm 0 . 0 5 4$ </td><td> $0 . 6 8 1 \pm 0 . 0 5 5$ </td><td> $0 . 6 8 1 \pm 0 . 0 5 3$ </td></tr><tr><td>99%</td><td> $0 . 6 8 3 \pm 0 . 0 5 6$ </td><td> $0 . 6 8 3 \pm 0 . 0 5 6$ </td><td> $0 . 6 8 1 \pm 0 . 0 5 5$ </td></tr></table>

Table 2: Relative one-step message error $\| L _ { H } X - L X \| _ { F } / \| L X \| _ { F }$ on the synthetic pilot.
<table><tr><td>Target edge ratio</td><td> $\mathrm { U n i f o r m + t r e e }$ </td><td> $\mathrm { M e s s a g e + t r e e }$ </td><td> $\mathrm { H y b r i d } + \mathrm { t r e e }$ </td></tr><tr><td>30%</td><td> $1 . 2 1 0 \pm 0 . 1 6 9$ </td><td> $1 . 2 1 6 \pm 0 . 1 7 3$ </td><td> $1 . 2 0 5 \pm 0 . 1 5 3$ </td></tr><tr><td>40%</td><td> $0 . 7 1 6 \pm 0 . 0 4 6$ </td><td> $0 . 6 9 2 \pm 0 . 0 4 3$ </td><td> $0 . 7 1 1 \pm 0 . 0 4 9$ </td></tr><tr><td>50%</td><td> $0 . 5 2 3 \pm 0 . 0 2 7$ </td><td> $0 . 5 0 7 \pm 0 . 0 2 3$ </td><td> $0 . 5 1 2 \pm 0 . 0 3 0$ </td></tr><tr><td>60%</td><td> $0 . 3 9 7 \pm 0 . 0 1 7$ </td><td> $0 . 3 8 0 \pm 0 . 0 1 6$ </td><td> $0 . 3 8 7 \pm 0 . 0 1 8$ </td></tr><tr><td>70%</td><td> $0 . 3 0 7 \pm 0 . 0 1 5$ </td><td> $0 . 2 9 0 \pm 0 . 0 1 1$ </td><td> $0 . 2 9 6 \pm 0 . 0 1 4$ </td></tr><tr><td>80%</td><td> $0 . 2 2 8 \pm 0 . 0 1 1$ </td><td> $0 . 2 0 7 \pm 0 . 0 0 8$ </td><td> $0 . 2 1 8 \pm 0 . 0 1 3$ </td></tr><tr><td>90%</td><td> $0 . 1 4 8 \pm 0 . 0 1 1$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 7$ </td><td> $0 . 1 3 5 \pm 0 . 0 1 0$ </td></tr><tr><td>95%</td><td> $0 . 0 9 9 \pm 0 . 0 0 9$ </td><td> $0 . 0 7 8 \pm 0 . 0 0 8$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 8$ </td></tr><tr><td>99%</td><td> $0 . 0 3 9 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 7 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 3 \pm 0 . 0 0 6$ </td></tr></table>

2. While sampling ratio grows from 30% to 99%, test accuracies in all three sampling methods remain in a narrow range with no loss but some minor gain. This supports out goal of not to sacrifice accuracy for message importance.

3. In both the message score and the accuracy tests, performance of the proposed hybrid sampling lies in between of that of the message-preserving only sampling and uniform random sampling. This meets our design expectation as well as mathematical proofs.

These experimental results, in addition to the theoretic analysis, strongly support that the heuristic approach to preserve both structural features and message importance is a correct direction worth further exploration. However, honestly, most data from our modest pilot are not significant enough to compete with peer work. We believe that this on one hand is due to the limitation of the quality and scale of our synthetic data, while on the other hand due to the preliminary simple design of our proposal.

## 6 Conclusion and Future Work

This paper studies edge sampling for approximating the one-step action of the combinatorial Laplacian on a fixed input feature matrix. By expressing the Laplacian message as a sum of edge-wise contributions, we derive an unbiased estimator and its exact expected squared error. With a fixed spanning-tree backbone, we characterize the minimum message error achievable at each expected edge budget and the corresponding error frontier. We also define a tree-based hybrid sampler that combines feature-conditioned message scores with efective-resistance leverage scores. An exact excess-error decomposition quantifies the cost of this hybrid allocation relative to the message-optimal allocation at the same budget. The controlled synthetic experiments provide preliminary empirical evidence consistent with the analysis. Together, these results provide a sound approach to study how edge selection afects one-step messages while retaining graph connectivity.

Future work will address these directions. First, the method should be evaluated on larger public attributedgraph benchmarks and integrated into standard GCN, GraphSAGE, and GAT pipelines. Such evaluations should report predictive performance as well as graph construction and propagation costs. Second, the analysis should be extended beyond the one-step action of the combinatorial Laplacian on fixed input features to normalized propagation operators and the evolving hidden representations of multi-layer networks. Developing guarantees over a feature-generated Krylov space is a related direction. Finally, we shall further explore alternative sampling strategies and conduct systematic experiments across graph sizes, feature conditions, and mixture parameters, with uncertainty estimates.

![](images/b070e3ae300cd49c22ea0055bc90e12a0fa0baeb3146aa35b5c7dcc16e4af1fa.jpg)  
Figure 1: Test accuracy versus target edge ratio. Markers show the mean at each of the eight tested ratios; error bars show one standard deviation. Dashed curves are quadratic least-squares fits using all eight means for each sampler and are descriptive only. The dotted horizontal line is the full-graph mean.

## References

[1] Joshua Batson, Daniel A. Spielman, and Nikhil Srivastava. Twice-ramanujan sparsifiers. SIAM Journal on Computing, 41(6):1704–1721, 2012. doi: 10.1137/090772873.

[2] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, Cambridge, 2004.

[3] Jie Chen, Tengfei Ma, and Cao Xiao. FastGCN: Fast learning with graph convolutional networks via importance sampling. In International Conference on Learning Representations, 2018.

[4] Yuhan Chen, Haojie Ye, Sanketh Vedula, Alex Bronstein, Ronald Dreslinski, Trevor Mudge, and Nishil Talati. Demystifying graph sparsification algorithms in graph properties preservation. Proceedings of the VLDB Endowment, 2024.

[5] Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-GCN: An eficient algorithm for training deep and large graph convolutional networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 257–266, 2019.

[6] Siddhartha Shankar Das, Naheed Anjum Arafat, Muftiqur Rahman, S. M. Ferdous, Alex Pothen, and Mahantesh M. Halappanavar. SGS-GNN: A supervised graph sparsification method for graph neural networks. arXiv preprint arXiv:2502.10208, 2025.

[7] Haipeng Ding, Zhewei Wei, and Yuhang Ye. Large-scale spectral graph neural networks via laplacian sparsification. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 213–223, 2025. doi: 10.1145/3690624.3709241.

[8] Mucong Ding, Yinhan He, Jundong Li, and Furong Huang. Spectral greedy coresets for graph neural networks. arXiv preprint arXiv:2405.17404, 2024.

[9] David Eppstein, Zvi Galil, Giuseppe F. Italiano, and Amnon Nissenzweig. Sparsification—a technique for speeding up dynamic graph algorithms. Journal of the ACM, 44(5):669–696, 1997. doi: 10.1145/265910.265914.

[10] Wai Shing Fung, Ramesh Hariharan, Nicholas J. A. Harvey, and Debmalya Panigrahi. A general framework for graph sparsification. SIAM Journal on Computing, 48(4), 2019. doi: 10.1137/16M1091666.

[11] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, volume 30, 2017.

[12] Mohammad Hashemi, Shengbo Gong, Juntong Ni, Wenqi Fan, B. Aditya Prakash, and Wei Jin. A comprehensive survey on graph reduction: Sparsification, coarsening, and condensation. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 8058–8066, 2024. doi: 10.24963/ijcai.2024/891.

[13] Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph representation learning. In Advances in Neural Information Processing Systems, volume 31, 2018.

[14] David R. Karger. Random sampling in cut, flow, and network design problems. Mathematics of Operations Research, 24(2):383–413, 1999. doi: 10.1287/moor.24.2.383.

[15] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

[16] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. DropEdge: Towards deep graph convolutional networks on node classification. In International Conference on Learning Representations, 2020.

[17] Xu Shen, Pietro Liò, Lintao Yang, Ru Yuan, et al. Graph rewiring and preprocessing for graph neural networks based on efective resistance. IEEE Transactions on Knowledge and Data Engineering, 36(11): 6330–6343, 2024. doi: 10.1109/TKDE.2024.3397692.

[18] Daniel A. Spielman and Nikhil Srivastava. Graph sparsification by efective resistances. SIAM Journal on Computing, 40(6):1913–1926, 2011. doi: 10.1137/080734029.

[19] Daniel A. Spielman and Shang-Hua Teng. Spectral sparsification of graphs. SIAM Journal on Computing, 40 (4):981–1025, 2011. doi: 10.1137/08074489X.

[20] Rakshith S. Srinivasa, Cao Xiao, Lucas Glass, Justin Romberg, and Jimeng Sun. Fast graph attention networks using efective resistance based graph sparsification. arXiv preprint arXiv:2006.08796, 2020.

[21] Jana Vatter, Maurice L. Rochau, Ruben Mayer, and Hans-Arno Jacobsen. Size does (not) matter? sparsification and graph neural network sampling for large-scale graphs. In Proceedings of the 3rd International Workshop on Large-Scale Graph Data Analytics (LSGDA), VLDB Workshops, 2024.

[22] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.

[23] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 6861–6871, 2019.

[24] Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. GraphSAINT: Graph sampling based inductive learning method. In International Conference on Learning Representations, 2020.

[25] Guibin Zhang, Xiangguo Sun, Yanwei Yue, Chonghe Jiang, Kun Wang, Tianlong Chen, and Shirui Pan. Graph sparsification via mixture of graphs. In International Conference on Learning Representations, 2025.

[26] Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. Layer-dependent importance sampling for training deep and large graph convolutional networks. In Advances in Neural Information Processing Systems, volume 32, 2019.