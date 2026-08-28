# Inductive Correlation Clustering with Graph Neural Networks

Francesco Paolo Nerini Sapienza University of Rome, Italy nerini@diag.uniroma1.it

Arijit Khan Bowling Green State University, OH, USA arijitk@bgsu.edu

## Abstract

Correlation Clustering (CC) is a natural formulation of clustering in combinatorial optimization, which uses a graph representation of the input and does not require a pre-specified number of clusters. Given � objects and a pairwise similarity function, the goal is to cluster the objects so that similar objects are put in the same cluster and dissimilar objects are put in diferent clusters. Despite its versatility, existing CC algorithms sufer from significant scalability issues and are inherently transductive: i.e., the algorithm must be executed from scratch for any new problem instance.

## CCS Concepts

In this work, we bridge this gap by leveraging Graph Neural Networks (GNNs) to solve Inductive Correlation Clustering, a novel generalization of the CC problem designed to handle unseen graph instances. By learning to exploit common structural patterns and node features during training, our framework generalizes to new graphs drawn from the same distribution with minimal computational overhead with respect to standard algorithms. We demonstrate the efectiveness and scalability of our approach through extensive experiments. Our framework not only excels in the inductive setting, e.g., lowering the inference time up to 5 order of magnitude, while maintaining an approximation ratio within 10% of the best baseline solution, but also achieves competitive results on standard (transductive) CC benchmarks. Finally, we showcase a practical application of our framework as a learnable pooling mechanism for graph classification. Our results indicate that our method serves as an eficient pooling layer, enhancing the ability of GNNs to capture hierarchical structural information in networks.

• Computing methodologies → Neural networks; • Information systems → Clustering.

## Keywords

Francesco Bonchi Intesa Sanpaolo AI Research, Italy francesco.bonchi@intesasanpaolo.com

Correlation Clustering, Graph Neural Networks, Graph Pooling

## 1 Introduction

André Panisson Intesa Sanpaolo AI Research, Italy andre.panisson@intesasanpaolo.com

Clustering is a foundational data analysis task aimed at grouping similar objects while separating dissimilar ones. A natural formulation of such objective is provided, in graph terminology, by Correlation Clustering (CC) [6, 17]: given an unweighted graph � = (�, �), where the set of nodes � represents the objects to be clustered and edges in � connect similar objects, the goal is to partition � into clusters so to minimize the total number of “disagreements”, defined as the sum of missing edges within clusters and existing edges between clusters. Such optimization problem naturally extends to non-binary, symmetric similarity functions, and to settings where the similarity information is incomplete, i.e., no penalty needs to be paid for some pairs of objects irrespective of whether they are placed together or not. A key feature of CC is that, in contrast to many other popular clustering techniques (such as �-means), it does not require the number of clusters to be given as input, instead the optimal number of clusters is naturally found by the process. Since it only requires as input a notion of similarity, CC is particularly appealing for the task of clustering structured ob jects, where the similarity function is domain-specific and can be designed or learned. Thanks to its broad generality, CC has been studied under many variants (see, e.g., [17, 18]), and in a multitude of downstream applications ranging from duplicate detection and similarity joins [31, 40, 50], to biology [10, 29] and computer vision, specifically to the problem of image segmentation [16, 47, 69].

However, CC faces significant challenges. Its general formulation is APX-hard [22, 31], and existing approximation algorithms often scale poorly. Many rely on linear programs with exhaustive constraint sets, making them infeasible for large graphs. Furthermore, traditional CC algorithms typically ignore node features, overlooking rich information that could better identify latent structures within the graph. Finally, traditional CC is inherently transductive, requiring to be re-run for each new problem instance, without the possibility of generalizing to entirely new instances.

Contributions. In this paper, we introduce Inductive Correlation Clustering, a new formulation of CC which can generalize to unseen instances. Hierarchical pooling, entity resolution, and subgraph detection are examples of tasks that can benefit from inductive clustering models. In hierarchical pooling, graph data must be compressed into meaningful “super-nodes”, for example, grouping atoms into functional units for molecular property prediction [65], or segment point clouds for computer vision tasks [23]. An entity resolution task, on the other hand, resolve identities across databases through relational clustering, which is vital for many dataset deduplication applications [24]. Subgraph detection can instead help identifying suspicious communities in transaction monitoring [59, 70], or groups on social networks [35]. In all these scenarios, graphs are processed in batches or continuous streams during both training and inference. Solving a CC instance from scratch for every unique graph in a batch is computationally prohibitive compared to inductive methodologies, especially for larger graphs. By adopting an inductive, scalable approach based on CC, we can perform these tasks even with very little information regarding the types of relationships across elements or the number of underlying clusters, while providing the speed and generalization required for largescale or real-time deployments on multiple graphs.

Graph Neural Networks (GNNs) [48, 57] have recently emerged as efective tools for tackling combinatorial optimization problems [20, 58]. Their key strength lies in generalizing across diverse problem instances [45, 55] by capturing structural patterns and exploiting node-level features. This makes GNN-based approaches strong candidates for Inductive Correlation Clustering.

We thus propose a framework to train GNNs for solving Inductive Correlation Clustering problems. By employing a loss function based on a diferentiable relaxation of the standard Correlation Clustering objective, our framework can learn to optimize via stochastic gradient descent in an unsupervised manner, allowing for an eficient training and inference. To address the fundamental diferences between transductive and inductive settings, our framework introduces two distinct architectural variants. The inductive variant, once trained, generalizes to previously unseen graph instances, producing solutions of comparable quality to traditional algorithms while being orders of magnitude faster. We demonstrate the utility of Inductive Correlation Clustering by applying it to graph pooling. As mentioned before, pooling is a critical technique for deep learning, allowing models to capture high-level structural information, increase the receptive field, and reduce computational overhead by pruning edges and coarsening the graph [66]. Our experiments indicate that our methodology serves as a highly efective pooling technique, competitive with state-of-the-art approaches.

In summary, we make the following contributions:

• Novel problem formulation: We introduce the Inductive Correlation Clustering problem, a generalization of Correlation Clustering that leverages graph structure and node features to infer clustering solutions on unseen graphs drawn from the same distribution (§3).

• Scalable framework: We propose a scalable GNN-based training framework designed specifically to solve Correlation Clustering instances, which we dub CC-GNN, featuring a transductive (NodeGNN) and an inductive (LinkGNN) variant (§4).

• Empirical validation: We demonstrate that LinkGNN provides high-quality and computationally eficient solutions for the inductive setting, while NodeGNN excels on standard Correlation Clustering instances (§5-§6).

• Downstream application: We showcase the practical utility of our framework as a learnable pooling layer for deep graph learning architectures (§6.4).

Our work, bridging two apparently remote areas, such as combinatorial optimization for clustering and deep learning on graphs, contributes to the growing body ofliterature showing the successful usage of GNNs in combinatorial optimization problems [20, 58].

## 2 Related Work

Correlation clustering. Correlation Clustering is a classical opti mization problem with a rich theoretical history [6, 17, 22]. Much of the existing literature focuses on the theoretical development of approximation algorithms [19, 21, 26]. However, implementations capable of providing high-quality approximations while scaling to large datasets remain scarce or are limited to simplified cases [7, 11, 62]. One of the seminal works introduced KwikCluster [4], a combinatorial algorithm providing a 3-approximation guarantee in expectation for unweighted, complete graphs. More recently, [7] proposed ModifiedPivot, which improves this bound to $3 - \epsilon _ { 0 }$ for some $\epsilon _ { 0 } \ > \ 0$ on the same type of graphs. Both algorithms have been shown to maintain their approximation guarantees in dynamic settings, where the graph undergoes edge insertions or deletions [9, 28]. A complementary line of research [22, 31] tackles CC by solving a Linear Programming (LP) relaxation followed by a rounding procedure. While these methods often yield tighter approximation bounds [19, 27], they need solving LPs with an extremely large number of constraints. To apply these techniques to real-world graphs, additional approximations are required to maintain computational feasibility [11, 62].

Online [25, 51], active [36, 49] streaming [3, 8, 21], and dynamic [7, 9, 28] variants of CC have been studied extensively: these variants typically assume a continuous flow of node/edge modifications to a single problem instance. In contrast, our work focuses on the inductive setting: learning from a distribution of graphs to cluster entirely new, previously unseen instances with minimal latency.

Inductive clustering on graphs. Clustering methodologies can be categorized into transductive (i.e., must be run from scratch on unseen problem instances) and inductive approaches (i.e., able to learn a generalizable function that can be applied to unseen problem instances) [52]. Within the domain of graph-structured data, most standard clustering techniques are transductive, e.g., Leiden algorithm [60], spectral clustering methods [54], and standard Correlation Clustering algorithms. Inductive clustering methods leverage node features to generalize across instances [42]. The most significant methods in this area use GNNs [48, 57], with the shift toward the inductive setting popularized by GraphSAGE [38]. Notable examples include DMoN [61], HOSC [32], GTV [39], and MinCutPool [14]. While all of these methods require setting a fixed or maximum number of clusters, our Inductive Correlation Clustering formulation allows to select the number of clusters according to the structure and density of the input graph.

Graph pooling. A primary application of inductive approaches is graph pooling [34, 66], where learnable clustering serves to coarsen graphs. This process not only manages computational complexity but also enhances the expressivity ofthe resulting GNN architecture [15]. We compare with many of these methods in a pooling task in §6.4. Besides, we also compare against �-MIS [5], MaxCutPool [1], and SEP [67] which allows to not set a fixed number of output nodes. However, these are not clustering techniques, since they coarsen graphs by selecting subsets of representative nodes instead of identifying communities; �-MIS is not even inductive, as it must run separately on each new graph and does not use node features.

## 3 Inductive Correlation Clustering

Consider an unweighted graph $G = \left( V , E \right)$ . The objective of Correlation Clustering is to partition the set of nodes � into clusters that minimize the total number of “disagreements”, defined as the sum of missing edges within clusters and existing edges between clusters. Let $\mathbf { C } \in \{ 0 , 1 \} ^ { | V | \times K }$ be a cluster assignment matrix, where each node is assigned to exactly one of � clusters (� is not fixed a priori). The cost of a clustering C is formalized as:

$$
\mathrm { c o s t } ( { \mathbf { C } } ) = \sum _ { ( i , j ) \in E } \sum _ { c = 1 } ^ { K } \mathbf { C } _ { i c } \big ( 1 - { \mathbf { C } } _ { j c } \big ) + \sum _ { ( i , j ) \notin E } \sum _ { c = 1 } ^ { K } \mathbf { C } _ { i c } \mathbf { C } _ { j c } .\tag{1}
$$

This objective can be expressed more intuitively by introducing the co-clustering matrix $\mathbf { M } = \mathbf { C } \mathbf { C } ^ { \top }$ , where $\mathbf { M } _ { i , j } = 1$ if nodes � and � are in the same cluster, and 0 otherwise. Therefore, the cost function simplifies to:

$$
\mathrm { c o s t } _ { G } ( \mathbf { M } ) = \sum _ { ( i , j ) \in E } \left( 1 - \mathbf { M } _ { i , j } \right) + \sum _ { ( i , j ) \notin E } \mathbf { M } _ { i , j } .\tag{2}
$$

Minimizing Eq. (2) is known to be APX-hard [6]. To address this, the problem is often relaxed by allowing $\mathbf { M } _ { i j } \in [ 0 , 1 ]$ subject to metric constraints (i.e., triangle inequality $1 - \mathbf { M } _ { i j } \leq \left( 1 - \mathbf { M } _ { i k } \right) + \left( 1 - \mathbf { M } _ { j k } \right)$ and symmetry $\mathbf { M } _ { i j } = \mathbf { M } _ { j i } )$ . This relaxation forms the basis of Linear Programming approaches [11, 22] and motivates the continuous relaxations we adopt for our GNN framework.

In the inductive setting, we assume that graphs are drawn from a distribution D. A graph instance is defined as a tuple $G = \left( V , E , \mathbf { X } \right)$ where $\mathbf { X } \in \mathbb { R } ^ { | V | \times \bar { F } }$ represents node features. We aim to learn a parameterized function $f _ { \theta } \left( \mathrm { e . g . , a G N N } \right)$ that maps a graph instance to a clustering assignment. Given a training graph $G _ { t r }$ (or a set of graphs), the goal is to find the parameters $\theta ^ { * }$ that minimize the clustering cost on the training structure:

$$
\boldsymbol { \theta } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \theta } } \cos \mathrm { t } _ { G _ { t r } } \left( f _ { \boldsymbol { \theta } } ( V _ { t r } , E _ { t r } , \mathbf { X } _ { t r } ) \right) .\tag{3}
$$

The ultimate objective is to generalize this solution to minimize the cost on a disjoint, previously unseen test graph $G _ { t e } \sim \mathcal { D }$ :

$$
\mathrm { O b j e c t i v e : \ m i n \ c o s t { \bf \Sigma } } _ { G _ { t e } } \left( f _ { \theta ^ { * } } \left( V _ { t e } , E _ { t e } , { \bf X } _ { t e } \right) \right) .\tag{4}
$$

We show in the following sections how we learn the model $f _ { \theta } ,$ and demonstrate via experiments the efectiveness ofour approach.

## 4 CC-GNN Framework

In this section, we present CC-GNN, a framework for Correlation Clustering in transductive and inductive settings. We distinguish between two optimization approaches: node-wise methods, which optimize a relaxed cluster assignment matrix $\mathbf { C } \in [ 0 , 1 ] ^ { | V | \times K }$ , and edge-wise methods, which optimize the relaxed co-clustering matrix $\mathbf { M } \in [ 0 , 1 ] ^ { | V | \times | V | }$

Formally, a GNN $f _ { \theta }$ maps node features X and the edge set � to a latent embedding matrix $\mathbf { O } \in \mathbb { R } ^ { | V | \times F }$

$$
\mathbf { O } = \mathrm { G N N } _ { \theta } ( \mathbf { X } , E ) .\tag{5}
$$

The initialization of X defines the model’s operating mode. In transductive settings lacking informative attributes, we follow [2, 56] and use random features as unique identifiers to enhance expressivity. While structural features $( \mathrm { e . g . }$ , Laplacian eigenvectors) are an alternative, they ofered no empirical advantage in our preliminary experiments. Conversely, for inductive generalization to unseen graphs, we utilize node attributes to capture transferable patterns.

The framework is organized into three distinct components. First, we introduce a pivot-based sampling strategy that ensures scalability to large graphs (§4.1). Second, we describe a node-wise optimization approach for standard, transductive Correlation Clustering (§4.2). Finally, we present our primary contribution: an inductive, link-wise formulation that enables the model to infer clustering solutions on entirely new graph instances without re-training (§4.3-§4.4).

Algorithm 1 Pivot-based Sampling   
Require: Graph $G = ( X , E )$ , signed adjacency W, # pivots �   
Ensure: Batched features $\mathbf { X } _ { B } ,$ edges $E _ { B } ,$ weight matrix $\mathbf { W } _ { B }$   
1: Select a set of � random pivots $P \subset V$   
2: $B  P$   
3: for each $v \in P$ do   
4: $B  B \cup N ( v )$   
5: end for   
6: $E _ { B } = \{ ( u , v ) \in E \mid u , v \in B \}$   
7: $\mathbf { X } _ { B } = \left\{ \mathbf { X } _ { u } \mid u \in B \right\}$   
8: $\mathbf { W } _ { B } = \mathbf { W } [ B , B ] \ \cdot \mathbf { \Phi }$ Submatrix induced by node set $B ^ { * } \backslash$   
9: return $\mathbf { X } _ { B } , E _ { B } , \mathbf { W } _ { B }$

Algorithm 2 NodeGNN Training   
Require: Graph $G = ( X , E )$ , signed adjacency W, epochs �, maxi   
mum number of clusters �, pivots number �.   
Ensure: Optimal GNN parameters $\theta ^ { * }$ , cluster assignments C   
1: Initialize GNN parameters �   
2: for epoch = 1 to � do   
3: $\mathbf { X } _ { B } , E _ { B } , \mathbf { W } _ { B } \gets$ PivotSampling(�, W, �) \<sup>∗</sup> Call Alg. 1 <sup>∗</sup>\   
4: $\mathbf { O } = \mathbf { G N N } _ { \theta } ( X _ { B } , E _ { B } )$   
5: $\mathbf { C } _ { i } = \mathrm { s o f t m a x } ( \mathbf { O } _ { i } )$   
6: Calculate loss: $\mathcal { L } ( \mathbf { C } ) = \| \mathbf { W } - \mathbf { C C } ^ { \intercal } \| _ { 2 } ^ { 2 } - \| \mathbf { C C } ^ { \intercal } \| _ { 2 } ^ { 2 }$   
7: Backpropagate $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$ and update �   
8: end for   
9: $\theta ^ { * } \gets \theta$   
10: Generate final embeddings ${ \bf O } ^ { * } = \mathrm { G N N } _ { \theta ^ { * } } ( X , E )$   
11: for each node $i \in V$ do   
12: $C _ { i } = \mathrm { a r g m a x } _ { j \in \{ 1 , . . . , K \} } { \bf O } _ { i j } ^ { * }$   
13: end for   
14: return $\theta ^ { * } , c$

## 4.1 Pivot-based Batch Sampling

Since message-passing operations scale poorly on large graphs, optimizing over the full training graph $G _ { t r }$ is often infeasible. Instead, we perform backpropagation on a subgraph induced by a node batch $B \subseteq V _ { t r }$ . As detailed in Algorithm 1, we utilize a sampling strategy inspired by KwikCluster [4]: at each epoch we select � random pivots � and include all their direct neighbors to form B.

In the original KwikCluster algorithm, each pivot and its neighbors are iteratively extracted to form clusters. By adopting this logic during training, we identify potential clusters and "refine" them by optimizing the model loss on these localized communities rather than the entire graph. This approach significantly reduces computational overhead, lowering the complexity of each GNN call from �(�) to $O \big ( \frac { M } { | V _ { t r } | } E \big )$ . This provides a substantial performance advantage when � ≪ $\left| V _ { t r } \right|$ . As we will see in §6.5, optimizing over a batch also drastically reduces the computational complexity of the loss computation, reducing training time by up to a factor of 40 and memory consumption by a factor of 4.

## 4.2 Transductive Node-wise Optimization

To train GNNs for Correlation Clustering, we relax the cost function into a continuous, diferentiable loss. A key advantage of deep learning here is the ability to satisfy constraints implicitly through model architecture rather than explicit enforcement. This makes local optimization feasible even for large graphs where the number of constraints would otherwise be prohibitive. We focus first on the node-wise approach, and call this method NodeGNN; its training procedure is detailed below and summarized in Algorithm 2.

In order to optimize Eq. 1, we first set the value of � arbitrarily and relax the node assignment matrix as $\mathbf { C } \in [ 0 , 1 ] ^ { | V | \times K }$ . This restricts the solution space to partitions with at most � clusters, a constraint absent from the original formulation with strong repercussions, as we will see.

In this setting, the cluster assignment matrix relaxation has the constraints $\textstyle \sum _ { j } \mathbf { C } _ { i j } = 1 , \forall i \in V$ . If we set the latent embeddings dimension � equal to �, these constraints are easily satisfied if we define C = softmax(O), with O the node embedding matrix output of a GNN (lines 5-6 of Algorithm 2).

To construct a diferentiable objective, we first define the signed adjacency matrix W as:

$$
\mathbf { W } _ { i j } = \left\{ { \begin{array} { l l } { + 1 , } & { { \mathrm { i f ~ } } i \neq j { \mathrm { ~ a n d ~ } } ( i , j ) \in E } \\ { - 1 , } & { { \mathrm { i f ~ } } i \neq j { \mathrm { ~ a n d ~ } } ( i , j ) \notin E } \\ { 0 , } & { { \mathrm { i f ~ } } i = j . } \end{array} } \right.\tag{6}
$$

Introducing W in Eq. 1 and simplifying we can reduce it to:

$$
\mathrm { c o s t } _ { G } ( { \bf C } ) = - \frac { 1 } { 2 } \sum _ { i j } { \bf W } _ { i j } \sum _ { c } { \bf C } _ { i c } { \bf C } _ { j c } + | { \cal E } |\tag{7}
$$

$$
\mathbf { \Sigma } = - \frac 1 2 \mathrm { T r a c e } ( \mathbf { W C C ^ { \top } } ) + \mathbf { c o n s t } .\tag{8}
$$

Exploiting the expansion of the squared Frobenius norm (the entrywise 2-norm), we can also cast the trace as:

$$
- \mathrm { T r a c e } ( { \bf W } { \bf C } { \bf C } ^ { \top } ) = \frac { 1 } { 2 } \left( { \| { \bf W } - { \bf C } { \bf C } ^ { \top } \| _ { 2 } ^ { 2 } } - { \| { \bf C } { \bf C } ^ { \top } \| _ { 2 } ^ { 2 } } - { \| { \bf W } \| _ { 2 } ^ { 2 } } \right) .\tag{9}
$$

Discarding the constant term $\| \mathbf { W } \| _ { 2 } ^ { 2 }$ , we arrive at the final loss function, which favours the alignment between W and $\mathbf { C } \mathbf { C } ^ { \top }$ :

$$
\mathcal { L } ( \mathbf { C } ) = \| \mathbf { W } - \mathbf { C C ^ { \intercal } } \| _ { 2 } ^ { 2 } - \| \mathbf { C C ^ { \intercal } } \| _ { 2 } ^ { 2 } .\tag{10}
$$

Computing this loss over the full graph has time complexity of $O ( | V | ^ { 2 } K )$ . Consequently, a regime where � ≪ |� | significantly reduces this overhead, though at the expense of restricting the reachable solutions.

At inference time, instead, we assign clusters through an argmax on the output $\mathbf { O } ^ { * }$ from the GNN (lines 12-14 of Algorithm 2).

Both the trace and matrix formulations facilitate local optimization of C; however, Eq. 9 specifically frames Correlation Clustering as a matrix factorization problem. The target matrix W is approxi mated by CC<sup>⊺</sup>, with ∥CC<sup>⊺</sup> ∥<sup>2</sup>serving as a regularization term. GNN architectures are well suited here, as W and the rows of C can naturally correspond to a weighted signed graph and �-dimensional node embeddings O, respectively. By leveraging the graph structure to learn these node embeddings, GNNs efectively perform the factorization required to optimize the clustering objective [44, 71].

Since this approach stems from fixing the maximum number of cluster � and optimizing the corresponding matrix $\mathbf { C } ,$ as the graph size increases, the approximation quality drops significantly unless � is scaled accordingly. Additionally, this matrix factorization approach is strictly transductive. Because the GNN learns to factorize the specific matrix W of the training graph, it fails to generalize to unseen data. Even when applied to new graphs with node features drawn from the same distribution, it cannot transfer its learned representations and outputs incoherent clusters (see §6.3). To address these challenges, the following sections introduce an inductive, edge-wise optimization approach.

![](images/1e2791fc5e5b18ba30fb81c704d3944900312123265d911cc99c63f58fcd13eb.jpg)  
(a)

![](images/60679e000117c5aff3836012dc65649c2e440a83bb690ecd8c6dff5dc54f2f62.jpg)  
(b)

![](images/cf69130f68a676ee56232fcf9971b6d6559942ef098751624b7aad45d2d19ebb.jpg)  
(c)  
Figure 1: (a) A toy graph instance where the optimal CC solution is highlighted by colored circles (ground truth). The corresponding node embeddings generated by LinkGNN are projected onto the unit circle $\left( F \right) = 2 )$ . Before training (b), embeddings are randomly distributed. After training (c), the model separates the nodes into 5 distinct angular sectors.

## 4.3 Intuition on Inductive Optimization

To overcome the inherent limitations of the node-wise approach, we consider a link-wise optimization approach, i.e., we optimize with respect to the co-clustering matrix $\dot { \mathbf { M } } \in [ 0 , 1 ] ^ { | V | \times | V | }$

Before detailing the method, which we call LinkGNN, we illustrate our intuition with an example. Consider the graph in Figure 1a, where the optimal Correlation Clustering solution consists of four small communities (5 nodes each) and a single isolated node. To enable visualization, we use a GNN with an output dimension of $F = 2$ and project the resulting node embeddings onto a circle to constrain possible distances (in general applications, this is extended to a hypersphere of dimension �). The distances between these projected embeddings then define the relaxed co-clustering matrix M, as will be described in the following section.

Our loss acts similarly to a contrastive learning objective. Initially, a GNN produces node embeddings that are clustered tightly together (Figure 1b). Through training, the loss pulls connected nodes closer and pushes disconnected ones apart until a local optimum is reached. In our example, this process projects nodes into clearly identifiable clusters along the unit circle (Figure 1c). Posttraining, we retrieve the optimal solution by grouping nodes based on their proximity in the latent space. Because clustering depends on these distances rather than fixed assignments, the model is not restricted to a maximum number of clusters (�), allowing it to adapt inductively to graphs of varying scales.

## 4.4 Inductive Link-wise Optimization

In LinkGNN, we optimize a relaxed link-formulation of CC. When solving the link-wise formulation, we need a relaxation M with the metric constraints described in §3. To do so, we first normalize the

Inductive Correlation Clustering with Graph Neural Networks

node embeddings:

$$
\tilde { \mathbf { O } } _ { i } = \frac { \mathbf { O } _ { i } } { \lVert \mathbf { O } _ { i } \rVert _ { 2 } } , \ : \forall i \in V .\tag{11}
$$

We can then obtain matrix M by computing:

$$
\mathbf { M } _ { i j } = 1 - \frac { 1 } { 2 } \| \tilde { \mathbf { O } } _ { i } - \tilde { \mathbf { O } } _ { j } \| _ { 2 } ,\tag{12}
$$

where we use $\left\| \cdot \right\|$ to indicate the Euclidean norm. It is necessary to do this step both during training (lines 5-7 of Algorithm 3) and inference (lines 1-3 of Algorithm 4).

Thanks to the properties of the Euclidean norm, the matrix M is now symmetric $( \mathbf { M } _ { i j } = \mathbf { M } _ { j i } )$ and satisfies the triangular inequalities $1 - \mathbf { M } _ { i j } \leq \left( 1 - \mathbf { M } _ { i k } \right) + \left( 1 - \mathbf { M } _ { j k } \right)$ for all $i , j , k ,$ as required by the constraint in $\ S 3 .$ However, by normalizing the node embeddings $^ { \mathbf { 0 , } }$ we are efectively limiting the matrix M to describe distances between points on an � dimensional hypersphere of unit radius. This is an additional constraint on the possible values of the matrix M with respect to the classical relaxation of [22, 31]. We can then cast the loss function in $\operatorname { E q . }$ 10 in terms of the co-clustering matrix M obtaining:

$$
\begin{array} { r } { \mathcal { L } ( \mathbf { M } ) = \| \mathbf { W } - \mathbf { M } \| _ { 2 } ^ { 2 } - \| \mathbf { M } \| _ { 2 } ^ { 2 } . } \end{array}\tag{13}
$$

The loss still requires the computation of M, which has a computational complexity of $O ( | V | ^ { 2 } F )$ on the full graph: however, since we are using the pivot-based sampling, we are able to reduce it to $O ( | \mathcal { B } | ^ { 2 } F )$ . Since we can also set the latent dimension � to a much smaller value than previous maximum number of clusters $K ,$ , it is also significantly faster than the node-wise loss.

During inference, we obtain the clusters by applying another heuristic approach (Algorithm 4), where we define the matrix:

$$
\begin{array} { r } { \hat { \mathbf { M } } _ { i j } ^ { t } = \left\{ \begin{array} { l l } { 1 , \ } & { \mathrm { i f } \ ( i , j ) \in E \ \mathrm { a n d } \ \mathbf { M } _ { i j } \geq t } \\ { 0 , \ } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{14}
$$

Since the matrix $\hat { \mathbf { M } } ^ { t }$ acts as an adjacency matrix derived from the similarity matrix M, we define the output clusters as its connected components. Finally, because the threshold � is arbitrary, we evaluate a range of values � ∈ [0, 1] during training and select the one that minimizes the cost (Algorithm 3, lines 11-18). The corresponding � is then used for inference as well. Increasing � sparsifies the graph, leading to a higher number of output clusters as more connected components are formed. After training, we can then use this heuristic to cluster the original training graph (as we do when we limit to a transductive application of the method) or to new graphs entirely (allowing for inductive applications). The main drawback is that this heuristics, together with our relaxation, can introduce a larger approximation error than NodeGNN (as we observe experimentally in §6.1 and §6.2).

Since during inference we only need to compute distances $\mathbf { M } _ { i j }$ for existing edges, we significantly reduce the auxiliary graph’s density and greatly accelerate the inference (reducing the computation of M from $O ( | V | ^ { 2 } F )$ to $O ( | E | F ) )$ . Focusing on already connected nodes can also be theoretically justified by the following observation: if a cluster contains two communities not linked by any edge, separat ing them yield a lower cost than keeping them merged. Accounting for the matrix multiplications in the message passing operations, inference time for LinkGNN scales as $O ( | V | \bar { F ^ { 2 } } + \bar { | E | } F )$ . Furthermore, inference memory consumption scales as $O ( F ^ { 2 } + | V | F + | E | )$ .

Algorithm 3 LinkGNN Training   
Require: Graph $G = ( X , E )$ , signed adjacency W, epochs �, num  
ber of pivots �, lists of thresholds �   
Ensure: Trained GNN parameters $\theta ^ { * } ,$ , optimal threshold $t ^ { * }$   
1: Initialize GNN parameters $\theta$   
2: for epoch = 1 to $E _ { m a x }$ do   
3: $\mathbf { X } _ { B } , E _ { B } , \mathbf { W } _ { B } \gets$ PivotSampling(�, W, �) \<sup>∗</sup> Call Alg. 1 <sup>∗</sup>\   
4: $\mathbf { O } = \mathbf { G } \mathbf { N N } _ { \theta } ( X _ { B } , E _ { B } )$   
5: $\tilde { \mathbf { O } } _ { i } = \mathbf { O } _ { i } / \lVert \mathbf { O } _ { i } \rVert _ { 2 } \ \backslash ^ { * }$ Row-wise Normalization $^ * \backslash$   
6: $\begin{array} { r } { \mathbf { M } _ { i j } = 1 - \frac { 1 } { 2 } | | \tilde { \mathbf { O } } _ { i } - \tilde { \mathbf { O } } _ { j } | | _ { 2 } } \end{array}$   
7: $\mathcal { L } ( \mathbf { M } ) = \| \mathbf { \bar { W } } _ { B } - \mathbf { M } \| _ { 2 } ^ { 2 } - \| \mathbf { M } \| _ { 2 } ^ { 2 }$   
8: Update � via backpropagation of $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$   
9: end for   
10: \<sup>∗</sup>Best threshold selection<sup>∗</sup>\   
11: ������� $ \infty , t ^ { * }  0$   
12: for each $t \in T$ do   
13: C<sub>�</sub> ← Inference $( G , { \mathrm { G N N } } _ { \theta ^ { * } } , t ) \ \backslash ^ { * }$ Call Alg. 4 <sup>∗</sup>\   
14: �������� = Cost� (C<sub>�</sub>) \<sup>∗</sup> Evaluate cost with eq. 1 <sup>∗</sup>\   
15: if �������� < ������� then   
16: ������� ← ��������, $t ^ { * } \gets t$   
17: end if   
18: end for   
19: return $\theta ^ { * } , t ^ { * }$   
Algorithm 4 LinkGNN Inference   
Require: Graph $G = ( X , E )$ , trained GNN, threshold �   
Ensure: Cluster assignments $C _ { t }$   
1: $\mathbf { O } = \mathrm { G N N } ( X , E )$   
2: $\tilde { \mathbf { O } } _ { i } = \mathbf { O } _ { i } / \lVert \mathbf { O } _ { i } \rVert _ { 2 } \ \backslash ^ { * }$ Row-wise Normalization $^ * \backslash$   
3: $\begin{array} { r } { \mathbf { M } _ { i j } = 1 - \frac { 1 } { 2 } \| \tilde { \mathbf { O } } _ { i } - \tilde { \mathbf { O } } _ { j } \| _ { 2 } } \end{array}$   
$\begin{array} { r } { \hat { \mathbf { M } } _ { i j } ^ { t } = \left\{ \begin{array} { l l } { 1 , } \\ { 0 , } \end{array} \right. } \end{array}$ $\mathrm { i f } \left( i , j \right) \in E$ and $\mathbf { M } _ { i j } \geq t$   
4:   
otherwise   
5: $C _ { t } \gets ($ ConnectedComponents(M<sup>ˆ</sup> <sup>�</sup>)   
6: return $C _ { t }$

Initializing nodes with random feature vectors can be efective for transductive tasks, but it inherently limits generalization to unseen graphs. In such cases, the GNN memorizes node identities from the training set rather than learning transferable structural patterns. Consequently, with LinkGNN, node features (when available) or structure-based embeddings such as Node2Vec must be used. For pure Correlation Clustering objectives, Node2Vec embeddings are often preferable to raw node features, since original features may be entirely uncorrelated with graph structure, the sole determinant of the optimal solution.

## 4.5 Pooling Adaptation

Pooling applications require to modify our method slightly. We consider only the link-wise optimization approach, motivated by the experimental results on inductive applications (§6.3).

While originally we optimize only the CC cost, and select the threshold � accordingly, in pooling we must always be able to regulate the coarsening of the graph through an hyperparameter. Therefore, we introduce an edge pool percentile parameter $r \in [ 0 , 1 0 0 ]$

![](images/964cdb997b461228e444e5854632e1dce56180371e3b5829c025d232f4623816.jpg)  
(a)

![](images/641eff0303f6eba8e789dcf9dafcc36a51af1eea5632f43a500c7d4a5398fd6b.jpg)  
(b)  
Figure 2: Example of pooling on the graph of Figure 1 with $r = 2 0 \% . ( \mathbf { a } )$ Red dotted lines indicate the 20% of edges connecting node pairs with the greatest embedding distance. (b) The pooling operation merges nodes within the same connected components and then re-introduce their original connections: the graph is significantly compressed, retaining fewer than 15% of original edges.

We dynamically set the threshold � to the (100 − �)-th percentile of the entries in the similarity matrix M. This ensures that we retain only the strongest connections, efectively controlling the coarsening ratio. After inferring the clusters from M, we first obtain the coarsened nodes representations by performing a sum aggregation across the nodes in each individual clusters. Then, we link any two of the clusters if their constituent nodes shared at least one edge in the original graph. These choices on aggregation and link reconstruction are in line with most pooling frameworks [37]. Finally, because optimal pooling depends on both the CC objective and the downstream task loss, we use the node representations from the downstream model as input embeddings instead of the Node2Vec embeddings of LinkGNN. These embeddings are initialized directly from the original node features before being updated by the downstream model. The CC loss is also not backpropagated to the downstream model. We present an example in Figure 2.

## 5 Experimental Setup

To rigorously assess the proposed CC-GNN framework, we construct a unified experimental suite, targeting 4 research questions: RQ1 (Approximation quality): Does the proposed diferentiable relaxation enable the GNN to recover high-quality solutions relative to exact solvers on tractable instances?

RQ2 (Transductive performance): How does our framework compare against state-of-the-art combinatorial algorithms on standard Correlation Clustering benchmarks without node features?

RQ3 (Inductive generalization): Can the model exploit structural patterns and node features to generalize to unseen graphs without retraining, and does this yield inference-time eficiency gains?

RQ4 (Downstream utility): Is Inductive Correlation Clustering efective when used as a learnable pooling layer (CCPool) within deep graph learning architectures?

Reproducibility: Our code and data are available at https://github.   
com/FrappaN/ccgnn.

## 5.1 Datasets

We use a diverse collection of graph datasets categorized into three groups. Summary statistics for all datasets are reported in Table 1. Standard correlation clustering benchmarks: To validate our method against traditional combinatorial algorithms in the standard transductive setting, we use unweighted graphs without node features. These include social, citation, and email networks from the SNAP and SuiteSparse collections.

Table 1: Statistics for all datasets. Standard Correlation Clustering benchmark consist of individual graph instances, while the Graph-Level Tasks and Pooling benchmark consist of collections of graphs (# Nodes and # Edges report average values). We report node features and number of labels exclusively of datasets used in pooling experiments. In the case of OGBG-ppa, we use only the first 30, 000 graphs of the dataset.
<table><tr><td>Dataset # Graphs # Nodes # Edges # Features</td></tr><tr><td># Labels Standard Correlation Clustering Benchmarks</td></tr><tr><td>polblogs[30] 1 1,224 16,715</td></tr><tr><td>ca-GrQc[46] 1 5,241 14,484</td></tr><tr><td>ca-HepTh[46] 1 9,875 25,973</td></tr><tr><td>ca-AstroPh[46] 1 18,771 198,050</td></tr><tr><td>email-Enron[46] 1 36,692 183,831</td></tr><tr><td>cond-mat-2005[30] 1 39,577 175,691</td></tr><tr><td>Graph-Level Tasks and Pooling Benchmarks</td></tr><tr><td>EXPWL1[15] 3,000 76.96 186.46 1 2</td></tr><tr><td>MUTAG[53] 188 17.93 39.59 7 2</td></tr><tr><td>NCI1[53] 4,110 29.87 64.60 37 2</td></tr><tr><td>REDDIT-BINARY[53] 2,000 429.63 497.75 一</td></tr><tr><td>GitHub-Stargazers[53] 12,725 113.79 234.64 一</td></tr><tr><td>OGBG-molhiv[43] 41,127 25.50 27.50 9 2</td></tr><tr><td>OGBG-ppa[43] 30,000 243.40 2,266.1 一 一</td></tr></table>

Graph-level benchmarks: To evaluate the inductive generalization of our model, we use datasets consisting in collections of multiple graphs. These include synthetic datasets (EXPWL1, a synthetic dataset designed to test Weisfeiler-Leman expressivity), molecular benchmarks (MUTAG, NCI1, OGBG-molhiv), protein association networks (OGBG-ppa), and social networks (GitHub-Stargazers, REDDIT-Binary).

Graph classification benchmarks: For the graph pooling application, we employ a subset of the previous datasets, all of which have a corresponding graph classification task: EXPWL1, MUTAG, NCI1, and OGBG-molhiv.

## 5.2 Baselines

We compare our framework against two distinct sets of state-of-theart baselines, corresponding to the two primary problem settings.

Correlation clustering baselines. We compare with KwikCluster [4], the seminal randomized 3-approximation algorithm; ModifiedPivot [7], which improves local cluster adjustments; and CFP (CoverFlipPivot) [11], a combinatorial method for the generalized LambdaCC problem. For RQ1, we also compare with an exact Gurob solver.

Graph pooling baselines. To assess the utility of Inductive CC as a pooling layer (CCPool), we compare against established pooling methods selected from the benchmark in [13], including Deep Modularity Networks (DMON) [61], High-Order Spectral Clustering (HOSC) [32], Just-Balance Pooling (JBPool) [12], Asymmetric Cheeger Cut (ACC) [39], k-Maximal Independent Sets pooling (k-MIS) [5], and MaxCut Pooling [1].

## 5.3 Experimental Protocol

For all Correlation Clustering experiments, performance is measured using the solution cost as defined in Eq. 1, normalized relative to the minimum cost achieved across all methods.

We evaluate our models, NodeGNN (transductive node-wise) and LinkGNN (inductive link-wise), across five distinct scenarios. For all experiments, we always report mean over diferent runs, together with the standard error of mean.

Approximation quality analysis: To validate the correctness of our framework on tractable instances (i.e., small instances where exact optimization is feasible), we utilize synthetic graphs generated via Stochastic Block Models (SBMs) [41]. We fix the graph size at |�| = 36 and divide nodes into � equal-sized blocks. Connection probabilities are set to 0.8 for intra-block edges and 0.1 for interblock edges. We vary � from 2 to 36 (where � = 36 approximates a random graph) and generate 10 random instances per configuration, reporting the average performance across these 10 instances. As the total graph size is fixed, increasing � corresponds to a decrease in cluster size as well. We compare the cost of the GNN solution against the global optimum computed using the Gurobi exact solver.

Transductive optimization: We train and evaluate on the same graph instance to verify if the GNN can recover high-quality clustering solutions compared to combinatorial solvers. We run all methods 5 times each to average against the stochasticity of the methods (i.e., the training algorithm of NodeGNN/LinkGNN, and the random pivot sampling which is used in diferent flavors in all the methods). All results are normalized with the minimum achieved in each dataset across all methods and iterations. We initialize nodes with random vectors, instead of structured-oriented embeddings (such as Laplacian positional encodings [33]), as these did not show to improve the models’ results in preliminary experiments. Indeed, GNNs are typically more able to discern structural patterns when nodes are initialized with random features [2, 56].

Inductive generalization: To evaluate inductive clustering on unseen instances, we split the graph-level datasets into train (80%), validation (10%), and test (10%) sets. For LinkGNN, we adapt Algorithm 3 by using the validation set to select the optimal threshold for the test set. For each dataset, the models are trained on the graphs in the train set, and we report the performance on the graphs in the test set. As mentioned in §4.4, since the optimal solution depends only on the graph structure, we discard original node features and use Node2Vec embeddings instead. We also run the transductive methods Kwikcluster and ModifiedPivot on the test set to serve as lower bounds on solution quality. As in the transductive setup, we execute 5 runs per method using diferent random splits. Because these splits are identical across methods, we normalize each result by the minimum cost achieved on that specific split.

Graph pooling: We perform 10-fold cross-validation on the graph classification benchmarks. Here, LinkGNN is used as a pooling operator (“CCPool”) to coarsen the graph between GNN layers, selecting the clustering threshold to retain a fixed percentage of edges rather than minimizing clustering cost (§4.5). Since the optimal pooling depends on both the downstream task and the graph structure, we use the downstream model embeddings as input for CCPool. In line with standard benchmarks, we report average test accuracy, except for OGBG-molhiv, where ROC-AUC is used due to class imbalance. For all the datasets and methods, we tune their corresponding hyperparameters to achieve a pooling ratio of at most 10% of the original nodes, including the parameter � of CCPool.

Ablation studies: In §6.5, we conduct comprehensive ablation and sensitivity studies to analyze our design choices. First, we vary model complexity: more complex configurations evaluate deeper architectures (2–3 GCN layers with nonlinearities between layers) or alternative convolutions (GIN [68], GraphSAGE [38], GAT [63]), while simpler variants remove message passing entirely by using a single linear layer ("Linear") or trainable node embeddings ("EMBOnly"). Second, we assess the operational impact of pivot sampling, embedding normalization, and final cluster selection. Finally, we perform a sensitivity analysis on both threshold selection and the embedding dimension �.

## 5.4 Model Architectures and Hyperparameters

We implement NodeGNN and LinkGNN, using a simplified architecture consisting of a single Graph Convolutional Network layer without nonlinearities. We do the same for CCPool. We provide additional implementation details in the following.

Architectures: Both NodeGNN and LinkGNN are composed of a single GCN layer [48], without nonlinearities. They adopt the approach described in §4.2 and §4.4. We refer to the implementation of LinkGNN for pooling desribed in §4.5 as CCPool. Referring to the terminology of [37], CCPool technically uses LinkGNN as the "select" function of the pooling method, while for the "reduce" and "connect" operations we adopt standard choices: the aggregated feature matrix is given by a sum across the pooled nodes, and nodes are connected in the pooled graph if the two corresponding clusters were originally connected by at least an edge. In addition, we do not perform Pivot-based sampling in CCPool: since the graphs are already quite small, we do not require sampling to train eficiently.

Hyperparameters: In both §6.2 and §6.3, we set the maximum number of clusters for NodeGNN to � = min (1�4, |�|), where |�| is the number of nodes in the training graph. For LinkGNN, we use 512 output channels in §6.2 and 64 in §6.3. Both models are trained for a maximum of 5000 epochs, with early stopping triggered if the loss does not improve for 100 consecutive epochs (500 epochs for §6.3). The configurations for features and sampling vary by section: §6.2 uses 1000 random pivots for sampling at each epoch alongside 512 random features per node. Meanwhile, §6.3 uses a batch size of 64 for all datasets except ogbg-molhiv and ogbg-ppa, which use a batch size of 128, and extracts one random pivot per each graph in the batch. For Node2Vec in §6.3, we use 128 embedding dimensions and train for 100 epochs with a walk length of 10, a context size of 10, and a learning rate of 0.01. In §6.1, we adjust the setup to use 32 random node features and 32 output channels for LinkGNN. Finally, §6.4 employs a GNN architecture consisting of two GIN convolutional layers [68], a pooling layer, a third GIN layer, a global pooling layer, and a three-layer MLP readout. Each layer contains 64 hidden channels, except the final layer of the MLP, which has 32. This model is trained for 1000 epochs with a batch size of 64, a learning rate of 1� − 4, and an early stopping patience of 100 epochs. Hardware and Compute Environment: Experiments were conducted across two distinct computing environments based on workload requirements:

![](images/2580ceebf7f137b1245b79a34ced5fc0b3f341b7b0df6a7fd111c54bd3d8203b.jpg)  
(a)

![](images/f7332e64b91fb8f2f5dafb3235c4ecae399dfa9f5f58837cb70a6c21ab5bc7b3.jpg)  
(b)  
Figure 3: Approximation ratio analysis on synthetic Stochastic Block Model (SBM) graphs (|� | = 36) relative to the optimal solution found by Gurobi. A ratio of 1.0 indicates an optimal solution. Shaded areas represent the standard error of the mean over 10 iterations. (a) Approximation ratio for NodeGNN (� = 36) and LinkGNN as a function of the number of blocks � (proxy for ground truth clusters). (b) Impact of the maximum cluster budget � on the approximation quality of NodeGNN for varying block densities.

• Large-scale & GPU: Experiments on real-world datasets (§6.2, 6.3, 6.4) were executed on a machine with an NVIDIA A100 GPU (80GB VRAM) and an Intel Xeon Gold 6248R 15-core CPU.

• Small-scale & CPU: Synthetic SBM experiments (§6.1) and all runs of the combinatorial baseline CFP were performed on an Apple M1 Pro processor (8-core CPU, 16GB RAM).

## 6 Experimental Results and Discussion

We report experimental results to evaluate the efectiveness, scalability, and versatility of the proposed framework. The study is organized around the four research questions in §5. In the following subsections, we address each question in turn, using the datasets and protocols defined in Section 5.

## 6.1 RQ1: Approximation Quality

We compare GNN solutions against the exact Gurobi optimum on synthetic SBM graphs. In Figure 3a, we report the ratio between the GNN solution cost and the optimal value while increasing the number of blocks � in the SBM. The results show that both models achieve tight approximations, consistently staying within 1.05× of the optimal cost. However, the error for both methods generally increases with the number of blocks �. In particular, LinkGNN exhibits a larger error than NodeGNN, indicating that link-wise optimization introduces a greater approximation error.

While LinkGNN has no upper limit to the maximum number of output clusters, in NodeGNN this is controlled by the hyperparameter �. Figure 3b reports the approximation ratio while varying � in NodeGNN. It highlights a structural limitation of NodeGNN: the error spikes significantly if the cluster budget � underestimates the true cluster count (� < �). To ensure robustness when the optimal number of clusters is unknown, NodeGNN requires setting � ≈ |� |, which severely impacts eficiency. This underscores LinkGNN’s advantage in resource-constrained scenarios involving large graphs with numerous clusters.

Table 2: Normalized cost comparison in the transductive setting. Values for each dataset are normalized relative to the minimum cost achieved across all methods (1.0 indicates the best performance). The horizontal line separates standard Correlation Clustering benchmarks (top) from node classification datasets (bottom). Best performance for each dataset is in bold, while second best performance is underlined. At the bottom, number of times a method appears as best or second-best out of all datasets.
<table><tr><td colspan="4">NodeGNN LinkGNN KwikClus. ModPivot</td><td></td><td>CFP</td></tr><tr><td>polblogs</td><td>1.00±3e-4</td><td>1.05±1e-3</td><td></td><td>1.36±0.04 1.12±4e-4</td><td>1.27±2e-3</td></tr><tr><td>GrQc</td><td>1.00±1e-3</td><td>1.39±4e-3</td><td>1.52±0.02</td><td>1.26±0.01</td><td>2.04±5e-3</td></tr><tr><td>HepTh</td><td>1.00±1e-3</td><td>1.23±2e-3</td><td>1.48±0.02</td><td>1.20±4e-3</td><td>1.81±2e-3</td></tr><tr><td>AstroPh</td><td>1.00±7e-4</td><td>1.35±7e-4</td><td>1.70±0.12</td><td>1.26±0.01</td><td>1.76±2e-3</td></tr><tr><td>Enron</td><td>1.14±5e-3</td><td>1.01±2e-4</td><td></td><td>2.40±0.82 1.01±4e-3</td><td>1.32±8e-4</td></tr><tr><td>cond-mat</td><td>1.01±5e-3</td><td>1.09±3e-4</td><td></td><td>1.25±0.02 1.01±3e-3</td><td>1.45±7e-4</td></tr><tr><td># 1st/2nd</td><td>5/1</td><td>1/2</td><td>0/0</td><td>2/3</td><td>0/0</td></tr></table>

## 6.2 RQ2: Transductive Performance

We first test our local optimization framework on a standard (non-inductive) setup. The results, reported in Table 2 shows that NodeGNN almost always obtains the solution with the minimum cost across competitors. By explicitly optimizing a relaxed Correlation Clustering objective for local minima, our method yields higher-quality solutions than competitors, which adopt combinatorial methods. In contrast, LinkGNN often provides a worse approximation ratio than NodeGNN and Modified Pivot. However, on the large scale datasets email-Enron and cond-mat-2005, where the optimal number of clusters likely exceeds the � = 10, 000 cluster limit, LinkGNN ranks as best or second best method because its distance-based approach is not restricted by a maximum cluster count, allowing it to scale more efectively than NodeGNN.

## 6.3 RQ3: Inductive Generalization and Eficiency

We now consider the inductive setup (§5). While previously NodeGNN usually performed better than LinkGNN, in this case (Figure 4) the cluster assignment learnt by NodeGNN is not well generalizable to a new graph. This performance collapse stems from architectural overfitting: by treating clusters as distinct classes, NodeGNN uses significantly more parameters than LinkGNN. Consequently, the model memorizes specific cluster identities rather than learning transferable structural motifs, causing it to fail on the novel distributions of unseen graphs. LinkGNN, instead, obtains results better than KwikCluster on 4 out of 7 datasets, and in a few cases is almost on par than ModifiedPivot. By mapping nodes into a latent space where proximity reflects structural features and connectivity, LinkGNN captures universal clustering patterns instead of graph-specific assignments which allow to perform clustering inductively. Additionally, as shown in Table 3, since we are not re-training LinkGNN from scratch for each graph, at inference time is significantly faster than ModifiedPivot and KwikCluster.

![](images/b3c0ebd74ca04750d017da8a254435af335f9006b79fe0c52797fba31a88c87a.jpg)  
Figure 4: Comparison of methods in an inductive setting. Each point has error bars corresponding to the standard error of the mean, but are often too small. (I) was used to demark inductive methods, (T) for transductive ones. Values were normalized in each train/test split according to the minimum cost reached across methods.

Table 3: Inference time (in seconds) with the methods in the inductive setting of §6.3.
<table><tr><td></td><td colspan="3">|LinkGNN KwikCluster ModifiedPivot</td></tr><tr><td>MUTAG</td><td>0.01±2e-3</td><td>1e-3±3e-5</td><td>0.21±0.7</td></tr><tr><td>NCI1</td><td>0.04±7e-3</td><td>0.08±7e-4</td><td>5.70±0.10</td></tr><tr><td>EXPWL1</td><td>0.02±7e-4</td><td>0.36±0.02</td><td>11.73±0.09</td></tr><tr><td>Reddit</td><td>0.02±1e-3</td><td>17.98±1.85</td><td>633.36±106.97</td></tr><tr><td>stargazers</td><td>0.09±9e-3</td><td>6.37±0.06</td><td>314.00±10.96</td></tr><tr><td>ogbg-molhiv</td><td>0.20±0.05</td><td>0.70±4e-03</td><td>46.72±0.31</td></tr><tr><td>ogbg-ppa</td><td>0.52±0.02</td><td>181.54±1.89</td><td>2234.55±11.17</td></tr></table>

## 6.4 RQ4: Application to Graph Pooling

We adapt our inductive LinkGNN as a pooling technique (§4.5).

Table 4, shows that CCPool delivers more consistent performance compared to state-of-the-art pooling mechanisms by combining structural flexibility with localized learning capabilities. On the EXPWL1 dataset, which evaluates the expressiveness of pooling architectures, dense poolers such as JBPool, HOSC, DMoN, and ACC are fundamentally constrained by a fixed maximum number of clusters �, often leading to failure in inductive clustering tasks. CCPool, like other sparse architectures such as MaxCut, SEP, and �-MIS, overcomes this limitation by dynamically adapting to new graphs through a variable number of output clusters.

Crucially, CCPool overcomes the traditional trade-ofs that limit other sparse poolers. While sparse methods typically underperform on molecular datasets like MUTAG, NCI1, and OGBG-molhiv (or, like SEP, fail to complete training on larger datasets), where independent functional groups dominate molecular properties, CCPool consistently matches the performance of dense methods on these chemical tasks. This demonstrates that CCPool is uniquely capable of learning dataset-specific structures while simultaneously retain ing the adaptive flexibility of a sparse pooling architecture, making it a more robust and versatile choice than its competitors.

Table 4: Accuracy of pooling methods on graph classification datasets. Accuracy is reported for all datasets except OGBGmolhiv, where ROC-AUC is used. For each dataset, best score is in bold and second-best is underlined. For OGBG-molhiv, SEP consistently exceeded our 48-hour time limit (OOT) without completing the training. Last column shows the average ranking of each method (lower is better).
<table><tr><td></td><td>EXPWL1</td><td>MUTAG</td><td>NCI1</td><td></td><td>molhiv | Avg. Rank</td></tr><tr><td>CCPool</td><td>100.0±0.0</td><td>84.2±2.2</td><td>78.0±0.7</td><td>80.1±0.8</td><td>2.88</td></tr><tr><td>JBPool</td><td>98.7±0.3</td><td>82.0±2.3</td><td>78.9±0.4</td><td>77.6±0.9</td><td>4.75</td></tr><tr><td>HOSC</td><td>91.8±0.7</td><td>85.1±2.3</td><td>77.8±1.0</td><td>79.0±0.6</td><td>5.25</td></tr><tr><td>DMoN</td><td>96.3±0.8</td><td>84.8±1.9</td><td>77.5±0.7</td><td>79.6±0.8</td><td>4.75</td></tr><tr><td>ACC</td><td>93.3±1.1</td><td>91.6±1.9</td><td>78.0±0.6</td><td>79.3±0.8</td><td>3.50</td></tr><tr><td>k-MIS</td><td>99.8±0.1</td><td>81.6±2.7</td><td>77.9±0.8</td><td>79.5±0.8</td><td>4.50</td></tr><tr><td>MaxCut</td><td>99.6±0.2</td><td>76.1±3.0</td><td>76.9±0.4</td><td>78.0±1.0</td><td>6.50</td></tr><tr><td>SEP</td><td>100.0±0.0</td><td>85.0±2.2</td><td>78.0±0.5</td><td>OOT</td><td>3.88</td></tr></table>

## 6.5 Ablation Studies

As a further test on our framework, we analyze the performance of both NodeGNN and LinkGNN when altering the number and the type of layers on the Correlation Clustering benchmarks. We show the results in Table 5 and Table 6, where the cost of the final solution is normalized with the minimum only across the methods used in this section (and not, e.g., the results of ModifiedPivot). We also analyze on REDDIT-binary additional ablations regarding our algorithm design choices (Table 7), and performed a sensitivity analysis on the embedding dimension � and the threshold selection (Table 8). In the sensitivity analysis, we are not using on the test set the threshold chosen during training, but perform a sweep to show how the solution quality varies according to the choice.

Architecture Ablation: Our single-layer NodeGNN variant consistently performs best. As shown in Tables 5 and 6, increasing layer depth, simplifying the architecture, or using alternative convolutions (except GraphSAGE) degrades performance. These results validate the eficacy of GNNs for this framework; constraining the model to a single layer keeps the parameter count manageable, which is crucial since the output dimension is fixed to � = min(1�4, |�|). For LinkGNN, results are more nuanced: a single convolutional layer is optimal for all datasets except ca-AstroPh, where the embedding-only variant ("EMBOnly") excels. However, EMBOnly is transductive and cannot generalize to unseen graphs. GCN and GAT yield comparable performance, so we choose GCN for architectural consistency with NodeGNN. In this link-wise setup, architectural choices are less impactful because the parameter count is significantly lower, though the optimization still benefits from structural graph information and simpler models.

Inductive Algorithm Ablation: We observe from Table 7 that each design choice afect the quality of our solution:

• Pivot Sampling: It drastically reduces memory and training time without degrading quality (as mentioned in §4.1).

• L2 Normalization: Normalization of the embeddings allows the method to find proper solutions for our CC problem relaxation by constraining distances in a [0,1] range, and speeds up convergence 5x. Additionally, it has theoretical grounding in contrastive learning [64]. We hypothesize that also in this case it grants convergence on alignment (similar intra-cluster nodes are grouped together) and uniformity (evenly distributed dissimilar clusters). While a low embedding dimension � on large graphs could cause small distinct clusters to overlap.

Table 5: Ablation study on model complexity of NodeGNN and LinkGNN, comparing the clustering cost (Eq. 10). The best model on each row is in bold; costs are normalized for each dataset across all methods used in the table.
<table><tr><td>Dataset</td><td>|EMBOnly Linear 1 Layer 2 Layers 3 Layers</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">NodeGNN</td></tr><tr><td rowspan="5">polblogs GrQc HepTh AstroPh Enron</td><td> $1 . 0 6 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 1 1 { \pm } 5 { \mathrm { e } } { \cdot } 4$ </td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 2 } e { - 4 } }$ </td><td> $3 . 7 5 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $2 5 . 8 9 { \scriptstyle \pm 4 . 6 3 }$ </td></tr><tr><td> $2 . 3 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $2 . 3 5 { \scriptstyle \pm 2 \mathrm { e } - 3 }$ </td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 6 } e { - 4 } }$ </td><td> $9 0 . 0 0 { \scriptstyle \pm 9 . 7 9 }$ </td><td> $2 2 2 4 . 2 5 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td> $1 . 7 4 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 6 4 \pm 1 \mathrm { e } { \cdot } 3$ </td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 9 } e { - 4 } }$ </td><td> $3 . 6 6 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $2 9 6 5 . 7 3 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td> $1 . 6 5 { \scriptstyle \pm 6 \mathrm { e } } - 3$ </td><td> $1 . 6 4 \pm 2 \mathrm { e } { \cdot } 4$ </td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 1 } e - 3 }$ </td><td> $2 . 1 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $1 3 7 1 . 6 4 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td> $1 . 5 0 { \scriptstyle \pm 2 \mathrm { e } - 3 }$   $1 . 8 0 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 . 5 6 { \scriptstyle \pm 2 \mathrm { e } } { \cdot 3 }$   $1 . 8 8 { \scriptstyle \pm 1 \mathrm { e } - 3 }$ </td><td> $1 . 1 2 { \scriptstyle \pm 5 \mathrm { e } } . 3$   $\mathbf { 1 . 0 1 } { \pm } 5 \mathbf { e } { \cdot } 3$ </td><td> $5 . 7 1 { \scriptstyle \pm 2 . 8 0 }$   $4 . 0 6 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $4 0 0 3 . 6 5 { \scriptstyle \pm 0 . 0 0 }$   $5 6 4 3 . 2 2 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td colspan="6">cond-mat</td></tr><tr><td rowspan="5">polblogs GrQc HepTh</td><td> $1 . 0 5 { \scriptstyle \pm 1 } \mathrm { e } { \cdot } 5$ </td><td> $1 . 0 6 { \scriptstyle \pm 8 \mathrm { e - 4 } }$ </td><td>LinkGNN  $\mathbf { 1 . 0 4 \pm 7 e { - 4 } }$ </td><td> $1 . 0 7 { \scriptstyle \pm 3 \mathrm { e } } . 3$ </td><td> $1 . 1 3 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td> $1 . 3 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 9 8 { \scriptstyle \pm 3 \mathrm { e } - 3 }$ </td><td> $1 . 3 2 { \scriptstyle \pm 1 } \mathbf { e } { \cdot } 3$ </td><td> $1 . 4 3 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 . 4 6 { \scriptstyle \pm 0 . 0 2 }$ </td></tr><tr><td> $1 . 4 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 5 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 2 3 { \scriptstyle \pm 3 \mathbf { e } } { \cdot } 4$ </td><td> $1 . 2 9 { \scriptstyle \pm 1 \mathrm { e } - 3 }$ </td><td> $1 . 3 0 { \scriptstyle \pm 6 \mathrm { e } } - 3$ </td></tr><tr><td> $1 . 2 9 { \scriptstyle \pm 4 } \mathrm { e } { \scriptscriptstyle - 3 }$ </td><td> $1 . 5 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 3 8 { \scriptstyle \pm 2 e - 4 }$ </td><td> $1 . 4 0 { \scriptstyle \pm 5 \mathrm { e } - 3 }$ </td><td> $1 . 3 7 { \pm } 9 \mathrm { e } { - } 3$ </td></tr><tr><td> $1 . 0 9 { \scriptstyle \pm 1 \mathrm { e } } { \scriptscriptstyle - 4 }$ </td><td> $1 . 0 9 { \scriptstyle \pm 1 } \mathrm { e } { \scriptscriptstyle - 6 }$ </td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 2 } e { - 4 } }$ </td><td> $1 . 0 4 { \pm } 4 \mathrm { e } { \cdot } 4$ </td><td> $1 . 1 4 { \scriptstyle \pm 0 . 0 3 }$ </td></tr><tr><td>Enron  $\mathrm { c o n d - m a t }$ </td><td> $1 . 2 3 { \scriptstyle \pm 1 } \mathrm { e } \cdot 3$ </td><td> $1 . 2 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $\mathbf { 1 . 0 5 { \scriptstyle \pm 2 e - 3 } }$ </td><td> $1 . 1 1 { \pm } 3 { \mathrm { e } } { \cdot } 3$ </td><td> $1 . 1 2 { \scriptstyle \pm 0 . 0 1 }$ </td></tr></table>

Table 6: Ablation study on the message passing operation by comparing clustering costs. In all models, we use a single message passing layer. As in Table 5, the costs are normalized for each dataset across all models used in the table. Best performance for each line is in bold.
<table><tr><td>Dataset</td><td>GCN</td><td>GraphSAGE</td><td>GIN</td><td>GAT</td></tr><tr><td colspan="5">NodeGNN</td></tr><tr><td>polblogs</td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 2 } e - 4 }$ </td><td> $1 . 0 5 { \scriptstyle \pm 1 } \mathrm { e } { \cdot } 3$ </td><td> $1 . 3 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $1 . 6 8 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>GrQc HepTh</td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 6 } e { - 4 } }$ </td><td> $1 . 3 0 { \scriptstyle \pm 4 \mathrm { e } } . 3$ </td><td> $3 0 . 0 0 { \scriptstyle \pm 2 . 9 9 }$ </td><td> $2 . 1 8 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td>AstroPh</td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 9 } e { - 4 } }$ </td><td> $1 . 2 5 { \scriptstyle \pm 1 } \mathrm { e } { \cdot } 3$ </td><td> $5 5 . 4 7 { \scriptstyle \pm 8 . 2 2 }$ </td><td> $2 . 1 8 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td></td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 1 } e - 3 }$ </td><td> $1 . 0 8 { \scriptstyle \pm 9 } \mathrm { e - } 4$ </td><td> $4 2 . 8 8 { \scriptstyle \pm 7 . 0 6 }$ </td><td> $2 . 2 6 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>Enron cond-mat</td><td> $1 . 1 2 { \scriptstyle \pm 5 \mathrm { e - 3 } }$   $\mathbf { 1 . 0 1 { \pm } 5 e { - } 3 }$ </td><td> $7 . 0 7 { \scriptstyle \pm 0 . 8 7 }$   $1 . 3 3 { \scriptstyle \pm 5 \mathrm { e } - 3 }$ </td><td> $1 , 0 5 6 . 9 { \scriptstyle \pm 3 5 1 . 4 }$   $^ { 3 , 3 5 0 . 3 \pm 5 8 3 . 1 }$ </td><td> $6 . 2 6 { \scriptstyle \pm 0 . 5 8 }$   $3 . 0 1 { \scriptstyle \pm 0 . 0 9 }$ </td></tr><tr><td colspan="5">LinkGNN</td></tr><tr><td>polblogs</td><td> $\mathbf { 1 . 0 4 \pm 7 e { - 4 } }$ </td><td> $1 . 0 6 { \pm } 1 { \mathrm { e } } { - } 3$ </td><td> $1 . 0 6 { \scriptstyle \pm 2 \mathrm { e } - 3 }$ </td><td> $1 . 0 8 { \scriptstyle \pm 8 \mathrm { e } } - 3$ </td></tr><tr><td>GrQc</td><td> $1 . 3 2 { \pm } 1 { \mathrm { e } } { - } 3$ </td><td> $1 . 3 7 { \pm } 4 \mathrm { e } { - } 3$ </td><td> $1 . 5 0 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 3 1 { \pm } 9 { \mathrm { e } } { \cdot } 3$ </td></tr><tr><td>HepTh</td><td> $1 . 2 3 { \scriptstyle \pm 3 \mathbf { e } } { \scriptscriptstyle - 4 }$ </td><td> $1 . 2 8 { \scriptstyle \pm 7 } { \mathrm { e } } { \cdot } 3$ </td><td> $1 . 3 6 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $1 . 2 4 { \pm } 3 { \mathrm { e } } { \cdot } 3$ </td></tr><tr><td>AstroPh</td><td> $1 . 3 8 { \scriptstyle \pm 2 e - 4 }$ </td><td> $1 . 3 6 { \pm } 2 \mathrm { e } { \cdot } 4$ </td><td> $1 . 3 7 { \pm } 2 \mathrm { e } { - } 3$ </td><td> $1 . 3 4 { \scriptstyle \pm 9 \mathbf { e } - 3 }$ </td></tr><tr><td>email-Enron</td><td> $\mathbf { 1 . 0 0 { \scriptstyle \pm 2 } e - 4 }$ </td><td> $5 . 9 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $1 . 1 2 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 0 8 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>cond-mat</td><td> $1 . 0 5 { \scriptstyle \pm 2 e - 3 }$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $1 . 1 1 { \pm } 5 { \mathrm { e } } { \cdot } 3$ </td><td> $1 . 1 9 { \scriptstyle \pm 4 \mathrm { e } } - 3$ </td><td> $\mathbf { 1 . 0 4 \pm 1 e { - 3 } }$ </td></tr></table>

• Rounding clusters: Substituting our clustering inference heuristic (Algorithm 4) with the Charikar rounding from [22] increases clustering costs, as this method assume a diferent CC relaxation.

Sensitivity Analysis: From Table 8 we see that the choice of embedding dimension � does not have a strong impact on the quality of the solution as long as it is large enough. Instead, while

Table 7: Ablation study on inductive algorithm design choices. We compute the cost ratio relative to the baseline configuration obtained without any ablation ("Baseline")).
<table><tr><td>Ablation</td><td> $\operatorname { A p p r o x } .$ </td><td>Ratio Train time (s)</td><td> $\mathrm { G P U } \left( \mathrm { M B } \right)$ </td></tr><tr><td>Baseline</td><td> $1 . 0 { \pm } 2 \mathrm { e } { - } 0 3$ </td><td> $1 9 0 { \pm } 9$ </td><td> $1 7 7 5 1 { \scriptstyle \pm 1 0 9 1 }$ </td></tr><tr><td>No pivot</td><td> $1 . 0 { \scriptstyle \pm 7 \mathrm { e - } } 0 4$ </td><td> $8 1 6 6 { \pm } 5 2 3$ </td><td> $7 5 7 0 9 { \scriptstyle \pm 1 2 8 7 }$ </td></tr><tr><td>No L2 norm.</td><td> $1 . 0 { \pm } 2 \mathrm { e } { - } 0 3$ </td><td> $9 1 8 { \pm } 1 8$ </td><td> $1 7 7 5 1 { \scriptstyle \pm 1 0 9 1 }$ </td></tr><tr><td>Charikar clusters</td><td> $1 . 1 { \pm } 3 { \mathrm { e } } { \cdot } 0 3$ </td><td> $1 9 7 \pm 9$ </td><td> $1 7 7 4 8 { \scriptstyle \pm 1 0 9 2 }$ </td></tr></table>

Table 8: Approximation ratio depending on embedding dimension � and threshold � selection. The ratio is computed with respect to the minimum across all configurations.
<table><tr><td>D</td><td> $t = 0 . 2 5$ </td><td> $t = 0 . 5 0$ </td><td> $t = 0 . 9 0$ </td><td> $t = 0 . 9 5$ </td><td> $t = 1 . 0 0$ </td></tr><tr><td>16</td><td> $2 3 2 . 5 0 { \scriptstyle \pm 2 2 . 7 4 }$ </td><td> $6 7 . 1 8 { \scriptstyle \pm 7 . 6 2 }$ </td><td> $1 . 0 6 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 0 3 { \scriptstyle \pm 8 \mathrm { e } - 0 3 }$ </td><td> $1 . 0 9 { \scriptstyle \pm 9 \mathrm { e } - 0 3 }$ </td></tr><tr><td>64</td><td> $2 2 2 . 5 1 { \scriptstyle \pm 2 3 . 7 3 }$ </td><td> $6 1 . 4 7 { \scriptstyle \pm 7 . 6 3 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 2 \mathrm { e } - 0 3 }$ </td><td> $1 . 0 3 { \scriptstyle \pm 9 \mathrm { e } - 0 3 }$ </td><td> $1 . 0 9 { \scriptstyle \pm 9 \mathrm { e } - 0 3 }$ </td></tr><tr><td>256</td><td> $2 2 1 . 5 3 { \scriptstyle \pm 2 3 . 9 8 }$ </td><td> $6 1 . 7 5 { \scriptstyle \pm 7 . 8 6 }$ </td><td> $1 . 0 1 { \pm } 5 { \mathrm { e } } { \cdot } 0 3$ </td><td> $1 . 0 3 { \scriptstyle \pm 9 \mathrm { e } - 0 3 }$ </td><td> $1 . 0 9 { \scriptstyle \pm 9 \mathrm { e } - 0 3 }$ </td></tr></table>

the exact threshold impacts quality, the training-optimized value (typically $t \approx 0 . 9 0 )$ transfers robustly to unseen graphs.

## 7 Conclusions

This work introduces Inductive Correlation Clustering, a novel extension of Correlation Clustering that learns a parametrized function to handle unseen graph instances. We solve this problem by establishing CC-GNN, a local optimization framework for Correlation Clustering which takes advantage ofthe inductive properties of GNNs, alongside a pivot-based sampling strategy to ensure scalable and eficient training on large graphs. In particular, we consider two alternative formulations, which we dub NodeGNN and LinkGNN. Through comprehensive experiments, we demonstrate the eficacy of NodeGNN in achieving superior solution quality in transductive settings with respect to competitors; LinkGNN, instead, successfully generalizes to new graphs providing solutions of quality compa rable to transductive baselines, while spending significantly less inference time. Finally, we demonstrate an application of Inductive Correlation Clustering to graph pooling. We adapt LinkGNN into a learnable pooling mechanism, CCPool, and showcase its competitiveness with state-of-the-art techniques in graph classification tasks. In summary, this research combines classic combinatorial optimization and deep learning techniques, ofering robust and scalable solutions for clustering graph data in large and dynamic batches. Future work will explore inductive applications of Correlation Clustering extensions (e.g. overlapping clusters or diversity constraints), and identify improved rounding algorithms.

## Acknowledgments

Part of this work was done while F.P.N. was visiting Bowling Green State University. F.P.N. and A.K. gratefully acknowledge the Ohio Supercomputer Center (OSC) for providing the computing resources and facilities used in conducting part of the experiments.

## GenAI Usage Disclosure

Generative AI tools were used in a assistive capacity in the preparation of this work: as a coding assistant during the implementation of the experimental pipeline, and for proofreading and editorial polishing of the manuscript. The research ideas, the problem formulation, the design of the proposed methods, the experimental protocol, and the analysis and interpretation of the results are entirely the authors’ own. All AI-assisted code and text were reviewed and validated by the authors, who take full responsibility for the content of this paper.

## References

[1] Carlo Abate and Filippo Maria Bianchi. 2025. MaxCutPool: diferentiable featureaware Maxcut for pooling in graph neural networks. In The Thirteenth International Conference on Learning Representations.

[2] Ralph Abboud, İsmail İlkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. 2021. The Surprising Power of Graph Neural Networks with Random Node Initialization. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, Zhi-Hua Zhou (Ed.). International Joint Conferences on Artificial Intelligence Organization, 2112–2118. doi:10.24963/ijcai.2021/291 Main Track.

[3] KookJin Ahn, Graham Cormode, Sudipto Guha, Andrew McGregor, and Anthony Wirth. 2015. Correlation Clustering in Data Streams. In Proceedings ofthe 32nd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 37), Francis Bach and David Blei (Eds.). PMLR, Lille, France, 2237– 2246. https://proceedings.mlr.press/v37/ahn15.html

[4] Nir Ailon, Moses Charikar, and Alantha Newman. 2008. Aggregating inconsistent information: ranking and clustering. Journal ofthe ACM (JACM) 55, 5 (2008), 1–27.

[5] Davide Bacciu, Alessio Conte, and Francesco Landolfi. 2023. Generalizing down sampling from regular data to graphs. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 6718–6727.

[6] Nikhil Bansal, Avrim Blum, and Shuchi Chawla. 2004. Correlation clustering. Machine learning 56, 1 (2004), 89–113.

[7] Soheil Behnezhad, Moses Charikar, Vincent Cohen-Addad, Alma Ghafari, and Weiyun ma. 2025. Correlation Clustering Beyond the Pivot Algorithm. In Fortysecond International Conference on Machine Learning. https://openreview.net/ forum?id=OzQLuoKMQZ

[8] Soheil Behnezhad, Moses Charikar, Weiyun Ma, and Li-Yang Tan. 2023. Single pass streaming algorithms for correlation clustering. In Proceedings ofthe 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM, 819–849.

[9] Soheil Behnezhad, Mahsa Derakhshan, MohammadTaghi Hajiaghayi, Clif Stein, and Madhu Sudan. 2019. Fully dynamic maximal independent set with polylog arithmic update time. In 2019 IEEE 60th Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 382–405.

[10] A. Ben-Dor, R. Shamir, and Z. Yakhini. 1999. Clustering Gene Expression Patterns. Journal ofComputational Biology 6, 3/4 (1999), 281–297.

[11] Vedangi Bengali and Nate Veldt. 2023. Faster approximation algorithms for parameterized graph clustering and edge labeling. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 78–87.

[12] Filippo Maria Bianchi. 2023. Simplifying Clustering with Graph Neural Networks. In Proceedings ofthe Northern Lights Deep Learning Workshop, Vol. 4.

[13] Filippo Maria Bianchi, Carlo Abate, and Ivan Marisca. 2025. Torch Geometric Pool: the Pytorch library for pooling in Graph Neural Networks. arXiv preprint arXiv:2512.12642 (2025).

[14] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. 2020. Spectral clustering with graph neural networks for graph pooling. In International conference on machine learning. PMLR, 874–883.

[15] Filippo Maria Bianchi and Veronica Lachi. 2023. The expressive power of pooling in graph neural networks. Advances in neural information processing systems 36 (2023), 71603–71618.

[16] A. Björn, H. K. Jörg, B. Thorsten, K. Ullrich, and A. H. Fred. 2011. Probabilistic image segmentation with closedness constraints. In Proc. ofIEEE Int. Conf. on Computer Vision (ICCV). 2611–2618.

[17] Francesco Bonchi, David García-Soriano, and Francesco Gullo. 2022. Correlation Clustering. Morgan & Claypool Publishers.

[18] Francesco Bonchi, David García-Soriano, and Edo Liberty. 2014. Correlation clustering: from theory to practice. In Proc. ofthe 20th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD). 1972.

[19] Nairen Cao, Vincent Cohen-Addad, Euiwoong Lee, Shi Li, David Rasmussen Lolck, Alantha Newman, Mikkel Thorup, Lukas Vogl, Shuyi Yan, and Hanwen Zhang. 2025. Solving the Correlation Cluster LP in Sublinear Time. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing. 1154–1165.

[20] Quentin Cappart, Didier Chételat, Elias B Khalil, Andrea Lodi, Christopher Morris, and Petar Veličković. 2023. Combinatorial optimization and reasoning with graph neural networks. Journal of Machine Learning Research 24, 130 (2023), 1–61.

[21] Sayak Chakrabarty and Konstantin Makarychev. 2023. Single-pass pivot al gorithm for correlation clustering. keep it simple!. In Proceedings of the 37th

International Conference on Neural Information Processing Systems. 6412–6421.

[22] Moses Charikar, Venkatesan Guruswami, and Anthony Wirth. 2005. Clustering with qualitative information. J. Comput. System Sci. 71, 3 (2005), 360–383.

[23] Yuhua Chen, Dengxin Dai, Jordi Pont-Tuset, and Luc Van Gool. 2016. Scale-Aware Alignment of Hierarchical Image Segmentation. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[24] Vassilis Christophides, Vasilis Efthymiou, Themis Palpanas, George Papadakis, and Kostas Stefanidis. 2020. An Overview of End-to-End Entity Resolution for Big Data. ACM Comput. Surv. 53, 6, Article 127 (Dec. 2020), 42 pages. doi:10. 1145/3418896

[25] Vincent Cohen-Addad, Silvio Lattanzi, Andreas Maggiori, and Nikos Parotsidis. 2022. Online and consistent correlation clustering. In International Conference on Machine Learning. PMLR, 4157–4179.

[26] Vincent Cohen-Addad, Silvio Lattanzi, Slobodan Mitrović, Ashkan Norouzi-Fard, Nikos Parotsidis, and Jakub Tarnawski. 2021. Correlation clustering in constant many parallel rounds. In International Conference on Machine Learning. PMLR, 2069–2078.

[27] Vincent Cohen-Addad, David Rasmussen Lolck, Marcin Pilipczuk, Mikkel Thorup, Shuyi Yan, and Hanwen Zhang. 2024. Combinatorial correlation clustering. In Proceedings ofthe 56th Annual ACM Symposium on Theory ofComputing. 1617– 1628.

[28] Mina Dalirrooyfard, Konstantin Makarychev, and Slobodan Mitrović. 2024. Pruned Pivot: Correlation Clustering Algorithm for Dynamic, Parallel, and Local Computation Models. In Forty-first International Conference on Machine Learning. https://openreview.net/forum?id=saP7s0ZgYE

[29] Bhaskar DasGupta, German Andres Enciso, Eduardo Sontag, and Yi Zhang. 2007. Algorithmic and complexity results for decompositions of biological networks into monotone subsystems. Biosystems 90, 1 (2007), 161–178.

[30] Timothy A Davis and Yifan Hu. 2011. The University of Florida sparse matrix collection. ACM Transactions on Mathematical Software (TOMS) 38, 1 (2011), 1–25.

[31] Erik D Demaine, Dotan Emanuel, Amos Fiat, and Nicole Immorlica. 2006. Correlation clustering in general weighted graphs. Theoretical Computer Science 361, 2-3 (2006), 172–187.

[32] Alexandre Duval and Fragkiskos Malliaros. 2022. Higher-order clustering and pooling for graph neural networks. In Proceedings of the 31st ACM international conference on information & knowledge management. 426–435.

[33] Vijay Prakash Dwivedi and Xavier Bresson. 2020. A generalization oftransformer networks to graphs. arXiv preprint arXiv:2012.09699. arXiv:2012.09699.

[34] Eshed Gal, Moshe Eliasof, Carola-Bibiane Schönlieb, Ivan Kyrchei, Eldad Haber, and Eran Treister. 2025. Towards Eficient Training of Graph Neural Networks: A Multiscale Approach. Transactions on Machine Learning Research (2025). https: //openreview.net/forum?id=2eZ8xkL2ZB

[35] Jun Gao, Jiazun Chen, Zhao Li, and Ji Zhang. 2021. ICS-GNN: lightweigh interactive community search via graph neural network. Proceedings ofthe VLDB Endowment 14, 6 (2021), 1006–1018.

[36] David García-Soriano, Konstantin Kutzkov, Francesco Bonchi, and Charalampos E. Tsourakakis. 2020. Query-Eficient Correlation Clustering. In WWW ’20: The Web Conference 2020, Taipei, Taiwan, April 20-24, 2020. 1468–1478.

[37] Daniele Grattarola, Daniele Zambon, Filippo Maria Bianchi, and Cesare Alippi. 2022. Understanding pooling in graph neural networks. IEEE transactions on neural networks and learning systems 35, 2 (2022), 2708–2718.

[38] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. Advances in neural information processing systems 30 (2017).

[39] Jonas Berg Hansen and Filippo Maria Bianchi. 2023. Total variation graph neural networks. In International Conference on Machine Learning. PMLR, 12445–12468.

[40] Oktie Hassanzadeh, Fei Chiang, Hyun Chul Lee, and Renée J Miller. 2009. Framework for evaluating clustering algorithms in duplicate detection. Proceedings of the VLDB Endowment 2, 1 (2009), 1282–1293.

[41] Paul W. Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. 1983. Stochastic blockmodels: First steps. Social Networks 5, 2 (1983), 109–137. doi:10.1016/0378-8733(83)90021-7

[42] Lun Hu, Shicheng Yang, Xin Luo, and MengChu Zhou. 2022. An Algorithm of Inductively Identifying Clusters From Attributed Graphs. IEEE Transactions on Big Data 8, 2 (2022), 523–534. doi:10.1109/TBDATA.2020.2964544

[43] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. 2020. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems 33 (2020), 22118–22133.

[44] Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, and Jie Tang. 2021. Mixgcf: An improved training method for graph neural network-based recommender systems. In Proceedings ofthe 27th ACM SIGKDD conference on knowledge discovery & data mining. 665–674.

[45] Chaitanya K Joshi, Quentin Cappart, Louis-Martin Rousseau, and Thomas Laurent. 2022. Learning the travelling salesperson problem requires rethinking generalization. Constraints 27, 1 (2022), 70–98.

[46] Leskovec Jure. 2014. Snap datasets: Stanford large network dataset collection. Retrieved December 2021 from http://snap. stanford. edu/data (2014).

[47] Sungwoong Kim, Sebastian Nowozin, Pushmeet Kohli, and Chang Dong Yoo. 2011. Higher-Order Correlation Clustering for Image Segmentation. In Proc. of Conf. on Advances in Neural Information Processing Systems (NeurIPS). 1530–1538.

[48] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations. https://openreview.net/forum?id=SJU4ayYg

[49] Yuko Kuroki, Atsushi Miyauchi, Francesco Bonchi, and Wei Chen. 2024. Query-Eficient Correlation Clustering with Noisy Oracle. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024.

[50] Shrinu Kushagra, Shai Ben-David, and Ihab Ilyas. 2019. Semi-supervised clustering for de-duplication. In The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 1659–1667.

[51] Claire Mathieu, Ocan Sankur, and Warren Schudy. 2010. Online Correlation Clustering. In 27th International Symposium on Theoretical Aspects ofComputer Science-STACS 2010. 573–584.

[52] Sadaaki Miyamoto. 2012. Inductive and non-inductive methods of clustering. In 2012 IEEE International Conference on Granular Computing. IEEE, 12–17.

[53] Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. 2020. TUDataset: A collection of benchmark datasets for learning with graphs. In ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020). arXiv:2007.08663 www.graphlearning.io

[54] Mariá C.V. Nascimento and André C.P.L.F. de Carvalho. 2011. Spectral methods for graph clustering – A survey. European Journal ofOperational Research 211, 2 (2011), 221–231. doi:10.1016/j.ejor.2010.08.012

[55] Robert R. Nerem, Samantha Chen, Sanjoy Dasgupta, and Yusu Wang. 2026. Graph neural networks extrapolate out-of-distribution for shortest paths. In Proceedings of Thirty Ninth Conference on Learning Theory (Proceedings of Machine Learning Research, Vol. 336), Steve Hanneke and Tor Lattimore (Eds.). PMLR, 5273–5331. https://proceedings.mlr.press/v336/nerem26a.html

[56] Ryoma Sato, Makoto Yamada, and Hisashi Kashima. 2021. Random features strengthen graph neural networks. In Proceedings of the 2021 SIAM international conference on data mining (SDM). SIAM, 333–341.

[57] Franco Scarselli, Sweah Liang Yong, Marco Gori, Markus Hagenbuchner, Ah Chung Tsoi, and Marco Maggini. 2005. Graph neural networks for ranking web pages. In The 2005 IEEE/WIC/ACM International Conference on Web Intelligence (WI’05). IEEE, 666–672.

[58] Martin JA Schuetz, J Kyle Brubaker, and Helmut G Katzgraber. 2022. Combinatorial optimization with physics-inspired graph neural networks. Nature Machine Intelligence 4, 4 (2022), 367–377.

[59] Michele Starnini, Charalampos E Tsourakakis, Maryam Zamanipour, André Panisson, Walter Allasia, Marco Fornasiero, Laura Li Puma, Valeria Ricci, Silvia Ronchiadin, Angela Ugrinoska, et al. 2021. Smurf-based anti-money laundering in time-evolving transaction networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 171–186.

[60] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. 2019. From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports 9, 1 (2019), 1–12.

[61] Anton Tsitsulin, John Palowitch, Bryan Perozzi, and Emmanuel Müller. 2023. Graph clustering with graph neural networks. Journal ofMachine Learning Research 24, 127 (2023), 1–21.

[62] Nate Veldt. 2022. Correlation clustering via strong triadic closure labeling: Fast approximation algorithms and practical lower bounds. In International Conference on Machine Learning. PMLR, 22060–22083

[63] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In International Conference on Learning Representations. https://openreview.net/forum?id=rJXMpikCZ

[64] Tongzhou Wang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning. PMLR, 9929–9939.

[65] Yan Wang, Wenju Hou, Nan Sheng, Ziqi Zhao, Jialin Liu, Lan Huang, and Juexin Wang. 2024. Graph pooling in graph neural networks: methods and their appli cations in omics studies. Artificial Intelligence Review 57, 11 (16 Sep 2024), 294. doi:10.1007/s10462-024-10918-9

[66] Yan Wang, Wenju Hou, Nan Sheng, Ziqi Zhao, Jialin Liu, Lan Huang, and Juexin Wang. 2024. Graph pooling in graph neural networks: methods and their appli cations in omics studies. Artificial Intelligence Review 57, 11 (2024), 294.

[67] Junran Wu, Xueyuan Chen, Ke Xu, and Shangzhe Li. 2022. Structural entropy guided graph hierarchical pooling. In International conference on machine learning. PMLR, 24017–24030.

[68] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019. How Powerful are Graph Neural Networks?. In International Conference on Learning Representations. https://openreview.net/forum?id=ryGs6iA5Km

[69] J. Yarkony, A. T. Ihler, and C. C. Fowlkes. 2012. Fast Planar Correlation Clustering for Image Segmentation. In Proc. of European Conf. on Computer Vision (ECCV). 568–581.

[70] Kai Siong Yow, Ningyi Liao, Siqiang Luo, and Reynold Cheng. 2023. Machine learning for subgraph extraction: Methods, applications and challenges. Proceedings ofthe VLDB Endowment 16, 12 (2023), 3864–3867.

[71] Muhan Zhang and Yixin Chen. 2020. Inductive Matrix Completion Based on Graph Neural Networks. In International Conference on Learning Representations. https://openreview.net/forum?id=ByxxgCEYDS