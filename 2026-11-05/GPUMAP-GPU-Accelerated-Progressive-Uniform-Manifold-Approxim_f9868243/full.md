# GPUMAP: GPU-Accelerated Progressive Uniform Manifold Approximation and Projection

Myeongwon Jung and Jaemin Jo

![](images/02537ce23efa06f7f7cd20423c6365b7485b6b39e3ad0c46717ea3c0d04581d3.jpg)  
Figure 1: Intermediate projections produced by GPUMAP on the HIGGS dataset (5M points) before standard UMAP completes. GPUMAP produces projections at 8.42, 19.86, and 27.03 seconds as more points are inserted, while standard UMAP returns its complete projection at 50.4 seconds. Points are colored by the m feature.

Abstract—Dimensionality reduction (DR) techniques are fundamental to the analysis of high-dimensional data, yet their substantia computational cost limits their use in interactive settings. To address this limitation, we present GPU-accelerated Progressive Uniform Manifold Approximation and Projection (GPUMAP), an efficient progressive DR algorithm designed for interactive use. We identify three atomic operations within UMAP and restructure them into a progressive pipeline. As part of this adaptation, we introduce modifications to UMAP, including operation queues, instance-wise local join, and scheduling strategies. We further accelerate computation by leveraging GPU parallelism, enabling interactive analysis of million-scale datasets on commodity GPU hardware. Finally, we employ our implementation to systematically study the behavior of progressive DR, investigating how projections evolve over time relative to the ground-truth projection and how biased data streams influence the quality and stability of the resulting projections.

Index Terms—Dimensionality Reduction, Uniform Manifold Approximation and Projection, Progressive Visual Analytics

## 1 INTRODUCTION

Dimensionality reduction (DR) techniques such as Uniform Manifold Approximation and Projection (UMAP) [17] have become a standard tool for exploring large-scale high-dimensional data. However, DR on large datasets is computationally expensive, dominated by neighborhood graph construction and embedding optimization. Conventional implementations produce no projection until this computation com pletes: analysts cannot inspect the data, identify emerging structures, or judge whether a projection is useful for their analysis until the very end. GPU-accelerated implementations [20] shorten this wait but retain the same sequential execution model, and as datasets continue to grow, this all-or-nothing workflow makes DR increasingly difficult to use within an interactive exploration loop.

Progressive Visual Analytics (PVA) offers a principled way to address this limitation by producing partial results while an analysis is still running [9, 27]. Rather than separating computation and inspection into distinct phases, PVA interleaves them, allowing users to examine emerging results before the underlying computation has finished. Sampling also lowers the latency of obtaining a result, but it requires committing to a sample size in advance, before the analyst knows how much data the question at hand demands, and a sample that turns out to be too small cannot generally be extended without recomputing the result. Applied to DR, this paradigm requires the system to produce a sequence of intermediate projections that are progressively refined as computation proceeds. Progressive DR thus imposes two requirements: the first inspectable projection must appear with low latency, and the projection must be continuously updated throughout execution.

Several progressive DR methods have been proposed [12, 14, 21], demonstrating that intermediate projections can be produced before the full computation completes. However, their applicability to modern large-scale datasets remains limited. These methods have primarily been evaluated on datasets containing tens of thousands of instances, leaving their scalability to million-scale data unexplored. Furthermore, while these methods produce intermediate projections, the behaviors arising from such progressive execution have not been systematically studied. For example, it remains unclear how rapidly the quality of intermediate projections converges to that of a sequentially computed projection and how distributing the computation budget among different operation types affects the projection quality.

In this work, we present GPU-accelerated Progressive Uniform Manifold Approximation and Projection (GPUMAP), an efficient and progressive implementation of the widely used DR method, UMAP [17]. Existing approaches address only one side of the scalability problem: RAPIDS UMAP exploits GPU parallelism but preserves the sequential execution model, while existing progressive DR methods suffer from scalability limitations. GPUMAP combines both directions, enabling progressive projection for million-scale data.

Our contributions are three-fold:

1. We present GPUMAP, an efficient progressive DR method that enables interactive analysis of large-scale high-dimensional data.

2. We investigate how GPUMAP approximates sequential DR projections in terms of runtime, nearest neighbor recall, and projec-

tion fidelity.

3. We systematically evaluate progressive DR under diverse usage scenarios, including scheduling strategies, time budgets, and biased data streams.

## 2 RELATED WORK

## 2.1 Uniform Manifold Approximation and Projection

We build our method on UMAP, inspired by both the original Python implementation [17] and its GPU-accelerated RAPIDS implementation [20]. To clarify how we improve UMAP to support progressive computation, we briefly review its computational pipeline. UMAP consists of two computation stages: k-Nearest Neighbor (kNN) graph construction and embedding optimization.

kNN Construction. UMAP first constructs a kNN graph to capture topological relationships among data points in the input space. Since exact kNN graph construction is expensive for large-scale datasets, UMAP mostly relies on approximate nearest neighbor algorithms such as Nearest-Neighbor Descent (NN-descent) [6]. The intuition behind NN-descent is that a neighbor of a neighbor is also likely to be a neighbor. In NN-descent, each point v maintains a neighbor list $N _ { k } ( \nu )$ and forms a candidate set $C ( \nu ) = N _ { k } ( \nu ) \cup N _ { k } ^ { - 1 } ( \nu )$ , where $N _ { k } ( \nu )$ denotes the forward neighbors (points in $\nu \mathbf { \dot { s } }$ neighbor list) and $N _ { k } ^ { - 1 } ( \nu ) = \{ u \}$ $\nu \in N _ { k } ( u ) \}$ denotes the backward neighbors (points that include v in their neighbor lists). NN-descent then evaluates pairwise distances for all pairs $( u , w )$ such that $u , w \in C ( \nu ) , u \neq$ w, updating neighbor lists accordingly, which is called local join.

Embedding Optimization. Using the constructed kNN graph, UMAP computes a fuzzy topological representation by assigning a membership strength to each edge between a point i and its neighbor $j \colon \nu _ { j \mid i } = \exp ( { - \operatorname* { m a x } ( 0 , d ( x _ { i } , x _ { j } ) - \rho _ { i } ) / \sigma _ { i } } )$ , where $x _ { i } \in \mathbb { R } ^ { D }$ denotes the high-dimensional input coordinates of point $i , \rho _ { i }$ controls the local connectivity around $x _ { i }$ and $\sigma _ { i }$ is a point-specific normalization factor. With the default local connectivity of 1.0, $\rho _ { i }$ corresponds to the distance from $x _ { i }$ to its nearest neighbor. The final symmetric edge weight is given by $\nu _ { i j } = ( \nu _ { j | i } + \nu _ { i | j } ) - \nu _ { j | i } \nu _ { i | j }$

In the projection space, UMAP defines the low-dimensional edge weight as $w _ { i j } = ( 1 + \bar { a } \| y _ { i } - y _ { j } \| _ { 2 } ^ { 2 b } ) ^ { - 1 }$ , where $y _ { i } \in \mathbb { R } ^ { d } ( d \ll D )$ denotes the low-dimensional embedding coordinates of point i, a and b are hyperparameters that control the shape of the embedding distribution. UMAP optimizes the embedding by minimizing the cross-entropy between $\nu _ { i j }$ and $w _ { i j } \colon$

$$
C E = \sum _ { i \neq j } \left[ \nu _ { i j } \cdot \log \left( \nu _ { i j } / w _ { i j } \right) - \left( 1 - \nu _ { i j } \right) \cdot \log \left( \left( 1 - \nu _ { i j } \right) / ( 1 - w _ { i j } ) \right) \right] .\tag{1}
$$

UMAP adopts a graph layout process with attractive and repulsive forces derived from the cross-entropy cost function.

$$
F _ { a t t } ( y _ { i } , y _ { j } , \nu _ { i j } ) = \frac { - 2 a b d _ { i j } ^ { 2 ( b - 1 ) } } { 1 + d _ { i j } ^ { 2 } } \nu _ { i j } ( y _ { i } - y _ { j } )\tag{2}
$$

$$
F _ { r e p } ( y _ { i } , y _ { j } , \nu _ { i j } ) = \frac { 2 b } { ( \varepsilon + d _ { i j } ^ { 2 } ) ( 1 + a d _ { i j } ^ { 2 b } ) } ( 1 - \nu _ { i j } ) ( y _ { i } - y _ { j } )\tag{3}
$$

where $d _ { i j } = \lVert y _ { i } - y _ { j } \rVert _ { 2 }$ and ε is a small constant to prevent division by zero.

From the implementation perspective, UMAP optimizes the embedding by computing attractive forces over kNN neighbors and repulsive forces over negatively sampled points, avoiding expensive all-pair com putations. The membership strength $\nu _ { i j }$ is not directly multiplied into the gradient; instead, it is reflected as a sampling schedule:

$$
n _ { i j } = n _ { \mathrm { e p o c h s } } \frac { \nu _ { i j } } { \operatorname* { m a x } _ { ( p , q ) } \nu _ { p q } } , \mathrm { e p o c h s \_ p e r \_ s a m p l e } _ { i j } = \frac { n _ { \mathrm { e p o c h s } } } { n _ { i j } } ,\tag{4}
$$

so that edges with larger weights are sampled more frequently. When an edge $( i , j )$ is sampled, an attractive update is applied:

$$
\mathrm { g r a d \_ c o e f f _ { } ^ { + } } _ { i j } = - 2 a b d _ { i j } ^ { 2 ( b - 1 ) } / ( 1 + a d _ { i j } ^ { 2 b } ) ,\tag{5}
$$

and then applies multiple repulsive updates on randomly sampled negative points using

$$
\mathrm { g r a d \_ c o e f f _ { \it i j } ^ { - } } = 2 \gamma b / ( ( \varepsilon + d _ { i j _ { k } } ^ { 2 } ) ( 1 + a d _ { i j _ { k } } ^ { 2 b } ) )\tag{6}
$$

where γ weights the repulsive term. The coefficients are then applied to update the embedding coordinates as

$$
y _ { i }  y _ { i } + \alpha \cdot \mathrm { g r a d \_ c o e f f } _ { i j } \cdot ( y _ { i } - y _ { j } ) ,\tag{7}
$$

where α is the learning rate that linearly decays from an initial value $\alpha _ { 0 }$ to zero over the course of optimization.

## 2.2 Projection and Visualization for Large-scale Data

To support the visual analysis of large-scale, high-dimensional data, prior work has explored methods to accelerate DR and reduce the computational burden. Early efforts addressed the cost of kNN graph construction by employing approximate nearest neighbor (ANN) methods. For example, A-tSNE [21] computes an approximate kNN graph using a forest of randomized KD-Trees [26], while LargeVis [28] constructs an approximate kNN graph using random projection trees [5] followed by neighbor exploration to improve graph quality.

In parallel, several studies aimed to reduce the embedding optimization process. For example, Barnes-Hut-SNE [32] approximates longrange repulsive forces using a Barnes-Hut tree [2], reducing the complexity of optimization from quadratic to ${ \cal O } ( N \log N )$ , and Fit-SNE [16] further accelerates t-SNE by approximating the repulsive-force computation using interpolation and the Fast Fourier Transform (FFT). LargeVis also improves the embedding optimization process by adopting asynchronous stochastic gradient descent to minimize its objective in linear time.

More recently, these acceleration methods, including approximate kNN graph construction and scalable embedding optimization, have also been implemented on the GPU in open-source libraries such as RAPIDS cuML [22] and TorchDR [30]. These libraries provide GPU-accelerated implementations of several DR techniques, including UMAP. However, such GPU-accelerated pipelines are typically benchmarked on high-end hardware such as an NVIDIA H100 GPU. On commodity GPUs, the achieved speedups are often insufficient to support interactive analysis of large-scale high-dimensional data.

Even with such acceleration, rendering and interactively exploring projections with millions of points remains computationally demanding. Systems such as WizMap [34] and Embedding Atlas [23] have therefore been proposed to support scalable rendering and interactive analysis of large projection spaces. However, because these systems still depend on completed projections, users must wait before visual inspection becomes possible.

In this work, we extend UMAP with a GPU-accelerated progressive computation pipeline, enabling users to inspect intermediate projections that approximate the final result before optimization is fully complete. While our approach is compatible with scalable visualization systems such as Embedding Atlas, our primary focus is not on rendering techniques themselves, but on enabling progressive projection generation that supports early and continuous data analysis.

## 2.3 Progressive Dimensionality Reduction

Progressive Visual Analytics (PVA) addresses the latency of longrunning computations by exposing partial results early and continuously, allowing analysts to inspect evolving outputs and steer computation before final convergence [9, 27]. Rather than blocking analysis until an exact result is available, PVA emphasizes meaningful intermediate results that are progressively refined over time. Applied to DR, this principle requires producing inspectable projections before neighborhood construction and embedding optimization have fully converged.

Prior work has explored progressive DR to reduce the initial latency. DimXplorer [29] combines incremental PCA and mini-batch k-means to support progressively updated projections and clusters during exploratory analysis. For non-linear DR, A-tSNE [21] introduces controllable approximation and user-steerable local refinement to make t-SNE usable in a progressive setting. PANENE [12] further reduces startup cost through progressive approximate kNN computation and demonstrates Responsive t-SNE as one of its applications. Progressive UMAP [14] extends this line of work to UMAP by incorporating progressive approximate kNN computation into the UMAP pipeline.

Algorithm 1 Progressive Execution in GPUMAP   
1: function $\mathrm { R U N } ( n _ { \mathrm { i n s } } , n _ { \mathrm { u p d } } , n _ { \mathrm { e m b } } )$   
2: $\mathrm { I N S E R T B A T C H } ( n _ { \mathrm { i n s } } ^ { \cdot } )$   
3: $\mathrm { U P D A T E B A T C H } ( n _ { \mathrm { u p d } } )$   
4: EMBEDB $\operatorname { A T C H } ( n _ { \operatorname { e m b } } )$   
5: end function   
6: function $\mathrm { R U N } ( t _ { \mathrm { b u d g e t } } )$   
7: Allocate $t _ { \mathrm { b u d g e t } }$ across operation types according to schedul  
ing weights   
8: Estimate the operation budget $( n _ { \mathrm { i n s } } , n _ { \mathrm { u p d } } , n _ { \mathrm { e m b } } )$   
9: $\mathrm { R U N } ( n _ { \mathrm { i n s } } , n _ { \mathrm { u p d } } , n _ { \mathrm { e m b } } )$   
10: end function

Neural parametric projection methods, such as Neural Network Projection (NNP) [7], parametric t-SNE [31], and parametric UMAP [24], also reduce projection latency by learning an explicit mapping that embeds new instances through a single forward pass. However, this speed applies only after training: they require a separate training stage on a representative subset of the data before any projection can be produced, so the latency of obtaining an initial result is shifted to, rather than eliminated by, this training stage. They also rely on a fixed mapping that does not revise the positions of previously embedded points as new data arrive. They therefore do not reduce the latency of the first projection nor progressively produce the projection of the current dataset, and are thus outside the scope of this work.

However, prior progressive DR methods have mainly been demonstrated at modest scales and with CPU-based implementations, leaving their practicality for large-scale interactive analysis unclear. Moreover, it remains insufficiently understood how closely intermediate projections approximate the final projection obtained through sequential computation and how the quality of the progressive kNN graph and projection changes throughout progressive execution. We address these gaps with a GPU-accelerated progressive UMAP method and a systematic analysis of its intermediate results.

## 3 GPUMAP

## 3.1 Overview of Progressive Execution

To progressively compute UMAP, GPUMAP decomposes the computation pipeline into a sequence of three types of operations that can be executed in an interleaved and controllable manner: INSERT, UPDATE, and EMBED as illustrated in Fig. 2.

An insertion operation, INSERT(i), incorporates a new point $x _ { i }$ into the current kNN graph, identifies approximate neighbors from the points inserted so far, and initializes its embedding. An update operation, UPDATE(i), refines the local neighborhood structure around $x _ { i }$ through a point-wise local join adapted from the NN-descent [6]. Finally, an embedding operation, EMBED(i), updates the embedding of x<sub>i</sub> using the latest graph state; the newly inserted points are processed with priority, while previously embedded points are revisited over time. Together, these operations progressively expand data coverage, improve the kNN graph, and refine the projection.

Each type of operation is managed by a dedicated queue on the host (i.e., CPU): the insertion queue $( Q _ { \mathrm { i n s } } )$ , update queue $( Q _ { \mathrm { u p d } } )$ , and two embedding queues, $Q _ { \mathrm { e m b } } ^ { \mathrm { H } }$ and $Q _ { \mathrm { e m b } } .$ The priority embedding queue $Q _ { \mathrm { e m b } } ^ { \mathrm { H } }$ contains the first embedding operation for each point and is processed with higher priority than Q .

We present two execution interfaces to the client (Algorithm 1). The first is $\mathrm { R U N } \big ( n _ { \mathrm { i n s } } , n _ { \mathrm { u p d } } , n _ { \mathrm { e m b } } \big )$ , where the client explicitly specifies an operation budget, requesting up to $n _ { \mathrm { i n s } }$ insertion operations, $n _ { \mathrm { u p d } }$ update operations, and $n _ { \mathrm { e m b } }$ embedding operations to be processed from their respective queues. Alternatively, the client can provide a wall-time bud get $t _ { \mathrm { b u d g e t } } ;$ GPUMAP allocates this budget across the three operation types according to a scheduling strategy and estimates the corresponding operation counts. Although the operations are conceptually defined at the level of individual points, GPUMAP dequeues multiple operations of the same type and executes them as GPU batches for efficiency. After each batch execution, GPUMAP produces the updated projection along with execution statistics, such as the actual time spent on each type of operation. We describe the procedures and design rationale in the following sections, while providing complete pseudocode in the supplementary material.

Algorithm 2 Insertion Operation   
1: function INSERT(i)   
2: $\mathcal { N } _ { k } ( i ) \gets \mathrm { S E E D N E I G H B O R S } ( i , \mathcal { N } )$   
3: $A \ { \stackrel { \cdots } { \longleftarrow } } \ \operatorname { R E F I N E F O R W A R D N E I G H B O R S } ( i , { \mathcal { N } } )$   
4: $y _ { i } \gets \mathrm { I N I T I A L I Z E E M B E D D I N G } ( \mathcal { N } _ { k } ( i ) , \mathbf { Y } )$   
5: $\mathrm { \dot { E } N Q U E U E } \big ( \mathcal { Q } _ { \mathrm { u p d } } , \{ i \} \cup N _ { k } ( i ) \cup \dot { A } \big )$   
6: $\operatorname { E N Q U E U E } ( Q _ { \mathrm { e m b } } ^ { \mathrm { H } } , i )$   
7: end function   
8: function REFINEFORWARDNEIGHBORS(i,N)   
9: $A  \emptyset$   
10: for $t \gets 1$ to $T$ do   
11: $C \gets \bigcup _ { j \in N _ { k } ( i ) } N _ { k } ( j ) \setminus \{ i \}$   
12: for all $c \in \mathcal { C } \backslash \{ i \}$ do   
13: $A  A \cup$ UPDATENEIGHBORS $\mathsf { i } ( i , c , \mathcal { N } )$   
14: end for   
15: end for   
16: return A   
17: end function   
18: function INSERTBATCH $( n _ { \mathrm { i n s } } )$   
19: $B _ { \mathrm { i n s } } \gets \sf D E Q U E U E ( Q _ { \mathrm { i n s } } , \operatorname* { m i n } ( n _ { \mathrm { i n s } } , | Q _ { \mathrm { i n s } } | ) )$   
20: for all $i \in B _ { \mathrm { i n s } }$ in parallel do   
21: INSERT(i)   
22: end for   
23: end function

## 3.2 The Insertion Operation

An insertion operation, INSERT(i), adds a data point $x _ { i }$ to the kNN graph and constructs its initial neighbor list using the points inserted so far. Given a dataset of M points, INSERT(1) $, \dots ,$ INSERT(M) are initially enqueued in $Q _ { \mathrm { i n s } }$ and await execution by the RUN call. The goal of each insertion is not to fully refine the neighborhood structure around $x _ { i } ,$ , but to rapidly establish a plausible local neighborhood. As illustrated in Algorithm 2, the operation consists of three main steps: neighbor seeding, forward neighbor refinement, and embedding initialization, followed by scheduling the affected points for subsequent refinement.

Neighbor Seeding. To initialize the neighbor list of the newly inserted point i, GPUMAP first randomly samples half of the candidates from the most recently inserted points and the remaining half from all points inserted so far. The original NN-descent algorithm randomly selects a seed set of neighbors from the entire dataset, but this is infeasible in progressive execution, as the algorithm has no access to points that have not yet been inserted. A straightforward alternative is to sample uniformly from the set of already inserted points, but this introduces bias: earlier points are more likely to be selected as initial neighbors, causing the graph to grow around the initial structure. To mitigate this bias, we maintain a recent pool $( { \mathcal { W } } )$ , defined as a sliding window over the most recent $W$ batches $( W = 3$ by default). For example, after inserting nine batches $B _ { 1 }$ through $B _ { 9 }$ , when inserting the 10th batch $B _ { 1 0 } .$ , the seed neighbors for each point in $B _ { 1 0 }$ are sampled such that half are drawn uniformly from all inserted points $( B _ { 1 }$ to $B _ { 1 0 } )$ , and the other half are drawn from the recent pool $( B _ { 8 }$ to $B _ { 1 0 } )$

Forward Neighbor Refinement. The random initial neighbor list of point i should be refined. The original NN-descent performs this refinement by repeatedly running local join; for example, it computes distances for every pair within the union of forward neighbors $N _ { k } ( i )$ and backward neighbors $N _ { k } ^ { - 1 } ( i )$ . We instead consider only secondorder forward neighbors as candidates, i.e., $C = \cup _ { j \in N _ { k } ( i ) } N _ { k } ( j ) \setminus \{ i \}$ the neighbors of i’s current neighbors. This is because (1) our goal is the rapid identification of plausible neighbors for i not global graph refinement and (2) computing backward neighbors incurs larger computational overhead. The local join on i, including backward neighbors, is deferred to the subsequent UPDATE operation. For each candidate in C, we compute the distance to i and update the neighbor lists of both points accordingly. Any point whose neighbor list changes is added to the affected set A, which records points that require subsequent UPDATE processing. This refinement is repeated $T = \bar { 1 0 }$ times by default.

![](images/05e9243d1e3ed4c7f11cd097ccbf327bd899453574766cded422f0ed791e7705.jpg)  
Figure 2: Overview of the three operations in the GPUMAP pipeline. (A) INSERT: points are dequeued from $Q _ { \mathrm { i n s } } ,$ inserted into the kNN graph with neighbors initialized via forward NN-descent, and their initial embeddings are set based on their neighbors; the inserted points are then enqueued into $\mathcal { Q } _ { \sf u p d }$ for neighbor refinement and $Q _ { \mathsf { e m b } } ^ { \mathsf { H } }$ for initial embedding optimization with priority. (B) UPDATE: points are dequeued from $\mathcal { Q } _ { \sf u p d }$ to have their neighbors refined, and the kNN graph is refined via local join; points whose neighbors change are enqueued back into $\mathcal { Q } _ { \sf u p d }$ for further refinement. (C) EMBED: points are dequeued from $Q _ { \mathsf { e m b } } ^ { \mathsf { H } }$ and $Q _ { \mathsf { e m b } }$ to optimize their embeddings; after optimization, points are enqueued into $Q _ { \mathsf { e m b } }$ for future revisits.

Embedding Initialization. Once the neighbor list of point i has been refined, we initialize its embedding as an exponentially weighted average of its neighbors’ embeddings, where the weights are determined by their distances. This is motivated by $\mathrm { U M A P ^ { \circ } s }$ use of an exponential kernel to model membership strength.

Queueing. At the end of the insertion operation, we enqueue UPDATE operations for points affected by the insertion. These points include the newly inserted point i, its neighbors in $N _ { k } ( i )$ , and the existing points whose neighbor lists were updated to include i during forward neighbor refinement (Line 5 in Algorithm 2). By restricting subsequent refinement to these points, GPUMAP avoids unnecessary computation on unaffected regions of the neighbor graph. Finally, an EMBED(i) operation is enqueued into the priority embedding queue $Q _ { \mathrm { e m b } } ^ { \mathrm { H } } ,$ so that the embedding of i is optimized with higher priority, as it has only been initialized but not yet optimized.

## 3.3 The Update Operation

An update operation, UPDATE(i), refines the kNN graph around point i by running the local join algorithm [6], propagating neighborhood information to points not directly touched by insertion. Rather than sweeping over all points at once as in the original NN-descent, we decompose this into independent UPDATE(i) calls, each queued in $Q _ { \mathrm { u p d } }$ and triggered only for points whose neighbor lists changed during a preceding INSERT or UPDATE. Algorithm 3 illustrates the procedure for a single point.

Candidate Sampling. For each UPDATE(i), GPUMAP collects nearby points that may reveal new neighbor relationships through a local join. These candidates are drawn from the forward and sampled backward neighbors of i. Following the original NN-descent, GPUMAP distinguishes candidates newly introduced to the local join from those already processed. This distinction allows the subsequent local join to focus on pairs involving new neighborhood information and avoid repeating comparisons between previously processed candidates. We denote the resulting sets by $C _ { \mathrm { n e w } }$ and $C _ { \mathrm { o l d } } .$

Local Join. GPUMAP evaluates candidate pairs consisting of at least one new candidate: new–new and new–old pairs (PAIRS in Algorithm 3); old–old pairs are skipped, since both points were already compared in a preceding update. For each pair, GPUMAP computes the distance between the two points and updates their neighbor lists if either point provides the other with a closer neighbor. Any point whose neighbor list changes is likewise added to A.

Algorithm 3 Update Operation   
1: function UPDATE(i)   
2: C<sub>new</sub>,C<sub>old</sub> ← SAMPLECANDIDATES $( i , \mathcal { N } )$   
3: A ← LOCALJOIN(C<sub>new</sub>,C<sub>old</sub>)   
4: $\operatorname { E N Q U E U E } ( Q _ { \mathrm { u p d } } , \dot { A } )$   
5: end function   
6: function LOCALJOIN(C<sub>new</sub>,C<sub>old</sub>)   
7: A ← /0   
8: for each $( c _ { 1 } , c _ { 2 } ) \in \mathrm { P A I R S } \big ( C _ { \mathrm { n e w } } , C _ { 0 \mathrm { l d } } \big )$ do   
9: A ← A∪ UPDATENEIGHBORS $( c _ { 1 } , c _ { 2 } , \mathcal { N } )$   
10: end for   
11: return A   
12: end function   
13: function UPDATEBATCH(n<sub>upd</sub>)   
14: $n _ { \mathrm { l e f t } }  n _ { \mathrm { u p d } }$   
15: while $| Q _ { \mathrm { u p d } } | \geq \delta \cdot M ^ { \prime }$ and $n _ { \mathrm { l e f t } } > 0$ do   
16: $B _ { \mathrm { u p d } }  \mathrm { N E X T B A T C H } ( \mathcal { Q } _ { \mathrm { u p d } } , n _ { \mathrm { l e f t } } , \mathcal { M } ^ { \prime } )$   
17: for all $i \in B _ { \mathrm { u p d } }$ in parallel do   
18: UPDATE(i)   
19: end for   
20: $n _ { \mathrm { l e f t } }  n _ { \mathrm { l e f t } } - | { B _ { \mathrm { u p d } } } |$   
21: end while   
22: end function

Queueing. After each UPDATE(i), GPUMAP enqueues the affected points in A into $Q _ { \mathrm { u p d } }$ . This allows newly discovered neighbor relationships to propagate incrementally through subsequent updates while avoiding computation on points whose neighbor lists remain unchanged.

Batch Scheduling. Beyond a single point, UPDATEBATCH schedules UPDATE calls across many points within a fixed budget $n _ { \mathrm { u p d } }$ per round. Each round, it draws a batch of points via NEXTBATCH and runs UPDATE on them in parallel. NEXTBATCH dequeues as many points as available from $Q _ { \mathrm { u p d } } .$ . If fewer points remain than the round’s budget, it supplements the batch with points randomly sampled from $\mathcal { M } ^ { \prime } \backslash B _ { \mathrm { u p d } } ,$ , where ${ \mathcal { M } } ^ { \prime }$ denotes the set of all points inserted so far. This covers points that still carry unprocessed new flags but were missed by backward sampling. The process repeats until either the budget $n _ { \mathrm { u p d } }$ is exhausted or $| Q _ { \mathrm { u p d } } |$ falls below the convergence threshold $\delta \cdot M ^ { \prime }$ (we use $\delta = 0 . 0 0 1$ , following the original NN-descent).

## 3.4 The Embedding Operation

An embedding operation. EMBED(i), refines the low-dimensional position of the point i based on the current kNN graph extended and refined by the INSERT and UPDATE operations. Unlike standard UMAP, which optimizes the embedding over all points using the full dataset, our progressive setting allocates a fixed computational budget of $n _ { \mathrm { e m b } }$ points to each run. Accordingly, each embedding run processes at most $n _ { \mathrm { e m b } }$ points, making the execution time controllable. The overall procedure for executing embedding operations is summarized in Algorithm 4.

Algorithm 4 Embedding Operation   
1: function EMBED(i)   
2: ADDPERTURBATION(y ) ▷ add small perturbation to y   
3: $E _ { i } \gets \{ ( i , j , \nu _ { i j } ) \ : | \ : j \in N _ { k } ( i ) , \nu _ { i j } \in \mathcal { G } \}$   
4: return $E _ { i }$   
5: end function   
6: function $\mathrm { E M B E D B A T C H } ( n _ { \mathrm { e m b } } )$   
7: $B _ { \mathrm { e m b } } \gets \sf { D E Q U E U E } ( Q _ { \mathrm { e m b } } ^ { \mathrm { H } } , n _ { \mathrm { e m b } } )$   
8: $B _ { \mathrm { e m b } } \gets B _ { \mathrm { e m b } } \cup \underline { { \mathrm { D E Q U E U E } } } ( Q _ { \mathrm { e m b } } , \operatorname* { m a x } ( 0 , n _ { \mathrm { e m b } } - | B _ { \mathrm { e m b } } | ) )$   
9: G ← COMPUTEFUZZYGRAPH(N)   
10: $\mathcal { E }  \emptyset$   
11: for all $i \in B _ { \mathrm { e m b } }$ in parallel do   
12: $\mathcal { E } ^ { {  } }  \mathcal { E } ^ { { \lfloor } } \lfloor$ EMBED(i)   
13: end for   
14: for $e \gets 1$ to $e _ { m a x }$ do ▷ synchronize epoch progression   
15: for all $( i , j , \nu _ { i j } ) \in \mathcal { E }$ in parallel do   
16: UPDATEEDGE $( i , j , \nu _ { i j } , e )$   
17: end for   
18: end for   
19: for all $k \in \mathcal { M } ^ { \prime } \backslash B _ { \mathrm { e m b } }$ in parallel do   
20: PROPAGATEPOSITION $\scriptstyle { \lceil ( k , Y , \mathcal { G } ) }$   
21: end for   
22: $\operatorname { E N Q U E U E } ( Q _ { \mathrm { e m b } } , B _ { \mathrm { e m b } } )$   
23: end function

An alternative design for allocating budgets would be to partition embedding updates at the epoch level, executing a few epochs for all points. While we initially explored this approach, it yielded suboptimal results. In particular, interrupting the optimization before completing all $e _ { \mathrm { m a x } }$ epochs led to unstructured and scattered embeddings. Instead, GPUMAP completes all epochs for the selected points and propagates the resulting changes to the remaining points, producing a more coherent partial layout.

Batch-wise Optimization. Although EMBED(i) is defined for an individual point, embedding operations are executed jointly in GPU batches. Given a budget of $n _ { \mathrm { e m b } }$ , EMBEDBATCH selects up to n<sub>emb</sub> points from the embedding queues and constructs the fuzzy graph G from the current kNN graph following standard UMAP. For each selected point i, EMBED(i) identifies its incident fuzzy graph edges E<sub>i</sub> as the targets of direct optimization. EMBEDBATCH combines the edge sets from all selected points and optimizes their union over a single shared loop of $e _ { \mathrm { m a x } }$ epochs. Finally, PROPAGATEPOSITION updates the remaining points to reflect the layout changes produced by the batch. This formulation allows GPUMAP to allocate its embedding budget by point while retaining the edge-parallel optimization of standard UMAP.

Maintaining Embedding Quality. We employ two mechanisms to maintain embedding quality across repeated, partial optimization runs. First, repeatedly optimizing the same points can lead to poor local optima and excessive contraction of local structures. To alleviate this issue, we add a small Gaussian perturbation to the points before embedding operations (ADDPERTURBATION in Algorithm 4). Second, because points outside $B _ { \mathrm { e m b } }$ receive no direct edge updates, we update their positions by repositioning to the weighted average of their neighbors’ embeddings (PROPAGATEPOSITION in Algorithm 4), which is a cheap operation.

Queueing. GPUMAP maintains two embedding queues, $Q _ { \mathrm { e m b } } ^ { \mathrm { H } }$ (high priority) and $Q _ { \mathrm { e m b } }$ (low priority), to control the order in which points undergo optimization. Without this separation, newly inserted points would be placed at the end of a single queue and could be delayed for several runs, resulting in poor embedding quality. Points are drawn from $Q _ { \mathrm { e m b } } ^ { \mathrm { H } }$ first, and if the embedding budget $n _ { \mathrm { e m b } }$ permits, additional points are drawn from $Q _ { \mathrm { e m b } } .$ . After each embedding run, all processed points, including those from $Q _ { \mathrm { e m b } } ^ { \mathrm { H } } ,$ are inserted into $Q _ { \mathrm { e m b } }$ again to enable continued refinement if either $Q _ { \mathrm { i n s } } \thinspace \mathrm { o r } \ Q _ { \mathrm { u p d } }$ is non-empty. This design can be viewed as a round-robin update scheme over all inserted points, ensuring that each point is revisited over time.

## 3.5 Execution Span Approximation

While GPUMAP supports incremental computation of UMAP by executing a specified number of operations of each type, users may instead wish to run the computation for a fixed time budget (e.g., 10 seconds), allowing the algorithm to determine how many operations can be executed within that time. To translate a physical time budget into the number of operations, we adopt a simple linear regression strategy similar to that used in ProgressiVis [8]. Specifically, we record the execution time of each operation type over the last five runs and use these measurements to predict the per-operation cost in the current run, while reserving a small portion of the budget to account for estimation error (Algorithm 1).

When no execution history is available, runtime estimation must rely on an initial calibration. A straightforward approach is to perform several warm-up runs with a small number of operations; however, this is inefficient because, for a fixed hardware configuration and dataset dimensionality, the runtime characteristics remain largely stable. Therefore, we store runtime statistics in a local database and reuse them in future runs. If no such record exists (e.g., during the first run on a given dataset and hardware configuration), we perform a short calibration procedure that begins with a very small number of operations (e.g., 1,000 operations per type) and repeatedly doubles the number of operations to initialize the regression model.

## 3.6 Scheduling

Given a fixed time budget $t _ { \mathrm { b u d g e t } } ,$ GPUMAP determines how to allocate this budget across the three operation types, which we refer to as a scheduling strategy. As a baseline, we define Sequential, a nonprogressive scheduling strategy equivalent to invoking $\mathbf { R U N } \big ( \infty , \infty , \infty \big )$ thereby ignoring the budget. Under Sequential, all INSERT operations are completed first, followed by UPDATE, and finally EMBED, making it the execution mode in GPUMAP that most closely resembles standard UMAP. We use this strategy both as a sanity check and as a controlled baseline for runtime comparison with progressive DR.

We parametrize each scheduling strategy by a weight vector $W =$ $( w _ { \mathrm { i n s } } , w _ { \mathrm { u p d } } , w _ { \mathrm { e m b } } )$ , which determines how the budget is distributed across INSERT, UPDATE, and EMBED. We define a baseline scheduling strategy Update-Focused with $W = \left( 1 , 2 , 1 \right)$ ; this update-biased weighting reflects our empirical observation that the quality of the kNN graph is the dominant factor influencing projection quality. We further define two variants, Insert-Focused and Embed-Focused, which increase the relative emphasis on INSERT and EMBED, respectively, by doubling their weights, resulting in $W = \left( 2 , 1 , 1 \right)$ and $W = \left( 1 , 1 , 2 \right)$

## 3.7 Implementation

GPUMAP is implemented in CUDA/C++ and exposed to Python via pybind11 [10]<sup>1</sup>. We build upon the NVIDIA RAPIDS cuML library [22], leveraging its parallel primitives, including GPU kernels for NN-descent and UMAP, while introducing our modifications to en able progressive execution. At present, GPUMAP supports single-GPU execution only and does not include multi-GPU support.

## 4 EVALUATION

Although several progressive DR methods have been proposed [12, 14, 21], it remains insufficiently understood how the quality of their intermediate results evolves over time and how progressive execution affects computational performance. In this section, we address this gap by using GPUMAP as a concrete realization of progressive DR and standard UMAP, denoted as $\mathrm { U } _ { \mathrm { s t d } } .$ , as its non-progressive counterpart. We first verify that the progressive decomposition of UMAP preserves the final neighborhood and projection quality across a diverse collection of datasets. We then examine how these qualities change throughout execution under different scheduling strategies and time budgets. Next, we compare GPUMAP with Progressive UMAP [14] to assess how efficiently each method improves projection fidelity and neighborhood quality over time. Finally, we examine the robustness of progressive execution under biased data insertion orders.

Table 1: Summary of datasets used in the evaluation.
<table><tr><td>Dataset</td><td>Size (M× dims)</td><td>Data Type</td></tr><tr><td>EMNIST Digits [4]</td><td> $2 8 0 , 0 0 0 \times 7 8 4$ </td><td>Image</td></tr><tr><td>Covertype [3]</td><td> $5 8 1 { , } 0 1 2 \times 5 4$ </td><td>Table</td></tr><tr><td>SVHN [19]</td><td> $6 3 0 { , } 4 2 0 \times 7 6 8$ </td><td>Image</td></tr><tr><td>SIFT1M [11]</td><td> $1 , 0 0 0 , 0 0 0 \times 1 2 8$ </td><td>Image</td></tr><tr><td>MARC [13]</td><td> $1 , 2 0 0 , 0 0 0 \times 7 6 8$ </td><td>Text</td></tr><tr><td>Mouse Brain [1]</td><td> $1 , 3 0 6 , 1 2 7 \times 5 0$ </td><td>Genomic</td></tr><tr><td>LAION [25]</td><td> $2 , 0 0 0 , 8 9 6 \times 5 1 2$ </td><td>Image</td></tr><tr><td>Google News [18]</td><td> $3 , 0 0 0 , 0 0 0 \times 3 0 0$ </td><td>Text</td></tr><tr><td>Spheres</td><td> $1 , 0 0 0 , 0 0 0 \times 3$ </td><td></td></tr><tr><td>Cylinder</td><td></td><td>Synthetic</td></tr><tr><td></td><td> $1 , 0 0 0 , 0 0 0 \times 3$ </td><td>Synthetic</td></tr></table>

Table 2: Results of sanity checks.
<table><tr><td>Dataset</td><td>Time (s)</td><td>Final Recall</td><td>Norm. Fidelity</td></tr><tr><td>EMNIST Digits</td><td> $1 9 . 9 7 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $0 . 9 9 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 9 8 { \scriptstyle \pm 0 . 0 2 }$ </td></tr><tr><td>Covertype</td><td> $2 7 . 4 1 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td></tr><tr><td>SVHN</td><td> $4 7 . 0 1 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $0 . 9 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 9 4 { \scriptstyle \pm 0 . 0 2 }$ </td></tr><tr><td>SIFT1M</td><td> $5 6 . 1 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.86±0.00</td></tr><tr><td>MARC</td><td> $9 5 . 8 1 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $0 . 8 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 7 4 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>Mouse Brain</td><td>68.88±0.70</td><td> $0 . 9 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td>1.00±0.00</td></tr><tr><td>LAION</td><td> $1 4 1 . 3 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $0 . 9 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.96±0.02</td></tr><tr><td>Google News</td><td> $1 9 7 . 8 6 { \pm } 1 . 1 4$ </td><td> $0 . 8 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>0.96±0.01</td></tr></table>

## 4.1 Experimental Setup

Datasets. We evaluate GPUMAP on a collection of real-world datasets spanning image, tabular, genomic, and text data, with dataset sizes ranging from hundreds of thousands to several million instances (Tab. 1). For the robustness assessment, we additionally use two synthetic datasets, Spheres and Cylinder, whose geometric structures and insertion orders can be explicitly controlled. Detailed descriptions and preprocessing procedures for each dataset are provided in the supplementary material.

Configurations. We use two baseline implementations. First, we use RAPIDS UMAP [20] as the specific implementation of $\mathrm { U } _ { \mathrm { s t d } }$ . Second, we use Progressive UMAP [14] as the prior progressive baseline. We do not use the official UMAP implementation [17], as it is neither GPUaccelerated nor progressive. For hyperparameters shared with $\mathrm { U } _ { \mathrm { s t d } } ,$ we use the default UMAP settings across experiments, except that cosine distance is used for the text embedding datasets. All UMAP-based methods use random initialization rather than spectral initialization, so that the comparisons do not rely on a global initialization computed from the full dataset. For GPUMAP-specific parameters, we use a fixed configuration across datasets unless otherwise specified; the complete parameter settings are reported in the supplementary material. For experiments that examine scheduling behavior, we use the four strategies defined in Sec. 3.6. All experiments were conducted on a desktop equipped with an AMD Ryzen 9 5950X CPU, 128GB of DDR4 RAM, and an NVIDIA RTX A6000 GPU with 48GB of VRAM.

Time Budgets. For progressive execution, we primarily use a per-run time budget of $t _ { \mathrm { b u d g e t } } = 1 0 \mathrm { s }$ . We additionally evaluate a 5-second budget to examine how more frequent updates affect progressive behavior and computational throughput.

## 4.2 Metrics

We evaluate GPUMAP in terms of execution time, nearest neighbor recall, and projection fidelity.

Execution Time. We measure the overall wall-clock time measured from the Python interface after all operations for each RUN call have completed. To isolate the performance of the algorithm itself, we assume a setting without a client; accordingly, external overheads such as network transmission and client-side rendering are excluded.

Nearest Neighbor Recall. We evaluate the quality of the constructed kNN graph using nearest neighbor recall. We measure final recall, which compares the current kNN graph against the ground-truth kNN graph over the full dataset. Note that the final recall is strictly upperbounded by the proportion of data inserted so far; since the kNN graph is built over only the points inserted up to the current iteration, it cannot capture neighbors that have yet to be inserted.

Normalized Fidelity. We quantify how faithfully a progressive projection preserves the structure of the sequentially computed projection (hereafter referred to as the ground-truth projection), which we denote as the fidelity of the progressive projection. We initially considered standard DR quality metrics such as Trustworthiness and Continuity [33]. However, these metrics are not suitable for our setting for two reasons. First, they are designed to measure the alignment between high- and low-dimensional spaces, rather than the similarity between two projections. Second, similar scores under these metrics do not necessarily imply that the projections themselves are structurally similar.

We found that measuring the fidelity of a progressive projection is not yet well established, and identified three fundamental challenges. First, at any intermediate iteration, only a subset of $M ^ { \prime } \leq M$ data points have been projected, whereas the ground-truth projection covers all M points. Comparing the two projections, therefore, requires a way to fill in the missing projections of $\bar { M } - M ^ { \prime }$ points. Second, two projections of the same data may differ substantially in their absolute coordinates despite preserving similar structures, making direct coordinate- or pixellevel comparisons inappropriate. Finally, UMAP and its progressive variants are stochastic algorithms due to the use of approximate kNN, stochastic gradient descent, and negative sampling. Therefore, a single reference projection is an insufficient ground truth.

To address the first challenge, we employ density-based filling; we fill the unprojected points by sampling positions proportional to the empirical density of the projected points. Concretely, we discretize the current projection into a $1 2 8 \times 1 2 8$ grid, estimate a density histogram, and draw positions for the missing points by first sampling a grid cell according to those counts and then adding Gaussian jitter of half a cell width. This preserves the global density structure of the progressive projection while its size matches the ground-truth projection.

For the second challenge, instead of comparing absolute coordinates, we compare the kNN of each point in a progressive projection with ground-truth neighbor sets. We define the per-point fidelity of point i as $\begin{array} { r } { f _ { i } = \frac { | N _ { k } ( i ) \cap \mathcal { G } _ { i } | } { k } } \end{array}$ where $N _ { k } ( i )$ is i’s kNN in the progressive projection and G the ground-truth neighbor set. The overall fidelity is $\begin{array} { r } { F = \frac { 1 } { M } \sum _ { i = 1 } ^ { M ^ { \prime } } f _ { i } . } \end{array}$

To overcome the third challenge, we generate R independent UMAP projections and take the union of their kNN sets as the ground-truth neighbor set for each point: $\mathscr { G } _ { i } = \bigcup _ { r = 1 } ^ { R } N _ { k } ^ { ( r ) } ( i )$ , where $N _ { k } ^ { ( r ) } ( i )$ is the kNN set of point i in the r-th UMAP run. Empirically, we increase r up to 300 UMAP runs and stop earlier when the average fraction of newly observed neighbors, normalized by k, falls below 0.2. We defer the justification of this stopping rule to the supplementary material.

In principle, the fidelity of a progressive projection can reach 1.0. However, we observed that even a newly computed ground-truth projection achieves a fidelity score below 1.0 due to inherent stochasticity. To account for this, we normalize the fidelity by the mean fidelity F<sub>UMAP</sub> of $\mathrm { U } _ { \mathrm { s t d } }$ runs, yielding the normalized fidelity: $\begin{array} { r } { \hat { F } = \frac { F } { F _ { \mathrm { U M A P } } } } \end{array}$ . A value of $\hat { F } = 1 . 0$ indicates that the progressive projection is as faithful as a newly computed ground-truth projection.

![](images/a73281f20881da89eddc980c7f81455f83126ccc0dd87fd6f5c9224f9630d179.jpg)  
Figure 3: Comparison of scheduling strategies in terms of normalized fidelity, final recall, and per-run wall-time distribution. In the top two rows, blue points indicate the final result of the sequential baseline (Sequential). The bottom row shows the per-run wall-time distribution of the Update-Focused strategies under a 10-second budget, with the red dashed line indicating the time budget.

![](images/e0743f03fd6927f8a124a5bf60a6a02005b1cb6e9818005bd2688de3c218f3ab.jpg)  
Figure 4: Comparison of two time-budget settings (5 seconds and 10 seconds) on Mouse Brain and LAION, showing normalized fidelity and final recall over cumulative wall time.

## 4.3 Sanity Checks

We first verify whether GPUMAP produces final results consistent with $\mathrm { U } _ { \mathrm { s t d } }$ when progressive scheduling is disabled. Specifically, we run GPUMAP using the Sequential strategy and compare its final recall and normalized fidelity with those obtained from $\bar { \mathrm { U } } _ { \mathrm { s t d } }$ (Tab. 2). Across most datasets, Sequential achieves high final recall and normalized fidelity, indicating that the modifications introduced by GPUMAP largely preserve the final behavior of $\mathrm { U } _ { \mathrm { s t d } }$ . The two text embedding datasets show comparatively lower final recall, while still maintaining moderate to high normalized fidelity.

## 4.4 Progressive Behavior

Fidelity and Scheduling Strategies. The first and second rows of Fig. 3 show the normalized fidelity and final recall, respectively, of different scheduling strategies over cumulative wall time. Sequential is shown as a single point, as it does not produce intermediate kNN graphs or projections before completing the full computation. In contrast, all progressive strategies expose intermediate results well before Sequential completes.

Among the evaluated strategies, Update-Focused shows the most balanced improvement in both normalized fidelity and final recall. This behavior is consistent with the relative amount of work required by the three operation types in our pipeline: neighborhood refinement generally requires more UPDATE operations than the corresponding insertion or embedding operations. Allocating additional budget to UPDATE therefore allows neighborhood refinement to better keep pace

with newly inserted points.

In contrast, Insert-Focused shows that normalized fidelity and final recall improve relatively slowly during insertion and then rise sharply once insertion is completed. After this point, the budget previously allocated to insertion can be redirected to the remaining update and embedding operations, allowing the neighborhood graph and projection to rapidly catch up. Embed-Focused shows a broadly similar pattern, although its trajectory generally lies between Insert-Focused and Update-Focused. Overall, these results suggest that scheduling mainly determines how well neighborhood refinement keeps pace with data insertion during progressive execution, while the differences among strategies become less pronounced after insertion is completed.

Time Budgets. We examine time budgets from two perspectives: (1) how faithfully the system adheres to a specified per-run budget, and (2) how different budget settings affect progressive behavior. We use the Update-Focused strategy for both analyses.

To assess the first aspect, we inspect the distribution of wall time between consecutive runs (third row of Fig. 3). Across datasets, the observed wall times remain close to the specified budget, although some runs terminate earlier when the number of available operations in the queues is insufficient. This indicates that the runtime of GPUMAP is predictable and can be controlled by adjusting the number of operations of each type, even with simple linear regression predictors.

For the second aspect, Fig. 4 compares $t _ { \mathrm { b u d g e t } } = 5 \mathrm { s }$ and 10s on Mouse Brain and LAION. We initially expected the shorter budget to require a longer total execution time because more frequent runs could incur greater progressive overhead. Contrary to this expectation, the 5-second setting completed earlier than the 10-second setting on both datasets. However, halving the time budget does not necessarily double the number of runs, since the amount of available work varies throughout progressive execution and several runs terminate before exhausting the allocated budget. Consequently, we did not observe a clear relationship between the per-run time budget and the total execution time.

A more noticeable difference appears in normalized fidelity. While the two budget settings exhibit relatively similar fidelity trajectories on Mouse Brain, the 5-second setting shows larger fluctuations on LAION. This observation suggests that shorter time budgets may lead to less stable intermediate projection quality, particularly for larger-scale data.

Overhead. We next quantify the computational cost introduced by supporting progressive execution, using Sequential as a non-progressive reference. For Update-Focused, the total runtime is 1.3×, 1.6×, and 1.8× that of Sequential across the three datasets, respectively. This overhead arises not only from coordination required for bounded-latency execution, such as CPU–GPU synchronization, but also because progressive execution performs additional insertion, update, and embedding operations to maintain reasonably high-fidelity intermediate results throughout the run. In sum, progressive execution introduces a clear trade-off between throughput and the ability to access faithful intermediate results during execution.

![](images/57ea73c1747823f48a27d64f0d0f170c949dd1d6d7c47e13ac20c0c1c2fa8ffa.jpg)  
Figure 5: Comparison of GPUMAP and PUMAP on EMNIST Digits, showing normalized fidelity and final recall over cumulative wall time.

## 4.5 Comparison with Progressive UMAP

We compare GPUMAP with Progressive UMAP (PUMAP) [14], a prior CPU-based progressive implementation of UMAP. Because PUMAP becomes impractical for the larger datasets in our evaluation, we perform the comparison on EMNIST Digits, the smallest real-world dataset in our benchmark with 280K points. PUMAP controls progressive computation using a fixed number of operations per batch rather than a wall-time budget. Following its original two-stage execution scheme, which allocates a larger batch at initialization and smaller batches thereafter, we use an initial batch of 60,000 points followed by batches of 6,000 points to adapt the method to the larger EMNIST Digits dataset. We otherwise follow the settings of the original implementation.

As shown in Fig. 5, GPUMAP improves both normalized fidelity and final recall substantially faster than PUMAP. GPUMAP reaches near-reference levels for both metrics within tens of seconds, whereas PUMAP remains at considerably lower fidelity and recall even after substantially longer cumulative execution time. Together with the difficulty of running the CPU-based implementation on our larger datasets, these results highlight the practical scalability advantage of GPUMAP.

## 4.6 Robustness Assessment

Fig. 6 illustrates how the projections change over time for the synthetic datasets under biased insertion orders. Although inputs would typically be shuffled at random in practice, we deliberately construct extreme insertion orders to evaluate the robustness of progressive DR under unfavorable ordering conditions.

Spheres. Spheres consists of ten compact 3D spherical clusters, each containing 100,000 points, with their centers evenly arranged along a circle. To introduce a strong temporal bias, we insert all points from each sphere contiguously before proceeding to the next sphere. Under this ordering (Fig. 6-Spheres), we observe an incorrect split of a sphere in the early projection (e.g., the blue cluster in the first projection). We attribute this artifact to stochastic initialization: when points from a new sphere $( \mathrm { e . g . }$ , the red sphere) begin to arrive, they are randomly placed near the center, forming an emerging cluster. Due to negative sampling, this newly formed structure exerts repulsive forces that can temporarily disrupt and split an existing cluster. Despite this early instability, the split is progressively resolved in subsequent projections, and the embedding ultimately converges to a clean and well-separated layout across all ten spheres.

Cylinder. Cylinder consists of points sampled from the lateral surface of a cylinder and labeled by angular sector, with one sector intentionally left empty to create a controlled geometric gap. To create a strongly biased temporal ordering, we insert the sectors in an interleaved order $( 1 , 9 , 2 , 8 , \ldots )$ , which separates spatially adjacent regions in the insertion sequence. Under this ordering (Fig. 6-Cylinder), the early projections already preserve the separation among the currently inserted sections. As progression continues, the currently inserted sections remain separated until the intermediate regions that connect them are introduced; as these points arrive, the separated structures gradually become connected through the center, progressively revealing the global relationships among the data observed so far. This suggests that the intermediate projections reflect the structure of the currently inserted subset reasonably well. At the same time, because newly inserted points are initialized near the center of the projection, points inserted earlier tend to be pushed outward as additional data arrive, suggesting that biased insertion continues to influence the progressive layout even while the projection continues to reflect the global relationships among the inserted regions.

Comparison with Standard UMAP. In Fig. 6, we also show reference projections generated by $\mathrm { U } _ { \mathrm { s t d } }$ . Surprisingly, $\mathrm { U } _ { \mathrm { s t d } }$ with random initialization fails to recover the underlying structure, reaffirming its sensitivity to initialization [15]. This behavior can be attributed to strong repulsive forces from negative sampling, which may dominate before meaningful local structures are established, leading to local minima that are difficult to escape. In contrast, progressive DR maintains a manageable number of points at each iteration, leaving sufficient space in the projection for points to move and reorganize freely, resulting in more faithful projections.

## 5 DISCUSSION

Our evaluation highlights both the practical opportunities and behavioral characteristics of progressive DR. In this section, we discuss how early intermediate projections can support interactive analysis, interpret projection fidelity over time, and summarize the limitations of the current approach.

## 5.1 Use Cases

Large-Scale Data Exploration. Fig. 1 demonstrates a use case of GPUMAP on the HIGGS dataset with 5M points [35]. While $\mathrm { U } _ { \mathrm { s t d } }$ provides its first projection only after completing the full computation at 50.4 seconds, GPUMAP produces intermediate projections as early as 8.42 seconds and continues to refine them throughout execution. The purpose of these early results is not to replace the final projection, but to allow users to begin inspecting the data while the computation is still running. By providing updates within a short and predictable latency, GPUMAP can keep the analysis loop active rather than forcing users to wait for the full projection and resume their analysis afterward. This distinction may become increasingly important when the completion time is not known in advance or grows with the dataset size.

Early Assessment of Data Quality and Transformations. DR is often applied after preprocessing steps such as feature transformation, normalization, or scaling, and errors or inappropriate choices in these steps can lead to undesirable projections. With a conventional sequential workflow, such problems become visible only after the full computation has completed. In contrast, an early projection from GPUMAP can provide a rapid sanity check of the current input, allowing users to detect unexpected structures and revise the preprocessing pipeline before spending the full computation time on an unsuitable configuration.

Hyperparameter Search. UMAP projections can vary substantially with hyperparameter choices such as the number of neighbors, minimum distance, and distance metric. When exploring such configurations, users often need an initial indication of how a parameter choice affects the emerging structure before deciding whether it is worth completing the computation. Intermediate projections from GPUMAP can therefore support a more iterative exploration process.

Tunable Update Frequency. In an integrated visualization pipeline, each update also incurs data transfer and rendering costs, which are not included in our experiments. Our implementation lets users adjust the frequency of progressive updates to account for these costs. An analysis may begin with a small time budget to obtain frequent early updates and establish an initial understanding of the data. Once frequent inspection is no longer necessary, the budget can be increased to avoid the transfer and rendering costs of updates the user no longer needs.

## 5.2 Progressive Behavior over Time

Based on our observation, we model the changes in the projection fidelity of progressive DR (Fig. 7), in which the timeline is divided into three distinct phases: the Bootstrap Phase, the Refinement Phase, and the Convergence Phase.

![](images/f29490cd6c3f1151ccd89f1ed8b4e18a5b002cd95a7fe0b6bd49cfb455fce09e.jpg)  
Figure 6: Progressive behavior of GPUMAP on synthetic datasets with controlled geometric structures and insertion orders. The left column show the original 3D structures, the middle columns show intermediate GPUMAP projections as the number of inserted points M<sup>′</sup> increases, and the right columns show standard UMAP projections with random and spectral initialization for reference.

![](images/4e55e0e91b89d5147870b4242c2f16642b67ee98fa189c0c1d9e87ba291e142c.jpg)  
Figure 7: Projection fidelity of progressive DR over time.

Bootstrap Phase. During the Bootstrap Phase, progressive DR rapidly constructs an initial partial projection as the first subsets of data are incorporated. Although this early projection is still approximate, normalized fidelity generally increases rapidly during this stage, allowing useful structure to emerge before the full computation completes. The duration of this phase varies across workloads rather than being determined by a fixed amount of wall-clock time.

Refinement Phase. During the Refinement Phase, progressive DR continuously updates the projection by incrementally adding data points. The fidelity improvement during this phase is gradual, as the projection must balance consistency with previously placed points while accommodating newly introduced ones. We even observed temporary fidelity drops depending on the dataset and scheduling strategy. Notably, sequential DR completes during this phase at time $t _ { \mathrm { s e q } } ,$ , at which point the fidelity gap $\Delta \hat { F } ( t _ { \mathrm { s e q } } ) = f _ { \mathrm { s e q } } - \hat { F } _ { \mathrm { p r o } } ( t _ { \mathrm { s e q } } )$ represents the disadvantage of progressive DR: had the user opted for sequential DR, they would have already obtained a complete projection.

Convergence Phase. In the Convergence Phase, the projection stabilizes as the majority of data points have been incorporated, rapidly approximating the sequential ground truth. The final fidelity difference $\dot { \Delta \hat { F } } ( t _ { \mathrm { p r o } } ) = f _ { \mathrm { s e q } } ^ { \mathrm { ^ { * } } } - \hat { F } _ { \mathrm { p r o } } ^ { \mathrm { ^ { * } } } ( t _ { \mathrm { p r o } } )$ represents the loss attributable to incremental computation. Surprisingly, we did not observe a significant drop in final fidelity, which contrasts with prior work on progressive DR [12], where the t-SNE optimization became trapped in local minima and ultimately yielded lower fidelity than its sequential counterpart. Instead, we found that the final fidelity of progressive DR is often higher than that of sequential DR. A possible explanation is the difference in the initialization during projection: whereas sequential DR starts from random coordinates, progressive DR places newly inserted points based on the projection of previously processed points. This more structured initialization provides a better starting point for optimization and helps the projection converge to a projection with high fidelity.

The ratio $\frac { t _ { \mathrm { p r o } } } { t _ { \mathrm { s e q } } }$ quantifies the computational overhead incurred by progressive execution. In our case, for the Update-Focused strategy, we found this ratio to be approximately 1.6.

Following this model, we summarize the performance characteristics

of GPUMAP below, which can serve as a baseline for future research:

$\Delta \hat { F } ( t _ { \mathrm { s e q } } ) ;$ : on average, the normalized fidelity gap between sequential and progressive DR at the moment sequential DR completes is approximately 0.3.

$\Delta \hat { F } ( t _ { \mathrm { p r o } } )$ : no significant fidelity loss is observed in the final projection due to progressive execution; in many cases, progressive DR achieved fidelity comparable to sequential DR.

• $\frac { t _ { \mathrm { p r o } } } { t _ { \mathrm { s e q } } }$ : the total execution time of progressive DR is approximately 1.6 times that of sequential DR on average.

One implication of this temporal model is the potential for automat ically adapting the time budget across different stages of progressive execution. For example, a scheduler could favor frequent updates during the early stage, when rapid feedback is most valuable, and reduce the update frequency during periods of gradual refinement. Such adaptation would require the system to infer the current stage from observable execution signals, since projection fidelity itself is not available during actual analysis. Developing such phase-aware scheduling strategies remains an interesting direction for future work.

## 5.3 Limitations

Our approach also has several limitations. First, we employ a simple linear regression model with a reserved margin to translate between time budgets and operation counts. While effective in practice, a more adaptive or learned cost model could better utilize the available budget. Second, we use a fixed scheduling strategy with empirically determined weights. Although this choice facilitates comparisons across settings, it limits the scheduler’s ability to adapt to dataset-specific characteristics or phase-dependent dynamics. Finally, our current implementation incurs additional overhead from generating intermediate projections between GPU operations, making it less efficient than a fully GPUresident batch implementation. Reducing this communication overhead remains an important direction for future work.

## 6 CONCLUSION

We present GPUMAP, a GPU-accelerated progressive implementation of UMAP that decomposes the original computation pipeline into three atomic operations, INSERT, UPDATE, and EMBED, and organizes them into a controllable progressive execution model. This design enables bounded-latency intermediate projections while scaling to million-point datasets on commodity GPU hardware. Building on this implementation, we systematically investigate the behavior of progressive DR under diverse conditions, including different scheduling strategies, time budgets, and biased insertion orders. Our results reveal trade-offs between runtime and projection fidelity, and establish an evaluation framework for progressive DR that we hope will inform the design of future progressive visualization systems.

## REFERENCES

[1] 10x Genomics. Single-cell RNA-seq of 1.3 million brain cells from E18 mice. NCBI Gene Expression Omnibus, accession GSE93421, 2017. 6

[2] J. Barnes and P. Hut. A hierarchical O(N log N) force-calculation algorithm. Nature, 324(6096):446–449, 1986. 2

[3] J. Blackard. Covertype. UCI Machine Learning Repository, 1998. doi: 10. 24432/C50K5N 6

[4] G. Cohen, S. Afshar, J. Tapson, and A. Van Schaik. EMNIST: Extending MNIST to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 2921–2926. IEEE, 2017. 6

[5] S. Dasgupta and Y. Freund. Random projection trees and low dimensional manifolds. In Proceedings of the Fortieth Annual ACM Symposium on Theory ofComputing, pp. 537–546, 2008. 2

[6] W. Dong, C. Moses, and K. Li. Efficient k-nearest neighbor graph construction for generic similarity measures. In Proceedings of the 20th International Conference on World Wide Web, pp. 577–586, 2011. 2, 3, 4

[7] M. Espadoto, N. S. T. Hirata, and A. C. Telea. Deep learning multidimensional projections. Information Visualization, 19(3):247–269, 2020. 3

[8] J.-D. Fekete. ProgressiVis: A toolkit for steerable progressive analytics and visualization. In 1st Workshop on Data Systemsfor Interactive Analysis, p. 5, 2015. 5

[9] J.-D. Fekete and R. Primet. Progressive analytics: A computation paradigm for exploratory data analysis. arXiv preprint arXiv:1607.05162, 2016. 1, 2

[10] W. Jakob, J. Rhinelander, and D. Moldovan. pybind11 – seamless operability between C++11 and Python, 2017. 5

[11] H. Jegou, M. Douze, and C. Schmid. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2010. 6

[12] J. Jo, J. Seo, and J.-D. Fekete. PANENE: A progressive algorithm for indexing and querying approximate k-nearest neighbors. IEEE Transactions on Visualization and Computer Graphics, 26(2):1347–1360, 2018. 1, 2, 5, 9

[13] P. Keung, Y. Lu, G. Szarvas, and N. A. Smith. The multilingual Amazon reviews corpus. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4563–4568, 2020. 6

[14] H.-K. Ko, J. Jo, and J. Seo. Progressive Uniform Manifold Approximation and Projection. In A. Kerren, C. Garth, and G. E. Marai, eds., EuroVis 2020 - Short Papers. The Eurographics Association, 2020. doi: 10.2312/ evs.20201061 1, 3, 5, 6, 8

[15] D. Kobak and G. C. Linderman. Initialization is critical for preserving global data structure in both t-SNE and UMAP. Nature Biotechnology, 39(2):156–157, 2021. 8

[16] G. C. Linderman, M. Rachh, J. G. Hoskins, S. Steinerberger, and Y. Kluger. Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nature Methods, 16(3):243–245, 2019. 2

[17] L. McInnes, J. Healy, N. Saul, and L. Grossberger. UMAP: Uniform manifold approximation and projection. Journal ofOpen Source Software, 3(29):861, 2018. 1, 2, 6

[18] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26, 2013. 6

[19] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervisedfeature learning, 2011. 6

[20] C. J. Nolet, V. Lafargue, E. Raff, T. Nanditale, T. Oates, J. Zedlewski et al. Bringing UMAP closer to the speed of light with GPU acceleration. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 418–426, 2021. 1, 2, 6

[21] N. Pezzotti, B. P. Lelieveldt, L. Van Der Maaten, T. Höllt, E. Eisemann, and A. Vilanova. Approximated and user steerable tSNE for progressive visual analytics. IEEE Transactions on Visualization and Computer Graphics, 23(7):1739–1752, 2016. 1, 2, 5

[22] S. Raschka, J. Patterson, and C. Nolet. Machine learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4):193, 2020. 2, 5

[23] D. Ren, F. Hohman, H. Lin, and D. Moritz. Embedding Atlas: Lowfriction, interactive embedding visualization. In 2025 IEEE Visualization and Visual Analytics (VIS), pp. 191–195. IEEE, 2025. 2

[24] T. Sainburg, L. McInnes, and T. Q. Gentner. Parametric UMAP embeddings for representation and semisupervised learning. Neural Computation,

33(11):2881–2907, 2021. 3

[25] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta et al. LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6

[26] C. Silpa-Anan and R. Hartley. Optimised KD-trees for fast image descriptor matching. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–8. IEEE, 2008. 2

[27] C. D. Stolper, A. Perer, and D. Gotz. Progressive visual analytics: Userdriven visual exploration of in-progress analytics. IEEE Transactions on Visualization and Computer Graphics, 20(12):1653–1662, 2014. 1, 2

[28] J. Tang, J. Liu, M. Zhang, and Q. Mei. Visualizing large-scale and highdimensional data. In Proceedings ofthe 25th International Conference on World Wide Web, pp. 287–297, 2016. 2

[29] C. Turkay, E. Kaya, S. Balcisoy, and H. Hauser. Designing progressive and interactive analytics processes for high-dimensional data analysis. IEEE Transactions on Visualization and Computer Graphics, 23(1):131–140, 2016. 2

[30] H. Van Assel, N. Courty, R. Flamary, A. Garivier, M. Massias, T. Vayer et al. TorchDR. 2

[31] L. Van Der Maaten. Learning a parametric embedding by preserving local structure. In Artificial Intelligence and Statistics, pp. 384–391. PMLR, 2009. 3

[32] L. Van Der Maaten. Accelerating t-SNE using tree-based algorithms. The Journal ofMachine Learning Research, 15(1):3221–3245, 2014. 2

[33] J. Venna and S. Kaski. Local multidimensional scaling. Neural Networks, 19(6-7):889–899, 2006. 6

[34] Z. J. Wang, F. Hohman, and D. H. Chau. WizMap: Scalable interactive visualization for exploring large machine learning embeddings. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 516–523, 2023. 2

[35] D. Whiteson. HIGGS. UCI Machine Learning Repository, 2014. doi: 10. 24432/C5V312 8