# Hierarchical Graph Representation Learning with Pooling-Induced Substructures

Luca Sbicego<sup>∗</sup> EPFL luca.sbicego@epfl.ch

Xiaowen Dong University of Oxford xdong@robots.ox.ac.uk

Dorina Thanou EPFL dorina.thanou@epfl.ch

## Abstract

Hierarchical graph-level tasks that require reasoning over substructures that extend beyond local neighborhoods are posing challenges for standard Graph Neural Networks (GNNs). While graph pooling aims to address this by coarsening graphs into hierarchical representations, existing methods often fail to explicitly preserve the substructures they induce, leading to inconsistent empirical performance. Moreover, evaluating such approaches remains challenging due to the lack of benchmarks with explicit hierarchical structure. We propose Pooled-Substructure-Aware GNNs, an approach to graph pooling that explicitly preserves and processes substructures uncovered by structure-based pooling schemes. We show that this design enhances the expressive power of pooling-based GNNs in the sense of the Weisfeiler-Lehman (WL) test. To enable principled evaluation, we introduce two new synthetic datasets with controllable hierarchical structure and propose a metric to characterize the prevalence of hierarchical patterns in realworld graph-level benchmarks. On tasks with relevant hierarchical structure, our approach outperforms matched pooling baselines, while remaining competitive on structurally simpler benchmarks. Code and datasets are publicly available at https://github.com/lsbicego/PSA-GNN/.

## 1 Introduction

Graph Neural Networks (GNNs) are widely used to process graph-structured data [1, 2], but their prevalent message passing formulation introduces a strong locality bias [3, 4]. Many graph tasks, however, require reasoning over structures extending beyond local neighborhoods, such as communities, or other forms of hierarchical organization [5–7] (Figure 1, left panel). While Message Passing Neural Networks (MPNNs) and Graph Transformers (GTs) can capture such patterns in principle, they must infer the underlying hierarchical organization implicitly, which can be challenging. Graph pooling has emerged as a standard mechanism for hierarchical graph representation learning [8], where graph convolutions are interleaved with pooling layers, which coarsen the input graph by grouping nodes into supernodes based on connectivity or learned node features [9].

In principle, this should enable multi-scale reasoning, making graph pooling a natural foundation for hierarchical modeling. However, in practice, pooling often yields limited or inconsistent gains over simpler flat MPNNs [10]. Widely used schemes that learn to coarsen a graph based on its node features in an end-to-end fashion have been shown to perform on par with randomized implementations that assign nodes to supernodes by chance [10]. Similarly, this type of pooling does not increase the expressivity of the underlying graph convolution in terms of the Weisfeiler-Lehman (WL) test, but at best preserves it [11, 12]. In contrast, structure-based pooling schemes that group nodes into hard clusters based on connectivity patterns inaccessible to standard MPNNs have been shown to increase expressivity [11]. This suggests that graph pooling can expose meaningful higher-order structure beyond simple coarsening of the input graph. Concretely, each supernode can be viewed as exposing a substructure, given by the induced subgraph over its constituent nodes and their internal edges. Yet, existing pooling-based approaches ignore this exposed structure: they collapse each cluster into a single supernode and discard the corresponding subgraph (Figure 1, top right panel).

![](images/aeda9bbc8d18fddce687b2f1eb67b92456a9172687ae61eb6aa5d5f0a19b20ef.jpg)  
Figure 1: Motivation and overview. Hierarchical tasks are challenging to solve for node-based MPNNs and GTs. We introduce PSA-GNN, a framework to process meaningful substructures exposed by graph pooling that is more expressive and improves performance on hierarchical graph-level tasks.

At the same time, evaluating graph pooling approaches remains a significant challenge. Hierarchical GNNs are typically assessed on standard benchmarks, where the extent of hierarchical patterns relevant to the graph-level label is unclear [13, 14]. Synthetic datasets with controllable hierarchical structure and difficulty are missing, making it difficult to isolate when and why pooling is beneficial.

These observations reveal both an architectural and an empirical gap in hierarchical graph learning: pooling-induced substructures are not explicitly preserved, and existing benchmarks make it difficult to assess when such structure matters. We address these gaps through the following contributions.

• Novel Framework for Hierarchical Modeling Using Graph Pooling (Section 3.1). We introduce Pooled-Substructure-Aware GNN (PSA-GNN; Figure 1, bottom right panel), a principled framework that processes pooling-induced clusters as explicit substructures rather than collapsing them into supernodes, enabling richer hierarchical representations.

• Theoretical Expressivity Analysis (Section 3.2). We provide a theoretical analysis showing that incorporating pooling-induced substructures increases expressivity in the WL sense compared to standard pooling.

• Hierarchical Benchmarks (Section 4.1). We introduce two synthetic datasets, MACRO-HIERARCHIES and GRANULAR-HIERARCHIES, enabling a controlled evaluation of hierarchical reasoning capabilities.

• Principled Evaluation. We further present a first characterization of the presence of hierarchical substructures in real-world benchmarks based on Nested Stochastic Block Models (Section 4.2). Together with our hierarchical benchmarks, this leads to a clear interpretability of empirical performances: on synthetic benchmarks and real-world datasets exhibiting hierarchical structure, our approach improves over existing pooling baselines, while remaining competitive on structurally simpler benchmarks (Section 5).

## 2 Background

Notation. Let an undirected, unweighted graph be denoted by $G = ( V , E )$ , where V and E are the node and edge sets, respectively, and $| V | = \bar { N }$ . We define the adjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { N \times N }$ and node features $\mathbf { X } \in \mathbb { R } ^ { \dot { N } \times F }$ , where F is the feature dimension. When comparing the expressivity of models, we use the following definition.

Definition 1 (Expressivity Comparisons, following [11]). Let $\theta , \phi$ be two permutation-invariant functions mapping graphs to vectors. Let G, H be any two graphs. Then:

• θ is at least as expressive as ϕ ${ \mathrm { i f ~ } } \phi ( G ) \neq \phi ( H ) \implies \theta ( G ) \neq \theta ( H )$

• θ and ϕ are equally expressive if $\phi ( G ) \neq \phi ( H ) \Longleftrightarrow \theta ( G ) \neq \theta ( H )$

• θ is more expressive than ϕ if θ is at least as expressive as $\phi$ and there exists a pair of graphs G, H such that $\theta ( G ) \neq \theta ( H )$ but $\phi ( G ) = \phi ( { \bar { H } } )$

Graph pooling. A graph pooling operator maps a graph G to a coarsened graph $G ^ { \prime }$ with $| V ^ { \prime } | < | V |$ We adopt the standard SEL-RED-CON view of pooling [9]. Nodes are first assigned to supernodes via SEL, node features are then aggregated within each supernode by RED, and connectivity between supernodes is determined by CON. Hierarchical GNNs interleave such pooling steps with message passing in order to process a graph at increasingly coarser levels.

Graph pooling in theory and practice. Graph pooling has often delivered inconsistent empirical gains, with hierarchical architectures frequently failing to outperform simpler flat baselines [10]. Many popular pooling operators are node-based, with assignments or scores learned from the same local node representations produced by the underlying MPNN. Recent expressivity results show that such pooling schemes cannot exceed the expressive power of the base encoder when their assignments depend only on these node representations [11, 12]. This may help explain why learned pooling can fail to introduce new structural information. These observations motivate us to investigate pooling from both theoretical and benchmarking perspectives.

Structure-based pooling. By contrast, recent work has shown that pooling schemes determining cluster assignments based on graph structure rather than learned node features can increase expressiv ity [11]. In this setting, pooling-induced clusters can be viewed not only as supernodes for coarsening, but also as structurally meaningful subgraphs of the original graph. This perspective forms the basis of our approach.

Subgraph GNNs. Subgraph GNN methods [15–18] aim to improve graph representations by processing a graph through a collection of overlapping subgraphs rather than only through the full graph itself. Intuitively, this allows nodes to be represented in multiple local contexts, strengthening expressivity beyond standard message passing [15, 16]. This raises the question of how to exploit pooling-induced subgraphs.

## 3 From Graph Pooling to Substructure-Aware Processing

We introduce a principled framework for processing meaningful pooling-induced substructures, instantiated by Pooled-Substructure-Aware GNNs (PSA-GNN). The key idea is to treat each pooled cluster as an induced subgraph rather than immediately collapsing it into a supernode. This contrasts with standard graph pooling, which reduces each cluster to a supernode and continues computation on the coarsened graph, thereby discarding the internal structure induced by the clusters.

## 3.1 Processing Pooling-Induced Substructures

Our perspective naturally connects graph pooling to the Subgraph GNN setting, where models operate not only on the input graph itself, but also on a collection of subgraphs extracted from it [15–18]. In our setting, the subgraph collection is not generated by a separate subgraph selection rule, but arises directly from the substructures exposed by the pooling operator:

Definition 2 (Pooling-induced subgraph policies). Let a graph pooling scheme partition the nodes of $G = ( V , E )$ ) into disjoint clusters $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { m } . \mathrm { ~ A ~ }$ subgraph policy π induces a set of subgraphs $\boldsymbol { S } = \{ G _ { i } = ( ^ { \prime } E _ { i } ) \} _ { i = 1 } ^ { \bar { m } }$ with $V _ { i } = A _ { i }$ and $E _ { i } = \{ ( u , v \bar { ) } \in \bar { E : } u , \bar { v } \in V _ { i } \}$ . Note that the induced subgraphs are vertex-disjoint. We further consider a padded policy πˆ that augments each subgraph with crossing edges and their incident boundary nodes, yielding an edge-covering set of subgraphs.

Building on these pooling-induced subgraph policies, we introduce PSA-GNN, which processes the pooling-induced subgraphs created by a policy π using a subgraph-processing layer inspired by DSS-GNN [15]. The layer combines subgraph-level updates with a graph-level context update, as illustrated in Figure 2. For each subgraph, a shared encoder $L ^ { \mathrm { s g } }$ updates node representations using only the internal structure of that subgraph. In parallel, subgraph-node features are scattered back to the original node space, processed on the original graph by $\hat { L ^ { \mathrm { g r a p h } } }$ , and then mapped back to each subgraph. This graph-level branch allows information to flow across different pooled subgraphs through the original graph adjacency.

![](images/31b8a860b27b2ba8156c4b81fb22ad569b3ef50628504acebdb3c4741a048c3d.jpg)  
Figure 2: Computational flow of PSA-GNN. We pre-compute the subgraph policy and initialize node features in subgraph space. Each layer then updates these representations by combining message passing on the original graph with message passing inside each induced subgraph. Finally, node representations are aggregated within each subgraph and passed to a DeepSets-style readout.

Formally, for each subgraph $i \in \{ 1 , \ldots , m \}$ , layer $L$ updates the node feature matrix as

$$
{ \bf X } _ { i } ^ { \prime } = L ( { \bf A } , { \bf X } , \pi ) _ { i } = L ^ { \mathrm { s g } } ( { \bf A } _ { i } , { \bf X } _ { i } ) + \left( L ^ { \mathrm { g r a p h } } ( { \bf A } , \bar { \bf X } ) \right) _ { \pi _ { i } } ,\tag{1}
$$

where $\mathbf { A } _ { i }$ and $\mathbf { X } _ { i }$ denote the adjacency and node feature matrices of subgraph $i ,$ and A is the adjacency matrix of the original graph. The matrix X¯ is obtained by scatter-aggregating all subgraph-node features back to the original node space. For vertex-disjoint subgraphs, this is simply a re-indexing step; for overlapping subgraphs, duplicate occurrences are averaged. The operator $( \cdot ) _ { \pi _ { i } }$ denotes gathering original-node features to the nodes in subgraph i.

For graph-level prediction, we first pool the node features of each subgraph into a subgraph representation $\mathbf { s } _ { i }$ , and then apply a DeepSets-style readout [19]:

$$
\hat { y } = \mathrm { M L P } _ { \mathrm { h e a d } } \left( \sum _ { i = 1 } ^ { m } \mathrm { M L P } _ { \mathrm { s g } } ( \mathbf { s } _ { i } ) \right) .\tag{2}
$$

Alternatively, the element-wise transformation ML $\mathrm { \Delta _ { \it P _ { \mathrm { s g } } } }$ can be replaced by a GNN operating over the pooled graph: $\mathrm { G N N } _ { \mathrm { s g } } ( \mathbf { A } ^ { \prime } , \mathbf { Z } )$ , where $\mathbf { Z } _ { i } = \mathbf { s } _ { i }$ and $\mathbf { A } ^ { \prime }$ is the adjacency matrix of the coarsened graph.

Unless otherwise specified, PSA-GNN uses a DeepSets-style readout and the padded variant from Definition 2, which augments each subgraph with crossing edges and their associated boundary nodes. This introduces controlled overlap between neighboring pooled substructures, allowing local subgraph updates to access boundary context while the graph-level branch couples information through the original graph adjacency. To distinguish original cluster nodes from added boundary nodes, we append a binary node feature indicating whether a node belongs to the core cluster or was introduced through padding.

## 3.2 Expressivity of Explicit Substructure Processing

In this section, we establish an expressivity hierarchy that characterizes the benefits of explicitly processing pooling-induced substructures. In particular, we investigate how within-subgraph message passing and information exchange across substructures contribute to the expressive power of pooling based GNNs.

To this end, we consider two comparison architectures. First, we introduce IP-GNN (Intra-Pooling-GNN), a simpler variant that uses the base policy π to produce disjoint clusters and extends the standard SEL-RED-CON pooling pipeline by performing message passing within each poolinginduced subgraph before the RED and CON steps. Second, we consider a standard SEL-RED-CON Pool-GNN as a matched pooling baseline.

We first analyze the effect of within-subgraph message passing. Proposition 3 formalizes the basic intuition that immediate coarsening can discard information already present in the substructures induced by pooling. Even a single message-passing step restricted to these subgraphs can therefore strengthen pooling.

Proposition 3. Let ${ \mathcal { C } } = { \mathcal { A } } _ { 1 } , \ldots , { \mathcal { A } } _ { m }$ be a fixed partition of the node set into disjoint clusters. Then, IP-GNN is at least as expressive as Pool-GNN in the WL sense. Moreover, there exists a partition $\widetilde { \mathcal { C } }$ for which IP-GNN is strictly more expressive than Pool-GNN.

We next show that PSA-GNN subsumes the simpler within-subgraph construction and can go strictly beyond it once pooled subgraphs are allowed to overlap through padding. Intuitively, this overlap enables the model to couple information across neighboring pooled substructures while preserving the underlying cluster structure identified by pooling.

Proposition 4. For any subgraph policy π, PSA-GNN with supergraph readout is at least as expressive as IP-GNN under the same policy. Moreover, there exists a policy π such that PSA-GNN using the corresponding padded, edge-covering policy πˆ together with appropriate node markings and any injective readout is strictly more expressive than IP-GNN under π.

Propositions 3 and 4 in combination yield the hierarchy between PSA-GNN, IP-GNN, and Pool-GNN. Full formal statements, proofs, and additional examples are deferred to Appendix A. Finally, we verify this relationship empirically on BREC, a standard expressivity benchmark in Appendix B.

## 4 Characterizing Hierarchical Graph Structure

The results of Section 3 suggest that explicit substructure processing can increase expressivity by leveraging graph pooling more effectively. We expect these gains to translate into improved downstream performance particularly when graph-level tasks depend on hierarchically organized substructures in the input graph rather than local node-level patterns. Figure 1 (left panel) illustrates this in an example: two graphs may exhibit identical local neighborhoods around selected nodes while differing in their global community organization. In such situations, local node-level views computed by bounded-depth MPNNs may be uninformative. Instead, GTs can enrich node-level representations with positional or structural encodings, but these mechanisms are not designed to explicitly identify task-relevant structures. Thus, while both model families can potentially capture higher-order patterns, they must learn to identify the relevant structures implicitly, whereas substructure-aware models can process them directly.

However, graph pooling methods are typically evaluated on standard graph-level benchmarks, where the relevance of hierarchical structure to the prediction task is often unclear [10, 20, 13]. To address this limitation, we adopt two complementary evaluation strategies: controlled synthetic benchmarks with explicit hierarchical signals, and a structural characterization of real-world datasets.

## 4.1 Synthetic Hierarchical Benchmarks

To study hierarchical graph modeling in a controlled setting, we introduce two synthetic benchmarks in which graph labels explicitly depend on hierarchical structure. The datasets are generated using SBMs [21]; full construction details are deferred to Appendix D.1. By design, the resulting prediction tasks do not impose an expressivity barrier and remain solvable, in principle, by sufficiently deep MPNNs.

MACRO-HIERARCHIES tests whether a model can reason about higher-level connectivity between clusters, inspired by the example in Figure 1. Each graph is generated from a two-level SBM, with micro-communities grouped into macro-communities. The class signal is encoded at the macro level: macro-communities are arranged either in a ring or in a bipartite pattern, as illustrated in Figure 3. Distinguishing the two classes therefore requires distinguishing different interaction patterns between larger substructures. We control task difficulty by adding random inter-cluster edges that obscure the underlying ring or bipartite pattern.

GRANULAR-HIERARCHIES complements the above dataset and tests whether a model can reason about the size distribution within clusters themselves. Graphs with class y = 0 consist of equally sized clusters, whereas graphs with class y = 1 contain a mixture of small and large clusters, as illustrated in Figure 3. The signal therefore lies in the relative granularity of the substructures. As in MACRO-HIERARCHIES, we can control the difficulty of this task by adding more inter-cluster edges and thus making the substructure signal noisier.

## 4.2 Structural Characterization of Real-World Benchmarks

Complementing our controlled synthetic benchmarks, we characterize the extent of hierarchical structure in real-world graph-level datasets. This provides a structural perspective for interpreting when explicit substructure processing may be beneficial in practice.

For each graph, we fit a degree-corrected nested SBM to its undirected, unweighted topology [21– 23], from which we extract two descriptors: the detected substructure count, i.e., the number of blocks at the finest inferred level, and the substructure depth, i.e., the number of inferred hierarchy levels. Together, these descriptors quantify how much partitionable substructure and hierarchical organization the graph exhibits under the fitted model.

![](images/0733799701356a307542aaec53e949f5a389685fe540b703b44005dc755ef7ce.jpg)  
Figure 3: Examples from two synthetic datasets. Left: MACRO-HIERARCHIES, where the class distinction is given by ring versus bipartite higher-level structure. Right: GRANULAR-HIERARCHIES, where the class distinction is given by even versus uneven cluster sizes.

Figure 4 shows the distribution of these descriptors across graphs from several widely used graphlevel benchmarks. Under this structural probe, ZINC [24] and NCI1 [25] appear structurally shallow, as almost all graphs are assigned a single detected substructure at depth 1. In contrast, REDDIT-5K, REDDIT-12K, COLLAB [25], and MalNet-Tiny [26] exhibit substantially richer inferred group structure and, for many graphs, multiple hierarchy levels. We use these findings to distinguish structurally flatter datasets from more structure-dominated benchmarks in the experiments below. To validate the methodology and to provide a point of reference, we also include an instance of our synthetic datasets at a medium difficulty level. As expected, the fitted models recover clear hierarchical structure in these graphs.

![](images/7ddc0d17bf1bb2a5cc69e68dd50d6f3c06d766910458d3beb70adb00169f8851.jpg)  
Figure 4: Structural characterization of real-world benchmarks using nested SBM fits. For each graph, we fit a degree-corrected nested SBM and record the number of detected substructures at the finest inferred level (left axis) and the detected substructure depth (right axis).

## 5 Experiments

We evaluate whether the expressivity gains from explicit substructure processing translate into improved graph-level performance when hierarchical structure is relevant to the prediction task. In all experiments, we compare PSA-GNN to the simple within-subgraph baseline (IP-GNN) as well as the standard pooling baseline (Pool-GNN) which immediately coarsens the graph, using matched pooling policies.

Setup. Our comparisons are centered on three structure-based pooling schemes: Leiden [27], a community-detection method; CurvPool [28], which clusters nodes using balanced Forman curvature; and CliquePool [29], which forms clusters from maximal cliques. For each policy, we evaluate the three pooling-based architectures for a single pooling step under matched depth and parameter budgets. Throughout all experiments, we use GIN [4] as the 1-WL base message passing layer. Hyperparameters are chosen on validation data, and the same dataset-specific training protocol is used across models. Full details and implementation choices are provided in Appendix E. Additional results for exploratory pooling-based subgraph policies are provided in Appendix F.

## 5.1 Controlled Hierarchical Benchmarks

The synthetic benchmarks introduced in Section 4.1 provide a controlled setting for testing our hypothesis, as graph labels explicitly depend on hierarchical graph structure. In addition to the matched pooling variants Pool-GNN, IP-GNN, and PSA-GNN, we compare against a flat MPNN based on GIN [4] and a GT baseline based on GPS [30]. Figure 5 reports test accuracy across Leiden, CurvPool, and CliquePool across varying difficulty levels.

![](images/74ea779499a164580f4c1c0a705aac36fe21dec10c7394cc9b25d640b671bb0b.jpg)  
Figure 5: Results on the synthetic hierarchical benchmarks. We compare Pool-GNN, IP-GNN, and PSA-GNN against flat MPNN and GT baselines across representative structure-based schemes.

Across both synthetic benchmarks, PSA-GNN achieves the best performance across the evaluated difficulty levels and remains substantially more robust as the hierarchical signal becomes harder to recover. The flat MPNN and GPS baselines often degrade more rapidly as difficulty increases, supporting the motivation that node-level models can struggle to infer the relevant higher-order organization implicitly.

The matched pooling variants further show the value of preserving the structures identified by pooling for tasks that depend on higher-order graph organization: IP-GNN often improves over immediate coarsening, while PSA-GNN achieves the strongest performance by processing pooling-induced substructures end-to-end. This pattern is consistent with the expressivity hierarchy established in Section 3.2 and provides controlled evidence that explicit substructure processing is useful when the task depends on hierarchical graph organization. The magnitude of these gains varies across pooling policies, highlighting that the benefits also depend on how well the resulting substructures align with the task-relevant hierarchy.

We additionally evaluate common learned pooling operators on the synthetic benchmarks in Appendix C.1. Sparse methods such as SAGPool and ASAPool [31, 32] can perform strongly on MACRO-HIERARCHIES, while dense methods like MinCutPool or DiffPool [33, 34] do not improve over baselines. These results highlight the diagnostic value of the benchmarks and suggest that different pooling mechanisms capture different hierarchical signals. We further assess the impact of informative node features by repeating the medium-difficulty experiments with contextual SBM features [35] in Appendix C.2. Informative features narrow the performance gap when directly relevant to the task, while PSA-GNN retains its advantage on higher-level structural reasoning.

## 5.2 Real-World Benchmarks

Table 1 presents results on the real-world benchmarks characterized in Section 4.2. We again focus on the matched comparison between PSA-GNN, IP-GNN, and Pool-GNN under the same three structure-based pooling policies. We additionally include popular hierarchical GNN baselines, namely ASAPool, SAGPool, MinCutPool, and DiffPool, as well as a flat MPNN baseline. Where computationally feasible, we also evaluate a Subgraph GNN baseline that applies PSA-GNN to all 2-hop egonets. This provides a reference for subgraph processing with a generic, non-pooling-based subgraph policy, while the matched pooling variants remain the focus of our architectural comparison.

Table 1: Results on real-world benchmarks. Best overall result per dataset is highlighted in brown, second best in purple. Among PSA-GNN, IP-GNN, and Pool-GNN, the best method for a given subgraph policy π is shown bolded. A ⋆ indicates that PSA-GNN significantly outperforms the matched Pool-GNN under a two-sided Welch test $( p < 0 . 0 5$ , unadjusted). The Avg. Rank reports the average rank of each architecture level across matched dataset–policy comparisons among PSA-GNN, IP-GNN, and Pool-GNN. Dense pooling schemes do not scale and ASAPool cannot use edge features by default.
<table><tr><td></td><td></td><td></td><td colspan="4">Hierarchical Structure</td><td colspan="2">Flat Structure</td></tr><tr><td>Model</td><td>Subgraph Policy</td><td>Avg. Rank (↓)</td><td>COLLAB  $A c c . \ [ \% ] \left( \uparrow \right)$ </td><td> $A c c . \ [ \% ] \left( \uparrow \right)$ </td><td>REDDIT-5K REDDIT-12K  $A c c . \ [ \% ] \left( \uparrow \right)$ </td><td>MalNet  $A c c . \ [ \% ] \ ( \uparrow )$ </td><td>NCI1  $A c c . \ [ \% ] \ ( \uparrow )$ </td><td>ZINC MAE (↓)</td></tr><tr><td>PSA-GNN</td><td>CliquePool</td><td></td><td> $7 8 . 1 \pm 1 . 3 \star$ </td><td> ${ \bf 5 5 . 0 \pm 0 . 8 }$ </td><td> $4 7 . 8 \pm 1 . 4$ </td><td> $9 0 . 6 \pm 2 . 4$ </td><td> ${ \bf 8 1 . 5 \pm 2 . 5 }$ </td><td> $\mathbf { 0 . 1 8 2 \pm 0 . 0 0 8 }$ </td></tr><tr><td></td><td>CurvPool</td><td>1.36</td><td> $7 6 . 8 \pm 1 . 5 \star$ </td><td> ${ \pm } 4 . 9 \pm 1 . 7$ </td><td> $5 0 . 1 \pm 1 . 7 \star$ </td><td> $9 2 . 0 \pm 1 . 9$ </td><td> $8 1 . 2 \pm { 1 . 6 }$ </td><td> $\mathbf { 0 . 1 7 7 \pm 0 . 0 0 5 }$ </td></tr><tr><td></td><td>Leiden</td><td></td><td> $7 6 . 4 \pm 2 . 2$ </td><td> $5 5 . 3 \pm 2 . 1 \star$ </td><td> ${ \bf 5 0 . 0 \pm 1 . 3 \star }$ </td><td> ${ \bf 9 1 . 1 \pm 1 . 4 \star }$ </td><td> $8 0 . 0 \pm 2 . 5$ </td><td> $0 . 2 1 3 \pm 0 . 0 0 7 \star$ </td></tr><tr><td>IP-GNN</td><td>CliquePool</td><td></td><td> $7 5 . 0 \pm 2 . 3$ </td><td> $5 3 . 3 \pm { 3 . 4 }$ </td><td> $4 5 . 1 \pm 2 . 1$ </td><td> $9 1 . 7 \pm 1 . 7$ </td><td> $8 0 . 6 \pm 2 . 1$ </td><td> $0 . 1 8 3 \pm 0 . 0 0 4$ </td></tr><tr><td></td><td>CurvPool</td><td>2.33</td><td> $7 7 . 4 \pm 0 . 9$ </td><td> $5 2 . 6 \pm 3 . 3$ </td><td> $4 2 . 0 \pm 2 . 5$ </td><td> $8 9 . 3 \pm 2 . 0$ </td><td> $8 0 . 3 \pm 1 . 9$ </td><td> $0 . 1 8 4 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td>Leiden</td><td></td><td> $7 3 . 6 \pm 3 . 1$ </td><td> $5 2 . 2 \pm 2 . 6$ </td><td> $4 5 . 6 \pm 0 . 8$ </td><td> $9 0 . 7 \pm 1 . 1$ </td><td> ${ \bf 8 0 . 1 } \pm 2 . 6$ </td><td> $\mathbf { 0 . 1 8 6 \pm 0 . 0 0 4 }$ </td></tr><tr><td>Pool-GNN</td><td>CliquePool</td><td></td><td> $7 0 . 4 \pm 4 . 1$ </td><td> $5 2 . 4 \pm 2 . 9$ </td><td> ${ \bf 4 8 . 3 \pm 1 . 7 }$ </td><td> $9 0 . 2 \pm 1 . 6$ </td><td> $8 1 . 0 \pm 2 . 4$ </td><td> $\mathbf { 0 . 1 8 2 \pm 0 . 0 0 4 }$ </td></tr><tr><td></td><td>CurvPool</td><td>2.31</td><td>71.1 ± 4.5</td><td> $5 3 . 6 \pm 2 . 6$ </td><td> $4 7 . 4 \pm 0 . 4$ </td><td> $9 1 . 3 \pm 1 . 1$ </td><td> ${ \bf 8 1 . 3 \pm 2 . 0 }$ </td><td> $0 . 1 7 8 \pm 0 . 0 0 4$ </td></tr><tr><td></td><td>Leiden</td><td></td><td> $7 4 . 1 \pm 2 . 3$ </td><td> $5 3 . 1 \pm 1 . 9$ </td><td> $4 4 . 7 \pm 1 . 6$ </td><td> $8 7 . 9 \pm 1 . 5$ </td><td> $7 9 . 9 \pm 2 . 0$ </td><td> $0 . 2 5 7 \pm 0 . 0 1 3$ </td></tr><tr><td>Hierarchical GNN</td><td>ASAPool</td><td></td><td> $7 1 . 6 \pm 5 . 4$ </td><td></td><td> $4 9 . 5 \pm 0 . 2$ </td><td> $8 8 . 2 \pm 3 . 1 $ </td><td> $8 1 . 2 \pm 2 . 0$ </td><td></td></tr><tr><td></td><td>SAGPool</td><td></td><td> $7 0 . 6 \pm 4 . 2$ </td><td></td><td> $4 6 . 6 \pm 1 . 8$ </td><td> $9 0 . 1 \pm 1 . 1$ </td><td> $8 0 . 1 \pm 1 . 9$ </td><td> $0 . 4 1 6 \pm 0 . 1 0 7$ </td></tr><tr><td></td><td>MinCutPool</td><td></td><td></td><td></td><td></td><td></td><td> $8 0 . 2 \pm 2 . 0$ </td><td> $0 . 1 6 1 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td>DiffPool</td><td></td><td></td><td></td><td></td><td></td><td> $8 1 . 0 \pm 1 . 7$ </td><td> $0 . 1 4 7 \pm 0 . 0 0 5$ </td></tr><tr><td>Baselines</td><td>Egonet</td><td></td><td></td><td></td><td></td><td> $9 1 . 4 \pm 1 . 4$ </td><td> $8 1 . 6 \pm 2 . 4$ </td><td> $0 . 4 0 3 \pm 0 . 0 6 0 ^ { \dag }$ </td></tr><tr><td></td><td> $\mathrm { \bar { M P N N } ( G I N ) }$ </td><td></td><td> $7 4 . 9 \pm 3 . 7$ </td><td> $5 3 . 4 \pm 2 . 4$ </td><td> $4 8 . 3 \pm 2 . 3$ </td><td> $9 1 . 0 \pm 1 . 6$ </td><td> $8 0 . 9 \pm 2 . 5$ </td><td> $0 . 1 7 2 \pm 0 . 0 0 5$ </td></tr></table>

<sup>†</sup> The Egonet result on ZINC is likely under-converged under the shared training budget.  
Across these real-world benchmarks, explicit substructure processing strengthens structure-based pooling. Aggregated over all dataset–policy comparisons, PSA-GNN achieves the best mean rank, ahead of IP-GNN and Pool-GNN (1.36 vs. 2.33 and 2.31 respectively), suggesting that its benefits extend across different pooling policies. These results position PSA-GNN as an effective architecture for real-world hierarchical graph modeling.

The choice of subgraph policy nevertheless remains task-dependent, as different schemes expose dif ferent structural patterns. Since no policy consistently dominates across datasets, domain knowledge or validation performance can guide its selection.

## 5.3 Validation of Structural Hierarchy Characterization

Figure 6 relates the performance gains of PSA-GNN to the structural characterization introduced in Section 4.2. For each dataset, we report the relative gain of PSA-GNN over the flat MPNN baseline using the best-performing structure-based pooling policy. In line with our hypothesis, the group-level average gain is positive on benchmarks with richer inferred hierarchical structure, but slightly negative on structurally flatter datasets. On COLLAB, REDDIT-5K, REDDIT-12K, and MalNet-Tiny, which exhibit richer inferred substructure, PSA-GNN achieves the strongest overall performance, suggesting that explicit substructure processing is particularly beneficial in these settings. On NCI1 and ZINC, PSA-GNN remains competitive, especially within the family of structure-based pooling models, although the overall performance differences are smaller. In these structurally flatter settings, MPNN baselines and pooling schemes with dense soft assignments, such as DiffPool and MinCutPool, can also perform strongly. Overall, these results are consistent with the topology-driven hierarchical structure detected by our SBM probe and support the use of our metric for interpreting when pooling is likely to help.

![](images/77cac9d097686f15ce43e4439992977a3da50c6481c4c3a106707c5ce7df6d2e.jpg)  
Figure 6: Relative gain over MPNN. Bars show group means; points show individual datasets. Positive values indicate improvement over MPNN.

Table 2: Test MAE on GraphBench CO benchmarks. We compare CO-ER (flat graphs) and CO-RB (with hierarchical structure) on 10K-graph subsets.
<table><tr><td>Model</td><td>Subgraph Policy</td><td>Flat CO-ER MAE (↓)</td><td>Hierarchical CO-RB MAE (↓)</td></tr><tr><td>PSA-GNN</td><td>CliquePool</td><td> $\overline { { 0 . 5 7 2 2 \pm 0 . 0 0 8 5 } }$ </td><td> $1 . 0 1 0 8 \pm 0 . 0 1 4 5$ </td></tr><tr><td></td><td>CurvPool</td><td> $0 . 5 9 1 7 \pm 0 . 0 1 7 7$ </td><td> $0 . 9 3 4 5 \pm 0 . 0 6 4 9$ </td></tr><tr><td></td><td>Leiden</td><td> $0 . 6 0 1 5 \pm 0 . 0 1 4 5$ </td><td> $1 . 2 5 5 1 \pm 0 . 0 5 9 8$ </td></tr><tr><td>Baseline</td><td>MPNN (GIN)</td><td> $0 . 5 8 4 4 \pm 0 . 0 3 4 7$ </td><td> $1 . 4 2 5 1 \pm 0 . 0 1 8 0$ </td></tr><tr><td>Avg. Gain vs. MPNN</td><td></td><td>-0.70%</td><td>+25.14%</td></tr></table>

We further validate our approach on a combinatorial optimization (CO) task of predicting the objective value of a maximum independent set task [36]. CO-ER [37] consists of flat ER graphs (one substructure across one level), whereas CO-RB [38] contains hierarchical graphs according to our structural descriptors, with 22.7 substructures across 4 levels on average. This contrast clearly illustrates that our structural characterization is indicative of the benefits of explicit substructure processing, with PSA-GNN achieving larger performance gains on datasets exhibiting richer inferred hierarchical structure.

## 5.4 Scalability Analysis

Next, we analyze the scalability of PSA-GNN across different subgraph policies. A standard MPNN has time and memory complexity of $\mathcal { O } ( N + M )$ , where N and M denote the number of nodes and edges, respectively. Excluding subgraph construction, PSA-GNN additionally processes the pooling-induced substructures and thus incurs a policy-dependent overhead factor $c _ { \pi } { : }$

$$
{ \mathcal { O } } { \Big ( } ( 1 + c _ { \pi } ) ( | V | + | E | ) { \Big ) } , \quad { \mathrm { w i t h ~ } } c _ { \pi } = { \frac { \sum _ { i } ( | V _ { i } | + | E _ { i } | ) } { N + M } } .\tag{3}
$$

The value of $c _ { \pi }$ depends on the size and overlap of the selected substructures. For example, for a disjoint partition π (see Def. 2), we have $\begin{array} { r } { \sum _ { i } | \dot { V _ { i } } | = N } \end{array}$ and $\textstyle \sum _ { i } | E _ { i } | \leq M$ , so the subgraph branch adds at most one additional graph-sized computation.

Figure 7 displays the total number of subgraph-node occurrences $\textstyle \sum _ { i } | V _ { i } |$ induced by different padded policies πˆ as a function of the original graph size. This quantity drives the factor $c _ { \pi }$ and serves as a simple proxy for memory usage during training and inference.

![](images/b590fb73cf3cd024f12919362e2716b4fbe00e804e440e3c5371a1d5c6ef5aa2.jpg)  
(a) Scalability on NCI1

![](images/176f956c3bf8f227b9970b7299856e21e2d874ad22543ff4ff8f16725a3fdd35.jpg)  
(b) Scalability on REDDIT  
Figure 7: Scalability of subgraph policies. We compare the total number of subgraph-node occurrences $\textstyle \sum _ { i } | V _ { i } |$ induced by different policies. Observations are grouped into 20 bins and averaged for readability; very large REDDIT graphs are omitted from the plot.

Pooling-induced policies generate substantially fewer subgraph-node occurrences than generic egonet policies, often by orders of magnitude on larger graphs. Processing these substructures therefore remains scalable, making pooled-substructure processing a promising approach for hierarchical modeling on larger graphs. Per-epoch training times are reported in Appendix E.3.

## 6 Related Work

Graph pooling. Many widely used pooling architectures [8, 9] rely on learned dense assignments or sparse node-selection mechanisms, making learned pooling a central paradigm in hierarchical GNNs. Motivated by recent results showing that structure-based pooling can increase expressive power [11], we revisit simpler pooling policies derived from graph topology. While some pooling methods use local subgraph operations for scoring or compression [32, 39], we propose a different view of structure-based pooling by leveraging the exposed substructures as central objects for hierarchical modeling.

Higher-order Graph Modeling. Our work also connects graph pooling to higher-order graph learning approaches that operate on structures beyond individual nodes. In particular, our architecture is inspired by Subgraph GNNs [15–18], which improve graph representations by processing collections of subgraphs. Rather than relying on generic subgraph policies such as egonets or node-deleted subgraphs, we leverage the clusters exposed by graph pooling, directly linking substructure processing to hierarchical graph modeling. This also yields collections of substructures that remain scalable to larger graphs, as is often required in hierarchical modeling. The same distinction applies to graph patching methods such as Graph MLP-Mixer [40], which tokenize input graphs into node groups. Finally, our work is conceptually related to hypergraph learning [41–43], where hyperedges can be viewed as meaningful overlapping node groups, providing a useful perspective for future extensions of our framework.

## 7 Conclusion

We introduced a pooled-substructure view of graph pooling by treating induced clusters as valuable substructures for hierarchical tasks. This approach improves hierarchical graph modeling in practice and strengthens pooling from an expressivity perspective. To enable principled evaluation, we proposed two new synthetic datasets with tasks requiring hierarchical reasoning and characterized hierarchical structure in real-world benchmarks. On tasks with meaningful hierarchical structure, PSA-GNN consistently outperforms immediate coarsening, showing that the substructures exposed by pooling are useful to preserve and process rather than intermediate artifacts to discard.

At the same time, our experiments clarify the scope of our method. Pooling is beneficial only when graphs exhibit hierarchical structure, and the pooling scheme must expose this structure appropriately, which may vary across datasets. Moreover, our structural characterization should be viewed as a first step toward quantifying hierarchy, as it does not yet assess whether the detected structure is directly predictive of the graph label.

Overall, our results position structure-based pooling combined with explicit substructure processing as a promising direction for hierarchical graph representation learning. Rather than serving solely as a coarsening mechanism, graph pooling can be viewed as a way to expose relevant substructures for computation. This opens a broader design space for hierarchical GNNs and reveals useful connections to other higher-order graph learning frameworks, which can inform future hierarchical modeling architectures.

## Acknowledgments and Disclosure of Funding

This work received no specific external funding and the authors declare no competing interests.

## References

[1] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.

[2] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1):4–24, 2021.

[3] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning, pages 1263–1272, 2017.

[4] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.

[5] Michelle Girvan and Mark EJ Newman. Community structure in social and biological networks. Proceedings ofthe National Academy ofSciences, 99(12):7821–7826, 2002.

[6] Erzsébet Ravasz and Albert-László Barabási. Hierarchical organization in complex networks. Physical Review E, 67(2):026112, 2003.

[7] Aaron Clauset, Cristopher Moore, and Mark EJ Newman. Hierarchical structure and the prediction of missing links in networks. Nature, 453(7191):98–101, 2008.

[8] Chuang Liu, Yibing Zhan, Jia Wu, Chang Li, Bo Du, Wenbin Hu, Tongliang Liu, and Dacheng Tao. Graph pooling for graph neural networks: Progress, challenges, and opportunities. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 6712–6722, 2023.

[9] Daniele Grattarola, Daniele Zambon, Filippo Maria Bianchi, and Cesare Alippi. Understanding Pooling in Graph Neural Networks. IEEE Transactions on Neural Networks and Learning Systems, 2022.

[10] Diego Mesquita, Amauri Souza, and Samuel Kaski. Rethinking pooling in graph neural networks. In Advances in Neural Information Processing Systems, volume 33, pages 2220– 2231, 2020.

[11] Veronica Lachi, Alice Moallemy-Oureh, Andreas Roth, and Pascal Welke. Expressive pooling for graph neural networks. Transactions on Machine Learning Research, 2025.

[12] Filippo Maria Bianchi and Veronica Lachi. The expressive power of pooling in graph neural networks. In Advances in Neural Information Processing Systems, 2023.

[13] Pengyun Wang, Junyu Luo, Yanxin Shen, Ming Zhang, Shaoen Qin, Siyu Heng, and Xiao Luo. A comprehensive graph pooling benchmark: Effectiveness, robustness and generalizability. arXiv preprint arXiv:2406.09031, 2024.

[14] Federico Errica, Marco Podda, Davide Bacciu, and Alessio Micheli. A fair comparison of graph neural networks for graph classification. In International Conference on Learning Representations, 2020.

[15] Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath Balamurugan, Michael M. Bronstein, and Haggai Maron. Equivariant subgraph aggregation networks. In International Conference on Learning Representations, 2022.

[16] Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any GNN with local structure awareness. In International Conference on Learning Representations, 2022.

[17] Muhan Zhang and Pan Li. Nested graph neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 15734–15747, 2021.

[18] Fabrizio Frasca, Beatrice Bevilacqua, Michael Bronstein, and Haggai Maron. Understanding and extending subgraph GNNs by rethinking their symmetries. In Advances in Neural Information Processing Systems, volume 35, pages 31376–31390, 2022.

[19] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. In Advances in Neural Information Processing Systems, volume 30, 2017.

[20] Luca Sbicego, Sevda Ögüt, Manuel Madeira, Yiming Qin, Dorina Thanou, and Pascal Frossard.˘ On the role of structure in hierarchical graph neural networks. In Proceedings on “I Can’t Believe It’s Not Better: Challenges in Applied Deep Learning” at ICLR 2025 Workshops, volume 296 of Proceedings ofMachine Learning Research, pages 141–150. PMLR, 2025.

[21] Paul W Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. Stochastic blockmodels: First steps. Social Networks, 5(2):109–137, 1983.

[22] Brian Karrer and Mark EJ Newman. Stochastic blockmodels and community structure in networks. Physical Review E, 83(1):016107, 2011.

[23] Tiago P Peixoto. Hierarchical block structures and high-resolution model selection in large networks. Physical Review X, 4(1):011047, 2014.

[24] Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS Central Science, 4(2):268–276, 2018.

[25] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. TUDataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663, 2020.

[26] Scott Freitas, Yuxiao Dong, Joshua Neil, and Duen Horng Chau. A large-scale database for graph representation learning. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

[27] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9(1):5233, 2019.

[28] Cedric Sanders, Andreas Roth, and Thomas Liebig. Curvature-based pooling within graph neural networks. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 471–485, 2023.

[29] Enxhell Luzhnica, Ben Day, and Pietro Lio. Clique pooling for graph classification. arXiv preprint arXiv:1904.00374, 2019.

[30] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems, volume 35, pages 14501–14515, 2022.

[31] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. In International Conference on Machine Learning, pages 3734–3743, 2019.

[32] Ekagra Ranjan, Soumya Sanyal, and Partha Talukdar. ASAP: Adaptive structure aware pooling for learning hierarchical graph representations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 5470–5477, 2020.

[33] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral Clustering with Graph Neural Networks for Graph Pooling. In International Conference on Machine Learning, 2020.

[34] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical Graph Representation Learning with Differentiable Pooling. In Advances in Neural Information Processing Systems, 2018.

[35] Yash Deshpande, Subhabrata Sen, Andrea Montanari, and Elchanan Mossel. Contextual stochastic block models. In Advances in Neural Information Processing Systems, volume 31, 2018.

[36] Timo Stoll, Chendi Qian, Ben Finkelshtein, Ali Parviz, Darius Weber, Fabrizio Frasca, Hadar Shavit, Antoine Siraudin, Arman Mielke, Marie Anastacio, Erik Müller, Maya Bechler-Speicher, Michael Bronstein, Mikhail Galkin, Holger Hoos, Mathias Niepert, Bryan Perozzi, Jan Tönshoff, and Christopher Morris. GraphBench: Next-generation graph learning benchmarking. arXiv preprint arXiv:2512.04475, 2025.

[37] Paul Erdos and Alfréd Rényi. On random graphs I. ˝ Publicationes Mathematicae Debrecen, 6: 290–297, 1959.

[38] Ke Xu, Frédéric Boussemart, Fred Hemery, and Christophe Lecoutre. Random constraint satisfaction: Easy generation of hard (satisfiable) instances. Artificial Intelligence, 171(8): 514–534, 2007.

[39] Zhuo Xu, Lu Bai, Lixin Cui, Ming Li, Hangyuan Du, Ziyu Lyu, Yue Wang, and Edwin R Hancock. SSHPool: The separated subgraph-based hierarchical pooling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 27405–27413, 2026.

[40] Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, and Xavier Bresson. A generalization of ViT/MLP-Mixer to graphs. In International Conference on Machine Learning, pages 12724–12745, 2023.

[41] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 3558–3565, 2019.

[42] Iulia Duta and Pietro Lio. Sphinx: Structural prediction using hypergraph inference network. In International Conference on Machine Learning, pages 14884–14901, 2025.

[43] Sunwoo Kim, Soo Yong Lee, Yue Gao, Alessia Antelmi, Mirko Polato, and Kijung Shin. A survey on hypergraph neural networks: An in-depth and step-by-step guide. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 6534–6544, 2024.

[44] Yanbo Wang and Muhan Zhang. An empirical study of realized GNN expressiveness. In International Conference on Machine Learning, pages 52134–52155, 2024.

[45] Vikraman Arvind, Frank Fuhlbrück, Johannes Köbler, and Oleg Verbitsky. On Weisfeiler-Leman invariance: Subgraph counts and related graph properties. Journal of Computer and System Sciences, 113:42–59, 2020.

[46] Hongyang Gao and Shuiwang Ji. Graph U-Nets. In International Conference on Machine Learning, pages 2083–2092, 2019.

[47] Boris Knyazev, Graham W Taylor, and Mohamed Amer. Understanding attention and generalization in graph neural networks. In Advances in Neural Information Processing Systems, 2019.

[48] Cheng Shi, Liming Pan, Hong Hu, and Ivan Dokmanic. Homophily modulates double descent´ generalization in graph convolution networks. Proceedings ofthe National Academy ofSciences, 121(8):e2309504121, 2024.

[49] Nikil Wale, Ian A Watson, and George Karypis. Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems, 14:347–375, 2008.

[50] Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. Journal of Machine Learning Research, 24(43):1–48, 2023.

[51] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[52] Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart. Exploring network structure, dynamics, and function using NetworkX. In Proceedings of the 7th Python in Science Conference, pages 11–15, 2008.

[53] Tiago P. Peixoto. The graph-tool Python library. figshare, 2014. doi: 10.6084/m9.figshare. 1164194.

[54] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448–456, 2015.

[55] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[56] Tianle Cai, Shengjie Luo, Keyulu Xu, Di He, Tie-yan Liu, and Liwei Wang. GraphNorm: A principled approach to accelerating graph neural network training. In International Conference on Machine Learning, pages 1204–1215, 2021.

[57] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[58] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021.

[59] Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022.

[60] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. In International Conference on Learning Representations, 2020.

[61] Reid Andersen, Fan Chung, and Kevin Lang. Local graph partitioning using PageRank vectors. In 2006 47th Annual IEEE Symposium on Foundations ofComputer Science (FOCS’06), pages 475–486. IEEE, 2006.

## Appendix Overview

The appendix contains the following sections:

• Appendix A: Proofs for Theoretical Analysis. We present formal proofs and additional examples for the theoretical analysis presented in Section 3.2.

• Appendix B: Realized Expressivity on BREC. We validate our theoretical results from Section 3.2 on a benchmark testing model expressivity [44].

• Appendix C: Additional Results on Synthetic Datasets. We discuss additional results on our synthetic datasets for learned pooling schemes popular in practice and we extend our synthetic datasets with meaningful node features.

• Appendix D: Datasets. We present details on the datasets used in our experimental evaluation. We first explain the exact generation procedure to generate our synthetic benchmarks. Then, we give a short overview and summary statistics across datasets.

• Appendix E: Experimental Details. We report the necessary details to fully reproduce our experiments. We also briefly discuss the impact of hyperparameters on real-world datasets.

• Appendix F: Results for Additional Subgraph Policies. We present and briefly discuss additional results on the synthetic and real-world datasets for three more subgraph policies. These are not included in the main text as these are mainly devised by us, and therefore do not have a strong basis in literature.

## A Proofs for Theoretical Analysis

Assumptions. Throughout this appendix, we assume that all message passing layers are permutation-equivariant and at most 1-WL expressive, as is standard for MPNNs such as GIN [4]. We further assume that the within-subgraph message passing layer used by IP-GNN can realize the identity map, for instance via residual connections. Finally, whenever pooling reduces node representations to supernode representations, we assume that the reduction operator RED is injective on multisets of node embeddings, so that it does not destroy distinctions already present in the pooled clusters [11, 12]. For PSA-GNN, we also assume that the final subgraph-level or supergraph-level readout is chosen injectively whenever needed. All expressivity comparisons are understood in the sense of Definition 1.

## A.1 Proof of Proposition 3

We show first that IP-GNN is at least as expressive as Pool-GNN, and then give a graph pair for which it is strictly more expressive.

At least as expressive. Let $M _ { p o o l }$ be any Pool-GNN under a fixed hard partition $\mathcal { C } = \{ A _ { 1 } , \ldots , A _ { m } \}$ Consider the corresponding IP-GNN under the same partition. By assumption, the additional withinsubgraph message passing layer can realize the identity map. Hence we may choose its parameters so that it leaves all node representations unchanged before the subsequent ${ \bar { R } } E D { \cdot } C O N$ pooling step. With this choice, IP-GNN reproduces exactly the same computation as $M _ { p o o l }$ on every input graph. Therefore, for every Pool-GNN there exists an IP-GNN with identical outputs, and IP-GNN is at least as expressive as Pool-GNN.

Strictly more expressive. It remains to show that there exists a partition $\widetilde { \mathcal { C } }$ and a pair of graphs ethat can be distinguished by IP-GNN but not by Pool-GNN. Consider two graphs $G$ and H whose node sets are partitioned into two clusters A and B of equal size, with $G [ A ] \widetilde \cong P _ { 4 } , G [ B ] \cong P _ { 4 } .$ and $H [ A ] \cong K _ { 1 , 3 } \mathrm { \dot { , } } H [ B ] \cong K _ { 1 , 3 }$ . Add inter-cluster edges so that both full graphs are regular and so that the coarsened graph obtained after pooling is identical in both cases.

A model in Pool-GNN immediately reduces each cluster to a supernode. Since the pooled graph is the same for G and H, and since the pre-pooling message passing stage cannot distinguish the regular unlabeled input graphs, Pool-GNN cannot separate the pair.

By contrast, IP-GNN performs an additional message passing step inside each induced subgraph before reduction. A 1-WL-bounded MPNN can distinguish $P _ { 4 }$ from $K _ { 1 , 3 }$ via their different internal degree structure, and the injective reduction preserves this distinction at the supernode level. Hence IP-GNN distinguishes the pair, establishing strict separation. □

Additional example. The above argument is not limited to hand-crafted partitions. In practice, structure-based pooling policies can induce subgraphs whose internal structure is informative even when the final coarsened graph is not. Figure 8 gives an additional example based on the Shrikhande graph and the 4  4 Rook’s graph, a classical pair of non-isomorphic strongly regular graphs with parameters SRG(16, 6, 2, 2) [45]. Under an empirical Leiden-based partition, the induced subgraphs expose different internal degree signatures. These are accessible to IP-GNN and PSA-GNN, but are discarded by immediate coarsening.

## A.2 Proof of Proposition 4

We prove the two claims in turn. First, PSA-GNN with supergraph readout subsumes IP-GNN under any fixed policy π. Second, there exists a policy π such that the padded version of PSA-GNN is strictly more expressive than IP-GNN under the base policy.

We equip PSA-GNN with a supergraph readout only to match the post-pooling interface of IP-GNN in the expressivity argument. In practice, we use a DeepSets-style readout by default, since our hypothesis is that the relevant hierarchical signal is already captured by the processed pooling-induced substructures and their coupling through the original graph branch.

ShrikhandeGraph

![](images/b98f97952d255114556112880f7d38eac825dc57e45182b8afa65ad95224ba0f.jpg)  
(a) Original graph

![](images/89a1288213200d5bb6d51b01bccb7a92b95cddce9a3c01a561e0afb62e53f570.jpg)  
(b) Induced subgraphs

![](images/646ad6c58c228526914dd585c2615033b029f8934b7589e620e23eb77e159f9d.jpg)  
(c) Internal degrees

![](images/9a7a27876e6509183acef3163abeaf2b0478474b6ea76edf79faa9ea39ecf1d3.jpg)  
(d) Pooled graph  
Rook’sGraph

![](images/83f8d6695fb4e35def1df351b167a0e5f12144f83f3c4c883ee86b7d6bcd2204.jpg)

![](images/ca18cf0a37d0a5022cf74dc0b564ffb9d7b071975fcc90d530b4496b811a8426.jpg)

![](images/7cf2585d78a3992e66a8268d720cabef7edecf1016bfcc9640996ccf0f7ebc5b.jpg)

![](images/f0366048736d29b9989e8d225402a441312e54e87d6acea910aac91c478beecd.jpg)  
Figure 8: Additional example for Proposition 3. The Shrikhande graph and the $4 \times 4$ Rook graph are not distinguished by immediate coarsening under this Leiden partition, but their pooling-induced subgraphs exhibit different internal degree signatures that can be exploited by IP-GNN or PSA-GNN.

(1) At least as expressive as IP-GNN. Fix any subgraph policy π. Recall that one PSA-GNN layer updates subgraph-node features as

$$
X _ { i } ^ { \prime } = L ^ { \mathrm { s g } } ( A _ { i } , X _ { i } ) + \left( L ^ { \mathrm { g r a p h } } ( A , \bar { X } ) \right) _ { \pi _ { i } } ,
$$

where $L ^ { \mathrm { s g } }$ acts inside each selected subgraph and $L ^ { \mathrm { g r a p h } }$ acts on the original graph after scatteraggregating subgraph-node features back to node space.

Consider an IP-GNN with a single pooling step under the same policy π. Its computation can be decomposed into three stages: (i) message passing on the original graph before pooling, (ii) message passing inside the induced subgraphs before reduction, and (iii) message passing on the pooled supergraph after RED-CON.

A PSA-GNN with supergraph readout can emulate these stages sequentially:

• Pre-pooling global message passing is recovered by setting $L ^ { \mathrm { s g } } = 0$ and choosing $L ^ { \mathrm { g r a p h } }$ to match the corresponding original-graph layers.

• Within-subgraph message passing is recovered by setting $L ^ { \mathrm { g r a p h } } = 0$ and choosing $L ^ { \mathrm { s g } }$ to match the intra-subgraph layers.

• Since both models use the same policy π and the same RED-CON mechanism, they induce the same pooled graph. The post-pooling computation of IP-GNN can therefore be reproduced by replacing the DeepSets readout of PSA-GNN with a supergraph readout of the same depth.

Thus every computation realizable by IP-GNN under π can also be realized by PSA-GNN under π.   
Hence PSA-GNN is always at least as expressive as IP-GNN.

(2) Strict improvement under padded policies. Having shown that PSA-GNN subsumes IP-GNN under π, we now give an example of a pair of graphs that can be distinguished by PSA-GNN but not IP-GNN under a fixed pooling policy. Consider the two 4-regular graphs G and H in Figure 9. Both graphs have vertex set $\boldsymbol { \dot { V } } = \{ \bar { 1 } , \dots , \bar { 8 } \}$ and the same hard partition $\mathbf { \bar { \Phi } } A = \{ 1 , 2 , 3 , 4 \} , B = \{ 5 , 6 , 7 , 8 \}$ where the induced subgraphs on A and B are identical diamonds $\tilde { D ^ { } } = K _ { 4 } \stackrel { \cdot } { - } e . \ G$ and H are indistinguishable by 1-WL from constant node features but differ in their crossing edges as shown.

IP-GNN cannot distinguish this pair. Under the policy π, IP-GNN sees the same multiset of induced subgraphs, namely $\{ \bar { D , } D \}$ , in both graphs. Moreover, the coarsened graph is the same in both cases: two supernodes $A$ and B connected by six crossing edges. Therefore IP-GNN cannot distinguish G and H under this policy.

PSA-GNN distinguishes this pair after two computational layers using πˆ with core/halo node markings. We focus on the padded A-subgraph. In both graphs, the core is the same diamond on nodes 1, 2, 3, 4 , while the halo nodes are 5, 6, 7, 8 . After one 1-WL refinement step on the padded subgraph, the halo nodes split into two types: nodes 5 and 7 have two core neighbors, while nodes 6 and 8 have one core neighbor. In G, the two low-internal-degree core nodes receive different halo-neighborhood multisets 1 : 6, 8 and 3 : 5, 7 . Thus node 1 sees two halo nodes of padded degree 1, while node 3 sees two halo nodes of padded degree 2. In H, however, 1 : 8, 7 and 3 : 6, 5 , so nodes 1 and 3 each see one halo node of padded degree 1 and one halo node of padded degree 2. Consequently, after a second 1-WL refinement step, nodes 1 and 3 receive different colors in the padded A-subgraph of G, but remain indistinguishable in the padded A-subgraph of H.

A two-layer subgraph encoder L<sup>sg</sup> with a 1-WL-expressive base layer can realize these computations. Together with an injective subgraph readout and an injective permutation-invariant graph-level readout, PSA-GNN distinguishes the padded substructure collections of G and H. Hence padded pooledsubstructure processing can be strictly more expressive than unpadded intra-pooling processing.

![](images/aaac61d36ce5d032fc4ad60ed276559e484a9a1f0152895b960a6cf5854b6d7f.jpg)  
Figure 9: Strict-separation example for padded pooled-substructure processing. The two original graphs G and H are both 4-regular and use the same hard partition into two diamond subgraphs. Under the unpadded policy π, both graphs yield the same subgraphs and coarsened graph with six crossing edges, so IP-GNN cannot distinguish them. Under the padded policy πˆ, however, the A-subgraphs expose different crossing-edge incidence patterns. In G, the core nodes 1 and 3 see different halo-neighbor types and split after two 1-WL refinement steps; while they remain symmetric in H.

## B Realized Expressivity on BREC

In this section we verify the expressivity hierarchy from Section 3.2 on a standard realized expressivity benchmark [44]. We present results for Leiden, CliquePool, and CurvPool as well as the Egonet baseline in Figure 10. Clearly, the expected relationship appears in practice: IP-GNN improves over immediate coarsening, while PSA-GNN performs best overall. Experimental details are deferred to Appendix E.

![](images/1360367e5b32507451b10c685927d255ed778bc6d8cbe38c95246037e6906635.jpg)  
Figure 10: Number of BREC pairs distinguished by each architecture-policy combination, out of 400 total. Following the official BREC protocol, we report the best score achieved over three seeds.

## C Further Results on Synthetic Datasets

## C.1 Hierarchical GNNs on Hierarchy Benchmarks

We further evaluated hierarchical GNNs based on popular pooling schemes that learn cluster assignments based on node features. We distinguish between dense and sparse methods. Dense methods, such as DiffPool and MinCutPool [34, 33], typically learn a soft assignment matrix $\mathbf { M } \in \mathbb { R } ^ { | V | \times | V ^ { \prime } | }$ that maps all input nodes to supernodes. Sparse pooling methods typically learn to assign a score to each node and retain only the top-k nodes, dropping non-selected nodes.

We include DiffPool and MinCut as representative dense pooling schemes [34, 33] as well as TopKPool, SAGPool and ASAPool as sparse variants [46, 47, 31, 32]. As an additional case study, we include randomized variants of ASAPool and SAGPool [10].

![](images/dc7b309be24a0dd35406c66515dfa5816b867c9c886bca2e5d7a4d0466dbf74c.jpg)  
Figure 11: Results on the synthetic hierarchical benchmarks for learned pooling operators. We compare a range of popular learned pooling operators as well as their randomized implementations.

On MACRO-HIERARCHIES, all dense pooling schemes fail to improve over the MPNN baseline, likely because the pooling step has little informative node-feature signal to exploit. By contrast, ASAPool and SAGPool, two sparse pooling baselines, achieve best-in-class performance, especially in the harder regimes, albeit at the cost of a higher variance across seeds. Both also clearly outperform their randomized implementations, indicating that they learn meaningful hierarchical structure. Furthermore, TopKPool, which implements a similar but simpler coarsening logic, performs substantially worse, suggesting that success in this setting depends on more than sparsity alone. These findings motivate further research on the promise and failure modes of pooling schemes and validate our synthetic approach as a valuable tool to that end.

On GRANULAR-HIERARCHIES, all learned pooling-based GNNs fail to beat the MPNN baseline. Similarly to MACRO-HIERARCHIES, dense pooling operators might not learn meaningful cluster assignments from graph topology alone. Sparse schemes on the other hand drop nodes, which might not be a suitable inductive bias as solving the task requires recognizing and estimating cluster sizes.

## C.2 Contextual Node Features on Hierarchy Benchmarks

The original experiments on our synthetic datasets were conducted with constant node features as these benchmarks are designed to isolate hierarchical structural reasoning. To test whether this design influences our empirical results, we repeated the medium-difficulty experiments with F = 16 node features across varying levels of informativeness.

Following the contextual SBM framework [35] and generalizing the feature construction in Eq. 9 of Shi et al. [48] from binary labels to multiple communities, we generate

$$
\mathbf { x } _ { i } = \sqrt { \frac { \mu } { N } } \mathbf { u } _ { c _ { i } } + \pmb { \xi } _ { i } ,\tag{4}
$$

Table 3: Impact of contextual node features on synthetic hierarchy benchmarks. Test accuracy (mean standard deviation) on the medium-difficulty GRANULAR-HIERARCHIES and MACRO-HIERARCHIES datasets with $F = 1 6$ contextual node features. The gap denotes the accuracy difference in percentage points between PSA-GNN and the GIN baseline.
<table><tr><td></td><td colspan="3">Granular-Hierarchies</td><td colspan="3">Macro-Hierarchies</td></tr><tr><td>Feature Level</td><td>PSA-GNN (Leiden)</td><td>MPNN (GIN)</td><td>Gap</td><td>PSA-GNN (Leiden)</td><td>MPNN (GIN)</td><td>Gap</td></tr><tr><td>Featureless</td><td> $9 5 . 7 0 \pm 0 . 5 6$ </td><td> $\overline { { 5 3 . 7 3 \pm 1 . 8 4 } }$ </td><td>+41.97</td><td> $9 9 . 0 6 \pm 0 . 3 6$ </td><td> $5 2 . 5 0 \pm 0 . 6 4$ </td><td>+46.56</td></tr><tr><td> $\mu = 1$ </td><td> $9 5 . 1 0 \pm 0 . 3 6$ </td><td> $4 9 . 6 3 \pm 2 . 1 5$ </td><td>+45.47</td><td> $9 9 . 3 7 \pm 0 . 3 1$ </td><td> $4 9 . 4 7 \pm 1 . 0 0$ </td><td>+49.90</td></tr><tr><td> $\mu = 5$ </td><td> $9 5 . 3 3 \pm 0 . 4 7$ </td><td> $5 7 . 0 7 \pm 2 . 7 4$ </td><td>+38.26</td><td> $9 9 . 2 3 \pm 0 . 2 5$ </td><td> $5 1 . 4 0 \pm 2 . 4 9$ </td><td>+47.83</td></tr><tr><td> $\dot { \mu } = 1 0$ </td><td> $9 5 . 9 7 \pm 0 . 4 7$ </td><td> $6 6 . 2 0 \pm 1 . 3 2$ </td><td>+29.77</td><td> $9 9 . 5 3 \pm 0 . 0 6$ </td><td> $4 9 . 6 0 \pm 2 . 5 5$ </td><td>+49.93</td></tr><tr><td> $\mu = 5 0$ </td><td> $9 7 . 4 0 \pm 0 . 9 0$ </td><td> $9 2 . 6 7 \pm 0 . 6 0$ </td><td>+4.73</td><td> $9 9 . 4 0 \pm 0 . 2 6$ </td><td> $5 0 . 3 0 \pm 2 . 9 5$ </td><td>+49.10</td></tr></table>

where $c _ { i }$ is the lowest-level community of node i, $u _ { c } \sim \mathcal { N } ( 0 , I _ { F } / F )$ are random community prototypes, and $\xi _ { i } \sim \mathcal { N } ( 0 , I _ { F } / F )$ . The prototypes are resampled independently for every graph, so they provide noisy community information without directly encoding the graph-level label. Larger µ corresponds to more informative features.

On GRANULAR-HIERARCHIES, increasingly informative features help GIN identify the lowest-level communities and estimate their sizes, narrowing the performance gap. On MACRO-HIERARCHIES, however, information about these communities alone is insufficient to recover their higher-level ring or bipartite organization, and GIN remains near chance. These results suggest that informative node features can reduce the need for structural reasoning when closely aligned with the prediction target as expected, but do not eliminate PSA-GNN’s advantage when the task requires reasoning about relationships between communities.

## D Dataset Details

Here, we provide additional details on all datasets used in our experiments.

## D.1 Synthetic Datasets

This appendix section provides details of the exact construction procedure of graphs in both MACRO-HIERARCHIES and GRANULAR-HIERARCHIES. We generate 10,000 graphs for both benchmarks for various difficulty levels. All nodes are endowed with constant features, unless otherwise specified.

MACRO-HIERARCHIES. This dataset is generated by a two-level stochastic block model with M macro-communities, each consisting of K micro-communities of size S. The class signal is encoded in the connectivity pattern between macro-communities: at the macro level, they are arranged either as a ring or as a bipartite graph, as illustrated in Figure 3.

Concretely, we sample $M \in \{ 6 , 7 , 8 \}$ and $K \in \{ 3 , 4 , 5 \}$ , and then choose $S \in \{ 6 , 8 , 1 0 , 1 2 \}$ such that $M K S \approx 2 0 0$ . Nodes within the same micro-community are connected with probability $p _ { \mathrm { m i c r o } } = 0 . 3 5$ while nodes from different micro-communities within the same macro-community are connected with probability $p _ { \mathrm { m a c r o } } = 0 . 1 2$ . Connections between nodes in different macro-communities are then sampled according to the class pattern (ring or bipartite). If two macro-communities are adjacent in the underlying pattern, edges between them are generated with probability $p _ { \mathrm { s i g n a l } } = p _ { \mathrm { b a s e } } + \delta$ , where $p _ { \mathrm { b a s e } } = 0 . 0 1$ is a small background-noise term and $\delta \in ( 0 . 0 0 \bar { 2 } , 0 . 0 1 4 )$ controls the strength of the class signal. For pairs of macro-communities that are not connected in the underlying pattern, edges are sampled with probability $p _ { \mathrm { o u t s i d e } }$ . This quantity is chosen so that the expected node degree is matched across labels, preventing trivial separation by simple graph statistics. Varying δ therefore controls the difficulty of the task.

GRANULAR-HIERARCHIES. In our second dataset, the class signal is encoded in the relative sizes of the clusters. Graphs with $y = 0$ consist of equally sized clusters, whereas graphs with $y = 1$ contain clusters of unequal size, as illustrated in Figure 3.

As before, we sample the number of clusters as $M \in \{ 6 , 7 , 8 \}$ and fix the total number of nodes to $N = 2 0 0$ . We further set a target expected degree of $d _ { \mathrm { t a r g e t } } = 5$ . For class $y = 1$ , two clusters are chosen to be four times larger than the remaining ones, whereas for class $y = 0 ,$ , all clusters have equal size. Edges between nodes in different clusters are sampled with probability $p _ { \mathrm { g l o b a l } } \in \left( 0 . 0 0 2 , 0 . 0 2 2 \right)$ which acts as background noise and therefore controls the difficulty of the task. Within-cluster edges are then sampled with probability $p _ { \mathrm { w i t h i n } } .$ , chosen such that the expected node degree remains close to $d _ { \mathrm { t a r g e t } }$ . This prevents trivial separation based on the total number of edges.

## D.2 Dataset Statistics

Table 4 presents summary statistics of the main datasets used in our experiments.

Table 4: Summary statistics of the benchmark datasets. Dataset-level statistics are reported as mean standard deviation over graphs where applicable.
<table><tr><td>Dataset</td><td>Task</td><td># Graphs</td><td># Labels</td><td>Node feat. dim</td><td>Edge feat. dim</td><td>Avg. # Nodes</td><td>Avg. # Edges</td><td>Avg. Degree</td></tr><tr><td colspan="9">Hierarchical Structure</td></tr><tr><td>COLLAB</td><td>Classification</td><td>5,000</td><td>3</td><td>0</td><td>0</td><td> $7 4 \pm 6 2$ </td><td> $2 , 4 5 7 \pm 6 , 4 3 9$ </td><td> $3 7 . 4 \pm 4 4 . 0$ </td></tr><tr><td>REDDIT-5K</td><td>Classification</td><td>5,000</td><td>5</td><td>0</td><td>0</td><td> $5 0 9 \pm 4 5 3$ </td><td> $5 9 5 \pm 5 6 7$ </td><td> $2 . 3 \pm 0 . 2$ </td></tr><tr><td>REDDIT-12K</td><td>Classification</td><td>11,929</td><td>11</td><td></td><td>0</td><td> $3 9 1 \pm 4 2 9$ </td><td> $4 5 7 \pm 5 1 8$ </td><td> $2 . 3 \pm 0 . 2$ </td></tr><tr><td>MalNet</td><td>Classification</td><td>5,000</td><td>5</td><td>0 0</td><td>0</td><td> $1 , 5 2 2 \pm 1 , 6 5 0$ </td><td> $2 , 8 5 9 \pm 3 , 2 0 5$ </td><td> $3 . 0 \pm 1 . 2$ </td></tr><tr><td>CO-RB</td><td>Regression</td><td>10,000</td><td>1</td><td>0</td><td>0</td><td> $2 4 1 \pm 2 8$ </td><td> $4 , 4 9 7 \pm 1 , 5 0 4$ </td><td> $3 6 . 8 \pm 9 . 9$ </td></tr><tr><td colspan="9">Flat Structure</td></tr><tr><td>NCI1</td><td>Classification</td><td>4,110</td><td>2</td><td>37</td><td>0</td><td> $3 0 \pm 1 4$ </td><td> $3 2 \pm 1 5$ </td><td> $2 . 2 \pm 0 . 1$ </td></tr><tr><td>ZINC</td><td>Regression</td><td>12,000</td><td>1</td><td>1</td><td>1</td><td> $2 3 \pm 4$ </td><td> $2 5 \pm 5$ </td><td> $2 . 1 \pm 0 . 1$ </td></tr><tr><td>CO-ER</td><td>Regression</td><td>10,000</td><td>1</td><td>0</td><td>0</td><td> $2 5 0 \pm 2 9$ </td><td> $4 , 7 2 1 \pm 1 , 0 9 0$ </td><td> $3 7 . 3 \pm 4 . 4$ </td></tr><tr><td colspan="9">Synthetic, structure-controlled datasets</td></tr><tr><td>Macro-Hierarchies</td><td>Classification</td><td>10,000</td><td>2</td><td>0</td><td>0</td><td> $2 3 7 \pm 4 6$ </td><td> $9 4 2 \pm 3 2 2$ </td><td> $7 . 8 \pm 1 . 2$ </td></tr><tr><td>Granular-Hierarchies</td><td>Classification</td><td>10,000</td><td>2</td><td>0</td><td>0</td><td> $2 0 0 \pm 0$ </td><td> $6 0 0 \pm 2 3$ </td><td> $6 . 0 \pm 0 . 2$ </td></tr></table>

Our synthetic datasets are introduced in Section D.1. We briefly describe the real-world benchmarks in the domains of malware analysis, social interaction, scientific collaboration, and molecular property prediction. MalNet-Tiny contains function call graphs of malicious software and requires predicting the malware family, providing a large-scale and structurally diverse benchmark [26]. REDDIT-5K and REDDIT-MULTI-12K model Reddit discussion threads as user-interaction graphs and require predicting the originating subreddit [25]. COLLAB contains collaboration ego-networks of scientists, with the goal of classifying the research domain from graph topology [25]. NCI1 is a molecular benchmark in which the task is to predict anti-cancer activity from atom–bond graphs [25, 49]. ZINC is a popular molecular regression benchmark based on a 12K subset of the ZINC molecular graph dataset, where the task is to predict constrained solubility from atom–bond graphs [24, 50]. Finally, CO-ER and CO-RB are graph-level regression benchmarks from GraphBench [36], where the task is to predict the maximum independent set objective on Erdos–Rényi random graphs [˝ 37] and clique-structured RB graphs [38], respectively.

## E Experimental Details

We conduct all experiments with PyTorch Geometric [51], relying on NetworkX [52] and graphtool [53] for graph-specific algorithms. Computations are conducted on a single NVIDIA A100 GPU with 80GB $\mathrm { V R A M } ^ { 2 }$

Model Architecture. We combine all of our processing layers with a standardized base model architecture. Initial node features undergo a linear projection to the embedding space. Subgraph node markings (if used) are also projected and summed with the node embeddings. These features, defined either in node space or in subgraph-node space, are then transformed sequentially by the corresponding layer using a GIN with 2-layer MLPs. After each convolution, we apply batch normalization [54], ReLU activation, dropout, and a residual connection. We use a 2-layer MLP to transform aggregated subgraph features as in Eq. (2) and a 3-layer MLP to produce the final graph-level predictions, with all MLPs containing batch normalization, ReLU, and dropout. Unless otherwise indicated, we use mean aggregation throughout for stability, including for graph-level and supernode-level readout as well as for subgraph feature aggregation in Eq. (1) in PSA-GNN.

Experimental Setup. All models are matched in depth and overall parameter count, where appropriate. Generally, our goal is not to maximize absolute performance, but to compare the three model architectures in a controlled setting. For all deterministic pooling models, we use a single pooling step, ensuring a like-for-like comparison with PSA-GNN and allowing subgraph assignments to be precomputed. As discussed in Section 5, we compare all models under the same base policy π to isolate the effect of architecture, with PSA-GNN using the padded policy πˆ. Hyperparameters are selected based on validation performance. Across different seeds, we use a random 80%-10%-10% train-validation-test split, or a fixed split when one is provided.

Implementation of Subgraph Policies. We do not tune subgraph-policy hyperparameters and instead use a small set of fixed implementation defaults. Leiden is run with its default resolution parameter of 1.0. For CurvPool, we threshold Balanced Forman Curvature at the 0.8 quantile (which we compared against mean-thresholding), retain edges above this threshold, and merge overlapping selected edges into core clusters; uncovered nodes are subsequently added through a fallback coverage step. For CliquePool, we compute maximal cliques, then resolve overlaps by keeping each node only in its largest incident clique(s), iterating this pruning until stable. In practice, we add node markings beyond the core/halo indicators whenever they are naturally available from the extraction procedure.

## E.1 BREC

We largely follow the original BREC protocol [44]. For each pair of graphs, we train a model by optimizing the cosine embedding loss for a maximum of 40 epochs using Adam [55] with learning rate 10<sup>−4</sup> and weight decay 10<sup>−4</sup>. We also replace all batch normalization layers with GraphNorm layers [56] due to numerical stability issues. Finally, we use a sum-based graph-level readout.

After some heuristic tuning, we settled on 8 layers throughout. Table 5 shows how these layers are allocated for the Pool-GNN and IP-GNN models. With this setup, our Pool-GNN baseline already achieves stronger results than previous empirical investigations into expressive graph pooling (see Table 1 in [11]), supporting our hyperparameter choices and the use of a single pooling step. Intuitively, if a first pooling step does not already improve separation on BREC, additional pooling steps are unlikely to provide additional help.

Table 5: Layer allocations across datasets and model families. For IP-GNN and Pool-GNN, layers are split into pre, within, and post; for Hierarchical, they are split into pre, between, and post. PSA-GNN and any MPNN baselines always use the full stack, i.e., total depth. between layers are multiplied by the number of pooling steps.
<table><tr><td></td><td>Total Depth</td><td colspan="3">IP-GNN</td><td colspan="3">Pool-GNN</td><td colspan="3">Hierarchical</td></tr><tr><td>Dataset</td><td></td><td>pre within</td><td></td><td>post</td><td></td><td>pre within</td><td>post</td><td></td><td>pre between post</td><td></td></tr><tr><td>Hierarchical Structure</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COLLAB</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>2</td><td>2x2</td><td>0</td></tr><tr><td>REDDIT-12K</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>2</td><td>2x2</td><td>0</td></tr><tr><td>REDDIT-5K</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>2</td><td>2x2</td><td>0</td></tr><tr><td>MalNet</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>2</td><td>2x2</td><td>0</td></tr><tr><td>CO-RB</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Flat Structure</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NCI1</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td>2</td><td>2x2</td><td>0</td></tr><tr><td>ZINC</td><td>8</td><td>4</td><td>2</td><td>2</td><td>5</td><td>0</td><td>3</td><td>4</td><td>2x2</td><td>0</td></tr><tr><td>CO-ER</td><td>6</td><td>2</td><td>2</td><td>2</td><td>3</td><td>0</td><td>3</td><td></td><td>一</td><td>一</td></tr><tr><td colspan="2">Synthetic, structure-controlled datasets</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BREC</td><td>8</td><td>2</td><td>3</td><td>3</td><td>4</td><td>0</td><td>4</td><td></td><td></td><td>一</td></tr><tr><td>Macro-Hierarchies</td><td>8</td><td>4</td><td>2</td><td>2</td><td>5</td><td>0</td><td>3</td><td>4</td><td>2x2</td><td>2</td></tr><tr><td>Granular-Hierarchies</td><td>8</td><td>4</td><td>2</td><td>2</td><td>5</td><td>0</td><td>3</td><td>4</td><td>2x2</td><td>2</td></tr></table>

## E.2 Synthetic Datasets

We performed a small hyperparameter sweep over the number of layers 8, 12 and embedding size 64, 128 , evaluating each setting on two seeds. We find that larger models achieve slightly better absolute results, but the relative ordering of the models and the performance gaps between them remain largely unchanged (see Figure 5). We therefore settle on 8 layers with an embedding size of 64. To roughly match the number of parameters, we reduce the embedding size of PSA-GNN to 48. Exact layer allocations for each model are presented in Table 5. We use AdamW [57] for optimization. All other optimization and model hyperparameters are detailed in Table 6.

The GPS [30] baseline uses a GIN layer as the local message-passing component and employs Performer attention [58] for scalability. We set the embedding size to 18 to closely match the parameter budget of the other baselines. We further add Random Walk Positional/Structural Encodings [59] with length 16, a standard structural encoding for graph transformers. In preliminary experiments, GPS without positional/structural encodings performed substantially worse on our synthetic tasks.

Table 6: Dataset-specific training and architecture hyperparameters. Key hyperparameters used for each benchmark dataset. For BREC, no patience-based early stopping is used; instead, training stops once the loss falls below 0.2 or after 40 epochs.
<table><tr><td>Dataset</td><td># Layers</td><td>Max Epochs</td><td>Batch Size</td><td>Learning Rate</td><td>Weight Decay</td><td>Layers Dropout</td><td>Head Dropout</td><td>Early stop Patience</td><td># Seeds</td></tr><tr><td colspan="10">Hierarchical Structure</td></tr><tr><td>COLLAB</td><td>6</td><td>300</td><td>32</td><td> $1 0 ^ { - 3 }$ </td><td>0</td><td>0.05</td><td>0.10</td><td>50</td><td>5</td></tr><tr><td>REDDIT-12K</td><td>6</td><td>300</td><td>16</td><td> $1 0 ^ { - 3 }$ </td><td>0</td><td>0.05</td><td>0.10</td><td>50</td><td>5</td></tr><tr><td>REDDIT-5K</td><td>6</td><td>300</td><td>16</td><td> $1 0 ^ { - 3 }$ </td><td>0</td><td>0.05</td><td>0.10</td><td>50</td><td>5</td></tr><tr><td>MalNet</td><td>6</td><td>300</td><td>16</td><td> $1 0 ^ { - 3 }$ </td><td>0</td><td>0.05</td><td>0.10</td><td>50</td><td>5</td></tr><tr><td>CO-RB</td><td>6</td><td>300</td><td>32</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 5 }$ </td><td>0.00</td><td>0.10</td><td>50</td><td>3</td></tr><tr><td colspan="10">Flat Structure</td></tr><tr><td>NCI1</td><td>6</td><td>300</td><td>32</td><td> $1 0 ^ { - 3 }$ </td><td>0</td><td>0.00</td><td>0.10</td><td>50</td><td>10</td></tr><tr><td>ZINC</td><td>8</td><td>500</td><td>64</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 5 }$ </td><td>0.00</td><td>0.00</td><td>100</td><td>3</td></tr><tr><td>CO-ER</td><td>6</td><td>300</td><td>32</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 5 }$ </td><td>0.00</td><td>0.10</td><td>50</td><td>3</td></tr><tr><td colspan="10">Synthetic, structure-controlled datasets</td></tr><tr><td>BREC</td><td>8</td><td>40</td><td>16</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.00</td><td>0.00</td><td>一</td><td>3</td></tr><tr><td>Macro-Hierarchies</td><td>8</td><td>250</td><td>128</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td><td>0.00</td><td>0.00</td><td>50</td><td>5</td></tr><tr><td>Granular-Hierarchies</td><td>8</td><td>250</td><td>128</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td><td>0.00</td><td>0.00</td><td>50</td><td>3</td></tr></table>

We also tuned the number of pooling steps in 1, 2 and the pooling strength using either ratios 0.1, 0.5 or a fixed target size of 24 nodes, which roughly corresponds to the number of microcommunities on the MACRO-HIERARCHIES dataset. We find that a pooling ratio of 0.5 consistently performs best. For all methods except TopKPool, a single pooling step is optimal.

## E.3 Real-World Datasets

For the real-world benchmarks, we use 6 layers on all datasets except ZINC, where 8 layers is more appropriate. Model-specific allocations of layers are reported in Table 5. All other hyperparameters used are detailed in Table 6. Given this model specification, we set the embedding dimension such that each model fits within a 500K parameter budget. As ZINC uses edge features, we first embed edge features and subsequently employ the GINE layer [60] for processing. For datasets without node features, we use constant scalars.

We tune the number of pooling steps in 1, 2 for all hierarchical GNNs and find that using two pooling steps is generally beneficial. We further ablate the use of subgraph node markings and the edge-covering policy πˆ. Table 7 presents the result of this experiment. The base model always includes node markings and the padded policy. The impact on performance depends on the dataset and subgraph-policy combination. In most cases, however, the effect is small and remains within one standard deviation of the base model, with the clearest exception being clique-based policies on COLLAB.

Finally, Table 8 reports the average training time per epoch of the real-world datasets NCI1 and REDDIT-12K discussed in Section 5.4. Although the subgraph-based architectures introduce addi-

Table 7: Padding and node-marking impact for PSA-GNN on real-world datasets. The base row is the mean-aggregation configuration with node markings enabled and default padding. Italic rows report the change relative to the base. For accuracy metrics, positive values indicate improvement; for ZINC, negative values indicate improvement because lower MAE is better. Models without padding also contain no node markings, as there is no core vs. halo distinction. Empty cells indicate that a configuration was not tested due to computational limitations.
<table><tr><td colspan="2"></td><td colspan="3">Hierarchical Structure</td><td colspan="2">Flat Structure</td></tr><tr><td>Model</td><td>Subgraph Policy</td><td>COLLAB Acc. [%] (↑)</td><td>REDDIT-12K  $A c c . \ : \left. l \% \right. ] ( \uparrow )$ </td><td>MalNet  $A c c . \ : \left. l \% \right. ] ( \uparrow )$ </td><td>NCI1  $A c c . \ : \left. l \% \right. ] ( \uparrow )$ </td><td>ZINC MAE (↓)</td></tr><tr><td rowspan="7">PSA-GNN</td><td>CliquePool</td><td>78.1 ± 1.3</td><td>47.0 ± 3.8</td><td>90.6 ± 2.4</td><td> $8 1 . 5 \pm 2 . 0$ </td><td>0.181 ± 0.005</td></tr><tr><td>- padding</td><td>-4.4</td><td>+0.8</td><td>+1.0</td><td>-0.6</td><td>-0.002</td></tr><tr><td>- node markings</td><td>-2.3</td><td>+0.8</td><td>+1.0</td><td>+0.0</td><td>+0.010</td></tr><tr><td>CurvPool</td><td>76.8 ± 1.5</td><td>49.3 ± 1.7</td><td>92.0 ± 1.9</td><td>81.2 ± 1.6</td><td>0.175 ± 0.005</td></tr><tr><td>- padding</td><td>-0.5</td><td>-0.3</td><td>-1.2</td><td>-0.8</td><td>+0.003</td></tr><tr><td>- node markings</td><td>+0.5</td><td>+0.8</td><td>-0.7</td><td>-0.8</td><td>-0.001</td></tr><tr><td>Leiden</td><td>76.4 ± 2.2</td><td>50.0 ± 1.3</td><td>90.9 ± 1.8</td><td>77.8 ± 2.3</td><td>0.215 ± 0.012</td></tr><tr><td></td><td>- padding - node markings</td><td>-1.4 -1.8</td><td>-0.8 -0.0</td><td>+0.2 -0.4</td><td>+1.0 +2.1</td><td>-0.006 +0.002</td></tr></table>

tional computational overhead relative to the MPNN baseline, their runtime remains within the same order of magnitude on both datasets, consistent with the theoretical complexity analysis.

Table 8: Training time per epoch on NCI1 and REDDIT-12K. Mean training time in seconds per epoch ( standard deviation) for the datasets considered in Figure 7.
<table><tr><td>Model</td><td>Subgraph Policy</td><td>NCI1</td><td>REDDIT-12K</td></tr><tr><td rowspan="3">PSA-GNN</td><td>CliquePool</td><td> $2 . 4 7 \pm 0 . 1 2$ </td><td> $1 6 . 1 1 \pm 0 . 8 9$ </td></tr><tr><td>CurvPool</td><td> $2 . 7 5 \pm 0 . 1 8$ </td><td> $1 8 . 9 5 \pm 2 . 8 8$ </td></tr><tr><td>Leiden</td><td> $2 . 4 5 \pm 0 . 0 9$ </td><td> $1 7 . 1 1 \pm 0 . 7 5$ </td></tr><tr><td rowspan="3">IP-GNN</td><td>CliquePool</td><td> $2 . 1 5 \pm 0 . 1 7$ </td><td> $1 5 . 0 4 \pm 2 . 6 2$ </td></tr><tr><td>CurvPool</td><td> $2 . 0 8 \pm 0 . 1 0$ </td><td> $1 5 . 0 5 \pm 1 . 3 0$ </td></tr><tr><td>Leiden</td><td> $2 . 0 4 \pm 0 . 0 4$ </td><td> $1 3 . 7 9 \pm 1 . 4 9$ </td></tr><tr><td rowspan="3">Pool-GNN</td><td>CliquePool</td><td> $2 . 0 8 \pm 0 . 1 5$ </td><td> $1 3 . 4 1 \pm 1 . 9 2$ </td></tr><tr><td>CurvPool</td><td> $2 . 0 2 \pm 0 . 1 2$ </td><td> $1 4 . 6 8 \pm 2 . 3 7$ </td></tr><tr><td>Leiden</td><td> $1 . 9 9 \pm 0 . 0 5$ </td><td> $1 4 . 0 6 \pm 0 . 9 6$ </td></tr><tr><td>MPNN (GIN)</td><td></td><td> $1 . 1 6 \pm 0 . 0 5$ </td><td> $9 . 1 6 \pm 0 . 5 4$ </td></tr><tr><td>Egonet</td><td></td><td> $2 . 5 2 \pm 0 . 1 6$ </td><td></td></tr></table>

## F Additional Results for Other Policies

There are not many widely used structure-based pooling schemes in the graph pooling literature [11]. To expand our evaluation, we therefore repeat our experiments with three additional policies described below.

• CliquePool-Maximal is our overlapping variant of CliquePool, in which we keep all maximal cliques as subgraphs instead of resolving overlaps. In the padded version, we add any missing edge to every incident clique-subgraph rather than performing a full 1-hop expansion, so the resulting subgraphs are no longer uniform and contain additional structural variation.

• ACL applies the Andersen–Chung–Lang localized clustering algorithm [61] iteratively, each time seeding it at the highest-degree node that has not yet been assigned to a cluster. In detail, clusters are extracted from a personalized PageRank sweep and constrained to cover between 0.02 and 0.3 of the graph volume; if no sweep prefix falls in this range, we choose the prefix closest to the target range. For computational reasons, we cap the iterative ACL procedure at 1000 extracted subgraphs, after which any remaining uncovered nodes are assigned through a rescue step.

• SeededHops is our iterative seed-based variant, which forms a 1-hop egonet around the highest-degree node that has not yet been assigned to a cluster, and repeats this process until all nodes are assigned.

Note that the latter two are seed-based policies, where we seed a subgraph extraction logic at some node and repeat until coverage is achieved. These are closer in spirit to the egonet baseline, which extracts 2-hop expansions around every node.

These policies also give the opportunity to benchmark the usage of additional subgraph node features. In particular, for seed-based policies we add a root-node indicator together with distance-to-root features. For ACL, we additionally include the normalized personalized PageRank score of each node within the extracted cluster.

Figure 12 reports the results for the BREC dataset. These new policies follow the exact same trend as the structure-based policies, with PSA-GNN outperforming Pool-GNN and IP-GNN occupying the middle ground. CliquePool-Maximal stands out as a particularly successful approach. This could be potentially explained by the increase in overlap across clusters among all maximal cliques. Notably, this more powerful approach only outperforms standard CliquePool when combined with PSA-GNN but not when processed by Pool-GNN, indicating that more powerful policies need more powerful processing approaches.

![](images/72d4fe5f90b8417f4f24d5043842c2b8927f0181b66eee4e34673d259ec8cd02.jpg)  
Figure 12: Number of BREC graph pairs (out of 400) distinguished by each architecture-policy combination. We report the best score achieved over three seeds.

Next, we repeated the experiment on our synthetic datasets, with results presented in Figure 13, excluding CliquePool-Maximal for computational reasons. Again, the pattern discussed in the main text is clearly confirmed on these additional policies. Note that due to the structure of the dataset, even Egonets arguably group nodes belonging to the same community, as clusters are more densely connected.

Finally, we present an expanded Table 9 reporting results on real-world benchmarks including our new policies. PSA-GNN remains the top performing model. CliquePool-Maximal emerges as a powerful policy, reaching the best result on all hierarchical datasets tested, while remaining computationally feasible.

![](images/45406c507d5d367e0cef7df9bc43835a81fd8cb5b4669ccd2853204e892fd2ea.jpg)  
Figure 13: Results on the synthetic hierarchical benchmarks for additional subgraph policies. We compare immediate coarsening (Pool-GNN), the within-subgraph baseline (IP-GNN), and the full pooled-subgraph model (PSA-GNN) across three additional subgraph policies.

Table 9: Results on real-world benchmark datasets. The best overall result per dataset is shown in brown, second best in purple. Among PSA-GNN, IP-GNN, and Pool-GNN, the best method for a given subgraph policy π is shown in bold. The Avg. Rank column reports the average rank of these three architecture levels across all available policy–dataset comparisons.
<table><tr><td></td><td></td><td></td><td colspan="3">Hierarchical Structure</td><td colspan="2">Flat Structure</td></tr><tr><td>Model</td><td>Subgraph Policy</td><td>Avg. Rank (↓)</td><td>COLLAB Acc. [%] (↑)</td><td>REDDIT-12K Acc. [%] (↑)</td><td>MalNet-T Acc. [%] (↑)</td><td>NCI1 Acc. [%] (↑)</td><td>ZINC MAE (↓)</td></tr><tr><td>PSA-GNN</td><td>CliquePool</td><td></td><td>78.1 ± 1.3</td><td>47.8 ± 1.4</td><td>90.6 ± 2.4</td><td>81.5 ± 2.5</td><td>0.182 ± 0.008</td></tr><tr><td></td><td>CliquePool-max</td><td></td><td>78.9 ± 0.7</td><td>51.2 ± 0.0</td><td>92.3 ± 1.6</td><td>80.9 ± 1.6</td><td>0.179 ± 0.003</td></tr><tr><td></td><td>CurvPool</td><td>1.34</td><td>76.8 ± 1.5</td><td>50.1 ± 1.7</td><td>92.0 ± 1.9</td><td>81.2 ± 1.6</td><td>0.177 ± 0.005</td></tr><tr><td></td><td>Leiden</td><td></td><td>76.4 ± 2.2</td><td>50.0 ± 1.3</td><td>91.1 ± 1.4</td><td>80.0 ± 2.5</td><td>0.213 ± 0.007</td></tr><tr><td></td><td>ACL</td><td></td><td></td><td></td><td>92.1 ± 0.9</td><td>80.9 ± 2.8</td><td>0.189 ± 0.002</td></tr><tr><td></td><td>SeededHops</td><td></td><td>72.2 ± 4.6</td><td>50.4 ± 1.4</td><td>91.4 ± 2.0</td><td>81.1 ± 2.6</td><td>0.184 ± 0.006</td></tr><tr><td>IP-GNN</td><td>CliquePool</td><td></td><td>75.0 ± 2.3</td><td>45.1 ± 2.1</td><td>91.7 ± 1.7</td><td>80.6 ± 2.1</td><td>0.183 ± 0.004</td></tr><tr><td></td><td>CliquePool-max</td><td></td><td>78.1 ± 0.4</td><td>49.3 ± 0.7</td><td>91.6 ± 1.4</td><td>80.2 ± 2.4</td><td>0.187 ± 0.002</td></tr><tr><td></td><td>CurvPool</td><td>2.32</td><td>77.4 ± 0.9</td><td>42.0 ± 2.5</td><td>89.3 ± 2.0</td><td>80.3 ± 1.9</td><td>0.184 ± 0.005</td></tr><tr><td></td><td>Leiden</td><td></td><td>73.6 ± 3.1</td><td>45.6 ± 0.8</td><td>90.7 ± 1.1</td><td>80.1 ± 2.6</td><td>0.186 ± 0.004</td></tr><tr><td></td><td>ACL</td><td></td><td></td><td></td><td>90.5 ± 1.5</td><td>80.7 ± 3.6</td><td>0.248 ± 0.005</td></tr><tr><td></td><td>SeededHops</td><td></td><td>71.5 ± 5.9</td><td>48.8 ± 1.0</td><td>89.6 ± 2.3</td><td>81.5 ± 1.6</td><td>0.209 ± 0.002</td></tr><tr><td>Pool-GNN</td><td>CliquePool</td><td></td><td>70.4 ± 4.1</td><td>48.3 ± 1.7</td><td>90.2 ± 1.6</td><td>81.0 ± 2.4</td><td></td></tr><tr><td></td><td>CliquePool-max</td><td></td><td>76.0 ± 2.3</td><td>46.1 ± 6.4</td><td>90.2 ± 2.3</td><td>80.3 ± 2.4</td><td>0.182 ± 0.004 0.181 ± 0.004</td></tr><tr><td></td><td>CurvPool</td><td>2.34</td><td>71.1 ± 4.5</td><td>47.4 ± 0.4</td><td>91.3 ± 1.1</td><td>81.3 ± 2.0</td><td>0.178 ± 0.004</td></tr><tr><td></td><td>Leiden</td><td></td><td>74.1 ± 2.3</td><td>44.7 ± 1.6</td><td>87.9 ± 1.5</td><td>79.9 ± 2.0</td><td>0.257 ± 0.013</td></tr><tr><td></td><td>ACL</td><td></td><td></td><td></td><td>89.8 ± 2.0</td><td>80.8 ± 2.1</td><td>0.270 ± 0.005</td></tr><tr><td></td><td>SeededHops</td><td></td><td>73.2 ± 2.9</td><td>48.4 ± 1.9</td><td>90.4 ± 1.6</td><td>81.6 ± 2.0</td><td>0.229 ± 0.003</td></tr><tr><td>Hierarchical GNN</td><td>ASAPool</td><td></td><td>71.6 ± 5.4</td><td>49.5 ± 0.2</td><td>88.2 ± 3.1</td><td></td><td></td></tr><tr><td></td><td>SAGPool</td><td></td><td>70.6 ± 4.2</td><td>46.6 ± 1.8</td><td>90.1 ± 1.1</td><td>81.2 ± 2.0 80.1 ± 1.9</td><td></td></tr><tr><td></td><td>MinCutPool</td><td></td><td></td><td></td><td></td><td>80.2 ± 2.0</td><td>0.416 ± 0.107 0.161 ± 0.003</td></tr><tr><td></td><td>DiffPool</td><td></td><td></td><td></td><td></td><td>81.0 ± 1.7</td><td>0.147 ± 0.005</td></tr><tr><td>Baselines</td><td>Egonet</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>MPNN (GIN)</td><td></td><td>74.9 ± 3.7</td><td>48.3 ± 2.3</td><td>91.4 ± 1.4 91.0 ± 1.6</td><td>81.6 ± 2.4 80.9 ± 2.5</td><td>0.403 ± 0.060 0.172 ± 0.005</td></tr></table>