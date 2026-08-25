# GRAPH REPRESENTATION LEARNING OFLIGHTWEIGHT IOT CIPHERS

PREPRINT VERSION

Jonathan Cook\*, Sabih ur Rehman\* and M. Arif Khan\* \*School of Computing, Mathematics and Engineering, Charles Sturt University, Australia

## ABSTRACT

SIMON and SIMECK belong to a family of Lightweight Cryptographic Algorithms (LCAs) based on the Feistel block cipher, designed for Internet of Things (IoT) devices. As with all Feistel ciphers, they are susceptible to differential cryptanalysis, necessitating rigorous resilience evaluations. While state of-the-art techniques leverage heuristics and sampling to improve efficiency, little work has applied Machine Learning (ML) guided Graph Representation Learning (GRL) to efficiently identify and visualise high-probability differential clusters. We address this gap by introducing an efficient feature engineering strategy that extracts four differential attributes from a partial Difference Distribution Table (pDDT), revealing structural information concealed in raw differential data. Utilising the enriched features, we construct and compare three ML-guided directed graphs for SIMON32 and SIMECK32 using K-Nearest Neighbour (KNN), Decision Trees (DT), and Random Forests (RF). To the best of our knowledge, our framework produces the first graph-based visualisation of the differential clustering effect, in which high-probability single-bit differentials form geometrically close clusters in the learned embedding. All three models achieve a precision of 1.0 in identifying high-probability differentials, confirming zero false positives. KNN achieves the strongest cluster separation, the highest F1 score and the lowest graph construction time of approximately 2.3 seconds, while DT and RF produce optimal paths with near-perfect regression. The results are consistent across both LCAs, demonstrating the applicability of the framework to other AND-rotation LCA families.

Keywords Differential Cryptanalysis, Graph Representation Learning, Lightweight Encryption, Machine Learning, Internet of Things (IoT)

Preprint note: This is the author’s version of a paper accepted at the 33rd International Conference on Neural Information Processing (ICONIP 2026). The final authenticated version will appear in Springer Communications in Computer and Information Science (CCIS).

## 1 Introduction

Internet of Things (IoT) devices face significant security, privacy, and integrity challenges due to their resourceconstrained environments [1, 8, 18]. Several Lightweight Cryptographic Algorithms (LCAs) suitable for resourceconstrained applications have been developed in recent years, beginning with SIMON and SPECK in 2013 [2], with SIMON being designed specifically for IoT devices. In 2015, researchers at the University of Waterloo proposed SIMECK, combining properties of both [23]. SIMON and SIMECK employ bitwise-AND rotation operations [21], which, although efficient, are known to be vulnerable to adversarial attacks, necessitating rigorous security evaluations using cryptanalytic methods [13], ideally targeting as many rounds as possible while maintaining efficiency [11].

Existing techniques improve cryptanalysis through optimisation strategies, Machine Learning (ML) and graphs, with many using a Difference Distribution Table (DDT) or partial DDT (pDDT) [3, 6, 7, 10, 12, 14, 20]. However, a pDDT comprises independent differential records with limited differential information, hindering advances utilising contemporary digital technologies, with efficiency degrading as the table size grows with word size [7, 12]. To address this, [10] proposed efficient FE techniques of pDDT differentials to extract additional features from characteristics identified in [9], aiding ML predictions in a recommendation system. The optimal model exploiting the observed clustering effect of differentials in SIMON [16] via K-Means clustering. However, the clustering effect in SIMON and SIMECK via graph representations has not yet been investigated. While a pDDT has previously been transformed into a graph structure using Neo4j [6], a significant clustering effect was not observed due to the absence of learnable features. By leveraging additional engineered features and Graph Representation Learning (GRL) [5, 15], high-probability differential regions can be identified, leading to the following research question: How can ML algorithms be utilised in GRL to identify high probability differential clusters, and how do various models compare?

![](images/e4bd2b8a7ec1f6f9c5bc67d067e36de9be1f79c6fda50154311659cea5ee652c.jpg)  
Figure 1: Our methodology

This paper presents the following contributions. First, we introduce a GRL framework that converts the pDDT into a directed graph in which nodes represent differential pairs and edges encode structural and probabilistic proximity, extending the work of [10] and [9]. Secondly, to the best of our knowledge, we present the first graph-based visualisation of the differential clustering effect [16], demonstrating that high-probability differentials manifest as geometric clusters in a learned feature embedding, transforming an analytically described observation into an algorithmically exploitable structure. Third, we conduct a comparative evaluation of K-Nearest Neighbour (KNN), Decision Trees (DTs), and Random Forests (RFs) as graph construction engines, assessing cluster quality and structural graph qualities. Fourth, we demonstrate that these models yield meaningfully distinct graph topologies, providing insights into the trade-offs between efficiency, graph construction and differential neighbourhood accuracy. The remainder of the document is presented as follows. Section 2 describes the methods used in this study. The results of this work are presented in Section 3. We present a formal discussion in Section 4. Finally, Section 5 presents our concluding comments.

## 2 Methodology

While graphs are typically utilised to describe complex relationships in vast data, dense graphs can complicate interpretation, making them challenging to understand [17]. With a pDDT in place, we utilise ML to guide a KG towards high-probability differentials and the stable state. However, we must first apply Feature Engineer (FE) to extract additional differential characteristics from the pDDT. A concise illustration of our methodology is presented in Figure 1.

## 2.1 Feature Engineering pDDT differentials

To effectively guide ML in identifying high-probability differentials, we augment the pDDT with additional features. The pDDT file is imported into the Python Pandas package, producing a dataframe with null-initialised feature columns to be populated during FE.

## 2.1.1 Hamming Weight (HW) of $\Delta _ { X } ( \Delta _ { X _ { w } } ) , \Delta _ { Y } ( \Delta _ { Y _ { w } } )$ and $\Delta _ { Z } \left( \Delta _ { Z _ { w } } \right)$

Counts the number of bits set to one in the binary string of both input differentials, is used to identify single-bit (power-of-two) differentials and the min/max values.

## 2.1.2 Inputs match (∆<sup>ˆ</sup> )

Determines if both input differentials share identical bit positions $( \mathbf { e . g . } , \Delta _ { X } = \Delta _ { Y } )$ . A Boolean value is returned as a differential feature.

## 2.1.3 Probability transformation

Transforms the probability score to a logarithmic scale, enabling a scalable representation of probabilities across low threshold configurations.

## 2.1.4 arg min $m a x ( \Delta _ { X } )$ and arg min max $\left( \Delta _ { Y } \right)$

Used to identify if input differentials are minima or maxima values in the pDDT word size space, returning a Boolean value derived from the HW.

## 2.2 Graph Creation

For our analysis, we compare three ML models, KNN, DT and RF, in guiding a graph construction towards highprobability, non-trivial differentials. Although k-means has previously been utilised to improve efficiency and increase the number of rounds targeted, KNN and DT offered efficiency advantages with natural branching to nearby data points [10]. The efficiency of KNN and DT underscores their potential to guide a graph towards high-probability differentials. Graph size is limited to 500 nodes by importance, producing a transparent structure in which relationships between nodes are readily interpretable. To evaluate each model, we employ the following definitions:

Definition 1 Let the experimental time recordings be a set of ${ \mathrm { ' } } T _ { E } = \left\{ t _ { 1 } , t _ { 2 } , t _ { 3 } \ldots t _ { n } \right\}$ recorded in seconds. The total time $( T )$ is the sum of all time recordings of the set $T _ { E } .$ . From the set of time recordings, we derive the mean graph execution time $( \hat { T } _ { G } )$ as $\begin{array} { r } { \bar { T } _ { G } = \frac { T } { | T _ { E } | } } \end{array}$

Definition 2 The total number of edges in a graph highlights the overall potential attack paths.

Definition 3 Total number of critical attack paths. Critical attack paths are edges connecting the highest probability differentials.

Definition 4 Hops from random differential to stable-state differential via edge traversal $( E _ { h o p } ) .$ . Many optimisation strategies employ heuristic search methods beginning at a random differential and continually sampling until a highprobability differential is identified. Employing a random starting differential, the number ofhopsfrom the random starting differential to the stable-state node.

Definition 5 Time per random hop $( T _ { h o p } )$ evaluates the time required to create the graph and the number of hops from a random starting node and is defined as $\begin{array} { r } { T _ { h o p } = \frac { \bar { T } _ { G } } { E _ { h o p } } . } \end{array}$

Graphs are rendered using a seeded Fruchterman-Reingold force-directed layout [22], where node coordinates are initialised by layer towards concentric arrangement. Nodes and edges are coloured and scaled separately to highlight key structural features and facilitate the identification of shortest paths to the stable-state root node. Graph construction utilises 5 complementary distance and penalty concepts:

1. Euclidean Distance (ED) is a measure that drives the neighbour (KNN) or branch (DT) selection process as a distance to the stable state node and is used in edge weight calculations.

2. Importance score $( I _ { s } )$ is a custom heuristic that ranks nodes by their differential properties, prioritising $\Delta _ { X } ^ { \mathrm { ~ ~ } } = \Delta _ { Y } , _ { } \mathrm { H W } = 1$ , and a high Differential Probability (DP).

3. PageRank (PR) [19] is utilised as a centrality measure, ranking nodes by the significance of their connectivity within the graph. PR scores are boosted post-computation for the stable-state node and high-priority differential types to reflect their cryptanalytic significance.

4. Hamming Distance (HD) measure on $\Delta _ { X }$ to $\Delta _ { X } ^ { \prime }$ and $\Delta _ { Y }$ to $\Delta _ { Y } ^ { \prime }$ evaluates single-bit transitions between differentials.

5. Shortest path from each node to the root node, computed using Dijkstra’s Algorithm with edge cost (cost $( v ^ { \prime } $ $v ) = 1 / w e i g h t )$ , where $( \boldsymbol { v } , \boldsymbol { v } ^ { \prime } )$ is the direct link from node v to node $v ^ { \prime }$ , quantifying proximity to the root.

## 2.2.1 K-Nearest Neighbour

The KNN-guided graph utilises the HW of $\Delta _ { X } , \Delta _ { Y } , \Delta _ { Z }$ , and the logarithmic probability as features in the feature space. The stable-state root node is defined as the differential with properties of $\bar { \Delta _ { X _ { w } } } = 0 , \bar { \Delta } _ { Y _ { w } } = 0$ and $\Delta _ { Z _ { w } } = 0$ and the highest probability. If absent, the fallback is $\hat { \Delta } = 1$ with minimum HW and the highest probability. Node layers are assigned as:

$$
l a y e r = \left\{ \begin{array} { l l } { \Delta _ { X _ { w } } } & { \mathrm { i f } \hat { \Delta } = 1 } \\ { \operatorname* { m i n } ( \Delta _ { X _ { w } } , \Delta _ { Y _ { w } } ) + 1 } & { \mathrm { \bf { o t h e r w i s e } . } } \end{array} \right.\tag{1}
$$

For each node, the nearest neighbours are enumerated, and directed edges are created towards neighbours with shorter ED to the root. The directed edge weight $( W _ { e } )$ is defined as:

$$
W _ { e } = S i m \times P _ { t } ,\tag{2}
$$

where $P _ { t }$ is the target probability and $S i m$ is defined as:

$$
S i m = \frac { 1 } { 1 + E D } .\tag{3}
$$

During rendering, a random non-root node is selected as a starting position, and the shortest path to the stable-state root is visualised, with red edges denoting the optimal traversal path.

## 2.2.2 Decision Tree

The DT-guided graph utilises the same feature space as the KNN, training a Classification And Regression Tree (CART) regressor [4] on a blended target:

$$
y = 0 . 6 \cdot D P + 0 . 4 \cdot S ,\tag{4}
$$

where S is the structural score as: $\hat { \Delta } ( 0 . 4 4 )$ , power of two of both inputs (0.24), inverse total bit count (0.2), and the minimum or maximum status (0.12). Each node receives a predicted score and leaf ID. Edges are created in two tiers: within-leaf edges connect to higher-scoring neighbours within the same partition, and up-tree edges connect to higher-ranked leaves with a higher mean tree score, consistently directing traversal towards better-scoring nodes. Edge weights combine tree score, structural features, and a Proximity Factor (PF), which penalises long-range edges across the graph. PF is defined as:

$$
P F = { \frac { 1 } { 1 + E D } } .\tag{5}
$$

## 2.2.3 Random Forest

The RF-guided graph uses the same feature space and blended target as the DT model. However, rather than a single tree, RF aggregates the predictions of independent DTs, each trained on a different sample, with the final score taken as the mean prediction across all estimators.

$$
\hat { y } _ { R F } = \frac { 1 } { N _ { e } } \sum _ { t = 1 } ^ { N _ { e } } \hat { y } _ { t } ,\tag{6}
$$

where $N _ { e }$ is the number of estimators and $\hat { y } _ { t }$ is the prediction of the $t ^ { t h }$ tree. A key distinction of the RF model is the ensemble variance tracking, where the per-node prediction variance across all estimators is recorded as a confidence measure as:

$$
\sigma _ { i } ^ { 2 } = \mathrm { V a r } ( \hat { y } _ { 1 , i } , \hat { y } _ { 2 , i } , . . . , \hat { y } _ { N _ { e } , i } ) , \quad C _ { i } = \frac { 1 } { 1 + \sigma _ { i } ^ { 2 } } ,\tag{7}
$$

where $C _ { i }$ is the confidence score for node i, with higher values indicating greater estimator agreement. Leaf assignment for graph edge construction is derived from the first estimator, and the same two-tier within-leaf and up-tree edge creation strategy as DT is applied.

## 3 Results

In previous sections, we described our feature engineering for differentials and the implementation of ML-guided graphs for identifying high-probability differentials. In this section, we present the results of our work, highlighting efficiency gains, differential feature enhancements and the interconnected relationships of differentials. Through a comparative analysis of 3 ML models, we highlight their key advantages and potential limitations.

## 3.1 Graph Analysis

We now provide a comparative analysis of the ML-guided graphs of SIMON and SIMECK, which will help us identify key differences in how models process and represent the pDDT data. The graphs consist of multiple layers of vertices, with the stable-state differential centred and highlighted in yellow. Orange and blue vertices in the inner ring denote high-probability differentials with specific bitwise transition states, such as a Hamming weight of 1 and matching input differences. The graph consists of three edges of different colours. Light grey edges represent background connectivity, typically with longer, suboptimal traversals. The red edges emphasise the critical guiding paths to the stable-state node. The blue edge highlights the shortest path from a randomly selected starting node to the central stable-state node.

![](images/b47b87a55d6a514e495b71484b4b52fe5fee91b3fe29348d26e306061d557184.jpg)  
(a) KNN guided graph of SIMON32  
(b) KNN guided graph of SIMECK32  
Figure 2: KNN guided graphs.

We begin our graph analysis of the KNN guided graph of SIMON32 and SIMECK32, presented in Figures 2a and 2b respectively. Both graphs render 500 nodes under an identical threshold configuration, enabling direct structural comparison between the two $\operatorname { L C A s } .$ The most prominent observation across both KNN figures is the tight spatial clustering of orange nodes, which are differentials that satisfy both $\hat { \Delta } = 1$ and $\Delta _ { X _ { w } } = 1 $ , corresponding to the highest-probability single-bit differentials. In Figure 2b, this cluster forms a dense, geometrically cohesive group at the centre of the graph. In Figure 2a, the same cluster occupies the upper-central region and exhibits a slightly greater dispersion, reflecting the broader spread of high probability differentials in SIMON relative to SIMECK. This geometric separation of high-probability differentials from broader node populations provides graph-based visual confirmation of the differential clustering effect [16].

Beyond the orange cluster, both graphs exhibit a structure in which green nodes form broad peripheral groupings. In both graphs, three distinct lobes are visible. However, SIMECK exhibits tighter cohesion between the two outer lobes than SIMON. These structural contrasts reflect the distinct design differences of the LCAs, with SIMON’s wider lobe distribution suggesting a less centralised differential dependency. Additionally, the stable-state root nodes occupy different positions in both graphs. In SIMECK, the root node is anchored at the top centre, with red paths fanning down across the graph. In SIMON, the root node settles in the upper-right centre, with red paths fanning outward toward prominent differentials. In both figures, KNN consistently produces paths to the root node, regardless of the random node’s starting position. The overall density of the grey edges reflects the lazy learning nature of KNN, in which each node retains connections to multiple neighbours rather than committing to a single branch. While this technique produces a dense graph, the structural hierarchy remains interpretable, with the red critical paths being clearly distinguished from the broader connectivity.

Figures 3a and 3b present the DT-guided graphs for SIMON32 and SIMECK32, respectively. The structural contrast with KNN is immediately apparent. Where KNN produces a fan-shaped topology with a tight peripheral orange cluster and distinct green lobes, the DT guided graph exhibits a dense, approximately circular arrangement in which nodes are distributed more uniformly across the graph interior. The most notable structural difference concerns the placement of orange single-bit differentials. Rather than forming isolated clusters, as in KNN, high-probability nodes with $\hat { \Delta } = 1$ and $\bar { \Delta } _ { X _ { w } } \overset { = } { = } 1$ are distributed throughout the central region, intermixed with the broader node population. This reflect the two-tier edge construction of the DT model, in which within-leaf edges bind structurally similar nodes locally and up-tree edges connect across partitions, pulling high-scoring differentials inward rather than allowing them to aggregate at a spatial periphery. Additionally, the black edges are clearly visible in both figures, a feature absent in the KNN graphs, and provide evidence that the leaf-based partitioning guides the connectivity. The density of red critical-path edges is notably higher in both graphs relative to their KNN counterparts. This indicates that more nodes lie on paths to the root node. The root node is positioned in the left centre of both graphs, with the random starting node located at the far left, near the root, with a recoverable path in both instances. Both DT graphs are also structurally more similar to one another than under KNN guidance, suggesting that the CART regressor’s blended probability-structure target produces a consistent partitioning topology across the two LCAs.

![](images/bde34be320393284d0b8a60a5ffcbc1943c6908b2ddd3be4dea7fa318a4e198b.jpg)  
Figure 3: DT guided graphs.

![](images/2f7cdc55e7d9fba17a07bebbba12e77ecd523b0d382aff1d359579c9d35a3cb8.jpg)  
(a) RF guided graph of SIMON32  
(b) RF guided graph of SIMECK32  
Figure 4: RF guided graphs.

Figures 4a and 4b present the RF-guided graphs for SIMON32 and SIMECK32, respectively. The RF graphs exhibit a topology distinct from both KNN and DT, with several structurally notable features. The most visually prominent characteristic is the emergence of orange single-bit differentials along the left boundary of both graphs, which is spatially separated from the broader node population. This peripheral positioning reflects the ensemble-averaging effect of RF, in which the model consistently assigns high scores to the most structurally favoured differentials, thereby pulling them outward as high-centrality anchor points. Red critical-path edges concentrate heavily around these left-spine nodes, indicating they serve as primary intermediaries on paths to the root.

Table 1: Graph construction and regression metrics
<table><tr><td>LCA</td><td>Model</td><td>Time (s)</td><td>Edges</td><td>R2</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="3">SIMON32</td><td>KNN</td><td>2.40</td><td>12,475</td><td>-6.6182*</td><td>0.1438</td><td>0.1522</td></tr><tr><td>DT</td><td>11.49</td><td>1,961</td><td>0.9974</td><td>0.0003</td><td>0.0040</td></tr><tr><td>RF</td><td>12.02</td><td>1,931</td><td>0.9964</td><td>0.0002</td><td>0.0047</td></tr><tr><td rowspan="3">SIMECK32</td><td>KNN</td><td>2.29</td><td>12,475</td><td>-6.6182*</td><td>0.1438</td><td>0.1522</td></tr><tr><td>DT</td><td>11.39</td><td>1,961</td><td>0.9974</td><td>0.0003</td><td>0.0040</td></tr><tr><td>RF</td><td>11.76</td><td>1,931</td><td>0.9964</td><td>0.0002</td><td>0.0047</td></tr></table>

A secondary orange cluster is visible in the central region of both graphs, interspersed with the green node population, as in the DT graphs. However, a further distinctive feature is the presence of a lone high-probability orange node at the lower periphery in SIMECK, isolated from the main body by several red convergence edges, suggesting that the RF model has assigned this node a particularly high importance whilst placing it outside the primary leaf partitions. Black edges are more dominant in the RF graphs than in DT, reflecting the denser within-leaf and up-tree scaffolding that results from leaf assignments derived from the ensemble scores. The green lower-probability nodes exhibit a more structured, concentric ring pattern around the graph interior than is observed in DT, indicative of the smoother score gradients produced by ensemble averaging. Overall, red critical paths are fewer in number but more spatially localised than in the DT graphs, concentrated on the orange spine and central cluster rather than uniformly distributed, suggesting that RF produces a sparser yet more targeted set of high-confidence differential pathways to the root node.

While the above figures provide insights into the graph’s structure, additional information can be extracted from the corresponding data. As shown, KNN is significantly more efficient than both DT and RF at guiding a graph to the stable state. However, as stated above, it does so by exploring many more paths, which may not be as optimal as the paths identified by DT and RF. Indeed, examining the total number of relationships reveals a more exhaustive search by KNN, in which all nodes are considered potential neighbours. Across all relationships, DT and RF show far fewer optimal paths, demonstrating the rigidity of the decision-making process. While the number of optimal paths of DT and RF is significantly lower than KNN, all models illustrate similar reductions of $\geq 9 0 \%$ from the total number of relationships. However, a notable feature is the model’s ability to identify a path from a random node to the stable state. As shown, KNN reliably identified a path from a random node to the stable state, while both DT and RF fail with smaller samples of the pDDT data. Although the paths identified by KNN typically include more steps over a greater distance, its lazy learning approach facilitates robust pathfinding, reliably guiding a path from a random node to the stable state.

## 3.2 Model Evaluation

While the above figures in Section 3.1 provide insights into the structure of the graph, additional information can be extracted from the corresponding data. Table 1 presents the construction metrics for all three models and LCA configurations. An initial observation is that metrics are identical within each model type across both LCAs, confirming that the graph construction algorithms respond deterministically to the shared pDDT structural properties of SIMON32 and SIMECK32 rather than to algorithm-specific designs. As shown in Figure 5a, KNN constructs graphs approximately five times faster than DT or RF. Figure 5b illustrates that KNN produces 12475 edges compared to approximately 1931 and 1961 for DT and RF models, respectively. This increase in edge density reflects KNN’s lazy learning strategy, in which all K nearest neighbours of each node are retained as potential edges, whereas DT and RF commit only the within-leaf and up-tree connections mandated by the CART partition structure.

For DT and RF, regression performance is near-perfect, with $R ^ { 2 } = 0 . 9 9 7 4$ and 0.9964 respectively, Mean Absolute Error (MAE) below 0.0004 and Root Mean Square Error (RMSE) below 0.005, as shown in Figure 6. The marginally higher RMSE of RF relative to DT (0.0047 versus 0.0040) reflects the smoothing effect of ensemble averaging on the decision boundaries of individual trees. KNN reports a strongly negative $R ^ { 2 }$ because its surrogate score, inverted distance-to-root in the feature embedding, is not a regression prediction in the supervised sense and exhibits high variance relative to the probability distribution. This value is therefore not comparable with the DT and RF results.

## 3.2.1 Classification Metrics

Table 2 presents supplementary classification metrics for the task of identifying single-bit, matching input differentials. A precision of 1.0000 across all models and both LCAs is the most significant result presented. No model produces a false positive, meaning that whenever a node is classified as belonging to the highest-priority single-bit group, that classification is always correct. This is cryptographically meaningful, as all three graphs correctly identify high-priority candidate differentials for attack. As shown in Figure 7, KNN achieves a higher F1 score than DT and RF, driven by higher recall, indicating that KNN’s dense neighbourhood structure recovers a larger proportion of the high-priority differentials within the graph’s top-ranked nodes. DT and RF yield identical classification results, consistent with their shared target function and equivalent leaf-based node prioritisation.

![](images/51a805b69f48bafa0ec4a90d5f8e5eb7b840c7487ca076884c1acedb7998d7e1.jpg)  
(a) Graph creation time

![](images/bd8c843ecaf9069c7557fa500f2038e81a7d3105c890c7927843444ecb7d491d.jpg)  
(b) Total relationships

Figure 5: Model comparison for SIMON32 and SIMECK32  
![](images/16f273843f114aaab5fd3049261282521e498b466cf27843b4bdc86098c0f451.jpg)  
Figure 6: Regression metrics for DT and RF.

## 3.2.2 Clustering Metrics

Table 3 presents the primary cryptographic evaluation metric, which is the mean Dijkstra distance from each node group to the stable-state root, and the derived Cluster Quality (CQ) score. The CQ score quantifies the degree to which the graph separates high-probability single-bit differentials from the broader node population. A higher CQ score indicates that such nodes sit substantially closer to the root than the remaining nodes, providing a graph-structural representation of the differential clustering effect observed in [16].

As illustrated in Figure 8a, KNN achieves the highest CQ score of 1.9221, meaning other nodes are positioned further from the root than high-priority single-bit differentials. DT and RF produce an identical CQ score of 1.5450. This result is counterintuitive given that DT and RF are explicitly optimised for differential priority, while KNN is not. Indeed,

Table 2: Classification metrics for $\hat { \Delta } = 1 , \Delta _ { X _ { w } } = 1$ differential identification
<table><tr><td>LCA</td><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">SIMON32</td><td>KNN</td><td>0.5474</td><td>1.0000</td><td>0.5474</td><td>0.7075</td></tr><tr><td>DT</td><td>0.5048</td><td>1.0000</td><td>0.5048</td><td>0.6709</td></tr><tr><td>RF</td><td>0.5048</td><td>1.0000</td><td>0.5048</td><td>0.6709</td></tr><tr><td rowspan="3">SIMECK32</td><td>KNN</td><td>0.5474</td><td>1.0000</td><td>0.5474</td><td>0.7075</td></tr><tr><td>DT</td><td>0.5048</td><td>1.0000</td><td>0.5048</td><td>0.6709</td></tr><tr><td>RF</td><td>0.5048</td><td>1.0000</td><td>0.5048</td><td>0.6709</td></tr></table>

![](images/97193ca21f0d244bd47099b4adf2c038b5e72c0a7cb213cb397aab5cc8858d41.jpg)  
Figure 7: F1 score

Table 3: Clustering proximity metrics
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Node Group</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Mean dist.</td><td rowspan=1 colspan=1>Std dist.</td><td rowspan=1 colspan=1>CQ Score</td></tr><tr><td rowspan=1 colspan=1>KNN</td><td rowspan=1 colspan=1> $\overline { { \mathrm { R o o t } \left( \alpha = 0 , \beta = 0 \right) } }$  $\hat { \Delta } = 1 , \Delta _ { X w } = 1 , p \ge 0 . 2 5$  $\hat { \Delta } = 1 ( \mathrm { n o n { - } s i n g l e { - } b i t ) }$ Other nodes</td><td rowspan=1 colspan=1>1325074,460</td><td rowspan=1 colspan=1>0.00003.54317.20906.8103</td><td rowspan=1 colspan=1>0.00000.12690.80171.1776</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>DT</td><td rowspan=1 colspan=1> $\overline { { \mathrm { R o o t } \left( \alpha = 0 , \beta = 0 \right) } }$  $\hat { \Delta } = 1 , \Delta _ { X _ { w } } = 1 , p \ge 0 . 2 5$  $\hat { \Delta } = 1 ( \mathrm { n o n { - } s i n g l e { - } b i t ) }$ Other nodes</td><td rowspan=1 colspan=1>1325074,460</td><td rowspan=1 colspan=1>0.00001.57313.15192.4305</td><td rowspan=1 colspan=1>0.00000.15890.41020.5999</td><td rowspan=1 colspan=1>1.5450</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1> $\overline { { \mathrm { R o o t } \left( \alpha = 0 , \beta = 0 \right) } }$  $\hat { \Delta } = 1 , \Delta _ { X _ { w } } = 1 , p \ge 0 . 2 5$  $\hat { \Delta } = 1 ( \mathrm { n o n { - } s i n g l e { - } b i t ) }$ Other nodes</td><td rowspan=1 colspan=1>1325074,460</td><td rowspan=1 colspan=1>0.00001.57313.15192.4305</td><td rowspan=1 colspan=1>0.00000.15890.41020.5999</td><td rowspan=1 colspan=1>1.5450</td></tr></table>

KNN’s neighbourhood structure produces stronger cluster separation in relative terms, albeit at larger absolute distances, as shown in Figure 8b. The tighter intra-cohesion of KNN’s high-priority group further confirms that KNN clusters these differentials more compactly around the root in relative graph-distance terms. Meanwhile, DT and RF produce shorter absolute distances throughout, high-priority nodes sit at a mean distance of 1.5731 from the root compared to 3.5431 for KNN, reflecting the sparser, more directed edge structure of the CART-guided graphs. The tighter absolute topology of DT and RF may be preferable in practice, where minimising $E _ { h o p }$ is the primary objective, while KNN’s stronger relative separation may be more informative for differential identification tasks where distinguishing high-priority from non-priority nodes is the goal. The best differential trail heuristic search results derived from the cluster analysis for all SIMON and SIMECK variants are provided at GitHub <sup>1</sup>, with trails for the largest variants extending beyond previously reported bounds at significantly reduced computational costs [12, 10].

## 4 Discussion

This work shows that ML algorithms can effectively identify and structurally separate high-probability differential clusters in GRL, directly addressing the RQ. By enhancing the pDDT with four engineered features, we reveal hidden structural properties, allowing all three models to generate graphs that focus on the stable-state root and significant differentials. The consistent results across both LCAs indicate a response to shared structural properties rather than specific algorithms. Notably, all three models achieve a precision of 1.0 for the optimal $\hat { \Delta } = 1$ and $\Delta _ { X _ { w } } = 1 $ ensuring no misclassification of differentials. Any node marked as high-priority is guaranteed to be correct, yielding actionable cryptographic insights. The spatial clustering of these nodes visually confirms the differential clustering effect, representing the first such visualisation within a GRL framework.

KNN demonstrates the best cluster separation (CQ = 1.9221, compared to 1.545 for DT and RF) and the highest F1 score (0.7075), along with approximately 5 times faster graph construction. It achieves equivalent differential isolation in $\leq 2 . { }$ 4 seconds on the same hardware as noted in previous studies, showing significant speedup without sacrificing efficiency. While KNN’s neighbourhood structure results in tighter intra-cluster cohesion, DT and RF excel in regression accuracy, producing sparser graphs with shorter Dijkstra distances, making them ideal for minimising hops. Overall, KNN is preferable for differential cluster identification and efficiency, while DT and RF are better for compact, path-optimised graphs.

![](images/117e35c2aa3051fdf39bf56890aea053af29f21c28d0f26c92c423a9158a596c.jpg)  
(a) Cluster quality.

![](images/2617813514e8ae995cd38d826ab55beea8bb59c161abe769a3c1e7b49ed4a772.jpg)  
(b) Dijkstra distance to root by group.  
Figure 8: Clustering proximity and quality.

## 5 Conclusion and Future Research

This paper presents a GRL framework for the differential cryptanalysis of LCAs by enhancing pDDT differentials with four engineered features, revealing previously hidden structural information for supervised ML models. The resulting features informed three graph-construction algorithms, KNN, DT, and RF, that successfully identified stable-state, high-probability differentials for SIMON32 and SIMECK32 without false positives. The clustering of differentials in the graph embeddings represents a novel visualisation of the differential clustering effect for both algorithms. The model comparisons highlight a trade-off between strong cluster separation and graph construction time, with KNN showing better separation but at a higher computational cost for DT and RF. The framework appears generalisable across both SIMON32 and SIMECK32, and future work may expand its evaluation to additional LCAs and larger word sizes, while exploring ways to utilise the identified differential clusters to reduce attack round counts.

## References

[1] Ali, B., Awad, A.I.: Cyber and physical security vulnerability assessment for IoT-based smart homes. sensors 18(3), 817 (2018)

[2] Beaulieu, R., Shors, D., Smith, J., Treatman-Clark, S., Weeks, B., Wingers, L.: The SIMON and SPECK lightweight block ciphers. In: Proceedings of the 52nd annual design automation conference. pp. 1–6 (2015)

[3] Biryukov, A., Velichkov, V.: Automatic search for differential trails in ARX ciphers. In: Topics in Cryptology–CT-RSA 2014: The Cryptographer’s Track at the RSA Conference 2014, San Francisco, CA, USA, February 25-28, 2014. Proceedings. pp. 227–250. Springer (2014)

[4] Bittencourt, H.R., Clarke, R.: Feature selection by using classification and regression trees (CART). The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences (2004)

[5] Celikkanat, A., Malliaros, F.D., Papadopoulos, A.N.: NodeSig: Binary Node Embeddings via Random Walk Diffusion. In: 2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM). pp. 68–75. IEEE (2022)

[6] Cook, J., ur Rehman, S., Khan, M.A.: Cryptanalysis of the SIMON Cypher Using Neo4j. In: 2024 International Conference on Electrical, Computer and Energy Technologies (ICECET). pp. 1–6. IEEE (2024)

[7] Cook, J., ur Rehman, S., Khan, M.A.: Lightweight Cryptanalysis of IoT Encryption Algorithms: Is Quota Sampling the Answer? IEEE Access (2024)

[8] Cook, J., Rehman, S.U., Khan, M.A.: Security and Privacy for Low Power IoT Devices on 5G and Beyond Networks: Challenges and Future Directions. IEEE Access 11, 39295–39317 (2023)

[9] Cook, J., Rehman, S.U., Khan, M.A.: Impact of Differentials in SIMON32 Algorithm for Lightweight Security of Internet of Things. In: 2025 IEEE Globecom Workshops (GC Wkshps). pp. 472–477. IEEE (2025). doi:10.1109/GCWkshps68340.2025.11590866

[10] Cook, J., Rehman, S.U., Khan, M.A.: RAID-Crypt: Rounding Alignment of Intermediate Differentials for Cryptanalysis of Lightweight Encryption. IEEE Access (2026)

[11] De Canniere, C., Biryukov, A., Preneel, B.: An introduction to block cipher cryptanalysis. Proceedings of the IEEE 94(2), 346–356 (2006)

[12] Dwivedi, A.D., Srivastava, G.: Security analysis of lightweight IoT encryption algorithms: SIMON and SIMECK. Internet of Things p. 100677 (2023)

[13] Edgar, T.W., Manz, D.O.: Science and cyber security. Research methods for cyber security pp. 33–62 (2017)

[14] Gohr, A.: Improving attacks on round-reduced speck32/64 using deep learning. In: Advances in Cryptology– CRYPTO 2019: 39th Annual International Cryptology Conference, Santa Barbara, CA, USA, August 18–22, 2019, Proceedings, Part II 39. pp. 150–179. Springer (2019)

[15] Kammari, M., Durga Bhavani, S.: Adaptive Temporal Random Walks for Graph Representation. In: International Conference on Big Data Analytics. pp. 75–91. Springer (2024)

[16] Leurent, G., Pernot, C., Schrottenloher, A.: Clustering effect in Simon and Simeck. In: International Conference on the Theory and Application of Cryptology and Information Security. pp. 272–302. Springer (2021)

[17] Li, H., Appleby, G., Brumar, C.D., Chang, R., Suh, A.: Knowledge graphs in practice: Characterizing their users, challenges, and visualization opportunities. IEEE Transactions on Visualization and Computer Graphics 30(1), 584–594 (2023)

[18] Meneghello, F., Calore, M., Zucchetto, D., Polese, M., Zanella, A.: IoT: Internet of threats? A survey of practical security vulnerabilities in real IoT devices. IEEE Internet of Things Journal 6(5), 8182–8201 (2019)

[19] Page, L., Brin, S., Motwani, R., Winograd, T.: The PageRank citation ranking: Bringing order to the web. Tech. rep., Stanford infolab (1999)

[20] So, J.: Deep learning-based cryptanalysis of lightweight block ciphers. Security and Communication Networks 2020, 1–11 (2020)

[21] Stallings, W.: The principles and practice of cryptography and network security 7th edition, isbn-10: 0134444280. Pearson Education 20(1), 130–131 (2017)

[22] Syafiq, M., Firdaus, A., Aimaan, M., Chuin, L.S., Kamarudin, N.K., Ab Razak, M.F.: FR-VIT: DDoS Detection Using Fruchtermanreingold Layouts and Vision Transformers. In: 2026 IEEE 2nd International Conference on Robotics and Technologies for Industrial Automation (ROBOTHIA). pp. 1–6. IEEE (2026)

[23] Yang, G., Zhu, B., Suder, V., Aagaard, M.D., Gong, G.: The simeck family of lightweight block ciphers. In: International workshop on cryptographic hardware and embedded systems. pp. 307–329. Springer (2015)