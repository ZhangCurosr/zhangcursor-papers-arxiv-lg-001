# Selective Hypergraph Refinement for Frozen Graph Clustering

Zimo Si

Department of Mathematics, Faculty of Science, University of Macau

Macao SAR, China

Email: uc32772@umac.mo

## Abstract

Existing graph-clustering methods typically improve clustering performance by optimizing model parameters and node representations. Efective means of further improving the clustering results of an already trained and frozen model, however, remain limited. We study post-processing for frozen graph clustering. After checkpoint fixation, the procedure uses no labels and updates neither model parameters, node representations nor the original graph structure. Instead, it exploits an attribute hypergraph to supplement higher-order relations that ordinary graphs cannot readily express, thereby refining existing cluster assignments. Because global hypergraph refinement can yield both performance gains and erroneous updates, we propose Selective Hypergraph Refinement (SHR). The method generates candidate residual directions from the hypergraph and evaluates their reliability using graph structure, node attributes and matched-null evidence. It updates only nodes with suficient support and otherwise retains their original assignments. Further analysis shows that whether a node changes cluster is jointly governed by its native assignment gap and the directional strength of the refinement. In a controlled common-suite evaluation, 13 of 15 backbone–dataset cells had a positive mean macro gain, one produced exact no-action, and one was negative. The cell-equal macro gain was 0.066 pp (95% bootstrap CI, [0.030, 0.107] pp), while only 0.209% of hard assignments changed on average. A broader 15-combination nativeinterface evaluation yielded a macro gain of 0.137 pp at a mean change ratio of 0.375%. These results indicate that frozen clustering outputs retain a limited but measurable refinement space after training. The efect is heterogeneous across backbone–dataset pairs, and broader coverage also increases exposure to negative transfer.

Keywords: graph clustering; hypergraph learning; frozen models; label-free post-processing; cluster assignments; selective refinement

## 1 Introduction

Existing graph-clustering methods primarily improve clustering performance by optimizing model parameters, node representations or graph structure[1–7]. However, further improving the clustering output of an already trained and frozen model remains challenging. We therefore study label-free post-processing for frozen graph clustering: whether existing cluster assignments can be improved without updating model parameters, node representations or the original graph structure.

Ordinary graphs primarily describe pairwise relations between nodes, whereas attribute hypergraphs can supplement higher-order associations induced by attribute similarity. For a frozen model, such information can no longer be incorporated through representation learning and can only be used to refine the existing clustering output. Our experiments reveal further scope for improvement through hypergraph refinement, but not every modification is beneficial. Broader refinement coverage can reach more potential errors while also disrupting assignments that were already correct. The key issue is therefore which candidate refinements should be accepted and how far the intervention should extend.

Motivated by this observation, we propose Selective Hypergraph Refinement (SHR). SHR uses an attribute hypergraph to generate candidate residual directions in the native cluster-coordinate system. It then evaluates diferent refinement states without labels by combining graph structure, node attributes and matched-null evidence. Only suficiently supported modifications are retained;

otherwise, the original assignments remain unchanged. We further find that, under a common global refinement strength, diferent nodes change their cluster assignments at diferent strength levels. We therefore define the critical refinement strength $\eta _ { \mathrm { c r i t } }$ as the minimum strength required for a node to change its current cluster assignment for the first time. We evaluate SHR across multiple graph-clustering backbones and datasets. The results show a positive aggregate gain under limited refinement coverage, together with pronounced backbone–dataset heterogeneity. Repair– harm analysis further shows that broader coverage creates more opportunities for repair but also greater exposure to negative transfer. Existing label-free signals, however, do not yet distinguish beneficial from harmful refinements consistently.

The main contributions of this work are as follows:

1. We formulate label-free post-processing for frozen graph clustering. After checkpoint fixation, model parameters, node representations and the original graph structure remain unchanged, and only the existing cluster assignments are refined.

2. We propose Selective Hypergraph Refinement (SHR). The method uses an attribute hypergraph to generate candidate refinements and evaluates refinement states without labels through graph structure, node attributes and matched-null evidence. It retains the original result when the evidence is insuficient.

3. We characterize the relationship between refinement coverage and gain–risk behavior. Critical-strength analysis explains implicit node selection under global refinement. Cross-backbone repair–harm experiments further show that broader coverage provides more opportunities for repair while increasing the risk of negative transfer.

## 2 Related Work

## 2.1 Graph Clustering and Hypergraph Learning

Existing deep graph-clustering methods typically improve clustering performance by jointly optimizing node representations, graph structure or clustering objectives[1–7]. Their efectiveness depends substantially on continued updates to model parameters and node representations during training. Whereas an edge in an ordinary graph links only two nodes, a hyperedge can connect multiple nodes and thereby represent higher-order associations[8]. Hypergraphs have therefore been increasingly used in graph clustering and representation learning. HCN learns clustering representations through hypergraph smoothing and structural alignment[9]. HCN-PAI imputes missing attributes through higher-order propagation and further optimizes clustering results[10]. JKHR jointly learns a consensus kernel and hypergraph regularization to integrate higher-order structural information[11]. Joint graph–hypergraph representations, higher-order encodings and multi-channel hypergraph models have also been used to integrate diferent forms of relational information[12–15]. These studies show that higher-order relations can complement ordinary graph structure, but most require node representations, graph structure, or the clustering space to be relearned during training. By contrast, we consider a graph-clustering backbone that has already been trained and remains frozen. The attribute hypergraph does not participate in new representation learning and serves only as auxiliary information for refining existing cluster assignments.

## 2.2 Post-training Adaptation

For an already trained graph model, test-time adaptation ofers a means of improving performance without repeating the entire training process. Existing graph test-time adaptation methods typically use unlabeled test data to continue adjusting local model parameters, node representations or propagation mechanisms. For example, ASSESS adapts a pretrained model at test time through adaptive subgraph selection and prototype supervision[16]. Other methods address structural shift by adjusting representations on the test graph through self-supervised low-rank feature tuning[17]. Although TOTF adopts a train-once-then-freeze encoder design, it still extracts shared cross-view information and trains an auxiliary graph network after freezing to complete clustering[18]. These approaches reduce the need to retrain the complete model, but still alter model representations, propagation processes or decision boundaries during testing or downstream clustering. SHR imposes stricter constraints: after checkpoint fixation, the backbone, node embeddings and original graph structure remain unchanged, and all subsequent operations act only on the existing cluster assignments.

## 2.3 Relational Post-processing

Another line of work does not continue adapting the base model, but instead uses additional relational information to post-process existing predictions. After fixing the base predictions, Correct and Smooth refines semi-supervised node-classification results through residual propagation from labeled nodes and prediction smoothing[19]. NLCS (Nonlinear Correct and Smooth) further introduces nonlinearity and higher-order relational modeling, allowing residuals to propagate along more complex node relations[20]. Graph structure has also been used in other forms of prediction post-processing. HGPF targets pretrained heterogeneous GNNs and adds an auxiliary relational-reasoning module after base-model training to improve classification[21]. These studies show that additional relational information can further refine the predictions of a base model. They nevertheless focus primarily on supervised or semi-supervised node classification, and some still depend on labels or post-training parameter optimization. SHR instead targets a frozen model that has already completed clustering. It uses no labels after checkpoint fixation and does not create a new clustering space. Rather, an auxiliary hypergraph selectively refines existing cluster assignments, while the original result is retained when the evidence is insuficient.

## 2.4 Uncertainty Modeling and State Averaging

When several candidate models or clustering states are plausible, selecting a single optimum ignores uncertainty arising from model selection. Bayesian model averaging provides a classical means of retaining this uncertainty by weighting multiple candidate models[22]. Related ideas have also been applied to clustering. Cluster-ensemble methods integrate multiple base clusterings through probabilistic or structured models[23]. Dynamic anchor-based methods using hypergraph reconstruction further construct and optimize a consensus structure from base clusterings[24]. clusterBMA weights multiple unsupervised clustering results using approximate posterior model probabilities, yielding a probabilistic final cluster assignment[25]. In parallel, research on safe multi-view clustering has shown that adding a view does not necessarily improve clustering performance and explicitly seeks to prevent performance degradation caused by additional views[26]. This concern is closely related to the gain–risk issue considered here, although these methods still learn new multi-view clustering representations through end-to-end training.

SHR draws on the principle of retaining uncertainty among several plausible candidates rather than constructing a new Bayesian clustering model. For the frozen native soft cluster-assignment matrix $Q _ { 0 }$ , diferent refinement states represent diferent refinement coverages and strengths. SHR assigns relative weights according to the label-free evidence and coverage complexity of each state, while jointly including the no-action state that leaves $Q _ { 0 }$ unchanged. Unlike the Bayesian clustering methods above, which integrate multiple clustering results or re-estimate a consensus clustering, SHR always refines $Q _ { 0 }$ within the native cluster-coordinate system and creates no new clustering space.

## 3 Preliminaries

Definition 3.1 (Unsupervised Graph Clustering). Consider an attributed undirected graph $\mathcal { G } =$ $( V , E , X )$ , where $V = \{ v _ { 1 } , \ldots , v _ { N } \}$ is the node set and E is the edge set, while $X \in \mathbb { R } ^ { \bar { N } \times \bar { F } }$ is the node-attribute matrix and $A \in \{ 0 , 1 \} ^ { N \times N }$ is the corresponding adjacency matrix. Unsupervised graph clustering aims to partition the N nodes into K clusters based on graph structure and node attributes, without using node labels.

For an already trained graph-clustering model, denote its frozen node representation by

$$
Z _ { 0 } \in \mathbb { R } ^ { N \times d } ,\tag{1}
$$

and its soft cluster-assignment matrix by

$$
Q _ { 0 } \in [ 0 , 1 ] ^ { N \times K } , \qquad \sum _ { k = 1 } ^ { K } Q _ { 0 , i k } = 1 .\tag{2}
$$

Here, $Q _ { 0 , i k }$ is the soft assignment of node $v _ { i }$ to cluster k. The current hard cluster assignment of node $v _ { i }$ is

$$
a _ { i } = \arg \operatorname* { m a x } _ { k } Q _ { 0 , i k } .\tag{3}
$$

Definition 3.2 (Attribute Hypergraph). Unlike an edge in an ordinary graph, which connects only two nodes, a hyperedge can connect multiple nodes simultaneously. We define the attribute hypergraph as

$$
\mathcal { H } = ( V , \mathcal { E } _ { h } ) ,\tag{4}
$$

where $\mathcal { E } _ { h } = \{ e _ { 1 } , . . . , e _ { M } \}$ is the hyperedge set. Its node–hyperedge incidence matrix is

$$
B \in \{ 0 , 1 \} ^ { N \times M } , \qquad B _ { i j } = \left\{ 1 , \quad v _ { i } \in e _ { j } , \right.\tag{5}
$$

Following the standard normalized form of hypergraph propagation[8], let $W$ denote the hyperedge-weight matrix, and let $D _ { v }$ and $D _ { e }$ denote the node-degree and hyperedge-degree matrices, respectively. The hypergraph propagation operator is

$$
P _ { h } = D _ { v } ^ { - 1 / 2 } B W D _ { e } ^ { - 1 } B ^ { \top } D _ { v } ^ { - 1 / 2 } .\tag{6}
$$

The node and hyperedge degrees are respectively given by

$$
d ( v _ { i } ) = \sum _ { j = 1 } ^ { M } W _ { j j } B _ { i j } , \qquad \delta ( e _ { j } ) = \sum _ { i = 1 } ^ { N } B _ { i j }
$$

with $D _ { v } = \operatorname { d i a g } ( d ( v _ { 1 } ) , \ldots , d ( v _ { N } ) )$ and $D _ { e } = \mathrm { d i a g } ( \delta ( e _ { 1 } ) , \dots , \delta ( e _ { M } ) )$ . Our implementation assigns equal weight to all hyperedges, and hence $W = I$

We construct the auxiliary hypergraph from node attributes and use $P _ { h }$ to aggregate higher-order relations in the attribute space. This hypergraph is used only as auxiliary information for refining the frozen backbone output and neither replaces nor modifies the original graph structure.

Definition 3.3 (Frozen Graph-Clustering Refinement). We consider an already trained and frozen graph-clustering model. The available quantities are the original adjacency matrix $A ,$ node attributes $X$ , the frozen node representation $Z _ { 0 } ,$ the native soft cluster-assignment matrix $Q _ { 0 }$ and the number of clusters $K$ . The goal is to obtain a refined cluster-assignment matrix without using node labels after checkpoint fixation or updating model parameters, node representations, or the original graph structure:

$$
\begin{array} { r l r } { Q ^ { * } = \mathcal { R } ( A , X , Z _ { 0 } , Q _ { 0 } , K ) , } & { { } } & { Q ^ { * } \in [ 0 , 1 ] ^ { N \times K } . } \end{array}\tag{7}
$$

The refinement acts only on the existing cluster assignments; therefore, $Q ^ { * }$ and $Q _ { 0 }$ share the same K-dimensional cluster-coordinate system. When label-free evidence is insuficient, SHR simply retains the original result:

$$
Q ^ { * } = Q _ { 0 } .\tag{8}
$$

Table 1: Principal notation and definitions used in this work.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $\mathcal { G } = ( V , E , X )$ </td><td>Input graph with node attributes</td></tr><tr><td> $N$ </td><td>Number of nodes</td></tr><tr><td> $F$ </td><td>Dimensionality of node attributes</td></tr><tr><td>A</td><td>Adjacency matrix of the original graph</td></tr><tr><td> $X$ </td><td>Node-attribute matrix</td></tr><tr><td> $K$ </td><td>Number of clusters</td></tr><tr><td> $Z _ { 0 }$ </td><td>Node representation output by the frozen graph-clustering backbone</td></tr><tr><td> $Q _ { 0 }$ </td><td>Native soft cluster-assignment matrix output by the frozen backbone</td></tr><tr><td> $a _ { i }$ </td><td>Current hard cluster assignment of node  $v _ { i }$  under  $Q _ { 0 }$ </td></tr><tr><td> $\mathcal { H }$ </td><td>Auxiliary hypergraph constructed from node attributes</td></tr><tr><td> $\mathcal { E } _ { h }$ </td><td>Hyperedge set</td></tr><tr><td> $B$ </td><td>Node-hyperedge incidence matrix</td></tr><tr><td> $P _ { h }$ </td><td>Normalized hypergraph propagation operator</td></tr><tr><td> $\Delta$ </td><td>Candidate refinement residual mapped into the native cluster-coordinate system and formally defined in Section 4.2</td></tr><tr><td>η  $Q ^ { * }$ </td><td>Candidate refinement strength</td></tr><tr><td></td><td>Refined soft cluster-assignment matrix output by SHR</td></tr></table>

## 4 Methods

## 4.1 Method Overview

Under the setting in Section 3, SHR takes the original graph A, node attributes X, frozen representation $Z _ { 0 }$ , native soft assignments $Q _ { 0 } .$ , and the number of clusters K as input. It returns refined assignments $Q ^ { * }$ in the same cluster-coordinate system as $Q _ { 0 }$ . The backbone parameters, $Z _ { 0 } .$ , and the original graph structure remain fixed throughout the procedure.

Figure 1 separates candidate proposal from selective execution. The attribute hypergraph is used to construct a structured candidate residual $\Delta$ in the native cluster-logit coordinates. The proposal defines the available direction of change but does not by itself alter any assignment. SHR evaluates the proposal at the state, node, and run levels. At the state level, topology and attribute evidence, matched-null calibration, and coverage complexity assign relative support to diferent intervention extents. At the node level, a lower confidence bound removes changes whose support is unstable across repeated evidence estimates. At the run level, an attribute-provenance veto returns the complete output to $Q _ { 0 }$ when the retained changes lack aggregate attribute support. Global shrinkage, change-rate control, and cluster-survival constraints limit the extent of the final update. SHR therefore uses label-free evidence to determine which hypergraph-generated changes are retained. The final update remains an additive correction in the native clustering logits and requires neither backbone retraining nor a transport plan.

## 4.2 Hypergraph-Guided Candidate Proposal

Attribute-hypergraph context. The original attributes X are typically high-dimensional and sparse. SHR first applies truncated SVD and per-dimension standardization to obtain a low-rank attribute representation $Z _ { x }$ , and then searches for neighbors by cosine similarity. Each node and its $k _ { h } = 1 0$ attribute neighbors form a candidate hyperedge, and duplicate hyperedges are removed. Each hyperedge therefore represents a group of nodes that are similar in the attribute space. Its contribution is examined in the source ablation in Section 5.3.

Using the hypergraph propagation operator $P _ { h }$ defined in Section $^ { 3 , }$ the node attributes and the outputs after one and two propagation steps are concatenated as

$$
Z _ { h } = \mathrm { S t d } \left( [ Z _ { x } \vert \vert P _ { h } Z _ { x } \vert \vert P _ { h } ^ { 2 } Z _ { x } ] \right) .\tag{9}
$$

![](images/46462a2eaa191e75def082ff48dd2f8ae342f0bde8d52e87218a584965410888.jpg)  
Figure 1: Overview of SHR. The frozen backbone provides $Z _ { 0 }$ and $Q _ { 0 }$ , and the attribute hypergraph generates a candidate residual. Graph structure, node attributes, and matched-null evidence are used to evaluate the candidate states. The method ultimately updates only a subset of cluster assignments, while the model parameters, $Z _ { 0 } .$ , and original graph structure remain frozen.

Here, Std(·) denotes column-wise standardization and ∥ denotes feature concatenation. $Z _ { x }$ retains the node’s own attributes, while $P _ { h } Z _ { x }$ and $P _ { h } ^ { 2 } Z _ { x }$ introduce higher-order context over diferent ranges. The concatenated representation is standardized to prevent one propagation scale from dominating the subsequent projection.

To extract cluster-level structure from $Z _ { h }$ , SHR applies a fixed, label-free $K \cdot$ -means configuration with K clusters. Let $\mu _ { k }$ denote the kth cluster center and $D _ { i , k }$ the squared distance from node $v _ { i }$ to that center. A common softening scale s is defined by the median, across nodes, of the distance to the nearest center:

$$
\begin{array} { r l } & { D _ { i , k } = \Vert Z _ { h , i } - \mu _ { k } \Vert _ { 2 } ^ { 2 } , } \\ & { \quad s = \operatorname* { m a x } \left\{ \operatorname* { m e d i a n } \underset { k } { \operatorname* { m i n } } D _ { i , k } , 1 0 ^ { - 6 } \right\} . } \end{array}\tag{10}
$$

The distances are then converted into auxiliary soft assignments:

$$
\widetilde { Q } _ { h , i , k } = \frac { \exp \left[ - ( D _ { i , k } - \operatorname* { m i n } _ { j } D _ { i , j } ) / s \right] } { \sum _ { \ell = 1 } ^ { K } \exp \left[ - ( D _ { i , \ell } - \operatorname* { m i n } _ { j } D _ { i , j } ) / s \right] } .\tag{11}
$$

$\widetilde { Q } _ { h , i , k }$ gives the soft membership of node $v _ { i }$ in the kth auxiliary K-means cluster. Its columns have not yet been aligned with the semantics of $Q _ { 0 }$

Because the cluster indices produced by K-means are permutation-invariant, let ${ \mathfrak { S } } _ { K }$ denote all permutations of the K cluster indices. SHR obtains a column permutation by maximizing the soft overlap between ${ \widetilde { Q } } _ { h }$ and $Q _ { 0 }$ :

$$
\begin{array} { r } { \pi ^ { * } = \arg \underset { \pi \in \mathfrak { S } _ { K } } { \operatorname* { m a x } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \widetilde { Q } _ { h , i , k } Q _ { 0 , i , \pi ( k ) } , } \\ { Q _ { h , i , \pi ^ { * } ( k ) } = \widetilde { Q } _ { h , i , k } . \qquad } \end{array}\tag{12}
$$

This Hungarian matching removes only the cluster-ID permutation and does not use node labels. After alignment, $Q _ { h }$ provides a cluster-level summary of the hypergraph representation. We concatenate it with $Z _ { h }$ as

$$
H _ { h } = [ Z _ { h } \vert \vert Q _ { h } ] .\tag{13}
$$

$Q _ { h }$ contributes only to constructing the candidate residual and is not used as the final clustering output.

Representation-space alignment. $H _ { h }$ and the frozen representation $Z _ { 0 }$ have no direct correspondence in either dimensionality or coordinate axes. They are first projected to the common dimension

$$
d _ { c } = \operatorname* { m i n } \left\{ 3 2 , \operatorname { c o l } ( Z _ { 0 } ) , \operatorname { c o l } ( H _ { h } ) \right\} ,
$$

where $\operatorname { c o l } ( \cdot )$ denotes the number of matrix columns. After per-dimension centering and scaling, the projected representations are denoted by the frozen anchor coordinates $H _ { 0 }$ and the hypergraph coordinates ${ \widetilde { H } } _ { h }$ , respectively.

We then solve the orthogonal Procrustes alignment

$$
R ^ { * } = \arg \operatorname* { m i n } _ { R ^ { \top } R = I } \left\| \widetilde H _ { h } R - H _ { 0 } \right\| _ { F } ^ { 2 } , \qquad H _ { h } ^ { \| } = \widetilde H _ { h } R ^ { * } .\tag{14}
$$

Here, R is an orthogonal matrix, I is the identity matrix of the corresponding dimension, and $\| \cdot \| _ { F }$ denotes the Frobenius norm. The orthogonality constraint restricts the alignment to rotations and reflections, avoiding additional scaling or nonlinear transformations.

Mapping to the native clustering coordinates. Attribute similarity need not agree with local relationships in the original graph. Let

$$
{ \widetilde { A } } = \operatorname { R o w N o r m } ( A + I ) ,
$$

where RowNorm(·) denotes row normalization and I is the node-dimensional identity matrix. The aligned hypergraph context is combined with its neighborhood aggregation on the original graph in a fixed ratio:

$$
C _ { h } = \frac { 1 } { 2 } H _ { h } ^ { \parallel } + \frac { 1 } { 2 } { \widetilde { A } } H _ { h } ^ { \parallel } .\tag{15}
$$

$C _ { h }$ therefore combines higher-order attribute context with local graph information.

$C _ { h }$ remains in a representation space, whereas the refinement must act on the K cluster logits of $Q _ { 0 }$ . SHR therefore fits a ridge decoder between the frozen anchor coordinates $H _ { 0 }$ and the logit representation $\log ( Q _ { 0 } + \epsilon )$ :

$$
W _ { d } = ( H _ { 0 } ^ { \top } H _ { 0 } + \lambda I ) ^ { - 1 } H _ { 0 } ^ { \top } \log ( Q _ { 0 } + \epsilon ) .\tag{16}
$$

Here, $\lambda > 0$ is the ridge regularization coeficient, $\epsilon > 0$ is a numerical-stability constant that prevents log 0, and $\log ( \cdot )$ is applied elementwise. $W _ { d }$ maps the frozen representation coordinates to the current cluster-logit space.

Passing $C _ { h }$ through the same decoder yields

$$
\Delta = C _ { h } W _ { d } .\tag{17}
$$

Here, $\Delta \in \mathbb { R } ^ { N \times K }$ , and $\Delta _ { i , k }$ is the candidate increment for node $v _ { i }$ along the kth existing cluster logit. Applying the same decoder to $C _ { h }$ expresses the auxiliary context in the native cluster-logit coordinates of $Q _ { 0 }$

The resulting $\Delta$ is the fixed candidate direction used in the following stages and remains expressed in the native cluster-logit coordinates. No assignment is changed at this point, and the hypergraph does not determine the final intervention. Sections 4.3–4.5 use $\Delta$ to define an ordered set of candidate states; label-free evidence then determines the supported intervention extent, the rows that retain the proposed change, and whether the complete run reverts to $Q _ { 0 }$ . The subsequent label-free stage therefore operates on a fixed set of candidate states rather than optimizing an unrestricted $N \times K$ assignment update. The SVD dimension, K-means configuration, numerical safeguards, and ridge parameters are specified in Appendix A.2.

## 4.3 Critical Strength and an Efect-Calibrated Candidate Path

Given the candidate direction $\Delta$ , a scalar $\eta \geq 0$ controls the global refinement strength. The candidate soft assignment for node $v _ { i }$ is

$$
Q _ { i } ( \eta ) = \mathrm { s o f t m a x } \left( \log ( Q _ { 0 , i } + \epsilon ) + \eta \Delta _ { i } \right) .\tag{18}
$$

Here, $Q _ { 0 , i }$ and $\Delta _ { i }$ are the ith rows of $Q _ { 0 }$ and $\Delta ,$ respectively. Setting $\eta = 0$ recovers the original assignment; the softmax ensures nonnegative rows that sum to one. All nodes share the same $\eta ,$ but each node has its own direction $\Delta _ { i }$

Let $a _ { i } = \arg \operatorname* { m a x } _ { k } Q _ { 0 , i , k }$ denote the current assignment of node $v _ { i }$ . For any candidate cluster $b \neq a _ { i }$ , define

$$
\begin{array} { r l r } { \left. { m _ { i , b } = \log Q _ { 0 , i , a _ { i } } - \log Q _ { 0 , i , b } , } \right.} \\ & { { } \quad d _ { i , b } = \Delta _ { i , b } - \Delta _ { i , a _ { i } } , } \\ & { { } \quad \eta _ { i  b } = \frac { m _ { i , b } } { d _ { i , b } } , \quad } & { d _ { i , b } > 0 . } \end{array}\tag{19}
$$

$m _ { i , b }$ is the native logit margin of the current cluster over cluster $b ,$ and $d _ { i , b }$ is the residual push toward cluster b relative to the current cluster. Equating the two refined logits gives the critical value $\eta _ { i  b } ~ = ~ m _ { i , b } / d _ { i , b }$ . If $d _ { i , b } ~ \leq ~ 0$ , cluster b cannot reach the score of the current cluster as η increases.

Define

$$
B _ { i } = \{ b \neq a _ { i } : d _ { i , b } > 0 \}
$$

as the set of candidate clusters that can catch up with the current cluster. The strength at which node $v _ { i }$ first reaches any assignment boundary is

$$
\eta _ { \mathrm { c r i t } } ( i ) = \left\{ \begin{array} { l l } { \displaystyle \operatorname* { m i n } _ { b \in \mathcal { B } _ { i } } \frac { m _ { i , b } } { d _ { i , b } } , } & { B _ { i } \neq \emptyset , } \\ { + \infty , } & { B _ { i } = \emptyset , } \end{array} \right.\tag{20}
$$

When $\boldsymbol { B } _ { i } \neq \boldsymbol { \mathcal { O } }$ , the corresponding first target cluster is

$$
b _ { \mathrm { c r i t } } ( i ) = \arg \operatorname* { m i n } _ { b \in { \mathcal { B } } _ { i } } \frac { m _ { i , b } } { d _ { i , b } } .\tag{21}
$$

A smaller native margin or a stronger relative push lowers $\eta _ { \mathrm { c r i t } } ( i )$ . Under a common global η, nodes with smaller critical strengths cross their assignment boundaries earlier. The boundary $\eta = \eta _ { \mathrm { c r i t } } ( i )$ corresponds to tied logits; a strict hard-assignment change occurs only after the boundary is crossed.

The margins of $Q _ { 0 }$ and the residual scale difer across backbones, so the same numerical η is not comparable across systems. SHR uses seven target hard-assignment change ratios—0.5%, 1%, 2%, 3%, 5%, 8%, and 12%—and selects, from a frozen η grid, the state closest to each target coverage. The grid and selection rule are specified in Appendix A.2.

The critical strengths induce an ordering of candidate assignment changes along the fixed residual. $\eta _ { \mathrm { c r i t } }$ determines when each node reaches an assignment boundary. The subsequent coverage calibration allows candidate states to be compared across backbones. It provides no evidence that the change is beneficial. The states $Q ^ { ( \eta _ { r } ) }$ are therefore treated only as proposals and are compared with the unchanged state using label-free evidence in the next subsection.

## 4.4 Evidence-Driven State Selection

The efect-calibrated path from Section 4.3 contains seven nonzero candidate states that difer in intervention extent. Their order reflects how readily nodes respond to the fixed residual, not whether the resulting changes are beneficial. SHR therefore treats every $Q ^ { ( \eta _ { r } ) }$ as a proposal and compares it with the unchanged output $Q ^ { ( 0 ) } = Q _ { 0 }$ using label-free evidence. Let S denote the index set of the nonzero states. For $\boldsymbol { r } \in \boldsymbol { S }$ , define

$$
\mathcal { C } _ { r } = \left\{ i : \arg \operatorname* { m a x } _ { k } Q _ { i , k } ^ { ( \eta _ { r } ) } \neq a _ { i } \right\} , \qquad c _ { r } = | \mathcal { C } _ { r } | ,
$$

where $\mathcal { C } _ { r }$ is the set of nodes whose hard assignments change in state r, and $c _ { r } = | \mathcal { C } _ { r } |$ is its intervention size.

![](images/4a2478cdf408c0c0d6ea7cdf40225113f528a21f734d92cb2753c816e844ddaf.jpg)  
Figure 2: Candidate-residual construction and critical refinement strength. (a) The multiscale attribute context is summarized by auxiliary clusters, aligned with the frozen representation, combined with local graph context, and decoded through the frozen readout to obtain $\Delta ;$ no backbone parameter is updated. (b) After $\Delta$ is fixed, the logits of the current and candidate clusters vary with $\eta$ and intersect at $\eta _ { \mathrm { c r i t } }$

We evaluate each state using topology and attribute evidence. Topology evidence asks whether the candidate cluster is more compatible than the current cluster with the node’s neighborhood pattern in the original graph. Attribute evidence measures the same candidate-versus-current compatibility in the low-rank attribute space. Both are candidate-versus-current scores estimated through held-out folds. The node utility also subtracts a Jensen–Shannon divergence penalty [27] to discourage large changes in the soft assignments. The exact predictive scores and utility are given in Appendix A.1.

High internal support alone is not suficient, because low-confidence nodes and broad states can receive favorable scores even when the node–residual correspondence is uninformative. SHR therefore constructs matched-null controls by permuting the rows of $\Delta$ within the same baseline cluster and confidence quintile, followed by recalibration to the same target coverage. The permutation approximately preserves node dificulty and intervention size while removing the original node– direction correspondence. The empirical odds $e _ { r }$ summarize whether the observed correspondence receives more support than its matched controls. They are relative evidence scores, not probabilities of correctness or Bayes factors.

Broader states have more operational freedom and can obtain favorable aggregate support by modifying more nodes. SHR accounts for this diference through the coverage complexity $L ( c _ { r } )$ which combines a sparse count prior and an MDL penalty. The coverage term therefore penalizes states that modify more nodes. The unchanged output remains an explicit competing state rather than a fallback added after selection. The log weights are

$$
\ell _ { 0 } = L ( 0 ) ,
$$

$$
\ell _ { r } = \log e _ { r } + L ( c _ { r } ) - \log | S | , \quad r \in S .\tag{22}
$$

Here, $| S |$ is the number of nonzero proposals, and the term − log |S| controls the total weight introduced by considering several nonzero states.

For $j \in \{ 0 \} \cup \mathcal { S }$ , the normalized weights are defined as

$$
\omega _ { j } = \frac { \exp ( \ell _ { j } ) } { \exp ( \ell _ { 0 } ) + \sum _ { s \in \mathcal { S } } \exp ( \ell _ { s } ) } .\tag{23}
$$

The mixed candidate is then

$$
Q _ { \mathrm { m i x } } = \omega _ { 0 } Q _ { 0 } + \sum _ { r \in S } \omega _ { r } Q ^ { ( \eta _ { r } ) } .\tag{24}
$$

The normalized weights $\omega _ { j }$ encode relative label-free support and are not posterior probabilities.   
Including $Q _ { 0 }$ allows SHR to select no action when the nonzero proposals are weakly supported.

State averaging also reduces abrupt coverage changes caused by small diferences in evidence. $Q _ { \mathrm { m i x } }$ summarizes the state-level evidence and is passed to the node-level filtering stage rather than used directly as the final output. Section 4.5 applies node-level evidence filtering and the run-level noaction decision.

## 4.5 Node- and Run-Level Selective Execution

State-level weighting determines the overall intervention extent, but individual node updates may still be unreliable. Before producing the final output, SHR first limits the global update and then filters the remaining node changes. It first constructs the fixed shrinkage path

$$
Q _ { \tau } = \mathrm { N o r m a l i z e } \left( ( 1 - \tau ) Q _ { 0 } + \tau Q _ { \mathrm { m i x } } \right) .\tag{25}
$$

Here, $\tau \in [ 0 , 1 ]$ is the shrinkage coeficient, and Normalize(·) denotes row-wise normalization. The path is evaluated from larger to smaller values of $\tau .$ . For each candidate, SHR checks the hardassignment change ratio, mean Jensen–Shannon divergence, and number of active clusters. They constrain the magnitude and structural validity of the update. The first candidate that satisfies all three conditions is denoted by $Q _ { \mathrm { s a f e } }$

SHR then considers only nodes whose hard assignments change under $Q _ { \mathrm { s a f e } }$ . For node $v _ { i }$ , let $\bar { u } _ { i }$ and $s _ { i }$ denote the mean and standard deviation of its support across the repeated label-free evidence calculations. The node-level lower confidence bound is

$$
\mathrm { L C B } _ { i } = \bar { u } _ { i } - 1 . 6 4 5 \frac { s _ { i } } { \sqrt { R } } .\tag{26}
$$

Here, $R = 5$ is the number of evidence repetitions, and 1.645 is the one-sided 95% normal quantile used by the frozen protocol. A positive LCB indicates that the node retains positive support after accounting for variation across evidence repeats. Nodes with a nonpositive LCB revert to $Q _ { 0 , i }$ Cluster survival is applied afterward as a structural constraint so that no original cluster loses all of its nodes.

Let $\mathcal { A } _ { \mathrm { a c c } }$ denote the nodes that pass the LCB and cluster-survival checks. Finally, the retained nodes must also receive positive mean attribute support at the run level. If $\mathcal { A } _ { \mathrm { a c c } }$ is empty or its mean attribute support is nonpositive, the attribute-provenance veto returns the complete output to $Q _ { 0 }$ . Otherwise, only the accepted rows are retained:

$$
\begin{array} { r } { Q _ { i } ^ { * } = \left\{ \begin{array} { l l } { Q _ { \mathrm { s a f e } , i } , } & { i \in \mathcal { A } _ { \mathrm { a c c } } , } \\ { Q _ { 0 , i } , } & { i \notin \mathcal { A } _ { \mathrm { a c c } } . } \end{array} \right. } \end{array}\tag{27}
$$

If no node passes the filtering stage, or if the run-level attribute provenance check fails, the entire run returns

$$
Q ^ { * } = Q _ { 0 } .
$$

Thus, state-level weighting determines the intervention extent, node-level LCB filtering retains supported rows, and the provenance check can still return the entire run to $Q _ { 0 }$ . Shrinkage and clustersurvival checks enforce feasibility. These controls restrict the intervention but do not guarantee an improvement in external clustering metrics. The fixed projection path, thresholds, and other execution constants are reported in Appendix A.2.

## 5 Experiments

We evaluate SHR from four perspectives: overall clustering performance, the contribution of its main components, the mechanism of selective refinement, and robustness across frozen backbones and datasets. The experiments address the following questions:

Q1: Can SHR improve existing cluster assignments across diferent frozen graph-clustering backbones and datasets, and is its performance consistent across backbones?

Q2: Do the key components of SHR and the higher-order information supplied by the attribute hypergraph contribute to the final refinement results?

Q3: How do refinement strength and refinement coverage afect node changes, clustering gains, and negative transfer, and how does the selective-refinement mechanism of SHR operate?

Q4: Are the aggregate conclusions for SHR stable across backbones, aggregation schemes, and boundary conditions, and what are its principal applicability limitations?

## 5.1 Experimental Setup

Datasets and Backbones. We evaluated SHR on five frozen graph-clustering backbones, DeSE, HALO, DGAC, SynC, and DGM, across attributed-graph datasets including Cora, Citeseer, Photo, ACM, UAT, EAT, BAT, DBLP, and Adam. Multiple backbones are included to test whether SHR can operate on diferent frozen clustering outputs. SHR reads the frozen node representations and native cluster assignments produced by each backbone without retraining the backbone. The broader native-interface evaluation comprises 15 backbone–dataset combinations with valid native cluster assignments. The frozen-embedding KMeans readout for DeSE–Computers and cases that do not satisfy the interface requirements are included only in supplementary analyses and are excluded from that aggregate. Table 2 summarizes the evaluation scope, and Appendix C provides dataset statistics, preprocessing details, and checkpoint provenance.

Controlled and Broader Evaluation Protocols. We use two complementary evaluation protocols. First, we constructed a complete controlled common suite comprising HALO, DGAC, and SynC on Cora, Citeseer, ACM, UAT, and DBLP. Each backbone–dataset cell contains five paired frozen runs, giving 75 experimental units. All three backbones provide a valid native $Q _ { 0 }$ on all five datasets; inclusion was determined by frozen-interface validity rather than SHR performance. Second, we retained a broader native-interface evaluation covering DeSE, HALO, DGAC, SynC, and DGM, with 15 valid backbone–dataset combinations and 113 frozen units. The broader evaluation covers more backbone-native datasets, although the dataset sets and frozen-unit counts difer across backbones. It is used to examine cross-system heterogeneity and robustness, while the controlled common suite provides the direct paired comparison. The preregistered $4 \times 5$ matrix and its DeSE interface failures are documented in Appendix C.3.

Table 2: Backbones, datasets, and their evaluation scope. Broader combinations enter the 15-combination native-interface evaluation; Development and Boundary combinations are used only for supplementary analyses.
<table><tr><td>Backbone</td><td>Dataset</td><td>Evaluation scope</td><td>Notes</td></tr><tr><td>DeSE[1]</td><td>Cora, Citeseer, Photo</td><td>Broader</td><td>Native assignment; label-free after checkpoint fixation</td></tr><tr><td>HALO[2]</td><td>ACM, UAT, EAT, BAT</td><td>Broader</td><td>Broader native-interface evaluation</td></tr><tr><td>DGAC[3]</td><td>Cora, Citeseer</td><td>Broader</td><td>Includes valid no-action</td></tr><tr><td>SynC[4]</td><td>Cora, Citeseer, ACM, UAT, DBLP</td><td>Broader</td><td>Includes negative cases</td></tr><tr><td>DGM[5]</td><td>Adam</td><td>Broader</td><td>Native near-no-action case</td></tr><tr><td>DBCD[6]</td><td>Cora</td><td>Boundary</td><td>Interface boundary</td></tr><tr><td>DeSE strict-final</td><td>Photo, Computers</td><td>Boundary</td><td>Degenerated-anchor boundary</td></tr><tr><td>DeSE legacy</td><td>Computers</td><td>Development</td><td>Development-only non-native readout</td></tr></table>

Comparison Methods. All comparison methods share the same frozen assignments $Q _ { 0 }$ and candidate residual $\Delta .$ allowing the efects of refinement strength, coverage, and node-level selection to be compared under a common proposal. Global-A is the simplest global baseline and applies a fixed refinement strength to every node. Strength-Bayes replaces this single strength with a weighted set of refinement-strength states to examine the efect of strength uncertainty. CS-BAYES further considers refinement coverage and strength jointly, whereas SHR adds node-level selection after state assessment. CM-Global matches the changed-node count of CS-BAYES and is used only to diagnose the efect of broader coverage. The exact state spaces, priors, and computational rules are reported in Appendix D.2 (Table 12).

Evaluation Metrics and Protocol. We evaluated final clustering quality using ACC, NMI, ARI, and macro-F1; ACC and macro-F1 were computed after Hungarian matching. For each metric, the change relative to the frozen baseline is reported in percentage points. Macro gain is the equally weighted mean of the four metric gains, and Change is the proportion of nodes whose final hard assignments difer from the frozen baseline. In addition to the final clustering metrics, we used Repair Recall and Harm Fraction for node-level post hoc diagnosis. They describe, respectively, the extent to which repairable errors in the fixed candidate space are ultimately corrected and the proportion of changed nodes that move from baseline-correct to incorrect assignments. Both are post-hoc diagnostics computed after the SHR outputs are frozen; their definitions are given in Appendix B. Common-suite cell-level summaries report the mean, SD, median, range, and a descriptive 95% interval from 10,000 paired seed-level bootstrap resamples. The aggregate common-suite estimator first forms 15 cell means and then uses the cells as sampling units in 10,000 bootstrap resamples, with random seed 4102026. The broader native-interface evaluation uses combination-equal aggregation: frozen runs are first aggregated within each backbone–dataset combination and the 15 combinationlevel results are then averaged equally. Backbone-equal, run-weighted, and leave-one-backbone-out summaries are used only for robustness assessment. To describe negative transfer, a single run with macro gain $g _ { r } \ < \ 0$ was defined as a negative run; a more pronounced negative-tail outcome with $g _ { r } \le - 0 . 2 5 \mathrm { p p }$ was defined as a catastrophic run; and a combination whose mean macro gain was below zero was defined as a negative combination. Runtime measurement begins after A, X, $Z _ { 0 } ,$ , and $Q _ { 0 }$ have been prepared and ends when formal SHR returns $Q ^ { * }$ . Each frozen unit receives one warm-up and three repetitions timed with time.perf counter(), whose median is reported; backbone training is excluded. The SHR refinement stage was label-free after checkpoint fixation. The DeSE paper-best checkpoints were selected upstream using ground-truth labels, so the label-free claim does not extend to backbone training or checkpoint selection.

Protocol summary. The controlled common suite contains 75 frozen units, while the broader native-interface evaluation contains 113. Separate historical and mechanism registries are used only for component and post-hoc diagnostic analyses. Results from these registries are reported separately and are not pooled.

## 5.2 Overall Performance

Controlled common-suite performance. Table 3 reports the absolute ACC, NMI, ARI, and F1 before and after SHR for each frozen backbone. The comparison is paired within each backbone rather than across diferent backbone models.

The paired results show small positive changes in most backbone–dataset cells, but not improvement in every case. Thirteen of the 15 cells had a positive mean macro gain, DGAC–Citeseer retained exact no-action, and SynC–UAT was negative. Equally weighting the 15 cell means yielded a macro gain of 0.066 pp with a 95% combination-bootstrap CI of [0.030, 0.107] pp; the cell median was 0.050 pp, the IQR was [0.006, 0.105] pp, and the range was [−0.042, 0.224] pp. The common suite therefore shows a small overall positive efect, with clear variation across backbone–dataset pairs. Complete seed-level dispersion and cell-level bootstrap intervals are reported in Appendix D.1.

Table 3: Controlled common-suite results before and after SHR. For each frozen backbone, the baseline and its SHR-refined output are reported consecutively on the same five datasets. Each entry is averaged over five paired frozen runs. ACC, NMI, ARI, and F1 are reported on the [0, 1] scale. Boldface indicates the better value within each backbone–SHR pair based on the unrounded means and is not used to rank diferent backbones.
<table><tr><td colspan="5">(a) Cora, Citeseer, and ACM</td><td colspan="5"></td><td colspan="4">ACM</td></tr><tr><td></td><td colspan="4">Cora</td><td colspan="4">Citeseer</td><td colspan="4"></td></tr><tr><td>Method</td><td>ACC</td><td>NMI</td><td>ARI</td><td></td><td>F1 ACC</td><td></td><td>NMI</td><td>ARI</td><td>F1</td><td>ACC</td><td>NMI</td><td>ARI</td><td>F1</td></tr><tr><td>HALO</td><td></td><td>0.5078</td><td>0.3853</td><td>0.2634</td><td>0.4403</td><td>0.4916</td><td>0.2597</td><td>0.2354</td><td>0.4415</td><td>0.8282</td><td>0.5334</td><td>0.5755</td><td>0.8262</td></tr><tr><td>HALO + SHR</td><td>0.5086</td><td>0.3865</td><td>0.2639</td><td>0.4399</td><td></td><td>0.4926</td><td>0.2603</td><td>0.2363</td><td>0.4424</td><td>0.8294</td><td>0.5354</td><td>0.5778</td><td>0.8275</td></tr><tr><td>DGAC</td><td>0.6393</td><td>0.5339</td><td>0.4381</td><td>0.6344</td><td></td><td>0.6909</td><td>0.4350</td><td>0.4450</td><td>0.6444</td><td>0.3647</td><td>0.0067</td><td>0.0051</td><td>0.2890</td></tr><tr><td>DGAC + SHR</td><td>0.6405</td><td>0.5351</td><td>0.4397</td><td>0.6350</td><td></td><td>0.6909</td><td>0.4350</td><td>0.4450</td><td>0.6444</td><td>0.3648</td><td>0.0068</td><td>0.0052</td><td>0.2891</td></tr><tr><td>SynC</td><td>0.6744</td><td>0.5034</td><td>0.4336</td><td>0.6676</td><td></td><td>0.6939</td><td>0.4426</td><td>0.4538</td><td>0.6410</td><td>0.9155</td><td>0.7036</td><td>0.7647</td><td>0.9157</td></tr><tr><td>SynC + SHR</td><td>0.6763</td><td>0.5058</td><td>0.4364</td><td>0.6694</td><td></td><td>0.6938</td><td>0.4428</td><td>0.4538</td><td>0.6410</td><td>0.9157</td><td>0.7040</td><td>0.7652</td><td>0.9159</td></tr></table>

<table><tr><td colspan="5">(b) UAT and DBLP</td><td colspan="4">DBLP</td></tr><tr><td>Method</td><td>ACC</td><td>UAT NMI</td><td>ARI</td><td>F1</td><td>ACC</td><td>NMI</td><td>ARI</td><td>F1</td></tr><tr><td></td><td></td><td>0.2281</td><td>0.1933</td><td>0.4642</td><td>0.3062</td><td>0.0259</td><td>-0.0010</td><td>0.1554</td></tr><tr><td>HALO HALO + SHR</td><td>0.4951 0.4966</td><td>0.2305</td><td>0.1955</td><td>0.4653</td><td>0.3062</td><td>0.0260</td><td>-0.0010</td><td>0.1555</td></tr><tr><td></td><td></td><td>0.2182</td><td>0.1961</td><td>0.4740</td><td>0.3202</td><td>0.0232</td><td>0.0213</td><td></td></tr><tr><td>DGAC DGAC + SHR</td><td>0.4963 0.4973</td><td>0.2189</td><td>0.1970</td><td>0.4753</td><td>0.3204</td><td>0.0232</td><td>0.0214</td><td>0.2733 0.2734</td></tr><tr><td></td><td>0.5555</td><td>0.2584</td><td></td><td>0.5478</td><td>0.8104</td><td>0.5140</td><td>0.5695</td><td></td></tr><tr><td>SynC SynC + SHR</td><td>0.5551</td><td>0.2579</td><td>0.2485 0.2484</td><td>0.5471</td><td>0.8107</td><td>0.5150</td><td>0.5703</td><td>0.8049 0.8052</td></tr></table>

## Broader native-interface evaluation.

We further evaluate 15 valid native-interface combinations spanning five backbones. Figure 3a shows that 10 combinations had positive macro gain, DGAC–Citeseer returned exact no-action, and four combinations were negative. With equal weighting across combinations<sup>1</sup>, the mean ACC, NMI, ARI, and F1 gains were 0.103, 0.171, 0.152, and 0.123 pp, respectively, yielding a macro gain of 0.137 pp with a 95% combination-bootstrap CI of [0.059, 0.228] pp. Because the dataset sets difer across backbones, this analysis is used to assess cross-system behavior rather than as a direct replacement for the common-suite comparison. Complete combination-level absolute baseline/final values and gains are provided in Appendix D.4.

![](images/4c246ad9a1b761cfcdb9422bb56f4165681b085350cefa11e906d79598461eaf.jpg)  
Raw points are frozen units (seeds, folds, or one frozen checkpoint). Panel a intervals are descriptive  
within-combination 95% bootstrap intervals; n=1 combinations have no interval. Panel b shows 95% bootstrap  
CIs only for the predefined combination-equal and backbone-equal estimators; other entries are exploratory point estimates.

Figure 3: Performance heterogeneity and robustness of SHR across backbone–dataset combinations. (a) Macro gain for each of the 15 combinations in the broader native-interface evaluation. Pale points denote frozen runs, highlighted markers denote combination means, and the vertical zero line denotes no change relative to the frozen baseline. Combinations with multiple frozen runs also show descriptive intervals; the three single-checkpoint DeSE combinations are shown as points only, without artificial intervals. (b) Aggregate macro gain under alternative aggregation schemes and leave-one-backbone-out analyses. Combination-equal aggregation is the main estimand for this broader evaluation; all other results are robustness or exploratory analyses. Frozen units difer by backbone and comprise seeds, folds, or one frozen checkpoint.

Robustness of broader evaluation. Figure 3b compares alternative aggregation rules and leaveone-backbone-out estimates. The combination-equal, backbone-equal, and run-weighted macro gains remained positive at 0.137, 0.109, and 0.154 pp, respectively. Leave-one-backbone-out estimates ranged from 0.068 to 0.170 pp, with the lowest value obtained after removing HALO. Thus, HALO contributes substantially to the aggregate efect, although the mean remains positive without it.

Intervention scale and post-processing cost. Performance changes should be interpreted together with their intervention scale. Table 4 shows that the equally weighted mean of the 15 cell-level change-ratio means was 0.209%; the median was 0.159%, the IQR was [0.077%, 0.343%], and the mean corresponded to 2.09 changed nodes per 1,000. Fourteen of the 15 cells had a mean change ratio below 0.5%, and all were below 1%. The broader 15-combination/113-run evaluation had a separate mean change ratio of 0.375%; these statistics describe diferent registries and are not pooled. The observed gains therefore arise from changes to only a small fraction of the frozen assignments.

SHR operates after checkpoint fixation and requires no backbone retraining. Under the current CPU/NumPy–SciPy implementation and tested graph scales, the median post-processing wall time from frozen inputs to $Q ^ { * }$ across the 75 units was 1.71 s (IQR, 1.48–2.01 s). Dataset-level medians ranged from 0.90 to 2.09 s. Backbone training time is excluded because it was not measured under the same hardware and protocol. Full intervention and timing details for all 15 cells are reported in Appendix D.1.

Table 4: Intervention scale and post-processing cost on the common suite. Macro gain and Change first average the five seeds within a backbone–dataset cell and then weight the three backbones equally for each dataset. Runtime is the median across 15 frozen units per dataset after one warm-up and three timed repetitions; it excludes backbone training.
<table><tr><td>Dataset</td><td>N</td><td>Macro gain (pp)</td><td>Change (%)</td><td>SHR time (s)</td></tr><tr><td>Cora</td><td>2708</td><td>+0.129</td><td>0.438</td><td>1.71</td></tr><tr><td>Citeseer</td><td>3327</td><td>+0.028</td><td>0.064</td><td>2.09</td></tr><tr><td>ACM</td><td>3025</td><td>+0.070</td><td>0.132</td><td>1.50</td></tr><tr><td>UAT</td><td>1190</td><td>+0.078</td><td>0.314</td><td>0.90</td></tr><tr><td>DBLP</td><td>4057</td><td>+0.026</td><td>0.097</td><td>1.98</td></tr><tr><td>Overall</td><td></td><td>+0.066</td><td>0.209</td><td>1.71</td></tr></table>

## 5.3 Ablation Study

We study two types of ablation: method components and auxiliary relation sources. Component ablations examine matched-null evidence, coverage control, state averaging, and node-level execution. Source ablations replace only the relation used to construct the candidate residual while keeping the frozen backbone and the downstream SHR procedure fixed.

Component ablation. We compare the complete method with several simplified variants. Efectonly inherits the changed-node count selected by complete SHR and therefore serves only as a matched-coverage diagnostic. No matched-null retains the label-free evidence from graph structure, node attributes, and assignment displacement, but removes the matched-null random control, so a real candidate state no longer has to outperform null states obtained after disrupting the node–residual correspondence. No coverage/MDL removes both the coverage prior based on changed-node count and the MDL complexity penalty, so states that modify more nodes are no longer additionally constrained for their greater freedom. No node-level filtering + provenance veto directly uses a candidate result after it passes the global magnitude constraints; it no longer applies Combined-LCB or cluster-survival screening to changed nodes and no longer rejects an entire run according to the mean attribute support of retained nodes. MAP retains the candidate states and their weight calculation, but replaces averaging over the no-action and multiple non-zero states with the single state having the largest weight.

Table 5: Component ablation of SHR. Results are from the frozen historical distillation registry and are used only to compare the performance–risk profiles of method simplifications; their absolute values are not mixed with those from the current broader native-interface evaluation.
<table><tr><td>Variant</td><td>Macro gain (pp)</td><td>Neg. run (%)</td><td>Worst combo (pp)</td><td>Interpretation</td></tr><tr><td>Full SHR</td><td>+0.118</td><td>18.4</td><td>-0.041</td><td>Complete method/reference</td></tr><tr><td>Effect-only†</td><td>+0.048</td><td>21.3</td><td>0.000</td><td>Matched-coverage diagnostic</td></tr><tr><td>No matched-null</td><td>+0.001</td><td>25.7</td><td>-0.075</td><td>Null calibration removed</td></tr><tr><td>No coverage/MDL</td><td>-0.436</td><td>45.6</td><td>-4.146</td><td>Coverage control removed</td></tr><tr><td>No node-level filtering + provenance veto</td><td>+0.068</td><td>28.7</td><td>-0.209</td><td>LCB/survival filtering and</td></tr><tr><td>MAP</td><td> $- 3 . 9 \times 1 0 ^ { - 5 }$ </td><td>1.5</td><td>-0.005</td><td>provenance veto removed State averaging replaced by MAP;</td></tr></table>

<sup>†</sup>Efect-only inherits SHR’s frozen changed-count budget and is not independently deployable. Node-level filtering comprises Combined-LCB screening and cluster-survival protection. Worst combo denotes the minimum combinationlevel mean and is distinct from a catastrophic-run count.

Table 5 compares the variants by macro gain, negative-run rate, and worst-combination performance. This experiment uses the historical frozen 15-combination/136-run registry, whereas the current broader native-interface evaluation uses the 15-combination/113-run protocol. The results are therefore used only to characterize changes in the gain–risk profile after method simplification and are not compared directly with the absolute gains in the broader evaluation. No tested simplification simultaneously preserves the mean gain, negative-run behavior, and execution constraints of complete SHR. This ablation does not isolate the causal contribution of each component.

Auxiliary relation-source ablation. For the source ablation, the downstream SHR procedure is fixed and only the relation used to construct the candidate residual is changed. We compared original-graph propagation, an attribute k-NN graph, a single-scale attribute hypergraph, the full multiscale attribute hypergraph, and a randomized hypergraph that preserved node incidence degree and hyperedge size. The randomized hypergraph preserves node incidence degree and hyperedge size while shufling the node–hyperedge correspondence. All auxiliary sources used the same attribute preprocessing, representation alignment, and mapping to the clustering coordinate system. Refinement strength was calibrated against the same target hard-assignment change ratios, so the sources had the same target coverage states at the candidate-generation stage.

Under the current frozen 15-combination/113-run protocol, the full multiscale attribute hypergraph achieved a combination-equal macro gain of +0.137 pp, with a 95% CI of [0.062, 0.228] pp. As shown in Figure 4, its paired diference<sup>2</sup> relative to the randomized hypergraph was +0.071 pp, with a 95% CI of [0.019, 0.133] pp, indicating an incremental contribution of the true node–hyperedge correspondence to the complete SHR refinement process. By contrast, the diferences relative to the attribute k-NN graph, original-graph propagation, and single-scale attribute hypergraph were +0.005 pp, −0.015 pp, and −0.002 pp, with corresponding 95% CIs of [−0.025, 0.041] pp, [−0.088, 0.056] pp, and [−0.044, 0.046] pp, all of which crossed zero. The current results therefore do not establish that the multiscale hypergraph is consistently superior to these ordinary relation sources. Meanwhile, the randomized hypergraph itself achieved a macro gain of +0.067 pp (95% CI, [0.037, 0.099] pp), whereas the multiscale hypergraph had the highest negative-run rate among the five sources, at 23.3%. Although the sources were calibrated to the same target change ratios during candidate generation, their final change ratios were not identical after safety projection and node screening; for example, the multiscale and randomized sources changed 0.375% and 0.229% of nodes, respectively. Because the final change ratios difer after projection and node filtering, these comparisons measure the operational efect of replacing the relation source rather than a strictly coverage-matched structural efect. The true node–hyperedge correspondence shows an incremental advantage over the shufled control, while the current results do not show a consistent advantage over the ordinary graph, attribute k-NN, or single-scale hypergraph sources. Additional protocol details are provided in Appendix E.2.

![](images/79da4db0ca4c3700b2526c7614cb76dbb4f33239177259ac0f70d5291ab4cd37.jpg)

![](images/474bc3e706c42f8ce9f4ead5afa9c2a35ecf07517040744905b13ec06874886a.jpg)  
Points are combination-equal estimates over 15 frozen backbone–dataset combinations. Error bars in a and b are 95% combination-bootstrap CIs. Panel c uses fixed-size markers; zero remains visible on all effect axes.

![](images/aaf358fc943c2dfd72459639749b691749ca80361f63b1ff6eb36835467495c4.jpg)

Figure 4: Auxiliary relation-source ablation under the unified frozen SHR protocol. (a) Combination-equal four-metric macro gain for five auxiliary relation sources across the 15 backbone–dataset combinations; error bars denote 95% bootstrap CIs with the combination as the sampling unit. (b) Paired macro-gain diference between the full multiscale hypergraph source and each control; the vertical dashed line denotes zero diference. (c) Combination-equal macro gain and negative-run rate for each source. The five shufled hypergraphs are first averaged within the same frozen run and are not treated as independent runs. Final change ratios can difer among sources, so panel b represents the overall efect of replacing the source under a common operational protocol rather than a strictly final-coverage-matched direct structural efect.

## 5.4 Mechanism Analysis

Critical refinement strength and implicit node selection. Section 4.3 shows that, even when all nodes receive the same global refinement strength, they have diferent critical strengths $\eta _ { \mathrm { c r i t } }$ because their native assignment gaps and relative residual pushes difer. For a fixed global strength, the changed mask predicted by $\eta \geq \eta _ { \mathrm { c r i t } } ( i )$ exactly matched the changes obtained by direct application of the residual. We then ranked nodes by $\eta _ { \mathrm { c r i t } }$ , formed a candidate changed set at the same target change ratio, and compared it with the globally changed set obtained by direct calibration. Across all 100 frozen runs, the mean Jaccard similarity between the two sets was 0.930; when only the 89 runs with nonzero refinement were considered, it was 0.922.<sup>3</sup> The global residual therefore reaches nodes with smaller critical strengths first, producing an implicit ordering of candidate changes. SHR subsequently decides which of these candidate changes are retained. As shown in Figure 5, for a fixed global strength $\eta ^ { * }$ , the condition for a node to reach the assignment boundary is $\eta ^ { * } \geq \eta _ { \mathrm { c r i t } }$ , equivalently $d \geq m / \eta ^ { * }$ . Equality denotes tied cluster scores, whereas a strict hard-assignment change requires the node to cross the boundary.

Critical-strength decision boundary for implicit node selection  
![](images/5a33e58246eb30724807eda0ace8f4a3abf8c0c5445670817b8a8e1755c331d2.jpg)  
Mechanism schematic only: points are illustrative; the boundary follows the analytical condition $\eta ^ { * } = \eta _ { \mathsf { c r i t } } .$  
Figure 5: Geometric interpretation of critical refinement strength. For a fixed global strength $\eta ^ { * }$ , the assignment boundary satisfies $d = m / \eta ^ { * }$ . Nodes above the boundary have smaller $\eta _ { \mathrm { c r i t } } = m / d$ and therefore change assignment earlier. Equality denotes tied cluster scores. The plotted nodes illustrate the analytical mechanism and do not represent an empirical distribution.

Refinement coverage and node-level outcomes. Critical refinement strength determines the order in which nodes enter the changed set, whereas the final coverage determines how many potential errors a method can reach. The DeSE results show that broader-coverage refinements generally achieve higher Repair Recall. SHR ultimately changes fewer nodes and therefore also has substantially lower Repair Recall. Smaller coverage, however, does not automatically produce a lower Harm Fraction: on Citeseer and Photo, the proportion of harmful changes under SHR was not lower than that under CS-BAYES. SHR is conservative mainly because it changes fewer nodes, not because the current evidence reliably identifies every beneficial refinement. On Cora, Citeseer, and Photo under the native DeSE soft-assignment interface, the historical DeSE development baseline achieved a three-dataset mean macro gain of 1.195 pp over the frozen outputs. Re-executing the saved historical procedure produced a maximum reproduction error of no more than $4 . 3 \times 1 0 ^ { - 6 }$ among the reported metrics; complete results are provided in Appendix D.3. Under the same frozen residual, the macro gains of CS-BAYES on the three datasets were 0.166, 0.350, and 3.136 pp, respectively. SHR changed 6, 6, and 49 nodes, whereas the historical development baseline changed 74, 116, and 463 nodes. Figure 6 shows that the gain diferences occur together with changes in intervention coverage and node-level filtering, while the candidate residual remains fixed. Strict definitions of repair, harm, and their diagnostic coordinate system are provided in Appendix B.3.

a Historical DeSE development gain

![](images/00aac212fd7b7c16c2862edec2bfdd73158b852574f527fa91c573e02cec362c.jpg)  
b Coverage contraction

![](images/a2a15d23b2fcce18c47f62b04ff5411a38e67986339cea0a671b43a97a5a4fed.jpg)

![](images/d9e53674d3b3159d8b8dbe861dc2f45ef1128ef81c3332239543af456d3af95b.jpg)

![](images/166f80d7438d7ba65b4eb0a76d6b33503a64d9edef02c452348c5e852c90eb7a.jpg)  
Figure 6: Development-stage improvement space and coverage diferences on DeSE. (a) Clustering gains of the historical DeSE development baseline over the frozen outputs on Cora, Citeseer, and Photo. (b) Changednode counts for the historical development baseline, CS-BAYES, and SHR. (c) Repair Recall for CS-BAYES and SHR. (d) Harm Fraction among changed nodes. SHR ultimately changes substantially fewer nodes, but its node-level Harm Fraction is not always lower. Repair and harm are post hoc diagnostics computed only after method outputs are frozen and do not enter SHR decisions.

Identifiability of label-free evidence. Post hoc identifiability diagnostics further indicate that the current unlabeled selector remains a limiting factor. In the fixed candidate space, 9 of the 10 active combinations retain at least 0.10 pp of oracle–global headroom, while the Combined-LCB repair–harm ROC–AUC is 0.684. Direction-stability and cross-system probes also show no consistent improvement. The current selector therefore appears more efective at limiting intervention coverage than at identifying every beneficial refinement. Full results are reported in Appendix E.1.

## 5.5 Risk and Applicability Boundaries

Gain and risk. The refinement strategies show diferent gain–risk profiles under the same frozen outputs and candidate residual. CS-BAYES had a combination-equal macro gain of 0.348 pp, compared with 0.137 pp for SHR; meanwhile, SHR had 3 catastrophic runs among 113 frozen units, whereas CS-BAYES had 12, and their worst-combination macro gains were −0.041 pp and −0.993 pp, respectively. The paired bootstrap intervals for the SHR–CS-BAYES macro-gain diference and negative-run-rate diference both crossed zero. The results show diferent empirical gain–risk profiles, but they do not support a statistically significant safety advantage for SHR.

![](images/be6fc01918b304b61cc77524ad3d3e0dc7dec566f8b12948c0014bd892cd5f74.jpg)

![](images/ebc1250f82561b75925a0bf7119c2e15b5f34dd18bf4784ffde12ada9e3fe123.jpg)

![](images/5b0ba10456ecf40a96e377290bb77fcaf058064cdf3f93be5e1fac293ce80e24.jpg)

![](images/0610499b72f8d993df61276eab9963aa5bcb348a2b9cda3fb6a0ba92fa97c135.jpg)  
† Diagnostic matched-coverage control; not independently deployable. Red indicates the preregistered catastrophic threshold or observed catastrophic runs.  
Figure 7: Gain–risk comparison under a shared residual. (a) Combination-equal macro gain. (b) Combination-equal negative-run rate. (c) Worst-combination macro gain. (d) Catastrophic-run count. Error bars denote 95% combination-bootstrap CIs. CM-Global inherits its coverage from CS-BAYES and therefore serves only as a matched-coverage diagnostic rather than an independent method.

Complete gain–risk values are reported in Appendix D.6.

Role of coverage. Coverage-Matched Global Residual (CM-Global) inherits the changed-node count of CS-BAYES in each run and approximately matches that coverage by adjusting global residual strength; it is therefore a mechanism diagnostic rather than an independently deployable method. CS-BAYES exceeds CM-Global by only 0.030 pp in combination-equal macro gain. Together with Section 5.4, this result shows that broader coverage coincides with larger observed gains in the current candidate space, while also exposing more nodes to harmful changes. The comparison does not isolate a causal efect of coverage.

Boundaries and applicability. SHR requires the frozen model to provide valid soft cluster assignments aligned with the original clustering coordinate system. When this interface condition is not met, the method can return the no-action state. For example, strict-final DeSE on Photo and Computers had fewer active clusters than the declared K, so the cluster-survival constraint prevented refinement. DBCD–Cora was likewise retained only as a boundary case. The preregistered 4 × 5 matrix further showed that native cluster collapse for DeSE on Cora, ACM, and UAT caused the matrix to fail its interface gate; common-suite backbone inclusion was therefore determined by five-dataset interface validity rather than SHR gain. A valid interface does not guarantee a positive SHR gain. Under the common-suite estimand, DGAC–Citeseer produced exact no-action and SynC–UAT had a macro gain of −0.042 pp; the latter was −0.041 pp in the broader native-interface registry. The current label-free evidence is more reliable for restricting the refinement range than for consistently identifying beneficial changes across datasets. Coverage and gain have no simple monotonic relation: broader coverage can increase both repair and harm. The observed gain–risk pattern is empirical and does not provide a performance guarantee. The label-free claim applies only to the SHR refinement stage after checkpoint fixation. Complete boundary cases, interface failures, and collapse counts are reported in Appendix C.3.

## 6 Conclusion

We study label-free post-processing of frozen graph-clustering models after checkpoint fixation and propose Selective Hypergraph Refinement (SHR). SHR keeps the backbone parameters, node representations, and original graph structure fixed. An attribute hypergraph provides the candidate residual, while label-free evidence determines which assignment changes are retained. The label-free constraint applies only to the SHR refinement stage after checkpoint fixation and does not include prior backbone training or checkpoint selection.

The controlled common suite shows that frozen clustering outputs retain a limited but measurable refinement space after training. Among 15 backbone–dataset cells, 13 had a positive mean gain, one produced exact no-action, and one was negative; the cell-equal macro gain was 0.066 pp, while only 0.209% of hard assignments changed on average. The broader five-backbone, 15-combination nativeinterface evaluation yielded a macro gain of 0.137 pp at a mean change ratio of 0.375%, again with substantial heterogeneity. Because only a small fraction of assignments is changed, SHR is better viewed as local post-processing of the frozen clustering output rather than reconstruction of a new global clustering result. Critical-strength analysis shows that node responses depend on both the native assignment gap and the relative residual push toward candidate clusters. Broader coverage reaches more potentially repairable nodes but also exposes more nodes to harmful changes.

The source ablation shows an incremental efect of the true node–hyperedge correspondence relative to the shufled hypergraph under the same frozen protocol. However, the full multiscale attribute hypergraph did not consistently outperform an attribute k-NN graph or original-graph propagation. The results support the use of genuine auxiliary relational information, but the multiscale hypergraph does not show a consistent advantage over the alternative relation sources considered here. The current label-free evidence also remains limited in distinguishing beneficial from harmful refinements. Improving this label-free selection remains an important direction for frozen graph-clustering post-processing.

## Data and code availability

All datasets used in this study are publicly available; their sources and preprocessing details are provided in the experimental setup and Supplementary Information. The code used for SHR, result aggregation, and metric computation, together with machine-readable records supporting the primary results, will be released through a public code repository upon acceptance.

During the SHR refinement stage, node labels are used only for external evaluation and post hoc analysis after all method outputs have been frozen. They do not enter candidate refinement, state weighting, or node selection.

## Generative AI Disclosure

Generative AI tools were used extensively during manuscript preparation, including for language editing and code-related assistance. All AI-assisted material was reviewed and verified by the author, who takes full responsibility for the scientific content, experimental results, references, figures, and conclusions. Generative AI tools are not authors.

## References

[1] Jingyun Zhang, Hao Peng, Li Sun, Guanlin Wu, Chunyang Liu, and Zhengtao Yu. Unsupervised graph clustering with deep structural entropy. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pages 3752–3763. Association for Computing Machinery, 2025. doi: 10.1145/3711896.3737173.

[2] Yuchen Zhu, Kuang Zhou, Haishan Ye, Guang Dai, and Ivor W. Tsang. HALO: Hardness-aware bilevel-inspired contrastive graph clustering. International Journal of Approximate Reasoning, 193:109657, 2026. doi: 10.1016/j.ijar.2026.109657.

[3] Kun Xie, Renchi Yang, and Sibo Wang. Difusion-based graph-agnostic clustering. In Proceedings of the ACM on Web Conference 2025, pages 1353–1364. Association for Computing Machinery, 2025. doi: 10.1145/3696410.3714652.

[4] Shifei Ding, Benyu Wu, Xiao Xu, Ling Ding, and Xindong Wu. SynC: Synergistic boosting of structure and representation for deep graph clustering. IEEE Transactions on Neural Networks and Learning Systems, 37(6):2959–2968, 2026. doi: 10.1109/TNNLS.2025.3643594.

[5] Xi Liu, Xiaolin Chen, Wenqian Yang, and Yong Yu. DGM: deep graph clustering with mincut for analysis of single-cell transcriptomics. BMC Genomics, 27(1):383, 2026. doi: 10.1186/ s12864-026-12694-y.

[6] Yan Wang, Yupeng Liu, Xiaojie Sun, and Jun Fu. DBCD: Deep balanced community detection via consensus-guided joint optimization in attributed networks. Expert Systems with Applications, 298:129487, 2026. doi: 10.1016/j.eswa.2025.129487.

[7] Xuanting Xie, Bingheng Li, Erlin Pan, Zhaochen Guo, Zhao Kang, and Wenyu Chen. One node one model: Featuring the missing-half for graph clustering. Proceedings of the AAAI Conference on Artificial Intelligence, 39(20):21688–21696, 2025. doi: 10.1609/aaai.v39i20.35473.

[8] Dengyong Zhou, Jiayuan Huang, and Bernhard Sch¨olkopf. Learning with hypergraphs: Clustering, classification, and embedding. In Advances in Neural Information Processing Systems 19, volume 19, pages 1601–1608, 2006. doi: 10.7551/mitpress/7503.003.0205.

[9] Qianqian Wang, Bowen Zhao, Zhengming Ding, Xiangdong Zhang, and Quanxue Gao. A simple yet efective hypergraph clustering network. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 6352–6360, 2025. doi: 10.24963/ijcai.2025/707.

[10] Qianqian Wang, Bowen Zhao, Zhengming Ding, Wei Feng, and Quanxue Gao. Hypergraph clustering network with partial attribute imputation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2697–2706, 2025. doi: 10.1109/ICCV51701.2025.00259.

[11] Ju Niu and Yuhui Du. Joint consensus kernel learning and adaptive hypergraph regularization for graph-based clustering. Information Sciences, 689:121468, 2025. doi: 10.1016/j.ins.2024. 121468.

[12] Dehua Peng, Guangyao Fang, Zhipeng Gui, Yuhang Liu, and Huayi Wu. Topological and semantic contrastive graph clustering by ricci curvature augmentation and hypergraph fusion. Knowledge-Based Systems, 334:115130, 2026. doi: 10.1016/j.knosys.2025.115130.

[13] Rapha¨el Pellegrin, Lukas Fesser, and Melanie Weber. Higher-order learning with graph neural networks via hypergraph encodings. In Advances in Neural Information Processing Systems 38, volume 38, pages 192577–192620, 2025. doi: 10.52202/085713-5782.

[14] A. Quadir and M. Tanveer. Hypergraph neural network with state space models for node classification. Engineering Applications of Artificial Intelligence, 163(Part 3):112922, 2026. doi: 10.1016/j.engappai.2025.112922.

[15] Ziwei Chen, Jianjian Jiang, Xiangmin Luo, Fangyuan Lei, Xiaochen Yuan, and Jin Zhan. Dual-channel hypergraph networks in the time–frequency domain for learning advanced spatiotemporal dependencies in multivariate time series. Neurocomputing, 648:130600, 2025. doi: 10.1016/j.neucom.2025.130600.

[16] Yusheng Zhao, Qixin Zhang, Xiao Luo, Junyu Luo, Wei Ju, Zhiping Xiao, and Ming Zhang. Test-time adaptation on graphs via adaptive subgraph-based selection and regularized prototypes. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 78003–78022, 2025. URL https://proceedings.mlr.press/v267/zhao25ai.html.

[17] Haoxiang Zhang, Zhuofeng Li, Qiannan Zhang, Ziyi Kou, Juncheng Li, and Shichao Pei. Avoiding structural pitfalls: Self-supervised low-rank feature tuning for graph test-time adaptation. Transactions on Machine Learning Research, November 2025. URL https://openreview.net/ forum?id=yiS6q42LLt.

[18] Mengyao Li, Xu Zhou, Jiapeng Zhang, Zhibang Yang, Cen Chen, and Kenli Li. Totf: Missingaware encoders for clustering on multi-view incomplete attributed graphs. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 3054–3062, 2025. doi: 10.24963/ijcai.2025/340.

[19] Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin R. Benson. Combining label propagation and simple models out-performs graph neural networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= 8E1-f3VhX1o.

[20] Yuanhang Shao and Xiuwen Liu. Nonlinear correct and smooth for graph-based semi-supervised learning. ACM Transactions on Knowledge Discovery from Data, 19(3):1–32, 2025. doi: 10. 1145/3712604.

[21] Cheng Yang, Xumeng Gong, Chuan Shi, and Philip S. Yu. A post-training framework for improving heterogeneous graph neural networks. In Proceedings of the ACM Web Conference 2023, pages 251–262. Association for Computing Machinery, 2023. doi: 10.1145/3543507.3583282.

[22] Jennifer A. Hoeting, David Madigan, Adrian E. Raftery, and Chris T. Volinsky. Bayesian model averaging: A tutorial. Statistical Science, 14(4):382–401, 1999. doi: 10.1214/ss/1009212519.

[23] Peng Zhou, Xia Wang, Liang Du, and Xuejun Li. Clustering ensemble via structured hypergraph learning. Information Fusion, 78:171–179, 2022. doi: 10.1016/j.infus.2021.09.003.

[24] Jiaxuan Xu, Lei Duan, Xinye Wang, and Liang Du. Dynamic anchor-based ensemble clustering via hypergraph reconstruction. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, pages 6740–6748, 2025. doi: 10.24963/ijcai.2025/750.

[25] Owen Forbes, Edgar Santos-Fernandez, Paul Pao-Yen Wu, Hong-Bo Xie, Paul E. Schwenn, Jim Lagopoulos, Lia Mills, Dashiell D. Sacks, Daniel F. Hermens, and Kerrie Mengersen. clusterbma: Bayesian model averaging for clustering. PLOS ONE, 18(8):e0288000, 2023. doi: 10.1371/ journal.pone.0288000.

[26] Huayi Tang and Yong Liu. Deep safe multi-view clustering: Reducing the risk of clustering performance degradation caused by view increase. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 202–211, 2022. doi: 10.1109/CVPR52688. 2022.00030.

[27] Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information Theory, 37(1):145–151, 1991. doi: 10.1109/18.61115.

## A Implementation Details and Frozen Protocol

## A.1 SHR Implementation Details

After truncated SVD and standardization of the attributes, an attribute hypergraph is constructed with $k _ { h } = 1 0 ;$ the multiscale representation is constructed according to Eq. (9). The auxiliary $Q _ { h }$ enters the context only and is not used as a direct output. We next specify the state weighting used in the formal implementation. For $r \in S$ , let $C _ { r } = \{ i :$ : arg max ${ Q _ { i } ^ { \left( \eta _ { r } \right) } \neq }$ arg max $Q _ { 0 , i } \}$ and $c _ { r } = | C _ { r } |$ . In the mth repeated held-out fold, let $T ^ { ( m ) }$ and $F ^ { ( m ) }$ denote the topology and attribute score matrices. The node utility and state statistic implemented in the code are

$$
\begin{array} { r } { u _ { i , r } ^ { ( m ) } = \frac { 1 } { 2 } \langle Q _ { i } ^ { ( \eta _ { r } ) } - Q _ { 0 , i } , T _ { i } ^ { ( m ) } \rangle + \frac { 1 } { 2 } \langle Q _ { i } ^ { ( \eta _ { r } ) } - Q _ { 0 , i } , F _ { i } ^ { ( m ) } \rangle - 0 . 0 5 ~ \mathrm { J S } ( Q _ { i } ^ { ( \eta _ { r } ) } , Q _ { 0 , i } ) , } \end{array}\tag{28}
$$

$$
s _ { r } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { 1 } { c _ { r } } \sum _ { i \in C _ { r } } u _ { i , r } ^ { ( m ) } ,\tag{29}
$$

where $M = 5 ;$ when $c _ { r } = 0$ , the implementation sets $s _ { r } = 0$ . Each matched-null repetition permutes the rows of $\Delta$ within strata defined by the baseline cluster and confidence quintile, and recalibrates the null state to the same target hard-assignment change ratio. For the $B = 1 9$ null statistics $s _ { r , j } ^ { \mathrm { n u l l } }$ we define

$$
b _ { r } = \sum _ { j = 1 } ^ { B } \mathbf 1 [ s _ { r , j } ^ { \mathrm { n u l l } } \geq s _ { r } ] , \qquad e _ { r } = \operatorname* { m a x } \left( \frac { B - b _ { r } } { 1 + b _ { r } } , \frac 1 { B + 1 } \right) .\tag{30}
$$

The coverage complexity is computed exactly as in the code using the Beta–Binomial count prior and the MDL term:

$$
\begin{array} { r } { L ( c ) = \log \mathrm { B e t a B i n o m i a l } ( c ; N , 1 , 1 9 ) - \frac { 1 } { 2 } \log ( 1 + c ) . } \end{array}\tag{31}
$$

Let $s$ contain the seven nonzero efect states. The log weights for the no-action and nonzero states, their normalized weights, and the resulting output are

$$
\ell _ { 0 } = L ( 0 ) ,
$$

$$
\ell _ { r } = \log e _ { r } + L ( c _ { r } ) - \log | S | ,\tag{32}
$$

$$
\omega _ { j } = \frac { \exp ( \ell _ { j } ) } { \exp ( \ell _ { 0 } ) + \sum _ { r \in S } \exp ( \ell _ { r } ) } , \qquad Q _ { \mathrm { m i x } } = \mathrm { N o r m a l i z e } \left( \omega _ { 0 } Q _ { 0 } + \sum _ { r \in S } \omega _ { r } Q ^ { ( \eta _ { r } ) } \right) .\tag{33}
$$

Thus, the no-action state participates directly in the averaging through $Q _ { 0 }$ and its weight $\omega _ { 0 }$ , rather than being added after the averaging step. The projection sequentially evaluates

$$
Q _ { \tau } = \mathrm { N o r m a l i z e } ( ( 1 - \tau ) Q _ { 0 } + \tau Q _ { \mathrm { m i x } } ) , \qquad \tau \in \{ 1 , 0 . 7 5 , 0 . 5 , 0 . 2 5 , 0 \} ,\tag{34}
$$

and adopts the first state that satisfies the frozen hard-assignment-change-ratio, mean-JS, and active-cluster constraints. Positive-LCB filtering then reverts unsupported rows, and failure of the attribute-provenance condition causes the entire run to return $Q _ { 0 }$ . These normalized evidence weights are based solely on label-free empirical evidence and the complexity term; they are not posterior probabilities or probabilities of correctness.

## A.2 Frozen Implementation Constants

Table 6 summarizes the implementation constants shared by all frozen evaluations.

Table 6: Frozen implementation constants for the formal SHR configuration.
<table><tr><td>Parameter</td><td>Value</td><td>Role</td><td>Selection rule / provenance</td></tr><tr><td>Attribute SVD width</td><td> $\operatorname* { m i n } ( 6 4 , D _ { X } - 1 , N - 1 )$ </td><td>Auxiliary attribute signal</td><td>Fixed implementation rule</td></tr><tr><td>Alignment width</td><td> $\operatorname* { m i n } ( 3 2 , D z _ { 0 } , D _ { H _ { h } } )$ </td><td>Common Procrustes coordinates</td><td>Fixed implementation rule</td></tr><tr><td>Ridge λ</td><td> $1 0 ^ { - 2 }$ </td><td>Native-logit decoder</td><td>Frozen pre-intervention implementation</td></tr><tr><td>Hypergraph  $k _ { h }$ </td><td>10</td><td>Attribute kNN hyperedges</td><td>Shared across</td></tr><tr><td>Effect targets</td><td>.005, .01, .02, .03, .05, .08, .12</td><td>Target hard-assignment</td><td>datasets/backbones Frozen target grid</td></tr><tr><td>η calibration grid</td><td>0 plus 80 log-spaced points in [.01, 20]</td><td>change ratios Match effect targets</td><td>Frozen search grid</td></tr><tr><td>Evidence repeats/folds</td><td>5/5</td><td>Predictive evidence</td><td>Fixed repeated-fold protocol</td></tr><tr><td>JS evidence penalty</td><td>.05</td><td>Penalize assignment</td><td>Frozen implementation</td></tr><tr><td>Matched-null repetitions</td><td>19</td><td>displacement Empirical odds er</td><td>Frozen permutation</td></tr><tr><td>Coverage prior</td><td> $\operatorname { B e t a - B i n o m i a l } ( N , 1 , 1 9 )$ </td><td>Sparse count prior</td><td>budget Frozen implementation</td></tr><tr><td>MDL coefficient</td><td>.5</td><td>log(1 + c) complexity</td><td>Frozen implementation</td></tr><tr><td>Projection τ</td><td>1, .75, .5, .25, 0</td><td>penalty Conservative shrinkage</td><td>Fixed ordered path</td></tr><tr><td>Max hard-change ratio</td><td>.20</td><td>Projection constraint</td><td>Frozen protocol</td></tr><tr><td>Mean JS threshold</td><td>.10</td><td>Projection constraint</td><td>Frozen protocol</td></tr><tr><td>LCB coefficient</td><td>1.645</td><td>One-sided node filter</td><td>Frozen 95% coefficient</td></tr><tr><td>Provenance condition</td><td>retained attribute support mean</td><td>Run-level accept/reject</td><td>Frozen hard-reject rule</td></tr></table>

## B Evaluation Protocol and Metric Definitions

## B.1 Label-Isolation Protocol

The label-isolation protocol separates the construction and evaluation stages. The construction stage reads only the frozen graph, attributes, representation, $Q _ { 0 }$ , residual, and existing label-free signals, and writes the candidate states, changed masks, coverage values, and selection outputs to disk before labels are accessed. The evaluation stage then reads the labels to compute external clustering metrics, repair/harm, oracle quantities, AUC, and tail precision. All intervention outputs are frozen before labels are accessed, and no post-hoc quantity is fed back into candidate generation, state weighting, node filtering, or the final output.

## B.2 Clustering Metrics

For each run and prediction ˆy, ACC first constructs a contingency table between the ground-truth labels and predicted clusters, and then applies Hungarian assignment to maximize the matched count; the baseline and intervention mappings are obtained independently. F1 is computed as the macro average after applying the same prediction-specific mapping, with zero division=0. NMI uses arithmetic normalization, and ARI uses the adjusted Rand definition; both are invariant to permutations of cluster identifiers.

## B.3 Repair, Harm, and Oracle Diagnostics

Repair/harm uses a distinct but fixed diagnostic coordinate system: a Hungarian mapping $M _ { 0 }$ is fitted only to the baseline prediction, and $M _ { 0 }$ is then used to assess both baseline and intervention correctness. Let

$$
\begin{array} { r } { h _ { i } = { \bf 1 } [ \hat { y } _ { i } ^ { * } \neq \hat { y } _ { i } ^ { 0 } ] , \quad \quad c _ { i } ^ { 0 } = { \bf 1 } [ M _ { 0 } ( \hat { y } _ { i } ^ { 0 } ) = y _ { i } ] , \quad \quad c _ { i } ^ { * } = { \bf 1 } [ M _ { 0 } ( \hat { y } _ { i } ^ { * } ) = y _ { i } ] , } \end{array}\tag{35}
$$

then

$$
\mathrm { r e p a i r } _ { i } = h _ { i } ( 1 - c _ { i } ^ { 0 } ) c _ { i } ^ { * } , \quad \mathrm { h a r m } _ { i } = h _ { i } c _ { i } ^ { 0 } ( 1 - c _ { i } ^ { * } ) , \quad \mathrm { n e u t r a l } _ { i } = h _ { i } - \mathrm { r e p a i r } _ { i } - \mathrm { h a r m } _ { i } .\tag{36}
$$

Repairable residual candidates have finite $\eta _ { \mathrm { c r i t } , i }$ , a defined $b _ { \mathrm { c r i t } , i } , c _ { i } ^ { 0 } = 0$ , and $M _ { 0 } ( b _ { \mathrm { c r i t } , i } ) = y _ { i }$ . The oracle is allowed to evaluate only the residual-specified $b _ { \mathrm { c r i t } , i }$ and cannot choose a diferent target class. Repair Recall is the number of actual repairs divided by this fixed candidate pool and is recorded as NA when the denominator is empty. Harm Fraction is the number of harmful changes divided by the changed-node count; when the changed-node count is 0, the implementation records it as 0.

Accepted-harm nodes are changed nodes retained by SHR for which har $\mathrm { n } _ { i } = 1 \mathrm { : }$ ; that is, they were correct under the fixed baseline mapping $M _ { 0 }$ but became incorrect after SHR refinement. Missedrepair nodes are drawn from the same fixed eligible residual-candidate pool; they were not ultimately selected by SHR, although the residual-specified $b _ { \mathrm { c r i t } , i }$ would change a baseline error into a correct assignment under $M _ { 0 }$ . Both categories are determined with labels only after method outputs are frozen and do not enter SHR node selection.

## C Datasets, Backbones and Provenance

## C.1 Dataset Statistics and Preprocessing

Table 7 reports the actual inputs included in the current, development, or boundary registries. Edge counts are the numbers of undirected edges after the loader removes self-loops, deduplicates edges, and symmetrizes the graph. SHR does not override each backbone’s feature preprocessing; instead, it reconstructs the auxiliary signal from the frozen input attributes by applying truncated SVD followed by standardization, and uses $k _ { h } = 1 0$ for the attribute kNN hypergraph. For graph evidence and the residual context, self-loops are added temporarily only within the propagation operator. When the same dataset is used by diferent backbones, backbone-specific upstream preprocessing is inherited from the corresponding released implementation.

Table 7: Dataset statistics and sources corresponding to the actual loaders and artifacts. DBCD–Cora is a separate interface boundary; its declared K = 19 is inherited from this frozen interface and does not replace the standard Cora value of $K = 7 .$
<table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td><td>Features</td><td>K</td><td>Backbone(s)</td><td>Input / preprocessing source</td></tr><tr><td>Cora</td><td>2708</td><td>5278</td><td>1433</td><td>7</td><td>DeSE, DGAC, SynC</td><td>PyG Planetoid canonical undirected graph</td></tr><tr><td>Citeseer</td><td>3327</td><td>4552</td><td>3703</td><td>6</td><td>DeSE, DGAC, SynC</td><td>PyG Planetoid canonical undirected graph</td></tr><tr><td>Photo</td><td>7650</td><td>119081</td><td>745</td><td>8</td><td>DeSE</td><td>PyG Amazon canonical undirected graph</td></tr><tr><td>Computers</td><td>13752</td><td>245861</td><td>767</td><td>10</td><td>DeSE legacy/boundary</td><td>PyG Amazon canonical undirected graph</td></tr><tr><td>ACM</td><td>3025</td><td>13128</td><td>1870</td><td>3</td><td>HALO, SynC</td><td>Released NPY; diagonal removed; symmetrized</td></tr><tr><td>UAT</td><td>1190</td><td>13599</td><td>239</td><td>4</td><td>HALO, SynC</td><td>Released NPY; diagonal removed; symmetrized</td></tr><tr><td>EAT</td><td>399</td><td>5993</td><td>203</td><td>4</td><td>HALO</td><td>Released HALO NPY; diagonal removed; symmetrized</td></tr><tr><td>BAT</td><td>131</td><td>1003</td><td>81</td><td>4</td><td>HALO</td><td>Released HALO NPY; diagonal removed; symmetrized</td></tr><tr><td>DBLP</td><td>4057</td><td>3528</td><td>334</td><td>4</td><td>SynC</td><td>Released SynC NPY; diagonal removed; symmetrized</td></tr><tr><td>Adam</td><td>3660</td><td>14270</td><td>10</td><td>8</td><td>DGM</td><td>Frozen strict feature/edge artifacts; symmetrized</td></tr><tr><td>DBCD-Cora</td><td>2708</td><td>5278</td><td>1433</td><td>19</td><td>DBCD</td><td>Planetoid raw reconstruction; interface boundary</td></tr></table>

## C.2 Backbone and Checkpoint Provenance

Upstream training and checkpoint-selection protocols follow the corresponding backbone implementations and are reported separately in Table 8. Once $Z _ { 0 }$ and $Q _ { 0 }$ are fixed, every SHR intervention decision follows the same label-free protocol. A paper citation identifies the backbone model, whereas local code and artifact provenance establish the checkpoint rule; the DeSE label-aware checkpoint claim is therefore not attributed to the paper alone.

Table 8: Upstream provenance separated from SHR intervention. “GT” denotes ground-truth labels; $^ { 6 6 } \mathrm { V a l } ^ { 5 }$ denotes validation labels. UNKNOWN evidence is not silently interpreted as label-free.
<table><tr><td>Backbone</td><td>Datasets</td><td>Checkpoint rule</td><td>GT upstream</td><td>Val upstream</td><td>Native  $Q _ { 0 }$ </td><td>SHR labels</td></tr><tr><td>DeSE [1]</td><td>Cora/Citeseer/Photo</td><td>released paper-best, ACC-selected</td><td>YES</td><td>NO</td><td>YES</td><td>NO</td></tr><tr><td>HALO [2]</td><td>ACM/UAT/EAT/BAT</td><td>fixed final epoch</td><td>NO</td><td>NO</td><td>YES</td><td>NO</td></tr><tr><td>DGAC [3]</td><td>Cora/Citeseer</td><td>fixed final epoch</td><td>NO</td><td>NO</td><td>YES</td><td>NO</td></tr><tr><td>SynC [4]</td><td>Cora/Citeseer/ACM/UAT/DBLP</td><td>fixed final epochs</td><td>NO</td><td>NO</td><td>YES</td><td>NO</td></tr><tr><td>DGM [5]</td><td>Adam</td><td>AE epoch 200; DGM epoch 500</td><td>NO</td><td>NO</td><td>YES</td><td>NO</td></tr><tr><td>DBCD [6]</td><td>Cora</td><td>fixed final epoch</td><td>NO</td><td>NO</td><td>YES</td><td>NO</td></tr></table>

The numerical DBCD final epoch and exact released-code commits for DeSE, DGAC and DGM were not preserved in the current frozen provenance records. This does not alter the locally recorded selection rule, but limits commit-level reproducibility. Additional code provenance will be provided with the anonymized code repository.

## C.3 Boundary and Interface Cases

Strict-final DeSE Photo/Computers. For these strict-final anchors, the number of active clusters is smaller than the declared K, so the cluster-survival constraint returns no-action. These runs are not included in the primary efectiveness analysis; the resulting zero intervention gain does not imply zero baseline clustering performance.

DBCD–Cora. The native assignment and active-cluster interface do not satisfy the primary protocol requirements, so this combination is reported only as an interface boundary case.

Computers legacy readout. The development study uses a frozen DeSE embedding followed by an unsupervised KMeans readout. This is a legacy, non-native diagnostic interface: it is reported separately, excluded from the native DeSE mean, and not used to support a four-dataset same-interface claim. Replaying the historical DeSE development baseline on the native Cora/Citeseer/Photo in terfaces yields a maximum absolute metric error of $4 . 3 \times 1 0 ^ { - 6 }$ ; the Computers readout does not fully reproduce the paper-reported Computers row.

Negative boundary behavior. SynC–UAT remains a negative primary case with a macro gain of $- 0 . 0 4 1 \mathrm { p p }$ , illustrating that satisfying the interface conditions does not guarantee positive refinement gain. DGAC–BAT provides an additional negative outcome under aggressive coverage and is reported separately from the primary 15-combination analysis.

Failed preregistered $4 \times 5$ matrix and common-suite derivation. The unified-matrix audit completed all 20 smoke cells and all 100 formal runs for DeSE, HALO, DGAC, and SynC on Cora, Citeseer, ACM, UAT, and DBLP. For DeSE on Cora, ACM, and UAT, at least three seeds had fewer native active clusters than the declared $K .$ , so the complete $4 \times 5$ matrix failed the preregistered native-interface validity gate. The subsequent common suite retains only HALO, DGAC, and $\mathrm { S y n C } .$ each of which has a valid native $Q _ { 0 }$ on all five datasets. This subset is determined by interface validity rather than SHR gain. Table 9 retains the complete collapse counts and inclusion decisions.

Table 9: Technical audit of the attempted preregistered $4 \times 5$ matrix. All 20 smoke cells and 100 formal runs completed. DeSE failed the native-interface validity gate on three datasets because at least three of five runs had fewer active native clusters than the declared $K .$ . The subsequent $3 \times 5$ common suite is therefore interface-validity-driven rather than performance-selected.
<table><tr><td>Backbone</td><td>Dataset</td><td>Completed</td><td>Collapsed</td><td>Valid interface</td><td>Common-suite decision</td></tr><tr><td>DeSE</td><td>Cora</td><td>5</td><td>5</td><td>No</td><td>Excluded: interface invalid</td></tr><tr><td>DeSE</td><td>Citeseer</td><td>5</td><td>2</td><td>Yes</td><td>Not retained: backbone incomplete across suite</td></tr><tr><td>DeSE</td><td>ACM</td><td>5</td><td>4</td><td>No</td><td>Excluded: interface invalid</td></tr><tr><td>DeSE</td><td>UAT</td><td>5</td><td>5</td><td>No</td><td>Excluded: interface invalid</td></tr><tr><td>DeSE</td><td>DBLP</td><td>5</td><td>2</td><td>Yes</td><td>Not retained: backbone incomplete across suite</td></tr><tr><td>HALO</td><td>All five</td><td>25</td><td>0</td><td>Yes</td><td>Included by interface validity</td></tr><tr><td>DGAC</td><td>All five</td><td>25</td><td>0</td><td>Yes</td><td>Included by interface validity</td></tr><tr><td> $\mathrm { S y n C }$ </td><td>All five</td><td>25</td><td>0</td><td>Yes</td><td>Included by interface validity</td></tr></table>

## D Additional Results and Robustness

## D.1 Common-Suite Statistics and Post-Processing Cost

Table 10 reports complete cell-level statistics for the common suite. All five paired seeds are retained in every cell; the cell-level bootstrap intervals are descriptive for these small samples and are not used for individual-cell significance claims. Table 11 reports the changed-node count, change ratio, changed nodes per 1,000, and post-processing runtime. Timing begins after $A , X , Z _ { 0 }$ , and $Q _ { 0 }$ have been prepared and ends when formal SHR returns $Q ^ { * }$ . Each unit receives one warm-up and three timed repetitions, and backbone training is excluded. All 75 timed outputs exactly match their corresponding frozen SHR outputs.

Table 10: Cell-level common-suite macro-gain statistics. Values are percentage points. Intervals are descriptive percentile intervals from 10,000 paired seed-level bootstrap resamples (n = 5 per cell; seed 4102026). Counts give positive/exact-zero/negative frozen runs.
<table><tr><td>Backbone</td><td>Dataset</td><td>Mean ± SD</td><td>Median</td><td>95% CI</td><td></td><td>+/0/-</td><td></td><td></td><td>n</td></tr><tr><td>HALO</td><td>Cora</td><td>+0.050 ± 0.015</td><td>+0.054</td><td>[+0.036, +0.058]</td><td>5/0/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>HALO</td><td>Citeseer</td><td>+0.083 ± 0.074</td><td>+0.055</td><td>[+0.033, +0.148]</td><td>5/0/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>HALO</td><td>ACM</td><td>+0.170 ± 0.107</td><td>+0.114</td><td>[+0.095, +0.264]</td><td>5/0/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>HALO</td><td>UAT</td><td>+0.181 ± 0.235</td><td>+0.263</td><td>[-0.002, +0.362]</td><td>3 / 1 / 1</td><td></td><td></td><td></td><td>5</td></tr><tr><td>HALO</td><td>DBLP</td><td>+0.004 ± 0.010</td><td>0.000</td><td>[0.000, +0.013]</td><td>1/4/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>DGAC</td><td>Cora</td><td>+0.115 ± 0.078</td><td>+0.124</td><td>[+0.054, +0.170]</td><td>4/1/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>DGAC</td><td>Citeseer</td><td>0.000 ± 0.000</td><td>0.000</td><td>[0.000, 0.000]</td><td>0/5/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>DGAC</td><td>ACM</td><td>+0.008 ± 0.011</td><td>0.000</td><td>[0.000, +0.017]</td><td>2/3 /0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>DGAC</td><td>UAT</td><td>+0.096 ± 0.137</td><td>+0.062</td><td>[+0.012, +0.216]</td><td>3 /2 /0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>DGAC</td><td>DBLP</td><td>+0.011 ± 0.015</td><td>+0.017</td><td>[0.000, +0.022]</td><td>3 / 1 / 1</td><td></td><td></td><td></td><td>5</td></tr><tr><td>SynC</td><td>Cora</td><td>+0.224 ± 0.097</td><td>+0.184</td><td>[+0.150, +0.302]</td><td>5/0/0</td><td></td><td></td><td></td><td>5</td></tr><tr><td>SynC</td><td>Citeseer</td><td>+0.002 ± 0.029</td><td>0.000</td><td>[-0.021, +0.027]</td><td>1 / 3 / 1</td><td></td><td></td><td></td><td>5</td></tr><tr><td>SynC</td><td>ACM</td><td>+0.033 ± 0.065</td><td>+0.011</td><td>[-0.012, +0.088]</td><td>3/0/2</td><td></td><td></td><td></td><td>5</td></tr><tr><td>SynC</td><td>UAT</td><td>-0.042 ± 0.139</td><td>-0.004</td><td>[-0.157, +0.055]</td><td>2 /0 / 3</td><td></td><td></td><td></td><td>5</td></tr><tr><td>SynC</td><td>DBLP</td><td> $+ 0 . 0 6 3 \pm 0 . 1 2 1$ </td><td>+0.046</td><td>[-0.022, +0.163]</td><td>4 / 0 / 1</td><td></td><td></td><td></td><td>5</td></tr></table>

Table 11: Detailed intervention scale and runtime for the 15 common-suite cells. Change statistics summarize five paired frozen runs; runtime is the median of the five per-unit median timings.
<table><tr><td>Backbone</td><td>Dataset</td><td>Changed nodes</td><td>Change (%)</td><td>Range (%)</td><td>Changed/1000</td><td>Time (s)</td></tr><tr><td>HALO</td><td>Cora</td><td>9.4</td><td>0.347</td><td>0.222-0.591</td><td>3.47</td><td>1.74</td></tr><tr><td>HALO</td><td>Citeseer</td><td>5.4</td><td>0.162</td><td>0.090-0.301</td><td>1.62</td><td>2.15</td></tr><tr><td>HALO</td><td>ACM</td><td>4.8</td><td>0.159</td><td>0.066-0.331</td><td>1.59</td><td>1.53</td></tr><tr><td>HALO</td><td>UAT</td><td>4.8</td><td>0.403</td><td>0.000-0.588</td><td>4.03</td><td>0.92</td></tr><tr><td>HALO</td><td>DBLP</td><td>0.2</td><td>0.005</td><td>0.000-0.025</td><td>0.05</td><td>1.98</td></tr><tr><td>DGAC</td><td>Cora</td><td>9.2</td><td>0.340</td><td>0.000-0.554</td><td>3.40</td><td>1.76</td></tr><tr><td>DGAC</td><td>Citeseer</td><td>0.0</td><td>0.000</td><td>0.000-0.000</td><td>0.00</td><td>2.13</td></tr><tr><td>DGAC</td><td>ACM</td><td>2.4</td><td>0.079</td><td>0.000-0.298</td><td>0.79</td><td>1.50</td></tr><tr><td>DGAC</td><td>UAT</td><td>1.8</td><td>0.151</td><td>0.000-0.504</td><td>1.51</td><td>0.74</td></tr><tr><td>DGAC</td><td>DBLP</td><td>3.0</td><td>0.074</td><td>0.000-0.173</td><td>0.74</td><td>2.04</td></tr><tr><td>SynC</td><td>Cora</td><td>17.0</td><td>0.628</td><td>0.517-0.739</td><td>6.28</td><td>1.68</td></tr><tr><td>SynC</td><td>Citeseer</td><td>1.0</td><td>0.030</td><td>0.000-0.120</td><td>0.30</td><td>2.04</td></tr><tr><td>SynC</td><td>ACM</td><td>4.8</td><td>0.159</td><td>0.066-0.231</td><td>1.59</td><td>1.46</td></tr><tr><td>SynC</td><td>UAT</td><td>4.6</td><td>0.387</td><td>0.168-0.588</td><td>3.87</td><td>1.00</td></tr><tr><td>SynC</td><td>DBLP</td><td>8.6</td><td>0.212</td><td>0.099-0.345</td><td>2.12</td><td>1.90</td></tr></table>

## D.2 Comparator Details

Table 12 summarizes the coverage, strength, state-averaging scheme, and experimental role of each comparison method under the shared candidate residual; the corresponding state spaces and com putational rules are specified below.

Global-A is defined as $Q ^ { A } ~ = ~ \mathrm { s o f t m a x } ( \log ( Q _ { 0 } + 1 0 ^ { - 8 } ) + 0 . 1 \Delta )$ and uses neither a posterior nor the output of another method. For Strength-Bayes, the nonzero states fix

Table 12: Comparison methods under the shared candidate residual. CM-Global inherits its refinement coverage from CS-BAYES and is therefore used only as a matched-coverage diagnostic.
<table><tr><td>Method</td><td>Coverage</td><td>Strength</td><td>State averaging</td><td>Standalone</td><td>Role</td></tr><tr><td>Global-A</td><td>Global</td><td>Fixed η = .1</td><td>None</td><td>Yes</td><td>Fixed-strength baseline</td></tr><tr><td>Strength-Bayes</td><td>π = 1 if active</td><td>Eight η states</td><td>Posterior average + no-action</td><td>Yes</td><td>Strength uncertainty</td></tr><tr><td>CS-BAYES</td><td>Nine positive π states</td><td>Eight η states</td><td>Joint  ${ \pi } / { \eta } / { z _ { i } }$  average</td><td>Yes</td><td>Coverage-strength comparator</td></tr><tr><td>SHR</td><td>Effect-state proposal + node filter</td><td>Effect-calibrated</td><td>Evidence-weighted + no-action</td><td>Yes</td><td>Proposed coverage- conservative method</td></tr><tr><td>CM-Global</td><td>Inherits CS-BAYES count</td><td>Coverage-matched search</td><td>None</td><td>No</td><td>Diagnostic only</td></tr></table>

$\pi ~ = ~ 1$ and use $\eta ~ \in ~ \{ 0 . 7 , 1 , 1 . 5 , 2 , 3 , 5 , 7 , 1 0 \}$ ; for CS-BAYES, the positive-coverage states are $\pi \in \{ . 0 1 , . 0 3 , . 0 5 , . 1 , . 2 , . 4 , . 6 , . 8 , 1 \}$ , with $\gamma = 0$ fixed. Both share the no-action prior $p _ { 0 } = . 1$ and the discrete strength prior

$$
\begin{array} { r } { \log p ( \eta _ { e } ) = - \frac { 1 } { 2 } ( \eta _ { e } / 5 ) ^ { 2 } - \log \mathrm { s u m e x p } _ { e ^ { \prime } } [ - \frac { 1 } { 2 } ( \eta _ { e ^ { \prime } } / 5 ) ^ { 2 } ] . } \end{array}\tag{37}
$$

Let $\ell _ { i , e }$ denote the local label-free log-evidence for node i at strength e. The run-level generalized evidence is $\begin{array} { r } { T _ { N } \sum _ { i } \log [ ( 1 - \pi ) + \pi \exp ( \ell _ { i , e } ) ] } \end{array}$ , where $T _ { N } = \mathrm { m i n } ( 1 , 5 1 2 / N )$ . Strength-Bayes fixes the positive states at $\pi = 1$ ; CS-BAYES performs posterior-style averaging over $( \pi , \eta )$ and the conditional node keep/action probabilities rather than using the posterior mode. Both then evaluate the maximum change rate, mean JS, and active-cluster survival along the fixed shrinkage path $\tau \in \{ 1 , . 7 5 , . 5 , . 2 5 , 0 \}$

CM-Global first reads the changed-node count k produced by CS-BAYES for the same run, and then selects, from $\eta = 0$ and [0.01, 20] sampled on an 80-point geometric grid, the global residual state with the smallest lexicographic key $( | \# \mathrm { c h a n g e d } - k | , \overline { { \mathrm { J S } } } )$ CM-Global therefore cannot determine coverage independently and serves only as a same-coverage diagnostic.

## D.3 DeSE Development Results

The historical DeSE development baseline records the development-stage improvement space observed when post-processing frozen DeSE outputs. On Cora, Citeseer, and Photo under the native soft-assignment interface, its three-dataset mean macro gain over the frozen outputs is 1.195 pp. Reexecuting the saved historical procedure yields a maximum absolute reproduction error of $4 . 3 \times 1 0 ^ { - 6 }$ among the reported metrics. This development gain is improvement space observed in the historical development procedure, not a theoretical performance upper bound. Complete values are listed in Table 13. The Computers row uses the legacy non-native readout described in Appendix C.3 and is excluded from the mean over the three native-interface datasets.

Table 13: DeSE development results on native assignments and the legacy Computers readout.
<table><tr><td colspan="7">Historical DeSE development baseline on native frozen assignments</td></tr><tr><td>Dataset</td><td>Role</td><td>∆ACC</td><td>∆NMI</td><td>∆ARI</td><td>∆F1</td><td>Changed</td></tr><tr><td>Cora</td><td>development</td><td>.148</td><td>.566</td><td>-.018</td><td>.293</td><td>74</td></tr><tr><td>Citeseer</td><td>development</td><td>.240</td><td>.249</td><td>.199</td><td>.404</td><td>116</td></tr><tr><td>Photo</td><td>development</td><td>2.235</td><td>1.703</td><td>3.679</td><td>4.645</td><td>463</td></tr><tr><td>Native mean</td><td>development</td><td>.874</td><td>.839</td><td>1.287</td><td>1.780</td><td></td></tr><tr><td></td><td>Legacy non-native readout (excluded from native mean)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Computers</td><td>frozen embedding + KMeans</td><td>1.825</td><td>.819</td><td>1.454</td><td>.624</td><td>1315</td></tr></table>

## D.4 Complete SHR Results

Table 14 reports the combination-level absolute results for the fixed 15-combination primary registry. It is aggregated directly from the existing frozen records for 113 units by taking the arithmetic mean over frozen units within each backbone–dataset combination. Strict-final degenerate anchors, the DBCD interface boundary, the Computers legacy readout, and additional aggressive-control diagnostics are excluded from this table.

Table 14: Complete absolute results for SHR across the 15 primary backbone–dataset combinations (Part I: frozen-baseline and final SHR metrics). Absolute metrics are reported on the [0, 1] scale, and each row is the descriptive arithmetic mean over the corresponding frozen units. Depending on the backbone, units are seeds, folds, or a single frozen checkpoint; therefore, the DeSE rows with n = 1 do not estimate run-to-run uncertainty. The frozen baseline denotes the locally frozen output rather than a directly transcribed paperreported score. The upstream selection of the DeSE checkpoints used labels; the label-free constraint applies only to the SHR stage after checkpoint fixation.
<table><tr><td colspan="3"></td><td colspan="4">Frozen baseline</td><td colspan="4">SHR final</td></tr><tr><td>Backbone</td><td>Dataset</td><td>n</td><td>ACC</td><td>NMI</td><td>ARI</td><td>F1</td><td>ACC</td><td>NMI</td><td>ARI</td><td>F1</td></tr><tr><td>DeSE</td><td>Cora</td><td>1</td><td>0.7518</td><td>0.5703</td><td>0.5270</td><td>0.7099</td><td>0.7526</td><td>0.5716</td><td>0.5275</td><td>0.7108</td></tr><tr><td>DeSE</td><td>Citeseer</td><td>1</td><td>0.6889</td><td>0.4398</td><td>0.4474</td><td>0.6399</td><td>0.6886</td><td>0.4394</td><td>0.4468</td><td>0.6397</td></tr><tr><td>DeSE</td><td>Photo</td><td>1</td><td>0.8046</td><td>0.6901</td><td>0.6241</td><td>0.7318</td><td>0.8055</td><td>0.6917</td><td>0.6252</td><td>0.7378</td></tr><tr><td>DGAC</td><td>Cora</td><td>5</td><td>0.6393</td><td>0.5339</td><td>0.4381</td><td>0.6344</td><td>0.6405</td><td>0.5351</td><td>0.4397</td><td>0.6350</td></tr><tr><td>DGAC</td><td>Citeseer</td><td>5</td><td>0.6909</td><td>0.4350</td><td>0.4450</td><td>0.6444</td><td>0.6909</td><td>0.4350</td><td>0.4450</td><td>0.6444</td></tr><tr><td>DGM</td><td>Adam</td><td>10</td><td>0.7518</td><td>0.7589</td><td>0.6851</td><td>0.7002</td><td>0.7517</td><td>0.7589</td><td>0.6851</td><td>0.7001</td></tr><tr><td>HALO</td><td>ACM</td><td>10</td><td>0.8125</td><td>0.5070</td><td>0.5466</td><td>0.8112</td><td>0.8137</td><td>0.5090</td><td>0.5488</td><td>0.8124</td></tr><tr><td>HALO</td><td>BAT</td><td>10</td><td>0.5237</td><td>0.2911</td><td>0.2184</td><td>0.4962</td><td>0.5290</td><td>0.2992</td><td>0.2254</td><td>0.5003</td></tr><tr><td>HALO</td><td>EAT</td><td>10</td><td>0.4739</td><td>0.1926</td><td>0.1690</td><td>0.4617</td><td>0.4757</td><td>0.1981</td><td>0.1728</td><td>0.4633</td></tr><tr><td>HALO</td><td>UAT</td><td>10</td><td>0.4861</td><td>0.2221</td><td>0.1861</td><td>0.4552</td><td>0.4882</td><td>0.2243</td><td>0.1883</td><td>0.4574</td></tr><tr><td>SynC</td><td>ACM</td><td>10</td><td>0.9148</td><td>0.7023</td><td>0.7628</td><td>0.9151</td><td>0.9152</td><td>0.7030</td><td>0.7639</td><td>0.9155</td></tr><tr><td>SynC</td><td>Citeseer</td><td>10</td><td>0.6951</td><td>0.4415</td><td>0.4525</td><td>0.6449</td><td>0.6950</td><td>0.4415</td><td>0.4525</td><td>0.6447</td></tr><tr><td>SynC</td><td>Cora</td><td>10</td><td>0.6613</td><td>0.4978</td><td>0.4292</td><td>0.6494</td><td>0.6635</td><td>0.5008</td><td>0.4325</td><td>0.6514</td></tr><tr><td>SynC</td><td>DBLP</td><td>10</td><td>0.8042</td><td>0.5054</td><td>0.5595</td><td>0.7961</td><td>0.8046</td><td>0.5065</td><td>0.5604</td><td>0.7964</td></tr><tr><td>SynC</td><td>UAT</td><td>10</td><td>0.5561</td><td>0.2601</td><td>0.2481</td><td>0.5489</td><td>0.5557</td><td>0.2596</td><td>0.2480</td><td>0.5484</td></tr></table>

Table 14: Complete absolute results for SHR (continued: metric gains relative to the frozen baseline, macro gain, and hard-assignment change ratio). Metric gains and macro gain are reported in percentage points; Change is reported as a percentage. The valid no-action result for DGAC–Citeseer and all negative combinations are retained exactly as frozen. Machine-readable records for every frozen unit and unrounded combination-level values will be released with the code repository.
<table><tr><td>Backbone</td><td>Dataset</td><td>n</td><td>∆ACC</td><td>∆NMI</td><td>∆ARI</td><td>∆F1</td><td>Macro</td><td>Change (%)</td></tr><tr><td>DeSE</td><td>Cora</td><td>1</td><td>+0.074</td><td>+0.126</td><td>+0.047</td><td>+0.093</td><td>+0.085</td><td>0.222</td></tr><tr><td>DeSE</td><td>Citeseer</td><td>1</td><td>-0.030</td><td>-0.037</td><td>-0.062</td><td>-0.023</td><td>-0.038</td><td>0.180</td></tr><tr><td>DeSE</td><td>Photo</td><td>1</td><td>+0.092</td><td>+0.158</td><td>+0.106</td><td>+0.597</td><td>+0.238</td><td>0.641</td></tr><tr><td>DGAC</td><td>Cora</td><td>5</td><td>+0.118</td><td>+0.120</td><td>+0.161</td><td>+0.060</td><td>+0.115</td><td>0.340</td></tr><tr><td>DGAC</td><td>Citeseer</td><td>5</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>DGM</td><td>Adam</td><td>10</td><td>-0.008</td><td>+0.000</td><td>+0.000</td><td>-0.009</td><td>-0.004</td><td>0.030</td></tr><tr><td>HALO</td><td>ACM</td><td>10</td><td>+0.116</td><td>+0.200</td><td>+0.222</td><td>+0.119</td><td>+0.164</td><td>0.162</td></tr><tr><td>HALO</td><td>BAT</td><td>10</td><td>+0.534</td><td>+0.807</td><td>+0.699</td><td>+0.411</td><td>+0.613</td><td>1.221</td></tr><tr><td>HALO</td><td>EAT</td><td>10</td><td>+0.175</td><td>+0.552</td><td>+0.376</td><td>+0.159</td><td>+0.316</td><td>1.103</td></tr><tr><td>HALO</td><td>UAT</td><td>10</td><td>+0.210</td><td>+0.214</td><td>+0.214</td><td>+0.224</td><td>+0.215</td><td>0.395</td></tr><tr><td>SynC</td><td>ACM</td><td>10</td><td>+0.043</td><td>+0.070</td><td>+0.109</td><td>+0.042</td><td>+0.066</td><td>0.139</td></tr><tr><td>SynC</td><td>Citeseer</td><td>10</td><td>-0.009</td><td>+0.008</td><td>+0.002</td><td>-0.012</td><td>-0.003</td><td>0.042</td></tr><tr><td>SynC</td><td>Cora</td><td>10</td><td>+0.218</td><td>+0.292</td><td>+0.330</td><td>+0.206</td><td>+0.261</td><td>0.654</td></tr><tr><td>SynC</td><td>DBLP</td><td>10</td><td>+0.042</td><td>+0.117</td><td>+0.087</td><td>+0.039</td><td>+0.071</td><td>0.197</td></tr><tr><td>SynC</td><td>UAT</td><td>10</td><td>-0.034</td><td>-0.055</td><td>-0.016</td><td>-0.059</td><td>-0.041</td><td>0.303</td></tr></table>

## D.5 Aggregation and Backbone Sensitivity

Table 15 reports the two-stage aggregation: datasets are first averaged within each backbone, and the backbone-level means are then weighted equally; combination-equal remains the primary statistic. For macro gain, SHR−CS-BAYES is −0.228 pp, with a 95% backbone-bootstrap CI of

Table 15: Equal-backbone sensitivity (gains in percentage points). Summary rows average the five backbonelevel means equally.
<table><tr><td>Method</td><td>ΔACC</td><td>∆NMI</td><td>∆ARI</td><td>∆F1</td><td>Macro</td></tr><tr><td>Global-A</td><td>.023</td><td>.098</td><td>.077</td><td>.047</td><td>.061</td></tr><tr><td>Strength-Bayes</td><td>.133</td><td>.343</td><td>.398</td><td>.135</td><td>.252</td></tr><tr><td>CS-BAYES</td><td>.161</td><td>.542</td><td>.515</td><td>.132</td><td>.337</td></tr><tr><td>SHR</td><td>.081</td><td>.134</td><td>.118</td><td>.103</td><td>.109</td></tr></table>

[−0.681, 0.118]. For negative-run rate, the diference is −2.13 percentage points, with a 95% CI of [−26.8, 22.0]. SHR itself has an equal-backbone macro gain of 0.109 pp (exploratory 95% bootstrap CI [0.031, 0.222] pp). With only five bootstrap units, these are exploratory sensitivity intervals rather than confirmatory evidence.

## D.6 Detailed Gain–Risk Comparison

Table 16 preserves the complete numerical gain–risk comparison moved from the main text. Standalone methods use the combination-equal risk estimand; CM-Global inherits the coverage of CS-BAYES and is only a matched-coverage mechanism diagnostic.

Table 16: Gain–risk summary under a shared residual. Standalone methods use combination-equal risk; CM Global is only a matched-coverage mechanism diagnostic. Catastrophic counts are reported over $N = 1 1 3$ frozen paired runs per method.
<table><tr><td colspan="7">Standalone interventions, 15-combination audit</td></tr><tr><td>Method</td><td>Macro gain</td><td>Eq.-comb. neg.</td><td>Pooled neg.</td><td>Neg. combos</td><td>Catastrophic</td><td>Worst combo</td></tr><tr><td>Global-A</td><td>.081</td><td>28.67%</td><td>24.78%</td><td>4/15</td><td>6/113</td><td>-.225</td></tr><tr><td>Strength-Bayes</td><td>.233</td><td>38.67%</td><td>39.82%</td><td>7/15</td><td>24/113</td><td>-.993</td></tr><tr><td>CS-BÃYES</td><td>.348</td><td>26.00%</td><td>30.09%</td><td>4/15</td><td>12/113</td><td>-.993</td></tr><tr><td>SHR</td><td>.137</td><td>23.33%</td><td>23.01%</td><td>4/15</td><td>3/113</td><td>-.041</td></tr><tr><td colspan="7">Panel C: Coverage-matched mechanism diagnostic (not standalone)</td></tr><tr><td>CM-Global</td><td>.318</td><td>37.33%</td><td>37.17%</td><td>5/15</td><td>16/113</td><td>-.982</td></tr><tr><td>CS-BAYES</td><td>.348</td><td>26.00%</td><td>30.09%</td><td>4/15</td><td>12/113</td><td>-.993</td></tr><tr><td>CS - CM</td><td>+.030</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## D.7 Additional Statistical Comparison with CS-BAYES

Across the 15 combination-level paired means, the macro-gain diference between SHR and CS-BAYES yields a two-sided Wilcoxon signed-rank statistic of 46.0 $( p = 0 . 4 5 4 )$ and a two-sided exact sign-test $p = 0 . 6 0 7$ . For the within-combination negative-run rates, the Wilcoxon statistic is 24.0 $( p = 0 . 4 2 1 )$ , and the exact sign-test gives $p = 1 . 0 0 0$ . These auxiliary tests are consistent with the paired bootstrap results in the main text: the current data support neither a universal advantage for either method nor a description of SHR as statistically safer.

## E Additional Diagnostics and Ablation

## E.1 Additional Identifiability Diagnostics

Diagnostic candidate set and oracle. This diagnostic includes only backbone–dataset combinations with at least one frozen unit whose final SHR changed-node count is nonzero. The resulting 10 combinations are DGAC–Cora; HALO–ACM, HALO–BAT, HALO–EAT, and HALO–UAT; and SynC–ACM, SynC–Citeseer, SynC–Cora, SynC–DBLP, and SynC–UAT. DGAC–Citeseer is excluded from this subset because all five frozen units return no-action. The candidate universe consists of eligible residual candidates with finite $\eta _ { \mathrm { c r i t } }$ not exceeding the frozen upper bound of the η search, and each target class is fixed to the residual-specified $b _ { \mathrm { c r i t } }$ . Within each run, the oracle selects k nodes, where k is the final SHR hard-assignment change count. The corresponding global result selects the first k candidates ordered by $\eta _ { \mathrm { c r i t } }$ and then node ID, again using $b _ { \mathrm { c r i t } }$ . The macro oracle first forms a baseline label–cluster contingency table and groups candidates by true-label row, baseline cluster, and $b _ { \mathrm { c r i t } }$ target, with candidates within each group ordered by $\eta _ { \mathrm { c r i t } }$ and node ID. It then adds the feasible move that maximizes the four-metric macro score at each step and performs up to 20 deterministic one-swap passes, accepting only swaps that improve the macro score by more than $1 0 ^ { - 1 2 }$ . This greedy-plus-one-swap solution is a feasible lower bound within the fixed candidate directions and is not guaranteed to be the combinatorial optimum. It uses labels only after output freezing, generates no new direction, does not enter SHR, and is not deployable. Let g<sub>oracle</sub>, g<sub>global</sub>, and g<sub>SHR</sub> denote their respective macro gains. Figure 8 uses

$$
H _ { \mathrm { o r a c l e - g l o b a l } } = g _ { \mathrm { o r a c l e } } - g _ { \mathrm { g l o b a l } } , \qquad \mathrm { R e c o v e r y } = { \frac { g _ { \mathrm { S H R } } - g _ { \mathrm { g l o b a l } } } { g _ { \mathrm { o r a c l e } } - g _ { \mathrm { g l o b a l } } } } ,
$$

with Recovery recorded as NA when the denominator is nonpositive. Nine of the 10 included combinations have combination-level mean headroom of at least 0.10 pp.

Combined-LCB and repair–harm AUC. The node-level Combined-LCB used in Section 4.5 and the frozen score re-audited here share the same label-free definition. For each of five evidence repeats, target-direction topology and attribute supports are combined as $\begin{array} { r } { u _ { i } ^ { ( t ) } = \frac { 1 } { 2 } ( u _ { i , \mathrm { t o p o } } ^ { ( t ) } + u _ { i , \mathrm { a t t r } } ^ { ( t ) } ) } \end{array}$ followed by the lower confidence bound ${ { \overline { { u } } } _ { i } } \mathrm { ~ - ~ } 1 . 6 4 5 { s _ { i } } / \sqrt { 5 } .$ To align the score direction with the residual-fixed repair/harm target, the reported AUC of 0.684 uses the frozen current-candidate LCB and retains only eligible decisive candidates whose stored target equals $b _ { \mathrm { c r i t } }$ . Repair and harm labels follow the fixed baseline mapping in Appendix B.3, and neutral nodes are excluded from AUC. ROC–AUC is computed within a frozen unit only when at least two observations, both repair and harm classes, and a nonconstant finite score are available; otherwise it is NA. Valid frozen-unit AUCs are first averaged within each backbone–dataset combination, and the resulting values are averaged equally over the 10 included combinations, yielding 0.684.

Residual-direction consensus. For every frozen run, four types of mild label-free perturbation are generated: 5% edge dropout, 5% feature masking, feature noise with standard deviation equal to 0.05 times the global feature standard deviation, and 5% hypergraph-neighbor membership perturbation. Each type has 10 fixed repetitions, for T = 40 perturbations. The auxiliary residual is reconstructed for every perturbation, and the earliest target state $b _ { i } ^ { ( t ) }$ is recorded; the absence of a finite switching strength is represented as a distinct no-switch state. Node-level directional consistency is

$$
\mathrm { d i r e c t i o n \_ c o n s e n s u s } _ { i } = \operatorname* { m a x } _ { s \in \{ - 1 , 0 , \ldots , K - 1 \} } \frac { 1 } { 4 0 } \sum _ { t = 1 } ^ { 4 0 } \mathbf { 1 } [ b _ { i } ^ { ( t ) } = s ] .
$$

Accepted-harm and missed-repair nodes are pooled according to Appendix B.3. Their pooled nodelevel medians are 0.962 (n = 118) and 0.850 (n = 11,637), respectively. Here, n denotes diagnostic nodes rather than independent runs; within-node and within-run dependence means that these descriptive distributions are not independent-sample inference.

Cross-system diagnostics. The supervised cross-system probes use only repair and harm candidates, with repair coded as 1 and harm as 0; neutral nodes are excluded. Leave-one-dataset-out (LODO) holds out one complete backbone–dataset combination and trains on the other nine combinations, whereas leave-one-backbone-out (LOBO) holds out one complete backbone (DGAC, HALO, or SynC). The full XGBoost probe uses 28 frozen label-free features: baseline logit margin; residual directional advantage; $\eta _ { \mathrm { c r i t } } ;$ Combined-LCB for the current, $b _ { \mathrm { c r i t } }$ , and action targets; topology target support; attribute target support; their support gap and source agreement; topology action sup port; attribute action support; candidate and actual-SHR assignment JS; baseline entropy; baseline maximum confidence; cluster size; feature-kNN target agreement; graph-neighbor target agreement; direction consensus; original-target survival; the coeficient of variation and finite fraction of $\eta _ { \mathrm { c r i t } } ;$ residual-direction margin; and the direction-consensus scores under edge dropout, feature masking, feature noise, and hypergraph perturbation. Missing values receive training-set median imputation, and training weights are the normalized inverse sample counts within each dataset–class group. XGBoost is fixed at 300 trees, depth 3, learning rate 0.05, row and column subsampling of 0.8, $\lambda = 1$ , and random seed 20260807, with no tuning. Average precision is computed over all held-out repair/harm nodes. Operating precision ranks candidates within each held-out run, selects the top $k ,$ and divides the repair count by that run’s observed SHR changed count k before averaging over runs. The separate single-signal tail analysis retains repair, harm, and neutral candidates and evaluates operating-k, 0.1%, 0.25%, 0.5%, 1%, 2%, and top-5/10/20 cutofs; neutral candidates remain in the Precision@k denominator. Relative to the current Combined-LCB baseline, LODO XGBoost has mean ∆AP of approximately +0.002 and mean $\Delta$ operating precision of approximately −0.006. Only 3 of 10 holdouts satisfy the preregistered joint-win rule, and LOBO results are likewise unstable. All supervised probes are post hoc diagnostics performed after method outputs are frozen. They are not part of unsupervised SHR and must not be interpreted as unsupervised generalization performance.

Figure 8 summarizes the oracle-headroom, residual-direction-consensus, repair–harm AUC, and cross-system probe results reported above.

## a Headroom exists, recovery remains weak

![](images/2d6ae16b43015ec70cacf16b26959436ffbcf258cacde91771cceac1b827b5c5.jpg)

b Wrong directions can be stable  
![](images/9a1dc434f1987b218cb4ae6f06ed505fc068325cd9ba49366a7bff8cb544ed81.jpg)

c Ranking signals vary by system  
![](images/ab6cbda438a96a2266869a9d69fa8b6f14a166723af0085720812ccf25e7f88f.jpg)

d Cross-dataset probe is inconsistent  
![](images/a7dad6d9ed53005f87dc0158cd65877530e32a247d5ab5ac949be7ff89d8dcf3.jpg)  
Δ average precision Δ operating precision  
a, marker area scales with oracle–global headroom; blue intensity encodes SHR/global Jaccard. d, post-hoc diagnostic only.  
Figure 8: Post hoc diagnosis of label-free selection capability. (a) Additional improvement space of the oracle over the global refinement result and its recovery by SHR for the included combinations. (b) Residual direction consensus for nodes that were accepted but ultimately became incorrect and for unaccepted nodes that could have been repaired. (c) Repair–harm ROC–AUC for $- \eta _ { \mathrm { c r i t } }$ and Combined-LCB. (d) Changes relative to the current LCB baseline in cross-dataset diagnosis. Repair, harm, oracle, and AUC are computed with labels only after method outputs are frozen and do not enter SHR decisions.

## E.2 Additional Source-Ablation Protocol Details

The auxiliary-source experiment fixes $Q _ { 0 }$ , state calibration, matched-null, coverage constraints, LCB, and provenance execution, while replacing only the relational source used to construct the residual. The comparisons include original-graph propagation, an attribute-based cosine 10-NN graph, a single-scale attribute hypergraph, the full multiscale attribute hypergraph, and a randomized (shufled) hypergraph that preserves node incidence degree and hyperedge size. The five shufled hypergraphs are averaged within the same frozen run and are not treated as independent runs. All sources use the same target coverage-state calibration protocol, but projection and node filtering can produce diferent final change ratios. The diferences between sources therefore represent the overall operational efect of replacing the information source under the same operational protocol, rather than a direct structural efect under strictly matched final coverage.

Across the 10 frozen runs of DGM–Adam, the prespecified common-dimension rule gives $d _ { c } = 2 6$ for the single-scale source and $d _ { c } = 3 2$ for all other sources. The comparison between the multiscale and single-scale sources is therefore only a descriptive scale diagnostic and cannot be interpreted as a pure scale efect under a strictly dimension-matched condition; this dimensional diference was not adjusted according to the evaluation results.

## E.3 Method Distillation and Simplification

Distillation uses the frozen historical 15-combination/136-run registry, in which the complete-SHR macro gain is 0.118 pp; the corresponding macro gain in the current primary 15-combination/113-run registry is 0.137 pp. These values have diferent scopes and must not be combined. Efect-only retains 40.6% of the observed gain of the complete configuration, but inherits its frozen changed-count budget and cannot be deployed independently. Among the simplified variants tested, No coverage/MDL, No matched-null, No node-level filtering + provenance veto, MAP, and Efect-only fail to preserve simultaneously the observed gain characteristics, negative-run behavior, and execution constraints of the complete SHR. Figure 9 presents a representative performance–risk–complexity profile.

No node-level filtering + provenance veto. This variant retains candidate-state construction, evidence assessment, state weighting, and global safety projection, and denotes by $Q _ { \mathrm { s a f e } }$ the candidate result that passes the global change-rate, mean-JS, and active-cluster constraints. Unlike complete SHR, it outputs $Q _ { \mathrm { s a f e } }$ directly and omits two final layers of selective control. First, it does not apply Combined-LCB screening to the actually changed nodes or the associated cluster-survival protection. Second, it does not apply run-level hard rejection based on the mean attribute support of retained nodes. Its macro gain, negative-run rate, and worst-combination result are +0.068 pp, 28.7%, and −0.209 pp, respectively, compared with +0.118 pp, 18.4%, and −0.041 pp for complete SHR. Under this historical protocol, the observed profile therefore combines lower mean gain, more negative runs, and a worse tail outcome; however, this ablation alone does not establish an independent causal role for either control layer.

a Performance  
![](images/ba9938e6e2dd1e215d81b9535447864b8cbfd0f26ae5982c0c0e48b8af31ba1f.jpg)  
Combination-equal macro gain (pp)

b Observed risk  
![](images/c0ee36a2370824900bd1f31c3ee58240200b63774c4cca805ef8b36d3789b11c.jpg)  
Negative-run rate (%)

c Boundary behavior  
![](images/0fab9d4f54c2caf2e82a59363f7830a34136ad1680481d25fb6c232707b7277c.jpg)  
Worst-combination macro gain (pp)  
† Effect-only inherits SHR's frozen changed-count budget and is not independently deployable. Red denotes values beyond the preregistered −0.25 pp catastrophic threshold.

Figure 9: Ablation profile of the frozen method-distillation registry. (a) Macro gain with equal weighting of backbone–dataset combinations; (b) run-weighted negative-run rate; (c) worst-combination macro gain and the catastrophic threshold defined as −0.25 pp, with the number of logical blocks shown on the right. No node-level filtering + provenance veto removes both Combined-LCB/cluster-survival screening and the attribute-provenance veto. Efect-only inherits the frozen changed-count budget of complete SHR and cannot be deployed independently. This figure compares the observed gain, negative-run behavior, and execution complexity of the simplified configurations and does not redefine the primary mean.