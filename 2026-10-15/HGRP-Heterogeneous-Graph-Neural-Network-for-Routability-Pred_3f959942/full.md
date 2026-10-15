# HGRP: Heterogeneous Graph Neural Network for Routability Prediction

Yule Sheng Nanjing University of Science and Technology Nanjing, China 124132011758@njust.edu.cn

Shan Shen Nanjing University of Science and Technology Nanjing, China shanshen@njust.edu.cn

Yue Yuan Nanjing University of Science and Technology Nanjing, China yuanyue1801@njust.edu.cn

Daying Sun Nanjing University of Science and Technology Nanjing, China hasdysun@njust.edu.cn

## Abstract

Routability prediction enables rapid evaluation ofplacement quality without routing, providing efective feedback for routability-driven placement optimization. Recent machine learning methods have achieved eficient routability prediction through data-driven approach. However, their prediction performance highly depends on the capability of input representations to capture essential circuit information. To address the limitations of existing circuit representations in simultaneously preserving logical topology, physical layout and routing resource information, this paper proposes HGRP, a heterogeneous graph neural network for routability prediction. HGRP models the cells, nets, and routing grids as distinct types of nodes and explicitly represents topological connectivity, spatial mapping, and grid adjacency, thereby constructing a heterogeneous graph representation that preserves multi-level circuit semantics. Based on the representation, a heterogeneous message passing mechanism is developed to enable information interaction and feature fusion among distinct types of nodes, efectively learning the complex correlations between circuit structures and routing behaviors. Experimental results on the CircuitNet dataset show that HGRP im proves F1-score by 18.69% relative improvements compared with the state-of-the-art in the congestion prediction task. The source code is publicly available at https://github.com/ShenShan123/HGPR.

## Keywords

Physical Design, Machine Learning, Routability Prediction

## ACM Reference Format:

Yule Sheng, Yue Yuan, Shan Shen, and Daying Sun. 2027. HGRP: Heterogeneous Graph Neural Network for Routability Prediction. In 32nd Asia and South Pacific Design Automation Conference (ASPDAC ’27), January 25–28, 2027, Tokyo, Japan. ACM, New York, NY, USA, 7 pages. https: //doi.org/10.1145/3840404.3848321

## 1 Introduction

As one of the most complex and time-consuming stages in physical design, if routability is not adequately considered early in the design flow, routing may sufer from congestion, design rule violations (DRVs), timing violations, and signal integrity issues. In such cases, the design flow often needs to return to the placement or even floorplan stage to adjust cell locations and rerun the routing process. Therefore, accurate routability prediction is essential for improving design quality and reducing design iterations.

Early routability-driven placement methods commonly rely on global routing [8] or trial global routing, where routing results are used as feedback to guide placement optimization. However, repeatedly invoking global router during placement optimization often introduces prohibitive computational overhead. In contrast, fast estimation methods based on net density, such as probabilistic congestion estimation and RUDY [10], can substantially reduce the cost. Nevertheless, these methods lack explicit modeling of routing path selection and competition for routing resource, making it dificult to accurately characterize actual local congestion.

Recent advances in machine learning provide solutions for rapid and accurate routability prediction, as large-scale EDA datasets [2] [1] have become increasingly available for model training and evaluation. Due to the inherent two dimensional spatial structure of physical layout, a typical approach is to formulate routability prediction as an image regression or image segmentation task. Methods such as RouteNet [13], J-Net [6], GPDL [7] and Pros [3] encode physical design information into multichannel feature maps and apply convolutional neural networks (CNNs) to predict routability at grid-level, thereby providing rapid feedback for routability-driven design. However, modeling circuits as images may weaken or even discard topological information, potentially limiting generalization across netlists with diferent connectivity structures.

Graph neural networks (GNNs) have been introduced into routability prediction to improve the representation of topological information. Since very large scale integrated (VLSI) circuits can be naturally represented as a graph, methods such as CongestionNet [5], NetlistGNN [14], HybridNet [15], and LHNN [12] learn circuit representations via GNNs to predict routability metrics such as congestion and DRVs. Graph-based approaches generally model circuits at the cell, net, or grid-level. However, existing methods often emphasize net connectivity to incorporate topological information and perform prediction at either the cell-level or the grid-level, while overlooking the influence of the spatial mapping between cells and routing grids on routability. The absence ofcell-level weakens the representation of cell-level structural information, whereas the absence of grid-level limits the capability to characterize routing resource competition and local congestion propagation. Conse quently, both types of graph constructions sufer from incomplete semantic representation.

To address these limitations, we propose HGRP, a Heterogeneous Graph neural network for Routability Prediction. HGRP represents circuits as heterogeneous graphs, and introduces a heterogeneous message passing mechanism coupled with an adaptive feature ag gregation strategy to update node representations. In this way, the model captures the complex coupling among circuit netlist, physical layout and routing resource. Moreover, the message passing order can be flexibly adjusted according to the dependency of different tasks on node representations, enabling HGRP to adapt to various physical design tasks beyond congestion prediction. The main contributions of this work are summarized as follows:

• We propose a heterogeneous graph representation for routability prediction, in which cells, nets, and routing grids are modeled as distinct types of nodes. Topological connectivity, spatial mapping, and grid adjacency relations are explicitly encoded, providing a more complete circuit representation for routability prediction.

• We develop a heterogeneous message passing mechanism that jointly learns routing demands induced by netlist topology and local congestion characteristics constrained by physical layout.

• We introduce an attention-based adaptive feature aggregation strategy that dynamically determines the contributions of neighboring nodes according to their node representations and edge attributes during message passing.

## 2 Problem Formulation

## 2.1 Congestion Prediction

Routing congestion is an important metric for evaluating the routability of placement solution. During global routing, the chip region is partitioned into a regular � × ℎ grid graph. Based on the netlist and cell placement, the global router estimates the routing paths of individual nets and assigns the routing demand to the corresponding grid regions. Congestion occurs when the demand within a grid exceeds its available routing capacity, potentially leading to increased routing detours, timing degradation, and even DRVs during detailed routing. Following the definition adopted in CircuitNet [2], the congestion of grid $g _ { i , j }$ is formulated as

$$
c o n g e s t i o n _ { i , j } = \frac { m a x ( d e m a n d _ { i , j } - c a p a c i t y _ { i , j } , 0 ) } { c a p a c i t y _ { i , j } }\tag{1}
$$

where ����� $d _ { i , j }$ and $c a p a c i t y _ { i , j }$ are the routing demand and capacity of $\cdot { g } _ { i , j } ,$ respectively, ��������� $n _ { i , j }$ measures the degree of routing congestion in the corresponding region and is clipped to zero when routing demand does not exceed capacity. The congestion of all grids are arranged according to their spatial locations to form the complete congestion map $\bar { \Upsilon } _ { c o n g } \in \mathbb { R } ^ { \bar { w } \times h }$ , and congestion prediction aims to learn a parameterized mapping $f _ { \theta , c o n g }$ from circuit information X to the congestion map.

$$
\hat { \mathbf { Y } } _ { c o n g } = f _ { \theta , c o n g } ( \mathbf { X } ) \in \mathbb { R } ^ { w \times h }\tag{2}
$$

## 2.2 Net Wirelength Prediction

Net wirelength is an important measure of subsequent routing cost, and is closely associated with interconnect delay, power consumption and routing-resource utilization. During placement, metrics such as half-perimeter wirelength (HPWL) are commonly used to rapidly estimate net length. However, HPWL does not explicitly account for routing layers, obstacles, design rules, resource competition or congestion-induced detours, therefore may deviate from the actual wirelength in the final physical implementation.

Predicting the final wirelength before detailed routing can facilitate the early identification of nets likely to require long routes, providing realistic feedback for cell repositioning, wirelength optimization and timing closure. Here, we use the actual post-detailedrouting wirelength as the prediction target. For a net $n _ { j } ,$ the detailed router generates its final routing solution that satisfies connectivity and design-rule constraints

$$
\mathcal { R } _ { j } = ( S _ { j } , \mathcal { U } _ { j } )\tag{3}
$$

where $S _ { j }$ denotes the set of routing segments of net $n _ { j }$ across all metal layers, and $\mathcal { U } _ { j }$ denotes the set of vias connecting diferent routing layers. The actual post-detailed-routing wirelength of net $n _ { j }$ is formulated as

$$
w i r e l e n g t h _ { j } = \sum _ { s \in S _ { j } } \ell ( s )\tag{4}
$$

where $\ell ( s )$ denotes the physical length of routed segment �. Net wirelength prediction aims to learn a parameterized mapping �<sub>�,�����ℎ</sub> form circuit information $\mathbf { X }$ to the actual post-detailed-routing wirelengths of all nets.

$$
\hat { \mathbf { Y } } _ { l e n g t h } = f _ { \theta , l e n g t h } ( \mathbf { X } )\tag{5}
$$

## 2.3 Evaluation Metrics

Since routability prediction is formulated as a grid-level congestion regression task in this work, we employ the normalized root mean square error (NRMSE) and structural similarity index measure (SSIM) to comprehensively evaluate model performance.

Definition 1 (NRMSE). NRMSE measures the normalized deviation between the predicted and ground-truth congestion.

$$
{ \mathrm { N R M S E } } = { \frac { 1 } { y _ { { \mathrm { m a x } } } - y _ { { \mathrm { m i n } } } } } \cdot { \sqrt { \frac { \sum _ { i = 1 } ^ { h } \sum _ { j = 1 } ^ { w } \left( y _ { i , j } - { \hat { y } } _ { i , j } \right) ^ { 2 } } { h \times w } } }\tag{6}
$$

where $y _ { i , j }$ and $\hat { y } _ { i , j }$ are the ground-truth and predicted congestion of grid $g _ { i , j } ,$ , ℎ and � are the height and width of the congestion map, $y _ { m a x }$ and $y _ { m i n }$ are the maximum and minimum ground-truth. The normalization eliminates the influence of scale diferences across circuit designs, thereby enabling fair performance comparisons. A lower NRMSE indicates higher numerical prediction accuracy.

Definition 2 (SSIM). SSIM measures the structural similarity between the predicted and ground-truth congestion maps.

$$
\mathrm { S S I M } = \frac { \big ( 2 \mu _ { y } \mu _ { \hat { y } } + C _ { 1 } \big ) \big ( 2 \sigma _ { y \hat { y } } + C _ { 2 } \big ) } { \big ( \mu _ { y } ^ { 2 } + \mu _ { \hat { y } } ^ { 2 } + C _ { 1 } \big ) \big ( \sigma _ { y } ^ { 2 } + \sigma _ { \hat { y } } ^ { 2 } + C _ { 2 } \big ) }\tag{7}
$$

where $\mu _ { y }$ and $\mu _ { \hat { y } }$ are the local means of the ground-truth and pre dicted congestion maps, $\sigma _ { y } ^ { 2 }$ and $\sigma _ { \hat { y } } ^ { 2 }$ are the variances, $\sigma _ { y \hat { y } }$ is the covariance. SSIM evaluates the similarity between the two congestion maps in terms of local intensity, contrast, and spatial structure. A higher SSIM indicates greater structural similarity between the predicted and ground-truth congestion maps.

In addition, congestion labels exhibit a highly long-tailed distribution. Therefore, we set the classification threshold to the 90th percentile of the congestion distribution in the training set. Grid exceeding the threshold are labelled as congestion hotpots. Accu racy, F1-score, ROC-AUC, and PR-AUC are employed to evaluate the capability of the model to identify high-risk congested regions.

Definition 3 (Accuracy). Accuracy measures the proportion of correctly classified grids among all evaluated grids.

$$
\mathrm { A c c u r a c y } = { \frac { T P + T N } { T P + T N + F P + F N } }\tag{8}
$$

where � �, � �, ��, �� are the numbers of true-positive, truenegative, false-positive, and false-negative grids, respectively.

Definition 4 (F1-score). F1-score is the harmonic mean of Precision and Recall.

$$
\mathrm { P r e c i s i o n } = { \frac { T P } { T P + F P } }\tag{9}
$$

$$
\mathrm { R e c a l l } = { \frac { T P } { T P + F N } }\tag{10}
$$

$$
{ \mathrm { F 1 - s c o r e } } = 2 \times { \frac { { \mathrm { P r e c i s i o n } } \times { \mathrm { R e c a l l } } } { { \mathrm { P r e c i s i o n } } + { \mathrm { R e c a l l } } } }\tag{11}
$$

Precision measures the proportion of predicted hotspots that are truly congested, whereas Recall measures the proportion of actual congestion hotspots successfully identified by the model.

Definition 5 (ROC-AUC). The receiver operating characteristic curve plots the true-positive rate against the false-positive rate across diferent decision thresholds, and ROC-AUC is the area under this curve.

Definition 6 (PR-AUC). The precision–recall curve describes the trade-of between Precision and Recall across diferent decision thresholds, and PR-AUC is the area under this curve.

Accuracy measures overall classification correctness, whereas F1-score jointly considers the precision and recall of congestion hotpots detection. ROC-AUC evaluates discrimination between congested and non-congested region across decision thresholds. PR-AUC focuses on detection performance for the congestion class. Since congested regions are relatively rare, F1-score and PR-AUC are used as the primary metrics for evaluating hotspot detection performance.

For the wirelength prediction task, the model predicts the wirelength of individual nets rather than reconstructing a spatial congestion map. Therefore, we adopt MAE, RMSE, Pearson correlation, Spearman correlation, and Kendall correlation as the evaluation metrics.

## 3 Methodology

## 3.1 Graph Construction

To preserve semantic information from logical topology, physical layout and routing resource, we represent each circuit as heterogeneous graph, as shown in Figure 1(a). HGRP models the cells, nets,

![](images/2674442236ddeec5417e01d9a679cde3a717e827d081714550a249e0cf999705.jpg)  
Figure 1: The HGRP architecture.

and routing grids as distinct types of nodes, forming the node sets C, N and $^ { \mathcal { G } , }$ respectively.Their input feature matrices are

$$
\mathbf { X } _ { C } \in \mathbb { R } ^ { | C | \times d _ { C } } , \mathbf { X } _ { N } \in \mathbb { R } ^ { | N | \times d _ { N } } , \mathbf { X } _ { \mathcal { G } } \in \mathbb { R } ^ { | \mathcal { G } | \times d _ { \mathcal { G } } }\tag{12}
$$

where $d _ { C } , d _ { N }$ and $d _ { \mathcal { G } }$ are the corresponding feature dimensions. HGRP defines three types of relation to encode topological connectivity, spatial mapping and grid adjacency

$$
\mathcal { T } \subseteq C \times N , M \subseteq C \times \mathcal { G } , \mathcal { A } \subseteq \mathcal { G } \times \mathcal { G }\tag{13}
$$

Specifically, $\mathcal { T }$ connects cells to nets, M connects cells to grids, and A connects adjacent grids. To enable bidirectional information propagation across node types, HGRP introduces the reverse relations

$$
\mathcal { T } ^ { - 1 } = \left\{ ( n _ { j } , c _ { i } ) | ( c _ { i } , n _ { j } ) \in \mathcal { T } \right\} , \mathcal { M } ^ { - 1 } = \left\{ ( g _ { k } , c _ { i } ) | ( c _ { i } , g _ { k } ) \in \mathcal { M } \right\}\tag{14}
$$

In addition, edge feature matrices are constructed for the spatial mapping, reverse spatial mapping, and grid adjacency relations

$$
\begin{array} { r } { \mathbf { Z } _ { \mathcal { M } } \in \mathbb { R } ^ { | \mathcal { M } | \times d _ { \mathcal { M } } } , \mathbf { Z } _ { \mathcal { M } ^ { - 1 } } \in \mathbb { R } ^ { | \mathcal { M } ^ { - 1 } | \times d _ { \mathcal { M } ^ { - 1 } } } , \mathbf { Z } _ { \mathcal { R } } \in \mathbb { R } ^ { | \mathcal { R } | \times d _ { \mathcal { A } } } } \end{array}\tag{15}
$$

providing fine-grained relational information for the subsequent attention-based adaptive message aggregation. Based on the above definitions, the heterogeneous graph representation of a circuit is formulated as

$$
\mathcal { H } = ( \mathcal { V } , \mathcal { E } , \mathbf { X } , \mathbf { Z } )\tag{16}
$$

where $\mathcal { V } = C \cup N \cup \mathcal { G }$ and $\mathcal { E } = \mathcal { T } \cup \mathcal { T } ^ { - 1 } \cup \mathcal { M } \cup \mathcal { M } ^ { - 1 } \cup \mathcal { A }$ are the node and edge sets, X and Z are their feature matrices.

## 3.2 Model Architecture

The overall architecture of HGRP is illustrated in Figure 1(b). Given the heterogeneous graph representation $\mathcal { H } = ( \mathcal { V } , \mathcal { E } , \mathbf { X } , \mathbf { Z } )$ , the raw features of diferent node and edge types may difer in both dimensionality and physical semantics. Therefore, HGRP employs independent linear projection layers to map the input features of each type into a unified latent representation space

$$
\mathbf { H } _ { t } ^ { ( 0 ) } = \sigma ( \mathbf { X } _ { t } \mathbf { W } _ { t } + \mathbf { b } _ { t } ) , t \in \{ C , N , g \}\tag{17}
$$

$$
\mathbf { E } _ { r } = \sigma ( \mathbf { Z } _ { r } \mathbf { W } _ { r } ^ { e } + \mathbf { b } _ { r } ^ { e } ) , r \in \left\{ \mathcal { M } , \mathcal { M } ^ { - 1 } , \mathcal { A } \right\}\tag{18}
$$

where $\mathbf { H } _ { t } ^ { ( 0 ) }$ is the initial representation of nodes of type $t , \mathbf { E } _ { r }$ is the representation ofedges associated with relation $r , \sigma ( \cdot )$ is a nonlinear activation function, $\mathbf { W } _ { t }$ and $\mathbf { W } _ { r } ^ { e } ,$ together with their corresponding biases, are learnable parameters. After feature encoding, the node representations are sequentially processed by topology-aware, geometry-aware, and spatial-aware message passing modules

$$
( \mathbf { H } _ { C } ^ { ( T ) } , \mathbf { H } _ { N } ^ { ( T ) } ) = \Phi _ { T } ( \mathbf { H } _ { C } ^ { ( 0 ) } , \mathbf { H } _ { N } ^ { ( 0 ) } ; \mathcal { T } , \mathcal { T } ^ { - 1 } )\tag{19}
$$

$$
( \mathbf { H } _ { C } ^ { ( T + M ) } , \mathbf { H } _ { \mathcal { G } } ^ { ( M ) } ) = \Phi _ { M } ( \mathbf { H } _ { C } ^ { ( T ) } , \mathbf { H } _ { \mathcal { G } } ^ { ( 0 ) } ; \mathcal { M } , \mathcal { M } ^ { - 1 } ; \mathbf { E } _ { \mathcal { M } } , \mathbf { E } _ { \mathcal { M } ^ { - 1 } } )\tag{20}
$$

$$
{ \bf Q } _ { { \mathcal G } } ^ { ( A ) } = \Phi _ { A } ( { \bf Q } _ { { \mathcal G } } ^ { ( 0 ) } ; { \mathcal R } ; { \bf E } _ { { \mathcal R } } ) , { \bf Q } _ { { \mathcal G } } ^ { ( 0 ) } = \sigma ( [ { \bf X } _ { { \mathcal G } } | | { \bf H } _ { { \mathcal G } } ^ { ( M ) } ] { \bf W } _ { f } + { \bf b } _ { f } )\tag{21}
$$

where || denotes feature concatenation, �, �, � are the numbers of layers in the corresponding modules. The topology-aware module Φ<sub>�</sub> incorporates contextual information from multi-pin nets into the cell representations. Then the geometry-aware module $\Phi _ { M }$ transfers cell-level topology and placement information to the corresponding grid. Finally, the spatial-aware module $\Phi _ { A }$ aggregates placement and routing resource information from adjacent grids, enlarging the spatial receptive field and capturing routing resource competition and spatial correlations among congested regions.

The grid representations before and after spatial propagation are concatenated and passed to a multi-layer perceptron (MLP) to predict grid-level congestion

$$
\hat { \bf y } _ { c o n g } = \mathrm { M L P } [ { \bf Q } _ { \mathcal { G } } ^ { ( 0 ) } | | { \bf Q } _ { \mathcal { G } } ^ { ( A ) } ] \in \mathbb { R } ^ { | \mathcal { G } | }\tag{22}
$$

Then the predictions are rearranged according to the physical coordinates, forming the congestion map $\hat { \mathbf { Y } } _ { c o n g } \in \mathbb { R } ^ { w \times h }$

## 3.3 Heterogeneous Message Passing

HGRP employs a staged message passing mechanism according to the physical semantics of each edge relation as shown in Figure 1(b). Node representations propagate sequentially along the cell-net connection, cell-grid mapping and grid adjacency. For any relation �, a target node �<sub>�</sub> aggregates information from its relation-specific neighbourhood $N _ { r } ( v _ { i } )$

$$
\mathbf { m } _ { i , r } ^ { ( l ) } = \mathrm { A G G } _ { v _ { j } \in N _ { r } ( v _ { i } ) } \mathcal { F } _ { r } ^ { ( l ) } ( h _ { i } ^ { ( l ) } , h _ { j } ^ { ( l ) } , e _ { j i , r } )\tag{23}
$$

where $\mathcal { F } r ^ { ( l ) } ( \cdot )$ is the message generation function associated with relation �, AGG(·) is the neighbourhood aggregation operator, $e _ { j i , r }$ is the attribute of edge $( v _ { j } , v _ { i } )$ under relation $r .$ The edge attribute term is omitted when the corresponding relation does not contain explicit edge features. The node representation is updated through a residual connection [4] and layer normalization

$$
\mathbf { h } _ { i } ^ { ( l + 1 ) } = \mathrm { L N } ( \mathbf { h } _ { i } ^ { ( l ) } + \sigma ( \mathbf { W } _ { r } ^ { o , ( l ) } \mathbf { m } _ { i , r } ^ { ( l ) } + \mathbf { b } _ { r } ^ { o , ( l ) } ) )\tag{24}
$$

where LN(·) is layer normalization. The residual connection preserves the previous node representation and improves training stability across multiple message passing layers. All message passing stages follow the same node update formulation, while adopting relation-specific message generation and neighbourhood aggregation operations.

Topology-aware Message Passing: The topology-aware stage performs bidirectional message passing along $\mathcal { T }$ and $\mathcal { T } ^ { - 1 }$ . The neighbourhood messages for net node $n _ { j }$ and cell node $c _ { i }$ are formulated as

$$
\mathbf { m } _ { n _ { j } } ^ { ( l ) } = \mathbf { W } _ { \mathcal { T } , s } ^ { ( l ) } \mathbf { h } _ { n _ { j } } ^ { ( l ) } + \mathbf { W } _ { \mathcal { T } , n } ^ { ( l ) } \sum _ { c _ { i } \in N _ { \mathcal { T } } ( n _ { j } ) } \mathbf { h } _ { c _ { i } } ^ { ( l ) }\tag{25}
$$

$$
\mathbf { m } _ { c _ { i } } ^ { ( l ) } = \mathbf { W } _ { \mathcal { T } ^ { - 1 } , s } ^ { ( l ) } \mathbf { h } _ { c _ { i } } ^ { ( l ) } + \mathbf { W } _ { \mathcal { T } ^ { - 1 } , n } ^ { ( l ) } \sum _ { n _ { j } \in N _ { \mathcal { T } ^ { - 1 } } ( c _ { i } ) } \mathbf { h } _ { n _ { j } } ^ { ( l ) }\tag{26}
$$

In the cell-to-net direction, each net node aggregates its incident cell features to capture multi-pin net context. The aggregated net information is then propagated back to connected cells, enabling each cell to incorporate information from other cells connected through the same net. Through multiple bidirectional propagation layers, cell representations encode both intrinsic attributes and topological context.

Geometry-aware Message Passing: The updated cell representations $\mathbf { H } _ { C } ^ { ( T ) }$ and the initial grid representations $\mathbf { H } _ { \mathcal { G } } ^ { ( 0 ) }$ are passed to the geometry-aware stage, where bidirectional message passing is performed along M and $M ^ { - 1 }$ . An edge-aware multi-head attention mechanism [11] is introduced to dynamically adjust the contributions of neighboring nodes. For relation $r \in \mathcal { M } , \mathcal { M } ^ { - 1 }$ , the attention weight from source node $v _ { j }$ to target node $v _ { i }$ under head $\boldsymbol { p }$ is formulated as

$$
\alpha _ { j i , r } ^ { p } = \mathrm { s o f t m a x } _ { v _ { j } \in N _ { r } ( v _ { i } ) } ( \frac { ( \mathbf { W } _ { Q , r } ^ { p } \mathbf { h } _ { i } ) ^ { \mathrm { T } } ( \mathbf { W } _ { K , r } ^ { p } \mathbf { h } _ { j } + \mathbf { W } _ { E , r } ^ { p } \mathbf { e } _ { j i , r } ) } { \sqrt { d _ { \rho } } } )\tag{27}
$$

where $d _ { p }$ is the feature dimension of each attention head. The message received by $v _ { i }$ under attention head (p) is

$$
\mathbf { m } _ { i , r } ^ { p } = \sum _ { v _ { j } \in N _ { r } ( v _ { i } ) } \alpha _ { j i , r } ^ { p } ( \mathbf { W } _ { V , r } ^ { p } \mathbf { h } _ { j } + \mathbf { W } _ { E , r } ^ { p } \mathbf { e } _ { j i , r } )\tag{28}
$$

Outputs from the $P$ attention heads are concatenated to obtain the final aggregated message

$$
\mathbf { m } _ { i , r } = | | _ { p = 1 } ^ { P } \mathbf { m } _ { i , r } ^ { p }\tag{29}
$$

In the cell-to-grid direction, each grid node aggregates information from its spatially mapped cells, thereby transferring cell-level topology and placement features to the corresponding physical region. In the reverse direction, each cell aggregates information from its associated grid region, allowing its representation to capture local routing resource conditions.

Spatial-aware Message Passing: The grid representations obtained from the geometry-aware stage are concatenated with the original grid features, thereby preserving both routing resource information and higher-level semantics aggregated from the cell nodes. Then HGRP performs spatial message passing over the gridadjacency relation A to model routing resource competition and congestion propagation between neighbouring regions.

Let h<sub>�</sub> and ${ \bf h } _ { j }$ denote the fused representations of target grid $g _ { i }$ and adjacent grid $g _ { j } .$ , respectively. The spatial attention weight under attention head (p) is formulated as

$$
\alpha _ { j i } ^ { p } = \mathrm { s o f t m a x } _ { g _ { j } \in N _ { \mathcal { R } } ( g _ { i } ) } ( \frac { ( \mathbf { W } _ { Q } ^ { p } \mathbf { h } _ { i } ) ^ { \mathrm { T } } ( \mathbf { W } _ { K } ^ { p } \mathbf { h } _ { j } + \mathbf { W } _ { E } ^ { p } \mathbf { e } _ { j i , \mathcal { R } } ) ) } { \sqrt { d _ { \rho } } } )\tag{30}
$$

The messages from adjacent grids are aggregated across $P$ attention heads as

$$
\mathbf { m } _ { i } = | | _ { p = 1 } ^ { P } \sum _ { g _ { j } \in N _ { \mathcal { A } } ( g _ { i } ) } \alpha _ { j i } ^ { p } ( \mathbf { W } _ { V } ^ { p } \mathbf { h } _ { j } + \mathbf { W } _ { E } ^ { p } \mathbf { e } _ { j i , \mathcal { A } } )\tag{31}
$$

As spatial message-passing layers are stacked, each grid node progressively receives placement density and routing resource information from its multi-hop neighborhood. The resulting spatial context captures the local continuity of congested regions and improves the representation of routing resource competition and congestion propagation.

Table 1: Congestion Prediction Results
<table><tr><td>Model</td><td>NRMSE↓</td><td>SSIM↑</td><td>Accuracy↑</td><td>F1-score↑</td><td>ROC-AUC↑</td><td>PR-AUC↑</td><td>Param</td><td>Memory</td><td>Train</td><td>Inference</td></tr><tr><td>GPDL</td><td>0.0495</td><td>0.7789</td><td>0.8682</td><td>0.3558</td><td>0.7773</td><td>0.4055</td><td>0.119M</td><td>0.309G</td><td>8.100s</td><td>0.415s</td></tr><tr><td>NetlistGNN</td><td>0.0406</td><td>0.7511</td><td>0.8966</td><td>0.3992</td><td>0.8699</td><td>0.4118</td><td>0.271M</td><td>5.310G</td><td>33.678s</td><td>4.279s</td></tr><tr><td>LHNN</td><td>0.0350</td><td>0.7750</td><td>0.8830</td><td>0.4589</td><td>0.8658</td><td>0.4731</td><td>0.054M</td><td>3.486G</td><td>28.615s</td><td>3.354s</td></tr><tr><td>HGRP</td><td>0.0338</td><td>0.8090</td><td>0.8803</td><td>0.5447</td><td>0.8998</td><td>0.5428</td><td>0.200M</td><td>9.495G</td><td>60.356s</td><td>8.542s</td></tr></table>

Table 2: Multi-seed Experimental Results.
<table><tr><td>Random Seed</td><td>NRMSE↓</td><td>SSIM↑</td><td>Accuracy↑</td><td>F1-score↑</td><td>ROC-AUC↑</td><td>PR-AUC↑</td></tr><tr><td>1</td><td>0.0338</td><td>0.8083</td><td>0.8847</td><td>0.5360</td><td>0.8994</td><td>0.5084</td></tr><tr><td>2</td><td>0.0339</td><td>0.8061</td><td>0.8801</td><td>0.5470</td><td>0.8807</td><td>0.5313</td></tr><tr><td>3</td><td>0.0338</td><td>0.8088</td><td>0.8844</td><td>0.5313</td><td>0.9026</td><td>0.5095</td></tr><tr><td>4</td><td>0.0338</td><td>0.8092</td><td>0.8821</td><td>0.5325</td><td>0.9013</td><td>0.5174</td></tr><tr><td>Mean±Std</td><td>0.0338±0.0000</td><td>0.8081±0.0014</td><td>0.8828±0.0021</td><td>0.5367±0.0071</td><td>0.8960±0.0102</td><td>0.5167±0.0105</td></tr></table>

## 4 Experiments

## 4.1 Experimental Settings

We conduct experiments on the CircuitNet [2] dataset to evaluate the efectiveness of HGRP for routability prediction. Following the experimental settings of CircuitNet, RISCY-a, RISCY-b, RISCY-FPU-a, and RISCY-FPU-b are used for training, while zero-riscya and zero-riscy-b are used for validation and testing. This split is intended to demonstrate the generalization capability of the model outside the training dataset. All samples utilize the routability features and routing congestion labels provided by CircuitNet. The extracted routability features consist of topological features from circuit connectivity, geometric features from physical layout, and local statistical features at the grid level.

We implement HGRP using PyTorch Geometric with an NVIDIA A100 GPU. The hidden dimensions of cell, net, and grid node representations are uniformly set to 64, while the hidden dimension of edge features is set to 16. The topology-aware, geometry-aware, and spatial-aware stages each consist of two message passing layers. We used AdamW[9] optimizer with initial learning rate of $5 e ^ { - 4 }$ and MSE loss to train HGRP.

## 4.2 Experimental Results

We compared HGRP with GPDL [7], NetlistGNN [14] and LHNN [12] to evaluate its predictive performance. These methods adopt distinct circuit representations, enabling the efectiveness of the proposed heterogeneous circuit representation to be evaluated from multiple perspectives.

GPDL encodes physical layout information into multichannel feature maps and resizes feature maps and congestion maps of dif ferent dimensions to 256 × 256 through interpolation. NetlistGNN models cells and nets as heterogeneous nodes and introduces edges between spatially adjacent cells using a sliding window, thereby incorporating geometric information into the topology-based graph representation. The geometric window size is set to 2.25 × 2.25, matching the routing grid dimensions. LHNN represents each circuit as a lattice hypergraph, allowing congestion information to propagate through both geometric and topological relations. In our implementation, the network architectures and unspecified parameters followed the corresponding publications or their public implementations. All methods were retrained using above dataset partition and supervised with routing congestion labels from the same source. Predictions from each method were arranged into the corresponding grid-level congestion maps before evaluation.

Congestion Prediction: The experimental results for congestion prediction are reported in Table 1, HGRP achieves the best performance in terms of both NRMSE and SSIM, demonstrating that HGRP can reduce the overall prediction error of grid-level congestion while accurately reconstructing the spatial structure of congested regions.

The congestion labels in CircuitNet exhibit substantial class imbalance, with non-congested grids substantially outnumbering congested grids. Accuracy measures overall classification correctness but is easily dominated by the majority class, thus failing to adequately reflect the capability of model to identify minority congestion hotpots. In contrast, F1-score jointly measures the precision and recall of congestion hotpots, whereas PR-AUC evaluates performance across the precision-recall trade-of. These metrics therefore provided more informative assessments of minority hotspot detection. Compared with the best-performing baseline, HGRP improves F1-score and PR-AUC by 18.69% and 14.73%, respectively, demonstrating its efectiveness in identifying high-risk congested regions.

To evaluate the robustness of HGRP against random initialization, we conduct experiments with four diferent random seeds. As shown in Table 2, the regression metrics remain highly stable with negligible variations, while the hotspot prediction metrics exhibit slightly larger fluctuations due to their sensitivity to classification thresholds. Nevertheless, HGRP consistently maintains competitive performance across all runs, demonstrating its robustness.

Net Wirelength Prediction: We further evaluated the applicability of the proposed heterogeneous graph representation through net wirelength prediction. Unlike congestion prediction, wirelength is predicted from net node representations. Therefore, we place geometry-aware stage before topology-aware stage. Cell nodes first incorporate layout information from their corresponding physical regions. Then net nodes aggregated the combined topological and geometric representations of their incident cells for wirelength regression..

Table 3: Net Wirelength Prediction Results.
<table><tr><td>Model</td><td>MAE↓</td><td>RMSE↓</td><td>Pearson↑</td><td>Spearman↑</td><td>Kendall↑</td></tr><tr><td>NetlistGNN</td><td>0.2945</td><td>0.5652</td><td>0.6621</td><td>0.7109</td><td>0.5352</td></tr><tr><td>LHNN</td><td>0.3157</td><td>0.5587</td><td>0.6214</td><td>0.6966</td><td>0.5270</td></tr><tr><td>HGRP</td><td>0.2925</td><td>0.5138</td><td>0.6932</td><td>0.7354</td><td>0.5636</td></tr></table>

Table 4: Ablation Study Results.
<table><tr><td>Variant</td><td>NRMSE↓</td><td>SSIM↑</td><td>Accuracy↑</td><td>F1-score↑</td><td>ROC-AUC↑</td><td>PR-AUC↑</td></tr><tr><td>W/o grid</td><td>0.0392</td><td>0.7594</td><td>0.9117</td><td>0.3810</td><td>0.8011</td><td>0.3178</td></tr><tr><td>W/o attention</td><td>0.0340</td><td>0.8039</td><td>0.9172</td><td>0.4577</td><td>0.8698</td><td>0.4592</td></tr><tr><td>Mixed</td><td>0.0340</td><td>0.8076</td><td>0.9147</td><td>0.4533</td><td>0.8783</td><td>0.4800</td></tr><tr><td>HGRP</td><td>0.0338</td><td>0.8090</td><td>0.8803</td><td>0.5447</td><td>0.8998</td><td>0.5428</td></tr></table>

Table 5: Performance Comparison under Diferent Hyperparameter Settings.
<table><tr><td>Node Dimension</td><td>Edge Dimension</td><td>Layers</td><td>NRMSE↓</td><td>SSIM↑</td><td>Accuracy↑</td><td>F1-score↑</td><td>ROC-AUC↑</td><td>PR-AUC↑</td></tr><tr><td>32</td><td>16</td><td>2</td><td>0.0337</td><td>0.8082</td><td>0.9055</td><td>0.4846</td><td>0.8569</td><td>0.4825</td></tr><tr><td>64</td><td>16</td><td>2</td><td>0.0338</td><td>0.8090</td><td>0.8803</td><td>0.5447</td><td>0.8998</td><td>0.5428</td></tr><tr><td>128</td><td>16</td><td>2</td><td>0.0336</td><td>0.8059</td><td>0.9211</td><td>0.4688</td><td>0.8789</td><td>0.4562</td></tr><tr><td>64</td><td>32</td><td>2</td><td>0.0337</td><td>0.8083</td><td>0.9194</td><td>0.4388</td><td>0.8743</td><td>0.4460</td></tr><tr><td>64</td><td>16</td><td>3</td><td>0.0336</td><td>0.8104</td><td>0.9216</td><td>0.4522</td><td>0.9022</td><td>0.4565</td></tr></table>

The experimental results for net wirelength prediction are reported in Table 3, HGRP achieves the best performance across all evaluation metrics, demonstrating that HGRP can estimate net wire length more accurately while maintaining higher consistency with the ground-truth wirelengths in terms of both numerical variation and net ranking.

These results demonstrate that the proposed heterogeneous graph representation efectively preserves both circuit netlist and physical layout information. Moreover, the propagation order of diferent relations can be dynamically adjusted according to the prediction target, enabling HGRP to accommodate physical design tasks at diferent abstraction levels.

## 4.3 Ablation Experiment

To evaluate the contributions of the heterogeneous graph representation and heterogeneous message-passing mechanism, we constructed three ablation variants of the complete HGRP model. W/o grid maps grid features to cell nodes before removing grid nodes and their associated relations. Cell representations are learned solely through the topological relations, after which the cell-level predictions are projected onto the grid graph. W/o attention replaces the attention-based adaptive aggregation in the geometryaware and spatial-aware stages with equal-weight neighborhood aggregation. Mixed jointly aggregates information from diferent relation-specific neighborhoods rather than propagating sequentially through the topology-aware, geometry-aware, and spatialaware stages.

The ablation study results are reported in Table 4. Removing grid nodes produces substantial decreases across all key metrics. Cell-level representations alone are therefore insuficient to capture the spatial distribution of routing demand. In contrast, explicitly introducing grid nodes enables the model to integrate logical topology, physical layout, and routing resource information at the spatial resolution consistent with the prediction target. Removing the attention mechanism reduced both F1-score and PR-AUC, demonstrating that adaptively adjusting neighbor contributions based on node representations and edge attributes improves the model’s capability to identify congestion hotspots. Mixed message passing also reduced predictive performance. Jointly aggregating diferent relation types may blur their distinct semantics, whereas staged propagation preserves relation-specific information and captures dependencies across circuit design hierarchies.

## 4.4 Hyperparameter Analysis

To analyze the impact of architectural hyperparameters on prediction performance, we compared HGRP under diferent numbers of node hidden dimensions, edge hidden dimensions, and messagepassing layers, as reported in Table 5. Regression metrics remained relatively stable across these settings, whereas congestion hotspot detection metrics varied more substantially. This observation suggests that model capacity primarily afects the discrimination of the minority congested samples, with relatively limited influence on fitting the overall congestion distribution.

As the node hidden dimension increases from 32 to 64, the additional capacity may facilitate representation of interactions among logical topology, physical layout and routing resource, which contributes to the overall improvement in classification metrics. However, further increasing the dimension to 128 markedly reduces F1-score and PR-AUC. A possible explanation is that the increased number of model parameters produced more conservative predictions, favouring the majority low-congestion class while reducing recall for minority congestion hotspots.

Similarly, increasing the edge hidden dimension does not produce consistent performance improvements. Since spatial-mapping and grid-adjacency relations have relatively explicit physical seman tics, low-dimensional edge representations may already capture the principal spatial information. Further increasing the dimension may introduce redundant features that interfere with attention learning, making neighbour importance more dificult to distinguish.

Increasing the message-passing depth expands the receptive field, allowing grid nodes to aggregate topological and spatial information from larger neighbourhoods. Nevertheless, excessive propagation depth may cause neighbouring node representations increasingly similar, weakening the feature distinction between high-congestion grids and their surroundings, thereby reducing hotspot-classification performance.

In summary, moderate node and edge representation dimensions, together with appropriate message-passing depth, provide a better balance among representational capacity, training stability and generalization performance.

## 5 Conclusion

In this work, we propose HGRP, which models circuits as heterogeneous graphs to provide a more complete circuit representation for routability prediction. Based on the heterogeneous graph, we introduces a heterogeneous message passing mechanism coupled with an adaptive feature aggregation strategy to update node representations, thereby capturing the complex coupling among circuit netlist, physical layout and routing resource. Experimental results on the CircuitNet dataset evaluate the efectiveness of the the proposed method. In future work, we will further explore its application to routability-driven placement optimization.

## References

[1] First Author and Second Author. 2026. R2G: A Multi-View Circuit Graph Bench mark Suite from RTL to GDSII. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[2] Zhuomin Chai, Yuxiang Zhao, Wei Liu, Yibo Lin, Runsheng Wang, and Ru Huang. 2023. CircuitNet: An Open-Source Dataset for Machine Learning in VLSI CAD Applications With Improved Domain-Specific Evaluation Metric and Learning Strategies. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 42, 12 (2023), 5034–5047. doi:10.1109/TCAD.2023.3287970

[3] Jingsong Chen, Jian Kuang, Guowei Zhao, Dennis J.-H. Huang, and Evangeline F.Y. Young. 2020. PROS: A Plug-in for Routability Optimization applied in the State-of the-art commercial EDA tool using deep learning. In 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD). 1–8.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 770–778. doi:10.1109/CVPR.2016.90

[5] Robert Kirby, Saad Godil, Rajarshi Roy, and Bryan Catanzaro. 2019. Congestion-Net: Routing Congestion Prediction Using Deep Graph Neural Networks. In 2019 IFIP/IEEE 27th International Conference on Very Large Scale Integration (VLSI-SoC). 217–222. doi:10.1109/VLSI-SoC.2019.8920342

[6] Rongjian Liang, Hua Xiang, Diwesh Pandey, Lakshmi Reddy, Shyam Ramji, Gi-Joon Nam, and Jiang Hu. 2020. DRC Hotspot Prediction at Sub-10nm Process Nodes Using Customized Convolutional Network. In Proceedings of the 2020 International Symposium on Physical Design (Taipei, Taiwan) (ISPD ’20). Association for Computing Machinery, New York, NY, USA, 135–142. doi:10.1145/3372780.3375560

[7] Siting Liu, Qi Sun, Peiyu Liao, Yibo Lin, and Bei Yu. 2021. Global Placement with Deep Learning-Enabled Explicit Routability Optimization. In 2021 Design, Automation & Test in Europe Conference & Exhibition (DATE). 1821–1824. doi:10. 23919/DATE51398.2021.9473959

[8] Wen-Hao Liu, Wei-Chun Kao, Yih-Lang Li, and Kai-Yuan Chao. 2013. NCTU-GR 2.0: Multithreaded Collision-Aware Global Routing With Bounded-Length Maze Routing. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems 32, 5 (2013), 709–722. doi:10.1109/TCAD.2012.2235124

[9] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations. https://mlanthology.org/ iclr/2019/loshchilov2019iclr-decoupled/

[10] Peter Spindler and Frank M. Johannes. 2007. Fast and Accurate Routing Demand Estimation for Eficient Routability-driven Placement. In 2007Design, Automation & Test in Europe Conference & Exhibition. 1–6. doi:10.1109/DATE.2007.364463

[11] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention Is All You Need. arXiv:1706.03762 [cs.CL] https://arxiv.org/abs/1706.03762

[12] Bowen Wang, Guibao Shen, Dong Li, Jianye Hao, Wulong Liu, Yu Huang, Hongzhong Wu, Yibo Lin, Guangyong Chen, and Pheng Ann Heng. 2022. LHNN: lattice hypergraph neural network for VLSI congestion prediction. In Proceedings ofthe 59th ACM/IEEE Design Automation Conference (San Francisco, California) (DAC ’22). Association for Computing Machinery, New York, NY, USA, 1297–1302. doi:10.1145/3489517.3530675

[13] Zhiyao Xie, Yu-Hung Huang, Guan-Qi Fang, Haoxing Ren, Shao-Yun Fang, Yiran Chen, and Jiang Hu. 2018. RouteNet: Routability prediction for Mixed-Size Designs Using Convolutional Neural Network. In 2018 IEEE/ACM International Conference on Computer-Aided Design (ICCAD). 1–8. doi:10.1145/3240765.3240843

[14] Shuwen Yang, Zhihao Yang, Dong Li, Yingxue Zhang, Zhanguang Zhang, Guojie Song, and Jianye Hao. 2022. Versatile multi-stage graph neural network for circuit representation. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 1477, 12 pages.

[15] Yuxiang Zhao, Zhuomin Chai, Yibo Lin, Runsheng Wang, and Ru Huang. 2023. HybridNet: Dual-Branch Fusion of Geometrical and Topological Views for VLSI Congestion Prediction. CoRR abs/2305.05374 (2023). arXiv:2305.05374 doi:10. 48550/ARXIV.2305.05374