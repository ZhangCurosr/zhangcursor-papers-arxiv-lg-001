# GSLAD: Prototype-Regularized Graph Structure Learning for Multivariate Time Series Anomaly Detection

Zepeng Zhang, Fuad Khuri, Keivan Faghih Niresi, Olga Fink<sup>∗</sup>

Intelligent Maintenance and Operations Systems (IMOS) Lab École Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland {zepeng.zhang, fuad.khuri, keivan.faghihniresi, olga.fink}@epfl.ch

## Abstract

Unsupervised multivariate time series anomaly detection methods typically identify anomalies through forecasting, reconstruction, or representation discrepancies. However, industrial faults may first alter inter-variable structural patterns while individual trajectories remain close to normal, resulting in weak anomaly signals. In this paper, we propose GSLAD, a prototype-regularized graph structure learning framework that uses structural deviations for anomaly scoring. GSLAD adopts a two-phase training strategy. First, a condition-aware graph learner and a graph-based forecaster are optimized with predictive supervision. The inferred normal graphs are then clustered into multiple structural prototypes representing diferent normal operating regimes, with edge-wise variability characterizing structural uncertainty. Deviations from these prototypes regularize the graph learner in the second phase, encouraging stable and regime-specific structural patterns. During inference, uncertainty-normalized structural deviation is combined with predictive deviation for anomaly scoring. Experiments on four industrial benchmarks demonstrate strong overall performance of GSLAD and confirm the efectiveness of structural deviation for anomaly detection and diagnosis.

## Introduction

Many industrial systems are monitored with sensor networks that generate vast amounts of multivariate time series (MTS) data. Detecting anomalies in MTS data is essential for preventing failures, reducing downtime, and maintaining the reliable operation of complex industrial systems (Wu, Dai, and Tang 2021; Deng and Hooi 2021; Fink et al. 2026). In practice, however, fault labels are often scarce, incomplete, and expensive to obtain, motivating increasing interest in unsupervised MTS anomaly detection (Zhang et al. 2019; Audibert et al. 2020; Belay et al. 2023).

Unsupervised MTS anomaly detection models are typically trained to forecast (Deng and Hooi 2021) or reconstruct (Zhao and Fink 2024) normal observations. Then, during inference, anomalies are identified by measuring discrepancies between observed signals and model outputs (Jin et al. 2024; Chen and Eldardiry 2024; Ho, Karami, and Armanfard 2025). Despite their architectural diferences, most existing methods rely on a similar assumption: anomalous samples are expected to produce larger predictive, reconstructive, or representation discrepancies than normal samples (Zhao et al. 2020; Cho et al. 2025). This principle is efective when faults cause pronounced deviations in individual sensor trajectories. However, many industrial faults do not immediately induce large marginal deviations. Instead, they may primarily alter the structural patterns while individual sensor trajectories remain close to their normal ranges. As a result, a model may continue to forecast or reconstruct individual trajectories accurately, producing weak anomaly signal.

To capture structural patterns of the system, several MTS anomaly detection methods propose to incorporate graph structure learning into graph neural network (GNN)-based forecasting or reconstruction models (Deng and Hooi 2021; Zheng et al. 2023; Zhao and Fink 2024). In these approaches, however, the learned graph primarily serves as an intermediate computational structure for more efective message passing, while anomaly scores are still derived solely from prediction or reconstruction discrepancies. More recently, several methods have explicitly exploited structural information for anomaly detection, for example by constructing relational graphs from model gradients (Liu, Gao, and Jiao 2025) or intermediate representations (Cho et al. 2025). Nonetheless, the graph structures in these works used for anomaly scoring are derived post hoc from trained predictors or latent representations, rather than being jointly inferred as explicit model outputs.

Turning structural deviation into a reliable anomaly signal is nontrivial. Structural patterns may vary across diferent operating regimes, and complex systems often exhibit multiple distinct normal operating regimes. Moreover, normal structural variability is heterogeneous across edges. Deviations on stable edges should provide stronger anomaly evidence than deviations on edges that fluctuate naturally during normal operation. To address these challenges, we introduce GSLAD, a prototype-regularized graph structure learning framework for unsupervised MTS anomaly detection. GSLAD jointly infers condition-dependent graph structures and forecasts future observations with a GNN. Its twophase training procedure first learns predictive normal graphs and then summarizes them into multiple uncertainty-aware structural prototypes. These prototypes are expected to capture distinct normal operating regimes, while their edge-wise variability quantifies the reliability of individual structural relations. Deviation from these prototypes subsequently regularizes the graph learner, encouraging normal graphs to remain compact around regime-specific prototypes. During inference, uncertainty-normalized structural deviation from the nearest normal prototype is combined with predictive deviation for anomaly scoring.

The main contributions are summarized as follows:

• We propose GSLAD and develop a two-phase training strategy. The first phase learns condition-aware graphs under predictive supervision, while the second phase regularizes graph learning using structural prototypes, thereby capturing stable and regime-specific structural patterns.

• Beyond conventional residual-based anomaly scoring, we perform anomaly detection in a jointly learned structural space, where uncertainty-aware structural deviation is combined with predictive deviation for anomaly scoring.

• Experiments on four industrial benchmarks demonstrate the strong overall anomaly detection performance of GSLAD. The structural analysis and fault characterization demonstrate the potential for further fault diagnosis.

## Related Work

## Multivariate Time-Series Anomaly Detection

Unsupervised MTS anomaly detection methods typically learn normal system behavior and identify anomalies according to deviations from the learned normality. Most existing methods derive anomaly scores from prediction or reconstruction residuals. For example, MTAD-GAT (Zhao et al. 2020) jointly optimizes forecasting and reconstruction objectives on normal data and computes anomaly scores from the discrepancies between observations and model outputs. Graph-based methods further model intervariable dependencies to improve forecasting or reconstruction. GDN (Deng and Hooi 2021) learns sensor relationships for forecasting-based anomaly detection, whereas DyEdge-GAT (Zhao and Fink 2024) infers input-dependent graph structures for reconstruction-based anomaly detection. Beyond value-space residuals, other methods detect anomalies through association discrepancies (Xu et al. 2022), representation discrepancies (Yang et al. 2023), or frequency-domain deviations (Wu et al. 2025). Despite their architectural diferences, these methods primarily quantify abnormality through deviations in observed values or latent representations. They may therefore provide weak anomaly evidence when faults alter structural patterns without immediately inducing pronounced deviations in individual trajectories.

More closely related to GSLAD, several recent methods exploit structural changes for anomaly detection. Specifically, GRELEN (Zhang, Zhang, and Tsung 2022) characterizes anomalies through changes in the in-degree and outdegree distributions of learned relational graphs, GCAD (Liu, Gao, and Jiao 2025) detects anomalies from changes in relational graphs derived from the gradients of a trained predictor, while OracleAD (Cho et al. 2025) constructs relational graphs from intermediate representations and measures their structural deviations for anomaly scoring. Unlike methods that use graphs primarily as message-passing topologies or derive structural evidence from a trained model, GSLAD explicitly measures structural deviations in a jointly learned structural space.

## Graph Structure Learning

Graph structure learning aims to infer or refine graph topologies directly from data rather than relying exclusively on predefined structures (Zhu et al. 2021; Li et al. 2023). Existing graph structure learning methods have been developed to address problems where observed graphs are noisy, incomplete, or unavailable (Jin et al. 2020; Liu et al. 2022; Zhang et al. 2024). They typically optimize the learned graph structure together with node representations for downstream tasks such as classification, prediction, or representation learning. Accordingly, the graph mainly functions as an intermediate computational structure that determines how information is propagated by a GNN. GSLAD difers from conventional graph structure learning methods in that it not only uses the learned graph structure for more efective message passing but also explicitly perform anomaly detection in the learned structural space.

## Prototype-Regularized Graph Structure Learning

In this section, we first introduce the GSLAD architecture, which consists of a temporal encoder, a condition-aware graph learner, and a GNN forecaster. Then we introduce the two-phase training strategy and the anomaly scoring policy. A schematic overview of the GSLAD framework is provided in Figure 1.

## Predictive Dynamic Graph Learning

We denote by $\mathbf { X } _ { t }$ a window containing past W observations from N variables:

$$
{ \bf X } _ { t } = [ { \bf x } _ { : , t - W + 1 } , \ldots , { \bf x } _ { : , t } ] \in \mathbb { R } ^ { N \times W } ,\tag{1}
$$

and by $\mathbf { Y } _ { t }$ the subsequent H-step forecasting target:

$$
\mathbf { Y } _ { t } = [ \mathbf { x } _ { : , t + 1 } , \dots , \mathbf { x } _ { : , t + H } ] \in \mathbb { R } ^ { N \times H } .\tag{2}
$$

Given $\mathbf { X } _ { t } , \mathbf { G S L A D }$ predicts $\hat { \mathbf Y } _ { t }$ and simultaneously infers a dense directed graph $\mathbf { S } _ { t } \in [ 0 , 1 ] ^ { N \times N }$ . The model consists of a temporal encoder, a dynamic graph learner, and a GNN forecaster. The details of these three components are introduced in the following.

Temporal Encoding The historical window of each variable is encoded independently using a shared temporal encoder. Specifically, for input $\dot { \mathbf X } _ { t } = [ \mathbf { \check { x } } _ { 1 , : } , \dots , \mathbf { x } _ { N , : } ] ^ { \top }$ , we use an one-dimensional convolutional neural network (Kiranyaz et al. 2021) to encode it into latent space as follows:

$$
\mathbf { h } _ { i , : } = \mathrm { C o n v 1 D } ( \mathbf { x } _ { i , : } ) \in \mathbb { R } ^ { d } ,\tag{3}
$$

yielding node representations

$$
\mathbf { H } _ { t } = [ \mathbf { h } _ { 1 , : } , \ldots , \mathbf { h } _ { N , : } ] ^ { \top } \in \mathbb { R } ^ { N \times d } ,\tag{4}
$$

where d is the representation dimension. Sharing the encoder across nodes provides a consistent representation space while preserving node-specific temporal patterns. These node representations will then be used as inputs for both the following dynamic graph learner and the GNN forecaster.

![](images/beede96cbfdbdb86625aee917ec806c3cc4a911e4b3d06bf28468974f2c0acb4.jpg)  
Figure 1: Overview of GSLAD. In Phase I, the temporal encoder, condition-aware graph learner, and GNN forecaster are optimized using predictive supervision. Dense normal graphs are clustered into multiple uncertainty-aware prototypes, while sparsified graphs are used for eficient GNN processing. In Phase II, deviation from the prototype bank regularizes graph learning. During inference, uncertainty-aware structural deviation and predictive deviation are combined for anomaly scoring.

Condition-Aware Graph Inference Given the node representations $\mathbf { H } _ { t } .$ the graph learner constructs a dynamic graph through an attention-based module. Considering that the normal operation state of a system normally contains multiple operating conditions, we propose a condition-aware attention mechanism for graph learning. Specifically, we modulate node-wise query and key projections using a global state representation computed by mean-pooling all node representations as follows:

$$
\mathbf { c } _ { t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { h } _ { t , i } .\tag{5}
$$

This mechanism allows the same variable pair to receive diferent afinities under diferent operating conditions. The global state vector $\mathbf { c } _ { t }$ is used to modulate the query projection $\mathbf { q } _ { t , i }$ and key projection $\mathbf { k } _ { t , i }$ of each node:

$$
\begin{array} { r } { \mathbf { q } _ { t , i } = \left( \mathbf { W } _ { q } \mathbf { h } _ { t , i } \right) \odot \sigma \left( \mathbf { W } _ { q } ^ { c } \mathbf { c } _ { t } \right) , } \\ { \mathbf { k } _ { t , i } = \left( \mathbf { W } _ { k } \mathbf { h } _ { t , i } \right) \odot \sigma \left( \mathbf { W } _ { k } ^ { c } \mathbf { c } _ { t } \right) , } \end{array}\tag{6}
$$

where $\odot$ denotes element-wise multiplication, $\mathbf { W } _ { q } , \ \mathbf { W } _ { q } ^ { c } ,$ $\mathbf { W } _ { k }$ , and $\mathbf { W } _ { k } ^ { c } \in \mathbb { R } ^ { d \times d }$ are weight matrices shared across nodes. The edge weight for each node pair $( i , j )$ is computed as follows:

$$
\begin{array} { l } { { \displaystyle e _ { t , i j } = \frac { \mathbf { q } _ { t , i } ^ { \top } \mathbf { k } _ { t , j } } { \sqrt { d } } } , } \\ { { \displaystyle s _ { t , i j } = \sigma ( e _ { t , i j } ) \in ( 0 , 1 ) , } } \end{array}\tag{7}
$$

where σ represents the sigmoid function to map the pairwise score to (0, 1). The resulting dense graph $\mathbf { S } _ { t }$ is retained for the following prototype construction, which will be used to regularize graph learning during training and structural anomaly scoring during testing.

Graph-Based Forecasting With the learned dynamic graph, we use a GNN model to perform MTS forecasting. To facilitate eficient computation, we first convert the dense adjacency matrix into a sparse weighted graph $\mathbf { A } _ { t } .$ Specifically, we retain the top-k outgoing neighbors of each node and apply a masked softmax over the retained logits, which is computed by

$$
a _ { t , i j } = \left\{ \begin{array} { l l } { \frac { \exp \left( e _ { t , i j } \right) } { \sum _ { m \in \mathrm { T o p K } ( e _ { t , i : } ) } \exp \left( e _ { t , i m } \right) } , } & { j \in \mathrm { T o p K } ( e _ { t , i : } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

We distinguish the sparse graph $\mathbf { A } _ { t } ,$ , used only for message passing, from the dense graph $\mathbf { S } _ { t }$ , used for prototype learning and anomaly scoring. The sparse graph ${ \bf A } _ { t }$ is used by the GNN forecaster to eficiently propagate information across variables. With ${ \bf Z } _ { t } ^ { ( 0 ) } = { \bf H } _ { t } .$ , each layer updates node representations using

$$
\begin{array} { r } { \mathbf { Z } _ { t } ^ { ( \ell + 1 ) } = \operatorname { G N N } ^ { ( \ell ) } \left( \mathbf { Z } _ { t } ^ { ( \ell ) } , \mathbf { A } _ { t } \right) , \quad \ell = 1 , \ldots , L - 1 , } \end{array}\tag{9}
$$

where $\mathrm { G N N } ^ { ( \ell ) } ( \cdot )$ denotes one GNN layer, including a message passing step followed by nonlinearity. In our implementation, we use a GAT-v2 layer (Brody, Alon, and Yahav 2022), while the framework also supports other GNN layer options. After L layers of message passing, a node-wise MLP decodes the final representation to the forecasting horizon as follows:

$$
\begin{array} { r } { \hat { { \bf y } } _ { t , i } = \mathrm { M L P } \left( { \bf z } _ { t , i } ^ { ( L ) } \right) \in \mathbb { R } ^ { H } . } \end{array}\tag{10}
$$

## Prototype-Regularized Two-Phase Training

GSLAD follows a two-phase training procedure. Phase I learns predictive dependency structures using forecasting supervision. The dense graphs inferred from normal training windows are then summarized into a prototype bank, which serves as the structural reference for Phase II. Phase II training further refines the graph learner using both forecasting supervision and prototype-based regularization.

Phase I: Predictive Graph Learning Following the standard setting in graph structure learning literature (Zhu et al. 2021; Li et al. 2023), we alternate between updating the graph learner while freezing the forecaster and updating the forecaster while freezing the graph learner. Both steps are optimized with a unified predictive objective:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { N H } \left. \hat { \mathbf { Y } } _ { t } - \mathbf { Y } _ { t } \right. _ { F } ^ { 2 } .\tag{11}
$$

This phase allows the temporal encoder, graph learner, and GNN forecaster to learn predictive structural patterns. Note that only the parameters in the graph learner and GNN forecaster are optimized alternately, while the encoder is optimized in both alternating stages.

Normal Graph Prototype Construction After Phase I, we collect the learned dense graphs $\{ \mathbf { S } _ { t } \} _ { t \in \mathcal { T } }$ , where $\tau$ denotes the training set windows. A straightforward strategy is to use the average graph as the prototype to regularize the graph learning and to compute the structural deviation. However, interpolating between distinct operating regimes may produce a prototype graph that corresponds to no actual operating regime, and comparing every sample with such an averaged graph may incorrectly penalize normal structural changes. Therefore, we assume that normal graph structures concentrate around a small number of operatingregime-dependent prototypes. Specifically, we cluster the dense graphs inferred from normal training windows into K prototypes using K-means. For k-th prototype graph, we compute the edge-wise mean and empirical standard deviation within the corresponding cluster:

$$
\begin{array} { l } { \displaystyle \mu _ { k , i j } = \frac { 1 } { \vert { \mathcal T } _ { k } \vert } \sum _ { t \in { \mathcal T } _ { k } } s _ { t , i j } , } \\ { \displaystyle \sigma _ { k , i j } = \sqrt { \frac { 1 } { \vert { \mathcal T } _ { k } \vert } \sum _ { t \in { \mathcal T } _ { k } } \left( s _ { t , i j } - \mu _ { k , i j } \right) ^ { 2 } } . } \end{array}\tag{12}
$$

where $\mathcal { T } _ { k }$ is the set of windows assigned to the k-th prototype. The mean graph $\pmb { \mu } _ { k }$ represents the structural pattern of regime k, while $\sigma _ { k }$ captures edge-wise variability within that regime. The resulting set $\{ ( \mu _ { k } , \pmb { \sigma } _ { k } ) \} _ { k = 1 } ^ { K }$ forms the normal graph prototype bank used in prototype-regularized refinement and structural deviation computation. These standard deviations $\sigma _ { k , i j }$ will then be used to weight the deviation of edge (i, j). Specifically, a small $\sigma _ { k , i j }$ indicates a stable and clear structural pattern and therefore should receive larger weights. For edges with large $\sigma _ { k , i j } .$ , we assign lower weights as they fluctuate naturally even during normal operation conditions.

Phase II: Prototype-Regularized Refinement With the constructed normal graph prototype bank, we perform graph refinement by augmenting the loss in (11) with an additional prototype-regularized refinement term, while the GNN forecaster is trained with the same predictive objective in (11). Specifically, for a learned graph $\mathbf { S } _ { t } .$ , we measure its uncertainty-normalized graph deviation for each prototype graph as follows:

$$
d _ { k } ( \mathbf { S } _ { t } ) = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \frac { ( s _ { t , i j } - \mu _ { k , i j } ) ^ { 2 } } { \sigma _ { k , i j } ^ { 2 } + \sigma _ { 0 } ^ { 2 } } ,\tag{13}
$$

where $\sigma _ { 0 }$ is a small global stabilization term preventing near-zero variances from dominating the objective. This uncertainty-aware graph deviation is implemented as a diagonal Mahalanobis-style distance that assigns larger penalties to deviations on stable edges and smaller penalties to deviations on edges with naturally large variations. During training, we use a soft nearest prototype graph assignment as follows:

$$
w _ { t , k } = \frac { \exp ( - d _ { k } ( \mathbf { S } _ { t } ) / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( - d _ { j } ( \mathbf { S } _ { t } ) / \tau ) } ,\tag{14}
$$

where $\tau$ is a temperature parameter. The prototype graph regularization is computed as

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \sum _ { k = 1 } ^ { K } w _ { t , k } d _ { k } ( { \mathbf { S } } _ { t } ) .\tag{15}
$$

During training, the gradients only back-propagate through the graph distance terms but are stopped through the weights $w _ { t , k } ,$ , so the soft weights act only as prototype-selection coeficients rather than being optimized to trivially reduce the regularization objective. In summary, the training loss during Phase II is defined by

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda \mathcal { L } _ { \mathrm { g r a p h } } . } \end{array}\tag{16}
$$

Note that the prototype bank obtained in Phase I remains fixed throughout Phase II, preventing the reference distribution from drifting together with the graph learner. After Phase II training, we recompute the prototype graph bank and the corresponding statistics based on the newly obtained inferred graphs, forming the refined prototype graph bank, which is then used for structural deviation computing during inference.

## Structural and Predictive Anomaly Scoring

Since the GSLAD model performs graph learning and GNN forecasting simultaneously, it produces two complementary anomaly scores, namely, a predictive deviation and a structural deviation. The predictive deviation measures the distance between the forecasted time series and the real time series:

$$
r _ { t } = \left\| \hat { \mathbf { Y } } _ { t } - \mathbf { Y } _ { t } \right\| _ { F } ^ { 2 } .\tag{17}
$$

The uncertainty-aware structural deviation measures the distance between the learned graph and the closest prototype graph:

$$
d _ { t } = d ( \mathbf { S } _ { t } ) = \operatorname* { m i n } _ { k \in \{ 1 , \dots , K \} } \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \frac { ( \mathbf { s } _ { t , i j } - \pmb { \mu } _ { k , i j } ) ^ { 2 } } { \pmb { \sigma } _ { k , i j } ^ { 2 } + \pmb { \sigma } _ { 0 } ^ { 2 } } .\tag{18}
$$

<table><tr><td></td><td colspan="2">SWAT</td><td colspan="2">WADI</td><td colspan="2">PSM</td><td colspan="2">TEP</td></tr><tr><td></td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td></tr><tr><td>MTAD-GAT</td><td>0.7619</td><td>0.2280</td><td>0.6105</td><td>0.2284</td><td>0.7885</td><td>0.5464</td><td>0.9021</td><td>0.9136</td></tr><tr><td>GDN</td><td>0.7360</td><td>0.1975</td><td>0.6059</td><td>0.2276</td><td>0.7568</td><td>0.5812</td><td>0.8990</td><td>0.9130</td></tr><tr><td>GRELEN</td><td>0.7708</td><td>0.2320</td><td>0.5848</td><td>0.2282</td><td>0.6189</td><td>0.4003</td><td>0.8190</td><td>0.8355</td></tr><tr><td>Anomaly Transformer</td><td>0.7605</td><td>0.2367</td><td>0.4940</td><td>0.1590</td><td>0.7202</td><td>0.5391</td><td>0.9326</td><td>0.9494</td></tr><tr><td>NLinear</td><td>0.8187</td><td>0.7214</td><td>0.7086</td><td>0.4469</td><td>0.6843</td><td>0.5482</td><td>0.7382</td><td>0.7874</td></tr><tr><td>DLinear</td><td>0.5931</td><td>0.1263</td><td>0.6993</td><td>0.4462</td><td>0.6661</td><td>0.5237</td><td>0.8006</td><td>0.8201</td></tr><tr><td>DCDetector</td><td>0.7359</td><td>0.6264</td><td>0.6929</td><td>0.4442</td><td>0.6513</td><td>0.4896</td><td>0.8862</td><td>0.9131</td></tr><tr><td>PatchTST</td><td>0.8149</td><td>0.7198</td><td>0.7011</td><td>0.4458</td><td>0.6722</td><td>0.5071</td><td>0.8248</td><td>0.8667</td></tr><tr><td>DyEdgeGAT</td><td>0.7394</td><td>0.2406</td><td>0.6517</td><td>0.4426</td><td>0.6976</td><td>0.5407</td><td>0.9325</td><td>0.9510</td></tr><tr><td>iTransformer</td><td>0.8213</td><td>0.7231</td><td>0.7129</td><td>0.4482</td><td>0.6576</td><td>0.5075</td><td>0.7346</td><td>0.7436</td></tr><tr><td>ModernTCN</td><td>0.7555</td><td>0.6559</td><td>0.7064</td><td>0.4512</td><td>0.7074</td><td>0.5453</td><td>0.8903</td><td>0.9166</td></tr><tr><td>CATCH</td><td>0.8110</td><td>0.7178</td><td>0.7112</td><td>0.4467</td><td>0.6710</td><td>0.5056</td><td>0.8575</td><td>0.8868</td></tr><tr><td>GCAD</td><td>0.5597</td><td>0.1196</td><td>0.7197</td><td>0.4490</td><td>0.6611</td><td>0.4899</td><td>0.8500</td><td>0.8820</td></tr><tr><td>GSLAD</td><td>0.8607</td><td>0.7612</td><td>0.7018</td><td>0.5157</td><td>0.8193</td><td>0.7038</td><td>0.9563</td><td>0.9687</td></tr></table>

Table 1: Multivariate time series anomaly detection results.

We use the minimum over diferent prototype graphs instead of the average because a test window should be considered as normal if its behavior is close to any learned normal operating condition. Since structural deviation and predictive deviation may have diferent scales, we use the sum ofz-score normalized deviations as the anomaly score.

## Experiments

In this section, we evaluate the efectiveness of the proposed GSLAD approach for MTS anomaly detection. First, we introduce the experimental settings. Then, we assess the effectiveness of GSLAD on anomaly detection tasks on four industrial datasets and we conduct ablation studies and parameter sensitivity analysis to investigate the contributions of individual components in GSLAD. Finally, we analyze the structural deviation patterns to perform anomaly diagnosis.

## Experiment Settings

Datasets We conduct experiments on four industrial datasets, namely, the Secure Water Treatment (SWaT) dataset (Mathur and Tippenhauer 2016), the Water Distribution (WADI) dataset (Ahmed, Palleti, and Mathur 2017), the Pooled Server Metrics (PSM) dataset (Abdulaal, Liu, and Lancewicki 2021), and the Tennessee Eastman Process (TEP) dataset (Reinartz, Kulahci, and Ravn 2021). SWaT and WADI contain sensor and actuator measurements collected from water-treatment and water-distribution testbeds, respectively. PSM consists ofserver-level monitoring metrics, while TEP simulates a multivariable chemical process with multiple fault types.

Baselines We compare the proposed GSLAD model with thirteen representative baselines covering forecasting-based, reconstruction-based, representation-based, and graph-based anomaly detection methods, including MTAD-GAT (Zhao et al. 2020), GDN (Deng and Hooi 2021), GRELEN (Zhang, Zhang, and Tsung 2022), Anomaly Transformer (Xu et al. 2022), NLinear (Zeng et al. 2023), DLinear (Zeng et al.

2023), DCdetector (Yang et al. 2023), PatchTST (Nie et al. 2023), DyEdgeGAT (Zhao and Fink 2024), iTransformer (Liu et al. 2024), ModernTCN (donghao and wang xue 2024), CATCH (Wu et al. 2025), and GCAD (Liu, Gao, and Jiao 2025). Details on the baselines are given in the Appendix.

Implementation Details Each dataset consists of two parts: unlabeled data under normal working conditions and labeled data containing some anomalies. We use 80% of the unlabeled normal data for training, and the remaining 20% is used for validation. Testing is conducted on the labeled data containing anomalies. The data is standardized using the mean and standard deviation computed from the normal training split, and the resulting time series is segmented into sliding windows. Since most methods do not provide a way to set predetermined thresholds, we evaluate using two threshold-independent metrics: the area under the curve (AUC) of the Receiver Operating Characteristic (ROC) (Fawcett 2006) and the Precision-Recall Curve (PRC) (Davis and Goadrich 2006). More implementation details are provided in Appendix.

## Multivariate Time Series Anomaly Detection

The anomaly detection results of the proposed GSLAD model and thirteen baseline models on four industrial anomaly detection benchmarks are summarized in Table 1. GSLAD achieves the best ROC-AUC on SWaT, PSM, and TEP, and the best PR-AUC on all four datasets. These results indicate that GSLAD consistently achieves state-of-the-art performance in most cases. iTransformer performs well on two water network datasets, obtaining second-best results on three metrics. However, it fails to obtain competitive performance on PSM and TEP, especially on TEP where it obtains the worst results among all the baselines. GSLAD does not achieve the best result on every metric. On WADI, GCAD obtains a higher ROC-AUC. Nevertheless, GSLAD remains competitive on these metrics and provides the strongest overall performance across the four datasets.

<table><tr><td></td><td colspan="2">SWAT</td><td colspan="2">WADI</td><td colspan="2">PSM</td><td colspan="2">TEP</td></tr><tr><td></td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td><td>AUC-ROC</td><td>AUC-PR</td></tr><tr><td>GSLAD</td><td>0.8607</td><td>0.7612</td><td>0.7018</td><td>0.5157</td><td>0.8193</td><td>0.7038</td><td>0.9563</td><td>0.9687</td></tr><tr><td>w/o structural deviation</td><td>0.7449</td><td>0.2052</td><td>0.5925</td><td>0.2263</td><td>0.6585</td><td>0.3905</td><td>0.9169</td><td>0.9293</td></tr><tr><td>w/o Phase II training</td><td>0.7561</td><td>0.2170</td><td>0.6805</td><td>0.4781</td><td>0.7721</td><td>0.5344</td><td>0.9494</td><td>0.9620</td></tr><tr><td>w/o uncertainty awareness</td><td>0.8605</td><td>0.7630</td><td>0.6595</td><td>0.1217</td><td>0.7478</td><td>0.5874</td><td>0.9124</td><td>0.9199</td></tr><tr><td>w/o condition awareness</td><td>0.8489</td><td>0.7099</td><td>0.6756</td><td>0.4511</td><td>0.7785</td><td>0.5909</td><td>0.9405</td><td>0.9509</td></tr></table>

Table 2: Ablation Study.

## Ablation Study

To have a better understanding on how each individual component in GSLAD contribute to the improved performance, we conduct ablation studies in this section. Specifically, we consider four variants of GSLAD by removing individual components from the model: 1) ’w/o structural deviation’, which uses only the predictive deviation for anomaly scoring; 2) ’w/o Phase II training’, which trains the model only with the Phase I predictive objective $\mathcal { L } _ { \mathrm { p r e d } }$ and without the Phase II prototype-regularized refinement; 3) ’w/o uncertainty awareness’, which removes the use of $\sigma _ { k }$ as uncertainty weighting and computes the unweighted structural deviation defined as

$$
d _ { k } \big ( \mathbf { S } _ { t } \big ) = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } ( \mathbf { s } _ { t , i j } - \mu _ { k , i j } ) ^ { 2 } ,\tag{19}
$$

during both Phase II refinement training and inference stage; 4) ’w/o condition awareness’, which does not modulate the query projection and key projection with the global state vector. Table 2 summarizes the results for these four variants of GSLAD.

From the results, we observe that all four individual components of GSLAD contribute positively to the anomaly detection performance. Among them, removing the structural deviation from anomaly scoring leads to the largest overall degradation, confirming that structural deviation provides the critical anomaly evidence in GSLAD. The performance drop is particularly significant on SWaT, WADI, and PSM, indicating that predictive deviation alone is insuficient to distinguish anomalies in such cases. Removing Phase II training also substantially degrades performance, especially on SWaT and PSM. This observation validates that the more compact and stable structural patterns facilitated by prototyperegularized refinement can help anomaly detection. Replacing the uncertainty-aware distance with an unweighted one consistently reduces performance, with especially large decreases on WADI, PSM, and TEP. This result demonstrates that deviations on stable edges provide stronger anomaly signals than deviations on edges that naturally fluctuate during normal operating regimes.

## Sensitivity analysis

The number of prototypes used in GSLAD is an important hyperparameter for obtaining efective anomaly evidence. We perform a sensitivity analysis of the model performance with respect to the prototype number K using the TEP dataset.

![](images/3db4a82e22d9f81bce57174a5acac0ef31b705e074908147a561867699a88019.jpg)  
Figure 2: Sensitivity to prototype number K.

The results are visualized in Figure 2. The best result is obtained when $K = 4$ . The results show that GSLAD performs consistently well when the number of prototypes is within a reasonable range close to the optimal prototype number. Specifically, with K = 2 or $K = 8 .$ , compared with the second-best baseline model in Table 1, GSLAD still obtains better AUC-ROC and similar AUC-PR. However, with an overly large or small number of prototypes, the model performance becomes worse than other baselines. This is because too many prototypes make the clusters and statistics inaccurate and unstable, while too few prototypes are not enough to cover all the diferent normal operating regimes and may cause prototypes that correspond to no actual operating regime.

## Cross-Fault Structural Analysis

In this section, we examine whether the learned structural space captures fault-specific structural patterns. This analysis is conducted on TEP because it contains 17 diferent fault types covering diferent fault mechanisms with annotations. Note that these labels are used only for post-hoc interpretation and are never used during model training or anomaly scoring. For each fault type i, we first construct a class-level reference graph as follows:

$$
\mathbf { F } _ { i } = \frac { 1 } { | \mathcal { M } _ { i } | } \sum _ { m \in \mathcal { M } _ { i } } \mathbf { S } ^ { ( m ) } ,\tag{20}
$$

where $\mathcal { M } _ { i }$ denotes the set of windows belonging to fault type i and $\mathbf { S } ^ { ( m ) }$ is the inferred graph of window m. We then

![](images/cb5d483c61547e10134781480939eb98fad2a24eb6df62456216e2d324b28e11.jpg)  
Figure 3: Cross-fault structural matrix.

compute the mean distance from samples of fault type j to the reference graph of fault type i as follows:

$$
C _ { i j } = \frac { 1 } { | \mathcal { M } _ { j } | } \sum _ { m \in \mathcal { M } _ { j } } \| \mathbf { S } ^ { ( m ) } - \mathbf { F } _ { i } \| .\tag{21}
$$

We visualize the cross-fault structural matrix in Figure 4. The matrix exhibits a clear diagonal pattern that all the diagonal entries are the minimum in the corresponding rows and columns. On average, the within-class distance is 34.1% lower than the corresponding nearest competing class. This observation indicates that the inferred graphs are generally closest to the reference graph of the same fault type. Thus, we can conclude that empirically, the inferred graph generated with GSLAD would deviate in diferent directions for diferent fault types, despite that the model is trained without fault type supervision. Among all the faults, F2, F12, and F14 exhibit particularly large margins from competing classes, suggesting that they induce most distinctive changes in structural patterns. Overall, the analysis shows that graph structures learned by GSLAD provide information beyond binary anomaly detection and may support fault characterization.

## Structural Deviation Interpretation

To investigate whether structural deviation can support fault diagnosis such as root-cause localization, we analyze Fault 14 as an example, which represents a reactor cooling-water valve-sticking fault that mainly afects the reactor coolingwater subsystem. For each anomalous window, we first compute the uncertainty-normalized edge-deviation matrix with respect to the closest normal prototype:

![](images/92b8656f3bdf10285a3f3a792c8122a8ff1c3c0604d59dc6e1efe593f6b5fd54.jpg)  
Figure 4: Node-level deviation ranking.

$$
D _ { i j } = \operatorname* { m i n } _ { k \in \{ 1 , \dots , K \} } \frac { ( s _ { i j } - \mu _ { k , i j } ) ^ { 2 } } { \sigma _ { k , i j } ^ { 2 } + \sigma _ { 0 } ^ { 2 } } .\tag{22}
$$

We then define the node-level deviation score as the average of its incoming and outgoing edge deviations:

$$
\rho _ { v } = \frac { 1 } { 2 } \left( \frac { 1 } { N } \sum _ { j } D _ { v j } + \frac { 1 } { N } \sum _ { i } D _ { i v } \right) .\tag{23}
$$

Figure 4 presents the variables with largest $\rho _ { v }$ . The three most afected nodes are XMV(10), XMEAS(9), and XMEAS(21), corresponding to reactor cooling-water flow, reactor temperature, and cooling-water outlet temperature, respectively. These variables belong to the subsystem directly associated with Fault 14, and the third-ranked variable has a deviation more than 6.4 times larger than the fourth-ranked variable. This concentration of node-level deviation suggests that the uncertainty-aware structural deviation provides a useful signal for localizing the root cause.

More experiments on runtime analysis and anomaly score time series analysis are provided in the Appendix.

## Conclusion

In this paper, we introduced GSLAD, a prototyperegularized graph structure learning framework for unsupervised MTS anomaly detection. GSLAD learns windowdependent graphs and represents normal structural behavior with a bank of prototypes, enabling structural deviation to complement predictive deviation for anomaly scoring. Experiments demonstrate strong overall performance of GSLAD, while structural analyses suggest potential for fault characterization and root-cause localization. Future work will investigate graph structure learning with stronger causal interpretability for industrial anomaly detection.

## References

Abdulaal, A.; Liu, Z.; and Lancewicki, T. 2021. Practical approach to asynchronous multivariate time series anomaly detection and localization. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, 2485–2494.

Ahmed, C. M.; Palleti, V. R.; and Mathur, A. P. 2017. WADI: a water distribution testbed for research in the design of secure cyber physical systems. In Proceedings of the 3rd international workshop on cyber-physical systems for smart water networks, 25–28.

Audibert, J.; Michiardi, P.; Guyard, F.; Marti, S.; and Zuluaga, M. A. 2020. Usad: Unsupervised anomaly detection on multivariate time series. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 3395–3404.

Belay, M. A.; Blakseth, S. S.; Rasheed, A.; and Salvo Rossi, P. 2023. Unsupervised anomaly detection for IoT-based multivariate time series: Existing solutions, performance analysis and future directions. Sensors, 23(5): 2844.

Brody, S.; Alon, U.; and Yahav, E. 2022. How Attentive are Graph Attention Networks? In International Conference on Learning Representations.

Chen, H.; and Eldardiry, H. 2024. Graph time-series modeling in deep learning: A survey. ACM Transactions on Knowledge Discoveryfrom Data, 18(5): 1–35.

Cho, D.; Han, J.; Kang, K.; Kim, M.; Ryu, H.; and Jung, N. 2025. Structured temporal causality for interpretable multivariate time series anomaly detection. Advances in Neural Information Processing Systems, 38: 59257–59294.

Davis, J.; and Goadrich, M. 2006. The relationship between Precision-Recall and ROC curves. In Proceedings of the 23rd international conference on Machine learning, 233–240.

Deng, A.; and Hooi, B. 2021. Graph neural network-based anomaly detection in multivariate time series. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, 4027–4035.

donghao, L.; and wang xue. 2024. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. In The Twelfth International Conference on Learning Representations.

Fawcett, T. 2006. An introduction to ROC analysis. Pattern recognition letters, 27(8): 861–874.

Fink, O.; Sharma, V.; Nejjar, I.; Von Krannichfeldt, L.; Garmaev, S.; Zhang, Z.; Wei, A.; Frusque, G.; Forest, F.; Zhao, M.; et al. 2026. From Physics to Machine Learning and Back: Part I-Learning with Inductive Biases in Prognostics and Health Management. Reliability Engineering & System Safety, 112213.

Ho, T. K. K.; Karami, A.; and Armanfard, N. 2025. Graph anomaly detection in time series: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Jin, M.; Koh, H. Y.; Wen, Q.; Zambon, D.; Alippi, C.; Webb, G. I.; King, I.; and Pan, S. 2024. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. IEEE transactions on pattern analysis and machine intelligence, 46(12): 10466–10485.

Jin, W.; Ma, Y.; Liu, X.; Tang, X.; Wang, S.; and Tang, J. 2020. Graph structure learning for robust graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 66–74.

Kingma, D.; and Ba, J. 2015. Adam: A Method for Stochastic Optimization. In Proceedings of the International Conference on Learning Representations.

Kiranyaz, S.; Avci, O.; Abdeljaber, O.; Ince, T.; Gabbouj, M.; and Inman, D. J. 2021. 1D convolutional neural networks and applications: A survey. Mechanical systems and signal processing, 151: 107398.

Li, Z.; Wang, L.; Sun, X.; Luo, Y.; Zhu, Y.; Chen, D.; Luo, Y.; Zhou, X.; Liu, Q.; Wu, S.; et al. 2023. Gslb: The graph structure learning benchmark. Advances in Neural Information Processing Systems, 36: 30306–30318.

Liu, Y.; Hu, T.; Zhang, H.; Wu, H.; Wang, S.; Ma, L.; and Long, M. 2024. iTransformer: Inverted transformers are effective for time series forecasting. In International conference on learning representations, volume 2024, 11116– 11140.

Liu, Y.; Zheng, Y.; Zhang, D.; Chen, H.; Peng, H.; and Pan, S. 2022. Towards unsupervised deep graph structure learning. In Proceedings of the ACM web conference 2022, 1392– 1403.

Liu, Z.; Gao, M.; and Jiao, P. 2025. Gcad: Anomaly detection in multivariate time series from the perspective of granger causality. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 19041–19049.

Mathur, A. P.; and Tippenhauer, N. O. 2016. SWaT: A water treatment testbed for research and training on ICS security. In 2016 international workshop on cyber-physical systems for smart water networks (CySWater), 31–36. IEEE.

Nie, Y.; Nguyen, N. H.; Sinthong, P.; and Kalagnanam, J. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In The Eleventh International Conference on Learning Representations.

Reinartz, C.; Kulahci, M.; and Ravn, O. 2021. An extended Tennessee Eastman simulation dataset for fault-detection and decision support systems. Computers & chemical engineering, 149: 107281.

Wu, X.; Qiu, X.; Li, Z.; Wang, Y.; Hu, J.; Guo, C.; Xiong, H.; and Yang, B. 2025. Catch: Channel-aware multivariate time series anomaly detection via frequency patching. In International conference on learning representations, volume 2025, 17017–17045.

Wu, Y.; Dai, H.-N.; and Tang, H. 2021. Graph neural networks for anomaly detection in industrial Internet of Things. IEEE Internet ofThings Journal, 9(12): 9214–9231.

Xu, J.; Wu, H.; Wang, J.; and Long, M. 2022. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. In International Conference on Learning Representations.

Yang, Y.; Zhang, C.; Zhou, T.; Wen, Q.; and Sun, L. 2023. Dcdetector: Dual attention contrastive representation learning for time series anomaly detection. In Proceedings ofthe

29th ACM SIGKDD conference on knowledge discovery and data mining, 3033–3045.

Zeng, A.; Chen, M.; Zhang, L.; and Xu, Q. 2023. Are transformers efective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, 11121–11128.

Zhang, C.; Song, D.; Chen, Y.; Feng, X.; Lumezanu, C.; Cheng, W.; Ni, J.; Zong, B.; Chen, H.; and Chawla, N. V. 2019. A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data. In Proceedings of the AAAI conference on artificial intelligence, volume 33, 1409–1416.

Zhang, W.; Zhang, C.; and Tsung, F. 2022. GRELEN: Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning. In IJCAI, 2390–2397.

Zhang, Z.; Lu, S.; Huang, Z.; and Zhao, Z. 2024. Graph Neural Networks With Adaptive Structures. IEEE Journal of Selected Topics in Signal Processing, 19(1): 181–194.

Zhao, H.; Wang, Y.; Duan, J.; Huang, C.; Cao, D.; Tong, Y.; Xu, B.; Bai, J.; Tong, J.; and Zhang, Q. 2020. Multivariate time-series anomaly detection via graph attention network. In 2020 IEEE international conference on data mining (ICDM), 841–850. IEEE.

Zhao, M.; and Fink, O. 2024. DyEdgeGAT: Dynamic Edge via Graph Attention for Early Fault Detection in IIoT Systems.

Zheng, Y.; Koh, H. Y.; Jin, M.; Chi, L.; Phan, K. T.; Pan, S.; Chen, Y.-P. P.; and Xiang, W. 2023. Correlation-aware spatial–temporal graph learning for multivariate time-series anomaly detection. IEEE transactions on neural networks and learning systems, 35(9): 11802–11816.

Zhu, Y.; Xu, W.; Zhang, J.; Du, Y.; Zhang, J.; Liu, Q.; Yang, C.; and Wu, S. 2021. A survey on graph structure learning: Progress and opportunities. arXiv preprint arXiv:2103.03036.

# Appendix

## Baselines

To evaluate the performance of our proposed GSLAD method, we compare it with various baselines. We briefly introduce these baselines below:

• MTAD-GAT (Zhao et al. 2020) employs parallel feature- and temporal-oriented graph attention layers and jointly optimizes forecasting and reconstruction objectives for multivariate time-series anomaly detection.

• GDN (Deng and Hooi 2021) learns a sparse dependency graph among sensors through node embeddings and graph attention, and detects anomalies based on forecasting deviations from the learned normal behavior.

• GRELEN (Zhang, Zhang, and Tsung 2022) integrates a variational autoencoder with stochastic graph relational learning to capture inter-sensor dependencies and construct a relation-aware anomaly score.

• Anomaly Transformer (Xu et al. 2022) introduces anomaly attention and a minimax learning strategy to distinguish anomalies through the discrepancy between prior and series associations.

• NLinear (Zeng et al. 2023) mitigates distribution shifts by subtracting the last observed value before linear forecasting and adding it back to the prediction.

• DLinear (Zeng et al. 2023) decomposes each time series into trend and seasonal components and forecasts them using separate linear mappings.

• DCdetector (Yang et al. 2023) adopts multi-scale dual-attention contrastive learning to obtain discriminative representations without relying on a reconstruction objective.

• PatchTST (Nie et al. 2023) divides each time series into temporal patches and processes diferent variables independently with a weight-shared Transformer encoder.

• DyEdgeGAT (Zhao and Fink 2024) dynamically infers evolving inter-sensor edges and incorporates operating-condition context into graph-attention-based signal reconstruction.

• iTransformer (Liu et al. 2024) represents individual variables as tokens, using self-attention to capture inter-variable dependencies and feed-forward networks to learn temporal representations.

• ModernTCN (donghao and wang xue 2024) modernizes temporal convolutional networks with large-kernel depthwise convolutions and convolutional feed-forward blocks to capture long-range temporal patterns eficiently.

• CATCH (Wu et al. 2025) partitions signals into frequency-domain patches and employs masked channel fusion to model fine-grained spectral characteristics and channel correlations.

• GCAD (Liu, Gao, and Jiao 2025) dynamically extracts Granger-causality graphs from predictor gradients and detects anomalies through deviations in the learned causal patterns.

For forecasting models, anomaly scores are computed from forecasting deviations under the same evaluation protocol. Similar to GCAD, there is another recent work that infers the graph for anomaly detection based on intermediate embeddings (Cho et al. 2025). We do not include it for comparison since there is no public code available. For methods implemented in (Wu et al. 2025), we use their implementation. For the other methods we use their publicly available oficial implementations.

## Implementation Details

All the experiments are conducted on a NVIDIA A100 80G GPU. For all the experimental results, we give the average performance and standard deviation with 5 independent trials. For all the datasets, we select windows of length 128. The Adam optimizer is used in all experiments for model training (Kingma and Ba 2015). We fix the training epochs of Phase I to 30 and the training epochs of Phase II to 10. We use a batch size of 256 for model training. The models’ hyperparameters are tuned based on the results of the validation set. The search space of hyperparameters are as follows: 1) horizon: {1,4,8}; 2) GNN layers: {2, 4}; 3) embedding dimension: {64, 128, 256}; 4) weight parameter λ in the loss function: {1, 3, 10, 30, 100}; 5) number of prototypes: {1, 2, 4, 8}. The learning rate and weight decay are set to 1e-4 and 1e-5, respectively. For generating the sparse graphs, we keep the top-k outgoing neighbors for each node with k = 5. The temperature parameter τ for soft nearest prototype graph assignment during Phase II training is set to 0.05. The global stabilization term σ<sub>0</sub> used for computing uncertainty-aware graph deviation is set to 0.05. During inference, we use the sum of z-score normalized structural deviation and predictive deviation as the anomaly score, where we use the mean and standard deviation computed from training set to perform z-score normalization.

## Runtime Comparison

Since GSLAD involves additional computation for graph structure learning, we perform a runtime comparison to evaluate the eficiency of GSLAD. Specifically, we evaluate the per-sample inference time (in seconds) on four industrial datasets. The results are shown in Figure 5. From the results, we can see that the models that jointly learn graph structures generally require longer inference time. However, we observe that GSLAD remains substantially more eficient than the other baselines that also perform structure learning, including DyEdgeGAT, GCAD, and GRELEN. For example, on the SWAT dataset, GSLAD is 12.5 times, 3.3 times, and 4.8 times faster than DyEdgeGAT, GCAD, and GRELEN, respectively.

![](images/6fc95f54e85a98230c1a0aadce92a42b33a393085af39ef569d0c4a273eabc7b.jpg)  
Figure 5: Per-sample inference time comparison.

## Anomaly Score Time Series Analysis

In this section, We qualitatively examine how the anomaly score evolves around fault onset. Figure 6 visualizes the anomaly scores time series for two faults in TEP, with the anomaly score containing both uncertainty-aware structural deviation and predictive deviation. Note that GSLAD treats each window independently and computes the anomaly score for each window separately. Both examples show a clear and sustained increase on anomaly score immediately after fault onset, indicating GSLAD provides strong anomaly signals.

![](images/4a382745a9d28600cc792a27b3ea460967cc8dc4edc4620a5469679137c811d3.jpg)

![](images/3e10f122b2ead5ebd56eb0e0e1b6179f1649073fd29b5fc24f59abe1253a1679.jpg)  
Figure 6: Anomaly score time series. The orange line represents the start time of the anomaly case.