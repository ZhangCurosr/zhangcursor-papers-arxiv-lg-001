# Evolving Federated Density Peak Clustering with Shufler-Based Anonymization

Miha Oˇzbot<sup>1\*</sup>, Seiichi Ozawa<sup>2</sup> and Igor Skrjanc <sup>ˇ</sup> <sup>1</sup>

<sup>1\*</sup>Faculty of Electrical Engineering, University of Ljubljana, Trˇzaˇska cesta 25, Ljubljana, 1000, Slovenia.

<sup>2</sup>Center for Mathematical and Data Sciences, Kobe University, 1-1 Rokkodai-cho, Nada-ku, Kobe, 657-8501, Japan.

\*Corresponding author(s). E-mail(s): miha.ozbot@fe.uni-lj.si; Contributing authors: ozawasei@kobe-u.ac.jp; igor.skrjanc@fe.uni-lj.si;

## Abstract

In this paper, we present a novel one-shot federated clustering approach based on evolving fuzzy clustering and shufler-based anonymization. In federated clustering, each round reveals additional model information to the server and can increase the opportunity to infer properties of local datasets, even when raw samples are never shared. Consequently, fewer client–server exchanges and unlinkability are important, so transmitted information cannot be attributed to any single client. Data-stream clustering algorithms, which learn in a single pass over the data, align well with the one-shot federated learning paradigm, where a single communication round can be suficient. The proposed Evolving Federated Density Peak Clustering (eFDPC) method combines evolving fuzzy data-stream clustering and density peak clustering in a federated setting. Each client independently evolves micro-clusters from private data mini-batches through cluster addition and merging. To limit communication, density peak clustering is applied to the learned micro-clusters and only the selected peaks are transmitted, rather than the full local models. A shufler-based layer is introduced between the clients and the server to anonymize contributions prior to reconstruction and global peak selection. We consider a peak-only variant, eFDPC-P, as well as eFDPC-G with server-side label propagation. Experiments on synthetic and real-world benchmark datasets show that eFDPC outperforms recent federated clustering baselines in one-shot settings and scales to higher-dimensional, non-IID data.

Keywords: Federated learning, Streaming clustering, Non-IID data, Evolving fuzzy systems, unsupervised learning, Density peak clustering

## 1 Introduction

A fundamental challenge in recent years is the sharing of information between diferent entities in sectors like manufacturing, government, finance, and healthcare, etc., while maintaining data privacy [1]. Federated Learning (FL) addresses the challenge of learning from distributed data that cannot be shared directly due to concerns such as privacy, economics, and security, inter alia. The contributors of the data, typically referred to as clients, benefit from shared information. However, supervised learning poses a challenge in a federated setting, as most real-world data are unlabeled. Additionally, the interpretability of the federated model is crucial in fields like finance and healthcare, where understanding the underlying system and trusting the model are essential [2, 3]. Federated clustering (FC) ofers a solution to this problem by enabling clustering analysis and the sharing of expert knowledge without requiring access to the data itself. One client might possess a large amount of data, while another might have expert knowledge. Unsupervised Federated Learning can enable the utilization of previously unavailable data to increase information utilization. The data are commonly non-independently and non-identically distributed (non-IID) among clients [4, 5], i.e., the task is shared, but local distributions difer, with each client having a distinct number of samples and class representation. None of the clients possess the global cluster centroids, so they must collaborate by sharing their local models to obtain a consistent global solution. Collaboration in creating the federated model is beneficial to each individual client, from one that desires information but does not have data, to a client that possesses the data but lacks the resources or knowledge to utilize it, to the client that has biased or partial data.

Traditional clustering methods require specifying the number of clusters, typically focusing on minimizing the average distance between samples and cluster centers or maximizing the likelihood of the observed data under the model. However, this can be problematic in a federated learning setting, where the number of clusters may not be known in advance. The number of clusters may vary for each client, and the clusters and their number may change over time, e.g., fraudulent financial transactions [6], malicious network activity [7], etc. Clients would benefit from shared information to address new challenges quickly. We propose that evolving data stream clustering, which was designed to adapt clusters incrementally over time and add new ones when necessary, is well suited to address these challenges.

Evolving Fuzzy Systems (EFS) [8] are designed to learn from streaming data, where each sample arrives sequentially and is subject to concept drift. As data characteristics change over time, these algorithms adapt clusters incrementally and introduce new clusters as needed. The system aims to extract maximum information from each individual sample before discarding it. While a limitation of this approach is that processing samples sequentially does not scale with available processing capacity and results in significant memory access overhead, we modify the usual streaming clustering approach to learn from mini-batches of data instead, which aligns better with federated learning. Evolving systems typically generate a large number of micro-clusters that capture local structure in the data, so a subsequent selection step is required to identify representative peaks and remove redundant or outlier micro-clusters. While the evolving-systems literature provides mature mechanisms for adding, merging, splitting, removing, and forgetting clusters, using full multivariate covariance matrices becomes costly as dimensionality increases. Gaussian micro-clusters using diagonal covariances scale linearly with data dimensionality, and the final clustering can be done by selecting and grouping these micro-clusters.

![](images/534c985045f0e473d068f30671a2d7939ec34f027264ff78dcb1cef44b93f04c.jpg)  
Fig. 1 Overview of the proposed eFDPC pipeline. (1) Each client learns evolving micro-clusters from its private data using mini-batch updates. (2) Each client applies a local DPC step on its micro-... ...clusters to select a small set of local peaks. (3) The selected peak statistics are split into additive shares and sent to multiple shuflers. Each shufler permutes the received messages and forwards them to the server. (4) The server reconstructs the peak statistics by matching share identifiers and summing the shares, obtaining an unlinkable pool of candidate peaks. (5) The server performs a second DPC step on the reconstructed candidates to select the global peak set, and (6) broadcasts the global peaks back to clients for the next communication round.

Density Peak Clustering (DPC) [9] identifies cluster centers as points that are both locally dense and well separated from higher-density points. Classical DPC relies on global pairwise comparisons to rank densities and to determine nearest-higher relationships, which is incompatible with federated settings where raw samples cannot be centralized. Each client first performs evolving clustering on private mini-batches that results in local micro-clusters. DPC is then applied locally to the learned microclusters to select a small set of representative peak candidates, rather than sharing all micro-clusters or raw samples.

Although raw samples are never shared, transmitted peak candidates may still be attributable to individual clients across rounds, potentially leaking information about local data distributions. To prevent attribution, we introduce a shufler-based anonymization layer between clients and the server. Each client splits each peak statistic into additive shares, shuflers permute the received share-messages, and the server reconstructs peaks only after mixing. The server thus obtains an unlinkable pool of candidate peaks, performs a second DPC step to select the global peak set, and can optionally propagate peak labels to obtain a full grouping over the reconstructed representatives.

The proposed Evolving Federated Density Peak Clustering (eFDPC) algorithm is a one-shot federated adaptive clustering framework for heterogeneous decentralized data and is illustrated in Fig. 1. The main contributions of the paper are:

• An improved evolving fuzzy clustering methodology that operates on mini-batches instead of the traditional online-learning setting, together with a novel measure for computing cluster overlap for merging.

• A novel density peak clustering approach that operates on micro-clusters rather than data samples.

• A federated clustering approach that combines evolving clustering and density peak clustering with a shufler layer to provide client anonymity.

The paper is structured as follows. Section 1 introduces the problem and reviews related work on federated clustering, evolving systems, and Density Peak Clustering. Section 2 describes the proposed evolving systems methodology, the DPC peak selection, and the federated clustering framework. The experimental setup and results are presented in Section 3. A discussion is provided in Section 4, and the conclusions are presented in Section 5.

## 1.1 Related Literature

Prototype-based federated clustering methods like k-means [10–13] and fuzzy C-means (FCM) [11, 14, 15] were proposed by researchers to address diferent challenges of federated learning, i.e., one-shot learning [10, 13], non-IID data [10, 14], multiview data [15], unlearning client data [12], vertical feature partitioning [11]. A key methodological distinction of these clustering approaches lies in the aggregation process at the server. For instance, clusters may have one-to-one relationships, the clustering process may be repeated at the server using the cluster centers as samples, or clusters may be iteratively merged based on specific selection criteria. One of the first proposed federated clustering method was k-FED [10], which modified the classic k-means method for federated learning. In k-FED, local clusters are generated using a k-means method and then aggregated at the server by iteratively selecting the most separated cluster centers from the local clusters, followed by performing k-means on all local centers. This approach allows each local client to have a diferent number of clusters, though the number must be specified in advance. Similarly, a one-shot learning Federated Centroid Aggregation (FeCA) method was proposed in [13], also based on k-means, which combines clusters by iteratively selecting the cluster with the largest radius and merging it with nearby clusters. In contrast, the k-means++ clustering approach was used in MUFC [12] to improve cluster initialization for the local clients and to aggregate the local clusters at the server. The LossLess Federation k-means (LLF-CM) and FCM (LLF-FCM) methods proposed in [11] address horizontal and vertical data partitioning. The initial set of cluster centers is selected from one of the clients. The server then aggregates the local counts of points and the sum of sample values for each cluster, updating the global cluster centers accordingly. In FedFCM [14], clusters are also initialized by one of the clients. Each client clusters its data using FCM with the global clusters as initial values and communicates the updated cluster centers and sum of memberships or support. Notably, the authors highlight the challenge of selecting the appropriate number of clusters as an unsolved problem. A federated multiview fuzzy C-means consensus prototypes clustering (FedMVFPC) approach was proposed in [15], demonstrating that achieving consensus among clusters from multiple views is beneficial. In federated multiple imputation-based fuzzy (FeMIFuzzy) method [3] clients vote on the optimal number of clusters using the Xie-Beni index. Those federated prototype-based clustering methods are based on iterative procedures, repeating local reclustering and server aggregation over multiple communication rounds. Even when a fixed message size enables secure aggregation, repeated rounds increase communication cost and can make prototype trajectories linkable across rounds.

Fuzzy systems were designed to handle data uncertainty and noise, making them suitable for federated learning scenarios where data can be non-IID across devices [16]. Fuzzy models have recently been proposed for distributed and federated learning. In this context, clustering approaches are used to define the antecedent structure of the model. Fuzzy neural networks have been applied in distributed learning [17], using k-means clustering, and in distributed semi-supervised learning [18], employing the distributed FCM clustering method. Furthermore, fuzzy models commonly rely on a consequent local linear model for regression and classification tasks. A federated learning system (Fed-FRBS) based on the Takagi-Sugeno Fuzzy Rule-Based System was proposed in [19], with an emphasis on Explainable AI. Fuzzy c-means clustering is used to determine the model antecedents and weigh contributions to the federated learning model based on prediction quality. FedFNN [16] proposed a federated fuzzy neural network that learns antecedent Gaussian centers and variances through backpropagation. To handle non-IID data and uncertainties, an evolutionary algorithm is used. EFS extend the fuzzy modeling approach by integrating unsupervised clustering techniques to incrementally learn and partition data in an online manner [8].

EFS incrementally cluster data to create membership functions for fuzzy rules and can add, merge, split, and remove clusters based on diferent criteria. These methods are designed for online classification [20, 21], regression [22], time series forecasting [23–25], and clustering [26–28] in evolving environments or streaming data. Several evolving clustering approaches were proposed based on Gaussian distribution [23, 24, 26, 27], Cauchy distribution [7], data clouds [29, 30] data granules [31], interval prototypes [28], among others. Each cluster definition also requires a tailored methodology for incremental learning, as well as a cluster merging metric and computation. Specifically, evolving Gaussian clustering is used to determine information granulation in [23]. The granules are compared based on the Kullback–Leibler (KL) divergence, which is a measure of distribution dissimilarity. An evolving Cauchy clustering (eCauchy) method was proposed in [7] for monitoring and identification in cybersecurity intrusion detection. The evolving volume-based Gaussian Clustering (eGauss+) approach [26] was designed to learn from data streams for clustering and regression tasks. An incremental version of Gustafson–Kessel clustering method was proposed in [24] to identify the antecedent clusters of an evolving fuzzy system for modeling a waste-water treatment facility. Incrementally learned granules of data were used to describe the antecedent of an evolving granular fuzzy-rule-based modeling method in [31]. An autonomous learning multi-model (ALMMo) evolving classifier, which uses the data cloud-based Cauchy clusters model, was proposed in [29]. Similarly, Auto-Cloud [30] relies on Typicality and Eccentricity of the samples instead of defining a prototype shape. An approach using the Recursive least squares was used in [27] to cluster streaming data based on the error of the model prediction. Evolving learning from data chunks instead of per sample was explored in [25] similar to mini-batches with a changing number of samples. Recently, an evolving federated learning method was introduced in [32] for supervised learning, that uses evolving fuzzy systems that can dynamically add clusters, starting from scratch. Most evolving clustering methods are designed for centralized streams and rely on sample-level access and second-order statistics, with merge and evaluation steps that are sensitive to tuning. Extending these mechanisms to federated settings is challenging due to limited communication, heterogeneous local evolution, and privacy constraints on transmissible parameters.

DPC has recently received increased attention due to its ability to identify density peaks and model manifold-like cluster structures. The core idea of DPC is to compute a local density estimate for each sample and the distance from that sample to its nearest neighbor of higher density. Samples that have simultaneously high density and large distance to higher-density samples are selected as cluster centers, and remaining samples are assigned by following the density gradient [33]. Recently, the DPC research focuses more on local density for neighboring points rather than relying on global distances, especially in high-dimensional settings. A common strategy is to restrict computations to k-nearest neighbors (kNN), which reduces the impact of irrelevant distances in high dimensions and makes distance estimates more local and comparable [34–36]. DPC and its variants have been applied to intrusion detection systems (IDS) as well [37]. Further extensions combine kNN with dimensionality reduction (e.g., principal component analysis (PCA)) to stabilize local neighborhoods and improve center selection [36]. Robust variants of DPC have also been proposed to improve stability and reduce sensitivity to density perturbations [38]. To reduce computational cost, one approach first approximates the dataset using representative granular-ball centers and then performs clustering on these representatives instead of on raw samples [38]. In federated or distributed settings, several designs combine local summarization with server-side peak selection. For example, one-shot pipelines can perform clientside k-means summarization and then apply SNN-DPC center selection at the server [39]. Horizontal federated DPC (HFDPC) has also been studied as a direct adaptation of DPC to horizontally partitioned data [40]. To mitigate the so-called domino efect in assignments caused by unstable nearest-higher relations, some methods explicitly restrict the distance-to-higher computation to the k nearest neighbors [41]. While this improves stability, it typically introduces an additional parameter that constrains the acceptable neighbor range (or neighborhood validity), which must be selected carefully for each dataset [41].

## 2 Methods

In this section, we present the methodological components of the eFDPC algorithm, including evolving clustering, density peak clustering, and federated learning.

## 2.1 Evolving Clustering

Let $\pmb { X } = [ \pmb { x } _ { i } ] ^ { \top } \in \mathbb { R } ^ { N \times d }$ denote the data matrix, where $\mathcal { I } = \{ 1 , \ldots , N \}$ is the index set of data points, $N \in \mathbb { Z } ^ { + }$ is the number of samples, and $d \in \mathbb { Z } ^ { + }$ is the feature dimension. The objective of the local clustering is to partition X into a set of micro-clusters

$$
\mathcal { C } = \{ \mathcal { C } _ { 1 } , \ldots , \mathcal { C } _ { C } \} = \big \{ ( \Sigma _ { c } , \pmb { \mu } _ { c } , n _ { c } ) \big | c \in \{ 1 , \ldots , C \} \big \} ,\tag{1}
$$

where each micro-cluster is defined by a diagonal covariance matrix $\pmb { \Sigma } _ { c } \in \mathbb { R } ^ { d \times d }$ , a center $\mu _ { c } \in \mathbb { R } ^ { d }$ , and the support $n _ { c } \in \mathbb { R } _ { + }$ . Evolving clustering incrementally adapts the number of micro-clusters $C$ to represent all samples while suppressing redundant overlap by merging. This can be summarized by the representation condition:

$$
d _ { c ^ { * } i } ^ { 2 } \leq \kappa _ { \sigma } ^ { 2 } , \quad \forall i \in \mathcal { I } , \quad \exists c ^ { * } \in \{ 1 , \ldots , C \} ,\tag{2}
$$

where $d _ { c ^ { * } i } ^ { 2 }$ is the squared Mahalanobis distance from sample $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ to its nearest cluster center, and $\kappa _ { \sigma } \in \mathbb { R } ^ { + }$ is the representation threshold. The representation condition is complemented by the post-merge separability condition:

$$
\kappa _ { p q } > \kappa _ { m } , \qquad \forall p \neq q , \quad p , q \in \{ 1 , \ldots , C \} ,\tag{3}
$$

where $\kappa _ { p q } \in ( 1 , \infty )$ denotes the overlap score between micro-clusters $\mathcal { C } _ { p }$ and $\mathcal { C } _ { q } ,$ and $\kappa _ { m } \in ( 1 , \infty )$ is the merging threshold.

Evolving clustering consists of the following main steps: membership computation, incremental clustering, cluster adding mechanism, and cluster merging mechanism [8]. The evolving per-sample learning methodology is not directly applicable to federated learning since updates cannot be sent for each sample due to privacy and communication cost. eFDPC evolving operations are performed in parallel for all samples in the mini-batch and clusters simultaneously.

## 2.1.1 Membership computation

The Mahalanobis distance is used to calculate the distance between each sample $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and each cluster $\mathcal { C } _ { c }$

$$
\begin{array} { r } { d _ { c i } ^ { 2 } = \left( \pmb { x } _ { i } - \pmb { \mu } _ { c } \right) ^ { \top } \pmb { \Sigma } _ { c } ^ { - 1 } ( \pmb { x } _ { i } - \pmb { \mu } _ { c } ) , } \end{array}\tag{4}
$$

$$
u _ { c i } = \exp \left( - d _ { c i } ^ { 2 } / d \right) ,\tag{5}
$$

where $d _ { c i } ^ { 2 } \in \mathbb { R } ^ { + }$ is the squared Mahalanobis distance, $\pmb { u } _ { c } = [ u _ { c i } ] \in \mathbb { R } ^ { N }$ is the degree of association of the samples X with the cluster $\mathcal { C } _ { c } .$ . The scaling factor $1 / d$ is used to improve high-dimensional stability $\mathbb { E } [ d _ { c i } ^ { 2 } ] = d \ [ 4 2 , \ 4 3 ]$

## 2.1.2 Cluster adaptation

Let $\mathcal { T } _ { c }$ ∗ denote the index set of newly assigned samples that satisfy the representation constraint:

$$
\begin{array} { r } { \mathcal { T } _ { c ^ { * } } = \left\{ i \Big | c ^ { * } = \arg \underset { c } { \operatorname* { m i n } } d _ { c i } , { u } _ { c ^ { * } i } > \exp \left( - \kappa _ { \sigma } ^ { 2 } / d \right) \right\} . } \end{array}\tag{6}
$$

The parameters of the nearest assigned cluster are incrementally updated with the new samples:

$$
S _ { c ^ { * } } \gets S _ { c ^ { * } } + \sum _ { i \in \mathcal { T } _ { c ^ { * } } } ( { \pmb x } _ { i } - { \pmb \mu } _ { c ^ { * } } ) ^ { \odot 2 } ,\tag{7}
$$

$$
\pmb { \mu } _ { c ^ { * } } \gets \frac { n _ { c ^ { * } } \pmb { \mu } _ { c ^ { * } } + \sum _ { i \in \mathcal { T } _ { c ^ { * } } } \pmb { x } _ { i } } { n _ { c ^ { * } } + | \mathbb { T } _ { c ^ { * } } | } ,\tag{8}
$$

$$
n _ { c ^ { * } }  n _ { c ^ { * } } + | \mathcal { T } _ { c ^ { * } } | .\tag{9}
$$

where $\odot$ denotes the Hadamard (element-wise) product, so $( \pmb { x } _ { i } - \pmb { \mu } _ { c ^ { * } } ) ^ { \odot 2 }$ is the elementwise square, $| \mathcal { T } _ { c ^ { * } } |$ | is the cardinality of $\mathcal { T } _ { c ^ { * } } , \mathrm { i . e }$ . the number of samples assigned to cluster $c ^ { * }$ , and we store the diagonal scatter matrix $S _ { c }$ for eficient updates and compute $\pmb { \Sigma } _ { c } = \pmb { S } _ { c } / n _ { c }$ when needed. The update has $\mathcal O ( d )$ complexity for diagonal $\Sigma _ { c }$

## 2.1.3 Cluster adding

Let $c ^ { * } ( i ) = \arg \operatorname* { m i n } _ { c \in \{ 1 , \ldots , C \} } d _ { c i } ^ { 2 }$ . Samples that are not represented by any existing micro-cluster are indexed as:

$$
\begin{array} { r } { \mathcal { T } _ { + } = \left\{ i \in \mathcal { I } \ \middle | \ u _ { c ^ { * } ( i ) , i } \leq \exp \left( - \kappa _ { \sigma } ^ { 2 } / d \right) \right\} . } \end{array}\tag{10}
$$

For each $i \in \mathcal { Z } _ { + }$ , a new micro-cluster is initialized as:

$$
\mathcal C  \mathcal C \cup \{ ( \Sigma = \Sigma _ { 0 } , \mu = \pmb x _ { i } , n = 1 ) \} , \qquad i \in \mathcal { T } _ { + } .\tag{11}
$$

The initial covariance is initialized as:

$$
{ \Sigma } _ { 0 } = \mathrm { d i a g } \left( \pmb { \sigma } ^ { 2 } / \kappa _ { r } \right) ,\tag{12}
$$

where $\pmb { \sigma } ^ { 2 } \in \mathbb { R } ^ { d }$ is the incrementally estimated per-dimension data variance and $\kappa _ { r } \in$ $\mathbb { R } ^ { + }$ is a quantization parameter.

## 2.1.4 Merging mechanism

To account for the separability criterion in Eq. (3), overlapping clusters are merged. For two clusters $\mathcal { C } _ { p }$ and $\mathcal { C } _ { q } ,$ , the overlap score is defined as:

$$
\Sigma _ { p q } = \frac { { \cal S } _ { p } + { \cal S } _ { q } } { n _ { p } + n _ { q } } ,\tag{13}
$$

$$
d _ { p q } ^ { 2 } = ( { \pmb { \mu } } _ { p } - { \pmb { \mu } } _ { q } ) ^ { \top } { \pmb { \Sigma } } _ { p q } ^ { - 1 } ( { \pmb { \mu } } _ { p } - { \pmb { \mu } } _ { q } ) ,\tag{14}
$$

$$
\kappa _ { p q } = 1 + \frac { n _ { p } n _ { q } } { ( n _ { p } + n _ { q } ) ^ { 2 } } d _ { p q } ^ { 2 } .\tag{15}
$$

Smaller values of $\kappa _ { p q } \in ( 1 , \infty )$ correspond to stronger cluster overlap, with $\kappa _ { p q } = 1$ when $\mu _ { p } = \mu _ { q }$ . When $\Sigma _ { p q }$ is diagonal, the evaluation of $\kappa _ { p q }$ has complexity $\mathcal O ( d )$ . A derivation is provided in Appendix $\mathrm { A }$

<table><tr><td colspan="2">Algorithm 1 Evolving Micro-Cluster Learning (mini-batch, parallel)</td></tr><tr><td>Input: Mini-batch  $X _ { b } ,$  micro-clusters Output: Updated micro-clusters C</td><td> $\mathcal { C } , \kappa _ { \sigma } , \kappa _ { m } , \kappa _ { r }$ </td></tr><tr><td>Membership computation 1: Compute Mahalanobis distances</td><td> $d _ { c i } ^ { 2 }$  ▶ Eq. (4)</td></tr><tr><td>2: Compute memberships</td><td> $u _ { c i } .$  Eq. (5)</td></tr><tr><td></td><td>Incremental clustering (representation)</td></tr><tr><td></td><td>3: Determine assignment sets  $\mathcal { T } _ { c } .$  Eq. (6)</td></tr><tr><td>4: Update</td><td> $\{ n _ { c } , \pmb { \mu } _ { c } , S _ { c } \}$  simultaneously. ▶ Eqs. (7)−(9)</td></tr><tr><td></td><td>Cluster adding (representation failure)</td></tr><tr><td></td><td>5: Initialize new micro-clusters. ▶ Eqs. (11), (12)</td></tr><tr><td></td><td>Merging mechanism (separability)</td></tr><tr><td></td><td>6: Construct candidate pairs  $E _ { \kappa }$  ▶ Eq. (16)</td></tr><tr><td></td><td>7: Compute overlap scores  $\kappa _ { p q } .$  Eqs. (13)−(15)</td></tr><tr><td></td><td>8: Merge pairs satisfying  $\kappa _ { p q } \leq \kappa _ { m } .$  ▶ Eqs. (17)−(19)</td></tr></table>

Exhaustive evaluation of all cluster pairs $\{ p , q \}$ scales as $\mathcal { O } ( C ^ { 2 } d )$ and can become prohibitive in high dimensional problems [44]. We therefore restrict evaluation to a pruned $K _ { \kappa } .$ -nearest-neighbor (kNN) graph constructed on the micro-cluster centers:

$$
E _ { \kappa } = \Big \{ \{ p , q \} \Big | q \in \mathrm { k N N } ( \mu _ { p } , K _ { \kappa } ) , p \in \mathrm { k N N } ( \mu _ { q } , K _ { \kappa } ) , p \neq q \Big \} ,\tag{16}
$$

where $\mathrm { k N N } ( \mu _ { p } , K _ { \kappa } )$ returns the indices of the $K _ { \kappa }$ closest centers to µ under Euclidean $\mu _ { p }$ distance, and only mutual kNN pairs are retained to form an undirected sparse candidate graph. In the worst case, this results in at most $C K _ { \kappa } / 2$ undirected edges and restricts subsequent pairwise computations to these candidates.

Merging is performed greedily on the pruned graph $E _ { \kappa }$ using nearest neighbors. Each cluster $p$ selects the neighbor q with minimal $\kappa _ { p q }$ among those satisfying $\kappa _ { p q } \leq$ $\kappa _ { m }$ . The pair {p, q} is merged as [26, 32]:

$$
n _ { p q } = n _ { p } + n _ { q } ,
$$

$$
\mu _ { p q } = \frac { n _ { p } \mu _ { p } + n _ { q } \mu _ { q } } { n _ { p q } } ,\tag{17}
$$

(18)

$$
S _ { p q } = S _ { p } + S _ { q } + \frac { n _ { p } n _ { q } } { n _ { p q } } ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) ^ { \odot 2 } .\tag{19}
$$

The complete evolving clustering in mini-batches is presented in Algorithm 1.

## 2.2 Density Peak Clustering

Evolving micro-cluster learning can create too many micro-clusters, so C is often larger than the true number of clusters in the data. We adapt DPC [9] to select P representative peaks from the micro-clusters C. DPC assumes that cluster centers

exhibit both high local density and strong separation from other higher-density microclusters. For each micro-cluster $i \in \{ 1 , \ldots , C \}$ , we compute its density $\rho _ { i }$ and its separation [9]:

$$
{ \delta _ { i } = \operatorname* { m i n } _ { j : \rho _ { j } > \rho _ { i } } \| \pmb { \mu } _ { i } - \pmb { \mu } _ { j } \| _ { 2 } ^ { 2 } , }\tag{20}
$$

$$
\gamma _ { i } = \rho _ { i } \delta _ { i } ,\tag{21}
$$

where $\delta _ { i }$ is the squared Euclidean distance from $\pmb { \mu } _ { i }$ to the nearest micro-cluster center of higher density. We select the P peaks with the largest $\gamma _ { c }$ and denote the corresponding peak micro-clusters by ${ \mathcal { P } } \subseteq { \mathcal { C } }$ , as in Algorithm 2.

DPC applied to micro-clusters $\mathcal { C }$ rather than to individual samples reduces the computational cost from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( N C ) + \mathcal { O } ( C ^ { 2 } )$ under the assumption $N \gg C$ Standard DPC defines the local density $\rho _ { i }$ from pairwise distances using either a cutof $d _ { c }$ or a kNN-based kernel. The overlap graph $E _ { \kappa }$ provides a sparse, scale-controlled neighborhood for each micro-cluster, so the density can be estimated from local centerto-center distances only. We define the micro-cluster density as the accumulated mass within the $E _ { \kappa }$ neighborhood:

$$
\rho _ { i } = n _ { i } + \sum _ { { \{ i , j \} \in E _ { \kappa } \ n _ { j } \ \exp ( - d _ { i j } ^ { 2 } / d ) } , \nonumber }\tag{22}
$$

where $d _ { i j } ^ { 2 }$ is computed as in Eq. (14) and divided by the dimension d to combat the curse of dimensionality. The accumulation is distance-weighted and asymmetric $( n _ { j } > n _ { i } )$ , so that intermediate micro-clusters are lifted toward the dominant one in each region and obtain a small $\delta _ { i }$ , leaving a single peak per dense region even when cluster sizes are highly imbalanced. This addresses the main failure mode of assigning too many peaks to a single cluster under imbalanced data.

The non-peak micro-clusters are assigned by multi-source shortest-path propagation from the selected peaks over $E _ { \kappa }$ , rather than by nearest-higher assignment as in classical DPC. After selecting the $P$ peaks ${ \mathcal { P } } \subseteq { \mathcal { C } }$ , each micro-cluster i is assigned to the peak that minimizes the shortest-path cost:

$$
\ell ( i ) = \arg \operatorname* { m i n } _ { p : \mathcal { C } _ { p } \in \mathcal { P } } d _ { E _ { \kappa } } ^ { 2 } ( p , i ) ,\tag{23}
$$

$$
d _ { E _ { \kappa } } ^ { 2 } ( p , i ) = \operatorname* { m i n } _ { \pi \in \Pi ( p , i ) } \sum _ { \{ u , v \} \in \pi } d _ { u v } ^ { 2 } / d ,\tag{24}
$$

where $\Pi ( p , i )$ denotes the set of paths from node $p$ to i in the graph $( \{ 1 , \ldots , C \} , E _ { \kappa } )$ and the edge cost $d _ { u v } ^ { 2 } / d$ is computed as in Eq. (14). Disconnected components that contain no peak are assigned to the nearest already-assigned micro-cluster by Euclidean distance between centers. We denote the assigned groups by $\mathcal { G } _ { p } = \{ c \vert \ell ( c ) = p \}$ for each peak $\mathcal { C } _ { p } \in \mathcal { P }$ , and the grouping variant as eFDPC-G (Algorithm 3).

Algorithm 2 Peak selection on micro-clusters   
Input: Micro-clusters ${ \overline { { \boldsymbol { c } } } } ,$ overlap graph $E _ { \kappa } ,$ number of peaks $P$   
Output: Peak set ${ \mathcal { P } } \subseteq { \mathcal { C } }$   
1: Compute density $\rho _ { c } .$ ▷ Eq. (22)   
2: Compute distance to nearest higher $\delta _ { c } .$ ▷ Eq. (20)   
3: Compute peak score $\gamma _ { c } .$ ▷ Eq. (21)   
4: Select $\mathcal { P }$ as the $P$ micro-clusters with the largest values of $\gamma _ { c } .$   
Algorithm 3 Micro-cluster assignment (multi-source geodesic propagation)   
Input: Micro-clusters ${ \overline { { \boldsymbol { c } } } } ,$ overlap graph $E _ { \kappa } ,$ peaks ${ \overline { { \mathcal { P } \subseteq { \mathcal { C } } } } }$   
Output: Peak label $\ell ( c ) \in \{ p \mid \mathcal { C } _ { p } \in \mathcal { P } \}$ for all $c \in \{ 1 , \ldots , C \}$   
1: Initialize $\ell ( c ) \gets - 1$ and $d ( c ) \gets + \infty$ for all $c .$   
2: for each peak $\mathcal { C } _ { p } \in \mathcal { P }$ do   
3: $d ( p ) \gets 0$ and $\ell ( p )  p .$   
4: end for   
5: Set the edge cost on $E _ { \kappa }$ to $d _ { u v } ^ { 2 } / d .$ ▷ Eq. (14)   
6: $\ell ( \cdot ) $ multi-source Dijkstra labels from $\mathcal { P }$ on $E _ { \kappa }$ ▷ Eqs. (23)–(24)   
Fallback (assign disconnected components)   
7: for each c with $\ell ( c ) = - 1$ do   
8: $\ell ( c ) \gets \ell \big ( \arg \operatorname* { m i n } _ { j : \ell ( j ) \neq - 1 } \| \pmb { \mu } _ { c } - \pmb { \mu } _ { j } \| _ { 2 } ^ { 2 } \big )$   
9: end for

## 2.3 Federated Learning

In horizontal federated learning, the data are partitioned across $K \in \mathbb { Z } ^ { + }$ clients, where client k holds $\boldsymbol { X } ^ { ( k ) } \in \mathbb { R } ^ { N ^ { ( k ) } \times d }$ and $\begin{array} { r } { N = \sum _ { k = 1 } ^ { K ^ { - } } N ^ { ( k ) } } \end{array}$ . Each client learns micro-clusters $\mathcal { C } ^ { ( k ) }$ from its local data and then selects a subset of peaks $\mathcal { P } ^ { ( k ) }$ from $\mathcal { C } ^ { ( k ) }$ . We assume an honest-but-curious server, $i . e .$ , the server follows the algorithm exactly but may attempt to extract additional information from received messages [6, 45]. We also assume that the server needs to have access to the global model, so privacy protection focuses on preventing exposure of raw samples and ensuring unlinkability of client contributions. To avoid revealing which client produced which peaks, we use an additional shufler-based aggregation layer between the clients and the server [46]. Unlinkability assumes that the shuflers do not collude with the server. For each transmitted peak, the client creates new additive shares and a new random identifier that is used only to match those shares during reconstruction. Therefore, the server can reconstruct the peak statistics from the shufled shares, but does not know which client produced each reconstructed peak.

Each transmitted peak statistic consists of the center $\mu _ { c } ^ { ( k ) }$ and the min–max normalized support $\tilde { n } _ { c } ^ { ( k ) }$ , obtained by scaling the local peak support $\{ n _ { c } ^ { ( k ) } \} _ { c = 1 } ^ { C ^ { ( k ) } }$ to the unit interval [0, 1]. For each peak $c ,$ the client generates $S - 1$ random additive shares:

$$
( \mu _ { c } ^ { ( k , s ) } , \tilde { n } _ { c } ^ { ( k , s ) } ) \sim \mathcal { D } _ { \mu , \tilde { n } } , \qquad s = 1 , \dots , S - 1 ,\tag{25}
$$

and sets the last share so that the $S$ shares sum to the true values:

$$
\pmb { \mu } _ { c } ^ { ( k , S ) } = \pmb { \mu } _ { c } ^ { ( k ) } - \sum _ { s = 1 } ^ { S - 1 } \pmb { \mu } _ { c } ^ { ( k , s ) } , \qquad \tilde { n } _ { c } ^ { ( k , S ) } = \tilde { n } _ { c } ^ { ( k ) } - \sum _ { s = 1 } ^ { S - 1 } \tilde { n } _ { c } ^ { ( k , s ) } .\tag{26}
$$

To enable reconstruction at the server, a public identifier is used to match shares of the same peak within a round. For each peak $^ { c , }$ the client attaches a random perpeak identifier $t _ { c }$ and sends share-messages to each shufler $s \in { 1 , \dots , S }$ . We denote the message set sent from client k to shufler s and the unordered set forwarded by shufler s as

$$
\mathcal { M } ^ { ( k , s ) } = \{ ( t _ { c } , \mu _ { c } ^ { ( k , s ) } , \tilde { n } _ { c } ^ { ( k , s ) } ) \ | \ \mathcal { C } _ { c } ^ { ( k ) } \in \mathcal { P } ^ { ( k ) } \} ,
$$

$$
{ \mathcal { M } } ^ { ( s ) } = \bigcup _ { k = 1 } ^ { K } { \mathcal { M } } ^ { ( k , s ) } .\tag{27}
$$

(28)

The shufler forwards the received messages in a uniformly random order. The server reconstructs each peak by grouping messages with the same identifier t and summing the corresponding shares:

$$
{ \pmb \mu } _ { t } = \sum _ { s = 1 } ^ { S } { \pmb \mu } _ { t } ^ { ( s ) } , \qquad { \tilde { n } } _ { t } = \sum _ { s = 1 } ^ { S } { \tilde { n } } _ { t } ^ { ( s ) } .\tag{29}
$$

The shufling step breaks the association between reconstructed peaks and their client of origin at the server. Each shufler receives only one additive share per peak, which is insuficient to reconstruct the peaks.

The server still requires a diagonal covariance to evaluate the overlap score in Eqs. (13)–(15) and the edge costs used in DPC. The server therefore assigns each reconstructed peak a diagonal covariance proxy using Eq. (12). After reconstructing the peak centers and their normalized supports, the server estimates a per-dimension spread across the reconstructed peaks and scales it by $\kappa _ { r }$ to obtain a diagonal covariance for each peak. Each reconstructed peak is then treated as a diagonal-covariance micro-cluster and processed by the same overlap and DPC routines as at the clientside. eFDPC is presented in Algorithm 4. The algorithm is written for the general multi-round federated setting, and the while loop denotes standard iterative communication. All our experiments use a single round, which is suficient. eFDPC therefore operates one-shot in practice while still supporting multi-round refinement.

## 2.4 Computational cost

Table 1 summarizes the computational and memory cost of eFDPC. Since the microclusters use diagonal covariance matrices, the main operations scale linearly with the dimension $d ,$ and each micro-cluster requires $\mathcal O ( d )$ memory. For client $k ,$ the main learning cost is ${ \mathcal { O } } ( N ^ { ( k ) } C ^ { ( k ) } d )$ , while local peak selection costs ${ \mathcal { O } } ( ( C ^ { ( k ) } ) ^ { 2 } d )$ . The local model requires ${ \mathcal { O } } ( C ^ { ( k ) } d )$ memory. The shufler layer adds $\mathcal { O } ( S P d )$ operations for generating additive shares. Since eFDPC is one-shot, this is done only once. Each client sends P peak summaries as S additive shares in a single communication round. The reported runtimes include only local computation and exclude communication.

Algorithm 4 eFDPC   
1: Parameters: $\kappa _ { m } , \kappa _ { r }$   
2: Initialization: $\mathcal { P }  \emptyset$   
3: while federated learning do   
Local learning (client $k = 1 , \ldots , K )$ :   
4: for $k \in \{ 1 , 2 , \ldots , K \}$ do   
5: $\mathcal { C } ^ { ( k ) }$ ← Evolving micro-cluster learning on local data ▷ Algorithm 1   
6: P<sup>(k)</sup> ← Peak selection on $\mathcal { C } ^ { ( k ) }$ ▷ Algorithm 2   
7: end for   
Private communication:   
8: Form $\mathcal { M } ^ { ( k , s ) }$ from $\mathcal { P } ^ { ( k ) }$ and send to each shufler s. ▷ Eq. (27)   
9: Shuflers compute $\mathcal { M } ^ { ( s ) }$ and forward to server. ▷ Eq. (28)   
10: Reconstruct $\mathcal { C } _ { g }$ by summing shares with the same identifier t. ▷ Eq. (29)   
Server refinement:   
11: $\mathcal { C } _ { g } \gets$ Merging mechanism on $\mathcal { C } _ { g }$ ▷ Section 2.1.4   
12: $\mathcal { P } _ { g } $ Peak selection on $\mathcal { C } _ { g }$ ▷ Algorithm 2   
13: (Optional) $\mathcal { C } _ { g }  \mathrm { P 1 }$ ropagate labels from $\mathcal { P } _ { g }$ ▷ Algorithm 3   
Synchronization:   
14: Broadcast $\mathcal { P } _ { g }$ to all clients   
15: end while

Table 1 Computational and memory complexity of eFDPC.
<table><tr><td>Stage</td><td>Operation</td><td>Time</td><td>Memory</td></tr><tr><td rowspan="4">Client</td><td>Micro-cluster update (per batch)</td><td> $\mathcal { O } ( N _ { b } C d )$ </td><td rowspan="4"> $O ( ( C + N _ { b } ) d )$ </td></tr><tr><td>Micro-cluster update (full pass)</td><td>O(NCd)</td></tr><tr><td>Local DPC peak selection</td><td>O(C2d)</td></tr><tr><td>Share generation</td><td>O(SPd)</td></tr><tr><td>Communication</td><td>Per client, single round</td><td>O(Pd)</td><td>O(Pd)</td></tr><tr><td rowspan="2">Server</td><td>Peak reconstruction</td><td>O(SKPd)</td><td rowspan="2">O(KPd)</td></tr><tr><td>Global DPC &amp; label propagation</td><td>O((KP)2d)</td></tr></table>

## 3 Experimentation

The proposed eFDPC method was evaluated on a collection of synthetic and real-world clustering benchmarks. The synthetic datasets in Table 2 include Compound [47], D31 and R15 [48], S3 and S4 [49], Flame [50] Aggregation [51], Jain [52], Pathbased, and the Variations datasets Asymmetric, Skewed, Unbalance, and Overlap [53]. These datasets are commonly used to evaluate clustering performance under controlled geometric conditions in low-dimensional space. Real-world datasets in Table 3 include tabular benchmarks from UCI and related sources [54], namely Dermatology, Seeds, WDBC, Heart, Iris, Leaf, Libras, Texture, Thyroid, and Wine, as well as higherdimensional datasets including MNIST and KDD99. These datasets cover a wide range of dimensionalities and sample sizes, from small tabular problems to large-scale highdimensional settings. For baseline comparison, we used recent federated clustering approaches with publicly available implementations, namely FKDC [39], k-FED [10], MUFC [12], FedFCM [14], and LLF-CM and LLF-FCM [11].

The datasets were partitioned using the horizontal federated setting described in Section 2.3. Performance was measured using clustering accuracy (ACC), Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and total wall-clock runtime in seconds (local computation only, excluding communication). For the non-IID setting, the dataset was split across K clients using a class-wise Dirichlet partition [55, 56] with $\alpha = 0 . 9$ . In each run, samples within each class were randomly permuted and assigned to clients according to these proportions, with a nonzero number of samples enforced for each client. The same client partition was reused across all compared methods within that run. All experiments were repeated for 20 independent non-IID partitions, and the mean measures were averaged across them. In all experiments, the true number of clusters is used as the number of peaks $P$ for $\mathrm { e F D P C }$ and as the cluster count for all baselines, so that all methods are given the true number of clusters as input, which in practice is unknown. The $\mathrm { e F D P C }$ parameters were set using simple heuristic rules: $\kappa _ { \sigma } = d$ since squared Mahalanobis distances satisfy $d _ { M } ^ { 2 } \sim$ $\kappa _ { r } \chi ^ { 2 } ( d )$ , so the acceptance radius $\kappa _ { \sigma } ^ { 2 } / \kappa _ { r }$ matches the expected chi-squared distance $\mathbb { E } [ \chi _ { d } ^ { 2 } ] = d$ (with $\kappa _ { r } = d )$ ; the overlap-based merging threshold used the normalized chi-squared rule $\begin{array} { r } { \kappa _ { m } = 1 + \frac { 1 } { 4 } \frac { \chi _ { d } ^ { 2 - 1 } ( p ) } { d } } \end{array}$ with the chi-squared median $p = 0 . 5 ; N _ { b } = \sqrt { N }$ followed the common heuristic C ≈ $\sqrt { N }$ ; $K _ { \kappa } = \sqrt { C } ;$ and $\kappa _ { r } = \operatorname* { m a x } ( 4 P , d )$ based on the empirical observation that P dominates in low dimensions, whereas d dominates in high dimensions; all parameters were fixed across all datasets and were not tuned per dataset.

Figure 2 illustrates server-side peak selection and label propagation for Skewed $( K = 1 0$ clients) and MNIST (K = 3 clients). Each client transmits P local peaks, and the server selects P global peaks from the reconstructed pool. Increasing the client-count improves the aggregated connectivity among reconstructed candidates, making the propagation step more informative on Skewed. In contrast, MNIST is more uniform, so peak selection is less clear and less stable. However, label propagation stabilizes the result.

Tables 2–3 report clustering performance and runtime. We consider eFDPC-P, which clusters using the selected peaks only, and eFDPC-G, which additionally propagates peak labels to the remaining micro-clusters at the server. Best results are highlighted in bold. Overall, both eFDPC variants are consistently among the bestperforming methods across datasets, with particularly strong gains on datasets with overlap, non-spherical structure, or varying density. In contrast, faster baselines often trade accuracy for speed, while some methods degrade substantially when the number of clusters/classes is large or when cluster shapes are not axis-aligned. On the synthetic datasets (Table 2), eFDPC achieves the best performance on most benchmarks, especially on datasets with pronounced density variations or overlap. In particular, on Overlap, S3–S4, D31, and R15 density-peak selection results in notably higher ARI and NMI, because operating on incrementally learned micro-clusters reduces the chance of missing true density peaks. The selection then primarily depends on whether candidate peaks have high enough local density and suficient separation from higherdensity regions. On real-world datasets (Table 3), eFDPC achieves the best overall performance on several datasets, with the highest ACC and typically the highest ARI and NMI. On KDD99, which has large N and high d, k-means-based baselines (k-FED and FKDC) sometimes fail to recover meaningful cluster structure, whereas eFDPC maintains high ACC and achieves higher ARI and NMI. The fuzzy C-means-based FedFCM achieves very strong results, but with an order-of-magnitude higher runtime. On high-dimensional MNIST (d = 784), eFDPC-G substantially improves over most baselines in ACC and NMI, while k-FED achieves a comparable ARI.

![](images/25b7883804c5b3efe72f67d53e442e246fd0bb7ef55e272c525736a22a7e4b3b.jpg)  
(a) Skewed, decision graph

![](images/aa90cc7ca7fc84bbd1d7f8481e34ef53128dd9e56a8334f12af6acc250866bdb.jpg)  
(b) MNIST, decision graph

![](images/0be4e5a8cd7a88a4cd18b1f30f5691a2c971d415c86513aa214e2539e46d7d2a.jpg)  
(c) Skewed, grouping

![](images/9f14c62d2cae89664161265cc420f25224693673d04eba53ebffcc7499a16900.jpg)  
(d) MNIST, grouping

![](images/9ad7b3fdee9414701df48df72f78c8c3b933f348c3df47dd97f8546410a56ca8.jpg)  
(e) Skewed, accuracy

![](images/743e1e258bcccc3c3ca05cfc5d6af2997331b42711334e8d1fc0e3ab88aa020b.jpg)  
(f) MNIST, accuracy  
Fig. 2 Comparison of eFDPC-G behavior on Skewed $( K = 1 0 ,$ left column) and MNIST $( K = 3 ,$ right column). From top to bottom, rows depict the decision graph (a,b), cluster grouping (c,d), and clustering accuracy (e,f) at the server-side. The evaluation is performed using the propagated labels on the shared peaks.

Table 2 Clustering comparison on synthetic datasets $( K = 3 , \alpha = 0 . 9 )$
<table><tr><td rowspan=1 colspan=2>Dataset</td><td rowspan=1 colspan=2>|Metric</td><td rowspan=1 colspan=5>|eFDPC-PeFDPC-GFKDCk-FEDMUFC FedFCM LLF-FCM LLF-CM</td></tr><tr><td rowspan=2 colspan=2>Aggregation $N = 7 8 8$ </td><td rowspan=1 colspan=2> $\mathrm { A C C ~ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=5>92±02   92±02  85±0389±0388±0382±05  79±04   80±0586±04</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $\mathrm { A R I ~ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=1>86±0586</td><td></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=2> $d = 2$ </td><td rowspan=1 colspan=2> $\mathrm { N M I \ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=2>89±0290±0</td><td rowspan=2 colspan=3>90±02  81±0486±0485±0376±06  75±05   76±0598±07   98±07   94±1329±0488±0627±10  09±00   02±00</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>Time $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td></td><td></td></tr><tr><td rowspan=2 colspan=2>Asymmetric $N = 1 0 0 0$ d = 2</td><td rowspan=2 colspan=2> $\mathrm { A C C } \ [ 1 0 ^ { - 2 } ]$  $\mathrm { A R I } \ [ \mathrm { i } 0 ^ { - 2 } ]$  $\mathrm { N M I \ [ 1 0 ^ { - 2 } ] }$  $\mathrm { T i m e \ [ 1 0 ^ { - 2 } s ] }$ </td><td rowspan=1 colspan=5>|96±02   95±03   80±0987±1094±0589±07  77±09   76±1090±04  90±05   71±1181±1288±0681±09  66±12   66±1191±02   91±03  83±0487±0590±0386±05  77±07   77±07</td></tr><tr><td rowspan=1 colspan=5>10±00    $1 0 { \pm } 0 0 $    09±0102±0008±0000±00  00±00   00±00</td></tr><tr><td rowspan=2 colspan=2>Compound $N = 3 9 9$ d = 2</td><td rowspan=1 colspan=1> $\mathrm { A C CARI [1</td><td rowspan=1 colspan=1>} \ [ 1 0 ^ { - 2 } ]$ 0−2]</td><td rowspan=1 colspan=5>83±05   83±04   79±0380±0481±0375±02  76±04   78±0476±04   77±02  74±0275±0375±0370±03  68±08   70±08</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I</td><td rowspan=1 colspan=1>\ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=4>79±04   81±02  75±0278±04</td><td rowspan=1 colspan=1>76±0372±03  70±07   73±07</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=3>68±03   68±03   76±1427±03</td><td rowspan=1 colspan=1>65±0911±03  04±00   01±00</td></tr><tr><td rowspan=3 colspan=2>D31 $N = 3 1 0 0$  $d = 2$ </td><td rowspan=3 colspan=2> $\mathrm { A C C } \ [ 1 0 ^ { - 2 } ]$  $\mathrm { A R I ~ } [ 1 0 ^ { - 2 } ]$ NMI [10−2] $\mathrm { T i m e \ [ 1 0 ^ { - 1 } s ] }$ </td><td rowspan=1 colspan=3>97±00  97±00   63±03</td><td rowspan=1 colspan=1>82±03</td><td rowspan=1 colspan=1>93±0275±02  70±03   71±03</td></tr><tr><td rowspan=1 colspan=3>94±00  94±00   56±03</td><td rowspan=1 colspan=1>79±03</td><td rowspan=1 colspan=1>88±0268±02  62±03   65±03</td></tr><tr><td rowspan=1 colspan=5>96±00  96±00  81±0191±0102±00   02±00   02±0000±0005±0014±05  01±00   00±00</td><td rowspan=1 colspan=1>94±01 86±01 84±0185±01</td></tr><tr><td rowspan=4 colspan=2>Flame $N = 2 4 0$ d = 2</td><td rowspan=2 colspan=2> $\mathrm { A C C ~ [ 1 0 ^ { - 2 } ] }$  $\mathrm { A R I ~ } [ 1 0 ^ { - 2 } ]$ </td><td rowspan=2 colspan=3>|80±07   82±09  80±0739±17   45±26  37±18</td><td rowspan=1 colspan=1>81±07</td><td rowspan=1 colspan=1>78±0878±07  80±05   77±06</td></tr><tr><td rowspan=1 colspan=1>42±16</td><td rowspan=1 colspan=1>33±1833±17  37±12   32±16</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I</td><td rowspan=1 colspan=1>\ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=3>35±15   43±21  35±15</td><td rowspan=1 colspan=1>37±13</td><td rowspan=1 colspan=1>32±1531±16  36±11   33±12</td></tr><tr><td rowspan=1 colspan=1>Time [</td><td rowspan=1 colspan=1>10−3s]</td><td rowspan=1 colspan=3>48±04   48±04   54±09</td><td rowspan=1 colspan=1>28±02</td><td rowspan=1 colspan=1>45±0302±00  01±00   00±00</td></tr><tr><td rowspan=3 colspan=2>JainN = 373d = 2</td><td rowspan=2 colspan=1> $\mathrm { A C C ~ARI<eq</td><td rowspan=2 colspan=1>[ 1 0 ^ { - 2 } ] }$ >[ \bar { 1 } 0 ^ { - 2 } ]</eq></td><td rowspan=2 colspan=3>90±02  90±05   85±0864±08   64±16  49±25</td><td rowspan=1 colspan=1>88±03</td><td rowspan=1 colspan=1>88±0584±03  83±05   85±07</td></tr><tr><td rowspan=1 colspan=1>57±10</td><td rowspan=1 colspan=1>59±1847±08  45±16   50±24</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\mathrm { N M I</td><td rowspan=1 colspan=1>\ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=3>56±04   57±11  45±20</td><td rowspan=1 colspan=1>50±04</td><td rowspan=1 colspan=1>51±1447±05  44±13   46±18</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>|Time</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=3>61±03   61±03   66±11</td><td rowspan=1 colspan=1>27±03</td><td rowspan=1 colspan=1>51±0602±00  01±00   00±00</td></tr><tr><td rowspan=4 colspan=2>OverlapN = 1000d = 2</td><td rowspan=4 colspan=1>ACC [1 $\mathrm { A R I ~NMI [1Time</td><td rowspan=4 colspan=1>0−2]} [ 1 0 ^ { - 2 } ]$ 0−2] $[ 1 0 ^ { - 2 } \mathrm { s } ]$ </td><td rowspan=1 colspan=3>82±04   80±04   78±04</td><td rowspan=1 colspan=1>79±06</td><td rowspan=1 colspan=1>76±0573±05  49±06   22±07</td></tr><tr><td rowspan=1 colspan=3>68±03  64±04   66±02</td><td rowspan=1 colspan=1>65±05</td><td rowspan=1 colspan=1>59±0455±06  36±05   04±08</td></tr><tr><td rowspan=2 colspan=3>73±02   72±02   75±0010±00   10±00   09±01</td><td rowspan=1 colspan=1>72±03</td><td rowspan=1 colspan=1>68±0363±04  54±04   08±15</td></tr><tr><td rowspan=1 colspan=1>02±00</td><td rowspan=1 colspan=1>09±0003±01  00±00   00±00</td></tr><tr><td rowspan=4 colspan=2>R15N = 600d = 2</td><td rowspan=4 colspan=1> $\mathrm { A C CARI [1NMI [Time </td><td rowspan=4 colspan=1>} \ [ 1 0 ^ { - 2 } ]$ 0−2]10−21[102s]</td><td rowspan=3 colspan=3>99±00   99±00   59±0498±00  98±00   46±1099±00  98±00   78±03</td><td rowspan=1 colspan=1>88±05</td><td rowspan=1 colspan=1>97±0281±05  73±05   68±04</td></tr><tr><td rowspan=1 colspan=1>86±05</td><td rowspan=1 colspan=1>96±0376±06  65±05   60±06</td></tr><tr><td rowspan=1 colspan=1>94±02</td><td rowspan=1 colspan=1>97±0189±02  85±02   84±02</td></tr><tr><td rowspan=1 colspan=3>|09±00   09±00   09±01</td><td rowspan=1 colspan=1>03±00</td><td rowspan=1 colspan=1>10±0004±02  01±00   00±00</td></tr><tr><td rowspan=4 colspan=2>S3N = 5000d = 2</td><td rowspan=2 colspan=1> $\mathrm { A C C<eq>\mathrm { A R I ~</td><td rowspan=2 colspan=1>~ [ 1 0 ^ { - 2 } ] }$ } [ 1 0 ^ { - 2 } ]</eq></td><td rowspan=2 colspan=3>|84±01   84±01   59±0470±01   70±01   47±04</td><td rowspan=1 colspan=1>75±03</td><td rowspan=1 colspan=1>78±0272±04  63±04   60±04</td></tr><tr><td rowspan=1 colspan=1>63±03</td><td rowspan=1 colspan=1>64±0258±04  51±03   49±04</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I \</td><td rowspan=1 colspan=1>[ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=3>78±00   78±00  68±02</td><td rowspan=1 colspan=1>76±01</td><td rowspan=1 colspan=1>76±0173±02  69±01   69±02</td></tr><tr><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 2 } \mathrm { s } ]$ </td><td rowspan=1 colspan=3>34±02   34±02  21±01</td><td rowspan=1 colspan=1>05±00</td><td rowspan=1 colspan=1>43±0185±29  12±04   03±01</td></tr><tr><td rowspan=4 colspan=2>S4 $N = 5 0 0 0$  $d = 2$ </td><td rowspan=1 colspan=1>ACC [1</td><td rowspan=1 colspan=1>0−2]</td><td rowspan=1 colspan=3>78±01   77±01   56±04</td><td rowspan=1 colspan=1>71±02</td><td rowspan=1 colspan=1>72±0368±03  59±03   55±03</td></tr><tr><td rowspan=3 colspan=2> $\mathrm { A R I ~ [ 1 0 ^ { - 2 } ] }$  $\mathrm { N M I } \stackrel { \cdot } { \scriptstyle \left[ 1 0 ^ { - 2 } \right] }$ Time [10−2s]</td><td rowspan=1 colspan=1>10⁻21</td><td rowspan=1 colspan=3>62±01    $6 0 { \pm } 0 1$    42±03</td><td rowspan=1 colspan=1>56±01</td><td rowspan=1 colspan=1>56±0352±03  45±02   42±03</td></tr><tr><td rowspan=1 colspan=1>10⁻2]</td><td rowspan=1 colspan=3>71±00   71±00   61±01</td><td rowspan=1 colspan=1>69±00</td><td rowspan=1 colspan=1>69±0167±01  63±01   62±01</td></tr><tr><td rowspan=1 colspan=4>30±03   30±03   33±4706±00</td><td rowspan=1 colspan=1>45±0189±26  13±09   04±02</td></tr><tr><td rowspan=2 colspan=2>SkewedN = 1000</td><td rowspan=2 colspan=2> $\mathrm { A C C } \ [ 1 0 ^ { - 2 } ]$ ARI [10−21</td><td rowspan=2 colspan=4>|81±04   81±05  81±0673±0470±04    ${ \bf 7 1 \pm 0 7 }$   70±0762±05</td><td rowspan=1 colspan=1>73±0764±06  44±09   22±07</td></tr><tr><td rowspan=1 colspan=1>60±0848±06  33±10   05±09</td></tr><tr><td rowspan=2 colspan=2>d = 2</td><td rowspan=2 colspan=1> $\mathrm { N M I } \stackrel { \cdTime</td><td rowspan=2 colspan=1>ot } { \mathrm { [ 1 0 ^ { - 2 } ] } }$  $[ 1 0 ^ { - 2 } \mathrm { s } ]$ </td><td rowspan=1 colspan=4>77±02   79±04  78±0473±03</td><td rowspan=1 colspan=1>71±0561±05  51±08   10±16</td></tr><tr><td rowspan=1 colspan=4>10±00   10±00   10±0103±00</td><td rowspan=1 colspan=1>09±0002±00  01±00   00±00</td></tr><tr><td rowspan=3 colspan=2>UnbalanceN = 6500d = 2</td><td rowspan=1 colspan=2>ACC [10−2]ARI [10−2]</td><td rowspan=1 colspan=5>|99±00   99±00   99±0098±0198±0072±10  86±10   35±1199±00   99±00  99±0197±0198±0158±14  78±15   07±17</td></tr><tr><td rowspan=1 colspan=1>NMI [1</td><td rowspan=1 colspan=1>0−2]</td><td rowspan=1 colspan=5>99±00   99±00  98±0196±0297±0168±08  80±09   10±22</td></tr><tr><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 2 } \mathrm { s } ]$ </td><td rowspan=1 colspan=5>31±00   31±00   18±0104±0025±0100±00 08±00   02±00</td></tr></table>

Table 3 Clustering comparison on real-world datasets (K = 3, α = 0.9).
<table><tr><td rowspan=1 colspan=4>Dataset</td><td rowspan=1 colspan=3>|Metric</td><td rowspan=1 colspan=5>[eFDPC-P eFDPC-GFKDCk-FEDMUFCFedFCM LLF-FCM LLF-CM</td></tr><tr><td rowspan=4 colspan=4>Dermatology $N = 3 5 8$  $d = 3 4$ </td><td rowspan=1 colspan=3>|ACC $[ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=5>88±04    $^ \mathrm { 8 8 \pm 0 3 }$    68±0479±0880±0474±05   $7 1 { \pm } 0 6$     66±08</td></tr><tr><td rowspan=2 colspan=3>ARI [10−2]NMI $[ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2>87±03    $8 6 { \pm } 0 4$    57±07</td><td rowspan=1 colspan=3>75±1172±0767±07   $6 1 \pm 1 1$      $5 1 \pm 1 7$ </td></tr><tr><td rowspan=1 colspan=2>91±01    ${ 9 1 \pm 0 2 }$    73±04</td><td rowspan=1 colspan=3>85±0778±0476±05   $6 7 \pm 1 0$     65±10</td></tr><tr><td rowspan=1 colspan=3>Time $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2>48±03   48±03   76±08</td><td rowspan=1 colspan=3>29±0359±0719±07  04±01     $\mathbf { 0 1 } \pm \mathbf { 0 0 }$ </td></tr><tr><td rowspan=4 colspan=4>Heart $N = 2 7 0$  $d = 1 3$ </td><td rowspan=3 colspan=3>ACC $[ 1 0 ^ { - 2 } ]$  $\mathrm { A R I \ [ i 0 ^ { - 2 } ] }$ [NMI [10−2]</td><td rowspan=2 colspan=2> $7 8 \pm 0 5$     ${ \bf 7 9 } \pm { \bf 0 4 }$   76±05 ${ 3 2 \pm 1 0 }$     $\mathbf { 3 4 \pm 0 8 }$    29±09</td><td rowspan=1 colspan=3>75±0771±0976±05  73±07    70±07</td></tr><tr><td rowspan=1 colspan=3>27±1321±1428±09   $2 4 \pm 1 1$     18±11</td></tr><tr><td rowspan=1 colspan=2> $2 6 { \pm } 0 6$      ${ \bf 2 7 \pm 0 5 }$   22±07</td><td rowspan=1 colspan=3>21±0917±1022±07   $1 9 { \pm } 0 9 $     16±09</td></tr><tr><td rowspan=1 colspan=3>Time [10−3s]</td><td rowspan=1 colspan=2>42±02     $4 2 \pm 0 2$    60±08</td><td rowspan=1 colspan=3>28±0348±0505±02   $0 1 \pm 0 0$    00±00</td></tr><tr><td rowspan=3 colspan=4>Iris $N = 1 5 0$  $d = 4$ </td><td rowspan=2 colspan=3> $\mathrm { A C C } \ [ 1 0 ^ { - 2 } ]$ ARI [10−2]</td><td rowspan=2 colspan=2> $^ { 8 9 \pm 0 2 }$     ${ \bf 9 0 } { \bf \pm 0 4 }$   68±06 $7 4 { \pm } 0 6 $     ${ \bf 7 5 \pm 0 8 }$    45±14</td><td rowspan=1 colspan=1>80±10</td><td rowspan=1 colspan=1>84±07</td><td rowspan=1 colspan=1>87±02  79±07    75±09</td></tr><tr><td rowspan=1 colspan=1>65±08</td><td rowspan=1 colspan=1>67±09</td><td rowspan=1 colspan=1>70±05  59±09    57±11</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I</td><td rowspan=1 colspan=2>\ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=2> $7 6 { \pm } 0 3$     ${ \bf 7 8 \pm 0 5 }$    56±16</td><td rowspan=1 colspan=1>71±04</td><td rowspan=1 colspan=1>73±05</td><td rowspan=1 colspan=1> $7 2 \pm 0 4$    $6 5 \pm 0 8$      $6 5 { \pm } 1 0 $ </td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2> $3 5 { \pm } 0 2$     $3 5 { \pm } 0 2$    39±04</td><td rowspan=1 colspan=1>30±02</td><td rowspan=1 colspan=1>48±04</td><td rowspan=1 colspan=1>02±00  01±00     $\mathbf { 0 0 } { \pm } \mathbf { 0 0 }$ </td></tr><tr><td rowspan=2 colspan=4>KDD99 $N = 4 9 4 0 2 1$ </td><td rowspan=1 colspan=1>ACC</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2> $9 8 { \pm } 0 0 \ $     $9 8 \pm 0 0$    98±00</td><td rowspan=1 colspan=1>98±00</td><td rowspan=1 colspan=1> $\mathbf { 9 8 \pm 0 0 }$ </td><td rowspan=1 colspan=1>98±00  92±06    91±07</td></tr><tr><td rowspan=1 colspan=1>ARI</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2> ${ 9 3 2 0 1 }$     $^ \mathrm { 9 3 \pm 0 1 }$    94±00</td><td rowspan=1 colspan=1>93±02</td><td rowspan=1 colspan=1> $\mathbf { 9 4 } \pm \mathbf { 0 1 }$ </td><td rowspan=1 colspan=1>92±02  62±34    59±36</td></tr><tr><td rowspan=2 colspan=4> $d = 1 2 1$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NMI</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=3> $^ \mathrm { 8 7 \pm 0 3 }$     $^ \mathrm { 8 7 \pm 0 3 }$    88±01</td><td rowspan=1 colspan=1>85±04</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { T i m e</td><td rowspan=1 colspan=2>\ [ 1 0 ^ { 0 } s ] }$ </td><td rowspan=1 colspan=2> $_ { 0 4 \pm 0 0 }$     $\boldsymbol { 0 } 4 \pm \boldsymbol { 0 } \boldsymbol { 0 }$    14±00</td><td rowspan=1 colspan=1>02±00</td><td rowspan=1 colspan=1> $2 6 { \pm } 0 0$ </td><td rowspan=1 colspan=1>56±16  09±00   02±00</td></tr><tr><td rowspan=4 colspan=4>Leaf $N = 3 4 0$  $d = 1 5$ </td><td rowspan=1 colspan=1> $\mathrm { A C C</td><td rowspan=1 colspan=2>} \ [ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2>48±02    $4 7 { \pm } 0 2$    36±02</td><td rowspan=1 colspan=1>43±01</td><td rowspan=1 colspan=1>46±02</td><td rowspan=1 colspan=1>40±02  28±01    23±03</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { A R I ~</td><td rowspan=1 colspan=1>[ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>30±02    $3 0 { \pm } 0 2$    20±02</td><td rowspan=1 colspan=1>27±01</td><td rowspan=1 colspan=1>28±01</td><td rowspan=1 colspan=1>25±02  16±01    10±03</td></tr><tr><td rowspan=1 colspan=1>[NMI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2> $6 4 { \pm } 0 1$     ${ \bf 6 4 } \pm { \bf 0 1 }$    56±02</td><td rowspan=1 colspan=1>62±01</td><td rowspan=1 colspan=1>63±01</td><td rowspan=1 colspan=1>58±01  51±02    46±04</td></tr><tr><td rowspan=1 colspan=1>Time [</td><td rowspan=1 colspan=2>10−</td><td rowspan=1 colspan=2>07±00    $0 7 { \pm } 0 0$    09±00</td><td rowspan=1 colspan=1>03±00</td><td rowspan=1 colspan=1>12±00</td><td rowspan=1 colspan=1>28±07   $0 1 \pm 0 0$    00±00</td></tr><tr><td rowspan=4 colspan=4>Libras $N = 3 6 0$  $d = 9 0$ </td><td rowspan=2 colspan=1> $\mathrm { A C CARI [1</td><td rowspan=2 colspan=2>~ [ 1 0 ^ { - 2 } ] }$ 0−2]</td><td rowspan=2 colspan=2>47±01    ${ \bf 4 7 \pm 0 1 }$    38±03 $_ { 3 1 \pm 0 2 }$     ${ \bf 3 3 \pm 0 2 }$ </td><td rowspan=1 colspan=1>43±03</td><td rowspan=1 colspan=1>45±01</td><td rowspan=1 colspan=1>29±02</td></tr><tr><td rowspan=1 colspan=1>20±03</td><td rowspan=1 colspan=1>29±03</td><td rowspan=1 colspan=1>28±02</td><td rowspan=1 colspan=1> $1 5 { \pm } 0 4$     18±03</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I</td><td rowspan=1 colspan=2>\ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=1> $5 7 \pm 0 2$     ${ \bf 5 9 \pm 0 2 }$ </td><td rowspan=1 colspan=1>46±03</td><td rowspan=1 colspan=1>55±03</td><td rowspan=1 colspan=1>55±02</td><td rowspan=1 colspan=1>38±04   $3 8 { \pm } 0 4$     42±04</td></tr><tr><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=2>[10−²s]</td><td rowspan=1 colspan=2>05±00     $0 5 { \pm } 0 0$    09±01</td><td rowspan=1 colspan=1>03±00</td><td rowspan=1 colspan=1>09±00</td><td rowspan=1 colspan=1>14±04   $0 0 { \pm } 0 0$     00±00</td></tr><tr><td rowspan=4 colspan=4>MNIST $N = 6 0 0 0 0$  $d = 7 8 4$ </td><td rowspan=1 colspan=1>|ACC [</td><td rowspan=1 colspan=2>10−2]</td><td rowspan=1 colspan=2> $5 7 \pm 0 2$     ${ \bf 5 9 } \pm { \bf 0 1 }$    47±05</td><td rowspan=1 colspan=1>55±02</td><td rowspan=1 colspan=1>43±04</td><td rowspan=1 colspan=1>30±03</td></tr><tr><td rowspan=1 colspan=1>ARI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2> ${ 3 8 \pm 0 3 }$     ${ \bf 4 2 \pm 0 2 }$    26±04</td><td rowspan=1 colspan=1>38±02</td><td rowspan=1 colspan=1>20±04</td><td rowspan=1 colspan=1>19±03    14±04</td></tr><tr><td rowspan=1 colspan=1>[NMI [</td><td rowspan=1 colspan=2>10−2]</td><td rowspan=1 colspan=2> $4 8 { \pm } 0 1$      ${ \bf 5 3 \pm 0 2 }$    38±04</td><td rowspan=1 colspan=1>49±02</td><td rowspan=1 colspan=1>32±03</td><td rowspan=1 colspan=1>28±02    26±04</td></tr><tr><td rowspan=1 colspan=1>Time [</td><td rowspan=1 colspan=2>10−1s]</td><td rowspan=1 colspan=2>16±00     $1 6 \pm 0 0$    42±02</td><td rowspan=1 colspan=1>08±00</td><td rowspan=1 colspan=1>58±01</td><td rowspan=1 colspan=1>45±02  15±01    05±01</td></tr><tr><td rowspan=4 colspan=4>Seeds $N = 2 1 0$  $d = 7$ </td><td rowspan=1 colspan=1> $\mathrm { A C C</td><td rowspan=1 colspan=2>~ } [ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2>89±02    $^ \mathrm { 8 7 \pm 0 3 }$    80±10</td><td rowspan=1 colspan=1>87±06</td><td rowspan=1 colspan=1>83±05</td><td rowspan=1 colspan=1>87±03  76±06    69±08</td></tr><tr><td rowspan=1 colspan=1>ARI [1</td><td rowspan=1 colspan=1>0−2]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>71±04    $6 8 { \pm } 0 6 $    60±12</td><td rowspan=1 colspan=1>68±08</td><td rowspan=1 colspan=1>60±08</td><td rowspan=1 colspan=1>67±06  53±07    48±11</td></tr><tr><td rowspan=1 colspan=1>NMI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2> ${ \bf 6 9 \pm 0 2 }$     $\mathrm { 6 7 \pm 0 3 }$    62±07</td><td rowspan=1 colspan=1>67±04</td><td rowspan=1 colspan=1>63±05</td><td rowspan=1 colspan=1>65±04  58±05    54±07</td></tr><tr><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2>45±44    $4 5 { \pm } 4 4$    52±05</td><td rowspan=1 colspan=1>29±02</td><td rowspan=1 colspan=1>49±07</td><td rowspan=1 colspan=1>02±00  01±00    00±00</td></tr><tr><td rowspan=3 colspan=4> $_ \mathrm { T e x t u r e }$  $N = 5 5 0 0$  $d = 4 0$ </td><td rowspan=2 colspan=1> $\mathrm { A C C<eq>\mathrm { A R I</td><td rowspan=2 colspan=2>~ } [ 1 0 ^ { - 2 } ]$ ~ [ 1 0 ^ { - 2 } ] }</eq></td><td rowspan=2 colspan=2>63±03    $6 3 { \pm } 0 2$    54±04 $4 9 { \pm } 0 2$     ${ \bf 4 9 \pm 0 2 }$    39±05</td><td rowspan=1 colspan=1>58±03</td><td rowspan=1 colspan=1>62±03</td><td rowspan=1 colspan=1>53±03</td></tr><tr><td rowspan=1 colspan=1>45±02</td><td rowspan=1 colspan=1>47±03</td><td rowspan=1 colspan=1> $1 8 { \pm } 0 1$     24±06</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { N M I \</td><td rowspan=1 colspan=2>[ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=2> $6 4 { \pm } 0 1$      ${ \bf 6 5 \pm 0 1 }$    58±03</td><td rowspan=1 colspan=1>62±01</td><td rowspan=1 colspan=1>63±02</td><td rowspan=1 colspan=1>54±02  38±02    46±05</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=1> $[ 1 0 ^ { - 2 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2>13±01     ${ 1 3 } { \pm } 0 1$    21±01</td><td rowspan=1 colspan=1>06±00</td><td rowspan=1 colspan=1>41±01</td><td rowspan=1 colspan=1>72±23  10±04    02±00</td></tr><tr><td rowspan=4 colspan=4>Thyroid $N = 2 1 5$  $d = 5$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ACC [1</td><td rowspan=1 colspan=1>0−2]</td><td rowspan=1 colspan=2>87±05    $8 6 { \pm } 0 6 $    77±03</td><td rowspan=1 colspan=1>84±04</td><td rowspan=1 colspan=1>85±05</td><td rowspan=1 colspan=1>82±06  73±05    79±05</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { A R I</td><td rowspan=1 colspan=2>~ [ 1 0 ^ { - 2 } ] }$ </td><td rowspan=1 colspan=2>59±16    $5 4 \pm 1 9$    29±11</td><td rowspan=1 colspan=1>50±13</td><td rowspan=1 colspan=1>54±16</td><td rowspan=1 colspan=1>46±19  20±19    34±17</td></tr><tr><td rowspan=1 colspan=1>[NMI [</td><td rowspan=1 colspan=2>10−2]</td><td rowspan=1 colspan=2> ${ \bf 5 7 \pm 1 3 }$     ${ 5 3 \pm 1 6 }$    30±10</td><td rowspan=1 colspan=1>49±11</td><td rowspan=1 colspan=1>51±14</td><td rowspan=1 colspan=1>46±17  23±20    36±16</td></tr><tr><td rowspan=1 colspan=1>Time [</td><td rowspan=1 colspan=2>10−3s]</td><td rowspan=1 colspan=2>47±05     $4 7 { \pm } 0 5$    50±08</td><td rowspan=1 colspan=1>28±02</td><td rowspan=1 colspan=1>48±04</td><td rowspan=1 colspan=1>04±02  01±00   00±00</td></tr><tr><td rowspan=2 colspan=4> $\mathrm { W D B C }$ N = 569</td><td rowspan=1 colspan=1> $\mathrm { A C C</td><td rowspan=1 colspan=2>} \ [ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2> ${ 9 2 \pm 0 2 }$     ${ 9 1 \pm 0 2 }$    89±08</td><td rowspan=1 colspan=1>91±01</td><td rowspan=1 colspan=1>86±09</td><td rowspan=1 colspan=1> ${ \bf 9 2 \pm 0 0 }$    ${ } 8 7 \pm 0 7  $     75±11</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ARI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2> $7 1 \pm 0 7$     $6 9 \pm 0 8$    63±23</td><td rowspan=1 colspan=1>69±06</td><td rowspan=1 colspan=1>55±23</td><td rowspan=1 colspan=1> ${ \bf 7 2 \pm 0 2 }$    $5 8 \pm 2 0$      $2 7 \pm 2 7$ </td></tr><tr><td rowspan=1 colspan=1> $d = 3 0$ </td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\mathrm { N M I } \ \mathr</td><td rowspan=1 colspan=2>m { \bar { [ 1 0 ^ { - 2 } ] } }$ </td><td rowspan=1 colspan=2> ${ \bf 6 1 \pm 0 5 }$     $6 0 { \pm } 0 6$     $5 5 { \pm } 1 9 $ </td><td rowspan=1 colspan=1>60±04</td><td rowspan=1 colspan=1>47±20</td><td rowspan=1 colspan=1> $^ \mathrm { 6 0 \pm 0 2 }$    $4 8 \pm 1 8$      ${ 2 4 \pm 2 3 }$ </td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2>58±02    $5 8 { \pm } 0 2$    77±13</td><td rowspan=1 colspan=1>28±03</td><td rowspan=1 colspan=1>58±08</td><td rowspan=1 colspan=1>05±06   $0 2 \pm 0 0$      $\mathbf { 0 1 } \pm \mathbf { 0 0 }$ </td></tr><tr><td rowspan=4 colspan=4>Wine $N = 1 7 8$  $d = 1 3$ </td><td rowspan=1 colspan=1> $\mathrm { A C C</td><td rowspan=1 colspan=2>} \ [ 1 0 ^ { - 2 } ]$ </td><td rowspan=1 colspan=2>94±01    $9 2 \pm 0 4$    78±14</td><td rowspan=1 colspan=1>90±08</td><td rowspan=1 colspan=1>83±09</td><td rowspan=1 colspan=1>85±06  74±09    66±13</td></tr><tr><td rowspan=1 colspan=1>ARI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2>83±04    $7 9 { \pm } 1 0 $     $5 6 { \pm } 2 0 $ </td><td rowspan=1 colspan=1>77±12</td><td rowspan=1 colspan=1>62±14</td><td rowspan=1 colspan=1>62±12  47±13    36±18</td></tr><tr><td rowspan=1 colspan=1>NMI [1</td><td rowspan=1 colspan=2>0−2]</td><td rowspan=1 colspan=2>81±03    $^ { 7 9 \pm 0 7 }$     ${ 6 1 \pm 1 5 }$ </td><td rowspan=1 colspan=1>77±08</td><td rowspan=1 colspan=1>66±09</td><td rowspan=1 colspan=1>66±09  53±12    42±17</td></tr><tr><td rowspan=1 colspan=1>Time</td><td rowspan=1 colspan=2> $[ 1 0 ^ { - 3 } \mathrm { s } ]$ </td><td rowspan=1 colspan=2>36±01     $_ { 3 6 \pm 0 1 }$    45±05</td><td rowspan=1 colspan=1>31±02</td><td rowspan=1 colspan=1>46±02</td><td rowspan=1 colspan=1>07±03  01±00    00±00</td></tr></table>

In terms of runtime, eFDPC is slower than the fastest centroid-based baselines (e.g., k-FED and LLF-CM/LLF-FCM), but it remains substantially faster than heavier approaches such as FedFCM on large-scale datasets, while providing stronger ARI and NMI. This reflects the added cost of forming and merging micro-clusters and performing peak selection, which improves robustness at a moderate computational overhead. However, the reported runtimes measure local computation only and do not include the communication cost for any method, which would add overhead in a real deployment.

## 3.1 Federated learning sensitivity

We evaluate the sensitivity of eFDPC under changing federated conditions and learning settings. We vary the non-IID level of client data distributions with a Dirichlet α sweep, we vary the amount of data per client with a sweep over the number of clients, and we vary the mini-batch size used by the client-side evolving learning. All other settings are the same as in the benchmark-dataset experiments.

First, we performed a client-count sweep to test how clustering changes when the data are distributed among more clients. As K increases, each client observes fewer samples and fewer classes, so local models become less representative, and serverside aggregation must compute peaks from more partial and more heterogeneous client contributions. We keep the experiment setup identical to the benchmark-dataset experiments and vary only the number of clients, $K \in \{ 5 , 1 0 , 3 0 \}$

The results of the client-count sweep are presented in Table 4. The metrics stay stable or improve as K increases. The end-to-end runtime drops for most methods when the data are split across more clients because each client processes a smaller local partition. When the data are partitioned among more clients, each client processes fewer samples and maintains fewer micro-clusters than in a centralized setting. Local distance computations scale as ${ \mathcal { O } } ( C ^ { ( k ) } N ^ { ( k ) } )$ and pairwise cluster operations scale as ${ \mathcal { O } } { \big ( } ( C ^ { ( k ) } ) ^ { 2 } N ^ { ( { \bar { k } } ) } { \big ) }$ . Under an approximately balanced split, $N ^ { ( k ) } \approx N / K$ and $C ^ { ( k ) }$ ≈ $C / K$ , so the per-client cost decreases to $\mathcal { O } ( C N / K ^ { 2 } )$ and $\mathcal { O } ( C ^ { 2 } / K ^ { 2 } )$ , and the total work across clients decreases roughly as $\mathcal { O } ( C N / K )$ and ${ \mathcal { O } } ( C ^ { 2 } / K )$ . The increase in metrics for larger K is expected because every compared method sends a fixed number of clusters per client, so a larger K means that the server receives more clusters in total. This makes aggregation easier and can improve ACC, ARI, and NMI across the board.

Second, we performed a mini-batch size sweep to test how the batch size B afects model quality. The batch size controls how many samples are processed in one update.

Table 4 Client-count K sweep for Dirichlet $\alpha = 0 . 9 .$
<table><tr><td>Dataset</td><td colspan="2">|Metric</td><td>eFDPC-P</td><td>eFDPC-G</td><td>FKDC</td><td>k-FED</td><td>MUFC</td><td></td><td>FedFCM LLF-FCM LLF-CM</td><td></td></tr><tr><td rowspan="4">MNIST  $K = 5$ </td><td>ACC [10−2]</td><td></td><td> $5 2 \pm 0 2$ </td><td> ${ \bf 5 9 2 0 4 }$ </td><td>55±02</td><td>55±03</td><td>45±03</td><td>31±04</td><td>38±03</td><td>35±04</td></tr><tr><td> $\mathrm { A R I ~ } [ 1 0 ^ { - 2 } ]$ </td><td></td><td> $^ { 2 8 \pm 0 3 }$ </td><td> ${ \bf 3 9 \pm 0 5 }$ </td><td>36±02</td><td>38±02</td><td>24±04</td><td>13±04</td><td>19±03</td><td>14±04</td></tr><tr><td> $\mathrm { N M I } \ \mathrm { \bar { [ 1 0 ^ { - 2 } ] } }$ </td><td></td><td> $4 2 { \pm } 0 1$ </td><td> ${ \bf 5 5 \pm 0 3 }$ </td><td>46±02</td><td>50±02</td><td>35±03</td><td>23±05</td><td>28±02</td><td>26±04</td></tr><tr><td>Time</td><td> $[ 1 0 ^ { - 1 } \mathrm { s } ]$ </td><td> $1 6 { \pm } 0 0$ </td><td>16±00</td><td>43±01</td><td>07±00</td><td>55±00</td><td>50±27</td><td>15±00</td><td>05±01</td></tr><tr><td rowspan="4">MNIST  $K = 1 0$ </td><td>ACC</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> ${ 5 1 \pm 0 3 }$ </td><td>60±04</td><td>54±02</td><td>55±05</td><td>47±02</td><td>33±03</td><td>36±03</td><td>33±06</td></tr><tr><td>ARI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $^ { 2 9 \pm 0 3 }$ </td><td>43±06</td><td>32±02</td><td>39±04</td><td>27±03</td><td>13±03</td><td>17±02</td><td>11±04</td></tr><tr><td>NMI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $4 2 { \pm } 0 2$ </td><td>58±03</td><td>42±02</td><td>49±04</td><td>38±02</td><td>24±04</td><td>26±02</td><td> $2 4 \pm 0 6$ </td></tr><tr><td>Time</td><td> $[ 1 0 ^ { 0 } \mathrm { s } ]$ </td><td> $1 0 { \pm } 0 8 $ </td><td>10±08</td><td>04±01</td><td>00±00</td><td>06±00</td><td>04±00</td><td>01±00</td><td> ${ \bf 0 0 } { \pm } { \bf 0 0 }$ </td></tr><tr><td rowspan="4">MNIST  $K = 3 0$ </td><td>ACC [10</td><td>-2]</td><td>55±04</td><td>60±04</td><td>48±02</td><td>57±03</td><td>50±02</td><td>34±03</td><td>38±03</td><td>35±04</td></tr><tr><td>ARI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td>33±04</td><td>41±05</td><td>26±02</td><td>40±02</td><td>30±02</td><td>15±02</td><td>18±03</td><td>14±03</td></tr><tr><td>NMI [10−2]</td><td></td><td> $\mathrm { 4 5 { \pm } 0 3 }$ </td><td>55±03</td><td>36±02</td><td>51±01</td><td>41±02</td><td>26±02</td><td>27±02</td><td>26±03</td></tr><tr><td> $\mathrm { T i m e } \mathbf { \bar { \Gamma } } [ 1 0 ^ { 0 } \mathbf { s } ]$ </td><td></td><td> $0 1 { \pm } 0 0$ </td><td> $0 1 \pm 0 0$ </td><td>08±00</td><td>00±00</td><td>05±00</td><td>04±00</td><td> $_ { 0 1 \pm 0 0 }$ </td><td> ${ \bf 0 0 } { \pm } { \bf 0 0 }$ </td></tr></table>

Small batches update the model more frequently, but they are slower because dataloading overheads occur more often. This also generally produces fewer clusters. In contrast, large batches reduce the number of updates, but they can produce too many clusters during the early mini-batches. We tested B ∈ {32, 64, 128, 256, 512, 1024} and the number of clients $K \in \{ 3 , 1 0 , 3 0 \}$ at $\alpha \ : = \ : 0 . 9$ for both the peak-only variant eFDPC-P and the grouped variant eFDPC-G.

The results of the sweep experiment are shown in Fig. 3. Clustering quality is stable across a broad range of batch sizes, with an optimum around $B = 6 4$ to 128, and a small degradation at the largest batch (B = 1024), i.e., the default $B \approx \sqrt { N }$ therefore lies within this stable region. Local runtime falls as B grows, because fewer minibatches are processed, so larger batches reduce computation at little cost in accuracy. Increasing the number of clients improves accuracy, with eFDPC-G rising from about 0.59 at K = 3 to 0.66 at $K = 3 0$ , consistent with the client-count sweep in the previous experiment. In contrast to the per-client cost, the eFDPC runtime grows modestly with K, because the server-side reconstruction and global density peak clustering operate on the larger pooled set of KP peaks at cost $\mathcal { O } ( ( K P ) ^ { 2 } d )$ (Table 1). Across all settings eFDPC-G outperforms eFDPC-P, confirming the benefit of the server-side label propagation.

Third, we performed a non-IID Dirichlet α sweep to test how clustering changes as the client data split becomes more or less non-IID. Smaller α gives a more skewed split, where each client has samples from fewer classes, while larger α produces a more balanced split across clients. We keep the experiment setup identical to the benchmark-dataset experiments and vary only the Dirichlet concentration parameter, using $\alpha \in \{ 0 . 1 , 0 . 3 , 1 , 3 . 0 \}$ with K = 10 clients.

The results of the α sweep experiment are shown in Table 5. As α increases, eFDPC-G improves consistently in ACC, ARI, and NMI, which matches the expectation that a less skewed client split makes server-side peak reconstruction and grouping easier. At lower α, clients agree less on which micro-clusters form global peaks, so more non-peak micro-clusters are shared and the peaks become less pronounced, which can lead to less stable selection. Across all tested α values, eFDPC-G achieves the strongest clustering quality among the compared methods. Runtime is broadly comparable across α for all methods. A likely reason for higher runtime at larger α is that micro-clusters from diferent clients become closer, so the server keeps more neighbor edges and evaluates more candidate pairs during merging and grouping even when the number of micro-clusters C is similar.

![](images/090535ec157d38710927d8466c352648531c7595dd7c3155c46486af82ff67a2.jpg)  
(a) Accuracy (ACC)

![](images/73dcdb822fe3148a42b5159aa18d2ef24ade87c9649c12d6ac8727e89fb3f3ab.jpg)  
(b) Adjusted Rand Index (ARI)

![](images/27966ed3445b9401fa0241b7ff7ada1b77403ca9bf7495c92254e7e83c25414f.jpg)  
(c) Normalized Mutual Information (NMI)

![](images/b54d1f9bec77462ce690617591a97ca71902a46d2c39e5feaff3f7d2d4dde553.jpg)  
(d) Wall-clock runtime  
Fig. 3 Mini-batch size sensitivity of eFDPC-G on MNIST for K = 3, 10, 30 clients (Dirichlet α = 0.9). ACC, ARI, and NMI are stable across the mini-batch size B, with a mild optimum at moderate sizes, and improve with the number of clients; runtime falls as B grows and grows with K. Bands show ±1 standard deviation over the repetitions.

![](images/83a3b923c223c2cd89ea66af9f0c4dbedf4f2f275fd7c92c9c021f814492886f.jpg)  
Client k  
(a) α = 0.1

![](images/8fad4b6216565eb6973d94b9fb99389556add57e6327204dce338cd3399df483.jpg)  
Client k  
(b) α = 0.3

![](images/92ba59c64e52405325e44b0663e46599a051dadbac1b8b50a5952007f2d62dd4.jpg)  
Client k  
(c) α = 1

![](images/10666b479f6c9979d9547e7f1540539e8b57842b84d9fc9172f8792250f54edc.jpg)  
Client k  
(d) α = 3.0  
Fig. 4 Non-IID label partition of MNIST across K = 10 clients with the Dirichlet split used in the experiments. Small α results in each true cluster being split between a few clients (a), while large α approaches an IID split where every client holds every cluster in similar proportions (d).

## 4 Discussion

The experimental results indicate that eFDPC-P is consistently among the bestperforming methods across datasets, with particularly strong gains on datasets with overlap and varying density, while eFDPC-G performs especially well on non-spherical structure. In this section, we discuss why the proposed micro-cluster representation and peak-selection strategy are efective under federated constraints, and we highlight limitations and potential extensions.

Table 5 Non-IID Dirichlet α sweep for $K = 1 0 .$
<table><tr><td>Dataset |Metric</td><td></td><td></td><td>[eFDPC-P eFDPC-G</td><td></td><td>FKDC</td><td>k-FED</td><td>MUFC</td><td></td><td>FedFCM LLF-FCM LLF-CM</td><td></td></tr><tr><td rowspan="4">MNIST  $\alpha = 0 . 1$ </td><td>ACC [10</td><td>-21</td><td> $4 8 { \pm } 0 5$ </td><td> ${ \bf 5 4 } \pm { \bf 0 6 }$ </td><td>53±04</td><td>48±06</td><td>48±03</td><td> ${ 3 0 \pm 0 6 }$ </td><td> ${ 3 8 \pm 0 3 }$ </td><td>35±04</td></tr><tr><td>ARI [i0</td><td>-21</td><td> $2 9 { \pm } 0 4$ </td><td> ${ \bf 3 7 \pm 0 7 }$ </td><td>33±04</td><td> $3 1 \pm 0 6$ </td><td>26±04</td><td> $1 1 \pm 0 5$ </td><td> ${ 1 9 2 0 3 }$ </td><td>14±04</td></tr><tr><td>NMI</td><td> $\dot { \lceil 1 0 ^ { - 2 } \rceil }$ </td><td> $3 8 { \pm } 0 4$ </td><td> ${ \bf 5 2 \pm 0 5 }$ </td><td>43±04</td><td>44±05</td><td>38±03</td><td> $2 0 { \pm } 0 6$ </td><td> $2 8 \pm 0 2$ </td><td>26±04</td></tr><tr><td>Time</td><td> $[ 1 0 ^ { 0 } \mathrm { s } ]$ </td><td> $\mathrm { 0 3 \pm 0 3 }$ </td><td> $_ { 0 3 \pm 0 3 }$ </td><td>04±00</td><td>00±00</td><td>06±00</td><td>12±14</td><td> $0 1 \pm 0 0$ </td><td>00±00</td></tr><tr><td rowspan="4"> $\mathrm { M N I S T }$   $\alpha = 0 . 3$ </td><td>ACC</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td>49±04</td><td>57±05</td><td>54±03</td><td>52±04</td><td>48±02</td><td> ${ 3 3 \pm 0 3 }$ </td><td> ${ 3 8 \pm 0 3 }$ </td><td>35±04</td></tr><tr><td>ARI</td><td> $\bar { [ 1 0 ^ { - 2 } ] }$ </td><td> $2 8 { \pm } 0 4$ </td><td>41±06</td><td>33±03</td><td>35±03</td><td>27±02</td><td> ${ 1 4 2 0 3 }$ </td><td> ${ 1 9 2 0 3 }$ </td><td>14±04</td></tr><tr><td>NMI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $^ { 3 9 \pm 0 3 }$ </td><td> ${ \bf 5 6 \pm 0 4 }$ </td><td>43±03</td><td> $4 7 { \pm } 0 3$ </td><td>38±02</td><td> $2 4 { \pm } 0 4$ </td><td> $2 8 \pm 0 2$ </td><td>26±04</td></tr><tr><td>Time</td><td> $[ 1 0 ^ { 0 } \mathrm { s } ]$ </td><td> $0 2 \pm 0 0$ </td><td> $0 2 \pm 0 0$ </td><td>05±00</td><td> $_ { 0 0 \pm 0 0 }$ </td><td>06±00</td><td> $\mathrm { 0 5 \pm 0 4 }$ </td><td> $0 1 \pm 0 0$ </td><td> $\mathbf { 0 0 } { \pm } \mathbf { 0 0 }$ </td></tr><tr><td rowspan="4">MNIST  $\alpha = 1 . 0$ </td><td>ACC [10−2]]</td><td></td><td> $5 0 { \pm } 0 4$ </td><td> ${ \bf 6 1 \pm 0 5 }$ </td><td>55±03</td><td>57±03</td><td>46±03</td><td> ${ 3 2 \pm 0 3 }$ </td><td> ${ 3 8 \pm 0 3 }$ </td><td>35±04</td></tr><tr><td>ARI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $2 8 \pm 0 5$ </td><td> $\mathbf { 4 4 } \pm \mathbf { 0 6 }$ </td><td>34±03</td><td>39±02</td><td>27±03</td><td>13±03</td><td> ${ 1 9 2 0 3 }$ </td><td>14±04</td></tr><tr><td>NMI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $\mathrm { 4 1 \pm 0 3 }$ </td><td> ${ \bf 6 0 \pm 0 3 }$ </td><td>44±02</td><td>50±02</td><td>37±02</td><td> $^ { 2 4 \pm 0 3 }$ </td><td> $2 8 \pm 0 2$ </td><td>26±04</td></tr><tr><td>Time</td><td> $[ 1 0 ^ { 0 } \mathrm { s } ]$ </td><td> $0 3 { \pm } 0 0$ </td><td> $_ { 0 3 \pm 0 0 }$ </td><td>05±00</td><td>00±00</td><td>07±00</td><td> $_ { 0 3 \pm 0 0 }$ </td><td> $0 2 \pm 0 0$ </td><td> $\mathbf { 0 0 } { \pm } \mathbf { 0 0 }$ </td></tr><tr><td rowspan="4"> $\mathrm { M N I S T }$   $\alpha = 3 . 0$ </td><td>ACC [10−2]</td><td></td><td> ${ 5 1 \pm 0 3 }$ </td><td> ${ \bf 6 3 \pm 0 4 }$ </td><td>53±02</td><td>56±02</td><td>47±02</td><td> ${ 3 0 \pm 0 3 }$ </td><td>38±03</td><td>35±04</td></tr><tr><td>ARI</td><td> $[ 1 0 ^ { - 2 } ]$ </td><td> $2 9 { \pm } 0 4$ </td><td>48±05</td><td>33±02</td><td>39±02</td><td>26±02</td><td> ${ 1 2 \pm 0 3 }$ </td><td>19±03</td><td>14±04</td></tr><tr><td>NMI</td><td> $\dot { [ 1 0 ^ { - 2 } ] }$ </td><td> $\mathrm { 4 1 \pm 0 3 }$ </td><td>61±03</td><td>42±01</td><td>50±02</td><td>37±02</td><td>22±03</td><td>28±02</td><td>27±04</td></tr><tr><td>Time</td><td> $[ 1 0 ^ { 0 } \mathrm { s } ]$ </td><td> $1 0 { \pm } 0 7 $ </td><td>10±07</td><td>05±00</td><td>00±00</td><td>07±00</td><td>03±00</td><td>02±00</td><td>00±00</td></tr></table>

The main idea of eFDPC is to form clusters from axis-parallel (diagonal-covariance) micro-clusters, which have substantially lower computational cost, and to retain much of the descriptive power of full-covariance Gaussian models used in evolving fuzzy systems [8, 44, 57] through density-peak selection and peak-based label propagation. Full-covariance clusters have an advantage over axis-parallel (diagonal) clusters because they model correlations between features, which can improve accuracy and/or reduce the number of required clusters compared to spherical models [57]. However, a full covariance matrix has $d ( d + 1 ) / 2$ free parameters, and reliable estimation requires at least that number of samples per cluster, whereas high-dimensional data are often sparse. Consequently, for high-dimensional data, a diagonal covariance with d parameters, or even a spherical model, often performs better while requiring far less computation [44, 58]. Further, the time complexity to compute the inverse of a covariance matrix used in distance and overlap computations is $\mathcal { O } ( d ^ { 3 } )$ , while for diagonal covariances it is $\mathcal O ( d )$ , implying that multivariate methods are typically much slower than methods that rely on centers and variances only.

A key component is overlap-aware merging during micro-cluster evolution. Existing dissimilarity measures used in evolving clustering sufer from several limitations. First, measures based on covariance determinants or explicit volume computation are computationally expensive, especially with multivariate Gaussian covariances, since determinant and inverse operations are costly and pairwise comparisons lead to high overall cost [26]. Second, many measures do not include sample counts, so clusters with diferent support are treated equally [32]. Third, measures that rely on the geometric mean of covariance determinants, such as the Bhattacharyya distance [57, 59], can fail to detect overlap between small and large clusters because they measure absolute overlap, which can be small even when clusters share the same center. In eFDPC, the overlap-aware merging criterion is designed to account for both dispersion and relative support, which reduces premature merging and improves robustness when clusters are skewed, overlapping, or have very dissimilar densities.

The eFDPC clients do not transmit absolute micro-cluster supports, as these may leak additional information about local data. Interestingly, avoiding absolute counts can also improve the quality of density peak selection. The resulting peak set is less dominated by large local densities, $i . e .$ , dense regions do not overwhelm the final model, and lower-density peaks can still contribute meaningfully, $e . g .$ , in the Unbalance and KDD99 datasets. Further, using bounded supports $\tilde { n } _ { c } ^ { ( k ) } \in [ 0 , 1 ]$ together with equal treatment of transmitted centers acts as a soft voting mechanism across clients, so each client can have comparable influence on the aggregated model even under non-IID partitions. This may reduce the impact of strongly biased local datasets, although defenses against adversarial clients require further investigation. The number of peaks $P$ is provided in our experiments for a fair comparison with the baselines. $P$ could be selected automatically, for example as an elbow on the sorted decision scores $\gamma _ { i } = \rho _ { i } \delta _ { i }$ 2 but no unsupervised criterion recovers the true number of clusters reliably.

The proposed framework also has limitations. First, peak selection and propagation depend on the stability of micro-cluster centers. When the decision graph is flat and no clear peaks exist, grouping can become sensitive to local perturbations and neighborhood construction. Second, the overall quality depends on hyperparameters such as the merging aggressiveness and neighborhood size used to localize density or propagation, which must be selected consistently across datasets. Third, while microcluster peaks reduce communication compared to transmitting samples, they still introduce communication overhead relative to pure centroid-only baselines, and the shufler-based anonymization layer adds system complexity and communication overhead. Finally, propagation can blur boundaries when micro-clusters are mixed. For applications where strict boundary separation is critical, peak-only grouping (eFDPC-P) may be preferable. eFDPC-G is more suitable when the data are not well described by centroid-based clusters and instead lie on a lower-dimensional manifold. Because the server reconstructs global peaks from all clients, a peak missed by one client is recovered from another (Fig. 3). The cluster is missed only when a true cluster is under-represented for all clients and no client transmits a peak for it.

The peaks computed with DPC are derived from micro-cluster centers, so peak selection can be sensitive to perturbations in micro-cluster assignments during cluster evolution. If a label propagation stage is used, it can also reduce clustering accuracy, since boundary micro-clusters may contain samples from multiple clusters and can shift the assignment balance. Rather than assigning each micro-cluster to its nearest higherdensity neighbor as in classical $\mathrm { D P C } .$ , we propagate labels simultaneously from all selected peaks over $E _ { \kappa }$ . This results in a partition that follows the local connectivity of $E _ { \kappa }$ and typically produces smoother decision boundaries. In our experiments, nearesthigher propagation was often unstable for micro-clusters, as small shifts in center positions can change nearest-higher relations and produce uneven groups, especially on flatter datasets with no clear density peaks. In contrast, multi-source Dijkstra propagation from peaks was more stable because it depends on local paths in $E _ { \kappa }$ rather than a global density ordering.

Privacy could be further increased by applying Diferential Privacy (DP). However, applying DP directly to the transmitted centers is not viable in our setting because, for typical clustering datasets and $\varepsilon = 1$ , the required noise is on the order of the data spread (or larger). As a result, global peak selection becomes unstable and clustering utility degrades substantially. DP is therefore most viable when only securely aggregated information is revealed to the server, $e . g .$ , when the server proposes candidates, clients vote, and the votes are securely aggregated. Multi-party computation could also be used to hide the global-peak computation from all parties. However, such secure computation is typically several orders of magnitude slower than the proposed eFDPC pipeline [60], which makes it dificult to apply at the scale of the considered datasets.

## 5 Conclusion

This work introduced eFDPC, a one-shot federated clustering framework that combines evolving fuzzy clustering and density peak clustering in a federated setting. Each client evolves micro-clusters from private mini-batches through incremental clustering, cluster addition, and merging, and then transmits only density-peak-selected local representatives through a shufler-based anonymization layer to enable unlinkable global peak selection at the server. Experiments on synthetic and real-world benchmarks showed that eFDPC consistently ranks among the best-performing approaches under non-IID partitions. It achieves strong ACC and typically improves ARI/NMI on challenging structures such as overlap, varying density, and high dimensionality, while maintaining runtime comparable to k-means methods. A key limitation of eFDPC is that peak selection and propagation can be sensitive when no clear density peaks exist, and performance depends on quantization and merging hyperparameters. Future work may include extending eFDPC to semi-supervised federated clustering and evolving environments, as well as studying robustness to adversarial clients, privacy mechanisms at the aggregation stage, and communication-aware privacy budgeting under multi-round updates.

## Appendix A Derivation of the Overlap score

Consider two multivariate Gaussian clusters $\mathcal { C } _ { p }$ and $\mathcal { C } _ { q }$ with means $\boldsymbol { \mu _ { p } } , \boldsymbol { \mu _ { q } } ~ \in ~ \mathbb { R } ^ { d }$ covariance matrices $ { \Sigma } _ { p } ,  { \Sigma } _ { q } \in \mathbb { R } ^ { d \times d }$ , and supports $n _ { p } , n _ { q }$ . The goal is to determine if $p , q$ should be merged.

Based on the multivariate Gaussian merging formulation [32], the covariance matrix of the merged cluster can be split into the average covariance and spread:

$$
\pmb { \Sigma } = \pmb { \Sigma } ^ { \circ } + \pmb { \Sigma } ^ { - } ,\tag{A1}
$$

$$
\Sigma ^ { \circ } = \frac { n _ { p } \Sigma _ { p } + n _ { q } \Sigma _ { q } } { n _ { p } + n _ { q } } ,\tag{A2}
$$

$$
\pmb { \Sigma } ^ { - } = \frac { n _ { p } n _ { q } } { ( n _ { p } + n _ { q } ) ^ { 2 } } ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) ^ { \top } .\tag{A3}
$$

Here, $\pmb { \Sigma } ^ { \circ } \in \mathbb { R } ^ { d \times d }$ denotes the pooled covariance without spread, while $\pmb { \Sigma } ^ { - } \in \mathbb { R } ^ { d \times d }$ is a rank-one positive semidefinite matrix capturing the dispersion induced by the separation between the cluster centers. The hyper-volume of a d-dimensional Gaussian cluster with covariance $\pmb { \Sigma } \in \mathbb { R } ^ { d \times d }$ is proportional to [26]:

$$
V ( { \bf { \Sigma } } ) = \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } \sqrt { \operatorname* { d e t } ( { \bf { \Sigma } } ) } .\tag{A4}
$$

and relative volume changes depend only on determinant ratios.

We propose to measure dissimilarity based on the relative covariance volume contribution of the center-dispersion term $\Sigma ^ { - }$ . However, this cannot be measured directly because its hyper-volume is 0. Accordingly, we consider the ratio between the hyper-volume of the merged covariance Σ and that with the pooled covariance $\pmb { \Sigma } ^ { \circ }$

$$
{ \frac { V } { V ^ { \circ } } } = { \sqrt { \frac { \operatorname* { d e t } ( \Sigma ) } { \operatorname* { d e t } ( \Sigma ^ { \circ } ) } } } = { \sqrt { \operatorname* { d e t } ( ( \Sigma ^ { \circ } ) ^ { - 1 } \Sigma ) } } .\tag{A5}
$$

We define the overlap score as the squared relative volume ratio:

$$
\kappa _ { p q } = ( V / V ^ { \circ } ) ^ { 2 } = \operatorname* { d e t } \bigl ( ( \Sigma ^ { \circ } ) ^ { - 1 } \Sigma \bigr ) .\tag{A6}
$$

Substituting the covariance decomposition in Eq. (A1):

$$
( \Sigma ^ { \circ } ) ^ { - 1 } \Sigma = I + \frac { n _ { p } n _ { q } } { ( n _ { p } + n _ { q } ) ^ { 2 } } ( \Sigma ^ { \circ } ) ^ { - 1 } ( \mu _ { p } - \mu _ { q } ) ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) ^ { \top } .\tag{A7}
$$

Using the matrix determinant lemma [61], de $\begin{array} { r } { \dot { \boldsymbol { \mathscr { \mathbf { \rho } } } } ( \pmb { I } + \pmb { u } \pmb { v } ^ { \top } ) = 1 + \pmb { v } ^ { \top } \pmb { u } . } \end{array}$ the determinant of Eq. (A7) simplifies to:

$$
\kappa _ { p q } = 1 + \frac { n _ { p } n _ { q } } { ( n _ { p } + n _ { q } ) ^ { 2 } } ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) ^ { \top } ( \pmb { \Sigma } ^ { \circ } ) ^ { - 1 } ( \pmb { \mu } _ { p } - \pmb { \mu } _ { q } ) .\tag{A8}
$$

The resulting overlap score $\kappa _ { p q }$ is symmetric, dimensionless, scales with the cluster spreads, and depends on the cluster means $\mu _ { p } , \mu _ { q } ,$ covariance matrices $\Sigma _ { p } , \Sigma _ { q } ,$ and supports $n _ { p } , n _ { q }$

## Statements and Declarations.

Competing Interests. The authors declare that they have no competing interests.

Funding. The authors acknowledge the financial support from the Slovenian Research and Innovation Agency (Ph.D. grant for Miha Oˇzbot, funding number P2-0219).

Data Availability. The datasets analyzed during the current study are publicly available from the sources cited in the manuscript.

Code Availability. The eFDPC PyTorch implementation and the experimental scripts are available at github.com/mihaozbot/eFDPC.

Ethics Approval. Not applicable.

Consent to Participate. Not applicable.

Consent for Publication. Not applicable.

## References

[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.y.: Communication-eficient learning of deep networks from decentralized data. In: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pp. 1273–1282. PMLR, Fort Lauderdale, FL, USA (2017). https://proceedings.mlr.press/v54/mcmahan17a.html

[2] Hu, L., Jiang, M., Dong, J., Liu, X., He, Z.: Interpretable clustering: A survey. ACM Computing Surveys 58(8), 215–121521 (2026) https://doi.org/10.1145/ 3789495

[3] Ngo, H., Fang, H., Rumbut, J., Wang, H.: Federated Fuzzy Clustering for Decentralized Incomplete Longitudinal Behavioral Data. IEEE Internet of Things Journal 11(8), 14657–14670 (2024) https://doi.org/10.1109/JIOT.2023.3343719

[4] Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.: Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems 2, 429–450 (2020)

[5] Li, X., Huang, K., Yang, W., Wang, S., Zhang, Z.: On the Convergence of FedAvg on Non-IID Data. In: International Conference on Learning Representations (ICLR), Addis Ababa, Ethiopia (2020)

[6] Kanamori, S., Abe, T., Ito, T., Emura, K., Wang, L., Yamamoto, S., Phong, L.T., Abe, K., Kim, S., Nojima, R., Ozawa, S., Moriai, S.: Privacy-Preserving Federated Learning for Detecting Fraudulent Financial Transactions in Japanese Banks. Journal of Information Processing 30(0), 789–795 (2022) https://doi.org/ 10.2197/ipsjjip.30.789

[7] Skrjanc, I., Ozawa, S., Ban, T., Dovˇzan, D.: Large-scale cyber attacks monitoring<sup>ˇ</sup> using Evolving Cauchy Possibilistic Clustering. Applied Soft Computing 62, 592– 601 (2018) https://doi.org/10.1016/j.asoc.2017.11.008

[8] Skrjanc, I., Iglesias, J., Sanchis, A., Leite, D., Lughofer, E., Gomide, F.: Evolving<sup>ˇ</sup> fuzzy and neuro-fuzzy approaches in clustering, regression, identification, and classification: A Survey. Information Sciences 490, 344–368 (2019) https://doi. org/10.1016/J.INS.2019.03.060

[9] Rodriguez, A., Laio, A.: Clustering by fast search and find of density peaks. Science 344(6191), 1492–1496 (2014) https://doi.org/10.1126/science.1242072

[10] Dennis, D.K., Li, T., Smith, V.: Heterogeneity for the Win: One-Shot Federated Clustering. In: Proceedings of the 38th International Conference on Machine Learning, vol. 139, pp. 2611–2620. PMLR, Online (2021)

[11] Corcuera B´arcena, J.L., Marcelloni, F., Renda, A., Bechini, A., Ducange, P.: Federated c-means and Fuzzy c-means Clustering Algorithms for Horizontally and Vertically Partitioned Data. IEEE Transactions on Artificial Intelligence, 1–15 (2024) https://doi.org/10.1109/TAI.2024.3426408

[12] Pan, C., Sima, J., Prakash, S., Rana, V., Milenkovic, O.: Machine unlearning of federated clusters. In: International Conference on Learning Representations (ICLR), Kigali, Rwanda (2023)

[13] Xu, J., Chen, H.-Y., Chao, W.-L., Zhang, Y.: Jigsaw Game: Federated Clustering. Transactions on Machine Learning Research (2024) https://doi.org/10. 48550/arXiv.2407.12764 . arXiv:2407.12764 [cs]

[14] Stallmann, M., Wilbik, A.: On a Framework for Federated Cluster Analysis. Applied Sciences 12(2020), 10455 (2022) https://doi.org/10.3390/app122010455

[15] Hu, X., Qin, J., Shen, Y., Pedrycz, W., Liu, X., Liu, J.: An Eficient Federated Multiview Fuzzy C-Means Clustering Method. IEEE Transactions on Fuzzy Systems 32(4), 1886–1899 (2024) https://doi.org/10.1109/TFUZZ.2023.3335361

[16] Zhang, L., Shi, Y., Chang, Y.-C., Lin, C.-T.: Federated Fuzzy Neural Network with Evolutionary Rule Learning. IEEE Transactions on Fuzzy Systems 31(5), 1653–1664 (2023) https://doi.org/10.1109/TFUZZ.2022.3207607

[17] Shi, Y., Lin, C.-T., Chang, Y.-C., Ding, W., Shi, Y., Yao, X.: Consensus Learning for Distributed Fuzzy Neural Network in Big Data Environment. IEEE Transactions on Emerging Topics in Computational Intelligence 5(1), 29–41 (2021) https://doi.org/10.1109/TETCI.2020.2998919

[18] Shi, Y., Zhang, L., Cao, Z., Tanveer, M., Lin, C.-T.: Distributed Semi-supervised Fuzzy Regression with Interpolation Consistency Regularization. IEEE Transactions on Fuzzy Systems 30(8), 3125–3137 (2022) https://doi.org/10.1109/ TFUZZ.2021.3104339

[19] Corcuera Barcena, J.L., Ducange, P., Ercolani, A., Marcelloni, F., Renda, A.: An Approach to Federated Learning of Explainable Fuzzy Regression Models. In: 2022 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), pp. 1– 8. IEEE, Padua, Italy (2022). https://doi.org/10.1109/FUZZ-IEEE55066.2022. 9882881

[20] Angelov, P.P., Zhou, X.: Evolving Fuzzy-Rule-Based Classifiers From Data Streams. IEEE Transactions on Fuzzy Systems 16(6), 1462–1475 (2008) https: //doi.org/10.1109/TFUZZ.2008.925904

[21] Pratama, M., Pedrycz, W., Lughofer, E.: Evolving Ensemble Fuzzy Classifier. IEEE Transactions on Fuzzy Systems 26(5), 2552–2567 (2018) https://doi.org/ 10.1109/TFUZZ.2018.2796099

[22] Lughofer, E., Skrjanc, I.: Online Active Learning for Evolving Error Feedback<sup>ˇ</sup> Fuzzy Models within a Multi-Innovation Context. IEEE Transactions on Fuzzy Systems, 1–14 (2023) https://doi.org/10.1109/TFUZZ.2023.3302403

[23] Wang, W., Liu, W., Chen, H.: Time-Series Forecasting via Fuzzy-Probabilistic Approach With Evolving Clustering-Based Granulation. IEEE Transactions on Fuzzy Systems 30(12), 5324–5336 (2022) https://doi.org/10.1109/TFUZZ.2022. 3173684

[24] Dovˇzan, D., Logar, V., Skrjanc, I.: Implementation of an Evolving Fuzzy Model <sup>ˇ</sup> (eFuMo) in a Monitoring System for a Waste-Water Treatment Process. IEEE Transactions on Fuzzy Systems 23(5), 1761–1776 (2015) https://doi.org/10.1109/ TFUZZ.2014.2379252

[25] Lughofer, E., Pratama, M.: Online sequential ensembling of predictive fuzzy systems. Evolving Systems 13(2), 361–386 (2022) https://doi.org/10.1007/ s12530-021-09398-x

[26] Skrjanc, I.: Cluster-Volume-Based Merging Approach for Incrementally Evolv-<sup>ˇ</sup> ing Fuzzy Gaussian Clustering-eGAUSS+. IEEE Transactions on Fuzzy Systems 28(9), 2222–2231 (2020) https://doi.org/10.1109/TFUZZ.2019.2931874

[27] Blaˇziˇc, S., Skrjanc, I.: Incremental Fuzzy C-Regression Clustering from Stream-<sup>ˇ</sup> ing Data for Local-Model-Network Identification. IEEE Transactions on Fuzzy Systems 28(4), 758–767 (2020) https://doi.org/10.1109/TFUZZ.2019.2916036

[28] Strˇzinar, Z.,<sup>ˇ</sup> Skrjanc, I., Pregelj, B.: Evolving interval-based time series clustering<sup>ˇ</sup> for streaming industrial data. Evolving Systems 16(3), 82 (2025) https://doi.org/ 10.1007/s12530-025-09713-w

[29] Angelov, P.P., Gu, X., Pr´ıncipe, J.C.: Autonomous Learning Multimodel Systems From Data Streams. IEEE Transactions on Fuzzy Systems 26(4), 2213–2224 (2018) https://doi.org/10.1109/TFUZZ.2017.2769039

[30] Bezerra, C.G., Costa, B.S.J., Guedes, L.A., Angelov, P.P.: An evolving approach to data streams clustering based on typicality and eccentricity data analytics. Information Sciences 518, 13–28 (2020) https://doi.org/10.1016/j.ins.2019.12. 022

[31] Garcia, C., Leite, D., Skrjanc, I.: Incremental Missing-Data Imputation for Evolving Fuzzy Granular Prediction. IEEE Transactions on Fuzzy Systems 28(10), 2348–2362 (2020) https://doi.org/10.1109/TFUZZ.2019.2935688

[32] Oˇzbot, M., Ozawa, S., Skrjanc, I.: eFedGauss: A Federated Approach to<sup>ˇ</sup> Fuzzy Multivariate Gaussian Clustering. In: 2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), pp. 1–8 (2024). https://doi.org/10.1109/ FUZZ-IEEE60900.2024.10611844

[33] Tobin, J., Zhang, M.: A theoretical analysis of density peaks clustering and the component-wise peak-finding algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(2), 1109–1120 (2024) https://doi.org/10.1109/ TPAMI.2023.3327471

[34] Yang, X., Xiao, F.: An improved density peaks clustering algorithm based on the generalized neighbors similarity. Engineering Applications of Artificial Intelligence 136, 108883 (2024) https://doi.org/10.1016/j.engappai.2024.108883

[35] Xie, J., Zhong, C., Tian, Y., Hu, L., Pan, T.: Robust density clustering based on density peak regions and k-nearest neighbors. Pattern Recognition 174, 113021 (2026) https://doi.org/10.1016/j.patcog.2025.113021

[36] Du, M., Ding, S., Jia, H.: Study on density peaks clustering based on k-nearest neighbors and principal component analysis. Know.-Based Syst. 99(C), 135–145 (2016) https://doi.org/10.1016/j.knosys.2016.02.001

[37] Wang, H., Zhang, J., Shen, Y., Wang, S., Deng, B., Zhao, W.: Improved density peak clustering with a flexible manifold distance and natural nearest neighbors for network intrusion detection. Scientific Reports 15(1), 8510 (2025) https:// doi.org/10.1038/s41598-025-92509-4

[38] Ding, L., Li, C., Ding, S., Xu, X., Guo, L., Wu, X.: Robust density peaks clustering for manifold data with multiple peaks. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(11), 10696–10708 (2025) https://doi.org/10.1109/ TPAMI.2025.3594121

[39] Wang, Y., Pang, W., Wang, D., Pedrycz, W.: One-shot secure federated k-means clustering based on density cores. IEEE Transactions on Neural Networks and Learning Systems 36(8), 14131–14143 (2025) https://doi.org/10.1109/TNNLS. 2025.3547362

[40] Ding, S., Li, C., Xu, X., Guo, L., Ding, L., Wu, X.: Horizontal federated density peaks clustering. IEEE Transactions on Neural Networks and Learning Systems 36(1), 820–829 (2025) https://doi.org/10.1109/TNNLS.2023.3329720

[41] Chen, D., Du, T., Zhou, J., Wu, Y., Wang, X.: Dwdp-stream: A dynamic weight and density peaks clustering algorithm for data stream. International Journal of Computational Intelligence Systems 15(1), 96 (2022) https://doi.org/10.1007/ s44196-022-00157-7

[42] Etherington, T.R.: Mahalanobis distances and ecological niche modelling: correcting a chi-squared probability error. PeerJ 7, 6678 (2019) https://doi.org/10. 7717/peerj.6678

[43] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., Red Hook, NY, USA (2017)

[44] Oˇzbot, M., Ozawa, S., Skrjanc, I.: Unsupervised federated learning based on evolv-<sup>ˇ</sup> ing gaussian clustering. In: 2025 IEEE International Conference on Fuzzy Systems (FUZZ), pp. 1–6 (2025). https://doi.org/10.1109/FUZZ62266.2025.11152249 . ISSN: 1558-4739. https://ieeexplore.ieee.org/abstract/document/11152249

[45] Hardy, S., Henecka, W., Ivey-Law, H., Nock, R., Patrini, G., Smith, G., Thorne, B.: Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption. arXiv preprint arXiv:1711.10677 (2017). arXiv:1711.10677

[46] Bittau, A., Erlingsson, U., Maniatis, P., Mironov, I., Raghunathan, A., Lie, D.,<sup>´</sup> Rudominer, M., Kode, U., Tinnes, J., Seefeld, B.: Prochlo: Strong privacy for analytics in the crowd. In: Proceedings of the 26th Symposium on Operating Systems Principles. SOSP ’17, pp. 441–459. Association for Computing Machinery, New York, NY, USA (2017). https://doi.org/10.1145/3132747.3132769 . https://dl.acm.org/doi/10.1145/3132747.3132769

[47] Zahn, C.T.: Graph-theoretical methods for detecting and describing gestalt clusters. IEEE Transactions on Computers C–20(1), 68–86 (1971) https://doi.org/ 10.1109/T-C.1971.223083

[48] Veenman, C.J., Reinders, M.J.T., Backer, E.: A maximum variance cluster algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence 24(9), 1273–1280 (2002) https://doi.org/10.1109/TPAMI.2002.1033218

[49] Fr¨anti, P., Virmajoki, O.: Iterative shrinking method for clustering problems. Pattern Recognition 39(5), 761–765 (2006) https://doi.org/10.1016/j.patcog.2005. 09.012

[50] Fu, L., Medico, E.: Flame, a novel fuzzy clustering method for the analysis of dna microarray data. BMC Bioinformatics 8(1), 3 (2007) https://doi.org/10.1186/ 1471-2105-8-3

[51] Gionis, A., Mannila, H., Tsaparas, P.: Clustering aggregation. ACM Trans. Knowl. Discov. Data 1(1), 4 (2007) https://doi.org/10.1145/1217299.1217303

[52] Jain, A.K., Law, M.H.C.: Data clustering: A user’s dilemma. In: Pal, S.K., Bandyopadhyay, S., Biswas, S. (eds.) Pattern Recognition and Machine Intelligence, pp.

1–10. Springer, Berlin, Heidelberg (2005). https://doi.org/10.1007/11590316 1

[53] Rezaei, M., Fr¨anti, P.: Can the number of clusters be determined by external indices? IEEE Access 8, 89239–89257 (2020) https://doi.org/10.1109/ACCESS. 2020.2993295

[54] Kelly, M., Longjohn, R., Nottingham, K.: The UCI Machine Learning Repository. https://archive.ics.uci.edu. University of California, Irvine, School of Information and Computer Sciences (2023)

[55] Li, Q., Diao, Y., Chen, Q., He, B.: Federated Learning on Non-IID Data Silos: An Experimental Study. In: 2022 IEEE 38th International Conference on Data Engineering (ICDE), Kuala Lumpur, Malaysia, pp. 965–978 (2022). https://doi. org/10.1109/ICDE53745.2022.00077

[56] Hsu, T.-M.H., Qi, H., Brown, M.: Measuring the Efects of Non-Identical Data Distribution for Federated Visual Classification. arXiv preprint arXiv:1909.06335 (2019). arXiv:1909.06335

[57] Lughofer, E., Cernuda, C., Kindermann, S., Pratama, M.: Generalized smart evolving fuzzy systems. Evolving Systems 6(4), 269–292 (2015) https://doi.org/ 10.1007/S12530-015-9132-6/FIGURES/12

[58] Bouveyron, C., Girard, S., Schmid, C.: High-dimensional data clustering. Computational Statistics & Data Analysis 52(1), 502–519 (2007) https://doi.org/10. 1016/j.csda.2007.02.009

[59] Oˇzbot, M., Skrjanc, I.: Online Design of Experiments with Fuzzy Confidence <sup>ˇ</sup> Interval for Identification of Nonlinear Dynamical Processes. In: 2024 IEEE International Conference on Evolving and Adaptive Intelligent Systems (EAIS), pp. 1–8 (2024). https://doi.org/10.1109/EAIS58494.2024.10569113

[60] Keller, M.: Mp-spdz: A versatile framework for multi-party computation. In: Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security. CCS ’20, pp. 1575–1590. Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3372297.3417872 . https://dl.acm.org/doi/10.1145/3372297.3417872

[61] Petersen, K.B., Pedersen, M.S.: The Matrix Cookbook. Technical University of Denmark. Version 20121115 (2012). http://www2.compute.dtu.dk/pubdb/pubs/ 3274-full.html