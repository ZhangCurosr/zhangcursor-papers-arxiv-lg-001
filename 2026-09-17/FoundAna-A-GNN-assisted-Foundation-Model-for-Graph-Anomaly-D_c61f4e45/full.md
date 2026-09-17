# FoundAna: A GNN-assisted Foundation Model for Graph Anomaly Detection

Suprim Nakarmi<sup>∗</sup>, Chahana Dahal<sup>∗</sup>, Yue Zhao<sup>†</sup>, Junggab Son<sup>∗</sup> and Zuobin Xiong<sup>∗</sup>

<sup>∗</sup>Department of Computer Science, University of Nevada Las Vegas, Las Vegas, USA

<sup>†</sup>Department of Computer Science, University of Southern California, Los Angeles, USA

{suprim.nakarmi, chahana.dahal, junggab.son, zuobin.xiong}@unlv.edu; <sup>†</sup> yue.z@usc.edu

Abstract—Graph anomaly detection aims to identify graph structures (e.g., nodes, edges, or subgraphs) that deviate significantly from expected patterns, which supports critical applications in fraud detection, spam identification, network intrusion, etc. Despite the growing methods in the field, existing approaches follow a one-model-per-dataset paradigm, limiting their transferability across diverse real-world scenarios due to task heterogeneity, label scarcity, and domain variability. In this work, we introduce FoundAna, a GNN-assisted Foundation Model for Graph Anomaly Detection – the first foundation model framework designated for generalizable, cross-graph anomaly detection by combining GNNs and transformers. FoundAna integrates an anomaly detection-specific GNN component with a standard transformer encoder augmented by four complementary positional encodings, which enable the model to capture both local and global structural information. Specifically, the positional encoding enriched node representations are passed through attribute and adjacency decoders, and the reconstruction errors serve as the anomaly score. Extensive experiments on nine benchmark datasets spanning financial, social, and citation network domains demonstrate that FoundAna consistently outperforms state-of-the-art baselines. The code implementation and Supplementary materials are here: https://github.com/FoundAna331/FoundAna.

Index Terms—Graph Anomaly Detection, Foundation Model, Cross-domain Graphs, Cross-task Graphs

## I. INTRODUCTION

Anomaly Detection (AD) has been an important topic in the field of research and practice, which refers to identifying abnormal data samples (also called outliers) that deviate significantly from the distribution of the majority [1], [2]. A wide spectrum of applications, including detecting fraudulent transactions in insurance, banking, etc. [2]; identifying malicious network intrusion in cybersecurity infrastructure [3]; and ensuring safety in aviation and industrial control systems [2], [4], heavily rely on such anomaly detection techniques. Classical approaches in AD employ statistical, distance-based, densitybased, and kernel-based methods to characterize abnormality through mathematical criteria [4]. However, these methods fail to scale to the complex, high-dimensional data distributions prevalent in modern applications, leading to the advent of deep learning-based models such as deep autoencoders, variational generative models, and self-supervised architectures [5]. Despite these advances, a critical limitation persists in almost all learning-based methods, as they are designed for Euclidean data modalities (e.g., images and tabular records), which are less effective in capturing topological dependencies inherent in graph-structured data [6].

The semantic and topological complexity of graphstructured data is determined by the attributes of nodes and their structural connectivity within the graph. This dual nature induces a heterogeneous anomaly landscape that spans multiple levels: node anomaly may represent a bot account generating thousands of spurious connections in a social network [7]; an edge anomaly may correspond to a sudden, high-value financial transaction between accounts that rarely interact [8]; and a (sub)graph anomaly may indicate a rare or structurally deviant molecular ring deviating from known chemical norms [9]. In the traditional Graph Anomaly Detection (GAD), Graph Neural Networks (GNNs) are the dominant backbone, enabling joint encoding of structural and attribute information through message passing mechanisms [6]. However, GAD using GNNs faces three fundamental challenges: (i) task heterogeneity: node, edge, and subgraph-level anomaly detection require distinct model designs, (ii) label scarcity: anomalies are inherently rare and expensive to annotate in a graph at scale, and (iii) domain variability: graph characteristics differ across application domains, limiting cross-dataset transferability. These challenges have motivated a recent shift toward Graph Foundation Models (GFMs) for GAD, which uses a single and powerful framework to generalize across tasks and domains, with limited labeled data.

Foundation models [10] are typically pre-trained on large and diverse data corpora and subsequently adapted to downstream tasks with minimal fine-tuning. Specifically, all of the modern FMs in the Natural Language Processing and Computer Vision domains use transformers as the backbone [11], [12]. Inspired by this success, the graph learning community has developed transformer-based GFMs that generalize across diverse graph tasks and domains [13], [14], which provide unbounded global self-attention to capture long-range dependencies, thereby overcoming the over-smoothing and oversquashing issues of message-passing GNNs [15]. However, leveraging standard transformers for graphs is non-trivial, as they treat nodes as an unordered set of tokens and lack any built-in mechanism to encode graph topology. Therefore, the selection of positional encoding, which provides the structural information for node representations before attention, is of critical importance. For example, using Laplacian eigenvectors as positional encoding provides a mathematical map of the graph’s global geometry, and a myriad of works have demonstrated graph positional encoding for general graph tasks in the literature [16], [17]. Yet, very few studies have investigated positional encoding specific to anomaly detection methods.

![](images/aaaa30f2f83a3752d217e083a7bb08d495841da96cabbf081f48259418e27b47.jpg)  
(a) Node degree

![](images/1091d081bb455eda1bbd05eac8208d010323217a4bc8ec7ae16a2a8e318b5b0a.jpg)  
(b) Residual distance  
Fig. 1. Distribution of node degree and residual distance in normal and anomaly nodes. The figure was plotted using the Facebook dataset.

In this work, we develop a GFM by integrating the benefit of GNNs for local structural learning (w.r.t. local anomaly) and the strength of the transformer for capturing the global graph attention (w.r.t. universal performance). Furthermore, we adopt four different positional encoding methods, including the Stable and Expressive Positional Encoding (SPE) [18], Random Walk Structural Encoding (RWSE) [19], node degree encoding, and residual distance encoding. In addition, the node degree and residual distance encoding are specialized to capture anomaly nodes with a distinct distribution from normal nodes, as shown in Figure 1(a) and Figure 1(b). To the best of our knowledge, no prior work has proposed an anomalyspecific GNN coupled with transformers as the foundational graph anomaly detection model. The main contributions are summarized as follows:

• We propose the first GNN-assisted transformer-based GFM specifically designed for GAD that generalizes well in cross-domain and cross-task detections.

• To echo the GNN-assisted transformer, we introduce a novel multi-scale positional encoding scheme integrating SPE, RWSE, residual distance, and node degree, providing both anomaly-relevant local structure and global context for transformer attention.

• Experimental results on nine datasets and multiple baselines, including GNNs and GFMs, confirm the superiority of the proposed method, especially in the few-shot learning scenario.

## II. RELATED WORK

## A. Graph Anomaly Detection

Early approaches to GAD relied on statistical or shallow learning methods; however, leveraging GNNs has gained much popularity for their structurally aware representations, which substantially improved detection performance [6], [20]. Over time, GNN-based GAD has evolved into three primary directions: reconstruction-based, contrastive learning-based, and GNN aggregation-based. Reconstruction-based methods typically train models to encode normal graph patterns and detect anomalies as nodes or structures that are poorly reconstructed [21]. Contrastive-learning based method provides a semi-supervised learning mechanism for GAD by pushing normal nodes to be similar to their local neighborhoods and pulling anomalies away [22]. GNN-aggregation consists of a family of methods that addresses GAD through careful design of GNN aggregation and spectral filtering [23]. All these methods are designed to detect anomalies for a specific data domain and a specific graph task, such as node, edge, and graph level.

Even though GAD models have superior performances, they follow a one model per dataset setting, meaning a model trained on a fraud detection graph cannot be reused for a social network anomaly task without retraining, which leads to a common issue of heavy data annotation demands and poor generalization [24], [25].

## B. Graph Foundation Models

Inspired by foundation models in NLP and computer vision, GFMs emerged to move from dataset-specific GNNs toward unified, transferable graph models. GraphBERT [16] introduced pure-attention graph learning, while self-supervised pretraining on molecular graphs demonstrated that graph transformers can learn transferable representations without taskspecific supervision. One For All (OFA) [26] reformulated all graph tasks into a unified node-classification problem using prompt graphs. OpenGraph [17] resolved feature heterogeneity through LLM-based shared embedding spaces, and GFT [27] introduced a transferable tree-structured vocabulary for graphagnostic learning.

Building on these foundations, recent works have extended GFMs to graph anomaly detection in two directions: GNN-backbone and transformer-backbone. GNN-backbone approaches leverage message-passing architectures as the core encoder. UniGAD [28] unified node, edge, and graph-level AD under a single model, showing that multi-level joint training regularizes representations and improves performance at each graph-level task. UniFORM [29] handles AD across diverse graph types under a single training objective, and GFM-UAD [30] extended this to a fully unified framework pretrained once and deployed across all three graph-level tasks. Similarly, AnomalyGFM [31] addresses cross-graph feature heterogeneity by encoding anomaly signals as residual distances, which is the deviation of a node’s representation from its neighbors. Transformer-backbone methods replace GNN encoders with attention mechanisms. This direction includes TFM4GAD [32], which encodes structural position through Laplacian embeddings and adds PageRank-based structural characteristics and neighborhood aggregation using beta wavelet filters. UNIP [25], which designs transferable neighborhood prompts enabling zero-shot AD on entirely unseen graphs; and ARC [24], which reframes GAD as an incontext learning problem where a few labeled examples guide anomaly scoring without retraining.

However, existing GFMs for anomaly detection function on fixed or pre-aligned feature spaces and do not fully leverage transformer architectures [24], [28], [31], which limits their ability to model both local neighborhood structure and global graph topology simultaneously. To address this gap, we propose FoundAna, which combines a GNN backbone with a transformer to jointly capture local and global structural context. The GNN encodes local neighborhood information, while the transformer leverages a combination of multi-scale positional encoding to model global structural position.

## III. PRELIMINARIES

In this section, we briefly conceptualize GFMs, GAD, and the problem formulation.

## A. Graph Anomaly Detection

GAD aims to identify nodes, edges, or subgraphs that deviate significantly from the expected normal data samples within a graph. Here, we define the attributed graphs and GAD.

Definition 1 (Attributed Graph). An attributed graph is defined as $\mathcal { G } = ( \nu , \mathcal { E } , \mathbf { X } )$ , where $\mathcal { V } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \}$ is the set of nodes with $| \mathcal { V } | = n , \mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ is the set of edges, and $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ is the node feature matrix with input dimensionality d, where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ denotes the feature vector of node $u _ { i }$ . The graph topology is encoded by an adjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { n \times n }$ , where:

$$
\begin{array} { r } { \mathbf { A } _ { i j } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ a n ~ e d g e ~ e x i s t s ~ b e t w e e n ~ } v _ { i } \mathrm { ~ a n d ~ } v _ { j } , } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}
$$

Definition 2 (Graph Anomaly Detection). Given an attributed graph G, graph anomaly detection aims to learn a scoring function $\phi : \mathcal { V } \to \mathbb { R }$ that assigns an anomaly score: $s _ { i } = \phi ( v _ { i } )$ to each node $v _ { i } \in \mathcal V$ (or E, G for edge and graph anomaly, respectively).

## B. Graph Foundation Models

Definition 3 (Graph Foundation Model). A graph foundation model is a pretrained encoder $\mathcal { F } _ { G }$ that maps an attributed graph $\mathcal { G } = ( \nu , \mathcal { E } , \mathbf { X } )$ , with node feature matrix $\mathbf { X } \in \mathbb { R } ^ { | \nu | \times d } ;$ , to a domain-agnostic latent space: $\mathbf { Z } = { \mathcal { F } } _ { G } ( { \mathcal { G } } ; \theta ^ { * } )$ , where $\theta ^ { * } =$ arg min<sub>θ</sub> $\mathcal { L } _ { \mathrm { p r e } } \left( \mathcal { F } _ { G } ( \cdot ; \theta ) , \mathcal { G } \right)$ denotes the optimal parameters obtained by pretraining on a corpus of graphs, $\mathbf { Z } \in \mathbf { \bar { \mathbb { R } } } ^ { | \mathcal { V } | \times d _ { h } }$ is the latent space node embedding matrix, and $d _ { h }$ is the hidden dimension, such that Z generalizes across domains and tasks with minimal adaptation.

A graph transformer treats each pre-processed node $u _ { i } \in$ V as a token and computes pairwise attention over all node pairs via the scaled dot-product mechanism. Formally, it can be defined as follows:

Definition 4 (Graph Transformer). A graph transformer is a function $\mathcal { T } : \mathbb { R } ^ { n \times \bar { d } }  \mathbb { R } ^ { n \times d _ { h } }$ that operates on a set of node tokens $\{ u _ { i } \} _ { i = 1 } ^ { n }$ without the canonical ordering assumption. Given an attributed graph $\mathcal { G } = ( \nu , \mathcal { E } , \mathbf { X } )$ , the input to the first layer is initialized as $\mathbf { H } ^ { ( 0 ) } = \mathbf { X } \in \mathbb { R } ^ { n \times d }$ . At each layer l, given the node representation matrix $\mathbf { H } ^ { ( l ) } \in \mathbb { R } ^ { n \times d _ { h } }$ , the transformer computes projections $\mathbf { Q } = \mathbf { H } ^ { ( l ) } \mathbf { W } ^ { Q } , \mathbf { K } = \mathbf { H } ^ { ( l ) } \mathbf { W } ^ { K } , \mathbf { V } = $ $\mathbf { H } ^ { ( l ) } \mathbf { W } ^ { V }$ with learnable weights $\mathbf { W } ^ { Q } , \mathbf { W } ^ { K } , \mathbf { W } ^ { V } \in \mathbb { R } ^ { d \times d _ { k } }$ <sup>k</sup> , and produces an output via scaled dot-product attention:

$$
{ \mathrm { A t t e n t i o n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { T } } { \sqrt { d _ { k } } } } \right) \mathbf { V }\tag{1}
$$

where $d _ { k }$ is the key dimension used for scaling. The attention output $\mathbf { \Lambda } \in \mathbb { R } ^ { n \times d _ { k } }$ is subsequently projected to the hidden dimension via ${ \bf W } ^ { O } ~ \in ~ \mathbb { R } ^ { \hat { d _ { k } } \times d _ { h } }$ , yielding the output node representations $\mathbf { H } ^ { ( l + 1 ) } \in \mathbb { R } ^ { n \times d _ { h } }$ . This attends to all n×n node pairs simultaneously and gives every node a global receptive field over the entire graph [33].

Unlike sequential text and vision-based transformers, selfattention treats the graph as a fully connected set of tokens, providing a global receptive field but discarding local structure inductive biases [33]. Therefore, the positional encoding in graph transformers augments H from graph topology prior to attention computation, restoring structural awareness without sacrificing global receptive fields, as discussed in Section IV-B.

## C. Problem Formulation

Definition 5 (Multi-Domain Graph Collection). A multi-domain graph collection is defined as G = $\mathcal { G } ^ { ( 1 ) } , \mathcal { G } ^ { ( 2 ) } , \ldots , \mathcal { \bar { G } } ^ { ( M ) }$ , where each $\mathcal { G } ^ { ( m ) } = ( \mathcal { V } ^ { ( m ) } , \mathcal { E } ^ { ( m ) } , \mathbf { X } ^ { ( m ) } )$ is an attributed graph from a distinct domain with potentially different feature dimensionality $d ^ { ( m ) }$ , graph size ${ \boldsymbol n } ^ { ( m ) }$ , and structural characteristics.

Problem Statement (Foundation Model for Graph Anomaly Detection). Given a multi-domain graph collection G with heterogeneous feature spaces $\mathbf { X } ^ { \left( m \right) } \in \mathbb { R } ^ { n ^ { \left( m \right) } \times d ^ { \left( m \right) } }$ , our objective is to pretrain a unified encoder $\mathcal { F } _ { G }$ with shared parameters $\theta ^ { * }$ and a decoder D that reconstructs both the node feature matrix $\hat { \mathbf { X } } ^ { ( m ) }$ and the adjacency matrix $\hat { \mathbf { A } } ^ { ( m ) }$ from the latent node embeddings $\mathbf { Z } ^ { ( m ) } \dot { = } \mathcal { F } _ { G } \dot { ( } \mathcal { G } ^ { ( m ) } ; \theta ^ { * } )$ . The anomaly score for each node $v _ { i }$ is defined as the combined reconstruction error over its features and structural connections:

$$
s _ { i } = \gamma _ { 1 } \cdot \Vert \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } \Vert _ { 2 } + \gamma _ { 2 } \cdot \sum _ { j \in \mathcal { N } ( i ) } \left. \mathbf { A } _ { i j } - \hat { \mathbf { A } } _ { i j } \right.\tag{2}
$$

where $\gamma _ { 1 } , \gamma _ { 2 } \geq 0$ are weighting coefficients balancing feature and structural reconstruction errors. The model operates in an unsupervised setting and is designed to generalize across unseen graphs from new domains with minimal adaptation (e.g., zero-shot or few-shot settings).

## IV. METHODOLOGY: FOUNDANA

In this section, we propose FoundAna, a universal GAD framework designed to operate across graph datasets without dataset-specific retraining (illustrated in Figure 2 and Algorithm 1). It consists of four stages: (1) SVD-based feature alignment (Section IV-A), (2) multi-scale positional encoding (Section IV-B), (3) a GNN-assisted transformer that fuses local spectral and global attention representations (Section IV-C), and (4) a dual reconstruction decoder for anomaly score (Section IV-D).

![](images/fac0f9256754bd12fd3ee25a3a52f78a24da54122970cc6c9112a7e50b1ac41f.jpg)  
Fig. 2. The framework of the proposed FoundAna.

## A. SVD-based Feature Alignment

To enable a single model to work on heterogeneous graph datasets with different node feature dimensions, we first project all node features into a shared latent space of dimension $d _ { u } .$ . Formally, for each dataset with feature matrix $\mathbf { X ^ { ( m ) } } \in \mathbb { R } ^ { N ^ { ( m ) } \times d ^ { ( m ) } }$ , we compute its truncated Singular Value Decomposition (SVD) $\begin{array} { r } { \dot { \mathbf { X } ^ { ( m ) } } = \mathbf { U } ^ { ( m ) } \sum ^ { ( m ) } ( \mathbf { V } ^ { ( \tilde { m } ) } ) ^ { T } } \end{array}$ and construct a dataset-specific projection matrix $\dot { \mathbf { W } } ^ { ( m ) } =$ $\mathbf { V } _ { 1 : d _ { \boldsymbol { u } } } ^ { ( m ) } \in \mathbb { R } ^ { d ^ { ( m ) } \times d _ { \boldsymbol { u } } }$ . The unified representation is then obtained as $\tilde { \mathbf { X } } ^ { ( m ) } \ = \ \mathbf { X } ^ { ( m ) } W ^ { ( m ) } \ \in \ \mathbb { R } ^ { N ^ { ( m ) } \times d _ { u } }$ , which gives lowdimension embedding that approximately preserves the dominant variance structure of the original features while aligning all datasets into a common feature space for downstream training and tasks.

## B. Graph Positional Encoding

The goal of positional encoding (PE) is to augment each node $v _ { i }$ with a structural descriptor $p _ { i } \in \mathbb { R } ^ { \mathfrak { p } }$ , such that the transformer input token for node $v _ { i }$ becomes $u _ { i } = [ x _ { i } | | p _ { i } ]$ $u _ { i } ~ \in ~ \mathbb { R } ^ { d _ { u } + p }$ , where || denotes concatenation. To enhance the graph structural properties in anomaly detection, we selected four PE: SPE [18], RWSE [34], residual distance [31], and node degree. SPE encodes stable and expressive global structural position and resolves the instability of prior methods (e.g., Laplacian PE) by replacing the hard partition of eigenspaces with a soft, eigenvalue-dependent partition [18]. Given the k smallest eigenpairs $( E _ { v } , \lambda )$ of the normalized graph Laplacian defined by $\mathbf { \bar {  { L } } } = \bar { \boldsymbol { I } } - D ^ { - 1 / 2 } A D ^ { - 1 / 2 }$ , where D is the diagonal degree matrix, SPE can be computed as:

$$
S P E ( V , \lambda ) _ { i } = \rho ( \sum _ { l = 1 } ^ { k } \phi _ { l } ( \lambda _ { l } ) . e _ { v i } ^ { ( l ) } ,\tag{3}
$$

where $\phi _ { l } : \mathbb { R } \to \mathbb { R } ^ { r }$ are learnable functions applied to the $l ^ { t h }$ eigenvalue, $e _ { v i } ^ { ( l ) }$ is the l-th eigenvector entry for node $v _ { i } ,$ and $\rho : \mathbb { R } ^ { r }  \mathbb { R } ^ { p }$ is a learnable aggregation.

To characterize the local topological neighborhood of a node, we include RWSE as another PE that encodes the k-step random walk return probability for each node, represented by:

$$
\mathrm { R W S E } ( v _ { i } ) = [ ( A D ^ { - 1 } ) _ { i i } ^ { 1 } , ( A D ^ { - 1 } ) _ { i i } ^ { 2 } , \ldots , ( A D ^ { - 1 } ) _ { i i } ^ { k } ] \in \mathbb { R } ^ { k }\tag{4}
$$

where the t-th entry is the probability that a random walk starting from $v _ { i }$ returns to $v _ { i }$ after exactly t steps. RWSE is entirely local and invariant to node labeling [19], therefore it can be defined for any graph without ambiguity.

In the GAD setting, the core inductive signal is that anomalous nodes exhibit atypical feature patterns relative to their neighbors, as shown in Figure 1(b). To capture this featurelevel deviation from the local neighborhood, we adopted the node representation residual introduced in [31] as the third encoding component. Instead of capturing the residuals in the GNN-derived node embeddings, we measure the residual in the raw node features, mathematically defined as:

Algorithm 1: FoundAna: Training and Inference   
Input: A set of graphs $\{ \mathcal { G } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ , each with adjacency   
$\mathbf { A } ^ { ( m ) }$ and feature matrix $\bar { \mathbf { X } ^ { ( m ) } } \in \mathbb { R } ^ { n ^ { ( m ) } \times \breve { d } ^ { ( m ) } }$   
unified dimension $d _ { u } ;$ loss weights $\gamma _ { 1 } , \gamma _ { 2 } ;$ training   
epochs $T$   
Output: Anomaly scores $\left\{ s _ { i } \right\}$ for all nodes at inference   
1 for each graph $\check { g } ^ { ( m ) }$ do   
2 Compute truncated SVD: $\mathbf { X } ^ { ( m ) } \approx \mathbf { U } ^ { ( m ) } \pmb { \Sigma } ^ { ( m ) } ( \mathbf { V } ^ { ( m ) } ) ^ { \top } ;$   
3 Construct projection: $\mathbf { W } ^ { ( m ) } = \mathbf { V } _ { 1 : d _ { u } } ^ { ( m ) }$   
4 Obtain aligned features:   
$\begin{array} { r } { \tilde { \mathbf { X } } ^ { ( m ) } = \mathbf { X } ^ { ( m ) } \mathbf { W } ^ { ( m ) } \in \mathbb { R } ^ { n ^ { ( m ) } \times d _ { u } } ; } \end{array}$   
5 end   
6 for each node $v _ { i }$ do   
7 Compute $\mathrm { S P E } ( E _ { v } , \lambda )$ , RWSE(v<sub>i</sub>), residual $\mathbf { r } _ { i } ,$ and   
$\log ( 1 + \deg ( \upsilon _ { i } ) ) ;$   
8 Form descriptor:   
$\mathbf { p } _ { i } = [ \mathrm { S P E } ] | \mathrm { R W S E } | | \mathbf { r } _ { i } | | \log ( 1 + \deg ( v _ { i } ) ) ] ;$   
9 Form input token: $\mathbf { u } _ { i } = [ \tilde { \mathbf { x } } _ { i } \lVert \mathbf { p } _ { i } ] ;$   
10 end   
11 for each training epoch $t = 1 , \dots , T$ do   
12 Compute Beta-wavelet filters $\{ F _ { B } ^ { a , A - a } \} _ { a = 0 } ^ { A }$ from $L ;$   
13 Aggregate: $\begin{array} { r } { \mathbf { h } _ { i } ^ { \mathrm { G N N } } = \sum _ { a = 0 } ^ { A } F _ { B } ^ { a , A - a } \mathbf { H } ^ { ( a ) } ; } \end{array}$   
14 Apply multi-head self-attention over tokens $\{ \mathbf { u } _ { i } \} $   
15 Obtain: $\mathbf { h } _ { i } ^ { \mathrm { A t t n } }$ for all $v _ { i } ;$   
16 $\mathbf { z } _ { i } = [ \mathbf { h } _ { i } ^ { \mathrm { A t t i } } \lVert \mathbf { h } _ { i } ^ { \mathrm { G N N } } ] \in \mathbb { R } ^ { 2 d _ { h } } ;$   
17 Decode attributes: $\hat { \mathbf { x } } _ { i } = { \mathcal { D } } _ { X } ( \mathbf { z } _ { i } ) ;$   
18 Decode adjacency: $\hat { \mathbf { A } } _ { i j } = \mathcal { D } _ { A } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) = \sigma ( \mathbf { z } _ { i } ^ { \top } \mathbf { W } _ { A } \mathbf { z } _ { j } ) ;$   
19 Compute $\begin{array} { r } { \mathcal { L } _ { X } = \frac { 1 } { n } \sum _ { i } \Vert \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } \Vert _ { 2 } ^ { 2 } ; } \end{array}$   
20 Compute $\mathcal { L } _ { A } =$   
$\begin{array} { r } { - \frac { 1 } { n ^ { 2 } } \sum _ { i , j } \left[ { \bf A } _ { i j } \log \hat { \bf A } _ { i j } + ( 1 - { \bf A } _ { i j } ) \log ( 1 - \hat { \bf A } _ { i j } ) \right] ; } \end{array}$   
21 Update parameters by minimizing $\mathcal { L } = \gamma _ { 1 } \mathcal { L } _ { X } + \gamma _ { 2 } \bar { \mathcal { L } } _ { A }$   
22 end   
23 for each node $v _ { i }$ do   
24 $s _ { i } = \gamma _ { 1 } \cdot \Vert \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } \Vert _ { 2 } + \gamma _ { 2 } \cdot \sum _ { j \in \mathcal { N } ( i ) } \vert \mathbf { A } _ { i j } - \hat { \mathbf { A } } _ { i j } \vert ;$   
25 end   
26 if graph-level inference then   
27 $\begin{array} { r } { s _ { G } = \sum _ { i \in \mathcal { V } } \alpha _ { i } \cdot s _ { i } ; } \end{array}$   
28 return s<sub>G</sub>;   
29 else   
30 return $s _ { i } ;$   
31 end

$$
\mathbf { r } _ { i } = v _ { i } - \frac { 1 } { | \mathcal { N } ( v _ { i } ) | } \sum _ { v _ { j } \in \mathcal { N } ( v _ { i } ) } v _ { j }\tag{5}
$$

where $\mathcal { N } ( v _ { i } )$ denotes the neighbor set of $v _ { i } .$ . Anomaly nodes generally have large residuals as they differ structurally or feature-wise from their context, compared with normal nodes.

The degree of a node deg $\begin{array} { r } { \{ v _ { i } \} = \sum _ { j } \mathbf { A } _ { i j } } \end{array}$ provides a direct measure of local connectivity. As the node degree distribution is different for normal and anomalous nodes, as shown in Figure 1(a), we include the log-degree $\log ( 1 + \deg ( v _ { i } ) )$ as a scalar appended to the feature vector. These four encoding components are concatenated to form the final PE for each

node as follows:

$$
\mathrm { p } _ { i } = [ \mathrm { S P E } ( E _ { v } , \lambda ) | | \mathrm { R W S E } ( v _ { i } ) | | \mathbf { r } _ { i } | | \mathbf { l o g } ( 1 + \mathbf { d e g } ( v _ { i } ) ) ]\tag{6}
$$

where the dimension for ${ \mathfrak { p } } _ { i }$ is $\mathbb { R } ^ { \in \mathbb { R } ^ { \rho _ { \mathrm { S P E } } + k + d _ { u } + 1 } }$ that is the sum of all PE length.

## C. GNN-assisted Transformer

Inspired by GraphGPS [19] architecture, we design a transformer backbone coupled with an anomaly-aware Beta wavelet filter-based GNN. The idea is to integrate both the local and global graph structural information by leveraging a GNN and a transformer, respectively.

GNN: The GNN component captures the local structural information and high-frequency spectral patterns, a characteristic of an anomalous node [23]. Tang et al. [23] showed that the presence of anomalous nodes induces a rightward shift of graph-signal energy toward high Laplacian eigenvalues (described in Theorem 1), which suggests emphasizing spectrally localized band-pass filters over the traditional low-pass GNN (e.g., Graph Convolutional Network).

Theorem 1 (Right-shift Aware Locality [23]). Under the Gaussian anomaly model, the high-frequency area, $S _ { h i g h } ( x )$ increases monotonically with anomaly degree, and Beta wavelet filters of sufficiently high order, $C ,$ can concentrate their spectral response on any interval $[ \lambda _ { 1 } , \lambda _ { 2 } ] \subset ( 0 , 2 ]$ while remaining C-hop localized in the graph.

To model this spectral information, we adopt the Betawavelet graph filters from BWGNN as the default local GNN encoder. Formally, the band pass filters $F$ can be obtained as:

$$
F _ { B } ^ { ( \alpha , \beta ) } ( L ) = U F ( \Lambda ) U ^ { \top } = \frac { \left( \frac { L } { 2 } \right) ^ { \alpha } \left( I - \frac { L } { 2 } \right) ^ { \beta } } { 2 C ( \alpha + 1 , \beta + 1 ) } ,\tag{7}
$$

where $\alpha , \beta > 0$ controls the beta wavelet shape, $C ( \alpha { + } 1 , \beta { + } 1 )$ is a normalization constant, L denotes the normalized graph Laplacian with eigen-decomposition ${ \cal { L } } = U \Lambda U ^ { T }$ , and I is the identity matrix. During message passing, the SVD-projected node features $\mathbf { X } ^ { ( 0 ) }$ are filtered in parallel by all kernels in $F _ { B }$ and the outputs are aggregated features as follows:

$$
h _ { i } ^ { G N N } = \sum _ { a = 0 } ^ { A } F _ { B } ^ { a , A - a } H ^ { ( a ) } ,\tag{8}
$$

where $A = \alpha + \beta$ controls the maximum scale, and $h _ { i } ^ { G N N }$ is the graph encoding.

Transformer: Given the node tokens $u _ { i } .$ , constructed by concatenating the SVD-projected features and PE, we apply a stack of standard transformer encoder layers with multihead self-attention (shown in Eq. (1)) and position-wise feedforward networks. In contrast to the GNN, the transformer’s global attention layers were trained without providing explicit edge features in the attention kernel but we relied on PE to include the topological information. Therefore, the transformer backbone produces node embeddings $h _ { i } ^ { A t t n }$ , which capture the global, long-range dependencies across the graph [19], [35].

Finally, we fuse the local and global information by concatenating the outputs of the GNN and transformer backbone at the node level, which produces the new features:

$$
\mathbf { z } _ { i } = [ \mathbf { h } _ { i } ^ { \mathrm { A t t n } } | | \mathbf { h } _ { i } ^ { \mathrm { G N N } } ] \in \mathbb { R } ^ { 2 d _ { h } } ,\tag{9}
$$

where $\mathbf { z } _ { i }$ is the combined node embedding representation.

## D. Reconstruction and Anomaly Score Computation

We employ separate decoders to reconstruct both node attributes and the graph adjacency structure from $\mathbf { z } _ { i } .$ For the attribute decoder $\mathcal { D } _ { \mathcal { X } }$ , we use a two-layer Multi Layer Perceptron that maps $\mathbf { z } _ { i }$ to the original feature space:

$$
\hat { \mathbf { x } } _ { i } = { \mathcal { D } } _ { X } \big ( \mathbf { z } _ { i } \big ) = \sigma \big ( \mathbf { W } _ { 2 } \cdot \mathrm { R e L U } ( \mathbf { W } _ { 1 } \mathbf { z } _ { i } + \mathbf { b } _ { 1 } ) + \mathbf { b } _ { 2 } \big )\tag{10}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d _ { h } \times 2 d _ { h } }$ compresses the fused representation to a bottleneck, and $\mathbf { W } _ { 2 } ~ \in ~ \mathbb { R } ^ { d ^ { ( m ) } \times d _ { h } }$ maps back to the original feature dimensionality $d ^ { ( m ) }$ of graph $\mathcal { G } ^ { ( m ) }$ . Similarly, since $\mathbf { z } _ { i }$ consists of PE concatenated with node features, the latent representations of two connected nodes $v _ { i }$ and $v _ { j }$ implicitly encode their structural relationship. Edge existence can therefore be decoded via a bilinear decoder $\mathcal { D } _ { A }$

$$
\hat { \mathbf { A } } _ { i j } = \mathcal { D } _ { A } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) = \sigma ( \mathbf { z } _ { i } ^ { \top } \mathbf { W } _ { A } \mathbf { z } _ { j } )\tag{11}
$$

where $\mathbf { W } _ { A } \in \mathbb { R } ^ { 2 d _ { h } \times 2 d _ { h } }$ is a learnable bilinear weight matrix. The overall reconstruction loss is defined as:

$$
\mathcal { L } = \gamma _ { 1 } \mathcal { L } _ { X } + \gamma _ { 2 } \mathcal { L } _ { A } ,\tag{12}
$$

where $\gamma _ { 1 } , \gamma _ { 2 } \geq 0$ balance the two reconstruction objectives, and the feature and structure reconstruction losses are respectively given by:

$$
\mathcal { L } _ { X } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Vert \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } \Vert _ { 2 } ^ { 2 } , a n d\tag{13}
$$

$$
\mathcal { L } _ { A } = - \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \left[ \mathbf { A } _ { i j } \log \hat { \mathbf { A } } _ { i j } + ( 1 - \mathbf { A } _ { i j } ) \log ( 1 - \hat { \mathbf { A } } _ { i j } ) \right] .\tag{14}
$$

In inference, a node is considered anomalous if it cannot be reconstructed in either attribute or structural space. We therefore define a composite anomaly score for node $v _ { i }$ in Eq. (15). For graph-level anomaly detection, a readout layer aggregates node attributes and adjacency reconstruction errors through attention-weighted pooling, which produces a single graph-level anomaly score that reflects the collective structural and feature deviation across the graph in Eq. (16).

$$
s _ { i } = \gamma _ { 1 } \cdot \Vert \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { i } \Vert _ { 2 } + \gamma _ { 2 } \cdot \sum _ { j \in \mathcal { N } ( i ) } \left. \mathbf { A } _ { i j } - \hat { \mathbf { A } } _ { i j } \right.\tag{15}
$$

$$
s _ { G } = \mathrm { R e a d o u t } \left( \left\{ s _ { i } \right\} _ { i \in \mathcal { V } } \right) = \sum _ { i \in \mathcal { V } } \alpha _ { i } \cdot s _ { i }\tag{16}
$$

## V. EXPERIMENTS

We validate the proposed FoundAna framework through experiments addressing the following Research Questions (RQs):

• RQ1: Does the proposed framework outperform baselines across evaluated datasets?

• RQ2: Does the model generalize to unseen graph data under zero-shot and few-shot scenarios?

• RQ3: Does concatenating PE with raw features improve GAD performance over using raw features alone?

## A. Datasets

We evaluate on 9 GAD benchmark datasets summarized in Table I, which span social networks, financial and e-commerce networks, and academic citation graphs.

Social networks: The Question dataset is derived from Community Question Answering (CQA) platforms and represents a bipartite user–question graph, where anomalies correspond to malicious users or low-quality automated content [20]. Weibo captures user–post interactions from a Chinese microblog platform, commonly used to detect Sybil accounts and botdriven rumor propagation. Facebook maps ego-network social connections and profile features, originally introduced for social circle discovery [20]. Reddit captures cross-subreddit user interactions, with anomalies defined by outlier engagement patterns deviating from the regular user base [20].

Financial and e-commerce networks: YelpChi is a multirelational review graph from the Chicago area, where fraudulent reviewers are labeled as anomalies via filtering heuristics [20]. Amazon is an e-commerce platform targeting fraud ulent reviewers who artificially inflate product ratings [20]. T-Finance is a transaction network designed for detecting fraudulent loan accounts with atypical topological patterns [23]. DGraph is a massive real-world fintech dataset with millions of user nodes and financial or social ties, specifically targeting fraudulent loan applications [20]. Elliptic is a large-scale Bitcoin transaction graph where illicit transactions are identified by association with known criminal entities [20].

Citation network: Ogbn-arxiv is a directed Computer Science (CS) paper citation network, where anomalies represent structural outliers or misclassified papers within the citation hierarchy [20].

1) Node and Edge Duality Assumption: Since none of these datasets consists of the edge features, we consider a common anomaly prediction for node and edge-based tasks. To support this assumption, we use a graph-theoretic perspective, where node and edge anomalies are unified under a common representational framework through the concept of the line graph transformation (Definition 6).

Definition 6 (Line Graph Transformation [36], [37]). Given a graph G, its line graph $L ( { \mathcal { G } } )$ is constructed such that each edge $e \in { \mathcal { E } }$ in the original graph becomes a node $\boldsymbol { v } ^ { \prime } \in \mathbf { V } ^ { \prime }$ in $L ( { \mathcal { G } } )$ , with two nodes in $L ( { \mathcal { G } } )$ connected if their corresponding edges in $\mathcal { G }$ share a common endpoint. This makes detecting anomalous edges in $\mathcal { G }$ mathematically equivalent to detecting anomalous nodes in L(G), which establishes a formal duality between node-level and edge-level GAD.

TABLE I  
DESCRIPTION OF THE DATASET. T-FINANCE IS PRIMARILY USED FOR TRAINING, WHEREAS THE REST OF THE DATASET IS USED FOR TESTING.
<table><tr><td>Dataset</td><td>Domain</td><td>Anomaly percentage</td><td>Dimension</td><td>Nodes</td><td>Edges</td><td>Normal degree</td><td>Anomaly degree</td></tr><tr><td>T-Finance</td><td>Finance</td><td>4.580</td><td>10</td><td>39,357</td><td>42,445,086</td><td>1098.960</td><td>651.460</td></tr><tr><td>Question</td><td>Q&amp;A platform</td><td>2.980</td><td>301</td><td>48,921</td><td>202,461</td><td>3.960</td><td>9.840</td></tr><tr><td>YelpChi</td><td>Reviews</td><td>5.110</td><td>32</td><td>23,831</td><td>98,630</td><td>4.280</td><td>1.590</td></tr><tr><td>Weibo</td><td>Social media</td><td>10.330</td><td>400</td><td>8,405</td><td>754,542</td><td>95.240</td><td>42.340</td></tr><tr><td>Dgraphfin</td><td>Finance</td><td>67.300</td><td>17</td><td>3,700,550</td><td>4,300,999</td><td>1.730</td><td>0.890</td></tr><tr><td>Facebook</td><td>Social network</td><td>2.310</td><td>128</td><td>1,081</td><td>55,104</td><td>52.060</td><td>5.280</td></tr><tr><td>Amazon</td><td>Reviews</td><td>6.780</td><td>25</td><td>10,224</td><td>351216</td><td>32.590</td><td>58.590</td></tr><tr><td>Ogbn-arixv</td><td>Citation network</td><td>3.540</td><td>128</td><td>169,343</td><td>2,357,596</td><td>13.680</td><td>20.500</td></tr><tr><td>Elliptic</td><td>Finance</td><td>9.760</td><td>165</td><td>46,564</td><td>73,248</td><td>1.660</td><td>0.810</td></tr><tr><td>Reddit</td><td>Social media</td><td>3.330</td><td>64</td><td>10,984</td><td>168,016</td><td>15.400</td><td>12.380</td></tr></table>

## B. Baselines and Metrics

We evaluate the proposed FoundAna against two categories of state-of-the-art methods: GNN-based GAD and GFM-based GAD. The GNN-based baselines include GCN [38], BWGNN [23], DOMINANT [39], AnomalyDAE [40], and CoLA [22]. These methods have various design choices, ranging from lowpass spectral filters and graph autoencoders to contrastive selfsupervised learning, representing the landscape of GAD. The GFM baselines include ARC [24], AnomalyGFM [31], and UNPrompt [25], each of which aims at cross-graph generalization under different pretraining and prompting approaches.

Since GAD datasets are inherently class-imbalanced, we select three complementary evaluation metrics: AUROC, which measures ranking quality independent of threshold; AUPRC, which is sensitive to performance on the minority anomaly class; and Macro F1, which evaluates per-class balance after thresholding. Also, to assess the statistical significance of the performance, we employ the Wilcoxon signed-rank test [41], a non-parametric statistical test that evaluates whether the differences between paired observations are systematically different from zero without assuming a normal distribution of the data. We consider the performance metrics on individual data as statistically significant results when $p < 0 . 0 5$

## C. Implementation Details

All experiments are implemented in Python v3.10.20, leveraging PyTorch v2.4.0, PyTorch Geometric v2.6.0, PyTorch Scatter v2.1.2, and scikit-learn v1.7.2. Training and inference for both the proposed model and all baselines were conducted on a single NVIDIA L40S GPU with 48 GB of memory. Since most of the baselines used a single dataset to train the models, and T-Finance being the widely adopted large-scale benchmark for anomaly detection in the GAD literature [23], [31], we selected T-Finance to train the model and evaluated zero-shot and few-shot on 9 anomaly detection datasets. Also, it comprises a huge network of connections with real-world fraud annotations. The weights are frozen when inferring in a zero-shot setting, whereas they are fine-tuned when inferring in a few-shot setting.

The final configuration uses 8 attention heads, a batch size of 2,048, and 100 training epochs with a learning rate of $1 \times 1 0 ^ { - 3 }$ . Feature unification via SVD projects all node attributes to a common dimension of 32. For PEs, SPE uses the 8 non-trivial Laplacian eigenpairs, and RWSE is computed with 16 random walk steps. The reconstruction loss is a weighted combination of the feature reconstruction and adjacency reconstruction terms, with loss weights $\gamma _ { 1 } = \gamma _ { 2 } = 0 . 5 .$ which gives equal importance assigned to attribute fidelity and structural reconstruction. For the few-shot inference, we use 20 data samples to train the model for 30 epochs.

## D. Results of FoundAna

Table II illustrates the performance comparison of the baseline and FoundAna on the nine benchmark datasets under three evaluation metrics. The rightmost columns report the average performance across all datasets and the p-value from the Wilcoxon signed-rank test to assess statistical significance. The bold values denote the best performance per dataset, and underlined values indicate the second best. Out of Memory (OOM) entries refer to methods that failed due to memory constraints, which generally occur on larger datasets such as Dgraphfin and Ogbn-arxiv of some baselines, indicating their intensive computation cost.

Generally, our proposed method achieves the highest average performance across all three metrics under the few-shot inference with AUROC, AUPRC, and Macro F1 scores of 0.647, 0.227, and 0.568, respectively. Also, our method has a clear superiority over the GNN-based baselines. For instance, our method has higher performance than the best AUROC and Macro F1 of 0.569 and 0.494, respectively, achieved in Dominant and AUPRC of 0.150 in BWGNN. Similarly, we observe comparable or superior results on the GFM baselines. A higher performance on all three metrics is observed on the four datasets (i.e., Question, YelpChi, Amazon, and Reddit) under a few-shot learning setting. Moreover, the highest gains are observed on YelpChi (AUROC 0.915, Macro F1 0.693), Weibo (Macro F1 0.740), and Dgraphfin (AUPRC 0.682), which support an affirmative answer to RQ1.

Also, FoundAna-zero shot already achieves competitive generalization with an average AUROC of 0.525, AUPRC of 0.176, and Macro F1 of 0.511, frequently surpassing fully supervised GNN baselines without any labeled data at test time. This supports that FoundAna generalizes to unseen graph data under zero-shot and few-shot scenarios (RQ2). The performance superiority is statistically significant for all GNNbased baselines, except BWGNN for AUPRC (i.e., p-value of 0.300); however, for the GFM baseline, only the Macro F1 score on ARC is statistically significant.

TABLE II  
PERFORMANCE COMPARISON ACROSS DATASETS USING AUROC, AUPRC, AND MACRO F1 SCORES. THE NUMBERS INDICATE RESULTS WHEN TRAINING ONLY ON THE T-FINANCE DATASET. AVG. IS THE AVERAGE PERFORMANCE ON ALL NINE BENCHMARK DATASETS. BOLD INDICATES THE BEST RESULT PER COLUMN; UNDERLINE INDICATES THE SECOND BEST.
<table><tr><td rowspan="2">Metrics</td><td rowspan="2">Methods</td><td colspan="9">Dataset</td><td rowspan="2">Avg.</td><td rowspan="2">p-value</td></tr><tr><td>Question</td><td>YelpChi</td><td>Weibo</td><td>Dgraphfin</td><td>Facebook</td><td>Amazon</td><td>Ogbn-arxiv</td><td>Elliptic</td><td>Reddit</td></tr><tr><td rowspan="9">AUROC</td><td>GCN</td><td>0.434</td><td>0.592</td><td>0.227</td><td>0.434</td><td>0.406</td><td>0.447</td><td>0.421</td><td>0.442</td><td>0.453</td><td>0.431</td><td>0.003</td></tr><tr><td>BWGNN</td><td>0.492</td><td>0.642</td><td>0.204</td><td>0.505</td><td>0.401</td><td>0.602</td><td>0.456</td><td>0.462</td><td>0.506</td><td>0.474</td><td>0.003</td></tr><tr><td>Dominant</td><td>0.533</td><td>0.717</td><td>0.767</td><td>00M</td><td>0.576</td><td>0.585</td><td>O0M</td><td>0.307</td><td>0.503</td><td>0.569</td><td>0.031</td></tr><tr><td>AnomalyDAE</td><td>0.519</td><td>0.404</td><td>0.499</td><td>O0M</td><td>0.023</td><td>0.397</td><td>0.425</td><td>0.448</td><td>0.421</td><td>0.392</td><td>0.007</td></tr><tr><td>CoLA</td><td>0.489</td><td>0.400</td><td>0.388</td><td>0.555</td><td>0.515</td><td>0.513</td><td>0.564</td><td>0.501</td><td>0.490</td><td>0.490</td><td>0.003</td></tr><tr><td>ARC</td><td>0.572</td><td>0.488</td><td>0.877</td><td>0.498</td><td>0.643</td><td>0.609</td><td>0.814</td><td>0.257</td><td>0.571</td><td>0.584</td><td>0.570</td></tr><tr><td>AnomalyGFM</td><td>0.546</td><td>0.524</td><td>0.556</td><td>00M</td><td>0.733</td><td>0.511</td><td>00M</td><td>0.606</td><td>0.533</td><td>0.572</td><td>0.375</td></tr><tr><td>UNPrompt</td><td>0.468</td><td>0.522</td><td>0.854</td><td>0.419</td><td>0.732</td><td>0.661</td><td>0.759</td><td>0.262</td><td>0.534</td><td>0.568</td><td>0.359</td></tr><tr><td>FoundAna-zero shot</td><td>0.477</td><td>0.566</td><td>0.749</td><td>0.364</td><td>0.483</td><td>0.521</td><td>0.589</td><td>0.466</td><td>0.512</td><td>0.525</td><td></td></tr><tr><td rowspan="10">AURC</td><td>FoundAna-few shot</td><td>0.585</td><td>0.915</td><td>0.801</td><td>0.560</td><td>0.546</td><td>0.695</td><td>0.689</td><td>0.508</td><td>0.586</td><td>0.647</td><td>-</td></tr><tr><td>GCN</td><td>0.024</td><td>0.087</td><td>0.236</td><td>0.598</td><td>0.019</td><td>0.068</td><td>0.028</td><td>0.023</td><td>0.035</td><td>0.124</td><td>0.003</td></tr><tr><td>BWGNN</td><td>0.027</td><td>0.098</td><td>0.064</td><td>0.780</td><td>0.097</td><td>0.101</td><td>0.030</td><td>0.120</td><td>0.038</td><td>0.150</td><td>0.300</td></tr><tr><td>Dominant</td><td>0.032</td><td>0.166</td><td>0.481</td><td>OOM</td><td>0.108</td><td>0.083</td><td>00M</td><td>0.064</td><td>0.035</td><td>0.138</td><td>0.109</td></tr><tr><td>AnomalyDAE</td><td>0.031</td><td>0.040</td><td>0.130</td><td>00M</td><td>0.012</td><td>0.063</td><td>0.029</td><td>0.083</td><td>0.028</td><td>0.052</td><td>0.007</td></tr><tr><td>CoLA</td><td>0.033</td><td>0.048</td><td>0.231</td><td>0.715</td><td>0.033</td><td>0.077</td><td>0.052</td><td>0.091</td><td>0.034</td><td>0.146</td><td>0.054</td></tr><tr><td>ARC</td><td>0.039</td><td>0.052</td><td>0.618</td><td>0.673</td><td>0.098</td><td>0.090</td><td>0.285</td><td>0.067</td><td>0.040</td><td>0.218</td><td>0.652</td></tr><tr><td>AnomalyGFM</td><td>0.034</td><td>0.060</td><td>0.113</td><td>00M</td><td>0.115</td><td>0.063</td><td>00M</td><td>0.127</td><td>0.034</td><td>0.078</td><td>0.156</td></tr><tr><td>UNPrompt</td><td>0.031</td><td>0.058</td><td>0.491</td><td>0.608</td><td>0.113</td><td>0.106</td><td>0.247</td><td>0.062</td><td>0.036</td><td>0.177</td><td>0.250</td></tr><tr><td>FoundAna-zero shot FoundAna-few shot</td><td>0.028 0.041</td><td>0.146 0.324</td><td>0.557 0.601</td><td>0.586 0.682</td><td>0.042 0.048</td><td>0.068 0.122</td><td>0.042 0.050</td><td>0.083 0.132</td><td>0.037 0.044</td><td>0.176 0.227</td><td></td></tr><tr><td rowspan="10">MMcF1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GCN</td><td>0.455</td><td>0.490</td><td>0.548</td><td>0.441</td><td>0.445</td><td>0.480</td><td>0.436</td><td>0.525</td><td>0.479</td><td>0.476</td><td>0.003</td></tr><tr><td>BWGNN</td><td>0.488</td><td>0.529</td><td>0.474</td><td>0.427</td><td>0.495</td><td>0.504</td><td>0.485</td><td>0.478</td><td>0.502</td><td>0.486</td><td>0.003</td></tr><tr><td>Dominant</td><td>0.490</td><td>0.486</td><td>0.551</td><td>O0M</td><td>0.493</td><td>0.504</td><td>00M</td><td>0.476</td><td>0.459</td><td>0.494</td><td>0.015</td></tr><tr><td>AnomalyDAE CoLA</td><td>0.455 0.472</td><td>0.480</td><td>0.088</td><td>00M</td><td>0.494</td><td>0.467</td><td>0.148</td><td>0.474</td><td>0.475</td><td>0.385</td><td>0.007</td></tr><tr><td></td><td></td><td>0.451</td><td>0.517</td><td>0.450</td><td>0.484</td><td>0.484</td><td>0.472</td><td>0.479</td><td>0.481</td><td>0.471</td><td>0.003</td></tr><tr><td>ARC</td><td>0.497</td><td>0.486</td><td>0.095</td><td>0.496</td><td>0.356</td><td>0.482</td><td>0.491</td><td>0.258</td><td>0.491</td><td>0.368</td><td>0.007</td></tr><tr><td>AnomalyGFM</td><td>0.403</td><td>0.331</td><td>0.413</td><td>OOM</td><td>0.519</td><td>0.347</td><td>00M</td><td>0.580</td><td>0.260</td><td>0.407</td><td>0.078</td></tr><tr><td>UNPrompt</td><td>0.520</td><td>0.514</td><td>0.722</td><td>0.494</td><td>0.592</td><td>0.537</td><td>0.653</td><td>0.474</td><td>0.503</td><td>0.550</td><td>0.425</td></tr><tr><td>FoundAna-zero shot FoundAna-few shot</td><td>0.497 0.533</td><td>0.540 0.693</td><td>0.728 0.740</td><td>0.427 0.499</td><td>0.488 0.508</td><td>0.486 0.583</td><td>0.490 0.502</td><td>0.454 0.528</td><td>0.496 0.532</td><td>0.511 0.568</td><td></td></tr></table>

Figure 3 illustrates the few-shot performance of FoundAna as a function of the number of few-shot examples k ∈ {1, 3, 5, 10, 15, 20} across nine datasets under AUROC, AUPRC, and Macro F1. Weibo consistently achieves the strongest and most stable performance across all k values (e.g., AUROC ≈ 0.790–0.810), while YelpChi exhibits a clear upward trend, peaking at AUROC ≈ 0.840 and Macro F1 ≈ 0.60 at k=15. Dgraphfin achieves notably high AUPRC values peaking near 0.70 at k=5, whereas the remaining datasets cluster in lower performance ranges.

## E. Ablation Study

In this section, we discuss the contributions of various components of the proposed framework, specifically the effect of different PEs used, the performance comparison across three GNN backbones, and the performance when using different types of features in the foundation model.

1) Effect of PEs: Table III ablates the contribution of each PE component on the YelpChi dataset (results on other datasets are in the supplementary material). The full combination of all PEs achieves the best performance across AUROC (0.804), AUPRC (0.165), and Macro F1 (0.581). Removing RWSE causes the smallest drop, whereas removing RD and ND leads to the sharpest decline of AUROC to 0.612 and 0.660, respectively. This suggests that RWSE carries partially redundant information with other PEs, while RD and NE are important for distinguishing anomalous nodes. Also, we observed poor performance when none of the PEs were used.

2) Effect of GNN Backbone Selection: Table IV compares three GNN backbone choices on the Ogbn-arxiv dataset (results on other datasets are in the supplementary material).

![](images/5536dd519e0c8e28369fc3718faada1bfa220ab4a34896573a72704c22982ee1.jpg)  
Fig. 3. Performance results on nine datasets under different values of k in few-shot learning settings.

## TABLE III

RESULTS WHEN USING DIFFERENT COMBINATIONS OF PE. THE REPORTED   
RESULTS ARE ON THE YELPCHI DATASET (OTHER DATASETS ARE IN THE   
SUPPLEMENTARY MATERIAL). PE: POSITIONAL ENCODING; SPE: STABLE AND EXPRESSIVE POSITIONAL ENCODING; RWSE: RANDOM WALK STRUCTURAL ENCODING; RD: RESIDUAL DISTANCE; ND: NODE DEGREE.

<table><tr><td>PE</td><td>AUROC</td><td>AUPRC</td><td>Macro F1</td><td>Training time</td></tr><tr><td>No PE</td><td>0.634</td><td>0.088</td><td>0.537</td><td>0.160</td></tr><tr><td>w/o SPE</td><td>0.745</td><td>0.118</td><td>0.546</td><td>0.230</td></tr><tr><td>w/o RWSE</td><td>0.764</td><td>0.149</td><td>0.580</td><td>0.220</td></tr><tr><td>w/o RD</td><td>0.612</td><td>0.085</td><td>0.536</td><td>0.210</td></tr><tr><td>w/o ND</td><td>0.660</td><td>0.082</td><td>0.526</td><td>0.230</td></tr><tr><td>w/ all PE</td><td>0.804</td><td>0.165</td><td>0.581</td><td>0.220</td></tr></table>

TABLE IV

PERFORMANCE RESULTS ON DIFFERENT GNN BACKBONES WHEN TESTED ON THE OGBN-ARIXV DATASET. T: TRAINING; I: INFERENCE.
<table><tr><td>Model</td><td>AUROC</td><td>AUPRC</td><td>Macro F1</td><td>T/I time</td></tr><tr><td>GCN</td><td>0.419</td><td>0.028</td><td>0.491</td><td>318.100/3.350</td></tr><tr><td>GIN</td><td>0.301</td><td>0.024</td><td>0.489</td><td>304.800/3.140</td></tr><tr><td>BWGNN</td><td>0.546</td><td>0.037</td><td>0.496</td><td>465.100/4.870</td></tr><tr><td>w/o GNN</td><td>0.334</td><td>0.039</td><td>0.386</td><td>305.300/ 2.300</td></tr></table>

BWGNN achieves the best overall performance on AUROC (0.546) and Macro F1 (0.496), which is consistent with prior work showing that BWGNN’s spectral filtering is particularly effective at capturing anomalous patterns [23], [42]. GCN and Graph Isomorphism Network (GIN) perform comparably but fall short of BWGNN, whereas the best AUPRC (0.039) is observed when completely removing the GNN backbone. This indicates that general-purpose message-passing backbones are suboptimal for anomaly-specific tasks in our framework.

3) Effect of PE-Augmented Input Features on GAD: Table V evaluates whether augmenting raw features with PE benefits BWGNN across three representative datasets from each dataset group: social networks (Weibo), financial and e-commerce (YelpChi), and citation (Ogbn-arxiv) networks.

## TABLE V

PERFORMANCE RESULTS WHEN INPUT IS RAW FEATURES AND PE CONCATENATED FEATURES (FULL RESULTS ARE IN SUPPLEMENTARY MATERIAL). PE: POSITIONAL ENCODING; RF: RAW FEATURES; T: TRAINING; I: INFERENCE.

<table><tr><td>Model</td><td>AUROC</td><td>AUPRC</td><td>Macro F1</td><td>T/I time</td></tr><tr><td colspan="5">Weibo</td></tr><tr><td>BWGNN (RF)</td><td>0.758</td><td>0.562</td><td>0.730</td><td>760.9/ 0.410</td></tr><tr><td>BWGNN (RF + PE)</td><td>0.752</td><td>0.564</td><td>0.732</td><td>769.1/ 0.440</td></tr><tr><td colspan="5">YelpChi</td></tr><tr><td>BWGNN (RF)</td><td>0.563</td><td>0.067</td><td>0.521</td><td>760.9/ 1.220</td></tr><tr><td>BWGNN (RF + PE)</td><td>0.666</td><td>0.082</td><td>0.526</td><td>769.1/1.230</td></tr><tr><td colspan="5">Ogbn Arxiv</td></tr><tr><td>BWGNN (RF)</td><td>0.496</td><td>0.035</td><td>0.497</td><td>760.9/10.690</td></tr><tr><td>BWGNN (RF + PE)</td><td>0.581</td><td>0.039</td><td>0.495</td><td>769.1/10.630</td></tr></table>

For this experiment, we select only BWGNN as it performs the best compared to all other GNN models, as presented in Table IV.

We observe a dataset-dependent pattern: on YelpChi and Ogbn-arxiv, the raw feature combined with PE consistently outperforms the raw feature alone, with an AUROC gain of 0.103 and 0.085 on YelpChi and Ogbn-arxiv, respectively. However, on Weibo, raw feature slightly outperforms the case of PE combination, with an AUROC of 0.006, and marginally achieves better on AUPRC and Macro F1. This indicates that in the dense social networks, raw feathers already capture sufficient discriminative signals and PE only contributes marginally, which is consistent in other datasets as well. These results provide an answer to our RQ3: PE-augmented features generally improve AD, though the magnitude of improvement is domain-dependent.

## VI. CONCLUSION

In this work, we investigated the graph anomaly detection problem through the lens of foundation models. Specifically, we introduced FoundAna, which integrates positional encoding with anomaly-specific GNNs to produce structural representations that generalize across diverse graph domains. The experimental results demonstrate that FoundAna consistently achieves better average results across 9 GAD benchmarks, validating its effectiveness as a graph-agnostic framework. Moreover, we have conducted extensive ablation studies on PE combinations and GNN backbone selection, revealing the critical role of PE in capturing anomalous patterns across heterogeneous graph domains. Nevertheless, the applicability of FoundAna to knowledge graphs remains an open challenge, representing a promising direction for future work in extending foundational GAD frameworks to more complex, heterogeneous graph structures.

## REFERENCES

[1] F. E. Grubbs, “Procedures for detecting outlying observations in samples,” Technometrics, vol. 11, no. 1, pp. 1–21, 1969.

[2] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” ACM computing surveys (CSUR), vol. 41, no. 3, pp. 1–58, 2009.

[3] M. Munir, S. A. Siddiqui, A. Dengel, and S. Ahmed, “Deepant: A deep learning approach for unsupervised anomaly detection in time series,” IEEE access, vol. 7, pp. 1991–2005, 2018.

[4] M. Munir, M. A. Chattha, A. Dengel, and S. Ahmed, “A comparative analysis of traditional and deep learning-based anomaly detection methods for streaming data,” in 2019 18th IEEE international conference on machine learning and applications (ICMLA). IEEE, 2019, pp. 561–566.

[5] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv preprint arXiv:1901.03407, 2019.

[6] X. Ma, J. Wu, S. Xue, J. Yang, C. Zhou, Q. Z. Sheng, H. Xiong, and L. Akoglu, “A comprehensive survey on graph anomaly detection with deep learning,” IEEE transactions on knowledge and data engineering, vol. 35, no. 12, pp. 12 012–12 038, 2021.

[7] J. Wang and I. C. Paschalidis, “Botnet detection based on anomaly and community detection,” IEEE Transactions on Control of Network Systems, vol. 4, no. 2, pp. 392–404, 2016.

[8] R. M. Zakaria, M. M. Rahman, H. Rahman, M. A. Rafi, A. Minto, M. S. Hossain, S. I. Saimon et al., “Detecting financial fraud in realtime transactions using graph neural networks and anomaly detection techniques,” Journal of Economics, Finance and Accounting Studies, vol. 7, no. 6, pp. 01–13, 2025.

[9] T. Wang and Z.-P. Liu, “Anomaly detection with graph embedding in bioinformatics: A survey,” Current Genomics, 2026.

[10] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.

[11] C. Zhou, Q. Li, C. Li, J. Yu, Y. Liu, G. Wang, K. Zhang, C. Ji, Q. Yan, L. He et al., “A comprehensive survey on pretrained foundation models: A history from bert to chatgpt,” International Journal of Machine Learning and Cybernetics, vol. 16, no. 12, pp. 9851–9915, 2025.

[12] M. Awais, M. Naseer, S. Khan, R. M. Anwer, H. Cholakkal, M. Shah, M.-H. Yang, and F. S. Khan, “Foundation models defining a new era in vision: a survey and outlook,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2245–2264, 2025.

[13] J. Liu, C. Yang, Z. Lu, J. Chen, Y. Li, M. Zhang, T. Bai, Y. Fang, L. Sun, P. S. Yu et al., “Graph foundation models: Concepts, opportunities and challenges,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[14] Z. Wang, Z. Liu, T. Ma, J. Li, Z. Zhang, X. Fu, Y. Li, Z. Yuan, W. Song, Y. Ma et al., “Graph foundation models: A comprehensive survey,” arXiv preprint arXiv:2505.15116, 2025.

[15] J. Liu, C. Yang, Z. Lu, J. Chen, Y. Li, M. Zhang, T. Bai, Y. Fang, L. Sun, P. S. Yu et al., “Towards graph foundation models: A survey and beyond,” arXiv preprint arXiv:2310.11829, 2023.

[16] J. Zhang, H. Zhang, C. Xia, and L. Sun, “Graph-bert: Only attention is needed for learning graph representations,” arXiv preprint arXiv:2001.05140, 2020.

[17] L. Xia, B. Kao, and C. Huang, “Opengraph: Towards open graph foundation models,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 2365–2379.

[18] Y. Huang, W. Lu, J. Robinson, Y. Yang, M. Zhang, S. Jegelka, and P. Li, “On the stability of expressive positional encodings for graphs,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 39 745–39 774.

[19] L. Rampa´sek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, andˇ D. Beaini, “Recipe for a general, powerful, scalable graph transformer,” Advances in Neural Information Processing Systems, vol. 35, pp. 14 501–14 515, 2022.

[20] H. Qiao, H. Tong, B. An, I. King, C. Aggarwal, and G. Pang, “Deep graph anomaly detection: A survey and new perspectives,” IEEE Transactions on Knowledge and Data Engineering, 2025.

[21] A. Roy, J. Shu, J. Li, C. Yang, O. Elshocht, J. Smeets, and P. Li, “Gad-nr: Graph anomaly detection via neighborhood reconstruction,” in Proceedings of the 17th ACM international conference on web search and data mining, 2024, pp. 576–585.

[22] Y. Liu, Z. Li, S. Pan, C. Gong, C. Zhou, and G. Karypis, “Anomaly detection on attributed networks via contrastive self-supervised learning,” IEEE transactions on neural networks and learning systems, vol. 33, no. 6, pp. 2378–2392, 2021.

[23] J. Tang, J. Li, Z. Gao, and J. Li, “Rethinking graph neural networks for anomaly detection,” in International conference on machine learning. PMLR, 2022, pp. 21 076–21 089.

[24] Y. Liu, S. Li, Y. Zheng, Q. Chen, C. Zhang, and S. Pan, “Arc: A generalist graph anomaly detector with in-context learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 50 772–50 804, 2024.

[25] C. Niu, H. Qiao, C. Chen, L. Chen, and G. Pang, “Zero-shot generalist graph anomaly detection with unified neighborhood prompts,” arXiv preprint arXiv:2410.14886, 2024.

[26] H. Liu, J. Feng, L. Kong, N. Liang, D. Tao, Y. Chen, and M. Zhang, “One for all: Towards training one graph model for all classification tasks,” in International conference on learning representations, vol. 2024, 2024, pp. 20 188–20 210.

[27] Z. Wang, Z. Zhang, N. V. Chawla, C. Zhang, and Y. Ye, “Gft: Graph foundation model with transferable tree vocabulary,” Advances in neural information processing systems, vol. 37, pp. 107 403–107 443, 2024.

[28] Y. Lin, J. Tang, C. Zi, H. V. Zhao, Y. Yao, and J. Li, “Unigad: Unifying multi-level graph anomaly detection,” Advances in neural information processing systems, vol. 37, pp. 136 120–136 148, 2024.

[29] C. Song, X. Lin, H. Shen, Y. Shang, and Y. Cao, “Uniform: Towards unified framework for anomaly detection on graphs,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 12, 2025, pp. 12 559–12 567.

[30] R. Han, X. Wang, L. Wang, W. Zhang, G. Yao, and H. Liang, “A graph foundation model for unified anomaly detection,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 523–534.

[31] H. Qiao, C. Niu, L. Chen, and G. Pang, “Anomalygfm: Graph foundation model for zero/few-shot anomaly detection,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 2326–2337.

[32] Y. Liu, T. He, Y. Liu, C. Yi, H. Jin, and C. Hong, “Tabular foundation models are strong graph anomaly detectors,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 8785–8788.

[33] C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T.-Y. Liu, “Do transformers really perform badly for graph representation?” Advances in neural information processing systems, vol. 34, pp. 28 877– 28 888, 2021.

[34] L. Ma, C. Lin, D. Lim, A. Romero-Soriano, P. K. Dokania, M. Coates, P. Torr, and S.-N. Lim, “Graph inductive biases in transformers without message passing,” in International Conference on Machine Learning. PMLR, 2023, pp. 23 321–23 337.

[35] E. Min, R. Chen, Y. Bian, T. Xu, K. Zhao, W. Huang, P. Zhao, J. Huang, S. Ananiadou, and Y. Rong, “Transformer for graphs: An overview from architecture perspective,” arXiv preprint arXiv:2202.08455, 2022.

[36] H. Whitney, “Congruent graphs and the connectivity of graphs,” American Journal of Mathematics, vol. 54, no. 1, p. 150, 1932.

[37] J. A. Bondy and U. S. R. Murty, Graph theory. Springer Publishing Company, Incorporated, 2008.

[38] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[39] K. Ding, J. Li, R. Bhanushali, and H. Liu, “Deep anomaly detection on attributed networks,” in Proceedings of the 2019 SIAM international conference on data mining. SIAM, 2019, pp. 594–602.

[40] H. Fan, F. Zhang, and Z. Li, “Anomalydae: Dual autoencoder for anomaly detection on attributed networks,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 5685–5689.

[41] F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics bulletin, vol. 1, no. 6, pp. 80–83, 1945.

[42] X. Zhang, C. Zhou, Y. Liu, S. Zhang, P. Zhang, Y. Zhu, and Q. Liu, “Heterogeneous graph anomaly detection with graph wavelet transformer,” in 2025 IEEE International Conference on Data Mining (ICDM). IEEE, 2025, pp. 923–932.