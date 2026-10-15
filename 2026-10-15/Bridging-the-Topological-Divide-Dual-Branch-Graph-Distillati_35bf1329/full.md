# Bridging the Topological Divide: Dual-Branch Graph Distillation with Semantic Priors for Robust Cold-Start Recommendation

Yuanfeng QIU

The Hong Kong University of Science and Technology (Guangzhou)

## Abstract

Graph Collaborative Filtering models have achieved remarkable success in recommendation systems. However, they inherently sufer from a catastrophic representation collapse in strict cold-start scenarios due to the topological isolation of unobserved items. While integrating multi-modal semantic features ofers a potential remedy, directly mapping isolated semantic attributes into a globally topologydriven collaborative space frequently precipitates representation degradation and optimization conflicts. To bridge this fundamental structural divide, we propose DualGCN, an innovative Dual-Branch Graph Knowledge Distillation Framework. Operating under a decoupled two-stage training paradigm, DualGCN first constructs a robust collaborative teacher network where item representations are anchored by a topology-smoothed semantic prior, efectively reshaping the latent manifold. Subsequently, a semantic student branch employs a residual mapping mechanism and a magnitude-agnostic cosine distillation objective to achieve high-fidelity crossmodal alignment without disrupting established collaborative signals. Furthermore, we introduce a dynamic norm reconstruction mechanism to rigorously calibrate the physical scales of novel and warm items during global retrieval. Extensive experiments across three real-world benchmark datasets demonstrate that DualGCN significantly outperforms state-of-the-art baselines in strict cold-start scenarios while comprehensively preserving recommendation accuracy for highly active items.

## 1 Introduction

Recommender systems are core components of modern online platforms, including ecommerce and social media [1–4]. Among existing paradigms, Collaborative Filtering (CF) remains fundamental due to its domain-agnostic nature and proficiency in capturing user interests from historical interactions [5–8]. Because these interactions naturally form bipartite graphs, Graph Collaborative Filtering (GCF) has emerged as a dominant approach [9–11]. Specifically, ID-based GCF models explicitly capture high-order collaborative signals via multihop message passing [12]. Architectures such as LightGCN [13] and LR-GCCF [14] further streamline this paradigm by omitting non-linear transformations, achieving high training and inference eficiency through a minimalist representation space.

However, ID-only models inherently struggle with extreme data sparsity and the cold-start problem. Without suficient interaction history, cold-start items cannot be efectively mapped into the mature collaborative space. This creates a severe distribution shift between warm and cold items, significantly bottlenecking model performance [15, 16]. To mitigate this, a common practice is to incorporate side semantic features (e.g., text or multimodal metadata) to enrich the representation space [17–23].

While providing valuable contextual priors to bridge sparse interaction chains [21, 24–26], these hybrid models typically rely on complex feature extractors and multi-layer fusion modules. This complexity not only escalates computational overhead but often compromises the purity of collaborative signals, thereby degrading general recommendation performance [27,28]. This dilemma raises a critical question: Can we leverage the rich priors of semantic features for cold-start items while preserving the lightweight architecture and collaborative eficiency of ID-based models?

To answer this, we propose DualGCN, a novel dual-path graph knowledge distillation framework for cold-start recommendation. DualGCN enables approximate topological-to-semantic knowledge transfer without relying on heavy fusion modules. Our main contributions are summarized as follows:

• We introduce DualGCN, a lightweight distillation framework featuring an asymmetric dual-path architecture. By replacing traditional heavy fusion modules with this design, it achieves approximate semanticknowledge transfer to the topological space, drastically reducing online inference costs.

• We design a two-stage optimization paradigm that decouples topological updates from semantic alignment, efectively resolving gradient conflicts common in multitask graph training. Coupled with a novel topology-smoothed semantic prior, the model successfully aligns cold-start nodes within the collaborative space.

• Extensive experiments on multiple real-world datasets demonstrate that DualGCN significantly outperforms state-of-the-art baselines in graph collaborative recommendation and achieves superior performance in cold-start scenarios.

## 2 Related Work

## 2.1 Graph-based Collaborative Filtering

Conventional Matrix Factorization (MF) models [29] primarily capture first-order interactions between users and items, often overlooking the intricate topological dependencies inherent in user-item data. To address this limitation, NGCF [12] introduced a message-passing mechanism via Graph Neural Networks to explicitly model high-order collaborative signals. Building on this, [13] proposed a streamlined architecture by removing redundant feature transformations and non-linear activations. This simplification not only reduced computational complexity but also established LightGCN as a state-of-the-art benchmark in graph-based recommendation. However, these models remain heavily reliant on the connectivity of the interaction bipartite graph. In scenarios characterized by long-tail items or extreme data sparsity, the high-order information propagation mechanism often falters [16], which fundamentally constrains the model’s representation capability.

## 2.2 Cold-start Recommendation

To compensate for the scarcity of interaction signals, researchers have extensively explored leveraging item metadata, such as textual descriptions, categories, or multimodal attributes, to enhance recommendation performance. Current cold-start solutions generally bifurcate into approach-driven and data-driven strategies. Approach-driven strategies, such as Dropout-Net [18], explicitly optimize cold-start performance by randomly dropping collaborative features during training. Meta-learning models, such as MetaEmbedding [19], utilize an embedding generator to produce high-quality initial ID embeddings for new items based on their side information and a limited set of interactions. Conversely, data-driven strategies focus on augmenting the interaction matrix. For instance, content-boosted models [30] utilize content features to generate pseudo-ratings for sparse user-item pairs before applying traditional CF. DeepMusic [17] pioneered the use of deep neural networks to extract textual and acoustic features from side attributes and map them into the CF latent space, demonstrating that integrating content features efectively mitigates the cold-start dilemma. Furthermore, weakly supervised methods [31] leverage user content preferences and item attributes to score new items, bridging the gap left by missing interactions.

## 2.3 Representation Alignment in Recommendation

While content features provide a viable path to alleviate cold-start issues, an inherent heterogeneous gap exists between the topology-based ID space and the attribute-based semantic space. To bridge this discrepancy, frontier research often relies on complex external graph structures for cross-view alignment. For example, DCIB [32] employs a dual contrastive information bottleneck to facilitate topological knowledge transfer from knowledge graphs to collaborative views. Similarly, DFHE [33] injects structural information into the semantic view by fusing meta-graphs and meta-paths. However, these methods depend heavily on complete prior knowledge bases, such as high-quality knowledge graphs or predefined heterogeneous information networks. In many practical and stringent industrial scenarios, such external knowledge is often unavailable [34–36], leaving the model to rely solely on highly sparse bipartite graphs and minimal item features. Consequently, achieving eficient cross-space representation alignment, while decoupling the model from external graph structures and relying only on minimalist features, remains a critical and unresolved challenge.

## 3 Preliminaries

In this chapter, we first formalize the candidate retrieval and strict cold-start problems in recommender systems, alongside a brief review of the graph convolutional paradigm exemplified by LightGCN. Subsequently, we analyze the fundamental causes of GCN failure in cold-start scenarios from a message-passing perspective. Finally, through preliminary empirical analysis, we demonstrate the eficacy of substituting random ID embeddings with pre-trained semantic features, providing both theoretical and empirical motivation for the proposed dual-path

architecture.

## 3.1 Problem Definition

In modern recommender systems, the core objective of the candidate retrieval stage is to eficiently retrieve a highly relevant subset of items from a massive corpus for a specific user. Formally, let U and I denote the user and item sets, respectively. Based on the visibility of historical interactions, the item set is strictly partitioned into warm and cold items, such that $J = \mathcal { I } _ { w a r m } \cup \mathcal { I } _ { c o l d }$ and $\mathcal { I } _ { w a r m } \cap \mathcal { I } _ { c o l d } = \emptyset$ . The historical interactions between users and warm items naturally form a bipartite graph $\mathcal { G } = ( \mathcal { V } _ { w a r m } , \mathcal { E } )$ .

In this context, Graph Collaborative Filtering (GCF) paradigms, particularly LightGCN, have achieved remarkable success. These transductive paradigms assign learnable ID embeddings $\mathbf { e } _ { u } , \mathbf { e } _ { i } \in \mathbb { R } ^ { d }$ to each user $u \in \mathcal { U }$ and warm item $i \in \mathcal { I } _ { w a r m }$ participating in training. High-order topological signals are then explicitly aggregated by performing eficient message passing on the interaction graph $\mathcal { G }$ . Concurrently, items in modern systems are typically associated with rich multimodal content. Encoded by pre-trained foundation models, these multimodal attributes are formalized as initial semantic vectors $\mathbf { s } _ { i } \in \mathbb { R } ^ { d _ { s } }$

However, traditional GCF paradigms expose critical limitations when confronting the strict cold-start (or Out-Of-Vocabulary, OOV) challenge. Specifically, when a completely novel item $i _ { n e w } \in \mathcal { I } _ { c o l d }$ enters the system during inference, existing models sufer a dual structural failure: First, constrained by the transductive parameterized lookup table mechanism, the unseen $i _ { n e w }$ lacks a pre-learned ID embedding, resulting in an immediate representation absence in the initial collaborative space. Second, devoid of historical interactions, its degree in the interaction graph $\mathcal { G }$ is strictly zero. This structural isolation halts the neighborhood-aggregation message-passing mechanism, completely severing the new item from the established collaborative signal space.

Consequently, the core challenge of this work is formulated as an inductive cross-modal representation alignment and latent preference prediction task. The objective is to design and jointly optimize a cross-modal mapping function $f _ { \theta } : \mathbb { R } ^ { d _ { s } }  \mathbb { R } ^ { d }$ , enabling the independent inference of a pseudo-collaborative representation $\tilde { \mathbf { e } } _ { i _ { n e w } } = f _ { \theta } ( \mathbf { s } _ { i _ { n e w } } )$ for any strict cold-start item $i _ { n e w } \in \mathcal { I } _ { c o l d }$ , relying exclusively on its semantic prior $\mathbf { s } _ { i _ { n e w } }$ . To ensure the retrieval validity of $\tilde { \mathbf { e } } _ { i _ { n e w } }$ , the optimization of $f _ { \theta }$ is governed by two key constraints:

1. Heterogeneous Spatial Alignment: On the visible warm subgraph $\mathcal { G }$ , the mapped representation distribution $\{ f _ { \theta } ( \mathbf { s } _ { i } ) | i \in \mathcal { I } _ { w a r m } \}$ must strictly approximate the high-order collaborative space aggregated via message passing. This minimizes the heterogeneous distribution shift between the semantic space $\mathbb { R } ^ { d _ { s } }$ and the collaborative space $\mathbb { R } ^ { d }$

2. Zero-Shot Preference Fidelity: The generated cold-start vector $\tilde { \mathbf { e } } _ { i _ { n e w } }$ must seamlessly integrate into the existing GCF retrieval space. This ensures that the similarity metric can accurately estimate the true interaction probability between a user $u \in \mathcal { U }$ and the unseen item $i _ { n e w }$ , thereby achieving zero-shot candidate retrieval under the extreme condition of zero topological degree.

## 3.2 Review of Graph Neural Networks for Recommendation

To capture high-order collaborative signals on bipartite interaction graphs, the core paradigm of GCF focuses on optimizing initial representations via iterative message passing between nodes. Following our previous definitions, let $\mathcal { G } = ( \mathcal { V } _ { \mathrm { w a r m } } , \mathcal { E } )$ be the interaction graph comprising users and warm items. We define $\mathcal { N } _ { u }$ and $N _ { i }$ as the first-order neighbor sets for user $u \in \mathcal { U }$ and item $i \in \mathcal { I } _ { \mathrm { w a r m } }$ , respectively. Let ${ \bf e } _ { u } ^ { ( k ) } , { \bf e } _ { i } ^ { ( k ) } \in \mathbb { R } ^ { d }$ denote the node embeddings generated at the �-th graph convolutional layer, where $k = 0$ corresponds to the initialized, learnable ID embeddings.

NGCF [12] established the foundational paradigm for topology-driven feature evolution. It explicitly models the information interaction between a target node and its neighbors by incorporating parameterized feature transformations and non-linear activations. Specifically, the aggregation operation in the �-th layer of NGCF is formulated as:

$$
\mathbf { e } _ { u } ^ { ( k + 1 ) } = \sigma \left( \mathbf { W } _ { 1 } \mathbf { e } _ { u } ^ { ( k ) } + \sum _ { i \in N _ { u } } { \frac { 1 } { \sqrt { | N _ { u } | | N _ { i } | } } } \left( \mathbf { W } _ { 1 } \mathbf { e } _ { i } ^ { ( k ) } + \mathbf { W } _ { 2 } ( \mathbf { e } _ { i } ^ { ( k ) } \odot \mathbf { e } _ { u } ^ { ( k ) } ) \right) \right)\tag{1}
$$

Here, $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } \in \mathbb { R } ^ { d \times d }$ are layer-specific trainable weight matrices, ⊙ denotes the element-wise product used to capture second-order feature interactions, and $\sigma ( \cdot )$ is the non-linear activation function, typically LeakyReLU. The symmetric normalization term $\frac { 1 } { \sqrt { | N _ { u } | | N _ { i } | } }$ is introduced to mitigate scale shifts caused by degree discrepancies.

While NGCF validates the eficacy of topological signal aggregation, subsequent research by LightGCN reveals a critical ineficiency: in an ID-only embedding space devoid of explicit semantics, complex weight matrices W and non-linear activations $\sigma$ are not merely redundant, but significantly exacerbate training dificulty and induce overfitting. Consequently, LightGCN [13] drastically streamlines the GCF paradigm by discarding all feature transformations and non-linearities, retaining only the essential symmetric normalized neighborhood smoothing:

$$
{ \bf e } _ { u } ^ { ( k + 1 ) } = \sum _ { i \in N _ { u } } \frac { 1 } { \sqrt { | N _ { u } | | N _ { i } | } } { \bf e } _ { i } ^ { ( k ) }\tag{2}
$$

$$
{ \bf e } _ { i } ^ { ( k + 1 ) } = \sum _ { u \in { \cal N } _ { i } } \frac { 1 } { \sqrt { | { \cal N } _ { i } | | { \cal N } _ { u } | } } { \bf e } _ { u } ^ { ( k ) }\tag{3}
$$

After executing � layers of message passing, to synthesize collaborative signals across diferent hops and alleviate the over-smoothing problem, both NGCF and LightGCN employ a layer combination mechanism. This computes the final representations, $\tilde { \mathbf { e } } _ { u }$ and $\tilde { \mathbf { e } } _ { i } ,$ , for downstream prediction by aggregating the node embeddings across all layers via a weighted sum:

$$
\tilde { \mathbf { e } } _ { u } = \sum _ { k = 0 } ^ { K } \alpha _ { k } \mathbf { e } _ { u } ^ { ( k ) } , \quad \tilde { \mathbf { e } } _ { i } = \sum _ { k = 0 } ^ { K } \alpha _ { k } \mathbf { e } _ { i } ^ { ( k ) }\tag{4}
$$

where � is the total number of graph convolutional layers, and the scalar $\alpha _ { k } \geq 0$ is a hyperparameter controlling the weight of the �-th layer features. To maintain model simplicity and avoid additional tuning overhead, it is empirically set to a uniform distribution $\alpha _ { k } = 1 / ( K + 1 )$ in practice. Ultimately, the model computes the predicted preference score of a user for an item via the inner product $\hat { y } _ { u i } = \tilde { \mathbf { e } } _ { u } ^ { \top } \tilde { \mathbf { e } } _ { i }$

Although these architectures achieve remarkable performance on active items with abundant interactions, their underlying transductive learning paradigm renders them highly vulnerable. This heavy reliance on topological connectivity inevitably leads to catastrophic performance collapse in strict cold-start scenarios. In the following subsection, we will provide concrete experimental evidence through an empirical analysis to uncover the mechanisms behind this failure.

## 3.3 DualGCN’s Motivation

## 3.3.1 Strict Cold-Start Breakdown in GCNs

While LightGCN has achieved significant success on active items, the eficacy of its representation learning is inherently bottlenecked by the rigid coupling of historical interaction topology and pre-trained ID embeddings. From a matrix perspective, the smoothing propaga-

tion mechanism in LightGCN can be formulated as:

$$
\mathbf { E } ^ { ( k + 1 ) } = \tilde { \mathbf { A } } \mathbf { E } ^ { ( k ) } , \quad \tilde { \mathbf { A } } = \mathbf { D } ^ { - 1 / 2 } \mathbf { A } \mathbf { D } ^ { - 1 / 2 }\tag{5}
$$

where $\mathbf { E } ^ { ( k ) } \in \mathbb { R } ^ { ( N + M ) \times d }$ denotes the embedding matrix at the �-th layer, $\mathbf { E } ^ { ( 0 ) }$ represents the initial ID embeddings, and A is the adjacency matrix of the user-item interaction graph. However, when a novel item $i _ { \mathrm { n e w } } \in \mathcal { I } _ { \mathrm { c o l d } }$ enters the system during inference, traditional GCF models encounter a dual structural failure.

First, due to the constraints of the transductive parameterized lookup table mechanism, unseen items $i _ { \mathrm { n e w } }$ lack pre-learned ID embeddings. This results in an immediate representation void in the initial collaborative space, causing a collapse in model performance. Second, the absence of historical interactions implies that the degree of $i _ { \mathrm { n e w } }$ in the interaction graph $\mathcal { G }$ is strictly zero. Consequently, the corresponding row and column vectors in the normalized adjacency matrix $\tilde { \mathbf { A } }$ are null, leading to a structural fracture in the topology-dependent neighborhood aggregation mechanism. After � layers of propagation, the final output representation of such an item degenerates into an unoptimized, isolated state:

$$
\tilde { \mathbf { e } } _ { i _ { \mathrm { n e w } } } = \mathbf { e } _ { i _ { \mathrm { n e w } } } ^ { ( 0 ) }\tag{6}
$$

This isolated feature sufers from a severe distribution gap relative to the highly clustered and mature collaborative manifold. As a result, cold-start items are significantly marginalized during vector retrieval. As illustrated in Table 1, on the Amazon-Book2023 dataset, pure IDdriven models such as NGCF and LightGCN experience a catastrophic collapse of over 97% in Recall@20 and NDCG@20 metrics when transitioning from warm items to strict cold-start scenarios.

Since topological isolation severs the acquisition of collaborative signals for cold items, leveraging graph-independent prior knowledge to bridge the representation gap becomes essential. To this end, we explore the potential of substituting uninformative random IDs with high-dimensional continuous semantic features $\mathbf { s } _ { i } ~ \in ~ \mathbb { R } ^ { d _ { s } }$ extracted from pre-trained content models for node initialization. Specifically, on the Amazon-Book2023 dataset, we utilize the pre-trained all-mpnet-base-v2 model from the Sentence Transformers framework [37] as the text encoder to extract item titles and categories to generate dense semantic vectors. These vectors are subsequently reduced to the model’s latent dimension via Principal Component Analysis

Table 1: Performance comparison on Warm vs. Strict Cold-Start scenarios on Amazon-Book2023. $\Delta \downarrow$ denotes the relative performance drop when transitioning from warm to cold-start items.
<table><tr><td rowspan="2">Method</td><td colspan="3">Recall@20</td><td colspan="3">NDCG@20</td></tr><tr><td>Warm</td><td>Cold</td><td>Δ↓</td><td>Warm</td><td>Cold</td><td>Δ↓</td></tr><tr><td>NGCF LightGCN</td><td>0.0736</td><td>0.0021</td><td>-97.1%</td><td>0.0463</td><td>0.0009</td><td>-98.1%</td></tr><tr><td>NGCF-Semantic LightGCN-Semantic</td><td>0.1244 0.0753 0.0515</td><td>0.0027 0.0948</td><td>-97.8% +25.9%</td><td>0.0791 0.0459</td><td>0.0013 0.0515</td><td>-98.4% +12.2%</td></tr></table>

(PCA) [38] to serve as initial embeddings:

$$
{ \bf e } _ { i } ^ { ( 0 ) } = { \bf s } _ { i }\tag{7}
$$

As shown in the lower half of Table 1, the introduction of semantic initialization significantly mitigates or even reverses the decline in cold-start performance, providing preliminary evidence of the eficacy of semantic priors.

![](images/95705a5c5961a9ad79d3e79941e48f1b81bd06731f7248cf728aac87c5f2a56e.jpg)  
Figure 1: Uniform Manifold Approximation and Projection visualization of item representations in the latent space before and after graph message passing. With random ID initialization, the final smoothed collaborative manifold completely decouples from the initial random state, leaving degree-0 cold items severely isolated. In contrast, semantic initialization exerts a strong anchoring efect, preserving the topological clustering structure and inherently aligning the isolated cold items with the mature collaborative signal space.

## 3.3.2 Semantic Anchoring Efect

To intuitively illustrate the proposed Semantic Anchoring Efect, we employ the UMAP algorithm in Figure 1 to visualize the evolution of representations within the latent space. Following the message-passing mechanism of LightGCN, we extract both the initial ID embeddings and the final output representations after multi-layer aggregation and residual weighting. To construct a discriminative sample distribution, we randomly select anchor items from three distinct semantic categories and retrieve the 50 warm items with the highest semantic similarity to each anchor to form visible clusters. Furthermore, to simulate strict cold-start scenarios, we identify 5 isolated nodes with zero degree and the highest semantic proximity to the anchors for each category.

Our observations indicate that under random ID initialization, representations exhibit significant representation drift after high-order graph smoothing. Cold-start nodes, lacking topological connections, remain completely dissociated from the collaborative manifold. Conversely, the introduction of semantic vectors provides a robust anchoring efect for the nodes. For active items, the feature distribution remains tightly clustered around the initial semantic anchors even after low-pass graph filtering.

This empirical finding strongly supports the core motivation of this work: since semantic features can naturally anchor the distribution within the graph structure during the smoothing process, we can design a cross-modal mapping mechanism to align the semantic features $\mathbf { s } _ { i _ { \mathrm { n e w } } }$ of isolated items into the mature collaborative space. This alignment allows the model to efectively overcome the cold-start dilemma caused by the failure of graph topology.

## 4 Methodology

Building on the empirical motivations and problem scope established previously, this chapter systematically delineates the proposed Dual-Branch Graph Knowledge Distillation framework (DualGCN). The framework leverages the anchoring efect of semantic features to construct asymmetric teacher and student branches, detailing their respective architectures and training strategies. Furthermore, we introduce a dynamic magnitude reconstruction mechanism and conclude with an analysis of the computational complexity of DualGCN.

## 4.1 The Dual-Branch Alignment Architecture

## 4.1.1 Collaborative Teacher Branch

The collaborative teacher branch is designed to extract the topological structures of warmstart nodes from the bipartite interaction graph $\mathcal { G } = ( \mathcal { V } _ { \mathrm { w a r m } } , \mathcal { E } )$ , thereby establishing a stable and generalizable collaborative reference frame. To integrate the anchoring efect of the static semantic space into the collaborative topological space, we first define and align the physical sources of the two heterogeneous features involved. On one hand, the collaborative features $\mathbf { e } _ { i } \in \mathbb { R } ^ { d }$ are learnable embeddings initialized via random IDs, intended to capture pure interaction preferences.

On the other hand, static semantic features $\mathbf { s } _ { i }$ are derived from item metadata. Since coldstart items typically lack user reviews or rich interaction histories, we extract only their most fundamental attributes by concatenating the item name and category into an input description $t _ { i } .$ . Specifically, we employ a pre-trained language model all-mpnet-base-v2 [37], denoted as $\Phi _ { \mathrm { e n c } } ( \cdot )$ , to encode $t _ { i }$ into high-dimensional dense representations. These are subsequently projected via PCA [38] into a �-dimensional space aligned with the collaborative features. The specific settings for the original dimensions and � are elaborated in the Experiments section. This extraction and dimensionality reduction process is formalized as:

$$
{ \mathbf { s } } _ { i } = \mathrm { P C A } \big ( \Phi _ { \mathrm { e n c } } ( t _ { i } ) \big )\tag{8}
$$

Notably, to maintain the semantic integrity of $\mathbf { s } _ { i } .$ , this semantic prior remains strictly frozen throughout the joint training process and does not participate in backpropagation updates.

However, an inherent magnitude discrepancy exists between the dense semantic vectors generated by pre-trained models and the randomly initialized collaborative vectors. To prevent this scale variance from inducing severe representation oscillation during the early stages of optimization, DualGCN first applies $L _ { 2 }$ normalization to the semantic features. This strips away the absolute magnitude, extracting a pure semantic directional prior: $\begin{array} { r } { \bar { \bf s } _ { i } = \frac { { \bf s } _ { i } } { \| { \bf s } _ { i } \| _ { 2 } } } \end{array}$ . Subsequently, we introduce a global learnable scalar $\gamma$ to adaptively regulate the injection intensity of this semantic prior. Ultimately, the 0-th layer initial embedding for a warm item $i \in \mathcal { I } _ { \mathrm { w a r m } }$ is formulated as a composition of the collaborative basis and the semantic anchor:

$$
{ \bf e } _ { i } ^ { ( 0 ) } = { \bf e } _ { i } + \gamma \bar { \bf s } _ { i }\tag{9}
$$

For a user � $\in { \mathcal { U } } ,$ , the initial state retains only the ID embedding, i.e., ${ \bf e } _ { u } ^ { ( 0 ) } = { \bf e } _ { u }$ . During the �- th layer of graph convolutional propagation, nodes aggregate the (� − 1)-th layer representations of their first-order interactive neighbors $N _ { i }$ and $\mathcal { N } _ { u }$ via symmetric normalization:

$$
{ \bf e } _ { i } ^ { ( k ) } = \sum _ { u \in N _ { i } } \frac { 1 } { \sqrt { | N _ { i } | | N _ { u } | } } { \bf e } _ { u } ^ { ( k - 1 ) }\tag{10}
$$

After � layers of propagation, a layer combination mechanism is employed to perform mean aggregation across all layers, generating the final teacher representation $\tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } }$ for downstream tasks:

$$
\tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } } = \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \mathbf { e } _ { i } ^ { ( k ) }\tag{11}
$$

The final user representation $\tilde { \mathbf { e } } _ { u } ^ { \mathcal { T } }$ is obtained analogously.

Matrix Formulation Analysis: Let $\mathbf { R } \in \mathbb { R } ^ { | \mathcal { U } | \times | \mathcal { I } _ { \mathrm { w a r m } } | }$ denote the interaction matrix between users and warm-start items. The adjacency matrix A of the bipartite interaction graph $\mathcal { G }$ and its symmetrically normalized form $\tilde { \mathbf { A } }$ are defined as:

$$
\mathbf { A } = \left[ \begin{array} { l l } { \mathbf { 0 } } & { \mathbf { R } } \\ { \mathbf { R } ^ { T } } & { \mathbf { 0 } } \end{array} \right] , \quad \tilde { \mathbf { A } } = \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { A } \mathbf { D } ^ { - \frac { 1 } { 2 } }\tag{12}
$$

Let $\mathbf { E } _ { u } \in \mathbb { R } ^ { | \mathcal { U } | \times d }$ and $\mathbf { E } _ { i } \in \mathbb { R } ^ { | \mathcal { I } _ { \mathrm { w a r m } } | \times d }$ define the initial ID embedding matrices for users and warm items, respectively, and $\bar { \bf S } \in \mathbb { R } ^ { | { \cal T } _ { \mathrm { w a r m } } | \times d }$ denote the normalized semantic matrix. The comprehensive node embedding matrix at the 0-th layer, $\mathbf { E } ^ { ( 0 ) }$ , can be represented in block form as:

$$
\mathbf { E } ^ { ( 0 ) } = \left[ \begin{array} { c } { \mathbf { E } _ { u } } \\ { \mathbf { E } _ { i } + \gamma \bar { \mathbf { S } } } \end{array} \right]\tag{13}
$$

Consequently, the final global teacher feature matrix $\begin{array} { r } { \tilde { \mathbf { E } } ^ { \mathcal { T } } = \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \mathbf { E } ^ { ( k ) } } \end{array}$ can be linearly expanded as:

$$
\tilde { \mathbf { E } } ^ { \mathcal { T } } = \underbrace { \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \tilde { \mathbf { A } } ^ { k } \left[ \mathbf { E } _ { u } \right] } _ { \mathrm { G r a p h - P r o p a g a t e d ~ I D ~ E m b e d d i n g s } } + \underbrace { \gamma \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \tilde { \mathbf { A } } ^ { k } \left[ \mathbf { 0 } \right] } _ { \mathrm { T o p o l o g y - S m o o t h e d ~ S e m a n t i c ~ P r i o r } }\tag{14}
$$

The second term in Eq. 14 introduces the Topology-Smoothed Semantic Prior, where S<sup>¯</sup> represents the initial multimodal/content features on the item side. Notably, although the user-

side semantic prior is initially empty, the multi-hop information difusion via the graph adjacency matrix $\tilde { \mathbf { A } } ^ { k }$ enables the seamless injection of item semantic features into user representations along the interaction edges.

## 4.1.2 Semantic Student Branch

To address the structural failure caused by the zero-degree isolation of cold-start items $i _ { \mathrm { n e w } } \in \mathcal { I } _ { \mathrm { c o l d } }$ , the student branch relies exclusively on the static semantic prior $\bar { \bf s } _ { i _ { \mathrm { n e w } } }$ to construct a cross-modal alignment network. Because of the inherent distribution discrepancy between the static semantic space and the dynamic collaborative space, direct mapping is highly susceptible to representation degeneration. Therefore, this branch employs an incremental prediction mechanism, using the fundamental semantic direction as an anchor to predict the feature deviation of the cold-start target within the collaborative space.

Let $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 }$ denote the weight matrices, $\mathbf { b } _ { 1 } , \mathbf { b } _ { 2 }$ the bias terms, and LN(·) Layer Normalization. The raw student representation $\tilde { \mathbf { e } } _ { i } ^ { S }$ for item � is formally defined as:

$$
\widetilde { \mathbf { e } } _ { i } ^ { S } = \mathbf { W } _ { 2 } \sigma \left( \mathrm { L N } ( \mathbf { W } _ { 1 } \bar { \mathbf { s } } _ { i } + \mathbf { b } _ { 1 } ) \right) + \mathbf { b } _ { 2 } + \gamma \bar { \mathbf { s } } _ { i }\tag{15}
$$

Through residual connections, this branch learns the semantic-to-collaborative alignment mapping. As elaborated in the subsequent dynamic representation reconstruction section, cold-start items obtain an approximate collaborative projection via this semantic mapping.

## 4.2 Decoupled Two-stage Optimization

Graph neural networks inevitably sufer from the over-smoothing efect during message passing. If topological updates and semantic alignment are jointly optimized in an end-to-end framework, the dominant gradients from the highly sparse graph reconstruction can easily overwhelm the low-frequency alignment signals, thereby disrupting the inherent representations of the static semantic space. To isolate this gradient interference and ensure that the original recommendation accuracy for warm items is not compromised by the mapping network, DualGCN adopts a decoupled two-stage training paradigm.

The primary objective of the first stage is to construct a robust collaborative reference frame. During this phase, the parameters of the semantic alignment network $\{ \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } , \mathbf { b } _ { 1 } , \mathbf { b } _ { 2 } \}$ are strictly frozen. The semantic features serve merely as structural anchors and do not participate in the gradient updates of the learnable embeddings. The model activates $\mathbf { e } _ { u } , \mathbf { e } _ { i } .$ , and $\gamma _ { \mathrm { { ; } } }$ , and computes the predicted score via the inner product $\hat { y } _ { u i } = ( \tilde { \mathbf { e } } _ { u } ^ { \mathcal { T } } ) ^ { T } \tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } }$ . Optimization is driven by the Bayesian Personalized Ranking (BPR) loss:

$$
\mathcal { L } \mathrm { B P R } = - \sum _ { \left( u , i , j \right) } \ln \sigma \left( \hat { y } _ { u i } - \hat { y } _ { u j } \right) + \lambda \Theta\tag{16}
$$

where Θ represents the $L _ { 2 }$ regularization term for the learnable parameters. To capture highquality collaborative signals as efectively as possible, we employ a training strategy similar to LightGCN in this first stage: the trainable parameters are restricted solely to the 0-th layer embeddings, which makes $\Theta = \lVert { \bf E } ^ { ( 0 ) } \rVert ^ { 2 }$ . In this stage we also omit the dropout mechanisms commonly used in GCN [39] and NGCF.

The second stage aims to achieve targeted cross-spatial transfer of modal knowledge. The parameter space $\{ \mathbf { e } _ { u } , \mathbf { e } _ { i } , \gamma \}$ converged in the first stage is permanently frozen. The final teacher representations $\tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } }$ serve as static soft labels to exclusively supervise the alignment network parameters. The model employs a bilaterally normalized cosine distillation loss, ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ , which discards absolute constraints in the Euclidean space and solely aligns the correlation of warm and cold features on the unit hypersphere:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = 1 - \frac { 1 } { | \bar { J } _ { \mathrm { w a r m } } | } \sum _ { i \in \bar { J } _ { \mathrm { w a r m } } } \frac { ( \tilde { \mathbf { e } } _ { i } ^ { S } ) ^ { T } \tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } } } { | | \tilde { \mathbf { e } } _ { i } ^ { S } | | _ { 2 } | | \tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } } | | _ { 2 } }\tag{17}
$$

During the second-stage alignment, we investigated alternative optimization criteria, including Mean Squared Error (MSE) and InfoNCE [40]. However, empirical evidence indicates that imposing absolute magnitude constraints on a mapping network devoid of interaction frequency priors, or incorporating negative samples, yields no performance dividends. Consequently, we empirically adopt the cosine distillation loss as our final objective.

## 4.3 Adaptive Inference and Dynamic Norm Reconstruction

As the distillation stage decouples feature magnitudes from their directional information, directly utilizing the student branch’s unit vectors for global retrieval would precipitate performance degradation due to geometric misalignment between the representation spaces. To circumvent this, we propose a dynamic norm reconstruction mechanism. Specifically, for each inference batch, the model first computes the average $L _ { 2 }$ norm of the warm-start representations

![](images/8baf4b9b312aafc638b8b38cd1d604ec6b4e77288be418b540e03899d7311905.jpg)  
Figure 2: UMAP Visualization of feature evolution from initial to final representation. (a) LightGCN starts from chaotic random initialization. (b) DualGCN systematically expands from frozen semantic anchors $\tilde { \mathbf { E } } _ { \mathrm { p r i o r } }$ towards the global collaborative manifold.

from the teacher branch:

$$
\mu _ { \mathrm { n o r m } } = \frac { 1 } { | \mathcal { I } _ { \mathrm { w a r m } } | } \sum _ { i \in \mathcal { I } _ { \mathrm { w a r m } } } \big \| \tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } } \big \| _ { 2 }\tag{18}
$$

To unify the downstream retrieval space, a routing mechanism is introduced to dynamically address and reconstruct the final inference representation $\mathbf { e } _ { i } ^ { * }$ . The routing logic is governed by an interaction threshold � relative to the item’s real-time degree $d _ { i } .$ . For items in a cold-start or extremely sparse state $( d _ { i } < \tau )$ , we extract the unit directional vector from the student network and scale it by the dynamic mean $\mu _ { \mathrm { { n o r m } } }$ to achieve scale calibration. This inference process is formally defined as:

$$
\mathbf { e } _ { i } ^ { * } = \left\{ \begin{array} { l l } { \tilde { \mathbf { e } } _ { i } ^ { \mathcal { T } } , } & { \mathrm { i f } \ d _ { i } \geq \tau , } \\ { \frac { \tilde { \mathbf { e } } _ { i } ^ { S } } { \left\| \tilde { \mathbf { e } } _ { i } ^ { S } \right\| _ { 2 } } \mu _ { \mathrm { n o r m } } , } & { \mathrm { i f } \ d _ { i } < \tau . } \end{array} \right.\tag{19}
$$

Finally, the preference score for user � toward any item � is computed as the inner product $\hat { y } _ { u , i } = ( \tilde { \mathbf { e } } _ { u } ^ { \mathcal { T } } ) ^ { \top } \mathbf { e } _ { i } ^ { * }$ . By aligning the physical scales of novel and established items within a global retrieval manifold, this mechanism efectively eliminates representation bias arising from uneven magnitudes without necessitating additional training overhead.

## 4.4 Theoretical Validity Analysis of Cross-Space Mapping

An inherent distribution gap exists between the static semantic feature space and the dynamic collaborative manifold. Consequently, compelling a mapping function to reconstruct globally topology-driven representations using solely localized and isolated semantic attributes is intrinsically unreasonable. To address this, we employ an early fusion strategy for semantic and ID representations at the input of the teacher network, introducing a Topology-Smoothed Semantic Prior:

$$
\tilde { \mathbf { E } } _ { \mathrm { p r i o r } } = \frac { \gamma } { K + 1 } \sum _ { k = 0 } ^ { K } \tilde { \mathbf { A } } ^ { k } \left[ \mathbf { 0 } \right]\tag{20}
$$

By freezing this prior during training, the collaborative space is reshaped into a constrained expansion manifold, anchored by semantic information and propelled by topological connectivity. To visually trace the evolution of this feature space, we sample items from five distinct categories and project their initial and final representations onto a 2D plane using UMAP. As depicted in Figure 2, the initial representations based exclusively on random IDs or local semantics exhibit a highly scattered and entangled distribution. Following the introduction of the semantic prior mechanism, the evolved representations transform from discrete or overlapping states into dense intra-class clusters. Furthermore, they maintain distinct decision boundaries across diferent semantic categories, efectively bridging the initial cross-modal discrepancy at a geometric level.

Empowered by this manifold alignment, the student branch is provided with robust reliability guarantees for the representation reconstruction of cold-start items. Formally, this branch employs a residual architecture, decomposing the cross-space mapping task into a superposition of a fundamental semantic anchor and a non-linear topological deviation:

$$
\tilde { \mathbf { e } } _ { i } ^ { S } = \mathbf { M } \mathbf { L } \mathbf { P } ( \bar { \mathbf { s } } _ { i } ) + \gamma \bar { \mathbf { s } } _ { i }\tag{21}
$$

This design fundamentally exploits the semantic anchoring property established by the teacher branch, compelling the reconstructed coordinates of cold-start items to converge within a valid joint semantic-collaborative subspace. This structural constraint ensures that even under conditions of extreme interaction sparsity, the model successfully anchors item representations within a plausible topological neighborhood, thereby preserving a robust partial order relationship for global retrieval.

## 4.5 Complexity Analysis

Despite the introduction of the dual-branch alignment architecture and the dynamic norm reconstruction mechanism, DualGCN preserves rigorous computational eficiency. The time complexity per training epoch is predominantly governed by the message-passing operations, which scale at $O ( K | \mathcal { E } | d )$ , where �, |E |, and � denote the number of graph convolutional layers, the total number of interaction edges, and the embedding dimensionality, respectively. The space complexity is constrained to $O ( | \mathcal { V } | d + | \mathcal { E } | )$ to maintain the trainable embedding parameters and the sparse adjacency matrix. Consequently, DualGCN delivers substantial performance improvements in strict cold-start scenarios without compromising the linear scalability inherent to minimalist graph recommenders such as LightGCN.

## 5 Experimental Evaluation

In this chapter, we first detail the experimental setup. Subsequently, we conduct a comprehensive comparative evaluation against various state-of-the-art baselines under both conventional recommendation and strict cold-start scenarios. Finally, to validate the rationality and eficacy of DualGCN, we provide in-depth ablation studies and empirical analyses.

## 5.1 Experimental Settings

This section delineates the experimental setup, encompassing dataset characteristics, data preprocessing and partitioning strategies, evaluation metrics, baseline models, and specific hyperparameter configurations.

## 5.1.1 Datasets

We conduct extensive experiments on the widely recognized Amazon Reviews’23 dataset [41], which serves as a standard benchmark in the recommendation domain by providing both rich multimodal textual attributes and real-world interaction records. Specifically, we select three domain-specific subsets: Amazon-Books, Amazon-Electronics, and Amazon-Sports. To ensure evaluation reliability and maintain consistency with standard protocols in GCF research [12,13], we apply a rigorous 10-core filtering strategy. This process eliminates users and items with fewer than 10 global interactions, thereby guaranteeing a suficient baseline of collaborative signals within the initial feature space. Detailed statistics for each dataset are summarized in Table 2.

Table 2: Statistics of the datasets.
<table><tr><td>Dataset</td><td>#Users</td><td>#Items</td><td>#Interactions</td><td>Density</td></tr><tr><td>Amazon-Books</td><td>144,032</td><td>116,665</td><td>2,754,236</td><td>0.00016</td></tr><tr><td>Amazon-Electronics</td><td>358,354</td><td>125,761</td><td>4,883,752</td><td>0.00011</td></tr><tr><td>Amazon-Sports</td><td>25,640</td><td>17,543</td><td>313,993</td><td>0.00070</td></tr></table>

## 5.1.2 Data Preprocessing and Splitting

In the feature extraction phase, we leverage the all-mpnet-base-v2 [37] encoder to map the composite textual descriptions of items, specifically their titles and categories, into 768- dimensional dense vectors. A subsequent PCA projects these high-dimensional representations into a compact 64-dimensional latent space. To establish a rigorous data partitioning scheme, we randomly isolate 10% of the items to form the strict cold-start group. The interaction records of this subset are completely masked during the training phase to preclude any potential data leakage. The remaining 90% of items constitute the warm-start group, and their historical interactions are randomly partitioned into training, validation, and test sets according to a 7:1:2 ratio.

## 5.1.3 Evaluation Metrics

To rigorously evaluate the ranking performance in conventional warm-start scenarios, we adopt a full-ranking strategy. For each user in the test set, the model computes preference scores and globally ranks the ground-truth interacted items against all unobserved items. Conversely, when assessing out-of-distribution cold-start performance, it is crucial to eliminate the confounding efect of popularity bias from trending items. Therefore, the candidate pool for the full ranking evaluation is strictly restricted to the pure cold-start item set.

We employ Recall@� and NDCG@� as our primary evaluation metrics. The final reported performance results are the average scores across all users in the test set. To ensure alignment with the baseline models, the truncation length is configured to $K \in \{ 2 0 , 5 0 \}$

## 5.1.4 Baselines

To validate the superiority and eficacy of our proposed framework, we benchmark DualGCN against two representative categories of baseline methods: pure ID-based collaborative filtering

models and feature-augmented cold-start models. For a fair and rigorous comparison, the semantic inputs for all feature-driven baselines are standardized to the identical 64-dimensional vectors extracted from item titles and categories.

## Pure ID-based Collaborative Filtering

• MF [29]: As the foundational baseline for collaborative filtering, Matrix Factorization projects users and items into low-dimensional latent vectors. It relies exclusively on direct interaction histories and is optimized via the Bayesian Personalized Ranking objective.

• NGCF [12]: This graph-based model enriches representation learning by executing embedding propagation over the bipartite interaction graph. Such a mechanism explicitly captures the high-order connectivity inherent in the topological structure.

• LightGCN [13]: By innovatively stripping away the redundant non-linear activations and feature transformations prevalent in traditional graph convolutional networks, this architecture drastically enhances both training eficiency and generalization capability. It currently stands as the state-of-the-art benchmark for pure identity-driven graph collaborative filtering.

## Feature-based and Cold-start Models

• DeepMusic [17]: This approach employs deep neural networks to extract heterogeneous modal features, such as textual and acoustic attributes, from side metadata. These extracted features are subsequently mapped into the collaborative latent space to mitigate the coldstart dilemma.

• DropoutNet [18]: Designed specifically for cold-start scenarios, this framework applies a targeted dropout mechanism to user and item identity embeddings during the training phase. This deliberate perturbation forces the model to rely on content features for preference prediction under conditions of severe interaction sparsity.

• MetaEmbedding [19]: Operating as a meta-learning paradigm for cold-start recommendation, this method optimizes an embedding generator. It leverages the associated side information alongside limited interaction records of novel items to synthesize highly expressive initial identity representations.

Table 3: Overall Performance Comparison. The best results in each scenario are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">Books</td><td colspan="4">Electronics</td><td colspan="4">Sports</td></tr><tr><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td></tr><tr><td colspan="10">Warm-Start</td></tr><tr><td>MetaEmbedding</td><td>0.0631</td><td>0.1028</td><td>0.0402</td><td>0.0521</td><td>0.0098</td><td>0.0180</td><td>0.0051</td><td>0.0075</td><td>0.0187</td><td>0.0380</td><td>0.0097</td><td>0.0151</td></tr><tr><td>DropoutNet</td><td>0.0459</td><td>0.0819</td><td>0.0275</td><td>0.0382</td><td>0.0157</td><td>0.0292</td><td>0.0090</td><td>0.0128</td><td>0.0361</td><td>0.0707</td><td>0.0196</td><td>0.0291</td></tr><tr><td>DeepMusic</td><td>0.0275</td><td>0.0484</td><td>0.0158</td><td>0.0217</td><td>0.0108</td><td>0.0190</td><td>0.0063</td><td>0.0087</td><td>0.0210</td><td>0.0409</td><td>0.0111</td><td>0.0166</td></tr><tr><td>MF</td><td>0.0063</td><td>0.0132</td><td>0.0035</td><td>0.0056</td><td>0.0103</td><td>0.0186</td><td>0.0059</td><td>0.0084</td><td>0.0058</td><td>0.0105</td><td>0.0029</td><td>0.0042</td></tr><tr><td>NGCF</td><td>0.0736</td><td>0.1187</td><td>0.0463</td><td>0.0598</td><td>0.0173</td><td>0.0300</td><td>0.0101</td><td>0.0138</td><td>0.0371</td><td>0.0699</td><td>0.0211</td><td>0.0302</td></tr><tr><td>LightGCN</td><td>0.1244</td><td>0.1986</td><td>0.0791</td><td>0.1012</td><td>0.0327</td><td>0.0533</td><td>0.0204</td><td>0.0263</td><td>0.0597</td><td>0.1024</td><td>0.0360</td><td>0.0479</td></tr><tr><td>DualGCN (Phase 1)</td><td>0.1268</td><td>0.2032</td><td>0.0797</td><td>0.1024</td><td>0.0457</td><td>0.0746</td><td>0.0279</td><td>0.0361</td><td>0.0736</td><td>0.1267</td><td>0.0439</td><td>0.0586</td></tr><tr><td>DualGCN (Phase 2)</td><td>0.1262</td><td>0.2016</td><td>0.0794</td><td>0.1018</td><td>0.0456</td><td>0.0743</td><td>0.0279</td><td>0.0360</td><td>0.0717</td><td>0.1210</td><td>0.0429</td><td>0.0565</td></tr><tr><td colspan="10">Cold-Start</td></tr><tr><td>MetaEmbedding</td><td>0.0382</td><td>0.0810</td><td>0.0193</td><td>0.0302</td><td>0.0197</td><td>0.0348</td><td>0.0106</td><td>0.0142</td><td>0.0785</td><td>0.1585</td><td>0.0379</td><td>0.0568</td></tr><tr><td>DropoutNet</td><td>0.0780</td><td>0.1369</td><td>0.0406</td><td>0.0555</td><td>0.0225</td><td>0.0446</td><td>0.0104</td><td>0.0157</td><td>0.0926</td><td>0.1853</td><td>0.0428</td><td>0.0645</td></tr><tr><td>DeepMusic</td><td>0.0890</td><td>0.1441</td><td>0.0489</td><td>0.0630</td><td>0.0279</td><td>0.0506</td><td>0.0138</td><td>0.0192</td><td>0.0827</td><td>0.1565</td><td>0.0393</td><td>0.0567</td></tr><tr><td>MF</td><td>0.0016</td><td>0.0041</td><td>0.0008</td><td>0.0014</td><td>0.0016</td><td>0.0039</td><td>0.0007</td><td>0.0013</td><td>0.0112</td><td>0.0297</td><td>0.0047</td><td>0.0091</td></tr><tr><td>NGCF</td><td>0.0021</td><td>0.0052</td><td>0.0009</td><td>0.0017</td><td>0.0016</td><td>0.0040</td><td>0.0007</td><td>0.0012</td><td>0.0115</td><td>0.0279</td><td>0.0051</td><td>0.0090</td></tr><tr><td>LightGCN</td><td>0.0027</td><td>0.0053</td><td>0.0013</td><td>0.0020</td><td>0.0022</td><td>0.0052</td><td>0.0010</td><td>0.0017</td><td>0.0117</td><td>0.0283</td><td>0.0049</td><td>0.0087</td></tr><tr><td>DualGCN (Phase 1)</td><td>0.0870</td><td>0.1439</td><td>0.0478</td><td>0.0623</td><td>0.0206</td><td>0.0415</td><td>0.0096</td><td>0.0147</td><td>0.0701</td><td>0.1405</td><td>0.0320</td><td>0.0485</td></tr><tr><td>DualGCN (Phase 2)</td><td>0.1007</td><td>0.1666</td><td>0.0554</td><td>0.0723</td><td>0.0248</td><td>0.0492</td><td>0.0116</td><td>0.0175</td><td>0.1119</td><td>0.2180</td><td>0.0512</td><td>0.0760</td></tr></table>

## 5.1.5 Hyperparameter Settings

Adhering to the standard configurations established by the baseline NGCF and LightGCN models [12,13], we uniformly fix the embedding dimensionality at � = 64 for all experiments and initialize identity embeddings via the Xavier method [42]. The depth of the graph convolutional network is set to � = 3 layers. For model optimization, we employ the Adam optimizer [43] with a learning rate of 0.012, a weight decay coeficient of $1 0 ^ { - 4 }$ , and a batch size of 262, 144. The training procedure follows a decoupled two-stage schedule. Specifically, the teacher network undergoes pre-training for 600 epochs, followed by a knowledge distillation phase of 200 epochs to guarantee comprehensive convergence of the model parameters at each stage.

## 5.1.6 Performance Evaluation

This section presents a comprehensive performance comparison between the proposed Dual-GCN and various baseline models across both conventional and strict cold-start recommendation scenarios, followed by an in-depth analysis of its intrinsic advantages. The overall empirical results are summarized in Table 3.

Warm-start Recommendation Performance. We first evaluate the conventional warmstart recommendation performance on the standard test set. As demonstrated in the upper half of Table 3, the first-stage DualGCN matches or surpasses existing state-of-the-art pure ID collaborative filtering baselines, including LightGCN and NGCF, across all evaluation metrics on the three datasets. This result substantiates that integrating feature fusion and distillation mechanisms for cold-start scenarios does not compromise the model’s capacity to capture and model abundant existing interaction signals. Specifically, compared to the best-performing pure ID model, LightGCN, the first-stage DualGCN achieves a remarkable improvement of 39.7% and 23.3% in Recall@20 on the Amazon-Electronics and Amazon-Sports datasets, respectively. Even on the Amazon-Books dataset, which features relatively denser interactions, DualGCN maintains a distinct leading advantage. This superiority is primarily attributed to the teacher network’s ability to fully exploit multimodal side information, efectively bridging potential sparse connections within the pure ID interaction graph and thereby augmenting representation learning in warm-start scenarios. Concurrently, we observe that the second-stage DualGCN exhibits remarkable stability, maintaining performance highly consistent with the first stage. This confirms the eficacy of the knowledge distillation mechanism in preserving warm-start collaborative signals. Notably, while DeepMusic demonstrates competitiveness in cold-start scenarios, its performance in warm-start settings is significantly inferior to both LightGCN and DualGCN.

Strict Cold-start Recommendation Performance. To rigorously validate the model’s true generalization capability for completely novel items, we conduct further evaluations under a strictly restricted pure cold-start scenario. As illustrated in the lower half of Table 3, given the complete absence of historical interaction records for target items, pure ID-based models reliant on graph propagation, such as LightGCN, NGCF, and MF, sufer catastrophic performance degradation with metrics plummeting to near zero. In stark contrast, DualGCN demonstrates exceptional robustness. Empirical data validates that the second-stage DualGCN efectively circumvents the cold-start failure inherent to ID embeddings, significantly outperforming traditional feature-driven solutions like DropoutNet and MetaEmbedding across multiple datasets. For instance, on the Amazon-Sports dataset, the second-stage DualGCN achieves a Recall@20 of 0.1119, yielding a 20.8% improvement over the runner-up model, DropoutNet, while also consistently maintaining superiority across all metrics on the Amazon-Books dataset. Although DeepMusic exhibits competitive cold-start performance on Amazon-Electronics, it falters significantly in conventional warm-start evaluations when compared to LightGCN and DualGCN. This phenomenon profoundly substantiates that the decoupled two-stage training paradigm successfully distills high-order collaborative patterns inherent in complex graph structures and injects them into the side-information generator. Consequently, the model is empowered to synthesize high-quality initial collaborative representations for zero-interaction items, relying entirely on textual modality inputs.

![](images/0bedc228d58f2a18e2b21283aaa8f5bb2d4b91819c0859e496fe3bbd87d00e50.jpg)  
Figure 3: Ablation study on the early ID and semantic fusion strategy across diferent datasets and start scenarios. LightGCN-Semantic represents the variant where the randomly initialized ID embeddings in LightGCN are replaced by trainable semantic vectors.

## 5.2 Ablation Study

In this section, we conduct comprehensive ablation studies to evaluate the individual contributions of the core components within the DualGCN framework. Specifically, we investigate the eficacy of the feature fusion mechanism, the distillation objective, and the semantic residual design by monitoring the Recall and NDCG metrics at truncation levels of 20 and 50.

## 5.2.1 Efectiveness of Early ID and Semantic Fusion Strategy

As illustrated in Figure 3, isolating the semantic branch and subjecting it to fine-tuning precipitates a pronounced performance degradation across all datasets. This empirical observation compellingly substantiates the necessity of the anchoring efect introduced within our collaborative teacher branch. When the high-dimensional semantic features $\mathbf { s } _ { i }$ are updated via backpropagation alongside the collaborative signals, the extreme sparsity inherent in recommendation interactions renders the model highly susceptible to overfitting. Consequently, this joint optimization severely disrupts the robust topological structure originally imbued by the largescale pre-trained language model. Conversely, by strictly freezing s<sub>�</sub>, our complete DualGCN framework successfully establishes these features as a static reference frame. This mechanism efectively constrains the learning trajectory of the collaborative features $\mathbf { e } _ { i } ,$ , thereby yielding substantial improvements in both conventional retrieval and strict cold-start scenarios.

Table 4: Ablation study of DualGCN on Warm-start and Cold-start scenarios. The best results of variants are highlighted in bold. DualGCN represents our default model with Cosine Loss and Semantic Residual.
<table><tr><td rowspan="2">Architecture</td><td colspan="4">Books</td><td colspan="4">Electronics</td><td colspan="4">Sports</td></tr><tr><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td><td>R@20</td><td>R@50</td><td>N@20</td><td>N@50</td></tr><tr><td colspan="10">Part 1: Variants of Distillation Optimization</td><td></td><td></td><td></td></tr><tr><td>Warm-Start</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DualGCN-MSE</td><td>0.1264 0.1262</td><td>0.2021</td><td>0.0795</td><td>0.1020</td><td>0.0456</td><td>0.0744</td><td>0.0279</td><td>0.0360</td><td>0.0716</td><td>0.1219 0.1232</td><td>0.0424</td><td>0.0563</td></tr><tr><td>DualGCN-InfoNCE DualGCN (Default)</td><td>0.1262</td><td>0.2051</td><td>0.0796 0.0794</td><td>0.1022</td><td>0.0380</td><td>0.0622</td><td>0.0235 0.0279</td><td>0.0304 0.0360</td><td>0.0717 0.0717</td><td>0.1210</td><td>0.0430</td><td>0.0571 0.0565</td></tr><tr><td></td><td></td><td>0.2016</td><td></td><td>0.1018</td><td>0.0456</td><td>0.0743</td><td></td><td></td><td></td><td></td><td>0.0429</td><td></td></tr><tr><td>Cold-Start</td><td colspan="10"></td><td></td><td></td></tr><tr><td>DualGCN-MSE</td><td>0.0951</td><td>0.1576</td><td>0.0521</td><td>0.0681</td><td>0.0236</td><td>0.0476</td><td>0.0111</td><td>0.0169</td><td>0.1127</td><td>0.2191</td><td>0.0515</td><td>0.0766</td></tr><tr><td>DualGCN-InfoNCE</td><td>0.0972</td><td>0.1603</td><td>0.0539</td><td>0.0701</td><td>0.0143</td><td>0.0301</td><td>0.0067</td><td>0.0105</td><td>0.1090</td><td>0.2112</td><td>0.0503</td><td>0.0744</td></tr><tr><td>DualGCN (Default)</td><td>0.1007</td><td>0.1666</td><td>0.0554</td><td>0.0723</td><td>0.0248</td><td>0.0492</td><td>0.0116</td><td>0.0175</td><td>0.1119</td><td>0.2180</td><td>0.0512</td><td>0.0760</td></tr><tr><td colspan="10">Part 2: Variants of Semantic Student Branch</td><td colspan="3"></td></tr><tr><td>Warm-Start w/o Semantic Residual</td><td>0.1262</td><td>0.2015</td><td>0.0794</td><td></td><td></td><td></td><td>0.0279</td><td>0.0360</td><td></td><td></td><td></td><td>0.0565</td></tr><tr><td>DualGCN (Default)</td><td>0.1262</td><td>0.2016</td><td>0.0794</td><td>0.1018 0.1018</td><td>0.0456 0.0456</td><td>0.0743 0.0743</td><td>0.0279</td><td>0.0360</td><td>0.0717 0.0717</td><td>0.1210 0.1210</td><td>0.0429 0.0429</td><td>0.0565</td></tr><tr><td>Cold-Start</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Semantic Residual</td><td>0.0905</td><td>0.1565</td><td>0.0447</td><td>0.0616</td><td>0.0237</td><td>0.0482</td><td>0.0105</td><td>0.0164</td><td>0.1010</td><td>0.2068</td><td>0.0408</td><td>0.0658</td></tr><tr><td>DualGCN (Default)</td><td>0.1007</td><td>0.1666</td><td>0.0554</td><td>0.0723</td><td>0.0248</td><td>0.0492</td><td>0.0116</td><td>0.0175</td><td>0.1119</td><td>0.2180</td><td>0.0512</td><td>0.0760</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.2.2 Efect of Various Loss Functions

To rigorously isolate the contributions of core components within DualGCN, we conduct extensive ablation studies across the Books, Electronics, and Sports datasets. As detailed in Table 4, all model variants maintain highly consistent evaluation metrics in warm-start scenarios. For instance, the Recall@20 metric on the Books dataset uniformly stabilizes at approximately 0.126. This consistency perfectly aligns with the design rationale of our dynamic norm reconstruction mechanism, which aims to resolve the cold-start bottleneck without degrading established collaborative signals. Consequently, the subsequent analysis predominantly focuses on strict cold-start performance.

During the second-stage cross-spatial targeted transfer, we evaluate our default cosine distillation loss against alternative objectives, specifically Mean Squared Error and InfoNCE. Empirical results confirm that the default cosine formulation achieves the optimal comprehensive performance in cold-start settings. Conversely, the application of either Mean Squared Error or InfoNCE precipitates noticeable performance degradation. Specifically, substituting the objective with Mean Squared Error causes the Recall@20 on the Books and Electronics datasets to drop to 0.0951 and 0.0236, respectively, falling significantly short of the default scores of 0.1007 and 0.0248.

The fundamental mechanism driving this phenomenon lies in the rigid constraint imposed by Mean Squared Error on absolute Euclidean distances. In graph collaborative filtering, the absolute magnitude of a node representation implicitly encodes the interaction frequency prior of the corresponding item—a prior inherently absent for cold-start items. Forcing the mapping network to approximate the absolute magnitudes of the teacher model induces magnitude distortion, thereby misleading the model into overfitting spurious popularity signals. In stark contrast, the cosine distillation adopted by DualGCN strips away these absolute constraints, executing correlation alignment exclusively on the unit hypersphere, which fundamentally enhances the model’s generalization capabilities.

Furthermore, incorporating the InfoNCE objective similarly fails to yield performance dividends, even precipitating a severe decline in Recall@20 to 0.0143 on the Electronics dataset. From a mechanistic perspective, the primary objective of targeted distillation in this asymmetric framework is to achieve precise coordinate projection. Indiscriminately introducing negative samples generates extraneous repulsive forces. These forces actively disrupt the delicate geometric alignment of local features, ultimately culminating in the observed performance deterioration.

## 5.2.3 Efectiveness of Semantic Residual Design

To further validate the architectural soundness of the student branch, we evaluate a model variant devoid of the semantic residual connection, denoted as w/o Semantic Residual. Empirical results indicate that eliminating this component precipitates a consistent and stable decline in cold-start performance across all three datasets. For instance, the Recall@20 metric drops from 0.1007 to 0.0905 on the Books dataset and from 0.1119 to 0.1010 on the Sports dataset.

Given the inherent distribution gap between the static semantic space and the dynamic collaborative manifold, compelling the network to execute a direct, deep non-linear mapping renders it highly susceptible to representation degradation and the subsequent loss of initial semantic priors. The residual mechanism efectively circumvents this vulnerability. By establishing explicit semantic features as foundational structural anchors, this design empirically alleviates the optimization complexity inherent in cross-modal alignment. Consequently, it substantially enhances the fidelity of the approximate collaborative projections generated for cold-start items.

## 6 Conclusion

In this work, we systematically investigate the fundamental structural vulnerability of traditional graph collaborative filtering models when confronted with strict cold-start scenarios. Specifically, we identify that the topological isolation of unobserved items severs the messagepassing mechanism, precipitating a catastrophic collapse in representation learning. To overcome this inherent limitation, we propose DualGCN, an innovative dual-branch graph knowledge distillation framework. By seamlessly integrating high-dimensional semantic priors extracted from pre-trained language models, we fundamentally reshape the collaborative latent space.

The proposed framework operates through a decoupled two-stage training paradigm. Initially, the teacher branch constructs a robust, topology-smoothed collaborative reference frame anchored by semantic features. Subsequently, the student branch leverages a residual mapping network and a cosine distillation objective to achieve highly reliable cross-modal alignment. Furthermore, to unify the global retrieval space, we introduce a dynamic norm reconstruction mechanism that rigorously calibrates physical scales during inference, efectively circumventing magnitude-induced representation bias.

Extensive empirical evaluations across three real-world benchmark datasets comprehensively validate the superiority of DualGCN. The proposed architecture not only establishes a new state-of-the-art for strict cold-start recommendation but also strictly preserves the representation capacity for conventional warm-start items without incurring substantial computational overhead. Moving forward, we intend to extend this semantic anchoring paradigm to dynamic graph environments and explore the integration of richer multi-modal signals to further enhance out-of-distribution generalization capabilities.

## Bibliography and References

[1] T. Trinh, V.-H. Nguyen, N. Nguyen, and D.-N. Nguyen, “Product collaborative filtering based recommendation systems for large-scale e-commerce,” Int. J. Inf. Manag. Data Insights, vol. 5, p. 100322, 2025.

[2] Y. Wu and Y. Yusof, “Emerging trends in real-time recommendation systems: A deep dive into multi-behavior streaming processing and recommendation for e-commerce platforms,” J. Internet Serv. Inf. Secur., vol. 14, pp. 45–66, 2024.

[3] M. G. Silva, S. C. Madeira, and R. Henriques, “A comprehensive survey on biclusteringbased collaborative filtering,” ACM Computing Surveys, vol. 56, pp. 1 – 32, 2024.

[4] L. Chen, R. Xiong, and Y. Ji, “Application of svm model based on collaborative filtering hybrid algorithm in e-commerce recommendation,” International Journal of Computers and Applications, vol. 46, pp. 292 – 300, 2024.

[5] M. F. Aljunid, D. Manjaiah, M. K. Hooshmand, W. A. Ali, A. M. Shetty, and S. Q. Alzoubah, “A collaborative filtering recommender systems: Survey,” Neurocomputing, vol. 617, p. 128718, 2025.

[6] X. Su and T. M. Khoshgoftaar, “A survey of collaborative filtering techniques,” Advances in artificial intelligence, vol. 2009, no. 1, p. 421425, 2009.

[7] Y. Shi, M. Larson, and A. Hanjalic, “Collaborative filtering beyond the user-item matrix: A survey of the state of the art and future challenges,” ACM Computing Surveys (CSUR), vol. 47, no. 1, pp. 1–45, 2014.

[8] P. B. Thorat, R. M. Goudar, and S. Barve, “Survey on collaborative filtering, contentbased filtering and hybrid recommendation system,” International Journal of Computer Applications, vol. 110, no. 4, 2015.

[9] T. T. Tran, L. T. Nguyen, and N. N. Phien, “Graph neural networks for collaborative filtering: A survey on ranking prediction,” IEEE Access, 2026.

[10] S. Wu, F. Sun, F. Sun, and B. Cui, “Graph neural networks in recommender systems: A survey,” ACM Computing Surveys, vol. 55, pp. 1 – 37, 2020.

[11] T. Duricic, D. Kowald, E. Lacic, and E. Lex, “Beyond-accuracy: a review on diversity,´ serendipity, and fairness in recommender systems based on graph neural networks,” Frontiers in Big Data, vol. 6, 2023.

[12] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, “Neural graph collaborative filtering,” in Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval, pp. 165–174, 2019.

[13] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, “Lightgcn: Simplifying and powering graph convolution network for recommendation,” in Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pp. 639–648, 2020.

[14] L. Chen, L. Wu, R. Hong, K. Zhang, and M. Wang, “Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach,” ArXiv, vol. abs/2001.10167, 2020.

[15] H. He, X. He, Y. Peng, Z. Shan, and X. Su, “Firzen: Firing strict cold-start items with frozen heterogeneous and homogeneous graphs for recommendation,” 2024 IEEE 40th International Conference on Data Engineering (ICDE), pp. 4657–4670, 2024.

[16] J. Liu, Z. Zhang, and R. C. Cheung, “Motorec: Sparse-regularized multimodal tokenization for cold-start recommendation,” 2026.

[17] A. M. Elbir, “Deepmusic: Multiple signal classification via deep learning,” IEEE Sensors Letters, vol. 4, no. 4, pp. 1–4, 2020.

[18] M. Volkovs, G. Yu, and T. Poutanen, “Dropoutnet: Addressing cold start in recommender systems,” Advances in neural information processing systems, vol. 30, 2017.

[19] F. Pan, S. Li, X. Ao, P. Tang, and Q. He, “Warm up cold-start advertisements: Improving ctr predictions via learning to learn id embeddings,” in Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 695– 704, 2019.

[20] X. Pan, Y. Chen, C. Tian, Z. Lin, H. Hu, and W. X. Zhao, “Multimodal meta-learning for cold-start sequential recommendation,” Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022.

[21] Y. wei Wei, X. Wang, Q. Li, L. Nie, Y. Li, X. Li, and T. seng Chua, “Contrastive learning for cold-start recommendation,” Proceedings of the 29th ACM International Conference on Multimedia, 2021.

[22] H. He, X. He, Y. Peng, Z. Shan, and X. Su, “Firzen: Firing strict cold-start items with frozen heterogeneous and homogeneous graphs for recommendation,” 2024 IEEE 40th International Conference on Data Engineering (ICDE), pp. 4657–4670, 2024.

[23] H. Bai, M. Hou, L. Wu, Y. Yang, K. Zhang, R. Hong, and M. Wang, “Unified representation learning for discrete attribute enhanced completely cold-start recommendation,” IEEE Transactions on Big Data, vol. 11, pp. 1091–1102, 2025.

[24] P. Magron and C. F’evotte, “Neural content-aware collaborative filtering for cold-start music recommendation,” Data Mining and Knowledge Discovery, vol. 36, pp. 1971 – 2005, 2021.

[25] G. B. Martins, J. Papa, and H. Adeli, “Deep learning techniques for recommender systems based on collaborative filtering,” Expert Systems, vol. 37, 2020.

[26] M. Moscati, “A multimodal single-branch embedding network for recommendation in cold-start and missing modality scenarios,” Proceedings of the 18th ACM Conference on Recommender Systems, 2024.

[27] X. Zhou and Z. Shen, “A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation,” Proceedings ofthe 31st ACM International Conference on Multimedia, 2022.

[28] J. Xu, Z. Chen, S. Yang, J. Li, and E. C. H. Ngai, “The best is yet to come: Graph convolution in the testing phase for multimodal recommendation,” Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025.

[29] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, “Bpr: Bayesian personalized ranking from implicit feedback,” arXiv preprint arXiv:1205.2618, 2012.

[30] P. Melville, R. Mooney, and R. Nagarajan, “Content-boosted collaborative filtering for improved recommendations,” pp. 187–192, 2002.

[31] S.-M. Choi, K. Jang, T.-D. Lee, A. Khreishah, and W. Noh, “Alleviating item-side coldstart problems in recommender systems using weak supervision,” IEEE Access, vol. 8, pp. 167747–167756, 2020.

[32] Q. Guo, J. Hai, Z. Sun, B. Wu, and Y. Ye, “Dcib: Dual contrastive information bottleneck for knowledge-aware recommendation,” Inf. Process. Manag., vol. 62, p. 103980, 2025.

[33] J. Ma and R. Wang, “Dual-view fusion of heterogeneous information network embedding for recommendation,” IEEE Latin America Transactions, vol. 22, pp. 557–565, 2024.

[34] M. Caro-Mart´ınez, G. Jimenez-D´ ´ıaz, and J. A. Recio-Garc´ıa, “A graph-based approach for minimising the knowledge requirement of explainable recommender systems,” Knowledge and Information Systems, vol. 65, pp. 4379–4409, 2023.

[35] Y. Yang, C. Huang, L. Xia, and C. Li, “Knowledge graph contrastive learning for recommendation,” Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022.

[36] H. Wang, F. Zhang, M. Zhao, W. Li, X. Xie, and M. Guo, “Multi-task feature learning for knowledge graph enhanced recommendation,” The World Wide Web Conference, 2019.

[37] C. Galli, N. Donos, and E. Calciolari, “Performance of 4 pre-trained sentence transformer models in the semantic query of a systematic review dataset on peri-implantitis,” Inf., vol. 15, p. 68, 2024.

[38] K. P. F.R.S., “Liii. on lines and planes of closest fit to systems of points in space,” Philosophical Magazine Series 1, vol. 2, pp. 559–572, 1901.

[39] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[40] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” ArXiv, vol. abs/1807.03748, 2018.

[41] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024.

[42] X. Glorot and Y. Bengio, “Understanding the dificulty of training deep feedforward neural networks,” in Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256, JMLR Workshop and Conference Proceedings, 2010.

[43] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.