# An Analysis of Semantic and Spatio-Temporal Co-evolution for Edge Classification on Dynamic Text-Atributed Graphs

Tyler Bonnet t.bonnet24@imperial.ac.uk Imperial College London London, United Kingdom

Marek Rei marek.rei@imperial.ac.uk Imperial College London London, United Kingdom

## Abstract

Dynamic edge classification on text-attributed Web graphs, such as review networks, community forums, and news-derived knowledge graphs, requires high quality, adaptive semantic representations in the context of network evolution. Existing dynamic graph architectures typically initialize text feature vectors with pretrained language models and hold them static during training to preserve their semantic structure, forgoing the potential benefits of allowing them to co-evolve with network topology. In this work, we investigate the efects of enabling such co-evolution with a uni fied text-graph alignment framework that integrates a lightweight Transformer adapter to update token-level text representations via spatio-temporal gradients from a contrastive pretraining ob jective. To prevent detrimental semantic distortion, a language model maintenance objective anchors the evolving text embeddings to their original semantic structure. Evaluations on the Dynamic Text-Attributed Graph Benchmark demonstrate that this anchored text-graph co-evolution consistently improves upon the edge classification performance of frozen-text baselines across di verse Web-related domains. Representational analyses reveal that the framework reduces anisotropy, improving the linear separability of text embeddings, and efectively encodes structural and temporal information into the semantic space. A computational eficiency analysis confirms that this beneficial semantic and spatio temporal co-evolution is achieved without incurring the prohibitive computational overhead typical of end-to-end joint training.

## Keywords

Graph Representation Learning, Dynamic Graphs, Text-Attributed Graphs, Self-Supervised Learning, Edge Classification

## 1 Introduction

Evolving networks of the Web, where interactions carry rich text such as product reviews and community Q&A, can be represented as dynamic text-attributed graphs. Edge classification on these graphs drives Web applications such as predicting review ratings and filtering useful community forum responses. These tasks require capturing dependencies between textual content and spatio-temporal dynamics. By studying how to represent text within an evolving network, our work addresses representation learning on temporal, attributed Web-related graphs and falls within the scope of the Graph Algorithms and Modelingfor the Web track. Continuous-Time Dynamic Graph Networks (CTDGNs) have become standard for such applications, processing interactions as streaming sequences to capture dependencies [9, 15, 19, 24]. Historically, the efective use of text attributes in dynamic graph modeling was not a primary research focus, in part due to the limited availability of datasets that contained such attributes. As a result, subsequent CTDGNs [3, 21, 22, 28] were designed with the assumption that text processing was secondary or handled externally. Such models remain standard baselines in dynamic text-attributed graph modeling [30].

To apply these architectures to text-rich domains, the prevailing approach is to initialize text attributes with a pretrained language model (PLM) and use them as static features during training. While evaluations in DTGB confirm that the frozen text embeddings offer a performance improvement over random vectors, this design remains fundamentally limited. The predictive value of the text in an interaction often depends heavily on the evolving historical topology between the participating nodes. By freezing the text representations, the model is forced to map generic semantic spaces to dynamic structural patterns. Consequently, the text attributes remain entirely decoupled from temporal interaction patterns, forcing the graph encoder to bridge a substantial representational gap alone and ultimately restricting dynamic edge classification performance.

Motivated by these limitations, we investigate the efects of enabling the co-evolution of text representations and network topology in dynamic text-attributed Web graphs with a unified textgraph alignment (UTGA) framework<sup>1</sup>. Our approach integrates a dynamic graph encoder with a trainable language model adapter that receives gradient updates from a contrastive pretraining objective, encoding spatio-temporal information into the text embeddings. To prevent detrimental semantic distortion, we use a language model maintenance loss that anchors the outputs of the adapter to the original PLM embeddings.

Evaluation on DTGB demonstrates that UTGA consistently outperforms baselines relying on static text embeddings for dynamic edge classification across a range oflabel budgets. An analysis ofthe learned textual latent space shows that UTGA reduces anisotropy, substantially improving the linear separability of the text embeddings, and efectively encodes structural and temporal properties into the semantic space. Crucially, this co-evolution avoids the prohibitive computational overhead typical of end-to-end training, operating within the same memory footprint order of magnitude as frozen-text baselines.

## 2 Related Work

This section reviews text attribute processing in standard CTDGNs and semantic alignment on text-attributed graphs.

## 2.1 Text Attribute Processing in CTDGNs

The prevailing approach to processing text attributes in CTDGNs was established by JODIE [9], which treats them as fixed inputs rather than learnable representations in its state update equation as it compresses historical interactions into latent states. DyRep [19] adopts this strategy, incorporating fixed representations of text attributes to update latent node states, which in turn parameterize an intensity function to modulate event occurrence rates. This treat ment of text was formalized by TGN [15], which unified memory based methods into a generic framework where text representations continued to serve as static components of the message vector passed to the recurrent module. In these architectures, learning is confined to the spatio-temporal dynamics of the memory update, leaving semantic representations immutable throughout training.

The development of more expressive architectures improved spatio-temporal modeling but retained this passive treatment of text attributes. TGAT [24] aggregates interactions with temporal selfattention, while CAWN [22] extracts network motifs by sampling causal anonymous walks and encoding them with an RNN. Despite these innovations, both architectures process text representations as fixed inputs. In Transformer-based models such as DyGFormer [28] and TCL [21], interaction histories are processed as sequences where text embeddings are combined with other feature vectors, such as time delta encodings. Although these models introduce mechanisms such as patching to capture long-range dependencies or contrastive objectives to improve robustness, they do not update the text representations. This paradigm is further cemented by DTGB [30], which, despite shifting attention towards the textual modality of dynamic graphs, standardizes an evaluation protocol that relies on frozen text embeddings.

## 2.2 Semantic Alignment on Text-Attributed Graphs

Semantic alignment strategies for text-attributed graphs can be organized into two broad categories: (1) those that rely on jointoptimization driven by downstream task labels; and (2) task-agnostic self-supervised pretraining frameworks. While there have been recent advancements in the former category in both the static [33, 34] and dynamic [16, 23, 26, 31] domains, our work is positioned in the latter. Therefore, our review focuses on methods with a decoupled self-supervised pretraining phase.

Early approaches in the static graph domain reduced text to bagof-words [17] or TF-IDF [27] vectors, but with the advent of PLMs came a shift towards deep semantic alignment. GIANT [2] aligns semantic and topological spaces by instructing the text encoder to solve a self-supervised neighborhood prediction task, efectively distilling neighborhood structure into the semantic embeddings. To capture more complex topological semantics, Patton [8] pretrains language models with network-contextualized masked language modeling and masked node prediction, grounding token representations in the text ofadjacent nodes. Because jointly training large language models and graph structures is computationally costly, other lines of research prioritized eficiency. For instance, GraphAdapter [7] uses a GNN as a lightweight, trainable adapter to fuse structural representations directly into the hidden states of frozen PLMs, pretraining the framework via task-agnostic next-token prediction. More recently, the field has expanded its focus from singledomain alignment to cross-domain transferability. GraphCLIP [35] addresses this by applying a self-supervised contrastive objective that aligns tunable graph encoders with LLM-generated text summaries in a shared semantic subspace. Overall, these approaches demonstrate that actively aligning semantic and structural spaces through end-to-end pretraining or parameter-eficient adaptation is essential to overcome the limitations of frozen-text architectures on static graphs.

In the dynamic graph domain, however, deep semantic alignment remains relatively underexplored. While self-supervised pretraining frameworks spanning generative [5, 12] and contrastive [1, 25, 36] approaches have advanced generalizable representation learning for dynamic graphs, they lack mechanisms for semantic alignment and retain the inherent limitations of frozen text embeddings.

## 3 Methodology

This section establishes preliminaries and details the UTGA framework (Figure 1), including the base framework, text adaptation module, and language model maintenance objective.

## 3.1 Preliminaries

A dynamic text-attributed graph is defined as $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathcal { T } , \mathcal { D } , \mathcal { R } )$ where $_ \mathrm { c } { } _ { V }$ is the node set, $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V } \times \mathcal { T }$ contains timestamped edges, $\mathcal { T }$ is the set of timestamps at which edges occur, D is the set of node text attributes, and R is the set of edge text attributes. An edge $( u , v , t ) \in \mathcal { E } ,$ , with its associated node and edge text attributes, connects a source node � to a destination node � at time �, and has a label $y _ { u v } ^ { t }$ representing the edge class.

At �, with a set of edges $\mathcal { E } _ { \leq t } \ = \ \{ ( u , v , t ^ { \prime } , y _ { u v } ^ { t ^ { \prime } } ) \ \in \ \mathcal { E } \ | \ t ^ { \prime } \ \leq \ t \}$ the evaluation follows the dynamic edge classification protocol of DTGB [30], where the task is defined as predicting the label of an edge at � + 1 given historical information, with the objective of learning a function $f : ( \mathcal { V } , \mathcal { E } _ { \leq t } , \mathcal { D } , \mathcal { R } ) \to y _ { u v } ^ { t + 1 }$ , which predicts edge classes between node pairs at future timestamps.

## 3.2 Base Framework

While our unified text-graph alignment framework is inherently encoder-agnostic, we adopt the role-aware sequence modeling framework of DyGnROLE [1] as our experimental vehicle. This framework belongs to the broader family of Transformer-based CTDGNs that process the historical interactions of query nodes as a sequence [21, 28]. Given a query interaction $( u , v , t )$ , the encoder constructs temporal neighborhood sequences for the source and destination nodes. Each node position in the sequence is represented by a concatenation of five feature channels: node text features, edge text features, time encodings, within-sequence frequency embeddings, and cross-sequence frequency embeddings. These representations, along with prepended role-specific CLS tokens, are processed by a Transformer encoder to extract the final source and destination node representations, $\mathbf { z } _ { u }$ and $\mathbf { z } _ { v }$

To optimize these representations and structure the latent space around the directional interaction process, the framework uses the directional role alignment (DRA) objective. DRA formulates pretraining as an InfoNCE [13] retrieval task formulated as

![](images/f4d18065ff3c632d3d4fac5f14d706dd3e9d412e864ca11412329efd8fcb46dd.jpg)  
Figure 1: Overview of the Unified Text-Graph Alignment Framework. (Left) For a query interaction between nodes � and � at time $t ,$ historical one-hop neighbors are retrieved. � and � combined with their respective neighbors form node sequences $S _ { u }$ and $S _ { v } .$ For each event in these sequences, its raw node text attribute $n _ { \mathnormal { \star } }$ raw edge text attribute $e ,$ interaction time delta $t ,$ cross-sequence neighbor frequency �, and within-sequence neighbor frequency � are extracted, and the edge text attribute of the query nodes is masked. (Center) � and � are processed by a frozen PLM into raw token-level text embeddings, which are subsequently transformed by the PLM adapter. For each node and edge text sequence, the final hidden state corresponding to the EOS token is extracted to serve as the feature vector representing the full text. To prevent semantic collapse, these adapted text feature vectors are anchored to their corresponding raw text embeddings via the language model maintenance (LMM) objective. The text feature vectors and the bypassed temporal and structural feature vectors $( t , c , w )$ are independently projected via separate linear layers $\mathbf { W } _ { n , e , t , c , w }$ and concatenated. Role-semantic positional encodings (RSPE) are added to the projected vectors to distinguish source query, destination query, source neighbors, and destination neighbors. Distinct CLS tokens are prepended to form the final augmented input matrices $\mathbf { X } _ { u }$ and $\mathbf { X } _ { v } .$ (Right) The sequences $\mathbf { X } _ { u }$ and $\mathbf { X } _ { v }$ are processed independently by the Transformer encoder, using shared weights. The CLS tokens are extracted to serve as the final source and destination representations $\mathbf { z } _ { u }$ and ${ \bf z } _ { v } ,$ which are optimized via the directional role alignment (DRA) objective.

$$
\mathcal { L } _ { \mathrm { { D R A } } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp \left( \frac { \mathbf { z } _ { u } ^ { ( i ) } \cdot \mathbf { z } _ { v } ^ { ( i ) } } { \tau } \right) } { \sum _ { j = 1 } ^ { N } M _ { i j } \exp \left( \frac { \mathbf { z } _ { u } ^ { ( i ) } \cdot \mathbf { z } _ { v } ^ { ( j ) } } { \tau } \right) } ,\tag{1}
$$

where � is the batch size, � is the temperature parameter, and $\mathbf { z } _ { u } ^ { ( i ) }$ and $\mathbf { z } _ { \upsilon } ^ { ( i ) }$ represent the ℓ<sub>2</sub>-normalized embeddings of the source and destination nodes. To prevent penalizing established structural relationships, DRA applies a binary historical mask $M _ { i j } \in \{ 0 , 1 \}$ that removes previously observed interactions from the denominator.

By default, the DyGnROLE framework relies on static text features generated by a frozen language model. In the following sections, we detail how our approach replaces these static inputs with a fully diferentiable text adaptation module, using the gradients from $\mathcal { L } _ { \mathrm { D R A } }$ to drive the semantic and topological co-evolution.

## 3.3 PLM Adapter

Our framework replaces static features with a fully diferentiable PLM adapter, which is a lightweight Transformer [20] operating on token-level representations produced by a frozen PLM. In our implementation, we use Qwen3-Embedding-0.6B [32], although the proposed adapter architecture is agnostic to the choice of PLM and can be applied to any model capable of producing contextualized token representations. Node and edge attributes are first tokenized and passed through the frozen PLM, from which the full sequence of final-layer hidden states are extracted. Learnable absolute positional embeddings are added to these contextualized token representations, which are then processed by the adapter, providing a trainable semantic transformation between the PLM latent space and the graph encoder.

The adapter consists of two Transformer layers, each containing bidirectional multi-head self-attention followed by a position-wise SwiGLU feed-forward network [18]. Both sub-layers use residual connections and pre-layer normalization using RMSNorm [29]. For an input sequence of token representations �, the adapter computes

$$
H ^ { \prime } = H + \operatorname { D r o p o u t } ( \mathrm { M H A } ( \mathrm { R M S N o r m } ( H ) ) )\tag{2}
$$

$$
H ^ { \prime \prime } = H ^ { \prime } + \mathrm { S w i G L U } ( \mathrm { R M S N o r m } ( H ^ { \prime } ) ) ,\tag{3}
$$

where MHA denotes multi-head self-attention and RMSNorm denotes Root Mean Square Normalization. The feed-forward transformation is defined as

$$
\mathrm { S w i G L U } ( H ^ { \prime } ) = ( \mathrm { S i L U } ( H ^ { \prime } W _ { \mathrm { g a t e } } ) \odot H ^ { \prime } W _ { \mathrm { v a l } } ) W _ { \mathrm { o u t } } ,\tag{4}
$$

where SiLU is the Sigmoid Linear Unit activation function, $W _ { \mathrm { g a t e } } ,$ $W _ { \mathrm { v a l } }$ , and $W _ { \mathrm { o u t } }$ are learnable projection matrices, and ⊙ denotes element-wise multiplication. Following the adapter transformation, the final hidden state corresponding to the EOS token is extracted as the resulting text representation ${ \bf { z } } _ { t e x t } = H _ { \mathrm { { E O } } } ^ { \prime \prime }$ <sub>S</sub>.

This representation is then passed through the feature channels of the graph encoder, enabling structural gradients from the DRA objective to propagate through the adapter and adapt the semantic space according to temporal interaction patterns.

## 3.4 LMM Objective

To prevent detrimental semantic distortion during DRA optimization, we introduce a language model maintenance (LMM) loss. Exposing a text encoder purely to structural gradients risks an unconstrained adaptation that prioritizes network topology at the expense of broader semantic relationships. While standard mitiga tion strategies often rely on computationally intensive language modeling tasks or rigid distance penalties, these approaches are either highly ineficient or too restrictive, heavily penalizing any deviation from the original embedding and hindering meaningful structural adaptation. To address this, the LMM objective acts as a batch-wise contrastive regularizer. Rather than forcing the adapted vector to match the original, it anchors the adapted representations to their corresponding frozen PLM representations through an InfoNCE objective [13] formulated as

$$
\mathcal { L } _ { \mathrm { L M M } } = - \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \log \frac { \exp \left( \frac { \tilde { \mathbf { z } } ^ { ( k ) } \cdot \mathbf { z } ^ { \mathrm { f r o z e n } ( k ) } } { \tau } \right) } { \sum _ { j = 1 } ^ { N } \exp \left( \frac { \tilde { \mathbf { z } } ^ { ( k ) } \cdot \mathbf { z } ^ { \mathrm { f r o z e n } ( j ) } } { \tau } \right) } ,\tag{5}
$$

where � denotes the number of text representations in the minibatch, $\tilde { \mathbf { z } } ^ { ( k ) }$ represents the adapted EOS representation for the �-th text attribute processed through an MLP projection task head, � is set to 0.07 following standard practice in contrastive learning, and z<sup>frozen(�)</sup> denotes the corresponding frozen PLM EOS representation before adaptation. By regularizing the latent vector with respect to the batch, this formulation ensures that an adapted representation maintains its distinct semantic identity relative to other texts. It grants the text embeddings the freedom to shift and organize around network topology, provided they remain more similar to their original semantic anchor than to the anchors of unrelated interactions. This provides a flexible and eficient mechanism to preserve linguistic structure without limiting the structural expressivity gained through DRA.

## 3.5 Implementation

We provide the hyperparameter settings of UTGA in Appendix B and text embedding extraction details, including storage and GPU compute hour requirements, in Appendix C. During pretraining, UTGA jointly optimizes $\mathcal { L } _ { \mathrm { D R A } }$ and $\mathcal { L } _ { \mathrm { L M M } }$ using dynamic weight averaging [11] to balance task weights based on their historical loss drop rates over recent training epochs. Performance on the validation split is monitored with a composite score:

$$
\mathrm { S c o r e = \frac { M R R _ { v a l } - M R R _ { b e s t } } { M R R _ { b e s t } } + \frac { L o s s _ { b e s t } - L o s s _ { v a l } } { L o s s _ { b e s t } } . }\tag{6}
$$

Early stopping is implemented with a patience of 10 epochs based on the composite score.

## 4 Experiments

In this section, we investigate the impact of semantic and topological co-evolution on dynamic edge classification. We outline our experimental setup and evaluate our framework alongside diverse baselines on multiple datasets.

## 4.1 Setup

Datasets. We evaluate the unified text-graph framework on all eight datasets from the Dynamic Text-Attributed Graph Benchmark (DTGB) [30]. These datasets feature timestamped edges accompanied by raw text attributes, spanning four domains: e-commerce reviews (Amazon, Google, Yelp), technical Q&A forums (Stack E, Stack U), corporate communications (Enron), and temporal knowledge graphs (GDELT, ICEWS). The datasets are available through the oficial DTGB repository<sup>2</sup> as chronologically ordered edge lists with source and destination node IDs, timestamps, and edge labels. Detailed descriptions, summary statistics, label distributions, and average node and edge text lengths are provided in Appendix A.

Crucially, because our investigation centers on unified text-graph alignment, DTGB provides a rigorous setting due to the substantial variation in the length of its node and edge text attributes. Average text length varies considerably across datasets and between modalities. Node attributes are particularly long in the technical Q&A datasets, averaging 142 words for Stack E and 109 words for Stack U, while they are much shorter in Enron and GDELT, averaging only 1 and 2 words, respectively. The extremely short node attributes in Enron arise because they consist of email addresses rather than natural-language descriptions. Edge attributes exhibit a diferent pattern: Enron, Yelp, and Amazon contain comparatively long interaction text, averaging 151, 105, and 96 words, respectively, whereas the temporal knowledge graphs contain very short edge descriptions, averaging only 4 words for both GDELT and ICEWS. Google similarly provides relatively short text in both modalities, with averages of 8 words for nodes and 23 words for edges. These diferences expose the framework to substantially diferent distributions of linguistic information, from text-rich node attributes and long-form interactions to very short textual descriptions, allowing us to evaluate whether structural information can be efectively integrated across diverse semantic conditions.

Tasks. Performance is assessed across four downstream tasks corresponding to the dataset domains. On the e-commerce networks (Amazon, Google, Yelp), models perform rating prediction to predict a 1-5 star rating accompanying a future review. On the technical Q&A forums (Stack E, Stack U), the task is post usefulness prediction, determining if a future reply will be upvoted by the community. For corporate communications (Enron), models perform email topic classification, categorizing future messages into one of 10 subjects (e.g., “calendar”, “deal”). On the temporal knowledge graphs (GDELT, ICEWS), the models are evaluated on political relation classification, a highly complex task requiring the model to predict future geopolitical interactions from a large label space (236 categories for GDELT, 266 for ICEWS).

Table 1: Finetuning results for dynamic edge classification on DTGB datasets. Values show average Macro F1 scores in percent with subscripts denoting ± standard deviation. Bold indicates best performance per dataset. Relative performance gains of UTGA over the base framework DyGnROLE are shown in green superscript. $^ { * } , \dagger ,$ and ‡ respectively indicate statistically significant improvement at $p < 0 . 1 , p < 0 . 0 5 ,$ and $\textstyle p < 0 . 0 1$ over the best-in-column baseline for individual datasets using a paired two-tailed �-test, and over the best overall baseline for the Avg column using a Wilcoxon signed-rank test.
<table><tr><td>Method</td><td>Amazon</td><td>Google</td><td>Yelp</td><td>Enron</td><td>Stack U</td><td> $\mathrm { S t a c k E }$ </td><td>GDELT</td><td>ICEWS</td><td>Avg</td></tr><tr><td>Random</td><td> $1 5 . 2 7 _ { \pm 0 . 0 4 }$ </td><td> $1 5 . 0 3 _ { \pm 0 . 0 5 }$ </td><td> $1 6 . 9 5 _ { \pm 0 . 0 3 }$ </td><td> $7 . 0 1 _ { \pm 0 . 0 4 }$ </td><td> $4 4 . 7 6 _ { \pm 0 . 0 5 }$ </td><td> $4 6 . 7 2 _ { \pm 0 . 0 7 }$ </td><td> $0 . 1 6 _ { \pm 0 . 0 1 }$ </td><td> $0 . 1 3 _ { \pm 0 . 0 0 }$ </td><td>18.25</td></tr><tr><td>Majority</td><td> $1 5 . 7 6 _ { \pm 0 . 0 0 }$ </td><td> $1 5 . 9 3 _ { \pm 0 . 0 0 }$ </td><td> $1 3 . 9 9 _ { \pm 0 . 0 0 }$ </td><td> $4 . 0 6 _ { \pm 0 . 0 0 }$ </td><td> $4 4 . 6 5 _ { \pm 0 . 0 0 }$ </td><td> $4 2 . 8 2 _ { \pm 0 . 0 0 }$ </td><td> $0 . 0 7 _ { \pm 0 . 0 0 }$ </td><td> $0 . 1 2 _ { \pm 0 . 0 0 }$ </td><td>17.17</td></tr><tr><td>DyRep</td><td> $1 6 . 0 9 _ { \pm 0 . 6 1 }$ </td><td> $1 3 . 9 5 _ { \pm 3 . 5 8 }$ </td><td> $1 5 . 1 2 _ { \pm 1 . 1 0 }$ </td><td> $8 . 7 5 _ { \pm 0 . 5 2 }$ </td><td> $4 4 . 9 1 _ { \pm 0 . 3 2 }$ </td><td> $4 3 . 9 2 _ { \pm 2 . 1 4 }$ </td><td> $0 . 1 3 _ { \pm 0 . 0 2 }$ </td><td> $0 . 1 4 _ { \pm 0 . 0 2 }$ </td><td>17.88</td></tr><tr><td>JODIE</td><td> $1 7 . 2 5 _ { \pm 1 . 9 0 }$ </td><td> $1 7 . 2 2 _ { \pm 1 . 3 5 }$ </td><td> $1 5 . 1 7 _ { \pm 2 . 3 4 }$ </td><td> $6 . 8 1 _ { \pm 1 . 4 8 }$ </td><td> $4 4 . 9 0 _ { \pm 0 . 8 3 }$ </td><td> $4 2 . 9 8 _ { \pm 1 . 6 3 }$ </td><td> $0 . 1 1 _ { \pm 0 . 0 2 }$ </td><td> $0 . 1 5 _ { \pm 0 . 0 2 }$ </td><td>18.07</td></tr><tr><td>DVGMAE</td><td> $1 6 . 1 0 _ { \pm 0 . 4 5 }$ </td><td> $1 6 . 2 3 _ { \pm 0 . 4 6 }$ </td><td> $1 8 . 3 5 _ { \pm 5 . 3 4 }$ </td><td> $1 0 . 1 8 _ { \pm 0 . 7 8 }$ </td><td> $4 5 . 5 9 _ { \pm 1 . 8 8 }$ </td><td> $4 2 . 8 2 _ { \pm 0 . 0 1 }$ </td><td> $0 . 1 5 _ { \pm 0 . 0 2 }$ </td><td> $0 . 2 2 _ { \pm 0 . 1 4 }$ </td><td>18.71</td></tr><tr><td>IDOL</td><td> $1 5 . 9 1 _ { \pm 0 . 3 0 }$ </td><td> $1 5 . 9 3 _ { \pm 0 . 0 0 }$ </td><td> $2 4 . 8 0 _ { \pm 0 . 1 0 }$ </td><td> $8 . 1 4 _ { \pm 0 . 2 5 }$ </td><td> $4 4 . 6 5 _ { \pm 0 . 0 0 }$ </td><td> $4 2 . 8 2 _ { \pm 0 . 0 0 }$ </td><td> $0 . 1 2 _ { \pm 0 . 0 1 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 1 8 }$ </td><td>19.11</td></tr><tr><td>CLDG</td><td> $1 5 . 5 7 _ { \pm 0 . 3 8 }$ </td><td> $1 6 . 2 0 _ { \pm 0 . 5 3 }$ </td><td> $2 6 . 9 0 _ { \pm 1 . 7 7 }$ </td><td> $8 . 2 8 _ { \pm 1 . 1 2 }$ </td><td> $4 4 . 6 5 _ { \pm 0 . 0 0 }$ </td><td> $4 5 . 1 5 _ { \pm 2 . 4 8 }$ </td><td> $0 . 1 7 _ { \pm 0 . 0 0 }$ </td><td> $0 . 5 6 _ { \pm 0 . 0 7 }$ </td><td>19.68</td></tr><tr><td>DyGMAE</td><td> $1 6 . 0 5 _ { \pm 0 . 5 8 }$ </td><td> $1 5 . 9 3 _ { \pm 0 . 0 0 }$ </td><td> $2 4 . 8 6 _ { \pm 0 . 2 6 }$ </td><td> $8 . 5 9 _ { \pm 1 . 1 0 }$ </td><td> $4 7 . 1 2 _ { \pm 2 . 0 8 }$ </td><td> $4 8 . 0 8 _ { \pm 0 . 4 1 }$ </td><td> $0 . 1 9 _ { \pm 0 . 0 0 }$ </td><td> $0 . 5 0 _ { \pm 0 . 0 7 }$ </td><td>20.17</td></tr><tr><td>CAWN</td><td> $1 8 . 5 9 _ { \pm 0 . 6 9 }$ </td><td> $1 9 . 6 4 _ { \pm 0 . 5 4 }$ </td><td> $2 8 . 7 7 _ { \pm 1 . 0 2 }$ </td><td> $9 . 8 8 _ { \pm 0 . 8 4 }$ </td><td> $4 4 . 6 5 _ { \pm 0 . 0 0 }$ </td><td> $4 2 . 8 2 _ { \pm 0 . 0 0 }$ </td><td> $0 . 3 3 _ { \pm 0 . 0 5 }$ </td><td> $0 . 9 4 _ { \pm 0 . 0 8 }$ </td><td>20.70</td></tr><tr><td>DyGFormer</td><td> $1 5 . 7 6 _ { \pm 0 . 0 0 }$ </td><td> $1 8 . 0 0 _ { \pm 1 . 6 7 }$ </td><td> $2 5 . 6 9 _ { \pm 2 . 6 8 }$ </td><td> $9 . 2 9 _ { \pm 0 . 4 2 }$ </td><td> $4 9 . 6 8 _ { \pm 4 . 1 2 }$ </td><td> $5 0 . 7 3 _ { \pm 3 . 1 6 }$ </td><td> $0 . 3 6 _ { \pm 0 . 0 2 }$ </td><td> $1 . 0 3 _ { \pm 0 . 0 3 }$ </td><td>21.32</td></tr><tr><td>TCL</td><td> $1 8 . 5 2 _ { \pm 0 . 4 0 }$ </td><td> $1 9 . 4 9 _ { \pm 0 . 4 4 }$ </td><td> $2 7 . 8 5 _ { \pm 0 . 4 3 }$ </td><td> $9 . 8 6 _ { \pm 0 . 3 8 }$ </td><td> $4 6 . 3 5 _ { \pm 3 . 4 0 }$ </td><td> $4 8 . 7 7 _ { \pm 4 . 9 3 }$ </td><td> $0 . 3 8 _ { \pm 0 . 0 3 }$ </td><td> $0 . 9 6 _ { \pm 0 . 0 4 }$ </td><td>21.52</td></tr><tr><td>DyGnROLE</td><td> $1 6 . 5 3 _ { \pm 0 . 3 0 }$ </td><td> $1 9 . 8 5 _ { \pm 0 . 5 4 }$ </td><td> $2 9 . 3 5 _ { \pm 0 . 9 2 }$ </td><td> $1 0 . 6 9 _ { \pm 0 . 4 4 }$ </td><td> $5 0 . 9 9 _ { \pm 2 . 0 8 }$ </td><td> $5 1 . 4 9 _ { \pm 2 . 1 6 }$ </td><td> $0 . 3 8 _ { \pm 0 . 0 6 }$ </td><td> $1 . 1 7 _ { \pm 0 . 0 9 }$ </td><td>22.56</td></tr><tr><td>GraphMixer</td><td> $1 6 . 8 0 _ { \pm 1 . 2 8 }$ </td><td> $1 9 . 0 6 _ { \pm 1 . 5 8 }$ </td><td> $2 5 . 9 4 _ { \pm 1 . 2 0 }$ </td><td> $9 . 3 6 _ { \pm 0 . 4 3 }$ </td><td> $5 4 . 9 9 _ { \pm 5 . 2 0 }$ </td><td> $5 4 . 6 6 _ { \pm 1 . 8 1 }$ </td><td> $0 . 3 1 _ { \pm 0 . 0 6 }$ </td><td> $0 . 7 0 _ { \pm 0 . 1 0 }$ </td><td>22.73</td></tr><tr><td>TGAT</td><td> $1 7 . 9 9 _ { \pm 1 . 1 7 }$ </td><td> $1 9 . 6 5 _ { \pm 0 . 8 7 }$ </td><td> $2 8 . 9 7 _ { \pm 0 . 4 3 }$ </td><td> $9 . 4 3 _ { \pm 0 . 2 8 }$ </td><td> $5 4 . 8 2 _ { \pm 0 . 6 8 }$ </td><td> $5 5 . 5 3 _ { \pm 1 . 1 0 }$ </td><td> $0 . 3 6 _ { \pm 0 . 0 6 }$ </td><td> $1 . 0 5 _ { \pm 0 . 0 5 }$ </td><td>23.48</td></tr><tr><td>UTGA</td><td> $\pm 4 . 7 9 _ { \pm 0 . 5 6 } ^ { + 5 0 . 0 \% \ddagger }$ </td><td> ${ \bf 2 5 . 1 3 } _ { \pm 0 . 3 2 } ^ { + 2 6 . 6 \% \ddagger }$ </td><td> ${ \bf 3 6 . 8 1 } _ { \pm 0 . 2 3 } ^ { + 2 5 . 4 \% \ddagger }$ </td><td> $1 8 . 8 2 _ { \pm 1 . 1 3 } ^ { + 7 6 . 1 \% \div }$ </td><td> ${ 5 5 . 5 0 } _ { \pm 0 . 1 9 } ^ { + 8 . 8 \% }$ </td><td> ${ \bf 5 6 . 3 3 _ { \pm 0 . 7 5 } ^ { + 9 . 4 \% } }$ </td><td> $\mathbf { 0 . 6 3 } _ { \pm 0 . 0 3 } ^ { + 6 5 . 8 \% \ddagger }$ </td><td> $2 . 2 6 _ { \pm 0 . 1 7 } ^ { + 9 3 . 2 \% \ddagger }$ </td><td> $2 7 . 5 3 ^ { + 2 2 . 0 \% \ddagger }$ </td></tr></table>

Self-supervised models are then finetuned while supervised models are trained from random initialization for dynamic edge classification. Following the standard DTGB evaluation protocol, the node representations generated by the models are concatenated and passed through an MLP classifier consisting of two hidden layers with ReLU activations and dropout regularization, followed by a linear output layer producing class logits. The full architecture and classification head are trained end-to-end using cross-entropy loss. We perform this experiment with varying label budgets for dynamic edge classification (1k, 5k, 10k, and 20k training interactions, with proportional 150, 750, 1,500, and 3,000 validation interactions, respectively), holding out the full DTGB test split for final evaluation. Early stopping uses a 20-epoch patience and an initial 20-epoch grace period based on improvements in the validation Macro F1 score.

Baselines. To provide a comprehensive comparison, we benchmark our framework against fourteen models. These baselines span fundamental heuristic predictors (Random, Majority) and established CTDGNs, including memory-centric models [9, 19], those using attention mechanisms or random walks [21, 22, 24, 28], and an MLP-based architecture [3]. Furthermore, we include recent self-supervised approaches designed for dynamic graphs, covering both generative masked autoencoders [5, 12] and contrastive learning methods [1, 25, 36]. All baseline models use text representations computed by Qwen3-Embedding-0.6B [32] to ensure fair comparison with UTGA. Detailed hyperparameter configurations are provided in Appendix B.

## 4.3 Dynamic Edge Classification Performance

## 4.2 Training Procedure

Table 1 reports dynamic edge classification results at the 10,000- label budget. UTGA achieves the highest Macro F1 on every dataset and on average (27.53%), a 22.0% relative gain over the base framework DyGnROLE (22.56%) and 17.2% over the strongest overall baseline, TGAT (23.48%). Since DyGnROLE isolates the efect of text-graph co-evolution, we use it as the primary reference in the discussion below, and we name the best-in-column baseline wherever it is a diferent model. The improvement over the best-incolumn baseline is statistically significant $( p < 0 . 0 1 )$ on six of the eight datasets.

We follow the standard 70-15-15 chronological training-validationtest split defined by DTGB. All methods with a self-supervised pretraining phase first pretrain on the 200,000 most recent interactions of the training split and monitor performance on the 30,000 most recent interactions of the validation set. Consistent with UTGA, early stopping of self-supervised baseline models uses a patience of 10 epochs during pretraining.

On Amazon, Google, and Yelp, UTGA improves over DyGn-ROLE by 50.0%, 26.6%, and 25.4%, respectively. DyGnROLE is also the best-in-column baseline on Google and Yelp. On Amazon the best baseline is CAWN (18.59%), over which UTGA achieves a gain of 33.4%. On Enron, UTGA scores 18.82%, a 76.1% relative improvement over DyGnROLE (10.69%), which is also the best-in-column baseline.

Table 2: Minority class (“useful”) F1 scores on the Stack Exchange datasets. Values show average F1 scores in percent with subscripts denoting ± standard deviation. Bold indicates best performance per dataset. Relative performance gains of UTGA over the base framework DyGnROLE are shown in green superscript.
<table><tr><td>Method</td><td>Stack U</td><td>Stack E</td><td>Avg</td></tr><tr><td>DyGnROLE</td><td> $1 5 . 1 6 _ { \pm 5 . 4 6 }$ </td><td> $2 1 . 7 5 _ { \pm 6 . 2 2 }$ </td><td>18.45</td></tr><tr><td>GraphMixer</td><td> $2 6 . 1 1 _ { \pm 1 3 . 2 2 }$ </td><td> $3 1 . 1 2 _ { \pm 7 . 4 8 }$ </td><td>28.61</td></tr><tr><td>TGAT</td><td> $2 4 . 4 5 _ { \pm 1 . 9 8 }$ </td><td> $3 1 . 6 0 _ { \pm 4 . 2 0 }$ </td><td>28.02</td></tr><tr><td>UTGA</td><td> ${ \bf 2 9 . 5 5 } _ { \pm 0 . 9 0 } ^ { + 9 4 . 9 \% }$ </td><td> ${ \bf 3 3 . 5 9 } _ { \pm 3 . 2 5 } ^ { + 5 4 . 4 \% }$ </td><td> $3 1 . 5 7 ^ { + 7 1 . 1 \% }$ </td></tr></table>

![](images/e07d765210e4010cea645cd2c3622f5b8a071466945a640d76e70f02a05e5cc7.jpg)  
Figure 2: Average Macro F1 performance across the DTGB benchmark as the number of training labels scales from 1k to 20k. UTGA maintains a consistent and substantial advantage over the top three performing baselines (TGAT, GraphMixer, and DyGnROLE) at every data scale.

Stack U and Stack E expose the vulnerability of standard architectures to class imbalance, with the majority class comprising roughly 81% and 75% of interactions, respectively. Several advanced models fail to meaningfully surpass the majority heuristic (44.65% and 42.82% Macro F1). On these datasets, UTGA reaches 55.50% and 56.33%, improving over DyGnROLE by 8.8% and 9.4%. Its margin over the best-in-column baselines (GraphMixer on Stack U, TGAT on Stack E) is narrower (+0.9% and +1.4%). The per-class F1 on the minority “useful” class separates the strongest baselines and UTGA more clearly (Table 2). Here the framework outperforms the best-in-column baselines by 13.2% (Stack U) and 6.3% (Stack E), and DyGnROLE by 94.9% and 54.4%, while exhibiting tighter standard deviations across random seeds.

GDELT and ICEWS are the most challenging settings, with massive label vocabularies and very short text attributes. On GDELT, UTGA reaches 0.63%, a 65.8% improvement over DyGnROLE, which is tied with TCL as the best-in-column baseline (0.38%). On ICEWS, UTGA achieves 2.26%, a 93.2% relative gain over DyGnROLE (1.17%), again the best-in-column baseline, nearly doubling its performance.

![](images/f727b7aefe8564b1eba211c4aa85b385275763966cb72fc8b62a280cc8bcba86.jpg)  
Figure 3: Relationship between log-transformed average node text length and relative Macro F1 performance gain of UTGA over the base framework. The red dashed line represents the Ordinary Least Squares (OLS) fit with a 95% confidence interval band. Markers display dataset-level means with standard error bars.

Overall, these observations indicate that allowing the textual latent space to co-evolve with the graph topology during pretraining yields a superior and more robust foundation for downstream classification across a highly diverse set of text-attributed domains.

Robustness to label availability. To evaluate the robustness of the learned representations under varying degrees of label availability, we scale the finetuning data budget from 1,000 to 20,000 training interactions. Figure 2 plots the average Macro F1 score across all datasets for UTGA and the top three performing baselines: TGAT, GraphMixer, and DyGnROLE. UTGA demonstrates superior sample eficiency, maintaining a dominant performance margin at every scale. Notably, UTGA trained with only 1,000 labels (25.1% Macro F1) outperforms all three baseline models when they are trained with 20,000 labels.

Impact of text length. As visualized in Figure 3, revisiting the text length variations discussed in Section 4.1 reveals an empirical pattern regarding where text-graph co-evolution provides the most value. We observe a strong, statistically significant negative correlation between log-transformed average node text length and relative Macro F1 gain at the aggregated dataset level $( N = 8 ;$ Pearson $r = - 0 . 7 2 6 , p = 0 . 0 4 1 )$ . This analysis suggests that semantic and spatio-temporal co-evolution yields the highest relative performance improvements in environments with short node text.

Because datasets with shorter node texts frequently feature larger label spaces in this benchmark (dataset details in Appendix

Table 3: Ablation results for dynamic edge classification showing average Macro F1 scores (%). Bold indicates best on dataset.
<table><tr><td>UTGA Variants</td><td>Amazon</td><td>Google</td><td>Yelp</td><td>Enron</td><td>Stack U</td><td>Stack E</td><td>GDELT</td><td>ICEWS</td><td>Avg</td></tr><tr><td>w/o Pretraining</td><td> $1 8 . 5 7 \pm 0 . 4 2$ </td><td> $1 5 . 9 3 \pm 0 . 0 0$ </td><td> $3 0 . 2 4 \pm 1 . 3 5$ </td><td> $8 . 8 9 \pm 0 . 7 1$ </td><td> $4 7 . 6 4 \pm 3 . 6 8$ </td><td> $4 7 . 6 0 \pm 5 . 9 0$ </td><td> $0 . 3 2 \pm 0 . 0 5$ </td><td> $1 . 7 7 \pm 0 . 0 8$ </td><td>21.37</td></tr><tr><td>w/o LMM</td><td> $2 3 . 4 0 \pm 1 . 0 2$ </td><td> $2 4 . 0 6 \pm 0 . 3 5$ </td><td> $3 1 . 3 5 \pm 0 . 4 4$ </td><td> $9 . 0 8 \pm 0 . 7 7$ </td><td> $5 4 . 6 6 \pm 1 . 0 6$ </td><td> ${ \bf 5 6 . 5 3 \pm 0 . 6 0 }$ </td><td> $0 . 5 6 \pm 0 . 0 1$ </td><td> $2 . 0 7 \pm 0 . 1 2$ </td><td>25.21</td></tr><tr><td>MLP PLM Adapter</td><td> $2 1 . 9 1 \pm 0 . 2 5$ </td><td> $2 1 . 4 5 \pm 0 . 4 0$ </td><td> $3 4 . 1 7 \pm 0 . 3 5$ </td><td> ${ \bf 1 9 . 2 3 \pm 1 . 1 3 }$ </td><td> $5 3 . 5 2 \pm 1 . 0 5$ </td><td> $5 5 . 6 4 \pm 1 . 0 2$ </td><td> $0 . 4 3 \pm 0 . 0 4$ </td><td> $1 . 2 4 \pm 0 . 0 6$ </td><td>25.95</td></tr><tr><td>UTGA</td><td> $\mathbf { 2 4 . 7 9 \pm 0 . 5 6 }$ </td><td> $\mathbf { 2 5 . 1 3 \pm 0 . 3 2 }$ </td><td> ${ \bf 3 6 . 8 1 \pm 0 . 2 3 }$ </td><td> $1 8 . 8 2 \pm 1 . 1 3$ </td><td> ${ \bf 5 5 . 5 0 \pm 0 . 1 9 }$ </td><td> $5 6 . 3 3 \pm 0 . 7 5$ </td><td> $\mathbf { 0 . 6 3 \pm 0 . 0 3 }$ </td><td> $\mathbf { 2 . 2 6 \pm 0 . 1 7 }$ </td><td>27.53</td></tr></table>

A), we conducted a confounding analysis to ensure this performance gain is not merely an artifact of task dificulty. To separate the efect of text length from the high class counts (200+) of GDELT and ICEWS, we performed a subgroup analysis restricted to the other six datasets, which have 10 or fewer classes. In this controlled setting, the negative relationship intensifies and remains statistically significant at the aggregated dataset level $( N = 6 ;$ Pearson $r = - 0 . 8 6 9 , p = 0 . 0 2 4 )$

## 5 Analysis

In this section, we analyze the specific mechanisms driving the impact of semantic and topological co-evolution. This includes isolating component contributions, measuring semantic anisotropy, and probing the structural expressivity of the text embeddings.

## 5.1 Ablation Study

To isolate the efects of pretraining objectives and the PLM adapter of UTGA, we conduct an ablation study. We evaluate three variants of the architecture: (1) w/o Pretraining, where the DRA and LMM objectives are removed, the model is initialized with random weights for dynamic edge classification, and the PLM Adapter op erates on the token-level Qwen3 embeddings as in the complete model; (2) w/o LMM, where the LMM objective is dropped during pretraining, allowing the full architecture to update solely via structural DRA gradients; and (3) MLP PLM Adapter, where the Transformer-based PLM Adapter is replaced by an MLP-based PLM Adapter, which still adapts the semantic space, but it operates purely on the pooled EOS representations of the texts with an MLP rather than adapting the full token sequence with a Transformer.

The results in Table 3 highlight the necessity of the joint pretraining framework. Removing pretraining entirely causes the average Macro F1 to fall from 27.53% to 21.37%, confirming the benefits of initializing with a PLM Adapter that encodes structural and temporal properties into the text embeddings. Removing the LMM objective results in a drop to 25.21%. Without this anchor, the structural gradients from DRA optimize solely for topology, which limits the semantic utility of the representations and reduces downstream classification performance. Substituting the Transformer for an MLP in the PLM Adapter drops overall performance to 25.95%, demonstrating that adapting the full sequence of text tokens generally provides a better alignment than adapting a single pooled sentence representation. While the complete UTGA framework scores marginally lower than MLP PLM Adapter on Enron and w/o LMM on Stack E, these specific variations fall within the standard deviations of the complete model (± 1.13 and ± 0.75, respectively), indicating no evidence of a diference on these datasets rather than a systematic disadvantage.

![](images/becdb5a1b14c5fc2d16a2e4045165567d5787a252241d482f21c261ceb2d01bd.jpg)  
Figure 4: Average pairwise cosine distance for edge text embeddings. The representations of the raw Qwen3-Embedding-0.6B exhibit high levels of anisotropy, whereas the UTGA framework efectively reduces this collapse to expand the embedding space.

## 5.2 Anisotropy Reduction

General-purpose language models frequently map diverse texts into a narrow, anisotropic cone, causing even semantically unrelated texts to exhibit artificially low cosine distances [4, 10]. This geometric homogeneity makes it dificult for downstream graph encoders to diferentiate distinct interactions. To evaluate how our framework mitigates this degradation, we compute the average ofdiagonal pairwise cosine distance for uniformly sampled subsets of up to 10,000 test-split edges using Qwen3-Embedding-0.6B and the UTGA framework (pretrained, but not finetuned). As shown in Figure 4, raw Qwen3 embeddings exhibit severe anisotropy, with average distances frequently around 0.1. The UTGA framework effectively mitigates this by increasing cosine distances substantially across all datasets.

To confirm the anisotropy reduction provided by UTGA yields more discriminable features, we probe the linear separability of the frozen edge representations of the e-commerce datasets by predicting 1-5 star review ratings using the same training and validation data sampled for pretraining and evaluating on the full test splits. The results in Table 5 show that the narrow geometry of the raw Qwen3 embeddings restricts the spatial resolution required to separate sentiments, limiting performance. By expanding the embedding space while preserving semantic integrity via LMM, UTGA constructs representations that are both highly discriminable and linguistically meaningful, achieving substantially higher linear separability across all three datasets.

Table 4: Node structural probing analysis on GDELT comparing the linear separability (Macro F1 %) of node text embeddings on graph structural properties. Bold indicates best performance per task.
<table><tr><td>Embedding</td><td>In-Degree</td><td>Out-Degree</td><td>PageRank</td><td>Clustering</td><td>Lifespan</td></tr><tr><td>Random</td><td> $2 5 . 0 9 \pm 0 . 5 8$ </td><td> $2 4 . 0 9 \pm 0 . 4 6$ </td><td> $2 5 . 3 1 \pm 0 . 6 5$ </td><td> $3 3 . 0 4 \pm 0 . 8 2$ </td><td> $2 1 . 2 9 \pm 0 . 3 5$ </td></tr><tr><td>Qwen3-Embedding</td><td> $2 3 . 7 5 \pm 0 . 0 8$ </td><td> $2 3 . 9 3 \pm 3 . 5 3$ </td><td> $2 7 . 6 9 \pm 0 . 3 5$ </td><td> $3 2 . 9 0 \pm 4 . 8 3$ </td><td> $2 2 . 4 3 \pm 0 . 0 0$ </td></tr><tr><td>UTGA</td><td> $\mathbf { 3 0 . 0 2 \pm 3 . 9 1 }$ </td><td> $\mathbf { 2 9 . 5 5 \pm 1 . 8 8 }$ </td><td> ${ \bf 3 8 . 8 5 \pm 0 . 6 8 }$ </td><td> ${ \bf 4 4 . 8 6 \pm 6 . 0 5 }$ </td><td> $\mathbf { 2 7 . 0 3 \pm 1 . 5 4 }$ </td></tr></table>

Table 5: Edge probing analysis comparing the linear separability (Macro F1 %) of edge text embeddings on e-commerce datasets. Bold indicates best performance per dataset.
<table><tr><td>Embedding</td><td>Amazon</td><td>Google</td><td>Yelp</td></tr><tr><td>Random</td><td> $1 5 . 2 7 \pm 0 . 0 4$ </td><td> $1 5 . 0 3 \pm 0 . 0 5$ </td><td> $1 6 . 9 5 \pm 0 . 0 3$ </td></tr><tr><td>Qwen3-Embedding</td><td> $1 8 . 9 9 \pm 0 . 1 3$ </td><td> $2 3 . 7 9 \pm 0 . 0 5$ </td><td> $2 4 . 2 2 \pm 0 . 2 3$ </td></tr><tr><td>UTGA</td><td> ${ \bf 3 6 . 7 1 \pm 0 . 5 2 }$ </td><td> ${ \bf 3 5 . 6 9 \pm 0 . 5 3 }$ </td><td> $\mathbf { 4 4 . 2 8 \pm 0 . 5 6 }$ </td></tr></table>

## 5.3 Topological Alignment

A central hypothesis of our investigation is that the UTGA framework encodes network topology into the text embeddings. We evaluate the structural expressivity of the learned representations through a series of node text embedding probing tasks. Using the GDELT temporal knowledge graph as a case study, we extract five fundamental graph-theoretic properties for each node: in-degree, out-degree, PageRank centrality [14], temporal lifespan, and local clustering density. The first four are divided into four quartile tiers while local clustering density is categorized into three classes: zero local triangles, below the non-zero median, and at or above the non-zero median. The discretization thresholds are calculated on the 200,000-interaction pretraining split. We train a single linear layer classifier on the frozen node text embeddings using the same training and validation data sampled for pretraining. Evaluation is performed on the full test splits.

The results of the probing analysis are presented in Table 4. As expected, the raw Qwen3 features lack structural awareness, yielding classification scores near or below random guessing on almost every task. When the representations are updated through UTGA, performance improves on all tasks. The framework yields substantially higher linear separability on PageRank centrality and local clustering density, indicating that the adapted representations encode both global network positioning and local topological motifs.

## 5.4 Computational Eficiency

To quantify the overhead of our framework, we compare UTGA to the base framework (DyGnROLE) and a joint text-graph training baseline that replaces the PLM Adapter with Low-Rank Adapta tion (LoRA) [6] on the full Qwen3-Embedding-0.6B backbone. As reported in Table 6, joint training with LoRA restricts throughput to 28.53 samples/s (4,485.74 ms per batch) and raises peak GPU memory allocation to 4,335.87 MB.

In contrast, with 2.03M additional trainable parameters over DyGnROLE (3.58M total), UTGA reaches a throughput of 1,600.62 samples/s (79.76 ms per batch). This yields a 56× speedup and an 88.8% memory reduction (485.34 MB) compared to LoRA. While slower than static feature processing (21.24 ms per batch), UTGA enables dynamic semantic-structural co-evolution within the same memory footprint order of magnitude. Details regarding our computing hardware are provided in Appendix D.

Table 6: Computational eficiency on the Enron dataset. Metrics include trainable parameters (Params), average per-batch execution time in milliseconds (Time), overall throughput (Samples/s), and peak GPU memory allocation in megabytes (Mem) during pretraining.
<table><tr><td>Architecture</td><td>Params</td><td>Time</td><td>Samples/s</td><td>Mem</td></tr><tr><td>DyGnROLE</td><td>1.55M</td><td> $2 1 . 2 4 \pm 6 . 2 1$ </td><td>5980.29</td><td>171.45</td></tr><tr><td>UTGA w/ LoRA</td><td>6.86M</td><td> $4 4 8 5 . 7 4 \pm 1 7 4 3 . 9 1$ </td><td>28.53</td><td>4335.87</td></tr><tr><td>UTGA</td><td>3.58M</td><td> $7 9 . 7 6 \pm 1 4 . 6 1$ </td><td>1600.62</td><td>485.34</td></tr></table>

## 6 Conclusion

In this work, we propose a unified text-graph alignment (UTGA) framework to investigate the efects of enabling the co-evolution of text representations and network topology in dynamic textattributed graphs. With this framework, we demonstrate that updating token-level text representations via spatio-temporal gradients consistently improves dynamic edge classification performance over standard static-feature baselines, particularly while simultaneously anchoring them with a language model maintenance objective to prevent detrimental semantic distortion. Through representational analyses, we revealed that UTGA reduces anisotropy, improving the linear separability of the embeddings, and efectively encodes structural and temporal properties into the semantic space. Crucially, a computational eficiency analysis confirms that text-graph co-evolution via UTGA is achieved without incurring the prohibitive computational overhead typical of end-to-end joint training.

## References

[1] Tyler Bonnet and Marek Rei. 2026. DyGnROLE: Modeling Asymmetry in Dynamic Graphs with Node-Role-Oriented Latent Encoding. arXiv preprint arXiv:2602.23135.

[2] Eli Chien, Wei-Cheng Chang, Cho-Jui Hsieh, Hsiang-Fu Yu, Jiong Zhang, Olgica Milenkovic, and Inderjit S. Dhillon. 2022. Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction. In International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=KJggliHbs8

[3] Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. 2023. Do We Really Need Complicated Model Architectures For Temporal Networks?. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=ayPPc0SyLv1

[4] Kawin Ethayarajh. 2019. How Contextual are Contextualized Word Represen tations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 55–65.

[5] Mengzhou Gao, Xinxun Zhang, Pengfei Jiao, Tianpeng Li, and Zhidong Zhao. 2025. DVGMAE: Self-Supervised Dynamic Variational Graph Masked Autoen coder. IEEE Transactions on Neural Networks and Learning Systems (2025).

[6] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9

[7] Xuanwen Huang, Kaiqiao Han, Yang Yang, Dezheng Bao, Quanjin Tao, Ziwei Chai, and Qi Zhu. 2024. Can GNN be Good Adapter for LLMs?. In Proceedings ofthe ACM Web Conference 2024. 1–12. https://dl.acm.org/doi/10.1145/3589334. 3645627

[8] Bowen Jin, Wentao Zhang, Yu Zhang, Yu Meng, Xinyang Zhang, Qi Zhu, and Jiawei Han. 2023. PATTON: Language Model Pretraining on Text-Rich Networks. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Toronto, Canada, 7027–7043. https://aclanthology.org/2023.acl-long.387

[9] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, Anchorage, AK, USA, 1269–1278.

[10] Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. 2020. On the Sentence Embeddings from Pre-trained Language Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 9119–9130.

[11] Shikun Liu, Edward Johns, and Andrew J. Davison. 2019. End-To-End Multi-Task Learning With Attention. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 1871–1880. doi:10.1109/CVPR.2019.00197

[12] Weixiong Liu, Junwei Cheng, Quanlong Guan, Zhongyu Pan, and Chaobo He. 2025. DyGMAE: A Novel Dynamic Graph Masked Autoencoder for Link Prediction. In UAI.

[13] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

[14] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. The PageRank citation ranking: Bringing order to the web. Technical Report. Stanford InfoLab.

[15] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. 2020. Temporal Graph Networks for Deep Learning on Dynamic Graphs. In ICML 2020 Workshop on Graph Representation Learning and Beyond.

[16] Amit Roy, Ning Yan, and Masood Mortazavi. 2025. LLM-driven knowledge distillation for dynamic text-attributed graphs. arXiv preprint arXiv:2502.10914 (2025).

[17] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29, 3 (2008), 93–106.

[18] Noam Shazeer. 2020. GLU variants improve transformer. arXiv preprint arXiv:2002.05202.

[19] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. 2019. DyRep: Learning representations over dynamic graphs. In International Conference on Learning Representations (ICLR).

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[21] Lu Wang, Xiaofu Chang, Shuang Li, Yunfei Chu, Hui Li, Wei Zhang, Xiaofeng He, Le Song, Jingren Zhou, and Hongxia Yang. 2021. TCL: Transformer-based dynamic graph modelling via contrastive learning. arXiv preprint arXiv:2105.07944.

[22] Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. 2021. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. In International Conference on Learning Representations (ICLR).

[23] Yunan Wang, Jianxin Li, and Ziwei Zhang. 2026. Global-Recent Semantic Reasoning on Dynamic Text-Attributed Graphs with Large Language Models. In The Fourteenth International Conference on Learning Representations. https://openreview.net/forum?id=puocvrFZRl

[24] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2020. Inductive Representation Learning on Temporal Graphs. In International Conference on Learning Representations (ICLR).

[25] Yifei Xu, Bowen Shi, Tengfei Ma, Bo Dong, Hongyu Zhou, and Qing Zheng. 2023. CLDG: Contrastive Learning on Dynamic Graphs. In 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 696–707.

[26] Yuanyuan Xu, Wenjie Zhang, Ying Zhang, Xuemin Lin, and Xiwei Xu. 2026. Unlocking multi-modal potentials for link prediction on dynamic text-attributed graphs. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 27386–27394.

[27] Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, and Edward Y Chang. 2015. Network representation learning with rich text information. In IJCAI, Vol. 2015. 2111–2117.

[28] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. 2023. Towards better dynamic graph learning: New architecture and unified library. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. 67686–67700.

[29] Biao Zhang and Rico Sennrich. 2019. Root Mean Square Layer Normalization. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019). 12381– 12392.

[30] Jiasheng Zhang, Jialin Chen, Menglin Yang, Aosong Feng, Shuang Liang, Jie Shao, and Rex Ying. 2024. DTGB: A comprehensive benchmark for dynamic text-attributed graphs. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 37. Curran Associates, Inc., 91405–91429.

[31] Siwei Zhang, Yun Xiong, Yateng Tang, Jiarong Xu, Xi Chen, Zehao Gu, Xuehao Zheng, Zi’an Jia, and Jiawei Zhang. 2025. Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=9env0BdcDV

[32] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

[33] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. 2023. Learning on Large-scale Text-attributed Graphs via Variational Inference. In International Conference on Learning Representations (ICLR). https: //openreview.net/forum?id=q0nmYciuuZN

[34] Xi Zhu, Haochen Xue, Ziwei Zhao, Wujiang Xu, Jingyuan Huang, Minghao Guo, Qifan Wang, Kaixiong Zhou, Imran Razzak, and Yongfeng Zhang. 2025. LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models. In The Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=1FY1apsMxc

[35] Yun Zhu, Haizhou Shi, Xiaotang Wang, Yongchao Liu, Yaoke Wang, Boci Peng, Chuntao Hong, and Siliang Tang. 2025. GraphCLIP: Enhancing transferability in graph foundation models for text-attributed graphs. In Proceedings of the ACM on Web Conference 2025. ACM, 2183–2197.

[36] Zhenyu Zhu, Kunpeng Wang, Huan Liu, Jia Li, and Shijun Luo. 2024. Topology-Monitorable Contrastive Learning on Dynamic Graphs. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, 4700–4711.

## A Dataset Details

Table 7: Summary statistics of the DTGB datasets.
<table><tr><td>Dataset</td><td>#Nodes</td><td>#Edges</td><td>#Classes</td><td>#Words</td></tr><tr><td>Amazon</td><td>293,566</td><td>3,217,324</td><td>5</td><td>623,447,701</td></tr><tr><td>Enron</td><td>42,711</td><td>797,907</td><td>10</td><td>187,576,324</td></tr><tr><td>GDELT</td><td>6,786</td><td>1,339,245</td><td>236</td><td>7,632,528</td></tr><tr><td>Google</td><td>111,168</td><td>1,380,623</td><td>5</td><td>74,401,703</td></tr><tr><td>ICEWS</td><td>31,796</td><td>1,100,071</td><td>266</td><td>27,003,734</td></tr><tr><td>Stack E</td><td>397,702</td><td>1,262,225</td><td>2</td><td>340,401,237</td></tr><tr><td>Stack U</td><td>674,248</td><td>1,497,006</td><td>2</td><td>347,878,722</td></tr><tr><td>Yelp</td><td>2,138,242</td><td>6,990,189</td><td>5</td><td>987,141,388</td></tr></table>

Enron<sup>3</sup>: Emails between Enron employees from 1999 to 2002.

• Nodes: Employees

• Node texts: Professional email addresses

• Edges: Email exchanges between employees

• Edge texts: Raw email content

• Edge labels: 10 distinct communication categories (e.g., ’deal’, ’calendar’)

GDELT<sup>4</sup>: Global Database of Events, Language, and Tone knowledge graph representing geopolitical events.

![](images/2109033c8191369e0be2a6a6065b69269434bfe38c52e57c8cbc82eda1fa6580.jpg)

Figure 5: Label distributions of the DTGB datasets.  
![](images/2bfd84b57fe1f2d30d81fe1c4a932b343528ec1b09a0c34bbefa8adeef206d4b.jpg)

![](images/479ae2c581e9aae631502ecf331c08f5fcc0e783f67d545f21df2c1afe5fb898.jpg)  
Figure 6: Average node and edge text lengths of the DTGB datasets.

• Nodes: Political entities or actors

• Node texts: Entity names

• Edges: Relationships or actions between two political entities

• Edge texts: Concise descriptions of the relationship

• Edge labels: 236 unique relationship types (e.g., ’provide economic aid’)

ICEWS1819 (ICEWS)<sup>5</sup>: Integrated Crisis Early Warning System knowledge graph representing geopolitical events.

• Nodes: Political entities

• Node texts: Entity names

• Edges: Documented relationships between political actors

• Edge texts: Short descriptions of the event or relationship

• Edge labels: 266 unique interaction categories (e.g., ’Engage in negotiation’)

Stack elec (Stack E)<sup>6</sup>: Stack Exchange forum interactions focused on electronics and electrical engineering.

• Nodes: Users or specific questions

• Node texts: User introduction and location: question title and post content

• Edges: The submission of an answer or a comment on a post

• Edge texts: Textual content of answers or comments

• Edge labels: Binary classification (’useful’ or ’useless’) based on a voting score where ����� = �������−���������. An edge is ’useful’ if ����� > 1.

Stack ubuntu (Stack U)<sup>7</sup>: Stack Exchange forum interactions focused on the Ubuntu operating system.

• Nodes: Users or questions

• Node texts: User introduction and location: question title and post content

• Edges: Answer submissions or comments on existing posts

• Edge texts: Textual content of answers or comments

• Edge labels: Binary classification (’useful’ or ’useless’) where ����� = ������� − ��������� and ’useful’ requires ����� > 1.

Googlemap CT (Google)<sup>8</sup>: Reviews for businesses in the state of Connecticut on Google.

• Nodes: Users or local businesses

• Node texts: Usernames for individuals: business name, address, category, and introduction

• Edges: Review interactions between users and businesses

• Edge texts: The textual content of user reviews

• Edge labels: 1 to 5 based on the standard five-star rating system

Amazon movies (Amazon)<sup>9</sup>: Reviews for the Movies and TV category on Amazon.

• Nodes: Users or products in the Movies and TV category

• Node texts: Usernames for individuals: product title, description, category, and rank score

• Edges: Reviews submitted by users for specific products

• Edge texts: The textual content of user reviews

• Edge labels: 1 to 5 based on the five-star rating system Yelp<sup>10</sup>: Reviews for businesses on Yelp.

• Nodes: Users or businesses

• Node texts: Name, register date, review count, and average rating for users: business name, city, address, and category

• Edges: Review interactions between users and businesses

• Edge texts: The textual content of user reviews

• Edge labels: 1 to 5 based on the five-star rating system

## B Model Hyperparameters

## Shared Settings

• Optimizer: AdamW

• Maximum number of neighbors (�): 10

• Gradient clipping max norm: 1.0

• Max training epochs: 1000

• Pretraining random seed: 42

• Pretraining early stopping patience: 10 epochs

• Finetuning random seeds: 0, 1, 2, 3, 4

• Finetuning early stopping patience: 20 epochs

• PyTorch automatic mixed precision FP16

## UTGA

• Weight decay: 0.01

• Batch size: 128

• Pretraining learning rate: $5 \times 1 0 ^ { - 5 }$

• Finetuning learning rate: $3 \times 1 0 ^ { - 5 }$

• Token feature dimension: 256

• Maximum tokens per text: 256

• PLM Adapter layers: 2

• PLM Adapter heads: 4

• PLM Adapter dropout rate: 0.1

• PLM Adapter SwiGLU hidden dimension: 768

• PLM Adapter RMSNorm �: 10<sup>−6</sup>

• Transformer encoder layers: 2

• Transformer encoder heads: 2

• Transformer encoder dropout rate: 0.1

• Transformer encoder feed-forward dimension: 1000

• Channel dimension: 50

• Time dimension: 100

• Number of channels: 5

• Maximum number of neighbors (�): 10

• LMM projection head: 256 → 512 → 256, dropout 0.1

• DWA temperature: 2.0

• DWA warm-up epochs: 2

## DyGnROLE

• Transformer encoder layers: 2

• Transformer encoder heads: 2

• Transformer encoder dropout rate: 0.1

• Channel dimension: 50

• Time dimension: 100

• Occurrence embedding dimension: 50

• Number of channels: 5

• InfoNCE temperature (�): 0.07

## CLDG

• InfoNCE temperature: 0.07

• Hidden dimension: 128

• Embedding dimension: 64

• Encoder layers: 2

• Dropout rate: 0.1

• Number of snapshots: 5

## IDOL

• Hidden dimension: 512

• Embedding dimension: 128

• Neighborhood samples (�): 4

• Restart probability (�): 0.1

• Maximum bias $( r _ { m a x } ^ { b } ) : 0 . 0 1$

• Loss trade-of (�): 0.01

• Validation temperature: 0.07

## DVGMAE

• Hidden dimension: 128

• Embedding dimension: 32

• Dropout rate: 0.1

• Epsilon (�): 0.25

• Mask ratio: 0.5

• Loss lambda weight (�): 0.5

• Number of snapshots: 5

## DyGMAE

• Hidden dimension: 128

• Encoder layers: 2

• Dropout rate: 0.1

• Number of masks: 2

• Mask ratios: 0.2, 0.4

• Mask types: path, random

• Fusion type: attention

• Walk length: 2

• Loss gamma (�): 0.1

• Loss delta (�): 0.1

• Loss lambda � (�<sub>�</sub>): 0.1

• Number of snapshots: 5

## TGAT

• Layers: 2

• Heads: 2

• Dropout rate: 0.1

• Time dimension: 100

• Maximum number of neighbors (�): 10

## DyRep

• Memory dimension: 128

• Mailbox size: 10

• Feature projection: 256 → 128

• Memory reset: each epoch

## JODIE

• Memory dimension: 128

• Recurrent unit: single-layer RNN cell

• Time dimension: 128

• Dropout rate: 0.1

• Memory reset: each epoch

## GraphMixer

• Mixer layers: 2

• Number of tokens: 10

• Channel dimension: 256

• Time dimension: 100 (fixed encoder)

• Token expansion factor: 0.5

• Channel expansion factor: 4.0

• Dropout rate: 0.5

## TCL

• Transformer encoder layers: 2

• Transformer encoder heads: 2

• Transformer encoder dropout rate: 0.1

• Hidden dimension: 256

• Feed-forward dimension: 1024

• Time dimension: 100

• Number of depth embeddings: 11

• Maximum number of neighbors (�): 10

## CAWN

• Walk length: 2

• Neighbors per hop: 10

• Attention heads: 8

• Dropout rate: 0.1

• Time dimension: 100

• Position encoding dimension: 172

• Feature BiLSTM hidden dimension: 392

• Position BiLSTM hidden dimension: 86

## DyGFormer

• Transformer encoder layers: 2

• Transformer encoder heads: 2

• Transformer encoder dropout rate: 0.1

• Transformer encoder feed-forward dimension: 800

• Channel dimension: 50

• Time dimension: 100

• Neighbor co-occurrence dimension: 50

• Maximum sequence length (�): 10

## C Text Embedding Extraction

The text attributes are embedded at the token level using Qwen3- Embedding-0.6B [32] and truncated to 256 dimensions to balance representational capacity with computational and memory eficiency. Across all datasets, the active token cache comprises 607.2M tokens (289.6 GiB in FP16 at D = 256), with text sequences capped at a maximum of 256 tokens. The embeddings are stored in a ragged CSR binary format, discarding all padding tokens. Feature extraction required 10 GPU-hours. Details regarding our computing hardware are provided in Appendix D.

## D Computing Hardware

All experiments were performed using an Intel Xeon Silver 4214 CPU (2.20GHz), 128 GB of system RAM, and four NVIDIA GeForce RTX 2080 Ti GPUs (11 GB VRAM each).

## E Limitations and Future Work

While our investigation establishes the empirical benefits of anchored text-graph co-evolution for a diverse set of dynamic edge classification tasks, extending this framework to link prediction and node classification remains a valuable direction for future research. A compelling question regarding link prediction is whether a dedicated pretraining phase remains advantageous, given that the structural supervision signal $( \mathrm { i . e . , }$ the existence ofedges) is naturally abundant, unlike the often limited annotations typical of edge and node classification. Consequently, subsequent studies could contrast the eficacy of the two-stage pretraining paradigm against an end-to-end framework where the PLM Adapter and graph encoder are jointly optimized from random initialization using only the downstream objective, as we have done with our “w/o Pretraining” ablation.