# HOPE: Heterophily-Aware Open-Set Node Classification with Pseudo-Extrapolation

Yumeng Dai<sup>1</sup>, Yue Tan<sup>2</sup>, Yixin Liu<sup>2</sup>, Chenxu Wang<sup>1,\*</sup>, Pinghui Wang<sup>1</sup>, Tao Qin<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University, Xi’an, China

<sup>2</sup>Griffith University, Brisbane, Australia

ymdai@stu.xjtu.edu.cn, yue.tan@griffith.edu.au, yixin.liu@griffith.edu.au,

cxwang@mail.xjtu.edu.cn, phwang@mail.xjtu.edu.cn, qin.tao@mail.xjtu.edu.cn

Corresponding author: Chenxu Wang

Abstract—Standard open-set node classification methods heavily rely on the homophily assumption, where connected nodes share identical labels. However, real-world graphs are often heterophilic, revealing the sub-optimal performance of current methods and posing new challenges to open-set node classification. On the one hand, the cross-class connectivity nature of heterophilic graphs causes node representations from different known or unknown classes to be intertwined after aggregation, undermining the discriminative capacity of learned representations. On the other hand, the structural mixture invalidates traditional threshold-based open-set classification methods and breaks conventional cross-class feature interpolation paradigms, leading to unreliable unknown-class rejection. To address these critical challenges, we propose a novel framework named HOPE Heterophily-aware Open-set node classification method with Pseudo-Extrapolation, abbreviated as HOPE. To adapt open-set graph neural networks (GNNs) to heterophilic scenarios, HOPE utilizes a structure-augmented feature initialization layer to capture multi-hop structural patterns for feature augmentation. Meanwhile, we design a trustworthy neighborhood aggregation mechanism adaptable to standard GNNs to dynamically filter out noisy cross-class neighbors. To enhance the unknown-class rejection capability of HOPE, we introduce a heterophily-guided pseudo-extrapolation strategy. It dynamically maintains the representations of known-class centers and extrapolates from these centers along cross-class neighborhood displacement directions, thereby synthesizing pseudo-unknown proxies near structurally ambiguous regions. Finally, we optimize the network via a joint classification framework with logit margin regularization, routing synthetic proxies into a dedicated rejection slot without imposing geometric margin constraints in the representation space. Extensive experiments on multiple datasets demonstrate that HOPE consistently outperforms state-of-the-art models, validating its effectiveness, robustness, and efficiency.

Index Terms—Graph Neural Networks, Open-Set Node Classification, Graph Heterophily, Pseudo-Unknown Proxies.

## I. INTRODUCTION

Graph Neural Networks (GNNs) have achieved remarkable success in various graph-structured tasks, particularly in semisupervised node classification [1]. Standard node classification methods operate under a closed-set assumption, where the training and testing environments share an identical label space. However, real-world graph database applications often encounter open-set scenarios [2]–[4], where test graphs contain nodes from previously unseen or unknown categories. When conventional closed-set methods face these unknown nodes, they often mistakenly force them into one of the known classes with high confidence, neglecting the presence of novel categories beyond the predefined label space. To address this limitation, open-set node classification has emerged as a crucial research problem, aiming to classify nodes from known categories while identifying nodes from previously unseen categories [2], [5]. By jointly modeling known-class discrimination and unknown-class rejection, open-set node classification methods can support both accurate known-class identification and robust unknown-class detection.

![](images/e4b0e7f57bcb87105afcc72c85b1bcf15cb4a1ad5bbc38ca3e1faaff015595bf.jpg)  
(a) Unknown classes in homophilic graphs

![](images/b6c46a3400013f524d3f410a8f185a8cdc0487b0ddcb1b930c1c710f265c3ef7.jpg)  
(b) Unknown categories in heterophilic graphs  
Fig. 1: The distinct topological behaviors of unknown class nodes in homophilic versus heterophilic graphs.

Despite the progress of open-set node classification in recent years, existing methods often assume that graphs are homophilic, meaning that connected nodes tend to share the same class label [6]. This assumption may not always hold in practi cal scenarios, since heterophilic graphs, which break the above homophily assumption, are ubiquitous in practical domains, such as e-commerce networks, social networks, and molecular structures [6]–[11]. Due to their distinct neighborhood-label correlations, the structural distribution of unknown classes exhibits fundamental differences between homophilic and heterophilic settings. As shown in Fig. 1a, unseen categories in homophilic graphs are characterized by dense localized clustering with limited cross-boundary expansion. Conversely, as depicted in Fig. 1b, unknown nodes in heterophilic graph manifest as highly intertwined topological neighbors, generating extensive and complex mixtures that deeply blur the classification boundaries of adjacent known class manifolds. As a result, existing open-set node classification methods often struggle on heterophilic graphs, as their homophilyoriented mechanisms, such as label propagation [12], [13] and consistency regularization [14], [15], may over-smooth naturally dissimilar neighbors and misinterpret normal interclass edges as label noise. Moreover, the boundary-based proxy generation strategies (e.g., Mixup) used in some openset classification approaches [16], [17] may scatter pseudounknown nodes across the feature space in heterophilic graphs, thereby blurring the distinction between known and unknown classes. These limitations motivate a fundamental question:

## Can we develop an open-set graph learning approach that can identify unknown nodes in heterophilic graphs?

Answering the above question is non-trivial, as the structural discrepancies between homophilic and heterophilic graphs introduce two key challenges for open-set node classification. Challenge 1: Representation boundary confusion under heterophilic aggregation. In a heterophilic graph, a known-class node is surrounded by neighbors from different known classes and potential unknown classes [18], [19]. In this case, standard neighborhood aggregation, which acts as a low-pass filter, may smooth out essential high-frequency distinctive features. When open-set nodes are mixed into these neighborhoods, uniform aggregation can cause catastrophic representation overlap, making known-class discrimination and unknown-class rejection mutually entangled. As a result, the model struggles to separate different known classes in the representation space, since the neighbor label distributions are highly noisy and complex, leading to ambiguous decision boundaries among the known classes.

While the above challenge focuses on maintaining separable known-class representations, open-set learning further requires reliable rejection of nodes beyond the known label space, leading to Challenge 2: Unreliable unknown-class rejection under structural entanglement. In heterophilic graphs, unknown nodes are structurally intertwined with the clusters from known classes through cross-class connections, rendering existing rejection criteria for open-set node classification ineffective. Specifically, threshold-based rejection methods [2], [20], [21] may misinterpret heterophilic connections as unknown evidence, leading to both false positives and false negatives. Meanwhile, interpolation-based proxy generation methods [16], [17] may also become unreliable, as simple feature interpolation between known classes cannot faithfully capture the complex cross-class distributions and topological boundaries induced by heterophily. Consequently, the generated proxies may be scattered across the feature space, further misleading the model in separating unknown nodes from known ones. In this case, a dedicated open-set node classification framework is needed to detect unknown nodes under such structurally entangled heterophilic settings.

To address these challenges, we propose a novel

Heterophily-aware Open-set node classification method with Pseudo-Extrapolation (HOPE for short). HOPE consists of four components designed to support known-class node classification and unknown-class node rejection simultaneously. Specifically, to address Challenge 1, we develop a structureaugmented feature initialization layer that computes multi-hop geometric descriptors to enrich raw node attributes, ensuring the model recognizes latent open-set boundaries before message passing. Meanwhile, we implement a trustworthy neighborhood aggregation scheme that uses an edge discriminator to evaluate homophily compatibility scores and dynamically prunes untrustworthy cross-class neighbors. Instead of relying on a specific GNN backbone, the proposed aggregation scheme can be applied to various standard GNN models in a plug-and-play manner, enhancing their ability to preserve discriminative known-class representations under heterophily. To handle Challenge 2, we introduce a pseudo-unknown proxy generation strategy based on structural extrapolation. This strategy extrapolates from known-class anchors along heterophilic neighborhood directions to synthesize representative pseudo-unknown boundary instances, providing informative supervision for learning reliable known-unknown rejection boundaries. Moreover, we optimize the network via a unified classification framework integrated with a logit margin regularization loss. By routing the synthetic proxies to a dedicated (K + 1)-th classification slot, HOPE establishes a robust classification-driven shield around known categories without relying on explicit geometric margin constraints in the representation space. The primary contributions of this paper are summarized as follows:

• Problem: We formalize the problem of open-set node classification under heterophily and investigate the topological distribution differences of unknown classes between homophilic and heterophilic graphs.

• Methodology: We propose HOPE, which incorporates a structure-augmented initialization layer, a trustworthy neighbor filtering mechanism adaptable to standard GNNs, a heterophily-guided pseudo-extrapolation strategy, and a unified (K + 1)-way classification framework with logit regularization.

• Experiments: Extensive evaluations on multiple heterophilic graph benchmarks demonstrate that HOPE significantly outperforms state-of-the-art closed-set and open-set models, showcasing superior generalizability and robustness.

## II. RELATED WORK

## A. Graph Neural Networks under Heterophily

Most foundational Graph Neural Networks (GNNs), such as GCN [1] and GAT [22], assume graph homophily, where connected nodes share similar labels or features [6]. However, these models fail on heterophilic graphs where links connect nodes from different classes, a pattern common in e-commerce and social networks [6]–[9], [23], [24].

To capture heterophilic graph structures, various messagepassing and spectral mechanisms have been designed. For instance, H2GCN [9] separates ego-embeddings from neighborhood representations, Geom-GCN [25] maps topologies into geometric spaces, and JK-Net [26] aggregates layers dynamically. Additionally, GPR-GNN [27] utilizes generalized PageRank for adaptive filtering, GCNII [28] incorporates initial residuals, EG-GCN [18] employs edge discriminators, and DiRW [29] uses path-aware random walks. Yet, these methods assume a fixed and fully known label space during training, causing them to fail when encountering unknown classes at test time.

## B. Open-Set Node Classification on Graphs

Open-set node classification addresses this by identifying unknown semantic classes during inference [2], [30], [31], primarily through threshold-based calibration or generative proxy modeling. Threshold-based frameworks like OpenWGL [2] utilize rejection metrics on softmax confidence or uncertainty scores [20], [21], but deep networks often output overconfident scores for unknown samples [32]. Generative proxy modeling methods, such as G2Pxy [16] and EGonc [17], instead introduce virtual open-set nodes via hiddenlayer manifold mixup or energy-based density optimization to simulate external distributions.

Nevertheless, existing open-set methods strongly depend on structural homophily, assuming unknown categories always appear as localized, tight groups. Under severe heterophily, known and unknown nodes connect tightly, causing standard propagation to blur semantic spaces and create severe overlap at classification boundaries. To address this, HOPE introduces a classification-driven structural extrapolation mechanism optimized within local batches. By mapping synthetic boundary proxies to a dedicated (K + 1)-th classification slot, this framework maintains high closed-set accuracy while ensuring adaptive open-set node rejection on heterophilic graphs.

## III. PRELIMINARIES

In this section, we present the formal definitions of graph concepts, heterophily, and the formulation of the open-set node classification task on heterophilic graphs.

## A. Graph Definitions and Heterophily

Let $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } , \mathbf { X } )$ denote an attributed graph, where $\mathcal { V } ~ = ~ \{ v _ { 1 } , v _ { 2 } , . . . , v _ { N } \}$ represents the set of N nodes, and $ { \mathcal { E } } \subseteq  { \mathcal { V } } \times  { \mathcal { V } }$ represents the set of edges. The topological structure of G can be uniquely represented by an adjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { N \times N }$ , where $\mathbf { A } _ { i j } = 1$ if there exists an edge $( v _ { i } , v _ { j } ) \in \mathcal { E }$ , and $\mathbf { A } _ { i j } = 0$ otherwise. Each node $v _ { i } \in \mathcal V$ is associated with a D-dimensional feature vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { D }$ and the collective features of all nodes form the node attribute matrix $\mathbf { X } = [ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$ . The neighborhood of a node $v _ { i }$ is defined as $\mathcal { N } ( v _ { i } ) = \{ v _ { j } \in \mathcal { V } \mid ( v _ { i } , v _ { j } ) \in \mathcal { E } \}$

The connection patterns between nodes in a graph can be quantified by homophily and heterophily. Formally, given a fully labeled graph where each node $v _ { i }$ has a class label $y _ { i }$ the edge homophily ratio h is defined as the proportion of edges that connect nodes sharing the same label:

$$
h = \frac { \sum _ { ( v _ { i } , v _ { j } ) \in \mathcal { E } } \mathbb { I } ( y _ { i } = y _ { j } ) } { | \mathcal { E } | } ,\tag{1}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. A graph is conventionally categorized as a homophilic graph when h is close to 1, implying that “like attracts $\mathrm { l i k } \mathbf { e } ^ { \mathbf { \prime } \mathbf { \prime } }$ . Conversely, a graph is designated as a heterophilic graph when h is close to 0, which signifies that edges predominantly link nodes belonging to distinct categories (i.e., $\mathcal { N } ( v _ { i } )$ contains substantial semantic diversity).

## B. Open-Set Node Detection Formulation

Unlike classical closed-set semi-supervised node classification, where the training and testing phases share an identical label space, Open-Set Recognition (OSR) accommodates the presence of unknown semantic classes during inference. Formally, let $\mathcal { V } _ { L } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { K } \}$ be the set of K known classes available during the training stage. In the openset deployment phase, the test nodes may originate from an expanded label space $y ~ = ~ y _ { L } \cup \mathcal { V } _ { U }$ , where $\begin{array} { r l } { ) _ { \mathcal { V } } } & { { } = } \end{array}$ $\left\{ c _ { K + 1 } , c _ { K + 2 } , \dots \right\}$ denotes the set of unseen or unknown classes that never appear in the training dataset, satisfying $y _ { L } \cap y _ { U } = \emptyset$

During training, we are given the graph G along with a set of labeled training nodes $\triangleright _ { t r a i n } \subset \mathcal { V } _ { : }$ , where each node $v _ { i } \in \mathcal { V } _ { t r a i n }$ is assigned a known label $y _ { i } \in \mathcal { V } _ { L }$ . The remaining nodes are partitioned into a validation set $\mathcal { V } _ { v a l }$ and a test set $\nu _ { t e s t } .$ . Critically, $\nu _ { t e s t }$ comprises both known-class nodes (whose labels belong to $y _ { L } )$ and unknown-class nodes (whose true labels belong to $y _ { U } )$ . The objective of open-set node detection is to learn a mapping function $\mathcal { F } : \mathcal { V }  \mathcal { V } _ { L } \cup \{ c _ { u n k } \}$ capable of precisely classifying nodes from known classes into their respective categories while simultaneously rejecting nodes from any unseen classes by categorizing them into a single unified unknown slot $c _ { u n k }$ (typically mapped as the $( K + 1 )$ -th class).

## C. Open-Set GNNs under Heterophily

An Open-Set Graph Neural Network (GNN) model typically consists of a structural encoder followed by an openset classifier. Existing traditional open-set learning paradigms frequently assume that data samples are independent and identically distributed (i.i.d.). Open-set GNNs break this assumption by leveraging spatial dependencies, propagating egofeatures across the topological structure via message-passing mechanisms to generate robust node representations ${ \textbf { Z } } =$ Encoder(X, A).

However, in heterophilic environments, standard open-set GNNs that rely on uniform low-pass aggregation (such as GCN) inherently aggregate conflicting representations from dissimilar neighbors. This drawback distorts the open-set decision boundaries by mixing known and unknown semantic spaces in the neighborhood. To resolve this challenge, HOPE addresses the coupled heterophily and open-set configuration by learning heterophily-aware structural mappings that separate multi-hop structural patterns. Instead of relying on representation-space distance margins as the primary openset objective, HOPE introduces a dedicated $( K + 1 )$ -way classification mechanism. This framework ensures that for any node $v _ { i } \in \mathcal { V } _ { t e s t }$ , the prediction is derived via:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c \in \{ 1 , \ldots , K , K + 1 \} } \mathbf { P } ( y _ { i } = c \mid \mathbf { x } _ { i } , \mathbf { A } ) ,\tag{2}
$$

where $c ~ = ~ K + 1$ represents the dynamic rejection slot for pseudo-unknown node variations synthetically extrapolated along heterophilic structural directions.

## IV. METHODOLOGY

In this section, we elaborate on the architectural design of HOPE, a novel framework tailored for semi-supervised open-set node classification on heterophilic graphs. As shown in Fig. 2, HOPE consists of four components: a structureaugmented feature initialization layer, a trustworthy neighborhood aggregation mechanism, a classification loss guided by a structural pseudo-extrapolation strategy, and a joint classification optimization framework integrated with logit margin regularization. By dynamically generating out-of-distribution (OOD) node boundaries without relying on brittle marginbased constraints, HOPE effectively prevents the overlapping of known and unseen class distributions under severe structural heterophily.

## A. Structure-Augmented Feature Initialization

In heterophilic open-set graphs, known and unknown nodes are often structurally mixed, making it difficult to distinguish unknown classes after neighborhood aggregation. Therefore, the model should establish discriminative node representations before the message passing stage. An appropriate initial representation should capture both semantic attributes and topological structures, because in heterophilic graphs, topological connections carry important boundary patterns that can indicate whether a node lies in a mixed neighborhood containing open-set nodes. To construct a complete initial node representation and avoid structural confusion at the beginning of the network, we enrich the raw feature space by explicitly adding structural encodings. This design ensures that the model recognizes structural identities at open-set boundaries before feature propagation.

For each node $v _ { i } ,$ we pre-compute a 16-dimensional topology-only structural encoding to capture its multi-hop structural characteristics before message passing. Specifically, let $\mathbf { P } = \mathbf { A } \mathbf { D } ^ { - 1 }$ denote the degree-normalized propagation matrix, where D is the degree matrix. We define the structural encoding as

$$
\mathbf { s } _ { i } = \left[ ( \mathbf { P } ) _ { i i } , ( \mathbf { P } ^ { 2 } ) _ { i i } , \ldots , ( \mathbf { P } ^ { 1 6 } ) _ { i i } \right] \in \mathbb { R } ^ { 1 6 } .
$$

The k-th component $( \mathbf { P } ^ { k } ) _ { i i }$ characterizes the structural return pattern of node $v _ { i }$ after k propagation steps, allowing s<sub>i</sub> to summarize local topology over multiple neighborhood ranges.

This encoding depends only on graph structure, requires no node labels, and is shared across different backbone choices.

We then combine the structural encoding with the original node attributes to obtain the initial representation:

$$
{ \bf h } _ { i } ^ { ( 0 ) } = \mathrm { M L P } _ { \mathrm { i n i t } } \left( \left[ { \bf x } _ { i } \| { \bf s } _ { i } \right] \right) ,\tag{3}
$$

where ∥ denotes feature concatenation. This structureaugmented representation provides topology-aware initialization for the subsequent heterophily-aware message passing.

## B. Trustworthy Neighborhood Aggregation

In heterophilic open-set graphs, not all neighbors provide useful information, as cross-class and open-set connections may introduce misleading messages during feature propagation. However, standard low-pass aggregation unconditionally averages all adjacent representations, which allows openset node variants to indiscriminately poison the surrounding known-class representations across heterophilic links.

To alleviate this problem, we design a trustworthy neighborhood aggregation module to selectively aggregate messages from reliable, contextually consistent neighbors, which shields the ego-node from noisy or conflicting semantic categories. By implementing a trustworthy neighborhood filtering mechanism, edge compatibility is evaluated to ensure that only highly reliable homophilic neighbors participate in the final neighborhood aggregation. Concretely, we utilize an edge discriminator to estimate the structural relationship between connected nodes. For each incoming edge $( v _ { j } , v _ { i } )$ , the discriminator computes a homophily probability $p _ { i j }$ . Meanwhile, we compute the cosine similarity between the intermediate hidden states of node $v _ { i }$ and node $v _ { j } .$ . Let $\mathbf { h } _ { i } ^ { ( t - 1 ) }$ denote the hidden embedding of node $v _ { i }$ at the $( t - 1 )$ -th iteration step. The model evaluates a homophily selection score $s _ { i j }$ for each connection:

$$
s _ { i j } = p _ { i j } \cdot \frac { \mathrm { R e L U } ( \mathrm { S i m } ( \mathbf h _ { j } ^ { ( t - 1 ) } , \mathbf h _ { i } ^ { ( t - 1 ) } ) ) } { \tau } ,\tag{4}
$$

where $\mathrm { { S i m } ( \cdot ) }$ is the cosine similarity function, and $\tau$ represents the aggregation temperature. Rather than keeping all neighbors, a strict masking threshold is applied based on $s _ { i j } .$ An edge is preserved only if its homophily selection score is sufficiently high. This step filters out untrustworthy cross-class neighbors and extracts a clean, homophilic neighbor subset $\mathcal { N } _ { t r u s t } ( v _ { i } )$

Finally, the model performs normalized aggregation exclusively over this trustworthy homophilic subset $\mathcal { N } _ { t r u s t } ( v _ { i } )$ . The layer-wise representation update for node $v _ { i }$ at the t-th step is formulated as follows:

$$
\begin{array} { r } { \mathbf h _ { i } ^ { ( t ) } = \mathrm { L a y e r N o r m } \Bigg ( \mathbf M \mathrm { L P } _ { f u s e } \Big ( \big [ \mathbf h _ { i } ^ { ( t - 1 ) } \ | \ } \\ { \displaystyle \sum _ { v _ { j } \in \mathcal N _ { t r u s t } ( v _ { i } ) } \alpha _ { i j } \cdot \mathbf h _ { j } ^ { ( t - 1 ) } \big ] \Big ) + \mathbf W _ { s e l f } \cdot \mathbf h _ { i } ^ { ( 0 ) } \Bigg ) , } \end{array}\tag{5}
$$

where ∥ denotes the concatenation operation, $\alpha _ { i j }$ represents the normalized edge weight computed via Softmax over the trusted subset, and ${ \bf W } _ { s e l f } \cdot { \bf h } _ { i } ^ { ( 0 ) }$ provides a self-loop residual connection from the initialization layer. After T iterations, the final output representation is denoted as ${ \bf z } _ { i } = { \bf h } _ { i } ^ { ( T ) }$ for all $v _ { i } ~ \in ~ \mathcal { V } .$ . For different backbones, the backbone-specific propagation first produces a graph-aware representation, which is combined with $\mathbf { h } ^ { ( 0 ) }$ through a residual connection and subsequently refined by the same trustworthy aggregation module. Thus, HOPE does not alter the internal propagation rule of the underlying backbone.

![](images/3639c1189ed2f9be162936b8b83942a7d750c83f6b7cc8d6da20feedc64610a6.jpg)  
Fig. 2: The overall architecture of HOPE. Raw node features are enriched with structural encodings into initial representations $\mathbf { h } _ { 0 } .$ The structure encoder then extracts multi-hop structural patterns through selective neighborhood pathways to output enhanced intermediate embeddings. Finally, a unified $( K { + } 1 ) { \mathrm { - w a y } }$ classifier categorizes nodes, jointly optimized via a multi-task learning paradigm comprising known-class loss $\mathcal { L } _ { r e a l }$ , pseudo-extrapolated unknown-class loss $\mathcal { L } _ { \boldsymbol { s y n } }$ , and classification-driven regularization penalty $\mathcal { L } _ { r e g }$ without geometric margin contractions.

Overall, this module enables HOPE to perform more robust and discriminative message passing by preserving beneficial homophilic signals while suppressing misleading heterophilic noise, thereby improving open-set recognition in complex graph structures.

## C. Pseudo-Unknown Proxy Generation and Classification Loss

In open-set recognition, the absence of labeled unknown samples makes it difficult for the model to explicitly learn where known-class decision boundaries should stop. To bridge this gap, we synthesize representative out-of-distribution (OOD) boundary instances to explicitly populate the $( K + 1 ) \cdot$ th classification slot, forcing the open-set model to learn a compact and closed decision boundary for known classes within a standard classification framework. Under heterophilic environments, graph nodes generally exhibit a low edge homophily ratio $h ,$ meaning that their neighborhoods $\mathcal { N } ( v _ { i } )$ contain substantial semantic diversity across multiple categories. Crucially, this structural characteristic applies to all entities in the graph, including both known and unknown classes. During message-passing propagation, when a Graph Neural Network (GNN) aggregates multi-hop structural patterns, the features of an open-set node are simultaneously subjected to multi-directional traction exerted by its semantically diverse neighbors from distinct known categories. This omnidirectional structural pulling prevents unknown nodes from forming isolated, well-segregated clusters in the latent space. Instead, their latent representations are inherently driven into the intersecting zones, peripheral margins, and ambiguous boundaries of the established known manifolds.

Consequently, the latent representations of nodes near these frontiers, regardless of whether they belong to known or unseen categories, would suffer from severe territorial overlap, which blurs the closed-set rejection boundaries. To decouple this overlap without modifying or distorting the internal representation spaces of known classes, HOPE leverages a structural extrapolation paradigm optimized within mini-batches. Instead of forcing the learned clusters of seen categories to become overly compact, our method places synthetic boundary proxies in the ambiguous regions between different classes. By assigning these boundary proxies to the unique classification slot $c _ { u n k } = K + 1 \quad$ , the standard cross-entropy loss forces the $( K + 1 )$ -th logit to become dominant exactly within these overlapping regions. As a result, the linear decision boundaries of the classifier are driven to adaptively wrap around and seal the known manifolds, effectively delegating the contaminated intersection zones to the unknown slot while leaving the internal latent structures of known classes uncompromised and largely preserved.

Specifically, during each training epoch, we optimize HOPE on the subgraph induced by labeled training nodes. To provide stable class anchors, we maintain an EMA center $\mu _ { c } \in \mathbb { R } ^ { d }$ for

each known class $c \in \mathcal { V } _ { L }$

$$
\pmb { \mu } _ { c }  \rho \pmb { \mu } _ { c } + ( 1 - \rho ) \frac { 1 } { | \mathscr { V } _ { \mathrm { t r a i n } } ^ { c } | } \sum _ { v _ { i } \in \mathscr { V } _ { \mathrm { t r a i n } } ^ { c } } \mathbf { z } _ { i } ,\tag{6}
$$

where $\mathcal { V } _ { \mathrm { t r a i n } } ^ { c }$ denotes the labeled training nodes belonging to class $c ,$ and $\rho$ is the EMA smoothing factor.

To synthesize pseudo-unknown proxies, we sample structurally ambiguous known-class nodes as anchors according to a softened structural score that combines neighborhood entropy and local cross-class connectivity. All statistics involved in anchor selection are computed exclusively on the labeled training-induced subgraph.

For an anchor node $v _ { i }$ with label $y _ { i } ,$ we define its known heterophilic neighborhood as $\mathcal { N } _ { \mathrm { h e t } } ( v _ { i } ) ~ = ~ \{ v _ { j } ~ \in ~ \mathcal { N } ( v _ { i } ) ~ \cap ~$ $\mathcal { V } _ { \mathrm { t r a i n } } \mid y _ { j } \ne y _ { i } , y _ { j } \in \mathcal { V } _ { L } \}$ . We then construct an outward extrapolation direction from the center of the anchor class toward its heterophilic neighbors: $\begin{array} { r } { \mathbf { d } _ { i } = \frac { 1 } { | \mathcal { N } _ { \mathrm { h e t } } ( v _ { i } ) | } \sum _ { v _ { j } \in \mathcal { N } _ { \mathrm { h e t } } ( v _ { i } ) } ( \mathbf { z } _ { j } - \mathbf { \lambda } } \end{array}$ $\mu _ { y _ { i } } )$ . Accordingly, the pseudo-unknown representation is generated as

$$
\tilde { \mathbf { z } } = \pmb { \mu } _ { y _ { i } } + \beta \mathbf { d } _ { i } + \epsilon ,\tag{7}
$$

where $\beta \sim \mathcal { U } ( 1 , \beta _ { \mathrm { m a x } } )$ and $\beta _ { \mathrm { m a x } } = \operatorname* { m a x } ( 1 , \beta _ { \mathrm { b a s e } } ( 1 + \eta ) )$ Here, $\eta$ denotes the heterophily ratio of known-class edges in the current training-induced subgraph, which adaptively controls the extrapolation range, while $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ introduces mild perturbations to improve boundary coverage.

To jointly preserve known-class discrimination and learn the additional rejection slot, we optimize real known nodes and synthesized pseudo-unknown proxies in a unified $( K { + } 1 )$ -way classification space. For a labeled training node $v _ { i } \in \mathbb { V } _ { \mathrm { t r a i n } }$ with $y _ { i } ~ \in ~ \mathcal { D } _ { L }$ , we apply the supervised cross-entropy loss only over the first K known-class logits:

$$
\mathcal { L } _ { \mathrm { r e a l } } = - \frac { 1 } { \vert \mathcal { V } _ { \mathrm { t r a i n } } \vert } \sum _ { v _ { i } \in \mathcal { V } _ { \mathrm { t r a i n } } } \log \frac { \exp ( \mathbf { o } _ { i , y _ { i } } ) } { \sum _ { c = 1 } ^ { K } \exp ( \mathbf { o } _ { i , c } ) } ,\tag{8}
$$

where $\mathbf { o } _ { i } = \mathrm { L i n e a r } ( \mathbf { z } _ { i } ) \in \mathbb { R } ^ { K + 1 }$ is the output logit vector and $\mathbf { o } _ { i , c }$ denotes its c-th component.

Concurrently, we optimize the generated proxies toward the $( K + 1 )$ -th unknown slot using a weighted cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { s y n } } = - \frac { 1 } { \sum _ { q = 1 } ^ { | \mathcal { V } _ { \mathrm { s y n } } | } w _ { q } } \sum _ { q = 1 } ^ { | \mathcal { V } _ { \mathrm { s y n } } | } w _ { q } \log \frac { \exp \left( \tilde { \mathbf { o } } _ { q , K + 1 } \right) } { \sum _ { c = 1 } ^ { K + 1 } \exp \left( \tilde { \mathbf { o } } _ { q , c } \right) } ,\tag{9}
$$

where $\tilde { \mathbf { o } } _ { q } = \mathrm { L i n e a r } ( \tilde { \mathbf { z } } _ { q } )$ , and $w _ { q }$ denotes the structural sampling weight inherited from the corresponding anchor.

## D. Logit Margin Regularization

To prevent the unknown slot from dominating the logit space of known-class nodes during training, we introduce a classification-driven margin penalty $\mathcal { L } _ { r e g }$ . This regularization term explicitly encourages that for any known-class training node $v _ { i }$ (with $y _ { i } ~ \in ~ \mathcal { V } _ { L } )$ , the logit corresponding to the unknown rejection slot $c _ { u n k } = K + 1 $ remains strictly lower than the maximum logit among the known classes. Without such a constraint, the model might push the unknown logit excessively high in order to fit synthetic pseudo-unknown proxies, thereby eroding the discriminative power of known classes. Formally, the logit margin regularization loss $\mathcal { L } _ { r e g }$ enforces a margin between the strongest known-class logit and the unknown-class logit for each known training node:

TABLE I: Detailed statistics of the evaluated heterophilous graph datasets.
<table><tr><td rowspan="2">Dataset Name</td><td colspan="4">Graph Properties</td></tr><tr><td>Nodes</td><td>Edges</td><td>Features</td><td>Classes</td></tr><tr><td>Chameleon</td><td>2,277</td><td>31,421</td><td>2,325</td><td>5</td></tr><tr><td>Squirrel</td><td>5,201</td><td>198,493</td><td>2,089</td><td>5</td></tr><tr><td>Wisconsin</td><td>251</td><td>466</td><td>1,703</td><td>5</td></tr><tr><td>Amazon-Ratings</td><td>24,492</td><td>93,050</td><td>300</td><td>5</td></tr><tr><td>Roman-Empire</td><td>22,662</td><td>32,927</td><td>300</td><td>18</td></tr><tr><td>Actor</td><td>7,600</td><td>26,752</td><td>932</td><td>5</td></tr><tr><td> $\mathbf { A r x i v - Y e a r }$ </td><td>169,343</td><td>1,166,243</td><td>128</td><td>5</td></tr></table>

<sup>a</sup>The class space indicates K observed known classes and 1 unobserved unknown class.

$$
\mathcal { L } _ { r e g } = \frac { 1 } { \vert \mathcal { V } _ { t r a i n } \vert } \sum _ { \substack { v _ { i } \in \mathcal { V } _ { t r a i n } } } \Big [ \operatorname* { m a x } \big ( 0 , \mathbf { o } _ { i , K + 1 } -\tag{10}
$$

where $\mathcal { V } _ { t r a i n } ^ { k n o w n }$ is the set of training nodes whose labels belong to the known classes $\mathcal { V } _ { L } , \ m > 0$ is a fixed margin hyperparameter, and max(0, ·) denotes the ReLU function. This loss penalizes a known-class node whenever its maximum known-class logit fails to exceed the unknown-slot logit by at least the margin m, thereby encouraging a clear separation between known-class predictions and the rejection option.

Without this regularization, the joint optimization of $\mathcal { L } _ { r e a l }$ and $\mathcal { L } _ { \boldsymbol { s y n } }$ may inadvertently drive the unknown logit to become active even for known nodes, especially in heterophilic graphs where known and unknown neighborhoods are heavily intertwined. By imposing a soft margin constraint directly on the logits, rather than on geometric distances in the representation space. HOPE avoids brittle boundary tuning while preserving the full expressiveness of the $( K + 1 ) { \mathrm { - w a y } }$ classifier.

Finally, the total objective function of HOPE is formulated as a multi-task learning paradigm:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { r e a l } + \gamma _ { 1 } \mathcal { L } _ { s y n } + \gamma _ { 2 } \mathcal { L } _ { r e g } ,\tag{11}
$$

where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are non-negative hyperparameters that scale the contributions of the pseudo-unknown classification risk and the known-class logit regularization penalty, respectively. Overall, this unified objective allows HOPE to jointly optimize known-class discrimination, unknown-boundary modeling, and known-class rejection regularization, leading to more reliable open-set recognition under heterophilic graph structures.

## V. EXPERIMENTS

## A. Experimental Setup

To comprehensively evaluate the performance of our proposed framework, we conduct benchmark evaluations across seven widely-used heterophilic graph datasets, namely Roman-Empire [33], Amazon-Ratings [33], Wisconsin [25], Actor [34], Chameleon [25], [35], Squirrel [25], [35], and Arxiv-Year [36]. These benchmarks span diverse topological scales, attribute dimensionalities, and feature densities, providing a robust and challenging testbed for open-set node classification under heterophilic environments. Following standard semisupervised open-set evaluation protocols established in graph domains, we designate the specific category with the fewest instances as the unobserved unknown novel class, ensuring it remains completely inaccessible during the optimization phase, while the remaining classes constitute the observed known label space. The source code and experimental configurations are available at: https://github.com/Solkattkgo/HOPE.

TABLE II: Experimental results (Acc and F1) across multi-datasets
<table><tr><td></td><td></td><td colspan="6">Accuracy (%)</td><td colspan="6"></td><td></td><td></td></tr><tr><td>Model</td><td>Method</td><td>Roman Empire</td><td>Amazon Ratings</td><td>Wis- consin</td><td>Actor</td><td>Chame- leon</td><td>Squirrel</td><td>Arxiv- Year</td><td>Roman Empire</td><td>Amazon Ratings</td><td>Wis- consin</td><td>Actor</td><td>Chame- leon</td><td>Squirrel</td><td>Arxiv- Year</td></tr><tr><td rowspan="6">GCN</td><td>GCN</td><td>20.58</td><td>31.63</td><td>47.76</td><td>12.48</td><td>43.24</td><td>14.68</td><td>44.91</td><td>16.13</td><td>11.20</td><td>29.57</td><td>12.35</td><td>36.57</td><td>22.35</td><td>32.23</td></tr><tr><td>ROG_PL</td><td>35.15</td><td>31.68</td><td>52.94</td><td>25.48</td><td>33.04</td><td>21.64</td><td>41.52</td><td>29.48</td><td>19.68</td><td>19.66</td><td>19.90</td><td>15.18</td><td>15.83</td><td>28.21</td></tr><tr><td>G2Pxy</td><td>31.60</td><td>30.85</td><td>41.88</td><td>20.43</td><td>31.92</td><td>18.55</td><td>42.62</td><td>37.01</td><td>21.98</td><td>33.72</td><td>13.23</td><td>18.26</td><td>20.01</td><td>30.22</td></tr><tr><td>EGonc</td><td>34.24</td><td>37.54</td><td>49.14</td><td>17.43</td><td>46.27</td><td>15.98</td><td>40.98</td><td>29.57</td><td>22.67</td><td>19.23</td><td>18.70</td><td>31.44</td><td>16.77</td><td>30.60</td></tr><tr><td>CONC</td><td>28.72</td><td>31.38</td><td>42.64</td><td>11.76</td><td>15.53</td><td>17.02</td><td>38.60</td><td>15.32</td><td>11.47</td><td>13.16</td><td>7.48</td><td>18.38</td><td>19.10</td><td>29.66</td></tr><tr><td>HOPE</td><td>53.95</td><td>39.92</td><td>59.32</td><td>43.85</td><td>55.16</td><td>39.00</td><td>45.41</td><td>54.78</td><td>55.28</td><td>43.09</td><td>28.87</td><td>38.76</td><td>21.29</td><td>35.85</td></tr><tr><td rowspan="7">GPR-GNN</td><td>GPR-GNN</td><td>40.05</td><td>30.43</td><td>49.52</td><td>11.97</td><td>43.08</td><td>16.44</td><td>42.39</td><td>33.81</td><td>12.52</td><td>32.08</td><td>12.21</td><td>36.49</td><td>23.87</td><td>28.33</td></tr><tr><td>ROG_PL</td><td>47.14</td><td>32.28</td><td>47.05</td><td>28.13</td><td>24.50</td><td>26.82</td><td>34.79</td><td>41.07</td><td>17.26</td><td>20.60</td><td>21.19</td><td>21.34</td><td>17.27</td><td>20.26</td></tr><tr><td>G2Pxy</td><td>42.53</td><td>30.74</td><td>40.68</td><td>22.09</td><td>27.44</td><td>19.72</td><td>32.17</td><td>27.34</td><td>21.76</td><td>34.29</td><td>16.33</td><td>18.26</td><td>20.23</td><td>19.77</td></tr><tr><td>EGonc</td><td>34.21</td><td>37.62</td><td>45.33</td><td>21.34</td><td>36.58</td><td>16.33</td><td>37.52</td><td>34.56</td><td>28.51</td><td>23.12</td><td>17.71</td><td>20.57</td><td>17.45</td><td>24.85</td></tr><tr><td>CONC</td><td>36.66</td><td>31.38</td><td>36.64</td><td>12.22</td><td>21.27</td><td>19.20</td><td>40.60</td><td>32.78</td><td>19.62</td><td>33.28</td><td>11.44</td><td>18.38</td><td>20.18</td><td>29.33</td></tr><tr><td>HOPE</td><td>50.04</td><td>38.81</td><td>59.32</td><td>42.22</td><td>56.99</td><td>46.21</td><td>44.58</td><td>51.04</td><td>33.80</td><td>41.44</td><td>27.69</td><td>39.93</td><td>29.53</td><td>32.01</td></tr><tr><td>GCNII</td><td>44.48</td><td>21.47</td><td>47.76</td><td>34.18</td><td>29.87</td><td>31.85</td><td>42.13</td><td>33.87</td><td>12.00</td><td>28.85</td><td>19.08</td><td>28.71</td><td></td><td>26.59</td></tr><tr><td rowspan="6">GCNII</td><td>ROG_PL</td><td>46.22</td><td>31.66</td><td>46.80</td><td>27.05</td><td>34.46</td><td>41.39</td><td>38.55</td><td>34.55</td><td>14.14</td><td>15.67</td><td>17.58</td><td>11.33</td><td>22.38 23.54</td><td>23.52</td></tr><tr><td>G2Pxy</td><td>45.54</td><td>29.19</td><td>47.06</td><td>28.64</td><td>42.56</td><td>40.76</td><td>35.23</td><td>36.27</td><td>25.22</td><td>24.20</td><td>11.21</td><td>22.60</td><td>23.62</td><td>22.17</td></tr><tr><td>EGonc</td><td>44.91</td><td>30.61</td><td>42.64</td><td>28.40</td><td>46.20</td><td>34.03</td><td>30.76</td><td>26.28</td><td>20.53</td><td>21.05</td><td>17.76</td><td>23.74</td><td>21.77</td><td>22.01</td></tr><tr><td>CONC</td><td>33.65</td><td>32.05</td><td>42.21</td><td>25.64</td><td>33.54</td><td>34.13</td><td>36.61</td><td>24.61</td><td>22.34</td><td>26.87</td><td>12.09</td><td>28.18</td><td>22.14</td><td>25.40</td></tr><tr><td>HOPE</td><td>51.51</td><td>38.67</td><td>55.93</td><td>30.95</td><td>48.16</td><td>42.84</td><td>43.99</td><td>52.06</td><td>32.69</td><td>38.73</td><td>29.15</td><td>38.04</td><td>27.60</td><td>29.96</td></tr><tr><td>EG-GCN</td><td>52.66</td><td>21.89</td><td>62.50</td><td>40.76</td><td>52.77</td><td>41.03</td><td>45.52</td><td>48.64</td><td>22.01</td><td>50.24</td><td>21.87</td><td>38.79</td><td></td><td>33.63</td></tr><tr><td rowspan="6">EG-GCN</td><td>ROG_PL</td><td>49.62</td><td>31.38</td><td>49.15</td><td>18.02</td><td>44.07</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.63 22.33</td><td>29.33</td></tr><tr><td>G2Pxy</td><td>47.43</td><td>28.71</td><td>37.50</td><td>22.09</td><td></td><td>38.89</td><td>37.90</td><td>33.55</td><td>17.14</td><td>22.83 12.00</td><td>9.97</td><td>14.34</td><td></td><td></td></tr><tr><td></td><td>34.43</td><td>28.17</td><td>42.76</td><td>21.34</td><td>41.22</td><td>39.48</td><td>40.28 34.55</td><td>27.74 24.84</td><td>23.12 20.42</td><td>25.32</td><td>16.36</td><td>28.61</td><td>22.76</td><td>30.24</td></tr><tr><td>EGonc</td><td>31.98</td><td>27.92</td><td>40.67</td><td>12.22</td><td>38.36 50.54</td><td>34.13 26.82</td><td>41.36</td><td>33.44</td><td>16.33</td><td>11.56</td><td>17.73 17.62</td><td>29.65</td><td>21.77</td><td>26.86</td></tr><tr><td>CONC</td><td>53.13</td><td>29.33</td><td>53.70</td><td>43.94</td><td>53.07</td><td>43.22</td><td>47.62</td><td>51.78</td><td>25.71</td><td>40.13</td><td>26.73</td><td>26.27 46.23</td><td>24.75 25.02</td><td>30.86</td></tr><tr><td>HOPE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>33.97</td></tr></table>

As a plug-and-play framework adaptable to standard back bones, we evaluate HOPE by comparing it against two representative groups of state-of-the-art graph baselines. The first group consists of graph neural network architectures including GCN [1], GPR-GNN [27], GCNII [28], and EG-GCN [18], which serve as foundational structural encoders covering both homophilous assumptions and heterophilic designs. The second group comprises open-set detection frameworks including ROG PL [20], G2Pxy [16], EGonc [17] and CONC [21], which represent advanced open-set node classification and boundary modeling methods. Crucially, for the specific evaluations conducted in the open-set and closed-set performance analysis, robustness evaluation, and computational efficiency analysis, GCNII, GPR-GNN, and EG-GCN utilize their own specialized architectures as structural encoders, whereas all other compared open-set detection methods consistently adopt GCN as their default underlying backbone network.

To evaluate open-set node classification, we use Overall Accuracy and Macro-F1-Score. Overall Accuracy measures global prediction performance, while Macro-F1 provides a balanced evaluation across imbalanced classes. Unless otherwise specified, we set $\gamma _ { 1 } = 0 . 5 , \gamma _ { 2 } = 0 . 1$ , and $m = 0 . 3$ . The random seed is fixed to 42 for all randomized operations and dataset splits.

## B. Performance Comparison

The comparison results of HOPE are illustrated in Table II. From the table, we make the following key observations:

Our proposed HOPE consistently secures the optimal or competitive second-best performance across almost all evaluation slots under both metrics. When paired with standard backbones, HOPE yields substantial performance gains compared to the vanilla versions. For instance, on the Roman-Empire dataset under the GPR-GNN framework, our method elevates the accuracy from 40.05% to the best 50.04% and the macro-F1 score from 33.81% to the best 51.04%. This performance stability highlights that our approach generalizes well to various heterophilic structural patterns.

The strong performance of HOPE is rooted in its dedicated design for open-set node recognition under severe structural heterophily. Specifically, it enriches raw features with multi-hop structural patterns via structure-augmented feature initialization, selectively propagates semantically consistent messages using a trustworthy neighborhood aggregation module, and synthesizes realistic pseudo-unknown boundaries via heterophily-guided structural extrapolation. Together with a joint classification framework with logit margin regularization, HOPE successfully circumvents severe statistical trade-offs, yielding balanced and stable leads.

TABLE III: Ablation study of the proposed model on Roman-Empire, Wisconsin, and Squirrel datasets.
<table><tr><td></td><td colspan="2">Roman-Empire</td><td colspan="2">Wisconsin</td><td colspan="2">Squirrel</td></tr><tr><td></td><td>acc</td><td>fl</td><td>acc</td><td>f1</td><td>acc</td><td>fl</td></tr><tr><td>w/o reg</td><td>18.63</td><td>17.68</td><td>16.95</td><td>5.80</td><td>55.50</td><td>14.28</td></tr><tr><td>w/o init</td><td>48.39</td><td>40.41</td><td>59.32</td><td>45.12</td><td>37.87</td><td>20.28</td></tr><tr><td>w/o trust</td><td>47.16</td><td>48.17</td><td>57.63</td><td>40.52</td><td>38.51</td><td>20.64</td></tr><tr><td>Full Model</td><td>53.95</td><td>54.78</td><td>59.32</td><td>55.28</td><td>39.00</td><td>21.29</td></tr></table>

In contrast, alternative baselines encounter clear algorithmic bottlenecks. Traditional closed-set encoders including GCN, GCNII, GPR-GNN, and EG-GCN lack native open-set awareness, and their confidence calibration breaks down under heterophilic linking patterns when evaluating under post-hoc threshold deployment protocols. This explains why, within the GCNII-backbone group, the vanilla GCNII achieves the highest Overall Accuracy on the Actor dataset but exhibits a depressed Macro-F1 score. On the other hand, established open-set baselines like ROG PL, G2Pxy, EGonc, and CONC assume structural homophily. When applied to heterophilic graphs, their rigid boundaries aggressively classify valid normal nodes into the open-set rejection slot due to severe falsepositive errors. This over-rejection behavior accounts for the severe statistical trade-offs observed in the results, such as when ROG PL is paired with GPR-GNN on the Amazon-Ratings dataset, or combined with GCNII on the Squirrel dataset, where it achieves a relatively high Overall Accuracy while its Macro-F1 score lags far behind due to the catastrophic collapse of known-class precision.

## C. Ablation Study

To examine the contribution of key designs in HOPE, we conduct ablation studies on 3 datasets with a GCN backbone. We compare the full model against three variants: w/o init, which removes the structure-augmented feature initialization layer; w/o trust, which disables the trustworthy neighborhood aggregation mechanism; and w/o reg, which omits the logit margin regularization loss. We have the following observations from Table III: ❶ The full model achieves the best overall performance, demonstrating that all designed modules are essential and mutually reinforcing. ❷ Removing the structural initialization layer triggers a significant performance degradation. This decline occurs because raw node attributes lack geometric awareness, making the model blind to local topological positions. These findings confirm that relying solely on semantic features is insufficient in heterophilic environments. ❸ Disabling the edge-filtering mechanism leads to noticeable performance drops across all datasets. The root cause is that standard low-pass aggregation unconditionally averages all adjacent representations, allowing open-set node variants and heterophilic cross-class neighbors to indiscriminately poison the ego-node features. ❹ Omitting the logit margin regularization loss causes a catastrophic failure mode where the model collapses to classifying almost all nodes as the open-set class. This dramatic collapse verifies that our classification-driven margin penalty is indispensable for anchoring known-class logits to sustain a stable open-set decision space.

![](images/24486f616ea2235d14d55b3bbecc5d4d674b7732ffa8520cc3dceb102a6bb606.jpg)  
(a) Chameleon

![](images/086832ebfd60ea7021da6dc4c0c201979ce12745c8a63e3b94591c66f8e55203.jpg)  
(b) Roman-Empire

Fig. 3: ACC of known/unknown classes comparison.  
![](images/aafa5b0cc9c90ecdf6829eec189bbaa041abcd9d7c46f1bb6d1e169e01132ab2.jpg)  
(a) Accuracy comparison

![](images/bb62f3c0aad78285e56310f60721bd13a714956345a58319ba3e94ef7b914f2e.jpg)  
(b) F1-score comparison  
Fig. 4: ACC/F1 with the increasing #open-set classes.

## D. Open-set and Closed-set Performance

To evaluate the fine-grained discriminative capability of HOPE in open-set scenarios, we analyze the classification performance on both known and unknown classes across the Chameleon and Roman-Empire datasets, as illustrated in Fig. 3. From the results, we observe that HOPE exhibits an outstanding capability to simultaneously maintain high classification precision on observed known classes and achieve superior detection rates on unobserved unknown nodes. Although G2Pxy secures a higher individual accuracy for identifying unknown class nodes on the Roman-Empire dataset, it severely sacrifices the prediction accuracy of the normal known classes, leading to massive false-positive classification errors. In contrast, by utilizing structure-augmented initialization and trustworthy aggregation alongside balanced margin regularization, HOPE successfully manages the decision spaces and prevents the unknown slot from aggressively absorbing normal nodes, thereby establishing stable and comprehensive leads under severe structural heterophily.

## E. Robustness Analysis

To evaluate the operational stability and robustness of our proposed framework under volatile open-set environments, we show the performance variations on the Roman-Empire dataset by incrementally increasing the number of unseen classes from one to eight. As observed from the results in Fig. 4, HOPE consistently maintains the optimal performance across all evaluation phases, exhibiting strong resistance against environmental volatility compared to alternative baselines. Even when the open-set class space expands during deployment, our framework establishes a steady performance superiority. This robust behavior is mainly attributed to our specialized pseudounknown proxy generation strategy, which models the invariant topological mixture mechanism within local mini-batches by adaptively shifting proxies outwards along heterophilic neighborhood displacement vectors. Consequently, our synthetic proxies continue to tend to populate representation regions associated with known-class boundaries to maintain stable decision hyperplanes under structural heterophily.

![](images/a264f0a68af8523631f47c4bed74f77e6c06014eeda6a6a43bef6f0cc674eb62.jpg)  
Fig. 5: The t-SNE visualization of node representations on the Chameleon dataset. The synthesized proxies tend to occupy ambiguous regions near known-class boundaries and show substantial overlap with regions containing real unknown nodes.

## F. Latent Space Topology Visualization

To intuitively demonstrate the geometric soundness of our proposed structural pseudo-extrapolation strategy, we conduct a latent space topology visualization experiment using the t-SNE algorithm on the Chameleon dataset. The resulting visual distribution mapping is illustrated in Fig. 5. As shown in the t-SNE visualization, the real unknown test nodes are distributed across specific regions adjacent to the known class boundaries. Importantly, rather than scattering randomly or encroaching upon the dense cores of known clusters, the synthetic boundary proxies tend to locate near the spatial positions occupied by the real unknown test nodes. This observation provides qualitative evidence that our proxies tend to appear near regions where real unknown nodes reside, offering visual support for the strategic rationality of our structural extrapolation paradigm under heterophily.

## G. Efficiency Analysis

Complexity Analysis. Let N, M, and d denote the numbers of nodes, edges, and hidden dimensions, respectively. Excluding the structural encoding preprocessing, trustworthy edge scoring and aggregation require $O ( M d )$ operations, while the edge MLP costs $O ( M d ^ { \bar { 2 } } )$ when its hidden width scales with d. Node transformations require $O ( N d ^ { 2 } )$ , and proxy construction requires $O ( M d + S d )$ for S synthesized proxies. Therefore, for fixed d and S, the per-epoch complexity scales linearly with $N + M$ , with memory complexity $O ( N d + M d )$ Empirical Evaluation. We illustrate the execution trade-offs on the Chameleon dataset in Fig. 6, where Fig. 6a and Fig. 6b showcase accuracy versus time and memory usage, respectively. The plots indicate that alternative closed-set, open-set, and heterophilic baselines either suffer from poor performance or incur heavy computational overhead. In contrast, HOPE secures a substantial performance margin over these competitors while maintaining a small computational/memory footprint, achieving high accuracy with lower time and memory expenditures than heavy paradigms like EGonc.

![](images/2ff499f5108671217b97336f5d79a2c3f3bbb2a85d1194fd0cc187330c39ea34.jpg)

![](images/35ab738c21e832bf1709146527d17b5365cff49d6383c65816218fb6d1f8c074.jpg)  
(a) Accuracy vs. time usage (b) Accuracy vs. memory usage Fig. 6: Computational costs comparison.

![](images/e850b4e135ef4216bf60d6f45642a41870f74c6cebac726142d53fa82f92d5fc.jpg)

![](images/a4de060fb7bbee1c6a8ccc8202777b6ff4b2f8ca55486c6fdd02b22650ae4205.jpg)  
(a) Sensitivity of Accuracy  
(b) Sensitivity of F1 score  
Fig. 7: Parameter sensitivity analysis of HOPE with respect to $\gamma 1 , \gamma 2 ,$ and m . The model shows consistent performance across a broad range of parameter settings.

## H. Parameter Sensitivity Analysis

To investigate the operational stability of HOPE under varied optimization balances, we perform a parameter sensitivity analysis on the Squirrel dataset. We evaluate the performance fluctuations in terms of Accuracy and Macro-F1 Score by varying the critical loss scaling factors $\gamma _ { 1 } , ~ \gamma _ { 2 }$ , and the soft margin m independently within the range from 0.1 to 1.0. As illustrated in Fig. 7, different hyperparameter configurations present smooth and predictable performance variations. Specifically, $\gamma _ { 1 }$ and $\gamma _ { 2 }$ exhibit complementary tendencies due to the balance between synthetic open-set optimization and topological soft margin anchoring, while both metrics remain relatively stable and flat across the entire variation spectrum of the soft margin parameter m. Overall, HOPE remains stable across hyperparameter settings, demonstrating low sensitivity under heterophilic distributions.

## VI. CONCLUSION

In this paper, we propose HOPE, a novel framework designed for open-set node classification on heterophilic graphs. To handle severe structural heterophily, our approach enriches raw features with multi-hop structural patterns and filters out noisy cross-class connections via a trustworthy aggregation mechanism. Furthermore, we introduce a heterophily-guided pseudo-extrapolation strategy to synthesize realistic pseudounknown boundaries at the intersections of known classes. This process is jointly optimized with a known-class logit regularization loss to maintain a balanced decision space. Extensive experiments across multiple benchmarks demonstrate that HOPE achieves the best or competitive performance across the evaluated settings against state-of-the-art baselines, verifying its effectiveness, efficiency, and robustness in entangled label environments.

## REFERENCES

[1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[2] M. Wu, S. Pan, and X. Zhu, “Openwgl: Open-world graph learning,” in 2020 IEEE international conference on data mining (icdm). IEEE, 2020, pp. 681–690.

[3] X. Gao, T. Chen, W. Zhang, Y. Li, X. Sun, and H. Yin, “Graph condensation for open-world graph learning,” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 851–862.

[4] Y. Liu, S. Li, Y. Zheng, Q. Chen, C. Zhang, P. S. Yu, and S. Pan, “From few-shot to zero-shot: Towards generalist graph anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[5] T. Huang, D. Wang, Y. Fang, and C. Zhengyu, “End-to-end open-set semi-supervised node classification with out-of-distribution detection,” in IJCAI, 2022.

[6] X. Zheng, Y. Wang, Y. Liu, M. Li, M. Zhang, D. Jin, P. S. Yu, and S. Pan, “Graph neural networks for graphs with heterophily: A survey,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[7] J. Lin, X. Guo, S. Zhang, Y. Zhu, and J. Shun, “When heterophily meets heterogeneity: Challenges and a new large-scale graph benchmark,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 5607–5618.

[8] T. Wang, D. Jin, R. Wang, D. He, and Y. Huang, “Powerful graph convolutional networks with adaptive propagation mechanism for homophily and heterophily,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 4, 2022, pp. 4210–4218.

[9] J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, “Beyond homophily in graph neural networks: Current limitations and effective designs,” Advances in neural information processing systems, vol. 33, pp. 7793–7804, 2020.

[10] Y. Tan, G. Long, J. Jiang, and C. Zhang, “Influence-oriented personalized federated learning,” in IEEE International Conference on Data Mining, 2026.

[11] B. Chen, W. Wongso, X. Hu, Y. Tan, and F. D. Salim, “Multi-stage verification-centric framework for mitigating hallucination in multimodal rag,” in 2025 KDD Cup Workshop for Multimodal Retrieval Augmented Generation.

[12] A. Iscen, G. Tolias, Y. Avrithis, and O. Chum, “Label propagation for deep semi-supervised learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5070– 5079.

[13] H. Wang and J. Leskovec, “Unifying graph convolutional neural networks and label propagation,” arXiv preprint arXiv:2002.06755, 2020.

[14] D. Bo, B. Hu, X. Wang, Z. Zhang, C. Shi, and J. Zhou, “Regularizing graph neural networks via consistency-diversity graph augmentations,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 4, 2022, pp. 3913–3921.

[15] J. Li, C. Xiong, and S. C. Hoi, “Comatch: Semi-supervised learning with contrastive graph regularization,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9475–9484.

[16] Q. Zhang, Z. Shi, X. Zhang, X. Chen, P. Fournier-Viger, and S. Pan, “G2pxy: generative open-set node classification on graphs with proxy unknowns,” in International Joint Conference on Artificial Intelligence, 2023, pp. 4576–4583.

[17] Q. Zhang, Z. Shi, S. Pan, J. Chen, H. Wu, and X. Chen, “Egonc: Energybased open-set node classification with substitute unknowns,” Advances in Neural Information Processing Systems, vol. 37, pp. 66 147–66 177, 2024.

[18] S. Liu, D. He, Z. Yu, D. Jin, Z. Feng, and W. Zhang, “Integrating co-training with edge discrimination to enhance graph neural networks under heterophily,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 18, 2025, pp. 18 960–18 968.

[19] X. Shen, Y. Liu, Y. Wang, R. Miao, Y. Dai, S. Pan, Y. Chang, and X. Wang, “Raising the bar in graph ood generalization: Invariant learning beyond explicit environment modeling,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[20] Q. Zhang, X. Li, J. Lu, L. Qiu, S. Pan, X. Chen, and J. Chen, “Rog pl: Robust open-set graph learning via region-based prototype learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 8, 2024, pp. 9350–9358.

[21] Q. Zhang, J. Lu, X. Li, H. Wu, S. Pan, and J. Chen, “Conc: complexnoise-resistant open-set node classification with adaptive noise detection,” in Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24). IJCAI, 2024.

[22] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Ben-´ gio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[23] Y. Zhao, Y. Liu, Q. Chen, S. Li, Y. Tan, and S. Pan, “Fedcigar: A personalized reconstruction approach for federated graph-level anomaly detection,” in International Joint Conference on Artificial Intelligence, 2026.

[24] S. Li, Y. Zhao, Y. Tan, Q. Chen, Y. Liu, and S. Pan, “Towards anomaly detection on relational data,” arXiv preprint arXiv:2606.18621, 2026.

[25] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang, “Geom-gcn: Geometric graph convolutional networks,” arXiv preprint arXiv:2002.05287, 2020.

[26] K. Xu, C. Li, Y. Tian, T. Sonobe, K.-i. Kawarabayashi, and S. Jegelka, “Representation learning on graphs with jumping knowledge networks,” in International conference on machine learning. pmlr, 2018, pp. 5453– 5462.

[27] E. Chien, J. Peng, P. Li, and O. Milenkovic, “Adaptive universal generalized pagerank graph neural network,” arXiv preprint arXiv:2006.07988, 2020.

[28] M. Chen, Z. Wei, Z. Huang, B. Ding, and Y. Li, “Simple and deep graph convolutional networks,” in International conference on machine learning. PMLR, 2020, pp. 1725–1735.

[29] D. Su, X. Li, Z. Li, Y. Liao, R.-H. Li, and G. Wang, “Dirw: Pathaware digraph learning for heterophily,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 2771–2780.

[30] H. Xu, K. Liu, Z. Yao, P. S. Yu, M. Li, K. Ding, and Y. Zhao, “Lego-learn: Label-efficient graph open-set learning,” arXiv preprint arXiv:2410.16386, 2024.

[31] Y. Tan, C. Chen, W. Zhuang, X. Dong, L. Lyu, and G. Long, “Taming heterogeneity to deal with test-time shift in federated learning,” in International Workshop on Federated Learning for Distributed Data Mining, 2023.

[32] J. Gawlikowski, C. R. N. Tassi, M. Ali, J. Lee, M. Humt, J. Feng, A. Kruspe, R. Triebel, P. Jung, R. Roscher et al., “A survey of uncertainty in deep neural networks,” Artificial intelligence review, vol. 56, no. Suppl 1, pp. 1513–1589, 2023.

[33] O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, and L. Prokhorenkova, “A critical look at the evaluation of gnns under heterophily: Are we really making progress?” arXiv preprint arXiv:2302.11640, 2023.

[34] J. Tang, J. Sun, C. Wang, and Z. Yang, “Social influence analysis in large-scale networks,” in Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, 2009, pp. 807–816.

[35] B. Rozemberczki, C. Allen, and R. Sarkar, “Multi-scale attributed node embedding,” Journal of Complex Networks, vol. 9, no. 2, p. cnab014, 2021.

[36] D. Lim, X. Li, F. Hohne, and S.-N. Lim, “New benchmarks for learning on non-homophilous graphs,” arXiv preprint arXiv:2104.01404, 2021.