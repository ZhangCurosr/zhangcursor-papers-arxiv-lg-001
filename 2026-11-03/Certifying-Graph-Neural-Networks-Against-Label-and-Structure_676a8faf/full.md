# Certifying Graph Neural Networks Against Label and Structure Poisoning

Lukas Gosch <sup>\*</sup> <sup>1</sup> <sup>2</sup> <sup>3</sup> Xichuan Chen <sup>\*</sup> <sup>1</sup> Yan Scholten <sup>\*</sup> <sup>1</sup> <sup>2</sup> <sup>3</sup> Stephan Gunnemann¨ <sup>1</sup> <sup>2</sup> <sup>3</sup>

## Abstract

Robust machine learning for graph-structured data has made significant progress against testtime attacks, yet certified robustness to poisoning – where adversaries manipulate the training data – remains largely underexplored. For image data, state-of-the-art poisoning certificates rely on partitioning-and-aggregation schemes. However, we show that these methods fail when applied in the graph domain due to the inherent label and structure sparsity found in common graph datasets, making effective graph-partitioning difficult. To address this challenge, we propose a novel semi-supervised learning framework called deep Self-Training Graph Partition Aggregation (ST-GPA), which enriches each graph partition with informative pseudo-labels and synthetic edges, enabling effective certification against node-label and graph-structure poisoning under sparse conditions. Our method is architectureagnostic, scales to large numbers of partitions, and consistently and significantly improves robustness guarantees against both label and structure poisoning across multiple benchmarks, while maintaining strong clean accuracy. Overall, our results establish a promising direction for certifiably robust learning on graph-structured data against poisoning under sparse conditions.

## 1. Introduction

Graph Neural Networks (GNNs) are highly susceptible to adversarial perturbations in their input graph applied at test or training time (Zugner et al. ¨ , 2018; Dai et al., 2018). Subsequently developed empirical defenses are at the continual risk of being rendered ineffective by more sophisticated ways to choose adversarial perturbations (Koh et al., 2022; Mujkanovic et al., 2022). This motivates the development of robustness certificates, which provide provable guarantees about the stability of predictions under worst-case data perturbations, allowing to rigorously assess and mitigate adversarial vulnerabilities. While significant advances in providing such provable guarantees for GNNs against test-time attacks have been made (Gunnemann¨ , 2022; Hojny et al., 2024), certifying robustness of GNNs against data poisoning, where an adversary can manipulate the graph structure (Zugner & G¨ unnemann¨ , 2019) or node-labels (Lingam et al., 2024) at training time, remains largely underexplored.

The most effective approaches to derive poisoning robustness guarantees in the image domain rely on partitioning the training data, learning separate (base) classifiers, and aggregating predictions (e.g., via majority vote) (Levine & Feizi, 2021). However, we demonstrate that these methods fail when directly applied to common graph learning tasks such as node classification. In particular, we find that the core challenge is sparsity: both labels and graph structure are often too sparse to provide effective training signals under data partitioning, leading to poor performance for label certificates and vacuous robustness guarantees for structure poisoning. This raises a critical question:

## How can we effectively analyze and guarantee the trustworthiness of GNNs in the presence of structure and label poisoning, while maintaining their utility?

In this work, we address this challenge by introducing a novel semi-supervised learning framework called deep Self-Training Graph Partition Aggregation (ST-GPA) that successfully overcomes the sparsity problem and enables effective poisoning certification of GNNs against structure and label poisoning (see Fig. 1). In particular, our method enriches the subgraphs created through partitioning the original graph, with synthetic data generated using carefully designed self-training approaches. Self-training is a concept from semi-supervised learning (Chapelle et al., 2006) that refers to training a model on its own predictions to expand the leverageable training set. Concretely, for each sparsely labeled graph partition, our method efficiently generates synthetic (pseudo) labels, expanding the available nodelabel set. In a similar spirit, for each sparsely connected subgraph, our method generates synthetic edges through

![](images/23b6f8eca958f393400d4033f3a0db8a7fb9abe60c19c17b45351c8a16846e0a.jpg)  
Figure 1. Deep Self-Training Graph Partition Aggregation (ST-GPA). To certify against structure poisoning, the graph’s edges are partitioned. The resulting set of sparse graphs are enriched with informative pseudo-edges and then, an ensemble of GNNs is trained on the partitions. To certify against label poisoning, the graph’s node labels are partitions, and the resulting sparsely labeled graphs enhanced with pseudo-labels, before training the ensemble. To jointly certify against both poisoning, ST-GPA partitions both the edges and labels and adds pseudo-edges and pseudo-labels.

## 2. Preliminaries

solving a link prediction problem, endowing each subgraph with a more informative structure. Then, by training GNNs as base classifiers on these enriched subgraphs and aggregating their outputs using a majority vote, we obtain strong certificates for label poisoning, structure poisoning, and joint label-structure poisoning. Our contributions are:

(i) In Sec. 3 we generalize the partition-and-aggregate paradigm prevalent in the image domain to derive poisoning robustness certificates to non-i.i.d structured data. Then, in Sec. 4, we identify and characterize a failure mode of certifying robustness against poisoning using this paradigm on graph-structured data rooted in the label and structure sparsity found in common graph datasets.

(ii) In Sec. 5 we propose deep Self-Training Graph Partition Aggregation (ST-GPA), a novel semi-supervised learning framework that augments partitions with carefully obtained pseudo-labels and/or synthetic edges, overcoming the sparsity problem when partitioning graph-structured data.

(iii) ST-GPA is the first effective certificate against structure and label poisoning for node classification with graph neural networks. In particular, we set a new state-of-the-art certified poisoning robustness for GNNs across multiple benchmarks, while being efficient and obtaining competitive clean accuracies (see Sec. 6).

Overall, our work highlights the importance of self-training in certifying poisoning robustness of GNNs and lays the foundation for future work against training-time attacks in the graph domain. We believe that our insights on improving partition-based certification using self-training may be of independent interest beyond graph learning.

We consider an attributed, undirected graph $G \subseteq \mathbb { G }$ described by a feature matrix $\pmb { X } \in \mathbb { R } ^ { n \times m }$ of m-dimensional features for n nodes and a set of edges $\mathcal { E } \subseteq \{ \{ i , j \} , i , j \in$ [n]} with $[ n ] = \{ 1 , \dots , n \}$ , where two nodes i and j are connected if and only ${ \mathrm { i f ~ } } \{ i , j \} \in { \mathcal { E } }$ , and each node belongs to one out of C classes with $\mathbb { Y } = [ C ]$ We study semisupervised transductive node classification. That is, the full graph $G = ( \mathcal { E } , X , y )$ is available during training but its nodes are only partially labeled. This is modeled with a label vector $\pmb { y } \in ( \mathbb { Y } \cup \{ - 1 \} ) ^ { n }$ , where entry $y _ { i } \in \mathbb { Y } = [ C ]$ indicates the label of node i, and $y _ { i } = - 1$ means the node i is unlabeled. We model GNNs as functions $f : \mathbb { G } \to \mathbb { Y } ^ { n }$ , and collect the labels in the graph in the set $\mathcal { V } = \cup _ { v \in V _ { \mathrm { l a b e l e d } } } \{ ( v , \pmb { y } _ { v } ) \}$ 1 and the node attributes in the set $\mathcal { X } = \cup _ { v \in V } \{ ( v , X _ { v } ) \}$ . We write graph datasets as $D = ( \mathcal { E } , \mathcal { X } , \mathcal { Y } )$

Perturbation model. In this work we study two types of poisoning attacks on graph-structured data: label flipping and structure perturbations. To quantify the strength of an attack, we define attack budgets $r _ { l } , r _ { s }$ , which bound the number of allowed modifications to labels and structure, respectively. We model the set of possible poisoned graphs G<sup>˜</sup> as a ball centered around the clean graph G:

$$
\begin{array} { r } { \mathcal { B } _ { r _ { l } , r _ { s } } ( G ) = \{ \tilde { G } = ( \tilde { \mathcal { E } } , \pmb { X } , \tilde { \pmb { y } } ) \ | \ \delta ( \tilde { \pmb { y } } , \pmb { y } ) \leq r _ { l } , \ } \\ { \Delta ( \tilde { \mathcal { E } } , \mathcal { E } ) \leq r _ { s } \} } \end{array}\tag{1}
$$

where $\Delta ( \mathcal { X } , \mathcal { Y } ) = | ( \mathcal { X } \setminus \mathcal { Y } ) \cup ( \mathcal { Y } \setminus \mathcal { X } ) |$ is the symmetric difference of two sets, and $\begin{array} { r } { \delta ( \tilde { \pmb { y } } , \pmb { y } ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { \tilde { y } _ { i } \neq y _ { i } } } \end{array}$ is the number of different entries (Hamming distance) of two vectors. Eq. (1) models three different perturbation models: (i) label flipping $( r _ { l } > 0 , r _ { s } = 0 )$ ; (ii) structure poisoning $( r _ { l } = 0 , r _ { s } > 0 )$ ; and (iii) both together $( r _ { l } > 0 , r _ { s } > 0 )$

Deep partition aggregation. State-of-the-art certified robustness guarantees for supervised image classification under poisoning attacks have been achieved via the partitionand-aggregate paradigm. The most prominent implementation of which is Deep Partition Aggregation (DPA) (Levine & Feizi, 2021). The core idea is to partition an i.i.d. training set D into k disjoint subsets and to train independent classifiers $f _ { D } ^ { ( i ) } : \mathcal { X }  \mathcal { Y }$ deterministically on each partition i. At inference time, for a given input $x \in \mathcal { X }$ , the final prediction is made via majority vote across all base classifiers:

$$
g _ { D } ( x ) = \arg \operatorname* { m a x } _ { c \in \mathcal { V } } n _ { c } ( D , x ) ,\tag{2}
$$

where $\begin{array} { r } { n _ { c } ( D , x ) = \sum _ { i = 1 } ^ { k } \mathbf { 1 } [ f _ { D } ^ { ( i ) } ( x ) = c ] } \end{array}$ counts the number of classifiers predicting class c for input x. This setup enables a formal robustness guarantee for the aggregated prediction under poisoning:

Theorem 2.1 (Levine & Feizi (2021)). Given a clean dataset D, the majority-vote prediction remains unchanged, $i . e . g _ { D } ( x ) = g _ { \tilde { D } } ( x )$ , for any perturbed dataset $\tilde { D } \in B _ { r } ( D )$ bounded by the attack budget r, as long as

$$
r \leq \left\lfloor \frac { n _ { c } ( D , x ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c } \left( n _ { c ^ { \prime } } ( D , x ) + \mathbf { 1 } _ { c ^ { \prime } < c } \right) } { 2 } \right\rfloor .\tag{3}
$$

where $c = g _ { D } ( x )$ is the predicted class on the clean dataset.

## 3. Generalized Deep Partition Aggregation for Non-i.i.d. Data

Even though DPA has been formulated for image data, the general paradigm to derive poisoning guarantees can be readily generalized to non-i.i.d. structured data. The main idea is based on recognizing that deriving a guarantee like Thm. 2.1 does not depend on the i.i.d. nature of the dataset, but rather on a partitioning scheme h where the potentially poisoned objects are partitioned into subsets independent of one another. In particular, it can be formulated w.r.t. a general set of objects O that may be poisoned, where $o _ { 1 } \in \mathcal { O }$ may not be independent of another $o _ { 2 } \in \mathcal { O } \left( \mathrm { e . g . } \right.$ , in a graph, one node may not be independently sampled from another node), as long as the partition $o _ { 1 }$ is grouped into, is not affected by the value of any other element in O, and thus, poisoning any object can only affect one partition.

To formulate such a split-and-majority voting certificate for general (potentially structured) data, assume a dataset $D = ( \mathcal { T } , \mathcal { O } )$ consisting of a set of objects T that are known to be clean and a set of objects O that may be poisoned. For example, for i.i.d. image data $\mathcal { O } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ and $\tau = \emptyset$ Then, k partitions $\mathcal { P } _ { i }$ with $i \in [ k ]$ are created as follows:

$$
\mathcal { P } _ { i } : = ( \mathcal { T } , \{ o \in \mathcal { O } \mid h ( \mathcal { T } , o ) = i \pmod { k } \} )\tag{4}
$$

where h is a deterministic (hash) function that takes $\tau$ and one $o \in \mathcal { O }$ as input and outputs a number in N representing the partition index, into which the given object o should be grouped. Then, one independently trains base classifiers $f _ { D } ^ { ( i ) } : \mathbb { X }  \mathbb { Y }$ on each partition $\mathcal { P } _ { i } .$ , where X represents a general data domain. Without loss of generality, we assume $f _ { D } ^ { ( i ) }$ to output a scalar class prediction, i.e., $\mathbb { Y } \subseteq \mathbb { N }$ . For example, for node classification, the data domain can be defined as $\mathbb { X } = \left( \mathbb { G } , \mathbb { N } _ { 0 } \right)$ where G is the set of possible graphs and the second element in the tuple refers to the index of a node, for which a class prediction is sought.<sup>1</sup> Now, given an input $x \in \mathbb { X }$ , the final prediction is made as in DPA for images via a majority vote across all base classifiers (Eq. (2)), which we denote $g _ { D } ( x )$

Lastly, for general data domains, the set difference $\Delta ( D , \tilde { D } )$ may not always be an appropriate distance measure. Thus, assume a general distance function $d ( D , \tilde { D } )$ between the clean dataset D and the perturbed dataset ${ \tilde { D } } .$ Further, different choices of hash function h may affect how many partitions $d _ { h } \in \mathbb { N }$ are at most affected when poisoning an object $o \in \mathcal { O }$ . If a perturbation of size $d ( D , \tilde { D } )$ leads to at most p changed partitions given h. Then, the scalar $d _ { h }$ links $d ( D , \tilde { D } )$ to $p$ as follows: $d _ { h } \cdot d ( D , { \tilde { D } } ) \geq p .$ . Exemplary, $d _ { h } > 1$ if one object is partitioned (duplicated) into multiple partitions as done by Wang et al. (2022). Now, we can state the following general theorem that follows the proof strategy outlined by Levine & Feizi (2021) and we refer to App. C.1 for the proof:

Theorem 3.1 (Generalized DPA). Given a clean, possible non-i.i.d and structured dataset D, and a poisoned dataset ${ \tilde { D } } ,$ , the majority-vote classifier prediction remains unchanged, $i . e . , g _ { D } ( x ) = g _ { \tilde { D } } ( x )$ , as long as $d _ { h } d ( D , \tilde { D } ) \leq$ $\lfloor ( n _ { c } ( D , x ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c } { \left( n _ { c ^ { \prime } } ( \bar { D } , x ) + \mathbb { 1 } _ { c ^ { \prime } < c } \right) } ) / 2 \rfloor { = } r _ { m } ( D , x )$ where $c { = } g _ { D } ( x )$ is the predicted class on the clean dataset.

We refer to the right-hand side of the condition in Thm. 3.1 as the robust margin $r _ { m } ( D , x )$ of a sample x given a dataset $D$ . To conclude, we need to find a function $h ,$ a scalar $d _ { h }$ , and a distance measure $d ( D , \tilde { D } )$ to get a robustness guarantee for an ensemble classifier with Thm. 3.1.

Generalized DPA for graphs. We define a general hash function in Eq. (5) that allows to partition a graph based on its node features $\mathcal { X } .$ . This hash function can be used for all three perturbation models captured by Eq. (1):

$$
h ( \mathcal { X } , o ) = \left\{ \begin{array} { l l } { h ( X _ { i } | | X _ { j } ) + h ( X _ { j } | | X _ { i } ) } & { \mathrm { i f ~ } o \in \mathcal { E } } \\ { h ( X _ { v } ) } & { \mathrm { i f ~ } o \in \mathcal { Y } } \end{array} \right.\tag{5}
$$

The hash function $h ( \mathcal { X } , o )$ determines the partition index for an edge, by taking the features of the incident nodes and concatenating them. We add both orderings to make the partitioning process invariant to the actual node order. When partitioning labels, $h ( \mathcal { X } , o )$ takes the corresponding node’s features as input. The h in Eq. (5) can be any hash function, in our experiments we choose the MD5 hash. As each edge or label is in exactly one partition, $d _ { h } = 1$

Finally, we choose three distance metrics for the three different threat models. For label poisoning, we take the Hamming distance $\delta ( { \pmb y } , \tilde { { \pmb y } } )$ between label vectors. For graph structure poisoning, we use the symmetric set difference $\Delta ( \mathcal { E } , \tilde { \mathcal { E } } )$ between edge sets. Inserting or deleting an edge in the graph leads to $\Delta ( \mathcal { E } , \tilde { \mathcal { E } } ) = 1$ and the modification of an edge to $\Delta ( \mathcal { E } , \tilde { \mathcal { E } } ) = 2$ . Similarly, the insertion or deletion of an edge leads to exactly one affected partition based on h, but a modification of an edge affects two. For a combination of label and structure poisoning, we add up the two distances $d ( ( { \mathcal { E } } , { \pmb y } ) , ( { \tilde { \mathcal { E } } } , { \tilde { \pmb y } } ) ) = \delta ( { \pmb y } , { \tilde { \pmb y } } ) + \Delta ( { \mathcal { E } } , { \tilde { \mathcal { E } } } )$ . This is indeed a distance metric (see App. C.2). Given the graph dataset D from Sec. 2, plugging in these distance measures along with our h from Eq. (5) and $d _ { h } = 1$ into Thm. 3.1, we get the following conditions for our certificates to hold:

$$
\tilde { G } \in \mathcal { B } _ { r _ { l } > 0 , r _ { s } = 0 } ( G ) : \delta ( \pmb { y } , \tilde { \pmb { y } } ) \leq r _ { m } ( D , x )\tag{6}
$$

$$
\tilde { G } \in \mathcal { B } _ { r _ { l } = 0 , r _ { s } > 0 } ( G ) : \Delta ( \mathcal { E } , \tilde { \mathcal { E } } ) \leq r _ { m } ( D , x )\tag{7}
$$

$$
\tilde { G } \in \mathcal { B } _ { r _ { l } > 0 , r _ { s } > 0 } ( G ) : d ( ( \mathcal { E } , \pmb { y } ) , ( \tilde { \mathcal { E } } , \tilde { \pmb { y } } ) ) \leq r _ { m } ( D , x )\tag{8}
$$

where $r _ { m } ( D , x )$ is the robust margin introduced in Thm. 3.1. In the context of transductive node classification, x refers to a node that we seek to classify in the graph D. In general, x could refer to a node in an arbitrary graph different to D.

## 4. Limitations of the Simple Partitioning in the Graph Domain

We first expose that the partitioning scheme used by DPA does not work well on graph datasets due to their inherent label and structure sparsity. In the context of image classification, obtaining provable poisoning robustness through partitioning yields significant results (Levine & Feizi, 2021). For example, using 1,000 partitions on CIFAR-10 achieves a certified accuracy of 50% against 392 label flips. The key factor in deriving strong robustness guarantees through partitioning is the number of partitions k. Increasing k increases the possible perturbation budget one can certify. Notably, the maximum number of tolerated perturbations before the certified ratio drops to 0% is $\left\lfloor { \frac { k } { 2 } } \right\rfloor$ . Thus, one can argue that the number of partitions k should be as large as possible to produce good certified radii. The primary constraint on increasing k is the size of the training dataset. With 50,000 training images in CIFAR-10, it is feasible to use k = 1, 000 partitions while maintaining acceptable performance of the base classifiers $f _ { D } ^ { ( i ) }$ . Critically, this approach does not straightforwardly translate to graph learning as we outline in the following.

Commonly used semi-supervised graph datasets come with significantly lower label rates compared to image datasets (see Table 1 in App. A). Regarding label-flipping, applying the partition-

![](images/2c3a417e2f398b3435b8c0b78273983eda45e78188c757c8d02882fe87fa51ad.jpg)  
Figure 2. Label sparsity (Cora-ML)

ing scheme naively to node labels results in partitions with few labeled nodes – often fewer than the number of classes. Fig. 2 shows that when using 40% of nodes for training on Cora-ML, having k = 281 partitions yields an average of only 4 labeled nodes per partition, even though there are 7 classes. This restricts the certification bound to 140 label flips, and base classifiers suffer from significantly reduced performance, as each model only has access to on average 4 labels during training. Even worse, Fig. 4(a) shows that $k = 8 0$ partitions already leads to a clean accuracy of only close to 40% for an ensemble of GCNs (Kipf & Welling, 2017) on Cora-ML, whereas a GCN trained without partitioning achieves ∼78.77% accuracy.

For structure partitioning, the graph structure rapidly deteriorates as edges are divided among partitions, rendering the approach ineffective for GNNs. This is illustrated in

![](images/dfbe6899523a3ed88ab03f5d3779c49c972552a720a7f151900c56fdf731dbac.jpg)  
Figure 3. Structure sparsity (Cora-ML)

Fig. 3, where we plot the mean of average node degrees across partitions. When the number of partitions exceeds $k > 5 0$ , connectivity in the different subgraphs becomes virtually nonexistent. Since GNNs rely on message passing to aggregate information from neighbors, this severely impairs their ability to learn meaningful representations. Under such conditions, the classifiers’ performance deteriorates to that of a multilayer perceptron (MLP), which, however, is already robust to structure perturbations since it does not utilize any edge information. Overall, partitioningbased robustness certificates cannot be naively applied in the graph domain due to the inherent label and structure sparsity, motivating the need for more sophisticated methods.

## 5. Deep Self-Training Graph Partition Aggregation

To address these sparsity challenges, we propose various semi-supervised learning methods to enhance the performance of the weak classifiers by selecting either pseudolabels or edges, or both, on each partition’s limited training data. This enables us to obtain strong certificates on graphs.

Algorithm 1 ST-GPA against label flipping Algorithm 2 ST-GPA against structure perturbations   
Input: Graph data $\boldsymbol { D } \ : = \ : ( \boldsymbol { \mathcal { E } } , \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } )$ , selectiveness t, co- Input: Graph dataset $D { = } ( \mathcal { E } , \mathcal { X } , \mathcal { Y } )$ , selectiveness $\varepsilon ,$ link   
training method $\operatorname { C T } ( { \mathcal { E } } , \chi , y , t )$ , self-training method prediction method $\mathrm { L P } ( e _ { i } , \mathcal { X } , \mathcal { Y } , \varepsilon )$ , hash function h,   
$\operatorname { S T } ( { \mathcal { E } } , { \mathcal { X } } , y , t )$ , training order $o _ { t r n } = ( o p _ { 1 } , \ldots , o p _ { R } ) \in$ number of partitions $k$   
$\{ \mathrm { C } \dot { \mathrm { T } } , \mathrm { S } \mathrm { T } \} ^ { \bar { R } } .$ hash function $h ,$ number of partitions $k$ Output: An ensemble classifier prediction and the associ-  
Output: An ensemble classifier prediction and the associ- ated certificate for a test node v.   
ated certificate for a test node $v .$ 1: Split edges into partitions   
1: Split labels into partitions $e _ { i } = \{ e | e \in \mathcal { E } , h ( \mathcal { X } , e ) \equiv i$ (mod k)}   
$\pmb { y } _ { i } = \{ y | y \in \mathcal { V } , h ( \mathcal { X } , y ) \equiv i$ (mod k)} 2: for each partition i do   
2: for each partition i do 3: $\hat { e } _ { i } = \mathrm { L P } ( e _ { i } , \mathcal { X } , \mathcal { Y } , \varepsilon )$   
3: for each operation op in given order $o _ { t r n }$ do 4: $e _ { i } = e _ { i } \cup \hat { e } _ { i }$   
4: $\hat { \pmb { y } } = o p ( \varepsilon _ { i } , \pmb { \chi } , \pmb { y } _ { i } , t )$ 5: Train $f _ { i }$ on $\hat { D } = \{ e _ { i } , \chi , \mathcal { V } \}$   
5: ${ \pmb y } _ { i } = { \pmb y } _ { i } \cup \hat { \pmb y }$ 6: end for   
6: end for 7: Count base classifier predictions $n _ { c } ( v )$ for all $c \in [ C ]$   
7: Train $f _ { i }$ on $\hat { D } = \{ \mathcal { E } , \mathcal { X } , \boldsymbol { y } _ { i } \}$ 8: Obtain majority class $c ^ { * } = \mathrm { a r g }$ ma $\mathrm { x } _ { c \in [ C ] } n _ { c } ( v )$   
8: end for 9: Calculate robust margin (see Thm. 3.1)   
9: Count base classifier predictions $n _ { c } ( v )$ for all $c \in [ C ]$ $r _ { m } = \lfloor ( n _ { c ^ { * } } ( v ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c ^ { * } } { \left( n _ { c ^ { \prime } } ( v ) + \mathbb { 1 } _ { c ^ { \prime } < c ^ { * } } \right) } ) / 2 \rfloor$   
10: Obtain majority class $c ^ { * } = \arg \operatorname* { m a x } _ { c \in [ C ] } n _ { c } ( v )$ 10: return $c ^ { * } , r _ { m }$   
11: Calculate robust margin (see Thm. 3.1)   
$r _ { m } = \lfloor ( n _ { c ^ { * } } ( v ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c ^ { * } } { \left( n _ { c ^ { \prime } } ( v ) + \mathbb { 1 } _ { c ^ { \prime } < c ^ { * } } \right) } ) / 2 \rfloor$   
12: return $c ^ { * } , r _ { m }$

## 5.1. Semi-Supervised Learning for Label Generation

To maximize utilization of the limited available labels within each partition, we employ two complementary pseudo-label generation methods: (i) Co-Training (CT), and (ii) Self-Training (ST). First, co-training leverages graph structure to propagate existing labels to neighboring nodes. While Li et al. (2018) proposes to generate pseudo-labels for the training of a GNN using ParWalks (Wu et al., 2012), its high computational complexity renders this approach impractical for larger graphs. Instead, we propose the use of label propagation (LP) (Zhu & Ghahramani, 2003) to significantly speed up the co-training process. The key advantage of LP is that it does not require computing the inverse of a Laplacian matrix inherent to ParWalks, and thus can be efficiently applied to larger graphs. Second, for self-training, we propose training a GNN on the existing labeled data and selecting the most confident predictions (as determined by their softmax scores) to serve as pseudo-labels for subsequent training iterations. Notably, both methods (co- and self-training) can be applied consecutively as we demonstrate in Sec. 6, where we observe that applying first co-training and then self-training works best (see Fig. 5(a)).

In LP we propagate a score matrix $S \in \mathbb { R } ^ { n \times c }$ in which $S _ { i , j }$ represents the likelihood of node i belonging to class $j . s$ is initialized as one-hot encoded label matrix $S ^ { ( 0 ) } \stackrel { = } { = } Y$ (where an unlabeled node has a row of zeros in Y ) and propagated with the normalized adjacency matrix $\tilde { A } =$ $\mathbf { \bar { \cal D } } ^ { - \frac { 1 } { 2 } } \mathbf { \bar { \cal A } } \mathbf { \cal D } ^ { - \frac { 1 } { 2 } }$ , as shown in Eq. (9):

$$
\pmb { S } ^ { ( i + 1 ) } = \alpha \tilde { \pmb { A } } \pmb { S } ^ { ( i ) } + ( 1 - \alpha ) \pmb { Y }\tag{9}
$$

where 1 − α is the teleport probability. This propagation is iterated until convergence or a cutoff iteration. Both cotraining and self-training result in a score matrix $S \in \mathbb { R } ^ { n \times c }$ which is used for selecting pseudo-labels. We introduce a hyperparameter t representing the number of pseudo-labels with the highest scores added per class via either method to control the selectiveness. A lower t adds less but often higher-quality pseudo-labels, while higher t adds more but less-confident ones. We present the full certification pipeline against label flipping in Algo. 1. Note that only Lines 9-11 have to be repeated to get a certificate for other nodes.

## 5.2. Semi-Supervised Learning for Edge Generation

Motivated by the concept of self-training in graph machine learning, we extend semi-supervised learning to edge prediction by generating pseudo-edges instead of pseudo-labels. Guided by the homophily assumption, we generate pseudoedges that connect node pairs likely belonging to the same class based on a model trained on the current graph. Here, modern link prediction methods (Zhang & Chen, 2018; Kazi et al., 2023) struggle as the edges in every partition are extremely sparse (see Sec. 4). Instead, we try to capture the overall graph structure and dilute the effect of possible interclass edges by adding in a denser graph than the original one. Within each partition, we first train a GNN to obtain initial node predictions and confidence scores, typically represented by the softmax outputs of the final GNN layer. We then iteratively add edges between node pairs with the highest sum of confidence of being in the same class. The number of edges added is controlled by a hyperparameter ε, which specifies the multiplier of the number of added edges within each partition relative to the original graph. Using this expanded edge set, a second GNN is trained to produce final predictions used for certification. Our certification pipeline against structure perturbation is outlined in Algo. 2. On first sight, a downside of our link prediction scheme is that we sample a denser graph for each partition than the original one. If this is implemented naively, it will have roughly $O ( k n ^ { 2 } )$ time and memory complexity, which does not scale well with the graph size and the number of partitions. To address this, we introduce an efficient algorithm in App. D.1, which has a near-linear complexity based on managing a global edge-candidate heap.

![](images/6b4d6ef353a387c9ad651fa824bd998d49eb6761f764889bb65897173e574cf1.jpg)  
(a) Cora-ML, LF

![](images/a2b8e83579854285f907070623fb4feb1bb4801734de6ba105992c578369291d.jpg)  
(b) PubMed, LF

![](images/3e73513bd51fe07b9edac00e8aae9f012011599c1974456249c9e3ec7e766077.jpg)  
(c) Cora-ML, SP

![](images/d4f716bb4ff858880d86fbdc49375d1fba2da39b9d4acd1a86173ce8b3249675.jpg)  
(d) PubMed, SP

![](images/59e73d0b4ca779d62c39a5377350eea586db16e3e0e2249066217731139e0d0a.jpg)  
(e) PubMed, LF & SP  
Figure 4. The effectiveness of our method is demonstrated by the increase in certified accuracy compared to the naive partition-based approach from Sec. 4. (a) and (b) are certified against Label Flipping (LF) with k = 80 and k = 500 partitions, respectively; (c) and (d) are certified against Structure Perturbations (SP) with k = 800 partitions; (e) is certified against both (LF&SP) with k = 250 partitions. We showcase the stark improvement of robustness both on Cora-ML and PubMed, demonstrating that our method works for both smaller and larger graphs. The red lines in (c) and (d) represent the performance of an infinitely robust MLP, serving as a trivial baseline for structure perturbation. The green dots are the non-robust clean accuracies of the same model trained without any partitions.

With the proposed methods addressing either label flipping or structure perturbations, a scheme for certifying against both types ofattacks becomes possible. We partition both labels and edges into partitions, perform semi-supervised edge and label generation iteratively, and obtain the final base classifier by training on the extended edges and labels, before we take the majority vote and compute certificates. We provide pseudocode for this joint pipeline in App. D.2.

## 6. Experimental Evaluation

In this section, we investigate the robustness guarantees derived by Deep Self-Training Graph Partition Aggregation (ST-GPA) and showcase the improvement of ST-GPA compared to simple partition aggregation on graphs.<sup>2</sup>

Experimental details. We demonstrate results for transductive node classification on four datasets: Cora-ML (Bojchevski & Gunnemann¨ , 2018), and the three Planetoid datasets CiteSeer, Cora, and PubMed (Yang et al., 2016); and for three GNNs: Graph Convolutional Networks (GCN) (Kipf & Welling, 2017), Graph Attention Networks (GAT) (Velickoviˇ c et al.´ , 2018), and APPNP (Klicpera et al., 2019). To train a robust classifier, we partition graphs as described in Sec. 5. Each round of semi-supervised learning adds pseudo-labels or edges to the training sets of the individual partitions, while keeping the partitions isolated. We train an ensemble classifier after each round of pseudo-label or pseudo-edge generation to investigate the effect of the individual semi-supervised learning steps. Results are reported using certified accuracy, which is the percentage of test nodes whose predictions are correct and provably robust, as a function of the perturbation size defined in Eq. (1). In all figures, colored areas represent the standard deviation over 3 seeds. We represent a baseline ensemble classifier trained on partitions without any semi-supervised learning as dashed lines. It is important to note that MLPs exhibit infinite robustness against structure perturbations, since they do not utilize edges during training. Consequently, any model with certified accuracy against structure perturbations below that of an MLP is trivial. Thus, we include an MLP as baseline as a red-dotted line in the structure perturbation plots. In these plots, we discard the trivial part of the curve below this baseline. We provide further details on our setup in App. A.

ST-GPA yields strong certified robustness. Fig. 4 demonstrates the stark improvement of certified accuracies by our proposed certification method ST-GPA, against Label Flipping (LF) in Figs. 4(a) and 4(b), Structure Perturbation (SP) in Figs. 4(c) and 4(d), and both label and structure perturbations in Fig. 4(e). If partitioning is applied naively, i.e., without our semi-supervised learning schemes from Sec. 5, we only get marginal certified accuracy curves (orange dashed lines) due to the sparsity of labels or edges as discussed in Sec. 4. In the structure case on Cora-ML (Fig. 4(c)), the clean accuracy of an ensemble GCN is even worse than an infinitely robust MLP, rendering the naive approach ineffective. In contrast, with our method we restore the ensemble classifier’s clean accuracy to a higher level compared to an ensemble without semi-supervised learning, typically around 70% to 80%, and this also allows the certified accuracy curves to drop down slower, meaning higher certified accuracy against the same number of perturbations. We include as a reference the clean accuracy of a GCN trained without any partitioning, and thus leading

Partition-based certificate w/o semi-supervised learning  
![](images/427b71cb28e12dad8b7cbbc74c2b04a7d314bdc238783f67378fbeb1a9d1b5c7.jpg)  
(a) Cora-ML, LF

![](images/6b1f2f983af2ab9917d2fd88344d1972e2f653521b2f72afeafcc91a69424f78.jpg)  
(b) Cora-ML, LF

![](images/b437036e7cc8fd6bc76b1b24679717126de5b0d4596eddf5cd12344be54862c2.jpg)  
(c) PubMed, LF & SP  
Figure 5. (a) Different orders of co-training and self-training against Label Flipping (LF) on Cora-ML with k = 80 and label propagation as co-training method; (b) Label propagation provides similar performance in co-training compared to ParWalks (Wu et al., 2012) while being scalable to larger graphs, demonstrated on Cora-ML with 80 partitions; (c) Our method provides significant improvements agains Label Flipping (LF) and Structure Perturbations (SP) on PubMed with 200 partitions with link prediction, co-training then self-training, while stacking more rounds of semi-supervised learning in said order yields further but marginal improvement.

to certified accuracies of 0% for any non-zero perturbation.   
Table 2 in App. A reports clean accuracies on other datasets.

For certifying label flipping, we demonstrate the effect of the order of co- and self-training in Fig. 5(a). We find that first co- and then self-training generally works best. This can be understood as self-training involves training a GNN on the available labels in a partition. With k = 80 partitions, each partition contains only ∼10 labels on smaller datasets. In contrast, co-training with label propagation does not require a representation learning step and effectively leverages the graph structure despite the extremely low label rate, producing high-quality pseudo-labels that can be further improved by subsequent self-training rounds. This effect is consistent across all investigated datasets as we show in Fig. 8 in App. B.2. Consequently, we adopt this order for all experiments involving label flipping.

In our certificate against both label flipping and structure perturbations, we perform link prediction, then co-training with label propagation, and finally self-training on labels. We perform several rounds of edge and label generation in this order because label propagation relies on the meaningful graph structure that link prediction generates, and self-training generally works better after co-training, as we find out in Fig. 5(a). In Fig. 5(c) we show that our method introduces significant improvements in the first round, where additional rounds can lead to further improvements.

Co-training with label propagation is similarly performant as ParWalks yet scalable. Since Li et al. (2018) proposed ParWalks in combination with self-training to address low label rates, we compare ParWalks with the Label Propagation (LP) from Zhu & Ghahramani (2003). As shown in Fig. 5(b), the performance advantage of ParWalks over LP is negligible. This is consistent across different datasets as shown in Fig. 10 in App. B.2. Given that LP with random teleportation does not require computing the inverse of the Laplacian, we adopt LP as our primary method.

Our method works with any GNN. The first row in Fig. 6 shows the certification performance of a GCN, GAT, and APPNP with and without our approach. The results show similar performance improvements and trends across all models, with clean accuracy boosted to around 80% for LF, 76% for SP, and 67% for both. We highlight this because our partition-based certificate is architecture-agnostic, and does not assume anything about the underlying classifiers themselves. Thus, our method stays compatible with better underlying GNNs, for which one can expect to produce better certificates. Exemplary, in App. B.4 we use a modern graph transformer as base classifier (Wu et al., 2022).

Our method scales well with partitions. The second row in Fig. 6 shows the effect of varying the number of partitions k. The results demonstrate the effectiveness and scalability of our method: as k increases, which is necessary to derive stronger guarantees, the baseline performance rapidly declines due to the sparsity of labels and structure per partition. However, our experiments indicate that link prediction, co-training, and self-training are required for non-trivial robustness guarantees. This is supported by more results in Fig. 11 in App. B.2. We observe that dividing labels to more than 100 partitions becomes impractical given the training size of 30% on Cora-ML, as partitioning would result in some partitions containing no labeled nodes, making training on those partitions infeasible. Further experiments on the selectiveness hyperparameter t and ε, and label propagation teleportation parameter α are included in App. B.5.

Our label flipping certificate scales well to large graphs. In App. B.3 we show that our label flipping certificate scales well to large graphs such as ogbn-arxiv, or significantly more dense graphs such as Wiki-CS. For more details on the time complexities and scaling behavior of both our label flipping and structure certificate, we refer to App. F, as well as to our discussion in Sec. 8.

Ours Partition-based certificate w/o semi-supervised learning  
![](images/785e5a6928c4eb183bac3033f06a9305e40652b3d670d84775d459d9f5508d8d.jpg)  
(a) LF, Cora-ML, k = 80

![](images/85e34b3654b351b8242dcb5817c12343bf9a86dccda40f13bbde0a18347d4cd2.jpg)  
(b) SP, Cora-ML, k = 800

![](images/a4e55fe32638e57bf60d0ba75cfc4b8e6387f17f47a5a1ea196c605e7f773c52.jpg)  
(c) LF & SP, PubMed, k = 200

![](images/771fe2aaaaf54f44316acffe1cf25bca62f068b05c2e79483cbbe32f03809e3a.jpg)  
(d) LF, Cora-ML

![](images/910fd8fcd645208b79e2af4af6b29871e819a75035aac47ca5c485916be8cde3.jpg)  
(e) SP, Cora-ML

![](images/f40eef5818c02b2a5ab4be4f00f4677e57b56fd5f67796b66824c4ad301dcf8d.jpg)  
(f) LF & SP, PubMed  
Figure 6. In (a) to (c) our method improves upon naive partition-based certification w/o semi-supervised learning, regardless of the GNN type; (d) to (f) demonstrate for GCNs that our method scales very well for larger number of partitions k, which provides better certificates.

## 7. Related Work

Studying upper bounds on adversarial robustness against test- and training-time perturbations through adversarial attacks is well established for message-passing GNNs (Dai et al., 2018; Zugner et al.¨ , 2018; Zugner & G¨ unnemann¨ , 2019; Mujkanovic et al., 2022; Gunnemann¨ , 2022), as well as for more modern graph transformers (Foth et al., 2025). Furthermore, self-training has been successfully used in empirical defenses against such attacks (Li et al., 2024; Lee & Park, 2025). In contrast to adversarial attacks, robustness certificates provide a lower bound on adversarial robustness, and certificates against test-time attacks are well researched for i.i.d. data (Li et al., 2023), as well as for the graph domain (Gunnemann ¨ , 2022; Scholten et al., 2022; 2023; Hojny et al., 2024). However, there are few works studying certification against changes to the training data. For the image domain, there are three main approaches: (i) partition-and-aggregate (Levine & Feizi, 2021), (ii) randomized smoothing (over the training data) (Weber et al., 2023), and (iii) differential privacy (Ma et al., 2019); and we refer to Gosch et al. (2025) for a representative survey.

Most related to our work is the partition-and-aggregate scheme by Levine & Feizi (2021), with various follow-up works (Wang et al., 2022; Chen et al., 2022; Rezaei et al., 2023; Scholten & Gunnemann¨ , 2025; Mohgaonkar et al., 2026; Saxena et al., 2026). However, it was mainly studied in the image domain so far. Regarding graphs, Lai et al.

(2024b) develop a collective, probabilistic poisoning certificate against node injection based on randomized smoothing (Lai et al., 2024a), which however, is not applicable to the perturbation models studied in this work. Gosch et al. (2025) develop a certification paradigm to certify node-feature poisoning, extended to label poisoning by Sabanayagam et al. (2025), but both cases are limited to infinite-width GNNs. Further, the label certificate by Sabanayagam et al. (2025) does only scale to graph datasets having at most 100-200 training labels and thus, does not scale to even our smallest datasets. However, we still provide a comparison to Sabanayagam et al. (2025) in App. E.1. Concurrently to our work, Mohgaonkar et al. (2026) extended the scalability of the label certificate by Sabanayagam et al. (2025) to scale to image datasets. However, the scaling is only possible at additional costs of the clean accuracy, and again, only valid for infinite-width networks. Further, they did not study graph certification. Li et al. (2025) apply partitioning to derive poisoning certificates for GNNs. However, they do not certify against label flipping, and their structure certificates are below the performance of an MLP (see App. E.2) and thus, vacuous. Further, their feature certification is only applicable to graph classification, as every node is in all partitions. Overall, our work is the first to provide effective, non-vacuous certified robustness guarantees for GNNs against both label and structure poisoning in node classification tasks.

## 8. Discussion and Limitations

We’ve shown that our semi-supervised training schemes are essential for meaningful robustness guarantees in the graph domain with partition-based methods. In this section, we address and discuss the limitations of our method and show how this hints to promising research directions.

Application to heterophilic graphs. While our certification framework is general, the chosen semi-supervised training components rely fundamentally on homophily, which limits their applicability to heterophilic graphs, which we empirically quantify through a study on synthetic graphs in App. H. Thus, addressing heterophily requires different semi-supervised learning approaches, such as novel label propagation or link prediction methods tailored to heterophilic structures, each of which represents its own independent research challenge. However, the core ideas of partitioning and self-training remain applicable and could inspire effective robustness guarantees if adapted properly to heterophilic settings.

Link prediction can be expensive. Furthermore, while the label certificate scales well to larger graphs, another important limitation is the computational overhead of link prediction on denser or larger graphs, necessary for effective structure certificates. Our method can produce semi-dense partitions by adding edges to capture the graph structure and diluting the effect of noisy inter-class edges, increasing memory and runtime costs. This overhead prevents scaling link prediction effectively to large datasets like ogbn-arxiv, as shown in our scalability analysis in App. F. While alternative link prediction algorithms might alleviate this, most are designed for full graphs rather than sparse partitions, making this a non-trivial challenge.

For a detailed discussion on tightness, we refer to App. G.

## 9. Conclusion

We derive a self-training framework (ST-GPA) that significantly improves certified robustness against poisoning in sparse graph-structured domains. By adding both synthetic labels and structure through effective semi-supervised learning techniques, our method overcomes the limitations of existing partition-based approaches. Empirical results show large improvements in certified robustness to both label and structure poisoning without compromising clean accuracy. Our findings highlight that effectively leveraging semi-supervised learning on sparse data is essential for provably robust graph machine learning against poisoning through partition-based approaches. This potentially offers a promising direction for building more robust models in semi-supervised settings beyond the graph domain.

## Impact statement

This work advances the field of robustness certification by enabling robustness of graph neural networks against structure and label poisoning attacks, thereby fostering more reliable and trustworthy machine learning. While there might be many further potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

This work has been funded by the DAAD program Konrad Zuse Schools of Excellence in Artificial Intelligence (sponsored by the Federal Ministry of Education and Research) and by the German Research Foundation, grant GU 1409/4-1. The authors of this work take full responsibility for its content.

## References

Bojchevski, A. and Gunnemann, S. Deep gaussian em-¨ bedding of graphs: Unsupervised inductive learning via ranking. In International Conference on Learning Representations (ICLR), 2018.

Chapelle, O., Scholkopf, B., and Zien, A.¨ Semi-Supervised Learning (Adaptive Computation and Machine Learning). MIT Press, 2006.

Chen, R., Li, Z., Li, J., Wu, C., and Yan, J. On collective robustness of bagging against data poisoning. In International Conference on Machine Learning (ICML), 2022.

Dai, H., Li, H., Tian, T., Huang, X., Wang, L., Zhu, J., and Song, L. Adversarial attack on graph structured data. In International Conference on Machine Learning (ICML), 2018.

Foth, P., Gosch, L., Geisler, S., Schwinn, L., and Gunnemann, S. Adversarial robustness of graph trans-¨ formers. Transactions of Machine Learning Research (TMLR), 2025.

Gosch, L., Sturm, D., Geisler, S., and Gunnemann, S. Re-¨ visiting robustness in graph machine learning. In International Conference on Learning Representations (ICLR), 2023.

Gosch, L., Sabanayagam, M., Ghoshdastidar, D., and Gunnemann, S. Provable robustness of (graph) neural¨ networks against data poisoning and backdoor attacks. Transactions on Machine Learning Research (TMLR), 2025.

Gunnemann, S.¨ Graph Neural Networks: Adversarial Robustness, pp. 149–176. Springer Nature Singapore, 2022.

Hojny, C., Zhang, S., Campos, J. S., and Misener, R. Verifying message-passing neural networks via topology-based bounds tightening. In International Conference on Machine Learning (ICML), 2024.

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. Open graph benchmark: Datasets for machine learning on graphs. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

Kazi, A., Cosmo, L., Ahmadi, S.-A., Navab, N., and Bronstein, M. M. Differentiable graph module (dgm) for graph convolutional networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):1606– 1617, 2023.

Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.

Klicpera, J., Bojchevski, A., and Gunnemann, S. Predict¨ then propagate: Graph neural networks meet personalized pagerank. International Conference on Learning Representations (ICLR), 2019.

Koh, P. W., Steinhardt, J., and Liang, P. Stronger data poisoning attacks break data sanitization defenses. Machine Learning, 111:1–47, 2022.

Lai, Y., Pan, B., Chen, K., Yuan, Y., and Zhou, K. Collective certified robustness against graph injection attacks. In International Conference on Machine Learning (ICML), 2024a.

Lai, Y., Zhu, Y., Pan, B., and Zhou, K. Node-aware bismoothing: Certified robustness against graph injection attacks. In IEEE Symposium on Security and Privacy (SP), 2024b.

Lee, W. and Park, H. Self-supervised adversarial purification for graph neural networks. In International Conference on Machine Learning (ICML), 2025.

Levine, A. and Feizi, S. Deep partition aggregation: Provable defense against general poisoning attacks. In International Conference on Learning Representations (ICLR), 2021.

Li, J., Pang, M., Dong, Y., and Wang, B. Deterministic certification of graph neural networks against graph poisoning attacks with arbitrary perturbations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Li, K., Chen, Y., Liu, Y., Wang, J., He, Q., Cheng, M., and Ao, X. Boosting the adversarial robustness of graph neural networks: An OOD perspective. In International Conference on Learning Representations (ICLR), 2024.

Li, L., Xie, T., and Li, B. Sok: Certified robustness for deep neural networks. In IEEE Symposium on Security and Privacy, SP, 2023.

Li, Q., Han, Z., and Wu, X. Deeper insights into graph convolutional networks for semi-supervised learning. In AAAI Conference on Artificial Intelligence (AAAI-18), 2018.

Lingam, V., Akhondzadeh, M. S., and Bojchevski, A. Rethinking label poisoning for GNNs: Pitfalls and attacks. In International Conference on Learning Representations (ICLR), 2024.

Ma, Y., Zhu, X., and Hsu, J. Data poisoning against differentially-private learners: attacks and defenses. In International Joint Conference on Artificial Intelligence (IJCAI), 2019.

Mernyei, P. and Cangea, C. Wiki-cs: A wikipedia-based benchmark for graph neural networks. arXiv preprint arXiv:2007.02901, 2020.

Mohgaonkar, A., Gosch, L., Sabanayagam, M., Ghoshdastidar, D., and Gunnemann, S. Exact certification of neural¨ networks and partition aggregation ensembles against label poisoning. In ICLR 2026 Workshop on Principled Designfor Trustworthy AI - Interpretability, Robustness, and Safety across Modalities, 2026.

Mujkanovic, F., Geisler, S., Gunnemann, S., and Bojchevski,¨ A. Are defenses for graph neural networks robust? In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Rezaei, K., Banihashem, K., Chegini, A., and Feizi, S. Runoff election: Improved provable defense against data poisoning attacks. In International Conference on Machine Learning (ICML), 2023.

Sabanayagam, M., Gosch, L., Gunnemann, S., and Ghosh-¨ dastidar, D. Exact certification of (graph) neural networks against label poisoning. In International Conference on Learning Representations (ICLR), 2025.

Saxena, A., Schuchardt, J., Scholten, Y., and Gunnemann,¨ S. Provable robustness against backdoor attacks via the primal-dual perspective on differential privacy. arXiv preprint arXiv:2605.21780, 2026.

Scholten, Y. and Gunnemann, S. Provably reliable confor-¨ mal prediction sets in the presence of data poisoning. In International Conference on Learning Representations (ICLR), 2025.

Scholten, Y., Schuchardt, J., Geisler, S., Bojchevski, A., and Gunnemann, S. Randomized message-interception¨

smoothing: Gray-box certificates for graph neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Scholten, Y., Schuchardt, J., Bojchevski, A., and Gunnemann, S. Hierarchical randomized smoothing. In¨ Conference on Neural Information Processing Systems (NeurIPS), 2023.

Shchur, O., Mumme, M., Bojchevski, A., and Gunnemann,¨ S. Pitfalls of graph neural network evaluation. Relational Representation Learning Workshop, NeurIPS 2018, 2018.

Velickoviˇ c, P., Cucurull, G., Casanova, A., Romero, A.,´ Lio, P., and Bengio, Y. Graph attention networks. In\` International Conference on Learning Representations (ICLR), 2018.

Wang, W., Levine, A., and Feizi, S. Improved certified defenses against data poisoning with (deterministic) finite aggregation. In International Conference on Machine Learning (ICML), 2022.

Weber, M., Xu, X., Karlas, B., Zhang, C., and Li, B. RAB: Provable Robustness Against Backdoor Attacks . In IEEE Symposium on Security and Privacy (SP), 2023.

Wu, Q., Zhao, W., Li, Z., Wipf, D., and Yan, J. Nodeformer: A scalable graph structure learning transformer for node classification. In Conference on Neural Information Processing Systems (NeurIPS), 2022.

Wu, X.-m., Li, Z., So, A., Wright, J., and Chang, S.-f. Learning with partially absorbing random walks. In Advances in Neural Information Processing Systems (NeurIPS), 2012.

Yang, Z., Cohen, W. W., and Salakhutdinov, R. Revisiting semi-supervised learning with graph embeddings. In International Conference on Machine Learning (ICML), 2016.

Zhang, M. and Chen, Y. Link prediction based on graph neural networks. Conference on Neural Information Processing Systems (NeurIPS), 2018.

Zhu, X. and Ghahramani, Z. Learning from labeled and unlabeled data with label propagation. Journal of Software Engineering and Applications, 5(7), 2003.

Zugner, D. and G¨ unnemann, S. Adversarial attacks on¨ graph neural networks via meta learning. In International Conference on Learning Representations (ICLR), 2019.

Zugner, D., Akbarnejad, A., and G¨ unnemann, S. Adversarial¨ attacks on neural networks for graph data. In SIGKDD, pp. 2847–2856, 2018.

## A. Experiment Setup

Datasets. As described in the beginning of Sec. 6, we use the 3 Planetoid datasets (Yang et al., 2016) Cora, CiteSeer and PubMed, available on PyTorch Geometric,<sup>3</sup> and the citation dataset Cora-ML (Bojchevski & Gunnemann¨ , 2018). As a standard procedure in graph machine learning (Shchur et al., 2018), we preprocess all the datasets by taking the largest connected component and force the graph to be undirected. The statistics of the dataset we use can be found in Table 1. The training, validation and test set nodes are determined with scikit-learn’s train test split() function,<sup>4</sup> with random state fixed to 12138. We first separate the test nodes’ indices from training and validation (and unused), then use this function again to separate the training set from the validation with the same seed. Only the 30% training labels are available to the model during training. Additionally, we evaluate the label certificates on Wiki-CS (Mernyei & Cangea, 2020) and ogbn-arxiv (Hu et al., 2020), and we sample 1/10th of their original training labels. The 2 Amazon datasets (Shchur et al., 2018), Amazon-Photo and Amazon-Computers, are also used for evaluating label certificates, and the same scheme as the Planetoid datasets for splitting the nodes is used.

Table 1. Statistics of Datasets. The number of training labels is 30% of all nodes for Cora-ML, CiteSeer, Cora, PubMed; 60% for Amazon-Computers and Amazon-Photo; and 5% for Wiki-CS and ogbn-arxiv.
<table><tr><td>Name</td><td>#Nodes</td><td># Training Labels</td><td># Edges</td><td># Features</td><td># Classes</td><td>Avg. Degree</td></tr><tr><td>Cora-ML</td><td>2810</td><td>843</td><td>7981</td><td>2879</td><td>7</td><td>2.84</td></tr><tr><td>CiteSeer</td><td>2110</td><td>633</td><td>3668</td><td>3703</td><td>6</td><td>1.74</td></tr><tr><td>Cora</td><td>2708</td><td>812</td><td>5069</td><td>1433</td><td>7</td><td>1.87</td></tr><tr><td>PubMed</td><td>19717</td><td>5915</td><td>44324</td><td>500</td><td>3</td><td>2.25</td></tr><tr><td>Wiki-CS</td><td>11701</td><td>586</td><td>216123</td><td>300</td><td>10</td><td>18.47</td></tr><tr><td>Amazon-Computers</td><td>13752</td><td>8251</td><td>491722</td><td>767</td><td>10</td><td>35.76</td></tr><tr><td>Amazon-Photo</td><td>7650</td><td>4590</td><td>238162</td><td>745</td><td>8</td><td>31.13</td></tr><tr><td>ogbn-arxiv</td><td>169343</td><td>9600</td><td>1166243</td><td>128</td><td>40</td><td>6.89</td></tr></table>

Table 2. Model Clean Accuracies on Datasets. We report the clean accuracy (percentage) on the test nodes with given dataset and model with the standard deviation of 3 repeated experiments with different initialization seeds to the model.
<table><tr><td></td><td>MLP</td><td>GCN</td><td>GAT</td><td>APPNP</td></tr><tr><td>Cora-ML</td><td>74.02(0.26)</td><td>78.77(2.10)</td><td>76.95(0.37)</td><td>85.13(0.24)</td></tr><tr><td>CiteSeer</td><td>70.35(0.20)</td><td>67.46(0.45)</td><td>64.98(2.38)</td><td>72.30(0.61)</td></tr><tr><td>Cora</td><td>67.78(0.06)</td><td>76.18(0.41)</td><td>75.77(1.22)</td><td>83.39(0.82)</td></tr><tr><td>PubMed</td><td>73.09(0.05)</td><td>83.66(1.33)</td><td>79.01(0.56)</td><td>87.49(0.41)</td></tr></table>

We follow a 30%-10%-30% train-validation-test split of node labels for all experiments except for larger datasets following Li et al. (2025). Note that the 10% validation set is typically used for regularization tasks such as hyperparameter tuning and early stopping when training a single GNN. However, since we use a fixed set of hyperparameters and do not employ early stopping, the validation set is not utilized during training.

Model Parameters. We used Graph Convolutional Networks (GCN)(Kipf & Welling, 2017), Graph Attention Networks (GAT), and Approximate Personalized Propagation of Neural Predictions (APPNP)(Klicpera et al., 2019) throughout our evaluations.

Our GCNs consist of 2 layers of GCNConv layer with added self-loops from pytorch geometric.<sup>5</sup> At the bottleneck we use 8 hidden channels, a dropout layer, and ReLU activation. For GAT we use 2 GATConv layers from pytorch geometric.<sup>6</sup> The first layer condense the feature channels to 8 hidden channels with 8 attention heads, and applies dropout and ELU activation, and the second layer then condense the 8 × 8 dimension in the middle to class-wise logits. The APPNP model consists of a 2 layer MLP with 8 hidden channels and then an APPNP layer from pytorch geometric<sup>7</sup> to propagate representations, with number of iterations K = 10, and teleport probability α = 0.1. In structure perturbation experiments specifically, we don’t use dropout as it provides us with better results. For additional evaluation of label certificates on a modern graph transformer, we use NodeFormer (Wu et al., 2022) and the exact setup as their published code. Additionally, we use an 2 layered MLP as baseline against structure perturbations. It also has 8 hidden units and no dropouts. Note that here in our setup, the GCN with an empty graph is strictly equivalent to the MLP. Both other attack models use dropouts with a probability of 0.5. All other unstated parameters follow the pytorch geometric default.

Hyperparameters for Training. For GNN training, we use a fixed set of hyperparameters inherited from Li et al. (2018) , which are commonly used in GNN models. Specifically, all models are trained for 200 epochs without early stopping, using the Adam optimizer with a learning rate of 0.01 and a weight decay of $5 \times 1 0 ^ { - 4 }$ . However, due to the unstable nature of GNN training, we choose the epoch with the lowest loss for prediction. To calculate this without overfitting, we separate the data in each partition in halfs, and use one half for training and another for evaluation of the loss. Note that this spli happens to the partitioned 30% labels or edges each partition have access to.

For label propagation, we iterate until the score matrix converges, but cap off at 100 iterations.

Evaluation Process. As described in Sec. 5, either all edges, the 30% training set labels, or both are partitioned for the weak classifiers, depending on different types of attacks. We perform transductive node classification, meaning that the weak classifiers then have access to all the graph data $G = ( \mathcal { E } , X , y )$ except for the poisoned items, which are partitioned in the first place.

All models are evaluated with at least 3 different deterministically chosen seeds for model initialization to evaluate the repeatability of our results. The seeds are chosen by python’s random module. We seed the random module with a fixed seed of 123456, and then use random.randint(0, 2\*\*32) to generate a deterministic random seed each time we repeat an experiment.

After the training is done, we evaluate the certified accuracies on the 30% test nodes. This is done by storing the class predictions of each weak classifier for each node and calculate the robust margin as introduced in Eq. (8). Then the certified accuracies we reported are given by calculating the ratio of test nodes whose robust margin, together with a specific perturbation size, satisfies Eq. (8).

Hyperparameters Summary. Here we provide a table of all hyperparameters involved as a summary to our description of the experiment setup, alongside their default values and selection criteria. Throughout this paper, all hyperparameters take their default values (if any) unless otherwise stated.

Table 3. Hyperparameters
<table><tr><td>hyperparameter</td><td>description</td><td>attack models</td><td>related models</td><td>default value</td><td>selection criteria</td></tr><tr><td>lr</td><td>learning rate</td><td>all</td><td>all</td><td>0.01</td><td>follows Li et al. (2018)</td></tr><tr><td>wd</td><td>weight decay</td><td>all</td><td>all</td><td>5 × 10 -4</td><td>follows Li et al. (2018)</td></tr><tr><td>ep</td><td>training epochs</td><td>all</td><td>all</td><td>200</td><td>follows Li et al. (2018)</td></tr><tr><td>es</td><td>early stopping</td><td>all</td><td>all</td><td>None</td><td>follows Li et al. (2018)</td></tr><tr><td>init_seed</td><td>seed for python random module</td><td>all</td><td>all</td><td>123456</td><td>1</td></tr><tr><td>repetition</td><td>number of repeated experiments</td><td>all</td><td>all</td><td>3</td><td>1</td></tr><tr><td>num_layers</td><td>number of layers in the model</td><td>all</td><td>all</td><td>2</td><td>prevents overfitting</td></tr><tr><td>hidden_size</td><td>hidden channels</td><td>all</td><td>all</td><td>8</td><td>prevents overfitting</td></tr><tr><td>dropout</td><td>dropout probability in dropout layers</td><td>all</td><td>GCN, GAT</td><td>0.5</td><td>prevents overfitting</td></tr><tr><td>activation</td><td>activation function between 2 layers</td><td>all all</td><td>GCN, APPNP, MLP GAT</td><td>ReLU ELU</td><td>1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>/</td></tr><tr><td>train_size val_size</td><td>% of labels for training</td><td>all all</td><td>all</td><td>30% 10%</td><td>1 /</td></tr><tr><td>test_size</td><td>% of labels for validation % of labels for testing</td><td>all</td><td>all all</td><td>30%</td><td>1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">k</td><td rowspan="3">number of partitions</td><td>LF</td><td rowspan="3">all</td><td>80</td><td>Fig. 6 Fig. 11</td></tr><tr><td>SP LF&amp;SP</td><td></td><td></td></tr><tr><td></td><td>200</td><td>Fig. 6</td></tr><tr><td>order</td><td>order of co-training and self-training</td><td>LF, LF&amp;SP</td><td>all</td><td>C/T then S/T</td><td>Fig. 8</td></tr><tr><td>co-train method</td><td>ParWalks(PW) or label propagation(LP)</td><td>LF, LF&amp;SP</td><td>all</td><td>LP</td><td>Fig. 10</td></tr><tr><td>t</td><td>number of pseudo-labels per class</td><td>LF, LF&amp;SP</td><td>all</td><td>50</td><td>Fig. 14(a)</td></tr><tr><td>α</td><td>teleport probability in label propagation</td><td>LF, LF&amp;SP</td><td>all</td><td>0.9</td><td>Fig. 14(b)</td></tr><tr><td>ε</td><td>× ε pseudo-edges than original graph</td><td>SP</td><td>all</td><td>dataset specific</td><td>Fig. 15</td></tr></table>

## B. Additional Experiment Results

## B.1. Clean Accuracy of Ensembles

![](images/1cd08b0687287da1348a5297238ee3b902973e0145ceab7d8ff106154734112f.jpg)  
(a) Partitioning labels

![](images/98a00a02a2d4bd516c737a97dcf3ca48ca416afe065bbd7261f6119b99dc92b6.jpg)  
(b) Partitioning Edges  
Figure 7. Deterioration of clean accuracy of ensembles on Cora-ML. Standard deviations are reported as colored areas over 3 different seeds.

In Fig. 7 we show the clean accuracies of an ensemble classifier if the partitioning scheme is applied trivially, i.e. without semi-supervised learning. In the label case, the sparsity degrades the clean accuracy of the ensemble as k goes larger. With k = 100 partitions, the clean accuracy drops to just over 20%. In the edge partitioning case, lack of edge information degrades the clean accuracy of an ensemble below that of an MLP’s over about k = 30 partitions. The deterioration of performance shown here necessitate the introduction of semi-supervised learning schemes within the sparse partitions.

## B.2. Additional Results on Other Datasets

Here we present results on other datasets supporting our conclusion from Sec. 6, which have been presented in the main part based on one exemplary dataset. Note that the subfigures marked with (\*) are already present in Sec. 6.

![](images/7f80a80e0f23cccedb102719838bb75f80312f973e6bb2828006eb1de6ec876d.jpg)  
(a) Cora-ML<sup>(∗)</sup>

![](images/e79c69c42886dc152b988ee771c07b3b709be806ba9e5aa56eba6750c7f55d91.jpg)  
(b) CiteSeer

![](images/c90ca1e66772989d616b4f0e916d6cdd1b2d5118de600b6496a02126e3accef7.jpg)  
(c) Cora

![](images/5efa5ded81ea842236610ef6819c82ddc4d0a811f18a9e973164e0ee47bf03e6.jpg)  
(d) PubMed, k = 500  
Figure 8. We show different orders of co-training and self-training against label flipping on different datasets. k = 80 unless otherwise stated.

The order of co-train then self-train generally works the best against label flipping. Fig. 8 showcases this. Our method restores the clean accuracy to generally 70% to 80%, while having higher certified accuracies, despite the poor performance with vanilla partitions.

Against both label and structure perturbations on smaller datasets, our method provides improvement but the limit of k prevents meaningful joint certificates. Due to the extreme sparsity of labels and edges if we are facing poisoning of both, we can’t use too many partitions, typically limiting k to around 10. Although the curve drops too fast due to a small k, yielding hardly any usable certified accuracy against even 1 poisoned label or edge, our method still provides 15% to 20% clean accuracy increase.

Co-training with label propagation is similarly performant as ParWalks yet scalable. Our method has similar certified accuracy compared to ParWalks across datasets, except for CiteSeer where it’s about 10% less. However, we argue that this is compensated by the vastly shorter training time and the characteristic that scales easily to larger datasets by our method.

Link prediction allows arbitrarily large k which generates even better certificates. In Fig. 11, we demonstrate the near-perfect scalability of link prediction on generating certificates. The number of partitions k determines the maximum achievable robustness with the partitioning scheme. Therefore, given sufficient computational resources, increasing k

w/ 2 rounds of edge and label generation

Ours Partition-based certificate w/o semi-supervised learning MLP baseline

![](images/1e79ca6fb0e190134164632d5a1ba1b843545646fab9112568caa20a18a1b56d.jpg)  
(a) Cora-ML, k = 10

![](images/a7965c4d1c5f0c6718047c191661ddcc58ee3ca41c1b44295b9bf5f8835c9a80.jpg)  
(b) CiteSeer, k = 10

w/ 1 round of edge and label generation  
![](images/6a17eed0c66e987f8637f297c3bcaecef3f8e0c85e494552fce776de844b642c.jpg)  
(c) Cora, k = 10

![](images/52e8b286a96bfcacd20582db7d11a9a921cbc644e37daf4e1737c04e02b87269.jpg)  
(d) PubMed<sup>(∗)</sup>, k = 200

Figure 9. Different number of rounds of link prediction + co-training + self-training against both label and structure perturbations.  
![](images/ba6c7de236143aa748d26dcf50543565dd1c8868d2d5141204a51a8ed92f36aa.jpg)  
(a) Cora-ML<sup>(∗)</sup>

![](images/9d97b2d081337be14075ef658d42a1eb2e228467f72c388632bbb52a46c08ccf.jpg)  
(b) CiteSeer

![](images/7edc0e147d46abcd1dfa6dc68ea57ac2b8f9157f697ed74971a43089f5d9f649.jpg)  
(c) Cora

![](images/7675e55efbc45d2f429fc47be861a920273fbdaadc82d0f91d6d0ce0136a29da.jpg)  
(d) PubMed, k = 500  
Figure 10. Comparison between our method (co-training with label propagation) and ParWalks(Wu et al., 2012). The results are achieved by co-training then self-training on given datasets, with k = 80 partitions unless otherwise stated.

![](images/5f559f7e2d797cefc2bc247764dd2fcc3b71a9e784296a9dc8d3c9960bf9aba9.jpg)  
(a) Cora-ML<sup>(∗)</sup>, ε = 12

![](images/60255ee22698183b0148e9c7e89bb51ef3fc994006308ebdef9dfde4b8984608.jpg)  
(b) CiteSeer, ε = 20

![](images/723de2dd73a2b175920bc071e5b9bc562849e91a1d2ca5e7c8f7ff89f3ee2f02.jpg)  
(c) Cora, ε = 20

![](images/fd66601ec7e4b44cf111fba530dabe10ec760e390f8fe6295ae5c7942ba2eebb.jpg)  
(d) PubMed, ε = 12  
Figure 11. Scalability over k with our link prediction method.

directly leads to improved robustness. Furthermore, we emphasize that k is theoretically unbounded in its capacity to enhance robustness against structure perturbations.

## B.3. Certifying Label Flipping on Larger and More Diverse Datasets

Our co-training and self-training scheme also scales to larger and more diverse datasets. Fig. 12 shows the performance gain upon a trivial partitioning scheme on Wiki-CS, amazon-computers, amazon-photos, and ogbn-arxiv. Wiki-CS and the amazon datasets include significantly more edges than the citation datasets. ogbn-arxiv includes significantly more nodes and edges (see also Table 1 in App. A). Our co-training and self-training generally restores clean accuracy well, while also having a higher certified accuracy. For Wiki-CS and ogbn-arxiv we used 1/10 of the label rate as other datasets to demonstrate this improvement with fewer partitions. Using the original split, a larger k would also be possible, and likely improve results (compare Figs. 12(d) and 12(e)). Our structure certificate struggles to scale well to such large and dense datasets, as we discuss in Sec. 8.

![](images/d59e8eff42c19fd6d650dc9ad8c64c706d005e625d46d56380413376e195e324.jpg)  
Number of Labels Pertu(a) Wiki-CS, k = 300

![](images/86030087206b4c5a14cac02f760fb8fec142602564406f28c7c643713687336e.jpg)  
(b) amazon-computers, k = 400

![](images/b00d840dadce9e564a4f5d12b11e4cf6196a39ff4855d4b133063bc65ad2fb52.jpg)  
(c) amazon-photo, k = 400

![](images/80f0aa0397f0640dd081b8349cb94a30ee0a57b116c7041f10e68f3657bcf788.jpg)  
(d) ogbn-arxiv, k = 400

![](images/84297709611a39e13585c5fa32851ca24912d37277298201d34256ddb00db5b6.jpg)  
(e) ogbn-arxiv, k = 800  
Figure 12. Certifying label flipping on larger and denser graph datasets. MLP baseline for ogbn-arxiv refers to an MLP ensemble.

## B.4. Additional Results for Graph Transformers

Our co-training and self-training scheme also applies to modern graph transformers. Fig. 13 shows that our method improves certified accuracy from 20% to over 70%, showcasing that it works well with modern graph transformer architec tures, apart from the result in Figs. 6(a) and 6(b) in Sec. 6. In particular, we find that NodeFormer w/o our semi-supervised learning strategies struggles to learn from sparsely labeled partitions. However, enriching the partitions with pseudo-labels using our proposed co-training scheme leads to good clean accuracies and certified robustness of the GT-ensemble against label poisoning.

![](images/09567bf85a841c2d82e121567dd5255156ec6045427f56e6f9f9798736564524.jpg)  
Number of Labels PerturbedFigure 13. Certifying label flipping on Cora-ML using NodeFormer (k = 40).

Partition-based certificate w/o semi-supervised learning

## B.5. Ablations on Hyperparameters

![](images/cd53a88690eb61092069fb70923256a7366b7cf28ca607a8cccd2f33d7b7ab26.jpg)  
(a) Ablation of t

![](images/13704db1741101bdf1373799230c4e34606976aba38291378ea6d78295f52249.jpg)  
(b) Ablation of α  
Figure 14. Certifying label flipping on Cora-ML with k = 80 partitions.

The hyperparameter t offers a way to balance between performance and robustness. Fig. 14(a) illustrates the impact of varying the number of pseudo-labels t added during training. When t is small, such as $t = 1 \mathrm { o r } t = 1 0 .$ , the pseudo-labels are too selective and insufficient in quantity to effectively train the subsequent GNN. Conversely, when t is too large, for example $t \geq 6 0 0$ , the quality of the pseudo-labels deteriorates, which negatively affects the ensemble classifier’s performance. Therefore, selecting an appropriate range for t is critical to achieving optimal certified accuracy. For Cora-ML, this corresponds roughly to the range $5 0 \leq t \leq 2 0 0$ . Since $t = 5 0$ works already quite well and introduce less computation overhead, we fix $t = 5 0$ for all experiments.

The random teleport probability α is pretty robust, but need to be properly chosen. Fig. 14(b) shows the certificates with co-training then self-training, but varying the random teleport probability α An $\alpha = 0$ means always randomly teleport to a labeled node in co-training, while an $\alpha = 1$ means no random teleport. As demonstrated by the results, the performance generally plateaus when $0 . 0 1 \leq \alpha \leq 0 . 9 9$ . As long as α isn’t chosen to be too large or too small, the certified accuracy remains the best achievable one. We choose to fix $\alpha = 0 . 9$ for all experiments.

![](images/46fb4f991cd8956f5d5d4e88d1214cfa290dbf92c28792ae049926f2e9d22a02.jpg)  
(a) Cora-ML

![](images/0f25b6ef3c0b5dd8eec6e78bc6c57d2ed83864881f0bc1ca4956ef8d87c95060.jpg)  
(b) CiteSeer

![](images/f25303b2e7a9e60524930f11ad4c39bfec8c6a1a33160a249037c3715a1231b7.jpg)  
(c) Cora  
Figure 15. Certifying structure perturbations on the three smaller datasets with k = 800 partitions.

There is a best ε for each dataset that we can tune to maximize robustness. ε serves as a control parameter for the selectiveness in adding pseudo-edges. A too small ε fails to capture sufficient graph structure, while a too large ε introduces noise that reduces the signal-to-noise ratio in the generated graph, thereby degrading performance. So there is a best ε in between, and according to Fig. 15, this best value varies a lot between datasets, even though the size of which are similar. To avoid high overheads in all other experiments, we cap ε to 20.

## C. Proof of Theorems

## C.1. Proof of Generalized DPA

We restate Thm. 3.1 before proving it:

Theorem 3.1 (Generalized DPA). Given a clean, possible non-i.i.d and structured dataset D, and a poisoned dataset ${ \tilde { D } } ,$ , the majority-vote classifier prediction remains unchanged, i.e., $g _ { D } ( x ) \ = \ g _ { \tilde { D } } ( x )$ , as long as $d _ { h } d ( D , \tilde { D } ) \ \leq$ $\big \lfloor \big ( n _ { c } ( D , x ) - \operatorname* { m a x } _ { c ^ { \prime } \neq c } \big ( n _ { c ^ { \prime } } ( D , x ) + \mathbb { 1 } _ { c ^ { \prime } < c } \big ) \big ) / 2 \big \rfloor = r _ { m } ( D , x )$ , where $c { = } g _ { D } ( x )$ is the predicted class on the clean dataset.

Proof. We first introduce a lemma that gets rid of the floor operation:

Lemma C.1 (Floor operation equivalency).

$$
a \leq \lfloor \frac { b } { 2 } \rfloor \Leftrightarrow 2 a \leq b , \forall a , b \in \mathbb { N }\tag{10}
$$

Proof. $\begin{array} { r } { a \leq \lfloor \frac { b } { 2 } \rfloor \leq \frac { b } { 2 } \Rightarrow a \leq \frac { b } { 2 } } \end{array}$ and 2a $\leq b \Rightarrow a \leq { \frac { b } { 2 } } \Rightarrow a = \lfloor a \rfloor \leq \lfloor { \frac { b } { 2 } } \rfloor$

We use the shorthand notation of $n _ { c }$ being $n _ { c } ( D , x )$ and $\tilde { n } _ { c }$ being $n _ { c } ( \tilde { D } , x )$ , and $r = d ( D , \tilde { D } )$ Given Thm. 3.1 and Lemma C.1

$$
2 d _ { h } \cdot r \le n _ { c } - \operatorname* { m a x } _ { c ^ { \prime } \ne c } ( n _ { c ^ { \prime } } + { \bf 1 } _ { c ^ { \prime } < c } )\tag{11}
$$

We get rid of the max operation by taking any class and move all terms to one side

$$
0 \leq n _ { c } - n _ { c ^ { \prime } } - \mathbf { 1 } _ { c ^ { \prime } < c } - 2 d _ { h } \cdot r , \forall c ^ { \prime } \neq c\tag{12}
$$

Because the training is conducted in a deterministic manner, the classifier will give the same prediction if the training data is the same. So the number of weak classifiers that predicts differently for a node $f _ { i } ( v ) \neq \tilde { f } _ { i } ( v )$ is also at most $d _ { h } \cdot r .$ So for any class, the number of predictions changed is bounded by

$$
\forall \bar { c } \in [ C ] , | n _ { \bar { c } } - \tilde { n } _ { \bar { c } } | \leq d _ { h } \cdot r\tag{13}
$$

From Eq. (13) we plug in $\bar { c } = c$ and $\bar { c } = c ^ { \prime }$ to get

$$
n _ { c } - \tilde { n } _ { c } \leq d _ { h } \cdot r\tag{14}
$$

$$
\tilde { n } _ { c ^ { \prime } } - n _ { c ^ { \prime } } \leq d _ { h } \cdot r\tag{15}
$$

which is equivalent to

$$
n _ { c } \le \tilde { n } _ { c } + d _ { h } \cdot r
$$

$$
- \boldsymbol { n _ { c ^ { \prime } } } \le - \boldsymbol { \tilde { n } _ { c ^ { \prime } } } + \boldsymbol { d _ { h } } \cdot \boldsymbol { r }\tag{16}
$$

(17)

Plugging this to Eq. (12), we have

$$
\forall c ^ { \prime } \neq c , 0 \leq n _ { c } - n _ { c ^ { \prime } } - \mathbf { 1 } _ { c ^ { \prime } < c } - 2 d _ { h } \cdot r\tag{18}
$$

$$
\leq \underbrace { \tilde { n } _ { c } + d _ { h } \cdot r } _ { \mathrm { E q . ~ ( 1 6 ) } } { \underbrace { - \tilde { n } _ { c ^ { \prime } } + d _ { h } \cdot r } _ { \mathrm { E q . ~ ( 1 7 ) } } } - \mathbf { 1 } _ { c ^ { \prime } < c } - 2 d _ { h } \cdot r\tag{19}
$$

$$
= \tilde { n } _ { c } - \tilde { n } _ { c ^ { \prime } } - \mathbf { 1 } _ { c ^ { \prime } < c }\tag{20}
$$

So for the ensemble classifier trained on poisoned data

$$
\tilde { n } _ { c } \ge \operatorname* { m a x } _ { c ^ { \prime } \ne c } ( \tilde { n } _ { c ^ { \prime } } + \mathbf { 1 } _ { c ^ { \prime } < c } )\tag{21}
$$

and the ensemble classifier’s prediction is unchanged because c is the majority class.

## C.2. Proof of Distance Function

The following is a proof that $d ( ( \mathcal { E } , \pmb { y } ) , ( \tilde { \pmb { \mathcal { E } } } , \tilde { \pmb { y } } ) )$ is a distance.

Proof. We recall that $d ( ( \mathcal { E } , \pmb { y } ) , ( \tilde { \pmb { \mathcal { E } } } , \tilde { \pmb { y } } ) )$ ) is defined to be the sum of symmetric set difference between edges and the hamming distance between labels

$$
d ( ( \mathcal { E } , \pmb { y } ) , ( \tilde { \mathcal { E } } , \tilde { \pmb { y } } ) ) = \Delta ( \mathcal { E } , \tilde { \mathcal { E } } ) + \delta ( \pmb { y } , \tilde { \pmb { y } } )\tag{22}
$$

and a function $d \colon M \times M \to \mathbb { R }$ is a distance on metric space M if $\forall x \in M , d ( x , x ) = 0 , d ( x , y ) > 0 , x \neq y$ $d ( x , y ) = d ( y , x )$ , and $d ( x , y ) \leq d ( x , z ) + d ( y , z )$

We first note that $d ( \cdot , \cdot )$ between one $( \mathcal E , \pmb { y } )$ and itself is 0 because both its terms are 0; It’s value is always positive as it’s the sum of two positive numbers, if both input are distinct; $d ( \cdot , \cdot )$ is symmetric because both its terms are symmetric and changing the order of objects doesn’t affect the value.

To prove triangle inequality

$$
d ( ( { \mathcal { E } } _ { 1 } , y _ { 1 } ) , ( { \mathcal { E } } _ { 3 } , y _ { 3 } ) ) = \Delta ( { \mathcal { E } } _ { 1 } , { \mathcal { E } } _ { 3 } ) + \delta ( y _ { 1 } , y _ { 3 } )\tag{23}
$$

$$
\leq \Delta ( \mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } ) + \Delta ( \mathcal { E } _ { 2 } , \mathcal { E } _ { 3 } ) + \delta ( \pmb { y } _ { 1 } , \pmb { y } _ { 2 } ) + \delta ( \pmb { y } _ { 2 } , \pmb { y } _ { 3 } )\tag{24}
$$

$$
= d ( ( { \mathcal { E } } _ { 1 } , y _ { 1 } ) , ( { \mathcal { E } } _ { 2 } , y _ { 2 } ) ) + d ( ( { \mathcal { E } } _ { 2 } , y _ { 2 } ) , ( { \mathcal { E } } _ { 3 } , y _ { 3 } ) )\tag{25}
$$

So $d ( ( \mathcal { E } , \pmb { y } ) , ( \tilde { \mathcal { E } } , \tilde { \pmb { y } } )$ is indeed a distance metric.

Algorithm 3 Efficient Edge Candidate Selection for Link Augmentation   
Input: Number of nodes per class $n _ { c } ,$ node confidence scores per class:   
$\{ S _ { c } = [ s _ { c , 1 } , s _ { c , 2 } , \ldots , s _ { c , n _ { c } } ] \}$ sorted descending, desired number of edges to add $\varepsilon \cdot e$   
Output: Top-ε · e pseudo-edges with highest sum of confidence scores   
1: Initialize global max-heap G (size = number of classes)   
2: for each class c do   
3: Initialize empty local max-heap $L _ { c }$   
4: Insert initial pair (0, 1) with score $s = S _ { c } [ u ] + S _ { c } [ v ]$ and class index as tuple $( u , v , s , c )$ into $G$   
5: end for   
6: while number of added edges smaller than $\varepsilon \cdot e$ do   
7: Extract top $( u , v , s , c )$ from global heap G   
8: Add $\boldsymbol { e } = \left( u , v \right)$ to candidate edge set   
9: For local heap $L _ { c } ,$ consider children pairs:   
10: $c h i l d _ { 1 } = ( u , v + 1 ) { \mathrm { i f } } v + 1 \leq n _ { c }$   
11: $c h i l d _ { 2 } = ( u + 1 , v ) { \mathrm { i f } } \ u + 2 = v$   
12: for each valid child pair $( u , v )$ do   
13: Compute sum of score $s = \dot { S } _ { c } [ u ] + S _ { c } [ v ]$   
14: Insert tuple $( u , v , s )$ into $L _ { c }$   
15: end for   
16: Extract next top tuple $( u ^ { \prime } , v ^ { \prime } , s )$ from $L _ { c }$   
17: Insert $( u ^ { \prime } , v ^ { \prime } , s , c )$ into global heap $G$   
18: end while   
19: return Final candidate edges across all classes

## D. Algorithms

## D.1. Efficient Selection of Pseudo-Edge Candidates

As the number of nodes, k or ε goes up, the number of possible edge pairs also scales up, which results in significant computational overhead. To address this challenge, we propose a time and memory-efficient algorithm for selecting edge candidates with the highest combined confidence scores across all classes. The algorithm for selecting edges is described in Algo. 3. A global max-heap keeps track of which class has the next best edge candidate according to the sum of scores. We initialize the global heap by inserting the first pair (0, 1) from each class, as these pairs hold the maximum possible sum of scores within their respective classes. To add edges, we repeatedly pick the top element from the global heap until the number of edges selected exceeds the threshold defined by ε.

To supply the global heap with the best candidates from each class, every class maintains a local heap, which is initially empty. Each time an edge is selected from the global heap, the algorithm accesses the corresponding class’s local heap and inserts the next candidate pairs. To prevent local heaps from becoming prohibitively large, we do not insert all possible node pairs at once. Instead, node pairs are added only when they can potentially represent the best candidate. This is enabled by pre-sorting nodes and their confidence scores within each class in descending order. Consequently, nodes with smaller indices correspond to higher scores. This ordering induces a binary tree structure over node pairs, illustrated in Fig. 16, where an edge $( u , v )$ has a larger combined score than $( u , v + 1 )$ , and $( u + 1 , v )$ . As the criterion, the sum of scores, is commutative, $( u , v )$ and $( v , u )$ represent the same edge candidate, so we consider only node pairs $( u , v )$ where $u < v$ . This results in the half binary tree structure in Fig. 16.

![](images/388be94d0fbb9eccc1525622b79de5ca5c3761be60b7bc1ab05e4c6b8669af90.jpg)  
Figure 16. Tree structure for spawning edge candidates. An edge in layer i is guaranteed to have a higher score than in layer $i + 1$ and a lower score than in layer $i - 1$

Since most nodes have two parents, simply adding both $( u , v + 1 )$ and $( u + 1 , v )$ to the local heap would result in duplicate insertions of many nodes. To avoid this, we add the pair $( u + 1 , v )$ only when $u + 2 = v$ . This condition effectively removes the red dashed connections shown in Fig. 16 and ensures that each edge pair is added to the local heap exactly once. This also guarantees that the local heap always contains the node pair with the highest possible sum of scores.

## D.2. Proposed General Certification Pipeline Against Both Label Flipping and Structure Perturbations

Algo. 4 shows the joint certification pipeline for ST-GPA. Regarding the training order, we do edge generation first, since we find effective co-training relies on meaningful graph structure. Further, we assume the number of label partitions $k _ { y }$ and structure partitions $k _ { e }$ are the same, i.e., $k = k _ { e } = k _ { y } .$ Otherwise, the partitioning in Lines 1-2 has to be adapted to duplicate the objects associated to the smaller partition number into all partitions, affecting $d _ { h }$ when calculating the certificate using the base classifier counts based on Thm. 3.1.

```latex
Algorithm 4 ST-GPA joint pipeline
Input: Graph dataset $\boldsymbol { D } \ : = \ : ( \boldsymbol { \mathcal { E } } , \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { Y } } )$ , a label propagation method $\hat { \pmb y } = L P ( e _ { i } , \pmb \chi , \pmb y )$ , a label self-training method
$\pmb { \hat { y } } = L S T ( e _ { i } , \pmb { \chi } , \pmb { y } )$ , an link prediction method $\hat { e } _ { i } = E S T ( e _ { i } , \mathscr { X } , \pmb { y } )$ , a training order $o _ { t r n } = ( o p _ { 1 } , \ldots , o p _ { R } ) \in$
$\{ { \mathrm { L P } } , { \mathrm { L S T } } , { \mathrm { E S T } } \} ^ { R }$ , number of partitions $k$
Output: An ensemble classifier prediction and the associated certificate for a test node v.
1: Split labels into partitions $\pmb { y } _ { i } = \{ y | y \in \mathcal { V } , h ( \mathcal { X } , y ) \equiv i$ (mod k)}
2: Split edges into partitions $e _ { i } = \{ e | e \in \mathcal { E } , h ( \mathcal { X } , e ) \equiv i$ (mod $k ) \}$
3: for each operation op in the given order $o _ { t r n }$ do
4: for each partition i do
5: $\boldsymbol { \hat { e } } _ { i }$ or $\hat { \pmb y } = o p ( e _ { i } , \pmb { \chi } , \pmb { y } _ { i } )$
6: $e _ { i } = e _ { i } \cup \hat { e } _ { i }$ or ${ \pmb y } _ { i } = { \pmb y } _ { i } \cup \hat { \pmb y }$
7: end for
8: end for
9: Train base classifiers $f _ { i }$ for each partition i on final edges, labels and features, i.e., on $\hat { D } = \{ e _ { i } , \chi , y _ { i } \}$
10: Count base classifier predictions $n _ { c } ( v )$ for all $c \in [ C ]$
11: Obtain majority class $c ^ { * } = \arg \operatorname* { m a x } _ { c \in [ C ] } n _ { c } ( v )$
12: Calculate robust margin (see Thm. 3.1)
$\begin{array} { r } { r _ { m } = \lfloor ( n _ { c ^ { * } } - \operatorname* { m a x } _ { c ^ { \prime } \neq c ^ { * } } { \left( n _ { c ^ { \prime } } ( v ) + \mathbb { 1 } _ { c ^ { \prime } < c ^ { * } } \right) } ) / 2 \rfloor } \end{array}$
13: return $c ^ { * } , r _ { m }$
```

## E. Comparison to Existing Works

## E.1. Comparison to LabelCert for Label Flipping

LabelCert (Sabanayagam et al., 2025) introduce the first exact certification method for GNNs against label poisoning, leveraging the Neural Tangent Kernel to reformulate the bilevel poisoning problem as a Mixed-Integer Linear Program. While their certificates are exact in the infinite-width limit, the MILP-based approach is computationally tractable only for binary classification tasks with a very small number of labels (20-40 in total), which represents a highly constrained experimental regime. Evaluation on a binary class Cora-ML dataset (Table 4) reveals that our method achieves certified accuracies that are remarkably competitive despite this sub-optimal setting for label partitioning. Specifically, our approach matches LabelCert within a small margin at a 5% perturbation budget and outperforms it at 10% perturbations. Furthermore, our method scales to graph datasets several orders of magnitude larger in both number of nodes and number of label classes than what LabelCert can handle, demonstrating a practical advantage that extends well beyond the narrow binary setting in which exact MILP-based certification is feasible.

Table 4. Certified accuracy (%) on Cora-ML (binary) under label poisoning, comparing our method to LabelCert.
<table><tr><td rowspan="2">Method</td><td colspan="2">Label Perturbations</td></tr><tr><td>5%</td><td>10%</td></tr><tr><td>Ours (co-train then self-train)</td><td> $6 6 . 5 \pm 1 7 . 8$ </td><td> $3 6 . 2 \pm 1 . 4$ </td></tr><tr><td>LabelCert</td><td> $6 8 . 7 \pm 6 . 8$ </td><td> $3 2 . 6 \pm 2 . 0$ </td></tr></table>

## E.2. Discussion on the Relation to PGNNCert Against Structure Perturbations

Li et al. (2025) apply a similar partitioning scheme PGNNCert to derive poisoning certificates for GNNs. In their work, a threat model of arbitrarily perturbing edges, nodes, and node features (but not labels) is considered. In this section, we provide comparison between our results and theirs. Due to that their certified accuracy does not outperform an infinitely robust MLP, we do not report them in Sec. 6.

To compare our works directly, we adopted their published code and use their exact dataset splits. We then run our proposed approach on their data split. Note that the only difference in the setup is the model and hyperparameters for training, where we use a 2-layered GCN as described in App. A, and they use a 3-layered GCN with skip connections and linear layer. In PGNNCert, 2 different partitioning schemes were proposed, namely edge centric and node centric, which has similar performance against structure perturbations, so we report the result with the node centric variant only. Due to the limitation of implementation of PGNNCert, their code does not scale to larger number of partitions due to memory limits, so we choose k = 60 as a compromise and keep other parameters exactly the same for a fair comparison. Note that normally our link prediction scheme can be used with significantly larger k-s to generate competitive certificates, so we also report k = 600 with the same setup. Although PGNNCert doesn’t report results on Cora, their code can be easily adapted as the Planetoid datasets share the same data loader in PyTorch, so we report results on Cora as well.

As shown in Fig. 17, PGNNCert-N’s performance is consistently outperformed by the infinitely robust MLP baseline on al datasets. In the mean time, our link prediction method shows similar improvements to certified accuracy over the MLP baseline as previously reported in Sec. 6 and App. B. At k = 60, the allowed perturbation budget is very small. But the reported k = 600 curves shows the normal performance of our method with more partitions, yielding larger perturbation budgets.

Provided that PGNNCert’s performance is below the MLP baseline in the case of structure perturbation, we don’t report a comparison to PGNNCert in the results in this paper, as we already use the MLP baseline as a lowest acceptable case in al our results against structure perturbation.

In PGNNCert, no certificate against label flipping is reported. As we have already shown in App. B.1, the partitioning scheme does not readily transfer to labels, so we don’t compare our results with PGNNCert on label flipping.

![](images/f8aae36e5d6e4a86b8b3dc2f036a020348d05f8a70a48866a5d75565a38c2cda.jpg)  
(a) Cora-ML, magnified

![](images/09474f564754b1aa27e1153c5fa55c7f05392a1bc1c01a377f58dfe6600ba1f0.jpg)  
(b) CiteSeer, magnified

![](images/496a306203d0163e42bc631d60df2b7031106613a2a45dc323c202e7bc8940fb.jpg)  
(c) Cora, magnified  
Figure 17. Certified accuracies of our method compared to Li et al. (2025), ε = 30. PGNNCert-N’s performance falls steadily below the MLP baseline which is infinitely robust to structure perturbation; a trivially applied partitioning scheme has only on-par clean accuracy with an MLP, yielding hardly any meaningful perturbation budget; with our proposed link prediction method, the certified accuracies is most competitive.

## F. Scalability of Our Methods

In this section, we analyze the scalability of our three proposed self-training methods by reporting the relative time used in our experiments. In all our experiments, we use a single NVIDIA GTX1080Ti GPU. The link prediction uses CPU only. While it can be easily parallelized, we report the performance on 1 CPU core only, as link prediction on individual partitions are usually done so. The times reported are for relative reference, considering the number of nodes and edges in each dataset.

We first point out that the time complexity scales linearly with the number of partitions k, since the partitions are disjoint once they are created. Partitioning the dataset takes a negligible amount of time, and can easily be accelerated by pre-processing the dataset and storing all the partition indices. Consequently, it is reasonable that we only report the time taken per partition in this section. The memory complexity is irrelevant w.r.t. k by the same reason that the partitions are disjoint. The training on individual partitions could be done in a parallel or distributed manner.

Table 5. Time used for co-training and self-training with t = 50
<table><tr><td></td><td>Cora-ML</td><td>CiteSeer</td><td>Cora</td><td>PubMed</td><td>Wiki-CS</td><td>ogbn-arxiv</td></tr><tr><td>graph size (Nr. nodes)</td><td>2810</td><td>2110</td><td>2708</td><td>19717</td><td>11701</td><td>169343</td></tr><tr><td>training time per partition (s)</td><td>6.4</td><td>5.5</td><td>5.5</td><td>2.7</td><td>3.1</td><td>40.9</td></tr><tr><td>thereof training w/o SSL</td><td>36.1%</td><td>37.3%</td><td>36.4%</td><td>31.9%</td><td>29.0%</td><td>31.4%</td></tr><tr><td>thereof C/T</td><td>26.8%</td><td>24.7%</td><td>25.2%</td><td>30.8%</td><td>36.4%</td><td>37.0%</td></tr><tr><td>thereof S/T</td><td>37.1%</td><td>38.0%</td><td>38.4%</td><td>37.3%</td><td>34.6%</td><td>31.6%</td></tr></table>

Table 5 shows the time taken to train an individual partition on all datasets we tested. Label partition training is generally very fast. Note that the time portion reported for training without semi-supervised learning represents the time baseline for training on the raw partition, which has also similar time complexity of training on the clean graph. Judging from the percentages reported, co-training and self-training is just another round of training of a GNN, which takes similar time as training the first one on the raw partition. The results show good scalability of our label self-training methods as the time portion stays roughly equal regardless of the size of the dataset, and the total time necessary to train the ensemble scales roughly linearly with the graph size (number of nodes)

Table 6 shows the time taken per partition for edge semi-supervised training. Although smaller datasets has faster training time per partition, more partitions are required in structure perturbation to generate a meaningful certificate. Therefore, the time taken to train an edge ensemble is usually comparably longer than a label ensemble. In the smaller datasets, training without and with the pseudo-edges take roughly the same time. However, as the number of edges grows, training with pseudo-edges becomes the predominant factor in total training time, and the time taken to find the pseudo-edges becomes non-negligible. On larger graphs, this is especially a choking factor as we usually add more pseudo-edges than there originally are (ε ≫ 1.0), making the link prediction very time-consuming.

Table 6. Time used for link prediction with ε = 1.0
<table><tr><td></td><td>Cora-ML</td><td>CiteSeer</td><td>Cora</td><td>PubMed</td><td>Wiki-CS</td><td>ogbn-arxiv</td></tr><tr><td>graph size (Nr. edges)</td><td>7981</td><td>3668</td><td>5069</td><td>44324</td><td>216343</td><td>1166243</td></tr><tr><td>training time per partition (s)</td><td>1.9</td><td>1.0</td><td>2.0</td><td>9.3</td><td>19.3</td><td>67.8</td></tr><tr><td>thereof training w/o SSL</td><td>32.3%</td><td>58.7%</td><td>57.5%</td><td>14.4%</td><td>8.2%</td><td>6.6%</td></tr><tr><td>thereof L/P</td><td>7.3%</td><td>3.8%</td><td>5.5%</td><td>13.8%</td><td>34.5%</td><td>42.8%</td></tr><tr><td>thereof training with pseudo-edges</td><td>60.4%</td><td>37.5%</td><td>37.0%</td><td>71.8%</td><td>57.2%</td><td>50.6%</td></tr></table>

## G. Empirical Tightness and Transfer Attacks

Since our method provides exact certification that already incorporates the lower bound on robustness, a key open question is how tight these certificates are in practice. To investigate this empirically, we conduct a transfer attack using Meta-PGD from Mujkanovic et al. (2022), transferring adversarial edges crafted against a GCN surrogate to our partition-and-ensemble framework. We note that gradient-based attacks such as PR-BCD (Mujkanovic et al., 2022) cannot be directly applied to ensemble-based methods, since doing so would require differentiating through the random partitioning and unrolling the training of many base classifiers on distinct graph subsets, a significant technical challenge for which no established solution currently exists. Shown in Fig. 18, the transfer attack proves largely ineffective: robust accuracy remains stable at k = 800. We attribute this to two factors: the non-adaptiveness of the attack, which is optimized against a single GCN rather than the ensemble, and the strong graph partitioning inherent to our method, which dilutes the influence of any fixed set of adversarial edges across independently trained base classifiers. Here we want to note the findings from Mohgaonkar et al. (2026), who demonstrate on image datasets that DPA-based methods for label certification tend to be overly conservative for small k, but approach tightness for larger k. Thus, one could expect similar behavior for graphs.

![](images/83c07937d3ea4df7365095d52cba1f7ca2d8d44f603583fb10304eed6eae47e8.jpg)  
Figure 18. Performing a transfer attack from a GCN using Meta-PGD for structural poisoning. We find the transfer attack from a GCN to be mostly ineffective in reducing the accuracy of the ensemble (k = 800 for both, certified and robust accuracy).

These results motivate an important open problem: the development of adaptive poisoning attacks tailored specifically to partition-and-ensemble methods. Such attacks would need to simultaneously optimize adversarial edges with respect to the partitioning scheme and the ensemble decision rule, which is a non-trivial objective given the discrete and stochastic nature of graph partitioning. Progress in this direction would enable a rigorous empirical assessment of certificate tightness beyond what transfer attacks can currently provide, and we view it as an important avenue for future work.

## H. Discussion on the Impact of Homophily

In this section, we conduct a systematic study on Contextual Stochastic Block Models (CSBMs), investigating what threshold of homophily our certification method requires for good empirical results. We vary homophily from 0% (pure heterophily) to 100% (pure homophily) in terms of the percentage of edges that connect same-class nodes. We find that both our certificates against label and structure perturbations start to provide benefits from 60% homophily, with strong results when the homophily exceeds 70%. (For context, Cora has 81% homophily.) The detailed results are shown in Figs. 19 and 20, and we refer to the figure captions for details on the experimental setup.

![](images/73b3ef45026b7451d203ba7340ab6db6141e54e500f14ac67b5679201a7c5f48.jpg)  
(a) h = 100%

![](images/5f6dc548a489130002e7f83ccec3a821d6d0987312bdbfb7851e130566197900.jpg)  
(b) h = 90%

Partition-based certificate w/o semi-supervised learning

![](images/c1d2ad0b2d774130199871a9d432b1735a514a13362c937f1c173adeb9135803.jpg)  
(c) h = 81% (Cora)

![](images/cadd31d584b267119b521dcd58312c489b5f080a932af998d4c9f174155a577d.jpg)  
(d) h = 75%

![](images/7603f0a9e2f471603734335966f33ade4934c158f4e1c0db51e305c71a731adf.jpg)  
(e) h = 70%

![](images/b64b1b0e5780a8a58304c275b604a8edcce385142cb8523196cc166e4f611916.jpg)  
(f) h = 60%

![](images/a4807e1fa8d71f288fc10f8fa647e5ba5e7d105f19831b4f763522cf4bf69d9d.jpg)  
(g) h = 50%

![](images/d1206dd6c9278aba4de5614c9ce810605db364f9a400f2dbd47f7e1a54a6a4f6.jpg)  
(h) h = 40%

![](images/33aa7b7787210dfefdd41103f479a1bcb4dd94526003316048a1b1064426a987.jpg)  
(i) h = 30%

![](images/f8ffe00d8314c042c2f91aafa23f4a324513006f111c76b47738b5c955e3a3ef.jpg)  
(j) h = 20%

![](images/56e066b06c515e684768d81d906642733c8c590844d7dcbd13c04fa3ca4cd448.jpg)  
(k) h = 10%

![](images/21468c33f368d71f9143d73480e7d43bc02c0fd371f162de21b545d8c0bfcea6.jpg)  
(l) h = 0%

Figure 19. Label flipping certificate results with k = 50 on CSBM graphs with different levels of homophily. h = percentage of edges tha connect same class nodes. CSBM: 774 nodes (same number of nodes/class and edge density as Cora). Other parametrization following Gosch et al. (2023) with $K = 1 . 5 ,$ , and σ = 1. We use 60% nodes for training and 20% for validation and testing respectively.

![](images/de13d7dac4d0bd004094011fd1034132ece0a31bcf1d79acd6838d9030be0e1e.jpg)  
(a) h = 100%

![](images/f806ee4d677dbe97d0f13f8563134b190bfabdf5736f47c7441a83b6bc3729cd.jpg)  
(b) h = 90%

![](images/5e60cb3eb77f49081f952640afbc63b304ab55e9ae35b64f56a7d07f3080684a.jpg)  
(c) h = 81% (Cora)

MLP baseline  
![](images/8598aae73f991a24ac5345fd4c87793306da9947b69f3a82f8f1b8e1a14109c6.jpg)  
(d) h = 75%

![](images/ebca6d551274326fc16bcb4b88a97a4659e8151e38fecf91b9db968ff4a19f2c.jpg)  
(e) h = 70%

![](images/f12d57c0d74921f0419eaf2c5461a3cdb92c1255a7c7908336d14f6ca8a99eb3.jpg)  
(f) h = 60%

![](images/8a0757dd2839cbfcb2ed98f1b6a921bb6bc05b7febf0c214fb23d69bdfd62df1.jpg)  
(g) h = 50%

![](images/b5780cbab78df6e18084fe309bccde4f4d49d9fae39fb790cd7827f596dbaa17.jpg)  
(h) h = 40%

![](images/1f70104da2251849305d3c7921f6f430317c1eaa8cef79a623eda1a4a96b9152.jpg)  
(i) h = 30%

![](images/8d00cf5191ad6da7eacadb4c9cdecef1034a17179b1dfc6a4ad6392c944856c0.jpg)  
(j) h = 20%

![](images/9e4ff2e7ffdf320839f60fa9e57952840bc64e1d8d472b9263c9eef29f911b68.jpg)  
(k) h = 10%

![](images/21bb091fc82fa6baf584b11119fa90430ae7ae22a9f3fcab20951d6f774e7679.jpg)  
(l) h = 0%

Figure 20. Our structural certificate on CSBM graphs with diff level of homophily. h = percentage of edges that connect same class nodes. CSBM: 774 nodes (same number of nodes per class as Cora), other parametrization following Gosch et al. (2023) with $K = 1 . 0 ,$ and $\sigma = 1 .$ . For certification we use k = 100 partitions and ε = 1. 60% nodes are used for training and 20% for validation and testing respectively.