# MDTE: Minority-Aware Difusion over Temporal Edge Events for Imbalanced Node Classification

Zelong Zhou Zhejiang University of Technology Hangzhou, China 221125120233@zjut.edu.cn

Yifu Tang   
Vecton AI   
Australia   
yves.tang@vectonai.com

Tianming Zhang<sup>∗</sup> Zhejiang University of Technology Hangzhou, China tmzhang@zjut.edu.cn

Chenyu Hou Zhejiang University of Technology Hangzhou, China houcy@zjut.edu.cn

Zhengyi Yang The University of Sydney Sydney, Australia zhengyi.yang@sydney.edu.au

Bin Cao Zhejiang University of Technology Hangzhou, China bincao@zjut.edu.cn

Jing Fan Zhejiang University of Technology Hangzhou, China fanjing@zjut.edu.cn

## Abstract

Class-imbalanced node classification on temporal graphs is challenging because majority-dominated temporal propagation progressively assimilates minority representations, while conventional node and neighborhood information provides insuficient discrimi native evidence for minority classes. To address these issues, we propose MDTE, a minority-aware difusion framework that reconstructs stable and discriminative temporal edge-event representations through conditional difusion denoising. Specifically, MDTE introduces Distribution-Aware Selective Propagation, which combines Local Outlier Factor (LOF)-based propagation filtering with cluster-aware low-frequency propagation. The module preserves informative neighborhood dependencies while mitigating harmful propagation and majority-class information assimilation. It further develops Multi-View Discriminative Fusion, which exploits feature reconstruction and topology prediction to characterize class-wise diferences in distribution learning and extracts complementary discriminability signals to guide denoising. Experiments on five real-world datasets demonstrate that MDTE consistently achieves the best performance on minority-class-oriented metrics, improving minority-class recall by up to 23.53 percentage points, minority class F1 by 8.68 percentage points, and AUPRC by 2.67 percentage points over the strongest baselines.

## Keywords

temporal graph learning, class-imbalanced node classification, graph difusion models

## 1 Introduction

Complex interaction networks, including financial transaction, blockchain transfer, and social networks, continuously evolve and can be naturally represented as temporal graphs of timestamped interactions [24, 36, 38]. Many real-world temporal graphs also exhibit class imbalance, where minority-class nodes, such as fraudulent accounts, illicit transaction nodes, and malicious users, are scarce but often of greater practical interest. Under temporal message propagation, their representations can be distorted by deviating neighboring representations and progressively assimilated toward majority-class patterns as majority information repeatedly dominates historical aggregation. Meanwhile, minority-specific patterns are underrepresented in the overall interaction distribution, providing insuficient discriminative evidence for separating minority and majority classes. Therefore, learning temporally stable and class-discriminative minority representations remains a challenge in class-imbalanced temporal graph learning.

In class-imbalanced temporal graphs, continuous interactions can progressively destabilize minority-class node representations, exacerbating the representational bias caused by class imbalance. Most existing class-imbalanced graph learning methods are designed for static graphs and mainly alleviate classification bias through data-level and algorithm-level strategies [19]. In these methods, indiscriminate neighborhood aggregation may propagate representations that deviate from the local neighborhood distribution and introduce majority-class information into minority representations. We refer to such propagation as harmful propagation. In temporal graphs, evolving interactions and continuous neighborhood propagation cause the efects of harmful propagation to accumulate and amplify over time, driving minority representations toward majority-class patterns. We refer to this process as majority-class information assimilation.

Difusion models can recover informative representations from perturbed inputs through denoising. Applying this denoising mechanism to graphs helps obtain more stable node representations. Existing graph difusion methods have been explored for node representation learning [5, 37, 45], and recent studies have further extended difusion to temporal graphs [10, 28]. These methods are generally optimized according to the overall graph distribution. Under class imbalance, majority-class patterns are more prevalent in the training distribution and can dominate the denoising objective, leaving insuficient signals for learning minority-specific patterns. Consequently, directly applying existing temporal difusion models may recover globally plausible representations while failing to preserve minority-specific characteristics and sharpen inter-class boundaries.

Efectively handling class-imbalanced node classification on temporal graphs requires simultaneously preserving the stability of minority-class representations and enhancing their class discriminability. This gives rise to two key technical challenges:

Challenge I: How can temporal difusion mitigate majorityclass information assimilation while preserving informative neighborhood dependencies? In temporal graphs, repeated neighborhood interactions and message propagation cause the efects of harmful propagation to accumulate and amplify, gradually shifting minority representations toward majority-class patterns and thereby resulting in majority-class information assimilation. This process destabilizes minority representations and weakens minority-specific patterns. However, simply reducing the overall propagation strength may also discard informative neighborhood dependencies. Therefore, the key challenge is to mitigate majority-class information assimilation while preserving useful neighborhood information, thereby maintaining the stability of minority representations.

Challenge II: How can latent class-discriminative evidence be extracted to guide minority-aware denoising? The scarcity of minority interactions limits the discriminative information available for learning minority-specific distributions. Node representations alone may not suficiently expose subtle class diferences, while topology alone may overlook feature-level distinctions. A more efective difusion process therefore requires complementary evidence from multiple views that can characterize how well the learned representations capture class-specific distributions. The key challenge is to extract and integrate complementary discriminability signals and incorporate them into the denoising process, so that difusion reconstruction is guided toward representations with clearer minority-majority separation.

To address these challenges, we propose MDTE, a Minorityaware Difusion framework over Temporal Edge events for imbalanced node classification. MDTE represents timestamped interactions as temporal edge events and learns their distributions through forward difusion and conditional reverse denoising. To address Challenge I, we propose Distribution-Aware Selective Propagation, which provides propagation-aware denoising conditions through two complementary mechanisms: Local Outlier Factor (LOF)-based propagation filtering restricts the transmission of deviating representations, while cluster-aware low-frequency propagation suppresses cross-cluster information to reduce majorityclass information assimilation without discarding useful neighborhood dependencies. To address Challenge II, we propose Multi-View Discriminative Fusion, which performs feature reconstruction and topology prediction on original and low-frequency-propagated representations. By characterizing class-wise diferences in distribution learning from feature and topology views, it extracts complementary discriminability signals and fuses them to guide the recon struction of stable and class-discriminative temporal edge-event representations. Finally, the denoised edge-event representations are aggregated to the node level, followed by Debiased Contrastive Learning for downstream node classification.

In summary, our main contributions are as follows:

• We propose MDTE, a minority-aware difusion framework over temporal edge events that explicitly addresses the two fundamental issues of majority-class information assimilation and insuficient class discriminability in class-imbalanced temporal graphs. By performing condition-guided difusion denoising on temporal edge events, MDTE learns representations that are both temporally stable and class-discriminative.

• We develop two complementary mechanisms for constructing denoising conditions during reverse difusion: Distribution-Aware Selective Propagation and Multi-View Discriminative Fusion. The former combines LOF-based propagation filtering and clusteraware low-frequency propagation to suppress deviating representations and majority-class information assimilation while retaining useful neighborhood dependencies; the latter jointly exploits feature reconstruction and topology prediction to extract complementary discriminability signals and guide the reconstruction toward more class-discriminative temporal representations.

• Extensive experiments on five real-world datasets demonstrate that MDTE consistently outperforms thirteen baseline methods, achieving particularly substantial improvements in minorityclass recognition, with minority-class recall improved by up to 23.53 percentage points over the strongest baselines.

## 2 Related Work

## 2.1 Class-Imbalanced Learning Methods

Class imbalance is common in real-world classification tasks and is characterized by unequal sample sizes across classes. In graph node classification, minority-class nodes are scarce and sparsely labeled, while their representations can be dominated by information from majority-class neighbors, further exacerbating classification bias. Existing methods can be divided into data-level and algorithm-level approaches [19].

Data-level methods include data interpolation [17, 21, 35, 40], adversarial generation [9, 22], and pseudo-labeling [43, 44]. Among data-interpolation methods, GraphSMOTE [40] generates minorityclass nodes in the embedding space and predicts their structural connections, while GraphENS [21] synthesizes complete ego-networks centered on minority-class nodes. Among adversarial-generation methods, ImGAGN [22] generates minority-class nodes and their connections through adversarial learning, while SORAG [9] extends this approach to imbalanced multi-label graphs. Among pseudolabeling methods, SPARC [43] combines label propagation with self-paced learning to select minority-class pseudo-labels, while GraphSR [44] uses node similarity and reinforcement learning to improve pseudo-label reliability.

Algorithm-level methods include model refinement [25, 30, 31], loss-function design [4, 26], and long-tailed representation enhancement [39]. Among model-refinement methods, RSDNE [31] constrains node embeddings using intra-class similarity and inter-class dissimilarity, EGCN [30] adjusts cross-class neighborhood aggregation from local and global perspectives, and DR-GCN [25] aligns the latent representation distributions of labeled and unlabeled nodes. Among loss-function methods, ReNode [4] adjusts training weights according to node positions relative to topological class boundaries, while TAM [26] adjusts classification margins according to local neighborhood label distributions. For long-tailed representation enhancement, LTE4G [39] combines expert models, knowledge distillation, and class prototypes to address class-level and node-degree-level long tails.

However, most existing methods are designed for static graphs and therefore do not explicitly capture interaction order or the ac cumulation of historical information. They also do not adequately address propagation interference from deviating representations or the repeated influence of majority-class information during temporal aggregation, thereby progressively destabilizing minorityclass representations. Moreover, the discriminative information extracted by these methods remains insuficient to further distinguish minority-class nodes from majority-class nodes.

## 2.2 Graph Denoising Difusion Models

Graph denoising difusion models learn graph representations or graph data distributions over node features, graph structures, or latent representations through forward perturbation and reverse denoising. Existing static-graph methods mainly include representation-space difusion [5, 37, 45], latent-space difusion [8, 42], and structure-oriented difusion [29, 34].

In representation space, DDM [37] introduces data-dependent directional noise to learn semantically informative graph representations, Grafe [5] uses encoder-produced graph representations to guide node-feature recovery, and SDMG [45] combines lowfrequency encoding with multi-scale smoothing constraints. In latent space, LGD [42] performs difusion over latent graph representations to unify graph generation and prediction. DifGCL [8] difuses low-dimensional latent embeddings produced by a shared graph encoder and combines difusion with graph contrastive learning. For graph structures, DiGress [29] difuses discrete node and edge types, while DDGAE [34] reconstructs graph structures through a discrete difusion decoder.

Recently, a small number of studies have extended difusion mechanisms to continuous-time dynamic graphs [10, 28]: Conda [28] generates historical neighbor embeddings through latent conditional difusion for dynamic graph augmentation, while SDG [10] reconstructs historical interaction sequences through conditional denoising for temporal link prediction.

However, existing difusion methods for continuous-time dynamic graphs do not explicitly address class imbalance and lack targeted modeling of minority-class node representations. Majorityclass interactions tend to dominate difusion training, biasing the model toward recovering the majority-class data distribution. Meanwhile, the limited occurrence of minority-class patterns makes them dificult to learn adequately. Denoising and propagation can further weaken the distinctive characteristics of minority-class representations, thereby blurring inter-class boundaries and degrading minority-class classification performance.

## 3 Preliminaries

Definition 3.1 (Continuous-time Temporal Subgraph). Given a batch of temporal edge events observed up to time $T ,$ the induced temporal subgraph is denoted by $\boldsymbol { \mathcal { G } } ^ { T } = \dot { ( } \mathcal { V } ^ { T } , \mathcal { E } ^ { T } , \mathbf { A } ^ { T } , \mathbf { H } )$ , where $\mathcal { V } ^ { T } = \{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ is the set of nodes involved in the batch up to time �, and $N = | \mathcal { V } ^ { T } |$ . The temporal edge-event set is $\mathcal { E } ^ { T } = \{ e _ { i j } ^ { \hat { t } } =$ $( v _ { i } , v _ { j } , t ) \mid v _ { i } , v _ { j } \in \mathcal { V } ^ { T } , t \leq T \}$ , where $e _ { i j } ^ { t }$ denotes an interaction between $v _ { i }$ and $v _ { j }$ at time $t ;$ the same node pair can interact multiple times. $\mathbf { A } ^ { T }$ is the adjacency matrix constructed from the interactions in $\mathcal { E } ^ { T }$ , and $\mathbf { H } = [ \bar { \mathbf { h } } _ { 1 } , . . . , \bar { \mathbf { h } } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d }$ is the node-representation matrix, where h<sub>�</sub> is the �-dimensional representation of $v _ { i } .$

Definition 3.2 (Continuous-time Encoding). For an edge event occurring at time �, let $\Delta t = T - t$ denote the interval between its occurrence time and the observation cutof time. The continuoustime encoding is generated using multi-scale sine and cosine basis functions [36]:

$$
\Phi _ { \mathrm { t i m e } } ( \Delta t ) = \sqrt { \frac { 1 } { 2 d _ { T } } } \big [ \cos ( \omega _ { 1 } \Delta t ) , \sin ( \omega _ { 1 } \Delta t ) , . . . ,
$$

Here, $d _ { T }$ is the number of frequencies, and $\pmb { \omega } = ( \omega _ { 1 } , \ldots , \omega _ { d _ { T } } ) ^ { \top }$ contains learnable frequency parameters. This encoding represents the temporal position of an edge event relative to the observation cutof time at multiple temporal scales.

Definition 3.3 (Random-Walk Positional Encoding). Given a graph with adjacency matrix A, the random-walk transition matrix is defined as $\mathbf { P } = \mathbf { D } ^ { - 1 } \mathbf { A }$ , where D is the degree matrix. For node $v _ { i } ,$ , the random-walk positional encoding is defined as $\mathbf { p } _ { i } = [ ( \mathbf { P } ) _ { i i } \lVert ( \mathbf { P } ^ { 2 } ) _ { i i } \rVert \cdot \cdot \cdot \lVert ( \mathbf { P } ^ { K _ { \mathrm { r w } } } ) _ { i i } ]$ , where $( \mathbf { P } ^ { s } ) _ { i i }$ denotes the probability that a random walk starting from $v _ { i }$ returns to itself after � steps, and $K _ { \mathrm { r w } }$ is the maximum walk length. These return probabilities characterize the structural position of a node across diferent neighborhood ranges [23].

Definition 3.4 (Imbalanced Temporal Graph Node Classification). Given the continuous-time temporal subgraph $\mathcal { G } ^ { T }$ and a set of labeled nodes $\mathcal { V } _ { L }$ with an imbalanced class distribution, partitioned into training, validation, and test sets, the objective is to learn a node-classification function under class imbalance, $f _ { \boldsymbol { \Theta } } : ( v _ { i } , \mathcal { E } ^ { T } , \mathbf { A } ^ { T } , \mathbf { H } ) \mapsto \widehat { y } _ { i } ^ { T }$ . Here, Θ denotes the learnable parameters, and $\widehat { y } _ { i } ^ { T }$ is the predicted label of node $v _ { i }$ at time �. The function predicts node classes from node representations and historical interactions up to time $T ,$ with particular emphasis on accurately identifying minority-class nodes.

## 4 Methodology

To address propagation interference from deviating representations, majority-class information assimilation, and insuficient discriminative information, we propose MDTE, a minority-aware difusion framework over temporal edge events for imbalanced node classification. As illustrated in Figure 1, MDTE consists of four stages: Temporal Edge-Event Construction, Minority-Aware Temporal Edge-Event Difusion, Edge-to-Node Aggregation, and Debiased Contrastive Learning. The second stage proceeds through Forward Difusion, Denoising Condition Construction, and Conditional Reverse Process. The denoising condition is constructed by two key modules: Distribution-Aware Selective Propagation and Multi-View Discriminative Fusion.

## 4.1 Temporal Edge-Event Construction

Existing methods for class-imbalanced graph learning and graph difusion typically aggregate historical interactions into a fixed graph structure, making it dificult to preserve the temporal information of individual interactions and the endpoint states before they occur. Therefore, MDTE represents each timestamped node interaction as a temporal edge event.

![](images/dc8624076d4b4f1f96d8a953f36ca2b1848c2bc720dc9a3e367659c9ba31863d.jpg)  
Figure 1: Overall architecture of MDTE.

For a temporal edge event $e _ { i j } ^ { t } = ( v _ { i } , v _ { j } , t )$ , its representation is

$$
\begin{array} { r } { \mathbf { E } _ { i j } ^ { t } = \left[ \mathbf { h } _ { i } \lVert \mathbf { h } _ { j } \rVert \Phi _ { \mathrm { t i m e } } ( \Delta t ) \rVert \mathbf { e } _ { i j } ^ { \mathrm { h i s t } } ( t ) \right] , } \end{array}\tag{1}
$$

where h<sub>�</sub> and ${ \bf h } _ { j }$ are the representations ofthe interacting endpoints, $\Phi _ { \mathrm { t i m e } } ( \Delta t )$ is the continuous-time encoding, ${ \bf e } _ { i j } ^ { \mathrm { h i s t } } ( t )$ encodes the endpoints’ historical interactions before the event, and ∥ denotes vector concatenation.

Let $d _ { i } ( t ^ { - } )$ and $d _ { j } ( t ^ { - } )$ be the numbers of historical interactions involving nodes $v _ { i }$ and $v _ { j { \mathrm { : } } }$ , respectively, before time �. The historicalinteraction encoding is

$$
\mathbf { e } _ { i j } ^ { \mathrm { h i s t } } ( t ) = \mathrm { M L P } \left( \log \left( 1 + d _ { i } ( t ^ { - } ) + d _ { j } ( t ^ { - } ) \right) \right) .\tag{2}
$$

The logarithmic transformation reduces the excessive influence of large interaction counts, and the resulting $\mathbf { E } _ { i j } ^ { t }$ serves as the difusion object in the subsequent edge-event difusion process.

## 4.2 Minority-Aware Temporal Edge-Event Difusion

To learn stable edge-event representations while reducing majorityclass bias, MDTE adopts conditional difusion with controlled noise and minority-oriented denoising conditions.

4.2.1 Forward Difusion. To enable the conditional reverse process to learn the distribution of temporal edge-event representations, we add controlled noise to $\mathbf { E } _ { i j } ^ { t }$ . During the forward process [12], we randomly sample a difusion step $q \in \{ 1 , \ldots , Q \}$ , where $Q$ denotes the total number of difusion steps, and add directional noise $\epsilon _ { i j }$ constructed from the distributional characteristics of the edge-event

representation [37] to obtain

$$
{ \bf E } _ { i j } ^ { t , q } = \sqrt { \bar { \alpha } _ { q } } { \bf E } _ { i j } ^ { t } + \sqrt { 1 - \bar { \alpha } _ { q } } \epsilon _ { i j } ,\tag{3}
$$

$$
\bar { \alpha } _ { q } = \prod _ { r = 1 } ^ { q } \alpha _ { r } .\tag{4}
$$

Here, $\alpha _ { r } \in ( 0 , 1 )$ denotes the information-retention coeficient at difusion step �, and $\bar { \alpha } _ { q }$ represents the cumulative proportion of original edge-event information retained after $q$ difusion steps. Thus, $\sqrt { \bar { \alpha } _ { q } }$ controls the proportion of the original representation, whereas $\sqrt { 1 - \bar { \alpha } _ { q } }$ controls the proportion of noise.

4.2.2 Denoising Condition Construction. To mitigate propagation interference and majority-class information assimilation while extracting additional discriminative information, we propose Distribution-Aware Selective Propagation and Multi-View EDiscriminative Fusion to construct the denoising conditions.

Distribution-Aware Selective Propagation. Some node representations deviate from their neighborhood distribution and interfere with other nodes during propagation. Therefore, we propose Distribution-Aware Selective Propagation, which uses LOF to limit their propagation to other nodes while preserving their own information through self-loops [3].

We compute LOF over the nodes in $\mathcal { G } ^ { T }$ to quantify how strongly each node representation deviates from its neighborhood distribution. Given the �-nearest-neighbor set $N _ { k } ( i )$ of node $v _ { i } ,$ its reachability distance to neighbor �<sub>�</sub> is

$$
\begin{array} { r } { \mathrm { r d } _ { k } ( i , j ) = \operatorname* { m a x } \left( k { - } \mathrm { d i s t } ( j ) , \mathrm { d i s t } ( i , j ) \right) , } \end{array}\tag{5}
$$

where �-dist(�) is the distance from $v _ { j }$ to its �-th nearest neighbor, and dist(�, �) is the cosine distance between $v _ { i }$ and $v _ { j }$ in the representation space. The reachability distance uses the local neighborhood radius �-dist(�) of $v _ { j }$ as a lower bound, preventing overly small pairwise distances in locally dense regions from exerting excessive influence on density estimation and thereby improving its stability.

Based on the reachability distance, the local reachability density of $v _ { i }$ is

$$
\mathrm { l r d } _ { k } ( i ) = \left( \frac { \sum _ { v _ { j } \in N _ { k } ( i ) } \mathrm { r d } _ { k } ( i , j ) } { | N _ { k } ( i ) | } \right) ^ { - 1 } .\tag{6}
$$

The local reachability density is the inverse of the average reachability distance between a node and its neighbors and measures local compactness; a larger lrd<sub>�</sub> (�) indicates a denser local region.

The local outlier factor of �<sub>�</sub> is

$$
\mathrm { L O F } _ { k } ( i ) = \frac { 1 } { | N _ { k } ( i ) | } \sum _ { v _ { j } \in N _ { k } ( i ) } \frac { \mathrm { l r d } _ { k } ( j ) } { \mathrm { l r d } _ { k } ( i ) } .\tag{7}
$$

LOF measures how strongly a node representation deviates from its neighborhood by comparing the node’s local reachability density with those of its neighbors. When the local reachability density of�<sub>�</sub> is substantially lower than those of its neighbors, $\mathrm { L O F } _ { k } ( i )$ increases, indicating a stronger deviation from the dominant representation pattern of its neighborhood.

We then rank-normalize the logarithmic values of $\mathrm { L O F } _ { k } ( i )$ to obtain relative deviation scores $u _ { i } \in [ 0 , 1 ]$ . Given a propagationfiltering threshold $\tau ,$ we construct a node mask m $\in \{ \bar { 0 } , 1 \} ^ { \bar { N } }$ , whose �-th entry is set to 1 if $u _ { i } < \tau$ and 0 otherwise. The filtered adjacency matrix is

$$
\widetilde { \mathbf { A } } ^ { T } = \mathbf { A } ^ { T } \odot ( \mathbf { m m } ^ { \top } ) + \mathbf { I } .\tag{8}
$$

Here, I is the identity matrix that retains a self-loop for every node, and ⊙ denotes element-wise multiplication.

Cross-cluster propagation mixes dissimilar representation patterns and amplifies majority-dominated information. Therefore, we propose cluster-aware low-frequency propagation to preserve intra-cluster propagation while downweighting cross-cluster messages.

We apply K-means [18] to the nodes retained after propagation filtering and use the elbow method [27] to adaptively determine the number of clusters $K _ { \mathrm { c } } .$ . The elbow method selects the point at which the decrease in within-cluster inertia changes from rapid to gradual, where the inertia is the mean squared Euclidean distance from each node to its assigned cluster center and measures cluster compactness. Let $c _ { i } \in \{ 1 , . . . , K _ { \mathrm { c } } \}$ denote the resulting cluster label of node �<sub>�</sub>. The propagation weight is

$$
B _ { i j } = \left\{ \begin{array} { l l } { 1 , \quad c _ { i } = c _ { j } , } \\ { \alpha , \quad c _ { i } \neq c _ { j } , } \end{array} \right. \quad \quad 0 < \alpha < 1 ,\tag{9}
$$

where � is a learnable cross-cluster propagation weight. The weighted adjacency matrix for low-frequency propagation is

$$
\overline { { A } } _ { i j } ^ { T } = \widetilde { A } _ { i j } ^ { T } B _ { i j } .\tag{10}
$$

Following the Node Feature Low-Frequency Information Encoder in SDMG [45], we extend it to the weighted filtered graph, where the cluster-aware edge weights modulate the GAT attention coeficients. The resulting low-frequency node representations are

$$
{ \bf H } ^ { \mathrm { l o w } } = \mathrm { L P } ( \overline { { { \bf A } } } ^ { T } , { \bf H } ) .\tag{11}
$$

Here, LP(·) denotes the GAT-based low-frequency encoder, ${ \bf H } ^ { \mathrm { l o w } } =$ $[ \mathbf { h } _ { 1 } ^ { \mathrm { l o w } } , \ldots , \mathbf { h } _ { N } ^ { \mathrm { l o w } } ] ^ { \intercal }$ <sup>⊤</sup>, and $\mathbf { h } _ { i } ^ { \mathrm { l o w } }$ is the low-frequency representation of node �<sub>�</sub>.

For a minority-class node, same-cluster neighbors usually exhibit similar representation patterns, and preserving within-cluster propagation captures smooth neighborhood information. In contrast, cross-cluster neighbors may introduce inconsistent information due to larger representation discrepancies. Therefore, downweighting cross-cluster propagation preserves useful information from similar neighbors while limiting dissimilar majority-class information, thereby alleviating majority-class information assimilation.

Multi-View Discriminative Fusion. To further distinguish minorityclass nodes from majority-class nodes, we propose Multi-View Discriminative Fusion, which extracts and fuses feature-based and topology-based discriminability signals from the original and low-frequency node representations [33].

The attribute patterns and connection relations of majority-class nodes are observed more frequently during training, making their distribution easier for the model to learn. In contrast, minorityclass nodes and their associated interactions are limited, so their distinctive patterns can remain insuficiently learned. Based on this diference in how well the underlying distributions are learned, we extract discriminability signals through feature reconstruction and topology prediction to capture additional diferences between minority-class nodes and majority-class nodes.

Feature Discriminability Signal. Directly reconstructing the complete node representation yields only an overall reconstruction error, whereas dimension-wise Normal-Inverse-Gamma (NIG) modeling estimates the distributional uncertainty of individual dimensions, providing finer-grained discriminative information for characterizing feature diferences between minority-class and majority-class nodes. Specifically, we model each representation dimension as a Gaussian variable with unknown mean and variance and use an NIG distribution [1] to describe parameter uncertainty. Given the original node representation $\mathbf { h } _ { i } ,$ the feature signal head outputs four groups of NIG parameters:

$$
\begin{array} { r } { ( \gamma _ { i } , \kappa _ { i } , \alpha _ { i } , \beta _ { i } ) = \mathrm { M L P } ( \mathbf h _ { i } ) . } \end{array}\tag{12}
$$

The four output vectors contain a set of NIG parameters for each dimension of the node representation. For dimension $\ell , \gamma _ { i , \ell }$ is the reconstructed value of $h _ { i , \ell } ; \kappa _ { i , \ell }$ reflects the certainty of the reconstructed value, with a larger value indicating higher certainty; and $\alpha _ { i , \ell }$ and $\beta _ { i , \ell }$ control the shape and scale of the variance distribution, respectively. Together, $\kappa _ { i , \ell } , \alpha _ { i , \ell } ,$ , and $\beta _ { i , \ell }$ characterize the reconstruction uncertainty of dimension $\ell .$ For each dimension, $\kappa _ { i , \ell } ~ > ~ 0 _ { : }$ $\alpha _ { i , \ell } > 1 _ { : }$ , and $\beta _ { i , \ell } > 0$

The feature discriminability signal of node $v _ { i }$ is derived from the reconstruction uncertainty:

$$
s _ { i } = \mathrm { M e a n } \left( \sqrt { \frac { \beta _ { i , \ell } } { \kappa _ { i , \ell } ( \alpha _ { i , \ell } - 1 ) } } \right) .\tag{13}
$$

The quantity inside the square root is the reconstruction uncertainty in dimension $\ell ,$ and Mean(·) averages over all representation dimensions. A larger $s _ { i }$ indicates higher uncertainty about reconstructing the node’s feature pattern. Under class imbalance, minority-class feature patterns occur less frequently and are more dificult to reconstruct stably; therefore, a node with a larger �<sub>�</sub> is more likely to exhibit a minority-class feature pattern.

To obtain reliable reconstruction uncertainty, we use $\mathcal { L } _ { \mathrm { N L I } }$ to fit the predictive distribution to the original node representation and $\mathcal { L } _ { \mathrm { r e g } }$ to prevent overconfidence when the reconstruction error is large. The NIG objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N I G } } = \mathcal { L } _ { \mathrm { N L L } } + \lambda _ { e } \mathcal { L } _ { \mathrm { r e g } } , } \end{array}\tag{14}
$$

where $\lambda _ { e }$ controls the contribution of the regularization term.

Following evidential regression [1], the negative log-likelihood for node $v _ { i }$ is

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { N L } , i } = \mathrm { M e a n } [ \frac { 1 } { 2 } \log \frac { \pi } { \kappa _ { i , \ell } } - \alpha _ { i , \ell } \log ( 2 \beta _ { i , \ell } ( 1 + \kappa _ { i , \ell } ) )  } } \\ & { } & {  + ( \alpha _ { i , \ell } + \frac { 1 } { 2 } ) \log ( \kappa _ { i , \ell } ( h _ { i , \ell } - \gamma _ { i , \ell } ) ^ { 2 } + 2 \beta _ { i , \ell } ( 1 + \kappa _ { i , \ell } ) )  } \\ & { } & {  + \log \Gamma ( \alpha _ { i , \ell } ) - \log \Gamma ( \alpha _ { i , \ell } + \frac { 1 } { 2 } ) \ ] . } \end{array}
$$

The squared reconstruction deviation $( h _ { i , \ell } - \gamma _ { i , \ell } ) ^ { 2 }$ appears in the logarithmic residual term. A larger deviation increases the loss and encourages $\gamma _ { i , \ell }$ to approach $h _ { i , \ell } .$ . Meanwhile, $\kappa _ { i , \ell } , \alpha _ { i , \ell } .$ , and $\beta _ { i , \ell }$ adjust the penalty according to the estimated uncertainty: for the same reconstruction deviation, lower uncertainty results in a stronger penalty. Consequently, minimizing $\mathcal { L } _ { \mathrm { N L L } , i }$ jointly promotes accurate reconstruction and learns uncertainty consistent with the reconstruction deviation.

For node $v _ { i }$ , the NIG regularization term is

$$
\mathcal { L } _ { \mathrm { r e g } , i } = \mathrm { M e a n } \left[ | h _ { i , \ell } - \gamma _ { i , \ell } | \left( 2 \kappa _ { i , \ell } + \alpha _ { i , \ell } \right) \right] ,\tag{16}
$$

A larger reconstruction deviation $\lvert h _ { i , \ell } - \gamma _ { i , \ell } \rvert$ , together with a larger value of $2 \kappa _ { i , \ell } + \alpha _ { i , \ell } ;$ results in a stronger penalty, thereby encouraging higher uncertainty for inaccurately reconstructed feature patterns. Both $\mathcal { L } _ { \mathrm { N L I } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ are obtained by averaging their node-level terms over the current batch.

Topology Discriminability Signal. Feature-reconstruction uncertainty reflects how well node attributes are learned but does not capture discriminative information in connection patterns. Therefore, from the topology view, a topology signal head predicts whether a connection exists between a pair of nodes. For a node pair $( v _ { i } , v _ { j } )$ the connection probability is

$$
\epsilon _ { i j } ^ { + } = \mathrm { M L P } ( [ \mathbf { h } _ { i } | | \mathbf { h } _ { j } ] ) .\tag{17}
$$

The output layer of the MLP uses a sigmoid activation to map its output to $( 0 , 1 )$ , and the probability that no connection exists is $\epsilon _ { i j } ^ { - } = 1 - \epsilon _ { i j } ^ { + }$

Topology prediction uses observed connections as positive samples and randomly sampled unconnected node pairs as negative samples. For a node pair $( v _ { i } , v _ { j } )$ , the binary cross-entropy loss is

$$
\mathcal { L } _ { i j } ^ { \mathrm { B C E } } = - \left[ A _ { i j } ^ { T } \log \epsilon _ { i j } ^ { + } + \left( 1 - A _ { i j } ^ { T } \right) \log \epsilon _ { i j } ^ { - } \right] ,\tag{18}
$$

where $A _ { i j } ^ { T } = 1$ for an observed connection and $A _ { i j } ^ { T } = 0$ for a sampled unconnected pair. The uncertainty of the connection prediction is measured using predictive entropy:

$$
\eta _ { i j } ^ { \mathrm { t o p o } } = - \epsilon _ { i j } ^ { + } \log \epsilon _ { i j } ^ { + } - \epsilon _ { i j } ^ { - } \log \epsilon _ { i j } ^ { - } .\tag{19}
$$

The topology discriminability signal combines connection-prediction error and prediction uncertainty:

$$
u _ { i j } ^ { \mathrm { t o p o } } = \frac { 1 } { 2 } \left( \mathcal { L } _ { i j } ^ { \mathrm { B C E } } + \eta _ { i j } ^ { \mathrm { t o p o } } \right) .\tag{20}
$$

A larger $\mathcal { L } _ { i j } ^ { \mathrm { B C E } }$ indicates greater prediction error, whereas a larger $\eta _ { i j } ^ { \mathrm { t o p o } }$ indicates that the connection is predicted with high uncertainty. Under class imbalance, node pairs with larger prediction errors or uncertainties are more likely to correspond to insuficiently learned minority-class connection patterns.

Let $\Omega _ { \mathrm { t o p o } }$ denote the positive and negative node pairs sampled for topology prediction in the current batch. For each node $v _ { i } ,$ the topology discriminability signals associated with �<sub>�</sub> are averaged to obtain the node-level topology discriminability signal

$$
s _ { i } ^ { \mathrm { t o p o } } = \operatorname * { M e a n } _ { ( v _ { i } , v _ { j } ) \in \Omega _ { \mathrm { t o p o } } } \left( u _ { i j } ^ { \mathrm { t o p o } } \right) .\tag{21}
$$

To extract more discriminative signals and further distinguish minority-class nodes from majority-class nodes, we further propose extracting feature-based and topology-based discriminability signals from the low-frequency representations and fusing them with those obtained from the original representations.

The original node representation $\mathbf { h } _ { i }$ mainly retains nodeattribute information, whereas $\mathbf { h } _ { i } ^ { \mathrm { l o w } }$ incorporates smoothed structural information from the filtered neighborhood. We apply the same feature-reconstruction and topology-prediction processes to both representation spaces using signal heads with the same structure but independent parameters.

To measure whether the propagated representation preserves node-specific patterns, we reconstruct $\mathbf { h } _ { i }$ from $\mathbf { h } _ { i } ^ { \mathrm { l o w } }$ and use the reconstruction uncertainty as the discriminability signal $s _ { i } ^ { \mathrm { l o w } }$ , with the corresponding loss $\mathcal { L } _ { \mathrm { N I G } } ^ { \mathrm { l o w } }$ . The joint feature objective is

$$
\mathcal { L } _ { \mathrm { f e a t } } = \mathcal { L } _ { \mathrm { N I G } } + \mathcal { L } _ { \mathrm { N I G } } ^ { \mathrm { l o w } } .\tag{22}
$$

In the low-frequency topology view, we use the low-frequency node representations to predict the same positive and negative node pairs. This yields the low-frequency topology discriminability signal $u _ { i j } ^ { \mathrm { t o p o , l o w } }$ and the corresponding node-level signal $s _ { i } ^ { \mathrm { t o p o , l o w } }$ The joint topology objective for the original and low-frequency views is

$$
\mathcal { L } _ { \mathrm { t o p o } } = \operatorname* { M e a n } _ { ( v _ { i } , v _ { j } ) \in \Omega _ { \mathrm { t o p o } } } \left( \mathcal { L } _ { i j } ^ { \mathrm { B C E } } + \mathcal { L } _ { i j } ^ { \mathrm { B C E , l o w } } \right) .\tag{23}
$$

To jointly learn the feature-based and topology-based discriminability signals from the original and low-frequency representation spaces, we combine the feature and topology objectives as

$$
\mathcal { L } _ { \mathrm { a u x } } = \lambda _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } + \lambda _ { \mathrm { t o p o } } \mathcal { L } _ { \mathrm { t o p o } } ,\tag{24}
$$

where $\lambda _ { \mathrm { f e a t } }$ and $\lambda _ { \mathrm { t o p o } }$ control the contributions of the feature and topology objectives, respectively.

To provide complementary discriminative information for further distinguishing minority-class nodes from majority-class nodes, we concatenate the feature-based and topology-based discriminabil ity signals from the original and low-frequency representation spaces:

$$
\begin{array} { r } { { \bf S } _ { i } = [ s _ { i } \| s _ { i } ^ { \mathrm { l o w } } \| s _ { i } ^ { \mathrm { t o p o } } \| s _ { i } ^ { \mathrm { t o p o , l o w } } ] . } \end{array}\tag{25}
$$

The resulting signal $\mathsf { S } _ { i }$ integrates feature-reconstruction uncertainty and topology-prediction error and uncertainty from the two representation spaces.

The mean of the four signals in $\mathbf { S } _ { i }$ is used as the node-level discriminability weight $w _ { i }$ . For each temporal edge event $e _ { i j } ^ { t } ,$ the corresponding edge-event weight $w _ { i j }$ is obtained by averaging the weights of its two endpoints.

Condition Integration. To construct an informative denoising condition that more efectively guides the reverse process, MDTE integrates the low-frequency node representation $\mathbf { h } _ { i } ^ { \mathrm { l o w } }$ and fused discriminability signal $\mathbf { S } _ { i } ,$ while incorporating the random-walk positional representation $\mathbf { p } _ { i }$ to complement structural-position information. To maintain consistency between the random-walk positional representations and the filtered propagation topology, we apply standard graph-Laplacian smoothness regularization [2], denoted by ${ \mathcal { L } } _ { \mathrm { p o s } }$

The resulting node-level denoising condition is

$$
\widetilde { \mathbf { h } } _ { i } = \mathrm { M L P } \left( [ \mathbf { h } _ { i } ^ { \mathrm { l o w } } + \mathbf { p } _ { i } \| \mathbf { S } _ { i } ] \right) .\tag{26}
$$

To further incorporate event-specific temporal and historical information, we then combine the node-level conditions of the two endpoints with the temporal and historical-interaction encodings to construct the edge-event-level denoising condition:

$$
\mathbf { h } _ { i j } ^ { \mathrm { c o n d } } = [ \mathbf { W } _ { s } \widetilde { \mathbf { h } } _ { i } \Vert \mathbf { W } _ { d } \widetilde { \mathbf { h } } _ { j } \Vert \mathbf { W } _ { t } \boldsymbol { \Phi } _ { \mathrm { t i m e } } ( \Delta t ) \Vert \mathbf { W } _ { g } \mathbf { e } _ { i j } ^ { \mathrm { h i s t } } ( t ) ] .\tag{27}
$$

Here, $\mathbf { W } _ { s } , \mathbf { W } _ { d } , \mathbf { W } _ { t }$ , and $\mathbf { W } _ { g }$ are the respective projection matrices.

4.2.3 Conditional Reverse Process. To learn the distribution of stable and discriminative edge-event representations, MDTE performs conditional reverse denoising guided by $\mathbf { h } _ { i j } ^ { \mathrm { c o n d } }$ . Given $\mathbf { E } _ { i j } ^ { t , q }$ , difusion step $q ,$ and $\mathbf { h } _ { i j } ^ { \mathrm { c o n d } }$ , a conditional U-Net reconstructs the edge-event representation:

$$
\widehat { \mathbf { E } } _ { i j } ^ { t } = \mathrm { U - N e t } ( \mathbf { E } _ { i j } ^ { t , q } , q , \mathbf { h } _ { i j } ^ { \mathrm { c o n d } } ) .\tag{28}
$$

To prevent frequent majority-class interactions from dominating difusion training, the discriminability weight $w _ { i j }$ is used to weight the squared reconstruction error of each edge event. The difusion loss is

$$
\mathcal { L } _ { \mathrm { d i f f } } = \frac { \sum _ { e _ { i j } ^ { t } \in \mathcal { B } _ { e } } w _ { i j } \left\| \widehat { \mathbf { E } } _ { i j } ^ { t } - \mathbf { E } _ { i j } ^ { t } \right\| _ { 2 } ^ { 2 } } { \sum _ { e _ { i j } ^ { t } \in \mathcal { B } _ { e } } w _ { i j } } .\tag{29}
$$

Here, $\mathcal { B } _ { e }$ denotes the edge events in the current batch. A larger $w _ { i j }$ increases the contribution of the corresponding edge event to difusion training, thereby reducing majority-dominated learning.

## 4.3 Edge-to-Node Aggregation and Debiased Contrastive Learning

To obtain node representations for downstream classification while preserving edge-level information, MDTE combines each denoised edge-event representation with its temporal encoding, historical interaction encoding, and source/destination role encoding after dimensional alignment, and aggregates the resulting representations by attention:

$$
\begin{array} { r } { \mathbf { h } _ { i } ^ { \mathrm { a g g } } = \mathrm { A t t n } _ { e _ { i j } ^ { t } \in \mathcal { R } ( i ) } \left( \widehat { \mathbf { E } } _ { i j } ^ { t } + \Phi _ { \mathrm { t i m e } } ( \Delta t ) + { \mathbf { e } } _ { i j } ^ { \mathrm { h i s t } } ( t ) + { \mathbf { r } } _ { i | i j } ^ { t } \right) . } \end{array}\tag{30}
$$

Table 1: Dataset statistics.
<table><tr><td>Dataset</td><td>Nodes</td><td>Temporal edges</td><td>Minority</td><td>Majority</td><td>IR</td></tr><tr><td>DGraph-Fin</td><td>3,700,550</td><td>4,300,999</td><td>15,509</td><td>1,210,092</td><td>78.03</td></tr><tr><td>Elliptic</td><td>203,769</td><td>234,355</td><td>4,545</td><td>42,019</td><td>9.25</td></tr><tr><td>Elliptic++ Transactions</td><td>203,769</td><td>234,355</td><td>4,545</td><td>42,019</td><td>9.25</td></tr><tr><td>Elliptic++ Actors</td><td>822,942</td><td>2,868,964</td><td>14,266</td><td>251,088</td><td>17.60</td></tr><tr><td>Ethereum</td><td>2,973,489</td><td>13,551,303</td><td>1,165</td><td>2,972,324</td><td>2,551.35</td></tr></table>

Here, $\mathcal { R } ( i )$ denotes the temporal edge events involving $v _ { i }$ in the current batch, and $\mathbf { r } _ { i \mid i j } ^ { t }$ is the role encoding that indicates whether $v _ { i }$ is the source or destination of $e _ { i j } ^ { t }$

To maintain consistency between the denoised aggregated representation $\mathbf { h } _ { i } ^ { \mathrm { a g g } }$ and the original node representation $\mathbf { h } _ { i } ,$ MDTE treats them as a positive pair and uses the original representations of other nodes in the current batch as negatives. Since these negatives may contain semantically similar false negatives, we adopt a simplified correction inspired by Debiased Contrastive Learning [7]. The corrected negative term is

$$
n _ { i } = \frac { \displaystyle \sum _ { j = 1 , j \neq i } ^ { N } \exp \left( \sin ( \mathbf { h } _ { i } ^ { \mathrm { a g g } } , \mathbf { h } _ { j } ) / \tau _ { c } \right) - \tau _ { + } \exp \left( \sin ( \mathbf { h } _ { i } ^ { \mathrm { a g g } } , \mathbf { h } _ { i } ) / \tau _ { c } \right) } { 1 - \tau _ { + } } .\tag{31}
$$

Here, sim $( \cdot , \cdot )$ denotes cosine similarity, $\tau _ { c }$ is the temperature coefficient, and $\tau _ { + }$ is the prior proportion of potential false negatives. The debiased contrastive loss is

$$
\mathcal { L } _ { \mathrm { c l } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp { \left( \sin ( \mathbf { h } _ { i } ^ { \mathrm { a g g } } , \mathbf { h } _ { i } ) / \tau _ { c } \right) } } { \exp { \left( \sin ( \mathbf { h } _ { i } ^ { \mathrm { a g g } } , \mathbf { h } _ { i } ) / \tau _ { c } \right) } + n _ { i } } .\tag{32}
$$

If a node appears in multiple edge-event batches, its denoised aggregated representations are averaged to obtain a unified node representation $\mathbf { h } _ { i } ^ { \mathrm { n o d e } }$ for downstream classification.

Combining the difusion, auxiliary, positional, and debiased contrastive objectives, the overall training objective of MDTE is

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { c l } } { \mathcal { L } } _ { \mathrm { c l } } + { \mathcal { L } } _ { \mathrm { a u x } } + \lambda _ { \mathrm { p o s } } { \mathcal { L } } _ { \mathrm { p o s } } .\tag{33}
$$

The coeficients $\lambda _ { \mathrm { c l } }$ and $\lambda _ { \mathrm { p o s } }$ control the debiased contrastive loss and positional-encoding constraint, respectively. This process does not use node class labels; after training, the parameters of MDTE are frozen, and a unified downstream classifier is trained using labeled training nodes.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We conduct experiments on five financial transaction graph datasets: DGraph-Fin [13], Elliptic [32], Elliptic++ Transactions and Actors [11], and Ethereum [6]. Table 1 reports their statistics, where the imbalance ratio (IR) is the ratio of majority- to minority-class nodes; a larger IR indicates more severe class imbalance. DGraph-Fin identifies fraudulent users, Elliptic and Elliptic++ Transactions detect illicit transactions, Elliptic++ Actors identifies illicit participants, and Ethereum detects phishing accounts.

Compared Baselines. We compare with thirteen baselines from four categories: (1) temporal graph methods, TGC [15], TGAT [36],

TGN [24], and DyGFormer [38]; (2) static graph autoencoder methods, GraphPAE [16] and DGMAE [41]; (3) imbalanced graph methods, GraphSMOTE [40], LTE4G [39], and GraphENS [21]; and (4) graph difusion methods, DDM [37], SDMG [45], LGD [42], and TGAT+Conda [28, 36].

Evaluation Protocol. All methods use the same data preprocessing, node splits, and downstream classifier. MDTE performs labelfree self-supervised learning on the complete temporal graph, processing all edge events chronologically. DGraph-Fin follows its oficial 70%/15%/15% split; labeled nodes in the other datasets are randomly divided into training, validation, and test sets at 60%/20%/20% using a fixed seed. After representation learning, each encoder is frozen and evaluated using the same two-layer MLP.

We report AUROC, AUPRC, minority-class Recall, minority-class F1, and Macro-F1, treating the minority class as positive.

AUROC denotes the area under the receiver operating characteristic (ROC) curve and measures the model’s ability to distinguish positive from negative samples across classification thresholds.

AUPRC summarizes the precision–recall trade-of across diferent classification thresholds and is computed as the area under the precision–recall curve. In its discrete form:

$$
{ \mathrm { A U P R C } } \approx \sum _ { c = 1 } ^ { C } ( R _ { c } - R _ { c - 1 } ) P _ { c } ,\tag{34}
$$

where $P _ { c }$ and $R _ { c }$ denote precision and recall at the �-th classification threshold, respectively. Minority-class Recall measures the proportion of minority-class nodes correctly identified, while minorityclass F1 is the harmonic mean of minority-class precision and recall. Macro-F1 is the arithmetic mean of the class-wise F1 scores. Higher values indicate better performance for all metrics.

For DGraph-Fin, we search [0.01, 0.50] at intervals of 0.01 and select the threshold maximizing validation minority-class F1; the other datasets use 0.5. All results report the mean and standard deviation over ten random seeds.

Implementation Details. Experiments run on an NVIDIA GeForce RTX 3090 GPU. For MDTE, we set the total number of difusion steps to � = 200 and adopt a linear noise schedule. The conditional U-Net contains three encoder blocks and three decoder blocks. The low-frequency encoder uses a two-layer GAT with two attention heads and a hidden dimension of 384. Baselines follow their oficial implementations, original training procedures, and recommended hyperparameters. The downstream classifier uses Adam [14] with a learning rate of 0.01, weight decay of $5 \times 1 0 ^ { - 7 }$ batch size 8,192, and 200 epochs. Hyperparameters are selected on the validation set.

Next, we seek to answer the following research questions: RQ1 (Classification Performance). How does MDTE compare with representative baselines for class-imbalanced node classification on temporal graphs?

RQ2 (Ablation: Necessity of Components). Are the key components of MDTE necessary for its classification performance?

RQ3 (Parameter Sensitivity). How sensitive is the performance of MDTE to changes in key hyperparameters?

RQ4 (Core Mechanism Analysis). How do Distribution-Aware Selective Propagation and Multi-View Discriminative Fusion afect minority representation assimilation and class discriminability?

## 5.2 Experimental Results

RQ1 (Classification Performance). As shown in Table 2, we observe that (i) compared with temporal graph methods (TGC, DyGFormer, TGAT, and TGN), MDTE achieves the best results on all non-AUROC metrics across the five datasets. On Ethereum, minority-class Recall and F1 improve by 14.94 and 8.68 percentage points, respectively, over the strongest temporal baselines. Its comparable AUROC further shows that explicitly accounting for class imbalance complements temporal modeling, particularly for minority-class recognition.

(ii) Compared with static graph autoencoder methods (GraphPAE and DGMAE), MDTE ranks first across all metrics on four datasets. On Ethereum, AUROC remains comparable, while minority-class Recall and F1 increase by 19.64 and 9.94 percentage points, respectively, reflecting the benefit of capturing temporal edge events and minority-related interaction patterns.

(iii) Compared with class-imbalanced graph methods (GraphENS, LTE4G, and GraphSMOTE), MDTE consistently achieves the best non-AUROC results. The advantage is especially clear on Ethereum, with gains of 26.60 and 17.68 percentage points in minority-class Recall and F1, respectively, while maintaining comparable AUROC. Such results reveal that class-imbalance handling can be further strengthened by incorporating temporal information.

(iv) For graph difusion methods (DDM, SDMG, LGD, and TGAT+Conda), MDTE achieves the best results across all metrics on four datasets and trails only slightly in AUROC on Elliptic. On Ethereum, minority-class Recall and F1 are improved by 29.78 and 23.43 percentage points, respectively, showing the benefit of incorporating minority-relevant information into the difusion condition.

Overall, MDTE consistently achieves stronger minority-class recognition across all four categories of baselines and maintains stable performance across datasets with varying imbalance levels, demonstrating its efectiveness and robustness.

RQ2 (Ablation: Necessity of Components). Table 3 shows that removing any core component degrades performance.

Removing Temporal Edge-Event Construction reduces minorityclass Recall by 14.19 percentage points on DGraph-Fin. Without Distribution-Aware Selective Propagation, minority-class Recall decreases by 4.66 percentage points on DGraph-Fin. Without Multi-View Discriminative Fusion, minority-class Recall decreases by 2.48 percentage points on Elliptic. Without Debiased Contrastive Learning, minority-class Recall drops by 5.47 percentage points on DGraph-Fin. These results confirm that Temporal Edge-Event Construction captures temporal interactions; Distribution-Aware Selective Propagation suppresses propagation interference and majorityclass assimilation while retaining useful dependencies; Multi-View Discriminative Fusion enhances discriminative evidence for minority representations; and Debiased Contrastive Learning maintains consistency between original node representations and denoised interaction-aggregated representations.

RQ3 (Parameter Sensitivity). Figure 2 evaluates four hyperparameters: the propagation-filtering threshold � for the node mask m in Eq. (8), the LOF neighborhood size � in Eq. (5), and the potential-false-negative prior $\tau _ { + }$ and contrastive temperature $\tau _ { c }$ in Eq. (31). Mean-centered F1-min is calculated by subtracting the mean minority-class F1 across all tested settings from the minorityclass F1 at each setting. The settings $\tau = 0 . 8 5 , \tau _ { + } = 0 . 0 2 5 , k = 3 0 .$

Table 2: Node classification results (%). The best and second-best results in each metric row are shown in bold and underlined, respectively.
<table><tr><td>Dataset</td><td>Metric</td><td>TGC</td><td>DyGFormer</td><td>TGAT</td><td>TGN</td><td>GraphPAE</td><td>DGMAE</td><td>GraphSMOTE</td><td>LTE4G</td><td>GraphENS</td><td>DDM</td><td>SDMG</td><td>LGD TGAT+Conda</td><td>MDTE</td></tr><tr><td rowspan="6">DGraph-Fin</td><td>AUROC AUPRC</td><td>72.89±.27 2.89±.09</td><td>76.10±.23 3.69±.11</td><td>72.06±.12 3.14±.02</td><td>73.41±.15 3.12±.04</td><td>70.92±.31 2.92±.05</td><td>70.61±.18</td><td>72.46±.08 71.90±.04</td><td>71.85±.06</td><td>72.34±.06 3.32±.06</td><td>74.59±.60 3.54±.13</td><td>75.48±.11 3.67±.04</td><td>74.38±.14 3.25±.06</td><td>81.19±0.23 4.76±0.07</td></tr><tr><td>Recall-min.</td><td>32.01±6.46</td><td></td><td>28.95±7.42</td><td></td><td>3.00±.05 21.02±11.21 18.30±4.44</td><td>2.73±.02 21.90±14.72</td><td>2.64±.01 20.68±11.92 17.33±4.00</td><td>3.07±.03</td><td>19.25±5.21</td><td>19.60±4.41</td><td></td><td></td><td>55.54±0.95</td></tr><tr><td></td><td>21.47±12.03</td><td></td><td>30.37±6.03</td><td></td><td>6.75±.20</td><td>5.50±.29</td><td>5.45±.30</td><td></td><td></td><td></td><td>25.16±7.80</td><td>24.73±5.59</td><td>8.24±0.10</td></tr><tr><td>F1-min.</td><td>5.64±.32 49.69±0.02</td><td>7.91±.51</td><td>6.82±.04</td><td>6.61±.06 49.68±0.00</td><td>5.90±.25 49.68±0.00</td><td></td><td></td><td>6.34±.20</td><td>6.95±.09</td><td>7.18±.23</td><td>7.47±.29 49.68±0.00</td><td>6.64±.11 49.68±0.00</td><td></td></tr><tr><td>Macro-F1</td><td></td><td>49.68±0.00</td><td>49.68±0.00</td><td></td><td>49.68±0.00</td><td>49.68±0.00</td><td>49.68±0.00</td><td>49.68±0.00</td><td>49.68±0.00</td><td>49.68±0.00</td><td></td><td></td><td>49.72±0.06</td></tr><tr><td>AUROC AUPRC</td><td>98.38±.13</td><td>92.74±.44</td><td>96.95±.13</td><td>98.47±.05</td><td>97.07±.14</td><td>94.88±.24 95.53±.07</td><td>96.54±.16</td><td>98.28±.04</td><td>97.94±.14</td><td>98.29±.09</td><td>95.23±.24</td><td>98.33±.06</td><td>98.21±0.06</td></tr><tr><td>Elliptic</td><td>93.32±.49</td><td>76.83±.65 65.27±3.95</td><td>86.31±.68 76.59±4.07</td><td>93.27±.23 84.38±1.85</td><td>86.21±.60 73.69±4.09</td><td>77.85±.39</td><td>74.59±.55</td><td>85.66±.39</td><td>92.30±.07</td><td>89.83±.41</td><td>92.52±.26</td><td>88.10±.29</td><td>93.13±.38</td><td>93.61±0.15 86.44±0.37</td></tr><tr><td>Recall-min.</td><td>85.23±1.54 87.82±.67</td><td>70.63±1.95</td><td>79.40±.96</td><td></td><td>77.54±1.27</td><td>65.10±2.80 69.92±.77</td><td>68.88±3.54 69.68±1.16</td><td>70.05±1.90 77.23±.81</td><td></td><td>82.11±1.36 79.38±11.04 78.30±6.03</td><td>85.02±3.49 85.03±6.47</td><td>78.36±.81 85.35±.19</td><td>84.16±1.01 87.08±.91</td><td>88.54±0.31</td></tr><tr><td>F1-min. Macro-F1</td><td>93.26±0.38</td><td>75.65±4.48</td><td>88.61±0.53</td><td>87.86±.59 93.36±0.29</td><td>87.60±0.70</td><td>83.42±0.38</td><td>83.18±0.61</td><td>87.73±0.47</td><td>85.32±.20 91.75±0.40</td><td>87.90±3.46</td><td>91.61±3.89</td><td>91.93±0.10</td><td>93.16±0.40</td><td>93.68±0.17</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AUROC</td><td>98.07±0.01</td><td>87.67±1.01</td><td>96.31±.10</td><td>98.07±.06</td><td>96.99±.09 95.28±.10</td><td>94.61±.17</td><td>96.39±.17</td><td>97.90±.13</td><td>97.33±.14</td><td>95.33±.66</td><td>92.36±.40</td><td>98.08±0.03</td><td>98.32±0.04</td></tr><tr><td>Elliptic++ Trans. Recall-min.</td><td>AUPRC</td><td>92.79±0.25 84.12±2.54</td><td>68.47±.88</td><td>85.04±.29 75.08±2.30</td><td>92.28±.30</td><td>85.35±.28 64.83±.58</td><td>72.88±.89</td><td>85.25±.50</td><td>90.31±.11</td><td>89.60±.45</td><td>84.67±1.69</td><td>81.92±.94</td><td>93.22±0.26</td><td>93.70±0.14 86.20±0.70</td></tr><tr><td>F1-min.</td><td></td><td></td><td>56.16±4.75</td><td></td><td>84.20±1.80</td><td>75.80±1.38 41.48±6.18</td><td>69.08±4.39</td><td>74.37±1.94</td><td>84.89±.39</td><td>80.62±3.63</td><td>68.37±3.26</td><td>69.96±.75</td><td>84.59±1.14</td><td>88.70±0.29</td></tr><tr><td>Macro-F1</td><td></td><td>87.31±0.23</td><td>61.91±2.50</td><td>79.18±.50</td><td>87.58±.77</td><td>77.43±.68 52.69±3.92</td><td>67.85±.92</td><td>77.82±.72</td><td>83.84±.32</td><td>83.31±.94</td><td>78.20±2.26</td><td>78.30±.15</td><td>87.89±0.59</td><td>93.77±0.16</td></tr><tr><td></td><td></td><td>93.47±0.32</td><td>54.82±3.10</td><td>88.54±0.27</td><td>93.22±0.34</td><td>87.53±0.40 82.01±0.84</td><td>82.17±0.50</td><td>87.74±0.47</td><td>89.77±1.71</td><td>90.79±0.54</td><td>88.10±1.21</td><td>88.13±0.08</td><td>93.22±0.48</td><td></td></tr><tr><td></td><td>AUROC</td><td>98.85±0.06</td><td>97.87±.09</td><td>97.30±.09</td><td>96.59±.13</td><td>98.37±.07 95.28±.10</td><td>97.24±.10</td><td>96.92±.10</td><td>98.75±.04</td><td>98.44±.08</td><td>98.40±.08</td><td>90.16±.83</td><td>97.88±.06</td><td>98.62±0.02</td></tr><tr><td>AUPRC</td><td></td><td>89.69±0.82</td><td>85.60±.42</td><td>81.85±.53</td><td>78.84±.73</td><td>88.92±.42 64.83±.58</td><td>82.23±.38</td><td>82.27±.53</td><td>90.18±.33</td><td>87.97±.47</td><td>87.17±.48</td><td>79.22±2.10</td><td>84.66±.37</td><td>90.28±0.17</td></tr><tr><td>Elliptic++ Actors Recall-min.</td><td></td><td>75.06±1.70</td><td>68.55±1.93</td><td>61.84±2.68</td><td>60.62±4.30</td><td>70.58±1.32 41.48±6.18</td><td>66.42±3.10</td><td>67.86±2.24</td><td>75.50±2.33</td><td>72.35±3.31</td><td>71.39±2.77</td><td>65.49±3.45</td><td>67.83±2.65</td><td>79.90±0.55</td></tr><tr><td></td><td>F1-min.</td><td>81.21±0.97</td><td>78.25±.43</td><td>72.35±1.20</td><td>70.38±1.99</td><td>80.67±.68 52.69±3.92</td><td>74.78±1.37</td><td>76.23±.65</td><td>82.08±.53</td><td>79.50±.70</td><td>78.62±.89</td><td>72.48±2.30</td><td>76.73±.83</td><td>83.30±0.35</td></tr><tr><td></td><td>Macro-F1</td><td>90.75±0.48</td><td>88.59±0.23</td><td>85.52±0.61</td><td>84.48±1.03</td><td>89.86±0.39 75.31±1.96</td><td>86.76±0.74</td><td>87.52±0.35</td><td>90.58±0.28</td><td>89.22±0.35</td><td>88.77±0.46</td><td>66.90±2.77</td><td>87.73±0.64</td><td>91.33±0.26</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AUROC</td><td>99.26±.09</td><td>99.33±.17</td><td>99.63±.06</td><td>99.29±.09</td><td>99.48±.04 97.88±.13</td><td>99.22±.09</td><td>99.24±.08</td><td>99.28±.07</td><td>98.43±.21</td><td>99.15±.09</td><td>91.01±.39</td><td>99.17±.11</td><td>99.18±0.09</td></tr><tr><td></td><td>AUPRC</td><td>36.95±2.05</td><td>46.35±.60</td><td>47.40±2.69</td><td>42.86±2.10</td><td>42.94±2.11</td><td>32.08±1.04 38.65±1.42</td><td>39.42±1.36</td><td>40.76±1.52</td><td>26.30±1.33</td><td>33.09±2.55</td><td>19.90±1.25</td><td>34.34±1.92</td><td>50.07±0.89</td></tr><tr><td>Ethereum</td><td></td><td>23.24±12.98</td><td>36.71±8.75</td><td>39.61±11.50</td><td>30.18±10.24</td><td>34.91±4.40</td><td>31.17±5.26 21.85±6.93</td><td>24.68±7.84</td><td>27.95±7.26</td><td>24.77±3.67</td><td>11.89±15.29</td><td>8.24±3.10</td><td>23.61±9.33</td><td>54.55±2.99</td></tr><tr><td></td><td>Recall-min.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>F1-min.</td><td></td><td></td><td></td><td>38.64±9.87</td><td></td><td>31.68±6.73</td><td>34.21±7.18</td><td>37.58±6.64</td><td>31.79±1.03</td><td>14.94±15.72</td><td>14.96±4.20</td><td>31.83±9.84</td><td></td></tr><tr><td></td><td>Macro-F1</td><td>66.82±6.26</td><td>73.03±3.65</td><td>74.99±3.16</td><td>69.32±4.94</td><td>66.64±11.61 70.13±1.77</td><td></td><td>65.83±3.55 49.99±0.00</td><td>68.79±3.32</td><td>61.21±4.18</td><td>57.46±7.86</td><td>50.12±0.28</td><td>65.91±4.92</td><td>77.60±0.39</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>45.32±3.51</td><td>40.28±3.54</td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.26±0.76</td></tr><tr><td></td><td></td><td>30.84±15.46</td><td>46.08±7.65</td><td>46.58±8.87</td><td></td></table>

Table 3: Ablation results (%). The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Variant</td><td colspan="5">DGraph-Fin</td><td colspan="5">Elliptic</td></tr><tr><td>AUROC</td><td>AUPRC</td><td>Recall-min.</td><td>F1-min.</td><td>Macro-F1</td><td>AUROC</td><td>AUPRC</td><td>Recall-min.</td><td>F1-min.</td><td>Macro-F1</td></tr><tr><td>w/o Temporal Edge-Event Construction</td><td>75.54±0.12</td><td>3.52±0.09</td><td>41.35±3.54</td><td>6.79±0.18</td><td>49.69±0.02</td><td>97.77±0.11</td><td>91.83±0.24</td><td>83.99±1.51</td><td>85.59±0.30</td><td>92.10±0.56</td></tr><tr><td>w/o Distribution-Aware Selective Propagation</td><td>80.64±0.23</td><td>4.57±0.08</td><td>50.88±3.34</td><td>8.11±0.07</td><td>49.68±0.00</td><td>98.11±0.09</td><td>93.13±0.13</td><td>85.00±1.11</td><td>86.71±0.48</td><td>92.85±0.56</td></tr><tr><td>w/o Multi-View Discriminative Fusion</td><td>80.90±0.29</td><td>4.67±0.06</td><td>53.44±0.99</td><td>8.08±0.04</td><td>49.71±0.02</td><td>97.93±0.18</td><td>92.35±0.50</td><td>83.96±1.92</td><td>86.29±0.41</td><td>92.91±0.39</td></tr><tr><td>w/o Debiased Contrastive Learning</td><td>81.01±0.39</td><td>4.71±0.19</td><td>50.07±7.76</td><td>8.09±0.05</td><td>49.69±0.02</td><td>97.75±0.14</td><td>91.63±0.18</td><td>83.34±1.25</td><td>86.04±0.66</td><td>92.31±0.38</td></tr><tr><td>Full Model</td><td>81.19±0.23</td><td>4.76±0.07</td><td>55.54±0.95</td><td>8.24±0.10</td><td>49.72±0.06</td><td>98.21±0.06</td><td>93.61±0.15</td><td>86.44±0.37</td><td>88.54±0.31</td><td>93.68±0.17</td></tr></table>

![](images/a3f8ce1c31134b4d269b795baaa9c348c0f2b65f58db74d31edaab5e1059a8d9.jpg)  
Figure 2: Hyperparameter sensitivity of $\tau , \tau _ { { + } } , k ,$ and $\tau _ { c }$

• Licit  
Illicit  
![](images/f60d92de84fd31c71303b23f513be0462209f9f850701af8ed8d1e3ef5d83680.jpg)  
Figure 3: Visualization of Node Representations on Elliptic

and $\tau _ { c } = 0 . 0 5$ yield favorable performance across datasets. Performance remains relatively stable on four datasets, demonstrating good robustness to parameter variations. Ethereum is more sensitive due to its extremely high class imbalance and limited minority samples, making minority-class performance more susceptible to parameter changes.

RQ4 (Core Mechanism Analysis). Figure 3 uses UMAP [20]to visualize 500 licit and 500 illicit nodes from the Elliptic test set. Removing Distribution-Aware Selective Propagation causes illicit nodes to move toward dense licit regions, indicating increased minority representation assimilation. Without Multi-View Discriminative Fusion, the class boundary becomes less distinct, reflecting weaker discriminability for minority representations. In contrast, the full model produces less overlap and clearer class boundaries, validating the efectiveness of both mechanisms.

## 6 Conclusion

This paper proposes MDTE, a minority-aware difusion framework over temporal edge events for imbalanced node classification, to accurately identify minority-class nodes. MDTE develops two com plementary mechanisms, Distribution-Aware Selective Propagation and Multi-View Discriminative Fusion, to mitigate majority-class interference and enhance minority-class discriminability. Extensive experiments on five real-world datasets validate the efectiveness of MDTE. In future work, we plan to further improve the discriminability and robustness of minority-node representations.

## 7 Ethical Considerations

This work studies node classification on imbalanced temporal graphs, with potential applications to fraud, phishing, and illicitactivity detection. False positives cause legitimate accounts to be misclassified. Therefore, MDTE should support rather than replace human review in consequential decisions. Practical deployments should calibrate classification thresholds to application-specific costs and evaluate performance across relevant subgroups. All experiments use publicly released research datasets.

## References

[1] Alexander Amini, Wilko Schwarting, Ava Soleimany, and Daniela Rus. 2020. Deep Evidential Regression. In Advances in Neural Information Processing Systems, Vol. 33. https://proceedings.neurips.cc/paper/2020/hash/ aab085461de182608ee9f607f3f7d18f-Abstract.html

[2] Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani. 2006. Manifold Regularization: A Geometric Framework for Learning from Labeled and Unla beled Examples. Journal of Machine Learning Research 7 (2006), 2399–2434. https://www.jmlr.org/papers/v7/belkin06a.html

[3] Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: Identifying Density-Based Local Outliers. In Proceedings ofthe 2000 ACM SIGMOD International Conference on Management ofData. 93–104. doi:10.1145/ 335191.335388

[4] Deli Chen, Yankai Lin, Guangxiang Zhao, Xuancheng Ren, Peng Li, Jie Zhou, and Xu Sun. 2021. Topology-Imbalance Learning for Semi-Supervised Node Classification. In Advances in Neural Information Processing Systems, Vol. 34. 29885–29897.

[5] Dingshuo Chen, Shuchen Xue, Liuji Chen, Yingheng Wang, Qiang Liu, Shu Wu, Zhi-Ming Ma, and Liang Wang. 2025. Grafe: Graph Representation Learning via Difusion Probabilistic Models. arXiv preprint arXiv:2505.04956 (2025). https: //arxiv.org/abs/2505.04956

[6] Liang Chen, Jiaying Peng, Yang Liu, Jintang Li, Fenfang Xie, and Zibin Zheng. 2021. Phishing Scams Detection in Ethereum Transaction Network. ACM Transactions on Internet Technology 21, 1 (2021), 10:1–10:16. doi:10.1145/3398071

[7] Ching-Yao Chuang, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. 2020. Debiased Contrastive Learning. In Advances in Neural Information Processing Systems, Vol. 33. https://proceedings.neurips.cc/paper/ 2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html

[8] Qi Dai, Yumeng Song, Yu Gu, Fangfang Li, and Xiaohua Li. 2024. Difusion Model-Enhanced Contrastive Learning for Graph Representation. In Database Systems for Advanced Applications (Lecture Notes in Computer Science, Vol. 14855). 332–341. doi:10.1007/978-981-97-5572-1\_22

[9] Yijun Duan, Xin Liu, AdamJatowt, Hai Tao Yu, Steven Lynden, Kyoung-Sook Kim, and Akiyoshi Matono. 2022. SORAG: Synthetic Data Over-Sampling Strategy on Multi-Label Graphs. Remote Sensing 14, 18 (2022), 4479. doi:10.3390/rs14184479

[10] Nguyen Minh Duc and Viet Cuong Ta. 2026. Sequence Difusion Model for Temporal Link Prediction in Continuous-Time Dynamic Graph. arXiv preprint arXiv:2601.23233 (2026). https://arxiv.org/abs/2601.23233

[11] Youssef Elmougy and Ling Liu. 2023. Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3979–3990. doi:10.1145/3580305.3599803

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Dif fusion Probabilistic Models. In Advances in Neural Information Processing Systems, Vol. 33. https://proceedings.neurips.cc/paper/2020/hash/ 4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html

[13] Xuanwen Huang, Yang Yang, Yang Wang, Chunping Wang, Zhisheng Zhang, Jiarong Xu, Lei Chen, and Michalis Vazirgiannis. 2022. DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. In Advances in Neural Information Processing Systems, Vol. 35. https://proceedings.neurips.cc/ paper\_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract Datasets\_and\_Benchmarks.htm

[14] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations. https: //arxiv.org/abs/1412.6980

[15] Meng Liu, Yue Liu, Ke Liang, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2024. Deep Temporal Graph Clustering. In International Conference on Learning Representations. https://openreview.net/forum?id=ViNe1fjGME

[16] Yang Liu, Deyu Bo, Wenxuan Cao, Yuan Fang, Yawen Li, and Chuan Shi. 2025. Graph Positional Autoencoders as Self-supervised Learners. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1867–1878. doi:10.1145/3711896.3736990

[17] Yongxu Liu, Zhi Zhang, Yan Liu, and Yao Zhu. 2022. GATSMOTE: Improving Imbalanced Node Classification on Graphs via Attention and Homophily. Mathematics 10, 11 (2022), 1799. doi:10.3390/math10111799

[18] Stuart P. Lloyd. 1982. Least Squares Quantization in PCM. IEEE Transactions on Information Theory 28, 2 (1982), 129–137. doi:10.1109/TIT.1982.1056489

[19] Yihong Ma, Yijun Tian, Nuno Moniz, and Nitesh V. Chawla. 2025. Class-Imbalanced Learning on Graphs: A Survey. Comput. Surveys 57, 8 (2025), 1–16. doi:10.1145/3718734

[20] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. 2018. UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software 3, 29 (2018), 861. doi:10.21105/joss.00861

[21] Joonhyung Park, Jaeyun Song, and Eunho Yang. 2022. GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification. In International Conference on Learning Representations. https://openreview.net/ forum?id=MXEl7i-iru

[22] Liang Qu, Huaisheng Zhu, Ruiqi Zheng, Yuhui Shi, and Hongzhi Yin. 2021. ImGAGN: Imbalanced Network Embedding via Generative Adversarial Graph Networks. In Proceedings ofthe 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1390–1398. doi:10.1145/3447548.3467334

[23] Ladislav Rampšek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. 2022. Recipe for a General, Powerful, Scalable Graph Transformer. In Advances in Neural Information Processing Systems, Vol. 35. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html

[24] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael M. Bronstein. 2020. Temporal Graph Networks for Deep Learning on Dynamic Graphs. arXiv preprint arXiv:2006.10637 (2020). https: //arxiv.org/abs/2006.10637

[25] Min Shi, Yufei Tang, Xingquan Zhu, David A. Wilson, and Jianxun Liu. 2020. Multi-Class Imbalanced Graph Convolutional Network Learning. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence. 2879– 2885. doi:10.24963/ijcai.2020/398

[26] Jaeyun Song, Joonhyung Park, and Eunho Yang. 2022. TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification. In Proceedings ofthe 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162). 20369–20383. https://proceedings.mlr.press/v162/song22a. html

[27] Robert L. Thorndike. 1953. Who Belongs in the Family? Psychometrika 18, 4 (1953), 267–276. doi:10.1007/BF02289263

[28] Yuxing Tian, Yiyan Qi, Aiwen Jiang, Qi Huang, and Jian Guo. 2024. Latent Conditional Difusion-based Data Augmentation for Continuous-Time Dynamic Graph Model. arXiv preprint arXiv:2407.08500 (2024). https://arxiv.org/abs/2407. 08500

[29] Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. 2023. DiGress: Discrete Denoising Difusion for Graph Generation. In International Conference on Learning Representations. https: //openreview.net/forum?id=UaAD-Nu86WX

[30] Kefan Wang, Jing An, and Qi Kang. 2022. Efective-Aggregation Graph Convolutional Network for Imbalanced Classification. In 2022 IEEE International Conference on Networking, Sensing and Control. 1–5. doi:10.1109/ICNSC55942. 2022.10004069

[31] Zheng Wang, Xiaojun Ye, Chaokun Wang, Yuexin Wu, Changping Wang, and Kaiwen Liang. 2018. RSDNE: Exploring Relaxed Similarity and Dissimilarity from Completely-Imbalanced Labels for Network Embedding. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32. doi:10.1609/aaai.v32i1.11242

[32] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I. Weidele, Claudio Bellei, Tom Robinson, and Charles E. Leiserson. 2019. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. arXiv preprint arXiv:1908.02591 (2019). https://arxiv.org/abs/1908. 02591

[33] Chunyu Wei, Wenji Hu, Xingjia Hao, Yunhai Wang, Yueguo Chen, Bing Bai, and Fei Wang. 2025. Graph Evidential Learning for Anomaly Detection. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3122–3133. doi:10.1145/3711896.3736989

[34] Daniel Wesego. 2025. Graph Representation Learning with Difusion Generative Models. arXiv preprint arXiv:2501.13133 (2025). https://arxiv.org/abs/2501.13133

[35] Lirong Wu, Haitao Lin, Zhangyang Gao, Cheng Tan, and Stan Z. Li. 2021. GraphMixup: Improving Class-Imbalanced Node Classification on Graphs by Self-supervised Context Prediction. arXiv preprint arXiv:2106.11133 (2021). https://arxiv.org/abs/2106.11133

[36] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2020. Inductive Representation Learning on Temporal Graphs. In International Conference on Learning Representations. https://openreview.net/forum?id= rJeW1yHYwH

[37] Run Yang, Yuling Yang, Fan Zhou, and Qiang Sun. 2023. Directional Difusion Models for Graph Representation Learning. In Advances in Neural Information Processing Systems, Vol. 36. https://proceedings.neurips.cc/paper\_files/paper/

2023/hash/6751ee6546b31ceb7d4ee12276b9f4d9-Abstract-Conference.html

[38] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. 2023. Towards Better Dynamic Graph Learning: New Architecture and Unified Library. In Advances in Neural Information Processing Systems, Vol. 36. https://proceedings.neurips.cc/ paper\_files/paper/2023/hash/d611019afba70d547bd595e8a4158f55-Abstract-Conference.html

[39] Sukwon Yun, Kibum Kim, Kanghoon Yoon, and Chanyoung Park. 2022. LTE4G: Long-Tail Experts for Graph Neural Networks. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management. 2434–2443. doi:10.1145/3511808.3557381

[40] Tianxiang Zhao, Xiang Zhang, and Suhang Wang. 2021. GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks. In Proceedings ofthe 14th ACM International Conference on Web Search and Data Mining. doi:10.1145/3437963.3441720

[41] Ziyu Zheng, Yaming Yang, Ziyu Guan, Wei Zhao, and Weigang Lu. 2025. Discrepancy-Aware Graph Mask Auto-Encoder. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4038–4049. https: //arxiv.org/abs/2506.19343

[42] Cai Zhou, Xiyuan Wang, and Muhan Zhang. 2024. Unifying Generation and Prediction on Graphs with Latent Graph Difusion. In Advances in Neural Information Processing Systems, Vol. 37. https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html

[43] Dawei Zhou, Jingrui He, Hongxia Yang, and Wei Fan. 2018. SPARC: Self-Paced Network Representation for Few-Shot Rare Category Characterization. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. doi:10.1145/3219819.3219952

[44] Mengting Zhou and Zhiguo Gong. 2023. GraphSR: A Data Augmentation Algorithm for Imbalanced Node Classification. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 37. 4954–4962. doi:10.1609/aaai.v37i4.25622

[45] Junyou Zhu, Langzhou He, Chao Gao, Dongpeng Hou, Zhen Su, Philip S. Yu, Juergen Kurths, and Frank Hellmann. 2025. SDMG: Smoothing Your Difusion Models for Powerful Graph Representation Learning. In Proceedings of the 42nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 267). 79815–79835. https://proceedings.mlr.press/v267/zhu25g.htm