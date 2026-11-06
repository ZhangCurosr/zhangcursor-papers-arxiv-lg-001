# GTAFL: Addressing Test-Agnostic Long-Tailed Federated Learning in Medical Image

Guangyu Chen, Jingyun Zeng, Hui Jiang

Abstract— In healthcare, protecting patient privacy is crucial due to the sensitivity of medical data and its extensive accessibility. Federated Learning (FL) offers a decentralized and privacy-preserving training paradigm, making it an ideal solution for healthcare applications. A critical challenge in healthcare is that real-world medical data often exhibits long-tailed distributions in local and global views. Existing methods addressing long-tailed FL problem typically assume that the model will be evaluated on uniform test data distribution. However, Practical test data in healthcare systems is often agnostic and unpredictable, leading to potential model failures in realworld scenarios. In this paper, we introduce a novel task termed Test-Agnostic Long-Tailed Federated Learning and propose GTAFL, a comprehensive framework to address this challenge. During the training stage, GTAFL employs adaptive re-sampling, expert classifier retraining, and selfsupervised learning to correct biased classifiers and distorted feature spaces caused by long-tailed training distributions. During the inference stage, an ensemble mechanism combines retrained expert classifiers to handle test data with unknown distributions. Extensive experiments on CIFAR10 and two medical datasets manifest that our framework outperforms other state-of-the art methods. The code is available at https://github.com/cgy-lavine/GTAFL

Index Terms— Federated Learning, Class Imbalance, Test-Agnostic, Healthcare, Data Privacy

## I. INTRODUCTION

Recently, the rapid proliferation of Internet of Things (IoT) technologies has significantly expanded the application of deep learning techniques for processing heterogeneous data from diverse sources. The Internet of Medical Things (IoMT), a specialized subset of IoT infrastructure interconnected through computer networks [1], enables comprehensive data collection for patient monitoring and treatment optimization [2]. However, data privacy protection has emerged as a critical concern in IoMT applications. Federated Learning (FL) addresses this challenge by allowing multiple nodes to collaboratively train powerful models without exposing raw data [3]–[6], offering a privacy-preserving training strategy with considerable potential in medical analysis [7]–[10].

![](images/92704dd4d61386cc889192d3bff83a0921d54e0a3dd65c4ec97da98a0926a342.jpg)

![](images/0c0da9f966f8be4eabf946a4fcdbbcf479dfae8f96c6da06c398360f7b194735.jpg)  
Fig. 1: The illustration of two practical challenges: class imbalance and test agnostics. Existing method is only focused on uniform tests in federated settings.

However, real-world federated learning (FL) faces significant challenges, notably class imbalance, which is common in medical datasets due to the rarity of certain conditions [11]. Such imbalance often biases models toward head classes, leading to poor performance on tail classes [12], [13]. To mitigate this, several studies [14]–[16] have investigated longtailed recognition in federated settings. However, existing methods assume that the test data is uniformly distributed. As illustrated in Fig. 1, this assumption rarely holds in practice as the test distribution is often unknown. For example, a model trained primarily on healthy individuals catastrophically degrades during an epidemic when the proportion of patient data surges. We refer to this practical yet underexplored task as Test-Agnostic Long-Tailed Federated Learning. To the best of our knowledge, this task has not been previously studied in the federated learning literature. <sup>1</sup>

To address this task, two primary challenges emerge. First, class imbalance manifests at both global and local levels. Globally, the aggregated client datasets typically exhibit longtailed distributions where head classes substantially outnumber tail classes, despite the critical importance of rare classes in safety-sensitive domains such as medical diagnosis [17]. Consequently, FL models demonstrate degraded performance on infrequent classes due to insufficient representation during training. Locally, imbalance occurs at the individual client level, where certain classes dominate specific clients while remaining scarce or entirely absent on others, thereby exacerbating inter-client heterogeneity (non-IID data). This distributional inconsistency results in biased aggregation, as clients develop divergent feature representations and decision boundaries for identical classes, ultimately compromising global model performance [18], [19]. Although recent FL approaches, including re-sampling strategies [20]–[22], classifier re-training [23], and feature-space adjustments [24], attempt to mitigate these imbalances, they predominantly assume uniform test distributions, an assumption that rarely holds in practice.

The second challenge lies in generalization to unknown test distributions. In centralized training, methods can leverage prior knowledge for post-hoc adjustments [25] or train multiple expert models [26] to accommodate various test distributions. However, FL settings impose fundamental constraints that render such approaches impractical. Resource limitations on edge devices preclude training multiple models locally [27], [28], while the distributed nature of FL complicates adaptation to unpredictable distribution shifts. For instance, while SADE [26] successfully trains multiple expert models with different loss functions for ensemble prediction, this approach fails in FL contexts where class absence at individual clients results in poorly trained experts, thereby compromising aggregation and ensemble effectiveness. Consequently, practical FL solutions must achieve robustness to shifting test distributions while operating within inherent constraints imposed by client resource limitations and heterogeneous training distributions.

Therefore, we propose GTAFL, a framework for relieving test-agnostic long-tailed federated learning. Rather than requiring resource-constrained clients to train multiple potentially biased expert models, GTAFL leverages global information, specifically the aggregated training distribution, to align local and global training distributions through a global informationguided sampling strategy. Additionally, this global information guides the generation of synthetic data (termed global federated features), which enables the server to train multiple unbiased expert classifiers. This design draws inspiration from twostage training paradigms [23], [29], wherein a fixed feature extractor and balanced data are used to retrain biased classifiers. To better capture intrinsic visual features under unknown test distributions, we employ server-side SimCLR [30] on unlabeled test set, without exposing any personally identifiable metadata [31]. Since SimCLR is purely self-supervised and never accesses label information, this does not constitute data leakage, consistent with established FL works that leverage unlabeled server-side data for model refinement [32], [33]. These sampling strategies and strengthened representations, combined with classifier retraining and expert ensemble during the inference stage, enable GTAFL to maintain robustness under unknown and shifting test distributions. Extensive experiments on three datasets demonstrate that GTAFL outperforms state-of-the-art federated learning methods across various data distributions. The main contributions of this paper are summarized as follows:

• We introduce and analyze a new federated learning setting, i.e., test-agnostic long-tailed federated learning, in which models must contend simultaneously with longtailed training data and unknown, potentially shifting testtime distributions.

TABLE I: Notation Table
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $K$ </td><td>Number of federated clients</td></tr><tr><td> $C$ </td><td>Number of classes</td></tr><tr><td> $\mathbf { w } = \{ \mathbf { u } , \mathbf { v } \}$ </td><td>Global model parameters</td></tr><tr><td> ${ \mathrm { \mathbf { u } } _ { } ^ { t } }$ </td><td>Feature extractor parameters at round t</td></tr><tr><td> $\mathrm { ~ \bf ~ v ~ } ^ { t }$ </td><td>Original classifier parameters at round t</td></tr><tr><td> $\hat { \mathbf { v } } ^ { t }$ </td><td>Expert classifiers parameters at round t</td></tr><tr><td> $\hat { \mathbf { v } } _ { f } , \hat { \mathbf { v } } _ { u } , \hat { \mathbf { v } } _ { b }$ </td><td>Forward, uniform, backward experts</td></tr><tr><td> $f _ { \mathbf { u } }$ </td><td>Feature extractor function</td></tr><tr><td> $h _ { \mathbf { V } }$ </td><td>Classifier function</td></tr><tr><td> $m _ { c } ^ { ( i ) }$ </td><td>Number of class c samples on client i</td></tr><tr><td> $\smash { d _ { 1 , \ldots , 3 } ^ { ( i ) } }$  local</td><td>Local class-count vector for client i</td></tr><tr><td> $d _ { \mathrm { g l o b a l } }$ </td><td>Global class-count vector</td></tr><tr><td> $\breve { \mathbf { I F } }$ </td><td>Imbalance factor</td></tr><tr><td> $A ^ { t }$ </td><td>Selected clients subset at round t</td></tr><tr><td> $w _ { c } ^ { i }$ </td><td>Sampling weight for class c on client ¿i</td></tr><tr><td> $\alpha _ { i }$  a</td><td>Adaptive mixing parameter for client i</td></tr><tr><td> $\mathbf { g } _ { c , \hat { \mathbf { v } } _ { h } ^ { t } } ^ { J }$   $\mathbf { \Pi } _ { \mathrm { a g g } } ^ { \mathrm { c } , \mathrm { v } }$ </td><td>Class c gradient from client j for expert k</td></tr><tr><td> $\mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathbf { a } _ { \infty } }$  gff</td><td>Aggregated class c gradient for expert k</td></tr><tr><td> $\mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \circ ^ { \scriptscriptstyle - } }$ </td><td>Class c gradient from global federated features</td></tr><tr><td> $\mathbf { F } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { t }$ </td><td>Global federated features for class c, expert k</td></tr><tr><td> $\tilde { m } _ { c }$ </td><td>Global federated features count for class c</td></tr><tr><td> $w _ { k }$ </td><td>Confidence weight for expert k</td></tr></table>

• We present GTAFL, a novel framework designed to address this challenge through four key components: adaptive re-sampling, expert classifier retraining, selfsupervised learning, and ensemble learning.

• We investigate synthetic global federated features and find that mild distributional imbalance does not significantly impair classifier retraining. Experiments show GTAFL achieves 8.5% average improvements in backward test distributions while maintaining competitive performance in forward and uniform scenarios.

## II. PRELIMINARIES AND PROBLEM FORMULATION

## A. Federated Learning

Federated learning (FL) aims to obtain a global model without sharing local data. In a typical FL setting, suppose that we have K clients, each indexed by $k \in \{ 1 , 2 , \ldots , K \}$ Each client k holds a local dataset $\mathcal D ^ { k } \ = \ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N ^ { k } } ,$ where $N ^ { k }$ denotes the number of samples on the k-th client. Client datasets are often heterogeneous and non-independent and identically distributed (non-IID). The goal of FL is to collaboratively learn a global model $\phi _ { \mathbf { w } }$ over the union of all local datasets $\textstyle { \mathcal { D } } = { \bigsqcup } _ { k = 1 } ^ { K } { \mathcal { D } } ^ { k }$ with the coordination of a central server, while preserving data privacy by avoiding direct data transmission. The global model $\phi _ { \mathbf { w } }$ is parameterized by learnable weights $\mathbf { w } = \{ \mathbf { u } , \mathbf { v } \}$ and typically consists of two components: (1) a feature extractor $\bar { f } _ { \mathbf { u } } : \bar { \mathcal { X } } \ \to \ \mathcal { Z } \ \subseteq \ \mathbb { R } ^ { d }$ with parameters u, which maps each input $\textbf { x } \in \mathcal { X }$ to a ddimensional feature vector $\mathbf { z } = f _ { \mathbf { u } } ( \mathbf { x } ) \in \mathcal { Z } ; ( 2 )$ a classifier $h _ { \mathbf { v } } : \mathcal { Z }  \mathbb { R } ^ { | \mathbb { Z } | }$ with parameters v, which maps a feature vector $\mathbf { z } \in { \mathcal { Z } }$ to a vector of logits. It is typically implemented as a fully connected layer that outputs class confidence scores, where $\mathcal { T } = 1 , 2 , \dotsc , C$ denotes the set of C classes.

## B. Execution Progress of Federated Learning

In FL, the training process consists of multiple communication rounds, where each round t involves the following two steps. The first step is the local model training. At the beginning of round t, the central server broadcasts the current global model parameters $\mathbf { w } ^ { t }$ to a selected subset of clients $\mathcal { A } ^ { t } \subseteq \{ 1 , 2 , \ldots , K \}$ . Suppose that the parameters of the $k \mathrm { - }$ th client’s local model are denoted as ${ \bf w } _ { k }$ . Each participating client $k \in \mathcal { A } ^ { t }$ receives the global model and performs local training on their private dataset $\mathcal { D } ^ { k }$ using gradient descent. The local model update is given by:

$$
\mathbf { w } _ { k } ^ { t + 1 } \gets \mathbf { w } _ { k } ^ { t } - \eta \nabla _ { \mathbf { w } } \ell ( \mathbf { w } _ { k } ^ { t } ; \mathcal { D } ^ { k } )\tag{1}
$$

where ℓ is cross-entropy loss.

The second step is the model aggregation. After local training, the participating clients upload their updated model parameters $\bar { \mathbf w } _ { k } ^ { t + 1 }$ to the server. The server then performs a weighted averaging operation to update the global model for the next communication round:

$$
\mathbf { w } ^ { t + 1 } = \sum _ { k \in \mathcal { A } ^ { t } } \frac { \left| \mathcal { D } ^ { k } \right| } { \sum _ { j \in \mathcal { A } ^ { t } } \left| \mathcal { D } ^ { j } \right| } \mathbf { w } _ { k } ^ { t + 1 }\tag{2}
$$

Note that we use FedAvg [3] as our aggregation method, which serves as the foundation for most federated learning approaches.

## C. Local and Global Data Distributions

Let $m _ { c } ^ { ( i ) }$ denote the number of samples from class $c \in$ $\{ 1 , \ldots , C \}$ on client $i \in \{ 1 , \ldots , K \}$ . The local class-count vector for client i is formally defined as

$$
\pmb { d } _ { \mathrm { l o c a l } } ^ { ( i ) } = \big ( m _ { 1 } ^ { ( i ) } , \dots , m _ { C } ^ { ( i ) } \big ) \in \mathbb { R } _ { + } ^ { C }\tag{3}
$$

The global class-count vector, which aggregates the class distributions across all K participating clients, is given by

$$
d _ { \mathrm { g l o b a l } } = \left( \sum _ { i = 1 } ^ { K } m _ { 1 } ^ { ( i ) } , \ldots , \sum _ { i = 1 } ^ { K } m _ { C } ^ { ( i ) } \right) \in \mathbb { R } _ { + } ^ { C }\tag{4}
$$

To quantify the degree of class imbalance inherent in the global distribution, we introduce the imbalance factor (IF), also defined as the imbalanced ratio between the maximum and minimum global class counts:

$$
\mathbf { I F } \ = \ { \frac { \operatorname* { m a x } _ { c \in \{ 1 , . . . , C \} } \sum _ { i = 1 } ^ { K } m _ { c } ^ { ( i ) } } { \operatorname* { m i n } _ { c \in \{ 1 , . . . , C \} } \sum _ { i = 1 } ^ { K } m _ { c } ^ { ( i ) } } }\tag{5}
$$

## D. Problem Formulation

We provide a problem formulation of test-agnostic longtailed federated learning, where there exists distribution discrepancy between training and testing data. Specifically, the global training set $\mathcal { D } _ { \mathrm { t r a i n } }$ follows a long-tailed distribution characterized by the imbalance factor IF, while the test set $\mathcal { D } _ { \mathrm { t e s t } }$ has an unknown class distribution that may differ significantly from the training distribution.

Let $\mathcal { D } _ { \mathrm { t e s t } }$ with class counts $( m _ { 1 } , m _ { 2 } , \ldots , m _ { C } )$ where each $m _ { c }$ is unknown and potentially different from the training class frequencies. The test distribution can follow forward, uniform, or backward patterns relative to the training long-tail.

Our objective is to learn a robust global model that performs well across different potential test distributions:

$$
\operatorname* { m i n } _ { \mathbf { w } } \mathbb { E } _ { \mathcal { D } _ { \mathrm { t e s t } } } \left[ \frac { 1 } { | \mathcal { D } _ { \mathrm { t e s t } } | } \sum _ { ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \mathrm { t e s t } } } \ell ( \phi _ { \mathbf { w } } ( x _ { i } ) , y _ { i } ) \right]\tag{6}
$$

where the expectation is over the unknown test distribution and ℓ is the cross-entropy loss.

## III. MOTIVATION AND OVERVIEW

## A. Motivation Study

In centralized training, long-tailed data distributions and distribution shifts between training and testing data typically degrade model performance [34]–[37]. In federated learning (FL) with imbalanced data, such shifts present even greater challenges due to data heterogeneity across clients. Previous work [24], [38] demonstrates that in long-tailed FL scenarios, fine-tuning only the classifier can effectively mitigate longtailed bias without complete model retraining. The key challenge lies in training multiple expert classifiers to handle various tests with unknown distribution. Due to privacy constraints inherent in federated learning, servers cannot directly access training data. Consequently, auxiliary datasets or synthetic data are commonly used for expert model training. CReFF [23] exemplifies this approach by leveraging federated features, balanced synthetic data trained from client-transmitted gradients, to retrain biased classifiers. Since federated features are balanced, a natural extension involves modifying their distribution to enable optimal classifier performance on categories with synthetic majority representation. Our experimental analysis on CIFAR10-LT, illustrated in Figure 3, reveals that retraining classifiers on federated features with different distribution yields varying performance in different tests. Specifically, classifiers retrained on distribution-specific federated features demonstrate superior performance on their corresponding test distributions. For instance, the backwardtrained classifier achieves exceptional performance on backward test distributions while maintaining competitive accuracy across other distribution types.

The train-test distribution discrepancy becomes more pronounced when data is limited, as demonstrated in Fig. 2. In medical imaging, particularly during the pandemic, substantial volumes of unlabeled imaging data have become available. To effectively leverage this data while addressing the high annotation costs and stringent privacy requirements inherent to medical applications, we employ a self-supervised contrastive learning approach [31], [39]. Specifically, we implement the SimCLR framework [30] on the server side to enable efficient feature learning from unlabeled medical images. This approach is particularly well-suited for federated medical scenarios as it exclusively utilizes image data without accessing any patient information or other sensitive metadata. The self-supervised learning process focuses solely on extracting intrinsic visual features, thereby maintaining the privacy-preserving principles fundamental to federated learning while enhancing the model’s representational capabilities on medical imaging tasks.

![](images/487836ee4972bc093f1f381ee9c179ba14ac8630ff3f2064221ce48945635f86.jpg)  
Fig. 2: CReFF performance degradation with reduced training data size across different test distributions. Performance drops significantly as training data decreases from 100% to 10%, with Backward50 test showing the most severe degradation

![](images/53c61b6ee8ab08ba70ff3ace783d708c8ec9ace25df45d2f3697372e2d5d9bc2.jpg)  
Fig. 3: Expert classifier specialization demonstrates that each classifier achieves optimal performance on its corresponding test distribution while maintaining competitive accuracy across other test distribution.

## B. Overview ofGTAFL

The primary goal of GTAFL is to train specialized expert classifiers and effectively combine them during the inference stage to handle varying test distributions. As illustrated in Fig. 4, the proposed GTAFL framework consists of four components: data sampling, expert classifiers retraining, selfsupervised refinement, and class-wise experts ensemble. Each communication round consists of the following steps: (1) Clients train the received global model based on the GI-GDS sampling strategy while computing class-wise gradients with respect to the expert classifiers from server. (2) Clients transmit the updated local model and class-wise gradients to the server. (3) On the server side, the server aggregates feature extractor, original classifier and the class-wise gradients. The expert classifiers are retrained based on the optimized global federated features. The feature extractor is further improved based on self-supervised method. (4) The server sends the aggregated model and retrained expert classifiers back to the clients, returning to stage (1). (5) During the inference stage, the expert classifiers with the shared feature extractor vote collaboratively to make predictions on test data.

As illustrated in Section IV-A, global information is leveraged for local data sampling to achieve a pseudo-balanced state during training. At this stage, only the feature extractor and the original classifier are updated locally. On the server side, the feature extractor and the original classifier are aggregated based on standard FedAvg approach [3]. Expert classifier retraining uses synthetic data, specifically global federated features (GFF) in GTAFL. In order to optimize the GFF, matching loss [40] is used for minimizing the gradient distance between synthetic data and real data toward the same classifiers. Subsequently, expert classifiers are trained using these updated GFF to achieve specialization for specific data distributions. After retraining the expert classifiers, the feature extractor is further refined using unlabeled data through the SimCLR technique, enabling the model to learn robust and generalized feature representations. This process ensures strict data privacy, as no sensitive information such as patient names is exposed.

## IV. METHODOLOGY

## A. Global Information–Guided Data Sampling (GI-GDS)

Sampling strategy that inverts the local class distribution can preserve minority-class signal on each client but often overcompensate, causing severe oversampling and noisy updates for rare classes [20]. Conversely, sampling solely from the global distribution can smooth extremes but may ignore clientspecific information that is important for personalization.

To balance these effects, we propose Global Information–Guided Data Sampling (GI-GDS), a hybrid sampling scheme that interpolates between local inverse-frequency and global inverse-frequency signals. For client i and class $c ,$ let $m _ { c } ^ { i }$ be the number of local samples in class c, and let $\begin{array} { r } { M _ { c } = \sum _ { j = 1 } ^ { N } m _ { c } ^ { j } } \end{array}$ denote the total across all N clients. The sampling weight is

$$
w _ { c } ^ { i } ( \alpha ) ~ = ~ \alpha \cdot \frac { 1 } { m _ { c } ^ { i } + \varepsilon } ~ + ~ ( 1 - \alpha ) \cdot \frac { 1 } { M _ { c } + \varepsilon }\tag{7}
$$

where $\varepsilon > 0 \ ( \mathrm { e . g . , 1 0 ^ { - 6 } } )$ prevents division by zero. We use normalized weights $\begin{array} { r } { \tilde { w } _ { c } ^ { i } = w _ { c } ^ { i } / \sum _ { c ^ { \prime } } w _ { c ^ { \prime } } ^ { i } } \end{array}$ to form the per-client sampling distribution.

At the start of each communication round, each participating client uploads its local class-count vector $\pmb { d } _ { \mathrm { l o c a l } } ^ { ( i ) }$ to the server.

![](images/235ed1f251fb676c9af069c9a1f842cecc99e99de67247cd4caf8e6c868258c8.jpg)  
Fig. 4: GTAFL framework workflow: (1) Clients receive global model and expert parameters, then perform local training with GI-GDS sampling and gradient computation. (2) Clients transmit class-wise gradients and model parameters to the server. (3) Server performs aggregation, expert retraining using optimized global federated features, and self-supervised refinement with unlabeled data. (4) The server sends the updated global model and expert classifiers parameters to clients. (5) Expert ensemble inference for test-agnostic prediction.

The server computes the global class-count vector M<sub>c</sub> by summation and broadcasts it back to all clients for use in the GI-GDS sampling weights. If the local class distribution $d _ { \mathrm { l o c a l } } ^ { i }$ substantially differs from the global distribution $d _ { \mathrm { g l o b a l } }$ , relying on global-only weights can erase informative local patterns. Conversely, when the two distributions are similar, emphasizing the global signal helps avoid aggressive oversampling of very rare local classes.

Let $s _ { i } = \cos \left( d _ { \mathrm { l o c a l } } ^ { i } , \ : d _ { \mathrm { g l o b a l } } \right)$ be the cosine similarity between the client’s class-frequency vector and the global frequency vector. We set

$$
\alpha _ { i } \ = \ \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f } \ s _ { i } < \tau , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. \qquad \mathrm { w i t h } \ \tau = 0 . 9\tag{8}
$$

Thus, when the local and global distributions diverge $( s _ { i } < \tau )$ we set $\alpha _ { i } = 1$ to preserve client-specific information; when they are similar, we set $\alpha _ { i } = 0$ to limit minority oversampling.

## B. Expert Classifier Retraining

In long-tailed federated learning, a single global classifier often fails to generalize uniformly, especially under trainingtesting distribution mismatch. To mitigate this mismatch, we maintain multiple classifier experts to specialize for a target prior while sharing a common feature extractor.

At communication round t, the server broadcasts global parameters $\{ \mathbf { u } ^ { t } , \mathbf { v } ^ { t } \}$ and expert classifiers parameters $\bigl \{ \hat { \mathbf { v } } _ { f _ { 1 } } ^ { t } , \hdots , \hat { \mathbf { v } } _ { f _ { p } } ^ { t } , \hat { \mathbf { v } } _ { u _ { 1 } } ^ { t } , \hdots , \hat { \mathbf { v } } _ { u _ { q } } ^ { t } , \hat { \mathbf { v } } _ { b _ { 1 } } ^ { t } , \hdots , \hat { \mathbf { v } } _ { b _ { r } } ^ { t } \bigr \}$ targeting fo uniform, and backward long-tail priors, respectively. Each client trains $\{ \mathbf { u } ^ { t } , \mathbf { v } ^ { t } \}$ locally following FedAvg. Furthermore, for class c, client $j$ samples $n _ { c } ^ { j }$ instances and reports the average gradient of cross-entropy loss for every expert classifier. The gradient computation for each expert classifier parameters $\hat { \mathbf { v } } _ { k } ^ { t } \in \hat { \mathbf { v } } ^ { t }$ , is formulated as

$$
\mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { j } = \frac { 1 } { n _ { c } ^ { j } } \sum _ { i = 1 } ^ { n _ { c } ^ { j } } \nabla _ { \hat { \mathbf { v } } _ { k } ^ { t } } \ell \left( h _ { \hat { \mathbf { v } } _ { k } ^ { t } } ( f _ { \mathbf { u } ^ { t } } ( x _ { c , i } ^ { j } ) ) , y _ { c , i } ^ { j } \right)\tag{9}
$$

Upon receiving updates from clients $\mathcal { A } ^ { t }$ , the server aggregates model parameters to obtain $\{ \tilde { \mathbf { u } } ^ { t } , \mathbf { v } ^ { t + 1 } \}$ and averages class-wise expert gradients :

$$
\mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { a g g } } = \frac { 1 } { | \mathcal { A } _ { c } ^ { t } | } \sum _ { j \in \mathcal { A } _ { c } ^ { t } } \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { j }\tag{10}
$$

To enable server-side expert retraining without client raw data, the server maintains global federated features $\mathbf { F } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { t } \in$ $\mathbb { R } ^ { \tilde { m } _ { c } \times d }$ for each class c and expert $\hat { \mathbf { v } } _ { k } ^ { t }$ . These features are designed to generate gradients that approximate those from raw data. The gradients generated by GFF for $\hat { \mathbf { v } } _ { k } ^ { t }$ and class c are as following:

$$
\mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { g f } } = \frac { 1 } { \tilde { m } _ { c } } \sum _ { i = 1 } ^ { \tilde { m } _ { c } } \nabla _ { \hat { \mathbf { v } } _ { k } ^ { t } } \ell \left( h _ { \hat { \mathbf { v } } _ { k } ^ { t } } ( \mathbf { F } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } , i } ^ { t } ) , y _ { c } \right)\tag{11}
$$

The GFF’s class-specific count is defined as $\tilde { m } _ { c } = m +$ $\begin{array} { r l } { \hat { \mathbf { v } } ^ { t } = } & { { } \lambda \tilde { d } _ { \mathrm { g l o b a l } } ^ { c } , } \end{array}$ , where m is a balanced base count and $\tilde { d } _ { \mathrm { g l o b a l } } ^ { c }$ is the rward, global class frequency. For the balanced expert, we set $\lambda = 0 .$

These features are refined by aligning their induced gradi ents to the aggregated client gradients. We optimize the GFF

by minimizing the cosine matching loss:

$$
D \left( \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { g f } } , \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { a g g } } \right) = 1 - \frac { \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { g f } } \cdot \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { a g g } } } { \left\| \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { g f } } \right\| \left\| \mathbf { g } _ { c , \hat { \mathbf { v } } _ { k } ^ { t } } ^ { \mathrm { a g g } } \right\| }\tag{12}
$$

After refinement, we obtain $\mathbf { F } _ { \hat { \mathbf { v } } _ { k } ^ { t } } ^ { t }$ for all classes and retrain each expert parameters $\hat { \mathbf { v } } _ { k } ^ { t }$ while freezing $\tilde { \textbf { u } } ^ { t }$ . The server updates the expert classifiers as:

$$
\hat { \mathbf { v } } _ { k } ^ { t + 1 } = \hat { \mathbf { v } } _ { k } ^ { t } - \eta \nabla _ { \hat { \mathbf { v } } _ { k } ^ { t } } \ell \left( h _ { \hat { \mathbf { v } } _ { k } ^ { t } } ( \mathbf { F } _ { \hat { \mathbf { v } } _ { k } ^ { t } } ^ { t } ) , y \right)\tag{13}
$$

## C. Self-Supervised Refinement

After aggregating the feature extractor parameters $\widetilde { \mathbf { u } } ^ { t }$ and completing expert retraining, we further refine the feature extractor using self-supervised refinement, i.e., contrastive learning, on unlabeled images available at test time. This step leverages medical images only, preserving the patients’ privacy and federated learning protocol.

At communication round t, the server draws N unlabeled images and applies two stochastic augmentations to each. Each view forms one positive pair with its counterpart and $2 N - 2$ negatives with the remaining views in the batch. The selfsupervised refinement loss is defined as follows.

$$
\mathcal { L } = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \mathcal { L } _ { 2 i - 1 , 2 i } + \mathcal { L } _ { 2 i , 2 i - 1 } \right]\tag{14}
$$

L denotes the NT-Xent loss [30]

$$
\mathcal { L } _ { i , j } = - \log \frac { \exp ( \sin ( z _ { i } , z _ { j } ) / \tau ) } { \sum _ { k = 1 , k \neq i } ^ { 2 N } \exp ( \sin ( z _ { i } , z _ { k } ) / \tau ) }\tag{15}
$$

where $\sin ( z _ { i } , \mathrm { z _ { j } } ) = z _ { i } \cdot z _ { j } / \| z _ { i } \| \| z _ { j } \|$ denote the dot product between $\ell _ { 2 }$ normalized $z _ { i }$ and $z _ { j } . \ z _ { i }$ is the projected feature of view i, and $\tau$ is the temperature.

We update the feature extractor by gradient descent,

$$
\mathbf { u } ^ { t + 1 } = \tilde { \mathbf { u } } ^ { t } - \eta \nabla _ { \tilde { \mathbf { u } } ^ { t } } \mathcal { L }\tag{16}
$$

using learning rate η for this self-supervised phase, while keeping $\mathbf { v } ^ { t + 1 }$ and the experts $\hat { \mathbf { v } } ^ { t + 1 }$ fixed. The refined global model parameters $\{ \mathbf u ^ { t + 1 } , \mathbf v ^ { t + 1 } \}$ and expert classifiers parameters $\hat { \mathbf { v } } ^ { \bar { t } + 1 }$ are then disseminated to clients for round $t + 1$

## D. Class-wise Experts Ensemble

After t round training, GTAFL aggregates predictions from multiple experts through a hard voting mechanism to accommodate unknown test priors. Given an input x, features $\begin{array} { r } { z = f _ { \mathbf { u } ^ { t } } ( x ) } \end{array}$ are extracted and evaluated by each expert classifier, which produces a discrete class prediction $\hat { y } _ { k } ~ =$ arg max<sub>c</sub> $h _ { \hat { \mathbf { v } } _ { k } ^ { t } } ( z )$ by selecting the class with the highest logits score.

We combine these discrete predictions through a weighted voting mechanism. For each candidate class c, we count the weighted votes from all experts who predict that class. The final prediction is determined by selecting the class that receives the maximum weighted vote count:

$$
\hat { y } = \arg \operatorname* { m a x } _ { c } \sum _ { k = 1 } ^ { p + q + r } w _ { k } \cdot \mathbb { I } [ \hat { y } _ { k } = c ]\tag{17}
$$

where $w _ { k } \geq 0$ represents the confidence weight for the k-th expert, and I[·] is the indicator function that equals 1 when the condition is true and 0 otherwise.

In the default implementation, we use uniform weighting $( w _ { k } \ = \ 1$ for all experts) with $p \ = \ q \ = \ r \ = \ 1$ , yielding one expert per distribution type. This democratic approach ensures equal contribution from each expert regardless of test distribution shifts, without requiring probability calibration or confidence estimation. The framework can be extended to support adaptive weighting schemes based on expert reliability or validation performance when such information is available.

## V. EXPERIMENTS SETUP

## A. Datasets

To evaluate GTAFL’s effectiveness, we conduct experiments on three benchmark datasets: CIFAR10-LT, PathMNIST-LT, and OCTMNIST-LT. These datasets represent diverse domains including natural images and medical imaging applications. Table II summarizes the key characteristics of each dataset. We follow a standardized two-stage protocol for creating long-tailed federated datasets. First, we generate globally imbalanced datasets with an imbalance factor of 50 using exponential decay. Second, we partition datasets among clients using Dirichlet distribution with $\alpha = 0 . 5 [ 4 1 ]$ to simulate non-IID federated environments. The following provides detailed specifications for each dataset.

• CIFAR10-LT: Long-tailed variant of CIFAR-10 [42] containing 50,000 training images (32×32 RGB) across 10 object categories, providing a standard computer vision benchmark for evaluating general image classification under federated long-tail conditions.

• PathMNIST-LT: Derived from the MedMNIST collection [43], this colon pathology dataset comprises 89,996 training samples across 9 diagnostic classes with 7,180 test samples, representing realistic medical imaging scenarios with inherent class imbalance.

• OCTMNIST-LT: Derived from the MedMNIST collection [43], OCTMNIST is a retinal optical coherence tomography dataset containing 97,477 training samples across 4 retinal condition categories with 1,000 test samples, providing validation for ophthalmological applications where rare conditions are clinically significant.

## B. Baselines

In this section, we conduct extensive experiments to evaluate the performance of GTAFL, First, we evaluate the accuracy on various test with different IF on three datasets. Next, we compare the class-wise performance on six baselines. Last but not least, we use an ablation study to find the effectiveness of resampling, classifier retraining, contrastive learning and ensemble strategies. In particular, the baseline includes traditional FL methods and other long-tailed oriented FL methods are compared with GTAFL.

• FedAvg [3]: The canonical FL baseline that alternates local SGD on each client with server-side aggregation via data-size–weighted averaging of model parameters.

TABLE II: Summary of datasets used in experiments.
<table><tr><td>Dataset</td><td>Modality</td><td>Task</td><td># of TrainSet</td><td># of Testset</td><td># of Classes</td><td>Imbalance Ratio</td></tr><tr><td>CIFAR10-LT</td><td>RGB</td><td>Image Recognition</td><td>50,000</td><td>10,000</td><td>10</td><td>50</td></tr><tr><td>PathMNIST-LT</td><td>Colon Pathology</td><td>Med-Image Classification</td><td>89,996</td><td>7,180</td><td>9</td><td>50</td></tr><tr><td>OCTMNIST-LT</td><td>Retinal OCT</td><td>Med-Image Classification</td><td>97,477</td><td>1,000</td><td>4</td><td>50</td></tr></table>

Clients train on their private data for a fixed number of local steps, then upload updates to the server for synchronous aggregation.

• FedProx [44]: Extends FedAvg by adding a proximal (quadratic) regularization term to each client’s local objective, penalizing deviation from the current global model to mitigate client drift under non-IID data; it also accommodates variable amounts of local work and partial participation.

• BalanceFL [20]: Using resampling and model distillation techniques to solve label imbalance. Re-sampling strategy incorporates class-frequency–aware reweighting into the training and aggregation pipeline (e.g., balanced sampling/losses and balanced aggregation of client updates) to counter majority-class dominance.

• CReFF [23]: employs a classifier retraining approach that generates class-wise synthetic features comparable to real features in a privacy-preserving manner. Server retrains the classifier using these synthetic features, enabling effective classifier recalibration to mitigate long-tailed distribution bias. .

• FedGraB [45]: Proposes gradient- and parameter-level rebalancing with regularization to dampen noisy or biased client updates; includes mechanisms that emphasize reliable contributions (e.g., by magnitude/consistency) while suppressing outliers caused by heterogeneity.

• FedLoGe [24]: simultaneously optimizing both local and global models. It uses a shared backbone for global trends while employing individualized classifiers for clientspecific features, achieving superior performance through representation learning and classifier alignment.

## C. Implementation Details

We use ResNet-8 as the backbone for all experiments. All methods are implemented in PyTorch and evaluated on a single NVIDIA GeForce RTX 4090 or two VGPU-48G GPUs. Federated learning is configured with 20 clients and 40% client participation per round, running for 200 rounds on CIFAR10- LT and PathMNIST-LT, and 100 rounds on OCTMNIST-LT. For a fair comparison, all compared FL methods use the same model and optimizer. Each selected client trains locally for 5 epochs with a batch size of 32 for CIFAR10-LT, 1024 for PathMNIST-LT, and 64 for OCTMNIST-LT. and the server aggregates updates using FedAvg. On the server side, we initialize the number of global federated features per class to 100. We use SGD with learning rate 0.1 for all optimization. Unless otherwise noted, the global federated feature coefficient is set to λ = 0.01 (CIFAR10-LT), λ = 0.001 (PathMNIST-LT), and λ = 0.0001 (OCTMNIST-LT). For SimCLR pretraining, the learning rates are $1 \times 1 0 ^ { - 3 }$ (CIFAR10-LT), $1 \times 1 0 ^ { - 4 }$ (PathMNIST-LT), and $1 \times 1 0 ^ { - 5 }$ (OCTMNIST-LT), with 10, 2, and 2 epochs, respectively.

## VI. EXPERIMENTS

## A. Main Results

Table III presents comprehensive performance comparisons across three datasets under varying test distribution scenarios. The results validate our central hypothesis: GTAFL achieves robust test-agnostic performance by trading minimal accuracy in predictable scenarios for substantial gains under distribution shifts.

1) Results on CIFAR10-LT: Averaged across all 11 test distributions, GTAFL achieves 75.02% accuracy, surpassing CReFF’s 73.69% accuracy and all other baselines by a clear margin, demonstrating consistent superiority regardless of test distribution type. When test distributions reverse training class frequencies, particularly in the Backward50 long-tail setting, GTAFL achieves 78.14% accuracy, representing a remarkable 76% improvement over FedAvg and a meaningful 6% gain over CReFF. This substantial improvement validates our approach’s effectiveness when deployment conditions fundamentally differ from training assumptions. In forward scenarios, GTAFL maintains competitive performance with CReFF while showing consistent improvements at moderate imbalance factors. At IF=50 and IF=25, GTAFL outperforms CReFF by 0.97% and 0.68% respectively, demonstrating stable performance across varying imbalance severities. Although GTAFL performs marginally below CReFF in uniform and some forward settings, the gap remains within 0.8%, confirming competitive accuracy while maintaining substantially superior backward robustness.

2) Medical Domain Validation: GTAFL similarly demonstrates average performance advantages in medical imaging datasets, supporting its clinical applicability. As shown in Table III, GTAFL achieves 83.87% at Forward50 on PathMNIST-LT, substantially outperforming all baselines, and attains nearly 2% improvement over the best competing method in backward distributions. Averaged across all 11 test distributions on PathMNIST-LT, GTAFL achieves 77.71%, confirming consistent superiority beyond individual distribution settings. On OCTMNIST-LT, while FedAvg and FedProx achieve marginally higher Forward50 accuracy, however, this advantage is misleading, as these methods simply assign test samples to the training majority classes, inflating forward accuracy while collapsing under backward distributions. By contrast, GTAFL maintains robust performance across the full distribution spectrum, achieving 74.12% at Backward50, a 26.18% improvement over FedAvg. These results demonstrate that GTAFL’s test-agnostic design is particularly valuable for real-world federated healthcare applications where deployment distributions are unpredictable.

TABLE III: Performance comparison of federated learning methods on CIFAR10-LT, PathMNIST-LT, and OCTMNIST-LT across various test distributions.
<table><tr><td colspan="10">CIFAR10-LT</td></tr><tr><td>Method</td><td colspan="4">Forward-LT</td><td>Uni.</td><td colspan="5">Backward-LT</td><td rowspan="2">Avg.</td></tr><tr><td></td><td>50</td><td>25</td><td>10</td><td>5</td><td>1</td><td>2</td><td>5</td><td>10</td><td>25</td><td>50</td></tr><tr><td>FedAvg</td><td>42.15</td><td>37.57</td><td>31.61</td><td>26.55</td><td>20.20</td><td>15.75</td><td>11.77</td><td>7.48</td><td>5.17</td><td>2.97</td><td>1.93</td></tr><tr><td>FedProx</td><td>38.21</td><td>34.90</td><td>29.73</td><td>25.41</td><td>19.50</td><td>15.29</td><td>11.34 7.28</td><td>4.95</td><td>2.85</td><td>1.82</td><td>18.47 17.39</td></tr><tr><td>BalanceFL</td><td>39.75</td><td>36.02</td><td>30.17</td><td>25.57</td><td>19.97</td><td>19.00</td><td>18.35 18.40</td><td>20.27</td><td>22.33</td><td>23.58</td><td>24.86</td></tr><tr><td>CReFF</td><td>79.64</td><td>78.26</td><td>75.24</td><td>73.61</td><td>71.99</td><td>71.60</td><td>71.52 72.05</td><td>72.38</td><td>71.91</td><td>72.34</td><td>73.69</td></tr><tr><td>FedGraB</td><td>73.20</td><td>69.06</td><td>62.56</td><td>56.43</td><td>47.77</td><td>41.07</td><td>34.80</td><td>27.30 22.18</td><td>16.23</td><td>12.88</td><td>42.13</td></tr><tr><td>FedLoGe</td><td>42.58</td><td>37.66</td><td>30.53</td><td>25.17</td><td>18.17</td><td>13.44</td><td>9.50 5.67</td><td>3.62</td><td>1.77</td><td>1.14</td><td>17.20</td></tr><tr><td>Ours</td><td>80.61</td><td>78.94</td><td>75.20</td><td>73.47</td><td>71.19</td><td>70.80 72.09</td><td>74.08</td><td>75.10</td><td>75.57</td><td>78.14</td><td>75.02</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>PathMNIST-LT</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Method</td><td colspan="7">Forward-LT</td><td colspan="4"></td></tr><tr><td>50 25</td><td></td><td>10</td><td>5</td><td></td><td>Uni.</td><td>5</td><td>Backward-LT 10</td><td></td><td></td><td>Avg.</td></tr><tr><td>FedAvg</td><td>50.35</td><td>46.67</td><td>41.60</td><td>36.94</td><td>31.45 26.45</td><td>2 25.69</td><td>24.58</td><td>23.28</td><td>25 21.17</td><td>50 19.72</td><td>31.63</td></tr><tr><td>FedProx</td><td>44.32</td><td>42.14</td><td>37.52</td><td>35.85</td><td>32.76 29.63</td><td>25.78</td><td>21.08</td><td>19.44</td><td>17.74</td><td>16.36</td><td>29.33</td></tr><tr><td>BalanceFL CReFF</td><td>32.71 78.19</td><td>33.77</td><td>34.48</td><td>34.75 35.66 75.63</td><td>34.94 74.57</td><td>33.35</td><td>30.24</td><td>30.80</td><td>33.77</td><td>35.61</td><td>32.74 76.13</td></tr><tr><td>FedGraB</td><td>21.69</td><td>77.62 22.48</td><td>76.56 24.24</td><td>74.44 24.89 25.69</td><td>25.40</td><td>74.54 24.88</td><td>75.63 22.89</td><td>76.24 20.80</td><td>76.81 17.94</td><td>77.26 16.36</td><td>22.48</td></tr><tr><td>FedLoGe</td><td>64.85</td><td>64.48</td><td>52.88</td><td>47.32</td><td>38.20 31.69</td><td>25.60</td><td>18.96</td><td>14.56</td><td>10.28</td><td>7.31</td><td>34.19</td></tr><tr><td>Ours</td><td>83.87</td><td>81.75</td><td>79.76</td><td>76.85</td><td>76.21</td><td>72.96 73.99</td><td>76.02</td><td>76.32</td><td>77.92</td><td>79.12</td><td>77.71</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>OCTMNIST-LT</td><td></td><td colspan="5"></td></tr><tr><td rowspan="2">Method</td><td colspan="7">Forward-LT</td><td colspan="5">Backward-LT</td></tr><tr><td>50</td><td>25</td><td>10</td><td>5</td><td>2</td><td>Uni. 1</td><td>2</td><td>5</td><td>10</td><td>25</td><td>Avg. 50</td></tr><tr><td>FedAvg</td><td>73.82</td><td>66.84</td><td>56.30</td><td></td><td>34.52</td><td>25.20</td><td>21.78</td><td>16.20</td><td></td><td></td><td>33.38</td></tr><tr><td>FedProx</td><td>75.29</td><td></td><td></td><td>47.08</td><td>38.77</td><td>29.50</td><td>21.51</td><td>12.16</td><td>8.02</td><td>5.29 5.29</td><td>35.21</td></tr><tr><td></td><td>75.29</td><td>69.25</td><td>60.81</td><td>51.22</td><td></td><td></td><td>16.01</td><td>11.94</td><td>7.75</td><td></td><td></td></tr><tr><td>BalanceFL</td><td>38.53</td><td>68.45</td><td>58.78</td><td>50.28</td><td>37.53</td><td>28.50</td><td>35.21 45.95</td><td>54.95 41.89</td><td>64.97 45.45</td><td>71.47 47.94</td><td>53.76 39.68</td></tr><tr><td>CReFF</td><td>73.53</td><td>39.04</td><td>39.19</td><td>37.10</td><td>36.58</td><td>35.30</td><td>35.34</td><td>40.11 12.61</td><td></td><td>7.06</td><td>33.68</td></tr><tr><td>FedGraB FedLoGe</td><td>73.53</td><td>66.84</td><td>56.31</td><td>47.08</td><td>34.66</td><td>25.80</td><td>21.51</td><td>16.01</td><td>9.09</td><td></td><td></td></tr><tr><td></td><td></td><td>66.84</td><td>56.31</td><td>47.08</td><td>34.25</td><td>25.00</td><td>17.12</td><td>9.42</td><td>5.63 2.67</td><td>1.47</td><td>30.85</td></tr><tr><td>Ours</td><td>72.65</td><td>68.72</td><td>63.06</td><td>57.63</td><td>59.86</td><td>58.90</td><td>59.59</td><td>65.73</td><td>69.82</td><td>80.20 74.41</td><td>66.42</td></tr></table>

## B. Ablation Study

We systematically analyze each component’s contribution through controlled ablations on CIFAR10-LT across forward, uniform, and backward test scenarios. Table IV reveals the distinct roles of our four components: Classifier Retraining (CR), Expert Ensemble (EN), Contrastive Learning (CL), and Global Sampling (GS).

Classifier Retraining (CR) establishes the fundamental capability for test-agnostic generalization. Without CR, the combination of CL and GS achieves competitive forward performance but suffers 41% degradation under backward distributions. This confirms that self-supervised learning and global sampling alone cannot compensate for the distributional bias introduced by long-tailed training.

The most significant improvement beyond CR emerges when Contrastive Learning (CL) is integrated with CR and EN: Backward50 performance increases from 71.45% to 78.14%, a substantial 6.69% gain. This validates that CL effectively acquires distribution-invariant representations, which are critical for generalizing to unknown test priors. Global Sampling (GS) provides measurable but more modest benefits. When combined with CR and EN, GS contributes consistent improvements on backward distributions such as Backward10 and Backward50.

In summary, the ablation confirms that each component targets a distinct aspect of test-agnostic learning: CR addresses classifier bias through synthetic expert retraining, EN exploits complementary expert knowledge across distributional priors, CL strengthens feature robustness through self-supervised learning, and GS refines sampling balance to amplify the effectiveness of the other components. This systematic decomposition validates our architectural design for handling unknown test distributions in federated environments.

## C. In-depth Analysis

1) Class-Wise Analysis: Table V presents a comprehensive class-wise performance breakdown on CIFAR10-LT, revealing how GTAFL adapts across different class categories under varying test priors. This granular analysis provides critical insights into the method’s behavior when handling head, middle, and tail classes under distribution shifts.

TABLE IV: Ablation study evaluating individual and combined effects of GTAFL components on CIFAR10-LT. Components: CR = Classifier Retraining, GS = Global Sampling, CL = Contrastive Learning, EN = Ensemble.
<table><tr><td colspan="10">CIFAR10-LT Ablation Study</td></tr><tr><td></td><td colspan="3">Components</td><td colspan="3">Forward-LT</td><td rowspan="2">Uni.</td><td colspan="3">Backward-LT</td><td rowspan="2">Avg.</td></tr><tr><td>CR</td><td>EN</td><td>CL</td><td>GS</td><td>50</td><td>25</td><td>10</td><td>1</td><td>25</td><td></td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>83.72</td><td>79.28</td><td>77.42</td><td>66.32</td><td>54.92</td><td>47.20</td><td>42.29</td><td>64.45</td></tr><tr><td>√</td><td></td><td></td><td></td><td>79.64</td><td>78.26</td><td>75.24</td><td>71.60</td><td>72.38</td><td>71.91</td><td>72.34</td><td>74.48</td></tr><tr><td>√</td><td></td><td></td><td>V</td><td>79.25</td><td>77.98</td><td>75.51</td><td>70.56</td><td>71.35</td><td>71.14</td><td>72.02</td><td>73.97</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>79.64</td><td>78.35</td><td>75.76</td><td>70.54</td><td>71.23</td><td>70.73</td><td>71.45</td><td>73.96</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>81.22</td><td>79.03</td><td>76.22</td><td>71.51</td><td>74.71</td><td>76.18</td><td>77.50</td><td>76.62</td></tr><tr><td>√</td><td>V</td><td>√</td><td>V</td><td>80.61</td><td>78.94</td><td>75.20</td><td>70.80</td><td>75.10</td><td>75.57</td><td>78.14</td><td>76.34</td></tr></table>

TABLE V: Top-1 test accuracy (%) achieved by compared FL methods and CReFF on CIFAR10-LT Forward 50, Uniform, and Backward 50 settings.
<table><tr><td>CIFAR10-LT</td><td colspan="4">Forward 50</td><td colspan="4">Uni.</td><td colspan="4">Backward 50</td></tr><tr><td>Method</td><td>All</td><td>Head</td><td>Middle</td><td>Tail</td><td>All</td><td>Head</td><td>Middle</td><td>Tail</td><td>All</td><td>Head</td><td>Middle</td><td>Tail</td></tr><tr><td>FedAvg</td><td>42.15</td><td>56.83</td><td>0.53</td><td>0.00</td><td>15.75</td><td>52.50</td><td>0.00</td><td>0.00</td><td>1.93</td><td>55.67</td><td>0.00</td><td>0.00</td></tr><tr><td>FedProx</td><td>38.21</td><td>51.69</td><td>0.00</td><td>0.00</td><td>15.29</td><td>50.97</td><td>0.00</td><td>0.00</td><td>1.82</td><td>52.58</td><td>0.00</td><td>0.00</td></tr><tr><td>BalanceFL</td><td>39.75</td><td>53.73</td><td>0.00</td><td>0.59</td><td>19.00</td><td>31.63</td><td>9.07</td><td>16.98</td><td>23.58</td><td>19.59</td><td>0.00</td><td>27.39</td></tr><tr><td>CReFF</td><td>79.64</td><td>86.11</td><td>62.79</td><td>56.47</td><td>71.60</td><td>78.23</td><td>63.50</td><td>72.70</td><td>72.34</td><td>75.26</td><td>57.06</td><td>74.58</td></tr><tr><td>FedGraB</td><td>73.20</td><td>87.08</td><td>40.79</td><td>11.18</td><td>41.07</td><td>78.57</td><td>40.33</td><td>13.50</td><td>12.88</td><td>72.16</td><td>39.34</td><td>6.33</td></tr><tr><td>FedLoGe</td><td>42.58</td><td>57.55</td><td>0.18</td><td>0.00</td><td>13.44</td><td>44.73</td><td>0.07</td><td>0.00</td><td>0.14</td><td>32.99</td><td>0.00</td><td>0.00</td></tr><tr><td>Ours</td><td>80.61</td><td>86.93</td><td>65.65</td><td>52.94</td><td>70.8</td><td>75.23</td><td>59.83</td><td>75.70</td><td>78.14</td><td>59.79</td><td>58.73</td><td>81.90</td></tr></table>

The most compelling finding emerges under backward conditions. GTAFL achieves 81.90% tail-class accuracy in the backward setting, confirming that the model has successfully reallocated its representational capacity toward classes that are prevalent at test time. However, under test-agnostic condition, other models that maintain high accuracy on trainingtime head classes is failing to adapt to the new distribution. For example, CReFF produces higher head-class accuracy in backward50, this comes at the cost of overall performance, achieving only 72.34% overall versus GTAFL’s 78.14% and substantially lower tail-class accuracy. Furthermore, in clinical contexts such as epidemic outbreaks, identifying the newly dominant pathology is far more critical than maintaining precision on now-comparatively-rare conditions.

Under forward test conditions, GTAFL achieves competitive or superior performance across all class categories. Notably, GTAFL surpasses CReFF on both head classes and middle classes, confirming that head-class reduction is not a systematic limitation. In uniform test scenarios, GTAFL delivers strong tail-class and head-class performance. Both performance validates GTAFL’s ability to redistribute classification capacity effectively when test distribution is agnostic.

2) Robustness to Limited Training Data: To evaluate GTAFL’s performance under data scarcity conditions prevalent in medical imaging scenarios, we conduct experiments with varying training data fractions (100%, 50%, 20%, 10%) on CIFAR10-LT. Figure 5 presents the comparative analysis between GTAFL and CReFF across different data availability conditions.

![](images/704e3485bc6b106640fc3625141f1a40ec05875bd2d2adbccf782bcb87dbfca5.jpg)

![](images/f392f5f3260675dec7e1a1d147d68de89b1ce3d71db407ee3f3cc17811060179.jpg)

![](images/0d93081a140bb3c41b306486171972904c1c315bad07eac766e04b060fdceb5a.jpg)

![](images/6094ed0924b39de600661ad2fa34f6d7b7e4ca28b739edb276eae9f31af68368.jpg)  
Fig. 5: Performance comparison of GTAFL and CReFF on varying CIFAR10 data fractions. GTAFL demonstrates greater robustness as the data size decreases. (FW50 = Forward50, Uni. = Uniform, BW50 = Backward50). When the data fraction reduces to 10% of original dataset, GTAFL outperforms CReFF by nearly 5%. For more details, please refer to VI-C.2.

The results show that GTAFL consistently outperforms CReFF on backward distributions across all data fractions. For instance, at 10% data, GTAFL achieves 50.70% versus CReFF’s 44.94% on Backward50, demonstrating robust testagnostic generalization even under severe data scarcity. This advantage stems from GTAFL’s distribution-specialized expert classifiers, which maintain meaningful distributional diversity even when trained on limited federated features.

Furthermore, as training data decreases, GTAFL’s advantage extends to uniform and forward distributions. At 50% data, GTAFL surpasses CReFF on forward scenarios by 4.32%, and at 10% data, it also leads on uniform accuracy by 2.63%. This suggests that GTAFL’s server-side self-supervised refinement provides increasingly competitive feature representations as labeled data becomes more scarce. This is valuable for realworld medical federated learning, where annotation costs are high and labeled samples are often limited.

TABLE VI: Computational overhead comparison between FedAvg, CReFF, and GTAFL on CIFAR10-LT. Client = Client Time, Server = Server Time, GFF Opt = GFF Optimization, CR = Classifier Retraining, CL = Contrastive Learning.
<table><tr><td>Method</td><td>Client</td><td>Server</td><td>GFF Opt</td><td>CR</td><td>CL</td></tr><tr><td>FedAvg</td><td>1129s</td><td>43s</td><td></td><td></td><td>一</td></tr><tr><td>CReFF</td><td>1376s</td><td>597s</td><td>95s</td><td>502s</td><td></td></tr><tr><td>Ours</td><td>1712s</td><td>4687s</td><td>277s</td><td>1586s</td><td>2699s</td></tr></table>

TABLE VII: Communication overhead comparison over 200 training rounds on CIFAR10-LT.
<table><tr><td>Method</td><td>Uplink</td><td>Downlink</td><td>Extra vs. FedAvg</td></tr><tr><td>FedAvg</td><td>7.333 GB</td><td>7.333 GB</td><td></td></tr><tr><td>CReFF</td><td>7.423 GB</td><td>7.349 GB</td><td>105 MB</td></tr><tr><td>GTAFL (Ours)</td><td>7.602 GB</td><td>7.380 GB</td><td>316MB</td></tr></table>

3) Computational and Communication Overhead Analysis: To evaluate the practical feasibility of GTAFL, we compare its training and communication overhead against FedAvg and CReFF on CIFAR10-LT under the backward50 test distribution. As reported in Tables VI and VII. GTAFL’s client time represents a 24.4% increase over CReFF, attributable solely to the additional backward passes required for computing 3 classwise expert gradients. This is a lightweight operation making GTAFL practical for resource-constrained edge devices.

The total server time is higher than CReFF , driven primarily by contrastive learning refinement and extra classifier retraining. As shown in Table VII, GTAFL requires only 316 MB of additional communication compared to FedAvg over 200 communication rounds with 40% participants each round . This marginal cost arises from uploading class-wise gradients for three expert classifiers and downloading the retrained expert heads, both of which involve only the lightweight classifier layer rather than the full model.

In summary, GTAFL achieves its test-agnostic performance gains through a server-centric design that shields clients from additional complexity.

4) Ensemble Mechanism Analysis: We further investigate the ensemble strategy by comparing our default hard voting, soft voting via averaged softmax probabilities, and confidenceweighted voting proportional to each expert’s maximum predicted probability. As shown in Table VIII, all three strategies yield nearly identical performance across all test distributions, with maximum differences below 0.006. This negligible gap is theoretically expected: confidence-based weighting assumes that softmax probabilities are reliable expert-selection signals, but deep neural networks are known to produce poorly calibrated confidence scores under distribution shift [46], causing experts to assign high confidence to incorrect predictions on unseen distributions. These results confirm that uniform hard voting is sufficient for test-agnostic federated learning.

TABLE VIII: Comparison of ensemble strategies on CIFAR10- LT. FW = Forward, BW = Backward, Conf.-Weighted = Confidence-Weighted.
<table><tr><td>Strategy</td><td>FW50</td><td>FW25</td><td>Uni.</td><td>BW25</td><td>BW50</td></tr><tr><td>Hard Voting (Ours)</td><td>0.7456</td><td>0.7321</td><td>0.6749</td><td>0.7684</td><td>0.7660</td></tr><tr><td>Soft Voting</td><td>0.7452</td><td>0.7342</td><td>0.6755</td><td>0.7680</td><td>0.7713</td></tr><tr><td>Conf.-Weighted</td><td>0.7456</td><td>0.7349</td><td>0.6753</td><td>0.7685</td><td>0.7717</td></tr></table>

5) Sensitivity Analysis of GFF Imbalance:

![](images/08e591cd049bfecb7287f9b9b823d168d43b2c0598b755d648b4900c9979d110.jpg)  
Fig. 6: Sensitivity analysis of GFF Imbalance on CIFAR10- LT. A larger λ induces a more skewed GFF distribution across expert classifiers.

Figure 6 reports the sensitivity of λ, and it controls the distributional imbalance of the synthetic global federated features (GFF) used for expert classifier retraining. When λ is too small, all three experts degenerate into equivalent uniform classifiers, losing distribution-specific specialization. When λ is too large, such as λ = 0.1, overly skewed synthetic features introduce distributional mismatch, degrading forward and uniform performance. The value $\lambda = 0 . 0 0 5$ achieves the best overall balance. Across $\lambda \in \ [ 0 . 0 0 1 , 0 . 0 1 ]$ , performance variations remain below 2%, confirming that GTAFL is robust to moderate perturbations of this parameter.

## VII. RELATED WORK

## A. Long-tail distribution FL

Long-tail learning has recently attracted significant attention due to the prevalence of imbalanced data in real-world applications. In the context of federated learning, the long-tail problem becomes even more pronounced as it is exacerbated by data heterogeneity across local clients. Several solutions, such as re-sampling, representation learning, and classifier re-training, which are commonly employed in centralized learning, can also be adapted in federated setting. Re-sampling strategies assign different training weights to classes, oversample minority classes, and under-sample majority classes based on their local frequencies to achieve a balanced training process [20], [47]. Recent methods often decouple the training process into two stages: classifier re-training and representation learning. During the classifier re-training phase, techniques such as post-calibration [29], [48], [49] are applied to address class imbalance. In the federated learning scenario, CReFF [23] introduces federated features, a form of synthetic data, to re-train biased classifiers. Some studies have observed that balancing the feature space is critical for addressing long-tail problems. For instance, RUCR [19] employs global unified prototypes to represent class-wise characteristics and adjust the distorted feature space. Meanwhile, Moon [39] leverages contrastive learning to align the current local model with the global model while distancing it from previous local models. However, most existing methods are designed under the assumption of a uniform test distribution, which may not hold in real medical scenarios.

## B. Test-agnostic learning

Test-agnostic long-tailed tasks have gained significant attention in centralized training. Existing methods address varying test distributions by adjusting loss functions or training multiple experts tailored to different scenarios [50], [51]. For example, LADE [25] uses prior knowledge of test distributions for post-adjustment, though this is impractical in real-world settings. SADE [26] trains experts with specialized loss functions and ensembles them using unsupervised learning strategies. While effective in centralized settings, these approaches face challenges in federated learning. Local clients often lack the computational resources to train multiple models, and data heterogeneity can result in missing classes, leading to degraded global performance. Building on the idea of training and ensembling experts, GTAFL retrains expert classifiers on the server and ensembles them during inference, avoiding the need for resource-intensive training on local clients. Local models are trained following standard FedAvg [3], overcoming the limitations of centralized strategies in federated environments. While prior test -agnostic federated learning work has focused on challenges like fairness [35], to the best of our knowledge, we are the first to identify and formulate the test-agnostic long-tailed federated learning, which presents fundamentally different challenges from existing problem formulations.

## VIII. DISCUSSION

## A. Privacy and Security Considerations

GTAFL operates within the standard federated learning paradigm, requiring exchanges of model parameters and gradients between clients and server. While raw data remains localized, such updates can leak information under sophisticated adversaries. Previous work has demonstrated that gradient and model inversion attacks may partially reconstruct sensitive training data [52]–[55], which is particularly concerning in healthcare applications. Our framework is compatible with privacy-preserving mechanisms to mitigate these risks. For example, client-side differential privacy (DP) [4], [56], [57] can be applied to expert gradients and model updates via per-example clipping and calibrated noise, providing formal privacy guarantees with controllable utility-privacy trade-offs.

## B. Performance Analysis and Trade-offs

GTAFL delivers substantial gains when test distribution differs from training setting, while maintaining competitive accuracy in other settings. This aligns with our design objective of test-agnostic generalization. However, a current limitation is relatively modest accuracy on majority classes under backward distribution conditions, reflecting an inherent trade-off when optimizing for substantial shifts in class priors. In clinical practice, this trade-off may be acceptable when deployment scenarios emphasize previously rare pathologies, such as emerging diseases or underdiagnosed conditions gaining recognition. Nevertheless, improving head-class fidelity while preserving tail-class robustness remains an important optimization target for comprehensive clinical deployment.

## C. Future Directions and Clinical Translation

Future work will focus on three directions. First, privacyby-design instantiations: integrating DP with adaptive noise schedules for expert gradients, combined with secure aggregation, to obtain end-to-end privacy guarantees tailored to classwise expert updates. Second, we aim to improve the efficiency of contrastive learning refinement by incorporating more computationally efficient contrastive objectives [58], [59], reducing training overhead while preserving the representation quality needed for robust feature learning. Third, we find that excessively large λ values can cause GFF gradient matching to degrade due to extremely severe mismatch, leading to reversed expert specialization. Future work will investigate more robust GFF optimization strategies and adaptive ensemble mechanisms that can automatically detect and down-weight degenerated expert classifiers at test time, improving reliability under suboptimal hyperparameter configurations. We plan to evaluate GTAFL under broader medical datasets and crosssite acquisition variability, and report privacy-utility trade-offs across different DP budgets to support clinical adoption.

## IX. CONCLUSION

We introduce Test-Agnostic Long-Tailed Federated Learning and propose GTAFL to address distribution shifts through expert classifiers ensemble. GTAFL generates diverse global federated features for privacy-preserving expert retraining, demonstrating robust performance across distribution shifts without serious degradation on reverse or imbalanced distributions. Extensive experiments show GTAFL outperforms state-of-the-art long-tailed federated learning methods, validating the framework’s effectiveness for healthcare applications where both privacy preservation and robustness to distribution shifts are critical.

## REFERENCES

[1] B. Jiang, J. Li, G. Yue, and H. Song, “Differential privacy for industrial internet of things: Opportunities, applications, and challenges,” IEEE Internet of Things Journal, vol. 8, no. 13, pp. 10 430–10 451, 2021.

[2] Z. Lian, Q. Yang, W. Wang, Q. Zeng, M. Alazab, H. Zhao, and C. Su, “Deep-fel: Decentralized, efficient and privacy-enhanced federated edge learning for healthcare cyber physical systems,” IEEE Transactions on Network Science and Engineering, vol. 9, no. 5, pp. 3558–3569, 2022.

[3] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. PMLR, 2017, pp. 1273– 1282.

[4] Q. Xie, S. Jiang, L. Jiang, Y. Huang, Z. Zhao, S. Khan, W. Dai, Z. Liu, and K. Wu, “Efficiency optimization techniques in privacy-preserving federated learning with homomorphic encryption: A brief survey,” IEEE Internet of Things Journal, vol. 11, no. 14, pp. 24 569–24 580, 2024.

[5] S. Jiang, H. Yang, Q. Xie, C. Ma, S. Wang, Z. Liu, T. Xiang, and G. Xing, “Towards compute-efficient byzantine-robust federated learning with fully homomorphic encryption,” Nature Machine Intelligence, vol. 7, no. 10, pp. 1657–1668, 2025.

[6] Z. Lian, Q. Zeng, W. Wang, T. R. Gadekallu, and C. Su, “Blockchainbased two-stage federated learning with non-iid data in iomt system,” IEEE Transactions on Computational Social Systems, vol. 10, no. 4, pp. 1701–1710, 2022.

[7] M. Jiang, Z. Wang, and Q. Dou, “Harmofl: Harmonizing local and global drifts in federated learning on heterogeneous medical images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 1, 2022, pp. 1087–1095.

[8] Q. Dou, T. Y. So, M. Jiang, Q. Liu, V. Vardhanabhuti, G. Kaissis, Z. Li, W. Si, H. H. Lee, K. Yu et al., “Federated deep learning for detecting covid-19 lung abnormalities in ct: a privacy-preserving multinational validation study,” NPJ digital medicine, vol. 4, no. 1, p. 60, 2021.

[9] T. R. Gadekallu, M. Alazab, J. Hemanth, and W. Wang, “Guest editorial federated learning for privacy preservation of healthcare data in internet of medical things and patient monitoring,” IEEE journal of biomedical and health informatics, vol. 27, no. 2, pp. 648–651, 2023.

[10] M. Jiang, H. Yang, X. Li, Q. Liu, P.-A. Heng, and Q. Dou, “Dynamic bank learning for semi-supervised federated image diagnosis with class imbalance,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2022, pp. 196–206.

[11] M. Ye, Y. Huangfu, S. Gao, W. Ren, W. Liu, and Z. Yu, “Fedgsca: Medical federated learning with global sample selector and client adaptive adjuster under label noise,” IEEE Journal of Biomedical and Health Informatics, 2026.

[12] J. Cai, Y. Wang, and J.-N. Hwang, “Ace: Ally complementary experts for solving long-tailed recognition in one-shot,” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 112–121, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID: 236924344

[13] Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, “Deep long-tailed learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, pp. 10 795–10 816, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:238583240

[14] S. Hou, X. Shang, S. N. Gowda, Y. Lu, C. Wu, Y. Yan, and H. Wang, “Capt: Class-aware prompt tuning for federated long-tailed learning with vision-language model,” ArXiv, vol. abs/2503.06993, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:276903333

[15] T. Li, Y. Huang, L. Jiang, Q. Xie, C. Liu, W. Du, L. Wang, and K. Wu, “Fedwcm: Unleashing the potential of momentum-based federated learning in long-tailed scenarios,” in Proceedings of the 54th International Conference on Parallel Processing, 2025, pp. 73–82.

[16] X. Lu, P. Li, and X. Jiang, “FedLF: Adaptive logit adjustment and feature optimization in federated long-tailed learning,” in Proceedings of the 16th Asian Conference on Machine Learning, ser. Proceedings of Machine Learning Research, V. Nguyen and H.-T. Lin, Eds., vol. 260. PMLR, 05–08 Dec 2025, pp. 303–318. [Online]. Available: https://proceedings.mlr.press/v260/lu25a.html

[17] B. Jiang, Y. Shang, G. Yue, H. H. Wang, and H. H. Song, “Long-tailed federated learning in internet of medical things based on ensemble distillation and imbalanced calibration,” IEEE Transactions on Consumer Electronics, pp. 1–1, 2025.

[18] L. Wang, S. Xu, X. Wang, and Q. Zhu, “Addressing class imbalance in federated learning,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 11, 2021, pp. 10 165–10 173.

[19] W. Huang, Y. Liu, M. Ye, J. Chen, and B. Du, “Federated learning with long-tailed data via representation unification and classifier rectification,” IEEE Transactions on Information Forensics and Security, 2024.

[20] X. Shuai, Y. Shen, S. Jiang, Z. Zhao, Z. Yan, and G. Xing, “Balancefl: Addressing class imbalance in long-tail federated learning,” in 2022 21st ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN). IEEE, 2022, pp. 271–284.

[21] Z. Zhu, Y. Shi, P. Fan, C. Peng, and K. B. Letaief, “Isfl: Federated learning for non-i.i.d. data with local importance sampling,” IEEE Internet of Things Journal, vol. 11, no. 16, pp. 27 448–27 462, 2024.

[22] N. Kang, Y. Lim, and J. Im, “Overcoming data imbalance in federated learning with calibration weighting,” in 2024 IEEE International Conference on Big Data (BigData), 2024, pp. 7832–7841.

[23] X. Shang, Y. Lu, G. Huang, and H. Wang, “Federated learning on heterogeneous and long-tailed data via classifier re-training with federated features,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, 7 2022, pp. 2218–2224, main Track. [Online]. Available: https: //doi.org/10.24963/ijcai.2022/308

[24] Z. Xiao, Z. Chen, L. Liu, Y. Feng, J. T. Zhou, J. Wu, W. Liu, H. Yang, and Z. Liu, “Fedloge: Joint local and generic federated learning under long-tailed data,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 50 357–50 376.

[25] Y. Hong, S. Han, K. Choi, S. Seo, B. Kim, and B. Chang, “Disentangling label distribution for long-tailed visual recognition,” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6622–6632, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:227239080

[26] Y. Zhang, B. Hooi, L. Hong, and J. Feng, “Self-supervised aggregation of diverse experts for test-agnostic long-tailed recognition,” in Neural Information Processing Systems, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:252684042

[27] B. Fan, S. Jiang, X. Su, S. Tarkoma, and P. Hui, “A survey on modelheterogeneous federated learning: Problems, methods, and prospects,” in 2024 IEEE International Conference on Big Data (BigData), 2024, pp. 7725–7734.

[28] S. Jiang, X. Shuai, and G. Xing, “Artfl: Exploiting data resolution in federated learning for dynamic runtime inference via multi-scale training,” in 2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN). IEEE, 2024, pp. 27–38.

[29] B. Kang, S. Xie, M. Rohrbach, Z. Yan, A. Gordo, J. Feng, and Y. Kalantidis, “Decoupling representation and classifier for long-tailed recognition,” in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/ forum?id=r1gRTCVFvB

[30] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597–1607.

[31] S. Jiang, W. Ding, H.-W. Chen, and M.-S. Chen, “Pgada: Perturbationguided adversarial alignment for few-shot learning under the supportquery shift,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2022, pp. 3–15.

[32] J.-M. Park, W.-J. Jang, T.-H. Oh, and S.-H. Lee, “Overcoming client data deficiency in federated learning by exploiting unlabeled data on the server,” IEEE Access, vol. 12, pp. 130 007–130 021, 2024.

[33] W. Jeong, J. Yoon, E. Yang, and S. J. Hwang, “Federated semisupervised learning with inter-client consistency & disjoint learning,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=ce6CFXBh30h

[34] T. Wei, Z. Mao, Z.-H. Zhou, Y. Wan, and M.-L. Zhang, “Learning label shift correction for test-agnostic long-tailed recognition,” in Forty-first International Conference on Machine Learning, 2024.

[35] Z. Yang, Q. Xu, Z. Wang, S. Li, B. Han, S. Bao, X. Cao, and Q. Huang, “Harnessing hierarchical label distribution variations in test agnostic long-tail recognition,” in Forty-first International Conference on Machine Learning, 2024.

[36] R. Ye, Y. Guo, X. Shuai, R. Ye, S. Jiang, and H. Jiang, “Licam: Long-tailed instance segmentation with real-time classification accuracy monitoring,” Journal of Circuits, Systems and Computers, vol. 32, no. 02, p. 2350032, 2023.

[37] S. Jiang, M. Yuan, X. Ji, B. Yang, Z. Liu, L. Xu, Y. Li, Y. He, L. Dong, W. Lu et al., “A large-scale multimodal dataset and benchmarks for human activity scene understanding and reasoning,” arXiv preprint arXiv:2512.07136, 2025.

[38] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, and C. Wu, “Federated learning with label distribution skew via logits calibration,” in International Conference on Machine Learning. PMLR, 2022, pp. 26 311– 26 329.

[39] Q. Li, B. He, and D. Song, “Model-contrastive federated learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 10 713–10 722.

[40] B. Zhao, K. R. Mopuri, and H. Bilen, “Dataset condensation with gradient matching,” in International Conference on Learning

Representations, 2021. [Online]. Available: https://openreview.net/ forum?id=mSAKhLYLSsl

[41] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble distillation for robust model fusion in federated learning,” Advances in neural information processing systems, vol. 33, pp. 2351–2363, 2020.

[42] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[43] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification,” Scientific Data, vol. 10, no. 1, p. 41, 2023.

[44] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” Proceedings of Machine learning and systems, vol. 2, pp. 429–450, 2020.

[45] Z. Xiao, Z. Chen, S. Liu, H. Wang, Y. Feng, J. Hao, J. T. Zhou, J. Wu, H. Yang, and Z. Liu, “Fed-grab: Federated long-tailed learning with selfadjusting gradient balancer,” Advances in Neural Information Processing Systems, vol. 36, pp. 77 745–77 757, 2023.

[46] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in International conference on machine learning. PMLR, 2017, pp. 1321–1330.

[47] K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, “Learning imbalanced datasets with label-distribution-aware margin loss,” Advances in neural information processing systems, vol. 32, 2019.

[48] T. Wang, Y. Li, B. Kang, J. Li, J. Liew, S. Tang, S. Hoi, and J. Feng, “The devil is in classification: A simple framework for long-tail instance segmentation,” in European conference on computer vision. Springer, 2020, pp. 728–744.

[49] Y. Li, T. Wang, B. Kang, S. Tang, C. Wang, J. Li, and J. Feng, “Overcoming classifier imbalance for long-tail object detection with balanced group softmax,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 10 991–11 000.

[50] C. Guo, W. Chen, A. Huang, and T. Zhao, “Prototype alignment with dedicated experts for test-agnostic long-tailed recognition,” IEEE Transactions on Multimedia, 2024.

[51] E. S. Aimar, A. Jonnarth, M. Felsberg, and M. Kuhlmann, “Balanced product of calibrated experts for long-tailed recognition,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19 967–19 977, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:254018198

[52] J. Geiping, H. Bauermeister, H. Droge, and M. Moeller, “Inverting¨ gradients-how easy is it to break privacy in federated learning?” Advances in neural information processing systems, vol. 33, pp. 16 937– 16 947, 2020.

[53] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and P. Molchanov, “See through gradients: Image batch recovery via gradinversion,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 16 337–16 346.

[54] G. Kaissis, A. Ziller, J. Passerat-Palmbach, T. Ryffel, D. Usynin, A. Trask, I. Lima Jr, J. Mancuso, F. Jungmann, M.-M. Steinborn et al., “End-to-end privacy preserving deep learning on multi-institutional medical imaging,” Nature Machine Intelligence, vol. 3, no. 6, pp. 473– 484, 2021.

[55] W. Wang, F. H. Memon, Z. Lian, Z. Yin, T. R. Gadekallu, Q.- V. Pham, K. Dev, and C. Su, “Secure-enhanced federated learning for ai-empowered electric vehicle energy prediction,” IEEE Consumer Electronics Magazine, vol. 12, no. 2, pp. 27–34, 2023.

[56] K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q. Quek, and H. V. Poor, “Federated learning with differential privacy: Algorithms and performance analysis,” IEEE transactions on information forensics and security, vol. 15, pp. 3454–3469, 2020.

[57] Z. Lian, W. Wang, H. Huang, and C. Su, “Layer-based communicationefficient federated learning with privacy preservation,” IEICE TRANS-ACTIONS on Information and Systems, vol. 105, no. 2, pp. 256–263, 2022.

[58] Y. Tan, G. Long, J. Ma, L. Liu, T. Zhou, and J. Jiang, “Federated learning from pre-trained models: A contrastive learning approach,” Advances in neural information processing systems, vol. 35, pp. 19 332–19 344, 2022.

[59] Q. Yu, Y. Liu, Y. Wang, K. Xu, and J. Liu, “Multimodal federated learning via contrastive representation ensemble,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=Hnk1WRMAYqg