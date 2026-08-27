# PASTA: Noisy Node Classification with Partial Label Learning

Yujing Liu, Yixin Liu, Yu Zheng, Yue Tan, Alan Wee-Chung Liew, Shirui Pan<sup>∗</sup> School of Information and Communication Technology, Griffith University, Brisbane, Australia {yujing.liu, yixin.liu, yu.zheng, yue.tan, a.liew, s.pan}@griffith.edu.au

Abstract—Noisy node classification problem is a fundamental yet challenging task for real-world graph-related web services, where node labels are often corrupted or unreliable due to weak supervision or automatic annotation. However, existing methods typically train models based on one-hot labels, which not only makes models susceptible to overfitting on noisy labels, but also leads to error accumulation after pseudo-label-guided enhancement. In this paper, we propose a novel Partial labelbased Self-training framework (PASTA for short) that leverages partial label learning technique to overcome the limitations of existing methods. Specifically, PASTA first trains multiple annotators to comprehensively capture the class distribution of nodes and aggregates their predictions to construct high-quality partial labels. Subsequently, we design a partial label-based classification model with two well-crafted loss functions to guide the model learning at both label and representation spaces. To further enhance the robustness against noisy labels, we introduce a self-training strategy where the labels refined by partial label learning are then used to further optimize the annotators in a closed-loop iterative manner. Extensive experiments on five datasets demonstrate that, compared with existing state-of-theart methods, PASTA achieves an average improvement of 1.1% in classification performance under various noise settings. The source code is at https://github.com/Yujingcn/PaSta-code.

Index Terms—Graph neural networks, noisy labels, partial labels, knowledge distillation

## I. INTRODUCTION

With the widespread application of graph-structured data in real-world web services such as social networks [1], recommendation systems [2], and financial networks [3], node classification has become a key technique for tasks like user profiling [4], automatic product categorization [5], and fraud detection [6]. However, real-world graph data inevitably contains erroneous labels (i.e., noisy labels) due to factors such as manual annotation errors or delays in information updates. Existing node classification methods are prone to overfitting these noisy labels in practice, resulting in a significant drop in generalization performance [7]. Consequently, the problem of noisy node classification (NNC) has gradually emerged as a central challenge for achieving reliable analysis and efficient utilization of graph-structured data in practical applications, attracting increasing research attention [8], [9].

To address the NNC problem, researchers have proposed various algorithms to equip graph neural networks (GNNs) with enhanced techniques that improve robustness against noisy labels, such as graph structure optimization, consistency regularization, and label correction [8]–[11]. Despite their effectiveness, the majority of current methods rely on onehot labels to supervise the classifier, which can lead to two inherent limitations. Limitation 1: Sensitivity to label noise. Driven by the cross-entropy loss, one-hot labels always force the classifier to assign high-confidence predictions to a single category. Nevertheless, when the labels are noisy, the model is easily misled to fit the wrong labels, which significantly undermines both the performance and robustness of models [10], [12]. Limitation 2: Difficulty in correcting label errors. To address the NNC problem, existing methods often refine noisy labels or assign pseudo-labels to unlabeled nodes. However, pseudo-labels or refined labels may still be unreliable, and the absolute certainty imposed by one-hot label-guided training can further reinforce incorrect supervision and even amplify error accumulation [8]. Even though several methods adopt soft labels [12], they are essentially probability-smoothed variants of one-hot labels (i.e., label smoothing), which still fail to capture the inherent uncertainty and ambiguity of labels.

![](images/69dce6808926784b0baebe78632da244adf2a46dc98028c66ed3d71749eeddac.jpg)  
Fig. 1: Sketch maps of noisy label and partial label learning. While noisy labels tend to drive the model toward confident but potentially incorrect predictions, partial labels help the model produce ambiguous yet more accurate predictions.

Going beyond one-hot label modeling, partial label learn ing [13], [14] provides a promising paradigm to alleviate the above limitations. As shown in Figure 1, partial label learning relaxes the rigid one-hot assumption by allowing each sample to be annotated with multiple candidate labels, hence providing an effective way to model uncertainty and ambiguity in supervision. In this case, as long as the correct label is included in the candidate set, the model is able to learn the correct decision boundary [15], [16], which substantially improves its robustness against noisy labels and thereby alleviates Limitation 1. On the other hand, if we convert the pseudo-labels and refined labels into partial labels, the inherent uncertainty can be explicitly preserved, which prevents the model from being constrained by potentially incorrect one-hot supervision. In this way, Limitation 2 can be mitigated by reducing the risk introduced by faulty label correction.

Although partial label learning has shown great potential in handling noisy labels, how to apply this advanced paradigm to the NNC problem still remains non-trivial. Specifically, two critical questions remain to be solved: Question 1: How to obtain reliable partial labels from graph data? Unlike in computer vision domains, where high-quality partial labels can often be naturally available [17], [18] or easily acquired from pre-trained models (e.g., CLIP [19]), it is difficult to construct reliable partial labels for real-world graph-structured data. Without high-quality partial labels, the classifier will be misled by vague and low-informative supervision signals, resulting in blurred decision boundaries and even model collapse. Hence, a robust annotation strategy is required to generate reliable partial label that are both complete and accurate. Even though accurate partial labels can be obtained, we have to face a followup challenge: Question 2: How to effectively leverage partial labels to learn a GNN-based classifier? Unlike one-hot labels that convey a probabilistic meaning with absolute certainty, partial labels characterize the possibility of multiple candidate classes and reflect the inherent ambiguity of supervision. In this case, directly applying the conventional cross-entropy loss for partial label learning may dilute the informative signal and ultimately hinder effective classifier training. Consequently, it is crucial to design learning objectives that can fully exploit the signals in partial labels.

In response to the above two questions, in this paper, we propose a Partial label-based Self-training framework (PASTA for short) to tackle the NNC problem. To answer Question 1, we introduce a self-supervised ensemble annotation module to generate high-quality partial labels automatically. Specifically, we construct multiple annotators using different self-supervised learning strategies to pre-train their feature extractors, followed by learnable predictors. In this way, the diverse annotators can assign labels to nodes from different perspectives, and by ensembling them, we can acquire partial labels with accuracy and completeness. To address Question 2, we develop a dual-space partial label learning module, where a node classifier is trained with two complementary loss functions to effectively exploit partial labels. At label space, we design a vote-aggregated Cross Entropy (VaCE) loss, which leverages the partial labels obtained by aggregating votes from multiple annotators to optimize the output class probabilities, thereby explicitly leveraging the consensus distribution formed by annotators. At the latent representation space, we further propose a partial label similarity (PaSim) loss, which constrains node representations based on the pairwise similarity of their partial labels. The PaSim loss encourages nodes with similar candidate label sets to be closer in the representation space, enhancing the model’s ability to discriminate under label uncertainty. Furthermore, to improve robustness against noisy supervision in NNC, we employ a self-training strategy that iteratively optimizes the annotation module and the partial label learning module in a bootstrapping manner. In each iteration, we enhance the labels for annotator training using the predictions from the partial label learning module; the refined partial labels, in turn, provide more reliable supervision to further improve the classifier. To sum up, the main contributions of our method are summarized as follows:

• New Paradigm. To the best of our knowledge, we make the first attempt to use partial labels to address the NNC problem, which provides a new learning paradigm that explicitly models label uncertainty and enhances robustness against label noise.

• Novel Method. We propose PASTA, a self-training framework for NNC, which consists of a self-supervised ensemble annotation module and a dual-space partial label learning module. The two modules are jointly optimized in an iterative manner to mutually enhance partial label annotation quality and classifier performance.

• Ample Experiments. We conduct extensive experiments on five real-world datasets, and the results show our method outperforms all existing SOTA methods across all noise levels.

## II. RELATED WORKS

## A. Graph Neural Networks

Graph Neural Networks (GNNs) have received widespread research attention due to their powerful capability in capturing structural topology and node attributes through recursive message-passing mechanisms. Consequently, GNNs are increasingly applied to a broad spectrum of downstream tasks, ranging from fundamental node classification [20], [21] and graph anomaly detection [22]–[24] to complex topology design [25]. Alongside this broadening scope of applications, advanced learning paradigms have developed rapidly to address real-world deployment challenges; for instance, federated graph learning frameworks [26], [27] manage data heterogeneity across distributed clients, while robust learning techniques [10], [28] aim to mitigate noise interference during model training. Despite these broad successes, standard GNN architectures remain inherently vulnerable to corrupted graph data, as bad signals can easily propagate and compound across neighboring nodes. As a result, achieving robust model performance on real-world noisy graph data remains a pressingly unresolved problem that warrants further investigation.

## B. Noisy Node Classification

GNNs are inherently sensitive to label noise because the recursive message-passing mechanism tends to propagate and amplify label errors across neighboring nodes, severely undermining model generalization [29], [30]. To address this, various robust GNN strategies have been proposed. Graph structure optimization methods, such as NRGNN [21], refine edge connections based on node similarity to ensure noisy nodes are supervised by reliable neighbors. Multi-teacher self-training frameworks (e.g.MTS-GNN [10] and BO-NNC [8]) utilize historical model states or distillation strategies to provide more stable supervision and guide label correction. Additionally, consistency-based models like CGCN [28] leverage data augmentation and contrastive constraints to encourage the model to learn noise-invariant representations. Despite these advancements, existing methods often struggle with overfitting to noisy signals during late-stage training, and ensuring the absolute reliability of label correction remains an open challenge that PASTA seeks to overcome.

## C. Partial Label Learning

Partial Label Learning (PLL) addresses scenarios where each instance is associated with a candidate label set containing the ground-truth [31]. Traditional PLL algorithms focus on progressive disambiguation to identify the true label from uncertain sets [17], [18]. Recently, PLL has been successfully adapted for noisy image classification (e.g., PALS [14] and NPN [16]), where noisy labels are treated as candidates and refined through iterative optimization. However, the application of PLL to the NNC problem remains unexplored, primarily due to the difficulty of constructing high-quality candidate sets from noisy graph data, a gap that our proposed PASTA framework effectively fills.

## III. PRELIMINARY

Notations. A graph can be represented by $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\mathcal { V } = \{ v _ { 1 } , \cdots , v _ { n } \}$ denotes the node set with n nodes, and $\mathcal { E } \subseteq$ $\nu \times \nu$ corresponds to the edge set that encodes their pairwise relations. Each node $v _ { i }$ is associated with a feature vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ , and assembling all node features yields the feature matrix $\mathbf { X } = \{ { \mathbf { x } } _ { 1 } , \cdot \cdot \cdot , { \mathbf { x } } _ { n } \} \in \mathbb { R } ^ { n \times d }$ . The structure of the graph is represented by an adjacency matrix $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ , where each entry $\mathbf { a } _ { i j } = 1 \mathrm { ~ i f ~ } ( v _ { i } , v _ { j } ) \in \mathcal { E }$ and $\mathbf { A } _ { i j } = 0$ otherwise.

Noisy node classification (NNC). In the NNC setting, only a small portion of nodes $\smash { \mathcal { V } _ { L } \subseteq \mathcal { V } }$ are provided with labels, denoted as $\mathbf { Y } = \{ \mathbf { y } _ { 1 } , \dots , \mathbf { y } _ { b } \}$ , where $b = | \nu _ { L } |$ | is the number of labeled nodes. Each label $\mathbf { y } _ { i } \in \mathbb { R } ^ { c }$ is expressed as a one-hot vector over c possible classes. Given $\mathcal { G } = ( \nu , \mathcal { E } )$ with feature matrix X, adjacency matrix A, and a noisy label set Y<sup>ˆ</sup> (in which some annotations may be erroneous, i.e., $\hat { \mathbf { y } } _ { i } \neq \mathbf { y } _ { i }$ for certain $v _ { i } \in \mathcal { V } _ { L } )$ , the task of NNC is to develop a model that can accurately predict the ground-truth class of each node by exploiting $\hat { \mathbf Y }$ as supervisory signals.

Partial label learning. In partial label learning, each node $v _ { i } ~ \in ~ \mathcal { V } _ { L }$ is associated not with a single label but with a candidate label set ${ { Y } _ { i } } ~ \in ~ \{ 0 , 1 \} ^ { C }$ , where the j-th element $Y _ { i j } ~ = ~ 1$ indicates that class j is a candidate label and $Y _ { i j } = 0$ otherwise. The ground-truth label $y _ { i }$ is guaranteed to be contained in $Y _ { i } ,$ i.e., $y _ { i } \in Y _ { i }$ , but its exact identity is unknown. The task of partial label learning is to develop a model that, given the candidate label set $Y _ { i }$ for each labeled node $v _ { i } \in \mathcal V _ { L }$ , can accurately infer the true class $y _ { i }$ , thereby learning classification patterns that generalize to all nodes $v \in \mathcal V$

## IV. METHODOLOGY

In this section, we introduce the proposed method, Partial label-based Self-training framework (PASTA), for noisy node classification (NNC). With its overall pipeline illustrated in Figure 2, PASTA is composed of two modules, namely selfsupervised ensemble annotation and dual-space partial label learning. In the self-supervised ensemble annotation module, our method employs multiple self-supervised graph techniques to build diverse partial label annotators, enabling them to capture node category information more comprehensively and thereby generate partial labels with high correct information quantity. Subsequently, in the dual-space partial label learning module, we design two novel loss functions to guide the classifier in learning from partial labels, which enables PASTA to exploit the supervisory information at both label space and representation space. To further refine the annotated partial labels and classifier, we incorporate a self-training strategy that iteratively updates the two modules in a bootstrapping manner. To achieve this, we design a label enhancement process for noisy label filtering and pseudo-label expansion, which polishes the label information for annotator training.

## A. Self-Supervised Ensemble Annotation

To build a robust partial label learning model, constructing reliable partial labels is the cornerstone of the entire framework. While acquiring high-quality partial labels in other domains, such as computer vision and natural language processing, is relatively easy through multi-class annotations or large pre-trained models (e.g., CLIP or BERT [32]), obtaining them in graph-structured data is far more challenging due to the absence of such annotation resources and the complexity of graph topology. In the context of NNC, the construction of partial labels becomes far more challenging since noisy labels can further contaminate the candidate label sets and degrade their overall quality. As a result, it is crucial to design a robust annotation strategy that can ensure the completeness and accuracy of partial labels.

To tackle this challenging task, we design a self-supervised ensemble annotation module, in which multiple annotators collaborate to generate partial labels together. For each annotator, we use a specialized self-supervised learning strategy to pre-train the feature extractor, which ensures diversity in the representation spaces among different annotators. Then, we train the predictors with enhanced labels (the enhancement process is detailed in Section IV-C), and the predictions from different annotators are finally ensembled to form high-quality consensus partial labels. These sub-components are introduced in the following paragraphs.

1) Self-Supervised Feature Extractors: In order to construct diverse annotation sources for high-quality partial label construction, we first adopt multiple self-supervised graph learning methods to train the feature extractors of the annotators. Formally, for the i-th annotator, the learned node embeddings are learned from different perspectives:

![](images/1981281008063e458e1ea524c8f77110f421c4923db35184b45b928dc68f1fd6.jpg)  
Fig. 2: The framework of the proposed PASTA. On the left, we leverage diverse self-supervised graph learning algorithms to construct annotators and generate consensus labels. On the right, we introduce $\mathcal { L } _ { p a s i m }$ and $\mathcal { L } _ { v a c e }$ to guide the classifier in learning robust patterns from the consensus labels. The trained classifier is then used to filter and expand dataset labels, which are subsequently used to further optimize the annotators’ predictors. Finally, the two modules are iteratively optimized.

$$
\mathbf { H } ^ { ( i ) } = \mathbf { F } _ { e n c } ^ { ( i ) } ( \mathbf { X } , \mathbf { A } ) , \quad \mathrm { w h e r e ~ } \mathbf { F } _ { e n c } ^ { ( i ) } = \arg \operatorname* { m i n } _ { \mathbf { F } } \mathcal { L } _ { s s l } ^ { ( i ) } .\tag{1}
$$

In this equation, $\mathbf { F } _ { e n c } ^ { ( i ) } ( \cdot )$ denotes the feature extractor of i-th annotator, $i \in [ 1 , . . . , t ]$ with t as the number of annotators, and $\mathcal { L } _ { s s l } ^ { ( i ) }$ is a specialized self-supervised loss function to optimize $\mathbf { F } _ { e n c } ^ { ( i ) } ( \cdot )$ . The candidates of $\bar { \mathcal { L } } _ { s s l } ^ { ( i ) }$ can be mainstream graph selfsupervised learning objectives, such as DGI [33], GCA [34], and SUGRL [35].

The design of multiple self-supervised feature extractors has several advantages. Firstly, since different graph selfsupervised learning methods capture distinct structural or semantic properties, they can project nodes into diverse representation spaces, thereby ensuring the completeness of the constructed partial labels [36]. Secondly, as self-supervised training does not rely on ground-truth labels, the learned extractors are immune to the influence of noisy labels, which satisfies the requirement of accuracy for partial labels. Thirdly, self-supervised extractors can be pre-trained in advance during model initialization, which improves the training efficiency of PASTA.

2) Supervised Predictors: After the pre-training of feature extractors, for each annotator, we train a light-weight predictor (i.e., a fully connected layer with Softmax activation) via supervised learning. Specifically, the predictor is trained by the cross-entropy loss with a labeled node set:

$$
\mathcal { L } _ { c e } ^ { ( i ) } = - \sum _ { v _ { j } \in S } \sum _ { k = 1 } ^ { c } \mathbb { I } ( y _ { j k } \neq 0 ) \log \mathbf { p } _ { j } ^ { ( i ) } ,\tag{2}
$$

where $s$ denotes the labeled training set, and $\mathbf { p } _ { j } ^ { ( i ) } \in \mathbb { R } ^ { c }$ and $\mathbf { y } _ { j }$ denote the prediction of the predictor $\mathbf { F } _ { p r e } ^ { ( i ) } ( \cdot )$ and the onehot label of node $v _ { j }$ , respectively. Here, the training set $s$ can be initialized as the original labeled set $\gamma _ { L }$ . As the model iteratively updates, S is progressively refined into enhanced labels (see Section IV-C).

Although different predictors are trained with the same supervision signals, their distinct underlying representation spaces still lead them to exhibit diverse feature extraction and classification behaviors. Consequently, different annotators can still exhibit diverse classification behaviors, thereby differentiating the impact of noisy labels on them and making their predictions complementary.

3) Ensembled into Consensus Partial Labels: Finally, the prediction results of these annotators are aggregated to form consensus partial labels. Concretely, the consensus label vector for node $v _ { i }$ is given by collecting the most confident category from each annotator:

$$
\tilde { \mathbf { y } } _ { i } = \left[ \tilde { y } _ { i 1 } , \ldots , \tilde { \mathbf { y } } _ { i c } \right] , \quad \mathrm { w h e r e ~ } \tilde { y } _ { i j } = \sum _ { k = 1 } ^ { t } \mathbb { I } \big ( \arg \operatorname* { m a x } \mathbf { p } _ { i } ^ { ( k ) } = j \big ) .\tag{3}
$$

It is worth noting that, in PASTA, we construct a welldesigned consensus partial labels, rather than the traditional binarized partial labels, because we argue that non-binarized consensus labels can carry richer supervisory signals. Firstly, non-binarized labels can more precisely reflect the positional relationships of nodes in the partial label space, thereby providing more detailed inter-node semantic information for the classifier during subsequent training. Secondly, non-binarized consensus labels effectively serve as adaptive sample reweighting, guiding the model to prioritize nodes on which the annotators agree, which helps constrain the model to learn correct classification information from the remaining consensus labels rather than fitting error labels.

## B. Dual-Space Partial Label Learning

Once we obtain the consensus partial labels, the following challenge is how to effectively leverage them to train a node classifier. While using a conventional training objective (e.g., cross-entropy) can provide straightforward supervision, it would treat partial labels as one-hot targets, leading to the loss of uncertainty information and the risk of reinforcing noisy signals. In order to fully exploit the informative consensus partial labels, in PASTA, we design two complementary loss functions for classifier training: the vote-aggregated crossentropy (VaCE) loss on the label space and the partial label similarity (PaSim) loss on the representation space.

1) Classification Model: In this module, we first initialize the parameters randomly to construct a two-layer GCN [29] as the classification model. Specifically, the node embeddings ${ \bf H } ^ { ( c l s ) }$ of all nodes and the prediction probability matrix $\mathbf { P } ^ { ( c l s ) }$ of the classification model are obtained by:

$$
\begin{array} { r l } & { \left\{ \mathbf { H } ^ { ( c l s ) } = \sigma ( \hat { \mathbf { A } } \mathbf { X } \mathbf { W } _ { e m b } ) , \right. } \\ & { \left. \mathbf { P } ^ { ( c l s ) } = \mathrm { s o f t m a x } ( \sigma ( \hat { \mathbf { A } } \mathbf { H } ^ { ( c l s ) } \mathbf { W } _ { p r d } ) ) , \right. } \end{array}\tag{4}
$$

where $\mathbf { W } _ { e m b }$ and $\mathbf { W } _ { p r d }$ are trainable parameters of the student model, $\hat { \bf A } = \tilde { \bf D } ^ { - \frac { 1 } { 2 } } \tilde { \bf A } \tilde { \bf D } ^ { - \frac { 1 } { 2 } }$ where $\tilde { \mathbf { A } } = \mathbf { A } + \mathbf { I } , \tilde { \mathbf { D } }$ is a diagonal matrix with $\begin{array} { r } { \tilde { d } _ { i i } \ = \ \sum _ { j } \tilde { a } _ { i j } } \end{array}$ and I is the identity matrix, $\sigma$ denotes the activation function.

Note that we adopt such a basic GNN as our classifier to clearly demonstrate the effectiveness of our proposed framework without the interference of complex architectures. In practice, the classification model can be alternatively replaced by any advanced GNNs for node classification, such as Graph-SAGE [37] or GAT [38].

2) Vote-Aggregated Cross-Entropy (VaCE) Loss: Different from one-hot labels or binary partial labels, the consensus partial labels in PASTA contain richer information about the class distribution, which is modeled by the annotators’ voting during label ensembling. To fully exploit the distributional information in consensus partial labels, it is crucial to design a classification loss that considers the voting behaviors. To this end, we design a Vote-aggregated Cross-Entropy (VaCE) loss $\mathcal { L } _ { v a c e } ,$ , an improvement over the conventional cross-entropy, to leverage the collective intelligence of the consensus labels for learning more robust decision patterns. The loss function is defined as:

$$
\mathcal { L } _ { v a c e } = - \sum _ { i \in \mathcal { T } } \sum _ { j = 1 } ^ { c } \mathbb { I } ( \tilde { y } _ { i j } \neq 0 ) \log \mathbf { p } _ { i j } ^ { ( c l s ) } \cdot \tilde { y } _ { i j } ,\tag{5}
$$

where $\tau$ denotes the training set. Different from the conventional cross-entropy loss, our VaCE loss explicitly accounts for the voting tendencies of annotators by assigning greater update strength to categories with higher vote counts. In this way, the model benefits from fine-grained supervision signals and avoids the overconfidence issue of one-hot training.

3) Partial label Similarity (PaSim) Loss: In the field of partial label learning, partial labels of the samples are typically constructed via manual annotation, rule-based generation, or random noise injection, where the non-target classes are often meaningless. Consequently, existing algorithms mainly focus on identifying the correct labels from the partial labels. However, the consensus labels constructed in PASTA not only contain the class information of nodes, but their non-target classes also carry rich semantic information (e.g., reflecting the node’s position near classification boundaries). To leverage this semantic information to improve model generalization, we design a novel loss function, referred to as the Partial label Similarity (PaSim) loss. Specifically, we first compute the consensus partial label similarity matrix $\mathbf { S } _ { l a b }$ and the embedding similarity matrix $\mathbf { S } _ { e m b } \colon$

$$
{ \bf S } _ { l a b } = \tilde { { \bf Y } } \cdot \tilde { { \bf Y } } ^ { \top } , ~ { \bf S } _ { e m b } = { \bf H } ^ { ( c l s ) } \cdot { \bf H } ^ { ( c l s ) \top } ,\tag{6}
$$

where <sup>⊤</sup> denotes the matrix transpose operation., and · represents matrix multiplication. By calculating the pair-wise similarity of partial labels, the consensus partial label similarity matrix $\mathbf { S } _ { l a b }$ captures the semantic affinity information of nodes in the partial label space. Such a semantic affinity can serve as a valuable supervision signal for guiding the embedding distribution in the representation space. Subsequently, we constrain the node distribution in the classifier’s representation space to be consistent with the label space through the following consistency constraint:

$$
\mathcal { L } _ { p a s i m } = \Vert \mathbf { S } _ { e m b } - \mathbf { S } _ { l a b } \Vert _ { 1 } = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \left. s _ { i j } ^ { e m b } - s _ { i j } ^ { l a b } \right. .\tag{7}
$$

Finally, by combining the VaCE loss and PaSim loss, we obtain the overall partial learning objective for the classification model:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { v a c e } + \lambda \mathcal { L } _ { p a s i m } , } \end{array}\tag{8}
$$

where λ is a hyperparameter that controls the weight of the PaSim loss. The two losses constrain the classifier in different spaces (i.e., the label space and the representation space) and leverage the class distribution and similarity information of consensus partial labels, and hence complement each other to enhance model robustness and discriminability.

## C. Iterative Self-Training Framework

Although the two modules in PASTA can realize the annotation and utilization of partial labels, if they are executed in a purely sequential manner, the overall effectiveness of PASTA may be undermined by noisy labels. Concretely, the lowquality labels from the original data may affect the annotation performance, leading to inaccurate partial labels. Then, the sub-optimal partial labels can further hinder the training of the classification model.

To alleviate the impact of original noisy labels and improve learning robustness, we employ an iterative self-training strategy [39] for PASTA, where the annotation module and the partial label learning module are jointly optimized in a closedloop and bootstrapping manner. In each iteration, we use the annotated partial labels to update the partial label learning module, and then use the predictions generated by the partial label learning module to further refine the annotators. Running in the loop with τ iterations, both modules are progressively refined and mutually enhanced, resulting in better partial labels and classification results.

In order to form a closed loop, a label enhancement strategy is needed to bridge the predictions of the classifier back to the annotator. In PASTA, we use a simple yet effective label enhancement strategy that consists of two steps: noisy label filtering and pseudo-label expansion. The label enhancement strategy focuses on the refinement of training node sets $s$ and the corresponding labels for annotator training, where $s$ is initialized as the original labeled node set $\mathcal { V } _ { L }$

1) Noisy Label Filtering: First, we filter out noisy labels in $s$ to reduce incorrect supervisory information. Specifically, after the classification model has converged after training with partial labels, we compute the loss set L of the labeled nodes for each class (e.g., the $j \mathrm { - t h }$ class) based on its predictions:

$$
{ \bf L } _ { j } = \{ l _ { i } \ | \ y _ { i } = j \} , \quad \mathrm { w h e r e } j \in [ 1 , . . . , c ] ,\tag{9}
$$

where $l _ { i }$ denotes the conventional cross-entropy loss of node $v _ { i }$ given class $j$

Next, following the small-loss strategy, nodes with higher loss are more likely to contain noisy labels [40], [41]. Therefore, we compute a noisy loss threshold using the following equation to select the nodes with the top r% highest loss values within each class:

$$
\varphi _ { i } = s o r t ( \mathbf { L } _ { i } ) _ { [ \sigma _ { i } r ] } ,\tag{10}
$$

where $\sigma _ { i }$ denotes the number of labelled nodes of class $i , \ r$ is a hyper-parameter and $\varphi _ { i }$ is the noisy loss threshold for class i. Finally, we regard nodes with loss values exceeding this threshold as noisy nodes and remove their labels. In subsequent training, these nodes are treated as unlabeled nodes. By executing the noisy label filtering, the unreliable nodes in the previous S will be removed.

2) Pseudo-Label Expansion: After filtering out noisy labels in the dataset, we increase the supervisory information by assigning pseudo-labels to the unlabeled nodes. Specifically, we first compute the predicted classes $y ^ { ( c l s ) }$ and confidence $\delta _ { i }$ of all unlabeled samples using the classification model:

$$
\begin{array} { r } { \left\{ { y } _ { i } ^ { ( c l s ) } = \arg \operatorname* { m a x } ( p _ { i } ^ { ( c l s ) } ) , \right. } \\ { \left. \delta _ { i } = \operatorname* { m a x } ( p _ { i } ^ { ( c l s ) } ) . \right. \qquad } \end{array}\tag{11}
$$

Subsequently, we assign pseudo labels to the top α (α is a hyper-parameter) nodes with the highest prediction confidence in each class. These pseudo labels are then added to $s$ to further optimize the annotators. To sum up, the pseudo-label expansion enables us to exploit high-confidence predictions to enrich the supervision signals and improve annotation quality.

After the two-step label enhancement process, the training node set S for the annotation module can be refined with cleaner and more informative supervision signals. Then, we can continue training the predictor of each annotator in the next iteration, which improves both annotation quality and then improves the classifier performance.

## D. Complexity Analysis

Time Complexity. The computational overhead of PaSta primarily stems from its iterative self-training process. Notably, the graph convolutional feature extractors of the annotators are pre-trained once and then frozen, which significantly reduces ongoing training costs. In each self-training iteration, the computational cost of PaSta is dominated by three components: (1) Forward propagation through t frozen GCN annotators to obtain node representations, costing $\mathrm { O } ( t \cdot n ^ { 2 } d )$ where $t ,$ n and $d$ are the number of annotators, nodes and feature dimensions, respectively; (2) Training t lightweight predictors (fully connected layers) on the labeled nodes, requiring $\operatorname { O } ( t \cdot { \mathcal { T } } \cdot d \cdot c )$ , where $\tau$ and c are the number of training nodes and classes, respectively; and (3) Training the main GCN classifier with dual-space losses, where the GCN forward pass and PaSim loss computation both cost $\mathrm { O } ( n ^ { 2 } d )$ and the VaCE loss adds $\mathrm { O } ( { \mathcal { T } } \cdot c )$ . Here, the graph convolution and similarity computation are the dominant factors in the overall complexity. As a result, the overall time complexity over $\tau$ iterations is $\operatorname { O } ( \tau \cdot t \cdot n ^ { 2 } d )$ . Obviously, it is quadratic to the node number $n ,$ differing from standard GNN training only in the constant factors.

Space Complexity. The space consumption of PASTA is dominated by model parameters and intermediate computation during dual-space optimization. The parameter storage includes t frozen GCN feature extractors $( O ( t \cdot d ^ { 2 } ) )$ , t lightweight predictors $( O ( t \cdot d \cdot c ) )$ , and the main GCN classifier $( O ( d ^ { 2 } ) )$ . Beyond parameters, storing node embeddings incurs $O ( n d )$ space, while the similarity matrices required for the PaSim loss dominate memory with $O ( n ^ { 2 } )$ complexity to capture pairwise semantic affinities. Overall, PASTA has a total space complexity of $O ( n ^ { 2 } + t \cdot d ^ { 2 } )$ , scaling quadratically with the number of nodes n due to dense similarity matrices and multiple GCN annotators.

## V. EXPERIMENTS

## A. Experimental Settings

1) Datasets: We evaluate the proposed method on five realworld graph datasets: three citation networks (Cora and Citeseer [42]), an academic collaboration network (DBLP [43]), and two product co-purchase network (Computers and Photo [44]). The detailed statistics of the datasets used in this work are provided in Appendix B.

2) Comparison Methods: The following comparison methods are considered for a comprehensive comparison: two classical GNN architectures (GCN [29] and GAT [38]), three representative self-supervised graph learning approaches (DGI [33], GCA [34], and SUGRL [35]), and four stateof-the-art noisy node classification methods (JoCoR [45], NRGNN [21], MTS-GNN [10], and BO-NNC [8]). This selection encompasses traditional supervised models, labelefficient self-supervised paradigms, and specialized noiserobust techniques, providing a thorough evaluation across diverse methodological categories.

TABLE I: Classification accuracy (%) of all methods under different noise rates on four datasets. Best results are in bold.  
Cora
<table><tr><td>Methods</td><td colspan="3">Uniform</td><td colspan="2">Pair</td></tr><tr><td>GCN</td><td>20%</td><td>40%</td><td>60%</td><td>20%</td><td>40%</td></tr><tr><td>GAT</td><td>74.8 (0.9) 76.4 (0.7)</td><td>61.5 (0.3) 61.6 (4.4)</td><td>36.1 (1.8)</td><td>69.4 (0.6)</td><td>53.6 (0.9)</td></tr><tr><td>DGI</td><td>82.3 (0.2)</td><td>70.2 (0.2)</td><td>36.2 (2.9) 46.9 (0.2)</td><td>72.1 (0.7) 78.2 (0.2)</td><td>56.6 (2.5) 65.3 (0.3)</td></tr><tr><td>GCA</td><td>78.9 (0.7)</td><td>76.7 (0.7)</td><td>50.4 (3.7)</td><td>78.3 (1.0)</td><td>68.9 (1.7)</td></tr><tr><td>SUGRL</td><td>83.2 (0.4)</td><td>70.5 5 (0.1)</td><td>43.1 (0.2)</td><td>78.0 (0.2)</td><td>64.3 (0.6)</td></tr><tr><td>JoCoR NRGNN MTS-GNN 79.5 (1.2)</td><td>76.0 (1.0) 79.6 (0.4)</td><td>63.5 5 (2.2) 74.5 (1.4) 76.9 (1.4)</td><td>38.9 (2.7) 45.8 (2.5) 60.1 (3.8)</td><td>71.6 (2.0) 75.3 (0.7) 78.9 (0.9)</td><td>54.9 (0.9) 65.4 (2.1) 73.2 (2.0)</td></tr></table>

DBLP
<table><tr><td>Methods</td><td></td><td>Uniform</td><td></td><td>Pair</td></tr><tr><td>GCN</td><td>20%</td><td>40%</td><td>60% 20%</td><td>40%</td></tr><tr><td>GAT</td><td>80.3 (0.4) 81.2 (0.2)</td><td>77.5 (1.2) 59.9 (3.9) 80.1 (0.4)</td><td>78.2 (1.0) 80.1 (0.5)</td><td>60.2 (2.8) 68.7 (1.4)</td></tr><tr><td>DGI</td><td>80.5 (0.1)</td><td>61.4 (1.2) 78.8 (0.2)</td><td>80.3 (0.2)</td><td>67.4 (1.1)</td></tr><tr><td>GCA</td><td>83.5 (0.1)</td><td>82.5 5 (0.4)</td><td>75.7 (0.1) 66.7 (3.0)</td><td></td></tr><tr><td>SUGRL</td><td>78.5 (0.1)</td><td>77.4 (0.3) 60.9 (0.8)</td><td>82.7 (0.5) 76.0 (0.3)</td><td>70.3 (2.4) 62.4 (1.2)</td></tr><tr><td>JoCoR NRGNN</td><td>80.6 (0.3) 78.2 80.6 (0.6) 81.7 (0.4)</td><td>(0.9) 60.7 (3.5) 68.7 (2.0)</td><td>78.7 (0.8) 80.3 (0.9)</td><td>61.6 (2.4) 69.5 (2.6)</td></tr><tr><td>MTS-GNN BO-NNC</td><td>82.3 (0.3) 82.2 (0.3) 83.6 (0.1) 83.4 (0.8)</td><td>72.0 (3.0) 79.6 (0.5)</td><td>83.4 (0.2) 83.6 (0.2)</td><td>79.0 (0.3) 77.8 (1.3)</td></tr></table>

3) Experimental Setup: In our experiments, we follow previous studies [46], [47] to partition each dataset into three disjoint subsets: training, validation, and test. For Cora and Citeseer, we adopt the standard splits provided by the PyTorch Geometric library. For DBLP, Computers and Photo, we follow [10], randomly selecting 5%, 10%, and 60% of the nodes for the training, validation, and test sets, respectively. In addition, following [46], [47], we inject noise into the datasets with different noise ratios. We consider two types of noise, namely uniform noise and pair noise:

• Uniform noise: Each node label is randomly flipped to any other class with equal probability.

• Pair noise: Node labels are flipped to predefined paired classes according to specified class mappings.

## B. Result Analysis

We assess the effectiveness of PASTA by comparing it against all comparison methods across five datasets under different noise levels. The results on four representative datasets are summarized in Table I, and the results on the remaining dataset, Photo, are provided in Appendix A.

From the results, we have the following observations. ❶ The proposed PASTA consistently achieves the best performance across four datasets. For instance, PASTA achieves an average improvement of 1.1% over the strongest competitor (BO-NNC) and 12.7% over the weakest one (GCN). The reason is that the proposed VaCE loss and PaSim loss enable the classifier to effectively learn the correct classification patterns from consensus labels, thereby preventing overfitting to noisy labels in the dataset and achieving better generalization performance. ❷ PASTA shows substantial advantages over all semi-supervised and self-supervised node classification methods (i.e.GCN, GAT, DGI, GCA, and SUGRL). For example, compared with DGI, which achieves the highest performance among the four methods, PASTA achieves average improvements of 8.3% under uniform noise and 8.5% under pair noise across all datasets. This is because PASTA effectively enhances the quality of label information in the dataset through noise filtering and label enrichment, providing more reliable supervision for model training. ❸ PASTA also outperforms existing NNC methods (e.g.JoCoR, NRGNN, MTS-GNN, and BO-NNC). Compared to BO-NNC, PASTA achieves the highest performance among these four methods, by an average of 1.1% across all experimental conditions. The reason is that PASTA mitigates the limitations of existing NNC methods by iteratively optimizing the annotation module and the partial label learning module, thereby achieving stronger robustness against label noise. This also demonstrates that introducing partial labels to tackle the NNC problem is reasonable and effective.

CiteSeer
<table><tr><td rowspan="2">Methods</td><td colspan="3">Uniform</td><td colspan="2">Pair</td></tr><tr><td>20%</td><td>40%</td><td>60%</td><td>20%</td><td>40%</td></tr><tr><td>GCN</td><td>63.1 (1.0)</td><td>59.8 (0.5)</td><td>38.3 (0.6)</td><td>61.7 (0.2)</td><td>51.7 (0.7)</td></tr><tr><td>GAT</td><td>64.4 (1.1)</td><td>60.8 (2.0)</td><td>41.5 (1.4)</td><td>64.5 (1.7)</td><td>51.5 (1.9)</td></tr><tr><td>DGI</td><td>66.6 (0.2)</td><td>68.2 (0.3)</td><td>45.9 (0.1)</td><td>69.2 (0.1)</td><td>54.5 (0.3)</td></tr><tr><td>GCA</td><td>54.9 (0.6)</td><td>53.3 (1.0)</td><td>33.6 (2.2)</td><td>58.6 (0.7)</td><td>53.3 (0.6)</td></tr><tr><td>SUGRL</td><td>59.9 (0.3)</td><td>55.3 (0.3)</td><td>19.8 (2.2)</td><td>64.8 (0.1)</td><td>40.9 (0.2)</td></tr><tr><td>JoCoR</td><td>65.2 (2.6)</td><td>63.4 (1.1)</td><td>39.6 (5.1)</td><td>65.0 (2.3)</td><td>51.4 (2.4)</td></tr><tr><td>NRGNN</td><td>67.4 (1.2)</td><td>60.6 (1.8)</td><td>46.5 (1.9)</td><td>66.9 (2.8)</td><td>53.1 (2.3)</td></tr><tr><td>MTS-GNN</td><td>72.3 (2.3)</td><td>68.9 (0.7)</td><td>57.1 (1.9)</td><td>69.1 (1.5)</td><td>58.2 (3.0)</td></tr><tr><td>BO-NNC</td><td>70.1 (0.3)</td><td>69.6 (0.8)</td><td>67.1 (1.1)</td><td>72.2 (0.8)</td><td>66.4 (1.5)</td></tr><tr><td>PASTA</td><td>72.7 (1.0)</td><td>71.2 (0.3)</td><td>69.3 (0.8)</td><td>72.2 (1.2)</td><td>67.6 (1.4)</td></tr></table>

Computers
<table><tr><td>Methods</td><td></td><td>Uniform</td><td></td><td>Pair</td><td></td></tr><tr><td></td><td>20%</td><td>40%</td><td>60%</td><td>20%</td><td>40%</td></tr><tr><td>GCN</td><td>84.9 (0.6)</td><td>83.5 (0.6)</td><td>77.9 (0.3)</td><td>84.0 (0.4)</td><td>71.2 (1.7)</td></tr><tr><td>GAT DGI</td><td>84.5 (0.6)</td><td>80.6 (2.0)</td><td>76.0 (1.1)</td><td>84.7 (1.0)</td><td>75.6 (1.1)</td></tr><tr><td>GCA</td><td>81.5 (0.1)</td><td>79.4 (0.2)</td><td>75.9 (0.3)</td><td>79.6 (0.1)</td><td>69.2 (0.9)</td></tr><tr><td>SUGRL</td><td>78.0 (0.6)</td><td>74.7 (0.9)</td><td>63.6 (0.5)</td><td>80.9 (0.2)</td><td>69.7 (1.8)</td></tr><tr><td>JoCoR</td><td>85.0 (0.1)</td><td>83.1 (0.1)</td><td>78.5 (0.2)</td><td>84.5 (0.1)</td><td>74.0 (0.4)</td></tr><tr><td>NRGNN</td><td>85.2 (0.8)</td><td>84.0 (0.4)</td><td>77.9 (0.6)</td><td>84.4 (0.9)</td><td>75.4 (1.3)</td></tr><tr><td></td><td>85.2 (0.9)</td><td>84.1 (0.9)</td><td>77.5 (0.8)</td><td>84.3 (0.3)</td><td>75.0 (1.6)</td></tr><tr><td>MTS-GNN</td><td>86.1 (0.6)</td><td>83.6 (2.2)</td><td>80.2 (1.2)</td><td>86.7 (0.3)</td><td>79.7 (1.7)</td></tr><tr><td>BO-NNC PASTA</td><td>85.6 (0.6) 86.5 (0.3)</td><td>84.4 (0.6) 84.8 (0.2)</td><td>80.3 (0.7) 80.7 (0.6)</td><td>84.1 (0.8) 85.6 (0.3)</td><td>80.0 (3.4) 83.5 (0.7)</td></tr></table>

## C. Ablation Study

PASTA has four key designs, namely annotators to construct consensus labels, the VaCE loss, the PaSim loss, and a selftraining strategy. To evaluate the contribution of each component, we conducted extensive experiments under different settings, including uniform noise with a rate of 60% and pair noise with a rate of 40%.

First, we evaluated the performance of different combinations of the PaSim loss (C1), VaCE loss (C2), and self-training strategy (C3). The results are summarized in Table II. We can observe that: ❶ Our method with all components improves by 4.2% on average compared to the versions with any two components. This suggests that all three components are essential to the overall performance of our method, confirming the soundness of our approach. ❷ Compared with the case without using the PaSim loss and VaCE loss, incorporating these two losses yields an average improvement of 1.7%. This indicates that the proposed losses effectively help the classifier learn correct classification patterns from the consensus labels. ❸ Performing the self-training strategy results in an average improvement of 9.6% compared to methods that do not utilize it. This demonstrates that the designed iterative optimization process can effectively combine the strengths of one-hot label learning and partial label learning, thereby enabling the student model to achieve stronger generalization performance.

Second, we evaluated the contribution of annotators to construct consensus labels. Specifically, we compared the performance of our method with that of using a single annotator, where partial labels are constructed by selecting the top-2 predicted classes with the highest confidence. The results are shown in Figure 3. The experimental results demonstrate that the proposed strategy of collaboratively constructing partial labels with multiple annotators significantly outperforms the approach that relies on a single annotator. Specifically, compared with using DGI, GCA, and SUGRL individually to generate partial labels, our method achieves an average improvement of 7.5%, 6.9%, and 6.1% in classification accuracy, respectively. These results verify the effectiveness of the multi-annotator strategy, which fully exploits the complementarity of different annotators’ predictions to substantially enhance the quality of partial labels, thereby further improving the generalization performance of the classifier.

## D. Parameter Sensitivity Analysis

In this subsection, we investigate the sensitivity of PASTA to the selection of iterative optimization rounds τ . We vary τ from $\{ 0 , \ 1 , \ . . . , \ 5 \}$ , and report the results in Figure 4. Sensitivity to the loss balance hyperparameter λ is provided in Appendix A.

First, according to the Figure 4, we can see that: ❶ The initial few iterations bring significant performance improvement compared to the non-iterative setting. This is because the self-training mechanism effectively refines the annotators with more reliable supervision, leading to better partial labels and more robust classifier training. ❷ The model performance improves as τ increases and tends to stabilize when τ exceeds

TABLE II: Classification accuracy (%) of PASTA with different component combinations under the highest noise rates for uniform and pair noise.  
Uniform noise (60%)
<table><tr><td>C1</td><td>C2</td><td>C3</td><td>Cora</td><td>Citeseer</td><td>DBLP</td><td>Computers</td></tr><tr><td> $\overline { { \checkmark } }$ </td><td></td><td></td><td>59.9 (3.1)</td><td>36.0 (11.3)</td><td>71.5 (1.5)</td><td>79.3 (0.4)</td></tr><tr><td></td><td> $\checkmark$ </td><td></td><td>62.6 (1.7)</td><td>39.9 (9.3)</td><td>74.1 (1.4)</td><td>79.4 (0.4)</td></tr><tr><td> $\checkmark$ </td><td></td><td>√</td><td>70.8 (1.0)</td><td>64.0 (6.2)</td><td>80.3 (0.4)</td><td>80.7 (0.3)</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>63.1 (1.9)</td><td>38.4 (12.7)</td><td>73.6 (1.5)</td><td>79.6 (0.4)</td></tr><tr><td></td><td></td><td>√</td><td>70.7 (1.7)</td><td>64.6 (4.2)</td><td>80.3 (0.4)</td><td>80.8 (0.3)</td></tr><tr><td></td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>70.3 (2.4)</td><td>64.5 (8.5)</td><td>80.3 (0.5)</td><td>80.7 (0.7)</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>71.9 (1.2)</td><td>69.3 (0.8)</td><td>80.8 (0.5)</td><td>80.7 (0.6)</td></tr></table>

<table><tr><td colspan="6">Tan molse (40 %)</td></tr><tr><td>C1</td><td>C2</td><td>C3</td><td>Cora 73.0 (1.1)</td><td>Citeseer DBLP 54.3 (5.1)</td><td>Computers</td></tr><tr><td></td><td></td><td></td><td></td><td>72.1 (0.6)</td><td>77.9 (2.8)</td></tr><tr><td></td><td> $\checkmark$ </td><td></td><td>72.8 (1.2) 55.0 (3.7)</td><td>73.8 (0.4) 77.7 (1.2)</td><td>79.0 (3.9)</td></tr><tr><td> $\checkmark$ </td><td></td><td> $\checkmark$ </td><td>78.0 (1.1)</td><td>64.4 (1.1)</td><td>82.6 (1.1)</td></tr><tr><td> $\checkmark$ </td><td>√</td><td></td><td>73.1 (1.1)</td><td>54.9 (3.8) 73.1 (0.3)</td><td>79.5 (3.4)</td></tr><tr><td></td><td></td><td>√</td><td>77.7 (0.8)</td><td>64.2 (1.3) 76.6 (1.4)</td><td>82.7 (0.6)</td></tr><tr><td></td><td> $\checkmark$ </td><td>√</td><td>78.5 (0.8)</td><td>67.0 (1.1) 77.9 (1.2)</td><td>82.5 (0.7)</td></tr><tr><td> $\overline { { \checkmark } }$ </td><td> $\overline { { \checkmark } }$ </td><td>√</td><td>78.7 (0.5)</td><td>67.6 (1.4) 79.2 (0.6)</td><td>83.5 (0.7)</td></tr></table>

![](images/6764ca3fc8da8738ae05dc8d3f14efd38ac081dc322098508a7c15026634a03c.jpg)  
(a) Uniform noise (60%)

![](images/886ddf41c71ca56369ad6d064065154171ccb3035388f24416309a9dc372e22e.jpg)  
(b) Pair noise (40%)

Fig. 3: Node classification performance of PASTA with different annotators under the highest noise rates for uniform and pair noise.  
![](images/305914c99f897058c4f53733b6bbc62e86159e5f13d82fa6596b616f5dc711d3.jpg)

![](images/c3fe5be8f74abd8ee4f773df29e64be82f3481681cc08412d022a33fc8c2c550.jpg)  
(a) Uniform noise (60%)  
(b) Pair noise (40%)  
Fig. 4: Sensitivity Analysis of the Parameter τ in PASTA.

3. The reason is that the self-training strategy in our method enables the annotators and the classifier to mutually enhance each other, thereby improving classification performance, and the training converges after approximately three iterations.

## E. Consensus Partial Label Visualization

To intuitively illustrate the difference between the consensus partial labels constructed in this work and the traditional onehot pseudo-labels, we select ten nodes with noisy labels from each of the three datasets for label visualization. In this experiment, one-hot pseudo-labels were generated by summing the predictions of the three annotators and selecting the class with the highest confidence. As shown in Figure 5, although some annotators can correctly predict these nodes, the traditional one-hot pseudo-labels based on aggregated probabilities may still assign incorrect classes. In contrast, the consensus partial labels given by PASTA can effectively prevent the loss of accurate information by retaining multiple candidate categories, thereby providing more reliable supervisory signals.

![](images/ea9266e97f920e3f685f819ff2748faeb305e0f3c7cf5c7e5d613d6ae1657331.jpg)  
Fig. 5: Visualization of consensus partial label and one-hot pseudo-label (ten sampled noisy nodes).

## VI. CONCLUSION

As noisy labels are commonly encountered in real-world graph data, noisy node classification has become a crucial problem for advancing the practical use of graph-structured data. To address this challenge, we propose a novel Partial label-based Self-training framework (PASTA), which breaks the limitations of conventional one-hot label supervision. PASTA consists of two synergistic modules: a self-supervised ensemble annotation module for generating high-quality partial labels, and a dual-space partial label learning module that exploits partial labels via complementary loss in both label and representation spaces, and also refines node labels. Through iterative optimization, the modules mutually reinforce each other, progressively improving label quality and classifier robustness. Experiments on five real-world benchmarks show that PASTA consistently outperforms existing SOTA methods across different datasets and noise levels.

## REFERENCES

[1] S. A. Myers, A. Sharma, P. Gupta, and J. Lin, “Information network or social network? the structure of the twitter follow graph,” in Proceedings of the 23rd international conference on world wide web, 2014, pp. 493– 498.

[2] S. Wu, F. Sun, W. Zhang, X. Xie, and B. Cui, “Graph neural networks in recommender systems: a survey,” ACM Computing Surveys, vol. 55, no. 5, pp. 1–37, 2022.

[3] J. Wang, S. Zhang, Y. Xiao, and R. Song, “A review on graph neural network methods in financial applications,” Journal of Data Science, vol. 20, no. 2, pp. 111–134, 2022.

[4] E. Purificato, L. Boratto, and E. W. De Luca, “Leveraging graph neural networks for user profiling: Recent advances and open challenges,” in Proceedings of the 32nd ACM international conference on information and knowledge management, 2023, pp. 5216–5219.

[5] J. Yi and Y. Deng, “Research on online shopping mall product classification recommendation based on graph neural network,” in 2024 IEEE 3rd International Conference on Electrical Engineering, Big Data and Algorithms (EEBDA). IEEE, 2024, pp. 297–301.

[6] Z. Liu, Y. Dou, P. S. Yu, Y. Deng, and H. Peng, “Alleviating the inconsistency problem of applying graph neural network to fraud detection,” in Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, 2020, pp. 1569–1572.

[7] M. Zhang, L. Hu, C. Shi, and X. Wang, “Adversarial label-flipping attack and defense for graph neural networks,” in 2020 IEEE International Conference on Data Mining (ICDM). IEEE, 2020, pp. 791–800.

[8] Y. Liu, Z. Wu, Z. Lu, C. Nie, G. Wen, Y. Zhu, and X. Zhu, “Noisy node classification by bi-level optimization based multi-teacher distillation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 18, 2025, pp. 19 033–19 040.

[9] Z. Lu, J. Ma, Z. Wu, B. Zhou, and X. Zhu, “A noise-resistant graph neural network by semi-supervised contrastive learning,” Information Sciences, vol. 658, p. 120001, 2024.

[10] Y. Liu, Z. Wu, Z. Lu, G. Wen, J. Ma, G. Lu, and X. Zhu, “Multi-teacher self-training for semi-supervised node classification with noisy labels,” in ACM MM, 2023, pp. 2946–2954.

[11] Z. Lu, Y. Liu, G. Wen, B. Zhou, W. Zhang, and J. Zhang, “Noiseresistant graph neural networks with manifold consistency and label consistency,” Expert Systems with Applications, vol. 245, p. 123120, 2024.

[12] Y. Li, J. Yin, and L. Chen, “Unified robust training for graph neural networks against label noise,” in PAKDD. Springer, 2021, pp. 528– 540.

[13] L. Feng and B. An, “Partial label learning with self-guided retraining,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, no. 01, 2019, pp. 3542–3549.

[14] D. Saravanan, N. Manwani, and V. Gandhi, “Pseudo-labelling meets label smoothing for noisy partial label learning,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2025, pp. 2143–2152.

[15] J. Lv, Y. Liu, S. Xia, N. Xu, M. Xu, G. Niu, M.-L. Zhang, M. Sugiyama, and X. Geng, “What makes partial-label learning algorithms effective?” Advances in Neural Information Processing Systems, vol. 37, pp. 89 513–89 534, 2024.

[16] M. Sheng, Z. Sun, Z. Cai, T. Chen, Y. Zhou, and Y. Yao, “Adaptive integration of partial label learning and negative learning for enhanced noisy label learning,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 38, no. 5, 2024, pp. 4820–4828.

[17] H. Wang, R. Xiao, Y. Li, L. Feng, G. Niu, G. Chen, and J. Zhao, “Pico: Contrastive label disambiguation for partial label learning.” ICLR, vol. 1, no. 2, p. 5, 2022.

[18] J. Lv, M. Xu, L. Feng, G. Niu, X. Geng, and M. Sugiyama, “Progressive identification of true labels for partial-label learning,” in international conference on machine learning. PMLR, 2020, pp. 6500–6510.

[19] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[20] X. Shen, Y. Liu, Y. Wang, R. Miao, Y. Dai, S. Pan, Y. Chang, and X. Wang, “Raising the bar in graph ood generalization: Invariant learning beyond explicit environment modeling,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[21] E. Dai, C. Aggarwal, and S. Wang, “Nrgnn: Learning a label noise resistant graph neural network on sparsely and noisily labeled graphs,” in KDD, 2021, pp. 227–236.

[22] Y. Liu, S. Li, Y. Zheng, Q. Chen, C. Zhang, P. S. Yu, and S. Pan, “From few-shot to zero-shot: Towards generalist graph anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[23] Y. Liu, Y. Liu, Y. Zheng, A. W.-C. Liew, X. Cao, and S. Pan, “Rethinking feature alignment in generalist graph anomaly detection: A relational fingerprint-based approach,” in International Conference on Machine Learning, 2026.

[24] S. Li, Y. Liu, Y. Zheng, X. Cao, S. Pan, and H. T. Shen, “Towards onefor-all anomaly detection for tabular data,” in International Conference on Machine Learning, 2026.

[25] S. Li, Y. Liu, Y. Zheng, M. Li, Q. V. H. Nguyen, and S. Pan, “Ofamas: One-for-all multi-agent system topology design based on mixtureof-experts graph generative models,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 1333–1344.

[26] Y. Tan, G. Long, J. Jiang, and C. Zhang, “Influence-oriented personalized federated learning,” in IEEE International Conference on Data Mining, 2026.

[27] Y. Tan, C. Chen, W. Zhuang, X. Dong, L. Lyu, and G. Long, “Taming heterogeneity to deal with test-time shift in federated learning,” in International Workshop on Federated Learning for Distributed Data Mining, 2023.

[28] J. Yuan, X. Luo, Y. Qin, Y. Zhao, W. Ju, and M. Zhang, “Learning on graphs under label noise,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[29] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in International Conference on Learning Representations, 2017.

[30] E. Dai and S. Wang, “Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information,” in WSDM, 2021, pp. 680–688.

[31] X. Gong, D. Yuan, and W. Bao, “Top-k partial label machine,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 11, pp. 6775–6788, 2021.

[32] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[33] P. Velickoviˇ c, W. Fedus, W. L. Hamilton, P. Li´ o, Y. Bengio, and R. D.\` Hjelm, “Deep graph infomax,” International Conference on Learning Representations, 2019.

[34] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang, “Graph contrastive learning with adaptive augmentation,” in Proceedings of the Web Conference 2021, 2021, pp. 2069–2080.

[35] Y. Mo, L. Peng, J. Xu, X. Shi, and X. Zhu, “Simple unsupervised graph representation learning,” in AAAI, vol. 36, no. 7, 2022, pp. 7797–7805.

[36] L. Wu, Y. Huang, H. Lin, Z. Liu, T. Fan, and S. Z. Li, “Automated graph self-supervised learning via multi-teacher knowledge distillation,” arXiv preprint arXiv:2210.02099, 2022.

[37] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” NeurIPS, vol. 30, 2017.

[38] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and\` Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations, 2018.

[39] Q. Li, Z. Han, and X.-M. Wu, “Deeper insights into graph convolutional networks for semi-supervised learning,” in AAAI, vol. 32, no. 1, 2018.

[40] X.-J. Gui, W. Wang, and Z.-H. Tian, “Towards understanding deep learning from noisy labels with small-loss criterion,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence Organization, 2021, pp. 2469–2475.

[41] Y. Lu and W. He, “Selc: Self-ensemble label correction improves learning with noisy labels,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence Organization, 2022, pp. 3278– 3284.

[42] Z. Yang, W. Cohen, and R. Salakhudinov, “Revisiting semi-supervised learning with graph embeddings,” in ICML. PMLR, 2016, pp. 40–48.

[43] A. Bojchevski and S. Gunnemann, “Deep gaussian embedding of¨ graphs: Unsupervised inductive learning via ranking,” in International Conference on Learning Representations, 2018, pp. 1–13.

[44] O. Shchur, M. Mumme, A. Bojchevski, and S. Gunnemann, “Pitfalls¨ of graph neural network evaluation,” arXiv preprint arXiv:1811.05868, 2018.

[45] H. Wei, L. Feng, X. Chen, and B. An, “Combating noisy labels by agreement: A joint training method with co-regularization,” in CVPR, 2020, pp. 13 726–13 735.

[46] B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. Tsang, and M. Sugiyama, “Co-teaching: Robust training of deep neural networks with extremely noisy labels,” NeurIPS, vol. 31, 2018.

[47] L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei, “Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels,” in ICML. PMLR, 2018, pp. 2304–2313.

## A. Additional Results

Classification Results of Photo Dataset: We report the node classification performance of PASTA and all comparison methods on the Photo dataset under different noise rates in Table III. Obviously, PASTA also achieves the best performance on the Photo dataset. Specifically, it surpasses the existing SOTA method BO-NNC by 2.27% and 2.57% under the uniform and pair noise settings, respectively. This result further demonstrates the robustness and effectiveness of PASTA in handling label noise with varying types and intensities.

TABLE III: Classification accuracy (%) of all methods under different noise rates on dataset Photo. Best results are in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Uniform</td><td colspan="2">Pair 60%</td></tr><tr><td>20%</td><td>40%</td><td>20%</td><td>40%</td></tr><tr><td>GCN</td><td>89.90 (0.47)</td><td>87.03 (1.21)</td><td>81.07 (2.13) 85.74 (0.95)</td><td>64.27 (1.66)</td></tr><tr><td>GAT</td><td>89.86 (1.24)</td><td>88.17 (1.03)</td><td>81.24 (1.57) 87.36 (1.25)</td><td>68.67 (1.35)</td></tr><tr><td>DGI</td><td>90.22 (0.29)</td><td>85.20 (0.14)</td><td>83.49 9 (0.83) 85.97</td><td>(0.10) 63.43 (0.90)</td></tr><tr><td>GCA</td><td>86.91 (0.19)</td><td>82.94 (1.04)</td><td>73.73 (1.76) 86.03 (1.46)</td><td>77.34 (0.33)</td></tr><tr><td>SUGRL</td><td>91.47 (0.10)</td><td>89.22 (0.20)</td><td>83.25 (0.86) 89.92</td><td>(0.13) 70.88 (0.71)</td></tr><tr><td>JoCoR</td><td>89.90 (0.45)</td><td>87.49 (0.64)</td><td>82.08 3 (1.79) 86.93</td><td>(0.24) 68.73 (1.65)</td></tr><tr><td>NRGNN</td><td>89.79 (0.79)</td><td>87.45 (1.15)</td><td>81.45 (1.50) 86.37 (1.44)</td><td>67.11 (2.47)</td></tr><tr><td>MTS-GNN</td><td>91.97 (0.23)</td><td>88.48 (0.37)</td><td>87.89 (1.64) 89.21 (0.42)</td><td>79.78 (5.61)</td></tr><tr><td>BO-NNC</td><td>90.82 (1.10)</td><td>88.78 (2.35)</td><td>87.43 (2.64) 89.63</td><td>(0.16) 81.58 (2.74)</td></tr><tr><td>PASTA</td><td>92.07 (0.25)</td><td>91.01 (0.70)</td><td>90.75 (0.90) 91.43</td><td>(0.99) 84.92 (2.02)</td></tr></table>

Sensitivity to λ: Our method has another key hyperparameter, i.e., λ controlling the weight of the $\mathcal { L } _ { p a s i m }$ . To investigate the sensitivity of our method to the hyper-parameter, we vary λ in {0.01, 0.1, 1, 10, 100}, and report the results in Figure 6. From the figure, we can see that our method is not sensitive to the value of λ. This is because the optimization objectives of $\mathcal { L } _ { p a s i m }$ and $\mathcal { L } _ { v a c e }$ are inherently aligned, with $\mathcal { L } _ { p a s i m }$ designed to help $\mathcal { L } _ { v a c e }$ achieve its goal more robustly. As a result, the model can effectively optimize both objectives simultaneously regardless of the value of λ.

![](images/1afe809e2076740eb18362a9471ed98004b7397412d86a9260f1fc5e1d805f9b.jpg)  
(a) Uniform noise (60%)

![](images/c7bd2dd024e0dda10f7925a91038dd96dc7bdac2ad6bca06faf0c2f0d025d78e.jpg)  
(b) Pair noise (40%)  
Fig. 6: Sensitivity Analysis of the Parameter λ in PASTA.