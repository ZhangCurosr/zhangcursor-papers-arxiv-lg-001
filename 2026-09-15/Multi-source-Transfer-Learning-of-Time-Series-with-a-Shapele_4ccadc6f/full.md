# Multi-source Transfer Learning of Time Series with a Shapelet-based Distance Measure

Jiseok Lee<sup>a,∗</sup>, Brian Kenji Iwana<sup>b</sup>

<sup>a</sup>Graduate School of Information Science and Electrical Engineering, Kyushu University, 744 Motooka, Nishi Ward, Fukuoka, 819-0395, Fukuoka, Japan

<sup>b</sup>Faculty of Information Science and Electrical Engineering, Kyushu University, 744 Motooka, Nishi Ward, Fukuoka, 819-0395, Fukuoka, Japan

## Abstract

Transfer learning is an efective technique for addressing data scarcity in deep learning for time series classification, but its success depends on the selection of source datasets. Conventional transferability estimation methods are often computationally expensive, as they require fully pre-training a model on each potential source dataset to assess its suitability. This paper introduces a novel, training-free source selection method named Shapelet Matching. Our approach first identifies discriminative shapelets from the target and potential source datasets. Then, Shapelet Matching quantifies dataset similarity by comparing the extracted sets of shapelets. To mitigate the risk of negative transfer from selecting an unsuitable single source, we introduce a multi-source transfer learning method. We select several source datasets based on their shapelet-based similarity scores, combine them into a single multi-source dataset, and use this aggregated dataset for pre-training. The model is then fine-tuned on the target task. We evaluated our method on 128 datasets from the UCR Archive using both temporal CNN and Transformer architectures. The empirical results demonstrate that our multi-source pre-training reduces the risk of negative transfer on average. Shapelet Matching achieves the strongest performance for the CNN backbone and remains competitive for patch-based Transformer architectures, while avoiding the cost of pre-training a separate model for every candidate source.

Keywords:

Transfer learning, Time series classification, Transferability estimation

## 1. Introduction

Deep neural networks have become a dominant and efective methodology for time series recognition. For instance, temporal convolutional neural networks (CNN) [1] have demonstrated considerable success across various time series domains. A primary limitation of these models, however, is their reliance on large training datasets [2, 3]. This dependency presents a significant challenge, as most time series data is unlabeled, making it costly to acquire suficient labeled examples for supervised learning.

To address the challenge of limited labeled data, several techniques such as transfer learning, self-supervised learning, and data augmentation have proven efective. Among these, transfer learning is an efective method for initializing the weights of neural networks [4]. In this study, we adopt a standard pre-training and fine-tuning paradigm. A model is first trained on a large-scale source dataset and subsequently adapted to a smaller target dataset. This process can significantly reduce the data requirements for the target task by enabling the model to learn general-purpose feature representations during pre-training [5]. Recent time series foundation models, including Timer [6], MOMENT [7], and UniTS [8], further highlight the value of large-scale pre-training. However, many practical supervised time-series classification settings still lack access to large external corpora or released foundation-model checkpoints. In such cases, users are often faced with a finite pool of heterogeneous labeled datasets and must decide which of them should be used as sources for pre-training.

However, the usefulness of pre-training often depends on the domain gap or similarity between the source and target datasets. Such a gap may arise from diferences in temporal scale, sampling characteristics, noise patterns, or the location and granularity of class-discriminative patterns. Consequently, source selection is critical because pre-training on mismatched sources can bias the feature extractor toward non-transferable patterns and induce negative transfer. A common way to address this problem is to estimate transferability using a model that has already been pre-trained on each candidate source. However, for time series datasets, this model-based strategy can be costly because every candidate source must be pre-trained for each backbone architecture before it can be evaluated. It is also less transparent than pattern-based comparison, since the selected source is justified by model outputs or internal features rather than by explicit temporal subsequences.

To address the scarcity of large, suitable datasets for pre-training, we propose a multi-source transfer learning approach. This method involves consolidating multiple smaller datasets into a single, larger source dataset for pre-training. Multi-source pretraining can reduce dependence on a single source, but it also makes source selection more important because mismatched sources can still introduce negative transfer. To mitigate this risk, we propose using shapelet-based similarity as a metric to assess the transferability between potential source datasets and the target task, thereby guiding the selection process. Our method is based on the assumption that transferable local discriminative subsequences exist across at least a subset of source and target datasets. When two datasets share such shapelets, pre-training on the source is more likely to induce representations that remain useful after fine-tuning on the target.

This paper proposes a novel source selection method for multi-source transfer learning based on a shapelet-based distance measure, which serves as an alternative to conventional transferability estimation. The proposed multi-source transfer learning framework, shown in Fig. 1, comprises four steps. First, class-discriminative shapelets are identified from each dataset using a matrix profile-based discovery algorithm [9]. Second, each candidate source is scored and ranked according to its shapelet-based distance to the target dataset. Third, the selected sources are resized, balanced, relabeled, and concatenated into a unified multi-source dataset. Finally, the backbone is pre-trained on the aggregated source dataset and fine-tuned on the target task. The underlying principle of this approach is that datasets sharing discriminative subsequences are presumed to possess similar feature distributions, thereby enabling efective knowledge transfer. Because the proposed criterion is computed directly from source and target datasets, it does not require pre-training a model for each candidate source. The resulting source rankings are therefore reusable across backbone architectures and ofer an explicit pattern-level rationale through the matched shapelets. This combination targets a practical gap between conventional transferability estimation and large-scale foundation-model pre-training: it keeps source selection lightweight while retaining an explicit pattern-level explanation of why a source is selected.

The contributions of this paper are as follows:

• We propose a new shapelet-based similarity measure called Shapelet Matching. This method focuses on dense matching of subsequences extracted from the target dataset and the candidate source datasets.

• We integrate this criterion into a multi-source transfer learning framework that aggregates the top-ranked source datasets for pre-training and then fine-tunes the backbone on the target task.

• We evaluate the method on all 128 univariate datasets from the 2018 UCR Time Series Archive (UCR Archive) [10], using CNN, Vision Transformer (ViT) [11], and PatchTST [12] backbones.

• We report statistical significance tests, computational cost comparisons, and architecture-dependent behavior to clarify both the strengths and limitations of the proposed method. We note that the method’s efectiveness varies by backbone architecture, which is further discussed in Section 7.3.

• We provide the implementation code for multi-source transfer learning with shapelet similarity-based source selection on GitHub<sup>1</sup>.

## 2. Related Work

## 2.1. Transfer Learning for Time Series

Transfer learning has long been recognized as an efective strategy when labeled data in the target domain are limited, and Zhuang et al. [5] provide a broad survey of this literature. For time series classification (TSC), however, the literature remains substantially smaller than in computer vision. Fawaz et al. [2] provided one of the earliest large-scale studies of transfer learning for TSC, showing on the UCR Archive that pre-training a CNN on a source dataset can improve target accuracy, while also revealing the risk of negative transfer when the source and target are poorly matched. Weber et al. [13] later systematized the broader literature on transfer learning with time series data and highlighted that TSC-specific studies with temporal neural networks are still relatively limited. More application-oriented work has explored transfer in specific settings: Clark and Doyle [14] studied transferability in cyber-physical health systems, whereas Gikunda and Jouandeau [15] considered homogeneous transfer active learning for time series classification. Taken together, these studies suggest that transfer learning is promising for TSC, but that its success depends strongly on source–target compatibility.

![](images/0701618bdfdc8d8f689b123e570e0c26379cffcc3d5e1dc4f54d88347ef972d2.jpg)  
Figure 1: End-to-end overview of the proposed multi-source transfer learning framework. In the Similarity Measure stage, class-discriminative shapelets are discovered from the target dataset and each candidate source dataset using the matrix profile-based procedure, and each source is assigned a Shapelet Matching distance to the target. In the Selection stage, sources are ranked by this distance and the top-K sources (dashed boxes) are retained. In the Concatenation stage, the selected sources are resized to the target temporal length, their label spaces are remapped into a unified multi-source label set, and the datasets are balanced so that each source contributes the same number of samples. In the Transfer Learning stage, a backbone network is pre-trained on the aggregated multi-source dataset and then fine-tuned on the target dataset with a target-specific classifier head.

A more recent line of work studies large pre-trained models for time series. Zhou et al. [16] showed that a pre-trained language model can be adapted to multiple time series tasks within a unified framework. Timer [6], MOMENT [7], and UniTS [8] further push this direction by learning general-purpose representations from large and heterogeneous time series data. These models are powerful when large pretraining data and substantial computational resources are available. However, this setting difers from the practical transfer-learning scenario studied in this paper, where users have limited computational resources and need task-specific portable models. Our Shapelet Matching method is therefore complementary to time-series foundation models rather than a replacement for them. Our method provides a lightweight, training-free source-selection criterion whose matched subsequences ofer potential interpretability when full-scale foundation-model pre-training is unavailable, unnecessary, or computationally prohibitive.

Recent shape-aware time-series classification research has also advanced eficiency and interpretability. SoftShape learns sparsified shape representations for eficient classification [17], whereas UniShape pre-trains a shape-aware foundation model to capture transferable patterns across domains [18]. In contrast to these approaches, which learn shape-aware representations within the classifier, Shapelet Matching uses discovered shapelets as a training-free criterion for ranking candidate source datasets before downstream model training.

## 2.2. Multi-Source Transfer Learning

Research on multi-source transfer learning, which leverages multiple source datasets for pre-training, is relatively less explored. The existing work in this area can be broadly categorized. One category includes boosting-based methods, such as the application of TrAdaBoost [19] by Yao et al. [20] and the SharedBoost algorithm proposed by Huang et al. [21]. Another category focuses on developing novel learning frameworks or weighting methods; for example, Multi-transfer [22] combines multiview with multi-source transfer learning, while Song et al. [23] introduced a method to weight source domains based on conditional probability diferences.

Multi-source transfer learning has also been specifically applied to time series classification. Some approaches focus on selecting the most relevant sources for a target domain. In the context of electroencephalogram (EEG) data, for example, Li et al. [24] trained a separate model for each source and selected the top-performing one, while Ren et al. [25] employed a preliminary classifier for the same purpose. Other methodologies aim to combine information from all sources. For sensor modality classification, Li et al. [26] trained a neural network to predict the source of time series segments, thereby learning an aligned representation.

Furthermore, this research presents several significant extensions of our preliminary study [27]. In this work, we propose the Shapelet Matching method, a substantial generalization of our previously introduced Minimum Shapelet method, by densely matching mutually preferable subsequences rather than matching only single pairs. Additionally, we have evaluated our methods on CNNs, ViT [11], and PatchTST [12], and based on the ViT results, we provide a more precise analysis of architecture dependency. Finally, we evaluate multi-source training with conventional transferability measures, which was not examined in our prior work.

## 3. Transferability Measures

## 3.1. Problem Setting

Assume a neural network $\theta \ : = \ : ( \omega , h )$ , where ω is a feature extractor and h is a linear classifier (head). Consider a transfer learning scenario where a neural network θ is first pre-trained on a source dataset S:

$$
S = \{ ( \mathbf { s } _ { 1 } , z _ { 1 } ) , \ldots , ( \mathbf { s } _ { m } , z _ { m } ) , \ldots , ( \mathbf { s } _ { M } , z _ { M } ) \} .\tag{1}
$$

Let $\mathbf { S } = \{ \mathbf { s } _ { 1 } , \ldots , \mathbf { s } _ { m } , \ldots , \mathbf { s } _ { M } \}$ be the set of source samples, and $\mathbf { Z } = \{ z _ { 1 } , \dots , z _ { m } , \dots z _ { M } \}$ be the corresponding source labels, with each label $z _ { m } \in \{ 1 , \dots , z , \dots , C _ { S } \}$ where $C _ { S }$ represents the number of source classes. The network θ is then fine-tuned on a target dataset T:

$$
\mathcal { T } = \{ ( \mathbf { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \mathbf { x } _ { n } , y _ { n } ) , \dotsc , ( \mathbf { x } _ { N } , y _ { N } ) \} .\tag{2}
$$

Let $\begin{array} { r l r } { { \bf X } } & { { } = } & { \{ { \bf x } _ { 1 } , \ldots , { \bf x } _ { n } , \ldots , { \bf x } _ { N } \} } \end{array}$ be the set of target samples, and let ${ \textbf { Y } } =$ $\{ y _ { 1 } , . . . , y _ { n } , . . . , y _ { N } \}$ be the corresponding target labels, where $y _ { n } \in \{ 1 , . . . , C _ { T } \}$ and

$C _ { T }$ denotes the number of target classes. Unlike the domain adaptation problem setting, there is no assumption that the datasets are similar tasks, that source label $z _ { m }$ and target label $y _ { n }$ are related, or that there exists a hypothesis (or model) that is suitable for both datasets [28].

Our setting is related to, but distinct from, standard domain adaptation and domain generalization. Domain adaptation typically assumes the same or closely related prediction task across domains and aims to reduce distribution shift [29], whereas domain generalization learns from multiple source domains to generalize to unseen domains [30]. By contrast, our setting allows heterogeneous source datasets with dataset-specific label spaces and potentially diferent task semantics. The objective is not domain alignment or domain-invariant prediction, but target-guided source selection: identifying source datasets that provide useful pre-training signals before finetuning on the target task.

## 3.2. Transferability Estimation

Given a source-trained model $\theta = \left( \omega , h \right)$ , a transferability estimator predicts how well the model is expected to perform after fine-tuning on $\mathcal { T }$ , without explicitly finetuning the model. The estimator computes a score from the target labels Y together with either the source-model predictions $\theta ( \mathbf { X } )$ or the source-trained features $\omega ( \mathbf { X } )$ The source and target label spaces need not correspond; the source predictions are used only to construct the transferability score.

## 3.2.1. H-score

H-score [31] is a transferability estimator computed from the features extracted by ω and the target labels Y, or:

$$
\operatorname { H - s c o r e } ( S , { \mathcal { T } } ) = \operatorname { T r } { \big ( } \operatorname { c o v } ( \omega ( \mathbf { X } ) ) ^ { - 1 } \operatorname { c o v } \left( \mathbb { E } [ \omega ( \mathbf { X } ) \mid \mathbf { Y } ] \right) { \big ) } ,\tag{3}
$$

where $\mathrm { T r } ( A ) = \textstyle \sum _ { i } ( A ) _ { i i }$ , and $\mathbb { E } [ \omega ( \mathbf { X } ) \mid \mathbf { Y } ]$ is the mean of a set of features from each label.

## 3.2.2. Log Expected Empirical Prediction (LEEP)

LEEP [32] attempts to predict the transferability for T using θ pre-trained on S. LEEP is then calculated by:

$$
\mathrm { L E E P } ( S , \mathcal { T } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log \left( \sum _ { z = 1 } ^ { C _ { S } } \hat { P } ( y _ { n } | z ) \theta ( \mathbf { x } _ { n } ) [ z ] \right) ,\tag{4}
$$

where $\theta ( \mathbf { x } _ { n } ) [ z ]$ is the probability assigned by the source-trained model to source class z for the input $\mathbf { X } _ { n } ,$ and $\hat { P } ( y | z )$ is an empirical conditional distribution estimating how likely target label y is, given that the model’s prediction lies in source class z. Specifically:

$$
{ \hat { P } } ( y \mid z ) = { \frac { { \hat { P } } ( y , z ) } { { \hat { P } } ( z ) } } = { \frac { \sum _ { n : y _ { n } = y } \theta ( \mathbf { x } _ { n } ) [ z ] } { \sum _ { n = 1 } ^ { N } \theta ( \mathbf { x } _ { n } ) [ z ] } } .\tag{5}
$$

## 3.2.3. Log Maximum Evidence (LogME)

LogME [33] is a transferability estimation that assesses how well pre-trained ω can explain Y. Specifically, the key quantity of LogME is the marginal log-likelihood (evidence):

$$
\begin{array} { r } { \mathcal { L } ( \alpha , \beta ) = \log P ( \mathbf { Y } | \omega ( \mathbf { X } ) , \alpha , \beta ) . } \end{array}\tag{6}
$$

LogME finds $\alpha ^ { * }$ and $\beta ^ { * }$ that maximize the evidence, $\mathcal { L } ( \alpha , \beta )$ . LogME is then calculated by normalizing the maximum evidence by N:

$$
\operatorname { L o g M E } ( S , { \mathcal { T } } ) = { \frac { { \mathcal { L } } ( \alpha ^ { * } , \beta ^ { * } ) } { N } } ,\tag{7}
$$

where

$$
\begin{array} { r } { ( \alpha ^ { * } , \beta ^ { * } ) = \operatorname * { a r g m a x } _ { \alpha , \beta } \mathcal { L } ( \alpha , \beta ) . } \end{array}\tag{8}
$$

## 3.2.4. Negative Conditional Entropy (NCE)

NCE [34] estimates transferability by measuring the uncertainty of target labels given source predictions. A lower conditional entropy suggests that source labels provide more information about target labels, implying better transferability.

NCE is computed as:

$$
\mathrm { N C E } ( S , \mathcal { T } ) = \sum _ { y = 1 } ^ { C _ { T } } \sum _ { z = 1 } ^ { C _ { S } } \hat { P } ( y , z ) \log \frac { \hat { P } ( y , z ) } { \hat { P } ( z ) } ,\tag{9}
$$

where $\hat { P } ( y , z )$ is the empirical joint probability of target label y and predicted source label z, and $\begin{array} { r } { \hat { P } ( z ) = \sum _ { y = 1 } ^ { C _ { T } } \hat { P } ( y , z ) } \end{array}$

## 3.2.5. TransRate

Similar to LogME, TransRate [35] measures how well the features $\omega ( \mathbf { X } )$ are separated into class-specific clusters. Formally, TransRate is defined as:

$$
\mathrm { T r a n s R a t e } ( S , \mathcal { T } ) = R ( \omega ( { \mathbf X } ) , \epsilon ) - R ( \omega ( { \mathbf X } ) , \epsilon | { \mathbf Y } ) ,\tag{10}
$$

where $R ( \omega ( \mathbf { X } ) , \epsilon )$ is the coding rate, measuring how many bits are required to represent $\omega ( \mathbf { X } )$ under a distortion ϵ:

$$
R ( \omega ( { \mathbf X } ) , \epsilon ) = \frac { 1 } { 2 } \log \mid { \mathbf I } + \frac { 1 } { N \epsilon } \omega ( { \mathbf X } ) ^ { T } \omega ( { \mathbf X } ) \mid ,\tag{11}
$$

and $R ( \omega ( \mathbf { X } ) , \epsilon \mid \mathbf { Y } )$ is the conditional coding rate computed by averaging coding rates over each class. Because features that are well-separated by class are easier to compress, the conditional coding rate tends to be lower when classes form distinct clusters.

## 3.2.6. Transferability Measurement with Intra-classfeature Variance (TMI)

TMI [36] focuses on generalization ability by calculating intra-class variance, whereas other measures focus on the clustering ability of pre-trained models, or:

$$
\mathrm { T M I } ( \boldsymbol { S } , \mathcal { T } ) = H ( \omega ( \mathbf { X } ) | \mathbf { Y } ) = \sum _ { y = 1 } ^ { C _ { T } } \frac { N _ { y } } { N } H ( \omega ( \mathbf { X } _ { y } ) ) ,\tag{12}
$$

where $H ( \omega ( \mathbf { X } ) | \mathbf { Y } )$ is the conditional entropy of the target feature $\omega ( \mathbf { X } )$ on the model pre-trained with S given the target labels $Y , \mathbf { X } _ { y }$ is the subset of X belonging to the label y, and $N _ { y }$ is the number of instances in class y.

## 3.2.7. Dataset Similarity Measure for Source Selection

Similar to the transferability estimation, dataset similarity measures are often used for source selection. The previous method, transferability estimation, is only useful when the pre-trained θ is available. However, unlike image recognition, there are fewer standard pre-trained weights available for time series recognition. Therefore, to find a suitable source with those transferability estimations for time series recognition, pre-training all available datasets is necessary, which incurs substantial computational cost.

By contrast, dataset distance measures only require information from the datasets. Fawaz et al. [2] demonstrated that Dynamic Time Warping (DTW) [37] can be used to compare prototypes of each time-series class in the target and source datasets. The prototypes are obtained by computing the average time series of each class found by DTW Barycenter Averaging (DBA) [38]. They defined the dataset distance as the distance between the most similar pairs of prototypes gained by DBA from each dataset [2]. Using the dataset distance measure for source selection helped to find an appropriate source dataset for the target task. A benefit of this method and our proposed method is that these methods do not require a trained model to predict transferability.

## 4. Multi-Source Transfer Learning

We propose a multi-source transfer learning framework that aggregates the top-K source datasets selected by the shapelet-based similarity measure into a single unified pre-training dataset, denoted as $\boldsymbol { S _ { \mathrm { M u l i } } }$ . Let $S _ { \mathrm { s e l } } = \{ S _ { \pi _ { 1 } } , . . . , S _ { \pi _ { K } } \}$ denote the selected source datasets ranked by their similarity to the target dataset T. An overview of the framework is shown in Fig. 1, and the complete end-to-end procedure is summarized in Algorithm 1.

After source selection, the retained datasets undergo several pre-processing steps to enable multi-source pre-training. First, all selected source datasets in $\boldsymbol { S } _ { \mathrm { s e l } }$ are resampled to match the temporal length of the target dataset T. This operation ensures that the neural network input dimensionality remains consistent during multi-source pretraining. If the target sequence length exceeds that of a source dataset, interpolationbased upsampling is applied to expand the source sequences to the target length. However, since the primary objective of pre-training is to obtain efective initial weights for the target domain, the potential loss of source-specific characteristics is acceptable. Second, the aggregated dataset $S _ { \mathrm { M u l i } }$ is balanced to ensure that each constituent source dataset $S _ { i }$ contributes an equal number of time series samples. This is achieved via oversampling while maintaining the original class distribution within each source. Finally, to ensure a fair comparison across experiments with varying sizes of $S _ { \mathrm { M u l t i } }$ the model is trained for a fixed number of iterations rather than for a fixed number of epochs.

Algorithm 1 End-to-end multi-source transfer learning with Shapelet Matching   
Input: Target dataset $T ,$ candidate source datasets $S = \{ S _ { 1 } , \ldots , S _ { I } \} .$ , number of se  
lected sources $K ,$ shapelet length $\ell _ { s } ,$ number of shapelets per class $Q ,$ backbone   
$f _ { \theta } ,$ pre-training iterations $N _ { \mathrm { p r e } } .$ , fine-tuning iterations $N _ { \mathrm { f t } }$   
Output: Fine-tuned target model $g _ { \psi } \circ f _ { \theta }$   
1: $P ^ { ( T ) }$ ← DiscoverShapeletsMP(T $\ell _ { s } , Q )$   
2: for $i \gets 1$ to I do   
3: $P ^ { ( S _ { i } ) } \gets$ DiscoverShapeletsM $\mathrm { P } ( S _ { i } , \ell _ { s } , Q )$   
4: $d _ { i }$ ← ShapeletMatchingDistance $( P ^ { ( S _ { i } ) } , P ^ { ( T ) } )$   
5: end for   
6: $\pi  \mathrm { A }$ rgsortAscending $( d _ { 1 } , \ldots , d _ { I } )$   
7: $S _ { K } \gets \{ S _ { \pi _ { 1 } } , \ldots , S _ { \pi _ { K } } \}$   
8: ℓ ← TemporalLength(T)   
9: $\hat { S } _ { K } \gets \emptyset$   
10: for $S \in S _ { K }$ do   
11: $\hat { S } _ { K } \gets \hat { S } _ { K } \cup$ {ResizeToLength $( S , \ell _ { T } ) \}$   
12: end for   
13: $M ^ { \star } \gets \operatorname* { m a x } _ { \hat { S } \in \hat { S } _ { K } } | \hat { S } |$   
14: $S _ { \mathrm { M u l t i } }  \emptyset$   
15: $o \gets 0$   
16: for $\hat { S } \in \hat { S } _ { K }$ do   
17: S<sup>˜</sup> ← OversamplePreservingClassRatio(S<sup>ˆ</sup> M<sup>⋆</sup>)   
18: $( \tilde { X } , \tilde { Z } , o )$ ← RelabelWithOffset(S<sup>˜</sup> o)   
19: $S _ { \mathrm { M u l t i } }  S _ { \mathrm { M u l t i } } \cup ( \tilde { X } , \tilde { Z } )$   
20: end for   
21: $C _ { \mathrm { M u l t i } }  | \{ z \mid ( \mathbf { x } , z ) \in S _ { \mathrm { M u l t i } } \} |$   
22: $( f _ { \theta } , h _ { \phi } )$ ← InitializeModel $. ( C _ { \mathrm { M u l t i } } )$   
23: $( f _ { \theta } , h _ { \phi } )$ ← TrainForIterations(f<sub>θ,</sub> h<sub>ϕ,</sub> S <sub>Multi,</sub> $N _ { \mathrm { p r e } } )$   
24: $C _ { T } \gets | \{ y \mid ( \mathbf { x } , y ) \in T \} |$   
25: g ← InitializeTargetHead(C )   
26: $( f _ { \theta } , g _ { \psi } ) $ TrainForIterations $( f _ { \theta } , g _ { \psi } , T , N _ { \mathrm { f t } } )$   
27: return $g _ { \psi } \circ f _ { \theta }$   
12 12

After preprocessing, the selected source datasets in $\boldsymbol { S } _ { \mathrm { s e l } }$ are merged by modifying the label space. Specifically, the corresponding label sets from each dataset, $\mathbf { Z } _ { \pi 1 } , \ldots , \mathbf { Z } _ { \pi k } , \ldots , \mathbf { Z } _ { \pi K }$ , are concatenated to form a single, unified label set $\mathbf { Z _ { \mathrm { M u l t i } } }$ . Consequently, the output layer of the network is extended to accommodate this expanded set of classes. The network is then pre-trained on the aggregated dataset $S _ { \mathrm { M u l t i } }$ using the unified labels $\mathbf { Z _ { \mathrm { M u l t i } } }$ . We emphasize that this aggregation is used to learn transferable feature extractors rather than to merge class semantics across domains. Each source label remains dataset-specific after relabeling, and pre-training exposes the network to a broader collection of discriminative temporal patterns. The benefit of aggregation is therefore not semantic equivalence across sources, but improved initialization from diverse yet target-relevant sources. This strategy enables the pre-training of a model on a substantially larger and more diverse dataset than would be possible with a single source.

After multi-source pre-training, the multi-source classifier head is replaced with a target-specific classifier head, and the transferred backbone is fine-tuned on the target task following the standard transfer learning procedure. Although the source-selection score can be computed independently of the downstream backbone, its empirical effectiveness may still depend on the model architecture. This architecture-dependent behavior is analyzed in Section 7.3 and discussed as a limitation in Section 8.

## 5. Shapelet Similarity-based Source Selection

It is essential to select a number of source datasets for the proposed multi-source transfer learning. However, selecting proper source datasets is not a straightforward task. Thus, for source selection, we propose to use a novel shapelet-based similarity measure instead of using conventional transferability measures.

## 5.1. Shapelet

A shapelet is a subsequence of time series data that is highly discriminative for a given class [39]. Shapelets imply class-representative patterns or features of each

class.

## 5.2. Shapelet Discovery

From a time series, a shapelet can be any subsequence depending on its definition. Thus, sometimes finding the most discriminative shapelet can be computationally expensive. In this research, we used matrix profile [9] to find discriminative shapelets for eficient shapelet discovery.

Matrix profile represents a time series as distances between subsequences and their nearest neighbor. Matrix profile p of given time series t and its list of all subsequences $\mathcal { A }$ is a sequence of the distances between every subsequence ${ \mathcal { A } } _ { r }$ to its nearest neighbor, or:

$$
\mathbf { p } = \parallel \mathcal { A } _ { 1 } - \mathcal { E } _ { 1 } \parallel , . . . , \parallel \mathcal { A } _ { r } - \mathcal { E } _ { r } \parallel , . . . , \parallel \mathcal { A } _ { R } - \mathcal { E } _ { R } \parallel .\tag{13}
$$

Here, $\mathcal { E } _ { r }$ denotes the subsequence of A closest to the corresponding subsequence ${ \mathcal { A } } _ { r } ,$ and ∥ · ∥ denotes the Euclidean norm. By computing the matrix profile p from the time series t, motifs and discords can be eficiently identified.

To apply matrix profile for shapelet discovery, we introduce several modifications. Given a dataset S, we first concatenate all time series belonging to each class c into a single class-specific time series $\mathbf { t } ^ { ( c ) }$ . For instance, if there are two classes (e.g., class 1 and class 2), two concatenated series $\mathbf { t } ^ { ( 1 ) }$ and $\mathbf { t } ^ { ( 2 ) }$ are created. Next, instead of calculating the matrix profile solely using nearest neighbors within the same class (as in (13)), we calculate matrix profiles across all class combinations: $\mathbf { p } ^ { ( 1 , 1 ) } , \mathbf { p } ^ { ( 1 , 2 ) } , \mathbf { p } ^ { ( 2 , 2 ) }$ and $\mathbf { p } ^ { ( 2 , 1 ) }$ . Subsequently, the largest values in the diference profiles $\mathbf { p } ^ { ( 1 , 2 ) } - \mathbf { p } ^ { ( 1 , 1 ) }$ and $\mathbf { p } ^ { ( 2 , 1 ) } - \mathbf { p } ^ { ( 2 , 2 ) }$ correspond to the most representative shapelet candidates for class 1 and class 2, respectively. As this procedure inherently supports binary classification, we generalize the approach for multi-class scenarios by employing a one-versus-all strategy to identify distinctive shapelets for each class separately. In implementation, time series from the same class are concatenated with boundary separators before matrix profile computation. Windows yielding invalid matrix-profile values near sequence boundaries are assigned invalid scores and deprioritized during candidate ranking, which prevents candidates from spanning two original series.

## 5.3. Source Selection based on Shapelet Similarity

We propose the shapelet-based similarity measures using Shapelet Matching for source selection, each of which compares the class-representative shapelets $\mathcal { P } ^ { ( S ) }$ and $\mathcal { P } ^ { ( \mathcal { T } ) }$ from the source and target datasets, respectively. Notably, the other two shapeletbased source selection schemes, Minimum Shapelet and Average Shapelet, are from the preliminary conference paper of this work [27]. Average Shapelet summarizes similarity across all source–target shapelet pairs, Minimum Shapelet emphasizes the closest local match, and Shapelet Matching balances these extremes by aggregating multiple greedy matches. Figure 2 illustrates how each scheme handles pairing $\mathcal { P } ^ { ( S ) }$ and $\mathcal { P } ^ { ( \mathcal { T } ) }$ . All three schemes rely on pairwise Euclidean distances but adopt distinct strategies to match and average those distances, leading to diferent similarity scores.

![](images/878741563588c6453af01a6bd6740ac311c21222fbb2420784304f7ab5617fd0.jpg)  
Figure 2: Comparison of the three shapelet-based dataset-similarity measures. Green boxes denote shapelets extracted from a source dataset, and blue boxes denote shapelets extracted from the target dataset. Red connections indicate the source–target shapelet pairs whose Euclidean distances are averaged to compute the dataset distance, whereas dashed black connections indicate candidate pairs that are not used in the final aggregation. Average Shapelet averages all possible pairs, Minimum Shapelet uses only the closest pair, and Shapelet Matching greedily selects multiple non-overlapping closest pairs

Average Shapelet measures the shapelet similarity from the mean distance of all

possible pairs of $\mathcal { P } _ { i } ^ { ( S ) }$ and $\mathcal { P } _ { j } ^ { ( \mathcal { T } ) }$ , or:

$$
D _ { \mathrm { a s } } = \underset { i , j } { \mathbb { E } } ( \Vert \mathcal { P } _ { i } ^ { ( S ) } - \mathcal { P } _ { j } ^ { ( \mathcal { T } ) } \Vert ) ,\tag{14}
$$

where $\mathcal { P } _ { i } ^ { ( S ) }$ and $\mathcal { P } _ { j } ^ { ( \mathcal { T } ) }$ are the i-th and j-th shapelet of $\mathcal { P } ^ { ( S ) }$ and $\mathcal { P } ^ { ( \mathcal { T } ) }$ , respectively.

Minimum Shapelet is inspired by the DBA-DTW [2] that measures DBA-based similarity with its most similar pair. This measures the shapelet similarity by measuring the distance of the most similar pair between source and target shapelets, $\mathcal { P } _ { i } ^ { ( S ) }$ and $\mathcal { P } _ { j } ^ { ( \mathcal { T } ) }$ , or:

$$
D _ { \mathrm { m s } } = \operatorname* { m i n } _ { i , j } \parallel \mathcal { P } _ { i } ^ { ( S ) } - \mathcal { P } _ { j } ^ { ( \mathcal { T } ) } \parallel .\tag{15}
$$

Shapelet Matching is defined using a greedy matching approach. The primary motivation for Shapelet Matching is to balance the sparse matching of Minimum Shapelet and the dense matching of Average Shapelet. Shapelet Matching iteratively selects the closest pairs based on a pre-computed distance matrix between two sets of shapelets. Given a set of source shapelets and a set of target shapelets, the algorithm identifies and matches pairs based on their Euclidean distance between each pair of shapelets, ensuring that each target shapelet is matched exactly once, while each entry in the expanded source-shapelet list is used at most once.

When the number of source shapelets is smaller than the number of target shapelets, Algorithm 2 repeats the source shapelet list before greedy matching. The duplicated entries are treated as distinct matching candidates, which allows every target-side discriminative shapelet to contribute to the final distance.

## 5.4. Comparison between Dataset Similarity Metrics and Transferability Estimation Methods

As summarized in Table 1, source-selection methods can be divided into transferability estimation methods and dataset similarity metrics. Transferability estimation methods are model dependent: each new backbone or source dataset requires sourcemodel pre-training, and each new target must be evaluated through the trained source models. In our setting, this required about 36,000 s for one new backbone, 280 s for one new source dataset, and 170 s for one new target dataset.

Algorithm 2 Shapelet Matching Distance   
Input: $\mathcal { P } ^ { ( S ) } = \{ s _ { 1 } , . . . , s _ { m } \}$ (source shapelets), $\mathcal { P } ^ { ( \mathcal { T } ) } = \{ t _ { 1 } , \ldots , t _ { n } \}$ (target shapelets)   
Output: Matched triples M and mean pairwise distance $\bar { d }$   
1: $m  \vert \mathcal { P } ^ { ( S ) } \vert , n  \vert \mathcal { P } ^ { ( \mathcal { T } ) } \vert$   
2: if $m < n$ then   
3: $r \gets \lceil n / m \rceil$   
4: $\widetilde { \mathcal { P } } ^ { ( S ) } \gets \mathrm { R e p e a t } ( \mathcal { P } ^ { ( S ) } , r )$ ▷ keep duplicated shapelets as distinct entries   
5: else   
6: $\smash { \widetilde { \mathcal { P } } ^ { ( S ) } \gets \mathcal { P } ^ { ( S ) } }$   
7: end if   
8: $\widetilde { \mathcal { P } } ^ { ( S ) } = \{ \tilde { s } _ { 1 } , \hdots , \tilde { s } _ { m ^ { \prime } } \} .$ , where $m ^ { \prime } = | \widetilde { \mathcal { P } } ^ { ( S ) } |$   
9: $I \gets \{ 1 , \ldots , m ^ { \prime } \}$ ▷ active source-shapelet indices   
10: $J  \{ 1 , \ldots , n \}$ ▷ active target-shapelet indices   
11: $M \gets [ ]$ ▷ matched triples   
12: $D \gets \left[ | | \tilde { s } _ { i } - t _ { j } | | _ { 2 } \right] _ { i = 1 , \dots , m ^ { \prime } ,  j = 1 , \dots , n }$ ▷ pre-computed distance matrix   
13: while $J \neq \emptyset$ do   
14: $( i ^ { \star } , j ^ { \star } ) \gets \arg \operatorname* { m i n } _ { i \in I , j \in J } D _ { i , j }$   
15: append $\left( M , ( i ^ { \star } , j ^ { \star } , D _ { i ^ { \star } , j ^ { \star } } ) \right)$   
16: $I  I \setminus \{ i ^ { \star } \}$   
17: $J  J \setminus \{ j ^ { \star } \}$   
18: end while   
19: $\bar { d }  \frac { 1 } { | M | } \sum d$   
(i<sub>,</sub> j<sub>,</sub>d)∈M   
20: return $( M , { \bar { d } } )$

Dataset similarity metrics avoid source-model pre-training because they compare source and target datasets directly. DBA-DTW is also training-free, but its target-side comparison is costly because DBA prototypes must be compared with DTW, requiring about 25,000 s per new target in our setting. The proposed shapelet-based metrics require one-time shapelet extraction, about 98 s per new source dataset and 110 s per new target dataset, after which source–target scores are computed using Euclidean distances between discovered shapelets. Thus, the ranking can be computed once without training a backbone and reused as an input to diferent architectures, although the downstream benefit of that ranking remains architecture-dependent.

Table 1: Comparison of source-selection methods in terms of training requirement and average computational time (seconds). The last three columns report the incremental cost of adding a new backbone architecture, one candidate source dataset, or one target dataset under our experimental setting. Actual computation times may vary across datasets.
<table><tr><td>Method</td><td>Type</td><td>Training</td><td>Per New Model</td><td>Per New Source Dataset</td><td>Per New Target Dataset</td></tr><tr><td colspan="6">Transferability Estimation</td></tr><tr><td>H-score</td><td>Feature-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td>LEEP</td><td>Prediction-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td>LogME</td><td>Feature-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td>NCE</td><td>Prediction-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td>TransRate</td><td>Feature-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td>TMI</td><td>Feature-based</td><td>√</td><td>36,000 s (GPU)</td><td>280 s (GPU)</td><td>170 s (GPU)</td></tr><tr><td colspan="6">Dataset Similarity Metrics</td></tr><tr><td>DBA-DTW</td><td>Global similarity</td><td>x</td><td></td><td>52 s (CPU only)</td><td>25,000 s (CPU only)</td></tr><tr><td>Avg. Shapelet (Ours)</td><td>Local similarity</td><td>x</td><td></td><td>98 s (CPU only)</td><td>110 s (CPU only)</td></tr><tr><td>Min. Shapelet (Ours)</td><td>Local similarity</td><td>x</td><td></td><td>98 s (CPU only)</td><td>110 s (CPU only)</td></tr><tr><td>Proposed SM (Ours)</td><td>Local similarity</td><td>x</td><td></td><td>98 s (CPU only)</td><td>110 s (CPU only)</td></tr></table>

## 6. Experimental Results

## 6.1. Dataset

We evaluated the proposed method using 128 univariate time series datasets from the UCR Archive [10]<sup>2</sup>. We use the division of training and test sets as determined by the archive. Except for the temporal-length resizing described in Section 4, we used the archive values as released and did not apply additional z-normalization. Importantly, the UCR 2018 archive is not uniformly normalized: the legacy Summer 2015 datasets are z-normalized, whereas the Fall 2018 additions are kept in their original scale unless they were already normalized by the data donor. Therefore, the normalization status depends on the dataset. Table 2 shows characteristics of datasets in UCR Archive.

Table 2: Summary of the 128 univariate UCR datasets used in the experiments.
<table><tr><td>Property</td><td>Min</td><td>Median</td><td>Max</td></tr><tr><td>Series length</td><td>15</td><td>344</td><td>2,844</td></tr><tr><td>Number of classes</td><td>2</td><td>4</td><td>60</td></tr><tr><td>Training samples</td><td>16</td><td>190.5</td><td>8,926</td></tr><tr><td>Test samples</td><td>20</td><td>316</td><td>16,800</td></tr></table>

## 6.2. Settings and Architecture

We evaluated our proposed method on three representative architectures for time series classification: CNNs, ViT and PatchTST. For CNNs, we adopted the VGG [40]- based architecture with three blocks of convolutional layers and pooling layers. Max pooling follows the first two blocks, and global average pooling (GAP) follows the final block. We used GAP to obtain a fixed-dimensional representation across datasets of diferent temporal lengths and to provide a common interface for the evaluated transferability estimators.

We implemented two patch-based Transformer models: ViT, originally introduced for image recognition [11], and PatchTST, adapted to time-series tasks [12]. Each model’s architecture is configured with three Transformer encoder blocks, a model dimension of 16, four attention heads, and a feed-forward network dimension of 128. The input data is segmented into patches of length 16; ViT uses non-overlapping patches (stride 16), whereas PatchTST uses 50%-overlapping patches (stride 8). For the final classification task, a head consisting of a flatten layer followed by a linear classification layer with a softmax activation function is appended to the encoder output.

Table 3: Training and source-selection hyperparameters used in the experiments. The shapelet length and number of candidate shapelets follow the preliminary study, whereas $K \ : = \ : 1 6$ is specific to the present experiments. Fine-tuning was repeated three times from a fixed pre-trained checkpoint. A dash indicates that the setting is not applicable.
<table><tr><td>Setting</td><td>VGG</td><td>ViT</td><td>PatchTST</td></tr><tr><td>Pre-training iterations</td><td>10,000</td><td>20,000</td><td>20,000</td></tr><tr><td>Fine-tuning iterations</td><td>5,000</td><td>10,000</td><td>10,000</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Fine-tuning repetitions</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Patch size</td><td>一</td><td>16</td><td>16</td></tr><tr><td>Shapelet length  $\ell _ { s }$ </td><td>15</td><td>15</td><td>15</td></tr><tr><td>Candidate shapelets per class  $\boldsymbol { Q }$ </td><td>10</td><td>10</td><td>10</td></tr><tr><td>Selected sources K (multi-source)</td><td>16</td><td>16</td><td>16</td></tr></table>

Training and source-selection settings are summarized in Table 3. Following our preliminary study [27], we set the shapelet length to 15 and retained 10 candidate shapelets per class. The preliminary study used $K = 1 4$ , whereas the present study selected $K \ = \ 1 6$ as a representative operating point within the near-saturation region observed in the exploratory VGG analysis in Fig. 3. The same value was used across all source-selection methods and backbones. We do not claim that $K = 1 6$ is uniquely optimal, and comparisons at this setting should be interpreted in light of this exploratory choice.

## 6.3. Comparative Evaluation

We compare three learning strategies: no transfer learning, single-source transfer learning, and multi-source transfer learning with $K = 1 6$ selected sources. The source selection criteria include conventional transferability estimators (H-score [31],

LEEP [32], LogME [33], NCE [34], TransRate [35], and TMI [36]), the DBA-DTW dataset similarity baseline [2], and the proposed shapelet-based measures. Results are reported as mean ± standard deviation over three fine-tuning runs from a fixed pre-trained checkpoint. For both Tables 4 and 5, the 95% percentile bootstrap confidence intervals were computed using 10,000 resamples of the 128 target datasets. The three fine-tuning runs were averaged within each target dataset before resampling, and the same bootstrap resamples were used across all methods. For significance testing, Holm correction was applied in both Tables 4 and 5. The reported standard deviation is the average of the per-target standard deviations computed over these three runs; it is not the standard deviation of the mean accuracies across the 128 target datasets. The paired statistical unit is the target dataset, and corrected paired comparisons for the main K = 16 setting are reported in Table 6. Paired t-tests were conducted on the 128 dataset-level paired diferences after averaging the three runs within each dataset. We used the paired t-test as the primary analysis because the statistical quantity of interest is the mean paired diference and the number of paired datasets is relatively large (n = 128). To assess the robustness of the conclusions to the parametric assumptions of the t-test, we additionally performed dataset-level bootstrap confidence interval estimation and Wilcoxon signed-rank sensitivity analyses. These analyses yielded the same qualitative conclusions for the primary comparisons between Proposed SM and no transfer learning.

Tables 4 and 5 show two main trends. First, multi-source transfer learning generally improves over the corresponding single-source setting, indicating that aggregating several selected sources reduces dependence on a single potentially mismatched source. On VGG, the proposed Shapelet Matching (SM) obtains the best average multi-source accuracy, improving from the no-transfer baseline of 74.18% to 80.78%. Second, the best source-selection criterion depends on the backbone architecture. For ViT and PatchTST, DBA-DTW achieves the highest mean multi-source accuracy, whereas the proposed SM remains competitive. This result suggests that shapeletbased source selection is especially well aligned with the VGG-style CNN, while patch-based Transformer models may benefit more from global similarity criteria.

For Proposed SM, single-source transfer yielded lower accuracy than no transfer learning on 36, 52, and 60 of the 128 target datasets for VGG, ViT, and PatchTST, respectively. Under multi-source transfer, these numbers decreased to 21, 40, and 45, corresponding to net reductions of 15, 12, and 15 targets. Multi-source SM also outperformed the corresponding single-source setting on 82 target datasets for each backbone, although the mean improvement was not statistically significant for ViT. Thus, multi-source transfer improved average accuracy while reducing the risk of negative transfer.

Table 4: Average classification accuracy (%) across the 128 UCR target datasets for the VGG backbone. Single-source transfer uses the top-ranked source dataset, whereas multi-source transfer uses the top-K sources with $K \ = \ 1 6 .$ Each entry reports the across-target mean accuracy (±) the average within-target standard deviation over three fine-tuning runs, followed by a 95% percentile bootstrap confidence interval in brackets. Methods are grouped according to whether they require source-model training or use trainingfree dataset similarity. Boldface indicates the best result in each column, and asterisks indicate significance relative to no transfer learning based on Holm-adjusted paired t-tests.
<table><tr><td colspan="2">Method Single</td></tr><tr><td colspan="2">Baseline: No Transfer Learning  $7 4 . 1 8 \pm 3 . 6 0$ </td></tr><tr><td colspan="2"> $[ 7 0 . 7 8 , 7 7 . 5 1 ]$ </td></tr><tr><td>Transferability Metrics  $7 7 . 9 2 \pm 2 . 1 1 ^ { \ast \ast \ast }$ </td><td> $8 0 . 0 5 \pm 1 . 4 3 ^ { \ast \ast \ast }$ </td></tr><tr><td>H-score LEEP</td><td>[74.69, 80.96] [77.05, 82.88]  $7 7 . 7 3 \pm 2 . 0 5 ^ { ^ { \ast } { } ^ { \ast } { } ^ { \ast } { } ^ { \ast } { } }$   $8 0 . 1 2 \pm 1 . 6 0 ^ { ^ { \ast \ast \ast } }$ </td></tr><tr><td></td><td>[74.47, 80.85]  $[ 7 7 . 1 4 , 8 2 . 9 5 ]$   $7 9 . 1 7 \pm 1 . 8 9 ^ { \ast \ast \ast }$   $7 9 . 9 8 \pm 1 . 7 8 ^ { \ast \ast \ast }$ </td></tr><tr><td>LogME</td><td> $[ 7 5 . 9 3 , 8 2 . 1 5 ]$   $[ 7 6 . 9 3 , 8 2 . 8 0 ]$   $7 8 . 7 5 \pm 1 . 6 8 ^ { ^ { \ast \ast \ast } }$   $8 0 . 2 2 \pm 1 . 4 4 ^ { ^ { \ast \ast \ast } }$ </td></tr><tr><td>NCE</td><td> $[ 7 5 . 4 2 , 8 1 . 8 2 ]$   $[ 7 7 . 2 3 , 8 3 . 0 4 ]$   $7 7 . 0 3 \pm 1 . 9 0 ^ { * * }$   $7 9 . 6 8 \pm 1 . 6 1 ^ { * * }$ </td></tr><tr><td>TransRate TMI</td><td> $[ 7 3 . 6 7 , 8 0 . 1 8 ]$   $[ 7 6 . 6 7 , 8 2 . 4 9 ]$   $7 2 . 2 1 \pm 2 . 3 2$   $8 0 . 1 8 \pm 1 . 5 6 ^ { * * * }$   $[ 6 7 . 9 9 , 7 6 . 1 8 ]$  [77.25, 82.90]</td></tr><tr><td colspan="2">Global Dataset Similarity Metrics</td></tr><tr><td> $7 8 . 4 2 \pm 2 . 1 2 ^ { \ast \ast \ast }$  DBA-DTW  $[ 7 5 . 1 7 , 8 1 . 5 0 ]$ </td><td> $8 0 . 4 1 \pm 1 . 6 7 ^ { \ast \ast \ast }$   $[ 7 7 . 3 7 , 8 3 . 2 2 ]$ </td></tr><tr><td colspan="2">Local Shapelet Similarity Metrics</td></tr><tr><td>Avg. Shapelet (Ours)</td><td> $7 6 . 4 7 \pm 2 . 1 2 ^ { * * }$   $7 9 . 9 4 \pm 1 . 6 4 ^ { ^ { \ast \ast \ast } }$  [73.15, 79.60]  $[ 7 7 . 0 1 , 8 2 . 7 0 ]$ </td></tr><tr><td>Min. Shapelet (Ours)</td><td> $7 8 . 7 0 \pm 1 . 6 5 ^ { ^ { \ast \ast \ast } }$   $7 9 . 9 7 \pm 1 . 7 9 ^ { \ast \ast \ast }$   $[ 7 5 . 5 0 , 8 1 . 7 0 ]$   $[ 7 6 . 9 0 , 8 2 . 8 0 ]$ </td></tr><tr><td>Proposed SM (Ours)</td><td> $7 8 . 3 2 \pm 1 . 9 6 ^ { ^ { \ast \ast \ast } }$   $\mathbf { 8 0 . 7 8 \pm 1 . 6 9 } ^ { \mathbf { * * } }$  [75.18, 81.32] [77.87, 83.49]</td></tr></table>

\* p<sub>Holm</sub> < 0<sub>.</sub>05, \*\* p<sub>Holm</sub> < 0<sub>.</sub>01, \*\*\* p<sub>Holm</sub> < 0<sub>.</sub>001

Table 5: Average classification accuracy (%) across the 128 UCR target datasets for the ViT and PatchTST backbones. Single-source transfer uses the top-ranked source dataset, whereas multi-source transfer uses the top-K sources with $K = 1 6 .$ Each entry reports the across-target mean accuracy (±) the average withintarget standard deviation over three fine-tuning runs, followed by a 95% percentile bootstrap confidence interval in brackets. Methods are grouped according to whether they require source-model training or use training-free dataset similarity. Boldface indicates the best result within each backbone and transfer setting, and asterisks indicate significance relative to no transfer learning based on Holm-adjusted paired t-tests.
<table><tr><td></td><td colspan="2">ViT</td><td colspan="2">PatchTST</td></tr><tr><td>Method</td><td>Single</td><td>Multi</td><td>Single</td><td>Multi</td></tr><tr><td colspan="5">Baseline: No Transfer Learning  $7 0 . 2 2 \pm 2 . 0 1$  [66.44, 73.85]</td></tr><tr><td>Transferability Metrics</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>H-score</td><td> $6 7 . 0 6 \pm 3 . 4 6 ^ { * * }$   $[ 6 3 . 0 2 , 7 0 . 9 3 ]$ </td><td> $7 1 . 0 7 \pm 1 . 4 1$  [67.36, 74.55]</td><td> $6 6 . 3 9 \pm 4 . 2 4 ^ { * * * }$  [62.56, 70.06]</td><td> $7 5 . 2 0 \pm 1 . 7 1 ^ { \ast }$   $[ 7 1 . 9 2 , 7 8 . 3 6 ]$ </td></tr><tr><td>LEEP</td><td> $^ { 7 1 . 6 9 \pm 1 . 4 6 ^ { \ast \ast } } _ { [ 6 7 . 9 5 , 7 5 . 2 5 ] }$ </td><td> $^ { 7 1 . 0 9 } _ { [ 6 7 . 4 0 , 7 4 . 5 8 ] }$ </td><td> $7 5 . 6 5 \pm 1 . 9 2$   $[ 7 2 . 1 7 , 7 8 . 9 7 ]$ </td><td> $7 6 . 8 6 \pm 1 . 7 2 ^ { * * * }$  [73.58, 80.00]</td></tr><tr><td>LogME</td><td> $6 8 . 7 3 \pm 1 . 6 4 ^ { * }$  [65.02, 72.24]</td><td> $^ { 7 0 . 2 9 \pm 1 . 5 5 } _ { [ 6 6 . 4 7 , 7 3 . 9 0 ] }$ </td><td> $7 4 . 0 6 \pm 1 . 8 2$  [70.45, 77.42]</td><td> $7 5 . 5 2 \pm 1 . 9 9 ^ { \ast \ast }$ </td></tr><tr><td>NCE</td><td> $7 1 . 5 0 \pm 1 . 5 2 ^ { \ast }$ </td><td> $^ { 7 1 . 2 6 \pm 1 . 7 3 } _ { [ 6 7 . 4 5 , 7 4 . 8 4 ] }$ </td><td> $^ { 7 5 . 8 1 \pm 1 . 7 7 ^ { \ast } } _ { [ 7 2 . 4 1 , 7 9 . 0 6 ] }$ </td><td> $^ { 7 6 . 6 2 \pm 1 . 9 3 ^ { \ast \ast \ast } } _ { [ 7 3 . 3 8 , 7 9 . 7 1 ] }$ </td></tr><tr><td>TransRate</td><td> $^ { 7 0 . 3 3 \pm 1 . 6 5 } _ { [ 6 6 . 5 9 , 7 3 . 9 0 ] }$ </td><td> $7 1 . 7 5 \pm 1 . 4 6 ^ { \ast \ast }$ </td><td> $7 5 . 1 8 \pm 1 . 8 4$  [71.82, 78.40]</td><td> $^ { 7 6 . 9 1 \pm 1 . 6 9 ^ { \ast \ast } } _ { [ 7 3 . 6 8 , 7 9 . 9 9 ] }$ </td></tr><tr><td>TMI</td><td> $6 8 . 3 0 \pm 1 . 8 6 ^ { \ast }$   $[ 6 4 . { \bar { 5 } } 0 , 7 1 . 8 { \bar { 8 } } ]$ </td><td> $^ { 7 0 . 7 8 \pm 1 . 5 1 } _ { [ 6 7 . 0 6 , 7 4 . 3 2 ] }$ </td><td> $7 2 . 9 6 \pm 2 . 1 0$   $[ 6 9 . 6 3 , 7 6 . 1 7 ]$ </td><td> $7 5 . 8 5 \pm 1 . 6 9 ^ { \ast }$   $[ 7 2 . 5 5 , 7 8 . 9 7 ]$ </td></tr><tr><td colspan="5">Global Dataset Similarity Metrics</td></tr><tr><td>DBA-DTW</td><td> $7 1 . 9 4 \pm 1 . 5 9 ^ { \ast \ast }$  [68.31, 75.42]</td><td> $^ { 7 2 . 3 8 \pm 1 . 7 2 ^ { \ast \ast } } _ { [ 6 8 . 7 6 , 7 5 . 7 7 ] }$ </td><td> $^ { 7 5 . 8 3 \pm 1 . 8 6 ^ { \ast } } _ { [ 7 2 . 4 8 , 7 9 . 0 9 ] }$ </td><td> $^ { 7 7 . 3 0 \pm 1 . 5 9 ^ { \ast \ast } } _ { [ 7 4 . 0 4 , 8 0 . 3 6 ] }$ </td></tr><tr><td colspan="5">Local Shapelet Similarity Metrics</td></tr><tr><td>Avg. Shapelet (Ours)</td><td> $6 9 . 8 1 \pm 1 . 5 2$   $[ 6 6 . 1 0 , 7 3 . 3 8 ]$ </td><td> $^ { 7 1 . 2 2 \pm 1 . 4 8 ^ { * } } _ { [ 6 7 . 5 2 , 7 4 . 7 4 ] }$ </td><td> $^ { 7 3 . 2 6 \pm 2 . 1 5 } _ { [ 6 9 . 8 7 , 7 6 . 5 4 ] }$ </td><td> $7 6 . 4 8 \pm 1 . 6 8 ^ { ^ { \ast \ast } }$  [73.19, 79.67]</td></tr><tr><td>Min. Shapelet (Ours)</td><td> $7 0 . 6 5 \pm 1 . 3 9$  [67.00, 74.15]</td><td> $^ { 7 1 . 0 9 \pm 1 . 5 3 } _ { [ 6 7 . 4 1 , 7 4 . 5 6 ] }$ </td><td> $7 5 . 7 5 \pm 1 . 8 6$  [72.48, 78.92]</td><td> $^ { 7 6 . 2 2 \pm 1 . 7 8 ^ { \ast \ast } } _ { [ 7 2 . 9 7 , 7 9 . 3 4 ] }$ </td></tr><tr><td>Proposed SM (Ours)</td><td> $7 1 . 1 6 \pm 1 . 5 0 ^ { * }$  [67.50, 74.59]</td><td> $^ { 7 1 . 8 8 \pm 1 . 6 9 ^ { \ast \ast } } _ { [ 6 8 . 2 2 , 7 5 . 3 6 ] }$ </td><td> $7 5 . 4 3 \pm 1 . 6 2$   $[ 7 2 . 0 8 , 7 8 . 5 7 ]$ </td><td> $7 6 . 6 5 \pm 1 . 9 2 ^ { ^ { \ast \ast \ast } }$  [73.37, 79.80]</td></tr></table>

\* p<sub>Holm</sub> < 0<sub>.</sub>05, \*\* p<sub>Holm</sub> < 0<sub>.</sub>01, \*\*\* p<sub>Holm</sub> < 0<sub>.</sub>001

Table 6: Paired accuracy comparisons across 128 UCR target datasets. For each backbone, diferences are computed per target dataset as Proposed (K = 16, multi-source transfer with Shapelet Matching) minus the comparator and then summarized across datasets. CIs are percentile-bootstrap 95% intervals, and p-values are from two-sided paired t-tests with Holm adjustment applied jointly across all 12 comparisons shown (four comparisons for each of the three backbones). W/L/T denotes wins, losses, and ties.
<table><tr><td>Comparison</td><td>Mean ∆ acc. (pp)</td><td>95% CI</td><td>Holm p</td><td>Cohen&#x27;s  $d _ { z }$ </td><td>Rank-biserial</td><td>W/L/T</td></tr><tr><td>VGG</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Proposed - No TL</td><td>6.60</td><td>[5.03, 8.26]</td><td>&lt;0.001</td><td>0.71</td><td>0.80</td><td>103/21/4</td></tr><tr><td>Proposed - DBA-DTW</td><td>0.37</td><td>[-0.11, 0.87]</td><td>0.462</td><td>0.13</td><td>0.15</td><td>62/54/12</td></tr><tr><td>Proposed - Minimum Shapelet</td><td>0.81</td><td>[0.31, 1.36]</td><td>0.021</td><td>0.27</td><td>0.25</td><td>71/48/9</td></tr><tr><td>Proposed - Single-source (SM)</td><td>2.47</td><td>[1.44, 3.64]</td><td>&lt;0.001</td><td>0.39</td><td>0.53</td><td>82/38/8</td></tr><tr><td>ViT</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Proposed - No TL</td><td>1.66</td><td>[0.84, 2.50]</td><td>0.001</td><td>0.35</td><td>0.46</td><td>81/40/7</td></tr><tr><td>Proposed - DBA-DTW</td><td>-0.50</td><td>[-1.28, 0.18]</td><td>0.462</td><td>-0.12</td><td>-0.05</td><td>63/59/6</td></tr><tr><td>Proposed - Minimum Shapelet</td><td>0.79</td><td>[0.22, 1.34]</td><td>0.042</td><td>0.24</td><td>0.40</td><td>83/36/9</td></tr><tr><td>Proposed - Single-source (SM)</td><td>0.72</td><td>[-0.14, 1.53]</td><td>0.462</td><td>0.15</td><td>0.36</td><td>82/40/6</td></tr><tr><td>PatchTST</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Proposed - No TL</td><td>2.79</td><td>[1.52, 4.26]</td><td>0.001</td><td>0.35</td><td>0.43</td><td>77/45/6</td></tr><tr><td>Proposed - DBA-DTW</td><td>-0.65</td><td>[-1.42, 0.13]</td><td>0.462</td><td>-0.14</td><td>-0.15</td><td>60/62/6</td></tr><tr><td>Proposed - Minimum Shapelet</td><td>0.43</td><td>[-0.14, 1.02]</td><td>0.462</td><td>0.13</td><td>0.16</td><td>69/53/6</td></tr><tr><td>Proposed - Single-source (SM)</td><td>1.22</td><td>[0.47, 1.97]</td><td>0.015</td><td>0.28</td><td>0.42</td><td>82/41/5</td></tr></table>

Note. As descriptive benchmarks, absolute Cohen’s d<sub>z</sub> values of approximately 0.2, 0.5, and 0.8 are conventionally considered small, medium, and large, respectively.

Table 6 provides the corrected paired analysis for the main claims. The proposed multi-source SM significantly improves over no transfer learning for all three backbones after Holm correction. The efect is largest for VGG $( \Delta = 6 . 6 0$ percentage points, 95% CI [5.03, 8.26], $p \ < \ 0 . 0 0 1$ , Cohen’s $d _ { z } ~ = ~ 0 . 7 1 )$ , and smaller but still significant for ViT and PatchTST. The $\mathrm { W } / \mathrm { L } / \mathrm { T }$ counts further show that the VGG improvement is broadly distributed across target datasets.

The comparison with alternative selection strategies gives a more critical view. On VGG, the proposed SM significantly improves over Minimum Shapelet, supporting the benefit of target-guided dense shapelet matching. However, the diference from DBA-DTW is not statistically significant for any backbone, and DBA-DTW obtains slightly higher mean accuracy on the two Transformer-based models. Therefore, the proposed method should not be interpreted as universally superior in accuracy. Its main advantage is a training-free, inspectable source-selection criterion whose rankings can be reused across backbones.

## 7. Discussion

## 7.1. Efect of Number of Source Datasets in Multi-Source Transfer Learning

![](images/c0c75fd680379d79912238793978f64d78d0a3e579537647a29a45f17ef9e42e.jpg)  
Figure 3: Efect of the number K of selected source datasets on VGG multi-source transfer learning. The y-axis reports the mean classification accuracy over the 128 UCR target datasets, and the dashed horizontal line indicates the no-transfer baseline. Error bars indicate the average per-target standard deviation over three fine-tuning runs.

Fig. 3 shows that performance enters a near-saturation region at approximately $K = 1 4 .$ . Although the exact accuracy fluctuates across neighboring values, increasing K beyond this region provides only marginal additional benefit. Under the fixed training budget, increasing K beyond this region yields limited additional accuracy gains in the evaluated setting.

## 7.2. Connecting Minimum and Shapelet Matching Approaches

![](images/2faace996a34075535e665f4409e6980db16aaf57c5f43324d365d71903da5f7.jpg)  
Figure 4: Performance transition from Minimum Shapelet to Shapelet Matching on the VGG backbone under the $K = 1 6$ multi-source setting. The point at threshold 0 denotes the Minimum Shapelet, which uses the single closest shapelet pair. For each positive threshold T, the score averages the smallest ⌊(T/100)M⌋ distances among the M one-to-one pairs produced by the greedy matching procedure; at threshold 100, all matched pairs are included, yielding full Shapelet Matching as described in Algorithm 2. For each run, classification accuracy is averaged over the 128 UCR target datasets; diamonds and error bars show the mean and standard deviation across three runs, respectively. The gray dotted curve represents LOWESS smoothing (frac = 1 0, three robust reweighting iterations) and is included only as a visual guide.

Fig. 4 presents a sensitivity analysis of the matching-coverage threshold, which controls the extent of source–target shapelet matching. Threshold 0 retains only the single closest pair and thus corresponds to Minimum Shapelet, whereas threshold 100 uses the full set of pairs obtained through the greedy matching. The mean accuracies at intermediate thresholds are non-monotonic. The endpoint accuracy increases from 79.97% for Minimum Shapelet to 80.78% for full Shapelet Matching. The highest observed mean is 80.82% at threshold 70, only 0.04 percentage points above full matching. We use full matching in the main experiments without introducing an additional coverage-tuning parameter. The LOWESS curve is included only as a visual guide.

## 7.3. Architecture-dependent Performance

Fig. 5 shows the dataset-wise comparisons with no transfer learning for the VGG and ViT backbones. The benefit of the proposed multi-source transfer pipeline varies across the evaluated backbones. Relative to no transfer learning, Proposed SM yields mean accuracy gains of 6.60, 1.66, and 2.79 percentage points for VGG, ViT, and

![](images/4a99bc7997e910c6093abb845163b14a7cd2addc8f39b32aab83578d4c2a4315.jpg)

![](images/d94df1648c34e204a522d7856274b30faf74bab973c90d32abdcb953a3f85c3c.jpg)  
Figure 5: Dataset-wise comparison between the proposed Shapelet Matching multi-source transfer learning and no transfer learning. Each point represents one UCR target dataset and reports the average accuracy over three fine-tuning runs. The x-axis shows the no-transfer accuracy, and the y-axis shows the accuracy obtained by the proposed method; points above the diagonal indicate improvement and points below the diagonal indicate degradation. Red points indicate wins for the proposed method, and the win/loss/draw counts summarize the 128 target datasets for each displayed backbone.

PatchTST, respectively, with win/loss/tie counts of 103/21/4, 81/40/7, and 77/45/6 (Table 6). The largest observed gain is therefore obtained with the VGG-based CNN. However, Proposed SM does not significantly outperform DBA-DTW on any of the three backbones.

One plausible explanation is that shapelet-based ranking emphasizes local discriminative subsequences that may be particularly compatible with the VGG backbone. However, the present experiments do not isolate this mechanism or establish that patch tokenization causes the smaller gains observed for the Transformers. These findings describe architecture-dependent behavior under the evaluated configurations and should not be generalized to all CNN or Transformer architectures.

## 7.4. Dataset-Level Trends and Failure Cases

The efectiveness of the proposed shapelet-based transfer learning method depends on the dataset–method fit, especially whether discriminative and class-representative local subsequences can be extracted. To examine this dependence, we analyze normalized PSD-based spectral entropy and qualitative examples of the discovered shapelets.

![](images/17bc017d211261114d333bb92231fe7a416d6db92de78e41713d5c7ce2a7a03c.jpg)  
Figure 6: Accuracy gain of Shapelet Matching over no transfer learning versus normalized spectral entropy across 128 UCR datasets (VGG, K = 16). Each point represents a target dataset; the red line and shading show the linear regression fit and its 95% confidence interval. Lower (higher) entropy indicates spectral power concentrated in fewer (spread across more) frequencies.

![](images/93fe738f2d65cf792a681cf65ebf20facd64a9029571b5a572b18258a8aa0dac.jpg)  
Figure 7: Illustrative examples of discriminative shapelets extracted from datasets with large improvement or degradation under the proposed method. Each small plot shows the discovered shapelets for one class of the corresponding UCR dataset. Panel (a) shows datasets where Shapelet Matching improved over no transfer learning, whereas panel (b) shows datasets where it degraded. These examples suggest that the proposed method is more efective when compact local class-discriminative subsequences are visible, and less efective when such local patterns are weak or ambiguous.

For each time series in the training split of each target dataset, we estimated the PSD using Welch’s method, excluded the DC bin, and normalized the PSD values at the remaining frequency bins to sum to one. Let B denote the number of retained frequency bins and $p _ { b }$ the normalized PSD value at bin b. We then computed the normalized spectral entropy as $\begin{array} { r } { H _ { \mathrm { n o r m } } = - \frac { \sum _ { b = 1 } ^ { B } p _ { b } \log p _ { b } } { \log B } } \end{array}$ <sub>,</sub> and averaged $H _ { \mathrm { n o r m } }$ across the training series for each target dataset.

Fig. 6 shows the relationship between spectral entropy and the dataset-wise accuracy gain of Shapelet Matching over no transfer learning under the VGG multi-source setting with $K = 1 6$ . The association is moderately negative, with Pearson $r = - 0 . 3 3$ and Spearman $\rho ~ = ~ - 0 . 3 2$ , suggesting that larger gains tend to occur when target datasets have more concentrated and regular spectral structure. However, the presence of outliers indicates that spectral entropy is only a partial predictor of transfer performance.

The qualitative examples in Fig. 7 support this interpretation. Datasets with large positive gains often exhibit compact and visually consistent shapelets, whereas datasets with degradation tend to show weak, ambiguous, or poorly localized patterns. These cases suggest that the target decision boundary may depend more on global shape, long-range temporal structure, or heterogeneous signal characteristics than on compact local subsequences.

Shapelet Matching tends to be more beneficial when the target exhibits clear spectral structure and extractable local subsequences, but spectral entropy alone is insufficient to predict transfer performance. When the data are spectrally complex or the discriminative information is not well represented by local shapelets, the method may select sources that appear locally similar but do not provide useful initialization for the target task.

## 7.5. Relation to Recent Time Series Foundation Models

Recent time-series foundation models shift the focus from dataset-specific transfer to large-scale pre-training and adaptation. Our method addresses a diferent problem: given a target classification dataset and a finite pool of heterogeneous labeled sources, it identifies sources that contain target-relevant local discriminative subsequences. Its advantages are that it requires no separate source-model training, produces source rankings reusable across backbones, and remains potentially interpretable through matched shapelets. These properties are useful when large-scale foundation-model pre-training or released checkpoints are unavailable. However, Shapelet Matching does not learn a universal representation, does not provide zero-shot inference, and depends on the existence of transferable local shapelets. It may therefore be less suitable when the target task is governed mainly by long-range temporal structure, global shape, or multivariate interactions. Future work should investigate how shapelet-based source ranking can support foundation-model adaptation, for example by selecting data for continued pre-training or by providing interpretable transfer diagnostics.

## 8. Limitations

The focus of this study was confined to univariate time series. Therefore, the applicability and performance of the proposed method on multivariate time series have not been explored. Extending the shapelet discovery and matching process to multivariate time series would require channel-wise or joint multichannel shapelet discovery, together with a matching criterion that accounts for cross-channel interactions.

Another limitation is the architecture-dependent behavior observed in our experiments. Among the three evaluated backbones, Shapelet Matching produced the largest gains with the VGG-based CNN, whose local convolutional filters are well aligned with local discriminative subsequences. For patch-based Transformers, the gains are smaller and DBA-DTW sometimes achieves higher mean accuracy. This suggests that shapelet-based source selection may not be universally optimal across all backbone architectures. Therefore, the proposed method should be viewed as particularly suitable for backbones with a strong local-pattern bias, rather than as a universally optimal source-selection criterion for all architectures.

Finally, the shapelet discovery hyperparameters were fixed in this study. We used a shapelet length of 15 and 10 candidate shapelets per class for all datasets, following the previous setting, but did not conduct a sensitivity analysis over either hyperparameter. Since the appropriate subsequence scale and number of discriminative patterns may vary across datasets, future work should explore multi-scale shapelets, validationbased hyperparameter selection, or adaptive shapelet discovery.

## 9. Conclusion

We introduced Shapelet Matching, a training-free source-selection method for multi-source transfer learning in time series classification. The method ranks candidate source datasets by comparing class-discriminative shapelets between source and target datasets, avoiding the need to pre-train a separate model for every candidate source. Experiments on 128 UCR datasets show that multi-source pre-training can reduce the risk of negative transfer, with the strongest gains observed for CNN-based models and competitive performance for patch-based Transformers. These results suggest that local shapelet-level similarity is a useful and potentially interpretable criterion for selecting source datasets when large-scale foundation-model pre-training is not available. This provides practitioners with a lightweight source-selection tool when large external pre-training corpora or foundation-model checkpoints are unavailable. The main limitations are the reliance on transferable local subsequences, architecture-dependent performance, the univariate-only evaluation, and the use of fixed shapelet hyperparameters. Future work will extend the method to multivariate time series, adaptive shapelet discovery, and architecture-aware source ranking.

Declaration of Generative AI and AI-assisted Technologies in the Manuscript Preparation Process

During the preparation of this work, the first author used Cursor for coding assistance, Grammarly for grammar checking, and Gemini and ChatGPT for wording assistance. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of this article.

## Acknowledgement

This work was partially supported by JST BOOST, Japan Grant Number JPMJBS2406.

## References

[1] Y. Lecun, L. Bottou, Y. Bengio, and P. Hafner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[2] H. I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, “Transfer learning for time series classification,” in Int. Conf. Big Data, 2018.

[3] B. K. Iwana and S. Uchida, “An empirical survey of data augmentation for time series classification with neural networks,” PLOS ONE, 2021.

[4] S. Bozinovski, “Reminder of the first paper on transfer learning in neural networks, 1976,” Informatica, vol. 44, no. 3, 2020.

[5] F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, and Q. He, “A comprehensive survey on transfer learning,” Proc. IEEE, vol. 109, no. 1, pp. 43–76, 2021.

[6] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer: generative pre-trained transformers are large time series models,” in Int. Conf. Mach. Learn., 2024, pp. 32 369–32 399.

[7] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “MO-MENT: A family of open time-series foundation models,” in Int. Conf. Mach. Learn., 2024, pp. 16 115–16 152.

[8] S. Gao, T. Koker, O. Queen, T. Hartvigsen, T. Tsiligkaridis, and M. Zitnik, “UniTS: A unified multi-task time series model,” Adv. Neural Inf. Process. Sys., vol. 37, pp. 140 589–140 631, 2024.

[9] C.-C. M. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H. A. Dau, D. F. Silva, A. Mueen, and E. Keogh, “Matrix profile i: All pairs similarity joins for time series: A unifying view that includes motifs, discords and shapelets,” in Int. Conf. Data Mining, 2016.

[10] H. A. Dau, A. Bagnall, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A. Ratanamahatana, and E. Keogh, “The UCR time series archive,” J. Automatica Sinica, vol. 6, no. 6, pp. 1293–1305, 2019.

[11] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and

N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in Int. Conf. Learn. Rep., 2021.

[12] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Int. Conf. Learn. Rep., 2023.

[13] M. Weber, M. Auch, C. Doblander, P. Mandl, and H.-A. Jacobsen, “Transfer learning with time series data: A systematic mapping study,” IEEE Access, vol. 9, pp. 165 409–165 432, 2021.

[14] R. Clark and T. E. Doyle, “A priori quantification of transfer learning performance on time series classification for cyber-physical health systems,” in Canadian Conf. Electr. and Comput. Eng., 2022.

[15] P. Gikunda and N. Jouandeau, “Homogeneous transfer active learning for time series classification,” in Int. Conf. Mach. Learn. and Appl., 2021.

[16] T. Zhou, P. Niu, L. Sun, R. Jin et al., “One fits all: Power general time series analysis by pretrained lm,” Adv. Neural Inf. Process. Sys., vol. 36, pp. 43 322– 43 355, 2023.

[17] Z. Liu, Y. Luo, B. Li, E. Eldele, M. Wu, and Q. Ma, “Learning soft sparse shapes for eficient time-series classification,” in Int. Conf. Mach. Learn., vol. 267. PMLR, 2025, pp. 39 032–39 059.

[18] Z. Liu, Y. Wang, B. Li, J. Zheng, E. Eldele, M. Wu, and Q. Ma, “A unified shapeaware foundation model for time series classification,” AAAI Conf. Artif. Intell., vol. 40, no. 28, pp. 23 972–23 980, 2026.

[19] W. Dai, Q. Yang, G.-R. Xue, and Y. Yu, “Boosting for transfer learning,” in Int. Conf. Mach. Learn., 2007.

[20] Y. Yao and G. Doretto, “Boosting for transfer learning with multiple sources,” in Conf. Comput. Vis. and Pattern Recognit., 2010.

[21] P. Huang, G. Wang, and S. Qin, “Boosting for transfer learning from multiple data sources,” Pattern Recognit. Lett., vol. 33, no. 5, pp. 568–579, 2012.

[22] B. Tan, E. Zhong, E. W. Xiang, and Q. Yang, “Multi-transfer: Transfer learning with multiple views and multiple sources,” in Int. Conf. Data Mining, 2013.

[23] H.-J. Song and S.-B. Park, “Identifying intention posts in discussion forums using multi-instance learning and multiple sources transfer learning,” Soft Comput., vol. 22, no. 24, pp. 8107–8118, 2018.

[24] J. Li, S. Qiu, Y.-Y. Shen, C.-L. Liu, and H. He, “Multisource transfer learning for cross-subject EEG emotion recognition,” IEEE Trans. Cyber., vol. 50, no. 7, pp. 3281–3293, 2020.

[25] R. Ren, Y. Yang, and H. Ren, “EEG emotion recognition using multisource instance transfer learning framework,” in Int. Conf. Image Process. Comput. Vis. and Mach. Learn., 2022.

[26] F. Li, K. Shirahama, M. A. Nisar, X. Huang, and M. Grzegorzek, “Deep transfer learning for time series data based on sensor modality classification,” Sensors, vol. 20, no. 15, p. 4271, 2020.

[27] J. Lee and B. K. Iwana, “Model selection with a shapelet-based distance measure for multi-source transfer learning in time series classification,” in Int. Conf. Pattern Recognit. Springer, 2025, pp. 160–175.

[28] W. Zhang, L. Deng, L. Zhang, and D. Wu, “A survey on negative transfer,” J. Automatica Sinica, vol. 10, no. 2, pp. 305–329, 2023.

[29] M. Ragab, E. Eldele, Z. Chen, M. Wu, C.-K. Kwoh, and X. Li, “Self-supervised autoregressive domain adaptation for time series data,” IEEE Trans. Neural Netw. and Learn. Syst., vol. 35, no. 1, pp. 1341–1351, 2024.

[30] S. Deng, O. Sprangers, M. Li, S. Schelter, and M. De Rijke, “Domain generalization in time series forecasting,” ACM Trans. Knowl. Discov. from Data, vol. 18, no. 5, pp. 1–24, 2024.

[31] Y. Bao, Y. Li, S.-L. Huang, L. Zhang, L. Zheng, A. Zamir, and L. Guibas, “An information-theoretic approach to transferability in task transfer learning,” in Int. Conf. Image Process., 2019.

[32] C. Nguyen, T. Hassner, M. Seeger, and C. Archambeau, “LEEP: A new measure to evaluate transferability of learned representations,” in Int. Conf. Mach. Learn., 2020, pp. 7294–7305.

[33] K. You, Y. Liu, J. Wang, and M. Long, “LogME: Practical assessment of pretrained models for transfer learning,” in Int. Conf. Mach. Learn., 2021, pp. 12 133–12 143.

[34] A. Tran, C. Nguyen, and T. Hassner, “Transferability and hardness of supervised classification tasks,” in Int. Conf. Comput. Vis., 2019.

[35] L.-K. Huang, J. Huang, Y. Rong, Q. Yang, and Y. Wei, “Frustratingly easy transferability estimation,” in Int. Conf. Mach. Learn., 2022, pp. 9201–9225.

[36] H. Xu and U. Kang, “Fast and accurate transferability measurement by evaluating intra-class feature variance,” in Int. Conf. Comput. Vis., 2023, pp. 11 474– 11 482.

[37] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization for spoken word recognition,” Readings Speech Recognit., pp. 159–165, 1990.

[38] F. Petitjean, A. Ketterlin, and P. Gançarski, “A global averaging method for dynamic time warping, with applications to clustering,” Pattern Recognit., vol. 44, no. 3, pp. 678–693, 2011.

[39] L. Ye and E. Keogh, “Time series shapelets,” in Knowl. Discov. and Data Mining, 2009.

[40] K. Simonyan and A. Zisserman, “Very deep convolutional networks for largescale image recognition,” in Int. Conf. Learn. Rep., 2015.