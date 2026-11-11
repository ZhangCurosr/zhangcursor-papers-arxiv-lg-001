# LUMIR: Decoupled Multimodal Representation Learning from Heterogeneous Data for Next-POI Recommendation

Zehui Wang<sup>1,2,\*</sup>, Wolfram Höpken<sup>1</sup> and Dietmar Jannach<sup>2</sup>

<sup>1</sup>University of Applied Sciences Ravensburg-Weingarten, Doggenriedstrasse, Weingarten, Weingarten, 88250, Germany <sup>2</sup>University ofKlagenfurt, Universitätsstraße 65–67, Klagenfurt am Wörthersee, 9020, Austria

## Abstract

As multimodal tourism-related information collected from heterogeneous data sources has become increasingly available, learning efective representations has attracted attention for improving recommendation performance. Existing approaches often tightly couple multimodal fusion with the architecture and optimization objective of a specific downstream model, which limits the generality and reusability of the learned representations. To address these limitations, we propose LUMIR, a decoupled multimodal representation learning framework based on contrastive learning. Multimodal information collected from heterogeneous data sources is first semantically unified into a shared textual representation space using large language models (LLMs). An interaction-aware fusion mechanism is then designed to model cross-modal interactions, while sequential signals are incorporated as a structural constraint to guide representation learning. To evaluate the learned representations, experiments are conducted on a multimodal point-of-interest (POI) dataset under the next-POI recommendation setting across multiple sequential recommendation backbones. The results show that LUMIR consistently outperforms existing multimodal fusion baselines in both accuracy-oriented and beyond-accuracy metrics, demonstrating that decoupling representation learning from downstream optimization provides an efective paradigm for learning reusable multimodal representations. The code is publicly available at https://github.com/wzehui/LUMIR.

## Keywords

Multimodal Representation Learning, Next-POI Recommendation, Contrastive Learning, Large Language Models,

## 1. Introduction

With the growing availability of multimodal data, modern recommender systems increasingly leverage cross-modal representation learning in data-driven applications [1]. In scenarios such as POI recommendation, item-related information is typically derived from heterogeneous data sources, including textual descriptions, visual content, and structured attributes. Efectively integrating multimodal tourismrelated information from heterogeneous data sources into a unified and informative representation remains a fundamental challenge [2].

Recent advances in large language models (LLMs) provide a promising direction for modeling multimodal tourism-related information by transforming information from heterogeneous data sources into a shared semantic representation space [3]. However, heterogeneous data sources may contain semantically diverse, redundant, inconsistent, or only partially informative signals that cannot be fully resolved through semantic unification alone and can introduce noise in multimodal data, thereby degrading the quality of the learned representations.

Despite these challenges, multimodal representation learning is typically tightly coupled with downstream tasks, where the learned representations are directly optimized based on task-specific performance signals [4]. While this design can improve task-specific performance, it introduces several limitations. First, the learned representations become tied to a particular downstream architecture and its optimization process, limiting their generality and transferability across diferent tasks or domains [5]. Second, joint optimization of multimodal fusion and downstream recommendation models requires representation learning to be repeated when the downstream architecture changes, increasing computational cost and reducing scalability [6]. More importantly, supervision from downstream objectives biases multimodal fusion toward task-optimal solutions rather than intrinsic cross-modal relationships, limiting cross-modal interaction modeling [2]. These challenges become particularly evident in multimodal POI recommendation, where multimodal tourism-related information collected from heterogeneous data sources must be integrated with sequential user interactions. Such heterogeneous information characteristics motivate the need for a more flexible and decoupled representation learning framework.

As an important task within POI recommendation, next-POI recommendation aims to predict the next location users will visit based on their historical interaction sequence [7]. Compared to general POI recommendation, next-POI recommendation focuses on modeling short-term sequential patterns and temporal dynamics [8]. In this setting, multimodal POI information provides complementary signals for capturing contextual characteristics and user preferences [9], and its efective integration in sequential recommendation has attracted increasing attention in recent work [10].

To address these limitations, we propose LLM-based Unified Multimodal Information Representation (LUMIR), a decoupled multimodal representation learning framework. Unlike existing approaches that tightly couple representation learning with downstream models, LUMIR learns representations independently of downstream model architectures and their optimization objectives. Leveraging the prior knowledge and representation capabilities of LLMs, LUMIR learns unified representations from multimodal tourism-related information collected from heterogeneous data sources (e.g., images, reviews, and metadata). On top of this unified representation space, LUMIR models cross-modal interactions while reducing the influence of redundant information and incorporating sequential signals as structural guidance for representation learning. Through stochastic modality augmentation and information compression, the framework is designed to reduce its dependence on noisy or partially informative modality-specific signals without requiring explicit data-quality annotations. The learned representations are evaluated on POI recommendation as a downstream task to assess their efectiveness in addressing key challenges in multimodal recommender systems. To the best of our knowledge, this work is the first to investigate a multimodal representation learning framework decoupled from downstream architectures and their optimization in the context of next-POI recommendation. The key contributions are summarized as follows:

<sup>•</sup> We propose LUMIR, a decoupled multimodal representation learning framework that separates representation learning from the architecture and optimization of downstream recommendation tasks. This design facilitates a modular approach to multimodal modeling.

<sup>•</sup> We propose an LLM-based representation unification framework for multimodal tourism-related information collected from heterogeneous data sources, in which the encoder is trainable and jointly optimized with the multimodal fusion process, enabling adaptive unified representations tailored to cross-modal interactions.

<sup>•</sup> We interpret multimodal representation learning under the information bottleneck framework and propose a unified objective function that jointly integrates modality compression and sequenceaware alignment, enabling the learned representations to capture both cross-modal interactions and sequential structure.

<sup>•</sup> We conduct experiments on a multimodal POI dataset under the next-POI recommendation setting, showing that the representations improve performance compared to multimodal baselines.

## 2. Related Work

This section reviews recent work on task-dependent multimodal representation learning, LLM-based representation unification, and next-POI recommendation.

Multimodal Representation Learning. Multimodal representation learning has been extensively studied to learn unified representations from heterogeneous data sources, where information is distributed across textual, visual, and structured modalities [4]. Recent advances therefore move beyond simple alignment to model cross-modal interactions by capturing redundant, unique, and synergistic information across modalities [11].

Despite diferences in application domains and model architectures, these approaches generally learn multimodal representations through joint optimization with downstream models, where the fusion process is tightly coupled with downstream objectives. For instance, prior work has evaluated multimodal representations directly based on recommendation performance [12], learned joint vision-language representations through task-driven supervision [13], or optimized multimodal models for downstream tasks such as classification and question answering [14]. Such designs produce representations that are tightly optimized together with a particular downstream architecture, limiting their modular reuse across diferent recommendation models.

LLM-based Representation Unification. Recent advances in LLMs provide a promising direction for multimodal representation learning by enabling multimodal information collected from heterogeneous data sources to be represented within a shared semantic embedding space [15]. Such semantic unification helps reduce modality heterogeneity while providing a common foundation for subsequent multimodal representation learning.

Recent studies leverage LLMs as a semantic bridge across application domains. For example, LLMbased semantic enhancement frameworks have been applied to unify textual and visual signals for POI representation learning in the recommendation domain [16]. In vision-language tasks, models such as BLIP-2 connect visual encoders with LLMs, enabling images to be represented and processed within a language-centric semantic space [17]. Similarly, models such as LLaVA align visual and language features via instruction tuning, enabling unified multimodal understanding [18]. Building on this prior work, our work uses LLMs to semantically unify multimodal tourism-related information collected from heterogeneous data sources before downstream representation learning.

Next-POI Recommendation. Next-POI recommendation is a key task in tourism recommendation, which aims to predict the next location a user will visit based on historical trajectory [8]. POI data is inherently multimodal, with rich information sources associated with each location. Recent work has explored incorporating additional contextual and multimodal information, such as geographical features [19], semantic information [20], and visual content [21].

With the rapid development of LLMs, their capabilities have increasingly been incorporated into POI recommendation to improve recommendation performance. For example, LLMs have been leveraged to enrich multimodal POI representations by transforming textual and visual information into structured representations [16]. In addition, methods such as SeCor align LLM-generated representations with collaborative signals [22]. These works highlight the potential of LLM-based representations for integrating multimodal information, motivating our use of LLM-based encoders within a decoupled representation learning framework.

Despite these advances, existing approaches generally couple multimodal representation learning with downstream recommendation objectives, making the learned representations dependent on a particular model configuration. Such end-to-end optimization may dominate the fusion process and limit the modeling of intrinsic relationships among heterogeneous information sources. Our work therefore investigates a decoupled framework that learns multimodal representations independently of the downstream recommendation objective and evaluates their efectiveness in next-POI recommendation.

## 3. Problem Formulation

This section defines the problem setting and notation for multimodal representation learning and next-POI recommendation.

![](images/ff2546a47298763ff40ace2279230298aff984e97be2377fa08e81751a58c871.jpg)  
Figure 1: Overview of the proposed LUMIR framework.

## 3.1. Multimodal Representation Learning

Let $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { | \mathcal { V } | } \}$ denote a set of recommendable items. Each item $v \in \mathcal V$ is associated with multimodal information:

$$
\mathbf { m } _ { v } = \{ m _ { v } ^ { ( 1 ) } , \dots , m _ { v } ^ { ( M ) } \} ,\tag{1}
$$

where $m _ { v } ^ { ( k ) }$ denotes the �-th modality, each corresponding to heterogeneous data sources (e.g., text, images, structured data).

The multimodal representation learning problem aims to map heterogeneous modalities into a unified latent space [13, 23] and can be abstracted as:

$$
\mathbf { z } _ { v } = \mathcal { F } ( \{ m ^ { ( k ) } \} _ { k = 1 } ^ { M } ) ,\tag{2}
$$

where $\mathcal F ( \cdot )$ denotes a multimodal representation function and $\mathbf { z } _ { v }$ is the representation of item �.

## 3.2. Next-POI Recommendation

In this work, next-POI recommendation is considered as the downstream application, where each item $v \in \mathcal V$ corresponds to a POI. Let $\mathcal { U } = \{ u _ { 1 } , \ldots , u _ { | \mathcal { U } | } \}$ denote the set of users and $\mathcal { P } = \{ p _ { 1 } , . . . , p _ { | \mathcal { P } | } \}$ denote the set of POIs. For each user � $\iota \in { \mathcal { U } } _ { : }$ , the historical interaction sequence is defined as:

$$
{ \cal S } _ { u } = \langle p _ { 1 } , p _ { 2 } , . . . , p _ { T } \rangle ,\tag{3}
$$

where $p _ { t } \in \mathcal P$ is the POI visited at time step �.

Given the observed sequence $S _ { u } .$ the next-POI recommendation task aims to predict the next POI $p _ { T + 1 }$ . The prediction is performed based on the sequence $S _ { u }$ and the multimodal representations $\mathbf { z } _ { p }$

## 4. Methodology

This section presents the proposed LUMIR framework, which consists of three main components: (1) LLM-based representation unification, (2) augmented multimodal representation learning, and (3) downstream evaluation via next-POI recommendation. An overview of the framework architecture is illustrated in Figure 1.

## 4.1. LLM-based Representation Unification

In the multimodal POI dataset used in this work, each POI $p \in \mathcal P$ is associated with multiple feature sources, including textual, visual, and structured information. To focus on multimodal representation learning, these heterogeneous feature sources are grouped into three modalities: (1) Metadata (�):

basic business descriptors such as name and category, structured attributes, and spatial information including geolocation and neighborhood representations based on H3 spatial indexing [24, 25]; (2) Review (�): LLM-generated summaries of given user reviews and rating statistics; (3) Photo (�): LLM-generated visual content summaries and additional visual features extracted from images.

Based on Eq. 1, the multimodal information of each POI can be written as:

$$
\mathbf { m } _ { p } = \{ m _ { p } ^ { ( k ) } \mid k \in \{ M , R , P \} \} ,\tag{4}
$$

where $m _ { p } ^ { ( M ) } , m _ { p } ^ { ( R ) }$ , and $m _ { p } ^ { ( P ) }$ correspond to metadata, review, and photo modalities, respectively. Following [16], each modality is transformed into a structured textual representation and encoded by a tunable LLM-based text encoder to generate modality-specific embeddings:

$$
\begin{array} { r } { \mathbf { z } _ { p } ^ { ( k ) } = f _ { \theta } \big ( m _ { p } ^ { ( k ) } \big ) , \quad k \in \{ M , R , P \} , } \end{array}\tag{5}
$$

where $f _ { \theta } ( \cdot )$ denotes the encoding function.

This semantic unification transforms information from diferent modalities into a shared textual representation space, thereby reducing the heterogeneity of the original modality formats. These modality-specific embeddings $\{ \mathbf { z } _ { p } ^ { ( k ) } \}$ are subsequently used as inputs for multimodal representation learning, where a unified POI representation $\mathbf { z } _ { p }$ is learned by fusion across modalities.

## 4.2. Multimodal Representation Learning

Given the modality-specific embeddings $\{ \mathbf { z } _ { p } ^ { ( k ) } \mid k \in \{ M , R , P \} \}$ , the goal of LUMIR is to learn a unified POI representation $\mathbf { z } _ { p }$ by integrating information across modalities. Instead of optimizing multimodal representations through the loss of a downstream model, LUMIR adopts a self-supervised learning paradigm to capture intrinsic relationships across modalities. In particular, contrastive learning efectively models correlations among multimodal representations by grouping related instances and separating unrelated ones [26], making it well suited for decoupled representation learning. Based on this, a contrastive objective is introduced to guide the fusion of modality-specific embeddings.

Modality Compression Loss While contrastive learning can align representations across modalities, efective integration of multimodal information goes beyond alignment. In this work, a modality fusion framework proposed in [11] is adopted. In particular, a modality augmentation transformation, denoted by $\tau ,$ is applied to modality-specific embeddings $\{ \mathbf { z } _ { p } ^ { ( k ) } \mid k \in \{ M , R , P \} \}$ , where each embedding corresponding to a modality $k \in \{ M , R , P \}$ is randomly perturbed or masked to generate stochastic partial views of the same POI. This encourages the model to learn robust representations by leveraging complementary information across modalities rather than relying on any single modality. As a result, two independent modality augmentation transformations, denoted by $\tau ^ { \prime }$ and $\mathcal { T } ^ { \prime \prime }$ , are applied to $\{ \mathbf { z } _ { p } ^ { ( k ) } \mid k \in \{ M , R , P \} \}$ . These are then fused via an attention-based multimodal fusion function $g _ { \phi } ( \cdot )$ to obtain two stochastic views:

$$
\tilde { \mathbf { z } } _ { p } ^ { \prime } = g _ { \phi } \big ( \mathcal { T } ^ { \prime } ( \mathbf { z } _ { p } ^ { ( k ) } ) \big ) , \qquad \tilde { \mathbf { z } } _ { p } ^ { \prime \prime } = g _ { \phi } \big ( \mathcal { T } ^ { \prime \prime } ( \mathbf { z } _ { p } ^ { ( k ) } ) \big ) .\tag{6}
$$

These two fused views correspond to diferent incomplete observations of the same POI and serve as positive pairs in contrastive learning. The modality compression objective enforces consistency between these two fused views while simultaneously aligning each modality-specific embedding with the fused representation.

Compared with existing approaches that restrict negative sampling within a mini-batch, a practical challenge in this work arises from the high dimensionality of multimodal representations. This leads to increased memory consumption and limits the feasible batch size during training. As contrastive learning relies on a large number of negative samples, small batch sizes reduce the diversity of negatives and can degrade representation quality [27]. To address this issue, LUMIR employs a memory bank ℳ to store representations from previous mini-batches and reuse them as additional negative samples [27].

This mechanism efectively enlarges the pool of negative samples without increasing the batch size. With the Information Noise-Contrastive Estimation (InfoNCE) objective denoted as $\hat { I } _ { \mathrm { N C E } } ( \cdot , \cdot )$ , the modality compression loss is defined as:

$$
\mathcal { L } _ { \mathrm { M C } } = - \frac { 1 } { | \mathcal { M } | } \sum _ { p \in \mathcal { M } } \left[ \hat { I } _ { \mathrm { N C E } } \big ( \tilde { \mathbf { z } } _ { p } ^ { \prime } , \tilde { \mathbf { z } } _ { p } ^ { \prime \prime } \big ) + \frac { 1 } { | \mathcal { K } | } \sum _ { k \in \mathcal { K } } \bar { I } _ { \mathrm { N C E } } \Big ( \mathbf { z } _ { p } ^ { ( k ) } , \tilde { \mathbf { z } } _ { p } \Big ) \right] ,\tag{7}
$$

where

$$
\bar { I } _ { \mathrm { N C E } } \left( \mathbf { z } _ { p } ^ { ( k ) } , \tilde { \mathbf { z } } _ { p } \right) = \frac { 1 } { 2 } \left( \hat { I } _ { \mathrm { N C E } } \left( \mathbf { z } _ { p } ^ { ( k ) } , \tilde { \mathbf { z } } _ { p } ^ { \prime } \right) + \hat { I } _ { \mathrm { N C E } } \left( \mathbf { z } _ { p } ^ { ( k ) } , \tilde { \mathbf { z } } _ { p } ^ { \prime \prime } \right) \right) .\tag{8}
$$

Sequence Alignment Loss Inspired by recent advances in applying contrastive learning to sequential recommendation [28, 29], a sequence-aware alignment objective is introduced to enhance the multimodal representations. The key assumption is that items appearing within the same user sequence exhibit semantic relatedness. Aligning their representations therefore helps $\mathbf { z } _ { p }$ capture sequential patterns that are useful for sequential modeling. This sequence-derived signal introduces task-relevant structural information into representation learning, while remaining independent of the parameters and optimization objective of any particular downstream sequential recommendation architecture.

To this end, a sequence-aware contrastive objective is constructed for the next-POI recommendation setting. For a given current POI $p ,$ positive samples are defined as POIs that appear as subsequent transitions in user sequences, while negative samples are drawn from other POIs stored in a memory bank ℳ. The sequence alignment loss is defined as:

$$
\mathcal { L } _ { \mathrm { S A } } = - \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { p \in \mathcal { M } } \hat { I } _ { \mathrm { N C E } } ( \mathbf { z } _ { p } , \mathcal { P } ( p ) ; \mathcal { M } ) ,\tag{9}
$$

where $\mathcal { P } ( \boldsymbol { p } )$ denotes the set of positive samples, i.e., POIs observed as next-step transitions for the current POI � across user sequences, and negative samples are sampled from ℳ.

Joint Objective The two contrastive objectives introduced above provide complementary learning signals for multimodal POI representation learning. To better understand how they jointly shape the learned representation, the framework can be interpreted from an information-theoretic perspective. Let $X _ { p }$ denote the original multimodal input of POI $p ,$ defined as the collection of modality-specific information, i.e., $X _ { p } = \{ m _ { p } ^ { ( k ) } \mid k \in \{ M , R , P \} \}$ . Multimodal representation learning can be viewed as optimizing the mutual information between $X _ { p }$ and $\mathbf { z } _ { p } ,$ aiming to learn a compact representation that suppresses redundant information while retaining task-relevant signals. Following the Information Bottleneck (IB) principle [30, 31, 32], the objective can be conceptually formulated in terms of mutual information � as:

$$
\mathcal { L } _ { \mathrm { I B } } = I ( X _ { p } ; \mathbf { z } _ { p } ) - \beta I ( \mathbf { z } _ { p } ; y ) ,\tag{10}
$$

where � denotes the prediction target in the next-POI recommendation task.

In this work, the prediction target � cannot be directly used during representation learning due to the decoupled training paradigm. To address this issue, the relational structure within user sequences is used as a surrogate signal. The underlying intuition is that the prediction target � is typically derived from, or conditioned on, the user sequence. Therefore, representations that capture sequence-level dependencies are likely to retain information relevant to the downstream task. This relationship can be interpreted through the Data Processing Inequality (DPI), which states that processing a signal cannot increase the information it contains about an underlying target. Under this assumption, maximizing the mutual information between the representation $\mathbf { z } _ { p }$ and sequence-level signals can serve as a surrogate objective for preserving task-relevant information. While this surrogate is not strictly equivalent to directly maximizing $I ( \mathbf { z } _ { p } ; y )$ , it provides a tractable and informative optimization signal under the decoupled setting.

Based on Eq. 10, LUMIR approximates the IB objective using contrastive learning. Specifically, the modality compression term is formulated as Eq. 7. By constructing multiple augmented views of the same POI and enforcing consistency between fused representations and modality-specific embeddings, $\mathcal { L } _ { \mathrm { M C } }$ encourages the model to retain shared and complementary information across modalities while suppressing redundant signals. Similarly, the sequence alignment term is implemented as Eq. 9, which provides a tractable lower-bound estimator of mutual information between paired variables [27]. As a result, minimizing $\mathcal { L } _ { \mathrm { S A } }$ implicitly maximizes the mutual information between $\mathbf { z } _ { p }$ and the sequencederived target signal. The final training objective is formulated as:

$$
\mathcal { L } _ { \mathrm { T o t a l } } = \lambda _ { \mathrm { M C } } \cdot \mathcal { L } _ { \mathrm { M C } } + \lambda _ { \mathrm { S A } } \cdot \mathcal { L } _ { \mathrm { S A } } ,\tag{11}
$$

where $\lambda _ { \mathrm { M C } }$ and $\lambda _ { \mathrm { S A } }$ balance multimodal compression and sequence-aware alignment, respectively.

Diferently from existing studies [33, 34] that couple multimodal fusion with downstream sequential recommendation models via joint optimization, LUMIR adopts a decoupled representation learning paradigm. The representation is learned by jointly optimizing the multimodal compression loss $\mathcal { L } _ { \mathrm { M C } }$ and the sequence-aware alignment loss $\mathcal { L } _ { \mathrm { S A } }$ . The optimization process updates the parameters of the tunable LLM-based encoder $f _ { \theta } ( \cdot )$ and the multimodal fusion function $g _ { \phi } ( \cdot )$ , which together define the mapping from modality-specific inputs to the unified representation. The final embedding configuration is selected based on a proxy objective defined in this work, which measures how frequently similar embeddings co-occur within the same user sequence, reflecting the ability of the learned representations to capture sequence-level consistency.

## 4.3. Downstream Sequential Recommendation

After modality fusion, the learned POI representations $\tilde { \mathbf { z } } _ { p }$ are evaluated as input representations for downstream next-POI recommendation. In this work, BERT4Rec [35], SASRec [36], and GRU4Rec [37] are employed as representative sequential recommendation models. To incorporate the learned POI embeddings, the original ID-based embedding layer in these models is replaced with a pre-initialized embedding matrix constructed from $\tilde { \mathbf { z } } _ { p } .$ , following prior work [38].

Given a user interaction sequence $s = [ p _ { 1 } , p _ { 2 } , \dotsc , p _ { t } ]$ , each POI $p _ { i }$ is represented by its embedding $\tilde { \mathbf { z } } _ { p _ { i } }$ . The sequential model takes the embedding sequence as input and learns to predict the next POI $p _ { t + 1 }$ based on the history. During this stage, the POI embeddings $\tilde { \mathbf { z } } _ { p }$ remain fixed, and the sequential model is trained to capture temporal or sequential dependencies over the embedding sequence.

## 5. Experiments

This section describes the experimental setup of the LUMIR framework, including the dataset, implementation details and baseline methods. Multimodal POI representations are learned using diferent fusion strategies and evaluated on downstream next-POI recommendation under a unified setting.

## 5.1. Dataset

Experiments are conducted on the Yelp multimodal dataset<sup>1</sup>, which is one of the publicly available multimodal datasets in the tourism domain accessible during this study. It contains rich business-related information, including metadata, user reviews, visual content, and user interaction data organized into sequential visit trajectories.<sup>2</sup> The statistics of the preprocessed dataset are summarized in Table 1.

Table 1  
Statistics of the Yelp multimodal POI dataset
<table><tr><td>Metric</td><td>Value</td><td>Metric</td><td>Value</td></tr><tr><td>#Interactions</td><td>509,892</td><td>Avg. User Seq. Length</td><td>23.0</td></tr><tr><td>#Businesses</td><td>16,702</td><td>Avg. Reviews per Business</td><td>30.5</td></tr><tr><td>#Users</td><td>22,207</td><td>Avg. Photos per Business</td><td>9.0</td></tr><tr><td>Sparsity</td><td>99.86%</td><td>Businesses with Photos</td><td>80%</td></tr></table>

## 5.2. Implementation Details

The experimental pipeline follows a decoupled setting. First, all modality-specific information is transformed into a unified embedding space via LLM-based representation unification. In this work, Linq-Embed-Mistral<sup>3</sup> is employed as a tunable LLM encoder to obtain modality-specific embeddings. The model is selected based on its strong embedding performance on the MTEB benchmark. POI embeddings are then learned independently for each method. During this stage, hyperparameters for representation learning are tuned via grid search, where the best-performing embeddings are selected based on the proxy objective. For LUMIR, modality augmentation is implemented through modality-level hard masking. During the first three warm-up epochs, all three modalities are retained. Afterwards, the probability of retaining the full modality set decreases linearly during training, with a lower bound of 0.2. Otherwise, the number of modalities to be retained is randomly selected with equal probability. The two augmented views are generated independently using this procedure.

The learned embeddings are subsequently fixed and used as input to downstream sequential recommendation models. All downstream models are optimized using the Adam optimizer with a crossentropy objective over POIs. BERT4Rec is trained through masked-item prediction, whereas SASRec and GRU4Rec are trained for next-item prediction. For the downstream models, hyperparameter tuning is performed using Optuna with 50 trials, where the best configuration is selected based on validation performance measured by NDCG@20. After identifying the best-performing hyperparameter configuration for each method, both LUMIR and the baselines are retrained under their respective optimal configurations for 10 runs with diferent random initializations. The final performance is reported as the average over these runs after removing the highest and lowest values to mitigate the efect of outliers.

All experiments are conducted on NVIDIA A40 (48GB) GPU for the representation learning stage and NVIDIA A800 (80GB) GPU for the downstream next-POI recommendation tasks.

## 5.3. Baselines

In this work, by applying the POI embeddings generated from various baselines to the same downstream recommendation task, LUMIR can be systematically evaluated under a unified experimental setting. Although recent approaches such as SeCor similarly leverage LLMs to integrate diferent information sources, SeCor additionally incorporates collaborative information derived from user interactions, which prevents a fair comparison with LUMIR under the same input conditions and is therefore not included as a baseline. In this work, four strategies based on diferent principles are adopted to construct POI representations:

Trajectory-Only Strategy. This strategy represents an ID-based baseline that does not incorporate any additional POI representations derived from multimodal information. In this setting, the next-POI recommendation task relies on the sequential patterns observed in users’ historical visit trajectories.

Full Concatenation Strategy. This strategy represents a parameter-free feature-level fusion strategy that concatenates embeddings from diferent modalities while preserving the complete information of each modality [16]. In this study, it serves as a baseline to evaluate the performance that can be achieved through feature-level fusion without explicitly modeling interactions between modalities.

Modality-Weighted Strategy. This strategy extends the Full Concatenation Strategy by adjusting diferent modality weights. The final POI representation is obtained through a weighted concatenation of modality embeddings. This strategy maintains a simple fusion structure while introducing explicit modeling of modality importance, which is commonly studied in multimodal fusion methods [23].

FLAVA-based Strategy. FLAVA is a vision–language model that learns joint representations via a shared encoder and cross-modal alignment objectives [39]. In the dataset used in this work, modalityspecific information (e.g., reviews and images) is summarized into textual representations by LLMs. The model is adapted to take the textual representations of the three modalities as input, while preserving the input structure as LUMIR. During fusion, attention mechanisms capture interactions across modalities within a unified encoding space, enabling implicit cross-modal integration beyond simple concatenation.

CoMM-based Strategy. The CoMM strategy learns multimodal representations by aligning modalityspecific embeddings in a shared space [11]. It focuses on modality-level alignment, which is conceptually similar to LUMIR, but does not incorporate sequential semantics or address the limitation of insuficient negative samples under small batch sizes.

## 6. Results

This section evaluates LUMIR framework from two perspectives. (1) The multimodal fusion process in LUMIR is analyzed to understand how meaningful structures emerge during representation learning. (2) The learned embeddings are assessed through their performance on downstream next-POI recommendation tasks.

## 6.1. Multimodal Fusion Dynamics

Due to the absence of direct supervision from downstream recommendation tasks, the quality of the learned representations in the multimodal fusion stage emerges from the training dynamics of the representation learning process itself. Diferent from existing unsupervised embedding methods that rely on fixed iterative propagation [40], LUMIR involves learnable multimodal fusion with complex training dynamics, making the training process a critical factor in determining representation quality. As a result, insuficient training may lead to under-expressive representations that fail to capture meaningful multimodal structures, while excessive training may overfit dominant interaction patterns, biasing representations toward popular POIs and under-representing long-tail items [41]. Therefore, two aspects are examined: (a) how the learning process improves the structural properties of the embeddings; and (b) how these changes afect downstream recommendation performance. Representations obtained from direct concatenation of untrained modality features are used as an initial status and compared with those learned at diferent epochs (5–25). The range of epochs is selected to capture the evolution of representations at diferent training stages within a practical computational budget.

Embedding Structure Evolution. As shown in Figure 2, the learned embeddings form more coherent clusters, as visualized using t-SNE. As representation learning progresses, the embedding space evolves from fragmented, attribute-driven groupings to more coherent and semantically structured clusters. While these visualizations provide important insights into the structural evolution of the embedding space, quantitative evaluation is still required to fully assess representation quality.

To this end, three metrics are used to characterize the structural properties of the learned embeddings: (1) Mean Similarity, which reflects the overall compactness of the embedding space; (2) Nearest Neighbor Recall@10 (NN Recall@10), which measures local neighborhood consistency [42]; and (3) the Intra–Inter Similarity Gap (Δ), which captures the separation between intra- and inter-session similarities [43]. As shown in Figure 3a, Mean Similarity increases at early epochs and then decreases, indicating a shift from overly compact representations to a more dispersed structure. In contrast, NN Recall@10 and Δ exhibit upward trends, suggesting improved local consistency and stronger discriminability between intra- and inter-session relationships as representation learning progresses.

![](images/1d4674afc22a7d674665e78e27cc0650edd1e63d848d3e8ed55f071baf9f7400.jpg)  
(a) Concatenation

![](images/43b960100a9c77882b60170dd4241c6e151b9c135a69192c2d548367d37eef8d.jpg)  
(b) Epoch 10

![](images/f7890d276e04ca8e32e6fc00d86866b06490f7a158fc1021bf244dab5a370feb.jpg)  
(c) Epoch 25  
Figure 2: Visualization of Embeddings Across Training.

![](images/7c2a857a298ba0f6d47a611c8de09052d7edfab87bf5065a3358b00496c44ccb.jpg)  
(a) Structural properties of learned representations

![](images/1e95d471fe5046f359892dc1d2b9251543856c4de5b8e1aa8566b301d7fb4ec9.jpg)  
(b) Downstream performance across epochs  
Figure 3: Evolution of representation structure and downstream performance during representation learning

Impact on Downstream Performance. While the above analysis provides insights into the structural properties of the learned embeddings, improvements in embedding structure do not necessarily translate into gains in downstream performance. Therefore, this section focuses on the evolution of downstream performance during representation training, rather than final evaluation results. As shown in Figure 3b, all metrics exhibit consistent improvements during early training stages. HR@10, NDCG@10, and MRR@10 increase notably from epoch 0 to epoch 10, indicating that representation learning efectively enhances recommendation performance. After this stage, performance gains become marginal, with slight fluctuations observed across later epochs.

Therefore, considering the trade-of between training cost, embedding structure, and downstream performance, epoch 15 is selected as the final configuration. At this point, the learned representations achieve a balanced state, where intra–inter separation is well established while downstream performance remains stable.

## 6.2. Downstream Recommendation Performance

Overall Performance Comparison. The learned POI representations from LUMIR and baseline methods are evaluated on downstream sequential models for next-POI recommendation. Performance is assessed using both accuracy-oriented metrics (HR@K, NDCG@K, MRR@K) and beyond-accuracy metrics (Coverage [44], Gini index, Novelty [45], and Serendipity [46]), with all metrics reported at cutof values $K \in \{ 1 0 , 2 0 \}$

Table 2  
Performance on the Yelp dataset with diferent sequential recommendation backbones  
(a) BERT4Rec Backbone
<table><tr><td rowspan="2">Embedding</td><td colspan="3">@10</td><td colspan="3">@20</td></tr><tr><td>NDCG</td><td>HitRate</td><td>MRR</td><td>NDCG</td><td>HitRate</td><td>MRR</td></tr><tr><td>Trajectory-Only</td><td>0.01940 ± 0.0019 (-20.0%)</td><td>0.03901 ± 0.0051 (-18.6%)</td><td>0.01353 ± 0.0012 (-21.0%)</td><td>0.02618 ± 0.0021 (-16.7%)</td><td>0.06608 ± 0.0058 (-13.6%)</td><td>0.01535 ± 0.0012 (-19.6%)</td></tr><tr><td>Full Concatenation</td><td>0.02302 ± 0.0016 (-5.1%)</td><td>0.04399 ± 0.0042 (-8.2%)</td><td>0.01671 ± 0.0011 (-2.5%)</td><td>0.02941 ± 0.0016 (-6.5%)</td><td>0.06961 ± 0.0051 (-9.0%)</td><td>0.01841 ± 0.0010 (-3.5%)</td></tr><tr><td>Modality-Weighted</td><td>0.02426 ± 0.0018 (+0.0%)</td><td>0.04794 ± 0.0039 (+0.0%)</td><td>0.01713 ± 0.0015</td><td>0.03144 ± 0.0022</td><td>0.07647 ± 0.0066</td><td>0.01908 ± 0.0015</td></tr><tr><td>FLAVA-Based</td><td>0.02263 ± 0.00086 (-6.7%)</td><td>0.04686 ± 0.0021 (-2.3%)</td><td>(+0.0%) 0.01540 ± 0.0014 (-10.1%)</td><td>(+0.0%) 0.02875 ± 0.0012 (-8.6%)</td><td>(+0.0%) 0.07127 ± 0.0030 (-6.8%)</td><td>(+0.0%) 0.01704 ± 0.0015 (-10.7%)</td></tr><tr><td>CoMM-Based</td><td>0.02341 ± 0.00064 (-3.5%)</td><td>0.04526 ± 0.0016 (-5.6%)</td><td>0.01681 ± 0.0010 (-1.9%)</td><td>0.02993 ± 0.0012 (-4.8%)</td><td>0.07123 ± 0.0040 (-6.9%)</td><td>0.01857 ± 0.0011 (-2.7%)</td></tr><tr><td>LUMIRMC</td><td>0.02663 ± 0.0014 (+9.8%)</td><td>0.05304 ± 0.0021 (+10.6%)</td><td>0.01873 ± 0.0015 (+9.3%)</td><td>0.03591 ± 0.0017 (+14.2%)</td><td>0.08988 ± 0.0056 (+17.5%)</td><td>0.02126 ± 0.0015 (+11.4%)</td></tr><tr><td>LUMIRMC+SA</td><td>0.02915 ± 0.0029 (+20.2%)</td><td>0.05956 ± 0.0051 (+24.2%)</td><td>0.02010 ± 0.0025 (+17.3%)</td><td>0.03896 ± 0.0029 (+23.9%)</td><td>0.09857 ± 0.0057 (+28.9%)</td><td>0.02276 ± 0.0025 (+19.3%)</td></tr></table>

(b) SASRec Backbone
<table><tr><td rowspan="2">Embedding</td><td colspan="3">@10</td><td colspan="3">@20</td></tr><tr><td>NDCG</td><td>HitRate</td><td>MRR</td><td>NDCG</td><td>HitRate</td><td>MRR</td></tr><tr><td>Trajectory-Only</td><td>0.01789 ± 0.0016 (-21.5%)</td><td>0.03304 ± 0.0043 (-21.5%)</td><td>0.01331 ± 0.00090 (-21.8%)</td><td>0.02352 ± 0.0023 (-18.1%)</td><td>0.05530 ± 0.0068 (-17.9%)</td><td>0.01485 ± 0.0011 (-20.2%)</td></tr><tr><td>Full Concatenation</td><td>0.01731 ± 0.0025 (-24.0%)</td><td>0.03358 ± 0.0049 (-20.2%)</td><td>0.01242 ± 0.0019 (-27.0%)</td><td>0.02322 ± 0.0033 (-19.2%)</td><td>0.05720 ± 0.0087 (-15.1%)</td><td>0.01401 ± 0.0021 (-24.7%)</td></tr><tr><td>Modality-Weighted</td><td>0.02167 ± 0.0022 (-4.9%)</td><td>0.04200 ± 0.0032 (-0.2%)</td><td>0.01557 ± 0.0021 (-8.5%)</td><td>0.02802 ± 0.0025</td><td>0.06734 ± 0.0063</td><td>0.01729 ± 0.0021</td></tr><tr><td>FLAVA-Based</td><td>0.02279 ± 0.00096</td><td>0.04209 ± 0.0023</td><td>0.01701 ± 0.0010</td><td>(-2.4%) 0.02872 ± 0.0011</td><td>(+0.0%) 0.06571 ± 0.0044</td><td>(-7.1%) 0.01861 ± 0.00089</td></tr><tr><td>CoMM-Based</td><td>(+0.0%) 0.01770 ± 0.0028 (-22.3%)</td><td>(+0.0%) 0.03458 ± 0.0057 (-17.8%)</td><td>(+0.0%) 0.01262 ± 0.0020</td><td>(+0.0%) 0.02366 ± 0.0028</td><td>(-2.4%) 0.05829 ± 0.0065</td><td>(+0.0%) 0.01424 ± 0.0020</td></tr><tr><td>LUMIRMC</td><td>0.02538 ± 0.0021 (+11.4%)</td><td>0.05241 ± 0.0038 (+24.5%)</td><td>(-25.8%) 0.01733 ± 0.0017 (+1.9%)</td><td>(-17.6%) 0.03361 ± 0.0020 (+17.0%)</td><td>(-13.4%) 0.08517 ± 0.0039 (+26.5%)</td><td>(-23.5%) 0.01955 ± 0.0017</td></tr><tr><td>LUMIRMC+SA</td><td>0.02317 ± 0.0025 (+1.7%)</td><td>0.04480 ± 0.0056 (+6.4%)</td><td>0.01664 ± 0.0017 (-2.2%)</td><td>0.03032 ± 0.0031 (+5.6%)</td><td>0.07332 ± 0.0081 (+8.9%)</td><td>(+5.1%) 0.01857 ± 0.0019 (-0.2%)</td></tr></table>

(c) GRU4Rec Backbone
<table><tr><td rowspan="2">Embedding</td><td colspan="3">@10</td><td colspan="3">@20</td></tr><tr><td>NDCG</td><td>HitRate</td><td>MRR</td><td>NDCG</td><td>HitRate</td><td>MRR</td></tr><tr><td>Trajectory-Only</td><td>0.01650 ± 0.0016 (-13.5%)</td><td>0.03376 ± 0.0040 (-10.1%)</td><td>0.01132 ± 0.0010 (-16.3%)</td><td>0.02159 ± 0.0016 (-14.5%)</td><td>0.05395 ± 0.0041 (-14.7%)</td><td>0.01271 ± 0.0010 (-16.3%)</td></tr><tr><td>Full Concatenation</td><td>0.01464 ± 0.0010 (-23.2%)</td><td>0.03068 ± 0.0017 (-18.3%)</td><td>0.009856 ± 0.0012 (-27.1%)</td><td>0.01906 ± 0.0013 (-24.5%)</td><td>0.04843 ± 0.0023 (-23.5%)</td><td>0.01104 ± 0.0012 (-27.3%)</td></tr><tr><td>Modality-Weighted</td><td>0.01814 ± 0.0019 (-4.9%)</td><td>0.03621 ± 0.0049</td><td>0.01273 ± 0.0012</td><td>0.02495 ± 0.0018</td><td>0.06327 ± 0.0046</td><td>0.01459 ± 0.0011</td></tr><tr><td>FLAVA-Based</td><td>0.01907 ± 0.0017</td><td>(-3.6%) 0.03756 ± 0.0039</td><td>(-5.8%) 0.01352 ± 0.0014</td><td>(-1.2%) 0.02525 ± 0.0016</td><td>(+0.0%) 0.06218 ± 0.0042</td><td>(-3.9%) 0.01519 ± 0.0014</td></tr><tr><td>CoMM-Based</td><td>(+0.0%) 0.01748 ± 0.0014</td><td>(+0.0%) 0.03476 ± 0.0031</td><td>(+0.0%) 0.01230 ± 0.0016</td><td>(+0.0%) 0.02314 ± 0.0014</td><td>(-1.7%) 0.05730 ± 0.0043</td><td>(+0.0%) 0.01384 ± 0.0015</td></tr><tr><td>LUMIRMC</td><td>(-8.3%) 0.02103 ± 0.0024</td><td>(-7.5%) 0.04191 ± 0.0035</td><td>(-9.0%) 0.01476 ± 0.0023</td><td>(-8.4%) 0.02629 ± 0.0023</td><td>(-9.4%) 0.06309 ± 0.0033</td><td>(-8.9%) 0.01615 ± 0.0023</td></tr><tr><td>LUMIRMC+SA</td><td>(+10.3%) 0.01758 ± 0.0022 (-7.8%)</td><td>(+11.6%) 0.03467 ± 0.0043 (-7.7%)</td><td>(+9.2%) 0.01244 ± 0.0019 (-8.0%)</td><td>(+4.1%) 0.02235 ± 0.0024 (-11.5%)</td><td>(-0.3%) 0.05358 ± 0.0054</td><td>(+6.3%) 0.01374 ± 0.0019 (-9.5%)</td></tr></table>

Bold and underline indicate the highest mean value for each metric and the best-performing baseline, respectively. Results are averaged over 8 runs after removing the highest and lowest from 10 independent runs. Percentages denote relative performance diferences with respect to the best-performing baseline.

Statistical significance results across backbones
<table><tr><td></td><td colspan="3">BERT4Rec</td><td colspan="3">SASRec</td><td colspan="3">GRU4Rec</td></tr><tr><td>Embedding</td><td>NDCG</td><td>HitRate</td><td>MRR</td><td>NDCG</td><td>HitRate</td><td>MRR</td><td>NDCG</td><td>HitRate</td><td>MRR</td></tr><tr><td colspan="10"></td></tr><tr><td>LUMIRMC LUMIRMC+SA</td><td>0.005087 0.002189‡</td><td>0.000327‡ 0.000265</td><td>0.065465</td><td>0.009222</td><td>0.000028</td><td>0.660621</td><td>0.084496</td><td>0.035103†</td><td>0.219529</td></tr><tr><td></td><td></td><td></td><td>0.017876†</td><td>0.689722</td><td>0.235593 @20</td><td>0.613663↓</td><td>0.159233↓</td><td>0.179227↓</td><td>0.212216↓</td></tr><tr><td></td><td colspan="9"></td></tr><tr><td>LUMIRMC LUMIRMC+SA</td><td>0.001152 0.000091</td><td>0.001738 0.000018</td><td>0.017595† 0.005237</td><td>0.000070 0.202630</td><td>0.000020 0.122294</td><td>0.191109 0.955319↓</td><td>0.307402 0.013921↓†</td><td>0.929254↓ 0.001717↓</td><td>0.325471 0.096983↓</td></tr></table>

†: � < 0.05, ‡: � < 0.01, based on Welch’s t-test comparing each LUMIR variant with the corresponding strongest baseline. ↓ indicates that the corresponding LUMIR variant yields lower performance than the strongest baseline.

Table 2 reports the performance ofLUMIR and baseline embedding strategies across three downstream sequential recommendation backbones.<sup>4</sup> To examine the impact of the sequential supervision signal $\mathcal { L } _ { \mathrm { S A } }$ on embedding quality, an ablation variant is considered in which representations are learned using only the modality compression objective $\mathcal { L } _ { \mathrm { M C } }$ , without incorporating $\mathcal { L } _ { \mathrm { S A } }$ . These two configurations are denoted as $\mathrm { L U M I R } _ { \mathrm { M C + S A } }$ and $\mathrm { L U M I R } _ { \mathrm { M C } }$ , respectively. <sup>5</sup> The following discussion is organized from the most pronounced performance diferences across backbones to the least.

The most pronounced improvements are observed under the BERT4Rec backbone. POI representations learned by LUMIR exhibit consistent and substantial improvements over all baselines across accuracy-oriented metrics. Compared to the strongest baseline, $\mathrm { L U M I R } _ { \mathrm { M C + S A } }$ achieves improvements of up to +24.2% in HR@10 and +28.9% in HR@20. Beyond accuracy, LUMIR also demonstrates clear advantages. It substantially increases coverage while reducing the Gini index, indicating a more balanced recommendation distribution. At the same time, it achieves the highest serendipity scores across both cutofs, suggesting improved ability to recommend relevant yet unexpected POIs. These results indicate that the joint objective enhances both predictive performance and the diversity-related properties of the learned representations.

Moderate but consistent improvements are observed under the SASRec backbone. LUMIR remains competitive with, and in several cases superior to, baseline methods. In particular, $\mathrm { L U M R } _ { \mathrm { M C } }$ consistently delivers the strongest performance, indicating that modality compression alone is suficient to yield substantial improvements in this setting. In terms of accuracy-oriented metrics, $\mathrm { L U M I R _ { M C } }$ achieves the highest performance across most metrics, with notable gains such as +24.5% in HR@10 and +26.5% in HR@20 compared to the strongest baseline. In contrast, incorporating the sequence-aware alignment objective does not lead to equally pronounced improvements, suggesting that its contribution is less dominant under this backbone. For beyond-accuracy metrics, LUMIR continues to demonstrate clear advantages. LUMIR variants achieve the best results across all metrics, improving both predictive accuracy and diversity-related properties.

In contrast, the diferences become less pronounced under the GRU4Rec backbone. $\mathrm { L U M I R } _ { \mathrm { M C } }$ continues to provide consistent improvements over most baselines across accuracy-oriented metrics. Except for HitRate@20, where it remains comparable to the strongest baseline, $\mathrm { L U M I R } _ { \mathrm { M C } }$ achieves higher performance on all other metrics, with gains such as +10.3% in NDCG@10, +11.6% in HR@10, and +9.2% in MRR@10. Compared to LUMIR<sub>MC</sub>, incorporating the sequence-aware alignment objective does not yield additional benefits for GRU4Rec. For beyond-accuracy metrics, $\mathrm { L U M I R } _ { \mathrm { M C } }$ achieves the best performance on key indicators such as serendipity and novelty, but the overall advantage is less pronounced compared to BERT4Rec and SASRec.

Statistical Significance Analysis. To assess whether the observed performance improvements are statistically reliable, statistical significance is evaluated based on per-user evaluation results. Specifically, for each backbone and accuracy-oriented metric, performance distributions are collected across multiple trials, and Welch’s t-test is applied to compare each LUMIR variant against the strongest baseline.

As reported in Table 3, the improvements achieved by LUMIR are statistically significant under the BERT4Rec backbone across most accuracy-oriented metrics, with $p < 0 . 0 1$ in the majority of cases. This confirms that the substantial gains observed in Table 2 are not due to random variation but reflect consistent improvements.

Under SASRec and GRU4Rec, the statistical evidence is comparatively weaker, with fewer metrics reaching significance thresholds than in BERT4Rec. Nevertheless, LUMIR continues to achieve statisti cally significant improvements on several metrics, particularly for the modality compression variant, indicating that the gains remain consistent across instances.

Cross-Backbone Analysis. The performance of LUMIR varies across downstream backbones. While $\mathrm { L U M I R } _ { \mathrm { M C + S A } }$ improves performance under BERT4Rec, its impact is weaker or less consistent under SASRec and GRU4Rec. The reasons for these diferences across backbones are as follows.

This discrepancy can be attributed to the interaction between the alignment objective and the inductive biases of diferent sequential models. BERT4Rec adopts a bidirectional self-attention mechanism, where each position is conditioned on both past and future context [35]. In this setting, the sequenceaware alignment objective is naturally compatible, as it encourages representations to capture global relational dependencies within the sequence. In contrast, SASRec and GRU4Rec follow an autoregressive paradigm, where predictions are strictly conditioned on past interactions [36, 37]. Under such unidirectional modeling, the contribution of the alignment objective becomes less dominant, as these models primarily rely on directional sequential dependencies. Nevertheless, the learned representations remain efective, and the modality compression objective alone already provides strong performance gains. These observations suggest that LUMIR can adapt to diferent sequential architectures, while the alignment objective yields the greatest benefits when global contextual modeling is supported.

## 7. Conclusion and Future Work

This work proposes LUMIR, a decoupled multimodal representation learning framework that learns unified POI representations from multimodal tourism-related information collected from heterogeneous data sources by leveraging LLM-based semantic unification and contrastive learning. By separating representation learning from downstream objectives, LUMIR captures intrinsic cross-modal interactions while incorporating sequence-aware signals as a weak structural constraint. Experimental results across multiple sequential recommendation backbones show that LUMIR consistently improves both accuracyoriented and beyond-accuracy metrics, demonstrating the efectiveness of decoupled multimodal representation learning from heterogeneous data sources.

Future work will explore more efective representation learning strategies for multimodal information collected from heterogeneous data sources, particularly in the presence of noisy, redundant, or only partially informative modality-specific signals. Specifically, new learning objectives could be developed to better distinguish informative cross-modal interactions from noise while capturing redundancy, complementarity, and semantic consistency across multimodal information. Extending LUMIR to more diverse multimodal recommendation scenarios, such as user-centric recommendation and crossdomain applications, will enable further evaluate of the robustness and transferability of the learned representations across heterogeneous data sources.

## Declaration on Generative AI

During the preparation of this work, the authors used ChatGPT in order to perform grammar and spelling checks and minor language polishing. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the publication’s content.

## References

[1] Q. Liu, J. Hu, Y. Xiao, X. Zhao, J. Gao, W. Wang, Q. Li, J. Tang, Multimodal recommender systems: A survey, ACM Comput. Surv. 57 (2024). doi:10.1145/3695461.

[2] Y.-H. H. Tsai, P. P. Liang, A. Zadeh, L.-P. Morency, R. Salakhutdinov, Learning factorized multimodal representations, in: International Conference on Learning Representations, 2019.

[3] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, E. Chen, A survey on multimodal large language models, National Science Review 11 (2024). doi:10.1093/nsr/nwae403.

[4] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, A. Y. Ng, Multimodal deep learning, in: Proceedings of the 28th International Conference on Machine Learning, 2011, pp. 689–696.

[5] Y. Bengio, A. Courville, P. Vincent, Representation learning: A review and new perspectives, IEEE Trans. Pattern Anal. Mach. Intell. 35 (2013) 1798–1828. doi:10.1109/TPAMI.2013.50.

[6] Y. Xie, P. Zhou, S. Kim, Decoupled side information fusion for sequential recommendation, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 1611–1621. doi:10.1145/3477495.3531963.

[7] J. Yu, L. Guo, J. Zhang, G. Wang, A survey on graph neural network-based next POI recommendation for smart cities, Journal of Reliable Intelligent Environments 10 (2024) 299–318. doi:10.1007/s40860-024-00233-z.

[8] Z. Wang, W. Höpken, D. Jannach, A survey on point-of-interest recommendations leveraging heterogeneous data, Information Technology & Tourism 27 (2025) 29–73. doi:10.1007/ s40558-024-00301-3.

[9] Y. Xu, G. Cong, L. Zhu, L. Cui, MMPOI: A multi-modal content-aware framework for POI recommendations, in: Proceedings of the ACM Web Conference 2024, WWW ’24, 2024, pp. 3454–3463. doi:10.1145/3589334.3645449.

[10] Z. Huang, J. Ma, Y. Dong, N. Z. Foutz, J. Li, Empowering next POI recommendation with multirelational modeling, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, 2022, pp. 2034–2038. doi:10.1145/3477495. 3531801.

[11] B. Dufumier, J. Castillo Navarro, D. Tuia, J.-P. Thiran, What to align in multimodal contrastive learning?, in: International Conference on Learning Representations, 2025, pp. 5408–5432.

[12] C. Pomo, M. Attimonelli, D. Danese, F. Narducci, T. Di Noia, Do recommender systems really leverage multimodal content? A comprehensive analysis on multimodal representations for recommendation, in: Proceedings of the 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, 2025, pp. 2377–2387. doi:10.1145/3746252.3761398.

[13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning, 2021, pp. 8748–8763.

[14] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, K. Simonyan, Flamingo: A visual language model for few-shot learning, in: Advances in Neural Information Processing Systems, volume 35, 2022, pp. 23716–23736. doi:10.52202/068431-1723.

[15] Z. Liang, Y. Xu, Y. Hong, P. Shang, Q. Wang, Q. Fu, K. Liu, A survey of multimodel large language

models, in: Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering, CAICE ’24, 2024, pp. 405–409. doi:10.1145/3672758.3672824.

[16] Z. Wang, W. Höpken, D. Jannach, Beyond visit trajectories: Enhancing POI recommendation via LLM-augmented text and image representations, in: Proceedings of the 19th ACM Conference on Recommender Systems, 2025, pp. 521–526. doi:10.1145/3705328.3748014.

[17] J. Li, D. Li, S. Savarese, S. Hoi, BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, in: Proceedings of the 40th International Conference on Machine Learning, ICML’23, 2023, pp. 19730–19742.

[18] H. Liu, C. Li, Q. Wu, Y. J. Lee, Visual instruction tuning, in: Advances in Neural Information Processing Systems, volume 36, 2023, pp. 34892–34916. doi:10.52202/075280-1516.

[19] Y. Qin, Y. Wang, F. Sun, W. Ju, X. Hou, Z. Wang, J. Cheng, J. Lei, M. Zhang, DisenPOI: Disentangling sequential and geographical influence for point-of-interest recommendation, in: Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, WSDM ’23, 2023, pp. 508–516. doi:10.1145/3539597.3570408.

[20] T. Alenezi, S. Hirtle, Normalized attraction travel personality representation for improving travel recommender systems, IEEE Access 10 (2022) 56493–56503. doi:10.1109/ACCESS.2022. 3178439.

[21] L. Chen, D. Lyu, S. Yu, G. Chen, Multi-level visual similarity based personalized tourist attraction recommendation using geo-tagged photos, ACM Trans. Knowl. Discov. Data 17 (2023). doi:10. 1145/3582015.

[22] S. Wang, B. Xie, L. Ding, X. Gao, J. Chen, Y. Xiang, SeCor: Aligning semantic and collaborative representations by large language models for next-point-of-interest recommendations, in: Proceedings of the 18th ACM Conference on Recommender Systems, RecSys ’24, 2024, pp. 1–11. doi:10.1145/3640457.3688124.

[23] T. Baltrusaitis, C. Ahuja, L.-P. Morency, Multimodal machine learning: A survey and taxonomy, IEEE Trans. Pattern Anal. Mach. Intell. 41 (2019) 423–443. doi:10.1109/TPAMI.2018.2798607.

[24] P. Gramacki, S. Woźniak, P. Szymański, gtfs2vec: Learning GTFS embeddings for comparing public transport ofer in microregions, in: Proceedings of the 1st ACM SIGSPATIAL International Workshop on Searching and Mining Large Collections of Geospatial Data, GeoSearch’21, 2021, pp. 5–12. doi:10.1145/3486640.3491392.

[25] S. Woźniak, P. Szymański, hex2vec: Context-aware embedding H3 hexagons with OpenStreetMap tags, in: Proceedings of the 4th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery, GEOAI ’21, 2021, pp. 61–71. doi:10.1145/3486635.3491076.

[26] L. Zhang, X. Zhou, Z. Zeng, Z. Shen, Multimodal pre-training for sequential recommendation via contrastive learning, ACM Trans. Recomm. Syst. 3 (2024). doi:10.1145/3682075.

[27] K. He, H. Fan, Y. Wu, S. Xie, R. Girshick, Momentum contrast for unsupervised visual representation learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 9729–9738.

[28] Y. Hou, S. Mu, W. X. Zhao, Y. Li, B. Ding, J.-R. Wen, Towards universal sequence representation learning for recommender systems, in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, 2022, pp. 585–593. doi:10.1145/3534678. 3539381.

[29] R. Qiu, Z. Huang, H. Yin, Z. Wang, Contrastive learning for representation degeneration problem in sequential recommendation, in: Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, WSDM ’22, 2022, pp. 813–823. doi:10.1145/3488560.3498433.

[30] W. Zhao, S. Zhong, Y. Liu, W. Wen, J. Qin, M. Liang, Z. Huang, DVIB: Towards robust multimodal recommender systems via variational information bottleneck distillation, in: Proceedings of the ACM on Web Conference 2025, WWW ’25, 2025, pp. 2549–2561. doi:10.1145/3696410. 3714840.

[31] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, C. Sun, Attention bottlenecks for multimodal fusion, in: Advances in Neural Information Processing Systems, volume 34 of NIPS ’21, 2021, pp. 14200–14213.

[32] S. Mai, Y. Zeng, H. Hu, Multimodal information bottleneck: Learning minimal suficient unimodal and multimodal representations, IEEE Transactions on Multimedia 25 (2023) 4121–4134. doi:10. 1109/TMM.2022.3171679.

[33] J. Fu, X. Ge, X. Xin, A. Karatzoglou, I. Arapakis, J. Wang, J. M. Jose, IISAN: Eficiently adapting multimodal representation for sequential recommendation with decoupled PEFT, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, 2024, pp. 687–697. doi:10.1145/3626772.3657725.

[34] J. Wang, Z. Zeng, Y. Wang, Y. Wang, X. Lu, T. Li, J. Yuan, R. Zhang, H.-T. Zheng, S.-T. Xia, MISSRec: Pre-training and transferring multi-modal interest-aware sequence representation for recommendation, in: Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, 2023, pp. 6548–6557. doi:10.1145/3581783.3611967.

[35] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, P. Jiang, BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer, in: Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019, pp. 1441–1450. doi:10.1145/3357384.3357895.

[36] W.-C. Kang, J. McAuley, Self-attentive sequential recommendation, in: 2018 IEEE International Conference on Data Mining, 2018, pp. 197–206. doi:10.1109/ICDM.2018.00035.

[37] B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk, Session-based recommendations with recurrent neural networks, in: International Conference on Learning Representations, 2016.

[38] A. Boz, W. Zorgdrager, Z. Kotti, J. Harte, P. Louridas, V. Karakoidas, D. Jannach, M. Fragkoulis, Improving sequential recommendations with LLMs, ACM Trans. Recomm. Syst. 4 (2025). doi:10. 1145/3711667.

[39] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, D. Kiela, FLAVA: A foundational language and vision alignment model, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15638–15650.

[40] B. Rychalska, P. Bąbel, K. Gołuchowski, A. Michałowski, J. Dąbrowski, P. Biecek, Cleora: A simple, strong and scalable graph embedding scheme, in: International Conference on Neural Information Processing, 2021, pp. 338–352. doi:10.1007/978-3-030-92273-3\_28.

[41] J. Wu, B. Yang, R. Mao, Q. Li, Popularity-aware sequential recommendation with user desire, Expert Systems with Applications 237 (2024) 121429. doi:10.1016/j.eswa.2023.121429.

[42] H. Li, J. Song, L. Gao, P. Zeng, H. Zhang, G. Li, A diferentiable semantic metric approximation in probabilistic embedding for cross-modal retrieval, in: Advances in Neural Information Processing Systems, volume 35, 2022, pp. 11934–11946. doi:10.52202/068431-0867.

[43] X. Wang, Y. Hua, E. Kodirov, N. M. Robertson, Ranked list loss for deep metric learning, IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (2022) 5414–5429. doi:10.1109/ TPAMI.2021.3068449.

[44] D. Jannach, L. Lerche, I. Kamehkhosh, M. Jugovac, What recommenders recommend: An analysis of recommendation biases and possible countermeasures, User Modeling and User-Adapted Interaction 25 (2015) 427–491. doi:10.1007/s11257-015-9165-3.

[45] T. Zhou, Z. Kuscsik, J.-G. Liu, M. Medo, J. R. Wakeling, Y.-C. Zhang, Solving the apparent diversityaccuracy dilemma of recommender systems, Proceedings of the National Academy of Sciences 107 (2010) 4511–4515. doi:10.1073/pnas.1000488107.

[46] M. Ge, C. Delgado-Battenfeld, D. Jannach, Beyond accuracy: Evaluating recommender systems by coverage and serendipity, in: Proceedings of the 4th ACM Conference on Recommender Systems, RecSys ’10, 2010, pp. 257–260. doi:10.1145/1864708.1864761.

## A Additional Beyond-accuracy Results on the Yelp Dataset

## Table 1: Performance with diferent sequential recommendation backbones

(a) BERT4Rec Backbone
<table><tr><td></td><td>Embedding</td><td>Coverage</td><td>Gini</td><td>Serendipity</td><td>Novelty</td></tr><tr><td></td><td></td><td colspan="4">@10</td></tr><tr><td></td><td>Trajectory-Only</td><td>0.08957±0.013</td><td>0.9657±0.0051</td><td>0.0377±0.0049</td><td>13.26±0.10</td></tr><tr><td></td><td>Full Concatenation</td><td>0.02292±0.0046</td><td>0.9914±0.0013</td><td>0.04073±0.0042</td><td>12.23±0.37</td></tr><tr><td></td><td>Modality-Weighted</td><td>0.04937±0.018</td><td>0.9825±0.0065</td><td>0.04716±0.0023</td><td>12.69±0.43</td></tr><tr><td></td><td>FLAVA-Based</td><td>0.03321±0.018</td><td>0.9874±0.0065</td><td>0.04417±0.0019</td><td>12.49±0.43</td></tr><tr><td></td><td>CoMM-Based</td><td>0.03319±0.013</td><td>0.9874±0.0049</td><td>0.04354±0.0015</td><td>12.76±0.20</td></tr><tr><td></td><td>LUMIRMC</td><td>0.1253±0.012</td><td>0.9516±0.0055</td><td>0.05196±0.0021</td><td>12.83±0.33</td></tr><tr><td></td><td> $\mathrm { L U M I R } _ { \mathrm { M C + S A } }$ </td><td>0.1515±0.019</td><td>0.9385±0.010</td><td>0.05739±0.0057</td><td>12.88±0.48</td></tr><tr><td></td><td></td><td colspan="4"></td></tr><tr><td>Beyaoaacy</td><td>Trajectory-Only</td><td>0.1431±0.019</td><td>0.9454±0.0074</td><td>0.06236±0.0058</td><td>13.29±0.10</td></tr><tr><td></td><td>Full Concatenation</td><td>0.03882±0.0072</td><td></td><td>0.06191±0.0055</td><td>12.32±0.37</td></tr><tr><td></td><td>Modality-Weighted</td><td>0.08263±0.029</td><td>0.9845±0.0023</td><td>0.07205±0.0080</td><td></td></tr><tr><td></td><td>FLAVA-Based</td><td> $0 . 0 5 6 0 6 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td>0.9696±0.010</td><td>0.06434±0.0036</td><td>12.79±0.39</td></tr><tr><td></td><td>CoMM-Based</td><td> $0 . 0 5 5 8 8 { \pm } 0 . 0 1 9$ </td><td>0.9774±0.011</td><td>0.06571±0.0041</td><td>12.57±0.42</td></tr><tr><td></td><td></td><td>0.2004±0.017</td><td>0.9776±0.0077</td><td>0.08526±0.0054</td><td>12.83±0.18</td></tr><tr><td></td><td>LUMIRMC</td><td></td><td>0.9237±0.0080</td><td></td><td>12.90±0.29</td></tr><tr><td></td><td>LUMIRMC+SA</td><td>0.2408±0.025</td><td>0.9057±0.014</td><td>0.09377±0.0062</td><td>12.95±0.41</td></tr></table>

(b) SASRec Backbone
<table><tr><td></td><td>Embedding</td><td>Coverage</td><td>Gini</td><td>Serendipity</td><td>Novelty</td></tr><tr><td></td><td></td><td colspan="4">@10</td></tr><tr><td></td><td>Trajectory-Only</td><td>0.02308±0.013</td><td>0.9915±0.0035</td><td>0.02969±0.0051</td><td>11.58±0.65</td></tr><tr><td></td><td>Full Concatenation</td><td> $0 . 0 4 8 5 8 { \scriptstyle \pm 0 . 0 0 7 3 }$ </td><td>0.9833±0.0029</td><td> $0 . 0 3 2 7 7 { \scriptstyle \pm 0 . 0 0 4 5 }$ </td><td>12.61±0.18</td></tr><tr><td></td><td>Modality-Weighted</td><td>0.04860±0.011</td><td>0.9849±0.0042</td><td>0.04064±0.0028</td><td>12.52±0.21</td></tr><tr><td></td><td>FLAVA-Based</td><td>0.04901±0.0097</td><td>0.9830±0.0035</td><td>0.04019±0.0025</td><td>12.23±0.14</td></tr><tr><td></td><td>CoMM-Based</td><td>0.04633±0.016</td><td>0.9836±0.0048</td><td>0.03313±0.0059</td><td>12.52±0.47</td></tr><tr><td></td><td>LUMIRMC</td><td>0.1013±0.010</td><td>0.9658±0.0052</td><td>0.05114±0.0038</td><td>12.73±0.14</td></tr><tr><td></td><td>LUMIR MC+SA</td><td>0.1434±0.010</td><td>0.9467±0.0050</td><td>0.04327±0.0062</td><td>12.59±0.45</td></tr><tr><td></td><td colspan="5">@20</td></tr><tr><td>Beyd-aoacy</td><td>Trajectory-Only</td><td>0.03916±0.022</td><td>0.9847±0.0062</td><td>0.04779±0.0084</td><td>11.76±0.60</td></tr><tr><td></td><td>Full Concatenation</td><td>0.07992±0.011</td><td>0.9719±0.0042</td><td>0.05413±0.0071</td><td>12.70±0.16</td></tr><tr><td></td><td>Modality-Weighted</td><td>0.08107±0.018</td><td>0.9746±0.0068</td><td> $0 . 0 6 1 1 9 { \scriptstyle \pm 0 . 0 0 5 9 }$ </td><td>12.65±0.21</td></tr><tr><td></td><td>FLAVA-Based</td><td> $0 . 0 8 0 4 2 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>0.9717±0.0055</td><td> $0 . 0 5 9 4 7 { \scriptstyle \pm 0 . 0 0 4 5 }$ </td><td>12.39±0.12</td></tr><tr><td></td><td>CoMM-Based</td><td> $0 . 0 7 6 2 1 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.9719±0.0076</td><td> $0 . 0 5 3 6 7 { \scriptstyle \pm 0 . 0 0 6 6 }$ </td><td>12.63±0.43</td></tr><tr><td></td><td>LUMIRMC</td><td>0.1607±0.014</td><td> $0 . 9 4 7 0 { \scriptstyle \pm 0 . 0 0 7 2 }$ </td><td>0.07974±0.0041</td><td>12.81±0.15</td></tr><tr><td></td><td>LUMIR MC+SA</td><td>0.2315±0.018</td><td>0.9161±0.0076</td><td>0.06942±0.0083</td><td>12.74±0.39</td></tr></table>

(c) GRU4Rec Backbone
<table><tr><td></td><td>Embedding</td><td>Coverage</td><td>Gini</td><td>Serendipity</td><td>Novelty</td></tr><tr><td></td><td>0.9826±0.0031</td><td colspan="4">@10</td></tr><tr><td></td><td>Trajectory-Only</td><td>0.06417±0.013</td><td></td><td>0.03050±0.0039</td><td>11.62±0.11</td></tr><tr><td></td><td>Full Concatenation</td><td>0.01215±0.0047</td><td>0.9949±0.0016</td><td>0.02552±0.0016</td><td>11.34±0.21</td></tr><tr><td></td><td>Modality-Weighted</td><td>0.04122±0.012</td><td>0.9848±0.0046</td><td>0.03358±0.0057</td><td>11.99±0.29</td></tr><tr><td></td><td>FLAVA-Based</td><td>0.02857±0.0089</td><td>0.9889±0.0035</td><td>0.03394±0.0046</td><td>11.77±0.24</td></tr><tr><td></td><td>CoMM-Based</td><td> $0 . 0 2 7 8 6 { \scriptstyle \pm 0 . 0 0 5 7 }$ </td><td>0.9893±0.0021</td><td>0.03114±0.0036</td><td>11.78±0.14</td></tr><tr><td></td><td>LUMIRMC</td><td> $0 . 0 4 3 9 8 { \pm } 0 . 0 1 9$ </td><td>0.9833±0.0067</td><td>0.03892±0.0045</td><td>12.14±0.44</td></tr><tr><td></td><td> $\mathrm { L U M I R } _ { \mathrm { M C + S A } }$ </td><td>0.02653±0.011</td><td>0.9898±0.0039</td><td>0.03105±0.0045</td><td>11.73±0.38</td></tr><tr><td></td><td colspan="4">@20</td><td></td></tr><tr><td>BEeyaarac</td><td>Trajectory-Only</td><td>0.1081±0.027</td><td>0.9716±0.0048</td><td>0.04499±0.0040</td><td>11.83±0.095</td></tr><tr><td></td><td>Full Concatenation</td><td>0.02090±0.0080</td><td>0.9905±0.0027</td><td>0.03856±0.0029</td><td>11.55±0.18</td></tr><tr><td></td><td>Modality-Weighted</td><td> $0 . 0 6 9 4 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td>0.9735±0.0077</td><td>0.05621±0.0060</td><td>12.16±0.27</td></tr><tr><td></td><td>FLAVA-Based</td><td> $0 . 0 4 8 0 4 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td>0.9804±0.0059</td><td>0.05377±0.0050</td><td>11.95±0.22</td></tr><tr><td></td><td>CoMM-Based</td><td>0.04705±0.0093</td><td>0.9809±0.0034</td><td>0.04915±0.0044</td><td>11.96±0.12</td></tr><tr><td></td><td>LUMIRMC</td><td> $0 . 0 7 2 6 4 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td>0.9715±0.011</td><td>0.05567±0.0047</td><td>12.28±0.39</td></tr><tr><td></td><td>LUMIRMC+SA</td><td>0.04502±0.018</td><td>0.9821±0.0065</td><td>0.04526±0.0064</td><td>11.91±0.35</td></tr></table>

Bold and underline indicate the highest mean value for each metric and the best-performing baseline, respectively. Results are averaged over 8 runs after removing the highest and lowest from 10 independent runs.