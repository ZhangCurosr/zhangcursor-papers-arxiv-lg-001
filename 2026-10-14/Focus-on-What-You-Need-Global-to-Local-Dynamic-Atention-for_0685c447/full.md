# Focus on What You Need: Global-to-Local Dynamic Atention for Multimodal Sarcasm Detection

Dehao Wu<sup>∗</sup> University of Illinois Urbana-Champaign dehaowu2@illinois.edu

## Abstract

Multimodal sarcasm detection has attracted increasing attention due to the widespread use of images and text on social media platforms. Existing studies generally treat the semantic or emotional inconsistency between images and text as a key cue for identifying sarcastic expressions. However, images in real-world scenarios often contain multiple visual objects, many of which are irrelevant to sarcasm-related semantics. Such redundant visual information may introduce noise during cross-modal interaction, distracting models from focusing on critical sarcasm cues and potentially leading to inaccurate predictions. To address this issue, we propose GLDA (Global-to-Local Dynamic Attention), a multimodal sarcasm detection framework that models cross-modal interactions from a globalto-local perspective. Specifically, GLDA first employs a global multimodal fusion module to capture high-level semantic interactions between visual and textual modalities, enabling the model to identify coarse-grained cross-modal inconsistencies. To further reduce the influence of irrelevant objects, GLDA introduces a local gated cross-modal attention mechanism that dynamically filters redundant visual and textual features and refines multimodal representations at a finer granularity. In addition, a multi-branch learning objective is designed to jointly optimize unimodal and multimodal representations. Experimental results demonstrate that GLDA effectively improves sarcasm detection performance and achieves competitive results compared with existing approaches. The code will be available at https://anonymous.4open.science/r/GLDA.

## CCS Concepts

• Information systems → Multimedia information systems; Sentiment analysis; • Computing methodologies → Information extraction.

## Keywords

Multimedia Content Understanding, Sarcasm Detection, Sentiment Analysis, Data Mining

## 1 Introduction

Sarcasm, a ubiquitous form of figurative language, operates on the delicate tension between literal expression and intended sentiment. Given its profound impact on sentiment analysis and public opinion monitoring, the task of sarcasm detection has evolved into a pivotal research frontier [18, 30]. While early methodologies focused exclusively on linguistic cues [1, 17], the surge of multimedia content has rendered text-only analysis insuficient. In the absence of visual or acoustic context, the subtle "ironic gap" in user-generated content often remains elusive, driving the field toward multimodal sarcasm detection (MSD) [29, 33, 36, 49].

Xingjie Zhuang<sup>∗</sup> Xiamen University zhuangxj@xmu.stu.edu.cn

![](images/ec30a85a109133cc8651a9c71e0fdcd0adbd28a401657a398c75a3bcbd70bba6.jpg)  
Figure 1: Comparison of our method with the other exist multimodal methods on the multimodal sarcasm detection task.

Contemporary MSD research has shifted from simple feature aggregation to sophisticated cross-modal interaction. For instance, researchers have leveraged interactive graph convolutional networks to map intricate intra-modal and inter-modal inconsistencies [22], or harnessed multi-granularity cues within the multi-view framework to capture nuanced sarcastic signals [33]. Furthermore, by integrating external knowledge, recent models have begun to simulate the hierarchical semantic coherence between text and imagery [24]. Beyond architectural innovation, a new paradigm is emerging that addresses model robustness and fairness. To combat inherent dataset flaws, recent studies have introduced causal-based debiasing frameworks to strip away statistical label biases [49]. Others have pioneered counterfactual data augmentation, utilizing adaptive contrastive learning to decouple sarcastic intent from biased textual factors [13], thereby ensuring more generalized and objective detection

Based on the aforementioned studies, it is not dificult to observe that a general consensus has been formed in the field of multimodal sarcasm detection: most researchers regard the semantic or emotional inconsistency between images and text as an important cue for understanding sarcastic expressions.

However, in practical sarcasm examples, the input image often contains multiple visual objects. It is important to note that not all objects in an image are strongly related to the sarcastic cues expressed in the corresponding text. In many cases, a large portion of the visual content is essentially redundant information. Such redundant visual signals not only fail to assist the model in capturing the key sarcasm cues, but may also distract the model from focusing on truly relevant elements, thereby weakening the efectiveness of cross-modal reasoning. Even recent multimodal large language models are not immune to this problem when dealing with images that contain substantial visual distractions, as illustrated in Figure 1. This is largely due to architectural characteristics and context-length constraints of multimodal large language models, such as Gemini, which prevent them from simultaneously reasoning over all visual objects in complex scenes. As a result, these models tend to adaptively focus on a limited number of objects that they consider most relevant to the textual context during in ference. Nonetheless, when redundant objects are present in the image, they may interfere with the model’s selection process and lead the model to attend to suboptimal or irrelevant visual cues, thereby afecting the overall reasoning quality. This issue becomes particularly pronounced in approaches that rely on graph-based structures or region-level interaction modeling, which are widely regarded as mainstream paradigms for modeling fine-grained crossmodal interactions, where a large number of irrelevant nodes can introduce significant noise and hinder the modeling of meaningful cross-modal relationships.

Notably, to alleviate this issue, some recent studies attempt to introduce additional visual features extracted by object detection models such as Faster R-CNN and YOLO, aiming to obtain more finegrained region-level representations for subsequent cross-modal interaction. Although this strategy may reduce the risk of missing potentially important visual objects during feature extraction, it essentially increases the number of candidate regions rather than addressing the core challenge of filtering irrelevant information. Consequently, the additional object-level features may still contain a large amount of noise that is unrelated to sarcasm understanding. Moreover, incorporating extra detection models inevitably introduces additional computational overhead and increases the overall model complexity.

To tackle these challenges, this paper proposes a multimodal sarcasm detection framework that models cross-modal interactions from a global-to-local perspective and adopts a multi-branch collaborative learning strategy. Specifically, the model first employs a global multimodal fusion module to capture the overall semantic interactions between visual and textual modalities, enabling the model to identify high-level cross-modal inconsistencies. Subsequently, a local gated cross-modal attention mechanism is introduced to refine the information flow at a finer granularity. Through a gating mechanism, the model can dynamically filter irrelevant visual objects and textual tokens, thereby focusing on sarcasmrelated semantic cues while suppressing redundant information. In addition, a multi-branch training strategy is adopted to jointly op timize unimodal and multimodal representations, which helps the model learn robust modality-specific features while maintaining efective cross-modal interaction.

By modeling multimodal interactions in a global-to-local manner, the proposed framework is able to capture both coarse-grained semantic inconsistencies and fine-grained sarcasm cues between images and text, thereby improving the overall performance of multimodal sarcasm detection. The main contributions of this work are summarized as follows:

• We investigate the impact of redundant visual objects in multimodal sarcasm detection, revealing that irrelevant visual cues can disrupt cross-modal interaction and hinder the capture of critical sarcasm signals.

• We propose a global-to-local multimodal framework that combines global semantic inconsistency modeling with a local gated cross-modal attention mechanism to refine representations and suppress redundant visual information.

• We introduce a multi-branch optimization strategy to jointly enhance unimodal robustness and cross-modal interaction, leading to consistent improvements on benchmark datasets.

## 2 Related Work

## 2.1 Multimodal Sarcasm Detection

Traditional sarcasm detection primarily focuses on identifying sarcastic expressions by analyzing the semantic relationships among words or sentences within textual contexts [8, 40, 47]. However, with the rapid proliferation of social media platforms, user-generated content increasingly combines multiple modalities such as images and text. This trend has driven growing interest in multimodal sarcasm detection, which aims to understand sarcastic intent by jointly modeling information from diferent modalities.

Early studies laid the foundation for this task. [34] first formally defined the problem of multimodal sarcasm detection, while [4] introduced a public dataset containing multimodal sarcastic content and proposed a hierarchical fusion framework for sarcasm recognition. Building upon these works, subsequent studies explored various cross-modal interaction mechanisms. For instance, [21, 22, 32] employed graph neural networks to model semantic relationships across modalities from multiple perspectives, enabling models to capture both intra-modal and cross-modal inconsistencies. Meanwhile, [33] constructed multiple multimodal views, allowing the model to capture sarcasm-related cues at diferent semantic granularities. More recently, researchers have also investigated the impact of data bias in multimodal sarcasm detection. For example, [13] proposed an adaptive debiasing contrastive learning strategy to mitigate statistical label bias and non-sarcastic word bias, while [49] introduced a causal graph framework to model variable relationships and reduce bias through counterfactual reasoning

In contrast to these approaches that mainly rely on complex cross-modal interaction architectures, Global-to-Local Dynamic Attention (GLDA) framework focuses on dynamically identifying sarcasm-relevant multimodal cues while suppressing redundant visual information.

## 2.2 Cross-modal Alignment

Cross-modal alignment is generally analyzed via coarse-grained and fine-grained perspectives. Coarse-grained methods often rely on global metrics like AC Score[42] or Modality Integration Rate (MIR) [12], which efectively evaluate overall pre-training but struggle to capture token-level details. For fine-grained alignment, while Logit Lens Analysis [27] identifies object-specific tokens, it lacks explicit optimization. Other works use QA-based coordination [35] or predefined vocabularies with contrastive loss (e.g., SEA [43]), though the latter is limited by fixed categories. Recent improvements [15] leverage automatic annotation from RAM [48] and eficient cosine similarity to bypass vocabulary constraints.

To achieve precise local alignment, vision grounding models like GLIP [20] and Grounding DINO [25] learn directly from region-text pairs, while RegionCLIP [15] uses pseudo-labels. However, high annotation costs remain a bottleneck. To mitigate this, self-distillation methods (e.g., CLIPSelf [38], MaskEmbed [6]) align region features with [CLS] tokens or use reconstruction tasks, though they are prone to teacher model bias and foreground interference. Lastly, audio-visual alignment utilizes attention mechanisms (e.g., TPAVI [45]) for spatiotemporal synchronization. While filtering silent regions via masks [26] reduces noise, such rigid strategies may hinder the model’s ability to distinguish between active and silent objects, limiting complex multimodal interaction modeling.

In contrast to previous cross-modal alignment methods that pri marily aim to explicitly match visual regions with textual tokens, our proposed framework focuses on dynamically selecting sarcasmrelevant multimodal cues and suppressing redundant visual objects, enabling more efective cross-modal reasoning for multimodal sar casm detection.

## 3 Methodology

## 3.1 Problem Formulation

Given an image–text pair (�,� ), where � denotes an input image and $T = \left\{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \right\}$ represents a textual sequence containing � tokens, the goal of multimodal sarcasm detection is to predict whether the pair expresses sarcasm. The corresponding label is defined as $y \in \{ 0 , 1 \}$

In real-world social media scenarios, images usually contain multiple visual objects, while texts consist of many words. However, only a subset of these objects and words are directly related to sarcasm expression. The remaining elements typically constitute redundant information that may introduce noise and mislead the model during cross-modal interaction. Such redundancy can distract the model from focusing on key sarcasm cues and weaken its capability of identifying semantic incongruity between modalities.

To address this challenge, we propose a multimodal sarcasm detection framework that progressively filters redundant information and highlights critical semantic cues. The framework consists of three major components:

• Unimodal Representation Learning

• Cross-modal Interaction Learning

• Multi-branch Optimization Criteria

The overall architecture of the proposed framework is illustrated in Figure 2.

## 3.2 Unimodal Representation Learning

Given an image–text pair (�, � ), our goal is to learn reliable modalityspecific representations before performing cross-modal reasoning. We first employ a pretrained vision–language encoder to independently extract visual and textual features.

For the textual modality, the text encoder transforms the input sequence $T = \{ w _ { 1 } , w _ { 2 } , . . . , w _ { L } \}$ into token-level representations:

$$
\mathbf { H } _ { T } = \mathrm { E n c } _ { T } ( T ) , \quad \mathbf { H } _ { T } \in \mathbb { R } ^ { L \times d } ,\tag{1}
$$

where � denotes the number of tokens and � is the embedding dimension. Similarly, the visual encoder extracts patch-level representations from the input image:

$$
\mathbf { H } _ { I } = \mathrm { E n c } _ { I } ( I ) , \quad \mathbf { H } _ { I } \in \mathbb { R } ^ { N \times d } ,\tag{2}
$$

where � represents the number of visual patches. To align the two modalities into a shared embedding space, we apply learnable

projection layers:

$$
{ \bf Z } _ { T } = { \bf W } _ { T } { \bf H } _ { T } , \qquad { \bf Z } _ { I } = { \bf W } _ { I } { \bf H } _ { I } ,\tag{3}
$$

where $\mathbf { W } _ { T }$ and $\mathbf { W } _ { I }$ denote projection matrices. Global modality representations are obtained via pooling:

$$
{ \bf z } _ { t } = \mathrm { P o o l } ( { \bf Z } _ { T } ) , \qquad { \bf z } _ { i } = \mathrm { P o o l } ( { \bf Z } _ { I } ) ,\tag{4}
$$

These representations are used to generate unimodal predictions:

$$
\hat { y } _ { t } = \mathbf { W } _ { t } \mathbf { z } _ { t } + b _ { t } , \qquad \hat { y } _ { i } = \mathbf { W } _ { i } \mathbf { z } _ { i } + b _ { i } ,\tag{5}
$$

where $\hat { y } _ { t }$ and $\hat { y } _ { i }$ denote the predicted sarcasm probabilities from the textual and visual branches, respectively.

## 3.3 Cross-modal Interaction Learning

Sarcasm often emerges from semantic incongruity between visual and textual modalities. However, both images and texts typically contain redundant information. Images may include multiple objects unrelated to sarcasm, while textual descriptions may contain tokens that do not contribute to sarcastic meaning. Such redundancy can mislead cross-modal interaction and dilute the model’s attention to relevant cues. To mitigate this, we develop a dualgranularity interaction mechanism: 1) a Global Multimodal Fusion module to capture holistic dependencies for comprehensive context understanding; 2) a Local Gated Cross-modal Attention module that acts as a precision filter. The latter assigns gating weights to individual elements, suppressing noise to distill the subtle, localized incongruities essential for detection.

3.3.1 Global Multimodal Fusion Module. To capture global crossmodal relationships while preserving modality-specific semantics, we design a global multimodal fusion module based on a Transformer encoder.

We first concatenate visual and textual representations to construct a joint multimodal sequence:

$$
\mathbf { H } = [ \mathbf { Z } _ { I } ; \mathbf { Z } _ { T } ] ,\tag{6}
$$

where ${ \mathbf Z } _ { I } ~ \in ~ \mathbb { R } ^ { N \times d }$ and ${ \bf Z } _ { T } ~ \in ~ \mathbb { R } ^ { L \times d }$ denote the projected visual and textual representations, respectively. Here � represents the number of visual patches, � denotes the number of textual tokens, and � is the embedding dimension. H $\in \mathbb { R } ^ { ( N + L ) \times d }$ therefore forms a unified multimodal sequence that integrates both visual and textual features.

A Transformer encoder is then applied to model global crossmodal interactions:

$$
\mathbf { H } _ { f } = \mathrm { T r a n s f o r m e r } ( \mathbf { H } ) ,\tag{7}
$$

where $\mathbf { H } _ { f }$ denotes the fused multimodal representations produced by the Transformer encoder after modeling interactions between visual and textual elements.

After interaction, modality-specific representations are obtained via pooling:

$$
\mathbf { h } _ { i } = \mathrm { P o o l } ( \mathbf { H } _ { f } ^ { i m g } ) , \qquad \mathbf { h } _ { t } = \mathrm { P o o l } ( \mathbf { H } _ { f } ^ { t x t } ) ,\tag{8}
$$

where $\mathbf { H } _ { f } ^ { i m g }$ and $\mathbf { H } _ { f } ^ { t x t }$ denote the visual and textual parts ofthe fused feature sequence, respectively. The pooling operation aggregates token-level representations into global modality-level vectors $\mathbf { h } _ { i }$ and $\mathbf { h } _ { t }$

![](images/587dcbf0c43a2d8f90b6a680e4b689e13e1afdc9686f3981b6f96c342d8c0366.jpg)  
Figure 2: The overall architecture of the proposed GLDA framework. In Stage 1, a CLIP encoder extracts image and text token representations $Z _ { I }$ and $Z _ { T } ,$ which are pooled into global vectors $\mathbf { z } _ { i }$ (image) and $\mathbf { z } _ { t }$ (text). In Stage $^ { 2 , }$ Path A (Global) fuses $Z _ { I }$ and $Z _ { T }$ via self-attention to produce the global representation $\mathbf { F } ,$ while Path B (Local) applies bidirectional cross-attention with a gating mechanism to produce the gated representation $F _ { g } .$ . In Stage 3, three linear classifiers are applied: a shared classifier for F and $F _ { g ; }$ , and two modality-specific classifiers for $\mathbf { z } _ { i }$ and $\mathbf { z } _ { t }$ , yielding predictions �ˆ , �ˆ , �ˆ , and $\hat { y } _ { \mathbf { t } }$ . These predictions are aggregated in Stage 4 for final sarcasm prediction.

Since diferent modalities may contribute unequally to sarcasm detection, we introduce a dynamic weighting mechanism:

$$
\left( { { w } _ { t } } , { { w } _ { i } } \right) = \mathrm { S o f t m a x } \left( { { \mathbf { W } } _ { a } } { { \mathbf { h } } _ { t } } , { { \mathbf { W } } _ { a } } { { \mathbf { h } } _ { i } } \right) ,\tag{9}
$$

where $\mathbf { W } _ { a }$ is a learnable parameter matrix used to estimate the importance of each modality. The Softmax function normalizes the modality scores into importance weights $w _ { t }$ and $w _ { i } ,$ , representing the relative contributions of textual and visual features.

The final fused representation is computed as:

$$
\mathbf { F } = w _ { t } \mathbf { h } _ { t } + w _ { i } \mathbf { h } _ { i } ,\tag{10}
$$

where F denotes the final multimodal representation that dynamically integrates textual and visual information and is used for subsequent sarcasm classification.

3.3.2 Local Gated Cross-modal Atention. While global fusion captures high-level multimodal interactions, it may inadvertently incorporate noise from irrelevant objects or tokens.

To address this issue and enhance the selection of relevant infor mation, we introduce a local gated cross-modal attention mechanism to refine the information flow at a finer granularity. By focusing on local-level correlations, this gated module acts as a selective filter to suppress redundant features, as shown in Figure 3. Specifically, we first perform bidirectional cross-modal attention:

$$
\begin{array} { r } { { \bf H } _ { I } ^ { \prime } = \mathrm { A t t n } ( { \bf Z } _ { I } , { \bf Z } _ { T } , { \bf Z } _ { T } ) , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { { \bf H } _ { T } ^ { \prime } = \mathrm { A t t n } ( { \bf Z } _ { T } , { \bf Z } _ { I } , { \bf Z } _ { I } ) , } \end{array}\tag{12}
$$

where $\mathbf { Z } _ { I }$ and ${ \bf Z } _ { T }$ denote the projected visual and textual representations introduced in the previous module. $\mathbf { H } _ { I } ^ { \prime }$ and ${ \bf { H } } _ { T } ^ { \prime }$ represent the updated visual and textual features after cross-modal attention. The attention function is defined as:

$$
{ \mathrm { A t t n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = { \mathrm { S o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } ,
$$

where Q, $\mathbf { K } ,$ and V denote the query, key, and value matrices, respectively, and � represents the feature embedding dimension used for scaling.

![](images/84446f16db6140194aabe2e409a99394b878ab2480856c3ea12b20289b5ce89f.jpg)  
Figure 3: Local gated cross-modal attention schematic diagram.

To control the amount of cross-modal information integrated into the original representations, a gating mechanism is introduced:

$$
\mathbf { G } = \sigma \left( \mathbf { W } _ { g } [ \mathbf { H } _ { T } ^ { \prime } ; \mathbf { Z } _ { T } ] \right) ,\tag{13}
$$

where $\mathbf { W } _ { g }$ denotes a learnable parameter matrix and $\sigma ( \cdot )$ represents the sigmoid activation function. The gate vector G determines how much cross-modal information should be incorporated into the original textual representation.

The updated textual representation is computed as:

$$
\tilde { \mathbf { H } } _ { T } = \mathbf { G } \odot \mathbf { H } _ { T } ^ { \prime } + \left( 1 - \mathbf { G } \right) \odot \mathbf { Z } _ { T } ,\tag{14}
$$

where ⊙ denotes element-wise multiplication. $\tilde { \mathbf { H } } _ { T }$ represents the refined textual feature that adaptively integrates the original representation and the cross-modal interaction.

A similar operation is applied to obtain the refined visual representation $\tilde { \mathbf { H } } _ { I } .$ . After pooling these gated features, we obtain modalitylevel representations $\mathbf { h } _ { t } ^ { g }$ and $\mathbf { h } _ { i } ^ { g }$

The gated multimodal representation is then computed as:

$$
\mathbf { F } _ { g } = w _ { t } \mathbf { h } _ { t } ^ { g } + w _ { i } \mathbf { h } _ { i } ^ { g } ,\tag{15}
$$

where $w _ { t }$ and $w _ { i }$ are the modality importance weights introduced in the global fusion module. $\mathrm { \bf F } _ { g }$ denotes the final gated multimodal representation used for subsequent sarcasm prediction.

## 3.4 Multi-branch Optimization Criteria

To efectively train the proposed framework, we adopt a multibranch optimization criteria that primarily optimizes cross-modal interaction representations while introducing auxiliary supervision for unimodal branches. This design is consistent with the architecture of our model, which consists of a global multimodal fusion module, a local gated cross-modal attention module, and unimodal prediction branches. By jointly supervising these components, the model is encouraged to capture the semantic inconsistency between image and text—one of the key characteristics of sarcasm—while maintaining strong modality-specific representations.

3.4.1 Global Fusion Loss. The primary cross-modal supervision is defined on the fused multimodal representation produced by the global fusion module:

$$
\begin{array} { r } { L _ { \mathrm { f u s e } } = \mathrm { C E } ( \hat { y } _ { f } , y ) , } \end{array}\tag{16}
$$

where CE(·) denotes the cross-entropy loss, $\hat { y } _ { f }$ represents the prediction generated from the fused multimodal representation, and � is the ground-truth sarcasm label. This loss encourages the model to learn global cross-modal representations that capture interactions between visual and textual modalities.

3.4.2 Local GatedInteraction Loss. To further enhance fine-grained cross-modal reasoning, we impose an additional supervision signal on the local gated cross-modal attention branch:

$$
\begin{array} { r } { L _ { \mathrm { g a t e d } } = \mathrm { C E } ( \hat { y } _ { g } , y ) , } \end{array}\tag{17}
$$

This objective guides the gating mechanism to filter redundant visual objects and irrelevant textual tokens, allowing the model to focus on sarcasm-related semantic cues at a finer granularity.

3.4.3 Unimodal AuxiliaryLosses. Although sarcasm detection mainly relies on cross-modal interaction, individual modalities may still contain useful semantic signals. Therefore, we introduce auxiliary supervision for the unimodal prediction branches:

$$
L _ { \mathrm { t e x t } } = \mathrm { C E } ( \hat { y } _ { t } , y ) , \qquad L _ { \mathrm { i m a g e } } = \mathrm { C E } ( \hat { y } _ { i } , y ) ,\tag{18}
$$

where $\hat { y } _ { t }$ and $\hat { y } _ { i }$ denote predictions derived from the textual and visual representations, respectively. These auxiliary objectives en courage each modality to learn informative features independently and help stabilize multimodal training.

Finally, the overall training objective is defined as the weighted combination of the above losses:

$$
L = L _ { \mathrm { f u s e } } + L _ { \mathrm { g a t e d } } + \lambda _ { 1 } L _ { \mathrm { t e x t } } + \lambda _ { 2 } L _ { \mathrm { i m a g e } } .\tag{19}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ control the contribution of unimodal supervision during training. By jointly optimizing these objectives, the model learns robust unimodal representations while efectively capturing both global multimodal interactions and fine-grained cross-modal relationships for sarcasm detection.

## 4 Experimental Setup

## 4.1 Dataset

For multimodal sarcasm detection, we conduct experiments on MMSD and MMSD 2.0 [33]. MMSD [4] consists of English imagetext pairs from Twitter, where labels are automatically assigned based on sarcasm-related hashtags. MMSD 2.0 [33] improves upon

MMSD by removing misleading cues and re-annotating noisy samples, providing a more reliable benchmark. Both datasets are split into training, validation, and test sets with an 80%/10%/10% ratio.

For multimodal fake news detection, we evaluate on the Weibo [16] and Twitter [3] datasets. The Weibo dataset contains 13,272 Chinese image-text pairs labeled as rumor or non-rumor, while the Twitter dataset includes 15,991 image-text pairs collected from realworld events and labeled as fake or real. Both datasets follow the same 80%/10%/10% data split. The basic statistics of these datasets are summarized in Tables 1.

Table 1: Statistics of datasets for multi-modal sarcasm detection and rumor detection.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Datasets</td><td rowspan="2">Total</td><td colspan="2">Train</td><td colspan="2">Val</td><td colspan="2">Test</td></tr><tr><td>#Positive</td><td>#Negative</td><td>#Positive</td><td>#Negative</td><td>#Positive</td><td>#Negative</td></tr><tr><td rowspan="2">Sarcasm Detection</td><td>MMSD[4]</td><td>19816</td><td>8642</td><td>11174</td><td>959</td><td>1451</td><td>959</td><td>1450</td></tr><tr><td>MMSD2.0[33]</td><td>19816</td><td>9576</td><td>10240</td><td>1042</td><td>1368</td><td>1037</td><td>1372</td></tr><tr><td rowspan="2">Rumor Detection</td><td>Weibo[16]</td><td>13272</td><td>6263</td><td>4290</td><td>864</td><td>528</td><td>827</td><td>500</td></tr><tr><td>Twitter[3]</td><td>15991</td><td>7518</td><td>6187</td><td>608</td><td>621</td><td>660</td><td>397</td></tr></table>

## 4.2 Experimental Settings

GLDA is implemented based on CLIP (ViT-B/32) as the pre-trained vision–language backbone. The visual encoder extracts �=50 patch tokens, while the text encoder processes sequences of up to �=77 tokens. Both modalities are projected into a shared embedding space of dimension �=512 for cross-modal interaction. The model is trained end-to-end using the multi-branch objective in Eq. (19).

We use AdamW as the optimizer with a batch size of 128 and train for 10 epochs. The learning rate is set to $5 \times 1 0 ^ { - 7 }$ for the CLIP backbone and $1 \times 1 0 ^ { - 4 }$ for the cross-modal modules, with a warmup proportion of 0.1 and weight decay of 0.01. The maximum gradient norm is clipped to 5, and a dropout rate of 0.3 is applied. The hidden size is set to 512 with 8 attention heads, and the number of gated cross-modal layers is 3. The unimodal loss weights are $\lambda _ { 1 } { = } 0 . 8$ and $\lambda _ { 2 } { = } 0 . 7$ . All hyperparameters are tuned on the validation set. Experiments are conducted on a single NVIDIA A6000 48G GPU.

## 4.3 Comparison Methods

4.3.1 Single-modal methods. SMSD [40] is a self-matching network using low-rank bilinear pooling. BERT [9], LLaMA3.1-8B [19], and Mixtral [14] provide bidirectional, general-purpose, and sparse MoEbased textual representations, respectively. For visual modeling, ResNet [11] and ViT [10] extract features via pooling layers and CLS tokens to support multimodal detection.

4.3.2 Multi-modal methods. HFM [4] uses attribute-initialized Bi-LSTM for feature fusion. D&R Net [41] captures cross-modal contrastive relationships. InCrossMGs [21] and CMGCN [22] employ GCNs to learn modal interactions, with the latter using attributeobject pairs. Multi-View CLIP [33] integrates multi-granularity cues, while DIP [36] applies contrastive reweighting for factual-emotional analysis. MoBa [39] optimizes parameter eficiency. ESAM [44] decouples features with sentimental constraints. ITFNet [46] models tone-polarization via tension fields, and LLaVA-v1.5 [23] combines CLIP and Vicuna for instruction-based reasoning.

## 5 Experimental Results and Analysis

## 5.1 Performance Comparison

The efectiveness of our method is validated through extensive comparisons with strong baselines (Table 2). First, GLDA consistently outperforms the strongest unimodal baselines, highlighting the necessity of multimodal modeling for sarcasm detection. This is largely due to the intrinsic contradiction between literal expression and intended meaning in sarcasm, which is dificult to capture from a single modality but can be efectively modeled through cross-modal cues.

Second, large language models (e.g., LLaMA3.1-8B and Mistral-7B) perform poorly in zero-shot settings, indicating limited capa bility in understanding sarcasm without task-specific fine-tuning. Notably, the multimodal model LLaVA-v1.5-7B underperforms even text-only LLMs in zero-shot scenarios, revealing the challenge of directly applying general-purpose multimodal models to this task. Only after supervised fine-tuning does LLaVA achieve competitive performance, even surpassing strong proprietary models such as GPT-4V.

Table 2: Comparison of diferent methods on MMSD [4] and MMSD2.0 [33] dataset, where LoRA denotes the model finetuned with LoRA-based supervised fine-tuning (SFT).
<table><tr><td rowspan="2">Method</td><td colspan="4">MMSD</td><td colspan="4">MMSD2.0</td></tr><tr><td>Acc.(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td><td>Acc.(%)</td><td>P(%)</td><td>R(%)</td><td>F1(%)</td></tr><tr><td colspan="9">Image-Only Methods</td></tr><tr><td>ResNet[11]</td><td>64.76</td><td>54.41</td><td>70.80</td><td>61.53</td><td>65.50</td><td>61.17</td><td>54.39</td><td>57.58</td></tr><tr><td>ViT[10]</td><td>67.83</td><td>57.93</td><td>70.07</td><td>63.40</td><td>72.02</td><td>65.26</td><td>74.83</td><td>69.72</td></tr><tr><td colspan="9">Text-Only Methods</td></tr><tr><td>SMSD[40]</td><td>80.90</td><td>76.46</td><td>75.18</td><td>75.82</td><td>73.56</td><td>68.45</td><td>71.55</td><td>69.97</td></tr><tr><td>BERT[9]</td><td>83.60</td><td>78.50</td><td>82.51</td><td>80.45</td><td>76.52</td><td>74.48</td><td>73.09</td><td>73.91</td></tr><tr><td>LLaMA3.1-8B (Zero-Shot) [19]</td><td>60.58</td><td>58.48</td><td>59.97</td><td>59.21</td><td>58.52</td><td>56.15</td><td>57.46</td><td>56.79</td></tr><tr><td>Mistral-7B (Zero-Shot) [14]</td><td>60.15</td><td>57.99</td><td>58.42</td><td>58.20</td><td>57.81</td><td>55.86</td><td>56.31</td><td>56.03</td></tr><tr><td colspan="9">Multi-Modal Methods</td></tr><tr><td>HFM[4]</td><td>83.44</td><td>76.57</td><td>84.15</td><td>80.18</td><td>70.57</td><td>64.84</td><td>69.05</td><td>66.88</td></tr><tr><td>D&amp;R Net[41]</td><td>84.02</td><td>77.97</td><td>83.42</td><td>80.60</td><td>=</td><td></td><td>=</td><td></td></tr><tr><td>InCrossMGs[21]</td><td>86.10</td><td>81.38</td><td>84.36</td><td>82.84</td><td>=</td><td></td><td>一</td><td></td></tr><tr><td>CMGCN[22]</td><td>86.54</td><td>1</td><td></td><td>82.73</td><td>79.83</td><td>75.82</td><td>78.01</td><td>76.90</td></tr><tr><td>Multi-View CLIP[33]</td><td>88.33</td><td>82.66</td><td>88.65</td><td>85.55</td><td>85.64</td><td>80.33</td><td>88.24</td><td>84.10</td></tr><tr><td>DIP[36]</td><td>89.59</td><td>87.76</td><td>86.58</td><td>87.17</td><td>85.32</td><td>81.76</td><td>85.94</td><td>83.79</td></tr><tr><td>MoBA[39]</td><td>88.96</td><td>82.84</td><td>88.12</td><td>85.40</td><td>85.83</td><td>80.42</td><td>88.67</td><td>84.34</td></tr><tr><td>ESAM[44]</td><td>90.11</td><td>86.87</td><td>89.54</td><td>88.19</td><td>85.87</td><td>83.12</td><td>86.05</td><td>84.56</td></tr><tr><td>ITFNet[46]</td><td>92.04</td><td>90.21</td><td>90.30</td><td>90.25</td><td>86.73</td><td>81.08</td><td>88.03</td><td>84.41</td></tr><tr><td>GPT-4V (Zero-Shot)[2]</td><td>71.04</td><td>59.43</td><td>96.52</td><td>73.56</td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-v1.5-7B (Zero-Shot)[23]</td><td></td><td>=</td><td></td><td>=</td><td>47.92</td><td>44.16</td><td>78.64</td><td>56.53</td></tr><tr><td>LLaVA-v1.5-7B (LoRA)[23]</td><td>93.67</td><td>93.70</td><td>93.14</td><td>93.40</td><td>85.18</td><td>85.59</td><td>85.20</td><td>85.11</td></tr><tr><td>GLDA</td><td>92.31</td><td>87.83</td><td>94.18</td><td>90.91</td><td>86.45</td><td>81.61</td><td>89.64</td><td>85.44</td></tr></table>

Finally, compared with classical multimodal approaches, GLDA achieves consistent improvements across datasets, validating its efectiveness. Although slightly inferior to SFT-enhanced LLaVA on certain metrics, GLDA demonstrates clear advantages in eficiency, requiring substantially less training time and approximately half the GPU memory (Table 3). Overall, it achieves a favorable balance between performance and computational cost.

## 5.2 Ablation Studies

To evaluate the contribution of each component in GLDA, we conduct ablation studies from both branch-level and interaction-level perspectives (Table 4).

Table 3: Training time and GPU memory usage of diferent models.
<table><tr><td>Model</td><td>Training time(Hour)</td><td>GPU memory(GB)</td></tr><tr><td>LLaVA-v1.5-7B (LoRA)</td><td>12.3</td><td>39.2</td></tr><tr><td>GLDA</td><td>0.6</td><td>19.8</td></tr></table>

5.2.1 Branch-based ablation. Removing any branch degrades performance, confirming all components are beneficial. In particular, removing the global cross-modal branch causes the most significant drop, indicating that high-level semantic inconsistency modeling is the dominant factor in sarcasm detection. Meanwhile, removing the local gated branch also leads to noticeable degradation, demonstrating its role in refining representations by filtering redundant information. These results suggest that the global and local branches are complementary, with the former capturing coarse cross-modal alignment and the latter enabling fine-grained refinement.

5.2.2 Interaction-based ablation. We further analyze the role of cross-modal interaction by comparing models with and without interaction. Results show that models without cross-modal interaction perform significantly worse, highlighting the necessity of modeling cross-modal inconsistency. Moreover, combining global and local interactions yields further gains, validating the efectiveness of multi-level cross-modal interaction.

Table 4: The ablation results of modules in the model. $\mathcal { B } _ { t s }$ B<sub>�</sub>, B<sub>�</sub> and $\mathcal { B } _ { g }$ denote the text-only modality learning branch, the image-only modality learning branch, the global crossmodal learning branch, and the local cross-modal learning branch, respectively.
<table><tr><td rowspan="2">Bt Bi Bl Bg</td><td colspan="2">MMSD</td><td colspan="2">MMSD2.0</td></tr><tr><td>Acc.(%)</td><td>F1(%)</td><td>Acc.(%)</td><td>F1(%)</td></tr><tr><td colspan="5">Category 1: Branch-based Ablation</td></tr><tr><td rowspan="2">V L √ √</td><td>90.15</td><td>89.23</td><td>85.06</td><td>84.75</td></tr><tr><td>90.58</td><td>89.67</td><td>85.44</td><td>84.96</td></tr><tr><td rowspan="2">L</td><td>89.42</td><td>88.15</td><td>84.68</td><td>83.23</td></tr><tr><td>90.41</td><td>89.12</td><td>85.31</td><td>84.39</td></tr><tr><td colspan="5">Category 2: Modality-based Ablation</td></tr><tr><td rowspan="2">V V √</td><td>87.86 L</td><td>86.95</td><td>82.02</td><td>81.70</td></tr><tr><td>89.98</td><td>89.01</td><td>84.92</td><td>84.26</td></tr><tr><td colspan="5">Category 3: Full Model</td></tr><tr><td>√ V</td><td>V V</td><td>92.31</td><td>90.91</td><td>86.45 85.44</td></tr></table>

## 5.3 Hyperparameter Analysis

To evaluate the sensitivity of GLDA to key hyperparameters, we conduct all analyses on MMSD2.0, whose reduced noise provides a more reliable setting. Considering the inherent class imbalance, we adopt F1 score as the primary metric.

We first analyze the unimodal loss weights $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in the multibranch objective (Figure 4). Both exhibit a bell-shaped trend, with performance peaking at $\lambda _ { 1 } = 0 . 8$ and $\lambda _ { 2 } = 0 . 7$ . This indicates that moderate unimodal supervision improves representation learning, while excessive weighting disrupts the balance between unimodal learning and cross-modal fusion. Notably, the optimal visual weight is lower than the textual one, reflecting that visual inputs often contain redundant objects. A relatively smaller $\lambda _ { 2 }$ prevents overfitting to noisy visual signals, while the gated mechanism further mitigates sensitivity to visual supervision.

![](images/29a0539550d0f10df631738e6c5d2e7b7053c20a54586db333da146c9b20b062.jpg)  
Figure 4: Parameter sensitivity analysis of $\lambda _ { 1 }$ and $\lambda _ { 2 } .$

![](images/ebfa9d9cdeffebbde8645e9717f1b70fd38491a6adbfc512dfbbb430749e7152.jpg)  
Figure 5: Parameter sensitivity analysis of gated crossattention layers.

We further vary the number of gated cross-attention layers, as summarized in Figure 5. Performance increases with depth and peaks at three layers, then degrades with additional layers, forming an inverted-U pattern. This suggests a trade-of in local cross-modal interaction: insuficient depth limits fine-grained alignment, while excessive depth leads to over-smoothing, weakening discrimina tive cross-modal contrasts. These findings highlight that efective sarcasm detection requires controlled, rather than excessive, crossmodal interaction.

## 5.4 Generalization Ability Analysis

Whether a method can be efectively generalized to other tasks has always been a highly concerned issue. To explore this, we extend evaluation to multimodal rumor detection, a related task that also relies on identifying cross-modal inconsistency. While sarcasm involves sentiment incongruity between text and image, rumor detection often exhibits mismatches between visual evidence and textual claims. This shared property provides a natural testbed for evaluating the generality of GLDA. Notably, we retrain the same architecture on Twitter and Weibo rumor detection datasets and compare against several strong baselines. No structural changes are made, and only dataset-specific hyperparameters are tuned. The

![](images/274e403a7fb2a7af32abac10beaf269f6468f4cd90aa060a2e803408a5bc30dd.jpg)  
Figure 6: Cross-modal attention visualization for a sarcastic sample from MMSD2.0. Left: Grad-CAM heatmap (Image←Text). Right: gated cross-attention weights (Text←Image).

Weibo dataset further evaluates cross-lingual generalization due to its Chinese content. Results are reported in Table 5.

GLDA achieves accuracies of 92.7% and 93.6% on Twitter and Weibo, respectively, outperforming all baselines. This strong transfer performance stems from two key factors. First, the gated crossmodal attention mechanism models token-level interactions without task-specific bias, enabling it to capture diverse forms of crossmodal inconsistency while filtering irrelevant visual content. Second, the multi-branch objective provides complementary supervision that improves robustness across tasks. The strong performance on Weibo further suggests that GLDA is robust to language variation, as cross-modal reasoning primarily operates in aligned embedding spaces. Overall, these results indicate that modeling cross-modal incongruity provides a general and transferable foundation for multimodal understanding.

Table 5: Performance comparison on the Weibo and Twitter datasets.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Acc</td><td colspan="3">Rumor</td><td colspan="3">Non-Rumor</td></tr><tr><td>R</td><td></td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="8">Weibo Dataset</td></tr><tr><td>CAFE [5] MRML [28]</td><td>0.840 0.897</td><td>0.855 0.898</td><td>0.830</td><td>0.842</td><td>0.825</td><td>0.851</td><td>0.837</td></tr><tr><td>HMCAN [31]</td><td>0.885</td><td>0.920</td><td>0.887 0.845</td><td>0.892 0.881</td><td>0.896 0.856</td><td>0.905</td><td>0.901</td></tr><tr><td>MFIR [37]</td><td>0.925</td><td>0.901</td><td></td><td></td><td></td><td>0.926</td><td>0.890</td></tr><tr><td>MIGCL [7]</td><td>0.932</td><td>0.925</td><td>0.840 0.899</td><td>0.869</td><td>0.906</td><td>0.848</td><td>0.876</td></tr><tr><td>GLDA (Ours)</td><td>0.936</td><td>0.936</td><td>0.961</td><td>0.912 0.949</td><td>0.936 0.935</td><td>0.954 0.893</td><td>0.944 0.913</td></tr><tr><td colspan="8">Twitter Dataset</td></tr><tr><td>CAFE [5] MRML [28]</td><td>0.806 0.803</td><td>0.807</td><td>0.799</td><td>0.803</td><td>0.805</td><td>0.813</td><td>0.809</td></tr><tr><td>HMCAN [31]</td><td>0.897</td><td>0.821 0.971</td><td>0.844</td><td>0.832 0.878</td><td>0.896 0.853</td><td>0.777 0.979</td><td>0.747</td></tr><tr><td>MFIR [37]</td><td>0.858</td><td></td><td>0.801</td><td>0.849</td><td>0.867</td><td></td><td>0.912</td></tr><tr><td>MIGCL [7]</td><td>0.906</td><td>0.850</td><td>0.848</td><td></td><td></td><td>0.871</td><td>0.869</td></tr><tr><td>GLDA (Ours)</td><td></td><td>0.886</td><td>0.932</td><td>0.908</td><td>0.929</td><td>0.881</td><td>0.904</td></tr><tr><td></td><td>0.927</td><td>0.935</td><td>0.915</td><td>0.925</td><td>0.920</td><td>0.939</td><td>0.929</td></tr></table>

## 5.5 Case Study

We investigate three research questions: (RQ1) How does GLDA identify sarcasm-relevant cues at both the image and token levels? (RQ2) Does cross-modal inconsistency modeling generalize to other tasks? (RQ3) What are the current limitations of the framework?

![](images/e57302a0f566271badb0ca5d6e61df723031d4ed3f6e40d889538f79f3a632b0.jpg)  
Figure 7: Cross-modal attention visualization for a rumor sample from the Twitter dataset. Left: Grad-CAM heatmap (Image←Text). Right: gated cross-attention weights (Text←Image).

5.5.1 RQ1: Sarcasm Cue Identification. As shown in Figure 6, the text “I love attitudes” is paired with an image of a person sitting on a desk in a classroom, using a phone instead of attending the lecture. In the Grad-CAM heatmap, high-activation regions are concentrated on the person and the phone, while background elements such as walls and distant desks are suppressed, indicating that the model attends selectively to content that contradicts the text. In the cross-attention weights, the token “attitudes” receives the highest weight, as the gating mechanism identifies it as the key token whose situated meaning—determined by the depicted behavior—directly contradicts the positive framing of “love.” These results confirm that GLDA filters irrelevant visual regions at the image level and pinpoints inconsistency-anchoring words at the token level.

5.5.2 RQ2: Cross-task Generalization. To evaluate the transferability of the learned cross-modal interaction, we apply the same model to a rumor detection sample from the Twitter dataset without any architectural modification. As shown in Figure 7, the text “Little Syrian girl sells chewing gum on the street so she can feed herself” is paired with an image of a young girl wearing a clean purple fleece jacket and a winter hat, smiling at the camera. The Grad-CAM heatmap highlights the clean, well-maintained clothing rather than the face or background, focusing on visual evidence that contradicts the hardship narrative. Meanwhile, the cross-attention weights are distributed across multiple tokens, reflecting that the inconsistency originates from the overall narrative rather than from a single keyword. This pattern mirrors the mechanism observed in RQ1—detecting semantic discrepancies between textual claims and visual evidence—and demonstrates the ability of the architecture to transfer across tasks without modification.

5.5.3 RQ3: Limitations. We identify two representative failure patterns from misclassified samples, as illustrated in Figure 8. In Figure 8(a), the sarcasm relies on a widely recognized meme face whose meaning derives from cultural convention rather than low-level visual attributes, which the visual encoder is unable to capture. In Figure 8(b), recognizing the sarcasm requires interpreting that 103°F denotes extreme heat—a character-level numerical reasoning step that lies beyond the scope of the current vision-language alignment. These cases reveal a clear boundary: GLDA efectively captures inconsistencies grounded in visual appearance, but struggles when comprehension requires external cultural knowledge or symbolic numerical reasoning.

![](images/ea88f826b5123ccb5c5d6a91528457d83e76f69eab2ffe0e123d953ca78e9897.jpg)  
(a) When my best friend tells me he got full marks in the exam!

![](images/74c07878228ee1147b45f22f85510177e946e0a203eb211822f1bf9638356d63.jpg)  
(b) It was nice enjoying this cooler weather for a change today.  
Figure 8: Two representative failure cases from MMSD2.0. (a) A meme-based sarcastic sample. (b) A weather-based sarcastic sample.

## 6 Conclusion and Future Work

This paper revisits multimodal sarcasm detection from the perspective ofinformation redundancy and selective cross-modal reasoning. We show that irrelevant visual content fundamentally hinders efective interaction, and argue that what to filter is as important as how to fuse. To this end, we propose GLDA, a global-to-local dynamic attention framework that captures coarse-grained semantic inconsistency and progressively refines representations via gated finegrained filtering, enabling the model to focus on sarcasm-relevant cues while suppressing noise. Extensive experiments demonstrate that GLDA achieves strong performance with improved eficiency. Ablation and transfer studies consistently highlight a key insight: modeling cross-modal inconsistency under controlled information flow is crucial for robust sarcasm understanding. Moreover, its efectiveness on multimodal rumor detection suggests that such modeling generalizes beyond task-specific patterns.

Despite these advances, challenges remain in handling sarcasm requiring external knowledge (e.g., cultural symbols) and finegrained symbolic reasoning (e.g., numerical inconsistencies). Future work will explore knowledge-enhanced reasoning and improved fine-grained perception to address these limitations, moving toward more generalizable multimodal understanding frameworks.

## References

[1] Gavin Abercrombie and Dirk Hovy. 2016. Putting sarcasm detection into context: The efects of class imbalance and manual labelling on supervised machine classification of twitter conversations. In Proceedings ofthe ACL 2016 student research workshop. 107–113.

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal

Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[3] Christina Boididou, Symeon Papadopoulos, Duc Tien Dang Nguyen, Gianluigi Boato, Michael Riegler, Andreas Petlund, and Ioannis Kompatsiaris. 2016. Verifying multimedia use at MediaEval 2016. In Proceedings of the MediaEval 2016 Workshop.

[4] Yitao Cai, Huiyu Cai, and Xiaojun Wan. 2019. Multi-modal sarcasm detection in twitter with hierarchical fusion model. In Proceedings of the 57th annual meeting ofthe association for computational linguistics. 2506–2515.

[5] Yixuan Chen, Dongsheng Li, Peng Zhang, Jie Sui, Qin Lv, Lu Tun, and Li Shang. 2022. Cross-modal ambiguity learning for multimodal fake news detection. In Proceedings ofthe ACM Web Conference 2022. 2897–2905.

[6] Ian Covert, Tony Sun, James Zou, and Tatsunori Hashimoto. 2024. Locality alignment improves vision-language models. arXiv preprint arXiv:2410.11087 (2024).

[7] Wei Cui and Mingsheng Shang. 2025. Migcl: Fake news detection with multimodal interaction and graph contrastive learning networks. Applied Intelligence 55, 1 (2025), 78.

[8] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. arXiv preprint arXiv:2010.00904 (2020).

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[10] Alexey Dosovitskiy. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 770–778.

[12] Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Weiming Zhang, and Nenghai Yu. 2025. Deciphering Cross-Modal Alignment in Large Vision-Language Models via Modality Integration Rate. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 218–227.

[13] Mengzhao Jia, Can Xie, and Liqiang Jing. 2024. Debiasing Multimodal Sarcasm Detection with Contrastive Learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 38. 18354–18362.

[14] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).

[15] Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, and Zhihui Zhu. 2025. Analyzing fine-grained alignment and enhancing vision understanding in multimodal language models. arXiv preprint arXiv:2505.17316 (2025).

[16] Zhiwei Jin, Juan Cao, Han Guo, Yongdong Zhang, and Jiebo Luo. 2017. Multi modal fusion with recurrent neural networks for rumor detection on microblogs. In Proceedings ofthe 25th ACM international conference on Multimedia. 795–816.

[17] Aditya Joshi, Vinita Sharma, and Pushpak Bhattacharyya. 2015. Harnessing context incongruity for sarcasm detection. In Proceedings ofthe 53rd Annual Meeting ofthe Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 757– 762.

[18] Raquel Justo, Thomas Corcoran, Stephanie M Lukin, Marilyn Walker, and M Inés Torres. 2014. Extracting relevant knowledge for the detection of sarcasm and nastiness in the social web. Knowledge-Based Systems 69 (2014), 124–133.

[19] Paul Kassianik, Baturay Saglam, Alexander Chen, Blaine Nelson, Anu Vellore, Massimo Aufiero, Fraser Burch, Dhruv Kedia, Avi Zohary, Sajana Weeraward hena, et al. 2025. Llama-3.1-foundationai-securityllm-base-8b technical report. arXiv preprint arXiv:2504.21039 (2025).

[20] Yunheng Li, ZhongYu Li, Quansheng Zeng, Qibin Hou, and Ming-Ming Cheng. 2024. Cascade-clip: Cascaded vision-language embeddings alignment for zeroshot semantic segmentation. arXiv preprint arXiv:2406.00670 (2024).

[21] Bin Liang, Chenwei Lou, Xiang Li, Lin Gui, Min Yang, and Ruifeng Xu. 2021. Multi modal sarcasm detection with interactive in-modal and cross-modal graphs. In Proceedings of the 29th ACM international conference on multimedia. 4707–4715.

[22] Bin Liang, Chenwei Lou, Xiang Li, Min Yang, Lin Gui, Yulan He, Wenjie Pei, and Ruifeng Xu. 2022. Multi-modal sarcasm detection via cross-modal graph convolutional network. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Vol. 1. 1767–1777.

[23] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26296–26306.

[24] Hui Liu, Wenya Wang, and Haoliang Li. 2022. Towards Multi-Modal Sarcasm Detection via Hierarchical Congruity Modeling with Knowledge Enhancement. arXiv preprint arXiv:2210.03501 (2022).

[25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision. Springer, 38–55.

[26] Juncheng Ma, Peiwen Sun, Yaoting Wang, and Di Hu. 2024. Stepping stones: A progressive training strategy for audio-visual semantic segmentation. In European Conference on Computer Vision. Springer, 311–327.

[27] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. 2024. Towards interpreting visual information processing in vision-language models. arXiv preprint arXiv:2410.07149 (2024).

[28] Liwen Peng, Songlei Jian, Dongsheng Li, and Siqi Shen. 2023. Mrml: Multimoda rumor detection by deep metric learning. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 1–5.

[29] Shraman Pramanick, Aniket Roy, and Vishal M Patel. 2022. Multimodal learning using optimal transport for sarcasm and humor detection. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision. 3930–3940.

[30] Tomáš Ptáček, Ivan Habernal, and Jun Hong. 2014. Sarcasm detection on czech and english twitter. In Proceedings ofCOLING 2014, the 25th international conference on computational linguistics: Technical papers. 213–223.

[31] Shengsheng Qian,Jinguang Wang,Jun Hu, Quan Fang, and Changsheng Xu. 2021. Hierarchical multi-modal contextual attention network for fake news detection. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 153–162.

[32] Yang Qiao, Liqiang Jing, Xuemeng Song, Xiaolin Chen, Lei Zhu, and Liqiang Nie. 2023. Mutual-enhanced incongruity learning network for multi-modal sarcasm detection. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 37. 9507–9515.

[33] Libo Qin, Shijue Huang, Qiguang Chen, Chenran Cai, Yudi Zhang, Bin Liang, Wanxiang Che, and Ruifeng Xu. 2023. MMSD2.0: Towards a Reliable Multimodal Sarcasm Detection System. In Findings ofthe Associationfor Computational Linguistics: ACL 2023. 10834–10845.

[34] Rossano Schifanella, Paloma De Juan, Joel Tetreault, and Liangliang Cao. 2016. Detecting sarcasm in multimodal social platforms. In Proceedings ofthe 24th ACM international conference on Multimedia. 1136–1145.

[35] Wei Wang, Zhaowei Li, Qi Xu, Linfeng Li, YiQing Cai, Botian Jiang, Hang Song, Xingcan Hu, Pengyu Wang, and Li Xiao. 2025. Advancing fine-grained visual understanding with multi-scale alignment in multi-modal models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 14271–14290.

[36] Changsong Wen, Guoli Jia, and Jufeng Yang. 2023. Dip: Dual incongruity perceiving network for sarcasm detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2540–2550.

[37] L Wu, Y Long, C Gao, Z Wang, and Y Zhang. 2023. Mfir: Multimodal fusion and inconsistency reasoning for explainable fake news detection. Information Fusion 100 (2023), 101944.

[38] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, and Chen Change Loy. 2023. Clipself: Vision transformer distills itself for open vocabulary dense prediction. arXiv preprint arXiv:2310.01403 (2023).

[39] Yifeng Xie, Zhihong Zhu, Xin Chen, Zhanpeng Chen, and Zhiqi Huang. 2024. MoBA: Mixture of Bi-directional Adapter for Multi-modal Sarcasm Detection. In Proceedings ofthe 32nd ACM International Conference on Multimedia. 4264–4272.

[40] Tao Xiong, Peiran Zhang, Hongbo Zhu, and Yihui Yang. 2019. Sarcasm detection with self-matching networks and low-rank bilinear pooling. In Proceedings of the World Wide Web Conference. 2115–2124.

[41] Nan Xu, Zhixiong Zeng, and Wenji Mao. 2020. Reasoning with multimodal sarcastic tweets via modeling cross-modality contrast and semantic association. In Proceedings ofthe 58th annual meeting ofthe association for computational linguistics. 3777–3786.

[42] Shijia Yang, Bohan Zhai, Quanzeng You, Jianbo Yuan, Hongxia Yang, and Chenfeng Xu. 2024. Law of vision representation in mllms. arXiv preprint arXiv:2408.16357 (2024).

[43] Yuanyang Yin, Yaqi Zhao, Yajie Zhang, Yuanxing Zhang, Ke Lin, Jiahao Wang, Xin Tao, Pengfei Wan, Wentao Zhang, and Feng Zhao. 2025. Sea: Supervised embedding alignment for token-level visual-textual integration in mllms. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. 1058–1070.

[44] Shaozu Yuan, Yiwei Wei, Hengyang Zhou, Qinfu Xu, Meng Chen, and Xiaodong He. 2025. Enhancing Semantic Awareness by Sentimental Constraint with Automatic Outlier Masking for Multimodal Sarcasm Detection. IEEE Transactions on Multimedia (2025).

[45] Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang. 2022. Not all tokens are equal: Human-centric visual analysis via token clustering transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11101–11111.

[46] Jiecheng Zhang, CL Philip Chen, Shuzhen Li, and Tong Zhang. 2025. Incongruityaware Tension Field Network for Multi-modal Sarcasm Detection. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 14499–14508.

[47] Meishan Zhang, Yue Zhang, and Guohong Fu. 2016. Tweet sarcasm detection using deep neural network. In Proceedings of the 26th International Conference on Computational Linguistics. 2449–2460.

[48] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. 2024. Recognize anything: A strong image tagging model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1724–1732.

[49] Zhihong Zhu, Xianwei Zhuang, Yunyan Zhang, Derong Xu, Guimin Hu, Xian Wu, and Yefeng Zheng. 2024. TFCD: Towards Multi-modal Sarcasm Detection via Training-Free Counterfactual Debiasing. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence. 6687–6695