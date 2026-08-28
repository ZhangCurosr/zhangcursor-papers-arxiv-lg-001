# Mitigating Strong-Modality Collapse in Multimodal Learning via Inverted Asymmetric Fusion

Mary Ogbuka Kenneth<sup>1</sup>, Foaad Khosmood<sup>2</sup>, Abbas Edalat<sup>1</sup>

<sup>1</sup> Algorithmic Human Development group, Department of Computing Imperial College London, United Kingdom <sup>2</sup> Computer Engineering Department, California Polytechnic State University San Luis Obispo, United States Correspondence: m.kenneth22@imperial.ac.uk

## Abstract

Fusing multiple modalities is expected to im prove model performance. However, on the MultiHuSE dataset, early, late, and symmetric attention fusion often fail to outperform the best unimodal baseline (text). Pathway isolation of a symmetric attention fusion model reveals that the text-pathway accuracy drops from 74.9% to 56.4% after fusion in one such setting, indicating that the dominant modality can be degraded during integration. We term this strong-modality collapse and argue that it helps explain why some multimodal mod els fail to surpass unimodal baselines. We propose Inverted Asymmetric Fusion (IAF), which avoids forcing mutual attention across modalities. The dominant modality is preserved by passing through fusion unchanged, while weaker modalities attend to it as a contextual anchor. Before fusion, weaker modalities are strengthened using Modality-Aware Knowledge Distillation. We evaluate IAF on three benchmarks with different modality hierarchies: text-dominant datasets (MultiHuSE, UR-FUNNY) and an audio-visual-dominant dataset (MUStARD). Pathway isolation shows that IAF preserves the dominant modality’s internal accuracy at its unimodal ceiling across all tested configurations, whereas symmetric fusion degrades it by up to 18.5% on MultiHuSE. IAF improves over the strongest unimodal baseline by up to 8.25%.

## 1 Introduction

Early, late, and symmetric attention fusion form the standard toolkit for integrating text, audio, and video, resting on the intuition that combining heterogeneous signals should enable models to outperform any single modality alone (Baltrusaitis et al., 2019; Li and Tang, 2025). Yet on MultiHuSE (Kenneth et al., 2025b), we observe that these approaches often fail to surpass a text-only baseline and can even degrade performance in some settings. This pattern is not isolated: prior work in affective computing and action recognition reports similar findings, where multimodal models trained with conventional fusion strategies underperform their strongest unimodal component despite having access to additional modalities (Peng et al., 2022; Wang et al., 2020b).

![](images/211055fec689e2c495f6e1ef4bdfbb0934ea4a7263a74016d28ed62a7337d7b3.jpg)  
Figure 1: Text-pathway accuracy on MultiHuSE (E5 encoder). Symmetric fusion degrades the dominant modality by 18.5 pp; IAF preserves it (Table 5).

Prior work attributes such failures to optimisation dynamics: modalities overfit and generalise at different rates, causing training to gravitate toward whichever modality converges fastest (Chaudhuri et al., 2025; Huang et al., 2022; Wang et al., 2020b). We extend this view by identifying an additional architectural mechanism that may contribute to the problem. Pathway isolation analysis of a symmetric attention model on MultiHuSE shows that the dominant text pathway drops from 74.9% to 56.4% after fusion (Figure 1).

This decline suggests that symmetric crossattention forces the dominant modality to attend to weaker, noisier signals, degrading the representation that initially carried the most predictive information. We term this phenomenon strong-modality collapse: a failure mode in which fusion undermines the model’s strongest modality.

This diagnosis suggests an architectural alternative. If symmetric attention contributes to this degradation, the solution lies in restructuring the fusion pathway rather than solely reweighting gradients or losses (Wei et al., 2025; Fan et al., 2023; Javaloy et al., 2022), which adjust training dynamics while leaving the cross-attention topology intact. To this end, we propose Inverted Asymmetric Fusion (IAF), which imposes an explicit structural hierarchy. In IAF, the dominant modality bypasses cross-modal attention entirely to preserve its representation, while weaker modalities attend to it as a contextual anchor rather than as standalone classifiers. Additionally, a modality-aware knowledge distillation stage (Hinton et al., 2015) precedes fusion, strengthening weaker modality representations so they can contribute a more informative complementary signal to the dominant pathway.

Testing across inverted modality hierarchies is essential to establish that IAF’s structural approach generalises beyond any single dominance configuration. We therefore evaluate on MultiHuSE (Kenneth et al., 2025b) and UR-FUNNY (Kamrul Hasan et al., 2019), where text is dominant, and MUStARD (Castro et al., 2019), where acoustic and visual cues carry more predictive signal. Our contributions are as follows:

• We identify strong-modality collapse as an architectural failure mode, distinct from optimisation-based methods that reweight gradients while preserving symmetric crossattention, and provide pathway-level evidence in symmetric fusion.

• We propose IAF, a fusion framework that prevents strong-modality collapse by design and preserves the dominant modality’s pathway accuracy at its unimodal ceiling across all tested configurations.

• We introduce a Modality-Aware Knowledge Distillation pipeline that adapts distillation to each dataset’s modality hierarchy, strengthening weaker encoders prior to fusion.

• We demonstrate cross-hierarchy robustness across 3 benchmarks with differing dominance structures, improving on prior baselines on UR-FUNNY and MUStARD.

## 2 Related Work

## 2.1 Multimodal Fusion Architectures

Multimodal fusion has evolved from simple concatenation and prediction averaging to learned attention mechanisms (Baltrusaitis et al., 2019; Pu Liang et al., 2021). Tensor-based approaches explicitly model cross-modal interactions via outer products (Zadeh et al., 2017), with later work reducing computational cost through low-rank factorisation (Liu et al., 2018). The advent of attentionbased fusion, led by the Multimodal Transformer (MulT) (Tsai et al., 2020) established symmetric cross-attention as the dominant paradigm. Subsequent improvements incorporated pretrained language models (Rahman et al., 2020), bottleneck token constraints (Nagrani et al., 2021), and modality subspace disentanglement (Hazarika et al., 2020).

While these designs capture rich cross-modal dynamics, they apply attention uniformly, regardless of the relative modality strengths (Feng et al., 2024). Forcing a dominant modality to symmetrically attend to weaker, noisier signals can corrupt its internal representation. Recent work has begun to address this by using text to guide multimodal fusion in affective tasks (Zhang et al., 2025), though such designs typically retain mutual attention rather than fully shielding the dominant stream from cross-modal interference. This vulnerability motivates our inverted asymmetric approach, which preserves the dominant modality while using weaker streams as contextual anchors.

## 2.2 Modality Collapse and Pathway Analysis

Representational degradation in fusion is one instance of a broader failure mode: modality collapse. Empirical studies show multimodal models often exploit only a subset of their input modalities (Gong et al., 2025; Kenneth et al., 2024b; Zhang et al., 2024). Because different modalities overfit and generalise at uneven rates, training tends to favour the fastest-converging modality (Wang et al., 2020b; Wu et al., 2022). Theoretical analyses attribute this behaviour to conflicting gradient directions during joint optimisation (Huang et al., 2022; Chaudhuri et al., 2025).

To mitigate modality imbalance, prior work has proposed optimisation-based strategies such as on-the-fly gradient modulation (OGM-GE) (Wei et al., 2025), Pareto-based integration (Wei and Hu, 2024), prototype-guided clustering (PMR) (Fan et al., 2023), and impartial multitask optimisation (Javaloy et al., 2022). These methods address imbalance by adjusting gradients or training objectives, but generally retain the same fusion topology, leaving dominant representations vulnerable to cross-modal interference. In contrast, our work takes a complementary architectural approach. IAF restructures the fusion pathway to limit such interference by design. However, structural shielding alone is insufficient unless weaker encoders are first strengthened, motivating the pre-fusion distillation stage.

## 2.3 Knowledge Distillation in Multimodal Learning

Knowledge distillation (KD) transfers generalisation by aligning a student’s predictions with a high-capacity teacher (Hinton et al., 2015; Gou et al., 2021). Cross-modal distillation extends this across domain boundaries and has proven effective for depth, optical flow, action recognition, and emotion analysis (Gupta et al., 2016; Thoker and Gall, 2019; Albanie et al., 2018). In multimodal fusion, KD is commonly used within the fusion stage for compression, dropout robustness, missingmodality learning, or representational alignment (Fang et al., 2021; Wang et al., 2023, 2020a; Wei et al., 2023; Li et al., 2023b; Lin and Hu, 2024). By contrast, our Modality-Aware Knowledge Distillation (MAKD) is a pre-fusion stage that uses only the empirically dominant unimodal modality as teacher and is designed specifically to strengthen weaker encoders before asymmetric integration. A detailed design comparison is provided in $\mathsf { A p - }$ pendix H.

## 2.4 Humor and Sarcasm Detection

Humor and sarcasm benchmarks provide a useful testbed for modality imbalance because their predictive hierarchies differ substantially across datasets. UR-FUNNY (Kamrul Hasan et al., 2019) and MultiHuSE (Kenneth et al., 2025b, 2024a, 2025a) are text-dominant, whereas MUStARD (Castro et al., 2019) places greater weight on acoustic and visual cues. This contrast makes the domain well suited for evaluating whether a fusion architecture can preserve strong modalities without sacrificing cross-modal contribution.

## 3 Methodology

Our framework comprises two sequential stages (Figure 2). Stage 1 applies Modality-Aware Knowledge Distillation to strengthen weaker modalities prior to fusion, adapting the strategy to each dataset’s modality hierarchy. Stage 2 performs Inverted Asymmetric Fusion (IAF), integrating streams within a strict hierarchy that shields the dominant modality from cross-modal interference and prevents strong-modality collapse.

## 3.1 Problem Formulation

Let $\mathbf { f } ^ { ( t ) } \in \mathbb { R } ^ { d _ { t } } , \mathbf { f } ^ { ( a ) } \in \mathbb { R } ^ { d _ { a } }$ , and $\mathbf { f } ^ { ( v ) } \in \mathbb { R } ^ { d _ { v } }$ denote pre-extracted feature vectors for text, audio, and video modalities of a single utterance, respectively. Each is projected to a shared $d { = } 5 1 2$ space via a modality-specific encoder $\phi _ { m }$ , yielding $\mathbf { h } ^ { ( m ) }$ $\phi _ { m } ( \mathbf { f } ^ { ( m ) } ) \bar { \in \mathbb { R } } ^ { 5 1 2 }$ . The dominant modality $m ^ { * }$ is defined as $m ^ { * } =$ arg $\operatorname* { m a x } _ { m }$ $\mathrm { A c c u r a c y } ( \phi _ { m } )$ on the validation set.

## 3.2 Stage 1: Modality-Aware Knowledge Distillation

The dominant modality $m ^ { * }$ acts as a teacher to elevate weaker modalities m $\neq m ^ { * }$ before fusion.

## 3.2.1 Distillation Loss $( \mathcal { L } _ { \mathbf { K D } } )$

Each student S is trained against a frozen teacher T (logits ${ \mathbf z } ^ { T } \in \mathbb { R } ^ { C }$ , bottleneck features $\mathbf { g } ^ { T } \in \mathbb { R } ^ { 5 1 2 } )$ using:

$$
{ \mathcal { L } } _ { \mathrm { K D } } = \alpha { \mathcal { L } } _ { \mathrm { C E } } + \beta { \mathcal { L } } _ { \mathrm { s o f t } } + \gamma { \mathcal { L } } _ { \mathrm { f e a t } }\tag{1}
$$

The loss terms are: hard-label cross-entropy $\mathcal { L } _ { \mathrm { C E } }$ temperature-scaled KL divergence ${ \mathcal { L } } _ { \mathrm { s o f t } }$ (with $T =$ 3.5), and bottleneck feature alignment ${ \mathcal { L } } _ { \mathrm { f e a t } }$ :

$$
\mathcal { L } _ { \mathrm { C E } } = - \sum _ { c = 1 } ^ { C } \mathbf { 1 } [ y = c ] \log \left( \operatorname { s o f } \mathrm { t m a x } ( \mathbf { z } ^ { S } ) _ { c } \right)\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { s o f t } } = T ^ { 2 } \mathrm { K L } \bigg ( \sigma \bigg ( \frac { \mathbf { z } ^ { S } } { T } \bigg ) \bigg \| \sigma \bigg ( \frac { \mathbf { z } ^ { T } } { T } \bigg ) \bigg )\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { f e a t } } = \left\| \mathbf { g } ^ { S } - \mathbf { g } ^ { T } \right\| _ { 2 } ^ { 2 }\tag{4}
$$

where $\sigma$ denotes the softmax function. We use fixed coefficients $\alpha = 0 . 4 , \beta = 0 . 3 5$ , and $\gamma =$ 0.25, selected by manual tuning on MultiHuSE and held constant across all datasets and encoder combinations. A sensitivity sweep of coefficients over four configurations shows that fusion accuracy varies by at most 1.20 percentage points, with both the default and equal-weight settings achieving the best result; see Appendix G for details.

Weighted Dual-Teacher Distillation (MUStARD). When both audio and video dominate (MUStARD), students are supervised by weighted dual teachers. The student is jointly supervised by a primary teacher $\mathcal { T } _ { 1 }$ and a secondary teacher $\tau _ { 2 } .$ whose losses are combined via a scalar weight $w _ { 1 } \colon$

$$
\mathcal { L } _ { \mathrm { d u a l } } = w _ { 1 } \mathcal { L } _ { \mathrm { K D } } ( \mathcal { T } _ { 1 } ) + \left( 1 - w _ { 1 } \right) \mathcal { L } _ { \mathrm { K D } } ( \mathcal { T } _ { 2 } )\tag{5}
$$

where $w _ { 1 }$ is tuned per configuration $( w _ { 1 } = 0 . 8$ for audio-primary, $w _ { 1 } = 0 . 2$ for video-primary).

![](images/0794df7832cea86ca60b1dbe341cd771831c373829b19274f5c9258109121c64.jpg)  
Figure 2: Overview of the proposed two-stage framework. Stage 1 performs Modality-Aware Knowledge Distillation, and Stage 2 applies the Inverted Asymmetric Fusion.

Cross-Architecture Distillation (UR-FUNNY). On UR-FUNNY, the narrow unimodal performance gap limits the representational surplus available for cross-modal transfer. This motivated a same-modality, cross-architecture distillation: high-capacity static teachers (E5, Dasheng, PE-Core) distill into BiLSTM students on benchmark features (GloVe, COVAREP, OpenFace), isolating IAF’s architectural contribution. The same threeterm loss in Eq. (1) applies, with teacher features $\mathbf { g } ^ { T }$ projected to 512-dim prior to alignment.

With all modality encoders operating at an enhanced representation, Stage 2 integrates these streams using IAF.

## 3.3 Stage 2: Inverted Asymmetric Fusion

## 3.3.1 CrossModal Anchor (CMA)

The CrossModal Anchor module serves as the core attention primitive in our architecture. Given a query-modality vector h $\in \mathbb { R } ^ { 5 1 2 }$ and a context set $\bar { \mathbf { C } } \in \mathbb { R } ^ { K \times 5 1 2 }$ containing K other modalities, the module computes:

$$
\hat { \mathbf { h } } = \mathbf { M H A } ( \mathbf { h } , \mathbf { C } , \mathbf { C } )\tag{6}
$$

$$
\mathbf { h } ^ { \prime } = \mathrm { L a y e r N o r m } \Big ( \mathbf { h } + \mathrm { D r o p o u t } ( \hat { \mathbf { h } } ) \Big )\tag{7}
$$

where MHA(·) denotes multi-head attention (Vaswani et al., 2017) with $d _ { \mathrm { m o d e l } } { = } 5 1 2$ and $H { = } 8$ heads. The residual connection Eq. (7) preserves the query modality’s original representation.

## 3.3.2 Structural Hierarchy: PURE vs. ATTEND

Let $m ^ { * }$ denote the dominant modality (or modalities). IAF enforces the following constraint on the

forward pass:

$$
\mathbf { o } ^ { ( m ) } = \left\{ \begin{array} { l l } { \mathbf { h } ^ { ( m ) } } & { \mathrm { i f } m = m ^ { * } \mathrm { ( P U R E ) } } \\ { \mathrm { C M A } \Big ( \mathbf { h } ^ { ( m ) } , \mathbf { \Omega } [ \mathbf { h } ^ { ( j ) } ] _ { j \neq m } \Big ) } & { \mathrm { i f } m \neq m ^ { * } \mathrm { ( A T T E N D ) } } \end{array} \right.\tag{8}
$$

The PURE pathway remains unmodified. AT-TEND pathways query all other modalities (including the shielded dominant) as contextual filters. On MultiHuSE/UR-FUNNY, text is PURE; on MUStARD, audio/video.

## 3.3.3 Adaptive Gating and Final Classification

Each modality produces pathway logits via a dedicated classifier head $\psi _ { m } .$ . For the PURE modality, $\psi _ { m ^ { * } }$ is a frozen copy of its standalone classifier, preserving its decision boundary. For ATTEND modalities, $\psi _ { m }$ is a trainable linear head suited to their contextually-enriched representations.

A gating network G computes sample-adaptive weights from pre-attention features:

$$
\mathbf { w } = \mathrm { s o f t m a x } \Big ( \mathcal { G } \Big ( [ \mathbf { h } ^ { ( t ) } \| \mathbf { h } ^ { ( a ) } \| \mathbf { h } ^ { ( v ) } ] \Big ) \Big )\tag{9}
$$

where $\mathbf { w } \in \mathbb { R } ^ { 3 }$ denotes the modality-gate weights, and $\mathcal { G }$ is a two-layer MLP with batch normalisation and dropout. The final prediction is the weighted sum of per-pathway logits:

$$
\hat { \mathbf { y } } = \sum _ { m \in \{ t , a , v \} } w _ { m } \cdot \psi _ { m } ( \mathbf { o } ^ { ( m ) } )\tag{10}
$$

Gating on pre-attention features preserves the textscpure/ATTEND separation.

## 3.4 Training Regularisation

We apply two complementary regularisation strategies during fusion training.

## 3.4.1 Mixup

For each mini-batch, we sample a mixing coefficient $\lambda \sim \mathrm { B e t a } ( \alpha _ { \mathrm { m i x } } , \alpha _ { \mathrm { m i x } } )$ with $\alpha _ { \mathrm { m i x } } { = } 0 . 0 5$ and interpolate pairs of feature vectors and their labels:

$$
\tilde { \mathbf { h } } _ { i } ^ { ( m ) } = \lambda \mathbf { h } _ { i } ^ { ( m ) } + ( 1 - \lambda ) \mathbf { h } _ { \sigma ( i ) } ^ { ( m ) } , \quad \tilde { y } _ { i } = ( \lambda , y _ { i } , y _ { \sigma ( i ) } )\tag{11}
$$

where $\sigma$ is a random permutation of batch indices. Mixup discourages overconfident predictions, which is critical for preventing overfitting (ablation in Appendix C)

## 3.4.2 Curriculum Modality Dropout

To prevent over-reliance on any single modality, we apply a two-phase stochastic masking schedule: all modalities are present during warm-up (epochs <15), after which each sample is assigned one of seven masking configurations (trimodal $p = 0 . 4 0$ , each bimodal pair $p = 0 . 1 0$ , each unimodal $p = 0 . 1 0 )$ , with masked modalities zeroed before the forward pass.

## 3.5 Pathway Analysis

To verify dominant pathway preservation, we isolate each modality’s internal contribution by routing $\mathbf { h } ^ { ( m ) }$ directly through $\psi _ { m }$ , bypassing gating. For the PURE modality $m ^ { * } { \mathrm { . } }$ :

$$
\mathrm { A c c u r a c y } _ { \mathrm { p a t h } } ^ { ( m ^ { * } ) } = \mathrm { A c c u r a c y } _ { \mathrm { u n i } } ^ { ( m ^ { * } ) }\tag{12}
$$

That is, $m ^ { * }$ pathway accuracy equals its standalone unimodal performance. Any deviation from this equality indicates cross-modal interference. Symmetric fusion violates this equality; IAF enforces it by construction. For ATTEND modalities, pathway accuracy is expected to drop below their unimodal baselines, as their representations are no longer optimised for standalone classification but instead tuned to provide contextual signal.

## 4 Experiments

We describe the datasets, feature extraction procedures, baseline models, and implementation details.

## 4.1 Datasets

We evaluate on three multimodal benchmarks for humor and sarcasm detection, chosen to span contrasting modality hierarchies and task complexity (binary and multiclass classification).

MultiHuSE (Kenneth et al., 2025b) is a 5-class humor style classification dataset comprising 2,407 video utterances from 50 diverse actors, including 1,463 unique text scripts and 943 re-performances.

Standard random splits risk data leakage across performances of the same script, so we introduce a strict 5-fold cross-validation protocol grouped by unique text ID, ensuring all performances of a given script appear exclusively in either training or evaluation. All baseline models are re-implemented and re-evaluated under this protocol. Modality hierarchy: Text ≫ Audio ≫ Video.

UR-FUNNY (Kamrul Hasan et al., 2019) is a binary humor detection dataset of 16,514 context– punchline pairs drawn from TED talks, using the standard train/dev/test split $( N _ { \mathrm { t e s t } } { = } 9 9 4 )$ . Following the original benchmark, we use sequential features GloVe (text), COVAREP (audio), and OpenFace (video) throughout, isolating our architectural contribution from feature quality and enabling direct comparison with prior work using the same feature set. Modality hierarchy: Text > Audio ≈ Video.

MUStARD (Castro et al., 2019) is a binary sarcasm detection dataset of 690 balanced utterances drawn from TV sitcoms, using 5-fold crossvalidation $( N _ { \mathrm { t e s t } } { = } 1 3 8$ per fold) from the original paper. Modality hierarchy: Audio ≈ Video > Text, is the clearest inversion relative to MultiHuSE and motivates our cross-hierarchy evaluation.

## 4.2 Feature Extraction

All features are extracted offline using frozen pretrained encoders and serialised to disk prior to training. This design decouples representation quality from the fusion architecture, ensuring that observed performance differences reflect architectural choices rather than effects of encoder fine-tuning.

MultiHuSE and MUStARD. For text, we evaluate two encoders: BERT-based-uncased (Devlin et al., 2019) (768-d) and Multilingual-E5-largeinstruct (Wang et al., 2024) (1024-d). Using both allows us to verify that our findings are not specific to a single text backbone. For audio, we use Dasheng-0.6B (Dinkel et al., 2024) (1280-d), a large-scale general-purpose audio encoder. For video, we evaluate two encoders: VideoMAE (Tong et al., 2022) (768-d) and PE-Core (Bolya et al., 2025) (1024-d), providing coverage of both masked autoencoder and contrastive video representations.

UR-FUNNY. We use GloVe (Pennington et al., 2014) (text), COVAREP (Degottex et al., 2014) (audio), and OpenFace (Baltrusaitis et al., 2018) (video), consistent with the original benchmark. The role of modern encoders as distillation teachers within this established feature setting is described in Section 3.2.

## 4.3 Implementation Details

Infrastructure. Feature extraction is performed on an HPC cluster (NVIDIA A16, 15GB VRAM), while all subsequent training is conducted on a consumer laptop (NVIDIA GeForce RTX 3050, 4GB VRAM). All primary results use a fixed random seed of 999. For MultiHuSE and MUStARD, results are reported as mean ± SD over 5-fold crossvalidation; for UR-FUNNY, we additionally report a five-seed analysis using seeds 42, 100, 123, 999, and 600. The unimodal pre-evaluation used to identify the dominant modality adds minimal overhead: on our hardware, training a single unimodal encoder takes less than 15 minutes, compared with a few hours for full fusion training.

Model Parameters. Pre-trained encoders used for feature extraction have the following parameter counts: BERT-base (110M), E5-multilingual (560M), Dasheng (600M), VideoMAE-base (94M), and PE-Core (320M visual encoder + 310M language tower). These encoders are frozen throughout; no encoder parameters are updated during distillation or fusion training.

Distillation and Fusion. We used AdamW for distillation, CosineAnnealingLR for fusion, a batch size of 64, Mixup with $\alpha _ { \mathrm { m i x } } = 0 . 0 5$ , a curriculum dropout warm-up of 15 epochs, and a CrossModal Anchor with $d _ { \mathrm { m o d e l } } = 5 1 2$ and H = 8 heads. Full hyperparameters are provided in Appendix A.

Evaluation. We report accuracy and macroaveraged F1. For MultiHuSE and MUStARD, results are reported as mean ± SD over 5 folds. For UR-FUNNY, we use the standard train/dev/test split; the result at seed 999 is reported in Table 3 for direct comparability with prior published baselines, and a multi-seed analysis is reported in Section 5.

## 4.4 Baselines

We compare against three internal baselines trained on identical features under identical hyperparameters: (1) Early Fusion: feature concatenation followed by an MLP; (2) Late Fusion: weighted averaging of unimodal prediction vectors; (3) Symmetric Fusion: mutual cross-attention across all three modalities using the same distilled features as IAF, isolating the inverted asymmetric design.

Comparisons with prior published work appear in the results tables in Section 5.

## 5 Results and Analysis

Table 1 reports unimodal performance. Distillation consistently improves student models (gains of 0.88–6.23%, mean 2.44%), with the largest gain on UR-FUNNY’s COVAREP audio (+6.23%), highlighting the benefit of cross-architecture distillation. These distilled students, together with their teachers, serve as the inputs to IAF and symmetry fusion.

<table><tr><td colspan="5">Modality / Feature Base Acc(%) Base F1(%) Dist. Acc(%) Dist. F1(%)</td></tr><tr><td colspan="5">MultiHuSE — 5-class humor style | Text  Audio  Video</td></tr><tr><td>Text: BERT</td><td> $6 8 . 2 6 { \pm } 2 . 2 5 $ </td><td>68</td><td>70.21 ± 2.82</td><td>70</td></tr><tr><td>Text: E5*</td><td> $\mathbf { 7 4 . 8 6 { \pm } 3 . 0 8 }$ </td><td>75</td><td></td><td>一</td></tr><tr><td>Audio: Dasheng</td><td> $6 0 . 4 5 { \pm } 1 . 5 8 $ </td><td>60</td><td> $6 2 . 9 0 \pm 1 . 8 3$ </td><td>63</td></tr><tr><td>Video: PE-Core</td><td> $4 3 . 6 6 \pm 1 . 2 9$ </td><td>43</td><td> $4 5 . 5 7 \pm 1 . 2 2$ </td><td>46</td></tr><tr><td>Video: VideoMAE</td><td> $4 0 . 1 3 { \pm } 0 . 7 3$ </td><td>40</td><td> $4 1 . 0 1 \pm 1 . 6 0$ </td><td>41</td></tr><tr><td colspan="5">UR-FUNNY — binary humor | Text &gt; Audio ≈ Video</td></tr><tr><td>Text: GloVe*</td><td>62.47</td><td>64</td><td>65.19</td><td>65</td></tr><tr><td>Audio: COVAREP</td><td>57.55</td><td>48</td><td>63.78</td><td>64</td></tr><tr><td>Video: OpenFace</td><td>57.75</td><td>58</td><td>58.95</td><td>59</td></tr><tr><td colspan="5">MUStARD — binary sarcasm | Audio ≈ Video &gt; Text</td></tr><tr><td>Text: BERT</td><td> $6 9 . 4 2 { \pm } 2 . 8 0 $ </td><td>69</td><td> $7 2 . 0 3 \pm 2 . 4 9$ </td><td>72</td></tr><tr><td>Text: E5</td><td> $7 0 . 2 9 { \scriptstyle \pm 2 . 0 0 }$ </td><td>70</td><td> $7 2 . 3 2 \pm 2 . 2 6$ </td><td>72</td></tr><tr><td>Audio: Dasheng*</td><td> $\mathbf { 7 8 . 1 2 \pm 5 . 4 5 }$ </td><td>78</td><td></td><td>一</td></tr><tr><td>Video: PE-Core</td><td> $7 7 . 1 0 { \pm } 3 . 1 3 $ </td><td>77</td><td></td><td>一</td></tr><tr><td>Video: VideoMAE</td><td> $7 4 . 4 9 { \pm } 1 . 1 6$ </td><td>74</td><td>1</td><td></td></tr></table>

Table 1: Unimodal results across all three benchmarks. Base: raw pre-trained; Dist.: knowledge-distilled. <sup>⋆</sup> = dominant modality. MultiHuSE and MUStARD: mean ± SD over 5-fold CV; UR-FUNNY: single split.

Table 2 presents trimodal fusion results on Multi-HuSE. Early and late fusion fail to reliably surpass the dominant unimodal text baseline, with early fusion degrading sharply in the E5+VideoMAE setting (−10.55%), suggesting that naive concatenation may disrupt strong text representations when combined with noisier video inputs. Symmetric fusion also fails to recover the E5 baseline in both E5 configurations (−2.49%, −0.54%), indicating that expressive attention alone is insufficient to preserve the dominant modality. IAF is the only architecture that consistently exceeds the unimodal ceiling across all four configurations, attaining gains of 2.06–6.48%. Paired Wilcoxon signed-rank tests across the five folds show that IAF significantly outperforms symmetric fusion in the PE-Core configurations (BERT: W=0, $\scriptstyle { p = 0 . 0 3 1 }$ ; E5: W=0, p=0.031; one-tailed), a result further supported by paired t-tests $\scriptstyle ( p = 0 . 0 2 3$ and p=0.005, respectively); Here, W=0 indicates IAF outperformed symmetric fusion on every fold.

<table><tr><td>Text</td><td>Video</td><td>Architecture</td><td>Acc</td><td>F1</td><td>∆</td></tr><tr><td colspan="6">Audio: Dasheng used across all configurations</td></tr><tr><td>Text: BERT (base 68.26%)</td><td></td><td>Early Fusion</td><td>67.22 ±2.00 67</td><td></td><td>-1.04 -0.21</td></tr><tr><td></td><td>BERT VideoMAE</td><td>Late Fusion Symmetric Fusion Proposed (IAF)</td><td>68.05 ±2.12  $6 9 . 4 2 \pm 1 . 4 8 $   $\mathbf { 7 3 . 1 6 \pm 2 . 5 9 }$ </td><td>68 69 72</td><td> $+ 1 . 1 6$   $+ 4 . 9 0$ </td></tr><tr><td>BERT PE-Core</td><td></td><td>Early Fusion Late Fusion Symmetric Fusion Proposed (IAF)</td><td> $7 0 . 3 8 \pm 2 . 5 7$   $6 8 . 6 3 \pm 2 . 3 6$   $7 2 . 0 0 \pm 1 . 2 0$   $\mathbf { 7 4 . 7 4 \pm 1 . 7 1 }$ </td><td>70 68 70 74</td><td> $+ 2 . 1 2$   $+ 0 . 3 7$   $+ 3 . 7 4$   $+ 6 . 4 8$ </td></tr><tr><td colspan="4">Text: E5 (base 74.86%)</td><td>64</td><td>-10.55</td></tr><tr><td>E5</td><td>VideoMAE</td><td>Early Fusion Late Fusion Symmetric Fusion Proposed (IAF)</td><td> $6 4 . 3 1 \pm 1 . 5 1$  71.25 ±4.07  $7 2 . 3 7 \pm 2 . 7 3$  76.92 ±3.53</td><td>71 72 76</td><td>-3.61 -2.49 +2.06</td></tr><tr><td>E5</td><td>PE-Core</td><td>Early Fusion Late Fusion Symmetric Fusion Proposed (IAF)</td><td> $7 6 . 2 8 \pm 2 . 9 1$   $7 1 . 5 4 \pm 4 . 2 0$   $7 4 . 3 2 \pm 3 . 0 3$   ${ \bf 7 8 . 0 6 \pm 3 . 8 2 }$ </td><td>76 71 74 78</td><td> $+ 1 . 4 2$  -3.32 -0.54 +3.20</td></tr></table>

Table 2: Trimodal fusion results on MultiHuSE. ∆ = gain over the strongest unimodal text baseline (BERT = 68.26%; E5 = 74.86%). All metrics in %.

Table 3 compares IAF against prior work and our internal baselines on UR-FUNNY. IAF achieves 70.72% accuracy, surpassing the strongest prior method (MISA, 68.60%) by 2.12% and symmetric fusion by 1.81%. Interestingly, symmetric fusion (68.91%) outperforms previously published methods, demonstrating that strengthening weaker modalities before fusion is beneficial even without architectural asymmetry. To assess initialisation sensitivity, we evaluated IAF over five random seeds (42, 100, 123, 600, 999), obtaining a mean accuracy of $6 9 . 7 0 \pm 0 . 6 1 \%$ (min 69.32%; seed 999 is reported in Table 3 for comparability with prior work). A one-sample t-test shows that this mean significantly exceeds both symmetric fusion (t(4)=2.90, p=0.022, one-tailed) and MISA (t(4)=4.03, p=0.008, one-tailed), and each seed individually outperforms both baselines.

Table 4 reports results on MUStARD. Symmetric fusion already improves substantially over the strongest unimodal audio baseline (+3.47% and +4.34% for BERT and E5, respectively), and IAF yields further gains in both text-encoder settings. In particular, IAF (E5) attains 83.33% accuracy, exceeding the state-of-the-art method (MHA with RoBERTa, 79.32%) by 4.01%. We note that the IAF-over-symmetric margin on MUStARD (0.87– 1.02 pp) falls within one standard deviation of both models and should be interpreted accordingly. Paired Wilcoxon signed-rank tests are consistent with this pattern: neither BERT+PE-Core (W=2,

<table><tr><td>Method</td><td></td><td>Context Acc (%) F1 (%)</td><td></td><td> $\Delta _ { \mathrm { u n i } }$ </td></tr><tr><td>Prior published methods</td><td></td><td></td><td></td><td></td></tr><tr><td>C-MFN (Kamrul Hasan et al., 2019)</td><td>×</td><td>64.47</td><td></td><td></td></tr><tr><td>C-MFN (Kamrul Hasan et al., 2019)</td><td>√</td><td>65.23</td><td></td><td></td></tr><tr><td>AGM(Late fusion) (Li et al., 2023a)</td><td>×</td><td>65.97</td><td></td><td></td></tr><tr><td>AGM(Early fusion) (Li et al., 2023a)</td><td>×</td><td>66.07</td><td></td><td></td></tr><tr><td>MULTIBENCH (Pu Liang et al., 2021)</td><td>√</td><td>66.70</td><td></td><td></td></tr><tr><td>BCFNet (Deng et al., 2025)</td><td>×</td><td>66.78</td><td>68</td><td></td></tr><tr><td>HF (Choube and Soleymani, 2020)</td><td>√</td><td>67.84</td><td>69</td><td></td></tr><tr><td>MISA (GloVe) (Hazarika et al., 2020)</td><td>×</td><td>68.60</td><td>一</td><td></td></tr><tr><td colspan="5">Our internal baselines — Text: GloVe (base 62.47%)</td></tr><tr><td>Early Fusion</td><td>√</td><td>65.19</td><td>66</td><td>+2.72</td></tr><tr><td>Late Fusion</td><td>√</td><td>64.19</td><td>65</td><td>+1.72</td></tr><tr><td>Symmetric Fusion</td><td>√</td><td>68.91</td><td>69</td><td>+6.44</td></tr><tr><td>Proposed (IAF)</td><td>√</td><td>70.72†</td><td>71</td><td>+8.25</td></tr></table>

Table 3: Results on UR-FUNNY. All methods use GloVe/COVAREP/OpenFace features (Kamrul Hasan et al., 2019). $\Delta _ { \mathrm { u n i } } = \mathrm { g a i n }$ over the GloVe baseline (62.47%) for our internal baselines. Context: ✓ = punchline + context; × = punchline only. <sup>†</sup>Seed 999; mean 69.70 ± 0.61% over 5 seeds.

p=0.188) nor E5+PE-Core (W=1, p=0.125) is significant (one-tailed, α=0.05).
<table><tr><td>Method</td><td>Context</td><td>Acc(%)</td><td>F1(%)</td><td>∆uni</td></tr><tr><td>Prior published methods</td><td></td><td></td><td></td><td></td></tr><tr><td>MULTIBENCH (Pu Liang et al., 2021)</td><td>√</td><td>71.8±0.3</td><td></td><td></td></tr><tr><td>SVM (BERT) (Castro et al., 2019)</td><td>√</td><td></td><td>71.5</td><td></td></tr><tr><td>SWIA (Chauhan et al., 2020)</td><td>√</td><td></td><td>72.6</td><td></td></tr><tr><td>CE (BART) (Ray et al., 2022)</td><td>√</td><td></td><td>74.2</td><td></td></tr><tr><td>MuLOT (BERT) (Pramanick et al., 2022)</td><td>×</td><td>74.52</td><td></td><td></td></tr><tr><td>IWAN (BERT) (Wu et al., 2021)</td><td>×</td><td></td><td>74.5</td><td></td></tr><tr><td>IWAN (BERT) (Wu et al., 2021)</td><td>√</td><td></td><td>75.1</td><td></td></tr><tr><td>MuLOT (BERT) (Pramanick et al., 2022)</td><td>√</td><td>76.82</td><td></td><td></td></tr><tr><td>MHA (RoBERTa) (Aggarwal et al., 2023)</td><td>√</td><td>79.32</td><td>77.6</td><td></td></tr><tr><td colspan="5">Our internal baselines – Audio: Dasheng (base 78.12%)</td></tr><tr><td>Early Fusion</td><td>√</td><td> $7 9 . 7 1 \pm 3 . 0 4$ </td><td>80.0</td><td>+1.59</td></tr><tr><td>Late Fusion (Average)</td><td>√</td><td> $7 9 . 8 6 \pm 4 . 7 1$ </td><td>80.0</td><td>+1.74</td></tr><tr><td>Symmetric Fusion (BERT)</td><td>√</td><td> $8 1 . 5 9 \pm 4 . 1 7$ </td><td>82.0</td><td>+3.47</td></tr><tr><td>Symmetric Fusion (E5)</td><td>√</td><td> $8 2 . 4 6 \pm 4 . 5 2$ </td><td>82.0</td><td>+4.34</td></tr><tr><td>Proposed (IAF (BERT))</td><td>√</td><td> $8 2 . 6 1 \pm 3 . 9 2$ </td><td>83.0</td><td>+4.49</td></tr><tr><td>Proposed (IAF (E5))</td><td>√</td><td> $\mathbf { 8 3 . 3 3 \pm 4 . 2 0 }$ </td><td>84.0</td><td>+5.21</td></tr></table>

Table 4: Results on MUStARD. $\Delta _ { \mathrm { u n i } } = \mathrm { g a i n }$ over Dasheng audio unimodal baseline (our internal baselines only).

Table 5 summarises pathway isolation analysis. Across all three datasets, IAF preserves the dominant modality’s pathway at its unimodal baseline, whereas symmetric fusion degrades the same pathways by up to 18.5pp: the E5 text pathway drops from 74.9% (Table 1) to 56.4% under symmetric fusion (Table 5). This directly supports our claim that IAF’s structural asymmetry functions as a representation-preservation guarantee that symmetric fusion cannot provide. Visual summaries of this collapse pattern are provided in Appendix E.

## 5.1 Ablation Study

(1) Role of Modalities. We assess modality contribution through bimodal ablations, reported in

<table><tr><td></td><td>Text Video</td><td>Archi.</td><td>Full Fusion</td><td>Text Path</td><td>Audio Path</td><td>Video Path</td></tr><tr><td colspan="7">MultiHuSE – Dominant Modality: TEXT – BERT=70.2%, E5=74.9%</td></tr><tr><td rowspan="3">BERT</td><td rowspan="3">VideoMAE</td><td rowspan="3">Symmetric IAF</td><td>69.42±1.48</td><td>61.3±2.5</td><td>59.0±4.5</td><td>60.1±1.3</td></tr><tr><td>73.16±2.59</td><td>70.2±2.8</td><td>55.8±7.4</td><td>57.5±5.0</td></tr><tr><td>Symmetric</td><td>72.00±1.20</td><td>63.5±1.6 63.6±2.3</td><td>61.6±7.2</td></tr><tr><td rowspan="3"></td><td rowspan="3">PE-Core VideoMAE</td><td rowspan="3">IAF Symmetric</td><td>74.74±1.71</td><td>70.2±2.8</td><td>62.9±2.5</td><td>65.6±3.1</td></tr><tr><td>72.37±2.73</td><td>56.4±4.6</td><td>63.7±9.2</td><td>67.4±3.3</td></tr><tr><td>IAF 76.92±3.53</td><td>74.9±3.1</td><td>48.4±8.9</td><td>56.7±7.0</td></tr><tr><td rowspan="3">5</td><td rowspan="3">PE-Core</td><td rowspan="3">Symmetric IAF</td><td>74.32±3.03</td><td>55.3±8.7</td><td>65.5±2.2</td><td>69.7±4.8</td></tr><tr><td>78.06±3.82</td><td>74.9±3.1</td><td>54.9±9.6</td><td>62.2±6.6</td></tr><tr><td></td><td>UR-FUNNY – Dominant Modality: TEXT=65.19%</td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>Goe</td><td>OpenFace</td><td>Symmetric IAF</td><td>68.91 70.72</td><td>63.08 65.19</td><td>49.30 50.91</td><td>50.70 49.30</td></tr><tr><td colspan="7">MUStARD– Dominant Modalities: AUDIO=78.12%; VIDEO (PE)=77.10%</td></tr><tr><td rowspan="3">BERT</td><td rowspan="3">VideoMAE PE-Core</td><td rowspan="3">Symmetric IAF</td><td>79.28±4.26</td><td>62.0±5.0</td><td>64.4±15.2</td><td>67.8±8.0</td></tr><tr><td>80.43±2.63</td><td>64.6±9.0</td><td>78.1±5.5</td><td>74.5±1.2</td></tr><tr><td>Symmetric 81.59±4.17</td><td>64.9±9.5</td><td>69.4±6.3</td><td>73.5±9.0</td></tr><tr><td rowspan="4">B</td><td rowspan="2">VideoMAE</td><td>IAF</td><td>82.61±3.92</td><td>63.9±6.2</td><td>78.1±5.5</td><td>77.1±3.1</td></tr><tr><td>Symmetric</td><td>79.86±3.09</td><td>60.3±12.3</td><td>77.3±5.1</td><td>71.5±6.2</td></tr><tr><td rowspan="2">PE-Core</td><td>IAF</td><td>79.42±4.34</td><td>66.4±7.6</td><td>78.1±5.5</td><td>74.5±1.2</td></tr><tr><td>Symmetric</td><td>82.46±4.52</td><td>74.4±3.5</td><td>74.9±4.6</td><td>73.3±10.4</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">IAF</td><td>83.33±4.20</td><td>69.0±7.8</td><td>78.1±5.5</td><td>77.1±3.1</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 5: Pathway isolation analysis. Each score reflects accuracy using only the corresponding modality pathway, bypassing fusion aggregation. Bold = dominant modality pathway. All metrics in %.

Appendix B. Across all three datasets, full trimodal IAF remains the best configuration. The ablations are consistent with the empirical modality hierarchies identified by unimodal validation: removing text hurts most on MultiHuSE and UR-FUNNY, whereas on MUStARD the text-free A+V setting remains close to full trimodal performance, reflecting the stronger audio-visual signal.

(2) Role of Distilled Feature Representations. We contrast IAF trained on raw pre-trained features with IAF trained on distillation-enhanced features, isolating the impact of Stage 1. As seen in Table 6, incorporating distilled representations improves performance: gains of 1.02–2.41% on MultiHuSE, 1.02–1.59% on MUStARD, and 1.81% on UR-FUNNY. This underscores the importance of raising weaker modality representations before fusion, rather than relying solely on the architecture.

<table><tr><td>Dataset</td><td>Config</td><td>Feat.</td><td>Acc(%) ± SD</td><td>F1(%)</td><td>∆(%)</td></tr><tr><td rowspan="3">MultiHuSE</td><td>BERT+PEC</td><td>Base Dist</td><td>72.33±2.02 74.74±1.71</td><td>72 74</td><td>-2.41</td></tr><tr><td>E5+PEC</td><td>Base</td><td>75.90±6.17</td><td>76</td><td>一 -2.16</td></tr><tr><td></td><td>Dist</td><td>78.06±3.82</td><td>78</td><td>一</td></tr><tr><td rowspan="2">MUStARD</td><td>BERT+PEC</td><td>Base Dist</td><td>81.59±3.85 82.61±3.92</td><td>82 83</td><td>-1.02 一</td></tr><tr><td>E5+PEC</td><td>Base Dist</td><td>81.74±4.24 83.33±4.20</td><td>82 84</td><td>-1.59</td></tr><tr><td>UR-FUNNY</td><td>GloVe/COV/OF</td><td>Base Dist</td><td>68.91 70.72</td><td>69 71</td><td>一 -1.81 一</td></tr></table>

Table 6: Distillation ablation: raw (Base) vs. distilled (Dist.) features within IAF. ∆ = Dist − Base.

(3) Role of Regularisation. Appendix C, Table 11, reports the contribution of Mixup and Curriculum Modality Dropout. Removing both yields the largest drops across all datasets (up to 4.07% on MultiHuSE, 1.61% on UR-FUNNY, and 2.90% on MUStARD), confirming their complementary roles. Curriculum Dropout provides a consistently larger isolated gain than Mixup, suggesting that progressive difficulty scheduling is particularly effective in noisy multimodal feature spaces.

(4) Role of the Gate Mechanism. We evaluate three gating strategies within IAF: a Learnable gate (sample-adaptive weights), a Hierarchy gate (fixed weights derived from modality ranking), and a Uniform gate (equal weights, T=A=V=0.33). Results (Appendix D, Table 12) shows that the learnable gate generally performs best, while the hierarchy gate remains competitive under strong modality imbalance. Appendix F further shows that mean gate weights remain distributed across modalities in all configurations, with no modality exceeding 0.53 on average.

## 6 Conclusion

We introduced Inverted Asymmetric Fusion (IAF), a two-stage multimodal framework for mitigating strong-modality collapse in discriminative fusion. By preserving the dominant modality through a PURE/ATTEND hierarchy and strengthening weaker modalities via knowledge distillation prior to fusion, IAF maintains the dominant pathway at its unimodal ceiling while improving fused performance in text-dominant and audio-visual-dominant settings. Across three benchmarks with different modality hierarchies, IAF consistently outperforms early, late, and symmetric fusion baselines, and surpasses prior published methods on UR-FUNNY and MUStARD. These findings suggest that protecting strong modalities from cross-modal interference is an important architectural principle for robust multimodal learning.

## Limitations

This work has several limitations. First, IAF requires the dominant modality to be identified empirically through unimodal evaluation before fusion training. When modalities have similar predictive strength, the PURE/ATTEND assignment becomes less clear and the benefits of structural asymmetry may diminish. Moreover, the inferred hierarchy depends on the encoder set: a stronger text encoder may appear dominant over a weaker video encoder even when the task itself relies more heavily on visual information. The hierarchy therefore reflects empirical encoder performance rather than intrinsic task structure. Although the dual-teacher design used for MUStARD partially addresses codominant settings, a more principled treatment remains open, for example through dynamic hierarchy assignment at the sample level.

Second, all experiments use frozen, preextracted features. This isolates the architectural contribution of IAF from encoder fine-tuning, but the framework has not been evaluated end to end, where joint optimisation could alter both the modality hierarchy and the effectiveness of distillation. In addition, our pathway analysis uses the original frozen unimodal classifier as a fixed probe, so it tests whether the dominant modality’s decision boundary is preserved rather than whether it remains recoverable in principle. Future work could examine this more directly through probing or representational analyses such as Centred Kernel Alignment (CKA).

Finally, all datasets are English-only and confined to humor and sarcasm. It therefore remains unclear whether the modality hierarchies observed here, and the benefits of IAF, generalise to other languages or to tasks such as emotion recognition, visual question answering, or action recognition. Although the inclusion of E5-multilingual suggests potential cross-lingual applicability, this has not been evaluated, and cross-dataset generalisation remains for future work.

## Ethical considerations

This work uses three publicly available datasets (MultiHuSE, UR-FUNNY, MUStARD), collected and released under standard academic licences by their respective authors. No new data collection or human annotation was conducted. All experiments operate on pre-extracted, anonymised feature representations with no access to personally identifiable information. We note that automated humor and sarcasm detection systems may misclassify culturally specific communication styles and are not intended for deployment beyond controlled research settings.

## Acknowledgments

This research was supported by the Petroleum Technology Development Fund (PTDF) of Nigeria.

## References

Sajal Aggarwal, Ananya Pandey, and Dinesh Kumar Vishwakarma. 2023. Multimodal Sarcasm Recognition by Fusing Textual, Visual and Acoustic content via Multi-Headed Attention for Video Dataset. In 2023 World Conference on Communication and Computing, WCONF 2023. Institute of Electrical and Electronics Engineers Inc.

Samuel Albanie, Arsha Nagrani, Andrea Vedaldi, and Andrew Zisserman. 2018. Emotion recognition in speech using cross-modal transfer in the wild. In MM 2018 - Proceedings of the 2018 ACM Multimedia Conference, pages 292–301. Association for Computing Machinery, Inc.

Tadas Baltrusaitis, Chaitanya Ahuja, and Louis Philippe Morency. 2019. Multimodal Machine Learning: A Survey and Taxonomy.

Tadas Baltrusaitis, Amir Zadeh, Yao Chong Lim, and Louis Philippe Morency. 2018. OpenFace 2.0: Facial behavior analysis toolkit. In Proceedings - 13th IEEE International Conference on Automatic Face and Gesture Recognition, FG 2018, pages 59–66. Institute of Electrical and Electronics Engineers Inc.

Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. 2025. Perception Encoder: The best visual embeddings are not at the output of the network. arXiv Preprint (Facebook research).

Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, and Soujanya Poria. 2019. Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper). In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4619–4629, Florence, Italy. Association for Computational Linguistics.

Abhra Chaudhuri, Anjan Dutta, Tu Bui, and Serban Georgescu. 2025. A Closer Look at Multimodal Representation Collapse. In Proceedings of the 42nd International Conference on Machine Learning, Vancouver.

Dushyant Singh Chauhan, Asif Ekbal, and Pushpak Bhattacharyya. 2020. Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4351–4360. Association for Computational Linguistics.

Akshat Choube and Mohammad Soleymani. 2020. Punchline Detection using Context-Aware Hierarchical Multimodal Fusion. In ICMI 2020 - Proceedings ofthe 2020 International Conference on Multimodal Interaction, pages 675–679. Association for Computing Machinery, Inc.

Gilles Degottex, John Kane, Thomas Drugman, Tuomo Raitio, and Stefan Scherer. 2014. COVAREP: A Collaborative Voice Analysis Repository for Speech Technologies. In IEEE International Conference on Acoustic, Speech and Signal Processing (ICASSP). IEEE.

Boya Deng, Jianzhao Li, Maoguo Gong, Zedong Tang, Yourun Zhang, Kaiyuan Feng, and Yue Wu. 2025. BCFNet: Bi-temporal collaborative fusion network for multi-modal humor detection. Pattern Recognition, 172.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova Google, and A I Language. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT, pages 4171–4186, Minnesota. Association for Computational Linguistics.

Heinrich Dinkel, Zhiyong Yan, Yongqing Wang, Junbo Zhang, Yujun Wang, and Bin Wang. 2024. Scaling up masked audio encoder learning for general audio classification. In Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, pages 547–551. International Speech Communication Association.

Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang, and Song Guo. 2023. PMR: Prototypical Modal Rebalance for Multimodal Learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20029–20038, Vancouver. IEEE.

Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lijuan Wang, Yezhou Yang, and Zicheng Liu. 2021. Compressing Visual-linguistic Model via Knowledge Distillation. In Proceedings of the IEEE International Conference on Computer Vision, pages 1408–1418. Institute of Electrical and Electronics Engineers Inc.

Xinyu Feng, Yuming Lin, Lihua He, You Li, Liang Chang, and Ya Zhou. 2024. Knowledge-Guided Dynamic Modality Attention Fusion Framework for Multimodal Sentiment Analysis. In Findings ofthe Associationfor Computational Linguistics: EMNLP, pages 14755–14766. Association for Computational Linguistics.

Baoquan Gong, Xiyuan Gao, Pengfei Zhu, Qinghua Hu, and Bing Cao. 2025. Multimodal Negative Learning. In 39th Conference on Neural Information Processing Systems. NeurIPS.

Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. 2021. Knowledge Distillation: A Survey. International Journal of Computer Vision, 129(6):1789–1819.

Saurabh Gupta, Judy Hoffman, and Jitendra Malik. 2016. Cross Modal Distillation for Supervision Transfer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2827–2836. IEEE.

Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. 2020. MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis. In MM 2020 - Proceedings of the 28th ACM International Conference on Multimedia, pages 1122–1131. Association for Computing Machinery, Inc.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the Knowledge in a Neural Network. arXiv Preprint.

Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. 2022. Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably). In Proceedings of the 39th International Conference on Machine Learning, Maryland.

Adrián Javaloy, Maryam Meghdadi, and Isabel Valera. 2022. Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization. In Proceedings of the 39th International Conference on Machine Learning,, Maryland.

Md Kamrul Hasan, Wasifur Rahman, Amir Zadeh, Jianyuan Zhong, Md Iftekhar Tanveer, Louis-Philippe Morency, and Mohammed Hoque. 2019. UR-FUNNY: A Multimodal Language Dataset for Understanding Humor. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 2046–2056.

Mary Ogbuka Kenneth, Foaad Khosmood, and Abbas Edalat. 2024a. A Two-Model Approach for Humour Style Recognition. In Proceedings ofthe 4th International Conference on Natural Language Processing for Digital Humanities, pages 259–274, Miami. Association for Computational Linguistics.

Mary Ogbuka Kenneth, Foaad Khosmood, and Abbas Edalat. 2024b. Systematic Literature Review: Computational Approaches for Humour Style Classification. ArXiv.

Mary Ogbuka Kenneth, Foaad Khosmood, and Abbas Edalat. 2025a. Explaining Humour Style Classifications: An XAI Approach to Understanding Computational Humour Analysis. Journal ofData Mining & Digital Humanities, NLP4DH.

Mary Ogbuka Kenneth, Foaad Khosmood, and Abbas Edalat. 2025b. MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions. In International Conference on Content-Based Multimedia Indexing (CBMI), pages 1–7. Institute of Electrical and Electronics Engineers (IEEE).

Hong Li, Xingyu Li, Pengbo Hu, Yinuo Lei, Chunxiao Li, and Yi Zhou. 2023a. Boosting Multi-modal Model Performance with Adaptive Gradient Modulation. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 22214–22224. Computer Vision Foundation.

Songtao Li and Hao Tang. 2025. Multimodal Alignment and Fusion: A Survey. International Journal ofComputer Vision.

Yong Li, Yuanzhi Wang, and Zhen Cui. 2023b. Decoupled Multimodal Distilling for Emotion Recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6631–6640. IEEE.

Ronghao Lin and Haifeng Hu. 2024. Multi-Task Momentum Distillation for Multimodal Sentiment Analysis. IEEE Transactions on Affective Computing, 15(2):549–565.

Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. 2018. Efficient Low-rank Multimodal Fusion with Modality-Specific Factors. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics, pages 2247– 2256. Association for Computational Linguistics.

Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun, and Google Research. 2021. Attention Bottlenecks for Multimodal Fusion. In 35th Conference on Neural Information Processing Systems. NeurIPS.

Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. 2022. Balanced Multimodal Learning via On-the-fly Gradient Modulation. In IEEE Conference on Computer Vision and Pattern Recognition, pages 8238–8247.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543. Association for Computational Linguistics.

Shraman Pramanick, Aniket Roy, and Vishal M Patel. 2022. Multimodal Learning using Optimal Transport for Sarcasm and Humor Detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3930–3940. IEEE.

Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A Lee, Yuke Zhu, Ruslan Salakhutdinov, Louis-Philippe Morency, and Johns Hopkins. 2021. MULTIBENCH: Multiscale Benchmarks for Multi modal Representation Learning. In 35th Conference on Neural Information Processing Systems, Track on Datasets and Benchmarks., pages 1–20. NeurIPS.

Wasifur Rahman, Md Kamrul Hasan, Sangwu Lee, Amir Zadeh, Chengfeng Mao, Louis-Philippe

Morency, and Ehsan Hoque. 2020. Integrating Multimodal Information in Large Pretrained Transformers. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pages 2359– 2369. Association for Computational Linguistics.

Anupama Ray, Shubham Mishra, Apoorva Nunna, and Pushpak Bhattacharyya. 2022. A Multimodal Corpus for Emotion Recognition in Sarcasm. In Proceedings ofthe 13th Conference on Language Resources and Evaluation, pages 20–25.

Fida Mohammad Thoker and Juergen Gall. 2019. Cross-Modal Knowledge Distillation for Action Recognition. In 2019 IEEE International Conference on Image Processing (ICIP). IEEE.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. 2022. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. In 36th Conference on Neural Information Processing Systems 9(NeurIPS), New Orleans.

Yao Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis Philippe Morency, and Ruslan Salakhutdinov. 2020. Multimodal transformer for unaligned multimodal language sequences. In ACL 2019 - 57th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, pages 6558–6569. Association for Computational Linguistics (ACL).

Ashish Vaswani, Google Brain, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In 31st Conference on Neural Information Processing Systems.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv preprint.

Qi Wang, Liang Zhan, Paul Thompson, and Jiayu Zhou. 2020a. Multimodal Learning with Incomplete Modalities by Knowledge Distillation. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1828– 1838. Association for Computing Machinery.

Tiannan Wang, Wangchunshu Zhou, Eth Zurich, Yan Zeng, and Xinsong Zhang. 2023. EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning. In Findings ofthe Associationfor Computational Linguistics, pages 13899–13913. Association for Computational Linguistics.

Weiyao Wang, Du Tran, and Matt Feiszli. 2020b. What Makes Training Multi-Modal Classification Networks Hard? In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 12692–12702. IEEE Computer Society.

Shicai Wei, Chunbo Luo, and Yang Luo. 2023. MMANet: Margin-aware Distillation and Modalityaware Regularization for Incomplete Multimodal Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20039–20049. IEEE.

Yake Wei and Di Hu. 2024. MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance. In Proceedings ofthe 41st International Conference on Machine Learning, Vienna.

Yake Wei, Di Hu, Henghui Du, and Ji Rong Wen. 2025. On-the-Fly Modulation for Balanced Multimodal Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(1):469–485.

Nan Wu, Stanisław Jastrz˛ Ebski, Kyunghyun Cho, and Krzysztof J Geras. 2022. Characterizing and Overcoming the Greedy Nature of Learning in Multimodal Deep Neural Networks. In Proceedings of the 39th International Conference on Machine Learning.

Yang Wu, Yanyan Zhao, Xin Lu, Bing Qin, Yin Wu, Jian Sheng, and Jinlong Li. 2021. Modeling Incongruity between Modalities for Multimodal Sarcasm Detection. IEEE Multimedia, 28(2):86–95.

Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017. Tensor Fusion Network for Multimodal Sentiment Analysis. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1103–1114, Denmark. Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics.

Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, and Huaxiu Yao. 2024. Multimodal Representation Learning by Alternating Unimodal Adaptation. In IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR). Computer Vision and Pattern Recognition Conference (CVPR).

Yu Zhang, Bin Chen, Hongfei Ye, Zijian Gao, Tianjiao Wan, Long Lan, and Kele Xu. 2025. Text-guided Multimodal Fusion for the Multimodal Emotion and Intent Joint Understanding. In ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings. Institute of Electrical and Electronics Engineers Inc.

## A Full Hyperparameter Configurations

Table 8 and Table 7 present the complete hyperparameter configurations for the distillation and fusion stages, across all datasets. Hyperparameters were selected through manual tuning: values were adjusted iteratively based on validation performance until stable, well-performing configurations were identified, after which a single final experiment was run per configuration and reported in the paper. No automated search was conducted.

<table><tr><td>Hyperparameter</td><td></td><td>MultiHuSE UR-FUNNY MUStARD</td><td></td></tr><tr><td>Optimiser</td><td>AdamW</td><td>Adam</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $6 \times 1 0 ^ { - 5 }$ </td><td>1  $. \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td>10⁻3</td><td>10⁻3</td><td>10⁻2</td></tr><tr><td>Max epochs</td><td>100</td><td>80</td><td>100</td></tr><tr><td>Early stopping patience</td><td>35</td><td>20</td><td>40</td></tr><tr><td>Gradient clipping</td><td>1.0</td><td></td><td>1.0</td></tr></table>

Table 7: Dataset-specific hyperparameters for the fusion
<table><tr><td>Hyperparameter Value</td></tr><tr><td>KD loss weights  $( \alpha , \beta , \gamma )$  0.4, 0.35, 0.25</td></tr><tr><td>Temperature T 3.5 Optimiser AdamW</td></tr><tr><td>Weight decay 10−3</td></tr><tr><td>Text/Audio LR  $1 0 ^ { - 4 } – 1 0 _ { \cdot } ^ { - 3 }$  Video LR  $9 \times 1 0 ^ { - 5 }$ </td></tr></table>

Table 8: Distillation stage hyperparameters.

## B Role of Modalities Ablation

We remove one modality at a time to quantify each modality’s contribution to full trimodal performance. Table 9 shows that full trimodal IAF outperforms all bimodal variants on every dataset. On MultiHuSE, the text-free A+V pair drops to 65.85%, confirming text’s dominant role. On UR-FUNNY, T+A outperforms T+V (68.91% vs. 65.29%), consistent with the Text > Audio ≈ Video hierarchy. On MUStARD, the text-free A+V pair (82.61%) nearly matches full trimodal performance (83.33%), reflecting the co-dominance of audio and video and the smaller marginal benefit of text.

<table><tr><td>Configuration</td><td>Pure</td><td>Attends</td><td>Acc(%) ± SD</td><td>F1(%)</td></tr><tr><td colspan="5">MultiHuSE — Text dominant (E5, Dasheng, PE-Core)</td></tr><tr><td>Full Trimodal</td><td>Text</td><td>Audio, Video</td><td>78.06 ±3.82</td><td>78</td></tr><tr><td>T+V</td><td>Text</td><td>Video</td><td>76.61 ±4.58</td><td>77</td></tr><tr><td>T+A</td><td>Text</td><td>Audio</td><td>76.57 ±2.92</td><td>76</td></tr><tr><td>A+V (Text-Free)</td><td>Audio</td><td>Video</td><td>65.85 ±1.63</td><td>65</td></tr><tr><td colspan="5">UR-FUNNY — Text dominant (GloVe, COVAREP, OpenFace)</td></tr><tr><td>Full Trimodal</td><td>Text</td><td>Audio, Video</td><td>70.72</td><td>71</td></tr><tr><td>T+A</td><td>Text</td><td>Audio</td><td>68.91</td><td>69</td></tr><tr><td>T+V</td><td>Text</td><td>Video</td><td>65.29</td><td>65</td></tr><tr><td>A+V (Text-Free)</td><td>Audio</td><td>Video</td><td>65.29</td><td>65</td></tr><tr><td colspan="5">MUStARD — Audio/Video dominant (E5, Dasheng, PE-Core)</td></tr><tr><td>Full Trimodal</td><td>Audio, Video</td><td>Text</td><td>83.33 ±4.20</td><td>84</td></tr><tr><td>A+T</td><td>Audio</td><td>Text</td><td>79.57 ±5.17</td><td>80</td></tr><tr><td>V+T</td><td>Video</td><td>Text</td><td>79.57 ±5.41</td><td>80</td></tr><tr><td>A+V (Text-Free)</td><td>Audio</td><td>Video</td><td>82.61 ±3.01</td><td>83</td></tr></table>

Table 9: Bimodal ablation of E5 and GloVe-based configuration for IAF

Table 10 reports bimodal ablation results for BERT encoder configurations on MultiHuSE and MUStARD. Results follow the same directional patterns as the E5 configurations in Table 9, confirming that the modality hierarchy findings are consistent across text encoders.

<table><tr><td>Configuration</td><td>Pure</td><td>Attends</td><td>Acc (%) ± SD F1(%)</td><td></td></tr><tr><td></td><td colspan="4">MultiHuSE — Text dominant (BERT, Dasheng, PE-Core)</td></tr><tr><td>Full Trimodal</td><td>Text</td><td>Audio, Video</td><td> $7 4 . 7 4 \pm 1 . 7 1$ </td><td>74</td></tr><tr><td>T+V</td><td>Text</td><td>Video</td><td> $7 2 . 4 5 \pm 3 . 2 7$ </td><td>72</td></tr><tr><td>T+A</td><td>Text</td><td>Audio</td><td> $7 2 . 0 8 \pm 2 . 7 1$ </td><td>72</td></tr><tr><td colspan="5">MUStARD — Audio/Video dominant (BERT, Dasheng, PE-Core)</td></tr><tr><td>Full Trimodal</td><td>Audio, Video</td><td>Text</td><td> ${ \bf 8 2 . 6 1 \pm 3 . 9 2 }$ </td><td>83</td></tr><tr><td>A+T</td><td>Audio</td><td>Text</td><td> $7 9 . 7 1 \pm 3 . 3 0$ </td><td>80</td></tr><tr><td>V+T</td><td>Video</td><td>Text</td><td> $7 7 . 2 5 \pm 4 . 7 5$ </td><td>77</td></tr></table>

Table 10: Bimodal fusion for BERT configurations on MultiHuSE and MUStARD.

## C Role of Regularisation Ablation

Table 11 reports the full regularisation ablation across all datasets and encoder configurations. BERT- and E5-based configurations are shown for MultiHuSE and MUStARD; results are directionally consistent across both encoders.

<table><tr><td>Mixup</td><td>Curriculum Acc (%) ± SD</td><td>F1 ∆(%)</td></tr><tr><td colspan="3">MultiHuSE (BERT + Dasheng + PE-Core)</td></tr><tr><td>√</td><td>√</td><td>74.74 ±1.71 74</td></tr><tr><td>x</td><td>√</td><td>73.49 ±3.15 73 -1.25</td></tr><tr><td>V</td><td>X</td><td>72.58 ±3.80 72 -2.16</td></tr><tr><td>x</td><td>x</td><td>70.75 ±2.67 70 -3.99</td></tr><tr><td colspan="3">MultiHuSE (E5 + Dasheng + PE-Core)</td></tr><tr><td>√</td><td></td><td>78.06 ±3.82 78</td></tr><tr><td>x</td><td>√ √</td><td>76.61 ±4.40 77 -1.45</td></tr><tr><td>V</td><td>X</td><td>75.74 ±5.53 76 -2.32</td></tr><tr><td>x</td><td>x</td><td>73.99 ±5.04 74 -4.07</td></tr><tr><td colspan="3">UR-FUNNY (GloVe + COVAREP + OpenFace)</td></tr><tr><td></td><td></td><td>71</td></tr><tr><td>VV</td><td>√ x</td><td>70.72 70.62 71 -0.10 -1.00</td></tr><tr><td>x</td><td>V</td><td>69.72 70</td></tr><tr><td>x</td><td>x</td><td>69.11 69</td></tr><tr><td colspan="3">MUStARD (BERT + Dasheng + PE-Core)</td></tr><tr><td>√</td><td>√</td><td>82.61 ±3.92 83</td></tr><tr><td>V</td><td>x</td><td>82.03 ±4.08 82 -0.58</td></tr><tr><td>X</td><td>V</td><td>81.45 ±4.11 81 -1.16</td></tr><tr><td>x</td><td>x</td><td>79.71 ±4.25 80 -2.90</td></tr><tr><td colspan="3">MUStARD (E5 + Dasheng + PE-Core)</td></tr><tr><td>VV</td><td>√</td><td>83.33 ±4.20 84</td></tr><tr><td></td><td>x</td><td>82.03 ±3.73 82 -1.30</td></tr><tr><td>X</td><td>V</td><td>81.59 ±3.65 82 -1.74</td></tr><tr><td>x</td><td>x</td><td>81.16 ±4.51 81 -2.17</td></tr></table>

Table 11: Regularisation ablation across MultiHuSE, UR-FUNNY, and MUStARD. ✓✓ = both Mixup and Curriculum Dropout enabled (full model); ∆ = accuracy change relative to the full model. All metrics in %.

## D Role of Gate Mechanism Ablation

Table 12 reports the effect of different gating strategies. The learnable gate attains the highest accuracy on UR-FUNNY and MUStARD, demonstrating the benefit of sample-adaptive weighting.

<table><tr><td colspan="4">Config Gate Type Acc(%) ± SD ∆(%)</td></tr><tr><td colspan="4">MultiHuSE Dataset (Hierarchy: T=0.50, A=0.30, V=0.20)</td></tr><tr><td>BERT + PE-Core</td><td>Learnable Hierarchy Uniform</td><td>74.74 ±1.71 74.99 ±2.68 74.86 ±3.31</td><td>+0.25 +0.12</td></tr><tr><td>E5 + PE-Core</td><td>Learnable Hierarchy Uniform</td><td> $7 8 . 0 6 \pm 3 . 8 2$   $7 8 . 5 2 \pm 3 . 6 4$   $7 8 . 0 2 \pm 3 . 8 1$ </td><td>+0.46 -0.04</td></tr><tr><td colspan="4">UR-FUNNY Dataset (Hierarchy: T=0.50, A=0.30, V=0.20)</td></tr><tr><td>GloVe/COVAREP/OpenFace</td><td>Learnable Hierarchy Uniform</td><td>70.72 70.32 69.42</td><td>-0.40 -1.30</td></tr><tr><td colspan="4">MUStARD Dataset (Hierarchy: T=0.25, A=0.40, V=0.35)</td></tr><tr><td rowspan="3">BERT + PE-Core</td><td>Learnable</td><td>82.61 ±3.92</td><td></td></tr><tr><td>Hierarchy</td><td>81.74 ±4.11</td><td>-0.87</td></tr><tr><td>Uniform</td><td>81.88 ±3.94</td><td>-0.73</td></tr><tr><td rowspan="3">E5 + PE-Core</td><td>Learnable</td><td> $\mathbf { 8 3 . 3 3 \pm 4 . 2 0 }$ </td><td></td></tr><tr><td>Hierarchy</td><td> $8 2 . 0 3 \pm 4 . 4 8$ </td><td>-1.30</td></tr><tr><td>Uniform</td><td> $8 1 . 8 8 \pm 4 . 4 2 $ </td><td>-1.45</td></tr></table>

Table 12: Gate mechanism ablation across all three datasets. Learnable = sample-adaptive weights; Hierarchy = fixed weights derived from modality ranking; Uniform = equal weights (T=A=V=0.33). ∆ = accuracy difference relative to the Learnable gate.

MultiHuSE is a mild exception where the fixed hierarchy gate performs marginally better, likely due to its extreme modality imbalance: under highly skewed conditions, a learnable gate can over-emphasize the dominant modality. Overall differences between gates are small (≤1.45%), indicating that IAF is largely insensitive to the precise gating strategy.

## E Pathway Collapse: Visualisation

This section visualises pathway collapse in the distilled IAF and symmetric models (E5 + PE-Core, MultiHuSE) from two complementary perspectives: aggregate pathway accuracy across configurations and feature-space behaviour of the text pathway.

Figure 3 summarises dominant-modality pathway accuracy for all configurations in Table 5, comparing the unimodal ceiling, symmetric fusion, and IAF. Across all seven settings, symmetric fusion consistently reduces pathway accuracy, with a mean drop of 11.6 pp, whereas IAF matches the unimodal ceiling in every case. This pattern holds across datasets and encoder combinations, suggesting that strong-modality collapse is a structural property of symmetric cross-attention fusion rather than an artefact of any particular encoder.

Feature-space visualisation of collapse (t-SNE). Figure 4 shows a joint t-SNE embedding of the text-pathway features used at the final classification step, coloured by prediction correctness. Both panels share a single embedding fitted on the combined feature matrix, so differences in error density reflect classifier behaviour rather than changes in the underlying feature geometry.

![](images/c923a86917d8b73acd8ad1028a789322569e3d60e0b817e7da5f41a61eca2026.jpg)  
Figure 3: Dominant-modality pathway accuracy for all configurations (data from Table 5). Shaded regions group MultiHuSE (blue) and MUStARD (purple) configurations.

In the IAF panel (left), the frozen unimodal classifier $\psi _ { T }$ operates on the backbone features $\mathbf { \delta } _ { h } ( T )$ on which it was trained; text-pathway accuracy on fold 4 is 76.7%. In the symmetric fusion panel (right), the shared classifier operates on post-crossattention features $\tilde { h } ^ { ( T ) }$ ; accuracy drops to 40.3%. The residual connection preserves a broadly similar manifold structure across both panels, yet the symmetric pathway produces substantially more errors in the same geometric space. This pattern is consistent with classifier mismatch: the shared head is not optimised for single-pathway inputs.

![](images/3faff9e85dce250988031cb41f38bf8ce7bd14001dfb6da508c59be300d8a968.jpg)  
Figure 4: Joint t-SNE of features used at the final classification step for the text pathway under IAF (left) and symmetric fusion (right), fold 4 of MultiHuSE (E5 + PE-Core, distilled). Green = correctly classified by that pathway; red = misclassified.

## F Gate Weight Analysis

The IAF gate network produces per-sample weights $( w _ { T } , w _ { A } , w _ { V } )$ that sum to one and determine each modality’s contribution before classification. Table 13 reports mean gate weights averaged across cross-validation folds at the best checkpoint for all nine configurations. Two population-level patterns emerge. First, the modality assigned the PURE role consistently receives the highest mean weight: text leads on MultiHuSE $( \bar { w } _ { T } ~ \in ~ [ 0 . 3 5 , 0 . 4 7 ] )$ whereas audio and video dominate on MUStARD $( \bar { w } _ { A } + \bar { w } _ { V } > 0 . 8 0$ in all configurations). On UR-FUNNY, weights are more evenly distributed $( \bar { w } _ { T } = 0 . 3 7 2 .$ $\bar { w } _ { A } = 0 . 3 1 0 .$ $\bar { w } _ { V } = 0 . 3 1 8 )$ , consistent with more balanced cross-modal cues in that corpus. Second, no modality exceeds a mean weight of 0.53 in any configuration, suggesting that IAF preserves multimodal integration rather than collapsing into a regularised unimodal model.

<table><tr><td>Text</td><td>Feat.</td><td>∆(%)</td><td> $\bar { w } _ { T }$ </td><td> $\bar { w } _ { A }$ </td><td> $\bar { w } _ { V }$ </td></tr><tr><td colspan="6">MultiHuSE (Text PURE | Dasheng+PE-Core)</td></tr><tr><td>BERT+PE-Core</td><td>Base Dist.</td><td> $+ 4 . 0 7 ^ { [ \mathrm { B ] } }$   $+ 6 . 4 8 ^ { [ \mathrm { { B } ] } }$ </td><td>0.474 0.355</td><td>0.355 0.206</td><td>0.171 0.439</td></tr><tr><td rowspan="2">E5+PE-Core</td><td>Base</td><td> $+ 1 . 0 4 ^ { \mathrm { [ E ] } }$ </td><td>0.431</td><td>0.425</td><td>0.144</td></tr><tr><td>Dist.</td><td> $+ 3 . 2 0 ^ { \mathrm { [ E ] } }$ </td><td>0.412</td><td>0.197</td><td>0.391</td></tr><tr><td>UR-FUNNY (Text PURE</td><td></td><td></td><td>|COVAREP + OpenFace)</td><td></td><td></td></tr><tr><td colspan="6">GloVe</td></tr><tr><td></td><td>Dist.</td><td>+8.25</td><td>0.372</td><td>0.310</td><td>0.318</td></tr><tr><td rowspan="2">MUStARD (Audio+Video PURE BERT+PE-Core</td><td></td><td></td><td></td><td>| Text ATTEND)</td><td></td></tr><tr><td>Base</td><td> $+ 3 . 4 7 ^ { [ \mathrm { B ] } }$   $+ 4 . 4 9 ^ { [ \mathrm { { B } ] } }$ </td><td>0.194</td><td>0.530</td><td>0.276 0.449</td></tr><tr><td rowspan="2">E5+PE-Core</td><td>Dist.</td><td> $+ 3 . 6 2 ^ { [ \mathrm { E ] } }$ </td><td>0.161</td><td>0.389</td><td></td></tr><tr><td>Base Dist.</td><td> $+ 5 . 2 1 ^ { [ \mathrm { E } ] }$ </td><td>0.166 0.103</td><td>0.440 0.470</td><td>0.394 0.428</td></tr></table>

Table 13: Mean learned gate weights across CV folds at the best checkpoint. ∆ = gain over dominant unimodal baseline (MultiHuSE: $\mathrm { B E R T } = 6 8 . 2 6 \%$ $\mathrm { E } 5 = 7 4 . 8 6 \%$ ; UR-FUNNY: GloVe = 62.47%; MUStARD: Dasheng = 78.12%). ${ \mathrm { [ B ] } } _ { / { \mathrm { [ E ] } } } .$ : gap from BERT/E5 baseline. Base = raw pre-trained; Dist. = knowledgedistilled.

These averages, however, mask substantial within-dataset variation. Table 14 presents representative samples from the distilled IAF model (E5 + PE-Core) on MultiHuSE. Although the population average is $\bar { w } _ { T } \approx 0 . 5 1$ , individual samples range from near-exclusive text weighting (w<sub>T</sub> > 0.95) to strong audio dominance $( w _ { A } > 0 . 8 5 )$ Four cases illustrate this adaptive behaviour. In Category (i), unambiguous headline-style utterances labelled as neutral humour are classified primarily from text. In Category (ii), deadpan or context-dependent phrasing provides weak textual evidence, and the gate shifts weight toward audio and visual cues while preserving correct classification. In Category (iii), the model shows overreliance on text: utterances such as Age is of no importance unless you’re a cheese depend on delivery cues absent from the transcript, leading to misclassification. In Category (iv), audio becomes dominant when the transcript is relatively uninformative; for example, Tomorrow is 2-22-22. Happy Tuesday! appears textually plain, but vocal warmth and intonation support affiliative classification, and the gate assigns about 90% of the weight to audio. Overall, the aggregate trends align with the PURE/ATTEND assignments, while the case studies show that the gate adapts to sample-level variation in modality reliability.

<table><tr><td>Sample text</td><td>True</td><td>Pred</td><td>WT</td><td>WA</td><td>wv</td></tr><tr><td>(i) High text weight, correctly classified Joe Kennedy III reveals how his GOP</td><td>Neu</td><td>Neu</td><td>0.970</td><td>0.006</td><td>0.024√</td></tr><tr><td>counterparts really feel about Trump&#x27;s tweets. Katy Perry wears American flag dress to</td><td>Neu</td><td>Neu</td><td>0.961</td><td>0.018</td><td>0.020 √</td></tr><tr><td>kids in overall concerts. Porn actress confirms Trump affair after</td><td>Neu</td><td>Neu</td><td>0.959</td><td>0.020</td><td>0.021✓</td></tr><tr><td>unpublished 2011 interview (ii) Low text weight, correctly classified</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>The only positive thing about you is your</td><td>Agg</td><td>Agg</td><td>0.023</td><td>0.548</td><td>0.429 √</td></tr><tr><td>HIV status. Advice to ice skaters: you can&#x27;t always</td><td>Neu</td><td>Neu</td><td>0.024</td><td>0.643</td><td>0.333 √</td></tr><tr><td>tell a brick by its cover. He who laughs last probably doesn&#x27;t</td><td>Neu</td><td>Neu</td><td>0.028</td><td>0.685</td><td>0.287 √</td></tr><tr><td>understand the joke. (iii) High text weight, incorrectly classified</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Age is of no importance unless you&#x27;re a S-enh</td><td></td><td>Agg</td><td>0.885</td><td>0.032</td><td>0.082 x</td></tr><tr><td>cheese. Don&#x27;t look back. You&#x27;re not going that S-enh</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>way.</td><td></td><td>Agg</td><td>0.863</td><td>0.049</td><td>0.088 x</td></tr><tr><td>What is the best contraceptive for old peo- ple? Nudity.</td><td>Agg</td><td>Aff</td><td>0.857</td><td>0.084</td><td>0.058 x</td></tr><tr><td>(iv) Audio-dominant, correctly classified How many Californians does it take to</td><td>Neu</td><td>Neu</td><td>0.039</td><td>0.900</td><td>0.061√</td></tr><tr><td>screw in a light bulb? None. They screw in hot tubs.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tomorrow is 2-22-22. Happy Tuesday!</td><td>Aff</td><td>Aff</td><td>0.135</td><td>0.856</td><td>0.009√</td></tr><tr><td>Why couldn&#x27;t the leopard play hide and seek? Because he was always spotted.</td><td>Aff</td><td>Aff</td><td>0.102</td><td>0.851</td><td>0.048√</td></tr></table>

Table 14: Representative IAF gate weight profiles (E5 + PE-Core, MultiHuSE, all folds). This table contains potentially offensive or sensitive example text reproduced from the dataset for analysis. Class labels: Aff = Affiliative, Agg = Aggressive, Neu = Neutral, Senh = Self-enhancing. w<sub>T</sub>+w<sub>A</sub>+w<sub>V</sub> = 1 per sample. ✓ correct; ✗ incorrect.

## G Distillation Coefficient Sensitivity

Table 15 reports the effect of varying the three distillation loss coefficients (α: hard CE, β: soft KD, γ: feature alignment) on MultiHuSE. We evaluate three alternative configurations alongside the default setting: equal weights, a CE-heavy variant, and a soft-target-heavy variant. All distilled students are then used as inputs to IAF under the same fusion training procedure.

Fusion accuracy for IAF with E5, audio, and PE-Core varies by at most 1.20 percentage points across the four configurations. The two bestperforming settings, the default and equal-weight variants, achieve the same top-line accuracy of 78.06%. CE-heavy weighting yields a small improvement in video distillation (+0.38 pp relative to the default) but reduces cross-modal soft transfer, leading to a 1.20 pp drop in fusion accuracy.

<table><tr><td>Setting</td><td>Coeff. (α/β/γ)</td><td>BERT Dist.</td><td>Audio Dist.</td><td>Video Dist.</td><td>IAF (BERT)</td><td>IAF (E5)</td></tr><tr><td>Default†</td><td>0.40 / 0.35 / 0.25</td><td>70.21±2.82</td><td>62.90±1.83</td><td>45.57±1.22</td><td>74.74±1.71</td><td>78.06±3.82</td></tr><tr><td>Equal weights</td><td>0.33 / 0.33 / 0.33</td><td>69.92±2.79</td><td>63.11±1.54</td><td>44.66±1.57</td><td>74.37±2.33</td><td>78.06±3.36</td></tr><tr><td>CE-heavy</td><td>0.60 / 0.25 / 0.15</td><td>69.63±2.75</td><td>62.78±1.18</td><td>45.95±0.98</td><td>74.03±2.48</td><td>76.86±4.32</td></tr><tr><td>Soft-target-heavy</td><td>0.25 / 0.55 / 0.20</td><td>70.17±2.58</td><td>62.86±1.10</td><td>44.74±1.35</td><td>73.33±2.32</td><td>76.90±4.18</td></tr><tr><td>Range</td><td></td><td>0.58</td><td>0.33</td><td>1.29</td><td>1.41</td><td>1.20</td></tr></table>

Table 15: Distillation coefficient sensitivity on Multi-HuSE (5-fold CV, mean ± SD %). † = setting used in all main experiments. IAF columns use distilled students; Audio = Dasheng, Video = PE-Core throughout. “Range” reports max − min across the four settings.

The soft-target-heavy variant lowers video distillation performance (-0.83 pp relative to the default), suggesting that the weakest modality depends more on direct task supervision through CE than on soft targets from a teacher trained on a different modality. In this setting, increasing the weight on soft targets from the text teacher appears to overwhelm the video student’s task-specific learning signal. Audio distillation remains stable across all settings, with a range of at most 0.33 pp, indicating that the Dasheng encoder adapts consistently regardless of coefficient choice. Overall, these results suggest that IAF is not highly sensitive to the precise coefficient values; the default setting was selected by manual tuning and then held fixed across all datasets.

## H Knowledge Distillation Design Comparison

Table 16 situates MAKD relative to prior multimodal distillation approaches along five design axes: training stage, teacher source, objective, and optimisation procedure. The key distinction is that MAKD is a standalone pre-fusion stage that uses only the empirically dominant unimodal modality as teacher to strengthen weaker encoders before fusion, whereas prior multimodal KD methods typically operate within the fusion stage itself.

<table><tr><td>Method</td><td>Stage</td><td>Teacher</td><td>Purpose</td><td>Procedure</td></tr><tr><td>Cross-M KD (Albanie et al., 2018) (Thoker and Gall, 2019)</td><td>Pre-fusion†</td><td>Paired modality</td><td>Unimodal transfer</td><td>Sequential</td></tr><tr><td>Wang et al. (Wang et al., 2020a)</td><td></td><td>Within fusion Full multimodal</td><td>Dropout robustness</td><td>Joint</td></tr><tr><td>MMANet (Wei et al., 2023)</td><td></td><td>Within fusion Full multimodal</td><td>Missing-modality</td><td>Joint</td></tr><tr><td>DMD (Li et al., 2023b)</td><td></td><td>Within fusion Subspace teachers</td><td>Repr. disentanglement</td><td>Concurrent</td></tr><tr><td>MTMD (Lin and Hu, 2024)</td><td></td><td>Within fusion Unimodal + fused</td><td>Heterogeneity reduction</td><td>Concurrent</td></tr><tr><td>MAKD (Ours)</td><td>Pre-fusion</td><td>Dom. unimodal</td><td>Encoder strengthening Sequential</td><td></td></tr></table>

Table 16: Design comparison of multimodal knowledge distillation methods. <sup>†</sup>Not designed in the context of a subsequent fusion stage.