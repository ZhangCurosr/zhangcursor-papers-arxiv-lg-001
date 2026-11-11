# Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms

Siyuan Yan, Haibo Zheng, Feng Lu, Wei Li

Abstract—Electrocardiogram (ECG) signals are critical for cardiovascular disease (CVD) detection. Most current deep learning models rely on supervised, task-specific training, which limits their ability to generalize across diseases, patient populations, and acquisition settings. This paper presents a self-supervised ECG feature extraction framework that learns from nearly one million unlabeled 12-lead ECG recordings. The framework includes two modules: Deviation Localization Embedding (DLE), which detects waveform-level deviations from normal patterns, and Association Reconstruction Embedding (ARE), which models crosslead associations through deviation-guided reconstruction to capture abnormal patterns expressed across multiple leads. These modules enable the framework to capture subtle deviation-related patterns and their cross-lead context. The framework achieves up to 6.5% AUROC and 12.8% F1 improvements over prior methods on multiple public datasets, even without additional fine-tuning the pretrained feature extractor. Embedding visualizations show that the learned features support meaningful disease grouping and anomaly identification. The approach reduces the need for labeled data and can support clinicians in diverse healthcare settings. The framework has potential to improve early detection and monitoring of cardiovascular diseases, contributing to more accessible and equitable heart care.

Impact Statement—This work provides a self-supervised ECG feature extraction framework that can learn transferable representations from large-scale unlabeled 12-lead ECG recordings. By reducing dependence on disease-specific labels and downstream fine-tuning, the framework may lower the cost of developing ECG analysis models for diverse clinical scenarios. The proposed deviation localization and association reconstruction modules help identify subtle waveform abnormalities and crosslead pathological patterns, offering more interpretable features for cardiovascular disease detection. Its generalization across multiple public and clinical datasets suggests potential value for early screening, risk assessment, and long-term monitoring, especially in settings where expert annotations are limited. More broadly, this study supports the development of reusable ECG representation models that may improve the accessibility, scalability, and reliability of AI-assisted cardiovascular care.

Index Terms—Anomaly detection, cardiovascular disease, electrocardiography, feature extraction, multi-lead ECG, selfsupervised learning.

## I. INTRODUCTION

cardiovascular medicine, providing non-invasive, realtime insights into heart function. ECG analysis plays a key role in the assessment and management of risk of arrhythmias, myocardial infarction, heart failure, and sudden cardiac death [1], [2], [3], [4]. The increasing availability of large-scale ECG datasets from clinical systems and wearable devices has supported the application of deep learning to heartbeat classification [5], [6] and ECG-based heart disease classification [7].

![](images/f2d9f361e51ab7a4f250f1b8f33231e981955757a883e87ccc29e202634336bc.jpg)  
Fig. 1. The deviation localization and association reconstruction of multilead ECG data. (a)–(c) Deviation localization: per-lead anomaly localization maps for leads II, III, and V4. (d) Vectorized deviation from continuous view to capture anomaly. (e)–(g) Original ECG waveforms for a MI patient. (h) Deviation-related contextual information derived from leads II and III. (i) Deviation-guided cross-lead association for the target lead. (j) Reconstructed lead V4 after association reconstruction. The red squares mark the position of obvious deviations.

Other applications include detection of atrial fibrillation [8], identification and risk stratification of coronary cardiovascular risk [9], [10]. Most existing models are trained in a supervised, task-specific manner, with labels tied to particular diagnostic outcomes. These models typically learn associations specific to the training dataset, constrained by patient demographics, disease distribution, and recording conditions. As a result, learned representations often lack diversity and fail to generalize across samples, tasks, and databases.

Self-supervised representation learning uses unlabeled ECGs to learn representations of temporal patterns and inter-lead dependencies. Recent ECG representation-learning and anomaly-detection approaches include BeatGAN [11], ASTCL [12], and ECGAD [13]. ECG-FM [14] combines selfsupervised pretraining objectives, whereas ECGFounder [15] learns from large-scale clinical diagnostic annotations. Prior work has also explored the structure and interpretability of learned representations [14]. Foundation models developed for vision or general time-series analysis include DINOv2 [16], TimesFM [17], MOIRAI [18], and Chronos [19]. In this study, we focus on lead-wise deviations relative to normal ECG morphology and their integration through deviation-guided cross-lead reconstruction.

We propose a self-supervised ECG feature extraction framework that explicitly targets the extraction of both effective and generalizable features. Unlike prior approaches that focus on building connection between ECG and cardiac disease labels, our framework learns transferable representations from unlabeled ECG signals using two structured embeddings. This enables the model to learn label-free, transferable representations applicable and better understand the meaning of vibrations across diverse diagnostic tasks.

To this end, we introduce two complementary modules that capture generalizable pathological deviations from distinct yet synergistic perspectives in ECGs. The Deviation Localization Embedding (DLE) module identifies fine-grained waveformlevel deviations by learning deviations between test sample and standardized normal sample without CVDs (Fig.1(a)- (d)). This highlights subtle anomalies that may carry diagnostic significance. The Association Reconstruction Embedding (ARE) module then uses the localized deviation cues to guide cross-lead association reconstruction through a DLE-guided gating mechanism (Fig.1(h)-(j)). By reweighting the contribution of peer leads during target-lead reconstruction, ARE encourages the learned representation to capture cross-lead consistency that may not be fully characterized by individual leads alone. DLE and ARE jointly provide complementary deviation-related and cross-lead contextual information for masked reconstruction, encouraging the model to retain informative ECG representations without relying on diagnostic labels. These features remain expressive enough to distinguish pathological signals from normal ECGs, without relying on annotated labels.

By combining DLE and ARE, our model captures beat-level anomalies and multi-lead patterns, addressing low-SNR and patient variability issues. We pretrain on unlabeled ECGs and adapt to classifiers without any fine-tuning. Our contributions are:

• A self-supervised ECG feature extractor pre-trained on 918,915 recordings, learning generalizable intra-lead and cross-lead representations.

• A dual-module Deviation Learning Embedding(DLE) and Association Reconstruction Embedding(ARE) framework that identifies subtle waveform deviations and integrates deviation-guided cross-lead associations for abnormallyaware ECG representation learning.

• Our approach has SOTA transfer performance on four public datasets, outperforming prior self-supervised extractors and ECG classifiers without downstream adaptation.

## II. RELATED WORK

## A. Feature Representation and Extraction

Feature extraction is central to electrocardiogram (ECG) analysis, as it converts high-dimensional physiological signals into compact and discriminative representations [20], [21]. Classical methods, including statistical descriptors, Fourier and wavelet transforms [22], [23], and principal component analysis (PCA) [24], rely heavily on hand-crafted features and prior signal-processing assumptions. In contrast, deep learning models, such as convolutional neural networks (CNNs) [25], [26] and recurrent neural networks (RNNs) [27], [28], can automatically learn task-relevant ECG embeddings from data. Their representation ability can be further improved by selfsupervised pretraining and data augmentation strategies [29].

Recent self-supervised ECG methods have explored different pretext tasks for feature learning. Contrastive Multisegment Coding (CMSC) [30] divides ECG recordings into segments and aligns their representations, while Contrastive Predictive Coding (CPC) [31], [32] learns temporal representations by predicting future latent representations. However, recurrent models used in such frameworks may face scalability challenges when modeling long ECG sequences. Masked autoencoder-based methods further introduce temporal and lead-wise recovery objectives for ECG representation learning [33], and ST-MEM applies random interval masking to each lead to capture spatio-temporal dependencies [34]. Although these approaches improve local context modeling, many of them still rely strongly on inter-lead correlations or generic reconstruction objectives. Such designs may obscure lead-specific characteristics and limit the detection of subtle, syndrome-specific abnormalities. This limitation reduces their clinical utility in scenarios where cross-lead consistency and fine-grained morphological distinctions are both important.

## B. Self-Supervised ECG Representation Learning

Self-supervised learning (SSL) enables representation learning without manual annotations by exploiting the inherent structure and contextual information within data [35]. Representative SSL frameworks include SimCLR [36], Contrastive Predictive Coding (CPC) [31], Momentum Contrast (MoCo) [37], and Bootstrap Your Own Latent (BYOL) [38]. These methods usually define pretext tasks that encourage models to learn invariant and discriminative representations, such as distinguishing augmented views of the same input or predicting future latent representations. In ECG analysis, adaptations of CPC learn temporal representations by predicting future latent representations from preceding context [32]. Liu et al. [39] investigated self-supervised pretraining for ECG morphology recognition using ECG-specific data augmentations. Beyond contrastive learning, generative SSL methods, such as autoencoder-based frameworks, reconstruct ECG signals to preserve morphological and structural information in the learned embeddings [40].

Despite their effectiveness in ECG classification, existing SSL methods still face limitations in clinical generalization. Many methods are primarily optimized for within-dataset performance and remain sensitive to distribution shifts caused by differences in patient populations, acquisition devices, recording protocols, and disease distributions. Moreover, most existing frameworks focus on temporal consistency or global signal reconstruction, while lacking explicit mechanisms to model inter-lead structure and clinically meaningful abnormal patterns across 12-lead ECG recordings. As a result, they may fail to capture rare, subtle, or previously unseen abnormalities, limiting their reliability in real-world diagnostic settings.

## III. METHODOLOGY

Our proposed framework is designed to learn robust ECG representations through a self-supervised task. The pre-training process adopts a Self-Supervised ScalesMamba2 Block (SSMb) as the backbone. The feature extraction part consists of two embedding extractors: DLE, which captures weak anomalies within ECG leads through Deviation Scales-Mamba2 (DMb), and ARE, which performs deviation-guided cross-lead association reconstruction in a latent representation space. The extracted features and the original data segment are sent to the SSMb block for mask reconstruction. The entire self-supervised framework incorporates Rotational Partial Freezing to ensure the effectiveness of the embedding extraction, see framework overview in Fig. 2.

![](images/160a95ea4c32c32324e175694c58d7448bceacf83ab9b53520b03fee913335b3.jpg)  
Fig. 2. Overview framework of ScalesMamba2-based ECG feature extraction model to learn generic dissimilarity features from unlabeled ECGs

## A. ScalesMamba2 Architecture

To effectively model the hierarchical nature of ECG signals, where localized morphological patterns are nested within global rhythm dynamics, we introduce the ScalesMamba2 architecture. ECG segments contain both fine-grained P-QRS-T morphology and long-range rhythm context; therefore, the model needs to preserve local waveform details while capturing broader temporal dependencies. ScalesMamba2 adopts a U-Net-like multi-scale topology and uses Mamba2 [41] as its temporal modeling core.

Given an input sequence $u _ { t } ,$ Mamba2 first projects it into a hidden representation and updates an input-dependent latent state:

$$
\begin{array} { r l } & { h _ { t } = W _ { h } u _ { t } + b _ { h } , } \\ & { x _ { t } = A _ { t } ( h _ { t } ) x _ { t - 1 } + B _ { t } ( h _ { t } ) h _ { t } , } \\ & { y _ { t } = C _ { t } ( h _ { t } ) x _ { t } . } \end{array}\tag{1}
$$

Here, the effective state update is dynamically parameterized by the current representation $h _ { t }$ . This content-dependent update enables the model to adapt to non-stationary ECG dy-

namics, including local morphological variations and rhythm changes.

The ScalesMamba2 layer wraps this temporal modeling core into a residual multi-scale operation:

$$
\operatorname { S c a l e s M a m b a 2 } ( U ) = U + \operatorname { U p } \left( \operatorname { M a m b a 2 } \left( \operatorname { D o w n } ( U ) \right) \right) .\tag{2}
$$

where Down(·) and $\mathrm { U p } ( \cdot )$ denote temporal resolution changes. The downsampled path allows Mamba2 to model broader temporal context on compressed representations, while the residual connection preserves fine-grained local morphological details that may be weakened during resolution changes.

Based on the ScalesMamba2 layer, we construct two U-Net-like blocks: the Deviation Mamba2 Block (DMb) and the Self-Supervised Mamba2 Block (SSMb). Both blocks stack ScalesMamba2 layers in an encoder-decoder structure with skip connections, but they are designed for different objectives. DMb is a 3-scale block used to refine the precomputed deviation signal σ from DLE, focusing on localized and midrange deviation patterns. In contrast, SSMb is a deeper 5-scale block used for self-supervised masked reconstruction from raw ECG segments, where broader temporal context is required to recover missing waveform regions. This design assigns different depths according to task complexity: deviation refinement mainly focuses on deviation-related temporal patterns, whereas masked reconstruction requires both local morphology and global rhythm context.

## B. Self-Supervised Pretraining Framework

The model is pretrained on large-scale unlabeled ECG databases. The core objective is a deviation-aware symmetric masked reconstruction task, together with a training strategy that encourages the learning of meaningful ECG representations.

![](images/caffbd0e0fc15fc5280ab4d35b4a2dc005db9d9f0da793fc55a165bf423bc9c1.jpg)  
Fig. 3. Overall architecture of the formation of NormECG Cluster and NormECG Codebook.

1) Data Segmentation and Input Formulation: Each 12-lead ECG recording is first segmented into non-overlapping windows of 500 samples, a duration typically sufficient to cover a full P-QRS-T cardiac cycle at 500 Hz. Let $x _ { k } ^ { ( i , j ) } \in \mathbb { R } ^ { 5 0 0 }$ denote the i-th segment from the j-th lead of the k-th ECG recording.

For the reconstruction task, two embeddings are computed for each segment: $\sigma _ { k } ^ { \left( i \right) }$ from DLE, which provides deviationrelated cues, and $E _ { k } ^ { \left( i \right) }$ from ARE, which provides cross-lead association information. We then employ a symmetric masking strategy to ensure full temporal reconstruction coverage and reduce reliance on trivial local interpolation. For each lead $j$ in segment i, we generate a random binary mask $M _ { k } ^ { ( i , j , 1 ) }$ and its complement $\bar { M } _ { k } ^ { ( i , j , 2 ) } = 1 - M _ { k } ^ { ( i , j , 1 ) }$ , where $M = 1$ denotes visible positions used as input. Thus, each temporal position is masked and reconstructed once across the two complementary views.

The model input for each view $m \in \{ 1 , 2 \}$ is formed by concatenating the embeddings with the corresponding masked signal:

$$
F _ { k } ^ { ( i , j , m ) } = \left[ \sigma _ { k } ^ { ( i ) } ; E _ { k } ^ { ( i ) } ; M _ { k } ^ { ( i , j , m ) } \odot x _ { k } ^ { ( i , j ) } \right] .\tag{3}
$$

This setup encourages the model to reconstruct masked regions by leveraging both deviation-related cues and cross-lead association information.

2) Reconstruction Loss: The reconstruction loss is computed only on the masked-out regions. We first calculate the masked residual for each view m:

$$
r _ { k } ^ { ( i , j , m ) } = \left( \hat { x } _ { k } ^ { ( i , j , m ) } - x _ { k } ^ { ( i , j ) } \right) \odot \left( 1 - M _ { k } ^ { ( i , j , m ) } \right) ,\tag{4}
$$

where $\hat { x } _ { k } ^ { ( i , j , m ) }$ is the model reconstruction. The total loss is the sum of Smooth L1 losses over all dimensions, which balances robustness and smoothness:

$$
\mathcal { L } _ { \mathrm { S S } } = \sum _ { i , j , k , m } \mathcal { L } _ { \delta } ^ { \mathrm { S m o o t h L 1 } } \left( r _ { k } ^ { ( i , j , m ) } , 0 \right) .\tag{5}
$$

3) Weighted Rotational Partial Freezing: To ensure that the model learns generalizable features rather than overfitting to the masked reconstruction objective, we employ the Weighted Rotational Partial Freezing (RPF) strategy. Specifically, the model parameters θ are partitioned into G disjoint blocks $\theta =$ $[ \theta _ { 1 } , \dots , \theta _ { G } ]$ , which are grouped into the feature extractor $\mathcal { E }$ and self-supervision head H. The feature extractor contains the representation-learning modules, while the self-supervision head contains the reconstruction-specific layers responsible for masked signal recovery. At each training step t, one block $g ( t )$ is selected for update according to a non-uniform probability distribution $\mathbf { p } ~ = ~ [ p _ { 1 } , \ldots , p _ { G } ]$ , where $p _ { g } \ = \ \operatorname* { P r } ( g ( t ) \ = \ g )$ while the remaining blocks are temporarily frozen. To prioritize representation learning, extractor blocks are assigned higher update probabilities than head blocks, i.e., $p _ { i } > p _ { j }$ for $\theta _ { i } ~ \in ~ \mathcal { E }$ and $\theta _ { j } \ \in \ { \mathcal { H } } .$ The total update probability assigned to the self-supervision head can be further bounded by $\gamma _ { \mathrm { m a x } } ,$ preventing the reconstruction-specific layers from dominating the optimization. The update rule is:

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \alpha ^ { ( t ) } M _ { g ( t ) } \odot \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S S } } ,\tag{6}
$$

where $M _ { g ( t ) }$ is the mask activating only block $g ( t )$ . This weighted block-wise update prioritizes the learning of transferable deviation and association embeddings while regularizing the reconstruction-specific head.

## C. Feature Extraction Modules

1) Deviation Localization Embedding: The Deviation Localization Embedding (DLE) quantifies a segment’s deviation from a learned codebook of normal cardiac morphologies.

NormECG Codebook: A codebook $\boldsymbol { B } _ { \mathrm { N o r m E C G } } = \{ \mu _ { c } \} _ { c = 1 } ^ { C }$ of C normal ECG prototypes is built via Bayesian clustering with Soft-DTW distance, governed by a smoothing parameter λ. The prototypes $\mu _ { c }$ are updated via a standard M-step, as shown in Fig. 3. The codebook serves as a morphologylevel reference of normal ECG patterns. Before clustering, normal ECG segments are temporally aligned to reduce phase shifts among cardiac cycles. Multi-lead DTW and barycenter aggregation are then used to obtain representative normal prototypes, which are later used for per-lead deviation calculation.

Deviation Calculation: For an input segment $\boldsymbol { x } _ { k } ^ { ( i ) }$ , we compute its similarity index $H ^ { S }$ through an MLP layer for dynamic hashing. We then find its M nearest neighbors $\{ \mu _ { c _ { n } ^ { * } } \} _ { n = 1 } ^ { M }$ from the codebook using cosine similarity on $H ^ { S }$ The per-lead deviation is computed as the RMSE between the input lead and the retrieved normal prototypes:

$$
\sigma _ { k , j } ^ { \left( i \right) } = \mathrm { R M S E } \left( x _ { k , j } ^ { \left( i \right) } - \mu _ { c _ { n } ^ { * } , j } \right) .\tag{7}
$$

Deviation Refinement and Embedding: Each per-lead deviation signal $\sigma _ { k , j } ^ { ( i ) }$ is refined by a 3-scale Deviation Mamba2 Block (DMb) to capture deviation patterns across multiple

temporal scales. The final embedding is the sum of these refined signals:

$$
\sigma _ { k } ^ { \left( i \right) } = \sum _ { j = 1 } ^ { L } \mathrm { D M b } \left( \sigma _ { k , j } ^ { \left( i \right) } \right) .\tag{8}
$$

This embedding is also passed through a scaled sigmoid function $\tau ( \cdot )$ to produce a diagonal Selective Fusion Matrix (SFM), $\mathrm { S F M } _ { k } ^ { ( i ) } \in \mathbb { R } ^ { L \times L }$ , which assigns larger weights to leads with stronger deviations and provides deviation-guided lead weights for ARE

2) Multi-Lead Association Reconstruction Embedding: The Multi-Lead Association Reconstruction Embedding (ARE) module reconstructs each target lead from its peer leads under the guidance of DLE-derived deviation cues. ARE aims to align lead representations in a shared latent association space and integrate cross-lead contextual information through association reconstruction. For each target lead j, the following steps are performed.

Context Assembly and Selection: The initial context $Q _ { 0 , k } ^ { ( j ) }$ is formed by the other $L - 1$ leads. This context is then pruned by a composite selector matrix $S _ { k } ^ { ( j ) }$ to form the final context $\begin{array} { r } { \dot { Q } _ { k } ^ { ( j ) } = \hat { S _ { k } ^ { ( j ) } } Q _ { 0 , k } ^ { ( j ) } } \end{array}$ . The selector $S _ { k } ^ { ( j ) }$ is constructed to: 1) capture sparse and structured lead-to-lead associations using a learnable sparse matrix $M ^ { ( j ) }$ , and 2) incorporate peer leads with stronger deviation cues emphasized by the DLE-derived SFM<sub>k</sub> (i) matrix:

$$
S _ { k } ^ { ( j ) } = M ^ { ( j ) } \vee \mathrm { d i a g } \left( \mathbf { 1 } _ { \{ \mathrm { d i a g } ( \mathrm { S F M } _ { k } ^ { ( \setminus j ) } ) > 0 \} } \right) .\tag{9}
$$

Here, $\mathrm { S F M } _ { k } ^ { ( \backslash j ) }$ denotes the segment-level aggregated $\mathrm { S F M } _ { k }$ excluding lead $j$

Deviation-Guided Attention: The context $Q _ { k } ^ { ( j ) }$ is processed by attention layers. To bridge the DLE and ARE modules, we introduce a deviation-guided gating mechanism that uses the DLE-derived SFM to modulate the attention computation. The key vectors are reweighted according to the per-lead deviation cues of the context leads::

$$
K _ { t } ^ { ( j ) } = \left( I + \beta \mathrm { S F M } _ { k } ^ { ( \backslash j ) } \right) W _ { t } ^ { K } Q _ { k } ^ { ( j ) } .\tag{10}
$$

This guides the attention module to align the target-lead representation with peer-lead representations carrying stronger deviation cues, enabling the reconstruction process to model cross-lead consistency in the latent association space. The attention output is then computed as:

$$
Z _ { t } ^ { ( j ) } = \mathrm { A t t n } \left( Q _ { k } ^ { ( j ) } , K _ { k } ^ { ( j ) } , V _ { k } ^ { ( j ) } \right) .\tag{11}
$$

Reconstruction and Fusion: The attention output $Z _ { t } ^ { ( j ) }$ is used to generate a reconstruction $\hat { x } _ { k , j }$ of the target lead $j .$ The final ARE embedding $E _ { k }$ is the aggregation of all such per-lead reconstructions, producing a context-aware embedding that integrates aligned cross-lead representations while reducing isolated lead-specific fluctuations:

$$
E _ { k } ^ { ( i ) } = \sum _ { j = 1 } ^ { L } \hat { x } _ { k , j } .\tag{12}
$$

## IV. EXPERIMENTS AND RESULTS

## A. Datasets

Pretraining data. We pretrained our feature extractor on 918,915 ECG recordings, comprising 800,137 from the MIMIC-IV-ECG [42] and 118,778 from a clinical partner. The data were randomly split into 80% for training and 20% for validation.

Evaluation datasets. We used four publicly available 12- lead ECG datasets for validation experiments. PTB-XL [43] contains 21,837 10-second recordings at 500 Hz annotated into five diagnostic classes from Europe. ML12L [44] includes 25,770 recordings with durations of 10–60 s at 500 Hz, grouped into eight multi-label classes from China. CPSC2020 [45] provides 10,330 recordings with durations of 6–60 s at 500 Hz, labeled across nine classes from China. WGEX [46] comprises 12,600 10-second recordings at 500 Hz spanning seven classes from China. Among them, PTB-XL and ML12L are used for frozen-feature evaluation, while CPSC2020 and WGEX are utilized for external generalizability comparison.

## B. Models for Comparison

We compared our method against state-of-the-art selfsupervised feature extractors for frozen-feature evaluation and state-of-the-art zero-shot ECG classification models for generalization validation.

The compared models in frozen-feature evaluation include DINOv2 [16], TimesFM [17], Chronos [19], and MOIRAI [18]. The models used in the generalizability comparison include ASTCL [12], BeatGAN [11], MERL [47], and ECGAD [13].

## C. Implementation Details

Our experiments used 24 V100 GPUs to train selfsupervised models, run benchmark comparisons, and perform ablation studies. Pretrained parameters for DINOv2, TimesFM, Chronos, and MOIRAI were configured for largescale deployment in frozen-feature evaluation. The zero-shot classifiers, e.g., ASTCL, BeatGAN, MERL, and ECGAD, were loaded with large-scale parameter tables and trained using the same parameter settings. All configurations and hyperparameter setups are consistent with our model.

For self-supervised masking training, we partitioned the pretraining datasets into 100 subsets. For each subset, we tracked the best-performing parameters based on validation loss and propagated them to the next subset, enabling a continuous training process. For downstream evaluation, all datasets followed a 10-fold cross-validation protocol. PTB-XL and ML12L were used as benchmark datasets, while CPSC2020 and WGEX served as external test sets. The downstream learning rate was initialized at $1 \times 1 0 ^ { - 4 }$ and halved every 5 epochs over 20 epochs. To isolate the effect of feature representations, we conducted a frozen-feature evaluation by feeding different extracted features into the same downstream classifiers. The downstream classifiers contain three different encoders with different structures, including a four-layer convolutional network, a Transformer with four encoder layers and attention heads, and a Mamba-based sequence model, followed by the same linear decoder.

TABLE I  
CLASSIFICATION PERFORMANCE OF DIFFERENT DOWNSTREAM CLASSIFIERS ON PTB-XL AND ML12L USING 10-FOLD CROSS-VALIDATION, AND ON CPSC2020 AND WGEX USING EXTERNAL VALIDATION. UNTOKENIZED DENOTES MODELS TRAINED DIRECTLY ON RAW ECG SIGNALS WITHOUT A FEATURE EXTRACTION FRAMEWORK. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD, THE SECOND-BEST IS UNDERLINED, AND THE WORST IS italicized.
<table><tr><td rowspan="2">Classifier</td><td rowspan="2">Extractor</td><td colspan="3">PTB-XL</td><td colspan="3">ML12L</td></tr><tr><td>B-Acc</td><td>F1</td><td>AUROC</td><td>B-Acc</td><td>F1</td><td>AUROC</td></tr><tr><td rowspan="7">CNN</td><td>Untokenized</td><td>0.399±0.024</td><td>0.338±0.042</td><td>0.694±0.011</td><td>0.701±0.162</td><td>0.425±0.056</td><td>0.661±0.069</td></tr><tr><td>Dinov2</td><td>0.452±0.014</td><td>0.383±0.040</td><td>0.757±0.006</td><td>0.740±0.118</td><td>0.368±0.055</td><td>0.725±0.137</td></tr><tr><td>TimesFM</td><td> $0 . 4 7 4 { \pm } 0 . 0 2 6$ </td><td>0.225±0.053</td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.684±0.018</td><td>0.352±0.048</td><td>0.669±0.051</td></tr><tr><td>Chronos</td><td>0.382±0.055</td><td>0.162±0.071</td><td> $0 . 5 9 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>0.681±0.082</td><td>0.306±0.045</td><td>0.723±0.065</td></tr><tr><td>MOIRAI</td><td>0.387±0.115</td><td>0.075±0.059</td><td> $0 . 5 6 2 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td>0.487±0.243</td><td>0.175±0.060</td><td>0.562±0.237</td></tr><tr><td>Ours</td><td>0.493±0.040</td><td>0.511±0.017</td><td>0.795±0.007</td><td>0.771±0.015</td><td>0.440±0.045</td><td>0.719±0.075</td></tr><tr><td>Untokenized</td><td></td><td>0.492±0.024</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Mamba</td><td></td><td>0.498±0.033</td><td></td><td>0.797±0.022</td><td>0.700±0.168</td><td>0.424±0.058</td><td>0.679±0.069</td></tr><tr><td>Dinov2</td><td>0.480±0.014</td><td>0.302±0.047</td><td>0.767±0.015</td><td>0.716±0.157</td><td>0.377±0.055</td><td>0.658±0.149</td></tr><tr><td>TimesFM</td><td>0.477±0.015 0.455±0.013</td><td>0.259±0.039</td><td> $0 . 7 4 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.570±0.041 0.772±0.010</td><td>0.342±0.039 0.343±0.008</td><td>0.680±0.150</td></tr><tr><td>Chronos MOIRAI</td><td>0.198±0.030</td><td>0.025±0.008</td><td> $0 . 6 2 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td>0.298±0.087</td><td>0.355±0.014</td><td> $\underline { { 0 . 7 3 4 \pm 0 . 0 3 3 } }$  0.331±0.008</td></tr><tr><td>Ours</td><td>0.552±0.017</td><td>0.255±0.017  $\mathbf { 0 . 5 5 6 { \pm 0 . 0 1 3 } }$ </td><td> $0 . 5 2 0 { \scriptstyle \pm 0 . 0 0 5 }$   $\mathbf { 0 . 8 1 4 \bot 0 . 0 0 9 }$ </td><td>0.785±0.012</td><td>0.428±0.028</td><td>0.758±0.145</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Transformer</td><td>Untokenized</td><td>0.448±0.027</td><td>0.314±0.034</td><td>0.686±0.008</td><td>0.548±0.035</td><td>0.414±0.039</td><td>0.566±0.068</td></tr><tr><td>Dinov2</td><td> $0 . 4 7 1 { \pm } 0 . 0 1 4$ </td><td>0.279±0.028</td><td>0.736±0.007</td><td>0.571±0.045</td><td>0.379±0.048</td><td>0.638±0.110</td></tr><tr><td>TimesFM</td><td> $0 . 4 7 9 { \pm } 0 . 0 1 9$ </td><td>0.338±0.069</td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td>0.579±0.025</td><td>0.438±0.056</td><td>0.673±0.040</td></tr><tr><td>Chronos</td><td> $0 . 3 6 2 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.218±0.028</td><td> $\overline { { 0 . 5 9 I \pm 0 . 0 I O } }$ </td><td>0.462±0.011</td><td>0.318±0.042</td><td>0.403±0.054</td></tr><tr><td>MOIRAI Ours</td><td> $0 . 3 3 I \pm 0 . 0 2 I$   $\mathbf { 0 . 5 2 0 { \overset { . } { \bot } } 0 . 0 2 3 }$ </td><td>0.285±0.021</td><td> $0 . 6 1 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td>0.431±0.023 0.620±0.040</td><td>0.385±0.021</td><td>0.416±0.148</td></tr><tr><td></td><td></td><td>0.507±0.033</td><td>0.797±0.008</td><td></td><td>0.607±0.044</td><td>0.715±0.061</td></tr><tr><td rowspan="2"></td><td colspan="5"></td><td colspan="3">WGEX</td></tr><tr><td></td><td>B-Acc</td><td>F1</td><td>AUROC</td><td>B-Acc</td><td>F1</td><td>AUROC</td></tr><tr><td rowspan="6">CNN</td><td>Untokenized</td><td>0.356±0.046</td><td>0.385±0.041</td><td>0.696±0.020</td><td>0.485±0.104</td><td>0.431±0.041</td><td>0.569±0.031</td></tr><tr><td>Dinov2</td><td>0.483±0.089</td><td>0.372±0.057</td><td>0.760±0.031</td><td>0.512±0.091</td><td>0.404±0.041</td><td>0.472±0.048</td></tr><tr><td>TimesFM</td><td>0.464±0.013</td><td>0.379±0.073</td><td> $0 . 7 4 3 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td>0.509±0.075</td><td>0.364±0.046</td><td>0.447±0.047</td></tr><tr><td>Chronos</td><td>0.450±0.020</td><td>0.046±0.060</td><td> $0 . 5 9 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td>0.444±0.033</td><td>0.320±0.034</td><td>0.816±0.049</td></tr><tr><td>MOIRAI</td><td>0.322±0.060</td><td>0.162±0.024</td><td> $0 . 6 1 8 { \pm } 0 . 0 2 0$ </td><td>0.322±0.060</td><td>0.262±0.024</td><td>0.439±0.045</td></tr><tr><td>Ours</td><td>0.534±0.002</td><td>0.505±0.017</td><td>0.800±0.004</td><td>0.521±0.009</td><td>0.478±0.010</td><td>0.798±0.070</td></tr><tr><td rowspan="7">Mamba</td><td>Untokenized</td><td>0.478±0.038</td><td>0.460±0.057</td><td>0.770±0.037</td><td>0.494±0.006</td><td>0.423±0.057</td><td>0.654±0.086</td></tr><tr><td>Dinov2</td><td>0.488±0.097</td><td>0.333±0.026</td><td>0.776±0.045</td><td>0.529±0.033</td><td>0.418±0.027</td><td>0.493±0.073</td></tr><tr><td>TimesFM</td><td>0.450±0.021</td><td>0.167±0.057</td><td>0.700±0.016</td><td>0.516±0.046</td><td>0.378±0.113</td><td>0.597±0.130</td></tr><tr><td>Chronos</td><td>0.450±0.008</td><td>0.060±0.007</td><td>0.626±0.008</td><td>0.570±0.005</td><td>0.342±0.004</td><td>0.722±0.005</td></tr><tr><td>MOIRAI</td><td>0.263±0.027</td><td>0.211±0.021</td><td>0.519±0.007</td><td>0.363±0.027</td><td>0.311±0.021</td><td>0.327±0.005</td></tr><tr><td>Ours</td><td>0.547±0.000</td><td>0.557±0.011</td><td>0.780±0.007</td><td>0.584±0.010</td><td>0.426±0.003</td><td>0.775±0.140</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Transformer</td><td>Untokenized</td><td>0.478±0.014</td><td>0.299±0.102</td><td>0.700±0.006</td><td>0.378±0.014</td><td>0.399±0.102</td><td>0.606±0.032</td></tr><tr><td>Dinov2</td><td>0.487±0.045</td><td>0.250±0.037</td><td>0.752±0.009</td><td>0.387±0.045</td><td>0.350±0.037</td><td>0.622±0.025</td></tr><tr><td>TimesFM</td><td>0.464±0.011</td><td>0.343±0.014</td><td>0.676±0.027</td><td>0.364±0.011</td><td>0.443±0.014</td><td>0.652±0.035</td></tr><tr><td>Chronos MOIRAI</td><td>0.388±0.020 0.314±0.044</td><td>0.176±0.012 0.302±0.021</td><td>0.591±0.009 0.603±0.013</td><td>0.288±0.020 0.314±0.044</td><td>0.276±0.012 0.402±0.021</td><td>0.452±0.005 0.402±0.045</td></tr><tr><td>Ours</td><td>0.533±0.035</td><td>0.548±0.002</td><td>0.791±0.007</td><td>0.433±0.035</td><td>0.538±0.002</td><td>0.809±0.019</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE II

PERFORMANCE COMPARISON OF SOTA SELF-SUPERVISED ECGCLASSIFIERS ON THE UNSEEN TEST SETS CPSC2020 AND WGEX USING10-FOLD VALIDATION.
<table><tr><td rowspan="2">Model</td><td colspan="3">CPSC2020</td><td colspan="4">WGEX</td></tr><tr><td>B-Acc</td><td>F1</td><td>AUC</td><td>Sens.</td><td>B-Acc</td><td>F1</td><td>AUC Sens.</td></tr><tr><td>ASTCL</td><td>0.496</td><td>0.522</td><td>0.744</td><td>0.700</td><td>0.530</td><td>0.462 0.724</td><td>0.699</td></tr><tr><td>BeatGAN</td><td>0.480</td><td>0.546</td><td>0.762</td><td>0.690</td><td>0.501</td><td>0.476 0.765</td><td>0.711</td></tr><tr><td>MERL</td><td>0.243</td><td>0.462</td><td>0.724</td><td>0.686</td><td>0.208</td><td>0.260 0.673</td><td>0.522</td></tr><tr><td>ECGAD</td><td>0.502</td><td>0.436</td><td>0.740</td><td>0.723</td><td>0.515</td><td>0.338 0.730</td><td>0.667</td></tr><tr><td>Ours</td><td>0.547</td><td>0.557</td><td>0.800</td><td>0.740</td><td>0.584</td><td>0.538 0.809</td><td>0.712</td></tr></table>

## D. Comparison with State-of-the-Art Models

We first conducted a frozen-feature evaluation by training lightweight classifiers, including CNN, Mamba, and Transformer, on feature representations extracted from different pretrained models. As shown in Table I, our method consistently achieved the best balanced accuracy, F1 score, and AUROC on both PTB-XL and ML12L.

With downstream training, our features outperformed all baselines on PTB-XL and ML12L. On PTB-XL, under the CNN classifier, our method achieved superior performance compared with TimesFM. Under the Mamba classifier, our method also outperformed the strongest baseline. On ML12L, our method achieved better performance under the Transformer classifier, exceeding TimesFM by 4.1% in F1 score and 4.2% in AUROC. When the pretrained feature extractor was frozen and evaluated on external datasets, including CPSC2020 and WGEX, our model also showed the best overall performance. Under the Transformer classifier, our model exceeded TimesFM by over 20% in F1 score. These results demonstrate the robustness of our learned features across different classifiers and their resilience to label imbalance in multi-label tasks.

Our model showed better performance than other methods on unseen datasets. We evaluated its generalizability to out-ofdistribution data using CPSC2020 and WGEX. As shown in

![](images/6b506f8ee90e5a5836f7de88fbaa896d5b8bb1768d20c0e2c724a5c29d71adc0.jpg)

Fig. 4. Visualization of extracted features using different models. Different columns show features extracted by different models. Original denotes the raw ECG signals, while Deviation and Association denote features from the DLE and ARE extractors, respectively.  
![](images/790de3c3388c0c6f378467c1df3e0073049be6fd08de77f0403f31301f0b91e3.jpg)

![](images/a68a0d63cb3a459e1f14b65cfc04f854858d0fa98c6a687c5363afcf5de211b7.jpg)

![](images/3aab4180cb965e0f26f312a3f4e9692444400c98d8b6da5d94eb081570559f9a.jpg)

![](images/e04c31470729127b976ff99e339abeeef12ee16d9006fcd154b93f72796c2057.jpg)  
Fig. 5. Two-dimensional projection of extracted features of DLE and ARE on PTB-XL, ML12L, CPSC2020, and WGEX datasets. The X-axis shows the ARE mapping, and the Y-axis shows the DLE mapping. Gray points represent healthy patients; points in other colors correspond to different cardiac diseases.

Table II, our approach consistently outperformed all baselines. On CPSC2020, it exceeded ECGAD by 4.5–6.4% across B-Acc, F1 score, AUROC, and sensitivity. On WGEX, which uses a different acquisition protocol and sampling rate, it surpassed BeatGAN by 4.4–6.3% across the same metrics.

Overall, our approach delivers transferable ECG features across models, label settings, and datasets. Its consistent performance suggests that the proposed pretraining strategy captures both local waveform deviations and lead-level associations more effectively than prior methods.

## E. Generated Embedding Visualization Analysis

Fig. 4 compares the structure of raw ECG waveforms and the embeddings learned by different methods. Fig. 4(a) shows the raw signals, which display different waveforms among normal (Norm), myocardial infarction (MI), ST-T changes (STTC), hypertrophy (HYP), and conduction disorder (CD) classes, without any anomaly indicators. DINOv2 in Fig. 4(b) and TimesFM in Fig. 4(c) partially separate some classes but exhibit considerable inter-class mixing. Chronos in Fig. 4(d) and MOIRAI in Fig. 4(e) produce fragmented and inconsistent groupings, reflecting weak discriminative capability.

In contrast, our deviation localization embedding in Fig. 4(f) forms compact and class-consistent clusters that clearly distinguish pathological patterns from normal ones. The association reconstruction embedding in Fig. 4. (g) further improves class separation by integrating deviation-guided cross-lead contextual information, yielding more compact and distinct clusters. Finally, the NormECG codebook projection outlines the reference manifold of normal segments used to compute per-lead deviations, as detailed in the supplementary material. These qualitative patterns are consistent with the quantitative improvements reported in Table I for frozen-feature evaluation on all datasets and Table II for generalizability comparison on external datasets, confirming that our learned representations are both more robust and more discriminative.

## F. Embedding Space Interpretability Analysis

Furthermore, we analyze the interpretability and crossdataset consistency of the learned DLE and ARE embeddings. DLE reflects lead-wise deviations from normal ECG morphology, whereas ARE captures deviation-guided cross-lead associations in the latent representation space. Fig. 5 visualizes their joint distributions across different datasets to examine whether the learned embeddings form consistent and separable structures for normal and abnormal ECG categories.

DLE Embedding. DLE encodes the magnitude of deviation of each lead from the NormECG codebook. In the projected space, samples with pronounced cardiovascular abnormalities, such as ventricular ectopy and atrial fibrillation, form tight and well-defined clusters. In contrast, cases with subtle deviations, such as mild ST-segment elevation, appear more dispersed with overlapping boundaries. This pattern indicates that DLE alone provides limited discriminative power for weaker anomalies.

TABLE III  
CLASSIFICATION PERFORMANCE OF DIFFERENT DOWNSTREAM CLASSIFIERS USING BASELINE AND DLE&ARE AS ECG REPRESENTATIONS ON EXTERNAL CLINICAL DATASETS. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td>Classifier</td><td>Representation</td><td>Accuracy</td><td>F1 Score</td><td>Precision</td><td>Recall</td><td>AUC</td></tr><tr><td>CNN</td><td>Baseline</td><td> $0 . 2 2 1 \pm 0 . 0 1 1$ </td><td> $0 . 3 5 6 \pm 0 . 0 1 2$ </td><td> $0 . 4 4 9 \pm 0 . 0 1 4$ </td><td> $0 . 3 0 3 \pm 0 . 0 1 8$ </td><td> $0 . 5 7 3 \pm 0 . 0 1 5$ </td></tr><tr><td></td><td>DLE&amp;ARE</td><td> ${ \bf 0 . 2 4 8 \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 0 . 3 9 5 \pm 0 . 0 1 6 }$ </td><td> ${ \bf 0 . 4 6 6 \pm 0 . 0 0 9 }$ </td><td> ${ \bf 0 . 3 4 8 \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 6 0 2 \pm 0 . 0 1 1 }$ </td></tr><tr><td>Transformer</td><td>Baseline</td><td> $0 . 2 4 2 \pm 0 . 0 2 4$ </td><td> $0 . 3 5 2 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 5 3 2 \pm 0 . 0 0 9 }$ </td><td> $0 . 2 6 9 \pm 0 . 0 2 6$ </td><td> $0 . 5 7 4 \pm 0 . 0 1 1$ </td></tr><tr><td></td><td>DLE&amp;ARE</td><td> ${ \bf 0 . 2 7 6 \pm 0 . 0 2 7 }$ </td><td> ${ \bf 0 . 4 0 1 \pm 0 . 0 2 2 }$ </td><td> $0 . 5 1 2 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 3 4 1 \pm 0 . 0 3 0 }$ </td><td> $\mathbf { 0 . 5 8 8 \pm 0 . 0 0 7 }$ </td></tr><tr><td>Mamba</td><td>Baseline</td><td> $0 . 2 4 1 \pm 0 . 0 1 7$ </td><td> $0 . 3 8 6 \pm 0 . 0 2 2$ </td><td> $0 . 3 9 9 \pm 0 . 0 2 1$ </td><td> $0 . 3 7 7 \pm 0 . 0 2 5$ </td><td> $0 . 4 9 9 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td>DLE&amp;ARE</td><td> ${ \bf 0 . 3 4 8 \pm 0 . 0 2 9 }$ </td><td> ${ \bf 0 . 4 2 9 \pm 0 . 0 2 8 }$ </td><td> ${ \bf 0 . 5 2 4 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 3 7 7 \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 0 . 5 4 3 \pm 0 . 0 0 9 }$ </td></tr></table>

TABLE IV

ABLATION STUDY ON CPSC2020 AND WGEX EXTERNAL VALIDATION.SSMB, SYMMASK, RPF, DLE, DMB, AND ARE DENOTE THESELF-SUPERVISED 5-LAYER SCALESMAMBA2 PRETRAINING BACKBONE,SYMMETRIC MASKING STRATEGY, WEIGHTED ROTATIONAL PARTIALFREEZING, DEVIATION LOCALIZATION EMBEDDING, DEVIATIONMAMBA2 BLOCK, AND ASSOCIATION RECONSTRUCTION EMBEDDING,RESPECTIVELY. A CHECKMARK INDICATES INCLUSION OF THECOMPONENT; A BLANK INDICATES OMISSION.
<table><tr><td>SSMb SymMask RPF DLE DMb ARE</td><td></td><td></td><td></td><td></td><td></td><td colspan="3">CPSC2020</td><td colspan="3">WGEX</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>B-Acc</td><td>F1</td><td>AUC</td><td>B-Acc</td><td>F1</td><td>AUC</td></tr><tr><td></td><td></td><td></td><td>√</td><td>√</td><td>√</td><td>0.289</td><td>0.295</td><td>0.552</td><td>0.270</td><td>0.285</td><td>0.557</td></tr><tr><td>√</td><td></td><td></td><td>√</td><td>√</td><td>V</td><td>0.397</td><td>0.403</td><td>0.659</td><td>0.337</td><td>0.390</td><td>0.663</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>5</td><td>L</td><td>0.466</td><td>0.470</td><td>0.726</td><td>0.464</td><td></td><td>0.4560.730</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>0.454</td><td>0.458</td><td>30.712</td><td>0.432</td><td></td><td>0.4520.716</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>0.473</td><td>0.475</td><td>0.729</td><td>0.449</td><td></td><td>0.458 0.732</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>0.500</td><td></td><td>0.501 0.755</td><td>0.476</td><td></td><td>0.462 0.748</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.538</td><td></td><td>0.536 0.763</td><td>0.513</td><td></td><td>0.471 0.787</td></tr></table>

ARE Embedding. ARE captures deviation-guided crosslead associations through attention-based reconstruction in the latent representation space. Compared with DLE, ARE produces more compact intra-class distributions and clearer inter-class separation across all datasets. In particular, class clusters in ML12L and CPSC2020 are more clearly delineated, indicating that ARE improves the discriminative structure of the learned representations.

Cross-Dataset Consistency and Classification Insight. Across all four datasets, the embeddings separate normal and abnormal cases and distinguish among major pathological classes. This consistency demonstrates the model’s ability to generalize deviation detection and localization across diverse clinical sources. Moreover, cluster tightness correlates with downstream classification performance: datasets with more cohesive and well-separated embeddings tend to yield higher detection accuracy. These visualizations thus validate the discriminative quality of our features and provide practical guidance for architecture design, including attention mechanisms and hyperparameter tuning.

## G. Analysis of DLE-Guided Lead Associations

To analyze the role of DLE in ARE, we visualize the sourceto-target lead associations used during cross-lead reconstruction. Fig. 6 compares a correlation-only reference map with the DLE-guided association map produced by ARE. Rows correspond to target leads to be reconstructed, and columns correspond to source leads used as reconstruction context. Each entry represents the association strength from a source lead to a target lead. Diagonal entries are masked because selflead dependencies are not considered cross-lead associations.

The correlation-only reference reflects the pairwise lead coupling estimated directly from the input ECG signals. Its association pattern is relatively diffuse, indicating that multiple limb and precordial leads exhibit non-negligible correlations. After introducing DLE/SFM guidance, the ARE association map becomes more selective, with stronger weights concentrated on a smaller subset of source leads. This suggests that ARE does not merely follow the raw correlation structure of the input signals. Instead, DLE-derived cues reweight the source-to-target association pattern used for latent cross-lead reconstruction.

The guidance map further illustrates the change induced by DLE/SFM guidance. Positive regions indicate enhanced source-to-target associations after guidance, whereas negative regions indicate suppressed associations. The observed pattern shows that DLE/SFM guidance selectively redistributes association weights instead of uniformly increasing all lead relations. This supports the interpretation that DLE provides lead-wise cues for ARE, enabling cue-guided cross-lead consistency modeling in the latent representation space.

## H. Observational Study

We further conducted an observational study on real-world clinical ECG data from an external clinical cohort, without intervening in patient care. This cohort contains population and demographic distributions different from those of the public benchmark datasets, providing a more challenging setting for evaluating representation generalizability. As shown in Table III, DLE&ARE representations improve most metrics across CNN, Transformer, and Mamba classifiers compared with baseline representations. The gains are particularly consistent in accuracy, F1 score, and AUC, indicating that the learned deviation-association representations provide additional transferable information under real-world distribution shifts. Although the absolute performance remains limited due to the heterogeneity of the observational cohort, the consistent relative improvements support the robustness of the proposed feature extraction framework.

![](images/3fb37e38c9e441311332c0a76347ccb3302899117f1138d0d97f49717d1a7d20.jpg)  
Fig. 6. DLE-guided lead association analysis in ARE. Rows and columns denote target and source leads, respectively. The three columns show the correlationonly reference, DLE-guided ARE association, and guidance-induced change. Positive and negative values indicate enhanced and suppressed source-to-target associations after DLE/SFM guidance; gray diagonal entries are excluded from cross-lead interpretation.

## I. Ablation Study

Table IV summarizes the performance impact of individual components in our framework.

Pretraining phase. Beginning with a random masking strategy and a Transformer backbone, the baseline model achieved B-Acc = 0.289 on CPSC2020 and 0.270 on WGEX. Replacing the Transformer backbone with SSMb under the same hyperparameter setting improved B-Acc to 0.397 and 0.337, respectively, demonstrating the benefits of hierarchical temporal modeling. Adding self-supervised symmetric masking further boosted B-Acc to 0.466 on CPSC2020 and 0.464 on WGEX, while increasing AUROC by over 0.07 on both datasets, highlighting the value of reconstruction-based pretraining. Incorporating weighted rotational partial freezing without DLE and DMb yielded B-Acc = 0.454 on CPSC2020, confirming that selective parameter updates contribute to training stability.

Component contributions. The variant combining SSMb, SymMask, RPF, DLE, and DMb without ARE achieved B-$\mathrm { A c c } = 0 . 4 7 3$ on CPSC2020 and 0.449 on WGEX, showing cumulative performance gains from these modules. Removing DMb while adding ARE raised B-Acc to 0.500 on CPSC2020 and 0.476 on WGEX, underscoring the role of ARE in interlead integration. The full model, incorporating all components, achieved the best results: B-Acc = 0.538, F1 = 0.536, and $\mathrm { A U R O C } = 0 . 7 6 3$ on CPSC2020; and B- $\cdot \mathrm { A c c } = 0 . 5 1 3 , \mathrm { F 1 } =$ 0.471, and $\mathrm { A U R O C } = 0 . 7 8 7$ on WGEX. These results confirm the complementary contributions of each module and the overall synergy of the full architecture.

## V. CONCLUSION

In this study, we proposed a new ECG feature extraction framework that integrates multi-lead association embedding and deviation localization embedding. Our method effectively captures deviations from normal ECG morphology while integrating complementary cross-lead contextual information through association reconstruction. Through extensive experiments across diverse classifiers and datasets, we validated the reliability and generalizability of the framework. Our future work will focus on integrating SOTA ECG-based CVD diagnosis models to further enhance performance and provide more precise, dependable clinical decision support tools.

## REFERENCES

[1] Q. Xiao, K. Lee, S. A. Mokhtar, I. Ismail, A. L. b. M. Pauzi, Q. Zhang, and P. Y. Lim, “Deep learning-based ECG arrhythmia classification: A systematic review,” Applied Sciences, vol. 13, no. 8, p. 4964, 2023.

[2] S. S. Al-Zaiti, C. Martin-Gill, J. K. Zegre-Hemsey, Z. Bouzid, Z. Fara-\` mand, M. O. Alrawashdeh, R. E. Gregg, S. Helman, N. T. Riek, K. Kraevsky-Phillips et al., “Machine learning for ECG diagnosis and risk stratification of occlusion myocardial infarction,” Nature Medicine, vol. 29, no. 7, pp. 1804–1813, 2023.

[3] E. Silvetti, O. Lanza, F. Romeo, A. Martino, E. Fedele, C. Lanzillo, C. Crescenzi, F. Fanisio, and L. Calo, “The pivotal role of ECG in\` cardiomyopathies,” Frontiers in Cardiovascular Medicine, vol. 10, p. 1178163, 2023.

[4] M. A. Muzammil, S. Javid, A. K. Afridi, R. Siddineni, M. Shahabi, M. Haseeb, F. Fariha, S. Kumar, S. Zaveri, and A. J. Nashwan, “Artificial intelligence-enhanced electrocardiography for accurate diagnosis and management of cardiovascular diseases,” Journal of Electrocardiology, 2024.

[5] N. N. Arslan, D. Ozdemir, and H. Temurtas, “ECG heartbeats classification with dilated convolutional autoencoder,” Signal, Image and Video Processing, vol. 18, no. 1, pp. 417–426, 2024.

[6] R. Avanzato and F. Beritelli, “Automatic ECG diagnosis using convolutional neural network,” Electronics, vol. 9, no. 6, p. 951, 2020.

[7] T. N. Nguyen and T. H. Nguyen, “Deep learning framework with ECG feature-based kernels for heart disease classification,” Elektronika ir Elektrotechnika, vol. 27, no. 1, pp. 48–59, 2021.

[8] A. F. Gund¨ uz and M. F. Talu, “Atrial fibrillation classification and¨ detection from ECG recordings,” Biomedical Signal Processing and Control, vol. 82, p. 104531, 2023.

[9] Y. Shiraishi, S. Goto, N. Niimi, Y. Katsumata, A. Goda, M. Takei, M. Saji, M. Sano, K. Fukuda, T. Kohno et al., “Improved prediction of sudden cardiac death in patients with heart failure through digital processing of electrocardiography,” Europace, vol. 25, no. 3, pp. 922– 930, 2023.

[10] J. W. Hughes, J. Tooley, J. Torres Soto, A. Ostropolets, T. Poterucha, M. K. Christensen, N. Yuan, B. Ehlert, D. Kaur, G. Kang et al., “A deep learning-based electrocardiogram risk score for long term cardiovascular death and disease,” NPJ Digital Medicine, vol. 6, no. 1, p. 169, 2023.

[11] B. Zhou, S. Liu, B. Hooi, X. Cheng, and J. Ye, “BeatGAN: Anomalous rhythm detection using adversarially generated time series,” in Proc. Int. Joint Conf. Artificial Intelligence, 2019, pp. 4433–4439.

[12] N. Wang, P. Feng, Z. Ge, Y. Zhou, B. Zhou, and Z. Wang, “Adversarial spatiotemporal contrastive learning for electrocardiogram signals,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 10, pp. 13 845–13 859, 2024.

[13] A. Jiang, C. Huang, Q. Cao, Y. Xu, Z. Zeng, K. Chen, Y. Zhang, and Y. Wang, “Self-supervised anomaly detection pretraining enhances longtail ECG diagnosis,” arXiv preprint arXiv:2408.17154, 2024.

[14] K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang, “ECG-FM: An open electrocardiogram foundation model,” JAMIA Open, vol. 8, no. 5, p. ooaf122, 2025.

[15] J. Li, A. D. Aguirre, V. M. Junior, J. Jin, C. Liu, L. Zhong, C. Sun, G. Clifford, B. Westover, and S. Hong, “An electrocardiogram foundation model built on over 10 million recordings,” NEJM AI, vol. 2, no. 7, Jul. 2025, accepted Mar. 28, 2025; revised Mar. 20, 2025.

[16] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023, version 1.

[17] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” arXiv preprint arXiv:2310.10688, 2023.

[18] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” arXiv preprint arXiv:2402.02592, 2024.

[19] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” arXiv preprint arXiv:2403.07815, 2024, version 3.

[20] R. Kavya et al., “Feature extraction technique for robust and fast visual tracking: A typical review,” International Journal of Emerging Engineering Research and Technology, vol. 3, no. 1, pp. 98–104, 2015.

[21] E. Alpaydin, Introduction to Machine Learning, 4th ed. MIT Press, 2020.

[22] R. M. Gray and J. W. Goodman, Fourier Transforms: An Introduction for Engineers. Springer Science & Business Media, 2012, vol. 322.

[23] A. Gholami, “Sparse time–frequency decomposition and some applications,” IEEE Transactions on Geoscience and Remote Sensing, vol. 51, no. 6, pp. 3598–3604, 2013.

[24] T. Kurita, “Principal component analysis (PCA),” in Computer Vision: A Reference Guide. Springer, 2021, pp. 1013–1016.

[25] L. Sadouk, “CNN approaches for time series classification,” in Time Series Analysis - Data, Methods, and Applications, C.-K. Ngan, Ed. IntechOpen, 2019, ch. 4.

[26] X. Jin, X. Yu, X. Wang, Y. Bai, T. Su, and J. Kong, “Prediction for time series with CNN and LSTM,” in Proc. 11th Int. Conf. Modelling, Identification and Control, 2020, pp. 631–641.

[27] J. Wang, X. Li, J. Li, Q. Sun, and H. Wang, “NGCU: A new RNN model for time-series data prediction,” Big Data Research, vol. 27, p. 100296, 2022.

[28] R. Khaldi, A. El Afia, R. Chiheb, and S. Tabik, “What is the best RNNcell structure to forecast each time series behavior?” Expert Systems with Applications, vol. 215, p. 119140, 2023.

[29] J. Lai, H. Tan, J. Wang, L. Ji, J. Guo, B. Han, Y. Shi, Q. Feng, and W. Yang, “Practical intelligent diagnostic algorithm for wearable 12- lead ECG via self-supervised learning on large-scale dataset,” Nature Communications, vol. 14, no. 1, p. 3741, 2023.

[30] D. Kiyasseh, T. Zhu, and D. A. Clifton, “CLOCS: Contrastive learning of cardiac signals across space, time, and patients,” in Proc. Int. Conf. Machine Learning, 2021, pp. 5606–5615.

[31] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[32] T. Mehari and N. Strodthoff, “Self-supervised representation learning from 12-lead ECG data,” Computers in Biology and Medicine, vol. 141, p. 105114, 2022.

[33] H. Zhang, W. Liu, J. Shi, S. Chang, H. Wang, J. He, and Q. Huang, “MaeFE: Masked autoencoders family of electrocardiogram for selfsupervised pretraining and transfer learning,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–15, 2023.

[34] Y. Na, M. Park, Y. Tae, and S. Joo, “Guiding masked representation learning to capture spatio-temporal relationship of electrocardiogram,” arXiv preprint arXiv:2402.09450, 2024.

[35] L. Jing and Y. Tian, “Self-supervised visual feature learning with deep neural networks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, pp. 4037–4058, 2021.

[36] T. Chen, S. Kornblith, M. Norouzi, and G. E. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv preprint arXiv:2002.05709, 2020.

[37] K. He, H. Fan, Y. Wu, S. Xie, and R. B. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2020, pp. 9726–9735.

[38] J.-B. Grill, F. Strub, F. Altche, C. Tallec, P. H. Richemond,´ E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar,<sup>´</sup> B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap your own latent: A new approach to self-supervised learning,” arXiv preprint arXiv:2006.07733, 2020.

[39] H. Liu, Z. Zhao, and Q. She, “Self-supervised ECG pre-training,” Biomedical Signal Processing and Control, vol. 70, p. 103010, 2021.

[40] D. Gedon, A. H. Ribeiro, N. Wahlstrom, and T. B. Sch¨ on, “First¨ steps towards self-supervised pretraining of the 12-lead ECG,” in Proc. Computing in Cardiology, vol. 48, 2021, pp. 1–4.

[41] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 10 041–10 071.

[42] B. Gow, T. Pollard, L. A. Nathanson, A. Johnson, B. Moody, C. Fernandes, N. Greenbaum, J. W. Waks, P. Eslami, T. Carbonati, A. Chaudhari, E. Herbst, D. Moukheiber, S. Berkowitz, R. Mark, and S. Horng, “MIMIC-IV-ECG: Diagnostic electrocardiogram matched subset,” PhysioNet, 2023, version 1.0.

[43] P. Wagner, N. Strodthoff, R. D. Bousseljot et al., “PTB-XL, a large publicly available electrocardiography dataset,” Scientific Data, vol. 7, no. 1, p. 154, 2020.

[44] H. Liu, D. Chen, D. Chen, X. Zhang, H. Li, L. Bian, M. Shu, and Y. Wang, “A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements,” Scientific Data, vol. 9, p. 272, 2022.

[45] E. A. Perez Alday, A. Gu, A. J. Shah, C. Robichaux, A.-K. I. Wong, C. Liu, F. Liu, A. Bahrami Rad, A. Elola, S. Seyedi, Q. Li, A. Sharma, G. D. Clifford, and M. A. Reyna, “Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020,” Physiological Measurement, vol. 41, no. 12, p. 124003, 2020.

[46] J. Zheng, H. Guo, and H. Chu, “A large scale 12-lead electrocardiogram database for arrhythmia study,” PhysioNet, 2022, version 1.0.0.

[47] C. Liu, Z. Wan, C. Ouyang, A. Shah, W. Bai, and R. Arcucci, “Zeroshot ECG classification with multimodal learning and test-time clinical knowledge enhancement,” arXiv preprint arXiv:2403.06659, 2024.

# Supplementary Material for Deviation-Aware Pretraining for Cardiovascular Disease Detection from Electrocardiograms

Siyuan Yan, Haibo Zheng, Feng Lu, Wei Li

## APPENDIX A

TWO-DIMENSIONAL PROJECTION OF DLE AND ARE EMBEDDINGS

Fig. 1 shows the following observations:

• PTB-XL: DLE in Fig. 1(a) forms clear clusters for Myocardial Infarction (MI) and Normal cases but overlaps ST–T Changes (STTC) and Conduction Disorders (CD), reflecting similarities in local waveform deviations. ARE in Fig. 1(e) tightens class boundaries by diffusing subtle MI and CD anomalies across leads, improving class separability.

• ML12L: DLE in Fig. 1(b) effectively isolates Hypertrophy (HYP) and Repolarization (Repolari) classes via localized amplitude deviations, but mixes multi-label groups like Rhythms and Conduction. ARE in Fig. 1(f) resolves these by capturing cross-lead associations, yielding more compact, distinct clusters for complex multilabel categories.

• CPSC2020: In DLE, Fig. 1(c), atrial fibrillation (AF) and normal samples separate well, yet PVC and PAC remain entangled. ARE in Fig. 1(g) amplifies inter-lead PVC patterns, disentangling PAC and reducing false overlaps with Normal.

• WGEX: DLE in Fig. 1(d) highlights PVC and AF anomalies strongly but under-separates RBBB and LBBB, due to subtle morphological differences. ARE in Fig. 1(h) balances this by integrating minor deviations across leads, improving discrimination among bundlebranch blocks.

These projections confirm that DLE excels at localizing beat-level deviations, while ARE captures cross-lead anomaly patterns. Their combination underpins the superior classification performance observed in our experiments.

## APPENDIX B

## DATASETS AND EXPERIMENTAL SETUP

## A. Datasets

Our study leverages two large-scale pretraining datasets and several public benchmark datasets for validation and evaluation.

1) Pretraining Datasets: The pretraining dataset was constructed by combining two large sources of 12-lead ECG recordings, each sampled at 500 Hz with a duration of 10 s:

• MIMIC-IV-ECG [5]: We included all available recordings from this version, excluding only samples with missing waveform data. This contributed approximately 800,000 recordings.

• Private Clinical Dataset: An additional set of approximately 118,000 recordings was sourced from a clinical partner under IRB approval. This dataset was collected in a clinical environment with equipment comparable to that used for the public datasets. It represents a general inpatient population and is not enriched for any specific cardiac condition, containing a mix of healthy and abnormal, but not finely categorized, samples.

In total, the pretraining corpus contains approximately 918,000 recordings.

2) External Evaluation Datasets: To assess generalization and transfer learning performance, we utilized four distinct public datasets as held-out cohorts:

• PTB-XL [1]: A large-scale clinical dataset from a general population, widely used for benchmarking general ECG interpretation algorithms.

• ML12L [2]: A large-scale multi-label 12-lead ECG dataset with standardized diagnostic statements, used to evaluate multi-label cardiovascular disease classification.

• CPSC2020 [3]: A dataset primarily targeted at the detection of atrial fibrillation and other rhythm disorders.

• WGEX: Our evaluation set drawn from A large scale 12-lead electrocardiogram database for arrhythmia study [4], referred to as WGEX in this study.

3) Observational Datasets: To evaluate the model extraction performance in real-world scenarios, we conducted observational experiments on real data from a clinical partner in another region.

## B. Data Preprocessing and Model Input

A minimalist preprocessing approach was intentionally adopted to encourage the model to learn robust features directly from near-raw signals. The same procedure was applied to all datasets.

• Signal Integrity Check: For the MIMIC dataset, only samples with non-empty ECG waveform data were retained. For the pretraining set, basic outlier detection was performed to remove samples exhibiting extreme artifacts likely due to lead disconnection or severe acquisition errors.

• No Filtering or Normalization: The raw waveform data were used directly without applying any signal filters, such as band-pass filtering, or amplitude normalization techniques, such as Z-score normalization. Each 10 s recording was treated as a sequence of 5,000 time points per lead.

TABLE I  
OVERVIEW OF THE VALIDATION DATASETS AND OBSERVATIONAL DATA.  
![](images/f9abc69f162c6c818f2a16f4cf695a4eb305cb8cc05e15322f479b8d56d002ca.jpg)

![](images/6ceecfe1e99837ee50eaede4a6e9bdb4760892fb6ef3a75117ef33a96d8cbd1d.jpg)

![](images/02813c3427bf7a07fb6890527050d86b9a83012fb1faa7c61eabba4b655cfd53.jpg)

![](images/032cf77c1b5987c253ac319ca5a87d47de24fc6cb2a4fd056bb409d29c866b5a.jpg)  
Fig. 1. Detailed two-dimensional projection of PTB-XL, ML12L, CPSC2020, and WGEX datasets under the DLE (Deviation Localization Embedding) and ARE (Association Reconstruction Embedding) feature embeddings. The x-axis represents the ARE mapping, and the y-axis represents the DLE mapping.

<table><tr><td>Dataset</td><td></td><td></td><td>#Samples #Classes Sampling Rate</td><td>Classes</td></tr><tr><td>PTB-XL [1]</td><td>21 873</td><td>5</td><td>500 Hz</td><td>Normal, MI, STTC, CD, HYP</td></tr><tr><td>ML12L [2]</td><td>25 770</td><td>8</td><td>500 Hz</td><td>Norm, Rhythms, Axis, Conduction, Extrasy, HYP, Ischemia, Repolari</td></tr><tr><td>CPSC2020 [3]</td><td>10330</td><td>9</td><td>500 Hz</td><td>Normal, AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE</td></tr><tr><td>WGEX [4]</td><td>12600</td><td>7</td><td>500 Hz</td><td>Normal, AF, PVC, PAC, LBBB, RBBB, STTC</td></tr><tr><td>Observational Data</td><td>5239</td><td>7</td><td>500 Hz</td><td>I21, I44, I45, I46, I47, I48, I49</td></tr></table>

• No Data Augmentation: No data augmentation techniques were applied during the pretraining phase.

## C. Experimental Design and Evaluation Strategy

Our experimental protocol is designed to first pretrain a powerful feature extractor in a self-supervised manner and then rigorously evaluate its feature quality and downstream performance.

1) Pretraining Phase: The entire pretraining corpus, including MIMIC-IV-ECG and private data, was used for selfsupervised model training. The dataset was shuffled at the recording level. During training, we partitioned this set dynamically, using 80% of the samples for training updates and 20% for validation in each epoch to monitor training progress and guide hyperparameter adjustments. It is important to note that this split was based on recording IDs, as the primary goal of our self-supervised pretraining is to learn universal waveform representations rather than patient-specific characteristics.

2) Evaluation Phase: The external datasets, including PTB-XL, ML12L, CPSC2020, and WGEX, were used exclusively for evaluation and were never seen by the model during pretraining. We employed two distinct evaluation strategies:

1) Linear Evaluation for Feature Quality Assessment: To assess the intrinsic quality of the learned representations, we performed linear evaluation on the PTB-

TABLE II  
CLASS-WISE SAMPLE COUNTS IN EACH EXTERNAL VALIDATION DATASET.
<table><tr><td>Class</td><td>PTB-XL</td><td>ML12L</td><td>CPSC2020</td><td>WGEX</td></tr><tr><td>Normal</td><td>5 038</td><td>13 905</td><td>3855</td><td>5000</td></tr><tr><td>MI</td><td>4508</td><td></td><td></td><td></td></tr><tr><td>STTC</td><td>5367</td><td></td><td></td><td>700</td></tr><tr><td>CD</td><td>4381</td><td></td><td></td><td></td></tr><tr><td>HYP</td><td>1079</td><td>229</td><td></td><td></td></tr><tr><td>Rhythms</td><td></td><td>5430</td><td></td><td></td></tr><tr><td>Axis</td><td></td><td>612</td><td></td><td></td></tr><tr><td>Conduction</td><td></td><td>2565</td><td></td><td></td></tr><tr><td>Extrasy</td><td></td><td>1689</td><td></td><td></td></tr><tr><td>Ischemia</td><td></td><td>260</td><td></td><td></td></tr><tr><td>Repolari</td><td></td><td>5125</td><td></td><td></td></tr><tr><td>AF</td><td></td><td></td><td>758</td><td>2000</td></tr><tr><td>I-AVB</td><td></td><td></td><td>328</td><td></td></tr><tr><td>LBBB</td><td></td><td></td><td>222</td><td>1000</td></tr><tr><td>RBBB</td><td></td><td></td><td>320</td><td>900</td></tr><tr><td>PAC</td><td></td><td></td><td>379</td><td>1200</td></tr><tr><td>PVC</td><td></td><td></td><td>640</td><td>1800</td></tr><tr><td>STD</td><td></td><td></td><td>154</td><td></td></tr><tr><td>STE</td><td>_</td><td></td><td>221</td><td>一</td></tr></table>

TABLE III  
DISEASE ABBREVIATIONS AND FULL NAMES IN THE PTB-XL DATASET.
<table><tr><td>Abbreviation</td><td>Full Name</td></tr><tr><td>Norm</td><td>Normal</td></tr><tr><td>CD</td><td>Conduction Disturbance</td></tr><tr><td>HYP</td><td>Hypertrophy</td></tr><tr><td>STTC</td><td>ST/T Change</td></tr><tr><td>MI</td><td>Myocardial Infarction</td></tr></table>

TABLE IV  
DISEASE ABBREVIATIONS AND FULL NAMES IN THE CPSC2020 DATASET.
<table><tr><td>Abbreviation</td><td>Full Name</td></tr><tr><td>Normal</td><td>Normal</td></tr><tr><td>AF</td><td>Atrial Fibrillation</td></tr><tr><td>I-AVB</td><td>First-degree Atrioventricular Block</td></tr><tr><td>LBBB</td><td>Left Bundle Branch Block</td></tr><tr><td>RBBB</td><td>Right Bundle Branch Block</td></tr><tr><td>PAC</td><td>Premature Atrial Contraction</td></tr><tr><td>PVC</td><td>Premature Ventricular Contraction</td></tr><tr><td>STD</td><td>ST-segment Depression</td></tr><tr><td>STE</td><td>ST-segment Elevation</td></tr></table>

XL and ML12L datasets. The pretrained ScalesMamba2 model was used as a frozen feature extractor. A simple linear classifier was then trained on top of these extracted features to perform the downstream classification task. The performance of this linear head serves as a direct measure of the features’ separability and utility.

2) Comparison With State-of-the-Art Models on Downstream Tasks: To benchmark against other unsupervised or weakly supervised methods, we evaluated our model on the CPSC2020 and WGEX datasets. Similar to linear probing, we extracted features using our frozen pretrained model and attached a lightweight classification head. The end-to-end performance was then compared against published results from other state-of-the-art models on these specific benchmarks.

TABLE V  
DISEASE ABBREVIATIONS AND FULL NAMES IN THE WGEX DATASET.
<table><tr><td>Abbreviation</td><td>Full Name</td></tr><tr><td>Normal</td><td>Normal</td></tr><tr><td>AF</td><td>Atrial Fibrillation</td></tr><tr><td>PVC</td><td>Premature Ventricular Contraction</td></tr><tr><td>PAC</td><td>Premature Atrial Contraction</td></tr><tr><td>LBBB</td><td>Left Bundle Branch Block</td></tr><tr><td>RBBB</td><td>Right Bundle Branch Block</td></tr><tr><td>STTC</td><td>ST/T Change</td></tr></table>

## APPENDIX C

## PRIOR KNOWLEDGE ABOUT SSMS AND DYNAMIC SPARSE MATRICES

## A. State-Space Models and Mambas

State-space models have consistently shown strong predictive power for time-series data [6], [7]. The recent development of Mamba and its variations has addressed the challenge of incorporating distant contexts as sequence length increases by integrating the attention mechanism of Transformers into a more streamlined state-space model [8], [9], [10], [11], [12]. ECG data, as a classic example of long time-series data, stands to benefit significantly from Mamba2. Furthermore, the deviation localization extracted from ECG signals, which includes a time dimension, can also be classified as long timeseries data. In a self-supervised task, Mamba2 can be used to extract the deviation localization embedding of ECG signals and adapt it dynamically during training. Generating this embedding through Mamba2, as opposed to direct numerical calculation, yields a thorough understanding and aids in the overall feature extraction of ECG signals.

## B. Dynamic Sparse Matrices and Feature Fusion for ECG

In the process of association reconstruction for each lead, it is essential to incorporate dynamic sparse matrix and feature fusion technology into the multi-lead association reconstruction embedding process. Studies have demonstrated the effectiveness of sparse representations in capturing ECG features and supporting heart rate and arrhythmia classification [13], [14], [15]. Concurrently, feature fusion techniques can amalgamate signals from multiple leads to extract comprehensive features and precisely reconstruct ECG signals [16], [17].

## APPENDIX D

## DETAILED MECHANISM FOR WEIGHTED ROTATIONAL PARTIAL FREEZING STRATEGY

Standard self-supervised training can sometimes lead to the model “over-optimizing” for the pretext task, e.g., masked signal reconstruction, at the expense of learning robust and generalizable features. To address this, we introduce a Weighted Rotational Partial Freezing (RPF) strategy. This advanced approach provides fine-grained control over the learning dynamics of different parts of the network, ensuring a balanced focus on both feature extraction and task-specific learning.

TABLE VI  
CARDIOVASCULAR DISEASE CATEGORIES AND SUBCATEGORIES IN THE ML12L DATASET.
<table><tr><td>Category</td><td>Subcategory</td></tr><tr><td colspan="2">Norm</td></tr><tr><td>Electric Axis of the Heart</td><td>Right-axis deviation</td></tr><tr><td></td><td>Left-axis deviation Sinus rhythm</td></tr><tr><td></td><td>Short PR interval</td></tr><tr><td></td><td>AV conduction ratio N:D</td></tr><tr><td></td><td>Prolonged PR interval</td></tr><tr><td></td><td>Second-degree AV block, Mobitz type I</td></tr><tr><td></td><td>Second-degree AV block, Mobitz type II</td></tr><tr><td></td><td>2:1 AV block</td></tr><tr><td>Conduction Abnormalities</td><td>AV block, varying conduction</td></tr><tr><td></td><td>AV block, advanced (high-grade)</td></tr><tr><td></td><td>AV block, complete (third-degree)</td></tr><tr><td></td><td>Left anterior fascicular block</td></tr><tr><td></td><td>Left posterior fascicular block</td></tr><tr><td></td><td>Left bundle-branch block</td></tr><tr><td></td><td>Incomplete right bundle-branch block</td></tr><tr><td></td><td>Right bundle-branch block</td></tr><tr><td></td><td>Ventricular preexcitation Junctional tachycardia</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Atrial premature complex(es) Atrial premature complexes, nonconducted</td></tr><tr><td>Extrasystolies</td><td>Junctional premature complex(es)</td></tr><tr><td></td><td>Junctional escape complex(es)</td></tr><tr><td></td><td>Ventricular premature complex(es)</td></tr><tr><td></td><td></td></tr><tr><td>Hypertrophies</td><td>Left atrial enlargement Left ventricular hypertrophy</td></tr><tr><td></td><td>Right ventricular hypertrophy</td></tr><tr><td></td><td>Atrial fibrillation</td></tr><tr><td></td><td>Atrial flutter</td></tr><tr><td></td><td>ST deviation</td></tr><tr><td></td><td>ST deviation with T-wave change</td></tr><tr><td></td><td>T-wave abnormality</td></tr><tr><td></td><td>Prolonged QT interval</td></tr><tr><td></td><td>TU fusion</td></tr><tr><td></td><td>Ventricular hypertrophy ST-T change</td></tr><tr><td>Repolarization Abnormalities</td><td>Low voltage</td></tr><tr><td></td><td>Early repolarization</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Anterior MI</td></tr><tr><td>Myocardial Infarction (MI)</td><td>Inferior MI</td></tr><tr><td></td><td>Anteroseptal MI</td></tr></table>

TABLE VII  
DISEASE CATEGORIES IN THE OBSERVATIONAL DATASET.
<table><tr><td>ICD</td><td>Full Name</td><td>Size</td></tr><tr><td>I21</td><td>Acute Myocardial Infarction</td><td>1,152</td></tr><tr><td>I44</td><td>Atrioventricular and left bundle block</td><td>450</td></tr><tr><td>I45</td><td>Other conduction disorders</td><td>207</td></tr><tr><td>I46</td><td>Cardiac arrest</td><td>44</td></tr><tr><td>I47</td><td>Paroxysmal tachycardia</td><td>700</td></tr><tr><td>I48</td><td>Atrial fibrillation and flutter</td><td>2,271</td></tr><tr><td>I49</td><td>Other cardiac arrhythmias</td><td>415</td></tr></table>

## A. Methodology of Weighted RPF

We partition the model’s full parameter vector θ into G disjoint blocks, $\theta = [ \theta _ { 1 } , \dots , \theta _ { G } ]$ . These blocks are conceptually grouped into two main components:

• The Feature Extractor (E): Comprising parameters $\{ \theta _ { g } \ | \ g \in \mathcal { T } \varepsilon \}$ from the deeper, foundational layers of our architecture, e.g., the encoder path of DMb/SSMb. These are intended to learn transferable representations.

• The Self-Supervision Head (H): Comprising parameters $\{ \theta _ { g } \mid g \in { \mathcal { T } } _ { \mathcal { H } } \}$ from the final layers responsible for the reconstruction task, e.g., the decoder and output layers.

Unlike a uniform cyclic schedule, weighted RPF selects a block g to unfreeze at each training step t according to a non-uniform probability distribution $\mathbf { p } = [ p _ { 1 } , \dots , p _ { G } ]$ , where $p _ { g } = \operatorname* { P r } ( g ( t ) = g )$ . The parameter update rule remains:

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \alpha ^ { ( t ) } M _ { g ( t ) } \odot \nabla _ { \boldsymbol { \theta } } \mathcal { L } \left( \boldsymbol { \theta } ^ { ( t ) } \right) ,\tag{1}
$$

where $M _ { g ( t ) }$ is the masking matrix that activates only block $g ( t )$

## B. Controlling Learning Dynamics via Update Probabilities

The core of weighted RPF is the deliberate, uneven allocation of optimization “budget.” We assign higher update probabilities to the feature extractor blocks and lower probabilities to the self-supervision head blocks. This is formalized by two principles:

1) Prioritizing Feature Learning: We set $p _ { i } > p _ { j }$ for any extractor block $\theta _ { i } \in \mathcal { E }$ and any head block $\theta _ { j } \in \mathcal { H }$ . This ensures that, on average, the feature extractor receives more frequent updates.

2) Capping Head Optimization: We introduce a hyperparameter $\gamma _ { \operatorname* { m a x } } \in ( 0 , 1 )$ to impose an upper bound on the total probability mass allocated to the self-supervision head:

$$
\sum _ { g \in \mathcal { Z } _ { \mathcal { H } } } p _ { g } \le \gamma _ { \operatorname* { m a x } } .\tag{2}
$$

For instance, setting $\gamma _ { \mathrm { m a x } } = 0 . 5 0$ guarantees that the reconstruction-specific parts of the model receive at most 25% of the total parameter updates.

## C. Architectural Justification and Benefits

This weighted strategy provides a principled approach to guiding the learning process with several key benefits:

• Guided Optimization Focus: By allocating more updates to the feature extractor E, W-RPF explicitly directs the model’s optimization effort toward learning deep, meaningful representations from the data, rather than superficial patterns sufficient only for the pretext task.

• Strong Regularization of the Task Head: Severely limiting the update frequency of the head H acts as a powerful form of regularization. It prevents the head from overfitting to the reconstruction task and forces the feature extractor to produce high-quality, informative features that make the reconstruction task easier for a “lazy” head.

• Mitigation of Gradient Interference: The fundamental benefit of modular updates is retained. By isolating the update of each block, W-RPF prevents large, clean gradients from the reconstruction head from interfering with the smaller, more nuanced gradients required to refine the deep feature extractor, leading to more stable training.

• Principled and Flexible Control: W-RPF transforms the training process from a simple uniform cycle into a controllable and targeted strategy. The probability vector p and the cap $\gamma _ { \mathrm { m a x } }$ become meaningful hyperparameters that allow a researcher to precisely balance the trade-off between pretext task performance and the quality of the learned features, adapting the strategy to different datasets and architectural depths.

## APPENDIX E

## APPLICATION OF SYMMETRIC MASKING TO SELF-SUPERVISED FEATURE LEARNING

We compare two masking strategies for self-supervised reconstruction: random masking, where each position is masked independently with probability $p ,$ and symmetric masking, where two complementary masks cover all positions.

## A. Loss Definitions

Let $\mathbf { x } \in \mathbb { R } ^ { T }$ be an ECG segment and $f ( \cdot )$ our reconstruction network. Under random masking $M \in \mathsf { \Gamma } \{ 0 , 1 \} ^ { T }$ with $M _ { t } \sim$ Bernoulli(p), the expected reconstruction loss is

$$
\mathcal { L } _ { \mathrm { r a n d } } = \mathbb { E } _ { M } \left[ \left\| M \odot \left( f \left( \left( \mathbf { 1 } - M \right) \odot \mathbf { x } \right) - \mathbf { x } \right) \right\| ^ { 2 } \right] .\tag{3}
$$

Under symmetric masking, we sample $M ^ { ( 1 ) } \sim$ Bernoulli(p) and set $M ^ { ( 2 ) } \ = \ \mathbf { 1 } - M ^ { ( 1 ) }$ . The combined loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s y m } } = \mathbb { E } _ { M ^ { ( 1 ) } } \Big [ \Big \| M ^ { ( 1 ) } \odot \Big ( f \Big ( M ^ { ( 2 ) } \odot \mathbf { x } \Big ) - \mathbf { x } \Big ) \Big \| ^ { 2 } } \\ { + \Big \| M ^ { ( 2 ) } \odot \Big ( f \Big ( M ^ { ( 1 ) } \odot \mathbf { x } \Big ) - \mathbf { x } \Big ) \Big \| ^ { 2 } \Big ] . } \end{array}\tag{4}
$$

## B. Unbiased Gradient Estimate

Denote the per-position gradient for coordinate t as $g _ { t } ( \mathbf { x } )$ Under random masking,

$$
\mathbb { E } \left[ M _ { t } g _ { t } ( \mathbf { x } ) \right] = p g _ { t } ( \mathbf { x } ) ,\tag{5}
$$

so the expected update only covers a fraction $p$ of the full gradient. In contrast, symmetric masking yields

$$
\mathbb { E } \left[ M _ { t } ^ { ( 1 ) } g _ { t } ^ { ( 2 ) } + M _ { t } ^ { ( 2 ) } g _ { t } ^ { ( 1 ) } \right] = p g _ { t } ^ { ( 2 ) } + ( 1 - p ) g _ { t } ^ { ( 1 ) } ,\tag{6}
$$

where $g _ { t } ^ { ( i ) }$ is the gradient from mask $M ^ { ( i ) }$ . If $f$ is sufficiently expressive, both gradients approximate the true gradient $g _ { t }$ , so

$$
\mathbb { E } \left[ \nabla _ { t } \mathcal { L } _ { \mathrm { s y m } } \right] \approx g _ { t } ( \mathbf x ) ,\tag{7}
$$

i.e., symmetric masking recovers an unbiased estimate of the full-signal gradient, unlike random masking, which scales it by $p .$

## C. Variance Reduction

Let $\sigma ^ { 2 }$ be the variance of the full gradient at coordinate t. For random masking,

$$
\mathrm { V a r } ( M _ { t } g _ { t } ) = p ( 1 - p ) g _ { t } ^ { 2 } < p \sigma ^ { 2 } .\tag{8}
$$

Under symmetric masking, the variance of the sum of two complementary terms is

$$
\begin{array} { r l } & { \operatorname { V a r } \Big ( M _ { t } ^ { ( 1 ) } g _ { t } ^ { ( 2 ) } + M _ { t } ^ { ( 2 ) } g _ { t } ^ { ( 1 ) } \Big ) } \\ & { = p ( 1 - p ) \left( g _ { t } ^ { ( 2 ) } - g _ { t } ^ { ( 1 ) } \right) ^ { 2 } \lesssim p ( 1 - p ) \sigma ^ { 2 } , } \end{array}\tag{9}
$$

which is at most half that of the random-mask case when $p \approx 0 . 5$ . Thus, symmetric masking provides a lower-variance gradient estimate.

## D. Full-Coverage Property

By construction, $M _ { t } ^ { ( 1 ) } + M _ { t } ^ { ( 2 ) } = 1$ , so every position is reconstructed exactly once per sample, ensuring no data region is systematically under-trained. Random masking leaves a fraction $1 - p$ of positions never seen in reconstruction for a given M.

Symmetric masking yields an unbiased, lower-variance gradient estimate that covers all positions every iteration, leading to more stable and efficient learning of ECG waveform morphology compared with standard random masking.

## APPENDIX F

## ARCHITECTURAL RATIONALE: MAMBA2’S SUPERIORITY FOR ECG ANOMALY DETECTION

The analysis of electrocardiogram (ECG) signals necessitates the detection of transient, often low-amplitude, anomalies within a long-duration, quasi-periodic context. An effective architecture must not only identify these localized events but also preserve their fidelity against dilution from the surrounding signal. Here, we provide a rationale for why the Mamba2 architecture, built upon the Selective State Space Model (Selective SSM), is structurally better suited for this task than Transformer, CNN, and traditional RNN models.

A. Mamba2 vs. Transformer: State Compression Over Global Averaging

The self-attention mechanism in Transformers computes a global, weighted average of the entire sequence. For an input signal $x _ { t } = s _ { t } + \delta \mathbf { 1 } _ { t = t _ { 0 } }$ , where $\delta$ is a weak, local anomaly, the output is:

$$
o _ { t } = \sum _ { k = 1 } ^ { T } \alpha _ { t , k } V _ { k } , \quad \mathrm { w h e r e } \quad \alpha _ { t , k } = \mathrm { s o f t m a x } \left( Q _ { t } ^ { \top } K _ { k } \right)\tag{10}
$$

This formulation poses a structural challenge: the anomaly $\delta$ must “compete” for attention against all other $T - 1$ time points. In a long and noisy ECG, this creates a risk of attention dilution, where the anomaly’s influence is smoothed out by the global averaging process, even if it receives a higher weight.

Mamba2 circumvents this by replacing global averaging with state compression. It sequentially processes the input, compressing the history into an evolving latent state $x _ { \tau }$ . The core of this process is the Selective SSM update rule:

$$
\begin{array} { r } { x _ { \tau } = A _ { \tau } ( h _ { \tau } ) \cdot x _ { \tau - 1 } + B _ { \tau } ( h _ { \tau } ) \cdot h _ { \tau } . } \end{array}\tag{11}
$$

Crucially, the state transition matrix $A _ { \tau }$ and input matrix $B _ { \tau }$ are functions of the current input $h _ { \tau }$ . This selective gating mechanism allows the model to:

1) Preserve Salient Inputs: Upon detecting an anomaly in $h _ { \tau _ { 0 } } ,$ the model can dynamically increase the norm of the input gate $B _ { \tau _ { 0 } } ,$ ensuring the anomaly is robustly injected into the state $x _ { \tau _ { 0 } }$ without global competition.

2) Memorize Critical Information: It can simultaneously adjust the memory gate $A _ { \tau _ { 0 } }$ to be close to identity, effectively “latching” the anomaly information in the state and preventing it from being overwritten by subsequent, less relevant signals.

B. Mamba2 vs. CNN: Dynamic, Content-Aware Filtering Over Static Kernels

A standard CNN applies a fixed, time-invariant kernel w to extract features:

$$
y _ { t } = \sum _ { k } w _ { k } \cdot x _ { t - k } .\tag{12}
$$

This static template matching is effective for regular, repeating patterns but struggles with the morphological diversity of

ECG anomalies. Mamba2’s mechanism, as described in (11), is inherently time-variant and content-aware. Its effective “filter” is dynamically configured by the input at each time step, allowing it to adaptively capture a wide variety of anomalous shapes. Furthermore, its recurrent nature inherently models long-range temporal dependencies, a task that requires deep CNN stacks or large dilated convolutions.

## C. Mamba2 vs. Traditional RNNs: State Expansion Over Information Bottleneck

While traditional RNNs, e.g., LSTM and GRU, also use a recurrent state, they suffer from an information bottleneck. All features across all input channels must be compressed into a single, fixed-size state vector $h _ { t } \in \mathbb { R } ^ { D }$ , leading to feature interference:

$$
h _ { t } = f ( h _ { t - 1 } , x _ { t } ) , \quad { \mathrm { w h e r e } } \quad \dim ( h _ { t } ) { \mathrm { ~ i s ~ f i x e d } } .\tag{13}
$$

Mamba2 overcomes this with state expansion. For each of its D input channels, or heads, it maintains an independent, higher-dimensional latent state $\boldsymbol { x } _ { \ u { \tau } } ~ \in ~ \mathbb { R } ^ { N }$ , where $N > 1$ is the state expansion factor. This expands the model’s total memory capacity to $\mathcal { O } ( D \times N )$ , allowing it to track multiple, distinct temporal features in parallel without them corrupting one another. This is critical for ECG analysis, where the model may need to simultaneously monitor baseline wander, heart rate, and QRS morphology. Additionally, Mamba2’s formulation allows for efficient, parallelizable training, $\mathcal O ( L )$ breaking the sequential computation bottleneck of traditional RNNs.

In summary, Mamba2’s architecture provides a powerful and well-suited inductive bias for detecting localized anomalies in long time-series data. By combining state compression to avoid signal dilution, selective gating to dynamically preserve important features, and state expansion to prevent information bottlenecks, it structurally addresses the key limitations of Transformer, CNN, and traditional RNN models for tasks such as ECG analysis.

## APPENDIX G ARCHITECTURE OF ANOMALY RECOGNITION WITH SCALESMAMBA2

## A. The ScalesMamba2 Layer: A Multi-Scale Selective SSM

At the core of our proposed architecture lies the Scales-Mamba2 layer, which integrates a Selective State Space Model (Selective SSM) within a multi-scale residual framework. The foundation is Mamba2’s SSM, which models a sequence $\mathbf { u } _ { t }$ through an evolving latent state $\mathbf { x } _ { t } \colon$

$$
\begin{array} { r l } & { \mathbf { h } _ { t } = W _ { h } \mathbf { u } _ { t } + \mathbf { b } _ { h } , } \\ & { \mathbf { x } _ { t } = \mathbf { A } _ { t } ( \mathbf { h } _ { t } ) \mathbf { x } _ { t - 1 } + \mathbf { B } _ { t } ( \mathbf { h } _ { t } ) \mathbf { h } _ { t } , } \\ & { \mathbf { y } _ { t } = \mathbf { C } _ { t } ( \mathbf { h } _ { t } ) \mathbf { x } _ { t } . } \end{array}\tag{14}
$$

Unlike a standard time-varying model, the state transition matrices ${ \bf A } _ { t }$ and input matrices $\mathbf { B } _ { t }$ are dynamically parameterized as functions of the current input $\mathbf { h } _ { t } .$ This content-aware selectivity allows the model to adapt its behavior in real time, for instance, by amplifying the influence of a detected anomaly via $\mathbf { B } _ { t }$ or extending its memory of that event via ${ \bf A } _ { t }$

The ScalesMamba2 layer wraps this powerful core in a residual, multi-scale pattern, crucial for preserving signal fidelity across different resolutions:

$$
\mathrm { S c a l e s M a m b a 2 ( { \bf U } ) = { \bf U } + U p ( M a m b a 2 ( D o w n ( { \bf U } ) ) ) } ,\tag{15}
$$

where Down and $\mathrm { U p }$ handle temporal resolution changes. This structure ensures that the SSM operates on a compressed representation while the residual connection preserves highfrequency details.

## B. The DMb and SSMb Blocks: Recurrent U-Net Structures

We propose two U-Net-like blocks, the Deviation Mamba2 Block (DMb) and the Self-Supervised Mamba2 Block (SSMb), which leverage stacked ScalesMamba2 layers in an encoderdecoder configuration with skip connections.

1) Deviation Mamba2 Block: Designed for refining a precomputed deviation signal σ, the 3-scale DMb follows this structure:

Encoder Path: e<sup>(0)</sup> = σ,

$$
\begin{array} { r l } & { \mathbf { e } ^ { ( i ) } = \mathrm { M a m b a 2 } \left( \mathrm { D o w n } \left( \mathbf { e } ^ { ( i - 1 ) } \right) \right) , } \\ & { i = 1 , 2 , 3 , } \end{array}
$$

Decoder Path: d<sup>(0)</sup> = e<sup>(3)</sup>,

$$
\begin{array} { r l } & { \mathbf { d } ^ { ( j ) } = \mathrm { M a m b a 2 } \left( \mathrm { U p } \left( \mathbf { d } ^ { ( j - 1 ) } \right) + \mathbf { e } ^ { ( 3 - j ) } \right) , } \\ & { j = 1 , 2 , 3 , } \end{array}
$$

$$
\mathrm { O u t p u t : \quad D M b } ( { \pmb \sigma } ) = { \bf d } ^ { ( 3 ) } .\tag{16}
$$

This architecture forces the model to capture anomaly “fingerprints” at progressively coarser temporal scales in the encoder, while the decoder uses skip connections to fuse this contextual understanding with fine-grained morphological details for precise anomaly reconstruction.

2) Self-Supervised Mamba2 Block: For the more demanding task of masked segment reconstruction from raw signal x, the deeper 5-scale SSMb is employed:

$$
\begin{array} { r l } { \mathrm { E n c o d e r ~ P a t h : } } & { \mathbf { u } ^ { ( 0 ) } = \mathbf { x } , } \\ & { \mathbf { u } ^ { ( i ) } = \mathrm { M a m b a 2 } \left( \mathrm { D o w n } \left( \mathbf { u } ^ { ( i - 1 ) } \right) \right) , } \\ & { i = 1 , . . . , 5 , } \\ { \mathrm { D e c o d e r ~ P a t h : } } & { \mathbf { v } ^ { ( 0 ) } = \mathbf { u } ^ { ( 5 ) } , } \\ & { \mathbf { v } ^ { ( j ) } = \mathrm { M a m b a 2 } \left( \mathrm { U p } \left( \mathbf { v } ^ { ( j - 1 ) } \right) + \mathbf { u } ^ { ( 5 - j ) } \right) , } \\ & { j = 1 , . . . , 5 , } \\ { \mathrm { O u t p u t : } } & { \mathrm { S S M b } ( \mathbf { x } ) = \mathbf { v } ^ { ( 5 ) } . } \end{array}\tag{17}
$$

The increased depth expands the model’s effective receptive field exponentially, enabling the bottleneck layer to reason about long-range global rhythm and context, which is essential for plausible reconstruction of heavily masked regions.

## C. Architectural Design Rationale

The efficacy of these recurrent U-Net structures for ECG analysis is rooted in the unique synergy between the U-Net topology and the Mamba2 core.

• Synergistic Multi-scale Processing: The U-Net’s downsampling path creates coarse-grained representations, compelling the model to learn context over longer time windows. While this can attenuate the amplitude of sharp, transient anomalies, Mamba2’s selective SSM excels at detecting and latching onto these weakened but contextually significant signals at low resolutions. This is a task where static CNN kernels might fail. The skip connections then re-inject high-fidelity, high-frequency information to the decoder, allowing the Mamba2 layers there to perform precise, context-informed reconstruction.

• Modeling of Non-Stationary Dynamics: ECG signals are inherently non-stationary, e.g., changing heart rates and ST-segment shifts. Mamba2’s content-aware state transitions, as defined in (14), provide the ideal mechanism for adaptively modeling these dynamics at each time scale, a feat beyond the reach of fixed-kernel CNNs.

• Hierarchical Context for Reconstruction: The deep, 5-scale SSMb captures a rich hierarchy of features. At the deepest level, i.e., the bottleneck, Mamba2’s longrange memory capacity is leveraged to understand global cardiac rhythm. At shallower levels, it models local P-QRS-T waveform dynamics. The decoder then masterfully integrates this hierarchical information, guided by skip connections, to generate reconstructions that are both globally plausible and morphologically precise.

• Specialization through Depth Trade-off: The choice of depth is a deliberate trade-off. The 3-scale DMb is optimized for refining a deviation signal, where the primary task is to enhance localized, pre-identified anomalies. A moderate receptive field is sufficient. In contrast, the 5-scale SSMb, with its vast receptive field $\left( \propto \ : \ : 2 ^ { 5 } \right)$ is designed for the more complex self-supervision task of reconstructing raw signals from large gaps, which demands a deep understanding of global context.

## APPENDIX H

## SHAPE-BASED BARYCENTER MDTW CLUSTERING WITH T-WAVE-BASED INITIALIZATION

Here we describe in detail a variant of our shape-based barycenter MDTW clustering that first aligns cluster centers to characteristic T-wave positions before performing multidimensional DTW (MDTW)-based clustering and barycenter averaging.

## A. Preprocessing and T-Wave Center Selection

Let $\{ \widetilde { n } _ { r } \} _ { r = 1 } ^ { m } \subset \mathbb { R } ^ { T \times L }$ be the preprocessed, normalized ECG segments, each of length $T$ in time and L leads.

1) T-wave detection: For each segment $\widetilde { n } _ { r } .$ , detect its Twave peak time $t _ { r }$

2) Sort and cycle computation: Sort these times:

$$
\begin{array} { l } { { \displaystyle t _ { ( 1 ) } \leq t _ { ( 2 ) } \leq \cdots \leq t _ { ( m ) } , } } \\ { { \displaystyle T _ { b } = \frac { 1 } { m - 1 } \sum _ { r = 1 } ^ { m - 1 } \left( t _ { ( r + 1 ) } - t _ { ( r ) } \right) . } } \end{array}\tag{18}
$$

3) Reference time points: Choose a target cluster count C. Define

$$
\tau _ { c } = t _ { ( 1 ) } + ( c - 1 ) \frac { T _ { b } } { C } , \quad c = 1 , \dots , C .\tag{19}
$$

4) Initial centers: For each c, pick

$$
\begin{array} { r l } & { r _ { c } = \arg \underset { 1 \leq r \leq m } { \operatorname* { m i n } } \left| t _ { r } - \tau _ { c } \right| , } \\ & { \mu _ { c } ^ { ( 0 ) } = \widetilde { n } _ { r _ { c } } . } \end{array}\tag{20}
$$

## B. Multi-Dimensional DTW Distance

For any two segments $X , Y \in \mathbb { R } ^ { T \times L }$ , define the per-point cost

$$
d \left( X ( \alpha ) , Y ( \beta ) \right) = \sum _ { \ell = 1 } ^ { L } \left| X _ { \ell } ( \alpha ) - Y _ { \ell } ( \beta ) \right| ^ { 2 } ,\tag{21}
$$

and let P be the set of all monotonic warping paths from (1, 1) to (T, T). Then

$$
D _ { \mathrm { M D T W } } ( X , Y ) = \operatorname* { m i n } _ { \pi \in { \mathcal { P } } } \sum _ { ( \alpha , \beta ) \in \pi } d \left( X ( \alpha ) , Y ( \beta ) \right) .\tag{22}
$$

## C. T-Wave-Initialized MDTW Clustering Algorithm

Given the T-wave-based initial centers $\{ \mu _ { c } ^ { ( 0 ) } \} _ { c = 1 } ^ { C }$ , we iterate for $q = 0 , \ldots , Q - 1$ as follows:

1) Assignment, Cluster E-step:

$$
\mathcal { C } _ { c } ^ { ( q ) } = \left\{ r : c = \arg \operatorname* { m i n } _ { j = 1 , \ldots , C } D _ { \mathrm { M D T W } } \left( \widetilde { n } _ { r } , \mu _ { j } ^ { ( q ) } \right) \right\} .\tag{23}
$$

2) Alignment, MDTW paths: For each cluster c and each $r \in \mathcal { C } _ { c } ^ { ( q ) }$ , compute

$$
\pi _ { r , c } = \arg \operatorname* { m i n } _ { \pi \in \mathcal { P } } \sum _ { ( \alpha , \beta ) \in \pi } d \left( \mu _ { c } ^ { ( q ) } ( \alpha ) , \widetilde { n } _ { r } ( \beta ) \right) .\tag{24}
$$

3) Barycenter Accumulation, M-step: For each cluster $c ,$ each time index $\alpha ,$ and lead ℓ:

$$
\begin{array} { r l } & { \mathrm { A c c } _ { c } ( \alpha , \ell ) = \displaystyle \sum _ { r \in \mathcal { C } _ { c } ^ { ( q ) } } \displaystyle \sum _ { ( \alpha , \beta ) \in \pi _ { r , c } } \widetilde { n } _ { r , \ell } ( \beta ) , } \\ & { \quad \mathrm { C n t } _ { c } ( \alpha ) = \displaystyle \sum _ { r \in \mathcal { C } _ { c } ^ { ( q ) } } \left. \left\{ \beta : ( \alpha , \beta ) \in \pi _ { r , c } \right\} \right. . } \end{array}\tag{25}
$$

4) Update Barycenter:

$$
\mu _ { c , \ell } ^ { ( q + 1 ) } ( \alpha ) = \left\{ \begin{array} { l l } { \displaystyle \frac { \mathrm { A c c } _ { c } ( \alpha , \ell ) } { \mathrm { C n t } _ { c } ( \alpha ) } , } & { \mathrm { C n t } _ { c } ( \alpha ) > 0 , } \\ { \displaystyle \mu _ { c , \ell } ^ { ( q ) } ( \alpha ) , } & { \mathrm { o t h e r w i s e } , } \\ { \ell = 1 , \ldots , L . } \end{array} \right.\tag{26}
$$

Repeat until convergence or $q = Q$ . The final set $\{ \mu _ { c } ^ { ( Q ) } \}$ are the cluster barycenters, each initialized at a representative T-wave position and refined via MDTW barycenter averaging.

D. Comparison Between Selected Sample and MDTW-Aligned Codebook Sample

Below we compare: 1) the raw ECG segment chosen as the cluster center based on T-wave-based initialization, as shown in Fig. 2; and 2) the corresponding MDTW-aligned codebook sample after barycenter fusion, as shown in Fig. 3. The fusion process yields a more standardized morphology, particularly correcting non-pathological fluctuations in leads 3 and 5.

## E. Soft-DTW Barycenter

Soft-DTW is a differentiable relaxation of the classic DTW objective, controlled by a smoothing parameter $\lambda > 0$ . Suppose we have the same cluster $\{ \widetilde { n } _ { 1 } , \dots , \widetilde { n } _ { m } \}$ as before. For a candidate barycenter $\mu ,$ the Soft-DTW distance to a segment $\widetilde { n } _ { r }$ can be viewed as:

$$
{ \mathcal { D } } _ { \mathrm { s o f t } } ( \mu , \widetilde { n } _ { r } ) = \operatorname* { m i n } _ { \pi \in { \mathcal { P } } } \sum _ { ( \alpha , \beta ) \in \pi } \left\| \mu ( \alpha ) - \widetilde { n } _ { r } ( \beta ) \right\| ^ { 2 } + \lambda \Phi ( \pi ) ,\tag{27}
$$

where $\Phi ( \pi )$ is a smooth penalty approximating the discrete warping cost. Soft-DTW then defines the barycenter $\mu$ by minimizing the aggregate cost over all segments:

$$
\mu _ { \mathrm { S D T W } } = \arg \operatorname* { m i n } _ { \mu } \sum _ { r = 1 } ^ { m } \mathcal { D } _ { \mathrm { s o f t } } ( \mu , \widetilde { n } _ { r } ) .\tag{28}
$$

Because $\mathcal { D } _ { \mathrm { s o f t } }$ is differentiable in $\mu ,$ we can solve (28) via gradient-based optimization, e.g.,

$$
\mu  \mu - \alpha \nabla _ { \mu } ( \sum _ { r = 1 } ^ { m } { \mathcal { D } } _ { \mathrm { s o f t } } ( \mu , \widetilde { n } _ { r } ) ) ,\tag{29}
$$

where α is the learning rate. In practice, if the implementation supports multi-dimensional Soft-DTW, one can directly handle $( T \times L )$ data within each iteration. The final solution is $\mu _ { c } ^ { \mathrm { S D T W } }$ , serving as the smooth DTW-based barycenter for cluster c.

## F. NormECG Codebook Construction

Using either DBI in Appendix H-C or Soft-DTW in $\mathsf { A p - }$ pendix H-E, we obtain a barycenter $\mu _ { c }$ for each cluster $c =$ $1 , \ldots , C .$ . The set of these barycenters forms our NormECG Codebook:

$$
B _ { \mathrm { N o r m E C G } } = \left\{ \mu _ { 1 } , \mu _ { 2 } , \dots , \mu _ { C } \right\} ,\tag{30}
$$

where each $\mu _ { c }$ is a time-warped representative of normal ECG segments within that cluster. Compared to naive pointwise averaging, these DTW-based barycenters better handle local temporal shifts and preserve waveform morphology more faithfully.

## APPENDIX I

## DETAILED ANALYSIS OF EXTRACTION MODULE

In this section, we illustrate how a slightly abnormal but still healthy ECG sample from a NormECG Cluster can be refined by Multi-dimensional DTW (MDTW) barycenter fusion. Specifically, we choose one central sample in the NormECG Cluster, as shown in Fig. 2, and show its corresponding aligned Codebook sample in Fig. 3.

![](images/7464d9c01dcd79b5131e1c04473f9d20bb4c55b1d2622e0da483e5579dad765d.jpg)

![](images/26b9ae58b00607c9db34341393ffd422eb5656b4676d331c28afd53e036801f1.jpg)

![](images/ed3f898d3778f7c1b018d8e44abee5435a090bbaeba152a99546d014470855d4.jpg)

![](images/4c01cf1f8eb1f7622982ac56e6c84ede4ec3c6657f6f9deecd1ab472a5c22b09.jpg)

![](images/65dd0b90242965a950d47531669ede3ba02cce68d920fc4292662dc40d67313d.jpg)

![](images/ad4fb1558f5f2d206d71bf641b8ad705bab29ba26a0eca0fe3a2a59edbe570e3.jpg)

![](images/b013a506ddbdfb71c3ff7bdc169d84226f5a3f9ea9f01143a52683d59d6e9370.jpg)

![](images/2f7614a7a266226f740854af1ff8c57a06be095b7fec1b7ca82fd3183f72b0b5.jpg)

![](images/41764c517f4b06d104176919e89d81679d9403ae3fb3f27b16213a4094fe5956.jpg)

![](images/7e8045e9d47b7ad87d00e7d300dfdc8d0c56d3ef1f41b1e0d37d740b92851ad8.jpg)

![](images/16874803ec489d84816431d07d9b66dec4bb42e3b0e3390afbaba2ffdb46abdd.jpg)

![](images/24b5859d8fbfdfcd8b45c0463fcf6b31061a1e439d5174f43932e3561ece3fda.jpg)

![](images/00908b2c3648e062e21a043bd3bce518f1f8961064a72ac9116a05f2ab7b47da.jpg)  
Fig. 2. One sample in the NormECG cluster selected as the central point. The red box shows leads with slightly abnormal signal acquisition due to factors unrelated to cardiovascular disease, while the blue or unmarked regions indicate the standard lead form of a normal ECG.  
Fig. 3. An MDTW-aligned sample in the NormECG Codebook corresponding to the selected sample in Fig. 2. The red box again highlights the leads that previously had slightly abnormal signal acquisition, which are now more consistent with typical normal ECG waveforms. The remaining leads, shown in blue or unmarked regions, are mostly unchanged and retain the characteristic shape. Such MDTW fusion helps mitigate external artifacts and preserve essential waveform patterns for subsequent R-peak detection and anomaly assessment.

As illustrated, the originally selected central sample contained mild signal deviations in Leads 3 and 5, as indicated by the red box in Fig. 2, possibly due to acquisition noise or external factors. Although these deviations are not caused by true cardiovascular pathology, they could still trigger false alarms in a feature-based anomaly detection system. By contrast, after MDTW-based fusion with other similar normal samples, the resulting Codebook sample in Fig. 3 exhibits smoother and more typical waveforms in the affected leads, while retaining the overall morphology of the original signal in other leads. Consequently, this aligned Codebook better reflects the intrinsic normal ECG pattern, facilitating more robust R-peak detection and providing a high-quality reference for subsequent anomaly-deviation calculations and

![](images/b8c45b4f4f9cd151c7d78f83850ed1a0f57cbd9d7576c9fa7d7792c968143c8a.jpg)  
Fig. 4. Comparison of ablation performance using CNN and Mamba2 as classifiers. The left panel shows ARE, the right panel shows DLE, and the baseline denotes the combination of the two embeddings.

reconstruction-based analyses.

## APPENDIX J

## EMBEDDING COMPONENT CONTRIBUTION AND ANALYSIS

In Fig. 4, the fused DLE+ARE embedding sets the baseline, i.e., 0% change. We then measure the relative drop or gain when using only Association Reconstruction Embedding (ARE) or only Deviation Localization Embedding (DLE).

## A. ARE Only

• On PTB-XL, removing DLE causes Balanced Accuracy to fall by ∼ 4% and F1 by ∼ 6%, since ARE emphasizes cross-lead correlations but misses localized amplitude deviations, e.g., minor ST shifts, that DLE captures. AU-ROC decreases by ∼ 3% overall, yet F1 for myocardial infarction improves by ∼ 1.5%, reflecting ARE’s strength in integrating multi-lead MI patterns.

• On ML12L, ARE-only drops Balanced Accuracy by ∼ 5% but boosts AUROC for conduction disorders by ∼ 2%, because cross-lead attention highlights the broad QRS prolongation patterns there. F1 for repolarization abnormalities also gains ∼ 1%, as ARE smooths out noisy local fluctuations.

## B. DLE Only

• On PTB-XL, removing ARE reduces F1 by ∼ 7% and AUROC by ∼ 4%, while sensitivity actually increases by ∼ 3%, showing that DLE’s per-lead deviation maps are highly sensitive to subtle anomalies but lack ARE’s inter-lead context to suppress false positives.

• On ML12L, DLE-only lowers overall Balanced Accuracy by ∼ 6%, yet recall for hypertrophy and ischemia classes rises by ∼ 4%, because DLE pinpoints local morphological deviations, e.g., voltage criteria, critical for these diagnoses.

## C. Detailed Interpretation

• DLE excels at detecting beat-level morphological anomalies, e.g., ST/T deviations and voltage changes, and thus can increase sensitivity or recall for classes defined by sharp local deviations.

• ARE strengthens inter-lead consistency, improving precision and class separability, as reflected in AUROC and F1 for broader conduction or MI patterns, but may smooth away isolated spikes that DLE would flag.

• Fused embedding leverages both: DLE provides high recall for subtle, localized deviations, while ARE provides high precision by contextualizing anomalies across leads, yielding the best trade-off in Balanced Accuracy, F1, and AUROC.

## APPENDIX K

## PARAMETER SETTINGS AND MODEL STRUCTURE

In this section, we summarize the key hyperparameters and implementation choices used throughout our training pipeline.

## A. Model Architecture

• d\_model = 256: hidden dimension of the Mamba2 input and output.

• d\_state = 64: latent state dimension in the LTV-SSM.

• d\_conv = 4: kernel size for the small convolutional networks that parameterize the time-varying matrices A<sub>t</sub>, $B _ { t } , C _ { t } .$ and D<sub>t</sub>.

• dropout\_rate = 0.1: AlphaDropout after the SELU activation.

## B. Optimization

• Loss: Smooth L1.

• Optimizer: Adam on all model and tokenizer parameters.

• Learning rate: $\eta = 4 \times 1 0 ^ { - 5 }$

• Scheduler: StepLR(step\_size = 5, γ = 0.5), which decays η by half every 5 epochs.

• Number of epochs per split for pretraining: 30.

• Number of epochs in linear evaluation and generalization validation: 20.

These settings were chosen based on preliminary experi ments balancing training stability, GPU memory constraints, all hyper-parameters remain the same for all models.

## C. Linear Evaluation Classifier Architectures

The architectures of the three linear evaluation classifiers used in our experiments are as follows:

• CNN Classifier:

– Four sequential 1D convolutional blocks:

∗ Block 1: Conv1d 36 → 64, kernel size 7, stride 1, padding 3; BatchNorm1d; ReLU; MaxPool1d with kernel size 2 and stride 2.

TABLE VIII  
CLASSIFICATION PERFORMANCE OF DIFFERENT DOWNSTREAM CLASSIFIERS USING BASELINE AND DLE&ARE AS ECG REPRESENTATIONS ON EXTERNAL CLINICAL DATASETS. THE BEST RESULT IN EACH COLUMN IS SHOWN IN BOLD.
<table><tr><td>Classifier</td><td>Representation</td><td>Accuracy</td><td>F1 Score</td><td>Precision</td><td>Recall</td><td>AUC</td></tr><tr><td>CNN</td><td>Baseline DLE&amp;ARE</td><td> $0 . 2 2 1 \pm 0 . 0 1 1$   ${ \bf 0 . 2 4 8 \pm 0 . 0 1 2 }$ </td><td> $0 . 3 5 6 \pm 0 . 0 1 2$   $\mathbf { 0 . 3 9 5 \pm 0 . 0 1 6 }$ </td><td> $0 . 4 4 9 \pm 0 . 0 1 4$   ${ \bf 0 . 4 6 6 \pm 0 . 0 0 9 }$ </td><td> $0 . 3 0 3 \pm 0 . 0 1 8$   ${ \bf 0 . 3 4 8 \pm 0 . 0 2 2 }$ </td><td> $0 . 5 7 3 \pm 0 . 0 1 5$   $\mathbf { 0 . 6 0 2 \pm 0 . 0 1 1 }$ </td></tr><tr><td>Transformer</td><td>Baseline DLE&amp;ARE</td><td> $0 . 2 4 2 \pm 0 . 0 2 4$   ${ \bf 0 . 2 7 6 \pm 0 . 0 2 7 }$ </td><td> $0 . 3 5 2 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 5 3 2 \pm 0 . 0 0 9 }$ </td><td> $0 . 2 6 9 \pm 0 . 0 2 6$ </td><td> $0 . 5 7 4 \pm 0 . 0 1 1$ </td></tr><tr><td>Mamba</td><td>Baseline DLE&amp;ARE</td><td> $0 . 2 4 1 \pm 0 . 0 1 7$   ${ \bf 0 . 3 4 8 \pm 0 . 0 2 9 }$ </td><td> ${ \bf 0 . 4 0 1 \pm 0 . 0 2 2 }$   $0 . 3 8 6 \pm 0 . 0 2 2$ </td><td> $0 . 5 1 2 \pm 0 . 0 1 1$   $0 . 3 9 9 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 3 4 1 \pm 0 . 0 3 0 }$   $0 . 3 7 7 \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 5 8 8 \pm 0 . 0 0 7 }$   $0 . 4 9 9 \pm 0 . 0 0 3$   $\mathbf { 0 . 5 4 3 \pm 0 . 0 0 9 }$ </td></tr></table>

∗ Block 2: Conv1d 64 → 128, kernel size 5, stride 1, padding 2; BatchNorm1d; ReLU; MaxPool1d with kernel size 2 and stride 2.

∗ Block 3: Conv1d 128 → 256, kernel size 3, stride 1, padding 1; BatchNorm1d; ReLU; MaxPool1d with kernel size 2 and stride 2.

∗ Block 4: Conv1d 256 → 512, kernel size 3, stride 1, padding 1; BatchNorm1d; ReLU; MaxPool1d with kernel size 2 and stride 2.

– The output is flattened and passed through fully connected layers: 512 → 256 → 64 → number of classes.

## • Transformer Classifier:

– Transformer encoder: four layers, each with d\_model = 720, nhead = 4, feedforward dimension 512, and dropout rate 0.1.

– After encoding, the output is reshaped and flattened.

– Fully connected head: 512 → 256 → 64 → number of classes.

## • Mamba Classifier:

– Mamba block: d $\underline { { \mathrm { m o d e l } } } = 7 2 0 , \ : \mathrm { d \_ s t a t e } = 6 4 .$ and d\_conv = 4, followed by SELU activation and AlphaDropout with dropout rate 0.1.

– The output is reshaped to (batch, 36, 5000) and then flattened.

– Fully connected head: 512 → 256 → 64 → number of classes.

## APPENDIX L

## SOTA MODELS USED FOR PERFORMANCE COMPARISON

## A. SOTA Feature Extraction Models

DINOv2 [18] is a self-supervised vision transformer originally developed for masked image modeling. It is adapted here for ECG signals by treating them as images to leverage its strong unsupervised representation capabilities.

TimesFM [19] is a decoder-only foundation model trained via contrastive learning on large-scale time series, capturing temporal dependencies well for sequence-level ECG modeling.

Chronos [20] is a transformer-based model pretrained with time-warp augmentations and tokenized representations, designed to model complex temporal dynamics.

MOIRAI [21] is a GAN-based anomaly detection model for multivariate physiological signals, capturing fine-grained correlations and supporting unsupervised ECG analysis.

## B. SOTA ECG-CVD Classification Models

ASTCL [22] is a transformer classifier trained with selfsupervision for long ECG sequences, effective at learning discriminative features without labels.

BeatGAN [23] is a beat-level generative model that enables fine-grained rhythm and morphology analysis for arrhythmia detection.

MERL [24] is a multi-resolution residual autoencoder for morphological anomaly detection using hierarchical representations and residual learning.

ECGAD [25] is an autoencoder-based model that detects abnormal heartbeats via reconstruction error, tailored for ECG anomaly detection.

## APPENDIX M

## DETAILED OBSERVATIONAL STUDY RESULTS

We compared the characteristic performance of the baseline model and the fusion model on the real-world external clinical dataset generated by them. This comparison was conducted for three downstream classifier model architectures, including convolutional neural network, Transformer, and Mamba model. We evaluated two input types: baseline, i.e., original electrocardiogram, and the extracted feature combination of DLE + ARE. The performance was evaluated through accuracy (ACC), F1 score, precision, recall rate, and AUC. Table VIII provides detailed metric values. In most classifiers and evaluation metrics, the combined representation consistently outperformed other configurations, indicating its strong generalization ability in various real data scenarios.

## APPENDIX N

## FULL TABLE OF PERFORMANCE COMPARISON OF SOTA SELF-SUPERVISED ECG CLASSIFIERS ON THE UNSEEN TEST SETS CPSC2020 AND WGEX

We present a comprehensive evaluation of state-of-the-art self-supervised ECG classification models on two external datasets, CPSC2020 and WGEX, both of which were not seen during training. As shown in Table IX, our method consistently outperforms existing approaches, including ASTCL, BeatGAN, MERL, and ECGAD, across all four evaluation metrics: balanced accuracy (B-Acc), F1 score, AUC, and sensitivity. Notably, our model achieves the highest AUC, 0.800 and 0.809, and balanced accuracy, 0.547 and 0.584, on both datasets, indicating superior generalization ability under distributional shifts. These results demonstrate that our representation learning framework yields more robust and discriminative features for unseen database scenarios.

TABLE IX  
PERFORMANCE COMPARISON OF STATE-OF-THE-ART SELF-SUPERVISED ECG CLASSIFIERS ON TWO UNSEEN TEST SETS, CPSC2020 AND WGEX, EVALUATED USING 10-FOLD CROSS-VALIDATION.
<table><tr><td>Model</td><td colspan="4">CPSC2020</td><td colspan="4">WGEX</td></tr><tr><td></td><td>B-Acc</td><td>F1</td><td>AUC</td><td>Sensitivity</td><td>B-Acc</td><td>F1</td><td>AUC</td><td>Sensitivity</td></tr><tr><td>ASTCL</td><td> $0 . 4 9 6 \pm 0 . 1 1 4$ </td><td> $0 . 5 2 2 \pm 0 . 0 4 1$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 1$ </td><td> $0 . 7 0 0 \pm 0 . 0 3 4$ </td><td> $0 . 5 3 0 \pm 0 . 0 5 7$ </td><td> $0 . 4 6 2 \pm 0 . 0 7 0$ </td><td> $0 . 7 2 4 \pm 0 . 0 9 6$ </td><td> $0 . 6 9 9 \pm 0 . 1 1 1$ </td></tr><tr><td>BeatGAN</td><td> $0 . 4 8 0 \pm 0 . 0 6 9$ </td><td> $0 . 5 4 6 \pm 0 . 0 2 7$ </td><td> $0 . 7 6 2 \pm 0 . 0 8 8$ </td><td> $0 . 6 9 0 \pm 0 . 0 0 6$ </td><td> $0 . 5 0 1 \pm 0 . 1 1 1$ </td><td> $0 . 4 7 6 \pm 0 . 0 0 2$ </td><td> $0 . 7 6 5 \pm 0 . 0 0 3$ </td><td> $0 . 7 1 1 \pm 0 . 1 0 3$ </td></tr><tr><td>MERL</td><td> $0 . 2 4 3 \pm 0 . 0 7 7$ </td><td> $0 . 4 6 2 \pm 0 . 1 0 3$ </td><td> $0 . 7 2 4 \pm 0 . 0 8 3$ </td><td> $0 . 6 8 6 \pm 0 . 0 9 7$ </td><td> $0 . 2 0 8 \pm 0 . 0 4 6$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 2$ </td><td> $0 . 6 7 3 \pm 0 . 0 6 1$ </td><td> $0 . 5 2 2 \pm 0 . 0 0 8$ </td></tr><tr><td>ECGAD</td><td> $0 . 5 0 2 \pm 0 . 0 6 2$ </td><td> $0 . 4 3 6 \pm 0 . 0 4 2$ </td><td> $0 . 7 4 0 \pm 0 . 1 0 7$ </td><td> $0 . 7 2 3 \pm 0 . 0 3 0$ </td><td> $0 . 5 1 5 \pm 0 . 0 2 9$ </td><td> $0 . 3 3 8 \pm 0 . 0 9 9$ </td><td> $0 . 7 3 0 \pm 0 . 0 8 8$ </td><td> $0 . 6 6 7 \pm 0 . 0 8 0$ </td></tr><tr><td>Ours</td><td> ${ \bf 0 . 5 4 7 \pm 0 . 0 3 9 }$ </td><td> $\mathbf { 0 . 5 5 7 \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 8 0 0 \ : \pm 0 . 0 3 6 }$ </td><td> $\mathbf { 0 . 7 4 0 \pm 0 . 0 9 3 }$ </td><td> ${ \bf 0 . 5 8 4 \pm 0 . 0 4 9 }$ </td><td> ${ \bf 0 . 5 3 8 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 8 0 9 \ : \pm 0 . 0 6 1 }$ </td><td> $\mathbf { 0 . 7 1 2 \pm 0 . 0 9 2 }$ </td></tr></table>

## REFERENCES

[1] P. Wagner, N. Strodthoff, R. D. Bousseljot et al., “PTB-XL, a large publicly available electrocardiography dataset,” Scientific Data, vol. 7, no. 1, p. 154, 2020.

[2] H. Liu, D. Chen, D. Chen, X. Zhang, H. Li, L. Bian, M. Shu, and Y. Wang, “A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements,” Scientific Data, vol. 9, p. 272, 2022.

[3] E. A. Perez Alday, A. Gu, A. J. Shah, C. Robichaux, A.-K. I. Wong, C. Liu, F. Liu, A. Bahrami Rad, A. Elola, S. Seyedi, Q. Li, A. Sharma, G. D. Clifford, and M. A. Reyna, “Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020,” Physiological Measurement, vol. 41, no. 12, p. 124003, 2020.

[4] J. Zheng, H. Guo, and H. Chu, “A large scale 12-lead electrocardiogram database for arrhythmia study,” PhysioNet, 2022, version 1.0.0.

[5] B. Gow, T. Pollard, L. A. Nathanson, A. Johnson, B. Moody, C. Fernandes, N. Greenbaum, J. W. Waks, P. Eslami, T. Carbonati, A. Chaudhari, E. Herbst, D. Moukheiber, S. Berkowitz, R. Mark, and S. Horng, “MIMIC-IV-ECG: Diagnostic electrocardiogram matched subset,” PhysioNet, 2023, version 1.0.

[6] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences with´ structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.

[7] J. T. H. Smith, A. Warrington, and S. W. Linderman, “Simplified state space layers for sequence modeling,” arXiv preprint arXiv:2208.04933, 2022.

[8] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[9] T. Dao and A. Gu, “Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 10 041–10 071.

[10] B. N. Patro and V. S. Agneeswaran, “Mamba-360: Survey of state space models as transformer alternative for long sequence modelling: Methods, applications, and challenges,” arXiv preprint arXiv:2404.16112, 2024.

[11] ——, “SiMBA: Simplified Mamba-based architecture for vision and multivariate time series,” arXiv preprint arXiv:2403.15360, 2024.

[12] Z. Zhang, A. Liu, I. Reid, R. Hartley, B. Zhuang, and H. Tang, “Motion Mamba: Efficient and long sequence motion generation with hierarchical and bidirectional selective SSM,” arXiv preprint arXiv:2403.07487, 2024.

[13] E. Izci, M. A. Ozdemir, R. Sadighzadeh, and A. Akan, “Arrhythmia detection on ECG signals by using empirical mode decomposition,” in Proc. Medical Technologies National Congress (TIPTEKNO), 2018, pp. 1–4.

[14] N. Karimian, Z. Guo, M. Tehranipoor, and D. Forte, “Highly reliable key generation from electrocardiogram (ECG),” IEEE Transactions on Biomedical Engineering, vol. 64, no. 6, pp. 1400–1411, 2017.

[15] P. De Chazal, M. O’Dwyer, and R. B. Reilly, “Automatic classification of heartbeats using ECG morphology and heartbeat interval features,” IEEE Transactions on Biomedical Engineering, vol. 51, no. 7, pp. 1196– 1206, 2004.

[16] Y. Dai, F. Gieseke, S. Oehmcke, Y. Wu, and K. Barnard, “Attentional feature fusion,” in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision, 2021, pp. 3560–3569.

[17] Y. Hu, Y. Zhao, J. Liu, J. Pang, C. Zhang, and P. Li, “An effective frequency-domain feature of atrial fibrillation based on time–frequency analysis,” BMC Medical Informatics and Decision Making, vol. 20, pp. 1–11, 2020.

[18] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “DINOv2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023, version 1.

[19] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” arXiv preprint arXiv:2310.10688, 2023.

[20] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” arXiv preprint arXiv:2403.07815, 2024, version 3.

[21] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” arXiv preprint arXiv:2402.02592, 2024.

[22] N. Wang, P. Feng, Z. Ge, Y. Zhou, B. Zhou, and Z. Wang, “Adversarial spatiotemporal contrastive learning for electrocardiogram signals,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 10, pp. 13 845–13 859, 2024.

[23] B. Zhou, S. Liu, B. Hooi, X. Cheng, and J. Ye, “BeatGAN: Anomalous rhythm detection using adversarially generated time series,” in Proc. Int. Joint Conf. Artificial Intelligence, 2019, pp. 4433–4439.

[24] C. Liu, Z. Wan, C. Ouyang, A. Shah, W. Bai, and R. Arcucci, “Zeroshot ECG classification with multimodal learning and test-time clinical knowledge enhancement,” arXiv preprint arXiv:2403.06659, 2024.

[25] A. Jiang, C. Huang, Q. Cao, Y. Xu, Z. Zeng, K. Chen, Y. Zhang, and Y. Wang, “Self-supervised anomaly detection pretraining enhances longtail ECG diagnosis,” arXiv preprint arXiv:2408.17154, 2024.