# A Multidimensional Data-Driven Hybrid Transformer Framework for Non-invasive Continuous Blood Pressure Prediction

Yuexin Ma<sup>1,†</sup>, Jingqi Hou<sup>1,†</sup>, Yuxuan Kang<sup>1,†</sup> and Zhaoying Liu<sup>1,∗</sup>

<sup>1</sup>College of Computer Science, Beijing University of Technology, Beijing, China

<sup>†</sup>These authors contributed equally to this work.

<sup>∗</sup>Author to whom any correspondence should be addressed.

E-mail: zhaoying.liu@bjut.edu.cn

Other author e-mails: Yuexin Ma: andrain@emails.but.edu.cn; Jingqi Hou: lcstate@emails.bjut.edu.cn; Yuxuan Kang: 23070211@emails.but.edu.cn

ORCID iDs: Yuexin Ma: 0009-0007-9705-2374; Jingqi Hou: 0009-0002-4213-2661; Yuxuan Kang:

0009-0003-5020-1417; Zhaoying Liu: 0000-0001-6991-0123

Keywords: blood pressure prediction, non-invasive monitoring, electrocardiography, photoplethysmography,

Transformer, pulse transit time, physiological feature engineering

## Abstract

Objective. To develop and evaluate a cufless continuous blood pressure (BP) estimator using temporal physiological and demographic features. We propose a hybrid Transformer framework to estimate diastolic and systolic BP from ECG/PPG-derived feature sequences. Approach. Rather than raw waveforms, the framework models 10-step sequences of six physiological descriptors and two demographic covariates. A Multi-Source Temporal Encoder Module combines Transformer, Kolmogorov–Arnold Network, and XGBoost branches to capture complementary temporal, nonlinear, and tabular information. A Dynamic Conditional Fusion-Decoder applies diferential multi-head attention, token-weighted aggregation, and gated residual correction. A robust composite objective jointly optimizes DBP and SBP. Main results. Using the MIMIC-III Waveform and Clinical Databases, the source pool comprised 28,486 waveform segments from 203 subjects, and feature generation retained 53,621 observations from 166 subjects. On 2,431 segment-level held-out test windows, mean error ± standard deviation was 0.41 mmHg ± 3.74 mmHg for diastolic BP and −1.60 mmHg ± 5.95 mmHg for systolic BP, with 95% limits of agreement of [-6.93, 7.74] and [-13.25, 10.06] mmHg, respectively. The proportions within 10 mmHg were 98.48% and 94.36%. The framework achieved the lowest standard deviations and narrowest limits of agreement among the locally retrained baselines. Significance. The feature-sequence fusion framework improved agreement with reference BP and fell within numerical AAMI and BHS Grade A thresholds on this split. This retrospective analysis is not formal device validation; subject-disjoint and external evaluation remain necessary before clinical use.

## 1 Introduction

Hypertension is a major modifiable risk factor for cardiovascular disease and stroke worldwide (McEvoy et al., 2024; Zhou et al., 2021). Conventional non-invasive blood pressure (BP) measurement mainly relies on cuf-based techniques, including auscultatory and oscillometric methods (Muntner et al., 2019; Meidert and Saugel, 2018). However, these methods provide only intermittent measurements and are limited in long-term continuous physiological monitoring owing to cuf-related discomfort, device bulkiness, and operational constraints (Meidert and Saugel, 2018; Pilz et al., 2024; Hua et al., 2024; Min et al., 2025). To address these limitations, non-invasive cufless monitoring technologies have garnered significant attention because of their potential to revolutionize traditional practices by ofering undisturbed, sustainable measurements (Hua et al., 2024; Min et al., 2025; Zhao et al., 2023). In particular, the pulse transit time (PTT)—defined as the temporal delay between the cardiac electrical activation (ECG R-peak) and the peripheral pulse arrival (PPG foot)—has been widely validated as a surrogate marker negatively correlated with BP (Mukkamala et al., 2015; Block et al., 2020; Ding et al., 2016).

Nonetheless, BP estimation relying solely on PTT is susceptible to inter-subject variability and environmental interferences, often requiring frequent calibration to compensate for these ofsets (Mukkamala et al., 2015; Zhao et al., 2023). To enhance precision, recent studies have introduced auxiliary physiological descriptors; for instance, Ding et al. (2016) proposed the PPG intensity ratio (PIR) to complement PTT, significantly improving estimation accuracy and dynamic tracking capabilities. Concurrently, machine learning and deep learning paradigms have been deployed to extract features directly from signal waveforms. While some approaches utilize regression algorithms on statistical features to meet Association for the Advancement of Medical Instrumentation (AAMI) standards (Kachuee et al., 2017; Rastegar et al., 2023), others employ deep models like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to learn representations from raw ECG and PPG signals (Slapniˇcar et al., 2019; Tang et al., 2024). However, despite their initial success, these existing deep models often struggle with generalization across diverse populations, lack robustness against motion artifacts, and sufer from poor interpretability (Zhao et al., 2023; Weber-Boisvert et al., 2023; Mehta et al., 2024).

To overcome these challenges, we propose a multidimensional data-driven pipeline that fuses ECG/PPG-derived physiological descriptors with an improved hybrid Transformer architecture. Rather than feeding raw waveform samples directly to the network, the method models consecutive records of timing, cardiac, PPG morphology, and demographic features. The source ECG, PPG, and continuous arterial blood pressure (ABP) signals were obtained from the MIMIC-III Waveform Database, and age and body weight were linked from the MIMIC-III Clinical Database v1.4 (Moody et al., 2020; Johnson et al., 2016; Johnson, Pollard, and Mark, 2016).

The main contributions of this work are summarized as follows.

First, we propose a Multi-Source Temporal Encoder Module (MSEM) to process multivariate physiological feature sequences. By integrating a Transformer encoder with parallel Kolmogorov– Arnold Network (KAN) and XGBoost (XGB) branches, the MSEM captures complementary temporal, non-linear, and tabular representations of hemodynamic variation (Liu, Wang, et al., 2025; Chen and Guestrin, 2016).

Second, we introduce a Dynamic Conditional Fusion-Decoder (DCFD) to unify multi-branch representations. It employs a diferential multi-head attention mechanism to reduce feature redundancy and enhance cross-branch fusion eficiency (Ye et al., 2025).

Third, we formulate a composite robust point-regression objective as a weighted sum of complementary terms rather than treating them as separate training stages. Learnable target-scale parameters balance the DBP and SBP squared-error terms; scheduled focal-style and quantile penalties emphasize large and asymmetric residuals; an early-epoch Huber term stabilizes initial optimization; and KAN, target-scale, and residual-correction terms provide regularization and auxiliary supervision (Kendall et al., 2018; Lin et al., 2017; Koenker and Bassett, 1978; Huber, 1964). Evaluated on the MIMIC-III dataset (Moody et al., 2020), this approach falls within the numerical AAMI mean-error and SD criteria and the numerical BHS Grade A percentage thresholds on the segment-level held-out test set (ISO, 2018, 2022; O’Brien et al., 1993).

## 2 Related Work

Cufless BP estimation traditionally exploits the inverse correlation between pulse transit time (PTT) and BP (Mukkamala et al., 2015). While PTT remains a fundamental physiological marker, PTT-only models are sensitive to individual diferences and vascular compliance changes (Mukkamala et al., 2015; Zhao et al., 2023). To enhance precision, recent literature augments PTT with morphological and intensity descriptors, such as the pulse intensity ratio (PIR) (Ding et al., 2016), and applies deep learning paradigms to extract representations directly from ECG and PPG waveforms (Rastegar et al., 2023; Tang et al., 2024; Zhao et al., 2023). Existing studies suggest that robust preprocessing, multimodal feature fusion, and attention-based modeling are promising directions for improving cufless BP estimation performance.

## 2.1 Evolution of Deep Learning Architectures

The progression of BP prediction models has shifted from feature engineering to end-to-end learning, focusing on capturing spatial morphological features and temporal dependencies.

Convolutional Neural Networks (CNNs). Convolutional Neural Networks serve as powerful extractors for local waveform patterns. Variants like 1D-CNN and ResNet have proven efective in learning abstract features such as systolic upstroke time and notch width (Slapniˇcar et al., 2019; Liu, Qiao, et al., 2025). Notably, architectures like DSRUnet introduce sparse residual connections and deep supervision to prevent gradient vanishing, achieving AAMI-compliant results on public datasets (Lai et al., 2024). U-Net-based waveform reconstruction and lightweight Squeeze U-Net variants have also been explored for PPG-based BP estimation (Athaya and Choi, 2021, 2022). A recent image-encoding alternative, GHC-Net, transforms PPG signals into Gramian angular fields and combines hybrid dilated convolutions with channel attention; its reported task is BP-category classification rather than continuous-value regression (Cheng et al., 2026). Parallel or multi-stream CNN-based frameworks have also been employed to process signal derivatives (e.g., velocity and acceleration PPG), thereby mimicking clinically relevant waveform interpretation (Slapniˇcar et al., 2019).

Recurrent Neural Networks (RNNs). To model the dynamic nature of BP, Recurrent Neural Networks, particularly LSTMs and GRUs, are widely used to capture historical dependencies and autonomic regulation (Zhao et al., 2023). For instance, hybrid models combining CNNs for spatial extraction and Bi-LSTMs/Conv-LSTMs for temporal modeling have demonstrated high accuracy (Jeong and Lim, 2021; Kamanditya et al., 2024; Liu, Qiao, et al., 2025; Tanveer and Hasan, 2019). In 2026, LAST-CBPM combined wavelet-based PPG denoising, temporal self-attention, BiLSTM modeling, and subject-specific residual calibration for quasi-continuous BP monitoring on MIMIC-IV and an independent in-house dataset (Song et al., 2026). However, RNNs entail serial computation and may be less efective at modeling very long-range dependencies, which can limit their ability to capture ultra-long temporal patterns such as circadian rhythms (Vaswani et al., 2017).

Transformers and Generative Models. Recently, Transformer architectures utilizing self-attention have emerged as strong performers in cufless BP estimation and waveform synthesis (Tian et al., 2025; Zheng et al., 2025; Nawaz et al., 2024). Models like UTransBPNet combine U-Net backbones with cross-attention mechanisms to track hemodynamic changes in dynamic scenarios (Zheng et al., 2025). Recent 2026 studies further combine attention with complementary signal representations: MCAFNet uses ConvNeXt and Transformer branches with multi-scale cross-attention to reconstruct continuous ABP from PPG, whereas FFDM-GAT-Transformer couples Fourier-decomposition-derived PPG features with graph attention and a Transformer encoder for point BP estimation (Tian et al., 2026; Batra et al., 2026). Generative Adversarial Networks (GANs) have also been explored for “waveform translation” (mapping PPG to ABP), as exemplified by PPG2ABP and CycleGAN-based reconstruction approaches (Ibtehaz et al., 2022; Mehrabadi et al., 2022). Difusion models, such as TDSTF, have likewise been investigated for blood pressure time-series forecasting and probabilistic generation in ICU settings, helping address data sparsity and uncertainty (Chang et al., 2024). In another 2026 multimodal design, Ma et al. (2026) encoded ECG and PPG signals using Gramian angular fields and combined convolutional, recurrent, and temporal-attention modules to capture complementary spatial-temporal patterns for continuous BP prediction.

## 2.2 Multimodal Data Fusion Strategies

Single-modality approaches are vulnerable to motion artifacts and physiological noise. Consequently, efective fusion of ECG, PPG, and other complementary modalities (including bioimpedance) is increasingly regarded as a key strategy for improving robustness against artifact-induced degradation and sensor-specific noise (Zhao et al., 2023; Min et al., 2025).

Feature-, Decision-, and Model-level Fusion. Feature-level fusion typically involves concatenating handcrafted descriptors (PTT, PIR) or deep embeddings prior to regression. While straightforward, this requires strict temporal synchronization and may overlook complex nonlinear cross-modal interactions (Stahlschmidt et al., 2022; Duan et al., 2024; Ma et al., 2023). Decision-level fusion aggregates outputs from independent classifiers (e.g., SVM, RF) via voting or stacking (Stahlschmidt et al., 2022; Duan et al., 2024). In contrast, model-level fusion jointly learns cross-modal interactions within a shared architecture, often using Mixture-of-Experts (MoE) or attention mechanisms. For example, recent wearable systems use gating or attention networks to dynamically recalibrate sensor contributions according to contextual cues such as temperature, contact force, or motion state, thereby improving stability across resting and exercise conditions (Li et al., 2025; Chen et al., 2025).

## 2.3 Gap Analysis: Generalization and Calibration

Despite architectural advances, three key gaps remain between academic benchmarks and real-world deployment (Zhao et al., 2023; Min et al., 2025). A recent systematic review further identified protocol realism, calibration drift, population diversity, and fairness-aware reporting as major barriers to trustworthy real-world cufless BP monitoring (Cisnal et al., 2026).

Subject Variability. The mapping from PPG to BP is highly subject-specific, governed by arterial stifness, vascular compliance, and age. Most deep models can degrade markedly in subject-independent or out-of-distribution evaluation (Zhao et al., 2023; Moulaeifard et al., 2025; Mehta et al., 2024). While domain adaptation techniques exist, they often introduce additional methodological and computational complexity (Moulaeifard et al., 2025). Recent work has attempted to mitigate this variability through personalized adaptation. Tang et al. (2026) combined self-supervised representation learning with few-shot subject-specific adaptation using only a small number of labeled samples. Although promising, the requirement for subject-specific labeled BP data remains a practical obstacle to fully calibration-free deployment.

Calibration Dependency. Many recent solutions still require periodic calibration using cuf devices to determine baseline ofsets (Mukkamala et al., 2015; Kasbekar et al., 2023; Min et al., 2025). Calibration-free models incorporating demographic data (age, BMI) have been explored, but robustness across diverse populations remains an open issue, particularly given continued concerns about skin-tone-related limitations in PPG sensing and validation (Liang et al., 2026; Colvonen, 2021).

Computational Eficiency. High-performing attention-based models can impose non-trivial memory, latency, and power costs, posing challenges for deployment on resource-constrained wearable edge devices (Min et al., 2025; Pankaj et al., 2026).

Addressing these limitations, our proposed method introduces a hybrid framework. We combine Kolmogorov–Arnold Networks (KAN) with a diferential-attention-based fusion mechanism to enhance multi-branch feature integration (Liu, Wang, et al., 2025; Ye et al., 2025).

## 3 Methodology

3.1 Problem Definition

Let $\mathcal { D } = \{ ( \mathbf { X } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ denote the model-ready dataset with N prediction samples. Each input $\mathbf { X } _ { i } \in \mathbb { R } ^ { L \times \bar { F } }$ contains $L = 1 0$ consecutive feature records and $F = 8$ variables: pulse transit time $( \mathrm { P T T } )$ , heart rate, PPG foot-to-peak interval, maximum PPG slope, reflection index (RI), systolic-todiastolic timing (SD time), body weight, and age. The target $\mathbf { y } _ { i } = [ y _ { d b p } , y _ { s b p } ] ^ { \top } \in \mathbb { R } ^ { 2 }$ contains the DBP and SBP reference values of the subsequent record. Our goal is to learn a mapping $\mathcal { F } : \mathbf { X } _ { i }  \hat { \mathbf { y } } _ { i }$ that minimizes prediction error against the ABP-derived references.

## 3.2 Overall Framework

The proposed framework is a multidimensional data-driven pipeline designed for continuous cufless BP monitoring. As illustrated in Fig. 1 and Fig. 2, the system integrates four tightly coupled components: (1) source waveform and metadata construction, (2) upstream derivation of physiological descriptors, (3) model-ready feature-sequence construction, and (4) hybrid Transformer-based prediction with a composite optimization objective. In particular, the model consists of a Multi-Source Temporal Encoder Module (MSEM) and a Dynamic Conditional Fusion-Decoder (DCFD), which together capture temporal dependencies, nonlinear hemodynamic characteristics, and efective cross-branch feature interactions.

For clarity, Fig. 2 summarizes the complete execution flow from upstream signal preparation to model evaluation.

## 3.3 Hybrid Transformer Architecture

The deep learning backbone consists of two modules: the Multi-Source Temporal Encoder Module (MSEM) and the Dynamic Conditional Fusion-Decoder (DCFD).

3.3.1 Multi-Source Temporal Encoder Module (MSEM) The encoding stage is designed to capture both sequential dependencies and statistical distributions. Before window construction, missing observations are removed and the eight input variables are standardized using statistics fitted on the training partition. Each batch therefore contains $\mathbf { X } \in \mathbb { R } ^ { B \times 1 0 \times 8 }$ . Within the model, each window is additionally normalized along the temporal dimension for the neural branches. The KAN input is the resulting 80-dimensional flattened window. The XGBoost input is formed from the globally standardized window flattened to 80 dimensions and concatenated with eight feature-wise temporal means and eight temporal standard deviations, yielding 96 dimensions.

The backbone utilizes a Transformer encoder to extract temporal hidden states H. To leverage the strengths of diferent modeling paradigms, a branch generator produces three parallel token representations:

• Main Head: a linear projection of the mean-pooled Transformer representation, providing a temporal feature-sequence estimate;

• KAN Head: a Kolmogorov–Arnold Network operating on the normalized 80-dimensional flattened window to model explicit nonlinear physiological relationships;

• XGB Head: an XGBoost representation operating on the 96-dimensional flattened-window statistics to provide a robust tabular baseline estimate.

![](images/45f549f40c515175eef490b50ab7670c47b7d67f1c98c5a9b7c10e3556b5001f.jpg)  
Figure 1: Architecture of the proposed hybrid framework for continuous blood pressure prediction. Given a standardized feature window $\mathbf { X } \in \mathbb { R } ^ { 1 0 \times 8 }$ , the Multi-Source Temporal Encoder Module (MSEM) processes complementary temporal, nonlinear, and tabular views using Transformer, KAN, and XGBoost branches, respectively. The encoder states $\mathbf { H } _ { \mathrm { e n c } }$ serve as decoder memory, while branch-specific projections $\mathbf { t } _ { k } = \mathbf { W } _ { k } \mathbf { h } _ { k } + \mathbf { b } _ { k }$ map the three representations to the common tokens $\mathbf { t } _ { \mathrm { e n c } } .$ $\mathbf { t } _ { \mathrm { K A N } }$ , and $\mathbf { t } _ { \mathrm { X G B } }$ Their concatenation $\mathbf { T } _ { \mathrm { b r a n c h } }$ is processed by the Dynamic Conditional Fusion-Decoder (DCFD), producing the fused representation $\mathbf { H } _ { \mathrm { f u s e d } } ,$ decoder output $\mathbf { H } _ { \mathrm { d e c } }$ , token-weighted aggregate $\mathbf { z } _ { \mathrm { a g g } } .$ , and neural correction $\Delta \mathbf { y }$ . In parallel, XGBoost supplies the two-dimensional baseline $\mathbf { y } _ { \mathrm { b a s e , X G B } } ;$ the final prediction is $\hat { \mathbf { y } } = \mathbf { y } _ { \mathrm { b a s e , X G B } } + \Delta \mathbf { y }$ . The exact implemented decomposition of $\Delta \mathbf { y }$ is given in equation 3.

Unlike conventional Multi-Layer Perceptrons (MLPs) with fixed node-wise activation functions, Kolmogorov–Arnold Networks (KANs) place learnable activation functions on edges. For an input vector $\mathbf { x } \in \mathbb { R } ^ { d _ { i n } } ,$ a KAN layer is defined as a matrix of learnable one-dimensional functions $\Phi = \{ \phi _ { q , p } \}$ where $\phi _ { q , p }$ connects the p-th input to the q-th output node. The computation is expressed as:

$$
x _ { q } ^ { l + 1 } = \sum _ { p = 1 } ^ { d _ { i n } } \phi _ { q , p } ( x _ { p } ^ { l } ) , \qquad q = 1 , \ldots , d _ { o u t }\tag{1}
$$

where $\phi _ { q , p } ( \cdot )$ is typically parameterized as a B-spline or a weighted sum of basis functions. This design enables the network to better approximate the highly nonlinear hemodynamic variations involved in BP prediction.

3.3.2 Dynamic Conditional Fusion-Decoder $( D C F D )$ The fusion-decoder stage is designed to integrate heterogeneous representations produced by diferent modeling branches. Given the branch tokens $\mathbf { t } _ { i } ,$ we employ a Multi-Head Diferential Attention (MHDA) module to enhance informative interactions while suppressing redundant responses across branches.

Unlike standard self-attention, MHDA computes two distinct similarity maps $( A _ { 1 } , A _ { 2 } )$ and subtracts them, i.e.,

$$
A = A _ { 1 } - \lambda A _ { 2 } ,\tag{2}
$$

so as to suppress redundant common-mode information while emphasizing salient variations.

The attended features are then aggregated into a fused representation ${ \mathbf { H } } _ { f u s e d }$ for final prediction. To ensure training stability and gradual integration of complex features, we employ a gated residual fusion strategy:

$$
\hat { \mathbf { y } } = \mathbf { y } _ { \mathrm { b a s e , X G B } } + \sigma ( \mathbf { g } ) \odot r ( s ) \operatorname { L i n e a r } ( \mathbf { H } _ { \mathrm { f u s e d } } ) + \sum _ { k } \gamma _ { k } \mathbf { t } _ { k } ^ { \mathrm { r e s } }\tag{3}
$$

where $\mathbf { y } _ { \mathrm { b a s e , X G B } }$ is the robust two-dimensional XGBoost baseline estimate, $\sigma ( \mathbf { g } )$ is the sigmoid gate, $r ( s )$ is the residual ramp coeficient at training step s, and $\gamma _ { k }$ weights the residual token $\mathbf { t } _ { k } ^ { \mathrm { r e s } }$ from branch k. The terms following y jointly constitute the aggregate neural correction $\Delta \mathbf { y }$ shown compactly in Fig. 1; the neural branches therefore mainly learn residual corrections to the statistical baseline.

![](images/d720ec673764440453beb04aecc6b08ca3f7a1397c2eecb0b5b3c2f13e6c7279.jpg)  
Figure 2: The overall workflow of the proposed continuous blood pressure prediction system. Source waveform processing provides the physiological provenance of the eight retained variables. The prediction stage operates on standardized 10-step feature sequences rather than raw ECG/PPG waveform samples.

## 3.4 Dataset and Data Preparation

We used synchronized ECG, PPG, and ABP segments from the MIMIC-III Waveform Database v1.0 (Moody et al., 2020; Johnson et al., 2016), resampled to 125 Hz. Age and weight were linked from the MIMIC-III Clinical Database v1.4 (Johnson, Pollard, and Mark, 2016). Table 1 summarizes cohort composition before and after signal-quality screening and feature generation.

Table 1: Composition of the source metadata pool and final analytical cohort. Percentages are calculated within each row.
<table><tr><td>Stage</td><td>Statistical unit</td><td>Total, n</td><td>Male, n (%)</td><td>Female, n (%)</td></tr><tr><td rowspan="2">Source pool</td><td>Subjects</td><td>203</td><td>115 (56.65)</td><td>88 (43.35)</td></tr><tr><td>Waveform segments</td><td>28,486</td><td>16,791 (58.94)</td><td>11,695 (41.06)</td></tr><tr><td rowspan="2">Analytical cohort</td><td>Subjects</td><td>166</td><td>104 (62.65)</td><td>62 (37.35)</td></tr><tr><td>Observations</td><td>53,621</td><td>43,620 (81.35)</td><td>10,001 (18.65)</td></tr></table>

Because the exported model-ready table omitted subject identifiers and sex labels, provenance was reconstructed from the source metadata and waveform archive. Unique age–weight combinations resolved 52,890 observations; the remaining 731 observations from three shared combinations were assigned by matching continuous SBP–DBP pairs to raw MAT-file labels at 10<sup>−9</sup> precision. This mapping was used only for cohort accounting, not as a model input or partitioning variable.

Feature generation required finite BP references and detectable ECG/PPG fiducials. Completecase filtering then removed 9 of 53,630 rows with missing weight. No pre-specified physiological-range, SBP > DBP, or RI-clipping rule was applied; therefore, the analytical table retained two rows with $\mathrm { S B P \le D B P }$ , approximate minima of 3.8 mmHg for DBP and 50.0 mmHg for SBP, and extreme RI values. No retrospective exclusion was performed to preserve the original modeling pipeline. A post hoc audit of the 2,431 held-out test references found DBP and SBP ranges of 35.62–98.23 and 85.36–185.26 mmHg, respectively, with no test row satisfying SBP ≤ DBP.

## 3.5 Sex and Gender Considerations

In accordance with the Sex and Gender Equity in Research (SAGER) guidelines (Heidari et al., 2016), sex was considered as a biological variable when describing the source pool and final analytical cohort. The binary M/F field available in the source metadata was not retained as a model predictor, a partitioning variable, or a stratification factor when the model-ready table and experimental splits

(a) Effect of Zero-phase Filtering on PPG

were created; the subsequent subject mapping was used only for cohort accounting. The higher proportion of male subjects in the final cohort reflects the composition of the available secondary ICU data after signal-quality screening and was not caused by sex-specific recruitment or exclusion criteria. Gender identity was not available in the source databases; therefore, this article reports sex rather than gender and makes no gender-based inferences. Model performance was not evaluated separately by sex, and the reported aggregate results should not be interpreted as evidence of equivalent performance across sex groups.

## 3.6 Preprocessing and Signal Alignment

The modeling stage begins from an exported feature table and does not load raw ECG or PPG waveform arrays. The waveform processing described here documents the physiological provenance of the retained descriptors rather than the tensor representation supplied to the network. The source signals were sampled at 125 Hz; representative upstream processing used third-order zero-phase Butterworth filtering (ECG: 0.5 Hz to 40 Hz; PPG: 0.4 Hz to 8 Hz), robust median/IQR normalization, and PPG polarity correction before fiducial detection. Fig. 3 illustrates these operations and the timing landmarks from which the model-ready variables were derived.

![](images/2a282f4a64612f11b4e5adae5c90b28fab3baf3ea327835ecb0d7c915931ffc2.jpg)  
Figure 3: Upstream signal preparation and feature provenance. (a) Raw and zero-phase band-pass-filtered PPG. (b) Robust-normalized ECG and PPG after polarity correction. (c) Fiducial landmarks used to derive $\mathrm { P T T }$ , the foot-to-peak interval, and heart rate, together with the eight retained predictors. The waveforms illustrate feature derivation; the prediction model receives 10-step feature sequences rather than raw signals.

ECG R-peaks and the corresponding PPG pulse feet and systolic peaks provide the timing landmarks for three retained variables. Let $t _ { R } ^ { ( j ) }$ and $t _ { R } ^ { ( j + 1 ) }$ denote the sample indices of two consecutive ECG R-peaks, and let $t _ { \mathrm { f o o t } }$ and $t _ { \mathrm { p e a k } }$ denote the PPG foot and systolic-peak indices associated with the jth cardiac cycle. With sampling frequency $f _ { s } = 1 2 5 \mathrm { { H z } }$ , pulse transit time $( \mathrm { P T T } )$ , the PPG foot-to-peak interval $\left( T _ { \mathrm { f o o t - p e a k } } \right)$ , the RR interval $( T _ { R R } )$ , and heart rate (HR) are defined as

$$
\begin{array} { r l r } & { \mathrm { P T T } = \frac { t _ { \mathrm { f o o t } } - t _ { R } ^ { ( j ) } } { f _ { s } } , } & { T _ { \mathrm { f o o t - p e a k } } = \frac { t _ { \mathrm { p e a k } } - t _ { \mathrm { f o o t } } } { f _ { s } } , } \\ & { T _ { R R } = \frac { t _ { R } ^ { ( j + 1 ) } - t _ { R } ^ { ( j ) } } { f _ { s } } , } & { \mathrm { H R } = \frac { 6 0 } { T _ { R R } } . } \end{array}\tag{4}
$$

Here, PTT, $T _ { \mathrm { f o o t - p e a k } }$ , and $T _ { R R }$ are expressed in seconds, and HR is expressed in beats per minute.   
The remaining waveform-derived inputs are the maximum PPG slope, the reflection index (RI;

reflected-wave amplitude relative to the primary systolic amplitude), and SD time (the systolic-todiastolic PPG timing interval). Age and body weight are appended as static demographic covariates. These definitions describe the eight columns retained in the final model-ready table; waveform samples and other candidate morphology descriptors are not supplied to the prediction network.

## 3.7 Physiological Feature Engineering

The final analytical table contains exactly eight predictors per record. The six physiological descriptors derived upstream from synchronized ECG/PPG recordings are pulse transit time (PTT), PPG foot-to systolic-peak interval, heart rate, maximum PPG slope, reflection index (RI), and systolic-to-diastolic timing (SD time). The two static demographic covariates are body weight and age. Table 2 lists their stored variable names and definitions. No raw waveform samples, resampled waveform templates, wavelet coeficients, spectral descriptors, shapelets, AUC features, or P2 descriptors are included in the model input.

Table 2: Predictor variables retained in the model-ready feature table.
<table><tr><td>Category</td><td>Stored variable</td><td>Description</td></tr><tr><td>Timing</td><td>ptt</td><td>ECG R-peak to PPG foot interval (s)</td></tr><tr><td>Timing</td><td>ptt_foot_pk</td><td>PPG foot-to-systolic-peak interval (s)</td></tr><tr><td>Cardiac</td><td>hr</td><td>Heart rate (beats/min)</td></tr><tr><td>PPG morphology</td><td>ppg-slope_max</td><td>Normalized maximum PPG slope  $\mathrm { ( a . u . / s ) }$ </td></tr><tr><td>PPG morphology</td><td>ri</td><td>Reflection index (ratio)</td></tr><tr><td>PPG timing</td><td>sd_time</td><td>Systolic-to-diastolic timing descriptor (s)</td></tr><tr><td>Demographic</td><td>weight</td><td>Body weight (kg)</td></tr><tr><td>Demographic</td><td>age</td><td>Age (years)</td></tr></table>

The 53,621 complete observations described above were used without additional post hoc outlier exclusion. Segment boundaries were inferred from resets or discontinuities in the exported index, and complete segments were assigned to the training, validation, and test partitions before scaling or window construction. The input scaler and output scaler were fitted only on training-segment rows. Within each segment, a unit-stride window uses 10 consecutive standardized feature records to predict the DBP and SBP values of the next record. Thus, each neural input has shape $1 0 \times 8$ , while the prediction length is one record.

## 3.8 Composite Robust Loss Function

BP prediction is formulated as a two-target point-regression task requiring accurate and robust DBP and SBP estimates. The implemented network returns one DBP and one SBP estimate for each input window. We define the overall training objective as

$$
\mathcal { L } _ { t o t a l } = \sum _ { k \in \{ \mathrm { D B P } , \mathrm { S B P } \} } w _ { k } \mathcal { L } _ { A , k } + \alpha _ { F } \mathcal { L } _ { F } + \alpha _ { Q } \mathcal { L } _ { Q } + w _ { H } \mathcal { L } _ { H } + \mathcal { L } _ { r e g } + w _ { R } \mathcal { L } _ { R } ,\tag{5}
$$

where k indexes the two BP targets, $w _ { k }$ is the corresponding target weight, $\mathcal { L } _ { F }$ and $\mathcal { L } _ { Q }$ are focal-style and quantile residual penalties, $\mathcal { L } _ { H }$ is an early-epoch Huber term, $\mathcal { L } _ { \boldsymbol { r } \boldsymbol { e } \boldsymbol { g } }$ contains KAN and target-scale regularization, and $\mathcal { L } _ { R }$ supervises the fused residual correction. For every mini-batch, the active terms are reduced to scalars and combined by weighted addition into the single objective $\mathcal { L } _ { t o t a l }$ used for back-propagation. The word “adaptive” applies specifically to the target-wise squared-error weighting, not to every term in the sum. This target-weighting term is

$$
\mathcal { L } _ { A , k } = \frac { 1 } { 2 } \left[ \exp ( - s _ { k } ) \mathrm { M S E } _ { k } + s _ { k } + \log ( 2 \pi ) \right] ,\tag{6}
$$

where $s _ { k }$ is a learnable target-specific scale parameter shared across all input windows. Together with the fixed target coeficient $w _ { k } , \exp ( - s _ { k } )$ adjusts the relative contribution of the DBP and SBP squared errors during optimization. The coeficients $\alpha _ { F }$ and $\alpha _ { Q }$ follow an epoch-dependent ramp-up schedule, whereas $w _ { H } = 0 . 0 5$ for the first ten epochs and $w _ { H } = 0$ thereafter; the remaining regularization and residual-supervision coeficients are fixed during a run.

Each component addresses a diferent aspect of the regression problem:

• Adaptively weighted squared error $( \mathcal { L } _ { A , k } )$ : balances the two regression targets using learned global scale parameters.

• Focal-style regression loss $( \mathcal { L } _ { F } )$ : emphasizes examples with large absolute errors.

• Quantile residual loss $( \mathcal { L } _ { Q } )$ : introduces asymmetric penalties at $\tau = 0 . 1$ and $\tau = 0 . 9$ to improve robustness to uneven residual distributions.

• Huber loss $( \mathcal { L } _ { H } )$ : provides additional robustness during the first ten training epochs.

• Regularization and residual supervision $( \mathcal { L } _ { r e g }$ and $\mathcal { L } _ { R } )$ : regularize the KAN and target-scale parameters and align the fused correction with the observed residual.

## 3.9 Implementation Details

The proposed model was implemented in Python using PyTorch 2.10.0+cu128 and trained on a single NVIDIA GeForce RTX 4090 GPU. The model configuration and implementation settings are provided in Table 3.

Table 3: Model configuration and implementation settings.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Input Configuration</td><td></td></tr><tr><td>Source Waveform Sampling Rate</td><td>125 Hz</td></tr><tr><td>Predictor Dimension (F)</td><td>8 variables</td></tr><tr><td>Model Sequence Length (L)</td><td>10 feature records</td></tr><tr><td>Window Step</td><td>1 record</td></tr><tr><td>Prediction Length</td><td>1 record</td></tr><tr><td>KAN Input Dimension</td><td>80</td></tr><tr><td>XGBoost Input Dimension</td><td>96</td></tr><tr><td>Batch Size (B)</td><td>32</td></tr><tr><td>MSEM Encoder</td><td></td></tr><tr><td>Transformer Layers</td><td>6</td></tr><tr><td>Hidden Dimension  $\left( d _ { \mathrm { m o d e l } } \right)$ </td><td>128</td></tr><tr><td>Attention Heads</td><td></td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer</td><td>RAdam with Lookahead</td></tr><tr><td>Initial Learning Rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Warmup Epochs</td><td>10</td></tr><tr><td>LR Scheduler</td><td>Cosine Annealing</td></tr><tr><td>Software and Hardware Environment</td><td></td></tr><tr><td>Deep Learning Framework</td><td>PyTorch 2.10.0+cu128</td></tr><tr><td>GPU</td><td>NVIDIA GeForce RTX 4090</td></tr></table>

Training was performed using the RAdam optimizer wrapped with Lookahead to improve convergence stability. A cosine annealing learning rate schedule was adopted after a 10-epoch linear warmup phase.

## 4 Experiments

## 4.1 Experimental Protocol

All comparison and ablation experiments were conducted on the same model-ready table of 53,621 complete observations derived from the MIMIC dataset. To ensure a fair evaluation, all models used the same model-ready feature table, segment partition, training-only scaling procedure, sequence length, and target definitions for DBP and SBP prediction. Contiguous observations were first grouped into segments from discontinuities in the retained segment index. These segments were then randomly allocated to training, validation, and test partitions in an $8 0 \% / 1 0 \% / 1 0 \%$ ratio using a single random seed, and the same split was used for all baseline, comparison, and ablation experiments unless otherwise specified. This procedure prevented windows derived from the same detected segment from crossing partitions. However, because the exported model-ready table did not retain the MIMIC subject identifier, the split was not subject-disjoint and diferent segments from the same subject may occur in more than one partition. Accordingly, “held-out test set” in this work refers to a segment-level holdout. All reported results correspond to a single seeded run; repeated-run variability was not evaluated.

For neural-network-based models, training was performed with early stopping on the validation set. Unless otherwise specified, the patience value was set to 20, and the checkpoint with the best monitored validation loss or validation metric was selected for final testing. For tree-based models such as XGBoost, the model was trained on the same training split and evaluated on the same test split. All reported results were computed on the segment-level held-out test set.

The comparison experiments were designed to evaluate the proposed framework against representative baseline models, including conventional deep learning architectures and standalone statistical or neural regressors. The ablation experiments were designed to isolate the contribution of major components in the proposed framework, including the XGBoost baseline/prior branch and the KAN branch.

## 4.2 Evaluation Metrics

The model performance was evaluated using multiple error and clinical accuracy metrics. For each blood pressure target, the prediction error was defined as

$$
e _ { i } = { \hat { y } } _ { i } - y _ { i } ,\tag{7}
$$

where $\hat { y } _ { i }$ denotes the predicted BP value and $y _ { i }$ denotes the corresponding reference value.

The mean error was used to measure systematic bias:

$$
\mathrm { M e a n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e _ { i } .\tag{8}
$$

A mean error closer to zero indicates lower systematic bias. The standard deviation (SD) of the errors was used to quantify the dispersion of prediction errors:

$$
\mathrm { S D } = { \sqrt { \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( e _ { i } - { \bar { e } } ) ^ { 2 } } } ,\tag{9}
$$

where ¯e is the mean error. A smaller SD indicates more stable predictions.

We also reported the 95% limits of agreement (LoA) based on Bland–Altman analysis:

$$
\mathrm { L o A } _ { 9 5 \% } = \left[ \bar { e } - 1 . 9 6 \times \mathrm { S D } , \bar { e } + 1 . 9 6 \times \mathrm { S D } \right] .\tag{10}
$$

A narrower LoA interval indicates better agreement between the predicted and reference BP values. The reported bias, SD, and LoA are descriptive per-window point estimates for the 2,431 test windows. Because temporally adjacent windows and multiple segments can arise from the same subject, these observations are not statistically independent. Subject-clustered confidence intervals and repeated measures Bland–Altman analysis were not calculated because subject identifiers were not retained in the evaluation export. The reported LoA should therefore not be interpreted as subject-level precision and may be narrower than estimates obtained with subject-clustered inference.

In addition, cumulative error percentages were calculated at the 5 mmHg, 10 mmHg, and 15 mmHg thresholds:

$$
P _ { \leq T } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( | e _ { i } | \leq T ) \times 1 0 0 \% ,\tag{11}
$$

where $T \in \{ 5 , 1 0 , 1 5 \}$ mmHg and I(·) is the indicator function. These cumulative percentages were further compared with the numerical percentage thresholds defined by the British Hypertension Society (BHS). These numerical comparisons do not constitute formal compliance testing, clinical device validation, or certification under ISO 81060-3/AAMI or the BHS protocol.

## 4.3 Comparison with Representative Baseline Models

To evaluate the proposed framework against representative architectures, we selected six baseline families with reference to their established use in cufless BP estimation or their foundational formulations: CNN–LSTM (Jeong and Lim, 2021), ResNet-1D (Liu, Qiao, et al., 2025), U-Net 1D (Athaya and Choi, 2021), XGBoost (Chen and Guestrin, 2016), Transformer (Vaswani et al., 2017), and KAN (Liu, Wang, et al., 2025). The XGBoost baseline uses flattened statistical features for direct regression, the Transformer baseline uses a temporal neural backbone, and the KAN baseline uses a standalone nonlinear neural regressor. These references identify the corresponding architecture families; the numerical results in Table 4 were not taken from the cited studies. All six baselines were implemented and retrained locally, then evaluated using the same model-ready inputs, data split, preprocessing protocol, and target definitions as the proposed framework.

Table 4: Comparison with representative baseline models. Arrows indicate whether lower (↓) or higher (↑) values are better. The descriptive per-window 95% limits of agreement (LoA) are calculated as Mean ± 1.96 SD without repeated-measures adjustment.
<table><tr><td rowspan="3">Model</td><td colspan="6">DBP (mmHg)</td><td colspan="6">SBP (mmHg)</td></tr><tr><td colspan="3">Error Agreement Mean† SD ↓</td><td colspan="3">Cumulative Accuracy (%) ≤ 5 ↑ ≤ 10 ↑</td><td colspan="3">Error Agreement SD ↓</td><td colspan="3">Cumulative Accuracy (%) ≤ 5 ↑ ≤ 10 ↑</td></tr><tr><td></td><td></td><td>95% LoA</td><td></td><td></td><td>≤ 15 ↑</td><td>Mean†</td><td></td><td>95% LoA</td><td></td><td></td><td>≤ 15 ↑</td></tr><tr><td>CNN-LSTM</td><td>0.49</td><td>4.54</td><td>[-8.41, 9.39]</td><td>88.21</td><td>95.46</td><td>97.73</td><td>0.94</td><td>7.61</td><td>[-13.98, 15.86]</td><td>67.57</td><td>85.11</td><td>92.06</td></tr><tr><td>ResNet-1D</td><td>0.50</td><td>5.06</td><td>[-9.42, 10.42]</td><td>82.01</td><td>93.73</td><td>96.22</td><td>1.09</td><td>8.13</td><td>[-14.84, 17.02]</td><td>64.78</td><td>83.90</td><td>91.38</td></tr><tr><td>U-Net 1D</td><td>-0.41</td><td>4.55</td><td>[-9.33, 8.51]]</td><td>88.21</td><td>94.94</td><td>97.20</td><td>0.67</td><td>8.45</td><td>[-15.89, 17.23]</td><td>65.84</td><td>86.09</td><td>91.91</td></tr><tr><td>XGBoost</td><td>0.41</td><td>3.86</td><td>[-7.16, 7.98]</td><td>87.53</td><td>97.43</td><td>99.01</td><td>-0.60</td><td>7.10</td><td>[-14.52, 13.32]</td><td>64.78</td><td>87.23</td><td>93.73</td></tr><tr><td>Transformer</td><td>-0.75 -0.91</td><td>4.76 4.13</td><td>[-10.08, 8.58]</td><td>84.88</td><td>96.90</td><td>98.34</td><td>0.91</td><td>7.80</td><td>[-14.38, 16.20]</td><td>64.85 53.15</td><td>84.13</td><td>91.16</td></tr><tr><td>KAN</td><td></td><td></td><td>[-9.00, 7.18]</td><td>83.26</td><td>98.03</td><td>99.05</td><td>-3.55</td><td>8.57</td><td>[-20.35, 13.25]</td><td></td><td>75.65</td><td>93.09</td></tr><tr><td>Proposed</td><td>0.41</td><td>3.74</td><td>[-6.93, 7.74]</td><td>87.95</td><td>98.48</td><td>99.01</td><td>-1.60</td><td>5.95</td><td>[-13.25, 10.06] | 70.75</td><td></td><td>94.36</td><td>98.11</td></tr></table>

<sup>†</sup> For Mean error, a value closer to zero indicates lower systematic bias. For 95% LoA, a narrower interval indicates better agreement.

Table 4 reports the comparison results. Among all compared methods, the proposed model achieves the best overall DBP agreement, with the lowest DBP SD of 3.74 mmHg and the narrowest DBP 95% LoA of [-6.93, 7.74] mmHg. Its cumulative accuracies are 87.95%, 98.48%, and 99.01% at the 5 mmHg, 10 mmHg, and 15 mmHg thresholds, respectively. The 10 mmHg accuracy is the highest among the compared methods, while the 5 mmHg and 15 mmHg accuracies remain within 0.26 and 0.04 percentage points of the corresponding best results. These results indicate that the proposed framework provides the strongest DBP agreement while maintaining highly competitive threshold-based accuracy.

For SBP estimation, XGBoost obtains the mean error closest to zero, with a bias of -0.60 mmHg. However, its SBP SD and LoA remain larger than those of the proposed model. The proposed model achieves the lowest SBP SD of 5.95 mmHg, the narrowest SBP 95% LoA of [-13.25, 10.06] mmHg, and the highest SBP cumulative accuracies of 70.75%, 94.36%, and 98.11% at the three error thresholds. This suggests that the proposed framework provides better agreement and stability for SBP prediction, even though its mean error is not the closest to zero.

The XGBoost baseline remains a strong statistical baseline, especially compared with standalone neural architectures. Nevertheless, the proposed model outperforms XGBoost in DBP and SBP SD and 95% LoA, while matching or improving its cumulative accuracy at all thresholds. Compared with XGBoost, the proposed model reduces the SBP SD from 7.10 mmHg to 5.95 mmHg, narrows the SBP LoA from [-14.52, 13.32] mmHg to [-13.25, 10.06] mmHg, and improves the SBP ≤ 10 mmHg accuracy from 87.23% to 94.36%. These improvements indicate that the hybrid design does not merely rely on the XGBoost prior, but further refines it through neural residual correction.

The KAN baseline achieves reasonable DBP performance, but its SBP performance is weaker, with a wider SBP LoA of [-20.35, 13.25] mmHg and lower cumulative accuracy. This suggests that KAN alone has nonlinear regression capability but is insuficient for stable SBP prediction when used as an independent predictor. Overall, the proposed model provides the most balanced performance across DBP and SBP by combining statistical prior estimation with conservative neural refinement.

Key Observations:

• Improved agreement: The proposed model achieves the narrowest DBP and SBP LoA intervals, indicating better agreement with the reference BP values.

• Stronger SBP stability: Compared with XGBoost, the proposed framework reduces SBP SD and substantially improves SBP cumulative accuracy, especially at the ≤ 10 mmHg threshold.

• Balanced hybrid prediction: Standalone neural models and single-branch baselines show less consistent behavior across DBP and SBP, whereas the proposed model maintains strong performance for both targets.

## 4.4 Ablation Study

To assess the conditional contribution of each major component within the complete framework, we used a leave-one-component-out (LOCO) ablation design. Starting from the full model, each reduced variant removes exactly one component while retaining the other two components and the same data split, preprocessing, target definitions, and training protocol. The Hybrid w/o KAN variant removes the KAN branch while retaining the XGBoost baseline/prior branch and MHDA-based fusion; the Hybrid w/o XGB variant removes the XGBoost branch while retaining KAN and MHDA; and the Hybrid w/o MHDA variant removes MHDA while retaining the XGBoost and KAN branches. The Proposed model contains all three components. This LOCO design evaluates the marginal degradation caused by removing a component from the complete system; it is not a sequential-addition or full-factorial analysis of all possible module interactions.

Table 5: Leave-one-component-out ablation study of the proposed hybrid baseline-residual framework. A filled circle indicates that a component is retained, whereas an en dash indicates that it is removed. Arrows indicate whether lower (↓) or higher (↑) values are better. The descriptive per-window 95% limits of agreement (LoA) are calculated as $\mathrm { M e a n } \pm 1 . 9 6$ SD without repeated-measures adjustment.
<table><tr><td rowspan="3">Configuration</td><td colspan="3">|Components retained</td><td colspan="6">DBP (mmHg)</td><td colspan="6">SBP (mmHg)</td></tr><tr><td colspan="2">XGB KAN</td><td>MHDA</td><td>Mean†</td><td colspan="2">Error Agreement SD ↓</td><td>95% LoA</td><td colspan="2">Cumulative Accuracy (%) ≤ 5 ↑ ≤ 10 ↑</td><td></td><td colspan="2">Error Agreement</td><td colspan="2">|Cumulative Accuracy (%)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>≤ 15 ↑</td><td>Mean†</td><td>SD ↓</td><td>95% LoA</td><td>≤ 5 ↑ ≤ 10 ↑</td><td>≤ 15 ↑</td></tr><tr><td>Hybrid w/o KAN</td><td></td><td></td><td></td><td>2.26</td><td>4.70</td><td>[-6.95, 11.47]</td><td>67.46 95.48</td><td>99.18</td><td></td><td>7.26 15.15</td><td>[-22.43, 36.95]</td><td>26.74</td><td>40.07</td><td>56.56</td></tr><tr><td>Hybrid w/o XGB</td><td></td><td></td><td></td><td>2.63</td><td>12.00</td><td>[-20.89, 26.15]</td><td>19.09 51.21</td><td></td><td>68.74</td><td>-4.29</td><td>23.43</td><td>[-50.21, 41.63] 13.74</td><td>31.18</td><td>41.83</td></tr><tr><td>Hybrid w/o MHDA</td><td></td><td></td><td></td><td>2.84</td><td>4.79</td><td>[-6.55, 12.23]</td><td>65.49 92.80</td><td></td><td>99.30</td><td>9.95 13.94</td><td>[-17.37, 37.27]</td><td>26.12</td><td>42.62</td><td>59.11</td></tr><tr><td>Proposed</td><td></td><td></td><td></td><td>0.41</td><td>3.74</td><td>[-6.93, 7.74]</td><td>87.95 98.48</td><td></td><td>99.01</td><td>-1.60 5.95</td><td>[-13.25, 10.06]</td><td>70.75</td><td>94.36</td><td>98.11</td></tr></table>

<sup>†</sup> For Mean error, a value closer to zero indicates lower systematic bias. For 95% LoA, a narrower interval indicates better agreement.

Table 5 summarizes the ablation results. Removing the XGBoost baseline/prior branch causes the most severe degradation. Relative to the full model, the Hybrid w/o XGB variant increases DBP SD by 8.26 mmHg, from 3.74 to 12.00 mmHg, and SBP SD by 17.48 mmHg, from 5.95 to 23.43 mmHg. The corresponding 95% LoA intervals become substantially wider, reaching [-20.89, 26.15] mmHg for DBP and [-50.21, 41.63] mmHg for SBP. This large expansion of the LoA intervals indicates poor agreement with the reference BP values and supports the role of the XGBoost branch as a statistical anchor in the hybrid residual framework.

The Hybrid w/o KAN variant also shows clear performance degradation, especially for SBP. Although the XGBoost baseline/prior branch is retained, removing KAN increases SBP SD by 9.20 mmHg, widens the SBP LoA to [-22.43, 36.95] mmHg, and reduces the SBP ≤ 10 mmHg accuracy by 54.29 percentage points, from 94.36% to 40.07%. This suggests that the KAN branch contributes to nonlinear physiological approximation and helps stabilize the residual correction process, particularly for SBP estimation.

The Hybrid w/o MHDA variant further demonstrates the importance of the MHDA module. After removing MHDA, the DBP SD increases by 1.05 mmHg, from 3.74 to 4.79 mmHg, and the DBP ≤ 10 mmHg accuracy decreases by 5.68 percentage points, from 98.48% to 92.80%. The degradation is more pronounced for SBP: the mean error increases to 9.95 mmHg, the SD increases to 13.94 mmHg, and the 95% LoA widens to [-17.37, 37.27] mmHg. The SBP ≤ 10 mmHg accuracy also decreases markedly from 94.36% to 42.62%. These results indicate that MHDA plays an important role in temporal feature interaction and in suppressing unstable residual corrections, especially for SBP estimation.

Compared with the reduced variants, the Proposed model achieves the narrowest LoA intervals for both DBP and SBP, with [-6.93, 7.74] mmHg for DBP and [-13.25, 10.06] mmHg for SBP. It also achieves the best SBP cumulative accuracy at all three thresholds and the best DBP ≤ 5 mmHg and ≤ 10 mmHg accuracies. Although the Hybrid w/o MHDA variant has a slightly higher DBP ≤ 15 mmHg accuracy, its bias, SD, LoA, and SBP performance are substantially worse. Therefore, the full framework provides the most stable overall performance by combining the XGBoost statistical anchor, KAN-assisted nonlinear modeling, MHDA-based feature interaction, and conservative residual refinement.

## Analysis of Results:

• Role of XGBoost: Removing the XGBoost branch leads to the widest LoA intervals and the largest drop in cumulative accuracy, indicating that the XGBoost prior is important for stabilizing the hybrid prediction framework.

• Role of KAN: Removing KAN substantially degrades SBP performance, suggesting that KAN helps model nonlinear physiological patterns and prevents unstable residual correction.

• Role of MHDA: Removing MHDA causes clear degradation in both DBP and SBP, especially in SBP SD, LoA, and threshold accuracy, indicating that MHDA is important for efective temporal feature interaction and robust residual correction.

• Overall stability: The proposed model does not achieve the best value in every single threshold-related column, but it provides the best overall agreement-oriented performance, especially in SD, LoA, and SBP cumulative accuracy.

## 4.5 Statistical Analysis and Visualization

To comprehensively evaluate the model’s reliability and dynamic tracking capability, we performed a detailed visualization analysis using the segment-level held-out test set and a separate extended held-out visualization set.

Agreement and Error Distribution: Fig. 4 presents the descriptive per-window agreement analysis. The Bland–Altman plots in the top row show mean biases of 0.41 mmHg for DBP and −1.60 mmHg for SBP, with unadjusted 95% LoA intervals of [-6.93, 7.74] mmHg and [-13.25, 10.06] mmHg, respectively. The error distributions in the middle row remain concentrated near zero with limited tails, consistent with the small overall biases. Furthermore, the regression scatter plots in the bottom row have high coeficients of determination $( R ^ { 2 } > 0 . 9 0 )$ between the estimated and reference values across the evaluated range. These plots do not account for repeated measurements within subjects.

Representative Continuous Tracking Cases: Beyond static statistical metrics, the ability to continuously follow blood pressure variations under diferent physiological ranges is essential for real-time monitoring. To provide a clearer visual assessment, Fig. 5 presents six representative segments from an extended held-out visualization set evaluated with the same trained model used for Table 4. The segments cover low, normal, and high blood-pressure ranges, with two examples shown for each range. Compared with plotting a long and dense sequence, these locally zoomed segments allow the agreement between reference and predicted SBP/DBP trajectories to be inspected more clearly.

As shown in Fig. 5, the proposed model closely follows the reference BP trajectories across diferent pressure levels. In the low-pressure cases, the model tracks both gradual trends and local fluctuations. In the normal-pressure cases, the predicted curves remain stable and well aligned with the reference values. In the high-pressure cases, the model maintains accurate tracking under elevated SBP/DBP ranges, indicating that the proposed framework is capable of capturing BP dynamics across a broad physiological spectrum.

![](images/f827222e845fc8f4aa531b1edde53a1be89e9e0376febfb75bd10b8b89bbfd8f.jpg)

![](images/5361f59a502e6053538b35a65da708b713d758350b12726b1c59d14067d13371.jpg)

![](images/76e4913a9f87644d7e3af2a30bf4371794e80ac87cd7b9d015634f200e0889d5.jpg)

![](images/0cf8df996c7e60ea31c8e178dae34c71c5bcf007818cd3e6c5ad95e0da360760.jpg)

![](images/69ac2cb336b57d80f17b2e17e91bb9c7a651c4981e4f9beb7eb7c55cfe73192c.jpg)

![](images/0f4ac4672b30e34b59d8754d8a766130c19ce31c5005c4ee492916cf8a8d90a4.jpg)  
Figure 4: Visualization of prediction performance. The top row presents descriptive per-window Bland–Altman plots showing agreement between predicted and reference BP. The dotted red lines represent the unadjusted 95% limits of agreement (±1.96 SD), and the dashed black line represents the mean bias; repeated measurements within subjects are not accounted for. The middle row presents error-distribution histograms with kernel density estimation (KDE) curves, showing that the prediction errors remain concentrated near zero. The bottom row presents scatter plots with high coeficients of determination $( R ^ { 2 } > 0 . 9 0 )$ between the estimated and reference values.

Low BP case | samples 24017-24116 | SBP MAE=0.25, DBP MAE=0.26  
![](images/d24dc8bee6caabec3fd0aef37d05783bbef6c73b27e901ad638e85d896eea23b.jpg)  
(a) Low BP case 1

Low BP case | samples 77860-77959 | SBP MAE=0.28, DBP MAE=0.30  
![](images/f1f54425f6e1856d0edc9ffbd1671f245d932702e697f379a1ffb39770cdee31.jpg)  
(b) Low BP case 2

Normal BP case | samples 2312-2411 | SBP MAE=0.28, DBP MAE=0.15  
![](images/e90cd58a2ffd393127250a6ff72ec63522434dfd6b9c2818eff2cf855b55beab.jpg)  
(c) Normal BP case 1

Normal BP case | samples 50624-50723 | SBP MAE=0.41, DBP MAE=0.19  
![](images/2bcf430bf61731b9ed9d980adcc660e9b596aa005766ee906b97a98b567142cb.jpg)  
(d) Normal BP case 2

High BP case | samples 13214-13313 | SBP MAE=0.79, DBP MAE=1.28  
![](images/a0c995fb71258da377e1955a3a3d8a6c64373f6ea7e4d0e64f3b21ec40b6e9e9.jpg)  
(e) High BP case 1

High BP case | samples 65268-65367 | SBP MAE=1.04, DBP MAE=1.10  
![](images/e439ed297a30ab129677ccedf3d1602d9f8f77434bb7271996c825d020f30818.jpg)  
(f) High BP case 2  
Figure 5: Representative continuous blood pressure tracking cases under diferent BP ranges. Six locally zoomed segments from an extended held-out visualization set are shown, including two low-BP cases, two normal-BP cases, and two high-BP cases. This set is separate from the test set summarized in Table 4, while the same trained model is used. Each case separately presents SBP and DBP trajectories within a local coordinate range, making trajectory-level agreement and short-term prediction errors easier to inspect. The predicted curves remain closely aligned with the reference trajectories across the selected low, normal, and high BP cases, illustrating consistent tracking behavior across diferent pressure ranges.

## 4.6 Numerical Comparison with BHS Thresholds

Finally, we compared the cumulative error percentages of the proposed method with the numerical thresholds defined by the British Hypertension Society (BHS). Table 6 separates the BHS reference percentages from the two outputs of the proposed model: the first row lists the minimum cumulative percentages associated with the numerical Grade A thresholds, whereas the subsequent rows report the DBP and SBP results produced by the same proposed model on the segment-level test set.

For DBP, the cumulative error percentages reach 87.95%, 98.48%, and 99.01% at the 5 mmHg, 10 mmHg, and 15 mmHg thresholds, respectively, all of which exceed the numerical Grade A percentage thresholds. For SBP, the model achieves 70.75%, 94.36%, and 98.11% at the same three thresholds, likewise falling within these numerical thresholds.

Thus, both BP targets exceed all three reference percentages. This is a descriptive threshold comparison rather than a formal BHS grade assignment because the evaluation did not follow a protocol-specific clinical validation design.

Table 6: BHS Grade A reference percentages and the corresponding cumulative error percentages of the proposed model for DBP and SBP. This descriptive comparison is not a formal BHS grade assignment.
<table><tr><td rowspan="2">Row type</td><td rowspan="2">BP target</td><td colspan="3">Cumulative Error Percentage (%)</td><td rowspan="2">Numerical comparison</td></tr><tr><td>≤5 mmHg</td><td>≤10 mmHg</td><td>≤15 mmHg</td></tr><tr><td>BHS Grade A reference</td><td></td><td>60.00</td><td>85.00</td><td>95.00</td><td>Reference threshold</td></tr><tr><td rowspan="2">Proposed model output</td><td>DBP</td><td>87.95</td><td>98.48</td><td>99.01</td><td>All three met</td></tr><tr><td>SBP</td><td>70.75</td><td>94.36</td><td>98.11</td><td>All three met</td></tr></table>

## 5 Discussion

## 5.1 Interpretation of Performance

The experimental results demonstrate that the proposed hybrid framework achieves the best overall agreement-oriented performance among the compared models for continuous blood pressure estimation. Unlike end-to-end waveform models (e.g., ResNet-1D) that rely solely on latent feature extraction, our method explicitly integrates six ECG/PPG-derived descriptors (PTT, heart rate, foot-to-peak interval, maximum PPG slope, RI, and SD time) with age and body weight. The ablation study confirms that the inclusion of the XGBoost branch and KAN network significantly reduces prediction bias, particularly for SBP, which typically exhibits higher variability than DBP.

## 5.2 Clinical Implications

From a clinical perspective, the model’s values fall within the numerical BHS Grade A percentage thresholds and numerical AAMI mean-error and SD criteria on the segment-level held-out test set, motivating formal clinical validation. These retrospective numerical comparisons do not constitute compliance testing, device validation, or certification. The tracking behavior observed in the selected hypotensive segments (Fig. 5) further motivates evaluation in ICU monitoring and wearable health settings.

## 5.3 Limitations and Future Work

Despite the promising results, this study has limitations. First, the model was validated on the MIMIC-III database, which consists of data from ICU patients. The signal quality in ambulatory settings (e.g., daily activities) may be lower due to motion artifacts, and the model’s robustness in such scenarios requires further validation. Second, although we used calibration-free features, adding periodic calibration could further enhance long-term accuracy. Future work will focus on compressing the model for deployment on edge devices (e.g., smartwatches) and validating it on real-world wearable datasets.

Third, the present evaluation retained the original segment-level split because subject identifiers were not available when the model-ready feature table and experimental partitions were created. Subject mapping was subsequently reconstructed from the source metadata and waveform archive for cohort accounting, but it was not used to retrospectively alter the established partitions or to claim subject-disjoint performance. Although windows from an individual segment were confined to one partition, segments from the same subject may therefore have appeared in diferent partitions. This can yield more optimistic estimates of generalization than a strictly subject-disjoint evaluation.

Future work should repeat model development and evaluation using a de novo subject-disjoint split defined before scaling and window construction, together with external-cohort validation.

Fourth, the original complete-case pipeline did not apply pre-specified physiological-range exclusions or RI clipping. Although the held-out test references contained no SBP ≤ DBP rows, the influence of the extreme values retained elsewhere in the model-ready table on training was not quantified through a retrained sensitivity analysis. Future evaluation should predefine physiologically justified quality-control rules and compare results with and without those exclusions.

Fifth, the 2,431 test windows include temporally adjacent and repeated observations from some subjects. The reported SD and Bland–Altman LoA treat windows as individual observations and do not include subject-clustered confidence intervals or a repeated-measures Bland–Altman model. They should therefore be interpreted as descriptive per-window agreement estimates. Subject-clustered bootstrap inference or repeated-measures agreement analysis should accompany the de novo subject disjoint evaluation.

## 6 Conclusion

In this paper, we proposed a multi-source data-driven framework for cufless continuous blood pressure monitoring. The method models standardized 10-step sequences of six ECG/PPG-derived physiological descriptors and two demographic covariates using a hybrid Transformer architecture (MSEM and DCFD), while XGBoost and KAN provide complementary tabular and nonlinear representations. Experimental results on the MIMIC-III dataset show that our model achieves the best overall agreement-oriented performance among the compared baselines and falls within the numerical AAMI mean-error and SD criteria and numerical BHS Grade A percentage thresholds on the segment-level test set. These numerical comparisons are descriptive and do not constitute formal compliance testing or device certification. The ablation study and visualization further demonstrate the contribution of the model components and its dynamic tracking behavior on the evaluated data. These findings should be confirmed using pre-specified quality-control rules, repeated-measures inference, de novo subject-disjoint evaluation, and an external cohort before clinical generalization is inferred.

## Ethical statement

This study was a secondary analysis of de-identified physiological and linked demographic data from the MIMIC-III Waveform Database and MIMIC-III Clinical Database. No participants were recruited and no new physiological or clinical data were collected by the authors. The secondary analysis was conducted in accordance with the principles of the Declaration of Helsinki and the applicable PhysioNet credentialing and data-use requirements. According to the database documentation, the original MIMIC-III project was approved by the Institutional Review Boards of Beth Israel Deaconess Medical Center and the Massachusetts Institute of Technology, and the requirement for individual patient consent was waived because the project did not afect clinical care and all protected health information was de-identified (Johnson, Pollard, and Mark, 2016; Johnson et al., 2016).

## Funding

This research received no external funding. The costs of the computational resources used in this study were supported by Zhaoying Liu.

## Acknowledgements

The authors used OpenAI Codex (GPT-5.6) to assist with English-language editing, manuscript organization, LaTeX formatting, and literature-search support. All AI-assisted content was critically reviewed and verified by the authors, who take full responsibility for the accuracy, integrity, and originality of the manuscript.

## Data availability

The MIMIC-III Waveform Database used in this study is available from PhysioNet (doi: https: //doi.org/10.13026/c2607m) (Moody et al., 2020). The linked clinical and demographic data from the MIMIC-III Clinical Database are available from PhysioNet (doi: https://doi.org/10.13026/ C2XW26) to credentialed users who complete the required training and accept the applicable data use agreement (Johnson, Pollard, and Mark, 2016). The authors are not permitted to redistribute restricted source records. The analysis code and non-identifying derived experimental data supporting the tables and figures are available from the corresponding author upon reasonable request, where disclosure is permitted by institutional requirements and the applicable PhysioNet data-use terms.

Conflict of interest The authors declare no competing interests.

## Author contributions

Yuexin Ma: Conceptualization, Methodology, Investigation, Project administration, and Writing– review and editing. Yuexin Ma conceived the study, designed the initial model architecture and methodological innovations, established the evaluation framework, led the literature investigation, and coordinated the research direction.

Jingqi Hou: Methodology, Software, Data curation, Investigation, Validation, Formal analysis, Visualization, Writing–original draft, and Writing–review and editing. Jingqi Hou constructed the model-ready dataset from the source data, conducted the main model training and performance optimization, performed the baseline comparisons and ablation experiments, analyzed the results, prepared the figures, and led manuscript drafting, revision, and adaptation to the journal requirements.

Yuxuan Kang: Methodology, Software, Investigation, Validation, Visualization, and Writing– original draft. Yuxuan Kang contributed to exploratory evaluation of candidate model components, hyperparameter selection, figure preparation, and manuscript drafting.

Zhaoying Liu: Methodology, Validation, Supervision, Resources, and Writing–review and editing. Zhaoying Liu provided methodological guidance, critically evaluated the initial experimental evidence and manuscript, identified issues that motivated model refinement and the repetition of key experiments, guided interpretation of the revised results, provided computational resources, and critically revised the manuscript for important intellectual content.

All authors reviewed and approved the final manuscript.

## References

Athaya T and Choi S 2021 An estimation method of continuous non-invasive arterial blood pressure waveform using photoplethysmography: a U-Net architecture-based approach Sensors 21 1867 (doi: 10.3390/s21051867)

Athaya T and Choi S 2022 Real-time cufless continuous blood pressure estimation using 1D Squeeze U-Net model: a progress toward mHealth Biosensors 12 655 (doi: 10.3390/bios12080655)

Batra P, Maan P and Kumar M 2026 FFDM-GAT-transformer: a fast FDM-based graph-attention and transformer framework for cufless blood pressure estimation using PPG signal Measurement 121638 (doi: 10.1016/j.measurement.2026.121638)

Block R C et al 2020 Conventional pulse transit times as markers of blood pressure changes in humans Sci. Rep. 10 16373 (doi: 10.1038/s41598-020-73143-8)

Chang P et al 2024 A transformer-based difusion probabilistic model for heart rate and blood pressure forecasting in intensive care unit Comput. Methods Programs Biomed. 246 108060 (doi: 10.1016/j.cmpb.2024.108060)

Chen T and Guestrin C 2016 XGBoost: a scalable tree boosting system Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining pp 785–94 (doi: 10.1145/2939672.2939785)

Chen Y, Xu F, Huang Z, He J and Feng Z 2025 Cufless blood pressure estimation from six wearable sensor modalities in multi-motion-state scenarios arXiv:2512.01653

Cheng Y, Li T, Zhang Z, Huang Z, Mei Z and Vai M I 2026 GHC-net: a Gramian angular field based hybrid CNN for cufless blood pressure classification using PPG signals Comput. Biol. Med. 206 111622 (doi: 10.1016/j.compbiomed.2026.111622)

Cisnal A, Podder I, Grossmann L, Dheman K, Elgendi M, Valgimigli M and Paez-Granados D 2026 Towards trustworthy AI-driven cufless blood pressure monitoring npj Digit. Med. (doi: 10.1038/s41746-026-02898-7)

Colvonen P J 2021 Response to: Investigating sources of inaccuracy in wearable optical heart rate sensors npj Digit. Med. 4 38 (doi: 10.1038/s41746-021-00408-5)

Ding X-R, Zhang Y-T, Liu J, Dai W-X and Tsang H K 2016 Continuous cufless blood pressure estimation using pulse transit time and photoplethysmogram intensity ratio IEEE Trans. Biomed. Eng. 63 964–72 (doi: 10.1109/TBME.2015.2480679)

Duan J, Xiong J, Li Y and Ding W 2024 Deep learning based multimodal biomedical data fusion: an overview and comparative review Inf. Fusion 112 102536 (doi: 10.1016/j.infus.2024.102536)

Heidari S, Babor T F, De Castro P, Tort S and Curno M 2016 Sex and Gender Equity in Research: rationale for the SAGER guidelines and recommended use Res. Integr. Peer Rev. 1 2 (doi: 10.1186/s41073-016-0007-6)

Hua J et al 2024 Wearable cufless blood pressure monitoring: from flexible electronics to machine learning Wearable Electron. 1 78–90 (doi: 10.1016/j.wees.2024.05.004)

Huber P J 1964 Robust estimation of a location parameter Ann. Math. Stat. 35 73–101 (doi: 10.1214/aoms/1177703732)

Ibtehaz N et al 2022 PPG2ABP: translating photoplethysmogram (PPG) signals to arterial blood pressure (ABP) waveforms Bioengineering 9 692 (doi: 10.3390/bioengineering9110692)

International Organization for Standardization 2018 ISO 81060-2:2018 Non-Invasive Sphygmomanometers—Part 2: Clinical Investigation of Intermittent Automated Measurement Type 3rd edn (Geneva: ISO) (available at: https://www.iso.org/standard/73339.html)

International Organization for Standardization 2022 ISO 81060-3:2022 Non-Invasive Sphygmomanometers—Part 3: Clinical Investigation of Continuous Automated Measurement Type 1st edn (Geneva: ISO) (available at: https://www.iso.org/standard/71161.html)

Jeong D U and Lim K M 2021 Combined deep CNN-LSTM network-based multitasking learning architecture for noninvasive continuous blood pressure estimation using diference in ECG-PPG features Sci. Rep. 11 13539 (doi: 10.1038/s41598-021-92997-0)

Johnson A, Pollard T and Mark R 2016 MIMIC-III Clinical Database version 1.4 (PhysioNet) RRID:SCR 007345 (doi: 10.13026/C2XW26)

Johnson A E W, Pollard T J, Shen L, Lehman L H, Feng M, Ghassemi M, Moody B, Szolovits P, Celi L A and Mark R G 2016 MIMIC-III, a freely accessible critical care database Sci. Data 3 160035 (doi: 10.1038/sdata.2016.35)

Kachuee M, Kiani M M, Mohammadzade H and Shabany M 2017 Cufless blood pressure estimation algorithms for continuous health-care monitoring IEEE Trans. Biomed. Eng. 64 859–69 (doi: 10.1109/TBME.2016.2580904)

Kamanditya B, Fuadah Y N, Mahardika T N Q et al 2024 Continuous blood pressure prediction system using Conv-LSTM network on hybrid latent features of photoplethysmogram (PPG) and electrocardiogram (ECG) signals Sci. Rep. 14 16450 (doi: 10.1038/s41598-024-66514-y)

Kasbekar R S, Ji S, Clancy E A et al 2023 Optimizing the input feature sets and machine learning algorithms for reliable and accurate estimation of continuous, cufless blood pressure Sci. Rep. 13 7750 (doi: 10.1038/s41598-023-34677-9)

Kendall A, Gal Y and Cipolla R 2018 Multi-task learning using uncertainty to weigh losses for scene geometry and semantics Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition pp 7482–91 (doi: 10.1109/CVPR.2018.00781)

Koenker R and Bassett G Jr 1978 Regression quantiles Econometrica 46 33–50 (doi: 10.2307/1913643)

Lai K, Wang X and Cao C 2024 A continuous non-invasive blood pressure prediction method based on deep sparse residual U-Net combined with improved squeeze and excitation skip connections Sensors 24 2721 (doi: 10.3390/s24092721)

Li Z, Lu T, Liu R et al 2025 FuSenseRing: an open-source platform for robust cufless blood pressure monitoring via multimodal sensor fusion and temperature-adaptive attention Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 9 192 (doi: 10.1145/3770689)

Liang C et al 2026 A conformal piezoelectric microsystem for demographic-adaptive and calibration-free cufless blood pressure monitoring Nat. Commun. 17 439 (doi: 10.1038/s41467-025-67118-4)

Lin T-Y, Goyal P, Girshick R, He K and Doll´ar P 2017 Focal loss for dense object detection Proc. IEEE Int. Conf. on Computer Vision pp 2980–88 (doi: 10.1109/ICCV.2017.324)

Liu Z, Wang Y, Vaidya S, Ruehle F, Halverson J, Soljaˇci´c M, Hou T Y and Tegmark M 2025 KAN: Kolmogorov–Arnold networks Proc. Int. Conf. on Learning Representations (available at: https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ afaed89642ea100935e39d39a4da602c-Abstract-Conference.html)

Liu Z, Qiao M, Liu Y, Zhang J and He L 2025 A two-branch ResNet-BiLSTM deep learning framework for extracting multimodal features applied to PPG-based cufless blood pressure estimation Sensors 25 3975 (doi: 10.3390/s25133975)

Ma G, Zhang J, Liu J, Wang L and Yu Y 2023 A multi-parameter fusion method for cufless continuous blood pressure estimation based on electrocardiogram and photoplethysmogram Micromachines 14 804 (doi: 10.3390/mi14040804)

Ma S, Wu Y, Peng C, Zhao Z and Wang H 2026 Continuous blood pressure prediction based on GAF-driven multimodal PPG and ECG signal fusion Measurement 257 118738 (doi: 10.1016/j.measurement.2025.118738)

McEvoy J W et al 2024 2024 ESC Guidelines for the management of elevated blood pressure and hypertension Eur. Heart J. 45 3912–4018 (doi: 10.1093/eurheartj/ehae178)

Mehrabadi M A, Aqajari S A H, Zargari A H A, Dutt N and Rahmani A M 2022 Novel blood pressure waveform reconstruction from photoplethysmography using cycle generative adversarial networks Proc. 44th Annu. Int. Conf. of the IEEE Engineering in Medicine and Biology Society pp 1906–09 (doi: 10.1109/EMBC48229.2022.9871962)

Mehta S, Kwatra N, Jain M and McDuf D 2024 Examining the challenges of blood pressure estimation via photoplethysmogram Sci. Rep. 14 18318 (doi: 10.1038/s41598-024-68862-1)

Meidert A S and Saugel B 2018 Techniques for non-invasive monitoring of arterial blood pressure Front. Med. 4 231 (doi: 10.3389/fmed.2017.00231)

Min S, An J, Lee J H et al 2025 Wearable blood pressure sensors for cardiovascular monitoring and machine learning algorithms for blood pressure estimation Nat. Rev. Cardiol. 22 629–48 (doi: 10.1038/s41569-025-01127-0)

Moody B, Moody G, Villarroel M, Cliford G D and Silva I 2020 MIMIC-III Waveform Database version 1.0 (PhysioNet) (doi: 10.13026/c2607m)

Moulaeifard M, Charlton P H and Strodthof N 2025 Generalizable deep learning for photoplethysmography-based blood pressure estimation—a benchmarking study Mach. Learn.: Health 1 010501 (doi: 10.1088/3049-477X/ae01a8)

Mukkamala R et al 2015 Toward ubiquitous blood pressure monitoring via pulse transit time: theory and practice IEEE Trans. Biomed. Eng. 62 1879–1901 (doi: 10.1109/TBME.2015.2441951)

Muntner P, Shimbo D, Carey R M et al 2019 Measurement of blood pressure in humans: a scientific statement from the American Heart Association Hypertension 73 e35–e66 (doi: 10.1161/HYP.0000000000000087)

Nawaz M W, Tahir M A, Mehmood A, Rahman M M U, Riaz K and Abbasi Q H 2024 Cuf-less arterial blood pressure waveform synthesis from single-site PPG using Transformer and frequency-domain learning arXiv:2401.05452 (doi: 10.48550/arXiv.2401.05452)

O’Brien E et al 1993 Short report: an outline of the revised British Hypertension Society protocol for the evaluation of blood pressure measuring devices J. Hypertens. 11 677–79 (doi: 10.1097/00004872-199306000-00013)

Pankaj, Maan P, Kumar M, Kumar A and Komaragiri R 2026 Cufless monitoring of blood pressure using photoplethysmography signal: a comprehensive review of artificial intelligence and edge computing solutions Arch. Comput. Methods Eng. 33 3837–66 (doi: 10.1007/s11831-025-10415-4)

Pilz N et al 2024 Cuf-based blood pressure measurement: challenges and solutions Blood Press. 33 2402368 (doi: 10.1080/08037051.2024.2402368)

Rastegar S, Gholam Hosseini H and Lowe A 2023 Hybrid CNN-SVR blood pressure estimation model using ECG and PPG signals Sensors 23 1259 (doi: 10.3390/s23031259)

Slapniˇcar G, Mlakar N and Luˇstrek M 2019 Blood pressure estimation from photoplethysmogram using a spectro-temporal deep neural network Sensors 19 3420 (doi: 10.3390/s19153420)

Song J, Liu Z, Huang Y and Wu X 2026 LAST-CBPM: photoplethysmography-based quasi-continuous blood pressure monitoring algorithm Biomed. Signal Process. Control 120 109723 (doi: 10.1016/j.bspc.2026.109723)

Stahlschmidt S R, Ulfenborg B and Synnergren J 2022 Multimodal deep learning for biomedical data fusion: a review Brief. Bioinform. 23 bbab569 (doi: 10.1093/bib/bbab569)

Tang H et al 2024 Blood pressure estimation based on PPG and ECG signals using knowledge distillation Cardiovasc. Eng. Technol. 15 39–51 (doi: 10.1007/s13239-023-00695-x)

Tang L, Lin W-H, Zheng D and Chen F 2026 Enhancing few-shot personalized cufless blood pressure estimation with self-supervised learning Physiol. Meas. 47 035012 (doi: 10.1088/1361-6579/ae52a1)

Tanveer M S and Hasan M K 2019 Cufless blood pressure estimation from electrocardiogram and photoplethysmogram using waveform based ANN-LSTM network Biomed. Signal Process. Control 51 382–92 (doi: 10.1016/j.bspc.2019.02.028)

Tian Z, Liu A, Zhu G and Chen X 2025 A paralleled CNN and Transformer network for PPG-based cuf-less blood pressure estimation Biomed. Signal Process. Control 99 106741 (doi: 10.1016/j.bspc.2024.106741)

Tian Z, Liu A, Chen J, Wang D and Chen X 2026 PPG-based continuous arterial blood pressure estimation via multi-scale cross attention fusion Biomed. Signal Process. Control 113 108833 (doi: 10.1016/j.bspc.2025.108833)

Vaswani A et al 2017 Attention is all you need Advances in Neural Information Processing Systems 30 5998–6008

Weber-Boisvert G, Gosselin B and Sandberg F 2023 Intensive care photoplethysmogram datasets and machine-learning for blood pressure estimation: generalization not guarantied Front. Physiol. 14 1126957 (doi: 10.3389/fphys.2023.1126957)

Ye T, Dong L, Xia Y, Sun Y, Zhu Y, Huang G and Wei F 2025 Diferential Transformer Proc. Int. Conf. on Learning Representations (available at: https://proceedings.iclr.cc/paper\_files/ paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)

Zhao L et al 2023 Emerging sensing and modeling technologies for wearable and cufless blood pressure monitoring npj Digit. Med. 6 93 (doi: 10.1038/s41746-023-00835-6)

Zheng Y et al 2025 UTransBPNet for cufless and calibration-free blood pressure estimation under dynamic conditions Sci. Rep. 15 17654 (doi: 10.1038/s41598-025-02963-3)

Zhou B, Perel P, Mensah G A et al 2021 Global epidemiology, health burden and efective interventions for elevated blood pressure and hypertension Nat. Rev. Cardiol. 18 785–802 (doi: 10.1038/s41569-021-00559-8)