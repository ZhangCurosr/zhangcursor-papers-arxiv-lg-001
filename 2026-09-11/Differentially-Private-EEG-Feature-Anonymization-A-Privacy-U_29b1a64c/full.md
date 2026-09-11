# Differentially Private EEG Feature Anonymization: A Privacy–Utility Case Study in Clinical Neurophysiology

Noman Sadiq <sup>ID</sup>

Mohsen Toorani

Department of Science and Industry Systems University of South-Eastern Norway Kongsberg, Norway

## Abstract

Clinical electroencephalography (EEG) data are valuable for healthcare research and for developing artificial intelligence (AI)-based clinical decision-support systems, but EEG recordings and derived features may contain sensitive patient-specific information. This creates privacy risks when data are reused, analyzed, or shared across clinical and research environments. Conventional anonymization methods are often insufficient for high-dimensional biomedical signals, since removing direct identifiers does not necessarily prevent re-identification, linkage, or inference risks. At the same time, strong privacy protection may distort clinically relevant signal characteristics and reduce data utility. This paper studies subject-level differential privacy for protecting clinical EEG-derived feature representations using Gaussian and Laplace perturbations. The proposed framework considers three deployment scenarios: client-side anonymization, centralized server-side anonymization, and decentralized local training. Following EEG preprocessing and feature extraction, Gaussian and Laplace perturbations are applied to the resulting patient-level EEG feature representations. The Laplace experiments evaluate the implemented noise scales, while the scales required for formal full-vector calibration are derived separately. The effects of both perturbations are assessed using statistical utility measures and a downstream machine-learning-based utility check. The results show that differentially private perturbation can be integrated into EEG processing workflows, but the selected mechanism, privacy parameters, and sensitivity calibration strongly influence data utility. The study highlights the practical privacy–utility trade-off in DP-based EEG feature anonymization and the challenges of preserving downstream utility in small and imbalanced clinical EEG datasets.

## 1 Introduction

Healthcare increasingly depends on digital technologies for collecting, storing, and analyzing patient data to improve efficiency, accuracy, and quality of care. At the same time, the growing collection and sharing of healthcare data creates serious privacy challenges. Patient records are highly sensitive, and healthcare institutions are under increasing pressure not only to use data for direct treatment but also for secondary purposes such as clinical research, decision support, and training of artificial intelligence (AI) models. This creates a need for privacy-preserving data sharing mechanisms that can support innovation while still meeting legal and ethical requirements. If privacy protection is weak, patient trust may be reduced, and organizations may become more reluctant to share valuable data. On the other hand, if privacy protection is too strict, important medical knowledge may be lost. Therefore, the balance between privacy protection and data utility has become a central problem in healthcare informatics [1].

This challenge is especially important for clinical neurophysiological data such as EEG (Electroencephalography) and electromyography (EMG). Unlike many traditional clinical datasets, EEG and EMG data are not only tabular records but also high-dimensional time-series signals collected over longer periods for each patient. These signals contain complex temporal and frequency patterns that are important for diagnosis and machine learning tasks, but they may also reveal sensitive neurological and medical information. As a result, privacy protection for such data is more difficult than simply removing names or patient identifiers [2]. Traditional de-identification models such as k-anonymity seek to make each record indistinguishable from at least k − 1 other records with respect to selected quasi-identifiers [3].

Traditional anonymization methods, including suppression, generalization, and k-anonymity-based models, provide useful baselines for privacy-preserving data publishing, but they are often insufficient for high-dimensional biomedical signals. Such methods may remain vulnerable to re-identification, linkage, or attribute-inference risks and may also reduce the utility of time-series data when strong generalization or suppression is applied. Differential Privacy (DP) [4, 5] offers a stronger formal framework by adding calibrated randomness to data or query outputs, thereby limiting the influence of an individual patient on the released result. In the present work, DP is considered at the subject level and is applied to EEG-derived feature representations rather than directly to raw EEG recordings. This design is motivated by the need to support secondary analysis and machine-learning use cases while reducing patient-level disclosure risk.

This paper focuses on privacy-preserving processing and sharing of clinical EEG data using Gaussian and Laplace perturbations within a subject-level DP framework. The study evaluates how these perturbations affect data utility and downstream machine learning performance in a clinical EEG case study. The main contributions of this paper are as follows:

• A clinical EEG data anonymization case study is presented using subject-level DP applied to EEG-derived feature representations.

• Three deployment scenarios are considered for multi-hospital EEG processing: client-side anonymization, centralized server-side anonymization, and decentralized local training.

• Gaussian and Laplace perturbation mechanisms are implemented and compared after EEG preprocessing and feature extraction, together with a separate derivation of the full-vector $L _ { 1 }$ sensitivity and corresponding Laplace scales required for formal calibration.

• The impact of the selected Gaussian and Laplace configurations is assessed using statistical utility measures and a downstream machine-learning-based utility check.

• Practical challenges are discussed for applying formal privacy mechanisms to small and imbalanced clinical EEG datasets.

The scope of the study is limited to selected DP configurations and EEG-derived feature representations. The classification experiment is used as a downstream utility check rather than as the main objective of the paper. The results should therefore be interpreted in light of the limited dataset size, class imbalance, selected sensitivity/noise calibrations, and the absence of extensive empirical attack-based privacy testing. Larger datasets, broader privacy-budget exploration, rigorous privacy accounting, dedicated re-identification, linkage, membership-inference, and reconstruction analyses, and broader model benchmarking are left for future work.

## 2 Related Work

Privacy-preserving EEG analysis is an emerging research area. EEG data can reveal subject identity, neurological characteristics, cognitive states, and other sensitive information, creating risks in brain-computer interfaces, clinical neurophysiology, and collaborative model training. These risks are especially important in healthcare settings where EEG data may need to be reused for research, shared between institutions, or used for machine learning model development. Existing work on privacy-preserving EEG analysis can be grouped into three partly overlapping directions: formal privacy mechanisms such as differential privacy, federated or distributed learning approaches, and representation-learning methods that aim to suppress identity-related information while preserving task utility.

Several studies have applied differential privacy to healthcare and medical data analysis. Subramanian [6] studied the Laplace mechanism for healthcare data and discussed how noise affects analytical accuracy. Sun et al. [7] proposed a DP-based medical data publishing and training framework using normalization, weighted Laplace noise, gradient clipping, and Gaussian noise to improve privacy while preserving utility. Letafati and Otoum [8] considered distributed healthcare settings in which local models are clipped and perturbed with Gaussian noise before aggregation. These studies show that DP can provide a formal privacy framework for medical data analysis, but they also demonstrate the central challenge of balancing privacy protection against data utility. Recent work on medical deep learning has further examined the methodological choices, privacy–utility trade-offs, and deployment implications of differential privacy in healthcare applications [9]. It is important to distinguish DP applied to a released feature table from DP applied during model optimization. DP-SGD, for example, clips and perturbs per-example gradients and accounts for the privacy loss accumulated over repeated optimization steps, whereas the present study examines a one-time patient-level feature release [10, 11]. These settings protect different released objects and require different sensitivity and privacy-accounting arguments. More broadly, decentralized health-data sharing involves several complementary security and privacy requirements beyond perturbation of the released data, including access control, accountability, consent management, and protection during cross-institutional collaboration [12].

Machine learning methods are central to EEG analysis because EEG signals are high-dimensional, non-stationary, and subject-dependent. In privacy-preserving EEG research, machine learning is used both for classification and for evaluating whether transformed EEG data retain useful task information. Luo et al. [13] applied federated learning with differential privacy to EEG-based epilepsy recognition. Their framework keeps raw EEG data on local clients, extracts EEG features locally, trains local models, and shares only model parameters with the server. Privacy is supported through gradient clipping and Gaussian noise added to model parameters before aggregation. This work is closely related to the present paper because it combines EEG-based epilepsy recognition, federated learning, and DP. However, it focuses on model-parameter perturbation in a federated learning setting, whereas the present paper evaluates Gaussian and Laplace perturbation of patient-level EEG-derived feature representations. More generally, federated learning enables participating institutions to train a shared model while retaining their raw clinical data locally [14]. Nevertheless, gradients and model updates can leak information about the underlying training data through attacks such as gradient inversion [15]. Secure aggregation addresses part of this risk by allowing the aggregation server to recover an aggregate of the participating clients’ updates without observing each individual update in plaintext [16]. In a clinical EEG setting, Rajabi and Toorani [17] implemented and evaluated masking-based secure aggregation with dropout recovery, malicious-setting safeguards, and auxiliary-notary-based verifiability for cross-silo federated learning.

Another line of work focuses on making EEG data less identifiable while preserving task performance. Meng et al. [18] studied sample-wise and user-wise perturbations that add near-imperceptible noise to EEG data in order to reduce identity leakage while maintaining BCI task utility. Chen et al. [19] also studied user-wise perturbations for identity protection in EEG-based BCI systems, including random, synthetic, error-minimizing, and error-maximizing perturbations. These approaches are useful for reducing identity leakage against specific models or attack settings, but they do not provide the same type of formal privacy guarantee as differential privacy.

Representation-learning approaches have also been proposed for privacy-aware EEG analysis. Singh et al. [20] used a multi-objective autoencoder to suppress subject identity while maintaining task recognition. Wang et al. [21] proposed an identity-removal network that separates task-related and identity-related EEG features and removes identity-related components before decoding. Bethge et al. [22] introduced domain-aligned private encoders for EEG-based emotion recognition, where private encoders and a domain-alignment loss are used to support cross-dataset learning while reducing domain-specific information. These methods are important because they address the fact that EEG contains identity-related structure, but most of them focus primarily on empirical identity suppression rather than formal DP accounting.

Privacy-preserving and data-sharing issues in EEG and BCI have also been studied from a broader methodological perspective. Xia et al. [23] proposed augmentation-based source-free adaptation for motor-imagery EEG classification, where the source EEG data are not shared during adaptation to a target user. Although this method is not a DP mechanism, it is relevant because it addresses the practical problem of adapting EEG models without transferring raw EEG data. Xia et al. [24] reviewed privacy protection in brain-computer interfaces, with a strong focus on EEG data. They summarized data-level and model-level privacy risks and discussed defense strategies such as anonymization, cryptography, differential privacy, federated learning, and transfer learning. Their review highlights the need for clearer privacy–utility evaluation protocols and more standardized ways of comparing privacy-preserving EEG methods.

Compared with these works, the present paper focuses on a clinical EEG case study in which Gaussian and Laplace perturbation mechanisms are applied to EEG-derived patient-level feature representations and assessed using both statistical utility measures and a downstream machine-learning-based utility check. The emphasis is not on proposing a new DP mechanism or a new EEG classifier, but on examining the practical consequences of subject-level DP perturbation in a multi-hospital EEG processing setting. This focus differs from user-wise perturbation and identity-removal methods because the present work explicitly studies DP noise calibration, patient-level adjacency, sensitivity assumptions, and the resulting privacy–utility trade-off for EEG-derived clinical features.

## 3 Proposed Scheme

A scenario-based privacy-preserving EEG processing scheme is proposed for multi-hospital clinical neurophysiology settings. The scheme is instantiated through three deployment scenarios that differ in where anonymization, aggregation, and model training are performed: client-side anonymization, centralized server-side anonymization, and decentralized local training. Across all scenarios, the common objective is to transform raw EEG recordings into protected EEG-derived feature representations using preprocessing, feature extraction, and subject-level DP-based perturbation, while limiting the exposure of raw signals and patient metadata.

In Scenario 1, depicted in Figure 1, client-side anonymization is performed locally at each hospital before protected feature representations are transferred to the server for aggregation or analysis. In Scenario 2, depicted in Figure 2, pseudonymized EEG-derived feature data are transferred to a trusted server, where server-side anonymization, aggregation, duplicate detection and model training are performed. In Scenario 3, depicted in Figure 3, each hospital keeps its EEG data and extracted features locally, applies anonymization within its own environment, and shares only model-related outputs or updates for decentralized learning.

![](images/3572fb961092f8135d2e3e1c66422cbc238caf89b6391f4d22ad538e07acc281.jpg)  
Figure 1: High-level architecture for client-side anonymization (Scenario 1).

![](images/fd3f75513bf0e8556b596d582e0c0af5cc610d63b3e5a5fcbc44830d1409dade.jpg)  
LK = Linkage Key share from Key Management Service  
AD = Anonymized Data share from hospital

Figure 2: High-level architecture for centralized server-side anonymization (Scenario 2).  
![](images/8983e9cffed3240e8dbb1f8ec7ab99e562b2dba4626c522ce4b0a2bc909942da.jpg)  
Figure 3: High-level architecture for decentralized local training (Scenario 3).

In Scenario 1, the server receives only protected feature representations, while raw EEG data and patient identifiers remain within the local hospital environment. This provides stronger data minimization because the server does not receive identifiable or raw EEG data. A common challenge in multi-hospital settings is that the same patient may appear in more than one hospital. Duplicate patient records can bias aggregation or model training because repeated records give some patients more influence than others. To address this issue, the proposed scheme uses deterministic privacy-preserving linkage. In centralized data-sharing scenarios, a

Key Management Service (KMS) distributes a shared linkage key to participating hospitals. Each hospital computes a consistent pseudonym from a canonically encoded shared patient identifier using HMAC-SHA256 [25]:

$$
\mathrm { G l o b a l I D } = \mathrm { H M A C - S H A 2 5 6 } _ { L K } \left( \mathrm { E n c o d e } ( \mathrm { P a t i e n t I D } ) \right) ,\tag{1}
$$

where LK is the secret linkage key provided by the KMS and Encode(·) denotes an agreed canonical representation of the shared patient identifier. When participating hospitals possess the same underlying identifier and apply the same encoding, the same patient receives the same GlobalID across hospitals, allowing exact duplicate detection without transmitting the direct identifier.

The GlobalID is used only as an internal linkage identifier for duplicate detection and record management inside the trusted processing environment. It is not included in the differentially private feature table and is not released to downstream researchers. After duplicate records have been identified and resolved, the linkage identifier is separated from the DP-protected analytical dataset. This separation is important because releasing stable pseudonyms together with noisy features could still preserve linkage or membership information across datasets. In this paper, pseudonymization refers to replacing direct identifiers with linkage values, whereas DP-based anonymization refers to randomized perturbation of the patient-level feature representation. The HMAC-based GlobalID and Bloom-filter components support record linkage and do not themselves provide differential privacy.

In Scenario 2, anonymization is performed inside the trusted server environment rather than at each hospital. Each participating hospital retains the raw patient identifiers locally, performs basic EEG preprocessing, generates a pseudonymized patient identifier, and transmits pseudonymized EEG-derived feature data to the central server via a secure communication channel. The trusted server then performs aggregation, duplicate detection, DP-based anonymization, and model training.

In Scenario 3, raw EEG data and extracted feature vectors remain within each hospital. Instead of sending patient-level EEG data to the central server, each hospital trains a local model using its own anonymized EEG feature dataset and the privacy/model parameters shared by the central coordinator. The main motivation for this scenario is to reduce direct data sharing between hospitals and the central server. Each hospital performs EEG preprocessing, feature extraction, subject-level anonymization, and local model training within its own secure environment. The server receives only model-related outputs or updates, which can then be aggregated to compute a global model. A common aggregation method is federated averaging [26]:

$$
w ^ { ( t + 1 ) } = \sum _ { h = 1 } ^ { H } \frac { n _ { h } } { N } w _ { h } ^ { ( t + 1 ) } ,\tag{2}
$$

where H is the number of hospitals, $n _ { h }$ is the number of local training samples at hospital h, $\begin{array} { r } { N = \sum _ { h = 1 } ^ { H } n _ { h } } \end{array}$ is the total number of samples, and $w _ { h } ^ { ( t + 1 ) }$ is the locally updated model of hospital h in round $t + 1$ . Differential privacy is applied locally within each hospital before model training. Each hospital preprocesses the raw EEG data, extracts the relevant feature vectors, and applies subject-level DP noise to these extracted features inside its own secure environment. The DP-protected features are then used only for local model training and are not released to the central server or to other hospitals. The central server receives only model-related updates for aggregation. Therefore, the intended privacy mechanism in this scenario is feature-level DP before local training, not DP applied directly to gradients or model updates. This distinction is important because feature-level DP and gradient-level DP require different privacy accounting.

Figure 3 includes a Bloom-filter-based component for privacy-preserving record linkage across hospitals. This component supports approximate linkage when patient-identifying fields may contain spelling, typographical, or formatting variations that prevent exact matching. In this approach, normalized identifying fields are divided into character q-grams and encoded as Bloom filters, which can then be compared without transmitting the direct identifiers in plaintext [27]. Bloom-filter encodings, however, do not by themselves provide anonymization and may be vulnerable to frequency and cryptanalytic attacks. A practical deployment therefore requires keyed hashing, carefully selected encoding and matching parameters, and an explicit threat model [28]. The HMAC construction described above supports exact linkage when hospitals possess the same stable patient identifier, whereas the Bloom-filter component supports approximate linkage when identifying information is not represented identically across hospitals.

## 3.1 EEG Processing

The proposed scheme uses two EEG processing and anonymization workflows. The first workflow, shown in Figure 4, is used in Scenarios 1 and 2, where anonymized EEG-derived features are prepared for data sharing, aggregation, or analysis and are assessed through intrinsic utility measures. The second workflow, shown in Figure 5, is used in Scenario 3, where anonymized EEG-derived features remain local and are used for local model training and downstream utility assessment.

The workflow in Figure 4 consists of four main stages. First, raw EEG data are preprocessed through filtering and artifact reduction to improve signal quality by removing noise such as baseline drift, muscle

![](images/9bddf532ac0120a92b64ba212ac93e7a88c084b9edb638124c16b58379ee118f.jpg)  
Figure 4: Proposed workflow for EEG processing, DP anonymization, and intrinsic utility assessment in Scenarios 1 and 2.

![](images/5f47167441321b0fc61c90fe984844b9d612d4c364c349329a2f69adac296ff5.jpg)  
Figure 5: Proposed workflow for EEG processing, DP anonymization, and downstream utility assessment through local model training in Scenario 3.

activity, and eye-blink artifacts. Second, feature extraction converts the cleaned signals into compact statistical and frequency-based representations. Third, DP-based perturbation is applied by adding calibrated noise to the feature vectors to protect patient-level information. Finally, intrinsic utility evaluation compares the anonymized and original features using statistical and signal-based measures to assess whether useful data characteristics are preserved.

## 3.1.1 Decentralized EEG Processing and Local Model Training

The decentralized workflow extends the same EEG processing and anonymization stages to a local model-training setting. In this scenario, each hospital keeps the EEG data within its local environment. Raw data are preprocessed, features are extracted, and anonymization is applied locally. The anonymized EEG dataset is then used to train a local model. The central server does not receive raw EEG data or subject-level EEG feature data; instead, it receives only trained local model outputs or updates.

The workflow is shown in Figure 5. The main processing stages remain preprocessing, feature extraction, and DP-based anonymization. The key difference is the evaluation stage: instead of only checking the intrinsic quality of anonymized features, the decentralized workflow evaluates utility through model performance. A benchmark model is first trained using the original extracted EEG features, and another model is trained using the anonymized EEG features. The performance of these two models is then compared to estimate how much downstream utility is preserved after anonymization.

## 3.2 EEG Data Processing Stages

The EEG processing stage prepares raw EEG recordings for anonymization and model training. The input data may come from multiple sources and are assumed to be stored in formats such as MAT or HDF5. Each patient file contains multichannel EEG data arranged as channels by samples. A consistent recording structure is important because it allows the same preprocessing and feature extraction procedure to be applied across patients.

Preprocessing: EEG preprocessing removes noise and improves signal comparability across patients, channels, and recording sessions. In this work, preprocessing includes band-pass filtering, normalization, and segmentation. The implemented pipeline used a 0.5–70 Hz band-pass filter, a 50 Hz notch filter $( Q = 3 0 )$ , average re-referencing, and 2-s windows with 50% overlap. Window-level features were averaged separately for each patient to obtain one 363-dimensional patient-level vector. Before clipping and noise addition, the patient-level vectors were stacked into a patient-by-feature matrix and robustly scaled feature-wise using the median and interquartile range calculated across the patients. This normalization was performed after patient-level aggregation.

Band-pass filtering retains relevant EEG frequency components while removing low-frequency drift and high-frequency noise. Segmentation divides continuous EEG recordings into fixed-length epochs for consistent feature extraction. After patient-level aggregation, robust scaling reduces differences in scale among the feature dimensions before clipping and perturbation.

After preprocessing, feature extraction transforms the high-dimensional EEG time series into compact numerical feature vectors. In this work, frequency-domain features are derived using Power Spectral Density (PSD), which estimates how signal power is distributed across the main EEG bands: δ, θ, α, β, and $\gamma .$ The PSD was estimated using Welch’s averaged-periodogram method [29] with a Hann window, a segment length of 256 samples, 50% overlap (128 samples), and an FFT length of 256 samples. These band-power features summarize key neural activity patterns and provide a compact representation of the signals. The 363-dimensional feature representation was constructed from 33 EEG channels using 11 per-channel features: line length, root mean square, variance, zero-crossing rate, δ band power, θ band power, α band power, $\beta$ band power, γ band power, high-γ band power, and the 95% spectral edge frequency (SEF95). The frequency bands were defined as $\delta = 0 . 5 – 4$ Hz, $\theta = 4 { - } 8$ Hz, $\alpha = 8 – 1 3$ Hz, $\beta = 1 3 – 3 0$ Hz, $\gamma = 3 0 – 4 5$ Hz, and high-γ = 45–70 Hz. The resulting feature vectors are then used as input to the DP mechanism, allowing anonymization while preserving an interpretable privacy–utility trade-off. DP is applied to EEG-derived features rather than directly to raw EEG time-series signals because feature representations provide a more structured and bounded numerical form for sensitivity control, noise calibration, and utility assessment.

![](images/0fc7bd6e4e51fa8307bb5a808478245de35d1882e5f618a1b95570d47ff3b565.jpg)  
Figure 6: Subject-level anonymization of patient EEG feature data.

The utility is assessed in two ways. First, intrinsic utility is assessed by comparing the original and anonymized feature data using root mean square error (RMSE), mean absolute error (MAE), correlation, and signal-to-noise ratio (SNR). Lower RMSE and MAE, together with higher correlation and SNR, indicate better preservation of the original feature representation. Second, in the decentralized model-training workflow, utility is evaluated through downstream model performance by comparing a model trained on original EEG features with a model trained on anonymized EEG features. This combined evaluation helps determine whether the anonymized EEG data remain suitable for later machine-learning tasks.

## 3.3 Subject-Level DP

The anonymization stage applies subject-level DP to the extracted EEG feature vectors. The privacy setting is limited to subject-level DP, where the contribution of one patient is protected according to the selected adjacency relation and privacy parameters. Subject-level privacy is selected because the goal is to protect the contribution of an entire patient rather than a single EEG epoch or segment. Figure 6 illustrates the application of subject-level DP to patient EEG feature data.

For the formal description, let $D = ( x _ { 1 } , \ldots , x _ { n } ) $ denote a fixed-size table of patient-level EEG feature vectors, where each $x _ { i } \in \mathbb { R } ^ { d }$ represents the features extracted for one patient. This paper uses a bounded, replace-one subject-level adjacency relation: two datasets D and $D ^ { \prime }$ are neighboring if they contain the same number of patients and differ in the feature vector of exactly one patient. In this implementation, the privacy unit is one patient. The pipeline constructs one aggregated EEG feature vector for each patient after preprocessing and feature extraction. Differential privacy is therefore applied to patient-level feature vectors, not to individual EEG files, epochs, segments, or sliding windows. If a patient has multiple recordings or epochs, these are first summarized into a single patient-level representation before the DP mechanism is applied. Consequently, the neighboring datasets in this paper differ by the complete feature contribution of one patient, which is consistent with the bounded, replace-one subject-level adjacency relation used in the formal definition.

A randomized mechanism M satisfies $( \varepsilon , \delta )$ -differential privacy under this adjacency relation if, for all neighboring datasets D, D<sup>′</sup> and all measurable output sets S,

$$
\operatorname* { P r } [ M ( D ) \in S ] \leq e ^ { \varepsilon } \operatorname* { P r } [ M ( D ^ { \prime } ) \in S ] + \delta .\tag{3}
$$

The parameter ε controls the privacy loss, while δ denotes a small probability of failure. Smaller values of ε provide stronger privacy but require more noise, which can reduce feature utility.

The mechanism considered in this paper operates on a patient-level EEG feature table. In the implementation, the privacy unit is one patient, and each patient is represented by one feature vector after preprocessing and feature extraction. The resulting feature dimension is $d = 3 6 3$ . The implementation follows a bounded, replace-one subject-level adjacency relation, where two neighboring datasets contain the same number of patients and differ in the complete feature contribution of exactly one patient.

Before noise addition, the patient-level vectors were assembled into a matrix with 363 feature columns. Each value was centered by the corresponding feature-wise median and divided by the corresponding interquartile range, both calculated across patients. Interquartile ranges less than or equal to $1 0 ^ { - 9 }$ were replaced with 1.0 before division. The scaling statistics were fitted within each invocation of the perturbation routine and remained unchanged during its clipping and noise-addition steps. Because these statistics were calculated from the private patient dataset, the sensitivity bounds below assume a fixed scaling transformation and do not establish an end-to-end privacy guarantee for the complete preprocessing pipeline.

Each scaled vector is then clipped to a fixed $L _ { 2 }$ -norm threshold C:

$$
{ \bar { x } } _ { i } = x _ { i } \cdot \operatorname* { m i n } \left( 1 , { \frac { C } { \| x _ { i } \| _ { 2 } } } \right) .\tag{4}
$$

In the implementation, the clipping threshold is $C = 2 . 0$

The clipped table is denoted by $\bar { D } = ( \bar { x } _ { 1 } , \dots , \bar { x } _ { n } )$ . Conditional on fixed scaling parameters, and under the replace-one adjacency relation, the $L _ { 2 }$ sensitivity of the full clipped table, viewed as one concatenated vector, is bounded by

$$
\Delta _ { 2 } \leq 2 C .\tag{5}
$$

With $C = 2 . 0$ , this gives

$$
\Delta _ { 2 } = 4 . 0 .\tag{6}
$$

For coordinate-wise Laplace noise applied to the full feature vector, the corresponding $L _ { 1 }$ sensitivity must be considered. If only $L _ { 2 }$ clipping is applied and the scaling parameters are treated as fixed, the $L _ { 1 }$ sensitivity is bounded by

$$
\Delta _ { 1 } \leq 2 { \sqrt { d } } C ,\tag{7}
$$

where d is the feature dimension. Since the implementation uses $d = 3 6 3$ and $C = 2 . 0$ , the resulting bound is

$$
\Delta _ { 1 } \leq 2 { \sqrt { 3 6 3 } } \cdot 2 . 0 \approx 7 6 . 2 1 .\tag{8}
$$

Table 1 summarizes the differential privacy configuration used in the implementation. The Gaussian and Laplace mechanisms considered in this study add noise to the clipped patient-level feature vectors. For the Gaussian mechanism, noise is sampled from $\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ and added to each clipped vector. The implementation uses the analytic Gaussian mechanism with $\Delta _ { 2 } = 4 . 0$ and $\delta = 1 0 ^ { - 5 }$ . The Gaussian noise scale is therefore calibrated using the $L _ { 2 }$ sensitivity of the clipped patient-level vector and the selected privacy budget ε.

Table 1: Sensitivity bounds and perturbation parameters used in the patient-level EEG feature experiments.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Implemented value or derived bound</td></tr><tr><td rowspan=1 colspan=1>Privacy unit</td><td rowspan=1 colspan=1>Single patient</td></tr><tr><td rowspan=1 colspan=1>Feature dimension</td><td rowspan=1 colspan=1> $d = 3 6 3$ </td></tr><tr><td rowspan=1 colspan=1>Clipping threshold</td><td rowspan=1 colspan=1> $\overline { { C = 2 . 0 } }$ </td></tr><tr><td rowspan=1 colspan=1> $L _ { 2 }$ sensitivity bound</td><td rowspan=1 colspan=1> $\overline { { \Delta _ { 2 } } } = 2 C = 4 . 0$ </td></tr><tr><td rowspan=1 colspan=1> $L _ { 1 }$ sensitivity bound</td><td rowspan=1 colspan=1> $\Delta _ { 1 } \leq 2 \sqrt { d } C \approx 7 6 . 2 1$ </td></tr><tr><td rowspan=1 colspan=1>Gaussianδ</td><td rowspan=1 colspan=1>δ = 10−5</td></tr><tr><td rowspan=1 colspan=1>Tested ε values</td><td rowspan=1 colspan=1>{0.5, 1.0, 2.0, 5.0, 10.0, 20.0}</td></tr><tr><td rowspan=1 colspan=1>Gaussian sensitivity used</td><td rowspan=1 colspan=1>4.0</td></tr><tr><td rowspan=1 colspan=1>Laplace sensitivity used in implementation</td><td rowspan=1 colspan=1>4.0</td></tr></table>

For the Gaussian mechanism, the classical sufficient calibration is

$$
\sigma \ge \frac { \Delta _ { 2 } \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } { \varepsilon } .\tag{9}
$$

However, because this classical expression is commonly stated under restrictive parameter conditions, the implementation uses the analytic Gaussian mechanism. The Gaussian experiments are reported for

$$
\varepsilon \in \{ 0 . 5 , 1 . 0 , 2 . 0 , 5 . 0 , 1 0 . 0 , 2 0 . 0 \} , \qquad \delta = 1 0 ^ { - 5 } , \qquad \Delta _ { 2 } = 4 . 0 .\tag{10}
$$

For the Laplace mechanism, independent coordinate-wise noise is added to each clipped feature coordinate. The implementation used a sensitivity value of 4.0, corresponding to $2 C$ . Therefore, the Laplace scale used in the experiments was

$$
b _ { \mathrm { i m p l } } = { \frac { 4 . 0 } { \varepsilon } } .\tag{11}
$$

This gives the following implemented noise scales:

$$
b _ { \mathrm { i m p l } } \in \{ 8 . 0 , 4 . 0 , 2 . 0 , 0 . 8 , 0 . 4 , 0 . 2 \} \quad \mathrm { f o r } \quad \varepsilon \in \{ 0 . 5 , 1 . 0 , 2 . 0 , 5 . 0 , 1 0 . 0 , 2 0 . 0 \} .\tag{12}
$$

For a formal coordinate-wise Laplace release of the complete 363-dimensional patient-level vector under replace-one subject-level adjacency, the required $L _ { 1 }$ sensitivity bound is instead

$$
\Delta _ { 1 } \leq 2 \sqrt { 3 6 3 } \cdot 2 . 0 \approx 7 6 . 2 1\tag{13}
$$

The corresponding formally calibrated Laplace scale would be

$$
b _ { \mathrm { f o r m a l } } = { \frac { \Delta _ { 1 } } { \varepsilon } } = { \frac { 7 6 . 2 1 } { \varepsilon } }\tag{14}
$$

Thus, the formally calibrated Laplace scales would be approximately

$$
b _ { \mathrm { f o r m a l } } \in \{ 1 5 2 . 4 2 , 7 6 . 2 1 , 3 8 . 1 0 , 1 5 . 2 4 , 7 . 6 2 , 3 . 8 1 \} \quad \mathrm { f o r } \quad \varepsilon \in \{ 0 . 5 , 1 . 0 , 2 . 0 , 5 . 0 , 1 0 . 0 , 2 0 . 0 \} .\tag{15}
$$

Consequently, the Gaussian experiments use the implemented $L _ { 2 }$ sensitivity bound $\Delta _ { 2 } = 4 . 0 $ , whereas the Laplace implementation uses $b _ { \mathrm { i m p l } } = 4 . 0 / \varepsilon$ rather than the full-vector scale $b _ { \mathrm { f o r m a l } } = 7 6 . 2 1 / \varepsilon$ . The Laplace results therefore characterize utility under the implemented perturbation magnitudes and are not claimed to provide formal full-vector $\varepsilon { \mathrm { - D P } }$ guarantees under replace-one subject-level adjacency. This comparison remains informative for showing how perturbation magnitude affects EEG feature utility, while Equations 13–15 give the calibration required for a formal full-vector Laplace release. The Gaussian mechanism is calibrated using the analytic Gaussian accountant [30], including for the larger ε values evaluated in this study.

## 3.3.1 Applying anonymization

The implemented anonymization procedure follows the same high-level structure for both perturbation mechanisms. Each patient-level EEG feature vector is first scaled and clipped to bound its contribution. For each mechanism, a noise scale is determined from the selected parameters and sensitivity value, and random noise is then added to produce the perturbed feature vector. The resulting privatized feature table is used for intrinsic utility analysis or downstream model training. The clipping step is essential because it limits the maximum influence that any single patient can have on the released feature table.

The Gaussian mechanism uses the $L _ { 2 }$ sensitivity of the clipped patient-level feature vector. In contrast, a formally calibrated coordinate-wise Laplace mechanism for the full released feature vector requires the corresponding $L _ { 1 }$ sensitivity, with scale $b = \Delta _ { 1 } / \epsilon$ . The Laplace experiments used the implemented sensitivity value 4.0 and scale $b _ { \mathrm { i m p l } } = 4 . 0 / \epsilon$ . Accordingly, these results evaluate the implemented Laplace perturbation settings and are not claimed to provide a formally calibrated full-vector Laplace-DP guarantee under replace-one subject-level adjacency. The original raw EEG recordings and direct patient identifiers are not included in the released feature table.

Algorithms 1 and 2 summarize the two perturbation procedures considered. In both algorithms, the input feature vectors are assumed to be the patient-level feature vectors after the scaling step described above. Algorithm 1 describes the Gaussian perturbation procedure, where the noise scale is obtained from the selected Gaussian accountant. Algorithm 2 describes the formally calibrated coordinate-wise Laplace procedure based on the $L _ { 1 }$ sensitivity bound.

Algorithm 1 Gaussian perturbation of clipped EEG feature vectors   
1: function GAUSSIAN-ANONYMIZATION(D, C, ε, δ, accountant)   
Step 1: Clip patient-levelfeature vectors   
2: for each patient vector $x _ { i } \in D$ do   
3: $\bar { x } _ { i } \gets x _ { i }$ · min $( 1 , C / \| x _ { i } \| _ { 2 } )$   
4: end for   
Step 2: Calibrate noise to subject-level sensitivity   
5: $\Delta _ { 2 }  2 C$ ▷ replace-one adjacency   
6: σ ← GAUSSIANSCALE $( \Delta _ { 2 } , \varepsilon , \delta ,$ accountant)   
Step 3: Add Gaussian noise   
7: for each clipped vector x¯<sub>i</sub> do   
8: Draw $z _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$   
9: x˜<sub>i</sub> ← x¯<sub>i</sub> + z<sub>i</sub>   
10: Add x˜ to De   
11: end for   
12: return De   
13: end function

Algorithm 2 Laplace perturbation of clipped EEG feature vectors   
1: function LAPLACE-ANONYMIZATION(D, C, ε, d)   
Step 1: Clip patient-levelfeature vectors   
2: for each patient vector $x _ { i } \in D$ do   
3: $\bar { x } _ { i } \gets x _ { i }$ · min $( 1 , C / \| x _ { i } \| _ { 2 } )$   
4: end for   
Step 2: Calibrate coordinate-wise Laplace noise   
5: $\Delta _ { 1 }  2 \sqrt { d } C$ ▷ replace-one adjacency and $L _ { 2 }$ clipping   
6: $b  \Delta _ { 1 } / \varepsilon$   
Step 3: Add Laplace noise   
7: for each clipped vector $\textstyle { \bar { x } } _ { i }$ do   
8: Draw $z _ { i } \in \mathbb { R } ^ { d }$ , where each $z _ { i j } \sim \mathrm { L a p } ( 0 , b )$   
9: $\tilde { x } _ { i } \gets \bar { x } _ { i } + z _ { i }$   
10: Add $\tilde { x } _ { i }$ to $\widetilde { D }$   
11: end for   
12: return $\widetilde { D }$   
13: end function

## 4 Performance Analysis

The proposed privacy-preserving EEG framework was implemented in Python using MNE-Python for EEG preprocessing [31], NumPy and SciPy for feature extraction, scikit-learn for downstream classification [32], and IBM DiffPrivLib for implementing the Gaussian and Laplace perturbation mechanisms. Experiments were conducted in Docker-based local environments, with Jupyter Notebook used for development and testing. A clinical EEG dataset was used, containing 122 EEG recording files, 8 GB of recordings, around 33 channels, and sampling rates of 250–500 Hz.

The implementation constructs one patient-level EEG feature vector per patient. Raw EEG files are first grouped by patient identifier, preprocessed, segmented into 2-second windows with 50% overlap, and converted into statistical and frequency-domain EEG features. Window-level features belonging to the same patient are then aggregated by the mean to obtain a single patient-level representation. Labels are assigned at the patient-level from the available metadata files using strict seizure-related keywords.

Table 2 summarizes the main dataset, preprocessing, privacy, and model-evaluation settings used in the experiments. Each privacy configuration was evaluated using one noise realization. The resulting measurements are reported as a case-study comparison of utility trends across the tested privacy budgets.

The analysis focuses on three aspects: statistical utility preservation, frequency-domain signal distortion, and downstream machine learning performance. The objective is to assess the privacy–utility trade-off introduced by the Gaussian and Laplace perturbations and to determine how anonymization affects both signal quality and classification performance. The evaluation uses both statistical and classification-based metrics. Statistical metrics compare the original and anonymized EEG features without training a machine learning model, measuring the extent to which anonymization preserves the structure and distribution of the original data. Classification metrics are then used to evaluate whether anonymized EEG features remain useful for downstream epilepsy detection.

Table 2: Reproducibility configuration for the patient-level EEG anonymization experiments.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Value used in implementation</td></tr><tr><td rowspan=1 colspan=1>Programming environment</td><td rowspan=1 colspan=1>Python 3.13.15, Jupyter Notebook 7.2.2</td></tr><tr><td rowspan=1 colspan=1>Main libraries</td><td rowspan=1 colspan=1>MNE-Python 1.10.1, NumPy 2.1.3, SciPy 1.16.3, scikit-learn1.6.1, IBM DiffPrivLib 0.6.6</td></tr><tr><td rowspan=1 colspan=1>Dataset source</td><td rowspan=1 colspan=1>Anonymized clinical EEG recordings made available through theVIKING project</td></tr><tr><td rowspan=1 colspan=1>Dataset size</td><td rowspan=1 colspan=1>122 EEG recording files from 17 different patients, around 8 GBof recordings</td></tr><tr><td rowspan=1 colspan=1>EEG channels and sampling rate</td><td rowspan=1 colspan=1>Around 33 channels; sampling rates of 250–500 Hz</td></tr><tr><td rowspan=1 colspan=1>Scaling before perturbation</td><td rowspan=1 colspan=1>Feature-wise robust scaling of the aggregated patient vectors usingthe median and interquartile range, followed by $L _ { \mathrm { 2 } } ~ \mathrm { c l i p p i n g }$ </td></tr><tr><td rowspan=1 colspan=1>Patient-level aggregation</td><td rowspan=1 colspan=1>Mean aggregation of window-level features per patient</td></tr><tr><td rowspan=1 colspan=1>Feature dimension</td><td rowspan=1 colspan=1> $d = 3 6 3$ </td></tr><tr><td rowspan=1 colspan=1>Clipping threshold</td><td rowspan=1 colspan=1> $C = 2 . 0$ </td></tr><tr><td rowspan=1 colspan=1>Adjacency relation</td><td rowspan=1 colspan=1>Bounded replace-one subject-level adjacency</td></tr><tr><td rowspan=1 colspan=1>Privacy unit</td><td rowspan=1 colspan=1>One patient</td></tr><tr><td rowspan=1 colspan=1>σ values</td><td rowspan=1 colspan=1>Obtained   directly    from   the   IBM   DiffPrivLibGaussianAnalytic mechanism using  $\Delta _ { 2 } \quad = \quad 4 . 0$ and $\delta = 1 0 ^ { - 5 }$ for each tested €.</td></tr><tr><td rowspan=1 colspan=1>Gaussian $L _ { 2 }$ sensitivity used in theexperiments</td><td rowspan=1 colspan=1> $\Delta _ { 2 } = 2 C = 4 . 0$ </td></tr><tr><td rowspan=1 colspan=1>General  $L _ { 1 }$  sensitivity bound forcoordinate-wise Laplace release</td><td rowspan=1 colspan=1> $\Delta _ { 1 } \leq 2 \sqrt { d } C \approx 7 6 . 2 1$ </td></tr><tr><td rowspan=1 colspan=1>Gaussian $\delta$ </td><td rowspan=1 colspan=1>10-5</td></tr><tr><td rowspan=1 colspan=1>Laplace mechanism implementation</td><td rowspan=1 colspan=1>Coordinate-wise Laplace noise generated using IBM DiffPrivLib</td></tr><tr><td rowspan=1 colspan=1>Laplace   sensitivity    used   inimplementation</td><td rowspan=1 colspan=1>4.0</td></tr><tr><td rowspan=1 colspan=1>Tested ε values</td><td rowspan=1 colspan=1>{0.5, 1.0, 2.0, 5.0, 10.0, 20.0}</td></tr><tr><td rowspan=1 colspan=1>Implemented Laplace scales</td><td rowspan=1 colspan=1>{8.0, 4.0, 2.0, 0.8, 0.4, 0.2}         for        ε{0.5, 1.0, 2.0, 5.0, 10.0, 20.0}</td></tr><tr><td rowspan=1 colspan=1>Formal Laplace scales under full $L _ { 1 }$ calibration</td><td rowspan=1 colspan=1>{152.42, 76.21, 38.10, 15.24, 7.62, 3.81}    for   ε{0.5, 1.0, 2.0, 5.0, 10.0, 20.0}</td></tr><tr><td rowspan=1 colspan=1>Noise repetitions</td><td rowspan=1 colspan=1>One noise realization per ε setting</td></tr><tr><td rowspan=1 colspan=1>Threshold-selection status</td><td rowspan=1 colspan=1>Cohort-level threshold optimization using the aggregated LOSOpredictions</td></tr><tr><td rowspan=1 colspan=1>Threshold grid</td><td rowspan=1 colspan=1>Candidate thresholds considered in the experiments: 0.20–0.80for the baseline evaluation and 0.05–0.95 for the DP evaluations.</td></tr></table>

Table 3: Gaussian-DP calibration across ϵ values.
<table><tr><td rowspan=1 colspan=1>€</td><td rowspan=1 colspan=1>δ</td><td rowspan=1 colspan=1> $\Delta _ { 2 }$ </td><td rowspan=1 colspan=1>Noise scale σ</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>28.1273</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>14.9225</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>7.9752</td></tr><tr><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>3.5675</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>1.9996</td></tr><tr><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>1.1602</td></tr></table>

Table 4: Privacy–utility trend across ϵ values for the analytically calibrated Gaussian mechanism.
<table><tr><td rowspan=1 colspan=1>€</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>Correlation</td><td rowspan=1 colspan=1>SNR (dB)</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>28.7048</td><td rowspan=1 colspan=1>22.9830</td><td rowspan=1 colspan=1>-0.0090</td><td rowspan=1 colspan=1>-48.7376</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>14.7640</td><td rowspan=1 colspan=1>11.7113</td><td rowspan=1 colspan=1>-0.0052</td><td rowspan=1 colspan=1>-42.9626</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>7.9066</td><td rowspan=1 colspan=1>6.3135</td><td rowspan=1 colspan=1>0.0105</td><td rowspan=1 colspan=1>-37.5382</td></tr><tr><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>3.5468</td><td rowspan=1 colspan=1>2.8286</td><td rowspan=1 colspan=1>0.0220</td><td rowspan=1 colspan=1>-30.5752</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>1.9988</td><td rowspan=1 colspan=1>1.5913</td><td rowspan=1 colspan=1>0.0522</td><td rowspan=1 colspan=1>-25.5937</td></tr><tr><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>1.1561</td><td rowspan=1 colspan=1>0.9198</td><td rowspan=1 colspan=1>0.0743</td><td rowspan=1 colspan=1>-20.8387</td></tr></table>

![](images/016f70cc88dc7a50c94475e5b888364ed3c3dca57996862627f9277681f59de9.jpg)  
Figure 7: Utility trend under the analytically calibrated Gaussian mechanism.

## 4.1 Privacy–Utility Assessment of the Gaussian Mechanism

The Gaussian mechanism adds noise sampled from a Gaussian distribution, where the noise level is controlled by the privacy budget ϵ, the privacy failure probability δ, and the $L _ { 2 }$ sensitivity of the released feature vector. In this implementation, Gaussian noise was calibrated using IBM DiffPrivLib’s GaussianAnalytic mechanism with $\Delta _ { 2 } = 4 . 0$ and $\delta = 1 0 ^ { - 5 }$ . The implementation used analytic Gaussian noise calibration with $\Delta _ { 2 } ~ = ~ 4 . 0$ for the clipped feature representation. This sensitivity bound assumes a fixed scaling transformation; it does not account for changes in the median and interquartile range when a patient is replaced in the underlying dataset. The actual Gaussian noise scale σ used for each privacy budget is reported in Table 3. The evaluation then examines how these calibrated ϵ values affect distortion, correlation, and signal-to-noise ratio.

As shown in Table 4 and Figure 7, increasing ϵ reduces RMSE and MAE, indicating that the perturbed feature representation becomes closer to the original feature representation. The correlation and SNR also improve as ϵ increases. However, the correlation remains low across all tested values and the SNR remains negative, indicating that the added noise remains large relative to the evaluated feature representation. This indicates a substantial utility loss under the Gaussian mechanism, especially in stronger privacy settings.

The results indicate that preserving EEG feature utility with the Gaussian mechanism is difficult in this setting. Even at larger privacy budgets, the correlation remains low and the SNR remains negative. This suggests that the tested Gaussian configurations introduce substantial distortion even under analytic Gaussian calibration.

## 4.2 Privacy–Utility Assessment of the Laplace Mechanism

Coordinate-wise Laplace noise is formally calibrated using the $L _ { 1 }$ sensitivity of the released object and the privacy parameter ϵ. Here, the released object is the complete patient-level EEG feature vector under replace-one subject-level adjacency. With $L _ { 2 }$ clipping, the corresponding bound is $\Delta _ { 1 } \leq 2 { \sqrt { d } } C ;$ for $d = 3 6 3$ and $C = 2 . 0$ , this gives $\Delta _ { 1 } \leq 7 6 . 2 1$ . The implementation instead used the sensitivity value 4.0 and scale $b _ { \mathrm { i m p l } } = 4 . 0 / \epsilon$ . The resulting experiments quantify utility under these implemented Laplace perturbation magnitudes, while the scale required for a formal full-vector Laplace release, $b _ { \mathrm { f o r m a l } } = 7 6 . 2 1 / \epsilon$ , is derived separately above.

Table 5 shows that increasing ϵ reduces the Laplace perturbation scale used in the experiments and improves both correlation and SNR, while Figure 8 visualizes the corresponding correlation trend. Smaller ϵ values correspond to larger perturbation magnitudes and lower utility, while larger ϵ values preserve more feature structure and correspond to smaller perturbation magnitudes in the implemented sweep. Because the Laplace implementation used the sensitivity value 4.0 rather than the full $L _ { 1 }$ sensitivity bound, these results characterize the observed utility trend under the implemented Laplace scales. Based on this trend, $\epsilon = 2$ and $\epsilon = 1 0$ are selected as representative high-perturbation and lower-perturbation settings for comparison. In this table, ϵ indexes the implemented relation $b _ { \mathrm { i m p l } } = 4 . 0 / \epsilon$ ; the formal full-vector calibration is given separately in Equations 13–15.

Table 5: Utility trend under the implemented Laplace scales $b _ { \mathrm { i m p l } } = 4 . 0 / \epsilon$
<table><tr><td>€</td><td>Corr.</td><td>SNR (dB)</td><td>Interpretation</td></tr><tr><td>0.5</td><td>0.019</td><td>-40.79</td><td>Largest perturbation setting among the tested values, with very low utility.</td></tr><tr><td>1.0</td><td>0.024</td><td>-34.51</td><td>Large perturbation setting; utility remains low.</td></tr><tr><td>2.0</td><td>0.043</td><td>-28.66</td><td>Large perturbation setting; correlation and SNR begin to improve.</td></tr><tr><td>5.0</td><td>0.103</td><td>-20.54</td><td>Moderate perturbation setting; utility improves but remains limited.</td></tr><tr><td>10.0 20.0</td><td>0.201</td><td>-14.80</td><td>Lower perturbation setting; utility improves compared with smaller € values.</td></tr><tr><td></td><td>0.335</td><td>-8.61</td><td>Smallest perturbation setting among the tested values, with the highest observed utility.</td></tr></table>

![](images/007a0ecddad70d6b0a688a7247c6700a8b96c39a0d651fdb9bdb888ee8da8e7a.jpg)  
Figure 8: Correlation between the original and perturbed feature values under the implemented Laplace perturbation with $b _ { \mathrm { i m p l } } = 4 . 0 / \epsilon$

The results show that this perturbation setting preserves more statistical structure than the tested Gaussian configurations as ϵ increases. The results indicate that the implemented Laplace perturbation preserves more statistical utility than the analytically calibrated Gaussian mechanism in the evaluated configuration. However, higher-utility settings also correspond to smaller perturbation magnitudes in the implemented sweep. Therefore, these results show a utility trend rather than establishing a generally optimal privacy–utility setting for clinical EEG anonymization.

## 4.3 Machine Learning Model Assessment

The downstream classification task was evaluated using Leave-One-Subject-Out (LOSO) validation at the patient level. In each fold, one patient was held out for testing, and the remaining patients were used for training. All reported classification metrics and confusion matrices are therefore computed from patient-level predictions, not from individual files, epochs, or windows. Window-level features were first aggregated into one feature representation per patient, and the class distribution used in the evaluation refers to these patient-level labels. Patient-level labels were assigned using the positive seizure-related keywords “epileptic seizure,” “ictal EEG pattern,” and “ictal EEG activity.” The expressions “epileptiform interictal activity” and “abnormal interictal” did not by themselves trigger a positive label. No additional automated negation handling or manual label verification was performed. This patient-level split prevents files, epochs, or windows from the same patient from appearing in both the training and test portions of a LOSO fold and remains consistent with the patient-level privacy unit used in the DP analysis. The downstream classifier was implemented using scikit-learn’s MLPClassifier with hidden-layer sizes (100), activation relu, solver adam, alpha = 0.0001, learning-rate setting constant, max\_iter = 200, early\_stopping = False, and random\_state = None. Within each LOSO fold, StandardScaler was fitted on the training patients and applied to the held-out patient.

Table 6 shows the class distribution used in the patient-level evaluation, with 12 patients in class 0 and 5 patients in class 1. The epilepsy class was treated as the positive class (y = 1), and the non-epilepsy class as the negative class $( y = 0 )$ .

For each feature representation, LOSO validation produced one predicted probability per patient. The decision threshold was selected from the aggregated LOSO predictions to maximize the F1-score for the epilepsy class. Because threshold selection and evaluation used the same cohort, the classification results are interpreted as cohort-level downstream utility comparisons rather than estimates of performance in an independent clinical population [33]. The same evaluation procedure was applied consistently to the original, Gaussian-perturbed, and Laplace-perturbed feature representations. For the classification comparison in Table 7, the Gaussian-perturbed features were generated using $\epsilon = 1 0 . 0 , \delta = 1 0 ^ { - 5 } , \Delta _ { 2 } = 4 . 0$ and $\sigma = 1 . 9 9 9 6$ , whereas the Laplace-perturbed features were generated using $\epsilon = 1 0 . 0 ,$ implemented sensitivity 4.0, and $b = 0 . 4$ . The selected decision thresholds were 0.40 for the original features, 0.05 for the Gaussian-perturbed features, and 0.10 for the Laplace-perturbed features.

Table 6: Dataset configuration for the patient-level MLP evaluation.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Evaluation</td><td rowspan=1 colspan=1>No. of Patients / Class Distribution (0/1)</td></tr><tr><td rowspan=1 colspan=1>Original</td><td rowspan=1 colspan=1>LOSO</td><td rowspan=1 colspan=1>17 / 12-5</td></tr></table>

## 4.3.1 Evaluation on Original EEG Data

Original EEG features were first used to establish a non-private baseline for epilepsy detection. Model performance was evaluated using accuracy, precision, recall, F1-score, and confusion matrix. As shown in Table 7, the MLP model produced a patient-level LOSO confusion matrix with $\mathrm { T N } = 9 , \mathrm { F P } = 3 , \mathrm { F N } = 2$ and $\mathrm { T P } = 3$ on the original EEG dataset. Based directly on this matrix, the model achieved an accuracy of 0.706, precision of 0.500, recall of 0.600, specificity of 0.750, balanced accuracy of 0.675, and F1-score of 0.545 for the epilepsy class. For comparison, a classifier that predicts every patient as non-epilepsy would obtain the same overall accuracy of $1 2 / 1 7 = 0 . 7 0 6$ , but a balanced accuracy of 0.500. The original-feature MLP achieved a higher balanced accuracy of 0.675 because it correctly classified patients from both classes $( T N = 9$ and $T P = 3 )$ , rather than predicting only the majority class. These results provide a non-private baseline for the downstream utility check, but they should be interpreted cautiously due to the small and imbalanced dataset. In particular, cross-validation estimates obtained from such a small cohort can have large sampling variability and error bars [34].

To quantify the uncertainty associated with the small patient cohort, two-sided 95% Wilson score intervals were calculated directly from the patient-level confusion matrices. For the original features, the intervals are 0.469–0.867 for accuracy, 0.231–0.882 for recall, and 0.468–0.911 for specificity. The corresponding intervals are 0.133–0.531, 0.566–1.000, and 0.000–0.242 for the Gaussian-perturbed features, and 0.096–0.473, 0.231–0.882, and 0.015–0.354 for the Laplace-perturbed features. These intervals quantify the uncertainty associated with the limited cohort size. Because the decision thresholds were selected using the same cohort, the intervals are interpreted descriptively.

Figure 9a shows the patient-level LOSO confusion matrix for the original EEG model. The model produced TN = 9, FP = 3, FN = 2, and $\mathrm { T P } = 3$ . These counts are consistent with the patient-level class distribution, since $T N + F P = 1 2$ for the non-epilepsy patients and $F N + T P = 5$ for the epilepsy patients.

Table 7: Patient-level LOSO classification performance across original, Gaussian-perturbed, and Laplace-perturbed EEG features.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>Specificity</td><td rowspan=1 colspan=1>Bal. Acc.</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=1>Original EEG</td><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>0.706</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.750</td><td rowspan=1 colspan=1>0.675</td><td rowspan=1 colspan=1>0.545</td></tr><tr><td rowspan=1 colspan=1>Gaussian-perturbedEEG</td><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>0.294</td><td rowspan=1 colspan=1>0.294</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.455</td></tr><tr><td rowspan=1 colspan=1>Laplace-perturbedEEG</td><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>0.235</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.342</td><td rowspan=1 colspan=1>0.316</td></tr></table>

![](images/e65176ed36ebff3ae3ba584d6d50e4dda573a0cac822ecf1bebb7e97ead6ddc4.jpg)  
(a) Confusion matrix

![](images/ca3eec8f39c1d83eb8421e2784e4767cc1e0229667c575df1af412861827071f.jpg)  
(b) Accuracy and F1-score  
Figure 9: MLP performance on original EEG features.

## 4.3.2 Evaluation on Gaussian-Perturbed EEG Data

The Gaussian-perturbed features were evaluated using the patient-level LOSO procedure and threshold-selection approach described above. Accuracy, balanced accuracy, sensitivity, specificity, precision, F1-score, and the patient-level confusion matrix are reported. Table 7 shows that the Gaussian-perturbed model obtained a low patient-level classification performance. Although recall was 1.000, specificity was 0, and precision was 0.294, meaning that all non-epilepsy patients were misclassified as epilepsy. Accordingly, the Gaussian-perturbed representation provides limited downstream classification utility because the model does not preserve class separability under this configuration.

Figure 10a shows the confusion matrix for the MLP model trained on Gaussian-perturbed features. The model produced TN = 0, FP = 12, FN = 0, and $\mathrm { T P } = 5$ . Thus, all epilepsy cases were detected, but all non-epilepsy cases were incorrectly classified as epilepsy. Figure 10b summarizes the MLP performance on Gaussian-perturbed features. Because all patients were assigned to the positive class, the recall of 1.000 does not indicate clinically useful sensitivity. Instead, the zero specificity and all-positive prediction pattern demonstrate substantial loss of class separability under this Gaussian configuration.

## 4.3.3 Evaluation on Laplace-Perturbed EEG Data

The same LOSO evaluation was applied to the Laplace-perturbed patient-level EEG features generated in the experiments. Probability-based predictions were used, and the decision threshold was tuned to maximize the F1-score for the epilepsy class. Table 7 shows that the MLP model trained on Laplace-perturbed EEG features achieved limited patient-level classification performance by giving an accuracy of 0.235, precision of 0.214, recall of 0.600, specificity of 0.083, balanced accuracy of 0.342, and F1-score of 0.316. Although the model correctly detected three of the five epilepsy patients, it also misclassified eleven of the twelve non-epilepsy patients as epilepsy. The very low specificity and high false-positive count demonstrate substantial downstream utility degradation; this result should not be interpreted as evidence of clinical usefulness.

![](images/ac4941a5d60d6d0649df07fb01d840f2388013045a5d67be45ebfbbfd202c787.jpg)  
(a) Confusion matrix

![](images/6f2d95e74a76d3aac7310c0c48c77764c067d762f76eea0da4f401a168127106.jpg)  
(b) Accuracy and F1-score

Figure 10: MLP performance on Gaussian-perturbed EEG features.  
![](images/7aba5d5b3367e2ec04de3f2a433342e247e7800bdf30751405bbf298b33375c1.jpg)  
(a) Confusion matrix

![](images/5b9dc22ffcebfde18fcdb5c33f25da32a8db553c629d4d08fa57a2eef8eabef5.jpg)  
(b) Accuracy and F1-score  
Figure 11: MLP performance on Laplace-perturbed EEG features.

Figure 11a presents the confusion matrix for the MLP model trained on Laplace-perturbed features. The model produced TN = 1, FP = 11, FN = 2, and $\mathrm { T P } = 3$ . Compared with the Gaussian-perturbed setting, the implemented Laplace perturbation produced one correctly classified non-epilepsy case but missed two epilepsy cases. Figure 11b summarizes the corresponding accuracy and F1-score. Although the F1-score is slightly higher than the accuracy, the results remain weaker than the original-data baseline. This indicates that Laplace noise reduces feature separability and limits downstream classification performance.

Under the implemented settings, the Laplace perturbation preserves only limited classification utility for epilepsy detection and still produces a high number of false positives. Compared with the Gaussian-perturbed setting, it slightly improves specificity but also introduces false negatives. These results indicate that the Laplace-perturbed features have limited usefulness for classification in the evaluated configuration.

## 5 Conclusion

Differentially private anonymization of clinical EEG-derived feature representations was studied in a multi-hospital setting. Three deployment scenarios were considered: client-side anonymization, centralized server-side anonymization, and decentralized local training. Following EEG preprocessing and feature extraction, Gaussian and Laplace perturbations were applied to the resulting patient-level EEG feature representations, and their impact was assessed using statistical utility measures and a downstream patient-level machine-learning utility check. The Laplace experiments used the implemented scale $b _ { \mathrm { i m p l } } = 4 . 0 / \varepsilon$ , while the full-vector $L _ { 1 }$ calibration required for a formal Laplace guarantee was derived separately.

The results show that DP-based perturbation can be integrated into EEG processing workflows, while the selected mechanism, sensitivity calibration, privacy parameters, and evaluation unit strongly affect data utility. The Gaussian mechanism was analytically calibrated using the implemented $L _ { 2 }$ sensitivity, with the normalization parameters treated as fixed during the perturbation stage. The Laplace experiments quantify utility under the implemented perturbation magnitudes, while the corresponding full-vector $L _ { 1 }$ calibration is derived separately. In the evaluated small and imbalanced dataset, larger perturbation magnitudes introduced substantial distortion, and downstream classification performance was limited after anonymization. These findings support the practical integration of privacy-preserving perturbation mechanisms into clinical EEG processing, provided that the privacy unit, adjacency relation, sensitivity, noise calibration, and validation protocol are specified carefully. Future work should evaluate larger and more balanced EEG datasets, repeat experiments over multiple noise realizations, investigate end-to-end privacy accounting for additional data-dependent preprocessing choices, and include empirical privacy-risk analyses such as re-identification, linkage, membership inference, and reconstruction attacks.

## Acknowledgment

This work was supported in part by the Dam Foundation under project number SDAM\_UTV532216 and by the Research Council of Norway under project number 331903.

## Ethics Statement

The collection and processing of data were approved by the Norwegian South-Eastern Regional Ethics Committee (REC; reference no. 689562) and the local data protection officer at Oslo University Hospital in accordance with applicable European Union and Norwegian requirements. The project was granted an exemption by REC from the requirement of informed consent. Only anonymized data were made available to the authors.

## References

[1] J. Andrew, R. J. Eunice, and J. Karthikeyan, “An anonymization-based privacy-preserving data collection protocol for digital health data,” Frontiers in Public Health, vol. 11, 2023. [Online]. Available: https://doi.org/10.3389/fpubh.2023.1125011

[2] S. Nazir, S. Khan, H. U. Khan, S. Ali, I. García-Magariño, R. B. Atan, and M. Nawaz, “A comprehensive analysis of healthcare big data management, analytics and scientific programming,” IEEE Access, vol. 8, pp. 95 714–95 733, 2020. [Online]. Available: https://doi.org/10.1109/ACCESS.2020.2995572

[3] L. Sweeney, “k-anonymity: A model for protecting privacy,” International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, vol. 10, no. 5, pp. 557–570, 2002. [Online]. Available: https://doi.org/10.1142/S0218488502001648

[4] C. Dwork, F. McSherry, K. Nissim, and A. D. Smith, “Calibrating noise to sensitivity in private data analysis,” in Theory of Cryptography, Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006, Proceedings, ser. Lecture Notes in Computer Science, S. Halevi and T. Rabin, Eds., vol. 3876. Springer, 2006, pp. 265–284. [Online]. Available: https://doi.org/10.1007/11681878\_14

[5] C. Dwork and A. Roth, “The algorithmic foundations of differential privacy,” Found. Trends Theor. Comput. Sci., vol. 9, no. 3-4, pp. 211–407, 2014. [Online]. Available: https://doi.org/10.1561/0400000042

[6] R. Subramanian, “Differential privacy techniques for healthcare data,” in 3rd International Conference on Intelligent Data Science Technologies and Applications, IDSTA 2022, San Antonio, TX, USA, September 5-7, 2022, M. A. Alsmirat, Y. Jararweh, M. Aloqaily, and I. Alsmadi, Eds. IEEE, 2022, pp. 95–100. [Online]. Available: https://doi.org/10.1109/IDSTA55301.2022.9923037

[7] Z. Sun, Y. Wang, M. Shu, R. Liu, and H. Zhao, “Differential privacy for data and model publishing of medical data,” IEEE Access, vol. 7, pp. 152 103–152 114, 2019. [Online]. Available: https://doi.org/10.1109/ACCESS.2019.2947295

[8] M. Letafati and S. Otoum, “Global differential privacy for distributed metaverse healthcare systems,” in 2023 International Conference on Intelligent Metaverse Technologies & Applications (iMETA), 2023, pp. 01–08. [Online]. Available: https://doi.org/10.1109/iMETA59369.2023.10294469

[9] M. Mohammadi, M. Vejdanihemmat, M. Lotfinia, M. Rusu, D. Truhn, A. K. Maier, and S. T. Arasteh, “Differential privacy for medical deep learning: Methods, tradeoffs, and deployment implications,” npj Digital Medicine, vol. 9, no. 1, p. 93, 2026. [Online]. Available: https://doi.org/10.1038/s41746-025-02280-z

[10] M. Abadi, A. Chu, I. J. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with differential privacy,” in 2016 ACM SIGSAC Conference on Computer and Communications Security, E. R. Weippl, S. Katzenbeisser, C. Kruegel, A. C. Myers, and S. Halevi, Eds. Vienna, Austria: ACM, 2016, pp. 308–318. [Online]. Available: https://doi.org/10.1145/2976749.2978318

[11] J. Near, D. Darais, N. Lefkovitz, and G. Howarth, “Guidelines for evaluating differential privacy guarantees,” National Institute of Standards and Technology, Tech. Rep. Special Publication 800-226, 2025. [Online]. Available: https://doi.org/10.6028/NIST.SP.800-226

[12] N. Naeem and M. Toorani, “Decentralized, secure and privacy-preserving sharing of health data: A survey and future directions,” SSRN, Aug. 2026. [Online]. Available: https: //doi.org/10.2139/ssrn.7345361

[13] Y. Luo, B. Jiang, S. Qin, Q. Fu, and S. Zhang, “EEG-based epilepsy recognition via federated learning with differential privacy,” Concurr. Comput. Pract. Exp., vol. 37, no. 9-11, 2025. [Online]. Available: https://doi.org/10.1002/cpe.70072

[14] N. Rieke, J. Hancox, W. Li, F. Milletarì, H. R. Roth, S. Albarqouni, S. Bakas, M. N. Galtier, B. A. Landman, K. H. Maier-Hein, S. Ourselin, M. J. Sheller, R. M. Summers, A. Trask, D. Xu, M. Baust, and M. J. Cardoso, “The future of digital health with federated learning,” npj Digital Medicine, vol. 3, no. 1, p. 119, 2020. [Online]. Available: https://doi.org/10.1038/s41746-020-00323-1

[15] A. Hatamizadeh, H. Yin, P. Molchanov, A. Myronenko, W. Li, P. Dogra, A. Feng, M. G. Flores, J. Kautz, D. Xu, and H. R. Roth, “Do gradient inversion attacks make federated learning unsafe?” IEEE Transactions on Medical Imaging, vol. 42, no. 7, pp. 2044–2056, 2023. [Online]. Available: https://doi.org/10.1109/TMI.2023.3239391

[16] K. A. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for privacy-preserving machine learning,” in 2017 ACM SIGSAC Conference on Computer and Communications Security, B. Thuraisingham, D. Evans, T. Malkin, and D. Xu, Eds. Dallas, TX, USA: ACM, 2017, pp. 1175–1191. [Online]. Available: https://doi.org/10.1145/3133956.3133982

[17] P. Rajabi and M. Toorani, “Secure aggregation for privacy-preserving federated learning on clinical EEG data,” 2026, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2607.28191

[18] L. Meng, X. Jiang, J. Huang, W. Li, H. Luo, and D. Wu, “User identity protection in EEG-based brain-computer interfaces,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 3576–3586, 2023. [Online]. Available: https://doi.org/10.1109/TNSRE.2023.3310883

[19] X. Chen, S. Li, Y. Tu, Z. Wang, and D. Wu, “User-wise perturbations for user identity protection in EEG-based bcis,” Journal ofNeural Engineering, vol. 22, no. 1, p. 016040, 2025. [Online]. Available: https://doi.org/10.1088/1741-2552/ad88a5

[20] G. Singh, P. Patel, M. Asaduzzaman, and G. Bajwa, “Selective EEG signal anonymization using multi-objective autoencoders,” in 20th Annual International Conference on Privacy, Security and Trust, PST 2023, Copenhagen, Denmark, August 21-23, 2023. IEEE, 2023, pp. 1–7. [Online]. Available: https://doi.org/10.1109/PST58708.2023.10320167

[21] H. Wang, J. Ruan, C. Fan, Y. Cheng, and Z. Lv, “ID-RemovalNet: Identity removal network for EEG privacy protection with enhancing decoding tasks,” in 34th International Joint Conference on Artificial Intelligence, IJCAI 2025, Montreal, Canada, August 16-22, 2025. ijcai.org, 2025, pp. 4209–4217. [Online]. Available: https://doi.org/10.24963/ijcai.2025/469

[22] D. Bethge, P. Hallgarten, T. Grosse-Puppendahl, M. Kari, R. Mikut, A. Schmidt, and O. Özdenizci, “Domain-invariant representation learning from EEG with private encoders,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, Virtual and Singapore, 23-27 May 2022. IEEE, 2022, pp. 1236–1240. [Online]. Available: https://doi.org/10.1109/ICASSP43922.2022.9747398

[23] K. Xia, L. Deng, W. Duch, and D. Wu, “Privacy-preserving domain adaptation for motor imagery-based brain-computer interfaces,” IEEE Trans. Biomed. Eng., vol. 69, no. 11, pp. 3365–3376, 2022. [Online]. Available: https://doi.org/10.1109/TBME.2022.3168570

[24] K. Xia, W. Duch, Y. Sun, K. Xu, W. Fang, H. Luo, Y. Zhang, D. Sang, X. Xu, F. Wang, and D. Wu, “Privacy-preserving brain-computer interfaces: A systematic review,” IEEE Trans. Comput. Soc. Syst., vol. 10, no. 5, pp. 2312–2324, 2023. [Online]. Available: https://doi.org/10.1109/TCSS.2022.3184818

[25] H. Krawczyk, M. Bellare, and R. Canetti, “HMAC: Keyed-hashing for message authentication,” RFC Editor, Request for Comments RFC 2104, Feb. 1997. [Online]. Available: https: //www.rfc-editor.org/rfc/rfc2104

[26] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in 20th International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 54. PMLR, 2017, pp. 1273–1282.

[27] R. Schnell, T. Bachteler, and J. Reiher, “Privacy-preserving record linkage using bloom filters,” BMC Medical Informatics and Decision Making, vol. 9, no. 1, p. 41, Aug. 2009. [Online]. Available: https://doi.org/10.1186/1472-6947-9-41

[28] T. Ranbaduge and R. Schnell, “Securing bloom filters for privacy-preserving record linkage,” in 29th ACM International Conference on Information and Knowledge Management, M. d’Aquin, S. Dietze, C. Hauff, E. Curry, and P. Cudré-Mauroux, Eds. ACM, 2020, pp. 2185–2188. [Online]. Available: https://doi.org/10.1145/3340531.3412105

[29] P. D. Welch, “The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,” IEEE Transactions on Audio and Electroacoustics, vol. 15, no. 2, pp. 70–73, Jun. 1967. [Online]. Available: https://doi.org/10.1109/TAU.1967.1161901

[30] B. Balle and Y.-X. Wang, “Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising,” in 35th International Conference on Machine Learning. PMLR, 2018, pp. 394–403.

[31] A. Gramfort, M. Luessi, E. Larson, D. A. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, and M. S. Hämäläinen, “MEG and EEG data analysis with MNE-Python,” Frontiers in Neuroscience, vol. 7, p. 267, 2013. [Online]. Available: https://doi.org/10.3389/fnins.2013.00267

[32] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. VanderPlas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://dl.acm.org/doi/10.5555/1953048.2078195

[33] S. Varma and R. Simon, “Bias in error estimation when using cross-validation for model selection,” BMC Bioinformatics, vol. 7, no. 1, p. 91, Feb. 2006. [Online]. Available: https://doi.org/10.1186/1471-2105-7-91

[34] G. Varoquaux, “Cross-validation failure: Small sample sizes lead to large error bars,” NeuroImage, vol. 180, pp. 68–77, 2018, part A. [Online]. Available: https://doi.org/10.1016/j.neuroimage.2017.06.061