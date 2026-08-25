# Robustness of Anomaly Detection Models for Industrial Control Systems under Training-Time Data Contamination

Mustafa Umut Ozbek, Taiwo Ojo, Pooria Madani, Khalil El-Khatib, Li Yang

Ontario Tech University, Oshawa, Ontario, Canada

Emails: mustafaumut.ozbek@ontariotechu.net, taiwo.ojo2@ontariotechu.net,

pooria.madani@ontariotechu.ca, khalil.el-khatib@ontariotechu.ca, li.yang@ontariotechu.ca

Abstract—Machine-learning-based anomaly detection is increasingly used in industrial control systems (ICS), yet most studies assume that detector training data is trustworthy. In practice, training data may be corrupted through compromised logs, labeling errors, manipulated historian records, or unsafe retraining processes. This paper evaluates the robustness of offline ICS anomaly-detection pipelines on the Secure Water Treatment (SWaT) benchmark under training-time contamination. We assess 11 heterogeneous anomaly detectors under three contamination strategies: random injection, similaritytargeted injection, and feature-noise injection. The first two insert attack samples into the nominal training pool, while the third adds bounded Gaussian noise to selected normal training samples. These attacks are contamination-based rather than gradient-driven poisoning methods. Contamination budgets from 1% to 10% are evaluated using clean validation and test sets under a unified offline protocol. The results show that robustness is strongly model-dependent and cannot be predicted from clean-data performance alone. Injection-based contamination causes the greatest degradation, particularly for local-density and distance-based detectors, whereas feature-noise contamination has a comparatively limited effect. PCA, SVM, HBOS, and IForest remain relatively stable, while the tuned neural detectors demonstrate intermediate robustness. Overall, the findings highlight the importance of training-data integrity in ML-enabled ICS monitoring, subject to the evaluated dataset, models, and threat assumptions.

Index Terms—industrial control systems, adversarial machine learning, training-time data contamination, anomaly detection, critical infrastructure

## I. INTRODUCTION

Industrial control systems (ICS) support critical infrastructure such as water treatment, energy, manufacturing, and digital instrumentation and control systems, where cyber incidents can produce direct operational and safety consequences. As these systems become increasingly connected and data-rich, anomaly-based detection has received growing attention as a complement to signature-based and rule-based monitoring. In particular, anomaly-based detectors are attractive in ICS because normal operating behavior is often easier to collect than comprehensive attack data, and multivariate process measurements can reveal deviations that are difficult to capture with static signatures alone [1]–[3].

Most existing ICS anomaly-detection studies evaluate models under the implicit assumption that the training data used to build them is trustworthy. In practice, that assumption is often weak. Datasets used for offline detector development may be assembled from historian exports, archived sensor traces, engineering workstations, alarm logs, operator annotations, and automated preprocessing pipelines. Errors, compromise, or manipulation at any stage can contaminate the final training corpus. As a result, a detector that appears strong under clean offline evaluation may still become unreliable if its training data has already been corrupted before deployment [4], [5].

This concern is especially important in anomaly-based pipelines trained on normal data only. In such settings, the detector does not learn an explicit supervised boundary between normal and attack classes. Instead, it learns a representation, support region, density structure, geometric profile, or reconstruction pattern of nominal process behavior. Contaminating the normal training pool therefore distorts the learned notion of normality itself. Attack samples injected into nominal training data can weaken the anomaly boundary, while perturbations applied to normal samples can blur the distinction between benign and malicious behavior. Despite this risk, the comparative robustness of anomaly-based ICS detectors under offline training-time contamination remains insufficiently characterized across a broad and diverse set of detector families.

Prior work has demonstrated effective clean-condition detection on the Secure Water Treatment (SWaT) benchmark using recurrent, convolutional, one-class, PCA-based, and reconstruction models [1], [6]–[9]. Adversarial ICS studies, however, have concentrated mainly on inference-time evasion [10]–[12] or poisoning of online-adaptive neural detectors [13], [14].

We address this gap with a unified benchmark on SWaT [15], [16]. Eleven normal-only detectors are tested under three contamination strategies at 1%, 3%, 5%, and 10% budgets. Thresholds are calibrated on clean validation data and applied unchanged to a clean held-out test set. Our contributions are:

1) a comparative offline training-contamination benchmark spanning eleven heterogeneous anomaly detectors;

2) a common evaluation of random injection, similaritytargeted injection, and feature-noise injection across four budgets;

3) an analysis that jointly considers F1-score degradation, false-negative rate (FNR), and training cost rather than clean F1-score alone;

4) a transparent configuration-level evaluation that explicitly reports differences in data representation, tuning depth, training scale, and compute resources.

The remainder of this paper is organized as follows. Section II reviews related work on ICS anomaly detection, adversarial machine learning, and training-time contamination. Section III defines the threat model and contamination methodology. Section IV describes the dataset, detector configurations, and experimental protocol. Section V presents and discusses the experimental results, and Section VI concludes the paper and outlines directions for future work.

## II. RELATED WORK

## A. ICS Anomaly Detection

Anomaly-based detection has become a major line of research in ICS because process behavior is highly structured, while comprehensive attack labels are often limited or costly to obtain. The SWaT testbed and dataset have become widely used benchmarks because they provide a realistic and reproducible platform for evaluating process-aware attack-detection methods [15], [16]. Prior SWaT studies include recurrent models trained on normal data [1], convolutional models [6], unsupervised and one-class approaches [7], sequence-tosequence architectures [8], lightweight neural and PCA-based detection [9], and methodology-oriented work emphasizing preprocessing, thresholding, and evaluation [17].

Together, these studies support two recurring conclusions. First, anomaly-based monitoring is a natural fit for ICS settings because normal process behavior is repetitive, correlated, and usually easier to collect than diverse attack traces. Second, there is no universally dominant detector family. Broader reviews of machine learning for ICS security reach similar conclusions while identifying persistent challenges in crosspaper comparability, dataset dependence, and deployment realism [2], [3].

Our study also intersects with benchmark design for anomaly and time-series anomaly detection. Prior work shows that detector rankings can shift substantially with metric choice, hyperparameter treatment, and evaluation protocol [18]–[20], while time-series studies highlight temporal leakage, split construction, and inconsistent evaluation rules [21], [22]. These observations reinforce the need to interpret detector performance through both the model family and the evaluation pipeline used to generate the reported results.<sup>1</sup>

B. Adversarial Machine Learning in Industrial Control Systems

Although adversarial machine learning has received increasing attention in ICS security, most prior work has focused on evasion rather than poisoning. Zizzo et al. studied adversarial attacks against time-series intrusion-detection models for ICS and showed that recurrent detectors can be manipulated even when an attacker controls only a subset of variables [10]. Erba et al. showed that realistic ICS manipulation must satisfy process and physical constraints, making the attack space substantially different from conventional adversarial examples [11]. Anthi et al. likewise demonstrated that adversarial perturbations can materially reduce machine-learning-based cyber defenses in industrial environments [12]. Collectively, these studies show that strong clean-data detection performance should not be interpreted as evidence of adversarial robustness.

## C. Training-Time Poisoning and Contamination of Anomaly Detectors

Compared with evasion, training-time corruption of ICS anomaly detectors has received much less direct study. Kravchik, Biggio, and Shabtai analyzed poisoning attacks against online-trained neural anomaly detectors for ICS [13]. This work was later extended to a practical evaluation of poisoning attacks on online anomaly detectors, again emphasizing online adaptation as the primary threat surface [14]. That setting is highly relevant, but differs from the offline training regime used in many operational pipelines, where models are built or periodically refreshed from curated historian logs, archived process traces, and exported datasets before deployment. Online poisoning studies therefore do not fully characterize contamination of the normal training pool during offline model development.

Outside ICS, the broader adversarial machine-learning literature has shown that training-time corruption can substantially degrade anomaly-detection and classification systems. Rubinstein et al. demonstrated that poisoning can distort anomaly detectors, particularly PCA-based detectors, by shifting the learned model of normality [4]. Other work developed poisoning attacks against SVMs and deep learning, certified defenses, targeted clean-label attacks, and broader surveys [5], [23]–[26]. Recent CPS studies also address provable mitigation and dynamic backdoors [27], [28]. These studies provide the conceptual basis for training-time vulnerability, but primarily address optimization-based poisoning rather than the contamination-style attack families considered here.

## D. Research Gap

Prior ICS studies provide strong evidence that anomalybased detectors can perform effectively under clean conditions [1], [6]–[9], [17], while adversarial ICS studies have focused mainly on evasion or poisoning tied to online retraining [10]–[14]. What remains insufficiently characterized is the comparative robustness of a broad set of anomaly-based ICS detectors under offline training-time contamination within a unified benchmarking framework.

More specifically, to the best of our knowledge, existing literature does not provide a consistent comparison across heterogeneous anomaly-detector families under the same contamination definitions, budgets, threshold-calibration procedure, and clean held-out evaluation protocol. It also remains unclear whether detectors that appear strong on clean benchmark data retain that advantage once the normal training pool is contaminated. The novelty of this study is a unified, configuration-level robustness benchmark across heterogeneous anomaly-detector families under common training pools, contamination budgets, clean calibration data, held-out testing, and evaluation rules.

## III. THREAT MODEL AND CONTAMINATION METHODOLOGY

## A. System and Adversary Model

We consider an offline anomaly-detection pipeline for ICS under a normal-only training regime. Each detector is trained to model nominal process behavior from a clean training pool, while model selection and threshold calibration use a labeled validation split and final performance is measured on a heldout test split that is never modified during contamination. This setup reflects a practical workflow in which historical process data are curated, preprocessed, and used to develop anomaly detectors before deployment.

The adversary’s objective is to degrade the detector’s ability to distinguish attacks from normal behavior at test time while preserving the appearance of a plausible training workflow. More specifically, the attacker seeks to increase false negatives, reduce F1-score and recall, and distort the learned notion of normality so that malicious behavior is more likely to be accepted as benign.

We assume a gray-box adversary with training-time access only. The attacker knows the feature representation and general anomaly-detection pipeline, but does not modify the clean validation or test sets and does not interact with the deployed detector at inference time. Depending on the contamination strategy, the adversary either injects attack samples into the nominal training pool or perturbs selected normal training observations. The similarity-targeted strategy assumes more informed sample selection than random contamination, while all evaluated strategies remain weaker than white-box optimization-based attacks. The budget is at most 10% of the clean training-pool size.

For clarity, “uncontaminated” means that no training-time attack is applied to validation or test observations; it does not mean that the labeled validation split is attack-free. The restrictions to clean calibration and test data isolate trainingtime contamination effects, but may be optimistic relative to workflows in which curation errors or historian compromise propagate beyond the training data.

Offline ICS datasets can absorb attack traces or distorted values through annotation, alignment, curation, export, or preprocessing failures. These failures need not alter the running process; they affect a normal-only detector through data presented as benign. We retain poisoning for continuity with the literature, but evaluate contamination-style rather than gradient-driven worst-case attacks. Figure 1 summarizes the workflow.

Two practical corruption paths motivate the evaluated strategies. First, attack-period observations may be included in a nominal archive through missing labels, event-window misalignment, or unsafe merging of exports. Second, stored normal observations may be modified through historian compromise or preprocessing faults without changing the number of records. Injection attacks model the first path, while featurenoise injection models the second. The benchmark does not claim that these mechanisms exhaust real data-governance failures; they provide repeatable stress tests with a common budget.

TABLE I  
SUMMARY OF THE EVALUATED TRAINING-TIME CONTAMINATION STRATEGIES.
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>Training-pool manipulation</td><td rowspan=1 colspan=1>Final size</td></tr><tr><td rowspan=1 colspan=1>Random in-jection</td><td rowspan=1 colspan=1>Uniformly sample k observa-tions from attack pool A andappend them as nominal</td><td rowspan=1 colspan=1> $\overline { { N + k } }$ </td></tr><tr><td rowspan=1 colspan=1>Similarity-targetedinjection</td><td rowspan=1 colspan=1>Append the k attack observa-tions closest to the clean cen-troid</td><td rowspan=1 colspan=1> $\overline { { N + k } }$ </td></tr><tr><td rowspan=1 colspan=1>Feature-noiseinjection</td><td rowspan=1 colspan=1>Perturb k selected clean ob-servations with clipped, scaledGaussian noise</td><td rowspan=1 colspan=1>N</td></tr></table>

## B. Contamination Strategies

Let the clean normal training pool be $D _ { \mathrm { c l e a n } } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ and the available attack pool be $A = \{ a _ { j } \} _ { j = 1 } ^ { M }$ . For a trainingpool-relative budget $p ,$ the adversary manipulates $k = \lfloor p N \rfloor$ samples. For the injection attacks, appending k samples yields a final injected fraction $k / ( N { + } k )$ (approximately 9.09% when $p = 1 0 \% )$ ; feature noise instead modifies k of the original N samples. Thus, p denotes a common budget relative to the clean pool rather than an identical final-set fraction across attack families.

For the pointwise detectors and AE, the attack pool is the attack-labeled portion of the stratified training fold and is therefore disjoint from their validation and test observations. LSTM-AE uses this pointwise training-fold attack pool to contaminate its separate contiguous normal training block, as detailed in Section IV-A. This controlled resource isolates training contamination within each implemented protocol but is optimistic relative to end-to-end data-governance failures.

The injected pool should therefore be understood as an evaluation resource rather than an assumption that an operational attacker possesses a perfectly curated attack corpus. It enables controlled comparisons across budgets and seeds. Similarly, the clean validation and test assumption prevents threshold corruption from being conflated with model corruption, but may underestimate failures in pipelines where the same dataquality problem propagates across all splits.

1) Random Injection: The attacker uniformly samples $S \subseteq$ A and forms

$$
D _ { \mathrm { c o n t } } = D _ { \mathrm { c l e a n } } \cup S , \qquad | S | = \operatorname* { m i n } ( k , | A | ) .\tag{1}
$$

This represents unsafe curation or accidental inclusion of attack traces in data later treated as normal.

Random injection does not exploit detector gradients or model parameters. Its strength comes from relabeling attack behavior implicitly as normal during fitting. It provides a simple baseline for measuring whether a detector’s learned support expands around malicious process states.

![](images/ecf93a2555186c2d238ad40d0ec4b266db6d1a492371ecd82ef5fb7473d4672a.jpg)  
Fig. 1. Offline ICS anomaly-detection workflow. Contamination is confined to the normal training pool; validation and test data remain uncontaminated.

2) Similarity-Targeted Injection: The clean centroid is $\mu _ { \mathrm { c l e a n } } ~ = ~ N ^ { - 1 } \textstyle \sum _ { i } x _ { i }$ . Attack samples are ranked by $d _ { j } \ =$ $\lVert \boldsymbol { a } _ { j } - \boldsymbol { \mu } _ { \mathrm { c l e a n } } \rVert _ { 2 }$ , and the k closest samples are inserted. “Targeted” therefore denotes centroid-based data selection, not gradient-based optimization or guaranteed worst-case targeting.

This strategy prefers attack observations that appear globally similar to the clean pool after scaling. The centroid is a deliberately model-agnostic selection rule, which allows the same contaminated training set to be evaluated across all detector families. It may be weak for detectors governed by local structure or temporal context, so its results are interpreted as one reproducible similarity heuristic rather than a worstcase targeted attack.

3) Feature-Noise Injection: The attacker selects $\begin{array} { r l } { Q } & { { } \subseteq } \end{array}$ $D _ { \mathrm { c l e a n } } , \vert Q \vert = k$ , and replaces each selected sample with

$$
\begin{array} { r } { \tilde { x } _ { i } = \mathrm { c l i p } _ { [ 0 , 1 ] } ( x _ { i } + z _ { i } \odot \sigma _ { \mathrm { f e a t } } ) , } \end{array}\tag{2}
$$

where $z _ { i } \sim \mathcal { N } ( 0 , 0 . 1 5 ^ { 2 } I )$ and $\sigma _ { \mathrm { f e a t } }$ is the vector of per-feature standard deviations. Noise is added after MinMax scaling. This preserves training-set size while modeling corruption of stored process values.

Clipping maintains the normalized feature range and the per-feature scale factor prevents a common perturbation magnitude from dominating low-variance variables. The experiment fixes the Gaussian magnitude at 0.15 and varies only the fraction of modified samples. Consequently, a near-null result establishes limited sensitivity to this tested configuration, not general robustness to feature corruption.

The strategies cover unsystematic attack inclusion, centroidtargeted inclusion, and corruption of nominal records. All use the same clean held-out protocol.

## IV. DATASET AND EXPERIMENTAL PROTOCOL

## A. Dataset and Preprocessing

SWaT contains multivariate time-series data from normal operation and attack scenarios [15], [16]. Timestamp columns are removed, source files are aligned on common features, values are coerced to numeric form, invalid rows are dropped, and seven constant columns (P202, P301, P401, P404, P502, P601, and P603) are removed. After the implemented preprocessing, the pointwise benchmark pool contains 449,919 samples, 44 features, and a 12.14% attack ratio. MinMax parameters are fitted on training data only and then applied to validation and test data.

Pointwise detectors and AE use a stratified 70/15/15 split of the pooled normal- and attack-file observations. Only normal samples from the training fold are used for detector fitting; labeled validation observations calibrate the F1-oriented threshold. LSTM-AE instead preserves the source-file order: the first 70% of the normal-operation file is used for training, the next 15% for early stopping, and the final 15% supplies normal test observations. The first half of the attack-period file is used for threshold selection and the second half for testing. For contaminated LSTM-AE runs, attack observations are drawn from the attack-labeled portion of the pointwise training fold and appended to the contiguous normal training block. Windows of length W = 30 are then constructed independently within the training, early-stopping, thresholdselection, and test arrays, so no window crosses a boundary between these arrays. A window is labeled anomalous if any included time step is attacked. This temporal protocol is stricter, but its sequence-level labels are not identical to the pointwise evaluation and must be considered in cross-family comparisons.

TABLE II  
POINTWISE SWAT BENCHMARK POOL AFTER PREPROCESSING.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Processed samples</td><td>449,919</td></tr><tr><td>Retained features</td><td>44</td></tr><tr><td>Normal / attack samples</td><td>395,298 / 54,621</td></tr><tr><td>Attack ratio</td><td>12.14%</td></tr><tr><td>Pointwise split</td><td>70% /  15% / 15%</td></tr><tr><td>Normalization</td><td>Train-fitted min-max to [0, 1]</td></tr></table>

Preprocessing is fitted without using validation or test statistics. Timestamp columns are removed, the normal and attack files are aligned on their common process variables, and nonnumeric or invalid rows are discarded before splitting. The seven removed process variables are constant under the available traces and therefore provide no discriminatory variation. For the pointwise protocol, stratification preserves the attack ratio in validation and test data; only the normal portion of the training fold is passed to a detector. For LSTM-AE, preserving order avoids constructing windows from unrelated time points and prevents a window from crossing the boundary between the normal training sequence and the attack period.

The two protocols consequently use different units of evaluation. A pointwise model assigns one score to one 44- dimensional observation. LSTM-AE assigns reconstruction errors to overlapping windows and labels a window anomalous when at least one included time step belongs to an attack interval. We report this distinction because it affects both the number of evaluated instances and the meaning of a false negative. Cross-family comparisons remain useful at the implemented-configuration level, but should not be read as if every detector received an identical sample representation.

## B. Detectors, Thresholding, and Reproducibility

The retained models are Isolation Forest (IForest), One-Class SVM, Local Outlier Factor (LOF), Cluster-Based Local Outlier Factor (CBLOF), KNN, Histogram-Based Outlier Score (HBOS), PCA, Minimum Covariance Determinant (MCD), Angle-Based Outlier Detection (ABOD), a feedforward autoencoder (AE), and LSTM-AE [29]–[39]. Classical detectors use PyOD; neural detectors use PyTorch. PCA and SVM are tuned, while other classical models use default or near-default configurations. AE and LSTM-AE use staged hyperparameter searches with multi-seed confirmation.

Table III summarizes the heterogeneous scoring mechanisms, final retained configurations, and training inputs, allowing contamination sensitivity to be compared across incompatible definitions of normality without duplicating training-scale information.

## C. Implementation, Tuning, and Compute Policy

The classical benchmark uses PyOD implementations with a common normal-only fitting workflow. IForest, HBOS, CBLOF, and PCA use the full pointwise normal training pool, while SVM, LOF, KNN, ABOD, and MCD use a fixed 50,000- point uniform subsample to control super-linear memory or distance-computation costs. IForest uses 100 trees and the default 256-sample tree subsample. LOF uses 20 neighbors, KNN scores distance to the fifth neighbor, CBLOF uses eight clusters, and ABOD uses its fast 10-neighbor configuration. PCA retains 90% variance, while SVM uses an RBF kernel with ν = 0.01 and gamma = scale.

Table III records the retained settings needed to interpret and reproduce each reported configuration. The PyOD contamination argument is fixed at 0.05 during construction where applicable, but it does not define the final operating threshold. Every detector is converted to binary decisions using the separate validation-calibration procedure described below. Thus, the benchmark’s 1–10% attack budgets describe training-pool manipulation and should not be confused with the library’s internal contamination parameter.

Tuning follows a sensitivity-first policy. A classical parameter family is promoted only when a preliminary change improves F1 by at least 0.03 or reduces FNR by at least 0.05 relative to its default slice. Only PCA and SVM cross this promotion threshold; other classical detectors retain default or near-default configurations. This conservative policy avoids tailoring every model extensively to one benchmark, but it also means that the reported ordering is configuration-level evidence rather than a definitive comparison of optimally tuned families.

The neural models receive staged searches because architecture and optimization choices materially affect their performance. AE is evaluated over 426 configurations spanning architecture, optimizer, regularization, threshold, and scaling choices. LSTM-AE is evaluated over 216 configurations spanning sequence length, hidden capacity, and training settings. Final candidates are re-ranked over three seeds using a validation-only composite of clean F1-score and F1-score under 10% similarity-targeted injection. All hyperparameter screening, staged selection, and multi-seed re-ranking use validation performance only; the held-out test set is accessed only after selecting the final configuration. Table IV records the resulting policy and configurations.

Both neural models minimize mean-squared reconstruction error, use gradient clipping at 1.0, and apply early stopping with patience 15. AE permits up to 100 epochs and LSTM-AE up to 30. The LSTM-AE search also exposes an interaction that a one-factor sweep would miss: W = 30 and hidden size 256 are not the strongest isolated changes, but their joint configuration ranks highest after staged multi-seed evaluation. Dropout values are not interpreted for the selected singlelayer LSTM because PyTorch applies recurrent dropout only between stacked recurrent layers.

TABLE III  
DETECTOR FAMILIES, SCORING MECHANISMS, RETAINED CONFIGURATIONS, AND TRAINING INPUTS USED IN THE COMPARATIVE BENCHMARK. CLASSICAL DETECTORS USE PYOD AND NEURAL DETECTORS USE PYTORCH, AS DESCRIBED IN THE TEXT.
<table><tr><td>Model and family</td><td>Scoring mechanism</td><td>Retained configuration</td><td>Training input</td></tr><tr><td>IForest Tree ensemble</td><td>Average path length in random isolation trees</td><td>100 trees; tree subsample 256; contamination 0.05; seed-controlled initial- ization</td><td>Full normal pool</td></tr><tr><td>SVM Kernel boundary</td><td>RBF ν-SVM support boundary, ν = 0.01</td><td>RBF one-class SVM; ν = 0.01; gamma = scale; contamination 0.05</td><td>50K subsample</td></tr><tr><td>LOF Local density</td><td>Density ratio relative to a 20- neighbor neighborhood</td><td>Novelty mode; 20 neighbors; Minkowski distance; contamination 0.05</td><td>50K subsample</td></tr><tr><td>CBLOF</td><td>Cluster assignment, cluster size,</td><td>Eight clusters;  $\alpha = 0 . 9 ; \beta = 5 ;$  seed-controlled initialization</td><td>Full normal pool</td></tr><tr><td>Cluster/density KNN</td><td>and distance Distance to the fifth-nearest</td><td>Fifth-nearest-neighbor distance; largest-distance score; Minkowski metric</td><td>50K subsample</td></tr><tr><td>Distance HBOS</td><td>training point Feature-wise histogram outlier</td><td>10 bins; α = 0.1; tolerance 0.5; feature-wise histogram scoring</td><td>Full normal pool</td></tr><tr><td>Statistical PCA</td><td>scores Reconstruction error at 90% re-</td><td>90% retained variance; reconstruction-error scoring; seed-controlled initial-</td><td>Full normal pool</td></tr><tr><td>Linear subspace MCD</td><td>tained variance Robust Mahalanobis distance</td><td>ization Default support fraction; robust covariance and Mahalanobis-distance scor-</td><td>50K subsample</td></tr><tr><td>Robust covariance ABOD</td><td>from a trimmed center Angle variance with 10 neigh-</td><td>Fast variant; 10 neighbors; angle-variance scoring</td><td>50K subsample</td></tr><tr><td>Geometric AE</td><td>bors Bottleneck reconstruction error</td><td>Dimensions [256, 128, 64]; BatchNorm, LeakyReLU, dropout 0.1; Adam</td><td>Full normal pool</td></tr><tr><td>Neural pointwise LSTM-AE</td><td>Sequence reconstruction error</td><td> $5 \times 1 0 ^ { - 4 } ;$  batch 2048; MSE; maximum 100 epochs One 256-unit LSTM layer;  $W = 3 0 ; { \mathrm { ~ A d a m W ~ } } 1 0 ^ { - 3 } ;$  batch 256; MSE;</td><td>Full sequence set</td></tr></table>

TABLE IV  
TUNING POLICY AND FINAL MODEL-CONFIGURATION SUMMARY.
<table><tr><td rowspan=1 colspan=1>Group</td><td rowspan=1 colspan=1>Search policy</td><td rowspan=1 colspan=1>Retained configuration</td></tr><tr><td rowspan=1 colspan=1>Classical</td><td rowspan=1 colspan=1>Sensitivity-first;  onlyPCA  and   SVMpromoted</td><td rowspan=1 colspan=1>PCA: 90% variance; SVM: RBF,ν = 0.01</td></tr><tr><td rowspan=1 colspan=1>AE</td><td rowspan=1 colspan=1>426 staged configura-tions; multi-seed confir-mation</td><td rowspan=1 colspan=1>[256,128, 64],    LeakyReLU,dropout 0.1, Adam $5 \ \times \ 1 0 ^ { - 4 } .$ batch 2048</td></tr><tr><td rowspan=1 colspan=1>LSTM-AE</td><td rowspan=1 colspan=1>216          stagedconfigurations;   jointsequence/capacitysearch</td><td rowspan=1 colspan=1> $\begin{array} { r l r } { \overline { { W \mathrm { ~ \small ~ \alpha ~ } } } } & { { } = } & { 3 0 , } \end{array}$ hidden size 256,AdamW $1 0 ^ { - 3 }$ , batch 256</td></tr></table>

Experiments were run on the Narval cluster of the Digital Research Alliance of Canada. Classical jobs used six Intel Xeon CPU cores and 48 GiB of RAM; neural jobs additionally used one NVIDIA A100-SXM4-40 GB GPU. Fixed seeds 42, 123, and 456 are applied to splitting, contamination, initialization, and training. Each run is checkpointed independently before aggregation, supporting restartable execution and consistent generation of the reported tables and figures.

All detectors output continuous anomaly scores. An F1- oriented threshold is selected on validation data and applied unchanged to held-out test data. The primary metric is F1- score; FNR is reported because missed attacks are safetyrelevant. Experiments use seeds 42, 123, and 456 and trainingpool-relative budgets of 1%, 3%, 5%, and 10%. Execution produced 430 successful runs: 33 retained clean baselines, 396 retained contamination runs, and one SOD clean baseline used only to document exclusion. SOD was removed from the main benchmark because its clean ROC-AUC indicated inverted score ordering under the evaluated configuration, making contamination-degradation analysis ill-posed [40]. Its available baseline also showed severe precision collapse (F1 $= 0 . 2 0 1 3 .$ , precision = 0.1130, recall = 0.9218) and a 2011.4- s training time; no SOD contamination result is included in the main analysis. All reported comparative analyses therefore use 429 retained runs.

For every clean or contaminated model within each seed and detector configuration, a new threshold is calibrated using only the same clean labeled validation split. That threshold is then frozen for the corresponding held-out test evaluation. This prevents test-set optimization and measures whether each contaminated model’s anomaly-score distribution remains compatible with an operating point selected without contaminated calibration data. Precision, recall, F1-score, FNR, ROC-AUC, and average precision are retained in the experiment outputs; F1 and FNR are emphasized because the former balances the two error directions while the latter directly exposes missed attacks.

## V. RESULTS

## A. Clean Performance and Contamination Robustness

Table V combines clean performance, measured training cost, and worst-case F1 loss. LSTM-AE has the highest clean F1-score (0.8813), followed by LOF, ABOD, and AE. Yet the clean-data ranking changes substantially under contamination. LOF, KNN, and ABOD each lose more than 0.64 F1 in their worst condition, while IForest, HBOS, PCA, and SVM remain within 0.068 of their clean baselines. AE and LSTM-AE occupy an intermediate position.

The clean scores are descriptive means over three seeds rather than significance-tested orderings. The leading group is narrow: LSTM-AE, LOF, ABOD, and AE all lie between 0.867 and 0.881. PCA and SVM are mid-table at approximately 0.813–0.814, but require far less training time than

ABOD, KNN, AE, or LSTM-AE. Clean evaluation therefore defines the starting point for robustness analysis but does not identify a single operational winner.

The standard deviations also distinguish stable clean estimates from seed-sensitive ones. Most detectors vary by less than 0.01 F1 across the three seeds, whereas LOF and MCD vary more noticeably because their fitting pools are subsampled. These differences are small relative to the largest contamination losses, but they reinforce why degradation is computed from seed-averaged contaminated and clean scores rather than from a single favorable run.

Figure 2 shows distinct attack responses. Feature noise at σ = 0.15 changes little, whereas 1% random injection reduces LOF, KNN, ABOD, and LSTM-AE from 0.876, 0.848, 0.874, and 0.881 to approximately 0.398, 0.335, 0.305, and 0.440. AE degrades progressively; similarity-targeted injection becomes severe for selected distance- and density-based detectors at 10%.

The earliest failures are operationally important because a small clean-pool-relative budget is sufficient to reverse the clean ranking. LOF, KNN, and ABOD begin among the five strongest clean configurations, yet at 1% random injection each loses more than half of its original F1-score. Their reliance on local neighborhoods, distances, or angles makes them sensitive when attack samples are incorporated into the reference geometry. In contrast, the more aggregated representations of PCA, HBOS, IForest, and SVM change much less under the same evaluated contamination.

The neural response is neither uniformly fragile nor uniformly robust. AE deteriorates gradually and reaches its worst condition at 10% random injection, where F1 falls to approximately 0.566. LSTM-AE has the highest clean F1- score but drops to approximately 0.440 at only 1% random injection. Its response to similarity-targeted injection is much milder through 5%, indicating that the centroid-based pointwise selection rule does not transfer directly into a worstcase sequence attack. The neural results therefore support an intermediate robustness interpretation and demonstrate why attack-family labels should not be treated as equivalent across data representations.

![](images/5b1515fda64312a120aeedaf4f4f2ed2d81b1ecfbcbc7a9a49d8378d22ce977c.jpg)  
Fig. 2. F1-score versus training-pool-relative contamination budget. Injection attacks cause detector-specific degradation; feature noise has a limited effect at the tested magnitude.

Figure 3 complements the F1 analysis with missed-detection behavior and is discussed in Section V-C. In Figs. 2–4, the labels HISTOGRAM, CLUSTER, and AUTOENCODER denote HBOS, CBLOF, and AE, respectively; axes labeled “contamination rate” report the clean-pool-relative budget p defined in Section III-B.

The heatmap in Fig. 4 makes the model dependence explicit. IForest, HBOS, PCA, and SVM remain close to their clean values. Random injection sharply harms LOF, KNN, ABOD, and LSTM-AE across budgets, whereas AE degrades more gradually. Similarity-targeted injection produces delayed failures for LOF, KNN, and ABOD. These results show why a clean-only leaderboard is inadequate for adversarial model selection.

## B. Attack-Type Comparison

The three strategies produce qualitatively different degradation patterns. Random injection is the dominant family overall: it causes the earliest large drops and affects both pointwise neighborhood methods and the sequence model. Because the injected samples span the available attack pool rather than only its center-nearest region, they introduce diverse malicious states into the normal reference data. This pattern is consistent with an expansion or distortion of the learned nominal region for detectors whose scores depend strongly on the training geometry.

![](images/0bdb88b619e7cb0058e35c88d23c256912092df86a86c791bec67cd066d64aad.jpg)  
Fig. 3. FNR versus training-pool-relative contamination budget. LOF, KNN, ABOD, and LSTM-AE enter the highest missed-detection region under one or more injection settings.

TABLE V  
CLEAN PERFORMANCE (MEAN ± SD OVER THREE SEEDS), MEAN OBSERVED TRAINING TIME, AND WORST-CASE F1 DROP ACROSS TWELVE ATTACK-BUDGET COMBINATIONS. WORST-CASE DROP IS COMPUTED FROM SEED-AVERAGED F1 VALUES.
<table><tr><td>Detector</td><td>Clean F1</td><td>Clean FNR</td><td>Train (s)</td><td>Worst F1 drop</td></tr><tr><td>LSTM-AE</td><td>0.8813 ± 0.0032</td><td>0.2027 ± 0.0068</td><td>139.9</td><td>0.442</td></tr><tr><td>LOF</td><td>0.8761 ± 0.0208</td><td>0.1373 ± 0.0312</td><td>36.5</td><td>0.660</td></tr><tr><td>ABOD</td><td>0.8744 ± 0.0039</td><td>0.2012 ± 0.0171</td><td>249.8</td><td>0.674</td></tr><tr><td>AE</td><td>0.8674 ± 0.0029</td><td>0.2030 ± 0.0082</td><td>117.1</td><td>0.302</td></tr><tr><td>KNN</td><td>0.8482 ± 0.0016</td><td>0.2259 ± 0.0077</td><td>264.2</td><td>0.648</td></tr><tr><td>SVM</td><td>0.8144 ± 0.0019</td><td>0.2922 ± 0.0015</td><td>17.1</td><td>0.068</td></tr><tr><td>PCA</td><td>0.8134 ± 0.0038</td><td>0.3078 ± 0.0054</td><td>1.5</td><td>0.045</td></tr><tr><td>HBOS</td><td>0.7712 ± 0.0037</td><td>0.3425 ± 0.0088</td><td>4.1</td><td>0.007</td></tr><tr><td>CBLOF</td><td>0.7543 ± 0.0048</td><td>0.3874 ± 0.0034</td><td>6.9</td><td>0.182</td></tr><tr><td>IForest</td><td>0.7331 ± 0.0022</td><td>0.4162 ± 0.0036</td><td>4.5</td><td>0.000</td></tr><tr><td>MCD</td><td>0.7260 ± 0.0129</td><td>0.3211 ± 0.0559</td><td>6.2</td><td>0.126</td></tr></table>

F1 Change Across Attack Scenarios  
![](images/7bfd9247642fed5d87864cca185cdf1c41c73c75debc14494ba2f80efedea599.jpg)

# Attack Type and Contamination Rate

Fig. 4. Change in F1-score relative to the clean baseline. Stable classical models remain near zero, while injection attacks expose severe fragility in several high-clean-F1 detectors.

Similarity-targeted injection is more detector-dependent. At low budgets it is often less harmful than random injection, but LOF, KNN, and ABOD show delayed failures as the budget increases. The centroid rule selects attack samples that are globally close to the average clean state, yet global proximity does not guarantee local proximity in every detector’s representation. The strategy should therefore be viewed as a repeatable stealth-oriented heuristic. It does not establish that random sampling is intrinsically stronger than an optimized targeted attack.

Feature-noise injection remains close to the clean baseline across the grid. Three aspects limit the conclusion: the magnitude is fixed at σ = 0.15, perturbations are clipped to the normalized range, and only selected normal observations are modified. The result shows that the evaluated models tolerate this bounded configuration substantially better than attacksample injection. It does not show general robustness to biased, coordinated, temporally persistent, or physically constrained sensor corruption.

## C. Safety, Cost, and Practical Trade-offs

Figure 3 shows that FNR should be interpreted together with F1-score rather than in isolation. ABOD, KNN, and LOF exceed approximately 0.70 FNR in at least one contamination setting, while LSTM-AE reaches a lower but still concerning peak under random injection. By contrast, PCA, SVM, HBOS, AE, and MCD remain in the lower-FNR group, and IForest stays comparatively stable despite its lower clean baseline.

At some contamination rates, a lower FNR may reflect overly permissive decision behavior, which reduces missed detections at the cost of precision. For that reason, FNR is most informative when read together with F1-score rather than as a standalone safety measure.

## D. Overall Trade-offs Across Performance, Robustness, and Cost

Taken together, the results show that the ordering by cleandata F1-score is materially different from the ordering implied by robustness and training cost. LSTM-AE, LOF, ABOD, and AE form the strongest clean-data group. However, several of these models remain fragile once the normal training pool is contaminated, particularly LOF, KNN, and ABOD, and to a lesser but still meaningful degree AE and LSTM-AE. By contrast, PCA and SVM do not lead the clean baseline table, yet both combine moderate clean performance with small worst-case F1 loss and much lower training cost. IForest is also notable for its stability across the contamination grid, although its clean baseline remains lower than that of PCA and SVM.

The most defensible practical conclusion is therefore not that one detector dominates on every metric, but that PCA and SVM provide a favorable observed balance among clean performance, contamination robustness, safety-relevant FNR behavior, and computational cost under the evaluated setup. Because hardware and training scales differ, this balance is configuration-specific rather than a formal multi-objective ranking. AE and LSTM-AE remain intermediate rather than robust, and their contamination-time behavior is less dependable than that of the most stable classical configurations.

## E. Limitations

Several limitations should be kept in mind when interpreting the benchmark. First, the study is conducted on a single ICS dataset, so the findings should be interpreted at the benchmark level rather than as population-wide claims about all safety-critical ICS environments. Second, the evaluated attack families are contamination-style training-time scenarios rather than optimization-based worst-case poisoning attacks. Third, the benchmark assumes clean validation and test splits, which isolates training-time contamination effects but may be optimistic relative to workflows in which curation errors propagate beyond the training data. Fourth, the benchmark does not apply equally deep hyperparameter optimization to every detector, so cross-detector comparisons should be interpreted as configuration-level robustness evidence rather than definitive family-level rankings. Finally, the pointwise detectors and sequence model use different split structures: the pointwise random split can place temporally adjacent observations in different folds, whereas LSTM-AE preserves a contiguous protocol. The limited feature-noise effect applies only at σ = 0.15.

## VI. CONCLUSION

This paper compared eleven anomaly detectors under offline training-time contamination on SWaT. Across three attack families and four budgets, strong clean performance did not imply robustness. LOF, KNN, and ABOD ranked near the top on clean data but degraded most severely after attack-sample injection. AE and LSTM-AE showed intermediate robustness, with random injection as the dominant neural failure mode. PCA, SVM, HBOS, and IForest were comparatively stable; PCA and SVM balanced clean performance, robustness, and cost under the evaluated protocols. Feature noise had limited impact at the tested magnitude, whereas injection was more consequential. These findings motivate integrity controls for offline ICS training data and model selection procedures that explicitly account for contamination. Future work should evaluate additional ICS environments, stronger adaptive attacks, and defenses that jointly optimize accuracy, robustness, and operational cost.

## REFERENCES

[1] J. Goh, S. Adepu, M. Tan, and Z. S. Lee, “Anomaly detection in cyber physical systems using recurrent neural networks,” in 2017 IEEE 18th International Symposium on High Assurance Systems Engineering (HASE), 2017, pp. 140–145.

[2] A. M. Y. Koay, R. K. L. Ko, H. Hettema, and K. Radke, “Machine learning in industrial control system security: Current landscape, opportunities and challenges,” Journal of Intelligent Information Systems, vol. 60, pp. 377–405, 2023.

[3] G. R. M. R., C. M. Ahmed, and A. Mathur, “Machine learning for intrusion detection in industrial control systems: Challenges and lessons from experimental evaluation,” Cybersecurity, vol. 4, no. 1, p. 27, 2021.

[4] B. I. P. Rubinstein, B. Nelson, L. Huang, A. D. Joseph, S. hon Lau, S. Rao, N. Taft, and J. D. Tygar, “ANTIDOTE: Understanding and defending against poisoning of anomaly detectors,” in Proceedings of the 9th ACM SIGCOMM Conference on Internet Measurement (IMC), 2009, pp. 1–14.

[5] A. E. Cina, K. Grosse, A. Demontis, S. Vascon, W. Zellinger, B. A.\` Moser, A. Oprea, B. Biggio, M. Pelillo, and F. Roli, “Wild patterns reloaded: A survey of machine learning security against training data poisoning,” ACM Computing Surveys, vol. 55, no. 13s, pp. 1–39, 2023.

[6] M. Kravchik and A. Shabtai, “Detecting cyber attacks in industrial control systems using convolutional neural networks,” in Proceedings of the 2018 Workshop on Cyber-Physical Systems Security and PrivaCy (CPS-SPC), 2018, pp. 72–83.

[7] J. Inoue, Y. Yamagata, Y. Chen, C. M. Poskitt, and J. Sun, “Anomaly detection for a water treatment system using unsupervised machine learning,” in 2017 IEEE International Conference on Data Mining Workshops (ICDMW), 2017, pp. 1058–1065.

[8] J. Kim, J. Yun, and H. C. Kim, “Anomaly detection for industrial control systems using sequence-to-sequence neural networks,” in Computer Security, ser. Lecture Notes in Computer Science. Springer, 2020, vol. 11980, pp. 3–18.

[9] M. Kravchik and A. Shabtai, “Efficient cyber attack detection in industrial control systems using lightweight neural networks and PCA,” IEEE Transactions on Dependable and Secure Computing, vol. 19, no. 4, pp. 2179–2197, 2022.

[10] G. Zizzo, C. Hankin, S. Maffeis, and K. Jones, “Adversarial attacks on time-series intrusion detection for industrial control systems,” in 2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom), 2020, pp. 899–910.

[11] A. Erba, R. Taormina, S. Galelli, M. Pogliani, M. Carminati, S. Zanero, and N. O. Tippenhauer, “Constrained concealment attacks against reconstruction-based anomaly detectors in industrial control systems,” in Proceedings of the 36th Annual Computer Security Applications Conference, ser. ACSAC ’20. New York, NY, USA: Association for Computing Machinery, 2020, pp. 480–495. [Online]. Available: https://doi.org/10.1145/3427228.3427660

[12] E. Anthi, L. Williams, M. Rhode, P. Burnap, and A. Wedgbury, “Adversarial attacks on machine learning cybersecurity defences in industrial control systems,” Journal of Information Security and Applications, vol. 58, p. 102717, 2021.

[13] M. Kravchik, B. Biggio, and A. Shabtai, “Poisoning attacks on cyber attack detectors for industrial control systems,” in Proceedings of the 36th Annual ACM Symposium on Applied Computing (SAC), 2021, pp. 116–125.

[14] M. Kravchik, L. Demetrio, B. Biggio, and A. Shabtai, “Practical evaluation of poisoning attacks on online anomaly detectors in industrial control systems,” Computers & Security, vol. 122, p. 102901, 2022.

[15] A. P. Mathur and N. O. Tippenhauer, “SWaT: A water treatment testbed for research and training on ics security,” in Proceedings of the 2016 International Workshop on Cyber-physical Systems for Smart Water Networks (CySWater), 2016, pp. 31–36.

[16] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A dataset to support research in the design of secure water treatment systems,” in Critical Information Infrastructures Security, ser. Lecture Notes in Computer Science. Springer, 2017, vol. 10242, pp. 88–99.

[17] A. L. P. G <sup>´</sup> omez, L. F. Maim ´ o, A. H. Celdr ´ an, and F. J. G. Clemente,´ “MADICS: A methodology for anomaly detection in industrial control systems,” Symmetry, vol. 12, no. 10, p. 1583, 2020.

[18] G. O. Campos, A. Zimek, J. Sander, R. J. G. B. Campello, B. Micenkova,´ E. Schubert, I. Assent, and M. E. Houle, “On the evaluation of unsupervised outlier detection: Measures, datasets, and an empirical study,” Data Mining and Knowledge Discovery, vol. 30, no. 4, pp. 891–927, 2016.

[19] S. Schmidl, P. Wenig, and T. Papenbrock, “Anomaly detection in time series: A comprehensive evaluation,” Proceedings of the VLDB Endowment, vol. 15, no. 9, pp. 1779–1797, 2022.

[20] P. Wenig, S. Schmidl, and T. Papenbrock, “Timeeval: A benchmarking toolkit for time series anomaly detection algorithms,” Proceedings of the VLDB Endowment, vol. 15, no. 12, pp. 3678–3681, 2022.

[21] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon, “Towards a rigorous evaluation of time-series anomaly detection,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 36, no. 7, 2022, pp. 7194– 7201.

[22] A. Huet, J. M. Navarro, and D. Rossi, “Local evaluation of time series anomaly detection algorithms,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 635–645.

[23] B. Biggio, B. Nelson, and P. Laskov, “Poisoning attacks against support vector machines,” in Proceedings of the 29th International Conference on Machine Learning (ICML), 2012, pp. 1467–1474.

[24] L. Munoz-Gonz˜ alez, B. Biggio, A. Demontis, A. Paudice, V. Wongras-´ samee, E. C. Lupu, and F. Roli, “Towards poisoning of deep learning algorithms with back-gradient optimization,” in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security (AISec), 2017, pp. 27–38.

[25] J. Steinhardt, P. W. Koh, and P. Liang, “Certified defenses for data poisoning attacks,” in Advances in Neural Information Processing Systems, 2017.

[26] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer, T. Dumitras, and T. Goldstein, “Poison frogs! targeted clean-label poisoning attacks on neural networks,” in Advances in Neural Information Processing Systems, 2018.

[27] S. Abedzadeh and S. Bhattacharjee, “Mitigating impact of data poisoning attacks on CPS anomaly detection with provable guarantees,” Information, vol. 16, no. 6, p. 428, 2025.

[28] Q. Jiang, Y. Zu, Z. Zhu, W. Zhong, Y. Xu, Z. Qian, and X. Zhang, “Dy-

namic stealthy backdoor attack against anomaly detectors in industrial control systems,” Information Sciences, vol. 735, p. 123066, 2026.

[29] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413–422.

[30] B. Scholkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C.¨ Williamson, “Estimating the support of a high-dimensional distribution,” Neural Computation, vol. 13, no. 7, pp. 1443–1471, 2001.

[31] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “LOF: Identifying density-based local outliers,” in Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, 2000, pp. 93–104.

[32] Z. He, X. Xu, and S. Deng, “Discovering cluster-based local outliers,” Pattern Recognition Letters, vol. 24, no. 9–10, pp. 1641–1650, 2003.

[33] S. Ramaswamy, R. Rastogi, and K. Shim, “Efficient algorithms for mining outliers from large data sets,” in Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, 2000, pp. 427–438.

[34] M. Goldstein and A. Dengel, “Histogram-based outlier score (HBOS): A fast unsupervised anomaly detection algorithm,” in KI-2012: Poster and Demo Track, 2012.

[35] M.-L. Shyu, S.-C. Chen, K. Sarinnapakorn, and L. Chang, “A novel anomaly detection scheme based on principal component classifier,” in Proceedings of the IEEE Foundations and New Directions of Data Mining Workshop, 2003, pp. 172–179.

[36] P. J. Rousseeuw and K. V. Driessen, “A fast algorithm for the minimum covariance determinant estimator,” Technometrics, vol. 41, no. 3, pp. 212–223, 1999.

[37] H.-P. Kriegel, M. Schubert, and A. Zimek, “Angle-based outlier detection in high-dimensional data,” in Proceedings ofthe 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2008, pp. 444–452.

[38] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” in Proceedings ofthe MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, 2014, pp. 4:1–4:11.

[39] T. Kieu, B. Yang, C. Guo, and C. S. Jensen, “Outlier detection for time series with recurrent autoencoder ensembles,” in Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 2019, pp. 2725–2732.

[40] H.-P. Kriegel, P. Kroger, E. Schubert, and A. Zimek, “Outlier detection¨ in axis-parallel subspaces of high dimensional data,” in Advances in Knowledge Discovery and Data Mining, ser. Lecture Notes in Computer Science. Springer, 2009, vol. 5476, pp. 831–838.