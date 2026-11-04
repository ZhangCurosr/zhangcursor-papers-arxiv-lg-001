Research Article

# Hybrid acoustic and pretrained embedding feature fusion for unmanned aerial vehicle detection using stacking ensemble learning under a leakage-aware evaluation protocol

Zahid Iqbal<sup>1,</sup>\*

<sup>1</sup> Department of Computer Science, Faculty of Computing and Information Technology, University of Gujrat, 50700, Punjab, Pakistan

\* Corresponding author: Zahid Iqbal

## Abstract

Introduction: Small multirotor drones produce a characteristic acoustic signature, and microphones offer a low-cost way to detect them for airspace safety and drone-traffic monitoring. High accuracies have been reported on public drone audio benchmarks, often with random clip-level splits of datasets built from a few long recordings and from pieces of existing sound collections, so that pieces of the same source can appear in both training and test data.

Materials and methods: We study the binary task of the public DroneAudioDataset (11,704 one-second clips). Its 1,332 drone clips come from 257 five-second chunks of 7 recordings, and 96.4% of its non-drone clips are pieces of the 2,000 clips of the ESC-50 environmental sound dataset. A grouped, source-disjoint protocol (five folds, three repeats) is compared with a random split. Each clip is described by 57 hand-crafted acoustic descriptors, reduced by a hybrid minimum-redundancy maximum-relevance and ReliefF selection or by NSGA-II multi-objective search, and by a 1024-dimensional embedding from the pretrained YAMNet network. Random Forest and XGBoost learners on both views are combined by a stacking ensemble

Results: Under the grouped protocol, the two-view stack reached a mean accuracy of 95.29% (standard deviation 1.62 percentage points over 15 test folds), a specificity of 98.29% and an area under the receiver operating characteristic curve of 0.990, and it classified 98.78% of unseen ESC-50 source clips correctly as non-drone. The random split overstated accuracy by 1.89 to 3.22 percentage points for the four models compared. Adding YAMNet embeddings to the hand-crafted view raised accuracy by 1.11 percentage points (higher in 12 of 15 folds and lower in none). Hybrid selection matched the accuracy of all 44 baseline descriptors with 16 to 18 features, and NSGA-II retained 4 to 6 features at lower accuracy. The stack processed one clip in 51.0 ms on a single CPU thread.

Conclusions: Source-disjoint evaluation is needed for a fair benchmark on this dataset, and fusing hand-crafted descriptors with pretrained embeddings gives a small but consistent gain at modest computational cost.

Keywords: unmanned aerial vehicle detection; acoustic sensing; feature selection; mRMR; ReliefF; NSGA-II; YAMNet; stacking ensemble; data leakage; grouped cross-validation

## 1. Introduction

Small unmanned aerial vehicles (UAVs) have become inexpensive and easy to fly, and they are used for inspection, mapping, delivery and recreation. Their growing number in low-altitude airspace, near airports and over public events has made automatic drone detection a practical need for airspace safety and traffic monitoring. Radar, radio-frequency monitoring, cameras and microphones have all been applied to the task. Acoustic sensing is attractive because the motors and rotors of a multirotor produce a stable harmonic signature [1], microphones are cheap, and sound does not depend on lighting or a clear line of sight [2].

Acoustic drone detectors fall into two broad families. The first trains deep networks on time–frequency representations such as log-mel spectrograms or Mel-frequency cepstral coefficients (MFCCs). Djurek et al. showed that quadcopter sound carries harmonic structure that changes with the flight regime [1], and Dumitrescu et al. developed an acoustic system for UAV detection [2]. Al-Emadi et al. recorded the DroneAudioDataset and compared deep network architectures for drone detection and identification [3], and later enlarged the data with generative adversarial networks [4]. Casabianca and Zhang combined several deep networks by late fusion [5], and Cabrera-Ponce et al. detected nearby UAVs with a microphone array carried on board a UAV [6].

The second family computes a compact set of clip-level descriptors and applies classical learners. Tian and Yu recognised UAVs in outdoor areas in real time with a Random Forest algorithm [7], Wang et al. used engineered acoustic features to detect whether a UAV carries a payload [8], and Tejera-Berengue et al. analysed how distance and environmental conditions affect machine-learning detectors [9].The same pattern of compact descriptors, feature selection and tree-based ensembles is common in other sensing and data-analysis tasks, for example machine-learning systems built on wearable sensor data for continuous healthcare monitoring [10] and the classification of malicious accounts on the Ethereum network with information-gain selection and gradient-boosted trees [11]. Ensemble models have also been used to handle strongly imbalanced classes in industrial inspection, for example in wafer defect classification [12], which parallels the class-weighted configuration studied here. Deep models reach high accuracy but need more data and computation; descriptor-based models are cheap and interpretable but depend on the choice of features. Acoustic detection is also often combined with other modalities: deep learning on radio-frequency signals has been used for end-to-end detection and identification [13], and Frid et al. fused radio-frequency and acoustic features [14]. The present study stays within the acoustic modality and fuses two views of the same signal.

Because descriptor-based detectors depend on their features, the selection step matters. Minimum-redundancy maximumrelevance (mRMR) selection adds features with high mutual information with the class and low redundancy with features already chosen [15], whereas ReliefF scores features by how well they separate near neighbours of different classes [16, 17]. Evolutionary multi-objective search, typically with NSGA-II [18], treats selection as a trade-off between error and subset size [19], and population-based search more generally has been applied to optimisation problems in dynamic environments [20]. At the same time, networks pretrained on large audio corpora such as AudioSet [21] now provide general-purpose embeddings at little cost. YAMNet [22] is a MobileNetV1-style network [23] trained with supervision on the 521 AudioSet classes, and its 1024-dimensional penultimate activations are used as transferable embeddings [22, 24]. Because it was not trained for drone sounds, we measure whether it adds information to a well-chosen set of hand-crafted descriptors.

A second question concerns evaluation. Public drone audio datasets are often distributed as short clips cut from a few longer recordings, and their non-drone class is often assembled from existing sound collections. When such clips are split at random, pieces of one recording or of one source sound fall on both sides of the split, and test accuracy then rewards recognising a recording rather than detecting a drone. Leakage of this kind is a known cause of over-optimistic results across machinelearning research [25]. Shulman recently grouped the drone clips of the DroneAudioDataset by recording session and the non-drone clips by their source, and reported lower scores under grouped cross-validation than under file-level grouping [26]. The binary task of this dataset [3] is a clear case: 96.4% of its 10,372 non-drone clips are one-second pieces of the 2,000 clips of the ESC-50 dataset [27], and its 1,332 drone clips come from 257 five-second chunks of only 7 recordings. ESC-50 therefore cannot act as an independent test set for models trained on this dataset, and random splits overstate performance for both classes.

This paper revisits hybrid feature selection and two-view feature fusion for acoustic drone detection under an evaluation protocol that keeps every source on one side of each split. It compares a random split with a grouped protocol on identica test clips for every model, measures how the split affects specificity on unseen ESC-50 sources, and combines the protoco with feature-selection and fusion experiments. The study makes five contributions. First, we define a leakage-aware protoco for the DroneAudioDataset binary task that groups drone clips by recording chunk and non-drone clips by their source clip, following the official ESC-50 folds, and we measure side by side how much a random split inflates accuracy and ESC-50 specificity (Section 3.1). Second, we combine mRMR and ReliefF in a hybrid filter selection, with an ablation against each filter and against the full feature set and a selection-frequency analysis over 15 test folds (Section 3.2). Third, a multi-objective search with NSGA-II maps the trade-off between error and the number of features (Section 3.2). Fourth, a two-view stacking ensemble [28] of Random Forest [29] and XGBoost [30] learners on hand-crafted descriptors and YAMNet embeddings is evaluated with a fusion ablation, paired exact McNemar tests [31] with Holm correction [32], and bootstrap confidence intervals (Section 3.3). Fifth, a per-class analysis identifies the ESC-50 sounds most often mistaken for a drone, and single-thread CPU latency and model size are measured (Section 3.4).

## 2. Materials and methods

Figure 1 summarises the system. Each one-second clip is described in two ways: by 57 hand-crafted descriptors, reduced by hybrid mRMR+ReliefF selection, and by a 1024-dimensional YAMNet embedding. Random Forest and XGBoost learners are trained on each view, and a logistic-regression meta-learner combines their out-of-fold probabilities. A separate NSGA-II search yields a compact subset for a single lightweight classifier. All selection and training steps are fitted on the training part of each fold only, and every model is evaluated on the same test clips, so that all comparisons in Section 3 are paired.

## 2.1. Dataset and leakage-aware evaluation protocol

We use the binary part of the DroneAudioDataset [3, 33], which holds 1,332 drone clips and 10,372 non-drone clips of about one second, mono, at 16 kHz. According to its documentation, the drone clips are propeller noise recorded indoors from two quadcopters, labelled Bebop and Membo in the file names [33]. The file names identify the five-second chunk and the recording of every drone clip and the source of every non-drone clip: 10,001 non-drone clips are one-second pieces of the 2,000 ESC-50 clips [27], 70 are pieces of background-noise recordings of the Speech Commands dataset [34], and 301 are silence clips. Table 1 lists the composition. The dataset was obtained from its public repository (commit ec6139b3) and is used under its academic research licence.

Table 1. Composition of the binary DroneAudioDataset task (1-s clips, 16 kHz) and the groups used to keep each source on one side of every split.
<table><tr><td>Subset</td><td>Clips</td><td>Leakage groups</td></tr><tr><td>Drone (positive)</td><td>1,332</td><td>257 five-second chunks from 7 recordings</td></tr><tr><td>Non-drone: ESC-50 pieces</td><td>10,001</td><td>2,000 ESC-50 source clips (50 classes)</td></tr><tr><td>Non-drone: Speech Commands background noise</td><td>70</td><td>5 noise recordings</td></tr><tr><td>Non-drone: silence</td><td>301</td><td>1 group</td></tr><tr><td>Total</td><td>11,704</td><td></td></tr></table>

![](images/a18f844721ad36e12937a810cb670c340b294a028e6d351dea122765baace630.jpg)  
Evaluation: 5 source-disjoint folds × 3 repeats; drone pieces grouped by recording chunk, ESC-50 pieces by source clip (official ESC-50 folds).  
Figure 1. Overview of the detection pipeline. Feature selection, base learners and the meta-learner are fitted on the training folds only.

Two properties of this composition shape the evaluation design. The drone class rests on only 7 recordings, so neighbouring one-second pieces of one recording are acoustically very similar. The non-drone class is dominated by ESC-50, whose five-second clips each contribute about five pieces. A split that ignores these sources measures, in part, how well a mode recognises a recording it has already heard.

Two protocols are therefore compared (Figure 2). In the random piece-level split, clips are split 80/20 at random, stratified by class, three times with different seeds. Pieces of one recording chunk and of one ESC-50 clip can then appear in both training and test data, and every ESC-50 clip used for testing has contributed pieces to training. This arm reproduces the clip-leve evaluation that such datasets invite, and it runs the same models with the same settings as the main protocol, so that the two protocols differ only in how clips are assigned to the test set.

Random piece-level split

![](images/7407fdeb1c9ab14041717dcefef465507621900456cb727f0a24f2a72fa4e006.jpg)  
Grouped, source-disjoint split (this work)

![](images/fbfad90f819f5eab4c4e0314f43b14c23cbc21894542fe20c3024c6a652b6ad6.jpg)  
Figure 2. Random piece-level splitting places pieces of one source on both sides of the split (left); the grouped protocol keeps every recording chunk and every ESC-50 source clip on one side (right).

The grouped, source-disjoint split is the main protocol. In it, ESC-50 pieces keep the official ESC-50 fold of their source clip, so all pieces of a clip always fall in the same fold. Drone pieces are grouped by their five-second chunk (257 groups), and the chunks of each of the 7 recordings are spread over the five folds so that the folds receive similar numbers of clips from every recording. The Speech Commands noise sources and the silence group are assigned to folds as whole groups. Each fold is the test fold once, and the assignment is repeated three times with different seeds, giving 15 test folds.

Within each training pool all drone clips are kept, and non-drone clips are sampled at a 1:1 ratio (configuration B) or at a 1:3 ratio with class-weighted learners (configuration E). The 1:3 ratio exposes the learners to a wider range of non-drone sounds, while the class weights keep the two classes balanced in the loss. Each test fold contains all of its drone clips and an equal number of its non-drone clips, sampled once and shared by every model; test folds held 490 to 644 clips, and training pools 2,020 to 2,174 clips (B) or 4,040 to 4,348 clips (E). Every ESC-50 source clip of the test fold is also scored at clip level as the mean drone probability of its pieces. Because no piece of these clips was seen in training, the share of them classified as non-drone is an unseen-source specificity, which complements the balanced test folds with a much larger and more varied set of negatives.

## 2.2. Acoustic descriptors and feature selection

Clips are processed with librosa [35] at 16 kHz with a 2,048-sample FFT and a hop of 512 samples. The baseline set has 44 clip-level values: the mean and standard deviation of 13 MFCCs [36] (26 values); the means of spectral centroid, bandwidth, roll-off at 85% and zero-crossing rate (4); the means of 12 chroma bins (12); and the mean and standard deviation of a 128-band log-mel spectrogram (2). The extended set adds the means of seven spectral-contrast values [37], computed from 100 Hz so that the highest band stays below the 8 kHz Nyquist limit, and of six tonnetz dimensions [38], giving 57 features. The families describe different aspects of the same sound: MFCCs and the log-mel statistics summarise the spectral envelope, the spectral-shape measures locate the energy across frequency, and chroma, spectral contrast and tonnetz respond to tonal and harmonic structure such as the blade-passing harmonics of a rotor.

Feature selection combines two filters with different criteria. Let $X _ { j }$ be feature $j , c$ the class, S the set of features already selected and $I ( \cdot ; \cdot )$ mutual information. mRMR in its difference form adds at each step the feature that maximises

$$
J ( X _ { j } ) = I ( X _ { j } ; c ) - { \frac { 1 } { | S | } } \sum _ { X _ { s } \in S } I ( X _ { j } ; X _ { s } ) ,\tag{1}
$$

starting from the most relevant feature [15]. Mutual information is estimated with the nearest-neighbour estimators of scikitlearn [39] on standardised features. ReliefF updates the weight of each feature from the k nearest hits $H _ { i }$ and misses $M _ { i }$ of each sampled instance $R$

$$
W ( X _ { j } ) \xleftarrow { } W ( X _ { j } ) - \sum _ { i = 1 } ^ { k } \frac { \mathrm { d i f f } ( X _ { j } , R , H _ { i } ) } { m k } + \sum _ { i = 1 } ^ { k } \frac { \mathrm { d i f f } ( X _ { j } , R , M _ { i } ) } { m k } ,\tag{2}
$$

where dif is the range-normalised absolute difference and m the number of sampled instances [16, 17]. We use $k = 1 0 0$ on min–max-scaled features, as implemented in scikit-rebate [40]. Both estimates use a stratified sample of at most 3,000 training clips. mRMR favours features that are informative and not redundant with each other, whereas ReliefF rewards features that separate close examples of the two classes and can keep features that are useful only in combination. The hybrid subset is the union of the ten top-ranked features of each method, which gave 16 to 18 features per fold.

Filter selection fixes the subset size in advance. To examine how few descriptors are sufficient, NSGA-II [18] searches binary masks $z \in \{ 0 , 1 \} ^ { 4 4 }$ over the baseline descriptors and minimises

$$
\begin{array} { r } { f _ { 1 } ( z ) = 1 - \mathrm { A c c } _ { \mathrm { C V } } ( z ) , \qquad f _ { 2 } ( z ) = \sum _ { j } z _ { j } , } \end{array}\tag{3}
$$

where $\operatorname { A c c } _ { \operatorname { C V } }$ is the grouped three-fold cross-validation accuracy of a 100-tree Random Forest inside the training pool. The search uses a population of 40, 40 generations, two-point crossover and bit-flip mutation (pymoo [41]). The operating point was fixed before testing as the smallest subset on the final front whose inner accuracy lies within two percentage points of the best one. Because of its cost, NSGA-II was run on the five folds of the first repeat.

## 2.3. Pretrained embedding and stacking ensemble

YAMNet [22] frames the 16 kHz waveform into 0.96-s windows with a 0.48-s hop and returns a 1024-dimensional embedding per window; clips shorter than one window are zero-padded. The window embeddings are averaged into one vector per clip. YAMNet is used as a frozen feature extractor without fine-tuning, so it provides a second view of each clip that was learned from a large general-purpose audio corpus rather than designed for this task.

Baselines on the hybrid subset are logistic regression on standardised inputs, a Random Forest tuned by grouped grid search (100 or 200 trees, maximum depth unlimited, 10 or 20, minimum split size 2 or 5), XGBoost (300 trees, learning rate 0.05, depth 6) and a one-dimensional convolutional network (two convolutional layers with 32 and 64 filters, global average pooling and two dense layers, Adam, early stopping on 10% of the training data). The stacking ensemble uses four base learners, a Random Forest of 200 trees and XGBoost on the hand-crafted view and the same two learners on YAMNet embeddings. In configuration E the Random Forests use balanced class weights and XGBoost weights the positive class by the class ratio. Out-of-fold probabilities $p _ { 1 } , \ldots , p _ { 4 }$ from five-fold grouped cross-validation inside the training pool train a logistic-regression meta-learner,

$$
\hat { P } ( \mathrm { d r o n e } \mid x ) = \sigma \Bigl ( \beta _ { 0 } + \sum _ { m = 1 } ^ { 4 } \beta _ { m } p _ { m } ( x ) \Bigr ) ,\tag{4}
$$

after which the base learners are refitted on the whole training pool [28]. Because the meta-learner sees only probabilities for clips that its base learners did not train on, it weights each learner by how well it generalises rather than by how well it fits the training pool. A clip is labelled as drone when its probability is at least 0.5. The proposed model, fixed before any result was seen, is the two-view stack in configuration E; the hand-crafted view of this stack uses the hybrid subset selected from all 57 descriptors (17 to 18 features).

## 2.4. Performance metrics and statistical analysis

Drone is the positive class. With TP, TN, FP and FN the numbers of true positives, true negatives, false positives and false negatives, recall is TP/(TP+FN), specificity TN/(TN+FP), precision TP/(TP+FP), and the F1-score is the harmonic mean of precision and recall. We report these values, accuracy and the area under the ROC curve (AUC) as mean ± standard deviation over the 15 grouped test folds; because test folds are balanced, accuracy equals balanced accuracy. ESC-50 unseen-source specificity is pooled over the five folds of each repeat and summarised over the three repeats.

Fold accuracies are not independent, because the training pools of different folds overlap. The proposed stack is therefore compared with each alternative on identical clips by the exact McNemar test, which uses only the clips on which the two models disagree, applied to the predictions of each repeat with its folds pooled [31]. Holm correction controls the family-wise error rate over the five comparisons within each repeat [32], and per-fold paired differences with won, tied and lost counts are reported alongside. Pooled accuracies carry 95% bootstrap confidence intervals from 2,000 resamples [42]. Latency is measured on one CPU thread as the median over 100 clips and three repetitions.

## 2.5. Implementation

The pipeline is written in Python with librosa, scikit-learn, XGBoost, pymoo, scikit-rebate and TensorFlow [43]. Experiments ran on a workstation with two AMD EPYC 7513 processors. All 18 runs (15 grouped folds and 3 random splits) were produced by one version of the code, and every table and figure in Section 3 was generated directly from the saved per-fold predictions.

## 3. Results

This section first measures how the evaluation protocol changes the reported performance. All later experiments use the grouped protocol: single classifiers and feature selection (Section 3.2), two-view fusion and the training class ratio (Section 3.3), and the errors and computational cost of the proposed stack (Section 3.4).

## 3.1. Effect of the evaluation protocol

Table 2 and Figure 3 compare the two protocols for four representative models. The random split gave higher accuracy for every model, by 1.89 to 3.22 percentage points, because its test clips share recording chunks and ESC-50 sources with the training data. For the proposed stack, accuracy fell from 97.18% under the random split to 95.29% under the grouped protocol.

Table 2. Effect of the evaluation protocol.
<table><tr><td rowspan="2">Model</td><td colspan="3">Accuracy (%)</td><td rowspan="2">ESC-50 specificity (%)</td><td colspan="2"></td></tr><tr><td>Random</td><td>Grouped</td><td> $\Delta$ </td><td>Random</td><td>Grouped</td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td> $9 4 . 9 2 \pm 0 . 7 5$ </td><td> $9 1 . 7 1 \pm 3 . 3 0$ </td><td>3.22</td><td> $9 6 . 8 3 \pm 0 . 4 6$ </td><td></td><td> $9 5 . 6 8 \pm 0 . 0 6$ </td></tr><tr><td>XGBoost, hybrid-57</td><td> $9 6 . 8 7 \pm 0 . 3 9$ </td><td> $9 4 . 5 0 \pm 2 . 2 4$ </td><td>2.37</td><td> $9 9 . 5 3 \pm 0 . 0 3$ </td><td></td><td> $9 8 . 2 7 \pm 0 . 1 5$ </td></tr><tr><td>Stacking: hand-crafted only</td><td> $9 6 . 5 5 \pm 0 . 5 4$ </td><td> $9 4 . 1 8 \pm 2 . 1 0$ </td><td>2.37</td><td></td><td> $9 9 . 5 2 \pm 0 . 1 5$ </td><td> $9 8 . 4 5 \pm 0 . 1 8$ </td></tr><tr><td>Stacking: hand-crafted + YAMNet (proposed)</td><td> $9 7 . 1 8 \pm 0 . 5 0 $ </td><td> $9 5 . 2 9 \pm 1 . 6 2$ </td><td>1.89</td><td> $9 9 . 6 8 \pm 0 . 1 0$ </td><td></td><td> $9 8 . 7 8 \pm 0 . 1 0$ </td></tr></table>

Random: 80/20 split of 1-s pieces (3 repeats), so pieces of the same recording chunk and of the same ESC-50 clip fall on both sides. Grouped: source-disjoint 5-fold protocol of this work (15 folds). Mean ± SD; ∆ = random minus grouped mean accuracy in percentage points, from unrounded values. Under the random split the ESC-50 clips are scored by models trained on other pieces of the same clips

The ranking of the four models was the same under both protocols, so the random split did not reorder them; it moved all of them upward, by different amounts. The shift was largest for the Random Forest on the hybrid subset (3.22 points) and smallest for the proposed stack (1.89 points). Even the smallest shift is larger than every difference between the three strongest models under the grouped protocol, the largest of which is 1.11 points. The random split also reported standard deviations three to six times smaller than the grouped protocol, so it overstates both the level and the stability of performance.

![](images/8472bd65b816de2eb5dd90dc92a8fb494dc99d491a7f41d67de731e237ecf318.jpg)  
0 Random piece-level split (3 repeats)Grouped source-disjoint split (15 folds)  
Figure 3. Accuracy and ESC-50 specificity under the random and grouped protocols $( \mathsf { m e a n } \pm \mathsf { S D } )$

ESC-50 specificity followed the same pattern. Under the random split the three strongest models classified 99.52% to 99.68% of ESC-50 clips as non-drone, but these clips had contributed other pieces to training. When the clips were truly unseen, specificity fell to 98.78% for the proposed stack (Figure 3, right). The grouped values are the ones used in the rest of the paper.

## 3.2. Single classifiers and feature selection

Table 3 lists the single classifiers on the hybrid subset under the grouped protocol. XGBoost was the most accurate (94.10%), followed by the tuned Random Forest (91.71%), logistic regression (91.12%) and the one-dimensional CNN (88.96%). The

Random Forest had high specificity but the lowest recall of the four classifiers $( 8 8 . 5 7 \pm 7 . 0 5 \% )$ , with a large spread across folds. Logistic regression and the CNN showed the opposite profile, with recall above specificity (92.35% against 89.90% for logistic regression), and they rejected fewer unseen ESC-50 sources (91.12% and 89.88%) than the two tree ensembles (95.68% and 95.50%). On a few hand-crafted inputs, the tree ensembles were thus the more reliable single learners, and they are also the base learners of the stacking ensemble.

Table 3. Classifiers on the hybrid mRMR+ReliefF feature subset (configuration B, 1:1 training ratio).
<table><tr><td>Model</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Spec.</td><td>AUC</td><td>ESC-50 spec.</td></tr><tr><td>Logistic regression, hybrid</td><td> $9 1 . 1 2 \pm 2 . 1 3$ </td><td> $9 0 . 2 5 \pm 2 . 9 5$ </td><td> $9 2 . 3 5 \pm 3 . 4 8$ </td><td> $9 1 . 2 3 \pm 2 . 1 1$ </td><td> $8 9 . 9 0 \pm 3 . 6 1 $ </td><td> $0 . 9 6 5 \pm 0 . 0 1 6$ </td><td> $9 1 . 1 2 \pm 0 . 2 9$ </td></tr><tr><td>1-D CNN, hybrid</td><td> $8 8 . 9 6 \pm 3 . 1 9$ </td><td> $8 8 . 5 0 \pm 5 . 4 3$ </td><td> $9 0 . 1 3 \pm 5 . 1 4$ </td><td> $8 9 . 1 0 \pm 2 . 9 8$ </td><td> $8 7 . 7 9 \pm 6 . 6 9$ </td><td> $0 . 9 5 7 \pm 0 . 0 2 5$ </td><td> $8 9 . 8 8 \pm 1 . 8 3$ </td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td> $9 1 . 7 1 \pm 3 . 3 0$ </td><td> $9 4 . 5 7 \pm 1 . 9 6$ </td><td> $8 8 . 5 7 \pm 7 . 0 5$ </td><td> $9 1 . 3 1 \pm 3 . 8 6$ </td><td>94.85±2.07</td><td> $0 . 9 8 1 \pm 0 . 0 1 0$ </td><td> $9 5 . 6 8 \pm 0 . 0 6$ </td></tr><tr><td>XGBoost, hybrid</td><td> $9 4 . 1 0 \pm 1 . 7 8$ </td><td> $9 4 . 8 0 \pm 1 . 5 8$ </td><td>93.36± 3.84</td><td> $9 4 . 0 3 \pm 1 . 9 2$ </td><td>94.84±1.68</td><td> $0 . 9 8 5 \pm 0 . 0 0 9$ </td><td> $9 5 . 5 0 \pm 0 . 1 0$ </td></tr></table>

Values are percentages (AUC as a fraction), mean ± SD over 15 grouped test folds (3 repeats of 5 folds); test sets are balanced, so accuracy equals balanced accuracy. ESC-50 spec.: share of the 2,000 ESC-50 source clips scored as non-drone by models that never saw any of their pieces, pooled per repeat, mean ± SD over 3 repeats.

Table 4 isolates the selection step with the tuned Random Forest. The hybrid subset reached $9 1 . 7 1 \pm 3 . 3 0 \%$ accuracy, against $9 1 . 6 1 \pm 2 . 4 7 \%$ with all 44 descriptors, 91.33% with the mRMR top ten and 90.69% with the ReliefF top ten. The hybrid subset matched the full set within one standard deviation while using fewer than half of the descriptors. Each single filter fell slightly below the hybrid in accuracy and in ESC-50 specificity (94.42% and 94.88% against 95.68%), which supports combining the two criteria; the full set kept a slightly higher ESC-50 specificity (96.17%).

Table 4. Feature-selection ablation with a tuned Random Forest (configuration B).
<table><tr><td>Model</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Spec.</td><td>AUC</td><td>ESC-50 spec.</td></tr><tr><td>RF, all 44 baseline features</td><td> $9 1 . 6 1 \pm 2 . 4 7$ </td><td> $9 4 . 8 2 \pm 2 . 5 7$ </td><td> $8 8 . 1 2 \pm 5 . 0 6$ </td><td> $9 1 . 2 4 \pm 2 . 8 1$ </td><td> $9 5 . 0 9 \pm 2 . 6 3$ </td><td> $0 . 9 8 1 \pm 0 . 0 1 0$ </td><td> $9 6 . 1 7 \pm 0 . 3 9$ </td></tr><tr><td>RF, mRMR top-10</td><td> $9 1 . 3 3 \pm 2 . 5 4$ </td><td> $9 2 . 3 9 \pm 2 . 6 9$ </td><td> $9 0 . 1 5 \pm 4 . 5 5$ </td><td> $9 1 . 1 9 \pm 2 . 7 5$ </td><td> $9 2 . 5 0 \pm 2 . 9 2$ </td><td> $0 . 9 7 3 \pm 0 . 0 1 4$ </td><td> $9 4 . 4 2 \pm 0 . 4 6$ </td></tr><tr><td>RF, ReliefF top-10</td><td> $9 0 . 6 9 \pm 3 . 6 5$ </td><td> $9 3 . 4 2 \pm 2 . 1 0$ </td><td>87.60±7.51</td><td> $9 0 . 2 5 \pm 4 . 2 2$ </td><td> $9 3 . 7 8 \pm 2 . 2 0$ </td><td> $0 . 9 7 3 \pm 0 . 0 1 4$ </td><td> $9 4 . 8 8 \pm 0 . 2 3$ </td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td> $9 1 . 7 1 \pm 3 . 3 0$ </td><td> $9 4 . 5 7 \pm 1 . 9 6$ </td><td>88.57±7.05</td><td> $9 1 . 3 1 \pm 3 . 8 6$ </td><td> $9 4 . 8 5 \pm 2 . 0 7$ </td><td> $0 . 9 8 1 \pm 0 . 0 1 0$ </td><td> $9 5 . 6 8 \pm 0 . 0 6$ </td></tr></table>

Values are percentages (AUC as a fraction), mean ± SD over 15 grouped test folds (3 repeats of 5 folds); test sets are balanced, so accuracy equals balanced accuracy. ESC-50 spec.: share of the 2,000 ESC-50 source clips scored as non-drone by models that never saw any of their pieces, pooled per repeat, mean ± SD over 3 repeats.

Figure 4 shows how often each descriptor entered the hybrid subset of configuration E. The first and ninth MFCC means, spectral bandwidth, the spread of the log-mel spectrogram, one chroma bin and two spectral-contrast bands were selected in all 15 folds. MFCC statistics formed the largest group of selected descriptors, yet every descriptor family contributed at least one descriptor that was selected in most folds. The selection was stable across folds, and the information it kept is spread over several families rather than concentrated in the cepstral coefficients.

The NSGA-II search asks a different question: how few descriptors are enough? The Pareto fronts in Figure 5 rise steeply up to about five features and flatten beyond, and the fronts of the five folds lie close together. The pre-specified rule chose 4 to 6 features per fold (Table 5), mostly MFCC statistics. These compact subsets reached 86.54 to 91.54% test accuracy (mean 88.14%), against 87.38 to 94.62% (mean 91.37%) for the hybrid subset on the same folds. NSGA-II thus gives up about three percentage points of accuracy for a descriptor set roughly one third the size of the hybrid subset.

![](images/d37c26419c4e1fe5cddcef5c1e2d35ead38e0a9aeb772b0425039f46a37665f8.jpg)  
Folds in which the descriptor was selected (of 15)  
Figure 4. Number of grouped test folds (of 15) in which each descriptor entered the hybrid mRMR+ReliefF subset of configuration E.

Table 5. NSGA-II feature selection on the 44 baseline features (population 40, 40 generations; the five grouped folds of the first repeat).
<table><tr><td>Fold</td><td>Front size</td><td>Chosen k</td><td>Inner-CV acc.</td><td>Test acc.</td><td>Test AUC</td><td>Hybrid k</td><td>Hybrid test acc.</td></tr><tr><td>1</td><td>10</td><td>6</td><td>92.07</td><td>88.46</td><td>0.953</td><td>17</td><td>92.69</td></tr><tr><td>2</td><td>12</td><td>5</td><td>91.19</td><td>91.54</td><td>0.972</td><td>16</td><td>94.62</td></tr><tr><td>3</td><td>10</td><td>6</td><td>92.44</td><td>86.97</td><td>0.959</td><td>16</td><td>92.15</td></tr><tr><td>4</td><td>10</td><td>5</td><td>91.04</td><td>87.20</td><td>0.951</td><td>18</td><td>90.00</td></tr><tr><td>5</td><td>10</td><td>4</td><td>92.68</td><td>86.54</td><td>0.968</td><td>17</td><td>87.38</td></tr></table>

The chosen subset is the smallest one within 2 percentage points of the best inner-CV accuracy, fixed before testing. Random Forest classifier; accuracies in percent; hybrid: mRMR+ReliefF subset of the same fold.

![](images/a9c839624034489bfcbbb1dae3724beefc35ed808dc7dbdba76d9720830480c5.jpg)  
Figure 5. NSGA-II Pareto fronts (inner cross-validation accuracy against subset size) for the five grouped folds of the first repeat; circles mark the pre-specified operating points.

## 3.3. Two-view fusion and training class ratio

Table 6 and Figure 6 report the fusion ablation in configuration E. Single learners on YAMNet embeddings (91.98% and 92.38%) were less accurate than XGBoost on the hand-crafted subset (94.50%). Stacking within one view gave 94.18% (hand-crafted) and 92.38% (YAMNet), and the two-view stack reached 95 $. 2 9 \pm 1 . 6 2 \%$ with 98.29% specificity and an AUC of 0.990.

Table 6. Fusion ablation (configuration E, 1:3 training ratio with class weights): single learners, single-view stacks and the two-view stack.
<table><tr><td>Model</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Spec.</td><td>AUC</td><td>ESC-50 spec.</td></tr><tr><td>RF, hybrid-57</td><td> $9 1 . 8 7 \pm 3 . 0 5$ </td><td> $9 7 . 6 1 \pm 1 . 0 2$ </td><td> $8 5 . 8 6 \pm 6 . 3 1$ </td><td> $9 1 . 2 4 \pm 3 . 6 1$ </td><td> $9 7 . 8 8 \pm 0 . 9 5$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 9$ </td><td> $9 8 . 8 0 \pm 0 . 1 3$ </td></tr><tr><td>XGBoost, hybrid-57</td><td> $9 4 . 5 0 \pm 2 . 2 4$ </td><td> $9 7 . 5 1 \pm 0 . 8 9$ </td><td> $9 1 . 3 3 \pm 4 . 4 6$ </td><td> $9 4 . 2 6 \pm 2 . 5 0$ </td><td> $9 7 . 6 6 \pm 0 . 8 7$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 9$ </td><td> $9 8 . 2 7 \pm 0 . 1 5$ </td></tr><tr><td>RF, YAMNet</td><td> $9 1 . 9 8 \pm 1 . 4 6$ </td><td> $9 6 . 7 3 \pm 1 . 1 5$ </td><td> $8 6 . 9 0 \pm 2 . 7 7$ </td><td> $9 1 . 5 3 \pm 1 . 6 4$ </td><td> $9 7 . 0 5 \pm 1 . 0 6$ </td><td> $0 . 9 7 8 \pm 0 . 0 1 0$ </td><td> $9 8 . 2 5 \pm 0 . 0 9$ </td></tr><tr><td>XGBoost, YAMNet</td><td> $9 2 . 3 8 \pm 1 . 4 4$ </td><td> $9 6 . 9 0 \pm 0 . 9 9$ </td><td> $8 7 . 5 7 \pm 2 . 8 2$ </td><td> $9 1 . 9 7 \pm 1 . 6 1$ </td><td> $9 7 . 1 9 \pm 0 . 9 2$ </td><td> $0 . 9 8 0 \pm 0 . 0 1 0$ </td><td> $9 8 . 5 8 \pm 0 . 1 0$ </td></tr><tr><td>Stacking: hand-crafted only</td><td> $9 4 . 1 8 \pm 2 . 1 0$ </td><td> $9 7 . 6 8 \pm 0 . 9 5$ </td><td> $9 0 . 5 2 \pm 4 . 3 5$ </td><td> $9 3 . 9 1 \pm 2 . 3 9$ </td><td> $9 7 . 8 4 \pm 0 . 9 4$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 8$ </td><td> $9 8 . 4 5 \pm 0 . 1 8$ </td></tr><tr><td>Stacking: YAMNet only</td><td> $9 2 . 3 8 \pm 1 . 6 0$ </td><td> $9 7 . 1 9 \pm 1 . 0 3$ </td><td>87.29±3.05 91.95±1.79 97.47± 0.94</td><td></td><td></td><td> $0 . 9 7 9 \pm 0 . 0 1 0$ </td><td> $9 8 . 5 0 \pm 0 . 2 2$ </td></tr><tr><td>Stacking: hand-crafted + YAMNet (proposed)</td><td> $9 5 . 2 9 \pm 1 . 6 2$ </td><td></td><td>98.19± 0.92 92.30±3.14 95.12±1.76 98.29±0.89</td><td></td><td></td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $9 8 . 7 8 \pm 0 . 1 0$ </td></tr></table>

Values are percentages (AUC as a fraction), mean ± SD over 15 grouped test folds (3 repeats of 5 folds); test sets are balanced, so accuracy equals balanced accuracy. ESC-50 spec.: share of the 2,000 ESC-50 source clips scored as non-drone by models that never saw any of their pieces, pooled per repeat, mean ± SD over 3 repeats.

Most of the gain came from recall. The two-view stack detected 92.30% of drone clips, against 90.52% for the hand-crafted stack and 87.29% for the YAMNet stack, while its specificity changed little (98.29% against 97.84%). Its fold-to-fold spread was also smaller than that of the hand-crafted stack (standard deviation 1.62 against 2.10 percentage points), as the per-fold dots in Figure 6 show.

![](images/9615bf8e318caac93acc3a2e5496893818e1ca4ba4c71f3d734c9fda0a305e21.jpg)

![](images/96d1b47294875ae2447eb6d65580d83c4b7be2c1b52bd44bcf815b92e6c75cd2.jpg)  
Figure 6. Fusion ablation in configuration E: accuracy and AUC per grouped test fold (dots) with mean ± SD (markers).

The meta-learner gave similar weights to the hand-crafted and YAMNet Random Forests (3.39 and 3.29) and smaller weights to the two XGBoost learners (2.70 and 2.06), so both views contribute to the decision (Figure 7). The YAMNet Random Forest received almost the same weight as its hand-crafted counterpart although it was the weaker learner on its own, which indicates that its errors overlap only partly with those of the hand-crafted view.

![](images/879e21b1d445ba5af8efdb98752380699d11203649cc322e99ba38e6ace6cd4c.jpg)  
Figure 7. Logistic-regression meta-learner weights of the proposed stack (mean ± SD over 15 folds).

Tables 7 and 8 give the paired comparisons. Against the hand-crafted stack, the two-view stack gained 1.11 percentage points on average and won 12 folds, tied 3 and lost 0; the McNemar test remained significant after Holm correction in two of the three repeats (largest adjusted $p = 0 . 0 7 4 )$ . Against XGBoost on the same hand-crafted subset the gain was 0.80 percentage points (11 won, 1 tied, 3 lost; significant in two of the three repeats). Against the YAMNet-only stack (2.91 percentage points) and the Random Forest baseline (3.59 percentage points) the differences were significant in every repeat. The gain from fusion is therefore small but consistent across folds.

Table 7. Per-fold paired accuracy differences between the proposed stack and each alternative over the 15 grouped test folds (computed from unrounded per-fold accuracies).
<table><tr><td>Alternative</td><td>△ acc. (pp), mean ± SD Folds won / tied / lost</td><td></td></tr><tr><td>Stacking: hand-crafted only</td><td> $1 . 1 1 \pm 1 . 0 8$ </td><td>12/3/0</td></tr><tr><td>XGBoost, hybrid-57</td><td> $0 . 8 0 \pm 1 . 0 8$ </td><td>11/1/3</td></tr><tr><td>Stacking: YAMNet only</td><td> $2 . 9 1 \pm 1 . 7 4$ </td><td>14/0/1</td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td> $3 . 5 9 \pm 3 . 5 3$ </td><td>13/0/2</td></tr><tr><td>XGBoost, hybrid</td><td> $1 . 1 9 \pm 1 . 5 5$ </td><td>12/0/3</td></tr></table>

Table 8. Exact McNemar tests of the proposed stack against each alternative on identical test clips, with the five folds of each repeat pooled (2,664 clips per repeat). b: clips classified correctly only by the proposed stack; c: clips classified correctly only by the alternative; p: exact two-sided p-value; Holm: p-value after Holm correction over the five comparisons within the repeat.
<table><tr><td colspan="3">Repeat 1</td><td colspan="3">Repeat 2</td><td colspan="3">Repeat 3</td></tr><tr><td>Alternative</td><td>b/c</td><td>p</td><td>Holm</td><td>b/c</td><td>p</td><td>Holm</td><td>b/c p</td><td>Holm</td></tr><tr><td>Stacking: hand-crafted only</td><td>55/33</td><td>0.025</td><td>0.074</td><td>55/27</td><td>0.003</td><td>0.008 57/ 19</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>XGBoost, hybrid-57</td><td>53 / 40</td><td>0.213</td><td>0.213</td><td>52/28 0.010</td><td>0.012</td><td>48/25</td><td>0.010</td><td>0.010</td></tr><tr><td>Stacking: YAMNet only</td><td>98/23</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>114/32</td><td>&lt; 0.001 &lt; 0.001</td><td>108/34</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td>151/45</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>143/ 47</td><td>&lt; 0.001</td><td>&lt; 0.001 145/39</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>XGBoost, hybrid</td><td>76/55</td><td>0.080</td><td>0.160</td><td>95/60</td><td>0.006</td><td>0.012 95/48</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr></table>

Table 9 gives the pooled accuracy of each repeat with its 95% bootstrap confidence interval. The intervals of the two-view stack and the hand-crafted stack overlap in every repeat, which is consistent with a small effect that the paired tests detect in two of the three repeats. The paired tests are the more sensitive comparison here, because they use the fact that both models classify the same clips.

Table 9. Pooled test accuracy (%) of each repeat with its 95% bootstrap confidence interval (2,000 resamples; 2,664 clips per repeat, five folds pooled).
<table><tr><td>Model</td><td>Repeat 1</td><td>Repeat 2</td><td>Repeat 3</td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td>91.25 [90.17, 92.34]</td><td>91.55 [90.43, 92.61]</td><td>91.63 [90.61, 92.72]</td></tr><tr><td>XGBoost, hybrid-57</td><td>94.74 [93.88, 95.61]</td><td>94.26 [93.36, 95.16]</td><td>94.74 [93.88, 95.61]</td></tr><tr><td>Stacking: hand-crafted only</td><td>94.41 [93.54, 95.27]</td><td>94.11 [93.21, 95.01]</td><td>94.18 [93.28, 95.08]</td></tr><tr><td>Stacking: hand-crafted + YAMNet (proposed)</td><td>95.23 [94.41, 96.02]</td><td>95.16 [94.33, 95.98]</td><td>95.61 [94.78, 96.40]</td></tr></table>

Table 10 compares the two-view stack trained on the balanced pool (configuration B) with the class-weighted 1:3 pool (configuration E). The balanced version reached a higher accuracy (95.97%) and recall (95.48%), while configuration E gave higher specificity (98.29% against 96.46%) and fewer false positives on unseen ESC-50 clips (98.78% against 96.98%). The training ratio thus sets the balance between missed drones and false alarms at the fixed 0.5 threshold, and its effect on specificity is larger than the effect of fusion itself.

Table 10. Effect of the training class ratio on the two-view stack: configuration B (1:1, no class weights) versus configuration E (1:3, class-weighted).
<table><tr><td>Model</td><td>Acc.</td><td>Prec. Rec.</td><td>F1</td><td>Spec.</td><td>AUC</td><td>ESC-50 spec.</td></tr><tr><td>Stacking: hybrid-44 + YAMNet (1:1 data)</td><td> $9 5 . 5 2 \pm 1 . 0 6$ </td><td> $9 5 . 8 3 \pm 1 . 7 7$ </td><td> $9 5 . 2 3 \pm 2 . 1 0$   $9 5 . 5 0 \pm 1 . 0 8$ </td><td> $9 5 . 8 1 \pm 1 . 8 3$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 8$ </td><td> $9 6 . 7 3 \pm 0 . 3 4$ </td></tr><tr><td>Stacking: hybrid-57 + YAMNet (1:1 data)</td><td> $9 5 . 9 7 \pm 1 . 7 0$ </td><td> $9 6 . 4 5 \pm 1 . 7 7$ </td><td> $9 5 . 4 8 \pm 2 . 7 9$   $9 5 . 9 4 \pm 1 . 7 5$ </td><td> $9 6 . 4 6 \pm 1 . 8 1$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $9 6 . 9 8 \pm 0 . 3 5$ </td></tr><tr><td>Stacking: hand-crafted only</td><td> $9 4 . 1 8 \pm 2 . 1 0$ </td><td> $9 7 . 6 8 \pm 0 . 9 5$ </td><td> $9 0 . 5 2 \pm 4 . 3 5$   $9 3 . 9 1 \pm 2 . 3 9$ </td><td> $9 7 . 8 4 \pm 0 . 9 4$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 8$ </td><td> $9 8 . 4 5 \pm 0 . 1 8$ </td></tr><tr><td>Stacking: hand-crafted + YAMNet (proposed)</td><td> $9 5 . 2 9 \pm 1 . 6 2$ </td><td> $9 8 . 1 9 \pm 0 . 9 2$ </td><td> $9 2 . 3 0 \pm 3 . 1 4$   $9 5 . 1 2 \pm 1 . 7 6$ </td><td> $9 8 . 2 9 \pm 0 . 8 9$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 9$ </td><td> $9 8 . 7 8 \pm 0 . 1 0$ </td></tr></table>

Values are percentages (AUC as a fraction), mean ± SD over 15 grouped test folds (3 repeats of 5 folds); test sets are balanced, so accuracy equals balanced accuracy. ESC-50 spec.: share of the 2,000 ESC-50 source clips scored as non-drone by models that never saw any of their pieces, pooled per repeat, mean ± SD over 3 repeats.

## 3.4. Error analysis and computational cost

Pooled over the five folds of the first repeat (Figure 8), the proposed stack produced 28 false positives and 99 missed drones among 2,664 test clips, fewer of both kinds than the Random Forest baseline. Missed drones outnumber false alarms, in line with the high specificity of configuration E. The ROC curve of the proposed stack lies at or above those of the other models at low false-positive rates (Figure 9), the region that matters when false alarms are costly.

![](images/35d4ade4d856d2dd10e211ac9d98e4c28a9ee61514239e1f1ec78b8fdd607743.jpg)

(b) Proposed stack  
![](images/de68d1f8fd4328c3726d51a2d4b75f6e631350d8b52d82bd158590c7d56addd9.jpg)  
Figure 8. Confusion matrices of the Random Forest baseline and the proposed stack, grouped protocol, first repeat (five folds pooled, 2,664 test clips).

On unseen ESC-50 clips (Table 11 and Figure 10), the most frequent confusions were with steady broadband or motorised sounds: washing machine (20.00%), rain (9.17%), vacuum cleaner (5.00%), and chainsaw and train (4.17% each). Airplane and helicopter sounds were each mistaken for drones in 3.33% of clips. By major sound group (Table 12), no animal or human non-speech clip was labelled as a drone, and the interior/domestic group had the highest false-positive rate (2.58%). The errors are concentrated in a few classes with sustained mechanical or noise-like sound rather than spread over the whole collection.

1-D CNN, hybrid (AUC 0.956)

![](images/98a59cd445b9fe866f334b5190034a25960b90d5cdcf99f543213e053bbc7fb6.jpg)  
Logistic regression, hybrid (AUC 0.961)  
XGBoost, hybrid-57 (AUC 0.989)  
Stacking: YAMNet only (AUC 0.975)  
Stacking: hand-crafted + YAMNet (proposed) (AUC 0.989)  
Figure 9. ROC curves of representative models, grouped protocol, first repeat (five folds pooled).

Table 11. ESC-50 classes most often scored as drone by the proposed stack.
<table><tr><td>ESC-50 class</td><td>Major group False-positive rate</td><td> $( \% ) , { \bf m e a n } \pm \mathsf { s D }$ </td></tr><tr><td>washing machine</td><td>Interior/domestic</td><td> $2 0 . 0 0 \pm 2 . 5 0$ </td></tr><tr><td>rain</td><td>Natural soundscapes</td><td> $9 . 1 7 \pm 3 . 8 2$ </td></tr><tr><td>vacuum cleaner</td><td>Interior/domestic</td><td> $5 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>chainsaw</td><td>Exterior/urban</td><td> $4 . 1 7 \pm 1 . 4 4$ </td></tr><tr><td>train</td><td>Exterior/urban</td><td> $4 . 1 7 \pm 1 . 4 4$ </td></tr><tr><td>airplane</td><td>Exterior/urban</td><td> $3 . 3 3 \pm 1 . 4 4$ </td></tr><tr><td>crackling fire</td><td>Natural soundscapes</td><td> $3 . 3 3 \pm 1 . 4 4$ </td></tr><tr><td>helicopter</td><td>Exterior/urban</td><td> $3 . 3 3 \pm 1 . 4 4$ </td></tr><tr><td>car horn</td><td>Exterior/urban</td><td> $2 . 5 0 \pm 0 . 0 0$ </td></tr><tr><td>engine</td><td>Exterior/urban</td><td> $2 . 5 0 \pm 2 . 5 0$ </td></tr></table>

Each class has 40 source clips; each clip is scored once per repeat by a model that never saw any of its pieces; mean ± SD over 3 repeats.

Table 12. ESC-50 false-positive rate of the proposed stack by major sound group (400 source clips per group, mean over 3 repeats).
<table><tr><td rowspan=1 colspan=1>Major group                                                                                               False-positive rate (%)</td></tr><tr><td rowspan=1 colspan=1>Animals                                                                                                               0.00</td></tr><tr><td rowspan=1 colspan=1>Exterior/urban                                                                                                        2.00</td></tr><tr><td rowspan=1 colspan=1>Human non-speech                                                                                                  0.00</td></tr><tr><td rowspan=1 colspan=1>Interior/domestic                                                                                                     2.58</td></tr><tr><td rowspan=1 colspan=1>Natural soundscapes                                                                                                1.50</td></tr></table>

![](images/5917bd7ed7d6089f8e8a7370d176991ec900c484384c85cae7f795ffea2082e9.jpg)  
Figure 10. ESC-50 classes most often scored as drone by the proposed stack (each clip scored by a model that never saw it; mean over 3 repeats).

Table 13 and Figure 11 report single-thread CPU cost. Extracting the 57 descriptors took 14.4 ms per clip and YAMNet 7.7 ms. XGBoost on the hand-crafted subset needed 14.7 ms per clip in total with a 0.69 MB model, the hand-crafted stack 28.8 ms, and the two-view stack 51.0 ms with 12.6 MB of classifiers. For the single XGBoost model, feature extraction accounts for almost all of the time; for the two-view stack, the four base learners and the meta-learner take more than half. All pipelines process a one-second clip well within one second on a single CPU thread, which leaves a wide margin for continuous operation on the processor used here.

Table 13. Single-thread CPU cost per 1-s clip on an AMD EPYC 7513 32-Core Processor (median of 100 clips × 3 repeats).
<table><tr><td>Pipeline</td><td>Hand-crafted features YAMNet Features (ms)</td><td></td><td></td><td>Classifier (ms)</td><td>Total (ms)</td><td>Size (MB)</td><td>Accuracy (%)</td></tr><tr><td>RF, hybrid mRMR+ReliefF</td><td>17</td><td>no</td><td>14.43</td><td>7.06</td><td>21.49</td><td>1.68</td><td>91.71 ±3.30</td></tr><tr><td>RF, NSGA-II subset</td><td>6</td><td>no</td><td>14.43</td><td>7.02</td><td>21.45</td><td>2.15</td><td>88.14 ± 2.03ª</td></tr><tr><td>XGBoost, hybrid-57</td><td>18</td><td>no</td><td>14.43</td><td>0.24</td><td>14.67</td><td>0.69</td><td>94.50 ± 2.24</td></tr><tr><td>Stacking: hand-crafted only</td><td>18</td><td>no</td><td>14.43</td><td>14.32</td><td>28.75</td><td>5.44</td><td>94.18</td></tr><tr><td>Stacking: hand-crafted + YAMNet (proposed)</td><td>18</td><td>yes</td><td>22.14</td><td>28.86</td><td>51.00</td><td>12.60</td><td>±2.10 95.29 ± 1.62</td></tr></table>

Feature time covers the full 57-feature front end, plus YAMNet for the two-view stack; model size is the pickled classifier. Accuracy: grouped protocol, mean ± SD over 15 folds; <sup>a</sup> five folds of the first repeat.

![](images/30c93b5bed7cb7078d5118fe8b663e50858e21033435c726638454b29f63efe9.jpg)  
Figure 11. Accuracy against total single-thread CPU time per clip. The NSGA-II point averages the five folds of the first repeat; the others average 15 folds.

## 4. Discussion

The protocol comparison shows that a random split of the DroneAudioDataset rewards memorising recordings and source sounds. The inflation of 1.89 to 3.22 percentage points is of the same order as the differences between the models compared here, so accuracies obtained with random splits on this dataset cannot be compared with source-disjoint results or taken as evidence of generalisation. The grouped protocol also exposes much larger fold-to-fold variation, which a single split hides. These findings are consistent with Shulman’s grouped evaluation of the same dataset [26]. Because most non-drone clips derive from ESC-50, ESC-50 is also unsuitable as an external test set for models trained on this dataset unless its source clips are excluded from training. For future benchmarks, the recording and source identifiers already present in the file names make a grouped split easy to apply, and reporting the grouping, the number of folds and the fold assignment would let results from different studies be compared directly.

Hybrid mRMR+ReliefF selection kept the accuracy of the full descriptor set with fewer than half of the features, and the descriptors chosen in every fold (low-order MFCC means, spectral bandwidth, spectral contrast and the spread of the log-me spectrogram) are consistent with the broadband, harmonic character of rotor noise. Because the selected descriptors came from every family, a detector built on MFCCs alone would likely discard useful information. NSGA-II offers a different operating point: four to six descriptors at a few percentage points lower accuracy, which suits settings where feature computation or memory is tightly limited.

Fusion with YAMNet embeddings gave a small, consistent improvement over the best hand-crafted models. The YAMNet view alone was weaker than the hand-crafted view, yet the meta-learner relied on it almost as much as on the hand-crafted Random Forest, which suggests that the two views make partly different errors. The improvement appeared mainly as higher recall, so the embedding mainly recovers drone clips that the hand-crafted view misses. The gain over XGBoost on the hand-crafted subset was below one standard deviation of the fold accuracies and was significant in two of the three repeats, so it should be read as a modest benefit. The class-ratio comparison shows that the training pool, more than the fusion itself, controls the balance between recall and specificity; applications that must avoid false alarms favour the class-weighted configuration, while applications that must not miss a drone favour the balanced one or a lower threshold.

The ESC-50 error analysis indicates where acoustic detectors trained on this dataset are fragile: steady machine and broadband noises such as washing machines, vacuum cleaners and rain cause most false alarms, while aircraft sounds are rarely confused. This information is useful for choosing negative training material and for planning field tests. A detector intended for a particular site would benefit from negative recordings of the machines and weather typical of that site, and a field evaluation should include them on purpose rather than rely on random background sound.

For a fixed acoustic sensor, the results describe a set of operating points rather than a single best model. The single XGBoost model on the hand-crafted subset is the cheapest option with competitive accuracy, the two-view stack adds 0.80 percentage points of accuracy and higher recall at more than three times the CPU time, and an NSGA-II subset reduces the descriptor set further for the most constrained devices. All of these choices remain fast enough for real-time processing of one-second clips on a single core of the processor used in this study.

## 5. Conclusions

Under a grouped, source-disjoint protocol for the DroneAudioDataset binary task, a stacking ensemble of hand-crafted descriptors and pretrained YAMNet embeddings reached 95.29% accuracy and 98.29% specificity, and processed a clip in 51.0 ms on one CPU thread. Random clip-level splitting overstated accuracy by 1.89 to 3.22 percentage points, a margin comparable to the differences between models, so source-disjoint evaluation should be the default for this benchmark.

Hybrid mRMR+ReliefF selection matched the full descriptor set with fewer features, NSGA-II provided compact subsets for constrained settings, and fusion with pretrained embeddings added a small but consistent gain, mainly through higher recall. The unseen-source analysis showed that the remaining false alarms come from a few steady mechanical and broadband sounds, which identifies the negative material that future training sets should contain. Future work will test the models on independently recorded outdoor data with more drone types, evaluate further pretrained and self-supervised audio encoders, and study federated training of the detector across distributed sensor nodes [44], where differences in data and model capacity between nodes can be handled by confidence-weighted knowledge distillation [45, 46].

## References

1. Djurek I, Petosic A, Grubesa S, Suhanek M. Analysis of a quadcopter’s acoustic signature in different flight regimes. IEEE Access. 2020;8:10662–10670. doi: 10.1109/ACCESS.2020.2965177

2. Dumitrescu C, Minea M, Costea IM, Cosmin Chiva I, Semenescu A. Development of an acoustic system for UAV detection. Sensors. 2020;20(17):4870. doi: 10.3390/s20174870

3. Al-Emadi S, Al-Ali A, Mohammad A, Al-Ali A. Audio based drone detection and identification using deep learning. In: 2019 15th International Wireless Communications & Mobile Computing Conference (IWCMC); 2019 Jun 24–28; Tangier, Morocco. IEEE; 2019. p. 459–464. doi: 10.1109/IWCMC.2019.8766732

4. Al-Emadi S, Al-Ali A, Al-Ali A. Audio-based drone detection and identification using deep learning techniques with dataset enhancement through generative adversarial networks. Sensors. 2021;21(15):4953. doi: 10.3390/s21154953

5. Casabianca P, Zhang Y. Acoustic-based UAV detection using late fusion of deep neural networks. Drones. 2021;5(3):54. doi: 10.3390/drones5030054

6. Cabrera-Ponce AA, Martinez-Carranza J, Rascon C. Detection of nearby UAVs using a multi-microphone array on board a UAV. Int J Micro Air Veh. 2020;12:1756829320925748. doi: 10.1177/1756829320925748

7. Tian Z, Yu D. A study of real-time recognition of unmanned aerial vehicles in outdoor areas based on a random forest algorithm. Aviation. 2022;26(4):169–175. doi: 10.3846/aviation.2022.17910

8. Wang Y, Fagiani F, Ho K, Matson E. A feature engineering focused system for acoustic UAV payload detection. In: Proceedings of the 14th International Conference on Agents and Artificial Intelligence (ICAART 2022), Volume 3. SciTePress; 2022. p. 470–475. doi: 10.5220/0010843800003116

9. Tejera-Berengue D, Zhu-Zhou F, Utrilla-Manso M, Gil-Pita R, Rosa-Zurera M. Analysis of distance and environmenta impact on UAV acoustic detection. Electronics. 2024;13(3):643. doi: 10.3390/electronics13030643

10. Iqbal Z, Ilyas R, Shahzad W, Inayat I. A comparative study of machine learning techniques used in non-clinical systems for continuous healthcare of independent livings. In: 2018 IEEE Symposium on Computer Applications and Industria Electronics (ISCAIE). IEEE; 2018. p. 406–411. doi: 10.1109/ISCAIE.2018.8405507

11. Ehsan A, Iqbal Z, Abuowaida S, Aljaidi M, Zia HU, Alshdaifat N, et al. Enhanced anomaly detection in Ethereum: unveiling and classifying threats with machine learning. IEEE Access. 2024;12:176440–176456. doi: 10.1109/AC-CESS.2024.3504300

12. Batool U, Iqbal Z, et.al. Ensemble attention deep network for imbalanced silicon wafer defect inspection. PeerJ Comput Sci. 2026; Vol. e4025. doi: 10.7717/peerj-cs.4025

13. Alam SS, Chakma A, Rahman MH, Mofidul RB, Alam MM, Utama IBKY, et al. RF-enabled deep-learning-assisted drone detection and identification: an end-to-end approach. Sensors. 2023;23(9):4202. doi: 10.3390/s23094202

14. Frid A, Ben-Shimol Y, Manor E, Greenberg S. Drones detection using a fusion of RF and acoustic features and deep neura networks. Sensors. 2024;24(8):2427. doi: 10.3390/s24082427

15. Peng H, Long F, Ding C. Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE Trans Pattern Anal Mach Intell. 2005;27(8):1226–1238. doi: 10.1109/TPAMI.2005.159

16. Kononenko I. Estimating attributes: analysis and extensions of RELIEF. In: Bergadano F, De Raedt L, editors. Machine Learning: ECML-94. Lecture Notes in Computer Science. Berlin, Germany: Springer; 1994. p. 171–182. doi: 10.1007/3- 540-57868-4\_57

17. Robnik-Šikonja M, Kononenko I. Theoretical and empirical analysis of ReliefF and RReliefF. Mach Learn. 2003;53(1-2):23– 69. doi: 10.1023/A:1025667309714

18. Deb K, Pratap A, Agarwal S, Meyarivan T. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Trans Evol Comput. 2002;6(2):182–197. doi: 10.1109/4235.996017

19. Kundu R, Mallipeddi R. HFMOEA: a hybrid framework for multi-objective feature selection. J Comput Des Eng. 2022;9(3):949–965. doi: 10.1093/jcde/qwac040

20. Iqbal Z, Shahzad W, Faiza M. A diverse clustering particle swarm optimizer for dynamic environment: to locate and track multiple optima. In: 2015 10th IEEE Conference on Industrial Electronics and Applications (ICIEA). IEEE; 2015. p. 1755– 1760. doi: 10.1109/ICIEA.2015.7334395

21. Gemmeke JF, Ellis DPW, Freedman D, Jansen A, Lawrence W, Moore RC, et al. Audio Set: an ontology and human-labeled dataset for audio events. In: 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE; 2017. p. 776–780. doi: 10.1109/ICASSP.2017.7952261

22. Google. YAMNet [model and code]. TensorFlow Model Garden, GitHub; 2026. Available from: https://github.com/tensorflo w/models/tree/master/research/audioset/yamnet (accessed 27 Oct 2026)

23. Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, et al. MobileNets: efficient convolutional neural networks for mobile vision applications. arXiv preprint. 2017. doi: 10.48550/arXiv.1704.04861

24. Hershey S, Chaudhuri S, Ellis DPW, Gemmeke JF, Jansen A, Moore RC, et al. CNN architectures for large-scale audio classification. In: 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE; 2017. p. 131–135. doi: 10.1109/ICASSP.2017.7952132

25. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804. doi: 10.1016/j.patter.2023.100804

26. Shulman D. EchoHawk: a reproducible acoustic pipeline for drone detection, classification, and direction-finding, with a cautionary study of session-level data leakage. arXiv preprint. 2026. doi: 10.48550/arXiv.2606.29589

27. Piczak KJ. ESC: dataset for environmental sound classification. In: Proceedings of the 23rd ACM International Conference on Multimedia (MM ’15); Brisbane, Australia. ACM; 2015. p. 1015–1018. doi: 10.1145/2733373.2806390

28. Wolpert DH. Stacked generalization. Neural Netw. 1992;5(2):241–259. doi: 10.1016/S0893-6080(05)80023-1

29. Breiman L. Random forests. Mach Learn. 2001;45(1):5–32. doi: 10.1023/A:1010933404324

30. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’16); San Francisco, USA. ACM; 2016. p. 785–794. doi: 10.1145/2939672.2939785

31. Dietterich TG. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Comput. 1998;10(7):1895–1923. doi: 10.1162/089976698300017197

32. Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65–70.

33. Al-Emadi S. DroneAudioDataset [dataset and licence]. GitHub; 2019–2026. Available from: https://github.com/saraalemad /DroneAudioDataset (commit ec6139b3, accessed 27 Oct 2026)

34. Warden P. Speech Commands: a dataset for limited-vocabulary speech recognition. arXiv preprint. 2018. doi: 10.48550/arXiv.1804.03209

35. McFee B, Raffel C, Liang D, Ellis DPW, McVicar M, Battenberg E, et al. librosa: audio and music signal analysis in Python. In: Proceedings of the 14th Python in Science Conference (SciPy 2015); 2015. p. 18–24. doi: 10.25080/Majora-7b98e3ed 003

36. Davis SB, Mermelstein P. Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Trans Acoust Speech Signal Process. 1980;28(4):357–366. doi: 10.1109/TASSP.1980.1163420

37. Jiang DN, Lu L, Zhang HJ, Tao JH, Cai LH. Music type classification by spectral contrast feature. In: Proceedings of the IEEE International Conference on Multimedia and Expo (ICME ’02). IEEE; 2002. p. 113–116. doi: 10.1109/ICME.2002.1035731

38. Harte C, Sandler M, Gasser M. Detecting harmonic change in musical audio. In: Proceedings of the 1st ACM Workshop on Audio and Music Computing Multimedia (AMCMM ’06). ACM; 2006. p. 21–26. doi: 10.1145/1178723.1178727

39. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–2830.

40. Urbanowicz RJ, Olson RS, Schmitt P, Meeker M, Moore JH. Benchmarking relief-based feature selection methods for bioinformatics data mining. J Biomed Inform. 2018;85:168–188. doi: 10.1016/j.jbi.2018.07.015

41. Blank J, Deb K. pymoo: multi-objective optimization in Python. IEEE Access. 2020;8:89497–89509. doi: 10.1109/AC-CESS.2020.2990567

42. Efron B, Tibshirani RJ. An introduction to the bootstrap. New York, USA: Chapman & Hall; 1993. (Monographs on Statistics and Applied Probability 57)

43. Abadi M, Barham P, Chen J, Chen Z, Davis A, Dean J, et al. TensorFlow: a system for large-scale machine learning. In: 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’16); Savannah, USA. USENIX Association; 2016. p. 265–283.

44. Iqbal Z, Chan HY. Concepts, key challenges and open problems of federated learning. Int J Eng. 2021;34(7):1667–1683. doi: 10.5829/ije.2021.34.07a.11

45. Ma J, Yonetani R, Iqbal Z. Adaptive distillation for decentralized learning from heterogeneous clients. In: 2020 25th International Conference on Pattern Recognition (ICPR). IEEE; 2021. p. 7486–7492. doi: 10.1109/ICPR48806.2021.9412356

46. Iqbal Z, Al-Aswadi FN, Shamsudin H, Aljaidi M, Al-Qablan TA, Pavleska T, et al. Adaptive dual-mode distillation for robust and communication-efficient federated learning under statistical and model heterogeneity. IEEE Access. 2026;14:145907– 145925. doi: 10.1109/ACCESS.2026.3735432