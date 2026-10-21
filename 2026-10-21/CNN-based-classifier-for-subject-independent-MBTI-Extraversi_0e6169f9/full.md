# CNN-based classifier for subject-independent MBTI Extraversion/Introversion preference classification from short-time EEG signals

Filip Postepski<sup>1\*</sup> and Grzegorz M. Wojcik<sup>1</sup>

<sup>1\*</sup>Department of Neuroinformatics and Biomedical Engineering, Institute of Computer Science and Mathematics, Maria Curie-Sk lodowska University, Akademicka 9, Lublin, 20-031, Lubelskie, Poland .

\*Corresponding author(s). E-mail(s): filip.postepski@mail.umcs.pl; Contributing authors: gmwojcik@umcs.edu.pl;

## Abstract

This study examines whether short, minimally preprocessed EEG segments contain suficient information to distinguish between individuals classified as extraverted or introverted according to the Myers–Briggs Type Indicator (MBTI). EEG recordings from 50 participants were divided into 4-s segments and analyzed using a subject-independent nested 5-fold cross-validation procedure to compare EEGNet, EEGProgress, and a newly proposed compact convolutional architecture, SSCNet. SSCNet achieved the highest mean accuracy (0.6481), AUC (0.7063), and F1-score (0.6276). The proposed architecture also showed the lowest variability across folds. SSCNet demonstrated the best computational eficiency, with the shortest median training epoch time and the highest training throughput despite having a similar number of trainable parameters to EEG-Progress. The results suggest that MBTI Extraversion–Introversion preferences may be moderately predictable from short time-domain EEG segments without manual spectral feature extraction, while highlighting the importance of subject-independent validation and computationally eficient model design.

Keywords:

eeg, cnn, personality, machine learning

## 1 Introduction

Individual psychological characteristics are commonly assessed using psychological questionnaires. Such instruments provide standardized measures of personality traits, cognitive styles, emotional characteristics, and other individual diferences. However, questionnaire-based assessment relies on self-reported information and therefore reflects how individuals describe or perceive themselves. The growing field of personality neuroscience has introduced a complementary perspective by investigating whether individual psychological characteristics are also reflected in measurable patterns of brain activity[1].

Electroencephalography (EEG) is particularly suitable for this type of research because it is a non-invasive brain activity measurement. It records electrical activity from the scalp with high temporal resolution and at relatively low cost compared to other neuroimaging modalities, such as fMRI or PET. It can be also less stressful for patients.[2, 3]

Extraversion and introvesion are one of the key personality dichotomies covered by Myers-Briggs Type Indicator(MBTI) questionnaire, alongside Sensing-Intuition (”S-I”), Thinking-Feeling $( ^ { , } \mathrm { T - F ^ { , } } )$ and Judging-Perceiving (”J-P”). The questionnaire distinguishes between introversion and extraversion according to how individuals engage with their environment. People with an introverted preference (”I”) are more likely to focus on their internal world and often prefer reflection, contemplation, and activities performed independently. On the other hand, those with an extraverted preference (”E”) tend to seek stimulation from the external world and are more likely to feel energized by communication, social activity, and interaction with others. Although the MBTI remains widely used, its dichotomous representation of personality preferences and some of its psychometric properties have been criticized in the literature [4].

A series of studies by Bhardwaj et al. have demonstrated that classification of personality dimensions collected from MBTI questionnaire is possible with high accuracy using genetic programming[5, 6]. However in this cycle of papers fast-fouries tranform (FFT) was used for signal features extraction.

Researchers have investigated whether extraversion-introversion diferences are reflected in EEG recordings. However, evidence for stable electro-physiological markers of extraversion remains inconsistent. A meta-analysis of 79 independent samples comprising 5700 participants found that resting frontal EEG asymmetry explained less than 0.4% of the variance in extraversion and neuroticism [7]. More recent studies have also produced mixed results. Pacheco et al. reported that Big Five personality traits could be decoded from periodic resting-state EEG power with modest predictive correlations of approximately r = 0.20 [8], whereas a preregistered replication involving 772 participants did not find meaningful prediction of Big Five traits from resting-state spectral EEG power [9]. Although the Big Five framework was used in those studies rather than the MBTI, their results indicate that EEG–personality associations may be weak and sensitive to the signal representation and analysis pipeline.

Machine-learning studies have provided evidence that personality relatedinformation can be extracted from EEG under some specific conditions. Baumgartl et al. achieved a balanced accuracy of 60.6% when distinguishing extraversion from introversion using resting-state EEG and a random forest classifier [10]. More recently, Penava and Buettner proposed a subject-independent CNN for predicting all five Big Five personality traits from resting-state EEG [11]. However, their pipeline first transformed EEG recordings into fine-grained spectral power features, removing the temporal domain before classification. Therefore, it remains unclear whether the Extraversion–Introversion preference can be classified directly from short, minimally preprocessed time-domain EEG segments without manual spectral feature extraction, and whether CNN-based architectures can generalize across individuals. This constitutes the main research problem addressed in the present study.

Key contributions of this study are as follows:

• Find out if any 4-s EEG segment can be classified as signal taken from Extraverted or Introvered person using CNN-based classifiers according to MBTI.

• Compare classification performance with two existing architectures of CNN-based models for EEG signal: EEGNet and EEGProgress under identical and evaluation conditions.

• Propose and evaluate a new CNN-based architecture, with the emphasis on computational eficiency, predictive performance, and robustness across subjectindependent cross-validation folds

## 2 Materials and methods

All models were trained on the same dataset. The dataset consisted of 50 recordings of EEG signal recorded for 20 minutes. Participants were listening to guided imagery recording (25 subjects) or resolving mental tasks (also 25 subjects). Also half of a dataset were represented by extraverts and the other half was represented by introverts. The Extraversion–Introversion labels were also approximately balanced across the two experimental conditions, as shown in Table 1.

Table 1 Distribution of Extraversion–Introversion labels across the experimental conditions.
<table><tr><td>Condition</td><td>Introversion</td><td>Extraversion</td><td>Total</td></tr><tr><td>Guided Imagery</td><td>13</td><td>12</td><td>25</td></tr><tr><td>Mental Task</td><td>12</td><td>13</td><td>25</td></tr><tr><td>Total</td><td>25</td><td>25</td><td>50</td></tr></table>

## 2.1 Participants and ethical approval

The EEG data used in the present study were acquired as part of a previously established experimental protocol. During the experiment, participants underwent either a Guided Imagery session intended to induce relaxation or a Mental Workload condition involving cognitively demanding memory-recall tasks. EEG recordings were collected under controlled laboratory conditions. The experimental procedure, EEG acquisition setup, and characteristics of the investigated conditions have been described in previous studies. The same EEG recordings have previously been investigated primarily in the context of distinguishing experimentally induced mental states[12]. This time the research sample is bigger. The present analysis included EEG recordings from 50 anonymized participants recruited according to the same eligibility criteria as in our previous study, comprising healthy, right-handed young adults aged 19–24 years. Before the EEG recording the study participants provided written informed consent and completed the Myers–Briggs Type Indicator (MBTI) personality questionnaire. The study protocol was reviewed and approved by the Maria Curie-Sklodowska Bioethics Committee. All methods were performed in accordance with the relevant guidelines and regulations.

## 2.2 Experimental procedure and EEG acquisition

For this study EGI 256 dense-array was used. For the signal recording Net Station 4.5.4 was used. The participants were ask to lay down with eyes closed to reduce artifacts from eye blinks and movement. The recordings were 20 minutes long. EEG signal was collected with sampling rate set to 250 Hz. The EGI GSN 130 dense-array cap was used for signal acquisition.

## 2.3 EEG signal preprocessing and dataset preparation

For this study 50 signals recorded from extraverted and introverted subjects were selected. The signal was preprocessed using mne python toolkit. Minimal preprocessing pipeline was done as mentioned it our previous work[12]. Bad channels were removed and interpolated using RANSAC algorithm. Each signal was filtered using 1 - 45 Hz band-pass filter. Next the whole recordings were split into 4-s segments. Only segments where subjects were listening to Guided Imagery recording or were solving mentaltasks were selected for classification. From the whole recording segments where in mental tasks recording when participats were listening to instructions were removed. Accordingly, complementary segments in the guided imagery recordings were also removed. This resulted in total of 122 segments for each subject. From all of 256 electrodes 69 were left. The selection was backed by the reduction table from full cap to 10-10 system in the [13] Geodesics Sensor Net manual.

## 2.4 Subject-independent cross-validation

For this study subject-independent nested 5-folds cross-validation procedure was applied. 40 subjects in each fold were used for development and hyperparameters selection. 10 subjects were left from each fold for final evaluation of 3 models generalization performance. Within those 40 subjects the training set contained 32 training subjects and 8 validation subjects. All EEG windows originating from a given subject were assigned exclusively to one partition, preventing subject-level data leakage. Identical cross-validation splits were used for all three architectures to ensure a direct and fair comparison. The nested 5-fold cross-validation procedure is depicted in Figure 1. The inner-folds validation datasets and outer-folds test datasets were also balanced according to the target label and stratified with respect to the recording condition.

![](images/a8bc919462bf0bc2b0dbaa03176d5c0670a8edd64c800a9c2561812637705829.jpg)  
Fig. 1 Nested cross-validation procedure used for models development and evaluation.

## 2.5 CNN-based classification models

Two CNN-based models were selected to compare with proposed architecture. EEGNet was used as a benchmark architecture. We also selected EEGProgress for comparison as it proposed newer diferent approach to EEG signal classification.

## 2.5.1 EEGNet

EEGNet was selected as one of the reference architectures for EEG signal classification. The model is a compact convolutional neural network originally proposed by Lawhern et al. for EEG-based brain–computer interface applications. Its architecture combines temporal convolutions with depthwise spatial filtering and separable convolutions, enabling the extraction of temporal and spatial features while maintaining a relatively low number of trainable parameters[14].

The implementation used in this study was based on the original EEGNet architecture and was adapted for binary classification. The number of temporal filters F1 was set to 8, the number of pointwise filters F2 to 16, and the depth multiplier D to 2. The temporal kernel length was set to 125 samples. The original output layer was replaced with a single-neuron layer using the Sigmoid activation function to provide the probability of belonging to one of the two personality classes. The configuration was also used in previous research based on smaller dataset[12]. The resulting model contained 3,397 trainable parameters.

## 2.5.2 EEGProgress

EEGProgress was selected as the second reference architecture. It was described as fast and lightweight architecture. It introduces diferent architecture idea which is to extract progressively information from electrodes, brain regions and electrodes[15].

The original EEGProgress framework combines a topological permutation stage with a progressive convolutional feature extractor composed of prior, electrode, region, and hemisphere convolution blocks. This design replaces a single global spatial convolution with several successive convolutional operations, reducing model complexity while preserving multi-scale spatial information. In the implementation used in the present study, EEGProgress was adapted to the 69-channel EEG input and binary Extraversion–Introversion classification task. The resulting architecture contained 16,245 trainable parameters.

## 2.5.3 SSCNet

SSCNet was proposed as a compact CNN-based architecture designed to process short multichannel EEG segments while limiting computational complexity. The model receives 4-s EEG windows represented by 1000 time samples and 69 channels. The first layer is a separable one-dimensional convolution with 64 filters, a kernel length of 90 samples, and a stride of 5. At the sampling frequency of 250 Hz, the kernel covers approximately 360 ms of the original EEG signal. A relatively long temporal kernel was selected to enable the extraction of temporal patterns extending over several hundred milliseconds, while the stride simultaneously reduces the temporal resolution and computational cost of the subsequent layers. Separable convolution was employed to reduce the number of trainable parameters compared with a conventional convolution while still allowing temporal feature extraction and subsequent interaction between EEG channels.

The second separable convolution uses 32 filters, a shorter kernel of 9, and a stride of 4. At this stage, the convolution operates on the feature representation generated by the first layer and further reduces the temporal dimension while extracting more local combinations of previously detected patterns. Considering both convolutional layers, each resulting feature covers approximately 130 samples, corresponding to about 520 ms of the original recording. Spatial dropout with a rate of 0.05 was applied after both separable convolutional layers to reduce dependence on individual feature maps while preserving the temporal structure of the signal. The number of filters is progressively reduced from 64 to 32 and subsequently to 16, forming a compact feature bottleneck.

A 1 × 1 convolution with 16 filters was then used to combine and compress the extracted feature representation without further reducing its temporal resolution. Instead of flattening the convolutional output and introducing additional fully connected layers, Global Average Pooling was applied to aggregate the learned features over the complete EEG segment. This design substantially limits the number of trainable parameters and makes the final prediction less dependent on the exact temporal position of a particular activation within the 4-s window. Finally, a single neuron with a Sigmoid activation function was employed for binary Extraversion–Introversion classification. The resulting SSCNet architecture contained 13,891 trainable parameters. The architecture design is shown in Figure 2.

![](images/4e497c568cb462584c065e226de11076dc995c0ee5f591f2f0d4d4295751be9a.jpg)  
Fig. 2 SSCNet model architecture.

## 2.6 Training procedure and hyperparameter selection

Hyperparameter selection was performed exclusively within the inner cross-validation loop of each outer training partition. All of the models were checked of the outer folds only once with no further optimization for the fair comparison. Also each model in the final stage was retrained on the same fold using 3 diferent randomly selected initialization seeds. Each architecture was tested with the same optimizer Adam and the loss function was binary cross-entropy. The learning rates were tested from established set of values: 0.003, 0.001, 0.0003 and 0.0001. Each model during 5 inner-folds was trained for maximum number of 150 epochs, but early stopping was enabled with patience of 15 epochs to select the best averaged epoch. The loss parameter was monitored for the best epoch selection.

## 2.7 Evaluation metrics

The classification performance of the evaluated architectures was assessed using accuracy (ACC), area under the receiver operating characteristic curve (AUC), F1-score, and Matthews correlation coeficient (MCC). The use of multiple evaluation measures provides a more comprehensive assessment of binary classification performance than accuracy alone [16].

Accuracy represents the proportion of correctly classified samples among all evaluated samples and is defined as:

$$
A C C = { \frac { T P + T N } { T P + T N + F P + F N } } ,\tag{1}
$$

where TP, TN, FP, and $F N$ denote true positives, true negatives, false positives, and false negatives, respectively.

The F1-score is the harmonic mean of precision and recall and provides a combined measure of the classifier’s ability to correctly identify positive samples while limiting false-positive predictions [16]. Precision and recall are defined as:

$$
P r e c i s i o n = \frac { T P } { T P + F P } ,\tag{2}
$$

$$
R e c a l l = \frac { T P } { T P + F N } ,\tag{3}
$$

and the F1-score is calculated as:

$$
F 1 = 2 \cdot \frac { P r e c i s i o n \cdot R e c a l l } { P r e c i s i o n + R e c a l l } .\tag{4}
$$

The area under the receiver operating characteristic curve (AUC) was used as a threshold-independent measure of the model’s ability to discriminate between the two classes [17]. The ROC curve represents the relationship between the true-positive rate and false-positive rate across diferent classification thresholds. AUC was calculated directly from the continuous probability scores produced by the output layer rather than from thresholded class predictions.

Matthews correlation coeficient was additionally employed as a balanced measure of binary classification quality [18]. MCC takes all four elements of the confusion matrix into account and is defined as:

$$
M C C = { \frac { T P \cdot T N - F P \cdot F N } { \sqrt { ( T P + F P ) ( T P + F N ) ( T N + F P ) ( T N + F N ) } } } .\tag{5}
$$

For ACC, F1-score, and MCC, continuous model outputs were converted into binary predictions using a classification threshold of 0.5. Performance metrics were calculated independently for each random seed within each outer subject-independent cross-validation fold. The three seed-specific results were first averaged within each fold. Overall performance was subsequently reported as the mean and standard deviation of the five resulting outer-fold values.

## 2.8 Computational eficiency evaluation

For the computational eficency evaluation we selected three measures: number of model parameters, median training epoch time and throughput which was number of training samples processed by the model during training per second. All of the models were trained for 20 epochs excluding the first which is typically longer due to the training initialization procedure. Then the results were averaged.

## 3 Results

Taking into account ACC metric, EEGNet showed the greatest susceptibility to seed initialization. EEGProgress, despite showing less variability depending on the seed, achieved the worst average ACC scores. SSCNet outperformed EEGProgress in terms of average ACC across all folds after 3 initializations. It is also the most stable of all architectures in this regard. It also outperforms EEGNet when considering the average ACC scores in folds 3 and 1. The standard deviation of paired results across the test folds demonstrates SSCNet’s greatest resistance to random seed initialization. The best fold across all networks for the three tested seeds is fold 4. EEGNet achieved the best average ACC in this fold after three initializations. SSCNet demonstrated the greatest robustness to initialization. EEGProgress showed the worst results. Each model achieved its worst results in a diferent fold. EEGProgress showed results below 0.5 in its worst fold. Detailed results, aggregated as the average of 3 initialization trials with a random seed, are shown in the Table 2.

Table 2 Classification accuracy (ACC) for the evaluated architectures across subject-independent cross-validation folds. Results are presented as mean ± standard deviation over three random seed initializations.
<table><tr><td>Fold</td><td>EEGNet</td><td> $\mathbf { \overline { { E E G P r o g r e s s } } }$ </td><td>SSCNet</td></tr><tr><td>Fold 0</td><td> $\mathbf { 0 . 6 2 0 2 \ : \pm { \ : 0 . 0 9 3 7 } }$ </td><td>一  $\overline { { 0 . 5 6 7 2 \pm 0 . 0 8 1 5 } }$ </td><td> $\overline { { 0 . 5 9 1 7 \pm 0 . 0 2 8 3 } }$ </td></tr><tr><td>Fold 1</td><td> $0 . 5 6 7 2 \pm 0 . 1 1 5 7$ </td><td> $0 . 5 1 9 0 \pm 0 . 0 3 0 4$ </td><td> $\mathbf { 0 . 6 4 6 6 \ : \pm { \ : 0 . 0 3 8 6 } }$ </td></tr><tr><td>Fold 2</td><td> $\mathbf { 0 . 6 8 2 8 \ : \pm { \ : 0 . 0 5 7 5 } }$ </td><td> $0 . 4 7 2 5 \pm 0 . 0 5 1 3$ </td><td> $0 . 6 6 5 2 \pm 0 . 0 2 3 0$ </td></tr><tr><td>Fold 3</td><td> $0 . 5 4 1 4 \pm 0 . 1 1 8 9$ </td><td> $0 . 5 7 5 0 \pm 0 . 0 5 9 0$ </td><td> $\mathbf { 0 . 6 3 7 6 \ : \pm { \ : 0 . 0 4 6 6 } }$ </td></tr><tr><td>Fold 4</td><td> $\mathbf { 0 . 7 6 8 0 \ : \pm { \ : 0 . 1 3 4 5 } }$ </td><td> $0 . 6 1 5 6 \pm 0 . 0 7 6 6$ </td><td> $0 . 6 9 9 7 \pm 0 . 0 5 6 8$ </td></tr></table>

SSCNet achieved the best overall performance in terms of ACC (0.6481), AUC (0.7063), and F1-score (0.6276), while also showing the lowest variability across subject-independent folds for these metrics. EEGNet obtained the highest MCC (0.2699), but its results were more variable across folds, particularly for AUC and

F1-score. EEGProgress showed the weakest overall classification performance, with the lowest mean values of ACC, AUC, and MCC among the evaluated architectures. Results are shown in the Table 3.

Table 3 Overall classification performance of the evaluated architectures. Results are presented as mean ± standard deviation across five subject-independent cross-validation folds, after averaging over three random seed initializations within each fold.
<table><tr><td>Model</td><td>ACC</td><td>AUC</td><td>F1-score</td><td>MCC</td></tr><tr><td>EEGNet</td><td> $0 . 6 3 5 9 \pm 0 . 0 9 1 6$ </td><td> $0 . 6 9 4 3 \pm 0 . 1 0 9 8$ </td><td> $0 . 6 0 9 8 \pm 0 . 1 0 6 8$ </td><td>0.2699 ± 0.1583</td></tr><tr><td>EEGProgress</td><td> $0 . 5 4 9 9 \pm 0 . 0 5 5 2$ </td><td> $0 . 5 7 8 6 \pm 0 . 1 1 6 2$ </td><td> $0 . 5 6 1 6 \pm 0 . 0 3 9 7$ </td><td> $0 . 1 1 5 7 \pm 0 . 0 9 2 4$ </td></tr><tr><td>SSCNet</td><td> $\mathbf { 0 . 6 4 8 1 \ : \pm { \ : 0 . 0 3 9 5 } }$ </td><td> $\mathbf { 0 . 7 0 6 3 \ : \pm { \ : 0 . 0 5 1 3 } }$ </td><td> $\mathbf { 0 . 6 2 7 6 \ : \pm { \ : 0 . 0 4 4 4 } }$ </td><td> $0 . 1 9 2 8 \pm 0 . 1 2 1 7$ </td></tr></table>

SSCNet demonstrated the highest computational eficiency, achieving the shortest epoch time (2.4511 s) and the highest throughput (3728.81 samples/s). Compared with EEGProgress, SSCNet was approximately 1.57 times faster per epoch and processed about 1.89 times more samples per second, despite having a comparable number of trainable parameters. EEGNet contained substantially fewer parameters, but exhibited the longest epoch time and the lowest throughput among the evaluated architectures. Table 4 shows the results of the several performance metrics from 20 epochs of training for each architecture tested.

Table 4 Computational eficiency of the evaluated architectures. Epoch time represents the average training time per epoch, while throughput denotes the number of training samples processed per second.
<table><tr><td>Model</td><td>Parameters</td><td>Epoch time (s)</td><td>Throughput (samples/s)</td></tr><tr><td>EEGNet</td><td>3,397</td><td>10.2283</td><td>902.5927</td></tr><tr><td>EEGProgress</td><td>16,245</td><td>3.8402</td><td>1973.5584</td></tr><tr><td>SSCNet</td><td>13,891</td><td>2.4511</td><td>3728.8085</td></tr></table>

## 4 Discussion

The results of this study indicate that short, 4-s EEG segments contain information that can be used to distinguish between individuals classified as introverted and extraverted according to the MBTI. Among the three evaluated CNN-based architectures, SSCNet achieved the highest mean ACC (0.6481), AUC (0.7063), and F1-score (0.6276). Moreover, SSCNet achieved lower variablility accross subject-idependent folds for AAC, AUC, and F1-score than the EEGNet and EEGProgress. These results suggest that the proposed architecture provides favorable balance between performance of prediction and stability over subjects groups of a 5-fold cross-validation. The obtained classification performance should be considered as moderate than as a highly accurate personality classifier.

The obtained results are rather consistent with previous studies suggesting that extraversion-related information can be extracted from EEG signals, although direct comparisons should be made cautiously because of diferences in personality models, EEG acquisition protocols, signal representations, and evaluation procedures.

An important finding of the presented study cover model stability. EEGNet was the most sensitive to random initialization, whereas SSCNet produced the lowest level of seed-related variability in four of five subject-independent folds. After averaging the three random initializations within each fold, SSCNet also exhibited the lowest variability across folds for three selected evaluation metrics such as: ACC (0.6481 ± 0.0395), AUC (0.7063 ± 0.0513), and F1-score (0.6276 ± 0.0444). Those characteristics are relevant when it comes to EEG signal classification, where inter-subject variability can cause considerable diferences in classifer performance over validation folds, as mentioned in latest research findings[19].

Overall, the results show that classification of MBTI Extraversion-Introversion preferences from short EEG segments can be done at a moderate performance level. SSCNet achieved the best mean ACC, AUC and F1-score while at the same time showed lower variability accross subject-independent folds and greater computational eficency than the two reference architectures. The main advantage should therefore be considered not as a huge improvment in classification accuracy, but as a balanced combination of comparable predictive performance, improved stability and reduced computational cost.

## References

[1] DeYoung, C.G., Beaty, R.E., Gen¸c, E., Latzman, R.D., Passamonti, L., Servaas, M.N., Shackman, A.J., Smillie, L.D., Spreng, R.N., Viding, E., Wacker, J.: Personality neuroscience: An emerging field with bright prospects. Personality Science 3(1), 7269 (2022) https://doi.org/10.5964/ps.7269 https://doi.org/10.5964/ps.7269

[2] Murphy, K.J., Brunberg, J.A.: Adult claustrophobia, anxiety and sedation in mri. Magnetic resonance imaging 15(1), 51–54 (1997)

[3] Introduction to Electroencephalography, pp. 1–14. John Wiley & Sons, Ltd (2021). Chap. 1. https://doi.org/10.1002/9781119386957.ch1 . https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119386957.ch1

[4] King, S.P., Mason, B.A.: Myers briggs type indicator. In: Carducci, B.J., Nave, C.S. (eds.) Wiley Encyclopedia of Personality and Individual Diferences: Volume II. Measurement and Assessment. John Wiley & Sons, Hoboken, NJ (2020)

[5] Bhardwaj, H., Tomar, P., Sakalle, A., Sakalle, M., Asthana, R., Bhardwaj, A., Ibrahim, W.: Personality prediction with hybrid genetic programming using portable eeg device. Computational Intelligence and Neuroscience 2022(1), 4867630 (2022)

[6] Bhardwaj, H., Tomar, P., Sakalle, A., Sakalle, M., Asthana, R., Bhardwaj, A., Ibrahim, W.: Personality prediction with hybrid genetic programming using portable eeg device. Computational Intelligence and Neuroscience 2022(1), 4867630 (2022) https://doi.org/10.1155/2022/4867630 https://onlinelibrary.wiley.com/doi/pdf/10.1155/2022/4867630

[7] Kuper, N., K¨ackenmester, W., Wacker, J.: Resting frontal eeg asymmetry and personality traits: A meta-analysis. European Journal of Personality 33(2), 154–175 (2019) https://doi.org/10.1002/per.2197 https://onlinelibrary.wiley.com/doi/pdf/10.1002/per.2197

[8] Pacheco, L.B., Feuerriegel, D., Jach, H.K., Robinson, E., Duong, V.N., Bode, S., Smillie, L.D.: Disentangling periodic and aperiodic resting eeg correlates of personality. NeuroImage 293, 120628 (2024) https://doi.org/10.1016/j.neuroimage.2024.120628

[9] Fruehlinger, C., Paul, K., Wacker, J.: Can personality traits be predicted from resting-state eeg oscillations? a replication study. Biological Psychology 193, 108955 (2024) https://doi.org/10.1016/j.biopsycho.2024.108955

[10] Baumgartl, H., Bayerlein, S., Buettner, R.: Measuring extraversion using eeg data. In: NeuroIS Retreat (2020)

[11] Penava, P., Buettner, R.: A novel subject-independent deep learning approach for user behavior prediction in electronic markets based on electroencephalographic data. Electronic Markets 35(1), 37 (2025)

[12] Postepski, F., Wojcik, G.M., Wrobel, K., Kawiak, A., Zemla, K., Sedek, G.: Recurrent and convolutional neural networks in classification of eeg signal for guided imagery and mental workload detection. Scientific Reports 15(1), 10521 (2025)

[13] Geodesics, I.: Net station waveform tools technical manual. s-man-200-wtfr-001 (Tech. Rep.). EGI (2003)

[14] Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., Lance, B.J.: Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces. Journal of Neural Engineering 15(5), 056013 (2018) https://doi.org/10.1088/1741-2552/aace8c

[15] Chen, Z., Yang, R., Huang, M., Li, F., Lu, G., Wang, Z.: Eegprogress: A fast and lightweight progressive convolution architecture for eeg classification. Computers in Biology and Medicine 169, 107901 (2024) https://doi.org/10.1016/j.compbiomed.2023.107901

[16] Sokolova, M., Lapalme, G.: A systematic analysis of performance measures for classification tasks. Information Processing & Management 45(4), 427–437 (2009) https://doi.org/10.1016/j.ipm.2009.03.002

[17] Fawcett, T.: An introduction to ROC analysis. Pattern Recognition Letters 27(8), 861–874 (2006) https://doi.org/10.1016/j.patrec.2005.10.010

[18] Matthews, B.W.: Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochimica et Biophysica Acta (BBA) - Protein Structure 405(2), 442–451 (1975) https://doi.org/10.1016/0005-2795(75)90109-9

[19] Del Pup, F., Zanola, A., Tshimanga, L.F., Bertoldo, A., Finos, L., Atzori, M.: The role of data partitioning on the performance of eeg-based deep learning models in supervised cross-subject analysis: A preliminary study. Computers in Biology and Medicine 196, 110608 (2025)

## Declarations

Some journals require declarations to be submitted in a standardised format. Please check the Instructions for Authors of the journal to which you are submitting to see if you need to complete this section. If yes, your manuscript must contain the following sections under the heading ‘Declarations’:

• Funding: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

• Competing interests: The authors declare that they have no competing interests.

• Ethics approval and consent to participate: The studies involving human participants were reviewed and approved by the Maria Curie-Sk lodowska University Bioethical Commission. The participants provided their written informed consent to participate in this study.

• Consent for publication: Not applicable. The manuscript does not contain identifiable personal data, images, or other information requiring individual consent for publication.

• Data availability: The EEG data analyzed in the present study contain human participant data and are not publicly available. The data may be made available by the corresponding author upon reasonable request, subject to applicable ethical and privacy restrictions.

• Author contribution: F.P.: Principal Investigator of the project, research project conceptualization at applied neuroinformatics dimension, concept of convolutional neural networks application, data processing, data analysis, manuscript preparation, models design and implementation, EEG recordings; G.M.W.: Key Investigator of the research, research idea and methodological support, manuscript preparation;

If any of the sections are not relevant to your manuscript, please include the heading and write ‘Not applicable’ for that section.

## Editorial Policies for:

Springer journals and proceedings: https://www.springer.com/gp/editorial-policies

Nature Portfolio journals: https://www.nature.com/nature-research/editorial-policies Scientific Reports: https://www.nature.com/srep/journal-policies/editorial-policies BMC journals: https://www.biomedcentral.com/getpublished/editorial-policies