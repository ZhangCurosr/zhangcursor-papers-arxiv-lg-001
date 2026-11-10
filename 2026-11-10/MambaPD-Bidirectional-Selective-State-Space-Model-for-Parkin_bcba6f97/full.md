# MambaPD: Bidirectional Selective State Space Model for Parkinson's Disease Detection from Resting-State EEG

Ayesha Zulfiqar

Dept. of Electrical Engineering, UET Lahore

ayeshaazulfiqar07@gmail.com

## ABSTRACT

Typical diagnosis of Parkinson's disease (PD) occurs only after significant dopaminergic loss, while resting-state EEG abnormalities like alpha slowing and reduced coherence occur earlier. While many published EEG-based PD detectors claim accuracy greater than 95%, many use epochlevel splits which leak subject identity. I introduce MambaPD, a bidirectional selective state space model for resting-state EEG-based PD detection, blending the channel mixing, patch tokenization and bidirectional Mamba encoder. It is compared with four baselines (TCN, BiLSTM with attention, EEGNet and a patch Transformer) in two public datasets (UNM, SanDiego) using subject-independent 5-fold cross-validation (checkpoints were selected on the held-out fold, so absolute estimates are optimistic). On UNM the accuracy is between 0.67 and 0.71, and MambaPD (0.714) is not statistically distinguishable from the Transformer (0.715), and TCN (0.708), with the highest sensitivity (0.82), while the lowest specificity (0.52). For the smaller SanDiego cohort, accuracy is in the range of 0.60 to 0.70, with the highest scores (0.703) from TCN and the lowest (0.601) from EEGNet, while MambaPD scores 0.631, with the lowest specificity (0.46) among the five models. The differences among the models on UNM are minor compared to the differences among folds. I report the whole pipeline, per-fold results, and the shortcomings of the protocol.

## 1. INTRODUCTION

Parkinson's Disease is a disorder that affects millions of people around the world and the incidence of this disorder is increasing at an alarming rate. PD is the second most common neurodegenerative disorder after Alzheimer's disease [1], [2]. PD results from the degeneration of dopaminergic neurons in the SNc(substantia nigra pars compacta) and the malfunctioning of cortico striatal pathways that regulate voluntary movements [3], [4]. Four motor symptoms of PD resting tremor, bradykinesia, cogwheel rigidity and postural instability is caused by loss of dopamine[5]. In the clinic, PD patients are characterized on the basis of motor phenotype (tremor dominant, akinetic rigid, mixed) and disease severity according to the Hoehn and Yahr scale, from unilateral disease (Stage I) to severe bilateral disease with dependency (Stage V) [6]. PD also has a wide-ranging non-motor syndrome, which includes cognitive dysfunction, rapid eye movement (REM) sleep behavior disorder, autonomic dysfunction, depression, hyposmia and psychosis, which all have a huge negative

impact on patient quality of life and clinical management [6], [7]. The most urgent unmet need for PD is the significant delay between the start of the disease and the onset of motor symptoms diagnostic for PD. By this time, 60-80% of dopaminergic neurons have been irreversibly lost [4], [7]. The five to ten year diagnostic delay precludes the possibility of neuroprotection. Moreover, even for specialist neurologists, PD because of its phenotypic overlap is sometimes confused with progressive supranuclear palsy, multiple system atrophy and drug induced parkinsonism [5]. These facts establish a critical clinical need for an objective, non-invasive biomarker to diagnose PD sooner and with greater accuracy than is possible today.

## 1.1 EEG as a Biomarker for Parkinson's Disease

Electroencephalography (EEG) is one of the most promising techniques for developing PD biomarkers. It has millisecond temporal resolution, is directly sensitive to the neurophysiological effects of dopamine loss, is noninvasive, portable, and much cheaper to acquire than PET or functional MRI [8], [20]. Resting-state EEG obtained while the patient is relaxed and sitting with eyes closed, without actively engaging in a task is of special interest because it reflects spontaneous brain activity that most directly reflects the tonic deficit in dopamine, is independent of symptom severity during task performance, and can be recorded even in a patient who cannot perform a task due to severe motor impairment.

The resting-state EEG of PD patients show a consistent pattern of spectral changes: Alpha slowing (8-13 Hz): The predominant posterior alpha rhythm slows and attenuates, an early and robust EEG abnormality in PD [9], [10]. Elevation of delta/theta (0.5-8 Hz): Slow-wave activity is increased globally, due to diffuse deafferentation of cortex [10], [11]. Beta desynchronization (13-30 Hz): Frontal-central beta power drops, due to motor preparation dysfunction [11]. Changes in gamma (30-60 Hz): Central and parietal gamma is associated with severity of cognitive impairment in PD [13] Decreased coherence: Inter-regional connectivity is reduced, indicating cortico-subcortical dysfunction [8], [12]

Importantly, some of these EEG changes can be detected in the prodromal period of PD decades before progression to motor symptoms [12]. This suggests that resting-state EEG has the potential to be used as a screening tool but only if the computational approaches used to process it are capable of extracting diagnostic information from its complex time series, long-duration, multi-channel recordings. The first machine learning approaches relied on hand-engineered features (common spatial patterns, wavelet coefficients, spectral power, sample entropy) and support vector machines (SVM) or linear discriminant analysis [14], [15]. These methods suffered from a lack of domain knowledge and lack of generalizability across sites and patient populations.

Deep learning largely overcame the need for feature engineering. Shaban and Amara [15] used a 20-layer CNN on wavelets-transformed EEG. Obayya et al. [16] developed a lightweight deep learning architecture for analysis of resting-state EEG. Jibon et al. [17] proposed a hybrid autoencoder and RBFNN approach for PD. Recent approaches have also used hybrid models. Bunterngchit et al. [18] proposed CTESM using CNNs, Transformer attention, and LSTM layers on SanDiego (non-subjectindependent). An explainable Transformer-based multi-head attention mechanism [19] achieved PD classification with attention explanations. TransformEEG [26] was studied on four harmonized datasets with 290 subjects via rigorous nested leave-N-subjects-out cross-validation one of the highest generalizable benchmarks reported.

However, several basic architectural problems remain. First, CNNs learn local temporal dependencies but lack the ability to handle long-range dependencies extending multiple seconds into the EEG the very long-term spectral modifications that characterize PD [20], [21]. Second, LSTMs are recurrent networks that squash all past information into a constant length hidden state vector, resulting in lost history over the long EEG sequences required for PD diagnosis [22]. Third, Transformers need to consider long-range dependencies using self-attention, but the number of pairwise comparisons scales as O(n²) with sequence length making it costly to process long raw EEG sequences and prone to overfitting on the small datasets usually observed in PD research [23][24]. State space models, like Mamba [25] can process a sequence in linear time and are used in other tasks with EEG data [39] – [40] [42]– [46]. A single fixed subject-wise split was used to evaluate a bidirectional Mamba on a TDBrain Parkinson’s subset [39] (25 PD patients and 25 controls). To the best of my knowledge, no such comparison of bidirectional Mamba models with the baseline models TCN, BiLSTM, EEGNet and Transformer in repeated subject-independent crossvalidation has been conducted on the UNM and SanDiego resting-state datasets. To this end, I propose MambaPD, a bidirectional Mamba model with channel mixing and patch tokenization, and wonder if it can compete with these baselines on a leakage-free protocol.

Mamba's hidden state is maintained and updated as it processes each token of the sequence (in this case each patch embedding 0.5 s). The update is gated by three functions of the input: Δ (how much of the hidden state is decayed), B (how much of the current input is written) and C (how much of the memory is read out). The motivation is that the model might highlight informative segments if these are provided, though that is not explored here for PD EEG. The bidirectional variant uses two Mamba stacks, forwards and backwards with the same patch sequence and merges the outputs.

## 1.2 Objectives

To apply a bidirectional selective state space model to detect PD using resting-state EEG

To compare it fairly with TCN, BiLSTM, EEGNet and patch Transformer on two public datasets across strict cross validation where it is independent of the subject.

To analyze where the models differ (sensitivity vs.   
specificity and fold level variability).

Cross validation was conducted subject-independent on two public datasets: UNM and SanDiego, with recordings from both medication states labeled as PD and the results were compared between MambaPD and TCN, BiLSTM, EEGNet and a patch Transformer. MambaPD matched the best baselines on UNM and was below the best baselines on SanDiego.

## 2. LITERATURE REVIEW

The performance of EEG based PD detection has been reported with a huge gap, and most of the variance is attributed to the evaluation procedure and not the model. Splitting epochs (not subjects) between training and testing, the accuracy in the small San Diego cohort of 31 subjects is 99% or better, with 99.65% reported for classical classifiers on fractal-dimension features [27] and 99.84% for a wavelet-scattering AlexNet pipeline [34] and 99% for an autoencoder-RBFNN hybrid [35]. If the subjects are kept separate, the reported figures drop: an SVM trained on band-power achieved only 68% when evaluated by subjectindependent evaluation [29]; an XGBoost classifier achieved 79% on an oddball-task dataset with strict subjectwise splits [31] and a transformer based model TransformEEG which was evaluated on four pooled datasets of 290 subjects, achieved 80.1% (the median was 78.45%) [26]. A number of the remaining studies have very small sample sizes or private datasets (Tables 1 and 2), and are not easily reproducible. A recent preprint reports 97.59% (±0.47) on the San Diego data with 5-fold cross-validation, but does not state whether the folds are subject-wise [41]. Bidirectional Mamba has also been used for PD detection: On a TDBrain dataset consisting of 25 PD patients and 25 controls, BioMamba achieves 96.77% accuracy on a single, fixed dataset split containing 8 test subjects, [39]. These small single-split evaluations have a high variance and repeated cross-validation was therefore encouraged here, as in this case.

<table><tr><td colspan="5">Table 1. Machine Learning techniques for EEG based pd detection ML classifier</td><td colspan="2"></td><td rowspan="2">Limitations</td></tr><tr><td colspan="3">Authors ref year Dataset used Dataset statistics</td><td colspan="2">Experimental setup</td><td colspan="2">Performance results</td></tr><tr><td rowspan="2">Lal et al. [27], 2024</td><td rowspan="2">UC San Diego rs-EEG</td><td rowspan="2">31 subj: 15 PD (ON/OFF),</td><td rowspan="2">KNN, XGBoost, k-fold CV</td><td>Subject Split</td><td></td><td rowspan="2">Precision Recall</td><td rowspan="2"></td></tr><tr><td>Validation Strategy</td><td>Accuracy 99.65% (ON) /</td></tr><tr><td rowspan="2"></td><td></td><td>16 HC, 64-ch</td><td>LightGBM, Extra Trees</td><td></td><td>Epoch-level only (not</td><td>99.45% (OFF)</td><td>Small cohort (31 subj), no subject-independent validation, single dataset, accuracy likely inflated due</td></tr><tr><td>Private EEG (Erciyes</td><td></td><td>SVM, KNN, FNN,</td><td></td><td>subject-wise)</td><td></td><td>to epoch-level leakage Private dataset, not reproducible, no subject</td></tr><tr><td rowspan="4">Latifoğlu et al. [28], 2024 Bera et al. [29], 2025</td><td>Univ., Turkey)</td><td>PD ON, PD OFF, HC; 32-ch; 3- class</td><td>Decision Tree</td><td>k-fold CV</td><td>Not reported</td><td>~100%</td><td>independent test cohort size not clearly reported</td></tr><tr><td>UC San Diego + IOWA</td><td>UCSD: 31 subj (15 PD, 16 HC);</td><td>SVM (RBF kernel)</td><td>5-fold CV</td><td>(VKF+SVM) 82% (UCSD) /</td><td></td><td>SVM drops to 68% under subject-independent</td></tr><tr><td>rs-EEG</td><td>IOWA: separate cohort</td><td></td><td></td><td>Subject- dependent +</td><td>94% (IOWA);</td><td>setting, only standard band power features tested,</td></tr><tr><td></td><td></td><td></td><td>subject- independent</td><td>68% subj-indep</td><td></td><td>no ensemble ML explored</td></tr><tr><td>Sahota et al. [30], 2024</td><td>Private Early-Stage EEG (Univ. of Bristol)</td><td>30 subj: 11 PD (HY1), 19 HC; 5 arousal states</td><td>AdaBoost, Decision Tree</td><td>Hold-out</td><td>Not reported</td><td>91% (N1, 96% PLI+stats)</td><td>80% Very small cohort (11 PD), no cross-dataset validation, sleep EEG acquisition adds clinical</td></tr><tr><td>Avola et al. [31], 2025</td><td>Public EEG Oddball Task</td><td>PD vs HC; public; subject-wise enforced</td><td>XGBoost, k-NN, RF, LR, SVM</td><td>Subject-wise CV</td><td>Strict subject-</td><td>79% (XGBoost)</td><td>complexity Single oddball-task dataset, findings may not transfer</td></tr><tr><td rowspan="2">Allahbakhshi et al. [32], 2024</td><td>Multi-source EEG PD</td><td>Diverse PD + HC; multi-site</td><td>SVM (RBF +</td><td>k-fold CV</td><td>wise splits</td><td></td><td>to resting-state EEG, no Mamba/SSM model included Preprint only, not peer reviewed, no subject</td></tr><tr><td>Dataset</td><td>demographic</td><td>polynomial)</td><td></td><td>Not reported</td><td>~88%</td><td>independent evaluation, dataset demographic details</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td>unclear</td></tr><tr><td></td><td></td><td></td><td></td><td>Not reported</td><td>&gt;80%</td><td>&gt;80%</td></tr><tr><td rowspan="2">Sahota et al. [33], 2024</td><td>Public Early-Stage PD</td><td>11 PD, 19 HC; 15-variate band</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>power + peak freq series</td><td>Decision Tree</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td></td><td></td><td>AdaBoost,</td><td>Hold-out</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>(AdaBoost, N1)</td><td>validation, HY1-specific dataset limits generalizability</td></tr><tr><td rowspan="2">EEG</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Very small cohort (11 PD), no cross-dataset</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="5">Table 2. Deep Learning techniques for EEG based pd detection Authors ref year</td><td colspan="2"></td><td rowspan="2"></td><td rowspan="2">Limitations</td></tr><tr><td>Dataset used</td><td>Dataset statistics</td><td>ML classifier</td><td>Experimental setup</td><td></td><td></td><td>Performance results</td></tr><tr><td rowspan="2">Siuly et al. [34], 2024</td><td>UC San Diego rs-EEG +</td><td>San Diego: 31 subj (15 PD</td><td>WST + AlexNet</td><td>Validation Strategy Train/validation/test split</td><td>Subject Split Epoch level</td><td>Accuracy 99.84% (San</td><td rowspan="2">Precision Recall</td><td rowspan="2">No subject independent evaluation, epoch-level split inflates accuracy, single AlexNet architecture</td></tr><tr><td>IOWA EEG</td><td>ON/OFF, 16 HC), 64-ch, 512 Hz; IOWA: separate PD/HC cohort</td><td>CNN (2D time frequency</td><td></td><td>only (not Diego) subject-wise)</td><td>95.79% (IOWA)</td></tr><tr><td>Jibon et al. [35], 2024</td><td>UC San Diego rs-EEG</td><td>31 subj (15 PD, 16 HC); 93 min of recordings; PSD sub-band analysis</td><td>images) Autoencoder (AE) + RBFNN hybrid deep</td><td>Hold-out validation</td><td>Epoch level (not subject wise)</td><td>99%</td><td>91.77% 84.71%</td><td>Single small dataset (31 subj), no subject independent test, RBFNN classification layer limits deep feature learning; no cross-dataset validation</td></tr><tr><td>Bera et al. [29], 2025</td><td>UC San Diego rs-EEG + IOWA EEG</td><td>UCSD: 31 subj (15 PD, 16 HC); IOWA: separate PD/HC cohort; five freq. bands (α, β, θ, γ, δ)</td><td>learning Multi-band CNN (cross frequency feature fusion)</td><td>5-fold CV</td><td>Subject dependent + subject</td><td>96% (UCSD); 99% (IOWA)</td><td></td><td>No subject independent CNN result reported (only SVM tested under subject independent), small datasets, CNN drops sharply in generalization</td></tr><tr><td>Del Pup et al. (TransformEEG) [26], 2025</td><td>4 pooled public rs-EEG datasets</td><td>290 subj total: 140 PD, 150 HC; UCSD, UNM, lowa, Turku datasets harmonized</td><td>Depth wise Conv Tokenizer + Transformer</td><td>10-outer × 10-inner Nested LNSO-CV</td><td>independent tested Strict subject wise (N-LNSO)</td><td>80.1% (w/ augmentation + threshold</td><td></td><td>settings Still below clinical threshold (80%), unidirectional attention only, no backward temporal pass, sensitive to preprocessing pipeline choices</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>(Dongguan Univ. of Technology)</td><td>channel EEG; parallel and series CNN-LSTM configurations</td><td>LSTM + Series</td><td></td><td></td><td>CNN-LSTM, best)</td><td></td><td>transformer based or GNN architectures</td></tr><tr><td>Li et al. [36], 2024</td><td>Private EEG</td><td>PD patients vs. HC; multi-</td><td>Parallel CNN-</td><td></td><td></td><td>median 78.45%</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Encoder (TransformEEG)</td><td></td><td></td><td>correction);</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>k-fold CV</td><td>Not reported</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>97% (series</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Private dataset, not reproducible, no subject independent evaluation, no comparison against</td></tr></table>

<table><tr><td colspan="3"></td><td colspan="5">CNN-LSTM hybrid</td></tr><tr><td>Bunterngchit et al. [37], 2025</td><td>UC San Diego rs-EEG</td><td>31 subj (15 PD, 16 HC); 40-ch; 500 Hz; spectral power, frequency band ratios, wavelet coefficients</td><td>CTESM: CNN + Transformer Test split blocks + LSTM layers</td><td>Train / Validation / wise)</td><td>Epoch level 99.7% (not subject</td><td></td><td>Very small dataset (31 subj), no subject independent evaluation, high accuracy likely epoch level inflated, no multi-center validation</td></tr><tr><td>GEPD: GAN + CNN [38], 2025</td><td>UC San Diego + UNM rs-EEG</td><td>San Diego: 31 subj; UNM dataset: separate PD/HC cohort; cross-dataset generalization evaluated</td><td>GAN (data Cross-dataset augmentation) + validation CNN classifier (GEPD framework)</td><td>Strict cross dataset subject wise</td><td>84.3%</td><td></td><td>GAN training is unstable and computationally expensive, limited to two datasets, no transformer or SSM based temporal modeling, no medication state analysis</td></tr></table>

Table 3. Other Related Studies to MambaPD
<table><tr><td>Authors (ref), Year Qian et al. (BioMamba) [39],</td><td>Dataset Used 6 public</td><td>Dataset Statistics</td><td>Architecture / Method</td><td>Validation Strategy</td><td>Subject Split</td><td>Performance results</td><td>Limitations TDBrain PD subset (25 PD, 25 HC, eyes-closed);</td></tr><tr><td>2025 Neves et al. (EEG-ChiMamba)</td><td>biosignal datasets (EEG, ECG) CAUEEG (Chung-</td><td>5 EEG + 1 ECG; binary tasks (AD, PD, workload, emotion, MI); extension adds wearable HAR 1,155 subjects; HC vs MCl vs</td><td>Spectro-Temporal Embedding + Bidirectional Mamba + Sparse FFN Decoupled channel wise</td><td>Per-dataset (no cross- dataset); 5 seeds; 6 metrics vs 8 baselines Strict subject-wise</td><td>Subject-wise, fixed per dataset Strict subject wise</td><td>SOTA on 5/6 datasets (all 6 metrics); TDBrain PD: 96.77% acc 59.27% (3-class, val)</td><td>single fixed split, 8 test subjects; no cross- validation; not compared with UNM/San Diego; ~53% acc on DREAMER Dementia not PD, 3-class reduces accuracy vs</td></tr><tr><td>[40], 2026 Dhavapandiammal &amp;</td><td>Ang Univ. Hospital) UC San Diego rs- EEG +</td><td>dementia; 3-class; rs-EEG UCSD: 31 subjects (15 PD, 16</td><td>SSM + Mamba temporal layers + Inverted LayerNorm Common Spatial Pattern</td><td>split; validation + held out test 5-fold CV + external</td><td>Not reported (5-</td><td>97.59% ± 0.47 (UCSD, 5-</td><td>binary PD/HC, rs-EEG paradigm transferable but not tested on PD Preprint, not peer reviewed; fold split not stated</td></tr><tr><td>Palanisamy (CSFE-DAE)[41], 2026 Gui et al. (EEGMamba) [42],</td><td>independent OpenNeuro PD dataset 8 public EEG</td><td>HC); OpenNeuro cohort size not stated in abstract CHB-MIT (seizure), SleepEDF</td><td>features + Dual-Attention EEGNet (temporal + spatial attention); Gaussian-noise augmentation Bidirectional Mamba + ST</td><td>validation on OpenNeuro Dataset-wise per task;</td><td>fold CV) Subject wise per</td><td>fold CV) SOTA on majority of 8</td><td>as subject-wise; CNN-based, no long-range or bidirectional SSM modeling; small UCSD cohort Never applied to PD rs-EEG, MoE adds</td></tr><tr><td>2024 Tegon et al. (FEMBA) [43],</td><td>datasets TUAB + TUAR</td><td>(sleep), SEED (emotion), BCICIV2a (MI); variable channels/lengths 21,000+ hrs unlabeled</td><td>Adaptive module + Task aware MoE Bidirectional Mamba SSM +</td><td>ablation confirms BiMamba &gt; unidirectional Fine-tune + held-out</td><td>task Subject wise held</td><td>datasets; linear O(n) 81.82% bal. acc (TUAB);</td><td>overhead, no resting state or PD specific evaluation Not evaluated on PD; requires large pretraining</td></tr><tr><td>2025 Saeed et al. (EEG-SSM) [44],</td><td>(Temple Univ. Hospital) OpenNeuro</td><td>pretraining; TUAB: normal/abnormal EEG; TUAR: artifact rejection 88 subjects: 36 AD, 23 FTD, 29</td><td>self-supervised random masking; 4 model sizes (7.8M-large) ICA preprocessing +</td><td>test per task Leave-one-subject-</td><td>out Subject wise</td><td>AUROC 0.8921 AD class: sens 85.9%,</td><td>corpus; domain shift to PD rs-EEG unaddressed Dementia (AD/FTD), not PD; small cohort (88);</td></tr><tr><td>2026</td><td>ds006036 (open- eyes, photic- stimulation EEG)</td><td>HC; 3-class diagnosis + MMSE/duration regression</td><td>AutoML-driven few-shot learning + SSM dual-task head (classification + severity regression) + SHAP explainability</td><td>out (LOSO)</td><td>(LOSO)</td><td>spec 93.8%, F1 85.5, AUC 0.98</td><td>photic-stimulation EEG, not resting state; no bidirectional selective-SSM design reported; code not yet released</td></tr><tr><td>Guo et al. (MI-Mamba) [45], 2025 Behrouz &amp; Hashemi (Brain-</td><td>BCI Competition IV 2a + 2b 7 real-world</td><td>9 subjects each; 4-class and 2- class MI; multi-channel EEG EEG + fMRI; neurological disease</td><td>CNN (local spatial features) + Mamba (global temporal features) + FC classifier MLP variate fusion + S4</td><td>BCI competition protocol Cross-dataset per</td><td>Subject wise Subject wise per</td><td>80.59% (2a, 4-class); 84.42% (2b, 2-class)</td><td>Motor imagery only, unidirectional Mamba, no frequency band injection, ~6x fewer params but not adapted for PD Not PD specific, EEG only PD resting state not</td></tr><tr><td>Mamba) [46], 2024</td><td>datasets; 3 modalities</td><td>detection tasks; cross-modality and cross-resolution evaluation</td><td>SSM per time-series; multi modal neurological disease representation</td><td>modality</td><td>dataset</td><td>SOTA on 7 datasets across 3 modalities</td><td>evaluated, S4 (not bidirectional Mamba) as backbone</td></tr></table>

![](images/118f2c07dcab37fc67aea198b311fb4473b76e81b297ed8a3af7ab9936e7b298.jpg)  
Fig. 1. Overview of the proposed MambaPD framework for resting-state EEG-based Parkinson's Disease detection

## 3. METHODOLOGY

Figure 1 shows the end-to-end pipeline in the proposed MambaPD framework: raw EEG acquisition, raw EEG preprocessing, EEG epoch segmentation, channel mixing, patch tokenization, bidirectional Mamba encoding and binary classification. The methodology has five parts: dataset description, preprocessing, baseline architectures, the proposed MambaPD architecture, the training and evaluation.

## 3.1 Datasets

The University of New Mexico (UNM) resting-state EEG dataset was recorded at the University of New Mexico, and is among the largest publicly available Parkinson disease EEG datasets. This consists of 56 subjects: 28 clinically diagnosed Parkinson's Disease patients and 28 age matched healthy controls. EEG data were recorded with a 64-channel Brain Vision amplifier system (63 scalp EEG channels plus VEOG) with 3 auxiliary channels for the accelerometers giving 67 channels recorded at 500 Hz at the international 10-20 placement. After 12 hours of withdrawal from the dopaminergic drugs (OFF-medication condition), each PD subject participated in two separate recording sessions, one while taking the prescribed dopaminergic drug therapy (ONmedication condition) and one after the 12-hour period of medication withdrawal (OFF-medication condition). All healthy controls had a single session of rest-state recordings. In all sessions, the people sat quietly in a comfortable chair with their eyes closed for \~3 minutes to reduce motor artefacts and the neural activity associated with cognitive tasks. The raw recordings consist of 67 channels: 1-63 channels are scalp EEG electrodes to capture the electrical activity of the scalp, 64 is a channel sensitive to eyemovement artifacts (vertical electrooculography or VEOG), and 65-67 are the three channels of a triaxial accelerometer mounted on the dominant hand to separately measure tremor. The four non-cerebral channels excluded from analysis leaving 63 pure EEG channels for each recording. All epochs from PD patients were labeled as y = 1 and all healthy control epochs as y = 0. All sessions of ON and OFF are combined within one PD label.

The University of California San Diego dataset [47] has been acquired at the University of California San Diego and is made public in the standardized format of neuroimaging data known as Brain Imaging Data Structure (BIDS). It consists of 31 subjects (15 clinically diagnosed PD patients and 16 healthy controls). Recordings were made with a 40- channel amplifier system (Biosemi ActiveTwo) with a sampling rate of 512 Hz. The electrode layout is based on a modified 10-20 system that has been optimized for the Biosemi ActiveTwo headset. Two recording sessions ses-on (recorded on medication) and ses-off (recorded off medication) were completed by PD patients. Healthy controls underwent one session (ses-hc). All the recordings are resting state (eyes closed), and they are of about three minutes length. The BIDS structure contains the data in a hierarchical structure.

The two datasets vary in number of electrodes (63 vs 40), recording hardware (BrainVision vs Biosemi ActiveTwo), sampling frequency (500 vs 512 Hz), number of subjects (56 vs 31), and source of institutions (University of New Mexico vs UC San Diego). Each dataset was used separately for training and testing. No cross-dataset training and testing was implemented. The two sets of data provide two withindataset evaluations with different recording conditions.

## 3.2 EEG Preprocessing

A standardized preprocessing pipeline was applied to both datasets using MNE-Python. Figure 2 illustrates the complete pipeline.

The UNM dataset was subjected to the same procedure: the four channels (VEOG and the three accelerometer channels) not related to EEG were excluded, leaving 63 pure scalp EEG channels. The SanDiego dataset was one of the sets that did not need any channels to be removed so it contained all 40 EEG channels. A zero-phase Hamming-windowed finite impulse response (FIR) bandpass filter was

<table><tr><td colspan="3">Table 4. Dataset Configuration</td></tr><tr><td>Property</td><td>UNM Dataset</td><td>SanDiego Dataset</td></tr><tr><td>Institution</td><td>Univ. of New Mexico</td><td>UC San Diego</td></tr><tr><td>Total Subjects</td><td>56</td><td>31</td></tr><tr><td>PD Patients</td><td>28</td><td>15</td></tr><tr><td>Healthy Controls</td><td>28</td><td>16</td></tr><tr><td>EEG System</td><td>BrainVision 64 ch</td><td>Biosemi Activetwo 40 ch</td></tr><tr><td>EEG Channels (used)</td><td>63</td><td>40</td></tr><tr><td>Native Sampling Rate</td><td>500Hz</td><td>512Hz</td></tr><tr><td>After Resampling</td><td>256Hz</td><td>256Hz</td></tr><tr><td>Sessions per PD</td><td>2 (ON + OFF medication)</td><td>2 (ses-on + ses-off)</td></tr><tr><td>Sessions per HC</td><td>1</td><td>1 (ses-hc)</td></tr><tr><td>Recording Condition</td><td>Eyes closed, resting</td><td>Eyes closed, resting</td></tr><tr><td>Data Format</td><td>MATLAB .mat (EEGLAB)</td><td>BDF (BIDS)</td></tr><tr><td>Epoch Length</td><td>4 seconds (1024 samples)</td><td>4 seconds (1024 samples)</td></tr><tr><td>Overlap</td><td>50% (512 sample step)</td><td>50% (512 sample step)</td></tr><tr><td>Total Epochs</td><td>8882</td><td>4475</td></tr><tr><td>Healthy Epochs</td><td>2995</td><td>1513</td></tr><tr><td>PD Epochs</td><td>5887</td><td>2962</td></tr></table>

![](images/9117d7cd3b7c01b51671b2389b3c3de7ec8e9e6bcb288a19cef18d796982c7d2.jpg)  
Fig. 2. EEG preprocessing pipeline applied to both datasets.

used on the data, with passband 0.5–40 Hz:

$$
x _ { f i l t e r e d } = x _ { r a w } ( t ) * h ( t )\tag{1}
$$

The Hamming-windowed FIR is denoted by h(t) and \* represents linear convolution. Lower cutoff (0.5 Hz) will decrease the electrode drift and galvanic skin response. The high end of 40 Hz filters out power line (50/60 Hz) and electromyographic (EMG) interference. This passband keeps all 5 bands of frequencies that correspond to PD delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz) and low gamma (30–40 Hz), Gamma above 40 Hz is not analyzed. Both sets of data were uniformly resampled to 256 Hz using MNE’s FFT-based resampling. The 256 Hz sampling rate is 6.4 times the highest retained frequency (40 Hz), well above the Nyquist requirement of 80 Hz and the frequency of the signals that are diagnostically important is not aliased. Resampling will cut the dimension of the data by about half of the original rates.

The continuous preprocessed recording was divided into fixed length epochs with a sliding window:

$$
E _ { i } = x [ i . S : i . S + W ] , \qquad i = 0 , 1 , 2 , \ldots \ldots\tag{2}
$$

Where $\mathbf { x } { \in } \mathbf { R } ^ { \mathrm { C } \times \mathrm { T } }$ is the continuous EEG signal with C channels and T total timepoints, W=1024 samples (4 seconds at 256 Hz) is the window length, and S=512 samples (2 seconds, 50% overlap) is the step size. The window length of 4 seconds captures at least 2 full cycles of the slowest diagnostically relevant oscillation of 0.5 Hz (delta) (period 2 s). An 8,882/4,475 epoch overlap gives about double the number of training samples with no added information content at UNM and SanDiego respectively. Independently normalization per channel was done for each epoch to remove inter subject amplitude variability:

$$
\begin{array} { r } { \hat { x } \left( \mathrm { c } , \mathrm { t } \right) = \frac { \mathrm { x ( c , t ) - \mu } _ { c } } { \sigma _ { c } + \varepsilon } } \end{array}\tag{3}
$$

Where $\begin{array} { r } { \mu _ { c } = \frac { 1 } { W } \sum _ { t = 1 } ^ { W } x ( c , t ) } \end{array}$ is the channel mean over the epoch window, $\sigma _ { \mathrm { c } }$ is the corresponding standard deviation, and $\varepsilon { = } 1 0 ^ { - 8 }$ prevents division by zero. Once normalized the mean of each channel is zero and variance is one. This normalization eliminates absolute amplitude and channel amplitude differences within each epoch (hence absolute band power), maintains the same shape of the waveforms for each channel and correlations between channels. During the pre-processing a subject identity array s was stored, in which s<sub>i</sub> contained the subject identifier for the i-th epoch. This array ensures that no epoch of any test subject is added to the training partition and at the same time that cross validation is strictly performed at patient level.

## 3.3 Baseline models

Four baseline models of deep learning sequence architectures were implemented. All receive input tensors $\mathbf { X } { \in } \mathbf { R } ^ { \mathrm { { B } { \times } \mathbf { C } { \times } \mathrm { { T } } } }$ where B is batch size, C is EEG channels and T=1024.

## Temporal Convolutional Network (TCN)

The stacked dilated causal convolutions with residual connections are used in the TCN. Each block expands by two dilated convolutions, such that each block is expanded (dilation factor di=2i) at the i-th block:

$$
h _ { i } = \mathrm { E L U } \left( \mathrm { B N } \left( \mathrm { C o n v 1 D } \left( \mathrm { h } _ { i - 1 } , \mathrm { d } = 2 ^ { i } \right) \right) \right) + \mathrm { W } _ { s } \mathrm { h } _ { i - 1 }\tag{4}
$$

where BN(⋅) represents batch normalization, $\mathrm { W _ { s } }$ represents a 1×1 convolution that matches channel dimensions and ELU(⋅) represents the exponential linear unit activation. The total receptive field of the stacked blocks is:

$$
\begin{array} { r } { R F = 1 + 2 ( k - 1 ) \sum _ { i = 0 } ^ { n - 1 } 2 ^ { i } = 1 + 4 ( 2 ^ { n } - 1 ) } \\ { \qquad } \\ { = 2 5 3 \mathrm { ~ t i m e s t a m p s } ( \mathrm { k } = 3 , \mathrm { n } = 6 \mathrm { b l o c k s } ) } \end{array}\tag{5}
$$

This is roughly 25% of the entire 1024 timestep epoch, which represents medium range temporal dependencies. The temporal dimension is reduced in global average pooling and the binary output is created in a fully connected classifier.

## Bidirectional LSTM with Temporal Attention (BiLSTM)

The bidirectional LSTM model works with EEG sequences in both temporal directions. The forward LSTM at each time step t calculates:

$$
\mathbf { f } _ { \mathrm { t } } = \sigma ( \mathbf { W } _ { \mathrm { f } } [ \mathrm { h } _ { \mathrm { t } - 1 } , \mathrm { x } _ { \mathrm { t } } ] + \mathbf { b } _ { \mathrm { f } } )\tag{6}
$$

$$
\mathrm { i _ { t } = \sigma ( W _ { i } [ h _ { t - 1 } , x _ { t } ] + b _ { i } ) }\tag{7}
$$

$$
\widetilde { \mathrm { c } } _ { \mathrm { t } } = \mathrm { t a n h } ( \mathrm { W } _ { \mathrm { c } } [ \mathrm { h } _ { \mathrm { t } - 1 } , \mathrm { x } _ { \mathrm { t } } ] + \mathrm { b } _ { \mathrm { c } } )\tag{8}
$$

$$
\mathbf { c } _ { \mathrm { t } } = \mathbf { f } _ { \mathrm { t } } \odot \mathbf { c } _ { \mathrm { t - 1 } } + \mathrm { i } _ { \mathrm { t } } \odot \boldsymbol { \widetilde { \mathbf { c } } } _ { \mathrm { t } }\tag{9}
$$

$$
\begin{array} { r } { \boldsymbol { 0 } _ { \mathrm { t } } = \sigma ( \mathsf { W } _ { 0 } [ \mathsf { h } _ { \mathrm { t } - 1 } , \mathsf { x } _ { \mathrm { t } } ] + \mathsf { b } _ { \mathrm { o } } ) } \end{array}\tag{10}
$$

$$
\mathrm { h } _ { \mathrm { t } } = \mathbf { 0 } _ { \mathrm { t } } \odot \mathrm { t a n h } ( \mathrm { c } _ { \mathrm { t } } )\tag{11}
$$

where $\mathrm { f } _ { \mathrm { t } } , \mathrm { i } _ { \mathrm { t } } , \mathrm { o } _ { \mathrm { t } }$ are forget, input and output gate activations; ct is the cell state; σ(⋅) is the sigmoid function and ⊙ is the element-wise multiplication operator. The reversed sequence is passed through the backward LSTM, which yields $\tilde { h } _ { \mathrm { t } } .$ Concatenated bidirectional states $\mathrm { H } _ { \mathrm { t } } { = } [ \mathrm { h } _ { \mathrm { t } } \ ; \ \tilde { h } _ { \mathrm { t } } ] { \in } \mathrm { R } ^ { 2 \mathrm { d } }$ at each timestep are passed through a temporal attention mechanism:

$$
\mathbf { e } _ { \mathrm { t } } = \mathbf { w } _ { \mathrm { a } } ^ { \top } \operatorname { t a n h } ( \mathbf { W } _ { \mathrm { a } } \mathbf { H } _ { \mathrm { t } } + \mathbf { b } _ { \mathrm { a } } )\tag{12}
$$

$$
\begin{array} { r } { \alpha _ { t } = \frac { \exp ( e _ { t } ) } { \sum _ { t ^ { \prime } = 1 } ^ { T } \exp ( e _ { t ^ { \prime } } ) } } \end{array}\tag{13}
$$

$$
\begin{array} { r } { c = \sum _ { t = 1 } ^ { T } \alpha _ { t } H _ { t } } \end{array}\tag{14}
$$

where $\mathbf { \alpha } \mathbf { a } _ { \mathrm { t } }$ are the normalized attention weights and c is the context vector passed to the classifier.

## EEGNet

EEGNet consists of 3 in succession convolutional layers tailored specifically for EEG. Block 1 does a temporal convolution:

$$
\mathrm { Z _ { 1 } \mathrm { = B N ( C o n v 2 D ( X _ { i } , F _ { 1 } , k e r n e l = ( 1 , 6 4 ) ) ) } }
$$

The depth wise spatial convolution in the C electrode channels is carried out by block 2:

$$
\mathrm { Z _ { 2 } \mathrm { = A v g P o o l ( E L U ( B N ( D e p t h w i s e C o n v 2 D ( Z _ { 1 } , F _ { 1 } , k e r n e l = ( C , 1 ) ) ) ) } }
$$

block 3 is a standard convolution over the feature maps with a temporal kernel:

$$
\mathrm { Z _ { 3 } \mathrm { = A v g P o o l ( E L U ( B N ( C o n v 2 D ( Z _ { 2 } , F _ { 2 } , k e r n e l = ( 1 , 1 6 ) ) ) ) } ) }
$$

with parameters $\mathrm { F } _ { 1 } { = } 8 , \mathrm { D } { = } 2 , \mathrm { F } _ { 2 } { = } 1 6$ , The number of trainable parameters for EEGNet is roughly 6.7k.

## Patch Transformer

The Patch Transformer tokenizes the EEG into P=8 nonoverlapping patches of length L<sub>p</sub>=128:

$$
\mathrm { z _ { p } { = } W _ { e } \ . \ F l a t t e n ( X ( : , ( p { - } 1 ) L _ { p } { : } p L _ { p } ) ) { + } b _ { e } }\tag{15}
$$

A learnable class token $\mathbf { Z } _ { 0 }$ is attached and positional encodings $\mathrm { E _ { p o s } } { \in } \mathrm { R } ^ { ( \mathrm { P + 1 } ) \times \mathrm { d } }$ are attached:

$$
\mathrm { Z _ { 0 } \mathrm { = ( z _ { 0 } \mathrm { ; ~ z _ { 1 } \mathrm { ; ~ \ldots ; ~ z _ { P } ) + E _ { p o s } } } } }\tag{16}
$$

For each Transformer encoder layer, I use multi-head selfattention (MHSA) and pre-layer normalization.

$$
\mathrm { Z _ { l } ^ { \prime } = M H S A \big ( L N ( \mathrm { Z } _ { l - 1 } ) \big ) + Z _ { l - 1 } }\tag{17}
$$

$$
\mathrm { Z _ { l } = F F N \big ( L N ( Z _ { l } ^ { \prime } ) \big ) + Z _ { l } ^ { \prime } }\tag{18}
$$

The MHSA with H heads computes:

$$
\mathrm { M H S A ( Z ) } = \mathrm { C o n c a t ( h e a d _ { 1 } , \dots , h e a d _ { H } ) W ^ { 0 } }\tag{19}
$$

$$
\mathrm { { h e a d } _ { h } = \mathrm { S o f t m a x } ( \frac { Q _ { h } K _ { h } ^ { T } } { \sqrt { d _ { k } } } ) V _ { h } }\tag{20}
$$

where $\mathrm { { d } _ { k } { = } d / H }$ . The quadratic complexity of O(P²) with $\mathrm { P } = 8$ gives 64 pairwise interactions (81 with the CLS token). The Patch Transformer is based on the same channel mixer and patch tokenization as MambaPD, but has a single linear patch embedding network instead of the two-layer MLP, H = 4 attention heads and 2 layers (Table 6)

## 3.4 Proposed Architecture: MambaPD

MambaPD is a bidirectional selective state space model for Parkinson's disease detection. It has an architecture of stages shown in figure 3

A light weight channel mixing module first extracts the cross-electrode relationships and then does temporal modelling:

$$
\widetilde \mathrm { X } = \mathrm { L a y e r N o r m } ( \mathrm { W } _ { 2 } \mathrm { G E L U } ( \mathrm { W } _ { 1 } \mathrm { X } + \mathrm { b } _ { 1 } ) + \mathrm { b } _ { 2 } )\tag{21}
$$

These are linear layers which can operate across channels at each time point where $\mathbf { W } _ { 1 } { \in } \mathbf { R } ^ { 2 \mathbf { C } \times \mathbf { C } }$ $\mathrm { W } _ { 2 } \in \mathrm { R } ^ { \mathrm { C } \times 2 \mathrm { C } }$ and $\mathbf { X } { \in } \mathbf { R } ^ { \mathrm { { C } \times T } }$ Biases are shared across time and LayerNorm is performed across channels at each time step. The layers thus perform independent mixing of channels at each time step and do not perform mixing over time. This module allows the model to learn the PD discriminative spatial information before the temporal processing of the sequence.

The channel-mixed representation $\tilde { X } { \in } \mathrm { R } ^ { \mathrm { C } \times \mathrm { T } }$ is divided into P mutually disjoint temporal patches:

$$
\begin{array} { r } { \mathrm { p _ { i } = \widetilde { X } \left[ : , \ ( i - 1 ) L _ { p } : i L _ { p } \right] \in R ^ { C \times L _ { p } } , \quad \mathrm { i = 1 , \ldots , P } } } \end{array}\tag{22}
$$

where $\mathrm { L } _ { \mathrm { p } } { = } 1 2 8$ timesteps (0.5 seconds) and $\mathrm { P { = } T / L _ { p } { = } } 8$ . Half a second of multi-channel EEG activity captured for each patch, corresponding to the temporal duration of fast oscillatory events of interest for PD (alpha bursts, beta spindles) but short enough so that an EEG segment is locally stationary.

Each flattened patch flattens $( \mathsf { p } _ { \mathrm { i } } ) { \in } \mathsf { R } ^ { \mathrm { c . L } _ { \mathsf { P } } }$ is projected to embedding dimension d by a two-layer MLP:

$$
\mathbf { e _ { i } } = \mathrm { L a y e r N o r m } ( \mathrm { W _ { 4 } } \mathrm { D r o p o u t } ( \mathrm { G E L U } ( \mathrm { W _ { 3 } } \mathrm { f l a t t e n } ( \mathrm { p _ { i } } ) + \mathrm { b _ { 3 } } ) ) + \mathrm { b _ { 4 } } ) ( 2 3 )
$$

where $\mathrm { W } _ { 3 } { \in } \mathrm { R } ^ { 2 \mathrm { d } { \times } \mathrm { C } \mathrm { L } \mathrm { p } }$ and $\mathrm { W } _ { 4 } \mathrm { e R } ^ { \mathrm { d } \times 2 \mathrm { d } }$ . Learnable class token $\mathrm { e _ { 0 } } { \in } \mathrm { R } ^ { \mathrm { d } }$ is added in front and learnable positional encodings $\mathrm { E _ { p o s } } \in \mathrm { R } ( \mathrm { P } + 1 ) { \times } \mathrm { d }$ is added:

$$
\mathrm { E } { = } ( \mathrm { e } _ { 0 } ; \mathrm { e } _ { 1 } ; \ldots ; \mathrm { e } _ { \mathrm { P } } ) { + } \mathrm { E } _ { \mathrm { p o s } }\tag{24}
$$

Selective state space model (SSM) is the backbone of MambaPD. The continuous-time model is:

$$
\mathrm { h ^ { \prime } ( t ) } = \mathrm { A ( t ) } \mathrm { h ( t ) } + \mathrm { B ( t ) } \mathrm { x ( t ) }\tag{25}
$$

![](images/57aedb70134d79834023357276b9cf60ff362b3596c2b95ace382586bb13f871.jpg)  
Fig. 3. Proposed MambaPD architecture for bidirectional Mamba-based Parkinson's Disease detection from resting-state EEG.

$$
\mathbf { y } ( \mathrm { t } ) = \mathbf { C } ( \mathrm { t } ) \mathrm { h ( t ) } + \mathbf { D } \mathbf { x } ( \mathrm { t } )\tag{26}
$$

where $\mathbf { \boldsymbol { x } } ( \mathrm { t } ) \in \mathrm { R } ^ { \mathrm { d _ { i n n e r } } }$ , per channel the hidden state is $\mathrm { h ( t ) } \in$ $\mathtt { R } ^ { \mathtt { N } }$ (full state $\mathsf { h } ( \mathrm { t } ) \in \mathsf { R } ^ { \mathrm { d _ { i n n e r } \times N } } ) , \mathsf { B } \in \mathbb { R } ^ { \mathrm { N } \times 1 } , \mathsf { C } \in \mathbb { R } ^ { \mathrm { 1 } \times \mathrm { N } }$ and D is a skip connection scalar. In the selective formulation A is parameterized per channel as $\boldsymbol { \mathrm { A } } \in \mathrm { R } ^ { \mathrm { d _ { i n n e r } \times N } }$ over an expanded inner dimension ${ \mathrm { d } } _ { \mathrm { i n n e r } } = { \mathrm { E } } \cdot { \mathrm { d } } \left( { \mathrm { h e r e } } { \mathrm { E } } = 2 \right)$

The key innovation of Mamba over previous SSMs (e.g S4) is that B, C and the parameter Δ is a timescale which is all computed as a function of the current input, not being fixed learned constants:

$$
\Delta _ { \mathrm { t } } = \mathrm { s o f t p l u s } \big ( \mathrm { L i n e a r } _ { \Delta } ( \mathrm { x } _ { \mathrm { t } } ) \big ) \in \mathrm { R } ^ { d _ { i n n e r } }\tag{27}
$$

$$
\mathrm { B } _ { \mathrm { t } } = \mathrm { L i n e a r } _ { \mathrm { B } } ( \mathrm { x } _ { \mathrm { t } } ) \in \mathrm { R } ^ { \mathrm { N } }\tag{28}
$$

$$
\mathrm { C } _ { \mathrm { t } } = \mathrm { L i n e a r } _ { \mathrm { C } } ( \mathrm { x } _ { \mathrm { t } } ) \in \mathrm { R } ^ { \mathrm { N } }\tag{29}
$$

![](images/144544ac0069f292b5b77af3a9e72f7fefe1d5d98dcde8ede7c3c3a711d3921f.jpg)  
Fig. 4. Selective state space model (SSM) mechanism in MambaPD: input-dependent gating enables adaptive EEG memory management.

The model is input-dependent, meaning it dynamically allocates how much it can update its memory and what to read in and out of the memory, depending on the informativeness of the current EEG sample, diagnosed from it. A continuous-time system is sampled and used with a zero order hold (ZOH) method with step size Δt on the sampled EEG sequence:

$$
\overline { { \mathrm { A } _ { \mathrm { t } } } } = \exp ( \Delta _ { \mathrm { t } } \otimes \mathrm { A } )\tag{30}
$$

$$
\overline { { \mathrm { B _ { t } } } } = ( \Delta _ { \mathrm { t } } \mathrm { A } ) ^ { - 1 } ( \mathrm { e x p } ( \Delta _ { \mathrm { t } } \mathrm { A } ) - \mathrm { I } ) \Delta _ { \mathrm { t } } \mathrm { B _ { t } } \approx \Delta _ { \mathrm { t } } \otimes \mathrm { B _ { t } }\tag{31}
$$

Which gives the discrete recurrence:

$$
\begin{array} { r } { \mathbf { h } _ { \mathrm { t } } = \overline { { \mathrm { A } _ { \mathrm { t } } } } \odot \mathbf { h } _ { \mathrm { t - 1 } } + \overline { { \mathrm { B } _ { \mathrm { t } } } } \odot \mathbf { x } _ { \mathrm { t } } } \end{array}\tag{32}
$$

$$
\mathrm { y } _ { \mathrm { t } } = \mathrm { C } _ { \mathrm { t } } ^ { \mathrm { T } } \mathrm { h } _ { \mathrm { t } } + \mathrm { D } \mathrm { x } _ { \mathrm { t } }\tag{33}
$$

The bias of Δ projection is initialized to −2.0 and the recurrence is evaluated exactly (sequentially) over 9 tokens. The timescale parameter Δt is also used as the discretization step-size and forgetting rate. When the baseline EEG is uninformative, the hidden state should forget the past and a large Δt results in fast state decay. A small Δt retains the hidden state to be adequate in the case of a diagnostically meaningful event (alpha rhythm slowing, coherence disruption) that has taken place and whose time is to be preserved for later classification. In principle, this selective forgetting could let the model pass through uninformative patches while emphasizing spectral and coherence deviations; this is a motivation and is not tested here. Figure 4 shows the selective state space mechanism and the three input-dependent gates and their function in EEG based PD detection. Two separate stacks of Mamba blocks are used, processing the patch sequence E∈R<sup>(P+1)×d</sup> in opposite time directions.

The forward Mamba stack runs patches sequentially:

$$
\mathrm { H ^ { f } = M a m b a S t a c k _ { f w d } ( E ) }\tag{34}
$$

H<sub>i</sub><sup>f</sup> at i represents the context of patches $\scriptstyle { \mathrm { e } } _ { 0 }$ to $\mathrm { { \vec { e } } _ { i } }$ which is all the context that exists before patch i.The reversed sequence of patches is processed by the backward Mamba stack:

$$
\mathrm { H ^ { b } = f l i p \Big ( M a m b a S t a c k _ { b w d } \big ( f l i p ( E ) \big ) \Big ) }\tag{35}
$$

At position $\mathrm { i , H _ { i } ^ { . b } }$ encodes context from patches $\mathrm { { e _ { i } } }$ through $\mathrm { e _ { p } }$ the complete future temporal context from patch i onward.

Both the direction representations are concatenated and projected in a fusion layer:

$$
\mathrm { H _ { f s d } \mathrm { = L a y e r N o r m ( G E L U ( W _ { f } L a y e r n o r m ( [ H ^ { f } , H ^ { b } ] ) + b ^ { f } ) ) } }\tag{36}
$$

where $\mathrm { W } _ { \mathrm { f } } { \in } \mathrm { R } ^ { \mathrm { d } \times 2 \mathrm { d } }$ and [⋅;⋅] refers to concatenation in the feature dimension. For every patch position i, the model accesses the past context (forward Mamba) and future context (backward Mamba) simultaneously.

The class token output from the fused bidirectional representation serves as the global sequence summary:

$$
\mathbf { z } = \mathrm { H } _ { \mathrm { f s d 0 } } \in \mathrm { R } ^ { \mathrm { d } }
$$

This representation is passed through the classification head

$$
\hat { \bf { y } } = \mathrm { { S o f t m a x } \left( W _ { 2 } \ D r o p o u t _ { p / 2 } \left( G E L U \left( W _ { 1 } D r o p o u t _ { p } \ L a y e r N o r m ( z ) + b _ { 1 } \right) \right) + b _ { 2 } \right) }
$$

where p=0.2, W ∈R<sup>64×d</sup> & $\mathrm { W } _ { 2 } \mathrm { e R } ^ { 2 \times 6 4 }$ . The predicted class is:

$$
\hat { \mathbf { y } } = \mathop { \mathrm { a r g } \mathrm { m a x } } _ { \mathrm { c } } \widehat { \mathbf { y } _ { \mathrm { c } } } \in \{ 0 \mathrm { ( H e a l t h y ) , ~ } 1 \mathrm { ( P a r k i n s o n ' s ~ D i s e a s e ) } \}
$$

The CLS token is in position 0, thus its output in the forward stream is not providing any information about the EEG patches and, therefore, the output of the readout is effectively driven by the backward stream. With 9 tokens the linear time advantage of Mamba is not used here.

Table 5. MambaPD configuration used for each dataset.
<table><tr><td>Component</td><td>UNM</td><td>SanDiego</td></tr><tr><td>Input shape</td><td>(B, 63, 1024)</td><td>(B, 40, 1024)</td></tr><tr><td>Channel mixer dims</td><td>63 → 126 → 63</td><td>40 → 80 → 40</td></tr><tr><td>Patch size L p</td><td>128 (0.5 sec)</td><td>128 (0.5 sec)</td></tr><tr><td>Number of patches P</td><td>8</td><td>8</td></tr><tr><td>d_model</td><td>128</td><td>128</td></tr><tr><td>SSM state dim N</td><td>16</td><td>16</td></tr><tr><td>Mamba layers</td><td>2</td><td>2</td></tr><tr><td>Total Mamba layers</td><td>4 (2 fwd + 2 bwd)</td><td>4 (2 fwd + 2 bwd)</td></tr><tr><td>Dropout</td><td>0.2</td><td>0.2</td></tr><tr><td>Classifier hidden dim</td><td>64</td><td>64</td></tr><tr><td>Output classes</td><td>2</td><td>2</td></tr></table>

## 3.5 Training Configuration

To solve the class imbalance, weighted cross-entropy loss with label smoothing was used:

$$
\begin{array} { r } { \mathcal { L } = - \sum _ { c \in \{ 0 , 1 \} } w _ { c } \widetilde { y _ { c } } \log \widehat { y _ { c } } } \end{array}\tag{37}
$$

where $\begin{array} { r } { \widetilde { y _ { c } } = ( 1 - \varepsilon ) y _ { c } + \frac { \varepsilon } { 2 } } \end{array}$ is the smoothed target with smoothing factor $\scriptstyle \varepsilon = 0 . 1$ , and the class weight is:

$$
\begin{array} { r } { w _ { c } = \frac { N _ { \mathrm { { t o t a l } } } } { 2 \times N _ { c } } } \end{array}\tag{38}
$$

N<sub>c</sub> is the number of epochs of class c in the dataset (computed once from all labels, not per fold). This weight gives minority healthy class (label 0) a higher weight than the majority PD class (label 1). AdamW optimizer with $\lambda { = } 1 0 ^ { - 3 }$ , β<sub>1</sub>=0.9 and β<sub>2</sub>=0.999 (weight decay) were used in all models. The learning rate schedule was implemented in two phases, linear warmup for the first 10 epochs (W<sub>e</sub>) between η<sub>min</sub> and $\boldsymbol \eta _ { \mathrm { m a x } } ,$ and cosine annealing.

$$
\eta ( e ) = \left\{ \begin{array} { c c } { \eta _ { m i n } + ( \eta _ { m a x } - \eta _ { m i n } ) \frac { e } { W _ { e } } } & { e \leq W _ { e } } \\ { \eta _ { m i n } + \frac { \eta _ { m a x } - \eta _ { m i n } } { 2 } \bigg ( 1 + \cos \bigg ( \frac { \pi ( e - W _ { e } ) } { E - W _ { e } } \bigg ) \bigg ) } & { e > W _ { e } } \end{array} \right.
$$

Table 6. All models Hyperparameter configuration where E is the total number of training epochs. For MambaPD: η =5×10<sup>−4</sup>, η =10<sup>−6</sup>, W =10, E=100 (UNM) and E=80 (SanDiego). Gradient clipping with maximum norm 1.0 was applied at every update step.

## 3.6 Evaluation Metrics

Let TP, TN, FP, FN denote true positives, true negatives, false positives, and false negatives respectively. The six evaluation metrics are:

$$
\begin{array} { r } { \mathrm { A c c u r a c y } = \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } } \end{array}\tag{39}
$$

$$
\begin{array} { r } { \mathrm { { P r e c i s i o n } } = \frac { \mathrm { { T P } } } { \mathrm { { T P } + \mathrm { { F P } } } } } \end{array}\tag{40}
$$

$$
\begin{array} { r } { \mathrm { S e n s i t i v i t y } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } \end{array}\tag{41}
$$

$$
\begin{array} { r } { \mathrm { S p e c i f i c i t y } = \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } } \end{array}\tag{42}
$$

$$
\operatorname { F 1 } _ { \mathrm { c } } = { \frac { 2 \times { \mathrm { P r e c i s i o n } } \times { \mathrm { S e n s i t i v i t y } } } { \mathrm { P r e c i s i o n } + { \mathrm { S e n s i t i v i t y } } } }
$$

$$
\begin{array} { r } { \mathrm { F } _ { 1 } = \Sigma _ { \mathrm { c } } \left( \frac { \mathrm { N } _ { \mathrm { c } } } { \mathrm { N } } \right) \cdot \mathrm { F } 1 _ { \mathrm { c } } } \end{array}\tag{43}
$$

$$
\begin{array} { r } { \mathrm { A U C } = \int _ { 0 } ^ { 1 } \mathrm { T P R } \left( \mathrm { F P R } ^ { - 1 } ( t ) \right) d t } \end{array}\tag{44}
$$

$$
\begin{array} { r } { \kappa = \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } } \end{array}\tag{45}
$$

where ${ \mathfrak { p } } _ { 0 }$ is the observed agreement and $\begin{array} { r } { p _ { e } = \sum _ { c } p _ { c , \mathrm { p r e d } } \times } \end{array}$ $p _ { c , \mathrm { t r u e } }$ is the expected agreement by chance. All the metrics were calculated on the fold basis and then averaged over the 5 folds. The reported values are mean ± standard deviation from folds.

<table><tr><td>Hyperparameter</td><td>TCN</td><td>BiLSTM</td><td>EEGNet</td><td>Transformer</td><td>MambaPD</td></tr><tr><td>Epochs</td><td>100 (UNM)</td><td>100 (UNM)</td><td>100 (UNM)</td><td>100 (UNM)</td><td>100 (UNM)</td></tr><tr><td>Batch size</td><td>80 (SD) 64</td><td>80 (SD)</td><td>80 (SD)</td><td>80 (SD)</td><td>80 (SD)</td></tr><tr><td>Max LR</td><td></td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Min LR</td><td>1e-3</td><td>5e-4</td><td>1e-3</td><td>5e-4</td><td>5e-4</td></tr><tr><td>Warmup epochs</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td></tr><tr><td>Weight decay</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Dropout</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-3</td></tr><tr><td>Hidden dim</td><td>0.2</td><td>0.3</td><td>0.5</td><td>0.3</td><td>0.2</td></tr><tr><td>Depth</td><td>64 ch</td><td>128</td><td>F1=8, D=2</td><td>d=128</td><td>128</td></tr><tr><td>Label smoothing</td><td>6 blks 0.1</td><td>2 lyrs</td><td>F2=16</td><td>2 lyrs</td><td>2+2</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>LR schedule</td><td></td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Mixed precision</td><td>Warmup+Cosine</td><td>Warmup+Cosine</td><td>Warmup+Cosine</td><td>Warmup+Cosine</td><td>Warmup+Cosine</td></tr><tr><td>Grad clip norm</td><td>FP16 1.0</td><td>FP16 1.0</td><td>FP16 1.0</td><td>FP16 1.0</td><td>FP16 1.0</td></tr></table>

EEG Alpha Rhythm Pattern (PD biomarker: sustained slowing)  
![](images/7ccdcf32ecda2c60f698dc4b9f6c82ae161ca189ae9b4ebaeb2131b7f33446a6.jpg)

![](images/9b2bb0439fd0cb35eb6970b4066a5e851b22e2e06ad68cbd8ca07238ec8d4cef.jpg)  
Fig. 5. Schematic illustration of bidirectional mamba processing capturing bilateral temporal context of PD relevant EEG biomarkers

5-fold stratified cross validation at the subject level is used for all experiments. Subjects not individual epochs are partitioned into five disjoint folds:

$$
\begin{array} { r } { \mathrm { T r a i n } _ { k } = \{ E _ { s } \colon s \notin \mathrm { F o l d } _ { k } \} } \\ { \mathrm { V a l } _ { k } = \{ E _ { s } \colon s \in \mathrm { F o l d } _ { k } \} } \end{array}
$$

where $\mathrm { E _ { s } }$ represents all epochs for the s-th subject and Fold<sub>k</sub> represents all the subjects in the k-th validation fold. The PD/healthy subject ratio is approximately the same throughout folds due to stratification. Subject-level splitting does not include patient-specific EEG signatures that can easily be confused across training and testing sets, and helps to eliminate the possibility of significant over-optimistic accuracy estimates, unlike epoch-level splitting. Both sessions (ON and OFF) of a PD subject are in the same fold and the same folds were used for all models (seed 42).

## 4. EXPERIMENTAL SETUP

Data were preprocessed the same way with MNE-Python for both datasets. All PD-relevant frequency bands were preserved while all frequency bands of electrode drift and

muscle noise were suppressed by means of a zero-phase Hamming-windowed FIR bandpass filter (0.5–40 Hz). Both data sets have been down-sampled to a 256 Hz sampling rate. Recordings were divided into 4 second, 50% overlapped (step size 512) epochs, resulting in 8,882 epochs from UNM and 4,475 epochs from SanDiego. The epochs were z-scored for each channel. A patient-level separation was ensured by keeping the patient identity array throughout. The training of all models was done in 16-bit mixed precision on Kaggle on an NVIDIA Tesla T4 GPU. The AdamW optimizer was applied to the problem with weight decay value $1 0 ^ { - 3 }$ and $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9 .$ A linear warmup and cosine annealing learning rate schedule were used. Class imbalance was addressed using weighted cross-entropy loss with label smoothing (ε = 0.1), where $\mathrm { { N } } _ { \mathrm { { t o t a l } } }$ is the total number of epochs. Gradient clipping with maximum norm 1.0 was applied. Checkpoints were selected by validation AUC, the validation set was the held-out fold, so absolute numbers are optimistic. This applies to all models.

Evaluation Protocol Strict 5-fold stratified cross validation at the subject level was used to evaluate all models. No epoch of any test patient is ever used for training at any

Table 7. Experimental results for all five models on both datasets.
<table><tr><td>Model</td><td>Dataset</td><td>Acc</td><td>±std</td><td>F1</td><td>AUC</td><td>Kappa</td><td>Sens</td><td>Spec</td></tr><tr><td>TCN</td><td>UNM</td><td>0.7076</td><td>0.0969</td><td>0.7113</td><td>0.7639</td><td>0.3883</td><td>0.7439</td><td>0.6584</td></tr><tr><td>TCN</td><td>SanDiego</td><td>0.7029</td><td>0.0474</td><td>0.6957</td><td>0.6849</td><td>0.3114</td><td>0.8070</td><td>0.4918</td></tr><tr><td>BiLSTM+Attention</td><td>UNM</td><td>0.6991</td><td>0.0628</td><td>0.7065</td><td>0.7605</td><td>0.3673</td><td>0.6876</td><td>0.7063</td></tr><tr><td>BiLSTM+Attention</td><td>SanDiego</td><td>0.6669</td><td>0.0161</td><td>0.6540</td><td>0.7324</td><td>0.2529</td><td>0.7557</td><td>0.5072</td></tr><tr><td>EEGNet</td><td>UNM</td><td>0.6694</td><td>0.1032</td><td>0.6783</td><td>0.7501</td><td>0.3439</td><td>0.6276</td><td>0.7509</td></tr><tr><td>EEGNet</td><td>SanDiego</td><td>0.6007</td><td>0.1162</td><td>0.5985</td><td>0.6807</td><td>0.2136</td><td>0.5779</td><td>0.6445</td></tr><tr><td>Transformer</td><td>UNM</td><td>0.7146</td><td>0.0605</td><td>0.7139</td><td>0.7227</td><td>0.3492</td><td>0.7843</td><td>0.5601</td></tr><tr><td>Transformer</td><td>SanDiego</td><td>0.6737</td><td>0.0225</td><td>0.6656</td><td>0.7226</td><td>0.2611</td><td>0.7695</td><td>0.4941</td></tr><tr><td>MambaPD</td><td>UNM</td><td>0.7138</td><td>0.0793</td><td>0.7064</td><td>0.7647</td><td>0.3535</td><td>0.8244</td><td>0.5176</td></tr><tr><td>MambaPD</td><td>SanDiego</td><td>0.6310</td><td>0.0410</td><td>0.6305</td><td>0.6490</td><td>0.1762</td><td>0.7192</td><td>0.4611</td></tr></table>

time. For each fold six metrics were calculated and the averages computed across the folds: accuracy, weighted F1, AUC-ROC, Cohen’s Kappa, sensitivity, and specificity. All models share the optimizer, schedule, weight decay, label smoothing, batch size, mixed precision and checkpointselection protocol. Learning rate and dropout differ by model (Table 6) and were not tuned separately.

## 5. RESULTS

The complementary views are presented in Figures 6–11 and the mean cross-validated metrics for all five models are listed in Table 7 on both datasets. All metrics are computed on epochs pooled within each held-out fold and then averaged over the five folds and reported as mean ± std across held out folds. The classes are imbalanced, about two thirds of the epochs are PD (Table 4). If a classifier were to classify all epochs as PD, then its accuracy would be approximately 66%, sensitivity 1.0, and specificity 0. This figure is used as a guideline for accuracy.

## 5.1 Overall Classification Performance:

UNM. The mean accuracy of each model is between 66.9% (EEGNet) and 71.5% (Patch Transformer). MambaPD reaches 71.4% (±7.9%), ahead of TCN (70.8%), BiLSTM+Attention (69.9%) and EEGNet, and 0.08 points behind the Transformer. All 5 models are 0.6-5 points higher than the always-PD reference. MambaPD achieves the best AUC (0.765) (by 0.001), and the highest sensitivity (0.824) among all methods. Its F1 (0.706) and Kappa (0.354) are lower than TCN’s (0.711 and 0.388), and its specificity (0.518) is the lowest of the five models.

SanDiego. Mean accuracy ranges from 60.1% (EEGNet) to 70.3% (TCN). The best accuracy, F1 and Kappa scores are seen in the TCN model, followed by the Transformer model (67.4%) and the BiLSTM+Attention model (66.7%). MambaPD reaches 63.1% (±4.1%), ahead of EEGNet only. It has the lowest AUC (0.649), Kappa (0.176) and specificity (0.461), and its sensitivity (0.719) is fourth of five. The score for TCN increases by approximately 4 relative to the always-PD standard, while the Transformer and BiLSTM increase by around 1 each. MambaPD and EEGNet are lower than the always-PD standard. All models have weak agreement with Kappa values between 0.18 and 0.31 (with chance level at −0.02 for MambaPD's fold 2).

## 5.2 Statistical comparison

Table 7a provides paired comparisons of the accuracy per fold obtained from the MambaPD compared to the baseline networks (n = 5 folds, no multiple-comparison correction, paired t-test).

<table><tr><td colspan="6">Table 7a. Comparisons of the accuracy per fold</td></tr><tr><td>Comparison</td><td>UNM ∆acc</td><td>UNM wins</td><td>UNM p</td><td>SD ∆acc</td><td>SD SD wins p</td></tr><tr><td>MambaPD 一 TCN</td><td>+0.006</td><td>3/5</td><td>0.85</td><td>-0.072 1/5</td><td>0.11</td></tr><tr><td>MambaPD BiLSTM</td><td>+0.015</td><td>2/5</td><td>0.63</td><td>-0.036 1/5</td><td>0.10</td></tr><tr><td>MambaPD - EEGNet</td><td>+0.044</td><td>3/5</td><td>0.18</td><td>+0.030</td><td>3/5 0.69</td></tr><tr><td>MambaPD - Transformer</td><td>-0.001</td><td>3/5</td><td>0.98</td><td>-0.043 1/5</td><td>0.09</td></tr></table>

No comparison reaches p < 0.05. The standard deviation of the paired differences between MambaPD and the Transformer on UNM is 5.6 points, which means that for 5 folds, a difference of approximately 7 points would be required to obtain a significance level p < 0.05. The test is not particularly sensitive to the differences here (1–5 points) so that the lack of significance does not prove that the models are not equivalent. Note that it does not imply a ranking between them. On SanDiego, MambaPD's performance is inferior to TCN, BiLSTM+Attention and Transformer for four out of five folds (p = 0.09–0.11), a trend which does not achieve statistical significance on five folds.

## 5.3 Multi-metric comparison

All six metrics are displayed as both a radar chart and a connected line chart in Figure 7 and 9. On UNM, the five models are close on accuracy (0.67–0.71), F1 (0.68–0.71), AUC (0.72–0.76) and Kappa (0.34–0.39).

Figure 6: Accuracy and AUC-ROC Comparison  
![](images/36a625c89bdf71b395741f4e64fb20b6312676b3e54c9617e1a140041d92335c.jpg)

![](images/d75477ddda8af5651a7ccf06701f48958e12df85678a4c6a24dc0b2c638643ea.jpg)

![](images/e2e474bf40a2f72c58e50f9db16d07a4548c7df7544b9c963d50bffa12e1763d.jpg)  
Figure 7: Radar Chart — Multi-metric Comparison  
Note: the Cohen's Kappa axis is rescaled as (k + 1) / 2 to fit [0, 1]

Figure 8: Cross-validation Accuracy Distribution  
![](images/c97bf5aa272d68e102fd19a4a5588ebd774d8a388066fb15bace93497018130f.jpg)

![](images/e4a3de63ca302ec6aae03dc9ca942265dca374b9d467f664ef49d90ecc6c99f8.jpg)

Figure 9: All Metrics Across Models  
![](images/a370d84c494dcb416663c4519202b7a9e604576017d0d20bd7c338359e010d59.jpg)

The main differences are in the sensitivity-specificity balance: MambaPD (0.824/0.518) and the Transformer (0.784/0.560) favour sensitivity, EEGNet (0.628/0.751) and BiLSTM+Attention (0.688/0.706) favour specificity, and TCN (0.744/0.658) is the most balanced. All polygons are shrunk on SanDiego. TCN is highest in accuracy, F1 and Kappa, and EEGNet is highest in specificity while MambaPD is smallest on AUC, Kappa and specificity, and fourth on sensitivity. The metrics thus concur that SanDiego

![](images/33811f32336d026dcd29a25803747af185addd517b46ad451a543a2926ee0691.jpg)  
is the more challenging set, and they disagree on which model is best on it, based on which metric is emphasized.

## 5.4 Fold-level behavior and stability

The accuracy per fold is displayed in Figures 8 and 11. On UNM, fold accuracy varies widely for every model (MambaPD 0.599–0.847, TCN 0.585–0.829, EEGNet 0.507–0.798, BiLSTM+Attention 0.574–0.738, Transformer 0.655–0.802).

![](images/3da7f50ffb550a6f76c8afbc63b1006284583c68a0d62f697af903b9a332be42.jpg)

![](images/407b6bbf5157eee64d84392dbe31271df0b2187bb67b03e78e83be8a707aeab4.jpg)  
Figure 11: Fold Accuracy Distribution (Violin + Strip)

![](images/8890bdcb05ca24abfb1123a6d738f9935113e445c5f4b52f10aeb4e9bf3a0f20.jpg)  
The five model means are no more than 4.5 points apart while the fold means (averaged over the five models) are as much as 21.6 points apart, with averaged fold accuracy of 0.585 (fold 1) to 0.801 (fold 3). Per-fold accuracy of different models are highly correlated (mean pairwise Pearson r = 0.76, indicative with five folds) and fold 1 is the hardest for four of five models and fold 3 the easiest for four of five. This means that the variation in the test subjects is the source of a greater amount of the variance than the model type. The fold means on SanDiego are more uniform (0.624–0.690) and there exists no consistent trend across the models (mean pairwise r ≈ 0.03). BiLSTM+Attention (std 0.016) and the Transformer (0.023) are stable, MambaPD (0.041) and TCN (0.047) are moderately variable, and EEGNet (0.116) is unstable dropping to 0.393 on one fold. The smallest fold of MambaPD is 0.551. The standard deviations are based on 6-7 test subjects per fold, meaning that the accuracy of any of the folds could move significantly when a one test subject is added or removed. Per-fold accuracies are listed in Table 8

## 5.5 Sensitivity-Specificity Trade-off

Sensitivity and specificity are displayed side by side in Figure 10; for deployment, sensitivity is an indicator of missed PD cases while specificity is an indicator of false alarms in healthy individuals. MambaPD has the lowest specificity (0.518) and the highest sensitivity (0.824) on UNM. It has the fourth highest sensitivity (0.719) on san diego out of the five (behind TCN 0.807, the Transformer 0.770, and BiLSTM 0.756) and the lowest specificity of the five. This profile might suit a screening setting where

![](images/8e985aed86957d29705fe4e291e8cde05b1dbae0ad394b0d56fcf223ae9cd02a.jpg)  
missed cases are costly but is not appropriate for confirmatory testing on UNM and is not sensitive or specific enough on SanDiego.

## 5.6 Differences between the two datasets

All five models have lower Kappa on SanDiego than on UNM and four of five have lower AUC (the Transformer is unchanged at 0.723). There are differences between the two datasets in terms of cohort size, channel count used, recording devices and number of test subjects per fold and these differences co-vary, thus I cannot argue that this difference is due to any single factor. Note that no model was trained using one dataset and tested using the other: thus the results are not a measure of cross-site generalization. The data-set also influences the ranking (TCN is highest for SanDiego but not for UNM). The biggest reduction in AUC (0.116) and Kappa (0.177) is observed in MambaPD.

## 5.7 Limitations

Absolute estimates are optimistic as AUC selected the checkpoints on the held-out fold for all models. The reported results were obtained once for each fold of each model; a second run of MambaPD on SanDiego with identical code gave 63.1% against 65.3% in the first run, so small differences between models should not be interpreted. Both datasets are small and test folds contain only 6–12 subjects. The hyperparameters were not tuned individually for each baseline; and the design choices for MambaPD were optimized on the same folds on which it was evaluated.

<table><tr><td colspan="8">Table 8. Per Fold results</td></tr><tr><td>Dataset</td><td>Model</td><td>Fold 1</td><td>Fold 2</td><td>Fold 3</td><td>Fold 4</td><td>Fold 5</td><td>Mean</td></tr><tr><td>UNM</td><td>TCN</td><td>0.585</td><td>0.613</td><td>0.829</td><td>0.799</td><td>0.712</td><td>0.708</td></tr><tr><td>UNM</td><td>BiLSTM+Attn</td><td>0.574</td><td>0.738</td><td>0.728</td><td>0.721</td><td>0.734</td><td>0.699</td></tr><tr><td>UNM</td><td>EEGNet</td><td>0.507</td><td>0.603</td><td>0.798</td><td>0.742</td><td>0.697</td><td>0.669</td></tr><tr><td>UNM</td><td>Transformer</td><td>0.657</td><td>0.655</td><td>0.802</td><td>0.771</td><td>0.688</td><td>0.715</td></tr><tr><td>UNM</td><td>MambaPD</td><td>0.599</td><td>0.718</td><td>0.847</td><td>0.713</td><td>0.692</td><td>0.714</td></tr><tr><td>SanDiego</td><td>TCN</td><td>0.711</td><td>0.727</td><td>0.617</td><td>0.760</td><td>0.699</td><td>0.703</td></tr><tr><td>SanDiego</td><td>BiLSTM+Attn</td><td>0.675</td><td>0.642</td><td>0.670</td><td>0.658</td><td>0.690</td><td>0.667</td></tr><tr><td>SanDiego</td><td>EEGNet</td><td>0.614</td><td>0.680</td><td>0.582</td><td>0.393</td><td>0.734</td><td>0.601</td></tr><tr><td>SanDiego</td><td>Transformer</td><td>0.690</td><td>0.651</td><td>0.697</td><td>0.642</td><td>0.689</td><td>0.674</td></tr><tr><td>SanDiego</td><td>MambaPD</td><td>0.648</td><td>0.551</td><td>0.657</td><td>0.664</td><td>0.635</td><td>0.631</td></tr></table>

Training and testing were performed separately for each dataset, there were different numbers of channels and cohorts in the datasets, and there was a different type of hardware used to record each dataset; therefore, differences from one dataset to another are not necessarily due to any single factor. The metrics are calculated on an epoch level and the decisions are taken based on a threshold of 0.5. No artifact rejection or re-referencing was applied, and perepoch z-scoring discards absolute amplitude and band power. Medication state (ON/OFF) was not modelled. The CLS token is placed at position 0, and it has no patch data on its forward stream; thus, the patch data is effectively being driven by the backward stream. I did not make a controlled ablation and the linear-time advantage of Mamba was not used with only 9 tokens. The accuracy of approximately 60–72% is not as high as the accuracy reported on four pooled datasets for TransformEEG [26] under the nested leave-subjects-out validation (median 78.45%; 80.1% with augmentation and threshold correction), and there are differences in protocols.

## CONCLUSION

I introduced the bidirectional selective state space model for resting-state EEG PD detection (MambaPD) and compared to four baselines on two public datasets, with strict crossvalidation, subject independent. MambaPD and the patch Transformer and TCN are similar on UNM (around 71%), with differences to be seen in sensitivity and specificity. On SanDiego, TCN performs best, and MambaPD is below TCN, the Transformer and BiLSTM, with the lowest specificity. The absolute accuracy is much lower than that reported with epoch level splitting and the spread across folds is larger than the spread across models on UNM. I report the pipeline and per-fold results to allow for a more stringent comparison, and I suggest multi-site validation and repeated runs, as well as nested model selection and threshold calibration in future work.

## REFERENCES

[1] Dorsey E, Elbaz A, Nichols E et al. Global, regional, and national burden of Parkinson's disease, 1990–2016: a systematic analysis for the Global Burden of Disease Study 2016 The Lancet Neurology, 2018; 17, 939-953

[2] Tysnes OB, Storstein A. Epidemiology of Parkinson's disease. J Neural Transm (Vienna). 2017 Aug;124(8):901-905. doi: 10.1007/s00702-017- 1686-y. Epub 2017 Feb 1. PMID: 28150045.

[3] Lang AE, Lozano AM. Parkinson's disease. First of two parts. N Engl J Med. 1998 Oct 8;339(15):1044-53. doi: 10.1056/NEJM199810083391506. PMID: 9761807.

[4] Dauer W, Przedborski S. Parkinson's disease: mechanisms and models. Neuron. 2003 Sep 11;39(6):889-909. doi: 10.1016/s0896-6273(03)00568-3. PMID: 12971891.

[5] Hughes AJ, Daniel SE, Kilford L, Lees AJ. Accuracy of clinical diagnosis of idiopathic Parkinson's disease: a clinico-pathological study of 100 cases. J Neurol Neurosurg Psychiatry. 1992 Mar;55(3):181-4. doi: 10.1136/jnnp.55.3.181. PMID: 1564476; PMCID: PMC1014720.

[6] Chaudhuri KR, Schapira AH. Non-motor symptoms of Parkinson's disease: dopaminergic pathophysiology and treatment. Lancet Neurol. 2009 May;8(5):464-74. doi: 10.1016/S1474-4422(09)70068-7. PMID: 19375664.

[7] Brundin P, Melki R, Kopito R. Prion-like transmission of protein aggregates in neurodegenerative diseases. Nat Rev Mol Cell Biol. 2010 Apr;11(4):301-7. doi: 10.1038/nrm2873. PMID: 20308987; PMCID: PMC2892479.

[8] Anjum MF, Espinoza AI, Cole RC, Singh A, May P, Uc EY, Dasgupta S, Narayanan NS. Resting-state EEG measures cognitive impairment in Parkinson's disease. NPJ Parkinsons Dis. 2024 Jan 3;10(1):6. doi: 10.1038/s41531-023-00602-0. PMID: 38172519; PMCID: PMC10764756.

[9] Soikkeli R, Partanen J, Soininen H, Pääkkönen A, Riekkinen P Sr. Slowing of EEG in Parkinson's disease. Electroencephalogr Clin Neurophysiol. 1991 Sep;79(3):159-65. doi: 10.1016/0013-4694(91)90134- p. PMID: 1714807

[10] Yuvaraj R, Murugappan M. Hemispheric asymmetry non-linear analysis of EEG during emotional responses from idiopathic Parkinson's disease patients. Cogn Neurodyn. 2016 Jun;10(3):225-34. doi: 10.1007/s11571-016-9375-3. Epub 2016 Jan 28. PMID: 27275378; PMCID: PMC4870407.

[11] Betrouni, N., Delval, A., Chaton, L., Defebvre, L., Duits, A., Moonen, A., Leentjens, A., & Dujardin, K. (2019). Electroencephalography‐based machine learning for cognitive profiling in Parkinson's disease: Preliminary results. Movement Disorders, 34(2), 210–217. https://doi.org/10.1002/mds.27528

[12] Palanivel, M., Ghosh, K.K., Mallam, M. et al. Imaging advances to detect non-motor prodromal markers of Parkinson’s disease and explore

therapeutic translation opportunities. npj Parkinsons Dis. 11, 174 (2025). https://doi.org/10.1038/s41531-025-01004-0

[13] Lensky, A. (2025). Oscillatory Signatures of Parkinson's Disease: Central and Parietal EEG Alterations Across Multiple Frequency Bands. ArXiv, abs/2503.12392.

[14] Aljalal M, Aldosari SA, AlSharabi K, Abdurraqeeb AM, Alturki FA. Parkinson's Disease Detection from Resting-State EEG Signals Using Common Spatial Pattern, Entropy, and Machine Learning Techniques. Diagnostics (Basel). 2022 Apr 20;12(5):1033. doi: 10.3390/diagnostics12051033. PMID: 35626189; PMCID: PMC9139946.

[15] Shaban M, Amara AW. Resting-state electroencephalography based deep-learning for the detection of Parkinson's disease. PLoS One. 2022 Feb 24;17(2):e0263159. doi: 10.1371/journal.pone.0263159. PMID: 35202420; PMCID: PMC8870584.

[16] Obayya M, Saeed MK, Maashi M, Alotaibi SS, Salama AS, Ahmed Hamza M. A novel automated Parkinson's disease identification approach using deep learning and EEG. PeerJ Comput Sci. 2023 Nov 22;9:e1663. doi: 10.7717/peerj-cs.1663. PMID: 38077610; PMCID: PMC10703017.

[17] Jibon FA, Tasbir A, Talukder MA, Uddin MA, Rabbi F, Uddin MS, Alanazi FK, Kazi M. Parkinson's disease detection from EEG signal employing autoencoder and RBFNN-based hybrid deep learning framework utilizing power spectral density. Digit Health. 2024 Nov 12;10:20552076241297355. doi: 10.1177/20552076241297355. PMID: 39539721; PMCID: PMC11558743.

[18] Bunterngchit, C., Baniata, L. H., Albayati, H., Baniata, M. H., Alharbi, K., Alshammari, F. H., & Kang, S. (2025). A Hybrid Convolutional– Transformer Approach for Accurate Electroencephalography (EEG)-Based Parkinson’s Disease Detection. Bioengineering, 12(6), 583.

[19] Valeriana Mancazzo, Elena Sibilano, Domenico Buongiorno, Raffaele Carli, Vitoantonio Bevilacqua, Antonio Brunetti,, Parkinson’s disease detection from resting-state EEG using a Transformer model with Multi-Head Attention explanations, Biomedical Signal Processing and Control, Volume 119, Part A, 2026,109811,ISSN 1746-8094, doi: https://doi.org/10.1016/j.bspc.2026.109811.

[20] Huggins, C. J., Escudero, J., Parra, M. A., Scally, B., Anghinah, R., Vitória Lacerda De Araújo, A., Basile, L. F., & Abasolo, D. (2021). Deep learning of resting-state electroencephalogram signals for three-class classification of Alzheimer's disease, mild cognitive impairment and healthy ageing. Journal of Neural Engineering, 18(4), Article 046087.

[21] Najafi, S., & Mazinani, M. (2026). 3DCNN-SL framework for the diagnosis of Parkinson's disease using frequency-preserving 3D EEG tensors. Computers in Biology and Medicine, 210, Article 111716.

[22] Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-Term Memory. Neural Computation. 9. 1735-1780. 10.1162/neco.1997.9.8.1735.

[23] Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia. (2017). Attention Is All You Need. 10.48550/arXiv.1706.03762.

[24] Siddiqui F, Mohammad A, Alam MA, Naaz S, Agarwal P, Sohail SS, Madsen DØ. Deep Neural Network for EEG Signal-Based Subject-Independent Imaginary Mental Task Classification. Diagnostics (Basel). 2023 Feb 9;13(4):640. doi: 10.3390/diagnostics13040640. PMID: 36832128; PMCID: PMC9955721.

[25] A. Gu and T. Dao, "Mamba: linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023.

[26] Del Pup, F. Federico & Brun, Riccardo & Iotti, Filippo & Paccagnella, Edoardo & Pezzato, Mattia & Bertozzo, Sabrina & Zanola, Andrea & Tshimanga, Louis Fabrice & Müller, Henning & Atzori, Manfredo. (2025). TransformEEG: Towards Improving Model Generalizability in Deep Learning-based EEG Parkinson's Disease Detection. 10.48550/arXiv.2507.07622.

[27] Lal, U., Chikkankod, A.V. & Longo, L. Fractal dimensions and machine learning for detection of Parkinson’s disease in resting-state electroencephalography. Neural Comput & Applic 36, 8257–8280 (2024). https://doi.org/10.1007/s00521-024-09521-4

[28] Latifoglu, Fatma & Penekli, Sultan & Orhan Bulucu, Fırat. (2024). A novel approach for Parkinson’s disease detection using Vold-Kalman order filtering and machine learning algorithms. Neural Computing and Applications. 36. 1-15. 10.1007/s00521-024-09569-2.

[29] Bera, S., Geem, Z. W., Cho, Y.-I., & Singh, P. K. (2025). A Comparative Study of Machine Learning and Deep Learning Models for Automatic Parkinson’s Disease Detection from Electroencephalogram Signals. Diagnostics, 15(6), 773. https://doi.org/10.3390/diagnostics15060773

[30] Sahota, Amarpal & Roguski, Amber & Jones, Matthew & Abdallah, Zahraa & Santos-Rodriguez, Raul. (2024). Investigating Brain Connectivity and Regional Statistics from EEG for early stage Parkinson's Classification. 10.48550/arXiv.2408.00711.

[31] Avola, Danilo & Bernardini, Andrea & Crocetti, Giancarlo & Ladogana, Andrea & Lezoche, Mario & Mancini, Maurizio & Pannone, Daniele & Ranaldi, Amedeo. (2025). Benchmarking of EEG Analysis Techniques for Parkinson's Disease Diagnosis: A Comparison between Traditional ML Methods and Foundation DL Methods. 10.48550/arXiv.2507.13716.

[32] Allahbakhshi, M., Sadri, A., & Shahdi, S. O. (2024). Diagnosis of Parkinson's disease using EEG signals and machine learning techniques: A comprehensive study. arXiv. https://doi.org/10.48550/arXiv.2405.00741

[33] Sahota, A., Roguski, A., Jones, M. W., Rolinski, M., Whone, A., Santos-Rodriguez, R., & Abdallah, Z. S. (2024). Interpretable Classification of Early Stage Parkinson’s Disease from EEG. In A. Shaban-Nejad, M. Michalowski, & S. Bianco (Eds.), AI for Health Equity and Fairness: Leveraging AI to Address Social Determinants of Health (pp. 219-231). (Studies in Computational Intelligence; Vol. 1164 SCI). Springer Science and Business Media Deutschland GmbH. Advance online publication. https://doi.org/10.1007/978-3-031-63592-2\_17

[34] Siuly, S., Khare, S. K., Kabir, E., Sadiq, M. T., & Wang, H. (2024). An efficient Parkinson's disease detection framework: Leveraging timefrequency representation and AlexNet convolutional neural network. Computers in Biology and Medicine, 174, Article 108462. https://doi.org/10.1016/j.compbiomed.2024.108462

[35] Jibon FA, Tasbir A, Talukder MA, Uddin MA, Rabbi F, Uddin MS, Alanazi FK, Kazi M. Parkinson's disease detection from EEG signal employing autoencoder and RBFNN-based hybrid deep learning framework utilizing power spectral density. Digit Health. 2024 Nov 12;10:20552076241297355. doi: 10.1177/20552076241297355. PMID: 39539721; PMCID: PMC11558743.

[36] Li K, Ao B, Wu X, Wen Q, Ul Haq E, Yin J. Parkinson's disease detection and classification using EEG based on deep CNN-LSTM model. Biotechnol Genet Eng Rev. 2024 Nov;40(3):2577-2596. doi: 10.1080/02648725.2023.2200333. Epub 2023 Apr 11. PMID: 37039259.

[37] Bunterngchit, C., Baniata, L. H., Albayati, H., Baniata, M. H., Alharbi, K., Alshammari, F. H., & Kang, S. (2025). A Hybrid Convolutional– Transformer Approach for Accurate Electroencephalography (EEG)-Based Parkinson’s Disease Detection. Bioengineering, 12(6), 583. https://doi.org/10.3390/bioengineering12060583

[38] Zhang, Qian & Zhang, Ruilin & Zhu, Biaokai & Han, Xun & Xiao, Jun & Liu, Yifan & Wang, Zhe. (2025). GEPD:GAN-Enhanced Generalizable Model for EEG-Based Detection of Parkinson's Disease. 10.48550/arXiv.2508.14074.

[39] Qian, J., Goh, T. L., Xie, B., Zhu, C., Wan, B., Guan, Y., Chen, R. D., & Chiang, P. Y. (2025). BioMamba: Leveraging spectro-temporal embedding in bidirectional Mamba for enhanced biosignal classification. arXiv. https://doi.org/10.48550/arXiv.2503.11741

[40] Neves, C., Steele, C., & Xiao, Y. (2026). EEG-ChiMamba: Towards a robust Mamba-based architecture for dementia detection from resting state electroencephalography. bioRxiv. https://doi.org/10.64898/2026.03.23.713697

[41] Dhavapandiammal, A. & Palanisamy, Kalavathi. (2026). An EEG Signals-based Deep Spatio-Temporal Feature Extraction with Dual Attention for Detecting Parkinson’s Disease. 10.21203/rs.3.rs-9318311/v1.

[42] Gui, Y., Chen, M., Su, Y., Luo, G., & Yang, Y. (2024). EEGMamba: Bidirectional state space model with mixture of experts for EEG multi-task classification. arXiv. https://doi.org/10.48550/arXiv.2407.20254

[43] Tegon, A., Ingolfsson, T. M., Wang, X., Benini, L., & Li, Y. (2025). FEMBA: Efficient and scalable EEG analysis with a bidirectional Mamba foundation model. arXiv. https://doi.org/10.48550/arXiv.2502.06438

[44] Saeed F, Aldera S. Explainable EEG-based machine learning for early diagnosis of Alzheimer's disease and frontotemporal dementia. Sci Rep. 2026 Jun 16;16(1):19775. doi: 10.1038/s41598-026-57069-1. PMID: 42297902; PMCID: PMC13315799.

[45] Guo, M., Han, X., Liu, H., Zhu, J., Zhang, J., Bai, Y., & Ni, G. (2025). MI-Mamba: A hybrid motor imagery electroencephalograph classification model with Mamba's global scanning. Ann NY Acad Sci., 1544, 242–253. https://doi.org/10.1111/nyas.15288

[46] Behrouz, A. & amp; Hashemi, F. (2024). Brain-Mamba: Encoding Brain Activity via Selective State Space Models. Proceedings of the fifth Conference on Health, Inference, and Learning, Proceedings of Machine Learning Research 248:233-250 Available from https://proceedings.mlr.press/v248/behrouz24a.html

[47] Alexander P. Rockhill, Nicko Jackson, Jobi George, Adam Aron, and Nicole C. Swann (2021). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro. [Dataset] doi:10.18112/openneuro.ds002778.v1.0.5