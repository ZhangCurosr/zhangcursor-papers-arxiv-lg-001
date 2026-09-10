# Deep Neural Networks for Learning Intent from sEMG Signals to Support Hardware Devices for Post-Stroke Neurorehabilitation

Zakariyya Brewster<sup>∗</sup>   
Department of Engineering Science   
University of Toronto   
Emily Yan   
Department of Computer Science   
University of Toronto   
Karma Namgyal   
Department of Electrical & Computer Engineering   
University of Toronto   
Markiyan Konyk   
Department of Engineering Science   
University of Toronto   
Divy Wadhwani   
Computer and Mathematical Sciences   
University of Toronto Scarborough   
Aidan Wang   
Computer Science and Cognitive Science   
University of Toronto   
Shuting Xie   
Department of Computer Science   
University of Toronto   
Tala Abdelmaguid   
Department of Computer Engineering   
University of Toronto

September 2026

## Abstract

Finger-specific motor intent is a clinically meaningful control signal for post-stroke neurorehabilitation, where residual muscle activity may remain measurable despite weak or incomplete movement. We study five-finger multilabel intent decoding from impaired-arm high-density surface electromyography (sEMG) in PhysioMio, a bilateral longitudinal dataset collected from stroke patients [10]. A common processing protocol aligns movement labels, applies 20–450 Hz Butterworth filtering and Symlet-4 wavelet denoising, segments overlapping 200 ms windows, and extracts twelve time- and frequency-domain descriptors per channel. Direct LSTM, CNN, and GNN baselines reveal complementary behavior: the LSTM attains the highest subset accuracy (0.545), whereas the GNN attains the highest macro F1 (0.706) and macro AUPRC (0.776). Architecture search then identifies CNN-Large as the strongest single-split CNN, with 0.593 subset accuracy and 0.714 macro F1, while CNN-Micro provides a compact architecture for embedded inference. To match a four-sensor hardware design, we retrain CNN-Micro using channels associated with ECRB, ECRL, FDS, and FDP and exclude the ground electrode from model input. Across five seeds, cross-channel knowledge distillation improves the four-channel student over direct training, reaching 0.5219 ± 0.0114 subset accuracy, 0.7612 ± 0.0038 finger accuracy, and 0.6095 ± 0.0058 macro F1. The selected 123K-parameter model accepts nine windows of 48 features and has been exported to ONNX. These results establish a reproducible software path from post-stroke sEMG to compact five-finger intent prediction for subsequent hardware-in-the-loop evaluation.

## 1 Introduction

Stroke frequently disrupts the descending motor pathways needed for coordinated hand opening, grasping, and individuated finger control. For patients with residual forearm muscle activity, surface electromyography (sEMG) ofers a non-invasive sensing modality through which intended movement can be estimated even when visible motion is weak or incomplete [15, 21, 5]. Reliable finger-intent decoding is therefore a central software problem for rehabilitation devices that aim to provide timely assistance, feedback, or task-specific practice.

The decoding problem is dificult because post-stroke sEMG difers substantially from healthy-limb gesture-recognition benchmarks. Paretic recordings are afected by abnormal co-contraction, altered recruitment patterns, fatigue, sensor placement, and impairment severity. In addition, the intended output for rehabilitation is often not a single gesture class but a multi-finger activation pattern suitable for controlling or cueing assistive hardware. A useful model must therefore be evaluated together with the signal-processing and labeling decisions that define the task.

This paper studies that problem as a hardware-aware machine-learning question: how can impairedarm HD-sEMG be transformed into five-finger intent estimates with models small enough to support responsive Raspberry Pi 5 rehabilitation hardware? The study evaluates the full decoding chain from label alignment and signal processing through model-family comparison, transfer learning, and compact CNN selection. This framing is deliberately tied to embedded rehabilitation use, because a clinically useful decoder must balance prediction quality with model size, latency, and deployability.

Our study is organized around three questions. First, when LSTM, CNN, and GNN decoders are trained on the same PhysioMio-derived feature representation, which inductive biases are most useful for multilabel finger-intent prediction? Second, after the baseline comparison, how much can decoding performance be improved through compact CNN architecture search, ResNet teacher references, and healthy-to-impaired transfer learning? Third, how much predictive performance is retained when the high-density representation is reduced to four active channels compatible with the planned sensing hardware, and can cross-channel knowledge distillation narrow that loss?

The central contribution is the formulation and evaluation of post-stroke HD-sEMG finger-intent decoding as a single hardware-aware research problem. We define a reproducible sEMG-to-finger intent workflow for PhysioMio recordings, including label alignment, filtering, denoising, windowed feature extraction, patient-level splits, and shared tensor adaptation. We compare LSTM, CNN, and GNN predictors under this common formulation and report both aggregate and finger-wise multilabel metrics. We evaluate an optimized CNN student family against internal ResNet teachers and quantify healthy-to-impaired transfer behavior. Finally, we construct and evaluate a fouractive-channel CNN-Micro through direct training, full-to-reduced-channel transfer initialization, and cross-channel knowledge distillation over five random seeds. This last experiment supplies a validated model and explicit input contract for subsequent Raspberry Pi 5 integration while leaving hardware timing and therapeutic evaluation to dedicated studies.

## 2 Related Work

sEMG has long been studied as a non-invasive control channel for rehabilitation, prosthetics, and human-machine interaction, and stroke-specific reviews show that sEMG-driven interventions can support meaningful upper-limb rehabilitation when signal interpretation is reliable enough for assistive use [15, 21]. Broader reviews of wearable sensing for stroke recovery make a related point: clinical value depends on the full stack, including acquisition quality, body placement, labeling protocol, feature representation, and the way machine learning is coupled to assessment or assistive control [5]. For this reason, decoding architecture should not be discussed independently of data construction and deployment context.

Dataset choice is especially important in this area. Ninapro remains the dominant public benchmark family for hand-gesture decoding from sEMG, with ten sub-datasets spanning diferent electrode layouts, movement vocabularies, and sensor modalities [3, 17]. The widely used DB5 split, for example, records 52 movements from 10 healthy participants with two Myo armbands, while DB1 and DB2 cover larger healthy cohorts with diferent channel configurations and richer kinematic metadata [3, 17]. By contrast, PhysioMio was introduced in 2026 as a stroke-specific bilateral and longitudinal HD-sEMG resource with 48 stroke patients, 16 gestures, 64 electrodes, and repeated recordings across inpatient recovery [10]. That distinction matters because results on healthysubject multiclass gesture sets are not automatically transferable to impaired-arm intent decoding in post-stroke populations.

Stroke-specific decoding studies have historically been small and heterogeneous. Lee et al. [12] used subject-specific myoelectric pattern classification for six functional hand movements in chronic stroke survivors and reported a sharp impairment-dependent gap, with mean accuracy of 71.3% for moderately impaired participants and 37.9% for severely impaired participants. Anastasiev et al. [2] later studied post-acute stroke gesture recognition with portable eight-channel sEMG and found that afected-side SVM performance could still reach the high-80% range on reduced gesture sets, but the task remained sensitive to feature design, label count, and paretic signal quality. More recent deep-learning work has begun to explore richer feature domains and architectures for post-stroke myoelectric recognition; Bao et al. [4] evaluated CNN, CNN-LSTM, and CNN-LSTMattention designs across time, frequency, and wavelet features in chronic stroke participants, with the best reported intra-subject and inter-subject transfer settings reaching 72.95% and 68.38% average accuracy, respectively. These studies underscore a recurring challenge for post-stroke intent decoding: strong performance is possible, but dataset scale, subject condition, movement vocabulary, and train-test protocol strongly determine the meaning of any reported number.

Deep neural models broaden that picture but do not remove the comparability problem. Sequenceoriented recurrent models remain natural baselines because they aggregate temporal evidence across windows, while compact 1D CNNs emphasize local temporal motifs and can be compressed aggressively for eficient inference [9]. Graph neural networks ofer a complementary structural view by representing windows as related observations connected through message passing rather than as a purely linear sequence [11, 7, 19]. On healthy-dataset benchmarks such as Ninapro, deeper end-to-end networks can reach very high multiclass accuracies; for example, Sri-Iesaranusorn et al. [18] reported 93.87% accuracy on Ninapro DB5 and 91.69% on DB7 for 41-movement classification. Other eficient CNN designs on Ninapro DB5 report high-80% accuracy while explicitly optimizing computation or channel use [16]. Those results are useful context for model capacity, but they address a diferent population and a diferent task than impaired-arm multilabel finger-intent prediction.

Transfer learning has recently become one of the most relevant directions for rehabilitation-oriented sEMG. Li et al. [13] showed that transfer-based CNNs can materially improve inter-subject and interday robustness on high-density healthy-subject sEMG, and Mohammadiazni et al. [14] demonstrated an especially important stroke-specific result: pretraining on healthy Ninapro DB5 data raised stroke-survivor gesture-classification accuracy from 46% to 93.6% in a three-gesture setting. These findings motivate healthy-to-impaired transfer as a natural strategy for paretic-limb decoding, where afected-side training data are comparatively scarce and variable.

Deployment-oriented optimization provides the final piece of context for this manuscript. Knowledge distillation is a standard way to transfer behavior from larger teachers into smaller students [8], and Optuna is widely used for black-box hyperparameter search under practical constraints [1].

For rehabilitation wearables, these methods connect ofline decoding accuracy to the practical requirements of model size, latency, and hardware-constrained inference.

Table 1: Literature context for the current study. Reported numbers reflect diferent populations, sensors, gesture vocabularies, and label spaces.
<table><tr><td>Source</td><td>Population / dataset</td><td>Task</td><td>Reported performance</td><td>Relevance to this paper</td></tr><tr><td>Lee et al. 2010</td><td>20 chronic stroke sur- vivors; 10 forearm/hand electrodes</td><td>Six functional hand movements with subject- specific EMG classifica- tion</td><td>Mean accuracy 71.3% for moderate impairment stroke severity strongly and 37.9% for severe impairment</td><td>Early evidence that affects decoder quality</td></tr><tr><td>2022</td><td>Anastasiev et al. 19 post-acute stroke patients; portable eight- channel sEMG</td><td>Four- to seven-gesture affected and non-affected the four-gesture setting sides</td><td>Affected-side SVM accu- multiclass recognition on racy reached 88.7% on</td><td>Shows that reduced- vocabulary stroke ges- ture recognition can reach high accuracy with careful feature design</td></tr><tr><td>Bao et al. 2024</td><td>8 chronic stroke partici- pants; post-stroke sEMG feature-domain study</td><td>CNN, CNN-LSTM, and CNN-LSTM-attention recognition using time, frequency, and wavelet</td><td>Best reported averages: 72.95% intra-subject accuracy and 68.38% inter-subject transfer</td><td>Supports deep archi- tectures for post-stroke decoding while high- lighting the difficulty of</td></tr><tr><td>et al. 2026</td><td>Mohammadiazni 10 stroke patients plus healthy Ninapro DB5 pretraining data</td><td>Three-gesture classifica- tion across multiple arm postures with transfer</td><td>accuracy 93.6% with transfer learning versus 46% without transfer</td><td>Strong stroke-specific motivation for healthy- to-impaired transfer learning</td></tr><tr><td>Sri-Iesaranusorn et al. 2021</td><td>Ninapro DB5/DB7, mostly healthy-subject settings</td><td>learning 41-movement deep neu- ral network classification</td><td>93.87% on DB5 and 91.69% on DB7</td><td>Healthy-data ceiling is high, but the task is easier than impaired- arm multilabel finger</td></tr><tr><td>This project</td><td>PhysioMio impaired- arm split from 48 stroke patients; patient-level train/val/test split</td><td>Five-finger multilabel intent decoding and four-channel hardware- targeted retraining</td><td>CNN-Large: 0.593 sub- set accuracy and 0.714 macro F1; distilled four- channel CNN-Micro: 0.522 ± 0.011 and 0.609 ± 0.006</td><td>decoding Connects impaired-arm model comparison to a concrete reduced-sensor interface</td></tr></table>

## 3 Method

## 3.1 Overview

This project defines a modular decoding workflow from multichannel sEMG and gesture annotations to five-finger prediction logits. The pipeline contains four stages: data harmonization, signal processing, model inference, and evaluation or deployment analysis. Figure 1 summarizes the workflow used throughout the study.

For notation, let a pre-segmented raw recording be $X _ { \mathrm { r a w } } \in \mathbb { R } ^ { T \times C }$ , where T is the number of raw time samples and C is either 64 for the high-density benchmarks or 4 for hardware-targeted retraining. After preprocessing and windowing, the project extracts a feature tensor $X _ { \mathrm { f e a t } } \in \mathbb { R } ^ { C \times W \times F }$ , where W is the number of windows and $F = 1 2$ is the number of handcrafted descriptors per channel-window pair. Each sample is paired with a multilabel target vector $y \in \{ 0 , 1 \} ^ { 5 }$ indicating intended activation of thumb, index, middle, ring, and little finger.

![](images/8222678ab949301438bc0e04fda37795568d6f149057d3f01ae05cda52913a3d.jpg)  
Figure 1: Overview of the sEMG finger-intent decoding pipeline. Raw sEMG and gesture annotations are aligned, segmented, featurized, mapped into shared tensors, evaluated across multiple model families, and prepared for downstream deployment analysis.

## 3.2 Data harmonization and ingestion

The implementation builds processed datasets directly from raw PhysioMio parquet files. The loader searches the raw data tree, groups files by patient, aligns contiguous stretches of the same movement\_type, maps each label to a five-bit finger target, discards contiguous label segments shorter than 200 raw samples, and applies the preprocessing pipeline to each usable segment. At the default 2000 Hz sampling rate, this 200-sample rule is a permissive minimum-duration filter for label segments; feature extraction still requires complete 200 ms windows, corresponding to 400 raw samples at 2000 Hz, because the windowing stage does not zero-pad short segments. This converts gesture-level clinical recordings into segment-level multilabel examples.

The original 64-channel model-family experiments use the project loader’s 2000 Hz configuration and select the impaired arm. The later four-channel experiment follows the documented PhysioMio rate of 2048 Hz. In both paths, patients are partitioned into deterministic 70/10/20 train/validation/test splits, so samples from the same patient do not appear in multiple partitions. Processed samples are padded to a dataset-wide maximum window count for batched tensor operations.

## 3.3 Signal preprocessing

The preprocessing module is explicitly parameterized by a frozen configuration object in the implementation. Raw sEMG first passes through a fourth-order Butterworth band-pass filter with 20–450 Hz cutofs and zero-phase forward-backward application. This stage suppresses low-frequency motion artifacts and high-frequency electrical noise while preserving time alignment across channels.

The filtered signal is then denoised with wavelet soft-thresholding. The default configuration uses a Symlet-4 basis, four decomposition levels, universal threshold selection, and a medianbased noise estimate. The same preprocessing configuration is used for all model families, making downstream diferences attributable to the representation and predictor rather than to separate signal-conditioning choices.

After denoising, each continuous segment is divided into 200 ms windows with 50% overlap and no zero-padding. These settings balance temporal responsiveness against the sample count required for stable summary statistics.

## 3.4 Window-level feature representation

Each channel-window pair is summarized by twelve handcrafted descriptors spanning time and frequency domains. The time-domain set includes root mean square, mean absolute value, integrated EMG, waveform length, variance, zero crossings, slope sign changes, and Willison amplitude. The frequency-domain set includes mean frequency, median frequency, spectral entropy, and total power, with spectral quantities estimated from Welch periodograms using nperseg = min(256, len(x)) [20]. For the default 200 ms PhysioMio window, len $( x ) = 4 0 0$ , so the Welch segment length is 256 samples; shorter windows, if produced by alternate configurations, use their full length. Because no explicit noverlap is passed, Sci $\mathrm { P y }$ uses its default 50% periodogram overlap. Degenerate spectra are zeroed explicitly to avoid unstable feature values. The resulting tensor has shape $( C , W , F )$ , where $C$ is the number of channels, W is the number of windows, and $F = 1 2$ is the feature count.

This feature representation is a deliberate engineering choice rather than a claim that raw end-to-end learning is unnecessary. Handcrafted time- and frequency-domain summaries reduce the input dimensionality, expose standard sEMG descriptors to all model families, and support compact students intended for embedded inference. They are also appropriate for the current single-dataset setting, where learning stable low-level time-frequency filters directly from raw HD-sEMG would require additional data and ablation experiments.

## 3.5 Tensor formatting and shared adapter

The project standardizes cross-model comparison with one shared adapter. It transforms $( C , W , F )$ or $( N , C , W , F )$ tensors into $( N , W , C \times F )$ sequences, where windows act as time steps and all channel features are concatenated into one vector per step. For a single sample, the adapted sequence can be written as

$$
S = \left[ s _ { 1 } , \dots , s _ { W } \right] ^ { \top } \in \mathbb { R } ^ { W \times d } , \qquad s _ { w } = \mathrm { v e c } ( X _ { \mathrm { f e a t } , : , w , : } ) , \qquad d = C F = 7 6 8 .
$$

This adapter creates a common input space for recurrent, convolutional, and graph-based predictors. For the CNN branch, the adapted tensor is subsequently permuted to a channels-first layout. The high-density CNN stack treats the $6 4 \times 1 2$ channel-feature grid as 768 input channels for temporal

Conv1d layers, preserving the same upstream representation while allowing convolutions over window order. The reduced-channel stack analogously uses $4 \times 1 2 = 4 8$ features per window. This is best interpreted as temporal convolution over a feature sequence whose per-window vector already contains spatial-channel and spectral summaries. It is not asserted to be superior to raw-signal 1D CNNs or 2D channel-by-time convolutions; those alternatives remain important ablations for future work.

## 3.6 Hardware-targeted channel reduction

The reduced-input experiment selects four active channels associated with two wrist/finger extensors and two finger flexors in the canonical order [ECRB, ECRL, FDS, FDP]. For the left array, the one-based PhysioMio channel indices are [1, 3, 9, 14]; for the right array they are [15, 16, 9, 1]. The corresponding zero-based right-map indices used by the final model are [14, 15, 8, 0]. Because the available metadata do not provide a reliable patient-level laterality field for this hardware placement, preprocessing creates left-map and right-map views under an identical patient split. The final experiment uses the right-map view selected during development.

Only four active sEMG signals enter the feature extractor. The hardware ground or reference electrode is not a fifth signal channel and is never included in $X _ { \mathrm { r a w } } , X _ { \mathrm { f e a t } }$ , or the neural-network input. At 2048 Hz, a 200 ms window contains 410 samples after rounding and the 50% overlap produces a 205-sample stride. The final model receives nine consecutive windows, giving an input $S _ { 4 \mathrm { c h } } \in \mathbb { R } ^ { 9 \times 4 8 }$ , and preserves the same five output logits as the high-density models.

## 3.7 Predictor families

The LSTM branch applies two stacked recurrent layers with hidden sizes 128 and 256, followed by a 128-unit fully connected layer and dropout before five output logits. Given adapted sequence input $s _ { t } .$ , the recurrent dynamics follow the standard LSTM update

$$
\begin{array} { r l } & { i _ { t } = \sigma \big ( W _ { i } s _ { t } + U _ { i } h _ { t - 1 } + b _ { i } \big ) , } \\ & { f _ { t } = \sigma \big ( W _ { f } s _ { t } + U _ { f } h _ { t - 1 } + b _ { f } \big ) , } \\ & { o _ { t } = \sigma \big ( W _ { o } s _ { t } + U _ { o } h _ { t - 1 } + b _ { o } \big ) , } \\ & { \tilde { c } _ { t } = \operatorname { t a n h } \big ( W _ { c } s _ { t } + U _ { c } h _ { t - 1 } + b _ { c } \big ) , } \\ & { c _ { t } = f _ { t } \odot c _ { t - 1 } + i _ { t } \odot \tilde { c } _ { t } , } \\ & { h _ { t } = o _ { t } \odot \operatorname { t a n h } ( c _ { t } ) , } \end{array}
$$

and the project uses the final hidden state from the second recurrent layer for classification. This branch emphasizes temporal accumulation across windows and serves as the main sequential baseline for the direct benchmark.

The GNN branch interprets each time window as a node in a graph. The implementation supports GCN, GraphSAGE, and GAT operators [11, 7, 19]; the benchmarked direct model uses the default two-layer GCN setting with hidden dimension 64, ReLU activations between graph-convolution layers, dropout 0.5, and mean graph pooling. The benchmarked path constructs a complete undirected graph over windows within each sample before graph-level pooling and readout. The resulting edge set has $O ( W ^ { 2 } )$ edges per sample, which is computationally feasible for the short padded window sequences used here and lets each window exchange information with all other windows. Sparse temporal adjacency and k-nearest-neighbor graph construction are left as graph-design ablations.

At a generic level, the node update can be written as

$$
h _ { i } ^ { ( \ell + 1 ) } = \psi _ { \ell } \left( h _ { i } ^ { ( \ell ) } , \mathop { \mathrm { A G G } } _ { j \in \mathcal { N } ( i ) } \phi _ { \ell } \left( h _ { i } ^ { ( \ell ) } , h _ { j } ^ { ( \ell ) } \right) \right) ,
$$

followed by graph pooling

$$
g = \mathrm { P O O L } \Big ( \{ h _ { i } ^ { ( L ) } \} _ { i = 1 } ^ { W } \Big )
$$

and a multilabel readout head. In the benchmarked GCN setting, AGG is the normalized neighborhood aggregation implemented by GCNConv, $\psi _ { \ell }$ is a linear graph-convolution update followed by ReLU for non-final layers, and POOL is global mean pooling. This branch injects an explicit structural prior: windows are related observations rather than only a flat sequence.

The CNN branch includes five student variants ranging from Nano to XLarge, each built around a $1 \times 1$ projection, temporal convolution blocks, adaptive pooling, and a compact fully connected head. If the channels-first representation is denoted by $\bar { S } \in \bar { \mathbb { R } ^ { d \times W } }$ , the student stack can be summarized as

$$
H ^ { ( 0 ) } = \rho \Bigl ( \mathrm { B N } ( W _ { p } * \hat { S } + b _ { p } ) \Bigr ) , \qquad H ^ { ( \ell + 1 ) } = \rho \Bigl ( \mathrm { B N } ( W _ { \ell } * H ^ { ( \ell ) } + b _ { \ell } ) \Bigr ) ,
$$

where ∗ denotes 1D convolution over window order and $\rho$ is a ReLU nonlinearity. Adaptive pooling compresses the temporal axis before a multilayer perceptron produces the five logits. The student family spans approximately 54K parameters and 53 KB INT8 for Nano to 1.62M parameters and 1.58 MB INT8 for XLarge. The teacher family uses 1D ResNet-50, ResNet-101, and ResNet-152 variants with bottleneck residual blocks.

## 3.8 Training, tuning, and evaluation protocol

The training utilities unify optimization and reporting across model families. The current implementation uses binary cross-entropy with logits for five-label prediction, Adam-based optimization, plateau-triggered learning-rate reduction, early stopping, checkpointing, and saved metric curves. For logits $z \in \mathbb { R } ^ { 5 }$ , the base multilabel objective is

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } \left[ y _ { k } \log \sigma ( z _ { k } ) + ( 1 - y _ { k } ) \log \left( 1 - \sigma ( z _ { k } ) \right) \right] .
$$

Evaluation reports subset accuracy, also called exact match ratio, finger-level averaged accuracy, macro precision, macro recall, macro F1, macro AUROC, and macro AUPRC. Subset accuracy counts a sample as correct only when all five finger labels are predicted correctly. The metrics module also stores per-finger scores and task-level curve data, which enables both aggregate and finger-wise analysis after training.

For the CNN students, Optuna searches architecture and training choices for each model size [1]. For a trial with validation macro F1 $F _ { \mathrm { v a l } }$ reported on a 0–100 scale, measured latency t, and size-specific baseline latency $t _ { \mathrm { r e f } }$ measured after one baseline epoch, the objective is

$$
{ \mathrm { s c o r e } } = { \frac { F _ { \mathrm { v a l } } } { 1 0 0 } } p \biggl ( { \frac { t } { t _ { \mathrm { r e f } } } } \biggr ) ,
$$

where

$$
p ( r ) = \left\{ \begin{array} { l l } { 1 , } & { r \leq 1 , } \\ { 1 - 3 ( r - 1 ) , } & { 1 < r \leq 1 . 2 , } \\ { 0 . 0 5 , } & { r > 1 . 2 . } \end{array} \right.
$$

Thus trials at or below the baseline latency are not penalized, mildly slower trials receive a linear penalty, and substantially slower trials are strongly downweighted. Trial-level early stopping and median pruning are used during search.

The system also implements teacher-ensemble distillation, per-finger threshold tuning, and latency benchmarking. For multilabel distillation, the implemented loss blends the hard target term with a temperature-scaled soft target from the teacher,

$$
\mathcal { L } _ { \mathrm { K D } } = \alpha \mathcal { L } _ { \mathrm { B C E } } ( z _ { s } , y ) + ( 1 - \alpha ) T ^ { 2 } \mathrm { B C E L o g i t s } \bigg ( \frac { z _ { s } } { T } , \sigma \bigg ( \frac { z _ { t } } { T } \bigg ) \bigg ) ,
$$

where $z _ { s }$ and $z _ { t }$ are student and teacher logits, $T$ is the distillation temperature, and α controls the hard-soft balance. In the four-channel experiment, paired examples present the full 768-feature representation to the fixed teacher and the matching 48-feature representation to the student. Training uses $T = 2$ and $\alpha = 0 . 5$ . Per-finger thresholds are selected on validation probabilities and then applied once to test predictions. A separate latency utility can report mean, median, and 95th-percentile inference time on a chosen device, but no Raspberry Pi timing is claimed in this study.

## 3.9 Transfer-learning paths

In addition to direct impaired-arm benchmarks, the study evaluates two distinct transfer mechanisms. In the first, CNN and LSTM models are pretrained on healthy-arm recordings and finetuned on impaired-arm recordings while preserving the same five-finger multilabel objective. The reported CNN experiment uses the CNN-Base configuration selected by the two-stage tuning run, so its rows do not duplicate either the legacy direct CNN baseline or the optimized CNN student sweep.

The four-channel study instead transfers across input density. A compatible 64-channel CNN-Micro source model is trained on the same patient split. Its first projection-layer weights are sliced to the 48 feature indices corresponding to the selected four channels, while shape-compatible later layers are copied directly. This initialization tests whether a full-grid representation transfers to the hardware-feasible sensor subset; it is evaluated separately from cross-channel distillation.

## 4 Experiments

## 4.1 Dataset and split construction

All experiments use the PhysioMio processing path implemented for this study [10, 6]. Dataset construction extracts contiguous movement segments from raw parquet recordings, maps gesture labels to five-bit finger targets, applies the signal-processing and feature-extraction stages, and stores tensors as PyTorch split files.

Patients are shufled with a fixed seed and divided into 70% train, 10% validation, and 20% test partitions. The main benchmarks use impaired-arm recordings. The healthy-to-impaired experiments use healthy-arm recordings for pretraining and impaired-arm recordings for finetuning. The four-channel study holds the patient assignment fixed across all training modes and contains 34 training patients (2656 samples), five validation patients (432 samples), and nine test patients (720 samples).

## 4.2 Experimental groups

The first group compares direct LSTM, CNN, and GNN baselines trained under the same 64-channel preprocessing and feature representation. The second evaluates optimized CNN students and ResNet teachers, emphasizing multilabel performance and model footprint. The third evaluates healthy-to-impaired transfer for CNN-Base and LSTM.

The fourth group targets the four-active-channel hardware constraint. Development runs compare left, right, and dual channel-map views and context lengths corresponding to one, four, and nine windows. The selected right-map, nine-window configuration is then trained under three modes: direct supervision, first-layer-sliced transfer from a compatible 64-channel CNN-Micro, and crosschannel distillation from a full-representation teacher. Each final mode is run with seeds 0–4. A separate 64-channel CNN-Micro source run with seed 42 provides a same-split reference but is not included in the five-seed variance estimate.

## 4.3 Evaluation and model selection

The task is five-label multilabel classification. We report subset accuracy (exact match ratio), mean finger accuracy, macro precision, macro recall, macro F1, macro AUROC, macro AUPRC, and perfinger metrics. Subset accuracy counts a sample as correct only when all five finger labels are correct. The baseline, CNN-sweep, ResNet, and healthy-transfer tables contain single deterministic-split estimates and therefore do not support statistical-significance claims.

For the four-channel final runs, per-finger thresholds are tuned on validation probabilities and applied to the held-out test set. Results are reported as the mean and sample standard deviation across five seeds; the artifact bundle additionally records 95% confidence-interval half-widths computed as $1 . 9 6 s / { \sqrt { 5 } }$ . The training mode is chosen by mean validation macro F1, and the handof checkpoint is the seed within that mode with the highest validation macro F1. Test performance is not used to choose the checkpoint.

## 5 Results

## 5.1 Direct baseline models: LSTM, CNN, and GNN

Table 2 reports the direct comparison among LSTM, CNN, and GNN decoders on the impaired-arm test split. The three families express diferent assumptions about the same windowed feature sequence. The LSTM treats the input as an ordered temporal process, the CNN searches for local temporal motifs after channel-feature projection, and the GNN treats windows as nodes in a sample-level graph. Their results show complementary behavior: the LSTM leads subset accuracy and macro AUROC, while the GNN leads macro F1 and macro AUPRC.

Table 2: Held-out test performance for the three direct model families. Subset accuracy is the multilabel exact match ratio. Bold numbers mark the best score in each column within this baseline comparison.
<table><tr><td>Model</td><td>Subset Acc.</td><td>Finger Acc.</td><td>Macro F1</td><td>Macro AUROC</td><td>Macro AUPRC</td></tr><tr><td>LSTM</td><td>0.545</td><td>0.784</td><td>0.705</td><td>0.858</td><td>0.754</td></tr><tr><td>CNN legacy</td><td>0.442</td><td>0.694</td><td>0.676</td><td>0.767</td><td>0.764</td></tr><tr><td>GNN</td><td>0.448</td><td>0.683</td><td>0.706</td><td>0.787</td><td>0.776</td></tr></table>

![](images/ba9942804e56fb61b0abd4410123bb7a2d5e33f3568265281349cbf599d8decc.jpg)  
Figure 2: Aggregate-metric comparison across the direct baselines and the strongest internal teacher and student references from the optimized CNN branch.

The baseline comparison suggests that temporal recurrence and graph aggregation capture diferent aspects of paretic sEMG. The LSTM’s stronger subset accuracy indicates better all-label consistency, whereas the GNN’s macro F1 and AUPRC indicate stronger class-balanced discrimination despite lower finger accuracy. The legacy CNN is weaker in this stage, motivating a systematic convolutional architecture search.

## 5.2 Optimized CNN students and ResNet teachers

Table 3 evaluates the optimized CNN student family together with internal ResNet teacher models. CNN-Large gives the strongest single-split aggregate CNN result, reaching 0.593 subset accuracy, 0.798 finger accuracy, 0.714 macro F1, 0.881 macro AUROC, and 0.812 macro AUPRC. The 158K-parameter CNN-Micro retains 0.578 subset accuracy and 0.697 macro F1 at an estimated 154 KB INT8 weight footprint, motivating its selection as the compact architecture for subsequent hardware-targeted study.

Table 3: CNN-family evaluation. Student parameter counts and INT8 sizes describe model scale; ResNet rows provide high-capacity teacher references. Bold numbers mark the best score in each column, and † marks the compact architecture carried into deployment-oriented development.
<table><tr><td>Variant</td><td>Type</td><td>Params</td><td>INT8 Size</td><td>Subset Acc.</td><td>Finger Acc.</td><td>Macro F1</td><td>Macro AUROC</td><td>Macro AUPRC</td></tr><tr><td>ResNet-50</td><td>Teacher</td><td>16.3M</td><td>15.6 MB</td><td>0.570</td><td>0.788</td><td>0.656</td><td>0.840</td><td>0.772</td></tr><tr><td>ResNet-101</td><td>Teacher</td><td>28.6M</td><td>27.3 MB</td><td>0.569</td><td>0.795</td><td>0.681</td><td>0.852</td><td>0.767</td></tr><tr><td>ResNet-152</td><td>Teacher</td><td>38.8M</td><td>36.9 MB</td><td>0.588</td><td>0.792</td><td>0.698</td><td>0.870</td><td>0.787</td></tr><tr><td>Nano</td><td>Student</td><td>54K</td><td>53 KB</td><td>0.580</td><td>0.794</td><td>0.698</td><td>0.855</td><td>0.758</td></tr><tr><td>Micro†</td><td>Student</td><td>158K</td><td>154 KB</td><td>0.578</td><td>0.794</td><td>0.697</td><td>0.871</td><td>0.793</td></tr><tr><td>Base</td><td>Student</td><td>390K</td><td>381 KB</td><td>0.575</td><td>0.798</td><td>0.707</td><td>0.870</td><td>0.775</td></tr><tr><td>Large</td><td>Student</td><td>803K</td><td>784 KB</td><td>0.593</td><td>0.798</td><td>0.714</td><td>0.881</td><td>0.812</td></tr><tr><td>XLarge</td><td>Student</td><td>1.62M</td><td>1.58 MB</td><td>0.551</td><td>0.794</td><td>0.690</td><td>0.830</td><td>0.726</td></tr></table>

CNN-Micro is within 0.015 subset accuracy and 0.017 macro F1 of CNN-Large while using approximately one fifth of its estimated INT8 footprint. The sweep is not monotonic in parameter count: CNN-XLarge is the largest student but performs below CNN-Large on every reported metric. This pattern is consistent with an intermediate capacity being better matched to the available dataset and search budget; confirming overfitting would require learning-curve and regularization ablations. ResNet-152 is the strongest teacher, with 0.588 subset accuracy and 0.698 macro F1, while CNN-Large has higher single-split values on the principal aggregate metrics.

## 5.3 Healthy-to-impaired transfer

Table 4 summarizes healthy-to-impaired transfer learning. This experiment uses the tuned CNN-Base two-stage configuration and is separate from both the legacy direct CNN and optimized student sweep. CNN-Base finetuning improves subset accuracy from 0.465 to 0.585, finger accuracy from 0.760 to 0.800, macro AUROC from 0.833 to 0.863, and macro AUPRC from 0.699 to 0.767. The LSTM behaves diferently: subset and finger accuracy increase after finetuning, but macro F1, AUROC, and AUPRC decrease.

Table 4: Healthy-to-impaired transfer-learning summaries. CNN rows use the tuned CNN-Base configuration. Bold values compare pretraining and finetuning within each architecture.
<table><tr><td>Model / Stage</td><td>Subset Acc.</td><td>Finger Acc.</td><td>Macro F1</td><td>Macro AUROC</td><td>Macro AUPRC</td></tr><tr><td>CNN-Base pre.</td><td>0.465</td><td>0.760</td><td>0.664</td><td>0.833</td><td>0.699</td></tr><tr><td>CNN-Base fine.</td><td>0.585</td><td>0.800</td><td>0.680</td><td>0.863</td><td>0.767</td></tr><tr><td>LSTM pre.</td><td>0.523</td><td>0.752</td><td>0.679</td><td>0.858</td><td>0.773</td></tr><tr><td>LSTM fine.</td><td>0.574</td><td>0.777</td><td>0.660</td><td>0.830</td><td>0.736</td></tr></table>

These results support healthy-to-impaired transfer for the CNN-Base configuration, but they do not establish transfer as uniformly beneficial across architectures or metrics.

## 5.4 Four-channel hardware-targeted retraining

Table 5 reports the final CNN-Micro study under the four-active-channel input constraint. Direct training reaches $0 . 5 8 2 2 \pm 0 . 0 1 6 2$ macro F1. First-layer-sliced transfer does not improve this result, attaining $0 . 5 7 0 6 \pm 0 . 0 1 0 9$ . Cross-channel distillation gives the strongest mean performance among the reduced-input modes, with $0 . 5 2 1 9 \pm 0 . 0 1 1 4$ subset accuracy, $0 . 7 6 1 2 \pm 0 . 0 0 3 8$ finger accuracy, $0 . 6 0 9 5 \pm 0 . 0 0 5 8$ macro F1, $0 . 7 9 0 4 \pm 0 . 0 0 7 3$ macro $\mathrm { A U R O C } ,$ , and $0 . 6 9 3 3 \pm 0 . 0 0 3 6$ macro AUPRC.

Table 5: Hardware-targeted CNN-Micro results. Four-channel entries are mean ± sample standard deviation across five seeds. The 64-channel source is a separate single-seed reference and is not included in the variance comparison. Bold values mark the best four-channel mean.
<table><tr><td>Training mode</td><td>Seeds</td><td>Subset Acc.</td><td>Finger Acc.</td><td>Macro F1</td><td>Macro AUROC</td><td>Macro AUPRC</td></tr><tr><td>64-channel source</td><td>1</td><td>0.5658</td><td>0.7846</td><td>0.6810</td><td>0.8520</td><td>0.7696</td></tr><tr><td>Direct 4-channel</td><td>5</td><td> $0 . 4 8 0 1 \pm 0 . 0 0 8 0$ </td><td> $0 . 7 3 2 6 \pm 0 . 0 1 2 7$ </td><td> $0 . 5 8 2 2 \pm 0 . 0 1 6 2$ </td><td> $0 . 7 6 8 3 \pm 0 . 0 1 3 9$ </td><td> $0 . 6 4 0 1 \pm 0 . 0 2 3 0$ </td></tr><tr><td>Transfer 4-channel</td><td>5</td><td> $0 . 4 6 1 8 \pm 0 . 0 0 8 9$ </td><td> $0 . 7 1 1 1 \pm 0 . 0 0 7 8$ </td><td> $0 . 5 7 0 6 \pm 0 . 0 1 0 9$ </td><td> $0 . 7 4 7 8 \pm 0 . 0 0 6 4$ </td><td> $0 . 5 9 8 5 \pm 0 . 0 1 6 4$ </td></tr><tr><td>Distilled 4-channel</td><td>5</td><td> $\mathbf { 0 . 5 2 1 9 \pm 0 . 0 1 1 4 }$ </td><td> $\mathbf { 0 . 7 6 1 2 \pm 0 . 0 0 3 8 }$ </td><td> $\mathbf { 0 . 6 0 9 5 \pm 0 . 0 0 5 8 }$ </td><td> $\mathbf { 0 . 7 9 0 4 \pm 0 . 0 0 7 3 }$ </td><td> $\mathbf { 0 . 6 9 3 3 \pm 0 . 0 0 3 6 }$ </td></tr></table>

Relative to direct four-channel training, distillation increases mean subset accuracy by 0.0418, finger accuracy by 0.0287, and macro F1 by 0.0272. The separate 64-channel source remains stronger, with 0.6810 macro F1 and 0.5658 subset accuracy, quantifying the cost of reducing the sensor grid. The selected four-channel network has 123,317 parameters, accepts a fixed (N, 9, 48) tensor, and produces five logits. The seed-4 checkpoint was selected by validation macro F1 and exported to ONNX; no on-device timing result is reported.

![](images/a7460847e153ea55dc84dc8851a960e891576d4306e890784e7dd99171dfb6be.jpg)  
Figure 3: Mean performance of the three four-channel CNN-Micro training modes. Distillation improves all displayed aggregate metrics relative to direct training and transfer initialization.

## 5.5 Finger-level behavior and external positioning

Finger-wise scores in the original model-family study show that ranking varies across outputs. Table 6 compares the LSTM and GNN direct baselines with CNN-Large. The GNN is strongest on thumb, CNN-Large leads on index and little finger, and the LSTM is slightly strongest on middle and ring finger. These patterns are plausible in light of hand biomechanics but remain hypotheses rather than anatomical evidence. Thumb activation is more anatomically distinct from the ulnar fingers, whereas middle and ring fingers frequently participate in coupled grasping synergies. Architecture-specific per-finger behavior therefore deserves targeted validation before control policies prioritize particular outputs.

Table 6: Per-finger F1 score in the original model-family study. Bold values mark the strongest model for each finger.
<table><tr><td>Finger</td><td>LSTM</td><td>GNN</td><td>CNN-Large</td></tr><tr><td>Thumb</td><td>0.716</td><td>0.776</td><td>0.729</td></tr><tr><td>Index</td><td>0.723</td><td>0.696</td><td>0.750</td></tr><tr><td>Middle</td><td>0.712</td><td>0.689</td><td>0.698</td></tr><tr><td>Ring</td><td>0.696</td><td>0.688</td><td>0.695</td></tr><tr><td>Little</td><td>0.680</td><td>0.678</td><td>0.696</td></tr></table>

Table 1 places these results beside prior stroke and sEMG studies. The comparison is not a leaderboard because populations, sensors, labels, and split protocols difer. The present task uses impaired-arm PhysioMio recordings, patient-level splits, and a five-label subset-accuracy metric requiring all finger activations to be correct simultaneously. Within that setting, the optimized CNN branch is competitive with internal ResNet teachers, and the reduced-channel study demonstrates the measurable trade-of between sensing density and an implementable input contract.

![](images/068c1e3969125cbe69f0e1b9d935c2c55d624502bb45b92aa448623569e3ffe1.jpg)  
Figure 4: Per-finger F1 comparison derived from committed metric files. Aggregate CNN gains coexist with output-specific strengths for the LSTM and GNN baselines.

## 6 Discussion

The direct comparison shows that the five-finger task benefits from more than one representation of temporal structure. The LSTM’s subset-accuracy advantage indicates greater consistency across the complete output vector, whereas the GNN’s macro F1 and AUPRC indicate stronger class-balanced discrimination. These complementary results are consistent with paretic sEMG containing both ordered temporal dynamics and relationships among non-adjacent windows.

Architecture search changes the interpretation of the initial CNN baseline. The tuned CNN students outperform the legacy CNN on every aggregate metric, and CNN-Large gives the strongest ofline result in that family. The non-monotonic decline for CNN-XLarge indicates that capacity alone does not determine performance under the available data and search budget. CNN-Micro retains most of CNN-Large’s performance at a substantially smaller estimated footprint, providing the architectural basis for reduced-channel development.

The two transfer-learning studies produce diferent outcomes. Healthy-to-impaired staging improves the CNN-Base configuration but gives mixed LSTM results, consistent with transfer being architecture- and metric-dependent. In the reduced-channel study, directly slicing the compatible 64-channel CNN-Micro initialization does not outperform training from scratch. The selected four channels may not preserve the spatial basis encoded by the full first-layer weights, so reusing those weights constrains the student without supplying all of the source information.

Cross-channel distillation is more efective because it transfers output behavior rather than requiring direct correspondence between full-grid and reduced-grid filters. Across five seeds, the distilled four-channel student improves mean subset accuracy, finger accuracy, macro F1, macro AUROC, and macro AUPRC relative to direct training. Its lower variance on macro F1 and ranking metrics also suggests more consistent optimization under the tested seeds. Nevertheless, the 64-channel source remains stronger, and the 0.0715 macro-F1 diference quantifies information lost when the input is reduced to four sensors. This is a meaningful engineering trade-of: the four-channel model sacrifices some ofline discrimination in exchange for a sensor contract that can be implemented by the planned device.

The work advances post-stroke sEMG decoding in three connected ways. It evaluates recurrent, convolutional, and graph models under one impaired-arm multilabel formulation; measures modeldevelopment strategies rather than assuming that transfer or scaling will help; and links the final learning problem to a concrete four-sensor input. Comparisons with Ninapro and reduced-vocabulary stroke studies must account for diferences in participants, labels, sensors, and split protocols, but the internal comparison against 1D ResNet teachers shows that compact temporal CNNs are credible for this PhysioMio task. The contribution is therefore both empirical and systems-oriented: a reproducible analysis of model choice followed by a measured transition from high-density recordings to deployable sensing.

The selected 123K-parameter student has been exported to a fixed-context ONNX graph, completing the software path from raw signal chunks to five thresholded finger predictions. This establishes interface compatibility, not real-time hardware performance. The current preprocessing reruns zero-phase filtering and wavelet denoising over rolling history, and its latency must be measured on the Raspberry Pi 5 before timing guarantees can be made.

Several limitations define the next experiments. The original model-family, architecture-search, ResNet, and healthy-transfer results are single-run estimates. The four-channel comparison adds five random seeds but retains one patient split; patient-level cross-validation is still needed to characterize generalization uncertainty. Electrode maps were inferred from the intended hardware placement rather than validated with signals collected by the final sensor assembly. Raw-signal CNNs, alternative channel-selection methods, sparse GNN graphs, and causal preprocessing remain untested ablations. Finally, the reported endpoints measure decoding rather than therapeutic benefit. Hardware-in-the-loop timing, robustness to electrode shift and signal drift, and studies with the target rehabilitation population are required before the model can support clinical conclusions.

## 7 Conclusion

This study developed a post-stroke five-finger intent decoder from impaired-arm sEMG and evaluated its progression from model-family comparison to hardware-targeted channel reduction. LSTM and GNN baselines expose complementary temporal behavior, while CNN architecture search identifies a compact convolutional path. Healthy-to-impaired transfer helps the CNN-Base experiment but not every architecture or metric.

For the four-active-channel design, cross-channel distillation produces the strongest reduced-input CNN-Micro, reaching $0 . 6 0 9 5 { \pm } 0 . 0 0 5 8$ macro F1 and $0 . 5 2 1 9 \pm 0 . 0 1 1 4$ subset accuracy across five seeds. The resulting 123K-parameter model accepts nine windows of ECRB, ECRL, FDS, and FDP features and emits five finger-intent logits through a verified ONNX interface. The central engineering result is that useful post-stroke finger-intent information can be retained under a practical four-sensor constraint, providing a concrete software foundation for Raspberry Pi 5 timing tests and subsequent hardware-in-the-loop rehabilitation research.

## References

[1] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019.

[2] Alexey Anastasiev, Hideki Kadone, Aiki Marushima, Hiroki Watanabe, Alexander Zaboronok, Shinya Watanabe, Akira Matsumura, Kenji Suzuki, Yuji Matsumaru, and Eiichi Ishikawa. Supervised myoelectrical hand gesture recognition in post-acute stroke patients with upper limb paresis on afected and non-afected sides. Sensors, 22(22):8733, 2022. doi: 10.3390/s22228733.

[3] Manfredo Atzori, Arjan Gijsberts, Claudio Castellini, Barbara Caputo, Anne-Gabrielle Mittaz Hager, Simone Elsig, Giorgio Giatsidis, Franco Bassetto, and Henning Müller. Electromyography data for non-invasive naturally-controlled robotic hand prostheses. Scientific Data, 1:140053, 2014. doi: 10.1038/sdata.2014.53.

[4] Tianzhe Bao, Zhiyuan Lu, and Ping Zhou. Deep learning based post-stroke myoelectric gesture recognition: From feature construction to network design. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024. doi: 10.1109/TNSRE.2024.3521583. Early access.

[5] Issam Boukhennoufa, Xiaojun Zhai, Victor Utti, Jo Jackson, and Klaus McDonald-Maier. Wearable sensors and machine learning in post-stroke rehabilitation assessment: A systematic review. Biomedical Signal Processing and Control, 71:103197, 2022. doi: 10.1016/j.bspc.2021. 103197.

[6] Zakariyya Brewster, Divy Wadhwani, Emily Yan, Aidan Wang, Karma Namgyal, Shuting Xie, Markiyan Konyk, and Tala Abdelmaguid. sEMG Finger-Intent Decoding Software for Post-Stroke Neurorehabilitation. https://github.com/post-stroke-rehab/psr-pipeline, 2026. Project software implementation, accessed July 28, 2026.

[7] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, 2017.

[8] Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[9] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9 (8):1735–1780, 1997.

[10] Julian Ilg, Alexander C. R. Oldemeier, Marie Fieweger, Luca Deuschel, Peter Rieckmann, Peter Young, Sabine Krause, and Tim C. Lueth. Physiomio: Bilateral and longitudinal hd-semg dataset of 16 hand gestures from 48 stroke patients. Scientific Data, 13(1):19, 2026. doi: 10.1038/s41597-026-06557-0.

[11] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

[12] Sang Wook Lee, Kristin Wilson, Blair A. Lock, and Derek G. Kamper. Subject-specific myoelectric pattern classification of functional hand movements for stroke survivors. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 19(5):558–566, 2010. doi: 10.1109/TNSRE.2010.2079334.

[13] Jianfeng Li, Xinyu Jiang, Jiahao Fan, Yanjuan Geng, Fumin Jia, and Chenyun Dai. Deep end-to-end transfer learning for robust inter-subject and inter-day hand gesture recognition using surface emg. Biomedical Signal Processing and Control, 100:106892, 2025. doi: 10.1016/j. bspc.2024.106892.

[14] M. Mohammadiazni, K. Huszar, S. Peters, and A. L. Trejos. Hand gesture intention detection using semg and transfer learning in stroke survivors. IEEE Journal of Biomedical and Health Informatics, 2026. doi: 10.1109/JBHI.2026.3693109. Early access, published May 13, 2026.

[15] Maria Munoz-Novoa, Morten B. Kristofersen, Katharina S. Sunnerhagen, Autumn Naber, Margit Alt Murphy, and Max Ortiz-Catalan. Upper limb stroke rehabilitation using surface electromyography: A systematic review and meta-analysis. Frontiers in Human Neuroscience, 16:897870, 2022. doi: 10.3389/fnhum.2022.897870.

[16] Sike Ni, Mohammed A. A. Al-qaness, Ammar Hawbani, Dalal Al-Alimi, Mohamed E. Abd Elaziz, and Ahmed A. Ewees. A survey on hand gesture recognition based on surface electromyography: Fundamentals, methods, applications, challenges and future trends. Applied Soft Computing, 166:112235, 2024. doi: 10.1016/j.asoc.2024.112235.

[17] Ninapro Project. The Non-Invasive Adaptive Prosthetics (Ninapro) Database. https:// ninapro.hevs.ch/, 2026. Project site, accessed July 28, 2026.

[18] Panyawut Sri-Iesaranusorn, Attawit Chaiyaroj, Chatchai Buekban, Songphon Dumnin, Ronachai Pongthornseri, Chusak Thanawattano, and Decho Surangsrirat. Classification of 41 hand and wrist movements via surface electromyogram using deep neural network. Frontiers in Bioengineering and Biotechnology, 9:548357, 2021. doi: 10.3389/fbioe.2021.548357.

[19] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.

[20] Peter D. Welch. The use of fast fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. IEEE Transactions on Audio and Electroacoustics, 15(2):70–73, 1967.

[21] Mingde Zheng, Michael S. Crouch, and Michael S. Eggleston. Surface electromyography as a natural human-machine interface: A review. arXiv preprint arXiv:2101.04658, 2021.