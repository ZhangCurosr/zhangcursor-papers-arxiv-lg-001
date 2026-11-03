# CARE: Contrastive Representation Learning with Environment-Invariance Regularization for Behavioural Intention Prediction

Shang Gao<sup>∗1</sup>, Zhixin Huang<sup>†1</sup>, Franz Götz-Hahn<sup>‡1</sup>, Konrad Doll<sup>§2</sup>, and Bernhard Sick<sup>¶1</sup>

<sup>1</sup>Intelligent Embedded Systems, University of Kassel, Germany, {sgao,

zhixin.huang,bsick,franz.goetz-hahn}@uni-kassel.de

<sup>2</sup>Aschafenburg University of Applied Sciences, Germany, konrad.doll@th-ab.de

## Abstract

Trajectory-based behavioural intention prediction faces environmental variation, and representations learned with a pooled contrastive objective may retain environment-specific regularities. We present CARE, a self-supervised trajectory-learning framework that couples temporally asymmetric contrastive views with an invariant-risk-minimisation (IRM)-style gradient penalty on environment-wise contrastive risks. Each input window yields an anchor, a partially masked positive view, and a temporally perturbed negative view. Masked reconstruction and observability-aware evidence consistency complement this objective by promoting temporal information retention under partial input. On an IMPTC-derived four-class intention task, CARE achieves 0.840 accuracy and 0.023 expected calibration error (ECE) under IID fine-tuning, and 0.806 accuracy with a Brier score of 0.308 on the reported held-out environment. Under frozen-encoder linear probing, reconstruction provides the largest individual held-out accuracy gain, whereas the complete objective yields the lowest held-out ECE (0.049) among tested variants. These results support the combined objective as a competitive representation-learning approach under the evaluated shift, while the IRM-style regulariser is not taken as evidence of formal invariance.

Keywords: behavioural intention prediction; contrastive representation learning; environment regularization; invariant risk minimization; trajectory self-supervision; probability calibration

## 1 Introduction

Behavioural Intention Prediction (BIP) estimates a road user’s upcoming manoeuvre from a short observed motion history. For vulnerable road users (VRUs), including pedestrians and cyclists, such estimates are relevant to proactive planning in autonomous driving [1]. Models are expected not only to distinguish intentions such as moving straight, turning or stopping, but also to retain predictive value when conditions at deployment difer from those seen during training.

Trajectory data are often heterogeneous: lighting, precipitation and time of day can vary across recordings. Standard supervised learning and conventional self-supervised pretraining do not, by themselves, require an encoder to avoid environment-specific shortcuts. This motivates a testable methodological question: can contrastive trajectory learning be regularized toward shared behaviour across predefined environments without sacrificing intention-prediction performance? Invariance here is a learning objective, not a property established by the use of an IRM penalty alone [2].

![](images/04ac4beb4714bd1430df6613f2c4c1fd9615f6910d62ca1406c760e79d4ff624.jpg)  
Figure 1: CARE combines temporally asymmetric contrastive views with an environment-wise IRMstyle penalty; reconstruction and evidence consistency provide auxiliary regularization. The architecture and computation graph are retained from the original manuscript.

CARE addresses this question by combining temporally asymmetric trajectory views with an environment-wise penalty on contrastive risk. For each input window, an anchor is compared with a masked positive and a temporally corrupted negative. The earlier part of the observed input window remains fixed while the later part is perturbed. The distinction is important: the perturbed segment is still observed input and is not the separate future interval used to derive the intention label. Contrastive learning encourages discrimination between these views; an IRM-inspired objective penalizes disparities in optimality of a shared projection head across training environments.

To complement the environment-regularised contrastive objective, CARE also reconstructs masked motion steps and constrains the evidential head across diferently observed views. The resulting joint objective couples discriminative trajectory structure, environment-wise regularisation, and observability-dependent auxiliary learning without assuming that a single constraint can guarantee robustness.

The contributions of this study are as follows:

1. Environment-regularized contrastive training. We formulate a contrastive loss over asymmetric motion views and add an IRM-style gradient penalty on contrastive risks computed within precipitation/time-of-day environment groups.

2. A controlled BIP evaluation. We retain the original IMPTC-derived four-class proxylabel task and report IID and selected held-out-environment results for the same baseline and CARE configurations.

3. A component-level reliability assessment. Using full-model evaluations and frozenencoder ablations, we distinguish the accuracy contribution of reconstruction from the probability-quality improvement associated with the combined objective, without attributing unverified gains to IRM alone.

## 2 Related Work

## 2.1 Behavioural Intention Prediction and Trajectory Representations

Behavioural intention prediction uses motion histories and contextual cues to distinguish the likely actions of road users [1, 3]. Learning from unlabelled motion is attractive when manually annotated trajectories are scarce. Contrastive predictive coding and TS2Vec demonstrate how self-supervised objectives can extract structure from sequential data [4, 5]; masked autoencoding supplies a complementary reconstruction signal [6]. CARE uses both mechanisms but directs its main regularization toward the diferences between training environments.

## 2.2 Environment Shift and Invariance Regularization

Domain generalization methods seek predictors that continue to work under distribution shifts. Domain-adversarial training suppresses domain-discriminative features [7], whereas invariant risk minimization (IRM) seeks representations for which one predictor is simultaneously adequate across environments [2]. Causal representation learning motivates, but does not guarantee, invariance under changes in nuisance conditions [8]. CARE adapts an IRM-style gradient penalty to an environment-wise contrastive risk instead of applying it directly to supervised intention classification. It is therefore an invariance-regularized objective, not a proof that the learned embeddings contain only causal or invariant factors.

## 2.3 Reconstruction and Observability-Aware Regularisation

Masked reconstruction complements contrastive learning by encouraging an encoder to retain local trajectory information even when parts of its input are missing [6]. Evidential learning introduces structured uncertainty parameters [9, 10]; their interpretation under partial observability motivates regularisation of the response to missing context. In particular, the visible-mean consistency and masked-evidence principles developed for evidential masked pretraining in eCoMAE [11] motivate CARE’s auxiliary constraints. Work on observability-dependent uncertainty, including MORSE [12], further illustrates the importance of relating uncertainty claims to measurable changes in input context. CARE incorporates observability-sensitive constraints into a temporally asymmetric, environment-regularised contrastive objective and evaluates prediction and classification calibration with accuracy, expected calibration error (ECE), negative log-likelihood (NLL), and Brier score [13].

## 3 Method

CARE targets environment-regularized contrastive representation learning for BIP. Table 1 summarizes notation. An input window $\mathbf { X } \in \mathbb { R } ^ { L \times F }$ and an environment label $e \in \{ 1 , \ldots , E \}$ are available at pretraining time. The learned encoder $f _ { \theta }$ is subsequently evaluated on four rule-derived intention classes: straight, left, right and stop. No causal graph or physical causal efect is estimated.

Three components appear in the objective: temporally asymmetric contrastive views, an environment-wise IRM-style contrastive penalty, and auxiliary masked reconstruction/evidence consistency. The components and their loss equations are defined below. Here, $L _ { p }$ and $L _ { f }$ refer to the early and late input segments; the later input segment must not be confused with the separate prediction horizon H used to construct proxy intention labels.

Table 1: Notation.  
Symbol Meaning   
$\mathbf { X } \in \mathbb { R } ^ { L \times F }$ input trajectory window, length L, feature dim F   
$\mathbf { x } _ { t } \in \mathbb { R } ^ { F }$ feature vector at time step t   
$L _ { p }$ past length, $L _ { f } = L - L _ { p }$ future length   
$\mathbf { A } , \mathbf { P } , \mathbf { N }$ anchor, masked positive, corrupted negative views   
$\mathcal { M } \subset \{ 1 , \ldots , L \}$ masked time index set used to form P   
$\rho , \ b$ mask rate, block length   
$f _ { \theta }$ encoder, $\mathbf { Z } = f _ { \theta } ( \mathbf { X } ) \in \mathbb { R } ^ { L \times D }$   
$h _ { w }$ projection head for contrastive vectors z   
$e \in \{ 1 , \ldots , E \}$ environment index, E number of environments   
$\textstyle B , \ B _ { e }$ training batch and its subset in environment e   
$d _ { \phi }$ decoder for reconstruction $\hat { \mathbf { X } } = d _ { \phi } ( \mathbf { Z } _ { p } )$   
$u _ { \psi }$ evidential head outputting NIG parameters $\mu , \mathbf { v } , \alpha , \beta$   
$\tau , \lambda .$ InfoNCE temperature and loss weights

Algorithm 1 Temporally asymmetric contrastive views (original CARE procedure)   
1: Input: $\mathbf { X } \in \mathbb { R } ^ { L \times F } ,$ , past length $L _ { p } ,$ , mask rate $\rho ,$ block length b, mask token m   
2: Anchor: $\mathbf { A }  \mathbf { X }$   
3: Positive: $\mathbf { P } \gets \mathbf { X } , M \gets \emptyset$   
4: Partition later-input indices $\{ L _ { p } { + } 1 , \ldots , L \}$ into consecutive blocks of length b   
5: Uniformly sample $\textstyle \lceil \rho \cdot { \frac { L - L _ { p } } { b } } \rceil$ blocks without replacement   
6: for each sampled block with indices $\{ s , \ldots , e \}$ do   
7: $M \gets M \cup \{ s , \ldots , e \}$ ; set $\mathbf { P } _ { t } \gets \mathbf { m }$ for all $t \in \{ s , \ldots , e \}$   
8: end for   
9: Negative: $\mathbf N \gets \mathbf X$   
10: apply one random corruption to the late segment $\mathbf { N } _ { L _ { p } + 1 : L } :$   
11: Rotate (temporal roll), Swap (batch roll), or Reverse (time flip)   
12: Output: A, P, N, M

## 3.1 Temporally Asymmetric Contrastive Views

CARE forms anchor, positive and negative views A, $\mathbf { P } , \mathbf { N } \in \mathbb { R } ^ { L \times F }$ from the same input window. The anchor is unmodified. Positive views mask contiguous blocks within the later input segment $\{ L _ { p } + 1 , \ldots , L \}$ , replacing the selected entries with a learnable token. Negative views apply one of three temporal corruptions to this same segment: rotation, reversal or swapping across a batch. The early $L _ { p }$ input steps remain identical across all three views.

The mask ratio $\rho$ and block length b determine the positive-view perturbation. The purpose of this asymmetry is to preserve a common earlier context while making the later input portion informative for distinguishing views; it is not an intervention on a physical cause, nor evidence that the downstream model is protected from future-label leakage. Algorithm 1 specifies the view construction procedure.

## 3.2 Environment-Invariance Regularization of Contrastive Risk

The encoder processes the three views, producing latent sequences ${ \bf Z } _ { a } = f _ { \theta } ( { \bf A } ) , { \bf Z } _ { p } = f _ { \theta } ( { \bf P } )$ $\mathbf { Z } _ { n } = f _ { \theta } ( \mathbf { N } ) \in \mathbb { R } ^ { L \times D }$ . A projection head produces contrastive vectors ${ \bf z } _ { a } , { \bf z } _ { p } , { \bf z } _ { n } \in \mathbb { R } ^ { D }$ . The InfoNCE loss aligns anchor and positive while repelling the negative:

$$
L _ { \mathrm { n c e } } = - \log \frac { \exp ( \sin ( \mathbf { z } _ { a } , \mathbf { z } _ { p } ) / \tau ) } { \exp ( \sin ( \mathbf { z } _ { a } , \mathbf { z } _ { p } ) / \tau ) + \exp ( \sin ( \mathbf { z } _ { a } , \mathbf { z } _ { n } ) / \tau ) } ,\tag{1}
$$

where sim is cosine similarity and $\tau$ is the temperature.

To discourage reliance on environment-specific shortcuts, CARE applies an IRM-style penalty to the environment-wise contrastive risk. Each sample is associated with an environment index $e \in \{ 1 , \ldots , E \}$ . Let $\mathcal { D } _ { e }$ denote the set of all training samples from environment e. During training, a batch of samples $B \subset \cup _ { e = 1 } ^ { E } \mathcal { D } _ { e }$ is drawn and partitioned by environment. Let $B _ { e } = \{ { \bf x } \in \mathcal { B } : \mathrm { e n v } ( { \bf x } ) = e \}$ denote the subset of the current batch from environment e. The environment-wise contrastive risk is estimated as

$$
\widehat { R } _ { e } ( \theta , w ) = \frac { 1 } { | \mathcal { B } _ { e } | } \sum _ { { \bf x } \in \mathcal { B } _ { e } } \ell _ { \mathrm { n c e } } ( \theta , w ; { \bf x } ) .\tag{2}
$$

CARE adopts an IRM-style gradient penalty intended to favour a shared projection mechanism across training environments; minimizing this surrogate does not itself establish causal or statistical invariance. Following the IRM principle, the penalty is measured by the gradient of each environment risk with respect to the head parameters, evaluated at a fixed reference head w<sub>0</sub>,

$$
L _ { \mathrm { i r m } } = \sum _ { e = 1 } ^ { E } \left\| \nabla _ { w } \widehat { R } _ { e } ( \theta , w ) \right\| _ { 2 } ^ { 2 } \Big | _ { w = w _ { 0 } } .\tag{3}
$$

Here w<sub>0</sub> is a shared and frozen initialization, and is not updated during training. Minimising $L _ { \mathrm { i r m } }$ updates the encoder parameters θ such that the same simple head can achieve low contrastive risk in all environments. The penalty encourages shared-head compatibility across the observed training groups, although stability on unobserved groups requires empirical assessment.

The environment-regularized contrastive objective is

$$
L _ { \mathrm { c i } } = \lambda _ { \mathrm { n c e } } L _ { \mathrm { n c e } } + \lambda _ { \mathrm { i r m } } L _ { \mathrm { i r m } } ,\tag{4}
$$

where $L _ { \mathrm { n c e } }$ is averaged over samples in the batch B. The term $L _ { \mathrm { i r m } }$ penalises environment-wise gradients as a surrogate for a common predictor; it does not enforce exact invariance across either the observed groups or unseen environments.

## 3.3 Auxiliary Reconstruction and Evidence Consistency

In addition to contrastive learning, CARE retains masked reconstruction and an observabilitydependent evidential consistency penalty on the positive view P. A decoder $d _ { \varphi }$ reconstructs the input sequence from the latent representation $\mathbf { Z } _ { p } = f _ { \theta } ( \mathbf { P } )$

$$
\begin{array} { r } { \hat { \mathbf { X } } = d _ { \varphi } ( \mathbf { Z } _ { p } ) , \qquad \hat { \mathbf { x } } _ { t } \in \mathbb { R } ^ { F } . } \end{array}\tag{5}
$$

Let $\mathcal { M } \subset \{ 1 , \ldots , L \}$ denote the set of masked time indices used to form P. The reconstruction loss is computed only on masked steps and encourages the encoder to preserve fine-grained temporal structure under partial observability,

$$
L _ { \mathrm { r e c } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { t \in \mathcal { M } } \left\| \mathbf { x } _ { t } - \hat { \mathbf { x } } _ { t } \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

Reconstruction alone does not constrain the outputs of an evidential head. As an auxiliary regularizer, CARE also uses a head $u _ { \psi }$ that outputs Normal-Inverse-Gamma parameters for each time step,

$$
( \pmb { \mu } , \mathbf { v } , \pmb { \alpha } , \beta ) = u _ { \psi } ( \mathbf { Z } ) , \qquad \pmb { \mu } _ { t } \in \mathbb { R } ^ { F } , \ \pmb { \alpha } _ { t } \in \mathbb { R } ^ { F } .\tag{7}
$$

The parameter $\pmb { \mu } _ { t }$ represents a predictive mean, while $\pmb { \alpha } _ { t }$ is the NIG shape parameter regularised at masked positions. These observability-dependent constraints follow the motivation of evidential masked pretraining [11]. Because the specified auxiliary objective does not include an NIG likelihood, the parameters are used as training signals rather than interpreted as validated calibrated uncertainty estimates.

The evidence-consistency regularizer relates outputs to observability through two complementary constraints defined on the anchor view A and the positive view P. Let $( \pmb { \mu } ^ { a } , \pmb { \alpha } ^ { a } ) = u _ { \psi } ( \mathbf { Z } _ { a } )$ and $( \pmb { \mu } ^ { p } , \pmb { \alpha } ^ { p } ) = u _ { \psi } ( \mathbf { Z } _ { p } )$ , where $\mathbf { Z } _ { a } = f _ { \theta } ( \mathbf { A } )$ . Let $\bar { \mathcal { M } } = \{ 1 , . . . , L \} \setminus \mathcal { M }$ denote visible indices. First, on visible steps, predictive means are encouraged to agree between the anchor and the masked view,

$$
\mathcal { L } _ { \mu } ^ { \mathrm { v i s } } = \frac { 1 } { \vert \bar { \mathcal { M } } \vert } \sum _ { t \in \bar { \mathcal { M } } } \big \Vert \pmb { \mu } _ { t } ^ { p } - \pmb { \mu } _ { t } ^ { a } \big \Vert _ { 2 } ^ { 2 } .\tag{8}
$$

Second, on masked steps, large values of the NIG shape parameter are penalised,

$$
\mathcal { L } _ { \alpha } ^ { \mathrm { m a s k } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { t \in \mathcal { M } } \left\| \pmb { \alpha } _ { t } ^ { p } \right\| _ { 1 } .\tag{9}
$$

The evidence-consistency regulariser is then

$$
\begin{array} { r } { { \mathcal { L } } _ { \mathrm { e c r } } = { \mathcal { L } } _ { \mu } ^ { \mathrm { v i s } } + { \mathcal { L } } _ { \alpha } ^ { \mathrm { m a s k } } . } \end{array}\tag{10}
$$

## 3.4 Combined Training Objective

CARE combines reconstruction and evidence consistency as an auxiliary objective:

$$
L _ { \mathrm { c a l } } = \lambda _ { \mathrm { r e c } } L _ { \mathrm { r e c } } + \lambda _ { \mathrm { e c r } } L _ { \mathrm { e c r } } .\tag{11}
$$

Together with the environment-regularized contrastive objective $L _ { \mathrm { c i } }$ in Sec. 3.2, the total self-supervised objective is

$$
L _ { \mathrm { s s l } } = \sum _ { k \in \mathcal { K } } \lambda _ { k } L _ { k } ,\tag{12}
$$

where $\mathcal { K } = \{ \mathrm { c i } , \mathrm { c a l } \}$ and $\lambda _ { k } \geq 0$ are scalar weights.

CARE is trained under two alternative protocols: two-stage training and joint training. In the two-stage protocol, the encoder $f _ { \theta }$ is first pretrained by minimising $L _ { \mathrm { s s l } }$ , and then frozen while a linear classifier is trained for intention prediction. In the joint protocol, the same $L _ { \mathrm { s s l } }$ is added as a regulariser to the supervised cross-entropy loss, enabling end-to-end optimisation while preserving the same objective components.

## 4 Experiments: Predictive Performance and Component Tradeofs

## 4.1 Dataset Construction and Task

A window level behavioural intention prediction dataset is constructed from IMPTC trajectory data. The raw data are frame wise tables grouped by an entity identifier, together with an entity level metadata table. For each identifier, the sequence is sorted by a time field and samples are generated via sliding windows. Given a sequence of length $L ,$ a start index s is advanced with stride S, a history window $[ s , s + L )$ is extracted as the model input, and the subsequent segment $[ s + L , s + L + H )$ is used to derive manoeuvre cues for label generation. Unless otherwise stated, the window length is $L = 1 0 0$ , the stride is $S = 8 0$ , and the future horizon for cue extraction is $H = 5 0$

The downstream task is a four class intention classification problem with labels: 0 straight, 1 left turn, 2 right turn, 3 stop.

Table 2: IMPTC BIP dataset overview and window construction settings.
<table><tr><td>Split</td><td>Samples</td><td>Unique IDs</td><td>Window length  $L$ </td><td>Past length  $L _ { p }$ </td><td>Future length  $L _ { f }$ </td><td>Stride S</td></tr><tr><td>Train</td><td>28272</td><td>3328</td><td>100</td><td>60</td><td>40</td><td>80</td></tr><tr><td>Test</td><td>8155</td><td>1065</td><td>100</td><td>60</td><td>40</td><td>80</td></tr></table>

Table 3: Intention label distribution. Percentages are computed within each split.
<table><tr><td>Split</td><td>Straight</td><td>Left</td><td>Right</td><td>Stop</td></tr><tr><td>Train</td><td>15099 (53.41%)</td><td>2347 (8.30%)</td><td>2319 (8.20%)</td><td>8507 (30.09%)</td></tr><tr><td>Test</td><td>4761 (58.38%)</td><td>698 (8.56%)</td><td>677 (8.30%)</td><td>2019 (24.76%)</td></tr></table>

Input features Each sample contains a trajectory tensor $X \in \mathbb { R } ^ { L \times 7 }$ with seven per timestep features in a fixed order: $x , y ,$ speed magnitude, z height, ground\_type, z\_ofset, source\_type. In addition, each sequence is associated with two context variables used for environment construction: precipitation intensity and part of day.

Rule derived intention labels Window level intention annotations y are derived from kinematic rules. Stop is assigned if the mean speed over the last K steps of the history window is below $v _ { \mathrm { s t o p } } .$ . The configuration uses $v _ { \mathrm { s t o p } } = 0 . 2 5$ and $K \in [ 3 , 1 0 ]$ . Otherwise, the signed angle $\Delta \theta$ between the last displacement in the history and the displacement from the history end to the future end is computed. Left is assigned if $\Delta \theta > \theta .$ , right is assigned if $\Delta \theta < - \theta$ , and straight is assigned otherwise. The configuration uses $\theta = 1 2 . 5 ^ { \circ }$ . These are proxy labels, and their utility is assessed through calibration and robustness metrics in addition to accuracy.

Dataset scale and label distribution Tables 2 and 3 report dataset scale and label distribution. All supervised models are trained with class-weighted cross-entropy to reduce bias towards frequent classes. For evaluation, accuracy is used to enable a fair comparison with standard time-series baselines under identical settings. Although accuracy can be influenced by imbalance, the comparisons remain consistent because the same metric and splits are used for all methods.

## 4.2 Environment Construction and Robustness Setup

Robustness under domain shift is evaluated using discrete environment labels. Each sample is assigned an environment index $e \in \{ 1 , \ldots , E \}$ defined by the joint condition of precipitation intensity and part of day. Precipitation intensity is discretised into three ordered levels using quantile binning on the training split, which yields balanced partitions while preserving the ordering from light to heavy precipitation. Part of day is discretised into four six-hour intervals in local time: 00:00–06:00, 06:00–12:00, 12:00–18:00, and 18:00–24:00. The nominal crossproduct comprises up to twelve precipitation–time categories; the original experiment uses the environment indices produced by its data preparation pipeline.

The realized environment distribution is long tailed. In the training split, the largest realized group contains 4,396 samples, while the smallest realized group contains 97 samples, corresponding to a ratio of approximately 45:1 between the largest and smallest groups. In the test split, the largest realized group contains 1,946 samples, while the smallest realized group contains 74 samples, corresponding to a ratio of approximately 26:1. Some environment codes appear in training but are absent in testing, and several realized groups remain rare with fewer than one

hundred samples.

These environment labels are used in two ways. First, for invariance learning, each training batch is partitioned into environment-specific subsets $\{ B _ { e } \} _ { e = 1 } ^ { E }$ , and the IRM-style penalty in Eq. 3 is computed from the corresponding environment-wise risks $R _ { e }$ . Second, for robustness evaluation, a leave-one-environment-out protocol is adopted: one large realised environment is held out as test data and the remaining environments are used for training. This evaluates generalisation to an unseen group defined by precipitation and time-of-day metadata under a naturally imbalanced environment distribution; the setup does not isolate the efect of either environmental factor.

## 4.3 Evaluation Protocols

IID evaluation IID evaluation uses the provided train split for training and the provided test split for evaluation. In addition to accuracy, worst group accuracy is reported, defined as the minimum accuracy across realized environment groups.

Leave one environment out evaluation Leave one environment out evaluation holds out a target environment group as the test domain and trains on the remaining environment groups. Due to the long tailed distribution, exhaustive leave one environment out across all groups can be unstable when the held out group is small. A selected leave-one-environment-out protocol is therefore used: the selected held-out environment is described as the largest realised group, providing a more populated evaluation set while limiting the conclusions to this particular environment shift.

Two stage training A two stage pipeline is used. Stage one is linear probing, where the encoder is frozen and a linear classifier is trained to probe representation quality. Stage two is global finetuning, where the full model is finetuned end to end for final downstream performance. Baseline comparisons are reported using stage-two fine-tuning, whereas component ablations use stage-one linear probing. The two protocols must not be interpreted as directly interchangeable.

## 4.4 Baselines, Metrics, and Implementation Details

Baselines CARE is compared with DLinear [14], LiST [15], PatchTST [16], TCN [17], DAE [18] and EAE [19]. All baselines follow the same splits and evaluation protocols.

Metrics Accuracy and worst group accuracy measure predictive performance. Negative log likelihood and Brier score evaluate probability quality. Calibration is assessed using expected calibration error (ECE),

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | \mathcal { T } _ { m } | } { N } \left| \operatorname { a c c } ( \mathcal { T } _ { m } ) - \operatorname { c o n f } ( \mathcal { T } _ { m } ) \right| ,
$$

where $\mathcal { I } _ { m }$ denotes the index set of samples whose predicted confidence falls into the $m { \mathrm { - t h } }$ bin, and N is the number of evaluated samples.

Implementation details Unless otherwise stated, a Transformer backbone is used with hidden dimension $d _ { \mathrm { m o d e l } } = 6 4$ , feed-forward dimension $d _ { \mathrm { f f } } = 1 2 8$ , and number of encoder layers $e _ { \mathrm { l a y e r s } } = 2$ . Batch size is 16 and the random seed is 2. Self supervised pretraining uses 50 epochs with learning rate $1 0 ^ { - 4 }$ . Stage one linear probing uses 5 epochs with learning rate $5 \times 1 0 ^ { - 3 }$

Table 4: IID results using stage two global finetuning. Higher is better for accuracy and worst group accuracy. Lower is better for calibration and probability quality metric. Best values per column are in bold.
<table><tr><td>Model</td><td>Acc</td><td>Worst group</td><td>ECE</td><td>NLL</td><td>Brier</td></tr><tr><td>CARE</td><td>0.840</td><td>0.800</td><td>0.023</td><td>0.453</td><td>0.233</td></tr><tr><td>DLinear</td><td>0.821</td><td>0.794</td><td>0.044</td><td>0.539</td><td>0.254</td></tr><tr><td>LiST</td><td>0.830</td><td>0.782</td><td>0.030</td><td>0.459</td><td>0.236</td></tr><tr><td>PatchTST</td><td>0.817</td><td>0.774</td><td>0.070</td><td>0.544</td><td>0.269</td></tr><tr><td>TCN</td><td>0.771</td><td>0.738</td><td>0.025</td><td>0.717</td><td>0.356</td></tr><tr><td>DAE</td><td>0.791</td><td>0.770</td><td>0.041</td><td>0.677</td><td>0.334</td></tr><tr><td>EAE</td><td>0.795</td><td>0.771</td><td>0.043</td><td>0.665</td><td>0.327</td></tr></table>

Table 5: Representative leave one environment out results using stage two global finetuning. The held out environment is chosen from the largest environment groups due to long tailed environment distribution. Higher is better for accuracy. Lower is better for calibration and probability quality metrics. Best values per column are in bold.
<table><tr><td>Model</td><td>Acc</td><td>ECE</td><td>NLL</td><td>Brier</td></tr><tr><td>CARE</td><td>0.806</td><td>0.090</td><td>0.623</td><td>0.308</td></tr><tr><td>DLinear</td><td>0.794</td><td>0.095</td><td>0.756</td><td>0.333</td></tr><tr><td>LiST</td><td>0.785</td><td>0.070</td><td>0.622</td><td>0.312</td></tr><tr><td>PatchTST</td><td>0.779</td><td>0.124</td><td>0.719</td><td>0.350</td></tr><tr><td>TCN</td><td>0.741</td><td>0.039</td><td>0.773</td><td>0.396</td></tr><tr><td>DAE</td><td>0.774</td><td>0.053</td><td>0.753</td><td>0.377</td></tr><tr><td>EAE</td><td>0.771</td><td>0.061</td><td>0.747</td><td>0.375</td></tr></table>

Stage two global finetuning uses 5 epochs with learning rate $1 0 ^ { - 4 }$ . The past window length is 60 and the total sequence length is 100.

## 4.5 Downstream Performance after Global Fine-Tuning

IID results Table 4 reports stage two IID results. CARE leads the reported IID fine-tuning comparison on accuracy (0.840), worst-group accuracy (0.800), ECE (0.023), NLL (0.453), and Brier score (0.233). These are downstream classification metrics, rather than direct measurements of the NIG head.

Leave one environment out results Table 5 reports stage two representative leave one environment out results. CARE leads the reported selected held-out-environment comparison in accuracy (0.806) and Brier score (0.308). In terms of negative log-likelihood, CARE is competitive and essentially tied with the strongest baseline (0.623 vs. 0.622). TCN has the lowest ECE (0.039, compared with CARE’s 0.090), so this evaluation supports a trade-of rather than a uniformly best-calibrated model.

## 4.6 Component Analysis with Frozen Representations

This experiment evaluates the contribution of CARE components to encoder representation quality using stage one linear probing. The encoder is frozen and only a linear classifier is trained. Three components are ablated: reconstruction constraint REC, invariance regularization IRM, and evidential regularization EVI with weight 0.05.

Table 6: IID component ablation with a frozen encoder (linear probing). Base denotes contrastive NCE; REC, IRM and EVI denote reconstruction, IRM-style regularisation and auxiliary evidence consistency. Best values per column are bold (Acc/worst-group Acc ↑, NLL/Brier/ECE ↓).
<table><tr><td>Setting</td><td>Acc</td><td>Worst group</td><td>NLL</td><td>Brier</td><td>ECE</td></tr><tr><td>Base</td><td>0.783</td><td>0.755</td><td>0.695</td><td>0.342</td><td>0.040</td></tr><tr><td>REC</td><td>0.803</td><td>0.778</td><td>0.625</td><td>0.302</td><td>0.053</td></tr><tr><td>IRM</td><td>0.784</td><td>0.755</td><td>0.694</td><td>0.341</td><td>0.038</td></tr><tr><td>EVI</td><td>0.783</td><td>0.756</td><td>0.696</td><td>0.343</td><td>0.038</td></tr><tr><td>REC+IRM</td><td>0.792</td><td>0.768</td><td>0.661</td><td>0.327</td><td>0.028</td></tr><tr><td>REC+EVI</td><td>0.792</td><td>0.768</td><td>0.662</td><td>0.327</td><td>0.028</td></tr><tr><td>IRM+EVI</td><td>0.781</td><td>0.754</td><td>0.700</td><td>0.344</td><td>0.041</td></tr><tr><td>All</td><td>0.804</td><td>0.782</td><td>0.625</td><td>0.303</td><td>0.047</td></tr></table>

Table 7: Held-out-environment component ablation with a frozen encoder (linear probing). Base denotes contrastive NCE; REC, IRM and EVI denote reconstruction, IRM-style regularisation and auxiliary evidence consistency. Best values per column are bold (Acc ↑, NLL/Brier/ECE ↓).
<table><tr><td>Setting</td><td>Acc</td><td>NLL</td><td>Brier</td><td>ECE</td></tr><tr><td>Base</td><td>0.765</td><td>0.750</td><td>0.381</td><td>0.066</td></tr><tr><td>REC</td><td>0.779</td><td>0.720</td><td>0.356</td><td>0.068</td></tr><tr><td>IRM</td><td>0.765</td><td>0.746</td><td>0.380</td><td>0.073</td></tr><tr><td>EVI</td><td>0.765</td><td>0.749</td><td>0.381</td><td>0.070</td></tr><tr><td>REC+IRM</td><td>0.774</td><td>0.731</td><td>0.370</td><td>0.069</td></tr><tr><td>REC+EVI</td><td>0.774</td><td>0.731</td><td>0.370</td><td>0.070</td></tr><tr><td>IRM+EVI</td><td>0.762</td><td>0.752</td><td>0.382</td><td>0.075</td></tr><tr><td>All</td><td>0.777</td><td>0.714</td><td>0.356</td><td>0.049</td></tr></table>

IID ablation Table 6 reports IID linear-probe results. Adding reconstruction (REC) provides the largest single-component improvement, increasing accuracy and yielding the best probability quality, with the lowest NLL and the lowest Brier score. Enabling all components (All) achieves the best accuracy and worst-group accuracy, while its ECE is higher than the best combined settings (REC+IRM and REC+EVI), which attain the lowest ECE. Overall, REC is the main driver for probability quality, whereas the full configuration ofers the strongest accuracy under the linear-probe protocol.

Leave one environment out ablation Table 7 reports selected leave-one-environment-out linear-probe results. On the reported held-out environment, the full configuration (All) has the lowest ECE (0.049) and NLL (0.714), and matches REC on Brier score (0.356). REC has the highest accuracy in this linear-probing ablation (0.779), versus 0.777 for All; the combination therefore provides a calibration/likelihood trade-of rather than uniformly improving every metric.

## 4.7 Qualitative Inspection of Representation Geometry

Beyond quantitative metrics, a qualitative inspection of the representation geometry learned by CARE is provided. Embeddings are projected to two dimensions using UMAP [20]. Each point corresponds to one trajectory window of length L=100 with F=7 per-timestep features. Points are coloured by attributes derived from the input window to assess whether the embedding manifold reflects behaviour-relevant structure.

![](images/25483c748c0dc2cc22a44d602ee028c961e744759b7636906490438992384713.jpg)  
Figure 2: UMAP visualisation of CARE encoder embeddings on the test split, coloured by the discrete environment ID (used by CARE for environment-aware learning and evaluation). Colours appear interleaved in this two-dimensional projection; this alone does not establish environment invariance. Environment IDs 2 and 3 are absent in the test split and therefore do not appear in the plot.

![](images/1a59962fc7bc06aca42ca64822d91223efb19f0638ef8dbc5b9c95d3e8ee34a4.jpg)

![](images/b5c3cab49c45796c725ab53fe78b5ecbddf95803a35fce504e7e14ad3b6053ec.jpg)  
Figure 3: UMAP visualisation of CARE encoder embeddings on the test split, coloured by trajectorylevel statistics of the x coordinate over each window (L=100). Left: mean(x); right: std(x). The smooth gradient in mean(x) suggests coherent geometric organisation, while localised high std(x) highlights windows with larger spatial variability.

Fig. 2 shows that samples from diferent environments are largely interleaved across the embedding manifold, without forming clearly environment-separated islands. This descriptive visualization does not show obvious environment-separated islands, but UMAP projections alone cannot establish statistical independence of features and environment labels. Figure 3 colours the same embeddings by trajectory-level statistics of the x coordinate. A smooth gradient in mean(x) over the manifold suggests coherent geometric organisation, while localised regions with high std(x) highlight windows with larger motion variability. Finally, Figure 4 colours embeddings by ground-type context. The modal ground\_type is dominated by the most common VRU surfaces and does not partition the manifold into isolated semantic clusters. Instead, regions with higher ground-type diversity tend to concentrate in specific parts of the manifold and align with areas of higher motion variability, suggesting that CARE preserves interpretable scene-context cues in addition to continuous geometric factors. These qualitative patterns suggest locally interpretable geometry. They do not constitute a quantitative test of environment invariance.

![](images/9f6868ce42f2d46a3eff902cb9a8660f25a2318fad1bb04abb79ec6a1121d829.jpg)

![](images/2d3163a3a834b567baac761125b55eb0de9e02875e1839b6fa5939f99a1197ef.jpg)  
Figure 4: UMAP projection of CARE encoder embeddings on the test split, coloured by ground-type context aggregated within each trajectory window (L=100). Left: modal ground\_type in the window. Right: number of unique ground\_type values observed in the window, serving as a proxy for local context diversity. The dominant mode is consistent with typical VRU trajectories occurring on sidewalks or road surfaces. Colours remain largely interleaved across the manifold rather than forming isolated semantic islands, while regions with higher context diversity concentrate locally and align with areas of higher motion variability in other statistics, providing interpretable context cues without the representation being driven by discrete ground-type labels.

## 5 Discussion

## 5.1 Predictive Performance and Component Contributions

The reported global fine-tuning comparison favours CARE on all five IID measures, including accuracy of 0.840 and ECE of 0.023. In the selected held-out-environment test, CARE achieves the highest reported accuracy (0.806) and the lowest Brier score (0.308), but TCN has a lower ECE (0.039 versus 0.090) and LiST has a marginally lower NLL (0.622 versus 0.623). These distinctions matter: a strong score under one particular environment shift does not imply invariance or uniformly superior calibration.

Component ablations tell a more selective story. With a frozen encoder, masked reconstruction gives the clearest individual gain in held-out-environment accuracy (0.779 versus 0.765 for the contrastive baseline). IRM alone matches the baseline in held-out-environment accuracy and has higher ECE, while EVI alone does not improve that accuracy. The full configuration improves held-out-environment ECE to 0.049 and NLL to 0.714 at accuracy 0.777, slightly below the 0.779 of REC. This pattern motivates studying interactions among regularizers; it does not justify treating all auxiliary penalties as independently beneficial.

## 5.2 Limitations and Reproducibility

The benchmark uses rule-derived proxy intention labels and the reported environment groups are long-tailed. The reported leave-one-environment-out results represent one fixed, large held-out group, not an exhaustive average across all environment groups. Experiments use the reported random seed of 2 and do not quantify seed-to-seed variability. UMAP visualizations provide qualitative interpretation but cannot establish learned invariance. The temporal perturbations afect the later portion of the input window; the paper makes no causal intervention or futurelabel-leakage claim. The exact correspondence between numeric environment codes and the nominal precipitation–time categories is not fully documented in the available manuscript, limiting independent verification of environment group membership.

Future work could evaluate all adequately populated held-out environment groups, quantify multiple-seed uncertainty, and isolate the temporal-asymmetry and environment-risk penalty contributions under otherwise matched protocols.

## 6 Conclusion

CARE studies an environment-regularised contrastive objective for motion representation learning, combining temporally asymmetric views with an IRM-style penalty on environmentwise risks. Reconstruction and evidence consistency complement this main objective as auxiliary losses. On the IMPTC-derived intention benchmark, CARE obtains 0.840 IID accuracy and 0.806 accuracy on the selected held-out environment after fine-tuning. Under frozen-encoder evaluation, reconstruction is the strongest individual contributor to held-out accuracy, whereas the full objective provides the lowest ECE among the tested combinations. These results support the practical value of the proposed combination under the evaluated conditions, but do not establish causal invariance, broad environment-generalisation guarantees, or calibrated NIG uncertainty. The study provides an environment-wise contrastive regularisation formulation and a component-level evaluation of its predictive trade-ofs in behavioural intention prediction.

## Acknowledgments

This research has been partly funded by the Hessian State Ministry for Higher Education, Research, Science and the Arts (HMWK) as part of the LOEWE program (Kapitel 15 02, Förderbuchungskreis 2995, Förderprodukt 11 – Landes-Ofensive zur Entwicklung Wissenschaftlichökonomischer Exzellenz).

## References

[1] Jianwu Fang, Fan Wang, Jianru Xue, and Tat-Seng Chua. Behavioral intention prediction in driving scenes: A survey. IEEE Transactions on Intelligent Transportation Systems, 25(8):8334–8355, 2024. doi: 10.1109/TITS.2024.3374342.

[2] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization, 2020. URL https://arxiv.org/abs/1907.02893.

[3] Stefan Zernetsch, Viktor Kress, Maarten Bieshaar, Jan Schneegans, Günther Reitberger, Erich Fuchs, Bernhard Sick, and Konrad Doll. Detecting intentions of vulnerable road users based on collective intelligence as a basis for automated driving. In Christoph Stiller, Matthias Althof, Christoph Burger, Barbara Deml, Lutz Eckstein, and Frank Flemisch, editors, Cooperatively Interacting Vehicles: Methods and Efects of Automated Cooperation in Trafic, pages 35–87. Springer International Publishing, Cham, 2024. ISBN 978-3-031-60494-2. doi: 10.1007/978-3-031-60494-2\_3. URL https://doi.org/10.1007/978-3-031-60494-2\_3.

[4] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.

[5] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series, 2022. URL https: //arxiv.org/abs/2106.10466.

[6] Zhe Li, Zhongwen Rao, Lujia Pan, Pengyun Wang, and Zenglin Xu. Ti-mae: Self-supervised masked time series autoencoders, 2023. URL https://arxiv.org/abs/2301.08871.

[7] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks, 2016. URL https://arxiv.org/abs/1505.07818.

[8] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Towards causal representation learning, 2021. URL https: //arxiv.org/abs/2102.11107.

[9] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty, 2018. URL https://arxiv.org/abs/1806.01768.

[10] Alexander Amini, Wilko Schwarting, Ava Soleimany, and Daniela Rus. Deep evidential regression, 2020. URL https://arxiv.org/abs/1910.02600.

[11] Shang Gao, Franz Götz-Hahn, Konrad Doll, and Bernhard Sick. Observability-aligned evidential masked pretraining for reliable vru motion prediction, 2026. URL https://arxiv.org/abs/2610. 39958.

[12] Shang Gao, Bernhard Sick, and Franz Götz-Hahn. Grounding predictive reliability through masked observability, 2026. URL https://arxiv.org/abs/2610.39728.

[13] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks, 2017. URL https://arxiv.org/abs/1706.04599.

[14] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers efective for time series forecasting?, 2022. URL https://arxiv.org/abs/2205.13504.

[15] Zhixin Huang, Christian Gruhl, and Bernhard Sick. List: An all-linear-layer spatial-temporal feature extractor with uncertainty estimation for rul prediction. In 2024 IEEE 19th Conference on Industrial Electronics and Applications (ICIEA), pages 1–7, 2024. doi: 10.1109/ICIEA61579.2024.10665236.

[16] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers, 2023. URL https://arxiv.org/abs/2211.14730.

[17] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018. URL https://arxiv.org/abs/1803.01271.

[18] Ghassan Al-Falouji, Shang Gao, Lukas Haschke, Dirk Nowotka, and Sven Tomforde. Enhancing maritime behaviour analysis through novel feature engineering and digital shadow modelling: A case study in the kiel fjord. In International Conference on Architecture of Computing Systems, pages 97–111. Springer, 2024.

[19] Shang Gao, Zhixin Huang, Ghassan Al-Falouji, Bernhard Sick, and Sven Tomforde. Towards cognitive situational awareness in maritime trafic using federated evidential learning. In 2025 IEEE Conference on Cognitive and Computational Aspects of Situation Management (CogSIMA), pages 9–16, 2025. doi: 10.1109/CogSIMA64436.2025.11079521.

[20] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction, 2020. URL https://arxiv.org/abs/1802.03426.