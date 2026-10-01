# A COMPREHENSIVE BENCHMARK OF SOURCE-FREE UNIVERSAL DOMAIN ADAPTATION ON TIME SERIES REPRESENTATIONS

Romain Mussard<sup>⋆</sup> Fannia Pacheco<sup>⋆</sup> Maxime Berar<sup>⋆</sup> Paul Honeine<sup>⋆</sup> Gilles Gasso<sup>⋆</sup>

<sup>⋆</sup> Univ Rouen Normandie, INSA Rouen Normandie, Universite Le Havre Normandie, Normandie Univ,´ LITIS UR 4108, F-76000 Rouen, France

## ABSTRACT

Source-Free Universal Domain Adaptation (SF-UniDA) extends Universal Domain Adaptation by removing access to source data at adaptation time while still handling label-set mismatches between domains. Despite growing interest in this setting for image data, no benchmark exists for time series, which are more challenging. We present the first SF-UniDA benchmark on time series. In addition, we provide the first study of pretrained foundation models as feature extractors for time series domain adaptation. In this context, we identify a critical and previously underexplored limitation of all existing SF-UniDA methods: the inference threshold for unknown-sample rejection is highly sensitive. We address this by proposing a plug-in auto-thresholding module that can be integrated into any SF-UniDA method. Experiments on three well-known time series datasets confirm the suitability of this module. They also highlight that foundation models do not systematically outperform classical backbones and that SF-UniDA tailored for time series is yet to be developed.

Index Terms— Source-free universal domain adaptation, time series, benchmark, foundation models

## 1. INTRODUCTION

Source-Free Universal Domain Adaptation (SF-UniDA) consists of aligning a model pretrained on a source distribution to an unlabeled target distribution. Its main objectives are common classes alignment between the two distributions and detection of target-only (unknown) classes. This makes it applicable when source data cannot be shared due to privacy constraints [1]. When source data is available, one can instead resort to Universal Domain Adaptation (UniDA) [2] techniques.

UniDA and SF-UniDA have been extensively studied for image datasets [1, 3, 4, 5], while time series datasets remain significantly understudied. Two benchmarks have evaluated domain adaptation for time series datasets: ADATIME [6] addresses unsupervised domain adaptation (where both domains share the same label set), and UniDABench [7] evaluates the impact of time-series-specific backbone architectures for the UniDA setting. However, neither considers the SF-UniDA setting nor the use of foundation models as feature extractors. Our benchmark addresses both gaps within a unified framework for SF-UniDA.

SF-UniDA poses distinct challenges. For instance, without access to source data, adaptation must rely entirely on the source pretrained model, making the representational quality of its backbone all the more critical. The emergence of large pretrained foundation models for time series [8, 9, 10] raises a question that no prior work has addressed: can their embeddings serve as effective feature representations for domain adaptation? In addition, most SF-UniDA methods rely on a fixed and manually selected threshold to distinguish between common samples and unknown samples [3, 4, 5]. Although SF-UniDA applications in computer vision demonstrate high robustness over threshold selection, a recent study cast doubt on such robustness over time series [11] in UniDA. This raises questions about the validity of manually selected thresholds in SF-UniDA.

This paper makes three contributions. (1) We present the first benchmark for SF-UniDA on time series data, contextualizing the foundation model results against four time-seriesspecific backbone architectures (CNN, TFE [11], TSLANet [12], and S3 [13]). (2) We conduct thefirst study ofpretrained foundation models for time series domain adaptation, comparing MOMENT [8], Mantis [9], and Chronos [10]. (3) We identify and analyze threshold hypersensitivity as a key limitation of all existing SF-UniDA methods on time series, and propose a plug-in auto-thresholding that resolves it without requiring labeled target data.

## 2. PROBLEM FORMULATION

We consider a pretrained model $h = g \circ \phi _ { : }$ , with feature extractor $\phi : \mathcal { X }  \mathcal { Z }$ and classifier $g : \mathcal { Z }  \mathcal { V }$ , trained on a labeled source domain $\mathfrak { D } ^ { s } = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { n _ { s } }$ that is no longer accessible at adaptation time. The goal is to adapt h to an unlabeled target domain ${ \mathfrak { D } } ^ { t } = \{ x _ { i } ^ { t } \} _ { i = 1 } ^ { n _ { t } } .$ , drawn from a distribution $\mathcal { P } ^ { t } ( x , y ) \neq \mathcal { P } ^ { s } ( x , y )$ . We assume covariate shift: the marginal distributions differ while the labeling rule is preserved, i.e., $\mathcal { P } ^ { s } ( x ) \neq \mathcal { P } ^ { t } ( x )$ and $\mathcal { P } ^ { s } ( y | x ) = \mathcal { P } ^ { t } ( y | x )$

![](images/3f18e896b4e39e495419cca054f70cab13a72fdf9258ca9d77f73017d093f33b.jpg)  
Fig. 1: Overview of the SF-UniDA benchmark for time series. Task-specific backbone architectures and pretrained foundation models are evaluated as feature extractors under the SF-UniDA methods. Both fixed thresholding and auto-thresholding are applied to each SF-UniDA method. The performance is measured with H-score over 3 time series datasets.

Beyond distributional shift, the source and target label sets might not coincide [2]. They decompose into three disjoint subsets: the common label set $\mathcal { V } = \mathcal { V } ^ { s } \cap \mathcal { V } ^ { t }$ of classes shared by both domains, the private source label set $\overline { { \mathcal { V } } } ^ { s } = \mathcal { V } ^ { s } \setminus \mathcal { V }$ of source-only classes absent from the target, and the private target label set $\overline { { \mathcal { V } } } ^ { t } = \mathcal { V } ^ { t } \setminus \mathcal { V }$ of target-only unknown classes unseen during training. We focus on the open-partial DA setting where both $\overline { { \mathcal { V } } } ^ { s } \neq \emptyset$ and $\overline { { \boldsymbol { \mathcal { V } } } } ^ { t } \neq \boldsymbol { \emptyset }$ , the most complete and challenging scenario. This defines two key tasks: (i) aligning representations for common samples in Y across domains, and (ii) detecting target samples from Y and assigning them to an unknown category. This setting, referred to as Source-Free Universal Domain Adaptation (SF-UniDA), is practically important when sharing source data is prohibited by privacy or governance constraints. Multiple methods have been proposed for SF-UniDA in the image domain [3, 4, 5, 1], but none have been studied for time series applications. In addition, most of these methods rely on a fixed inference threshold to reject unknown samples, which have proven robust in the image domain, yet this robustness remains to be investigated for time series.

## 3. BENCHMARK DESIGN

This benchmark applies all recent SF-UniDA methods to time series datasets and evaluates them across a diverse set of taskspecific backbones and pretrained foundation models. We also investigate the robustness of the fixed inference threshold shared by all these methods in the time series setting, contrasting it against an auto-thresholding approach. The complete benchmark is illustrated in Fig. 1. Section 3.1 presents the backbones and foundation models, Section 3.2 describes the SF-UniDA methods, and Section 3.3 motivates and details the proposed auto-thresholding module. Our code is available at https://github.com/RomainMsrd/SF-UniDABe nch.

## 3.1. Backbones & Foundation Models

Comparison backbones. We use four time-series-specific architectures as baselines. 1D-CNN (three-block convolutional network) serves as the reference baseline. TFE [11] concatenates a Fourier-based encoder with a parallel CNN branch to jointly capture spectral and temporal features. TSLANet [12] combines learnable spectral filtering with gated convolutional mixing. S3Layer [13] segments the series into patches, reorders them via a learned priority vector, and stitches them back with a residual connection. Foundation models. Three foundation models are evaluated in this work. They have been trained to solve different tasks and have been chosen for their relatively small number of parameters. These foundation models are summarized in Table 1 and span a range of pretraining objectives and tasks.

MOMENT [8] is a general-purpose T5-style encoder pretrained via masked patch reconstruction on a large collection of public time series spanning multiple tasks (forecasting, classification, anomaly detection, and imputation). Its multi-task pretraining distinguishes it from Mantis, which is designed specifically for classification.

Mantis [9] is a ViT-based encoder designed specifically for time series classification. Its Token Generator Unit fuses three complementary views of each series (instancenormalized patches, first-order differences, and global statistics) and the encoder is pretrained via a contrastive objective, making it the most naturally suited model for the classification task at hand.

Chronos [10] adapts the T5 language model to time series forecasting by quantizing observations into discrete scalar tokens and training autoregressively on a large corpus of real and synthetic time series. Despite being designed for forecasting, its encoder embeddings are used here as a feature extractor, representing the extreme case of a model pretrained on a fundamentally different task than classification.

Table 1: Comparison of MOMENT, Mantis, and Chronos time series foundation models. General-purpose tasks include classification, forecasting, anomaly detection, and imputation.
<table><tr><td>Model</td><td>Type</td><td>Architecture</td><td>Param.</td><td>Input</td><td>Pre-training objective</td><td>Task</td></tr><tr><td>MOMENT [8]</td><td>Encoder T5</td><td></td><td>40M</td><td>Patches</td><td>Reconstruction</td><td>General-purpose</td></tr><tr><td>Mantis [9]</td><td>Encoder ViT</td><td></td><td>8M</td><td>Tokens</td><td>Contrastive learning</td><td>Classification</td></tr><tr><td>Chronos [10]</td><td></td><td>Seq2Seq T5 and LLM</td><td>8M</td><td>Scalar quantized</td><td>Autoregression</td><td>Forecasting</td></tr></table>

Table 2: Comparison of SF-UniDA methods. Threshold-free refers to inference only. “Weak” source-free indicates UMAD requires a dual-head architecture and auxiliary orthogonal loss during source pre-training.
<table><tr><td>Method</td><td>Key idea</td><td>Unknown detection</td><td>Source-free</td><td>Threshold-free (Inference)</td></tr><tr><td>UMAD [1]</td><td>Dual-head classifier consistency</td><td>Consistency + MixUP</td><td>Weak</td><td>√</td></tr><tr><td>GLC [3]</td><td>Global One Vs All + local k-NN</td><td>One Vs All clustering</td><td>Full</td><td>x</td></tr><tr><td>GLC++[4]</td><td>GLC + contrastive loss</td><td>One Vs All clustering</td><td>Full</td><td>x</td></tr><tr><td>LEAD [5]</td><td>Orthogonal feature decomposition</td><td>Gaussian Mixture Model</td><td>Full</td><td>x</td></tr></table>

## 3.2. SF-UniDA Methods

The considered SF-UniDA methods are summarized in Table 2 and further described below.

UMAD [1] trains a source model with a two-head classifier regularized by an orthogonality constraint, so that the two heads learn complementary decision boundaries. At adaptation and inference time, unknown samples are detected with an information-consistency score, defined as the inner product between the two heads’ softmax outputs.

GLC [3] proposes a Global and Local Clustering approach. It combines an adaptive one-vs-all global clustering algorithm to distinguish common and unknown target classes, with a local k-NN clustering strategy to mitigate negative transfer from unknown samples.

GLC++ [4] extends GLC by integrating a contrastive affinity learning strategy to overcome the limitation of uniform treatment of unknown data inherent to closed-set source architectures, enabling finer discrimination among distinct unknown categories.

LEAD [5] proposes a LEArning Decomposition framework that decouples target features into source-common and source-unknown components via orthogonal decomposition. Unknown samples are then identified by thresholding the norm of each sample’s source-unknown component, avoiding time-consuming iterative clustering.

All methods except UMAD rely on a manually selected threshold to reject unknown samples at inference. UMAD instead estimates its threshold from MixUp-interpolated target samples acting as synthetic unknowns, but requires a specific dual-head architecture and source pre-training objective, unlike other methods, which can adapt any off-the-shelf classifier. Therefore, UMAD is included for comparison while noting its slightly relaxed SF-UniDA assumption.

![](images/0f3b4460fb911a895710dfe1a0e877c098cf0cb909dcb642c05868fe675e0f6a.jpg)  
Fig. 2: Source model confidence distributions on target data (no adaptation) for known and unknown samples. Left: CNN / EDF (Time Series). Right: ResNet50 / Office-31 (Images).

## 3.3. Threshold Sensitivity in SF-UniDA

Strictly source-free SF-UniDA methods reject unknown target samples by thresholding the maximum softmax probability with a fixed manually selected value τ. In computer vision, sensitivity analyses report marginal H-score variations across broad ranges of τ , making this choice appear relatively harmless [3]. We argue this assumption should be revisited for time series, as dynamic auto-thresholding has already shown promise for unknown-class detection in this setting [11], suggesting that fixed thresholds may be less reliable.

We hypothesize that common and unknown confidence distributions are harder to separate due to noisier and less structured feature spaces producing less calibrated softmax outputs (see Fig. 2). Under this hypothesis, the H-score can become highly sensitive to τ, with small threshold changes causing large performance swings and narrow performance peaks that exhaustive grid search may miss. We therefore compare each SF-UniDA method with and without a drop-in auto-thresholding block replacing the fixed threshold, which additionally reduces the hyperparameter search space.

![](images/a32de9452cacf33e78222c871aa4502bd6409c0954cbea2499232d1ce2dda95b.jpg)

![](images/9d46f8eae3c4b30863e488c52bc7a19d3aa0c3fc3b2713840f8e53c5b502ee87.jpg)  
Fig. 3: Threshold sensitivity. Left: Mean H-score as a function of $\tau$ for GLC on time series (HAR) and image (Office) data with a CNN backbone and two foundation models, showing that the optimal threshold range is far narrower for time series. Right: Per-scenario H-score as a function of $\tau$ for GLC with a CNN backbone on HAR, illustrating that the optimal threshold also varies across adaptation scenarios.

Plug-in auto-thresholding module. To address this limitation, we propose a drop-in module applicable to any SF-UniDA method that replaces the fixed threshold with one estimated automatically from the unlabeled target data. After the adaptation step, the empirical distribution of classifier confidences $\{ s _ { i } \} _ { i = 1 } ^ { | \mathcal { D } _ { t r } ^ { t } | }$ , where $s _ { i } = \operatorname* { m a x } _ { c } p _ { i , c } ^ { t }$ , is computed on the unlabeled training target set $\mathcal { D } _ { t r } ^ { t } . \mathrm { A n }$ unsupervised histogrambased thresholding criterion $\mathcal { F }$ is then applied to this distribution to yield the optimal threshold:

$$
\tau ^ { * } = \underset { \tau } { \mathrm { a r g m i n } } \mathcal { F } \Big ( \{ s _ { i } \} _ { i = 1 } ^ { | \mathcal { D } _ { t r } ^ { t } | } \Big ) .\tag{1}
$$

Any criterion that partitions a one-dimensional distribution into two groups without requiring labels is a valid instantiation of ${ \mathcal F } .$ We consider three standard choices: Otsu’s criterion [14] which minimizes intra-class variance, Yen’s criterion [15] which maximizes the entropic correlation of each partition, and Li’s criterion [16] which minimizes crossentropy to group means. The resulting $\tau ^ { * }$ is computed once after adaptation and applied fixed at test time, adding negligible overhead, requiring no labels, and no modification to the adaptation procedure, making it a true plug-in compatible with any fully SF-UniDA method [11].

## 4. EXPERIMENTS

## 4.1. Experimental Setup

SF-UniDA methods are evaluated on the datasets used in [7]: HAR and HHAR are human activity recognition datasets with six activity classes (walking, sitting, biking, etc.) collected from body-worn sensors, both use 128-timestep windows and are multivariate (9 and 3 channels, respectively). EDF is a sleep stage classification dataset with five sleep stages from EEG recordings. We also consider the image dataset Office31 [17] for comparison with time series. Backbone hyperparameters are fixed across all experiments, foundation models are frozen, and a trainable single-hiddenlayer MLP is appended on top to let the SF-UniDA method adjust the feature space without modifying the foundation model weights. The performance is measured by the H-score $( 2 A _ { C } A _ { U } ) / ( A _ { C } + A _ { U } )$ [18], where $A _ { C }$ and $A _ { U }$ denote accuracy on common and unknown target classes, respectively.

All SF-UniDA methods rely on their own set of hyperparameters, which generally include a fixed threshold $\tau$ during inference, whose selection deeply impacts performance. Following [7], hyperparameters are selected by Bayesian optimization over $N _ { r }$ trials. Given that domain adaptation datasets are structured as a collection of d domains $\mathbf { D } = \{ \mathcal { D } ^ { 1 } , \ldots , \mathcal { D } ^ { d } \}$ , where any ordered pair $( \mathcal { D } ^ { s } , \mathcal { D } ^ { t } )$ with $s \neq t$ defines an adaptation scenario, the full set of scenarios $s$ can be partitioned into a validation subset $S _ { v a l }$ of size $N _ { v a l }$ and an evaluation subset $\boldsymbol { S _ { e v a l } }$ of size $N _ { e v a l }$ for final reporting. This scenario-level separation ensures that the H-score signal used for selection does not contaminate the final evaluation. We use 20 training epochs per model, $N _ { r } ~ = ~ 2 0$ Bayesian hyperparameter trials, $N _ { v a l } ~ = ~ 5$ validation scenarios for HAR and $N _ { v a l } ~ = ~ 3$ for HHAR and EDF, with final results over $N _ { e v a l } = 1 0$ test scenarios each averaged over 10 seeds. Note that this benchmark protocol requires labeled target validation scenarios.

## 4.2. Threshold Sensitivity Analysis

The Fig. 3 (left) reveals that the optimal fixed threshold for time series occupies a much narrower range than for image data. This is especially pronounced for foundation models such as Mantis and Chronos, whose H-score collapses to nearly zero across most of the range and only spikes sharply. Even with a CNN backbone on time series, although the Hscore increases more gradually from $\tau = 0 . 3 \mathrm { t o } \ \tau = 0 . 9 .$ it then drops abruptly to zero, with no flat plateau. This contrasts with GLC using a ResNet backbone on image data, which maintains a stable and high H-score over a wide range of thresholds before degrading near the extremes. The right panel further shows that the optimal threshold varies across adaptation scenarios, so a fixed threshold tuned at the dataset level may not generalize well within the same dataset.

Table 3: H-score (%) with Auto-Thresholding (Yen’s criterion) over 10 seeds. Parentheses show gain vs. fixed thresholding: (↑) gain, (↓) loss, (=) no change. Bold and italic indicate the best and second-best architectures for each method, respectively. UniJDOT in gray is a UniDA method for which backbone results marked with <sup>∗</sup> are reported directly from [7].
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Methods</td><td colspan="4">Backbones</td><td colspan="3">Foundation Models</td></tr><tr><td>CNN</td><td>TFE</td><td>S3</td><td>TSLANet</td><td>Mantis</td><td>Moment</td><td>Chronos</td></tr><tr><td rowspan="5">HAR</td><td>UniJDOT</td><td>61.0*(−)</td><td>64.6* (−)</td><td>54.1* (−)</td><td>59.7*(−)</td><td>82.5 (−)</td><td>30.1 (−)</td><td>72.2 (-)</td></tr><tr><td>UMAD</td><td>52.7 (↓ 2.5)</td><td>64.6 (↑ 29.7)</td><td>36.1 (↓ 7.5)</td><td>43.1 (↑ 11.5)</td><td>51.2 (↑ 16.2)</td><td>24.2 (↑ 11.4)</td><td>46.0 (↑ 13.3)</td></tr><tr><td>GLC</td><td>46.2 (↑ 4.2)</td><td>42.2 (↑ 29.0)</td><td>52.3 (↑ 11.2)</td><td>54.0 (↑ 5.8)</td><td>53.4 (↑ 15.8)</td><td>16.7 (↑ 16.6)</td><td>50.6 (↑ 28.9)</td></tr><tr><td>GLC++</td><td>44.7 (↑ 5.4)</td><td>36.4 (↑ 20.8)</td><td>50.7 (↑ 5.0)</td><td>54.1 (↑ 17.8)</td><td>59.6 (↑ 51.1)</td><td>13.6 (↑ 10.8)</td><td>50.9 (↑ 33.8)</td></tr><tr><td>LEAD</td><td>43.8 (↑ 7.2)</td><td>42.3 (↑ 28.1)</td><td>50.4 (↑ 4.4)</td><td>53.1 (↑ 15.5)</td><td>57.8 (↑ 13.3)</td><td>13.7 (↑ 11.5)</td><td>50.6 (↑ 30.7)</td></tr><tr><td rowspan="5">HHAR</td><td>UniJDOT</td><td>56.6*(−)</td><td>61.2* (−)</td><td>57.8*(−)</td><td>55.7*(-)</td><td>59.3 (−)</td><td>46.3 (-)</td><td>57.6 (-)</td></tr><tr><td>UMAD</td><td>53.7 (↑ 0.5)</td><td>42.6 (↑ 21.1)</td><td>51.5 (↓ 2.0)</td><td>41.1 (↑ 2.7)</td><td>28.6 (↑ 4.5)</td><td>34.1 (↓ 10.2)</td><td>40.2 (↑ 1.6)</td></tr><tr><td>GLC</td><td>43.2 (↑ 14.1)</td><td>45.3 (↑ 13.4)</td><td>45.5 (↑ 3.2)</td><td>29.2 (↑ 1.4)</td><td>39.4 (↑ 27.1)</td><td>31.3 (↑ 17.4)</td><td>31.6 (↓ 3.8)</td></tr><tr><td>GLC++</td><td>37.3 (↑ 10.3)</td><td>44.9 (↑ 13.2)</td><td>47.7 (↑ 7.9)</td><td>29.6 (↑ 2.7)</td><td>37.7 (↑ 27.0)</td><td>26.8 (↑ 18.7)</td><td>25.3 (↓ 14.9)</td></tr><tr><td>LEAD</td><td>41.0 (↑ 12.1)</td><td>45.6 (↑ 13.0)</td><td>52.2 (↑ 11.2)</td><td>17.8 (↑ 10.0)</td><td>40.2 (↑ 22.9)</td><td>35.0 (↑ 18.5)</td><td>29.4 (↑ 25.3)</td></tr><tr><td rowspan="5">EDF</td><td>UniJDOT</td><td>44.3* (−)</td><td>55.6* (−)</td><td>50.4* (一)</td><td>55.0*(−)</td><td>42.4 (−)</td><td>37.7 (-)</td><td>38.1 (−)</td></tr><tr><td>UMAD</td><td>46.5 (↑ 4.9)</td><td>32.9 (↓ 1.4)</td><td>48.1 (↑ 2.5)</td><td>49.3 (↑ 7.7)</td><td>29.3 (↑ 10.9)</td><td>30.5 (↑ 5.6)</td><td>34.8 (↑ 4.9)</td></tr><tr><td>GLC</td><td>46.7 (↑ 1.8)</td><td>24.5 (↓ 14.6)</td><td>38.8 (↑ 1.3)</td><td>31.8 (↑ 2.9)</td><td>30.9 (↑ 10.7)</td><td>31.3 (↑ 6.8)</td><td>30.4 (↑ 2.9)</td></tr><tr><td>GLC++</td><td>49.5 (↑ 1.6)</td><td>31.3 (↓ 10.8)</td><td>43.8 (↑ 1.7)</td><td>32.6 (↑ 2.9)</td><td>27.4 (↑ 12.4)</td><td>28.3 (↑ 2.5)</td><td>29.6 (↑ 1.5)</td></tr><tr><td>LEAD</td><td>46.2 (↑ 0.2)</td><td>36.3 (↑ 7.4)</td><td>44.6 (=)</td><td>33.7 (↑ 4.8)</td><td>28.9 (↑ 10.2)</td><td>37.0 (↑ 8.4)</td><td>27.2 (↑ 3.1)</td></tr></table>

## 4.3. Benchmark Results

Table 3 reports H-scores with gains over the best fixed thresholds found via hyperparameter search in parentheses. UMAD frequently leads among SF-UniDA methods as a weakly source-free method (i.e., requiring a specific architecture and source pretraining), while foundation models with a trainable MLP do not consistently exceed standard backbones: Mantis is competitive on HAR, whereas on HHAR and EDF, classical backbones generally match or outperform all foundation models. Moment’s low H-scores for HAR are traceable to poor source model accuracy despite equivalent hyperparameter tuning rather than to the adaptation procedure. The top non-source-free method, UniJDOT [11] outperforms SF-UniDA, confirming a persistent gap with UniDA.

Auto-thresholding generally improves performance across methods and backbones, with the largest gains for foundation models. On HAR, Mantis achieves the highest foundationmodel H-scores, with gains exceeding 50 points for GLC++ and 10 points otherwise, while Chronos gains around 30 points for most methods, confirming that fixed thresholds severely underestimate foundation models. On HHAR and EDF, gains remain predominant, with isolated losses for Chronos on HHAR and TFE on EDF. The proposed plug-in also outperforms UMAD’s own auto-thresholding on average.

As shown in Table 4, Yen’s criterion ranks first or second while Otsu’s and Li’s remain competitive, confirming that other criteria can be used with no or little performance loss. These results demonstrate that the proposed auto-thresholding module effectively addresses the critical threshold sensitivity issue in time-series SF-UniDA.

Table 4: H-score (%) over 10 seeds for GLC on HAR across backbones and thresholding methods
<table><tr><td rowspan="2">Threshold</td><td colspan="3">Backbones</td><td colspan="2">Foundation M.</td></tr><tr><td>CNN</td><td>TFE</td><td>S3Layer</td><td>Mantis</td><td>Chronos</td></tr><tr><td>yen</td><td>46.2</td><td>42.2</td><td>52.3</td><td>53.4</td><td>50.6</td></tr><tr><td>otsu</td><td>44.5</td><td>42.9</td><td>49.8</td><td>45.7</td><td>42.9</td></tr><tr><td>li</td><td>42.0</td><td>41.3</td><td>55.3</td><td>48.0</td><td>40.3</td></tr></table>

## 5. CONCLUSION

We presented the first SF-UniDA benchmark for time series, evaluating four backbones and four SF-UniDA methods across three datasets, as well as pretrained foundation models as feature extractors for time-series domain adaptation. Results show that S3Layer ranks among the top classical backbones, that SF-UniDA lags behind its UniDA counterpart, and that frozen foundation model features with an additional MLP do not systematically outperform task-specific backbones, although other fine-tuning strategies may yield different results. We further identified threshold hypersensitivity as a critical and underexplored limitation of existing SF-UniDA methods on time series, and proposed a label-free plug-in auto-thresholding module that consistently improves performance across all methods and backbones, with gains exceeding 50 H-score points for some foundation models. Finally, backbone selection currently has a larger impact on performance than the choice of SF-UniDA method, suggesting that developing time-series-specific SF-UniDA approaches is a promising direction for future work.

## 6. REFERENCES

[1] Jian Liang, Dapeng Hu, Jiashi Feng, and Ran He, “Umad: Universal model adaptation under domain and category shift,” 2021.

[2] Kaichao You, Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan, “Universal domain adaptation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[3] Sanqing Qu, Tianpei Zou, Florian Rohrbein, Cewu Lu,¨ Guang Chen, Dacheng Tao, and Changjun Jiang, “Upcycling models under domain and category shift,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[4] Sanqing Qu, Tianpei Zou, Florian Rohrbein, Cewu¨ Lu, Guang Chen, Dacheng Tao, and Changjun Jiang, “Glc++: Source-free universal domain adaptation through global-local clustering and contrastive affinity learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[5] Sanqing Qu, Tianpei Zou, Lianghua He, Florian Rohrbein, Alois Knoll, Guang Chen, and Changjun¨ Jiang, “Lead: Learning decomposition for source-free universal domain adaptation,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024, IEEE.

[6] Mohamed Ragab, Emadeldeen Eldele, Wee Ling Tan, Chuan-Sheng Foo, Zhenghua Chen, Min Wu, Chee-Keong Kwoh, and Xiaoli Li, “Adatime: A benchmarking suite for domain adaptation on time series data,” ACM Transactions on Knowledge Discoveryfrom Data, Sept. 2023.

[7] Romain Mussard, Fannia Pacheco, Maxime Berar, Gilles Gasso, and Paul Honeine, “Universal domain adaptation benchmark for time series data representation,” in 2025 33rd European Signal Processing Conference (EUSIPCO), 2025.

[8] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski, “Moment: A family of open time-series foundation models,” in Proceedings of the 41st International Conference on Machine Learning, 2024, ICML’24.

[9] Vasilii Feofanov, Marius Alonso, Songkang Wen, Romain Ilbert, Hongbo Guo, Malik Tiomoko, Lujia Pan, Jianfeng Zhang, and Ievgen Redko, “Mantis: Lightweight calibrated foundation model for userfriendly time series classification,” in 1st ICML Workshop on Foundation Modelsfor Structured Data, 2025.

[10] Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang, “Chronos: Learning the language of time series,” Transactions on Machine Learning Research, 2024.

[11] Romain Mussard, Fannia Pacheco, Maxime Berar, Gilles Gasso, and Paul Honeine, “Deep joint distribution optimal transport for universal domain adaptation on time series,” in 2025 International Joint Conference on Neural Networks (IJCNN), 2025.

[12] Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu, and Xiaoli Li, “Tslanet: Rethinking transformers for time series representation learning,” in Proceedings of the 41st International Conference on Machine Learning, 2024, ICML’24.

[13] Shivam Grover, Amin Jalali, and Ali Etemad, “Segment, shuffle, and stitch: A simple layer for improving time-series representations,” in The Thirty-Eighth Annual Conference on Neural Information Processing Systems, Nov. 2024.

[14] Nobuyuki Otsu, “A threshold selection method from gray-level histograms,” IEEE Transactions on Systems, Man, and Cybernetics, Jan. 1979.

[15] Jui-Cheng Yen, Fu-Juay Chang, and Shyang Chang, “A new criterion for automatic multilevel thresholding,” IEEE Transactions on Image Processing, 1995.

[16] C.H. Li and C.K. Lee, “Minimum cross entropy thresholding,” Pattern Recognition, vol. 26, no. 4, pp. 617– 625, Apr. 1993.

[17] David Hutchison, Takeo Kanade, Josef Kittler, Jon M. Kleinberg, Friedemann Mattern, John C. Mitchell, Moni Naor, Oscar Nierstrasz, C. Pandu Rangan, Bernhard Steffen, Madhu Sudan, Demetri Terzopoulos, Doug Tygar, Moshe Y. Vardi, Gerhard Weikum, Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell, “Adapting visual category models to new domains,” in Computer Vision – ECCV 2010. 2010, Springer Berlin Heidelberg.

[18] Bo Fu, Zhangjie Cao, Mingsheng Long, and Jianmin Wang, “Learning to detect open classes for universal domain adaptation,” in Computer Vision – ECCV 2020. Springer International Publishing, 2020.