# Robust Neural Stimulation Response Modeling Through Meta-Learning and Pretraining

Matthew J Bryan<sup>1,2,3</sup>, Daniel C Muir<sup>1,2,3</sup>, Felix Schwock<sup>2,3,5</sup>, Azadeh Yazdan-Shahmorad<sup>2,3,4,5</sup>, Rajesh P N Rao<sup>1,2,3</sup>

Abstract— Objective: Model-based closed-loop neural stimulation holds promise for therapeutic applications ranging from Parkinson’s disease to sensory restoration, but deployment has been limited by two obstacles: (1) forecasting models for predicting the consequences of stimulation fail catastrophically on a meaningful fraction of sessions, and (2) per-session calibration requirements are often incompatible with clinical constraints. We address both by demonstrating, for the first time, that meta-learning and pretraining can be applied to neural stimulation response modeling. Methods: Temporal basis function models (TBFMs) have previously been used to forecast statedependent neural responses to stimulation. We extend the application of TBFMs with cross-session pretraining using a novel architecture and algorithm based on modelagnostic meta-learning (MAML), and evaluate the approach on 40 sessions of optogenetic stimulation in primary somatosensory and motor cortex of two non-human primates. Results: Meta-learning substantially reduces catastrophic forecast failure: for a 1k calibration set size, sessions with test $\mathsf { R } ^ { 2 } < 0 . 0 5$ drop from 16/40 (single-session training) to 1/40 (MAML-pretrained), and prediction intervals become significantly narrower (Brown-Forsythe p < 0.05). Calibration requirements are reduced by 50–90% at matched accuracy, enabling experiments otherwise infeasible within clinical session-time constraints. Conclusion: Our results demonstrate that cross-session structure in stimulation responses is consistent enough to support pretraining, providing the first empirical evidence that meta-learning approaches are viable for neural stimulation. Significance: The robustness and sample efficiency gains directly ad-

dress known obstacles to deploying model-based stimulation controllers. Our results motivate community efforts to assemble standardized multi-site stimulation datasets and to further explore meta-learning for robust closed-loop stimulation.

Index Terms— Brain–computer interfaces, closed-loop stimulation, meta-learning, neural decoding, temporal basis function models, transfer learning

## I. INTRODUCTION

efficiency [1], and reduced side effects [2] compared to openloop stimulation. It uses sensed brain activity and potentially data from other sensors to adapt stimulation in real time (e.g. [3], [4]), allowing it to take advantage of “state-dependence” [5], i.e., the dependence of the stimulation response on neural or behavioral states. Clinical open-loop systems have been successfully demonstrated for sensory function restoration, symptom alleviation and targeted rehabilitation for a variety of neurological conditions, e.g., cochlear implants for deafness [6] to deep brain stimulation (DBS) for Parkinson’s disease (PD) [1]. Closed-loop stimulation seeks to improve on these, but remains difficult due to myriad technical challenges including controller latency, precise characterization and modeling of neural responses to stimulation, and reliable calibration [18].

Adaptive closed-loop stimulation guided by artificial neural networks (ANN) — often termed neural co-processing [3], [8] - offers a path toward stimulation controllers that adjust dynamically to a patient’s neural state, individual differences, and disease-specific features. Realizing such a neural coprocessor for clinical use however will require significant improvements in robust, sample efficient, accurate, and lowlatency model-based stimulation controllers.

Temporal basis function models (TBFMs) [18] seek to address these challenges by providing single-trial, multi-step, spatiotemporal forecasts of the state-dependent neural response to stimulation, while exhibiting high sample efficiency and low latency. They improve on the prior state-of-theart which, for example, did not consider state-dependence [9], did not operate with low enough latency for control, or offer sufficient sample efficiency for practical use in clinical settings [3]. TBFMs ameliorate these issues: Bryan et al. [18] demonstrated their potential to be used in the span of a 1-2 hour experimental session, including data collection for calibration and model training time.

In this article, we advance TBFMs by showing that pretraining on data from past sessions and using model-agnostic metalearning (MAML) [10] substantially improves the robustness of stimulation response forecasting. Pretrained TBFMs exhibit significantly fewer sessions where accuracy degrades catastrophically due to noise or other nuisance variables, addressing a known obstacle to deploying model-based stimulation controllers in clinical and experimental settings. The same pretraining approach additionally allows TBFMs to adapt to future sessions with calibration datasets 50−90% smaller than those needed for single-session TBFMs, enabling experiments otherwise infeasible within single session-time constraints.

These gains are notable given that TBFMs were previously demonstrated to have sample efficiency comparable to basic linear state-space models while exhibiting significantly higher prediction accuracy [18]. The pretraining and meta-learning approach we introduce here allows experimenters to (1) take advantage of previously collected stimulation data, (2) reduce the frequency of failed experimental sessions and (3) concentrate significantly more experiment time on exploring stimulation parameters in a closed-loop fashion rather than on calibration.

Meta-learning involves the pretraining of models which are easily adapted to unseen tasks [10], in our case stimulation experiment sessions. In the case of MAML specifically, model parameters are pushed into regimes where adaptation to an unseen session can be done with minimal training data. The adaptation process is referred to as “test-time adaptation” (TTA), and involves fine-tuning some part of the model for the unseen session using a small calibration data set. This pretraining improves robustness by biasing per-session models towards patterns identified across many previous sessions, anchoring the learning process and reducing the degenerate effects of nuisance variables. It has been applied recently to brain-computer interfacing (BCI) to minimize calibration data collection [11], but to our knowledge, neither MAML nor meta-learning more generally have previously been shown to be effective for neural stimulation modeling.

Meta-learning works best when there is similarity between sessions which pretraining can leverage. With varying numbers of usable electrodes, differing subjects and electrode placements, and varying stimulation timings and locations, there are many sources of dissimilarity between sessions. Yet local stimulation responses tend to be stereotyped, some spatial autocorrelation statistics may be similar (see example in Fig 1), and similarly for some temporal autocorrelations [19]. We show for the first time that pretraining and meta-learning can take advantage of these similarities, and that its adaptation process can overcome the cross-session dissimilarities, allowing the model to adapt to unseen sessions while exhibiting accuracy and robustness superior to single-session models.

In summary, this article provides:

• A brief introduction to TBFMs, based on prior work;

• A novel machine learning architecture based on TBFMs which allows for cross-session pretraining;

• A meta-learning algorithm for pretraining such models;

• A test-time adaptation algorithm for unseen sessions;

• A free-to-use open source implementation;

• Results demonstrating:

1) A significant improvement in robustness, defined as the the model’s accuracy on the highest noise sessions;

2) Significant savings in calibration data collection; and

3) Comparable or improved forecast accuracy relative to single-session models across all calibration data set sizes.

## II. METHODS

## A. Dataset

We evaluate the proposed method using a previously published µECoG dataset [13] comprising 40 optogenetic stimulation sessions collected from two rhesus macaques. In these experiments, excitatory neurons in left primary somatosensory (S1) and primary motor (M1) cortex were transduced to express the red-shifted channelrhodopsin variant C1V1. A semitransparent 96-channel µECoG grid enabled simultaneous recording of cortical local field potentials (LFPs) while paired optical pulses were delivered through two fiber-optic probes (Figs. 1–3) [14]–[17]. Each fiber optic was positioned adjacent to a specific ECoG electrode. Although the optical stimulation sites differed between sessions, they remained fixed within any given session. The number of electrodes meeting our quality criteria per session ranged from 42 to 94 (mean 78.6, SD 13.2), with electrodes being excluded based on impedance values measured during non-stimulation periods. Further descriptions of data acquisition and filtering procedures are provided in [12].

To suppress line noise, we applied an infinite impulse response (IIR) notch filter at 60, 180, and 300 Hz, using quality factors (Q) of 20, 60, and 100, respectively. Results in this manuscript focus on the broadband timedomain signals; however, parallel analyses using bandpassand time–frequency–filtered versions of the data are available upon request. Additional methodological details appear in [7], [12], [13].

In each recording session, paired light pulses were delivered every 200ms in five consecutive 10-minute stimulation blocks, yielding approximately 15k pulse pairs per session. Individual pulses were 5ms in duration, and the inter-pulse interval within each pair was either 10ms, 30ms, or 100ms (session dependent). Stimulation blocks were interleaved with 10-minute rest periods during which neural activity was recorded without optical stimulation. During the experiments the animals were in an awake state watching cartoons.

## B. Introduction to temporal basis function models (TBFMs)

This section provides a brief introduction to single-session TBFMs based on our early work [18], and a significantly expanded analysis of TBFMs in our journal article [7]. A more detailed description as well as a reference PyTorch implementation can be found here: https://github.com/ mmattb/py-tbfm. This implementation is fit for use on a variety of stimulation modeling tasks, and is free to use under an open source license. The new extension of the singlesession approach to multi-session is in Section II-C.

![](images/54c2eea2b29433f2290341431e463fbd7604f744853b836b589d1c7e78f34824.jpg)

Fig. 1. Stimulation response across the µECog array. Heatmap indicates LFP values (Gaussian-smoothed and Z-scored, single session average) at 10ms after the second stim pulse onset. Missing electrodes were given false values for graphing. Stim locations varied between sessions.  
![](images/010435c5a4afd0344768fc622e6bb2eb226901e01d98a6091351f91f6103b520.jpg)  
Fig. 2. Data windowing Trials are 184ms separated by a 16ms buffer, ∼ 15k trials per session. See Figure 3 for legend.

1) Key design goals: We constructed the TBFM to solve several key challenges with machine learning approaches to neural stimulation modeling:

1) Sample efficiency: Closed-loop experiments can collect only modest amounts of stimulation data due to safety and timing constraints; models must therefore learn effectively from small datasets.

2) Training efficiency: Computationally expensive models and learning algorithms limit the types of adaptive protocols or rapid-iteration experiments we can run. Fast training enables broader experimental flexibility.

![](images/6a72406d2eb2767f08359318e6537fd90ff4a3a4b8191e12150ace035d0657d5.jpg)  
Fig. 3. Average response across many trials for a single electrode within one session. The TBFM model predicts the stimulation-evoked activity (y) in each trial using the first 20 ms of recorded neural data (x, known as the “runway”). The initial optical pulse occurs at 40 ms (blue dashed line); we assume there is a 20 ms control-loop delay between measuring the brain state at 20 ms and computing parameters for the stimulation (see Sec. II-B.1). Note that the TBFM forecasts all channels simultaneously using the runway for all channels, but we depict only one channel here.

3) Low-latency inference: Real-time stimulation requires predictions that remain valid within real-time control constraints. If model latency exceeds the forecasting horizon (e.g., a 50ms prediction with 75ms compute time), the controller acts on stale information and may not be able to effectively shape neural activity better than open-loop.

2) Architecture: TBFMs provide single-trial, multi-step predictions of the neural response to stimulation. The model takes as input a short window of recently recorded neural activity—referred to as the “runway”, together with the stimulation parameters. For this study, the runway consists of 20ms of neural data, and stimulation is assumed to be delivered 20ms later. This offset reflects an estimated worst-case control-loop latency, noting that some closed-loop systems achieve substantially lower delays (e.g., 7.5ms in [20]). During training, the TBFM is tasked with forecasting activity up to a horizon of t=164ms (Fig. 2). In closed-loop demonstrations, only a part of this prediction window may be used, depending on the specific control objective [18].

In our dataset, within a given session, stimulation was applied at fixed electrode sites with fixed stimulation parameters. As a result, the primary task of the controller trained on these data is to decide when to apply stimulation based on brain activity, rather than picking other parameters such as stimulation power or pulse width. Our pretrained TBFMs however must adapt to differing stimulation parameters because they vary across sessions. Specifically, the inter-pulse interval and target stimulation locations varied across sessions.

Fig 4 depicts the architecture of the TBFM (a more detailed description can be found in [18]). In brief, the single-session TBFM consists of:

1) A “stimulation descriptor” which is a tensor that encodes the stimulation parameters in some manner - their spatial pattern, timing, power, etc.

2) A basis vector generator which receives as input any covariates useful for the forecast. In our stimulation setting, this includes the stimulation descriptor (see above). It outputs a set of basis vectors which the TBFM shares across all channels to make the forecast.

3) The “runway” as described above.

4) Channel-wise means and standard deviations used for Z-scoring the runway data.

5) An affine basis weight estimator which estimates the per-channel, time-invariant weights for each basis using the normalized runway as input.

6) The forecast, defined as a weighted sum of the basis vectors and the last runway element for the given channel. That is, for a single channel c the forecast becomes: $\begin{array} { r } { \hat { y _ { c } } = x _ { c , r } \mathbf { 1 } + \sum _ { i = 1 } ^ { b } W ( X ) _ { c , i } B _ { i , * } } \end{array}$ , where B is the matrix of basis vectors, b is the basis count, W is the weight matrix, r is the runway length (see Fig 4).

3) Training: TBFMs are trained end-to-end with supervised backpropagation. For each channel, we estimate the mean and variance from the training set and apply Z-scoring using those fixed statistics during inference. The loss consists of the $L _ { 2 }$ prediction error and a Frobenius-norm regularization on the

![](images/07c990d9d51221eb47da3c4c0cdf320199f0ac8a7215965dd0ac4931acbb6e8f.jpg)  
Fig. 4. TBFM architecture

basis weights.

Training uses mini-batches of 5k trials drawn from the early portion of each session, while evaluation is performed on the final 2.5k trials to assess robustness to within-session drift. This mimics a realistic deployment scenario where a model is calibrated early and applied later in the session. Optimization proceeds until the loss stabilizes, typically within 15k epochs.

4) State dependency of predictions: Fig 5 depicts a TBFM’s brain state-dependent predictions for a single channel and single session, where the TBFM was trained on that single session’s data alone. For graphing, training set trials are binned into quartile “initial states” based on their value at the black dashed line, which is the last runway value. The pink “high” state exhibits a significantly different stimulation response than the others - one example of the brain state-dependency of the stimulation response present throughout these data. In [7] we present statistical evidence that this sort of state-dependency appears widely across the dataset and is statistically significant.

5) Model compilation: If stimulation parameters are drawn from a set of values - e.g. five different power levels - then an optimization is available to speed up inference time; we refer to it as “compilation.” Here the basis set for each parameter set is computed by forward pass through the basis generator, and saved in a dictionary. At inference time, the bases for that parameter set are pulled from the dictionary rather than computed on-the-fly, saving significant processing time, ≈ 34% for a typical session. We reuse this idea below.

![](images/9fe71cdb6e3f579812ae25cabe22439ec3d049f738e3dd8455445df8f15dada6.jpg)  
Fig. 5. Mean neural response and mean prediction for four binned initial states. Single channel, single session. Stimulation onset is at t=40ms. y is the mean stimulation response for trials in the given initial state bin, yˆ is the mean forecast for the same trials.

6) Performance comparison to linear state space models and recurrentneural networks: In [18] we compared TBFMs to two other models which broadly represent common approaches to modeling neural dynamics, and more generally dynamical systems with control. The first was a simple linear state space model (LSSM) similar to one used previously for modeling neural responses to stimulation [9]. The other is a complex nonlinear recurrent neural network based on long short term memory (LSTM) cells [22], which is strictly more expressive than the LSSM.

In brief: we found TBFMs to exhibit inference latency two orders of magnitude lower than either of the comparison models, while having the highest overall accuracy. TBFMs have sample efficiency slightly better than LSSMs and significantly better than the more complex LSTM model. As a result, TBFMs did not exhibit a trade-off between efficiency and accuracy on these data.

7) TBFMs for closed-loop neural stimulation: Once we have built a TBFM we can specify a stimulation controller using approaches such as model predictive control (MPC) or modelbased reinforcement learning (MBRL). In [18] we demonstrated TBFMs on two simulated closed-loop tasks: 1) applying stimulation only when key neural features are forecast to occur in the near future; and 2) timing stimulation to drive neural activity towards target dynamics. The first task reflects closed-loop strategies that deliver stimulation when specific neural states are detected, as in [21], which triggers stimulation at particular beta-phase events, or [1], which adapts stimulation based on biomarkers of Parkinsonian symptoms. The second task parallels work in visual prostheses, where stimulation is used to drive neural activity toward patterns associated with particular visual inputs.

## C. A temporal basis function model (TBFM) with meta-learning

This paper builds on the single-session TBFMs by illustrating how they can be trained to generalize across many related experiment sessions, allowing for significant improvements in sample efficiency. The overall architecture is depicted visually in Fig 6. We provide a table of ablations in III-F to provide insight on the reasoning behind this design. The architecture extends the single-session model as follows:

1) Per-session normalizers are not based on Z-scoring, but rather per-channel median and interquartile range (IQR) calculated from the training set of the given session. That is: the per-channel Z-score is replaced with: $\frac { X _ { c } - Q _ { 5 0 } } { Q _ { 7 5 } - Q _ { 2 5 } }$ where $Q _ { p }$ indicates the $p t h$ quantile of that channel’s data from the training dataset. We found this to be marginally more robust than Z-scoring when working with small calibration data sets.

2) Per-session linear autoencoders which map all sessions into a common latent space of dimensionality l. These provide two functions: 1) accounting for dimensionality differences due to differing numbers of missing channels between sessions; and 2) aligning the channels of the sessions in a latent space so that a common TBFM may be shared between them. We warm start the autoencoders with PCA, and a zero bias but refine them during training (see below). PCs are calculated from the session’s training data.

3) A single session-shared TBFM which lives in the latent space.

4) A basis weight normalization step which passes the basis weights through a hyperbolic tangent and $L _ { 2 }$ normalization to force each channel’s weight vector onto the unit sphere. This helps to stabilize training.

5) Per-session resting state context $c _ { s } ^ { r e s t }$ . These context variables customize the bases generated by the TBFM to be more suitable for a particular session. The resting state context is constituted by autocorrelation statistics calculated on resting/baseline data gathered prior to stimulation being attempted. This is an optional covariate which informs the optimal basis shapes for the session (explained further in Section II-C.1).

6) Per-session stimulation context $c _ { s } ^ { s t i m }$ . This vector is learned through our MAML-based optimization process (see below), and is passed into an MLP which reshapes the bases to customize them for the given session. The MLP outputs a low-rank matrix which is summed into the basis matrix - an approach known as low-rank adaptation (LoRA) [23].

7) A training procedure (described below) based on model-agnostic meta-learning (MAML) which optimizes a base TBFM to generalize well across sessions with minimal calibration data, while also optimizing the autoencoders and stimulation contexts $c _ { s } ^ { s t i m }$

8) A test-time adaptation (TTA) procedure (described below) which fine tunes a pretrained TBFM for an unseen session using a small set of stimulation examples, known as the calibration data set. This is made efficient by the training procedure in the prior step, and is the primary way we take advantage of pretraining for unseen sessions.

1) Per-session resting state context: In a previous paper [19], we showed that autocorrelation statistics of baseline data were descriptive of the predictability of the stimulation data from the same session. Specifically, the average absolute value of the autocorrelation function (A-ACF) calculated on baseline data correlated highly with the performance of all forecast model types we attempted on the stimulation data, including TBFMs. Autocorrelation is a proxy for the $R ^ { 2 }$ of a linear autoregressive forecast, and so it captures the (linear) predictability of a time series. It can also indirectly capture the signal-to-noise ratio (SnR) of the series under certain assumptions.

We leverage A-ACFs as covariates to help shape the temporal bases for a given session. While this is an optional aspect of our approach, it illustrates how additional sources of information could be leveraged to help shape a session’s predictions. Conceptually, lower autocorrelations often mean that forecasts should be more conservative, and should revert quickly towards the channel mean. By passing A-ACF statistics to the TBFM, we can shape its bases to reflect a more conservative forecast if a channel or session have high noise.

For our case we measure A-ACF for each channel individually. $c _ { s } ^ { r e s t }$ then becomes a vector containing the 25th, 50th and 75th percentile A-ACF among all channels, i.e., the interquartile range of the A-ACFs. This context vector is calculated in advance on baseline data before any stimulation data is gathered.

2) Per-session stimulation context: An additional context vector $c _ { s } ^ { s t i m } \in \mathbb { R } ^ { 1 5 }$ is optimized during training and during TTA to customize the TBFM’s bases for that particular session. While sessions have similarities in their stimulation responses, the responses also have unique session-specific characteristics and bases need to be fine tuned accordingly.

The stimulation context vector is passed to an MLP which generates a low-rank correction summed into the matrix of bases. The correction is known as the “residual matrix”, and this adaptation approach is referred to as “low-rank adaptation $\mathrm { ( L o R A ) } ^ { \prime }$ [23]. The rank of this residual matrix is kept low so that it can be specified with a small number of parameters, which is necessary since it will be estimated using a very small dataset at test time. The basis matrix $B \in \mathbb { R } ^ { b , T }$ contains the b bases, each of length T (our forecast horizon). The residual matrix R is the same shape, but estimated using a low rank factorization: $R = W O$ where $O \in \mathbb { R } ^ { r }$ is the correction of rank r and output from the MLP, and $W \in \mathbb { R } ^ { b * T , r }$ is a matrix learned during training and frozen during TTA. The resulting bases then are computed as $B + R _ { u n f l a t t e n e d }$ . Unflattening here refers to the reshaping of R to match the shape of B. We used a residual rank r of 16 for the results presented throughout this paper.

3) Training: The training algorithm, based on MAML, can be found in Algorithm 1. In brief, the training loop contains a type of simulation of the test-time adaptation procedure we will use later on unseen sessions. This causes the pretraining to push the model’s cross-session parameters into a regime where test-time adaptation will yield an accurate model with a small number of training steps and a small calibration dataset. More specifically, the training procedure involves:

• Random sample $n _ { s u p p o r t }$ trials (usually 300) from the training batch, without replacement. These form the simulated “support set” - i.e. calibration set; the remaining trials are the “query set”. This simulates the sampling process we will see when gathering an unseen session’s support set, on which we will perform TTA.

![](images/be928bb75747380834bdfa9c16c950e8ca4c79964d340ca900adb721f077d097.jpg)  
Fig. 6. Multi-session TBFM architecture. $X _ { b , t , d }$ Neural activity in a session (each block depicted is one of n sessions), time span t, channel count $d _ { * } ,$ multiple trials per session: b is the batch dimension. r is the runway length, T is the forecast horizon. $Z _ { t , l }$ Latent space representation of neural activity for a session, time span t, latent dimensionality l. s<sub>i</sub> per-trial embedding of stimulation parameters, known as the “stimulation descriptor”. $c _ { s } ^ { r e s t }$ per-session context from resting/non-stimulation activity, repeated for all trials of that session. $c _ { s } ^ { s t i m }$ learned per-session stimulation context, repeated for all trials of that session.

• Train a newly-initialized stimulation context $c _ { s } ^ { \mathrm { s t i m } }$ until convergence using a simulated test-time adaptation procedure (TTA) and the support set. Training uses backpropagation and gradient descent. This is known as the “inner loop”.

• Using the query set, calculate the loss (see below) and update the model parameters θ using backpropagation and a gradient descent based optimizer. This is known as the “outer loop”.

This is a first-order MAML meta-learning algorithm which pushes the model parameters θ into a regime where they can be easily adapted through TTA. The random sampling in the inner loop ensures that the meta-learning objective is optimized over a distribution of possible support sets, rather than a fixed partition. This makes the learned embedding and adaptation procedure robust to the variability due to the calibration set sampling process we will see when we encounter an unseen session. By having a pretrained θ prior to TTA, we introduce a strong bias towards the use of crosssession patterns in the stimulation responses, which is likely the source of our robustness gains.

The inner loop loss function is based on $L _ { 2 }$ forecast error $\lVert y - \hat { y } \rVert _ { 2 } ^ { 2 }$ . The outer loop loss function has the following terms:

$L _ { 2 }$ Forecast error $\lVert y - \hat { y } \rVert _ { 2 } ^ { 2 }$

• Basis weight regularization based on the Frobenius norm: $\lambda _ { \mathrm { w e i g h t } } \Vert W _ { t b f m } \Vert _ { F } ^ { 2 }$ , where $W _ { t b f m }$ is the TBFM’s matrix of basis weights. This was found in [18] to reduce overfit on single-session TBFMs, and is retained here.

• Reconstruction error $\lambda _ { \mathrm { { r e c o n } } } \vert \vert x - D e c ( E n c ( x ) ) \vert \vert _ { 2 } ^ { 2 }$ which encourages the affine encoder and decoder to be true pseudo-inverses of each other.

• A bases orthonormality penalty $\lambda _ { \mathrm { o r t h o } } O r t h o ( B )$ , which encourages the bases to be both orthogonal to each other and unit length. Its mathematical definition can be found on our Github pages.

• Stimulation context $L _ { 2 }$ regularization $\lambda _ { L 2 } \| c _ { s } ^ { \mathrm { s t i m } } \| ^ { 2 }$ which biases the stimulation context towards 0.

4) Test-time adaptation (TTA) procedure: Test-time adaptation (TTA) refers to the adaptation of a pretrained base model to an unseen task - in our case forecasting the effect of stimulation for an unseen session. It is based on a calibration data set, which is significantly smaller in size than what we would need for a single-session model.

The test-time adaptation (TTA) procedure reflects the same process as the training loop in Algorithm 1, but differs in some key ways:

• The TTA support set is a small calibration data set we collected for the unseen session, rather than one we random sampled from a larger training set.

• The inner and outer loop structure remains the same, but only the autoencoder is fully trained in the outer loop. It is still warm started with PCA on the calibration dataset. The inner loop still optimizes $c ^ { s t i m }$ . This is necessary since the autoencoder is specific to the session. The basis generator is fine tuned using a lower learning rate $( 1 0 ^ { - 6 } )$

5) Multi-session model compilation: Like single-session TBFMs (Section II-B.5), multi-session model compilation is possible. Compilation here refers to a simplification of the calculation which is possible after training and TTA have been performed. The simplification drastically reduces inference latency of the model (see Section III-E).

Compilation in the cross-session case is performed after

Algorithm 1 MAML-Based Multi-Session Training   
Require: Training data $\{ \mathcal { D } _ { s } \} _ { s = 1 } ^ { S }$ , model $f _ { \theta }$ (TBFM + AE),   
config $\mathcal { C }$   
Ensure: Trained model parameters $\theta ^ { * }$   
1: Initialize model parameters θ, load rest embeddings $\{ c _ { s } ^ { \mathrm { r e s t } } \}$   
2: for epoch = 1 to $N _ { \mathrm { e p o c h s } }$ do   
3: for each minibatch $\mathcal { B } \subset \{ \mathcal { D } _ { s } \}$ do   
4: $\mathcal { B } _ { \mathrm { s u p p o r t } } , \mathcal { B } _ { \mathrm { q u e r y } }  \mathrm { R A N D O M S P L I T } ( \mathcal { B } , n _ { \mathrm { s u p p o r t } } )$   
5: // Inner loop: adapt stimulus embeddings, simu  
lating test-time adaptation (TTA)   
6: Freeze $\theta ;$ initialize $\{ c _ { s } ^ { \mathrm { s t i m } } \} \sim \mathcal { N } ( 0 , 0 . 0 1 )$   
7: for $t = 1$ to $T _ { \mathrm { i n n e r } }$ do   
8: $\begin{array} { r l } { \mathcal { L } _ { \mathrm { i n n e r } }  } & { { } \frac { 1 } { \vert \mathcal { B } _ { \mathrm { s u p p o r t } } \vert } \sum _ { s } \vert \vert \hat { y } _ { s } - y _ { s } \vert \vert ^ { 2 } + \lambda _ { L 2 } \vert \vert c _ { s } ^ { \mathrm { s t i m } } \vert \vert ^ { 2 } } \end{array}$   
9: $\{ c _ { s } ^ { \mathrm { s t i m } } \} \gets \{ c _ { s } ^ { \mathrm { s t i m } } \} - \alpha _ { \mathrm { i n n e r } } \nabla _ { c ^ { \mathrm { s t i m } } } \mathcal { L } _ { \mathrm { i n } }$ nner   
10: end for   
11: $\{ c _ { s } ^ { \mathrm { s t i m } } \} \gets \mathrm { D E T A C H } ( \{ c _ { s } ^ { \mathrm { s t i m } } \} )$   
12: // Outer loop: update shared parameters on query   
set   
13: Compute $\mathcal { L } ( \mathcal { B } _ { \mathrm { q u e r y } } , \theta , \{ c _ { s } ^ { \mathrm { r e s t } } \} , \{ c _ { s } ^ { \mathrm { s t i m } } \} )$ // see Loss   
function, Section $I I { - } C . 3$   
14: $\theta  \theta - \alpha _ { \mathrm { o u t e r } } \nabla _ { \theta } \mathcal { L }$   
15: end for   
16: end for   
17: return $\theta ^ { * }$

TTA, reducing the model to a moderately simple two layer neural network with a skip connection:

$$
\hat { \mathbf { y } } = \left( \boldsymbol { B } ^ { \top } \phi \big ( A _ { \mathrm { p r e } } \mathrm { v e c } ( \mathbf { x } ) + \nu _ { \mathrm { p r e } } \big ) + \mathbf { 1 } _ { T } \mathbf { z } _ { 0 } ^ { \top } \right) W _ { e n c }\tag{1}
$$

where :

(2)

$$
{ \bf { z } } _ { 0 } = { \bf { x } } _ { r } \tilde { W } _ { e n c } ^ { \top } + \tilde { b } _ { e n c }\tag{3}
$$

$$
\scriptstyle \phi ( P ) = r o w N o r m ( t a n h ( P ) )\tag{4}
$$

$$
\tilde { W } _ { e n c } = W _ { e n c } \odot \alpha ,\tag{5}
$$

$$
\widetilde { b } _ { e n c } = \beta W _ { e n c } ^ { \top } + b _ { e n c }\tag{6}
$$

Here $\alpha , \beta$ are the normalization constants used for the session’s interquartile range (IQR) normalization, and vec() vectorizes (flattens) a matrix. $A _ { \mathrm { p r e } }$ and $\nu _ { \mathrm { p r e } }$ fuse the normaliser, AE encoder, and basis-weighting layer into a single affine map: $\mathrm { v e c } ( \mathbf { x } ) \mapsto A _ { \mathrm { p r e } } \mathrm { v e c } ( \mathbf { x } ) + \nu _ { \mathrm { p r e } } \in \mathbb { R } ^ { l \cdot b }$ , precomputed once post-TTA. The skip connection consists of the last runway value $\mathbf { X } _ { r } \in \mathbb { R } ^ { d }$ . Additional details on this derivation can be found on our Github pages.

Once we have adapted a model to a single session with TTA, we compile to reduce it to this simpler computation, resulting in a significant speed up in inference time (Section III-E), and increasing the feasibility of deployment on embedded systems.

6) Ablation: pretraining without MAML: For comparison, we present results below for one additional training and TTA technique we call “coadaptation.” In this simpler pretraining method, MAML is not used, and instead $c ^ { s t i m }$ is optimized along with all other parameters (i.e. co-adapted). At test time as well, $c ^ { s t i m }$ is optimized jointly with the autoencoder. This model uses the same architecture but is trained and adapted with this simpler approach in order to measure the effect of MAML in our setting, relative to this simpler pretraining technique.

![](images/c0bbb182e5346de47b701c50eb6ec1386479a556371e29ea3dd322019a42fa40.jpg)  
Fig. 7. Example deployment. In this simple example we collect larger calibration (randomly timed stimulation) data sets during the initial 20 sessions. During those sessions we create a single-session TBFM for closed-loop experimentation. After 20 sessions, we use a meta-learned shared-session TBFM and test-time adaptation, allowing us to collect far smaller calibration data sets (5k versus 500 calibration examples).

7) Deployment: Deployment refers to the process of building the base model over the course of a campaign of experiments and using it throughout the later part of the campaign to maximize robustness and minimize calibration data collection. There are many conceivable methods for deployment, and the optimal approach will likely depend on the details of the campaign. We present a simple deployment below to motivate future work.

Any deployment must address some key details. First: how much training data must we collect before we pretrain and use the base model? What decision rule do we use to go from single-session model use to the pretrained model and TTA use? For example: do we wait until the pretrained model has achieved a forecast error within a certain percentage of a single-session model on the same session? The answer to this question will depend on the accuracy/sample efficiency trade off we are willing to accept. A related question is, if enough compute is available, should we continue to train single session models on our tiny calibration set and compare their relative performance before deciding?

Second, will we continue to collect training data and refine the base model once we begin using it? How do we balance data collection for improving the base model and using it improve robustness and reduce data collection?

Below we present results for a simple deployment, and leave more sophisticated data collection scheduling to future innovation. Our simple approach consists of collecting larger calibration data sets from the first N sessions of the campaign, and using single session models on those sessions. Once N sessions have passed, we pretrain the shared session model and deploy it for all future sessions. In these later sessions, we can collect much smaller calibration sets. This is graphically depicted in Fig 7, with the results described in section III.

## III. RESULTS

For all results below, we evaluate our pretrained model types - co-adapted and MAML - on a difficult test time adaptation task. We hold out 20 sessions of data for test-time adaptation (TTA), and pretrain on the remaining 20 sessions. We compare the accuracy of single-session models on the 20 held out sessions, using the same sized calibration data sets, allowing us to assess sample efficiency of the respective approaches. In all cases, we assess accuracy using a test set of 2.5k examples drawn from the end of the held out session, simulating the trials we would see later (typically 1 hour) after data calibration. The TTA is performed on trials from the first part of the session for the given calibration set size, once again simulating the flow of an experimental session. Rather than N-fold cross validation, we perform Monte Carlo Cross-Validation (MCCV). We repeat our process 20 times, each time assigning sessions randomly to the held-in/out portions. We use this in lieu of N-fold cross-validation to maintain sufficiently-sized held-in/out set proportions.

## A. Pretraining and MAML provide more robustness across sessions

A desirable property for stimulation models is robustness, meaning that their accuracy should degrade gracefully rather than failing catastrophically in the presence of stronger nuisance variables such as noise. Overall, we found that pretrained TBFMs exhibit significantly improved robustness across all calibration set sizes, defined as the mass of the left tail of the accuracy distribution across held-out sessions and MCCV splits.

In Fig 8, we see the distribution of test set $R ^ { 2 }$ across sessions for each calibration set size. In all cases, the left tail of the distribution is significantly longer for vanilla (single-session) models compared to the coadaptive and MAML approaches. At a calibration set size of 1k for example, 16 of 40 sessions have vanilla models with test set $R ^ { 2 }$ below 0.05, compared with 1 of 40 for MAML-trained TBFMs. At a threshold of $R ^ { 2 } < 0 . 1 5$ these numbers increase to 18 and 7, respectively. Measured in terms of variance, the variances at calibration set sizes $< 5 k$ is less for MAML models to a statistically significant degree under the single-sided Brown-Forsythe test at $p < 0 . 0 5$ , and $p = 0 . 1 5$ at the 5k set size.

Fig 9 further illustrates that we see the highest performance gains on high noise sessions (indicated by a low A-ACF value). A-ACF calculated on resting (i.e., non-stimulation) data can be a proxy for the noise in a session’s data, and for the ability of a model to forecast that session’s neural activity [19]. We see in Fig 9 that this relationship holds for pretrained and testtime adapted models, and the largest performance gains are on the noisiest sessions.

We also see this robustness effect in Fig 10, where the prediction interval (PI) for an unseen session test $R ^ { 2 }$ is significantly wider across all calibration set sizes for vanilla models than for MAML-trained models. Notably, we sometimes see vanilla model performance marginally exceed MAML models. At a 1k set size for example, 8 of 40 sessions had higher vanilla model performance. Luckily, since training times are so short (see Section III-E), an experimenter is free to try both methods in a given session and pick the one which works best on some validation set.

![](images/38397af2ac2a0dd89caad8e84e0648f36f4923bea3471e2090d4080007f4e7b0.jpg)  
Fig. 8. Distribution of test set $R ^ { 2 }$ across folds and sessions, by calibration set size and training method. Dashed lines indicate median and IQR. Left tails of the distributions are significantly heavier for vanilla (i.e., single-session) models, indicating greater robustness of the coadaptive- and MAML-pretrained models. At the calibration set size of 1k for example, 7 of 40 sessions with coadapted model, and 7 of 40 with MAML models have test set $R ^ { 2 } < 0 . 1 5$ , compared with 18 of 40 for vanilla models. At 2.5k it is 8, 6 and 11, respectively.

## B. Pretraining and meta-learning significantly increase sample efficiency on unseen sessions

Overall, our pretrained models significantly exceed in sample efficiency relative to our single session baseline. In all cases, we see diminishing returns on increasing calibration set size, but at drastically different rates between single session and pretrained models.

At moderate calibration set sizes (1k-2.5k), the accuracy of the pretrained models exceed single-session models by a significant margin $( p < 0 . 0 5 )$ on previously unseen (i.e., “held-out”) sessions (Fig 10). At a 1k calibration set size for example, single session models exhibit an average $R ^ { 2 }$ of 0.167 on held out session test sets, while our MAML pretrained and adapted models exhibit an average $R ^ { 2 }$ of $0 . 3 9 7 \mathrm { ~ - ~ } \mathrm { a }$ difference significant at $p = 9 . 1 ^ { - 5 }$ under the 1-sided Wilcoxon signedrank test.

By a 5k set size, we see the performances of single session and pretrain models nearly converge, with pretrained models still showing a slight but statistically insignificant advantage: 0.424 versus 0.398 for single-session models - a difference $p = 0 . 1 5$ under the 1-sided Wilcoxon signed-rank test. The convergence point represents the scale at which we have collected enough data that mean session performance begins to no longer benefit from pretraining. The experimenter saves on data collection by picking a calibration set size smaller than this. As we show in Section III-A, the pretraining approach additionally provides greater robustness, even at the point that the mean performances of the two approaches converge.

![](images/f4687f388125de3a8645c7bc5a93b04f61d306de9549df71bd8ca6cffaa3de65.jpg)  
Fig. 9. A-ACF predicts performance of pretrained models. Robustness is indicated by high gains on noisy sessions. Lower A-ACF values on resting data suggest a session’s data are noisier. The median A-ACF value across a session’s channels is predictive of both vanilla (single-session) model and MAML TTA model performance: the trend slope is positive at $p < 5 ^ { - 5 }$ in all cases. Pretraining flattens the susceptibility of the model to failing catastrophically on high noise (low A-ACF) sessions, and hence we see large improvements in the high noise regime. Vanilla models more frequently fail catastrophically $( R ^ { 2 } < 0 . 0 5 )$ on higher noise sessions. At 5k calibration trials (right), the two methods converge, indicating that pretraining’s advantage is specific to low-data regimes where anchoring to cross-session patterns compensates for limited per-session evidence. The slope of the difference in trends is negative at $1 ^ { - 3 }$ for 1k, insignificant in 5k. The 5k result is expected since prior work found this was the optimal training set size for single-session models [7].

![](images/49f372e210cf97bbbf2d371c198a0d81ba25f6434aab265585f7d87d6cb6bcaf.jpg)  
Fig. 10. Sample efficiency of single session versus MAML pretrained models. The plot shows mean and distribution of test set R<sup>2</sup> across held out sessions and Monte Carlo Cross-Validation (MCCV) splits. Confidence intervals (CIs) are for the mean of held-out sessions. The prediction intervals (PIs) capture the variability in performance across sessions and MCCV splits, reflecting both genuine cross-session differences in signal quality and the uncertainty introduced by the particular held-in/held-out assignment. Mean performances of MAMLpretrained models exceed vanilla (single session) models at $p < 0 . 0 5$ under the 1-sided Wilcoxon signed-rank test for calibration set sizes $\leq 2 . 5 k$ and $p = 0 . 1 5$ for 5k. Prediction intervals are significantly more narrow for MAML-pretrained models under the single-sided Brown-Forsythe test $( p < 0 . 0 5 )$ for calibration set sizes $\leq 2 . 5 \bar { k } .$ , indicating more consistent performance and higher robustness. At 5k, it drops to $p =$ 0.19.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Held out subject</td></tr><tr><td rowspan=1 colspan=1>Held in subject</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1> $\overline { { 0 . 4 0 5 \pm 0 . 0 5 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 3 5 4 \pm 0 . 0 9 4 } }$ </td></tr><tr><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1> $\overline { { 0 . 4 0 7 \pm 0 . 0 4 7 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 3 7 3 \pm 0 . 1 0 2 } }$ </td></tr></table>

TABLE I  
SUMMARY OF WITHIN- AND CROSS-SUBJECT GENERALIZATION. VALUES ARE HELD-OUT SESSION TEST SET PERFORMANCE, MEAN ACROSS FOLDS. INDICATED INTERVAL IS THE 95% CONFIDENCE INTERVAL OF THE MEAN.

## C. MAML improves accuracy the over simpler pretraining approach

MAML pretrained models exceed the co-adapted models in accuracy across all calibration set sizes (Fig 8). At a 1k calibration set size, for example, we see 0.345 $R ^ { 2 }$ for coadapted models versus 0.397 for MAML. At all set sizes MAML has significantly higher mean performance under the two-sided Wilcoxon signed-rank test $( p < 1 ^ { - 5 } )$ . These results indicate that meta-learning can help improve sample efficiency beyond mere pretraining.

The variance of results is also reduced by MAML across all calibration set sizes, though not to a statistically significant degree under the two-sided Brown-Forsythe test. We depict this visually in Fig 8. In brief: lower variance indicates more consistent results, and therefore may indicate a higher degree of robustness.

## D. Cross-animal generalization

One desirable property of a cross-session stimulation model is the ability to generalize across subjects. Future foundation modeling efforts will rely on cross-subject similarity in the data. Our dataset contains sessions from two subjects: Subject G (22 sessions) and Subject J (18 sessions).

To examine cross-subject generalization, we performed an MCCV procedure similar to the results above. Here, for a given fold, we randomly split each subject’s sessions into 10 held-in sessions, and the remainder are held-out. We pretrain each subject’s model using MAML and its held in sessions, and then perform test-time adaptation of the model to the other animal’s held out sessions. We repeat the TTA on the samesubject’s held out sessions as well, to measure the difference between within-animal and cross-animal generalization. Due to computation limits, we perform only four folds, but note that the mean results varied little between folds.

Overall, we found that pretrained TBFMs successfully generalize between our two subjects. Results are summarized in Table I. Held out sessions from Subject G exhibited test set accuracy effectively identical across models. Subject J’s held out sessions yielded somewhat higher accuracy on the Subject J pretrained model, but the difference is not statistically significant under the two-sample t-test. These results indicate that 1) there exists enough similarity between subjects that a model pretrained on one subject can usefully adapt to an unseen subject; and 2) multi-session TBFMs are capable of performing this adaptation.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CPU</td><td rowspan=1 colspan=1>GPU</td></tr><tr><td rowspan=1 colspan=1>Pretrained TBFM</td><td rowspan=1 colspan=1>2.798</td><td rowspan=1 colspan=1>0.790</td></tr><tr><td rowspan=1 colspan=1>Pretrained TBFM compiled</td><td rowspan=1 colspan=1>2.095</td><td rowspan=1 colspan=1>0.223</td></tr><tr><td rowspan=1 colspan=1>Vanilla TBFM</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.226</td></tr><tr><td rowspan=1 colspan=1>Vanilla TBFM compiled</td><td rowspan=1 colspan=1>0.115</td><td rowspan=1 colspan=1>0.121</td></tr><tr><td rowspan=1 colspan=1>AE-LSTM</td><td rowspan=1 colspan=1>41.628</td><td rowspan=1 colspan=1>34.156</td></tr><tr><td rowspan=1 colspan=1>LSSM</td><td rowspan=1 colspan=1>10.111</td><td rowspan=1 colspan=1>16.939</td></tr></table>

TABLE II  
AVERAGE SINGLE-TRIAL INFERENCE TIMES (MILLISECONDS)

E. Training time and inference latency slower, but still faster than comparison models

We benchmark training time and inference latency on a desktop computer built from common components: an AMD Ryzen Threadripper PRO 5955WX CPU, and an NVIDIA GeForce RTX 4090 GPU. In all cases training is performed on the GPU. We compare CPU versus GPU single-trial inference latencies in Table II.

Inference time for pretrained TBFMs is higher due to the higher model complexity, in particular the addition of an encoder/decoder. Typical GPU-based inference time for a single trial is 0.223ms for a pretrained and compiled TBFM, compared with 0.121ms for the compiled vanilla TBFM (see Table II). While slower, the pretrained TBFM remains well below our target 20ms real-time requirement, and faster than the latency of some existing clinical closed-loop systems at 7.5ms [20]. For comparison, the inference times were 10.111ms and 34.156ms for our baseline linear state space model (LSSM) and LSTM models, respectively.

Pretraining a multi-session TBFM on 20 held in sessions using MAML requires 3.6hr on average, which reflects the complexity of the training loop and scale of the training data set. We envision experimenters running this training between sessions rather than in the middle of a session, as needed.

During a session, the calibration data set can be collected, followed by TTA of a pretrained model or training of a vanilla model, followed by closed-loop stimulation. As a result, TTA and vanilla model training need to be fast. On our hardware, typical vanilla model training requires ≈ 3min for a calibration set size of 5k. TTA requires 13min for an average session with a support size of 1k, which is the support size with expected test set accuracy roughly equal to the vanilla model’s dataset size of 5k. While longer, this TTA training time is still well within typical experimental constraints. Compare to 5k training set times of 11.53hr and 90.6min for our complex LSTM and LSSM comparison models, respectively [7].

Note that MAML with TTA in the typical sense is usually faster to train than a single session model: the intent is to initialize a network such that it can be trained with few gradient descent steps. Here MAML is playing a somewhat different role: it is used to initialize only the session stimulation embeddings $( c _ { s } ^ { s t i m } )$ and therefore temporal bases, not the entire network. This makes the more complex training loop (Algorithm 1) slower than the simple gradient descent used for single-session models. We believe this trade-off is justified: with higher accuracy, greater robustness, and reduced subject fatigue from calibration stimulation, we argue the increased training time is worthwhile, particularly given the promise of rapidly evolving hardware which can further accelerate the learning.

## F. Ablations

Our model’s design was based on the result of many ablations. These ablations provided insight into the design decisions needed to make the model work well for multisession pretraining and adaptation. We believe these insights may generalize beyond our particular model and setting, and therefore outline the insights below.

• Tanh nonlinearity and weight norm were necessary for stabilizing training. Without these choices, we found that the basis weight estimator and basis generator parts of the TBFM network (Fig 6) have conflicting gradients, causing the learned parameters to oscillate in scale and never stabilize.

• Basis orthonormality regularization decreases held-in session accuracy but improves accuracy on held-out sessions, suggesting this regularization is a useful constraint which aids generalization.

• IQR normalization rather than z-score exhibits a lower median accuracy on held-out session test sets, across all support sizes. Its mean performance is higher at lower support sizes but to a statistically insignificant degree.

• L2 regularization of c<sup>stim</sup> improves test set performance, especially on held-out sessions and lower support sizes. This penalty pushes basis adaptation to be moderate, and likely is controlling basis overfit.

• Inclusion of $c ^ { r e s t }$ decreases held-in session performance, slightly decreases held-out session test set performance. We nevertheless include it in our results to illustrate the concept of leveraging exogenous correlates, as explained in Section II.

• MAML rather than co-adaptive pretraining improves mean and median accuracy as well as robustness.

• Autoencoder reconstruction loss improved mean and median held-out session accuracies, but to a statistically insignificant degree.

• Partial fine-tuning of latent TBFM during TTA, allowing the basis generator to adapt at a low learning rate $( 1 0 ^ { - 6 } )$ during TTA, helps accuracy on held-out session test sets, particularly at higher support sizes.

## IV. DISCUSSION

Our results show that pretrained temporal basis function models significantly exceed single-session models in sample efficiency and robustness, with MAML providing an additional and measurable improvement over simpler co-adaptive pretraining. Together our results provide, to our knowledge, the first demonstration that meta-learning can be applied to neural stimulation modeling to produce practical gains in efficiency and reliability. We discuss below the interpretation of these findings, their architectural underpinnings, key limitations, and the longer-term trajectory toward foundation models for neural stimulation.

## A. Interpretation of Robustness Gains

A finding at least as significant as the mean accuracy improvement is the compression of the left tail of the performance distribution (Fig 8). At a calibration set size of 1k, vanilla (single-session) models fall below $R ^ { 2 } = 0 . 0 5 ~ .$ effectively no better than a forecast consisting only of the channel mean - on 16 of 40 sessions, compared with just 1 of 40 for MAML-pretrained models. In a clinical deployment, a system that achieves strong mean performance but fails entirely in a material fraction of sessions cannot be considered reliable. The prediction intervals (PIs) in Fig 10 quantify this: PIs reflect the variability in performance across sessions and MCCV splits, capturing both genuine cross-session differences in signal quality and uncertainty due to the particular heldin/held-out assignment. The significantly narrower PIs for MAML-pretrained models (Brown-Forsythe test, $p < 0 . 0 5$ for calibration set sizes ≤ 2.5k) confirm that this is not merely a mean-shift effect.

We attribute the robustness improvement to two complementary mechanisms. First, the MAML objective explicitly optimizes for adaptation across a distribution of support sets, sampled randomly during the inner loop. This discourages the model from finding parameter configurations that are sensitive to the particular trials included in the calibration set - precisely the configurations most likely to fail under noisy or atypical sessions. Second, the resting-state context $c _ { s } ^ { \mathrm { r e s t } }$ , computed from autocorrelation statistics of baseline data prior to any stimulation, provides the model with an advance signal about session noise level. Our prior work showed that the average absolute value of the autocorrelation function (A-ACF) of baseline LFPs correlates strongly with the predictability of stimulation responses across model types [19]. By conditioning the basis generator on this context, the model can modulate the conservatism of its forecasts before seeing a single stimulation trial, attenuating the catastrophic failures that arise when an overconfident model is deployed in a highnoise session.

## B. Interpretation ofSample Efficiency Gains

Our core sample efficiency result, namely, that MAMLpretrained models achieve comparable or superior accuracy to single-session models using 50−90% less calibration data, has direct practical consequences for closed-loop neurostimulation experiments. Calibration data collection is costly in at least three respects: it requires session time that could otherwise be devoted to closed-loop exploration, it imposes repeated nontherapeutic stimulation on the subject, and in clinical contexts, it may not be feasible at all within a single session. The convergence of pretrained and vanilla model performance at a 5k calibration set size $( R ^ { 2 } = 0 . 4 2 4$ vs. 0.398, p = 0.19) identifies the regime in which cross-session structure is no longer required for improving model accuracy, and singlesession data alone is sufficient. Below that threshold, the pretrained model exploits structure that the vanilla model must rediscover from scratch each session.

The diminishing returns observed for both model types as calibration set size increases are consistent with a general property of supervised learning: the most informative examples are encountered early, and additional data yields progressively smaller improvements in generalization. The pretrained model begins the calibration process in a θ neighborhood suitable for efficient adaptation. That initialization provides not only temporal bases capturing stereotyped stimulation-evoked response shapes, but also a weight estimator design matrix $( \theta _ { A } )$ encoding shared spatiotemporal predictive structure - specifically, the cross-session consistency in how pre-stimulus neural state in the runway predicts the evoked response. The stimulation context vector $( c _ { s } ^ { s t i m } )$ then captures session-specific deviations from these shared structures, requiring only a small support set to converge.

## C. The Role ofCross-Session Structure

Meta-learning can succeed because the sessions share exploitable patterns. Local stimulation responses in primary somatosensory and motor cortex tend to be stereotyped at the level of individual channels: the canonical shape of a stimulation-evoked LFP response - an initial deflection followed by a rebound and recovery - is broadly conserved across animals and sessions [7]. This indirectly resembles the hemodynamic response function (HRF) modeled in functional magnetic resonance imaging (fMRI) studies, which is a temporal structure (i.e. temporal basis function) that can be leveraged across much of the brain in the context of a generalized linear model (GLM) [25]. Pretrained TBFMs likewise share temporal structure across both space and individual sessions, but learn the basis functions from data rather than using a canonical basis set. Inter-session variability reshapes the temporal bases, and the stimulation context $c _ { s } ^ { \mathrm { s t i m } }$ estimates the necessary reshaping in a low-rank manner.

The per-session linear autoencoders further absorb differences in spatial structure. They serve two functions simultaneously: they handle the variable electrode count across sessions (42 − 94 usable channels in the present dataset), and they project each session’s activity into a common latent space where the shared TBFM can operate. Warm-starting the autoencoders with PCA biases the encoders toward solutions that are compact. Together, these design choices explicitly factor out session-specific nuisance variation and expose the shared temporal structure that pretraining can leverage. This factored architecture may generalize naturally to other stimula tion modalities and brain regions where comparable stereotypy exists.

## D. MAML vs. Co-Adaptive Pretraining

The consistent accuracy advantage of MAML-pretrained models over co-adapted models $( p < 1 0 ^ { - 5 }$ across all calibration set sizes) isolates the contribution of the meta-learning objective itself. The co-adapted model uses the identical architecture and training corpus, but optimizes $c _ { s } ^ { \mathrm { s t i m } }$ jointly with all other parameters rather than through a simulated inner-loop adaptation. The fact that this simpler approach is already substantially better than a vanilla single-session model confirms that cross-session pretraining is broadly beneficial. The further improvement from MAML suggests that the structured inner/outer loop training genuinely shapes the loss landscape in a way that facilitates adaptation, rather than merely benefiting from the larger effective training set.

Intuitively, MAML ensures that the stimulation embedding $c _ { s } ^ { \mathrm { s t i m } }$ is initialized in a region of parameter space where a small calibration set suffices to capture session-specific response properties. Without this, the co-adapted model may converge to representations that fit the training sessions well but do not support fast adaptation at test time - a form of overfitting to the meta-training distribution rather than to any individual session. The residual-rank architecture (rank $r = 1 6$ for the basis residual matrix $R )$ further supports this by keeping the degrees of freedom available to $c _ { s } ^ { \mathrm { s t i m } }$ small, ensuring that TTA remains a well-conditioned estimation problem even on tiny calibration sets.

## E. Limitations

Several limitations of the present work warrant explicit discussion.

Dataset scope: The dataset used in the present study derives from two rhesus macaques in a pair of cortical regions (S1/M1) using a single stimulation modality (optogenetic, redshifted channelrhodopsin C1V1). While this allows careful characterization under controlled conditions, it limits the generalizability of our findings to other brain regions, species, and stimulation modalities. Electrical stimulation, for example, introduces stereotyped artifacts whose temporal shape may itself be well-captured by learned basis functions, but it remains unclear how to train a shared model on datasets that mix modalities with qualitatively different artifact time courses. Addressing this challenge - perhaps through modalityspecific basis components or artifact-aware normalization - is an important direction for future work.

Linear autoencoders: The choice of a linear autoencoder for cross-session alignment is deliberate, prioritizing computational simplicity and compatibility with small calibration sets. Nonlinear alignment methods might achieve better generalization when sessions are more heterogeneous, but introduce additional parameters that are harder to warm-start and regularize with limited data. The tradeoff between alignment expressivity and calibration efficiency deserves systematic study as larger, more heterogeneous stimulation datasets become available.

Test-time adaptation duration: TTA times of approximately 13 minutes, while acceptable within a standard experimental session, represent a practical constraint in rapid-iteration paradigms and would need to be reduced for some clinical applications. The TTA runtime is dominated by autoencoder adaptation in the outer loop, not by the inner-loop stimulation context optimization. Architectural modifications - such as a lower-dimensional autoencoder bottleneck, a stronger PCAbased initialization, or frozen encoder weights with only the decoder adapted - may reduce this substantially. Continued improvements in GPU hardware will also provide gains independently of architectural changes.

Evaluation under distribution shift due to behavioral states: Our evaluation protocol trains on the early portion of each session and evaluates on the late portion (final 2.5k trials), providing some protection against within-session drift. However, we have not evaluated performance under intentional distribution shift - for example, calibrating in one behavioral state and deploying in another. While known behavioral states may be captured as additional covariates that can be input to our basis weight generator, it remains unclear how well our model will generalize to unseen behavioral states. In clinical applications such as adaptive DBS, stimulation may be required across a wide range of patient states (rest, movement, sleep, emotional arousal, etc.), each potentially producing different response dynamics. Building explicit robustness to state-level distribution shifts into the meta-learning framework, perhaps through multi-state calibration protocols or explicit behavioral state conditioning, remains important future work for our approach and other stimulation modeling approaches.

## F. Active learning and adaptive deployment

The deployment scenario described in Section II-C.7 is deliberately simple: collect full calibration sets for an initial batch of sessions, then switch to TTA for all subsequent sessions. Myriad more sophisticated deployment strategies are conceivable. Active learning [22] is a particularly promising direction: rather than collecting calibration trials at random stimulation timings, the model could select the stimulation parameters or timing expected to maximally reduce its uncertainty about the session-specific response. In the closed-loop stimulation setting, this could take the form of an exploration phase - brief, model-guided stimulation trials chosen to be maximally informative - followed by a shorter exploitation phase using closed-loop control. Active learning has been shown to improve sample efficiency in related neural decoding settings, and its application to stimulation modeling is a natural next step.

Continuous online adaptation is another avenue: rather than treating TTA as a one-time calibration step, the model could be updated incrementally as new trials accumulate within a session, potentially tracking within-session non-stationarities such as habituation, electrode drift, or changes in arousal. A principled combination of online adaptation and active trial selection could substantially reduce the total number of stimulation trials required for reliable closed-loop control.

## G. Towards a foundation model for neural stimulation

The results presented here provide, to our knowledge, the first empirical evidence that a pretrained model based on meta-learning can exploit cross-session structure in neural stimulation data to improve generalization in stimulation response forecasting. This is a necessary condition - though not sufficient on its own - for any credible effort to build foundation models for neural stimulation.

Large pretrained models have recently begun to emerge in neural decoding, driven in part by efforts like the FAL-CON benchmark [24], which has standardized evaluation of few-shot decoding across multiple recording modalities and participants. Stimulation modeling has lagged behind decoding in this and many regards, partly because ground-truth stimulation-response data is harder to collect at scale. The present approach begins to address this gap by demonstrating that pretraining is practical with modest multi-session datasets (20 sessions here), and that even limited cross-session structure is sufficient for meaningful improvement.

A practical path toward a stimulation foundation model based on TBFMs might proceed as follows. First, assemble a heterogeneous corpus of stimulation experiments spanning modalities, brain regions, and species. Second, use the TBFM architecture proposed here - with flexible per-session autoencoders, stimulation context vectors, and resting-state context - to learn a shared prior over stimulation responses. Third, evaluate generalization to held-out sessions under the full range of cross-session variability present in the corpus, using a benchmark structured analogously to FALCON. The restingstate context already illustrates one avenue for including external covariates that do not require stimulation trials; further biological covariates (electrode impedance profiles, restingstate power spectra, subject demographics) could similarly inform adaptation without adding to calibration data requirements.

The key open question is whether the shared latent structure of stimulation responses is sufficiently consistent across the substantial biological heterogeneity of such a corpus to support a common prior. We consider this an important open question for the field, and call on experimenters to consider how standardized multi-site stimulation datasets - analogous to FALCON but for stimulation rather than decoding - could be assembled and shared to enable rigorous benchmarking. The present results provide early grounds for optimism that such an effort would be scientifically productive.

## H. Clinical translation considerations

Beyond sample efficiency and accuracy, a credible path toward clinical translation requires predictable behavior under the full range of conditions likely to be encountered in use. Our robustness results are encouraging in this respect: the reduction in catastrophic failure modes suggests that MAML-pretrained models are less sensitive to variability in the calibration data, which is one proxy for the kind of distribution shift that clinical systems routinely encounter.

Several additional considerations will need to be addressed as this work moves toward clinical evaluation. Regulatory requirements for closed-loop neural devices generally demand that stimulation controllers behave safely and predictably under both normal and degraded operating conditions: a model that performs well on average but fails unpredictably in a minority of sessions is unlikely to meet this bar. The improved tail behavior we observe is a step in the right direction, but formal uncertainty quantification - for example, conformal prediction bounds on the forecast error - would provide stronger guarantees suitable for safety-critical deployment.

Real-world closed-loop systems also operate under hardware and energy constraints that laboratory systems do not face. The inference latency of 0.223ms for a pretrained TBFM on GPU is well within our 20ms target and faster than some existing clinical systems [20]. The architectural simplicity of TBFMs makes them amenable to implementation on low-power embedded processors, a practical advantage over recurrent neural networks or transformer-based alternatives. Despite the training-time complexity, the post-TTA compiled model reduces to a shallow computation: two affine transformations separated by a normalized tanh nonlinearity, plus a skip connection carrying the last runway value forward as a baseline forecast. The training-time apparatus - persession autoencoders, MAML inner loops, MLP-generated LoRA-based basis adaptation - collapses at inference into a structure resembling a two layer perceptron with a residual connection. This separation between training-time complexity and inference-time simplicity is architecturally favorable for clinical deployment: the elaborate machinery needed to learn cross-session structure does not impose corresponding complexity on the deployed device.

Finally, the inter-session variability we observe across just two animals underscores the challenge of building systems that generalize across patients with different anatomies, disease states, and stimulation histories. Transfer learning approaches like the one demonstrated here will likely be a prerequisite for practical deployment, given that collecting large per-patient calibration datasets is often infeasible. We view the results reported here as an early proof of concept for this broader program.

## V. CONCLUSION

Our results provide, to our knowledge, the first demonstration that meta-learning can be applied to neural stimulation modeling to achieve meaningful improvements in both sample efficiency and robustness. Building on temporal basis function models, we show that model-agnostic meta-learning (MAML) enables a pretrained model to adapt to unseen sessions using calibration datasets 50 − 90% smaller than those required by single-session models, without a significant accuracy trade-off at moderate calibration set sizes. Critically, pretraining does not merely shift mean performance upward - it substantially compresses the left tail of the performance distribution, reducing the frequency of catastrophic failures that would otherwise make deployment unreliable.

The architectural foundation for these gains is a factored design in which shared temporal basis functions and a shared weight estimator design matrix capture cross-session structure, while per-session autoencoders and stimulation context vectors absorb session-specific variation. This division of labor reflects a broader principle with precedent in functional neuroimaging, where stereotyped stimulation response timecourses - analogous to the hemodynamic response function - have long been leveraged to improve signal estimation across brain regions and subjects [25]. Our results suggest that a similar principle holds for LFP responses evoked by neural stimulation applied directly to neural tissue (as opposed to, e.g., visual stimulation applied through the eyes). Our approach additionally has the advantage that the basis functions are learned from data rather than assumed canonical.

Our approach simultaneously yields inference latency well within real-time control constraints, and the MAML training objective provides robustness properties that begin to address the reliability requirements for clinical deployment. These results provide early empirical evidence that cross-session structure in neural stimulation data is consistent enough to support pretraining - a necessary condition for any future effort to build foundation models for neural stimulation.

Taken together, we believe this work meaningfully narrows the gap between complex data-driven approaches to closedloop neural stimulation and the practical constraints - limited calibration time, session-to-session variability, and noise - that have historically made such approaches difficult to deploy. The path forward involves extending this framework to more heterogeneous datasets spanning modalities, brain regions, and species, and developing standardized benchmarks that can drive progress toward a general-purpose neural stimulation foundation model.

## REFERENCES

[1] Castano-Candamil S, Ferleger B I, Haddock A, Cooper S S, Her-˜ ron J, Ko A, Chizeck H J and Tangermann M 2020 Frontiers in Human Neuroscience 14 421 A pilot study on data-driven adaptive deep brain stimulation in chronically implanted essential tremor patients https://www.frontiersin.org/article/10.3389/ fnhum.2020.541625

[2] Little S e a 2016 Journal of neurology, neurosurgery, and psychiatry 87(7) 717–21 Bilateral adaptive deep brain stimulation is effective in parkinson’s disease.

[3] Bryan M J, Jiang L P and Rao R P N 2023 Journal of Neural Engineering 20 036004 Neural co-processors for restoring brain function: results from a cortical model of grasping https://dx.doi.org/ 10.1088/1741-2552/accaa9

[4] Bolus M, Willats A, Rozell C and Stanley G 2021 Journal of neural engineering 18(3) State-space optimal feedback control of optogenetically driven neural activity

[5] Bradley C, Nydam A S, Dux P E and Mattingley J B 2022 Nature Reviews Neuroscience 23(8) 459–475 State-dependent effects of neural stimulation on brain function and cognition

[6] Niparko J K 2009 Lippincott Williams and Wilkins (Oxford University Press)

[7] Bryan M J, Schwock F, Yazdan-Shahmorad A and Rao R 2025 Journal of Neural Engineering 22 Temporal basis function models for closedloop neural stimulation 10.1088/1741-2552/ae036a

[8] Rao R P N 2019 Current Opinion in Neurobiology 55 142–151 Towards neural co-processors for the brain: Combining decoding and encoding in brain-computer interfaces

[9] Yang Y, Qiao S, Sani O, Sedillo J, Ferrentino B, Pesaran B and Shanechi M 2021 Nature Biomedical Engineering 5(4) 324–345 Modelling and prediction of the dynamic responses of large-scale brain networks during direct electrical stimulation https://doi.org/10.1038/ s41551-020-00666-w

[10] Finn C, Abbeel P and Levine S 2017 CoRR Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks http://arxiv. org/abs/1703.03400

[11] Li D, Ortega P, Wei X, and Faisal A 2021 Model-Agnostic Meta-Learning for EEG Motor Imagery Decoding in Brain-Computer-Interfacing https://arxiv.org/abs/2103.08664

[12] Bloch J, Greaves-Tunnell A, Shea-Brown E, Harchaoui Z, Shojaie A and Yazdan-Shahmorad A 2022 iScience 25 104285 Network structure mediates functional reorganization induced by optogenetic stimulation of non-human primate sensorimotor cortex ISSN 2589-0042 https://www.sciencedirect.com/science/article/ pii/S2589004222005557

[13] Yazdan-Shahmorad A, Silversmith D B, Kharazia V and Sabes P N 2018 eLife 7 e31034 Targeted cortical reorganization using optogenetics in non-human primates ISSN 2050-084X https://doi.org/10. 7554/eLife.31034

[14] Yazdan-Shahmorad A, Diaz-Botia C, Hanson T L, Kharazia V, Ledochowitsch P, Maharbiz M M and Sabes P N 2016 Neuron 89(5) 927– 39 A large-scale interface for optogenetic stimulation and recording in nonhuman primates

[15] Ledochowitsch P and Yazdan-Shahmorad e a 2015 Journal of neuroscience methods 256 220–31 Strategies for optical control and simultaneous electrical readout of extended cortical circuits

[16] Yazdan-Shahmorad A, Silversmith D B and Sabes P N 2018 Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference 2018 5479–5482 Novel techniques for large-scale manipulations of cortical networks in non-human primates

[17] Yazdan-Shahmorad A, Diaz-Botia C, Hanson T, Ledochowitsch P, Maharabiz M and Sabes P 2015 Progress in Biomedical Optics and Imaging - Proceedings of SPIE 9305 Demonstration of a setup for chronic optogenetic stimulation and recording across cortical areas in non-human primates

[18] Bryan M J, Schwock F, Yazdan-Shahmorad A and Rao R P N Proceedings of the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Copenhagen, Denmark 2025 (to appear) Learning Temporal Basis Vectors for Closed-Loop Neural Stimulation

[19] Bryan M J, Schwock F, Yazdan-Shahmorad A and Rao R P N Proceedings of the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Copenhagen, Denmark 2025 (to appear) Average of Baseline Autocorrelation Function is a Leading Indicator of Neural Stimulation Data Quality

[20] Guggenmos D J, Azin M, Barbay S, Mahnken J D, Dunham C, Mohseni P and Nudo R J 2013 Proceedings of the National Academy of Sciences 110 21177–21182 Restoration of function after brain damage using a neural prosthesis (Preprint https://www.pnas.org/doi/abs/ 10.1073/pnas.1316885110

[21] Zanos S, Rembado I, Chen D and Fetz E E 2018 Current Biology Phase-locked stimulation during cortical beta oscillations produces bidirectional synaptic plasticity in awake monkeys

[22] Murphy K P 2022 Probabilistic Machine Learning: An introduction (MIT Press) probml.ai

[23] Hu EJ et al. 2021 ArXiv LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685

[24] Karpowicz B et al. 2024 bioArxiv preprint Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark https://www.biorxiv.org/content/early/2024/10/ 31/2024.09.15.613126

[25] K. J. Friston, A. P. Holmes, K. J. Worsley, J.-P. Poline, C. D. Frith, and R. S. J. Frackowiak Human Brain Mapping 2(4) 189- 210 Statistical parametric maps in functional imaging: A general linear approach https://onlinelibrary.wiley.com/doi/10. 1002/hbm.460020402

## VI. APPENDIX

## A. Code availability

All code for this paper is available through an open-source license (BSD 3-Clause) in the ‘multisession’ branch of our Github repository.

## B. Hyperparameters

The reader can find most of these hyperparameters defined in our configuration files, allowing them to trace the values easily through the code.

## 1) Model architecture:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>TBFM</td><td></td></tr><tr><td>Ambient latent dimension (l)</td><td>96</td></tr><tr><td>Covariate (stim) dimension</td><td>3</td></tr><tr><td>Number of bases (b)</td><td>100</td></tr><tr><td>Basis-generator latent dim</td><td>16</td></tr><tr><td>Basis-generator depth</td><td>3</td></tr><tr><td>Basis-generator dropout</td><td>0.3</td></tr><tr><td>Meta projection dim</td><td>32</td></tr><tr><td>Rest-embedding dim  $( c _ { s } ^ { \mathrm { r e s t } } )$ </td><td>3</td></tr><tr><td>Stim-embedding dim  $( c _ { s } ^ { \mathrm { s t i m } } )$ </td><td>15</td></tr><tr><td>tanh basis weights</td><td>true</td></tr><tr><td>Basis-weight row norm</td><td>true</td></tr><tr><td>Shared bases</td><td>true</td></tr><tr><td>Meta-learning residual module</td><td></td></tr><tr><td>Basis residual rank</td><td>16</td></tr><tr><td>Residual MLP hidden dim</td><td>16</td></tr><tr><td>Autoencoder</td><td></td></tr><tr><td>Autoencoder (AE) type</td><td>LinearChannelAE</td></tr><tr><td>AE input dim</td><td>96</td></tr><tr><td>AE bias</td><td>true</td></tr></table>

## 2) Training:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Data</td><td></td></tr><tr><td>Trial length</td><td>184 bins</td></tr><tr><td>Runway (pre-stimulus conditioning window)</td><td>20 bins</td></tr><tr><td>Prediction length / forecast horizon (T)</td><td>164 bins</td></tr><tr><td>Train set size</td><td>5000</td></tr><tr><td>Test set size</td><td>2500</td></tr><tr><td>Held-out sessions</td><td>15</td></tr><tr><td>Normalizer</td><td>quantile</td></tr><tr><td>Outlier IQR multiplier</td><td>10.0</td></tr><tr><td>Max outliers per trialª</td><td>5</td></tr><tr><td>Outer loop</td><td></td></tr><tr><td>Training steps  $( N _ { \mathrm { e p o c h s } } )$ </td><td>12001</td></tr><tr><td>Batch size per session</td><td>500</td></tr><tr><td>Gradient clip (l2)</td><td>2.0</td></tr><tr><td>Basis weight estimator updates per basis generator updateb</td><td>3</td></tr><tr><td>Basis weight estimator LR</td><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Basis generator (outer) LR  $( \alpha _ { \mathrm { o u t e r } } )$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Basis weight estimator weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Basis generator weight decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Basis weight reg. weight  $( \lambda _ { \mathrm { w e i g h t } } )$ </td><td>75.0</td></tr><tr><td>Orthonormality penalty  $( \lambda _ { \mathrm { o r t h o } } )$ </td><td>0.05</td></tr><tr><td>AE reconstruction weight  $( \lambda _ { \mathrm { r e c o n } } )$ </td><td>0.03</td></tr><tr><td>AE co-adaptation (fine-tune after PCA warm-start)</td><td>true</td></tr><tr><td>AE freeze step (step at which AE is frozen)</td><td>5000</td></tr><tr><td>AE LR</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>AE optimizer</td><td> $\mathrm { A d a m } \ ( \varepsilon = 1 0 ^ { - 8 } ,$ </td></tr><tr><td>Normalizer LR</td><td>AMSGrad)  $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Inner loop</td><td></td></tr><tr><td>Inner-loop steps  $( T _ { \mathrm { i n n e r } } )$ </td><td>20</td></tr><tr><td>Tail inner stepsc</td><td>1000</td></tr><tr><td>Support set size  $( n _ { \mathrm { s u p p o r t } } )$ </td><td>300</td></tr><tr><td>Stim-embedding l2 weight  $( \lambda _ { L 2 } )$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Inner LR  $( \alpha _ { \mathrm { i n n e r } } )$ </td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Inner weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr></table>

<sup>a</sup> A trial is excluded if more than this many timepoints are flagged as outliers by the IQR criterion. <sup>b</sup> Decouples the optimization rates of the two parameter groups. <sup>c</sup> Additional inner-loop steps applied to the stim embedding after the final outer TTA loop, allowing a re-fit on the full support set.

## 3) Test-time adaptation (TTA):

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>TTA steps</td><td>7001</td></tr><tr><td>Support sizes swept</td><td>{500, 1000,2500,5000}</td></tr><tr><td>Max adapted sessions</td><td>20</td></tr><tr><td>Progressive-unfreezing thresholdª</td><td>0 (always)</td></tr><tr><td>Fine-tune basis generator (--unfreeze-bases)</td><td>yes</td></tr><tr><td>Basis-generator LR</td><td> $\dot { 1 } \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Fine-tune basis weights</td><td>no</td></tr><tr><td>Basis-weight LR (when unfrozen)</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr></table>

<sup>d</sup> Basis generator fine-tuning is enabled only when the support size exceeds this value; 0 means always enabled.