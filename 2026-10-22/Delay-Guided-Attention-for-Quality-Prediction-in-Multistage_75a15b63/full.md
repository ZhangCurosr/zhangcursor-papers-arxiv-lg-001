# Delay-Guided Attention for Quality Prediction in Multistage Manufacturing

Jiarui Zhang<sup>a,†</sup>, Zhaoyang Guan<sup>a,†</sup>, Zhixuan Xia<sup>a</sup>, and Naichen Shi<sup>b,∗</sup>

<sup>a</sup> Department of Engineering Sciences and Applied Mathematics, Northwestern University, Evanston, IL, USA;

<sup>b</sup> Department of Industrial Engineering and Management Sciences, Northwestern University, Evanston, IL, USA

## ABSTRACT

Quality prediction in multistage manufacturing is challenging when inter-stage lags vary over time, as measurements taken at the same timestamp may reflect diferent process states. Existing approaches often preserve production order but fuse timestamp-aligned features or model delayed dependencies implicitly, without identifying the upstream history relevant to each downstream observation. We propose Delay-Guided Attention for Multi-stage Forecasting (DAMF), which explicitly estimates inter-stage lag distributions and uses them to guide cross-stage informa tion transfer for final quality prediction. For each adjacent stage pair, a lag predictor compares the current downstream representation with candidate lagged upstream representations. An occurrence–magnitude factorization separately estimates the probability of a positive lag and its conditional magnitude, yielding a sampleand stage-pair-specific distribution over discrete lag states without requiring plantspecified delays at inference. DelayAlignment incorporates this soft distribution as a temporal prior into diferentiable cross-stage attention, while gated residual fusion regulates information propagation across production stages. For interpretation, segment-wise Viterbi decoding converts pointwise lag distributions into temporally coherent lag paths, whereas forecasting retains the soft distributions for alignment. On the real-world LiquidSugar dataset, DAMF improves final quality prediction over the compared baselines. Evaluations on two controlled-delay benchmarks show more accurate and stable lag recovery, and ablation studies support the contributions of lag prediction, delay-guided attention, and structured temporal decoding.

## KEYWORDS

Multistage manufacturing; Quality prediction; Lag recovery

## 1. Introduction

Modern industrial production systems often comprise several physically or functionally connected stages in which raw materials are transformed into a series of intermediate products and the final product (Ceglarek et al. 2004; Du et al. 2015; Filz, Bosse, and Herrmann 2024). Although diferent stages perform diferent operations, their process states are not independent. Variations in equipment condition and operating settings at an upstream stage may propagate to downstream measurements and ultimately afect final product quality (Agrawal, Lawless, and Mackay 1999; Cho and Lim 2024; Du et al. 2015; Lawless, Mackay, and Robinson 1999; Yan et al. 2021; Zhang et al. 2022).

The liquid sugar production process is a representative example. As illustrated in Figure 1, the feed first enters a melter in Stage 1. The resulting intermediate stream is processed by two parallel saturators in Stage 2, whose outputs are combined and transferred to Saturator 3 in Stage 3 before the final product is obtained (Cho and Lim 2024). Sensors associated with individual production stages continuously record stage-specific process variables, providing monitoring signals about the production process (Jiang et al. 2021; Qin 2012).

![](images/7763cd46131e918d8898e4094894626e3cc227868612b9c0d7002f83ce23e71e.jpg)  
Figure 1. Schematic illustration of the sequential multi-stage Liquid Sugar production process.

The objective of this study is to predict the final product quality from the complete multi-stage sensor history with structural understandings that can support process diagnosis. We formulate this objective as two related tasks: forecasting the final quality of the manufacturing system output and describing the signal-propagation structure of the system. The two tasks are connected because final quality depends not only on the states recorded at individual stages, but also on which upstream historical patterns are associated with the current downstream condition.

To understand the cross-stage signal propagation in multi-stage manufacturing, we study the signal lag. Although all measurements are often recorded under a common clock, equal timestamps do not necessarily refer to the same material condition or production event. Transportation, intermediate processing, and equipment response may cause a pattern recorded upstream to appear downstream after a temporal ofset (Cattaldo, Ferrer, and M˚age 2024; Meji´c et al. 2019; Ofermans et al. 2024). As a result, timestamp-based alignment is needed to associate a downstream observation with an upstream state (Ofermans et al. 2020; Xiong et al. 2017; Yao and Ge 2020). Estimating this ofset can also provide evidence about when and at which stage a quality-relevant deviation first becomes observable.

Accurate estimation of inter-stage lags is often challenging for several reasons. Diferent stage pairs involve diferent transportation and processing mechanisms, and their efective lags may vary with production rate, operating condition, equipment status, or material characteristics (Reitz et al. 2013; Wang et al. 2021; Zhao et al. 2021). A single global lag or static timestamp shift prescribed by domain knowledge is therefore unlikely to fully capture the dynamics of the multi-stage process (Griesing-Scheiwe et al. 2020; Yang and Li 2017; Yao and Ge 2020). Moreover, not every upstream variation remains equally informative after intermediate processing, as some patterns may be attenuated, whereas others become amplified. Another challenge is that lags are not directly observed and have been estimated from noisy observations. The statistical variations in monitoring signal complicates the estimation of lags across stages (Ni, Xiao, and Shah 2010; Wang, Li, and Cheng 2024; Zheng and Feng 1990).

In literature, a few lag-detection and temporal causal-discovery methods study delayed relationships between time-series variables (Pamfil et al. 2020; Runge et al. 2019). However, these methods often model lag prediction as a task independent of quality prediction. When the two tasks are handled separately, prediction models do not benefit from sample-dependent lag distributions for diferent consecutive stage, which implicitly discards the interaction patterns across stages.

Can we build a synergistic framework for multi-stage manufacturing signal analysis that transfers the information across the two tasks? Since lag prediction is often a local analysis between two adjacent manufacturing stages, while the quality prediction involves the comprehensive analysis of the entire system, combining them is not easy. To bridge the gap, we propose Delay-Guided Attention for Multi-stage Forecasting(DAMF), which leverages the information from cross-stage lag estimation for final quality prediction. DAMF consists of two stages: lag identification and quality prediction. In the lag identification stage, DAMF compares every current downstream representation with every representation from the upstream stage and estimates a data-dependent distribution of the lag. The resulting distribution serves as a prior in a diferentiable DelayAlignment module, which calculates an attention map that prescribes a greater attention to upstream positions supported by the estimated lag. For lag recovery and interpretation, a Viterbi decoder converts the pointwise lag distributions into temporally coherent lag paths. Quality forecasting instead retains the soft distributions and uses them as priors in DelayAlignment. In the quality prediction stage, the quality model inherits the structural information from lag prediction through cross-attention and outputs the final quality label.

The main contributions of this work are summarized as follows:

(1) Delay-aware multi-stage prediction framework. We propose DAMF, a multi-stage quality prediction framework that jointly models inter-stage delays and predicts final product quality.

(2) Inter-stage lag estimate. We develop an inter-stage attention-based lag estimator that compares downstream representations with candidate upstream histories for each adjacent stage pair. The lag estimation module outputs both the occurrence of the lag and the magnitude of the lag, providing flexibility in industrial practice. We also propose a Viterbi filter to stabilize the lag estimates.

(3) Lag-guided time series analysis. We introduce a diferentiable DelayAlignment module that incorporates the estimated lag distribution as a temporal prior to the attention map. Based on the attention map, we further propose a gated residual fusion module for final-quality prediction.

(4) Experimental validation. Experiments on the real LiquidSugar dataset and two controlled-delay benchmarks, together with systematic ablation studies, validate the efectiveness of DAMF for both final-quality prediction and inter-stage lag recovery.

## 2. Related Work

Multi-stage quality-prediction models. Data-driven soft sensors estimate dificult-to-measure quality variables from readily available process measurements (Kadlec, Gabrys, and Strandt 2009), but conventional soft-sensing models do not usually encode the ordered structure of a multi-stage manufacturing system. Several stage-aware models have been developed to address this limitation. The deep multi-stage multi-task learning model proposed by Yan et al. (2021) follows the sequential production structure and jointly predicts multiple stage-wise quality variables through shared representations. Wang et al. (2023) use a multi-scale convolutional network with control gates to represent cumulative variation across stages and a multi-gate mixture-of-experts model to learn multiple quality tasks. PGAT (Zhang et al. 2022) represents machines and production paths as a directed graph and applies path-enhanced bidirectional graph attention to aggregate quality-relevant information. For continuous processes without intermediate quality measurements, MMPNet (Cho et al. 2024) sequentially injects stage-specific inputs into a feedforward network, while MultiStageNet (Cho and Lim 2024) combines temporal encoders, interstage blocks, and stacked multi-stage modules to learn sequential dependencies without intermediate output or lead-time labels. These models capture stage-wise dependencies through multi-task learning, gated feature transfer, graph message passing, or sequential interstage modules. However, their cross-stage temporal relationships remain implicit: they do not expose a sample-dependent lag distribution for each adjacent stage pair and use it to align upstream and downstream observations.

Time-series forecasting models. Deep time-series models provide alternative architectures for extracting temporal patterns from industrial sensor data. PatchTST (Nie et al. 2023) divides each variable into temporal patches and applies channel-independent Transformer encoding, whereas iTransformer (Liu et al. 2024) treats individual variables as tokens and learns their dependencies through attention. TimeMixer (Wang et al. 2024a) represents temporal variation at multiple resolutions and mixes information across scales. State-space models such as MambaTS (Cai et al. 2024) and TimeMachine (Ahamed and Cheng 2024) provide eficient mechanisms for capturing long-range dependencies in multivariate sequences. These models are efective general-purpose forecasting architectures, but they treat the input primarily as a univariate or multivariate sequence. When applied directly to multi-stage manufacturing data, the known production order and adjacent-stage relationships are not explicitly used to determine which upstream historical position should be associated with a downstream observation. Temporal ofsets between stages are therefore absorbed into the learned representations rather than represented as explicit, interpretable variables.

Lag identification and structured temporal modeling. Methods for laggeddependency analysis explicitly examine delayed relationships between time-series variables. Granger causality analysis (Granger 1969) evaluates whether past observations of one variable improve the prediction of another. PCMCI (Runge et al. 2019) uses conditional-independence testing to identify lagged dependencies, while DYNOTEARS (Pamfil et al. 2020) learns time-lagged directed structures through continuous optimization. These methods identify candidate lagged relationships under their respective assumptions, but their outputs are generally variable-level dependencies or temporal graphs estimated over a dataset or analysis window. They are not designed to produce a sample-dependent lag distribution for each manufacturing-stage pair and use that distribution directly in final-quality forecasting. Once point-wise lag scores are available, structured sequence methods can be used to improve temporal consistency. The Viterbi algorithm recovers the most probable state path from emission and transition scores (Rabiner 1989; Viterbi 1967), while explicit-duration models reduce unrealistically short assignments by modeling state persistence (Yu 2010). These decoders smooth an existing score sequence but do not estimate lag probabilities from cross-stage sensor features or connect them to the forecasting mechanism. DAMF links these previously separate components by estimating sampleand stage-pair-specific lag distributions, using them to guide cross-stage alignment for quality forecasting, and decoding them into coherent lag paths for interpretation.

![](images/0d0cd48311f77f8c96a63b611f874aa48803e5d4088d20061e6025a6d55654a6.jpg)  
Figure 2. Overall architecture of DAMF. Edge-wise lag distributions guide DelayAlignment and gated residual fusion across successive processing stages.

## 3. Methodology

In this section, we will first introduce the modeling of inter-stage lag in the monitoring signal, and then present our methods to leverage the identified lag into quality prediction.

## 3.1. Lag modeling and DAMF

Consider a multi-stage manufacturing process with an ordered set of stages $\mathcal { G } =$ $\{ 0 , 1 , \ldots , M \}$ , where stage 0 denotes the system input and stages $1 , \ldots , M$ denote successive processing stages. At time t, the observation from stage g is denoted by $\mathbf { x } _ { t } ^ { ( g ) } \in \mathbb { R } ^ { d _ { g } }$ , where $d _ { g }$ is the number of variables measured at that stage. Given a window length L, the historical input from stage g is

$$
\mathbf { X } _ { t } ^ { \left( g \right) } = \left[ \mathbf { x } _ { t - L + 1 } ^ { \left( g \right) } , \mathbf { x } _ { t - L + 2 } ^ { \left( g \right) } , \ldots , \mathbf { x } _ { t } ^ { \left( g \right) } \right] ^ { \top } \in \mathbb { R } ^ { L \times d _ { g } } .
$$

Let y<sub>t</sub> denote the final-quality response at time t. Using the multi-stage history $\{ \mathbf { X } _ { t } ^ { ( g ) } \} _ { g = 0 } ^ { M }$ , we aim to predict $y _ { t + H }$ , where H is the forecasting horizon measured in sampling steps. Although the stage-wise signals share a common time index, corresponding process states may not capture the same physical process at the same time. We represent this temporal ofset by a discrete lag state between two adjacent stage

![](images/f739e6be84a1f351c8557771c595cbe3a967564e0e3253a0fd54e9c6c370d2d6.jpg)  
Figure 3. Illustration of an inter-stage lag. An upstream process feature observed at t − k corresponds to a downstream feature observed at t, giving a lag of k sampling steps.

$( g - 1 , g )$ for $g = 1 , \ldots , M$ as:

$$
z _ { t } ^ { ( g ) } \in \mathcal { K } , \qquad K = \{ 0 , 1 , \dots , K \} ,
$$

where K is the maximum candidate lag. The state $z _ { t } ^ { ( g ) } = k$ indicates that the downstream state at time t is aligned with an upstream state observed at time $t - k .$ . Here, the lag represents a signal-level alignment ofset. Figure 3 illustrates a lag of k sampling steps between two adjacent stages.

Based on this formulation, we propose Delay-Guided Attention for Multi-stage Forecasting (DAMF), which consists of inter-stage lag estimation and lag-guided forecasting. For each pair of adjacent stages $( g - 1 , g )$ , the lag predictor estimates the posterior probability distribution of the lag $z _ { t } ^ { ( g ) }$ after observing the signal from the two stages,

$$
q _ { t , k } ^ { ( g ) } = \operatorname* { P r } \left( z _ { t } ^ { ( g ) } = k \mid \mathbf { X } _ { t } ^ { ( g - 1 ) } , \mathbf { X } _ { t } ^ { ( g ) } \right) , \qquad k \in \mathcal { K } .
$$

DelayAlignment combines this distribution with content-based matching to construct an aligned upstream message, which is then integrated with the current-stage representation through gated residual fusion. Repeating this operation across successive stages constructs the representation needed for the final quality forecast $\widehat { y } _ { t + H }$ . Figure 2 summarizes the overall architecture, and the following subsections describe the two components in detail.

## 3.2. Inter-stage lag estimation

As discussed, for each adjacent-stage edge $( g - 1 , g )$ , the lag predictor maps the upstream and downstream sensor observations to a pointwise distribution over the candidate lags. This subsection introduces the lag-matching mechanism and the training strategy, then presents an inference strategy used to recover a temporally smooth lag path. The pointwise distribution of the lag is subsequently used for lag-guided forecasting.

## 3.2.1. Lag matching and estimation

For an adjacent-stage edge $( g - 1 , g )$ , we refer to $\mathbf { X } _ { t } ^ { ( g - 1 ) }$ as the source window and $\mathbf { X } _ { t } ^ { ( g ) }$ as the target window. Both contain the most recent $L$ observations up to time t. We encode both windows using gated recurrent unit (GRU) networks (Cho et al. 2014), whose gates control how information from previous observations is retained and updated. Intuitively, a GRU maintains a hidden state that serves as a running summary of the sequential observation. At each time step, a reset gate determines how much of the previous summary is used to inform the current hidden state. Then, an update gate interpolates between the previous hidden state and the current one. This gating mechanism allows the network to identify persistent sequential patterns over the history. We leverage the flexibility and denoising capability of GRU to extract important features from raw sensing measurements.

The output of GRU from source window is a sequence of hidden states $\{ \mathbf { h } _ { t , j } ^ { ( g - 1 ) } \} _ { j = 1 } ^ { L }$ representing the latent features extracted, where position $j$ corresponds to time $t -$ $L + j$ . On the target side, the GRU similarly summarizes the latent information from the corresponding time window into a context vector $\mathbf { c } _ { t } ^ { ( g ) }$ . For candidate lag $k , \mathbf { h } _ { t , L - k } ^ { ( g - 1 ) }$ therefore incorporated the source semantic information corresponding to time $t - k$ We assume $K \leq L - 1$

To model the semantic correspondence between the target context and the source representation under candidate lag k, we propose a similarity metric denoted by,

$$
s _ { t , k } ^ { ( g ) } = \frac { \left( W _ { m } \mathbf { h } _ { t , L - k } ^ { \left( g - 1 \right) } \right) ^ { \top } \mathbf { c } _ { t } ^ { ( g ) } } { \sqrt { d _ { h } } } + b _ { k } , \qquad k \in \mathcal { K } ,\tag{1}
$$

where $W _ { m }$ is a projection matrix that maps the source representation to the space of the target representation, $d _ { h }$ is the dimension of this space, and $b _ { k }$ is a lag-specific learnable bias. Ideally, a larger $s _ { t , k } ^ { ( g ) }$ indicates stronger correspondence under lag k.

After calculating the matching scores $s _ { t , k } ^ { ( g ) }$ for all candidate lags $k ,$ we pass them into two prediction heads that estimate lag occurrence and magnitude. The occurrence head will output $o _ { t } ^ { ( g ) }$ , which represents the probability that there is a non-zero lag $z _ { t } ^ { ( g ) } > 0$ For magnitude estimation, the positive-lag head produces $K$ outputs, one for each lag $k = 1 , \ldots , K$ , to predict probability that the lag is $k .$ Combining both outputs yields a categorical distribution over all candidate lags:

$$
q _ { t , 0 } ^ { ( g ) } = 1 - o _ { t } ^ { ( g ) } , \qquad q _ { t , k } ^ { ( g ) } = o _ { t } ^ { ( g ) } \pi _ { \mathrm { p o s } , t } ^ { ( g ) } ( k ) , \quad k = 1 , \ldots , K .\tag{2}
$$

One can verify that the vector $\mathbf { q } _ { t } ^ { \left( g \right) } = [ q _ { t , 0 } ^ { \left( g \right) } , \ldots , q _ { t , K } ^ { \left( g \right) } ]$ defines a normalized distribution over $\kappa ,$ since the positive-lag probabilities sum to one.

## 3.2.2. Lag-predictor training

After defining the parametric model to predict the occurrence of the lag and its distribution, we will introduce the training objectives to fit these models.

Remember that primary task of DAMF is to recover the signal delay dynamics and predict the quality label simultaneously. We therefore design two main empirical loss functions following the hierarchical factorization in Eq. (2). The occurrence loss ${ \mathcal { L } } _ { \mathrm { o c c } }$ encourages the lag occurrence prediction $o _ { t }$ to match the ground truth binary label. A focal-style objective is used to reduce the influence of the class imbalance in the training dataset, as most observations do not admit lags. Within the positive-lag subset, the magnitude loss ${ \mathcal { L } } _ { \mathrm { p o s } }$ is a cross-entropy loss that encourages $\pi _ { \mathrm { p o s } , t }$ to identify the correct ground-truth lag among $\{ 1 , \ldots , K \}$

To further strengthen the robustness of the prediction, we introduce three auxiliary objectives. The relaxed Top-N loss $\mathcal { L } _ { \mathrm { T o p N } }$ operates on the matching scores $s _ { t , k }$ and encourages the ground-truth lag $\ell _ { t }$ to remain among the leading N candidates without requiring it to be ranked first. The large-lag loss $\mathcal { L } _ { \mathrm { l a r g e } }$ emphasizes the positiveoccurrence supervision for samples satisfying $\ell _ { t } \geq K _ { \mathrm { l a r g e } }$ , reducing the tendency of large lags to collapse to the zero-lag state. Finally, for samples whose ground-truth lag is retained in the Top-N candidate set $\mathcal { C } _ { t } .$ , the selector loss $\mathcal { L } _ { \mathrm { s e l } }$ uses the selector probabilities $a _ { t } ( j )$ to refine the ranking among the retained candidates. The selector provides auxiliary supervision only and is not used to construct $\mathbf { q } _ { t } ^ { \left( g \right) }$ at inference.

The complete lag-predictor objective, denoted by $\mathcal { L } _ { \mathrm { l a g } }$ , combines $\mathcal { L } _ { \mathrm { o c c } } , \mathcal { L } _ { \mathrm { p o s } }$ , and $\mathcal { L } _ { \mathrm { T o p N } }$ , together with the weighted auxiliary terms $\mathcal { L } _ { \mathrm { l a r g e } }$ and $\mathcal { L } _ { \mathrm { s e l } }$ . Their complete mathematical definitions are provided in Appendix C.

## 3.2.3. Denoised inference of the lag

Even with well-trained lag predictors, we notice that the direct lag predictions tend to fluctuate over time. We attribute such fluctuations to the stochasticity of the sensing data and training dynamics. To encourage temporal consistency among lag predictions, we propose to use a Viterbi algorithm to smooth the fluctuations.

More specifically, we use $\mathcal { T } _ { r } = ( t _ { 1 } , \ldots , t _ { n _ { r } } )$ denote the $n _ { r }$ time indices within a time window, ordered chronologically. For the manufacturing stage $( g - 1 , g )$ , a candidate path $\mathbf { z } _ { r } ^ { ( g ) } \in \mathcal { K } ^ { n _ { r } }$ assigns a lag state $z _ { t _ { i } } ^ { ( g ) }$ to each time $t _ { i } .$ . Here, $\kappa ^ { n _ { \tau } }$ denotes the set of all lag sequences of length $n _ { r }$ with entries in K. We obtain the decoded path by maximizing the regularized score:

$$
\hat { \mathbf { z } } _ { r } ^ { ( g ) } = \arg \operatorname* { m a x } _ { \mathbf { z } _ { r } ^ { ( g ) } \in { \cal K } ^ { n _ { r } } } \left\{ \sum _ { i = 1 } ^ { n _ { r } } \log \left( q _ { t _ { i } , z _ { t _ { i } } ^ { ( g ) } } ^ { ( g ) } + \epsilon \right) + \sum _ { i = 2 } ^ { n _ { r } } A \left( z _ { t _ { i - 1 } } ^ { ( g ) } , z _ { t _ { i } } ^ { ( g ) } \right) \right\} ,\tag{3}
$$

where $\epsilon > 0$ is a small constant added for numerical stability, and $A ( a , b )$ denotes the transition score from lag state a to lag state b. In this objective, the first sum measures agreement between the selected lag and the pointwise predicted lag scores, while the second penalizes abrupt changes between successive lag states. For a transition from state a to state b, with $a , b \in \mathcal K$ , we define

$$
\begin{array} { r l } & { A ( a , b ) = - \eta _ { \mathrm { s w } } \mathbb { I } ( a \neq b ) - \eta _ { \mathrm { j u m p } } | a - b | } \\ & { \quad \quad - \eta _ { \mathrm { d r o p } } \mathbb { I } ( a \geq \kappa \wedge 0 < b < \kappa ) } \\ & { \quad \quad - \eta _ { \mathrm { o f f } } \mathbb { I } ( a \geq \kappa \wedge b = 0 ) - \eta _ { \mathrm { o n } } \mathbb { I } ( a = 0 \wedge b \geq \kappa ) . } \end{array}\tag{4}
$$

Here, $\kappa \in \{ 1 , \ldots , K \}$ is the threshold defining large-lag states, and $\mathbb { I } ( \cdot )$ is the indicator function. All penalty weights are nonnegative. The coeficients $\eta _ { \mathrm { s w } }$ and $\eta _ { \mathrm { j u m p } }$ penalize state switching and the magnitude of a lag change, respectively. Additional penalties $\eta _ { \mathrm { d r o p } } , \eta _ { \mathrm { o f f } }$ , and $\eta _ { \mathrm { o n } }$ apply to transitions from a large lag to a smaller positive lag, from a large lag to zero, and from zero to a large lag, respectively. Because the first two terms depend only on $| a - b |$ , they define a symmetric Toeplitz base transition matrix. The remaining terms introduce transition-specific corrections. Figure 4 illustrates the

resulting transition structure.

![](images/0cf153b78b2c43613755820b83bf0517898c003e2866a9767f886718b1a75b66.jpg)  
Figure 4. Transition-score structure for lag-path decoding under the experimental parameter settings. Darker cells indicate higher scores and smaller penalties.

We use the decoded path only for lag recovery and interpretation. For quality forecasting, DAMF instead uses the pointwise distribution $\mathbf { q } _ { t } ^ { \left( g \right) }$ as a soft prior in Delay-Alignment.

## 3.3. Lag-guided forecasting

When the lag distribution predictors are optimized, we use these predictions as alignment priors to predict the final quality label. For this quality prediction task, we similarly employ GRU to encode the temporal observations into semantic embeddings. For stage $^ { g , }$ we use ${ \bf e } _ { t } ^ { ( g ) }$ to denote the semantic representation at time t, and let $\mathbf { u } _ { t } ^ { ( g ) }$ to denote the representation of the upstream information. At the system input, we set ${ \bf u } _ { t } ^ { ( 0 ) } = { \bf e } _ { t } ^ { ( 0 ) }$

For two adjacent stages $( g - 1 , g )$ , the DelayAlignment module compares ${ \bf e } _ { t } ^ { ( g ) }$ with upstream representations at candidate lagged positions. For each valid lag $k ,$ we construct the cross-attention score as

$$
\widetilde { \alpha } _ { t , k } ^ { ( g ) } = \frac { \left. W _ { Q } \mathbf { e } _ { t } ^ { ( g ) } , W _ { K } \left( \mathbf { u } _ { t - k } ^ { ( g - 1 ) } + \mathbf { r } _ { k } \right) \right. } { \sqrt { d _ { a } } } + \beta _ { k } + \lambda \log \left( q _ { t , k } ^ { ( g ) } + \epsilon \right) .\tag{5}
$$

Here, $W _ { Q }$ and $W _ { K }$ project the downstream and upstream representations into query and key vectors of dimension $d _ { a } .$ and $\langle \cdot , \cdot \rangle$ denotes their inner product. The vector $\mathbf { r } _ { k }$ is an embedding of lag $k ,$ , and $\beta _ { k }$ is a lag-specific learnable bias in DelayAlignment. The coeficient λ controls the contribution of the lag prior, while $\epsilon > 0$ ensures numerical stability. Setting $\lambda = 0$ removes the lag prior.

Applying softmax to $\widetilde { \alpha } _ { t , k } ^ { ( g ) } / \tau$ over k gives the alignment weights $\pi _ { t , k } ^ { ( g ) }$ , where $\tau > 0$ is the temperature parameter. These weights depend on both the encoded representations and the estimated lag distribution. We use them to aggregate the upstream information

into a message:

$$
\mathbf { m } _ { t } ^ { ( g ) } = \sum _ { k \in \mathcal { K } _ { t } } \pi _ { t , k } ^ { ( g ) } W _ { V } \mathbf { u } _ { t - k } ^ { ( g - 1 ) } ,\tag{6}
$$

where $W _ { V }$ is the value projection matrix. This weighted aggregation allows multiple candidate lags to contribute, rather than selecting a single upstream position.

To combine the aligned message with the current-stage representation, we introduce two nonlinear information fusion mechanisms:

$$
\begin{array} { r l } & { \gamma _ { t } ^ { ( g ) } = \sigma \left( W _ { \gamma } ^ { ( g ) } [ \mathbf { e } _ { t } ^ { ( g ) } ; \mathbf { m } _ { t } ^ { ( g ) } ] + \mathbf { b } _ { \gamma } ^ { ( g ) } \right) , } \\ & { \mathbf { u } _ { t } ^ { ( g ) } = \mathbf { e } _ { t } ^ { ( g ) } + \gamma _ { t } ^ { ( g ) } \odot \psi ^ { ( g ) } \left( \mathbf { m } _ { t } ^ { ( g ) } \right) , \qquad g = 1 , \dots , M . } \end{array}\tag{7}
$$

Here, $\sigma$ is the nonlinear activation function such as a sigmoid function, $[ \cdot ; \cdot ]$ denotes vector concatenation, and $W _ { \gamma } ^ { ( g ) }$ and ${ \bf b } _ { \gamma } ^ { ( g ) }$ are learnable gate parameters. The mapping $\psi ^ { ( g ) }$ transforms the message to the dimension of ${ \bf e } _ { t } ^ { ( g ) }$ , and $\odot$ denotes elementwise multiplication. This update retains the current-stage representation while adding a gated contribution from upstream. Figure 5 illustrates the alignment and fusion operations for an adjacent-stage pair.

Repeating the update across successive stages produces ${ \bf u } _ { t } ^ { ( M ) }$ . A prediction head $\phi _ { y }$ then gives the final-quality forecast $\hat { y } _ { t + H } = \phi _ { y } ( \mathbf { u } _ { t } ^ { ( M ) } )$

Training proceeds in two phases with separate objectives. We first train the lag predictor with $\mathcal { L } _ { \mathrm { l a g } }$ , as described in Section 3.2.2. After completing lag-predictor training, we use its estimated distributions as alignment priors and train the forecasting model. The objective for this second phase is the mean absolute error (MAE) between observed and predicted quality:

$$
\mathcal { L } _ { y } = \frac { 1 } { N _ { y } } \sum _ { i = 1 } ^ { N _ { y } } \left| y _ { i + H } - \hat { y } _ { i + H } \right| ,\tag{8}
$$

where $N _ { y }$ is the number of forecasting training samples.

During forecasting training, the lag predictor is held fixed and provides the pointwise distributions $q _ { t } ^ { ( g ) }$ as soft alignment priors. The forecasting loss $L _ { y }$ updates only the stage encoders in the forecasting backbone, DelayAlignment, the gated fusion modules, and the prediction head.

## 4. Experiments

This section evaluates the proposed DAMF framework from three perspectives. First, we evaluate whether the proposed lag-recovery pipeline can recover meaningful interstage lag patterns on controlled-delay benchmarks. Second, we examine whether delayintegrated modeling improves final quality prediction on real multi-stage manufacturing data. Third, we conduct ablation studies to analyze the efects of lag-predictor architecture, loss components, structured temporal decoding, and delay priors.

![](images/5249dfcb371054d37a3d2b3064e0062094515e237723fb1e6ff2b36fe817c90f.jpg)  
Figure 5. Lag-guided alignment and fusion for an adjacent-stage pair. DelayAlignment combines contentbased matching with the estimated lag distribution to compute attention weights over upstream candidates. The resulting upstream message is combined with the current-stage representation through gated residual fusion.

## 4.1. Datasets

We evaluate DAMF using one real-world manufacturing dataset and two controlleddelay benchmarks. The original LiquidSugar dataset is used for final quality prediction, while a delay-injected version of LiquidSugar and a CausalAI-based synthetic benchmark are used for inter-stage lag recovery. LiquidSugar (Cho and Lim 2024) contains 22,854 observations with a nominal sampling interval of 15 minutes. Each observation consists of one timestamp, 24 process variables, and one final quality target, yield flow. The process variables include seven feed variables, two Stage-1 variables, twelve Stage-2 variables, and three Stage-3 variables. Observations separated by more than 120 minutes are treated as diferent continuous segments. Within each segment, measurements are regularized onto a 15-minute grid, and short missing intervals are filled by linear interpolation. The resulting complete segments are divided chronologically into training and test sets using an approximate 80/20 ratio, with identical preprocessing and data splits used for all compared methods.

For controlled lag-recovery evaluation, we further construct a delay-injected version of LiquidSugar. Within each split, 50% of the eligible segments are randomly selected for lag injection, while the remaining segments retain the zero-lag condition. The candidate lag states are {0, 1, 2, 3, 4, 5} sampling steps, corresponding to physical delays ranging from 0 to 75 minutes in 15-minute increments. The injected lag sequence is retained as the ground-truth trajectory at each time step and is used to evaluate the accuracy of inter-stage lag recovery.

We additionally construct a CausalAI-based synthetic benchmark to evaluate lag recovery under fully controlled temporal dependencies. The benchmark consists of a bivariate time series with one source variable and one target variable, where the target is generated from the source under a prescribed time-varying lag. For this benchmark, the candidate lag states are {0, 1, 2, 3, 4, 5, 6}, with an 80/20 train/test split and lag changes introduced into 50% of the eligible regions. Both block-shaped and smooth bump-shaped lag trajectories are included to represent persistent and transient delay variations, respectively. Since the ground-truth lag trajectory g(t) is explicitly specified during data generation, this benchmark enables direct evaluation of both lag-recovery accuracy and the ability to track time-varying delay dynamics.

## 4.2. Experimental Settings

The experiments address two tasks: final quality prediction and inter-stage lag recovery. All methods follow the data splits and preprocessing procedures described in Section 4.1. For final quality prediction, DAMF is compared with three groups of baselines. The machine-learning baselines include Linear Regression, XGBoost (Chen and Guestrin 2016), Random Forest (Breiman 2001), and LightGBM (Ke et al. 2017). The deep sequential baselines include LSTM (Hochreiter and Schmidhuber 1997), GRU (Cho et al. 2014), and Transformer (Vaswani et al. 2017). The multistage baselines include TimeXer (Wang et al. 2024b), PEGNN-SS (Faghih Niresi et al. 2024), and MultistageNet (Cho and Lim 2024).

These three baselines are selected for their complementary modeling capabilities. TimeXer captures temporal dependencies and exogenous-variable interactions, providing a reference for attention-based prediction. PEGNN-SS incorporates physical knowledge into sensor graphs, representing physics-enhanced modeling of process-variable interactions. MultistageNet learns sequential stage dependencies and lead-time relationships without intermediate output or lead-time labels, making it directly relevant to our manufacturing setting. Together, they provide complementary comparisons for evaluating DAMF’s explicit modeling of inter-stage delays. Lag recovery is evaluated separately on the controlled-delay versions of LiquidSugar and the CausalAI-based synthetic benchmark. Both benchmarks contain block-shaped and smooth bump-shaped lag patterns, representing persistent and transient delay changes, respectively.

For lag-recovery evaluation, DAMF is compared with MNet and WCC. MNet uses the hidden lag predictions produced by MultistageNet (Cho and Lim 2024) as its lag estimates. WCC applies window-based cross-correlation, a classical time-delay estimation approach (Carter 1987; Knapp and Carter 1976), followed by a conventional Viterbi decoder.

The input window length is set to $L = 1 2$ throughout the experiments. For lag recovery, the maximum candidate lag is set to $K = 5$ for LiquidSugar and $K = 6$ for CausalAI. The lag predictors use two-layer GRU encoders with a hidden dimension of 64 and are trained for 60 epochs using Adam with a learning rate of $5 \times 1 0 ^ { - 4 }$ . The batch sizes are 256 and 64 for LiquidSugar and CausalAI, respectively.

For final quality prediction on LiquidSugar, we use a forecasting horizon of $H = 1$ The DAMF forecasting backbone also employs two-layer GRU encoders with a hidden dimension of 64. It is trained for 50 epochs using Adam with a learning rate of $1 0 ^ { - 3 }$ and a batch size of 256, while the pretrained lag predictor remains fixed.

## 4.2.1. Evaluation metrics

Final quality prediction is evaluated using RMSE, MAE, and $R ^ { 2 }$ , where lower RMSE and MAE and higher $R ^ { 2 }$ indicate better performance. For lag recovery, let $\tau$ denote the set of evaluated test indices, and let $d _ { t }$ and $\hat { d } _ { t }$ denote the true and predicted lags, respectively, measured in sampling steps. We define $\mathcal { T } = \{ t \in \mathcal { T } : d _ { t } > 0 \} , \mathcal { O } = \{ t \in$ $\mathcal { T } : d _ { t } = 0 \}$ , and $\mathcal { L } = \{ t \in \mathcal { T } : d _ { t } \geq 4 \}$ as the sets of true positive-lag, zero-lag, and large-lag test points, respectively.

Lag occurrence. For the controlled-delay comparison, let $\mathrm { T P } _ { d }$ and $\mathrm { F P } _ { d }$ count positive-lag detections in $\mathcal { T }$ and ${ \mathcal { O } } _ { : }$ respectively. For DAMF, a positive lag is detected when $\hat { d } _ { t } > 0$ . Precision and the false alarm rate (FAR) are defined as

$$
\mathrm { P r e c i s i o n } = \frac { \mathrm { T P } _ { d } } { \mathrm { T P } _ { d } + \mathrm { F P } _ { d } } , \qquad \mathrm { F A R } = \frac { \mathrm { F P } _ { d } } { | \mathcal { O } | } .\tag{9}
$$

These metrics assess whether a lag is detected, rather than whether its magnitude is estimated exactly. For the lag-predictor ablations, occurrence F1 (Occ F1) uses the predicted occurrence probability $o _ { t }$ . A ground-truth label is positive when $d _ { t } \ > \ 0$ and a predicted label is positive when $o _ { t } \geq 0 . 5$ . Let $\mathrm { T P } _ { o } , \mathrm { F P } _ { o } ,$ and $\mathrm { F N } _ { o }$ denote the true-positive, false-positive, and false-negative counts obtained from these labels over all test points. Then

$$
\mathrm { O c c ~ F 1 } = \frac { 2 \mathrm { T P } _ { o } } { 2 \mathrm { T P } _ { o } + \mathrm { F P } _ { o } + \mathrm { F N } _ { o } } .\tag{10}
$$

Lag magnitude and peak recovery. The mean absolute error over true positive-lag points is reported as Lag MAE in the ablations and In-bump MAE in the controlleddelay comparison. Outside mean delay measures the average predicted lag at true zero-lag points. These metrics are defined as

$$
\begin{array} { c } { \displaystyle \mathrm { L a g ~ M A E } = \frac { 1 } { | \mathcal T | } \sum _ { t \in \mathcal T } | \hat { d } _ { t } - d _ { t } | , } \\ { \displaystyle \mathrm { O u t s i d e ~ m e a n ~ d e l a y } = \frac { 1 } { | \mathcal O | } \sum _ { t \in \mathcal O } \hat { d } _ { t } . } \end{array}\tag{11}
$$

For peak value error, let $\mathcal { T } _ { b } , b = 1 , \ldots , B .$ , denote the maximal contiguous positive-lag intervals within individual data segments. We define

$$
\mathrm { P e a k \ V a l u e \ E r r o r } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left| \operatorname* { m a x } _ { t \in \mathcal { T } _ { b } } \hat { d } _ { t } - \operatorname* { m a x } _ { t \in \mathcal { T } _ { b } } d _ { t } \right| .\tag{12}
$$

This metric compares peak heights rather than peak timing; the predicted and true maxima need not occur at the same time point.

Pointwise accuracy. Exact-match accuracy $( \operatorname { A c c } )$ and one-step-tolerance accuracy

(±1 Acc) are computed over all test points, including zero-lag points:

$$
\begin{array} { r } { \begin{array} { r } { \mathrm { A c c } = \frac { \left| \left\{ t \in { \mathscr { T } } : \hat { d } _ { t } = d _ { t } \right\} \right| } { \left| { \mathscr { T } } \right| } , } \\ { \pm 1 \ \mathrm { A c c } = \frac { \left| \left\{ t \in { \mathscr { T } } : \left| \hat { d } _ { t } - d _ { t } \right| \leq 1 \right\} \right| } { \left| { \mathscr { T } } \right| } . } \end{array} } \end{array}\tag{13}
$$

Large-lag recovery. The remaining three metrics are evaluated only over the true large-lag set $\mathcal { L } \mathrm { : }$

$$
{ \mathrm { L a r g e - l a g ~ R e c a l l } } = { \frac { | \{ t \in { \mathcal { L } } : { \hat { d } } _ { t } \geq 4 \} | } { | { \mathcal { L } } | } } ,
$$

$$
\mathrm { C o l l a p s e ~ t o ~ } 0 = \frac { | \{ t \in \mathcal { L } : \hat { d } _ { t } = 0 \} | } { | \mathcal { L } | } ,\tag{14}
$$

$$
{ \mathrm { C o l l a p s e ~ t o ~ s m a l l } } = { \frac { | \{ t \in { \mathcal { L } } : 1 \leq { \hat { d } } _ { t } \leq 3 \} | } { | { \mathcal { L } } | } } .
$$

Large-lag Recall measures whether a true large lag is still predicted as large, without requiring an exact match. Collapse to 0 and Collapse to small measure the fractions of true large lags underestimated as zero and as positive lags of one to three sampling steps, respectively. Higher precision, Occ F1, Acc, ±1 Acc, and Large-lag Recall indicate better performance; lower values are preferred for the other lag-recovery metrics.

## 4.3. Main Results

Table 1. Final quality prediction performance on the LiquidSugar dataset. The best result for each metric is highlighted in bold.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="3">LiquidSugar</td></tr><tr><td>RMSE↓</td><td>MAE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td rowspan="7">Machine Learning Methods</td><td>Linear Regression</td><td> $8 . 8 6 \pm 0 . 0 0$ </td><td> $6 . 2 3 \pm 0 . 0 0$ </td><td> $0 . 0 6 \pm 0 . 0 0$ </td></tr><tr><td>XGBoost</td><td> $6 . 1 2 \pm 0 . 0 5$ </td><td> $4 . 5 2 \pm 0 . 1 1$ </td><td> $0 . 5 5 \pm 0 . 0 1$ </td></tr><tr><td>LSTM</td><td> $1 0 . 2 4 \pm 3 . 5 8$ </td><td> $7 . 4 3 \pm 1 . 9 2$ </td><td> $- 0 . 3 6 \pm 0 . 9 7$ </td></tr><tr><td>Random Forest</td><td> $6 . 7 1 \pm 0 . 0 5$ </td><td> $5 . 0 6 \pm 0 . 0 6$ </td><td> $0 . 4 6 \pm 0 . 0 1$ </td></tr><tr><td>GRU</td><td> $7 . 4 0 \pm 0 . 3 1$ </td><td> $5 . 6 1 \pm 0 . 2 8$ </td><td> $0 . 3 4 \pm 0 . 0 6$ </td></tr><tr><td>LightGBM</td><td> $5 . 9 8 \pm 0 . 0 0$ </td><td> $4 . 3 9 \pm 0 . 0 0$ </td><td> $0 . 5 7 \pm 0 . 0 0$ </td></tr><tr><td>Transformer</td><td> $7 . 0 2 \pm 0 . 4 8$ </td><td> $5 . 1 4 \pm 0 . 3 1$ </td><td> $0 . 4 1 \pm 0 . 0 8$ </td></tr><tr><td rowspan="3">Multistage Methods</td><td>TimeXer</td><td> $7 . 9 8 \pm 0 . 1 4$ </td><td> $6 . 2 8 \pm 0 . 1 0$ </td><td> $0 . 2 4 \pm 0 . 0 3$ </td></tr><tr><td>PEGNN-SS</td><td> $9 . 7 4 \pm 0 . 5 3$ </td><td> $7 . 5 1 \pm 0 . 7 0$ </td><td> $- 0 . 1 0 \pm 0 . 1 2$ </td></tr><tr><td>MultistageNet</td><td> $5 . 7 5 \pm 0 . 2 4$ </td><td> $4 . 1 9 \pm 0 . 0 8$ </td><td> $0 . 6 1 \pm 0 . 0 3$ </td></tr><tr><td>Ours</td><td>DAMF</td><td> ${ \bf 5 . 2 4 \pm 0 . 1 3 }$ </td><td> ${ \bf 3 . 6 0 \pm 0 . 1 3 }$ </td><td> $\mathbf { 0 . 6 7 \pm 0 . 0 2 }$ </td></tr></table>

Results are reported as mean ± standard deviation over three random seeds. RMSE and MAE are lower-is-better metrics, while $\overline { { R ^ { 2 } } }$ is a higher-is-better metric.

Table 1 reports the final quality prediction results on the LiquidSugar dataset. Among traditional machine-learning baselines, LightGBM achieves the best overall performance, followed by Random Forest and XGBoost, suggesting that tree-based ensemble methods can capture part of the nonlinear relationship between process variables and final quality. By contrast, generic deep sequential models, including LSTM, GRU, and Transformer, do not consistently outperform traditional baselines, which suggests that directly applying sequence models to timestamp-aligned multi-stage observations is insuficient for this manufacturing prediction task. Multi-stage baselines show mixed performance. TimeXer and PEGNN-SS do not outperform the strongest traditional baselines, whereas MultistageNet achieves the best baseline performance across all three metrics. This comparison indicates that explicitly modeling the stage structure is useful, but representing the production flow alone is still insuficient. Without explicit modeling of dynamic inter-stage delays, upstream and downstream process states may remain imperfectly matched. DAMF achieves the best performance across all three metrics. Compared with the strongest baseline, MultistageNet, DAMF consistently reduces prediction errors and improves the explained variance of the final quality response. These improvements show that estimating inter-stage delays and using them to guide multi-stage fusion provides additional predictive benefit beyond modeling the stage structure alone.

## 4.4. Lag recovery on controlled-delay benchmarks

## 4.4.1. Quantitative comparison

We compare DAMF with MNet and WCC on the two controlled-delay benchmarks using the metrics defined in Section 4.2.1: precision, FAR, In-bump MAE, Outside mean delay, and Peak Value Error. Figure 6 summarizes the quantitative results on the two controlled-delay benchmarks. DAMF achieves higher precision and lower FAR than MNet and WCC on both benchmarks. On LiquidSugar, it also reduces in-bump MAE, outside mean delay, and peak value error. On CausalAI, DAMF achieves nearperfect precision and a low FAR. Its in-bump MAE is close to that of WCC, whereas its outside mean delay and peak value error are substantially lower. These results indicate that DAMF improves both the detection of delayed intervals and the estimation of delay magnitudes under controlled-delay settings.

## 4.4.2. Representative lag recovery examples

Figure 7 provides qualitative examples of the decoded lag paths produced by DAMF on the two controlled-delay benchmarks. The blue curve denotes the ground-truth lag path. Let $\hat { d } _ { t } ^ { ( r ) }$ denote the decoded delay at sample index t in the r-th random-seed run, with $r = 1 , \ldots , R$ . The predictions from all runs are aligned by sample index. At each index t, the orange curve shows the mean $\begin{array} { r } { \bar { d } _ { t } = R ^ { - 1 } \sum _ { r = 1 } ^ { R } \hat { d } _ { t } ^ { ( r ) } } \end{array}$ , while the lower and upper boundaries of the shaded region are min $1 1 { \leq } r { \leq } R \hat { d } _ { t } ^ { ( r ) }$ and $\operatorname* { m a x } _ { 1 \leq r \leq R } \hat { d } _ { t } ^ { ( r ) }$ 2 respectively.

These summaries are computed from the decoded lag paths of the individual runs. The shaded band therefore shows the observed variation across random seeds, rather than a confidence interval; a wider band indicates greater disagreement among runs. For LiquidSugar, the recovered paths capture the main injected delay intervals and follow the overall ground-truth trajectory. Both block-shaped and bump-shaped delay structures are retained, with discrepancies occurring mainly near transition boundaries and rapidly changing lag values. For CausalAI, the decoded paths show closer agreement with the ground truth. The sustained block and the smooth bump are both recovered, including their onset, peak, and decay regions. This pattern is consistent with the cleaner source–target dependence in the CausalAI benchmark. Additional region-level lag-recovery examples are provided in Appendix A.

(a) LiquidSugar benchmark  
![](images/cca7aeafd1f549990c9d01b73df45b2cda4d1633fffcff01df4e51e3993f8d54.jpg)

![](images/dac9521f10e50182778e4701c5c072703f10f85723f784cd01c728b3eb0aabec.jpg)

![](images/8ebf9b7a3a35e6747fea7e03b65a7a21ee5baeac2c6eace61273a008e709276b.jpg)

![](images/767fb61b0ab554b65a2465bee43d4b3cdc7dac8a2981d4e0ee0bf9277930f428.jpg)

![](images/9402df58f045a1f795964ff3ae91c18dc45d27dd270b7e7e83fba52378b9edc9.jpg)

(b) CausalAI benchmark  
![](images/95e3fbcd10f6a816d0788b364fa8beb33338207d44f1da110f27ea0a9ac2538a.jpg)

![](images/5fc3282194ecfacd7ec055b7c1d3063dc0a501b9a3417d2b7d7f08870d4a82e1.jpg)

![](images/2672d40bb1f608bc890cd6cb396ea86903451d7a2b3a787540a5788e0c16cbbf.jpg)

![](images/c047e6caec5833d9d3c7cc8d472ca8aef4001af917e6bae1c6803558ee9f5972.jpg)

![](images/133198de21cf4b49d83f8de775c5ba2cd924afbe33107762e6d7d8351bbc52ec.jpg)  
Figure 6. Lag-recovery comparison on the two controlled-delay benchmarks. Each panel reports one lagrecovery metric for MNet, WCC, and DAMF. MNet uses the hidden-lag output of MultistageNet, WCC combines window-based cross-correlation with a conventional Viterbi decoder, and DAMF denotes the proposed lag-recovery pipeline and is highlighted in red.

## 4.5. Ablation studies

The following ablations examine the lag-predictor architecture, training objective, temporal decoding, and lag prior. All lag-recovery metrics follow the definitions in Section 4.2.1.

## 4.5.1. Structural ablation of the lag predictor

The structural ablation evaluates three key design choices in the lag predictor: sourcetarget lag matching, lag-specific source value extraction, and the selector auxiliary head. These designs are introduced to enable explicit delay comparison, preserve source information under each candidate lag, and provide auxiliary supervision for candidate ranking, respectively. Table 2 shows that weakening any of these designs leads to worse lag recovery performance. When source-target lag matching is removed, Occ F1 drops substantially and Lag MAE increases, indicating that directly comparing upstream and downstream temporal windows is important for localizing delays. Removing lagspecific source value extraction causes the largest degradation, especially in large-lag recall and collapse-to-zero errors. This suggests that retaining source information at each candidate lag is critical for distinguishing nonzero delays. Removing the selector auxiliary head also reduces performance, although the drop is relatively smaller, which indicates that it mainly serves as an auxiliary regularizer for candidate ranking rather than the primary source of lag recovery.

![](images/46deeaf5d2a150106991dab443ee6a4f495abe976610b854e7d97a10054cab7a.jpg)

(b) CausalAl benchmark  
![](images/2ce57d67cff4358e514601803e4b8493f9320677eac4a77639782dcb33743401.jpg)  
Figure 7. Representative lag-recovery examples on the controlled-delay benchmarks. The blue curves denote the ground-truth lag paths, while the orange curves and shaded regions represent the mean decoded paths and their ranges across random seeds, respectively. Each benchmark contains both block-shaped and bump-shaped delay patterns.

Table 2. Structural ablation of the lag predictor. The best result for each metric is shown in bold.
<table><tr><td>Setting</td><td>Occ F1 ↑</td><td>Lag MAE↓</td><td> $\pm 1 \operatorname { A c c } \uparrow$ </td><td></td><td>Large-lag Recall ↑ Collapse to 0 ↓</td></tr><tr><td>Full lag predictor</td><td> $\mathbf { 0 . 9 1 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 6 5 \ : \pm { \ : 0 . 1 3 } }$ </td><td> $\mathbf { 0 . 9 0 \ : \pm { \ : 0 . 0 2 } }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 0 3 \ : \pm { \ : 0 . 0 3 } }$ </td></tr><tr><td>w/o Match</td><td> $0 . 6 0 \pm 0 . 0 6$ </td><td> $2 . 3 4 \pm 0 . 1 9$ </td><td> $0 . 6 4 \pm 0 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 1 1$ </td><td> $0 . 4 5 \pm 0 . 0 8$ </td></tr><tr><td>w/o LagVal</td><td> $0 . 4 2 \pm 0 . 0 5$ </td><td> $2 . 4 5 \pm 0 . 0 8$ </td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td> $0 . 3 3 \pm 0 . 0 0$ </td><td> $0 . 6 4 \pm 0 . 0 2$ </td></tr><tr><td>w/o Selector</td><td> $0 . 8 6 \pm 0 . 1 0$ </td><td> $0 . 9 0 \pm 0 . 3 2$ </td><td> $0 . 8 7 \pm 0 . 0 4$ </td><td> $0 . 6 7 \pm 0 . 1 1$ </td><td> $0 . 1 4 \pm 0 . 1 6$ </td></tr></table>

w/o Match removes source-target lag matching; w/o LagVal removes the lag-specific source value; w/o Selector removes the selector auxiliary head.

## 4.5.2. Loss ablation of the lag predictor

We further investigate the contribution of each loss term in the lag predictor training objective. The ablation settings are constructed by removing one loss term at a time from the full objective, including the relaxed-TopN loss, the positive-lag CE loss, the large-occurrence recall loss, and the selector loss. In addition, we include a simplified variant that retains only the occurrence loss and the positive-lag CE loss, so as to examine whether standard classification supervision is suficient for reliable lag recovery. Table 3 shows that the full lag-predictor objective achieves the strongest overall performance. It obtains the best results in Occ F1, Lag MAE, ±1 Acc, and Collapse to 0, while maintaining competitive large-lag recall. Although the simplified occurrenceplus-CE setting still yields reasonable results, it noticeably reduces large-lag recall and increases the collapse-to-zero rate. This indicates that basic classification supervision alone is insuficient to protect dificult large-lag regions, especially when the model needs to distinguish true nonzero delays from no-lag cases.

Among the individual loss terms, the relaxed-TopN loss mainly improves candidate coverage and helps preserve plausible large-lag candidates. When this term is removed, the recovered lag path becomes less stable and the overall lag estimation accuracy deteriorates. The positive-lag CE loss is essential for estimating the lag magnitude; removing it leads to the largest increase in Lag MAE and a substantial drop in ±1 Acc. The large-occurrence recall loss further prevents large delays from being misclassified as no lag or small lag, as reflected by the degradation in large-lag recall and collapse-related metrics. The selector loss has a relatively moderate but consistent efect, suggesting that it serves primarily as an auxiliary regularizer for candidate ranking rather than as the dominant source of lag recovery.

<table><tr><td>Setting</td><td></td><td></td><td></td><td>Occ F1 ↑ Lag MAE ↓ ±1 Acc ↑ Large-lag Recall ↑ Collapse to 0 ↓</td><td></td></tr><tr><td>Full lag-prediction objective</td><td> $\mathbf { 0 . 9 1 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 6 5 \ : \pm { \ : 0 . 1 3 } }$ </td><td> ${ \bf 0 . 9 0 \pm 0 . 0 2 }$ </td><td> $0 . 7 6 \pm 0 . 0 5$ </td><td> $\mathbf { 0 . 0 3 \ : \pm { \ : 0 . 0 3 } }$ </td></tr><tr><td>Occurrence + positive CE only</td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $0 . 8 2 \pm 0 . 1 5$ </td><td> $0 . 8 7 \pm 0 . 0 2$ </td><td> $0 . 6 8 \pm 0 . 0 5$ </td><td> $0 . 2 0 \pm 0 . 0 9$ </td></tr><tr><td>w/o relaxed-TopN loss</td><td> $0 . 8 9 \pm 0 . 0 4$ </td><td> $0 . 9 4 \pm 0 . 1 5$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 1 3 }$ </td><td> $0 . 1 0 \pm 0 . 0 8$ </td></tr><tr><td>w/o positive-lag CE loss</td><td> $0 . 6 4 \pm 0 . 1 1$ </td><td> $2 . 7 7 \pm 0 . 0 8$ </td><td> $0 . 6 0 \pm 0 . 0 1$ </td><td> $0 . 1 4 \pm 0 . 0 2$ </td><td> $0 . 6 5 \pm 0 . 0 4$ </td></tr><tr><td>w/o large-occurrence recall loss</td><td> $0 . 8 3 \pm 0 . 0 7$ </td><td> $1 . 3 2 \pm 0 . 2 3$ </td><td> $0 . 8 2 \pm 0 . 0 3$ </td><td> $0 . 6 2 \pm 0 . 1 6$ </td><td> $0 . 2 3 \pm 0 . 1 7$ </td></tr><tr><td>w/o selector loss</td><td> $0 . 8 6 \pm 0 . 0 5$ </td><td> $0 . 9 1 \pm 0 . 3 9$ </td><td> $0 . 8 5 \pm 0 . 0 5$ </td><td> $0 . 6 5 \pm 0 . 1 5$ </td><td> $0 . 1 2 \pm 0 . 0 9$ </td></tr></table>

Table 3. Loss ablation for lag recovery. The first row uses the complete lag-prediction objective. Each row removes one loss component from the full objective, except “Occurrence + positive-lag CE only”. The best numerical result for each metric is shown in bold.

## 4.5.3. Efect of Viterbi decoding

After the lag predictor produces raw edge-level lag emissions, the final lag path can be obtained either by pointwise raw-argmax decisions or by structured temporal decoding. The raw-argmax strategy treats each sample independently and therefore does not explicitly account for the temporal persistence of manufacturing delays. In contrast, Viterbi decoding imposes structured continuity on the recovered lag sequence, which is consistent with the fact that process delays usually persist over continuous intervals rather than switching arbitrarily at each time point.

Table 4 compares raw-argmax decoding with Viterbi decoding. The decoder substantially improves the main accuracy-related metrics, including Acc, ±1 Acc, Largelag Recall, and Lag MAE. It also reduces Collapse to small, indicating that structured decoding helps suppress unstable small-lag predictions. Raw argmax achieves a slightly lower Collapse to 0 value; however, the absolute values of this metric are small in both settings. Therefore, the overall comparison demonstrates that Viterbi decoding improves lag recovery accuracy and produces more stable lag paths. Figure 8 further provides a path-level illustration of this efect. Raw-argmax predictions are able to capture the broad delayed regions, but they contain local spikes, short-lived drops, and noticeable seed-to-seed variability. After decoding, the recovered paths become less fragmented and better aligned with the duration of the underlying delay regions. This qualitative observation is consistent with the aggregate improvements reported in Table 4. Taken together, these experiments provide consistent evidence for the proposed delay-integrated design. DAMF improves final quality prediction, lag predictor recovers interpretable time-varying lag patterns on controlled-delay benchmarks, and the ablation studies verify the necessity of the lag prior, the training objective, and the duration-aware decoder. The performance gain therefore arises from the joint efect of lag estimation and delay-aware fusion, rather than from any single part alone.

Table 4. Efect of Viterbi decoding on lag recovery. The best numerical result for each metric is shown in bold.
<table><tr><td>Setting</td><td> $\operatorname { A c c } \uparrow$ </td><td> $\pm 1 \operatorname { A c c } \uparrow$ </td><td></td><td></td><td></td><td>Lag MAE ↓ Large-lag Recall ↑ Collapse to 0 ↓ Collapse to small ↓</td></tr><tr><td>Raw argmax</td><td> $0 . 6 5 \pm 0 . 0 4$ </td><td> $0 . 7 9 \pm 0 . 0 4$ </td><td> $1 . 0 3 \pm 0 . 1 2$ </td><td> $0 . 5 5 \pm 0 . 0 4$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 2 }$ </td><td> $0 . 4 4 \pm 0 . 0 3$ </td></tr><tr><td>Viterbi</td><td> ${ \bf 0 . 7 6 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 9 0 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 6 5 \ : \pm { \ : 0 . 1 3 } }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 0 5 }$ </td><td> $0 . 0 3 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 2 2 \pm 0 . 0 3 }$ </td></tr></table>

![](images/9732ea77fd050eb42673b0f6345f00607c222c239e0a9b135d7344d13c16a6d8.jpg)  
Figure 8. Path-level comparison between raw pointwise argmax predictions and Viterbi decoding on representative test regions. The left column shows raw argmax predictions before decoding, whereas the right column shows the lag paths after Viterbi decoding. The blue curve denotes the ground-truth lag path, the orange curve denotes the prediction mean across random seeds, and the shaded region denotes the corresponding prediction range.

## 4.5.4. Efect of the lag prior on final quality prediction

This ablation examines the contribution of the estimated lag prior to final quality prediction. We compare DAMF without the lag prior against variants that incorporate the pretrained lag predictor’s soft distribution into DelayAlignment. All variants use the same forecasting architecture and training protocol, with only the prior weight varying. We evaluated diferent prior weights on the train set and adopted $\lambda = 0 . 4$ in the main experiments, as it achieved the best overall performance across RMSE, MAE, and $R ^ { 2 }$

![](images/a43fa3b3ca65787b2a49973379253162c081a398fcde3d75c59ba3eaae330e60.jpg)  
Figure 9. Sensitivity analysis of the lag prior weight. The four panels report RMSE, ∆RMSE, MAE, and $R ^ { 2 }$ under diferent prior-weight settings.

Figure 9 presents the results under diferent prior weights. Compared with the noprior baseline, most lag-prior settings yield lower RMSE and MAE and higher $R ^ { 2 }$ The improvement is marginal at the smallest weight, while $\lambda = 0 . 4$ gives the best results across all three metrics. These results suggest that the estimated edge-level lag distributions provide useful information for cross-stage alignment and can improve final quality prediction. Additional region-level lag recovery examples are provided in Appendix A.

## 5. Conclusion

This paper presented DAMF, a delay-aware framework for multi-stage manufacturing quality prediction. The central idea is to treat inter-stage propagation delay as prediction-relevant information, rather than relying on fixed timestamp alignment or leaving lag estimation as a separate diagnostic analysis. By integrating delay estimation into the forecasting process, DAMF improves the connection between upstream process histories and downstream quality outcomes while also providing interpretable temporal propagation patterns across production stages.

The experimental results show that delay information plays a practical role in multistage quality prediction. DAMF outperforms the compared baselines on final quality prediction, and the lag predictor recovers more accurate and stable lag paths on controlled-delay benchmarks. These findings indicate that the estimated inter-stage delays can serve not only as interpretable diagnostic outputs, but also as useful temporal guidance for aligning upstream and downstream process representations.

Ablation studies further clarify the source of the performance gains. Removing the lag prior weakens final quality prediction, simplifying the lag-predictor architecture or its training objective degrades lag recovery, and replacing structured decoding with pointwise decisions leads to less stable lag paths. These results support two distinct uses of the estimated lag distributions: guiding cross-stage alignment for quality prediction and recovering temporally coherent lag paths for interpretation.

Several directions remain for future work. First, the current framework focuses on adjacent-stage delay modeling, whereas more complex manufacturing systems may involve non-adjacent interactions, branching process structures, or feedback loops. Extending DAMF to more general process graphs would broaden its applicability. Second, the current delay representation is based on a discrete candidate lag space. Continuous or distributional delay modeling may provide a more flexible way to describe propagation patterns.

## References

Agrawal, R., J. F. Lawless, and R. J. Mackay. 1999. “Analysis of Variation Transmission in Manufacturing Processes—Part II.” Journal of Quality Technology 31 (2): 143–154.

Ahamed, Md Atik, and Qiang Cheng. 2024. “TimeMachine: A Time Series is Worth 4 Mambas for Long-Term Forecasting.” In ECAI 2024 – 27th European Conference on Artificial Intelligence, Including 13th Conference on Prestigious Applications of Intelligent Systems, PAIS 2024, Proceedings, Vol. 392 of Frontiers in Artificial Intelligence and Applications, 1688–1695. IOS Press.

Breiman, Leo. 2001. “Random Forests.” Machine Learning 45 (1): 5–32.

Cai, Xiuding, Yaoyao Zhu, Xueyao Wang, and Yu Yao. 2024. “MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting.” https://arxiv.org/abs/2405.16440.

Carter, G. Cliford. 1987. “Coherence and Time Delay Estimation.” Proceedings of the IEEE 75 (2): 236–255.

Cattaldo, Marco, Alberto Ferrer, and Ingrid M˚age. 2024. “Variable time delay estimation in continuous industrial processes.” Chemometrics and Intelligent Laboratory Systems 246: 105082.

Ceglarek, Darek, Wenzhen Huang, Shiyu Zhou, Yu Ding, R. Kumar, and Y. Zhou. 2004. “Time-Based Competition in Multistage Manufacturing: Stream-of-Variation Analysis (SOVA) Methodology—Review.” International Journal of Flexible Manufacturing Systems 16 (1): 11–44.

Chen, Tianqi, and Carlos Guestrin. 2016. “XGBoost: A Scalable Tree Boosting System.” In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

Cho, Hojin, Kyeongbin Kim, Kihyuk Yoon, Jaewook Chun, Jaeyong Kim, Kyeongmin Lee, Junghye Lee, and Chiehyeon Lim. 2024. “MMP Net: A Feedforward Neural Network Model with Sequential Inputs for Representing Continuous Multistage Manufacturing Processes without Intermediate Outputs.” IISE Transactions 56 (10): 1058–1069.

Cho, Hojin, and Chiehyeon Lim. 2024. “Multistage Net: Learning Continuous Multistage Manufacturing Processes of Liquid Products Without Intermediate Output and Lead-Time Labels.” IEEE Transactions on Industrial Informatics 20 (12): 13765–13775. https://doi.org/10.1109/TII.2024.3432134.

Cho, Kyunghyun, Bart van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.” In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 1724–1734.

Du, Shichang, Rui Xu, Delin Huang, and Xufeng Yao. 2015. “Markov Modeling and Anal-

ysis of Multi-Stage Manufacturing Systems with Remote Quality Information Feedback.” Computers & Industrial Engineering 88: 13–25.

Faghih Niresi, Keivan, Hugo Bissig, Henri Baumann, and Olga Fink. 2024. “Physics-Enhanced Graph Neural Networks for Soft Sensing in Industrial Internet of Things.” IEEE Internet of Things Journal 11 (21): 34978–34990.

Filz, Marc-Andr´e, Jan Philipp Bosse, and Christoph Herrmann. 2024. “Digitalization Platform for Data-Driven Quality Management in Multi-Stage Manufacturing Systems.” Journal of Intelligent Manufacturing 35 (6): 2699–2718.

Granger, C. W. J. 1969. “Investigating Causal Relations by Econometric Models and Crossspectral Methods.” Econometrica 37 (3): 424–438.

Griesing-Scheiwe, Fritjof, Yuri A. W. Shardt, Gustavo P´erez-Zu˜niga, and Xu Yang. 2020. “Soft sensor design for variable time delay and variable sampling time.” Journal of Process Control 92: 310–318.

Hochreiter, Sepp, and J¨urgen Schmidhuber. 1997. “Long Short-Term Memory.” Neural Computation 9 (8): 1735–1780.

Jiang, Yuchen, Shen Yin, Jingwei Dong, and Okyay Kaynak. 2021. “A Review on Soft Sensors for Monitoring, Control, and Optimization of Industrial Processes.” IEEE Sensors Journal 21 (11): 12868–12881.

Kadlec, Petr, Bogdan Gabrys, and Sibylle Strandt. 2009. “Data-driven Soft Sensors in the Process Industry.” Computers & Chemical Engineering 33 (4): 795–814.

Ke, Guolin, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. “LightGBM: A Highly Eficient Gradient Boosting Decision Tree.” In Advances in Neural Information Processing Systems, Vol. 30.

Knapp, Charles H., and G. Cliford Carter. 1976. “The Generalized Correlation Method for Estimation of Time Delay.” IEEE Transactions on Acoustics, Speech, and Signal Processing 24 (4): 320–327.

Lawless, J. F., R. J. Mackay, and J. A. Robinson. 1999. “Analysis of Variation Transmission in Manufacturing Processes—Part I.” Journal of Quality Technology 31 (2): 131–142.

Liu, Yong, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. 2024. “iTransformer: Inverted Transformers Are Efective for Time Series Forecasting.” In International Conference on Learning Representations, https://openreview.net/forum?id=JePfAI8fah.

Meji´c, Luka, Darko Staniˇsi´c, Vojin Ili´c, Boris Jakovljevi´c, and Nikola Jorgovanovi´c. 2019. “A novel algorithm for dead time estimation between soft sensor inputs and output.” Computers & Chemical Engineering 124: 153–161.

Ni, Boyi, Deyun Xiao, and Sirish L. Shah. 2010. “Time delay estimation for MIMO dynamical systems–With time-frequency domain analysis.” Journal of Process Control 20 (1): 83–94.

Nie, Yuqi, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023. “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.” In International Conference on Learning Representations, https://openreview.net/forum?id=Jbdc0vTOcol.

Ofermans, Tim, Ewa Szyma´nska, Lutgarde M. C. Buydens, and Jeroen J. Jansen. 2020. “Synchronizing process variables in time for industrial process monitoring and control.” Computers & Chemical Engineering 140: 106938.

Ofermans, Tim, Bente van Son, Carlo G. Bertinetto, Arjen Bot, Rogier Brussee, and Jeroen J. Jansen. 2024. “Data-driven sensor delay estimation in industrial processes using multivariate projection methods.” Chemometrics and Intelligent Laboratory Systems 246: 105090.

Pamfil, Roxana, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilgerstorfer, Konstantinos Georgatzis, Paul Beaumont, and Bryon Aragam. 2020. “DYNOTEARS: Structure Learning from Time-Series Data.” In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, Vol. 108 of Proceedings of Machine Learning Research, 1595–1605. PMLR. https://proceedings.mlr.press/v108/pamfil20a.html.

Qin, S. Joe. 2012. “Survey on Data-Driven Industrial Process Monitoring and Diagnosis.” Annual Reviews in Control 36 (2): 220–234.

Rabiner, Lawrence R. 1989. “A Tutorial on Hidden Markov Models and Selected Applications

in Speech Recognition.” Proceedings of the IEEE 77 (2): 257–286.

Reitz, Elena, Helmut Podhaisky, David Ely, and Markus Thommes. 2013. “Residence time modeling of hot melt extrusion processes.” European Journal of Pharmaceutics and Biopharmaceutics 85 (3): 1200–1205.

Runge, Jakob, Peer Nowack, Marlene Kretschmer, Seth Flaxman, and Dino Sejdinovic. 2019. “Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets.” Science Advances 5 (11): eaau4996.

Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” In Advances in Neural Information Processing Systems, Vol. 30.

Viterbi, Andrew J. 1967. “Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm.” IEEE Transactions on Information Theory 13 (2): 260–269.

Wang, Pei, Hai Qu, Qianle Zhang, Xun Xu, and Sheng Yang. 2023. “Production Quality Prediction of Multistage Manufacturing Systems Using Multi-Task Joint Deep Learning.” Journal of Manufacturing Systems 70: 48–68.

Wang, Shiyu, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, and Jun Zhou. 2024a. “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.” In International Conference on Learning Representations, https://openreview.net/forum?id=7oLshfEIC2.

Wang, Wei, Chunhua Yang, Jie Han, Wenting Li, and Yonggang Li. 2021. “A soft sensor modeling method with dynamic time-delay estimation and its application in wastewater treatment plant.” Biochemical Engineering Journal 172: 108048.

Wang, Yuang, Jiming Li, and Xuezhen Cheng. 2024. “A D-norm cross-correlation method for particle velocity measurement through electrostatic sensing.” Measurement 224: 113950.

Wang, Yuxuan, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, and Mingsheng Long. 2024b. “TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.” In Advances in Neural Information Processing Systems, Vol. 37, 469–498.

Xiong, Weili, Yanjun Li, Yujia Zhao, and Biao Huang. 2017. “Adaptive soft sensor based on time diference Gaussian process regression with local time-delay reconstruction.” Chemical Engineering Research and Design 117: 670–680.

Yan, Hao, Nurettin Dorukhan Sergin, William A. Brenneman, Stephen Joseph Lange, and Shan Ba. 2021. “Deep Multistage Multi-Task Learning for Quality Prediction of Multistage Manufacturing Systems.” Journal of Quality Technology 53 (5): 526–544. https://doi.org/10.1080/00224065.2021.1903822.

Yang, Bo, and Hongguang Li. 2017. “A similarity elastic window based approach to process dynamic time delay analysis.” Chemometrics and Intelligent Laboratory Systems 170: 13–24.

Yao, Le, and Zhiqiang Ge. 2020. “Refining data-driven soft sensor modeling framework with variable time reconstruction.” Journal of Process Control 87: 91–107.

Yu, Shun-Zheng. 2010. “Hidden Semi-Markov Models.” Artificial Intelligence 174 (2): 215–243.

Zhang, Donghao, Zhenyu Liu, Weiqiang Jia, Hui Liu, and Jianrong Tan. 2022. “Path Enhanced Bidirectional Graph Attention Network for Quality Prediction in Multistage Manufacturing Process.” IEEE Transactions on Industrial Informatics 18 (2): 1018–1027.

Zhao, Yantao, Bochuan Ding, Yuling Zhang, Liming Yang, and Xiaochen Hao. 2021. “Online cement clinker quality monitoring: A soft sensor model based on multivariate time series analysis and CNN.” ISA Transactions 117: 180–195.

Zheng, Wei Xing, and Chun Bo Feng. 1990. “Identification of stochastic time lag systems in the presence of colored noise.” Automatica 26 (4): 769–779.

![](images/aa6e70e0427d095fc62e6047d68d0afc4bb080c555941d0c2925e5a5bbf4afcb.jpg)

![](images/b5fdc9ac9ed78cc2610ef10a45482b867a59fba204962a7d7af1bf07dcc5aec0.jpg)

![](images/67232be41e66b27204e0ebd4986d20cfc015646101b66e2c96ab8ed65803ae5e.jpg)

![](images/6c9f79f14a329c801ce99e95d38ba9dd01b9b9f8b7934e8601b7663b7a256197.jpg)

![](images/01acb1570f6a28b1c2125a8d55696cd933f416ce1c908c5842433c10d87bb0e7.jpg)  
(a) LiquidSugar dataset with artificially injected lag.

![](images/393199546beb01eb236af3dc4eaaefec1a3b0ef380a7503736072ea739bda75a.jpg)

![](images/f979bcbdf943e8376613822d6a244da6ec715c8f963d450239b33748399f8d80.jpg)

![](images/634310a74bc479735029fe7a0136f8425b3f52482f5ffa19cf44e64d135eaa52.jpg)

![](images/f6c125d6474134cd31be938fb5fe5e80fa8493c5ecb12cac71caba25779e4644.jpg)

![](images/7b33818d2d6889060de816da673bcf67428d984194667fae8bccf5ba434e67f4.jpg)

![](images/1a00e771aad0363142c64591b4dd94d7634f8e0bb77e58096e4f44346915036e.jpg)  
(b) CausalAI dataset.

![](images/b5a7d6b025832d344c5b541708fa89951b25b31769b4cab22b3285fe46a17284.jpg)  
Figure A1. Lag recovery results of the edge-level lag predictor on two controlled-delay datasets. The blue line denotes the ground-truth lag path, the orange line denotes the mean Viterbi-decoded lag path across seeds, and the shaded region denotes the corresponding range across seeds 42, 43, and 44.

(a) Raw point-wise lag prediction  
![](images/3d4a6f5f4c4ff48b8a604bfb36b788445814095d8139e6b023cbf4419f0afa15.jpg)  
Figure B1. Efect of Viterbi decoding on lag-path recovery across five test regions. The left and right columns show the raw point-wise lag predictions and the corresponding Viterbi-decoded lag paths, respectively. Solid colored curves and shaded bands denote the mean and range across three random seeds, while blue curves represent the ground-truth lag paths. Vertical dashed lines delimit the evaluation intervals, and the reported MAE is computed within each interval.

## Appendix A. Additional lag recovery examples

## Appendix B. Additional lag path recovery results

## Appendix C. Detailed lag-predictor objective

For clarity, we present the objective for a fixed adjacent-stage edge and omit the edge superscript g. Let B denote a nonempty mini-batch, let $\ell _ { t } \in \mathcal { K } = \{ 0 , 1 , \ldots , K \}$ denote the ground-truth lag of sample $t ,$ and define

$$
r _ { t } = \mathbb { I } ( \ell _ { t } > 0 ) .
$$

The occurrence head outputs $o _ { t } = \operatorname* { P r } ( z _ { t } > 0 )$ , and the positive-lag head outputs

$$
\pi _ { \mathrm { p o s } , t } ( k ) = \operatorname* { P r } ( z _ { t } = k \mid z _ { t } > 0 ) , \qquad k \in { \mathcal { K } } _ { + } , \qquad { \mathcal { K } } _ { + } = \{ 1 , \ldots , K \} .
$$

For any index set ${ \mathcal { A } } ,$ define

$$
D ( { \mathcal { A } } ) = \operatorname* { m a x } \{ 1 , | { \mathcal { A } } | \} .
$$

Thus, a loss whose contributing set is empty evaluates to zero. The occurrence head is trained using a focal-style binary classification loss:

$$
\mathcal { L } _ { \mathrm { o c c } } = - \frac { 1 } { | \mathcal { B } | } \sum _ { t \in \mathcal { B } } \left[ \alpha r _ { t } ( 1 - o _ { t } ) ^ { \gamma } \log ( o _ { t } + \epsilon ) + ( 1 - \alpha ) ( 1 - r _ { t } ) o _ { t } ^ { \gamma } \log ( 1 - o _ { t } + \epsilon ) \right] ,\tag{C1}
$$

where $\alpha \in [ 0 , 1 ]$ controls class weighting, $\gamma \geq 0$ is the focal parameter, and $\epsilon > 0$ ensures numerical stability. Let

$$
\mathcal { P } = \{ t \in \mathcal { B } : \ell _ { t } > 0 \}
$$

denote the set of positive-lag samples. The positive-lag head is trained using the conditional cross-entropy loss

$$
\mathcal { L } _ { \mathrm { p o s } } = - \frac { 1 } { D ( \mathcal { P } ) } \sum _ { t \in \mathcal { P } } \log ( \pi _ { \mathrm { p o s } , t } ( \ell _ { t } ) + \epsilon ) .\tag{C2}
$$

For each $t \in \mathcal { P }$ , define the set of incorrect positive-lag candidates as

$$
{ \mathcal { N } } _ { t } = { \mathcal { K } } _ { + } \backslash \{ \ell _ { t } \} .
$$

Let

$$
s _ { t , [ 1 ] } ^ { ( - ) } \geq s _ { t , [ 2 ] } ^ { ( - ) } \geq \cdot \cdot \cdot \geq s _ { t , [ K - 1 ] } ^ { ( - ) }
$$

be the matching scores $\{ s _ { t } ( k ) : k \in \mathcal { N } _ { t } \}$ sorted in descending order, and let

$$
s _ { t , ( N ) } ^ { ( - ) } = s _ { t , [ N ] } ^ { ( - ) } , \qquad 1 \leq N \leq K - 1 .
$$

The relaxed Top-N loss is

$$
\mathcal { L } _ { \mathrm { T o p N } } = \frac { 1 } { D ( \mathcal { P } ) } \sum _ { t \in \mathcal { P } } \Big [ m + s _ { t , ( N ) } ^ { ( - ) } - s _ { t } ( \ell _ { t } ) \Big ] _ { + } , \qquad [ x ] _ { + } = \operatorname* { m a x } \{ 0 , x \} ,\tag{C3}
$$

where $m > 0$ is a margin. This term encourages the ground-truth lag to remain among the leading candidates without requiring it to have the largest matching score. To discourage large positive lags from being classified as zero, define

$$
\mathcal { P } _ { \mathrm { l a r g e } } = \{ t \in \mathcal { B } : \ell _ { t } \geq K _ { \mathrm { l a r g e } } \} , \qquad K _ { \mathrm { l a r g e } } \in \{ 1 , \ldots , K \} .
$$

The corresponding occurrence loss is

$$
\mathcal { L } _ { \mathrm { l a r g e } } = - \frac { 1 } { D ( \mathcal { P } _ { \mathrm { l a r g e } } ) } \sum _ { t \in \mathcal { P } _ { \mathrm { l a r g e } } } \log ( o _ { t } + \epsilon ) .\tag{C4}
$$

Let

$$
\mathcal { C } _ { t } = ( k _ { t , 1 } , \ldots , k _ { t , N } )
$$

denote the N highest-scoring candidates in $\kappa _ { + } .$ , ordered such that

$$
s _ { t } ( k _ { t , 1 } ) \geq s _ { t } ( k _ { t , 2 } ) \geq \cdot \cdot \cdot \geq s _ { t } ( k _ { t , N } ) .
$$

Let $a _ { t } ( j )$ be the selector probability assigned to candidate position $j ,$ satisfying

$$
a _ { t } ( j ) \geq 0 , \qquad \sum _ { j = 1 } ^ { N } a _ { t } ( j ) = 1 .
$$

For samples whose ground-truth lag is retained, define

$$
S = \{ t \in { \mathcal { P } } : \ell _ { t } \in \mathcal { C } _ { t } \} , \qquad j _ { t } ^ { * } = \operatorname { i n d e x } _ { \mathcal { C } _ { t } } ( \ell _ { t } ) .
$$

The selector loss is

$$
\mathcal { L } _ { \mathrm { s e l } } = - \frac { 1 } { D ( S ) } \sum _ { t \in S } \log ( a _ { t } ( j _ { t } ^ { * } ) + \epsilon ) .\tag{C5}
$$

The selector provides auxiliary ranking supervision only and is not used to construct the lag distribution at inference. The complete lag-predictor objective is

$$
\mathcal { L } _ { \mathrm { l a g } } = \mathcal { L } _ { \mathrm { o c c } } + \mathcal { L } _ { \mathrm { p o s } } + \mathcal { L } _ { \mathrm { T o p N } } + \lambda _ { \mathrm { l a r g e } } \mathcal { L } _ { \mathrm { l a r g e } } + \lambda _ { \mathrm { s e l } } \mathcal { L } _ { \mathrm { s e l } } ,\tag{C6}
$$

where $\lambda _ { \mathrm { l a r g e } } , \lambda _ { \mathrm { s e l } } \ge 0$ control the auxiliary terms. The occurrence, positive-lag, and relaxed Top-N terms have unit coeficients in this formulation.