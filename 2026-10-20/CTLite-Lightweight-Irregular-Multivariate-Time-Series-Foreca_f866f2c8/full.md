# CTLite: Lightweight Irregular Multivariate Time Series Forecasting with Reusable Continuous-Time Summaries

Shusen Ma

Institute of Advanced Technology

University of Science and Technology of China

Hefei, China

Yangchun Xie

Institute of Advanced Technology

University of Science and Technology of China

Hefei, China

Xudong Zuo<sup>\*</sup>

Institute of Advanced Technology

University of Science and Technology of China

Hefei, China

Yun-Bo Zhao<sup>†</sup>

Department of Automation

University of Science and Technology of China

Hefei, China

ybzhao@ustc.edu.cn

Abstract—Irregular multivariate time series forecasting (IMTSF) requires capturing temporal dynamics and crossvariable dependencies from sparse, asynchronous observations. Existing methods often rely on learned temporal encoders, coupling historical representation construction with model optimisation and repeatedly processing the same observations. We propose CTLite, a lightweight forecasting framework that decouples historical summarisation from learned prediction. Specifically, a parameter-free continuous-time moment encoder summarises local levels and trends at multiple temporal scales while suppressing poorly supported slope estimates. A compact readout combines damped local extrapolation with a lowrank cross-variable correction that accounts for observation freshness, target availability, and forecast horizon. Since the historical summaries are independent of trainable parameters and forecast queries, they can be reused across optimisation steps and repeated predictions for an unchanged history without altering predictions or parameter gradients. Experiments on four real-world benchmarks show that CTLite achieves the best or second-best mean MSE and MAE among the evaluated methods with only 88–2,537 trainable parameters. Under matched GPU workloads at batch size 32, CTLite reduces inference latency by 18.6–87.3% compared with the latest models, including the cost of historical encoding. Reusing summaries further reduces total runtime by approximately 52% over 16 forecast calls on the same history compared with recomputing the encoder for every call, including the initial encoding cost. The code is at: https://github.com/SSMa913/NICLab-CTLite.

Index Terms—Irregular multivariate time series forecasting, continuous-time modelling, lightweight models, representation reuse.

## I. INTRODUCTION

Irregular multivariate time series arise when variables are measured asynchronously, and observations are missing. In clinical monitoring, environmental sensing, and activity recognition, a forecast must account for both the recorded values and the time over which they were observed. Two histories with the same values and ordering can imply different dynamics when their timestamps differ; two equally long histories can provide different amounts of evidence when their observation masks differ. Effective irregular multivariate time series forecasting (IMTSF) therefore requires a representation of temporal dynamics, observation support, and dependencies across variables.

![](images/56a229182c1c107624e988a34973f4525677e7578609da5d4a0beb442f24cc14.jpg)

![](images/35681f8482c4e81a29c32b2c02c7b180a3816d34c167e7a2cbdcca85060496d0.jpg)  
Fig. 1. Compact prediction and historical reuse. (a) Test MSE normalised by last-value forecasting on the same test records. (b) Time for 16 forecast calls on one unchanged batch, normalised by recomputing the history for every call. Reuse includes one initial encoding. Bars show five-seed means and sample Standard Deviation (SD).

Recent methods address these requirements through continuous latent dynamics, graph interactions, and temporal patches [1]–[5]. Efficient architectures have also become a direct focus: APN uses time-aware aggregation with a shallow predictor, while KAFNet revisits pre-alignment for efficient irregular forecasting [6], [7]. These advances raise a more specific question about where learned computation is needed. Can historical information be represented by fixed, compact statistics while a small learned readout supplies the forecasting capacity?

This question matters when histories are processed repeatedly. Time-series databases such as Gorilla and Apache IoTDB provide efficient storage and access to historical measurements [8], [9]; a forecasting operator must then transform the retrieved observations into predictions. Training revisits the same windows across epochs, and inference can request several forecasts from an unchanged window. A learned encoder’s cached representation becomes outdated when its parameters change. A parameter-independent representation can instead remain valid across readout updates, allowing historical aggregation to be performed once for a fixed window and preprocessing configuration. The design challenge is to obtain this reuse without reducing the history to an inadequate statistic.

Compact regular-grid forecasters suggest that a carefully chosen representation can reduce the amount of learned computation [10]–[13]. Irregular observations require additional structure. Array position does not measure elapsed time, a last value contains no trend information, and a local slope can be unreliable when only one observation effectively supports it. Moreover, sparsely observed variables may benefit from related variables, but a dense interaction matrix introduces a quadratic parameter cost. A useful compact representation must preserve local dynamics together with the evidence supporting them, and expose this information to an economical cross-variable predictor.

We propose CTLite, which separates fixed historical summarisation from learned prediction. Its continuous-time moment encoder computes weighted levels and slopes at several fixed decay rates directly from observed timestamps and masks. An effective-support factor suppresses poorly supported slopes. For each forecast query, a compact readout combines these summaries with a damped extrapolation step, whose growth saturates at a learned scale. A low-rank residual transfers recent values across variables, accounting for source freshness, target observation state, and forecast distance. Both residual branches are initialised at zero, so learning starts from the last-value predictor. The resulting model contains 88–2,537 trainable parameters across the four evaluated datasets.

Fig. 1 illustrates two properties that motivate our design. First, a fixed historical representation with only 88–2,537 learned parameters reduces MSE by 27.82–54.71% relative to last-value forecasting across four datasets. The gains show that a small predictor can extract useful information beyond the latest observation. Second, reusing the representation reduces the cumulative time for 16 forecast calls by 52.05–52.32%, even when initial encoding is included. These observations motivate separating historical aggregation from learned prediction: the representation should retain useful temporal information while remaining reusable when the prediction weights change. The reuse experiment evaluates CTLite’s execution paths, not a unique advantage over every possible cached baseline.

Our contributions are summarised as follows:

We propose CTLite, a compact forecasting model that separates historical encoding from learned prediction. Its parameter-free encoder summarises irregular observations through masked moments at multiple temporal scales, while a support-aware local readout limits extrapolation from poorly observed trends. For an unchanged history and preprocessing configuration, these summaries can be reused across training updates and forecast queries without changing the model’s predictions or parameter gradients.

We introduce a low-rank correction that complements local temporal predictions with information from other variables. The correction accounts for the time since each source variable’s last observation, the target variable’s observation availability, and the forecast horizon. This design incorporates observation timing and missingness into cross-variable information transfer, with an interaction parameter count that grows linearly with the number of variables for a fixed rank.

We evaluate CTLite on four irregular forecasting benchmarks using five random seeds, with CTLite containing only 88–2,537 trainable parameters and achieving competitive forecasting accuracy. Matched measurements show lower full-forward latency and peak training memory.

## II. RELATED WORK

## A. Structured Forecasting on Regular Grids

DLinear studies decomposition with linear prediction [10]. FITS uses frequency interpolation, SparseTSF exploits crossperiod sparsity, and MixLinear combines temporal and lowrank spectral modelling to obtain small predictors [11]–[13]. CycleNet explicitly models recurring patterns and predicts the residual [14]. Other architectures impose structure through multiscale mixing in TimeMixer [15] and modern convolutional blocks in ModernTCN [16]. Cross-variable modelling also takes several forms: PCDformer [17] and iTransformer [18] attend over variable tokens, SOFTS aggregates and redistributes a shared series core [19], and DUET uses temporal and channel clustering [20]. C3RL jointly leverages channelindependence and channel-mixing through a Siamese contrastive learning framework with adaptive loss weighting to improve multivariate time-series forecasting across different model architectures [21]. THCVformer jointly models variable heterogeneity, temporal dependencies, and inter-variable correlations to enhance prediction accuracy [22]. These methods illustrate how temporal and inter-variable structure can guide model design on regularly indexed sequences. CTLite constructs its historical features from the actual timestamps and masks of irregular observations.

## B. Representations for Irregular Observations

GRU-D incorporates missingness and elapsed time into recurrent updates [23]. Latent ODEs infer a latent initial state and model continuous dynamics [1], whereas GRU-ODE-Bayes combines continuous evolution with updates at observations [2]. Set-function models aggregate timestamped observations [24], and mTAN learns attention-based temporal representations [25]. Graph Neural Flows further couple continuous flows with graph-structured dependencies [26].

Recent IMTSF methods organise observations and interactions at multiple resolutions. GraFITi represents irregular series using a sparse bipartite graph [3]. t-PatchGNN learns transformable temporal patches [4], and Hi-Patch builds hierarchical patch representations [5]. ASTGI learns adaptive spatio-temporal graph interactions [27], while DyH2 models heterogeneous interactions across observation and variable levels [28]. TimeCHEAT combines local channel-dependent and global channel-independent modelling [29]. ReIMTS constructs recursive multiscale representations while preserving original timestamps [30].

![](images/e9e230c26261922d83fcebef21680ddfd1761b15ec3f4a28bc56e41c790caa23.jpg)  
Fig. 2. Overall CTLite framework. F and L denote fixed and learned computations, respectively.

Efficiency and alternative temporal representations are also active directions. APN combines time-aware aggregation with a shallow predictor [6]. KAFNet uses kernel-based prealignment and efficient frequency-domain modelling [7]. Two recent preprints explore observation-time-dependent linear maps in AiT [31] and learned, density-corrected basis representations in DNBNet [32]. CTLite uses fixed exponential weights and masked moments of the observed samples. Keeping this historical computation independent of the learned readout is what permits its reuse across optimisation updates.

## C. Probabilistic Forecasting and Evaluation

ProFITi models joint predictive distributions through conditional normalising flows [33], and MOSES develops marginalisation-consistent probabilistic forecasts [34]. CTLite produces point forecasts; its independent evaluation of query times concerns deterministic predictions rather than consistency of predictive distributions.

Recent benchmarks also broaden evaluation beyond individual model comparisons. TFB studies comprehensive and fair forecasting evaluation under unified protocols [35]. Physiome-ODE provides irregular forecasting tasks derived from biological dynamical systems [36], and Time-IMM introduces irregular multimodal multivariate data [37]. PYRREGULAR provides a unified framework with classification benchmarks [38]. These resources cover complementary tasks and data regimes. The present experiments use four established forecasting datasets and report comparisons under the stated preprocessing and execution settings.

## D. Historical Representation and Reuse

Systems such as Gorilla and Apache IoTDB optimise historical data storage and access [8], [9]. CTLite operates on retrieved forecasting windows. With fixed model weights, learned historical representations can also be cached when their encoders are query-independent. CTLite’s parameterfree encoder additionally allows the same stored summary to remain valid while the readout weights change. Reuse requires an unchanged history and fixed encoder configuration; the proposed implementation reconstructs the summary when either changes.

## III. METHOD

Fig. 2 illustrates CTLite and its three computational modules. Given an irregular history, the Continuous-Time Moment Encoder (Fig. 2(b)) constructs a reusable summary for each variable. For a query time, the Bounded Local Correction (Fig. 2(c)) uses the variable’s own summary to produce a local residual, while the Low-Rank Cross-Variable Correction (Fig. 2(d)) combines information across variables to produce a cross-variable residual. The two residuals are added to the Last-value baseline to obtain the Target forecast, as shown in Fig. 2(a).

## A. Task Formulation and Overall Framework

For one record, let $\boldsymbol { X } \in \mathbb { R } ^ { L \times N }$ contain historical values at timestamps $t _ { 1 } , \ldots , t _ { L }$ . Here N is the number of variables and

$L$ is the padded history length. The mask $M \in \{ 0 , 1 \} ^ { L \times N }$ has $M _ { i n } = 1$ only when variable n is observed at timestamp $t _ { i } .$ Missing values and padding have mask zero. Given future query times $q _ { 1 } , \ldots , q _ { Q }$ , the predictor returns an estimate for each requested variable and time. The prediction interface takes only the historical values, timestamps, masks, and requested query times.

Times follow the nonnegative normalised convention of the data pipeline. The shared origin and centred historical times are

$$
t _ { 0 } = \operatorname* { m a x } _ { i : \sum _ { n } M _ { i n } > 0 } t _ { i } , \qquad u _ { i } = t _ { i } - t _ { 0 } .\tag{1}
$$

We set $t _ { 0 } = 0$ if the entire history is empty. Let $\ell _ { n }$ denote the latest observed value of variable $n ,$ or zero if it is unobserved. CTLite predicts

$$
\widehat { y } _ { n } ( q ) = \ell _ { n } + \delta _ { n } ^ { \mathrm { l o c a l } } ( q ) + \delta _ { n } ^ { \mathrm { c r o s s } } ( q ) .\tag{2}
$$

The local residual $( \delta _ { n } ^ { \mathrm { l o c a l } } )$ uses the variable’s own history. The cross-variable residual $( \delta _ { n } ^ { \mathrm { c r o s s } } )$ uses recent information from all variables. The origin is determined solely by the historical observations; query independence is established in the reuse analysis.

## B. Continuous-Time Moment Encoder

1) Weights: Choose K fixed nonnegative decay rates $\Lambda =$ $\{ \lambda _ { 1 } , \ldots , \lambda _ { K } \}$ , including zero. Their dataset-specific values are reported in Table IV. At scale $k ,$ observation i receives weight:

$$
w _ { k i } = \exp ( \exp ( \lambda _ { k } u _ { i } , - 6 0 , 0 ) ) .\tag{3}
$$

The function cl $\mathrm { i p } ( z , a , b ) ~ = ~ \mathrm { m i n } \{ b , \mathrm { m a x } \{ a , z \} \}$ limits its argument to $[ a , b ] . \ \mathrm { A }$ zero rate gives equal weights, while a larger rate emphasises recent observations. Rates are measured relative to normalised time. We clip the exponent below at −60 for numerical stability.

2) Masked moments: For a scalar sequence $f _ { i } ,$ define

$$
\begin{array} { c } { { D _ { k n } = \displaystyle \operatorname* { m a x } \Biggl \{ \sum _ { i } w _ { k i } M _ { i n } , \epsilon \Biggr \} ~ , } } \\ { { \displaystyle \mathcal E _ { k n } [ f ] = D _ { k n } ^ { - 1 } \sum _ { i } w _ { k i } M _ { i n } f _ { i } , } } \\ { { \displaystyle \epsilon = 1 0 ^ { - 6 } . } } \end{array}\tag{4}
$$

The index n determines which observations contribute. The denominator floor ϵ prevents division by zero. Masked values are set to zero before moment accumulation. When the floor is inactive, $\mathcal { E } _ { k n }$ is a normalised weighted average.

The level is $\mu _ { k n } = \mathcal { E } _ { k n } [ x _ { n } ]$ , where $x _ { n } = ( x _ { 1 n } , \ldots , x _ { L n } )$ With mean time $\bar { u } _ { k n } = \mathcal { E } _ { k n } [ u ]$ , define temporal variance and time–value covariance as

$$
\begin{array} { r l } & { v _ { k n } = \operatorname* { m a x } \{ \mathcal { E } _ { k n } [ u ^ { 2 } ] - \bar { u } _ { k n } ^ { 2 } , 0 \} , } \\ & { c _ { k n } = \mathcal { E } _ { k n } [ u x _ { n } ] - \bar { u } _ { k n } \mu _ { k n } . } \end{array}\tag{5}
$$

The nonnegative clamp removes small negative variances due to numerical cancellation. Actual time differences enter both the weights and the moments. The summary therefore changes when the observation times change, even if their order and values do not.

3) Supported slopes: A local slope estimated from very few effective observations can be unreliable. We use a stabilized effective count $\nu _ { k n }$ and support factor $\chi _ { k n } \colon$

$$
\begin{array} { r l } & { \nu _ { k n } = \frac { D _ { k n } ^ { 2 } } { \operatorname* { m a x } \{ \sum _ { i } w _ { k i } ^ { 2 } M _ { i n } , \epsilon \} } , } \\ & { \chi _ { k n } = \frac { \operatorname* { m a x } \{ \nu _ { k n } - 1 , 0 \} } { \nu _ { k n } + 1 } , } \\ & { b _ { k n } = \chi _ { k n } \frac { c _ { k n } } { v _ { k n } + \rho } , \quad \quad \rho = 1 0 ^ { - 3 } . } \end{array}\tag{6}
$$

The ridge coefficient $\rho$ limits sensitivity to small temporal variance. With inactive floors and normalized weights $p _ { i } ~ =$ $w _ { k i } M _ { i n } / D _ { k n } .$ , the effective count is $1 / \textstyle \sum _ { i } p _ { i } ^ { 2 }$ . Equal weights on m samples give $\nu _ { k n } ~ = ~ m$ , whereas a dominant single sample gives a value near one. Consequently, $\chi _ { k n }$ suppresses slopes with little support. It shrinks slopes according to their effective observation support. Empty and singleton histories have zero slope, including cases in which a numerical floor is active.

4) Observation state: For each variable, let $o _ { n } = \sum _ { i } M _ { i n }$ be its observation count and $a _ { n }$ its age at $t _ { 0 }$ . If variable n has a latest observation at time $t _ { n } ^ { * }$ , then $a _ { n } = \mathrm { c l i p } ( t _ { 0 } - t _ { n } ^ { * } , 0 , 1 )$ Otherwise, set $a _ { n } = 1$ and $\ell _ { n } = 0$ . The availability indicator is $z _ { n } ^ { \mathrm { o b s } } = \mathbf { 1 } [ o _ { n } > 0 ]$ , where 1 is the indicator function. Define the descriptor:

$$
\begin{array} { r } { \pmb { r } _ { n } = [ a _ { n } , ( o _ { n } + 1 ) ^ { - 1 } , z _ { n } ^ { \mathrm { o b s } } ] ^ { \mathsf { T } } . } \end{array}\tag{7}
$$

Its three entries describe staleness, sparsity, and availability. 5) Cached summary: The encoder outputs $\begin{array} { r l } { T _ { n } } & { { } = } \end{array}$ $\{ \mu _ { k n } , b _ { k n } \} _ { k = 1 } ^ { K } \cup \{ \ell _ { n } , a _ { n } , o _ { n } , z _ { n } ^ { \mathrm { { o b s } } } \}$ for each variable and stores $t _ { 0 }$ once. No part of this summary depends on a learned parameter or on the forecast queries.

## C. Bounded Local Correction

1) Damped horizon: The Damped horizon block converts the elapsed time into a bounded extrapolation step $h _ { n } ( q )$ which controls how strongly the historical slopes contribute to the prediction.

Specifically, for a query time $q ,$ define the nonnegative forecast distance $\Delta ( q ) ~ = ~ \operatorname* { m a x } \{ q - t _ { 0 } , 0 \}$ and represented elapsed time $e _ { n } ( q ) = \Delta ( q ) + a _ { n } . \mathrm { A }$ learned positive coefficient $d _ { n } = \mathrm { s o f t p l u s } ( \theta _ { n } )$ controls the extrapolation step

$$
h _ { n } ( q ) = \frac { 1 - \exp [ - d _ { n } e _ { n } ( q ) ] } { d _ { n } } .\tag{8}
$$

Here $\theta _ { n } \in$ R is trainable and softplus $( x ) \ = \ \log ( 1 + e ^ { x } )$ For small $d _ { n } e _ { n } , \ h _ { n }$ approximates elapsed time. For a long horizon, it approaches $1 / d _ { n }$ . Thus, a poorly estimated slope is not multiplied by an arbitrarily large forecast distance.

2) Feature vector: The local feature vector has $F = 2 K + 2$ entries:

$$
\phi _ { n } ( q ) = \left[ \begin{array} { c } { { \{ \mu _ { k n } - \ell _ { n } \} _ { k = 1 } ^ { K } } } \\ { { \{ b _ { k n } h _ { n } ( q ) \} _ { k = 1 } ^ { K } } } \\ { { \ell _ { n } } } \\ { { \ell _ { n } \left( \exp [ - d _ { n } e _ { n } ( q ) ] - 1 \right) } } \end{array} \right] .\tag{9}
$$

The first group describes historical levels relative to the latest value. The second describes supported trend increments. The final two entries retain the latest value and its damped change. They allow a small readout to combine level correction, trend extrapolation, and attenuation.

3) Channel-wise linear head + shared MLP (width H) : The residual combines a variable-specific linear map with a shared nonlinear map:

$$
\delta _ { n } ^ { \mathrm { l o c a l } } ( q ) = \pmb { \alpha } _ { n } ^ { \top } \phi _ { n } ( q ) + \beta _ { n } + g _ { \psi } ( [ \phi _ { n } ( q ) , \pmb { r } _ { n } ] ) .\tag{10}
$$

The coefficients are ${ \pmb { \alpha } } _ { n } \in \mathbb { R } ^ { F }$ and $\beta _ { n } \in \mathbb { R } .$ . Square brackets concatenate vectors. The shared function is $g _ { \psi } ( \xi ) \ =$ $w _ { 2 } ^ { \mathsf { T } } \operatorname { t a n h } ( W _ { 1 } \xi + b _ { 1 } ) + b _ { 2 }$ , where $W _ { 1 } \in \mathbb { R } ^ { H \times ( F + 3 ) } , \ b _ { 1 } , w _ { 2 } \ \in$ $\mathbb { R } ^ { \mathbf { \breve { H } } }$ , and $b _ { 2 } ~ \in ~ \mathbb { R }$ . The hidden width H is small, and the same network is used for every variable. This sharing avoids allocating a nonlinear network to each channel. We initialise $\alpha _ { n } , \beta _ { n } , w _ { 2 }$ , and $b _ { 2 }$ to zero, so the initial local residual is zero.

## D. Low-Rank Cross-Variable Correction

A variable’s own summary may be insufficient when its measurements are sparse or stale. We form a source vector $s ~ \in ~ \mathbb { R } ^ { N }$ from recent values and mix it through a rank-r factorisation:

$$
\begin{array} { r } { s _ { n } = \ell _ { n } z _ { n } ^ { \mathrm { o b s } } \exp ( - a _ { n } d _ { n } ) , \qquad z = s ^ { \mathsf { T } } U V . } \end{array}\tag{11}
$$

The learned matrices are $U \in \mathbb { R } ^ { N \times r }$ and $V \in \mathbb { R } ^ { r \times N }$ . Their product uses $2 N r$ parameters instead of $N ^ { 2 }$ . The integer r is distinct from the descriptor vector $\mathbf { \nabla } _ { \mathbf { r } _ { n } . }$ . The availability indicator removes unobserved sources, and the exponential factor reduces the contribution of older sources. Self-contributions are allowed because the diagonal of UV is not removed.

A target-specific gate then controls the context contribution:

$$
\begin{array} { c } { \gamma _ { n } = \sigma ( \pmb { \eta } _ { n } ^ { \top } \pmb { r } _ { n } ) , } \\ { \delta _ { n } ^ { \mathrm { c r o s s } } ( q ) = \gamma _ { n } \exp [ - \Delta ( q ) d _ { n } ] z _ { n } . } \end{array}\tag{12}
$$

Here $\eta _ { n } \in \mathbb { R } ^ { 3 }$ is learned and $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ . Source attenuation depends on the age of each input variable. The gate depends on the observation state of the target variable. The final decay depends on forecast distance. The learned gate is not constrained to increase monotonically with missingness.

We initialise U from a zero-mean Gaussian with standard deviation $1 / \sqrt { N }$ and set $V = 0$ . The initial cross residual is therefore zero. Together with the local initialisation, the complete model starts from persistence, $\widehat { y } _ { n } ( q ) = \ell _ { n }$ , and learns only the corrections supported by training.

## E. Training, Reuse, and Computational Properties

1) Objective: For a minibatch B, let $A _ { b j n }$ indicate whether the target of record $b ,$ query j, and variable n is observed. Training minimizes

$$
\mathcal { L } _ { B } = \frac { \sum _ { b \in \mathcal { B } , j , n } A _ { b j n } ( \widehat { y } _ { b j n } - y _ { b j n } ) ^ { 2 } } { \operatorname* { m a x } \{ 1 , \sum _ { b \in \mathcal { B } , j , n } A _ { b j n } \} } .\tag{13}
$$

All four datasets use Adam with a fixed learning rate, gradientnorm clipping at 1, at most 200 epochs, and early stopping after 30 epochs without validation-MSE improvement.

2) Exact reuse of the historical summary: Write the model as $f _ { \theta } ( X , M , t , q ) = D _ { \theta } ( E ( X , M , t ) , q )$ , where E is the fixed encoder and $D _ { \theta }$ the learned readout. For an unchanged history, replacing $E ( X , M , t )$ by its stored value leaves both the prediction and the gradient with respect to θ unchanged. The encoder has no parameter gradient that would otherwise need to be updated. Learned damping, gates, and query features remain in $D _ { \theta }$ and are recomputed at each step.

This reuse is valid only while the history, normalisation, decay bank, and numerical constants are unchanged. A new observation or a changed scale bank requires a new summary. The current implementation does not maintain an exact streaming update across a moving origin. For multiple queries against the same window, however, the history is encoded once, and each query is decoded independently. Adding or removing another query cannot change a prediction already requested.

3) Bounds and size: For $x \geq 0 , 0 \leq 1 - e ^ { - x } \leq \operatorname* { m i n } \{ x , 1 \}$ Substitution into Eq. (8) gives

$$
0 \leq h _ { n } ( q ) \leq \operatorname* { m i n } \{ e _ { n } ( q ) , 1 / d _ { n } \} .\tag{14}
$$

For fixed $d _ { n } > 0 ,$ , Eq. (14) bounds the extrapolation step $h _ { n } .$ Conditional on masks, times, and learned parameters, Eq. (12) has the form $\ell ^ { \mathsf { T } } D _ { s } U V D _ { q }$ , where $D _ { s }$ and $D _ { q }$ are diagonal source and target factors. Its rank is at most r. The rank statement concerns the cross residual only, not the complete nonlinear predictor.

The linear heads, biases, damping coefficients, and gates contain $N ( 2 K + 7 )$ parameters. The factorisation adds 2Nr and the shared MLP adds $H ( 2 K + 7 ) + 1$ . Hence

$$
P = N ( 2 K + 7 + 2 r ) + H ( 2 K + 7 ) + 1 .\tag{15}
$$

This count is independent of L and Q. For batch size $B ,$ encoding takes $O ( B L N K )$ work, the local readout takes $O ( B Q N K H )$ , and cross-variable mixing takes $O ( B N r \textrm { + }$ $B Q N )$ . The summary stores $B [ N ( 2 K + 4 ) + 1 ]$ scalar values. These bounds make clear which costs depend on the observed history and which remain for each query. Full inference still includes encoding whenever a fresh history is supplied.

## F. Why the Summary Is Stable and Reusable

The representation is deliberately restricted. Its usefulness depends on retaining a supported local trend, controlling extrapolation, and separating fixed historical work from learned computation. We establish the properties needed for those choices below.

1) Connection to weighted local regression: Fix a variable and a scale, and temporarily omit their subscripts. If the numerical floors are inactive, the normalised weights are $\begin{array} { r } { p _ { i } = w _ { i } M _ { i } / \sum _ { j } w _ { j } M _ { j } } \end{array}$ . Consider the local line with intercept a and slope b:

$$
\operatorname* { m i n } _ { a , b } \ \sum _ { i } p _ { i } ( x _ { i } - a - b u _ { i } ) ^ { 2 } + \rho b ^ { 2 } .\tag{16}
$$

Its intercept condition gives $a \ = \ \mu \ - \ b \bar { u }$ . Substitution into the slope condition gives $b _ { * } = c / ( v + \rho )$ . Thus Eq. (6) is a ridge slope followed by the support-dependent shrinkage χ. The shrinkage is an additional modelling choice, rather than the exact optimiser of Eq. (16). When denominator floors are active, E is not a probability average, and this regression interpretation does not hold exactly.

For m observed entries with positive weights, Cauchy– Schwarz gives $\begin{array} { r } { 1 / m \le \sum _ { i } p _ { i } ^ { 2 } \le 1 } \end{array}$ . Therefore $1 \leq \nu \leq m$ and $0 \leq \chi < 1$ . Equal weights recover the observation count, whereas concentration on one observation reduces the effective count toward one. For example, normalized weights $( 1 / 2 , 1 / 2 )$ give $\nu = 2$ and $\chi = 1 / 3$ . Weights (0.99, 0.01) give $\nu \approx 1 . 0 2$ and $\chi \approx 0 . 0 1$ . Two recorded points therefore need not provide two equally informative supports for a slope.

Proposition 1 (Empty and singleton histories). In real arithmetic, the stabilised moment encoder assigns zero slope to an empty history and to a history with one observed value.

Proof. An empty history has zero covariance. For a singleton of weight w, if $w ^ { 2 } \geq \epsilon ,$ the floors are inactive, covariance is zero, and $\nu = 1 . \mathrm { I f } \epsilon \leq w < \sqrt { \epsilon } .$ , then $D = w$ and $\nu = w ^ { 2 } / \epsilon <$ 1, so $\chi = 0$ . If $w < \epsilon ,$ then $D = \epsilon$ and $\nu = \epsilon < 1$ , again giving $\chi = 0$ . The boundary cases give the same slope.

This property prevents the model from interpreting one observation as evidence for a local trend. It does not force the entire prediction to be zero for a missing variable. Learned biases and information from other variables can still contribute.

2) Extrapolation and conditional interaction rank: For fixed $d > 0$ , differentiation of $h ( e ) = ( 1 - e ^ { - d e } ) / d$ yields $h ^ { \prime } ( e ) = e ^ { - d e } > 0$ and $h ^ { \prime \prime } ( e ) = - \dot { d e } ^ { - d e } < 0 .$ . The effective step increases with elapsed time, but its marginal increase decreases. Together with Eq. (14), this explains how the model retains short-horizon trend information without linearly extending the same slope indefinitely. The bound is conditional on d. As d approaches zero, its asymptotic limit 1/d increases. The levels, learned readout weights, and complete forecast remain unbounded.

The diagonal factors in the cross residual are

$$
\begin{array} { r l } & { D _ { s } = \mathrm { d i a g } \left( z _ { n } ^ { \mathrm { o b s } } e ^ { - a _ { n } d _ { n } } \right) , } \\ & { D _ { q } = \mathrm { d i a g } \left( \gamma _ { n } e ^ { - \Delta d _ { n } } \right) . } \end{array}\tag{17}
$$

Here diag places its arguments on the diagonal. Since multiplication cannot increase rank, rank $\left( D _ { s } U V D _ { q } \right) \leq r .$ . Missing source variables contribute zero through $D _ { s }$ , while each target’s gate can modulate the transferred value independently. The statement holds conditional on the observation state. It does not restrict the rank of the complete predictor across histories with different masks or times.

3) Exact reuse during learning:

Proposition 2 (Prediction and gradient preservation). Let E be a deterministic encoder independent of learned parameters θ. For a fixed history x, replacing $E ( x )$ in $D _ { \theta } ( E ( x ) , q )$ by its stored value preserves predictions and derivatives with respect to θ wherever those derivatives exist.

Proof. Set $z ~ = ~ E ( x )$ . Both evaluations compute $D _ { \theta } ( z , q )$ at the same argument. Since $\partial E ( x ) / \partial \theta = 0$ , the chain-rule contribution through the encoder is zero. Both derivatives are the derivative of the same readout.

Consequently, with identical initialisation, minibatch order, numerical execution, and optimiser state, replacing recomputation by exact stored summaries leaves the sequence of Adam updates unchanged. This claim concerns gradients with respect to model parameters. It does not preserve an input gradient through a detached cache. Changing preprocessing, augmenting the historical values, or learning the decay rates would invalidate the stated reuse rule.

4) Execution and storage: The operator has three stages. First, replace masked values with zero, find the historical origin, and accumulate the moments and observation state. Second, for each query, compute damping, local features, and the shared readout. Third, compute the source mixture once for the history, apply the target gates and query decay, and add both residuals to the latest value. No query-axis aggregation is required, so query chunks can be evaluated independently while sharing the same summary.

A stored summary occupies $4 [ N ( 2 K + 4 ) + 1 ]$ bytes per record in FP32. Under the final configurations, this is 676 bytes for HumanActivity, 164 for USHCN, 1,732 for PhysioNet, and 5,380 for MIMIC. These values exclude the targets, decoder activations, and storage-system overhead. Retaining all summaries trades memory for repeated encoder work, so materialisation is useful only when a history is revisited enough times to justify its construction and storage.

## IV. EXPERIMENTS

We evaluate CTLite on four IMTS benchmarks: Human-Activity, USHCN, PhysioNet, and MIMIC. Our experiments examine whether compact, reusable historical summaries can support accurate forecasting at a low computational cost. We first describe the datasets, evaluation protocol, and implementation settings, then compare forecasting accuracy and computational efficiency. We further investigate how inference and training costs change with batch size, quantify the benefit of reusing historical summaries, and assess the contribution of each modelling component through ablation studies. We report CTLite’s results over five random seeds to characterise performance variability.

## A. Experimental Setup

1) Datasets and evaluation: We use HumanActivity, USHCN, repaired PhysioNet, and MIMIC, with 12, 5, 36, and 96 variables. Clinical data originate from the PhysioNet 2012 challenge and MIMIC [39], [40]. Table II lists the supplied split sizes. We retain their normalisation, history windows, query timestamps, and masks. In particular, repaired PhysioNet uses history in [0, 36) hours and the first three retained future hourly bins within [36, 48). An observed target is a scalar value, not an entire query timestamp.

TABLE I  
FORECASTING PERFORMANCE ON FOUR IMTS DATASETS (MEAN ± STANDARD DEVIATION). BOLD AND UNDERLINE MARK THE BEST AND SECOND-BEST MEANS. † DENOTES REPAIRED PHYSIONET.
<table><tr><td>Dataset</td><td colspan="2">HumanActivity</td><td colspan="2">USHCN PhysioNet†</td><td colspan="2">MIMIC</td></tr><tr><td>Metric</td><td>MSE↓</td><td>MAE↓ MSE↓</td><td>MAE↓</td><td>MSE↓ MAE↓</td><td>MSE↓</td><td>MAE↓</td></tr><tr><td>PrimeNet</td><td>4.2507±0.0041 1.7018±0.0011</td><td></td><td></td><td>0.4930±0.00150.4954±0.0018|0.8128±0.000040.6977±0.00006</td><td>0.9073±0.0001 0.6614±0.0001</td><td></td></tr><tr><td>NeuralFlows</td><td>0.1722±0.00900.3150±0.0094</td><td>0.2087±0.02580.3157±0.0187</td><td></td><td>0.3910±0.003030.4360±0.00201</td><td>0.6085±0.0101 0.5306±0.0066</td><td></td></tr><tr><td>CRU</td><td></td><td></td><td></td><td>0.1387±0.00730.2607±0.0092|0.2168±0.01620.3180±0.0248|0.5917±0.00340 0.5604±0.00199|0.5895±0.00920.5151±0.0048</td><td></td><td></td></tr><tr><td>mTAN</td><td>0.0993±0.00260.2219±0.0047</td><td></td><td></td><td>0.5561±0.2020 0.5015±0.0968|0.3711±0.004590.4231±0.00356|0.9408±0.11260.6755±0.0459</td><td></td><td></td></tr><tr><td>SeFT</td><td>1.3786±0.0024 0.9762±0.0007</td><td></td><td></td><td>0.3345±0.0022 0.4083±0.0084|0.7912±0.002230.6860±0.00079|0.9230±0.00150.6628±0.0008</td><td></td><td></td></tr><tr><td></td><td>GNeuralFlow|0.3936±0.15850.4541±0.0841</td><td></td><td></td><td>0.2205±0.0421 0.3286±0.0412|0.8433±0.154580.6791±0.06523|0.8957±0.02090.6450±0.0072</td><td></td><td></td></tr><tr><td>GRU-D</td><td></td><td></td><td></td><td>0.1893±0.06270.3253±0.0485|0.2097±0.04930.3045±0.03050.3270±0.001570.3917±0.00091</td><td>0.4759±0.01000.4526±0.0055</td><td></td></tr><tr><td>Raindrop</td><td></td><td></td><td></td><td>0.0916±0.0072 0.2114±0.0072|0.2035±0.03360.3029±0.0264|0.3330±0.001570.3965±0.00138|0.6754±0.18290.5444±0.0868</td><td></td><td></td></tr><tr><td>Warpformer</td><td></td><td></td><td></td><td>0.0449±0.00100.1228±0.0018|0.1888±0.05980.2939±0.0591|0.2888±0.001410.3556±0.00095</td><td></td><td>0.4302±0.00350.4025±0.0014</td></tr><tr><td>tPatchGNN</td><td>0.0443±0.00090.1247±0.0031</td><td></td><td></td><td>0.1885±0.04030.3084±0.04790.2923±0.002910.3595±0.00321 0.4431±0.01150.4077±0.0088</td><td></td><td></td></tr><tr><td>GraFITi</td><td></td><td></td><td></td><td></td><td>0.0437±0.00050.1221±0.0017 0.1691±0.0093 0.2777±0.0248|0.2873±0.00213 0.3531±0.00214|0.4359±0.0455 0.4142±0.0297</td><td></td></tr><tr><td>APN</td><td></td><td></td><td></td><td></td><td>0.0421±0.0001 0.1159±0.0006|0.1590±0.01370.2611±0.0167|0.2947±0.00310 0.3610±0.00610|0.4292±0.0027 0.4016±0.0016</td><td></td></tr><tr><td>CTLite</td><td>0.0426±0.0010 0.1167±0.0005</td><td></td><td>0.1648±0.00080.2707±0.0024</td><td>0.2876±0.000170.3527±0.00096|0.4048±0.00080.3844±0.0009</td><td></td><td></td></tr></table>

TABLE II  
SPLIT SIZES AND OBSERVED TARGET COUNTS.
<table><tr><td>Dataset</td><td>N</td><td colspan="3">Records (Samples)</td><td colspan="3">Observed targets</td></tr><tr><td></td><td></td><td>Train</td><td>Val</td><td>Test</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>HumanActivity</td><td>12</td><td>949</td><td>193</td><td>218</td><td>25,872</td><td>5,205</td><td>5,910</td></tr><tr><td>USHCN</td><td>5</td><td>901</td><td>101</td><td>112</td><td>2,964</td><td>334</td><td>368</td></tr><tr><td>PhysioNet</td><td>36</td><td>9,703</td><td>1,079</td><td>1,199</td><td>217,124</td><td>24,061</td><td>27,008</td></tr><tr><td>MIMIC</td><td>96</td><td>17,212</td><td>1,913</td><td>2,125</td><td>187,419</td><td>20,874</td><td>23,015</td></tr></table>

Let $\Omega _ { S }$ be the observed scalar targets in split S, and let $T _ { S } = | \Omega _ { S }$ . The reported metrics are

$$
\begin{array} { r l } & { \mathrm { M S E } _ { S } = T _ { S } ^ { - 1 } \displaystyle \sum _ { ( b , j , n ) \in \Omega _ { S } } \left( \widehat { y } _ { b j n } - y _ { b j n } \right) ^ { 2 } , } \\ & { \mathrm { M A E } _ { S } = T _ { S } ^ { - 1 } \displaystyle \sum _ { ( b , j , n ) \in \Omega _ { S } } | \widehat { y } _ { b j n } - y _ { b j n } | . } \end{array}\tag{18}
$$

We aggregate errors over all observed scalar targets, including incomplete batches, and report five-seed (2024–2028) means and sample standard deviations: For seed errors $e _ { 1 } , \ldots , e _ { 5 } ,$ we report $\bar { e } = \sum _ { s } e _ { s } / 5$ and sample deviation $\begin{array} { r } { s _ { e } = [ \sum _ { s } ( e _ { s } - } \end{array}$ $\bar { e } ) ^ { 2 } / 4 ] ^ { 1 / 2 }$

2) Final configurations and selection: Table IV gives all dataset-specific model and training settings. The numerical constants $\epsilon = 1 0 ^ { - 6 }$ and $\rho \ : = \ : 1 0 ^ { - 3 }$ are shared. Dropout is not used. The weight decay is coupled Adam regularisation. Each dataset uses one configuration for all five seeds.

3) Baselines and Data Preparation: To evaluate the performance of CTLite, we compare it with 12 baseline models.

These baseline models can be broadly categorised into two groups: (1) IMTS Classification/Imputation Models, including PrimeNet [41], SeFT [24], mTAN [25], GRU-D [23], and Warpformer [42]. (2) IMTS Forecasting Models, including NeuralFlows [43], CRU [44], GNeuralFlow [26], tPatchGNN [4], GraFITi [3], and APN [6]. Baseline results for Human-Activity, USHCN, and MIMIC are adopted from [6]. For PhysioNet, we use a corrected version that reconstructs hourly observations from raw records, resolves variable aliases, treats negative values as missing, and replaces the legacy 35.5-hour cutoff with a history interval of [0, 36) hours. Prediction targets are the first three retained future hourly bins within [36, 48). Patient partitions remain unchanged, and normalisation uses training statistics only. The correction changes the observed test targets from 23,694 to 27,008, making errors across versions non-comparable. We therefore use the corrected PhysioNet baseline results alongside CTLite evaluated on the corrected caches.

## B. Predictive Accuracy

Forecasting performance on four IMTS datasets (mean ± standard deviation) is shown in TABLE I. CTLite has the lowest mean MSE and MAE on MIMIC among the reported models. The errors are 0.40476 and 0.38438, improving on the best published means by 5.69% and 4.29%. On repaired PhysioNet, its MSE of 0.28762 is second to GraFITi’s 0.2873, while its MAE of 0.35265 is the lowest reported mean.

On HumanActivity and USHCN, CTLite ranks second in both metrics. Its MSE is 1.09% and 3.66% above the best published mean, respectively. It shows that a very small readout can be competitive while preserving a clear cost advantage.

TABLE III  
MATCHED EFFICIENCY AT BATCH SIZE 32 (MEAN ± STANDARD DEVIATION). THROUGHPUT IS IN THOUSANDS OF RECORDS/S, AND MEMORY IS PEAK ALLOCATED MIB. BOLD IS BEST, UNDERLINE SECOND BEST.
<table><tr><td>Dataset</td><td>Model</td><td>Params↓</td><td>Train (ms) ↓</td><td>Infer (ms) ↓</td><td>Throughput ↑</td><td>Train MiB ↓</td><td>Infer MiB ↓</td></tr><tr><td rowspan="3">HumanActivity</td><td>APN</td><td>28,789</td><td>8.905±0.034</td><td>3.530±0.021</td><td>9.065±0.055</td><td> $4 6 3 . 8 3 { \pm } 1 . 0 0 $ </td><td>229.90±0.41</td></tr><tr><td>GraFITi</td><td>249,409</td><td>8.887±0.053</td><td>4.205±0.007</td><td>7.610±0.012</td><td> $2 8 0 . 2 1 { \pm } 1 . 0 8 $ </td><td>62.92±0.31</td></tr><tr><td>CTLite</td><td>365</td><td>2.232±0.017</td><td>0.665±0.008</td><td>48.119±0.579</td><td> $\mathbf { 1 8 . 4 6 { \scriptstyle \pm 0 . 0 4 } }$ </td><td>10.24±0.03</td></tr><tr><td rowspan="3">USHCN</td><td>APN</td><td>1,970</td><td>3.440±0.024</td><td>1.062±0.006</td><td>30.119±0.175</td><td> $\underline { { 1 7 6 . 3 0 { \pm } 2 . 7 1 } }$ </td><td>69.54±1.04</td></tr><tr><td>GraFITi</td><td>495,617</td><td>8.516±0.114</td><td>3.940±0.055</td><td>8.122±0.113</td><td> $3 3 9 . 1 3 { \pm } 8 . 8 0$ </td><td>137.81±3.01</td></tr><tr><td>CTLite</td><td>88</td><td>2.205±0.016</td><td>0.654±0.003</td><td>48.926±0.250</td><td> $\mathbf { 1 7 . 7 0 { \pm } 0 . 0 1 }$ </td><td>9.55±0.01</td></tr><tr><td rowspan="3">PhysioNet</td><td>APN</td><td>6,701</td><td>2.692±0.020</td><td>0.813±0.002</td><td>39.351±0.085</td><td>59.84±0.00</td><td>31.20±0.00</td></tr><tr><td>GraFITi</td><td>250,945</td><td>9.531±0.621</td><td>4.778±0.411</td><td>6.736±0.565</td><td>325.74±36.61</td><td>59.46±6.36</td></tr><tr><td>CTLite</td><td>889</td><td>2.223±0.035</td><td>0.662±0.007</td><td>48.346±0.480</td><td> $\mathbf { 1 8 . 2 5 { \scriptstyle \pm 0 . 0 0 } }$ </td><td>10.05±0.00</td></tr><tr><td rowspan="3">MIMIC</td><td>APN</td><td>13,777</td><td>7.749±0.449</td><td>3.028±0.157</td><td>10.593±0.579</td><td> $3 8 7 . 1 4 { \pm } 4 8 . 1 9$ </td><td>181.86±15.15</td></tr><tr><td>GraFITi</td><td>254,785</td><td>10.808±1.399</td><td>5.241±0.924</td><td>6.311±1.456</td><td> $3 6 3 . 2 1 { \pm } 7 6 . 2 4$ </td><td>78.28±13.94</td></tr><tr><td>CTLite</td><td>2,537</td><td>2.238±0.018</td><td>0.667±0.006</td><td>47.978±0.443</td><td>20.99±0.27</td><td>12.78±0.35</td></tr></table>

TABLE IV

FINAL MODEL AND TRAINING SETTINGS. ALL TASKS USE CONSTANT LEARNING RATES AND RAW ADAM WEIGHTS.
<table><tr><td>Dataset</td><td>Fixed rates Λ</td><td>r</td><td>H</td><td>Learning rate</td><td>Batch size</td><td>Weight decay</td><td>Params</td></tr><tr><td>HumanActivity</td><td>{0, 1, 4, 16, 64}</td><td>1</td><td>8</td><td>0.01</td><td>16</td><td>0.0001</td><td>365</td></tr><tr><td>USHCN</td><td>{0,8}</td><td>1</td><td>2</td><td>0.0003</td><td>4</td><td>0.01</td><td>88</td></tr><tr><td>PhysioNet</td><td>{0, 2, 8, 32}</td><td>4</td><td>4</td><td>0.001</td><td>64</td><td>0.0003</td><td>889</td></tr><tr><td>MIMIC</td><td>{0, 2, 8, 32, 128}</td><td>4</td><td>8</td><td>0.001</td><td>32</td><td>0.0001</td><td>2,537</td></tr></table>

1) Variation across training seeds: Table V gives every validation and test result of the complete model. Validation determines the checkpoint within a run, and test values are recorded only for that checkpoint. This exposes the variation behind the aggregate scores. Low variation across these runs indicates stability to the selected training seeds on this split.

## C. Efficiency on Matched Workloads

1) Measurement protocol: We compare CTLite, APN, and GraFITi on one RTX 3090 using PyTorch 2.5.1+cu124, FP32, four CPU threads, and disabled TF32 matrix multiplication. For each dataset and seed, the same 32 training records are supplied with identical values, masks, query times, and withinbatch padding. All efficiency experiments use the same five random seeds (2024–2028), with results reported as the mean and sample standard deviation across runs. APN and GraFITi retain their official architectures, optimiser settings, and native losses. In particular, GraFITi retains its variable-averaged loss and graph construction.

Peak memory is the CUDA allocator’s peak allocated memory after warm-up, including resident inputs and model state. Counters are reset before each measured operation. Training additionally includes gradients and Adam state. Allocatorreserved but not used memory and memory outside PyTorch are excluded from the displayed peak, so these values are not total device-memory occupancy. Throughput is averaged from the five individual rates, rather than calculated as the reciprocal

TABLE V  
VALIDATION AND TEST ERRORS OF THE COMPLETE MODEL BY SEED.
<table><tr><td>Dataset</td><td>Seed</td><td>Val MSE</td><td>Val MAE</td><td>Test MSE</td><td>Test MAE</td></tr><tr><td rowspan="5">HumanActivity</td><td>2024</td><td>0.045863</td><td>0.118920</td><td>0.042599</td><td>0.117128</td></tr><tr><td>2025</td><td>0.046127</td><td>0.119902</td><td>0.042174</td><td>0.116331</td></tr><tr><td>2026</td><td>0.047570</td><td>0.120034</td><td>0.044177</td><td>0.117122</td></tr><tr><td>2027</td><td>0.046258</td><td>0.118295</td><td>0.041547</td><td>0.116031</td></tr><tr><td>2028</td><td>0.045992</td><td>0.119011</td><td>0.042301</td><td>0.116980</td></tr><tr><td rowspan="5">USHCN</td><td>2024</td><td>0.353462</td><td>0.308676</td><td>0.166031</td><td>0.272905</td></tr><tr><td>2025</td><td>0.352962</td><td>0.305179</td><td>0.164978</td><td>0.268582</td></tr><tr><td>2026</td><td>0.358588</td><td>0.308169</td><td>0.164755</td><td>0.269127</td></tr><tr><td>2027</td><td>0.355586</td><td>0.307593</td><td>0.164687</td><td>0.273676</td></tr><tr><td>2028</td><td>0.353245</td><td>0.305297</td><td>0.163671</td><td>0.269114</td></tr><tr><td rowspan="5">PhysioNet</td><td>2024</td><td>0.299604</td><td>0.354958</td><td>0.287338</td><td>0.351516</td></tr><tr><td>2025</td><td>0.299745</td><td>0.356185</td><td>0.287629</td><td>0.352951</td></tr><tr><td>2026</td><td>0.299682</td><td>0.356594</td><td>0.287757</td><td>0.353648</td></tr><tr><td>2027</td><td>0.299585</td><td>0.356553</td><td>0.287721</td><td>0.353373</td></tr><tr><td>2028</td><td>0.299437</td><td>0.354797</td><td>0.287664</td><td>0.351764</td></tr><tr><td rowspan="5">MIMIC</td><td>2024</td><td>0.418445</td><td>0.390700</td><td>0.405562</td><td>0.383714</td></tr><tr><td>2025</td><td>0.420998</td><td>0.392848</td><td>0.404776</td><td>0.385185</td></tr><tr><td>2026</td><td>0.420267</td><td>0.391029</td><td>0.403561</td><td>0.383412</td></tr><tr><td>2027</td><td>0.419227</td><td>0.392369</td><td>0.405495</td><td>0.385361</td></tr><tr><td>2028</td><td>0.419151</td><td>0.391649</td><td>0.404404</td><td>0.384247</td></tr></table>

![](images/d2e0ff6868ea5df7cbf840861f2da17d87bc88adc566a5b177358bc712b126b4.jpg)  
Fig. 3. Relative reductions, $1 0 0 ( 1 { - } m _ { \mathrm { C T L i t e } } / m _ { \mathrm { b a s e } } )$ . Positive values favour CTLite.

of average latency.

2) Size, latency, and memory: Table III and Fig. 3 compare computational costs at a matched batch size of B = 32. CTLite achieves the lowest parameter count, training and inference latency, and peak allocated memory, together with the highest throughput, on all four datasets.

CTLite contains only 88–2,537 parameters, requiring 352– 10,148 bytes of FP32 parameter storage. Relative to APN and GraFITi, the parameter count is reduced by 81.59–98.73% and 99.00–99.98%, respectively.

The compact components also provide substantial runtime benefits. CTLite completes full inference in 0.654–0.667 ms, reducing latency by 18.60–81.16% compared with APN and 83.40–87.27% compared with GraFITi. Its throughput reaches 47.98–48.93 thousand records per second, corresponding to 1.23–5.31× that of APN and 6.02–7.60× that of GraFITi. For each run, throughput is computed as 1000B/t, where t is inference latency in milliseconds. Training steps require 2.205–2.238 ms, yielding reductions of 17.42–79.29% across the two baselines. The improvement over APN is smallest on PhysioNet, where its inference and training costs are already relatively low. Thus, the magnitude of the runtime benefit depends on both the dataset and the baseline, despite consistently lower costs for CTLite under this workload.

CTLite also reduces the memory required for execution. Peak allocated memory is 17.70–20.99 MiB during training and 9.55–12.78 MiB during inference, representing reductions of 69.50–96.02% and 67.80–95.54%, respectively, across the two baselines. Here, one MiB equals 2<sup>20</sup> bytes. Unlike parameter storage, these measurements include resident inputs and intermediate tensors, as well as gradients and optimiser state during training. The memory savings therefore extend beyond the model weights to the measured execution footprint. All primary measurements include historical encoding without using cached features.

![](images/74d9b4d0f8d24f5d39a541bd7ced5f1e62f35ab7bf6e2a780c34c8d9921cd4a9.jpg)  
Fig. 4. Full per-request CPU inference without cached histories. Bars show five-seed mean median latency, and sample SD. The log axis accommodates model differences. Input tensors are CPU resident; disk IO and process startup are excluded.

3) CPU-only deployment: We evaluate single-record inference (B = 1) on an Intel Core i7-12700KF using one and four CPU threads. All models process identical requests in FP32, with full history encoding and no feature caching. Inputs reside in CPU memory, and timing excludes data loading and process startup. We compute median and 95th-percentile (p95) request latency separately for each seed and report their means and sample standard deviations over five seeds.

CTLite achieves the lowest mean median latency in all eight dataset–thread settings. With one thread, its latency is 0.291± 0.003, 0.289 ± 0.002, 0.292 ± 0.001, and $0 . 3 0 0 \pm 0 . 0 0 4$ ms on HumanActivity, USHCN, PhysioNet, and MIMIC, respectively. Across both thread settings, CTLite reduces mean median latency by 27.03–88.86% relative to APN and 77.44– 93.01% relative to GraFITi. It also achieves the lowest mean p95 latency in every setting, indicating that the advantage extends beyond typical requests to the latency tail.

Increasing the thread count does not improve CTLite’s latency, suggesting limited benefit from additional parallelism for its small computational workload. Nevertheless, CTLite remains faster than both baselines with four threads. These results complement the GPU measurements by demonstrating efficient single-record execution without accelerator hardware or cached summaries.

## D. Batch Scaling

We repeat the complete benchmark at $B \in \{ 1 , 1 6 , 3 2 , 6 4 \}$ using the final model settings. Each seed selects 64 training records per dataset. All three models receive the same values, masks, times, and padding at a given batch size. Each model– dataset–seed–batch combination uses a fresh GPU process, yielding 240 isolated workloads. This prevents allocations and optimiser state from carrying across batch sizes. Table III and Fig. 3 use the B = 32 slice of this same measurement set.

Fig. 5 reports full inference and full training steps, including historical encoding. Both axes use logarithmic scales, and error bars show the sample standard deviation of five run medians. CTLite’s mean inference cost grows from 0.648– 0.650 ms at B = 1 to 0.660–0.669 ms at B = 64. Its trainingstep cost remains within 2.174–2.310 ms over all measured datasets and batch sizes. The shallow, shared readout therefore accommodates more records without a proportional increase in elapsed GPU time on these workloads.

Especially, as the batch size increases from 1 to 64, CTLite’s inference and training-step times remain nearly constant across all four datasets, whereas both baselines exhibit substantial increases, particularly at larger batches. CTLite therefore processes more records per step with little additional latency, translating larger batches into higher throughput. This behaviour benefits both batched prediction and training, where the computational cost per record decreases as more records are processed together. The runtime advantage becomes especially clear at batch sizes of 32 and 64, where CTLite is faster than both baselines on every dataset. Although APN achieves lower inference latency at batch size 1 and on PhysioNet at batch size 16, CTLite provides more favourable scaling over the evaluated batch sizes. These results demonstrate CTLite’s efficiency for batched workloads under the same tested hardware and sequence lengths.

![](images/2d21e1f5c3ab4bfc85c35773c7a1179491c116d99116a3d4e2e1b3dbffd5310e.jpg)  
Fig. 5. Batch scaling. Full inference above and full training steps below. Error bars show sample SD over five seeds.

TABLE VI  
HISTORICAL ENCODING AND COMPUTATION WITH A STORED SUMMARY (MS).
<table><tr><td>Dataset</td><td>Encode once Cached inference Cached train step</td><td></td><td></td></tr><tr><td>HumanActivity</td><td> $0 . 3 7 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 3 3 6 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $1 . 8 1 7 { \scriptstyle \pm 0 . 0 2 4 }$ </td></tr><tr><td>USHCN</td><td> $0 . 3 7 1 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 3 3 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $1 . 7 8 6 { \scriptstyle \pm 0 . 0 0 6 }$ </td></tr><tr><td>PhysioNet</td><td> $0 . 3 7 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 3 4 1 { \pm } 0 . 0 0 9$ </td><td> $1 . 8 0 1 { \pm } 0 . 0 3 8$ </td></tr><tr><td>MIMIC</td><td> $0 . 3 7 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 3 3 1 { \pm } 0 . 0 0 3$ </td><td> $1 . 8 0 6 { \pm } 0 . 0 3 1$ </td></tr></table>

## E. Efficiency Gains from Reusing Historical Summaries

CTLite separates historical encoding from prediction. The encoder converts observed values, timestamps, and masks into a fixed summary containing temporal moments and observation-state features. The prediction modules then combine this summary with the requested forecast times. Because the encoder contains no learned parameters, an unchanged history produces the same summary throughout training. The summary can therefore be computed once and reused across subsequent optimiser steps or repeated forecast queries, while the learned prediction modules are evaluated each time.

Table VI measures the computational benefit of this reuse at a batch size of 32. Encode once reports the cost of constructing the summary. Cached inference reports inference time when the summary is already available in GPU memory. Cached train step reports a training step using the stored summary, including prediction, loss computation, backpropagation, and the optimiser update. The latter two measurements exclude summary construction, which is performed before reuse begins.

Across the four datasets, summary construction takes 0.370– 0.376 ms. Once the summary is available, inference requires only 0.331–0.341 ms, reducing latency by 48.47–50.40% relative to CTLite’s full inference in Table III. Training steps require 1.786–1.817 ms, a reduction of 18.58–19.33%. These savings come from avoiding repeated historical encoding, without changing the forecasting computation. We verify exact equality between predictions obtained from stored summaries and complete forward execution in the measured implementation.

Reuse is beneficial when the same historical window is accessed repeatedly. A new observation, a shifted window, or a change in preprocessing or encoder settings requires recomputing the summary. Updating the learned prediction weights alone does not. The reported cached timings assume GPUresident summaries and exclude persistent-storage and retrieval overhead. All primary efficiency comparisons in Table III and Fig. 5 include historical encoding.

TABLE VII  
ABLATION STUDIES. BOLD MARKS THE LOWEST ERROR IN EACH COLUMN.
<table><tr><td>Dataset</td><td colspan="2">HumanActivity</td><td colspan="2">USHCN</td><td colspan="2">PhysioNet</td><td colspan="2">MIMIC</td></tr><tr><td>Variant</td><td>MSE↓</td><td>MAE↓</td><td>MSE↓</td><td>MAE↓</td><td>MSE↓</td><td>MAE↓</td><td>MSE↓</td><td>MAE↓</td></tr><tr><td>Full model</td><td>0.04256±0.00098 0.11672±0.00051</td><td></td><td>0.16482±0.00084</td><td>0.27068±0.00241</td><td>0.28762±0.00017 0.35265±0.00096</td><td></td><td>0.40476±0.00083</td><td>0.38438±0.00087</td></tr><tr><td>No cross residual</td><td>0.04259±0.00080 0.11672±0.00131</td><td></td><td>0.16475±0.00093</td><td>0.26739±0.00097</td><td>0.28982±0.00008 0.35309±0.00091</td><td></td><td>0.42632±0.00062</td><td>0.39337±0.00052</td></tr><tr><td>Index/no slope</td><td>0.04287±0.00060 0.11730±0.00114</td><td></td><td>0.16000±0.000640.26774±0.00265</td><td></td><td>0.28840±0.000150.35295±0.00114</td><td></td><td>0.40187±0.00135</td><td>0.38406±0.00183</td></tr></table>

## F. Component Analysis

We retrain two existing block ablations under each full model’s final settings and seeds 2024–2028, whose results are shown in TABLE VII. There is no additional hyperparameter search for the ablations. No cross residual disables Eq. (12). Index/no slope replaces the physical time offsets used in the moment bank with normalised row positions and sets the slope features to zero. Observation ages and query decay remain active in the latter variant. It therefore tests a combined temporal construction, not complete removal of all timestamp information.

The clearest cross-variable gains occur on the clinical tasks. Removing the cross residual increases mean MSE by 0.00220 on PhysioNet and 0.02156 on MIMIC, with an increase in all five paired seeds. HumanActivity changes by only 0.00003, while USHCN has a slightly lower MSE and a lower MAE without that residual. The temporal construction is also taskdependent. Replacing time-based moments by index moments and removing slopes raises MSE on HumanActivity and PhysioNet, but reduces it on USHCN and MIMIC.

## V. CONCLUSION

We presented CTLite, a compact framework for irregular multivariate time series forecasting that separates fixed historical summarisation from learned prediction. Continuoustime moments retain local levels and supported trends, while a small readout combines damped temporal corrections with observation-conditioned cross-variable information. This separation makes historical summaries reusable across optimisation steps and forecast queries without changing predictions or parameter gradients. Across four benchmarks, CTLite achieves the best or second-best mean MSE and MAE among the evaluated methods with only 88–2,537 trainable parameters, alongside lower latency and memory consumption than the latest models at a matched batch size of 32. These results show that a carefully structured historical representation can support accurate irregular forecasting with a small learned predictor and make repeated historical computation avoidable.

## REFERENCES

[1] Y. Rubanova, R. T. Q. Chen, and D. K. Duvenaud, “Latent Ordinary Differential Equations for Irregularly-Sampled Time Series,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[2] E. De Brouwer, J. Simm, A. Arany, and Y. Moreau, “GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[3] V. K. Yalavarthi, K. Madhusudhanan, R. Scholz, N. Ahmed, J. Burchert, S. Jawed, S. Born, and L. Schmidt-Thieme, “GraFITi: Graphs for Forecasting Irregularly Sampled Time Series,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 15, 2024, pp. 16 255–16 263.

[4] W. Zhang, C. Yin, H. Liu, X. Zhou, and H. Xiong, “Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach,” in Proceedings of the International Conference on Machine Learning, vol. 235, 2024, pp. 60 179–60 196.

[5] Y. Luo, B. Zhang, Z. Liu, and Q. Ma, “Hi-Patch: Hierarchical Patch GNN for Irregular Multivariate Time Series,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267, 2025, pp. 41 494–41 519.

[6] X. Liu, X. Qiu, X. Wu, Z. Li, C. Guo, J. Hu, and B. Yang, “Rethinking Irregular Time Series Forecasting: A Simple Yet Effective Baseline,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 28, pp. 23 873–23 881, 2026.

[7] Z. Zhou, Y. Huang, Y. Wang, Y. Wu, J. Kwok, and Y. Liang, “Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 34, pp. 29 115–29 123, 2026.

[8] T. Pelkonen, S. Franklin, J. Teller, P. Cavallaro, Q. Huang, J. Meza, and K. Veeraraghavan, “Gorilla: A Fast, Scalable, In-Memory Time Series Database,” Proceedings of the VLDB Endowment, vol. 8, no. 12, pp. 1816–1827, 2015.

[9] C. Wang, X. Huang, J. Qiao, T. Jiang, L. Rui, J. Zhang, R. Kang, J. Feinauer, K. A. McGrail, P. Wang, D. Luo, J. Yuan, J. Wang, and J. Sun, “Apache IoTDB: Time-series Database for Internet of Things,” Proceedings of the VLDB Endowment, vol. 13, no. 12, pp. 2901–2904, 2020.

[10] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are Transformers Effective for Time Series Forecasting?” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, 2023, pp. 11 121–11 128.

[11] Z. Xu, A. Zeng, and Q. Xu, “FITS: Modeling Time Series with 10k Parameters,” in International Conference on Learning Representations, 2024.

[12] S. Lin, W. Lin, W. Wu, H. Chen, and J. Yang, “SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters,” in Proceedings of the International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 30 211–30 226.

[13] A. Ma, D. Luo, and M. Sha, “MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters,” in International Conference on Learning Representations, 2026.

[14] S. Lin, W. Lin, X. Hu, W. Wu, R. Mo, and H. Zhong, “CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[15] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Zhang, and J. ZHOU, “TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting,” in International Conference on Learning Representations, 2024, pp. 38 626–38 652.

[16] D. Luo and X. Wang, “ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis,” in International Conference on Learning Representations, 2024.

[17] S. Ma, Y.-B. Zhao, Y. Kang, and P. Bai, “Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer,” IEEE Transactions on Artificial Intelligence, vol. 5, no. 10, pp. 5232–5243, 2024.

[18] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “iTransformer: Inverted Transformers Are Effective for Time Series Forecasting,” in International Conference on Learning Representations, 2024.

[19] L. Han, X.-Y. Chen, H.-J. Ye, and D.-C. Zhan, “SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[20] X. Qiu, X. Wu, Y. Lin, C. Guo, J. Hu, and B. Yang, “DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting,” in Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2025, pp. 1185–1196.

[21] S. Ma, Y. Zhao, and Y. Kang, “C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 29, 2026, pp. 24 281–24 289.

[22] S. Ma, Y.-B. Zhao, and Y. Kang, “THCVformer: Modeling temporaldependencies, heterogeneity, and correlations for enhanced time-series forecasting,” Knowledge-Based Systems, p. 115964, 2026.

[23] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent Neural Networks for Multivariate Time Series with Missing Values,” Scientific Reports, vol. 8, p. 6085, 2018.

[24] M. Horn, M. Moor, C. Bock, B. Rieck, and K. Borgwardt, “Set Functions for Time Series,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119, 2020, pp. 4353–4363.

[25] S. N. Shukla and B. Marlin, “Multi-Time Attention Networks for Irregularly Sampled Time Series,” in International Conference on Learning Representations, 2021.

[26] G. Mercatali, A. Freitas, and J. Chen, “Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[27] X. Liu, X. Qiu, H. Cheng, X. Wu, G. Guo, B. Yang, and J. Hu, “ASTGI: Adaptive Spatio-Temporal Graph Interactions for Irregular Multivariate Time Series Forecasting,” in International Conference on Learning Representations, 2026, pp. 23 940–23 958.

[28] X. Yan, Z. Li, J. Zhang, B. Li, J. Yan, and B. Tang, “Beyond Sequences: A Dynamic Hierarchical Heterogeneous Spatio-Temporal Graph for Irregular Multivariate Time Series Forecasting,” in Proceedings of the International Joint Conference on Artificial Intelligence, 2026, pp. 5146– 5154.

[29] J. Liu, M. Cao, and S. Chen, “TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 18, pp. 18 861–18 869, 2025.

[30] B. Li, Z. Liu, Y. Luo, and Q. Ma, “Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting,” in International Conference on Learning Representations, 2026.

[31] C. Wang, Q. Qi, J. Wang, H. Sun, Z. Zhuang, and J. Liao, “Unlocking the Potential of Linear Networks for Irregular Multivariate Time Series Forecasting,” arXiv preprint arXiv:2505.00590, 2025.

[32] R. Li and C. Chen, “Rethinking Irregular Time Series Forecasting from the Perspective of Basis Functions,” arXiv preprint arXiv:2608.17284, 2026.

[33] V. K. Yalavarthi, R. Scholz, S. Born, and L. Schmidt-Thieme, “Probabilistic Forecasting of Irregularly Sampled Time Series with Missing Values via Conditional Normalizing Flows,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 20, 2025, pp. 21 877– 21 885.

[34] V. K. Yalavarthi, R. Scholz, C. Klotergens, K. Madhusudhanan, S. Born,¨ and L. Schmidt-Thieme, “Reliable Probabilistic Forecasting of Irregular Time Series through Marginalization-Consistent Flows,” in International Conference on Learning Representations, 2026, pp. 85 403–85 427.

[35] X. Qiu, J. Hu, L. Zhou, X. Wu, J. Du, B. Zhang, C. Guo, A. Zhou, C. S. Jensen, Z. Sheng, and B. Yang, “TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods,” Proceedings of the VLDB Endowment, vol. 17, no. 9, pp. 2363–2377, 2024.

[36] C. Klotergens, V. K. Yalavarthi, R. Scholz, M. Stubbemann, S. Born,¨ and L. Schmidt-Thieme, “Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting Based on Biological ODEs,” in International Conference on Learning Representations, 2025, pp. 34 961–34 988.

[37] C. Chang, J. Hwang, Y. Shi, H. Wang, W. Wang, W.-C. Peng, and T.-F. Chen, “Time-IMM: A Dataset and Benchmark for Irregular Multimodal

Multivariate Time Series,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[38] F. Spinnato and C. Landi, “PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks,” in International Conference on Learning Representations, 2026.

[39] I. Silva, G. Moody, D. J. Scott, L. A. Celi, and R. G. Mark, “Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012,” in 2012 computing in cardiology, 2012, pp. 245–248.

[40] A. E. Johnson, T. J. Pollard, L. Shen, L.-w. H. Lehman, M. Feng, M. Ghassemi, B. Moody, P. Szolovits, L. Anthony Celi, and R. G. Mark, “MIMIC-III, a Freely Accessible Critical Care Database,” Scientific data, vol. 3, no. 1, p. 160035, 2016.

[41] R. R. Chowdhury, J. Li, X. Zhang, D. Hong, R. K. Gupta, and J. Shang, “Primenet: Pre-training for irregular multivariate time series,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 37, no. 6, 2023, pp. 7184–7192.

[42] J. Zhang, S. Zheng, W. Cao, J. Bian, and J. Li, “Warpformer: A multi-scale modeling approach for irregular clinical time series,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 3273–3285.

[43] M. Bilos, J. Sommer, S. S. Rangapuram, T. Januschowski, andˇ S. Gunnemann, “Neural flows: Efficient alternative to neural odes,” in¨ Advances in neural information processing systems, vol. 34, 2021, pp. 21 325–21 337.

[44] M. Schirmer, M. Eltayeb, S. Lessmann, and M. Rudolph, “Modeling irregular time series with continuous recurrent units,” in International conference on machine learning, 2022, pp. 19 388–19 405.