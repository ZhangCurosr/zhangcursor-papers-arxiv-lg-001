# FlowATC: Aircraft Trajectory Prediction via Flow Matching

Mathurin Petit\* École Polytechnique, Palaiseau, 91128, France

Emir Torun† Technische Universität Berlin, Berlin, 10623, Germany

Louis Brusset Mines Paris–PSL University, Paris, 75006, France

Jordan Kam§ California Institute of Technology, Pasadena, CA, 91125, USA

Alexandre M. Bayen" University of California, Berkeley, Berkeley, CA 94720, USA

Building accurate decision-support tools for next generation air traffic control requires robust trajectory prediction models. We present a flow-matching architecture trained exclusively on historical aircraft trajectories, with no route labels or chart supervision. Trained on 1.15 million Automatic Dependent Surveillance-Broadcast trajectory windows collected over the San Francisco Bay Area, the model generates aircraft trajectory distributions that closely match historical traffic, reproducing known airspace structure around San Francisco Airport such as the shape of SFO's published NITE FOUR departure procedure. Our model is trained directly on the native, irregular ADS-B sampling interval. Trajectory prediction is cast as sequence inpainting using a block-causal Transformer that denoises future state tokens conditioned on the observed history using Conditional Flow Matching or Denoising Diffusion Probabilistic Models. We compare our architecture against constant-velocity, deterministic-Long Short Term Memory, and Conditional Variational Autoencoders baselines. At matched parameter count, CFM outperforms DDPM by 11-26% in minADE@20, and both generative objectives surpass the CVAE baseline by 31–41%. We further show that the error degrades gracefully with prediction horizon, and the architecture remains effective when retrained on temporally decimated feeds. Lastly, we sample K independent completions, yielding spatial probabilistic occupancy estimates that can serve as input to downstream conflict-risk estimation.

## Nomenclature

X = trajectory window, $T \times F$ array of ADS-B samples   
$\mathbf { X } ^ { \mathrm { o b s } }$ = observation prefix, first $T _ { \mathrm { o b s } }$ steps   
$\mathbf { X } ^ { \mathrm { f u t } }$ = future suffix, next $T _ { \mathrm { f u t } }$ steps to predict   
$T$ = total sequence length   
$F$ = features per timestep   
$\phi , \lambda$ = geodetic latitude and longitude   
$d$ = Transformer hidden dimension   
$u _ { \theta }$ = CFM velocity field network   
$p _ { i } = ( x _ { i } , y _ { i } )$ = Lateral position at trajectory timestep i   
$K$ = number of samples in best-of-K evaluation   
minADE@ K = min-over-K average displacement error, m   
minFDE@ K = min-over-K final displacement error, m   
NLL@n = KDE negative log-likelihood at future step n

## I. Introduction

ow altitude air traffc control (ATC) relies on accurate short-horizon aircraft trajectory prediction to maintain safe separation. Automatic Dependent Surveillance-Broadcast (ADS-B) data provides continuous position updates across the National Airspace System (NAS), yet forecasting where an aircraft will be in the next two minutes remains hard. A flight on a standard San Francisco (SFO) arrival may turn left or continue straight depending on runway assignment and traffic sequencing, both invisible from position data alone. Prediction is therefore inherently multimodal, meaning a single observed history is consistent with several physically plausible futures (or modes) [1–3]. The quantity that matters operationally is not only where an aircraft will be, but how probable a conflict is at that location, which calls for a predictor that returns a full distribution over futures rather than a single point estimate [4–6].

The generative-modelling toolkit powering recent advances in image synthesis and robot motion planning transfers naturally to continuous trajectory data. Denoising Diffusion Probabilistic Models (DDPM) [7] learn to reverse a Markov noising chain and, with the Denoising Diffusion Implicit Models (DDIM) sampler [8], generate high-quality samples in a handful of steps. Conditional Flow Matching (CFM) [9] is a more direct alternative regressing a velocity field that transports Gaussian noise to data along straight-line paths, giving a simpler training objective and fewer integration steps. Both families produce samples rather than point estimates. Generating K independent samples at inference yields a distribution over future positions that covers the inherently multimodal space of plausible aircraft states even when the model is conditioned on a single input modality. The Transformer architecture [10] represents data as a set of tokens coupled by self-attention. This tokenized view is attractive for trajectory prediction for two reasons: it is flexible (heterogeneous inputs, i.e., positions, time deltas, and, in the future, ATC voice or weather fields, are simply additional tokens) and fast on modern hardware. Crucially, the Diffusion Transformer (DiT) [11] shows that a Transformer backbone, with the generation step injected through Adaptive Layer Normalization (AdaLN), is an excellent denoiser for diffusion. Specifically, we exploit exactly this synergy: a DiT style backbone hooks up cleanly with both DDPM and CFM, letting us frame trajectory prediction as token-level inpainting.

## Related work

Classical predictors fall into at least four families. Kinematic models (e.g. constant velocity / constant turn) propagate the last observed state forward under a motion assumption; they are interpretable and fast but produce a single deterministic forecast which does not take into account surrounding airspace information. We use constant velocity as a lower bound as it provides a good estimate of the order of magnitude of the error for such prediction tasks. Deterministic neural models, typically an LSTM/Gated Recurrent Unit (GRU) encoder-decoder or a Transformer regressor [12–14], learn data driven dynamics but still emit one trajectory per query, suppressing the uncertainty that matters for safety. Direct multimodal predictors instead decode a fixed set of plausible trajectory hypotheses. ASCENT [2], for example, uses a Transformer encoder with learnable mode queries to predict multiple 3D future trajectories together with associated mode scores in non-towered terminal airspace, achieving strong best-of-K performance on the TrajAir benchmark. Unlike stochastic generative models, however, such approaches represent multimodality through a finite set of explicitly decoded hypotheses rather than by sampling from a continuous conditional distribution. Probabilistic generative models output a distribution: Conditional Variational Autoencoders (CVAE) such as Trajectron++ [4] are widely used probabilistic baselines for multimodal trajectory forecasting, sampling a latent variable to produce diverse futures. In autonomous driving, diffusion-based predictors have recently emerged as a strong alternative to CVAEs. By iteratively denoising noise samples to in distribution data, this technique offers a probabilistic approach to prediction tasks. MotionDiffuser [5] applies DDPM to multi-agent road-traffic forecasting and shows that diffusion captures multimodal distributions without trajectory anchors. Diffusion has also been applied directly to aircraft trajectory prediction: Yin et al. [15] combine the aircraft's history with contextual information representing intent and environmental conditions in a diffusion-based decoder, evaluated at Singapore Changi Airport, and GooDFlight [3] first estimates goal positions, then generates diverse trajectories with a goal-guided diffusion decoder. In the aerospace domain, Briden et al. [16] apply diffusion to spacecraft descent planning, framing trajectory solutions as composable probability density functions; our setting is complementary, targeting probabilistic prediction of civil aircraft from surveillance observations, where maneuver structure is governed by ATC procedures rather than road geometry. More recently, diffusion models have also been widely adopted for robot motion planning: Janner et al. [17] showed that full action trajectories can be generated by iterative denoising guided by reward functions.

Closest to the present work in generative formulation, Figuet et al. [6] apply Conditional Flow Matching to short-term aircraft trajectory prediction, using a Transformer encoder-decoder pair. Their study targets en-route traffic above FL195 in Swiss Free Route Airspace, in an aircraft-centric frame normalized to the last observed state, with absolute position provided as an explicit 8-dimensional context vector rather than encoded in the trajectory features directly. ADS-B is resampled to a uniform 1 Hz grid. Our study is complementary along four axes. First, we target low-altitude terminal airspace, where published procedures dictate maneuver structure and traffic mixes commercial and general aviation; we retain absolute Cartesian coordinates so this structure can be learned without chart supervision (Section VII). Second, we predict a ≈128 s horizon at the native, irregular ADS-B sampling rate, rather than on a resampled grid. Third, we replace the encoder-decoder pair with a single DiT that ingests clean and noisy tokens by concatenation, casting the task as sequence inpainting. Fourth, we benchmark CFM against CVAE and DDPM at matched capacity across three model scales, isolating the contribution of the objective itself.

Our novel contributions include the following:

(i) A sequence-inpainting architecture concatenating observed and noisy tokens, block-causal self-attention, AdaLN time conditioning allowing a DiT to address trajectory completion without an intermediate encoder.

(ii) Benchmarking constant-velocity, deterministic LSTM, and CVAE (Trajectron++) baselines against DDPM and CFM variants of the same backbone, and showing that CFM is substantially more accurate at equal parameter count.

(iii) Characterizing operational flexibility: graceful degradation over the horizon, retraining on lower-rate streams, extrapolation beyond the training horizon, and inference cost.

(iv) Showing that FlowATC recovers Bay Area airspace structure without chart supervision and that its output distribution is accurate: at airspace branch points it reproduces the distribution of maneuvers actually flown and gives each aircraft a close to calibrated distribution over its next maneuver.

## II. Methodology

## A. Aircraft Trajectory Data

ADS-B is a cooperative surveillance technology in which an aircraft periodically broadcasts its own state: identity, position, altitude, velocity, and vertical rate, derived primarily from onboard Global Navigation Satellite System (GNSS). The broadcasts are received by a dense network of ground stations and can be aggregated by public feeds, making ADS-B a high-coverage, low-cost source of trajectory data over busy terminal airspace. We collect ADS-B continuously from the ADS-B LOL live feed (adsb.lol) using the native API that queries aircraft states every 2 s over a circular geofence of 60 nautical miles radius centered at 37.75°N, 122.25°W (San Francisco Bay). The scraper records, for each aircraft, the ICAO24 transponder code, callsign, Unix timestamp, longitude, latitude, barometric altitude, ground speed, true track, and vertical rate. Collection ran for 12 days, 10–22 April 2026, yielding 21,515,794 raw state vectors. The geofence and the airports referenced in this study are shown in Fig. 1; dataset statistics are summarized in Table 1.

![](images/223b3c60a2b0919d91450a8ebfcae253c323eb8f87db498ec8026fcc33102735.jpg)  
Fig. 1 Geographic coverage of the ADS-B collection used for the present work. The dashed circle shows the 60 nautical-mile radius geofence centered on San Francisco Bay (37.75°N, 122.25°W, red dot). Black dots mark the airports referenced in this study. The Cartesian coordinate frame used for trajectory representation (Eq. (13)) is centered on SFO (37.6213°N,122.3790°W).

The collection spans a broad mix of Bay Area traffic (shares below by unique aircraft): large commercial aircraft (ICAO category A3, 41.8%), light general aviation (A1, 32.1%), heavy aircraft such as B747/A380 (A5, 10.8%), and lighter traffic (remaining 15.3%).

## B. Trajectory processing

Raw ADS-B state vectors are segmented into continuous flight segments and windowed into 86-point sequences. Each window is split into a 43-point observed prefix and a 43-point future suffix to predict. The choice of a two-minute history and forecast horizon follows from discussions with pilots, who identified a 2-minute lookahead as the operationally relevant horizon. For maneuver prediction, each 43-point half spans roughly 128 s on average.

We project geodetic coordinates onto a local tangent-plane Cartesian frame centered on SFO, yielding a 6-dimensional feature vector $( x , y , z , \nu _ { x } , \nu _ { y } , \nu _ { z } )$ entirely in meters or meters per second. Because absolute Cartesian coordinates encode geographic position, the model implicitly learns location-specific structure (e.g. that a south-westbound aircraft at y ≈ –15 km is on SFO final approach). Each feature is independently z-score normalized using training-split statistics.

Table 1 Bay Area ADS-B dataset statistics.
<table><tr><td>Variable</td><td>Value</td></tr><tr><td>Collection period</td><td>10 Apr–22 Apr 2026 (12 days)</td></tr><tr><td>Raw ADS-B state vectors</td><td>21,515,794</td></tr><tr><td>Geographic coverage</td><td>60 nm radius circle, centre 37.75°N, 122.25°W</td></tr><tr><td>Cartesian reference (SFO)</td><td>37.6213°N, 122.3790°W</td></tr><tr><td>Segmentation cut</td><td>Gap &gt; 120 s or callsign change</td></tr><tr><td>Features</td><td>x, y, Z, νx, Vy, νz</td></tr><tr><td>Sequence length / stride</td><td>86 pts / 10 pts</td></tr><tr><td>Total 86-point windows</td><td>1,349,388</td></tr><tr><td>Training windows</td><td>1,149,245</td></tr><tr><td>Validation windows</td><td>137,127</td></tr><tr><td>Test windows</td><td>63,016</td></tr><tr><td>Intra-segment ∆t: mean / median / std</td><td> $3 . 0 0 \mathrm { s } / 2 . 6 0 \mathrm { s } / 2 . 3 6 \mathrm { s }$ </td></tr></table>

Rather than resampling to a fixed grid, we expose the native, irregular ADS-B timing to the network directly via a learned time-delta embedding; full preprocessing details are given in Appendix B.

## III. Evaluation

Predicting distributions over aircraft trajectories has until recently received moderate attention in the ATM literature [14], though generative formulations are now emerging [6]. To account for the inherently multi-modal aspect of trajectories, encompassing the different acceptable maneuvers at some given point, this distributional point of view is necessary and is hard to measure in practice. Building upon Salzmann et al.'s Trajectron++[4], we chose to evaluate our models on best of K for Average/Final Displacement Error (ADE/FDE) and Kernel Density Estimation Negative Log Likelihood (KDE-NLL).

## A. Average and Final Displacement Error

Let $\mathbf { p } _ { n } = ( x _ { n } , y _ { n } ) \in \mathbb { R } ^ { 2 }$ denote the horizontal Cartesian position at future step n, extracted from $\mathbf { X } ^ { \mathrm { f u t } }$ . Because the variance on the vertical axis is secondary to horizontal variations, we chose to compute $z _ { n }$ separately. Average displacement error accounts for how close the sampled trajectory is from the ground truth.

$$
\mathrm { A D E } = \frac { 1 } { T _ { \mathrm { f u t } } } \sum _ { n } \| \hat { \mathbf { p } } _ { n } - \mathbf { p } _ { n } \| _ { 2 }\tag{1}
$$

Note that this definition requires both predicted and ground-truth trajectories to share the same timestamps at each step n. In decimation experiments where the input sequence is downsampled, timestamp correspondence is preserved naturally. However, when predictions are requested at timestamps absent from the ground truth, the ground-truth positions ${ \bf p } _ { n }$ are obtained by linear interpolation to the desired evaluation points. To assess how far predictions drift at a fixed horizon, we report the Final Displacement Error, defined as the Euclidean distance between the predicted and ground-truth positions at the last future step $T _ { \mathrm { f u t } }$

$$
\mathrm { F D E } = \Vert \hat { \mathbf { p } } _ { T _ { \mathrm { { t u t } } } } - \mathbf { p } _ { T _ { \mathrm { { f u t } } } } \Vert _ { 2 }\tag{2}
$$

Because of the irregular ADS-B sampling, the final token is on average 128 s in the future, with the central 80% of windows spanning [95 s, 165 s], which equates to approximately one and a half to three minutes ahead.

## B. Best of K

ADE and FDE alone are well suited for deterministic trajectory forecasting, where a single predicted trajectory is compared against the ground truth. In our setting, however, the future is genuinely multimodal: given the same observed prefix, an aircraft may initiate a left or right turn, continue en route, or enter a holding pattern, all equally valid outcomes invisible from position data alone. A deterministic metric would penalize any model that hedges across modes, even if one of its samples matches the ground truth closely. We therefore adopt the best-of-K variants, minADE@ K and minFDE@K, standard in the multimodal forecasting literature [4, 5]:

$$
\operatorname* { m i n A D E } \mathcal { Q } K = \operatorname* { m i n } _ { k } \frac { 1 } { T _ { \mathrm { f u t } } } \sum _ { n } \| \hat { \mathbf { p } } _ { n } ^ { ( k ) } - \mathbf { p } _ { n } \| _ { 2 } , \quad \operatorname* { m i n F D E } \mathcal { Q } K = \operatorname* { m i n } _ { k } \| \hat { \mathbf { p } } _ { T _ { \mathrm { f u t } } } ^ { ( k ) } - \mathbf { p } _ { T _ { \mathrm { f u t } } } \| _ { 2 } .\tag{3}
$$

These metrics reward sample coverage: a model with good coverage places at least one sample close to the ground truth, and needs fewer samples to do so. We report $K \in \{ 1 , 5 , 2 0 \}$ , which quantifies how quickly additional samples improve coverage.

## C. Density Calibration

Best-of-K displacement errors reward coverage, whether at least one sample lands near the ground truth, but say nothing about how the remaining probability mass is distributed [18]. A model that places one sample on target and scatters the other $K - 1$ arbitrarily attains the same minADE@K as one whose entire sample cloud tightly brackets the truth, yet only the latter yields a density usable for conflict detection (Fig. 2). To assess the quality of the full predicted distribution, we adopt the kernel-density negative log-likelihood (KDE-NLL), a widely used distributional evaluation metric in multimodal trajectory forecasting [4].

At a future step n we draw $K = 5 0$ independent completions and retain their horizontal positions $\{ \hat { \mathbf { p } } _ { n } ^ { ( k ) } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { 2 }$ We fit a Gaussian kernel density estimate with Scott's-rule bandwidth,

$$
\hat { f } _ { n } ( \mathbf { p } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } N \big ( \mathbf { p } ; \hat { \mathbf { p } } _ { n } ^ { ( k ) } , h _ { n } ^ { 2 } \hat { \Sigma } _ { n } \big ) , \qquad h _ { n } = K ^ { - 1 / ( d + 4 ) } = K ^ { - 1 / 6 } ,\tag{4}
$$

(a) Sharp & calibrated  
![](images/da632cf5e769a09bb9044014243e4ed813a658f3e6975c73b47292f234019773.jpg)  
(c) Sharp & overconfident

(b) Multimodal & calibrated  
![](images/12165b0c62b7d0ec57c4af00a05806200ced581fa44e6f4821d5b7c80196e8ea.jpg)  
(d) Hyper-diffuse (always covers, low likelihood)

![](images/1e5a86c6679250092c51563c42efd41bca0e16fddd61de98fef2de6706481c56.jpg)

![](images/b124975ae501e55060669476ac189e928f94b25a363dc7c1932190190b6b9199.jpg)  
Fig. 2 Four synthetic K=20 sample-cloud scenarios, with the actual minFDE@20 and KDE NLL computed via Eq. (4). (a) Sharp and calibrated: both metrics agree. (b) Multimodal and calibrated: two equally plausible maneuvers; the bimodal density scores well against either outcome $( { \bf N L L } _ { A } { = } 1 5 . 5 , { \bf N L L } _ { B } { = } 1 4 . 9$ nats), (c) Sharp but displaced: over-confidence makes the NLL diverge even though the samples are tightly clustered. (d) Hyper-diffuse: spreading the samples widely keeps minFDE@20 comparable to (b), yet the likelihood assigned to the true outcome is markedly lower everywhere.

where $d = 2 , { \hat { \Sigma } } _ { n }$ is the empirical covariance of the K samples, and $h _ { n }$ is the Scott factor.\* We then report the negative log-likelihood of the ground-truth position under this density,

$$
\mathrm { N L L } @ n = - \log \hat { f } _ { n } ( \mathbf { p } _ { n } ) ,\tag{5}
$$

averaged over the test set (lower is better).

The log score is strictly proper [19]. The finite-K kernel estimate used here approximates it, with a Scott bandwidth and a floor, so we treat it as an approximate log-density score that jointly reflects sharpness and calibration rather than as an exactly proper rule. It rewards sharpness (concentrating mass), but penalizes over-confidence, since a tight cluster that excludes the ground truth drives ${ \hat { f } } _ { n } ( \mathbf { p } _ { n } ) \to 0$ and the score diverges. It thus captures exactly what best-of-K misses: whether the model assigns calibrated probability to where the aircraft actually goes. We evaluate NLL at the 10th, 20th, and 43rd future steps to track calibration as uncertainty accumulates over the horizon; as with the displacement metrics, the vertical axis is handled separately and the density is estimated in the horizontal plane.

## IV. Baseline Architectures

We compare against three baselines spanning the kinematic, deterministic-neural, and probabilistic-generative families. All baselines consume the same 43-point observed prefix and predict the same 43-point future suffix, enabling a like-for-like comparison.

## A. Constant Velocity

The constant-velocity (CV) model propagates the last observed state forward at constant velocity. Let $t _ { n }$ denote the (irregular) timestamp of the sample at sequence index n, and let $T _ { \mathrm { o b s } }$ index the last observed sample. The predicted position at sequence index $T _ { \mathrm { o b s } } + k$ is

$$
\hat { \mathbf { p } } _ { T _ { \mathrm { o b s } } + k } = \mathbf { p } _ { T _ { \mathrm { o b s } } } + \left( t _ { T _ { \mathrm { o b s } } + k } - t _ { T _ { \mathrm { o b s } } } \right) \mathbf { v } _ { T _ { \mathrm { o b s } } } , \qquad k = 1 , \dots , T _ { \mathrm { f u t } } ,\tag{6}
$$

where $\mathbf { p } _ { T _ { \mathrm { { o b s } } } }$ and ${ \bf v } _ { T _ { \mathrm { o b s } } }$ are the last observed position and velocity. The elapsed time $t _ { T _ { \mathrm { 0 b s } } + k } - t _ { T _ { \mathrm { 0 b s } } }$ is read from the target timestamps and therefore accounts for the irregular ADS-B sampling, while the position index $T _ { \mathrm { o b s } }$ + k remains a discrete sequence index. CV requires no training, is interpretable, and is the natural lower bound that any structure-aware model must beat. Being deterministic, its best-of-K metrics are constant in K.

## B. Deterministic LSTM Encoder-GRU Decoder

The deterministic neural baseline is a recurrent encoder-decoder. An LSTM encoder ingests the 43 observed tokens and produces a context vector; a GRU decoder then autoregressively rolls out the 43 future states. At each decoder step, the elapsed time since the start of the observation window is concatenated to the decoder input, letting the network condition its rollout on the irregular ADS-B sampling rather than assuming a fixed step. The model is trained with mean-squared error on the future per-step displacements (deltas), integrated at inference to recover absolute Cartesian positions. Because the output is a single trajectory, the model is deterministic and its @K metrics again collapse to the K = 1 values.

## C. Conditional Variational Autoencoder (Trajectron++)

The strongest non-diffusion baseline is a Conditional Variational Autoencoder in the style of Trajectron++ [4], a widely used probabilistic approach to multimodal trajectory forecasting. A recurrent encoder summarizes the observed past into a conditioning vector; a latent variable captures the discrete and continuous modes of the future (e.g. turn vs. straight); and a recurrent decoder generates a future trajectory conditioned jointly on the past and a latent sample. Drawing K independent latent samples yields K diverse trajectories, so, unlike CV and the deterministic LSTM, the CVAE supports genuine best-of-K evaluation and density estimation, and serves as our probabilistic non-diffusion reference.

## V. FlowATC: Generative Inpainting with Flow Matching

## A. Problem Formulation

Let $( \mathbf { X } ^ { \mathrm { o b s } } , \mathbf { X } ^ { \mathrm { f u t } } ) \sim p _ { \mathrm { d a t a } }$ denote a pair of observed prefix and future suffix drawn from the (unknown) joint distribution of Bay Area traffic, with $\mathbf { X } ^ { \mathrm { o b s } } \in \mathbb { R } ^ { T _ { \mathrm { o b s } } \times F }$ and $\mathbf { X } ^ { \mathrm { f u t } } \in \mathbb { R } ^ { T _ { \mathrm { f u t } } \times F }$ . Because runway assignment, controller instructions, and traffic sequencing are not observable in $\mathbf { X } ^ { \mathrm { o b s } }$ , the conditional law $p ( \mathbf { X } ^ { \mathrm { f u t } } \mid \mathbf { X } ^ { \mathrm { o b s } } )$ is in general multimodal: several distinct futures carry non-negligible probability mass. The object we seek is therefore not a point estimate but a sampler for this conditional distribution, that is, a mechanism producing $\hat { \mathbf { X } } ^ { \mathrm { f u t } } \sim p ( \cdot \mid \mathbf { X } ^ { \mathrm { o b s } } )$ , from which any downstream quantity (conflict probability, occupancy density, best of K forecasts) can be estimated by Monte Carlo.

This requirement is not merely a preference: any deterministic predictor $f _ { \theta }$ trained with mean squared error converges, at the population optimum, to the conditional mean,

$$
\arg \operatorname* { m i n } _ { f } \ \mathbb { E } _ { ( \mathbf { X } ^ { \mathrm { o b s } } , \mathbf { X } ^ { \mathrm { f u t } } ) } { \left\| f ( \mathbf { X } ^ { \mathrm { o b s } } ) - \mathbf { X } ^ { \mathrm { f u t } } \right\| } _ { 2 } ^ { 2 } \ = \ \mathbb { E } { \left[ \mathbf { X } ^ { \mathrm { f u t } } \ \vert \ \mathbf { X } ^ { \mathrm { o b s } } \right] } ,\tag{7}
$$

by the standard $L ^ { 2 }$ projection property of conditional expectation. When $p ( \mathbf { X } ^ { \mathrm { f u t } } \mid \mathbf { X } ^ { \mathrm { o b s } } )$ has two modes, say a left turn and a continued straight leg, their average is a trajectory that belongs to neither mode and may be physically implausible The deterministic baselines of Section IV are thus limited by construction, independently of their capacity: they solve a different (and, under multimodality, ill-suited) problem.

Conditional Flow Matching sidesteps this by regressing a velocity field $u _ { \theta }$ rather than a trajectory. Fix $\mathbf { X } ^ { \mathrm { o b s } }$ , draw $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $\mathbf { x } _ { 1 } = \mathbf { X } ^ { \mathrm { f u t } } \sim p ( \cdot \mid \mathbf { X } ^ { \mathrm { o b s } } )$ , and define the linear interpolant ${ \bf x } _ { t } = \left( 1 - t \right) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ . The population CFM objective regresses $u _ { \theta } ( \mathbf { x } _ { t } , t \mid \mathbf { X } ^ { \mathrm { o b s } } )$ onto the pair conditional velocity $\left( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \right)$ . Because the squared loss is minimized pointwise by a conditional expectation, its unique population minimizer is the marginal velocity field

$$
u ^ { \star } ( { \bf x } , t \mid { \bf X } ^ { \mathrm { o b s } } ) = \mathbb { E } \big [ { \bf x } _ { 1 } - { \bf x } _ { 0 } \big | { \bf x } _ { t } = { \bf x } , { \bf X } ^ { \mathrm { o b s } } \big ] ,\tag{8}
$$

and it is a standard result of the flow matching literature [9, 20] that the probability flow of this field, namely the solution of $\dot { { \mathbf x } } = u ^ { \star } ( { \mathbf x } , t \mid { \mathbf X } ^ { \mathrm { o b s } } )$ initialized at $\mathbf { x } ( 0 ) \sim N ( \mathbf { 0 } , \mathbf { I } )$ , has marginal law exactly $p ( \cdot \mid \mathbf { X } ^ { \mathrm { o b s } } )$ at $t = 1$ . In other words, exactly minimizing the CFM loss and exactly integrating the learned field is equivalent to sampling from the true conditional distribution of futures. Multimodality is preserved automatically: distinct noise draws $\mathbf { X } _ { 0 }$ are transported to distinct modes, and no averaging across modes ever occurs.

## B. The DiT Backbone

Following the Diffusion Transformer (DiT) framework [11], the denoiser is a stack of Transformer blocks with the generation step (flow time $t \in [ 0 , 1 ] ,$ injected into every block through Adaptive Layer Normalization (AdaLN). Let h $\epsilon \mathbb { R } ^ { d }$ denote a token's hidden representation entering a sub-layer, with $\mu ( \mathbf { h } )$ and $\sigma ( \mathbf { h } )$ its mean and standard deviation taken over the feature dimension d (per-token normalization). A small MLP maps t to scale and shift parameters $\gamma ( t ) , \beta ( t ) \in \mathbb { R } ^ { d }$ , applied via the same AdaLN mechanism at each of the two sub-layers (self-attention and feed-forward) through independently learned MLP heads:

$$
\operatorname { A d a L N } ( \mathbf { h } , t ) = \gamma ( t ) \cdot { \frac { \mathbf { h } - \mu ( \mathbf { h } ) } { \sigma ( \mathbf { h } ) } } + \beta ( t ) .\tag{9}
$$

The core block operation is multi-head self-attention; for tokens $\mathbf { Z } \in \mathbb { R } ^ { T \times d }$ , split into H heads of dimension $d _ { k } = d / H$

$$
\mathrm { A t t n } ( { \bf Z } ) = \mathrm { s o f t m a x } \left( { \frac { { \bf Q } { \bf K } ^ { \top } } { \sqrt { d _ { k } } } } \right) { \bf V } , \quad { \bf Q } , { \bf K } , { \bf V } = { \bf Z } W _ { Q } , { \bf Z } W _ { K } , { \bf Z } W _ { V } ,\tag{10}
$$

Attention is block-causal: observed tokens are prevented from attending to the noisy future tokens, while future tokens attend freely to the observed prefix and to one another. Writing $\mathbf { A } \in \{ 0 , - \infty \} ^ { T \times T }$ for the additive mask applied to the attention logits before the softmax, $A _ { i j } = - \infty$ iff $i < T _ { \mathrm { o b s } }$ and $j \ge T _ { \mathrm { o b s } }$ , and 0 otherwise. The observed representation is therefore independent of the noise realisation, while the future block keeps the full bidirectional attention motivated above, since all its tokens share the same noise level. The future timestamps supplied through $t _ { \mathrm { r e l } }$ are query times: they state when a prediction is requested and carry no information about the aircraft's future state.

Figure 3 shows the full architecture.

## C. Trajectory as an Image: Inpainting the Future

We represent each 86-step window as a $6 \times 8 6$ feature-time matrix, in effect a one-channel “image" of the trajectory. The first $T _ { \mathrm { o b s } } = 4 3$ tokens carry the observed (clean) state vectors; the last $T _ { \mathrm { f u t } } = 4 3$ tokens are initialized with Gaussian noise and treated as the masked region to be inpainted (Fig. 4).

Both the $T _ { \mathrm { o b s } }$ clean observation tokens and the $T _ { \mathrm { f u t } }$ noisy future tokens are embedded by a shared linear projection

![](images/6e1dc39ef8c2504e77fabf33b216f3b17e814e39a7bf08c2cea5c7201f4f6bce.jpg)  
Fig. 3 FlowATC DiT[11] architecture overview.

and concatenated into a single sequence of length $T = 8 6$

$$
{ \bf Z } = \left[ W _ { \mathrm { i n } } { \bf X } ^ { \mathrm { o b s } } \parallel W _ { \mathrm { i n } } \tilde { \bf X } ^ { \mathrm { f u t } } \right] \in \mathbb { R } ^ { T \times d } .\tag{11}
$$

The observed past is thus injected purely by concatenation: no separate encoder is needed. Because the past enters only as additional tokens, the architecture is a natural substrate for further conditioning: any extra signal (ATC voice, weather fields, charts) can be appended as context tokens without changing the loss or the inpainting mechanism. Because all future tokens are denoised at the same noise level t, there is no temporal ordering to preserve within the future block, so attending freely helps the model produce spatially consistent trajectories. Only the $T _ { \mathrm { f u t } }$ future positions are read off for the loss or the next integration step.

![](images/417a7f108f64df978eac8762933bce6e79bd80bd62b53c6a4b9b1b8836242dc4.jpg)  
Fig. 4 Trajectory represented as a $6 \times 8 6$ feature-time matrix. The observed prefix (solid, left of dashed line) provides context tokens; the noisy future suffix (light, right) is the region to be inpainted by the flow-matching model.

In addition to the AdaLN diffusion-time conditioning, each token receives a sinusoidal time embedding of its prediction date: the time offset of that token within the window which is added to the token representation. This lets the model distinguish near-future from far future positions even though all future tokens are denoised simultaneously.

The velocity field $u _ { \theta } ( { \bf x } , t )$ is trained on the straight-line interpolation between Gaussian noise and the target future:

$$
\mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } _ { t , \mathbf { x } _ { 0 } , \mathbf { X } ^ { \mathrm { f u t } } } \Big \| u _ { \theta } \big ( ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { X } ^ { \mathrm { f u t } } , ~ t ~ | ~ \mathbf { X } ^ { \mathrm { o b s } } \big ) - ( \mathbf { X } ^ { \mathrm { f u t } } - \mathbf { x } _ { 0 } ) \Big \| _ { 2 } ^ { 2 } ,\tag{12}
$$

with $\mathbf { x } _ { 0 } \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } )$ and t ∼ LogitNormal(1.0, 1.0). At inference, Euler integration from t = 0 to t = 1 in 20 steps produces one trajectory sample; repeating with independent noise draws yields K diverse completions. We refer to this flow-matching-trained model as FlowATC. The same backbone is also trained with a DDPM objective [7] (1000 forward steps, cosine schedule, DDIM sampling) as a baseline.

All CFM and DDPM variants share a Transformer backbone (multi-head self-attention, feed-forward factor 4, dropout 0.1) with the layer/head/width counts of Table 2. Training uses AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , weight decay $1 0 ^ { - 2 }$ , learning rate $1 0 ^ { - 4 } )$ with a 1000-step linear warm-up, batch size 512, for up to 200 epochs on a single GPU. Exponential Moving Average weights (decay 0.9999) are used at inference.

Latency is the wall-clock time to answer one request: a single input sequence (batch size 1) for which K=20 samples are drawn in parallel, using 20 Euler steps for flow models and 100 DDIM steps for diffusion models. It is measured in float32 on a single NVIDIA RTX 4000 Ada, averaged over at least 30 requests after 10 warm-up requests.

Table 2 Flow model architecture configurations.
<table><tr><td>Model</td><td>Layers</td><td>Heads</td><td>d</td><td>Parameters</td></tr><tr><td>flow_tiny</td><td>5</td><td>4</td><td>128</td><td>1.5M</td></tr><tr><td>flow_small</td><td>6</td><td>8</td><td>256</td><td>7.1M</td></tr><tr><td>flow_large</td><td>8</td><td>8</td><td>384</td><td>20.7M</td></tr></table>

## VI. Results

Drawing K completions and coloring each by its final-point KDE density yields a fan of plausible futures (Fig. 5); the corresponding final-point density forms spatial probabilistic occupancy estimates that can serve as an input to downstream conflict-risk estimation (Fig. 6). For straight en-route segments the density is narrow and elongated along the track; for aircraft entering an approach turn it broadens laterally, capturing genuine uncertainty about the turn-initiation point. In all cases the high-density region brackets the ground-truth final position.

![](images/8ce7e555af619b96357a20c53c6d87bc74e331f9a741d356d1d5fae644021a96.jpg)  
Fig. 5 Predicted trajectory fan (K = 20 samples). Blue: observed past; Green: ground-truth future; purpleyellow: predicted final-point density (purple = low, yellow = high).

![](images/b1a4c96437224a94ba1576ab2543d274607326637710bdbffbf1b6ece3a9b640.jpg)  
(a) Straight en-route

![](images/f22a8ddcfd31e5f957b259b4e61651bedfdafbab4d4737e5b418025d6bf199f4.jpg)  
(b) Turn

![](images/fb52acf06afe4b8637f05db77f8b90c65553c15472a5143a15278ded0d82dd2f.jpg)  
(c) Holding entry  
Fig. 6 Final-position probability density maps (K = 20 samples, Gaussian KDE, Scott bandwidth). Dashed blue: observed past; dashed green: ground-truth future; solid heatmap: sampled trajectories; green star: ground-truth final position; yellow-red: predicted final-point density (light = low, dark = high). Density width correlates with the complexity of the anticipated maneuver.

## A. Model Comparison

Table 3 compares all five model variants at matched capacity (1.5 M parameters) on the held-out test set. LSTM-Det improves over CV mainly on FDE (2748.5 m vs. 3480.2 m, -21%), reflecting that a learned decoder curves its rollout toward typical approach geometry rather than extrapolating a straight line; the ADE gain is comparatively modest (1305.3 m vs. 1343.6 m), since most of the 43-step horizon is still well approximated by locally linear motion. At K = 1, the generative models do not outperform the deterministic baselines outright: Diffusion Tiny (1370.6/3183.3) is in fact the worst model in the table on minADE@ 1, and Flow Tiny (1313.5/3147.7) is comparable to CV. Panel (a) of Fig. 7 shows this is a genuine crossover in horizon, not just an artefact of the final step: CV, LSTM, Diffusion, and Flow all start near zero displacement and grow with n, while CVAE starts at \~500 m even at n=1 and grows far more slowly, overtaking the other four models only around n ≈ 15–20. This is expected rather than a failure mode: a single diffusion or flow draw is one stochastic sample from the predicted distribution, not a mean or mode estimate, so it inherits the full spread of the multimodal posterior from the very first future step. CVAE's early offset has a different origin, isolated in Appendix D: it is decoder output variance rather than a mispredicted mean, and the model cannot shed this noise for free as doing so collapses minADE@20 instead, since its discrete latent supplies little diversity conditional on a single observed history. The generative models only become competitive once they are allowed to hedge across samples.

Table 3 Model comparison at equal capacity (1.5 M parameters). 43 obs → 43 fut steps (≈ 128 s horizon), $N = 6 3 { , } 0 1 6$ test trajectories. Distances in metres ± SEM. KDE NLL in nats (lower is better for all columns). Bold = best value per column.
<table><tr><td>Model</td><td> $\mathrm { T y p e }$ </td><td>Params ms/pred</td><td></td><td colspan="2"> $K = 1$ </td><td colspan="2"> $K = 5$ </td><td colspan="2"> $K = 2 0$ </td><td colspan="3">KDE NLL@n</td></tr><tr><td></td><td></td><td></td><td></td><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td> $n { = } 1 0$ </td><td> $n { = } 2 0$ </td><td> $n { = } 4 3$ </td></tr><tr><td>CV</td><td></td><td></td><td>0.0</td><td> $1 3 4 3 . 6 _ { \pm 8 . 1 }$ </td><td> $3 4 8 0 . 2 _ { \pm 1 9 . 3 }$ </td><td> $1 3 4 3 . 6 _ { \pm 8 . 1 }$ </td><td> $3 4 8 0 . 2 _ { \pm 1 9 . 3 }$ </td><td> $1 3 4 3 . 6 _ { \pm 8 . 1 }$ </td><td> $3 4 8 0 . 2 _ { \pm 1 9 . 3 }$ </td><td></td><td></td><td></td></tr><tr><td>LSTM-Det  $\left( + t _ { \mathrm { r e l } } \right)$ </td><td></td><td>1.5M</td><td>1.15</td><td> $1 3 0 5 . 3 _ { \pm 5 . 7 }$ </td><td> $2 7 4 8 . 5 _ { \pm 1 2 . 6 }$ </td><td> $1 3 0 5 . 3 _ { \pm 5 . 7 }$ </td><td> $2 7 4 8 . 5 _ { \pm 1 2 . 6 }$ </td><td> $1 3 0 5 . 3 _ { \pm 5 . 7 }$ </td><td> $2 7 4 8 . 5 _ { \pm 1 2 . 6 }$ </td><td></td><td></td><td></td></tr><tr><td>CVAE (Trajectron++ style)</td><td>CVAE</td><td>1.5M</td><td>4.06</td><td> $\mathbf { 1 2 1 9 . 7 _ { \pm 6 . 0 } }$ </td><td> $2 3 1 6 . 2 _ { \pm 1 2 . 8 }$ </td><td> $8 2 8 . 5 _ { \pm 4 . 3 }$ </td><td> $1 6 0 8 . 8 _ { \pm 1 0 . 2 }$ </td><td> $6 6 2 . 9 _ { \pm 3 . 8 }$ </td><td> $1 2 4 0 . 9 _ { \pm 9 . 1 }$ </td><td> $1 4 . 6 _ { \pm 0 . 0 }$ </td><td> $1 6 . 9 _ { \pm 0 . 1 }$ </td><td> $3 1 . 7 _ { \pm 0 . 3 }$ </td></tr><tr><td>Diffusion Tiny</td><td>DDPM</td><td>1.5M</td><td>119.7</td><td> $1 3 7 0 . 6 _ { \pm 7 . 5 }$ </td><td> $3 1 8 3 . 3 _ { \pm 1 7 . 6 }$ </td><td> $6 9 7 . 1 _ { \pm 4 . 2 }$ </td><td> $1 4 6 7 . 5 _ { \pm 9 . 3 }$ </td><td> $4 5 9 . 9 _ { \pm 2 . 9 }$ </td><td> $8 2 8 . 1 _ { \pm 6 . 1 }$ </td><td> $1 3 . 7 _ { \pm 0 . 0 }$ </td><td> $1 6 . 2 _ { \pm 0 . 1 }$ </td><td> $1 8 . 9 _ { \pm 0 . 1 }$ </td></tr><tr><td>Flow Tiny</td><td>CFM</td><td>1.5M</td><td>23.8</td><td> $1 3 1 3 . 5 _ { \pm 7 . 5 }$ </td><td> $3 1 4 7 . 7 _ { \pm 1 8 . 2 }$ </td><td> ${ \bf 6 1 4 . 5 _ { \pm 3 . 8 } }$ </td><td> $\mathbf { 1 3 0 2 . 7 _ { \pm 8 . 7 } }$ </td><td> ${ \bf 3 9 0 . 5 _ { \pm 2 . 4 } }$ </td><td> ${ \bf 6 8 6 . 1 _ { \pm 5 . 0 } }$ </td><td> ${ \bf 1 3 . 4 _ { \pm 0 . 1 } }$ </td><td> $1 5 . 7 _ { \pm 0 . 1 }$ </td><td> ${ \bf 1 8 . 1 _ { \pm 0 . 2 } }$ </td></tr></table>

The picture reverses sharply by $K = 2 0 !$ Flow Tiny (390.5/686.1) and Diffusion Tiny (459.9/828.1) both surpass CVAE (662.9/1240.9) by a wide margin, 41% and 31% lower minADE@20 respectively. Panel (b) of Fig. 7 shows this advantage holds across the full horizon, not just at n=43: flow and diffusion separate from CVAE as early as $K { = } 5$ (Appendix E), and the gap widens monotonically with n through K=20. This indicates that the diffusion/flow sample cloud covers the true multimodal distribution more efficiently than the CVAE's discrete-latent mixture; fewer samples are wasted on implausible modes, so additional draws pay off faster at every horizon, not only at the end of the window.

The NLL@n columns tell a complementary story about distributional quality rather than best-case coverage. All three generative models are similar at the 10-step horizon (13.4–14.6 nats), but diverge sharply by the 43-step horizon, as panel (c) of Fig. 7 makes clear: CVAE's calibration curve bends upward steeply after $n \approx 2 0$ , while Diffusion Tiny and Flow Tiny remain nearly flat over the same range. CVAE's NLL@43 (31.7) corresponds to a kernel density at the realized endpoint roughly $e ^ { 1 3 . 6 } \approx 8 \times 1 0 ^ { 5 }$ times smaller than Flow Tiny's (18.1), meaning that even though CVAE's best-of-20 samples can land close to the ground truth, its full predicted density is comparatively poorly calibrated at long horizons.

Comparing the two generative objectives directly, CFM (Flow Tiny) beats DDPM (Diffusion Tiny) on every column: 15% lower minADE@20 (390.5 vs. 459.9), 17% lower minFDE@20, and consistently lower NLL at all three horizons. In this matched-capacity comparison CFM consistently outperforms DDPM; since capacity is held fixed, the gap points to the training objective rather than to model size, though we report single runs and do not quantify seed variance, a point we revisit at scale in Section VI.B.

Table 4 Vertical error at matched capacity (1.5 M parameters) on the held-out test set, in metres, mean ± SEM over 63,016 windows. The altitude axis is evaluated separately from the horizontal metrics of Table $_ { 3 ; }$ the deterministic baselines are constant in K.
<table><tr><td rowspan="2">Model</td><td colspan="2"> $\mathrm { m i n A D E } _ { z }$ </td><td colspan="2"> $\mathrm { m i n F D E } _ { z }$ </td></tr><tr><td> $K { = } 1$ </td><td> $K { = } 2 0$ </td><td> $K { = } 1$ </td><td> $K { = } 2 0$ </td></tr><tr><td>CV</td><td> $1 0 9 . 7 _ { \pm 0 . 7 }$ </td><td> $1 0 9 . 7 _ { \pm 0 . 7 }$ </td><td> $2 4 8 . 7 _ { \pm 1 . 5 }$ </td><td> $2 4 8 . 7 _ { \pm 1 . 5 }$ </td></tr><tr><td>LSTM-Det</td><td> $7 8 . 4 _ { \pm 0 . 4 }$ </td><td> $7 8 . 4 _ { \pm 0 . 4 }$ </td><td> $\mathbf { 1 5 1 . 9 _ { \pm 0 . 9 } }$ </td><td> $1 5 1 . 9 _ { \pm 0 . 9 }$ </td></tr><tr><td>CVAE</td><td> $9 4 . 6 _ { \pm 0 . 6 }$ </td><td> $4 3 . 8 _ { \pm 0 . 3 }$ </td><td> $1 6 8 . 6 _ { \pm 1 . 2 }$ </td><td> $6 2 . 0 _ { \pm 0 . 7 }$ </td></tr><tr><td>Diffusion Tiny</td><td> $1 0 2 . 6 _ { \pm 0 . 5 }$ </td><td> $2 8 . 3 _ { \pm 0 . 2 }$ </td><td> $1 9 8 . 9 _ { \pm 1 . 1 }$ </td><td> $2 9 . 9 _ { \pm 0 . 4 }$ </td></tr><tr><td>Flow Tiny</td><td> $9 4 . 2 _ { \pm 0 . 5 }$ </td><td> $2 5 . 0 _ { \pm 0 . 2 }$ </td><td> $1 8 6 . 8 _ { \pm 1 . 1 }$ </td><td> $2 4 . 3 _ { \pm 0 . 3 }$ </td></tr></table>

![](images/565cde8da10af8f6547870f3a4e2160d2f41f1fc0e0f605fcf25138d1d2ca712.jpg)

![](images/0ba071d0755b05e050421138cb21afdb77f1fe2e6deea45ebdd2b10dea0320c3.jpg)

![](images/398d9229c70b1eac3e304157eb39ad0930a667090e148d7344a3baf4a5a971d8.jpg)  
Fig. 7 Prediction error and calibration across sample budget and future horizon. (a)–(b) Minimum displacement error at $K \in \{ 1 , 2 0 \}$ . (c) Density calibration for the two stochastic generative models (CV and the deterministic LSTM are excluded: their point estimate has no associated density). The full $K \in { 1 , 5 , 2 0 }$ sweep is given in Appendix E, Fig. 17.

Table 4 reports the vertical axis, evaluated separately. It reproduces the horizontal narrative rather than adding a new one. At K=1 the deterministic LSTM is the most accurate model $( 7 8 . 4 _ { \pm 0 . 4 }$ m against $9 4 . 2 _ { \pm 0 . 5 }$ m for Flow Tiny), since a single stochastic draw again inherits the spread of the predicted distribution rather than estimating its mean. By $K { = } 2 0$ the ordering reverses: Flow Tiny reaches $2 5 . 0 _ { \pm 0 . 2 }$ m against $2 8 . 3 _ { \pm 0 . 2 }$ m for Diffusion Tiny and $4 3 . 8 _ { \pm 0 . 3 }$ m for the CVAE, a 43% reduction over the latter, close to the 41% measured horizontally. Altitude at a two-minute horizon is dominated by the discrete choice to level off, climb or descend, precisely the kind of branching a single point estimate cannot represent.

## B. Scaling

Table 5 reports both CFM and DDPM across three model sizes. FlowATC's minADE@20 improves sharply from Tiny to Small $( 3 9 0 . 5  3 0 6 . 1 \mathrm { m } , - 2 1 . 6 \% )$ but plateaus from Small to Large $( 3 0 6 . 1  3 0 7 . 4 \mathrm { m } , + 0 . 4 \% )$ , and minFDE@20 shows the same pattern (568.7 → 576.8 m). This suggests CFM saturates the information available in the 12-day dataset by around 7 M parameters, approaching a data-limited rather than capacity-limited regime. The saturation has an operational corollary: Flow Small reaches the same minADE@20 as Flow Large (306.1 vs 307.4 m) at 2.3× lower latency (67.0 vs 157.1 ms per request, Table 6), so capacity beyond 7.1 M buys nothing but cost on this dataset.

Table 5 Flow Matching vs. Diffusion scaling. 43 obs → 43 fut steps (≈ 128 s horizon), $N = 6 3 { , } 0 1 6$ test trajectories. Distances in metres ± SEM. KDE NLL in nats. Bold = best within each architecture family.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Type</td><td rowspan="2">Params</td><td colspan="2"> $K = 1$ </td><td colspan="2"> $K = 5$ </td><td colspan="2"> $K = 2 0$ </td><td colspan="3">KDE NLL@n</td></tr><tr><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td> $n { = } 1 0$ </td><td> $\scriptstyle n = 2 0$ </td><td> $scriptstyle n = 4 3$ </td></tr><tr><td colspan="10">Flow Matching (CFM, 20 inference steps)</td><td></td><td></td><td></td></tr><tr><td>Flow Tiny</td><td>CFM</td><td>1.5M</td><td> $1 3 1 3 . 5 _ { \pm 7 . 5 }$ </td><td> $3 1 4 7 . 7 _ { \pm 1 8 . 2 }$ </td><td> $6 1 4 . 5 _ { \pm 3 . 8 }$ </td><td> $1 3 0 2 . 7 _ { \pm 8 . 7 }$ </td><td> $3 9 0 . 5 _ { \pm 2 . 4 }$ </td><td> $6 8 6 . 1 _ { \pm 5 . 0 }$ </td><td> $1 3 . 4 _ { \pm 0 . 1 }$ </td><td> $1 5 . 7 _ { \pm 0 . 1 }$ </td><td> ${ \bf 1 8 . 1 _ { \pm 0 . 2 } }$ </td></tr><tr><td>Flow Small</td><td>CFM</td><td>7.1M</td><td> $9 9 6 . 0 _ { \pm 6 . 5 }$ </td><td> $2 4 1 9 . 7 _ { \pm 1 5 . 8 }$ </td><td> $4 7 9 . 5 _ { \pm 3 . 4 }$ </td><td> $1 0 5 0 . 6 _ { \pm 7 . 9 }$ </td><td> $\mathbf { 3 0 6 . 1 } _ { \pm 2 . 2 }$ </td><td> ${ \bar { 5 } } 6 8 . 7 _ { \pm 4 . 7 }$ </td><td> $1 3 . 3 _ { \pm 0 . 1 }$ </td><td> $1 7 . 4 _ { \pm 0 . 6 }$ </td><td> $2 0 . 2 _ { \pm 0 . 5 }$ </td></tr><tr><td>Flow Large</td><td>CFM</td><td>20.7M</td><td> ${ \bf 9 0 3 . 4 } _ { \pm 5 . 9 }$ </td><td> $2 1 5 6 . 8 _ { \pm 1 4 . 2 }$ </td><td> $\mathbf { 4 6 7 . 3 _ { \pm 3 . 4 } }$ </td><td> $\mathbf { 1 0 1 3 . 2 _ { \pm 7 . 8 } }$ </td><td> $3 0 7 . 4 _ { \pm 2 . 4 }$ </td><td> $5 7 6 . 8 _ { \pm 5 . 3 }$ </td><td> $1 3 . 3 _ { \pm 0 . 1 }$ </td><td> $1 7 . 5 _ { \pm 0 . 5 }$ </td><td> $2 0 . 2 _ { \pm 0 . 4 }$ </td></tr><tr><td colspan="10">Diffusion (DDPM, 100 inference steps)</td><td colspan="3"></td></tr><tr><td>Diffusion Tiny</td><td>DDPM</td><td>1.5M</td><td> $1 3 7 0 . 6 _ { \pm 7 . 5 }$ </td><td> $3 1 8 3 . 3 _ { \pm 1 7 . 6 }$ </td><td> $6 9 7 . 1 _ { \pm 4 . 2 }$ </td><td> $1 4 6 7 . 5 _ { \pm 9 . 3 }$ </td><td> $4 5 9 . 9 _ { \pm 2 . 9 }$ </td><td> $8 2 8 . 1 _ { \pm 6 . 1 }$ </td><td> $1 3 . 7 _ { \pm 0 . 0 }$ </td><td> ${ \bf 1 6 . 2 _ { \pm 0 . 1 } }$ </td><td> $\mathbf { 1 8 . 9 _ { \pm 0 . 1 } }$ </td></tr><tr><td>Diffusion Small</td><td>DDPM</td><td>7.1M</td><td> $1 1 7 3 . 8 _ { \pm 6 . 9 }$ </td><td> $2 8 3 5 . 5 _ { \pm 1 6 . 5 }$ </td><td> $6 2 0 . 2 _ { \pm 4 . 0 }$ </td><td> $1 3 8 5 . 8 _ { \pm 9 . 3 }$ </td><td> $4 1 2 . 7 _ { \pm 2 . 8 }$ </td><td> $8 0 5 . 9 _ { \pm 6 . 3 }$ </td><td> $1 4 . 2 _ { \pm 0 . 1 }$ </td><td> $1 9 . 1 _ { \pm 0 . 3 }$ </td><td> $2 2 . 6 _ { \pm 0 . 4 }$ </td></tr><tr><td>Diffusion Large</td><td>DDPM</td><td>20.7M</td><td> $\mathbf { 1 0 5 1 . 6 _ { \pm 6 . 3 } }$ </td><td> $2 4 7 8 . 6 _ { \pm 1 5 . 3 }$ </td><td> ${ \bf 5 2 9 . 8 _ { \pm 3 . 5 } }$ </td><td> $\mathbf { 1 1 5 0 . 3 _ { \pm 8 . 2 } }$ </td><td> $3 4 4 . 0 _ { \pm 2 . 4 }$ </td><td> $\mathbf { 6 5 0 . 3 _ { \pm 5 . 5 } }$ </td><td> ${ \bf 1 3 . 2 } _ { \pm 0 . 1 }$ </td><td> $1 6 . 8 _ { \pm 0 . 3 }$ </td><td> $2 0 . 9 _ { \pm 0 . 6 }$ </td></tr></table>

DDPM shows the opposite trend: minADE@20 improves steadily at every step (459.9 → 412.7 → 344.0 m, -10.3% then -16.6%), with its largest single gain occurring exactly where flow stalls. Because DDPM has a harder denoising objective, this is consistent with DDPM still being capacity-limited at 20.7 M parameters where CFM is not. Consequently, the relative advantage of CFM over DDPM on minADE@20 is not monotonic in model size: 15% at Tiny, widening to 26% at Small, then narrowing to 11% at Large as DDPM catches up (each measured as the reduction in minADE@20 relative to DDPM at matched capacity). At the sizes tested here, CFM remains strictly better at every scale, but DDPM's steeper scaling curve suggests the gap would continue to narrow, or close, at larger capacity than we evaluate.

The KDE NLL columns decouple from minADE/minFDE in an interesting way: FlowATC's NLL@43 is best at Tiny (18.1 nats) and degrades with scale (20.2 nats at both Small and Large), even as displacement error improves. Larger FlowATC models thus produce sample clouds that are more accurate on average but slightly less well calibrated at long horizons, a reminder that best-of-K accuracy and distributional calibration are related but distinct properties, and that scaling helps one without guaranteeing the other.

## C. Extrapolation to Future Timesteps

Because the prediction-date embedding conditions each future token on its time offset within the window rather than on a fixed step count, the model can be queried at prediction dates beyond the $T _ { \mathrm { f u t } } = 4 3$ horizon seen during training, simply by resampling the embedding at longer offsets without retraining or architectural changes. We probe this by overwriting the $t _ { \mathrm { r e l } }$ timing channel with virtual timestamps spanning target horizons up to 360 s (roughly three times the native ≈ 128 s horizon), stitching three shards $( H _ { \mathrm { m a x } } \in \{ 9 0 , 1 8 0 , 3 6 0 \}$ s) to keep good temporal resolution across the full range. Ground truth beyond the observation window is obtained by cubic-spline interpolation of the continuing raw ADS-B trajectory: a window contributes at a given horizon only if its flight actually extends that far, and horizons where fewer than half the windows qualify are dropped, a filter that does not bind here since 85% of windows still have a continuing track at 360 s. Appendix G gives the full protocol, including a visible estimator-seam artefact at the shard boundaries that reflects a change of estimator, not of model behavior.

![](images/e426dc343274ee05ebb5475739fd1f1d3cadf6027bc80ec29b60932ee5ab33e1.jpg)  
Fig. 8 Portion of test windows reaching a given future horizon.

Figure 9 reports minADE@K (panels a-b) and KDE NLL (panel c) as a function of elapsed time since the last observation, from 0 to 360 s, for CVAE, Diffusion Tiny, and Flow Tiny (all 1.5 M parameters). The gray shading reports a distinct quantity: the fraction of test windows whose native 43-token future span alone reaches that horizon, 96% at 90 s but 16% at 150 s and under 1% at 360 s (Fig. 8). Beyond roughly 150 s the comparison therefore rests on the interpolated continuation of each flight rather than on tokens the observation window itself contains.

The diffusion-based models degrade faster than CVAE at long horizons when only a single sample is drawn (K=1, panel a); once $K \geq 5$ (panel b shows K=20, the full K=5 curve is in Appendix G) Flow remains the most accurate model across the full horizon, and only CVAE's K=1 curve overtakes it beyond ≈ 200 s. This is an artefact of the models being queried at time offsets never seen during training: the future block still contains 43 tokens, but the prediction dates attached to them lie beyond the training horizon, whereas CVAE's autoregressive decoder rolls out to arbitrary length by construction.

Panel (c) shows why this apparent CVAE advantage is misleading. KDE-NLL, which tests whether a model's uncertainty keeps pace with its actual error independent of its size, plateaus at ≈20 nats for Diffusion and Flow from t ≈ 150 s onward. Their sample clouds widen in step with the true uncertainty even as displacement error keeps growing (panel b) while CVAE's NLL diverges to ≈ 60 nats by $t = 3 6 0 \mathrm { s } .$ CVAE's lower displacement error beyond ≈ 200 s therefore reflects a single sampled trajectory landing closer to the truth on average, not a predicted distribution that knows how uncertain it should be: exactly the failure mode best-of-K metrics cannot detect and KDE-NLL is designed to catch.

![](images/b380fd4763cb0c7265b47535f5f2a5e48ef11b6a08de01a7fcfd81ea9b6d2674.jpg)

![](images/963c4b60ddee96b51c00bc5619ae7e1cec00e2931064fff5cc3f4a764ff832be.jpg)

![](images/3a5bd30932c3effab8aea299798014e361b7c90f90d23db16073d953dd0161d7.jpg)  
Fig. 9 Extrapolation error and calibration beyond the training horizon. (a)–(b) minDisp@K for K ∈ {1, 20}. (c) KDE NLL. Gray shading (right axis): fraction of test windows with native ground-truth coverage. The full $K \in \{ 1 , 5 , 2 0 \}$ sweep is given in Appendix G.

## D. Operational Robustness

Real ADS-B receivers report at rates that vary with receiver quality and congestion. To test whether the architecture remains effective at coarser temporal resolution, we retrain the flow model at strides of 2, 4, and 8 (effective intervals ≈ 5, 10, 20 s): a dedicated model per stride, not one fixed model queried at variable resolution, since obs\_len and fut\_len must change with the token count. Stride-2 costs +4.9% in minADE@20 while roughly halving inference time; stride-8 costs +44.4% but runs 5.0× faster. Near-term calibration (NLL@ 10) is essentially unaffected across strides; long-horizon calibration degrades more than the displacement numbers alone suggest, since NLL is a log-likelihood. The architecture thus degrades gracefully across temporal resolutions and is cheap to redeploy for a receiver's typical rate (full results and discussion in Appendix F, Table 8).

Because one request already occupies the GPU, batching several aircraft into a single forward pass barely raises throughput for the Transformer models: one RTX 4000 Ada serves about 50 aircraft per second with Flow Tiny (one 20-sample forecast each), 7 with Flow Large and 1.4 with Diffusion Large (Table 6). At the 3 s mean ADS-B update interval, a single GPU therefore keeps the forecasts of roughly 150 aircraft current with Flow Tiny, or 20 with Flow Large, and independent requests can be spread over further GPUs as traffic grows.

The observed prefix can also be shortened at inference without retraining, by freezing the oldest tokens (repeated from the earliest retained observation) rather than truncating the fixed-length input. Figure 10 reports minFDE@5 as the retained history $n _ { \mathrm { k e e p } }$ shrinks from 43 to 2 points (N=2000 windows per level, ≈ 1.4 million forward passes in total): error stays essentially flat down to $n _ { \mathrm { k e e p } } \approx 2 1$ (half the window, under 5% cost) and rises sharply below $n _ { \mathrm { k e e p } } \approx 1 5 .$ reaching 2.6× the full-history error at $n _ { \mathrm { k e e p } } = 2$ . Because the frozen prefix is also mildly out-of-distribution relative to training data, this should be read as a conservative upper bound on the true cost of a genuinely shorter history.

Table 6 Inference cost and monitoring capacity on one NVIDIA RTX 4000 Ada (float32). Latency: one request (a single input sequence, batch size 1), K samples drawn in parallel, 20 Euler steps (CFM) or 100 DDIM steps (DDPM). Throughput: best over concurrent requests, at K=20. Aircraft per GPU: throughput $\times 3 \mathbf { s } ,$ the mean ADS-B update interval, i.e. aircraft whose forecast can be refreshed at every update. ≥: the card was not saturated at the largest concurrency tested (1024).
<table><tr><td>Model</td><td>Type</td><td>Params</td><td colspan="2">Latency (ms) K=20 K=50</td><td>Throughput (aircraft/s)</td><td>Aircraft per GPU</td></tr><tr><td>LSTM-Det</td><td></td><td>1.5M</td><td>1.15</td><td>1.15</td><td>≥ 100,379</td><td>≥ 301,137</td></tr><tr><td>CVAE</td><td>CVAE</td><td>1.5M</td><td>4.06</td><td>4.10</td><td>16,388</td><td>49,164</td></tr><tr><td>Diffusion Tiny</td><td>DDPM</td><td>1.5M</td><td>119.68</td><td>229.50</td><td>10.3</td><td>30</td></tr><tr><td>Flow Tiny</td><td>CFM</td><td>1.5M</td><td>23.76</td><td>47.25</td><td>51.3</td><td>153</td></tr><tr><td>Flow Small</td><td>CFM</td><td>7.1M</td><td>66.95</td><td>160.80</td><td>15.8</td><td>47</td></tr><tr><td>Diffusion Large</td><td>DDPM</td><td>20.7M</td><td>773.16</td><td>1870.65</td><td>1.4</td><td>4</td></tr><tr><td>Flow Large</td><td>CFM</td><td>20.7M</td><td>157.07</td><td>375.82</td><td>6.8</td><td>20</td></tr><tr><td>Flow Large, stride 2</td><td>CFM</td><td>20.7M</td><td>83.57</td><td>205.27</td><td>13.6</td><td>40</td></tr><tr><td>Flow Large, stride 4</td><td>CFM</td><td>20.7M</td><td>49.85</td><td>107.80</td><td>25.9</td><td>77</td></tr><tr><td>Flow Large, stride 8</td><td>CFM</td><td>20.7M</td><td>31.55</td><td>54.68</td><td>54.7</td><td>164</td></tr></table>

![](images/ba0c96c8de90c6aa997e4e189ca3a2265dfb78083f9919096f4849ee104a2f28.jpg)  
Fig. 10 Displacement error as a function of retained observation history. minFDE@5 (mean ± SEM, N=2000) as the observed prefix is shrunk from 43 points $( n _ { \mathbf { k e e p } } )$ to progressively shorter spans; the earliest $4 3 - n _ { \mathbf { k e e p } }$ tokens are frozen (repeated from the earliest retained sample) since the backbone's positional embeddings fix the input length at 43. Error is essentially flat down to $n _ { \mathbf k \mathbf e \mathbf p } \approx 2 1$ and rises sharply below $n _ { \mathbf k \mathbf e \mathbf p } \approx 1 5 $

## VII. Spatial Bias

Without any chart or procedure supervision, FlowATC reproduces known Bay Area airspace structure such as approach turns, holding patterns, and descent profiles. This can be done by understanding the latent structures from the absolute Cartesian coordinates which encode geographic position. We examine this in three complementary ways: a catalogue of recurring turn geometries and their geographic relationship to published airspace procedures (A), qualitative evidence that the model's sampled trajectories reproduce them (B), and a quantitative comparison, at branch points where traffic divides, of the maneuvers FlowATC samples against those actually flown, including by aircraft that do not

## A. Identifying Recurring Turn Patterns From ADS-B Trajectories

To evaluate whether the model reproduces location-specific maneuver structure, we use an independent catalogue of recurring turns built from ADS-B trajectories; all counts and evaluations below use its events from the collection of Section II. A turn is defined as the transition between two locally stable ground-track segments. The catalogue is derived directly from observed trajectories and is not used during model training; the full detection, classification, and grouping protocol is given in Appendix H. Each turn is then associated geographically with published FAA navigation references using its detected start point. A turn with exactly one nearby reference within 1 nautical mile is assigned to the isolated-fix population, whereas a turn with multiple nearby references is assigned to the clustered/regional population. This association should be interpreted as spatial proximity rather than evidence that a particular fix, controller instruction, or published procedure caused the maneuver.

![](images/8a90f442110aea89e8364fa06bce9b2a72556b94885c4e1b1f9591d8813174b7.jpg)  
(a) Turn pattern associated with the fix HLHRS.

![](images/a9704d14b27fdb4dcb1082a4acc61c7d268fc98533ecaf8bf35a3cb87fde5808.jpg)  
(b) Full trajectory context of the recurring pattern.

![](images/e99270ef7cd5da4ffa617884b51a8782ae0426c909c503fcc349635c95bb97de.jpg)  
(c) NIITE FOUR departure chart.  
Fig. 11 Recurring departure geometry recovered from ADS-B trajectories near SFO. (a) Turn pattern associated with the fix HLHRS. (b) Wider trajectory context of the recurring pattern. (c) The broad corridor and branching structure of the NIITE FOUR departure are visible in the extracted ADS-B pattern visualisations. Chart source: Federal Aviation Administration [21]

The resulting catalog contains 433 recurring patterns and 32,765 non-overlapping turn events: 229 isolated-fix patterns comprising 12,076 events, and 204 clustered/regional patterns comprising 20,689 events. Each pattern is annotated with a pattern identifier, an associated fix or regional fix set, turn direction, central turn angle, empirical angular tolerance, qualitative compactness label, and event count. Compactness is reported as sharp, moderate, or broad. Beyond cataloguing frequent turns, grouping trajectories by recurring local geometry provides an empirical representation of the maneuver modes available at specific locations in the airspace. These modes often align with published procedure corridors, although geometry and fix proximity alone do not uniquely identify the procedure assigned to an individual flight.

Figure 11 illustrates this relationship for one recurring SFO departure pattern. The pattern is named HLHRS\_11 because HLHRS is the only eligible navigation reference within the association radius of its detected turn start location. This name denotes geographic proximity and does not imply that HLHRS caused the maneuver or is part of the corresponding published procedure. Nevertheless, the observed trajectory corridor is broadly consistent with the NIITE FOUR departure shown alongside it.

The catalogue is designed to record recurring turns, so it contains only aircraft that turned. Evaluating a predictor requires the complementary view: every aircraft arriving at a decision point, whatever it does next. We therefore build on the catalogue to define branch points and collect all passages through them, recording the heading change actually flown, which is zero for aircraft that continue straight. At the 28 branch points retained for evaluation, a third of the 2,329 test passages fly a catalogued turn of that point and a third continue straight; the others fly maneuvers that the catalogue, restricted by design to recurring turns near navigation references, does not retain there.

## B. Qualitative Evidence of Learned Spatial Bias

The turn-pattern catalogue of Section VII.A provides a reference set of geographically recurring maneuvers against which model behavior can be inspected qualitatively. In this subsection, we compare extracted turn patterns with sampled futures from FlowATC in order to assess whether the model reproduces not only plausible trajectories, but also the spatially structured branching behavior observed in the ADS-B data

![](images/ad5833884fd91ca54147cb2636c0f57eeea449864b441d32664b9a00839c3724.jpg)  
(a) Recurring SFO departure patterns CA0048 and HLHRS\_11.

![](images/5abddd33813095ec8b30b761cad4335ee04819a00a57ff4667b76254aa9aed21.jpg)  
(b) Flow model samples for HLHRS\_11 test and validation events.  
Fig. 12 Multimodal departure structure near SFO. (a) The recurring patterns CA0048 (n=1,408) and HLHRS\_11 (n=375) share a similar initial right turn departure geometry before diverging. Lowercase n denotes the total number of occurrences in the turn pattern catalogue. The CA0048 trajectories continue the turn toward an eastbound continuation, whereas HLHRS\_11 cuts the initial turn shorter, continues farther north, and subsequently forms the identified left turn pattern. (b) Predictions for N=40 distinct HLHRS\_11 turn events (15 test + 25 validation), with validation added because only 15 test aircraft yielded eligible prediction cases for this pattern. K=20 sampled futures per event. Blue shows the last 12 of the 43 observed points displayed for clarity, green the ground truth future, and red the model samples.

Figure 12 shows a representative two-mode example near San Francisco International Airport (SFO). The reference catalogue contains two departure patterns with similar initial geometry: clustered pattern CA0048 $( n = 1 , 4 0 8 )$ and HLHRS\_11 (n = 375). Both depart SFO and initially execute a similar right turn, but their later geometries diverge. The CA0048 trajectories maintain the turn for longer and continue predominantly eastward, whereas HLHRS\_11 cuts the initial turn shorter, continues farther north, and subsequently forms a left-turn pattern. The corresponding predictions are constructed from test and validation cases belonging only to HLHRS\_11. Even in this setting, FlowATC samples populate both the northbound and eastbound continuations before the trajectories become fully distinguishable, indicating that FlowATC preserves multimodal uncertainty when the observed history is still compatible with more than one learned continuation. At the same time, the ground-truth future remains covered by a substantial subset of the samples

A more complex example is shown in Fig. 13, where six recurring turn patterns and a derived No Turn cohort share a similar inbound segment and then diverge into a fan-shaped set of continuations near Daly City. The catalogue identifies the following turn patterns in this region: CA0012 $( n = 1 , 1 2 5 )$ , CA0011 $( n = 2 7 4 )$ , CA0005 (n = 168), CA0004 $( n = 1 0 5 )$ , CA0003 (n = 127), and CA0017 $( n = 8 6 1 )$ . In addition, the visualization includes a No Turn cohort $( n = 3 2 2 )$ , comprising historical flights that share the same incoming corridor and continue through the branching region without entering one of the detected turn patterns. The associated prediction panels show two separate test subsets, built from $N = 4 0 \left( \mathsf { C A 0 0 1 2 } \right)$ and N = 29 (CA0017) distinct turn events with $K = 2 0$ sampled futures per event. In both cases, conditioned on the shared incoming geometry, FlowATC samples across several plausible outgoing branches while still covering the ground-truth continuation. However, not all branches are populated equally. The predictions concentrate most strongly on the historically dominant continuations, which are the patterns CA0012, CA0017 and straight continuation. This suggests that the observed prefix does not fully disambiguate the downstream branch, so FlowATC spreads probability mass over the modes roughly in proportion to how often each is flown (Section VII.C). Supporting this interpretation, the branches overlap substantially in turn-start altitude and speed, indicating that the preference for some continuations over others is not cleanly explained by simple kinematic separation. Overall, this behavior suggests that FlowATC has learned that the relevant uncertainty is not purely pointwise, but is organized around a discrete set of geographically recurring maneuver modes.

Taken together, these examples provide qualitative evidence that FlowATC learns location-specific airspace structure from trajectory data alone. Rather than collapsing toward an average future, it places samples across multiple historically observed maneuver modes when the observed history remains ambiguous. These qualitative observations motivate the quantitative evaluation at branch points of Section VII.C

## C. Probabilistic Reproduction of maneuvers at Branch Points

Displacement error and KDE-NLL measure where probability mass lands in space, but neither asks whether the model reproduces the maneuver actually flown: at a given decision point, how often aircraft turn, how sharply, and with

![](images/5b784a4c84798e2c42f8082b1ecba8d2ce15788a4935f45a0f8aa9b9063407bc.jpg)  
(a) Six recurring catalogue patterns.

![](images/725c81acd705898ab0bb99df4dd1c087334f25b77315d10a16fef159718630ef.jpg)  
(b) Flow-model samples for CA0012 test events.

![](images/26e894677d9a6a2312a4f6d82d8130ce6cb564e2c64b8632127bbc149f1ee514.jpg)  
(c) Flow-model samples for CA0017 test events.

Fig. 13 A fan-shaped multimodal branching region near Daly City. (a) Six recurring catalogue patterns and a derived No Turn cohort share a similar incoming trajectory before diverging into distinct continuations: CA0012 $( n = 1 , 1 2 5 )$ , CA0011 $( n = 2 7 4 )$ , CA0005 (n = 168), CA0004 $( n = 1 0 5 )$ , CA0003 (n = 127), $\mathsf { C A } \boldsymbol { 0 } \boldsymbol { 0 } \mathsf { 1 } \boldsymbol { 7 } \left( n = 8 6 \boldsymbol { 1 } \right)$ , and No Turn $( n = 3 2 2 )$ . Lowercase n denotes the total number of occurrences in the turn pattern catalogue. The No Turn group is a visualization-only cohort of historical flights that traverse the shared incoming corridor and continue through the branching region without entering one of the detected turn patterns. (b)–(c) Flow-model predictions for two separate test-only subsets. Panel (b) shows $N = 4 0$ distinct CA0012 turn events, and panel (c) shows $N = 2 9$ distinct CA0017 turn events, with K = 20 sampled futures per event. Blue shows the last 12 of the 43 observed points displayed for clarity, green the ground-truth future, and red the model samples.

what spread. The branch passages of Section VII.A provide the reference needed to ask this directly, and, unlike the cataloged events, they include the aircraft that do not turn.

A branch point merges cataloged patterns lying within 1.5 NM of each other whose incoming headings differ by at most 30°. A passage is selected from the observed history only: at the last observed point, the aircraft heads within $3 0 ^ { \circ }$ of the branch point's incoming heading, points at it, would reach it within 15 to 90 s at its current speed, and flies within the altitude band of the aircraft that fly its cataloged turns. One window is kept per passage. The altitude condition removes cruise overflights far above the procedure, which would otherwise inflate the share of straight flight. The outcome θ is the net heading change, unwrapped along the path, from the end of the observation until the path leaves a 4 NM disc around the branch point. The same function is applied to the observed future and to each of the K=50 completions drawn by FlowATC, so the model and its reference are measured on the same footing. On passages that fly a cataloged turn, θ recovers the cataloged angle to a median error of $1 . 0 ^ { \circ }$ without using the catalog's turn timing. We evaluate the 28 branch points that have at least 40 such passages in the held-out test split, 2,329 passages in total; the full protocol is given in Appendix I.

We first ask whether FlowATC reproduces the distribution of maneuvers flown at each branch point, pooled over all aircraft that reach it. Figure 14 overlays the observed and sampled distributions of θ at the six branch points with the most test passages; all 28 are shown in Appendix I, Fig. 20. FlowATC places its mass on the same modes, at the same angles and with comparable weights, including the three-branch fan near Daly City, and its share of straight flight matches the observed one within a few percentage points in every panel. Across the 28 branch points, $W _ { 1 }$ between the pooled sampled and observed outcomes averages $3 . 3 ^ { \circ }$ , weighted by passages. This distance cannot vanish on a finite test set: an ideal sampler drawing from the true distribution of each branch point would itself score about 3.8°. FlowATC reaches that level at 16 of the 28 branch points, and predicts a straight-through share of 32.2% against 32.9% observed, with a correlation of 0.99 across branch points.

![](images/e10f6a299200a45c434ae2b60216b960e4accf4633e516962c6d4edf00665f14.jpg)  
Fig. 14 Observed and sampled maneuver distributions at the six branch points with the most test passages. Gray bars: share of observed test passages per bin of net heading change θ (left turns negative, right turns positive, straight flight near zero). Green line: share of FlowATC samples (Flow Large, K=50 per passage) in the same bins. Titles give the number of passages $n ,$ the 1-Wasserstein distance $W _ { 1 }$ between the two distributions, the value an ideal history-blind sampler would reach on this test set, and the observed and predicted shares of straight flight $( | \theta | < 1 0 ^ { \circ } )$ . The horizontal range covers the 1st to 99th percentiles of both distributions. No catalogue information enters the figure.

Reproducing the pooled distribution does not require using the history: a sampler that ignored each aircraft and drew from the distribution of its branch point would do so as well. We therefore score each passage individually with the fair ensemble continuous ranked probability score (CRPS) [22], a fair finite-ensemble score that compares the K sampled outcomes of one passage with the outcome actually flown, rewarding both accuracy and appropriate spread; for a single deterministic prediction it reduces to the absolute error. The reference is a history-blind sampler that, for each passage, draws from the observed outcomes of all other passages at the same branch point: it knows the exact test distribution but nothing about the aircraft. FlowATC reduces its CRPS by 47% (95% bootstrap interval over branch points: 36 to 62%), is better on 82% of passages, and improves on it at all 28 branch points (Table 7). Its sampled intervals are close to calibrated: the central 80% interval contains the observed outcome for 81% of passages and the central 95% interval for 92%, so its tails are slightly too narrow. FlowATC thus reproduces maneuvers at two levels: marginally, as the distribution of what all aircraft reaching a branch point do, and conditionally, as a close to calibrated distributionfor each aircraft given its observed history.

Table 7 maneuver fidelity at the 28 branch points (N=2,329 test passages, K=50 samples each). W1: 1- Wasserstein distance between pooled sampled and observed outcomes, averaged over branch points weighted by passages. Straight: share of outcomes with $| \theta | < 1 0 ^ { \circ }$ (observed: 32.9%). CRPS skill: relative CRPS reduction against the history-blind sampler, with 95% bootstrap interval over branch points. Coverage: share of observed outcomes inside the central 80% and 95% sample intervals.
<table><tr><td>Model</td><td>Params</td><td>W1 (deg)</td><td>Straight (%)</td><td>CRPS skill</td><td>Cov. 80%</td><td>Cov. 95%</td></tr><tr><td>Flow Large</td><td>20.7M</td><td>3.3</td><td>32.2</td><td>0.47 [0.36, 0.62]</td><td>81.2</td><td>91.9</td></tr><tr><td>Flow Small</td><td>7.1M</td><td>3.7</td><td>33.1</td><td>0.41 [0.30, 0.58]</td><td>80.9</td><td>93.3</td></tr><tr><td>Flow Tiny</td><td>1.5M</td><td>9.6</td><td>37.3</td><td>0.25 [0.12, 0.44]</td><td>84.4</td><td>96.0</td></tr><tr><td>CVAE</td><td>1.5M</td><td>10.4</td><td>31.4</td><td>0.25 [0.13, 0.44]</td><td>74.0</td><td>83.6</td></tr><tr><td>History-blind sampler</td><td></td><td>3.8</td><td>32.9</td><td>0</td><td>一</td><td></td></tr><tr><td>Catalogued angles only</td><td></td><td>27.2</td><td>0.0</td><td>–0.92 [-1.81, –0.35]</td><td></td><td></td></tr><tr><td>Always straight</td><td></td><td>37.1</td><td>100</td><td>-1.35</td><td></td><td></td></tr></table>

Table 7 mirrors the ordering of Section VI. Flow Small is almost as good as Flow Large, with a CRPS skill of 41% against 47%, consistent with the saturation observed in Section VI.B, whereas Flow Tiny and the CVAE reach only 25% and roughly triple the marginal distance. The CVAE is also the only model whose intervals are clearly too narrow: its central 95% interval covers 84% of outcomes. Drawing from the cataloged turn angles alone does worse than the history-blind sampler, because, by design, it describes only the aircraft that turn.

## VIII. Conclusion

We presented FlowATC, a flow-matching architecture for aircraft trajectory prediction that frames the task as sequence inpainting. By concatenating observed and noisy tokens in a single DiT conditioned through AdaLN, FlowATC samples from the conditional distribution of future trajectories given the observed past. Four findings stand out. First, best-of-K accuracy and distributional calibration are related but distinct properties, and generative models dominate deterministic and kinematic baselines primarily on the former: at matched capacity, CFM and DDPM both surpass the CVAE baseline by 31–41% on minADE@20, while a single sample from either is no better, and sometimes worse, than a deterministic point estimate. K independent samples are what turn this single-sample weakness into a calibrated spatial density over future positions that can serve as an input to downstream conflict-risk estimation. Second, CFM is consistently the strongest objective at matched parameter count, outperforming DDPM by 11–26% in minADE@20 across the model sizes we test, though the gap narrows as both objectives scale, with DDPM's steeper scaling curve suggesting it may close entirely beyond the capacities evaluated here. Third, FlowATC is operationally flexible along several axes: error degrades gracefully within the training horizon, temporal decimation to stride 2 costs under 5% in minADE@20 while roughly halving inference time, and stride 8 gives a 5× speedup at a 44% accuracy cost. Queried beyond the training horizon, FlowATC remains the most accurate model up to 360 s once several samples are drawn, and its calibration stays stable where the CVAE’s diverges.

Lastly, FlowATC learns airspace structure without chart or procedure supervision: approach turns, holding patterns and descent profiles emerge from data alone. At 28 branch points where traffic divides, it reproduces the distribution of maneuvers actually flown, including the third of aircraft that continue straight, and gives each aircraft a close to calibrated distribution that improves on a history-blind sampler by 47% in CRPS.

## Future Work

FlowATC, in its current single-modality form, is designed as the first stage of a multi-modal predictor. Because the past enters only as additional context tokens, three conditioning streams can be appended without changing the loss or the inpainting mechanism:

1) ATC voice instructions: radio transcripts from Bay Area towers carry controller intent at maneuver initiation points; aligning voice embeddings with the trajectory space [23] should sharpen the posterior at turn points.

2) Weather fields: wind forecasts and SIGMET polygons tokenized as additional context;

3) FAA airspace structure: the named fixes and directed procedure edges (STARs, SIDs, airways, holding patterns) of the Bay Area encoded as a graph conditioning signal.

Each modality is expected to reduce density variance precisely at decision points where a single observed history is consistent with multiple controller instructions. A further stream is the flight plan itself: planned trajectories could be inserted conditionally into the future tokens as additional information

## Funding Sources

This work was funded by École Polytechnique. Mathurin Petit was hosted at CITRIS and the Banatao Institute, University of California, Berkeley, under the Visiting Scholar Researcher program.

## Acknowledgments

We thank Professor Trevor Darrell and Jiahui Lei from the Berkeley Artificial Intelligence Research Lab (BAIR) for their advice regarding trajectory-applied flow motion. We would also like to thank Tom Davis and John Robinson from Crown Innovations LLC, Parimal Kopardekar and James Murphy of NASA Ames Research Center, and Dragos Margineantu from Boeing for their insightful discussions regarding next-generation airspace decision support tools. We also thank Professor Hans-Ludger Dienel from Technische Universität Berlin for coordinating with UC Berkeley.

## References

[1] Ivanovic, B., Schmerling, E., Leung, K., and Pavone, M., “Generative Modeling of Multimodal Multi-Human Behavior," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 3088–3095. https://doi.org/10.1109/ IROS.2018.8594393.

[2] Prutsch, A., Schinagl, D., and Possegger, H., “ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace," Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2026.

[3] Yang, S., Liu, L., Chen, B., Cheng, S., Shi, Z., and Zou, Z., “GooDFlight: Goal-Oriented Diffusion Model for Flight Trajectory Prediction," IEEE Transactions on Aerospace and Electronic Systems, Vol. 61, No. 3, 2025, pp. 7447–7465. https://doi.org/10.1109/TAES.2025.3536436.

[4] Salzmann, T., Ivanovic, B., Chakravarty, P., and Pavone, M., “Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data," European Conference on Computer Vision (ECCV), 2020, pp. 683–700. https://doi.org/10.1007/978-3- 030-58523-5\_40.

[5] Jiang, C. M., Cornman, A., Park, C., Sapp, B., Zhou, Y., and Anguelov, D., "MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9644–9653. https://doi.org/10.1109/CVPR52729.2023.00930.

[6] Figuet, B., Krauth, T., and Barry, S., “Generative Short-Term Aircraft Trajectory Prediction with Conditional Flow Matching,” Journal of Open Aviation Science, Vol. 4, No. 2, 2026. https://doi.org/10.59490/joas.2026.8468

[7] Ho, J., Jain, A., and Abbeel, P., "Denoising Diffusion Probabilistic Models," Advances in Neural Information Processing Systems, Vol. 33, 2020, pp. 6840–6851.

[8] Song, J., Meng, C., and Ermon, S., “Denoising Diffusion Implicit Models," International Conference on Learning Representations, 2021.

[9] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M., "Flow Matching for Generative Modeling," International Conference on Learning Representations, 2023.

[10] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I., "Attention Is All You Need," Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 5998–6008.

[11] Peebles, W., and Xie, S., “Scalable Diffusion Models with Transformers," Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205. https://doi.org/10.1109/ICCV51070.2023.00387.

[12] Tong, Q., Hu, J., Chen, Y., Guo, D., and Liu, X., "Long-Term Trajectory Prediction Model Based on Transformer," IEEE Access, Vol. 11, 2023, pp. 143695–143703. https://doi.org/10.1109/ACCESS.2023.3343800.

[13] Zhao, Z., Zeng, W., Quan, Z., Chen, M., and Yang, Z., “Aircraft Trajectory Prediction Using Deep Long Short-Term Memory Networks," CICTP 2019, 2019, pp. 124–135. https://doi.org/10.1061/9780784482292.012.

[14] Zeng, W., Chu, X., Xu, Z., Liu, Y., and Quan, Z., “Aircraft 4D Trajectory Prediction in Civil Aviation: A Review," Aerospace, Vol. 9, No. 2, 2022, p. 91. https://doi.org/10.3390/aerospace9020091.

[15] Yin, Y., Zhang, S., Zhang, Y., Zhang, Y., and Xiang, S., "Context-aware Aircraft Trajectory Prediction with Diffusion Models," 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), 2023, pp. 5312–5317. https://doi.org/10.1109/ITSC57777.2023.10422124.

[16] Briden, J., Johnson, B., Linares, R., and Cauligi, A., “Diffusion Policies for Generative Modeling of Spacecraft Trajectories,” AIAA SCITECH 2025 Forum, 2025. https://doi.org/10.2514/6.2025-2775, AIAA Paper 2025-2775.

[17] Janner, M., Du, Y., Tenenbaum, J. B., and Levine, S., "Planning with Diffusion for Flexible Behavior Synthesis," International Conference on Machine Learning (ICML), Vol. 162, PMLR, 2022, pp. 9902–9915.

[18] Thiede, L. A., and Brahma, P. P., “Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction," Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9954–9963. https://doi.org/10.1109/ICCV.2019.01005.

[19] Gneiting, T., and Raftery, A. E., “"Strictly Proper Scoring Rules, Prediction, and Estimation," Journal of the American Statistical Association, Vol. 102, No. 477, 2007, pp. 359–378. https://doi.org/10.1198/016214506000001437.

[20] Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E., “Stochastic Interpolants: A Unifying Framework for Flows and Diffusions," Journal of Machine Learning Research, Vol. 26, No. 209, 2025, pp. 1–80.

[21] Federal Aviation Administration, “"NITE FOUR DEPARTURE (RNAV) (NITE4.NITE), San Francisco International Airport (SFO)," https://www.faa.gov/aero\_docs/dtpp/2607/00375NITE.PDF, 2026. FAA Terminal Procedures Publication, cycle 2607, effective 9 July–6 August 2026; procedure dated 11 July 2024; accessed 18 July 2026.

[22] Ferro, C. A. T., “Fair Scores for Ensemble Forecasts," Quarterly Journal of the Royal Meteorological Society, Vol. 140, No. 683, 2014, pp. 1917–1923. https://doi.org/10.1002/qj.2270.

[23] Brusset, L., Petit, M., Kam, J., and Bayen, A., “V2TATC: A Joint Voice-Trajectory Embedding Framework and Dataset for Air Traffic Controller Situational Awareness," arXiv preprint arXiv:2608.28981, 2026. URL https://arxiv.org/abs/2608.28981.

## Appendices

The appendices collect supporting material that complements the main article and are useful for reproduction.

## A. Random Sample of Flow-Large Predictions on the Test Set

To complement the curated multimodal examples of Section VII, which are deliberately selected to illustrate branching behavior at known decision points, Figure 15 shows an uncurated sample: 40 test windows drawn uniformly at random (fixed seed, no cherry-picking), predicted by Flow Large (20.7 M parameters) at K=20 with the same 20-step Euler integration used throughout the paper. For each panel, blue marks the observed history (open circle: last observed point); the dashed green line and star mark the ground-truth future and its final position; the K=20 sampled trajectories are drawn as thin rays, coloured by the Gaussian KDE density (Scott's-rule bandwidth, Eq. (4)) of their endpoint, with the same density shown as filled contours.

The large majority of panels show the predicted density tightly bracketing the ground truth, on both straight segments and turns, including one holding pattern reproduced almost exactly. A small minority (2 of the 40 panels shown) instead show the sampled density missing the realized outcome entirely. This is not hidden or excluded: an honest sample from a genuinely uncertain predictor occasionally misses, and this is precisely the behavior that KDE-NLL penalizes and that a best-of-K figure alone would let a model hide.

## B. Data Preprocessing details

Raw CSV files are ingested per aircraft, deduplicated on timestamp, and segmented into continuous flight segments: a new segment begins whenever the callsign changes or the inter-sample gap exceeds 120 s. Raw geodetic coordinates present three difficulties for neural processing: longitude is non-linear (one degree covers different physical distances at different latitudes), true track is a circular variable $( 3 5 9 ^ { \circ }  0 ^ { \circ } )$ , and position and velocity live at incompatible scales. We resolve these by projecting onto a local tangent-plane Cartesian frame centered on SFO $( \phi _ { 0 } = 3 7 . 6 2 1 3 ^ { \circ } \mathrm { N } .$ $\lambda _ { 0 } = 1 2 2 . 3 7 9 0 ^ { \circ } \mathrm { W } )$

$$
x = ( R + h ) \cos \phi ( \lambda - \lambda _ { 0 } ) , \quad y = R ( \phi - \phi _ { 0 } ) , \quad z = h ,\tag{13}
$$

with velocity components $\nu _ { x } = \nu \sin \theta , \nu _ { y } = \nu \cos \theta , \nu _ { z } = \dot { h } . \ \dot { \uparrow }$ Segments are windowed into 86-point sequences with a stride of 10.

To expose the true timing to the network, we compute, for each sample, the elapsed time since the start of its window, apply a sinusoidal embedding [10] to it, and pass the result through a small MLP before adding it to the corresponding input token. Flight segments and windows that are entirely on the ground (ADS-B on\_ground flag) are discarded during preprocessing, together with a small number of windows containing corrupted altitude readings $( < - 5 0 \mathrm { m } ) ;$

![](images/a57096c7a3efdcb1d15db3d8506e1e22d24ad97c66bdf31fd9c1fc70c53b658d.jpg)  
Fig. 15 Uncurated random sample of 40 test-set predictions

![](images/004b43011ad17768c8d8439cd5f4db016ec625f1804f90e3e2bd0b7b1ce16906.jpg)  
Fig. 16 Conceptual comparison of Flow Matching and DDPM generation. Flow Matching transports samples along straight conditional paths, requiring fewer integration steps and yielding a simpler training objective. DDPM reverses a Markov noising chain along curved trajectories.

every retained window contains at least one airborne sample. The train/validation/test split is drawn at the level of individual aircraft: the set of unique ICAO24 transponder codes is shuffled once with a fixed seed (42) and partitioned 85%/10%/5%, and every window belonging to a given aircraft is assigned entirely to a single split, preventing correlated windows from leaking across splits.

## C. Flow matching and DDPM

Diffusion models and flow matching are two families of generative models that learn to sample from a data distribution $p ( \mathbf { x } )$ by reversing a noise corruption process; both can be made conditional by holding a context signal fixed while the target is denoised. Denoising Diffusion Probabilistic Models [7] define a Markov chain that gradually adds Gaussian noise to a data sample over $T _ { \mathrm { d i f f } }$ steps until it is indistinguishable from pure noise. A network $\varepsilon _ { \theta }$ is trained to predict the added noise, enabling step-by-step denoising from a pure-noise sample at inference. The number of inference steps can be reduced (e.g. from 1000 to 100) with the DDIM sampler [8] without significant quality loss. Conditional Flow Matching [9] learns a velocity field $u _ { \theta } ( { \bf x } , t )$ that transports samples along straight-line paths from a Gaussian base distribution (t = 0) to the data distribution $( t = 1 )$ . The objective is simple and deterministic: at any interpolation ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ , the target velocity is the constant $\left( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \right)$ . At inference, integrating the learned field with a few Euler steps produces a clean sample. Compared to DDPM, CFM has a simpler loss, requires fewer steps, and yields straighter sampling trajectories in data space (Fig. 16).

## D. CVAE Decoder Variance at the First Prediction Step

The CVAE's near-flat error curve at short horizon (Fig. 7a) and its NLL divergence at long horizon share a common cause. Decomposing the step-1 prediction on the held-out test set: the true one-step displacement averages 311 m, the decoder's conditional-mean error is 111 m (about twice the 56 m of constant velocity), but the sampled Gaussian noise adds a further $\sigma _ { 0 } \approx 6 0 5$ m horizontally. This σ sits at its trained floor $( \approx 5 0 \mathrm { m } )$ for every subsequent step, so the excess variance at n=1 is a boundary effect rather than a genuine displacement error. Deterministic sampling (only latent draw, no decoder noise) confirms this: step-1 minADE@ 1 drops from 542.6 m to 118.1 m. The noise cannot simply be removed, however: because the prior places $\approx 9 5 \%$ of its mass on a single category for a given observed history (12 of 25 categories carry non-negligible mass in aggregate, but conditionally the latent is nearly one-hot), disabling decoder noise collapses minADE@20 back toward minADE@1 (853.4 vs. 508.2 m) instead of the K draws exploring K distinct futures. The model's cumulative time encoding contributes to this: the decoder can recover the local inter-step gap for $n \geq 1$ by differencing two elapsed-time values it received itself, but the very first future gap (last observation to first prediction) is reachable only through the compressed history encoding. That defect, however, acts on the conditional mean, not on the decoder variance: at n=1 the mean is already within 111 m of the truth while $\sigma _ { 0 } \approx 6 0 5 \mathrm { m } .$ so the offset is five times larger than any error the mean could contribute, and a better-informed mean cannot remove it. The best-of-K comparisons that carry our headline results are driven by σ and by the latent's conditional diversity, neither of which depends on this input, so we do not expect a time-corrected CVAE to change the $K { = } 2 0$ ordering of Table 3.

The same limitation applies architecturally to the deterministic LSTM baseline, which uses an identical no-feedback decoder driven solely by cumulative $t _ { \mathrm { r e l } } ;$ its own step-1 error (94.3 m) is nearly twice the steady per-step increment observed from step 2 onward $( \approx 5 0 \mathrm { m } )$ , consistent with the same missing local time gap.

## E. Full $K \in \{ 1 , 5 , 2 0 \}$ Displacement Sweep

Figure 7, in the main text, shows minDisp@1, minDisp@20 and density calibration only; $K { = } 5$ is omitted there to keep the figure to three panels, since it sits between the $K { = } 1$ and $K { = } 2 0$ stories without adding a qualitatively new one. Figure 17 reports the complete sweep for completeness.

![](images/fbdb7ed097ac8f8279f67a0555e1a5adbb87c5a696ee5b0220f1c20bba864fb7.jpg)

![](images/e3b8ff30c34d8d663fce4b8ed3768848d87ae506b895b83629c70997e0de8c11.jpg)

![](images/7d597a18d958ec0fa2624385927c541f4757f1b35e3b425a98b959cbddbd0e2f.jpg)

![](images/73ed75d96ad7cf608d6778ea2e606e2d19c6b6be9d0ba4ed2f39712dda64106d.jpg)  
Fig. 17 Full $K \in \{ 1 , 5 , 2 0 \}$ displacement-error sweep, extending Fig. 7 with the intermediate $K { = } 5$ panel.

## F. ADS-B Downsampling — Full Results

Because downsampling changes the physical time each future token represents, the KDE NLL columns of Table 8 are evaluated at the token index whose horizon is closest to the stride-1 reference rather than at a fixed token index (see table caption for the resulting horizons). Under this matched-horizon comparison, NLL@43 rises from 20.2 to 22.2–23.0 nats across the three strides. Because NLL is a log-likelihood, this modest-looking increase in nats corresponds to the ground-truth density dropping by a factor of $e ^ { 2 . 0 } \approx 7 . 4 \times$ (stride-2) to $e ^ { 2 . 8 } \approx 1 6 . 4 \times ( \mathrm { s t r i d e } { - 4 } )$ , a substantially larger effective loss of calibration than the nat values alone suggest. NLL@ 10, by contrast, is essentially unchanged and even improves slightly at stride-8 (12.9 vs. 13.3 nats, $e ^ { - 0 . 4 } \approx 1 . 5 \times$ more likely). The architecture retrains effectively on lower-rate feeds, while the model's confidence at long horizons becomes markedly less trustworthy under coarser sampling, even though the displacement-error degradation (Table 8) looks comparable in relative terms across horizons

Table 8 Effect of temporal stride on Flow Large (20.7 M params), $N = 6 3 { , } 0 1 6$ test trajectories. Distances in metres ± SEM. KDE NLL in nats, reported at the token whose physical horizon is nearest the stride-1 reference $( n { = } 1 0 , 2 0 , 4 3 \approx 3 0 , 6 0$ , 129 s); stride models thus reach {30, 60, 132} s (stride 2), {36, 60, 132} s (stride 4) and {24, 48, 120} s (stride 8). Bold = reference.
<table><tr><td>Model</td><td>Stride</td><td colspan="2">K = 1</td><td colspan="2"> $K = 5$ </td><td colspan="2"> $K = 2 0$ </td><td colspan="3">KDE NLL@n</td><td>ms/pred</td></tr><tr><td></td><td></td><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td>minADE</td><td>minFDE</td><td> $n { = } 1 0$ </td><td> $n { = } 2 0$ </td><td> $n { = } 4 3$ </td><td></td></tr><tr><td>Flow Large</td><td>1</td><td> $\mathbf { 9 0 3 . 4 } _ { \pm 5 . 9 }$ </td><td> $2 1 5 6 . 8 _ { \pm 1 4 . 2 }$ </td><td> $\mathbf { 4 6 7 . 3 _ { \pm 3 . 4 } }$ </td><td> $\mathbf { 1 0 1 3 . 2 _ { \pm 7 . 8 } }$ </td><td> $\mathbf { 3 0 7 . 4 } _ { \pm 2 . 4 }$ </td><td> $\bar { 5 } 7 6 . 8 _ { \pm 5 . 3 }$ </td><td> ${ \bf 1 3 . 3 _ { \pm 0 . 1 } }$ </td><td> $1 7 . 5 _ { \pm 0 . 5 }$ </td><td> $\mathbf { 2 0 . 2 _ { \pm 0 . 4 } }$ </td><td>157.07</td></tr><tr><td>Flow Large</td><td>2</td><td> $9 8 6 . 7 _ { \pm 6 . 6 }$ </td><td> $2 3 1 5 . 9 _ { \pm 1 5 . 9 }$ </td><td> $4 9 4 . 2 _ { \pm 3 . 5 }$ </td><td> $1 0 4 4 . 4 _ { \pm 8 . 1 }$ </td><td> $3 2 2 . 6 _ { \pm 2 . 3 }$ </td><td> $5 8 5 . 5 _ { \pm 5 . 1 }$ </td><td> $1 3 . 8 _ { \pm 0 . 1 }$ </td><td> $1 8 . 2 _ { \pm 0 . 4 }$ </td><td> $2 2 . 2 _ { \pm 0 . 9 }$ </td><td>83.57</td></tr><tr><td>Flow Large</td><td>4</td><td> $1 0 3 1 . 2 _ { \pm 6 . 7 }$ </td><td> $2 2 7 1 . 0 _ { \pm 1 5 . 1 }$ </td><td> $5 3 1 . 2 _ { \pm 3 . 8 }$ </td><td> $1 0 6 6 . 3 _ { \pm 8 . 3 }$ </td><td> $3 5 2 . 5 _ { \pm 2 . 7 }$ </td><td> $6 1 9 . 8 _ { \pm 5 . 6 }$ </td><td> $1 5 . 1 _ { \pm 0 . 2 }$ </td><td> $1 8 . 8 _ { \pm 0 . 4 }$ </td><td> $2 3 . 0 _ { \pm 0 . 7 }$ </td><td>49.85</td></tr><tr><td>Flow Large</td><td>8</td><td> $1 3 1 0 . 7 _ { \pm 8 . 5 }$ </td><td> $2 6 3 1 . 3 _ { \pm 1 7 . 5 }$ </td><td> $6 7 4 . 7 _ { \pm 4 . 8 }$ </td><td> $1 2 3 9 . 2 _ { \pm 9 . 7 }$ </td><td> $4 4 3 . 8 _ { \pm 3 . 2 }$ </td><td> $7 1 1 . 3 _ { \pm 6 . 3 }$ </td><td> $1 2 . 9 _ { \pm 0 . 1 }$ </td><td> $1 6 . 2 _ { \pm 0 . 2 }$ </td><td> $2 2 . 5 _ { \pm 0 . 6 }$ </td><td>31.55</td></tr></table>

## G. Full Extrapolation Sweep and Shard-Stitching Detail

Figure 9 in the main text omits minDisp@5; Figure 18 reports the complete $K \in \{ 1 , 5 , 2 0 \}$ sweep. Because $\Delta t _ { \mathrm { v i r t } } = H _ { \mathrm { m a x } } / 4 3$ is fixed once $H _ { \mathrm { m a x } }$ is chosen, a single run only gives good local temporal resolution near its own $H _ { \mathrm { m a x } } ;$ a run with $H _ { \mathrm { m a x } } = 3 6 0 \mathrm { s }$ has $\Delta t _ { \mathrm { v i r t } } \approx 8 . 4$ s/token, under-resolving the 0–90 s range relative to a dedicated $H _ { \mathrm { m a x } } = 9 0 \mathrm { { s } }$ run $( \Delta t _ { \mathrm { v i r t } } \approx 2 . 1$ s/token). We therefore stitch three independent shards, each with its own 43-token resampling and its own K=50 posterior draws, keeping for each time band the shard with the finest resolution available: (0, 90] s from $H _ { \mathrm { m a x } } { = } 9 0 .$ (90, 180] s from $H _ { \mathrm { m a x } } { = } 1 8 0$ ,(180, 360] s from $H _ { \mathrm { m a x } } { = } 3 6 0$ . The visible seam at each band boundary (most noticeable at $t \approx 1 8 0 \mathrm { s } )$ is not a change in model behavior but a change of estimator: the two shards on either side differ simultaneously in token resolution, in the finite-sample statistics computed from a different draw of the K posterior samples, and in which subset of test windows passes the coverage filter at that instant.

![](images/beffaa7cd27f6dcc453f0643041bc8946fe27a2851235d13e88f92ffa4f3db26.jpg)

![](images/e51f64614ac230ceb94cd71494c877c90227aa1a6124f32f600c9a5ff012bb38.jpg)

![](images/280854f12074f28cb8339419c39ee0bd6f96a517de9641a2cbf4f67c295037c8.jpg)

![](images/9d4225792d7c9cb0a96374b9423eb53b5aea21d24db3bb7c31c4cfe36be9e139.jpg)  
Fig. 18 Full $K \in \{ 1 , 5 , 2 0 \}$ extrapolation sweep, extending Fig. 9 with the intermediate K=5 panel.

## H. Turn Catalogue: Detection, Classification, and Grouping Protocol

We first retain sustained airborne trajectory segments by requiring a geometric altitude of at least 150 m and a ground speed of at least $2 5 \mathrm { m s } ^ { - 1 }$ . After filtering, gaps longer than 30 s split a track into separate segments, and segments containing fewer than five points are discarded.

![](images/342df56e4dc31667d61afc7aca5e72646dfd8ac23cbd23b830be5af7d7d8d9f0.jpg)  
Fig. 19 A turn event is defined as the transition between two locally stable ground-track segments. The detected turn start and end delimit this transition, while the turn angle, $\Delta \psi _ { ; }$ , is the wrapped angular difference between the mean ground-track directions of the pre-turn and post-turn stable segments.

The detector uses the ADS-B-reported true track, smoothed with a short circular moving average to avoid discontinuities at $0 ^ { \circ } / 3 6 0 ^ { \circ }$ . A stable segment is a sequence over which the pointwise ground-track rate remains below $0 . 5 ^ { \circ } \mathrm { s } ^ { - 1 }$ and the cumulative drift from the initial course remains below $6 ^ { \circ }$ . For two consecutive stable segments with circular-mean directions $\bar { \psi } _ { \mathrm { p r e } }$ and $\bar { \psi } _ { \mathrm { p o s t } }$ , the signed turn angle is

$$
\Delta \psi = \left[ \left( \bar { \psi } _ { \mathrm { p o s t } } - \bar { \psi } _ { \mathrm { p r e } } + 1 8 0 ^ { \circ } \right) \mathrm { m o d } 3 6 0 ^ { \circ } \right] - 1 8 0 ^ { \circ } .\tag{14}
$$

Positive and negative values denote right and left turns, respectively, and the interval between the two stable segments defines the detected turn start and end. Changes of at least $2 0 ^ { \circ }$ are retained directly as major turns; smaller changes between $1 0 ^ { \circ }$ and $2 0 ^ { \circ }$ are retained only when the observed path geometry is consistent with the reported ground-track change. Smaller course adjustments are excluded from the pattern catalogue. Extending the association rule of Section VII.A, turns without an eligible navigation reference within 1 nautical mile are left unmatched and excluded from both pattern populations. After detecting individual turn events, we group events that occur in the same local area and exhibit similar incoming and outgoing ground-track directions, together with a similar overall signed ground-track change, $\Delta \psi$ . When a turn start is associated with a single navigation reference, patterns are formed at the fix level; in dense terminal regions with several nearby references, turns are grouped regionally. Only recurring and geometrically coherent groups are retained as catalogue patterns, with limited manual review used to resolve evident duplicates or inconsistent cases.

## I. Branch Points, Passages and Scores

Branch points. Catalogued patterns with at least 60 events are located at the position of their associated fix (isolated family) or at the median start of their turns (clustered family). Patterns lying within 1.5 NM of each other whose incoming headings differ by at most $3 0 ^ { \circ }$ are merged; the branch point's incoming heading is the event-weighted circular mean. This yields 80 branch points.

Passages. A window is a candidate passage if, at its last observed point, the aircraft is airborne (altitude at least 150 m, ground speed at least $2 5 \mathrm { m s } ^ { - 1 }$ , as in the catalogue), heads within $3 0 ^ { \circ }$ of the incoming heading, has the branch point within $2 5 ^ { \circ }$ of its heading, and would reach it in 15 to 90 s at its current speed. Candidate windows of the same aircraft separated by less than $9 0 0 \mathrm { s }$ form one passage, represented by the window whose time to the branch point is closest to 25 s. The altitude band of a branch point spans the 5th to 95th percentile, widened by $5 0 0 \mathrm { m } ,$ of the altitudes of training and validation passages that fly one of its catalogued turns; branch points with fewer than 10 such passages have no band. Test passages outside the band are discarded. The evaluation uses the 28 branch points with at least 40 remaining test passages.

Outcome. Headings are computed over four-step segments and unwrapped along the path, starting from the last observed heading. The outcome θ is the accumulated heading change at the first point, after the closest approach, where the path is more than 4 NM from the branch point. When the path is still within the disc at the end of the window (26.9% of test passages) or never enters it (0.6%), θ is taken at the end of the window; the same rule applies to every sample. On the 771 test passages that fly a catalogued turn of their branch point, θ matches the catalogued angle to a median of $1 . 0 ^ { \circ }$ , and to within $1 5 ^ { \circ }$ for 86% of them.

Scores. $W _ { 1 }$ compares the Kn pooled sampled outcomes of a branch point with its n observed outcomes. The level of an ideal history-blind sampler is estimated as half the mean $W _ { 1 }$ between two random halves of the observed outcomes since $W _ { 1 }$ between n empirical values and their law scales as $n ^ { - 1 / 2 }$ ; simulations on mixtures shaped like branch points agree within $4 \%$ . For a passage with samples $\theta ^ { ( 1 ) } , \ldots , \theta ^ { ( K ) }$ and observed outcome $\theta ,$ the ensemble CRPS is

$$
\mathrm { C R P S } = \frac { 1 } { K } \sum _ { k } \bigl | \theta ^ { ( k ) } - \theta \bigr | - \frac { 1 } { 2 K ( K - 1 ) } \sum _ { k \neq l } \bigl | \theta ^ { ( k ) } - \theta ^ { ( l ) } \bigr | .\tag{15}
$$

The history-blind sampler uses as ensemble the observed outcomes of all other test passages at the same branch point.   
Bootstrap intervals resample branch points 1000 times.

![](images/db807ec00618ba316e586531c8e3bb8924b12f376ace42d9cb50eb14db4b9801.jpg)  
net heading change (°), left < 0 < right