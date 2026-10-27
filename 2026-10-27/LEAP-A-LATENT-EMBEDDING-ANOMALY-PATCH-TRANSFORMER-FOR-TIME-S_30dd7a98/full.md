# LEAP: A LATENT EMBEDDING ANOMALY PATCH-TRANSFORMER FOR TIME-SERIES ANOMALY DETECTION

Cristhian Moya-Mota <sup>1,2</sup>, Ignacio Aguilera-Martos <sup>1,2</sup>, Diego García-Gil <sup>2,3</sup>, Julián Luengo <sup>1,2</sup>

<sup>1</sup>Department of Computer Science and Artificial Intelligence, University of Granada, Granada, Spain. <sup>2</sup>Andalusian Institute of Data Science and Computational Intelligence (DaSCI), University of Granada, Granada, Spain. <sup>3</sup>Department of Software Engineering, University of Granada, Granada, Spain <sup>1</sup>Emails: cris190402@correo.ugr.es, nacheteam@ugr.es, djgarcia@ugr.es and julianlm@decsai.ugr.es

## ABSTRACT

Time-series anomaly detection is dominated by reconstruction-based neural detectors, whose objective is structurally flawed. Since the target is also the input, an expressive model can learn to reconstruct anomalies too, leaving uninformative residuals. Recent large-scale benchmarks show that, on curated data, elaborate deep detectors rarely beat simple statistical baselines, while current leaders rely on representation learning. We propose LEAP (Latent Embedding Anomaly Patch-Transformer), which combines masked prediction with latent representation-based anomaly evidence, without relying on reconstruction or attention-based scoring. The model removes patches from a window and predicts them from the remaining temporal and cross-channel context, so prediction errors measure how well learned dynamics explain hidden segments. Deterministic complementary group masking ensures every time step is scored while hidden, with identical visible fractions during training and inference. LEAP adds three label-free evidence sources: latent-embedding novelty calibrated by local density, local temporal contrast, and per-channel novelty pooled over channels selected by six label-free criteria. Under the TSB-AD protocol, the full model achieves a VUS-PR of 0.5179 ± 0.0022 on 180 held-out multivariate series. On TSB-AD-U, a univariate variant combining multi-scale masked-prediction scoring with latent novelty, without contrast or channel-pool novelty, achieves 0.6194 ± 0.0016 on 350 held-out series. Both results average three random seeds. A fivestage ablation on TSB-AD-M, supported by paired Wilcoxon signed-rank tests, shows statistically significant gains from each successive component.

Keywords Anomaly detection; Time series; Masked modeling; Self-supervised learning; Representation learning; Benchmarking

## 1 Introduction

Time-series anomaly detection (TSAD) is the task of deciding, for every instant of a recorded signal, whether the behavior observed there is compatible with the process that produced the rest of the signal [1]. It underpins condition monitoring in industrial plants, fault diagnosis in spacecraft telemetry, intrusion detection in cyber-physical systems, arrhythmia screening in electrocardiography, and the operation of large software systems that must detect failures before their users do [2].

What distinguishes it from anomaly detection on unordered data is that the unit of analysis is rarely a single sample. Normality depends on temporal context, which encompasses periodicity, trend, and regime. As a result, a value that is unremarkable at one phase of a cycle can become pathological shortly thereafter, indicating that the true object of abnormality is usually a subsequence: a deformed heartbeat with the same mean, variance and range as a healthy one, a sustained shift of a sensor’s operating level, a burst of variance, or a configuration of channels that is impossible jointly even though each channel is plausible on its own [3]. Labels are scarce and unbalanced, the catalog of anomaly types is heterogeneous enough that a detector tuned for spikes fails on level shifts, and evaluation has to credit a detection that is correct in substance but a few samples early or late.

The families of methods addressing this task have accumulated rather than replaced one another. The earliest fit a model to the series and inspect its residuals, from ARIMA and exponential smoothing onwards [4]. A second embeds the series into a space of sliding windows and identifies windows that are dissimilar to the rest, with the time-series discord providing the canonical formulation [5]. A third learns a predictor of future values with a neural network and scores forecast residuals [6]. A fourth, dominant for much of the last decade, reconstructs its input and flags large reconstruction errors as anomalies [7], while a fifth learns short-subsequence representations through self-supervision and detects anomalies by their distance to normal representations [8].

Given its prevalence, the reconstruction family warrants closer examination. A key limitation of this approach is that the target is also used as input, a characteristic that cannot be fully addressed through architectural refinements alone. A model with enough capacity can learn the identity map, and what it copies includes the anomalies. The very property that makes a network expressive enough to represent complex normal dynamics is the property that lets it reproduce the abnormal ones. Practical systems compensate by integrating additional evidence into the score, such as inter-channel structure terms, amplitude statistics, hand-crafted shape metrics, and association discrepancies, thereby tacitly admitting that the residual alone does not carry the information the detector needs.

The practical significance of this issue is well established. Wu [9] demonstrated that standard benchmark datasets are severely compromised by triviality, mislabeling, and run-to-failure bias. Kim [10] showed that the widely used point-adjust protocol is sufficiently permissive for random scores to outperform published detectors. Schmidl [11] and, more recently, Liu [12] re-ran dozens of algorithms under controlled conditions and reached a conclusion that should reorganize the research agenda: simple architectures generally match or beat elaborate designs, and no single method dominates across data sources. The numbers are blunt. On the multivariate track of TSB-AD, which curates 1,070 manually inspected series across 40 datasets and reports VUS-PR [13] as its primary measure, the best-performing Transformer detector is TranAD at 0.18 VUS-PR, while Anomaly Transformer and DCdetector, the two most cited Transformer detectors in the field, achieve scores of 0.12 and 0.09, respectively, falling below the 0.20 of an Isolation Forest and far below the 0.31 of a plain convolutional forecaster [12].

Since then, the immediate context of this work has shifted. A new class of detectors has raised the multivariate state of the art from the 0.31–0.35 range to approximately 0.48 VUS-PR, largely by learning representations of short subsequences and scoring test windows by their distance to normal ones. VACE follows this principle and reaches 0.48 VUS-PR [14], while PaAno reaches 0.43 using nearest-neighbor scoring over learned patch representations [8]. KAN-AD [15] achieves 0.41 through Fourier-based temporal modeling, while CrossAD [16] reaches 0.33 using cross-scale reconstruction and cross-window context. Thus, the recent shift has been away from reconstruction-based Transformers toward representation learning with distance-based scoring. This convergence motivates our design, which combines a masked-prediction backbone with the same family of scoring rules.

Rather than mitigating the limitations of reconstruction-based detection, this paper addresses their underlying cause. We present LEAP (Latent Embedding Anomaly Patch-Transformer), a semi-supervised detector whose base objective is predict-from-context rather than standard reconstruction. In this approach, the input window is divided into fixedlength patches across the temporal and channel dimensions. A deterministic subset of these patches is replaced by a learned mask embedding, requiring the network to predict the missing content using the remaining temporal and cross-channel context. As the masked patch is not visible to the encoder, the model cannot simply copy its content during reconstruction. Therefore, a large prediction error indicates that the observed context and the learned normal dynamics of the series are insufficient to explain the hidden values. Because the masking groups form an exact partition of the patch grid, every time step is predicted exactly once and always from context, and because a single group is masked both during training and during scoring, there is no distribution shift between how the model was trained and how it is used.

The second contribution addresses the structural limitations of masked prediction error, which may miss sustained level shifts and globally novel but locally predictable patterns. LEAP first combines masked prediction error with local moments to capture level shifts (Stage 1), then adds three complementary sources of label-free evidence: (i) density-calibrated latent-window novelty (Stages 2–3); (ii) local temporal contrast to improve precision in non-stationary recordings (Stage 4); and (iii) per-channel novelty over channels selected by six label-free criteria (Stage 5). Their additive fusion preserves complementary anomaly evidence. The five-stage ablation evaluates density calibration separately from the latent novelty it refines, rather than treating it as an additional evidence source.

The contributions of this work are the following.

1. A masked-modeling detector for multivariate time-series anomaly detection, implemented over a Transformer encoder, with a deterministic complementary group masking scheme that staggers holes across channels, so that a broken inter-channel relationship surfaces in the same error signal as a deformed waveform or a level shift, with no hand-crafted structural metric (Sections 3.3–3.4).

2. A factorized temporal/channel encoder with an optional latent bottleneck on the channel axis, which reduces the cost of the cross-channel route from $\mathcal { O } ( C ^ { 2 } )$ to $\dot { \mathcal { O } } ( C K )$ making hundreds of channels affordable (Section 3.3).

3. A label-free scoring pipeline (comprising per-channel error calibration with a spread floor, soft channel aggregation, edge-weighted overlap-add, event-scale smoothing, and tolerance dilation) alongside three additive evidence layers: density-calibrated latent k-nearest-neighbor novelty, local contrast, and criterionbased per-channel novelty (Sections 3.6–3.7.4).

4. A five-stage decomposition of the model that doubles as a controlled ablation, where each stage is a strict extension of the previous one and reduces to it exactly by setting a single coefficient to zero (Section 3.7).

5. An evaluation under the TSB-AD protocol on 180 multivariate and 350 univariate held-out series, with VUS PR as the primary measure, paired Wilcoxon signed-rank tests for every increment, and full disclosure of the tuning procedure (Sections 4 and 5).

The remainder of the paper is organized as follows. Section 2 reviews the foundations of TSAD, major methodologica families, and current benchmarking evidence. Section 3 introduces the proposed LEAP framework. Section 4 details the experimental setup, and Section 5 presents and discusses the results. Finally, Section 6 concludes this manuscript.

## 2 Background

This section identifies the anomaly-detection mechanisms on which LEAP builds and the evaluation evidence that motivates their combination.

## 2.1 From Anomaly Detection to Time Series

Anomalies may be point-wise, contextual, or collective, depending on whether an observation is unusual in isolation, only in its context, or as part of an otherwise plausible group [17]. Detection may be supervised (labeled anomalies), semi-supervised (a training sample believed to be normal), or unsupervised (a potentially contaminated sample). LEAP follows the semi-supervised setting of the benchmark used here. It trains on a clean prefix and scores the complete series without using anomaly labels.

Several classical scoring rules remain relevant after features are learned. Nearest-neighbor distance measures novelty, but can over-score normal samples in sparse regions. Local-density correction, as in Local Outlier Factor, compares a sample with its neighbors on their own scale [18]. Isolation-based methods instead use the ease of separating a sample by random splits [19], while subspace methods score what a low-dimensional projection fails to explain [20]. LEAP uses distance and local-density correction in its latent space.

For time series, the anomalous object is often a subsequence whose meaning depends on temporal context, rather than an isolated value. Sliding windows preserve that context and permit distance, density, or subspace scoring of temporal patterns. They also expose shape and cross-channel relationships that point-wise scoring would discard.

## 2.2 Time-Series Anomaly Detection

Let $X = ( x _ { 1 } , \dots , x _ { T } )$ , with $x _ { t } \in R ^ { C }$ , denote a series of length T and C channels. A detector returns a score $s _ { t } \in R$ for each time step, with larger scores indicating stronger evidence of abnormality. In the semi-supervised setting it fits on a clean prefix $X _ { 1 : T _ { \mathrm { t r } } }$ . Relevant anomalies include global or contextual points, deformed subsequences, sustained level or variance changes, altered seasonality or trend, and joint channel configurations whose individual values remain plausible [1]. This diversity motivates complementary evidence rather than a single score.

## 2.2.1 Statistical Approaches

Classical models score departures from temporal behavior. ARIMA, exponential smoothing, and state-space models use prediction residuals. Control charts accumulate evidence of a sustained shift, and seasonal decomposition isolates departures from periodic structure [4]. These approaches are transparent, but a low-order or single-channel model may miss shape-preserving and relational anomalies. Applying classical methods to windows helps. Subsequence PCA [20], local density, and nearest-neighbor scoring are competitive on univariate benchmarks [12].

## 2.2.2 Discord and Bank-Based Approaches

A time-series discord is a subsequence whose nearest non-trivial match is unusually distant [5]. The Matrix Profile computes such distances across a series [21]. Other methods compare windows with representative normal patterns. NormA builds a normal-pattern set [22], while shape clustering supplies normal centroids [23]. Their reusable principle is comparison with a bank of normal subsequences, although raw-space distances can be sensitive to amplitude, phase, and channel count.

## 2.2.3 Deep Forecasting and Reconstruction Detectors

Deep forecasting extends residual scoring. LSTM-AD models prediction errors [6], DeepAnT uses a convolutional predictor [24], and Telemanom adds dynamic thresholding [25]. Forecast error can nevertheless be high on normal but unpredictable signals. Reconstruction methods instead score the error of reproducing an observed window. Examples include OmniAnomaly’s stochastic recurrent model [7], USAD’s adversarial decoders [26], and GDN’s explicit sensor dependencies [27]. Because a reconstruction model sees its target as input, sufficient capacity may also reproduce anomalies. Its residual alone need not measure abnormality reliably. LEAP addresses this risk by predicting content withheld from its encoder. Masked prediction differs from both, since it hides the target from the encoder while still using context on both sides of a gap and in other channels.

## 2.2.4 Attention-Based Detectors

Attention provides a way to model long temporal and cross-channel dependencies, but the scoring objective still matters. Anomaly Transformer compares learned and prior associations [28]. TranAD combines reconstruction with selfconditioning and adversarial training [29]. DCdetector contrasts patch representations rather than reconstructing input [30]. Their original benchmark results require caution, as common datasets contain flaws that undermine comparisons [9], while point-adjusted evaluation can artificially inflate detection performance [10]. Under the curated, range-aware TSB-AD evaluation, these published Transformer detectors rank below simpler alternatives [12]. This observation concerns their objectives and measured scores, not an inherent limitation of attention as an encoder.

## 2.2.5 Self-Supervised Representation Learning

Self-supervised methods learn features without labeled anomalies. TS2Vec learns multiscale contrastive representations [31]. CARLA injects synthetic anomalies as contrastive negatives [32], while COUTA uses anomaly-informed calibration of a one-class boundary [33]. Such approaches encode assumptions about future anomalies. Masked modeling instead hides part of an input and predicts it from the remainder, a principle established for image patches by masked autoencoders [34]. PatchTST introduced masked patch prediction for time-series representation learning [35]. SimMTM reconstructs masked points by aggregating representations from multiple masked views [36]. MOMENT extends masked patch reconstruction to general-purpose time-series foundation models [37], while TSPulse employs masked reconstruction in both temporal and frequency domains [38]. In LEAP, masked prediction error serves directly as the detection score, requiring each position to be scored under masking with comparable training and inference contexts; the corresponding rules are detailed in Section 3. Table A.1 in Appendix A summarizes the scoring mechanisms of these families and the anomaly types to which they are suited.

## 2.2.6 Representation Learning with Nearest-Neighbor Scoring

Recent detectors combine a learned encoder with geometric scoring against normal embeddings. PaAno trains a compact patch encoder with a triplet and consecutiveness objective, then scores distance to a coreset bank [8]. VACE regularizes embedding trajectories through velocity consistency and combines position and direction scores [14]. PaAno+ adds multiscale and cross-variable modeling [39]. PAI identifies an amplitude blind spot in window-normalized embeddings and restores raw-amplitude evidence [40]. LEAP likewise preserves level information and includes a local-moment term (Section 3.6.4).

The common design is an encoder trained on normal windows followed by a geometric anomaly score. The scoring geometry may involve a coreset distance, a trajectory-based distance, or an amplitude-aware variant. This family is closest to LEAP’s latent novelty term, but the learned representation is not the whole detector here. The masked prediction error provides an independent contextual signal, while local-density calibration reduces the tendency of a raw nearest-neighbor distance to mistake legitimate sparse normal regions for anomalies. Neither improvement requires synthetic anomalies.

Other recent models refine reconstruction or forecasting instead. CrossAD predicts finer from coarser scales with prototypes [16], CATCH models frequency-domain channel relations [41], DADA uses adaptive bottlenecks with adversarial decoders [42], xLSTMAD updates the recurrent core [43], and KAN-AD replaces the deep predictor with a Kolmogorov–Arnold expansion [15].

## 2.3 Positioning of This Work

LEAP combines two scoring mechanisms. Masked modeling [34] supplies an error computed only for content hidden from the encoder, while the discord and normal-bank tradition [5] supplies novelty relative to training windows. Local-density calibration [18] adjusts that novelty for unevenly populated parts of the latent space. Unlike PaAno [8] and VACE [14], whose auxiliary objectives yield geometric scores, LEAP has a contextual prediction score before latent novelty is added. The Transformer implements the predictor. Table A.1 therefore places its scoring logic across masked modeling and representation with a normal bank. The additive formulation is detailed in Section 3.7. Table 1 provides the family-level comparison. Per-detector results appear in Section 5.1.

The two signals address different questions. Masked error asks whether a local observation follows from its visible temporal and channel context. Latent novelty asks whether the shape of a larger window resembles normal training windows at all. A window can be locally predictable yet globally unfamiliar, just as a familiar overall shape can contain a brief contextual violation. Density correction then makes the novelty term relative to the structure of the normal bank instead of using one global distance scale. These signals are combined rather than choosing one of them for every series. Thus, attention only exchanges information among visible patches, and the anomaly evidence arises from the error on hidden patches and from a latent comparison that does not make the whole score depend on one embedding distance.

## 2.4 Evaluation Paradigms and Empirical Landscape

Reported TSAD gains depend strongly on the data and metric. Earlier benchmarks contained trivial, mislabeled, or run-to-failure cases [9], while point-adjusted F<sub>1</sub> can reward even random rankings [10]. Range-based precision and recall [44], affiliation scores [45], and VUS metrics [13] better account for anomalous events and small timing offsets. Curated re-evaluations [11] culminate here in TSB-AD, with 1,070 inspected series across 40 datasets, separated tuning and held-out sets, and VUS-PR as the primary measure [12].

The metric choice matters for the claim this paper makes. Point-wise measures penalize a detector whose score rises close to a labeled interval but outside it. Point adjustment can go too far in the other direction by crediting an entire event after one hit. VUS-PR integrates precision–recall performance across temporal tolerance widths, avoiding a single chosen buffer and a single chosen decision threshold. Its precision–recall basis also reflects the rarity of anomalous time steps. The full metric definitions and secondary measures are given in Section 4.3.

Table 1 summarizes published TSB-AD-M results by detector family, while Section 5.1 distinguishes reported figures from our re-measurements. The strongest representation-and-bank entry reaches 0.48 VUS-PR. This provides empirical motivation for incorporating latent geometry into LEAP, but not for treating latent geometry alone as a sufficient detector: different anomaly types can remain invisible to any single source of evidence. Since the table combines heterogeneous methods and results from different implementations and evaluation conditions, small differences should not be interpreted as definitive rankings. Section 5 reports individual baselines and explicitly identifies results remeasured under the same protocol; the experimental protocol and metrics are given in Section 4.

Table 1: How the method families of Table A.1 stand on the TSB-AD-M held-out evaluation set (180 series), mean VUS-PR, aggregated from the per-detector values published in the works cited in Section 2.2.6. Best is the strongest published member of the family, range the spread across its members, and period the publication span of those members. The per-detector ranking is deferred to Section 5.1, where it serves as the comparison for LEAP.
<table><tr><td>Family</td><td>Best member</td><td>VUS-PR</td><td>Range</td><td>Period</td></tr><tr><td>Representation + bank</td><td>VACE [14]</td><td>0.48</td><td>0.42-0.48</td><td>2026</td></tr><tr><td>Forecasting (deep)</td><td>KAN-AD [15]</td><td>0.41</td><td>0.31-0.41</td><td>2019–2025</td></tr><tr><td>Reconstruction (deep)</td><td>xLSTMAD-R [43]</td><td>0.37</td><td>0.19–0.37</td><td>2019–2025</td></tr><tr><td>Discord/matrix profile</td><td>MMPAD [46]</td><td>0.35</td><td></td><td>2026</td></tr><tr><td>Statistical</td><td>PCA [20]</td><td>0.31</td><td>0.20-0.31</td><td>2003-2012</td></tr><tr><td>Transformer</td><td>TranAD [29]</td><td>0.18</td><td>0.09–0.18</td><td>2022–2023</td></tr></table>

## 3 LEAP (Latent Embedding Anomaly Patch-Transformer)

This section defines the masked predictor, its score calibration and the additional evidence used by LEAP. Figure 1 shows the complete pipeline.

G complementary passes: every (c, n) token is masked in exactly one pass, so every timestep is scored while hidden, and the visible fraction 1 1/G is identical in training and in scoring  
(a) Masked-patch backbone — predict deleted patches from context  
![](images/17f57dbd9f5d2a5922884fbd8074b25f82f65f3273b6f260727d69f9dc750142.jpg)

(b) Score assembly — four label-free evidences, each added, never substituted  
![](images/bb0bfc24c82d52c4bde4dee9571f461c8c77a832a10c91532875e1caa2cd440b.jpg)  
Figure 1: Architecture and scoring flow of LEAP. Panel (a) shows a $C \times N$ patch grid, complementary masks staggered across channels, the factorized encoder and masked prediction. Each patch is scored once while hidden. Panel (b) shows per-channel calibration, overlap-add aggregation and event-scale smoothing. The score then combines local moments, density-calibrated latent novelty, local contrast and per-channel novelty from a six-criterion channel pool.

## 3.1 Overview and Design Principles

LEAP combines masked prediction with representation-and-bank scoring. Following the TSB-AD protocol [12], it trains on an anomaly-free prefix $X _ { \mathrm { t r } } = \mathbf { \bar { \Phi } } ^ { \mathbf { \Phi } } X _ { 1 : T _ { \mathrm { t r } } } \in \mathbf { \Phi } R ^ { T _ { \mathrm { t r } } \times C }$ and returns scores $s \in R ^ { T }$ for the complete series $X \in R ^ { T \times C }$ . Training, calibration and score rescaling use no anomaly labels. Appendix C gives the complete procedure in Algorithm 1, its computational cost and a detailed account of this label-free property.

The base model masks selected patches and predicts them from visible temporal and cross-channel context, scoring each position once while hidden. Staggered gaps across channels prevent copying the input, making the resulting error a contextual prediction signal rather than a reconstruction residual. Because local prediction may miss globally unusual patterns or sustained level shifts, LEAP complements it with local-moment evidence inside the base score and with three additive evidence layers: density-calibrated latent novelty, local temporal contrast, and per-channel novelty. The stages and their coefficients are defined below; notation is provided in Table B.1 in Appendix B.

## 3.2 Standardization and Window Length

The training prefix determines the channel scales and context window length for each series.

## 3.2.1 Global Per-Channel Standardization with Scale Floor

Let $\mu _ { c }$ and $\sigma _ { c }$ be the mean and standard deviation of channel c on $X _ { \mathrm { t r } }$ . Nearly constant training channels would amplify small changes if divided by $\sigma _ { c } .$ The following scale floor uses the median positive channel spread.

$$
a = \mathrm { m e d i a n } \{ \sigma _ { c } : \sigma _ { c } > 0 \} ,\tag{1}
$$

$$
\varphi = \rho _ { \sigma } a , \qquad Q = \{ c : \sigma _ { c } < \varphi \} ,\tag{2}
$$

$$
\tilde { \sigma } _ { c } = \operatorname* { m a x } ( \sigma _ { c } , \varphi ) , \qquad \tilde { x } _ { t , c } = \frac { x _ { t , c } - \mu _ { c } } { \tilde { \sigma } _ { c } } .\tag{3}
$$

Here $\rho _ { \sigma } = 0 . 0 5$ . Using only positive spreads keeps the anchor nonzero when most training channels are inactive. The quiescent set Q is retained for calibration. Each channel uses one scale across the complete series, which preserves level changes that per-window normalization could remove. Reversible Instance Normalization [47] uses statistics of each window. If these include masked patches, their values leak through the normalization parameters. Per-window scaling can also erase sustained level shifts. LEAP therefore uses training-prefix channel statistics throughout and predicts absolute patch levels from visible context.

## 3.2.2 Dominant-Period Windowing

The window length follows the training prefix’s dominant period. For a centered channel of length $n ,$ LEAP computes autocorrelation r by FFT, zero-padded to $2 ^ { \lceil \log _ { 2 } 2 n \rceil }$ for linear correlation and normalized to $r ( 0 ) = 1$ . It selects the largest local maximum beyond lag $\ell _ { 0 } = 3$

$$
\hat { P } _ { c } = \underset { \ell \in \mathcal { L } } { \arg \operatorname* { m a x } } r ( \ell ) , \qquad \mathcal { L } = \{ \ell : r ( \ell ) > r ( \ell - 1 ) , \ r ( \ell ) > r ( \ell + 1 ) \} ,\tag{4}
$$

and accepted only if it clears an extreme-value band,

$$
r ( \hat { P } _ { c } ) \geq \frac { \sqrt { 2 \ln m } } { \sqrt { n } } ,\tag{5}
$$

where $m = | \mathcal { L } |$ . Restricting the search to peaks avoids choosing the smallest lag of a monotonically decaying correlation. Unlike a per-lag band of ${ \bar { 2 / } } { \sqrt { n } } ,$ , which noise peaks cross more often as more lags are searched, this band follows the Gaussian extreme-value bound for the largest of m approximately $\mathcal { N } ( 0 , 1 / n )$ correlations. For multivariate series, $\hat { P }$ is the median of the periods of the channels that clear the band. Estimation uses at most the 64 highest-variance channels and the first 20,000 training samples. The window is then a multiple of the patch length:

$$
L = \mathrm { c l i p } \Bigg ( p \cdot \mathrm { r o u n d } \Bigg ( { \frac { \pi _ { \mathrm { p e r } } \hat { P } } { p } } \Bigg ) , ~ 4 p , ~ L _ { \mathrm { m a x } } \Bigg ) , \qquad L _ { \mathrm { m a x } } = \mathrm { m i n } \Bigg ( 3 8 4 , p \left\lfloor { \frac { T _ { \mathrm { t r } } } { 2 p } } \right\rfloor \Bigg ) .\tag{6}
$$

With $\pi _ { \mathrm { p e r } } = 2 $ , the window spans about two periods. If no significant period is found, $L = \operatorname* { m i n } ( L _ { \operatorname* { m a x } } , 1 6 p )$ . Multiples of p need no patch padding. The four-patch minimum leaves visible context, and the cap bounds attention cost.

Windows use stride $\Delta = \operatorname* { m a x } ( 1 , L / 8 )$ , reduced if needed to obtain at least 512 training windows. Uniform subsampling caps their number at 4,000. The tail window is always included.

## 3.3 Tokenization and Encoder Architecture

Patching reduces sequence length while preserving local temporal patterns [35], an approach widely explored in time-series Transformers [48]. Two-stage attention captures temporal and cross-channel dependencies efficiently [49]. Given the competitiveness of linear [50] and period-aware models [51], we evaluate the backbone separately in Section 5.2.

## 3.3.1 Tokens

An input window $x \in R ^ { L \times C }$ is partitioned into a $C \times N$ grid of patches $m u _ { c , n } \in R ^ { p }$ , where $n = 0 , \ldots , N - 1$ indexes temporal position and c denotes channel identity. Each patch is linearly projected into a D-dimensional latent space and combined with temporal and channel encodings. The patches contain the globally standardized values from Eq. (3).

$$
\begin{array} { r } { h _ { c , n } ^ { ( 0 ) } = \underbrace { \left[ m _ { c , n } m \epsilon _ { \mathrm { M A S K } } + \left( 1 - m _ { c , n } \right) \left( W _ { e } m u _ { c , n } + m b _ { e } \right) \right] } _ { \mathrm { c o n t e n t , e r a s e d w h e r e } \ m _ { c , n } = 1 } + \mathrm { P E } _ { n } + m \gamma _ { c } , } \end{array}\tag{7}
$$

where $m _ { c , n } \in \{ 0 , 1 \}$ is the mask indicator, $m \epsilon _ { \mathrm { M A S K } } \in R ^ { D }$ is a learned parameter, $\mathrm { P E } _ { n }$ is the fixed sinusoidal positional encoding of Vaswani [52] and $m \gamma _ { c } \in R ^ { D }$ is a learned channel embedding. The order of operations in Eq. (7) is deliberate. The substitution happens before the position and channel codes are added, so what survives of a masked token is only where the hole is located, given by its instant and channel, and never what was originally in it.

## 3.3.2 Factorized Attention

A multivariate window is a $C \times N$ grid of tokens. Attending over the flattened grid would cost $\mathcal { O } ( ( C N ) ^ { 2 } )$ , which is prohibitive on wide datasets. LEAP factorizes attention into two axes. Each of the B encoder blocks applies, in

pre-norm residual form,

$$
h \gets h + \mathrm { A t t n } _ { \mathrm { t i m e } } \big ( \mathrm { L N } ( h ) \big ) , \qquad \mathrm { o v e r } \ n , \mathrm { f o r } \mathrm { e a c h } \ c ,\tag{8}
$$

$$
h  h + \mathrm { A t t n } _ { \mathrm { c h a n } } \big ( \mathrm { L N } ( h ) \big ) , \qquad \mathrm { o v e r } \ c , \mathrm { f o r } \mathrm { e a c h } n ,\tag{9}
$$

$$
h \xleftarrow { } h + \mathrm { F F N } \big ( \mathrm { L N } ( h ) \big ) , \qquad &  \mathrm { F F N } ( m v ) = W _ { 2 } \mathrm { G E L U } ( W _ { 1 } m v ) ,\tag{10}
$$

so that the cost is $\mathcal { O } ( C N ^ { 2 } D + N C ^ { 2 } D )$ instead of $\mathcal { O } ( C ^ { 2 } N ^ { 2 } D )$ . Within one block, information can travel from $\left( c _ { 1 } , n _ { 1 } \right) \mathbf { t o } \left( c _ { 2 } , n _ { 2 } \right)$ through $( c _ { 1 } , n _ { 2 } )$ . Temporal attention supplies within-channel context. Channel attention supplies simultaneous observations from other variables, allowing relational disruptions to affect masked prediction. Fused scaled dot-product kernels avoid materializing attention matrices in GPU memory.

## 3.3.3 Latent Channel Bottleneck

When $C > C _ { \mathrm { l a t } } = 1 2 8$ , LEAP replaces quadratic channel attention with K learned channel latents $\pmb { \Lambda } \in \mathbb { R } ^ { K \times D }$ following the inducing-point idea of Lee [53]. They aggregate channel information and broadcast it back through two cross-attention steps:

$$
\mathbf { A } ^ { \prime } = \mathbf { A } + \operatorname { C r o s s A t t n } \bigl ( Q = \mathbf { A } , \ K V = \operatorname { L N } ( \mathbf { h } _ { : , n } ) \bigr ) ,\tag{11}
$$

$$
\mathrm { A t t n } _ { \mathrm { c h a n } } ( \mathbf { h } _ { : , n } ) = \mathrm { C r o s s A t t n } \big ( Q = \mathrm { L N } ( \mathbf { h } _ { : , n } ) , \ K V = \mathrm { L N } ( \boldsymbol { \Lambda } ^ { \prime } ) \big ) ,\tag{12}
$$

where $\operatorname { L N } ( \cdot )$ denotes layer normalization applied prior to attention. Equation (11) condenses $C$ channel representations into K latent modes, while Eq. (12) distributes this aggregated cross-channel context back to individual channels.

The channel-attention cost per patch falls from $\mathcal { O } ( C ^ { 2 } D )$ to $\mathcal { O } ( C K D )$ ). With $C = 2 4 8$ and $K = 1 6 ,$ this is roughly an eight-fold reduction in channel-attention operations. Gradient checkpointing can further reduce training memory.

After the final encoder block, a linear head $W _ { d } \in \mathbb { R } ^ { D \times p }$ predicts each patch from $\mathbf { h } _ { c , n } ^ { ( B ) }$ . The patches form $\hat { x } \in \mathbb { R } ^ { L \times C }$

## 3.4 Complementary Group Masking

The mask is a deterministic partition of the $( c , n )$ token grid into $G$ groups

$$
g ( c , n ) ~ = ~ \left( \lfloor n / b \rfloor + c + \varrho \right) { \bmod { G } } , \qquad M a s k ^ { ( g ) } = \{ ( c , n ) : g ( c , n ) = g \} .\tag{13}
$$

Here b is the block length in patches, giving gaps of bp samples, and $\varrho$ is a rotation offset. Evaluating all $G$ groups makes the assembled score independent of that offset. The partition has three properties.

1. Exact partition. Every token belongs to exactly one group, so masking the G groups in turn predicts every token exactly once and always from context. There is no position in the output that the model was allowed to see while predicting it.

2. Train–inference consistency. Training and inference each mask one group per pass. Pass g leaves a fraction $1 - | \mathcal { M } ^ { ( g ) } | / ( C N )$ visible, averaging $1 - 1 / G$ over all groups in both phases. Thus a masked prediction is evaluated with the same single-group context pattern used for training.

3. Staggering across channels. The additive c term rotates the pattern by one group per channel. Without it, a group would consist of whole time columns, removing every channel at the same instant and leaving the cross-channel route with nothing to read. The task would collapse to pure temporal interpolation. With it, any masked token retains both a temporal context and a set of simultaneous observations in other channels.

At inference, the masked error of a window is assembled from the G complementary passes,

$$
e _ { t , c } = \big | \tilde { x } _ { t , c } - \hat { x } _ { t , c } ^ { ( g ( c , \lfloor ( t - 1 ) / p \rfloor ) ) } \big | ,\tag{14}
$$

where $\hat { x } ^ { ( g ) } = f _ { \theta } ( \tilde { x } , M a s k ^ { ( g ) } )$ is the prediction of the pass that hid the group containing $( c , \lfloor ( t - 1 ) / p \rfloor )$

## 3.5 Training Objective

Training minimizes the mean absolute error restricted to the masked positions of one randomly drawn group,

$$
\mathcal { L } ( \theta ) \ = \ E _ { x \sim \mathcal { D } _ { \mathrm { t r } } , g , \varrho \sim \mathcal { U } \left\{ 0 , G - 1 \right\} } \left[ \frac { \sum _ { ( c , n ) \in M a s k ^ { ( g ) } } \sum _ { j = 1 } ^ { p } \left| \tilde { x } _ { n p + j , c } - \hat { x } _ { n p + j , c } \right| } { p \left| M a s k ^ { ( g ) } \right| } \right] .\tag{15}
$$

The ℓ loss limits the influence of contamination in a training prefix that is assumed, but not guaranteed, to be anomaly-free. A squared loss would let a single large residual dominate the gradient.

Optimization uses AdamW, gradient clipping, a ReduceLROnPlateau scheduler, and early stopping on the last 20% of the training prefix. A fixed validation-mask seed keeps validation loss independent of mask-sampling noise. BF16 mixed precision is optionally supported.

## 3.6 Calibration and Base Anomaly Scoring

The masked error of Eq. (14) is calibrated per channel, aggregated across channels and overlapping windows, and smoothed before it enters the Stage-1 score. Local moment evidence complements this error.

## 3.6.1 Per-Channel Error Calibration

Channels differ in how predictable they are, and a raw error is therefore not comparable across them. LEAP estimates the first two moments of the masked error on the (anomaly-free) training windows,

$$
m _ { c } = E [ e _ { \cdot , c } ] , \qquad s _ { c } = \sqrt { E [ e _ { \cdot , c } ^ { 2 } ] - m _ { c } ^ { 2 } } ,\tag{16}
$$

and regularizes the spread through optional pooling toward the median of active channels and a lower floor,

$$
\bar { s } = \mathrm { m e d i a n } _ { c \not \in { \mathscr { Q } } } s _ { c } , \qquad \tilde { s } _ { c } = \left( 1 - \lambda \right) s _ { c } + \lambda \bar { s } ,\tag{17}
$$

$$
s _ { c } ^ { \star } = \operatorname* { m a x } \bigl ( \tilde { s } _ { c } , \ \mathrm { m a x } ( \rho _ { f } \bar { s } , \varepsilon ) \bigr ) , \qquad z _ { t , c } = \frac { e _ { t , c } - m _ { c } } { s _ { c } ^ { \star } } .\tag{18}
$$

The floor prevents an unbounded z when $s _ { c } \approx 0$ . The parameter λ ranges from the raw spread at 0 to a pooled spread at 1. The reported setting $\lambda = 0$ keeps the floor while disabling pooling.

## 3.6.2 Channel Aggregation

The calibrated errors are reduced over channels by a temperature-controlled soft maximum:

$$
a _ { t } \ = \ \sum _ { c } w _ { t , c } \ \operatorname* { m a x } ( z _ { t , c } , 0 ) , \qquad w _ { t , c } = \frac { \ \exp ( z _ { t , c } / \tau ) } { \sum _ { c ^ { \prime } } \exp ( z _ { t , c ^ { \prime } } / \tau ) } .\tag{19}
$$

With $\tau = 0 . 5$ , the soft maximum balances the mean, which can dilute a single-channel anomaly, against the maximum, which can favor a noisy channel. Hard max and mean remain available.

$\mathbf { A } \mathbf { s } \ \tau$ increases, weights approach uniformity; as $\tau  0$ , the largest calibrated error dominates. The positive part in Eq. (19) prevents small errors from canceling evidence across channels.

## 3.6.3 Overlap-Add, Event-Scale Smoothing, and Tolerance Dilation

Windows overlap, so a time step receives one score per window that covers it. Positions near a window edge have seen less context and are down-weighted by a Gaussian profile before accumulation.

$$
s _ { t } = { \frac { \sum _ { ( w , j ) : \mathrm { s t a r t } _ { w } + j = t } \omega _ { j } a _ { w , j } } { \sum _ { ( w , j ) : \mathrm { s t a r t } _ { w } + j = t } \omega _ { j } } } , \qquad \omega _ { j } = \exp \left( - { \textstyle { \frac { 1 } { 2 } } } \left( { \frac { j - L / 2 } { \rho _ { \omega } L } } \right) ^ { 2 } \right) ,\tag{20}
$$

with $\rho _ { \omega } = 0 . 2 5$ . The result is then smoothed with a moving average of width $W _ { s } = \eta p ,$

$$
s  s m o o t h ( s ; W _ { s } ) , \qquad W _ { s } = \eta p , \quad \eta = 8 .\tag{21}
$$

This reads error at event scale and yields the largest base-score gain in the component analysis (Appendix G.2). A maximum filter of width $\kappa = \operatorname* { m a x } ( \bar { 3 } , \lfloor L / 1 0 \rfloor )$ ) then dilates the combined Stage-1 signal,

$$
( d i l _ { \kappa } { v } ) _ { t } = \operatorname* { m a x } _ { | u | \leq \kappa / 2 } v _ { t + u } ,\tag{22}
$$

which tolerates small temporal offsets (Section 4.3).

## 3.6.4 Local Moment Evidence

To detect sustained level or dispersion changes, LEAP computes local moments on the signal rather than inside a normalization layer. Let $M _ { t , c }$ and $S _ { t , c }$ be the mean and standard deviation in a centered window of $W _ { m } = \operatorname* { m a x } ( 8 , 4 p )$ samples. Cumulative sums compute them in $\mathcal { O } ( T )$ , with edge replication to avoid artificial endpoint shifts. Trainingprefix baselines $( \mu _ { c } ^ { M } , \sigma _ { c } ^ { M } )$ and $( \dot { \mu } _ { c } ^ { S } , \sigma _ { c } ^ { S } )$ use a positive-median scale floor. The evidence combines two-sided deviations in both moments,

$$
\ell _ { t } = u n i t { \Big ( } \operatorname* { m a x } _ { c } { \big | } M _ { t , c } - \mu _ { c } ^ { M } { \big | } / \sigma _ { c } ^ { M } { \Big ) } \ + u n i t { \Big ( } \operatorname* { m a x } _ { c } { \big | } S _ { t , c } - \mu _ { c } ^ { S } { \big | } / \sigma _ { c } ^ { S } { \Big ) } , \qquad u n i t ( v ) = { \frac { v } { q _ { 0 , 9 9 } ( v \mid v > 0 ) } } .\tag{23}
$$

Taking absolute deviations captures both increases and decreases, and summing the two profiles keeps level and dispersion evidence instead of only the larger of them. The operator unit divides each profile by the 99th percentile of its active values $( v > 0 )$ , so that a mostly silent signal does not shrink the scale, and it adds no ϵ to the denominator, since an additive floor could hide genuine anomalies. A profile without active values is left unchanged.

The per-series rescaling changes neither profile’s internal order, although their sum can produce a new ranking, and it leaves the two profiles on comparable scales. The maximum over channels lets a shift confined to a single sensor carry the evidence.

## 3.6.5 Stage-1 Base Anomaly Score

Stage 1 adds rescaled local moments to the smoothed prediction error before dilation,

$$
s ^ { ( 1 ) } = \mathrm { } d i l _ { \kappa } \Big ( u n i t ( s ) + w _ { \ell } u n i t ( \ell ) \Big ) , \qquad w _ { \ell } = 1 .\tag{24}
$$

At $w _ { \ell } = 0$ the score uses prediction error alone. Alternative moment combinations are compared in Section 5.2.

## 3.7 Additive Evidence Layers

Stage 1 is complemented by latent novelty, density calibration, local contrast, and per-channel novelty. These components form five nested ablation stages, with density calibration evaluated separately from latent novelty (Appendix G.1).

## 3.7.1 Latent Shape Novelty (Stage 2)

A window can have an unfamiliar shape even when its patches are locally predictable. Stage 2 measures its novelty in the encoder’s latent space. Let

$$
h ( x ) \ = \ \mathrm { E n c } \big ( W _ { e } \mathrm { p a t c h i f y } ( x ) + b _ { e } + \mathrm { P E } + \gamma \big ) \in R ^ { C \times N \times D }\tag{25}
$$

denote the encoder representation without masking. Let A contain the min $( C , \chi )$ highest-variance training channels, with $\chi = 4$ . Concatenating their patch representations and applying $\ell _ { 2 }$ normalization gives

$$
\phi ( x ) \ = \ \frac { \mathrm { v e c } \left( h _ { \mathsf { A } , : , : } ( x ) \right) } { \left\| \mathrm { v e c } \left( h _ { \mathsf { A } , : , : } ( x ) \right) \right\| _ { 2 } + \epsilon } \ \in \ R ^ { | { \cal A } | N D } .\tag{26}
$$

Concatenation preserves temporal order that pooling would discard. Two windows can have similar average features yet differ in the order of their patch-level changes. Retaining that order makes shape deformation visible to the bank distance. Because this vector is $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ , the bank distance depends on the direction of the representation, while amplitude information is retained by the separate local-moment term of Section 3.6.4.

A uniformly sampled bank B holds at most $N _ { \mathrm { b a n k } } = 1 2 0 0$ training-window embeddings. A test window’s novelty is its distance to the nearest bank element,

$$
d ( x ) \ = \ \operatorname* { m i n } _ { \beta \in B } \big | \big | \phi ( x ) - \beta \big | \big | _ { 2 } .\tag{27}
$$

This is analogous to an AB-join against a normal reference series [21]. Gaussian overlap-add from Eq. (20) maps window distances to $\tilde { d } _ { t }$ . Stage 2 adds them to the base score,

$$
s ^ { ( 2 ) } = u n i t { \left( s ^ { ( 1 ) } \right) } + \beta u n i t { \left( \tilde { d } \right) } , \qquad \beta = 1 .\tag{28}
$$

## 3.7.2 Density-Based Novelty Calibration (Stage 3)

Raw distance can be high in a valid but sparsely sampled normal mode. Following the local-density idea of LOF [18], Stage 3 scales novelty by the bank’s nearby spacing. Let $\delta _ { \beta }$ be the leave-one-out nearest-neighbor distance of bank element $\beta ,$ and let $\dot { \mathcal { N } _ { k } } ( x )$ contain the k nearest bank elements. Calibrated novelty is defined in Eq. (29), with $k = 4$ and $\alpha = 0 . 2 5$

$$
\hat { d } ( x ) = \frac { d ( x ) } { \alpha \cdot \bar { \delta } _ { k } ( x ) + ( 1 - \alpha ) \cdot \mathrm { m e d i a n } _ { \beta \in B } \delta _ { \beta } } , \qquad \bar { \delta } _ { k } ( x ) = \frac { 1 } { k } \sum _ { \beta \in \mathcal { N } _ { k } ( x ) } \delta _ { \beta } .\tag{29}
$$

The global median prevents the denominator from collapsing when local values approach zero. $\operatorname { A t } \alpha = 0 ,$ , the calculation is proportional to raw novelty, preserving its ranking after rescaling. Although $\alpha \in [ 0 , 1 ]$ is recommended to maintain a positive denominator, larger values are supported through a small positive numerical floor that prevents zero or negative denominators. The deployed value $\alpha = 0 . 2 5$ provides a conservative calibration. The Stage-3 score is:

$$
s ^ { ( 3 ) } = u n i t { \left( s ^ { ( 1 ) } \right) } + \beta u n i t { \left( \hat { \tilde { d } } \right) } .\tag{30}
$$

Local spacing is large in a sparse but valid normal region and small in a densely sampled one. Dividing by that spacing prevents a legitimate sparse mode from appearing novel solely because its neighbors are far apart. The global component limits unstable ratios when several bank embeddings nearly coincide.

Simple unit rescaling changes the scale of raw distance but does not remove a high-distance plateau caused by sparse sampling. Density calibration changes the relative values of windows in different parts of the normal manifold. It therefore addresses a distinct problem from the per-series scaling used for score fusion.

## 3.7.3 Local Temporal Contrast (Stage 4)

On long non-stationary series, a global score baseline can drift. Stage 4 compares each score with a running median over $W _ { \gamma }$ samples and adds the resulting contrast, defined in Eq. (31).

$$
r _ { t } ~ = ~ \frac { s _ { t } ^ { ( 3 ) } } { m e d _ { W _ { \gamma } } \bigl ( s ^ { ( 3 ) } \bigr ) _ { t } + \rho _ { \gamma } \operatorname * { m e d i a n } \bigl ( s ^ { ( 3 ) } \bigr ) + \epsilon } , ~ s ^ { ( 4 ) } ~ = ~ s ^ { ( 3 ) } + \gamma \cdot u n i t _ { q } \bigl ( r \bigr ) .\tag{31}
$$

Here med<sub>W</sub> is a running median, $u n i t _ { q }$ bounds the tail at the same quantile of the active support that unit divides by — the bound and the divisor are one quantity, and the clip is applied only once that scale is known to be usable — and $\rho _ { \gamma } = 1 0 ^ { - 3 }$ stabilizes the denominator. The stated coefficient is $\gamma = 1 . 5 . \mathrm { A t } \gamma = 0$ , Stage 4 equals Stage 3.

The ratio $r _ { t }$ highlights a point that rises above its own neighborhood on a non-stationary recording. Adding it, instead of multiplying the score by it, keeps the Stage-3 signal when a long anomaly also raises its local median, which a multiplicative correction would erase. The contrast is computed on the complete Stage-3 score, so it responds to masked error and latent novelty together. Winsorization limits the effect of near-zero denominators, and $W _ { \gamma } = \nu L$ ties the context width to the window length.

## 3.7.4 Criterion-Based Channel Selection and Novelty Scoring (Stage 5)

The high-variance channels of Eq. (26) can miss anomalies in quiet channels. Stage 5 therefore takes the union $\mathcal { P }$ of the top min $( C , \chi ^ { \prime } )$ channels selected by each of six label-free criteria. Let $a = X _ { 1 : T _ { \mathrm { t } } }$ be the training part and $b = X _ { T _ { \mathrm { t r } } + 1 : T }$ the remainder. The criteria are

$$
\begin{array} { r l r } & { \mathrm { V A R } _ { c } = \mathrm { s t d } ( a _ { c } ) , } & { \mathrm { P } ^ { 9 9 } c = \frac { 1 } { T } \sum _ { t } \mathcal { k } \big [ x _ { t , c } > q _ { 0 . 9 9 } \big ( a _ { c } \big ) \big ] , } \\ & { \mathrm { R A N G E } _ { c } = \frac { \mathrm { p t p } \big ( b _ { c } \big ) + \epsilon } { \mathrm { p t p } \big ( a _ { c } \big ) + \epsilon } , } & { \mathrm { Z M E A N } _ { c } = \frac { 1 } { T } \sum _ { t } \frac { \big | x _ { t , c } - \overline { { a } } _ { c } \big | } { \mathrm { s t d } ( a _ { c } ) + \epsilon } , } \\ & { \mathrm { S S T D } _ { c } = \frac { \mathrm { s t d } ( b _ { c } ) + \epsilon } { \mathrm { s t d } ( a _ { c } ) + \epsilon } , } & { \mathrm { K S } _ { c } = \operatorname* { s u p } _ { v } \big | F _ { a _ { c } } ( v ) - F _ { b _ { c } } ( v ) \big | . } \end{array}\tag{32}
$$

The last criterion is a two-sample Kolmogorov–Smirnov statistic [54] computed on a fixed-seed subsample. The criteria use the known split point $T _ { \mathrm { t r } }$ and no anomaly labels.

The variance criterion reflects training activity. The remaining criteria capture high-tail frequency, range, mean absolute deviation, dispersion and distribution changes between the training prefix and the unlabeled series. Because several inspect the complete remainder $^ { b , }$ channel selection is an offline step. Their union allows a low-variance channel to enter the pool even when it was absent from the concatenated embedding of Stage 2.

For each $c \in { \mathcal { P } } .$ , the normalized embedding $\phi _ { c } ( x ) = \mathrm { v e c } ( h _ { c , : , : } ) / \big ( \| \mathrm { v e c } ( h _ { c , : , : } ) \| _ { 2 } + \epsilon \big ) \in R ^ { N D }$ is scored by the density-calibrated novelty of Eq. (29), using a separate bank $\boldsymbol { B } _ { c }$ of that channel’s training-window embeddings and their leave-one-out distances. Overlap-add projects it to time steps. The rescaled channel scores are averaged,

$$
A _ { t } \ = \ \frac { 1 } { | \mathcal { P } | } \sum _ { c \in \mathcal { P } } u n i t \big ( \hat { \tilde { d } } _ { c } \big ) _ { t } .\tag{33}
$$

The pool evidence enters before contrast is recomputed on the combined score:

$$
s ^ { ( 5 \mathrm { a } ) } = s ^ { ( 3 ) } + \delta \cdot u n i t ( A ) , \qquad \delta = 1 . 5 ,\tag{34}
$$

$$
s ^ { \mathrm { L E a p } } = s ^ { \left( \mathrm { 5 a } \right) } + \gamma \cdot u n i t _ { q } \Bigl ( s ^ { \left( \mathrm { 5 a } \right) } \oslash \left( m e d _ { W _ { \gamma } } ( s ^ { \left( \mathrm { 5 a } \right) } ) + \rho _ { \gamma } \mathrm { m e d i a n } ( s ^ { \left( \mathrm { 5 a } \right) } ) + \epsilon \right) \Bigr ) .\tag{35}
$$

Here ⊘ denotes element-wise division. The pool supplements the concatenated embedding, as replacing it degraded individual-series performance. At δ = 0, Eq. (35) reduces to $s ^ { ( 4 ) }$ . Scoring each pooled channel independently preserves its novelty when only one sensor departs from normal patterns, helping series where the high-variance channels in Eq. (26) obscure the anomaly. Contrast is recomputed after adding this term to capture the complete Stage-5 evidence.

## 3.7.5 Five-Stage Score Summary

Table 2 summarizes the five scores and the coefficient that removes each successive mechanism for ablation. Further configuration details are provided in Section 4.2.

Table 2: The five LEAP scores and the coefficient that removes each successive mechanism for ablation. Setting $\beta = 0$ preserves the Stage-1 ranking after unit rescaling.
<table><tr><td>Stage</td><td>Integrated Mechanism</td><td>Score</td><td>Reduction Condition</td></tr><tr><td>1</td><td>Masked-patch prediction + local moments</td><td>Eq. (24)</td><td></td></tr><tr><td>2</td><td>Latent shape novelty</td><td>Eq. (28)</td><td> $\beta = 0$ </td></tr><tr><td>3</td><td>Local-density calibration of novelty</td><td>Eq. (30)</td><td> $\alpha = 0$ </td></tr><tr><td>4</td><td>Local temporal contrast</td><td>Eq. (31)</td><td> $\gamma = 0$ </td></tr><tr><td>5</td><td>Per-channel novelty over criterion pool</td><td>Eq. (35)</td><td> $\delta = 0$ </td></tr></table>

## 4 Experimental Framework

This section describes the benchmarks, the model configuration, the evaluation metrics, the tuning protocol and the computational environment, so that every reported result can be reproduced.

## 4.1 Benchmarks

Experimental evaluation uses the TSB-AD benchmark [12], comprising 1,070 curated time series from 40 public datasets with human-verified anomaly labels. It includes 870 univariate series (TSB-AD-U) and 200 multivariate series (TSB-AD-M), with separate tuning and held-out evaluation subsets. We use 48/350 tuning/evaluation series from 23 datasets and nine application domains for TSB-AD-U, and 20/180 from 17 datasets and six domains for TSB-AD-M The detailed composition of the evaluation sets is reported in Table D.1 in Appendix D.

The evaluation sets are deliberately heterogeneous: TSB-AD-M spans 2–248 channels and from a few hundred to over half a million time steps, while TSB-AD-U covers a broad range of univariate signals, including web-service monitoring traces. Each series specifies an anomaly-free training prefix. Semi-supervised detectors, including LEAP, are trained exclusively on this prefix and then score the complete series, whereas unsupervised methods operate directly on the full series. Ground-truth labels are never used for fitting or scoring, and metrics are computed per series and averaged uniformly so that each series contributes equally regardless of length.

## 4.2 Hyperparameters

Table E.1 in Appendix E lists the complete configuration. Three groups can be distinguished. Structural hyperparameters (p, D, H, B, G) define the architecture and are fixed a priori at conventional values rather than searched. Adaptive hyperparameters $( L , \Delta , W _ { s } , W _ { m } , \kappa , W _ { \gamma } )$ are instead derived per series from the estimated dominant period or patch length, allowing a single configuration to accommodate series with sampling rates that differ by orders of magnitude

Fusion hyperparameters $( w _ { \ell } , \beta , \alpha , k , \gamma , \delta , \chi , \chi ^ { \prime } )$ are the coefficients of the additive evidence layers and are the only ones selected on the tuning subset (Section 4.4). The architecture uses patches of $p = 1 6$ samples, width $D = 1 2 8$ $H = 8$ attention heads, $B = 4$ encoder blocks and $G = 4$ masking groups with blocks of $b = 1$ patch, and the fusion coefficients take the values given with their equations in Section 3.

A unified hyperparameter configuration is used across all 180 multivariate and, separately, all 350 univariate evaluation series, without per-series or per-dataset tuning. The univariate configuration (Table E.1) adapts LEAP to single-channel inputs $( C = 1 )$ in three ways.

First, the event-scale smoothing of Eq. (21) is replaced by a multi-scale readout: masked-prediction errors are smoothed at widths $W _ { s } \in \{ 1 , 8 , 3 2 , 1 2 8 \}$ and local moments (Section 3.6.4) are computed over $W _ { m } \in \{ 1 6 , 6 4 , 2 5 6 \}$ . Each scale is rescaled to unit range, the maximum response is retained per time step, and tolerance dilation is applied. This avoids a fixed width, whose optimum varied independently of series length, window size, and training duration.

Second, local contrast is disabled $( \gamma = 0 )$ , as it adds no signal beyond multi-scale scoring. Third, channel-pool novelty is disabled $( \delta = 0 )$ , since the pool in Eq. (32) reduces to one channel, making Stage 5 redundant with Stage 3. Latent novelty thus embeds that channel rather than the cross-channel concatenation of Eq. (26), with density calibration unchanged.

All experiments use three fixed random seeds (2024, 2025, and 2026), consistently applied to PyTorch, NumPy, Python’s random module, and the masking generators. Deterministic cuDNN is enabled where supported, with computations performed in FP32 by default.

## 4.3 Performance Metrics

The evaluation adopts the TSB-AD metric suite without modification and reports nine metrics per series.

VUS-ROC and VUS-PR [13] are threshold-free and range-aware. A buffer of width w is placed around each labeled anomalous region, inside which a detection receives a weight that decays with its distance to the region. The ROC or PR curve is computed for every $w \in \{ 0 , \ldots , 2 \ell _ { \mathrm { v u s } } \}$ and the resulting surface is integrated,

$$
\mathrm { V U S - P R } \ = \ \frac { 1 } { 2 \ell _ { \mathrm { v u s } } } \int _ { 0 } ^ { 2 \ell _ { \mathrm { v u s } } } \mathrm { A U C - P R } ( w ) \mathrm { d } w ,\tag{36}
$$

so the measure depends on neither the threshold nor the buffer width. The half-width $\ell _ { \mathrm { v u s } }$ is the dominant period of the series, estimated from the autocorrelation of the first channel with the peak-based rule of the benchmark and capped at 125 when the estimate exceeds 300. Following Liu [12], VUS-PR is the primary measure of this paper. It is threshold-free, tolerant to small temporal displacements and sensitive to class imbalance, and it showed the strongest correlation with human judgment in the meta-evaluation of the benchmark.

The remaining seven metrics are reported for comparability with the benchmark. AUC-ROC and AUC-PR are point-wise and threshold-free, so a detection shifted by a few samples is fully penalized, and AUC-PR is the more informative of the two because anomalies are rare. Standard-F1 is the point-wise $F _ { 1 }$ at the oracle threshold, and PA-F1 counts a whole labeled segment as detected when any of its points is flagged. Event-based-F1 treats each anomalous segment as one unit, R-based-F1 [44] rewards both the existence and the overlap of detections, and Affiliation-F [45] relies on the temporal distance between predicted and labeled events. Following Kim [10], PA-F1 is treated as diagnostic only, since even a random score can beat a trained detector under it, and no claim in this paper rests on it.

## 4.4 Tuning Strategy

The tuning protocol was designed to make the reported held-out figures defensible under the criticisms reviewed in Section 2.4. Five rules were enforced.

Strict Separation of Tuning and Evaluation Sets All architectural choices and hyperparameter selections were conducted exclusively on the benchmark’s designated tuning subsets (20 multivariate and 48 univariate series) or internal validation splits, without accessing held-out evaluation sequences. The held-out evaluation sets were reserved strictly for generating the final benchmark results in Section 5.

Stage-Wise Greedy Selection Each stage in the ablation hierarchy (Table 2) was fixed sequentially before introducing subsequent components, searching only the scaling coefficient governing the newly integrated mechanism. This greedy optimization strategy constrains search-space complexity to scale linearly with the number of stages rather than exponentially. While higher-order parameter interactions are not explicitly explored, this trade-off is justified because each stage incorporates a physically motivated domain mechanism rather than an unconstrained hyperparameter.

Paired Hypothesis Testing Because per-series VUS-PR distributions are paired and non-Gaussian, architectural modifications were evaluated using two-sided Wilcoxon signed-rank tests computed over per-series differences. Wins, losses, and ties were tracked alongside mean differences. A modification was accepted only if it yielded a positive mean difference and achieved statistical significance $( \alpha = 0 . 0 5 )$ . Where multiple variants were tested, p-values are descriptive; stage-wise increments are also reported with Bonferroni correction (Appendix G.1).

Drift Control Every stage was compared with a recomputed reference run rather than with a previously recorded value, so that changes in library versions, hardware or random state could not be mistaken for improvements, and these recomputations agree with the recorded values to six decimal places.

Mechanistic Transparency over Metric Optimization When two hyperparameter configurations yielded statistically indistinguishable performance on the tuning set, the variant with the clearer theoretical foundation was retained. For instance, the relative context rule $W _ { \gamma } = 6 \bar { 4 } L$ , which ties the contrast context to the window length L and hence to the dominant period of each series, was selected over an absolute window of 32,000 samples. Similarly, settings that raised overall mean metrics at the cost of severe performance degradation on a substantial minority of series were systematically rejected. Full hardware and software specifications are detailed in Appendix F.

## 5 Experimental Results and Analysis

This section compares LEAP with the state of the art on both tracks and isolates the mechanisms behind its performance through ablation, sensitivity and stratified analyses. All paired tests are computed on per-series values rather than on aggregates: baseline comparisons use three-seed means on the held-out series, and the sensitivity analyses use the tuning set.

## 5.1 Performance on TSB-AD-M

Table 3 compares LEAP with 21 baseline anomaly detectors on the 180 held-out multivariate series. Sixteen of the twenty-one baselines were re-run locally from their released source code using the same TSB-AD evaluation protocol, file list, and three random seeds (2024, 2025, and 2026). The table reports published values alongside locally measured means and sample standard deviations where available. LEAP was evaluated under the same protocol, with its per-series results archived alongside those of the locally evaluated baselines.

The comparison distinguishes published results from standardized local evaluations. Five baselines (VACE, PaAno+, MMPAD, DADA, and TranAD) are represented only by published values because their source code was unavailable. Among the 16 locally evaluated baselines, substantial discrepancies from published scores occur in both directions: KAN-AD drops from approximately 0.41 to $0 . 3 2 4 1 \pm 0 . 0 0 0 7$ , while USAD and TimesNet increase from 0.30 and 0.19 to $0 . 4 0 2 8 \pm \bar { 0 . 0 0 1 9 }$ and $0 . 3 8 7 4 \pm \mathrm { { \dot { 0 } } . 0 0 2 2 }$ , respectively. PaAno changes less, from 0.43 to $0 . 4 1 6 1 \pm 0 . 0 0 6 4$ . These discrepancies highlight the limitations of ranking methods solely by published scores.

LEAP achieves the highest locally measured VUS-PR $( 0 . 5 1 7 9 \pm 0 . 0 0 2 2 ;$ ; Table 3), outperforming the strongest measured baseline, PaAno (0.4161), by 24.5%. It also exceeds published PaAno (0.43) by 20.4% and the best published baseline, VACE (0.48), by 7.9%, although these cross-protocol comparisons should be interpreted cautiously. The absolute margins over $\mathrm { P a A n o _ { P A I } }$ , USAD, and measured KAN-AD are +0.1134, +0.1151, and +0.1938, respectively. The close agreement with the single-seed ablation result (0.5171; Table G.1) is consistent with the low variability across seeds. Baselines represented only by published results are excluded from subsequent paired statistical analyses.

Aggregate averages can obscure substantial differences across datasets: per-series VUS-PR scores exhibit a strongly bimodal distribution, with median scores $\leq 0 . 1 6$ for all six methods on GHL but $\geq 0 . 7 1$ on Exathlon and TAO. This motivates examining per-series distributions (Figure 3) and paired statistical tests, alongside the five-stage ablation in Appendix G. A Bayesian signed-rank test [55] compares LEAP with the five strongest locally measured baselines (PaAno, $\mathrm { P a A n o } _ { \mathrm { P A I } }$ , USAD, xLSTMAD-R, and TimesNet), using three-seed mean VUS-PR scores across the 180 paired series and a ROPE of ±0.05. As shown in Figure 2, LEAP achieves a posterior probability of superiority above 99.9% in all five comparisons, with negligible posterior probability of any competitor outperforming it.

The nine-metric comparison in Table 4 further supports this result. LEAP achieves the highest score on eight of nine metrics: AUC-ROC (0.8410), AUC-PR (0.4794), VUS-ROC (0.8535), VUS-PR (0.5179), Standard-F1 (0.5427), Event based-F1 (0.6285), R-based-F1 (0.5462), and Affiliation-F (0.8821). The only exception is PA-F1, where $\mathrm { P a A n o _ { P A I } }$ reaches 0.7684 versus 0.7290 for LEAP (0.0394 higher). This divergence reflects the point-wise versus range-aware trade-off discussed in Section 2.4: consistent with Section 4.3 and Kim [10], PA-F1 credits an entire ground-truth anomaly segment when a single anomalous point is detected, favoring PaAno ’s point-level calibration. PA-F1 should therefore be interpreted as a diagnostic indicator of metric behavior rather than a definitive measure of detector quality. TimesNet shows a related discrepancy: although it ranks last among the six methods on threshold-free and point-wise metrics (AUC-ROC, AUC-PR, VUS-ROC, VUS-PR, and Standard-F1), it ranks third on PA-F1 (0.6644) and fourth on Event-based-F1, R-based-F1, and Affiliation-F.

Table 3: Mean VUS-PR on the TSB-AD-M held-out set (180 series). Published and locally measured results are reported where available; measured values are mean ± sample standard deviation over three seeds. Source: reference from which the published value is taken; for rows without a published value, the reference describing the method. Daggers (†) designate standalone published detectors outside the standard TSB-AD benchmark suite.
<table><tr><td>Method</td><td>Family</td><td>Published</td><td>Measured (this work)</td><td>Source</td></tr><tr><td>LEAP (this work)</td><td>Masked + repr.</td><td></td><td> $\mathbf { 0 . 5 1 7 9 \pm 0 . 0 0 2 2 }$ </td><td>this work</td></tr><tr><td>VACE†</td><td>Representation</td><td>0.48</td><td></td><td>[14]</td></tr><tr><td> $\mathrm { P a A n o } ^ { \dagger }$ </td><td>Representation</td><td>0.43</td><td> $0 . 4 1 6 1 \pm 0 . 0 0 6 4$ </td><td>[8]</td></tr><tr><td> $\mathrm { P a A n o } + ^ { \dagger }$ </td><td>Representation</td><td>0.42</td><td></td><td>[39]</td></tr><tr><td> $\mathrm { P a A n o } _ { \mathrm { P A I } }$ </td><td>Representation</td><td></td><td> $0 . 4 0 4 5 \pm 0 . 0 0 2 4$ </td><td>[40]</td></tr><tr><td> ${ \mathrm { K A N - A D } } ^ { \dagger }$ </td><td>Forecasting</td><td>0.41</td><td> $0 . 3 2 4 1 \pm 0 . 0 0 0 7$ </td><td>[15]</td></tr><tr><td> $\mathbf { \Delta x L S T M A D - R } ^ { \dagger }$ </td><td>Reconstruction</td><td>0.37</td><td> $0 . 3 8 8 1 \pm 0 . 0 0 1 1$ </td><td>[43]</td></tr><tr><td> $\mathbf { M N P A D } ^ { \dagger }$ </td><td>Discord</td><td>0.35</td><td></td><td>[46]</td></tr><tr><td> $\mathrm { \ x L S T M A D  – F ^ { \dagger } }$ </td><td>Forecasting</td><td>0.35</td><td> $0 . 3 4 9 0 \pm 0 . 0 0 1 9$ </td><td>[43]</td></tr><tr><td> $\mathrm { C r o s s A D ^ { \dag } }$ </td><td>Reconstruction</td><td>0.33</td><td> $0 . 3 3 5 5 \pm 0 . 0 0 1 4$ </td><td>[16]</td></tr><tr><td>CNN</td><td>Forecasting</td><td>0.31</td><td> $0 . 3 5 3 8 \pm 0 . 0 0 0 7$ </td><td>[12]</td></tr><tr><td>OmniAnomaly</td><td>Reconstruction</td><td>0.31</td><td> $0 . 3 1 5 2 \pm 0 . 0 0 0 1$ </td><td>[14]</td></tr><tr><td>DADA†</td><td>Reconstruction</td><td>0.31</td><td></td><td>[42]</td></tr><tr><td>PCA</td><td>Statistical</td><td>0.31</td><td> $0 . 2 6 9 1 \pm 0 . 0 0 0 0$ </td><td>[12]</td></tr><tr><td>USAD</td><td>Reconstruction</td><td>0.30</td><td> $0 . 4 0 2 8 \pm 0 . 0 0 1 9$ </td><td>[14]</td></tr><tr><td>CATCH†</td><td>Reconstruction</td><td>0.30</td><td> $0 . 3 3 0 3 \pm 0 . 0 0 0 6$ </td><td>[41]</td></tr><tr><td>IForest</td><td>Statistical</td><td>0.20</td><td> $0 . 2 4 6 9 \pm 0 . 0 0 2 4$ </td><td>[14]</td></tr><tr><td>TimesNet</td><td>Reconstruction</td><td>0.19</td><td> $0 . 3 8 7 4 \pm 0 . 0 0 2 2$ </td><td>[8]</td></tr><tr><td>TranAD</td><td>Transformer</td><td>0.18</td><td></td><td>[39]</td></tr><tr><td> $\mathrm { C A R L A ^ { \dagger } }$ </td><td>Contrastive SSL</td><td></td><td> $0 . 1 3 5 1 \pm 0 . 0 2 0 4$ </td><td>[32]</td></tr><tr><td>Anomaly Transf.</td><td>Transformer</td><td>0.12</td><td> $0 . 1 1 4 7 \pm 0 . 0 0 1 8$ </td><td>[8]</td></tr><tr><td>DCdetector</td><td>Transformer</td><td>0.09</td><td> $0 . 0 9 6 2 \pm 0 . 0 0 0 5$ </td><td>[14]</td></tr></table>

![](images/fa2675f1cc47370ef75598389b72ec1d8d8423fbd64b46e371df7a80e4ec1d20.jpg)  
Figure 2: Bayesian signed-rank test comparing LEAP against each of the five measured competitors of Table $^ { 3 , }$ with a region of practical equivalence of ±0.05. Green is the posterior probability that LEAP is superior, gray the probability of practical equivalence, and red the probability that the competitor is superior.

Table 4: Nine-metric evaluation on the TSB-AD-M held-out set (180 series) for the top five measured competitors. Values are mean ± sample standard deviation over three seeds; the best value in each row is bold.
<table><tr><td>Metric</td><td>PaAno</td><td>PaAnoPAI</td><td>USAD</td><td>xLSTMAD-R</td><td>TimesNet</td><td>LEAP</td></tr><tr><td>AUC-ROC</td><td> $0 . 7 6 1 7 \pm 0 . 0 0 4 8$ </td><td> $0 . 7 8 3 3 \pm 0 . 0 0 0 8$ </td><td> $0 . 7 3 6 5 \pm 0 . 0 0 0 5$ </td><td> $0 . 7 3 8 7 \pm 0 . 0 0 1 6$ </td><td> $0 . 6 6 8 1 \pm 0 . 0 0 1 6$ </td><td> $\mathbf { 0 . 8 4 1 0 \pm 0 . 0 0 2 8 }$ </td></tr><tr><td>AUC-PR</td><td> $0 . 3 7 6 3 \pm 0 . 0 0 5 6$ </td><td> $0 . 3 9 7 9 \pm 0 . 0 0 0 8$ </td><td> $0 . 3 6 1 7 \pm 0 . 0 0 0 8$ </td><td> $0 . 3 4 1 3 \pm 0 . 0 0 1 7$ </td><td> $0 . 3 3 5 8 \pm 0 . 0 0 1 6$ </td><td> $\mathbf { 0 . 4 7 9 4 \pm 0 . 0 0 3 1 }$ </td></tr><tr><td>VUS-ROC</td><td> $0 . 7 8 8 7 \pm 0 . 0 0 4 4$ </td><td> $0 . 8 0 5 7 \pm 0 . 0 0 0 7$ </td><td> $0 . 7 6 5 9 \pm 0 . 0 0 1 2$ </td><td> $0 . 7 7 7 3 \pm 0 . 0 0 0 7$ </td><td> $0 . 7 0 9 0 \pm 0 . 0 0 2 3$ </td><td> $\mathbf { 0 . 8 5 3 5 \pm 0 . 0 0 3 6 }$ </td></tr><tr><td>VUS-PR</td><td> $0 . 4 1 6 1 \pm 0 . 0 0 6 4$ </td><td> $0 . 4 0 4 5 \pm 0 . 0 0 2 4$ </td><td> $0 . 4 0 2 8 \pm 0 . 0 0 1 9$ </td><td> $0 . 3 8 8 1 \pm 0 . 0 0 1 1$ </td><td> $0 . 3 8 7 4 \pm 0 . 0 0 2 2$ </td><td> $\mathbf { 0 . 5 1 7 9 \pm 0 . 0 0 2 2 }$ </td></tr><tr><td>Standard-F1</td><td> $0 . 4 2 7 1 \pm 0 . 0 0 4 6$ </td><td> $0 . 4 3 6 6 \pm 0 . 0 0 2 0$ </td><td> $0 . 4 3 3 3 \pm 0 . 0 0 0 7$ </td><td> $0 . 4 2 2 1 \pm 0 . 0 0 0 7$ </td><td> $0 . 4 0 9 8 \pm 0 . 0 0 1 7$ </td><td> $\mathbf { 0 . 5 4 2 7 \pm 0 . 0 0 1 7 }$ </td></tr><tr><td>PA-F1</td><td> $0 . 6 5 2 1 \pm 0 . 0 0 4 7$ </td><td> $\mathbf { 0 . 7 6 8 4 \pm 0 . 0 0 1 5 }$ </td><td> $0 . 6 2 4 8 \pm 0 . 0 0 3 3$ </td><td> $0 . 6 2 1 6 \pm 0 . 0 0 5 7$ </td><td> $0 . 6 6 4 4 \pm 0 . 0 0 1 5$ </td><td> $0 . 7 2 9 0 \pm 0 . 0 0 2 0$ </td></tr><tr><td>Event-based-F1</td><td> $0 . 5 1 2 4 \pm 0 . 0 0 6 1$ </td><td> $0 . 6 0 6 4 \pm 0 . 0 0 1 8$ </td><td> $0 . 5 1 9 8 \pm 0 . 0 0 1 5$ </td><td> $0 . 5 3 3 9 \pm 0 . 0 0 5 6$ </td><td> $0 . 5 3 2 3 \pm 0 . 0 0 4 3$ </td><td> $\mathbf { 0 . 6 2 8 5 \pm 0 . 0 0 4 5 }$ </td></tr><tr><td>R-based-F1</td><td> $0 . 4 0 6 3 \pm 0 . 0 1 2 4$ </td><td> $0 . 3 8 3 6 \pm 0 . 0 0 3 2$ </td><td> $0 . 4 2 2 9 \pm 0 . 0 0 1 1$ </td><td> $0 . 3 8 3 5 \pm 0 . 0 0 0 4$ </td><td> $0 . 4 0 3 1 \pm 0 . 0 0 3 6$ </td><td> $\mathbf { 0 . 5 4 6 2 \pm 0 . 0 0 2 4 }$ </td></tr><tr><td>Affiliation-F</td><td> $0 . 8 5 1 6 \pm 0 . 0 0 1 8$ </td><td> $0 . 8 6 5 9 \pm 0 . 0 0 1 3$ </td><td> $0 . 8 3 7 1 \pm 0 . 0 0 0 3$ </td><td> $0 . 8 3 3 2 \pm 0 . 0 0 1 2$ </td><td> $0 . 8 4 5 2 \pm 0 . 0 0 3 4$ </td><td> $\mathbf { 0 . 8 8 2 1 \pm 0 . 0 0 2 0 }$ </td></tr></table>

Figure 3 shows the per-series VUS-PR distributions across ten source datasets, covering 169 of the 180 series. LEAP obtains the highest mean on seven datasets (SVDB, SMAP, Exathlon, GHL, MSL, MITDB, and CATSv2), while individual baselines perform better on SMD, TAO, and OPPORTUNITY. The largest baseline advantage occurs on OPPORTUNITY, where USAD exceeds LEAP by 0.2808 in mean VUS-PR. These dataset-specific differences show that the overall ranking reflects performance across heterogeneous domains rather than uniform superiority on every dataset. Appendix H extends this analysis with mean VUS-PR by source dataset and application domain (Tables H.1 and H.2), by dominant anomaly type (Tables H.3 and H.4), and by series characteristics (Tables H.5 and H.6).

![](images/5c34c97f5be546b496ffb22164b3e822af6b5e5e6d7a1ba909980bad891d6b8f.jpg)  
Figure 3: Per-series VUS-PR distributions on TSB-AD-M eval set for LEAP and the top five measured competitors (Table 3), pooled across three random seeds and grouped by source dataset (restricted to 10 datasets with ≥ 5 series; 169 of 180 series total represented). Boxes indicate median and IQR; whiskers extend to $1 . 5 \times \mathrm { I Q R } ,$ , with outliers plotted individually.

## 5.2 Sensitivity to Fusion Coefficients

To assess generalization across heterogeneous time series, we swept $w _ { \ell } , \beta , \alpha , \gamma ,$ and $\delta$ over 20 tuning series from 11 dataset families. Figure 4 and Table 5 show that cross-series heterogeneity outweighs hyperparameter sensitivity:

deployed VUS-PR ranges from 0.049 to 0.999 (standard deviation = 0.36), while selecting each coefficient’s grid optimum improves the mean by at most 0.0384. The deployed values are therefore close to optimal for four coefficients, with α showing greater sensitivity.

Table 5: Sensitivity of the fusion coefficients on the 20-series tuning set. Each parameter is varied independently while the remaining parameters are held at their deployed values $( w _ { \ell } = \beta = 1 . 0 , \alpha = 0 . 2 5 , \gamma = \delta = 1 . 5 )$ . Values $\alpha > 1$ exceed the recommended range of Eq. (29) and are included only to illustrate the risk of near-zero denominators.
<table><tr><td>Coefficient</td><td>Grid tested</td><td>Points</td><td>Deployed value</td><td>VUS-PR (deployed)</td><td>Grid optimum</td></tr><tr><td>we</td><td>0 to 2.0</td><td>6</td><td>1.0</td><td>0.6717</td><td> $0 . 6 7 1 7 \mathrm { a t } 1 . 5 ( \Delta < 0 . 0 0 0 1 )$ </td></tr><tr><td>β</td><td>0 to 2.0</td><td>6</td><td>1.0</td><td>0.6717</td><td> $0 . 6 7 6 1 \mathrm { a t } 0 ( \Delta = 0 . 0 0 4 4 )$ </td></tr><tr><td>α</td><td>0 to 2.0</td><td>8</td><td>0.25</td><td>0.6717</td><td> $0 . 7 1 0 1 \mathrm { a t } 1 . 0 ( \Delta = 0 . 0 3 8 4 )$ </td></tr><tr><td>γ</td><td>0 to 3.0</td><td>9</td><td>1.5</td><td>0.6717</td><td> $0 . 6 7 5 3 \mathrm { a t } 2 . 5 ( \Delta = 0 . 0 0 3 6 )$ </td></tr><tr><td>δ</td><td>0 to 3.0</td><td>9</td><td>1.5</td><td>0.6717</td><td> $0 . 6 7 2 6 \mathrm { a t } 2 . 0 ( \Delta = 0 . 0 0 0 9 )$ </td></tr></table>

![](images/f6beb7af456d80722fbf88f0b6c1797eb8782bc3f8c69de40b5e2b673a50de8b.jpg)  
Figure 4: VUS-PR sensitivity to the five fusion coefficients on the 20-series tuning set. Shaded regions denote one standard deviation across benchmark series.

All evaluations were restricted to the tuning set, with parameter selection verified through leave-one-series-out (LOSO) cross-validation rather than direct peak selection. The apparent gains for $w _ { \ell }$ and $\beta$ disappeared when a single favorable series was excluded, while α required excluding two. These influential series (011\_MSL, 062\_SMD, and 131\_OPPORTUNITY) highlight cross-series heterogeneity. Although $\alpha = 1 . 0$ achieves the highest aggregate VUS-PR (0.7101), its gains are concentrated in OPPORTUNITY and MITDB, while degrading MSL, SMD, and GHL. This optimum also lies near the denominator instability described in Section 3.7.2. The sweep extends to 2.0 with a numerical safeguard, although [0, 1] remains the recommended range. Thus, the deployed configuration retains $w _ { \ell } = \beta = 1 . 0$ and $\alpha = 0 . 2 5$ , prioritizing robustness over aggregate optima.

In contrast, the local contrast coefficient $\gamma$ and channel-pool novelty weight δ were the only parameters showing consistent gains. Both improved over their zero-value baselines across all 20 LOSO folds, with mean VUS-PR gains of $+ 0 . 0 1 8 ( 1 0 / 3 / 7 \mathrm { ~ W / L / T } , p \mathrm { - v a l u e } = 0 . 0 4 9 ) \mathrm { ~ a n d ~ } + 0 . 0 4 9 ( 1 3 / 4 / 3 , p \mathrm { - v a l u e } = 0 . 0 1 6 ) ,$ , respectively. Their deployed value of 1.5 is also close to the grid optima, differing by only 0.0036 for γ and 0.0009 for $\delta ,$ suggesting broad performance plateaus rather than isolated optima (Figure 4).

The analysis also examined combination rules and five backbone parameters (Appendix I). No alternative combination rule demonstrated a consistent or statistically significant advantage over the deployed add and softmax operators. Among backbone parameters, only patch length significantly affected performance: $p = 8 \ : ( \Delta = - 0 . 0 4 5 4 .$ , p-value $= 0 . 0 0 3 )$ and $p = \bar { 3 } 2 \ : ( \Delta = - 0 . 0 3 \dot { 4 } \bar { 7 } , p \mathrm { - v a l u e } = 0 . \bar { 0 } 1 4 )$ both degraded performance, whereas depth, width, masking groups, and mask block size showed no significant effects $( p \mathrm { - v a l u e s } \geq 0 . 2 2 5 )$ . These findings suggest that $\mathbf { L E A P } \mathbf { \dot { s } }$ gains stem primarily from its scoring and fusion mechanisms rather than backbone capacity tuning.

## 5.3 Results on TSB-AD-U

Table 6 compares LEAP with 23 baseline methods on the 350 held-out univariate series of TSB-AD-U, following the protocol described in Section 5.1.

LEAP achieves the best overall performance, with a measured VUS-PR of $0 . 6 1 9 4 \pm 0 . 0 0 1 6$ , outperforming all 20 locally evaluated baselines, including PaAno<sub>PAI</sub> (0.5839, $\Delta = + 0 . 0 3 5 5 , + 6 . 1 \% )$ , and PaAno+ (0.55). The advantage reaches 0.1029 over the standalone PaAno implementation (0.5165). Conversely, $\mathrm { P a A n o _ { P A I } }$ substantially outperforms standalone PaAno despite sharing the same method family, increasing VUS-PR by more than 0.06 under the same evaluation protocol and highlighting the impact of benchmark-specific variants and scoring configurations.

The KAN-AD discrepancy reported for TSB-AD-M in Section 5.1 is also observed on TSB-AD-U: its measured score $( 0 . 3 6 9 5 \pm 0 . 0 1 2 5 )$ is approximately 0.03 below the lower bound of the published range (0.40–0.43), underscoring the sensitivity of reported performance to evaluation implementation. Performance also varies across algorithmic families. Representation-based methods lead most baseline rankings, while PAI post-processing lifts PaAno to 0.58 VUS-PR, above the foundation models TSPulse (0.54), MOMENT and TimesFM (0.30–0.37). The Forecasting family ranges from PatchTST (0.4569) to NLinear (0.3333), limiting the predictive value of family-level taxonomy. Reconstruction and Statistical/Shape-based methods form a narrower middle tier (0.37–0.42), while the contrastive CARLA (0.1136) and pure Transformer baselines remain at the bottom, consistent with the multivariate results.

Table 6: Mean VUS-PR performance on TSB-AD-U. Published columns report published values (ranges indicate cited intervals), while Measured columns report local mean ± standard deviation, with dashes denoting unavailable entries. Method Family classifications extend Table A.1 with Foundation model (TSPulse, MOMENT, TimesFM) and Shape clustering (KShapeAD). Daggers (†) designate standalone published detectors outside the standard TSB-AD benchmark suite. Source: reference from which the published value is taken; for rows without a published value, the reference describing the method.
<table><tr><td>Method</td><td>Family</td><td>Published</td><td>Measured (this work)</td><td>Source</td></tr><tr><td>LEAP (this work)</td><td>Masked + repr.</td><td></td><td>0.6194 ± 0.0016</td><td>this work</td></tr><tr><td>PaAnoPAI</td><td>Representation</td><td>0.58</td><td>0.5839 ± 0.0016</td><td>[40]</td></tr><tr><td>PaAno+†</td><td>Representation</td><td>0.55</td><td></td><td>[39]</td></tr><tr><td>TSPulse†</td><td>Foundation model</td><td>0.54</td><td></td><td>[38]</td></tr><tr><td>PaAno†</td><td>Representation</td><td>0.51–0.53</td><td>0.5165 ± 0.0025</td><td>[8]</td></tr><tr><td>CrossAD†</td><td>Reconstruction</td><td>0.45</td><td></td><td>[16]</td></tr><tr><td>PatchTST</td><td>Forecasting</td><td></td><td>0.4569 ± 0.0025</td><td>[12]</td></tr><tr><td>Sub-PCA</td><td>Statistical</td><td>0.42</td><td>0.4173 ± 0.0000</td><td>[46]</td></tr><tr><td>KAN-AD†</td><td>Forecasting</td><td>0.40-0.43</td><td>0.3695 ± 0.0125</td><td>[15]</td></tr><tr><td>USAD</td><td>Reconstruction</td><td></td><td>0.3959 ± 0.0017</td><td>[12]</td></tr><tr><td>FITS</td><td>Forecasting</td><td></td><td>0.3902 ± 0.0070</td><td>[12]</td></tr><tr><td>CNN</td><td>Forecasting</td><td></td><td>0.3823 ± 0.0068</td><td>[12]</td></tr><tr><td>KShapeAD</td><td>Shape clustering</td><td></td><td>0.3750 ± 0.0063</td><td>[12]</td></tr><tr><td>MOMENT-FT†</td><td>Foundation model</td><td></td><td>0.3707 ± 0.0012</td><td>[37]</td></tr><tr><td>TimesNet</td><td>Reconstruction</td><td></td><td>0.3707 ± 0.0044</td><td>[12]</td></tr><tr><td>MOMENT-ZS†</td><td>Foundation model</td><td></td><td>0.3662 ± 0.0009</td><td>[37]</td></tr><tr><td>iTransformer</td><td>Forecasting</td><td></td><td>0.3585 ± 0.0003</td><td>[12]</td></tr><tr><td>OFA</td><td>Forecasting</td><td></td><td>0.3581 ± 0.0029</td><td>[12]</td></tr><tr><td>DLinear</td><td>Forecasting</td><td></td><td>0.3493 ± 0.0017</td><td>[12]</td></tr><tr><td>NLinear</td><td>Forecasting</td><td></td><td>0.3333 ± 0.0015</td><td>[12]</td></tr><tr><td>TimesFM†</td><td>Foundation model</td><td></td><td>0.2964 ± 0.0000</td><td>[12]</td></tr><tr><td>CARLA†</td><td>Contrastive SSL</td><td></td><td>0.1136 ± 0.0012</td><td>[32]</td></tr><tr><td>Anomaly Transf.</td><td>Transformer</td><td></td><td>0.1035 ± 0.0049</td><td>[12]</td></tr><tr><td>DCdetector</td><td>Transformer</td><td></td><td>0.0884 ± 0.0006</td><td>[12]</td></tr></table>

Following Section 5.1, Figure 5 summarizes the Bayesian signed-rank comparisons between LEAP and the five strongest measured baselines across the 350 paired series (ROPE = ±0.05 VUS-PR). LEAP achieves a posterior probability of superiority above 99.9% against all five competitors (PaAno , PaAno, PatchTST, Sub-PCA, and USAD), with negligible posterior mass assigned to practical equivalence or competitor superiority. Against PaAno , LEAP wins on 202 of 350 series (57.7%), with a mean VUS-PR advantage of +0.0355.

Table 7 extends the comparison to nine evaluation metrics for the five strongest locally measured baselines, following the multivariate analysis in Table 4. LEAP leads in five metrics (AUC-ROC, VUS-ROC, VUS-PR, Standard-F1, and R-based-F1), while PaAno leads in the remaining four: AUC-PR (0.5506 vs. 0.5409), PA-F1 (0.8375 vs. 0.7447), Event-based-F1 (0.7532 vs. 0.6860), and Affiliation-F (0.9278 vs. 0.9194). These results indicate that LEAP’s advantage in VUS-PR does not extend uniformly across evaluation metrics.

The comparison with PaAno<sub>PAI</sub> highlights the importance of amplitude information. PAI restores magnitude sensitivity through post-hoc corrections [40], raising PaAno to 0.5839 VUS-PR, whereas LEAP preserves amplitude natively through unnormalized masked prediction and its local-moment module (Section 3.6.4), without requiring post-hoc correction. The remaining 15 locally evaluated baselines in Table 6 fall outside the ±0.05 ROPE relative to LEAP, so further pairwise tests are unnecessary.

Figure 6 shows that performance remains dataset-dependent, although TSB-AD-U exhibits more continuous variation than TSB-AD-M. LEAP is particularly competitive on UCR, where it achieves the highest mean VUS-PR (0.459)

![](images/e354576ca6be4d6982c7b245f68d2b1325e6eb9d469b73f62bee46b73af07eea.jpg)  
Figure 5: Bayesian signed-rank test comparing LEAP against the top five measured univariate competitors from Table 6.

Table 7: Nine-metric evaluation on the TSB-AD-U held-out set (350 series). Values are mean $\pm$ sample standard deviation across the three seeds; Sub-PCA is deterministic and therefore has zero spread. The six methods are evaluated locally, and the best value in each row is bold.
<table><tr><td>Metric</td><td>PaAnoPAI</td><td>PaAno</td><td>PatchTST</td><td>Sub-PCA</td><td>USAD</td><td>LEAP</td></tr><tr><td>AUC-ROC</td><td>0.9022±.0005</td><td>0.8663±.0005</td><td>0.7692±.0020</td><td>0.7143±.0000</td><td>0.7027±.0021</td><td>0.9128±.0005</td></tr><tr><td>AUC-PR</td><td>0.5506±.0015</td><td>0.4695±.0025</td><td>0.3676±.0024</td><td>0.3777±.0000</td><td>0.3631±.0013</td><td>0.5409±.0022</td></tr><tr><td>VUS-ROC</td><td>0.9280±.0002</td><td>0.8837±.0007</td><td>0.8235±.0004</td><td>0.7585±.0000</td><td>0.7493±.0016</td><td>0.9295±.0008</td></tr><tr><td>VUS-PR</td><td>0.5839±.0016</td><td>0.5165±.0025</td><td>0.4569±.0025</td><td>0.4173±.0000</td><td>0.3959±.0017</td><td>0.6194±.0016</td></tr><tr><td>Standard-F1</td><td>0.5789±.0027</td><td>0.5166±.0013</td><td>0.4330±.0025</td><td>0.4168±.0000</td><td>0.4029±.0014</td><td>0.5849±.0004</td></tr><tr><td>PA-F1</td><td>0.8375±.0019</td><td>0.6730±.0053</td><td>0.6822±.0023</td><td>0.5719±.0000</td><td>0.5431±.0019</td><td>0.7447±.0005</td></tr><tr><td>Event-based-F1</td><td>0.7532±.0029</td><td>0.5847±.0034</td><td>0.5575±.0038</td><td>0.5039±.0000</td><td>0.4660±.0010</td><td>0.6860±.0012</td></tr><tr><td>R-based-F1</td><td>0.5121±.0025</td><td>0.4879±.0031</td><td>0.4187±.0005</td><td>0.4097±.0000</td><td>0.3978±.0021</td><td>0.5500±.0024</td></tr><tr><td>Affiliation-F</td><td>0.9278±.0008</td><td>0.8959±.0014</td><td>0.8960±.0011</td><td>0.8529±.0000</td><td>0.8343±.0017</td><td>0.9194±.0011</td></tr></table>

compared with 0.402 for PaAno , and ranks first on seven of the fifteen datasets. YAHOO remains a clear weakness, with a median of 0.352 versus 0.929 for PaAno, whereas on OPPORTUNITY LEAP reaches a median of 0.734. Exathlon shows consistently high performance across methods, with LEAP reaching a median of 0.984 (mean 0.9626; Table H.2).

## 5.4 Qualitative Analysis

To quantify the behaviors in Figure 7, each evidence component was evaluated in isolation on its corresponding series using the standard VUS-PR protocol. Although some components are not standalone detectors (e.g., contrast depends on $s ^ { ( 1 ) } )$ , this isolates their contributions before score fusion:

• Base detector sufficiency in Exathlon (174\_Exathlon\_id\_1). Masked prediction already achieves 0.9998 VUS-PR $( s ^ { ( 1 ) } = 0 . 9 8 4 8 )$ , as the high-amplitude bursts closely match the anomaly patterns targeted by maskedpatch prediction. Subsequent layers reinforce this response, illustrating the regime where anomalies are clear departures from normal dynamics.

• Partial recovery via latent novelty in SMD (059\_SMD\_id\_3). Masked prediction is weak (0.1486 VUS-PR) because the brief, low-amplitude anomalies produce little prediction error. Latent k-nearest-neighbor novelty improves this to 0.2442, while local contrast reaches 0.3297, yielding 0.2669 after fusion. The recovery remains partial because some anomalous windows remain embedded in the normal latent manifold. The space can be seen geometrically in Figure 8.

• Decisive local-moment evidence in MSL (011\_MSL\_id\_10). Local-moment evidence reaches 0.8790 VUS-PR versus only 0.0936 for masked prediction, as 47 of 55 channels are near-constant command flags. It therefore captures abrupt state transitions missed by prediction, while the shape-based terms contribute little. This illustrates the failure mode motivating local-moment evidence.

• Synergy through per-channel pooling in Genesis (001\_Genesis\_id\_1). The components of $s ^ { ( 1 ) }$ conflict (masked prediction: 0.7004; level: 0.0580), reducing their combined score to 0.3966. Although pooling is weak in isolation (0.1102), its fusion with local contrast raises the final score to 0.7286. This suggests that complementary ranking errors can be corrected through additive fusion.

![](images/392a8fea1830451472c9a22b46c4866993c46f124caa10a458ac1faed30d8dfd.jpg)  
Figure 6: Per-series VUS-PR distributions on the TSB-AD-U held-out set for LEAP and the five competitors, grouped by source dataset. Results cover the fifteen datasets with at least eight series (328 of 350 series); boxes show the interquartile range and median, whiskers extend to 1.5× the interquartile range, and points beyond the whiskers are shown individually.

• Evidence-driven saturation failure in SMAP (166\_SMAP\_id\_23). This sequence is the key failure case in Stage 5’s leave-one-out robustness in Appendix G.1. The level component and $s ^ { ( 1 ) }$ effectively detect the sharp level shift (0.9753 and 0.8214), but latent novelty and channel pooling saturate after the transient. Their persistence beyond the 41-sample anomaly lowers precision to 0.3049, revealing a structural limitation of additive fusion: supplementary evidence can fail not by missing anomalies, but by remaining elevated after they have been detected.

## 5.5 Computational Cost

Table 8 reports the per-series wall-clock (fit plus scoring) and inference times for LEAP and the five strongest competitors on each track, pooled over the held-out series and three random seeds (540 evaluations for TSB-AD-M and 1050 for TSB-AD-U). Since runtime is heavy-tailed, mean ± standard deviation and the median (in parentheses) are reported. The divergence between these statistics is particularly pronounced for deep-learning methods, reflecting the disproportionate cost of high-dimensional and exceptionally long series.

Two patterns emerge. First, LEAP incurs substantially higher total runtime than most lightweight baselines. On TSB-AD-M, its mean wall-clock time (946 s) is 17–65 times that of PaAno, PaAno , USAD, and TimesNet; on TSB-AD-U (93.4 s), it is 4–24 times higher. The main exception is xLSTMAD-R, which is considerably slower than LEAP (3490 s vs. 946 s), although the gap narrows to 1.7 times when comparing medians (966 s vs. 566 s). Thus, LEAP trades computational efficiency for improved detection performance, particularly on the multivariate track.

Second, training accounts for most of LEAP’s runtime, with inference representing only 10.1% of mean wall-clock time on TSB-AD-M and 18.4% on TSB-AD-U. This profile is consistent with other learning-based methods, whereas non-learning approaches such as Sub-PCA incur negligible inference costs because most computation occurs during the initial matrix decomposition.

![](images/1010aef561b0db66460693959078674b87739d60302882c496df4337b6e094cc.jpg)  
Figure 7: Raw signal, ground-truth annotations, LEAP’s four score components $( s ^ { ( 1 ) }$ , which fuses masked prediction and local-moment evidence; latent novelty; per-channel pooling; local contrast) and the fused final score for five representative series. Signals correspond to the continuous channel with the largest standardized mean shift between anomalous and normal regions, avoiding maximum-variance selection that can favor binary telemetry flags.

![](images/8f9c503e962fe21d27d3726e067846c0e2906c1a0a635daae6aad8071a3ec071.jpg)  
Figure 8: Two-dimensional t-SNE projection of per-window latent embeddings (Eq. (26)) for 059\_SMD\_id\_3, showing training, normal test, and anomalous test windows.

Table 8: Per-series wall-clock (fit plus scoring) and inference time in seconds, pooled over the held-out series and three seeds (540 runs on TSB-AD-M, 1050 on TSB-AD-U). Mean ± std, with the median in parentheses; the mean training time is the difference between both columns. Dashes mark methods that are not among the five strongest competitors on that track.
<table><tr><td rowspan="2">Method</td><td colspan="2">TSB-AD-M</td><td colspan="2">TSB-AD-U</td></tr><tr><td>Wall (s)</td><td>Infer (s)</td><td>Wall (s)</td><td>Infer (s)</td></tr><tr><td>PaAno</td><td> $3 4 . 6 \pm 4 0 . 9 ( 1 3 . 7 )$ </td><td> $7 . 0 4 \pm 9 . 3 3 ( 2 . 8 3 )$ </td><td> $1 6 . 1 \pm 2 3 . 2 ( 8 . 0 5 )$ </td><td> $1 . 3 2 \pm 2 . 7 9 ( 0 . 4 2 )$ </td></tr><tr><td>PaAnoPAI</td><td> $1 4 . 5 \pm 8 . 0 5 ( 1 3 . 1 )$ </td><td> $1 . 5 3 \pm 3 . 0 3 ( 0 . 8 9 )$ </td><td> $1 3 . 0 \pm 6 . 7 6 \overset { \cdot } { ( 1 0 . 4 ) }$ </td><td> $0 . 5 6 \pm 1 . 1 9 ( 0 . 2 0 )$ </td></tr><tr><td>USAD</td><td> $2 4 . 6 \pm 2 7 . 6 ( 1 3 . 6 )$ </td><td> $7 . 4 1 \pm 7 . 9 0 ( 3 . 8 1 )$ </td><td> $9 . 7 4 \pm 1 4 . 7 ( 3 . 4 5 )$ </td><td> $2 . 5 2 \pm 5 . 4 0 ( 0 . 8 3 )$ </td></tr><tr><td>xLSTMAD-R</td><td> $3 4 9 0 \pm 6 1 1 2 ( 9 6 6 )$ </td><td> $1 5 4 \pm 1 6 9 ( 6 6 . 3 )$ </td><td></td><td></td></tr><tr><td>TimesNet</td><td> $5 5 . 4 \pm 6 8 . 6 ( 2 5 . 6 )$ </td><td> $1 1 . 4 \pm 1 4 . 3 ( 5 . 2 3 )$ </td><td></td><td></td></tr><tr><td>PatchTST</td><td></td><td></td><td> $2 1 . 1 \pm 3 1 . 6 ( 7 . 6 5 )$ </td><td> $5 . 6 8 \pm 1 2 . 1 ( 1 . 8 9 )$ </td></tr><tr><td>Sub-PCA</td><td></td><td></td><td> $3 . 8 4 \pm 1 9 . 8 ( 0 . 2 1 )$ </td><td> $0 . 0 0 \pm 0 . 0 0 ( 0 . 0 0 )$ </td></tr><tr><td>LEAP</td><td> $9 4 6 \pm 1 0 0 7 \ : ( 5 6 6 )$ </td><td> $9 5 . 8 \pm 1 9 2 ( 3 1 . 2 )$ </td><td> $9 3 . 4 \pm 8 2 . 6 ( 7 5 . 8 )$ </td><td> $1 7 . 2 \pm 2 8 . 8 ( 9 . 1 1 )$ </td></tr></table>

## 6 Conclusions

This paper introduced LEAP, a time-series anomaly detector that replaces reconstruction with masked patch prediction, using prediction error to assess how well normal dynamics explain each observation, and augments it with three label-free evidence sources: density-calibrated latent novelty, local contrast, and per-channel novelty.

Under the TSB-AD protocol with VUS-PR as the primary metric, performance improves monotonically across the five stages, with every increment significant under a paired Wilcoxon signed-rank test. Since the evidence layers address largely disjoint subsets of series, their additive fusion preserves these gains; moreover, setting any fusion coefficient to zero reduces the model to its predecessor, providing a built-in controlled ablation. On the 180 multivariate series, LEAP achieves $0 . 5 1 7 9 \pm 0 . 0 0 2 2$ , a 24.5% improvement over the strongest locally measured baseline (PaAno), with posterior win probabilities above 99.9% against the five strongest measured baselines. On the 350 univariate series, the multi-scale variant combining masked-prediction scoring with latent novelty achieves $0 . 6 1 9 4 \pm 0 . 0 0 1 6$ , outperforming all 20 locally measured baselines, including PaAno (+6.1%).

Two findings extend beyond the proposed architecture. First, the largest improvement came from evaluating prediction errors at the event rather than sample level. Second, the univariate advantage over $\mathrm { P a A n o _ { P A I } }$ is narrower, and $\mathrm { P a A n o } _ { \mathrm { P A I } }$ retains stronger performance on AUC-PR, PA-F1, Event-based-F1 and Affiliation-F. Its performance benefits from benchmark-specific adaptations and post-hoc amplitude correction, highlighting the need to evaluate architectural complexity against strong baselines under robust metrics.

More broadly, the detectors that displaced reconstruction Transformers, such as PaAno, VACE and PAI-augmented models, converged on learning short subsequence representations and scoring their proximity to a bank of normal windows. LEAP learns its representation through an objective that is itself a detector, so prediction error remains available as a second and independent signal, and the additive fusion exploits this redundancy.

Four directions follow from these results. A learned weighting of channels could replace the six fixed selection criteria of Eq. (32) and score each channel by its estimated reliability. Fusion coefficients that depend on observable properties of each series, such as its length, number of channels or dominant period, would move LEAP toward the per-series adaptivity that benchmark heterogeneity calls for [12]. The uniform bank of 1,200 windows could become a structured representation, such as the k-means coresets of Park[8]. Finally, since the masked objective mirrors the pre-training of time-series foundation models [37], a pre-trained backbone combined with the same scoring stages would decouple representation from evidence aggregation and show whether the gains come from the representation or from the evidence fusion.

## Acknowledgements

This publication is part of the IAFER TSI-100927-2023-1 project, funded by the Recovery, Transformation, and Resilience Plan of the Next Generation of the EU through the Spanish Ministry of Digital Transformation and the Civil

Service. This work was also supported by the Spanish Ministry of Science and Technology under grants PID2023- 150070NB-I00 and PID2025-170986NA-I00 funded by MICIU/AEI/10.13039/501100011033 and by “ERDF/EU” and the “European Union NextGenerationEU/PRTR”, respectively. This work was also supported by the University of Granada technology-transfer project OTRI-6915 (SmBuilding-X).

## Data and code availability

The datasets used in this study are publicly available as part of the TSB-AD benchmark suite [12]. The implementation of LEAP is available at https://github.com/ari-dasci/S-LEAP.

## References

[1] Ane Blázquez-García, Angel Conde, Usue Mori, and Jose A. Lozano. A review on outlier/anomaly detection in time series data. ACM Computing Surveys, 54(3):56:1–56:33, 2021.

[2] Peng Yan, Ahmed Abdulkadir, Paul-Philipp Luley, Matthias Rosenthal, Gerrit A Schatte, Benjamin F Grewe, and Thilo Stadelmann. A comprehensive survey of deep transfer learning for anomaly detection in industrial time series: Methods, applications, and directions. IEEE Access, 12:3768–3789, 2024.

[3] Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, Charu C. Aggarwal, and Mahsa Salehi. Deep learning for time series anomaly detection: A survey. ACM Computing Surveys, 57(1):15:1–15:42, 2025.

[4] George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung. Time Series Analysis: Forecasting and Control. Wiley, 5th edition, 2015.

[5] Eamonn Keogh, Jessica Lin, and Ada Fu. HOT SAX: Efficiently finding the most unusual time series subsequence. In Proceedings of the 5th IEEE International Conference on Data Mining (ICDM), pages 226–233, 2005.

[6] Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal. Long short term memory networks for anomaly detection in time series. In Proceedings ofthe 23rd European Symposium on Artificial Neural Networks (ESANN), pages 89–94, 2015.

[7] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural networks. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 2828–2837, 2019.

[8] Jinju Park and Seokho Kang. PaAno: Patch-based representation learning for time-series anomaly detection. In Proceedings ofthe 14th International Conference on Learning Representations (ICLR), 2026.

[9] Renjie Wu and Eamonn J. Keogh. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE Transactions on Knowledge and Data Engineering, 35(3):2421–2429, 2023.

[10] Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon. Towards a rigorous evaluation of time-series anomaly detection. In Proceedings ofthe 36th AAAI Conference on Artificial Intelligence, pages 7194–7201, 2022.

[11] Sebastian Schmidl, Phillip Wenig, and Thorsten Papenbrock. Anomaly detection in time series: A comprehensive evaluation. Proceedings ofthe VLDB Endowment, 15(9):1779–1797, 2022.

[12] Qinghua Liu and John Paparrizos. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In Advances in Neural Information Processing Systems 37 (NeurIPS), Datasets and Benchmarks Track, 2024.

[13] Paul Boniol, John Paparrizos, and Themis Palpanas. VUS: Effective and efficient accuracy measures for time-serie anomaly detection. The VLDB Journal, 34(3):31, 2025.

[14] Alberto D. Cencillo, Leonardo Concepción, Isaac Triguero, and Julián Luengo. VACE: Learning geometrically structured representations for time series anomaly detection. arXiv preprint arXiv:2605.23504, 2026.

[15] Quan Zhou, Changhua Pei, Fei Sun, Jing Han, Zhengwei Gao, Dan Pei, Haiming Zhang, Gaogang Xie, and Jianhui Li. KAN-AD: Time series anomaly detection with Kolmogorov–Arnold networks. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of PMLR, pages 79136–79149, 2025.

[16] Beibu Li, Qichao Shentu, Yang Shu, Hui Zhang, Ming Li, Ning Jin, Bin Yang, and Chenjuan Guo. CrossAD: Time series anomaly detection with cross-scale associations and cross-window modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[17] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM Computing Surveys, 41(3):15:1–15:58, 2009.

[18] Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. LOF: Identifying density-based local outliers. In Proceedings of the ACM SIGMOD International Conference on Management of Data, pages 93–104, 2000.

[19] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation-based anomaly detection. ACM Transactions on Knowledge Discoveryfrom Data, 6(1):3:1–3:39, 2012.

[20] Mei-Ling Shyu, Shu-Ching Chen, Kanoksri Sarinnapakorn, and LiWu Chang. A novel anomaly detection scheme based on principal component classifier. Proceedings ofthe IEEE Foundations and New Directions ofData Mining Workshop, pages 172–179, 2003.

[21] Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh. Matrix profile I: All pairs similarity joins for time series. In Proceedings ofthe 16th IEEE International Conference on Data Mining (ICDM), pages 1317–1322, 2016.

[22] Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed Meftah, and Emmanuel Remy. Unsupervised and scalable subsequence anomaly detection in large data series. The VLDB Journal, 30(6):909–931, 2021.

[23] John Paparrizos and Luis Gravano. k-Shape: Efficient and accurate clustering of time series. ACM SIGMOD Record, 45(1):69–76, 2016.

[24] Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. DeepAnT: A deep learning approach for unsupervised anomaly detection in time series. IEEE Access, 7:1991–2005, 2019.

[25] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Söderström. Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 387–395, 2018.

[26] Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A. Zuluaga. USAD: Unsupervised anomaly detection on multivariate time series. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 3395–3404, 2020.

[27] Ailin Deng and Bryan Hooi. Graph neural network-based anomaly detection in multivariate time series. In Proceedings ofthe 35th AAAI Conference on Artificial Intelligence, pages 4027–4035, 2021.

[28] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In Proceedings ofthe 10th International Conference on Learning Representations (ICLR), 2022.

[29] Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings. TranAD: Deep transformer networks for anomaly detection in multivariate time series data. Proceedings ofthe VLDB Endowment, 15(6):1201–1214, 2022.

[30] Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. DCdetector: Dual attention contrastive representation learning for time series anomaly detection. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3033–3045, 2023.

[31] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. TS2Vec: Towards universal representation of time series. In Proceedings ofthe 36th AAAI Conference on Artificial Intelligence, pages 8980–8987, 2022.

[32] Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, Charu C. Aggarwal, and Mahsa Salehi. CARLA: Self supervised contrastive representation learning for time series anomaly detection. Pattern Recognition, 157:110874, 2025.

[33] Hongzuo Xu, Yijie Wang, Songlei Jian, Qing Liao, Yongjun Wang, and Guansong Pang. Calibrated one-class classification for unsupervised time series anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 36(11):5723–5736, 2024.

[34] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, 2022.

[35] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023.

[36] Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, and Mingsheng Long. SimMTM: A simple pre-training framework for masked time-series modeling. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[37] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. MOMENT: A family of open time-series foundation models. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

[38] Vijay Ekambaram, Subodh Kumar, Arindam Jati, Sumanta Mukherjee, Tomoya Sakai, Pankaj Dayama, Wesley Gifford, and Jayant Kalagnanam. Tspulse: Tiny pre-trained models with disentangled representations for rapid time-series analysis. In International Conference on Learning Representations, volume 2026, pages 62090–62133, 2026.

[39] Youji Zhu, Hongbing Wang, Wenchao Liu, Xiaodong Liu, and Xiangguang Xiong. PaAno+: Multiscale encoding and cross-variable attention for time series anomaly detection. arXiv preprint arXiv:2606.20055, 2026.

[40] Kang Zhang, Wei Jian Lau, Shoushou Ren, Dong Lin, Joon Son Chung, and Chuanhao Sun. PAI: Preserving amplitude information in representation-based time-series anomaly detection. arXiv preprint arXiv:2606.08935, 2026.

[41] Xingjian Wu, Xiangfei Qiu, Zhengyu Li, Yihang Wang, Jilin Hu, Chenjuan Guo, Hui Xiong, and Bin Yang. CATCH: Channel-aware multivariate time series anomaly detection via frequency patching. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025.

[42] Qichao Shentu, Beibu Li, Kai Zhao, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, and Chenjuan Guo. Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders. In Proceedings of the 13th International Conference on Learning Representations (ICLR), 2025.

[43] Kamil Faber, Marcin Pietron, Dominik Zurek, and Roberto Corizzo. xlstmad: A powerful xlstm-based method for anomaly detection. In 2025 IEEE International Conference on Data Mining (ICDM), pages 247–256, 2025.

[44] Nesime Tatbul, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. Precision and recall for time series. In Advances in Neural Information Processing Systems (NeurIPS), pages 1920–1930, 2018.

[45] Alexis Huet, Jose Manuel Navarro, and Dario Rossi. Local evaluation of time series anomaly detection algorithms. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 635–645, 2022.

[46] Chin-Chia Michael Yeh. Matrix profile for time-series anomaly detection: A reproducible open-source benchmark on TSB-AD. arXiv preprint arXiv:2604.02445, 2026.

[47] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance nor malization for accurate time-series forecasting against distribution shift. In Proceedings ofthe 10th International Conference on Learning Representations (ICLR), 2022.

[48] Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. Transformers in time series: A survey. In Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI), pages 6778–6786, 2023.

[49] Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023.

[50] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings of the 37th AAAI Conference on Artificial Intelligence, pages 11121–11128, 2023.

[51] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2D-variation modeling for general time series analysis. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR), 2023.

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), pages 5998–6008, 2017.

[53] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings ofthe 36th International Conference on Machine Learning (ICML), pages 3744–3753, 2019.

[54] Frank J. Massey. The Kolmogorov-Smirnov test for goodness of fit. Journal ofthe American Statistical Association, 46(253):68–78, 1951.

[55] Alessio Benavoli, Giorgio Corani, Francesca Mangili, Marco Zaffalon, and Fabrizio Ruggeri. A bayesian wilcoxon signed-rank test based on the dirichlet process. In Eric P. Xing and Tony Jebara, editors, Proceedings ofthe 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1026–1034, Beijing, China, 22–24 Jun 2014. PMLR.

## A Families of Time-Series Anomaly Detectors

Table A.1 summarizes the families of time-series anomaly detectors reviewed in Section 2.2, together with their scoring mechanism and the anomaly types to which they are suited.

Table A.1: Families of time-series anomaly detectors, their scoring mechanism and the anomaly types they are structurally suited to. Semi denotes the semi-supervised (clean training prefix) regime, Un the unsupervised one.
<table><tr><td>Family</td><td>Score mechanism</td><td>Suited to</td><td>Regime</td></tr><tr><td>Statistical residual</td><td>residual of a fitted model (ARIMA, ETS, STL)</td><td>point, contextual</td><td>Semi/Un</td></tr><tr><td>Control charts, CPD</td><td>accumulated deviation of the mean</td><td>level shift, regime change</td><td>Semi/Un</td></tr><tr><td>Spectral</td><td>saliency of the amplitude spec- point, seasonal trum</td><td></td><td>Un</td></tr><tr><td>Discord/matrix profile</td><td>distance to nearest non-trivial match</td><td>subsequence, shape</td><td>Un</td></tr><tr><td>Normal-pattern banks</td><td>distance to a set of normal pat- subsequence, shape terns</td><td></td><td>Semi</td></tr><tr><td>Forecasting (deep)</td><td>prediction residual</td><td>point, contextual</td><td>Semi</td></tr><tr><td>Reconstruction (deep) Relational (graph)</td><td>reconstruction residual</td><td>point, subsequence</td><td>Semi/Un</td></tr><tr><td></td><td>deviation from predicted neigh- bor relations</td><td>relational</td><td>Semi</td></tr><tr><td>Contrastive SSL</td><td>discrepancy between views or neighborhoods</td><td>subsequence, contextual</td><td>Semi/Un</td></tr><tr><td>Representation + bank</td><td>distance in a learned space to nor- subsequence, shape mal windows</td><td></td><td>Semi</td></tr><tr><td>Masked modeling</td><td>error of predicting removed con- shape, level, relational tent</td><td></td><td>Semi</td></tr></table>

## B Mathematical Notation Summary

Table B.1 summarizes the notation used to define LEAP in Section 3.

## C Algorithm, Complexity and Label-Freeness

Algorithm 1 summarizes the procedure.

Complexity Let W be the number of windows. A forward pass costs $\mathcal { O } \big ( B \big / ( C N ^ { 2 } + N C ^ { 2 } ) D + C N D ^ { 2 } \big | \big )$ per window. With the latent bottleneck, this becomes $\mathcal { O } \big ( B \big / ( C N ^ { 2 } + N C K ) D + \dot { C } \dot { N } D ^ { 2 } \big | \big )$ . The $D ^ { 2 }$ term covers token projections and feed-forward layers. Scoring uses $G \stackrel { \cdot } { = } 4$ passes per window. Nearest-neighbor queries use a k-d tree over at most 1200 bank vectors per channel considered. Their cost depends on the embedding dimension and bank geometry and is at most $\mathcal { O } ( W ( \bar { 1 } + | \mathcal { P } | ) | B | )$ ) for brute-force search. The other evidence layers add O(T) for local moments and $\mathcal { O } ( T \log W _ { \gamma } )$ for the running median. In practice, the Transformer passes dominate.

Label-Freeness Every quantity used by LEAP depends only on X and the known split $T _ { \mathrm { t r } }$ . Standardization, window length, network weights, error moments and bank embeddings use the training prefix. The channel criteria of Eq. (32) compare it with the unlabeled remainder. Unit rescaling uses the 99th percentile of each score’s own active support. Ground-truth labels inform neither fitting nor per-series scoring. Global architecture and hyperparameters were selected on the labeled tuning subset under the protocol of Section 4.4.

## D TSB-AD Composition per Dataset

Table D.1 reports the number of held-out series and the application domains contributed by each source dataset of TSB-AD.

Table B.1: Notation used in Section 3. Deployed values of all coefficients and widths are listed in Table E.1.
<table><tr><td>Symbol Data, windowing and encoder</td><td>Meaning</td></tr><tr><td> $T , C$   $T _ { \mathrm { t r } }$   $\boldsymbol { x } _ { t } \in R ^ { C }$   $\tilde { x } _ { t }$   $\hat { P } , \ \pi _ { \mathrm { p e r } }$   $L , p , N = L / p$   $\dot { D , ~ H , ~ B }$   $K$   $h _ { c , n } \in R ^ { D }$   $G , b$  Base scoring (Stage 1)</td><td>series length, number of channels length of the anomaly-free training prefix observation at time t globally standardized observation estimated dominant period, periods spanned by a window window length, patch length, patches per window model width, attention heads, encoder depth number of channel latents (bottleneck) token of channel c, patch n number of masking groups, mask block length (in patches)</td></tr><tr><td> $e _ { t , c }$   $z _ { t , c }$  calibrated (standardized) error  $\lambda$   $\tau$  temperature of the soft channel aggregation  $W _ { s } = \eta p$  event-scale smoothing width  $\kappa$  tolerance-dilation width  $W _ { m } , ~ w _ { \ell }$  local-moment width, local-moment weight Evidence layers (Stages 2–5)</td><td>masked prediction error at  $( t , c )$  shrinkage of the per-channel error spread toward the median of active channels</td></tr><tr><td> $B , \ N _ { \mathrm { b a n k } }$  X  $d ( x )$  latent novelty: distance of window x to the bank  $\beta$  latent-novelty weight (Stage 2)  $k , \ \alpha$   $\gamma , W _ { \gamma } = \nu L$   $\chi ^ { \prime } , \delta$ </td><td>bank of training-window embeddings, bank size highest-variance channels in the concatenated embedding density-calibration neighbors, local-density weight (Stage 3) local-contrast weight, running-median width (Stage 4) channels per selection criterion, per-channel novelty weight (Stage 5)</td></tr><tr><td> $u n i t ( \pmb { v } )$   $s ^ { ( \cdot ) }$  score vector of a given stage p always denotes the patch length. Statistical significance is</td><td> ${ \pmb v } / q _ { 0 . 9 9 } ( { \pmb v } \mid { \pmb v } > 0 )$  , per-series rescaling</td></tr></table>

p always denotes the patch length. Statistical significance is  
reported throughout as a p-value, never as a bare $p .$

Table D.1: Composition of the two TSB-AD held-out evaluation sets used in this work (number of series per source dataset).
<table><tr><td colspan="2">TSB-AD-M (180 series)</td><td colspan="2">TSB-AD-U (350 series)</td></tr><tr><td>Source</td><td>#</td><td>Source</td><td>#</td></tr><tr><td>SVDB</td><td>28</td><td>UCR</td><td>70</td></tr><tr><td>Exathlon</td><td>25</td><td>SMD</td><td>33</td></tr><tr><td>SMAP</td><td>25</td><td>Exathlon</td><td>30</td></tr><tr><td>GHL</td><td>23</td><td>YAHOO</td><td>30</td></tr><tr><td>SMD</td><td>20</td><td>OPPORTUNITY</td><td>27</td></tr><tr><td>MSL</td><td>14</td><td>NAB</td><td>23</td></tr><tr><td>MITDB</td><td>11</td><td>WSD</td><td>20</td></tr><tr><td>TAO</td><td>11</td><td>SVDB</td><td>18</td></tr><tr><td>OPPORTUNITY</td><td>7</td><td>SMAP</td><td>17</td></tr><tr><td>CATSv2</td><td>5</td><td>IOPS</td><td>15</td></tr><tr><td>LTDB</td><td>4</td><td>TODS</td><td>13</td></tr><tr><td>SWaT</td><td>2</td><td>LTDB, MGAB, NEK, Stock</td><td>8 each</td></tr><tr><td rowspan="3">CreditCard, Daphnet, GECCO, Genesis, PSM</td><td rowspan="3">1 each</td><td>MITDB, MSL</td><td>7 each</td></tr><tr><td>SED, TAO</td><td></td></tr><tr><td>CATSv2, Daphnet, Power, SWaT</td><td>2 each 1 each</td></tr><tr><td colspan="2">Domains: Sensor 71, Facility 46,</td><td colspan="2">Domains: Facility 87, WebService 64,</td></tr><tr><td colspan="2">Medical 43, Environment 11,</td><td colspan="2">Medical 47, HumanActivity 43, Sensor 39,</td></tr></table>

Algorithm 1 LEAP fit and score   
Input series $X \in R ^ { T \times C }$ , split $T _ { \mathrm { t r } }$   
Output score $s \in R ^ { T }$   
$F i t$   
1: $( \mu , \tilde { \sigma } , \mathcal { Q } )$ ← standardizer on $X _ { 1 : T _ { \mathrm { t 1 } } }$ ▷ Eqs. (1)–(3)   
2: $L \gets$ window length from the dominant period ▷ Eqs. (4)–(6)   
3: train $f _ { \theta }$ by masked patch prediction with early stopping ▷ Eq. (15)   
4: $( m _ { c } , s _ { c } ^ { \star } ) \gets$ error calibration on training windows ▷ Eqs. (16)–(18)   
5: $( \mu ^ { \tilde { M } } , \sigma ^ { \tilde { M } } , \mu ^ { S } , \sigma ^ { S } )$ ← local-moment calibration ▷ Eq. (23)   
6: $B \gets$ concatenated training bank with LOO distances ▷ Eqs. (26), (29)   
Score   
7: e ← masked error over the $G$ complementary passes ▷ Eq. (14)   
8: z ← per-channel calibration of $e$ ▷ Eq. (18)   
9: a ← soft-max channel aggregation of $_ z$ ▷ Eq. (19)   
10: s ← smooth(overlap-add(a)) ▷ Eqs. (20)–(21)   
11: ℓ ← local-moment evidence ▷ Eq. (23)   
12: $s ^ { ( 1 ) } \gets d i l _ { \kappa } \bigl ( u n i t ( s ) + w _ { \ell } u n i t ( \ell ) \bigr )$ ▷ Eq. (24)   
13: $\hat { \tilde { d } } \gets$ calibrated bank novelty with overlap-add ▷ Eqs. (29), (20)   
14: $s ^ { ( 3 ) } \gets u n i t ( s ^ { ( 1 ) } ) + \beta u n i t ( \hat { \tilde { d } } )$ ▷ Eqs. (29), (30)   
15: $\mathcal { P } $ union of the top-min $( \dot { C } , \dot { \chi } ^ { \prime } )$ channels of six criteria ▷ Eq. (32)   
16: ${ { B } _ { c } } \gets$ training banks with $\dot { \mathrm { L O O } }$ distances for $c \in \mathcal { P }$ ▷ Eq. (29)   
17: A ← pooled per-channel novelty from $B _ { c }$ ▷ Eq. (33)   
18: $s ^ { ( 5 a ) } \stackrel { - } {  } s ^ { ( 3 ) } \stackrel { - } { + } \delta u n i t ( A )$ ▷ Eq. (34)   
19: return $s ^ { ( 5 a ) } + \gamma u n i t _ { q } ( \mathrm { c o n t r a s t } ( s ^ { ( 5 a ) } ) )$ ▷ Eq. (35)

## E Hyperparameters

Table E.1 summarizes the hyperparameters used for LEAP.

## F Hardware and Software Environment

All computational workloads were executed on an NVIDIA DGX-1 system equipped with dual Intel Xeon E5-2698 v4 processors, 512 GB of DDR4 RAM, and an NVIDIA Tesla V100 Tensor Core GPU (32 GB VRAM). The software environment was configured on Ubuntu 22.04 LTS, Python 3.11, PyTorch v2.10.0, CUDA 12.8, and cuDNN v9.10.2.

All reported evaluation runs use standard FP32 precision. Under this configuration, a complete evaluation pass requires on average 545 minutes for TSB-AD-U (350 series) and 2,839 minutes for TSB-AD-M (180 series), consistent with Table 8, with peak GPU memory usage reaching 15 GB.

## G Extended Empirical Ablation Analysis

This appendix presents an extended evaluation of the proposed architecture. We first report the stage-by-stage cumulative ablation across the full pipeline, followed by a micro-ablation isolating specific design decisions within the base detector.

## G.1 Stage-Wise Pipeline Ablation

Table G.1 details performance progression across the five architectural stages on the 180 held-out multivariate series of TSB-AD-M. Because each stage strictly extends its predecessor, consecutive rows evaluate the isolated impact of a single mechanism under unified weighting, windowing, embedding, and calibration settings.

Four core insights emerge from this stage-wise breakdown:

Statistical Significance and Complementarity The overall gain from Stage 1 to Stage 5 reaches +0.056 VUS-PR (a 12% relative improvement), composed of four distinct increments ranging between +0.004 and +0.034. This performance distribution underscores a modular design targeting largely disjoint failure modes: no single term universally dominates across all series, yet each addresses specific anomaly types and wins on more series than it loses. The high tie proportions in Stages 3 and 4 further reflect this behavior, as these mechanisms remain passive when local density or temporal context exhibits no anomaly signal. Importantly, gains overlap little at the individual series level: at a threshold of $\Delta \mathrm { V U S - P R } \geq 0 . 0 1$ , Stage 3 improves 35 series and Stage 4 improves 43 series, yet only 7 series benefit from both by that margin, and only 3 of the 180 series improve across all four incremental stages. All four increments remain significant after Bonferroni correction for the four stage-wise comparisons (largest adjusted p-value $= 3 . 8 \times 1 0 ^ { - 4 } )$

Table E.1: Hyperparameters of LEAP. Adaptive entries are rules evaluated per series. The track-specific rows are the only values that differ between the multivariate and the univariate configuration; on the univariate track each score is read out at every listed width and the maximum unit-rescaled response is kept.
<table><tr><td>Group</td><td>Symbol</td><td>Meaning</td><td>Value</td></tr><tr><td rowspan="6">Architecture</td><td> $p$ </td><td>patch length</td><td>16</td></tr><tr><td> $D$ </td><td>model width</td><td>128</td></tr><tr><td> $H$ </td><td>attention heads</td><td>8</td></tr><tr><td> $B$ </td><td>encoder blocks</td><td>4</td></tr><tr><td> $d _ { \mathrm { f f } }$ </td><td>feed-forward width</td><td> $4 D = 5 1 2$ </td></tr><tr><td>dropout</td><td></td><td>0.0</td></tr><tr><td rowspan="3">Channel axis</td><td> $C _ { \mathrm { l a t } }$ </td><td>latent-bottleneck threshold</td><td>128</td></tr><tr><td> $K$ </td><td>number of channel latents</td><td>16</td></tr><tr><td> $\tau$ </td><td>channel-aggregation temperature</td><td>0.5</td></tr><tr><td rowspan="2">Masking</td><td> $G$ </td><td>masking groups (masked fraction  $1 / G )$ </td><td>4</td></tr><tr><td> $^ { b }$ </td><td>mask block, in patches</td><td>1</td></tr><tr><td rowspan="4">Windowing (adaptive)</td><td> $L$ </td><td>window length</td><td>min(2 , 384), mult. of p</td></tr><tr><td> $\Delta$ </td><td>window stride</td><td> $L / 8$ </td></tr><tr><td></td><td>min./max. training windows</td><td> $5 1 2 / 4 0 0 0$ </td></tr><tr><td> $\kappa$ </td><td>dilation width</td><td> $L / 1 0$ </td></tr><tr><td rowspan="6">Optimization</td><td> $\eta _ { 0 }$ </td><td>learning rate (AdamW)</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>weight decay</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>batch size</td><td>32</td></tr><tr><td></td><td>max. epochs / patience</td><td>400 / 10</td></tr><tr><td></td><td>LR decay factor / patience</td><td>0.5/3</td></tr><tr><td></td><td>gradient clipping / validation split</td><td>1.0 /0.2</td></tr><tr><td rowspan="5">Calibration</td><td> $\rho _ { \sigma }$ </td><td>degenerate-channel floor fraction</td><td>0.05</td></tr><tr><td> $\lambda$ </td><td>error-spread shrinkage</td><td>0.0</td></tr><tr><td> $\rho _ { f }$ </td><td>error-spread floor fraction</td><td>0.1</td></tr><tr><td> $\rho _ { \omega }$ </td><td>edge-weight Gaussian fraction</td><td>0.25</td></tr><tr><td> $\rho _ { \gamma }$ </td><td>contrast denominator floor</td><td>10⁻3</td></tr><tr><td rowspan="7">Evidence fusion</td><td> $w _ { \ell }$ </td><td>local-moment weight (add)</td><td>1.0</td></tr><tr><td> $\beta$ </td><td>latent-novelty weight</td><td>1.0</td></tr><tr><td> $\chi$ </td><td>channels in the concat embedding</td><td>4</td></tr><tr><td> $N _ { \mathrm { b a n k } }$ </td><td>latent bank size</td><td>1200</td></tr><tr><td> $k , \alpha$ </td><td>LOF neighbors, density weight</td><td>4,0.25</td></tr><tr><td> $\nu$ </td><td>contrast width,  $W _ { \gamma } = \nu L$ </td><td>64</td></tr><tr><td> $\chi ^ { \prime }$ </td><td>channels per criterion in the pool</td><td>4</td></tr><tr><td rowspan="4">Track-specific</td><td> $W _ { s }$ </td><td>score smoothing width(s)</td><td> $ 8 p = 1 2 8 ( \mathbf { M } ) / \{ 1 , 8 , 3 2 , 1 2 8 \} ( \mathbf { U } )$ </td></tr><tr><td> $W _ { m }$ </td><td>local-moment width(s)</td><td> $\operatorname* { m a x } ( 8 , 4 p ) = 6 4 \left( \mathrm { M } \right) / \left\{ 1 6 , 6 4 , 2 5 6 \right\} \left( \mathrm { U } \right)$ </td></tr><tr><td>γ</td><td>contrast weight</td><td>1.5 (M) / 0 (U)</td></tr><tr><td> $\delta$ </td><td>per-channel novelty weight</td><td>1.5 (M) / 0 (U)</td></tr></table>

Table G.1: Stage-wise ablation on TSB-AD-M (180 held-out series) single seed (2024). ∆ denotes the mean per-series difference relative to the preceding stage; W/L/T indicates wins, losses, and ties; p-values correspond to two-sided Wilcoxon signed-rank tests on paired per-series VUS-PR scores.
<table><tr><td>Stage</td><td>Model</td><td>VUS-PR</td><td> $\Delta$ </td><td>W/L/T</td><td>p-value</td></tr><tr><td>1</td><td>Masked-patch base</td><td>0.4611</td><td></td><td></td><td></td></tr><tr><td>2</td><td>+ Latent novelty (β=1.0)</td><td>0.4954</td><td>+0.0343</td><td>113/55/12</td><td> $1 . 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>3</td><td>+ Density calibration</td><td>0.5034</td><td>+0.0080</td><td>84/42/54</td><td> $9 . 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>4</td><td>+ Local contrast</td><td>0.5074</td><td>+0.0040</td><td>87/35/58</td><td> $2 . 8 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>5</td><td>+ Per-channel pool (LEAP)</td><td>0.5171</td><td>+0.0097</td><td>90/51/39</td><td> $3 . 4 \times 1 0 ^ { - 5 }$ </td></tr></table>

Impact of Latent Novelty Stage 2 accounts for approximately 60% of the total cumulative improvement. This outcome supports the theoretical motivation in Section 3.7.1: local masked prediction can miss globally novel sequences that remain locally predictable. The deployed $\beta = 1 . 0$ was fixed a priori, assigning equal nominal weight to latent novelty and masked-prediction evidence, independently of held-out results. On the held-out set, $\beta = 0 . 6$ improved upon Stage 1 (+0.0242 VUS-PR, 111/54/15 W/L/T, p-value $= 5 . 7 \times 1 0 ^ { - 6 } )$ , while increasing β to 1.0 yielded a further significant gain (+0.0101 VUS-PR, 104/56/20 W/L/T, $p \mathrm { - v a l u e = 1 . 7 \times 1 0 ^ { - 4 } ) }$ . On the 20-series tuning set, $\beta = 0$ slightly outperformed the deployed $\beta = 1 . 0$ in mean VUS-PR $( 0 . 6 7 6 1 \mathrm { v s . } 0 . 6 7 1 7 )$ . However, LOSO analysis reversed this preference, favoring $\beta = 1 . \bar { 0 }$ and revealing the sensitivity of the apparent $\beta = 0$ advantage to individual series. The deployed value was therefore consistent with both its a priori equal-weighting rationale and the tuning-set robustness analysis. Its substantial benefit was subsequently observed on the held-out set, which was not used for parameter selection.

Manifold Standardization via Density Calibration Stage 3 makes nearest-neighbor distances comparable across regions of varying density. Although α = 1.0 achieves the highest aggregate VUS-PR on the tuning set (0.7101, $\Delta = + 0 . 0 3 8 4 ;$ Table 5), its gains are concentrated in two series, with zero median improvement and degradation in 9 of 20 series. This instability near $\bar { \delta } _ { k } \to 0$ motivates the conservative $\alpha = 0 . 2 5$ . Performance is also robust to neighborhood size: in a preliminary sweep on the 20 tuning series, $k \in \{ 1 , 4 , 8 \}$ improved over uncalibrated novelty by +0.0144, +0.0123, and +0.0125 VUS-PR, respectively, while on the held-out set the deployed $k = 4$ contributes +0.0080 (Table G.1).

Robustness Across Benchmark Series The performance improvement in Stage $5 \left( + 0 . 0 0 9 7 \mathrm { V U S - P R } \right)$ is distributed broadly across the evaluation suite rather than driven by extreme outliers. A leave-one-series-out sensitivity analysis demonstrates that removing the most influential sequence (166\_SMAP) slightly increases the mean gain to +0.0111 VUS-PR, confirming the generalizability of per-channel pooling across the benchmark.

## G.2 Micro-Ablation of the Base Detector

The Stage 1 baseline score $( s ^ { ( 1 ) }$ in Eq. (24); VUS-PR 0.4611) incorporates the score calibration, channel aggregation, event-scale smoothing, and local-moment formulations detailed in Section 3.6, rather than raw reconstruction errors. Table G.2 isolates the specific contributions of event-scale smoothing and local-moment evidence.

To disentangle individual design decisions, single-variable ablation experiments were conducted on the 20-series tuning set. Because each configuration is tested independently against its baseline, initial scores for smoothing (0.4109) and local moments (0.4439) differ. These development baselines also differ from the Stage 1 score (0.4611), which reflects final execution on the 180-series held-out evaluation set after locking hyperparameters during the tuning phase. The aggregation-rule comparison is additionally reported using Average Precision (AP), whereas the other micro-ablations use the primary VUS-PR metric. This component-level evaluation reveals two key findings:

Table G.2: Single-variable micro-ablation of base detector design decisions on the tuning set.
<table><tr><td>Design Decision</td><td>Configuration Comparison</td><td>Metric</td><td>Performance</td></tr><tr><td>Event-scale smoothing (Eq. (21))</td><td> $W _ { s } = 1  W _ { s } = 8 p$ </td><td>VUS-PR</td><td> $0 . 4 1 0 9  0 . 5 1 3 5$ </td></tr><tr><td>Local-moment evidence (Eq. (23))</td><td> $w _ { \ell } = 0 \to w _ { \ell } > 0$ </td><td>VUS-PR</td><td> $0 . 4 4 3 9  0 . 4 5 3 9$ </td></tr><tr><td>Moment combination scheme</td><td>max vs. rescaled sum</td><td>AP</td><td> $0 . 3 5 2 8 \to 0 . 3 9 7 6$ </td></tr></table>

1. Impact of Event-Scale Smoothing: In alignment with Eq. (21), event-scale smoothing serves as a primary performance driver in the base detector. Increasing the smoothing window from $\breve { W _ { s } } ~ = ~ 1$ to $\bar { W _ { s } } ~ = ~ 8 \bar { p }$ improves tuning VUS-PR from 0.4109 to 0.5135. This step establishes a solid scoring foundation upon which subsequent pipeline stages (Stages 2–5) achieve further cumulative gains across the held-out evaluation set (+0.056 VUS-PR total; see Table G.1).

2. Contribution of Local-Moment Evidence: Incorporating local-moment evidence (Section 3.6.4) yields a modest but statistically significant gain across the tuning set (0.4439 → 0.4539 VUS-PR, p-value = 0.021). This confirms its theoretical function as a targeted mechanism for detecting sustained level or dispersion shifts while remaining passive elsewhere. Additionally, the rescaled-sum aggregation in Eq. (23) improves Average Precision (AP) over the standard maximum operator (0.3528 → 0.3976), providing complementary evidence for the selected combination rule.

## H Impact of Data and Domain Characteristics

This section provides a granular breakdown of the per-series VUS-PR performance aggregated in Tables 3–7. Specifically, we compare the three-seed measured scores of LEAP against the five top-performing baselines identified for each track:

• Multivariate (M): PaAno, PaAno<sub>PAI</sub>, USAD, xLSTMAD-R, and TimesNet.

• Univariate (U): PaAno, PaAno , PatchTST, Sub-PCA, and USAD.

To ensure strict empirical consistency, this disaggregated evaluation is derived directly from the exact same experimental runs utilized for the main aggregate metrics.

Performance across Datasets and Application Domains Tables H.1 and H.2 stratify the per-series scores of both tracks according to the 17 and 23 source datasets detailed in Table D.1, as well as by their respective application domains. By examining this breakdown, we aim to investigate whether the central finding of Liu [12]—that no single anomaly detection method consistently dominates across diverse data sources—holds true when LEAP is introduced to the benchmark.

Table H.1: Mean VUS-PR by source dataset and by application domain on the TSB-AD-M held-out set (180 series), for LEAP and the five measured baselines of Table 4. Values are the three-seed per-series scores of Table 3, pooled within each group; boldface marks the best method per row.
<table><tr><td>Source</td><td>n</td><td>LEAP</td><td>PaAno</td><td>PaAnOPAI</td><td>USAD</td><td>xLSTMAD-R</td><td>TimesNet</td></tr><tr><td>SVDB</td><td>28</td><td>0.6411</td><td>0.5544</td><td>0.4583</td><td>0.4489</td><td>0.3462</td><td>0.4669</td></tr><tr><td>Exathlon</td><td>25</td><td>0.9907</td><td>0.7773</td><td>0.8705</td><td>0.9659</td><td>0.9454</td><td>0.8449</td></tr><tr><td>SMAP</td><td>25</td><td>0.5980</td><td>0.5316</td><td>0.3713</td><td>0.2657</td><td>0.2377</td><td>0.2922</td></tr><tr><td>GHL</td><td>23</td><td>0.1830</td><td>0.0085</td><td>0.0114</td><td>0.0130</td><td>0.0336</td><td>0.0078</td></tr><tr><td>SMD</td><td>20</td><td>0.3295</td><td>0.3319</td><td>0.3218</td><td>0.3959</td><td>0.3877</td><td>0.4714</td></tr><tr><td>MSL</td><td>14</td><td>0.3715</td><td>0.2108</td><td>0.3103</td><td>0.3058</td><td>0.3427</td><td>0.2556</td></tr><tr><td>MITDB</td><td>11</td><td>0.4497</td><td>0.3823</td><td>0.2539</td><td>0.1327</td><td>0.1440</td><td>0.1803</td></tr><tr><td>TAO</td><td>11</td><td>0.7161</td><td>0.6654</td><td>0.8498</td><td>0.7369</td><td>0.7393</td><td>0.7662</td></tr><tr><td>OPPORTUNITY</td><td>7</td><td>0.1133</td><td>0.1457</td><td>0.1491</td><td>0.3941</td><td>0.3229</td><td>0.0679</td></tr><tr><td>CATSv2</td><td>5</td><td>0.2498</td><td>0.0781</td><td>0.0816</td><td>0.1865</td><td>0.2211</td><td>0.1807</td></tr><tr><td>LTDB</td><td>4</td><td>0.5332</td><td>0.6161</td><td>0.5288</td><td>0.4903</td><td>0.4079</td><td>0.5121</td></tr><tr><td>SWaT</td><td>2</td><td>0.5219</td><td>0.2395</td><td>0.3162</td><td>0.4521</td><td>0.5072</td><td>0.1470</td></tr><tr><td>CreditCard</td><td>1</td><td>0.0663</td><td>0.0256</td><td>0.0496</td><td>0.0636</td><td>0.0663</td><td>0.0585</td></tr><tr><td>Daphnet</td><td>1</td><td>0.3842</td><td>0.2395</td><td>0.3351</td><td>0.1260</td><td>0.2361</td><td>0.3393</td></tr><tr><td>GECCO</td><td>1</td><td>0.1030</td><td>0.1934</td><td>0.1007</td><td>0.1233</td><td>0.1923</td><td>0.3129</td></tr><tr><td>Genesis</td><td>1</td><td>0.7286</td><td>0.3141</td><td>0.8998</td><td>0.0488</td><td>0.6535</td><td>0.0170</td></tr><tr><td>PSM</td><td>1</td><td>0.2344</td><td>0.2070</td><td>0.1662</td><td>0.1685</td><td>0.3868</td><td>0.1798</td></tr><tr><td colspan="2">By application domain</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sensor</td><td>71</td><td>0.3871</td><td>0.2509</td><td>0.2244</td><td>0.1864</td><td>0.2039</td><td>0.1773</td></tr><tr><td>Facility</td><td>46</td><td>0.6868</td><td>0.5713</td><td>0.6166</td><td>0.7007</td><td>0.6908</td><td>0.6680</td></tr><tr><td>Medical</td><td>43</td><td>0.5821</td><td>0.5161</td><td>0.4126</td><td>0.3718</td><td>0.3002</td><td>0.3978</td></tr><tr><td>Environment</td><td>11</td><td>0.7161</td><td>0.6654</td><td>0.8498</td><td>0.7369</td><td>0.7393</td><td>0.7662</td></tr><tr><td>HumanActivity</td><td>8</td><td>0.1471</td><td>0.1574</td><td>0.1724</td><td>0.3606</td><td>0.3120</td><td>0.1018</td></tr><tr><td>Finance</td><td>1</td><td>0.0663</td><td>0.0256</td><td>0.0496</td><td>0.0636</td><td>0.0663</td><td>0.0585</td></tr></table>

On the TSB-AD-M track, LEAP achieves the best performance on nine of the 17 source datasets and ties with xLSTMAD-R on CreditCard. The remaining seven datasets are led by five competitors: TimesNet on GECCO and SMD, PaAno on Genesis and TAO, PaAno on LTDB, USAD on OPPORTUNITY, and xLSTMAD-R on PSM. This

distribution mirrors the dataset fragmentation documented by Liu [12], confirming that performance heterogeneity persists even against a top-performing detector.

Two datasets exhibit notable performance shifts. On GHL, LEAP achieves its largest margin of superiority, outperforming the second-best model, xLSTMAD-R, by +0.1494. Where all competing methods fail to exceed a score of 0.034, LEAP reaches 0.183. This suggests that integrating local-moment and latent-novelty components (Section 3.6.4) extracts actionable anomaly signals from regimes where standard detectors fail completely.

Conversely, the OPPORTUNITY dataset accounts for LEAP’s largest relative deficit, falling −0.2808 behind USAD. Because OPPORTUNITY comprises 7 of the 8 series in the HumanActivity domain within the held-out set, this single dataset dictates the domain aggregate. As a result, LEAP ranks fifth out of six methods in HumanActivity, despite leading two of the remaining five domains (Medical and Sensor, with Sensor being the largest domain at 71 series).

Domain-level aggregation does not smooth out this underlying per-source fragmentation. LEAP leads outright in two of the six domains and ties for first in a third (Finance, comprising a single series), whereas USAD dominates Facility and HumanActivity, and PaAno leads Environment. This breakdown confirms that the “no free lunch” trade-off observed at the dataset level persists after domain grouping.

Table H.2: Mean VUS-PR by source dataset and by application domain on the TSB-AD-U held-out set (350 series), for LEAP and the five measured baselines of Table 7. Values are the three-seed per-series scores of Table 6, pooled within each group; boldface marks the best method per row.
<table><tr><td>Source</td><td>n</td><td>LEAP</td><td>PaAno</td><td>PaAnoPAI</td><td>PatchTST</td><td>Sub-PCA</td><td>USAD</td></tr><tr><td>UCR</td><td>70</td><td>0.4593</td><td>0.3739</td><td>0.4017</td><td>0.2520</td><td>0.1237</td><td>0.1045</td></tr><tr><td>SMD</td><td>33</td><td>0.8270</td><td>0.5005</td><td>0.5327</td><td>0.8065</td><td>0.4416</td><td>0.5439</td></tr><tr><td>Exathlon</td><td>30</td><td>0.9626</td><td>0.8186</td><td>0.9285</td><td>0.8559</td><td>0.8920</td><td>0.9364</td></tr><tr><td>YAHOO</td><td>30</td><td>0.4170</td><td>0.6810</td><td>0.6968</td><td>0.2731</td><td>0.1547</td><td>0.1027</td></tr><tr><td>OPPORTUNITY</td><td>27</td><td>0.6304</td><td>0.2646</td><td>0.7673</td><td>0.0726</td><td>0.9139</td><td>0.7977</td></tr><tr><td>NAB</td><td>23</td><td>0.6446</td><td>0.4982</td><td>0.4355</td><td>0.4868</td><td>0.3947</td><td>0.4946</td></tr><tr><td>WSD</td><td>20</td><td>0.6690</td><td>0.4275</td><td>0.5153</td><td>0.3032</td><td>0.1229</td><td>0.0323</td></tr><tr><td>SVDB</td><td>18</td><td>0.6386</td><td>0.7045</td><td>0.5074</td><td>0.6158</td><td>0.5222</td><td>0.4422</td></tr><tr><td>SMAP</td><td>17</td><td>0.8043</td><td>0.8045</td><td>0.8566</td><td>0.7795</td><td>0.5045</td><td>0.4631</td></tr><tr><td>IOPS</td><td>15</td><td>0.5254</td><td>0.2673</td><td>0.4308</td><td>0.3103</td><td>0.2478</td><td>0.1531</td></tr><tr><td>TODS</td><td>13</td><td>0.6242</td><td>0.7296</td><td>0.7789</td><td>0.7160</td><td>0.5394</td><td>0.5493</td></tr><tr><td>LTDB</td><td>8</td><td>0.6172</td><td>0.6950</td><td>0.6326</td><td>0.5514</td><td>0.5558</td><td>0.5504</td></tr><tr><td>MGAB</td><td>8</td><td>0.3085</td><td>0.2831</td><td>0.1873</td><td>0.0677</td><td>0.0052</td><td>0.0056</td></tr><tr><td>NEK</td><td>8</td><td>0.6929</td><td>0.5662</td><td>0.7603</td><td>0.6322</td><td>0.8856</td><td>0.7237</td></tr><tr><td>Stock</td><td>8</td><td>0.6351</td><td>0.5538</td><td>0.6790</td><td>0.6274</td><td>0.7631</td><td>0.6197</td></tr><tr><td>MITDB</td><td>7</td><td>0.3953</td><td>0.4296</td><td>0.3200</td><td>0.2579</td><td>0.3101</td><td>0.2174</td></tr><tr><td>MSL</td><td>7</td><td>0.5639</td><td>0.2515</td><td>0.4915</td><td>0.6146</td><td>0.5085</td><td>0.4832</td></tr><tr><td>SED</td><td>2</td><td>0.7558</td><td>0.9531</td><td>0.9319</td><td>0.0891</td><td>0.0314</td><td>0.0351</td></tr><tr><td>TAO</td><td>2</td><td>0.9314</td><td>0.8704</td><td>0.9493</td><td>0.9308</td><td>0.9299</td><td>0.9283</td></tr><tr><td>CATSv2</td><td>1</td><td>0.2872</td><td>0.3209</td><td>0.3743</td><td>0.3692</td><td>0.2647</td><td>0.3186</td></tr><tr><td>Daphnet</td><td>1</td><td>0.5049</td><td>0.3034</td><td>0.3543</td><td>0.5573</td><td>0.4218</td><td>0.4457</td></tr><tr><td>Power</td><td>1</td><td>0.1169</td><td>0.1857</td><td>0.1513</td><td>0.0718</td><td>0.0761</td><td>0.1650</td></tr><tr><td>SWaT</td><td>1</td><td>0.5346</td><td>0.0961</td><td>0.1786</td><td>0.0991</td><td>0.3930</td><td>0.3546</td></tr><tr><td colspan="8">By application domain</td></tr><tr><td>Facility</td><td>87</td><td>0.7499</td><td>0.5637</td><td>0.5953</td><td>0.6926</td><td>0.5359</td><td>0.5968</td></tr><tr><td>WebService</td><td>64</td><td>0.5500</td><td>0.4511</td><td>0.5379</td><td>0.3279</td><td>0.2976</td><td>0.2193</td></tr><tr><td>Medical</td><td>47</td><td>0.6345</td><td>0.6483</td><td>0.5704</td><td>0.4430</td><td>0.3956</td><td>0.3177</td></tr><tr><td>HumanActivity</td><td>43</td><td>0.5514</td><td>0.2445</td><td>0.5675</td><td>0.0989</td><td>0.6092</td><td>0.5365</td></tr><tr><td>Sensor</td><td>39</td><td>0.6998</td><td>0.6407</td><td>0.6967</td><td>0.5732</td><td>0.4073</td><td>0.3968</td></tr><tr><td>Synthetic</td><td>39</td><td>0.5391</td><td>0.6341</td><td>0.6324</td><td>0.4367</td><td>0.2423</td><td>0.2414</td></tr><tr><td>Environment</td><td>18</td><td>0.3288</td><td>0.2718</td><td>0.4108</td><td>0.3907</td><td>0.1370</td><td>0.1262</td></tr><tr><td>Finance</td><td>8</td><td>0.6351</td><td>0.5538</td><td>0.6790</td><td>0.6274</td><td>0.7631</td><td>0.6197</td></tr><tr><td>Traffic</td><td>5</td><td>0.6984</td><td>0.5691</td><td>0.4542</td><td>0.4339</td><td>0.3339</td><td>0.4983</td></tr></table>

On the TSB-AD-U track, performance fragmentation is even more pronounced. LEAP leads on 8 of the 23 source datasets (UCR, SMD, Exathlon, NAB, WSD, IOPS, MGAB, and SWaT), followed by PaAno<sub>PAI</sub> and PaAno on 5 each, Sub-PCA on 3, and PatchTST on 2 (with USAD leading none). Consequently, even the top-ranked method secures the best score on barely more than a third of the available sources.

LEAP’s two largest univariate deficits occur on OPPORTUNITY (0.6304 compared to 0.9139 for Sub-PCA, $\Delta =$ −0.2835) and YAHOO (0.4170 compared to 0.6968 for PaAno<sub>PAI</sub>, $\Delta = - 0 . 2 7 9 \bar { 8 } )$ . The YAHOO gap directly reflects the per-series instability previously noted in Figure 6. Specifically, 26 of the 30 YAHOO series are dominated by point-scale anomalies (Appendix H)—the exact anomaly category where LEAP demonstrates the lowest relative efficacy, as detailed in the subsequent type-level breakdown.

The OPPORTUNITY deficit, however, cannot be attributed to the same morphological weakness observed in YAHOO. In OPPORTUNITY, the series are dominated by level shifts (18 of 27) and dispersion changes (7 of 27), two categories on which LEAP leads every baseline on average (Table H.4). This discrepancy highlights a dataset-specific effect that type-level stratification cannot fully explain, serving as a reminder that categorical breakdowns capture average tendencies rather than guarantees for individual data sources.

Domain-level aggregation reflects a similarly mixed performance profile. LEAP leads four of the nine domains, including the two largest, Facility (87 series) and WebService (64 series), together with Sensor and Traffic. In WebService, the YAHOO deficit is outweighed by the leads on WSD and IOPS (+0.0121 over $\mathrm { P a A n o _ { P A I } ) }$ . Conversely, it trails Sub-PCA by −0.1280 in Finance and by −0.0578 in HumanActivity, PaAno by −0.0950 in Synthetic and narrowly by −0.0138 in Medical, and $\mathrm { P a A n o } _ { \mathrm { P A I } }$ by −0.0820 in Environment. From a methodological standpoint, it is important to note that datasets such as NAB, UCR, and YAHOO contribute series to multiple domains under the standard TSB-AD taxonomy (with NAB alone spanning five). Consequently, the source-dataset and domain-level views represent genuinely distinct partitions of the 350 series, rather than simple hierarchical coarsenings of one another.

By Anomaly Type To evaluate model performance across fine-grained anomaly behaviors, time series must be stratified by structural anomaly type. While TSB-AD provides collection-level tags distinguishing sequence-only (Seq) from mixed point and sequence (P&Seq) datasets [12], this metadata is insufficient for granular ablation. Specifically, the official tags are assigned per collection rather than per individual time series, and they collapse distinct structural phenomena (including level shifts, dispersion changes, shape anomalies, and relational disruptions) into a single generic sequence category.

To establish a precise and reproducible classification, we operationalize the six-way taxonomy from Section 2.2 through an automated feature-extraction procedure applied directly to ground-truth anomaly segmentations. For each labeled anomalous segment, the target channel exhibiting the largest standardized mean shift relative to its preceding normal history is identified. Short segments containing five or fewer samples are classified as point-scale anomalies, categorized as global $( z \geq 4$ relative to training marginals) or contextual $( z \geq 3$ relative to local pre-event windows).

Longer anomalous segments are evaluated across three descriptive statistics computed on the primary channel: standardized mean shift (level), log-ratio of standard deviations (dispersion), and z-normalized Euclidean distance to nearest-matching normal reference windows (shape). Segments are assigned to whichever category exceeds its calibrated threshold (3.0, 2 octaves, and 2.5, respectively) by the largest margin. For multivariate series, a segment is classified as relational when every individual channel among the top 20 highest-variance channels remains within normal bounds $( z < 2 . 5 )$ while the joint Mahalanobis distance exceeds 3.0, isolating instances where inter-channel correlations break despite normal univariate marginals.

Finally, a time series is designated with the anomaly type that accounts for at least 60% of its total anomalous duration, or classified as mixed if no single category dominates. This automated classification establishes the per-series taxonomy utilized throughout Tables H.3–H.4.

Table H.3: Mean VUS-PR by automatically-assigned dominant anomaly type (Appendix H) on the TSB-AD-M held-out set, for LEAP and the five measured baselines of Table 4. Boldface marks the best method per row.
<table><tr><td>Type</td><td>n</td><td>LEAP</td><td>PaAno</td><td> $\mathrm { P a A n o _ { P A I } }$ </td><td>USAD</td><td>xLSTMAD-R</td><td>TimesNet</td></tr><tr><td>Level shift</td><td>88</td><td>0.5662</td><td>0.4455</td><td>0.4531</td><td>0.5097</td><td>0.4875</td><td>0.4466</td></tr><tr><td>Subsequence/shape</td><td>34</td><td>0.5265</td><td>0.4890</td><td>0.3730</td><td>0.3197</td><td>0.2574</td><td>0.3563</td></tr><tr><td>Relational</td><td>16</td><td>0.3547</td><td>0.1781</td><td>0.1707</td><td>0.1419</td><td>0.2146</td><td>0.1476</td></tr><tr><td>Dispersion change</td><td>16</td><td>0.3426</td><td>0.2675</td><td>0.2225</td><td>0.1561</td><td>0.1619</td><td>0.1593</td></tr><tr><td>Mixed</td><td>14</td><td>0.4573</td><td>0.3277</td><td>0.3261</td><td>0.2746</td><td>0.2845</td><td>0.3512</td></tr><tr><td>Point (global)</td><td>8</td><td>0.5828</td><td>0.5217</td><td>0.7148</td><td>0.6043</td><td>0.6079</td><td>0.6315</td></tr><tr><td>Point (contextual)</td><td>4</td><td>0.8202</td><td>0.7929</td><td>0.9198</td><td>0.8338</td><td>0.8336</td><td>0.8589</td></tr></table>

Categorizing performance by anomaly type evaluates $\mathrm { L E A P ^ { \prime } s }$ scoring components directly against their targeted failure modes rather than aggregate metrics. Across both benchmark tracks, LEAP leads on level shifts, dispersion changes, shape anomalies, and mixed series, confirming the specific contributions of the local-moment term (Section 3.6.4), local-density calibration (Section 3.7.2), and latent discord scoring (Section 3.7.1). Furthermore, on TSB-AD-M, LEAP achieves its widest margin of advantage on relational anomalies (+0.1401 VUS-PR over xLSTMAD-R), demonstrating that group-masking (Section 3.4) effectively exposes broken cross-channel dependencies.

Table H.4: Mean VUS-PR by automatically-assigned dominant anomaly type (Appendix H) on the TSB-AD-U held-out set, for LEAP and the five measured baselines of Table 7. Relational anomalies do not apply to a univariate series and are absent by construction. Boldface marks the best method per row.
<table><tr><td>Type</td><td>n</td><td>LEAP</td><td>PaAno</td><td>PaAnoPAI</td><td>PatchTST</td><td>Sub-PCA</td><td>USAD</td></tr><tr><td>Level shift</td><td>115</td><td>0.8072</td><td>0.5183</td><td>0.7578</td><td>0.5954</td><td>0.7053</td><td>0.6768</td></tr><tr><td>Subsequence/shape</td><td>101</td><td>0.5255</td><td>0.4934</td><td>0.4102</td><td>0.4250</td><td>0.2243</td><td>0.2311</td></tr><tr><td>Dispersion change</td><td>68</td><td>0.5494</td><td>0.5102</td><td>0.5108</td><td>0.3323</td><td>0.3114</td><td>0.3109</td></tr><tr><td>Mixed</td><td>22</td><td>0.5895</td><td>0.4862</td><td>0.5796</td><td>0.4912</td><td>0.4823</td><td>0.4132</td></tr><tr><td>Point (global)</td><td>29</td><td>0.4455</td><td>0.6773</td><td>0.7188</td><td>0.3593</td><td>0.2403</td><td>0.1619</td></tr><tr><td>Point (contextual)</td><td>15</td><td>0.5083</td><td>0.4213</td><td>0.4965</td><td>0.3140</td><td>0.2347</td><td>0.1652</td></tr></table>

Conversely, LEAP displays a clear sensitivity limitation on point-scale anomalies. On TSB-AD-M, it ranks near the bottom of the field on both global and contextual point events, reaching 0.5828 and 0.8202 compared to PaAno ’s 0.7148 and 0.9198. On TSB-AD-U, LEAP leads narrowly on contextual point anomalies (0.5083 versus 0.4965 for PaAno<sub>PAI</sub>), but global point anomalies remain its weakest category, ranking third of six (0.4455 versus 0.7188 for PaAno and 0.6773 for PaAno). Structurally, masked-patch prediction, moving averages, and k-nearest-neighbor novelty metrics operate over patch-wide windows, rendering them inherently less sensitive to isolated sample spikes than density (or memory-bank) based scorers. Because point-scale anomalies represent a small subset of the evaluation sets (12 of 180 series on M; 44 of 350 on U), this failure mode has a constrained impact on aggregate scores, yet it establishes the primary operational limit of LEAP’s design.

By Series Characteristics To evaluate the empirical necessity and behavior of the adaptive hyper-parameter rules (Table E.1), per-series evaluation scores are stratified across four intrinsic properties: channel count (multivariate track only), sequence length, anomaly ratio, and the presence of a dominant period estimated via Eqs. (4)–(5) on the training prefix. Tables H.5 and H.6 summarize these performance breakdowns on TSB-AD-M and TSB-AD-U.

Table H.5: Mean VUS-PR by series characteristics on the TSB-AD-M held-out set, for LEAP and the five measured baselines of Table 4; boldface marks the best method per row.
<table><tr><td></td><td>n</td><td>LEAP</td><td>PaAno</td><td> $\mathrm { P a A n o _ { P A I } }$ </td><td>USAD</td><td>xLSTMAD-R</td><td>TimesNet</td></tr><tr><td colspan="8">By number of channels</td></tr><tr><td>2-10</td><td>58</td><td>0.6089</td><td>0.5505</td><td>0.5089</td><td>0.4524</td><td>0.4028</td><td>0.4587</td></tr><tr><td>11-30</td><td>72</td><td>0.5182</td><td>0.3660</td><td>0.3356</td><td>0.3291</td><td>0.3333</td><td>0.3185</td></tr><tr><td>31-248</td><td>50</td><td>0.4121</td><td>0.3322</td><td>0.3827</td><td>0.4514</td><td>0.4499</td><td>0.4037</td></tr><tr><td colspan="8">By series length (samples)</td></tr><tr><td>≤ 10,000</td><td>53</td><td>0.5424</td><td>0.4601</td><td>0.4517</td><td>0.3867</td><td>0.3812</td><td>0.3720</td></tr><tr><td>10,001-46,655</td><td>38</td><td>0.5043</td><td>0.4415</td><td>0.4719</td><td>0.5296</td><td>0.5391</td><td>0.5059</td></tr><tr><td>46,656–200,000</td><td>55</td><td>0.4932</td><td>0.3334</td><td>0.3293</td><td>0.3532</td><td>0.3443</td><td>0.3308</td></tr><tr><td>&gt; 200,000</td><td>34</td><td>0.5351</td><td>0.4527</td><td>0.3772</td><td>0.3666</td><td>0.3010</td><td>0.3703</td></tr><tr><td colspan="8">By anomaly ratio</td></tr><tr><td>≤ 0.0114</td><td>45</td><td>0.4094</td><td>0.2795</td><td>0.2427</td><td>0.1857</td><td>0.1993</td><td>0.2091</td></tr><tr><td>0.0115-0.0374</td><td>45</td><td>0.3830</td><td>0.3038</td><td>0.2221</td><td>0.2243</td><td>0.2020</td><td>0.2422</td></tr><tr><td>0.0375-0.0715</td><td>45</td><td>0.5584</td><td>0.4626</td><td>0.5092</td><td>0.5259</td><td>0.4928</td><td>0.4876</td></tr><tr><td>&gt; 0.0715</td><td>45</td><td>0.7210</td><td>0.6184</td><td>0.6440</td><td>0.6754</td><td>0.6583</td><td>0.6105</td></tr><tr><td colspan="8">By dominant period (Eq. (4))</td></tr><tr><td>Period found</td><td>176</td><td>0.5109</td><td>0.4072</td><td>0.3923</td><td>0.3930</td><td>0.3779</td><td>0.3767</td></tr><tr><td>No period found</td><td>4</td><td>0.8293</td><td>0.8062</td><td>0.9408</td><td>0.8352</td><td>0.8392</td><td>0.8556</td></tr></table>

Three primary empirical trends emerge from this stratification. First, LEAP’s advantage on TSB-AD-M does not extend to the widest series. It leads in the 2–10 channel band (+0.0584 VUS-PR over PaAno) and most clearly in the 11–30 band (+0.1522 over PaAno), but falls behind USAD and xLSTMAD-R in the 31–248 band (−0.0393 relative to USAD). Because the per-channel pooling step (Section 3.7.4) selects a representative channel subset rather than modeling all channels jointly, its selectivity degrades as the candidate channel pool expands. This empirical drop directly supports replacing fixed pooling criteria with learned reliability weighting (Section 6).

Table H.6: Mean VUS-PR by series characteristics on the TSB-AD-U held-out set, for LEAP and the five measured baselines of Table 7. Every series is univariate by construction, so the channel stratification of Table H.5 does not apply. Boldface marks the best method per row.
<table><tr><td></td><td>n</td><td>LEAP</td><td>PaAno</td><td>PaAnoPAI</td><td>PatchTST</td><td>Sub-PCA</td><td>USAD</td></tr><tr><td colspan="8">By series length (samples)</td></tr><tr><td> $\leq 7 { , } 5 0 0$ </td><td>88</td><td>0.5785</td><td>0.5866</td><td>0.6334</td><td>0.4839</td><td>0.3915</td><td>0.3733</td></tr><tr><td>7,501–18,228</td><td>87</td><td>0.6215</td><td>0.4511</td><td>0.6430</td><td>0.3175</td><td>0.3946</td><td>0.3296</td></tr><tr><td>18,229–43,066</td><td>96</td><td>0.7434</td><td>0.5540</td><td>0.6254</td><td>0.6177</td><td>0.5124</td><td>0.5459</td></tr><tr><td>&gt; 43,066</td><td>79</td><td>0.5119</td><td>0.4651</td><td>0.4130</td><td>0.3849</td><td>0.3553</td><td>0.3118</td></tr><tr><td colspan="8">By anomaly ratio</td></tr><tr><td>≤ 0.0048</td><td>88</td><td>0.4496</td><td>0.4402</td><td>0.4205</td><td>0.2330</td><td>0.0881</td><td>0.0437</td></tr><tr><td>0.0049-0.0215</td><td>87</td><td>0.6515</td><td>0.5124</td><td>0.5845</td><td>0.5153</td><td>0.2979</td><td>0.3023</td></tr><tr><td>0.0216-0.0774</td><td>87</td><td>0.6349</td><td>0.5053</td><td>0.6427</td><td>0.4904</td><td>0.6166</td><td>0.5575</td></tr><tr><td>&gt; 0.0774</td><td>88</td><td>0.7421</td><td>0.6080</td><td>0.6885</td><td>0.5900</td><td>0.6674</td><td>0.6810</td></tr><tr><td colspan="8">By dominant period (Eq. (4))</td></tr><tr><td>Period found</td><td>326</td><td>0.6106</td><td>0.5132</td><td>0.5714</td><td>0.4432</td><td>0.4059</td><td>0.3893</td></tr><tr><td>No period found</td><td>24</td><td>0.7388</td><td>0.5620</td><td>0.7532</td><td>0.6426</td><td>0.5709</td><td>0.4855</td></tr></table>

Second, sequence length has a minimal impact on overall rankings. On the multivariate track, LEAP ranks first in three of the four length bins and fourth in the 10,001–46,655 bin, where xLSTMAD-R, USAD and TimesNet lead by at most 0.035. On the univariate track, it leads the two longest bins outright, ranks second in the 7,501–18,228 bin, and drops to third of six in the shortest bin (0.5785 versus 0.6334 for PaAno<sub>PAI</sub>, $\Delta = - 0 . 0 5 4 9 )$ . Unlike channel count, this degradation is modest and confined to the short-series half of the set, where PaAno<sub>PAI</sub> leads, establishing sequence length as a secondary regime driver.

Third, absolute scores across all evaluated methods scale positively with the anomaly ratio, as ranking-based metrics inherently benefit from higher positive sample density (only 5 minor dips under 0.03 occur across 36 transitions between consecutive anomaly-ratio bins). However, LEAP’s advantage varies across tracks. On TSB-AD-M, its largest margin over the runner-up occurs in the lowest anomaly-ratio bin (+0.1299), decreasing to +0.0792, +0.0325, and +0.0456 in the remaining bins, highlighting its strength on sparse anomalies. On TSB-AD-U, LEAP also leads the sparsest bin, but only narrowly (+0.0094 over PaAno); its largest margin falls in the second bin (+0.0670 over $\mathrm { P a A n o _ { P A I } ) }$ , it trails $\mathrm { P a A n o _ { P A I } }$ marginally in the third (−0.0078), and it leads again in the densest band (+0.0536). The narrow univariate margin at low density aligns with structural differences in anomaly types across tracks. Sparse univariate anomalies are disproportionately point-scale (mean ratio of 0.017 versus 0.049 for other types), exposing $\mathbf { L E A P } \mathbf { \bar { s } }$ point-level sensitivity limitations. Conversely, multivariate point anomalies exhibit higher average density (0.079 versus 0.048 for other types), and the lowest multivariate ratio bin consists predominantly (80%) of level-shift, dispersion, or relational events, which directly match LEAP’s additive evidence criteria.

The dominant-period split requires an important caveat. All four multivariate series lacking a detected period belong to TAO recordings, whose anomaly ratios (0.094–0.141) fall in the highest anomaly-ratio bin (> 0.0715). Similarly, the 24 univariate series without a period display a higher mean anomaly ratio (0.061 versus 0.043). Because every method scores substantially higher on the no-period subset, this performance gap is confounded by anomaly density rather than reflecting an inherent advantage of the fallback window rule $( L = \operatorname* { m i n } ( L _ { \operatorname* { m a x } } , 1 6 p )$ in Eq. (6)). These results confirm that the fallback rule introduces no empirical liability, though the small sample sizes (4 and 24 series, respectively) preclude drawing stronger conclusions in either direction.

## I Extended Sensitivity Analysis

This appendix complements Section 5.2 by evaluating the two pipeline combination rules and the five architectural backbone constants.

## I.1 Combination Rules

The deployed architecture fixes two pipeline combination rules: local-moment fusion (level\_combine) and channel aggregation of the calibrated prediction errors (chan\_agg). Both were evaluated against candidate alternatives on the tuning set (Table I.1).

Table I.1: Effect of alternative combination rules for local-moment fusion (level\_combine) and channel aggregation (chan\_agg) on the 20-series tuning set. ∆ denotes the mean per-series VUS-PR difference relative to the deployed rule; Range indicates the minimum and maximum per-series $\bar { \Delta } ; p \cdot$ -value is from a paired Wilcoxon signed-rank test; W/L/T reports wins, losses, and ties relative to the deployed baseline at a 0.001 VUS-PR threshold.
<table><tr><td>Component</td><td>Rule</td><td>VUS-PR</td><td>∆</td><td>Range of ∆</td><td>p-value</td><td>W/L/T</td></tr><tr><td rowspan="4">level_combine</td><td>add (deployed)</td><td>0.6717</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>pmean</td><td>0.6715</td><td>-0.0002</td><td>[−0.022, +0.019]</td><td>0.275</td><td> $4 / 7 / 9$ </td></tr><tr><td>max</td><td>0.6725</td><td>+0.0008</td><td>[-0.041, +0.091]</td><td>0.395</td><td> $5 / 1 0 / 5$ </td></tr><tr><td>gate</td><td>0.6571</td><td>-0.0146</td><td>[−0.155, +0.017]</td><td>0.374</td><td> $8 / 8 \dot { / } 4$ </td></tr><tr><td rowspan="3">chan_agg</td><td>softmax (deployed)</td><td>0.6717</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>max</td><td>0.6717</td><td>+0.0000</td><td>[−0.0001, +0.0004]</td><td>0.135</td><td> $0 / 0 / 2 0$ </td></tr><tr><td>mean</td><td>0.6724</td><td>+0.0007</td><td>[-0.017, +0.013]</td><td>0.154</td><td> $6 / 3 / 1 1$ </td></tr></table>

Three of the four level\_combine rules achieve similar overall means, while gate trails by 0.0146. However, their per-series stability varies substantially. The max operator exhibits considerable variability across individual series (−0.041 to +0.091 VUS-PR) and achieves a marginal mean gain (+0.0008), despite recording twice as many losses as wins $\left( 5 / 1 0 / 5 \right)$ . The gate mechanism ranges from −0.155 to +0.017 VUS-PR, producing the largest mean decline (−0.0146) despite balanced wins and losses $( 8 / 8 / 4 )$ . While pmean exhibits greater stability $( - 0 . \bar { 0 } 2 2 { \mathrm { ~ t o ~ } } + 0 . 0 1 9 )$ , it trails add in both average score and win/loss ratio $( 4 / 7 / 9 )$ . As an unweighted linear sum, add provides a reliable baseline free of non-linear distortion. Because no alternative offers a statistically significant or systematic gain to justify added volatility, add is retained.

For chan\_agg, max is statistically indistinguishable from softmax, with all 20 tuning series tying within 0.001 VUS-PR. Although mean achieves a slightly higher overall average (+0.0007 VUS-PR; 6 wins, 3 losses), its per-series differences range from −0.017 to +0.013, with no statistically significant improvement (p-value = 0.154). This variability reinforces the sensitivity concerns noted for $w _ { \ell } , \beta ,$ and α in Section 5.2. Consequently, softmax is retained, as neither alternative demonstrates a significant or consistent advantage.

## I.2 Backbone Architecture Sensitivity

Five architectural constants of the masked-patch Transformer backbone were fixed prior to evaluation rather than jointly tuned: patch length p, depth B, width D, masking groups G, and mask block size b. To validate these choices, each constant was evaluated via single-variable grid sweeps on the 20-series tuning set while holding all remaining hyperparameters at their deployed values $( w _ { \ell } = 1 . \bar { 0 } , \beta = 1 . \bar { 0 } , \alpha = 0 . 2 5 , \gamma = 1 . 5 , \delta = 1 . 5 )$ . No candidate configuration outperformed the selected baseline $( p = 1 6 , B = 4 , D = 1 2 8 , G = 4 , b = 1 )$ , which was subsequently locked across all held-out evaluations. Table I.2 compares the top-performing alternative for each parameter against this baseline.

Table I.2: Sensitivity of the five architectural constants fixed a priori (patch length p, depth B, width D, masking groups G, mask block b), evaluated one-at-a-time on the 20-series tuning set while holding the remaining constants and all fusion coefficients at their deployed values. “Best alternative vs. selected” reports the single grid point with the highest mean VUS-PR among the non-selected values, its mean per-series ∆ against the selected value, and a paired Wilcoxon signed-rank p-value; W/L/T counts series with a gain, loss, or tie at a 0.001 VUS-PR threshold. The selected configuration attains a mean VUS-PR of 0.6717.
<table><tr><td>Constant</td><td>Grid tested</td><td>Selected</td><td>Best alternative (∆)</td><td>p-value</td><td>W/L/T</td></tr><tr><td>p (patch length)</td><td>{8, 16, 32}</td><td>16</td><td>32: -0.0347</td><td>0.014</td><td>4/14/2</td></tr><tr><td>B (depth)</td><td>{2, 4, 8}</td><td>4</td><td>2:-0.0070</td><td>0.891</td><td>7/7/6</td></tr><tr><td>D (width)</td><td>{64, 128, 256}</td><td>128</td><td>64: -0.0010</td><td>0.984</td><td>7/8/5</td></tr><tr><td>G (masking groups)</td><td>{2, 4,8}</td><td>4</td><td>2: -0.0053</td><td>0.225</td><td>6/8/6</td></tr><tr><td>b (mask block)</td><td>{1, 2, 4}</td><td>1</td><td>2: -0.0016</td><td>0.395</td><td> $5 / 9 / 6$ </td></tr></table>

Patch length is the sole architectural constant whose variation produces a statistically significant performance change. Both alternative patch resolutions induce notable drops $( p = 8 \colon \Delta = - 0 . 0 4 5 4 , \overbrace { p \mathrm { - } \mathrm { v a l u e } } = 0 . 0 0 3 ; p = 3 2 \colon \Delta =$ $- 0 . 0 3 4 7 , p \mathrm { - v a l u e = 0 . 0 1 4 ) }$ , confirming that patch granularity critically influences the scale of extracted temporal features. Conversely, varying depth, width, masking groups, or mask block size yields no statistically significant performance change (p-values between 0.225 and 0.984), with relatively balanced win/loss ratios and small mean differences $( \leq 0 . 0 0 7 0 \mathrm { { V U S - P R } } )$ . Once an appropriate patch size is fixed, backbone performance remains robust across the tested architectural variations.