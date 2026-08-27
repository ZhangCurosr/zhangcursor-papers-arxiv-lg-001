# CardioFusion-AI: Robust ECG–PPG Fusion for Multimodal Physiological Monitoring Under Signal Degradation

Navaneeth Krishnan K and Janakiraman K

Abstract— Wearable electrocardiogram (ECG) and photoplethysmogram (PPG) sensors are complementary but individually fragile: motion artifact, poor contact, and sensor dropout routinely degrade one or both signals. Fusion strategies that assume both modalities are equally trustworthy can become less reliable than a single clean modality once degradation sets in, yet the comparative robustness of different ECG–PPG fusion mechanisms under graded degradation and complete modality loss remains poorly characterized. We present CardioFusion-AI, a framework whose signal-processing front end (R-peak and systolic-peak detection, an Orphanidou-type signal-quality index, and beat-by-beat pulse transit time estimation) is validated on 53 real intensive-care recordings (848 windows, heart-rate mean absolute error 1.61 bpm ECG / 2.78 bpm PPG) and a real annotated fetal ECG database (R-peak F1 0.89–0.98). Building on this validated front end, we conduct a controlled, physiologically-grounded synthetic degradation study comparing eight ECG–PPG fusion strategies unimodal baselines, three fixed-weight fusion variants, attention fusion, and two sample-adaptive gated fusion mechanisms (one conditioned only on learned embeddings, one additionally conditioned on the validated signalquality index) – across six degradation regimes spanning graded corruption and complete modality loss, over five independent training seeds. Attention fusion achieved the lowest descriptive overall error (1.66±0.43 bpm). Both adaptive gates correctly reallocated weight toward the healthy modality under complete modality loss but showed near-zero correlation between gate weight and signal quality when both modalities were present but graded-degraded (r ≈ 0.10–0.24), and signal-quality conditioning produced a large, specific improvement under missing-PPG conditions (1.56±0.59 bpm, near the 1.48 bpm unimodal ceiling). With only five training seeds, no pairwise comparison survives Holm-corrected significance testing; we report effect sizes and confidence intervals accordingly. These results indicate that modality availability and modality quality are functionally distinct problems for adaptive fusion.

Index Terms— Adaptive fusion, electrocardiography, multimodal biosignal fusion, photoplethysmography, physiological monitoring, robust wearable sensing, signal quality assessment, signal-quality-aware fusion.

## I. INTRODUCTION

W <sup>EARABLE</sup> <sup>physiological</sup> <sup>monitoring</sup> <sup>increasingly</sup> <sup>re-</sup>lies on multiple complementary biosignals rather than lies on multiple complementary biosignals rather than any single sensor. Electrocardiography (ECG) and photoplethysmography (PPG) are the most common pairing in consumer and clinical wearables: ECG provides a direct electrical readout of cardiac timing, while PPG provides an optically derived pulse waveform from which heart rate, pulse timing, and – in combination with ECG – pulse transit time (PTT) can be estimated [1]. Because the two modalities are captured through different physical transduction mechanisms, they are also affected by different degradation processes, which is a large part of why fusing them is attractive: a corruption that disables one modality frequently leaves the other intact.

In practice, however, both modalities degrade routinely and often simultaneously. Motion artifact corrupts PPG far more readily than ECG; poor electrode contact or perspiration corrupts ECG independently of PPG; and either sensor can lose contact entirely for extended periods in ambulatory use. A fusion method that implicitly assumes both inputs are equally trustworthy can therefore become less reliable than a single well-behaved modality once one input deteriorates – naive averaging of a clean signal with a corrupted one degrades the estimate relative to simply discarding the corrupted input.

This motivates signal-quality-aware fusion: mechanisms that adjust how much each modality contributes based on how trustworthy it currently is. Several families of fusion strategy have been proposed for multimodal biosignal and, more broadly, multimodal machine learning problems, including early/input-level concatenation, feature-level fusion via a learned combination MLP, late/decision-level fusion with fixed or learned per-modality weights, cross-modal attention, and sample-adaptive gating in which a small network learns to reweight modalities on a per-input basis [2]. Adaptive gating is conceptually the most attractive of these for degraded wearable sensing, since it promises to reweight modalities exactly when and where degradation occurs, rather than applying a single global weighting scheme.

Existing evaluations of these fusion mechanisms are, however, typically conducted under clean or narrowly controlled conditions, or evaluate a single proposed mechanism against a small number of baselines. The comparative behavior of a broad set of fusion mechanisms – including whether an adaptive, signal-quality-conditioned gate actually learns to track graded signal degradation, as opposed to only the binary presence or absence of a modality – has not, to our knowledge, been systematically characterized under a controlled degradation protocol with matched architectures, matched training procedures, and an explicit accounting for the statistical power of the comparison.

In this work we address that gap directly. We build CardioFusion-AI, a two-part evaluation: (i) a signal-processing front end – peak detection, an Orphanidou-type signal-quality index (SQI) [3], and beat-by-beat PTT estimation – validated against real, independently sourced physiological data; and (ii) a controlled, physiologically-grounded synthetic degradation study in which eight ECG–PPG fusion architectures, sharing an identical encoder backbone and training protocol, are compared across six degradation regimes spanning graded single- and dual-modality corruption and complete modality loss. We deliberately frame this work in terms of physiological monitoring robustness rather than disease diagnosis or clinical risk stratification: the evidence we present concerns the reliability of physiological estimation under signal degradation, not diagnostic or prognostic performance, and should not be read as either.

The central hypothesis motivating this study is:

Signal-quality-aware and multimodal fusion mechanisms can improve the robustness of physiological estimates under ECG and/or PPG signal degradation, compared with fixed or unimodal estimation strategies; and sample-adaptive fusion guided explicitly by signal-quality information will show greater robustness to modality-specific degradation than fusion strategies without such conditioning.

As we show, the data support the first clause but only partially support the second, in a way that is itself informative about the mechanism.

The contributions of this paper are threefold:

1) A signal-quality-validated ECG–PPG fusion framework and controlled degradation-and-modality-loss evaluation protocol for physiological estimation, built on physiologically-grounded synthetic signal generation with independently controllable, graded per-modality corruption.

2) A systematic, matched-protocol comparison of eight unimodal and multimodal fusion strategies – including fixed-weight, attention-based, and implicit- and explicitsignal-quality-conditioned adaptive gating – across six degradation regimes and five independent training replicates, with pre-registered, multiplicity-corrected statistical analysis.

3) A mechanistic analysis of learned modality allocation showing that adaptive gates reliably detect complete modality absence but exhibit near-zero correlation with graded signal quality when both modalities are present, and that explicit signal-quality conditioning improves this behavior selectively rather than uniformly.

## II. METHODS

## A. Overview and Two-Part Evaluation Strategy

CardioFusion-AI combines two categories of evidence that are epistemically distinct and are kept explicitly separate throughout this paper. Claim A concerns the signal-processing front end – peak detection, SQI computation, and PTT estimation – and is supported by validation against real, thirdparty physiological recordings (Section II-B). Claim B concerns the comparative robustness of fusion architectures under degradation, which we study using a controlled synthetic protocol (Sections II-C–II-G) because no public dataset provides graded, independently-controllable, ground-truth-labeled degradation of both ECG and PPG simultaneously. Claim B numbers should not be interpreted as clinical performance figures; they establish comparative, mechanistic behavior among fusion strategies under a controlled corruption model.

## B. Real-Data Validation of the Signal-Processing Pipeline

The peak-detection and signal-quality components used throughout this work were validated, independently of the degradation study, against two public PhysioNet resources [4].

R-peak detection. Using the Abdominal and Direct Fetal ECG Database [5] (five records, cardiologist-verified beat annotations), direct-lead R-peak detection achieved an F1 score of 0.89–0.98 (mean 0.95) against expert annotation. Applying the same detector naively to the far noisier abdominal leads, without any signal-quality gating, collapsed F1 to 0.19–0.34, and per-record detection F1 correlated strongly with per-record SQI pass rate (Pearson r = 0.927), directly motivating qualityaware processing.

Heart-rate and pulse-rate agreement. Using the BIDMC PPG and Respiration Database [6] (53 real ICU patients, 30 s windows, 848 windows total), ECG-derived heart rate agreed with the bedside monitor reference with mean absolute error (MAE) 1.61 bpm (correlation 0.911) and PPG-derived pulse rate with MAE 2.78 bpm (correlation 0.879).

Pulse transit time. A beat-by-beat PTT estimator, validated on the same 53-subject BIDMC cohort (7,868 individually matched beats after fixing a boundary-artifact bug in an earlier whole-window cross-correlation approach), produced a physiologically plausible distribution (mean 113.9 ms, median 104.0 ms, SD 55.6 ms) consistent with a critically-ill ICU population.

These results establish that the front-end signal processing – and specifically the SQI computation used as an input to the signal-quality-conditioned fusion gate in Section II-E – is grounded in real, independently verifiable performance, even though the fusion-degradation comparison itself (Claim B) uses controlled synthetic data.

## C. Controlled Synthetic Degradation Study: Signal Generation

No public dataset provides simultaneous ECG and PPG with independently controllable, graded degradation severity and known ground-truth physiological state; real degraded recordings confound degradation type, severity, and physiological state in ways that make controlled comparison of fusion mechanisms impossible. We therefore generated physiologicallygrounded synthetic ECG and PPG using NeuroKit2 [7], whose ECG simulator implements the McSharry dynamical model [8] and whose PPG simulator includes built-in, literature-standard motion-artifact, baseline-drift, and burst-noise parameters.

TABLE I  
CONTROLLED DEGRADATION CONDITIONS. “LEVEL” SPANS CLEAN (0) THROUGH SEVERE (3); “MISSING” (4) DENOTES COMPLETE SENSOR DROPOUT AND IS EVALUATED ONLY AGAINST A CLEAN PARTNER MODALITY.
<table><tr><td>Regime</td><td>ECG level</td><td>PPG level</td><td># grid cells</td></tr><tr><td>Both clean</td><td>0</td><td>0</td><td>1</td></tr><tr><td>ECG degraded only</td><td>1-3</td><td>0</td><td>3</td></tr><tr><td>PPG degraded only</td><td>0</td><td>1-3</td><td>3</td></tr><tr><td>Both degraded</td><td>1-3</td><td>1-3</td><td>9</td></tr><tr><td>ECG missing</td><td>4</td><td>0</td><td>1</td></tr><tr><td>PPG missing</td><td>0</td><td>4</td><td>1</td></tr></table>

For each 8 s window (sampling rate 125 Hz, 1000 samples), a target heart rate was drawn as HR ∼ U(50, 110) bpm and used to generate a clean ECG and a clean PPG waveform sharing that heart rate. Each modality was then independently corrupted to one of five severity levels: clean (0), mild (1), moderate (2), severe (3), or missing (4, complete sensor dropout modeled as a noise floor with no cardiac structure). ECG degradation combined additive Gaussian noise, lowfrequency baseline wander, and randomly-timed motion bursts, with severity-scaled amplitudes; PPG degradation used NeuroKit2’s native motion-amplitude, powerline, drift, and burstnoise parameters, also severity-scaled. Table I summarizes the resulting evaluation grid, which is the full factorial of graded co-degradation (16 combinations, levels $0 { - } 3 \times 0 { - } 3 )$ plus two targeted complete-modality-loss conditions, collapsing to six analysis regimes.

Windows are i.i.d. synthetic draws with no subject identity, so there is no possibility of subject-level leakage across splits; independence across the train/validation/test partition instead follows directly from using disjoint random-seed ranges for signal generation in each split (Section II-F). Dataset sizes (792 train / 234 validation / 324 test window pairs, approximately 44/13/18 windows per grid cell) were chosen to keep the eight-architecture × five-seed sweep computationally tractable on a single CPU core while providing at least ∼20 test windows per regime cell for stable per-regime error estimates; we report this as a scale limitation in Section V rather than presenting it as an optimized choice.

The regression target is the true, generator-known heart rate (bpm) for the window, which – unlike reference values derived from noisy real recordings – provides an exact ground truth against which fusion-strategy accuracy can be compared without reference-signal error confounding the comparison.

## D. Signal Quality Assessment Applied to Synthetic Windows

The SQI used both diagnostically and as an input feature to the signal-quality-conditioned fusion gate is the same Orphanidou-type index validated in Section II-B [3]: beat detection followed by a template-correlation-based quality score and a feasibility flag (physiologically plausible rate and beat-to-beat regularity). We emphasize that this SQI is a signal-quality descriptor, derived from waveform morphology and rate regularity, not a ground-truth quality label; it can and does fail to discriminate in some conditions (Section IV). Applied to the synthetic test windows, mean SQI decreased monotonically with degradation severity for both modalities (ECG: 0.988 at level 0 to 0.946 at level 3; PPG: 0.995 to 0.952) and collapsed sharply under complete dropout (ECG missing: mean SQI 0.19, feasibility-pass rate 27%; PPG missing: mean SQI 0.05, feasibility-pass rate 14%), confirming the SQI is directionally informative before it is used as a gate input.

![](images/3bb0c02b227e777f7ae702c1c15adff1c58c352f0aa9cb3b437f690afe32ef58.jpg)  
Fig. 1. Shared CardioFusion-AI dual-stream architecture. Each modality is independently preprocessed, quality-assessed, and encoded; the fusion mechanism (the only component that differs across the eight compared strategies) combines the two embeddings into a physiological estimate. The signal-quality index is an explicit gate input only in the SQI-conditioned adaptive variant.

## E. Fusion Architectures

Fig. 1 shows the shared dual-stream architecture common to all eight strategies; they differ only in the fusion mechanism $g ( \cdot )$ that combines the two per-modality embeddings.

All eight architectures share an identical single-modality encoder backbone: a four-block 1-D CNN (kernel size 7, channel progression 16–32–64–64, batch normalization, ReLU, dropout 0.2, max-pooling after each block) followed by global average pooling and a linear projection to a $d \_ =$ 64-dimensional embedding, $\begin{array} { r c l c l } { e _ { \mathrm { e c g } } } & { = } & { f _ { \theta _ { \mathrm { e c g } } } ( x _ { \mathrm { e c g } } ) , } & { e _ { \mathrm { p p g } } } & { = } \end{array}$ $f _ { \theta _ { \mathrm { p p g } } } ( x _ { \mathrm { p p g } } ) \ \in \ \mathbb { R } ^ { 6 4 }$ . Holding the encoder architecture fixed isolates the fusion mechanism as the only varying factor. A regression head $h : \mathbb { R } ^ { 6 4 }  \mathbb { R }$ (two-layer MLP, ReLU, dropout 0.3) maps the fused representation to a scalar heartrate estimate $\hat { y } .$

1) ECG-only / PPG-only: $\hat { y } ~ = ~ h ( e _ { \mathrm { e c g } } )$ or $h ( e _ { \mathrm { p p g } } )$ ; no fusion.

2) Fixed-average fusion: $\begin{array} { r } { \hat { y } = h \big ( \frac { 1 } { 2 } e _ { \mathrm { e c g } } + \frac { 1 } { 2 } e _ { \mathrm { p p g } } \big ) } \end{array}$ , a static, input-independent 50/50 combination – the most literal realization of “fixed fusion.”

3) Feature-level fusion: $\hat { y } = h ( \mathrm { M L P } ( [ e _ { \mathrm { e c g } } ; e _ { \mathrm { p p g } } ] ) )$ , where the fusion MLP is learned during training but, once trained, applies the same transformation to every input regardless of quality.

4) Global-weighted late fusion: independent per-modality regression heads $h _ { \mathrm { e c g } } , h _ { \mathrm { p p g } }$ produce $\hat { y } _ { \mathrm { e c g } } = h _ { \mathrm { e c g } } ( e _ { \mathrm { e c g } } )$ $\hat { y } _ { \mathrm { p p g } } = h _ { \mathrm { p p g } } ( e _ { \mathrm { p p g } } )$ , combined as $\hat { y } = w _ { 1 } \hat { y } _ { \mathrm { e c g } } + w _ { 2 } \hat { y } _ { \mathrm { p p g } }$ with w = softmax(ϕ), $\phi \in \mathbb { R } ^ { 2 }$ a single learned parameter pair shared across all samples (fixed at inference).

5) Attention fusion: bidirectional multi-head crossattention [10] (4 heads) between the pooled embeddings, treated as length-1 token sequences: $\begin{array} { r l } { \tilde { e } _ { \mathrm { e c g } } } & { { } = } \end{array}$ $\mathrm { L N } ( e _ { \mathrm { e c g } } + \mathrm { M H A } ( e _ { \mathrm { e c g } } , e _ { \mathrm { p p g } } , e _ { \mathrm { p p g } } ) )$ and symmetrically for $\tilde { e } _ { \mathrm { p p g } } ; ~ \hat { y } ~ = ~ h ( [ \tilde { e } _ { \mathrm { e c g } } ; \tilde { e } _ { \mathrm { p p g } } ] )$ . Attention weights are input-dependent but are not explicitly supervised by any signal-quality signal.

6) Adaptive gate (implicit): a gate network $g \quad =$ softmax $\begin{array} { r } { \mathrm {  ~ \psi ~ } _ { \mathrm { { z } } } ( \mathrm { M L P } ( [ e _ { \mathrm { e c g } } ; e _ { \mathrm { p p g } } ] ) ) \in \mathrm { ~ \mathbb { \Delta } ~ } \Delta ^ { 1 } } \end{array}$ produces a $p e r { - }$ sample weight pair from the embeddings alone; $\hat { y } =$ $h ( g _ { 1 } e _ { \mathrm { e c g } } + g _ { 2 } e _ { \mathrm { p p g } } )$

7) Adaptive gate (SQI-conditioned): identical to (6) except the gate additionally receives the real SQI descriptor $s \ = \ [ \mathrm { S Q I } _ { \mathrm { e c g } } , \mathrm { f e a s } _ { \mathrm { e c g } } , \mathrm { S Q I } _ { \mathrm { p p g } } , \mathrm { f e a s } _ { \mathrm { p p g } } ] \ \in \ \mathbb { R } ^ { 4 }$ from Section $\mathrm { I I - B : } \bar { g } = \mathrm { s o f t m a x } ( \bar { \mathrm { M L P } } ( [ e _ { \mathrm { e c g } } ; e _ { \mathrm { p p g } } ; s ] ) )$

Architectures 3, 5, 6, and 7 reuse the CardioFusion-AI repository’s own implementations unmodified (with the classification head reduced to a single output unit for regression); architectures 2 and 4 were added for this study as literal fixed-fusion comparators, since no fusion strategy in the base repository was fully input-independent at inference.

## F. Experimental Protocol

Each of the eight architectures was trained independently under five random seeds (0–4), which govern both parameter initialization and minibatch ordering; the train/validation/test partition itself was fixed and shared across all seeds and architectures, generated once using disjoint random-numbergenerator seed ranges per split (Section II-C). All models were trained with Adam (learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ gradient-norm clipping at 1.0 – both introduced after an initial pilot run diverged for one architecture/seed combination, then applied uniformly to every architecture for fairness), batch size 32, mean-squared-error loss, up to 30 epochs with early stopping (patience 6) on validation MSE. Evaluation used the single fixed test partition (324 window pairs) for every architecture and seed.

## G. Statistical Analysis

For each architecture and regime we report the mean and standard deviation of test MAE across the five seeds. For a pre-registered family of three architecture comparisons (SQIconditioned vs. implicit adaptive gate; SQI-conditioned adaptive gate vs. attention; implicit adaptive gate vs. attention), evaluated across seven metrics (overall MAE and the six regime-specific MAEs; 21 tests total), we treat the seed – one independently initialized and trained model instance – as the unit of statistical inference, since the test partition is identical across seeds and pooling seed × window pairs as if independent would be pseudo-replication and would invalidate resulting p-values. We report the paired difference in per-seed mean MAE $( n = 5 )$ , an exact paired Wilcoxon signed-rank test, a paired t-test, a paired effect size (Cohen’s $d _ { z } ) _ { \ast }$ , and a t-based 95% confidence interval on the mean difference, explicitly caveated given $n \ : = \ : 5$ . Holm–Bonferroni correction is applied across the full 21-test family. We separately report the correlation between SQI and learned gate weight (Section III-D), computed per seed and averaged, to avoid the same pseudo-replication problem.

![](images/60cdd2697b57caebc37c46c399012732399dfebde592da424b52b3684e8bfaf0.jpg)  
Fig. 2. Test MAE vs. graded degradation severity $( 0 = { \mathsf { c l e a n } } \ { \mathsf { t o } } \ 3 =$ severe), with the partner modality held clean. Error bars: SD across 5 seeds.

## III. RESULTS

## A. Overall Fusion Performance

Table II reports test MAE (mean ± SD across five seeds) for all eight architectures across the six degradation regimes. Attention fusion exhibited the lowest descriptive overall MAE $( 1 . 6 6 \pm 0 . 4 3 \mathrm { b p m } )$ , followed by the implicit $( 2 . 0 6 \pm 0 . 5 1$ bpm) and SQI-conditioned $( 1 . 9 9 \pm 0 . 6 1$ bpm) adaptive gates; fixedweight fusion variants (fixed-average, feature-level, globalweighted late) and the single-modality baselines were descriptively worse (2.18–2.87 bpm). We report these as descriptive comparisons: as detailed in Section III-E, none of the pairwise comparisons among attention fusion and the two adaptive gates survives Holm-corrected significance testing at n = 5 seeds.

## B. Performance Across Degradation Severity

Fig. 2 shows test MAE as a function of graded degradation severity, isolated separately for ECG degradation (PPG held clean) and PPG degradation (ECG held clean). Performance trajectories differed markedly by degradation type and fusion mechanism rather than degrading uniformly. Under graded PPG degradation, attention fusion remained the strongest strategy at every severity level. Under graded ECG degradation, by contrast, attention fusion’s advantage eroded with increasing severity and reversed at the severe level, where a naive PPGonly estimator outperformed it (1.45 vs. 2.15 bpm) – attention fusion’s overall advantage in Table II is therefore not uniform across degradation types, and severe single-modality ECG degradation is a specific failure mode for the otherwisestrongest architecture.

## C. Robustness to Complete Modality Loss

Complete modality loss (Table II, rightmost two columns) separated the architectures far more sharply than graded degradation. Single-modality baselines behaved exactly as expected – catastrophic when their own modality is missing (ECGonly: 14.34 bpm with ECG missing; PPG-only: 15.02 bpm with PPG missing) and near-ceiling when the other modality is missing (ECG-only: 1.48 bpm with PPG missing; PPGonly: 1.47 bpm with ECG missing), which we treat as an

TABLE II  
TEST-SET MAE (BPM), $\mathsf { M E A N } \pm \mathsf { S D }$ ACROSS 5 INDEPENDENT TRAINING SEEDS, BY DEGRADATION REGIME.
<table><tr><td>Model</td><td>Overall</td><td>Both clean</td><td>ECG degraded</td><td>PPG degraded</td><td>Both degraded</td><td>ECG missing</td><td>PPG missing</td></tr><tr><td>ECG-only</td><td>2.18±0.45</td><td>1.86±0.76</td><td> $1 . 5 2 { \pm } 0 . 4 1$ </td><td> $1 . 4 1 { \pm } 0 . 5 6 $ </td><td>1.43±0.41</td><td>14.34±0.38</td><td>1.48±0.67</td></tr><tr><td>PPG-only</td><td>2.61±0.30</td><td>1.49±0.32</td><td> $1 . 3 8 \pm 0 . 3 0$ </td><td> $1 . 9 2 { \pm } 0 . 2 2$ </td><td> $2 . 1 3 { \pm } 0 . 3 5 $ </td><td>1.47±0.34</td><td>15.02±1.51</td></tr><tr><td>Fixed-average fusion</td><td>2.63±0.64</td><td>2.39±0.71</td><td> $2 . 1 1 { \pm } 0 . 8 9$ </td><td> $2 . 1 4 \pm 0 . 7 3$ </td><td>2.17±0.76</td><td>7.01±0.82</td><td>5.77±1.33</td></tr><tr><td>Feature-level fusion</td><td>2.87±0.53</td><td>2.53±0.57</td><td> $2 . 1 7 { \pm } 0 . 3 1$ </td><td> $2 . 2 5 { \pm } 0 . 5 3 $ </td><td>2.40±0.67</td><td>6.95±0.76</td><td>7.28±1.22</td></tr><tr><td>Attention fusion</td><td>1.66±0.43</td><td>1.24±0.26</td><td> $1 . 6 5 { \pm } 0 . 3 9$ </td><td> ${ \bf 1 . 1 8 \pm 0 . 1 8 }$ </td><td>1.55±0.31</td><td>3.17±1.91</td><td>2.97±1.73</td></tr><tr><td>Global-weighted late fusion</td><td>2.67±0.30</td><td>1.98±0.44</td><td> $1 . 8 3 { \pm } 0 . 6 2$ </td><td> $2 . 0 5 { \pm } 0 . 3 5$ </td><td>2.26±0.40</td><td>6.86±0.54</td><td>7.24±0.80</td></tr><tr><td>Adaptive gate (implicit)</td><td>2.06±0.51</td><td>1.52±0.36</td><td> $1 . 7 3 { \pm } 0 . 1 6$ </td><td> $1 . 7 2 { \pm } 0 . 4 4$ </td><td>1.77±0.40</td><td>5.38±4.11</td><td>3.85±5.54</td></tr><tr><td>Adaptive gate (SQI-cond.)</td><td>1.99±0.61</td><td>1.98±0.48</td><td> $1 . 8 8 { \pm } 0 . 6 9$ </td><td> $1 . 8 1 { \pm } 0 . 5 5 $ </td><td>1.87±0.68</td><td>4.36±1.59</td><td>1.56±0.59</td></tr></table>

![](images/2e9a2f7b74cd39e04956993598a435ee984291925438b507ac34fc55b75759e7.jpg)  
Fig. 3. Test MAE by degradation regime for five representative fusion strategies. Complete modality loss (right two groups) separates strategies far more sharply than graded degradation (left four groups).

Fixed-weight fusion strategies (fixed-average, feature-level, global-weighted late) could not approach this ceiling under either missing-modality condition (5.77–7.28 bpm), since none can reduce a dead channel’s contribution at inference. The SQI-conditioned adaptive gate came within 0.08 bpm of the unimodal ceiling under missing PPG $( 1 . 5 6 \pm 0 . 5 9$ bpm vs. 1.48 bpm), while under missing ECG it reached $4 . 3 6 \pm 1 . 5 9$ bpm – better than the implicit gate $( 5 . 3 8 \pm 4 . 1 1$ bpm) and the fixed-weight strategies, but well short of the corresponding 1.47 bpm ceiling. This ECG/PPG asymmetry is examined mechanistically in Section III-D.

approximate empirical ceiling for what any fusion strategy relying on the surviving modality alone could achieve.

## D. Adaptive Gate Behavior and Effect of Explicit SQI Conditioning

Fig. 4 shows the mean learned gate weight allocated to ECG, by regime, for both adaptive gates. Both gates allocate the large majority of weight to ECG when both modalities are present (implicit: 0.74–0.77 across clean and degradedbut-present regimes; SQI-conditioned: 0.975–0.998), and both correctly collapse to allocate $> 9 8 \%$ of weight to the surviving modality under complete dropout of the other (implicit: 0.009 with ECG missing, 1.000 with PPG missing; SQI-conditioned: 0.015 with ECG missing, 1.000 with PPG missing). Critically, however, neither gate’s weight allocation moves appreciably between the clean and severely-degraded-but-present conditions (implicit: $0 . 7 3 8  0 . 7 4 3 $ SQI-conditioned: 0.975 → 0.980, ECG-severity axis) – both gates behave as an approximately step-like function of modality presence, not a graded

![](images/183166ee8080c7b87b12aaef2a157759117ca83a32cf65874f5c839c36770fb6.jpg)  
Fig. 4. Mean learned gate weight on ECG vs. PPG by degradation regime, for the implicit and SQI-conditioned adaptive gates. Both gates reallocate weight sharply only under complete modality loss (rightmost two regimes).

function of modality quality.

We quantified this directly by correlating, per seed, the learned gate weight on ECG against the SQI difference $s _ { \mathrm { e c g } } - s _ { \mathrm { p p g } }$ across all test windows. Pooled across all regimes (including the missing-modality extremes), both gates show a strong association (implicit: mean Pearson $r = 0 . 6 8 ;$ SQIconditioned: $r = 0 . 7 0$ , across seeds). Restricted to windows where both modalities are present but graded-degraded – the condition that actually determines whether a gate is qualityproportional rather than merely presence-detecting – this association collapses (implicit: mean $r = 0 . 2 4 $ ; SQI-conditioned: mean $r ~ = ~ 0 . 1 0 )$ . That is, giving the gate direct numerical access to a validated signal-quality descriptor did not, in this experiment, teach it to use that information proportionally when both modalities remained present; the SQI-conditioned gate’s advantage over the implicit gate is concentrated specifically in the missing-PPG condition (Section III, Table II), not in graded-degradation responsiveness.

## E. Statistical Significance

Table III summarizes the pre-registered, seed-level paired comparisons $( n = 5 )$ with Holm correction across all 21 tests. No comparison survives correction. The two nominally closest comparisons (SQI-conditioned adaptive gate vs. attention fusion, on the “both clean” and “PPG degraded only” regimes) reached the minimum possible exact two-sided Wilcoxon $p \textmd { - }$ value at $n = 5 \ ( p = 0 . 0 6 2 5 , \mathrm { i . e . }$ , a consistent direction across all five seeds) with large paired effect sizes (Cohen’s $d _ { z } = 2 . 5 1$ and 1.50, respectively), but neither survives Holm correction $( p _ { \mathrm { H o l m } } = 1 . 0 ) $ . We report these as directionally consistent, large-effect-size, exploratory findings rather than confirmed differences, and we consider this an explicit limitation of a five-seed study rather than a null result to be minimized (Section V).

TABLE III  
SEED-LEVEL PAIRED COMPARISONS $( n = 5 )$ , SELECTED REGIMES. FULL 21-TEST FAMILY IN SUPPLEMENTARY MATERIAL.
<table><tr><td>Comparison</td><td>Regime</td><td>Mean diff.</td><td> $d _ { z }$ </td><td>p</td></tr><tr><td>SQI vs. Attn.</td><td>both clean</td><td>+0.74</td><td>2.51</td><td>0.063</td></tr><tr><td>SQI vs. Attn.</td><td>PPG degraded</td><td>+0.63</td><td>1.50</td><td>0.063</td></tr><tr><td>SQI vs. Implicit</td><td>overall</td><td>-0.07</td><td>0.18</td><td>0.81</td></tr><tr><td>Implicit vs. Attn.</td><td>overall</td><td>+0.40</td><td>0.56</td><td>0.31</td></tr></table>

## IV. DISCUSSION

## A. Why attention fusion performed best, with a specific exception

Attention fusion’s bidirectional cross-attention allows each modality’s contribution to the fused representation to vary with the input, without requiring an explicit, hand-specified quality signal; unlike the gates in this study, it also retains both attended representations (concatenated) rather than compressing to a single convex combination, giving it more representational capacity at the fusion step. This is consistent with its descriptively strongest overall performance. Its reversal under severe, single-modality ECG degradation (Section III) indicates that this capacity does not translate into robustness against every degradation type, and that the mechanism most often recommended for its flexibility is not uniformly the most robust choice.

## B. Why adaptive gating did not track graded quality

The gate architectures in this study learn from a 64- dimensional pooled embedding (optionally augmented with a 4-dimensional SQI vector) trained end to end against a single scalar heart-rate loss. A plausible explanation for the observed presence/quality asymmetry is that complete modality dropout produces a large, easily separable signal in embedding space (and an extreme SQI value), while graded degradation produces a comparatively subtle, continuous shift that a small gate network trained on a modest number of windows may simply not be incentivized to track, since the regression loss can often still be minimized by defaulting to whichever modality is globally more informative on average. This is consistent with – though our design does not by itself establish causally – a form of modality dominance during joint training [9], discussed further below.

## C. Why SQI conditioning helped selectively

Explicit SQI conditioning produced its clearest benefit exactly where the SQI signal itself is least ambiguous: complete PPG dropout collapses PPG SQI to near zero, a regime the gate can separate easily. Under graded degradation, where SQI changes are smaller and the underlying regression task may remain solvable without a large weight shift, the SQIconditioned gate did not show materially different weightallocation behavior from the implicit gate (Section III-D).

We regard this as the paper’s central mechanistic finding: modality availability and modality quality are functionally distinct problemsfor adaptivefusion, and providing a validated <sub>p</sub> quality descriptor as an input feature does not, by itself, $\overline { { 1 . 0 \mathfrak { G } } }$ uarantee the network learns to use it proportionally.

$$
\mathrm { 1 . 0 0 } ^ { \cdot } \mathrm { \bar { 2 } } .
$$

Modality competition as a plausible contributing factor .00 Even when the SQI-conditioned gate correctly allocated ∼99% of weight to PPG under ECG dropout, its accuracy (4.36 bpm) remained well short of a dedicated PPG-only model evaluated on the identical windows (1.47 bpm) – despite the gate’s decision being, in that regime, essentially correct. A plausible explanation is that because both gates favor ECG for the substantial majority of training windows (Section III-D), the PPG-specific encoder receives systematically less gradient signal during joint training than it would in isolation, a pattern consistent with previously reported modality competition or modality dominance effects in multimodal network training [9]. We phrase this as a plausible contributing explanation rather than an established causal mechanism, since this study was not designed with the ablations (e.g., modality-balanced training curricula, gradient-magnitude monitoring) needed to establish it directly.

## V. LIMITATIONS

1) Five independent training seeds. This provides adequate power to report consistent-direction trends and effect sizes but not confirmatory pairwise significance after multiplicity correction; the two largest observed effects $( d _ { z } > 1 . 5 )$ are promising candidates for confirmation in a higher-replicate follow-up.

2) Controlled, synthetic degradation. The degradation model is physiologically grounded (Section II-C) but is a controlled approximation of real-world corruption; absolute error magnitudes should not be read as realworld deployment performance, only as a comparative signal among architectures evaluated identically.

3) Modest dataset scale. 792/234/324 train/validation/test window pairs, chosen for computational tractability, is small by deep-learning standards; per-regime test cells (as few as ∼18–90 windows) limit the precision of regime-specific estimates.

4) The gate and correlation analyses are descriptive, not causal. They demonstrate consistent behavioral patterns but do not, by themselves, establish the underlying training-dynamics mechanism (Section IV).

5) Asymmetric strength of evidence. The signal-processing front end (Section II-B) is validated on real, independently sourced data; the comparative fusion-architecture findings (Sections III–III-D) are not, and real-world robustness of the learned fusion models has not been established.

## VI. CONCLUSION

We evaluated eight ECG–PPG fusion strategies under a controlled, physiologically-grounded degradation protocol spanning graded corruption and complete modality loss, building on a signal-processing front end independently validated against real ICU and fetal ECG recordings. Fusion-mechanism choice produced large, consistent differences in robustness. Attention fusion was the strongest general-purpose strategy but not uniformly $\mathbf { S O } ,$ failing specifically under severe singlemodality ECG degradation. Both adaptive gating mechanisms correctly detected complete modality loss but showed little evidence of tracking graded signal quality when both modalities remained present; explicit conditioning on a validated signal-quality descriptor improved missing-modality robustness in one specific, asymmetric direction without resolving this broader limitation. We interpret modality availability and modality quality as distinct problems that adaptive multimodal physiological monitoring systems must address separately, and we report our statistical evidence, including its limits, explicitly rather than overstating what a five-seed comparative study can support.

## DATA AND CODE AVAILABILITY

The CardioFusion-AI signal-processing and fusionarchitecture code is available at the project repository. Synthetic data generation code, trained-model evaluation scripts, and the full seed-level results tables underlying every reported statistic are available as supplementary material.

[1] R. Mukkamala, J.-O. Hahn, O. T. Inan, L. K. Mestha, C.-S. Kim, H. Toreyin, and S. Kyal, “Toward ubiquitous blood pressure monitoring via¨ pulse transit time: Theory and practice,” IEEE Trans. Biomed. Eng., vol. 62, no. 8, pp. 1879–1901, Aug. 2015.

[2] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machineˇ learning: A survey and taxonomy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 2, pp. 423–443, Feb. 2019.

[3] C. Orphanidou, T. Bonnici, P. Charlton, D. Clifton, D. Vallance, and L. Tarassenko, “Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring,” IEEE J. Biomed. Health Inform., vol. 19, no. 3, pp. 832–838, May 2015.

[4] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. Ch. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” Circulation, vol. 101, no. 23, pp. e215–e220, Jun. 2000.

[5] J. Jezewski, A. Matonia, T. Kupka, D. Roj, and R. Czabanski, “Determination of the fetal heart rate from abdominal signals: Evaluation of beat-to-beat accuracy in relation to the direct fetal electrocardiogram,” Biomed. Tech. (Berl.), vol. 57, no. 5, pp. 383–394, Jul. 2012.

[6] M. A. F. Pimentel, A. E. W. Johnson, P. H. Charlton, D. Birrenkott, P. J. Watkinson, L. Tarassenko, and D. A. Clifton, “Toward a robust estimation of respiratory rate from pulse oximeters,” IEEE Trans. Biomed. Eng., vol. 64, no. 8, pp. 1914–1923, Aug. 2017.

[7] D. Makowski, T. Pham, Z. J. Lau, J. C. Brammer, F. Lespinasse, H. Pham, C. Scholzel, and S. H. A. Chen, “NeuroKit2: A Python toolbox¨ for neurophysiological signal processing,” Behav. Res. Methods, vol. 53, no. 4, pp. 1689–1696, Aug. 2021.

[8] P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, “A dynamical model for generating synthetic electrocardiogram signals,” IEEE Trans. Biomed. Eng., vol. 50, no. 3, pp. 289–294, Mar. 2003.

[9] W. Wang, D. Tran, and M. Feiszli, “What makes training multi-modal classification networks hard?” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 12695–12705.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017.