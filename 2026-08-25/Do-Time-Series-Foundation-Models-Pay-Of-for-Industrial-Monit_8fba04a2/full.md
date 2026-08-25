# Do Time-Series Foundation Models Pay Of for Industrial Monitoring? A Cost-Aware Empirical Study

Guan-Hua Wen Kuan-Yu Chen   
Department of Computer Science and Information Engineering National Taiwan University of Science and Technology Taipei, Taiwan   
B11215005@mail.ntust.edu.tw, kychen@mail.ntust.edu.tw

## Abstract

Industrial monitoring models must detect operationally relevant deviations while satisfying target-specific data, calibration, and resource constraints. Time-series foundation models (TSFMs) promise reusable representations and zeroshot forecasts, yet evidence for their deployment value remains mixed when task definitions are heterogeneous and lightweight baselines are competitive. This work presents a protocol-aware empirical assessment across three settings: a C-MAPSS degradation-risk proxy, normal-only training for anomalous-sound detection on MIMII, and BDG2 forecasting-residual diagnostics with synthetic target perturbations. We assess classical one-class methods, compact neural autoencoders, residual forecasters, MOMENT-small, Chronos-T5, and TimesFM 2.5 in terms of anomaly-ranking performance, risk-horizon sensitivity, residual forecasting and perturbation sensitivity, and local implementation cost. Across 100 C-MAPSS engines evaluated out of fold, TCN-AE reaches fold-weighted AUROC/AUPRC 0.9570/0.8960, compared with 0.7310/0.3080 for MOMENT reconstruction; paired engine-cluster bootstrap confidence intervals exclude zero for both diferences. Across five matched MIMII pump evaluations, OCSVM also exceeds MOMENT reconstruction in AUROC and AUPRC. On a fixed 12-meter BDG2 panel, TimesFM 2.5 has the lowest aligned forecast error and the highest synthetic AUROC point estimate, although synthetic AUPRC is similar across TSFM and fitted residual models. Same-device measurements show that MOMENT incurs higher latency, peak allocated VRAM, and serialized state-dictionary size than TCN-AE. Under the evaluated frozen and zero-shot settings, TSFMs are task-dependent deployment options rather than default replacements for fitted lightweight models.

## 1 Introduction

Industrial monitoring supports decisions that range from maintenance scheduling to fault investigation and energysystem operation. Useful models must match target labels and evaluation units while meeting local resource constraints. TSFMs promise representations or forecasts that transfer with little target-specific adaptation [Goswami et al., 2024,

Ansari et al., 2024, Das et al., 2024], but their deployment value is uncertain when target-normal data and competitive lightweight models are available.

The target tasks do not share a native anomaly definition. C-MAPSS provides run-to-failure trajectories, MIMII supports normal-only acoustic anomaly detection, and the BDG2 data used here lack real-fault labels [Saxena and Goebel, 2008, Purohit et al., 2019, Koizumi et al., 2020, Miller et al., 2020]. We therefore compare models only within protocol, using domain-specific evaluation units, metrics, and uncertainty procedures rather than a cross-domain leaderboard. The study asks four deployment questions:

1. Under matched engine- or file-level protocols, do frozen TSFM modes improve anomaly ranking over fitted lightweight models?

2. Does the C-MAPSS model ordering persist when the degradation-risk horizon changes?

3. Do zero-shot forecasting TSFMs improve aligned BDG2 forecasting residuals and sensitivity to controlled perturbations?

4. What latency, memory, and model-size costs accompany any predictive gain under a standardized local workload?

Across classical estimators, compact neural models, and three TSFM families, we contribute matched C-MAPSS and MIMII comparisons, a prespecified multi-meter BDG2 diagnostic, grouped uncertainty analyses, and standardized local cost metadata. This study provides an empirical deployment comparison; it does not propose a new detector, claim state-of-the-art performance, or establish a universal ranking of model families.

## 2 Related Work

Time-series anomaly detection. Recent surveys organize deep approaches around reconstruction, forecasting, density estimation, contrastive learning, and graph- or attention-based architectures [Zamanzadeh Darban et al., 2025].

Industrial transfer and domain shift. Changes in operating regime, machine identity, or sensing conditions can degrade a detector fitted to a source distribution. Deep transfer-learning surveys emphasize the potential to reduce retraining and labeling requirements, while warning that source–target mismatch must be handled explicitly [Yan et al., 2024].

Acoustic machine anomaly detection. MIMII was introduced to support machine-sound anomaly detection under realistic factory noise, with normal and anomalous recordings from valves, pumps, fans, and slide rails [Purohit et al., 2019]. DCASE 2020 Task 2 formalized unsupervised anomalous sound detection in which only normal sounds are available for training and unknown abnormal sounds must be detected at test time [Koizumi et al., 2020]. Our experiments adopt the normal-only training premise of DCASE Task 2 but do not reproduce its complete machine-ID-specific scoring protocol.

Building energy residual monitoring. BDG2 contains large-scale building meter data associated with the ASHRAE Great Energy Predictor III context [Miller et al., 2020]. Because the BDG2 data used here do not provide equipment-fault labels, we report forecasting errors and separately use injected perturbations to probe residual sensitivity.

Time-series foundation models. MOMENT introduced an open family of pretrained models for forecasting, classification, anomaly detection, and other time-series tasks [Goswami et al., 2024]. Chronos adapts language-model architectures to probabilistic forecasting by scaling and quantizing time-series values into tokens [Ansari et al., 2024]. TimesFM uses a decoder-only architecture for zero-shot forecasting [Das et al., 2024]. Recent work such as ChronosAD explores foundation-model representations specifically for anomaly detection [Khan et al., 2026]. An empirical assessment in power-system forecasting evaluates TSFMs across zero-shot accuracy, fine-tuning eficiency, horizon sensitivity, covariate handling, and unseen-site generalization [Za’ter and Hodge, 2026]. We instead assess frozen or zero-shot deployment modes across matched industrial monitoring protocols and local implementation costs.

## 3 Methods

## 3.1 Task Framing

The three domains do not share a native anomaly definition. Treating them as entries in one leaderboard would therefore be misleading. Instead, we use a common interface,

$$
x \longmapsto s ( x ) \in \mathbb { R } ,
$$

mapping each input sequence or window to a scalar monitoring score while preserving domain-specific interpretation. Table 1 summarizes the resulting protocols.

For C-MAPSS, multivariate trajectories are divided into sliding windows. For engine i with terminal cycle $T _ { i } .$ , a window ending at cycle t receives

$$
\mathrm { R U L } _ { i , t } = T _ { i } - t , \qquad y _ { i , t } ^ { ( h ) } = \mathbf { 1 } \{ \mathrm { R U L } _ { i , t } \leq h \} .
$$

The main analysis uses $h \ = \ 3 0$ cycles and evaluates $h \in \{ 1 5 , 2 0 , 3 0 , 4 0 \}$ for sensitivity. This is a degradationrisk proxy, not a native anomaly label. For MIMII, models are calibrated on normal training audio and evaluated on test-normal and test-anomaly files. For BDG2, a forecasting model predicts electricity readings and absolute residual magnitude provides the monitoring score. Synthetic perturbations create pseudo labels only for controlled sensitivity analysis.

## 3.2 Datasets and Preprocessing

C-MAPSS. C-MAPSS contains simulated turbofan degradation trajectories generated with NASA’s Commercial Modular Aero-Propulsion System Simulation tool [Saxena and Goebel, 2008]. Each row represents one operating cycle with three operating settings and 21 sensor measurements [Saxena and Goebel, 2008]. We use the 21 sensor channels, exclude the three operational settings from the model input, and construct length-30 windows with stride 1. A seed-controlled five-fold manifest assigns each of the 100 FD001 engines to exactly one held-out fold; every fold contains 80 fitting and 20 test engines, with zero engine or sample-ID overlap. Within each fold, 512 proxy-normal windows are selected by a seed-controlled sample-ID rule shared by all models. Per-sensor standardization is fitted only on these selected windows, and every window from the held-out engines is evaluated.

MIMII. We use the pump archive from the MIMII/DCASE machine-sound data [Purohit et al., 2019, Koizumi et al., 2020]. Audio is preprocessed once into a shared cache of fixed-size 128 × 32 log-mel tensors using 32 mel bins, FFT size 1024, and hop length 512. For each of five seeds, a manifest selects 160 pump train-normal files and 320 balanced test files across machine IDs 00, 02, 04, and 06. Normalization is fitted on the train-normal tensors only. MOMENT consumes the normalized tensors directly; classical models consume a deterministic flattening of the same tensors. All anomaly scores and metrics remain file-level.

BDG2. We use cleaned hourly electricity meter series from BDG2 [Miller et al., 2020]. We prespecify a 12-meter electricity panel selected with seed 42 from eligible meters having at least 12,000 observations, at least 90% coverage, and nonzero variance. The panel spans nine sites and five primary-use categories. Lightweight regressors use calendar features and one- and 24-hour lags. Chronos and TimesFM receive a 48-hour historical context and make zero-shot one-step forecasts. Forecasts are aligned to the common held-out timestamps available from every model.

Table 1: Protocol summary. C-MAPSS uses degradation-risk proxy labels, MIMII uses native file-level test labels, and BDG2 uses synthetic perturbation diagnostics. Results are interpreted within each protocol and are not compared across datasets.
<table><tr><td>Dataset</td><td>Domain</td><td>Task framing</td><td>Training/evaluation protocol</td><td>Primary metrics</td></tr><tr><td>C-MAPSS</td><td>Turbofan sensors</td><td>Degradation trajectories converted to a high-risk-window proxy</td><td>Five engine-disjoint folds; 512 proxy- normal fitting windows per fold; 100 engines evaluated out of fold</td><td>AUROC, AUPRC, F1, precision, recall</td></tr><tr><td>MIMII</td><td></td><td>Machine acoustics Labeled anomalous sound detection</td><td>Five paired file manifests; 160 train- normal and 320 balanced test files per seed; four machine IDs</td><td>AUROC, AUPRC, F1, precision, recall</td></tr><tr><td>BDG2</td><td>Building energy</td><td>Forecast residual monitoring and synthetic sensitivity</td><td>Fixed 12-meter panel; aligned held- out timestamps; five injection seeds sMAPE; pseudo and ten perturbation types</td><td>MAE, RMSE, AUROC, AUPRC, F1</td></tr></table>

## 3.3 Anomaly Scores and Thresholds

All evaluated monitoring methods return scores for which larger values indicate greater deviation under the corresponding protocol. Classical one-class models use the negated estimator score, reconstruction models use mean squared reconstruction error, and forecasting models use absolute residuals. Unless otherwise noted, the operating threshold is the 95th percentile of training scores:

$$
\tau = Q _ { 0 . 9 5 } \left( \{ s ( x _ { i } ) : x _ { i } \in \mathcal { D } _ { \mathrm { t r a i n } } \} \right) .
$$

This avoids selecting decision thresholds using test labels.

## 3.4 Model Families

Classical one-class baselines. We evaluate Isolation Forest [Liu et al., 2008], One-Class SVM [Schölkopf et al., 2001], Local Outlier Factor in novelty mode [Breunig et al., 2000], and PCA reconstruction [Jackson and Mudholkar, 1979]. Windowed sensor inputs are flattened before model fitting.

Lightweight deep baselines. C-MAPSS uses compact LSTM [Hochreiter and Schmidhuber, 1997] and TCN-style [Bai et al., 2018] autoencoders. Both are trained on scaled normal windows, and per-window reconstruction error is used as the anomaly score.

Energy residual baselines. We compare lag-24 naive forecasts, rolling medians, Ridge regression [Hoerl and Kennard, 1970], and LightGBM [Ke et al., 2017]. These models are evaluated using forecasting residual metrics and, separately, synthetic pseudo-anomaly diagnostics.

Foundation models. MOMENT-small [Goswami et al., 2024] is evaluated in embedding-plus-Isolation-Forest and reconstruction modes for C-MAPSS; reconstruction is also evaluated on fixed-length MIMII log-mel windows. Chronos-T5 tiny [Ansari et al., 2024] and TimesFM 2.5 200M [Das et al., 2024, Google Research, 2025] are evaluated as zeroshot one-step BDG2 forecasters. The selected TSFMs operate in diferent task modes and domains; the study compares deployment cases rather than estimating a domainindependent ranking of model families. The selection is not exhaustive and does not cover fine-tuning or anomalyspecific adapters.

## 3.5 Cost and Reproducibility Metadata

Each formal run stores protocol and train-selection checksums together with Parquet score artifacts, allowing summary metrics to be recomputed and paired by engine or file. The standardized C-MAPSS cost audit uses one RTX 4090, float32 inference, identical scaled fold-0 windows, batch sizes 1 and 16, 10 warm-up runs, and 30 measured repetitions. It reports same-process warm-call p50/p95 per-batch model-call latency, peak allocated/reserved VRAM, and serialized state-dictionary size. Preprocessing and model loading are excluded and reported as such; CPU Isolation Forest is retained as a deployment configuration but is not used for same-device algorithmic dominance.

Dataset archives are tracked by path, size, and SHA256 checksum. Experiments use YAML configurations, and paper artifacts are regenerated from run-level metrics. The main Python 3.11 environment is locked with uv.lock; TimesFM 2.5 uses a separately specified environment because of incompatible NumPy requirements.

## 3.6 Evaluation Units

C-MAPSS uses engines as the split and bootstrap unit; overlapping windows from an engine remain in the same fold. MIMII uses audio files as the evaluation unit, and all compared models consume the same file manifests and cached tensors within a seed. BDG2 uses meters as the primary aggregation unit and injection realizations nested within meters. No result is interpreted as raw transfer across sensor, acoustic, and energy inputs.

## 3.7 Uncertainty

For C-MAPSS, point estimates are engine-count-weighted means of fold metrics; paired diferences use 10,000 bootstrap resamples of held-out engines within each fold, avoiding direct ranking of raw scores across fold-specific scales.

For MIMII, we report means and t-based 95% confidenceinterval half-widths over five paired file manifests; paired file-bootstrap diagnostics resample within label-by-machine-ID strata. BDG2 first averages injection seeds within a meter and then reports meter-macro means and dispersion.

## 3.8 TSFM Evaluation and Synthetic Diagnostics

Each TSFM uses the matched domain inputs and evaluation units defined above. Chronos-T5 and TimesFM forecast each held-out BDG2 target from a fixed context.

Five injection seeds place nonoverlapping six-point events at nominal point rate 0.01 across spike, drop, flatline, gradual-drift, level-shift, intermittent-missingness, stuckat, delayed-response, seasonal-distortion, and contextualanomaly types while forecasts and contexts remain fixed. Type-specific AUROC, AUPRC, F1, precision, and recall are reported alongside aggregate diagnostics. Directly observable missingness is a separate pipeline check excluded from aggregate metrics. These pseudo labels diagnose residual sensitivity, not equipment faults, occupancy events, commissioning issues, or meter failures.

## 4 Results

## 4.1 Target-Domain Anomaly Ranking

Figure 1 and Table 2 summarize the matched-protocol results. Across 100 held-out C-MAPSS engines and 17,731 out-of-fold windows, TCN-AE obtains fold-weighted AU-ROC 0.9570, AUPRC 0.8960, and F1 0.7959. Isolation Forest is close at 0.9521/0.8866/0.7847, whereas the better-performing MOMENT configuration, reconstruction, reaches only 0.7310 AUROC and 0.3080 AUPRC.

The fold-stratified engine-cluster paired bootstrap supports the C-MAPSS gap. The paired diferences, defined as MOMENT minus TCN-AE, are −0.2260 for reconstruction AUROC and −0.5880 for reconstruction AUPRC; the corresponding embedding diferences are −0.3199 and −0.6570. The respective 95% confidence intervals from 10,000 within-fold engine resamples are [−0.2557, −0.1938], [−0.6226, −0.5260], [−0.3458, −0.2945], and [−0.6886, −0.6235].

Under the paired MIMII protocol, OCSVM has the strongest five-seed AUROC/AUPRC means, 0.6944±0.0272 and $0 . 7 2 7 7 \pm 0 . 0 2 0 9$ , compared with 0.5501 ± 0.0271 and $0 . 5 8 9 0 \pm 0 . 0 3 6 7$ for MOMENT reconstruction. Within every seed, the label-by-machine-ID-stratified paired filebootstrap 95% confidence intervals for MOMENT minus OCSVM lie below zero for both AUROC and AUPRC. All methods share files, upstream features, normalization, and labels; MOMENT consumes the normalized tensor, whereas classical estimators consume its deterministic flattened view. Thus, neither anomaly-ranking protocol shows a benefit from frozen MOMENT-small over models fitted to target-normal data.

## 4.2 Risk-Horizon Sensitivity

The C-MAPSS ranking is stable across proxy horizons of 15, 20, 30, and 40 cycles. TCN-AE AUROC ranges from 0.9347 to 0.9805 and AUPRC from 0.8725 to 0.8960; Isolation Forest remains close. MOMENT reconstruction remains substantially lower at every horizon, with AUROC ranging from 0.7004 to 0.7682 and AUPRC from 0.1910 to 0.3646. Although the proxy definition changes absolute performance, MOMENT remains behind the fitted references at every tested horizon.

## 4.3 Forecasting and Perturbation Sensitivity

On the aligned 12-meter BDG2 panel, TimesFM 2.5 has the lowest meter-macro forecast errors (MAE 9.3059, RMSE 15.7695, sMAPE 0.1033). Chronos-T5 tiny reaches 10.3225/17.0758/0.2214, while Ridge and LightGBM have similar larger relative errors. TimesFM and Chronos achieve aggregate synthetic AUROC of 0.7239 and 0.7215, compared with 0.7039 for LightGBM and 0.6853 for Ridge. Aggregate AUPRC is close and ordered diferently: 0.1582, 0.1455, 0.1639, and 0.1549, respectively. This contrast illustrates why AUROC alone can overstate practical diferences under sparse perturbations.

Perturbation-specific results delimit this evidence. Missingness is perfectly separable by construction, while spike and drop events are comparatively easy. Delayed response, flatline, stuck-at, and seasonal distortion have low AUPRC for all evaluated models. Chronos has the highest metermacro AUROC point estimates on gradual drift and level shift, closely followed by TimesFM, but thresholded precision remains low. The aggregate diagnostic therefore mixes qualitatively diferent event dificulties and cannot establish real building-fault detection.

## 4.4 Local Resource Requirements

Figure 2 compares TCN-AE and MOMENT reconstruction on the same RTX 4090 and scaled fold-0 workload. At batch size 1, per-batch p50 latency is 0.294 ms for TCN-AE and 13.368 ms for MOMENT; at batch size 16 it is 0.265 and 50.013 ms. MOMENT’s serialized state-dictionary size is 138,053 KB versus 13.8 KB for TCN-AE, and its batch-16 peak allocated VRAM is 691 MB versus 2.6 MB. These are same-process warm-call distributions on a shared host, which exhibited latency dispersion; they establish implementation-specific scale, not a stable speed multiplier or hardware-independent frontier.

## 5 Discussion

## 5.1 When Do TSFMs Pay Of?

BDG2 is the only setting in which zero-shot forecasting priors improve selected point estimates relative to the fitted residual baselines, but model preference changes across

![](images/06bd45db9020c489d6b06eb7e7a51a642703573f174e3337588da147ebfc558b.jpg)

![](images/9176830caa5d35a08c8353fc5a98b3bc68653baf1e5634faba51ef3a21c91ba3.jpg)

![](images/77b6eaecd06c462f9bce7cc538aa78e81b9f94c80530f55017af376c25ef4a22.jpg)

![](images/7bbfe19267219455e7bcc0f66a47241bfe10a397368d10346fe3da0ba1c0bd41.jpg)

![](images/0aa46dd032ae109710e8bfcb9b3d0182a549644fd3690553cfc08de4a99baaad.jpg)

![](images/1cc160c14a7ba0c9384e4d7f46b57d1909aca858cca7c6c3783f3b31f390ec99.jpg)  
Figure 1: Protocol-specific AUROC and AUPRC; values should not be compared across datasets. Blue denotes classical or fitted baselines, green compact neural models, and red TSFMs. C-MAPSS reports engine-count-weighted means from engine-disjoint folds. MIMII reports five-seed means with t-based 95% confidence-interval half-widths. BDG2 reports meter-macro means with across-meter standard deviations after averaging injection seeds within each meter; these values measure synthetic perturbation sensitivity.

AUROC, AUPRC, and thresholded metrics. Because the labels are injected, this is evidence of perturbation sensitivity rather than real-fault detection.

The comparison evaluates deployment modes: TSFMs are frozen or zero-shot, whereas classical and compact neural models fit target-normal data. It does not assess the full finetuning capacity of TSFMs. Across the two anomaly-ranking protocols, avoiding TSFM adaptation does not ofset lower ranking performance, and the audited MOMENT mode adds local implementation cost. Model selection should therefore favor the lowest-cost model that satisfies prespecified protocol-specific performance requirements rather than treat pretraining as evidence of fitness by itself.

## 5.2 Threats to Validity

Task heterogeneity. C-MAPSS uses degradation-risk proxy labels, MIMII uses labeled acoustic anomalies, and BDG2 uses residual monitoring with synthetic sensitivity diagnostics. The study is a protocol-aware audit, not a unified benchmark.

C-MAPSS label and split semantics. High-risk labels are derived from each engine’s observed terminal cycle and are not native anomaly annotations. Engine-disjoint folds prevent engine and exact-window leakage, but stride-1 windows remain highly dependent within an engine. Withinfold engine-cluster bootstrap respects this grouping without comparing raw score ranks across fold-specific scales, and the four-horizon analysis shows that absolute performance remains proxy-dependent.

Synthetic energy anomalies. BDG2 pseudo metrics measure response to ten synthetic event types on 12 selected meters and five injection seeds. Missingness is directly observable, while several contextual or temporal types remain dificult. The selected panel broadens coverage beyond a single meter but is not a random sample of all buildings. It does not measure real equipment faults, occupancy anomalies, commissioning problems, or meter failures.

TSFM coverage. We evaluate MOMENT-small, Chronos-T5 tiny, and TimesFM 2.5 200M. Larger variants, fine-tuning, anomaly-specific adapters, and alternative context construction could change the results. Our conclusion is conditional on the tested protocols.

Timing boundaries. The standardized C-MAPSS audit fixes workload, GPU, precision, batch size, warm-up, and repetition count, but excludes preprocessing and model loading. The shared host shows nontrivial latency dispersion, and the CPU Isolation Forest configuration is not directly comparable to GPU implementations.

Table 2: Selected anomaly-ranking and synthetic-diagnostic results. MIMII values are five-seed means. BDG2 values first average five injection seeds within each meter and then macro-average the 12 meters; they do not represent real-fault detection.
<table><tr><td>Protocol</td><td>Model</td><td>Scope</td><td>AUROC</td><td>AUPRC</td><td>F1</td><td>Precision</td><td>Recall</td></tr><tr><td>C-MAPSS</td><td>TCN-AE</td><td>100 engines OOF</td><td>0.9570</td><td>0.8960</td><td>0.7959</td><td>0.7751</td><td>0.8239</td></tr><tr><td>C-MAPSS</td><td>Isolation Forest</td><td>100 engines OOF</td><td>0.9521</td><td>0.8866</td><td>0.7847</td><td>0.7659</td><td>0.8129</td></tr><tr><td>C-MAPSS</td><td>MOMENT Recon.</td><td>100 engines OOF</td><td>0.7310</td><td>0.3080</td><td>0.1190</td><td>0.2592</td><td>0.0790</td></tr><tr><td>MIMII</td><td>OCSVM</td><td>5 seeds, 320 files/seed</td><td>0.6944</td><td>0.7277</td><td>0.6109</td><td>0.6334</td><td>0.5912</td></tr><tr><td>MIMII</td><td>MOMENT Recon.</td><td>5 seeds, 320 files/seed</td><td>0.5501</td><td>0.5890</td><td>0.2232</td><td>0.7489</td><td>0.1325</td></tr><tr><td>BDG2 diagnostic</td><td>LightGBM</td><td>12 meters, 5 seeds</td><td>0.7039</td><td>0.1639</td><td>0.1050</td><td>0.0653</td><td>0.3562</td></tr><tr><td>BDG2 diagnostic</td><td>Ridge</td><td>12 meters, 5 seeds</td><td>0.6853</td><td>0.1549</td><td>0.1070</td><td>0.0738</td><td>0.3012</td></tr><tr><td>BDG2 diagnostic</td><td>Chronos-T5 tiny</td><td>12 meters, 5 seeds</td><td>0.7215</td><td>0.1455</td><td>0.0699</td><td>0.0392</td><td>0.6358</td></tr><tr><td>BDG2 diagnostic</td><td>TimesFM 2.5</td><td>12 meters, 5 seeds</td><td>0.7239</td><td>0.1582</td><td>0.0732</td><td>0.0412</td><td>0.6293</td></tr></table>

![](images/8653a4b637abf357391ad2f1ee6f1f79c33f545c99ffe5205cb91e8e0ff1ee57.jpg)

![](images/f3dd4e4703495a8080567d2567d2698197a43055ddb803dfb089f28e41c7c7c3.jpg)  
Figure 2: Standardized local resource audit on one RTX 4090. The left panel reports same-process warm-call per-batch p50 latency and p50–p95 ranges over 30 repetitions after 10 warm-ups. The right panel reports peak allocated VRAM and serialized state-dictionary size. Preprocessing and model loading are excluded.

Uncertainty. C-MAPSS uses fold-stratified enginecluster bootstrap and MIMII uses paired five-seed manifests with label-by-machine-ID-stratified file-bootstrap artifacts, but deep models are not repeatedly initialized within every fold. BDG2 reports meter dispersion rather than a population-level confidence interval, and injection seeds are nested within the fixed panel.

## 6 Conclusion

Across engine-disjoint C-MAPSS and paired MIMII protocols, fitted lightweight models remain stronger than the tested frozen MOMENT-small modes, and this ordering persists across four C-MAPSS risk horizons. Forecasting TSFMs show limited advantages on BDG2 in forecast error and synthetic AUROC, but not in AUPRC or thresholded F1; this is not evidence of real-fault detection. Foundation models should therefore be evaluated as task-dependent deployment choices rather than default replacements for fitted models: compact fitted models are strong references when target-normal data and resource constraints matter, while zero-shot forecasters merit evaluation when residual forecasting is itself valuable. Stronger claims require real energy-event labels and adapted or anomaly-specific TSFM evaluation.

In practice, deployment decisions should retain targetnormal baselines and report calibration boundaries alongside ranking metrics. Synthetic diagnostics can screen forecasters but cannot certify detectors without real event labels and end-to-end resource accounting.

## References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=gerNCVqqtR.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018.

Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. LOF: Identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, pages 93–104, 2000. doi: 10.1145/342009.335388.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10148–10167. PMLR, 2024. URL https: //proceedings.mlr.press/v235/das24c.html.

Google Research. TimesFM: timesfm-2.5-200m-pytorch model card. Hugging Face model card, 2025. URL https: //huggingface.co/google/timesfm-2.5-200m-pytorch. Updated 2025-10-02; accessed 2026-08-23.

Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. MOMENT: A family of open time-series foundation models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 16115–16152. PMLR, 2024. URL https://proceedings. mlr.press/v235/goswami24a.html.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

Arthur E. Hoerl and Robert W. Kennard. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67, 1970. doi: 10.1080/00401706.1970. 10488634. URL https://doi.org/10.1080/00401706. 1970.10488634.

J. Edward Jackson and Govind S. Mudholkar. Control procedures for residuals associated with principal component analysis. Technometrics, 21(3):341–349, 1979. doi: 10.1080/00401706.1979.10489779. URL https://doi.org/ 10.1080/00401706.1979.10489779.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly eficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, volume 30, pages 3146–3154, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 6449f44a102fde848669bdd9eb6b76fa-Abstract.html.

Uzair Khan, Luigi Capogrosso, Francesco Biondani, Michele Magno, Franco Fummi, Francesco Setti, and Marco Cristani. ChronosAD: Leveraging time series foundation models for

accurate anomaly detection, 2026. Preprint; accepted at IEEE INDIN 2026.

Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada. Description and discussion on DCASE2020 challenge task2: Unsupervised anomalous sound detection for machine condition monitoring. In Proceedings of the 5th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2020), pages 81–85, 2020. URL https://dcase.community/documents/workshop2020/ proceedings/DCASE2020Workshop\_Koizumi\_3.pdf.

Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining, pages 413–422, 2008. doi: 10.1109/ICDM.2008. 17.

Clayton Miller, Anjukan Kathirgamanathan, Bianca Picchetti, Pandarasamy Arjunan, June Young Park, Zoltan Nagy, Paul Raftery, Brodie W. Hobson, Zixiao Shi, and Forrest Meggers. The building data genome project 2, energy meter data from the ASHRAE great energy predictor III competition. Scientific Data, 7(1):368, 2020. doi: 10.1038/s41597-020-00712-x. URL https://doi.org/10.1038/s41597-020-00712-x.

Harsh Purohit, Ryo Tanabe, Takeshi Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi. MIMII dataset: Sound dataset for malfunctioning industrial machine investigation and inspection. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), pages 209–213, 2019. URL https://dcase.community/documents/workshop2019/ proceedings/DCASE2019Workshop\_Purohit\_21.pdf.

Abhinav Saxena and Kai Goebel. Turbofan engine degradation simulation data set. NASA Prognostics Data Repository, NASA Ames Research Center, 2008. URL https: //www.nasa.gov/intelligent-systems-division/ discovery-and-systems-health/pcoe/ pcoe-data-set-repository/.

Bernhard Schölkopf, John C. Platt, John Shawe-Taylor, Alex J. Smola, and Robert C. Williamson. Estimating the support of a high-dimensional distribution. Neural Computation, 13 (7):1443–1471, 2001. doi: 10.1162/089976601750264965.

Peng Yan, Ahmed Abdulkadir, Paul-Philipp Luley, Matthias Rosenthal, Gerrit A. Schatte, Benjamin F. Grewe, and Thilo Stadelmann. A comprehensive survey of deep transfer learn ing for anomaly detection in industrial time series: Methods, applications, and directions. IEEE Access, 12:3768– 3789, 2024. doi: 10.1109/ACCESS.2023.3349132. URL https://doi.org/10.1109/ACCESS.2023.3349132.

Zahra Zamanzadeh Darban, Geofrey I. Webb, Shirui Pan, Charu Aggarwal, and Mahsa Salehi. Deep learning for time series anomaly detection: A survey. ACM Computing Surveys, 57(1):1–42, 2025. doi: 10.1145/3691338. URL https://doi.org/10.1145/3691338. Article 15.

Muhy Eddin Za’ter and Bri-Mathias Hodge. Empirical assessment of time-series foundation models For power system forecasting applications, 2026. URL https://arxiv.org/ abs/2604.22077.