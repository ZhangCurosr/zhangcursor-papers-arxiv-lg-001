# An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates

Xu Wang<sup>1,2</sup>, Shuhan Yi<sup>2</sup>, and Zhen Cheng<sup>2\*</sup>

<sup>1</sup> School of Engineering, The Hong Kong University of Science and Technology, Hong Kong 999077, China

<sup>2</sup> School of Biomedical Engineering, Tsinghua University, Beijing 100084, China

\*Corresponding author:

Zhen Cheng, School of Biomedical Engineering, Tsinghua Medicine, Tsinghua University, Beijing 100084, P. R. China. Email: zcheng@mail.tsinghua.edu.cn

## Abstract

Wrist photoplethysmography (PPG) heart rate estimates are vulnerable to motion artifacts, and corrections that reduce mean error can worsen individual windows. We propose an agent architecture with tools and memory around a fixed heart rate estimator. Pulse, motion, and limited historical evidence generate candidate estimates, whose predicted benefits and harms guide selection. An independent risk gate determines whether to retain, correct, or reject each output. We trained the architecture on PPG-DaLiA and tested it on Wrist Exercise, GalaxyPPG, and WildPPG with frozen weights. Across these four public datasets, mean absolute error (MAE), macro-averaged over ten estimators, decreased by up to 44.5%, with the largest reduction on GalaxyPPG. Beneficial and harmful corrections decreased and increased absolute error by more than 1 bpm, respectively. Their ten-estimator macro-averaged proportions ranged from 16.06% to 36.64% and from 1.68% to 6.77%, respectively, across the four datasets. At 60% and 80% target coverage, numerical correction on the same accepted windows contributed more to total error reduction than window selection. We implemented real-time wearable acquisition, Bluetooth reception, and smartphone deployment of a selected estimator and the framework. In backend analysis of four prototype recordings from one participant, ten-estimator macro-averaged MAE decreased from 9.09 to 4.45 bpm without additional rejection. Under the evalu ated conditions, the architecture improved average accuracy while quantifying correction benefits and harms, sup porting wearable heart rate analysis that considers error, correction risk, and coverage.

![](images/10ca7df85c3f94be1eccad69ef87ea868e6f439e5d6620cf432978c67c6f62d3.jpg)  
Abstract figure

Index Terms—Heart rate estimation, photoplethysmography, selective correction, temporal memory, wearable sensors.

## I. INTRODUCTION

Wearable heart rate monitoring, predominantly enabled by photoplethysmography (PPG), has become central to wearable health sensing, enabling continuous characterization of daily activity and physiological status in real-world settings. PPG measures rhythmic changes in peripheral blood volume using a compact, noninvasive optical sensor [1], [2]. Motion can introduce interference that overlaps with pulse components and reduces estimation accuracy [1], [3]. This interference varies with activity and wearing conditions, and substantial errors can remain after artifact suppression. Continuous monitoring therefore requires both accurate estimates and a way to identify unreliable outputs before they enter subsequent analyses.

Signal processing and data-driven approaches have been developed to address motion interference. TROIKA [4], JOSS [5], SpaMA [6], and WFPV [7] use signal decomposition, joint spectral reconstruction, analysis of motionrelated frequencies, and filtering or tracking to improve heart rate estimation under interference. Deep PPG [3] and TimePPG [8] learn relationships between sensor signals and heart rate using time–frequency representations and temporal convolution. Output reliability has also received attention. Signal quality indices can identify contaminated segments; BeliefPPG models heart rate dynamics and probabilistic inference to provide estimates and their uncertainty [9], whereas KID-PPG combines knowledge-guided inference with identification of unreliable outputs [10]. Beyond improving estimates and detecting unreliable results, post-calibration methods use device outputs, activity, and historical information to adjust existing heart rate estimates [11]. Such additional information can support correction, but it also raises the question of which estimates should be changed and which should be retained.

These methods do not fully resolve when an existing estimate should be changed. Mean error measures overal accuracy but does not establish whether a correction benefits an individual window. BeliefPPG and some post-cali bration methods assess window-level reliability or adjust unreliable outputs [9], [11]. However, explicit assessment of correction gains and harm to originally accurate estimates remains limited. Average improvements can concea deterioration in individual windows. If a system also rejects outputs, lower error may partly reflect the exclusion of difficult windows [12], [13]. Evaluation must therefore distinguish correction benefits, correction harms, and output coverage.

Here, we propose a selective correction framework built around a fixed heart rate estimator. Its organization links observations, internal states, and action selection, following an agent architecture [14]. Signal evidence and recent history inform whether to retain, correct, or reject an estimate without changing the underlying estimator. Decisions account for predicted gains and harm relative to the original estimate, including deterioration of initially accurate outputs. We evaluate numerical correction separately from window rejection to identify the source of error reductions.

This study makes three contributions. First, we introduce a framework with signal analysis tools and memory tha evaluates candidate gains and harms before replacing a fixed estimator’s output. A shared base-error prediction and geometric constraints give candidates a consistent comparison against the original estimate. The original estimate remains available, and candidate selection precedes a separate output risk gate. Second, we evaluate ten prespecified estimators on four public datasets and prototype recordings, separating correction gains from selection effects and reporting correction harm and coverage. Third, we implement real-time wearable acquisition and smartphone de ployment of a selected estimator and the framework. The smartphone receives and analyzes the signals, while the computer backend supports comparisons across methods.

## II. MATERIALS AND METHODS

## A. Task Definition and Overall Architecture

![](images/62263becfc981d902cea39c9e6434b9feff9df197b32586f3d424270c620c3ab.jpg)

Fig. 1. Agent architecture with tools and memory for selective heart rate correction. Information flow among the fixed core estimator and M1–M4, with retain, correct, and reject outputs. M1/M2 extract pulse, harmonic, and motion evidence from PPG and triaxial ACC to form a 186 × 30 evidence tensor. M3 integrates evidence from five temporal positions (the current position and four preceding positions, spaced 4 s apart) into memory and combines it with current evidence and the core estimator prior to generate a heart rate posterior. M4 evaluates gain, harm, and error from 106-dimensional features for nine candidates, selects an estimate, and then checks output risk. Arrows indicate information flow. PPG: photoplethysmography; ACC: acceleration; HR: heart rate.

As illustrated in Fig. 1, the framework comprises four modules, including M1 and M2 which extract pulse and motion evidence, respectively. M3 combines this evidence with recent history and the core estimator prior to generate heart rate candidates. M4 predicts each candidate’s gain and harm relative to the original estimate, selects an estimate, and applies a separate risk gate. The architecture is shared across core estimators, but M3 and M4 are fitted inde pendently for each estimator. The evaluated framework therefore uses estimator-specific auxiliary weights rather than a single set of weights shared by all ten estimators.

The framework corrects the window-level outputs of a fixed heart rate estimator. For analysis window �, the raw sensor input is a synchronized � × 4 matrix comprising � samples from one PPG channel and three acceleration (ACC) channels. In addition, the framework receives three estimator-related quantities: the original heart rate esti mate $h _ { 0 , t }$ , an adapted heart rate prior $p _ { t } ,$ and a baseline risk score ${ \boldsymbol { r } } _ { 0 , t }$ . These are window-level quantities rather than additional sensor channels. The baseline risk score is an estimator-specific uncertainty indicator associated with the original estimate. For BeliefPPG and KID-PPG, it is the standard deviation of the predictive distribution; for the other estimators, it is an adapter-derived entropy measure of spectral evidence. Larger scores indicate greater uncertainty. The score is computed without reference heart rate at inference and is not the observed absolute error. Estimator specific definitions are provided in Supplementary Methods S6. The output is a retained estimate, a corrected estimate, or a missing-value indicator. Candidate heart rates are denoted by $a _ { t , j }$ , with $j = 0$ corresponding to the original estimate, i.e., $a _ { t , 0 } = h _ { 0 , t }$ . Given the reference heart rate $y _ { t }$ , candidate absolute error, net gain relative to the original estimate, and added error are defined as follows:

$$
\begin{array} { c } { { e _ { t , j } = | a _ { t , j } - y _ { t } | , } } \\ { { { } } } \\ { { G _ { t , j } ^ { * } = e _ { t , 0 } - e _ { t , j } , } } \\ { { { } } } \\ { { H _ { t , j } ^ { * } = m a x ( - G _ { t , j } ^ { * } , 0 ) . } } \end{array}\tag{1}
$$

Here, $e _ { t , j }$ is the candidate absolute error, $G _ { t , j } ^ { * }$ is the net gain, and $H _ { t , j } ^ { * }$ is the added error, all measured in beats per minute (bpm). Positive net gain indicates error reduction, whereas negative net gain indicates error increase. The original estimate always remains in the candidate set and has zero relative net gain and added error. Reference heart rate is used only for source-domain supervision, calibration, and offline evaluation; it is not used for candidate generation or action selection at test time.

M1 and M2 are deterministic signal processing modules; M3 and M4 are learned modules. The fixed workflow links observation analysis, internal-state updates, and action selection [14]. It does not use a language model, dynamically schedule tools, or combine votes from multiple core estimators. Fig. 1 summarizes evidence extraction, memory fusion, candidate selection, and output risk checking. Supplementary Methods S18 specifies the module interfaces.

## B. Pulse and Motion Evidence Extraction

M1 and M2 transform raw signals into evidence on a common heart rate grid to compare pulse and motion-related components. Analysis windows are 8 s long and updated every 4 s. At each update, M1 and M2 use both the current 8 s signal segment and a continuous 24 s signal segment ending at the same time, with a common feature-computation sampling rate of 64 Hz. The 24 s segment spans five consecutive, overlapping 8 s windows with a 4 s stride $( 8 + 4 \times$ $4 = 2 4 { \mathrm { s } } )$ . It provides a longer raw-signal context for evidence extraction and is distinct from the five-position feature memory used by M3. The heart rate grid spans 35–220 bpm in 1 bpm increments, giving 186 positions. Each position contains 30 features, forming the $1 8 6 \times 3 0$ evidence matrix $X _ { t }$ . M1 extracts PPG power, autocorrelation, and spectral features at the fundamental, half, and double frequencies. M2 characterizes motion coupling using magnitude squared coherence between PPG and triaxial ACC and computes residual power that downweights motion-related frequencies.

Pulse and motion frequencies can overlap, so the representation retains the original spectrum, coherence, and residual power. Its 30 channels contain spectra and autocorrelation at two time scales, together with interaction features at the fundamental, half, and double frequencies. Supplementary Methods S11 and S12 define spectral estimation, coherence, and residual power; Table S1 lists the channel order.

## C. Limited-History Fusion and Heart Rate Candidate Generation

M3 uses recent evidence to constrain the heart rate distribution of the current window. Its input comprises evidence matrices from five temporal positions: the current position and four preceding positions, spaced 4 s apart. Each feature channel c has a learnable decay coefficient $\alpha _ { c } ,$ obtained by applying a sigmoid to parameter $d _ { c }$ and shared across all 186 heart rate grid positions. The state is initialized using the evidence matrix at the earliest position and then updated chronologically:

$$
S _ { t } ^ { ( i ) } = \alpha \odot S _ { t } ^ { ( i - 1 ) } + ( 1 - \alpha ) \odot X _ { t - 4 + i } , i = 1 , \dots , 4 .\tag{2}
$$

In (2), ⊙ denotes elementwise multiplication, and i indexes updates within the five-position sequence. The final historical state $S _ { t }$ is the result of the fourth update. The state is recomputed over the most recent five positions at every inference step; missing history at the beginning of a recording is padded with the earliest available features. The five endpoints span 16 s, and each position uses a 24 s evidence window, so the earliest signal used precedes the current endpoint by 40 s. No future data are accessed. Initialization and the expanded recurrence are given in Supplementary Methods S13.2.

At heart rate grid point g, current evidence, the historical state, and their difference are concatenated into a 90- dimensional vector. A shared two-layer perceptron with 24 hidden units and sigmoid linear unit (SiLU) activation produces an energy correction $E _ { t } ( \mathrm { g } )$ . This correction is bounded between −6 and 6 by a hyperbolic tangent, added to the temperature-scaled log prior, and normalized over the heart rate grid �:

$$
\begin{array} { r } { q _ { _ t } ( g ) = \frac { \exp \{ \log p _ { _ t } ( g ) / T + E _ { t } ( g ) \} } { \sum _ { u \in \mathcal { G } } \exp \{ \log p _ { _ t } ( u ) / T + E _ { t } ( u ) \} } , T = 0 . 5 + 3 \sigma ( \theta ) . } \end{array}\tag{3}
$$

Here, T is a learnable temperature parameterized by θ and bounded between 0.5 and 3.5. Prior probabilities are floored at $1 0 ^ { - 1 0 }$ before taking logarithms. $p _ { t }$ is adapted from the core estimator output distribution or corresponding evidence. $q _ { t }$ is the fused heart rate distribution, referred to below as the posterior; this terminology does not imply that its probabilities have been calibrated in the target domain. Full expressions for feature concatenation and the energy network are given in Supplementary Methods S13.3.

Nine candidates are derived from the posterior: the original estimate, mean, mode, five percentiles, and a midpoint candidate. The percentiles are 10th, 25th, 50th, 75th, and 90th. The midpoint between the original estimate and posterior median offers a smaller correction. Candidate order is fixed, and duplicates are allowed. The original estimate occupies the first position and retains its continuous value, even outside the heart rate grid. Supplementary Methods S13.4 defines the candidate set.

## D. Modeling Candidate Gains and Harms

M4 evaluates candidates generated by M3 by learning how error changes when the original estimate is replaced. It first computes the expected candidate absolute error $e _ { t , j } ^ { q }$ and expected added error $h _ { t , j } ^ { q }$ under posterior $q _ { t }$ . These quantities are posterior-weighted averages, not the actual errors against reference heart rate. Their expressions are provided in Supplementary Methods S14.1.

To learn the discrepancy between posterior expectations and actual errors, M4 constructs 106 features for each candidate: 16 candidate and distribution statistics, plus 30-dimensional current evidence, historical evidence, and current-evidence differences relative to the original estimate at the candidate position. Evidence is sampled at the nearest heart rate grid position, while the candidate heart rate itself is unchanged. Features are standardized using source-training statistics and clipped to [−5, 5]. A shared linear branch produces three allocation coefficients for each candidate; a second branch reads only the original-estimate features to produce a nonnegative base error $b _ { t }$ shared within the window. Feature definitions, standardization, and network expressions are given in Supplementary Methods S14.2–S14.4.

The three allocation coefficients $\pi _ { t , j }$ partition the magnitude of the candidate change relative to the original estimate; they are not action probabilities for retain, correct, and reject. Joint decoding with the base error yields a beneficial component $\widehat { B } _ { t , j }$ , added error $\widehat { H } _ { t , j } .$ , net gain $\widehat { G } _ { t , j }$ , and candidate absolute error $\widehat { E } _ { t , j }$ . Net gain equals the beneficial component minus added error, and candidate absolute error equals base error minus net gain. The complete decoding expressions are given in (S22)–(S25) of Supplementary Methods S14.4.

The shared base error gives all candidates the same comparison reference within a window. Decoding keeps predicted absolute error nonnegative and limits predicted net gain to the magnitude of the candidate change. It also enforces the triangle inequality between the original estimate, candidate estimate, and reference. These constraints make the predictions geometrically consistent but do not guarantee actual error bounds. Supplementary Methods S16 and S17 provide the training objectives and proofs.

## E. Candidate Selection and Risk Control

Candidate selection precedes output rejection. A replacement candidate must have predicted net gain of at least 1 bpm and predicted added error of at most 2 bpm. Its posterior expected absolute error must also be lower than that of the original estimate. Eligible candidates receive utility equal to net gain minus twice the added error; the original estimate has zero utility:

$$
\begin{array} { r l } & { \quad U _ { t , 0 } = 0 , } \\ & { \quad U _ { t , j } = \left\{ \begin{array} { l l } { \widehat { G } _ { t , j } - 2 \widehat { H } _ { t , j } , \quad \widehat { G } _ { t , j } \geq 1 , \widehat { H } _ { t , j } \leq 2 , e _ { t , 0 } ^ { q } > e _ { t , j } ^ { q } , } \\ { - \infty , \quad } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \\ & { \quad j ^ { * } = a r g m a x _ { j } U _ { t , j } , \widehat { h } _ { t } = a _ { t , j ^ { * } } . } \end{array}\tag{4}
$$

(5)

$j ^ { * }$ is the index of the candidate with maximum utility, and $\widehat { h } _ { t }$ is the selected estimate before risk filtering. A tie at zero utility favors the original estimate, which occupies the first candidate position. The proposed method imposes no 20 bpm cap on the magnitude of correction; its utility can also be written as $\widehat { B } _ { t , j } - 3 \widehat { H } _ { t , j }$ . The 1 bpm and 2 bpm thresholds apply to predicted quantities and must not be interpreted as guarantees of actual improvement or harm.

The risk gate then determines whether the selected estimate can be output. Risk score $r _ { t }$ is either the baseline risk ${ \boldsymbol { r } } _ { 0 , t }$ from the core estimator or its adapter (Supplementary Methods S6), or the absolute error $\widehat { E } _ { t , j \ast }$ predicted by M4 for the selected candidate. Both the risk type and threshold τ are determined exclusively from source-domain cali bration data. A window is accepted only if the selected estimate and risk score are finite and $r _ { t } \leq \tau \colon$

$$
A _ { t } = \mathbf { 1 } ( \widehat { h } _ { t } \mathrm { f i n i t e } ) \mathbf { 1 } ( r _ { t } \mathrm { f i n i t e } ) \mathbf { 1 } ( r _ { t } \leq \tau ) .\tag{6}
$$

Here, $A _ { t }$ is the acceptance indicator. An accepted window outputs the selected estimate; a rejected window is marked as missing. The complete output expression is given in (S41) of Supplementary Methods S19. Retention and

correction describe estimate selection, whereas rejection describes the subsequent output status, so these operations should be evaluated separately. M3 and M4 contain 2,240 and 428 learnable parameters, respectively, totaling 2,668 auxiliary parameters per core estimator.

## F. Data Splits, Model Training, and Source-Domain Calibration

We used PPG-DaLiA to train and calibrate the auxiliary models and evaluated transfer to Wrist Exercise, Galaxy PPG, and WildPPG [3], [15]–[17]. These datasets contained 15, 8, 20, and 14 participants with eligible reference measurements, respectively. Table I lists the recording and window counts. Activity recordings were pooled within each participant, and windows without eligible reference measurements were excluded from error evaluation. Al datasets used 8 s windows and a 4 s stride. Supplementary Methods S1 specifies reference eligibility.

DATASETS AND ANALYSIS SCOPE
<table><tr><td>Dataset</td><td>Participants included</td><td>Participants with eligible reference</td><td>Recordings</td><td>Reference-eligible windows</td></tr><tr><td>PPG-DaLiA</td><td>15</td><td>15</td><td>15</td><td>32308</td></tr><tr><td>Wrist Exercise</td><td>8</td><td>8</td><td>19</td><td>1671</td></tr><tr><td>GalaxyPPG</td><td>23</td><td>20</td><td>23</td><td>16239</td></tr><tr><td>WildPPG</td><td>16</td><td>14</td><td>16</td><td>166132</td></tr></table>

Source-domain participants were divided into disjoint auxiliary-training, calibration, and evaluation groups. M3 first generated out-of-fold posteriors within the training group. M4 was fitted using these posteriors and historica features from the corresponding M3 models. The final M3 was then fitted on the full auxiliary-training group for calibration and evaluation. This procedure prevents M4 from training on M3 predictions fitted to the same participants. Participant assignments, folds, and random seeds are detailed in Supplementary Methods S1 and S5.

M3 represents reference heart rate as a Gaussian soft target with a standard deviation of 3 bpm. Its loss combines cross-entropy and cumulative-distribution discrepancy. M4 uses the true net gain, added error, and candidate absolute error in (1) as supervision. It minimizes SmoothL1 loss with an additional penalty for underestimating added error. The stages are trained sequentially; M4 gradients do not propagate to M3 or the core estimator. Both use participant balanced sampling and AdamW. M3 uses a learning rate of 0.002, weight decay of 0.03, and batch size of 96. The corresponding M4 settings are 0.003, 0.1, and 256. Each stage runs for 2,000 steps, using the final weights without checkpoint selection on the evaluation set. Supplementary Methods S5 and S16 specify loss weights, initialization, and other settings.

Source-domain calibration was performed separately from training. We selected either original risk or learned error according to macro-averaged mean absolute error (MAE) at 80% coverage in the calibration participants; ties favored original risk. This comparison used uncapped correction without the posterior-consistency condition. The selected risk type was then applied to the corresponding decision configurations. Two thresholds were calibrated separately: one for 80% source coverage and one for source MAE no greater than 5 bpm. The error-based rule additionally required at least 10% coverage for each calibration participant. Thresholds remained fixed during external evaluation. No feasible output was assumed when calibration constraints could not be met. Supplementary Methods S6 describes the search rules.

The ten core estimators were Deep PPG, KID-PPG, TimePPG, BeliefPPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV [3]–[10]. Supplementary Methods S6 describes their implementations. Core estimators remained fixed during auxiliary training, with M3 and M4 fitted independently for each estimator. External evaluation froze auxiliary weights, standardization statistics, decision rules, and source thresholds. Target references were not used to select source models or refit auxiliary modules. Each external participant was evaluated with each source configuration, and metrics were averaged without fusing predictions. Supplementary Methods S1 and S5 report the folds, seeds, and training procedure. KID-PPG retained its per-recording adaptation without heart rate labels [10]. Training-participant provenance was incomplete for some pretrained weights, and all four datasets had been used in earlier development. The comparisons therefore assess additional correction gains for the specified implementations, rather than reproduce a ranking of the original methods. Original and corrected errors are compared within each implementation under the same evaluation conditions; headline accuracies from the original publications are not directly comparable across different datasets, splits, and processing protocols.

## G. Evaluation Metrics and Statistical Analysis

Evaluation considers estimation error, correction harm, and output coverage. Let ℛ denote all windows with eligible reference measurements, � the subset accepted with finite outputs, and �₀ the subset of � with finite original estimates. The set �₀ supports paired comparisons before and after correction within the same windows. Mean ab solute error (MAE), measured in bpm, is the mean absolute difference between estimates and reference heart rates in accepted windows. Coverage is the fraction of all reference-eligible windows that are accepted.

Paired accepted windows are classified by the change in absolute error relative to the original estimate. A decrease exceeding 1 bpm is a beneficial correction; an increase exceeding 1 bpm is a harmful correction. Changes within ±1 bpm form the middle category. The three proportions share the same denominator and sum to 100%. The middle category includes unchanged estimates and small error changes, so it differs from the retention-action rate. The severe added-error proportion counts accepted windows with an error increase exceeding 10 bpm, divided by all referenceeligible windows. MAE and paired correction proportions are undefined when no valid outputs remain. Supplemen tary Methods S8 and S19 give the definitions. Comparators include original estimates, rejection-only strategies, can didate-selection variants, and ablations of harmonic evidence, motion evidence, or memory.

Both rejection-only comparators preserve the original heart rate values. Ridge rejection uses a ridge regression model trained on source data to predict each window’s original estimation error. It ranks test windows by predicted error and retains those with the lowest scores. This comparator measures what a simple error predictor can achieve through selection alone. In risk-ranked comparisons, same-risk rejection uses exactly the framework’s risk scores and accepted windows but outputs the original heart rates. Its MAE difference from the full framework therefore isolates numerical correction on the same windows. Neither comparator uses test-time reference errors to select windows. For thresholds targeting source MAE of 5 bpm, each strategy is calibrated independently, so accepted sets can differ. Supplementary Methods S6 describes their features, training, and threshold rules.

Risk–coverage curves are evaluated from 5% to 100% target coverage in steps of 5 percentage points. We selected 80% target coverage as a common operating point to retain most eligible outputs while allowing rejection of a limited fraction of high-risk windows. This choice represents a practical trade-off between output availability and error reduction, rather than an established standard or an optimal coverage level. At the prespecified 80% operating point, error changes are decomposed into numerical correction and window selection. Let � be the accepted set obtained by ranking the proposed method risk scores and ℬ the accepted set obtained using the original estimator risk scores. Then

$$
\Delta _ { \mathrm { t o t a l } } = \Delta _ { \mathrm { c o r r e c t i o n } } + \Delta _ { \mathrm { s e l e c t i o n } } .\tag{7}
$$

The total change is $\mathrm { M A E ( \hat { h } , \mathcal { A } ) - M A E ( h o , \mathcal { B } ) }$ . The correction term is $\mathrm { M A E ( \hat { h } , \mathcal { A } ) } - \mathrm { M A E ( h o , \mathcal { A } ) }$ , comparing estimates on a fixed accepted set. The selection term is $\mathbf { M A E ( h o , \mathcal { A } ) } - \mathbf { M A E ( h o , \mathcal { B } ) }$ , comparing window sets while keeping original estimates fixed. Fig. 4 reports the negatives of these terms as MAE reductions. This decomposition is a descriptive algebraic comparison. Target-coverage evaluation ranks risk scores within target recordings and retains at most the specified fraction of eligible windows without using reference heart rate values to select estimates. It does not imply that a fixed source-domain threshold maintains the same coverage in the target domain.

Participants were the statistical units for public-dataset evaluation. Source-configuration metrics were first averaged within each participant and estimator. Cross-estimator macro-averages then weighted the ten implementations equally, followed by equal weighting of participants. We calculated 95% percentile confidence intervals from 10,000 participant-bootstrap resamples. Paired effects were calculated before resampling; Supplementary Methods S8 specifies the random seed and aggregation details. These intervals are conditional on the fitted models and exclude retraining variability. Each ablation retrained the affected auxiliary modules on source data and froze their weights for external evaluation. Table S6 reports MAE changes and confidence intervals.

## H. Wearable Acquisition, Smartphone Deployment, and Prototype Evaluation

The wearable system uses a Seeed Studio XIAO nRF52840 Sense controller, a MAX30102 optical sensor, and the Android application PulseLink. It acquires red and infrared PPG and triaxial acceleration, then transmits signals and frame counts over Bluetooth Low Energy (BLE). A selected core model and the framework run on the smartphone to analyze received signals, display heart rate, and save recordings. The computer backend compares methods and calculates paired errors, correction harm, and risk–coverage curves. Fig. 5(a)–(c) show the measurement setup, hardware, and smartphone workflow. Supplementary Methods S20.3 specifies the smartphone and backend configurations. The quantitative results reported here were calculated from saved recordings.

One participant wore the wrist device and an independent electrocardiography (ECG) sensor on the upper sternum during static, walking, running, and recovery-rest tests. ECG provided the heart rate reference. Wrist signals were sent to the smartphone for real-time analysis and recording, followed by backend comparison of original and corrected estimates. Supplementary Methods S20 describes timing alignment, preprocessing, and evaluation settings.

## III. RESULTS

## A. Examples of Retention, Correction, and Rejection

Fig. 2 illustrates retention, correction, and rejection using GalaxyPPG wrist signals and synchronized ECG, with Deep PPG as the fixed core estimator. All examples use the configuration specified in Supplementary Methods S9. In the retained window, the original and final estimates were both 77.35 bpm, versus a reference of 75.73 bpm. In the corrected window, the estimate changed from 76.53 to 91.00 bpm, closer to the 88.64 bpm reference. Absolute error decreased from 12.10 to 2.36 bpm. In the rejected window, the estimate before risk filtering was 76.81 bpm, versus a reference of 86.19 bpm; the framework withheld the output. Corrections could also cause harm: Table S8 shows an example in which absolute error increased from 1.51 to 4.00 bpm.

The three illustrated windows start at 1412 s for P21, 420 s for P15, and 2508 s for P12. Each column aligns raw PPG, ECG, and window-end heart rates from 8 s before to 16 s after the central window start. Cases were classified retrospectively using reference heart rate; selection rules are specified in Supplementary Methods S9. They illustrate output behavior and cannot be used to estimate the overall frequencies of the three behaviors.

![](images/334adf641c483cd7c9369d92d7bfe4b7065ff440e739dbfab620a78e49942220.jpg)  
(a)

![](images/593e7b9dae46f2cdc0e7d2cae344a6accc870f281d99884805bf67fdc58cd5e4.jpg)  
(b)

![](images/6a0e70e2119070ba18a065ef9881b01f1189bc39e52882c2914ea6752e930b04.jpg)  
(c)  
Fig. 2. Selective correction examples from GalaxyPPG with Deep PPG as the fixed core estimator. (a) Retention. (b) Correction. (c) Rejection. From top to bottom, each column shows raw PPG, overlaid reference/original/framework heart rates, and raw ECG. Time is relative to the central window start; shading marks the 0–8 s central window, and heart rate points are located at window endpoints. Black dashed lines indicate the ECG reference, gray open squares the original estimates, blue circles framework outputs, and crosses rejected internal candidates only. Overlapping curves retain their actual values without artificial offsets. Signal amplitudes use the original recording units. Fold and seed are specified in S9.

## B. Error, Correction Benefits, and Harms With All Available Outputs

Without additional rejection, the framework reduced macro-averaged MAE on all four datasets [Fig. 3(a)]. MAE decreased from 13.68 to 9.23 bpm on DaLiA and from 45.41 to 34.61 bpm on Wrist. It decreased from 38.50 to 21.38 bpm on Galaxy and from 39.97 to 26.08 bpm on Wild. The Galaxy reduction from 38.50 to 21.38 bpm corresponds to the maximum relative reduction of 44.5% reported in the abstract. This percentage is calculated from the macroaveraged MAEs, rather than by averaging estimator-specific percentage reductions. Table S3 reports the confidence intervals. Shared coverage was 100.00%, 98.29%, 97.81%, and 99.34%, respectively; the first value is rounded (Table S10). These reductions therefore did not depend on rejecting additional windows. Results were averaged equally across estimators within participants and then across participants. In Fig. 3(a), each dot represents the participantaveraged MAE of one core estimator under the Original or Framework condition. Open diamonds indicate equal weight means across the ten estimators; these are summaries of errors, not ensemble predictions. Figs. S5, S7, and S8 show participant-level and estimator-level results, and Table S9 reports MAE reductions for selected estimators. These comparisons distinguish the final error after correction from the size of the correction gain; neither alone establishes a universally preferred core estimator.

Beneficial corrections were more frequent than harmful corrections on every dataset [Fig. 3(b)]. Beneficial proportions were 16.06%, 30.74%, 36.64%, and 26.99% on DaLiA, Wrist, Galaxy, and Wild, respectively. Harmful proportions were 1.68%, 6.77%, 3.52%, and 3.12%; the remaining 82.26%, 62.49%, 59.84%, and 69.88% had error changes within ±1 bpm. All three categories use the same paired-window denominator. The middle category includes unchanged outputs and small error changes, so it does not equal the retention rate. Severe added error occurred in 0.37%, 3.89%, 1.61%, and 1.35% of all reference-eligible windows, respectively (Table S3). These proportions describe the frequency of error changes and should be read alongside final MAE.

The primary analysis retained all ten prespecified implementations. Leave-one-estimator-out analyses tested whether a single implementation dominated the macro-averages. Supplementary Methods S21 and Figs. S7 and S21 report these comparisons, implementation differences, and the rationale for separately excluding the local TROIKA variant.

## C. Contributions of Correction and Selection at Fixed Target Coverage

At 80% target coverage, numerical correction accounted for most of the total MAE reduction [Fig. 4(e)]. Correction contributed 3.88, 11.24, 17.79, and 14.70 bpm on DaLiA, Wrist, Galaxy, and Wild, respectively. Window selection contributed 2.38, 0.95, 1.50, and 0.32 bpm (Table S5). Ratios of aggregate means attributed approximately 62%, 92%, 92%, and 98% of the reduction to correction. Same-risk rejection accepts the same windows as the framework and preserves original estimates, making this comparison independent of changes in the accepted set. The percentages summarize this operating point; they do not describe every participant or estimator.

Fig. 4(a)–(d) show error curves across target coverage levels; estimator-specific curves are provided in Figs. S1– S4. Reducing target coverage from 80% to 60% decreased macro-averaged MAE from 6.22 to 4.63 bpm on DaLiA, from 32.47 to 30.59 bpm on Wrist, from 18.39 to 15.70 bpm on Galaxy, and from 24.45 to 22.72 bpm on Wild. These reductions came at the cost of fewer retained outputs. Numerical correction contributed more than window selection to the total MAE reduction at both operating points. Accepted windows were determined by ranking risk scores within target recordings. Insufficient finite predictions and integer rounding of accepted counts could reduce actual coverage below the target. This evaluation differs from direct transfer of fixed source thresholds, which is reported separately.

(a)  
![](images/aeddc897c302cfca481bd0932c6e4a1714d4c39c69868fa056f1c8dbc2ea042b.jpg)

MAE across ten core estimators  
![](images/683f6ab89943742c5da933bf351654b053329edbd371597b961beb2e6c0330d4.jpg)

![](images/fbd88c69b3c08b499c0c6466db9407fd7585d9e03b312a4855dc20e659c656f0.jpg)

![](images/bec37ec59b5963ce5153ecb3c595364eb43e189daf8528a341bd8b68ebc1cd8f.jpg)

(b)  
![](images/28a4bc471515e8b8c9c4f136072356668d5e9c12110656d1a4e52fc07bbe9568.jpg)  
Fig. 3. Error and correction benefits and harms for ten prespecified core estimators using all available outputs. (a) Distributions of participant averaged MAE across the ten estimators before and after correction. Each dot represents one estimator, with configurations averaged within each participant and then participants weighted equally. Violin widths show kernel-density estimates truncated to the observed range; short horizontal lines indicate medians, and open diamonds indicate equal-weight means across estimators. The violins describe between-estimator variation, not participant-bootstrap confidence intervals. The four dataset panels use different vertical-axis ranges, and all ten estimators are retained. (b) Macro-averaged proportions of beneficial corrections (blue), error changes within ±1 bpm (gray), and harmful corrections (orange) in the same paired window set. Beneficial and harmful corrections decrease and increase absolute error, respectively, by more than 1 bpm relative to the original estimate. Gray includes unchanged outputs and small error changes; orange proportions are labeled to the right. Each estimator is compared on the same available windows without additional risk rejection. Source configurations are averaged within each partic ipant and estimator, followed by equal weighting of estimators and participants; heart rate predictions are not fused. Panel (b) shows composition point estimates. Table S3 reports these proportions and macro-averaged MAEs, together with participant-bootstrap 95% confidence intervals from 10,000 resamples. Participant counts and coverage are provided in Table S10, and estimator-composition sensitivity in Fig. S21.

![](images/d7fbc084f8dbd71c5d0256114d17e8cd7c4d6568eee457c98eaa2836d1de1a01.jpg)  
(a)

![](images/60b3ccb6b112cb477f71f91a01a9cfe0f7c42bb0aed849f5fb77cbe6c0df67d5.jpg)  
(b)

![](images/c9ca2d4859ffd66a0f1b030a4b5eb5c7592d16ce99a227b5650c257577a7faab.jpg)  
(c)

![](images/f24be9accbe3c1d302d993ab691da4ca331934f431a045ed0b075a44af2b066a.jpg)

(d)  
![](images/807977321569977822f1648be9201da61466ff54434413376a91966493a4c559.jpg)  
Fig. 4. Risk–coverage relationships and paired decomposition of MAE reduction. (a)–(d) DaLiA, Wrist, Galaxy, and Wild. Ridge rejection selects windows using original-error predictions from source-domain ridge regression; same-risk rejection uses the framework risk scores and accepted set. Both retain original heart rates. (e) At 80% target coverage, orange denotes correction gains on the same accepted set, green denotes selection gains with original heart rates fixed, and points indicate their sum. Positive values indicate error reduction. Curve shading and error bars for total reduction indicate participant-bootstrap 95% confidence intervals; total intervals are calculated from paired total effects.

## D. Effects of Module Ablations on Error and Correction Harm

Removing motion evidence or historical memory increased macro-averaged MAE on every dataset (Table II). Motion ablation increased MAE by 0.97, 5.98, 6.08, and 7.08 bpm on DaLiA, Wrist, Galaxy, and Wild, respectively. Memory ablation increased MAE by 0.87, 1.52, 1.17, and 1.35 bpm. Harmonic ablation increased MAE by 0.61 and 0.93 bpm on DaLiA and Wrist but reduced it by 1.62 and 0.94 bpm on Galaxy and Wild. The Wrist interval for harmonic ablation included zero (95% confidence interval: −0.23 to 2.10 bpm). Tables II and S6 report the remaining intervals.

TABLE II EFFECTS OF MODULE ABLATIONS ON MEAN ABSOLUTE ERROR
<table><tr><td>Dataset</td><td>No harmonic ∆MAE (bpm)</td><td>No motion ∆MAE (bpm)</td><td>No memory ∆MAE (bpm)</td></tr><tr><td>DaLiA</td><td>+0.61 [+0.36, +0.85]</td><td>+0.97 [+0.74, +1.24]</td><td>+0.87 [+0.71, +1.05]</td></tr><tr><td>Wrist</td><td>+0.93 [-0.23, +2.10]</td><td>+5.98 [+3.02, +8.67]</td><td>+1.52 [+0.52, +2.46]</td></tr><tr><td>Galaxy</td><td>-1.62 [-1.91, -1.31]</td><td>+6.08 [+5.51, +6.63]</td><td>+1.17 [+0.89, +1.48]</td></tr><tr><td>Wild</td><td>-0.94 [-1.35, -0.56]</td><td>+7.08 [+6.71, +7.43]</td><td>+1.35 [+1.10, +1.58]</td></tr></table>

Note: ΔMAE is ablated-model MAE minus full-framework MAE, in bpm; positive values indicate higher mean error. Brackets denote partic ipant-bootstrap 95% confidence intervals. DaLiA, Wrist, Galaxy, and Wild include 15, 8, 20, and 14 participants, respectively.

Average accuracy and severe added error did not respond consistently to ablation (Tables II and III). For example, removing motion evidence increased MAE by 6.08 and 7.08 bpm on Galaxy and Wild but decreased the severe added-error proportion by 1.25 and 0.94 percentage points, respectively. Removing historical memory also increased MAE while reducing the severe added-error proportion on these datasets. Thus, a module that improves average accuracy does not necessarily reduce every measure of correction harm.

EFFECTS OF MODULE ABLATIONS ON THE SEVERE ADDED-ERROR PROPORTION
<table><tr><td>Dataset</td><td>No harmonic ∆ severe added error (percentage points)</td><td>No motion ∆ severe added error (percentage points)</td><td>No memory ∆ severe added error (percentage points)</td></tr><tr><td>DaLiA</td><td>-0.12 [-0.24, -0.03]</td><td>-0.07 [-0.19, +0.02]</td><td>-0.07 [-0.19, +0.02]</td></tr><tr><td>Wrist</td><td>-0.40 [-1.42, +0.51]</td><td>+0.31 [-1.41, +3.05]</td><td>-1.02 [-1.87, -0.22]</td></tr><tr><td>Galaxy</td><td>-1.40 [-1.99, -0.97]</td><td>-1.25 [-1.85, -0.84]</td><td>-0.84 [-1.19, -0.57]</td></tr><tr><td>Wild</td><td>-0.86 [-1.11, -0.64]</td><td>-0.94 [-1.09, -0.81]</td><td>-0.44 [-0.68, -0.21]</td></tr></table>

Note: Values are the ablated-model severe added-error proportion minus the full-framework proportion, in percentage points. Positive values indicate more severe added error. Brackets denote participant-bootstrap 95% confidence intervals. Participant counts were 15, 8, 20, and 14, respectively.

## E. Transfer of Fixed Source-Domain Thresholds

Fixed source thresholds did not preserve their coverage or error targets on external datasets (Table IV). The nominal 80% coverage threshold yielded 79.07%, 42.29%, 45.48%, and 30.50% coverage on DaLiA, Wrist, Galaxy, and Wild, respectively. A separate rule targeted source MAE no greater than 5 bpm. After transfer, output MAE exceeded 5 bpm in 39.55%, 91.45%, 49.72%, and 73.47% of nonempty configurations, respectively. Zero-output configuration counts were 48/450, 183/1200, 365/3000, and 227/2100. These configurations were excluded from the exceedance denominator. The two rules test coverage transfer and error-budget transfer separately. Table S7 and Fig. S9 provide detailed results.

TABLE IV TRANSFER OF COVERAGE AND ERROR TARGETS WITH FIXED SOURCE THRESHOLDS
<table><tr><td>Dataset</td><td>80% source-coverage threshold Actual coverage (%)</td><td>Source MAE 5 bpm threshold Exceeding / nonempty (%)</td><td>Source MAE 5 bpm threshold Zero-output / all</td></tr><tr><td>DaLiA</td><td>79.07 [73.80, 84.16]</td><td>159/402 (39.55%)</td><td>48/450</td></tr><tr><td>Wrist</td><td>42.29 [38.46, 46.42]</td><td>930/1017 (91.45%)</td><td>183/1200</td></tr><tr><td>Galaxy</td><td>45.48 [40.50, 50.32]</td><td>1310/2635 (49.72%)</td><td>365/3000</td></tr><tr><td>Wild</td><td>30.50 [26.07, 35.73]</td><td>1376/1873 (73.47%)</td><td>227/2100</td></tr></table>

Note: Coverage is a participant macro-average; brackets denote bootstrap 95% confidence intervals. Exceedance means MAE > 5 bpm among retained outputs, with nonempty configurations as the denominator. Zero-output configurations are counted separately and excluded from that denominator. Configuration-count proportions are descriptive and have no reported confidence intervals. The two sets of results use different source-domain threshold-selection rules.

## F. Prototype System and Evaluation of Acquired Data

Fig. 5(d) compares ECG references, original Deep PPG estimates, and framework outputs during static, walking, and running recordings from one participant. Recovery-rest recordings were also included in the aggregate analysis Without additional rejection, ten-estimator macro-averaged MAE decreased from 9.09 to 4.45 bpm [Fig. 5(e)]. Eight estimators improved, BeliefPPG was essentially unchanged, and WFPV increased slightly from 1.09 to 1.10 bpm. Restricting the comparison to 289 windows available for all ten estimators reduced MAE from 9.24 to 4.33 bpm. Table S12 and Fig. S12 report estimator-specific counts and errors; Supplementary Methods S21 examines sensitivity to estimator composition.

At 80% target coverage, each method accepted 259 of 324 reference-eligible windows, giving 79.94% actual coverage. Macro-averaged MAE was 8.60 bpm with original-risk selection, 7.49 bpm with same-risk rejection, and 3.30 bpm with the full framework. Correction contributed 4.19 bpm of the total 5.30 bpm reduction, while selection contributed 1.11 bpm, or approximately 79% and 21%, respectively [Fig. 5(f)]. Ridge rejection achieved 6.83 bpm, outperforming same-risk rejection at this operating point. The framework’s risk ranking therefore did not improve on every selection baseline. Figs. S13 and S14 show the full curves and estimator-specific decompositions.

The prototype recordings also showed a trade-off between accuracy and conservative correction. Without additional rejection, adding M4 reduced the macro-averaged harmful correction rate from 2.18% to 0.42% but increased MAE from 2.57 to 4.45 bpm. Figs. S12 and S16 report these comparisons. Nominal 80% source-coverage thresholds yielded 58.21%–86.85% actual coverage across methods. The separate 5 bpm source-MAE thresholds produced both error exceedances and zero-output configurations (Table S13; Fig. S17). Macro-averaged MAE decreased during static, walking, and running recordings. However, estimator-specific gains and protection of originally accurate esti mates were inconsistent (Figs. S19 and S20). Figs. S10, S15, and S18 show synchronized raw signals, red-channel sensitivity, and preprocessing examples, respectively.

![](images/b9fbba7495fcca7a7e245e6aba4e0365d57dbf4c3d3c444c18575f8ce636645e.jpg)

![](images/94bd8b6516cfc0ff0f2a84c37a0bd2943cc7465342338f9b57791bda45eb2a09.jpg)  
(b)

![](images/4d81268e0eefffc8cc8d791f407307ac010a5cfd5f79d427c6a8f6cedd1aa307.jpg)  
(c)

![](images/11ee6e21f7f7b90dc73f8b1156df21856d742d638a368e6b67c6bbfdedbafcf4.jpg)

![](images/9a3ba07a21f3440d57598ef31f9dbb5452eb263f575afd522b04289387850f10.jpg)  
(d)

![](images/1320cdfb9de51e1b4be4a5030e3912032bfa4bdcf502123eb3821a8e83b105fa.jpg)

![](images/70926c743e6d3af88fa4123d5503288c15721224c4bb2490f24a356c03c39a9e.jpg)  
(e)

![](images/3f2e0f8ddaa3ca436683d53af37966b626553ad8a7c730260d8b553b36f93c21.jpg)  
(f)  
Fig. 5. Wearable acquisition, smartphone deployment, and prototype evaluation. (a) Measurement setup and independent ECG reference. (b) Hardware structure and appearance. (c) Smartphone interface and processing workflow for the deployed model and framework. (d) Original Deep PPG estimates, ECG references, and framework outputs in acquired static, walking, and running recordings using fixed source thresholds; crosses denote rejected internal candidates. The three states use different vertical-axis ranges to resolve within-state variation. Configurations are specified in S20.3. (e) Ten-estimator macro-averaged MAE pooled across four states without additional rejection, and results for the 289 common windows. (f) Method-specific correction and selection gains at 80% target coverage; points denote totals, and negative components are retained. Each strategy accepts 259/324 windows. Correction fixes the framework accepted set, whereas selection fixes original heart rates. Bars are means across frozen source configurations; error bars in (e) indicate configuration ranges. ${ \mathfrak { n } } = 1 ;$ statistics are from backend analysis of saved recordings. Panel (f) uses risk ranking, unlike (e).

## IV. DISCUSSION

## A. Selective Correction Relative to Original Estimates

Evaluating candidate gains and harms reduced average heart rate error while leaving the core estimators fixed. The all-output comparison separates this improvement from the effect of rejecting additional windows. Beneficial corrections outnumbered harmful corrections on all four datasets, although some windows still deteriorated [Fig. 3(b)]. These proportions measure frequency rather than the magnitude of error changes and must be interpreted alongside MAE. At 80% target coverage, correction on the same accepted windows contributed most of the average error reduction. Thus, the observed gains cannot be explained solely by withholding difficult outputs.

The architecture separates candidate generation from the decision to replace an estimate. M3 forms a distribution from current and historical evidence; M4 evaluates candidates relative to the original estimate. High posterior prob ability alone does not trigger replacement. A candidate must also satisfy the predicted-gain, predicted-harm, and posterior-error conditions. The original estimate remains available when no replacement qualifies, and a separate risk gate determines whether to release the selected value. The three outcomes therefore arise from sequential decisions rather than a single three-class network.

The term agent describes the organization of observations, a limited internal state, and constrained action selection [14]. Execution follows a fixed sequence. Historical states are updated from input evidence, without using previous output heart rates or online reference feedback. The results support this implementation of selective correction. They do not establish autonomous planning, collaboration among estimators, or one set of auxiliary weights that works for every estimator.

## B. Relationship to Existing Estimation Methods

The auxiliary modules reorganize existing signals for relative-error assessment rather than introduce new sensor measurements. TROIKA, JOSS, SpaMA, and WFPV already use spectral separation, motion processing, or tracking [4]–[7]. Deep PPG and BeliefPPG also use time–frequency information, and BeliefPPG additionally models heart rate dynamics and uncertainty [3], [9]. The observed gains therefore cannot be attributed simply to an absence of spectral or historical information in these methods.

The auxiliary modules may complement existing estimators through their representations and training objectives. M1 and M2 retain power, harmonic, and motion features at multiple time scales on a common heart rate grid. M3 summarizes recent history, and M4 learns paired-error differences for a specific estimator. Scalar heart rates or distribution summaries may omit information useful for these comparisons. Additional supervision may also help the auxiliary models learn estimator-specific biases. We did not independently vary representation, model capacity, and supervision, so their individual contributions remain unresolved.

Post-calibration is an established approach [11]. Device adaptation, temporal modeling, and multisite measurements offer other routes to improved estimation [18]–[20]. We did not compare all these approaches under matched inputs, training data, and computational budgets. Our focus is the assessment of relative gains and harms before replacement. If an estimator is already Bayes-optimal for absolute error, post-processing with identical information cannot universally reduce risk under the same target distribution [14]. Any additional benefit must therefore be demonstrated for the specific estimator and operating conditions.

## C. Core-Estimator Differences and Module Contributions

Model-specific results constrain the interpretation of macro-averages. Participant-paired MAE reductions for KID-PPG were 0.35, 8.56, 1.09, and 0.12 bpm on DaLiA, Wrist, Galaxy, and Wild, respectively. The prespecified JOSS, SpaMA, and WFPV implementations also improved on average across all four datasets (Table S9; Fig. S8). Additional gains can therefore occur for estimators that already incorporate spectral or temporal processing, although their magnitudes depend on the model and dataset.

BeliefPPG changed by less than 0.001 bpm in absolute MAE on every dataset, indicating that the framework largely retained its outputs. Overlap between its representation and the auxiliary modules is one possible explanation. How ever, near-zero change establishes neither redundancy nor optimality. Original error and scope for correction also differ across implementations. Macro-averages should therefore be read with estimator-specific results and the composition analyses in Supplementary Methods S21, rather than used to rank the original published methods.

The ablations support a contribution from motion evidence and recent history to average accuracy. Harmonic evidence showed less consistent benefits across datasets. Pulse and motion rhythms, including their harmonics, can occupy the same frequency bands [2], [4]–[7]. Feature weights learned on source data may therefore transfer poorly to different activities or sensors. These are possible explanations for the Galaxy and Wild ablation results, rather than established causes. Each ablation also removes related interaction features, preventing an independent attribution of the total gain to individual modules.

## D. Correction Harm and Decision Traceability

Mean error and correction harm capture different aspects of performance. Ablations with opposing effects in Tables II and III show that a module reducing overall MAE may still increase severe deterioration in a small number of windows. Model comparisons should therefore jointly report error changes on the same accepted set, harmful correction rates, and coverage. Uncertainty estimation and selective prediction provide a basis for identifying low-confidence outputs [9], [10], [12], [13]. When a system both modifies estimates and rejects outputs, the contributions of these operations must also be distinguished.

The framework exposes intermediate quantities that can be used to trace decisions, including candidate evidence, historical states, posterior distributions, predicted gains and harms, and utility rankings. Geometric constraints rule out incompatible error predictions but cannot guarantee accurate harm estimates. Fig. 2 aligns signals with actions without showing complete candidate-level trajectories. Similarly, high motion coherence alone does not prove that a spectral peak originates from motion. Tracing an algorithmic decision and explaining its physiological cause require different evidence.

The paired decomposition separates correction from selection but remains descriptive. Correction contributed more than selection at 80% target coverage (Fig. 4), yet some accepted windows still deteriorated. The harmful example in Table S8 illustrates why retention options and candidate constraints must be evaluated using actual outputs. Their presence alone does not establish protection against harmful corrections.

## E. Cross-Dataset Transfer and Deployment Limitations

Lower average error after transfer does not establish sufficient absolute accuracy for deployment. External-dataset MAE remained 21.38–34.61 bpm. Transferring thresholds calibrated for 80% source coverage produced only

30.50%–45.48% coverage on external data. Error-based thresholds also failed to consistently maintain the 5 bpm target. Risk scores and thresholds therefore need independent validation in the intended operating conditions. The 2 bpm predicted-harm limit is a decision criterion, not a bound on actual error.

Data history and implementation differences further limit generalization claims. All four datasets were used during earlier development, and participant provenance is incomplete for some pretrained weights. The current participant splits and weight freezing do not create a confirmatory set untouched throughout the research process. Reference eligibility rules differ across datasets, and Galaxy uses a historical ECG cache checked for consistency. Participant bootstrap intervals do not capture the uncertainty of full retraining. In addition, some preprocessing and core-estima tor operations are retrospective, and KID-PPG retains per-recording adaptation without heart rate labels. Crossmethod statistics therefore cannot be equated directly with the real-time accuracy of the deployed smartphone con figuration.

The prototype combines wearable acquisition, smartphone analysis, and backend evaluation. Its recordings support paired-error comparisons after input adaptation, but they comprise only four recordings from one participant. Fig. 5(e) and Supplementary Methods S21 report common-window and estimator-composition analyses. The cross-estimator averages represent neither the deployed smartphone model alone nor population-level performance. M4 reduced the overall harmful correction rate but increased MAE, and some protection metrics worsened for individual methods (Fig. S20). Its usefulness therefore depends on the acceptable balance between error, harm, and coverage.

Smartphone analysis runs in real time; the multi-method backend comparisons use saved recordings and standard ized preprocessing. Earlier exploration of the same participant, zero-phase filtering, and whole-block amplitude normalization limit these comparisons. Repeated source configurations do not remove those limitations. Reconstructed reception times do not measure hardware transmission latency, and the low effective reception rate limits recoverable bandwidth. Independent end-to-end tests should use a fixed deployed model, defined input units, causal preprocessing, and prespecified thresholds. Such tests should assess accuracy, correction harm, coverage, and latency against synchronized reference measurements.

## V. CONCLUSION

We presented a selective correction framework that combines signal evidence, recent history, and predicted gains and harms around a fixed heart rate estimator. Across ten estimators and four public datasets, the framework reduced macro-averaged MAE without additional rejection. At 80% target coverage, numerical correction on the same ac cepted windows accounted for most of the error reduction. We also implemented wearable acquisition and smartphone analysis, with backend comparisons on prototype recordings. Gains varied by estimator and dataset, harmful corrections remained, and source-calibrated thresholds did not transfer reliably. Further evaluation in inde pendent participants should measure end-to-end accuracy, correction harm, output coverage, and latency.

## DECLARATIONS AND REPRODUCIBILITY

Data and code availability: Public datasets are described in [3] and [15]–[17]. The local study archive contains participant summaries, window-level predictions, configuration results, reference-eligibility records, analysis and plotting scripts, prototype recordings, timing reconstruction, source-model bindings, figure data, and editable figures. Sources and local indices for complete public recordings are retained separately. Public repository links for the study code, weights, and results are not provided in this version.

Author self-recordings and ethics status: The new prototype recordings were collected from the first author for deployment and technical feasibility evaluation. These recordings do not establish generalization across participants or clinical validity. No institutional ethics approval or exemption determination was obtained for these author selfrecordings. No other participants were recruited for the prototype evaluation.

## REFERENCES

[1] J. Fine et al., “Sources of inaccuracy in photoplethysmography for continuous cardiovascular monitoring,” Biosensors, vol. 11, no. 4, Art. no. 126, 2021, doi: 10.3390/bios11040126.

[2] D. Biswas, N. Simões-Capela, C. Van Hoof, and N. Van Helleputte, “Heart rate estimation from wrist-worn photoplethysmography: A review,” IEEE Sensors J., vol. 19, no. 16, pp. 6560–6570, 2019, doi: 10.1109/JSEN.2019.2914166.

[3] A. Reiss, I. Indlekofer, P. Schmidt, and K. Van Laerhoven, “Deep PPG: Large-scale heart rate estimation with convolutional neural networks,” Sensors, vol. 19, no. 14, Art. no. 3079, 2019, doi: 10.3390/s19143079.

[4] Z. Zhang, Z. Pi, and B. Liu, “TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise,” IEEE Trans. Biomed. Eng., vol. 62, no. 2, pp. 522–531, 2015, doi: 10.1109/TBME.2014.2359372.

[5] Z. Zhang, “Photoplethysmography-based heart rate monitoring in physical activities via joint sparse spectrum reconstruction,” IEEE Trans. Biomed. Eng., vol. 62, no. 8, pp. 1902–1910, 2015, doi: 10.1109/TBME.2015.2406332.

[6] S. M. A. Salehizadeh, D. Dao, J. Bolkhovsky, C. Cho, Y. Mendelson, and K. H. Chon, “A novel time-varying spectral filtering algorithm for reconstruction of motion artifact corrupted heart rate signals during intense physical activities using a wearable photoplethysmogram sensor,” Sensors, vol. 16, no. 1, Art. no. 10, 2016, doi: 10.3390/s16010010.

[7] A. Temko, “Accurate heart rate monitoring during physical exercises using PPG,” IEEE Trans. Biomed. Eng., vol. 64, no. 9, pp. 2016–2024, 2017, doi: 10.1109/TBME.2017.2676243.

[8] A. Burrello et al., “Embedding temporal convolutional networks for energy-efficient PPG-based heart rate monitoring,” ACM Trans. Comput. Healthcare, vol. 3, no. 2, pp. 1–25, 2022, doi: 10.1145/3487910.

[9] V. Bieri, P. Streli, B. U. Demirel, and C. Holz, “BeliefPPG: Uncertainty-aware heart rate estimation from PPG signals via belief propagation,” in Proc. Conf. Uncertainty Artif. Intell. (UAI), vol. 216, 2023, pp. 173–183. [Online]. Available: https://proceedings.mlr.press/v216/bieri23a.html

[10] C. Kechris, J. Dan, J. Miranda, and D. Atienza, “KID-PPG: Knowledge informed deep learning for extracting heart rate from a smartwatch,” IEEE Trans. Biomed. Eng., vol. 72, no. 3, pp. 870–877, 2025, doi: 10.1109/TBME.2024.3477275.

[11] T. Choksatchawathi et al., “Improving heart rate estimation on consumer grade wrist-worn device using post-calibration approach,” IEEE Sensors J., vol. 20, no. 13, pp. 7433–7446, 2020, doi: 10.1109/JSEN.2020.2979191.

[12] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems, vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

[13] Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an integrated reject option,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 97, 2019, pp. 2151–2159. [Online]. Available: https://proceedings.mlr.press/v97/geif man19a.html

[14] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2021, chs. 2 and 16. [Online]. Available: https://aima.cs.berkeley.edu

[15] D. Jarchi and A. J. Casson, “Description of a database containing wrist PPG signals recorded during physical exercise with both accelerometer and gyroscope measures of motion,” Data, vol. 2, no. 1, Art. no. 1, 2017, doi: 10.3390/data2010001.

[16] S. Park, D. Zheng, and U. Lee, “A PPG signal dataset collected in semi-naturalistic settings using Galaxy Watch,” Sci. Data, vol. 12, Art. no. 892, 2025, doi: 10.1038/s41597-025-05152-z.

[17] M. Meier, B. U. Demirel, and C. Holz, “WildPPG: A real-world PPG dataset of long continuous recordings,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://papers.nips.cc/paper\_files/paper/2024/hash/0433292a3eb101df8f4d72a63f5410d4-Abstract-Datasets\_and\_Benchmarks\_Track.htm

[18] A. Shyam, V. Ravichandran, S. P. Preejith, J. Joseph, and M. Sivaprakasam, “PPGnet: Deep network for device independent heart rate estimation from photoplethysmogram,” in Proc. 41st Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2019, pp. 1899–1902, doi: 10.1109/EMBC.2019.8856989.

[19] M. Wilkosz and A. Szczęsna, “Multi-headed Conv-LSTM network for heart rate estimation during daily living activities,” Sensors, vol. 21, no. 15, Art. no. 5212, 2021, doi: 10.3390/s21155212.

[20] M. Meier and C. Holz, “Robust heart rate detection via multi-site photoplethysmography,” in Proc. 46th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2024, pp. 1–4, doi: 10.1109/EMBC53108.2024.10782155.

# Supplementary Information

# An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates

Xu Wang<sup>1,2</sup>, Shuhan Yi<sup>2</sup>, and Zhen Cheng<sup>2\*</sup>

<sup>1</sup> School of Engineering, The Hong Kong University of Science and Technology, Hong Kong 999077, China

<sup>2</sup> School of Biomedical Engineering, Tsinghua University, Beijing 100084, China

\*Corresponding author:

Zhen Cheng, School of Biomedical Engineering, Tsinghua Medicine, Tsinghua University, Beijing 100084, P. R. China. Email: zcheng@mail.tsinghua.edu.cn

This supplement provides the reference eligibility criteria, training and calibration procedures, module definitions, and proofs supporting the main manuscript. Sections S11–S14 specify the signal features, historical state, candidate construction, and gain–harm model. Section S19 defines the outputs and evaluation metrics. Section S20 describes prototype acquisition, preprocessing, and offline evaluation; Section S21 examines sensitivity to estimator composition. Tables S1–S13 and Figs. S1–S21 provide supporting results. Reference numbers are shared with the main manuscript.

## Table of contents

S1. Datasets, Reference Eligibility, and Participant Splits

S2. Retention, Correction, and Rejection

S3. Physiological Evidence and Candidate Construction

S4. Consistent Gain–Harm Decisions

S5. Auxiliary Training and Evaluation With Fixed Configurations

S6. Comparators, Risk Scores, and Thresholds

S7. Cross-Dataset Transfer With Fixed Auxiliary Models

S8. Metrics, Paired Decomposition, and Statistics

S9. Selection and Display of Signal Examples

S10. Scope of Prototype Wearable Evaluation

S11. M1 Pulse Periodicity and Harmonic Evidence

S12. M2 Motion Coherence and Residual Spectral Evidence

S13. M3 Five-Position Memory and Posterior Network

S14. M4 Consistent Gain–Harm Decisions

S15. Independent Output Risk Gate

S16. Two-Stage Training and Losses

S17. Decoding Constraints and Information Limits

S18. Roles and Interfaces in the Agent Architecture

S19. Mathematical Definitions of Outputs and Metrics

S20. Single-Participant Acquisition and Offline Evaluation

S21. Estimator Composition and TROIKA Sensitivity

Supplementary Tables S1-S13

Supplementary Figures S1-S21

## SUPPLEMENTARY METHODS

## S1. Datasets, Reference Eligibility, and Participant Splits

The public datasets are PPG-DaLiA, Wrist Exercise, GalaxyPPG, and WildPPG, abbreviated here as DaLiA, Wrist, Galaxy, and Wild [3], [15]–[17]. Heart rate and mean absolute error (MAE) are reported in beats per minute (bpm). Signal inputs are wrist photoplethysmography (PPG) and triaxial acceleration (ACC). Electrocardiography (ECG)- derived heart rates or published heartbeat annotations are used only for training targets, source calibration, and evaluation, not as deployment inputs. The common evaluation grid uses 8 s windows with a 4 s stride. Multiple Wrist activity recordings are pooled within each participant; its 19 recordings are not treated as 19 independent participants. Wild uses wrist green-channel PPG and the corresponding acceleration, without combining body sites. Registered native outputs from each core estimator are retained and aligned to the common time grid.

Table I reports the actual frozen analysis scope, rather than the full populations described in the dataset papers. Galaxy originally reported 24 participants [16], whereas the current input manifest contains 23 participants, P02– P24; no P01 results are imputed. Galaxy participants P05, P07, and P22 and Wild participants fex and p5d have no eligible references under the current rules. Their inputs, predictions, and eligibility records are retained, but their mean errors are not calculated. These exclusions depend only on reference eligibility, not PPG estimation error. The four datasets include 62 participants, of whom 57 have eligible references. Analyses remain separate by dataset. DaLiA uses the frozen ECG-derived window heart rates. Wrist uses published heartbeat events. Each 8 s window must contain at least three events, at least two RR intervals corresponding to 35–220 bpm, and at least 80% valid intervals. Heart rate is calculated from the median valid RR interval. Galaxy reuses a historical ECG reference cache whose bindings to raw inputs, processing code, and time grids were checked. Eligible windows require agreement within 5 bpm between XQRS and a second ECG estimator; the XQRS value is used. Cache consistency checks do not constitute independent validation of reference accuracy. Wild references are recomputed from the current raw ECG using segment-wise R peaks and stable RR intervals. Existing recording-level rules require at least 80% reference coverage, 90% physiological RR intervals, and 80% stable RR intervals. All windows are ineligible if a recording fails these rules. The two participants without eligible references were identified by the current rules, rather than by reusing an earlier 14-participant list. Reference rules differ across datasets, so crossdataset error differences cannot be attributed solely to devices.

DaLiA uses five fixed groups: G0={S1,S6,S11}, G1={S2,S7,S12}, G2={S3,S8,S13}, G3={S4,S9,S14}, and G4={S5,S10,S15}. Outer fold k evaluates Gk, calibrates on G(k+1 mod 5), and trains on the remaining nine participants. Each configuration uses seeds 219, 220, and 221. A participant contributes to DaLiA results only in their evaluation fold. All participants had been accessed during earlier development. These splits isolate participants during auxiliary fitting but cannot make the evaluation set untouched throughout the research process.

## S2. Retention, Correction, and Rejection

The original estimate is the output of the fixed core estimator. The auxiliary modules generate alternative candidate heart rates. After selecting the original estimate or a candidate, a separate risk gate determines whether to release the selected value. Reference heart rates are not used to generate candidates during inference.

A candidate’s true error is its absolute difference from the reference heart rate. Relative net gain is original-estimate error minus candidate error and may be negative. Added error is the positive part of candidate error minus originalestimate error. Retention outputs the original estimate; correction outputs another candidate; rejection withholds heart rate for the window. Rejection changes output status but does not remove the window from the referenceeligible set.

The harmful correction rate is the proportion of paired accepted windows in which absolute error increases by more than 1 bpm relative to the finite original estimate. This differs from a heart rate change exceeding 1 bpm and from the continuous added-error target used in training. We also record deterioration among windows whose original error is at most 5 bpm and tail measures for error or added error exceeding 10 bpm. These measures expose local failures that mean error can conceal.

## S3. Physiological Evidence and Candidate Construction

The auxiliary posterior combines an estimator-related prior with 30-dimensional physiological evidence. Features describe PPG structure around candidate frequencies, half- and double-frequency relationships, motion-related spectra, and their interactions. Estimators with native predictive distributions and those producing only scalar hear rates use separate adapters. Supplementary evidence-based risk for scalar estimators is not described as their native confidence. For KID-PPG, the actual continuous mean is retained as the original estimate, rather than replaced by a mean on a truncated grid.

The history module applies learnable channel-wise decay to five aligned feature positions, including the current window. Current evidence, historical state, and their difference at each candidate frequency are used to model the auxiliary posterior. The posterior head has 90 inputs, 24 hidden units, and sigmoid linear unit (SiLU) activation. It predicts a bounded energy adjustment, which is normalized with the temperature-scaled log prior to form a heart rate distribution. History indices never access future windows. This design compares current evidence with recent context, but some core-estimator preprocessing remains retrospective; it does not validate end-to-end real-time operation.

The auxiliary posterior provides nine candidates: the original estimate, mean, mode, 10th, 25th, 50th, 75th, and 90th percentiles, and the midpoint between the original estimate and posterior median. The original estimate always occupies the first position. Each candidate has a posterior expected absolute error, expected added error relative to the original estimate, and posterior support for error within 5 bpm. This finite candidate space allows the framework to retain an estimate instead of producing a new value in every window.

## S4. Consistent Gain–Harm Decisions

M4 uses 106 features per candidate. The first 16 describe candidate and original values, correction direction and magnitude, posterior error, added error, support, and distribution statistics. The remaining 90 comprise current evidence, historical state, and current-evidence differences between the candidate and original estimate. This candidate-relative difference serves a different purpose from the temporal difference in the posterior head.

M4 contains two linear branches. A branch mapping 106 features to three outputs produces allocation logits for every candidate; a branch mapping 106 features to one output produces a base-error parameter from the originalestimate features only. Softmax and softplus transformations are followed by consistent decoding to jointly predic gain, added error, and absolute error. Candidate distance limits achievable gain, while a nonnegative base error and triangle-inequality constraints maintain consistency among predictions. Training targets are the actual paired absolute errors, with an extra penalty for underestimating added error. These constraints govern relationships between predicted quantities. They do not guarantee calibration in unseen participants or safety in individual windows.

The primary configuration requires predicted net gain of at least 1 bpm, predicted added error no greater than 2 bpm, and lower posterior expected error than the original estimate. Eligible candidates are ranked by predicted net gain minus twice predicted added error. The original estimate remains available with utility zero. Ties favor earlier candidates, so equal utility does not force a correction. The primary configuration has no correction-magnitude cap. These thresholds apply to predictions and do not bound actual error increases to 2 bpm.

## S5. Auxiliary Training and Evaluation With Fixed Configurations

All core estimators remain fixed during auxiliary training, and auxiliary weights are fitted separately for each estimator. Within the nine source-training participants of each outer fold, cross-fitting uses the three existing participant groups. M3 is trained on six participants and generates out-of-fold posteriors for the other three. M4 uses only these posteriors and history features from the corresponding M3 models. The final M3 is trained on all nine source participants before predicting for three calibration and three evaluation participants. This avoids using M3 in-sample posteriors to train M4. It does not remove participant exposure during original-estimator pretraining or earlier research.

Each posterior head uses AdamW with learning rate 0.002, weight decay 0.03, batch size 96, and 2000 training steps. References become Gaussian soft targets with a standard deviation of 3 bpm. The loss combines crossentropy and cumulative-distribution discrepancy, with participant-balanced sampling, prior-channel perturbation, and gradient clipping. M4 uses participant-balanced sampling, batch size 256, learning rate 0.003, weight decay 0.1, and 2000 steps. Its features are standardized from source data and clipped. Final-step weights are used throughout; checkpoints are not selected on the evaluation set.

Ten estimators, five outer folds, three seeds, and four variants requiring retraining give 600 configurations. Each trains three cross-fitted posterior heads, one final posterior head, and one M4 module, totaling 2,400 posterior-head fits and 600 M4 fits. Calibration and evaluation predictions were archived before final scoring. The complete internal archive contains 81,000 participant-, strategy-, and coverage-mode metric records. This manuscript reports only the specified strategies, and the plotting package records row counts before and after filtering. Analyses use the archived results without retraining or configuration reselection.

## S6. Comparators, Risk Scores, and Thresholds

The ten registered implementations are BeliefPPG, KID-PPG, TimePPG, Deep PPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV. Names identify the implementations evaluated here, not the best results in their original papers. The local TROIKA implementation is a registered 25 Hz paper-guided variant, not the authors code or a validated reproduction. It uses custom singular spectrum analysis (SSA) component pairing and spectral tracking, with outputs constrained to 24–300 bpm. Training-participant exposure could not be fully established for estimators using author-provided weights. Source-training and early-stopping splits for locally trained estimators also differ from the current auxiliary outer folds. The study therefore measures auxiliary gains around fixed estimators, rather than providing a fair from-scratch ranking of the ten published methods.

Baseline risk definition. The score ${ \boldsymbol { r } } _ { 0 , t }$ is an estimator-specific indicator of uncertainty in the original estimate $h _ { 0 , t } .$ computed without reference heart rate at inference. It is not the observed absolute error. The risk score and the adapted heart rate prior $p _ { t }$ are separate quantities: they need not be constructed from the same weights. Scores are used for within-estimator ranking and estimator-specific source calibration; their magnitudes are not compared across estimators.

BeliefPPG uses the standard deviation of its native predictive distribution, evaluated on the native heart rate bins before adaptation to the common grid: $\mathrm { r o , \mathrm { _ t } = [ \sum _ { k } \pi _ { t , k } ( b _ { k } - \mu _ { t } ) ^ { 2 } ] ^ { 1 / 2 } }$ , where $\pi _ { \mathrm { t } , \mathrm { k } }$ are native predictive probabilities, $\mathsf { b } _ { \mathrm { k } }$ are native bin values, and $\begin{array} { r } { \mathsf { \Pi } \mu _ { \mathrm { t } } = \sum _ { \mathrm { k } } \pi \phantom { } _ { \mathrm { t } , \mathrm { k } } \mathsf { b } _ { \mathrm { k } } } \end{array}$ . KID-PPG uses its predicted Gaussian standard deviation, $\mathrm { r o , t } = 1 + \mathrm { s o f t p l u s } ( \mathrm { z _ { t } } )$ where $\mathrm { Z _ { t } }$ is the model output controlling scale and softplus $( \mathsf { z } ) = \ln ( 1 + \exp ( \mathsf { z } ) )$ . These standard deviations are measured in bpm and are retained from the fixed estimator outputs.

For the other eight estimators, the adapter computes spectral entropy: ${ \bf r _ { 0 , t } } = - \Sigma _ { \mathrm { k } } \ { \bf s } _ { \mathrm { t , k } } \ln ( { \bf s } _ { \mathrm { t , k } } )$ , where $\mathbf { S } _ { \mathrm { t } , \mathbf { k } } \geq 0$ and $\Sigma _ { \mathrm { k } }$ sₜ,ₖ = 1. Natural logarithms are used, without division by the logarithm of the number of bins. Larger entropy indicates more diffuse spectral evidence. These supplementary indicators are not native calibrated error estimates.

For Welch, $\mathbf { S } _ { \mathrm { t } , \mathrm { k } }$ is the normalized PPG Welch power interpolated onto the 35–220 bpm grid. For ACC-NLMS, it is normalized spectral power of the ACC-filtered PPG residual. TROIKA uses its normalized sparse spectrum, JOSS its cleaned sparse spectrum after motion suppression, SpaMA its surviving candidate spectral mass, and WFPV its exported spectral evidence. For these six estimators, spectra are interpolated onto the 1 bpm grid, floored at $1 0 ^ { - 1 2 }$ and divided by their sum. An all-zero finite spectrum consequently yields uniform weights. WFPV replaces nonfinite spectral rows by uniform evidence before normalization.

For TimePPG, the risk weights are obtained from the mean-centered input PPG: a 1024-point FFT at 32 Hz provides power within 35–220 bpm, which is floored at $1 0 ^ { - 1 2 }$ and normalized by its sum. For Deep PPG, the weights are obtained by applying softmax to the last PPG spectral-magnitude input segment; logarithm arguments are floored at $1 0 ^ { - 3 0 0 }$ for numerical stability. For these two scalar estimators, the adapted heart rate prior is a separate Gaussian centered on the predicted heart rate with a source-calibrated scale. Their baseline risk is therefore not the entropy of that Gaussian prior.

Seven output strategies are evaluated: original estimates, a 20 bpm correction cap, the cap with posterior-gain consistency, uncapped correction, uncapped correction with consistency, ridge rejection, and same-risk rejection. The proposed method is the complete M1–M4 framework with uncapped correction and the posterior-gain consistency check. All configurations were fixed before evaluation.

Ridge rejection predicts the natural logarithm of one plus the original absolute error from 17 source-domain features generated by participant cross-fitting. The ridge regression model uses participant-balanced weights and a fixed regularization strength of 1. At test time, predicted errors rank windows or are compared with a sourcecalibrated threshold. Accepted windows retain their original heart rates. This comparator evaluates selection using a separate, simple error predictor. Same-risk rejection requires no additional predictor. In risk-ranked comparisons, it uses the framework’s risk scores and accepted windows but retains the original heart rates. Its MAE difference from the framework therefore measures numerical correction on the same windows. Thresholds targeting source MAE of 5 bpm are calibrated separately for each strategy; accepted sets can differ under that evaluation mode.

After freezing weights, each configuration compares original risk with learned-error risk from the uncapped variant without consistency checking on three calibration participants. Risk family is selected by participant-macroaveraged MAE at 80% coverage, with ties favoring original risk. The same family is used for four M4 decision variants. Calibration also yields a pooled-quantile threshold for 80% coverage and a threshold targeting source macro-averaged MAE no greater than 5 bpm. The latter requires at least 10% coverage for every calibration participant. Zero output is allowed when no threshold is feasible. Each strategy uses thresholds calibrated for its own outputs; rejection-only strategies do not inherit corrected-output error-budget thresholds. This empirical calibration provides no finite-sample risk guarantee.

## S7. Cross-Dataset Transfer With Fixed Auxiliary Models

All 600 configurations, spanning estimators, five source folds, three seeds, and four trained variants, are applied separately to each external dataset. No additional auxiliary fitting is performed. Source-selected risk families, decision rules, and thresholds remain unchanged; external references do not select seeds, folds, or output variants. Each external participant thus has 15 complete primary source configurations. Their metrics are averaged, rather than averaging 15 predictions into an ensemble estimate.

External estimator caches retain the processing rules of their registered implementations. Supplementary spectral risk for scalar outputs is not described as native estimator probability. Input and history availability masks propagate downstream; interpolation is not used to create eligible predictions. Physiological evidence uses a common sampling representation, 8 s and 24 s contexts, and five aligned historical positions. KID-PPG retains 1000 steps of per-recording adaptation without heart rate labels. Thus, zero auxiliary retraining does not mean tha every core estimator performs no optimization. Whole-record resampling and some estimator preprocessing are retrospective. These batch results do not establish strict streaming deployment of all ten methods; S20.3 separately specifies the smartphone configuration.

Inputs, reference-file hashes, configurations, recordings, and predictions for each external dataset were archived before the final scoring procedure. All datasets had been accessed during earlier development. Fixing the analysis configurations prevents subsequent model selection at this stage, but does not create an independent confirmatory dataset. The experiments therefore assess cross-dataset transfer under fixed configurations.

## S8. Metrics, Paired Decomposition, and Statistics

Let h0 be the original estimate, h the framework output, and y the reference. Absolute error is e(h)=|h−y|, and paired error change is Δe=e(h)−e(h0), in bpm. Coverage is accepted windows with finite predictions divided by all reference-eligible windows; MAE is calculated only over accepted windows. Within the same paired accepted set, beneficial corrections have Δe<−1, small error changes have −1≤Δe≤1, and harmful corrections have Δe>1. These mutually exclusive, exhaustive categories share a denominator. The middle category includes unchanged outputs and small error changes; it is not a retention-action rate. Fig. 3(b) shows their macro-averaged composition, with confidence intervals in Table S3. Severe added error counts accepted windows with Δe>10 relative to all reference eligible windows. It is neither the proportion of absolute errors exceeding 10 bpm nor a measure of clinical harm. With zero output, coverage is zero, while MAE and paired proportions remain undefined.

Evaluation modes include all available outputs without additional rejection, a fixed original-estimator 80% accepted set, independently ranked 80% sets, and fixed source thresholds for 80% coverage or 5 bpm MAE. Risk– coverage curves span 5%–100% in 5-percentage-point steps. The 80% operating point was part of the existing protocol, not optimized for these results. Target accepted counts use each participant’s reference-eligible window count and are rounded down. Missing finite predictions or risk scores can reduce actual coverage below target. Ranking is a retrospective evaluation; fixed-threshold transfer is reported separately.

At 80% target coverage, let A be the framework accepted set and B the set selected by original-estimator risk. Total change, MAE(h,A)−MAE(h0,B), equals the correction term MAE(h,A)−MAE(h0,A) plus the selection term MAE(h0,A)−MAE(h0,B). Same-risk rejection preserves h0 and uses framework risk. Accepted indices were verified to match the framework for every configuration. This decomposition is a paired algebraic identity, not a causal effect estimate.

Participants are the primary aggregation units. DaLiA’s three seeds or the 15 external source configurations are averaged within each participant before equal weighting across participants. Cross-estimator macro-averages also weight the ten implementations equally within a participant. They summarize the registered estimator set; estimators are not treated as independently sampled algorithms, and windows are not pooled across datasets. Recording duration does not alter participant weights. Percentile 95% confidence intervals use 10000 participant bootstrap resamples with replacement and seed 260922. Paired differences are calculated before resampling. Intervals are conditional on fitted models and exclude full-retraining variability and dependence from shared training folds. No confirmatory p-values or multiplicity-adjusted significance claims are reported.

For source MAE5 thresholds, total, zero-output, nonempty, and above-5-bpm configuration counts are reported separately. Participants without eligible references are excluded from these failure denominators. Configuration counts describe runs and are not independent sample sizes. Changes in conditional MAE caused by zero outputs are accompanied by valid-participant counts and counts with complete model replicates. Exploratory leave-one estimator-out analyses assess sensitivity to individual estimators; none is excluded from the primary analysis Analyses use NumPy 2.2.5 and pandas 2.2.3, and all figures derive from actual scores or archived predictions.

## S9. Selection and Display of Signal Examples

Fig. 2 uses GalaxyPPG with Deep PPG as the fixed estimator and the complete framework (FourthUnbounded+agreement). M3/M4 weights are fixed to PPG-DaLiA outer fold 0 and seed 219; Section S1 defines the folds. The same configuration supplies retention, correction, and rejection examples. Across 23 recordings, central windows require eligible references, finite original and framework estimates and risk, and 15 consecutive neighboring windows on each side at 4 s intervals. There were 16,175 centers satisfying common validity and 15,882 satisfying context requirements, from 20 participants.

Retention requires original absolute error no greater than 5 bpm, an unchanged estimate, and acceptance by the source 80% threshold. Correction requires original error above 5 bpm, error reduction of at least 5 bpm, and acceptance. Rejection requires threshold rejection; harmful correction requires acceptance and added error above 1 bpm. Equality uses absolute tolerance 10⁻⁶ bpm. Examples are selected closest to their category median for original error, correction gain, risk, or added error, respectively. Ties are ordered by participant identifier and time. Candidate counts are 2,415, 1,946, 9,740, and 145, respectively.

Selection rules were recorded in FIGURE\_CONTRACT.md before drawing the examples, not preregistered before experimentation. ECG-based event identification serves retrospective explanation and changes neither inference nor thresholds. The four categories do not exhaustively partition all windows, and windows are not independen samples. Fig. 2 retains the selected retention, correction, and rejection events, showing raw PPG/ECG from −8 to 16 s and seven window-end heart rate points. This crop shortens the display without reselecting events or changing predictions or aggregate results. Full context for all four events, including harmful correction, remains in source data and Table S8. Lines break at invalid references, without smoothing or waveform reconstruction. CASE\_AUDIT.json and case\_selection.csv record selection; Fig3\_signals contains raw segments and timestamp mappings.

The Retain, Correct, and Reject waveforms in the Abstract figure come from the corresponding real GalaxyPPG examples in Fig. 2, using Deep PPG. Each shows only raw PPG from the central 0–8 s decision window, preserving amplitudes and samples. No waveform repair or visual selection is applied. The waveforms represent evidence for the corresponding decisions. Correct refers to changing the heart rate estimate, not reconstructing PPG; h₀ and h\* denote the original and candidate heart rates.

## S10. Scope of Prototype Wearable Evaluation

The study includes static, walking, running, and recovery-rest recordings from one participant, scored offline against independent ECG. These recordings are separate from GalaxyPPG and the other public datasets. Fig. 5 shows prototype results, whereas Fig. 2 retains the GalaxyPPG Deep PPG examples. Section S20 details acquisition, input processing, frozen configurations, and evaluation.

The wearable acquires signals in real time and transmits them to a smartphone running a selected estimator and the correction framework. The computer backend analyzes saved recordings using standardized preprocessing and multiple estimators. Preprocessing was informed by earlier recordings from the same participant. Although auxiliary weights and thresholds were not refitted to the new ECG measurements, this remains a retrospective evaluation after development. Its batch statistics do not directly measure the end-to-end performance of the deployed smartphone configuration.

Overlapping windows and repeated source configurations are not independent participants and were not used to estimate population confidence intervals. The prototype recordings were collected from the first author for deployment and technical feasibility evaluation. No institutional ethics approval or exemption determination was obtained for these author self-recordings. These self-recordings alone do not establish generalization across participants or clinical validity.

## S11. M1 Pulse Periodicity and Harmonic Evidence

## S11.1 Structure and Purpose

M1 is a deterministic signal analysis module that extracts fundamental, half-frequency, and double-frequency evidence for each heart rate hypothesis. For example, a 120 bpm hypothesis corresponds to 2 Hz, with additional power checks at 1 and 4 Hz. This representation helps distinguish fundamental and harmonic components. Candidate selection also uses motion, history, and relative-error evidence.

At each 4 s update, M1/M2 extract evidence from the current 8 s signal segment and a continuous 24 s signal segment with the same endpoint. The longer segment covers the union of five consecutive 8 s windows separated by $4 \mathrm { ~ s ~ } ( 8 + 4 \times 4 = 2 4 \mathrm { ~ s ~ } )$ ; it is not an average of five separately computed outputs. This raw-signal context differs from the five-position feature memory in M3. Because each stored feature position may itself use 24 s of signals, five feature positions span 16 s between endpoints and can require 40 s of raw-signal history, as described in S13.

## S11.2 Base Spectra and Autocorrelation

A signal of length � is linearly detrended, multiplied by a Hann window �[�], and transformed using a 4096-point fast Fourier transform (FFT). Power interpolated onto the heart rate grid is denoted by

$$
\begin{array} { r l } & { P _ { L } ( g ) = | F F T _ { 4 0 9 6 } \{ w [ n ] d e t r e n d ( s _ { L } [ n ] ) \} ( g / 6 0 ) | ^ { 2 } } \\ & { \overline { { P } } _ { L } ( g ) = \frac { P _ { L } ( g ) } { m a x _ { u \in \mathcal { G } } P _ { L } ( u ) + \varepsilon } . } \end{array}\tag{S1}
$$

(S2)

Here, $L \in \{ 8 , 2 4 \}$ . The implementation floors the denominator instead of explicitly adding � ; the equation is shorthand. Spectral features are then transformed by

$$
\begin{array} { r } { \ell ( z ) = \frac { \log ( 1 + 1 0 0 z ) } { \log 1 0 1 } } \end{array}\tag{S3}
$$

to compress their dynamic range. ACC base spectra are normalized separately by axis, then combined by taking the maximum across axes at each frequency.

After bandpass filtering and mean removal, normalized PPG autocorrelation is calculated as

$$
R _ { L } ( \tau ) = \frac { \sum _ { n } \tilde { s } _ { L } [ n ] \tilde { s } _ { L } [ n + \tau ] } { \sum _ { n } \tilde { s } _ { L } [ n ] ^ { 2 } } , A _ { L } ( g ) = I n t e r p \Big ( R _ { L } , \frac { 6 4 \times 6 0 } { g } \Big ) .\tag{S4}
$$

The common feature sampling rate is 64 Hz. Normalized autocorrelation measures waveform similarity at the lag corresponding to a heart rate period. The implementation uses an FFT and normalizes by zero-lag correlation.

## S11.3 Harmonic Observation Range

The posterior grid covers 35–220 bpm, while harmonic observations span 17.5–440 bpm. Thus, second harmonics of high heart rate candidates are not truncated at 220 bpm. Original estimates and midpoint candidates derived from them are not constrained to the posterior grid.

## S12. M2 Motion Coherence and Residual Spectral Evidence

## S12.1 Structure and Purpose

M2 is a deterministic spectral module. It uses ACC–PPG coherence to characterize motion-related interference and passes both original and residual spectra to subsequent models. It does not reconstruct time-domain artifact waveforms.

## S12.2 Coherence and Residual Power

Let segmented Fourier coefficients be $P _ { m } ( f )$ and $A _ { m , d } ( f )$ , where $d = 1 , 2 , 3$ indexes ACC axes and � indexes overlapping segments:

$$
\begin{array} { r } { S _ { p p } ( f ) = \frac { 1 } { M } \sum _ { m } | P _ { m } ( f ) | ^ { 2 } , S _ { a a , d } ( f ) = \frac { 1 } { M } \sum _ { m } | A _ { m , d } ( f ) | ^ { 2 } , S _ { p a , d } ( f ) = \frac { 1 } { M } \sum _ { m } P _ { m } ( f ) A _ { m , d } ( f ) ^ { * } . } \end{array}\tag{S5}
$$

$$
\begin{array} { r } { C _ { d } ( f ) = c l i p \left( \frac { \left| S _ { p a , d } ( f ) \right| ^ { 2 } } { S _ { p p } ( f ) S _ { a a , d } ( f ) + 1 0 ^ { - 2 0 } } , 0 , 1 \right) , C ( f ) = m a x _ { d } C _ { d } ( f ) . } \end{array}\tag{S6}
$$

$$
\begin{array} { r } { P _ { \mathrm { r e s } } ( f ) = S _ { p p } ( f ) [ 1 - C ( f ) ] . } \end{array}\tag{S7}
$$

High coherence indicates a frequency-specific association between PPG and at least one acceleration axis. Residual power attenuates the contribution at that frequency. This quantity provides motion-related evidence rather than an artifact-free reference spectrum. Pulse components can also be attenuated when pulse and motion frequencies overlap.

For each $r \in \{ 1 , 1 / 2 , 2 \}$ , four features are extracted at the frequencies corresponding to ��:

$$
\boldsymbol { \psi } _ { L , r } ( g ) = \big [ \ell ( u _ { L , r } ( g ) ) , C _ { L , r } ( g ) , \ell ( v _ { L , r } ( g ) ) , [ 1 - C _ { L , r } ( g ) ] u _ { L , r } ( g ) \big ] ,\tag{S8}
$$

Here, � is PPG power normalized by its maximum over 17.5–440 bpm, and � is residual power normalized by its own band maximum. The fourth feature uses � without logarithmic compression.

The two time scales, three frequency ratios, and four features produce 24 channels. These channels include PPG power and interactions with motion, so M1 and M2 have partly overlapping roles.

## S12.3 Complete Ordering of the 30 Features

Table S1 lists the 30 feature channels in the zero-based order used by the implementation.

Table S1. Physiological evidence channels
<table><tr><td>Channel</td><td>Definition</td></tr><tr><td>0,1</td><td>8 s PPG base spectrum and maximum ACC base spectrum across axes, compressed using e</td></tr><tr><td>2,3</td><td>Corresponding 24 s base spectra</td></tr><tr><td>4,5</td><td>8 s and 24 s PPG autocorrelation</td></tr><tr><td>6-9</td><td>8 s, fundamental  $g \colon \psi$ </td></tr><tr><td>10-13</td><td>8 s, half-frequency g/2: ψ</td></tr><tr><td>14-17</td><td>8 s, double-frequency  $2 g \colon \psi$ </td></tr><tr><td>18-21</td><td>24 s, fundamental  $g \colon \psi$ </td></tr><tr><td>22-25</td><td>24 s, half-frequency  $g / 2 \colon \psi$ </td></tr><tr><td>26-29</td><td>24 s, double-frequency  $2 g \colon \psi$ </td></tr></table>

Thus, each window has $X _ { t } \in \mathbb { R } ^ { 1 8 6 \times 3 0 }$ . The 4096-point full-window FFT for base spectra and the 2048-point segmented FFT for coherence are separate computations, not a single spectral estimator.

## S13. M3 Five-Position Memory and Posterior Network

## S13.1 Network Structure and Tensor Dimensions

The same small multilayer perceptron (MLP) is applied independently to each heart rate hypothesis. There is no convolution across frequencies; hypotheses compete through the final softmax. History uses channel-wise diagonal decay without additional gates or attention.

## S13.2 Exact Historical-State Recurrence

Let $X ^ { ( 0 ) } , \ldots , X ^ { ( 4 ) }$ index five positions from oldest to newest, including the current window. Channel decay is defined by

$$
\begin{array} { r l r } & { \alpha _ { c } = \sigma ( d _ { c } ) , c = 1 , \ldots , 3 0 . } & { \mathrm { ( S 9 ) } } \\ & { S ^ { ( 0 ) } = X ^ { ( 0 ) } , S ^ { ( m ) } = \alpha \odot S ^ { ( m - 1 ) } + ( 1 - \alpha ) \odot X ^ { ( m ) } , m = 1 , \ldots , 4 . } & \end{array}\tag{S10}
$$

The final state is $S _ { t } = S ^ { ( 4 ) }$ . For a complete five-position history, the expanded recurrence is

$$
\begin{array} { c } { S _ { t } = \alpha ^ { 4 } \odot X _ { t - 4 } } \\ { + ( 1 - \alpha ) \odot ( \alpha ^ { 3 } \odot X _ { t - 3 } + \alpha ^ { 2 } \odot X _ { t - 2 } } \\ { + \alpha \odot X _ { t - 1 } + X _ { t } ) . } \end{array}\tag{S11}
$$

Weights sum to 1 within each channel. Initially, $d _ { c } = \log 4$ , so $\alpha _ { c } = 0 . 8$ , giving five-position weights (0.4096, 0.1024, 0.128, 0.16, 0.2). The oldest position includes the initialization contribution; recurrence is limited to these five positions.

The historical state is recomputed from the most recent five positions. At the start of a recording, the earliest available features are repeated, and input validity is checked at every position. Positions are 4 s apart, spanning 16 s between endpoints. Combined with the 24 s evidence window at each position, the earliest signal is 40 s before the current endpoint. The smartphone retains the current 24 s of raw signals and features for five positions.

## S13.3 Energy Adjustment and Posterior

$$
z _ { t } ( k ) = [ X _ { t } ( k ) , S _ { t } ( k ) , X _ { t } ( k ) - S _ { t } ( k ) ] \in \mathbb { R } ^ { 9 0 } , \qquad \quad \mathrm { ~ \normalfont ~ ( S 1 2 ) ~ }
$$

$$
u _ { t } ( k ) = W _ { 2 } S i L U ( W _ { 1 } z _ { t } ( k ) + b _ { 1 } ) + b _ { 2 } , E _ { t } ( k ) = 6 \mathrm { t a n h } ( u _ { t } ( k ) / 6 ) ,\tag{S13}
$$

$$
\begin{array} { r } { T = 0 . 5 + 3 \sigma ( \theta ) , q _ { t } ( k ) = \frac { \exp \{ \log [ m a x ( \overline { p } _ { t } ( k ) , 1 0 ^ { - 1 0 } ) ] / T + E _ { t } ( k ) \} } { \sum _ { l } \exp \{ \log [ m a x ( \overline { p } _ { t } ( l ) , 1 0 ^ { - 1 0 } ) ] / T + E _ { t } ( l ) \} } . } \end{array}\tag{S14}
$$

$\overline { { p } }$ is the mean over prior channels. The evaluated input has only one channel, so ${ \overline { { p } } } = p$ . Each instance uses a single core estimator.

$T \in ( 0 . 5 , 3 . 5 )$ controls distribution sharpness, while $E \in ( - 6 , 6 )$ adjusts support for each heart rate hypothesis. Equivalently,

$$
q _ { t } ( k ) \propto p _ { t } ( k ) ^ { 1 / T } \mathrm { e x p } E _ { t } ( k ) .\tag{S15}
$$

Initialization uses $\theta = \log 0 . 2$ , giving � = 1. The final linear layer is initialized to zero, so the initial network approximately preserves the prior. The term posterior describes this distribution but does not imply guaranteed statistical calibration.

The parameter count is $( 9 0 \times 2 4 + 2 4 ) + ( 2 4 + 1 ) + 3 0 + 1 = 2 2 4 0$ . Memory decay, the MLP, and temperature are learned jointly.

## S13.4 Construction of Nine Candidates

Let $Q _ { \beta }$ be the first grid value at which the discrete posterior CDF reaches $\beta .$ Candidates follow the implementation order:

$$
\begin{array} { r } { \mathbf { a } _ { t } = [ h _ { 0 , t } , \sum _ { k } q _ { t } ( k ) g _ { k } , g _ { a r g m a x _ { k } q _ { t } ( k ) } , } \\ { Q _ { . 1 } , Q _ { . 2 5 } , Q _ { . 5 } , Q _ { . 7 5 } , Q _ { . 9 } , ( h _ { 0 , t } + Q _ { . 5 } ) / 2 ] . } \end{array}\tag{S16}
$$

These are the original estimate, mean, mode, five percentiles, and the midpoint between the original estimate and median. Duplicates are allowed. The original estimate occupies position 0 to give retention priority in ties.

## S14. M4 Consistent Gain–Harm Decisions

## S14.1 Posterior Expectations and Learned Error Predictions

The posterior provides each candidate’s expected absolute error, expected added error, and neighborhood support:

$$
\begin{array} { r l r } & { e _ { j } ^ { q } = \sum _ { k } q ( k ) | a _ { j } - g _ { k } | , } & { { \mathrm { ( S 1 7 ) } } } \\ & { h _ { j } ^ { q } = \sum _ { k } q ( k ) m a x \{ | a _ { j } - g _ { k } | - | h _ { 0 } - g _ { k } | , 0 \} , r _ { j } ^ { q } = \sum _ { k } q ( k ) { \bf 1 } ( | a _ { j } - g _ { k } | \leq 5 ) . } & \end{array}\tag{S18}
$$

Posterior expected net gain is $g _ { j } ^ { q } = e _ { 0 } ^ { q } - e _ { j } ^ { q }$ . These quantities are calculated under q. M4 separately learns $\widehat { G } _ { j } , \widehat { H } _ { j } , \widehat { E } _ { j }$ using actual source-participant errors as supervision. Eligibility requires both sufficient learned net gain and posterior expected error below that of the original estimate.

## S14.2 Complete 106-Dimensional Feature Vector

Let $\delta _ { j } = a _ { j } - h _ { 0 }$ , with posterior mean $\mu _ { q }$ , standard deviation $\sigma _ { q }$ , and entropy $\mathcal { H } _ { q }$ . The first 16 features are

$$
\begin{array} { r l } & { f _ { j } ^ { 1 6 } = [ a _ { j } / 2 0 0 , h _ { 0 } / 2 0 0 , \delta _ { j } / 4 0 , | \delta _ { j } | / 4 0 , } \\ & { \quad \quad \quad e _ { j } ^ { q } / 2 0 , h _ { j } ^ { q } / 2 0 , r _ { j } ^ { q } , e _ { 0 } ^ { q } / 2 0 , r _ { 0 } ^ { q } , } \\ & { \quad \quad ( e _ { 0 } ^ { q } - e _ { j } ^ { q } ) / 2 0 , r _ { j } ^ { q } - r _ { 0 } ^ { q } , \sigma _ { q } / 4 0 , \mathcal { H } _ { q } / 6 , } \\ & { \quad \quad \mu _ { q } / 2 0 0 , \mathbf { 1 } ( \delta _ { j } < 0 ) , \mathbf { 1 } ( | \delta _ { j } | > 2 0 ) ] . } \end{array}\tag{S19}
$$

For candidate $a _ { j }$ , the nearest grid position is $k _ { j } = a r g m i n _ { k } | a _ { j } - g _ { k } |$ . The remaining 90 features are

$$
f _ { j } ^ { 9 0 } = [ X _ { t } ( k _ { j } ) , S _ { t } ( k _ { j } ) , X _ { t } ( k _ { j } ) - X _ { t } ( k _ { 0 } ) ] .\tag{S20}
$$

The final vector is $f _ { j } = [ f _ { j } ^ { 1 6 } , f _ { j } ^ { 9 0 } ] \in \mathbb { R } ^ { 1 0 6 }$ . Its third 30-dimensional block is the current-evidence difference between candidate and original estimate. In contrast, M3 uses a temporal difference between current and historical evidence. For out-of-grid candidates, evidence comes from the nearest endpoint, but the candidate heart rate remains unchanged.

Features are standardized using source-training statistics:

$$
\begin{array} { r } { \tilde { { f } } _ { j , c } = c l i p \left( \frac { f _ { j , c } - \mu _ { c } } { m a x ( \sigma _ { c } , 0 . 0 5 ) } , - 5 , 5 \right) . } \end{array}\tag{S21}
$$

Means and standard deviations are calculated across training windows and all nine candidates, then reused unchanged during external inference.

## S14.3 Two-Branch Network

Each branch consists of one affine transformation. Softmax, softplus, and constrained decoding introduce the nonlinear operations. The second branch predicts a single base error for each window, shared by all nine candidates.

The parameter count is $( 1 0 6 \times 3 + 3 ) + ( 1 0 6 + 1 ) = 4 2 8 .$

## S14.4 Exact Consistent Decoding

$$
\pi _ { j } = s o f t m a x ( \tilde { W _ { h } f _ { j } } + b _ { h } ) , b = 2 0 s o f t p l u s ( \tilde { w _ { r } \tilde { f _ { 0 } } } + b _ { r } ) , d _ { j } = | a _ { j } - h _ { 0 } | .\tag{S22}
$$

Let $\pi _ { j , 1 } , \pi _ { j , 2 } , \pi _ { j , 3 }$ denote the three allocations. The implementation decodes them as

$$
\widehat { B } _ { j } = m i n ( d _ { j } \pi _ { j , 1 } , b ) ,\tag{S23}
$$

$$
\widehat { H } _ { j } = m a x \big ( d _ { j } \pi _ { j , 2 } , m a x ( d _ { j } - 2 b + \widehat { B } _ { j } , 0 ) \big ) ,\tag{S24}
$$

$$
\widehat { G } _ { j } = \widehat { B } _ { j } - \widehat { H } _ { j } , \widehat { E } _ { j } = b - \widehat { G } _ { j } .\tag{S25}
$$

The three logits are allocation parameters, not independent retention, correction, and rejection probabilities. The third component participates through the softmax sum constraint but is not returned directly. $\widehat { B }$ is an internal beneficial component; the returned gain is net gain after subtracting added error.

The decoding satisfies

$$
\widehat { E } _ { j } \geq 0 , | \widehat { G } _ { j } | \leq d _ { j } , b + \widehat { E } _ { j } \geq d _ { j } , \widehat { E } _ { j } - b = - \widehat { G } _ { j } .\tag{S26}
$$

because absolute-error distances obey the triangle inequality:

$$
| | a _ { j } - y | - | h _ { 0 } - y | | \leq | a _ { j } - h _ { 0 } | , | a _ { j } - y | + | h _ { 0 } - y | \geq | a _ { j } - h _ { 0 } | .\tag{S27}
$$

Embedding these geometric relationships prevents inconsistent predictions, such as claiming a 30 bpm improvement after changing an estimate by only 2 bpm. For the original estimate $d _ { 0 } = 0$ , the identities $\widehat { B } _ { 0 } = \widehat { H } _ { 0 } =$ $\widehat { G } _ { 0 } = 0 \mathrm { ~ a n d ~ } \widehat { E } _ { 0 } = b$ necessarily hold.

Decoding does not enforce $\widehat { H } _ { j } = m a x ( - \widehat { G } _ { j } , 0 )$ . Beneficial and harmful components can both be positive, with their difference giving net gain. This joint prediction parameterization differs from the definition of actual added error for an individual training sample.

## S14.5 Primary Decision Rule

For $j > 0$ , eligibility requires

$$
\begin{array} { r } { I _ { j } = \pmb { 1 } \Big ( \widehat { G } _ { j } \geq 1 \Big ) \pmb { 1 } \Big ( \widehat { H } _ { j } \leq 2 \Big ) \pmb { 1 } \big ( e _ { 0 } ^ { q } - e _ { j } ^ { q } > 0 \big ) . } \end{array}\tag{S28}
$$

$$
U _ { 0 } = 0 , U _ { j } = \left\{ \begin{array} { c l } { { \widehat { G } _ { j } - 2 \widehat { H } _ { j } , } } & { { I _ { j } = 1 , } } \\ { { - \infty , } } & { { I _ { j } = 0 , j ^ { * } = a r g m a x _ { j = 0 , \ldots , 8 } U _ { j } . ( \mathrm { S } 2 9 ) } } \end{array} \right.
$$

$$
\widehat { h } = a _ { j ^ { * } } .\tag{S30}
$$

The primary configuration, identified as FourthUnbounded+agreement in the analysis files, applies no 20 bpm correction cap. The maximization returns the first candidate attaining the highest utility. Because the original estimate is listed first, a tie at zero utility retains that estimate. A replacement must therefore satisfy all three eligibility conditions and have positive utility.

For illustration only, consider an original estimate of 140 bpm and a candidate of 100 bpm; these are not experimental observations. With $\widehat { G } = 8 , \widehat { H } = 1 , g ^ { q } = 5$ , utility is 6 and the candidate can compete. With ${ \widehat { G } } =$ $2 , \widehat { H } = 1 . 5 , g ^ { q } = 5$ , all three conditions hold but utility is −1, so the original estimate is retained. The added-error limit of 2 bpm applies to a prediction; actual error can still increase by more than 2 bpm.

## S15. Independent Output Risk Gate

M4 selects $\hat { h } ,$ and the risk gate then determines acceptance. Each configuration compares original-estimator risk $r _ { 0 }$ with learned candidate error $\widehat { E } _ { j ^ { \ast } }$ on three source-calibration participants. Selection uses participant-macro-averaged MAE80 for the uncapped variant without agreement checking, with ties favoring original risk. The selected family is then used by all four M4 decision strategies.

The rejection risk for the proposed method is therefore

$$
\begin{array} { r } { r _ { t } = \left\{ \begin{array} { c c } { r _ { 0 , t } , } & { \mathrm { r a w ~ r i s k ~ s e l e c t e d } , } \\ { \widehat { E } _ { t , j ^ { * } } , } & { \mathrm { l e a r n e d ~ r i s k ~ s e l e c t e d } . } \end{array} \right. \qquad \mathrm { ( S 3 1 ) } } \end{array}
$$

The base error � is not itself the final threshold, and not every configuration rejects according to learned error.   
When the selected family is raw, original-estimator risk remains in use even after heart rate correction.

For a frozen threshold $\tau ,$

$$
A c c e p t _ { t } = \mathbf { 1 } ( \hat { h } _ { t } \operatorname { f i n i t e } ) \mathbf { 1 } ( r _ { t } \operatorname { f i n i t e } ) \mathbf { 1 } ( r _ { t } \leq \tau ) .\tag{S32}
$$

For the source 80% threshold, the implementation calculates $m = \left\lfloor 0 . 8 N _ { \mathrm { r e f } } \right\rfloor$ from the complete reference-eligible count $N _ { \mathrm { r e f } }$ , sorts finite available risk scores, and selects value ��� $( m , N _ { \mathrm { a v a i l } } )$ . If no position is available, a rejection-sentinel threshold is used. Ties, missing outputs, and distribution shifts can move actual coverage away from 80%. A separate source MAE≤5 bpm rule requires at least 10% coverage per calibration participant and allows zero output if no solution is feasible.

When external auxiliary history is unavailable, restoration to the full timeline may retain the core estimate but set auxiliary risk to NaN. This does not pass the rejection gate. All-available-output evaluation and fixed-risk-gate evaluation therefore use different output sets.

## S16. Two-Stage Training and Losses

## S16.1 M3 Distribution Target

Reference heart rate is converted to a Gaussian soft target with a standard deviation of 3 bpm:

$$
\begin{array} { r } { \tilde { y } _ { k } = \frac { \exp [ - ( g _ { k } - y ) ^ { 2 } / ( 2 \cdot 3 ^ { 2 } ) ] } { \sum _ { l } \exp [ - ( g _ { l } - y ) ^ { 2 } / ( 2 \cdot 3 ^ { 2 } ) ] } . } \end{array}\tag{S33}
$$

Let $\begin{array} { r } { F _ { q } ( k ) = \sum _ { l \leq k } q _ { l } } \end{array}$ and $F _ { y } ( k ) = \sum _ { l \le k } \tilde { y } _ { l }$ . The implemented loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p o s t } } = \frac { 1 } { B } \displaystyle \sum _ { t } \Bigl [ - \sum _ { k } \tilde { y } _ { t , k } \log q _ { t , k } + \frac { 0 . 2 } { 2 0 } \sum _ { k } ( F _ { q _ { t } } ( k ) - F _ { y _ { t } } ( k ) ) ^ { 2 } \Bigr ] . } \end{array}\tag{S34}
$$

The first term favors probability near the reference, while the cumulative-distribution term incorporates distance on the heart rate grid. This is not a raw-waveform reconstruction loss.

M3 is trained for 2,000 steps using AdamW, a learning rate of 0.002, weight decay of 0.03, and a batch size of 96. The gradient norm is clipped at 2, and sampling is balanced across participants. With probability 0.1, each sample’s prior channel is replaced by the prior from another sampled window to reduce reliance on the prior.

## S16.2 M4 Supervision

For the original estimate and candidates in the same window,

$$
E _ { t , j } ^ { * } = | a _ { t , j } - y _ { t } | , G _ { t , j } ^ { * } = E _ { t , 0 } ^ { * } - E _ { t , j } ^ { * } , H _ { t , j } ^ { * } = m a x ( - G _ { t , j } ^ { * } , 0 ) . ( \mathrm { S } 3 5 )
$$

$$
\mathrm { L e t } \ \hat { v } = ( \hat { G } , \widehat { H } , \hat { E } ) / 2 0 \ \mathrm { a n d } \ v ^ { * } = ( G ^ { * } , H ^ { * } , E ^ { * } ) / 2 0 . \ \mathrm { T h e n }
$$

$$
\mathcal { L } _ { \mathrm { d e c i s i o n } } = m e a n [ \quad S m o o t h L 1 ( \widehat { \boldsymbol { v } } - \boldsymbol { v } ^ { * } ) ] + 2 m e a n \big [ m a x ( \boldsymbol { v } _ { H } ^ { * } - \widehat { \boldsymbol { v } } _ { H } , 0 ) ^ { 2 } \big ] .\tag{S36}
$$

PyTorch’s default SmoothL1 is $\rho ( u ) = u ^ { 2 } / 2$ when $| u | < 1$ , and $| u | - 1 / 2$ otherwise. The first term averages over batches, candidates, and three outputs; the second averages over batches and candidates. The additional term penalizes underestimation of actual added error.

The weights and biases of both M4 linear layers are initialized to zero. Training uses AdamW with a learning rate of 0.003, weight decay of 0.1, batch size of 256, and participant-balanced sampling. The model is trained for 2,000 steps. M3 posteriors are computed and fixed beforehand; the M4 loss does not update M3 or the core estimator.

## S16.3 Rationale for Participant Cross-Fitting

Training M4 on predictions from the same participants used to fit M3 could underestimate replacement risk. Crossfitting instead generates each participant’s posteriors using an M3 model trained without that participant. Historical decay parameters are taken from the corresponding M3 model.

Each estimator, outer fold, seed, and trained variant requires three internal M3 models, one final M3 model, and one M4 model. Ten estimators, five folds, three seeds, and four variants yield 600 configurations, comprising 2,400 M3 fits and 600 M4 fits. Each external participant is evaluated with 15 source configurations. Their metrics are averaged; their predictions are not combined into an ensemble.

The weights from training step 2,000 are used without checkpoint selection on the evaluation set.

## S17. Decoding Constraints and Information Limits

## S17.1 Proof of Geometric Consistency

Omitting window and candidate indices, let $d = | a - h _ { 0 } | \geq 0$ and $b \geq 0$ . Softmax ensures $\pi _ { i } \geq 0$ and $\textstyle \sum _ { i } \pi _ { i } = 1$ From $B = m i n ( d \pi _ { 1 } , b )$ , it follows that $0 \leq B \leq$ ���(�, �). Therefore,

$$
d \pi _ { 2 } \leq d - d \pi _ { 1 } \leq d - B , d - 2 b + B \leq d - B . \eqno ( \mathrm { S } 3 7 )
$$

Because $0 \leq d - B$ , maximum-based decoding gives $0 \leq H \leq d - B$ . Hence,

$$
- d \leq B - H = G \leq d , E = b - B + H \geq 0 .\tag{S38}
$$

Using $H \geq d - 2 b + B$ , we further obtain

$$
b + E = 2 b - B + H \geq d .\tag{S39}
$$

For the original estimate, $d = 0$ , so $B = H = G = 0$ and $E = b$ . These relationships constrain the algebra of predicted quantities. They do not equate predictions with actual errors or guarantee calibration or a true harm bound. The harm branch is not forced to equal ��� $( - G , 0 )$ ; it receives separate supervision from actual added error.

## S17.2 Why Estimators Using Spectral Information May Still Benefit

Deterministic spectral, harmonic, and difference features add no information to the raw signals, but they can make useful structure more accessible to finite-capacity models. The training objective also changes from a single heart rate or distribution to paired gain, added error, and absolute error relative to a fixed estimate. Candidate selection then applies an explicit cost. Spectral processing in a core estimator is therefore compatible with additional correction gains. Those gains cannot be attributed solely to recomputing spectra.

Complementarity is not guaranteed. Let ℐ represent all information shared by the estimator and auxiliary layer. If the estimator is already Bayes-optimal for absolute error in the same target domain,

$$
h ^ { * } ( \mathcal { I } ) \in a r g m i n _ { h } \mathbb { E } [ | h - Y | \mid \mathcal { I } ] ,\tag{S40}
$$

deterministic post-processing using the same information cannot further reduce conditional expected absolute error. Practical finite models need not satisfy this ideal condition, and the study also considers trade-offs involving harmful correction and coverage. This statement defines a theoretical boundary; it does not establish Bayes optimality for any evaluated estimator.

The experiments show average error reductions, gains from correction on the same accepted windows, and effects of the specified ablations. They do not isolate the separate contributions of representation, paired supervision, and constrained decoding. Controlled comparisons would be needed to attribute these gains to individual mechanisms.

## S18. Roles and Interfaces in the Agent Architecture

The agent architecture organizes signal analysis, a limited historical state, and output decisions around a fixed heart rate estimator. Each of the ten estimators is evaluated separately in the core-model role. The system uses neither a language model nor collaboration among estimators. Auxiliary parameters are fitted separately for each estimator and remain fixed during external evaluation.

Table S2. Module interfaces and outputs
<table><tr><td>Role</td><td>Implementation</td><td>Output or function</td></tr><tr><td>Core model</td><td>One registered heart rate estimator</td><td>Original estimate, adapted prior, and original risk</td></tr><tr><td>Tools</td><td>M1 pulse analysis and M2 motion analysis</td><td>30-dimensional evidence at 186 heart rate hypotheses</td></tr><tr><td>Memory</td><td>M3 limited history and posterior network</td><td>Five-position evidence state, posterior, and nine candidates</td></tr><tr><td>Decision</td><td>M4 evaluation and explicit selection rules</td><td>Shared original error, candidate gains and harms, and estimate selection</td></tr><tr><td>Delivery gate</td><td>Independent risk gate</td><td>Release or reject the selected estimate</td></tr></table>

Tools execute in a fixed sequence, and some M1/M2 interaction channels share PPG and ACC information. M3 states derive from input evidence, not released heart rates, ECG feedback, or generated text. M4 directly reads tool and memory features to compare candidates with the original estimate. Rejection follows selection, so both retained and corrected estimates can subsequently be rejected. Training uses no reinforcement learning. M4 loss does not propagate to M3 or the core estimator, and its three-way softmax does not represent action probabilities.

Fig. 1 shows the overall information flow through the fixed core estimator and M1–M4. Table S2 summarizes the module interfaces. M1/M2 use 8 s and 24 s contexts at 64 Hz and have no trainable weights. M3 uses five positions, including the current window, and shares network parameters across 186 heart rate hypotheses. M4 has 428 trainable parameters, with risk checking after candidate selection. The architecture includes no iterative feedback, dynamic tool routing, or additional training stage. Motion and history ablations support some tool and memory contributions. Prototype recordings also include an M4-removal comparison with the same candidates (S20; Fig. S16). Independent ablation of all harm constraints and online-feedback experiments remain unvalidated.

## S19. Mathematical Definitions of Outputs and Metrics

This section defines final outputs and evaluation metrics. The acceptance indicator $\mathbf { A } _ { \mathrm { t } }$ is defined in main-text (6), and the selected heart rate in (5). The final output is

$$
h _ { t } ^ { \mathrm { o u t } } = \left\{ \begin{array} { c c } { \widehat { h } _ { t } , } & { A _ { t } = 1 , } \\ { \operatorname* { m i s s i n g } , } & { A _ { t } = 0 . } \end{array} \right.\tag{S41}
$$

When ${ \mathrm { A } } _ { \mathrm { t } } { = } 1$ , the selected estimate is released. When ${ \bf A } _ { \mathrm { t } } { = } 0$ , output is missing and no heart rate is reported. Missing outputs are excluded from the accepted set but remain reference-eligible. Let $\mathcal { R }$ contain all reference-eligible windows, � the accepted windows with finite outputs, and $\mathcal { A } _ { 0 }$ those also having finite original estimates. MAE, coverage, and harmful correction rate are

$$
\begin{array} { r } { M A E ( \hat { h } , \mathcal { A } ) = \frac { 1 } { \vert \mathcal { A } \vert } \sum _ { t \in \mathcal { A } } \vert \hat { h } _ { t } - y _ { t } \vert , C o v e r a g e = \frac { \vert \mathcal { A } \vert } { \vert \mathcal { R } \vert } . } \end{array}\tag{S42}
$$

$$
\begin{array} { r } { H a r m R a t e = \frac { 1 } { | \mathcal { A } _ { 0 } | } \sum _ { t \in \mathcal { A } _ { 0 } } \mathbf { 1 } \{ | \hat { h } _ { t } - y _ { t } | - | h _ { 0 , t } - y _ { t } | > 1 \} . ( \mathrm { S } 4 3 ) } \end{array}
$$

MAE is measured in bpm. Coverage uses all reference-eligible windows as its denominator, whereas harmful correction uses paired accepted windows. The threshold in (S43) is an absolute-error increase exceeding 1 bpm, not a heart rate change exceeding 1 bpm. Severe added error counts accepted windows with an error increase exceeding 10 bpm, divided by all reference-eligible windows. With an empty accepted set, MAE remains undefined and coverage is zero. Section S8 describes aggregation, paired decomposition, and statistics.

## S20. Single-Participant Acquisition and Offline Evaluation

## S20.1 Recording Audit and Reference Timeline

On October 3, 2026, participant PILOT01 completed separate static, walk, run, and rest recordings; rest denotes recovery after exercise. Raw wrist logs contain dual-wavelength PPG, ACC, gyroscope signals, frame counts, and reception times. Independent upper-sternum ECG provides the reference. Fourteen raw files comprise wrist logs, ECG logs, and compressed exports for the four recordings, plus an earlier communication-test log pair. The earlier test is excluded from scoring. Archive integrity and decoded ECG consistency were audited; duplicate synchronization files are not independent recordings. Table S11 reports durations, reception rates, and eligiblewindow counts.

Repeated ECG timestamps are reconstructed from batch-end reception times at 512 Hz. Reference heart rates retain the XQRS/GQRS dual-detector quality checks and reference\_qualified flags. Window-level detector outputs, valid RR fractions, sampling coverage, and exclusions are stored in ecg\_reference\_windows.csv. PPG/ACC use the same recording clock, without an ECG-error-minimizing delay search. Reconstruction does not validate fixed interdevice transmission delays. Evaluation uses 8 s windows with a 4 s stride, requiring endpoints at or after 40 s and eligible references. There are 324 eligible windows, with dependence due to overlap.

PPG reception gaps exceeding 0.4 s split recordings into blocks, with 1 s boundary guards at both ends. Memory does not cross recordings or gaps. Auxiliary evidence needs 24 s of context, and five complete historical positions need approximately 40 s. When auxiliary quantities are unavailable but the core estimate is finite, all-availableoutput evaluation retains the original estimate. The matched\_aux\_full and full\_history\_only analyses separately restrict auxiliary availability and complete history; both are included in configuration-level sensitivity results.

## S20.2 Dual-Channel Preprocessing and Input Units

Infrared and red channels are processed separately. Each block undergoes linear detrending, root-mean-square (RMS) normalization, and third-order zero-phase bandpass filtering at 0.5–5 Hz. Triaxial ACC supplies motion references for continuous normalized least mean squares (NLMS) filtering, with 12 taps per axis, step size 0.1, and leakage 10⁻⁴. The residual is used when the motion-intensity ratio over up to 24 s exceeds 0.1; otherwise, the bandpass output is retained. Block baseline and RMS are then adjusted. Processing uses a 12.5 Hz grid before adaptation to source-model sampling rates; auxiliary evidence uses 64 Hz. Actual reception rates of 9.67–11.19 Hz differ from this grid, and upsampling cannot recover unrecorded bandwidth. Zero-phase filtering and whole-block normalization make the pipeline retrospective. Preprocessing was informed by earlier exploration of the same participant, so this is post-development retrospective evaluation.

Infrared is the prespecified primary channel, with red analyzed separately. ECG does not select channels by window, and the ten estimators are not fused. Each estimator and its framework receive identical processed signals. TimePPG input adaptation multiplies PPG by the source-training median amplitude 45.349861 and divides ACC by 2142.539615, the median magnitude during the first 60 s of the static recording. These constants apply to both channels and all four states, without ECG-error tuning. This is numerical scale adaptation, not physical calibration. Unadapted outputs are retained separately for diagnosis. Fig. S18 shows fixed pre- and post-processing segments; regular-looking waveforms do not establish complete recovery of physiological components.

## S20.3 Frozen Models and Figure Configurations

Each registered estimator reuses five source folds and seeds 219, 220, and 221, giving 15 frozen auxiliary configurations. Infrared includes full, no\_harmonic, no\_motion, and no\_memory; red includes full only. This gives 750 estimator/channel/configuration combinations corresponding to 600 unique auxiliary parameter sets. FROZEN\_MODEL\_BINDINGS.json and file hashes bind checkpoints, M4 parameters, and calibration thresholds. Metrics are calculated for each configuration before arithmetic averaging; predictions are not combined into an ensemble. KID-PPG retains unlabeled filter fitting for 1,000 steps per recording/channel, totaling 8,000 steps, without updating the heart rate network. Local implementations and input adapters follow the source protocol and do not represent the best official implementations.

Fig. 5(d) fixes Deep PPG, infrared input, source fold 0, seed 219, and the unchanged source80 threshold. It shows all evaluation windows from 40 s onward in static, walking, and running recordings. Rejected outputs remain missing; crosses mark internal candidates only. For Fig. S11, TimePPG is the first estimator in the prespecified order with all three actions. Using the same fold and seed, each action is illustrated by its first chronologica window, without selection for favorable ECG errors. The deployed research version of PulseLink 0.3.0 uses Welch with the full framework, source fold 0, seed 219, and a source 80% coverage threshold of 9.860092 for real-time analysis. The backend evaluates all ten methods and source configurations, calculates paired statistics, and generates figures. Fig. 5(c) shows the smartphone interface and workflow. Smartphone configuration and backend multi-method results are recorded separately; ten predictions are not fused into a phone output.

## S20.4 Comparison Sets and Aggregation

The available\_full mode applies no additional rejection and compares original and corrected outputs on the same available windows within each estimator. Most methods have 307/324 windows; BeliefPPG and Deep PPG have 289/324, and KID-PPG has 304/324. The common\_ten\_full mode uses 289 infrared windows available to all ten estimators. Eligible windows from all four states are pooled to calculate each estimator’s MAE before equa weighting across estimators. State-specific results are calculated separately. Cross-estimator macro-averages describe neither a single deployed estimator nor population effects.

The own\_rank80 mode ranks risk within target recordings and accepts at most 80% of reference-eligible windows, without ranking ECG errors. Each method accepts 259/324 windows overall. Same-risk rejection uses the ful framework’s accepted set but preserves original heart rates. Correction gain compares original and corrected errors on that set. Selection gain compares original-estimate errors on sets selected by original and framework risk. Their sum equals total gain within each configuration. The source80 and source\_mae5 modes reuse source-calibrated thresholds without target-ECG reselection. Source\_mae5 strategies are calibrated independently, so their accepted sets need not match.

A harmful correction increases absolute error by more than 1 bpm relative to the original estimate. Its rate uses paired output windows as the denominator. Initially accurate estimates have an original absolute error no greater than 5 bpm; deterioration rates are calculated within that subset. The severe added-error proportion counts error increases exceeding 10 bpm relative to all reference-eligible windows. Overall summaries weight estimatorspecific rates equally. For Fig. S20(b), method–window counts are pooled within each configuration before calculating rates. This avoids undefined rates for individual state–method pairs with no initially accurate windows, but differs from averaging estimator-specific rates. A zero denominator is reported as NA.

Error bars and shading show the minimum–maximum range across 15 frozen source configurations, not participant-bootstrap confidence intervals. Folds, seeds, wavelengths, and correlated windows are not additional participants. MAE is undefined for zero-output configurations. Source-threshold coverage averages include all 15 configurations, while MAE averages include only nonempty configurations; zero-output counts are reported separately.

## S20.5 Results and Interpretation

Without additional rejection, paired macro-averaged MAE decreased from 9.087 to 4.445 bpm. Excluding TROIKA gave a decrease from 7.102 to 4.454 bpm. Restricting all ten estimators to 289 common windows gave a decrease from 9.235 to 4.332 bpm. Across ten estimators, MAE decreased from 14.811 to 4.241 bpm during static recording, from 8.544 to 5.210 bpm during walking, and from 9.997 to 5.968 bpm during running. The large original TROIKA errors contributed to the static result; excluding TROIKA gave a decrease from 7.716 to 4.148 bpm. Table S12 reports estimator-specific results. Fig. S15 presents both channels and all four states; Fig. S19 presents state-specific sensitivity and matched-coverage comparisons.

At 80% target coverage, macro-averaged MAE was 8.598 bpm for original-risk selection, 7.492 bpm for same-risk rejection, 3.298 bpm for the framework, and 6.826 bpm for ridge rejection. Correction and selection contributed approximately 4.195 and 1.105 bpm, respectively, to the total reduction of 5.300 bpm. The decomposition compares numerical corrections on fixed windows and selection effects on unchanged estimates. It is descriptive and does not identify independent causal effects of individual modules.

Adding M4 reduced the harmful correction rate from 2.175% to 0.416% and the deterioration rate among initially accurate estimates from 2.644% to 0.516%. However, MAE increased from 2.574 to 4.445 bpm. These comparisons used all available outputs without additional rejection. The effect varied by estimator: KID-PPG had a harmful correction rate of 0.395% with M4 and 0.132% without it. After pooling initially accurate method–window counts by state, deterioration rates decreased from 1.040%, 6.000%, and 3.022% to 0.776%, 0.324%, and 0.954% for static, walking, and running recordings, respectively. Some individual estimators showed changes in the opposite direction (Fig. S20). Thus, more conservative correction did not improve every metric or estimator.

After transfer of thresholds targeting source MAE of 5 bpm, mean MAE among configurations with outputs was 8.871 bpm for KID-PPG and 8.244 bpm for TimePPG. ACC-NLMS, JOSS, and SpaMA produced no outputs in 7/15, 6/15, and 3/15 configurations, respectively. Source-domain calibration therefore did not guarantee the target error on prototype recordings. This retrospective single-participant evaluation is further limited by noncausal preprocessing, reconstructed timestamps, and reception bandwidth. Independent evaluation requires a fixed protocol and causal online processing.

## S21. Estimator Composition and TROIKA Sensitivity

All ten registered implementations remain in the primary analysis. Leave-one-out analyses test whether unusually large original errors from one implementation dominate the macro-average; they do not selectively remove unfavorable results. The local TROIKA variant runs at 25 Hz with custom SSA pairing and spectral tracking and is not equivalent to the authors’ implementation. Its median original output reaches 292.68 bpm on Galaxy and Wild, leaving substantial scope for correction. Excluding it, the other nine estimators still reduce macro-averaged MAE by 3.41, 9.17, 5.83, and 3.08 bpm on DaLiA, Wrist, Galaxy, and Wild. Fig. S21(a) compares all-ten and no-TROIKA gains; Fig. S7 reports the complete leave-one-out analysis using the same paired calculations and participant-bootstrap intervals.

For prototype infrared recordings, ten-estimator macro-averaged MAE changes from 9.087 to 4.445 bpm, compared with 7.102 to 4.454 bpm after excluding TROIKA [Fig. S21(b)]. Each estimator retains its paired available windows without additional rejection. The common 289-window result, 9.235 to 4.332 bpm, instead tests differences in window availability and is a distinct sensitivity condition. Fig. S19 reports state-specific analyses. Gains are therefore not solely attributable to TROIKA, but macro-averages must specify which implementations they include.

## SUPPLEMENTARY TABLES

TABLE S3  
ERROR, CORRECTION BENEFITS, AND HARMS WITHOUT ADDITIONAL REJECTION WITH 95% CONFIDENCE INTERVALS
<table><tr><td>Metric</td><td>DaLiA</td><td>Wrist</td><td>Galaxy</td><td>Wild</td></tr><tr><td>Original MAE (bpm)</td><td>13.68 [11.43, 16.12]</td><td>45.41 [36.55, 54.75]</td><td>38.50 [36.52, 40.86]</td><td>39.97 [38.15, 42.19]</td></tr><tr><td>Framework MAE (bpm)</td><td>9.23 [7.39, 11.28]</td><td>34.61 [26.47, 45.07]</td><td>21.38 [19.18, 24.10]</td><td>26.08 [23.70, 28.60]</td></tr><tr><td>Beneficial corrections (%)</td><td>16.06 [14.78, 17.48]</td><td>30.74 [24.00, 37.45]</td><td>36.64 [35.08, 38.27]</td><td>26.99 [24.43, 30.04]</td></tr><tr><td>Error change within ±1 bpm (%)</td><td>82.26 [80.75, 83.67]</td><td>62.49 [57.10, 67.46]</td><td>59.84 [58.43, 61.12]</td><td>69.88 [67.03, 72.28]</td></tr><tr><td>Harmful corrections (%)</td><td>1.68 [1.28, 2.16]</td><td>6.77 [3.97, 10.09]</td><td>3.52 [2.84, 4.46]</td><td>3.12 [2.80, 3.45]</td></tr><tr><td>Severe added error (%)</td><td>0.37 [0.16, 0.66]</td><td>3.89 [1.92, 6.14]</td><td>1.61 [1.14, 2.26]</td><td>1.35 [1.17, 1.54]</td></tr></table>

Note: Values are macro-averaged point estimates with participant-bootstrap 95% confidence intervals. DaLiA, Wrist, Galaxy, and Wild include 15, 8, 20, and 14 participants with eligible references. Three DaLiA seeds or 15 external source configurations are averaged within participant and estimator, followed by equal weighting of ten estimators and participants. The three correction categories use the same paired available-window denominator and sum to 100% before rounding. Severe added error instead uses all reference-eligible windows. Bootstrap resampling uses 10000 draws and seed 260922 with fixed model weights. Intervals reflect participant composition, not retraining uncertainty.

TABLE S4  
PAIRED MAE CHANGES RELATIVE TO ORIGINAL ESTIMATES WITH 95% CONFIDENCE INTERVALS
<table><tr><td>Dataset</td><td>Framework – original MAE (bpm)</td></tr><tr><td>DaLiA</td><td>-4.45 [-5.47, -3.65]</td></tr><tr><td>Wrist</td><td>-10.79 [-15.33, -5.98]</td></tr><tr><td>Galaxy</td><td>-17.11 [-18.07, -16.25]</td></tr><tr><td>Wild</td><td>-13.89 [-15.04, -12.98]</td></tr></table>

TABLE S5  
MAE CHANGE DECOMPOSITION AT 80% TARGET COVERAGE WITH 95% CONFIDENCE INTERVALS
<table><tr><td>Dataset</td><td>Total change (bpm)</td><td>Correction (bpm)</td><td>Selection (bpm)</td></tr><tr><td>DaLiA</td><td>-6.26 [-7.71, -5.12]</td><td>-3.88 [-4.87, -3.07]</td><td>-2.38 [-2.95, -1.90]</td></tr><tr><td>Wrist</td><td>-12.19 [-17.51, -6.51]</td><td>-11.24 [-16.05, -6.27]</td><td>-0.95 [-1.91, -0.08]</td></tr><tr><td>Galaxy</td><td>-19.29 [-20.40, -18.27]</td><td>-17.79 [-18.85, -16.85]</td><td>-1.50 [-1.97, -1.02]</td></tr><tr><td>Wild</td><td>-15.02 [-16.37, -13.89]</td><td>-14.70 [-15.85, -13.79]</td><td>-0.32 [-0.64, -0.00]</td></tr></table>

TABLE S6  
MAE CHANGES AFTER MODULE ABLATION WITH 95% CONFIDENCE INTERVALS
<table><tr><td>Dataset</td><td>No harmonic (bpm)</td><td>No motion (bpm)</td><td>No memory (bpm)</td></tr><tr><td>DaLiA</td><td>0.61 [0.36, 0.85]</td><td>0.97 [0.74, 1.24]</td><td>0.87 [0.71, 1.05]</td></tr><tr><td>Wrist</td><td>0.93 [-0.23, 2.10]</td><td>5.98 [3.02, 8.67]</td><td>1.52 [0.52, 2.46]</td></tr><tr><td>Galaxy</td><td>-1.62 [-1.91, -1.31]</td><td>6.08 [5.51, 6.63]</td><td>1.17 [0.89, 1.48]</td></tr><tr><td>Wild</td><td>-0.94 [-1.35, -0.56]</td><td>7.08 [6.71, 7.43]</td><td>1.35 [1.10, 1.58]</td></tr></table>

TABLE S7  
ACTUAL COVERAGE AND CONFIGURATION FAILURES WITH FIXED SOURCE THRESHOLDS
<table><tr><td>Dataset</td><td>Source80 coverage (%)</td><td>Source MAE5 coverage (%)</td><td>MAE5 zero-output / all</td><td>MAE5 above 5 / nonempty</td><td>Above 5 (%)</td></tr><tr><td>DaLiA</td><td>79.07</td><td>69.61</td><td>48/450</td><td>159/402</td><td>39.55</td></tr><tr><td>Wrist</td><td>42.29</td><td>44.87</td><td>183/1200</td><td>930/1017</td><td>91.45</td></tr><tr><td>Galaxy</td><td>45.48</td><td>47.24</td><td>365/3000</td><td>1310/2635</td><td>49.72</td></tr><tr><td>Wild</td><td>30.50</td><td>36.38</td><td>227/2100</td><td>1376/1873</td><td>73.47</td></tr></table>

TABLE S8  
CENTRAL-WINDOW ERRORS IN REPRESENTATIVE EXAMPLES
<table><tr><td>Outcome</td><td>Participant</td><td>Window start (s)</td><td>Original error (bpm)</td><td>Framework error (bpm)</td></tr><tr><td>Retention</td><td>P21</td><td>1412</td><td>1.621</td><td>1.621</td></tr><tr><td>Correction</td><td>P15</td><td>420</td><td>12.104</td><td>2.364</td></tr><tr><td>Rejection</td><td>P12</td><td>2508</td><td>9.382</td><td>9.382 (before rejection)</td></tr><tr><td>Harmful correction</td><td>P15</td><td>168</td><td>1.513</td><td>4.000</td></tr></table>

TABLE S9  
MEAN MAE REDUCTIONS FOR SELECTED CORE ESTIMATORS
<table><tr><td>Core estimator</td><td>DaLiA</td><td>Wrist</td><td>Galaxy</td><td>Wild</td></tr><tr><td>Deep PPG [3]</td><td>1.335</td><td>5.222</td><td>2.910</td><td>1.215</td></tr><tr><td>JOSS [5]</td><td>5.756</td><td>15.847</td><td>7.560</td><td>5.779</td></tr><tr><td>SpaMA [6]</td><td>7.071</td><td>10.991</td><td>13.902</td><td>5.256</td></tr><tr><td>WFPV [7]</td><td>1.134</td><td>19.611</td><td>14.571</td><td>4.720</td></tr><tr><td>KID-PPG [10]</td><td>0.348</td><td>8.562</td><td>1.093</td><td>0.120</td></tr><tr><td>BeliefPPG [9]</td><td>&lt;0.001 (near zero)</td><td>&lt;0.001 (near zero)</td><td>&lt;0.001 (near zero)</td><td>&lt;0.001 (near zero)</td></tr></table>

TABLE S10  
PARTICIPANT COUNTS AND SHARED COVERAGE FOR FIG. 3
<table><tr><td>Dataset</td><td>Participants</td><td>Shared coverage (%)</td></tr><tr><td>DaLiA</td><td>15</td><td>100.00</td></tr><tr><td>Wrist</td><td>8</td><td>98.29</td></tr><tr><td>Galaxy</td><td>20</td><td>97.81</td></tr><tr><td>Wild</td><td>14</td><td>99.34</td></tr></table>

Note: Original and framework estimates are compared on the same available windows. Shared coverage is a separate macro-average for each dataset. The unrounded DaLiA value is 99.9973%, displayed as 100.00% to two decimal places. These are the Fig. 3 analysis windows, not fixed-source-threshold transfer results.

TABLE S11  
PROTOTYPE RECORDINGS AND EVALUATION WINDOWS
<table><tr><td>State</td><td>Common duration (s)</td><td>Mean reception rate (Hz)</td><td>Reference-eligible windows</td></tr><tr><td>static</td><td>274.46</td><td>9.67</td><td>59</td></tr><tr><td>walk</td><td>432.56</td><td>9.89</td><td>54</td></tr><tr><td>run</td><td>347.49</td><td>11.19</td><td>77</td></tr><tr><td>rest</td><td>575.38</td><td>10.28</td><td>134</td></tr></table>

Note: n = 1 participant and four separate recordings. Durations are audited common recording intervals. Counts include reference-eligible 8 s windows with a 4 s stride and endpoints at or after 40 s, totaling 324. Reception rates differ from the 12.5 Hz processing grid and 64 Hz model input sampling rate.

TABLE S12  
PAIRED ALL-AVAILABLE-OUTPUT COMPARISONS FOR PROTOTYPE INFRARED RECORDINGS
<table><tr><td>Core estimator</td><td>Paired windows</td><td>Original MAE (bpm)</td><td>Framework MAE (bpm)</td><td>Reduction (bpm)</td></tr><tr><td>BeliefPPG</td><td>289</td><td>2.256</td><td>2.256</td><td>0.000</td></tr><tr><td>KID-PPG</td><td>304</td><td>12.601</td><td>9.887</td><td>2.714</td></tr><tr><td>TimePPG</td><td>307</td><td>14.484</td><td>9.571</td><td>4.913</td></tr><tr><td>Deep PPG</td><td>289</td><td>8.231</td><td>4.235</td><td>3.996</td></tr><tr><td>Welch</td><td>307</td><td>3.364</td><td>2.733</td><td>0.631</td></tr><tr><td>ACC-NLMS</td><td>307</td><td>7.049</td><td>3.533</td><td>3.516</td></tr><tr><td>TROIKA</td><td>307</td><td>26.945</td><td>4.366</td><td>22.579</td></tr><tr><td>JOSS</td><td>307</td><td>2.145</td><td>1.691</td><td>0.454</td></tr><tr><td>SpaMA</td><td>307</td><td>12.704</td><td>5.073</td><td>7.631</td></tr><tr><td>WFPV</td><td>307</td><td>1.086</td><td>1.104</td><td>-0.019</td></tr></table>

Note: The four states are pooled without additional rejection. Each estimator’s original and framework outputs use the same available windows. Values are means over 15 frozen source configurations. Negative reductions indicate increased error. n = 1; window counts are not independent participant counts.

TABLE S13  
FIXED SOURCE-THRESHOLD TRANSFER TO PROTOTYPE INFRARED RECORDINGS
<table><tr><td>Core estimator</td><td>Source80 coverage (%)</td><td>Source MAE5 coverage (%)</td><td>Source MAE5 nonempty MAE (bpm)</td><td>Zero-output configurations</td></tr><tr><td>BeliefPPG</td><td>81.05</td><td>87.35</td><td>2.250</td><td>0/15</td></tr><tr><td>KID-PPG</td><td>58.21</td><td>79.86</td><td>8.871</td><td>0/15</td></tr><tr><td>TimePPG</td><td>72.82</td><td>82.96</td><td>8.244</td><td>0/15</td></tr><tr><td>Deep PPG</td><td>84.44</td><td>83.83</td><td>3.490</td><td>0/15</td></tr><tr><td>Welch</td><td>85.08</td><td>84.61</td><td>1.279</td><td>0/15</td></tr><tr><td>ACC-NLMS</td><td>83.83</td><td>42.51</td><td>1.910</td><td>7/15</td></tr><tr><td>TROIKA</td><td>85.74</td><td>69.47</td><td>1.918</td><td>0/15</td></tr><tr><td>JOSS</td><td>86.13</td><td>50.51</td><td>1.252</td><td>6/15</td></tr><tr><td>SpaMA</td><td>80.58</td><td>56.50</td><td>3.171</td><td>3/15</td></tr><tr><td>WFPV</td><td>86.85</td><td>84.67</td><td>1.125</td><td>0/15</td></tr></table>

Note: Thresholds are reused from source calibration without adjustment to prototype ECG. Coverage includes all 15 configurations; MAE averages only nonempty configurations. Zero output is not assigned zero error. The two thresholds serve different calibration objectives.

![](images/0fce41e4c7484f0bd7c49d6b90d4685d4565bc83c98ddaf30c87cbc9e5f1726e.jpg)  
(a)

![](images/670d4b0da941ce402abeecf6ec43147b65da0b1932d58235aa0d5268cdfe0a9e.jpg)  
(b)  
DeepPPG

TimePPG  
![](images/59c9197db619ea0792e6ce724d77a453980bac3acba46abd6e23fa2590a5f62f.jpg)

![](images/0c2796149879d79bf351cbdd454e171b8eca56ad7e6a4d4dc452bc035b75c00f.jpg)

(c)  
![](images/4e504a87a42bb972130d6ae1d86206a3eb424aab01d174743d5096e89060311b.jpg)  
(e)

(d)  
![](images/3920ec7b884f794970a98873659b4864a3196545c1d542046501383ef1074465.jpg)  
(f)

![](images/208bf9a13ca76b6b95abf265c2f1ab73f204ec91a6be8b7941918fdaafb4c893.jpg)  
(g)

![](images/8cd151d6ea3b75261c804b04177deaabd0df78f5992caf0b3b0e7bada1282c37.jpg)  
(h)

![](images/4c8f97ada1607ca5942a8e60426f1e143e358bd0d41a1dede4df46cb2c5c8e06.jpg)  
(i)

![](images/12e257cc9f9d4af8d002bb1a885b7a63617f4e303eb744b4ba76ba5536129e1f.jpg)  
(i)  
Ridge rejection  
- Same-risk rejection  
Fig. S1. Estimator-specific risk–coverage curves on DaLiA. (a)–(j) BeliefPPG, KID-PPG, TimePPG, Deep PPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV. Curves compare original estimates, the framework, ridge rejection, and same-risk rejection, showing accepted-set MAE at different target coverage levels. Vertical scales differ by panel. Shading indicates participant-bootstrap 95% confidence intervals.

Ridge rejection

![](images/cfc8c76ed9783e6e7f809c3c159280d47b8a096c54450802d69ceafe33df442b.jpg)  
(a)  
TimePPG

![](images/43af8bac3b4196f4194ecf9c99db9e87133ea28a7f3c2a334fe28dbe57104e74.jpg)  
(b)

![](images/ab56f36120fd62e40965b21d85b045a29b0b2ad5d60493d66b3771212aa0088d.jpg)  
(c)

DeepPPG  
![](images/4443adf7e751be76594a33541c018d8ac1a5165386ac1d54945f6c331e64486a.jpg)  
(d)

Welch  
![](images/d44040a44248d0731ffffb8d7181e2829af583ad712f33782c45ab06d57a5ad5.jpg)  
(e)

ACC-NLMS  
![](images/bd06a27e3f36ecdedc42283de957fd3c6f0163c4292095e03061953c9ddc9b12.jpg)

![](images/c3600d7d8c208c2fa4ea0613a5663a44a8714b1db231c0f3f16d1b097c22e40b.jpg)  
(g)

(f)  
![](images/3ab23f365ab07233ee35784f96979b325be5866c62b689442455955ce2b18492.jpg)  
(h)

SpaMA  
![](images/aae585319f1000c1c256169b02757ebf10a6c8d9223861929c328c3a5fd07aed.jpg)

![](images/d6c682dcc59d8f70044383fab885c254933d1f88e7d9917a97168ed03df0c05e.jpg)

![](images/6a2a461bef60b0f10f52e692a5e3ed2eea8f292a6cb71ac77c68bb446abde830.jpg)  
(j)  
Fig. S2. Estimator-specific risk–coverage curves on Wrist. (a)–(j) BeliefPPG, KID-PPG, TimePPG, Deep PPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV. Curves compare original estimates, the framework, ridge rejection, and same-risk rejection, showing accepted-set MAE at different target coverage levels. Vertical scales differ by panel. Shading indicates participant-bootstrap 95% confidence intervals.

![](images/f39365b380be8fee40b1d737dc8ebc614f2fd63a750db1b650fd6f7decf9988d.jpg)

![](images/f9a29bf2131874f79c2222e9a12fda0cca92a19a6a7cb124ab9dca686b886f93.jpg)  
(a)  
TimePPG

![](images/7ee3ab08649848243cccc6356f0aa5f08cf6bd920357413fae5f5d74a63301ad.jpg)  
(b)

![](images/02d332daa7b0600c8aefc421d347fd74c08a562047cbdbfc1192d223e8cdb2ec.jpg)  
(c)

DeepPPG  
![](images/7d543684ddae407fbc18a380e90a82cf5ede44a24cf00de219b31e1e5701bd92.jpg)  
(d)

Welch  
![](images/661e510fa84bb8fe97b4f6e68ff87c5c58bcd4e5e8cd159f369012d0fca6a581.jpg)  
(e)

ACC-NLMS  
![](images/3e311a8a8b671cba8e4f97cd0388c8d2faecca6400d9e8f8cbc15420b4114060.jpg)

![](images/7000dd9ef3099a1f9fb781b7ff47808f4dabd6014cc189c8ae30cc9ea0d23daf.jpg)  
(g)

(f)  
![](images/14dc547919d1d3174cf1c470976c798c2d9db1a28476b5b64041009dbf896f72.jpg)  
(h)

SpaMA  
![](images/789f15d6980dc3ab9580d079639296487b5f262bc93c0ad20d4f79e772144222.jpg)

![](images/c280d55fcbd9a21206cbba4813de6341ef00f4bdc7b05d01712358dfbfd7e039.jpg)  
(j)  
Fig. S3. Estimator-specific risk–coverage curves on Galaxy. (a)–(j) BeliefPPG, KID-PPG, TimePPG, Deep PPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV. Curves compare original estimates, the framework, ridge rejection, and same-risk rejection, showing accepted-set MAE at different target coverage levels. Vertical scales differ by panel. Shading indicates participant-bootstrap 95% confidence intervals.

Ridge rejection

![](images/f03b0fcca6c6992724aebef2edb5437575afe9cacf1a9b7a9cb8838ac00d166c.jpg)  
(a)

KID-PPG  
![](images/ac22dcb3d0079da805b30d8dbeedea7d3cce5da479ce431f56558aeca38db3f8.jpg)  
(b)

TimePPG  
![](images/9faea7318b569202b6cbfd8305b3f045deaf810b86d4d71909f4771e1c120511.jpg)  
(c)

DeepPPG  
![](images/fed3859acb9ee2251228a6ddcfe940fce8c0d33234a101ed9e663d7553fbe8a9.jpg)  
(d)

Welch  
![](images/b93ced74e4d588f8a7137232c99b21eed7176b2509867e822ff0f45cb8dafa46.jpg)  
(e)

ACC-NLMS  
![](images/349594fb013ebc7c72e5634cbf07b36d49a7248070e985129ae95d525bcf6064.jpg)

![](images/8e543f93d559218a66b8c91b2239288acad3a794ddef44ace788164a746e030b.jpg)  
(g)

(f)  
![](images/0adb1f7e08627ac42bdfaa4c16020600f7c6bb026a1eea98879bcc10e00222ed.jpg)  
(h)

SpaMA  
![](images/356781113ddededa90d81addf20a2501581d441e898a843ce7afbafdd64ebd83.jpg)

![](images/c22df119868ca890478b11ca45abc830a8757fa4e4469d7ff768ae1785b6e3ed.jpg)

![](images/130fdf121b51c104757a83d34fd7af9c038a8ef776f83c1fc8e6fda9d30bc503.jpg)  
(j)  
Fig. S4. Estimator-specific risk–coverage curves on Wild. (a)–(j) BeliefPPG, KID-PPG, TimePPG, Deep PPG, Welch, ACC-NLMS, TROIKA, JOSS, SpaMA, and WFPV. Curves compare original estimates, the framework, ridge rejection, and same-risk rejection, showing accepted-set MAE at different target coverage levels. Vertical scales differ by panel. Shading indicates participant-bootstrap 95% confidence intervals.

DaLiA  
![](images/8dfd7b277fbef14d78b94918c35ecf941ca5fa3487dc0cc13c58e1f56c1fba96.jpg)  
(a)

Wrist  
![](images/d3e481da33c12a0041247fc6532f0c0299578cfb485860307fbbf44dacf618ae.jpg)  
(b)

Galaxy  
![](images/56011332cfb36e6af730f57bd26c66aec3a3d00bc343b23d641cddae06df7604.jpg)  
(c)

Wild  
![](images/02f6cd9e1611f001e384451ac6111aa5df3adcbbff0513986066d02092c29280.jpg)  
Fig. S5. MAE changes by participant and estimator. (a)–(d) DaLiA, Wrist, Galaxy, and Wild. Cells show framework minus original MAE; blue indicates decreases and red increases. The symmetric logarithmic scale is linear within ±5 bpm and spans ±140 bpm. Blank columns indicate participants without eligible references. Gains vary across participants and estimators.

DaLiA  
![](images/32e277931eafb7fc5833b1fa66bc6698e84a509f1ef5f36540ffd6656a6aff1f.jpg)  
(a)

Wrist  
![](images/0006fc883208a35dd92c87db2612a3f6d25717d3dc422ea319ddf1db6cc1ca67.jpg)  
(c)  
Galaxy

(b)  
Wrist  
![](images/92bcde074ae5db95b464d60b0c20cf173577261c8e83893141d916a316b14886.jpg)

![](images/691b6dc49e681d0018f3190880075c033568d097d4c32003ce962630086e9913.jpg)

![](images/cbb7e58be529bcf6797ba45203249fe80be16b3c1ba9ca0cc4b201a002ab2f3e.jpg)  
(e)  
Wild

(d)  
Galaxy  
![](images/3496539cdba95bdc33185524f41667cfa34878383f6aadbd5b0cb4fd7062aedf.jpg)  
(f)

![](images/a9deb10a39660a9ee0343b59555350f3e9f94ad625fa8e366efbb9162c13ea41.jpg)  
(g)

Wild  
![](images/f2e099ad346e3e4f55915faae7c22d5d47c45fcc1aa920a99f5aa9adad315b99.jpg)  
(h)  
Fig. S6. Error and harmful correction rates for seven output strategies. Rows correspond to DaLiA, Wrist, Galaxy, and Wild. Left: MAE. Right: harmful correction rate. Cap 20 denotes a 20 bpm correction cap, check denotes posterior-gain consistency checking, and No cap denotes uncapped correction. Blue identifies the proposed method. Error bars indicate participant-bootstrap 95% confidence intervals.

DaLiA  
![](images/e1280671895e70d8ff6d7c25f0a0f5cb718d71780ec62356e392b9d26a2646ed.jpg)  
(a)

Wrist  
![](images/629aeffc408353898bb6e5ead2a4c56b733b16ec75fc92b8af2e793ed9e2945e.jpg)  
(b)  
Wild

Galaxy  
![](images/eaf1c3a2bc73547edfd515312a45a4a74d463140af577aef3d0cbb456dbcce5b.jpg)  
(c)

![](images/3c444fd02b0dac335707f699f5d6ca46c01e88d438faf3d5c6a2422efe6f2e79.jpg)  
(d)  
Fig. S7. Macro-averaged MAE reductions after excluding one estimator. (a)–(d) DaLiA, Wrist, Galaxy, and Wild. The vertical axis identifies the excluded estimator. Positive values favor the framework. Orange marks exclusion of local TROIKA; other exclusions are blue. Average improvements remain under every exclusion, although their magnitudes differ. Error bars indicate participant-bootstrap 95% confidence intervals.

DaLiA  
![](images/7b9b2572e59e8ac5c3799ee7799eac9ad1d1c7eac93cc5e5a9ebe646fbeaaf8b.jpg)  
(a)  
Wrist

DaLiA  
![](images/28a59ecc8ae46d3db9c6205a0a9cf601df905aba17511731e6193edfe74f0c7b.jpg)  
(b)

![](images/fbf620a34457bb9b77186f18cd0b9b6221ca11a7c92b568fea7a8b92ac9f2e59.jpg)  
(c)

Wrist  
![](images/7c6176722b0be2d6d1975e6127c486b627484b1c0f16121408ec42758981824e.jpg)  
(d)

Galaxy  
![](images/5e7eb8ef8f0e7509b2eda79eec5101de43c4464dfdf261c4db807a151ce29722.jpg)  
(e)  
Wild

Galaxy  
![](images/d3880635e511d672d1727bff65279def12496f27b6c5e3b3cd946980308f2fc1.jpg)  
(f)  
Wild

![](images/7ae88ac1b1c212f0311d3e9923ee0d4152b1022104ffe3e13ecb0a5fc4bb18fc.jpg)  
(g)

![](images/a8b344b23294ce02680097c28c22400e80bb61055d760de60e852665308064a3.jpg)  
(h)  
Fig. S8. Correction outcomes by estimator. Rows correspond to DaLiA, Wrist, Galaxy, and Wild. Left: MAE reduction. Right: harmful correction rate. Gains differ across estimators, with BeliefPPG nearly unchanged. Horizontal scales are selected for each dataset. Error bars indicate participant-bootstrap 95% confidence intervals.

MAE5: configurations with no output (%)  
![](images/c96ecc815fbd74d18f6429ad477ec7752995af9c5e7b2b5a9f29202543012658.jpg)  
(a)

MAE5: MAE >5 bpm among nonempty configurations (%)  
![](images/7bf554f360d6784488151b767cb0706eda0762caff50f7d05338404d210a9d61.jpg)  
(b)

Source80: actual coverage (%)  
![](images/0bbf5f5b65f0079a7acb0614af674b08dd7e284127e83919bbba784ed9e708cc.jpg)  
(c)

MAE5: actual coverage (%)  
![](images/0638ba3f08c3195eefa75d2680cd88f2f1814b1e0c03a9003630bd3fd28c8211.jpg)  
(d)  
Fig. S9. Transfer of fixed source thresholds. (a) Zero-output configuration proportion under the source MAE 5 bpm threshold. (b) Proportion of nonempty configurations exceeding 5 bpm MAE. (c) Actual coverage under the source 80% threshold. (d) Actual coverage under the source MAE 5 bpm threshold. Cells are percentages on a common 0%–100% color scale. External datasets do not retain the source coverage or error targets.

![](images/9d627a173df6107b026c6eb81a97bdb57b545eaadab416ce3d189769b30c81ac.jpg)

![](images/31daa402544a5e23c59e576ae010f10cf7d214b62b607721a0e7ff73aa9e0198.jpg)

![](images/ed99755ae98dc7a611781311c16f9fc8dcb0cf03d8c4a45c10bece8e3bbd907e.jpg)

![](images/62b19f9e7efd04990365019b25b5d42eec32a801834b5234c29f89bf66617587.jpg)

![](images/c0f205c59f4d7c20d21041f4d20cffd765a997624eae9e7afbc75435bda3723a.jpg)  
(a)  
(c)  
Fig. S10. Fixed 20 s examples of synchronized raw prototype signals. Dual-wavelength PPG, acceleration, and independent ECG are shown for static, walking, and running recordings. Units follow panel labels; offsets are removed only for display. Separate recordings are not concatenated into one continuous acquisition. These are acquired signals; model inputs after preprocessing are shown in $\mathrm { F i g } .$ . S18. (a)–(c) Static, walking, and running. Gray, purple, and green acceleration curves denote $\mathbf { X } , \mathbf { y } ,$ and z axes.

Time from center-window start (s  
Time from center-window start (s  
![](images/38e8ecaa1c93c57e01de8832a9b3bb96b0daff176529618ac1aa002415a7db34.jpg)

![](images/66bc01cc14fddb512fd37283c4d51e4aff1218ccd54f5b46ea01858ddfb92bc8.jpg)

![](images/a9e91c2f38a66237c885478e17e982369b74c59319dfd2424864db222eb5838c.jpg)

![](images/b7d4bfcb7772d5375127a7f8831f8f12c828f1358f26660e12efed9860dafbd1.jpg)

![](images/6aadc57efc152583dfe1a1a1872866b6146ef46bf2bfcd2f1f7ef0270dd913e7.jpg)

![](images/bef912566a50ac6a7289df5ce591d397f782a2c8e244c5b1476cac86682fc5d4.jpg)

![](images/1a397a0a3f179322caf979396a8dc880084be6f58143c775e22b070451451d47.jpg)

![](images/7297b2f19d736b245d748cdb787f11e6db1b567170d533e2db53165cc6c0cff1.jpg)

![](images/158f02252500a6ab77ccbf3003b280882b0804264862a5f2f49f8c21b655ad32.jpg)

![](images/42a3dbf0179d7bf52609af725d02aee8ccec5dd301e26dbd36025c91e0f0ac8c.jpg)  
(a)

![](images/acb6bf3806207f806b25a472c6382dcbc8f76541f1827edf07d13bfdd7287e21.jpg)  
(b)

![](images/f97f10080541cbca7b4dec88c05657b057b781d096a307dd0226782dddae6ef7.jpg)  
(c)  
Fig. S11. Retention, correction, and rejection examples in prototype recordings using TimePPG. Each column aligns raw signals, heart rates, and references. The original estimator and framework receive identical processed inputs. Examples use source fold 0 and seed 219 and the first-eligible-window rule in S20.3, without selection for error reduction. Rejected windows have no final heart rate. Examples do not estimate overall action frequencies.

![](images/5055afbe90ac3875dd90faaa5ae883d8b8184926c1e605a164eb6aeefeca7d93.jpg)

![](images/c03d8708f1652c3753202c712ac8ea59ba98a8dfc8235ae8de80b8e0d63a2b67.jpg)  
(b)

![](images/51b013f56136997a0218b277572016c7bd1ff07e59a6c25a86b70e0e9eef90b4.jpg)  
(c)  
Fig. S12. All-available-output accuracy and harmful correction in prototype infrared recordings. (a) Paired MAE for ten estimators and their frameworks. (b) Proportions with absolute-error increases above 1 bpm, with and without M4. (c) Macro-averages for all ten estimators and after excluding TROIKA. Four states are pooled, with estimator-specific paired available windows and no additional rejection. Bars are means over 15 frozen source configurations; error bars show configuration ranges. n = 1.

Static  
![](images/4cb6c3b09fe8c62f7d15daa1ba15a5a328fe6c2d679b3dadf4a4d695d650f414.jpg)  
(a)

![](images/74c27c9a037bba0b47fd800a53f1b79f320a418b48ca3981837da788dbaec762.jpg)  
(b)

Run  
![](images/8fb1fce775f83d764a028b1231813f010152d1c1272b00c8c664b7a01eecd218.jpg)  
(c)

Rest  
![](images/36639e36bacd0a8b454b6dff4147840215fb60212687d50a4835553f03ac34df.jpg)  
(d)

All four states pooled  
![](images/bf640ea3c863fba8f7ad124eb0f6c1214ff4a2ba4e9ea39eda2348db2f171ab0.jpg)  
(e)

Gain decomposition  
![](images/79f7c61d6014b84ddd6a011c2b42d6e830daae1de0e157066db71eaf41fda579.jpg)  
(f)  
Fig. S13. Risk–coverage curves and gain decomposition in prototype infrared recordings. (a)–(d) Static, walking, running, and recovery rest. (e) Four states pooled. (f) Estimator-specific correction and selection gains at 80% target coverage. Curves show original-risk selection, the framework, ridge rejection, and same-risk rejection. The framework and same-risk rejection share accepted windows. Shading shows the range of 15 source configurations, not population confidence intervals.

OriginalSelective correction

![](images/23de29c59372189c7e36fa2504acb756d87f10fad413cb3afb5ffbfced41de55.jpg)  
(a)

![](images/9f989e21bf5aebd3c2facd7fb2656b1cb02b1d4e7c837859091bf58707c9a53e.jpg)  
(b)

![](images/468f6a6533d8201624fc8227c9d83b0f65a3193341a08bb928c8e3719a06b470.jpg)  
(c)

![](images/1e9cb459a335bfcbd747d40646f3e0f0be10591ece271ea65514d0950f440ab0.jpg)

![](images/ede8f7a19f5d109260ea07462c3e977a720641875866c079a563ca7a5d0f5bc8.jpg)  
(e)

(d)  
![](images/bd8cb567fdae36884b271cea161b63390a811562e453a818e2dbcd746a82f05c.jpg)  
(f)

![](images/8e998c3c523eb144eecb2ebbcc44c4908ff10848f37309a98d6a0a96464722d5.jpg)  
(g)

![](images/45ea4344eb9db0399a20111247b3d01d70088baf67fd92b0a2a231639b21e9cf.jpg)  
(h)

![](images/ba0ff1b910497932e2deff4d9271ae7d456a8c9f9b6f8d24f622d5f5fb096094.jpg)  
(i)

![](images/e681ba4a6a9db1e83216c4511e04b7817186df250c24413726fd4eaef89d4431.jpg)  
(j)  
Fig. S14. Risk–coverage curves for ten estimators in prototype infrared recordings, pooled across four states. Panel-specific vertical scales show original-risk selection, the framework, ridge rejection, and same-risk rejection. The horizontal axis is actual coverage. Curves rank risk within target recordings and differ from fixed-source-threshold transfer. Shading indicates frozen-source-configuration ranges. n = 1.

IR: Original MAE (bpm)  
![](images/27f17a26c73770c2657ced144352348c60415f52505afc47b51b1560657d62ef.jpg)

IR: Framework MAE (bpm)  
![](images/93c70fc91db311acea301055a6c0b5d52e5dc3047ebc22c8ef1be905a3c3e389.jpg)

Red: Original MAE (bpm)  
![](images/5396a73b98e69bb07dbcb31e968ed880d4ec4674fe16b310fc944bd0f4b8a2d7.jpg)  
(c)

Red: Framework MAE (bpm)  
![](images/65bf7fa605a0e39c5fe90a3ad16d9ab67af08131ec850ceededb91ce024ffa7c.jpg)  
(d)  
Fig. S15. Estimator-specific MAE across two channels and four prototype states. (a), (b) Original and framework infrared results. (c), (d) Original and framework red-channel results. Each method uses its own paired available windows. Values average 15 frozen source configurations. Color saturates at 40 bpm; labels retain actual values. Infrared is primary, and red is a separate sensitivity analysis. Channels are not separate participants.

![](images/ae6f8c2add5db59786bbd66760ec9c2fd7987e96f07ef4ff91b9e09e6e70ef1b.jpg)  
(a)

![](images/eb7e83aa10ba70672576d0968193c6ee55741363dbfbc2613685d0bf1eb734b1.jpg)  
(b)

![](images/3a83238550f6b96d384650fa8aa8efd104cb24e2c01eb3779d0dc6807506c664.jpg)  
(c)  
Fig. S16. Module-ablation trade-offs in prototype infrared recordings. (a) Macro-averaged MAE. (b) Harmful correction rate. (c) Severe added-error proportion relative to all reference-eligible windows. Full, No harmonic, No motion, and No memory reuse their corresponding source-trained weights. No M4 removes candidate gain–harm selection. All comparisons pool four states without additional rejection. Error bars show ranges across 15 frozen source configurations.

Source 80% threshold

![](images/ced5cd1ecf1a90a75372a72dd4e180fa6ee195da3e7529ad5b77ade305049d70.jpg)  
(a)

![](images/8c43d0f2c3b5fbdc0171a56f5018e51c7c6830a39725a053a709744a777d3192.jpg)  
(b)

![](images/6ef8e3956716dee00812c8279d08ace27b8abb41e3af6a7917131dcdcd2be61f.jpg)  
(c)  
Fig. S17. Fixed-source-threshold transfer to prototype infrared recordings. (a) Actual coverage under source80. (b) Actual coverage under source MAE 5 bpm thresholds. (c) MAE for nonempty configurations under the latter rule. Thresholds are not adjusted to prototype ECG. Coverage includes all 15 configurations; MAE averages only nonempty configurations, with zero-output counts labeled separately. Missing output is not zero error.

![](images/65458136a6b592ed74caf60eb97e88f2b06511f8d97cb5c5933dd70bf280fadd.jpg)

![](images/1f074355dc160b76fe96498883f8ed16c290fcce249c9393a62f625fd2411697.jpg)

![](images/400440009b58fe1273c7bc5f72a88ca36dd8b96fe9731451800d86511497f6c8.jpg)

![](images/e9a429384816c010db5cd207755653a40300c97b2d9f91e1a3adbcebbd1f7aac.jpg)

![](images/f9c93f5676701bbcc4be4f3027c71dd4846350be3f16d674f07c8176a9bf195b.jpg)

![](images/14aa6a6a587ee85ab5dccd185ac34a025ad142e6de5459cc4ac30413483ce0a5.jpg)

![](images/70948eddf1c8a59c39984e8ba79f0b0e0ac63bced81522391e7a2ba004b01ae3.jpg)

![](images/465f85cc6194e3a60985708acfea53bf488306d78a481b4440aaa1466ecea9ad.jpg)

![](images/7043ae3bd307b8413e443c85848fcd2b5b0d87e8a8263ac3529b1ae92dad8286.jpg)  
(a)

![](images/3cddf4f3eb1957291101d4bcd6e755550bf273bd4e140f14219ad101ab386b40.jpg)  
(b)

![](images/0406e9e1cf04aadd2419c202ab55876db34794b77e220d3887845ff8197f385a.jpg)  
(c)

![](images/6ee4b875c47b0f8bbbdb3bbefd37c6189d13f02e01dac0b1331f3b8d71a298f2.jpg)  
(d)  
Fig. S18. Fixed examples of prototype signal preprocessing before and after bandpass filtering and continuous acceleration-assisted processing. Segments are 20 s long; Section S20.2 specifies parameters. Each estimator and framework receive identical processed signals. Zero-phase filtering and whole-block normalization are retrospective. No clean PPG reference is available, so waveform appearance cannot establish physiological recovery. (a)–(d) Static, walking, running, and recovery rest. Rows show raw PPG, processed PPG, and acceleration.

![](images/11f3448b6f2db8f20c592e9c7bccf632184a226b042163e99c282e3869b0b05c.jpg)  
(a)

![](images/18b142eacc1f6fa8f89445fd813165db2259ea17097fe96cf2b9f3e4afeb3bce.jpg)  
(b)

![](images/98e014e7c3b5f5a720098aa18e19419babeda4fae5e8001eca35c50054cf2848.jpg)  
(c)

![](images/fb5c03cec2b7ea33ed229de0c671384181b8dfc96ba5f9567327ef984e735950.jpg)  
(d)

![](images/f693497f2a58b57aa13b7b0ab1e6c48dc07ed6e01935a08731ac582cb5778414.jpg)  
(e)

![](images/b443240999a77e2cad967d2e51f5703550d5391451a11bf33bfcf2d4c0a61978.jpg)  
(f)

![](images/cb70b43f1c8686d85da81185234fff96fb4dd090da52bcc20ccf203e1a2d1f51.jpg)  
(g)

![](images/3aa5d2dab40d853e8cf192cf53094838bd121143f6b4c7073e706c6f907a6344.jpg)  
(h)

![](images/4fc3ad05bbac1d3c41c8468d9a7828b46700a48b159587001743981f823951ba.jpg)  
(i)  
Fig. S19. State-specific gains for static, walking, and running recordings. (a)–(c) Estimator-specific paired all-output MAE on logarithmic horizontal axes. (d)–(f) Macro-averages for all ten estimators, excluding TROIKA, and strictly common windows. (g)–(i) MAE at 80% target coverage for original-risk selection, same-risk rejection, the framework, and ridge rejection. Accepted counts are 47/59, 43/54, and 61/77 reference-eligible windows, respectively. Ranking is performed separately by state, unlike pooled four-state ranking. Bars and points are configuration means, with configuration-range error bars. n = 1.

![](images/111fdd75c3670833018f19291bcff3f8eea587fa4e1ea82acbc00af2d1030e9d.jpg)

![](images/af786c77325cc40bcd692b414d3b6f4ee06b1030daff768ef67c79dcc9dd9d37.jpg)

![](images/20008a65ae64bd257c9d3c4448de17aa307942cbdaf830bd1dec20b3cfd2e1c4.jpg)

Reduction in damage to correct answers (positive = protection)
<table><tr><td rowspan=2 colspan=1>BeliefPPGKID-PPG</td><td rowspan=1 colspan=1>+0.00% (n=56)</td><td rowspan=1 colspan=2>+0.00% (n=50)                         +0.00% (n=63)</td></tr><tr><td rowspan=1 colspan=1>+0.00% (n=55)</td><td rowspan=1 colspan=1>+1.11% (n=24)</td><td rowspan=1 colspan=1>-16.00% (n=5)</td></tr><tr><td rowspan=1 colspan=1>TimePPG</td><td rowspan=1 colspan=1>+0.00% (n=1)</td><td rowspan=1 colspan=1>+13.75% (n=16)</td><td rowspan=1 colspan=1>-3.33% (n=6)</td></tr><tr><td rowspan=4 colspan=1>DeepPPGWelchACC-NLMSTROIKA</td><td rowspan=1 colspan=1>+0.00% (n=54)</td><td rowspan=1 colspan=1>+1.20% (n=39)</td><td rowspan=1 colspan=1>NA (0 correct)</td></tr><tr><td rowspan=1 colspan=1>+0.23% (n=59)</td><td rowspan=1 colspan=1>+14.40% (n=50)</td><td rowspan=1 colspan=1>+3.50% (n=61)</td></tr><tr><td rowspan=1 colspan=1>+0.85% (n=55)</td><td rowspan=1 colspan=1>+3.43% (n=35)</td><td rowspan=1 colspan=1>+0.47% (n=57)</td></tr><tr><td rowspan=1 colspan=1>NA (0 correct)</td><td rowspan=1 colspan=1>+10.18% (n=38)</td><td rowspan=1 colspan=1>-5.05% (n=66)</td></tr><tr><td rowspan=1 colspan=1>JOSS</td><td rowspan=1 colspan=1>+1.69% (n=59)</td><td rowspan=1 colspan=1>+2.90% (n=46)</td><td rowspan=1 colspan=1>+3.33% (n=66)</td></tr><tr><td rowspan=1 colspan=1>SpaMA</td><td rowspan=1 colspan=1>+0.00% (n=6)</td><td rowspan=1 colspan=1>NA (0 correct)</td><td rowspan=1 colspan=1>+6.09% (n=69)</td></tr><tr><td rowspan=1 colspan=4>WFPVStatic                                Walk                                Run(d)Without M4         Full framework</td></tr></table>

Fig. S20. Protection and accuracy costs of M4 in static, walking, and running recordings. (a) Harmful correction rate. (b) Deterioration among initially accurate estimates, calculated from pooled method–window counts within each configuration. (c) Macro-averaged MAE. (d) Deterioration rate without M4 minus the full-framework rate; positive differences indicate protection and negative differences indicate worsening. Initially accurate means original absolute error ≤5 bpm; deterioration means an absolute-error increase >1 bpm. n denotes initially accurate windows per method; NA denotes an undefined rate when none exist. Means and ranges use 15 frozen source configurations, not population confidence intervals. Percent symbols in (d) denote absolute percentage-point differences, not relative percentage reductions.

![](images/defdc4e727eefcefc8b80504549bd3743159a41225a1e401eeaa905369f3560c.jpg)  
(a)

Prototype recordings  
![](images/91108b5753ecf8bddf7f56ffad25f3b6306f13a84f796c2ebe6158f0411e1c4d.jpg)  
(b)  
Fig. S21. Sensitivity to estimator composition. (a) Macro-averaged MAE reductions on four public datasets for all ten estimators and after excluding local TROIKA, with participant-bootstrap 95% confidence intervals. (b) Original and framework macro-averaged MAE in prototype infrared recordings for all ten estimators and without TROIKA. Bars average frozen source configurations; n = 1. Both comparisons retain estimator-specific paired windows without additional rejection. All ten implementations remain in the primary analysis Section S21 describes the motivation, implementation differences, and interpretation limits.

## REFERENCES

[1] J. Fine et al., “Sources of inaccuracy in photoplethysmography for continuous cardiovascular monitoring,” Biosensors, vol. 11, no. 4, Art. no. 126, 2021, doi: 10.3390/bios11040126.

[2] D. Biswas, N. Simões-Capela, C. Van Hoof, and N. Van Helleputte, “Heart rate estimation from wrist-worn photoplethysmography: A review,” IEEE Sensors J., vol. 19, no. 16, pp. 6560–6570, 2019, doi: 10.1109/JSEN.2019.2914166.

[3] A. Reiss, I. Indlekofer, P. Schmidt, and K. Van Laerhoven, “Deep PPG: Large-scale heart rate estimation with convolutional neural networks,” Sensors, vol. 19, no. 14, Art. no. 3079, 2019, doi: 10.3390/s19143079.

[4] Z. Zhang, Z. Pi, and B. Liu, “TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise,” IEEE Trans. Biomed. Eng., vol. 62, no. 2, pp. 522–531, 2015, doi: 10.1109/TBME.2014.2359372.

[5] Z. Zhang, “Photoplethysmography-based heart rate monitoring in physical activities via joint sparse spectrum reconstruction,” IEEE Trans. Biomed. Eng., vol. 62, no. 8, pp. 1902–1910, 2015, doi: 10.1109/TBME.2015.2406332.

[6] S. M. A. Salehizadeh, D. Dao, J. Bolkhovsky, C. Cho, Y. Mendelson, and K. H. Chon, “A novel time-varying spectral filtering algorithm for reconstruction of motion artifact corrupted heart rate signals during intense physical activities using a wearable photoplethysmogram sensor,” Sensors, vol. 16, no. 1, Art. no. 10, 2016, doi: 10.3390/s16010010.

[7] A. Temko, “Accurate heart rate monitoring during physical exercises using PPG,” IEEE Trans. Biomed. Eng., vol. 64, no. 9, pp. 2016–2024, 2017, doi: 10.1109/TBME.2017.2676243.

[8] A. Burrello et al., “Embedding temporal convolutional networks for energy-efficient PPG-based heart rate monitoring,” ACM Trans. Comput. Healthcare, vol. 3, no. 2, pp. 1–25, 2022, doi: 10.1145/3487910.

[9] V. Bieri, P. Streli, B. U. Demirel, and C. Holz, “BeliefPPG: Uncertainty-aware heart rate estimation from PPG signals via belief propagation,” in Proc. Conf. Uncertainty Artif. Intell. (UAI), vol. 216, 2023, pp. 173–183. [Online]. Available: https://proceedings.mlr.press/v216/bieri23a.html

[10] C. Kechris, J. Dan, J. Miranda, and D. Atienza, “KID-PPG: Knowledge informed deep learning for extracting heart rate from a smartwatch,” IEEE Trans. Biomed. Eng., vol. 72, no. 3, pp. 870–877, 2025, doi: 10.1109/TBME.2024.3477275.

[11] T. Choksatchawathi et al., “Improving heart rate estimation on consumer grade wrist-worn device using post-calibration approach,” IEEE Sensors J., vol. 20, no. 13, pp. 7433–7446, 2020, doi: 10.1109/JSEN.2020.2979191.

[12] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems, vol. 30, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

[13] Y. Geifman and R. El-Yaniv, “SelectiveNet: A deep neural network with an integrated reject option,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 97, 2019, pp. 2151–2159. [Online]. Available: https://proceedings.mlr.press/v97/geifman19a.html

[14] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Pearson, 2021, chs. 2 and 16. [Online]. Available: https://aima.cs.berkeley.edu

[15] D. Jarchi and A. J. Casson, “Description of a database containing wrist PPG signals recorded during physical exercise with both accelerometer and gyroscope measures of motion,” Data, vol. 2, no. 1, Art. no. 1, 2017, doi: 10.3390/data2010001.

[16] S. Park, D. Zheng, and U. Lee, “A PPG signal dataset collected in semi-naturalistic settings using Galaxy Watch,” Sci. Data, vol. 12, Art. no. 892, 2025, doi: 10.1038/s41597-025-05152-z.

[17] M. Meier, B. U. Demirel, and C. Holz, “WildPPG: A real-world PPG dataset of long continuous recordings,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://papers.nips.cc/paper\_files/paper/2024/hash/0433292a3eb101df8f4d72a63f5410d4-Abstract-Datasets\_and\_Benchmarks\_Track.html

[18] A. Shyam, V. Ravichandran, S. P. Preejith, J. Joseph, and M. Sivaprakasam, “PPGnet: Deep network for device independent heart rate estimation from photoplethysmogram,” in Proc. 41st Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2019, pp. 1899–1902, doi: 10.1109/EMBC.2019.8856989.

[19] M. Wilkosz and A. Szczęsna, “Multi-headed Conv-LSTM network for heart rate estimation during daily living activities,” Sensors, vol. 21, no. 15, Art. no. 5212, 2021, doi: 10.3390/s21155212.

[20] M. Meier and C. Holz, “Robust heart rate detection via multi-site photoplethysmography,” in Proc. 46th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2024, pp. 1–4, doi: 10.1109/EMBC53108.2024.10782155.