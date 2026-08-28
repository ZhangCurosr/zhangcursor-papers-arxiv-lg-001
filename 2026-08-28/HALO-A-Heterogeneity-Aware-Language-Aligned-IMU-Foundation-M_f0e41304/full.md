# HALO: A Heterogeneity-Aware Language-Aligned IMU Foundation Model for Open-Set Human Activity Recognition

Zihan Ding   
Hong Kong University of Science   
and Technology   
Hong Kong, China   
alxd219p1@gmail.com

## ABSTRACT

Liyu Zhang   
Hong Kong University of Science   
and Technology   
Hong Kong, China   
lzhangcx@connect.ust.hk   
Xiaomin Ouyang   
Hong Kong University of Science   
and Technology   
Hong Kong, China   
xmouyang@cse.ust.hk

Human Activity Recognition (HAR) using inertial measurement units (IMUs) enables a wide range of applications, yet the field still lacks a unified model that can generalize across diverse subjects, devices, and activities. Training such a model is dificult due to two key challenges: sensing heterogeneity—diferences in sampling rates, channel configurations, and sensor placements—and poor generalization to unseen activities and label vocabularies. We introduce HALO (Heterogeneity-Aware Language-aligned Open-set model), a domain-specific IMU foundation model that addresses both challenges through a two-stage training framework<sup>1</sup>. Stage 1 pretrains the IMU encoder with heterogeneity-aware selfsupervised learning, including adaptive-pooling tokenization, channel-independent feature extraction, and contextualized sensor conditioning that injects natural-language sensor descriptions into each channel embedding. Stage 2 aligns this IMU encoder with text embeddings via synonym-aware soft contrastive learning, enabling open-set recognition via cosinesimilarity retrieval without per-dataset classifiers. Trained on 10 public HAR datasets and evaluated on 7 held-out datasets, HALO outperforms five state-of-the-art baselines on all 8 aggregate metrics, and still leads on 3 of 4 settings under baseline-matched inputs. Despite using only ∼35M trainable parameters—10× fewer than the latest foundation model MOMENT (341.2M)—HALO improves zero-shot open-set accuracy, measured over all 87 training labels, by 13.7 percentage points. On two further datasets with severe distribution shift, every model including HALO collapses zero-shot. A video demonstration of HALO’s performance in real world is available at https://youtu.be/rooVKragtFU.

## 1 INTRODUCTION

Inertial measurement units (IMUs) are among the most widely deployed sensors in consumer electronics such as smartwatches and smartphones, enabling human activity recognition (HAR) for applications in health monitoring [2], rehabilitation [25], sports analytics [3], and human– computer interaction. However, developing a unified model for HAR that generalizes across diverse subjects, devices, and activities remains challenging due to two fundamental issues. First, sensing heterogeneity: datasets difer substantially in sampling rates (20–100 Hz), sensor suites (3–45 channels spanning accelerometers, gyroscopes, and magnetometers at various body placements), and label vocabularies with inconsistently overlapping activity names. Second, limited generalization: models trained on one dataset typically transfer poorly to others, especially when faced with unseen activities or mismatched label spaces.

Addressing these challenges requires a model that can (i) operate at arbitrary sampling rates without architectural modifications, (ii) handle variable sensor suites and placements, and (iii) recognize activities absent from training— i.e., support open-vocabulary recognition under label shift. Existing approaches address these axes in isolation. Selfsupervised methods (LiMU-BERT [36], CrossHAR [4]) improve transferability but require closed-world classifiers. General-purpose foundation models (MOMENT [8]) cover diverse signal types but lack HAR-specific inductive biases. Language-aligned models (LanHAR [13]) decouple the label interface via text similarity, but without strong sensor pretraining, naive text alignment proves insuficient for robust zero-shot transfer.

We present HALO (Heterogeneity-Aware Languagealigned Open-set model), a domain-specific IMU foundation model that enables a new deployment capability: a single pretrained model that accepts diverse wearable and phone IMU configurations—varying sampling rates, channel counts, and sensor placements—and recognizes activities via open-vocabulary cosine-similarity retrieval, with no per-dataset classifier or architectural modification required. HALO achieves this through a two-stage framework driven by the three design requirements above. Stage 1 pretrains a heterogeneity-aware IMU encoder via self-supervised objectives (masked patch autoencoding and contrastive learning), using (1) seconds-based adaptive-pooling tokenization for rate invariance, (2) channel-independent processing for variable sensor suites, and (3) contextualized sensor conditioning that injects natural-language descriptions of each channel’s sensor configuration (e.g., “Accelerometer X-axis, waist-mounted phone”) to provide placement context. Stage 2 aligns this encoder with a frozen SentenceBERT backbone [24] via synonym-aware soft contrastive learning: soft targets derived from text–text similarity allow semantically related labels (e.g., “walking”, “strolling”, “ambulating”) to reinforce one another, resolving the label-conflict prob lem inherent to multi-dataset training. At inference, activity recognition is cosine-similarity retrieval over any text label library—eliminating per-dataset classifiers.

![](images/1c91b98937fd3b6e2efcf1d0ecdacfede31a90f6d544c6af26ef525ae5afbd0b.jpg)  
Figure 1: HALO is a domain-specific IMU foundation model that can tackle various sensing heterogeneity for open-set human activity recognition.

We train HALO on 10 public HAR datasets and evaluate it on seven held-out test datasets—five main test datasets with high label coverage and two severe out-of-domain datasets containing entirely unseen activities—under four evaluation settings: zero-shot open-set (predicting from all 87 training labels with no knowledge of which labels appear in the test set), zero-shot closed-set (predicting only from the labels present in the test dataset), and 1% and 10% su pervised fine-tuning. Against five state-of-the-art baselines spanning reconstruction-based, text-aligned, and LLM-based paradigms, HALO leads on all 8 aggregate metrics across the five main test datasets, improving zero-shot open-set accuracy by 13.7 percentage points over the general-purpose foundation model MOMENT [8] (42.0% vs. 28.3%) while using only ∼35M trainable parameters—10× fewer than MO-MENT (341.2M). On two severe out-of-domain datasets, all models—including HALO—collapse in zero-shot, confirming the limits of current cross-dataset transfer under extreme sensor or label shift; yet HALO adapts efectively with as little as 1% labeled data, recovering to the strongest supervised performance on the harder of the two datasets.

We make the following contributions:

• We systematically investigate sensing heterogeneity and open-set generalization as the two fundamental barriers preventing current IMU models from serving as general-purpose foundations for mobile HAR, and derive three concrete design requirements: rateagnostic tokenization, variable-channel processing, and synonym-robust label alignment.

• We propose HALO, a domain-specific IMU foundation model that jointly addresses all three requirements through a two-stage training framework. The key integration insight is that heterogeneity-aware pretraining (adaptive-pooling tokenization, channel-independent processing, and contextualized sensor conditioning via natural-language placement descriptions) provides a robust encoder that can then be aligned to text via synonym-aware soft contrastive learning, enabling a deployment capability no prior system achieves: a single model that accepts diverse IMU configurations and recognizes activities via open-vocabulary cosinesimilarity retrieval.

• We establish a unified cross-dataset benchmark spanning 10 training and 7 test datasets with four evaluation settings (eight metrics), and demonstrate that HALO outperforms five state-of-the-art baselines on all 8 aggregate metrics—including under baselineequivalent input conditions (20 Hz, generic descriptions)— with ∼35M trainable parameters. We will release the pretrained model checkpoint and codebase to facilitate reproducible research.

## 2 RELATED WORK

IMU-based HAR. Deep learning for HAR has progressed from single-dataset CNN-LSTM models [20] to approaches addressing practical heterogeneity: Cosmo [21] demonstrates contrastive fusion for small-data multimodal HAR, Uni-HAR [35] tackles cross-user and cross-device generalization via physics-informed augmentation, and ClusterFL [22] captures inter-user relationships through federated learning. Despite these advances, all remain architecturally bound to a fixed channel count, sampling rate, and closed-world label set, limiting transferability across sensing setups.

Self-supervised pretraining and foundation models for IMU sensing. Self-supervised learning reduces dependence on labeled IMU data: Contrastive methods learn augmentation-invariant representations [5], while masked autoencoding has been adapted from vision [9] and language [6] to IMU sequences. For example, LiMU-BERT [36] pretrains via span-masked reconstruction at 20 Hz, and CrossHAR [4] combines masked reconstruction with contrastive learning for cross-dataset HAR. PatchTST [19] demonstrates that patch-based channel-independent tokenization improves time-series modeling, a design principle we adopt. However, fixed-length positional encodings constrain variable sampling rates, and all methods require a dataset-specific classifier. Foundation models for general time-series data aim for broader coverage: MOMENT [8] is a T5-style encoder pretrained on 13 time-series domains via masked patch reconstruction (341.2M parameters), and UniMTS [38] targets unified pretraining for motion time series across heterogeneous sensor configurations. In the tabular domain, Mitra [37] demonstrates that curated mixtures of synthetic priors can train foundation models that generalize across diverse real-world datasets via in-context learning. These general-purpose approaches trade domain-specific in ductive biases for breadth; HALO occupies an intermediate position as a domain-specific foundation model for wearable sensing with HAR-specific designs and a language-based label interface.

Language-grounded perception. CLIP [23] demonstrated that aligning vision and text in a shared embedding space enables open-vocabulary zero-shot classification without per-class classifiers. ImageBind [7] extends this to multiple modalities including IMU, but is not designed for cross-dataset HAR or variable sensor configurations. Lan HAR [13] applies CLIP-style alignment to HAR by training a sensor Transformer against LLM-generated activity descriptions. LLaSA [11] takes an alternative approach, leveraging LLMs directly for natural language reasoning about IMU data. Unlike LanHAR, which lacks self-supervised pretraining and assumes fixed sensor configurations, HALO integrates heterogeneity-aware pretraining with synonymaware contrastive alignment, enabling a capability none of these prior systems achieve: a single deployable model that handles variable sampling rates, channel counts, and sensor placements for open-set HAR.

## 3 MOTIVATION

We identify two fundamental challenges that prevent current IMU-based HAR models from serving as general-purpose foundations: sensing heterogeneity across devices and deployments, and limited generalization to unseen activity distributions. We analyze each challenge through the lens of real-world applications and derive the design requirements that motivate HALO.

![](images/f375a6c3b1db6d25e34e25c30064f8efcae61a6a3e72ba204847b8a9e7dc51c7.jpg)  
Figure 2: Dataset heterogeneity across 14 HAR datasets in our benchmark. Training datasets (blue circles) and test datasets (red triangles) span 20–100 Hz in sampling rates, 3–45 channels, and 4–19 activity labels.

## 3.1 Sensing Heterogeneity Across Devices and Deployments

Wearable and mobile devices difer widely in their sensing configurations, leading to substantial data heterogeneity across HAR datasets. For example, a smartwatch with a 3- axis accelerometer at 50 Hz produces significantly diferent raw data than a multi-sensor body suit capturing 51 channels at $1 0 0 \mathrm { H z } ,$ even when both record the same physical activity. To systematically analyze such heterogeneity, we analyze the 17 datasets included in Table 1 of our benchmark. Figure 2 visualizes the distributions of the key characteristics of these datasets.

Sampling rate variation. Sampling rates range from 20 Hz to 100 Hz (Figure 2), so a 4-second window contains 80 to 400 timesteps depending on the device. A fixed positional encoding tied to timestep index conflates physical time with sequence length, causing identical motions at diferent rates to appear structurally diferent.

Channel and placement diversity. Channel counts range from 3-axis wrist accelerometers to 51-channel multi-IMU body suits. A fixed-width input layer cannot accommodate this variability without padding artifacts or datasetspecific input heads.

## 3.2 Limited Generalization to New Distributions and Activities

Beyond sensing heterogeneity, existing approaches share a structural limitation: they produce representations consumed by a fixed, closed-world classifier. This is adequate within a single dataset but fails in two important transfer scenarios.

Label vocabulary inconsistency. Label vocabularies differ in both naming and granularity: “walking,” “level walking,” “walking on a flat surface,” and “normal walk” all denote the same activity across diferent datasets, while “sport” and “rope skipping” represent very diferent levels of specificity. Because each research group defines its own taxonomy independently, standard one-hot supervision treats “jogging” and “running” as maximally diferent, producing contradictory gradients when both labels appear in the training set for similar motions.

The first is label shift: a model trained on ten datasets with a combined 87 unique activity labels is never exposed to the specific label strings used in a held-out test set, even when the underlying motions overlap. For classifier-based models, this means the output vocabulary cannot cover the test labels, and synonym mismatches (e.g., predicting “jogging” when the ground truth is “running”) count as errors even when semantically correct. The second is genuinely novel activities: our test datasets include activities with no semantic analog in the training label set (e.g., car\_step\_in in MobiAct, industrial activities in VTT-ConIoT), making correct predictions structurally impossible for any closed-world model.

Text-aligned models can avoid the closed-world limitation: inference is cosine-similarity retrieval over an arbitrary text library, so the candidate set can be changed at runtime without retraining. However, efective text alignment re quires both (i) a sensor encoder robust to the heterogeneity described above, and (ii) a training objective that handles synonym variation in label strings rather than treating them as distinct classes.

## 3.3 Design Requirements

The analysis above yields three concrete requirements for a cross-device IMU model: (R1) the model must handle arbitrary sampling rates, channel counts, and sensor placements without dataset-specific architectural changes; (R2) it must support open-vocabulary recognition over arbitrary label sets at test time, including labels not seen during training; and (R3) it must align semantically equivalent labels during training to avoid contradictory gradients from synonym variation. HALO achieves R1 through heterogeneity-aware pretraining (Section 5), and R2–R3 through synonym-aware IMU-to-text contrastive alignment (Section 5).

## 4 SYSTEM OVERVIEW

HALO is an IMU-to-text semantic alignment model trained across heterogeneous HAR datasets for open-set, zero-shot activity recognition. The system addresses the design requirements identified in Section 3 through a two-stage training framework. Figure 3 illustrates the end-to-end pipeline.

![](images/359a01f71cc17e1cbfaa8916ec34fbfd722a3c0a4b32c98b519de8994620fc38.jpg)  
Figure 3: System overview of HALO. The sensing heterogeneity is addressed through heterogeneity-aware pretraining of the IMU encoder (Stage 1), followed by synonym-aware soft contrastive learning to align sensor features with text semantics (Stage 2), enabling generalizable and open-set activity recognition. Zeroshot inference is achieved by swapping label banks at runtime without retraining.

Stage 1: Heterogeneity-aware pretraining. The IMU encoder maps variable-rate, variable-channel sessions to a fixed embedding $\mathbf { z } _ { \mathrm { i m u } } \in \mathbb { R } ^ { D }$ through adaptive-pooling tokenization, channel-independent processing, and contextualized sensor conditioning, then is pretrained via masked autoencoding and patch-level contrastive learning (Section 5.1).

Stage 2: Synonym-aware IMU-to-text alignment. The pretrained encoder is aligned with a frozen Sentence-BERT text encoder via synonym-aware soft contrastive learning that derives soft target distributions from text–text similarity, resolving label conflicts across datasets and mapping both modalities into a shared 384-dimensional space (Section 5.2).

Inference. At test time, recognition is cosine-similarity retrieval: given $\mathbf { z } _ { \mathrm { i m u } } .$ we select the label embedding with maximum similarity, supporting arbitrary candidate inventories including open-vocabulary settings. No per-dataset classifier is needed. The full model contains ∼35M trainable parameters (8-layer dual-branch IMU encoder ∼19M, alignment and projection heads ∼16M) plus a single frozen Sentence-BERT backbone (all-MiniLM-L6-v2, 22.7M) shared for both channel conditioning and text alignment. At deployment, the sensor description string is specified once when the app is configured; the corresponding text embeddings are computed once and cached, adding zero per-inference cost. Likewise, all candidate label embeddings can be precomputed ofline.

## 5 DESIGN OF HALO

This section details HALO’s two stages: the heterogeneityaware encoder with self-supervised pretraining (Stage 1, Section 5.1) and synonym-aware contrastive text alignment (Stage 2, Section 5.2).

## 5.1 Heterogeneity-aware Pretraining

The encoder addresses the three heterogeneity challenges identified in Section 3 through dedicated components (Figure 4). At inference time, a sample flows through them in sequence: tokenization produces per-channel tokens H<sub>0</sub>, sensor conditioning enriches them to H<sup>′</sup>, and the channelindependent encoder fuses them into the session embedding $\mathbf { z } _ { \mathrm { i m u } }$ . We present each component below, grouping the encoder with its fusion mechanism before describing conditioning, since the encoder architecture is easier to understand as a self-contained unit.

5.1.1 Adaptive-Pooling Tokenization for Variable Sampling Rates. HAR datasets are collected at diferent native sampling rates—for example, PAMAP2 [25] records at 100 Hz while Shoaib [28] uses 50 Hz and WISDM [12] operates at 20 Hz. A conventional fixed-length tokenizer that expects � samples per patch would either require resampling every dataset to a common rate (distorting high-frequency content or fabricating samples) or training separate models per rate. Rather than resampling, HALO segments sessions in seconds rather than samples, so every patch spans the same physical duration regardless of rate. The resulting variable-length patches are then mapped to fixed-dimension tokens via adaptive pooling, decoupling the entire encoder from any specific sampling rate.

Seconds-based patching. Each session is segmented into fixed-duration patches (dataset-specific, typically 0.75–2.0 s), yielding X $\in \bar { \mathbb { R } } ^ { \bar { B } \times P \times L \times \dot { C } }$ where � is the native patch length and � the channel count. A 1-second patch at 50 Hz contains �=50 samples while the same patch at 100 Hz contains �=100; the variable � is absorbed downstream by adaptive pooling rather than resampling. During training, patch durations are randomly sampled from a dataset-specific range (e.g., 0.75– 1.25 s) as a temporal augmentation; augmentation is disabled at evaluation.

![](images/9fa2a7771a54a07e430ab941a22f5e289800d7162affb03d981fa94ef17d47dc.jpg)  
Figure 4: Heterogeneity-aware Pretraining. Each row shows a challenge (left) and the corresponding design in HALO (right): adaptive-pooling tokenization for variable sampling rates; channel-independent feature extraction for variable channels; and contextualized sensor conditioning for variable sensor placements.

Per-patch, per-channel normalization. Each patchchannel pair is independently z-score normalized:

$$
\hat { \mathbf { X } } _ { b , p , : , c } = \frac { \mathbf { X } _ { b , p , : , c } - \mu _ { b , p , c } } { \sigma _ { b , p , c } + \epsilon } ,
$$

where $\mu _ { b , p , c }$ and $\sigma _ { b , p , c }$ are computed over the � timesteps of patch (�, �) channel �. This local normalization removes device- and subject-specific amplitude biases without requiring global statistics.

Spectral-temporal tokenizer. Each normalized patchchannel pair is converted into a �-dimensional token (�=384) via a hybrid feature extractor with shared weights across channels. The temporal branch applies a multi-scale 1-D CNN (kernel sizes 3, 5, 7) followed by adaptive average pooling to produce a fixed-length output regardless of $L ;$ the spectral branch computes a fixed-� FFT magnitude spectrum and projects it through a learned MLP. Combining both branches captures complementary time-domain shape and frequency-domain periodicity cues. The outputs are concatenated to form the initial token tensor $\mathbf { H } _ { 0 } \in \mathrm { \overline { { R } } } ^ { B \times P \times C \times D }$

Because both branches use adaptive pooling, the tokenizer natively accepts any sampling rate without resampling or architectural changes—a single trained model can process 20 Hz smartphone data and 100 Hz clinical IMU data through the same weights. This is a key advantage over fixed-length tokenizers such as PatchTST [19], which require a predetermined number of samples per patch and therefore either resample all inputs to a common rate (distorting highfrequency content at downsampled rates and fabricating samples at upsampled rates) or maintain separate tokenizer configurations per dataset. Patch-size sensitivity analysis (Section 6.4) confirms that the fixed 1.0 s default is within 1.3 pp of per-dataset optima across the full 20–100 Hz range.

5.1.2 Channel-Independent Feature Extraction for Variable Channel Counts. Diferent datasets expose vastly different sensor suites: WISDM [12] provides a single 3-axis accelerometer (�=3), while DSADS [3] uses 45 channels from accelerometers, gyroscopes, and magnetometers at five body locations. A fixed-width architecture would either waste capacity on small datasets or fail entirely on larger ones. To handle this, HALO treats every channel identically: the tokenizer, Transformer, and fusion modules all share parameters across channels, and variable-length channel dimensions are zero-padded with binary masks propagated to every attention and pooling layer so that padded channels never contribute to computation.

Dual-branch Transformer. The token tensor—after sensor conditioning (Section 5.1.3, below)—is processed by $L _ { e }$ Transformer layers $\left( L _ { e } { = } 8 \right)$ , each alternating two attention stages: (1) temporal self-attention over patches within each channel (reshape to (��, �, �)), which captures how each individual sensor signal evolves over time, and (2) cross-channel self-attention over channels within each patch (reshape to (��, �, �), padded channels masked), which captures correlations between diferent sensors at the same time step. Each stage is followed by a shared FFN, yielding H $\in \mathbb { R } ^ { B \times \mathbf { \tilde { P } } \times C \times D }$ Sinusoidal positional encodings [33] with a learnable scale parameter inject temporal order over the patch dimension. The alternating design allows the model to first build per sensor temporal representations and then exchange information across sensors at every layer, progressively building richer multi-sensor features.

Cross-channel fusion. For each patch, a small set of learnable queries attends over the $C$ channel tokens via multi-query attention, producing a fused patch sequence $\textbf { U } ~ \in ~ \bar { \mathbb { R } } ^ { B \times \bar { P } \times D }$ . Channel masks prevent padded channels from influencing attention. This reduces the variable-width channel dimension to a fixed-width representation, enabling downstream modules to operate regardless of the original sensor count.

Temporal pooling and projection. A lightweight Transformer over U followed by a second multi-query attention module pools the patch axis to yield $\mathbf { v } \in \mathbb { R } ^ { B \times D }$ , which an MLP projection head maps to the session embedding $\mathbf { z } _ { \mathrm { i m u } } \in \mathbb { R } ^ { B \times D _ { s } }$ $( D _ { s } { = } 3 8 4$ , ℓ<sub>2</sub>-normalized). HALO also supports per-patch prediction: each fused patch token $\mathbf { U } _ { b , p }$ is independently projected and scored against text labels via cosine similarity, with per-patch predictions aggregated by majority vote. All zero-shot results in this paper use per-patch majority voting; supervised fine-tuning uses the pooled global embedding z<sub>imu</sub>.

The channel-independent design means the same trained weights handle 3-channel smartphone data and 45-channel full-body sensor suites with no architectural changes across all 17 datasets. At training time, channels from diferent datasets are mixed within each batch, exposing the encoder to diverse sensor types and body locations in every gradient step. The shared-parameter approach also acts as an implicit regularizer: the model cannot memorize channel-specific shortcuts, encouraging generalizable motion features that transfer across sensor configurations.

5.1.3 ContextualizedSensorConditioningforVariable Sensor Placements. Because the tokenizer shares weights across all channels, the tokens $\mathbf { H } _ { 0 }$ carry temporal and spectral information but are agnostic to where on the body each channel was recorded or what type of sensor produced it. A wrist accelerometer X-axis and a waist gyroscope Z-axis may produce similar waveforms for diferent activities—for example, arm-swing acceleration during walking looks similar from a wrist accelerometer and a hip accelerometer, yet the two carry diferent biomechanical meaning. Without placement context, the encoder cannot disambiguate them. HALO addresses this by injecting sensor placement and modality context into each channel token via natural-language descriptions, using the same frozen text encoder that will later be used for activity-label alignment. This converts the heterogeneity problem into a representational advantage: the model learns that “Accelerometer X-axis, waist-mounted smartphone” and “Accelerometer X-axis, wrist-mounted smartwatch” are related but distinct, leveraging the semantic structure of natural language.

Concretely, each channel token $\mathrm { H } _ { 0 , b , p , c }$ is conditioned on a per-channel semantic description (e.g., “Accelerometer X-axis, waist-mounted smartphone”). A frozen Sentence-BERT backbone [24] (all-MiniLM-L6-v2, 384-dim)—shared with the text encoder used in Stage 2 alignment—encodes the description, and a learned two-layer residual projection with ReLU maps it to ${ \bf e } _ { c } \in \mathbb { R } ^ { D }$ . The conditioned token is: $\mathbf { H } _ { b , p , c } ^ { \prime } = \mathbf { H } _ { 0 , b , p , c } + \gamma \mathbf { e } _ { c } ,$ , where � is a learnable scale initialized to 0.1 so that conditioning acts as a gentle bias early in training and grows as the model learns to exploit it. The conditioned tokens H<sup>′</sup> are then passed to the dual-branch Transformer (Section 5.1.2).

Descriptions are constructed per dataset from a simple template combining modality, axis, and placement metadata (e.g., “Accelerometer X-axis, waist-mounted smartphone”). Each dataset’s channel descriptions are specified once during data preparation and remain fixed throughout training and evaluation. For datasets without placement annotations (e.g., UCI-HAR provides only “smartphone” without specifying pocket vs. hand), we use the available metadata; datasets with no sensor metadata fall back to generic descriptions $( \mathrm { e . g . }$ , “IMU channel”). At inference, the model leverages rich descriptions when available and falls back to generic ones otherwise. Sensor conditioning is the single largest contributor to zero-shot transfer (−19.6 pp ZS-O when removed;

Table 6); even with generic descriptions HALO outperforms baselines on 3 of 4 settings (Table 7).

5.1.4 Self-supervised Pretraining. Before alignment with text, the encoder must learn general-purpose IMU representations from unlabeled data across all 10 training datasets. The dificulty is that heterogeneous sensor configurations and activities can cause the encoder to overfit to dataset-specific artifacts (e.g., amplitude ranges, noise profiles) rather than learning transferable motion patterns. Stage 1 addresses this by training the encoder end-to-end with two complementary self-supervised objectives—masked autoencoding for local signal reconstruction and patch-level contrastive learning for global discriminability—combined with signal augmentation that forces the model to generalize across noise conditions.

Masked autoencoding. A fixed fraction of patches (mask ratio 0.5) are replaced by a learned mask token, and the temporal encoder reconstructs the normalized input signal at masked positions:

$$
\mathcal { L } _ { \mathrm { m a e } } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { ( b , p ) \in \mathcal { M } } \left. \hat { \mathbf { u } } _ { b , p } - \mathbf { u } _ { b , p } \right. _ { 2 } ^ { 2 } .
$$

This objective encourages the encoder to capture finegrained temporal patterns within each channel, such as the periodic swing phase in walking or the abrupt deceleration in sitting down. The 50% mask ratio is chosen to balance reconstruction dificulty: lower ratios make the task too easy (the encoder can interpolate from nearby unmasked patches), while higher ratios deprive the model of suficient context for meaningful reconstruction.

Patch-level contrastive learning. Two augmented views of each patch (jittering, scaling, time warping) are treated as positives and other in-batch patches as negatives under an InfoNCE objective [32] with temperature 0.2. This objective encourages patches from similar motions to cluster in embedding space while pushing dissimilar motions apart, building discriminative structure before any labels are introduced. Both objectives are combined with equal weights $( \lambda _ { \mathrm { m a e } } = \lambda _ { \mathrm { c o n } } = 1 . 0 )$

Together, the dual objectives produce an encoder that captures both local signal fidelity (via MAE) and global motion discriminability (via contrastive learning). Signal augmentation is critical: it closes a 9.5 pp generalization gap invisible on in-distribution validation but evident on held-out datasets (Section 6.3.1).

## 5.2 Synonym-aware IMU-to-text Alignment

A key challenge in multi-dataset IMU-to-text alignment is label synonym conflict: “walking,” “strolling,” and “ambulating” describe the same motion, yet standard contrastive objectives treat them as unrelated negatives, producing contradictory gradients when both appear in a batch. HALO resolves this

![](images/d5caec3baead02907c940e1430dc9cbf2da9027fb0695d735ae84a31e030f54d.jpg)  
Inference: Open-Set Zero-Shot Recognition

Figure 5: IMU-to-text alignment and zero-shot inference. Top: During training, the IMU encoder is aligned with a frozen text encoder in a shared embedding space via synonym-aware contrastive learning, so that semantically related activities are clustered together. Botom: At inference, the IMU segment is encoded and matched against an open label library via cosine similarity-enabling zero-shot recognition without retraining or dataset-specific classifiers.

through synonym-aware label augmentation (Section 5.2.1) and soft contrastive alignment (Section 5.2.2).

5.2.1 Synonym-Aware Label Augmentation. Each dataset defines a curated synonym map (e.g., walking ↔ strolling ↔ ambulating) and natural-language templates (e.g., “person {}”, “a person is {}”). Synonym maps are constructed once per dataset from WordNet and common HAR naming conventions, averaging 2.3 synonyms per activity across the 87-label vocabulary. During each Stage 2 step, with probability 0.8 the original label is replaced by a randomly sampled synonym wrapped in a random template (disabled at evaluation), diversifying the text embeddings seen for each activity and preventing the text encoder from overfitting to exact label strings in any single dataset.

Such label augmentation reduces synonym conflict frequency in each batch (−7.0 pp at 1% when removed; Table 6), but cannot eliminate it entirely—residual synonyms still appear as negatives—motivating the soft targets described next.

5.2.2 IMU-to-Text Alignment with Soft Contrastive Learning. Standard CLIP-style learning uses hard one-hot targets, forcing synonymous labels apart. HALO replaces these with soft target distributions derived from text–text similarity, so synonymous labels receive partial credit—converting synonym conflict from a source of gradient noise into a learning signal that reinforces semantic structure.

Text encoder and contrastive objective. A frozen Sentence-BERT backbone [24] (all-MiniLM-L6-v2, 384-dim)— the same model used for sensor conditioning—encodes augmented activity labels; learnable query-based pooling over its token outputs produces $\mathbf { z } _ { \mathrm { t e x t } } \in \dot { \mathbb { R } } ^ { B \times \dot { D } _ { s } }$ . Stage 2 aligns z<sub>imu</sub> with $\mathbf { z } _ { \mathrm { t e x t } }$ via a CLIP-style bidirectional contrastive objective. Because our batch sizes are limited by GPU memory (microbatch 32 with gradient accumulation), we maintain FIFO queues [10] of size $Q = 2 5 6$ for both modalities to increase the efective number of negatives per step from 32 to 288, pre-filling them with a warmup pass before training begins. Similarity logits are computed as

$$
\mathbf { S } _ { i  t } = \alpha \mathbf { Z } _ { \mathrm { i m u } } \mathbf { A } _ { \mathrm { t e x t } } ^ { \top } , \quad \mathbf { S } _ { t  i } = \alpha \mathbf { Z } _ { \mathrm { t e x t } } \mathbf { A } _ { \mathrm { i m u } } ^ { \top } ,
$$

where A denotes the concatenation of the current batch and queue, and � = exp(logit\_scale) is a learnable temperature (CLIP-initialized, clamped to [1, 50]).

Adaptive soft targets. Raw text–text similarities cluster in a narrow range (0.4–0.9), producing near-uniform targets that collapse the embedding space. We apply adaptive sharpening: z-score-normalize the in-batch similarity matrix $\mathbf { M } = \mathbf { Z } _ { \mathrm { t e x t } } \mathbf { Z } _ { \mathrm { t e x t } } ^ { \top }$ , rescale with temperature $\tau _ { s } { = } 0 . 5 ,$ and softmax to obtain sharpened distributions T (e.g., “walking”/“strolling” share 0.4 of target mass while “walking”/“cycling” share <0.01). Queue entries serve as hard negatives. The blended target P combines soft in-batch targets T with hard one-hot targets for queue entries. The bidirectional loss is:

$$
\begin{array} { r l r } {  { \mathcal L _ { i \to t } = - \frac { 1 } { B } \sum _ { b , j } \mathbf P _ { b , j } \log \operatorname { s o f t m a x } ( \mathbf S _ { i \to t } ) _ { b , j } , } } \\ & { } & { \quad \mathcal L = \frac { 1 } { 2 } \big ( \mathcal L _ { i \to t } + \mathcal L _ { t \to i } \big ) . \quad } \end{array}
$$

Zero-shot inference. At test time, the predicted activity is �ˆ = arg max<sub>�</sub> cos $( \mathbf { z } _ { \mathrm { i m u } } , \mathbf { z } _ { \mathrm { t e x t } } ^ { ( k ) } )$ over an arbitrary candidate label set. Swapping the label library at runtime enables openset recognition without retraining (Section 4).

Embedding-space analysis confirms 76.7% nearest-neighbor accuracy between IMU and text centroids (Section 6.3.3).

## 6 EXPERIMENTS

## 6.1 Methodology

We evaluate whether HALO supports (i) robust performance under heterogeneous multi-dataset training, (ii) open-set and closed-set zero-shot generalization to unseen datasets, and (iii) sample-eficient supervised transfer.

Datasets. We train jointly on ten public HAR datasets (Table 1). All HAR-pretrained models use the same 10 training datasets; MOMENT is pretrained on general time-series data; LLaSA uses published weights whose training data overlaps with two of our test sets (MotionSense, Shoaib). We evaluate on seven held-out datasets never seen during any model’s training: five main test datasets (85–100% label coverage) and two severe out-of-domain datasets reported separately. HALO evaluates at native rates and LanHAR at 50 Hz; the remaining baselines use 20 Hz resampled data via the LIMU-BERT pipeline (Section 6.1).

<table><tr><td>Dataset</td><td>Hz</td><td>#Subj</td><td>#Act</td><td>#Ch</td><td>Description</td><td>Cov. Diff.</td></tr><tr><td>UCI-HAR [1]</td><td>50</td><td>30</td><td>6</td><td>9 IMU</td><td>Smartphone</td><td></td></tr><tr><td>HHAR [30]</td><td>~50</td><td>9</td><td>6</td><td>6 Phone watch</td><td>+</td><td></td></tr><tr><td>UniMiB [18]</td><td>50</td><td>30</td><td>17</td><td>3</td><td>Phone acc, ADLs</td><td></td></tr><tr><td>MHEALTH [2]</td><td>50</td><td>10</td><td>12</td><td>6+ suite</td><td>Body IMU</td><td></td></tr><tr><td>PAMAP2 [25]</td><td>100</td><td>9</td><td>18</td><td>27 Multi- IMU</td><td></td><td></td></tr><tr><td>WISDM [12]</td><td>20</td><td>51</td><td>18</td><td>6 Phone watch</td><td>+</td><td></td></tr><tr><td>DSADS [3]</td><td>25</td><td>8</td><td>19</td><td>45</td><td>5 body po- sitions</td><td></td></tr><tr><td>HAPT [26]</td><td>50</td><td>30</td><td>12</td><td>6 Phone, transi-</td><td></td><td></td></tr><tr><td>KU-HAR [29]</td><td>100</td><td>90</td><td>18</td><td>6</td><td>tions Phone, 18</td><td></td></tr><tr><td>RecGym [39]</td><td>20</td><td>10</td><td>12</td><td>acts 6 Watch, gym</td><td></td><td></td></tr><tr><td>MotionSense† [16]</td><td>50</td><td>24</td><td>6</td><td>6 sic</td><td>Phone, ba-</td><td>100% Easy</td></tr><tr><td>RealWorld† [31]</td><td>50</td><td>15</td><td>8 3</td><td>Phone, acc-only</td><td>100%</td><td>Med</td></tr><tr><td>MobiAct† [34]</td><td>50</td><td>66</td><td>13</td><td>6 Falls ADLs</td><td>+ 85%</td><td>Hard</td></tr><tr><td>Shoaib† [28]</td><td>50</td><td>10</td><td>7</td><td>6 Multi- placement</td><td>100%</td><td>Med</td></tr><tr><td>Opportunity† [27]</td><td>30</td><td>4</td><td>4</td><td>6 XSens body-</td><td>worn</td><td>100% Med</td></tr><tr><td>HARTH [14]</td><td>50</td><td>22</td><td>12</td><td>6</td><td>Back/thigh</td><td>100% Severe</td></tr><tr><td>VTT-ConIoT‡ [15]</td><td>52</td><td>12</td><td>16</td><td>acc 6</td><td>Construction 50%</td><td>Severe</td></tr></table>

Table 1: Dataset overview. Top: 10 training datasets. Bottom: 7 held-out test datasets. † Main test datasets. ‡ Severe OOD datasets; reported separately. Cov.: Fraction of test activities that have an equivalent in the training set. Dif.: Dificulty level determined by distribution shift and label novelty.

Baselines. We compare HALO against five baselines representing diferent pretraining paradigms (Table 2):

Text-aligned models (HALO, LanHAR) predict via cosine similarity between sensor and text embeddings, enabling arbitrary label strings at test time (evaluated by exact string match). LanHAR fine-tunes SciBERT on activity text, then aligns a sensor Transformer. Classifier-based models (LiMU-BERT, MOMENT, CrossHAR) train a fixed 87-class classifier over the global training labels. LiMU-BERT uses spanmasked reconstruction with a GRU head; MOMENT is a T5-style encoder pretrained on 13 time-series domains, processing each channel independently (6×1024 = 6144-dim); CrossHAR combines masked reconstruction with contrastive regularization. For zero-shot evaluation, each uses its paper’s recommended classifier on training-set embeddings only— the term zero-shot means the test dataset is unseen at both training and classifier-fitting stages. These models benefit from group-based scoring but cannot predict unseen label strings. Generative language model (LLaSA) combines a LiMU-BERT encoder with Vicuna-7B and classifies by prompting the LLM with sensor tokens. These asymmetries are inherent to the architectures; where we deviated from a baseline’s protocol, we erred in that baseline’s favour (Section 6.1). Full adaptation details are in the supplementary material.

<table><tr><td>Model</td><td>Paradigm</td><td>Dim</td><td>Train.</td><td>Infer.</td></tr><tr><td>LiMU-BERT [36]</td><td>Masked recon.</td><td>72</td><td>72.7K</td><td>72.7K</td></tr><tr><td>MOMENT [8]</td><td>General TS</td><td>6144</td><td>341.2M</td><td>341.2M</td></tr><tr><td>CrossHAR [4]</td><td>Self-sup HAR</td><td>72</td><td>531.4K</td><td>531.4K</td></tr><tr><td>LanHAR [13]</td><td>Text-aligned</td><td>768</td><td>13.0M*</td><td>122.9M</td></tr><tr><td>LLaSA [11]</td><td>LLM generative</td><td>一</td><td>~6.7B</td><td>~6.7B</td></tr><tr><td>HALO (ours)</td><td>Text-aligned</td><td>384</td><td>~35M</td><td>~58M</td></tr></table>

Table 2: Baseline overview. Train.: parameters updated during pretraining and alignment. Infer.: Total parameters used at inference, including frozen modules. HALO’s inference total includes a single frozen MiniLM (22.7M) shared for both contrastive alignment and channel conditioning; all text embeddings are precomputable. <sup>∗</sup>LanHAR freezes SciBERT (109.9M) during stage-2 and unfreezes it for fine-tuning.

Evaluation protocol. To handle cross-dataset label inconsistency (e.g., “jogging” vs. “running”), we define synonym groups mapping variant strings to canonical activities; classifier-based models benefit from group-based scoring, while text-aligned models use exact string match. Novel test activities with no training equivalent form irreducible error floors. We report accuracy and macro F1 for all settings. Zero-shot open-set: predict from the full 87-label training vocabulary (HALO uses per-patch majority voting; classifierbased models use 87-class logits). Zero-shot closed-set: restrict predictions to test-dataset labels. 1%/10% supervised: finetune end-to-end on 1% or 10% of labeled test data (balanced subsampling, 80/10/10 split, seed 3431; 20 epochs, AdamW, encoder LR 10<sup>−5</sup>, head LR $1 0 ^ { - 3 }$ , batch 32, patience 5).

Fairness and input policy. Each model receives the richest input its architecture can consume. HALO evaluates at native rates with rich channel descriptions; LanHAR at 50 Hz; LiMU-BERT, CrossHAR, MOMENT, and LLaSA use the LIMU-BERT preprocessing pipeline (20 Hz, 120-step, 6- channel windows); MOMENT additionally left-pads each channel to its expected 512 timesteps per its published protocol. Conversely, MOMENT benefits from 341M parameters and 6144-dim embeddings, and classifier-based models ben efit from group scoring. To ensure a fair comparison: (i) all models receive identical windows, labels, and splits; (ii) each baseline uses its recommended classifier; (iii) where we deviated from a baseline’s original protocol, we chose the option favouring that baseline (e.g., unfreezing CrossHAR, giving

<table><tr><td rowspan="2">Model</td><td colspan="2">ZS-O</td><td colspan="2">ZS-C</td><td colspan="2">1%</td><td colspan="2">10%</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>HALO (ours)</td><td>42.0</td><td>21.0</td><td>53.1</td><td>37.2</td><td>76.5</td><td>70.0</td><td>85.7</td><td>81.7</td></tr><tr><td>MOMENT</td><td>28.3</td><td>7.2</td><td>44.7</td><td>33.2</td><td>73.8</td><td>69.8</td><td>81.3</td><td>76.8</td></tr><tr><td>CrossHAR</td><td>21.2</td><td>5.5</td><td>38.2</td><td>31.7</td><td>61.1</td><td>55.2</td><td>78.6</td><td>75.2</td></tr><tr><td>LiMU-BERT</td><td>21.7</td><td>7.5</td><td>33.1</td><td>23.1</td><td>35.7</td><td>22.7</td><td>51.3</td><td>43.6</td></tr><tr><td>LanHAR</td><td>15.8</td><td>6.2</td><td>28.4</td><td>21.7</td><td>43.3</td><td>35.2</td><td>56.2</td><td>51.5</td></tr><tr><td>LLaSA†</td><td>1.4</td><td>1.2</td><td>17.4</td><td>8.4</td><td>一</td><td></td><td></td><td></td></tr></table>

Table 3: Overall performance in average accuracy (%) and macro F1 (%) across 5 main test datasets. Bold: best result for each metric. ZS-O/ZS-C: zero-shot openset/closed-set settings. HALO and LanHAR are evaluated at each dataset’s native sampling rate; other baselines use data resampled to 20 Hz. <sup>†</sup>LLaSA: published 7B model, zero-shot only (no supervised)

MOMENT end-to-end fine-tuning, 10× LLaSA’s per-class budget); (iv) classifier-based models use group scoring, while text-aligned models use exact string match. We report exact match for text-aligned models because open-vocabulary deployment depends on raw label-string retrieval; synonymgroup scoring would not reflect real deployment accuracy. As a direct fairness control, Section 6.3.2 evaluates HALO under baseline-equivalent inputs (20 Hz, generic descriptions) and shows it still leads on 3 of 4 settings (Table 7).

Implementation details. Training proceeds in two stages with a single model jointly across all 10 datasets. Stage 1 pretrains the IMU encoder (Section 5.1.4) for 100 epochs (batch 20, AdamW, $\mathrm { L R } ~ 1 0 ^ { - 4 }$ , weight decay ${ { 1 0 } ^ { - 5 } }$ 10 warmup epochs, MAE mask ratio 0.5, contrastive temperature 0.2). Stage 2 initializes from the pretrained encoder and trains for 200 epochs with 70/15/15 splits per dataset, mixed precision, and gradient accumulation (micro-batch 32, accumulation 16). Key hyperparameters: learned temperature $\scriptstyle \alpha _ { 0 } = 0 . 0 7$ , soft-target temperature $\tau _ { s } { = } 0 . 5$ , memory bank �=256. Group-balanced sampling mitigates class imbalance (weights inversely proportional to group frequency, capped at 20×). HALO uses a fixed 1.0 s evaluation patch size for all datasets with no per-dataset tuning; sensitivity is analyzed in Section 6.4. All training is performed on a single NVIDIA A100 80 GB GPU; on-device inference profiling uses a smartphone (Section 6.5).

## 6.2 Overall Performance

6.2.1 Overall performance in diferent setings. Table 3 summarizes average performance. HALO leads on all eight metrics; notably, these gains hold even under baselineequivalent inputs (20 Hz, generic descriptions), as confirmed in the fairness control (Table 7). The largest margin is in zero-shot open-set (42.0% vs. MOMENT’s 28.3%, +13.7 pp), where predicting from 87 candidates without test-set knowledge is hardest. Even at 1% supervision, HALO leads despite MOMENT’s 6144-dim embeddings (16× HALO’s 384-dim) giving its classifier substantially more capacity. LanHAR underperforms despite being text-aligned and evaluating at native 50 Hz—likely because its sensor encoder lacks selfsupervised pretraining.

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Accuracy (%)</td><td colspan="3">Macro F1 (%)</td></tr><tr><td>ZS-O</td><td>ZS-C</td><td>1%</td><td>10%</td><td>ZS-O ZS-C</td><td>1%</td><td>10%</td></tr><tr><td>MotionSense</td><td>49.3</td><td>64.1</td><td>88.6</td><td>95.0</td><td>30.3</td><td>49.3</td><td>87.1 94.6</td></tr><tr><td>RealWorld</td><td>48.0</td><td>48.0</td><td>75.1</td><td>85.6</td><td>33.3</td><td>37.9</td><td>74.0 86.4</td></tr><tr><td>MobiAct</td><td>42.2</td><td>50.0</td><td>65.3</td><td>72.9</td><td>16.3</td><td>12.8</td><td>35.2 51.7</td></tr><tr><td>Shoaib</td><td>49.2</td><td>54.1</td><td>81.6</td><td>95.9</td><td>17.6</td><td>51.5</td><td>81.2 95.7</td></tr><tr><td>Opportunity</td><td>21.1</td><td>49.3</td><td>72.0</td><td>79.3</td><td>7.4</td><td>34.2</td><td>72.6 80.0</td></tr><tr><td>Average</td><td>42.0</td><td>53.1</td><td>76.5</td><td>85.7</td><td>21.0</td><td>37.2</td><td>70.0 81.7</td></tr></table>

Table 4: Per-dataset results of HALO on the five main test datasets. ZS-O/ZS-C: zero-shot open/closed-set. All datasets are held out from training.

<table><tr><td rowspan="2"></td><td colspan="4">HARTH</td><td colspan="4">VTT-ConIoT</td></tr><tr><td>ZS-O</td><td>ZS-C</td><td>1%</td><td>10%</td><td>ZS-O</td><td>ZS-C</td><td>1%</td><td>10%</td></tr><tr><td>HALO</td><td>2.0</td><td>2.5</td><td>64.7</td><td>78.3</td><td>1.3</td><td>2.2</td><td>1.9</td><td>16.9</td></tr><tr><td>MOMENT</td><td>0.4</td><td>1.9</td><td>66.7</td><td>65.4</td><td>1.6</td><td>5.2</td><td>18.4</td><td>34.8</td></tr><tr><td>CrossHAR</td><td>0.3</td><td>0.9</td><td>42.4</td><td>71.9</td><td>0.7</td><td>5.0</td><td>10.6</td><td>30.9</td></tr><tr><td>LiMU-BERT</td><td>0.0</td><td>20.2</td><td>54.4</td><td>19.1</td><td>3.4</td><td>7.1</td><td>4.3</td><td>15.0</td></tr><tr><td>LanHAR</td><td>0.9</td><td>1.1</td><td>3.4</td><td>70.0</td><td>8.3</td><td>6.9</td><td>7.7</td><td>7.7</td></tr><tr><td>LLaSA†</td><td>11.9</td><td>12.5</td><td>一</td><td></td><td>一</td><td>6.9</td><td>一</td><td>一</td></tr></table>

Table 5: Accuracy on severe out-of-domain datasets (%). ZS-O/ZS-C: zero-shot open/closed-set. All models collapse on HARTH zero-shot (sensor distribution shift); VTT-ConIoT is capped at ∼50% by novel activities. <sup>†</sup>LLaSA: zero-shot only, subsampled.

6.2.2 Results in each dataset. HALO leads zero-shot on all five datasets except Opportunity closed-set, where MOMENT edges ahead (53.9% vs. 49.3%), possibly because its 6144- dim SVM-RBF classifier benefits from the small label set. MobiAct—the hardest with 13 classes including falls—shows the largest zero-shot gap (HALO 42.2% vs. MOMENT 28.7% open-set). At 10% supervision, HALO leads on all 5 datasets; at 1%, HALO leads on 4 of 5 (MOMENT edges ahead only on Opportunity: 74.3% vs. 72.0%).

6.2.3 Severe out-of-domain transfer performance. Table 5 reports HARTH and VTT-ConIoT separately. HARTH uses back/thigh-mounted accelerometers only (no gyroscope), with gravity-contaminated data that difers substantially from training; despite 100% label coverage, this sensor shift causes all models to collapse in zero-shot. VTT-ConIoT is an industrial dataset where only 50% of activities have training equivalents; the rest (e.g., kneeling work, roll painting) are entirely novel. On HARTH, all embedding-based models achieve near-zero zero-shot accuracy (<3%), confirming genuine sensor distribution shift rather than a model-specific failure. LLaSA’s generative reasoning is less sensitive to signal distribution (11.9% open-set), but still far from useful. With supervised data, HALO leads at 10% (78.3% vs. CrossHAR’s 71.9%); at 1%, MOMENT edges ahead (66.7% vs. 64.7%). On VTT-ConIoT, all models score near random on zero-shot (<9%). With supervised data, MOMENT leads (34.8% at 10%), benefiting from its general time-series pretraining; HALO’s HAR-specific encoder provides strong in-domain transfer but less benefit when the target domain is fundamentally diferent.

<table><tr><td>Configuration</td><td>ZS-0</td><td>ZS-C</td><td>1%</td><td>10%</td></tr><tr><td>Full model</td><td>44.9</td><td>51.9</td><td>74.7</td><td>85.4</td></tr><tr><td>– Channel-text fusion</td><td>25.3 (-19.6)</td><td>43.8 (-8.1)</td><td>72.9 (-1.8)</td><td>82.5 (−2.9)</td></tr><tr><td>– Signal augmentation</td><td>45.3 (+0.4)</td><td>53.0 (+1.1)</td><td>70.9 (-3.8)</td><td>83.5 (−1.9)</td></tr><tr><td>– Text augmentation</td><td>42.7 (−2.2)</td><td>49.9 (−2.0)</td><td>67.7 (-7.0)</td><td>84.2 (−1.2)</td></tr></table>

Table 6: Ablation study of design components. Each row removes one component while keeping all others fixed. Results report average accuracy (%) across five main test datasets, with performance deltas relative to the full model shown in parentheses.

## 6.3 Understanding HALO’s Performance

6.3.1 Ablation study on design components. We isolate three architectural and training contributions by disabling each independently from the full model.<sup>2</sup> Table 6 reports 5-dataset averages. Channel-text fusion (i.e., sensor conditioning, Section 5.1.3) is the single most important zero-shot component (−19.6 pp ZS-O), as the model must otherwise infer sensor semantics purely from signal patterns; the efect fades with supervision (−2.9 pp at 10%). Text augmentation has the largest supervised impact (−7.0 pp at 1%), preventing the text encoder from memorizing exact label strings. Signal augmentation closes a 9.5 pp generalization gap invisible on in-distribution validation but critical on held-out datasets; on severe OOD the efect strengthens to −6.8 pp at 10%.

6.3.2 Analysis on native sampling rate and fairness. We compare HALO at native rates with rich descriptions against the same model on 20 Hz resampled data with generic channel names (e.g., “acc\_x”), matching the LIMU-BERT pipeline used by four of five baselines. Table 7 shows that even under these conservative inputs, HALO leads on 3 of 4 settings (+2.3 pp ZS-O, +4.7 pp ZS-C, +3.4 pp at 10% over MOMENT); only at 1% does MOMENT edge ahead. The native-rate advantage is dataset-dependent: MotionSense and MobiAct show large zero-shot gains (+24.5 and +19.7 pp) where rich channel text aids placement disambiguation, while RealWorld drops (−9.9 pp) because its heterogeneous body placements are poorly captured by a single dataset-level description. Supervised performance is less afected (within 3 pp at 10%).

<table><tr><td>Configuration</td><td>ZS-0</td><td>ZS-C</td><td>1%</td><td>10%</td></tr><tr><td>HALO (native Hz, rich desc.) HALO (20 Hz, generic desc.)</td><td>42.0 30.6</td><td>53.1 49.4</td><td>76.5</td><td>85.7</td></tr><tr><td>Best baseline (MOMENT)</td><td>28.3</td><td>44.7</td><td>71.7 73.8</td><td>84.7 81.3</td></tr></table>

Table 7: Analysis of native sampling rate and evaluation fairness with average accuracy (%) across five main test datasets. In Row 2, HALO is evaluated using the same 20Hz, 6-channel inputs with generic textual descriptions as four of the five baselines (LanHAR uses 50Hz data). Even under these conservative and standardized inputs, HALO outperforms baselines in three of the four evaluation settings.

![](images/db2fb70d13597a8e1f6d0f1a4784f3f05d5e41f783ac3312f97e039f9b63a064.jpg)  
Figure 6: Analysis of embedding space. UMAP projection of learned IMU and text embeddings at epoch 100. Circles: per-activity IMU centroids; diamonds: SBERT text centroids; lines connect matched IMU–text pairs.

6.3.3 Analysis of embedding space. Figure 6 visualizes the learned embedding space via UMAP [17]. Semantically related groups (e.g., running, stairs, walking) cluster together with tight cross-modal alignment: nearest-neighbor accuracy reaches 76.7%, positive similarity 0.857, and the similarity gap between matched and unmatched pairs is 0.377. Stationary activities form a distinct cluster separated from locomotion, mirroring kinematic structure. Text centroids align closely with IMU counterparts, confirming the contrastive objective maps both modalities into a shared semantic space.

![](images/bb65394dbd4318ddd00f0fdc2d6563ca57244f0bf27f873a0df5a01bd8779646.jpg)

Figure 7: Patch size sensitivity. Zero-shot closed-set accuracy (%) across candidate patch durations. The dashed vertical line indicates the selected patch size of 1.0 s. Original datasets are evaluated at 20 Hz, while Shoaib and Opportunity are evaluated at their native sampling rates.
<table><tr><td>Model</td><td>Params</td><td>ZS-0</td><td>ZS-C</td><td>1%</td><td>10%</td></tr><tr><td>Tiny</td><td>6M</td><td>32.8</td><td>52.1</td><td>72.6</td><td>86.4</td></tr><tr><td>Small</td><td>35M</td><td>46.0</td><td>53.4</td><td>77.5</td><td>85.3</td></tr><tr><td>Medium</td><td>63M</td><td>37.8</td><td>47.7</td><td>72.6</td><td>84.5</td></tr></table>

Table 8: Scaling of model size. Average accuracy (%) on 3 main test datasets at 20 Hz. The Small model (35M) achieves the best performance, while the Medium model regresses on all zero-shot metrics despite having twice as many parameters.

## 6.4 Patch Size Sensitivity

HALO uses a fixed 1.0 s patch size for all results—no perdataset tuning. Figure 7 shows that 1.0–1.25 s patches consistently perform best; the fixed 1.0 s choice is within 1.3 pp of the per-dataset optimum across all five datasets. This range aligns with typical gait cycle durations (0.8–1.2 s), capturing approximately one full motion cycle per patch. Shorter patches (0.75 s) degrade on complex-activity datasets like Opportunity, likely splitting compound activities across patch boundaries; longer patches (1.5–2.0 s) show diminishing returns as they average over multiple motion phases. The sensitivity curve is relatively flat between 0.75–1.5 s, and consistency across sampling rates (20–100 Hz) confirms that seconds-based patching with adaptive pooling decouples temporal resolution from patch semantics.

## 6.5 Scaling of Model Sizes

We train three model sizes—Tiny (�=192, 4 layers, 6M), Small (�=384, 8 layers, 35M), and Medium (�=512, 8 layers, 63M)— with identical training recipes, hyperparameters, and data to isolate the efect ofmodel capacity. Table 8 shows Small is the sweet spot: Medium regresses by 8.2 pp on ZS-O despite doubling parameters, achieving the highest validation accuracy but the worst held-out generalization—classic overfitting to dataset-specific patterns rather than transferable motion representations. Tiny underfits, falling 13.2 pp below Small on ZS-O, though it slightly outperforms Small at 10% supervision (86.4% vs. 85.3%), suggesting smaller models benefit more from supervised regularization. The ∼13K-session corpus supports ∼35M parameters before overfitting dominates; larger HAR corpora could shift this ceiling upward.

<table><tr><td>Model</td><td>Params</td><td>Avg (ms)</td><td>p95 (ms)</td><td>Peak Mem. (MB)</td><td>CPU Wt/ms</td></tr><tr><td>CrossHAR</td><td>110K</td><td>0.29</td><td>0.30</td><td>1.20</td><td>224.0K</td></tr><tr><td>LiMU-BERT</td><td>110K</td><td>0.31</td><td>0.36</td><td>0.27</td><td>188.0K</td></tr><tr><td>LLaSA</td><td>105K</td><td>0.33</td><td>0.46</td><td>0.02</td><td>187.2K</td></tr><tr><td>LanHAR</td><td>8.8M</td><td>0.77</td><td>0.89</td><td>0.05</td><td>71.8K</td></tr><tr><td>MOMENT</td><td>24.0M</td><td>37.69</td><td>37.98</td><td>25.05</td><td>1.8K</td></tr><tr><td>HALO (ours)</td><td>34.0M</td><td>7.16</td><td>7.37</td><td>32.81</td><td>2414.4K</td></tr></table>

Table 9: Inference overhead on an iPhone 16 Pro. All models are benchmarked with same setting: identical 6-channel IMU inputs using Core ML, with 10 warmup iterations followed by 100 timed iterations. Params denotes the profiled model size. CPU Wt/ms reports cycle-weighted CPU activity normalized by benchmark duration.

## 6.6 Inference Overhead on Smartphones

Table 9 shows that HALO achieves an average inference latency of 7.16 ms (p95: 7.37 ms) on an iPhone 16 Pro, well within real-time requirements for on-device activity recognition. Under the same Core ML deployment setting (identical 6-channel IMU input, 10 warm-up and 100 timed iterations), HALO is 5.26× faster than MOMENT (7.16 ms vs. 37.69 ms) while jointly modeling all six IMU channels in a single forward pass. Although HALO is slower than lightweight baselines such as CrossHAR (0.29 ms) and LiMU-BERT (0.31 ms) due to its dual-branch Transformer architecture, its latency remains practical for smartphone deployment. The speedup over MOMENT mainly stems from inference structure: MO MENT relies on a large T5-style encoder and requires six separate passes for a 6-channel input, whereas HALO processes all channels jointly. HALO reaches a peak memory usage of 32.81 MB, and label embeddings are precomputed and cached ofline, incurring negligible runtime overhead.

## 6.7 Real-World Case Study

We implement the full HALO pipeline as an iOS application and test under continuous placement and label shifts. As shown in Figure 8, the pipeline consists offour stages: (1) SFT data collection on the smartphone, where the user performs each target activity for 30 s per class; (2) SFT head training on a laptop (under 2 minutes for 6 classes); (3) weight transfer back to the device via a local network; and (4) real-time inference. The demo covers three body locations (left pocket, wrist, upper arm) with six seen classes (Sitting, Standing, Walking, Running, Jumping, Lying) and two unseen labels (Marching, Jogging) introduced at inference to test robustness beyond the SFT label space.

![](images/a88d80e2892cf97db6e58640b9b09e7a9e3c3ca888718559ed9d899027c023f8.jpg)  
Figure 8: Deployment setting of smartphone-based IMU HAR in the real-world case study.

![](images/af23c82e546899379c12e549df7b426802cea7885a45cdf7965b136356442b4f.jpg)  
Figure 9: Performance continuous IMU-based HAR on the smartphone.

Across 40 inference windows (Figure 9), HALO achieves 97.5% accuracy with only one transient error near the pocketto-wrist transition, recovering immediately. The placement transition from pocket to wrist represents a significant sensor distribution shift mid-session, yet the model adapts within a single window. The unseen-label phase remains fully correct, confirming that the personalized head preserves semantic generalization even when inference-time labels difer from the SFT classes—the model correctly distinguishes “Marching” from “Walking” and “Jogging” from “Running” despite never training on these labels during SFT. Mean end-to-end pipeline latency is 30.35 ms (P95: 33.66 ms), higher than the isolated model benchmark (7.16 ms, Table 9) due to app-level overhead; memory stays within 73–78 MB, and the deployed model is 21.8M parameters (∼34 MB on disk)—practical for continuous on-device HAR.

## 7 CONCLUSION AND DISCUSSIONS

We presented HALO, an IMU foundation model that supports diverse wearable configurations and enables openvocabulary activity recognition without per-dataset adaptation. With ∼35M parameters, HALO outperforms five baselines across all eight aggregate metrics. Our results also motivate several directions for future work. First, moving from dataset-level to per-sample channel–text conditioning could better capture heterogeneous body placements, as evidenced by a 9.9 pp zero-shot drop on RealWorld with richer global descriptions (Section 6.3.2). Second, robustness to severe distribution shift remains an open challenge, particularly for datasets such as HARTH and VTT-ConIoT under zeroshot transfer. Finally, expanding beyond the current 87-label vocabulary and 384-dimensional embedding space may further enhance semantic coverage and generalization ability of HALO.

## REFERENCES

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge Luis Reyes-Ortiz. 2013. A Public Domain Dataset for Human Activity Recognition Using Smartphones. In Proceedings ofthe European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN). Bruges, Belgium.

[2] Oresti Banos, Rafael Garcia, Juan A. Holgado-Terriza, Miguel Damas, Hector Pomares, Ignacio Rojas, Alejandro Saez, and Claudia Villalonga. 2014. mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications. In Proceedings of the 6th International Work-conference on Ambient Assisted Living and Daily Activities (IWAAL). Springer, Belfast, UK, 91–98.

[3] Billur Barshan and Murat C. Yüksek. 2014. Recognizing Daily and Sports Activities in Two Open-Source Machine Learning Environments Using Body-Worn Sensor Units. Comput. J. 57, 11 (2014), 1649–1667.

[4] Zhiqiang Chang, Patrizia Di Campli San Vito, Mirco Musolesi, and Cecilia Mascolo. 2024. CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 1 (2024), 1–27.

[5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. 2020. A Simple Framework for Contrastive Learning of Visual Representations. In Proceedings ofthe 37th International Conference on Machine Learning (ICML). PMLR, Virtual, 1597–1607.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings ofthe 2019 Conference ofthe North American Chapterofthe AssociationforComputational Linguistics (NAACL). Association for Computational Linguistics, Minneapolis, MN, 4171–4186.

[7] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. ImageBind: One Embedding Space to Bind Them All. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Vancouver, Canada, 15180–15190.

[8] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. 2024. MOMENT: A Family of Open Time-Series Foundation Models. In Proceedings ofthe 41st International Conference on Machine Learning (ICML). PMLR, Vienna, Austria, 16115–16152.

[9] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. Masked Autoencoders Are Scalable Vision Learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, New Orleans, LA, 16000–16009.

[10] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum Contrast for Unsupervised Visual Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Seattle, WA, 9729–9738.

[11] Sheikh Asif Imran, Mohammad Nur Hossain Khan, Subrata Biswas, and Bashima Islam. 2025. LLaSA: A Sensor-Aware LLM for Natural

Language Reasoning of Human Activity from IMU Data. In Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp). ACM.

[12] Jennifer R. Kwapisz, Gary M. Weiss, and Samuel A. Moore. 2011. Activity Recognition Using Cell Phone Accelerometers. ACM SIGKDD Explorations Newsletter 12, 2 (2011), 74–82.

[13] Quan Li, Kecheng Li, Yuqian Hu, Bin Guo, and Zhiwen Yu. 2024. Lan-HAR: Language-Based Zero-Shot Human Activity Recognition via Large Language Models. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 4 (2024), 1–25.

[14] Aleksej Logacjov, Kerstin Bach, Atle Kongsvold, Hilde Bremseth Bårdstu, and Paul Jarle Mork. 2021. HARTH: A Human Activity Recognition Dataset for Machine Learning. Sensors 21, 21 (2021), 7261.

[15] Satu-Marja Mäkelä, Ari Lämsä, Joonas S. Keränen, Jari Liikka, Jari Ronkainen, Jari Peltola, Jukka Häikiö, Sami Järvinen, and Miguel Bordallo López. 2021. Introducing VTT-ConIoT: A Realistic Dataset for Activity Recognition of Construction Workers Using IMU Devices. Sustainability 14, 1 (2021), 220. https://doi.org/10.3390/su14010220

[16] Mohammad Malekzadeh, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. 2019. Mobile Sensor Data Anonymization. In Proceedings ofthe International Conference on Internet ofThings Design and Implementation (IoTDI). ACM, Montreal, Canada, 49–58.

[17] Leland McInnes, John Healy, and James Melville. 2018. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426 (2018).

[18] Daniela Micucci, Marco Mobilio, and Paolo Napoletano. 2017. UniMiB SHAR: A Dataset for Human Activity Recognition Using Acceleration Data from Smartphones. Applied Sciences 7, 10 (2017), 1101.

[19] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In The Eleventh International Conference on Learning Representations (ICLR). OpenReview.net, Kigali, Rwanda, 1–22.

[20] Francisco Javier Ordóñez and Daniel Roggen. 2016. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. Sensors 16, 1 (2016), 115.

[21] Xiaomin Ouyang, Xian Shuai, Jiayu Zhou, Ivy Wang Shi, Zhiyuan Xie, Guoliang Xing, and Jianwei Huang. 2022. Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition. In Proceedings ofthe 28th Annual International Conference on Mobile Computing and Networking (MobiCom). ACM, 324–337.

[22] Xiaomin Ouyang, Zhiyuan Xie, Jiayu Zhou, Jianwei Huang, and Guoliang Xing. 2021. ClusterFL: A Similarity-Aware Federated Learning System for Human Activity Recognition. In Proceedings ofthe 19th Annual International Conference on Mobile Systems, Applications, and Services (ACM MobiSys). ACM, 54–67. https://doi.org/10.1145/3458864. 3467681

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML). PMLR, Virtual, 8748–8763.

[24] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Hong Kong, China, 3982–3992.

[25] Attila Reiss and Didier Stricker. 2012. Introducing a New Benchmarked Dataset for Activity Monitoring. In Proceedings ofthe 16th International Symposium on Wearable Computers (ISWC). IEEE, Newcastle, UK, 108– 109.

[26] Jorge-Luis Reyes-Ortiz, Luca Oneto, Albert Samà, Xavier Parra, and Davide Anguita. 2016. Transition-Aware Human Activity Recognition Using Smartphones. Neurocomputing 171 (2016), 754–767.

[27] Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Kilian Förster, Gerhard Tröster, Paul Lukowicz, David Bannach, Gerald Pirkl, Alois Ferscha, et al. 2010. Collecting Complex Activity Datasets in Highly Rich Networked Sensor Environments. In Proceedings of the 7th International Conference on Networked Sensing Systems (INSS). IEEE, 233–240.

[28] Muhammad Shoaib, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and Paul J. M. Havinga. 2014. Fusion of Smartphone Motion Sensors for Physical Activity Recognition. Sensors 14, 6 (2014), 10146–10176.

[29] Nilufar Sikder, Muhammad Sakib Ibne Chowdhury, and Abdullah-Al Nahid. 2021. KU-HAR: An Open Dataset for Heterogeneous Human Activity Recognition. Pattern Recognition Letters 146 (2021), 46–54.

[30] Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Bent Kjærgaard, Anind Dey, Tobias Sonne, and Mads Møller Jensen. 2015. Smart Devices Are Diferent: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition. In Proceedings of the 13th ACM Conference on Embedded Networked Sensor Systems (SenSys). ACM, Seoul, South Korea, 127–140.

[31] Timo Sztyler and Heiner Stuckenschmidt. 2016. On-Body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition. In Proceedings ofthe IEEE International Conference on Pervasive Computing and Communications (PerCom). IEEE, Sydney, Australia, 1–9.

[32] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748 (2018).

[33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems 30 (NeurIPS). Curran Associates, Long Beach, CA, 5998–6008.

[34] George Vavoulas, Charikleia Chatzaki, Thodoris Malliotakis, Matthew Pediaditis, and Manolis Tsiknakis. 2016. The MobiAct Dataset: Recognition of Activities of Daily Living Using Smartphones. In Proceedings ofthe International Conference on Information and Communication Technologies for Ageing Well and e-Health (ICT4AWE). SciTePress, Rome, Italy, 143–151.

[35] Huatao Xu, Pengfei Zhou, Rui Tan, and Mo Li. 2023. Practically Adopting Human Activity Recognition. In Proceedings of the 29th Annual International Conference on Mobile Computing and Networking (ACM MobiCom). ACM, 10–21. https://doi.org/10.1145/3570361.3613299

[36] Huatao Xu, Pengfei Zhou, Rui Tan, Mo Li, and Guobin Shen. 2021. LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications. In Proceedings ofthe 19th ACM Conference on Embedded Networked SensorSystems (SenSys). ACM, Coimbra, Portugal, 220–233.

[37] Xiyuan Zhang, Danielle Maddix, Hao Wang, Megha Gupta, Michael Mahoney, Priyank Goyal, Oleksandr Shchur, Andrew Merrill, Caner Turkmen, Qingsong Wen, Vasudev Rajan, Renqian Luo, Christos Faloutsos, and Yuyang Wang. 2025. Mitra: Mixed Synthetic Priors for Enhanc ing Tabular Foundation Models. In Advances in Neural Information Processing Systems (NeurIPS). arXiv:2510.21204.

[38] Xiyuan Zhang, Diyan Teng, Ranak Roy Chowdhury, Shuheng Li, Dezhi Hong, Rajesh K. Gupta, and Jingbo Shang. 2024. UniMTS: Unified Pretraining for Motion Time Series. In Advances in Neural Information Processing Systems 37 (NeurIPS).

[39] Zhaxidele, Jiawei Bian, et al. 2022. RecGym: Gym Workouts Recognition Dataset with IMU and Capacitive Sensor. UCI Machine Learning Repository. (2022). https://archive.ics.uci.edu/dataset/1128.