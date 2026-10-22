# NeuroCodec: EEG-Guided Latent-Domain Generative Model for High-Fidelity Target Speaker Extraction

Julian Silva , Jaliya Kodithuwakku , Navindu Sirimanna , Shaveen Herath , Wageesha N. Manamperi , Member, IEEE, and Joshua P. Kulasingham , Member, IEEE

Abstract—Driven by recent advancements in hearing assistive devices, target speaker extraction (TSE) in multi-talker environments has drawn significant attention. To address this, hearing technologies identify attention using neural signals recorded via electroencephalography (EEG) to guide TSE, enabling neurosteered auditory attention decoding. However, most existing techniques rely on time-frequency masking, which suffers from low fidelity in spectro-temporally overlapping regions and is often restricted to narrowband audio. In contrast, we introduce NeuroCodec, which, to the best of our knowledge, is the first generative EEG-guided TSE framework operating in the latent space of a neural audio codec. Specifically, NeuroCodec models spatial EEG dependencies via graph convolutions and fuses cross-modal EEG and speech representations through attention mechanisms and Mamba-based temporal modeling. We further introduce a six-phase training curriculum that progressively incorporates latent reconstruction, multi-scale spectral, and adversarial objectives to ensure stable, high-fidelity speech generation. Experiments on the public Cocktail Party and KUL datasets show NeuroCodec outperforms baselines, across several perceptual metrics. A listening experiment with 20 participants validated that NeuroCodec offers greater perceptual accuracy, achieving a mean score of 82.24 compared with 49.63 for M3ANet. Additionally, the proposed method is seen to enable 44.1 kHz reconstruction with spectral content extending up to 20 kHz, further supporting the perceptual results. Finally, we extend NeuroCodec with a causal streaming formulation suited for real-time processing, providing a basis for hearing assistive devices.

Index Terms—Target speaker extraction, audio codec, generative speech reconstruction, electroencephalography (EEG), hearing assistive devices

## I. INTRODUCTION

N complex, multi-talker acoustic environments, the human auditory system exhibits a remarkable capacity to isolate a target speech stream while suppressing interfering voices and background noise, an ability commonly referred to as selective auditory attention in a “cocktail party” scenario [1], [2]. However, this selective auditory attention is severely compromised in listeners with hearing impairments [3], leading to degraded speech intelligibility and elevated cognitive load [4], [5]. While advanced hearing aid technologies and digital acoustic prostheses have successfully restored broad auditory accessibility for millions of individuals, enabling these devices to operate effectively in dynamic, multi-speaker

scenes remains a critical hurdle. For modern speech enhancement systems, one fundamental challenge lies not merely in isolating overlapping speech audio, but in decoding the listener’s attention to determine which speaker to extract from the acoustic mixture.

Target Speaker Extraction (TSE) aims to address this problem by isolating a desired speaker from a speech mixture using auxiliary information about the target speaker [6]. Recent approaches exploit a variety of cues, including acoustic embeddings, directional or binaural spatial cues [7], [8], and visual signals [9], [10], to guide the extraction process and improve robustness in challenging acoustic environments. Although these approaches have achieved strong extraction performance, they rely on external reference information that may be unavailable or unreliable in dynamic real-world environments.

An alternative approach is to decode the attentional target from the listener’s brain activity, which has been shown to be strongly correlated with the attended speech signal. Several studies in the field of auditory neuroscience have demonstrated that the electroencephalography (EEG) signal tracks the attended speech, which is also called neural speech tracking [11], [12]. Consequently, EEG serves as a direct, non-invasive, and robust indicator of attention, enabling TSE without relying on restrictive external cues. While early computational approaches leveraged this neural-acoustic correlation solely to detect the attended speaker [13] without speech extraction, a task known as Auditory Attention Decoding (AAD), recent advances have established integrated neuro-steered TSE frameworks [14]–[17]. By utilizing the continuous EEG signal as the reference cue, these systems directly isolate and reconstruct the attended speech waveform from the multi-talker acoustic mixture [18].

Neuro-steered TSE can be categorized into two main processing paradigms: offline and online extraction. Offline neurosteered TSE processes complete audio-EEG sequence before producing the target speech, allowing the model to exploit full temporal context during extraction. Early work, such as brain-informed speech separation (BISS) [18], used an EEGdecoded speech envelope to guide TSE. The brain-assisted speech enhancement network (BASEN) [19] later introduced convolutional cross-attention for direct EEG-guided speech enhancement, while NeuroHeed [15] employed self-attentionbased EEG encoding and an end-to-end extraction framework. NeuroHeed+ [17] extended this approach through joint auditory attention detection and extraction. NeuroSpex+ [16] further improves extraction through dual-task training on speech-envelope and waveform objectives and is evaluated under a subject-independent (SI) setting, where test subjects are unseen during training. Although SI evaluation is more challenging than subject-dependent (SD) testing, it provides a rigorous measure of cross-subject generalization for practical deployment. More recently, M3ANet [14] incorporated graphbased EEG modeling and GroupMamba blocks.

Despite these architectural advances, the inherent reliance on offline processing renders these models unsuitable for realtime deployment in hearing assistive devices that require continuous, low-latency operation. To bridge this gap, online processing paradigms handle the input audio stream using short, overlapping causal windows. NeuroHeed [15] introduced an online formulation for causal neuro-steered TSE, while CA-NeuroSpex [20] more recently explored a related causal setting. However, causal neuro-steered extraction remains highly challenging, with existing online systems generally showing lower performance than their offline counterparts [15].

Taken together, prior work leaves two main gaps. First, existing EEG-guided TSE systems still rely on discriminative mixture-domain mask prediction [14], [15]. When target and interfering speech overlap, an inaccurate mask may retain components of the interfering speaker, resulting in speaker leakage. Second, because prior systems have been evaluated at limited audio sampling rates, at a maximum of 8 kHz or 14.7 kHz [14], [15], their ability to reconstruct wider-band attended speech has not been established. This narrowband constraint yields an output with a muffled, robotic acoustic quality that fails to meet the perceptual naturalness required for real-world applications.

Generative approaches have recently shown promising performance in speech enhancement and for TSE conditioned on acoustic or visual cues [21]–[23]. However, these models are computationally intensive, rely on slow iterative generation, and typically benefit from large-scale training data. These requirements are particularly challenging for EEG-guided TSE, where paired EEG-speech datasets are limited and low-latency processing is desirable. To address these constraints, neural audio codecs such as SoundStream [24], EnCodec [25], and the Descript Audio Codec (DAC) [26], offer an alternative by using Residual Vector Quantization (RVQ) to compress highrate audio into compact, low-frame-rate representations that can be reconstructed by a learned decoder. Predicting these target speech latents fundamentally transforms the extraction task from discriminative mixture-domain mask estimation to direct, generative latent synthesis.

In this paper, we propose NeuroCodec, a generative neurosteered TSE framework that addresses the limitations of timefrequency mask estimation using a generative neural audio codec. Specifically, the framework predicts target speech representations within the latent space of a frozen, pretrained DAC [26]. This generative formulation aims to reduce masking-related artifacts such as spectral holes and residual speaker leakage, while enabling substantially wider-band reconstruction. Unlike prior neuro-steered systems limited to an output bandwidth of approximately 4-7 kHz, NeuroCodec reconstructs spectral content up to 20 kHz. Cross-modal attention and Mamba-based temporal modeling fuse EEG and mixture audio representations at the compact codec frame rate. To the best of our knowledge, there has not been prior work on generative EEG-guided TSE frameworks. Moreover, we present OnlineNeuroCodec, a causal streaming extension tailored for real-time hearing assistive devices.

Our main contributions are as follows:

• A generative EEG-guided target speech extraction framework, NeuroCodec, that predicts target speech in the latent space of a neural audio codec, featuring a sixphase training curriculum to stabilize latent, spectral, and adversarial objectives while mitigating traditional timefrequency masking artifacts.

• A causal streaming extension, OnlineNeuroCodec, incorporating previous target-latent estimates through an auditory attractor to enable real-time, neuro-steered extraction.

• A comprehensive experimental evaluation under the SI setting on the Cocktail Party [27] and KUL [13] datasets, featuring objective evaluations, ablation studies, streaming benchmarks, spectrogram analysis, and a perceptual listening study demonstrating reconstruction of spectral content up to approximately 20 kHz.

The remainder of the paper is organized as follows. Section II introduces the proposed NeuroCodec framework, Section III presents OnlineNeuroCodec, the causal extension for online streaming. Section IV describes the implementation details and datasets, Section V presents the results and discussion, and Section VI is the conclusion.

## II. NEUROCODEC FRAMEWORK

## A. Model Overview

We formulate the TSE problem as a generative mapping from a multi-speaker mixture and multi-channel EEG signals to the target speaker’s latent representations, which are subsequently decoded into a high-fidelity speech waveform. By predicting these latents within the structured space of a frozen pretrained DAC [26], the framework shifts from mixture-domain interference suppression to cross-modal latent alignment. This enables the model to leverage pretrained perceptual priors for speech extraction while focusing its learning capacity on EEG-guided attention detection. The codec’s decoder is utilized for final waveform synthesis. As shown in Fig. 1 the architecture consists of four stages: (i) audio encoder, (ii) EEG encoder, (iii) cross-modal selective state space block (CM-S3), and (iv) codec latent generator and decoder. The complete offline training algorithm is provided in the supplementary materials.

## B. Audio Encoder

A pretrained and frozen DAC [26] encoder $\mathcal { E } _ { c }$ maps the mixture waveform $\mathbf { x } \ \in \ \mathbb { R } ^ { B \times 1 \times T }$ to a compact latent representation $\mathbf { Z } _ { \mathrm { m i x } } \in \mathbb { R } ^ { B \times D \times T _ { c } }$ , where B is the batch size, $T$ is the number of waveform samples, $T _ { c }$ is the number of latent samples, and $D = 1 0 2 4$ is the codec latent dimension size. The compact codec representation substantially reduces the temporal resolution relative to the raw audio, enabling efficient subsequent EEG-audio fusion. The DAC encoder remains frozen throughout training to preserve its pretrained acoustic representation.

![](images/5515e930cf817bcee0181d1a2018276dd41201241f01b5dbc98b99913f4f43c9.jpg)  
Fig. 1. Illustration of NeuroCodec architecture. A frozen DAC encoder extracts mixture latents $\mathbf { Z } _ { \mathrm { m i x } } ,$ which are fused with EEG embeddings through stacked CM-S3 blocks to predict target latents $\mathbf { Z } _ { \mathrm { t g t } }$ . The predicted latents are quantized via codebooks $\left( Q _ { 1 } { - } Q _ { 9 } \right)$ , and a frozen DAC decoder reconstructs the target speech waveform ˆs. The EEG encoder captures spatial-temporal neural features, and multi-period/multi-band discriminators provide adversarial supervision.

## C. EEG Encoder

Adapted from M3ANet [14], the EEG encoder applies third order Chebyshev graph convolutions [28] to model spatial dependencies across electrodes. Unlike prior approaches that upsample EEG toward the audio rate [14]–[16], NeuroCodec operates at the compact codec latent rate (50 or 86 Hz), close to the native 128 Hz EEG resolution, thereby avoiding the initial upsampling and substantially reducing interpolation and computational overhead. The graph features are then processed by parallel 1-D convolutions with kernel sizes 3, 7, and 11, followed by a temporal encoder stack of 1-D convolutions to capture deeper sequential dependencies, that yields a latent output $\bar { \textbf { E } } \doteq \mathbb { R } ^ { B \times } \mathbf { \dot { H } } \times T _ { e } ^ { \prime }$ , where H is the EEG embedding dimension and $T _ { e } ^ { \prime }$ is the number of latent samples.

## D. Cross Modal Selective State Space Block

Positional encodings are then injected into both modalities and the temporal sequences are aligned via multi-head crossattention [29], defined as

$$
{ \bf Z } _ { 1 } = { \bf Z } _ { \mathrm { m i x } } + \mathrm { A t t n } ( \mathrm { L N } ( { \bf Z } _ { \mathrm { m i x } } ) , \mathrm { L N } ( \bar { \bf E } ) , \mathrm { L N } ( \bar { \bf E } ) )\tag{1}
$$

where LN(·) denotes layer normalization and $\mathrm { A t t n } ( \cdot , \cdot , \cdot )$ denotes cross-modal attention with inputs of queries, keys and values. The acoustic tokens act as queries, while the normalized EEG features serve as keys and values to steer the extraction. Long-range temporal dependencies are subsequently modeled using a Mamba block [30], defined as

$$
\mathbf { Z } _ { \mathrm { t g t } } = \mathbf { Z } _ { 1 } + \mathrm { M a m b a } ( \mathrm { L N } ( \mathbf { Z } _ { 1 } ) )\tag{2}
$$

By replacing self-attention with selective state-space modeling, Mamba provides efficient linear-time temporal modeling of codec-latent sequences.

## E. Codec Latent Generator and Decoder

The output of the four stacked CM-S3 blocks is projected back to the codec latent dimension to yield the continuous target latents $\tilde { \mathbf { Z } } _ { \mathrm { t g t } } \in \mathbb { R } ^ { B \times D \times T _ { c } }$ . These predicted latents are then quantized using the residual vector quantization (RVQ) [26] codebooks $( Q _ { 1 } { - } Q _ { 9 } )$ of the frozen DAC, yielding the discrete latent representation defined as $\hat { \mathbf { Z } } _ { \mathrm { t g t } } \ = \ Q ( \tilde { \mathbf { Z } } _ { \mathrm { t g t } } )$ , where $\mathcal { Q }$ denotes the quantization operation. Finally, a pretrained and frozen DAC decoder D generates the target speech waveform from these quantized latents, providing the target reconstruction defined as $\hat { \mathbf { s } } = \mathcal { D } _ { c } ( \hat { \mathbf { Z } } _ { \mathrm { t g t } } )$

## F. Loss Functions

Unlike conventional models that optimize scale-invariant signal-to-distortion ratio (SI-SDR) loss directly on the decoded audio [14]–[16], [18], Neurocodec is trained using a composite objective function for accurate latent alignment and highfidelity acoustic reconstruction.

Reconstruction loss, ${ \mathcal { L } } _ { \mathrm { r e c o n } } ,$ is computed as the mean squared error between the predicted and ground-truth target continuous latents and is used to guide the model to reconstruct the target speaker representation in the codec latent space.

However, reconstruction loss alone can yield perceptual artifacts. Spectral loss, $\mathcal { L } _ { \mathrm { m e l } }$ [26], [31], improves spectral fidelity by minimizing the $L _ { 1 }$ distance between log-Mel spectrograms at multiple resolutions. We use seven scales up to 20 kHz for the Cocktail Party [27] dataset and four scales up to 4 kHz for KUL dataset [13] to capture both transient and steady-state spectral characteristics across their respective bandwidths.

To further improve perceptual realism, we use an adversarial ensemble of Multi-Period Discriminators (MPD) [32] to model waveform periodicity and a Multi-band Multi-scale Short-Time Fourier Transform (STFT) Discriminator (MBMSD) [26] to enforce multi-band spectral consistency. The generator is optimized using hinge adversarial loss $\mathcal { L } _ { \mathrm { a d v } }$ and $L _ { 1 }$ featurematching loss ${ \mathcal L } _ { \mathrm { f m } }$ on intermediate discriminator features, while the discriminators are trained with the standard hinge objective.

## G. Multi-phase Training Curriculum

The overall generator objective is given by,

$$
\mathcal { L } _ { G } = \lambda _ { \mathrm { { r e c o n } } } \mathcal { L } _ { \mathrm { { r e c o n } } } + \lambda _ { \mathrm { { m e l } } } \mathcal { L } _ { \mathrm { { m e l } } } + \lambda _ { \mathrm { { a d v } } } \mathcal { L } _ { \mathrm { { a d v } } } + \lambda _ { \mathrm { { f m } } } \mathcal { L } _ { \mathrm { { f m } } } ,\tag{3}
$$

where $\lambda _ { \mathrm { { r e c o n } } } , \lambda _ { \mathrm { { m e l } } } , \lambda _ { \mathrm { { a d v } } }$ , and $\lambda _ { \mathrm { f m } }$ are weighting coefficients balancing their respective loss terms throughout training. We set $\lambda _ { \mathrm { r e c o n } } = 1$ to be fixed throughout training.

To preserve GAN stability and mitigate mode collapse during training, we employ a six-phase curriculum that progressively introduces spectral and adversarial objectives. In Phase 1, the model optimizes exclusively $\mathcal { L } _ { \mathrm { r e c o n } }$ until the validation latent mean squared error (MSE) falls below 0.87, establishing a stable latent mapping prior to multi-modal supervision. Phase 2 gradually introduces the Mel loss $\left( \mathcal { L } _ { \mathrm { m e l } } \right)$ ramping $\lambda _ { \mathrm { m e l } }$ from 2 to 15 to progressively incorporate spectral supervision without destabilizing the learned latent representation. Upon reaching $\lambda _ { \mathrm { m e l } } = 1 5 .$ , Phase 3 maintains joint latent and spectral optimization until the validation loss decreases for at least one epoch. In Phase 4, the generator parameters are frozen for 2 epochs to allow warm-up pretraining of the discriminators, providing stable adversarial supervision. Phase 5 then jointly optimizes the generator and discriminators, progressively increasing $\left( \lambda _ { \mathrm { a d v } } , \lambda _ { \mathrm { f m } } \right)$ in steps of (0.25, 0.5) until the target values of (1.0, 2.0) are reached. Finally, Phase 6 conducts full adversarial training until validation performance plateaus, employing a ReduceLROnPlateau scheduler with patience of 3 epochs and a reduction factor of 0.5.

In the following section, we extend this framework into a causal streaming formulation, OnlineNeuroCodec, which integrates an auditory attractor mechanism to enable real-time target speech extraction.

## III. CAUSAL STREAMING EXTENSION OF NEUROCODEC

We extend NeuroCodec to a causal formulation, termed OnlineNeuroCodec. The causal model preserves the latentdomain processing of the offline formulation while restricting each prediction to the current input window and the information accumulated from preceding windows.

Inference uses a sliding window comprising buffered context $w _ { b }$ and a new input hop $w _ { c } ,$ following NeuroHeed [15]. At each hop, only the latent frames corresponding to $w _ { c }$ are generated, while the preceding $w _ { b }$ frames provide causal context. The complete streaming procedure, including the cold start and the running speaker state introduced below, is given in Algorithm 1.

## A. Auditory Attractor via Running Speaker State

OnlineNeuroCodec preserves attended-speaker identity across hops by maintaining a running latent representation of previous target estimates, $\mathbf { Z } _ { \mathrm { p a s t } }$ . From the second hop onward, it is updated using an Exponential Moving Average (EMA) [33]:

$$
\mathbf { Z } _ { \mathrm { p a s t } }  \alpha \mathbf { Z } _ { \mathrm { p a s t } } + ( 1 - \alpha ) \hat { \mathbf { Z } } _ { \mathrm { t g t } } ^ { h } ,\tag{4}
$$

where $\hat { \mathbf { Z } } _ { \mathrm { t g t } } ^ { h }$ denotes the target-latent estimate of the current hop and α is the EMA smoothing coefficient, set to 0.7 based on validation performance. The EMA combines accumulated speaker information with the latest estimate, providing a stable speaker memory for subsequent extraction.

OnlineNeuroCodec therefore uses two complementary cues: a neuronal attractor, in which EEG features provide the keys and values for cross-attention, and an auditory attractor derived from $\mathbf { Z } _ { \mathrm { p a s t } }$ . The latter is mean-pooled and linearly projected to FiLM [34] scale and shift parameters $( \gamma , \beta )$

Algorithm 1 OnlineNeuroCodec: causal streaming inference.   
Require: Streaming mixture x, EEG E; window $w _ { b } ,$ hop $w _ { c }$   
Require: Frozen $\mathcal { E } _ { c } , \ \mathcal { Q } ,$ and $\mathcal { D } _ { c } \mathbf { ; }$ EMA rate α   
Ensure: Extracted waveform ˆs   
1: $\mathbf { Z } _ { \mathrm { p a s t } }  \varnothing ; \hat { \mathbf { s } }  \varnothing$   
2: for each hop $h = 1 , \ldots , N _ { \mathrm { h o p s } }$ do   
3: $\mathbf { x } ^ { ( h ) } , \mathbf { R } ^ { ( h ) } \gets$ window of length w ending at hop h   
4: $\mathbf { Z } _ { \mathrm { m i x } } ^ { ( h ) }  \mathcal { E } _ { c } ( \mathbf { x } ^ { ( h ) } ) ; \bar { \mathbf { E } } ^ { ( h ) }  \mathrm { E } \mathrm { E } \mathrm { G } \mathrm { E n c } ( \mathbf { R } ^ { ( \bar { h } ) } )$   
5: if $h = 1$ then   
6: $\mathbf { Z } _ { \mathrm { p s e u d o } } ^ { ( h ) }  \mathrm { C M - S 3 } ( \mathbf { Z } _ { \mathrm { m i x } } ^ { ( h ) } , \bar { \mathbf { E } } ^ { ( h ) } )$   
7: $\mathbf { Z } _ { \mathrm { p a s t } }  \mathbf { Z } _ { \mathrm { p s e u d c } } ^ { ( h ) }$   
8: end if   
9: $[ \gamma , \beta ] \gets \mathrm { L i n e a r } ( \mathrm { M e a n } ( \mathbf { Z } _ { \mathrm { p a s t } } ) )$   
10: $\widetilde { \mathbf Z } _ { \mathrm { m i x } } ^ { ( h ) } \gets ( 1 + \gamma ) \odot \mathrm { L N } ( \mathbf Z _ { \mathrm { m i x } } ^ { ( h ) } ) + \beta$   
11: $\widetilde { \mathbf { Z } } _ { \mathrm { t g t } } ^ { ( h ) }  \mathrm { C M - S 3 } ( \widetilde { \mathbf { Z } } _ { \mathrm { m i x } } ^ { ( h ) } , \bar { \mathbf { E } } ^ { ( h ) } )$   
12: $\hat { \mathbf { Z } } _ { \mathrm { t g t } } ^ { ( \breve { h } ) }  \mathcal { Q } ( \widetilde { \mathbf { Z } } _ { \mathrm { t g t } } ^ { ( h ) } )$   
13: $\mathbf { Z } _ { \mathrm { p a s t } }  \alpha \mathbf { Z } _ { \mathrm { p a s t } } + ( 1 - \alpha ) \hat { \mathbf { Z } } _ { \mathrm { t g t } } ^ { ( h ) }$   
14: $\mathbf { Z } _ { \mathrm { p a s t } }  \mathrm { d e t a c h } ( \mathbf { Z } _ { \mathrm { p a s t } } )$   
15: $\hat { \mathbf { s } }  \hat { \mathbf { s } } \Vert \mathcal { D } _ { c } ( \hat { \mathbf { Z } } _ { \mathrm { t g t } } ^ { ( h ) } )$   
16: end for   
17: return ˆs

which modulate the current mixture-latent query $\mathbf { Z } _ { \mathrm { m i x } }$ before EEG cross-attention:

$$
[ \gamma , \beta ] = \mathrm { L i n e a r } ( \mathrm { M e a n } ( \mathbf { Z } _ { \mathrm { p a s t } } ) ) ,
$$

$$
\widetilde { \mathbf Z } _ { \mathrm { m i x } } = ( 1 + \gamma ) \odot \mathrm { L N } ( \mathbf Z _ { \mathrm { m i x } } ) + \boldsymbol { \beta } .\tag{5}
$$

(6)

Thus, previous target estimates provide speaker-dependent conditioning for the current hop. Similar to the autoregressive attractor in NeuroHeed [15], this mechanism self-enrolls the attended speaker over time. However, NeuroHeed re-encodes previous waveform outputs using a dedicated LSTM speaker encoder, whereas OnlineNeuroCodec updates the speaker state directly in codec latent space using a parameter-free EMA, avoiding an additional recurrent speaker module.

## B. Hopwise Training Strategy

At the first hop, no preceding prediction is available. Therefore, $\mathbf { Z } _ { \mathrm { p a s t } }$ is initialized from an EEG-conditioned forward pass without auditory-history conditioning. During training, speaker dropout with probability $p = 0 . 2$ is applied so that the model remains functional when the historical context is unavailable [35].

Training follows the streaming setting by applying the reconstruction and spectral objectives of Section II-F to each output hop. For hop $h , \mathcal { L } _ { \mathrm { r e c o n } } ^ { ( h ) }$ operates in the latent domain, whereas $\dot { \mathcal { L } } _ { \mathrm { m e l } } ^ { ( h ) }$ is evaluated on the corresponding decoded speech segment. The overall online objective is

$$
\mathcal { L } _ { \mathrm { o n l i n e } } = \frac { 1 } { N _ { \mathrm { h o p s } } } \sum _ { h = 1 } ^ { N _ { \mathrm { h o p s } } } \left( \lambda _ { \mathrm { r e c o n } } \mathcal { L } _ { \mathrm { r e c o n } } ^ { ( h ) } + \lambda _ { \mathrm { m e l } } \mathcal { L } _ { \mathrm { m e l } } ^ { ( h ) } \right) ,\tag{7}
$$

where $N _ { \mathrm { h o p s } }$ is the number of output hops, and $\lambda _ { \mathrm { { r e c o n } } }$ and $\lambda _ { \mathrm { m e l } }$ weight the reconstruction and spectral terms, respectively. Averaging across hops keeps the objective scale independent of sequence length.

During training, gradients are stopped at $\mathbf { Z } _ { \mathrm { p a s t } }$ between consecutive hops, truncating backpropagation across hop boundaries while maintaining the state for causal conditioning.

## IV. DATASETS AND EXPERIMENTS

In this section, we detail the experimental setup and the datasets used to evaluate the proposed method.

## A. Datasets

To evaluate our proposed method, we used two publicly available multi-speaker EEG-audio datasets: Cocktail Party and KUL. The Cocktail Party dataset [27] includes EEG recordings from 33 subjects (128 channels, 512 Hz) who completed 30 dichotic listening trials (60 s each), attending to one of two simultaneous narrative speech streams presented to opposite ears. The KUL data set [13] consists of 16 subjects with normal-hearing with 20 trials each; in our experiments, we used the first eight trials (similar to prior work for fair comparison [15], [16]), such that subjects do not attend to the same speech stimulus repeatedly, where participants selectively attended one of two concurrent Dutch narrative stories presented dichotically.

## B. Implementation Details

EEG signals are sampled at 128 Hz and were band-pass filtered between 0.1-45 Hz, followed by ICA-based artifact removal [36]. All experiments use subject independent (SI) evaluation, where test subjects are unseen during training, providing a more stringent assessment of cross-subject generalization.

For the Cocktail Party dataset, EEG is additionally rereferenced to the average mastoids. Subjects are randomly split with a fixed seed into 23/5/5, training/validation/test subjects, and the same SI split is used for all compared models. For fair comparison with M3ANet [14], BASEN [19], and MSFNet [37], trials are segmented into 2s non-overlapping windows. The native 44.1 kHz audio is processed using the pretrained 44.1 kHz DAC encoder [26], which has a downsampling factor of 512 and yields an approximately 86 Hz latent frame rate.

For KUL, we use the preprocessed data released with NeuroHeed [15] and perform 16-fold SI cross-validation. For fair comparison with the SI NeuroSpex+ setting [16], recordings are segmented into 4s windows with a 1-s hop. The original 8 kHz audio is upsampled to 16 kHz and processed using the pretrained 16 kHz DAC encoder [26], which has a downsampling factor of 320 and yields a 50 Hz latent frame rate. Note that the difference in window sizes between the two datasets is to ensure fair comparison with the prior work on each dataset. Additional implementation details and hyperparameters are provided in the supplementary material. The source code will be released upon acceptance.

## C. Evaluation Metrics

Performance is evaluated using standard objective speech enhancement metrics: Short-Time Objective Intelligibility (STOI) [38], Extended Short-Time Objective Intelligibility (ESTOI) [39], and Perceptual Evaluation of Speech Quality (PESQ) [40]. To ensure fair comparison with existing baselines that only operate at 8 kHz sample rate, all outputs are resampled to 8 kHz before objective evaluation. This prevents the wider output bandwidth of NeuroCodec from providing an inherent advantage in the reported objective metrics.

## V. RESULTS AND DISCUSSION

This section details the experimental evaluation of the proposed framework across two public datasets. We present (i) objective speech quality metrics, (ii) an ablation study validating our core architectural choices, (iii) a subjective listening test, and (iv) a performance evaluation of the causal extension.

## A. Experimental Results

Table I compares NeuroCodec with previous EEG-guided target speaker extraction systems on the Cocktail Party and KUL datasets.

On the Cocktail Party dataset, BASEN [19] and M3ANet [14] are reproduced using the authors’ publicly available implementations to ensure a fair comparison by evaluating under the same SI split used for NeuroCodec. NeuroCodec achieves the highest STOI and PESQ values, improving STOI from 0.812 (M3ANet) to 0.827, and PESQ from 2.23 to 2.31. While M3ANet retains a marginal advantage in ESTOI, these results demonstrate that NeuroCodec improves overall speech intelligibility and perceptual quality, while remaining highly competitive in ESTOI.

On the KUL dataset, the performance metrics for prior baselines BISS [18], NeuroHeed [15], and Neuro-Spex [41] are sourced directly from the Neuro-Spex+ paper [16], which reports the identical SI configuration. The BISS baseline shows substantially lower performance compared to all other models, as it employs a subject-dependent linear decoder for envelope reconstruction that is trained exclusively on singletalker data. In contrast, when compared with the state-of-theart NeuroSpex+, our NeuroCodec improves STOI from 0.859 to 0.868 and PESQ from 2.514 to 2.562.

Note that other relevant models such as NeuroHeed+ [17] and MSFNet [37] are omitted from Table I because the lack of publicly available code prevents re-evaluation under the same rigorous SI setting.

TABLE I  
STOI, ESTOI, AND PESQ RESULTS ON COCKTAIL PARTY AND KUL
<table><tr><td>Model</td><td>STOI</td><td>ESTOI</td><td>PESQ</td></tr><tr><td>(a) Cocktail Party Dataset</td><td></td><td></td><td></td></tr><tr><td>BASEN [19]</td><td>0.790</td><td>0.715</td><td>1.91</td></tr><tr><td>M3ANet [14]</td><td>0.812</td><td>0.725</td><td>2.23</td></tr><tr><td>NeuroCodec</td><td>0.8273</td><td>0.721</td><td>2.31</td></tr><tr><td>(b) KUL Dataset</td><td></td><td></td><td></td></tr><tr><td>BISS [18]</td><td>0.106</td><td></td><td>-0.045</td></tr><tr><td>NeuroHeed [15]</td><td>0.637</td><td></td><td>2.282</td></tr><tr><td>NeuroSpex [41]</td><td>0.795</td><td></td><td>2.319</td></tr><tr><td>NeuroSpex+ [16]</td><td>0.859</td><td></td><td>2.514</td></tr><tr><td>NeuroCodec</td><td>0.868</td><td>0.754</td><td>2.562</td></tr></table>

Scores for the Cocktail Party dataset were produced using publicly available code from the relevant authors. Scores for prior models on the KUL dataset are sourced from NeuroSpex+ [16]. Baselines differ across datasets according to their reported evaluation settings.

Qualitative evaluation of the representative spectrograms shown in Fig. 2 further illustrates these differences. M3ANet [14] exhibits high-frequency truncation around 7 kHz, whereas NeuroCodec reconstructs spectral content up to 20 kHz with clearer harmonic structure. In overlappingspeech regions, NeuroCodec also shows cleaner target-speaker separation, consistent with its direct generation of target speech latents rather than masking the mixture representation.

![](images/6067b3f3ca9dfabac8819d6e95d03b3b5112eb7a41b9015c0451a0862c6af640.jpg)  
Fig. 2. Qualitative spectrogram comparison on the Cocktail Party dataset: (i) mixture, (ii) ground-truth target, (iii) M3ANet output, and (iv) NeuroCodec output. Solid boxes highlight NeuroCodec’s preserved content above 7 kHz, while dashed region mark overlapping-speech regions with reduced interference. Note that M3ANet does not preserve content above 7 kHz, nor does it avoid speaker leakage within the dashed region.

## B. Ablation Study

To understand the contribution of each architectural component, we conduct controlled ablation experiments on the KUL dataset. As summarized in Table II, we evaluate the proposed architecture across three core dimensions: the fusion block backbone, the internal EEG encoder architecture, and the training objectives.

1) Temporal modeling architecture: We first investigate the temporal modeling architecture by comparing three sequence modeling strategies: (i) a standard Transformer [29], (ii) a Transformer with Snake activations [42], and (iii) the proposed Causal Mamba. The baseline Transformer achieves lower scores, while the Snake-enhanced Transformer improves all three metrics over the standard Transformer. However, the Causal Mamba achieves the highest scores across all three metrics.

2) EEG encoder guidance and components: Next, we validate the necessity and design of the EEG encoder. To assess the model’s dependence on EEG guidance, we replace the EEG input with Gaussian noise. The substantial performance drop supports the conclusion that the model relies on neural guidance for target extraction. Furthermore, ablating specific internal components of the EEG encoder, namely the GCN spatial mapping and the Multi-Scale 1D Convolutions (MS-Conv1D), confirmed that capturing both cross-channel spatial topologies and multi-scale temporal receptive fields is crucial to extract better EEG features.

3) Progressive training objectives: Finally, we compare the progressive addition of training objectives. Using solely the latent-domain reconstruction loss $( { \mathcal { L } } _ { \mathrm { r e c o n } } )$ serves as our baseline. Integrating the multi-scale log-Mel spectrogram loss $\left( \mathcal { L } _ { \mathrm { m e l } } \right)$ drastically improves perceptual quality, while the further addition of the adversarial feature-matching loss $( \mathcal { L } _ { \mathrm { a d v / f m } } )$ leads to even more improvement. The combined objective $( \mathcal { L } _ { \mathrm { r e c o n } } { + } \mathcal { L } _ { \mathrm { m e l } } { + } \mathcal { L } _ { \mathrm { a d v / f m } } )$ achieves the highest overall performance, reaching a PESQ of 2.5623 and an ESTOI of 0.7544.

TABLE II  
ABLATION STUDIES OF NEUROCODEC ON THE KUL DATASET
<table><tr><td>Configuration</td><td>STOI</td><td>ESTOI</td><td>PESQ</td></tr><tr><td colspan="4">(a) Temporal Modeling Architecture (with full EEG Encoder &amp;  $\mathcal { L } _ { r e c o n } )$ </td></tr><tr><td>Transformer</td><td>0.6120</td><td>0.4703</td><td>0.8702</td></tr><tr><td>Transformer + Snake</td><td>0.6480</td><td>0.5162</td><td>1.0340</td></tr><tr><td>Mamba</td><td>0.7112</td><td>0.5734</td><td>1.5030</td></tr><tr><td colspan="4">(b) EEG Encoder Architecture (with Mamba &amp;  $\mathcal { L } _ { r e c o n } )$ </td></tr><tr><td>w/o EEG guidance (noise instead of EEG)</td><td>0.5412</td><td>0.3921</td><td>0.7845</td></tr><tr><td>w/o spatial mapping (w/o GCN)</td><td>0.6725</td><td>0.5312</td><td>1.2840</td></tr><tr><td>w/o Multi-scale conv1D</td><td>0.6854</td><td>0.5408</td><td>1.3912</td></tr><tr><td>Full Proposed EEG Encoder</td><td>0.7112</td><td>0.5734</td><td>1.5030</td></tr><tr><td colspan="4">(c) Training Objectives (with Mamba &amp; full EEG Encoder)</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c o n } } ~ ( \mathrm { B a s e } )$ </td><td>0.7112</td><td>0.5734</td><td>1.5030</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { m e l } }$ </td><td>0.8015</td><td>0.6861</td><td>1.9204</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { a d v / f m } }$ </td><td>0.8242</td><td>0.7122</td><td>2.1032</td></tr><tr><td> $\mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { m e l } } + \mathcal { L } _ { \mathrm { a d v / f m } }$  (Final)</td><td>0.8681</td><td>0.7544</td><td>2.5623</td></tr></table>

## C. Perceptual Evaluation

To verify the perceptual quality of the generated output, we conducted a controlled multi-stimulus listening test with 20 participants. The study was approved by the Ethics Review Committee of the University of Moratuwa (No EDN/2025/012) and all participants provided written informed consent. We used web Multiple Stimulus with Hidden Reference and Anchor (MUSHRA) [43] to compare NeuroCodec’s native 44.1kHz output with the native output of the strongest reproducible baseline, M3ANet [14]. The Cocktail Party dataset was used; KUL was excluded because NeuroSpex+ [16] has no public implementation. Eight 2s samples were randomly selected from the held-out test set.

![](images/b40bcda9e74c122b4b1522afbea3247f45affc8129764ef6176b2e01780e5fac.jpg)  
Fig. 3. MUSHRA listening-test results on the Cocktail Party dataset. Bandwidth-matched NeuroCodec outperforms M3ANet, while native NeuroCodec achieves a substantially larger perceptual gain.

Each trial presented five signals under blinded and randomized conditions: the hidden clean reference, mixture, M3ANet, bandwidth-matched NeuroCodec, and NeuroCodec under blinded and randomized presentation. As shown in Fig. 3, the hidden clean reference achieved a mean score of 94.76 (95% CI: 94.41-95.14) while NeuroCodec achieved 82.24 (CI: 78.86-85.39).Bandwidth-matched NeuroCodec, obtained by low-pass filtering the NeuroCodec output at 7.296 kHz to match the effective bandwidth of M3ANet, achieved 58.49 (CI: 55.06-62.31), compared with 49.63 (CI: 45.42-53.71) for M3ANet. This improvement under matched bandwidth indicates that NeuroCodec’s perceptual advantage is not solely attributable to its wider output bandwidth and is consistent with benefits from the proposed extraction architecture. The substantially higher score of the original NeuroCodec output further demonstrates the perceptual benefit of widerband speech reconstruction. The mixture obtained a mean score of 22.41 (CI: 20.24-24.80). The NeuroCodec-M3ANet difference was significant under a paired permutation test $( p = 3 . 0 \times 1 0 ^ { - 5 } )$ .

## D. OnlineNeuroCodec Evaluation

In addition to the offline evaluation, we assess OnlineNeuroCodec against our reimplementation of online Neuro-Heed [15].<sup>1</sup> OnlineNeuroCodec and the reimplemented NeuroHeed are both evaluated under an identical streaming configuration with a window length of $w = 2 . 0 \ s$ and a hop size of $w _ { c } = 0 . 5 \mathrm { ~ s ~ } ,$ with all benchmarks measured on a single GTX 1080 Ti GPU. Computational efficiency is evaluated using the Real-Time Factor $( \mathrm { R T F } = T _ { \mathrm { p r o c } } / w _ { c } )$ , where $T _ { \mathrm { p r o c } }$ is the processing time. We note that NeuroHeed [15] defines RTF using the reciprocal convention, $w _ { c } / T _ { \mathrm { p r o c } }$

As shown in Table III, OnlineNeuroCodec attains substantially higher extraction accuracy. Relative to NeuroHeed [15], STOI improves from 0.612 to 0.803, ESTOI from 0.5716 to 0.7105, and PESQ from 1.290 to 1.662.

TABLE III  
ONLINE EVALUATION OF ONLINENEUROCODEC AGAINST NEUROHEED [15] ON THE KUL DATASET.
<table><tr><td>Model</td><td>STOI↑</td><td>ESTOI↑</td><td>PESQ↑</td><td>RTF↓</td></tr><tr><td>NeuroHeed [15]</td><td>0.588</td><td>0.497</td><td>1.037</td><td>0.27</td></tr><tr><td>NeuroHeed [15] (with autoregressive Speaker Encoder)</td><td>0.612</td><td>0.572</td><td>1.290</td><td>0.41</td></tr><tr><td>OnlineNeuroCodec</td><td>0.742</td><td>0.686</td><td>1.501</td><td>0.13</td></tr><tr><td>OnlineNeuroCodec (with Auditory Attractor)</td><td>0.803</td><td>0.711</td><td>1.662</td><td>0.18</td></tr></table>

NeuroHeed results are from our reimplementation, as the authors release code for the offline model only.

Enabling the auditory attractor improves OnlineNeuroCodec from 0.742 to 0.803 STOI and from 1.501 to 1.662 PESQ. Since $\mathbf { Z } _ { \mathrm { p a s t } }$ affects the model only in mean-pooled form (Eq. 6), this improvement is consistent with more stable tracking of the attended speaker across hops. A comparable trend is observed for NeuroHeed, whose autoregressive speaker encoder raises STOI from 0.588 to 0.612, consistent with the self-enrollment behaviour reported in their work [15].

OnlineNeuroCodec without the attractor achieves the lowest RTF (0.13). Enabling the auditory attractor increases RTF modestly from 0.13 to 0.18. NeuroHeed is the most expensive of the four configurations (0.27-0.41). OnlineNeuroCodec is therefore approximately 2.1× faster than NeuroHeed without the attractor mechanism (0.13 against 0.27) and 2.3× faster with it (0.18 against 0.41), so its accuracy advantage is obtained at approximately half the measured RTF, despite operating on a generative rather than a masking objective.

Table IV reports the effect of the two streaming parameters. Lengthening the buffered context $w _ { b }$ improves PESQ, since a longer causal history is available for cross-modal alignment, but raises the RTF because each hop must reprocess a longer window. The gain in PESQ saturates beyond $w _ { b } ~ = ~ 1 . 5$ s while the RTF continues to grow. Enlarging the hop $w _ { c }$ lowers the $\mathrm { R T F , }$ as fewer forward passes are required per second of emitted audio, but increases algorithmic latency because the system must wait for a full hop of new audio before it can respond. Hence, the configuration $w _ { b } = 1 . 5$ s and $w _ { c } = 0 . 5 \mathrm { ~ s ~ }$ is adopted throughout, as it attains near-maximal PESQ while maintaining a low RTF.

TABLE IV  
EFFECT OF THE BUFFERED CONTEXT SIZE w<sub>b</sub> AND THE HOP SIZE $w _ { c }$ ON ONLINENEUROCODEC.
<table><tr><td></td><td>w (s)</td><td>PESQ ↑</td><td>RTF↓</td></tr><tr><td>(a) Buffered context wb  $( w _ { c } = 0 . 5 ~ s )$ </td><td></td><td></td><td></td></tr><tr><td> $w _ { b } = 0 . 5 \mathrm { ~ s ~ }$ </td><td>1.0</td><td>1.412</td><td>0.09</td></tr><tr><td> $w _ { b } = 1 . 0 \mathrm { ~ s ~ }$ </td><td>1.5</td><td>1.558</td><td>0.13</td></tr><tr><td> ${ \bf w _ { b } } = { \bf 1 . 5 ~ s }$ </td><td>2.0</td><td>1.662</td><td>0.18</td></tr><tr><td> $w _ { b } = 2 . 5 \mathrm { ~ s ~ }$ </td><td>3.0</td><td>1.681</td><td>0.27</td></tr><tr><td> $w _ { b } = 3 . 5 \mathrm { ~ s ~ }$ </td><td>4.0</td><td>1.689</td><td>0.36</td></tr><tr><td>(b)  $H o p \ w _ { c } \ ( w _ { b } = 1 . 5 \ s )$ </td><td></td><td></td><td></td></tr><tr><td> $w _ { c } = 0 . 1 0 ~ \mathrm { s }$ </td><td>1.6</td><td>1.674</td><td>0.71</td></tr><tr><td> $w _ { c } = 0 . 2 5 \mathrm { ~ s ~ }$ </td><td>1.75</td><td>1.669</td><td>0.31</td></tr><tr><td> $\mathbf { w _ { c } } = \mathbf { 0 . 5 0 s }$ </td><td>2.0</td><td>1.662</td><td>0.18</td></tr><tr><td> $w _ { c } = 1 . 0 0 ~ \mathrm { s }$ </td><td>2.5</td><td>1.594</td><td>0.11</td></tr></table>

The model was evaluated on the KUL dataset with the auditory attractor enabled. The total window is $\begin{array} { r } { w = w _ { b } + w _ { c } . } \end{array}$ . Bold text denotes the operating point used in Table III.

## VI. CONCLUSION

In this paper we introduce NeuroCodec, a generative EEGguided target speaker extraction framework that predicts latent representations of a frozen high-fidelity audio codec instead of estimating time-frequency masks. By combining graphbased EEG modeling, cross-modal attention, and Mambabased temporal modeling, NeuroCodec achieves effective EEG-audio fusion in a compact latent space. Experiments on two datasets demonstrate improved speech intelligibility and perceptual quality over existing approaches, as measured by both objective metrics and a subjective listening study. NeuroCodec also enables 44.1kHz reconstruction with spectral content extending to approximately 20 kHz with minimal speaker interference, unlike prior work with limited output bandwidth and speaker leakage. We extend the model to a causal streaming version, OnlineNeuroCodec, which enables streaming TSE using a causal auditory attractor derived from previous target estimates. This model achieves vastly superior computational speed while retaining competitive extraction quality. These results demonstrate the potential of generative latent modeling for neuro-steered speech extraction and future assistive hearing systems. However, OnlineNeuroCodec, like most other contemporary models, remains far from deployment on embedded hearing-aid hardware. Future work will explore autoregressive codec-token prediction for fully causal generation with reduced buffering requirements, potentially lowering algorithmic latency, alongside model compression and low-power implementation.

## ACKNOWLEDGMENTS

This work was supported by the Accelerating Higher Education Expansion and Development (AHEAD) operation (Grant No. 6026-LK/8743-LK, World Bank). The authors also thank all participants who took part in the listening study.

## REFERENCES

[1] A. W. Bronkhorst, “The cocktail party phenomenon: A review of research on speech intelligibility in multiple-talker conditions,” Acta Acust. united Ac., vol. 86, no. 1, pp. 117–128, 2000.

[2] E. C. Cherry, “Some experiments on the recognition of speech, with one and with two ears,” J. Acoust. Soc. Amer., vol. 25, no. 5, pp. 975–979, 1953.

[3] B. G. Shinn-Cunningham and V. Best, “Selective attention in normal and impaired hearing,” Trends Amplif., vol. 12, no. 4, pp. 283–299, 2008.

[4] M. K. Pichora-Fuller, S. E. Kramer, M. A. Eckert, B. Edwards, B. W. Y. Hornsby, L. E. Humes et al., “Hearing impairment and cognitive energy: The framework for understanding effortful listening (FUEL),” Ear Hear., vol. 37, no. Suppl. 1, pp. 5S–27S, 2016.

[5] J. E. Peelle, “Listening effort: How the cognitive consequences of acoustic challenge are reflected in brain and behavior,” Ear Hear., vol. 39, no. 2, pp. 204–214, 2018.

[6] K. Zmolikova, M. Delcroix, T. Ochiai, K. Kinoshita, J. Cernock<sup>ˇ</sup> y, and´ D. Yu, “Neural target speech extraction: An overview,” IEEE Signal Process. Mag., vol. 40, no. 3, pp. 8–29, 2023.

[7] D.-J. Alcala Padilla, N. L. Westhausen, S. Vivekananthan, and B. T. Meyer, “Location-aware target speaker extraction for hearing aids,” in Proc. Interspeech, 2025, pp. 2975–2979.

[8] S. A. Indrapala and W. N. Manamperi, “BiTSE: Binaural target speaker extraction in noisy multi-talker environments for AR glass arrays,” arXiv preprint arXiv:2608.10106, Aug. 2026.

[9] T. Wu, S. He, J. Pan, H. Huang, Z. Mo, and X. Zhang, “Unified audio visual cues for target speaker extraction,” in Proc. Interspeech, 2024, pp. 4343–4347.

[10] J. Li, K. Zhang, S. Wang, K. A. Lee, M.-W. Mak, and H. Li, “MoMuSE: Momentum multi-modal target speaker extraction for real-time scenarios with impaired visual cues,” in Proc. IEEE Int. Conf. Multimedia Expo (ICME), 2025, pp. 1–6.

[11] N. Mesgarani and E. F. Chang, “Selective cortical representation of attended speaker in multi-talker speech perception,” Nature, vol. 485, no. 7397, pp. 233–236, 2012.

[12] N. Ding and J. Z. Simon, “Emergence of neural encoding of attended speech in a multi-speaker environment,” J. Neurosci., vol. 32, no. 43, pp. 14 772–14 778, 2012.

[13] W. Biesmans, N. Das, T. Francart, and A. Bertrand, “Auditory-inspired speech envelope extraction methods for improved EEG-based auditory attention detection in a cocktail party scenario,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 5, pp. 402–412, 2017.

[14] C. Fan, Y. Chen, J. Zhou, Z. Pan, J. Zhang, Y. Gao, X. Yang, Z. Wen, and Z. Lv, “M3ANet: Multi-scale and multi-modal alignment network for brain-assisted target speaker extraction,” in Proc. Int. Joint Conf. Artif. Intell. (IJCAI), 2025, pp. 8040–8048.

[15] Z. Pan, M. Borsdorf, S. Cai, T. Schultz, and H. Li, “NeuroHeed: Neurosteered speaker extraction using EEG signals,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 4456–4470, 2024.

[16] D. Da Silva, S. Cai, S. Pahuja, T. Schultz, and H. Li, “NeuroSpex+: Dual-task training of neuro-guided speaker extraction with speech envelope and waveform,” in Proc. Interspeech, 2025, pp. 5568–5572.

[17] Z. Pan, G. Wichern, F. G. Germain, S. Khurana, and J. Le Roux, “NeuroHeed+: Improving neuro-steered speaker extraction with joint auditory attention detection,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2024, pp. 11 456–11 460.

[18] E. Ceolini, J. Hjortkjær, D. D. E. Wong, J. O’Sullivan, V. S. Raghavan, J. Herrero, A. D. Mehta, S.-C. Liu, and N. Mesgarani, “Brain-informed speech separation (BISS) for enhancement of target speaker in multitalker speech perception,” NeuroImage, vol. 223, p. 117282, 2020.

[19] J. Zhang, Q. Xu, Q.-S. Zhu, and Z.-H. Ling, “BASEN: Time-domain brain-assisted speech enhancement network with convolutional cross attention in multi-talker conditions,” in Proc. Interspeech, 2023, pp. 3117–3121.

[20] D. D. Silva, S. Cai, S. Pahuja, T. Schultz, and H. Li, “CA-NeuroSpex: Context-informed autoregressive neuro-guided speaker extraction,” in Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), 2025, pp. 1–7.

[21] J. Richter, S. Welker, J.-M. Lemercier, B. Lay, and T. Gerkmann, “Speech enhancement and dereverberation with diffusion-based generative models,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, pp. 2351–2364, 2023.

[22] A. Navon, A. Shamsian, Y. Segal-Feldman, N. Glazer, G. Hetz, and J. Keshet, “FlowTSE: Target speaker extraction with flow matching,” in Proc. Interspeech, 2025, pp. 2965–2969.

[23] X. Wang, M. Thakker, Z. Chen, N. Kanda, S. E. Eskimez, S. Chen, M. Tang, S. Liu, J. Li, and T. Yoshioka, “SpeechX: Neural codec language model as a versatile speech transformer,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 3355–3364, 2024.

[24] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “SoundStream: An end-to-end neural audio codec,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 30, pp. 495–507, 2022.

[25] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” Trans. Mach. Learn. Res., 2023.

[26] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “Highfidelity audio compression with improved RVQGAN,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 27 980–27 993.

[27] M. P. Broderick, A. J. Anderson, G. M. D. Liberto, M. J. Crosse, and E. C. Lalor, “Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech,” Curr. Biol., vol. 28, no. 5, pp. 803–809.e3, 2018.

[28] X. Bai, J. Tan, H. Hu, C. Zhang, and D. Gu, “SGCRNN: A ChebNet-GRU fusion model for EEG emotion recognition,” J. Intell. Fuzzy Syst., vol. 45, no. 6, pp. 10 545–10 561, 2023.

[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 5998–6008.

[30] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2024.

[31] H. Guo, H. Lu, X. Wu, and H. Meng, “A multi-scale time-frequency spectrogram discriminator for GAN-based non-autoregressive TTS,” in Proc. Interspeech, 2022, pp. 1566–1570.

[32] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis,” in Adv. Neural Inf. Process. Syst., vol. 33, 2020, pp. 17 022–17 033.

[33] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1195–1204.

[34] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., vol. 32, no. 1, 2018, pp. 3942–3951.

[35] S. Bengio, O. Vinyals, N. Jaitly, and N. Shazeer, “Scheduled sampling for sequence prediction with recurrent neural networks,” in Adv. Neural Inf. Process. Syst., vol. 28, 2015, pp. 1171–1179.

[36] A. Delorme and S. Makeig, “EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics including independent component analysis,” J. Neurosci. Methods, vol. 134, no. 1, pp. 9–21, 2004.

[37] C. Fan, J. Zhang, H. Zhang, W. Xiang, J. Tao, X. Li, J. Yi, D. Sui, and Z. Lv, “MSFNet: Multi-scale fusion network for brain-controlled speaker extraction,” in Proc. 32nd ACM Int. Conf. Multimedia, 2024, pp. 1652–1661.

[38] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An algorithm for intelligibility prediction of time-frequency weighted noisy speech,” IEEE Trans. Audio, Speech, Lang. Process., vol. 19, no. 7, pp. 2125– 2136, 2011.

[39] J. Jensen and C. H. Taal, “An algorithm for predicting the intelligibility of speech masked by modulated noise maskers,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 24, no. 11, pp. 2009–2022, 2016.

[40] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (PESQ)—a new method for speech quality assessment of telephone networks and codecs,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), vol. 2, 2001, pp. 749–752.

[41] D. De Silva, S. Cai, S. Pahuja, T. Schultz, and H. Li, “NeuroSpex: Neuro-guided speaker extraction with cross-modal fusion,” in Proc. IEEE Spoken Lang. Technol. Workshop (SLT), 2024, pp. 341–348.

[42] S. Li, S. Liu, L. Zhang, X. Li, Y. Bian, C. Weng, Z. Wu, and H. Meng, “SnakeGAN: A universal vocoder leveraging DDSP prior knowledge and periodic inductive bias,” in Proc. IEEE Int. Conf. Multimedia Expo (ICME), 2023, pp. 1703–1708.

[43] M. Schoeffler, S. Bartoschek, F.-R. Stoter, M. Roess, S. Westphal,¨ B. Edler, and J. Herre, “webMUSHRA—a comprehensive framework for web-based listening tests,” J. Open Res. Softw., vol. 6, no. 1, p. 8, 2018.