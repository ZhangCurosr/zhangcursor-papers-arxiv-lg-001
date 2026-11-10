# AUDIOCODEX: STREAMING AUTOENCODERS FOR LIVE AUDIO

Marco Pasini<sup>2</sup> Javier Nistal<sup>1</sup> Cyran Aouameur<sup>1</sup> Stefan Lattner<sup>1</sup> György Fazekas<sup>2</sup>

<sup>1</sup>Sony Computer Science Laboratories, Paris, France

<sup>2</sup>Queen Mary University of London, UK

m.pasini@qmul.ac.uk

## ABSTRACT

We introduce AudioCodex, a fully causal autoencoder for live generative audio applications that compresses stereo audio into compact continuous latent sequences at ∼ 10 Hz. It combines a Mean Flow magnitude autoencoder with latent hypersphere projection, bounded-noise augmentation, representation alignment, and adversarial phase recovery to learn representations that support both efficient downstream generation and high-quality reconstruction. Through extensive ablations, we show that AudioCodex outperforms the state-of-the-art in terms of generation-reconstruction trade-off, whilst remaining strictly causal in both encoding and decoding. This work paves the way towards truly real-time interactive and streaming generative audio applications. Supplementary results and audio samples are available in the website. <sup>1</sup>

## 1. INTRODUCTION

Compact, structured latent representations of audio are central to both generative modeling [1–3] and music information retrieval [4–6]. Downstream models operating in such spaces, rather than directly on waveforms, become substantially more efficient during training and inference.

In reconstruction-oriented applications, especially in generative modeling, autoencoders are the standard choice, typically learning deterministic mappings between audio signals and latent representations [7–10]. Yet existing audio autoencoders often expose a fundamental tension: representations that preserve detailed acoustic information can be difficult for generative models to learn [11–14], whereas highly regularized or compressed latents may improve generative modeling at the cost of reconstruction fidelity [15]. Stochastic generative decoders help relax this tension [2, 16, 17] by decoupling reconstruction and audio quality: the latent carries global structure, while the decoder plausibly synthesizes missing fine detail. The objective is therefore not exact sample reconstruction, but a representation that is both generative-friendly and perceptually high-quality after decoding [18].

Recently, real-time audio applications are gaining traction—e.g., conversational speech [19, 20], musical accompaniment [3, 21, 22], interactive synthesis [23–25]— requiring these systems to remain strictly causal. While existing work addresses causal audio autoencoding for speech, these methods typically produce discrete tokens rather than continuous latents [7, 8, 20, 26, 27], rendering them less suitable for certain generative modeling frameworks [28–32].

We introduce AudioCodex, afully causal and continuous audio autoencoder designed for real-time generative use. The model is trained as a generative autoencoder on log-magnitude spectrograms, with a separate adversarial phase estimator for waveform recovery. This decoupling facilitates high-quality reconstruction by removing the noisy phase components from the autoencoder training process. The magnitude decoder is trained as a Mean Flow model [33], enabling single-step decoding and stable training compared to other one-step frameworks [34]. To the best of our knowledge, this is the first Mean Flow-based autoencoder in any modality.

Our contributions are fivefold: (1) a fully causal autoencoder for general high sample rate stereo audio, designed for streaming to and from ∼ 10 Hz compressed representations; (2) a magnitude-only formulation with separate adversarial phase estimation, trained without direct reconstruction objectives; (3) a sphere-normalized bottleneck with bounded latent noise augmentation [15, 35–37]; (4) Representation Alignment (REPA) [38] with a frozen selfsupervised audio encoder [39]; and (5) a comprehensive analysis of the reconstruction-generation trade-off using both reconstruction and downstream-generative metrics.

## 2. RELATED WORK

Neural audio autoencoders have largely followed two directions. Discrete codecs combine convolutional encoder– decoder architectures with residual vector quantization [7– 9, 40], yielding representations that are well suited to reconstruction-based language-modeling objectives such as next-token prediction [41, 42]. However, they are less directly compatible with generative frameworks commonly defined in continuous spaces [28–32]. Inspired by works in vision [36, 43, 44], continuous audio autoencoders address this limitation [2, 16, 17, 45]. Yet they still face a reconstruction-generation trade-off: latents optimized for faithful reconstruction can be difficult to model downstream, whereas more regularized representations may compromise reconstruction fidelity [11–13, 18, 46]. AudioCodex is designed to improve this trade-off while remaining strictly causal, unlike prior continuous autoencoders for general audio, which typically rely on bidirectional processing and non-causal decoding [1,2,10,16,45]. This makes it suitable for streaming generative applications, where causal autoencoders have so far been mostly speech-oriented and tokenbased [20, 26, 27]. Finally, AudioCodex separates magnitude and phase modeling. Since wrapped phase induces an irregular geometry in complex STFTs [2, 16, 17], and phase errors can interfere with magnitude prediction [47], we encode only log-magnitude spectrograms and recover phase separately. This connects our design to neural vocoding [48–51] while allowing the magnitude autoencoder to be trained with a single flow-based objective.

## 3. BACKGROUND

AudioCodex builds on improved Mean Flows [33, 52] for generative autoencoding, Representation Alignment (REPA) [38] for semantic feature distillation into the compressed latents, noise augmentation for latent regularization, and the R3GAN adversarial framework [53] for phaseestimator training.

3.1 Improved Mean Flow (iMF) [52]. A Rectified Flow (RF) [54] model learns to reach a data sample from noise by integrating a velocity field. It is trained with data-noise interpolations $x _ { t } = ( 1 - t ) x +$ tϵ as input to approximate the conditional instantaneous velocity $v _ { c } = \epsilon - x .$ . Mean Flow [33] generalizes this idea by modeling the average velocity between two noise levels $r < t ,$ rather than only the instantaneous velocity used in standard RF [54]:

$$
u ( x _ { t } , r , t ) = \frac { 1 } { t - r } \int _ { r } ^ { t } v ( x _ { \tau } , \tau ) d \tau .\tag{1}
$$

We can write this as $\begin{array} { r } { ( t - r ) u \ = \ \int _ { r } ^ { t } v d \tau } \end{array}$ , and differentiating w.r.t. t and rearranging the terms gives us the Mean Flow identity:

$$
u ( x _ { t } , r , t ) = v ( x _ { t } , t ) - ( t - r ) \frac { d } { d t } u ( x _ { t } , r , t ) .\tag{2}
$$

Mean Flow then trains $u _ { \theta }$ by matching this identity with a stop-gradient target:

$$
\mathcal { L } _ { \mathrm { M F } } = \mathbb { E } \left. u _ { \theta } ( x _ { t } , r , t ) - \mathrm { s g } [ u _ { \mathrm { t g t } } ] \right. _ { 2 } ^ { 2 }\tag{3}
$$

$$
u _ { \mathrm { t g t } } = v - ( t - r ) \left( v \partial _ { z } u _ { \theta } + \partial _ { t } u _ { \theta } \right)\tag{4}
$$

where $\scriptstyle { \frac { d } { d t } } u _ { \theta }$ is computed as a JVP. In practice, v is replaced by the conditional velocity $v _ { c } = \epsilon - x$ . iMF [52] reformulates training as a regression on instantaneous velocity. By rearranging Eq. 4 we get the predicted instantaneous velocity:

$$
V _ { \theta } ( x _ { t } , r , t ) = u _ { \theta } ( x _ { t } , r , t ) + ( t - r ) \mathrm { s g } [ \mathrm { J V P } ( u _ { \theta } ; v _ { \theta } ) ] ,\tag{5}
$$

where $v _ { \theta }$ is predicted from the same network at the boundary condition $r = t .$ . The iMF objective is then

$$
\mathcal { L } _ { \mathrm { i M F } } = \mathbb { E } \left. V _ { \theta } ( x _ { t } , r , t ) - v _ { c } \right. _ { 2 } ^ { 2 } ,\tag{6}
$$

This loss provides an empirically stable [52] velocityregression view of the Mean Flow objective.

3.2 Representation Alignment (REPA) [38] augments the training objective by aligning intermediate model features with those of a frozen teacher network using a cosine similarity loss. This provides semantically informed gradients early in training and encourages the latent space to organize around the structure already captured by the teacher encoder. In AudioCodex, REPA guides the magnitude autoencoder toward more generative-friendly representations without significantly affecting reconstruction performance.

3.3 Noise Augmentation [15, 36] perturbs latent representations during training to improve downstream generation. Rather than training only on clean latents $z ,$ the decoder is trained on noisy interpolants,

$$
\begin{array} { r } { z ^ { \prime } = ( 1 - t ) z + t n , \qquad n \sim \mathcal { N } ( 0 , I ) , \qquad t \sim \mathcal { U } ( 0 , 1 ) . } \end{array}\tag{7}
$$

This regularization promotes perceptually meaningful latent hierarchies, with coarse structures capturing salient content and finer variations encoding high-frequency detail. The latent variance is fixed through layer normalization before interpolation, preventing the encoder from counteracting the noise by increasing its scale. Inspired by prior work that synchronizes latent and data noise levels [36], we propose a novel bounded noise augmentation strategy that significantly improves generative performance at limited to no reconstruction cost.

3.4 R3GAN [53] uses a relativistic pairing objective [55] where the discriminator compares fake and real samples directly:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e l } } ( \theta , \psi ) = \mathbb { E } _ { z \sim p ( z ) , x \sim p _ { \mathcal { D } } } \Big [ f \big ( D _ { \psi } ( G _ { \theta } ( z ) ) - D _ { \psi } ( x ) \big ) \Big ] . } \end{array}\tag{8}
$$

To keep training stable, R3GAN applies zero-centered penalties on both real and fake data:

$$
R _ { 1 } ( \psi ) = \frac { \gamma } { 2 } \mathbb { E } _ { x \sim p _ { \mathcal { D } } } \left[ \lVert \nabla _ { x } D _ { \psi } ( x ) \rVert _ { 2 } ^ { 2 } \right]\tag{9}
$$

$$
R _ { 2 } ( \theta , \psi ) = \frac { \gamma } { 2 } \mathbb { E } _ { x \sim p _ { \theta } } \left[ \| \nabla _ { x } D _ { \psi } ( x ) \| _ { 2 } ^ { 2 } \right]\tag{10}
$$

The discriminator maximizes $\mathscr { L } _ { \mathrm { r e l } } - R _ { 1 } - R _ { 2 }$ , while the generator minimizes ${ \mathcal { L } } _ { \mathrm { r e l } } .$ . The relativistic objective improves diversity and avoids mode collapse, and $R _ { 1 } + R _ { 2 }$ damps unstable gradients on both real and fake data manifolds.

## 4. AUDIOCODEX

AudioCodex consists of two independently trained stages: a magnitude autoencoder that maps log-magnitude spectrograms to and from a continuous latent representation, and a phase estimator that reconstructs phase from the decoded magnitudes. This decoupling keeps the rapidly varying phase structure out of the autoencoder objective, addressing the poor phase reconstructions often observed in complex-STFT autoencoders [2, 16, 17]. While this decoupling certainly adds complexity, our goal is to keep the training processes of the two modules simple and scalable.

4.1 Audio representation. Given an input waveform $w ,$ we compute its Short-Time Fourier Transform (STFT) and retain only the log-magnitude $x = \big ( 2 0 \log _ { 1 0 } ( | \mathrm { S T F T } ( w ) | +$ $\epsilon ) - \mu ) / \sigma$ , where $\mu$ and σ are scalar normalization parameters. Stereo audio is processed by treating the two channels as feature maps stacked along the channel axis.

![](images/f786f9ca84f2958db2de0f0c51dec94a2fac58162b9a5fcee1fe0c22b1d5d213.jpg)

![](images/8751d52542773bfff61396f36200716b7fb536f8427265f463c2b3403ca4240e.jpg)  
(a) The causal encoder produces compact sphere-normalized latents. Learnable summary embeddings (Sum Emb) are appended to each frame before the frequency-wise transformer.  
(b) G receives upsampled latent representations and spectrogram-shaped gaussian noise. U provides input-level conditioning and multi-scale guidance to G before magnitude-to-waveform reconstruction.  
Figure 1: AudioCodex’s autoencoder main architecture. Blue denotes learnable components.

4.2 Magnitude Autoencoder. Fig. 1 depicts AudioCodex’s magnitude autoencoder. In the encoder, a convolutional patchifier progressively downsamples the input spectrogram along time and frequency, resulting in a sequence of feature frames with temporal resolution $f _ { z } , \mathbf { A }$ set of K learnable summary embeddings is appended to each temporal frame [16, 17]. Each summary-augmented frame is then processed independently by a frequency-wise transformer. After the transformer, only the updated summary embeddings are retained, concatenated, flattened, and projected through the bottleneck to form a compact latent representation at resolution $f _ { z }$

The decoder consists of an upsampler U and a generator $G .$ The upsampler mirrors the encoder, lifting the compact latents into a spectrogram-shaped representation that matches the input resolution of $G .$ This representation is concatenated with Gaussian noise at the generator input [17], while intermediate feature maps from U are injected into $G \ ' \mathrm { s }$ U-Net at matching resolution levels. In this way, $U$ provides both early conditioning and multiscale guidance for the 1-step generation. The generator follows the same architectural principles as the encoder: a time-causal convolutional patchifier, a frequency-wise transformer, and a de-patchifier, augmented with standard U-Net skip connections.

Causality is enforced by using time-causal convolutions in all patchifiers and causal group normalization in all convolutional blocks, where normalization statistics are accumulated only from past context. Transformer layers use layer normalization and operate along the frequency axis independently for each ∼ 10 Hz time frame. At inference time, the model can run incrementally in streaming mode by caching activations and processing only newly available input [23, 25].

4.3 Bottleneck and Bounded Noise Augmentation. Sphere Encoder [37] shows that projecting autoencoder latents onto the hypersphere and performing noise augmentation results in a latent distribution that is easy to sample from. We thus adopt the idea and RMS-normalize the encoder latents $z \in \bar { \mathbb { R } } ^ { d _ { z } }$ [56]. During training, we perform bounded noise augmentation. This matches the Sphere Encoder idea: noisy latent clouds are formed by adding

Gaussian noise and projecting back to the sphere. Our implementation keeps the same intuition but uses interpolation rather than direct addition:

$$
\tilde { z } _ { i } = \mathrm { R M S N o r m } \big ( ( 1 - \alpha _ { i } ) \hat { z } _ { i } + \alpha _ { i } \hat { n } _ { i } \big ) , \quad \alpha _ { i } \sim \mathcal { U } ( 0 , t _ { i } ) ,\tag{11}
$$

where $\hat { n } _ { i }$ is a sphere-normalized Gaussian noise vector and $t _ { i }$ is the data-space noise level used for the corresponding noisy magnitude spectrogram in batch element i. Thus the latent corruption is capped per batch element: when the input is only mildly noisy, the latent stays mildly noisy too, which preserves access to latent high-frequency information. This can also be considered as a simplified form of noise synchronization [36], without additional hyperparameters or layernorm. In practice, this preserves reconstruction performance, while still encouraging a generationfriendly representation [15].

4.4 Training. The magnitude autoencoder is trained using the iMF framework. For a normalized log-magnitude spectrogram x and noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , we define

$$
x _ { t } = ( 1 - t ) x + t \epsilon , \qquad v _ { c } = \epsilon - x .\tag{12}
$$

The start noise level t is sampled from a logit-normal distribution, and the end time $r$ is sampled as

$$
r = \bigl ( 1 - \xi ^ { \alpha } \bigr ) t , \quad \xi \sim \mathcal { U } ( 0 , 1 ) ,\tag{13}
$$

The original Mean Flow formulation samples t, r independently, enforces $r < t$ by swapping, and finally sets $r = t$ 75% of the times. We instead first sample t as in standard rectified-flow training, and only then sample r conditionally from t. This avoids distorting the marginal distribution of t. Many noise levels where r ≈ t are still sampled (through the α-exponentiated transform), so the model keeps a strong instantaneous-velocity learning signal and smoothly propagates it to larger jumps. Given $( x _ { t } , t , r )$ , we first estimate the instantaneous velocity at $r \ = \ t$

$$
\hat { v } _ { \theta } ( x _ { t } , t ) = u _ { \theta } ( x _ { t } , t , t ) ,\tag{14}
$$

which is then used as the JVP tangent to evaluate $\frac { d } { d t }$ u<sub>θ</sub> at $( x _ { t } , r , t )$ . The final velocity prediction is

$$
V _ { \theta } ( x _ { t } , r , t ) = u _ { \theta } ( x _ { t } , r , t ) + ( t - r ) \mathrm { s g } \biggl [ \frac { d } { d t } u _ { \theta } ( x _ { t } , r , t ) \biggr ] .\tag{15}
$$

We use the adaptive MSE loss proposed in Mean Flow, with an additional perceptual frequency weighting. Let $\Delta _ { i } = V _ { \theta , i } - v _ { c , i }$ be the per-sample velocity error and M a fixed mel-derived frequency weight. The implemented per-sample loss is

$$
\mathcal { L } _ { \mathrm { M F } } = \frac { 1 } { B } \sum _ { i } \frac { | | M \odot \Delta _ { i } | | _ { 2 } ^ { 2 } } { ( \mathrm { s g } | | \Delta _ { i } | | _ { 2 } ^ { 2 } + c ) ^ { p } } , \quad M = \frac { 1 } { 1 + \frac { F } { 7 0 0 } } .\tag{16}
$$

where $p \in \ [ 0 , 1 ]$ controls sub-quadratic behaviour, $c >$ 0 is a numerical constant, stop-gradient is applied to the denominator term, and $F$ is the range of frequency bins of the STFT configuration in use.

Classifier-free guidance (CFG) is distilled into one-step training targets. Concretely, for ∼ 10% of batch elements during training, we run two predictions at $r \ = \ t :$ one conditional and one unconditional,

$$
\hat { v } _ { i } ^ { z } = V _ { \theta } ( z _ { t , i } , t _ { i } , t _ { i } \mid z ) , \quad \hat { v } _ { i } ^ { \varpi } = V _ { \theta } ( z _ { t , i } , t _ { i } , t _ { i } \mid \emptyset ) .\tag{17}
$$

We then form a distilled target velocity

$$
\tilde { v } _ { c , i } = v _ { c , i } + \mathbf { 1 } _ { [ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ] } ( t _ { i } ) \omega \big ( \hat { v } _ { i } ^ { z } - \hat { v } _ { i } ^ { \mathcal { O } } \big ) ,\tag{18}
$$

and train the main forward pass against $\tilde { v } _ { c }$ on that subset. Outside the subset, $\tilde { v } _ { c } ~ = ~ v _ { c }$ . Also note that we apply CFG only in the (0.2, 0.8) noise level interval [57]. The model is explicitly informed about mode through a learned conditioning embedding with three states (distilled-CFG mode, conditional mode, unconditional mode), injected through the adaptive normalization layers of all modules.

We further use Representation Alignment (REPA) [38]. A frozen self-supervised teacher encoder produces a feature sequence $Y ,$ . We extract one designated hidden representation from each of our three modules (encoder, upsampler, and decoder transformers), and project each sequence to the teacher representation shape with distinct 2-layer alignment probes. If Y<sup>ˆ</sup> <sup>(m)</sup> denotes the projected feature sequence from each module $m ,$ the implemented REPA loss is:

$$
\mathcal { L } _ { \mathrm { R E P A } } = \frac { 1 } { 3 } \sum _ { m } \Bigl ( 1 - \cos \bigl ( \hat { Y } ^ { ( m ) } , Y \Bigr ) \Bigr ) ,\tag{19}
$$

evaluated over all time positions and batch elements. The final magnitude-autoencoder objective is

$$
\mathcal { L } _ { \mathrm { A E } } = \mathcal { L } _ { \mathrm { M F } } + \lambda _ { \mathrm { R E P A } } \mathcal { L } _ { \mathrm { R E P A } }\tag{20}
$$

4.5 Phase Estimator. The magnitude autoencoder does not model phase, so waveform reconstruction requires a separate, fully causal phase estimator. During training, the phase estimator takes ground-truth log-magnitude spectrograms and Gaussian noise as input, and predicts phase information. To avoid phase discontinuities, we represent phase as interleaved cosine and sine channels: each frequency bin k is represented as $[ \cos ( \phi _ { k } ) , \sin ( \phi _ { k } ) \rangle$ ], normalized to unit magnitude during training. The architecture mirrors the magnitude generator: a time-causal convolutional U-Net with the same temporal downsampling factor (∼ 10 Hz), but without transformer layers. Training follows the R3GAN framework (Sec. 3) with two discriminators. The main discriminator is magnitude-conditioned and evaluates whether a predicted phase matches a given magnitude. We then reconstruct the waveform using the inverse STFT, and compute the power-compressed magnitude mel-spectrogram, possibly using a different STFT configuration and powercompression factor γ, and use it as the input to an auxiliary discriminator [50, 58]. Let $D _ { \mathrm { m a i n } }$ denote the main magnitude-conditioned discriminator and $D _ { \mathrm { a u x } }$ denote the auxiliary discriminator. The generator loss is

$$
\begin{array} { r } { \mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { r e l } } ( D _ { \mathrm { m a i n } } ) + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { r e l } } ( D _ { \mathrm { a u x } } ) , } \end{array}\tag{21}
$$

where $\mathcal { L } _ { \mathrm { r e l } }$ is the relativistic loss from R3GAN (Sec. 3). The discriminator loss is

$$
\mathcal { L } _ { D } = \mathcal { L } _ { \mathrm { r e l } } ( D _ { \mathrm { m a i n } } , D _ { \mathrm { a u x } } ) + \gamma _ { \mathrm { m a i n } } R _ { 1 + 2 , \mathrm { m a i n } } + \gamma _ { \mathrm { a u x } } R _ { 1 + 2 , \mathrm { a u x } } .\tag{22}
$$

where the relativistic loss is evaluated jointly across both discriminators, and $R _ { 1 } , R _ { 2 }$ denote the zero-centered gradient penalties on real and fake data respectively, with fixed weights $\gamma _ { 1 }$ and $\gamma _ { 2 }$

## 5. EXPERIMENTAL SETUP

5.1 Training data. We train AudioCodex on a mixed collection of permissively licensed datasets, using the same data recipe for both the magnitude autoencoder and the phase estimator. The training mixture is: Unsupervised People’s Speech [59] (in-the-wild speech/vocal data that also includes music and general audio, 40% sampling weight), Freesound [60] (general audio, 30%), and FMA [61] (music, 30%). All audio samples are kept in their original sampling rate (44.1 or 48 kHz).

5.2 Evaluation protocol. Our evaluation is organized into three parts: (i) targeted ablations, (ii) comparison to baselines, and (iii) streaming inference speed. The ablations are the core of this study and cover the main design choices of both the magnitude autoencoder and the phase estimator. Baseline comparison and latency benchmarking are then used to position the final selected configuration in terms of quality and real-time usability.

For evaluation, we build four domain-specific sets and collect 5k non-silent clips of 10 s per domain: stems from MUSDB18-HQ [62], music from MusicCaps [42], speech from HiFiTTS [63], and general audio from SoundBible [64]. Before cropping, all source audio is resampled to 44.1 kHz when needed, and mono files are converted to stereo by channel duplication. Most source material is already stereo at either 44.1 kHz or 48 kHz. We release metadata for all evaluation samples, including the specific crops sampled, to ensure reproducibility.

5.3 Metrics. We use Fréchet Audio Distance (FAD) [65] computed with both VGGish [66] and CLAP [4] embeddings. For reconstruction experiments, to evaluate audio quality, we report reconstruction FAD (rFAD), obtained by comparing embeddings from 10 s reconstructions with those from the original 10 s test samples. rFAD uses VGGish embeddings, while $\mathrm { r F A D _ { C } }$ uses CLAP embeddings. When comparing to baselines, we also report L1 distances on dB-scale spectrograms and mel-spectrograms, respectively denoted STFT and MEL. We do not include SI-SDR [67] since it measures exact-phase matching, and in our case phase is stochastically reconstructed.

To evaluate the quality of the latents for downstream generation, we use a fixed, lightweight proxy task: we first obtain latent sequences from a trained encoder, then train a small-scale latent rectified-flow transformer (4 layers, model dimension 512) for 40k iterations on latent sequences with a length of 16, and decode generated samples back to waveform space. FAD is then measured. A length of 16 corresponds to ∼ 1.5 s at our target latent rate $\mathrm { { ( \sim 1 0 H z , } }$ assuming 44.1 kHz audio). For baseline representations produced at higher temporal rates (e.g., 20 Hz), we concatenate adjacent timesteps along the channel axis so the transformer is still trained on length-16 sequences covering the same ∼ 1.5 s span, which keeps the metric fair for different rates and compression ratios. For downstream generation experiments, we report generative $F A D \left( \mathrm { g F A D } \right)$ obtained by comparing samples from the lightweight latent generator against the target data distribution after decoding. gFAD only uses VGGish embeddings since the $\sim 1 . 5 \ : \mathrm { s }$ samples are too short for CLAP.

5.4 Ablation protocol. All ablations for both the magnitude autoencoder and the phase estimator are trained for 200k iterations with fixed learning rate $5 e - 5$ . For the magnitude autoencoder, the reference configuration is: Mean Flow with data-space prediction (x-prediction), sphere-normalized bottleneck with $d _ { z } = 2 5 6$ , bounded latent noise augmentation, CFG distillation weight 0.5, causal encoder/decoder, and $\alpha = 2 . 5$ . For the phase estimator, the selected reference configuration is: causal model, both discriminators enabled, power-compression factor $\gamma = 0 . 4$ and $\lambda _ { \mathrm { a u x } } = 1$ . Non-targeted factors remain fixed for each ablation. For ablations, we reduce the evaluation sets to 1k clips per domain and for brevity only report the average of the in-domain r/gFADs. This makes ablation metrics not directly comparable with Tab. 2.

5.5 Baselines. We compare against five representative autoencoders. M2L refers to a recent continuous audio autoencoder using consistency models [2]; CDC denotes a hybrid continuous/discrete representation model [17], which we run in continuous mode; SAO denotes the VAE used in Stable Audio Open [10]; and DAC denotes a discrete residual-vector-quantized codec [9], which is only included as a reference since it is not directly comparable. Our evaluation deliberately does not consider compression ratios: we are mainly interested in finding the best trade-off between reconstruction and generation, even at the cost of lower compression.

5.6 Implementation details. AudioCodex processes stereo audio using the representation defined in Sec. 4.1. We use an STFT with window size W = 2048 and hop size $H = 1 0 2 4 ,$ and normalize log-magnitude spectrograms with $\mu \ : = \ : - 3 5$ and $\sigma = 7 5$

For the magnitude autoencoder (Sec. 4.2), the convolutional patchifiers use 4 resolution levels, with 64 base channels and per-resolution channel multipliers [1, 2, 4, 8], [3, 3, 3, 1] layers per-level, and downsampling strides [(2, 2), (1, 4), (2, 2)] across (time, frequency) axes. Everything is inverted for all de-patchifiers. Two 3x3 convolutions and a middle SiLU [68] are used in the residual blocks. The transformer backbones in the encoder, upsampler, and generator each use 12 layers, a model dimension of 512, a head dimension of 128, and a feed-forward expansion factor of 4 with SwiGLU activations. At the bottleneck, the encoder outputs $K = 8$ summary embeddings. We jointly train different compression regimes by randomly sampling $d _ { z } \in [ 1 2 8 , 2 5 6 , 5 1 2 ]$ for each element in the batch, and investigate $d _ { z } \mathrm { ' s }$ impact on the reconstruction-generation tradeoff (see Tab. 1). For REPA, we use USAD-Small [39] as the frozen teacher with $\lambda _ { \mathrm { R E P A } } = 0 . 1 ( \mathrm { E q . } 2 0 )$ , extracting features from the 4th layer in the upsampler and the generator, and the 8th layer in the encoder. We choose it since it was trained with both general audio and speech signals. The dimension of the alignment probes is 512. iMF uses $p = 0 . 5$ $t \sim L o g i t N o r m a l ( \mu = 0 , \sigma = 1 )$ , and r is sampled using $\alpha = 2 . 5 \ : ( \mathrm { E q . } 5 )$ . We optimize the magnitude autoencoder with CRAdam [69–71], using peak learning rate $1 0 ^ { - 4 }$ , cosine decay to zero, $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ , weight decay 0.01, and 10k warm-up steps. Training runs for 500k iterations with a batch size of 72 on $3 \mathrm { A l 0 0 { \mathrm { G P U s } } , }$ , and we maintain an exponential moving average with decay 0.9999. The full magnitude autoencoder contains ∼ 150 M parameters.

The phase estimator uses the same causal UNet as $G ,$ and does not use the transformer blocks, which results in a ∼ 70 M parameter model. Both discriminators use 64 base channels, 6 resolutions levels, [1, 2, 4, 8, 8, 8] as the per-resolution channel multipliers, [3, 3, 3, 3, 3, 3] layer per-level. The downsampling strides for the main discriminator are [(2, 2), (1, 4), (2, 2), (1, 4), (2, 2)] for the (time, freq) axes, and for the auxiliary one we use [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]. Leaky ReLUs are used in the residual blocks. For the auxiliary discriminator, we use a STFT window of 1024 and hop size of 256, with 108 mel bins. We use $\lambda _ { \mathrm { a u x } } = 1 , \gamma _ { \mathrm { m a i n } } = 5 0 , \gamma _ { \mathrm { a u x } } = 0 . 1$ We train for 200k iterations with a fixed learning rate of $1 0 ^ { - 4 }$ , using Adam [70] with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 5 , 0 . 9 )$ for both generator and discriminator.

## 6. RESULTS

We provide audio examples and supplementary results on all audio domains at this link <sup>2</sup> .

6.1 Magnitude Autoencoder Ablations. Tab. 1 reports rFAD and gFAD for all magnitude autoencoder ablations, averaged over our four audio domains. We first train a consistency model using complex STFT representations, similarly to Music2Latent and CoDiCodec (we use the framework presented in [17] with our causal architecture). Results show how this variant underperforms our Mean Flow decoupled framework. We also show how x-prediction [72] is slightly better than v-prediction. For these first ablations, we do not distill CFG, so that we can compare with the consistency-trained variant where CFG distillation is not directly possible. For bottleneck design, using bounded noise augmentation and REPA improves generative quality compared to tanh [2, 73], layernorm [15, 36], and plain RMS-Norm, at a minor reconstruction cost. Uniform noise augmentation (where the noise level is always sampled from $\boldsymbol { \mathcal { U } } ( 0 , 1 ) )$ severely degrades reconstruction. Bottleneck dimensionality shows a clear frontier: larger $d _ { z }$ helps reconstruction, while smaller $d _ { z }$ helps generation. We thus use multiple $d _ { z } = [ 1 2 8 , 2 5 6 , 5 1 2 ]$ for our final model, so users can choose the trade-off that best suits their setting. For CFG distillation, 0.5 gives the best trade-off. For the r noise level sampling parameter, $\alpha = 1 . 5$ is the strongest. However, we notice frequent loss spikes during training with such a low α. We note that when $\alpha  \infty \mathrm { o u r }$ framework becomes equivalent to rectified flow (stable), while $\alpha = 0$ corresponds to consistency training (generally unstable). We thus choose to use $\alpha = 2 . 5$ in our final model, which results in smooth training. Finally, the non-causal variant performs better, showing that the causal constraint has a detrimental effect on both metrics.

Table 1: Architecture ablations for: a) the autoencoder, evaluated using r/gFAD; b) the phase estimator using rFAD.
<table><tr><td colspan="3">a) Autoencoder</td></tr><tr><td>Formulation</td><td>rFAD</td><td>gFAD</td></tr><tr><td>CMcSTFT</td><td>2.17</td><td>4.81</td></tr><tr><td> $v { \mathrm { - } } \mathbf { M } \mathbf { F } _ { \mathrm { m S T F T } } ^ { \sigma { \mathrm { C F G } } }$ </td><td>0.83</td><td>3.63</td></tr><tr><td> $\boldsymbol { x } { \mathrm { - } } \mathbf { M } \mathbf { F } _ { \mathrm { m S T F T } } ^ { \sigma { \mathrm { C F G } } }$  Bottleneck</td><td>0.82</td><td>3.61</td></tr><tr><td>tanh layernorm sphere + bounded noise  $+ \mathrm { R E P A }$   $+ n \sim \mathcal { U } _ { 0 , 1 }$  + REPA  $\underline { { d _ { z } } }$  64</td><td>0.82 0.76 0.77 0.73 0.79 0.99 1.29 0.90</td><td>5.14 4.63 4.57 3.95 3.45 3.21 2.58</td></tr><tr><td>256 512 CFG weight 0.0</td><td>0.79 0.61 0.82</td><td>2.60 3.45 4.64 3.61</td></tr><tr><td>0.5 1.0 α</td><td>0.79 0.86</td><td>3.45 3.45</td></tr><tr><td>1.5</td><td>0.76</td><td>3.34</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>2.5</td><td>0.79</td><td>3.45</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>3.5</td><td>0.84</td><td>3.51</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Causality</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Causal</td><td>0.79</td><td>3.45</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Non-causal</td><td>0.65</td><td>3.25</td></tr></table>

<table><tr><td colspan="2">b) Phase Estimator</td></tr><tr><td>D setup</td><td>rFAD</td></tr><tr><td> $D _ { \mathrm { p h } }$   $D _ { \mathrm { a u x } }$  both  $\underline { { \boldsymbol { \gamma } } }$ </td><td>0.41 0.31 0.25</td></tr><tr><td> $\gamma = 0 . 2$   $\gamma = 0 . 4$   $\gamma = 0 . 7$   $\underline { { \lambda _ { \mathrm { a u x } } } }$ </td><td>0.30 0.25 0.32</td></tr><tr><td>0.5 1.0 1.5 2.5 Causality</td><td>0.33 0.25 0.28 0.27</td></tr><tr><td>causal non-causal</td><td>0.25 0.27</td></tr></table>

6.2 Phase Estimator Ablations. Tab. 1 reports the rFAD for different ablations. The best results are obtained by using both main and auxiliary discriminators with powercompressed mel-magnitudes $( \gamma = 0 . 4 )$ , and an auxiliary discriminator weight of 1. Surprisingly, training a noncausal phase estimator does not improve our metrics.

6.3 Baseline Comparison. In Tab. 2, the final AudioCodex configuration is compared to M2L, CDC, SAO, and DAC under the same eval protocol (Sec. 5.2). Results show that AudioCodex matches or exceeds almost all baseline results, while being fully causal and enabling streaming. We note that none of the presented baselines is causal, and to the best of our knowledge, no general-purpose streamable autoencoder producing continuous latents currently exists. We also acknowledge that AudioCodex supports multi-step decoding, although we do not observe meaningful improvements on our metrics when using multiple steps. 2-step results are included in our supplementary page.

Table 2: Comparison to baselines. Metrics are defined in 5.3. For all metrics, lower is better.
<table><tr><td>Model rFAD rFADc gFAD STFT MEL</td></tr><tr><td>Codec Reference</td></tr><tr><td>DAC [9] 0.601 0.044 N/A 0.0940.057 Baselines</td></tr><tr><td>M2L [2] 0.943 0.233 5.241 0.132 0.090 CDC [17] 0.304 0.088 5.065 0.114 0.079</td></tr><tr><td>SAO [10] 1.151 0.155 9.518 0.1060.068</td></tr><tr><td>AudioCodex  $d _ { z } = 1 2 8$  0.773 0.098 3.696 0.1060.071  $d _ { z } = 2 5 6$  0.593 0.088 4.168 0.098 0.065</td></tr></table>

Table 3: Inference latency and real-time factor (RTF) measured on a NVIDIA A100 GPU with 48 kHz stereo audio.
<table><tr><td>Component</td><td>Time (ms)</td><td>RTF</td></tr><tr><td colspan="3">Single latent streaming</td></tr><tr><td>Encoding</td><td>6.80</td><td>12.56x</td></tr><tr><td>Combined decoding</td><td>41.38</td><td>2.06x</td></tr><tr><td colspan="3">10s sample</td></tr><tr><td>Encoding</td><td>18.91</td><td>528.8x</td></tr><tr><td>Combined decoding</td><td>87.64</td><td>114.1x</td></tr></table>

6.4 Inference speed. In Tab. 3, we evaluate the inference speed of AudioCodex on 48 kHz stereo audio using torch.compile on a single A100 GPU (batch size 1), with timings averaged over 128 runs. At the target latent rate, which is about 10 Hz for 44.1 kHz audio and 12 Hz for 48 kHz audio (roughly 0.1 s and 0.085 s per frame, respectively), the full causal pipeline operates in real time, with combined decoding (upsampler, generator, and phase estimator) achieving about 2× faster-than-real-time performance per latent step. When processing full 10-second sequences (non-streaming mode), efficiency improves due to reduced overhead, with the full decoding pipeline exceeding 100× real-time.

## 7. CONCLUSION

We presented AudioCodex, a fully causal continuous autoencoder for live audio. It compresses stereo audio to ∼ 10 Hz latents and reconstructs audio with a one-step magnitude decoder plus a separate causal phase estimator. In our ablations and baseline comparisons, AudioCodex shows a strong practical balance between reconstruction quality and downstream generative usability, while keeping both encoding and decoding streamable. This study also has clear limitations. Non-causal decoding can still achieve better quality than causal decoding. Our gFAD proxy uses short (∼ 1.5 s) samples, so it does not fully measure long-range structure. Future work should improve causal decoding quality and extend downstream evaluation to longer horizons.

## 8. ACKNOWLEDGEMENTS

This work is supported by the EPSRC UKRI Centre for Doctoral Training in Artificial Intelligence and Music (EP/S022694/1) and Sony Computer Science Laboratories Paris. Fazekas was supported by the Leverhulme Trust and the Royal Academy of Engineering under the RAEng / Leverhulme Trust Research Fellowships scheme.

## 9. REFERENCES

[1] Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, “Long-form music generation with latent diffusion,” in International Societyfor Music Information Retrieval Conference, 2024.

[2] M. Pasini, S. Lattner, and G. Fazekas, “Music2latent: Consistency autoencoders for latent audio compression,” in International Society for Music Information Retrieval Conference, 2024.

[3] J. Nistal, M. Pasini, C. Aouameur, M. Grachten, and S. Lattner, “Diff-a-riff: Musical accompaniment cocreation via latent diffusion models,” in International Society for Music Information Retrieval Conference, 2024, pp. 272–280.

[4] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive languageaudio pretraining with feature fusion and keywordto-caption augmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2023, pp. 1–5. [Online]. Available: https: //doi.org/10.1109/icassp49357.2023.10095969

[5] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Xiao, C. Lin, A. Ragni, E. Benetos, N. Gyenge, R. B. Dannenberg, R. Liu, W. Chen, G. Xia, Y. Shi, W. Huang, Z. Wang, Y. Guo, and J. Fu, “Mert: Acoustic music understanding model with large-scale self-supervised training.” in International Conference on Learning Representations, 2024, also available as arXiv:2306.00107. [Online]. Available: https://openreview.net/forum?id=w3YZ9MSlBu

[6] M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon, and A. F. Ehmann, “Supervised and unsupervised learning of audio representations for music understanding,” in International Society for Music Information Retrieval Conference, 2022.

[7] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2022. [Online]. Available: https: //doi.org/10.1109/taslp.2021.3129994

[8] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression.” Trans. Mach. Learn. Res., vol. 2023, 2023. [Online]. Available: https://openreview.net/forum?id=ivCd8z8zR2

[9] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved rvqgan,” in Advances in Neural Information Processing Systems. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023, pp. 27 980–27 993. [Online]. Available: https: //doi.org/10.52202/075280-1214

[10] Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, “Stable audio open,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2025, pp. 1–5. [Online]. Available: https: //doi.org/10.1109/icassp49660.2025.10888461

[11] P. Hansen-Estruch, D. Yan, C.-Y. Chuang, O. Zohar, J. Wang, T. Hou, T. Xu, S. Vishwanath, P. Vajda, and X. Chen, “Learnings from scaling visual tokenizers for reconstruction and generation.” in International Conference on Machine Learning, 2025. [Online]. Available: https://proceedings.mlr.press/v267/ hansen-estruch25a.html

[12] J. Casebeer, G. Zhu, Z. Wang, and N. J. Bryan, “A generative-first neural audio autoencoder,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2026, pp. 15 107– 15 111. [Online]. Available: https://doi.org/10.1109/ icassp55912.2026.11463285

[13] B. Zheng, N. Ma, S. Tong, and S. Xie, “Diffusion transformers with representation autoencoders,” in International Conference on Learning Representations, 2026, also available as arXiv:2510.11690. [Online]. Available: https://openreview.net/forum?id=0u1LigJaab

[14] S. Tong, B. Zheng, Z. Wang, B. Tang, N. Ma, E. Brown, J. Yang, R. Fergus, Y. LeCun, and S. Xie, “Scaling text-to-image diffusion transformers with representation autoencoders,” arXiv preprint, 2026.

[15] M. R. Bjare, G. Cantisani, M. Pasini, S. Lattner, and G. Widmer, “Perceptually aligning representations of music via noise-augmented autoencoders,” 2025. [Online]. Available: https://arxiv.org/abs/2511.05350

[16] M. Pasini, S. Lattner, and G. Fazekas, “Music2latent2: Audio compression with summary embeddings and autoregressive decoding,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2025, pp. 1–5. [Online]. Available: https: //doi.org/10.1109/icassp49660.2025.10890049

[17] ——, “Codicodec: Unifying continuous and discrete compressed representations of audio,” in International Society for Music Information Retrieval Conference, 2025.

[18] P. Cao, Y. Ma, F. Zhou, X. Yin, Q. Song, and L. Yang, “A tilted seesaw: Revisiting autoencoder trade-off for controllable diffusion,” Open MIND, 2026, also available as arXiv:2601.21633. [Online]. Available: https://doi.org/10.48550/arxiv.2601.21633

[19] P. Wang, S. Lu, Y. Tang, S. Yan, W. Xia, and Y. Xiong, “A full-duplex speech dialogue scheme based on large language model,” in Advances in Neural Information Processing Systems. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024, pp. 13 372–13 403. [Online]. Available: https: //doi.org/10.52202/079017-0427

[20] A. Défossez, L. Mazare, M. Orsini, A. Mourachko, P. Music, N. Zeghidour, J. Copet, and G. Synnaeve, “Moshi: A speech-text foundation model for real-time dialogue,” in International Conference on Learning Representations, 2025.

[21] Y. Wu, M. Wang, H. Lei, S. Brade, L. Blanchard, S.-L. Wu, A. C. Courville, and C.-Z. A. Huang, “Streaming generation for music accompaniment,” 2025. [Online]. Available: https://arxiv.org/abs/2510.22105

[22] T. Karchkhadze and S. Dubnov, “Towards real-time human-ai musical co-performance: Accompaniment generation with latent diffusion models and max/msp,” 2026. [Online]. Available: https://arxiv.org/abs/2604. 07612

[23] A. Caillon and P. Esling, “RAVE: A variational autoencoder for fast and high-quality neural audio synthesis,” 2021. [Online]. Available: https://arxiv.org/ abs/2111.05011

[24] N. Tokui and T. Baker, “Latent granular resynthesis using neural audio codecs,” 2025. [Online]. Available: https://arxiv.org/abs/2507.19202

[25] F. Caspe, J. Shier, M. Sandler, C. Saitis, and A. McPherson, “Designing Neural Synthesizers for Low-Latency Interaction,” Journal ofthe Audio Engineering Society, 2025.

[26] L. D. Libera, C. Subakan, and M. Ravanelli, “Focalcodec-stream: Streaming low-bitrate speech coding via causal distillation,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2026, pp. 17 002–17 006. [Online]. Available: https://doi.org/10.1109/icassp55912.2026.11464843

[27] X.-H. Jiang, Y. Ai, R.-C. Zheng, and Z.-H. Ling, “A streamable neural audio codec with residual scalarvector quantization for real-time communication,” IEEE Signal Processing Letters, vol. 32, pp. 1645–1649, 2025. [Online]. Available: https://doi.org/10.1109/lsp. 2025.3560172

[28] D. P. Kingma and M. Welling, “Auto-encoding variational bayes.” in International Conference on Learning Representations, 2014. [Online]. Available: http://arxiv.org/abs/1312.6114

[29] I. J. Goodfellow, J. Pouget-Abadie et al., “Generative adversarial nets,” in Advances in Neural Information Processing Systems, December 2014.

[30] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics.” in International Conference on Machine Learning, 2015, pp. 2256– 2265. [Online]. Available: http://proceedings.mlr.press/ v37/sohl-dickstein15.html

[31] J. Ho, A. Jain et al., “Denoising Diffusion Probabilistic Models,” in Advances in Neural Information Processing Systems, 2020.

[32] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, 2021.

[33] Z. Geng, M. Deng, X. Bai, Z. Kolter, and K. He, “Mean flows for one-step generative modeling.” in Advances in Neural Information Processing Systems, 2025, also available as arXiv:2505.13447.

[34] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 32 211–32 252.

[35] M. Pasini, J. nistal, S. Lattner, and G. Fazekas, “Continuous autoregressive models with noise augmentation avoid error accumulation,” in Advances in Neural Information Processing Systems, 2024, also available as arXiv:2411.18447. [Online]. Available: https://openreview.net/forum?id=dZNajz11Q8

[36] Y. Chen, R. Girdhar, X. Wang, S. S. Rambhatla, and I. Misra, “Diffusion autoencoders are scalable image tokenizers,” 2025. [Online]. Available: https: //arxiv.org/abs/2501.18593

[37] K. Yue, M. Jia, J. Hou, and T. Goldstein, “Image generation with a sphere encoder,” Open MIND, 2026, also available as arXiv:2602.15030. [Online]. Available: https://doi.org/10.48550/arxiv.2602.15030

[38] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think.” in International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=DJSZGGZYVi

[39] H.-J. Chang, S. Bhati, J. Glass, and A. H. Liu, “Usad: Universal speech and audio representation via distillation,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2025, pp. 1–8. [Online]. Available: https: //doi.org/10.1109/asru65441.2025.11434802

[40] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning.” in Advances in Neural Information Processing Systems, 2017, pp. 6306–6315. [Online]. Available: https://proceedings.neurips.cc/paper/2017/hash/ 7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html

[41] J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Defossez, “Simple and controllable music generation,” in Advances in Neural Information Processing Systems. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023, pp. 47 704–47 720. [Online]. Available: https: //doi.org/10.52202/075280-2066

[42] A. Agostinelli, T. I. Denk et al., “MusicLM: Generating Music From Text,” January 2023, arXiv:2301.11325 [cs, eess]. [Online]. Available: https://arxiv.org/abs/ 2301.11325

[43] K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn, “Diffusion autoencoders: Toward a meaningful and decodable representation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2022, pp. 10 609–10 619. [Online]. Available: https://doi.org/10.1109/cvpr52688. 2022.01036

[44] C. Park, J. C. Lee, and J. H. Ko, “Singlestep diffusion for image compression at ultra-low bitrates,” in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2026, pp. 6391–6400. [Online]. Available: https: //doi.org/10.1109/wacv61042.2026.00618

[45] Z. Evans, C. Carr, J. Taylor, S. H. Hawley, and J. Pons, “Fast timing-conditioned latent audio diffusion.” in International Conference on Machine Learning, 2024, pp. 12 652–12 665. [Online]. Available: https://proceedings.mlr.press/v235/evans24a.html

[46] T. Xiong, J. H. Liew, Z. Huang, J. Feng, and X. Liu, “Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation,” in IEEE/CVF International Conference on Computer Vision. IEEE, 2025, pp. 18 770–18 780. [Online]. Available: https://doi.org/10.1109/iccv51701.2025. 01744

[47] Y.-X. Lu, Y. Ai, and Z.-H. Ling, “Mp-senet: A speech enhancement model with parallel denoising of magnitude and phase spectra,” in Interspeech. ISCA, 2023, pp. 3834–3838. [Online]. Available: https://doi.org/10.21437/interspeech.2023-1441

[48] C. Donahue, J. J. McAuley, and M. S. Puckette, “Adversarial audio synthesis.” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=ByMVTsR5KQ

[49] K. Kumar, R. Kumar et al., “MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis,” in Advances in Neural Information Processing Systems, December 2019, place: Vancouver, BC, Canada.

[50] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” Advances in Neural Information Processing Systems, 2020.

[51] R. Yamamoto, E. Song, and J.-M. Kim, “Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram.” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2020, pp. 6199–6203, arXiv:1910.11480 [cs, eess] Also available as arXiv:1910.11480. [Online]. Available: https://doi.org/10.1109/icassp40776.2020.9053795

[52] Z. Geng, Y. Lu, Z. Wu, E. Shechtman, J. Z. Kolter, and K. He, “Improved mean flows: On the challenges of fastforward generative models,” 2025. [Online]. Available: https://arxiv.org/abs/2512.02012

[53] N. Huang, A. Gokaslan, V. Kuleshov, and J. Tompkin, “The gan is dead; long live the gan! a modern gan baseline.” in Advances in Neural Information Processing Systems, 2024.

[54] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow.” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=XVjTT1nw5z

[55] A. Jolicoeur-Martineau, “The relativistic discriminator: a key element missing from standard gan.” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/ forum?id=S1erHoR5t7

[56] B. Zhang and R. Sennrich, “Root mean square layer normalization.” in Advances in Neural Information Processing Systems, 2019, pp. 12 360–12 371. [Online]. Available: https://proceedings.neurips.cc/paper/2019/hash/ 1e8a19426224ca89e83cef47f1e7f53b-Abstract.html

[57] T. Kynkäänniemi, M. Aittala, T. Karras, S. Laine, T. Aila, and J. Lehtinen, “Applying guidance in a limited interval improves sample and distribution quality in diffusion models.” in Advances in Neural Information Processing Systems, 2024.

[58] S. gil Lee, W. Ping, B. Ginsburg, B. Catanzaro, and S. Yoon, “Bigvgan: A universal neural vocoder with large-scale training.” in International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=iTtGCMDEzS\_

[59] D. Galvez, R. Hileman, R. Mosquera, J. Ciro, K. Bollacker, P. Mattson, and D. Kanter, “Unsupervised people’s speech (the million hour audio dataset),” 2023. [Online]. Available: https://huggingface.co/datasets/ MLCommons/peoples\_speech

[60] Freesound Team, Music Technology Group, Universitat Pompeu Fabra, “About freesound,” 2026, accessed: 2026-04-24. [Online]. Available: https://freesound.org/ help/about/

[61] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson, “Fma: A dataset for music analysis.” in

International Societyfor Music Information Retrieval Conference, 2017, pp. 316–323, arXiv:1612.01840 [cs.SD] Also available as arXiv:1612.01840. [Online]. Available: https://ismir2017.smcnus.org/wp-content/ uploads/2017/10/75\_Paper.pdf

[62] Z. Rafii, A. Liutkus, F.-R. St"oter, S. I. Mimilakis, and R. Bittner, “MUSDB18-HQ - an uncompressed version of MUSDB18,” August 2019. [Online]. Available: https://doi.org/10.5281/zenodo.3338373

[63] E. Bakhturina, V. Lavrukhin, B. Ginsburg, and Y. Zhang, “Hi-fi multi-speaker english tts dataset.” in Interspeech, 2021, pp. 2776–2780, arXiv:2104.01497 [eess.AS] Also available as arXiv:2104.01497. [Online]. Available: https://doi.org/10.21437/interspeech.2021-1599

[64] SoundBible.com, “SoundBible: Free sound effects,” 2026, accessed: 2026-04-24. [Online]. Available: https://soundbible.com/

[65] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Fréchet audio distance: A reference-free metric for evaluating music enhancement algorithms,” in Interspeech. ISCA, 2019, pp. 2350–2354. [Online]. Available: https://doi.org/10.21437/interspeech.2019-2219

[66] S. Hershey, S. Chaudhuri, D. P. W. Ellis, J. F. Gemmeke, A. Jansen, R. C. Moore, M. Plakal, D. Platt, R. A. Saurous, B. Seybold, M. Slaney, R. J. Weiss, and K. Wilson, “Cnn architectures for large-scale audio classification,” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2017, pp. 131–135. [Online]. Available: https://doi.org/10.1109/icassp.2017.7952132

[67] J. L. Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “Sdr – half-baked or well done?” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2019, pp. 626–630. [Online]. Available: https://doi.org/10.1109/icassp.2019.8683855

[68] P. Ramachandran, B. Zoph, and Q. V. Le, “Searching for activation functions.” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=Hkuq2EkPf

[69] K. Liang, L. Chen, B. Liu, and qiang liu, “Cautious optimizers: Improving training with one line of code,” in International Conference on Learning Representations, 2026, also available as arXiv:2411.16085. [Online]. Available: https://openreview.net/forum?id=zBPZeRjfgu

[70] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization.” in International Conference on Learning Representations, 2015. [Online]. Available: http://arxiv.org/abs/1412.6980

[71] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization.” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[72] T. Li and K. He, “Back to basics: Let denoising generative models denoise,” 2025. [Online]. Available: https://arxiv.org/abs/2511.13720

[73] M. Pasini and J. Schlüter, “Musika! fast infinite waveform music generation,” in International Society for Music Information Retrieval Conference, 2022, pp. 568– 575.