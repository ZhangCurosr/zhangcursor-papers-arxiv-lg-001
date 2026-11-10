# BARS: BEAT-ADAPTIVE RAP SYNTHESIS VIA FLOW MATCHING

Taemin Cho

Hector Martel

Enyan Koh

BandLab Technologies, Singapore

taemin.cho@bandlab.com

## ABSTRACT

We present BARS (Beat-Adaptive Rap Synthesis), a nonautoregressive rapping voice generation model based on Flow Matching, conditioned on a beat track (instrumental backing), lyrics, and a short reference voice for zero-shot timbre control. Unlike prior autoregressive rap synthesis, where rap pace and output duration are strictly determined by the beat, BARS enables independent control over the rap tempo, allowing flexible-length generation without explicit duration prediction or phoneme-level alignment. We encode lyrics using a pretrained ByT5 encoder with a lightweight Transformer adapter trained jointly with the model, without relying on speech-pretrained text encoders, enabling natural handling of non-standard orthography (e.g., slang, abbreviations) and numerals common in rap lyrics. To address alignment instabilities, such as token skipping or repetition, under strong beat conditioning, we propose Asymmetric Scaled RoPE for cross-attention, which introduces a positional structure to facilitate sequential lyric-audio alignment learning. Furthermore, the model supports partial re-generation via span masking, enabling lyric-level editing while preserving the surrounding vocal style. Experiments demonstrate that the proposed model achieves high lyric fidelity, natural rap flow that is rhythmically coherent with the beat, and robust tempo control across a wide range of rapping styles and speeds.

## 1. INTRODUCTION

Rap music occupies a unique intersection of rhythm, language, and vocal expression, where the vocal delivery is governed by a rigid rhythmic contract with the underlying beat. Unlike melodic singing, rapping requires syllables to fall precisely within specific metrical windows while preserving the semantic integrity of the lyrics [1]. This dual constraint of temporal synchronization and semantic fidelity makes rapping voice generation significantly more challenging than conventional Text-to-Speech (TTS) or singing voice synthesis.

Existing Singing Voice Synthesis (SVS) systems [2–4] typically require predefined pitch and duration inputs, which are fundamentally incompatible with the nature of rap. Indeed, rap delivery is largely pitch-agnostic and characterized by irregular rhythmic variations. The essence of rap performance, commonly referred to as flow [1], emerges from dynamic stress patterns, syncopation, and deliberate timing deviations that cannot be captured by fixed pitch and duration templates. Recent work on beatconditioned rap generation, notably Freestyler [5], has made meaningful progress in this direction. However, its autoregressive nature results in slow sequential decoding, and both the rapping pace and output duration are solely determined by the beat track, leaving the model with no independent control over vocal delivery.

In this work, we propose BARS (Beat-Adaptive Rap Synthesis) <sup>1</sup> , a non-autoregressive rapping voice generation model built on Flow Matching [6]. The model is conditioned on a beat track, lyrics, and a short reference voice, and generates the corresponding rap vocal track, which can subsequently be mixed with the input beat track to produce the final output. It is designed as a creative tool to help rap artists prototype and refine their vocal delivery. Its nonautoregressive design enables independent control over the tempo of the generated vocals, allowing artists to explore diverse rhythmic interpretations of the same lyrics over a given beat. In addition, training with span masking allows partial re-generation, making it possible to revise specific phrases while preserving the vocal style and rhythmic consistency of the surrounding performance. The model further supports zero-shot timbre control, enabling synthesis in the voice of an unseen speaker using only a short reference voice sample.

For lyric encoding, we leverage a pretrained ByT5 encoder [7], which provides rich byte-level linguistic representations learned from large-scale text data. To map these embeddings for cross-modal alignment with audio latents, we introduce a lightweight Transformer-based adapter. We further observe that strong beat conditioning can disrupt positional tracking over the lyric sequence during crossattention, leading to skipped or repeated lyric tokens. To mitigate this, we propose an Asymmetric Scaled RoPE (A-RoPE) mechanism that applies standard RoPE to the audio query side while applying length-proportionally scaled RoPE to the lyric key side, encouraging more consistent sequential coverage of the lyric tokens.

To the best of our knowledge, this is the first nonautoregressive Flow Matching-based framework for rapping voice generation that achieves flexible-length output without relying on explicit duration prediction, phonemelevel alignment, or speech-specific encoders.

## 2. RELATED WORK

## 2.1 Rap Generation

Previous works on rap generation have largely focused on the linguistic dimensions of the genre. Early systems such as DopeLearning [8] and GhostWriter [9] addressed lyric composition through information retrieval using RankSVM and LSTM-based sequence modeling respectively, prioritizing rhyme quality and stylistic consistency. Rapformer [10] further extended this direction with a Transformer-based denoising autoencoder capable of transferring non-rap text into rap lyrics. DeepRapper [11] marked a step toward rhythmic awareness by explicitly modeling rhyme and beat alignment through a reverse-order language model with inserted beat tokens. Despite this progress, these systems operate entirely in the text domain and produce no vocal audio, leaving the acoustic realization of rap performance largely unaddressed.

The shift toward audio-based rap generation is more recent. Freestyler [5] represents the first system to synthesize rapping vocals directly from lyrics and accompaniment, employing a three-stage hierarchical framework to achieve stylistic and rhythmic alignment with the backing track. However, its autoregressive architecture introduces certain limitations. In particular, the model operates in a unidirectional manner, which restricts its ability to consider the global structure of the accompaniment when planning vocal delivery. As the model is trained to synchronize closely with local, frame-level acoustic features, it has limited capacity to adjust delivery speed based on the overall duration of the beat. Consequently, the generated vocals tend to follow the input track’s rhythm, with limited flexibility in pacing control across varying track lengths. These limitations result in early termination when the beat is too short and overly extended outputs when it is excessively long. Although Freestyler incorporates a random masking strategy during training to alleviate these issues, explicit control over pacing independent of the beat track remains limited. Additionally, as model weights are not publicly available, direct quantitative comparison is not feasible.

## 2.2 Non-Autoregressive Speech Synthesis

Non-autoregressive approaches to TTS have advanced rapidly through diffusion-based frameworks. E3 TTS [12] explores end-to-end diffusion-based TTS without explicit alignment supervision, but its U-Net backbone constrains generation to fixed-length outputs. NaturalSpeech [13, 14] and Voicebox [15] overcome this limitation by adopting Transformer architectures that support variable-length generation. However, they depend on phoneme-level alignment and separate duration prediction modules, which introduce additional training complexity.

E2 TTS [16] takes a further step toward simplicity by eliminating all explicit alignment constraints. By framing TTS as an audio infilling task and using filler tokens to reconcile text and audio lengths, E2 TTS learns alignment implicitly during training. DiTTo-TTS [17] similarly avoids explicit duration supervision via cross-attention between text and audio representations. However, it reports that ByT5, a text-only pretrained encoder, substantially underperforms speech-pretrained encoders such as SpeechT5 [18] when codec-level text-audio fine-tuning is absent. This finding suggests that at least one side of the text-audio interface must encode cross-modal information to effectively bridge the modality gap.

## 3. METHOD

Our approach builds on Flow Matching [6], ByT5 [7], RoPE [19], and an audio VAE [20]; our contributions lie specifically in the cross-modal adapter (Section 3.1) and the A-RoPE mechanism (Section 3.4).

## 3.1 Input Representations

The model takes three inputs: a beat track, lyrics, and a 3- second reference voice. The beat track and reference voice, along with the ground truth rap vocal during training, are encoded to compact latent sequences using an audio VAE trained following the architecture of Stable Audio 2.0 [20] on proprietary music data comprising diverse music genres, sound effects, and vocal recordings.

Lyrics are encoded using a pretrained ByT5-base encoder <sup>2</sup> . Rather than fine-tuning the audio encoder or relying on a speech-pretrained text encoder, we introduce a lightweight Transformer encoder [21] as a cross-modal adapter on the text side. This adapter is trained jointly with the Flow Matching network and learns to map ByT5 embeddings into a representation space compatible with audio latents for cross-attention. This design preserves the byte-level coverage of ByT5, which can naturally handle numbers, symbols, and non-standard orthography common in rap lyrics, while avoiding the limitations of speechpretrained encoders such as SpeechT5, which relies on a fixed phonetic vocabulary that treats such inputs as out-ofvocabulary tokens.

## 3.2 Model Input Construction

Inspired by [17], we incorporate span masking into training to enable lyric infilling. While DiTTo-TTS [17] leverages this objective to naturally acquire zero-shot voice cloning capability, our setting differs: since the generated vocal must cover the full lyric sequence from the start of the beat, we instead provide speaker identity via a 3-second reference voice prepended to the input sequence, enabling zero-shot timbre control without requiring a paired transcript.

Let $z _ { 1 } \in \mathbb { R } ^ { D \times N }$ denote the ground truth rap vocal latent sequence, where D is the latent dimension and N is the sequence length. A binary mask $m \in \{ 0 , 1 \} ^ { N }$ is constructed by randomly selecting a contiguous span. Following [17], the mask either covers the entire sequence with probability 0.1, or masks a random contiguous segment spanning

![](images/4f2ed0ec1d7c6d4f8e92670a531fa6847b2fa162e1e3b420915fa258466330d0.jpg)  
Figure 1: Input construction: the rap vocal, beat track, and reference voice are encoded via a shared VAE encoder. A span mask is applied to the vocal latent, and masked regions are replaced with the Flow Matching interpolant $z _ { t } .$ The beat latent is concatenated along the channel dimension and projected to form $x _ { t } ,$ which is prepended with $z _ { \mathrm { r e f } }$ and timestep embedding $t _ { \mathrm { e m b } }$

70% to 100% of the sequence length. The masked input is formed as:

$$
\tilde { z } _ { t } = m \odot z _ { t } + ( 1 - m ) \odot z _ { 1 }\tag{1}
$$

Here, ⊙ denotes element-wise multiplication, $z _ { t } = ( 1 -$ $t ) \cdot z _ { 0 } + t \cdot z _ { 1 }$ is the Flow Matching interpolant applied to masked positions, where $z _ { 0 } \sim \mathcal { N } ( 0 , \mathbf { I } )$ is Gaussian noise, and $t \in [ 0 , 1 ]$ is the timestep sampled uniformly during training. The unmasked positions (i.e., where $m \ : = \ : 0 )$ retain the ground truth values $z _ { 1 } .$ , conditioning the model on the surrounding context. The training loss is computed only over the masked regions:

$$
\mathcal { L } = \mathbb { E } _ { t , z _ { 0 } } \left\| m \odot ( u _ { t } - v _ { \theta } ( \tilde { z } _ { t } , t , c ) ) \right\| ^ { 2 }\tag{2}
$$

where $u _ { t } = z _ { 1 } - z _ { 0 }$ is the Flow Matching target, and c represents the conditioning inputs.

The beat track latent is projected to a lower-dimensional representation $z _ { \mathrm { b e a t } } ~ \in ~ \bar { \mathbb { R } } ^ { d _ { \mathrm { b e a t } } \times N }$ and concatenated with $\tilde { z } _ { t }$ along the channel dimension, providing a frame-level rhythmic scaffold. The concatenated sequence is then projected to the model dimension $d _ { \mathrm { m o d e l } }$ to produce the audio input tokens $x _ { t }$ . As mentioned above, the reference voice latent is projected to $d _ { \mathrm { m o d e l } }$ and prepended to the audio input tokens, along with a sinusoidal embedding of the timestep t, following Stable Audio 2.0 [20]. The full input sequence to the Transformer is thus:

$$
{ \mathrm { i n p u t } } = [ t _ { \mathrm { e m b } } , z _ { \mathrm { r e f } } , x _ { t } ]\tag{3}
$$

This process is illustrated in Figure 1.

## 3.3 Model Architecture

The input sequence is processed by a stack of Transformer blocks, as illustrated in Figure 2. Each block consists of a bidirectional self-attention layer, a cross-attention layer, and a feed-forward network. In the self-attention layer, Rotary Position Embeddings (RoPE) [19] are applied to both the query and key, with position indices reset to 0 at the start of each segment (i.e., the prepended conditioning tokens and the audio latent tokens are each indexed independently starting from 0). This position-decoupled scheme allows the timestep and reference to interact with the audio latents as global context without introducing positional bias.

![](images/478620e35645d51c98e892a4b45eaf39a0d7511452f9c3373fe07eb8e01344d5.jpg)  
Figure 2: Architecture of a single Transformer block: RoPE is applied in self-attention with position indices reset per segment. Prior to cross-attention, conditioning tokens are removed and only audio latent tokens serve as queries, with standard RoPE on the audio side and length-scaled RoPE on the lyric key side. The conditioning tokens are reintroduced before the feed-forward network.

Lyric representations are injected into the Transformer blocks via cross-attention layers, enabling alignment between audio and text without explicit duration prediction or phoneme-level supervision. Prior to cross-attention, the prepended conditioning tokens $( t _ { \mathrm { e m b } } ^ { \prime }$ and $z _ { \mathrm { r e f } } ^ { \prime } )$ are temporarily removed from the sequence and only the audio latent tokens $ { \boldsymbol { { x } } } _ { t } ^ { \prime }$ are used as queries, while the lyric representations serve as keys and values. Since the timestep and reference function as global conditioning signals rather than content tokens that align with lyrics, they are excluded from the cross-attention query. This is possible because their information is already propagated to the audio latent tokens through the preceding self-attention. This design ensures a positionally coherent application of A-RoPE, described in Section 3.4. Finally, the conditioning tokens are reintroduced before the feed-forward network.

## 3.4 Asymmetric Scaled RoPE for Cross-Attention

As previously noted, without positional guidance in crossattention, the model tends to prioritize rhythmic alignment with the beat at the expense of accurate lyric reproduction. We observe that this behavior correlates with the strength of the beat conditioning: as the beat input is scaled down or removed, lyric accuracy improves, albeit at the cost of rhythmic alignment. The issue becomes more pronounced when the lyric length deviates substantially from what the beat can naturally accommodate. To address this, we introduce an A-RoPE mechanism in the cross-attention layers.

![](images/beb936a4f6364c64ecdc00e41fd2b26826c8537722de833776cf021eaed3df6d.jpg)  
(a) w/o A-RoPE

![](images/61cb3d4717d8a038fad7b2685d544fa58487f2453618cdb9697c0e5a240a3a42.jpg)  
(b) w/ A-RoPE  
Figure 3: Cross-attention maps (mid-layer). (a) Abrupt horizontal shifts indicate skipping, while the double-line pattern denotes repetition via redundant attention. (b) Attention follows a smooth diagonal trajectory.

In standard RoPE, the relative position between query position i and key position j is encoded through the rotation angle difference, which biases attention toward nearby positions. In our cross-attention, the query (audio latent) and key (lyric embedding) sequences differ in length: the audio sequence has length N while the lyric sequence has length $L .$ We apply standard RoPE to the audio query side, while scaling the RoPE position indices on the lyric key side by a factor of $S \ = \ N / L$ . Specifically, for a lyric key at position $j ,$ its RoPE rotation angle is computed using the scaled position $j \cdot S$ instead of $j .$ . Intuitively, this stretches the lyric position indices to span the temporal extent of the audio as if a lyric sheet were laid out along the track’s timeline, encouraging the model to learn a sequential correspondence between audio frames and lyric tokens: audio frame i is biased toward lyric token $j \approx i \cdot L / N$ This mechanism allows the model to deviate from strict diagonal alignment where appropriate (e.g., held syllables or rhythmic pauses) while discouraging the skip and repeat patterns that arise in the unconstrained setting. Figure 3 visualizes cross-attention maps from models with and without A-RoPE. Without it, the attention pattern exhibits frequent skipping and repetition, whereas the model with A-RoPE maintains a coherent diagonal trajectory without skipping or repetition.

Closely related to our approach, LARoPE [22] proposes a symmetric length-normalization of RoPE for TTS crossattention, where both query and key positions are divided by their respective sequence lengths. Our approach instead scales only the key (lyric) side by the length ratio $S ~ = ~ N / L$ , leaving the query (audio) side as standard RoPE, which avoids introducing additional hyperparameters and is motivated by the distinct challenge of beatconditioned generation, where rhythmic conditioning competes with lyric alignment.

## 4. EXPERIMENT

## 4.1 Dataset and Implementation Details

We train on approximately 1,500 hours of English rap music from both male and female artists, covering a diverse range of subgenres. Each training sample consists of a rap vocal segment with its corresponding lyrics and beat track, along with a 3-second reference clip from a different segment of the same rapper. We split the data into train/validation/test sets (90/5/5), ensuring no rapper overlap across splits. Our model (∼200M parameters) is trained for 500k steps on 8 NVIDIA H100 GPUs using AdamW (peak $\mathrm { L R } \ 1 \times 1 0 ^ { - 4 }$ , OneCycleLR with 2k warmup, batch size 64) with EMA (decay 0.9999). While the model architecture imposes no inherent length constraint, training samples are capped at 60 seconds for memory efficiency; we have not evaluated inference beyond this duration. At inference, the output length is set to that of the given beat track, and we integrate the flow ODE using the Euler method (25 steps) with classifier-free guidance (scale 3.0). On a single NVIDIA RTX 4090 GPU, our model achieves a real-time factor (RTF) of 0.027 (10 seconds of audio in 0.27 seconds), including VAE encoding/decoding and text encoding.

## 4.2 Evaluation Metrics

We evaluate generated rap vocals using the following metrics. Word Error Rate (WER) is computed by transcribing generated vocals with Whisper large-v3 [23] and comparing the resulting transcripts against the ground-truth lyrics. We further decompose WER into deletion, insertion, and substitution rates to distinguish lyric skipping, repetition, and mispronunciation. As rap vocals pose inherent challenges for ASR systems, WER should be interpreted relative to the GT transcription error (5.1% in Table 1) rather than as an absolute measure of generation quality. Speaker similarity is measured as the cosine similarity between speaker embeddings extracted from the reference and generated audio using WavLM-Base+ [24], averaging over non-overlapping 3-second windows for vocals longer than the reference. DNSMOS [25] is included as a supplementary automatic proxy for audio quality. Perceptual quality is primarily assessed through a subjective MOS study on a 5-point Likert scale, evaluating rhythmic flow (MOS-F), speaker similarity (MOS-S), and audio quality (MOS-Q), with details provided in Section 4.7.

## 4.3 Ablation Study

We compare the proposed model (ByT5 + Adapter + A-RoPE) against ablation variants to evaluate the contribution of each component. Results are summarized in Table 1. The proposed model achieves a WER of 5.1%, comparable to that obtained on the ground truth (humanrecorded) audio, which reflects the inherent transcription difficulty of rap vocals for ASR systems.

Removing A-RoPE increases WER only marginally to 6.0%. Notably, this ByT5 + Adapter configuration alone already outperforms every SpeechT5-based variant except the one combining both the adapter and A-RoPE (5.6%), suggesting that a lightweight adapter can substitute for costly speech pretraining as a means of achieving effective cross-modal grounding. Removing the adapter while retaining A-RoPE yields a WER of 9.7%, on par with SpeechT5 without any adaptation (10.0%). This suggests that A-RoPE’s positional prior can substantially compensate for the lack of a learned cross-modal representation, by directly encouraging sequential correspondence between lyric and audio tokens. Without both adapter and A-RoPE, the model produces significantly degraded lyric coverage, reaching a WER of 37.7%. These results collectively demonstrate that neither component alone is sufficient to match the full configuration.

<table><tr><td></td><td></td><td>Encoder Adapter A-RoPE</td><td>WER↓</td><td>Del↓ Ins↓</td><td>Sub↓</td></tr><tr><td>GT</td><td></td><td></td><td>5.1 %</td><td>0.7 % 0.6 %</td><td>3.8 %</td></tr><tr><td>ByT5</td><td>√ √</td><td>√ √</td><td>5.1 % 6.0 % 9.7 % 37.7 % 5.3 %</td><td>1.1 % 1.1 % 1.5 % 1.2 % 1.3 % 3.2 % 7.4 %</td><td>2.9 % 3.3 % 5.2 % 25.0 %</td></tr><tr><td>SpT5</td><td>√ √</td><td>V √</td><td>5.6 % 1.1 % 8.8 % 1.9 % 7.8 % 1.7 % 10.0 % 2.4 %</td><td>1.2 % 1.7 % 1.5 % 2.2 %</td><td>3.3 % 5.2 % 4.6 % 5.4 %</td></tr></table>

Table 1: Ablation study on encoder, adapter, and A-RoPE. WER and its decomposition into deletion (Del), insertion (Ins), and substitution (Sub) rates are reported. The proposed configuration is shown in bold. GT denotes ground truth recordings. SpT5: SpeechT5.

Together, the adapter and A-RoPE are complementary: the adapter provides cross-modal grounding while A-RoPE supplies the positional structure needed to maintain lyric coverage under strong beat conditioning. This complementarity is more clearly observed in the SpeechT5 results, where the modality gap is already considerably addressed through speech pretraining. Adding A-RoPE alone yields a larger improvement (10.0% → 7.8%) than adding the adapter alone (10.0% → 8.8%), suggesting that under sufficient cross-modal grounding, A-RoPE’s positional alignment prior has a stronger effect on lyric coverage. The full effect of A-RoPE is further examined on variable tempo settings in Section 4.5. When both components are combined, SpeechT5 nearly matches the proposed ByT5 configuration. Despite the similar WER, we retain ByT5 for its byte-level flexibility, which avoids the vocabulary limitations of SpeechT5 discussed in Section 3.1.

## 4.4 Training Dynamics

To quantify how effectively the model utilizes lyric content during training, we define a lyrics utilization score as $( \mathcal { L } _ { \mathrm { i r r } } - \mathcal { L } _ { \mathrm { g t } } ) / \mathcal { L } _ { \mathrm { g t } }$ , where ${ \mathcal { L } } _ { \mathrm { g t } }$ is the loss computed with ground-truth lyrics and $\mathcal { L } _ { \mathrm { i r r } }$ is the loss computed with irrelevant lyrics from a different sample in the same batch. This score can be roughly interpreted as the relative contribution of lyric content to the model’s loss.

As shown in Figure 4, the model trained with A-RoPE exhibits immediate and steady growth in lyrics utilization from the start of training, whereas the model without A-RoPE shows near-zero utilization for approximately 50k steps before rising sharply. This delayed activation coincides with a visible discontinuity in the training loss curve, suggesting that the model abruptly shifts its reliance toward lyric content once the cross-attention becomes effective. Although both models converge to similar training loss, the following section shows that this does not translate into equivalent generalization under tempo mismatch.

![](images/b015aff910bd206a55495434a71a9db17924571d8b36de64c117f64d7e0660ed.jpg)  
(a) Training loss

![](images/19961462c4cf2496fea52fbf94f7c8025857a2d1374eae8bc33198bed1ec9b25.jpg)  
(b) Lyrics utilization

Figure 4: Training curves comparing models with and without A-RoPE. (a) Training loss. (b) Lyrics utilization, measured as the relative loss increase when replacing ground-truth lyrics with irrelevant lyrics in cross-attention.
<table><tr><td rowspan="2">Speed (CPS)</td><td colspan="3">w/ A-RoPE</td><td colspan="2">w/o A-RoPE</td></tr><tr><td>WER↓</td><td>Del↓</td><td>Ins↓</td><td>WER↓ Del↓</td><td>Ins↓</td></tr><tr><td>-2σ</td><td>20.5 %</td><td>2.2 %</td><td>9.3 %</td><td>33.6% 5.7 %</td><td>14.6%</td></tr><tr><td rowspan="4">-1σ μ  $+ 1 \sigma$ </td><td>14.2 %</td><td>2.6 %</td><td>5.2 %</td><td>25.5 % 5.9 %</td><td>8.3 %</td></tr><tr><td>12.3 %</td><td>3.3 %</td><td>2.8 %</td><td>22.0 % 7.2 %</td><td>4.7 %</td></tr><tr><td>13.8 %</td><td>4.1 %</td><td>2.2 %</td><td>24.4 % 9.3 %</td><td>3.9 %</td></tr><tr><td>15.4 %</td><td>5.8 %</td><td>2.2 %</td><td>26.9 % 12.6 %</td><td>3.1 %</td></tr></table>

Table 2: Tempo robustness evaluation (WER/Del/Ins, %) for the proposed model and its ablation without A-RoPE. Lyrics and beat tracks are randomly mismatched across test samples. Speed bin labels denote the offset from the training mean µ, each spanning 1σ (µ=17.9, σ=2.9 chars/sec).

## 4.5 Tempo Robustness

To evaluate tempo robustness, we randomly mismatch lyrics and beat tracks from the test set, controlling the ratio of lyric length to beat duration. Specifically, we partition the mismatched pairs into five bins based on the characters-per-second (CPS) rate, using the training distribution (µ = 17.9, σ = 2.9 chars/sec) to define bin boundaries. Each bin spans 1σ and is centered at $\mu - 2 \sigma$ (slowest), $\mu - 1 \sigma , \mu , \mu + 1 \sigma ,$ , and $\mu + 2 \sigma$ (fastest).

As shown in Table 2, the proposed model consistently achieves lower WER across all bins, with the largest improvements observed in the extreme bins where lyrics are significantly faster or slower than the beat. We note that WER values in this experiment are higher than those reported in Table 1, as the lyrics are not written to match the given beat. In the matched setting, lyrics are composed to fit the rhythmic structure of the beat, making it easier for the model to distribute syllables naturally. In the mismatched setting, the model must adapt its pacing to accommodate lyrics of varying lengths relative to the beat, which inherently increases the likelihood of deletion and insertion errors. The error pattern is directional: deletions increase at higher CPS while insertions increase at lower CPS, consistent with the skip and repeat behaviors described in Section 3.4.

<table><tr><td>Model</td><td>Group</td><td>WavLM Sim ↑</td><td>DNSMOS OVR ↑</td></tr><tr><td>GT</td><td>一</td><td> $0 . 8 0 \pm 0 . 0 5$ </td><td> $3 . 3 5 \pm 0 . 2 1$ </td></tr><tr><td>w/ A-RoPE</td><td>match mismatch</td><td> $0 . 8 0 \pm 0 . 0 5$   $0 . 7 5 \pm 0 . 0 5$ </td><td> $3 . 4 9 \pm 0 . 2 0$   $3 . 4 6 \pm 0 . 2 1$ </td></tr><tr><td>w/o A-RoPE</td><td>match mismatch</td><td> $0 . 8 0 \pm 0 . 0 5$   $0 . 7 6 \pm 0 . 0 5$ </td><td> $3 . 4 6 \pm 0 . 2 0$   $3 . 4 2 \pm 0 . 2 2$ </td></tr></table>

Table 3: Speaker similarity (WavLM Sim) and audio quality (DNSMOS OVR) under matched and mismatched conditions (mean ± std).

## 4.6 Speaker Similarity and Audio Quality

Table 3 reports speaker similarity and audio quality under matched (original lyric-beat pairs) and mismatched (randomly reassigned pairs from Section 4.5) conditions. In the matched setting, both models achieve a speaker similarity of 0.80, matching the GT upper bound and demonstrating effective speaker identity preservation from a 3-second reference voice. Under mismatched conditions, similarity decreases slightly to 0.75–0.76, likely due to the increased difficulty of maintaining consistent vocal delivery when lyrics and beat are not originally paired. The difference between the two models remains within the standard deviation, indicating that A-RoPE does not affect timbre reproduction.

DNSMOS scores are consistent across all conditions, with both models producing audio quality comparable to or slightly above GT (3.35). The higher DNSMOS of generated vocals (3.42–3.49) can be attributed to the smoothing effect of the generation process, which tends to suppress background noise and vocal artifacts present in real recordings. Under mismatched conditions, the model with A-RoPE scores marginally higher than the variant without it (3.46 vs. 3.42), indicating that improved lyric alignment does not come at the cost of audio quality.

## 4.7 Subjective Evaluation

We conduct a listening study to evaluate rhythmic flow (MOS-F), speaker similarity (MOS-S), and audio quality (MOS-Q) on a 5-point Likert scale. Two model variants are evaluated: the proposed model and its ablation without A-RoPE. For each model, 15 clips are randomly sampled from the generated outputs of the mismatched test set (3 per CPS bin), selecting only samples for which both models achieve WER below 5%. Without this control, differences in lyric accuracy dominate listener judgments: listeners tend to penalize samples with audible mispronunciations or missing words, overshadowing the perceptual dimensions we aim to evaluate. For speaker similarity, listeners are provided with the 3-second reference voice that was used as input during generation. Additionally, 10 ground truth clips are randomly sampled from the matched test set, with the reference voice drawn from a different segment by the same rapper. All 40 clips are randomly shuffled and presented to 40 listeners in a blind evaluation without any indication of their source.

<table><tr><td>Model</td><td>MOS-F↑</td><td>MOS-S ↑</td><td>MOS-Q↑</td></tr><tr><td>GT</td><td>3.81</td><td>3.95</td><td>3.73</td></tr><tr><td>w/ A-RoPE</td><td>3.51</td><td>3.82</td><td>3.35</td></tr><tr><td>w/o A-RoPE</td><td>3.34</td><td>3.98</td><td>3.46</td></tr></table>

Table 4: Subjective evaluation results (MOS, 5-point scale). MOS-F: rhythmic flow, MOS-S: speaker similarity, MOS-Q: audio quality.

Results are summarized in Table 4. Statistical significance is assessed using Welch’s t-test, which does not assume equal variances across groups. The proposed model achieves significantly higher MOS-F than the ablation (p = 0.01), suggesting that A-RoPE contributes to improved perceived rhythmic flow. No significant difference was found in MOS-Q (p = 0.06). Both models score below GT in flow and audio quality $( p < 0 . 0 0 1 )$ , which is expected, as GT clips reflect naturally aligned lyric-beat pairs, whereas the models are evaluated under mismatched conditions. For speaker similarity, neither model differs significantly from GT $( p > 0 . 0 5 )$ , indicating that both models preserve speaker identity at a level comparable to ground truth recordings.

## 5. CONCLUSION

We presented BARS, a Flow Matching-based rapping voice generation model, conditioned on a beat track, lyrics, and a reference voice. By introducing A-RoPE in crossattention, the model achieves robust lyric-audio alignment without explicit duration prediction or phoneme-level supervision, substantially reducing lyric skipping and repetition under varying tempo conditions. A lightweight Transformer adapter bridges the modality gap between text-only pretrained ByT5 embeddings and audio latents, matching or exceeding the performance of speech-pretrained text encoders while retaining byte-level flexibility. Experiments demonstrate that the proposed model achieves nearground-truth intelligibility on matched lyric-beat pairs, and maintains strong performance under mismatched conditions where the vocal pacing deviates significantly from the training distribution. Ablation studies confirm that both the A-RoPE mechanism and the adapter are essential to the model’s effectiveness.

A current limitation is that our evaluation is restricted to English. While ByT5 operates at the byte level and is inherently language-agnostic, differences in byte-percharacter ratios across non-ASCII languages may affect the relationship between lyric length and audio duration. Our demo page includes preliminary multilingual examples, but systematic evaluation and adaptation of the scaling mechanism are left to future work.

## 6. ETHICS STATEMENT

The proposed model, BARS, is capable of zero-shot timbre reproduction from a short reference voice clip. While this functionality is designed as a creative tool for rap artists to prototype and refine their vocal delivery, we acknowledge the potential for misuse, such as the unauthorized synthesis of a specific individual’s voice. We emphasize that this technology should be used responsibly and only with the explicit consent of the original speaker.

## 7. AI USAGE STATEMENT

No AI tools were used as part of the core methodology described in this paper. AI-assisted writing tools were used solely for editing purposes, including spell checkers and grammar correction. All content was reviewed and verified by the authors.

## 8. REFERENCES

[1] M. Ohriner, Flow: The rhythmic voice in rap music. Oxford University Press, 2019.

[2] J. Chen, X. Tan, J. Luan, T. Qin, and T.-Y. Liu, “Hi-FiSinger: Towards high-fidelity neural singing voice synthesis,” arXiv preprint arXiv:2009.01776, 2020.

[3] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, “Diff-Singer: Singing voice synthesis via shallow diffusion mechanism,” in Proc. of the AAAI Conf. on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 11 020–11 028.

[4] Z. Zhang, Y. Zheng, X. Li, and L. Lu, “WeSinger 2: fully parallel singing voice synthesis via multisinger conditional adversarial training,” in Proc. of the Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[5] Z. Ning, S. Wang, Y. Jiang, J. Yao, L. He, S. Pan, J. Ding, and L. Xie, “Drop the beat! freestyler for accompaniment conditioned rapping voice generation,” in Proc. of the AAAI Conf. on Artificial Intelligence, vol. 39, no. 23, 2025, pp. 24 966–24 974.

[6] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. of the 11th Int. Conf. on Learning Representations. ICLR, 2023.

[7] L. Xue, A. Barua, N. Constant, R. Al-Rfou, S. Narang, M. Kale, A. Roberts, and C. Raffel, “ByT5: Towards a token-free future with pre-trained byte-to-byte models,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 291–306, 2022.

[8] E. Malmi, P. Takala, H. Toivonen, T. Raiko, and A. Gionis, “DopeLearning: A computational approach to rap lyrics generation,” in Proc. of the 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining, 2016, pp. 195–204.

[9] P. Potash, A. Romanov, and A. Rumshisky, “Ghost-Writer: Using an lstm for automatic rap lyric generation,” in Proc. of the 2015 Conf. on Empirical Methods in Natural Language Processing, 2015, pp. 1919– 1924.

[10] N. I. Nikolov, E. Malmi, C. Northcutt, and L. Parisi, “Rapformer: Conditional rap lyrics generation with denoising autoencoders,” in Proc. of the 13th Int. Conf. on Natural Language Generation, 2020, pp. 360–373.

[11] L. Xue, K. Song, D. Wu, X. Tan, N. L. Zhang, T. Qin, W.-Q. Zhang, and T.-Y. Liu, “DeepRapper: Neural rap generation with rhyme and rhythm modeling,” in Proc. of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th Int. Joint Conf. on Natural Language Processing (Volume 1: Long Papers), 2021, pp. 69–81.

[12] Y. Gao, N. Morioka, Y. Zhang, and N. Chen, “E3 TTS: Easy end-to-end diffusion-based text to speech,” in Proc. ofthe Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.

[13] K. Shen, Z. Ju, X. Tan, E. Liu, Y. Leng, L. He, T. Qin, J. Bian et al., “Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers,” in Proc. of the 12th Int. Conf. on Learning Representations, 2024.

[14] Z. Ju, Y. Wang, K. Shen, X. Tan, D. Xin, D. Yang, E. Liu, Y. Leng, K. Song, S. Tang et al., “Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models,” in Proc. of the Int. Conf. on Machine Learning (ICML). PMLR, 2024, pp. 22 605– 22 623.

[15] M. Le, A. Vyas, B. Shi, B. Karrer, L. Sari, R. Moritz, M. Williamson, V. Manohar, Y. Adi, J. Mahadeokar et al., “Voicebox: Text-guided multilingual universal speech generation at scale,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 14 005–14 034, 2023.

[16] S. E. Eskimez, X. Wang, M. Thakker, C. Li, C.-H. Tsai, Z. Xiao, H. Yang, Z. Zhu, M. Tang, X. Tan et al., “E2 TTS: Embarrassingly easy fully non-autoregressive zero-shot tts,” in Proc. of the Spoken Language Technology Workshop (SLT). IEEE, 2024, pp. 682–689.

[17] K. Lee, D. W. Kim, J. Kim, S. Chung, and J. Cho, “DiTTo-TTS: Diffusion transformers for scalable textto-speech without domain-specific factors,” in Proc. of the 13th Int. Conf. on Learning Representations, 2025.

[18] J. Ao, R. Wang, L. Zhou, C. Wang, S. Ren, Y. Wu, S. Liu, T. Ko, Q. Li, Y. Zhang et al., “SpeechT5: Unified-modal encoder-decoder pre-training for spoken language processing,” in Proc. of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 5723–5738.

[19] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced transformer with rotary position embedding,” Neurocomputing, vol. 568, p. 127063, 2024.

[20] Z. Evans, J. D. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, “Long-form music generation with latent diffusion,” in Proc. of the 25th Int. Society for Music Information Retrieval Conf. ISMIR, Nov. 2024, pp. 429–437.

[21] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[22] H. Kim, J. Lee, J. Yang, and J. Morton, “Length-aware rotary position embedding for text-speech alignment,” in Proc. of the International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026.

[23] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ofthe Int. Conf. on Machine Learning. PMLR, 2023, pp. 28 492–28 518.

[24] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[25] C. K. Reddy, V. Gopal, and R. Cutler, “DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,” in Proc. of the Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 6493–6497.