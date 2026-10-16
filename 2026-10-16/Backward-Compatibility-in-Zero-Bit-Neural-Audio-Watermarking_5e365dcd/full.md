# Backward Compatibility in Zero-Bit Neural Audio Watermarking

Admir Krnić<sup>\*</sup>, Kosta Pavlović, Petar Nedić, Željko Bolević, Slavko Kovačević, Igor Djurović

DeepMark

Neural watermarking systems are typically trained as tightly coupled embedder-detector pairs and implicitly assume static deployment. In real-world deployments, embedders and detectors are frequently updated and selectively decoupled for security and versioning purposes, which can break detection in previously embedded content and, more critically, increase false-positive rates on embedders that a given detector is intentionally not intended to recognize.

In this paper the concept of backward compatibility in zero-bit audio watermarking is formalized. A training framework is proposed that pairs a new detector with a sequence of frozen legacy embedders (its legacy chain), using a performance-weighted sampling strategy so that the detector retains accuracy across all earlier versions. This coupling, while necessary for backward compatibility, is shown to induce an embedder-collapse efect that catastrophically degrades the true negative rate (TNR) against out-of-chain embedders. To counteract this, a Angular Diversity Loss (ADL) is introduced that explicitly forces each new embedder to occupy a distinct region of the output space. The combined approach restores near-perfect TNR with no measurable cost to detection accuracy or perceptual audio quality.

Keywords: audio watermarking; zero-bit watermarking; backward compatibility; legacy chain; revocation

## 1. Introduction

Recent advances in generative AI have substantially increased the realism and accessibility of synthetic audio content. State-of-the-art audio generation techniques are capable of producing high quality speech and audio at scale [1]. As a result, concerns surrounding large-scale misinformation, fraud and reputational damage have intensified, alongside a broader erosion of trust in digital media.

Audio watermarking, a technique that embeds imperceptible signals at generation time and later recovers them to verify provenance, has emerged as a practical mechanism for attribution and misuse prevention. It is also becoming a legal requirement: the European Union’s AI Act (Article 50) mandates that providers of generative AI mark and detect artificially generated content, and its accompanying Code of Practice further requires that a replacement detector be backward compatible, detecting previously marked content with no loss of capability [2].

A line of recent deep-learning approaches jointly trains embedders and detectors, which we survey in Section 2. These systems all share an implicit assumption: an embedder and its detector are trained as a single matched pair and deployed together. In practice, watermarking systems must evolve, with new embedder-detector pairs introduced over time to harden against new attacks, improve quality, or reduce cost, while previously distributed content stays in circulation and must remain verifiable. Existing methods provide no explicit mechanism for an updated detector to keep recognizing watermarks from earlier embedders while still rejecting unrelated content, leading to invalidated watermarks or elevated false positives after updates. Re-embedding old content is often infeasible, and maintaining many detector versions at inference is complex and itself raises the false-positive rate.

We call the ordered sequence of legacy embedders $E _ { 1 } , \ldots , E _ { v - 1 }$ , that a newly trained detector $D _ { v }$ must continue to support, the legacy chain. A backward-compatible $D _ { v }$ must (i) detect watermarks from all embedders in its chain, (ii) reject unwatermarked audio, and (iii) reject watermarks from embedders outside the chain. Objective (iii) is not merely an accuracy concern; it is what makes a compromised version revocable. If an embedder $E _ { i }$ is leaked, an adversary can stamp arbitrary audio with its watermark. The response is to revoke $E _ { i } ,$ dropping it from every future detector’s chain so that any detector issued after the leak rejects it outright. Without this, a single leak permanently invalidates the watermark, as every newer detector keeps accepting the forgeries. This motivates strict version separation rather than monotone forward acceptance.

In this work, we conduct the first systematic study of backward compatibility in neural audio watermarking. The following contributions are made:

• The backward-compatibility objective for zero-bit watermarking is formalized, and the legacy-chain abstraction is introduced together with its security implications.

• A training framework is proposed that combines a performance-weighted sampling scheme across frozen legacy embedders with ADL applied to the new embedder.

• The framework is evaluated across eight model versions that span architectural changes in both embedders (DenseNet, ViTMAE) and detectors (ResNet, AST), and the ADL is shown to substantially restore TNR against out-of-chain embedders with no measurable degradation in detection accuracy or audio quality.

• The same mechanism is shown to enable revocation: a leaked embedder can be dropped from the legacy chain and reliably rejected by detectors trained without it.

## 2. Related Work

Deep-learning-based audio watermarking jointly trains an embedder and a detector to balance imperceptibility and robustness. RobustDNN [3] introduced the basic recipe, WavMark [4], AudioSeal [5], Timbre Watermarking [6], and SilentCipher [7] subsequently improved robustness to common signal-processing attacks. VoiceMark [8] extends robustness to zero-shot voice-cloning attacks. All of these systems train and deploy embedders and detectors as fixed pairs and provide no explicit mechanism for cross-version compatibility.

Petrov et al. [9] study the coexistence and ensembling of independently trained watermarks. Their setting difers from ours in two important ways: embedder-detector pairs are trained independently (with no shared parameters or training signal), and there is no obligation for a later detector to recognize watermarks from an earlier one. As a consequence, their reported interference is minimal.

We instead consider the coupled-training setting required for backward compatibility, and show that this coupling introduces failure modes that independent training does not exhibit.

Continual learning addresses sequential task acquisition without catastrophic forgetting [10], through methods such as Elastic Weight Consolidation [11], Learning without Forgetting [12], and Synaptic Intelligence [13]. These approaches assume a single, architecturally fixed model updated over time and therefore do not accommodate the architectural changes that accompany real-world watermarking updates.

## 3. Problem Formulation

We first review zero-bit watermarking and then formalize the backward-compatibility objective on top of it.

Zero-Bit Watermarking. Zero-bit watermarking aims to decide whether a digital signal contains a watermark. Given an input signal $x \in X$ , an embedder � produces a watermarked signal $x _ { w } = E ( x )$ where the perturbation $E ( x ) - x$ is imperceptible. A detector � then produces a real-valued detection score $D ( x _ { w } ) \in \mathbb { R } _ {  }$ , with larger positive values indicating the presence of a watermark and smaller (negative) values indicating its absence. A binary decision is obtained by comparing the score against a threshold. The pair (�<sub>,</sub> �) is typically trained end-to-end: the detector learns to distinguish watermarked from unwatermarked audio, while the embedder learns to introduce minimal perturbations that the detector can reliably detect.

Backward-Compatibility Objective. We associate every training run with a discrete version index �. For the initial version, $v = 1$ , a single pair $( E _ { 1 } , D _ { 1 } )$ is trained end-to-end. For each subsequent version $v > 1$ , a new embedder $E _ { v }$ is trained jointly with a new detector $D _ { v }$ . The legacy chain at version � is the set of earlier embedders $\{ E _ { i } : i < v \}$ , minus a revocation set � of embedders deliberately dropped (e.g. a leaked one). The legacy chain is kept frozen during training, and $D _ { v }$ is trained against it together with $E _ { v } ,$ so its compatible (in-chain) set is $\{ E _ { i } : i \le v \} \setminus R$ . Every other embedder is out-of-chain: future versions $E _ { j }$ with $j > v$ that did not yet exist, and any revoked $E _ { i } \in R$

Formally, the detector $D _ { v }$ at version � must satisfy:

$D _ { v }$ correctly detects watermarks produced by every embedder from set: $\left\{ E _ { i } : i \leq v \right\} \mid$ � (high TPR within the chain).

• $D _ { v }$ rejects raw audio (high TNR on clean inputs).

$D _ { v }$ rejects watermarks produced by any embedder outside its legacy chain (high TNR against out-of-chain embedders): both future versions $E _ { j }$ with $j > v$ that did not yet exist at training, and any embedder from �.

The last objective we refer to as the version boundary: the line a detector holds between the in-chain embedders it accepts and the out-of-chain ones it rejects, whether future or revoked. This is what distinguishes backward compatibility from standard multi-embedder training: an out-of-chain embedder is not merely absent, it must be actively rejected, which is precisely what makes revoking a leaked embedder possible.

## 4. Method

Although backward compatibility is a general concern, we focus exclusively on zero-bit audio watermarking in this work. Our method has three components: a backward-compatible training setup, a performance-weighted sampling strategy that allocates training iterations across the legacy chain, and a diversity loss that prevents the new embedder from drifting toward its predecessors.

Backward-Compatible Training Setup. Embedders operate in the short-time Fourier transform (STFT) domain. The watermark is added to the magnitude spectrogram within the [500<sub>,</sub> 4000] Hz embedding band and the time-domain signal is reconstructed via inverse STFT using the original phase. Detectors mirror the embedder, they compute the STFT and a mel spectrogram restricted to the same embedding band, then feed it to the detection network.

We extend the idea from SilentCipher [7] of enforcing a hard constraint on the maximum perceptual distortion, which lets us train with a detection-only loss. This distortion constraint drives all embedders to a consistent quality level (PESQ 4.3–4.5). Because every model then sits at essentially the same perceptual quality, we can compare models by detection accuracy alone.

For the detection objective we adopt the push loss introduced by AWARE [14], which augments the standard detection MSE with a term that widens the margin between watermarked and unwatermarked frames:

$$
\mathcal { L } _ { \mathrm { p u s h } } = \mathrm { M S E } ( \hat { y } , y ) - \lambda \left( \frac { 1 } { | y \geq 0 | } \sum _ { y \geq 0 } \hat { y } + \frac { 1 } { | y < 0 | } \sum _ { y < 0 } ( - \hat { y } ) \right) ,\tag{1}
$$

where the second term pushes positive logits (watermarked frames) up and negative logits (unwatermarked frames) down, scaled by the push weight �.

For version $v = 1$ , training follows the standard procedure: a single embedder-detector pair is trained end-to-end with the push loss alone. For versions $v > 1$ , two things change with respect to the $v = 1$ recipe. First, the new embedder $E _ { v }$ and detector $D _ { v }$ may use a diferent architecture from any previous version. This is by design, since real watermarking deployments evolve their architectures over time (Section 5). Second, the training objective is extended with the diversity loss introduced below, which is needed to keep the new embedder distinguishable from the legacy chain and thereby preserve the version boundary.

At each iteration, the detector $D _ { v }$ is trained on the output of a single embedder sampled from the full chain $\{ E _ { 1 } , \ldots , E _ { v } \} \setminus R ,$ chosen by the performance-weighted sampling scheme described in the next paragraph. Because the legacy embedders are frozen and each embedder is drawn independently by this scheme, the training of $D _ { v }$ does not depend on the order of the embedders within the chain. The version index reflects only the order in which the embedders were created.

Performance-Weighted Sampling. Let’s denote with � the chain of embedders the detector is trained to detect $\{ E _ { 1 } , \ldots , E _ { v } \} \setminus R$ . At each training iteration we sample a single embedder $E _ { i } \in C$ according to a probability distribution p over $C ,$ and perform the forward and backward passes for the pair $( E _ { i } , D _ { v } )$ . The distribution p is initialized to uniform. After each validation step we obtain a per-embedder accuracy $A _ { i } \in [ 0 , 1 ]$ for every $E _ { i } \in C$ and update p via a temperature-controlled softmax [15] over negated accuracies:

$$
p _ { i } = \frac { \exp ( -  { \boldsymbol { k } } \cdot  { \boldsymbol { A } } _ { i } ) } { \sum _ { E _ { i } \in C } \exp ( -  { \boldsymbol { k } } \cdot  { \boldsymbol { A } } _ { j } ) } ,\tag{2}
$$

where $k > 0$ controls how aggressively the budget is shifted toward weaker embedders. The scheme adaptively emphasizes the embedders on which the detector currently performs worst. Based on our experiments $k = 8$ yielded the best results.

Embedder Collapse and the Need for a Diversity Loss. Performance-weighted sampling together with push loss is suficient to obtain high TPR within the legacy chain, but it leaves the new embedder $E _ { v }$ entirely free to position its residual watermark anywhere in the signal space. The detector $D _ { v }$ is itself trained from scratch, what is fixed are the legacy embedders $E _ { 1 } , \dots , E _ { v - 1 }$ , whose residuals are frozen throughout training while $E _ { v } \mathrm { ' s }$ residual is the only one free to move. The easiest path to a low detection loss is therefore for $E _ { v }$ to drift toward the frozen legacy embedders’ residual distributions, so that $D _ { v }$ can satisfy every in-chain pair with a single decision boundary. We refer to this failure mode as embedder collapse: the new embedder’s residual ends up implicitly close to those of the frozen legacy embedders, destroying the version boundary.

Angular Diversity Loss. We treat the mel spectrogram of the audio watermarked by each embedder as a “class” on the unit sphere and separate these classes with an additive angular margin, inspired by the margin formulation from ArcFace [16]. The setting, however, is not the fixed-identity classification ArcFace was designed for: an embedder has no identity vector to learn, and only $E _ { v }$ is trainable while the legacy embedders are frozen. The ADL adapts the angular margin to this setting with two changes.

First, we remove ArcFace’s learnable per-class prototypes. Let $f _ { i } ( x )$ denote the batch mean of the mel feature of the audio watermarked by embedder $E _ { i }$ . Each class center is then the normalized batch mean $\mu _ { i } = \bar { f } _ { i } / \| \bar { f } _ { i } \|$ , recomputed on every forward pass. Second, since only $E _ { v }$ moves, the margin is applied only to the mel features of the new embedder’s output. We denote by cos $\theta _ { i } ^ { ( b ) } = \langle f _ { v } ( x _ { b } ) , \mu _ { i } \rangle$ the cosine similarity between mel features of the sample � produced by the new embedder and class center �. Let $N _ { v }$ be the set of class centers from which $E _ { v }$ is repelled, typically $N _ { v }$ contains all predecessors $\{ E _ { i } : i < v \}$ , with the option of removing revoked embedders, as in our revocation experiments. The loss is the additive-margin cross-entropy that pulls every sample toward its own class �:

$$
\mathcal { L } _ { \mathrm { d i v } } = - \frac { 1 } { B } \sum _ { b } \log \frac { e ^ { s \cos ( \theta _ { v } ^ { ( b ) } + m ) } } { e ^ { s \cos ( \theta _ { v } ^ { ( b ) } + m ) } + \sum _ { i \in N _ { v } } e ^ { s \cos \theta _ { i } ^ { ( b ) } } } .\tag{3}
$$

The margin � pushes each sample of $E _ { v }$ at least � radians away from every class center in $N _ { v . }$ while the scale � controls the softmax sharpness.

The proposed loss is essential for preserving the separation between model versions. Without it, the embedding spaces of the legacy and new embedders collapses, eliminating the intended version boundary. The mel spectrogram is a robust representation in which perceptually similar signals produce very similar mel features, so a strong margin or sharp scale would destabilize training. This is why weak parameters $( m = 0 . 1 , s = 8 )$ sufice: the loss still enforces the separation, but needs only a gentle push.

Gradient Normalization. Combining ${ \mathcal { L } } _ { \mathrm { p u s h } }$ and ${ \mathcal { L } } _ { \mathrm { d i v } }$ through a fixed scalar weighting is brittle: the two losses operate on very diferent scales so one term tends to dominate the gradient. We instead balance them with a per-step gradient-norm scaling inspired by GradNorm [17].

At each step we compute the per-loss gradients with respect to the new embedder’s parameters, $\begin{array} { r } { \mathbf { g } _ { \mathrm { p u s h } } = \nabla \mathcal { L } _ { \mathrm { p u s h } } } \end{array}$ and $\mathbf { g } _ { \mathrm { d i v } } = \nabla \mathcal { L } _ { \mathrm { d i v } }$ , and their $L _ { 2 }$ norms. We then rescale both to a common target norm $\begin{array} { r } { \bar { n } = \frac { 1 } { 2 } \left( \lVert \mathbf { g } _ { \mathrm { p u s h } } \rVert + \lVert \mathbf { g } _ { \mathrm { d i v } } \rVert \right) } \end{array}$ , and sum the result:

$$
\mathbf { g } = \frac { \bar { n } } { \| \mathbf { g } _ { \mathrm { p u s h } } \| + \epsilon } \mathbf { g } _ { \mathrm { p u s h } } + \frac { \bar { n } } { \| \mathbf { g } _ { \mathrm { d i v } } \| + \epsilon } \mathbf { g } _ { \mathrm { d i v } } .\tag{4}
$$

Dividing each gradient by its own norm removes raw magnitude so both losses count equally, while the shared factor �<sup>¯</sup> rescales them to a common, representative size instead of unit norm.

## 5. Experimental Setup

This section describes the data, model architectures and evaluation protocol.

Dataset. Training data is drawn from the Mozilla Common Voice corpus [18], with all training audio resampled to 44.1 kHz during preprocessing. The test set consists of audio files sampled from three corpora, LibriSpeech [19], LJSpeech [20], and Mozilla Common Voice, and spans a range of native sampling rates, reflecting a realistic and heterogeneous evaluation condition.

Model Architectures. We evaluate two model families with complementary inductive biases: a convolutional family (DenseNet [21] embedder with a ResNet [22] detector) and a Transformer family (ViTMAE [23, 24] embedder with an AST [25] detector). Each architecture uses a fixed template, across versions we vary only its activation function and its depth, holding every other hyperparameter (widths, kernel sizes, attention heads, feature dimensions) constant. The depth knob is specific to each architecture: dense blocks for DenseNet, encoder/decoder layers for ViTMAE, residual stacks for ResNet, and encoder layers for AST. Table 1 gives the exact per-version configuration.

Table 1 | Model configurations per version.
<table><tr><td>Version</td><td>Embedder</td><td>Detector</td></tr><tr><td> $v _ { 1 }$ </td><td>DenseNet: 1 block, ReLU</td><td>ResNet: 1 stack, ReLU</td></tr><tr><td>v2</td><td>DenseNet: 1 block, Swish</td><td>ResNet: 1 stack, Swish</td></tr><tr><td>v3</td><td>DenseNet: 1 block, GELU</td><td>ResNet: 1 stack, GELU</td></tr><tr><td>v4</td><td>DenseNet: 2 blocks, GELU</td><td>ResNet: 2 stacks, GELU</td></tr><tr><td> $v _ { 5 }$ </td><td>ViTMAE: 1 enc, 2 dec, ReLU</td><td>AST: 4 layers, ReLU</td></tr><tr><td> $v _ { 6 }$ </td><td>ViTMAE: 1 enc, 2 dec, Swish</td><td>AST: 4 layers, Swish</td></tr><tr><td> $v _ { 7 }$ </td><td>ViTMAE: 1 enc, 2 dec, GELU</td><td>AST: 4 layers, GELU</td></tr><tr><td> $v _ { 8 }$ </td><td>ViTMAE: 2 enc, 3 dec, GELU</td><td>AST: 6 layers, GELU</td></tr></table>

The version-invariant hyperparameters are as follows. DenseNet: 64 initial filters, dense blocks of 16 bottleneck layers (3×3 convolutions, growth rate 64, bottleneck reduction 4). ViTMAE: token dimension 128, one attention head, feed-forward ratio 4, dropout 0<sub>.</sub>1. ResNet: 64 initial filters (kernel size 7), stacks of four 3×3 residual blocks, a three-layer LSTM head (128/256/128), and a dilated output convolution (256 channels). AST: hidden size 128, four attention heads, 16×16 patches at stride 10, dropout 0<sub>.</sub>1.

Modeling realistic architectural drift. We train a sequence of eight versions $( v _ { 1 }$ to $v _ { 8 } )$ , one model, i.e. one embedder-detector pair, per version. Over time, practitioners swap in newer activation functions, enlarge networks, and eventually replace entire architecture families with newer ones. Our version sequence is designed to mimic this trajectory. Concretely, we choose architectures along two orthogonal axes:

1. Activation rotation. Within a fixed embedder/detector size, we rotate the activation function in the order in which it became standard in the deep-learning literature: ReLU [26] first, then Swish [27], then GELU [28]. This mimics a team that periodically swaps the activation function for a more recent one without otherwise changing the network.

2. Capacity step-ups. After the activation rotation has cycled through ReLU/Swish/GELU at the original size, we step up the network’s capacity (more dense blocks / more encoder-decoder layers) while keeping the most recent activation. This mimics a team that, after exhausting easy activation upgrades, decides to scale the model.

The same two-axis schedule is applied first to the convolutional family $\left( v _ { 1 }  – v _ { 4 } \right)$ and then repeated on the Transformer family $\left( v _ { 5 } – v _ { 8 } \right)$ , since Transformer-based audio models entered the field later than convolutional ones. The transition from $v _ { 4 }$ to $v _ { 5 }$ simulates the discrete jump that occurs when a deployment switches to a fundamentally newer architecture family rather than enlarging the existing one.

Training Details. All models are trained with Adam at a learning rate of $3 \times 1 0 ^ { - 4 } .$ , decayed by an exponential schedule $( \gamma = 0 . 9 5 )$ , with a batch size of 128 and gradient-norm clipping at 0<sub>.</sub>1.

Evaluation Metrics. For each detector $D _ { v }$ we measure: TPR on in-chain pairs, TNR on unwatermarked audio and on out-of-chain pairs, respectively.

Results are presented as accuracy matrices, evaluated on the test set described above. A binary decision requires a confidence threshold �. We report two experiments: a base setup with performance-weighted sampling and the detection loss, and thefull method that adds the ADL on top. Both use all eight versions with no revocation (� = ∅). Revocation is evaluated separately.

## 6. Results

We fix the confidence threshold at $\tau = 0 . 9 9 .$ , which gives the best TPR and TNR for our objectives across both experiments.

Unwatermarked Audio. We evaluate every detector on the unwatermarked test set and measure the TNR. All detectors, across both training approaches, achieve 100% TNR.

Results Without Diversity Loss. Our base setup pairs each new detector with the legacy chain using performance-weighted sampling and the detection loss alone, with no diversity term. Fig. 1 shows its accuracy matrix.

In-chain TPR is near-perfect, between 99<sub>.</sub>3% and 100% (mean 99<sub>.</sub>8%), so performance-weighted sampling alone lets each detector serve its whole chain. The version boundary, however, fails catastrophically: mean out-of-chain TNR is only 11<sub>.</sub>3%, with 24 of 28 incompatible pairs $\mathrm { a t } \leq 3 \%$ . The collapse is total within the ViTMAE block (0% TNR) and near-total across cross-family transitions (0<sub>.</sub>7–1<sub>.</sub>4%). The sole exception is embedder $v _ { 4 } ,$ rejected by earlier detectors at high TNR.

Residual-Similarity Evidence for Embedder Collapse. The accuracy matrix shows that the version boundary collapses without a diversity loss, but not why. Our hypothesis is that, with no pressure to difer, a new embedder learns watermark residuals that resemble its predecessors’, so their detectors accept it as one of their own.

The residual of an embedder is the perturbation it adds to the input, $E ( x ) - x ,$ i.e. the watermark n GELU [28]. This<sup>signal</sup> <sup>itself.</sup> <sup>To</sup> <sup>test</sup> <sup>the</sup> <sup>hypothesis</sup> <sup>we</sup> <sup>take</sup> <sup>each</sup> <sup>incompatible</sup> <sup>pair</sup> $\left( E _ { M } , D _ { N } \right)$ with $M > N$ (the lower the activation func- <sup>VI.</sup> <sup>RESULTStriangle</sup> <sup>of</sup> <sup>Fig.</sup> <sup>1)</sup> <sup>and</sup> <sup>compare</sup> <sup>two</sup> <sup>quantities:</sup> <sup>its</sup> <sup>TNR</sup> <sup>and</sup>

![](images/0bba8d5deed28b6e588e16f2a22349e1fdac20e9df7c2b4b60479067f86eca9b.jpg)  
Figure 1 | Accuracy matrix for the base setup (no diversity loss). Rows index embedder versions, columns index detector versions. The main diagonal and entries above it show TPR entries below show TNR. In-chain TPR is near-perfect, but the lower triangle collapses: the detector accepts almost all out-of-chain watermarks embedder<sub>as its own.</sub>

$$
\bar { c } _ { M , N } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } c ( M , k ) ,\tag{5}
$$

dense blwhere $c ( M , k )$ <sub>ore</sub> across both training approaches, achieve 100% TNR.is the averaged cosine similarity between the residuals of $E _ { M }$ and $E _ { k }$ . A high $\hat { c } _ { M , N }$ ng themeans $E _ { M } { ' } \mathbf { s }$ <sub>ecent</sub> Results Without Diversity Loss. Our base setu <sub>watermark</sub> <sub>looks,</sub> <sub>in</sub> <sub>residual</sub> <sub>space,</sub> <sub>like</sub> <sub>the</sub> <sub>watermarks</sub> $D _ { N }$ airs<sub>already</sub> <sub>accepts.</sub> <sub>If</sub> <sub>the</sub> fter exhausting easy <sup>each</sup> <sup>new</sup> <sup>detector</sup>hypothesis holds, TNR should fall as $\bar { c } _ { M , N }$ the le<sub>rises.</sub>

diversity term. Fig. 1 shows its accuracy matrix.Table 2 | Out-of-chain TNR grouped by mean residual similarity $\hat { c } _ { M , N }$
<table><tr><td> $\bar { c } _ { M , N }$ </td><td>n</td><td>TNR (%)</td></tr><tr><td>[0.45,0.55)</td><td>3</td><td>98.9</td></tr><tr><td>[0.55, 0.65)</td><td>1</td><td>2.7</td></tr><tr><td>[0.65,0.75)</td><td>15</td><td>0.9</td></tr><tr><td>[0.75,0.85)</td><td>9</td><td>0.4</td></tr></table>

Across all pairs, $\bar { c } _ { M , N }$ embedder v<sub>4</sub>, rejected by earlier detectors at high TNR.and TNR are strongly negatively correlated (Pearson $r = - 0 . 8 6 ,$ Spearman $\rho = - 0 . 7 3$ both $p < 1 0 ^ { - 4 } )$ sidual-Similarity Evidence for Embedder Collapse.. The more a new embedder’s residual resembles those in the detector’s The accuracy matrix shows that the version bounda<sub>chain,</sub> <sub>the</sub> <sub>less</sub> <sub>reliably</sub> <sub>that</sub> <sub>detector</sub> <sub>rejects</sub> <sub>it.</sub> <sub>Binning</sub> <sub>the</sub> <sub>pairs</sub> <sub>by</sub> $\bar { c } _ { M , N }$ lapses<sub>(Table 2) makes the efect</sub> without a diversity loss, but not why. Our hypothesis is that,<sub>concrete: TNR stays near-perfect only in the lowest-similarity bin and falls of a clif the moment</sub> $\hat { c } _ { M , N }$ <sub>audio and on</sub> with no pressure to differ, a new embedder learns watermark<sub>crosses</sub> ∼<sub>0.55. This confirms that residual closeness directly drives the failed rejections and</sub> residuals that resemble its predecessors’, so their dete<sub>motivates</sub> <sub>a</sub> <sub>loss</sub> <sub>that</sub> <sub>keeps</sub> <sub>successive</sub> <sub>embedders</sub> <sub>apart</sub> <sub>in</sub> <sub>residual</sub> <sub>space.</sub>

<sup>rices,</sup> <sup>evaluated</sup> <sup>on</sup> The residual of an embedder is the perturbation it adds<sup>Results</sup> <sup>With</sup> <sup>Diversity</sup> <sup>Loss.</sup> <sup>The</sup> <sup>full</sup> <sup>method</sup> <sup>adds</sup> <sup>the</sup> <sup>ADL</sup> <sup>on</sup> <sup>top</sup> <sup>of</sup> <sup>the</sup> <sup>base</sup> <sup>setup,</sup> <sup>keeping</sup> compatibility itself<sub>performance-weighted</sub> <sub>sampling</sub> <sub>and</sub> <sub>the</sub> <sub>detection</sub> <sub>loss</sub> <sub>and</sub> <sub>combining</sub> <sub>losses</sub> <sub>through</sub> <sub>gradient</sub> normalization. Fig. 2 presents the resulting accuracy matrix.

![](images/0877dcbbbfd8bc19fe888657c6d449e9be9c7e4e53dbfa8822f442d4fec37b05.jpg)  
Figure 2 | Accuracy matrix for the full method (with the ADL). Same layout as Fig. 1. In-chain TPR is unchanged, while the lower triangle recovers to near-perfect TNR.

Across all pairs, c¯<sub>M,N</sub> and TNR are strongly negatively <sub>therefore test, rathCompatibility</sub> <sub>is</sub> <sub>preserved:</sub> <sub>in-chain</sub> <sub>TPR</sub> <sub>stays</sub> <sub>between</sub> <sub>99.3</sub> <sub>and</sub> <sub>100%</sub> <sub>(mean</sub> <sub>99.9%),</sub> <sub>matching</sub> <sub>the</sub> <sup>correlated</sup> <sup>(Pearson</sup> <sup>r</sup> <sup>=</sup> −<sup>0.86,</sup> <sup>Spearman</sup> <sup>ρ</sup> <sup>=</sup> −base setup. The diversity loss is therefore free in compatibility terms.

those in the detector’s chain, the less reliably that detector <sup>reports</sup> <sup>the</sup> <sup>min/maOut-of-chain rejection, the failure mode of the base setup, is recovered: mean TNR rises from</sup> rejects it. Binning the pairs11.3% to 99.98% (27 of 28 incompatible pairs at $1 0 0 \% ,$ (Table II) makes the  the last at 99.3%), with no residual failure effect concrete: TNR stays near-perfect only incluster. The version boundary holds within the convolutional family $( v _ { 1 } – v _ { 4 } )$ est-, within the transformer family $\left( v _ { 5 } – v _ { 8 } \right)$ similarity, and across the $v _ { 4 }  v _ { 5 }$ falls off a cliff the moment c¯ crosses  family switch, enforcing separation across both within- and 0cross-family transitions.

Table 3 | Revocation: TNR (%) against the revoked embedder.
<table><tr><td>Revoked</td><td>No div.</td><td>Div.,  $E _ { i }$  unused</td><td>Div.,  $E _ { i }$  pushed</td></tr><tr><td> $v _ { 3 }$  (conv.)</td><td>0.7</td><td>24.3</td><td>100.0</td></tr><tr><td> $v _ { 5 }$  (transf.)</td><td>0.0</td><td>95.9</td><td>100.0</td></tr></table>

Compatibility is pr<sub>Revocation.</sub> <sub>Revoking</sub> <sub>a</sub> <sub>leaked</sub> <sub>embedder</sub> $E _ { i }$ rved: in-chain TPR stays between We presented the <sub>means</sub> <sub>dropping</sub> <sub>it</sub> <sub>from</sub> <sub>every</sub> <sub>future</sub> <sub>detector’s</sub> <sub>legacy</sub> 99.3 and 100% (mean 99.9%), matching the base<sub>chain, which is considered successful if a detector trained without</sub> $E _ { i }$ etup. The ibility in zero-bit <sub>still rejects it. Starting from</sub> the chain of versions $v _ { 1 }$ ers<sub>to</sub> $v _ { 7 } ,$ loss is therefore free in compatibility terms.  <sub>,</sub> <sub>we</sub> <sub>revoke</sub> <sub>a</sub> <sub>single</sub> <sub>embedder,</sub> <sub>either</sub> <sub>a</sub> <sub>convolutional-family</sub> <sub>one</sub> $\left( v _ { 3 } \right)$ or a transformer-family one (� ), and for each retrain a new embedder–detector pair on the six surviving versions $( v _ { 1 }$ <sup>11.3</sup>  <sup>99.98</sup> <sup>27</sup>   to �<sub>7</sub> excluding the revoked one) under three conditions: (i) the baseline with no diversity loss, (ii) the diversity loss with $E _ { i }$ unused and (iii) the diversity loss with $E _ { i }$ added to the diversity term only (forcing the new embedder to difer from it) while kept out of <sup>1 4</sup>     the detection loss, so the detector is never trained to accept it. Table 3 reports TNR against $E _ { i }$ . The baseline barely rejects the revoked embedder, and merely omitting it from a diversity-trained chain is unreliable. Only when $E _ { i }$ is pushed away by the diversity loss does the new embedder produce distinct residuals, and the detector rejects the revoked embedder fully in both families.

Third-Party Watermarks. In production, our detectors must not accept third-party watermarks, and our watermarked content must not be accepted by third-party detectors. We tested both directions against the publicly released zero-bit AudioSeal model [5], using all eight pair of models trained with the full method: (i) the audio watermarked by each of our embedders is passed to the AudioSeal detector, and (ii) the audio watermarked by the AudioSeal embedder is passed to each of our detectors. Because AudioSeal operates at 16 kHz, audio is resampled to 16 kHz at its input.

We use AudioSeal’s default decision threshold of 0<sub>.</sub>5. For every one of the eight versions, and in both directions, not a single file is detected, so TNR is 100% for every pair. This agrees with the negligible interference that Petrov et al. [9] report for independently trained watermarks.

## 7. Robustness Analysis

Maximizing robustness to signal-processing and codec attacks is out of scope, and none of our models are trained with attack augmentation. We ask a narrower question: does backward compatibility itself survive distortion? A deployed watermark is transmitted and edited before detection, so a framework that held only on clean audio would be of little practical use. We therefore test, rather than optimize for, robustness, re-running the evaluation under the DeepMark benchmark [29]. Table 4 reports the min/max/mean TPR and TNR per attack for both training conditions; the version boundary is preserved under attack.

The version boundary survives attack. The baseline’s TNR was already broken on clean audio and stays broken under every attack (mean 11–14%), while ours was near-perfect on clean audio and stays that way, with mean TNR ≥ 98% on all 27 attacks and exactly 100% on 21.

TPR is essentially unchanged. In-chain detection tracks the baseline across attacks, confirming that adding the diversity loss has a negligible cost in compatibility under distortion.

## 8. Conclusion

We presented the first systematic study of backward compatibility in zero-bit deep-learning audio watermarking. We formalized the backward-compatibility objective via the legacy-chain abstraction, identified the embedder-collapse failure mode that breaks version boundaries when a new detector is trained against frozen legacy embedders, and proposed a training framework that combines performance-weighted sampling and push loss together with ADL balanced by gradient normalization. Across eight versions and two architecture families, the method restores out-of-chain TNR without degrading within-chain detection or perceptual quality, and the restored version boundary holds under distortion. The same mechanism enables revocation: a leaked embedder dropped from the chain is reliably rejected by detectors trained without it. We see the legacy chain as a useful abstraction for future watermarking systems that must evolve under deployment constraints, and we leave robustness under adversarial attack and multibit extensions to future work.

Table 4 | Per-attack TPR (%) and TNR (%) at $\tau = 0 . 9 9 .$
<table><tr><td rowspan="4"></td><td colspan="6">Baseline</td><td colspan="6">Ours</td></tr><tr><td colspan="3">TPR (%)</td><td colspan="3">TNR (%)</td><td colspan="3">TPR (%)</td><td colspan="3">TNR (%)</td></tr><tr><td>min</td><td>max</td><td>avg</td><td>min</td><td>max</td><td>avg</td><td>min</td><td>max</td><td>avg</td><td>min</td><td>max</td><td>avg</td></tr><tr><td>Additive Noise</td><td>25.0</td><td>69.5</td><td>52.7</td><td>26.3</td><td>100.0</td><td>49.5</td><td>16.8</td><td>76.3</td><td>44.3</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Bandstop</td><td>66.2</td><td>66.2</td><td>66.2</td><td>33.7</td><td>100.0</td><td>41.0</td><td>65.5</td><td>66.2</td><td>66.1</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Crop Beginning</td><td>99.3</td><td>100.0</td><td>99.7</td><td>0.0</td><td>100.0</td><td>11.6</td><td>98.6</td><td>100.0</td><td>99.7</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Crop Random</td><td>98.6</td><td>100.0</td><td>99.7</td><td>0.0</td><td>100.0</td><td>11.7</td><td>98.6</td><td>100.0</td><td>99.8</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Cut Samples</td><td>99.3</td><td>100.0</td><td>99.7</td><td>0.0</td><td>99.3</td><td>11.6</td><td>97.9</td><td>100.0</td><td>99.6</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Echo</td><td>98.6</td><td>100.0</td><td>99.7</td><td>0.0</td><td>100.0</td><td>11.5</td><td>96.6</td><td>100.0</td><td>99.4</td><td>98.6</td><td>100.0</td><td>99.9</td></tr><tr><td>Flanger</td><td>96.6</td><td>100.0</td><td>99.3</td><td>0.0</td><td>100.0</td><td>13.0</td><td>92.5</td><td>100.0</td><td>98.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Flip Samples</td><td>96.6</td><td>100.0</td><td>99.4</td><td>0.0</td><td>100.0</td><td>12.1</td><td>95.9</td><td>100.0</td><td>99.0</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Gaussian Noise</td><td>75.0</td><td>98.6</td><td>87.9</td><td>1.3</td><td>99.3</td><td>21.4</td><td>58.7</td><td>97.2</td><td>82.9</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Highpass</td><td>99.3</td><td>100.0</td><td>99.7</td><td>0.0</td><td>99.3</td><td>11.1</td><td>98.6</td><td>100.0</td><td>99.8</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Inverted Time-Str.</td><td>98.6</td><td>100.0</td><td>99.8</td><td>0.0</td><td>100.0</td><td>11.1</td><td>97.2</td><td>100.0</td><td>99.7</td><td>72.9</td><td>100.0</td><td>98.2</td></tr><tr><td>LPC</td><td>98.6</td><td>100.0</td><td>99.7</td><td>0.0</td><td>99.3</td><td>11.2</td><td>98.6</td><td>100.0</td><td>99.8</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Lowpass</td><td>99.3</td><td>100.0</td><td>99.8</td><td>0.0</td><td>99.3</td><td>11.2</td><td>99.3</td><td>100.0</td><td>99.8</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>MP3 Compression</td><td>97.2</td><td>100.0</td><td>99.3</td><td>0.0</td><td>99.3</td><td>11.6</td><td>97.2</td><td>100.0</td><td>99.0</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Network Transm.</td><td>62.8</td><td>100.0</td><td>88.6</td><td>0.0</td><td>100.0</td><td>21.4</td><td>23.6</td><td>95.9</td><td>74.4</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>PCM Quantization</td><td>98.6</td><td>100.0</td><td>99.5</td><td>0.0</td><td>99.3</td><td>11.3</td><td>97.9</td><td>100.0</td><td>99.4</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Pink Noise</td><td>26.3</td><td>78.3</td><td>57.2</td><td>20.2</td><td>100.0</td><td>46.2</td><td>18.2</td><td>85.8</td><td>50.1</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Pitch Shift</td><td>98.6</td><td>100.0</td><td>99.6</td><td>0.0</td><td>100.0</td><td>11.5</td><td>97.9</td><td>100.0</td><td>99.6</td><td>93.2</td><td>100.0</td><td>99.6</td></tr><tr><td>Quantization</td><td>73.6</td><td>99.3</td><td>89.3</td><td>0.6</td><td>100.0</td><td>20.2</td><td>56.7</td><td>96.6</td><td>81.9</td><td>97.2</td><td>100.0</td><td>99.8</td></tr><tr><td>Replacement</td><td>97.2</td><td>100.0</td><td>99.0</td><td>0.6</td><td>99.3</td><td>12.4</td><td>93.9</td><td>100.0</td><td>98.3</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Resampling</td><td>99.3</td><td>100.0</td><td>99.7</td><td>0.0</td><td>99.3</td><td>11.2</td><td>98.6</td><td>100.0</td><td>99.8</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>STFT Quantization</td><td>70.2</td><td>97.9</td><td>93.2</td><td>1.3</td><td>99.3</td><td>13.3</td><td>42.5</td><td>100.0</td><td>81.7</td><td>91.2</td><td>100.0</td><td>98.8</td></tr><tr><td>Sign Inversion</td><td>99.3</td><td>100.0</td><td>99.8</td><td>0.0</td><td>99.3</td><td>11.2</td><td>99.3</td><td>100.0</td><td>99.8</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Smoothing</td><td>96.6</td><td>100.0</td><td>98.9</td><td>0.0</td><td>100.0</td><td>13.8</td><td>66.2</td><td>100.0</td><td>93.5</td><td>85.1</td><td>100.0</td><td>98.2</td></tr><tr><td>Time Stretch</td><td>98.6</td><td>100.0</td><td>99.8</td><td>0.0</td><td>100.0</td><td>11.3</td><td>97.9</td><td>100.0</td><td>99.7</td><td>95.2</td><td>100.0</td><td>99.6</td></tr><tr><td>Collusion</td><td>94.5</td><td>100.0</td><td>98.2</td><td>0.0</td><td>100.0</td><td>13.9</td><td>92.5</td><td>100.0</td><td>97.6</td><td>99.3</td><td>100.0</td><td>99.9</td></tr><tr><td>Zero-Cross Inserts</td><td>99.3</td><td>100.0</td><td>99.8</td><td>0.0</td><td>100.0</td><td>11.4</td><td>98.6</td><td>100.0</td><td>99.7</td><td>99.3</td><td>100.0</td><td>99.9</td></tr></table>

## References

[1] C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023.

[2] European Commission and AI Ofice. Code of practice on transparency of AI-generated content. Technical report, European Commission, Directorate-General for Communications Networks, Content and Technology, June 2026. Section 1: Marking and Detection of AI-Generated Content.

[3] K. Pavlović, S. Kovačević, I. Djurović, and A. Wojciechowski. Robust speech watermarking by a jointly trained embedder and detector using a DNN. Digital Signal Processing, 122:103381, 2022.

[4] G. Chen, Y. Wu, S. Liu, T. Liu, X. Du, and F. Wei. WavMark: Watermarking for audio generation. arXiv preprint arXiv:2308.12770, 2023.

[5] R. S. Roman, P. Fernandez, A. Défossez, T. Furon, T. Tran, and H. Elsahar. Proactive detection of voice cloning with localized watermarking. arXiv preprint arXiv:2401.17264, 2024.

[6] C. Liu, J. Zhang, T. Zhang, X. Yang, W. Zhang, and N. Yu. Detecting voice cloning attacks via timbre watermarking. arXiv preprint arXiv:2312.03410, 2023.

[7] M. K. Singh, N. Takahashi, W.-H. Liao, and Y. Mitsufuji. SilentCipher: Deep audio watermarking. In INTERSPEECH, 2024.

[8] H. Li, Z. Wu, X. Xie, J. Xie, Y. Xu, and H. Peng. VoiceMark: Zero-shot voice cloning-resistant watermarking approach leveraging speaker-specific latents. arXiv preprint arXiv:2505.21568, 2025.

[9] A. Petrov, S. Agarwal, P. Torr, A. Bibi, and J. Collomosse. On the coexistence and ensembling of watermarks. Advances in Neural Information Processing Systems, 38:107058–107094, 2026.

[10] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54–71, 2019.

[11] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017.

[12] Z. Li and D. Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2935–2947, 2017.

[13] F. Zenke, B. Poole, and S. Ganguli. Continual learning through synaptic intelligence. In International Conference on Machine Learning, pages 3987–3995. PMLR, 2017.

[14] K. Pavlović, L. Stanarević, P. Nedić, E. Nešović, S. Kovačević, and I. Djurović. AWARE: Audio watermarking with adversarial resistance to edits. arXiv preprint arXiv:2510.17512, 2025.

[15] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[16] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4690–4699, 2019.

[17] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International Conference on Machine Learning, pages 794–803. PMLR, 2018.

[18] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4218–4222, 2020.

[19] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. LibriSpeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210. IEEE, 2015.

[20] K. Ito and L. Johnson. The LJ speech dataset. https://keithito.com/LJ-Speech-Dataset/, 2017.

[21] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4700–4708, 2017.

[22] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[23] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[24] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022.

[25] Y. Gong, Y.-A. Chung, and J. Glass. AST: Audio spectrogram transformer. arXiv preprint arXiv:2104.01778, 2021.

[26] V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pages 807–814, 2010.

[27] P. Ramachandran, B. Zoph, and Q. V. Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017.

[28] D. Hendrycks and K. Gimpel. Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415, 2016.

[29] S. Kovačević, E. Nešović, K. Pavlović, P. Nedić, and I. Djurović. DeepMark benchmark: Redefining audio watermarking robustness. IEEE Access, 2026.