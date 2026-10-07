# Feature Encoding in VAE-based Audio Decoders: Effects of Input, Depth and Distribution

Louis McCallum and Mick Grierson

Abstract—Neural audio synthesis models like the Realtime Audio Variational autoEncoder (RAVE) achieve impressive generation quality, yet how their internal representations encode musical features remains poorly understood. We present a systematic layer-wise and cross-layer cluster analysis of RAVE decoder activations across three models trained on different musical domains, tested with four stimulus types. We then evaluate architectural generalization with a general purpose EnCodec model. For RAVE, we find that synthetic stimuli are encoded well across models and audio features (pitch |ρ|=0.45, 5.1× the null, BPM $\lvert \rho \rvert = 0 . 7 6 , 8 . 6 \times$ the null). These results are reduced but still substantively apparent when using natural audio (mean across features |ρ|=0.25, 2.8× the null). Natural audio sees a stronger encoding when nonlinear probes are used (mean across features $R ^ { 2 } { = } { \bf 0 . 5 6 } ,$ 18× the null, +0.152 nonlinear gain over the linear probe $R ^ { 2 } )$ . Encoding strength varies throughout the layers of the decoder and an increased ability to joint-encode in the middle layers is seen across all audio features $( \beta _ { 2 }$ all negative, $p < 0 . 0 5 )$ The general purpose EnCodec decoder also sees similar strong synthetic responses across audio features, similar nonlinear gains for natural audio joint encoding and similar depth profiles. We find the best cross-layer cluster improves the strength $( r = 0 . 6 5 ,$ $p \ = \ 0 . 0 0 6 )$ and prevalence $( r \ = \ 0 . 7 5 , \ p \ = \ 0 . 0 0 1 )$ of BPM encoding when compared against the best whole layers within the same section, with no effect for joint encoding. These findings advance the interpretability of neural audio models and inform targeted control strategies for neural synthesis.

Index Terms—Audio synthesis, clustering methods, feature extraction, neural networks, variational autoencoders (VAEs)

## I. INTRODUCTION

DVANCES in neural audio synthesis have enabled realtime, high-fidelity music generation through models like RAVE [1], EnCodec [2], and SoundStream [3]. These variational autoencoder (VAE)-based architectures learn compressed latent representations that can reconstruct audio with perceptual quality rivaling traditional synthesis methods. However, understanding what musical features these models encode, and where in their architecture this encoding occurs, remains largely unexplored.

This knowledge gap matters for both scientific understanding and practical application. From a scientific perspective, neural audio models are increasingly deployed in creative and research contexts, yet we lack basic knowledge of whether pitch, tempo, or timbre occupy identifiable locations within these architectures or whether they are encoded in distributed, entangled representations that resist decomposition. From a practical standpoint, if we can identify which decoder layers or neuron populations represent specific musical features, this understanding could inform the design of more targeted control interfaces for neural synthesizers, structured pruning strategies, and diagnostic tools for model debugging.

Existing work has examined learning constraints in latent space [4] and controllability [5], but layer-wise activation analysis of the kind common in computer vision [6] has received limited attention in audio. Studies of self-supervised speech models such as Wav2Vec [7] have revealed hierarchical acoustic feature learning, with early layers capturing spectral patterns and later layers encoding semantic content. Layerwise probing of models including HuBERT and WavLM has demonstrated that pitch, tempo, and energy encoding varies systematically with network depth [8], [9]. Whether similar hierarchical organization exists in generative audio models remains an open question.

A methodological challenge in this domain is distinguishing genuine feature encoding from chance correlation. We address this through permutation testing [10], establishing empirical null distributions for each model-dataset combination and defining feature prevalence (the percentage of neurons exceed ing the permutation-derived 95th percentile threshold, referred to by $\% > p _ { 9 5 } )$ as a complement to conventional encoding strength (mean $| \rho | )$ . This dual-metric approach reveals aspects of encoding that neither measure captures alone: encoding strength reflects how intensely neurons track a feature, while prevalence reflects how broadly that tracking is distributed across the neural population. We further use nonlinear probes to establish the abilities of neuron subset to jointly encode these features (measured by $R ^ { 2 } )$ . Where valid, results are compared to EnCodec [2] to investigate whether the findings are specific to RAVE’s variational design.

We address:

RQ1: How do linear and nonlinear probes compare across synthetic and natural audio?

RQ2: How does feature encoding vary with depth in the network?

RQ3: Can cross-layer clustering identify neuron groups that encode features more effectively than individual layers?

We contribute:

1) Systematic layer-wise analysis across three models × four datasets using encoding strength, permutation-validated feature prevalence and nonlinear probes.

2) Evidence that RAVE decoders encode pitch $( | \rho | = 0 . 4 5$ 5.1× the null) and BPM $( | \rho | = 0 . 7 6 ,$ , 8.6× the null) strongly, linearly and broadly throughout layers when presented with synthetic stimuli and that this is maintained for natural audio even when spectral features are also tested (mean across features $| \rho | = 0 . 2 5 , 2 . 8 \times \mathrm { ~ t h e ~ n u l l } )$

Natural audio sees a stronger encoding when nonlinear probes are used (mean across features $R ^ { 2 } { = } 0 . 5 6$ , 18× the null, +0.152 nonlinear gain over the linear probe $R ^ { 2 } )$ .

3) We see some complete and some feature specific patterns in responses of layers through the model when considering depth. Evidence of all four features $( \beta _ { 2 }$ all negative, $p < 0 . 0 5 )$ shows non-monotonic joint encoding with midnetwork peaks. Three of the four features (pitch, centroid, and bandwidth $\beta _ { 2 }$ all positive, $p < 0 . 0 5 )$ show per-neuron concentration toward late layers.

4) We find the best cross-layer cluster improves the strength $( r ~ = ~ 0 . 6 5 , ~ p ~ = ~ 0 . 0 0 6 )$ and prevalence $( r ~ = ~ 0 . 7 5$ $p \ : = \ : 0 . 0 0 1 )$ of BPM encoding when compared against the best whole layers within the same section. We find no improvement for joint-encoding power.

5) A cross-architecture comparison showing the findings are not specific to RAVE’s variational design.

Our analysis proceeds in sequential steps, each building upon the results of previous work. We examine feature encoding on a per-layer basis, investigating how strongly and where musical features are represented across RAVE’s 21 decoder layers, and whether hierarchical patterns are shared across models or domain-specific. We then extend this to crosslayer analysis, using K-means clustering to identify neuron populations spanning multiple layers that collectively encode features more coherently than any single layer. Together, these analyses provide a detailed picture of how one of the most commonly used neural audio synthesis architectures internally represents musical structure and timbral information.

## II. RELATED WORK

Our work combines neural audio synthesis and network interpretability. We review prior work in each area, identify the gap our analysis addresses, and position our methodological contributions.

## A. Neural Audio Synthesis and Generative Models

1) Neural audio codecs: Recent advances in neural audio compression have produced high-quality generative models that learn compact latent representations of audio. Sound-Stream [3] introduced residual vector quantization for efficient audio coding, while EnCodec [2] extended this approach with improved perceptual quality. RAVE [1], the focus of our analysis, employs variational autoencoders with adversarial training to achieve real-time audio synthesis with latency suitable for musical applications. More recent generative architectures have moved in two directions with transformerbased autoregressive models such as MusicGen [11] and AudioLM [12] that operate over discrete codebook tokens, and diffusion-based synthesizers such as AudioLDM [13] that introduce iterative denoising. Despite this proliferation, the convolutional VAE family remains the dominant choice for real-time musical applications where latency constraints preclude autoregressive or diffusion-based alternatives. Across all of these architectures the question of what and where their internal representations encode has received limited empirical treatment.

2) Audio feature encoding: Understanding what neural networks learn about audio has been explored primarily through probing classifiers [14], in which a supervised model is trained to predict an acoustic or linguistic feature from frozen layer activations. Studies of Wav2Vec [7] and related self-supervised models reveal hierarchical acoustic feature learning, with early layers capturing spectral patterns and late layers encoding semantic content. Pasad et al. [15] extend this paradigm to lexical content, while Chiu et al. [9] conduct layer-wise probing of HuBERT, WavLM, and Wav2Vec 2.0 for pitch, tempo, and energy, finding a consistent three-stage hierarchy across discriminative speech models.

This body of work has, however, two limitations relevant to our setting. First, it focuses almost exclusively on encoders trained for discriminative or self-supervised objectives whereas the analogous question for generative decoders has not been systematically addressed. Second, reliance on trained probes raises capacity confounds [16] in that a sufficiently expressive probe can extract features that are computable from the representation rather than genuinely encoded in it. Our analysis addresses both gaps by extending layer-wise probing to a generative decoder and complementing trained probes with rank-correlation and permutation-based prevalence measures that are robust to capacity confounds.

## B. Neural Network Interpretability and Mechanistic Understanding

Neural network interpretability encompasses a broad range of approaches that can be organized along several dimensions: passive versus active methods, the type of explanation produced (examples, attribution, hidden semantics, or rules), and the scope of interpretation (local, semi-local, or global) [17]. Our work occupies a specific position in this landscape as we perform passive, post-hoc analysis aimed at uncovering hidden semantics at a global scope that characterizes what neuron populations across entire decoder hierarchies encode, rather than explaining individual predictions or modifying the architecture for interpretability.

1) Activation-based interpretation: Visualising and manipulating neuron activations has become a cornerstone of interpretability research. Feature visualization [18] demonstrates that scaling activations amplifies or suppresses feature detection in vision models. Deep visualization [19] reveals neuron selectivity through activation maximization. Network dissection [20] quantifies interpretability by measuring neuronconcept alignment against a labelled visual concept dataset. These methods share an assumption that meaningful features align with individual neurons or simple linear combinations thereof and subsequent work has shown this to be only partially correct as features are frequently encoded in superposition across overlapping neuron populations [21]. This motivates the more distributed, prevalence-based view of feature encoding we adopt here.

2) Attribution methods: A complementary family of interpretability methods assigns credit to input features for a model’s output. Layer-wise Relevance Propagation (LRP) [22] decomposes predictions by propagating relevance scores back-

![](images/4dc5456fe4f729a9602fc3f01759f1bf5e20076375dd838978ba7b8708e003c1.jpg)  
strides 2,4,4,4 · total upsampling ×128 · 28 probed convolutions across 21 named modules  
LeakyReLU at net.1, 5, 10, 15, 20 · each residual = 3×1 then 1×1 conv with skip

Fig. 1. Diagram of RAVE Decoder architecture

ward through the network. DeepLIFT [23] compares activations to a reference to propagate activation differences, while Integrated Gradients [24] accumulates gradients along a path from a baseline input to provide axiomatic attribution. SHAP [25] unifies several attribution approaches under a game-theoretic framework. Although powerful for explaining individual predictions, attribution methods are ill-suited to the question we address as they identify which inputs drive an output, not which internal representations carry specific information about a feature. For generative models in particular, where the meaningful ”output” is a high-dimensional waveform rather than a class label, the attribution framing maps poorly onto questions of musical feature encoding, motivating our focus on activation-based rather than attribution-based analysis.

3) Circuits and mechanistic interpretability: Cammarata et al. [26] seek to reverse-engineer neural networks by identifying computational subgraphs that implement specific functions. They emphasize hierarchical feature composition in which early layers detect simple patterns that later layers compose into complex representations. Subsequent work in this tradition has identified phenomena that offer increasingly fine-grained accounts of how trained networks represent and manipulate information. This includes induction heads in language models [27], polysemantic neurons and feature superposition [21], and sparse-autoencoder-based decomposition of activations into monosemantic features [28].

Two observations from this line of work bear directly on our setting. First, the mechanistic interpretability program has focused on language models and, to a lesser extent, vision models. Audio generative models remain largely unexamined despite their growing deployment. Second, the central tension between localized (per-neuron) and distributed (cross-neuron, cross-layer) encoding remains an open question for any new architecture. Our prevalence-based and cluster-based analyses address this tension directly for RAVE. Rather than presupposing that features are localized to individual neurons or that they are diffusely distributed, we measure where on this spectrum

each feature actually sits.

4) Neuron clustering: Grouping neurons by functional similarity has proven effective for understanding network organization. Raghu et al. [29] introduced SVCCA for comparing layer representations via canonical correlation analysis, revealing that similar layers cluster together. Bauerle et al. [30]¨ find groups of images that are treated similarly by extracting similar activation profiles in the activation space of a neural network layer. A common limitation of these approaches is that clustering is typically performed within individual layers, leaving open the question of whether functionally-related neurons spanning across multiple layers might cohere into more informative groupings than any single layer provides. We address this directly through cross-layer clustering.

## C. Summary of the Gap

Having covered the relevant literature, we summarize the gap our work addresses. First, layer-wise probing of audio representations has been concentrated on discriminative speech encoders and comparable analysis of generative musical decoders is largely absent. Second, existing audio interpretabil ity work rarely separates encoding strength (how intensely individual units track a feature) from encoding prevalence (how broadly that tracking is distributed across the neural population). This conflates two distinguishable properties that the superposition and circuits literatures suggest can dissociate. Third, neuron-clustering analyses have typically been confined to within-layer groupings, leaving cross-layer functional organization in generative architectures unexamined. Our contributions of a three-metric layer-wise analysis and cross-layer clustering address this gap for the RAVE family, with a methodological framework with potential to be portable to other architectures.

## III. METHOD

To balance focused internal validation against reliable musical properties and externally valid real world sounds, we present a probe analysis on three models and four datasets (three naturally occurring and one synthetic). This means that we can draw conclusions for 12 combinations and draw insights from differences and similarities between combinations. This helps us move towards broad and reliable claims about RAVE models’ encoding and decoding of music in more general terms. We further compare these results with EnCodec, a pretrained general purpose decoder [2].

## A. Training Datasets

Models tend to converge best on homogeneous datasets that contain a focused set of timbres. However, within this they can successfully train on sets of sounds that are polyphonic, multi-instrumental, a-rhythmic and even nonmusical (e.g. Foley sounds). As such, it is highly challenging to find a single RAVE model with enough timbral breadth to draw fully generalizable conclusions about universal modeling performance.

We collected three datasets of distinct categories of musical recordings to train three RAVE models from scratch. The strings (N=740) and drums (N=759) datasets consist of samples from a commercial online library [31]. The strings were selected to allow for a good distribution of key and tempo during downstream balancing and include solo and ensemble. Drums were also selected to allow good distribution of tempo during downstream balancing and are spread across a variety of contemporary genres including rock, pop, hip hop, jazz and jungle. The vocals dataset (N=486) is the full catalog of a commercially available pop singer, where vocals have been separated using Meta’s Demucs algorithm [32].

## B. Model

RAVE employs a variational autoencoder architecture in which the encoder compresses raw audio into a low dimensional latent space and the decoder reconstructs the waveform through a series of upsampling and convolutional stages (see Fig 1). This comparatively simple architecture (when compared to other models such as transformers) makes the extraction and analysis of activations straightforward.

We trained three models on the datasets described above (strings, drums, vocals). Each uses an identical decoder architecture and only the training data differs. The models are mono and have a 44.1 kHz sample rate with 128 latent dimensions. The datasets are within the good practice lengths (minimum 2–3 hours), and the models were trained for at least 1.5M steps using a standard RAVE V2 approach [33].

Treating each convolution as a separate layer yields 28 probed units across 21 named modules. Activations are captured after weight normalization and nonlinearity but before residual addition, isolating each layer’s transformation.

## C. Model Inputs

1) Sampling: To assess how well the models encode audio features, we constructed four sets of audio (three from the training datasets and one synthesized) and forward-passed them through each model while collecting activations. Each audio segment is paired with feature labels that are combined with the activations for subsequent analysis. Valid results require sets that are balanced in the distribution of each feature, drawn from diverse source audio, and sufficiently large to yield adequate statistical confidence.

As a baseline representing the most reduced signal with minimal confounding variation, we generated two synthetic datasets, one varying in pitch and the other in tempo. The pitch set comprises of sine tones at equally spaced interval and the tempo set is made of pulse trains.

To determine whether the observed effects extend to more complex, naturally occurring audio, and to assess the influence of in- and out-of-distribution inputs, we constructed representative, balanced subsets of the data used to train the three models.

For drums, the audio is divided into four second chunks and randomly sampled to obtain a uniform distribution across tempo, prioritizing source-file diversity. For strings, the same procedure is applied with uniform sampling across both tempo and pitch. For vocals, samples are selected to favor high RMS energy (unlikely to be silence) and samples are randomly sampled to obtain a uniform distribution across tempo and pitch, again prioritizing source-file diversity. This results in 500 BPM balanced and 500 pitch balanced samples where appropriate.

2) Features: We choose four features for this analysis, namely, pitch, tempo (BPM), spectral centroid and spectral bandwidth. Pitch is a fundamental perceptual attribute of musical audio that contributes to melody and harmony [34]. Tempo captures rhythmic structure that unfolds over time rather than within a single frame, testing whether the model encodes temporally-extended properties as well as instantaneous ones. Spectral centroid provides the perceived brightness of a sound and is a good correlate of timbre [35]. Spectral bandwidth measures the spread of energy around the centroid, complementing it by capturing the noisiness of the spectrum and distinguishing pure tones from broadband content [36].

Feature labels are obtained from sample-library metadata where available. For drums and strings, tempo is taken from the library metadata. For vocals, no such metadata is available so tempo is computed over the whole track using librosa.beat\_track [37], [38]. For strings and vocals, files are chunked into four second parts and pitch is estimated over the full four second clip using librosa.pyin [39]. A 100-ms segment most likely to correspond to a stable note is identified as that exhibiting the highest confidence and lowest variation. Its median pitch is stored in the pitch dataset along with the 100-ms segment.

Spectral centroid and spectral bandwidth are also calculated using librosa [38] for the full four seconds for the whole 1000 audio files (BPM and pitch balanced) and averaged into one label. A balanced subset of 500 is then used for consistency to avoid artefacts of dataset size.

Distributions are shown in Table I. Although well-matched throughout, feature distributions can vary across datasets due to differences in source material. This is a constraint of working with natural datasets that is considered acceptable to work with complex, naturally occurring audio.

TABLE I  
ACOUSTIC FEATURE DISTRIBUTIONS PER DATASET AFTER PER-FEATURE BALANCED SAMPLING. ALL CELLS USE N=500 SAMPLES, STRATIFIED TO SPAN THE FULL FEATURE RANGE. EXCLUSIONS FOR INAPPROPRIATE FEATURE / DATA COMBINATIONS APPLY
<table><tr><td rowspan="2">Dataset</td><td colspan="3">Pitch (Hz)</td><td colspan="3">BPM</td><td colspan="3">Spectral Centroid (Hz)</td><td colspan="3">Spectral Bandwidth (Hz)</td></tr><tr><td>Min</td><td>Max</td><td>Mean (σ)</td><td>Min</td><td>Max</td><td>Mean (σ)</td><td>Min</td><td>Max</td><td>Mean (σ)</td><td>Min</td><td>Max</td><td>Mean (σ)</td></tr><tr><td>Strings</td><td>80</td><td>788</td><td>293 (200)</td><td>70</td><td>140</td><td>104 (21)</td><td>306</td><td>7357</td><td>1976 (1123)</td><td>969</td><td>6082</td><td>2601 (813)</td></tr><tr><td>Drums</td><td></td><td></td><td></td><td>60</td><td>175</td><td>115 (34)</td><td>652</td><td>7902</td><td>3961 (1711)</td><td>948</td><td>6074</td><td>3553 (1105)</td></tr><tr><td>Vocals</td><td>80</td><td>761</td><td>284 (126)</td><td>65</td><td>123</td><td>90 (17)</td><td>918</td><td>7905</td><td>3938 (995)</td><td>1803</td><td>6608</td><td>3799 (779)</td></tr><tr><td>Synthetic</td><td>80</td><td>800</td><td>313 (200)</td><td>60</td><td>180</td><td>120 (35)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/469e2a7108f351411aabcb5b144e97699bc8d67e920965f75749648e0ee20c5a.jpg)  
Fig. 2. Diagram representing the three metrics used for comparing the relationship of acoustic feature values and neuron activations. |ρ| can be further averaged across layers or models. Null $p _ { 9 5 }$ derived from permutation tests. Predictive model for $R ^ { 2 }$ can be linear or MLP.

## D. Probes

For each (model, dataset, feature) cell we used three measures to examine relationships between acoustic features and decoder activations (see Fig 2). Each is computed on 500 audio files balanced for that particular feature (as described above). For each audio segment, we encoded and decoded audio with RAVE, capturing activations at each convolutional unit via forward hooks.

The first part of our approach follows the linear probing tradition established in [14], adapted to the generative audio domain. We use Spearman rank correlations between layer activations and acoustic features as a more constrained but more interpretable measure that avoids the capacity concerns raised in probing classifier literature [40]. This correlational approach has been applied to layer-wise analysis of speech SSL models [8], [9], where it has successfully revealed hierarchical feature encoding.

For layer level joint encoding, two probe types are used. A linear regression baseline and a nonlinear MLP probe. Reporting both allows us to characterize whether features are encoded in a directly linear form or require nonlinear extraction.

For linear baseline we use a ridge regression with regularization parameter $\alpha = 1 . 0$ . Selected as a baseline because it represents the simplest probe that recovers a linear projection from activations to the target feature. We use a two layer MLP with ReLU activation, followed by a linear output head. The hidden dimension scales with the number of input neurons as

$$
h = \operatorname* { m i n } ( 6 4 , \operatorname* { m a x } ( 8 , \ : n _ { \mathrm { n e u r o n s } } / 8 ) ) .
$$

This scaling provides sufficient capacity for layers and clusters with many neurons while preventing severe overparameterization for small neuron subsets. Training uses Adam optimizer with learning rate $1 0 ^ { - 3 }$ , batch size 64, for 100 epochs with early stopping on validation loss (patience 10).

Probe performance is evaluated with 5-fold cross-validation. Splits are random across the balanced sample. Predictions are pooled across folds to compute the coefficient of determination $( R ^ { 2 } )$ on held-out predictions. The same fold structure is used for linear and nonlinear probes to enable direct comparison. Target features are log-transformed for pitch and spectral centroid (which span multiple octaves and are perceptually logarithmic), and used directly for tempo and spectral bandwidth. $R ^ { 2 }$ values reflect variance explained on the transformed target.

## E. Metrics

We characterize feature encoding using three complementary measures (see Fig 2). Each captures a distinct aspect of how feature information is represented across neurons.

1) Encoding: We firstly provide the per-neuron mean Spearman $| \rho |$ for each neuron. This is the absolute Spearman rank correlation between activation and the target feature, averaged across neurons where appropriate. It provides the average per-neuron encoding strength.

Prevalence above null $( \% > p _ { 9 5 } )$ is the percentage of neurons whose $| \rho |$ exceeds the 95th percentile of a permutationderived null distribution. This measures how broadly distributed feature-encoding neurons are within the population.

We finally provide a joint probe $R ^ { 2 }$ to assess the performance of a multi-neuron probe (linear or nonlinear) on the neuron vector as a whole. This measures the joint-encoding capacity of the population by telling us how well the feature is recoverable from the full set of neurons together.

For each (model, dataset, feature) cell, we construct an empirical null distribution by permuting the target feature labels across samples and recomputing all three measures. Permutations are repeated $K = 5 0 0$ for Spearman correlations and $K = 1 0 0$ times for nonlinear probes (which are themselves each 5-fold cross-validated). For each measure, we record the mean and 95th percentile of the null distribution. The 95th percentile defines the chance threshold against which the observed values are compared.

2) Paired Comparisons: For paired comparisons across cells (natural vs synthetic), we use Wilcoxon signed-rank tests where pairing is matched by layer position and model.

Pairing within layer is deliberate because encoding strength varies strongly and non-monotonically with depth, comparing natural vs synthetic values at the same layer removes layer as a confound and isolates the effect of distributional match from the effect of network depth. Across the three models, this yields 84 layer-matched pairs per feature, which we tested with a two-sided Wilcoxon signed-rank test and report matchedpairs rank-biserial correlation r as the effect size. We note that pairs sharing a model are not strictly independent, as they inherit a common model-level component. We report the between-model intraclass correlations (ICC) and the exact sign-permutation p stratified by each factor.

3) Depth: Where we test whether layer position predicts encoding strength, for each (model, dataset) cell, we fit polynomial regressions of the metric on normalized layer depth $d \in \ [ 0 , 1 ]$ , including both linear $( \beta _ { 0 } + \beta _ { 1 } d )$ and quadratic $( \beta _ { 0 } + \beta _ { 1 } d + \beta _ { 2 } d ^ { 2 } )$ specifications. Slopes from each cell are treated as the unit of observation with the curvature coefficient $( \beta _ { 2 } )$ tested across cells with a one-sample Wilcoxon signedrank test. Per-cell regressions are well-conditioned (each cell contributes 28 layer observations).

We report the matched-pairs rank-biserial correlation r as the effect size. Cells are not fully independent as they share both models and datasets so we report the between-model and between-dataset ICC and the exact sign-permutation $p$ stratified by each factor.

4) Inferential and Descriptive: The confirmatory analyses specified by our research questions are the natural-vs-synthetic probe comparison (RQ1), the depth-dependence of encoding for natural audio (RQ2), and the cluster-versus-layer comparison (RQ3). Exploratory analyses are reported descriptively without inferential claims and include synthetic-stimulus depth profiles (3 cells, insufficient for inference). The comparison of models (RAVE and EnCodec) is also non-inferential as the difference in structure makes per-layer matching not possible and low N makes cell-level comparisons equally weak.

We control the false-discovery rate (Benjamini–Hochberg) within each family of related tests (a given measure across the four features) and report adjusted p-values. The three measures are treated as complementary views of the same representation rather than substitutable tests of a single hypothesis. Confidence intervals are 95% bootstrap.

## F. Cross-Layer Clustering

To identify functionally-coherent neuron groups beyond individual layers, we performed hierarchical cross-layer clustering. We divided the 21 decoder layers into three sections based on architectural depth: early (layers 0-7), middle (layers 8-14), and late (layers 15+). This sectioning reflects both architectural structure (early layers process compressed latents, late layers generate waveforms) and our per-layer findings (different metrics peak at different depths).

Within each section, we used K-means clustering on the selected neurons based on their activation patterns across the audio corpus. Each neuron’s feature vector consists of its activations across the full set of stimuli independently of any probed feature. We used $k = 6$ clusters to group neurons with similar response profiles (see Supplementary Materials Fig S2 to S4 for more details on the choice of k).

We apply the same layer level metrics described above by recalculating the observed mean $| \rho |$ and $\% > p _ { 9 5 }$ and retraining new linear and nonlinear probes to the new neuron subset to investigate the cluster’s joint encoding.

For each cell, feature, and measure, we identified the best-performing cross-layer cluster and the best-performing single layer within that section, and computed their difference. Differences were tested across cells using the same framework as the depth analysis where the cell is the unit of observation, and ICC for model and dataset is reported. Because clusters and their best-matching layer are selected from overlapping neuron pools, the two sets may not be disjoint. However, this does not undermine the paired comparison as shared neurons contribute equally to both members of a pair and so cancel during within-cell differencing.

## G. EnCodec Comparison

We also ran the above evaluations on Meta’s EnCodec model [2]. We chose the pretrained 32 kHz model designed for music and ran all four datasets through, collecting activations and calculating each of the three metrics across the model’s layers. It is structurally similar (input convolution, upsampling stages each followed by a residual block, output convolution), however, it uses residual vector quantization rather than variational continuous latents. We identified 13 appropriate layers to capture activations. This comparison allows us to see if results hold across a different decoder architecture.

TABLE II  
ENCODING METRICS AGGREGATED BY FEATURE AND CONDITION, REPORTED MEAN ACROSS CELLS. SYNTHETIC SPECTRAL VALUES NOT ANALYZED. × NULL IS RATIO OF OBSERVED MEAN AGAINST PERMUTATION DERIVED NULL MEAN.
<table><tr><td rowspan="2">Feature</td><td rowspan="2">Condition n</td><td rowspan="2"></td><td colspan="3">Per-neuron</td><td colspan="4">Joint probe  $R ^ { 2 }$ </td></tr><tr><td>Mean  $| \rho |$ </td><td>× null</td><td> $\% > \mathsf { p } 9 5$ </td><td>Linear</td><td>Nonlinear</td><td>× null</td><td>NL gain</td></tr><tr><td rowspan="4">Pitch</td><td>Natural</td><td>6</td><td>0.20</td><td>2.24</td><td>56.4</td><td>0.27</td><td>0.38</td><td>13.18</td><td>0.11</td></tr><tr><td>Synthetic</td><td>3</td><td>0.45</td><td>5.12</td><td>82.3</td><td>0.99</td><td>1.00</td><td>35.31</td><td>0.00</td></tr><tr><td>ID</td><td>2</td><td>0.18</td><td>2.10</td><td>50.5</td><td>0.17</td><td>0.43</td><td>14.41</td><td>0.11</td></tr><tr><td>OOD</td><td>4</td><td>0.20</td><td>2.32</td><td>59.3</td><td>0.25</td><td>0.36</td><td>12.55</td><td>0.11</td></tr><tr><td rowspan="4">BPM</td><td>Natural</td><td>9</td><td>0.20</td><td>2.29</td><td>59.7</td><td>0.27</td><td>0.42</td><td>13.56</td><td>0.15</td></tr><tr><td>Synthetic</td><td>3</td><td>0.76</td><td>8.63</td><td>91.6</td><td>1.00</td><td>0.99</td><td>28.57</td><td>0.00</td></tr><tr><td>ID</td><td>3</td><td>0.19</td><td>2.17</td><td>55.4</td><td>0.32</td><td>0.54</td><td>17.11</td><td>0.22</td></tr><tr><td>OOD</td><td>6</td><td>0.20</td><td>2.35</td><td>61.9</td><td>0.25</td><td>0.36</td><td>11.76</td><td>0.12</td></tr><tr><td rowspan="3">Spectral Centroid</td><td>Natural</td><td>9</td><td>0.31</td><td>3.45</td><td>69.8</td><td>0.61</td><td>0.76</td><td>22.48</td><td>0.15</td></tr><tr><td>ID</td><td>3</td><td>0.26</td><td>2.92</td><td>61.5</td><td>0.63</td><td>0.80</td><td>24.69</td><td>0.17</td></tr><tr><td>OOD</td><td>6</td><td>0.33</td><td>3.72</td><td>73.9</td><td>0.61</td><td>0.74</td><td>21.44</td><td>0.13</td></tr><tr><td rowspan="3">Spectral Bandwidth</td><td>Natural</td><td>9</td><td>0.27</td><td>3.15</td><td>69.9</td><td>0.47</td><td>0.68</td><td>21.52</td><td>0.20</td></tr><tr><td>ID</td><td>3</td><td>0.24</td><td>2.73</td><td>65.9</td><td>0.50</td><td>0.75</td><td>25.60</td><td>0.25</td></tr><tr><td>OOD</td><td>6</td><td>0.29</td><td>3.36</td><td>71.9</td><td>0.46</td><td>0.64</td><td>19.68</td><td>0.18</td></tr></table>

![](images/7be256cc4d90cc89ccb5de33d8d3d668dba0f9e56cbb95e423fbe13e63364382.jpg)

![](images/bd9fd9d983c4e89d2c6f267018311acc4dfc515895188b88f11ccb1fa204d025.jpg)  
Fig. 3. Plots demonstrating what percentage of neurons encode multiple features across all cells. Left shows how many features are $> p _ { 9 5 }$ for each neuron; synthetic can have max. 2 features as this is all that is measured. Right shows what percentage of eligible neurons are $>$ p<sub>95</sub> for each feature (N=121,472 for BPM (all data), N=91,104 for pitch (no drums), N=91,104 for spectral centroid and spectral bandwidth (no synthetic)).

## IV. RESULTS

## A. Synthetic Stimuli

Shown in Table II, we found that for pitch and BPM (the two features measured), synthetic stimuli showed high $| \rho |$ (0.45 and 0.76) showing a correlation substantially above chance between activation and features at a neuron level. Almost all neurons responded above chance $( \% > p _ { 9 5 }$ is 82% and 92%) and further shown in Fig 3, 86.2% of neurons responded to both and less than 1% neither. We see the encoding is mostly captured linearly and that joint encoding for $R ^ { 2 }$ is near the ceiling. This likely reflects the stimuli’s controlled single-dimension variation, which removes the confounding co-variation present in natural audio and makes each feature

easily recoverable.

## B. Natural Audio

For natural audio, we see again in Table II that all features are correlated above chance with $| \rho |$ between 0.20 and 0.31. $\% > p _ { 9 5 }$ is also high. We also see a much larger gain when analysing the joint encoding between linear and nonlinear models than we saw with synthetic inputs. $R ^ { 2 }$ is well above chance. Fig 3 shows encoding is often shared with 73.4% of neurons correlating over chance for at least 3 features.

When directly comparing synthetic and natural audio for $| \rho |$ and $\% > p _ { 9 5 }$ using layer matched paired Wilcoxon signed rank tests, synthetic stimuli were encoded far more strongly than natural audio across all measures for pitch (100% of layermatched pairs saw rank-biserial $\mathrm { ~ r ~ } \geq 0 . 9 9 9 ;$ all $p \ < \ 0 . 0 0 1 )$ . Between-model ICC was moderate to high (0.097–0.835), reflecting variation in the magnitude of the synthetic advantage across models rather than any disagreement in direction (the effect direction was unanimous across all models).

TABLE III  
MEDIAN PER-CELL QUADRATIC COEFFICIENT $\beta _ { 2 }$ OVER NORMALIZED DEPTH (POSITIVE = U-SHAPED, NEGATIVE = INVERTED-U) WITH [IQR]. r IS THE MATCHED-PAIRS RANK-BISERIAL EFFECT SIZE. BETWEEN-MODEL AND BETWEEN-DATASET INTRACLASS CORRELATIONS (ICC) ARE PROVIDED, ALONG WITH THE BENJAMINI–HOCHBERG-ADJUSTED EXACT SIGN-PERMUTATION p. ALL NATURAL CELLS; SYNTHETIC OMITTED (DESCRIPTIVE, N=3).
<table><tr><td colspan="6"></td><td colspan="2">ICC</td><td></td></tr><tr><td>Meas.</td><td>Feature</td><td>N</td><td>β2 [IQR]</td><td>r</td><td>model</td><td>data</td><td> $p _ { \mathrm { a d j } }$ </td></tr><tr><td rowspan="4">Mean |ρ|</td><td>Pitch</td><td>6</td><td>0.11 [0.09, 0.17]</td><td>1.00</td><td>0.37</td><td>0.00</td><td>0.042*</td></tr><tr><td>BPM</td><td>9</td><td>-0.09 [-0.11, 0.03]</td><td>-0.38</td><td>0.00</td><td>0.00</td><td>0.324</td></tr><tr><td>Spectral Centroid</td><td>9</td><td>0.25 [0.16, 0.55]</td><td>0.78</td><td>0.74</td><td>0.00</td><td>0.042*</td></tr><tr><td>Spectral Bandwidth</td><td>9</td><td>0.43 [0.17, 0.57]</td><td>0.87</td><td>0.32</td><td>0.00</td><td>0.042*</td></tr><tr><td rowspan="4"> $\% > \mathrm { p } 9 5$ </td><td>Pitch</td><td>6</td><td>26.16 [12.34, 43.61]</td><td>0.81</td><td>0.06</td><td>0.01</td><td>0.125</td></tr><tr><td>BPM</td><td>9</td><td>-12.79 [-24.61, 8.96]</td><td>-0.38</td><td>0.00</td><td>0.00</td><td>0.250</td></tr><tr><td>Spectral Centroid</td><td>9</td><td>16.42 [1.41, 41.15]</td><td>0.64</td><td>0.67</td><td>0.00</td><td>0.125</td></tr><tr><td>Spectral Bandwidth</td><td>9</td><td>41.62 [24.16, 47.15]</td><td>0.96</td><td>0.00</td><td>0.00</td><td>0.031*</td></tr><tr><td rowspan="4"> $R ^ { 2 }$ </td><td>Pitch</td><td>6</td><td>-0.38 [-0.45, -0.13]</td><td>-1.00</td><td>0.00</td><td>0.72</td><td>0.031*</td></tr><tr><td>BPM</td><td>9</td><td>-0.65 [-1.01, -0.55]</td><td>-1.00</td><td>0.00</td><td>0.00</td><td>0.005**</td></tr><tr><td>Spectral Centroid</td><td>9</td><td>-0.29 [-0.34, -0.20]</td><td>-1.00</td><td>0.70</td><td>0.00</td><td>0.005**</td></tr><tr><td>Spectral Bandwidth</td><td>9</td><td>-0.38 [-0.60, -0.35]</td><td>-1.00</td><td>0.43</td><td>0.00</td><td>0.005**</td></tr></table>

## C. Out-of-Distribution Audio

Feature encoding remains well above chance when a model was tested with out-of-distribution audio it had not been trained on. Out-of-distribution per-neuron correlations exceeded the null threshold by $2 { \ - } 4 \times$ and joint $R ^ { 2 }$ remained high (e.g. spectral centroid $R ^ { 2 } \ = \ 0 . 7 4$ , tempo $R ^ { 2 } = 0 . 3 6$ on unfamiliar audio), indicating the learned feature representations generalize substantially beyond the training domain. Notably, on per-neuron measures encoding was often as strong or stronger out-of-distribution than in-distribution, suggesting the network’s individual channels correlate with these features in a manner not specific to the training data.

## D. Effect of Layer Depth

Depth analysis is conducted on natural audio cells, where the per-cell aggregation has adequate statistical power. For synthetic stimuli, joint encoding is at ceiling across all layers $( R ^ { 2 } > 0 . 9 9 )$ , and depth-related variation is too small to be meaningful given the small sample of synthetic cells (N=3 per feature). Results are shown in Fig 4 and synthetic $| \rho |$ is still included to show its similar pattern to natural audio, although we are unable to statistically determine significance.

Joint $R ^ { 2 }$ followed an inverted-U depth profile (negative curvature, peak in mid-network) for every feature: pitch (−0.38; $N = 6 , p = 0 . 0 3 1 , r = - 1 . 0 0 )$ , BPM (median $\beta _ { 2 } = - 0 . 6 5$ $N = 9 , p = 0 . 0 0 5 , r = - 1 . 0 0 )$ , spectral centroid $( - 0 . 2 9 ;$ $N ~ = ~ 9 , ~ p ~ = ~ 0 . 0 0 5 , ~ r ~ = ~ - 1 . 0 0 )$ and spectral bandwidth $( - 0 . 3 8 ; N = 9 , p = 0 . 0 0 5 , r = - 1 . 0 0 )$

The effect was unanimous in direction across models and datasets. BPM showed no between-group dependence (both $\mathrm { I C C s } = 0 )$ , while pitch, spectral centroid, and spectral bandwidth carried moderate ICC on one grouping axis each (Table III), reflecting variation in the magnitude of the curvature across groups rather than any disagreement in its direction. The inverted-U is therefore robust as a directional finding but its precise magnitude is less certain for the high-ICC features.

Per-neuron mean $| \rho |$ followed the opposite, U-shaped profile (positive curvature, strongest at network extremes) for pitch (median $\beta _ { 2 } = + 0 . 1 1 ; N = 6 , p = 0 . 0 4 2 , r = + 1 . 0 0 )$ , spectral centroid $( + 0 . 2 5 ; N = 9 , p = 0 . 0 4 2 , r = + 0 . 7 8 )$ , and spectral bandwidth $( + 0 . 4 3 ; \ : N = 9 , \ : p = 0 . 0 4 2 , \ : r = + 0 . 8 7 ) ;$ tempo showed no significant curvature. ICC shows moderate effect of model but 0 for dataset.

We examined the five top layers in each cell to determine if there were any layers that consistently encoded particular features across combinations, however, no specific layers could be determined as shared when corrected for the multiple tests.

## E. Cross-Layer Clustering

For comparing clusters, we pick the best cluster per section (after filtering out clusters that are less than 1% of the total available) and compare this against the best layer in the corresponding section. Filtering prevents best selection picking a small cluster (approx. 1–5 neurons) that may have high correlation by chance.

We then compare the deltas across each cell (full results in Table S4 see also Fig 5). Comparing each cell’s best crosslayer cluster against its best single layer, tempo (BPM) was the only feature to show a reliable cluster advantage, and it did so across two measures: per-neuron mean |ρ| $( \mathrm { H L } = + 0 . 0 1 9 .$ $r = + 0 . 6 5 ) , \mathcal { I } _ { 0 } > p _ { 9 5 } ( + 6 . 1 5 , r = + 0 . 7 5 )$ . Both showed low to moderate dataset ICC but no effect of model. High concordant r indicates a consistent direction across cells but a magnitude that varies by dataset.

![](images/629cf013b6f13bfd6dd03d302ebcebe02815f789195919b5c87130d7d7f5bf56.jpg)

![](images/41fb37c5eff339744f5c35f058a5768fc1c3c02cb0887032eb94682157af82d0.jpg)

![](images/15da968a02773e44ccb469e26b2e59aa7659bc050687b4e14557d91cc0c2269d.jpg)

![](images/86990796e764bf2a3f1fedccf186800fe57e9150d5344a2994d396071984affc.jpg)  
Fig. 4. Plots of per layer mean $| \rho | ,$ neuron prevalence and $R ^ { 2 }$ across four audio features for natural audio, reported as median across cells. Only per layer mean |ρ| displayed for synthetic audio as other features are near ceiling throughout (e.g. flat). Significance values come from testing curvature coefficient (β<sub>2</sub>) across cells with a one-sample Wilcoxon signed-rank test. Not possible for synthetic as N=3 (too low).  
Within-section cluster vs best individual layer (one point = one section of one cell) Above identity line: cluster > layer. Below: layer > cluster.

![](images/fddd70dc735017654ddc3864631d8a77bcec6175275d06eb04f2efd5587b1757.jpg)

![](images/b986fbff6828ba75399d0c29d55c2f898aecd81cede649ac454305b35b4b8fe4.jpg)

![](images/4db27549c8acc831908414cce237e0cc74d75b395b7f9c4b36b7fa80d1631183.jpg)  
Fig. 5. Best cluster against best layer per section for each cell for all three metrics $( | \rho | , \mathcal { T } _ { O } > p _ { 9 5 }$ and $R ^ { 2 } )$ ; only natural BPM $| \rho |$ and > p<sub>95</sub> show improvement for clusters.

## F. EnCodec Comparison

Table IV demonstrates that findings from RAVE models are well replicated in another architecture. Synthetic stimuli show strong linear encoding, with BPM encoding with more strength and more prevalence than pitch. Synthetic stimuli were encoded more strongly than natural audio across every measure and feature, replicating the pattern observed in RAVE. For pitch: mean |ρ| (0.56 vs 0.24), $\% > p _ { 9 5 }$ (85.7% vs 64.6%) and joint $R ^ { 2 }$ (0.99 vs 0.45). For tempo: mean |ρ| (0.66 vs 0.20), $\% > p _ { 9 5 }$ (94.0% vs 55.6%) and joint $R ^ { 2 }$ (0.99 vs 0.59). All comparisons $N = 1 3 , p < 0 . 0 0 1 , r = + 1 . 0 0$

As with RAVE, spectral centroid had the highest $| \rho | ,$ , with good nonlinear joint encoding (nonlinear gain strong for BPM (+0.20), present but weaker for other features). Given the low number of cells we are only able to provide descriptive analysis of the effect of depth, but we do see in Table S3 that $\beta _ { 2 }$ is all negative for $R ^ { 2 }$ , indicating that joint encoding peaks at intermediate decoder layers. We also see similar peaks in later layers for $| \rho |$ (spectral centroid, spectral bandwidth, BPM for natural, pitch for synthetic). We also see little improvement in joint encoding for clustering.

## V. DISCUSSION

## A. Robustness to Complexity

When a synthetic stimulus varies along a single axis (pitch or BPM, with all else fixed), the corresponding RAVE activations vary in a linear manner with that axis. This is a strong result and it implies that RAVE has learned a representation where the feature has a near-linear coordinate in activation space and that small input changes produce proportional activation changes.

TABLE IV  
ENCODEC ENCODING METRICS AGGREGATED BY FEATURE AND CONDITION, RESULTS ARE MEAN ACROSS CELLS. MULTIPLES (× NULL) ARE THE OBSERVED VALUE DIVIDED BY THE CORRESPONDING NULL BASELINE (p95 FOR PER-NEURON, MEAN NULL $R ^ { 2 }$ FOR THE JOINT PROBE).
<table><tr><td rowspan="2">Feature</td><td rowspan="2">Condition</td><td rowspan="2">n</td><td colspan="3">Per-neuron</td><td colspan="3">Joint probe  $R ^ { 2 }$ </td></tr><tr><td>Mean  $| \rho |$ </td><td>× null  $\% > \mathsf { p } 9 5$ </td><td>Linear</td><td>Nonlinear</td><td>× null</td><td>NL gain</td></tr><tr><td rowspan="2">Pitch</td><td>Natural</td><td>2</td><td>0.24</td><td>2.76</td><td>64.6 0.44</td><td>0.45</td><td>18.03</td><td>0.05</td></tr><tr><td>Synthetic</td><td>1</td><td>0.56</td><td>6.41</td><td>85.7</td><td>0.99 0.99</td><td>35.91</td><td>0.01</td></tr><tr><td rowspan="2">BPM</td><td>Natural</td><td>3</td><td>0.20</td><td>2.26</td><td>55.6</td><td>0.46 0.59</td><td>21.85</td><td>0.20</td></tr><tr><td>Synthetic</td><td>1</td><td>0.66</td><td>7.45</td><td>94.0</td><td>1.00 0.99</td><td>32.81</td><td>0.00</td></tr><tr><td>Spectral Centroid</td><td>Natural</td><td>3</td><td>0.33</td><td>3.71</td><td>65.9 0.83</td><td>0.85</td><td>33.80</td><td>0.07</td></tr><tr><td>Spectral Bandwidth</td><td>Natural</td><td>3|</td><td>0.26</td><td>2.98</td><td>63.6</td><td>0.79 0.82</td><td>26.14</td><td>0.09</td></tr></table>

Beyond this, the substantial encoding of natural audio despite its acoustic complexity demonstrates that RAVE’s decoder also maintains detectable feature correlations under realistic conditions. While synthetic stimuli produced significantly stronger encoding, natural audio still showed meaningful correlations, indicating the network’s activations track features even in signals containing timbral variation, temporal irregularities, and polyphonic content.

When considering differing responses, we could also characterize synthetic and natural stimuli as triggering different aspects of the same representation. Synthetic stimuli demonstrate what the network’s representation is like if all variation aligned with a single feature axis. Natural stimuli show what the representation is when feature variation is entangled with other natural sources of variation. This is perhaps why it requires nonlinear probing to uncover.

## B. Feature Specialization

Neuron-level encoding is multi-feature. In natural audio, 58.7% of neurons encode both pitch and tempo above the null threshold, and 73% encode at least three of the four analyzed features. This overlap indicates that RAVE encodes acoustic features in a distributed, entangled manner rather than allocating dedicated neurons to individual properties. A single neuron typically contributes to the representation of several features at once.

This being said, spectral centroid shows the highest perneuron correlation and encoding across natural cells, suggesting that brightness is the most neuron-aligned acoustic property in the network’s representation.

The synthetic-to-natural comparison reveals asymmetry between pitch and tempo. Under controlled synthetic stimuli, tempo was the more strongly encoded of the two, yet on natural audio the two converged to the same value. The degradation of tempo from synthetic to natural conditions is far steeper than for pitch.

This could be a reflection of the different ways the two properties exist in the two formats. In natural audio, tempo must be inferred from irregular, expressively-timed onsets embedded in polyphonic textures. This is a far noisier estimation problem than reading a fixed periodicity.

Pitch, by contrast, is more of a locally-instantiated property. Although it too weakens on natural audio, its encoding is less dependent on the regularity that the synthetic condition supplies. Therefore, whilst it is an interesting result, we can caution against reading synthetic encoding strength as a direct proxy for how a model represents tempo in realistic use.

That tempo alone benefits from cross-layer aggregation is consistent with its status as the only temporally-extended feature examined. Unlike other features, tempo is defined over time and its encoding may draw on layers operating at different timescales, so that grouping neurons across layers recovers information no single layer holds in isolation.

## C. Joint-Layer and Per-Neuron Encoding

In some cases, we see dissociation between metrics, especially $| \rho |$ and $R ^ { 2 } .$ . These metrics measure different qualities of layers, and so their divergence can highlight differing behaviors, especially when observed across different model depths.

Indeed, our the depth analysis indicates these different layer level patterns. A plausible explanation is that the decoder distributes feature information across many neurons in its middle layers, where joint recovery $( R ^ { 2 } )$ is strongest, and consolidates it into fewer dominant neurons closer to the output, where per-neuron correlations $( | \rho | )$ are strongest and joint diversity is reduced.

## D. Application to Other Architectures

RAVE [1] is a deliberate choice for this initial investigation rather than a limiting one. Its decoder is a relatively shallow stack of transposed convolutions and residual blocks operating on a low-dimensional latent making it an architecture simple enough to permit exhaustive per-layer analysis. However, it is also representative of the convolutional VAE family that remains widely deployed in real-time musical applications where latency constraints preclude larger autoregressive or diffusion-based alternatives. The questions of whether musical features occupy identifiable locations in a generative decoder and whether cross-layer functional groupings improve feature readout are architecture-agnostic in formulation, even if our specific answers are not.

We expect the methodological framework (permutationvalidated prevalence alongside encoding strength, cross-layer clustering, neuron specialization taxonomy) to transfer directly to other architectures, and the broad findings about how features are organized to plausibly generalize to other convolutional decoders which share RAVE’s upsampling-throughstrided-convolution backbone. We have shown this is the case for EnCodec [2] and the results with this general purpose decoder align with our strong results for OOD audio in RAVE models.

We are more cautious about generalization to architectures with fundamentally different inductive biases. Transformerbased audio models such as MusicGen [11] and AudioLM [12] replace local convolutional receptive fields with global attention. This would likely redistribute feature encoding across token positions rather than network depth. We see this contrast paralleled in vision research, where layer-wise interpretability findings for convolutional networks do not transfer cleanly to vision transformers [41].

Diffusion-based synthesizers ( [42], [43]) introduce an additional temporal axis (denoising steps) along which features may be organized, complicating direct comparison to a feed-forward decoder hierarchy. Autoregressive token models operate on discrete codebook entries rather than continuous activations, which changes what ”neuron-level encoding” even means in those settings [44]. Specific localization claims should therefore be read as findings about VAE-family decoders, while the methodology and the higher-level organizational principles offer hypotheses to be tested in these other settings rather than conclusions to be assumed of them.

## VI. LIMITATIONS

Our correlational analysis establishes association between layer activations and musical features, not functional importance. A neuron that correlates with pitch may actually be tracking a correlated acoustic property. For example, in an extended dissection of the MusiCNN tagger Sturm [45] examines a single network on individual signals across its convolutional stack. Identical musical excerpts drawn from different source recordings yield markedly different tag predictions, and predictions shift with input amplitude alone.

The drum model is excluded from pitch analysis and although not tonal nor melodic in the traditional sense, drums do contain spectral content (e.g., toms, cymbals). Future work could investigate whether pitch encoding of percussive timbres differs from tonal instruments. While permutation testing confirms results are well above chance (approx. 2–4× the null mean across features), the effect sizes observed for natural audio encoding are modest (pitch mean $| \rho | { = } 0 . 2 0$ , tempo mean $| \rho | { = } 0 . 2 0$ , spectral centroid mean $| \rho | { = } 0 . 3 1$ , spectral bandwidth mean $| \rho | { = } 0 . 2 7 )$ , corresponding to approximately 4–10% explained variance.

## VII. CONCLUSION

We have presented a systematic layer-wise and cross-layer cluster analysis of spectro-temporal feature encoding in RAVE audio synthesis models, examining three models trained on different musical domains across four stimulus types using three complementary metrics: encoding strength, permutationvalidated feature prevalence and joint-encoding strength with a nonlinear probe. We also conducted these tests on a supplementary model to test the generalization of results beyond one specific architecture.

We have found that synthetic stimuli are strongly encoded throughout the decoder layers and these results hold for natural audio and out-of-distribution audio and across temporal, pitched and spectral features. Synthetic stimuli are strongly linearly correlated whereas natural audio shows stronger results when measuring joint encoding across a whole layer with a nonlinear probe. Monotonic encoding $( | \rho | )$ is strongest at the early and late layers for three out of four features, whereas joint encoding is strongest in the middle layers for all features. Clustering across layers shows small but consistent advantage when finding neuron subsets that encode tempo, in comparison to whole layers.

Future work should pursue several directions: (1) causal validation through targeted ablation of the identified clusters and layers, which would determine whether the correlational patterns reported here reflect functional roles; (2) further crossarchitecture comparisons to determine whether these findings generalize beyond VAEs; (3) investigation of whether the identified encoding structure can inform practical applications such as targeted network bending, structured pruning, or the design of more interpretable neural audio architectures.

## ACKNOWLEDGMENTS

The authors would like to thank the developers and maintainers of the RAVE library.

## SUPPLEMENTARY MATERIALS

This paper has supplementary downloadable material available at https://ieeexplore.ieee.org/Xplore provided by the author. The material includes additional plots (Figs S1–S4) and data tables (Tables S1–S4). Contact l.mccallum@arts.ac.uk for further questions about this work.

## REFERENCES

[1] A. Caillon and P. Esling, “Rave: A variational autoencoder for fast and high-quality neural audio synthesis,” arXiv preprint arXiv:2111.05011, 2021.

[2] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” Transactions on Machine Learning Research, 2022.

[3] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2021.

[4] J. Engel, M. Hoffman, and A. Roberts, “Latent constraints: Learning to generate conditionally from unconditional generative models,” in International Conference on Learning Representations, 2018.

[5] C.-Z. A. Huang, A. Vaswani, J. Uszkoreit, N. Shazeer, C. Hawthorne, A. M. Dai, M. D. Hoffman, and D. Eck, “An improved relative selfattention mechanism for transformer with application to music generation,” arXiv preprint arXiv:1809.04281, vol. 2, 2018.

[6] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in European conference on computer vision. Springer, 2014, pp. 818–833.

[7] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449– 12 460, 2020.

[8] A. Pasad, C.-M. Chien, S. Settle, and K. Livescu, “What do selfsupervised speech models know about words?” Transactions of the Association for Computational Linguistics, vol. 12, pp. 372–391, 2024.

[9] A. Y. F. Chiu, K. C. Fung, R. T. Y. Li, J. Li, and T. Lee, “A large-scale probing analysis of speaker-specific attributes in self-supervised speech representations,” 2026. [Online]. Available: https://arxiv.org/abs/2501.05310

[10] P. Good, Permutation, parametric and bootstrap tests of hypotheses. Springer, 2005.

[11] J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Defossez, “Simple and controllable music generation,”´ Advances in neural information processing systems, vol. 36, pp. 47 704–47 720, 2023.

[12] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour, “Audiolm: a language modeling approach to audio generation,” 2023. [Online]. Available: https://arxiv.org/abs/2209.03143

[13] H. Liu, Z. Chen, Y. Yuan, X. Mei, X. Liu, D. Mandic, W. Wang, and M. D. Plumbley, “Audioldm: text-to-audio generation with latent diffusion models,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[14] G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” arXiv preprint arXiv:1610.01644, 2016.

[15] A. Pasad, C.-M. Chien, S. Settle, and K. Livescu, “What do self-supervised speech models know about words?” Transactions of the Association for Computational Linguistics, vol. 12, pp. 372–391, 04 2024. [Online]. Available: https://doi.org/10.1162/tacl a 00656

[16] J. Hewitt and P. Liang, “Designing and interpreting probes with control tasks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (emnlp-ijcnlp), 2019, pp. 2733–2743.

[17] Y. Zhang, P. Tino, A. Leonardis, and K. Tang, “A survey on neuralˇ network interpretability,” IEEE transactions on emerging topics in computational intelligence, vol. 5, no. 5, pp. 726–742, 2021.

[18] C. Olah, A. Mordvintsev, and L. Schubert, “Feature visualization,” Distill, 2017. [Online]. Available: https://distill.pub/ 2017/feature-visualization

[19] J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, and H. Lipson, “Understanding neural networks through deep visualization,” in ICML Deep Learning Workshop, 2015.

[20] D. Bau, B. Zhou, A. Khosla, A. Oliva, and A. Torralba, “Network dissection: Quantifying interpretability of deep visual representations,” in CVPR, 2017, pp. 6541–6549.

[21] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah, “Toy models of superposition,” Transformer Circuits Thread, 2022, https://transformer-circuits.pub/2022/toy model/index.html.

[22] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Muller, and¨ W. Samek, “On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation,” PloS one, vol. 10, no. 7, 2015.

[23] A. Shrikumar, P. Greenside, and A. Kundaje, “Learning important features through propagating activation differences,” in International conference on machine learning. PMlR, 2017, pp. 3145–3153.

[24] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in International conference on machine learning. PMLR, 2017, pp. 3319–3328.

[25] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[26] N. Cammarata, S. Carter, G. Goh, C. Olah, M. Petrov, L. Schubert, C. Voss, B. Egan, and S. K. Lim, “Thread: Circuits,” Distill, 2020. [Online]. Available: https://distill.pub/2020/circuits

[27] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. Mc-Candlish, and C. Olah, “In-context learning and induction heads,” Transformer Circuits Thread, 2022, https://transformer-circuits.pub/2022/incontext-learning-and-induction-heads/index.html.

[28] T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. Hatfield-Dodds, A. Tamkin,

K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah, “Towards monosemanticity: Decomposing language models with dictionary learning,” Transformer Circuits Thread, 2023, https://transformer-circuits.pub/2023/monosemantic-features/index.html.

[29] M. Raghu, J. Gilmer, J. Yosinski, and J. Sohl-Dickstein, “Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability,” Advances in neural information processing systems, vol. 30, 2017.

[30] A. Bauerle, D. J¨ onsson, and T. Ropinski, “Neural activation patterns¨ (naps): Visual explainability of learned concepts,” 2022. [Online]. Available: https://arxiv.org/abs/2206.10611

[31] Splice.com, “Splice sounds,” https://splice.com, 2024, royalty-free sample library. Samples used under Splice license.

[32] S. Rouard, F. Massa, and A. Defossez, “Hybrid transformers for music´ source separation,” in ICASSP 23, 2023.

[33] A. Caillon and P. Esling, “Streamable neural audio synthesis with noncausal convolutions,” arXiv preprint arXiv:2204.07064, 2022.

[34] J. H. McDermott and A. J. Oxenham, “Music perception, pitch, and the auditory system,” Current opinion in neurobiology, vol. 18, no. 4, pp. 452–463, 2008.

[35] G. Peeters, B. L. Giordano, P. Susini, N. Misdariis, and S. McAdams, “The timbre toolbox: Extracting audio descriptors from musical signals,” The Journal of the Acoustical Society of America, vol. 130, no. 5, pp. 2902–2916, 2011.

[36] A. Klapuri and M. Davy, Signal Processing Methods for Music Transcription, 1st ed. Springer Publishing Company, Incorporated, 2010.

[37] D. P. Ellis, “Beat tracking by dynamic programming,” Journal of New Music Research, vol. 36, no. 1, pp. 51–60, 2007.

[38] B. McFee, C. Raffel, D. Liang, D. P. Ellis, M. McVicar, E. Battenberg, O. Nieto et al., “librosa: Audio and music signal analysis in python.” SciPy, vol. 2015, no. 18-24, p. 7, 2015.

[39] M. Mauch and S. Dixon, “pyin: A fundamental frequency estimator using probabilistic threshold distributions,” in 2014 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, 2014, pp. 659–663.

[40] J. Hewitt and C. D. Manning, “A structural probe for finding syntax in word representations,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 4129–4138.

[41] M. Raghu, T. Unterthiner, S. Kornblith, C. Zhang, and A. Dosovitskiy, “Do vision transformers see like convolutional neural networks?” Advances in neural information processing systems, vol. 34, pp. 12 116– 12 128, 2021.

[42] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” arXiv preprint arXiv:2009.09761, 2020.

[43] J. Gong, S. Zhao, S. Wang, S. Xu, and J. Guo, “Ace-step: A step towards music generation foundation model,” 2025. [Online]. Available: https://arxiv.org/abs/2506.00045

[44] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.

[45] B. L. T. Sturm, “Digging into MusiCNN,” Blog series, “Folk the Algorithms”, 2022, 12-part series; https://highnoongmt.wordpress.com/ 2022/10/31/digging-into-musicnn-pt-1/.

## SUPPLEMENTARY MATERIALS

## A. Permutation Tests

TABLE S1  
PER-CELL ENCODING METRICS FOR ALL (MODEL, DATASET, FEATURE) COMBINATIONS. PER-NEURON STATISTICS FROM SPEARMAN CORRELATIONS; JOINT STATISTICS FROM MLP PROBES. NULL % > p VALUES ARE FROM 500 PERMUTATIONS OF SHUFFLED FEATURE LABELS. LINEAR $R ^ { 2 }$ FROM RIDGE REGRESSION PROBES; NONLINEAR $R ^ { 2 }$ FROM MLP PROBES; NL GAIN = NONLINEAR − LINEAR $R ^ { 2 }$ . BRACKETED VALUES ARE 95% BOOTSTRAP CONFIDENCE INTERVALS CALCULATED AT A LAYER LEVEL.
<table><tr><td colspan="3">Model</td><td colspan="3">Per-neuron |ρ|</td><td colspan="4">Joint probe R2</td></tr><tr><td></td><td>Dataset</td><td>Feature Null p95</td><td>Obs. mean [95% CI]</td><td>Obs. max</td><td>% &gt; p95</td><td>Null p95</td><td>Linear [95% CI]</td><td>Nonlinear [95% CI]</td><td>NL gain</td></tr><tr><td rowspan="10">Strings</td><td>Pitch BPM</td><td></td><td>0.087 0.221 [0.210, 0.233]</td><td>0.729</td><td>58.6</td><td>0.023</td><td>0.460 [0.436, 0.475] 0.377 [0.323, 0.418]</td><td>0.619 [0.584, 0.635]</td><td>0.159</td></tr><tr><td>Strings Spec C.</td><td>0.087 0.088</td><td>0.153 [0.148, 0.158] 0.367 [0.349, 0.383]</td><td>0.548 0.904</td><td>45.1</td><td>0.037</td><td></td><td>0.512 [0.480, 0.541]</td><td>0.134</td></tr><tr><td>Spec B.</td><td>0.088</td><td></td><td></td><td>77.4</td><td>0.038</td><td>0.817 [0.787, 0.838]</td><td>0.908 [0.893, 0.921]</td><td>0.091</td></tr><tr><td></td><td></td><td>0.256 [0.239, 0.277]</td><td>0.838</td><td>65.2</td><td>0.026</td><td>0.751 [0.706, 0.786]</td><td>0.836 [0.784, 0.866]</td><td>0.086</td></tr><tr><td></td><td>BPM</td><td>0.085 0.172 [0.166, 0.176]</td><td>0.431</td><td>64.5</td><td>0.027</td><td>0.246 [0.201, 0.277]</td><td>0.329 [0.286, 0.361]</td><td>0.083</td></tr><tr><td>Drums Spec C.</td><td>0.087</td><td>0.376 [0.366, 0.384]</td><td>0.776</td><td>87.1</td><td>0.034</td><td>0.410 [0.349, 0.435]</td><td>0.641 [0.624, 0.654]</td><td>0.230</td></tr><tr><td>Spec B.</td><td>0.086</td><td>0.517 [0.501, 0.531]</td><td>0.857</td><td>92.8</td><td>0.039</td><td>0.503 [0.479, 0.525]</td><td>0.681 [0.662, 0.696]</td><td>0.178</td></tr><tr><td>Pitch Stimuli</td><td>0.088</td><td>0.468 [0.442, 0.500]</td><td>0.980</td><td>81.8</td><td>0.027</td><td>0.993 [0.991, 0.995]</td><td>0.994 [0.992, 0.995]</td><td>0.001</td></tr><tr><td>BPM</td><td>0.092</td><td>0.945 [0.932, 0.953]</td><td>0.998</td><td>99.3</td><td>0.041</td><td>0.998 [0.997, 0.998]</td><td>0.994 [0.992, 0.994]</td><td>-0.004</td></tr><tr><td></td><td>0.088</td><td>0.134 [0.128, 0.141]</td><td>0.440</td><td>42.9</td><td>0.035</td><td>0.066 [0.058, 0.074]</td><td>0.125 [0.113, 0.140]</td><td>0.059</td></tr><tr><td rowspan="8">Vocals</td><td>Pitch BPM</td><td>0.089</td><td>0.177 [0.170, 0.185]</td><td>0.468</td><td>55.3</td><td>0.035</td><td>0.202 [0.171, 0.227]</td><td>0.341 [0.318, 0.361]</td><td>0.139</td></tr><tr><td>Spec C.</td><td>0.088</td><td>0.315 [0.308, 0.323]</td><td>0.774</td><td>83.4</td><td>0.040</td><td>0.454 [0.417, 0.480]</td><td>0.614 [0.590, 0.642]</td><td>0.160</td></tr><tr><td>Spec B.</td><td>0.087</td><td>0.190 [0.176, 0.206]</td><td>0.725</td><td>52.6</td><td>0.030</td><td>0.402 [0.375, 0.418]</td><td>0.570 [0.530, 0.615]</td><td>0.169</td></tr><tr><td>Pitch</td><td>0.087</td><td>0.264 [0.252, 0.276]</td><td>0.720</td><td>72.7</td><td>0.025</td><td>0.354 [0.333, 0.365]</td><td>0.522 [0.503, 0.533]</td><td>0.168</td></tr><tr><td>BPM Strings</td><td>0.087</td><td>0.219 [0.194, 0.239]</td><td>0.629</td><td>50.5</td><td>0.039</td><td>0.440 [0.422, 0.455]</td><td>0.447 [0.425, 0.463]</td><td>0.008</td></tr><tr><td>Spec C. Spec B.</td><td>0.088 0.089</td><td>0.464 [0.448, 0.486]</td><td>0.897</td><td>87.3</td><td>0.030</td><td>0.803 [0.791, 0.811]</td><td>0.847 [0.834, 0.858]</td><td>0.044</td></tr><tr><td></td><td></td><td>0.245 [0.233, 0.259]</td><td>0.751</td><td>74.8</td><td>0.040</td><td>0.570 [0.540, 0.594]</td><td>0.711 [0.681, 0.740]</td><td>0.142</td></tr><tr><td>Drums</td><td>BPM</td><td>0.088</td><td>0.187 [0.174, 0.197]</td><td>0.577</td><td>55.1</td><td>0.165 [0.112, 0.225]</td><td>0.536 [0.452, 0.580]</td><td>0.370</td></tr><tr><td rowspan="10">Drums</td><td>Spec C.</td><td>0.090</td><td>0.236 [0.221, 0.256]</td><td>0.816</td><td>61.1</td><td>0.030 0.031</td><td>0.487 [0.459, 0.516]</td><td>0.795 [0.770, 0.814]</td><td>0.309</td></tr><tr><td>Spec B.</td><td>0.086</td><td>0.241 [0.225, 0.264]</td><td>0.830</td><td>66.3</td><td>0.038</td><td>0.257 [0.237, 0.290]</td><td>0.740 [0.683, 0.780]</td><td>0.482</td></tr><tr><td>Pitch</td><td>0.086</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stimuli BPM</td><td>0.087</td><td>0.490 [0.472, 0.505] 0.621 [0.561, 0.674]</td><td>0.969 0.995</td><td>85.3 83.4</td><td>0.027 0.032</td><td>0.991 [0.987, 0.993] 0.997 [0.994, 0.998]</td><td>0.993 [0.990, 0.995] 0.991 [0.988, 0.994]</td><td>0.002 -0.005</td></tr><tr><td></td><td></td><td></td><td>0.411</td><td>60.7</td><td>0.033</td><td></td><td></td><td></td></tr><tr><td>Vocals</td><td>Pitch BPM</td><td>0.089 0.173 [0.166, 0.182] 0.087 0.294 [0.279, 0.310]</td><td>0.522</td><td></td><td>0.032</td><td>0.045 [0.038, 0.051] 0.182 [0.176, 0.190]</td><td>0.115 [0.104, 0.127] 0.289 [0.282, 0.297]</td><td>0.070 0.107</td></tr><tr><td>Spec C.</td><td>0.088</td><td>0.179 [0.167, 0.191]</td><td>0.744</td><td>82.3 45.7</td><td>0.035</td><td>0.547 [0.532, 0.564]</td><td>0.661 [0.641, 0.682]</td><td>0.114</td></tr><tr><td>Spec B.</td><td>0.088</td><td>0.245 [0.232, 0.258]</td><td>0.718</td><td>76.4</td><td>0.032</td><td>0.456 [0.435, 0.480]</td><td>0.589 [0.563, 0.616]</td><td>0.134</td></tr><tr><td>Pitch</td><td>0.088</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.243 [0.223, 0.262]</td><td>0.717</td><td>60.7</td><td>0.023</td><td>0.529 [0.510, 0.541]</td><td>0.688 [0.661, 0.701]</td><td>0.159</td></tr><tr><td rowspan="9">Vocals</td><td>BPM Strings</td><td>0.087</td><td>0.157 [0.153, 0.162]</td><td>0.531</td><td>48.8</td><td>0.032</td><td>0.439 [0.416, 0.460] 0.770 [0.762, 0.779]</td><td>0.541 [0.516, 0.564] 0.852 [0.846, 0.859]</td><td>0.102 0.082</td></tr><tr><td>Spec C.</td><td>0.088</td><td>0.396 [0.365, 0.426]</td><td>0.901</td><td>79.2</td><td>0.029</td><td>0.648 [0.618, 0.673]</td><td></td><td></td></tr><tr><td>Spec B.</td><td>0.089</td><td>0.294 [0.272, 0.318]</td><td>0.806</td><td>72.6</td><td>0.031</td><td></td><td>0.758 [0.742, 0.773]</td><td>0.110</td></tr><tr><td>BPM</td><td>0.087</td><td>0.207 [0.198, 0.215]</td><td>0.473</td><td>70.3</td><td>0.021</td><td>-0.022 [−0.051, 0.007]</td><td>0.241 [0.197, 0.288]</td><td>0.263</td></tr><tr><td>Drums Spec C.</td><td>0.091</td><td>0.244 [0.227, 0.269]</td><td>0.849</td><td>60.8</td><td>0.038</td><td>0.661 [0.649, 0.671]</td><td>0.814 [0.802, 0.820]</td><td>0.153</td></tr><tr><td>Spec B.</td><td>0.086</td><td>0.266 [0.243, 0.300]</td><td>0.864</td><td>62.0</td><td>0.023</td><td>0.195 [0.145, 0.232]</td><td>0.528 [0.482, 0.591]</td><td>0.333</td></tr><tr><td>Pitch Stimuli</td><td>0.089</td><td>0.389 [0.371, 0.411]</td><td>0.957</td><td>79.9</td><td>0.030</td><td>0.998 [0.997, 0.999]</td><td>0.998 [0.997, 0.999]</td><td>-0.000</td></tr><tr><td>BPM</td><td>0.087</td><td>0.728 [0.710, 0.744]</td><td>0.998</td><td>92.1</td><td>0.031</td><td>0.999 [0.999, 0.999]</td><td>0.998 [0.997, 0.998]</td><td>-0.001</td></tr><tr><td>Pitch</td><td>0.088</td><td>0.148 [0.138, 0.159]</td><td>0.514</td><td>42.5</td><td>0.037</td><td>0.172 [0.158, 0.182]</td><td>0.233 [0.212, 0.247]</td><td>0.061</td></tr><tr><td rowspan="3">Vocals</td><td>BPM</td><td>0.088</td><td>0.232 [0.218, 0.245]</td><td>0.574</td><td>65.9</td><td>0.027</td><td>0.412 [0.384, 0.440]</td><td>0.575 [0.526, 0.614]</td><td>0.162</td></tr><tr><td>Spec C.</td><td>0.088</td><td>0.172 [0.152, 0.194]</td><td>0.790</td><td>46.0</td><td>0.028</td><td>0.574 [0.569, 0.578]</td><td>0.695 [0.684, 0.702]</td><td>0.122</td></tr><tr><td>Spec B.</td><td>0.088</td><td>0.220 [0.205, 0.239]</td><td>0.801</td><td>66.3</td><td>0.024</td><td>0.480 [0.474, 0.485]</td><td>0.665 [0.656, 0.672]</td><td>0.185</td></tr><tr><td rowspan="9">EnCodec</td><td>Pitch BPM</td><td>0.088</td><td>0.310 [0.294, 0.325]</td><td>0.764 0.592</td><td>75.4 50.8</td><td>0.025 0.036</td><td>0.697 [0.652, 0.724] 0.520 [0.472, 0.555]</td></table>

## B. Comparison Tests

TABLE S2  
NATURAL VERSUS SYNTHETIC ENCODING ACROSS ARCHITECTURES. LAYER-MATCHED WILCOXON SIGNED-RANK TESTS (UNIT = LAYER PAIR, N PAIRS) FOR RAVE AND A PRETRAINED ENCODEC DECODER. NAT AND SYN GIVE MEDIANS [IQR]. ∆ (HL) IS THE HODGES–LEHMANN SYNTHETIC−NATURAL DIFFERENCE WITH 95% CI (POSITIVE = SYNTHETIC HIGHER); r IS RANK-BISERIAL. p<sub>ADJ</sub> IS THE BENJAMINI–HOCHBERG-ADJUSTED EXACT SIGN-PERMUTATION p WITHIN EACH MEASURE FAMILY. SPECTRAL FEATURES EXCLUDED (NO SYNTHETIC SPECTRAL ANALYSIS).
<table><tr><td>Arch.</td><td>Feature</td><td>Measure</td><td>N</td><td>Nat</td><td>Syn</td><td>∆ (HL) [95% CI]</td><td>r</td><td>ICC</td><td></td><td>padj</td></tr><tr><td rowspan="5">RAVE</td><td>Pitch</td><td>Mean |ρ|</td><td>84</td><td>0.199</td><td>0.436</td><td>+0.248 [+0.232, +0.263]</td><td>+1.00</td><td>0.450</td><td>&lt;.001</td><td>***</td></tr><tr><td></td><td> $\% > \mathrm { p } 9 5$ </td><td>84</td><td>74.6</td><td>90.1</td><td>+15.76 [+14.35, +17.19]</td><td>+1.00</td><td>0.507</td><td>&lt;.001</td><td>***</td></tr><tr><td></td><td> $R ^ { 2 }$ </td><td>84</td><td>0.365</td><td>0.997</td><td>+0.612 [+0.594, +0.630]</td><td>+1.00</td><td>0.765</td><td>&lt;.001</td><td>***</td></tr><tr><td rowspan="2">BPM</td><td>Mean |ρ|</td><td>84</td><td>0.197</td><td>0.752</td><td>+0.562 [+0.524, +0.625]</td><td>+1.00</td><td>0.835</td><td>&lt;.001</td><td>***</td></tr><tr><td> $\% > \mathrm { p } 9 5$ </td><td>84</td><td>77.8</td><td>97.4</td><td>+18.27 [+16.67, +19.88]</td><td>+1.00</td><td>0.754</td><td>&lt;.001</td><td>***</td></tr><tr><td rowspan="5">EnCodec</td><td rowspan="4">Pitch</td><td> $R ^ { 2 }$ </td><td>84</td><td>0.432</td><td>0.996</td><td>+0.566 [+0.551, +0.580]</td><td>+1.00</td><td>0.097</td><td></td><td>&lt;.001 ***</td></tr><tr><td>Mean |ρ|</td><td>13</td><td>0.244</td><td>0.555</td><td>+0.319 [+0.256, +0.381]</td><td>+1.00</td><td></td><td>&lt;.001</td><td>***</td></tr><tr><td> $\% > \mathrm { p } 9 5$ </td><td>13</td><td>81.6</td><td>91.0</td><td>+12.11 [+9.38, +14.65]</td><td>+1.00</td><td></td><td></td><td>&lt;.001 ***</td></tr><tr><td> $R ^ { 2 }$ </td><td>13</td><td>0.478</td><td>0.994</td><td>+0.537 [+0.499, +0.582]</td><td>+1.00</td><td></td><td></td><td>&lt;.001 ***</td></tr><tr><td rowspan="3">BPM</td><td>Mean |ρ|</td><td>13</td><td>0.185</td><td>0.649</td><td>+0.470 [+0.401, +0.541]</td><td>+1.00</td><td></td><td>&lt;.001</td><td>***</td></tr><tr><td></td><td> $\% > \mathrm { p } 9 5$  13</td><td>75.5</td><td>96.5</td><td>+19.12 [+14.91, +23.77]</td><td></td><td>+1.00</td><td></td><td></td><td>&lt;.001 ***</td></tr><tr><td> $R ^ { 2 }$ </td><td></td><td>13</td><td>0.647</td><td>0.998</td><td>+0.388 [+0.322, +0.484]</td><td>+1.00</td><td></td><td></td><td>&lt;.001 ***</td></tr></table>

TABLE S3

DEPTH CURVATURE, ENCODEC (DESCRIPTIVE). MEDIAN PER-CELL QUADRATIC COEFFICIENT $\beta _ { 2 }$ OVER NORMALIZED DEPTH (POSITIVE = U-SHAPED, NEGATIVE = INVERTED-U) WITH [IQR] AND RANK-BISERIAL r. AS ENCODEC IS A SINGLE MODEL, EACH FEATURE CONTRIBUTES ONLY N=2–3 CELLS; THE ACROSS-CELL WILCOXON CANNOT REACH SIGNIFICANCE (p ≥ 0.25 THROUGHOUT), SO RESULTS ARE DESCRIPTIVE.
<table><tr><td>Measure</td><td>Feature</td><td> $N$ </td><td> $\beta _ { 2 }$  [IQR]</td><td>r</td></tr><tr><td rowspan="4">Mean |ρ|</td><td>Pitch</td><td>2</td><td>-0.14 [-0.16, -0.12]</td><td>-1.00</td></tr><tr><td>BPM</td><td>3</td><td>0.16 [0.08, 0.21]</td><td>0.67</td></tr><tr><td>Spectral Centroid</td><td>3</td><td>0.29 [0.03, 0.30]</td><td>0.67</td></tr><tr><td>Spectral Bandwidth</td><td>3</td><td>0.15 [0.10, 0.20]</td><td>1.00</td></tr><tr><td rowspan="4"> $\% > \mathrm { p } 9 5$ </td><td>Pitch</td><td>2</td><td>-26.59 [-33.70, -19.49]</td><td>-1.00</td></tr><tr><td>BPM</td><td>3</td><td>19.72 [6.30, 40.94]</td><td>0.67</td></tr><tr><td>Spectral Centroid</td><td>3</td><td>10.28 [-3.09, 31.14]</td><td>0.33</td></tr><tr><td>Spectral Bandwidth</td><td>3</td><td>0.60 [-9.86, 23.86]</td><td>0.33</td></tr><tr><td rowspan="4"> $R ^ { 2 }$ </td><td>Pitch</td><td>2</td><td>-0.48 [-0.56, -0.39]</td><td>-1.00</td></tr><tr><td>BPM</td><td>3</td><td>-0.78 [-0.85, -0.59]</td><td>-1.00</td></tr><tr><td>Spectral Centroid</td><td>3</td><td>-0.31 [-0.42, -0.29]</td><td>-1.00</td></tr><tr><td>Spectral Bandwidth</td><td>3</td><td>-0.60 [-0.62, -0.39]</td><td>-1.00</td></tr></table>

Hierarchical organisation: per-cell polynomial fits, aggregated across cells (solid = linear: dashed = quadratic: error bars = SEM across cells: lines anchored at data centroid)  
![](images/b3d6007f9ed5fb3f348100fa110a60bb9116ebabbb028591218dcd93b90b57ea.jpg)

![](images/c5910bf367ea6dcf82bf765b7646c2baaa09ae4c2848ce03533e9f38f846f4ec.jpg)

![](images/2949ce01f88a7cc781645017abb3ed3479e34f8a0d66613ce516184b063e14d1.jpg)

![](images/ed8edbb4a8cfada2bff09547ca63d3a5a512a7dedcb15363f76f1eefca1533d9.jpg)  
Fig. S1. Plots of per layer mean $| \rho | ,$ neuron prevalence and $R ^ { 2 }$ across four audio features for natural audio with the EnCodec model. Only per layer mean |ρ| displayed for synthetic audio as other features are near ceiling throughout (e.g. flat). Significance tests not possible as N=3 for natural and N=1 for synthetic.

TABLE S4  
CROSS-LAYER CLUSTER VERSUS BEST SINGLE LAYER, NATURAL AUDIO ONLY. HODGES–LEHMANN ESTIMATE OF THE MEDIAN PER-CELL DIFFERENCE (BEST CROSS-LAYER CLUSTER − BEST SINGLE LAYER WITHIN A SECTION) WITH 95% BOOTSTRAP CI, FOR RAVE AND A PRETRAINED ENCODEC DECODER; POSITIVE VALUES INDICATE THE CROSS-LAYER CLUSTER OUTPERFORMS THE BEST SINGLE LAYER. LAYER/CLUSTER MEAN ARE THE RAW (NON-PAIRED) MEANS; BEST CLUSTER/LAYER n ARE THE AVERAGE NUMBER OF UNITS IN THE SELECTED CLUSTER VS. THE BEST SINGLE LAYER. r IS THE MATCHED-PAIRS RANK-BISERIAL EFFECT SIZE; p IS THE BENJAMINI–HOCHBERG-ADJUSTED EXACT SIGN-PERMUTATION p WITHIN EACH MEASURE FAMILY. BETWEEN-MODEL (MOD) AND BETWEEN-DATASET (DAT) INTRACLASS CORRELATIONS INDEX DEPENDENCE AMONG CELLS. ENCODEC IS A SINGLE MODEL, SO BETWEEN-MODEL ICC IS UNDEFINED (–).
<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>ICC</td><td></td><td></td><td>Avg. best n</td><td></td></tr><tr><td>Arch.</td><td>Meas.</td><td>Feat.</td><td></td><td>N Layer</td><td>Cluster</td><td></td><td>HL [95% CI]</td><td>r</td><td>mod</td><td>dat</td><td>padj</td><td>cluster</td><td>layer</td></tr><tr><td rowspan="10">RAVE</td><td rowspan="4">Mean |ρ|</td><td>Pitch</td><td>18</td><td>0.239</td><td>0.237</td><td>-0.004 [−0.015, +0.012]</td><td></td><td>-0.05</td><td>0.154 0.339</td><td></td><td>0.769</td><td>779</td><td>480</td></tr><tr><td>BPM</td><td>27</td><td>0.233</td><td>0.257</td><td></td><td>+0.019 [+0.007, +0.033]</td><td>+0.65</td><td>0.000</td><td>0.411</td><td>0.006**</td><td>360</td><td>309</td></tr><tr><td>Spec. Centroid</td><td>27</td><td>0.377</td><td>0.362</td><td></td><td>-0.018 [−0.036, +0.009]</td><td>-0.29</td><td>0.268</td><td>0.008</td><td>0.400</td><td>1102</td><td>513</td></tr><tr><td>Spec. Bandwidth</td><td>27</td><td>0.348</td><td>0.335</td><td>-0.022 [−0.044, +0.004]</td><td></td><td>-0.34</td><td>0.000</td><td>0.000</td><td>0.605</td><td>831</td><td>466</td></tr><tr><td rowspan="4">% &gt; p95</td><td>Pitch</td><td>18</td><td>82.6</td><td>86.9</td><td></td><td>+3.91 [+0.13, +8.37]</td><td>+0.50</td><td>0.183</td><td>0.153</td><td>0.061</td><td>514</td><td>420</td></tr><tr><td>BPM</td><td>27</td><td>84.9</td><td>91.0</td><td>+6.15 [+3.23, +9.07]</td><td>+0.75</td><td>0.000</td><td>0.233</td><td></td><td>0.001**</td><td>486</td><td>309</td></tr><tr><td>Spec. Centroid</td><td>27</td><td>88.9</td><td>88.0</td><td>-0.59 [-3.17, +1.53]</td><td></td><td>-0.13 0.137</td><td></td><td>0.142</td><td>0.521</td><td>971</td><td>442</td></tr><tr><td>Spec. Bandwidth</td><td>27</td><td>89.3</td><td>89.8</td><td>−0.02 [-2.45, +3.18]</td><td></td><td>-0.01</td><td>0.014</td><td>0.083</td><td>0.686</td><td>738</td><td>423</td></tr><tr><td rowspan="4">R2</td><td>Pitch</td><td></td><td>18 0.412</td><td></td><td>0.418</td><td>+0.005 [+0.001, +0.010]</td><td>+0.56</td><td>0.000</td><td>0.000</td><td>0.083</td><td>738</td><td>352</td></tr><tr><td>BPM</td><td></td><td>27 0.490</td><td>0.500</td><td></td><td>+0.011 [−0.009, +0.036]</td><td>+0.17</td><td>0.000</td><td>0.000</td><td>0.860</td><td>953</td><td>338</td></tr><tr><td>Spec. Centroid</td><td>27</td><td>0.788</td><td>0.789</td><td>+0.001 [-0.008, +0.009]</td><td></td><td>+0.02</td><td>0.000</td><td>0.004</td><td>0.917</td><td>825</td><td>359</td></tr><tr><td>Spec. Bandwidth</td><td>27</td><td>0.744</td><td>0.733</td><td>-0.004 [-0.015, +0.012]</td><td></td><td>-0.15</td><td>0.002</td><td>0.093</td><td>0.860</td><td>698</td><td>352</td></tr><tr><td rowspan="8">EnCodec</td><td rowspan="4">Mean |ρ|</td><td>Pitch</td><td></td><td>6 0.273</td><td>0.355</td><td>+0.066 [+0.018, +0.156]</td><td>+1.00</td><td></td><td>0.556</td><td></td><td>0.125</td><td>55</td><td>235</td></tr><tr><td>BPM</td><td></td><td>9 0.228</td><td>0.227</td><td>-0.002 [-0.016, +0.014]</td><td>-0.11</td><td></td><td>0.000</td><td></td><td>0.840</td><td>396</td><td>331</td></tr><tr><td>Spec. Centroid</td><td></td><td>9 0.369</td><td>0.436</td><td>+0.046 [−0.002, +0.132]</td><td>+0.60</td><td></td><td>0.667</td><td></td><td>0.164</td><td>122</td><td>288</td></tr><tr><td>Spec. Bandwidth</td><td>9</td><td>0.299</td><td>0.336</td><td></td><td>+0.035 [-0.012, +0.079]</td><td>+0.51</td><td></td><td>0.070</td><td>0.167</td><td>259</td><td>188</td></tr><tr><td rowspan="4">% &gt; p95</td><td>Pitch</td><td>6</td><td>86.9</td><td>92.6</td><td></td><td>+5.66 [+3.66, +7.62]</td><td>+1.00</td><td></td><td>0.000</td><td>0.042*</td><td>55</td><td>219</td></tr><tr><td>BPM</td><td>9</td><td>83.9</td><td>84.5</td><td>+0.18 [-4.37, +6.55]</td><td></td><td>+0.06</td><td></td><td>0.139</td><td>0.859</td><td>207</td><td>295</td></tr><tr><td>Spec. Centroid</td><td>9</td><td>87.3</td><td>90.5</td><td>+3.28 [+0.96, +5.60]</td><td></td><td>+0.82</td><td></td><td>0.290</td><td>0.042*</td><td>79</td><td>228</td></tr><tr><td>Spec. Bandwidth</td><td>9</td><td>86.3</td><td>91.4</td><td>+5.69 [+0.71, +8.77]</td><td></td><td>+0.82</td><td></td><td>0.000</td><td>0.042*</td><td>211</td><td>203</td></tr><tr><td rowspan="4">R²</td><td>Pitch</td><td></td><td>6 0.479</td><td></td><td>0.480</td><td>-0.002 [-0.019, +0.024]</td><td>-0.14</td><td></td><td>0.138</td><td>0.938</td><td>243</td><td>277</td></tr><tr><td>BPM</td><td></td><td>9 0.615</td><td></td><td>0.629</td><td>+0.012 [−0.008, +0.038]</td><td>+0.33</td><td></td><td>0.000</td><td>0.344</td><td>309</td><td>277</td></tr><tr><td>Spec. Centroid</td><td></td><td>9 0.868</td><td>0.851</td><td></td><td>-0.014 [-0.050, +0.021]</td><td>-0.51</td><td></td><td>0.664</td><td>0.344</td><td>365</td><td>277</td></tr><tr><td>Spec. Bandwidth</td><td></td><td>9 0.843</td><td>0.777</td><td></td><td>-0.065 [-0.117, -0.014]</td><td>-0.73</td><td></td><td>0.623</td><td>0.125</td><td>378</td><td>277</td></tr></table>

![](images/0ca3ee121868b507216add06ec46c6dbe758bc9d2731f606cd896e47b357241c.jpg)

![](images/b929c69812f8b3e029413384019b16d046ea6dc426566a8b9933eac6124c6db4.jpg)

![](images/bdad37571783c10f46b5c0600961321aa7bed6223cf2ab91ba9a0271c08f7e38.jpg)  
Fig. S2. Plots of mean silhouette score, cluster imbalance and mean and max feature correlation across values of k.  
Average size of ranked clusters across k (mean ± SÉM across 12 model × dataset combos)

![](images/3c72c652609d2c5b115105b74b31134bf8a2a977332d17ba9708dad63afa5938.jpg)

![](images/d0fc34e51dbc1c8388b259143da0eedbf0842551c802f57e87604283935008e1.jpg)

![](images/3153b4f8ff4d02d553f2cb4b38fc11e80c1bcb1e6c0ad078369fbcb7669fc908.jpg)

Fig. S3. Plot of mean cluster size (ranked by size) across values of k.  
![](images/68d0efd2fba1187446c1b4b2be8cddba726a01bcfc423ec0ff089b46f61b8e34.jpg)

![](images/5074a8e8623f2e5d313aec78e4ac4e62ee71b194331fc43c074536eccaa8e86b.jpg)  
Fig. S4. Plot of mean ARI for all clusters, and mean Jaccard for best cluster across values of k.