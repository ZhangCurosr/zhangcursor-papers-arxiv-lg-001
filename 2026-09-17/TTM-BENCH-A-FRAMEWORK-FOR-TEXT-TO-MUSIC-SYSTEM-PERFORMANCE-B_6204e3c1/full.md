# TTM-BENCH: A FRAMEWORK FOR TEXT-TO-MUSIC SYSTEM PERFORMANCE BENCHMARKING

Giorgia Adorni<sup>1</sup> Michela Papandrea<sup>1</sup> Battista Rimoldi<sup>2</sup> Tiziano Leidi<sup>1</sup>

<sup>1</sup>Institute of Information Systems and Networking (ISIN), SUPSI, Lugano, Switzerland <sup>2</sup>t2b AG, Basel, Switzerland

## ABSTRACT

Text-to-music (TTM) systems are increasingly used to generate musical audio from natural-language descriptions. Robust evaluation is therefore essential, yet reliable performance comparison remains challenging. This difficulty stems from differences in system architecture, supported conditioning information, and access mode, as well as heterogeneous and fragmented metrics that cannot be applied uniformly across systems. To address these challenges, we introduce TTM-Bench, a framework that defines a common protocol for systematic, reproducible performance benchmarking of contemporary TTM systems. It evaluates performance along two dimensions: musical-content alignment, quantified by interpretable semantic, genre, and musical-descriptor agreement scores against a common musical specification and summarized by an aggregate score; and computational efficiency, characterized by generation latency and real-time factor, alongside resource use for local models and cost for hosted services. We demonstrate the framework through a preliminary comparative case study, illustrating the complementary evidence captured by these dimensions. The results show that higher musical-content alignment does not systematically coincide with lower computational demands, highlighting the importance of assessing TTM performance through distinct, interpretable measures rather than a reductive overall indicator.

Index Terms— text-to-music generation, benchmarking framework, musical-content alignment, computational efficiency, automatic evaluation

## 1. INTRODUCTION

Text-to-music (TTM) generation has emerged as a prominent research direction in generative audio, allowing systems to synthesize music from textual descriptions and other musicrelated conditioning information. Recent advances in this field have led to TTM systems with substantially different conditioning formats, ranging from free-form prompts to structured musical attributes. This heterogeneity poses a comparability problem: the same prompt may not express the intended musical content equivalently across systems, whereas system-specific reformulation may alter the underlying specification. A benchmarking protocol must therefore preserve a shared musical specification while expressing it in a form compatible with each system. Beyond conditioning, TTM systems also differ in architecture, supported output duration, and execution mode. These differences affect how computational efficiency can be measured and compared: local systems allow direct measurement of hardware-level resource use, whereas hosted APIs allow only service-level observations such as latency and cost. A further source of heterogeneity concerns evaluation. Existing automatic metrics target different performance dimensions and therefore capture separate aspects of system behavior. Contrastive Language-Audio Pretraining (CLAP)-based similarity measures semantic correspondence between a text prompt and generated audio [1], whereas Frechet Audio Distance (FAD)´ compares generated and reference audio distributions and therefore depends on the choice of reference corpus [2]. As a result, metric selection depends on the benchmarking objective and affects how system performance is characterized. To the best of our knowledge, available benchmarks address specific aspects of TTM evaluation under relatively controlled settings. However, they do not directly support systematic comparison of existing systems that differ in conditioning formats, architectures, and access modes.

In this work, we make two main contributions. First, we introduce TTM-Bench, a framework for benchmarking heterogeneous TTM systems across two distinct dimensions. It establishes a common protocol that preserves the underlying musical specification while adapting its representation to each system and explicitly separates musical-content alignment from access-aware computational efficiency. TTM-Bench integrates complementary evidence within two workflows. The musical-content alignment workflow evaluates how closely each generated track matches the intended musical content, using semantic correspondence with the conditioning text and genre and musical-descriptor similarity to the reference audio. It retains interpretable component scores alongside an aggregate alignment score. The computational efficiency workflow characterizes generation time, resource use, and cost within the observable boundaries of local and hosted execution. Second, we illustrate the framework through a preliminary case study involving 13 open and commercial TTM systems.

## 2. RELATED WORK

Research on TTM performance benchmarking spans four areas: automatic metrics, learned perceptual evaluators, computational efficiency studies, and cross-system benchmarks.

Automatic metrics quantify properties such as semantic correspondence with the conditioning text, differences between generated and reference audio distributions in learned embedding spaces, and audio-derived musical characteristics [1–4]. These measures capture different aspects of generated audio, and their results depend on methodological choices such as the embedding model, reference data, preprocessing, sample size, and aggregation [5].

Learned perceptual evaluators, such as MusicEval, MuQ-Eval, SongBench, AudioEval, and TuneJury, use humanannotated data to estimate perceived text-audio alignment, listener preference, and musical quality across aspects including vocals, composition, arrangement, and production [6–10]. These approaches complement automatic metrics by capturing perceptual aspects that are difficult to measure directly, but their reliability depends on the quality, coverage, and potential biases of the human judgments used for training.

Computational efficiency studies mainly characterize generation speed through measures such as inference latency and real-time factor [11–13]. These results are often reported under different hardware and experimental settings, limiting direct comparison, while other relevant factors such as memory use, energy consumption, and hosted-service cost are less consistently considered.

Cross-system benchmarks address different sources of evaluation variability. Grotschla et al. relate model and met-¨ ric outputs to human preferences, Music Arena facilitates live pairwise preference evaluation with LLM-based routing across systems with different conditioning formats, and the ATTM Grand Challenge standardizes training data and constraints in a fair-play setting [14–16]. These approaches emphasize preference-based or controlled evaluation, leaving already-available systems with heterogeneous conditioning and access modes less systematically addressed.

TTM-Bench instead targets already-available systems with heterogeneous conditioning formats and access modes by evaluating musical-content alignment at the component level and computational efficiency using the measurements observable under each access mode.

## 3. METHOD

The benchmarking procedure is organized into two workflows, described below. We will make the code, descriptor manifests, system-specific prompts, derived metrics, and analysis notebooks publicly available on Zenodo.

![](images/f274de3df6496bf506c49a95789ecc3b047c79829202bd1b32d2657f4d4ec9d1.jpg)  
Fig. 1. Musical-content alignment benchmarking workflow.

The musical-content alignment workflow (Fig. 1) assesses how well generated tracks match a common referencederived musical specification, expressed in the conditioning format supported by each system. We characterize alignment along three dimensions. (1) Semantic Alignment (SA) measures semantic similarity between the conditioning text and generated audio using LAION-CLAP cosine similarity [1]. (2) Genre Alignment (GA) combines linearly genre-set overlap (20%) and classifier-score similarity (80%). We estimate genre information using three pretrained classifiers with different taxonomies: Discogs519 [17], Discogs400 [18], and MTG-Jamendo [19]. The first two predict 519 and 400 styles from the Discogs taxonomy, respectively, while MTG-Jamendo uses a separate 87-genre taxonomy. Using multiple classifiers reduces dependence on a single genre representation. We compute genre-set overlap as the Jaccard similarity between high-level genre sets predicted for the reference and generated audio by the two Discogs classifiers. We calculate classifier-score similarity by combining cosine similarities between the corresponding genre-score vectors from all three classifiers weighted at 50%, 25%, and 25%, respectively. (3) Descriptor Alignment (DA) combines timbre-category (50%), BPM (40%), and tempo-label (10%) similarity, based respectively on a fixed distance matrix over warm, bright, and harsh, relative BPM difference, and normalized ordinal distance from slow to fast. These measures evaluate correspondence at the individual-generation level. We do not include distribution-level metrics such as FAD, as they compare audio sets rather than individual generations with their intended musical specification. We calculate the Musical-Content Alignment Score (MCAS) as a linear combination of the three components. It is not intended as an overall measure of TTM systems’ performance.

$$
\mathrm { M C A S } = 0 . 4 0 \cdot \mathrm { S A } + 0 . 3 5 \cdot \mathrm { G A } + 0 . 2 5 \cdot \mathrm { D A } .\tag{1}
$$

We fix the weights empirically, assigning greater weight to SA for overall conditioning correspondence than to the narrower musical properties captured by GA and DA. We assess MCAS weight sensitivity using equal weighting, local perturbations, and leave-one-component-out configurations, and measure agreement with the proposed weighting through Spearman’s $\rho$ and Kendall’s τ . We provide implementation details and normalization functions with the released code.

The computational efficiency workflow (Fig. 2) characterizes generation time, resource use, and cost within the measurement boundaries of local execution and hosted services. For local execution, latency covers model-side conditioning through construction of the usable waveform, with CUDA synchronization used for timing. For hosted services, it covers request submission through audio reception and decoding and may therefore include provider-side generation, queuing, network transfer, polling, download, and local postprocessing. When a system cannot natively generate the target duration, efficiency is measured for the computation required to obtain the standardized 30s output. For both access types, real-time factor (RTF) is computed as measured latency divided by usable output duration. The reported measures reflect what is observable under each access mode. Local execution provides latency, RTF, memory use, GPU utilization, and GPU energy consumption, whereas hosted services provide client-observed latency, RTF, and provider-listed API cost. We do not estimate unavailable server-side resources.

![](images/627e32a472599bc47c0199cffab7da1d9a7f6603a44bf5082667b9fcaf6a0b4f.jpg)  
Fig. 2. Computational efficiency benchmarking workflow.

To validate the proposed TTM-Bench, we conduct a preliminary case study and analyze the results. We restrict the evaluation to text-conditioned instrumental generation by using HTDemucs to extract instrumental stems from both reference and generated audio, reducing the influence of differences in vocal or lyric generation on the comparison [20]. We evaluate 13 TTM systems spanning different conditioning formats, access modes, and architectural families, including autoregressive, diffusion, flow-matching, and hybrid approaches. The set includes 9 open-source models: ACE-Step v1 3.5B [13], AudioLDM2 Music [21], DiffRhythm [12], HeartMuLa OSS 3B [22], InspireMusic 1.5B Long [23], MusicGen Large [24], MusicLDM [25], Stable Audio Open 1.0 [26], and YuE [27]; and 4 commercial APIs: Lyria 2 [28] and Lyria 3 [29], accessed via fal.ai, Stable Audio 2.5 [30], and Stable Audio 3 [31]. The reference collection comprises 100 commercially released tracks obtained from legitimate music sources. We curated the collection to span multiple decades, genres, and tempo ranges, providing heterogeneous case-study conditions rather than statistical representativeness. We release only the reference metadata and derived descriptors, not the original reference audio files. Our extraction pipeline (Fig. 3) derives 46 attributes per track from metadata, audio analysis, and controlled descriptor enrichment, including established music information retrieval (MIR) tools and models [32, 33]. For each reference-system pair (100 reference tracks × 13 TTM systems), we restrict the common musical specification to the attributes supported by that system. We use Llama 3.1 8B [34] to generate 10 prompt variants in the required input format, ranging from commaseparated attribute lists to natural-language descriptions of varying detail. We then validate each to ensure it preserves the selected attribute values without introducing unsupported musical information. For each pair, we retain the 5 prompt variants with the highest LAION-CLAP similarity to the in strumental audio reference [1], yielding 100 × 13 × 5 = 6500 prompts. Finally, we generate 4 30s tracks from each retained prompt, resulting in 6500 × 4 = 26000 tracks. We compute SA, GA, DA, and MCAS for each, average the score first within each reference-system pair, and then across references to obtain system-level scores. We assess robustness to reference-set composition through paired reference-level bootstrap resampling with 10000 iterations. At each iteration, we sample 100 references with replacement using the same reference indices for all systems, and recompute system-level scores and pairwise contrasts. We assess the complementarity of SA, GA, and DA using Pearson and Spearman correlations at both generation and system levels. We interpret these correlations descriptively because outputs within referencesystem pairs are not independent and only 13 systems are evaluated. We evaluate computational efficiency after one unmeasured warm-up using 10 distinct inputs per system, targeting 30s outputs. For local runs, we use an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB); we document hardware, software, and system-specific inference configurations in the Zenodo benchmark manifest. We sample resource measurements at approximately 10Hz during generation, excluding the time required to save the final audio. We define peak RAM as the maximum memory used by the generation process and its child processes, and peak VRAM as the maximum GPU memory reported by nvidia-smi. We average GPU utilization across samples and estimate GPU-board energy by integrating the reported power draw over time. We do not measure CPU or whole-system energy.

![](images/a30c795faf77c6ff377e9db07c7cd3978ef7530c430c9fa6d17506065cf0699c.jpg)  
Fig. 3. Descriptor-extraction pipeline.

## 4. RESULTS

Table 1 summarizes SA, GA, DA, and MCAS for the evaluated systems. Across the tested alternatives, agreement with the proposed weighting remains high (median/minimum Spearman $\rho ~ = ~ 0 . 9 7 8 / 0 . 9 2 9 ;$ Kendall $\tau ~ = ~ 0 . 9 2 3 / 0 . 8 2 1 \rangle$ .

This supports the adopted formulation as a reasonable choice, with relative system-level MCAS results remaining stable under the tested weighting alternatives. Commercial systems outperform open models across all musical-content alignment measures. Within the former group, Stable Audio 2.5 leads in SA and GA, while Lyria 3 records the highest DA; within the latter, Stable Audio Open 1.0 leads in SA and DA, with MusicGen Large obtaining the highest GA. These differences indicate that relative performance varies across the individual alignment components. For the three smallest pairwise MCAS contrasts, the 95% bootstrap intervals for the pairwise difference between Stable Audio 2.5 and Lyria 3 ([−0.012, 0.013]), Lyria 3 and Stable Audio 3 ([−0.008, 0.014]), and MusicGen Large and Stable Audio Open 1.0 ([−0.016, 0.020]) all include zero. Thus, small differences in MCAS are not robust to changes in reference-set composition. At generation level, SA, GA, and DA show weak positive pairwise correlations (Pearson $r ~ = ~ 0 . 2 1 7 \cdot$ 0.281; Spearman $\rho = 0 . 2 2 2 { - 0 . 2 7 0 } )$ , indicating limited overlap among the information captured by the three components. Correlations among the 13 system-level means are higher (Pearson $r ~ = ~ 0 . 7 6 7 – 0 . 8 7 9 ;$ Spearman $\rho ~ = ~ 0 . 6 8 7 – 0 . 8 9 6 )$ showing greater agreement after aggregation at system-level. Together, these findings support retaining the individual alignment components alongside the aggregate MCAS.

Table 1. Musical-content alignment case study results.  
(a) Open systems
<table><tr><td>System</td><td>SA</td><td>GA</td><td>DA</td><td>MCAS</td></tr><tr><td>ACE-Step v1 3.5B</td><td>0.337</td><td>0.558</td><td>0.749</td><td>0.517</td></tr><tr><td>AudioLDM2 Music</td><td>0.229</td><td>0.577</td><td>0.734</td><td>0.477</td></tr><tr><td>DiffRhythm</td><td>0.358</td><td>0.547</td><td>0.767</td><td>0.526</td></tr><tr><td>HeartMuLa OSS 3B</td><td>0.290</td><td>0.538</td><td>0.753</td><td>0.493</td></tr><tr><td>InspireMusic 1.5B Long</td><td>0.239</td><td>0.441</td><td>0.626</td><td>0.406</td></tr><tr><td>MusicGen Large</td><td>0.288</td><td>0.645</td><td>0.802</td><td>0.541</td></tr><tr><td>MusicLDM</td><td>0.287</td><td>0.583</td><td>0.764</td><td>0.510</td></tr><tr><td>Stable Audio Open 1.0</td><td>0.365</td><td>0.536</td><td>0.823</td><td>0.539</td></tr><tr><td>YuE</td><td>0.110</td><td>0.466</td><td>0.683</td><td>0.378</td></tr></table>

(b) Commercial systems
<table><tr><td>System</td><td>SA</td><td>GA</td><td>DA</td><td>MCAS</td></tr><tr><td>Lyria 2 (via fal.ai)</td><td>0.380</td><td>0.647</td><td>0.827</td><td>0.585</td></tr><tr><td>Lyria 3 (via fal.ai)</td><td>0.460</td><td>0.665</td><td>0.868</td><td>0.634</td></tr><tr><td>Stable Audio 2.5</td><td>0.476</td><td>0.676</td><td>0.829</td><td>0.634</td></tr><tr><td>Stable Audio 3</td><td>0.476</td><td>0.662</td><td>0.834</td><td>0.631</td></tr></table>

Table 2 summarizes computational efficiency across the evaluated systems. Among open-source models, ACE-Step has the lowest latency, RTF, and energy consumption, while MusicGen Large uses the least peak RAM and MusicLDM the least peak VRAM. Thus, no single system performs best on all efficiency measures. Among commercial APIs, Stable Audio 2.5 has the lowest observed latency and RTF, whereas Lyria 3 has the lowest API cost. This shows that faster generation does not necessarily correspond to lower cost.

Table 2. Computational efficiency case study results. Latency is reported as mean ± SD in seconds, RAM/VRAM in GB, GPU utilization in %, GPU energy in Wh, and Cost in USD.  
(a) Open systems
<table><tr><td>System</td><td>Latency</td><td></td><td>RTF* RAM VRAM</td><td></td><td>GPU util.</td><td>GPU energy</td></tr><tr><td>ACE-Step v1 3.5B</td><td> $\overline { { 2 . 3 \pm \ : \ : 0 . 0 3 } }$ </td><td>0.1</td><td>3.2</td><td>8.8</td><td>79.2</td><td>0.2</td></tr><tr><td>AudioLDM2 Music</td><td> $1 7 . 5 \pm \ : 0 . 0 5$ </td><td>0.6</td><td>4.1</td><td>4.5</td><td>93.0</td><td>2.2</td></tr><tr><td>DiffRhythm</td><td> $1 1 . 0 \pm \ : \ : 0 . 0 7$ </td><td>0.4</td><td>5.2</td><td>7.0</td><td>98.7</td><td>1.4</td></tr><tr><td>HeartMuLa OSS 3B</td><td> $1 7 . 4 \pm \ : 0 . 0 8$ </td><td>0.6</td><td>3.1</td><td>21.5</td><td>88.5</td><td>1.2</td></tr><tr><td>InspireMusic 1.5B Long</td><td> $2 0 . 0 \pm \ : \ : 0 . 0 3$ </td><td>0.7</td><td>7.0</td><td>11.5</td><td>85.0</td><td>1.2</td></tr><tr><td>MusicGen Large</td><td> $2 3 . 7 \pm \ : 0 . 0 8$ </td><td>0.8</td><td>2.5</td><td>19.1</td><td>92.9</td><td>1.7</td></tr><tr><td>MusicLDM</td><td> $9 . 3 \pm 0 . 0 1$ </td><td>0.3</td><td>3.1</td><td>3.2</td><td>92.4</td><td>1.1</td></tr><tr><td>Stable Audio Open 1.0</td><td> $6 2 . 0 \pm \ : \ : 0 . 1 2$ </td><td>2.1</td><td>3.1</td><td>11.0</td><td>97.6</td><td>7.7</td></tr><tr><td>YuE</td><td> $4 3 9 . 5 \pm 6 5 . 8 6$ </td><td>14.7</td><td>19.4</td><td>15.6</td><td>83.4</td><td>40.9</td></tr></table>

RTF values report means only; SDs are < 0.01, except for YuE (2.20).

(b) Commercial systems
<table><tr><td>System</td><td>Latency</td><td>RTF</td><td>Cost</td></tr><tr><td>Lyria 2 (via fal.ai)</td><td> $\overline { { 3 6 . 4 \pm 1 1 . 9 } }$ </td><td> $1 . 1 \pm 0 . 3 6$ </td><td>0.10</td></tr><tr><td>Lyria 3 (via fal.ai)</td><td> $1 1 . 6 \pm 1 . 8$ </td><td> $0 . 4 \pm 0 . 0 6$ </td><td>0.04</td></tr><tr><td>Stable Audio 2.5</td><td> $5 . 6 \pm \ : \ : 0 . 6$ </td><td> $0 . 2 \pm 0 . 0 2$ </td><td>0.20</td></tr><tr><td>Stable Audio 3</td><td> $9 . 8 \pm \ : 2 . 1$ </td><td> $0 . 3 \pm 0 . 0 7$ </td><td>0.26</td></tr></table>

## 5. CONCLUSION

In this paper, we present TTM-Bench, a reference-based framework for benchmarking TTM system performance across musical-content alignment and computational efficiency. The case study demonstrates the framework’s appli cation to 13 TTM systems using a corpus of 100 reference music tracks. The results show that alignment components capture distinct information at the generation level, supporting their individual reporting alongside the aggregate MCAS. Relative system-level MCAS results remain stable across the tested weighting alternatives, whereas small differences between similarly scoring systems are sensitive to reference-set composition. Computational results reveal that higher musical-content alignment does not systematically coincide with lower computational demands. These findings are limited to the evaluated references, system versions, configurations, and access conditions. Moreover, MCAS remains an experimentally defined aggregate whose weighting should be further validated against human judgments. Future work will extend TTM-Bench to broader reference collections and investigate the relationship between its measures and human judgments of TTM system performance.

## 6. ACKNOWLEDGMENT

No funding was received for conducting this study. The authors have no relevant financial or nonfinancial interests to disclose.

## 7. COMPLIANCE WITH ETHICAL STANDARDS

This study did not involve human participants or animals;   
therefore, ethical approval was not required.

## 8. REFERENCES

[1] Y. Wu et al., “Large-scale contrastive language–audio pretraining with feature fusion and keyword-to-caption augmentation,” in ICASSP, 2023.

[2] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Frechet audio distance: A reference-free metric for´ evaluating music enhancement algorithms,” in ISCA, 2019.

[3] L.-C. Yang and A. Lerch, “On the evaluation of generative models in music,” Neural Comput. Appl., vol. 32, 2020.

[4] A. Lerch et al., “Survey on the evaluation of generative models in music,” ACM Comput. Surv., vol. 58, 2025.

[5] A. Gui, H. Gamper, S. Braun, and D. Emmanouilidou, “Adapting frechet audio distance for generative music´ evaluation,” in ICASSP, 2024.

[6] C. Liu et al., “MusicEval: A generative music dataset with expert ratings for automatic text-to-music evaluation,” in ICASSP, 2025.

[7] D. Zhu and Z. Li, “MuQ-Eval: An open-source persample quality metric for AI music generation evaluation,” 2026.

[8] D. Wu et al., “SongBench: A fine-grained multi-aspect benchmark for song quality assessment,” 2026.

[9] H. Wang et al., “AudioEval: Automatic dualperspective and multi-dimensional evaluation of text-toaudio-generation,” 2026.

[10] Y. Kim et al., “TuneJury: An open metric for improving music generation preference alignment,” 2026.

[11] Z. Evans et al., “Fast timing-conditioned latent audio diffusion,” in ICML, 2024, vol. 235.

[12] Z. Ning et al., “DiffRhythm: Blazingly fast and embarrassingly simple end-to-end full-length song generation with latent diffusion,” 2025.

[13] J. Gong et al., “ACE-Step: A step towards music generation foundation model,” 2025.

[14] F. Grotschla, A. Solak, L. A. Lanzend¨ orfer, and R. Wat-¨ tenhofer, “Benchmarking music generation models and metrics via human preference studies,” in ICASSP, 2025.

[15] Y. Kim et al., “Music arena: Live evaluation for text-tomusic,” 2025.

[16] F.-C. Hsieh et al., “Academic text-to-music grand challenge: Datasets, baselines, and evaluation methods,” 2026.

[17] P. Alonso-Jimenez, X. Serra, and D. Bogdanov, “Effi-´ cient supervised training of audio transformers for music representation learning,” in ISMIR, 2023.

[18] P. Alonso-Jimenez, X. Serra, and D. Bogdanov, “Music´ representation learning based on editorial metadata from Discogs,” in ISMIR, 2022.

[19] D. Bogdanov et al., “The MTG-Jamendo dataset for automatic music tagging,” in ICML ML4MD, 2019.

[20] S. Rouard, F. Massa, and A. Defossez, “Hybrid trans-´ formers for music source separation,” in ICASSP, 2023.

[21] H. Liu et al., “AudioLDM 2: Learning holistic audio generation with self-supervised pretraining,” ASLP, vol. 32, 2024.

[22] D. Yang et al., “HeartMuLa: A family of open sourced music foundation models,” 2026.

[23] C. Zhang et al., “InspireMusic: Integrating super resolution and large language model for high-fidelity longform music generation,” 2025.

[24] J. Copet et al., “Simple and controllable music generation,” in NeurIPS, 2023, vol. 36.

[25] K. Chen et al., “MusicLDM: Enhancing novelty in text-to-music generation using beat-synchronous mixup strategies,” in ICASSP, 2024.

[26] Z. Evans et al., “Stable audio open,” 2024.

[27] R. Yuan et al., “YuE: Scaling open foundation models for long-form music generation,” 2025.

[28] Google DeepMind, “Lyria 2,” Online, 2025.

[29] Google DeepMind, “Lyria 3,” Online, 2026.

[30] Stability AI, “Stable Audio 2.5,” Online, 2025.

[31] Z. Evans et al., “Stable audio 3,” 2026.

[32] D. Bogdanov et al., “Essentia: An audio analysis library for music information retrieval,” in ISMIR, 2013.

[33] P. Alonso-Jimenez, D. Bogdanov, J. Pons, and X. Serra,´ “TensorFlow audio models in Essentia,” in ICASSP, 2020.

[34] A. Grattafiori et al., “The Llama 3 herd of models,” 2024.