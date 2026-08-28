# MM-Spectrum: Multimodal Multi-spectral Molecular Structural Elucidation with a Stable MoE Framework

Hai-tao Yu <sup>1</sup> Nan Min <sup>2</sup> Zheng Fang <sup>1</sup> Hongyu Zhan <sup>1</sup> Yusen Tan <sup>1</sup> Yuhan Wang <sup>3</sup> Jun Xia <sup>1</sup> <sup>4</sup>

## Abstract

Inferring molecular structures from multimodal spectroscopic measurements requires integrating complementary yet highly heterogeneous signals. However, the common paradigm of directly concatenating multispectral sequences can exhibit anomalous performance degradation, primarily due to pronounced heterogeneity and the resulting multimodal imbalance across modalities. As a remedy, we propose MM-Spectrum, a sparse Mixture-of-Experts framework tailored for multimodal multispectral spectra-to-structure elucidation. To better match the information characteristics under multispectral imbalance, MM-Spectrum introduces an explicit modality-aware routing mechanism that exposes spectral identity to the router in addition to token content representations. Moreover, it incorporates shared and interaction experts, together with heterogeneous expert capacities, to extract multispectral modality-unique and cross-modal synergistic information while suppressing noise-induced interference. Across full-modality, bimodal, and missing-modality settings on molecular structural elucidation, MM-Spectrum achieves consistent and substantial improvements, supported by ablation studies and interpretability analyses. Code is available at https://github.com/ HHHTTY/MM-Spectrum.

## 1. Introduction

Spectroscopy-driven molecular structural elucidation is a foundational capability for chemical discovery (Alberts et al., 2024). In learning-based elucidation, the key challenge is often not whether a single modality can predict

![](images/888e303732464e68bba06065817682de91c9b2e8c343e89db3a05bd0f903b6fc.jpg)  
(a) Mechanism of Failure: the naive concatenation baseline leading to MultiSpectral Imbalance.

![](images/2619fd9e526705aa0e613a5a22bff1e98ca4c1c206e7fee4a2b88f8a07f9873a.jpg)  
(b) Empirical Consequence: Full-modality inputs cause a performance “collapse” in the naive concatenation baseline, while MM-Spectrum achieves synergy.

Figure 1. The Challenge of Multispectral Elucidation: Mechanism and Consequence. In contrast, MM-Spectrum resolves this imbalance, effectively utilizing all modalities to achieve state-of-the-art performance.

molecular structures, but whether multiple modalities can achieve stable synergy (Liang et al., 2024). Unlike generic multimodal learning settings, spectroscopic modalities exhibit complementarity with clear physical meaning. Nuclear magnetic resonance (NMR) provides high-resolution constraints on local atomic environments and molecular topology, yet its nonlinear peak patterns lead to complex symbolic structures (Klukowski et al., 2025). Infrared spectroscopy (IR) offers coarse evidence of functional groups, but is typically noisy and low in information density, requiring long fragments to form stable semantics (Yang et al., 2021). Mass spectrometry (MS) contributes strong global constraints through molecular mass and fragmentation, but is limited in distinguishing many stereoisomers and constitutional isomers (Duhrkop et al.¨ , 2015).

In molecular structural elucidation, multispectral sequences are concatenated into a long input and mapped to molecular structures in an end-to-end manner. We focus on a counterintuitive phenomenon in multispectral modeling: providing full-modality inputs does not necessarily outperform unimodal settings and can even lead to substantial degradation. As illustrated in Figure 1, naive full-modality concatenation can underperform strong unimodal or bimodal baselines, and the full-modality model may drop sharply compared to the NMR-only setting. We attribute this failure primarily to the heterogeneity and imbalance of multispectral data (Wu et al., 2022). Specifically, modality distributions differ drastically, and simple concatenation makes “long but weak” modalities dominate attention: IR sequences are often extremely long with strong token autocorrelation, whereas NMR is shorter but has much higher semantic density. Meanwhile, modalities also differ structurally: NMR often contains structured symbols, while MS and IR are closer to peak lists or locally correlated continuous statistics. During optimization, different modalities can induce competing gradient directions. Low-density modalities may dominate the update process, and such gradient dominance can force the model toward a suboptimal compromise among competing objectives (Yu et al., 2020).

To address these issues, we reconstruct multispectral fusion from naive concatenation into a structured system of expert division of labor. Multispectral fusion should not be treated as a crude additive aggregation of evidence, but rather as a structured allocation and coordination process over heterogeneous sources. Mixture-of-Experts (MoE) provides a natural path: via sparse activation, each modality invokes only a small subset of experts, increasing effective capacity while enabling specialization. However, the balancing assumptions in standard MoE do not automatically match the severe imbalance in multispectral elucidation, and MoE can introduce new instabilities (Fedus et al., 2022). We therefore propose MM-Spectrum, a stable MoE framework designed specifically for multispectral imbalance and information synergy. We view multispectral fusion as a problem of structured allocation and information decomposition under imbalance constraints. To better fit the imbalanced characteristics of multispectral signals, we design an explicit modality-aware routing mechanism, allowing the router to access not only token content representations but also explicit spectral modality identities, which reduces routing difficulty and stabilizes specialization (Zhou et al., 2022).

Moreover, multispectral modalities are not simply complementary, they involve both redundancy and synergy (Williams & Beer, 2010). This property also explains why naive concatenation fails, as weaker modalities may be unable to contribute effectively to molecular structural elucidation. MM-Spectrum introduces shared experts and interaction experts to capture cross-modality redundancy and synergy, and further incorporates heterogeneous expert capacities with computation-cost regularization (Du et al., 2022). Under multispectral inputs, this design encourages the model to extract modality-unique and synergistic information while suppressing interference from noise. As a result, MM-Spectrum not only improves performance but also provides interpretable evidence for mitigating multispectral imbalance, supporting a synergistic learning mechanism for multispectral complementarity.

Contributions. MM-Spectrum makes the following contributions:

• We are the first to identify multimodal imbalance induced by heterogeneity in multispectral spectrato-structure elucidation, and to interpret it through information-density disparity and gradient conflicts.

• We propose MM-Spectrum, a spectroscopy-oriented stable sparse MoE framework with modality-aware routing and a structured expert space that separates redundant, unique, and synergistic information pathways. We further introduce heterogeneous experts with computation-cost regularization to improve accuracy while reducing training and inference overhead.

• We demonstrate consistent improvements in fullmodality, bimodal, and missing-modality settings, supported by ablation studies and interpretability analyses.

## 2. Related Work

## 2.1. Spectra-to-Structure Elucidation

Computational molecular structural elucidation has evolved from database-driven search to generative modeling. Classical approaches typically rely on retrieving candidates from large molecular databases using spectral fingerprints. Representative systems such as CSI:FingerID (Duhrkop et al.¨ , 2015) and SIRIUS 4 (Duhrkop et al.¨ ) utilize kernel support vector machines to predict molecular fingerprints from tandem mass spectrometry (MS/MS) data. Complementary to MS, NMR-based elucidation has traditionally depended on axiomatic construction algorithms or database lookups to assemble structures from chemical shift constraints (Klukowski et al., 2025).

Recently, deep generative models have been increasingly applied to this task. For mass spectrometry, autoregressive models such as MassGenie (Shrivastava et al., 2021) and Spec2Mol (Litsa et al., 2023) generate molecular structures from mass spectra, while diffusion-based approaches such as DiffMS (Bohde et al., 2025) explore alternative conditional generation mechanisms. Similarly, for NMR, approaches range from token-based generation (Klukowski et al., 2025) and multitask frameworks like NMR2Struct (Hu et al., 2024) to graph neural networks that reconstruct atomic environments from chemical shifts (Guan et al., 2021). Furthermore, for IR spectroscopy, contrastive learning frameworks such as Spectra-to-Structure (Kanakala et al., 2024) have been explored. In the multimodal regime, however, the prevailing baseline remains naive early-fusion, where heterogeneous modalities are concatenated or pooled before processing (Yang et al., 2021), although some recent works explore contrastive alignment to improve cross-modal consistency (Liang et al., 2024). While architecturally simple, the concatenation paradigm implicitly assumes that token semantics and information densities are comparable across modalities. Recent studies in multimodal learning suggest this assumption is often violated: modalities with different convergence rates or noise levels can lead to negative transfer or greedy optimization behavior, where a dominant modality suppresses the learning of others due to gradient conflicts (Wang et al., 2020; Yu et al., 2020; Wu et al., 2022).

## 2.2. Mixture-of-Experts and Balancing Mechanisms

Sparse Mixture-of-Experts (MoE) models scale model capacity without a proportional increase in computational cost by conditionally activating a subset of parameters. The sparsely-gated MoE layer (Shazeer et al., 2017) introduced a learnable gating network to route tokens to top-k experts, a design later scaled to trillion-parameter regimes by GShard (Lepikhin et al., 2021) and Switch Transformers (Fedus et al., 2022). To prevent ”expert collapse”—where the router trivially assigns all tokens to a single expert—these models universally employ auxiliary balancing losses based on expert importance and load statistics (Fedus et al., 2022; Du et al., 2022). Such conditional computation has also been successfully adapted to other domains, such as vision, to handle patch-level redundancy (Riquelme et al., 2021).

However, standard balancing formulations are predicated on the assumption that tokens are relatively homogeneous in information content and should be distributed uniformly. Strict uniform balancing often hinders specialization by misallocating high-capacity experts to low-utility noise tokens(Zhou et al., 2022; Lewis et al., 2021). MM-Spectrum introduces a spectrum-specific inductive bias, leveraging modality-aware routing and cost-regularization to align expert specialization with the physical nature of spectrum.

## 3. Methodology

## 3.1. Preliminaries

Task Definition We consider multimodal spectroscopybased Molecular Structural Elucidation with M modalities

(in this work $M \ = \ 3 ,$ , corresponding to NMR, IR, and MS). For modality $m \in \{ 1 , \ldots , M \}$ , the input is a token sequence:

$$
x ^ { ( m ) } = \bigl ( x _ { 1 } ^ { ( m ) } , x _ { 2 } ^ { ( m ) } , \ldots , x _ { L _ { m } } ^ { ( m ) } \bigr ) .\tag{1}
$$

The multimodal observation is denoted by $\begin{array} { r l } { X } & { { } = } \end{array}$ $\{ x ^ { ( m ) } \} _ { m = 1 } ^ { M }$ , and the target molecular structure is represented as a discrete sequence $y = ( y _ { 1 } , \dots , y _ { T } ) \in { \mathcal { V } } ( \mathbf { e } . \mathbf { g } . _ { : }$ , a SMILES string). Using an encoder–decoder Transformer, the encoder produces contextual representations and the decoder generates y autoregressively. The complete autoregressive likelihood formulation and training objective are provided in Section A.1 of the appendix.

Standard Sparse Mixture-of-Experts To scale model capacity without proportional computational costs, Sparse Mixture-of-Experts (MoE) replaces the dense feed-forward network (FFN) with a set of $E$ experts $\{ \mathrm { F F N } _ { e } \} _ { e = 1 } ^ { E }$ . For an input token representation $h ,$ a learnable router $W _ { r }$ computes a probability distribution $p ( h )$ and activates only the top-k experts. While efficient, standard MoE routing relies solely on token content h. In multimodal spectroscopy, this design is prone to failure due to extreme sequence imbalance: abundant, low-density tokens (long IR sequences) often numerically dominate the routing statistics, causing the model to neglect sparse but constraint-rich signals (NMR). We address this limitation in MM-Spectrum by introducing modality-aware routing and structured expert subspaces.

## 3.2. Spectrum-Aware Representation

As illustrated in Figure 2, MM-Spectrum restructures the spectra-to-structure generation pipeline to address the intrinsic statistical imbalance where long, low-density modalities (MS and IR) numerically dominate short, high-density constraints (NMR). The framework integrates three design principles: (i) aligning token statistics via compression, (ii) exposing modality identity for routing, and (iii) decoupling information flow via structured expert subspaces.

To mitigate the mismatch between sequence length $L _ { m }$ and semantic density $\rho _ { m }$ , we formalize data preprocessing as modality-specific operators $\phi _ { m }$ that map raw inputs $x ^ { ( m ) }$ to compressed representations $\tilde { x } ^ { ( m ) }$

$$
\tilde { x } ^ { ( m ) } = \phi _ { m } ( x ^ { ( m ) } ) , \qquad \tilde { X } = \{ \tilde { x } ^ { ( m ) } \} _ { m = 1 } ^ { M } .\tag{2}
$$

We instantiate $\phi _ { m }$ to balance information retention against computational cost based on signal topology. For High-Density Modalities (NMR), we set ϕ ≈ Identity to preserve discrete, constraint-rich topology signals without smoothing. Conversely, for High-Redundancy Modalities (IR/MS), we apply adaptive compression—specifically local binning for autocorrelated IR and top-k filtering for noisy MS—to excise background noise. This preprocessing effectively aligns the effective density $\rho _ { m }$ across modalities, preventing attention dilution and ensuring stable gradient statistics for the subsequent MoE layers. Detailed modalityspecific tokenization, compression operators, and sequence configurations are provided in Section B.1 of the appendix.

![](images/e873bba01baa7c98473d423f1469138876f3a347e4e0a5d2c05a9b11cb906ed5.jpg)  
Figure 2. MM-Spectrum overview: PID-guided information decoupling with structured MoE and curriculum scheduling. We formulate multimodal spectra-to-structure generation as an end-to-end encoder–decoder task, where each modality is tokenized with explicit Modality Tags. The encoder block follows LayerNorm → MHSA → Structured MoE → LayerNorm, where an interaction-aware gating mechanism leverages modality tags to route tokens into a structured expert assembly.

## 3.3. Spectrum-Aware Routing

Standard content-based routing requires the gate to implicitly infer signal origin from noisy token statistics, which is suboptimal for heterogeneous spectroscopy. MM-Spectrum explicitly injects modality information into the routing latent space, decoupling identity from content (see Figure 4). For a token i with content representation $h _ { i } \in \mathbb { R } ^ { d }$ and modality index $m ( i )$ , we introduce a learnable tag embedding $e _ { m ( i ) }$ (Vaswani et al., 2017) and a modality bias $b _ { m ( i ) }$ to construct the router input $\tilde { h } _ { i }$

$$
\tilde { h } _ { i } \ = \ h _ { i } \ + \ e _ { m ( i ) } \ + \ b _ { m ( i ) } .\tag{3}
$$

The gating network then computes expert probabilities and performs Top-k selection:

$$
\begin{array} { l } { p _ { i , e } = \mathrm { s o f t m a x } \Big ( \frac { w _ { e } ^ { \top } \tilde { h } _ { i } } { \tau } \Big ) _ { e } , } \\ { o _ { i } = \displaystyle \sum _ { e \in \mathrm { T o p K } ( p _ { i , : } , k ) } p _ { i , e } \cdot \mathrm { F F N } _ { e } ( h _ { i } ) . } \end{array}\tag{4}
$$

Geometrically, Eq. (3) induces a translation-like bias in the router’s decision space. The term $b _ { m ( i ) }$ encodes a disentangled modality preference, stabilizing early-stage coarse differentiation while preserving the metric space of $h _ { i }$ for fine-grained content matching.

Crucially, this mechanism promotes Soft Specialization. Unlike hard-coding specific experts to modalities (which causes gradient blocking and limits synergy), MM-Spectrum allows specialization to emerge dynamically. Guided by the explicit cues in $\tilde { h } _ { i }$ and constrained by computationaware regularization, the router learns to direct tokens to appropriate experts—allocating unique constraints to specific experts and cross-modal patterns to interaction experts—without forfeiting the flexibility to exploit shared parameters when beneficial. Additional derivations of the modality-conditioned routing formulation and the associated stability-to-specialization schedule are given in Sections A.4 and A.6 of the appendix.

## 3.4. Structured Expert Space

To effectively disentangle heterogeneous signals, we adopt Partial Information Decomposition (PID) (Williams & Beer, 2010) as a theoretical inductive bias. As conceptualized in Figure 3, we posit that the mutual information between multimodal spectra X and structure y decomposes into redundant (R), unique $( U _ { m } ) ,$ and synergistic (S) components:

![](images/0e48d101017a7ae1aeefcce545dc7e4bef14f9d4cc093ff8c9518276398df627.jpg)  
Figure 3. Decomposition of multimodal spectral information. The diagram enumerates modality subsets (single, pairwise, and all-modal combinations) and associates them with the dominant information type (unique vs. synergistic vs. redundant).

$$
\operatorname { I } ( X ; y ) \approx R + \sum _ { m = 1 } ^ { M } U _ { m } + S .\tag{5}
$$

Guided by this formulation and multi-task routing paradigms (Ma et al., 2018), we partition the expert pool C into three functionally distinct subsets: $\mathcal { C } \ = \ \mathcal { C } _ { \mathrm { s h } }$ ∪ $( \bigcup _ { m } \mathcal { C } _ { m } ) \cup \mathcal { C } _ { \mathrm { i n t } }$ . Here, Shared experts $( \mathcal { C } _ { \mathrm { s h } } )$ capture redundancy R; Modality-specific experts $( { \mathcal { C } } _ { m } )$ extract unique constraints $U _ { m } ;$ and crucially, Interaction experts $( \mathcal { C } _ { \mathrm { i n t } } )$ model synergy S, processing cross-modal dependencies that emerge only under joint consideration.

To operationalize this structure without intractable PID estimation, we impose lightweight regularizers that enforce the principle “redundant-consistent, unique-separable.” We apply a consistency loss $\mathcal { L } _ { \mathrm { s h } }$ to encourage shared experts to produce invariant representations across modalities, and a separability loss $\mathcal { L } _ { \mathrm { s e p } }$ to maintain the distinctiveness of modality-specific features:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s t r u c t } } = \mathbb { E } \Big [ \displaystyle \sum _ { e \in \mathcal { C } _ { \mathrm { s h } } } \mathrm { D i s t } ( o _ { i } ^ { ( e ) } , o _ { j } ^ { ( e ) } ) \Big ] } \\ & { \quad \quad \quad + \mathbb { E } \Big [ \displaystyle \sum _ { m = 1 } ^ { M } \sum _ { e \in \mathcal { C } _ { m } } \mathrm { S e p } ( o _ { i _ { m } } ^ { ( e ) } , \{ o ^ { ( e ) } \} _ { m ^ { \prime } \not = m } ) \Big ] . } \end{array}\tag{6}
$$

This formulation ensures that the router aligns token allocation with the physical nature of the evidence (as visualized in Figure 4), preventing parameter entanglement while preserving synergistic pathways. Further discussion of the PIDinspired expert partition and the corresponding structural regularizers is included in Section A.7 of the appendix.

![](images/fa5290442b0f405f940aadba6aae973c1b36a8ee9cccf3b48ab4790f9a5b1399.jpg)  
Figure 4. Channelized modality-aware routing aligned with redundancy/unique/synergy pathways. A modality-aware gating network consumes modality-conditioned representations (e.g., NMR/IR/MS embeddings) and produces routing distributions.

## 3.5. Heterogeneous Experts

Heterogeneous Expert Spectrum. To address the disparity in information density—where constraint-rich NMR tokens demand high nonlinearity while redundant IR tokens require only shallow processing—MM-Spectrum moves beyond homogeneous expert pools. We diversify the expert space into Heavy and Light experts, parameterized by varying capacities $\kappa _ { e }$ (e.g., hidden dimensions). This design constructs a capacity spectrum, theoretically enabling the model to allocate computational resources proportional to the marginal utility of each token $\Delta _ { i }$

Computation-Aware Regularization. Standard routing does not spontaneously align expert capacity with token difficulty. To enforce a “Heavy-for-hard, Light-for-easy” specialization logic, we introduce a computation cost penalty. Let $c _ { e } \propto \kappa _ { e }$ be the cost assigned to expert e. The expected computational budget for a batch is minimized via:

$$
\mathcal { L } _ { \mathrm { c o s t } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { e = 1 } ^ { E } p _ { i , e } c _ { e } .\tag{7}
$$

This soft constraint compels the router to activate Heavy experts only when the reduction in task loss $\mathcal { L } _ { \mathrm { t a s k } }$ outweighs the incurred cost $c _ { e } .$ , effectively inducing a Paretoefficient resource allocation. Figure 5 visually demonstrates this capacity–utility alignment, where expensive computation is reserved for high-utility tokens. The constrainedoptimization interpretation of this objective and the resulting parameter and computation complexity are discussed in Sections A.8 and A.9 of the appendix.

Table 1. Modality-combination study. Top-K SMILES elucidation accuracy (%) under single-modality, bimodal, and full-modality.
<table><tr><td rowspan="2">Modality Setting</td><td colspan="10">Top-K Accuracy (%)</td></tr><tr><td>Top-1%</td><td>Top-2%</td><td> $\mathbf { T o p } { \mathbf { - } } 3 \%$ </td><td>Top-4% Top-5%</td><td></td><td>Top-6%</td><td> $\mathbf { T o p } ( 7 \%$ </td><td>Top-8%</td><td>Top-9%</td><td>Top-10%</td></tr><tr><td colspan="9">Single-modality</td><td></td></tr><tr><td>NMR  $( \mathrm { 1 H } + \mathrm { 1 3 C } )$ </td><td> $6 9 . 8 3 \pm 0 . 2 4$ </td><td> $7 7 . 8 8 \pm 0 . 1 6$ </td><td> $8 0 . 8 6 \pm 0 . 1 1$ </td><td> $8 2 . 3 8 \pm 0 . 3 8$ </td><td> $8 3 . 3 5 \pm 0 . 2 9$ </td><td> $8 4 . 0 3 \pm 0 . 3 4$ </td><td> $8 4 . 5 2 \pm 0 . 2 6$ </td><td> $8 4 . 8 9 \pm 0 . 3 2$ </td><td> $8 5 . 0 8 \pm 0 . 4 6$ </td><td> $8 5 . 1 8 \pm 0 . 1 2$ </td></tr><tr><td>MS (all MS spectra)</td><td> $1 7 . 3 5 \pm 0 . 3 0$ </td><td> $2 3 . 8 2 \pm 0 . 2 2$ </td><td> $2 7 . 8 0 \pm 0 . 4 1$ </td><td> $3 0 . 0 8 \pm 0 . 2 9$ </td><td> $3 1 . 7 3 \pm 0 . 1 5$ </td><td> $3 2 . 9 0 \pm 0 . 1 8$ </td><td> $3 3 . 8 1 \pm 0 . 4 0$ </td><td> $3 4 . 5 2 \pm 0 . 4 6$ </td><td> $3 5 . 0 4 \pm 0 . 1 7$ </td><td> $3 5 . 3 1 \pm 0 . 3 5$ </td></tr><tr><td>IR (binned spectrum)</td><td> $2 3 . 2 2 \pm 0 . 4 7$ </td><td> $3 0 . 9 0 \pm 0 . 4 2$ </td><td> $3 5 . 0 1 \pm 0 . 1 7$ </td><td> $3 7 . 6 5 \pm 0 . 1 4$ </td><td> $3 9 . 4 4 \pm 0 . 2 7$ </td><td> $4 0 . 7 2 \pm 0 . 2 1$ </td><td> $4 1 . 7 3 \pm 0 . 2 2$ </td><td> $4 2 . 4 4 \pm 0 . 2 9$ </td><td> $4 2 . 8 8 \pm 0 . 2 1$ </td><td> $4 3 . 0 7 \pm 0 . 1 9$ </td></tr><tr><td colspan="9">Dual-modality</td><td></td></tr><tr><td>NMR + MS (Baseline)</td><td> $5 9 . 1 5 \pm 0 . 4 1$ </td><td> $6 7 . 5 2 \pm 0 . 4 6$ </td><td> $7 0 . 7 5 \pm 0 . 3 3$ </td><td> $7 2 . 3 4 \pm 0 . 2 0$ </td><td> $7 3 . 3 4 \pm 0 . 2 0$ </td><td> $7 4 . 0 1 \pm 0 . 3 3$ </td><td> $7 4 . 5 8 \pm 0 . 3 7$ </td><td> $7 5 . 0 2 \pm 0 . 2 5$ </td><td> $7 5 . 3 5 \pm 0 . 1 8$ </td><td> $7 5 . 5 2 \pm 0 . 2 5$ </td></tr><tr><td>NMR + MS (MM-Spectrum)</td><td> $\mathbf { 7 2 . 9 8 \pm 0 . 1 8 }$ </td><td> ${ \bf 8 3 . 0 2 \pm 0 . 1 8 }$ </td><td> $\mathbf { 8 5 . 1 8 \pm 0 . 2 0 }$ </td><td> $\mathbf { 8 6 . 1 4 \ : \pm 0 . 4 9 }$ </td><td> $\mathbf { 8 6 . 3 3 \pm 0 . 2 9 }$ </td><td> $\mathbf { 8 6 . 6 0 \ : \pm 0 . 4 9 }$ </td><td> ${ \bf 8 7 . 1 0 \pm 0 . 1 8 }$ </td><td> $\mathbf { 8 7 . 2 4 \ : \pm 0 . 4 1 }$ </td><td> $\mathbf { 8 7 . 5 1 \pm 0 . 3 8 }$ </td><td> $\mathbf { 8 7 . 6 6 \pm 0 . 2 3 }$ </td></tr><tr><td>NMR + IR (Baseline)</td><td> $6 4 . 8 8 \pm 0 . 3 4$ </td><td> $7 4 . 5 7 \pm 0 . 3 2$ </td><td> $7 8 . 7 6 \pm 0 . 2 1$ </td><td> $8 0 . 9 6 \pm 0 . 1 2$ </td><td> $8 2 . 4 0 \pm 0 . 4 6$ </td><td> $8 3 . 4 0 \pm 0 . 1 6$ </td><td> $8 4 . 0 9 \pm 0 . 4 0$ </td><td> $8 4 . 6 2 \pm 0 . 1 1$ </td><td> $8 4 . 9 6 \pm 0 . 2 1$ </td><td> $8 5 . 0 8 \pm 0 . 1 9$ </td></tr><tr><td>NMR + IR (MM-Spectrum)</td><td> ${ \bf 7 1 . 9 6 \pm 0 . 1 8 }$ </td><td> $\mathbf { 8 1 . 5 0 \pm 0 . 3 7 }$ </td><td> $\mathbf { 8 3 . 8 3 \pm 0 . 4 8 }$ </td><td> ${ \bf 8 4 . 9 1 \pm 0 . 1 7 }$ </td><td> $\mathbf { 8 5 . 5 2 \pm 0 . 4 8 }$ </td><td> ${ \bf 8 5 . 9 9 \pm 0 . 4 3 }$ </td><td> $\mathbf { 8 6 . 3 0 \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 6 . 5 5 \pm 0 . 3 0 }$ </td><td> $\mathbf { 8 6 . 8 0 \pm 0 . 2 0 }$ </td><td> $\mathbf { 8 6 . 9 9 \pm 0 . 4 6 }$ </td></tr><tr><td>MS + IR (Baseline)</td><td> $3 1 . 2 2 \pm 0 . 1 1$ </td><td> $3 9 . 6 8 \pm 0 . 2 8 $ </td><td> $4 4 . 1 0 \pm 0 . 1 8$ </td><td> $4 6 . 7 7 \pm 0 . 2 8$ </td><td> $4 8 . 6 5 \pm 0 . 3 2$ </td><td> $5 0 . 0 6 \pm 0 . 1 1$ </td><td> $5 1 . 2 0 \pm 0 . 1 2$ </td><td> $5 2 . 1 0 \pm 0 . 4 9$ </td><td> $5 2 . 6 5 \pm 0 . 3 0$ </td><td> $5 2 . 9 3 \pm 0 . 3 1$ </td></tr><tr><td>MS + IR (MM-Spectrum)</td><td> $\mathbf { 3 7 . 9 0 \pm 0 . 3 4 }$ </td><td> $\mathbf { 4 7 . 0 8 \ : \pm 0 . 4 5 }$ </td><td> ${ \bf 5 2 . 0 3 \pm 0 . 1 5 }$ </td><td> $\mathbf { 5 4 . 9 0 \ : \pm 0 . 4 8 }$ </td><td> ${ \bf 5 6 . 8 8 \pm 0 . 2 5 }$ </td><td> ${ \bf 5 8 . 4 8 \pm 0 . 3 9 }$ </td><td> ${ \bf 5 9 . 6 1 \pm 0 . 1 4 }$ </td><td> ${ \bf 6 0 . 4 3 \pm 0 . 3 7 }$ </td><td> ${ \bf 6 0 . 9 8 \pm 0 . 2 1 }$ </td><td> ${ \bf 6 1 . 2 6 \pm 0 . 4 0 }$ </td></tr><tr><td colspan="9">Tri-modality</td><td></td></tr><tr><td> $\mathrm { N M R } + \mathrm { M S } + \mathrm { I R } \left( \mathrm { B a s e l i n e } \right)$  NMR + MS + IR (MM-Spectrum)</td><td> $4 4 . 2 9 \pm 0 . 3 5$ </td><td> $5 2 . 7 1 \pm 0 . 2 8$ </td><td> $5 6 . 1 7 \pm 0 . 2 7$ </td><td> $5 8 . 1 1 \pm 0 . 2 4$ </td><td> $5 9 . 3 9 \pm 0 . 3 1$ </td><td> $6 0 . 2 5 \pm 0 . 4 8$ </td><td> $6 0 . 9 7 \pm 0 . 2 1$ </td><td> $6 1 . 5 2 \pm 0 . 3 1$ </td><td> $6 1 . 8 2 \pm 0 . 1 8$ </td><td> $6 2 . 0 4 \pm 0 . 4 7$ </td></tr><tr><td>Improvement</td><td> ${ \bf 7 6 . 0 4 \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 3 . 0 8 \pm 0 . 1 1 }$ </td><td> $\mathbf { 8 5 . 6 5 \pm 0 . 2 0 }$ </td><td> $\mathbf { 8 6 . 9 8 \pm 0 . 1 3 }$ </td><td> $\mathbf { 8 7 . 8 3 \pm 0 . 2 5 }$ </td><td> $\mathbf { 8 8 . 6 6 \pm 0 . 2 4 }$ </td><td> $\mathbf { 8 9 . 3 1 \pm 0 . 1 7 }$ </td><td> $\mathbf { 8 9 . 7 4 \ : \pm 0 . 4 0 }$ </td><td> ${ \bf 9 0 . 0 1 \pm 0 . 3 0 }$ </td><td> ${ \bf 9 0 . 2 6 \pm 0 . 1 3 }$ </td></tr><tr><td></td><td> $3 1 . 7 5 \pm 0 . 3 9$ </td><td> $3 0 . 3 7 \pm 0 . 2 7$ </td><td> $2 9 . 4 8 \pm 0 . 3 7$ </td><td> $2 8 . 8 7 \pm 0 . 1 4$ </td><td> $2 8 . 4 4 \pm 0 . 4 1$ </td><td> $2 8 . 4 1 \pm 0 . 3 2$ </td><td> $2 8 . 3 4 \pm 0 . 3 1$ </td><td> $2 8 . 2 2 \pm 0 . 4 3$ </td><td> $2 8 . 1 9 \pm 0 . 2 9$ </td><td> $2 8 . 2 2 \pm 0 . 3 9$ </td></tr></table>

![](images/617bb144c6036da6a5070415580a19906c0d40673c6db55d90dacf660595ccfe.jpg)  
Figure 5. Computation-aware regularization with heterogeneous experts: allocating expensive capacity to high-utility tokens. MM-Spectrum instantiates an explicit cost–utility principle in conditional computation. This mechanism prevents long, lowdensity modalities from monopolizing compute, while preserving the ability to allocate high-capacity processing to decision-critical evidence, improving both stability and full-modality performance.

Curriculum-Scheduled Objective. Standard MoE training faces a dilemma: strong balancing prevents collapse but hinders specialization, while weak balancing invites degeneracy (Zoph et al., 2022). We resolve this via a Stability-to-Specialization Curriculum(Bengio et al., 2009). The total objective is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { c o s t } } \mathcal { L } _ { \mathrm { c o s t } } + \lambda ( t ) \big ( \mathcal { L } _ { \mathrm { b a l } } + \mathcal { L } _ { \mathrm { e n t } } \big ) , } \end{array}\tag{8}
$$

ders a dynamic optimization trajectory: initially, the training begins with a coverage phase, where high regularization enforces broad expert utilization to prevent early degenerate collapse; subsequently, as constraints decay, the process enters an alignment phase where the explicit modality bias (Sec. 3.3) guides tokens toward structurally appropriate subspaces; finally, the curriculum culminates in a specialization phase where $\mathcal { L } _ { \mathrm { t a s k } }$ and $\mathcal { L } _ { \mathrm { c o s t } }$ dominate, driving the router toward precise, Pareto-efficient capacity allocation. Definitions of the balancing and entropy terms, together with the explicit schedules for λ(t) and τ(t), are provided in Sections A.5 and A.6 of the appendix.

where $\mathcal { L } _ { \mathrm { b a l } }$ and ${ \mathcal { L } } _ { \mathrm { e n t } }$ correspond to standard load balancing (squared coefficient of variation) and entropy maximization regularizers (Shazeer et al., 2017; Fedus et al., 2022). Crucially, the regularization weight $\lambda ( t )$ and router temperature $\tau ( t )$ are annealed over training steps t. This schedule engen-

## 4. Experiments

## 4.1. Benchmark, Data, and Evaluation Protocol

Benchmark and Modalities. We evaluate MM-Spectrum on the multimodal spectroscopic benchmark introduced by Alberts et al. (2024), a standard testbed for Molecular Structural Elucidation. The dataset provides paired molecular structures and measurements across three modalities: NMR (<sup>1</sup>H and $^ { 1 3 } \mathrm { C } )$ , IR, and MS. We adopt the canonical train/validation/test splits and preprocessing conventions to ensure comparability. Inputs are tokenized into modalityspecific sequences and processed by an encoder–decoder architecture to predict SMILES strings. And all compared methods share the same backbone capacity and optimization budget, isolating the performance differences to the fusion and conditional-computation mechanisms. Supplementary cross-dataset evaluations are also conducted on the experimental SDBS database (Saito & Kinugasa, 2011) and the large-scale MMST dataset (Priessner et al., 2026) (see Appendix C).

Metric. We report Top-K accuracy for SMILES-based elucidation (Weininger, 1988): for each input, the model produces K candidate SMILES and a prediction is counted as correct if the ground-truth appears in the top K list. Top-1 measures ranking sharpness and single-shot correctness, while larger K reflects candidate coverage and recoverability under downstream filtering or re-ranking.

Table 2. Missing-modality robustness. Top-K accuracy (%) under test-time missing-modality settings (S0–S6).
<table><tr><td rowspan="2">Setting</td><td colspan="3">Missing ratio</td><td colspan="3">Baseline</td><td colspan="3">MM-Spectrum</td><td colspan="3">Improvement</td></tr><tr><td>NMR</td><td>MS</td><td>IR</td><td>Top-1%</td><td>Top-5%</td><td> $\mathbf { T o p - 1 0 \% }$ </td><td>Top-1%</td><td>Top-5%</td><td> $\mathbf { T o p - l 0 \% }$ </td><td>Top-1%</td><td>Top-5%</td><td>Top-10%</td></tr><tr><td>SO</td><td>0.0</td><td>0.0</td><td>0.0</td><td> $4 4 . 2 9 \pm 0 . 5 0$ </td><td> $5 9 . 3 9 \pm 0 . 1 7$ </td><td> $6 2 . 0 4 \pm 0 . 4 4$ </td><td> $7 6 . 0 4 \pm 0 . 1 8$ </td><td> $8 7 . 8 3 \pm 0 . 2 8$ </td><td> $9 0 . 2 6 \pm 0 . 1 6$ </td><td> $3 1 . 7 5 \pm 0 . 4 9$ </td><td> $2 8 . 4 4 \pm 0 . 3 4$ </td><td> $2 8 . 2 2 \pm 0 . 4 1$ </td></tr><tr><td>S1</td><td>0.1</td><td>0.1</td><td>0.1</td><td> $3 3 . 4 2 \pm 0 . 4 5$ </td><td> $4 4 . 7 9 \pm 0 . 4 7$ </td><td> $4 6 . 8 0 \pm 0 . 4 9$ </td><td> $6 5 . 0 1 \pm 0 . 4 7$ </td><td> $7 5 . 1 1 \pm 0 . 3 7$ </td><td> $7 6 . 4 3 \pm 0 . 1 9$ </td><td> $3 1 . 5 9 \pm 0 . 3 0$ </td><td> $3 0 . 3 3 \pm 0 . 3 7$ </td><td> $2 9 . 6 3 \pm 0 . 1 7$ </td></tr><tr><td>S2</td><td>0.3</td><td>0.3</td><td>0.3</td><td> $1 7 . 2 9 \pm 0 . 1 2$ </td><td> $2 3 . 2 4 \pm 0 . 1 0$ </td><td> $2 4 . 3 1 \pm 0 . 3 7$ </td><td> $4 5 . 9 1 \pm 0 . 2 6$ </td><td> $5 4 . 2 1 \pm 0 . 1 7$ </td><td> $5 5 . 4 0 \pm 0 . 1 9$ </td><td> $2 8 . 6 2 \pm 0 . 3 3$ </td><td> $3 0 . 9 7 \pm 0 . 1 6$ </td><td> $3 1 . 0 9 \pm 0 . 4 4$ </td></tr><tr><td>S3</td><td>0.6</td><td>0.6</td><td>0.6</td><td> $4 . 2 8 \pm 0 . 1 4$ </td><td> $5 . 7 7 \pm 0 . 2 4$ </td><td> $6 . 0 5 \pm 0 . 4 8$ </td><td> $2 1 . 3 8 \pm 0 . 4 1$ </td><td> $2 6 . 2 0 \pm 0 . 4 2$ </td><td> $2 7 . 0 4 \pm 0 . 4 6$ </td><td> $1 7 . 1 0 \pm 0 . 3 2$ </td><td> $2 0 . 4 2 \pm 0 . 4 7$ </td><td> $2 1 . 0 0 \pm 0 . 3 4$ </td></tr><tr><td>S4</td><td>0.6</td><td>0.1</td><td>0.1</td><td> $1 4 . 8 8 \pm 0 . 4 6$ </td><td> $2 0 . 0 2 \pm 0 . 1 7$ </td><td> $2 0 . 9 4 \pm 0 . 1 7$ </td><td> $2 8 . 9 6 \pm 0 . 4 7$ </td><td> $3 3 . 4 7 \pm 0 . 2 1$ </td><td> $3 4 . 0 2 \pm 0 . 2 9$ </td><td> $1 4 . 0 9 \pm 0 . 4 0$ </td><td> $1 3 . 4 4 \pm 0 . 4 7$ </td><td> $1 3 . 0 9 \pm 0 . 2 8$ </td></tr><tr><td>S5</td><td>0.1</td><td>0.6</td><td>0.1</td><td> $1 4 . 8 9 \pm 0 . 2 8$ </td><td> $1 9 . 9 7 \pm 0 . 3 4$ </td><td> $2 0 . 8 9 \pm 0 . 3 5$ </td><td> $6 2 . 1 6 \pm 0 . 2 5$ </td><td> $7 2 . 6 8 \pm 0 . 4 9$ </td><td> $7 4 . 1 1 \pm 0 . 2 9$ </td><td> $4 7 . 2 8 \pm 0 . 1 8$ </td><td> $5 2 . 7 1 \pm 0 . 1 7$ </td><td> $5 3 . 2 2 \pm 0 . 1 2$ </td></tr><tr><td>S6</td><td>0.1</td><td>0.1</td><td>0.6</td><td> $2 1 . 3 7 \pm 0 . 4 8$ </td><td> $2 8 . 8 2 \pm 0 . 3 7$ </td><td> $3 0 . 1 0 \pm 0 . 1 1$ </td><td> $5 4 . 5 6 \pm 0 . 3 5$ </td><td> $6 5 . 8 4 \pm 0 . 2 0$ </td><td> $6 7 . 6 3 \pm 0 . 5 0$ </td><td> $3 3 . 1 9 \pm 0 . 3 8$ </td><td> $3 7 . 0 3 \pm 0 . 1 8$ </td><td> $3 7 . 5 3 \pm 0 . 4 9$ </td></tr></table>

Compared methods. We compare: Dense concatenation (the standard baseline employing naive concatenation of multimodal token sequences under a standard Transformer) (Alberts et al., 2024), and MM-Spectrum (our modalityaware sparse MoE encoder with structured expert subspaces, heterogeneous capacities, and computation-cost regularization). For a fair comparison, the decoder and non-MoE components are kept identical across methods.

To investigate whether MM-Spectrum mitigates negative transfer induced by spectral heterogeneity, we evaluate elucidation accuracy across all seven combinations of NMR, IR, and MS (unimodal, bimodal, and full-modality). We compare our method against a standard Dense baseline that uses early concatenation, keeping backbone capacities identical. As shown in Table 1, the single-modality baseline reveals significant imbalance, with NMR providing the strongest standalone constraints (69.83%). Crucially, the Dense concatenation baseline exhibits catastrophic negative transfer in the full-modality regime: instead of benefiting from additional evidence, its performance collapses to 44.29%,

Implementation Details. We implement all models using PyTorch on NVIDIA A40 (40GB) GPUs. The backbone is a standard Transformer Encoder-Decoder with $L = 6$ layers, $H \ = \ 8$ attention heads, and hidden dimension $d _ { \mathrm { m o d e l } } = 5 1 2$ . For the MM-Spectrum encoder, we configure the MoE layers with a total of E = 8 experts (partitioned into shared, specific, and interaction groups as described) and employ $\mathrm { T o p } { \cdot } k = 2$ gating with a capacity factor of $C = 1 . 2 5$ to balance load. During inference, we use beam search with a beam width of 10. Complete architectural hyperparameters, routing configurations, preprocessing settings, and decoding details are reported in Sections B.2, B.4 and C.1 of the appendix.

## 4.2. Modality Combination Study

significantly underperforming the unimodal NMR baseline. This confirms our hypothesis that naive fusion allows long, low-density modalities (IR/MS) to overwhelm optimization and dilute high-fidelity signals.

In contrast, MM-Spectrum yields consistent Pareto improvements across all settings. Most notably, it successfully reverses the full-modality collapse (44 $. 2 9 \%  7 6 . 0 4 \% )$ , transforming the additional noisy modalities from a source of interference into a source of synergy. The substantial gains in Top-1 accuracy indicate that our structured expert allocation effectively disentangles conflicting gradients, allowing the model to sharpen ranking precision by selectively integrating complementary evidence rather than succumbing to attention dilution. Additional comparisons against crossattention, contrastive-alignment, and gradient-modulation baselines are provided in Section C.2 of the appendix. And the large-scale evaluation on MMST is reported in Section C.3.

## 4.3. Missing-Modality Robustness

To evaluate model reliability under realistic instrumentation failures, we test elucidation performance across seven inference-time missingness settings (S0–S6), ranging from balanced dropout to heavy single-modality loss, without any test-time adaptation. As detailed in Table 2, the baseline exhibits brittle behavior with sharp performance drops as missingness increases, suggesting implicit co-adaptation that fails when specific inputs are absent. In contrast, MM-Spectrum demonstrates smoother degradation and superior candidate recoverability (Top-10). This resilience indicates that our modality-aware routing successfully establishes reconfigurable inference pathways, capable of dynamically re-allocating computational budget to the remaining reliable evidence rather than collapsing under distribution shift.

## 4.4. Generalization to Experimental Spectra

To evaluate the robustness and practical applicability of MM-Spectrum on empirical laboratory measurements, we conduct extensive evaluations on the Spectral Database for Organic Compounds (SDBS) (Saito & Kinugasa, 2011). We benchmark our model under two distinct paradigms: (1) trainingfrom scratch directly on the experimental dataset, and (2) utilizing a pre-training and fine-tuning protocol, where the network is initialized on simulated spectra and subsequently adapted to real laboratory observations.

![](images/3275748da292e6c0a80e9a87260043defee33abb4927bc67f79e9b1ab488f45b.jpg)

![](images/fdab2f4141897d38c7405444be6ebe21cd3fb0237653dbff89d883634cfcab9a.jpg)

![](images/025152714cdd508d12d1d2a51a82fed9ac68d73d0aeb1ec9aa16d4b32ba86caf.jpg)

Table 3. Stratified evaluation by molecular complexity. Top-K accuracy (%) across Heavy Atom Count (HAC) bins. MM-Spectrum yields larger relative improvements on more complex molecules.
<table><tr><td rowspan="2">Bucket</td><td rowspan="2">HAC range #Samples</td><td rowspan="2"></td><td colspan="3">Baseline</td><td colspan="3">MM-Spectrum</td><td colspan="3">Improvement</td></tr><tr><td> $\mathbf { T o p - 1 \% }$ </td><td> $\mathbf { T o p } { \cdot } 5 \%$ </td><td> $\mathbf { T o p - 1 0 \% }$ </td><td> $\mathbf { T o p - 1 \% }$ </td><td> $\mathbf { T o p } { \cdot } 5 \%$ </td><td> $\mathbf { T o p - 1 0 \% }$ </td><td> $\mathbf { T o p - 1 } \%$ </td><td> $\mathbf { T o p } { \cdot } 5 \%$ </td><td>Top-10%</td></tr><tr><td>HAC_S (Small)</td><td> $\leq 1 5$ </td><td>14821</td><td> $6 3 . 4 6 \pm 0 . 2 2$ </td><td> $7 8 . 4 8 \pm 0 . 3 7$ </td><td> $8 1 . 4 4 \pm 0 . 2 8$ </td><td> $8 2 . 2 8 \pm 0 . 4 2$ </td><td> $9 1 . 3 0 \pm 0 . 3 5$ </td><td> $9 2 . 4 5 \pm 0 . 3 2$ </td><td> $1 8 . 8 2 \pm 0 . 3 0$ </td><td> $1 2 . 8 2 \pm 0 . 3 8$ </td><td> $1 1 . 0 1 \pm 0 . 1 2$ </td></tr><tr><td>HAC_M (Medium)</td><td>16-25</td><td>35985</td><td> $4 6 . 8 9 \pm 0 . 2 9$ </td><td> $6 3 . 1 6 \pm 0 . 1 6$ </td><td> $6 6 . 0 4 \pm 0 . 3 9$ </td><td> $7 7 . 5 2 \pm 0 . 3 2$ </td><td> $8 8 . 0 3 \pm 0 . 3 7$ </td><td> $8 9 . 1 9 \pm 0 . 4 4$ </td><td> $3 0 . 6 3 \pm 0 . 3 4$ </td><td> $2 4 . 8 7 \pm 0 . 2 5$ </td><td> $2 3 . 1 5 \pm 0 . 3 2$ </td></tr><tr><td>HAC_L (Large)</td><td>≥ 26</td><td>28635</td><td> $2 9 . 9 3 \pm 0 . 3 5$ </td><td> $4 3 . 5 6 \pm 0 . 3 5$ </td><td> $4 6 . 4 8 \pm 0 . 4 5$ </td><td> $6 9 . 4 4 \pm 0 . 2 9$ </td><td> $8 1 . 5 0 \pm 0 . 2 3$ </td><td> $8 2 . 8 6 \pm 0 . 3 8$ </td><td> $3 9 . 5 1 \pm 0 . 3 1$ </td><td> $3 7 . 9 4 \pm 0 . 3 3$ </td><td> $3 6 . 3 8 \pm 0 . 1 6$ </td></tr></table>

Table 4. Generalization to real-world experimental spectra. Top-K SMILES elucidation accuracy (%) on the SDBS database under from-scratch and pre-training & fine-tuning (FT) protocols.
<table><tr><td rowspan="2">Protocol</td><td rowspan="2">Setting</td><td colspan="3">Top-K Accuracy (%)</td></tr><tr><td>Top-1%</td><td> $\mathbf { T o p } { \mathbf { - } } 5 \%$ </td><td>Top-10%</td></tr><tr><td rowspan="5">Scratch</td><td>NMR</td><td> $1 8 . 1 5 \pm 0 . 3 1$ </td><td> $3 1 . 7 4 \pm 0 . 2 5$ </td><td> $3 3 . 6 9 \pm 0 . 4 2$ </td></tr><tr><td>MS</td><td> $5 . 3 8 \pm 0 . 1 8$ </td><td> $1 1 . 0 7 \pm 0 . 2 2$ </td><td> $1 3 . 9 2 \pm 0 . 1 5$ </td></tr><tr><td>IR</td><td> $1 3 . 2 1 \pm 0 . 2 9$ </td><td> $2 5 . 4 8 \pm 0 . 3 4$ </td><td> $2 6 . 6 1 \pm 0 . 2 7$ </td></tr><tr><td>Baseline (Full)</td><td> $1 4 . 1 0 \pm 0 . 3 3$ </td><td> $2 7 . 1 3 \pm 0 . 4 1$ </td><td> $3 1 . 5 3 \pm 0 . 3 8$ </td></tr><tr><td>Ours (Full)</td><td> $\mathbf { 2 6 . 6 7 \pm 0 . 2 4 }$ </td><td> ${ \bf 3 8 . 9 1 \pm 0 . 1 9 }$ </td><td> $\mathbf { 4 7 . 3 7 \pm 0 . 2 6 }$ </td></tr><tr><td rowspan="2">Pretrain &amp;  $F T$ </td><td>Baseline (Full)</td><td> $3 7 . 5 6 \pm 0 . 3 6$ </td><td> $5 7 . 1 5 \pm 0 . 2 8$ </td><td> $5 9 . 8 4 \pm 0 . 3 1$ </td></tr><tr><td>Ours (Full)</td><td> $\pm \mathbf { 9 . 5 } 2 \pm \mathbf { 0 . 2 1 }$ </td><td> $\mathbf { 7 3 . 9 5 \ : \pm 0 . 2 5 }$ </td><td> ${ \bf 7 6 . 3 2 \pm 0 . 1 9 }$ </td></tr></table>

As summarized in Table 4, the single-modality baselines trained from scratch reveal the inherent complexity of uncurated experimental signals, with NMR remaining the most constraint-rich channel. Crucially, when combining all channels under full-modality inputs, the naive dense concatenation baseline exhibits a notable performance degradation compared to the standalone NMR setup, dropping from 18.15% to 14.10% Top-1 accuracy. In contrast, MM-Spectrum successfully circumvents this multi-source interference. When training from scratch, it completely reverses the full-modality degradation, boosting Top-1 accuracy to 26.67% and outperforming the dense multi-spectral baseline. Furthermore, under the pre-training and fine-tuning protocol, MM-Spectrum capitalizes on the large-scale prior representations to achieve a substantial performance leap.

## 4.5. Stratified Evaluation by Molecular Complexity

To probe performance beyond aggregate metrics, we stratify test molecules by Heavy Atom Count (HAC), a proxy for structural complexity and search-space ambiguity. We partition the data into small $( \mathrm { H A C } _ { S } ) _ { \mathrm { \Omega } }$ , medium $( { \mathrm { H A C } } _ { M } )$ , and large $( \mathrm { H A C } _ { L } )$ bins based on ground-truth structures. Results in Table 3 reveal a clear complexity-dependent degradation for the Dense concatenation baseline. In contrast, MM-Spectrum achieves consistent improvements across all strata, with the most substantial relative gains observed on $\mathrm { H A C } _ { M }$ and $\mathrm { H A C } _ { L }$ . This trend corroborates the efficacy of our capacity–utility principle: difficult instances trigger high-capacity and interaction-specific experts, effectively scaling computational resources to resolve greater posterior uncertainty. Complementary sensitivity analyses of expert capacity and routing sparsity are reported in Section C.5 of the appendix.

Figure 6. Representation evolution over training. Dense concatenation (top row) vs. MM-Spectrum (bottom row) at multiple training steps. MM-Spectrum exhibits more coherent and stable structure in the learned representation space, consistent with reduced cross-modal interference.  
![](images/654d55b2b710ada6797f7f0beed0fbd768c745237fdd3f98626bfe3f5a4f7b65.jpg)  
Figure 7. Layer-wise expert utilization. 100% stacked bar charts of Top-1 expert fractions per layer. Different layers display distinct utilization profiles, indicating layer-dependent specialization rather than trivial collapse or uniform routing.

## 4.6. Interpretability and Training Dynamics

Representation Evolution. Visualizing encoder representations (Figure 6) reveals that while the Dense concatenation baseline exhibits progressive mixing and instability—indicative of entanglement—MM-Spectrum develops a stable, well-separated organization. This confirms that modality-aware routing and structured subspaces effectively reduce interference and support disentangled evidence aggregation.

![](images/2e73647ce123f3114c0ecabddab1196c34126f8342a2709d2923f9404feecda2.jpg)  
(a) Step200

![](images/1402c056324e47d83601668e3176ecf5e0054d88192e7ececf07ab97b5df41f9.jpg)

![](images/12c38bcdf338b7ab5569c7ea702d4c8fb67e998737a2236ab325082be8493f9a.jpg)

(b) Step1200  
![](images/a7a0319436bd4a8cb0213cb6260893b1d6cb06b2c760979549dd99f5c1152771.jpg)  
(c) Step2400  
(d) Step5000  
Figure 8. Routing dynamics (Layer×Expert heatmaps). Top-1 expert occupancy across layers at multiple training steps. The routing pattern transitions from early exploration to structured specialization and stabilizes at later stages, providing direct evidence of emergent expert division of labor.

Routing Dynamics and Specialization. We further analyze expert behavior via occupancy heatmaps (Figure 8) and layer-wise profiles (Figure 7). Routing trajectories evolve from diffuse early exploration to stable, persistent hotspots, validating our curriculum-driven transition from coverage to specialization. Crucially, expert utilization varies distinctively across depths without degenerating into collapse or uniformity, implying that different layers autonomously acquire distinct functional roles (feature alignment vs. decision transformation) driven by the capacity–utility principle.

## 4.7. Ablation Studies

Table 5. Ablation: routing signals. Impact of Router-aware signals and Tag routing on Top-K accuracy (%).
<table><tr><td rowspan="2">Router-aware Tag-routing</td><td rowspan="2"></td><td colspan="3">Top-K Accuracy (%)</td></tr><tr><td>Top-1%</td><td>Top-5%</td><td>Top-10%</td></tr><tr><td>√</td><td>√</td><td> $\mathbf { 7 6 . 0 4 \ : \pm 0 . 2 0 }$ </td><td> ${ \bf 8 7 . 8 3 \pm 0 . 3 9 }$ </td><td> $\mathbf { 9 0 . 2 6 \pm 0 . 1 3 }$ </td></tr><tr><td>×</td><td>√</td><td> $7 2 . 5 7 \pm 0 . 2 6$ </td><td> $8 6 . 3 9 \pm 0 . 4 8$ </td><td> $8 8 . 9 3 \pm 0 . 3 9$ </td></tr><tr><td>√</td><td>×</td><td> $7 1 . 2 3 \pm 0 . 2 5$ </td><td> $8 6 . 0 6 \pm 0 . 1 6$ </td><td> $8 8 . 8 3 \pm 0 . 1 5$ </td></tr><tr><td>X</td><td>X</td><td> $7 0 . 1 1 \pm 0 . 3 0$ </td><td> $8 4 . 4 3 \pm 0 . 3 1$ </td><td> $8 6 . 0 6 \pm 0 . 3 4$ </td></tr></table>

Routing signals: Router-aware and Tag routing. We first assess how making modality cues explicit affects specialization. Table 5 shows that removing either Routeraware signals or Tag routing consistently reduces Top-K, with the largest relative degradation at Top-1. This pattern indicates that explicit routing supervision improves not only candidate coverage but also ranking precision, consistent with faster and cleaner formation of expert responsibilities.

Expert Subspaces and Dynamic Activation. Decomposing the expert space (Table 6) demonstrates that Interaction

Table 6. Ablation: expert subspaces and dynamic activation. Effects of Shared experts, Interaction experts, and Dynamic activation on Top-K accuracy (%).
<table><tr><td rowspan="2"></td><td rowspan="2">Shared Interaction Dynamic</td><td rowspan="2"></td><td colspan="3">Top-K Accuracy (%)</td></tr><tr><td> $\mathbf { T o p - 1 \% }$ </td><td> $\mathbf { T o p } { \mathbf { - } } 5 \%$ </td><td> $\mathbf { T o p - 1 0 \% }$ </td></tr><tr><td> $\checkmark$ </td><td>V</td><td>√</td><td> ${ \bf 7 6 . 0 4 \pm 0 . 1 5 }$ </td><td> $\mathbf { 8 7 . 8 3 \pm 0 . 3 2 }$ </td><td> $\mathbf { 9 0 . 2 6 \ : \pm 0 . 2 1 }$ </td></tr><tr><td> $\checkmark$ </td><td>√</td><td>X</td><td> $7 4 . 4 1 \pm 0 . 2 8$ </td><td> $8 6 . 5 4 \pm 0 . 4 1$ </td><td> $8 8 . 8 3 \pm 0 . 1 9$ </td></tr><tr><td> $\checkmark$ </td><td>X</td><td>√</td><td> $7 4 . 3 9 \pm 0 . 3 5$ </td><td> $8 6 . 3 4 \pm 0 . 2 2$ </td><td> $8 8 . 6 3 \pm 0 . 4 5$ </td></tr><tr><td> $\times$ </td><td> $\checkmark$ </td><td>√</td><td> $7 5 . 0 1 \pm 0 . 1 2$ </td><td> $8 6 . 9 0 \pm 0 . 3 3$ </td><td> $8 9 . 2 0 \pm 0 . 2 7$ </td></tr><tr><td>X</td><td> $\times$ </td><td> $\checkmark$ </td><td> $7 2 . 7 0 \pm 0 . 4 0$ </td><td> $8 4 . 6 5 \pm 0 . 1 8$ </td><td> $8 7 . 0 5 \pm 0 . 3 6$ </td></tr></table>

experts are pivotal for capturing cross-modal synergy, validating the need for dedicated parameter pathways over naive fusion. Shared experts concurrently facilitate redundant evidence transfer, while Dynamic activation further boosts robustness, confirming the benefit of adaptive computation allocation under sample-level variability.

Table 7. Ablation: heterogeneous experts and regularization. Effects of heterogeneous capacities (Hetero) and computationaware regularization on Top-K accuracy (%).
<table><tr><td rowspan="2"></td><td rowspan="2">Hetero. Regularizer</td><td colspan="3">Top-K Accuracy (%)</td></tr><tr><td>Top-1%</td><td>Top-5%</td><td>Top-10%</td></tr><tr><td>√</td><td>√</td><td> ${ \bf 7 6 . 0 4 \pm 0 . 1 4 }$ </td><td> ${ \bf 8 7 . 8 3 \pm 0 . 2 9 }$ </td><td> $\mathbf { 9 0 . 2 6 \ : \pm 0 . 1 8 }$ </td></tr><tr><td>√</td><td>×</td><td> $7 4 . 8 1 \pm 0 . 3 5$ </td><td> $8 6 . 6 1 \pm 0 . 2 2$ </td><td> $8 9 . 2 1 \pm 0 . 4 1$ </td></tr><tr><td>X</td><td>√</td><td> $7 4 . 0 7 \pm 0 . 1 9$ </td><td> $8 5 . 9 3 \pm 0 . 3 8$ </td><td> $8 8 . 6 5 \pm 0 . 2 6$ </td></tr><tr><td>X</td><td>X</td><td> $7 4 . 2 4 \pm 0 . 4 4$ </td><td> $8 6 . 0 8 \pm 0 . 1 5$ </td><td> $8 8 . 6 5 \pm 0 . 3 2$ </td></tr></table>

Heterogeneous Capacities and Regularization. Finally, Table 7 substantiates the synergy between heterogeneous expert capacities and computation-aware regularization. The combined improvement indicates that meaningful capacity diversity enables the regularizer to effectively match computational budget to token complexity. These findings strongly corroborate our proposed capacity–utility principle for conditional computation in multispectral elucidation.

## 5. Conclusion

We presented MM-Spectrum, a sparse MoE framework tailored to resolve the optimization instability in multimodal Molecular Structural Elucidation arising from severe spectral heterogeneity. By synergizing spectroscopy-aware compression, explicit modality routing, and structured expert subspaces with curriculum-driven heterogeneous computation, MM-Spectrum enforces Pareto-efficient specialization. This design mitigates imbalance-induced conflicts and successfully resolves full-modality collapse. While not intended to outcompete highly specialized single-modality models, MM-Spectrum yields substantial gains in the fullmodality regime, establishing a principled paradigm for extracting faithful cross-spectral synergy.

## Acknowledgements

We sincerely thank the anonymous reviewers for their insightful comments and constructive suggestions. This research is supported by the National Natural Science Foundation of China Project (No. 623B2086), Sponsored by CCF-GHFund (No. OF 2026005), Sponsored by CIPS-SMP-Zhipu Large Model Fund, Ant Group, and TeleAI of China Telecom.

## Impact Statement

This paper presents work whose goal is to advance the field of multimodal machine learning for scientific discovery, specifically in chemical structure elucidation. Potential societal consequences include accelerating drug discovery and materials science research. We do not foresee any negative ethical impacts or adverse societal consequences from this work.

## References

Alberts, M., Schilter, O., Zipoli, F., Hartrampf, N., and Laino, T. Unraveling molecular structure: A multimodal spectroscopic dataset for chemistry. Advances in Neural Information Processing Systems, 37:125780–125808, 2024.

Bengio, Y., Louradour, J., Collobert, R., and Weston, J. Curriculum learning. In Proceedings ofthe 26th annual international conference on machine learning, pp. 41–48, 2009.

Bohde, M., Manjrekar, M., Wang, R., Ji, S., and Coley, C. W. Diffms: Diffusion generation of molecules conditioned on mass spectra. arXiv preprint arXiv:2502.09571, 2025.

Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam: Efficient scaling of language models with mixture-ofexperts. In International Conference on Machine Learning, pp. 5547–5569. PMLR, 2022.

Duhrkop, K., Fleischauer, M., Ludwig, M., Aksenov,¨ A., Melnik, A., Meusel, M., Dorrestein, P., Rousu, J., and SIRIUS, S. B. 4: A rapid tool for turning tandem mass spectra into metabolite structure information., 2019, 16. DOI: https://doi. org/10.1038/s41592-019-0344-8. PMID: https://www. ncbi. nlm. nih. gov/pubmed/30886413, pp. 299–302.

Duhrkop, K., Shen, H., Mechtler, M., B¨ ocker, S., and Rousu,¨ J. Searching molecular structure databases with tandem mass spectra using csi: Fingerid. Proceedings of the National Academy of Sciences, 112(41):12580–12585, 2015.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022.

Guan, Y., Sowndarya, S. V. S., Gallegos, L. C., St. John, P. C., and Paton, R. S. Real-time prediction of 1h and 13c chemical shifts with dft accuracy using a 3d graph neural network. Chemical Science, 12(36):12012–12026, 2021.

Hu, F., Chen, M. S., Rotskoff, G. M., Kanan, M. W., and Markland, T. E. Accurate and efficient structure elucidation from routine one-dimensional nmr spectra using multitask machine learning. ACS Central Science, 10(11): 2162–2170, 2024.

Kanakala, G. C., Sridharan, B., and Priyakumar, U. D. Spectra to structure: contrastive learning framework for library ranking and generating molecular structures for infrared spectra. Digital Discovery, 3(12):2417–2423, 2024.

Klukowski, P., Riek, R., and Guntert, P. Machine learn-¨ ing in nmr spectroscopy. Progress in Nuclear Magnetic Resonance Spectroscopy, pp. 101575, 2025.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. GShard: Scaling giant models with conditional computation and automatic sharding. In Proceedings ofthe 9th International Conference on Learning Representations (ICLR), 2021.

Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., and Zettlemoyer, L. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pp. 6265–6274. PMLR, 2021.

Liang, P. P., Zadeh, A., and Morency, L.-P. Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. ACM computing surveys, 56(10):1–42, 2024.

Litsa, E. E., Chenthamarakshan, V., Das, P., and Kavraki, L. E. An end-to-end deep learning framework for translating mass spectra to de-novo molecules. Communications Chemistry, 6(1):132, 2023.

Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., and Chi, E. H. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1930–1939, 2018.

Priessner, M., Lewis, R. J., Lemurell, I., Johansson, M. J., Goodman, J., Janet, J. P., and Tomberg, A. Advancing structure elucidation with a flexible multi-spectral ai model. Angewandte Chemie, 138(2):e17611, 2026.

Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., and Houlsby, N. Scaling vision with sparse mixture of experts. In Advances in Neural Information Processing Systems, volume 34, pp. 8583–8595, 2021.

Saito, T. and Kinugasa, S. Development and release of a spectral database for organic compounds-key to the continual services and success of a large-scale database. Synthesiology English edition, 4(1):35–44, 2011.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017.

Shrivastava, A. D., Swainston, N., Samanta, S., Roberts, I., Wright Muelas, M., and Kell, D. B. Massgenie: a transformer-based deep learning method for identifying small molecules from their mass spectra. Biomolecules, 11(12):1793, 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in neural information processing systems, pp. 5998–6008, 2017.

Wang, W., Tran, D., and Feiszli, M. What makes training multi-modal classification networks hard? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12695–12705, 2020.

Weininger, D. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36, 1988.

Williams, P. L. and Beer, R. D. Nonnegative decomposition of multivariate information. arXiv preprint arXiv:1004.2515, 2010.

Wu, N., Jastrzebski, S., Cho, K., and Geras, K. J. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In International Conference on Machine Learning, pp. 24043–24055. PMLR, 2022.

Yang, Y. et al. Deep learning for vibrational spectral analysis: Recent progress and a practical guide. TrAC Trends in Analytical Chemistry, 143:116377, 2021.

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836, 2020.

Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A. M., Le, Q. V., Laudon, J., et al. Mixture-ofexperts with expert choice routing. In Advances in Neural Information Processing Systems, volume 35, pp. 7103– 7114, 2022.

Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., and Fedus, W. St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.

## Appendix: MM-Spectrum: Multimodal Multi-spectral Molecular Structural Elucidation with a Stable MoE Framework

## A. Theoretical Background and Derivations

## A.1. Notation and Task Objective

We consider multimodal spectra-to-structure generation with M modalities. For each modality $m \in \{ 1 , \ldots , M \}$ , the tokenized spectral sequence is $x ^ { ( m ) } = ( x _ { 1 } ^ { ( m ) } , \dots , x _ { L _ { m } } ^ { ( m ) } )$ with length $L _ { m }$ . Let $X = \{ x ^ { ( m ) } \} _ { m = 1 } ^ { M }$ denote the multimodal input and $y = ( y _ { 1 } , \dots , y _ { T } ) \in \mathcal { V }$ the target SMILES sequence. An encoder–decoder model parameterized by θ defines

$$
p _ { \theta } ( y \mid X ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( y _ { t } \mid y _ { < t } , X ) , \qquad \mathscr { L } _ { \mathrm { t a s k } } ( \theta ) = - \mathbb { E } _ { ( X , y ) \sim \mathcal { D } } \Big [ \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( y _ { t } \mid y _ { < t } , X ) \Big ] .\tag{9}
$$

(a) Existing Methods: Naive Concatenation  
![](images/fcd208f8116a6ef0aa614484725975ac392846f8ab2535d046e5a69e086d4f1b.jpg)  
(b) Our Method: MM-Spectrum with MoE-Encoder

![](images/c627b88390461a47f6f992863090929cae153c58729b76c8cd716a81da6bc67b.jpg)  
MoE Encoder: Dynamically routes modality embeddings to specialized experts via gating.

Figure 9. Visualizing the mechanism of failure versus the mechanism of synergy. (a) The Pathology of Naive Concatenation: Treating heterogeneous signals (NMR/IR/MS) as a uniform sequence leads to Modality Collapse. The shared parameter space succumbs to gradient conflicts (visualized as “Internal Chaos”), where high-redundancy modalities overwhelm high-density constraints, resulting in entangled representations and distorted molecular predictions. (b) The Solution via MM-Spectrum: We resolve this by introducing a Structured MoE Encoder. Through a modality-aware gating mechanism $G ( x )$ , the model dynamically disentangles information flows by routing tokens to functionally specialized modules—allocating Heavy Experts for complex topology inference and Light Experts for noise filtering—thereby restoring optimization stability and generation precision.

In this appendix we formalize: (i) why naive early-fusion concatenation can degrade in the full-modality regime, (ii) how modality-aware routing stabilizes specialization, and (iii) how structured experts and cost-aware regularization yield a Pareto-efficient allocation under multispectral imbalance.

## A.2. A Multi-Objective View of Multispectral Training

A useful lens is to treat multispectral learning as optimizing a sum of modality-induced objectives. Let $C _ { m } ( \theta )$ be the expected loss contribution “attributable” to modality m (e.g., the loss induced when modality m is present and the others are treated as nuisance factors). A generic multi-objective surrogate can be written as

$$
C ( \theta ) = \sum _ { m = 1 } ^ { M } w _ { m } C _ { m } ( \theta ) , \qquad w _ { m } \geq 0 , \sum _ { m } w _ { m } = 1 .\tag{10}
$$

Let $g _ { m } = \nabla _ { \theta } C _ { m } ( \theta )$ . Gradient compatibility between modalities i and $j$ can be measured by cosine similarity

$$
\cos ( g _ { i } , g _ { j } ) = { \frac { \langle g _ { i } , g _ { j } \rangle } { \| g _ { i } \| \ \| g _ { j } \| } } .\tag{11}
$$

Negative transfer occurs when cross-modal updates conflict, i.e., cos $( g _ { i } , g _ { j } ) < 0$ , so that improving one modality tends to degrade the other. In multispectral spectroscopy, conflicts are exacerbated by imbalance: modalities can differ by orders of magnitude in length $( L _ { m } )$ , noise, and per-token semantic density.

## A.3. Why Naive Concatenation Can Collapse Under Multispectral Imbalance

The standard baseline forms a single early-fusion sequence

$$
\boldsymbol { x } ^ { ( \mathrm { c a t } ) } = \mathrm { C o n c a t } \bigl ( \boldsymbol { x } ^ { ( 1 ) } , \ldots , \boldsymbol { x } ^ { ( M ) } \bigr ) , \qquad \boldsymbol { L } = \sum _ { m = 1 } ^ { M } L _ { m } .\tag{12}
$$

This design implicitly allocates attention budget and gradient updates roughly in proportion to token counts. When $L _ { m }$ is severely imbalanced, “long but weak” modalities can dominate the training signal even if their marginal utility per token is low.

A density-weighted intuition. Let $\rho _ { m }$ be an abstract (unobserved) measure of per-token information density for modality m with respect to the task. Under early fusion, the effective contribution of modality m to the overall update is influenced by both its token count and density. A simplified heuristic is that the expected magnitude of gradients contributed by modality m scales with the number of tokens that backpropagate through shared parameters:

$$
\mathbb { E } \bigl [ \bigl \| g _ { m } \bigr \| \bigr ] \propto L _ { m } \cdot \bar { \delta } _ { m } ,\tag{13}
$$

where $\hat { \delta } _ { m }$ summarizes average per-token backprop signal strength. In multispectral spectroscopy, it is common that $L _ { \mathrm { I R } } \gg L _ { \mathrm { N M R } }$ but ${ \bar { \delta } } _ { \mathrm { I R } } < { \bar { \delta } } _ { \mathrm { N M R } } , \mathrm { s o } L _ { m }$ can dominate even when $\rho _ { m }$ is low. This yields attention dilution (high-density constraints become “background”) and can also amplify cross-modal gradient conflicts, pushing the optimizer toward a suboptimal compromise solution.

Key implication. Full-modality improvement is not guaranteed by simply adding modalities. Instead, stable synergy requires an explicit mechanism that can (i) prevent low-density tokens from monopolizing updates, and (ii) structurally separate conflicting gradient components so that specialization can emerge.

## A.4. Modality-Aware Routing as an Explicit “Identity + Content” Decomposition

Standard MoE routing. A sparse MoE layer replaces the dense FFN with experts $\{ \mathrm { F F N } _ { e } \} _ { e = 1 } ^ { E }$ and a router that produces gating probabilities $p _ { i , e }$ for each token representation $h _ { i } \in \mathbb { R } ^ { d }$

$$
p _ { i , e } = \mathrm { s o f t m a x } \Big ( \frac { w _ { e } ^ { \top } h _ { i } } { \tau } \Big ) _ { e } , \qquad o _ { i } = \sum _ { e \in \mathrm { T o p K } ( p _ { i , : } , k ) } p _ { i , e } \mathrm { F F N } _ { e } ( h _ { i } ) .\tag{14}
$$

In heterogeneous spectroscopy, content-only routing forces the router to $i n f e r$ modality identity from token statistics, which is unnecessary and unstable when modalities have drastically different distributions.

MM-Spectrum: additive modality injection. For each token i, let m(i) be its modality index. We inject explicit modality information via a learnable tag embedding $e _ { m ( i ) }$ and a modality bias $b _ { m ( i ) } .$

$$
\tilde { h } _ { i } = h _ { i } + e _ { m ( i ) } + b _ { m ( i ) } .\tag{15}
$$

The router then gates on $\tilde { h } _ { i }$

$$
p _ { i , e } = \mathrm { s o f t m a x } \Big ( \frac { w _ { e } ^ { \top } \tilde { h } _ { i } } { \tau } \Big ) _ { e } , \qquad o _ { i } = \sum _ { e \in \mathrm { T o p K } ( p _ { i , : } , k ) } p _ { i , e } \mathrm { F F N } _ { e } ( h _ { i } ) .\tag{16}
$$

Interpretation: logit bias and stabilized coarse separation. Define logits $z _ { i , e } = \boldsymbol { w _ { e } ^ { \intercal } } \tilde { h } _ { i }$ . Then

$$
z _ { i , e } = w _ { e } ^ { \top } h _ { i } \ + \ w _ { e } ^ { \top } ( e _ { m ( i ) } + b _ { m ( i ) } ) .\tag{17}
$$

The second term is a modality-dependent expert preference that is disentangled from content. Geometrically, Eq. (15) induces a translation in the router input space, so tokens from different modalities become separable even when their content features overlap early in training. This yields a stable “coverage” behavior in early optimization (reducing collapse risk), while still allowing fine-grained content-based routing within each modality.

## A.5. Load Balancing and Entropy Regularization

Sparse MoE typically uses auxiliary losses to avoid expert collapse. Let ${ \cal K } _ { i } = \mathrm { T o p K } ( p _ { i , : } , k )$ be the activated set for token i in a minibatch of B tokens. Define expert importance and load:

$$
{ \mathrm { I m p } } _ { e } = \sum _ { i = 1 } ^ { B } p _ { i , e } , \qquad { \mathrm { L o a d } } _ { e } = \sum _ { i = 1 } ^ { B } \mathbf { 1 } [ e \in { \mathcal { K } } _ { i } ] .\tag{18}
$$

We use the squared coefficient of variation $\scriptstyle ( \mathbf { C V } ^ { 2 } )$ to penalize concentration:

$$
\mathcal { L } _ { \mathrm { b a l } } = \mathrm { C V } ^ { 2 } ( \mathrm { I m p } ) + \mathrm { C V } ^ { 2 } ( \mathrm { L o a d } ) = \frac { \mathrm { V a r } ( \mathrm { I m p } ) } { ( \mathbb { E } [ \mathrm { I m p } ] ) ^ { 2 } } + \frac { \mathrm { V a r } ( \mathrm { L o a d } ) } { ( \mathbb { E } [ \mathrm { L o a d } ] ) ^ { 2 } } .\tag{19}
$$

Additionally, to encourage exploration early in training, we regularize router entropy:

$$
\mathcal { L } _ { \mathrm { e n t } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { e = 1 } ^ { E } p _ { i , e } \log p _ { i , e } .\tag{20}
$$

## A.6. Stability-to-Specialization Curriculum: Scheduling λ(t) and $\tau ( t )$

In multispectral settings, strong balancing is beneficial early (preventing collapse) but can become harmful late (preventing specialization and synergy). We therefore schedule the strength of auxiliary regularizers and the router temperature over training steps t:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { c o s t } } \mathcal { L } _ { \mathrm { c o s t } } + \lambda ( t ) \big ( \mathcal { L } _ { \mathrm { b a l } } + \mathcal { L } _ { \mathrm { e n t } } \big ) , } \end{array}\tag{21}
$$

where $\lambda ( t )$ is annealed from a large value to a small value. We use a smooth cosine decay (other monotone decays are valid):

$$
\lambda ( t ) = \lambda _ { 0 } \cdot \frac { 1 } { 2 } \Big ( 1 + \cos ( \pi \cdot \operatorname * { m i n } ( 1 , t / T _ { \lambda } ) ) \Big ) .\tag{22}
$$

Similarly, we anneal temperature to move from exploration to confident routing:

$$
\tau ( t ) = \operatorname* { m a x } \Big ( \tau _ { \operatorname* { m i n } } , \tau _ { 0 } \cdot \exp ( - t / T _ { \tau } ) \Big ) .\tag{23}
$$

Three-phase dynamics. This induces an interpretable trajectory: (i) coverage phase: large λ(t) and higher $\tau ( t )$ encourage broad usage; (ii) alignment phase: decreasing $\lambda ( t )$ allows modality bias to form stable channels; (iii) specialization phase: $\mathcal { L } _ { \mathrm { t a s k } }$ and $\mathcal { L } _ { \mathrm { c o s t } }$ dominate, yielding sharp and cost-effective routing.

## A.7. Structured Expert Space via PID-Induced Inductive Bias

We adopt Partial Information Decomposition (PID) as an inductive bias to structure expert roles. Let R denote redundan information shared across modalities, $U _ { m }$ unique information of modality $m ,$ , and $S$ synergistic information:

$$
\operatorname { I } ( X ; y ) \approx R + \sum _ { m = 1 } ^ { M } U _ { m } + S .\tag{24}
$$

Guided by Eq. (24), we partition experts into three families:

$$
\mathscr { C } = \mathscr { C } _ { \mathrm { s h } } \cup \Big ( \bigcup _ { m = 1 } ^ { M } \mathscr { C } _ { m } \Big ) \cup \mathscr { C } _ { \mathrm { i n t } } ,\tag{25}
$$

where $\mathcal { C } _ { \mathrm { s h } }$ are shared experts (redundancy), $\mathcal { C } _ { m }$ are modality-specific experts (unique constraints), and $\mathcal { C } _ { \mathrm { i n t } }$ are interaction experts (synergy).

Lightweight structural regularizers. We impose weak, implementation-friendly constraints that encourage “redundantconsistent, unique-separable” behavior without explicitly estimating PID:

$$
\mathcal { L } _ { \mathrm { s t r u c t } } = \mathcal { L } _ { \mathrm { s h } } + \mathcal { L } _ { \mathrm { s e p } } .\tag{26}
$$

For shared experts, we encourage cross-modality consistency for representations associated with the same molecule:

$$
\mathcal { L } _ { \mathrm { s h } } = \mathbb { E } \Big [ \sum _ { e \in \mathcal { C } _ { \mathrm { s h } } } \big \| \mu _ { \mathrm { N M R } } ^ { ( e ) } - \mu _ { \mathrm { I R } } ^ { ( e ) } \big \| _ { 2 } ^ { 2 } + \big \| \mu _ { \mathrm { N M R } } ^ { ( e ) } - \mu _ { \mathrm { M S } } ^ { ( e ) } \big \| _ { 2 } ^ { 2 } \Big ] ,\tag{27}
$$

where $\mu _ { m } ^ { ( e ) }$ denotes the mean pooled expert-e output for modality m (within a sample or minibatch). For modality-specific experts, we enforce separability by a margin objective:

$$
\mathcal { L } _ { \mathrm { s e p } } = \mathbb { E } \Big [ \sum _ { m = 1 } ^ { M } \sum _ { e \in \mathcal { C } _ { m } } \operatorname* { m a x } \big ( 0 , \gamma - d ( \mu _ { m } ^ { ( e ) } , \mu _ { \neg m } ^ { ( e ) } ) \big ) \Big ] ,\tag{28}
$$

where $\mu _ { \lnot m } ^ { ( e ) }$ is the mean pooled output over other modalities and $d ( \cdot , \cdot )$ can be cosine distance or $\ell _ { 2 }$ distance. In practice, $\mathcal { L } _ { \mathrm { s t r u c t } }$ is assigned a small weight so it shapes the geometry without overriding the main task loss

## A.8. Heterogeneous Experts and Computation-Aware Regularization

Heterogeneous capacity spectrum. Different spectral tokens have different marginal utilities. We instantiate a spectrum of expert capacities via a per-expert descriptor $\kappa _ { e }$ (e.g., hidden width in FFN), allowing “heavy” and “light” experts:

$$
\mathrm { F F N } _ { e } ( h ) = W _ { 2 } ^ { ( e ) } \sigma ( W _ { 1 } ^ { ( e ) } h ) , \qquad \kappa _ { e } : = \dim ( W _ { 1 } ^ { ( e ) } h ) .\tag{29}
$$

Cost-aware routing as a soft constrained optimization. Associate each expert with a cost $c _ { e } \propto \kappa _ { e }$ . We penalize the expected routing mass sent to expensive experts:

$$
\mathcal { L } _ { \mathrm { c o s t } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { e = 1 } ^ { E } p _ { i , e } c _ { e } .\tag{30}
$$

Then minimizing $\mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { c o s t } } \mathcal { L } _ { \mathrm { c o s t } }$ is a Lagrangian relaxation of a constrained problem:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { t a s k } } ( \theta ) \quad \mathrm { s . t . } \quad \mathbb { E } \Big [ \sum _ { e } p _ { i , e } c _ { e } \Big ] \leq \mathcal { B } ,\tag{31}
$$

which encourages a Pareto-efficient allocation: high-utility tokens are allowed to use heavy experts, while low-utility redundant tokens are discouraged from monopolizing expensive capacity.

## A.9. Compute and Parameter Complexity

We provide a simple comparison between a dense Transformer FFN and a Top-k MoE FFN. Let the dense FFN cost per token be $C _ { \mathrm { { f f n } } }$ (FLOPs). In MoE, only k experts are executed, so ignoring dispatch overhead:

$$
C _ { \mathrm { m o e } } \approx k \cdot \bar { C } _ { \mathrm { e x p e r t } } + C _ { \mathrm { r o u t e r } } ,\tag{32}
$$

where $\bar { C } _ { \mathrm { e x p e r t } }$ is the average cost of an expert and $C _ { \mathrm { r o u t e r } }$ is small. If experts are heterogeneous, $\bar { C } _ { \mathrm { e x p e r t } }$ becomes routing-dependent; in that case $\mathcal { L } _ { \mathrm { c o s t } } \left( \mathrm { E q . } \left( 3 0 \right) \right)$ directly regularizes the expected compute.

Parameter-wise, MoE increases capacity approximately linearly with E:

$$
P _ { \mathrm { m o e } } \approx P _ { \mathrm { s h a r e d } } + E \cdot P _ { \mathrm { e x p e r t } } ,\tag{33}
$$

while keeping per-token compute controlled by k (not E).

## B. Additional Implementation Details and Reproducibility

## B.1. Tokenization and Compression Operators

To align sequence statistics across modalities, we apply modality-specific preprocessing $\phi _ { m } i$

$$
\tilde { x } ^ { ( m ) } = \phi _ { m } ( x ^ { ( m ) } ) , \qquad \tilde { X } = \{ \tilde { x } ^ { ( m ) } \} _ { m = 1 } ^ { M } .\tag{34}
$$

For high-density NMR, we preserve discrete constraints $( \phi _ { \mathrm { N M R } } \approx \mathrm { I d } )$ . For high-redundancy IR/MS, we use local binning (IR) and peak filtering (MS) to suppress background tokens. Modality tags are appended (or added as embeddings) so each token retains explicit modality identity.

Table 8. Modality preprocessing and sequence configuration (illustrative). Operators are chosen to reduce redundancy while preserving chemically critical evidence.
<table><tr><td>Modality</td><td>Token type</td><td>Compression  $\phi _ { m }$ </td><td>Max length</td></tr><tr><td>NMR  $( ^ { 1 } \mathrm { H } / ^ { 1 3 } \mathrm { C } )$ </td><td>symbolic / constraint-rich</td><td>identity / conservative parsing</td><td> $L _ { \mathrm { m a x } } ^ { \mathrm { N M R } }$ </td></tr><tr><td>IR</td><td>dense / autocorrelated</td><td>local binning / aggregation</td><td> $L _ { \mathrm { m a x } } ^ { \mathrm { I R } }$  max</td></tr><tr><td>MS</td><td>sparse peaks + noise tail</td><td>top-k peak filtering</td><td> $T . ^ { \mathrm { \tilde { M S } } ^ { - } }$  Lmax</td></tr></table>

## B.2. MoE Configuration and Routing Schedules

We use Top-k gating and a capacity factor to reduce dispatch overflow. The router temperature $\tau ( t )$ and regularizer strength λ(t) follow Eqs. (22)–(23). All ablations keep the decoder and non-MoE components fixed.

Table 9. Key MoE hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>#experts E</td><td>8</td></tr><tr><td>Top-k</td><td>2</td></tr><tr><td>Capacity factor C</td><td>1.25</td></tr><tr><td>Aux losses</td><td> $\mathcal { L } _ { \mathrm { b a l } } , \mathcal { L } _ { \mathrm { e n t } } , \mathcal { L } _ { \mathrm { c o s t } }$ </td></tr><tr><td>Schedules</td><td>λ(t) (cosine), τ(t) (exp or cosine)</td></tr><tr><td>Expert families</td><td>shared / modality-specific / interaction</td></tr></table>

## B.3. Missing-Modality Protocol

For missing-modality evaluation, we fix a trained model and only alter test-time inputs. Given missing ratios $( r _ { \mathrm { N M R } } , r _ { \mathrm { M S } } , r _ { \mathrm { I R } } )$ , we randomly drop (mask) the corresponding fraction of each modality’s tokens (or remove the modality channel), while preserving modality tags for remaining tokens. This evaluates whether routing can re-allocate compute to reliable evidence under distribution shift, without any test-time adaptation.

## B.4. Evaluation: Top-K Accuracy with Beam Search

We decode with beam search (beam width B) and report Top-K accuracy: a prediction is correct if the ground-truth SMILES appears among the top K candidates. Top-1 reflects single-shot ranking accuracy, whereas larger K values measure whether the correct structure remains recoverable within a candidate set.

## B.5. Data Preprocessing Details

For the dataset from Alberts et al. (2024), we applied the following modality-specific preprocessing:

• NMR: Discretized into symbolic tokens representing chemical shift ranges (0.1 ppm bins) and multiplicities. Max length truncated to 256.

• IR: Binned into 1024 distinct frequency intervals. Intensities are normalized to [0, 1].

• MS: Top-100 peak filtering is applied based on relative intensity.

## C. Detailed Experimental Setup and Results

## C.1. Hyperparameter Configuration

To facilitate full reproducibility of our results, we provide the comprehensive hyperparameter configuration in Table 10. These parameters were selected based on grid search performance on the validation set. All experiments were conducted on a computational node equipped with 4 × NVIDIA A40 (40GB) GPUs.

Table 10. Complete Hyperparameter Configuration. The settings cover model architecture, the specific MM-Spectrum MoE design, data preprocessing constraints, and optimization details.

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>Model ArchitectureLayers (L)Hidden Dimension (dmodel)Attention Heads (H)Feed-Forward Dimension</td><td rowspan=1 colspan=1>Transformer Encoder-Decoder6 (Encoder) / 6 (Decoder)51282048 (in non-MoE layers)</td></tr><tr><td rowspan=1 colspan=1>MM-Spectrum MoE</td><td rowspan=1 colspan=1>Total Experts (E)Experts per Token (Top-k)Expert Capacity Factor (C)Heavy Expert Hidden DimLight Expert Hidden DimRouter Type</td><td rowspan=1 colspan=1>8 (Partitioned: 2 Shared, 2 NMR, 2 IR/MS, 2 Interact)21.252048 (4× expansion)512 (1× expansion)Modality-Aware Dense Router</td></tr><tr><td rowspan=1 colspan=1>Data Preprocessing</td><td rowspan=1 colspan=1>NMR Max LengthIR/MS Max LengthIR Compression</td><td rowspan=1 colspan=1>256 (Symbolic tokens)1024 (Compressed continuous tokens)Local Binning (Window=4, Stride=4)</td></tr><tr><td rowspan=1 colspan=1>Optimization</td><td rowspan=1 colspan=1>HardwareOptimizerLearning RateWeight DecayBatch Size</td><td rowspan=1 colspan=1>4× NVIDIA A40 (40GB)AdamW (β1 = 0.9, β2 = 0.999, ∈ = 1e − 8)Peak 5 × 10−4 with Cosine Decay0.0164 per GPU (Global Batch = 256)</td></tr></table>

## C.2. Comparison with Alternative Multimodal Fusion Baselines

To verify that the massive improvements of MM-Spectrum stem from its structural conditional computation rather than generic multi-objective training or optimization tweaks, we benchmark our framework against several representative alternative fusion strategies. These include Cross-Attention Fusion, Contrastive Alignment (Liang et al., 2024), and gradient modulation algorithms such as PCGrad (Yu et al., 2020), Gradient Blending (Wang et al., 2020), and OGM-GE (Wu et al., 2022). All models are evaluated under the identical tri-modality (NMR + MS + IR) setup on the reference benchmark.

As illustrated in Table 11, optimization-level methods (e.g., Gradient Blending and OGM-GE) deliver marginal improvements over the naive Dense Concatenation baseline by alleviating gradient conflicts. However, representation alignment and generic gradient modulation alone remain insufficient to fundamentally counteract the severe multimodal imbalance induced by spectral scale and redundancy disparities. By contrast, MM-Spectrum achieves a decisive gain of +31.75% in Top-1% accuracy over the baseline, establishing the clear architectural necessity of physically grounded expert partitioning and modality-aware routing pathways.

## C.3. Large-Scale Generalization on the MMST Dataset

To rigorously assess the structural scalability of our model, we extend our experiments to the MultiModalSpectralTransformer (MMST) dataset (Priessner et al., 2026). This dataset presents an immense simulated corpus containing approximately 5, 997, 971 multi-spectral samples partitioned into 5, 275, 360 training examples, 659, 420 validation samples, and 63, 191 test observations.

The quantitative results for models trained from scratch are detailed in Table 12. Mirroring the empirical trajectory

Table 11. Comparison with alternative fusion baselines. Evaluation of Top-K SMILES elucidation accuracy (%) under full tri-modality inputs on the canonical benchmark dataset.
<table><tr><td>Method</td><td>Top-1%</td><td> $\mathbf { T o p } { \mathbf { - } } 5 \%$ </td><td>Top-10%</td></tr><tr><td>Dense Concatenation (Alberts et al., 2024)</td><td> $4 4 . 2 9 \pm 0 . 3 5$ </td><td> $5 9 . 3 9 \pm 0 . 3 1$ </td><td> $6 2 . 0 4 \pm 0 . 4 7$ </td></tr><tr><td>Cross-Attention Fusion</td><td> $4 2 . 1 5 \pm 0 . 4 1$ </td><td> $5 6 . 4 5 \pm 0 . 3 8$ </td><td> $6 1 . 9 1 \pm 0 . 5 2$ </td></tr><tr><td>Contrastive Alignment (Liang et al., 2024)</td><td> $3 9 . 0 1 \pm 0 . 2 8$ </td><td> $5 2 . 7 4 \pm 0 . 4 4$ </td><td> $5 7 . 0 6 \pm 0 . 3 9$ </td></tr><tr><td>OGM-GE (Wu et al., 2022)</td><td> $4 4 . 8 7 \pm 0 . 3 0$ </td><td> $6 0 . 9 3 \pm 0 . 2 7$ </td><td> $6 3 . 3 2 \pm 0 . 4 1$ </td></tr><tr><td>PCGrad (Yu et al., 2020)</td><td> $4 4 . 7 3 \pm 0 . 3 4$ </td><td> $6 1 . 2 4 \pm 0 . 4 9$ </td><td> $6 4 . 9 2 \pm 0 . 3 6$ </td></tr><tr><td>Gradient Blending (Wang et al., 2020)</td><td> $4 6 . 7 8 \pm 0 . 2 5$ </td><td> $6 3 . 4 5 \pm 0 . 3 3$ </td><td> $6 7 . 8 2 \pm 0 . 4 5$ </td></tr><tr><td>MM-Spectrum (Ours)</td><td> $\mathbf { 7 6 . 0 4 \ : \pm 0 . 1 2 }$ </td><td> ${ \bf 8 7 . 8 3 \pm 0 . 2 5 }$ </td><td> ${ \bf 9 0 . 2 6 \pm 0 . 1 3 }$ </td></tr></table>

observed on previous benchmarks, single-modality evaluations confirm that NMR delivers the most definitive chemical structure mappings, while IR and MS suffer from severe semantic sparsity when modeled independently. Under early-fusion concatenation, the dense baseline falls victim to severe optimization interference, causing its full-modality performance to slide to 65.71%, well below its standalone NMR counterpart (79.41%). Conversely, MM-Spectrum successfully circumvents this scaling pathology, optimizing multi-spectral parameters harmoniously to yield a peak accuracy of 87.35% Top-1, establishing a strict margin of +21.64% over the dense multi-modal equivalent.

Table 12. SMILES elucidation performance on the large-scale MMST dataset. All models are trained from scratch directly on the simulated data splits.
<table><tr><td>Modality Configuration</td><td>Method</td><td>Top-1%</td><td> $\mathbf { T o p } { \mathbf { - } } 5 \%$ </td><td>Top-10%</td></tr><tr><td>NMR-only  $\mathrm { ( ^ { 1 } H + ^ { 1 3 } C ) }$ </td><td>Dense Baseline</td><td> $7 9 . 4 1 \pm 0 . 1 6$ </td><td> $8 9 . 4 5 \pm 0 . 2 2$ </td><td> $9 4 . 7 1 \pm 0 . 1 1$ </td></tr><tr><td>MS-only</td><td>Dense Baseline</td><td> $2 5 . 1 3 \pm 0 . 4 2$ </td><td> $2 8 . 4 2 \pm 0 . 3 5$ </td><td> $3 1 . 2 8 \pm 0 . 5 1$ </td></tr><tr><td>IR-only</td><td>Dense Baseline</td><td> $1 9 . 3 5 \pm 0 . 3 8$ </td><td> $2 3 . 6 0 \pm 0 . 2 9$ </td><td> $2 7 . 0 4 \pm 0 . 3 4$ </td></tr><tr><td>Full-Modality Concatenation</td><td>Dense Baseline</td><td> $6 5 . 7 1 \pm 0 . 5 0$ </td><td> $8 5 . 0 7 \pm 0 . 3 4$ </td><td> $8 9 . 0 2 \pm 0 . 4 6$ </td></tr><tr><td>Full-Modality (Ours)</td><td>MM-Spectrum</td><td> $\mathbf { 8 7 . 3 5 \pm 0 . 1 9 }$ </td><td> $\mathbf { 9 5 . 8 7 \pm 0 . 1 4 }$ </td><td> $\mathbf { 9 8 . 7 2 \pm 0 . 1 2 }$ </td></tr></table>

## C.4. Robustness to the Complete Absence of an Entire Modality

In practical chemical discovery pipelines, complete spectral portfolios may be unavailable due to high instrument acquisition costs or sample-state incompatibilities. To evaluate the extreme bounds of robustness, we subject models trained on full tri-modality spectra to an unfavorable testing environment where one complete physical modality channel is entirely zeroed out during test-time inference.

As detailed in Table 13, the early-fusion Dense baseline demonstrates brittle co-dependency, showing massive performance degradation when any information pillar is dropped. Strikingly, when the pivotal NMR spectrum is completely removed, the baseline drops catastrophically to 18.65% Top-1 accuracy. Under identical missingness constraints, MM-Spectrum maintains a massive performance envelope, securing 39.24% Top-1 accuracy when NMR is missing, and holding above 73% accuracy when either MS or IR is fully omitted. This behavior indicates that our modality-aware router dynamically isolates degraded coordinate axes and safely re-routes tokens across unaffected specialized parameters.

Table 13. Inference evaluation under the complete absence of a single modality. Models are optimized on full modalities and evaluated with one modality entirely omitted at test time.
<table><tr><td>Omitted Modality Channel</td><td>Method</td><td>Top-1%</td><td>Top-5%</td><td>Top-10%</td></tr><tr><td rowspan="2">Mass Spectrometry (MS) Removed</td><td>Dense Baseline</td><td> $2 5 . 7 1 \pm 0 . 4 9$ </td><td> $2 8 . 5 2 \pm 0 . 3 6$ </td><td> $3 0 . 8 4 \pm 0 . 4 1$ </td></tr><tr><td>MM-Spectrum</td><td> $\mathbf { 7 3 . 9 4 \ : \pm 0 . 2 2 }$ </td><td> $\mathbf { 8 6 . 1 2 \pm 0 . 1 5 }$ </td><td> $\mathbf { 8 8 . 4 7 \pm 0 . 1 8 }$ </td></tr><tr><td rowspan="2">Infrared (IR) Removed</td><td>Dense Baseline</td><td> $3 1 . 5 2 \pm 0 . 3 8$ </td><td> $3 5 . 7 8 \pm 0 . 4 2$ </td><td> $3 7 . 7 9 \pm 0 . 3 5$ </td></tr><tr><td>MM-Spectrum</td><td> $\mathbf { 7 4 . 8 1 \pm 0 . 1 7 }$ </td><td> $\mathbf { 8 6 . 7 9 \ : \pm 0 . 2 0 }$ </td><td> $\mathbf { 8 9 . 0 5 \ : \pm 0 . 2 4 }$ </td></tr><tr><td rowspan="2">Nuclear Magnetic Resonance (NMR) Removed</td><td>Dense Baseline</td><td> $1 8 . 6 5 \pm 0 . 5 2$ </td><td> $2 0 . 4 6 \pm 0 . 4 7$ </td><td> $2 3 . 1 7 \pm 0 . 5 5$ </td></tr><tr><td>MM-Spectrum</td><td> $\mathbf { 3 9 . 2 4 \ : \pm 0 . 3 1 }$ </td><td> ${ \pm 9 . 3 6 \pm 0 . 2 8 }$ </td><td> ${ \bf 6 3 . 4 1 \pm 0 . 3 9 }$ </td></tr></table>

Table 14. Sensitivity to the number of experts E (Top-k = 2). We fix Top-k = 2 and vary the expert pool size. Best values are bold. The results show a consistent improvement from small E to a moderate regime, with performance saturating (or slightly degrading) when E becomes too large due to routing dilution. This supports our choice of E = 8 as the default setting.
<table><tr><td> $E$ </td><td>Top-1 ↑</td><td>Top-3 ↑</td><td> $\mathrm { T o p } { - } 5 \uparrow$ </td><td> $\mathrm { T o p - } 1 0 \uparrow$ </td><td>Latency ↓</td><td>Throughput ↑</td></tr><tr><td>2</td><td>73.15</td><td>85.02</td><td>87.10</td><td>89.05</td><td>19.8 ms</td><td>50.2 s/sample</td></tr><tr><td>4</td><td>74.82</td><td>86.95</td><td>88.45</td><td>90.12</td><td>20.1 ms</td><td>49.5 s/sample</td></tr><tr><td>8</td><td>76.04</td><td>87.83</td><td>90.26</td><td>91.80</td><td>21.3 ms</td><td>46.8 s/sample</td></tr><tr><td>16</td><td>75.88</td><td>87.60</td><td>90.15</td><td>91.75</td><td>24.5 ms</td><td>40.5 s/sample</td></tr></table>

Table 15. Sensitivity to Top-k routing (fix $E = 8 ) .$ . We fix $E = 8$ and vary Top-k. Best trade-off is achieved at Top-k = 2. Top-k = 1 limits expert composition capacity, while larger k increases compute and routing congestion without commensurate accuracy gains.

<table><tr><td>Top-k</td><td>Top-1 ↑</td><td>Top-3 ↑</td><td> $\mathrm { T o p } { - } 5 \uparrow$ </td><td>Top-10 ↑</td><td>Latency ↓</td><td>Throughput ↑</td><td>Activated Cost↓</td></tr><tr><td>1</td><td>73.50</td><td>85.20</td><td>87.35</td><td>89.20</td><td>13.5 ms</td><td>74.0 s/sample</td><td>1.0×</td></tr><tr><td>2</td><td>76.04</td><td>87.83</td><td>90.26</td><td>91.80</td><td>21.3 ms</td><td>46.8 s/sample</td><td>2.0×</td></tr><tr><td>4</td><td>76.12</td><td>87.85</td><td>90.30</td><td>91.85</td><td>38.6 ms</td><td>25.8 s/sample</td><td>4.0×</td></tr></table>

## C.5. Sensitivity Analysis on Expert Capacity and Top-k Routing

Experimental protocol. To study the sensitivity of conditional computation in MM-Spectrum, we vary (i) the number of experts E and (ii) the routing parameter Top-k while keeping all other hyperparameters fixed (optimizer, learning rate schedule, dropout, total steps, decoding settings, and data splits). Unless otherwise specified, we use the same training budget and evaluation protocol as in the main experiments. For efficiency, we additionally report resource-related statistics (e.g., inference latency/throughput and the relative compute proxy such as activated FFN FLOPs) to characterize the performance–efficiency trade-off.

Metrics. We report Top-K accuracy for $K \in \{ 1 , 3 , 5 , 1 0 \}$ (higher is better), and inference efficiency measured by through put (samples/s) or latency (ms/sample) (lower is better). When applicable, we also report the fraction of dropped/overflow tokens induced by limited expert capacity (capacity factor C) as an indicator of routing congestion.

Effect of expert pool size E. Table 14 reveals that increasing the expert pool size improves accuracy up to a saturation point. Concretely, moving from a small pool (e.g., E = 2 or 4) to a moderate pool (default E = 8) consistently improves Top-K performance, indicating that the multispectral elucidation task benefits from richer conditional capacity and stronger specialization. However, further increasing E beyond this regime (e.g., E = 16) yields diminishing returns and may even slightly degrade performance. We attribute this to two factors: (1) under-utilization and routing noise: with too many experts, the router must allocate probability mass across a larger set, making load balancing harder and increasing the chance of unstable or noisy assignments; (2) optimization and regularization pressure: larger expert pools intensify the burden of auxiliary balancing constraints and may lead to over-partitioning of representation space, which is undesirable under modality imbalance and limited training budget. Overall, these findings justify choosing E = 8 as the default configuration, which achieves the best accuracy while maintaining favorable inference efficiency.

Effect of Top-k. Table 15 demonstrates that Top-k directly controls the composition degree of conditional computation. With Top-k = 1, each token is restricted to a single expert, which reduces compute but also limits the model’s ability to combine complementary expertise across modalities (e.g., assigning shared/specific/interaction experts jointly for ambiguous spectral fragments). As a result, Top-k = 1 typically underperforms configurations that allow limited expert composition. Increasing Top-k improves accuracy initially, and Top-k = 2 achieves the best overall trade-off: it allows mild compositionality while keeping overhead small. In contrast, larger Top-k (e.g., k = 4) substantially increases activated expert computations, aggravates routing congestion under finite capacity factor C, and tends to introduce redundancy (multiple experts receiving similar tokens). Empirically, the additional compute does not translate into consistent Top-K gains, leading to a worse accuracy–efficiency frontier. Therefore, we adopt Top-k = 2 as the default setting throughout the paper.

Takeaway: a stable “capacity–utility” operating point. Combining Tables 14 and 15, we identify a stable operating point $( E = 8 , \mathrm { T o p } { - } k = 2 )$ that maximizes utility under modality imbalance. This operating point aligns with our capacity–utility principle: conditional capacity should be sufficient to enable meaningful specialization, yet constrained enough to avoid routing instability and unnecessary compute. These observations are consistent with the synergy discussed in Table 7, where heterogeneous capacities become most effective when the expert pool is neither under-parameterized nor excessively fragmented, enabling computation-aware regularization to allocate budget according to token complexity.

## C.6. Computational Cost and Complexity Analysis

Motivation. While multispectral inputs provide complementary structural evidence, they also increase computational burden due to longer concatenated sequences and heterogeneous noise densities across modalities. A naive early-fusion baseline (Dense concatenation Transformer) treats all tokens uniformly and therefore (i) pays quadratic attention cost on the concatenated length and (ii) wastes capacity on low-utility/noisy tokens. In contrast, MM-Spectrum introduces conditional computation via MoE routing, enabling selective capacity allocation and compute–utility trade-offs that are particularly important under modality imbalance.<sup>1</sup>

## C.6.1. THEORETICAL TIME/SPACE COMPLEXITY

Notation. Let the input token lengths for NMR/MS/IR be $L _ { \mathrm { N } } , L _ { \mathrm { M } } , L _ { \mathrm { I } }$ , respectively, and the concatenated length be $L _ { \mathrm { a l l } } = L _ { \mathrm { N } } + L _ { \mathrm { M } } + L _ { \mathrm { I } }$ . For a Transformer layer with model dimension d and FFN hidden dimension $d _ { \mathrm { f } }$ , we denote the self-attention cost as $\mathcal { O } ( L ^ { 2 } d )$ and FFN cost as $\mathcal { O } ( L d d _ { \mathrm { f f } } )$

Dense concatenation baseline. In early fusion, attention is computed on the full concatenated sequence, leading to per-layer time complexit

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { D e n s e } } = \mathcal { O } \big ( L _ { \mathrm { a l l } } ^ { 2 } d \big ) \ + \ \mathcal { O } ( L _ { \mathrm { a l l } } d d _ { \mathrm { f f } } ) , } \end{array}\tag{35}
$$

and activation memory scaling (dominantly from attention maps and intermediate activations) as

$$
\mathcal { M } _ { \mathrm { D e n s e } } \approx \mathcal { O } \big ( L _ { \mathrm { a l l } } ^ { 2 } \big ) + \mathcal { O } ( L _ { \mathrm { a l l } } d ) .\tag{36}
$$

This becomes unfavorable when multispectral sequences are long and modality imbalance introduces many low-utility tokens.

MM-Spectrum with conditional computation. MM-Spectrum replaces uniform FFN computation with a sparse MoE block. Let there be E experts and Top-k routing. Each token is processed by at most k experts. Denote the expert cost as cost(e) (e.g., proportional to expert FLOPs or parameter size). Then the expected MoE compute per token is

$$
\mathbb { E } \left[ \mathrm { c o s t } \right] = \sum _ { e = 1 } ^ { E } p ( e \mid x ) \mathrm { c o s t } ( e ) ,\tag{37}
$$

and the total MoE compute scales with $L _ { \mathrm { a l l } } \cdot \mathbb { E } [ \mathrm { c o s t } ]$ . To further encourage compute-aware specialization, we introduce a cost regularizer that penalizes routing probability mass on expensive experts, so that the router learns to spend heavy computation only when it yields sufficient marginal utility. This directly operationalizes a “performance vs. compute” accounting mechanism and tends to route low-density/noisy tokens to light experts while reserving heavy experts for hard tokens.

Capacity–utility principle. The above formulation suggests a capacity–utility principle: given a fixed budget, the model should allocate high-capacity computation to tokens/modalities with higher structural ambiguity or stronger cross-modal constraints, while using light computation for tokens that are either redundant or noisy. This principle is particularly compatible with multispectral elucidation, where token utility varies substantially across modalities and across molecule complexity regimes.

## C.6.2. EMPIRICAL PROFILING PROTOCOL

Metrics. We report: (i) #Params (total parameters); (ii) #Active Params (parameters effectively activated per token under Top-k); (iii) training throughput (tokens/sec); (iv) peak GPU memory (GB); (v) decoding latency (ms/sample) under fixed decoding configuration; and (vi) task performance (Top-1 / Top-10, %). All measurements are collected on the same hardware and software stack to ensure comparability.

Measurement settings (recommendation). Unless otherwise specified, we profile training with the same batch size and sequence truncation used in the main experiments, and profile inference with identical beam size, max decode length, and candidate set size. We recommend reporting mean±std over 3 runs and discarding the first several warm-up iterations for stable throughput.

## C.6.3. RESOURCE–ACCURACY TRADE-OFF

Table 16. Resource consumption and efficiency comparison. MM-Spectrum achieves significantly better accuracy while maintaining a favorable compute/memory profile compared to the Dense concatenation baseline. Note that while Total Parameters increase due to the expert pool, the Active Parameters (computational cost) remain comparable.

Profiling details. Hardware: 4× NVIDIA A40 (40GB); optimizer and training schedule follow Table 10. Throughput is measured during training with effective batch size 256. Latency is measured during inference with beam size 5.
<table><tr><td>Model</td><td>Top-1 ↑</td><td>Top-10 ↑</td><td>#Params ↓</td><td>#Active Params ↓</td><td>Throughput ↑</td><td>Peak Mem. ↓</td><td>Latency ↓</td></tr><tr><td>Dense (Concat)</td><td>44.29</td><td>62.04</td><td>110M</td><td>110M</td><td>4,850 tok/s</td><td>16.5 GB</td><td>18.2 ms</td></tr><tr><td>MM-Spectrum (MoE, Top-k=2)</td><td>76.04</td><td>90.26</td><td>450M</td><td>135M</td><td>4,210 tok/s</td><td>22.4 GB</td><td>21.3 ms</td></tr></table>

Discussion. Accuracy. As shown in Table 16, MM-Spectrum outperforms the Dense concatenation baseline in both Top-1 and Top-10 accuracy. This indicates that the proposed conditional-computation mechanism not only improves the best-guess generation quality (Top-1), but also increases the recoverability of correct structures within a candidate set (Top-10), which is crucial in practical elucidation pipelines.

Efficiency. Despite introducing more total parameters, MM-Spectrum activates only a sparse subset per token under Top-k routing, leading to a favorable active-parameter and compute footprint. Moreover, compute-aware regularization discourages unnecessary activation of heavy experts, thereby controlling the active computational footprint despite the increased total parameter capacity.

Why early fusion is inefficient under modality imbalance. In multispectral inputs, modality imbalance implies a skewed utility distribution: strong modalities (e.g., NMR) often dominate, while weaker modalities (e.g., MS/IR) contribute sparse but critical evidence. Dense concatenation is forced to process all tokens uniformly, paying quadratic attention cost on $L _ { \mathrm { a l l } }$ and risking negative transfer due to indiscriminate fusion. In contrast, MM-Spectrum can route tokens to modality-specialized or interaction experts, suppressing noisy token influence and allocating heavy computation primarily to ambiguous/hard regions, consistent with the capacity–utility principle.

Takeaway. Overall, MM-Spectrum provides a better Pareto frontier than Dense early fusion: it achieves higher elucidation accuracy while maintaining a more efficient resource profile in terms of activated capacity, memory footprint, and inference latency.

## D. Algorithmic Workflow

Algorithm 1 summarizes the complete training procedure of MM-Spectrum. It details the modality-specific preprocessing, the structured routing mechanism with explicit modality bias, and the multi-objective optimization strategy involving curriculum scheduling.

## E. Efficiency Analysis and Heterogeneous Experts

Why Heterogeneous Experts? In standard MoE, all experts share the same architecture (i.e., identical parameter count). However, in multispectral elucidation, processing a token from IR background noise requires significantly less computation than inferring a complex NMR coupling splitting. MM-Spectrum introduces two types of experts: Heavy $( d _ { f f n } = 2 0 4 8 )$

Algorithm 1 Training Procedure of MM-Spectrum   
1: Input: Multimodal dataset $\mathcal { D } = \{ ( \boldsymbol { X } ^ { ( m ) } , y ) \} _ { m = 1 } ^ { M }$ , where $m \in \{ \mathrm { N M R } , \mathrm { I R } , \mathrm { M S } \}$   
2: Parameters: Encoder-Decoder $\theta ,$ Router parameters $\{ W _ { r } , e _ { m } , b _ { m } \}$ , Experts $\{ \mathrm { F F N } _ { e } \} _ { e = 1 } ^ { E }$   
3: Hyperparameters: Total steps $T _ { m a x } ,$ Expert costs $\{ c _ { e } \}$ , Capacity factor $C ,$ Top-k.   
4: Initialization: Randomly initialize $\theta ,$ router, and experts. Define expert sets $\mathcal { C } _ { \mathrm { s h } } , \mathcal { C } _ { m } , \mathcal { C } _ { \mathrm { i n t } } .$   
5: for step $t = 1$ to $T _ { m a x }$ do   
6: 1. Spectrum-Aware Representation:   
7: for each modality m do   
8: Apply compression $\phi _ { m } \colon \tilde { x } ^ { ( m ) } \gets \phi _ { m } ( x ^ { ( m ) } )$ {Eq. 2: Binning/Filtering/Identity}   
9: end for   
10: Concatenate to form batch input $\tilde { X } = [ \tilde { x } ^ { ( \mathrm { N M R } ) } , \tilde { x } ^ { ( \mathrm { I R } ) } , \tilde { x } ^ { ( \mathrm { M S } ) } ]$   
11: 2. Spectrum-Aware Routing:   
12: for each token i in $\tilde { X }$ do   
13: Get content $h _ { i }$ and modality index $m ( i ) .$   
14: Inject modality bias: $\tilde { h } _ { i }  h _ { i } + e _ { m ( i ) } + b _ { m ( i ) }$ {Eq. 3}   
15: Compute logits: $s _ { i , e } \gets w _ { e } ^ { \top } \tilde { h } _ { i } .$   
16: Anneal temperature: $\tau \gets \tau ( t )$ {Curriculum Schedule}   
17: Calculate probabilities: $p _ { i , e } \gets$ softmax $( s _ { i , e } / \tau )$   
18: Select experts: ${ \displaystyle \mathcal { K } _ { i } \gets \mathrm { T o p K } ( p _ { i , : } , k ) }$   
19: Compute MoE output: $\begin{array} { r } { o _ { i } \gets \sum _ { e \in \mathcal { K } _ { i } } p _ { i , e } \cdot \mathrm { F F N } _ { e } ( h _ { i } ) } \end{array}$   
20: end for   
21: 3. Multi-Objective Loss Calculation:   
22: Compute task loss (Cross-Entropy): $\mathcal { L } _ { \mathrm { t a s k } }$   
23: Compute auxiliary losses:   
Balancing: $\mathcal { L } _ { \mathrm { b a l } }  \mathbf { C V } ^ { 2 } ( \mathrm { I m p } ) + \mathbf { C V } ^ { 2 } ( \mathrm { L o a d } ) .$   
Structural (PID): ${ \mathcal { L } } _ { \mathrm { s t r u c t } }  { \mathcal { L } } _ { \mathrm { s h } } + { \mathcal { L } } _ { \mathrm { s e p } }$ {Eq. 6}   
Cost-Aware: $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o s t } }  \frac { 1 } { B } \sum _ { i , e } p _ { i , e } c _ { e } } \end{array}$ {Eq. 7}   
24: Update curriculum weights $\lambda _ { \mathrm { b a l } } ( t )$   
25: Total Loss: $\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } }  \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { c o s t } } \mathcal { L } _ { \mathrm { c o s t } } + \lambda _ { \mathrm { b a l } } ( t ) ( \mathcal { L } _ { \mathrm { b a l } } + \mathcal { L } _ { \mathrm { e n t } } ) + \lambda _ { \mathrm { p i d } } \mathcal { L } _ { \mathrm { s t r u c t } } . } \end{array}$   
26: 4. Optimization:   
27: Compute gradients $\nabla { \mathcal { L } } _ { \mathrm { t o t a l } } .$   
28: Update parameters using AdamW.   
29: end for   
30: Output: Trained MM-Spectrum model.   
and Light $( d _ { f f n } = 5 1 2 ) .$   
Effect of Computation-Aware Regularization: By incorporating $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o s t } } = \frac { 1 } { B } \sum p _ { i , e } c _ { \epsilon } } \end{array}$ into the loss function, the model   
learns ”budget management.” Our experimental observations indicate:   
• In the late stages of training, Heavy Expert activation is concentrated on NMR tokens and the fingerprint region of IR   
spectra.   
• Light Experts primarily process low-intensity MS fragment peaks and smooth IR baseline regions.   
This mechanism allows MM-Spectrum to achieve an average inference FLOPs of only $\sim 4 0 \%$ compared to a Dense model   
of equivalent parameter scale, significantly reducing inference latency while improving accuracy.

![](images/f0ff0a4611c014eb00091b09b6642b9a23afde6d171cd4612f1c1bd31ebe31e9.jpg)  
Figure 10. Routing Evolution Dashboard. The top panel shows routing Uniformity (Entropy H), while the bottom panel shows Imbalance (Max/Mean load). Observation: A clear phase transition is visible at the end of the warmup. Imbalance drops initially (Coverage Phase) and then stabilizes at a non-zero level, indicating that the router has moved from ”random exploration” to ”structured specialization” rather than collapsing to a trivial solution.

![](images/6d34aae95a6f6a23c537dd26a52ff43678d298f4e82b407bb773069bac9f04db.jpg)  
Figure 11. Expert Composition Over Time. Stacked area plot of Top-1 expert usage. Different colors represent different experts. The emergence of stable bands (yellow/green/purple) confirms that distinct experts have successfully captured stable roles (e.g., Heavy vs. Light) driven by the capacity-utility principle.

![](images/932b1893aa1f8fcd29dec0d5f3497e1f2af0f919576d8a8c0a392ecec3f9f3c9.jpg)  
Figure 12. Thermodynamics of Routing. Scatter plot of Entropy (H) vs. Imbalance, colored by training step (purple→yellow). The trajectory shows a strong negative correlation (ρ = −0.889): as training progresses, the router trades high-entropy exploration for lower-entropy, imbalanced (specialized) allocation, consistent with Pareto optimization.