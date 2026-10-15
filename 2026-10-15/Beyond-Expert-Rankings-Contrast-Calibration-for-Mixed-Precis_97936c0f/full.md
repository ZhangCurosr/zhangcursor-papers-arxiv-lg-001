# Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization

Renjie Xie<sup>1,2</sup>, Juncheng Yang<sup>2</sup>, Aoting Hu<sup>3,2</sup>, Yan Liang<sup>5</sup> Mingxi Zhang<sup>1,2</sup>, Zheheng Hong<sup>4,2</sup>, Wei Xu<sup>5</sup>, Yanzhi Chen<sup>6,7</sup>

<sup>1</sup>Nanjing University of Posts and Telecommunications <sup>2</sup>Tylogi AI Lab / TAIL <sup>3</sup>Anhui University of Technology <sup>4</sup>Shanghai Jiao Tong University <sup>5</sup>Southeast University <sup>6</sup>Microsoft <sup>7</sup>University of Cambridge

renjie\_xie@njupt.edu.cn, juncheng\_yang@tylogi.com, aotinghu@ahut.edu.cn liang\_yan@seu.edu.cn, b23080111@njupt.edu.cn, zh2227@sjtu.edu.cn wxu@seu.edu.cn, yc514@cam.ac.uk

## Abstract

Mixture-of-experts (MoE) language models activate only a few experts per token, yet all expert weights must be stored. Mixed-precision quantization reduces this storage cost by assigning higher bit-widths to sensitive experts and lower bit-widths to resilient ones, but expert rankings alone do not determine how strongly to weight their errors. To address this gap, we propose NatScale, a contrast-calibrated allocation framework for mixed-precision MoE quantization. It uses a model-specific power transformation to calibrate gradient-based expert weights to an information-theoretic contrast target while preserving rankings. We jointly evaluate each expert’s gate, up and down con figurations to estimate its output error. We then combine these errors and calibrated weights with an existing prior based on weight spectra, variance and precision decay to select configurations under a storage budget. Across three models, NatScale achieves the lowest native-toquantized KL divergence and the highest toptoken agreement in all nine settings against five allocation baselines. It also yields the lowest perplexity in seven settings. These findings highlight the importance of objective-aware precision allocation.

## 1 Introduction

Sparse Mixture-of-Experts (MoE) language models decouple parameter count from computational cost by routing each token to a small subset of experts (Jiang et al., 2024; DeepSeek-AI et al., 2024). Although this routing limits per-token computation, all expert parameters must still be stored. Mixedprecision quantization offers an effective way to compress these models: under a strict memory budget, it allocates higher precision to sensitive weights and lower precision to robust ones.

Prior work uses gradients or curvature information to estimate how quantization errors affect the model’s loss, providing a basis for sensitivity-aware quantization (Kim et al., 2024, 2025). Choosing precisions under this budget requires two ingredients: estimating the quantization error of candidate configurations and evaluating the model’s sensitivity to those errors. Existing techniques rank experts by importance using diverse heuristics, including routing statistics and reconstruction errors (Huang et al., 2025a), block sensitivity combined with activations (Duanmu et al., 2025), dynamic importance updates during quantization (Deng et al., 2026), and weight spectra priors (Yang et al., 2026).

While these signals establish expert rankings, a ranking alone does not determine memory allocation: it indicates which experts are more important, but not how strongly to weight their errors. Consider two sensitivity score vectors: (2, 1) and (100, 1). Both vectors rank the first expert ahead of the second, but assign different relative weights to their errors. We refer to the magnitude of this relative separation between sensitivity scores as contrast. Under a global storage budget, contrast controls how strongly experts compete for storage. AlphaQ adapts the curvature of its spectral importance mapping to model-weight statistics (Yang et al., 2026). We study this weighting strength for gradient-based expert sensitivities.

In this work, we propose NatScale, an allocation framework designed to explicitly control sensitivity contrast in mixed-precision MoE quantization. NatScale applies a power transformation to gradient-based expert sensitivities, choosing the exponent so that normalized squared weights reach a prescribed Jeffreys (symmetric KL) divergence from a uniform distribution over positive-score experts. Because a power transformation is orderpreserving, this calibration leaves the expert rankings untouched, and the prescribed Jeffreys divergence expresses each model’s overall contrast on a common information-theoretic scale.

To score candidate configurations with these calibrated sensitivity weights, NatScale incorporates two additional design elements: (1) Joint Projection Error Measurement. Because gate and up outputs interact before the down projection, separately measured errors may miss their joint effect on the output. We measure output error with all three projections quantized jointly on fixed original-model inputs. (2) Spectral Structural Priors. The prior itself summarizes each block’s weight structure through three factors: weight spectra, variance and precision decay. Multiplying this prior by the joint errors and calibrated sensitivities yields a score for each configuration, and these scores determine the selection under a global storage budget.

The contributions of our work are:

• We propose NatScale for mixed-precision MoE quantization. Rather than using importance scores directly or applying fixed power transformations, NatScale calibrates expert sensitivity contrast to an explicit information-theoretic target, allowing for more effective allocation of quantization precision under a fixed budget.

• We show that expert rankings alone do not determine precision allocation: sensitivity contrast controls how strongly experts compete for storage. We further demonstrate that preserving native outputs and minimizing perplexity can favor different contrast levels.

• We evaluate NatScale on WikiText-2 across Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, and DeepSeek-V4-Flash-0731 at three storage targets, achieving the best native-output fidelity among the tested methods in all nine settings. We additionally evaluate FlashNext on AIME, GPQA, TruthfulQA, and LiveCodeBench.

## 2 Related Work

Post-Training Quantization. Post-training quantization (PTQ) reduces quantization error: GPTQ (Frantar et al., 2023) and AWQ (Lin et al., 2024) use calibration activations, while QuIP# (Tseng et al., 2024) and AQLM (Egiazarian et al., 2024) employ codebooks for low-bit quality. Sensitivity-aware quantization methods like BRECQ (Li et al., 2021), SqueezeLLM (Kim et al., 2024), GuidedQuant (Kim et al., 2025), and KronQ (Lee et al., 2026b) compute importance via gradients or curvature, with DASH-Q (Kim et al., 2026) mitigating noisy curvature. For precision allocation, SliM-LLM (Huang et al., 2025b) assigns bits to groups according to their salience, whereas CoopQ (Zhao et al., 2026) uses layer interactions to allocate precision across layers. These dense-model formulations do not directly specify expert-level choices under sparse routing.

MoE Quantization. Extending precision allocation to Mixture-of-Experts (MoE) models requires capturing expert importance and interactions. MC-MoE (Huang et al., 2025a) scores experts via activation frequency, routing probabilities, and error. MODE (Chen et al., 2026) estimates modalityspecific importance, and GEMQ (Deng et al., 2026) updates sensitivity dynamically. MxMoE (Duanmu et al., 2025) and FRI-MxMoE (Qi et al., 2026) allocate block-level precision using reconstruction costs or fuzzy interpolation, while AlphaQ (Yang et al., 2026) uses weight spectra and adapts the curvature of its importance mapping to model statistics. Recent work refines granularity: SFMoE (Qi et al., 2027) models cooperation, CAMERA (Xu et al., 2026) constructs micro-experts, and Q-Strata (Lee et al., 2026a) uses model-level objectives. NatScale calibrates gradient-based expert weights to a prescribed Jeffreys divergence. It jointly evaluates expert configurations, combining output errors with calibrated weights and AlphaQ’s structural factor, trading coordinate-level gradient alignment for global tractability.

Calibration and Robustness. Compression performance also depends on calibration data and robustness. Dataset selection critically impacts quality (Williams and Aletras, 2024), including selfgenerated text (Williams et al., 2025), sequence length (Oh and Oh, 2025), and language diversity (Chimoto et al., 2026). In multimodal models, MBQ (Li et al., 2025) uses output gradients to balance reconstruction across vision and language tokens. For MoE models, MoEQuant (Chen et al., 2025) improves calibration coverage through expert-balanced sampling and uses sample–expert affinities during quantization. Related work also addresses differences across data groups: Group DRO (Sagawa et al., 2020) optimizes worst-group loss, while FairQuant (Woergaard and Selvan, 2026) uses group-conditioned importance and fairness regularization for mixed-precision quantization. NatScale averages frequency-weighted RMS errors across folds. While its contrast target regulates weight separation, it remains dependent on calibration coverage and distribution shifts.

## 3 Methodology

## 3.1 Overall Framework

We formulate mixed-precision quantization for sparse Mixture-of-Experts (MoE) models as a constrained optimization problem (Jiang et al., 2024; DeepSeek-AI et al., 2024). Let $i = ( \ell , e ) \in \mathcal { I }$ index expert e in layer ℓ, and let $p \in \{ G , U , D \}$ denote its gate, up, or down projections. For each expert, a local configuration $c _ { i } = ( c _ { i , G } , c _ { i , U } , c _ { i , D } ) \in$ $\mathcal { C } _ { i }$ specifies an encoding format and quantized weights. Given an encoded-weight payload $C _ { i } ( c _ { i } )$ in exact bytes and a global storage budget B, our goal is to find an allocation $c = ( c _ { i } ) _ { i \in \mathcal { I } }$ that minimizes an empirical proxy for total model perturbation:

$$
\begin{array} { r l } { \underset { \{ c _ { i } \in \mathcal { C } _ { i } \} } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i \in \mathcal { T } } d _ { i } ( c _ { i } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \displaystyle \sum _ { i \in \mathcal { T } } C _ { i } ( c _ { i } ) \leq B , } \end{array}\tag{1}
$$

where $d _ { i } ( c _ { i } )$ is the expert’s proxy cost and nonrouted weights retain their native precision.

As illustrated in Figure 1, NatScale achieves this allocation through a principled three-stage pipeline. First, it derives gradient-based expert sensitivities and calibrates their relative separation, termed contrast, using an information-theoretic symmetric KL target. Second, it evaluates candidate distortions by jointly quantizing all three projections while incorporating a structural weight prior (Yang et al., 2026). Finally, it synthesizes these signals into composite expert costs and solves the global storage allocation using LP relaxation and feasible rounding.

## 3.2 Step 1: Sensitivity Calibration

To determine how strongly sensitive experts compete for storage, we measure the response of prediction loss to expert output changes. Let $v _ { i , t } =$ $\partial \mathcal { L } _ { \mathrm { C E } } / \partial y _ { i , t }$ denote the gradient of summed nexttoken cross-entropy with respect to expert output $y _ { i , t }$ at token position t. Over calibration folds, we compute the normalized gradient second moment

g<sub>i</sub>:

$$
\begin{array} { r l } & { G _ { i } = \frac { \sum _ { s } \sum _ { t \in { \cal T } _ { s , i } ^ { \mathrm { o r i g } } } \Vert v _ { i , t } \Vert _ { 2 } ^ { 2 } } { H _ { i } \operatorname* { m a x } \{ \sum _ { s } n _ { s , i } ^ { \mathrm { o r i g } } , 1 \} } , } \\ & { g _ { i } = \frac { G _ { i } } { \vert \mathcal { Z } \vert ^ { - 1 } \sum _ { j \in \mathcal { T } } G _ { j } } , } \end{array}\tag{2}
$$

where global mean normalization preserves relative inter-layer sensitivities. Here $H _ { i }$ is the expert output width, $\mathcal { T } _ { s , i } ^ { \mathrm { o r i g } }$ contains its routed token events in native calibration fold s, and $n _ { s , i } ^ { \mathrm { o r i g } } = | T _ { s , i } ^ { \mathrm { o r i g } } |$

Rather than relying on raw scores or heuristic powers, NatScale applies a power transformation $\dot { w } _ { i } = g _ { i } ^ { \beta _ { \tau } }$ on active experts ${ \mathcal { Z } } _ { + } = \{ i : g _ { i } > 0 \}$ and determines $\beta _ { \tau }$ from a target symmetric KL divergence $\tau > 0$ against a uniform distribution. For a candidate exponent $\beta \geq 0$ , the normalized squared weights and uniform reference are

$$
q _ { i } ( \beta ) = \frac { g _ { i } ^ { 2 \beta } } { \sum _ { j \in \mathcal { T } _ { + } } g _ { j } ^ { 2 \beta } } , \qquad u _ { i } = \frac { 1 } { | \mathcal { T } _ { + } | } .\tag{3}
$$

When the positive scores are nonconstant, each target $\tau > 0$ determines a unique exponent $\beta _ { \tau } > 0$ through

$$
\begin{array} { r l } & { ~ \mathcal { I } ( q ( \beta _ { \tau } ) , u ) } \\ & { = D _ { \mathrm { K L } } ( q ( \beta _ { \tau } ) \parallel u ) + D _ { \mathrm { K L } } ( u \parallel q ( \beta _ { \tau } ) ) } \\ & { = \tau , } \end{array}\tag{4}
$$

where target contrast τ is measured in nats. Zerogradient experts retain zero weight; calibration rejects empty support and constant positive scores. This formulation yields calibrated weights $w _ { i } =$ $g _ { i } ^ { \beta _ { \tau } }$ that strictly preserve expert rank ordering while setting the prescribed sensitivity contrast. We adopt $\tau = 1$ nat as our maximum-entropy reference default in evaluation.

## 3.3 Step 2: Joint Error Estimation

An expert’s gate and up projections nonlinearly interact before the down projection through gating operator $\Phi _ { i } \left( \mathbf { e . g . , S i L U } ( a ) \odot b \right)$ . We retain their error interactions by quantizing all three projections jointly on fixed replay inputs x:

$$
y _ { i } ^ { c } ( x ) = W _ { i , D } ^ { c _ { D } } \Phi _ { i } \Big ( W _ { i , G } ^ { c _ { G } } x , W _ { i , U } ^ { c _ { U } } x \Big ) .\tag{5}
$$

Letting $\Delta y _ { i , t } ( c ) = y _ { i } ^ { c } ( x _ { i , t } ) - y _ { i } ^ { 0 } ( x _ { i , t } )$ denote the output residual relative to native outputs, we define

![](images/ef1afd4eff5e3449d7f59f4c9de596703fb0cf61bd1409d4d52318df5172e959.jpg)  
Figure 1: Overview of NatScale. (a) Contrast Calibration: Maps gradient scores $g _ { i }$ via a power transformation $w _ { i } = g _ { i } ^ { \beta _ { \tau } }$ , where the exponent $\beta _ { \tau }$ is calibrated using a symmetric KL divergence target τ to control expert storage competition. (b) Joint Error Estimation: Evaluates quantization distortion by jointly quantizing gate, up, and down projections over replay inputs while incorporating structural priors. (c) Budgeted Allocation: Synthesizes calibrated weights, structural priors, and joint errors into expert costs, solving the global storage allocation by linear-programming (LP) relaxation and feasible rounding under a storage budget B.

the exposure-weighted RMS output error $E _ { i } ( c )$ using activation frequencies $f _ { s , i }$ across folds:

$$
\begin{array} { r l } & { \quad M _ { s , i } ( c ) = \frac { \sum _ { t \in { \cal T } _ { s , i } ^ { \mathrm { r e p } } } \| \Delta y _ { i , t } ( c ) \| _ { 2 } ^ { 2 } } { \operatorname* { m a x } \{ n _ { s , i } ^ { \mathrm { r e p } } , 1 \} H _ { i } } , } \\ & { \quad E _ { i } ( c ) = \displaystyle \sum _ { s } \pi _ { s } f _ { s , i } \sqrt { M _ { s , i } ( c ) } . } \end{array}\tag{6}
$$

Here $\mathcal { T } _ { s , i } ^ { \mathrm { r e p } }$ is the replay event set and $n _ { s , i } ^ { \mathrm { r e p } } = | T _ { s , i } ^ { \mathrm { r e p } } |$ Exposure is $f _ { s , i } = n _ { s , i } ^ { \mathrm { o r i g } } / T _ { s }$ , where $T _ { s }$ is the token count in fold $s ;$ the six folds receive equal weights $\pi _ { s } = 1 / 6$

To incorporate weight structure, we augment the output error with a structural prior $A _ { i } ( c )$ following AlphaQ (Yang et al., 2026), which incorporates spectral tail indices $\alpha _ { i , p } ,$ parameter variance $\mathrm { V a r } ( W _ { i , p } )$ , and effective bit precision decay:

$$
\begin{array} { l } { \displaystyle { A _ { i } ( c ) = \frac { 1 } { 3 } \sum _ { p \in \{ G , U , D \} } A _ { i , p } ( c _ { p } ) , } } \\ { \displaystyle { A _ { i , p } ( c _ { p } ) = \chi _ { i , p } ( c _ { p } ) \frac { \widetilde { \alpha } } { \alpha _ { i , p } } } } \\ { \displaystyle { \qquad \cdot \operatorname { V a r } ( W _ { i , p } ) 2 ^ { - 2 b _ { i , p } ( c _ { p } ) } . } } \end{array}\tag{7}
$$

Here $\widetilde { \alpha }$ is the median spectral tail index across all target projections, and $b _ { i , p } ( c _ { p } )$ uses nominal bits for integer formats and effective bits per weight for vector formats. The indicator $\chi _ { i , p } ( c _ { p } )$ is zero when that projection retains DeepSeek-V4-Flash-0731’s native MXFP4 encoding and one otherwise.

## 3.4 Step 3: Budget Allocation

With the calibrated sensitivity weights $w _ { i }$ , structural priors $A _ { i } ( c )$ , and joint output distortions $E _ { i } ( c )$ established, we formulate the composite expert proxy cost for configuration $c _ { i }$ as:

$$
\boxed  d _ { i } ( c _ { i } ) = w _ { i } A _ { i } ( c _ { i } ) E _ { i } ( c _ { i } ) \left| . \right.\tag{8}
$$

Under Lagrangian relaxation, the calibrated weight w<sub>i</sub> rescales the global byte price, making higherprecision configurations more attractive for critical experts. We construct each expert’s lower convex hull over cost-payload trade-offs and solve the LP relaxation, then round its solution to a feasible allocation. The gap to the LP optimum bounds the returned allocation’s optimization error for this proxy (Appendix C).

Implementation Note. Re-calibrating the contrast target τ or adjusting the global budget B only requires updating w and re-running the solver over pre-cached distortion terms $E _ { i } ( c )$ , requiring no additional model forward or backward passes. Further technical proofs and implementation details are provided in Appendices A, B, and C.

## 4 Theoretical Analysis

To substantiate NatScale’s design, this section provides theoretical answers to two fundamental questions:

1. Why is contrast calibration well-posed, and why choose $\tau = 1$ nat? Raw sensitivity distributions can differ in their overall contrast. NatScale puts this contrast on a common information scale by mapping scores through a power transformation to a prescribed symmetric KL divergence τ .

For any positive, non-constant gradient scores $g _ { i } ,$ the symmetric KL divergence $\mathcal { I } ( q ( \beta ) , u )$ is a monotonically increasing and continuous function of exponent $\beta \geq 0$ mapping from zero to infinity. Consequently, each target contrast $\tau > 0$ uniquely determines a single positive exponent $\beta _ { \tau }$ . Changing τ smoothly scales the relative separation (contrast) between expert weights without altering their rank ordering.

To set a default without tuning the contrast target, we map a two-parameter Weibull fit of amplitude scores $r _ { i } = \sqrt { g _ { i } }$ to a reference distribution whose squared amplitudes have maximum entropy at unit mean. Combining this map with the ordinary maximum-likelihood equation gives the following finite-sample identity:

Corollary 4.1 (Weibull Reference Target). For any positive, nonconstant amplitude scores $r _ { i } ,$ the unweighted two-parameter Weibull maximumlikelihood fit satisfies $\beta _ { 1 } = \widehat { k } / 4$ , yielding a calibrated contrast ofexactly $\mathcal { I } ( q ( \beta _ { 1 } ) , u ) = 1$ nat.

The derivation in Appendix A applies to any positive nonconstant sample, without assuming a Weibull data distribution. Thus, $\tau \ = \ 1$ nat acts as an information-theoretic anchor, while taskspecific performance determines whether a tighter or wider contrast is preferred.

2. How do calibrated weights regulate memory allocation? To understand how calibrated weights govern memory allocation during optimization, let $L _ { i } = A _ { i } E _ { i }$ represent the combined local distortion and structural prior cost for expert i. Under Lagrangian relaxation with a global byte price $\lambda \geq 0$ the expert-level selection objective minimizes:

$$
\operatorname* { m i n } _ { c _ { i } \in \mathcal { C } _ { i } } \quad w _ { i } L _ { i } ( c _ { i } ) + \lambda C _ { i } ( c _ { i } ) .\tag{9}
$$

Dividing this objective by the calibrated weight $w _ { i } > 0$ yields an equivalent minimization problem:

$$
\operatorname* { m i n } _ { c _ { i } \in \mathcal { C } _ { i } } \quad L _ { i } ( c _ { i } ) + \left( \frac { \lambda } { w _ { i } } \right) C _ { i } ( c _ { i } ) .\tag{10}
$$

Equation 10 gives contrast a direct allocation interpretation: larger $w _ { i }$ lowers the effective byte price $\lambda / w _ { i }$ for expert i.

When weights are close, their effective byte prices are similar; wider contrast separates these prices and can concentrate higher-precision storage on more heavily weighted experts. This relation explains how contrast changes storage competition, but it does not determine which target best preserves model outputs.

We therefore use $\tau = 1$ nat as a statistical reference and assess its effect empirically. Separately, the gap between the returned feasible allocation and the LP optimum gives an upper bound on its optimization error for the fixed empirical proxy (Appendix C).

## 5 Experiments

We evaluate NatScale on three MoE language models under three routed-weight storage budgets. Our experiments address four questions: (i) how well does contrast-calibrated allocation preserve model predictions relative to existing allocation strategies; (ii) how does the contrast target affect quality when expert rankings are held fixed; (iii) which score components contribute to performance; and (iv) whether the benefits extend to downstream tasks and whole-model compression.

## 5.1 Experimental Setup

Models. We study Qwen3.6-35B-A3B, Qwen3.8- Flash-Next, and DeepSeek-V4-Flash-0731, with reported total parameter counts of 35B, 125B, and 284B, respectively (Qwen Team, 2026a,b; DeepSeek-AI, 2026b,a). They activate approximately 3B, 6B, and 13B parameters per token and route each token to 8 of 256, 10 of 512, and 6 of 256 routed experts, respectively, in addition to a shared expert. The models span different sparse architectures and native weight formats. Details are specified in Appendix D.

Quantization and calibration. We allocate formats to the gate, up, and down projections of routed experts at nominal targets of 2.0, 2.5, and 3.0 bits per weight (bpw). All methods draw from the same precomputed candidate weights produced by MixedFlowQuant, while preserving their respective allocation rules. NatScale constrains encoded payload bytes and accounts for metadata separately; the adapted baselines include header overhead in their bit budgets. The targets specify nominal routed-weight storage. We use 512 fixed sequences of 2,048 tokens from Magpie, UltraChat, and WritingPrompts, divided into six equally weighted calibration folds (Xu et al., 2025; Ding et al., 2023; Fan et al., 2018). Gradients, routing frequencies, candidate weights, and joint output errors are computed once and reused across budgets and contrast targets. Unless stated otherwise, NatScale uses joint output RMS, equally weighted structural projection factors, and τ = 1 nat, without an importance matrix or router fine-tuning.

Table 1: WikiText-2 results at three nominal routed-weight budgets. Lower perplexity (PPL) and native-toquantized KL divergence (KLD) are better; higher native top-token agreement (Top1, %) is better. Bold indicates the best result in each column. All methods use the same candidate quantized weights, but retain method-specific allocation constraints and payload/header accounting (Appendix D). NatScale uses the fixed contrast target τ = 1 nat.
<table><tr><td>Qwen3.6-35B-A3B</td><td colspan="3">2.0 routed bpw</td><td colspan="3">2.5 routed bpw</td><td colspan="3">3.0 routed bpw</td></tr><tr><td>Method</td><td>PPL↓</td><td>KLD ↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td></tr><tr><td>MxMoE (Duanmu et al., 2025)</td><td>8.0690</td><td>0.2727</td><td>79.87</td><td>7.2930</td><td>0.1489</td><td>85.30</td><td>7.0630</td><td>0.0870</td><td>88.93</td></tr><tr><td>FRI-MxMoE (Qi et al., 2026)</td><td>8.4293</td><td>0.3118</td><td>78.14</td><td>7.9069</td><td>0.2359</td><td>80.90</td><td>7.6321</td><td>0.1992</td><td>82.55</td></tr><tr><td>AlphaQ (Yang et al., 2026)</td><td>7.7790</td><td>0.2047</td><td>81.76</td><td>7.3119</td><td>0.1236</td><td>85.70</td><td>6.9932</td><td>0.0671</td><td>89.49</td></tr><tr><td>MODE (Chen et al., 2026)</td><td>8.4724</td><td>0.3291</td><td>78.30</td><td>7.6407</td><td>0.1901</td><td>83.68</td><td>7.2792</td><td>0.1196</td><td>87.21</td></tr><tr><td>GEMQ (alloc.) (Deng et al., 2026)</td><td>7.9142</td><td>0.2189</td><td>81.89</td><td>7.3333</td><td>0.1160</td><td>87.01</td><td>7.1582</td><td>0.0762</td><td>89.85</td></tr><tr><td>NatScale (ours)</td><td>7.6767</td><td>0.1988</td><td>82.98</td><td>7.1507</td><td>0.0984</td><td>87.94</td><td>7.0149</td><td>0.0575</td><td>90.72</td></tr><tr><td>Qwen3.8-Flash-Next</td><td colspan="3">2.0 routed bpw</td><td colspan="3">2.5 routed bpw</td><td colspan="3">3.0 routed bpw</td></tr><tr><td>Method</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td></tr><tr><td>MxMoE (Duanmu et al., 2025)</td><td>6.1777</td><td>0.5873</td><td>73.14</td><td>5.3389</td><td>0.4109</td><td>78.66</td><td>4.9619</td><td>0.2765</td><td>82.69</td></tr><tr><td>FRI-MxMoE (Qi et al., 2026)</td><td>5.5050</td><td>0.4766</td><td>75.78</td><td>5.2422</td><td>0.4055</td><td>77.79</td><td>5.1490</td><td>0.3607</td><td>79.04</td></tr><tr><td>AlphaQ (Yang et al., 2026)</td><td>5.1147</td><td>0.3445</td><td>79.42</td><td>4.7987</td><td>0.2487</td><td>82.82</td><td>4.8261</td><td>0.1381</td><td>86.72</td></tr><tr><td>MODE (Chen et al., 2026)</td><td>5.5368</td><td>0.4446</td><td>76.93</td><td>4.9840</td><td>0.2594</td><td>82.48</td><td>4.7954</td><td>0.1663</td><td>86.04</td></tr><tr><td>GEMQ (alloc.) (Deng et al., 2026)</td><td>6.0587</td><td>0.5370</td><td>73.64</td><td>5.2986</td><td>0.3382</td><td>79.56</td><td>4.9891</td><td>0.2243</td><td>83.50</td></tr><tr><td>NatScale (ours)</td><td>5.0678</td><td>0.3293</td><td>80.22</td><td>4.8344</td><td>0.1941</td><td>84.86</td><td>4.7430</td><td>0.1243</td><td>87.67</td></tr><tr><td>DeepSeek-V4-Flash-0731</td><td colspan="3">2.0 routed bpw</td><td colspan="3">2.5 routed bpw</td><td colspan="3">3.0 routed bpw</td></tr><tr><td>Method</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 ↑</td></tr><tr><td>MxMoE (Duanmu et al., 2025)</td><td>21.1285</td><td>1.6657</td><td>58.43</td><td>16.4595</td><td>1.3704</td><td>62.85</td><td>9.8755</td><td>0.8292</td><td>71.11</td></tr><tr><td>FRI-MxMoE (Qi et al., 2026)</td><td>12.5689</td><td>1.1318</td><td>65.65</td><td>9.7246</td><td>0.8756</td><td>69.59</td><td>7.4664</td><td>0.6025</td><td>74.56</td></tr><tr><td>AlphaQ (Yang et al., 2026)</td><td>9.0467</td><td>0.7834</td><td>71.61</td><td>6.3970</td><td>0.4146</td><td>79.38</td><td>5.6550</td><td>0.2259</td><td>84.88</td></tr><tr><td>MODE (Chen et al., 2026)</td><td>15.2631</td><td>1.3185</td><td>63.07</td><td>9.1905</td><td>0.7947</td><td>71.26</td><td>6.4961</td><td>0.4273</td><td>78.63</td></tr><tr><td>GEMQ (alloc.) (Deng et al., 2026)</td><td>8.9568</td><td>0.8201</td><td>70.05</td><td>6.1106</td><td>0.3652</td><td>81.01</td><td>5.5451</td><td>0.2103</td><td>85.98</td></tr><tr><td>NatScale (ours)</td><td>8.3647</td><td>0.7293</td><td>72.15</td><td>5.9615</td><td>0.3361</td><td>81.57</td><td>5.4883</td><td>0.1936</td><td>86.16</td></tr></table>

Baselines. We adapt five allocation methods to these shared candidates: MxMoE (Duanmu et al., 2025), FRI-MxMoE (Qi et al., 2026), AlphaQ (Yang et al., 2026), MODE (Chen et al., 2026), and GEMQ (Deng et al., 2026). They respectively emphasize reconstruction sensitivity, interpolated reconstruction costs, weight-structure priors, source-dependent expert exposure, and gradient-weighted perturbations. The GEMQ comparison evaluates its allocation criterion without GPTQ, progressive quantization, or router finetuning; its Qwen3.8-Flash-Next and DeepSeek-V4- Flash-0731 results use corrected gradient scoring. The adaptations are specified in Appendix D.

Evaluation. We measure perplexity (PPL), nativeto-quantized KLD, and native-model top-token agreement (Top1) on WikiText-2 (Merity et al., 2016). Within each model, all methods use the same test stream and native reference. We score 255 next-token positions (256–510) in each independent 512-token window. Unlike the fixedinput local error estimates used to construct allocation costs, evaluation runs the complete quantized model, allowing quantization-induced changes to propagate through hidden states and routing decisions.

## 5.2 Quantization Performance

Performance on WikiText-2. Table 1 shows that NatScale consistently preserves native-model predictions across architectures and storage budgets. With the fixed τ = 1 contrast target, it achieves the lowest KLD and highest Top1 among the adapted allocation baselines in every model–budget setting. It also yields the lowest PPL in seven of the nine settings and improves all three metrics over the adapted GEMQ allocation throughout. The improvements are especially clear on Qwen3.8-Flash-Next: at the middle budget, NatScale reduces AlphaQ’s KLD from 0.2487 to 0.1941 while increasing top-token agreement from 82.82% to 84.86%. The results show consistent native-output preservation across the three tested architectures.

Divergence versus Likelihood. The PPL results qualify this advantage. AlphaQ obtains slightly lower perplexity on Qwen3.6-35B-A3B at the highest budget and on Qwen3.8-Flash-Next at the middle budget, although NatScale still achieves lower KLD and higher Top1 in both cases. This difference is expected because KLD measures agreement with the native output distribution, whereas perplexity measures likelihood assigned to observed next tokens. A quantized model can therefore more closely reproduce its native counterpart without attaining the best perplexity. These comparisons show stronger native-output fidelity at the fixed contrast target.

![](images/d25e922f93e8ec0177b799bfb820ca3cf7f6eeaa234bbd2d4dc2649efbb0f7e7.jpg)  
Figure 2: Sensitivity to the contrast target τ. Changes in KLD (%) and Top1 (percentage points) are measured relative to NatScale at $\tau = 1$ for each model and nominal routed-bpw budget. Solid curves show NatScale allocations at different targets; same-color horizontal dashed lines show the corresponding AlphaQ reference. The vertical dotted line marks the fixed default.

Downstream task evaluation. We further test whether the WikiText-2 gains carry over to reasoning, factuality, and coding tasks without taskspecific refitting. Table 3 reports Qwen3.8-Flash-Next at 2.5 routed bpw with thinking disabled. NatScale obtains the highest point estimate on three of four tasks: AIME 2025 (68.33%), GPQA Diamond (78.28%), and TruthfulQA binary (94.05%). MODE scores highest on LiveCodeBench v6 (62.86% versus NatScale’s 61.14%). Task rankings thus differ from native-output fidelity on WikiText-2. Evaluation protocols are given in Appendix G.

## 5.3 Ablation Studies

Expert weight contrast. To isolate the effect of contrast, we vary $\tau \in \{ 0 . 5 , \sqrt { 0 . 5 } , 1 , \sqrt { 2 } , 2 \}$ while retaining each model’s sensitivity scores, candidate weights, local errors, and storage ceiling. Each target sets the separation of expert weights on the same information scale across models. Expert rankings remain fixed, so the sweep isolates the effect of weighting strength on allocation quality.

Figure 2 shows that contrast materially affects output fidelity. On DeepSeek-V4-Flash-0731 at

2.0 bpw, increasing τ from 1 to 2 reduces KLD from 0.7293 to 0.6780 (7.03%) and increases Top1 by 1.02 percentage points. At 3.0 bpw, however, its smallest KLD among the five targets occurs at $\tau = \sqrt { 0 . 5 } .$ . Across all nine settings, $\tau = 2$ attains the smallest KLD in five, while $\tau = 1$ yields lower PPL than $\tau \ = \ 2$ in eight (Table 4). We retain one nat as the common statistical reference across models and budgets. The sweep shows that a fixed expert ranking can lead to different quality tradeoffs as contrast changes.

Score components. We remove local RMS error, precision-dependent decay, gradient weights, gradient smoothing, or routing-exposure magnitude in turn. All variants reuse identical candidate weights, calibration statistics, and payload ceilings. Removing gradient weights or bit decay increases KLD and decreases Top1 in every model–budget setting; removing bit decay also worsens PPL in all nine. At 2.5 bpw, each of the five removals degrades both KLD and Top1. Removing local RMS or gradient smoothing improves all three metrics for DeepSeek-V4-Flash-0731 at 2.0 bpw, but degrades them at its higher budgets. The complete results (Appendix F) support the roles of gradient weighting and precision decay.

Joint versus separate projection errors. To examine the effect of projection interactions, we compare joint error measurement with the sum of independently measured gate, up, and down projection errors, keeping all other allocation conditions fixed. As shown in Table 8, neither approach consistently dominates across storage budgets. Separate scoring performs slightly better at 2.0 bpw, while higher budgets exhibit mixed results across KLD and Top1. Nevertheless, joint measurement captures the combined effects of projection perturbations that separate measurements may overlook.

Table 2: Whole-model Qwen3.8-Flash-Next comparison by Unsloth Dynamic (UD) reference tier. NatScale uses each tier’s UD byte ceiling. GSQ-RCO release sizes differ (<sup>†</sup>). Total size includes fixed tensors and metadata, with GB defined as 10<sup>9</sup> bytes. Top1 is native-model agreement (%). UD scores are externally reported, while NatScale and GSQ-RCO are evaluated locally using the same WikiText-2 stream and native reference. Bold indicates the lowest KLD and highest Top1 within each tier. Additional details appear in Appendix H.
<table><tr><td rowspan="3">Method</td><td colspan="9">UD reference tier (NatScale byte ceiling)</td></tr><tr><td colspan="3">IQ1_M</td><td colspan="3">Q2_K_XL</td><td colspan="3">IQ3_XXS</td></tr><tr><td>Size (GB)</td><td>KLD↓</td><td>Top1 ↑</td><td>Size (GB)</td><td>KLD↓</td><td>Top1 ↑</td><td>Size (GB)</td><td>KLD↓</td><td>Top1 ↑</td></tr><tr><td>UD (reported)</td><td>74.5</td><td>0.314739</td><td>79.691</td><td>78.9</td><td>0.224607</td><td>82.715</td><td>82.0</td><td>0.165120</td><td>85.414</td></tr><tr><td>GSQ-RCO†</td><td>75.840</td><td>0.221337</td><td>82.998</td><td></td><td></td><td></td><td>83.618</td><td>0.137169</td><td>86.898</td></tr><tr><td>NatScale (ours)</td><td>74.529</td><td>0.196894</td><td>84.721</td><td>78.859</td><td>0.146297</td><td>86.543</td><td>81.952</td><td>0.124359</td><td>87.623</td></tr></table>

<sup>†</sup> GSQ-RCO uses the released IQ3\_XXS and IQ3\_S checkpoints in the first and third groups, respectively. Their sizes differ from the UD reference budgets.

Table 3: Downstream performance on Qwen3.8- Flash-Next at 2.5 routed bpw (%). Thinking is disabled. AIME is mean response accuracy on AIME 2025; GPQA is GPQA Diamond; TQA is TruthfulQA binary accuracy; and LCB is LiveCodeBench v6 pass@1. GEMQ denotes the allocation-only adaptation. Bold indicates the highest point estimate per task.
<table><tr><td>Method</td><td>AIME</td><td>GPQA</td><td>TQA</td><td>LCB</td></tr><tr><td>MxMoE</td><td>63.33</td><td>73.23</td><td>93.54</td><td>57.14</td></tr><tr><td>FRI-MxMoE</td><td>50.00</td><td>67.68</td><td>92.03</td><td>51.43</td></tr><tr><td>AlphaQ</td><td>62.50</td><td>73.74</td><td>93.04</td><td>58.86</td></tr><tr><td>MODE</td><td>60.83</td><td>74.75</td><td>93.42</td><td>62.86</td></tr><tr><td>GEMQ (alloc.)</td><td>15.83</td><td>63.13</td><td>93.80</td><td>47.43</td></tr><tr><td>NatScale (ours)</td><td>68.33</td><td>78.28</td><td>94.05</td><td>61.14</td></tr></table>

We further evaluate both approaches on downstream tasks using Qwen3.8-Flash-Next at 2.5 routed bpw with thinking disabled. Joint scoring outperforms separate scoring on three of four benchmarks, with particularly notable improvements on AIME and GPQA, while separate scoring performs better on LiveCodeBench (Table 9). These results support joint measurement as a reasonable default for capturing projection interactions, although its benefits remain task-dependent. Allocation patterns. We visualize expert-level precision and layer-level storage at 2.5 bpw for all three models in Appendix E. NatScale assigns different effective precisions to experts within the same layer, a variation obscured by layer-averaged budgets. The heatmaps, cumulative storage distributions, and layer-budget curves describe how the competing allocation rules distribute a global budget across experts and layers. They show how NatScale distributes precision within and across layers.

## 5.4 Whole-Model Storage and Quality

We further evaluate NatScale under whole-model storage constraints by extending precision allocation to shared experts and selected attention and gated-residual projections on Qwen3.8-Flash-Next. Unlike the routed-only experiments, this evaluation uses candidates fitted with the UD importance matrix and storage budgets aligned with Unsloth Dynamic (UD) reference tiers. We compare against UD and released GSQ-RCO checkpoints (Unsloth, 2026; ISTA-DASLab, 2026).

Table 2 shows that NatScale consistently achieves lower KLD and higher Top1 agreement than the available comparators across all three storage tiers. Notably, compared with GSQ-RCO IQ3\_XXS, NatScale reduces model size by 1.31 GB while lowering KLD by 11.04%. These results suggest that contrast-calibrated allocation remains effective beyond routed experts, offering better preservation of native outputs under practical storage constraints. The comparisons use different inference backends, while UD results are externally reported; further protocol details are provided in Appendix H.

## 6 Conclusion

We presented NatScale, a mixed-precision MoE allocation framework that explicitly calibrates the contrast between expert sensitivities instead of relying on their rankings alone. It combines a model-specific, rank-preserving power transformation with joint gate, up, and down output errors and a structural precision prior to allocate weight formats under a storage budget. At the fixed one-nat target, NatScale achieves the lowest KLD and highest top-token agreement in all nine comparisons with five adapted baselines, and the lowest PPL in seven. Downstream and whole-model evaluations provide additional evidence of its utility. The contrast sweeps show that native-output fidelity and perplexity can favor different targets.

## Limitations

While effective, NatScale requires additional computation to estimate expert sensitivity and select precision levels, adding to the cost of preparing a quantized model. Its allocation decisions also depend on how well the calibration data reflect the inputs encountered during deployment. Changes in input distribution or expert usage may therefore affect the quality of the resulting allocation. In addition, the error estimates used to guide precision selection are approximations and may not fully capture interactions between quantized components or changes in downstream task performance. Future work could explore more efficient calibration procedures and evaluate the stability of precision allocations across different data distributions. It could also examine whether calibration statistics can be reused when adapting a model to related application domains. Another direction is to study how the choice and amount of calibration data affect precision selection in task-specific settings.

## References

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola´ Jovanovic, and Martin Vechev. 2025.´ MathArena: Evaluating LLMs on uncontaminated math competitions. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Stephen Boyd and Lieven Vandenberghe. 2004. Convex Optimization. Cambridge University Press.

Yuanteng Chen, Nanxin Zeng, Peisong Wang, Zhilei Liu, Yuantian Shao, Shiqiang Lang, Tao Liu, Chuangyi Li, Qinghao Hu, Gang Li, Jing Liu, and Jian Cheng. 2026. MODE: Modality-decomposed expert-level mixed-precision quantization for MoE multimodal LLMs. arXiv preprint arXiv:2606.17118. EMNLP 2026 Main Conference; arXiv version 2.

Zhixuan Chen, Xing Hu, Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, and Jiangyong Yu. 2025. MoEQuant: Enhancing quantiza-

tion for mixture-of-experts large language models via expert-balanced sampling and affinity guidance. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 8245–8260. PMLR.

Everlyn Asiko Chimoto, Mostafa Elhoushi, and Bruce Bassett. 2026. Calibrating beyond English: Language diversity for better quantized multilingual LLMs. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4822–4838. Association for Computational Linguistics.

Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Maximilian Kleineg-´ ger, and Dan Alistarh. 2026. GSQ: Highlyaccurate low-precision scalar quantization for LLMs via gumbel-softmax sampling. arXiv preprint arXiv:2604.18556.

DeepSeek-AI. 2026a. DeepSeek-V4-Flash-0731. Hugging Face model card. Accessed 11 October 2026.

DeepSeek-AI. 2026b. DeepSeek-V4: Towards highly efficient million-token context intelligence. Preprint, arXiv:2606.19348.

DeepSeek-AI et al. 2024. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

Jianing Deng, Song Wang, Dongwei Wang, Zijie Liu, Tianlong Chen, Huanrui Yang, and Jingtong Hu. 2026. GEMQ: Global expert-level mixedprecision quantization for MoE LLMs. arXiv preprint arXiv:2605.23078.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051.

Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. 2025. MxMoE: Mixed-precision quantization for MoE with accuracy and performance co-design. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 14793–14806.

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. 2024. Extreme compression of large language models via additive quantization. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 12284–12303. PMLR.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings

of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations.

Michael Helcig and Dan Alistarh. 2026. Model compression with exact budget constraints via riemannian manifolds. arXiv preprint arXiv:2605.00649.

Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and Xiaojuan Qi. 2025a. Mixture compressor for mixtureof-experts LLMs gains more. In International Conference on Learning Representations.

Wei Huang, Haotong Qin, Yangdong Liu, Yawei Li, Qinshuo Liu, Xianglong Liu, Luca Benini, Michele Magno, Shiming Zhang, and Xiaojuan Qi. 2025b. SliM-LLM: Salience-driven mixed-precision quantization for large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 25672–25692. PMLR.

ISTA-DASLab. 2026. Qwen3.8-Flash-Next: GSQ-RCO GGUFs. Hugging Face model card, Deep Algorithms and Systems Lab, Institute of Science and Technology Austria. Accessed 10 October 2026.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida I. Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Jaemin Kim, Sungkyun Kim, Junyeol Lee, and Jiwon Seo. 2026. Robust ultra low-bit post-training quantization via stable diagonal curvature estimate. In Proceedings ofthe Sixth European Workshop on Machine Learning and Systems, pages 274–285.

Jinuk Kim, Marwa El Halabi, Wonpyo Park, Clemens JS Schaefer, Deokjae Lee, Yeonhong Park, Jae W. Lee, and Hyun Oh Song. 2025. GuidedQuant: Large language model quantization via exploiting end loss guidance. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 30011–30037.

Sehoon Kim, Coleman Richard Charles Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, and Kurt Keutzer. 2024. SqueezeLLM: Dense-and-sparse quantization. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 23901–23923. PMLR.

Deokjae Lee, Sihun Chu, and Hyun Oh Song. 2026a. Q-Strata: Hierarchical bit allocation for mixedprecision quantization of mixture-of-experts LLMs. arXiv preprint arXiv:2608.30564. Accepted to EMNLP 2026, Main Conference.

Donghyun Lee, Yuhang Li, Ruokai Yin, and Priyadarshini Panda. 2026b. KronQ: LLM quantization via Kronecker-factored Hessian. In Conference on Language Modeling. Version 2.

Shiyao Li, Yingchun Hu, Xuefei Ning, Xihui Liu, Ke Hong, Xiaotao Jia, Xiuhong Li, Yaqi Yan, Pei Ran, Guohao Dai, Shengen Yan, Huazhong Yang, and Yu Wang. 2025. MBQ: Modality-balanced quantization for large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4167–4177.

Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. 2021. BRECQ: Pushing the limit of post-training quantization by block reconstruction. In International Conference on Learning Representations.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware weight quantization for ondevice LLM compression and acceleration. In Proceedings of Machine Learning and Systems, volume 6, pages 87–100.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252. Association for Computational Linguistics.

Enes Makalic and Daniel F. Schmidt. 2023. Maximum likelihood estimation of the Weibull distribution with reduced bias. Statistics and Computing, 33:69.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843.

Jaehoon Oh and Dokwan Oh. 2025. Beyond fixedlength calibration for post-training compression of LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 19355– 19366. Association for Computational Linguistics.

Hua-Chen Qi, Rui-Yu Zhuo, Bo-Wen Shi, Fei Chao, Xiang Chang, Changjing Shang, and Qiang Shen. 2027. Synergistic mixed-precision quantization for MoE

LLMs: Integrating cooperative Shapley attribution with profiling-few fuzzy rule interpolation. Information Sciences, 759:124025.

Huachen Qi, Ruiyu Zhuo, Bowen Shi, Xiang Chang, Fei Chao, Changjing Shang, and Qiang Shen. 2026. Profiling-free mixed-precision quantization for MoE LLMs via fuzzy rule interpolation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21484–21499. Association for Computational Linguistics.

Qwen Team. 2026a. Qwen3.6-35B-A3B. Hugging Face model card. Accessed 11 October 2026.

Qwen Team. 2026b. Qwen3.8-Flash-Next. Hugging Face model card. Accessed 10 October 2026.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof Q&A benchmark. In Conference on Language Modeling.

Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. 2020. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations. ArXiv version 2.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. 2024. QuIP#: Even better LLM quantization with Hadamard incoherence and lattice codebooks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 48630–48656. PMLR.

Unsloth. 2026. Qwen3.8-Flash-Next: How to run locally. Unsloth Documentation. Quantization Analysis. Accessed 10 October 2026.

Miles Williams and Nikolaos Aletras. 2024. On the impact of calibration data in post-training quantization and pruning. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10100– 10118. Association for Computational Linguistics.

Miles Williams, George Chrysostomou, and Nikolaos Aletras. 2025. Self-calibration for language model quantization and pruning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10149–10167. Association for Computational Linguistics.

Thomas Woergaard and Raghavendra Selvan. 2026. FairQuant: Fairness-aware mixed-precision quantization for medical image classification. arXiv preprint arXiv:2602.23192.

Yuzhuang Xu, Xu Han, Yuanchi Zhang, Yixuan Wang, Yijun Liu, Shiyu Ji, Qingfu Zhu, and Wanxiang Che. 2026. CAMERA: Multi-matrix joint compression for MoE models via micro-expert redundancy analysis. Proceedings of the AAAI Conference on Artificial Intelligence, 40(32):27395–27404.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. 2025. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In International Conference on Learning Representations.

Wanqi Yang, Yuexiao Ma, Alexander Conzelmann, Xiawu Zheng, Michael W. Mahoney, T. Konstantin Rusch, and Shiwei Liu. 2026. AlphaQ: Calibrationfree bit allocation for mixture-of-experts quantization. arXiv preprint arXiv:2606.04980.

Junchen Zhao, Ali Derakhshan, Jayden Hyman, Junhao Dong, Sangeetha Abdu Jyothi, and Ian Harris. 2026. CoopQ: Cooperative game inspired layerwise mixed precision quantization for LLMs. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 7566–7578. Association for Computational Linguistics.

## Appendix Contents

A Contrast Calibration 12   
A.1 The Energy Reference . . 12   
A.2 Entropy-Regularized Reweighting 12   
A.3 Solving for the Exponent . 13   
A.4 The Weibull Case 13   
A.5 Weight Entropy and Power Invariance 13   
B Residual Modeling and Sensitivity 14   
C Implementation Details 15   
D Evaluation Setup 17   
E Precision Allocation 18   
F Design Ablations 19   
F.1 Contrast Targets . 19   
F.2 Component Ablations 19   
F.3 Joint and Separate Errors 20   
G Downstream Evaluation 21   
H Whole-Model Allocation 22   
H.1 Allocation Units and Scores 23   
H.2 Storage Budget . 26   
H.3 Evaluation and Results 26

## A Contrast Calibration

This section supplement the contrast calibration introduced in Section 3.2 and proves the guarantees stated in Section 4. On the positive support $\mathcal { T } _ { + }$ , let $N = | \mathbb { Z } _ { + } |$ and assume $x _ { i } = \log g _ { i }$ is nonconstant. The analysis uses the log-partition function and its associated probability distribution:

$$
\begin{array} { c } { \displaystyle \psi ( t ) = \log \left( \frac { 1 } { N } \sum _ { i } e ^ { t x _ { i } } \right) , } \\ { \displaystyle q _ { i } ^ { ( t ) } = \frac { e ^ { t x _ { i } } } { \sum _ { j } e ^ { t x _ { j } } } . } \end{array}\tag{11}
$$

These definitions give $\psi ^ { \prime } ( t ) = \mathbb { E } _ { q ^ { ( t ) } } x , \psi ^ { \prime \prime } ( t )$ $\operatorname { V a r } _ { a ^ { ( t ) } } ( x ) ~ > ~ 0 , ~ u ~ = ~ q ^ { ( 0 ) }$ , and $\bar { \boldsymbol { q } } ( \beta ) ~ = ~ \boldsymbol { q } ^ { ( 2 \beta ) }$ We write the change in expected log sensitivity as $\Delta ( t ) = \psi ^ { \prime } ( t ) - \psi ^ { \prime } ( 0 )$ throughout the analysis.

## A.1 The Energy Reference

To derive the energy reference behind the default target in Corollary 4.1, recall that the second moment $g _ { i }$ corresponds to amplitude $r _ { i } = \sqrt { g _ { i } }$ . After calibration, $q _ { i } = w _ { i } ^ { 2 } / \sum _ { j } w _ { j } ^ { 2 }$ describes how the corresponding energy is distributed across experts; at $\beta = 1 / 2 , q _ { i } = g _ { i } / \sum _ { j } g _ { j }$ . The following result identifies the reference distribution for a nonnegative energy variable with a fixed mean.

Proposition A.1 (Maximum-entropy energy reference). Among densities of a nonnegative variable V with $\mathbb { E } V = 1$ , the exponential density $e ^ { - v }$ uniquely maximizes differential entropy. Its amplitude $Z = { \sqrt { V } }$ has CDF

$$
\begin{array} { c } { { F _ { 0 } ( z ) = 1 - \exp ( - z ^ { 2 } ) , } } \\ { { z \geq 0 . } } \end{array}\tag{12}
$$

Proof of Proposition A.1. For a density $\rho$ with $\mathbb { E } _ { \rho } V = 1$ , define differential entropy with respect to the energy coordinate dv. Taking $\rho _ { 0 } ( v ) = e ^ { - v }$ gives

$$
\begin{array} { c l l } { \displaystyle D _ { \mathrm { K L } } ( \rho \| \rho _ { 0 } ) = \int _ { 0 } ^ { \infty } \rho ( v ) \log \rho ( v ) d v } \\ { \displaystyle + \mathbb { E } _ { \rho } V } \\ { \displaystyle = 1 - h ( \rho ) \geq 0 . } \end{array}\tag{13}
$$

Equality requires $\rho \quad = \quad \rho _ { 0 }$ almost everywhere. Transforming $Z = { \sqrt { V } }$ gives density $2 z e ^ { - z ^ { 2 } }$ and the stated CDF. □

Changing the positive mean rescales all amplitudes by a common factor. The entropy maximization is defined in the energy coordinate inherited from the second moment; defining it in the amplitude coordinate would give a different problem. This continuous reference provides an energy interpretation, while the finite expert distribution q is calibrated separately to its prescribed contrast.

## A.2 Entropy-Regularized Reweighting

To characterize this finite distribution, we express the power family as entropy-regularized reweighting. For $t \geq 0$ , the exponential family above is the unique solution of

$$
\begin{array} { r } { \boldsymbol { q } ^ { ( t ) } = \underset { \boldsymbol { q } \in \Delta _ { N } } { \mathrm { a r g m a x } } \left\{ \boldsymbol { t } \mathbb { E } _ { \boldsymbol { q } } \boldsymbol { x } - D _ { \mathrm { K L } } ( \boldsymbol { q } \| \boldsymbol { u } ) \right\} , } \end{array}\tag{14}
$$

where $\Delta _ { N }$ is the probability simplex. The objective equals $\psi ( t ) - D _ { \mathrm { { K L } } } ( q | | q ^ { ( t ) } )$ , so its unique maximizer is the stated distribution. Increasing t strengthens the preference for larger log sensitivities. NatScale determines this preference by setting $t = 2 \beta _ { \tau }$ through the realized contrast τ , placing different gradient distributions on a common information scale.

## A.3 Solving for the Exponent

This contrast-based specification determines a unique exponent because the divergence increases strictly along the power family.

Proposition A.2 (Well-defined contrast calibration). Ifthe positive gradients are nonconstant, ev-$e r y \tau > 0$ has a unique finite solution $\beta _ { \tau } > 0$ in Equation (4). This solution increases strictly with $\tau .$

Proof of Proposition A.2. Using $\log ( q _ { i } ^ { ( t ) } / u _ { i } ) =$ $t x _ { i } - \psi ( t )$ gives

$$
\mathcal { I } ( q ^ { ( t ) } , u ) = t \Delta ( t ) .\tag{15}
$$

Since $\Delta ^ { \prime } ( t ) = \psi ^ { \prime \prime } ( t ) > 0 .$ , the divergence is continuous, vanishes at $t ~ = ~ 0$ , and has derivative $\Delta ( t ) + t \psi ^ { \prime \prime } ( t ) > 0$ for $t ~ > ~ 0$ $\mathbf { A s } \ t \ \to \ \infty ,$ $\Delta ( t )  \operatorname* { m a x } _ { i } x _ { i } - \mathbb { E } _ { u } x > 0$ , so the divergence tends to infinity. Each $\tau > 0$ therefore has a unique positive root, increasing with τ ; set $t = 2 \beta$ □

The solver uses this monotonicity to find the exponent with a bracketed root finder. It checks the support before standardization, evaluates the distribution using log-sum-exp, and verifies the returned contrast. For a fixed nonconstant score vector, the behavior near zero contrast is described by the expansion

$$
\begin{array} { r l r } {  { \mathcal { I } ( q ( \beta ) , u ) = 4 \operatorname { V a r } _ { u } ( x ) \beta ^ { 2 } + O ( \beta ^ { 3 } ) , } } \\ & { } & { \beta _ { \tau } = \frac { \sqrt { \tau } } { 2 \sqrt { \operatorname { V a r } _ { u } ( x ) } } + O ( \tau ) . } \end{array}\tag{16}
$$

Zero and degenerate statistics. The positivesupport condition also determines how calibration handles degenerate statistics. Zero gradients retain zero weight and are excluded from calibration, while empty support or constant positive scores admit no positive-target solution and are rejected. As $\tau \downarrow 0$ , weights become uniform on positive support and remain zero elsewhere. When zero scores exist, this limiting distribution therefore differs from the all-one-weight ablation. All five evaluated targets are positive.

## A.4 The Weibull Case

We derive the one-nat identity in Corollary 4.1 by connecting calibration to Weibull amplitude fitting. A two-parameter Weibull CDF $F ( r ) =$ $1 - \exp [ - ( r / a ) ^ { k } ]$ maps native amplitudes $r = { \sqrt { g } }$ to the energy reference in Equation (12) as

$$
F _ { 0 } ^ { - 1 } ( F ( { \sqrt { g } } ) ) = a ^ { - k / 2 } g ^ { k / 4 } .\tag{17}
$$

The common scale leaves allocation unchanged, so the fitted shape determines the exponent. In particular, a full unweighted maximum-likelihood fit yields $\beta = \widehat { k } / 4$ , which corresponds to exactly one nat of contrast.

ProofofCorollary 4.1. Let $z _ { i } = \log r _ { i } = x _ { i } / 2$ and $\bar { z } = N ^ { - 1 } \sum _ { i } z _ { i }$ . For the ordinary unweighted fit to complete observations, profiling the Weibull likelihood over scale (Makalic and Schmidt, 2023) gives $\begin{array} { r } { \widehat { a } ( k ) ^ { k } = N ^ { - 1 } \sum _ { i } e ^ { k z _ { i } } } \end{array}$ and

$$
\frac { \ell _ { \mathrm { p r o f } } ( k ) } { N } = \log k - \log \left( \frac { 1 } { N } \sum _ { i } e ^ { k z _ { i } } \right)\tag{18}
$$

Its stationary equation is

$$
\frac { 1 } { \widehat { k } } = \sum _ { i } \frac { e ^ { \widehat { k } z _ { i } } } { \sum _ { j } e ^ { \widehat { k } z _ { j } } } z _ { i } - \bar { z } .\tag{19}
$$

The normalized profile has second derivative $- 1 / k ^ { 2 } - \mathrm { V a r } _ { \mathrm { s o f t m a x } ( k z ) } ( z ) < 0 .$ , so its stationary point is unique. To relate this point to contrast, observe that at $\beta = \widehat { k } / 4 , q = \mathrm { s o f t m a x } ( \widehat { k } z )$ . Multiplying Equation 19 by $\widehat { k }$ and applying Equation 15 then gives $\mathcal { I } = 1$ □

Because the identity follows from the likelihood equation, it holds for any positive nonconstant sample without assuming that the data follow a Weibull distribution. The same argument extends to other reference coordinates: for $p ~ > ~ 0$ , mapping to $1 - e ^ { - z ^ { p } }$ gives $w \propto g ^ { \widehat { k } / ( 2 p ) }$ and one nat for normalized $w ^ { p }$ . The energy coordinate selects $p = 2 .$ while the likelihood determines the reference contrast.

## A.5 Weight Entropy and Power Invariance

Having calibrated the squared weights, we next examine the linear sensitivity weights used in the allocation cost of Equation 8. Their entropy remains bounded below, and their normalized values are invariant to a common positive-power reparameterization of the gradient scores.

Proposition A.3 (Entropy of the linear weights). For $\begin{array} { r } { p _ { i } = g _ { i } ^ { \beta _ { \tau } } / \sum _ { j } g _ { j } ^ { \beta _ { \tau } } } \end{array}$

$$
\begin{array} { c } { { \mathcal { I } ( p , u ) < \tau / 2 , } } \\ { { H ( p ) > \log N - \tau / 2 . } } \end{array}\tag{20}
$$

Proof. Since $\Delta$ is strictly increasing,

$$
\mathcal { I } ( p , u ) = \beta _ { \tau } \Delta ( \beta _ { \tau } ) < \beta _ { \tau } \Delta ( 2 \beta _ { \tau } ) = \tau / 2 .\tag{21}
$$

Finally, $D _ { \mathrm { K L } } ( p \Vert u ) = \log N - H ( p ) \leq \mathcal { I } ( p , u )$

The entropy bound gives exp $H ( p ) > N e ^ { - \tau / 2 }$ for the gradient weights before candidatedependent costs are applied. It therefore concerns the weights themselves, not the resulting budget shares.

Proposition A.4 (Uniform positive-power reparameterization). $I f g _ { i } ^ { \prime } = a g _ { i } ^ { c } f o r$ common $a , c > 0 ,$ , recalibration at the same τ yields $\beta _ { \tau } ^ { \prime } = \beta _ { \tau } / c$ and identical normalized sensitivity weights.

Proof. Substituting $\beta ^ { \prime } = \beta / c$ gives $q ^ { \prime } ( \beta ^ { \prime } ) = q ( \beta )$ so uniqueness identifies the recalibrated root. Moreover, $\hat { ( g _ { i } ^ { \prime } ) } ^ { \beta ^ { \prime } } = a ^ { \beta / c } g _ { i } ^ { \beta }$ : the transformed weights differ only by a common factor. Their normalized values, and hence the allocation when other costs are fixed, are preserved. □

This invariance also explains the behavior of nearly constant scores. For $g _ { i } \propto \exp ( \varepsilon z _ { i } )$ with fixed nonconstant $z ,$ the exponent grows as $1 / \varepsilon$ while normalized weights remain nonuniform. A fixed positive contrast thus amplifies relative differences even $\mathsf { a s } \varepsilon \downarrow 0 ;$ exactly constant scores are rejected before standardization.

## B Residual Modeling and Sensitivity

The allocation cost in Equation 8 combines calibrated sensitivity weights with candidate residuals. To explain this combination, we examine raw-RMS stability and relate the gradient and error statistics of Sections 3.2 and 3.3 through native first-order influence.

Stability of raw RMS. We begin with the effect of small residual perturbations. For a residual vector $\boldsymbol { e } ~ \in ~ \mathbb { R } ^ { d }$ , let $R ( e ) = \| e \| _ { 2 } / \sqrt { d }$ . The reverse triangle inequality gives

$$
| R ( e + \delta ) - R ( e ) | \leq \| \delta \| _ { 2 } / \sqrt { d } .\tag{22}
$$

Centering the MSE before taking its square root changes this behavior. For two scalar candidates with residual $a > 0$ , perturbing one to $a { + } \delta$ changes raw RMS by δ but centered RMS by $\sqrt { 2 a \delta + \delta ^ { 2 } }$ The ratio of the centered change to the raw change diverges as $\delta \downarrow 0$ , showing how MSE centering amplifies near ties and makes errors depend on the candidate minimum. Shifting a complete cost row, however, preserves allocation (Equation 35).

A native first-order interpretation. To relate residual amplitude to model sensitivity, consider a differentiable real-valued graph with locally unchanged routing indices. Interpolate each expert function as

$$
y _ { i } ^ { ( \eta ) } ( x ) = y _ { i } ^ { 0 } ( x ) + \eta \left[ y _ { i } ^ { c _ { i } } ( x ) - y _ { i } ^ { 0 } ( x ) \right] .\tag{23}
$$

On native inputs, let $\epsilon _ { i , t } = y _ { i } ^ { c _ { i } } ( x _ { i , t } ) - y _ { i } ^ { 0 } ( x _ { i , t } )$ and $v _ { i , t }$ be the native summed-CE gradient. For $n _ { s , i } ^ { \mathrm { o r i g } } >$ 0, define

$$
\begin{array} { c } { \displaystyle { G _ { s , i } = \frac { \sum _ { t \in { \cal T } _ { s , i } ^ { \mathrm { o r i g } } } \| v _ { i , t } \| _ { 2 } ^ { 2 } } { n _ { s , i } ^ { \mathrm { o r i g } } H _ { i } } , } } \\ { \displaystyle { M _ { s , i } ^ { \mathrm { o r i g } } = \frac { \sum _ { t \in { \cal T } _ { s , i } ^ { \mathrm { o r i g } } } \| \epsilon _ { i , t } \| _ { 2 } ^ { 2 } } { n _ { s , i } ^ { \mathrm { o r i g } } H _ { i } } , } } \\ { \displaystyle { I _ { s } = \frac { 1 } { T _ { s } } \sum _ { i } \sum _ { t \in { \cal T } _ { s , i } ^ { \mathrm { o r i g } } } v _ { i , t } ^ { \top } \epsilon _ { i , t } . } } \end{array}\tag{24}
$$

Zero-exposure cells contribute zero. The commonη derivative is

$$
\left. \frac { d \mathcal { L } _ { \mathrm { C E } } ( \eta ) } { d \eta } \right| _ { \eta = 0 } = \sum _ { s } T _ { s } I _ { s } .\tag{25}
$$

To incorporate fold weights, the influence $I _ { \pi } =$ $\textstyle \sum _ { s } \pi _ { s } I _ { s }$ reweights injection events by $\pi _ { s } / T _ { s }$ while retaining summed-CE gradients. This definition also applies when folds split interacting sequence positions. Applying Cauchy–Schwarz gives

$$
| I _ { \pi } | \leq \sum _ { s , i } \pi _ { s } H _ { i } f _ { s , i } \sqrt { G _ { s , i } } \sqrt { M _ { s , i } ^ { \mathrm { o r i g } } } .\tag{26}
$$

Here, the concatenated norm product divided by $T _ { s }$ is $H _ { i } f _ { s , i } \sqrt { G _ { s , i } M _ { s , i } ^ { \mathrm { o r i g } } }$ . Because all experts within each evaluated checkpoint share an output width, $H _ { i }$ is a common allocation factor.

This bound motivates the use of exposure, gradient amplitude, and residual amplitude in the allocation cost. Pooling gradients, calibrating weights, and applying $A _ { i } ( c )$ remain empirical choices that the bound does not determine. The implementation also distinguishes the event sets used for error measurement: $M _ { s , i }$ uses separate replay events (Equation $6 ) .$ , whereas the derivative requires $M _ { s , i } ^ { \mathrm { o r i g } }$ from matched native events. These sets can differ near routing boundaries.

Effective byte price. We now develop the storage-price interpretation from Section 4, first for discrete candidates and then for a continuous rate model. Write $L _ { i } ( c ) = A _ { i } ( c ) E _ { i } ( c )$ . For $w _ { i } > 0$ and a common Lagrangian byte price $\lambda \geq 0$

$$
\begin{array} { l } { { \underset { c } { \mathrm { a r g } } \mathrm { m i n } \{ w _ { i } L _ { i } ( c ) + \lambda C _ { i } ( c ) \} } } \\ { { \ = \mathrm { a r g } \mathrm { m i n } \{ L _ { i } ( c ) + ( \lambda / w _ { i } ) C _ { i } ( c ) \} . } } \end{array}\tag{27}
$$

Dividing by $w _ { i }$ proves the identity and separates the roles of the two factors. The product $A _ { i } E _ { i }$ orders one expert’s candidates, while a larger $w _ { i }$ lowers that expert’s effective byte price and strengthens its competition for storage.

Continuous high-rate model. To examine how this competition affects precision, suppose one continuous bit width $b _ { i }$ describes each expert, with $M _ { i } \simeq K _ { i } 2 ^ { - 2 b _ { i } } , A _ { i } \simeq a _ { i } 2 ^ { - 2 b _ { i } }$ and $C _ { i } = N _ { i } b _ { i } / 8$ for $N _ { i }$ weights. Under these assumptions, $A _ { i } \sqrt { M _ { i } }$ decays as $\bar { 2 } ^ { - 3 b _ { i } }$ . Omitting explicit precision decay gives $2 ^ { - b _ { i } }$ for RMS or $2 ^ { - 2 b _ { i } }$ for MSE, with potentially different coefficients.

These cases can be expressed through the objective $\textstyle \sum _ { i } g _ { i } ^ { \beta } h _ { i } 2 ^ { - \nu b _ { i } }$ with $g _ { i } , h _ { i } , N _ { i } , \nu > 0$ . For an active linear budget and an interior solution, the KKT conditions give

$$
\begin{array} { l } { { \displaystyle b _ { i } - b _ { j } = \frac { \beta } { \nu } \log _ { 2 } \frac { g _ { i } } { g _ { j } } } } \\ { { \displaystyle ~ + \frac { 1 } { \nu } \log _ { 2 } \frac { h _ { i } / N _ { i } } { h _ { j } / N _ { j } } . } } \end{array}\tag{28}
$$

The relation separates sensitivity contrast $\beta$ from precision response ν in the interior model. In the implemented allocator, actual formats and projection choices instead provide discrete improvements per byte.

## C Implementation Details

Algorithm 1 implements the three stages of Section 3. The details below explain how joint errors are cached and how the final allocation is rounded and certified. Preparation produces native statistics and candidate tables that can be reused across allocation settings. Changing τ requires only contrast calibration and cost construction, while changing the payload ceiling reuses these quantities as well.

Cost normalization. We normalize complete cost rows using their minima, $m _ { i } = \operatorname* { m i n } _ { c \in \mathcal { C } _ { i } } d _ { i } ( c )$ and the global scale $\begin{array} { r } { S = \sum _ { i \in \mathcal { T } } \operatorname* { m a x } _ { c \in \mathcal { C } _ { i } } d _ { i } ( c ) } \end{array}$

$$
\widetilde { d } _ { i } ( c ) = \frac { d _ { i } ( c ) - m _ { i } } { S } .\tag{29}
$$

The scale must be finite and positive, and degenerate scores are rejected. Because a complete allocation selects one configuration per expert, subtracting the row minima shifts every allocation’s cost by the same constant. Dividing by the positive global scale also preserves the minimizer. Both operations therefore normalize the complete allocation cost $d _ { i }$ without changing the selected configuration; they do not center local MSE before taking its square root.

Calibration and candidate conventions. We collect native statistics from 1,048,576 chattemplated tokens divided into six folds, two per source. Gradients are measured at the downprojection output using summed next-token crossentropy and are shared across candidates. The candidate catalogs then provide $1 4 ^ { 3 } = 2 7 4 4$ complete configurations per expert for each Qwen model and $1 1 ^ { 3 } = 1 3 3 1$ for DeepSeek-V4-Flash-0731, including native MXFP4.

To evaluate these candidates, we fix the inputs and routes collected by a separate nativecheckpoint replay. Exposure uses $n _ { s , i } ^ { \mathrm { o r i g } } / T _ { s }$ while MSE divides the replay error sum by $H _ { i } \operatorname* { m a x } ( n _ { s , i } ^ { \mathrm { r e p } } , 1 )$ . An empty replay event set therefore stores zero MSE, including when the original collection recorded positive exposure. The collector checks total route counts at the fold level; these totals can agree even when expert-level counts differ.

Numerical expert operator. Candidate evaluation follows the numerical operator of each checkpoint. Let $R _ { B }$ denote BF16 rounding, and let the gate and up linear maps return BF16 values $a , b .$ For both Qwen models, the cached activation is $R _ { B } ( R _ { B } ( \mathrm { S i L U } ( a ) ) \odot b )$ . DeepSeek-V4-Flash-0731 first computes

$$
z = R _ { B } ( \mathrm { S i L U } ( \operatorname* { m i n } ( a , 1 0 ) ) \odot \mathrm { c l i p } ( b , - 1 0 , 1 0 ) ) ,\tag{30}
$$

with clipping, SiLU, and multiplication in FP32 before the displayed rounding. For each 128-channel block $z _ { B }$ , its FP8 scale and reconstructed activation are

$$
\begin{array} { r l } & { s _ { B } = 2 ^ { \left\lceil \log _ { 2 } ( \operatorname* { m a x } ( \| z _ { B } \| _ { \infty } , 1 0 ^ { - 4 } ) / 4 4 8 ) \right\rceil } , } \\ & { Q _ { 8 } ( z _ { B } ) = } \\ & { R _ { B } \big ( s _ { B } R _ { \mathrm { E 4 M 3 F N } } \big ( \mathrm { c l i p } ( z _ { B } / s _ { B } , - 4 4 8 , 4 4 8 ) \big ) \big ) _ { \boldsymbol { c } ^ { \prime } } } \end{array}\tag{31}
$$

We apply the same operator to native and candidate gate/up pairs. The moment identity in Equation 32, derived below, scores their final down-projection matrix maps without an additional output BF16 rounding. Its cross term therefore retains both quantized activations and down-weight errors under the specified local arithmetic.

Algorithm 1 NatScale: contrast-calibrated precision allocation   
Require: Native checkpoint, weighted calibration folds, candidate catalogs $\mathcal { C } _ { i } , \tau > 0 .$ payload ceiling B,   
optional parent map   
Ensure: Static configurations ${ \widehat { c } } ,$ payload, normalized objective and LP gap   
1: Collect native gradient moments and exposure $f _ { s , i } ;$ normalize the moments to $g _ { i } .$   
2: Solve $\mathcal { I } ( q ( \beta _ { \tau } ) , u ) = \tau$ on the nonconstant positive support; set $w _ { i } = g _ { i } ^ { \beta _ { \tau } }$ there and zero elsewhere.   
3: Replay fixed native inputs and cache joint $M _ { s , i } ( c )$ via the moment identity in Equation (32).   
4: Compute structural factors $A _ { i } ( c )$ and actual payload costs $C _ { i } ( c )$   
5: Form $\begin{array} { r } { d _ { i } ( c ) = w _ { i } A _ { i } ( c ) \sum _ { s } \pi _ { s } f _ { s , i } \sqrt { M _ { s , i } ( c ) } } \end{array}$ , setting zero-exposure terms to zero.   
6: Check $\begin{array} { r } { \sum _ { i } \operatorname* { m i n } _ { c } C _ { i } ( c ) \leq B } \end{array}$ and finite costs; normalize using Equation (29).   
7: Build lower hulls; solve the budgeted LP and round to a feasible map as described below.   
8: Keep a feasible parent map from the same catalog if its objective is smaller.   
9: Verify $\begin{array} { r } { \sum _ { i } C _ { i } ( \widehat { c } _ { i } ) \leq B ; } \end{array}$ return the map, payload, objective and gap $\widetilde J ( \widehat { c } ) - \widetilde J _ { B } ^ { \mathrm { L P } }$

For the structural factor, our AlphaQ adaptation estimates spectra on complete 128 × 128 weight blocks, fits the largest 10% of their spectra, and uses spectral exponent one. The variance term is the population weight variance. The precision factor uses nominal bits for NINT integer formats and effective bpw for NVQ vector formats, excluding the four-byte expert descriptor, while budget accounting uses actual candidate payload. Common positive score normalizers leave the minimizer unchanged. The adaptation uses no K-FAC attribution, projection-specific budget cap, or worstsource aggregation.

Moment evaluation. To construct the joint-error tables used in Section 3.3, we reuse activation and weight moments across candidate configurations. For one expert and fold with $n = n _ { s , i } ^ { \mathrm { r e p } } > 0$ replay events, let $Z _ { a b } \in \mathbb { R } ^ { n \times m }$ be gate/up activations and $D _ { c } \in \mathbb { R } ^ { H \times m }$ a down matrix. Using checkpoint values $Z _ { 0 } , D _ { 0 }$ on the same replay inputs, define $\Delta Z _ { a b } = Z _ { a b } - Z _ { 0 }$ and $\Delta D _ { c } = D _ { c } - D _ { 0 }$ . Expanding the residual $\Delta Z _ { a b } D _ { c } ^ { \mathsf { T } } + Z _ { 0 } \Delta D _ { c } ^ { \mathsf { T } }$ gives

$$
\begin{array} { r l } & { n H M ( a , b , c ) = \langle \Delta Z _ { a b } ^ { \mathsf { T } } \Delta Z _ { a b } , D _ { c } ^ { \mathsf { T } } D _ { c } \rangle _ { F } } \\ & { ~ + ~ 2 \langle \Delta Z _ { a b } ^ { \mathsf { T } } Z _ { 0 } , D _ { c } ^ { \mathsf { T } } \Delta D _ { c } \rangle _ { F } } \\ & { ~ + ~ \langle Z _ { 0 } ^ { \mathsf { T } } Z _ { 0 } , \Delta D _ { c } ^ { \mathsf { T } } \Delta D _ { c } \rangle _ { F } . } \end{array}\tag{32}
$$

The identity follows from trace cyclicity and retains all off-diagonal moments and the cross term. Because the gate/up-pair and down-candidate moments are reused across triples, each configuration can be scored without replaying every token. The identity is exact for the specified local operator; we check finite-precision accumulation against direct evaluation.

This reuse changes the computational cost. With $Q$ candidates per projection and $n _ { \mathrm { t o t } }$ replay events, direct replay costs $O ( Q ^ { 3 } n _ { \mathrm { t o t } } m H )$ . For F folds, evaluation with reused moments costs

$$
\begin{array} { c } { { O ( Q n _ { \mathrm { t o t } } H m ) + O ( Q ^ { 2 } n _ { \mathrm { t o t } } m ^ { 2 } ) } } \\ { { + O ( Q H m ^ { 2 } ) + O ( F Q ^ { 3 } m ^ { 2 } ) . } } \end{array}\tag{33}
$$

The first terms account for constructing outputs and moments, while the final contractions are independent of token count. Across six retained estimates, the results agree with direct evaluation within a relative discrepancy of $4 . 4 0 \times 1 0 ^ { - 6 }$ . We cache one MSE for each fold, expert, and configuration for subsequent allocation.

LP relaxation and certificate. Once the candidate costs are available, we obtain a lower bound by relaxing Equation 1 to mixtures $x _ { i , c } ~ \geq ~ 0 _ { \mathrm { { \ ; } } }$ $\begin{array} { l l l } { \sum _ { c } x _ { i , c } } & { = } & { 1 } \end{array}$ , under the same budget. This LP (Boyd and Vandenberghe, 2004) contains every discrete allocation, so its optimum $J _ { B } ^ { \mathrm { L P } }$ bounds the discrete optimum $J _ { B } ^ { \star }$ from below. Every feasible returned configuration bc therefore satisfies

$$
0 \leq J ( { \widehat { c } } ) - J _ { B } ^ { \star } \leq J ( { \widehat { c } } ) - J _ { B } ^ { \mathrm { L P } } .\tag{34}
$$

The resulting certificate bounds optimization error for the fixed proxy objective. To express it in normalized cost units, let $\begin{array} { r } { \widetilde { J } ( c ) = \sum _ { i } \widetilde { d } _ { i } ( c _ { i } ) } \end{array}$ . Then

$$
\begin{array} { c } { { J ( c ) = S \widetilde { J } ( c ) + \displaystyle \sum _ { i } m _ { i } , } } \\ { { J ( \widehat { c } ) - J _ { B } ^ { \mathrm { L P } } = S \big [ \widetilde { J } ( \widehat { c } ) - \widetilde { J } _ { B } ^ { \mathrm { L P } } \big ] . } } \end{array}\tag{35}
$$

Thus, the row offsets preserve both optimizers, and S converts the normalized gap back to raw units.

Deterministic boundary rounding. To construct the LP solution and a feasible discrete allocation, the solver first orders each expert’s candidates by ascending payload and then loss, preserving the original order for exact ties. It removes dominated points and constructs the lower convex hull, whose edges represent objective reductions obtained by spending additional bytes. The solver sorts these edges by decreasing reduction per byte. Since the hull slopes are ordered, this sorting preserves the required order of upgrades within each expert. The LP consumes complete edges until the first boundary edge exceeds the remaining budget, then takes the affordable fraction of that edge. If all upgrades fit, the endpoint allocation is already integral.

When the boundary is fractional, rounding starts from its lower-payload endpoint. The solver skips the boundary edge and scans the remaining edges in the same stable order, accepting an upgrade only when the current expert configuration is its preceding endpoint and the additional bytes fit. Each accepted move therefore preserves feasibility. If a feasible parent configuration from the same catalog has a strictly smaller objective, the solver retains it. The procedure returns a feasible map together with its measured gap to the LP optimum.

## D Evaluation Setup

To complement the comparison in Table 1, this appendix describes the model architectures, shared evaluation protocol, and baseline adaptations for Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, and DeepSeek-V4-Flash-0731.

Model scale and architecture. The three models span increasing reported total and activated parameter counts: 35B/3B for Qwen3.6-35B-A3B, 125B/6B for Qwen3.8-Flash-Next, and 284B/13B for DeepSeek-V4-Flash-0731 (Qwen Team, 2026a,b; DeepSeek-AI, 2026b,a). Qwen3.6- 35B-A3B has 40 layers and 256 routed experts per layer, activating eight plus one shared expert. Qwen3.8-Flash-Next has 48 layers and 512 routed experts, activating ten plus one shared expert. DeepSeek-V4-Flash-0731 has 43 layers and 256 routed experts, activating six plus one shared expert.

These reported parameter counts follow each model’s specification. For Qwen3.8-Flash-Next, the reported 125B excludes the additional 51B ngram embedding parameters and 4B multi-tokenprediction (MTP) parameters. In contrast, the official 284B Flash specification includes its MTP configuration, while the 0731 checkpoint also contains the DSpark speculative-decoding module. Reported specifications, complete checkpoint parameter counts, and routed-only byte budgets therefore describe different scopes.

Calibration. For each model, we use a frozen 1,048,576-token calibration pool rendered with its native chat template and assign equal weight to the six folds. Across NatScale score variants, we reuse the native gradient second moments, routing frequencies, candidate weights, and complete joint reconstruction errors. Changing the threshold or budget therefore requires no new collection of statistics.

Shared candidate formats. All methods use candidates produced by the MixedFlowQuant (MFQ) backend. Its integer and vector-quantization formats are denoted by NINT and NVQ, respectively, with suffixes identifying the fixed candidate encodings. Qwen3.6-35B-A3B and Qwen3.8-Flash-Next use NINT2, NINT3, NINT4, NINT5, NINT6, NINT8, NVQ2J, NVQ2J-L, NVQ2J-XL, NVQ3J, NVQ3J-512, NVQ3J-L, NVQ1-S, and NVQ1-L. DeepSeek-V4-Flash-0731 uses NINT2, NINT3, the six NVQ2J/NVQ3J formats, native MXFP4, NVQ1-S, and NVQ1-L. The resulting catalogs contain 14, 14, and 11 entries, respectively. DeepSeek-V4-Flash-0731 is not assigned NINT4/5/6/8 candidates above its native MXFP4 endpoint. Within these catalogs, NatScale evaluates complete mixed gate/up/down configurations while keeping nonrouted weights native.

To compare allocation methods over the same candidates, the five external baselines share these weights and the native evaluation protocol. Each baseline retains its own scoring rule and allocation constraints; the adaptations are described below in the order used in the result tables.

MxMoE. MxMoE assigns precision to linear blocks using their reconstruction sensitivity (Duanmu et al., 2025). Our adaptation measures layeroutput L2 error after a single-projection perturbation with r = 1, ties the gate and up formats within each expert, and uses a global storage budget. Hadamard transforms and GPTQ are disabled.

FRI-MxMoE. FRI-MxMoE uses fuzzy rule interpolation to estimate reconstruction costs from a subset of profiled experts (Qi et al., 2026). Our adaptation uses measured expert anchors to interpolate costs across the shared candidate formats, then allocates precision under their encoded-byte costs.

AlphaQ. AlphaQ derives its allocation scores from weight spectra without calibration text (Yang et al., 2026). We adapt this criterion to the shared candidate formats by combining the spectral factor, weight variance, and bit-decay factor in Equation 36. The shared candidates do not undergo the subsequent GPTQ reconstruction used in the original pipeline. The adaptation uses the released code’s default spectral exponent of one,<sup>1</sup> which differs from the model-dependent curvature rule given in the paper.

With this exponent, the adapted structural factor for projection $p$ is

$$
\begin{array} { r } { A _ { i , p } ( c _ { p } ) = \displaystyle \chi _ { i , p } ( c _ { p } ) \frac { \widetilde \alpha } { \alpha _ { i , p } } \qquad } \\ { \qquad \cdot \operatorname { V a r } ( W _ { i , p } ) 2 ^ { - 2 b _ { i , p } ( c _ { p } ) } . } \end{array}\tag{36}
$$

Here $\alpha _ { i , p }$ is the spectral tail index and α is its median across all target projections. The precision $b _ { i , p } ( c _ { p } )$ uses nominal bits for integer formats and effective bits per weight for vector formats. The indicator χ is zero only for DeepSeek-V4- Flash-0731’s native MXFP4 endpoint and one otherwise. To obtain an expert-level structural factor, NatScale averages the three projection factors equally, $\begin{array} { r } { A _ { i } ( c ) = \frac { 1 } { 3 } \sum _ { p \in \{ G , U , D \} } A _ { i , p } ( c _ { p } ) } \end{array}$

MODE. MODE decomposes expert importance by modality for multimodal models (Chen et al., 2026). Our text-only adaptation combines sourcedependent exposure with expert-output reconstruction error, using the calibration text sources in place of modalities. It assigns the same candidate format to gate, up and down within each expert.

GEMQ (allocation-only). The label GEMQ (alloc.) in tables and figures denotes the allocationonly adaptation of GEMQ (Deng et al., 2026). It uses the criterion in that paper’s Equation (6), combining per-token squared output gradients with whole-expert perturbations. Under this adaptation, gate, up and down share one candidate format per expert, and allocation uses encoded bytes while retaining the high-precision expert coverage constraints. GPTQ, router fine-tuning, and progressive quantization are excluded from the adapted pipeline.

To compute the gradient criterion, the evaluation protocol specifies all 512 calibration sequences with their loss masks and reuses native forward/backward statistics across budgets. We differentiate the per-sequence mean loss, with routing weights included in either the squared-gradient coefficients or the routed perturbation. The current Qwen3.8-Flash-Next and DeepSeek-V4-Flash-0731 measurements use the gradient-corrected reruns published on 2026-10-09; Qwen3.6-35B-A3B is unchanged.

Auxiliary weighting controls. Candidate fitting can also use an importance matrix (imatrix), which supplies activation-based weights. The routedonly experiments omit this weighting, whereas the whole-model extension uses the archived UD importance matrix described in Appendix H.

Native evaluation. We evaluate the resulting quantized models on WT2 test text without a chat template. Each window contains 512 tokens, with logits 256–510 providing 255 scored next-token predictions. The Qwen3.6-35B-A3B and Qwen3.8-Flash-Next streams contain 580 windows each, while DeepSeek-V4-Flash-0731 contains 573. Within a model, all methods use the same token stream, scored positions, and native reference. Using this common reference, KLD measures divergence from the native distribution to the quantized distribution, and Top1 measures native argmax agreement; PPL instead pools the true nexttoken negative log-likelihood. All three metrics are computed from complete quantized forward passes, which retain routing changes and accumulated errors.

Storage conventions. Storage budgets in these comparisons cover routed-expert weights, and the routed-bpw targets are nominal. NatScale constrains candidate payload and reports headers separately, whereas the adapted baselines reserve header bytes within their targets. For the NatScale threshold and component comparisons, we hold the payload ceiling fixed.

## E Precision Allocation

The allocation patterns discussed in Section 5.3 complement the quality results in Table 1 by showing where each method assigns storage. Figures 3– 5 compare NatScale with MxMoE, FRI-MxMoE, AlphaQ and MODE at 2.5 nominal routed bpw. The three model figures use a common method order and color scale, while retaining each model’s original expert order.

These figures present the allocations at three levels. The heatmaps show each expert’s stored bits, including overhead, divided by its combined gate/up/down parameter count. To account for this overhead, tensor headers are shared equally among a layer’s experts, while format and codebook headers are shared among experts using that format. The cumulative distributions then summarize storage across all experts. Finally, the layer curves show how storage varies with depth by dividing each layer’s total expert storage by its routed parameter count. NatScale’s variation in precision within layers is visible in the heatmaps, while the cumulative distributions show the overall spread of expert storage.

## F Design Ablations

This appendix provides the protocols and complete results behind the design comparisons in Section 5.3. We first vary the contrast target and score factors across three models and budgets, then compare joint and separate projection errors on Qwen3.8-Flash-Next. Throughout these comparisons, calibration statistics and candidate weights remain fixed, as does each setting’s exact payload ceiling.

## F.1 Contrast Targets

We begin with the calibration rule in Equation (4), varying its contrast target around the one-nat reference derived in Section 4. For each model, we solve the Jeffreys calibration condition at five logspaced targets $\tau \in \{ 0 . 5 , \sqrt { 0 . 5 } , 1 , \sqrt { 2 } , 2 \}$ nats on the positive-gradient support, using natural logarithms and no factor of $1 / 2 .$ . The resulting exponent enters the allocation score through $g ^ { \beta _ { \tau } }$ , whereas the normalized distribution used in the divergence is proportional to $g ^ { 2 \beta _ { \tau } }$ . At each target, one exponent is shared across layers and budgets, and zero-gradient experts retain zero weight. For the one-nat setting, we use the saved Weibull-MLE exponent exactly.

This comparison changes only the gradient exponent: raw joint RMS, the structural factor that equally weights the three AlphaQ projection factors, fold frequencies, candidate weights, and payload ceilings remain fixed. In the table, the labels 0.7071 and 1.4142 denote the exact midpoints $\sqrt { 0 . 5 }$ and ${ \sqrt { 2 } } .$ . Table 4 provides the KLD and Top1 values underlying Figure 2 and additionally reports PPL, showing how preferences for contrast differ between likelihood and native-output fidelity.

## F.2 Component Ablations

Having varied the contrast target, we next examine the factors combined in Equation (8) through five core removals from the default, which uses joint raw RMS and $\tau ~ = ~ 1$ Removing local RMS error sets the error factor to one. When bit decay is removed, the spectral and variance factors remain, and DeepSeek-V4-Flash-0731’s native MXFP4 endpoint retains zero added error.

The remaining three removals concern expert weighting. Removing gradient weights assigns every expert a weight of one, including experts with zero native gradient; removing gradient smoothing instead sets $\beta = 1$ . Finally, removing exposure magnitude replaces the frequency $f _ { s , i }$ with $\mathbf { 1 } [ f _ { s , i } > 0 ]$ , while preserving the routing dependence of the gradient statistics.

To examine the structural factors and error power in more detail, we introduce four additional controls for Qwen3.8-Flash-Next. The MSE variant also removes bit decay and therefore differs from the default in two factors. Comparing it with the RMS variant without bit decay isolates the effect of error power under that condition. The other three controls retain bit decay while removing the spectral factor, the variance factor, or both. Tables 5–7 report 66 component results across KLD, Top1 and PPL.

Across the nine model–budget settings, removing bit decay consistently worsens all three metrics. Gradient weights also improve native-output fidelity throughout: removing them worsens KLD and Top1 in every setting, although it improves PPL in two. By comparison, the effects of local RMS error and gradient smoothing depend on the setting. Removing either improves all three metrics for DeepSeek-V4-Flash-0731 at 2.0 bpw but worsens them in the other eight settings.

The exposure and spectral controls further distinguish the metrics. Removing exposure magnitude improves KLD once and PPL twice, while never improving Top1. On Qwen3.8-Flash-Next, removing the spectral factor improves all three metrics at 3.0 bpw, whereas at 2.5 bpw it trades worse KLD

![](images/c92a02c1c1782e408c80e028124c9a22b91094c4dfe99eb31854958854d5ba36.jpg)  
Figure 3: Expert precision and layer budgets for Qwen3.6-35B-A3B at nominal 2.5 routed bpw. Panels (a)–(e) show actual expert storage; (f) gives its cumulative distribution; (g) shows routed storage by layer. The five methods use the same order, storage definition and color scale across the three model figures.

for better Top1 and PPL.

## F.3 Joint and Separate Errors

The joint-error design in Section 3.3 retains interactions among gate, up and down perturbations. To evaluate this choice in the ablation reported in Section 5.3, we replace the joint MSE in Equation (6) with a sum of single-projection perturbation errors:

$$
\begin{array} { r l } { { \cal M } _ { s , i } ^ {  { \mathrm { s e p } } } ( c _ { G } , c _ { U } , c _ { D } ) = { \cal M } _ { s , i } ( c _ { G } , 0 , 0 ) } & { } \\ & { ~ + ~ { \cal M } _ { s , i } ( 0 , c _ { U } , 0 ) } \\ & { ~ + ~ { \cal M } _ { s , i } ( 0 , 0 , c _ { D } ) . } \end{array}\tag{37}
$$

Here 0 denotes the native projection, used only as a measurement reference. Each term changes one projection while keeping the other two native. To form the separate-error score, we sum these MSEs before taking the square root, while retaining the calibrated gradients, structural factors, exposure weights, candidate weights and global payload ceilings. Both variants allow G/U/D to select different formats; joint scoring remains the default, and separate-error scoring serves as the control.

This change in measurement reduces the number of local-error entries from $Q ^ { 3 }$ to 3Q per expert and fold, or from 2744 to 42 for Qwen3.8-Flash-Next. It leaves the same $Q ^ { 3 }$ deployment configurations available to both scores. The joint implementation evaluates its entries by reusing matrix moments.

The resulting comparison on Qwen3.8-Flash-Next shows that separate scoring lowers KLD at 2.0 and 3.0 bpw and raises Top1 at 2.0 and 2.5 bpw. At 2.5 bpw, for example, it raises Top1 from 84.8573% to 84.9175% but increases KLD from 0.194086 to 0.195639. At 3.0 bpw, the metric preferences reverse: separate scoring has lower KLD, while joint scoring has higher Top1.

To assess how these error scores affect task performance, the downstream comparison in Table 9 evaluates them on four tasks using Qwen3.8-Flash-Next at 2.5 routed bpw.

![](images/2b6933c7b3eb6c455a5f9f5e8228ff4c322845ef5ebbadcad6ba946fcbffb856.jpg)  
Figure 4: Expert precision and layer budgets for Qwen3.8-Flash-Next at nominal 2.5 routed bpw. Heatmaps, cumulative storage distributions and layer curves follow the definitions and shared scales in Figure 3. NatScale uses the joint-error default.

## G Downstream Evaluation

Section 5.2 asks whether the gains in native-output fidelity carry over to downstream tasks. This appendix specifies the protocols for that comparison in Table 3, which evaluates NatScale and five allocation baselines on Qwen3.8-Flash-Next at nominal 2.5 routed bpw. Using the same model and storage target, the error-score ablation in Table 9 further examines the choice of error estimate. The default NatScale configuration uses joint raw RMS and τ = 1, while the ablation replaces the joint estimate with the separate errors defined in Equation (37).

Shared inference conditions. For both comparisons, we use a modified Benchi pipeline with thinking disabled. The pipeline reuses the fixed allocations and candidate weights from WT2 evaluation without task-specific refitting. Non-routed weights remain native and candidate fitting uses no imatrix. The five baselines use the shared-MFQ adaptations described in Appendix D. For AIME and GPQA, evaluation uses the corrected chatcompletions interface and native chat template; earlier completion-interface runs are excluded.

AIME 2025. For AIME, we generate four responses to each of 30 problems at temperature 1, top-p 1 and top-k 20, with a 32,768-token limit. MathArena (Balunovic et al.´ , 2025) then scores mean accuracy across all generated responses. For joint-error NatScale on Qwen3.8-Flash-Next, we exclude the faulty run and use the corrected runs with seeds 22345 and 32345. Each corrected run provides two responses per problem, giving a combined score of 82/120 (68.33%). The separate-error runs use seed 12345 to generate all four responses per problem.

GPQA Diamond. For GPQA, all methods are evaluated on 198 questions (Rein et al., 2024) with greedy decoding and a 32,768-token limit. Each method skips the same long question, which is counted as incorrect. Thus, evaluation requires 197 requests but retains a denominator of 198. Refusals remain in the denominator.

![](images/ad0fc351ef261f7fcdd76218a3e3a0dc2455b423218363a49ef19b8dc1ac3143.jpg)  
Figure 5: Expert precision and layer budgets for DeepSeek-V4-Flash-0731 at nominal 2.5 routed bpw. All 43 layers and 256 routed experts per layer are shown in their original order. Heatmaps, cumulative distributions and layer curves use the same definitions and scales as the two Qwen figures.

TruthfulQA binary. We use the binary-2025 adaptation of TruthfulQA (Lin et al., 2022), which contains 790 questions. Responses are generated with greedy decoding and a 32-token limit, while seed 0 fixes the answer-option order. Accuracy is the fraction of questions answered correctly.

LiveCodeBench v6. We evaluate the 175 newly added v6 problems in test6.jsonl (Jain et al., 2025), using one greedy response per problem, seed 0 and a 32,768-token limit. The generated responses are evaluated by a checker with a sixsecond timeout and 12 scoring processes; pass@1 uses the first valid grade for each problem. Where task-index correspondence requires repair, we restore it and generate only the missing answers. Every reported configuration therefore covers all 175 unique problems.

Joint and separate errors. With these protocols fixed, Table 9 extends the WikiText-2 error comparison in Section 5.3 to task performance. The separate-error scores for Qwen3.8-Flash-Next come from the 12 October export, while the jointerror scores are those reported in the main table. Both follow the AIME protocol described above, with the joint-error result taken from the corrected runs.

Joint scoring achieves higher scores on AIME, GPQA and TruthfulQA for Qwen3.8-Flash-Next, whereas separate scoring performs better on Live-CodeBench. We retain joint scoring as the default because it preserves projection interactions.

## H Whole-Model Allocation

The whole-model comparison on Qwen3.8-Flash-Next in Section 5.4 tests whether NatScale’s allocation rule remains effective when shared experts, attention and gated-residual projections also receive quantized formats. To define this extension, we first generalize the expert cost in Equation (8) to these allocation units, then include fixed tensors and metadata in the global byte constraint. The gated-residual projections are archived as MHC, and the identifier G1\_raw denotes raw-RMS scoring under this whole-model protocol.

Table 4: All 45 threshold settings, including the nine defaults. Top1 is native-reference agreement (%). Bold marks each metric’s best value among the five contrasts within each model and budget.
<table><tr><td>Model</td><td>Target bpw</td><td>τ (nat)</td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td></tr><tr><td>Qwen3.6-</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>35B-A3B</td><td>2.0</td><td>0.5</td><td>0.202861</td><td>82.8668</td><td>7.703205</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.201441</td><td>82.9946</td><td>7.682874</td></tr><tr><td></td><td></td><td>1</td><td>0.198816</td><td>82.9824</td><td>7.676734</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.196132</td><td>83.1650</td><td>7.676333</td></tr><tr><td></td><td></td><td>2</td><td>0.195436</td><td>83.0852</td><td>7.689436</td></tr><tr><td></td><td>2.5</td><td>0.5</td><td>0.100756</td><td>87.7958</td><td>7.125175</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.099959</td><td>87.8188</td><td>7.154521</td></tr><tr><td></td><td></td><td>1</td><td>0.098355</td><td>87.9378</td><td>7.150688</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.099830</td><td>87.8431</td><td>7.204622</td></tr><tr><td></td><td></td><td>2</td><td>0.100853</td><td>87.8580</td><td>7.248561</td></tr><tr><td></td><td>3.0</td><td>0.5</td><td>0.058600</td><td>90.6437</td><td>7.037591</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.057835</td><td>90.6876</td><td>7.025139</td></tr><tr><td></td><td></td><td>1</td><td>0.057493</td><td>90.7228</td><td>7.014946</td></tr><tr><td></td><td></td><td>1.4142 2</td><td>0.057376</td><td>90.7897</td><td>7.022438</td></tr><tr><td></td><td></td><td></td><td>0.057317</td><td>90.7674</td><td>7.042827</td></tr><tr><td>Qwen3.8-</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Flash-Next</td><td>2.0</td><td>0.5</td><td>0.345286</td><td>80.1237</td><td>5.079686</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.337696</td><td>80.3198</td><td>5.074160</td></tr><tr><td></td><td></td><td>1</td><td>0.329276</td><td>80.2170</td><td>5.067797</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.336225</td><td>80.1034</td><td>5.073389</td></tr><tr><td></td><td></td><td>2</td><td>0.340955</td><td>80.0169</td><td>5.086687</td></tr><tr><td></td><td>2.5</td><td>0.5</td><td>0.201418</td><td>84.7796</td><td>4.823743</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.197665</td><td>84.8621</td><td>4.822990</td></tr><tr><td></td><td></td><td>1</td><td>0.194086</td><td>84.8573</td><td>4.834395</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.194985</td><td>84.8763</td><td>4.851031</td></tr><tr><td></td><td></td><td>2</td><td>0.196514</td><td>84.6606</td><td>4.878319</td></tr><tr><td></td><td>3.0</td><td>0.5</td><td>0.127064</td><td>87.6876</td><td>4.736719</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.125395</td><td>87.7735</td><td>4.740789</td></tr><tr><td></td><td></td><td>1</td><td>0.124340</td><td>87.6714</td><td>4.743009</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.121826</td><td>87.7437</td><td>4.749630</td></tr><tr><td></td><td></td><td>2</td><td>0.121556</td><td>87.7830</td><td>4.748906</td></tr><tr><td colspan="6">DeepSeek-V4-</td></tr><tr><td>Flash-0731</td><td>2.0</td><td>0.5</td><td>0.765042</td><td>71.3554</td><td>8.648208</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.751667</td><td>71.6217</td><td>8.579417</td></tr><tr><td></td><td></td><td>1</td><td>0.729311</td><td>72.1534</td><td>8.364731</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.701516</td><td>72.6845</td><td>8.144471</td></tr><tr><td></td><td></td><td>2</td><td>0.678030</td><td>73.1780</td><td>7.925372</td></tr><tr><td></td><td>2.5</td><td>0.5</td><td>0.346255</td><td>81.1532</td><td>5.978910</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.343613</td><td>81.2730</td><td>6.008364</td></tr><tr><td></td><td></td><td>1</td><td>0.336141</td><td>81.5714</td><td>5.961485</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.332286</td><td>81.6555</td><td>5.939674</td></tr><tr><td></td><td></td><td>2</td><td>0.331466</td><td>81.7363</td><td>5.967101</td></tr><tr><td></td><td>3.0</td><td>0.5</td><td>0.193404</td><td>86.1705</td><td>5.473308</td></tr><tr><td></td><td></td><td>0.7071</td><td>0.192386</td><td>86.2376</td><td>5.477620</td></tr><tr><td></td><td></td><td>1</td><td>0.193588</td><td>86.1609</td><td>5.488280</td></tr><tr><td></td><td></td><td>1.4142</td><td>0.194685</td><td>86.2430</td><td>5.492832</td></tr><tr><td></td><td></td><td>2</td><td>0.196102</td><td>86.1883</td><td>5.509159</td></tr></table>

## H.1 Allocation Units and Scores

To define allocation units for this extension, we follow the checkpoint’s modular structure, which combines experts with Qwen Sparse Attention (QSA), Gated DeltaNet (GDN) and gated-residual modules (Qwen Team, 2026b). The resulting set U contains 24,576 routed experts, 48 shared experts, 48 attention groups and 96 MHC groups, totaling 24,768 units and 74,304 projections. Each expert groups its gate/up/down projections. Attention groups contain Q/K/V/G/O in QSA layers or Q/K/V/Z/O in GDN layers, while MHC groups contain up/down. Within a unit, projections may receive different encodings from the 14-format catalog (Appendix D). Across all three budgets, we reuse candidates fitted with the UD importance matrix (Unsloth, 2026) and compute allocation statistics from our native calibration data.

Table 5: NatScale component variants for Qwen3.6-35B-A3B under fixed candidate weights and payload ceilings. Top1 is native agreement (%). Bold marks the best listed variant within each budget and metric; the joint-error default appears last.
<table><tr><td>bpw</td><td>Variant</td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td></tr><tr><td>2.0</td><td>NatScale w/o local RMS error</td><td>0.243853</td><td>81.1204</td><td>8.156114</td></tr><tr><td>2.0</td><td>NatScale w/o bit decay</td><td>0.310342</td><td>79.1062</td><td>8.450841</td></tr><tr><td>2.0</td><td>NatScale w/o gradient weights</td><td>0.222994</td><td>81.9155</td><td>7.854326</td></tr><tr><td>2.0</td><td>NatScale w/o gradient smoothing</td><td>0.245525</td><td>81.0257</td><td>8.210989</td></tr><tr><td>2.0</td><td>NatScale w/o exposure magnitude</td><td>0.194500</td><td>82.4719</td><td>7.719427</td></tr><tr><td>2.0</td><td>NatScale (ours, τ = 1)</td><td>0.198816</td><td>82.9824</td><td>7.676734</td></tr><tr><td>2.5</td><td>NatScale w/o local RMS error</td><td>0.125210</td><td>86.9506</td><td>7.445417</td></tr><tr><td>2.5</td><td>NatScale w/o bit decay</td><td>0.184559</td><td>84.4388</td><td>7.676710</td></tr><tr><td>2.5</td><td>NatScale w/o gradient weights</td><td>0.105059</td><td>87.5808</td><td>7.185638</td></tr><tr><td>2.5</td><td>NatScale w/o gradient smoothing</td><td>0.131737</td><td>86.2691</td><td>7.474272</td></tr><tr><td>2.5</td><td>NatScale w/o exposure magnitude</td><td>0.106081</td><td>86.8796</td><td>7.207796</td></tr><tr><td>2.5</td><td>NatScale (ours, τ = 1)</td><td>0.098355</td><td>87.9378</td><td>7.150688</td></tr><tr><td>3.0</td><td>NatScale w/o local RMS error</td><td>0.068296</td><td>90.0609</td><td>7.131690</td></tr><tr><td>3.0</td><td>NatScale w/o bit decay</td><td>0.121408</td><td>87.6362</td><td>7.296315</td></tr><tr><td>3.0</td><td>NatScale w/o gradient weights</td><td>0.061691</td><td>90.4320</td><td>7.016817</td></tr><tr><td>3.0</td><td>NatScale w/o gradient smoothing</td><td>0.075514</td><td>89.5612</td><td>7.201615</td></tr><tr><td>3.0</td><td>NatScale w/o exposure magnitude</td><td>0.059845</td><td>90.1657</td><td>7.014259</td></tr><tr><td>3.0</td><td>NatScale (ours, τ = 1)</td><td>0.057493</td><td>90.7228</td><td>7.014946</td></tr></table>

Table 6: NatScale component variants for Qwen3.8-Flash-Next, including error-power and spectral/variance controls. Bold marks the best variant within each budget and metric.
<table><tr><td>bpw</td><td>Variant</td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td></tr><tr><td>2.0</td><td>NatScale w/o local RMS error</td><td>0.378134</td><td>78.5598</td><td>5.302109</td></tr><tr><td>2.0</td><td>NatScale w/o bit decay</td><td>0.456162</td><td>76.5707</td><td>5.582535</td></tr><tr><td>2.0</td><td>NatScale w/o gradient weights</td><td>0.337510</td><td>79.8438</td><td>5.154556</td></tr><tr><td>2.0</td><td>NatScale w/o gradient smoothing</td><td>0.492273</td><td>75.1075</td><td>5.816586</td></tr><tr><td>2.0</td><td>NatScale w/o exposure magnitude</td><td>0.330178</td><td>80.1420</td><td>5.035539</td></tr><tr><td>2.0</td><td>NatScale w/ MSE, w/o bit decay</td><td>0.389347</td><td>78.4050</td><td>5.372212</td></tr><tr><td>2.0</td><td>NatScale w/o spectral factor</td><td>0.329165</td><td>80.1866</td><td>5.086408</td></tr><tr><td>2.0</td><td>NatScale w/o variance factor</td><td>0.332328</td><td>80.1501</td><td>5.082415</td></tr><tr><td>2.0</td><td>NatScale w/o spectral and variance factors</td><td>0.334005</td><td>80.0710</td><td>5.081566</td></tr><tr><td>2.0</td><td>NatScale (ours, τ = 1)</td><td>0.329276</td><td>80.2170</td><td>5.067797</td></tr><tr><td>2.5</td><td>NatScale w/o local RMS error</td><td>0.226839</td><td>83.6518</td><td>4.909631</td></tr><tr><td>2.5</td><td>NatScale w/o bit decay</td><td>0.292266</td><td>81.6971</td><td>5.072827</td></tr><tr><td>2.5</td><td>NatScale w/o gradient weights</td><td>0.202779</td><td>84.6815</td><td>4.802855</td></tr><tr><td>2.5</td><td>NatScale w/o gradient smoothing</td><td>0.314415</td><td>80.3550</td><td>5.236243</td></tr><tr><td>2.5</td><td>NatScale w/o exposure magnitude</td><td>0.223295</td><td>83.6180</td><td>4.858623</td></tr><tr><td>2.5</td><td>NatScale w/ MSÊ, w/o bit decay</td><td>0.238135</td><td>83.3698</td><td>4.925901</td></tr><tr><td>2.5</td><td>NatScale w/o spectral factor</td><td>0.196914</td><td>84.9601</td><td>4.829930</td></tr><tr><td>2.5</td><td>NatScale w/o variance factor</td><td>0.199316</td><td>84.8431</td><td>4.830544</td></tr><tr><td>2.5</td><td>NatScale w/o spectral and variance factors</td><td>0.197706</td><td>84.9020</td><td>4.835880</td></tr><tr><td>2.5</td><td>NatScale (ours, τ = 1)</td><td>0.194086</td><td>84.8573</td><td>4.834395</td></tr><tr><td>3.0</td><td>NatScale w/o local RMS error</td><td>0.138895</td><td>87.1481</td><td>4.822337</td></tr><tr><td>3.0</td><td>NatScale w/o bit decay</td><td>0.202124</td><td>84.8722</td><td>4.902857</td></tr><tr><td>3.0</td><td>NatScale w/o gradient weights</td><td>0.128369</td><td>87.6504</td><td>4.736839</td></tr><tr><td>3.0</td><td>NatScale w/o gradient smoothing</td><td>0.197610</td><td>84.6646</td><td>4.956623</td></tr><tr><td>3.0</td><td>NatScale w/o exposure magnitude</td><td>0.130632</td><td>87.3597</td><td>4.758108</td></tr><tr><td>3.0</td><td>NatScale w/ MSE, w/o bit decay</td><td>0.138096</td><td>86.9229</td><td>4.825457</td></tr><tr><td>3.0</td><td>NatScale w/o spectral factor</td><td>0.123582</td><td>87.7458</td><td>4.737910</td></tr><tr><td>3.0</td><td>NatScale w/o variance factor</td><td>0.125899</td><td>87.7363</td><td>4.737174</td></tr><tr><td>3.0</td><td>NatScale w/o spectral and variance factors</td><td>0.123974</td><td>87.6788</td><td>4.755737</td></tr><tr><td>3.0</td><td>NatScale (ours, τ = 1)</td><td>0.124340</td><td>87.6714</td><td>4.743009</td></tr></table>

Table 7: NatScale component variants for DeepSeek-V4-Flash-0731. Bold marks the best variant within each budget and metric.
<table><tr><td>bpw</td><td>Variant</td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td></tr><tr><td>2.0</td><td>NatScale w/o local RMS error</td><td>0.655823</td><td>73.9828</td><td>7.726484</td></tr><tr><td>2.0</td><td>NatScale w/o bit decay</td><td>1.102132</td><td>65.5826</td><td>11.946395</td></tr><tr><td>2.0</td><td>NatScale w/o gradient weights</td><td>0.912658</td><td>69.1134</td><td>10.175868</td></tr><tr><td>2.0</td><td>NatScale w/o gradient smoothing</td><td>0.609329</td><td>74.8965</td><td>7.378263</td></tr><tr><td>2.0</td><td>NatScale w/o exposure magnitude</td><td>0.787991</td><td>71.1932</td><td>8.884293</td></tr><tr><td>2.0</td><td>NatScale (ours, τ = 1)</td><td>0.729311</td><td>72.1534</td><td>8.364731</td></tr><tr><td>2.5</td><td>NatScale w/o local RMS error</td><td>0.391052</td><td>80.3449</td><td>6.339849</td></tr><tr><td>2.5</td><td>NatScale w/o bit decay</td><td>0.639630</td><td>73.7433</td><td>7.553834</td></tr><tr><td>2.5</td><td>NatScale w/o gradient weights</td><td>0.401876</td><td>79.5743</td><td>6.266410</td></tr><tr><td>2.5</td><td>NatScale w/o gradient smoothing</td><td>0.367891</td><td>81.3003</td><td>6.207998</td></tr><tr><td>2.5</td><td>NatScale w/o exposure magnitude</td><td>0.381792</td><td>80.2033</td><td>6.222373</td></tr><tr><td>2.5</td><td>NatScale (ours, τ = 1)</td><td>0.336141</td><td>81.5714</td><td>5.961485</td></tr><tr><td>3.0</td><td>NatScale w/o local RMS error</td><td>0.242478</td><td>84.7367</td><td>5.720269</td></tr><tr><td>3.0</td><td>NatScale w/o bit decay</td><td>0.295438</td><td>82.9395</td><td>5.789639</td></tr><tr><td>3.0</td><td>NatScale w/o gradient weights</td><td>0.198781</td><td>85.8331</td><td>5.530394</td></tr><tr><td>3.0</td><td>NatScale w/o gradient smoothing</td><td>0.217450</td><td>85.6435</td><td>5.595068</td></tr><tr><td>3.0</td><td>NatScale w/o exposure magnitude</td><td>0.211537</td><td>85.4875</td><td>5.545926</td></tr><tr><td>3.0</td><td>NatScale (ours, τ = 1)</td><td>0.193588</td><td>86.1609</td><td>5.488280</td></tr></table>

Table 8: Joint versus separate projection errors on Qwen3.8-Flash-Next at the NatScale default. Both variants use raw RMS and τ = 1 under each setting’s fixed payload ceiling. Top1 is native agreement (%). Bold marks the better of the two scores within each row and metric, using unrounded values.
<table><tr><td>Setting</td><td colspan="3">Joint error</td><td colspan="3">Separate error</td></tr><tr><td></td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td><td>KLD↓</td><td>Top1 (%) ↑</td><td>PPL↓</td></tr><tr><td colspan="7">Qwen3.8-Flash-Next</td></tr><tr><td>2.0 bpw</td><td>0.329276</td><td>80.2170</td><td>5.067797</td><td>0.326225</td><td>80.2569</td><td>5.073339</td></tr><tr><td>2.5 bpw</td><td>0.194086</td><td>84.8573</td><td>4.834395</td><td>0.195639</td><td>84.9175</td><td>4.831680</td></tr><tr><td>3.0 bpw</td><td>0.124340</td><td>87.6714</td><td>4.743009</td><td>0.123854</td><td>87.6423</td><td>4.741410</td></tr></table>

Table 9: Downstream comparison of joint and separate-error NatScale on Qwen3.8-Flash-Next at nominal 2.5 routed bpw, with thinking disabled. Cells show percentages and correct counts over the full denominator. AIME is mean response accuracy on AIME 2025; GPQA is Diamond; TQA is TruthfulQA binary; LCB is LiveCodeBench v6 pass@1. Bold marks the higher point estimate on each task.
<table><tr><td>Error score</td><td>AIME</td><td>GPQA</td><td>TQA</td><td>LCB</td></tr><tr><td>Qwen3.8-Flash-Next</td><td></td><td></td><td></td><td></td></tr><tr><td>NatScale w/ separate errors</td><td>65.00 (78/120)</td><td>74.75 (148/198)</td><td>93.42 (738/790)</td><td>62.86 (110/175)</td></tr><tr><td>NatScale (joint, ours)</td><td>68.33 (82/120)</td><td>78.28 (155/198)</td><td>94.05 (743/790)</td><td>61.14 (107/175)</td></tr></table>

We next score each unit’s candidate configurations. For a unit $u ,$ let $P _ { u }$ denote its projections and c their joint configuration. Routed and shared experts retain the complete joint G/U/D MSE $M _ { s , u } ( c )$ , including cross terms, measured on fixed native inputs. For attention and MHC, we instead sum the single-projection module-output MSEs $M _ { s , u , p } ( c _ { p } )$ , omitting cross terms between their projection perturbations. Averaging the AlphaQ structural factors with equal shares gives $\begin{array} { r } { \overline { { A } } _ { u } ( c ) \ : = \ : | P _ { u } | ^ { - 1 } \sum _ { p \in P _ { u } } A _ { u , p } ( c _ { p } ) } \end{array}$ . The resulting

score is

$$
\begin{array} { r l r } {  { \widetilde { M } _ { s , u } ( c ) = } } \\ & { } & \\ & { } & { \{ M _ { s , u } ( c ) , \qquad \mathrm { r o u t e d / s h a r e d ~ e x p e r t } ,  } \\ & { } & {  \sum _ { p \in { \cal P } _ { u } } { M _ { s , u , p } ( c _ { p } ) } , \quad \mathrm { a t t e n t i o n / M H C } ,  } \\ & { } & {  d _ { u } ( c ) = } \\ & { } & {  g _ { u } ^ { \beta } \overline { { A } } _ { u } ( c ) \sum _ { s = 1 } ^ { 6 } \pi _ { s } f _ { s , u } \sqrt { \operatorname* { m a x } \{ \widetilde { M } _ { s , u } ( c ) , 0 \} } .  } \end{array}\tag{38}
$$

Here, $\pi _ { s } = 1 / 6$ gives equal weight to each fold, while $f _ { s , u }$ is the native activation frequency for routed experts and one otherwise. The zero floor handles numerical roundoff without subtracting any candidate minimum. For attention and MHC, the single-projection MSEs are first added, then converted to RMS by taking the square root. As with the joint expert error, this square root is taken before averaging across folds.

To obtain the sensitivity weights in this score, we compute one conditional second moment per unit from native output gradients of the summed cross-entropy loss, then normalize these moments by a common mean over U. Fitting a Weibull distribution to positive $\sqrt { g _ { u } }$ gives $\dot { k } \ =$ 0.7956516957758913 and the shared exponent $\beta = \widehat { k } / 4 = 0 . 1 9 8 9 1 2 9 2 3 9 4 3 9 7 2 8 4$

The calibration data used for these statistics comprise 1,048,576 tokens: 220 Magpie, 234 UltraChat and 58 WritingPrompts sequences of length 2,048, rendered with the native chat template. Splitting each source by record-hash parity produces the six equally weighted folds used in the score. Each fold contributes all of its data to local error estimation. Neither the exponent nor candidate weights are refitted on WikiText-2.

## H.2 Storage Budget

The scores above guide allocation under a wholemodel storage constraint that also accounts for tensors with fixed encodings. Per-layer embedding (PLE) tables and token embeddings retain the UD encodings. The output head, routers, QSA indexer query/key, GDN alpha/beta and MHC inject projections retain fixed NINT8 encodings; normalization, convolution and the final gated-residual module retain their native configurations. These fixed tensors consume $B _ { \mathrm { f i x e d } } = 3 3 , 0 0 0 , 3 9 1 , 1 0 7$ bytes. After reserving a further $B _ { \mathrm { r e s e r v e } } = 3 6 { , } 4 3 6 { , } 9 9 2$ bytes for headers and codebooks, we solve

$$
\begin{array} { r l r } & { } & { \underset { c _ { u } \in \mathcal { C } _ { u } } { \mathrm { m i n } } \displaystyle \sum _ { u \in \mathcal { U } } d _ { u } ( c _ { u } ) } \\ & { } & { \mathrm { s u b j e c t ~ t o ~ } B _ { \mathrm { f i x e d } } + B _ { \mathrm { r e s e r v e } } } \\ & { } & { + \displaystyle \sum _ { u \in \mathcal { U } } C _ { u } ( c _ { u } ) \leq B _ { \mathrm { U D } } . } \end{array}\tag{39}
$$

We solve this problem using lower-hull allocation and feasible discrete rounding (Section 3.4), without imposing per-layer or projection-type ceilings.

## H.3 Evaluation and Results

We evaluate the resulting whole-model allocations against four released GSQ-RCO checkpoints (ISTA-DASLab, 2026). GSQ-RCO combines Gumbel-Softmax scalar quantization (Dadgarnia et al., 2026) with Riemannian constrained optimization (Helcig and Alistarh, 2026). For this comparison, NatScale and GSQ-RCO share 580 frozen WikiText-2 contexts of 512 tokens, the native reference hidden states and output head, and 147,900 predictions at logits 256–510. Full-vocabulary KLD uses FP64 reduction without probability cutoffs; Top1 compares native and candidate argmaxes.

With these evaluation inputs and metrics fixed, each method runs on its corresponding backend. NatScale propagates decoded BF16 candidates through the native Transformers graph with its fixed NINT8 head and recomputed routes. GSQ-RCO retains the released GGUF weights and llama.cpp revision 3cf03257, with lossless reference-logit transfer and full-vocabulary metrics.

Alongside these locally evaluated results, Table 2 includes UD scores reported by Unsloth (Unsloth, 2026).

## AI use statement

Generative AI assisted language editing, literature discovery, LAT<sub>E</sub>X formatting, and analysis-code review. The authors verified sources, code, and numerical claims, reviewed the manuscript, and take responsibility for all content and artifacts.