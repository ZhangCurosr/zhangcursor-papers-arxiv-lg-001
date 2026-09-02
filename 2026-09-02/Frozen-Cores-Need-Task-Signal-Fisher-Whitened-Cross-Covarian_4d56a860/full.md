# Frozen Cores Need Task Signal: Fisher-Whitened Cross-Covariance for Parameter-Efficient Adaptation

Wentao Ye<sup>1,2,\*</sup>, Zhanming Shen<sup>1,\*</sup>, Zhiqing Xiao<sup>1</sup>, Yao Ding<sup>2</sup>, Haobo Wang<sup>1,†</sup>, Gang Chen<sup>1</sup> <sup>1</sup> Zhejiang University <sup>2</sup>Hunyuan, Tencent {yewt01, z.shen, wanghaobo}@zju.edu.cn

## Abstract

Parameter-efficient fine-tuning is usually framed as a question of how many parameters to update. Under a severe trainable-state budget, however, where those coefficients act is equally consequential. We study this choice through frozen-core adaptation: a calibration pass fixes left and right bases for each weight matrix, and fine-tuning optimizes only an r × r core. This removes the ability of trainable factors to repair a poor initial span and makes subspace quality directly observable. We introduce FCCA, which estimates the signed input– error cross-covariance, whitens it with diagonal Fisher moments, truncates it in the resulting local metric, maps the selected directions back, and applies thin QR to obtain stable core coordinates. Under a matched r<sup>2</sup> budget, we compare eight basis constructors on 11 tasks, four model settings, and three seeds. On Qwen2.5- 3B, FCCA reaches an 83.0 macro-average, 2.3 points above the next-best matched-budget constructor, and exceeds its unwhitened RawGrad control on all 11 tasks. It ranks first at all three Qwen scales and finishes within 0.13 points of the best method on Llama-3.2-1B. Controlled ablations show gains of 2.7–17.2 points from whitening and identify QR as necessary for stable core optimization in the tested regime. Finally, FCCA comes within 0.32 and 0.23 average points of LoRA and DoRA while optimizing 36.9K rather than roughly 7.4M parameters. These results show that a carefully selected fixed span can recover most of the benefit of movable low-rank factors at a much smaller trainable and optimizer-state cost.<sup>1</sup>

## 1 Introduction

Parameter-efficient fine-tuning (PEFT) is commonly presented as a parameter-count problem: keep the pretrained model frozen, introduce a small adapter, and optimize as few new values as possible. This view explains much of LoRA’s appeal (Hu et al., 2022), but it leaves a second question implicit. When the budget is extremely small, the location of the trainable coefficients can matter as much as their number. A handful of coefficients placed in a task-relevant update span may be useful; the same coefficients placed in directions inherited from pretraining, random initialization, or poorly scaled gradients may be nearly inert.

Low-rank adapters normally hide this distinction. In LoRA, two trainable factors jointly determine the row and column spaces of the update and the coefficients within those spaces. They therefore perform two jobs at once: they must discover a useful subspace and optimize the update inside it. End-task accuracy conflates these effects. A favorable initialization may help, but an initially weak span can also rotate during fine-tuning, making it difficult to tell whether a method selected good directions or merely recovered from bad ones. This ambiguity is especially important when data, steps, optimizer state, or communication are constrained, because the opportunity to revise a span is limited.

We isolate subspace selection withfrozen-core adaptation. For a pretrained matrix $W \in \mathbb { R } ^ { m \times n } ,$ calibration fixes bases $P \in \mathbb { R } ^ { m \times r }$ and $Q \in \mathbb { R } ^ { n \times r } .$ and fine-tuning optimizes only the core in $P R Q ^ { \top }$ Matching the model, target matrices, rank, and $r ^ { 2 }$ trainable budget makes basis choice directly comparable. As a diagnostic, the protocol reveals how much downstream value is already present before a span can rotate; as an adaptation model, it represents settings where gradients, optimizer states, or transmitted updates must remain tiny. The fixed bases still have a storage cost, so the efficiency claim concerns trainable and optimizer state rather than a universally tiny task artifact.

What information should define such a span? Weight-spectrum bases emphasize directions prominent in pretraining but never observe the downstream loss (Bałazy et al., 2025; Wang et al., 2025). Activation- and marginal-Fisher methods observe the calibration distribution, yet they do not retain which activation produced which signed error (Paischer et al., 2025; Han and Guo, 2026). A raw task-gradient basis preserves that pairing, but an ordinary SVD imposes rank in Euclidean coordinates: high-variance input or error dimensions can dominate even when movement along them is locally expensive. These constructors therefore capture different pieces of the same object. The missing link is a way to preserve the signed task direction while judging it in the local geometry of the loss, rather than yet another source of marginal energy. A strong frozen span should do both before the rank constraint discards information.

![](images/907462d97a1ee26f4d3390306940c93b0de0fca801c3be6c38b57563cea4f5b6.jpg)  
Figure 1: Panel (a) sketches the design space of frozen-span construction: useful spans should preserve both signed task signal and local geometry, and the labeled positions indicate which information each constructor retains before truncation. Panel (b) shows FCCA: calibration estimates the signed cross-covariance and diagonal Fisher moments; whitening imposes the rank constraint in Fisher coordinates; inverse scaling and thin QR map the retained directions back to stable frozen bases; only the core R is trained.

Figure 1 organizes this design space and summarizes FCCA (Fisher-whitened cross-covariance adaptation). For layer inputs x and backpropagated output errors δ, calibration estimates the signed cross-moment $G = \mathbb { E } [ \delta x ^ { \top } ]$ together with input- and output-side second moments A and D. We first form $\widetilde { G } = D ^ { - 1 / 2 } G A ^ { - 1 / 2 }$ and apply the rank constraint in these Fisher-scaled coordinates. Its leading singular directions solve a local rankconstrained objective; mapping them back gives raw bases in the original parameter coordinates. Thin QR then orthonormalizes those bases without changing the representable update family. This order is essential: whitening after truncation could rescale a chosen span, but it could not recover directions already discarded by the unscaled SVD. QR addresses a separate problem by presenting the retained span in coordinates that a common core optimizer can train.

Across 11 tasks and four model settings, FCCA ranks first on Qwen2.5-1.5B, 3B, and 7B and is essentially tied for first on Llama-3.2-1B. At 3B, it reaches an 83.0 mean, leads the next-best matchedbudget constructor by 2.3 points, and exceeds Raw-Grad on every task. Whitening helps all six mechanism tasks; calibration-size experiments show that useful directions emerge from modest taskmatched data; and rank, module, QR, and densecurvature controls separate span selection from coordinate conditioning. The fixed span also remains within 0.32 points of LoRA and 0.23 points of DoRA while optimizing roughly 200× fewer values.

This work makes four contributions. (1) We formulate frozen-subspace selection in a local Fisher metric and derive the truncated whitened crosscovariance as its rank-r solution. (2) We introduce a unified frozen-core evaluation across eight constructors, 11 tasks, three Qwen scales, a Llama architecture, and three seeds, exposing basis quality under an exactly matched r<sup>2</sup> trainable budget. (3) Controlled interventions identify the roles of signed task signal, whitening, calibration coverage, span size, and adapted modules. (4) Additional QR and retuned no-QR controls show that even a useful span must still be expressed in coordinates the optimizer can train. Together, these results sharpen the main conclusion: low-rank adaptation benefits from choosing directions in the metric where information will later be discarded.

## 2 Related Work

Parameter-efficient adaptation. Adapters, prompt and prefix tuning, BitFit, and (IA)<sup>3</sup> reduce trainable state through different parameterizations (Houlsby et al., 2019; Li and Liang, 2021; Lester et al., 2021; Ben Zaken et al., 2022; Liu et al., 2022). LoRA trains both low-rank factors (Hu et al., 2022); DoRA separates direction from magnitude (Liu et al., 2024); PiSSA initializes trainable factors from principal weight components (Meng et al., 2024); and AdaLoRA allocates rank dynamically (Zhang et al., 2023). In contrast, our controlled setting freezes both bases and trains only a square core, isolating the quality of the selected subspace.

Fixed or data-derived subspaces. VeRA uses shared random low-rank matrices with lightweight trainable scaling (Kopiczko et al., 2024). LoRA-XS places a small trainable matrix between weight-SVD bases (Bałazy et al., 2025), whereas MiLoRA emphasizes minor singular components (Wang et al., 2025). EVA derives bases from activation variance (Paischer et al., 2025); FiLoRA uses Fisher-guided marginal subspaces (Han and Guo, 2026); LoRA-GA and gradient-compression methods exploit task gradients in trainable-factor settings (Wang et al., 2024; Hao et al., 2024). We instantiate each applicable idea in the same frozenbasis, trainable-core form so that performance reflects subspace choice instead of differences in native parameterization.

Curvature and natural gradients. Naturalgradient and K-FAC methods rescale optimization by local parameter geometry (Amari, 1998; Martens and Grosse, 2015). FCCA applies this principle to subspace selection: it preserves the signed input–error cross-covariance and normalizes both sides before the rank constraint is imposed. This is distinct from selecting input and output eigenspaces independently.

<table><tr><td>Constructor</td><td>Task signal</td><td>Geometry</td><td>Source</td></tr><tr><td>Random</td><td>no</td><td>no</td><td>random draw</td></tr><tr><td>LoRA-XS/MiLoRA</td><td>no</td><td>weight</td><td>W spectrum</td></tr><tr><td>EVA-core</td><td>indirect</td><td>input</td><td>activations</td></tr><tr><td>FiLoRA-core</td><td>indirect</td><td>input+output</td><td> $A , D$ </td></tr><tr><td>RawGrad</td><td>yes</td><td>no</td><td>G</td></tr><tr><td>FCCA</td><td>yes</td><td>input+output</td><td> $G , { \bar { A } } , D$ </td></tr></table>

Table 1: Information used by the matched-budget basis constructors. “Indirect” means the data distribution is observed without the signed input–error cross term.

Table 1 summarizes the information available to each constructor. Weight-spectrum methods retain directions prominent in pretraining but never observe the downstream task. EVA observes task inputs, and FiLoRA observes input and output marginals, but neither retains the signed pairing between an activation and the error it produced. RawGrad retains that pairing in a Euclidean metric. Among the tested methods, FCCA alone combines the signed cross term with both marginal geometries before truncation.

## 3 Method

## 3.1 Frozen-core adaptation

For a frozen linear map $y \ = \ W x$ , we use $y =$ $( W + s P R Q ^ { \top } ) x$ . The columns of $P \in \mathbb { R } ^ { m \times r }$ and $Q \in \mathbb { R } ^ { n \times r }$ are orthonormal, s is a fixed adapter scale, and only $R \in \mathbb { R } ^ { r \times r }$ is trainable. Calibration fixes the bases, and training begins from $R = 0$ Across L target matrices, the trainable budget is exactly $L r ^ { 2 }$ , independent of matrix width, and the update family $\mathcal { S } ( P , Q ) = \{ P R Q ^ { \top } : R \in \mathbb { R } ^ { r \times r } \}$ cannot rotate during optimization.

A directly observable criterion for subspace quality. Let $G = \nabla _ { W } \mathcal { L }$ be the task gradient of the frozen matrix. At the zero core, $\nabla _ { \boldsymbol { R } } \mathcal { L } \ : =$ $s P ^ { \top } G Q .$ , so a gradient step of size γ reduces the linearized loss in proportion to $\gamma s ^ { 2 } \| P ^ { \top } G Q \| _ { F } ^ { 2 }$ . Because s is shared across constructors, the projected norm $\| P ^ { \top } G Q \| _ { F }$ is a directly observable criterion for basis quality. A useful frozen subspace must capture signed task gradient in both its left and right spans. High activation variance, large weight singular values, or large marginal Fisher eigenvalues alone do not ensure this. Unlike trainable LoRA factors, a frozen basis cannot rotate toward a missed direction during fine-tuning.

The Euclidean criterion is nevertheless coordinate dependent: high-variance activation or error coordinates can dominate G even when movement along them is locally expensive. We therefore impose the rank constraint in a Fisher-scaled metric rather than truncating the raw gradient directly.

For each calibration token, let $x \in \mathbb { R } ^ { n }$ be the matrix input and $\delta = \partial \mathcal { L } / \partial y \in \mathbb { R } ^ { m }$ the backpropagated output error. We estimate the signed cross moment $G = \mathbb { E } [ \delta x ^ { \top } ]$ and diagonal moments $a = \mathbb { E } [ x ^ { \odot 2 } ]$ and $d = \mathbb { E } [ \delta ^ { \odot 2 } ]$ . The default factors are $A = \mathrm { d i a g } ( a ) + \epsilon I$ and $D = \mathrm { d i a g } ( d ) + \epsilon I$ This diagonal estimator avoids fitting two dense width-by-width covariances from only a few hundred calibration examples.

## 3.2 Fisher-whitened cross-covariance

Consider the local rank-constrained objective

$$
\begin{array} { r l } { \underset { \operatorname { r a n k } ( \Delta W ) \leq r } { \operatorname* { m i n } } } & { \left. G , \Delta W \right. } \\ & { + \displaystyle \frac { 1 } { 2 \eta } \left\| D ^ { 1 / 2 } \Delta W A ^ { 1 / 2 } \right\| _ { F } ^ { 2 } . } \end{array}\tag{1}
$$

The quadratic term defines a Kronecker-factored Fisher metric for the linear layer. Defining Ge = $D ^ { - 1 / 2 } G A ^ { - 1 / 2 }$ moves the signed task gradient into

Fisher-whitened coordinates before any low-rank truncation.

Theorem 1. Let A and D be positive definite, and let $[ { \widetilde { G } } ] ,$ <sub>r</sub> denote a best rank-r approximation to $\widetilde { G }$ A minimizer ofEq. (1) is

$$
\Delta W ^ { * } = - \eta D ^ { - 1 / 2 } [ \widetilde { G } ] _ { r } A ^ { - 1 / 2 } .\tag{2}
$$

The proof changes variables to Fisher-whitened coordinates and applies the Eckart–Young–Mirsky theorem (Eckart and Young, 1936; Mirsky, 1960); Appendix A gives the derivation. If $\widetilde { G } = U \Sigma V ^ { \top }$ Eq. (2) yields raw bases $P _ { 0 } = D ^ { - 1 / 2 } U _ { r }$ and $Q _ { 0 } =$ $A ^ { - 1 / 2 } V _ { r }$ . RawGrad is the exact no-whitening control obtained by factorizing $G$ instead. Marginalcurvature methods select directions from A and $D$ separately and therefore lose the signed activation– error pairing.

## 3.3 QR as a coordinate reparameterization

Inverse whitening can leave $P _ { 0 }$ and $Q _ { 0 }$ severely anisotropic. We compute thin QR factorizations $P _ { 0 } ~ = ~ P T _ { P }$ and $Q _ { 0 } ~ = ~ Q T _ { Q }$ and train the orthonormal-basis adapter $P R Q ^ { \top }$ . Because $P _ { 0 } Z Q _ { 0 } ^ { \top } = P ( T _ { P } Z T _ { Q } ^ { \top } ) Q ^ { \top }$ , QR preserves the representable update family and changes only the coordinates of the square core. The controlled ablation in Table 6 shows that these coordinates are essential under the shared learning rate.

```latex
Algorithm 1 FCCA basis construction and core
training
Require: frozen model, calibration set $\mathcal { D } _ { c }$ , target matrices
M, rank r
1: for $\ell \in \mathcal { M }$ do
2: initialize $G _ { \ell } , a _ { \ell } , d _ { \ell } \gets 0$
3: end for
4: for calibration batch $B \subset \mathcal { D } _ { c }$ do
5: run one forward/backward pass
6: for $\ell \in \mathcal { M }$ do
7: $\begin{array} { r } { G _ { \ell } + = \sum _ { t } \delta _ { \ell , t } x _ { \ell , t } ^ { \top } } \end{array}$
8: $\begin{array} { r } { \underset { \ldots } { a \ell } + = \sum _ { t } x _ { \ell , t } ^ { \odot 2 } ; d _ { \ell } + = \sum _ { t } \delta _ { \ell , t } ^ { \odot 2 } } \end{array}$
9: end for
10: end for
11: for $\ell \in \mathcal { M }$ do
12: $A _ { \ell } \gets \mathrm { d i a g } ( a _ { \ell } ) + \epsilon I ; D _ { \ell } \gets \mathrm { d i a g } ( d _ { \ell } ) + \epsilon I$
13: $\widetilde { G } _ { \ell } \gets D _ { \ell } ^ { - 1 / 2 } \dot { G } _ { \ell } A _ { \ell } ^ { - 1 / 2 }$
14: $U _ { r } , \Sigma _ { r } , V _ { r }  \mathrm { S V D } _ { r } ( \widetilde { G } _ { \ell } )$
15: $P _ { 0 }  D _ { \ell } ^ { - 1 / 2 } U _ { r } ; Q _ { 0 }  A _ { \ell } ^ { - 1 / 2 } V _ { r }$
16: thin QR of $P _ { 0 } , Q _ { 0 } ;$ attach a zero core $R _ { \ell }$
17: end for
18: discard calibration statistics and train only {R<sub>ℓ</sub>}
```

## 3.4 Construction, training, and storage costs

Algorithm 1 summarizes construction and training. For each $m \times n$ target matrix, calibration accumulates one $m \times n$ cross moment and $m + n$ diagonal moments, followed by a truncated rank-r $\operatorname { s v p }$ . Our implementation hooks all target layers in one grouped forward/backward pass and stores the statistics on CPU. Cross moments can dominate host memory, and the backward pass is more expensive than forward-only activation collection. This construction cost is paid once per task. Streamed or randomized accumulation could reduce the footprint but is not evaluated here.

After construction, gradients and optimizer states are required for only $r ^ { 2 }$ values per target matrix, versus $r ( m + n )$ values for a trainablebasis rank-r adapter. The fixed bases still occupy $r ( m + n )$ storage when retained separately for each task. Our efficiency claim therefore concerns trainable and optimizer state: an unmerged adapter must store P and $Q ,$ whereas a merged update requires a task-specific copy of the base model.

## 4 Experimental Setup

Models and tasks. Qwen2.5-3B-Instruct is the primary model; Qwen2.5-1.5B/7B-Instruct and Llama-3.2-1B-Instruct provide scale and architecture checks (Qwen Team, 2024; Llama Team, AI @ Meta, 2024). The 11-task suite contains SVAMP and GSM8K (Patel et al., 2021; Cobbe et al., 2021); SST-2, QNLI, CoLA, RTE, and MRPC (Wang et al., 2018); and ARC-Challenge, OpenBookQA, HellaSwag, and WinoGrande (Clark et al., 2018; Mihaylov et al., 2018; Zellers et al., 2019; Sakaguchi et al., 2020). We evaluate fixed subsets of 300 examples per math task and 500 per remaining task. Appendix B and Table 8 give dataset configurations, sizes, and training lengths.

Common adapter protocol. Every matchedbudget method uses rank $r \ = \ 1 6 ,$ , adapts all $q / k / v / o$ projections, runs in bf16 with effective batch size 8, initializes the core to zero, and trains one $r \times r$ core per target matrix. Qwen2.5-3B therefore has 36,864 trainable parameters. We compare FCCA, RawGrad, FiLoRA-core, MiLoRA-core, LoRA-XS-core, VeRA-core, EVA-core, and random orthonormal bases. The “-core” suffix denotes our controlled fixed-basis instantiation, not the original method’s full recipe; Appendix C gives the exact mappings. LoRA, DoRA, and PiSSA serve as trainable-basis rank-16 references.

Evaluation and selection. SVAMP and GSM8K use greedy decoding and exact match with a shared deterministic parser; the remaining tasks use perchoice conditional log likelihood. Appendix B.2 motivates this task-format-specific choice and clarifies how the cross-task mean should be interpreted. We run seeds 42, 43, and 44 and report mean ± standard deviation. Held-out validation selects checkpoints and hyperparameters. FCCA fixes the learning rate at $5 \times 1 0 ^ { - 3 }$ and searches warmup {0.03, 0.1}× calibration size {128, 256}; each matched-budget baseline searches learning rate $\{ 2 , 5 \} \times 1 0 ^ { - 3 }$ at warmup 0.1. Thus baselines can tune learning rate, although FCCA evaluates more total warmup/calibration configurations. Table 9 lists the complete protocol. The primary paired test is a two-sided Wilcoxon signed-rank test over the 11 unrounded task means.

## 5 Results

## 5.1 Matched-budget performance at 3B

Table 2 gives the complete Qwen2.5-3B comparison. FCCA reaches an 83.0 mean, 2.3 points above the next-best matched-budget constructor, LoRA-XS-core. It is best or tied on seven tasks and remains within 1.2 points of the best method on the other four. The largest gains over RawGrad occur on OpenBookQA (+10.0), HellaSwag (+7.4), and WinoGrande (+18.8), but the advantage is not carried by those tasks alone: every one of the 11 differences is positive.

Using unrounded task means, FCCA exceeds RawGrad by 4.78 points on average $( p = . 0 0 1 )$ and every unadjusted Wilcoxon comparison against a matched-budget baseline is below .05. Table 3 summarizes the paired tests; Appendix F reports win counts and paired-t references. The task is the unit of replication, so these tests support a cross-task comparison and should not be read as significance claims for each individual dataset.

At the principal 3B setting, FCCA’s advantage is broad across tasks and extends to every matchedbudget constructor. The $\mathrm { m a t c h e d } { - } r ^ { 2 }$ protocol is central to that interpretation. A trainable rank-r factor can rotate during optimization, so its final accuracy conflates initialization quality with the ability to repair a weak starting span. Freezing $P , Q$ removes that repair mechanism and makes basis selection directly observable. Frozen-core performance therefore diagnoses span quality and models state-constrained adaptation; it does not imply that practical adapters must always freeze their bases.

The ordering is not a simple complexity ladder. VeRA and FiLoRA-core win isolated tasks, showing that several low-dimensional spans can be adequate locally; their lower cross-task means show that this adequacy does not transfer reliably. RawGrad is the decisive control: it uses exactly the same signed cross-moment as FCCA, so its deficit on all 11 tasks isolates the metric applied at truncation, not access to additional supervision. EVA-core remains competitive at larger scales, but its high 3B variance on MRPC and HellaSwag exposes task-level brittleness in this campaign.

The weight-spectrum controls sharpen the mismatch. LoRA-XS-core freezes the dominant singular directions of W, whereas MiLoRA-core emphasizes the minor spectrum; neither prior transfers consistently. LoRA-XS-core is the strongest 3B alternative, yet EVA-core is runner-up in the other model settings, and MiLoRA-core is competitive only on a subset of classification and QA tasks. The problem is therefore not merely choosing the “wrong end” of the spectrum: spectral energy describes how pretraining uses a matrix, whereas a downstream update depends on how task errors couple to current activations. A frozen basis cannot rotate a generic spectral prior into that coupling.

The data-derived baselines expose the complementary boundary. FiLoRA-core edges FCCA on SVAMP (80.1 versus 79.7), and EVA-core nearly ties it on Llama-1B; marginal statistics can therefore suffice when the update aligns with highvariance activation or curvature directions. Their failures are concentrated: at 3B, FiLoRA-core trails FCCA by 7.7 points on HellaSwag and 12.8 on WinoGrande, while EVA-core is lower and highly variable on HellaSwag. This pattern does not identify a unique causal mechanism, but it matches the information distinction in Table 1: marginals identify energetic directions, whereas $G = \mathbb { E } [ \delta x ^ { \top } ]$ retains their signed pairing with task errors. FCCA helps when both that pairing and the local metric matter before truncation.

## 5.2 Scale, architecture, and task-family coverage

Figure 2 shows that the main result is not confined to one model size. FCCA ranks first on Qwen2.5- 1.5B, 3B, and 7B, with means of 77.2, 83.0, and 87.5. Its lead over the strongest alternative is 0.5, 2.3, and 0.8 points, respectively, even though the identity of that alternative changes. At 7B, FCCA significantly exceeds six of seven controls using the supplied unrounded task means; EVA-core remains statistically comparable $( p = . 0 5 4 )$ . On Llama-3.2- 1B, EVA-core leads by only 0.1 point (68.2 versus 68.1), while FCCA still exceeds RawGrad on eight tasks.

<table><tr><td>Task</td><td>FCCA</td><td></td><td></td><td></td><td>RawGrad FiLoRA-c MiLoRA-c LoRA-XS-c</td><td>VeRA-c</td><td>EVA-c</td><td>Random</td></tr><tr><td>SVAMP</td><td>79.7±1.5</td><td>74.9±2.0</td><td> ${ \bf 8 0 . 1 \pm 0 . 4 }$ </td><td> $7 1 . 2 { \pm } 1 . 1 $ </td><td> $6 9 . 9 2 1 . 8 $ </td><td> $7 2 . 6 { \pm } 1 . 2 $ </td><td> $7 8 . 9 \pm 1 . 2$ </td><td> $7 1 . 2 { \pm } 1 . 0 $ </td></tr><tr><td>GSM8K</td><td> $6 8 . 1 { \pm } 0 . 8 $ </td><td>66.7±1.5</td><td> $6 6 . 1 \pm 2 . 0$ </td><td> $6 5 . 2 { \pm } 1 . 1 $ </td><td> $6 5 . 0 { \pm } 1 . 6 $ </td><td> ${ \bf 6 9 . 3 \pm 0 . 8 }$ </td><td> $6 7 . 8 { \pm } 0 . 6 $ </td><td> $6 7 . 0 { \pm } 2 . 7 $ </td></tr><tr><td>SST-2</td><td> ${ \bf 9 5 . 1 } { \bf \pm 0 . 2 }$ </td><td>93.6±0.6</td><td> $9 4 . 4 \pm 0 . 2 $ </td><td> $9 3 . 7 { \pm } 0 . 2 $ </td><td> $9 3 . 5 { \scriptstyle \pm 0 . 7 }$ </td><td> $9 2 . 3 { \pm } 0 . 2 $ </td><td> $9 4 . 8 { \pm } 0 . 3 $ </td><td> $9 2 . 5 { \pm } 0 . 2 $ </td></tr><tr><td>QNLI</td><td> ${ \bf 8 8 . 9 2 1 . 2 }$ </td><td>86.9±0.9</td><td> $8 7 . 1 { \pm } 1 . 1 $ </td><td> $8 7 . 1 \pm 1 . 5$ </td><td> $8 6 . 9 { \pm } 1 . 3 $ </td><td> $8 4 . 8 { \pm } 0 . 9$ </td><td> ${ \bf 8 8 . 9 \pm 0 . 7 }$ </td><td> $8 5 . 1 \pm 0 . 2 $ </td></tr><tr><td>CoLA</td><td> $8 5 . 3 { \pm } 1 . 3 $ </td><td>84.1±1.0</td><td> $8 2 . 6 \pm 2 . 0$ </td><td> $8 5 . 4 \pm 1 . 0$ </td><td> $8 4 . 5 { \pm } 0 . 9 \ $ </td><td> $8 3 . 9 2 1 . 8$ </td><td> $\mathbf { 8 5 . 7 \pm 0 . 1 }$ </td><td> $8 4 . 1 \pm 0 . 9$ </td></tr><tr><td>RTE</td><td> ${ \bf 8 9 . 3 \pm 0 . 9 }$ </td><td> $8 7 . 5 { \pm } 0 . 6 $ </td><td> $8 8 . 2 \pm 0 . 7 $ </td><td> $8 8 . 2 \pm 0 . 5$ </td><td>89.3±1.0</td><td> $8 8 . 2 \pm 0 . 5$ </td><td> $8 8 . 7 \pm 1 . 2 $ </td><td> $8 7 . 8 { \pm } 0 . 2 \ $ </td></tr><tr><td>MRPC</td><td> $\mathbf { 8 9 . 4 \pm 1 . 3 }$ </td><td>87.7±0.7</td><td> $8 7 . 2 \pm 3 . 4 $ </td><td> $8 6 . 5 { \pm } 0 . 5 $ </td><td> $8 6 . 4 \pm 1 . 1$ </td><td> $8 0 . 7 { \pm } 0 . 3 $ </td><td> $8 1 . 9 { \pm } 9 . 6 $ </td><td> $8 3 . 3 { \pm } 0 . 9$ </td></tr><tr><td>ARC-C</td><td>83.1±1.1</td><td>81.3±0.9</td><td> $8 2 . 3 { \pm } 0 . 5 $ </td><td> $8 1 . 3 { \pm } 0 . 9$ </td><td> $8 3 . 0 { \pm } 1 . 0 \ $ </td><td> ${ \bf 8 3 . 9 2 0 . 7 }$ </td><td> $8 2 . 1 \pm 1 . 3$ </td><td> $8 2 . 5 { \pm } 0 . 5 $ </td></tr><tr><td>OBQA</td><td> ${ \bf 8 5 . 9 2 1 . 0 }$ </td><td>75.9±1.6</td><td> $7 8 . 8 \pm 0 . 2$ </td><td> $8 4 . 7 { \pm } 0 . 7 $ </td><td> $8 4 . 1 { \pm } 0 . 8 $ </td><td> $8 5 . 6 { \pm } 0 . 2 $ </td><td> $8 3 . 7 \pm 1 . 5$ </td><td> $8 3 . 7 \pm 1 . 1$ </td></tr><tr><td>HellaSwag</td><td>75.6±1.0</td><td>68.2±2.0</td><td> $6 7 . 9 { \pm } 0 . 3 $ </td><td> $7 4 . 1 \pm 0 . 2$ </td><td> $7 4 . 3 { \pm } 1 . 1$ </td><td> $7 2 . 4 \pm 0 . 6$ </td><td> $5 7 . 5 { \pm } 2 1 . 2 $ </td><td> $7 1 . 7 { \pm } 0 . 2 $ </td></tr><tr><td>WinoG</td><td>72.8±0.6</td><td>54.0±5.8</td><td> $6 0 . 0 { \pm } 8 . 0 \ $ </td><td> $6 7 . 5 { \pm } 0 . 9 \ $ </td><td> $7 1 . 1 { \pm } 1 . 3 $ </td><td> $6 9 . 5 { \pm } 0 . 8 $ </td><td> $6 9 . 3 \pm 1 . 7$ </td><td> $6 8 . 3 { \pm } 1 . 2 $ </td></tr><tr><td>Mean</td><td>83.0</td><td>78.3</td><td>79.5</td><td>80.4</td><td>80.7</td><td>80.3</td><td>79.9</td><td>79.7</td></tr></table>

Table 2: Matched-budget results on Qwen2.5-3B-Instruct. Every frozen-core method trains only an r × r core with r = 16 in all $q / k / v / o$ projections. Entries are mean±std over three seeds (×100); the Mean row is the arithmetic mean of the 11 displayed task means. Bold marks the best displayed mean per row.

![](images/d759c4ae7729f547332b4d9ccea9e09d0364ed0a027f0b17e61b879b83985439.jpg)  
(a) Qwen2.5-1.5B.

![](images/4be52e89b96441e3cf6d61789f32a5f6551837f279dabf07f26e769acdfc80ff.jpg)  
(b) Qwen2.5-3B.

![](images/d6d88e9e53cb6db0a9345fe1eecb9f94453005188d36b32a22c7801df96b2a07.jpg)

![](images/ac8344ebca61ce6bbfdbb14c63785f02dfda4f504c709b0e56437fb320a03114.jpg)  
(c) Qwen2.5-7B.  
(d) Llama-3.2-1B.

Figure 2: Cross-model matched-budget comparison. Each panel orders the eight frozen-core constructors by 11-task mean. FCCA is orange, RawGrad blue, EVA-core navy, and the remaining methods gray; Appendix D reports exact means and task-level results.
<table><tr><td>Baseline</td><td>Mean ∆</td><td>Wilcoxon p</td></tr><tr><td>RawGrad</td><td>+4.78</td><td>.001</td></tr><tr><td>FiLoRA-core</td><td>+3.51</td><td>.002</td></tr><tr><td>MiLoRA-core</td><td>+2.57</td><td>&lt; .01</td></tr><tr><td>LoRA-XS-core</td><td>+2.31</td><td>&lt; .01</td></tr><tr><td>VeRA-core</td><td>+2.74</td><td>.014</td></tr><tr><td>EVA-core</td><td>+3.09</td><td>.007</td></tr><tr><td>Random</td><td>+3.28</td><td>&lt; .01</td></tr></table>

Table 3: FCCA minus each 3B baseline over 11 unrounded task means. Complete win counts, paired-t references, and multiplicity caveats appear in Appendix F.

FCCA leads across all three Qwen scales and remains within 0.1 point of the best Llama result. This boundary is informative. Larger backbones can expose several competitive low-dimensional spans, yet FCCA’s margin over RawGrad remains positive in all four settings: using the displayed task means, it wins 38 of 44 task comparisons, ties one, and loses five. Because the two methods use the same signed cross-moment G, their repeated separation isolates the metric imposed before truncation, not additional task information. The margin peaks at 3B instead of growing monotonically; the evidence supports a robust geometric correction without implying a scaling law.

The breadth also holds across task families. On Qwen2.5-3B, FCCA averages 73.9 on the two math tasks, 89.6 on the five GLUE tasks, and 79.4 on the four reasoning/QA tasks. These are 0.5, 1.4, and 1.2 points above the strongest alternative within each family. The largest family-level gain over RawGrad is on reasoning/QA (+9.5), but the GLUE and math results rule out an explanation tied only to generated arithmetic answers. Nor is the ranking driven by unusually small seed variance: FCCA’s median standard deviation is 1.0 point, versus 0.7– 1.2 for the alternatives. Complete 1.5B, 7B, and Llama results are in Appendix D.

## 5.3 Whitening changes which directions survive truncation

Figure 3(a) isolates the paper’s central intervention. Whitening improves every tested mechanism task, with two-sided gains of 2.7–17.2 points over the unscaled SVD. The effect spans math, language understanding, and commonsense reasoning, and is largest on RTE and WinoGrande. Twosided whitening is best or tied on four of six tasks; input-only is 0.1 point higher on ARC-C and 1.1 higher on RTE. Two-sided whitening is therefore the strongest overall default in this study, although input-only is slightly better on two individual tasks.

![](images/afdbd61d7f0ef81f8ac79a7359725ddd647f77bd84cd0f32173bd95fb9e57f9f.jpg)  
(a) Whitening gains by task.

![](images/d8ab87755b7b597f4df343047e0030c34dba3873287d31126204f3d17cccf197.jpg)  
(b) Accuracy versus calibration size.

![](images/6abc0b9fdca60d13fd39dc94d44e24ad9907c2849329c831ee87596be3e519ba.jpg)  
(c) Basis convergence.  
Figure 3: Mechanism and calibration diagnostics on Qwen2.5-3B. (a) Gain from one- and two-sided whitening over the unwhitened SVD. (b) Accuracy versus calibration size (mean ± standard deviation, three seeds). (c) Mean principal-angle cosine to the 512-example bases. Appendix G reports exact values and additional controls.

The comparison also separates task signal from local geometry. RawGrad preserves the signed activation–error pairing but imposes rank in Euclidean coordinates. EVA-core and FiLoRA-core encode aspects of the calibration distribution or marginal curvature without retaining that signed pairing. FCCA combines both exactly where information is discarded: before the rank-r truncation. This ordering matters because whitening after truncation could rescale a chosen span but could not recover directions already removed. Appendix A formalizes the same point through the local Fisher objective, while Appendix G reports the six-task values underlying the figure.

## 5.4 Calibration saturates before the basis fully converges

Figure 3(b)–(c) distinguish downstream sufficiency from numerical convergence. Across 64–512 examples, FCCA varies by only 1.3 points on SVAMP and 0.4 on ARC-C, and it remains 5.9–7.3 points above RawGrad at every tested SVAMP size. The selected bases continue to move over the same range: their principal-angle cosines first exceed .93 on both sides at 128 examples and approach one only at 512. Accuracy therefore plateaus before the estimated span becomes identical to the largest-calibration reference.

A plausible interpretation is that several nearby rank-16 spans contain similarly useful directions and the square core can reweight coordinates within any of them. The mismatched-source experiment provides a sharper boundary. Calibrating SVAMP with GSM8K costs 1.7 points, whereas ARC-C and SST-2 calibration cost 5.1 and 7.1. Increasing sample count therefore cannot substitute for representative task signal. The ordering matches

FCCA’s statistic: calibration sets both marginal scales and the orientation of $\delta x ^ { \top }$ . Related math data preserve more of this joint structure; additional out-of-domain examples can reduce variance around a mismatched population quantity without correcting its bias. Basis reuse therefore appears more plausible within task families or after a small target-task refresh. In the tested regime, coverage is more consequential than exact convergence to one reference basis: modest, task-matched calibration is sufficient, but general cross-domain basis reuse is not established.

Principal-angle convergence and downstream utility therefore answer different questions. The former measures recovery of a particular reference basis; the latter only requires a span whose projected gradient and local geometry are adequate for the task. This distinction explains why accuracy can saturate before the basis stabilizes and cautions against choosing calibration size from geometric convergence alone.

## 5.5 Accuracy–state trade-offs

<table><tr><td>Method</td><td>Mean</td><td>Trainable</td><td>∆ vs. FCCA</td></tr><tr><td>FCCA</td><td>83.0</td><td>36.9K</td><td>reference</td></tr><tr><td>LoRA</td><td>83.3</td><td>7.37M</td><td>+0.32</td></tr><tr><td>DoRA</td><td>83.2</td><td>7.54M</td><td>+0.23</td></tr><tr><td>PiSSA</td><td>80.4</td><td>7.37M</td><td>-2.60</td></tr></table>

Table 4: Qwen2.5-3B accuracy and trainable state. Means are over the 11-task suite.

Table 4 asks how much accuracy is lost when the selected span cannot move. FCCA’s 83.0 mean is 0.318 below LoRA and 0.227 below DoRA despite optimizing 36,864 rather than 7.37–7.54M parameters. It exceeds LoRA on four tasks and DoRA on five; the trainable factors retain their clearest advantages on OpenBookQA and ARC-C. The result is near-parity at a 200× reduction in trainable state. The complete task-level comparison is in Appendix E.

<table><tr><td>Method Init./calib.</td><td></td><td>Step Projected total</td><td></td><td>Peak GPU</td></tr><tr><td>FCCA</td><td></td><td> $3 5 \mathrm { ~ s ~ } ~ 0 . 8 1 \mathrm { ~ s ~ }$ </td><td>4.1 min</td><td> $6 . 9 2 \ : \mathrm { G B }$ </td></tr><tr><td>LoRA</td><td>none</td><td>0.92 s</td><td>4.0 min</td><td> $7 . 1 9 \mathrm { G B }$ </td></tr><tr><td>DoRA</td><td>none</td><td>1.31 s</td><td>5.7 min</td><td>7.27 GB</td></tr><tr><td>PiSSA</td><td></td><td>41 s 1.76 s</td><td>8.4 min</td><td>7.20 GB</td></tr></table>

Table 5: SVAMP phase profile on one H20; totals project approximately 262 optimizer steps. The first column reports FCCA calibration and PiSSA initialization; PiSSA uses a one-time weight-SVD to initialize its trainable factors.

The ordering clarifies what a frozen span gives up. LoRA and DoRA can rotate their factors throughout training, so they can repair calibration directions that are incomplete or task-locally misaligned. Their largest gains on OpenBookQA and ARC-C are consistent with a greater need for span revision on those tasks, although this comparison alone cannot establish the cause. PiSSA starts from dominant weight directions and then learns both factors; its lower 3B mean, driven especially by GSM8K, shows that a favorable weight-space initialization is not sufficient when the downstream update is poorly aligned with the pretrained spectrum. In contrast, FCCA spends its calibration budget on the downstream loss and relies on the square core only to recombine selected directions.

Table 5 places the state reduction in a runtime context. FCCA adds a one-time 35-second calibration pass, but its core-only updates yield a projected 4.1-minute run: close to LoRA (4.0), below DoRA (5.7), and roughly half PiSSA (8.4). The GPU-memory difference is modest because frozen model weights dominate at this scale; the more substantial savings are in gradients, optimizer state, and communication. The storage claim is narrower: before merging, task-specific $P , Q$ bases remain width dependent, and after merging deployment requires a task-specific model copy. Appendix H gives the full protocol.

Under these measurements, FCCA amortizes its calibration relative to DoRA after roughly 70 optimizer steps and relative to LoRA after roughly 318. The 262-step SVAMP run is therefore well beyond the DoRA break-even point and just below the LoRA break-even point. These thresholds are hardware- and implementation-specific, but they make the runtime trade-off explicit.

<table><tr><td>Setting</td><td>SVAMP</td><td>ARC-C</td><td>GSM8K</td></tr><tr><td>Span size</td><td> $( a l l q / k / v / o )$ </td><td rowspan="3"> $6 7 . 1 { \pm } 2 . 6 $   ${ \bf 6 7 . 3 2 1 . 2 }$ </td></tr><tr><td> $r = 8$ </td><td> $7 3 . 9 2 0 . 3 $   ${ \bf 8 4 . 1 } { \pm } { \bf 0 . 2 }$ </td></tr><tr><td> $r = 1 6$ </td><td> ${ \bf 7 9 . 7 \pm 1 . 5 }$   $8 3 . 1 { \pm } 1 . 1 $ </td></tr><tr><td> $r = 3 2$ </td><td> $7 4 . 9 { \pm } 2 . 1 $   $8 1 . 3 { \pm } 1 . 6 $ </td><td rowspan="2"> $6 3 . 4 \pm 1 . 2$   $5 6 . 7 { \pm } 1 . 7 $ </td></tr><tr><td> $r = 6 4$ </td><td> $7 0 . 6 { \pm } 0 . 9 $   $6 9 . 8 { \pm } 1 . 4 $ </td></tr><tr><td>Core coordinates</td><td> $( f i x e d s p a n )$ </td><td></td></tr><tr><td>w/ thin QR</td><td> ${ \bf 7 9 . 7 \pm 1 . 5 }$   ${ \bf 8 3 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 6 9 . 3 { \pm 1 . 7 } }$ </td></tr><tr><td>w/o QR†</td><td>div. (0.0)  $2 2 . 9 { \pm } 1 . 9 $ </td><td>div. (0.0)</td></tr><tr><td></td><td>Whitening estimator  $( I 2 8 – 2 5 6$ </td><td>calibration)</td></tr><tr><td>Diagonal</td><td> ${ \bf 7 9 . 7 \pm 1 . 5 }$   ${ \bf 8 3 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 6 9 . 3 { \pm 1 . 7 } }$ </td></tr><tr><td>Full K-FAC</td><td> $7 1 . 0 { \pm } 2 . 6 $   $8 2 . 7 { \pm } 1 . 6 $ </td><td> $6 7 . 6 { \pm } 2 . 2 $ </td></tr></table>

Table 6: Design diagnostics on Qwen2.5-3B. <sup>†</sup>Without QR, SVAMP and GSM8K diverged under the shared learning rate (loss → ∞); 0.0 is the resulting exactmatch score, not a converged accuracy. GSM8K uses the dedicated ablation configuration (QR reference 69.3).

## 5.6 Span selection and coordinate conditioning are distinct

The span-size block of Table 6 shows a consistent small-rank regime and a clear breakdown at larger ranks. Performance peaks at $r = 1 6$ on SVAMP and GSM8K, while ARC-C is already strongest at $r = 8 ;$ all three tasks drop substantially at $r = 3 2$ and again at $r = 6 4$ . This pattern is not an expressivity paradox: for a fixed calibration statistic, the top-r SVD spans are nested, so a larger span can reproduce the lower-rank solution. What changes is the finite-sample estimation and optimization problem. Ranks above 16 admit weaker tail directions estimated from the same $1 2 8 \mathrm { - } 2 5 6$ calibration examples, while the core grows from $1 6 ^ { 2 } = 2 5 6$ to $3 2 ^ { 2 } = 1 0 2 4$ and $6 4 ^ { 2 }$ = 4096 coefficients under essentially fixed training data, steps, and learning-rate search. QR removes gross anisotropy, but it does not denoise tail directions or regularize the larger core. The evidence therefore supports a small-tomoderate rank sweet spot under this protocol, with $r = 8 – 1 6$ covering the best tested points across tasks.

<table><tr><td>Configuration</td><td>FCCA</td><td>FiLoRA-c</td><td>RawGrad</td></tr><tr><td> $r = 8 , q / v$ </td><td>74.7±2.1</td><td>75.4±0.8</td><td> $7 5 . 1 \pm 0 . 2 $ </td></tr><tr><td> $r = 8 , q / k / v / o$ </td><td>75.2±1.3</td><td>76.8±1.1</td><td>74.2±0.4</td></tr><tr><td> $r = 1 6 , \dot { q } / \dot { v }$ </td><td>76.0±2.8</td><td>77.3±1.2</td><td>76.9±0.3</td></tr><tr><td> $r = 1 6 , \stackrel { \sim } { q } / k / v / o$ </td><td>79.7±1.5</td><td>80.1±0.4</td><td> $7 4 . 9 2 2 . 0 $ </td></tr></table>

Table 7: SVAMP rank-by-target-module interaction. FCCA’s separation from RawGrad appears at the main $r = 1 6 , q / k / v / o$ budget; FiLoRA-core remains slightly higher in that cell.

Table 7 sharpens this boundary. FCCA is near RawGrad in the three smaller configurations and opens a 4.8-point gap only at r = 16, $q / k / v / o ;$ FiLoRA-core remains 0.4 point higher there. Fisher scaling therefore does not win every small-budget cell. Its clearest benefit appears when rank and module coverage are sufficient to retain the balanced directions selected by the whitened crosscovariance.

Taken together, the two budget studies favor distributing a moderate rank across all four attention projections. Broader module coverage exposes more types of task-relevant transformations, whereas increasing r beyond 16 mainly adds weaker tail directions under the same calibration and optimization budget. In these campaigns, breadth across q/k/v/o is more useful than indiscriminately expanding each frozen span.

The coordinate block isolates optimization geometry. QR preserves the update family, yet without it the shared-learning-rate runs diverge on SVAMP and GSM8K and reach 22.9 on ARC-C. Thus the 0.0 entries diagnose failed optimization in the raw coordinates, not zero span expressivity. The estimator block isolates finite-sample curvature estimation: dense K-FAC must infer two width-by-width covariances from only 128–256 examples and trails the diagonal estimator on all three tasks. This supports diagonal whitening in the tested low-calibration regime, while leaving open whether stronger shrinkage, structured factors, or substantially more calibration could reverse the result.

A useful frozen adapter requires both a wellchosen span and a trainable coordinate system. The controls separate three failure sources that endtask accuracy alone would conflate: the statistic can omit task information, the rank metric can discard locally useful directions, or the coordinates can make an adequate span untrainable.

The controls differ in inferential strength. Whitening-side and QR ablations hold the pipeline fixed, and QR preserves the update family; dense K-FAC and rank sweeps also change finite-sample conditioning or model size. We therefore interpret the latter as regime-specific design evidence, not monotonic laws.

## 6 Conclusion and Discussion

FCCA combines signed task signal, Fisher-scaled truncation, and QR-conditioned core coordinates. It leads at all Qwen scales, nearly ties on Llama-1B, and approaches LoRA and DoRA with roughly 200× less trainable state while keeping calibration and basis-storage costs explicit.

The baseline pattern separates three design choices. Weight-spectrum and random bases lack downstream error information; EVA-core and FiLoRA-core retain useful marginals but omit the signed input–error pairing; RawGrad preserves that pairing but imposes rank in an unscaled metric; and no-QR keeps the selected span yet makes its core effectively untrainable. Trainable factors retain one advantage: they can rotate to repair an incomplete span, which explains the remaining gap to LoRA and DoRA.

Frozen-core accuracy is therefore a subspace diagnostic. Under resource constraints, the practical lesson is to preserve signed task signal in the metric where rank is imposed, allocate moderate rank before admitting weak tail directions, and expose the retained span through trainable coordinates.

## Limitations

We evaluate fixed subsets (300 examples per math task and 500 per remaining task), so absolute scores are not directly comparable with full-test leaderboards. The heterogeneous 11-task macro-average weights tasks equally, and the paired Wilcoxon test has only 11 observations.

The study uses three seeds, one software stack, and one GPU family. Three Qwen scales are complemented by one Llama setting, and several ablations cover only one to three tasks; their design conclusions may be regime specific. We do not evaluate instruction following or open-ended generation beyond exact-match math.

Calibration depends on task match and requires a backward pass plus cross-covariances that can consume host memory. The r<sup>2</sup> budget measures trainable and optimizer state, not complete storage. Search trial counts also differ (four FCCA warmup/calibration configurations versus two baseline learning rates), and benchmark contamination was not audited.

## Acknowledgments

Supported by the Noncommunicable Chronic Diseases-National Science and Technology Major Project (2026ZD0558202) and the Ningbo Yongjiang Talent Introduction Programme (2023A-399-G).

## References

Shun-ichi Amari. 1998. Natural gradient works efficiently in learning. Neural Computation, 10(2):251– 276.

Klaudia Bałazy, Mohammadreza Banaei, Karl Aberer, and Jacek Tabor. 2025. LoRA-XS: Low-rank adaptation with extremely small number of parameters. In ECAI 2025 – 28th European Conference on Artificial Intelligence, Frontiers in Artificial Intelligence and Applications, pages 3194–3201. IOS Press.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. 2022. BitFit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 1–9. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Carl Eckart and Gale Young. 1936. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218.

Dezheng Han and Shuaishuai Guo. 2026. FiLoRA: Parameter-efficient fine-tuning with fisher information-guided low-rank adaptation. IEEE Signal Processing Letters, 33:604–608.

Yongchang Hao, Yanshuai Cao, and Lili Mou. 2024. Flora: Low-rank adapters are secretly gradient compressors. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17554–17571. PMLR.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Dawid J. Kopiczko, Tijmen Blankevoort, and Yuki M. Asano. 2024. VeRA: Vector-based random matrix

adaptation. In International Conference on Learning Representations.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059. Association for Computational Linguistics.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597. Association for Computational Linguistics.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems, volume 35, pages 1950–1965.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. DoRA: Weightdecomposed low-rank adaptation. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32100–32121. PMLR.

Llama Team, AI @ Meta. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

James Martens and Roger Grosse. 2015. Optimizing neural networks with kronecker-factored approximate curvature. In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pages 2408–2417. PMLR.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. 2024. PiSSA: Principal singular values and singular vectors adaptation of large language models. In Advances in Neural Information Processing Systems, volume 37.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391. Association for Computational Linguistics.

Leon Mirsky. 1960. Symmetric gauge functions and unitarily invariant norms. The Quarterly Journal of Mathematics, 11(1):50–59.

Fabian Paischer, Lukas Hauzenberger, Thomas Schmied, Benedikt Alkin, Marc Deisenroth, and Sepp Hochreiter. 2025. Parameter efficient finetuning via explained variance adaptation. In Advances in Neural Information Processing Systems, volume 38.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2080–2094. Association for Computational Linguistics.

Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networksfor NLP, pages 353–355. Association for Computational Linguistics.

Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. 2025. MiLoRA: Harnessing minor singular components for parameter-efficient LLM finetuning. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4823–4836. Association for Computational Linguistics.

Shaowen Wang, Linxi Yu, and Jian Li. 2024. LoRA-GA: Low-rank adaptation with gradient approximation. In Advances in Neural Information Processing Systems, volume 37, pages 54905–54931.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800. Association for Computational Linguistics.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In International Conference on Learning Representations.

## A Mathematical Details and Coordinate Invariance

## A.1 Projected-gradient criterion for a frozen core

Let $\phi ( R ) = \mathcal { L } ( W + s P R Q ^ { \top } )$ for fixed P and $Q .$ and let $G = \nabla _ { W } \mathcal { L } ( W )$ . The first-order expansion around the zero core is

$$
\phi ( R ) = \phi ( 0 ) + s \langle G , P R Q ^ { \top } \rangle + o ( \| R \| _ { F } )\tag{3}
$$

$$
\begin{array} { r } { = \phi ( 0 ) + s \langle P ^ { \top } G Q , R \rangle + o ( \| R \| _ { F } ) . } \end{array}\tag{4}
$$

Hence $\nabla _ { R } \phi ( 0 ) ~ = ~ s P ^ { \top } G Q$ . A gradient step $R _ { 1 } = - \gamma s P ^ { \top } G Q$ changes the linearized objective $\mathbf { b y } - \gamma s ^ { 2 } \| P ^ { \top } G Q \| _ { F } ^ { 2 }$ . The fixed scale s changes the magnitude of the step but not the ranking of candidate subspaces. This derivation motivates the matched-budget comparison: a useful constructor must capture the signed task cross term, not only high-variance or high-weight-energy directions.

## A.2 Proof of Theorem 1

Set $Z ~ = ~ D ^ { 1 / 2 } \Delta W A ^ { 1 / 2 }$ . Positive definiteness makes the change of variables invertible, gives $\Delta W = D ^ { - 1 / 2 } \bar { Z } A ^ { - 1 / 2 }$ , and preserves rank. Because the inverse square roots are symmetric,

$$
\langle G , \Delta W \rangle = \mathrm { T r } \Big ( G ^ { \top } D ^ { - 1 / 2 } Z A ^ { - 1 / 2 } \Big )\tag{5}
$$

$$
= \operatorname { T r } \left( ( D ^ { - 1 / 2 } G A ^ { - 1 / 2 } ) ^ { \top } Z \right)\tag{6}
$$

$$
= \langle { \widetilde { G } } , Z \rangle .\tag{7}
$$

Substituting into Eq. (1) and completing the square yields

$$
\operatorname* { m i n } _ { \mathrm {  ~ \ r a n k } ( Z ) \leq r } \frac { 1 } { 2 \eta } \| Z + \eta \widetilde { G } \| _ { F } ^ { 2 } - \frac { \eta } { 2 } \| \widetilde { G } \| _ { F } ^ { 2 } .\tag{8}
$$

The second term is constant in Z. By the Eckart– Young–Mirsky theorem, the best rank-r approximation to $- \eta \widetilde { G }$ in Frobenius norm is $- \eta [ \widetilde { G } ] _ { r }$ . Mapping this solution back through the inverse square roots gives Eq. (2).

Why truncation is performed after whitening. Except when A and D are scalar multiples of the identity (or share special symmetries with G), the leading singular vectors of $D ^ { - 1 / 2 } G A ^ { - 1 / 2 }$ are not obtained by simply rescaling the leading singular vectors of G. Whitening can change both singular directions and their ordering. FCCA therefore applies the rank constraint in Fisher coordinates before any raw-gradient directions are discarded.

## A.3 Thin QR leaves the update family unchanged

Let $P _ { 0 } = P T _ { P }$ and $Q _ { 0 } = Q T _ { Q }$ be thin QR decompositions. If the selected columns have full rank, the triangular factors $T _ { P } , T _ { Q } \in \mathbb { R } ^ { r \times r }$ are nonsingular and

$$
\{ P _ { 0 } Z Q _ { 0 } ^ { \top } : Z \in \mathbb { R } ^ { r \times r } \} = \{ P R Q ^ { \top } : R \in \mathbb { R } ^ { r \times r } \} ,\tag{9}
$$

with the bijection $R = T _ { P } Z T _ { Q } ^ { \top }$ . QR therefore changes optimization coordinates, not expressivity. Zero initialization is function preserving in either coordinate system, since $Z ~ = ~ 0$ if and only if $R = 0$

## A.4 Diagonal moments, damping, and terminology

For each target matrix, calibration accumulates the uncentered cross moment G and coordinatewise second moments $a = \mathbb { E } [ x ^ { \odot 2 } ]$ and $d = \mathbb { E } [ \delta ^ { \odot 2 } ]$ . The default factors are $A = \mathrm { d i a g } ( a ) + \epsilon I$ and $D =$ $\mathrm { d i a g } ( d ) + \epsilon I$ , where damping keeps inverse square roots finite in rarely activated coordinates. The term “cross-covariance” follows common gradientfactor terminology; no mean subtraction is applied because $\mathbb { E } [ \delta x ^ { \top } ]$ ] is the empirical loss gradient of the linear map. Dense A and D are evaluated only in the dedicated full K-FAC ablation.

## B Complete Experimental Protocol

## B.1 Datasets, fixed subsets, and training length

Table 8 gives the complete data protocol, including source identifiers, fixed evaluation subsets, and training lengths.

The campaign uses a common evaluation budget instead of each benchmark’s full official test set: 300 examples for each math task and 500 for every other task. This keeps the four-model, eightmethod, three-seed comparison tractable and ensures that every method is evaluated on identical examples. It also means the absolute scores should be interpreted within this protocol rather than compared directly with full-test leaderboard numbers.

## B.2 Scoring protocols and aggregate interpretation

For SVAMP and GSM8K, decoding is greedy. The parser takes the first occurrence of either “The answer is:” or “####” and then applies one deterministic numeric normalizer shared by all methods.

Taking the first marker prevents a model that repeats or revises an answer later in the generation from being scored under a different extraction rule.

For the remaining nine tasks, each candidate answer is scored by its conditional log likelihood under the same prompt and the highest-scoring choice is selected. These results do not depend on generation temperature, stopping rules, or label-token parsing. CoLA and the other GLUE tasks are therefore reported as accuracy-like choice scores in this campaign, not necessarily with each benchmark’s official leaderboard metric.

Why decoding and likelihood are combined. The two scoring modes follow task format, not method choice. Mathematical reasoning requires a generated numeric answer, whereas conditional likelihood avoids label-token and stopping-rule artifacts for classification and multiple choice. The 11-task mean is consequently an unweighted macroaverage of heterogeneous accuracy-like scores on a common 0–100 scale. We use it as a descriptive summary, not a calibrated utility measure, and accompany it with per-task results, family means, and rank-based paired tests.

## B.3 Common adapter and hyperparameter protocol

Table 9 summarizes the unified optimization protocol.

FCCA evaluates four warmup-by-calibration combinations at one learning rate. Each matchedbudget baseline evaluates two learning rates at fixed warmup. This gives baselines an LR-selection advantage but does not equalize total search trials: FCCA has four configurations and a baseline has two. The best configuration is selected by held-out validation before the three-seed evaluation. LoRA, DoRA, and PiSSA are separate trainable-basis references and use validation-selected learning rates appropriate to their parameterization.

## B.4 Calibration implementation

Calibration examples are drawn only from the task’s training split. With calib\_group\_size=999, one grouped hooked forward/backward pass collects statistics for all target layers. Cross moments and diagonal second moments are placed on CPU (stats\_device=cpu); after the SVD and QR steps, the statistics are discarded and only the bases and cores remain. Diagonal whitening is the default. Full dense

<table><tr><td>Task</td><td>Dataset source</td><td>Configuration</td><td>Evaluation</td><td>Train</td><td>Eval / epochs</td></tr><tr><td>SVAMP</td><td>ChilleD/SVAMP</td><td>default</td><td>greedy exact match</td><td>700</td><td>300 / 3</td></tr><tr><td>GSM8K</td><td>openai/gsm8k</td><td>main</td><td>greedy exact match</td><td>1500</td><td>300/2</td></tr><tr><td>SST-2</td><td>nyu-ml1/glue</td><td>sst2</td><td>likelihood MC</td><td>2000</td><td>500/ 1</td></tr><tr><td>QNLI</td><td>nyu-ml1/glue</td><td>qnli</td><td>likelihood MC</td><td>2000</td><td>500/ 1</td></tr><tr><td>CoLA</td><td>nyu-ml1/glue</td><td>cola</td><td>likelihood MC</td><td>2000</td><td>500 / 2</td></tr><tr><td>RTE</td><td>nyu-ml1/glue</td><td>rte</td><td>likelihood MC</td><td>2000</td><td>500/ 3</td></tr><tr><td>MRPC</td><td>nyu-mll/glue</td><td>mrpc</td><td>likelihood MC</td><td>2000</td><td>500/3</td></tr><tr><td>ARC-Challenge</td><td>allenai/ai2_arc</td><td>ARC-Challenge</td><td>likelihood MC</td><td>1119</td><td>500/ 3</td></tr><tr><td>OpenBookQA</td><td>allenai/openbookqa</td><td>main</td><td>likelihood MC</td><td>2000</td><td>500/ 3</td></tr><tr><td>HellaSwag</td><td>Rowan/hellaswag</td><td>default</td><td>likelihood MC</td><td>2000</td><td>500/2</td></tr><tr><td>WinoGrande</td><td>allenai/winogrande</td><td>winogrande_xl</td><td>likelihood MC</td><td>2000</td><td>500/2</td></tr></table>

Table 8: Complete dataset protocol. “Default” denotes a dataset source with no named Hugging Face configuration. “Eval / epochs” gives the fixed evaluation-subset size and training epochs. All datasets were loaded from an offline cache, and the same fixed subsets were reused for every method and seed.

<table><tr><td>Component</td><td>FCCA</td><td>Matched-budget baselines</td></tr><tr><td>Shared adapter</td><td>rank 16; all  $q / k / v / o$  projections; every layer; zero r × r core</td><td>same</td></tr><tr><td>Precision / batch</td><td>bf16; batch size 1; gradient accu- mulation 8 (effective batch 8)</td><td>same</td></tr><tr><td>Trainable budget</td><td>one r × r core per target matrix; 36,864 parameters at Qwen2.5-3B</td><td>same</td></tr><tr><td>Learning rate</td><td>fixed  $5 \times 1 0 ^ { - 3 }$ </td><td>validation search over  $\{ 2 , 5 \} \ \times$   $1 0 ^ { - 3 }$ </td></tr><tr><td>Warmup</td><td>validation search over {0.03, 0.1}</td><td>fixed 0.1</td></tr><tr><td>Calibration size</td><td>validation search over {128, 256}</td><td>method-specific basis construction under the common data budget</td></tr><tr><td>Final evaluation</td><td>seeds 42, 43, 44; mean ± sample standard deviation</td><td>same</td></tr></table>

Table 9: Unified optimization protocol. FCCA has a single fixed learning rate and searches its own warmup/calibration knobs; matched-budget baselines can select learning rate.

K-FAC factors are constructed only for the controlled comparison in Table 6.

## B.5 Statistical reporting

Every displayed cell is mean ± sample standard deviation over seeds 42, 43, and 44 unless explicitly marked as a mean-only mechanism table. The primary cross-task test is a two-sided paired Wilcoxon signed-rank test over the 11 unrounded task means. A paired t-test is shown as a reference, but the rank test is primary because the 11 task differences include large outliers and provide little basis for a normality assumption. The task, not the seed, is the unit of replication for this analysis.

## C Baseline Constructions

All matched-budget baselines are converted to the same frozen-basis, trainable-core architecture. Consequently, labels ending in “-core” denote controlled instantiations of a basis-selection idea, not a reproduction of the original method’s complete parameterization or native trainable budget.

RawGrad. Compute $G = U \Sigma V ^ { \top }$ on the calibration set and take $P = U _ { r } , Q = V _ { r }$ . No input- or output-side whitening is applied. RawGrad therefore has the same signed task cross moment as

FCCA and is the cleanest control for the effect of A and D.

FiLoRA-core. Select fixed input and output bases from the marginal K-FAC/Fisher factors A and D. This retains curvature-aware marginal eigenspaces but does not preserve the signed input– error pairing in G.

MiLoRA-core and LoRA-XS-core. Both use the pretrained weight SVD and are task agnostic during basis construction. MiLoRA-core selects the bottom-r weight singular directions. LoRA-XS-core uses the top-r directions and incorporates the singular values into the left basis before the common coordinate handling and core training.

EVA-core. The input-side basis is obtained from the leading singular directions of calibration activations, following the explained-variance principle. The campaign uses the corresponding fixed outputside construction from its EVA-core implementation and trains only the common r × r core.

VeRA-core and Random-core. VeRA-core preserves the frozen-random-basis principle but replaces VeRA’s native lightweight scaling parameterization with the common square core. Randomcore samples independent orthonormal left and right bases. Their purpose is to separate the value of task-derived structure from the capacity of the shared core itself.

Trainable-basis references. LoRA, DoRA, and PiSSA update rank-r factors of width-dependent size, approximately $r ( m + n )$ values per target matrix. PiSSA initializes those trainable factors from principal weight components; unlike LoRA-XScore, its factors remain trainable. These references answer how much accuracy is lost by freezing a carefully selected subspace, not which method wins under equal trainable parameters.

## D Full Cross-Scale Results

Figure 2 emphasizes the ranking structure; Table 10 gives the exact 11-task means. Tables 11, 12, and 13 report every task-level mean and standard deviation for the three non-primary settings, while the Qwen2.5-3B campaign remains in Table 2.

On Qwen2.5-1.5B, FCCA has the highest displayed 11-task mean (77.15 before rounding), 0.47 point above EVA-core and 1.90 above RawGrad. On Qwen2.5-7B, FCCA reaches 87.49, 0.79 above EVA-core; the supplied unrounded Wilcoxon comparison to EVA-core is $p = . 0 5 4$ , so the two are statistically comparable at that scale. On Llama-3.2-1B, EVA-core leads by 0.13 point before rounding, while FCCA exceeds RawGrad on eight of 11 tasks. These results support a repeated Qwen scaling pattern and a top-tier cross-architecture result, rather than a claim that one constructor must win on every model family.

All averages in Table 10 are recomputed from the 11 displayed task means. This avoids silently changing the task set across models and keeps the comparison aligned with the primary 3B campaign. The main-text figure highlights FCCA and Raw-Grad because they isolate the effect of Fisher scaling while sharing the same signed cross-moment; the complete tables retain all six additional constructors.

## E Trainable-Basis References

LoRA, DoRA, and PiSSA train both rank-16 factors rather than a square core. On Qwen2.5-3B, FCCA is 0.318 point below LoRA and 0.227 below DoRA on the 11-task mean despite training about 200× fewer parameters. FCCA is higher than LoRA on SVAMP, GSM8K, SST-2, and QNLI, and higher than DoRA on those four tasks plus

WinoGrande. The largest trainable-basis advantages are on OpenBookQA and ARC-C. The comparison therefore supports near-parity on average, not dominance over full trainable factors.

An earlier Qwen2.5-7B PiSSA GSM8K run at learning rate $1 0 ^ { - 3 }$ produced 0.5% exact match, validation loss 5.48, and degenerate numeric strings. This was an optimizer divergence caused by an illmatched learning rate, not evidence that PiSSA’s subspace is intrinsically defective. Re-running at $2 \times 1 0 ^ { - 4 }$ gives 77.2±2.1. The Qwen2.5-3B PiSSA value in Table 14 is a separate result: both candidate learning rates train stably, but GSM8K remains 51.4±1.7.

## F Task-Level Significance

Table 15 expands the compact main-text significance summary with win counts and paired-t references.

All reported tests are unadjusted. A Bonferroni threshold for seven comparisons would be $. 0 5 / 7 \approx . 0 0 7 1$ , but several supplied values are available only after rounding or as bounds such as $p < . 0 1$ , so the corrected decisions cannot be audited exactly from the summary record. We therefore report the pre-specified unadjusted comparisons transparently and do not make a multiplicitycorrected significance claim.

The paired-t reference for EVA-core is .090 even though the Wilcoxon value is .007. Two especially large positive FCCA margins (HellaSwag and MRPC) inflate the variance of the raw differences; the rank-based test is less sensitive to their magnitude. With only 11 task pairs, both the resolution and power of any cross-task test remain limited.

## G Ablations and Robustness

## G.1 Whitening sides

Figure 3(a) visualizes the gains, while Table 16 preserves the underlying six-task means. Table 17 is the strict exact-pipeline control in which learning rate, warmup, calibration size, and all other choices are fixed and only the whitening gates change.

Every whitened variant exceeds the nowhitening SVD on every populated task, and full whitening gains 2.7–17.2 points over None. Full whitening is best or tied on four of six tasks; inputonly is 0.1 higher on ARC-C and 1.1 higher on RTE. The exact SVAMP control shows the same

<table><tr><td>Model</td><td>FCCA</td><td>RawGrad</td><td> $_ \mathrm { F i L o R A - c }$ </td><td> $\mathbf { M i L o R A - c }$ </td><td>LoRA-XS-c</td><td>VeRA-c</td><td> ${ \mathrm { E V A - c } }$ </td><td>Random</td></tr><tr><td>Qwen2.5-1.5B</td><td>77.2</td><td>75.3</td><td>75.2</td><td>74.9</td><td>74.8</td><td>74.8</td><td>76.7</td><td>73.8</td></tr><tr><td>Qwen2.5-3B</td><td>83.0</td><td>78.3</td><td>79.5</td><td>80.4</td><td>80.7</td><td>80.3</td><td>79.9</td><td>79.7</td></tr><tr><td>Qwen2.5-7B</td><td>87.5</td><td>86.0</td><td>85.5</td><td>84.3</td><td>84.9</td><td>85.7</td><td>86.7</td><td>82.9</td></tr><tr><td>Llama-3.2-1B</td><td>68.1</td><td>66.3</td><td>67.5</td><td>64.7</td><td>66.6</td><td>66.4</td><td>68.2</td><td>63.1</td></tr></table>

Table 10: Cross-scale 11-task means under the same protocol. Values are recomputed from the displayed per-task means; bold/underline denote first/second within each model.

<table><tr><td>Task</td><td>FCCA</td><td>RawGrad</td><td> $_ { \mathrm { F i L o R A - c } }$ </td><td> $\mathbf { M i L o R A - c }$ </td><td> $_ { \mathrm { L o R A - X S - C } }$ </td><td> $\mathrm { V e R A - c }$ </td><td> $\mathrm { E V A – c }$ </td><td>Random</td></tr><tr><td>SVAMP</td><td> $6 8 . 4 \pm 1 . 0$ </td><td> $6 9 . 1 \pm 2 . 0$ </td><td> $6 7 . 2 { \pm } 0 . 6 $ </td><td> $6 2 . 7 { \pm } 1 . 0 $ </td><td> $5 9 . 1 { \pm } 1 . 1 $ </td><td> $6 4 . 3 { \pm } 3 . 4 $ </td><td> ${ \bf 6 9 . 2 \pm 0 . 8 }$ </td><td> $5 9 . 4 \pm 1 . 5$ </td></tr><tr><td>GSM8K</td><td> $4 7 . 3 { \pm } 0 . 7 $ </td><td> $4 5 . 7 \pm 1 . 2$ </td><td> ${ \bf 4 9 . 1 \pm 1 . 8 }$ </td><td> $4 6 . 8 \pm 1 . 1$ </td><td> $4 6 . 4 \pm 0 . 8$ </td><td> $4 8 . 1 \pm 0 . 9$ </td><td> $4 8 . 4 \pm 1 . 5$ </td><td> $4 6 . 4 \pm 0 . 9$ </td></tr><tr><td>SST-2</td><td> ${ \bf 9 5 . 8 \pm 0 . 3 }$ </td><td> $9 3 . 7 { \pm } 0 . 8 $ </td><td> $9 3 . 7 { \pm } 0 . 7 $ </td><td> $9 4 . 8 { \pm } 0 . 2 \ $ </td><td> $\begin{array} { l } { { \mp \mathrm { { u . + . \mathrm { { i . u . o } } } } } } \\ { { \ q _ { 4 } \ q _ { + 0 . 3 } } } \end{array}$ </td><td> $9 4 . 4 \pm 0 . 2 $ </td><td> $9 5 . 0 { \pm } 0 . 7 \ $ </td><td>94.4±0.4</td></tr><tr><td>QNLI</td><td> ${ \bf 8 7 . 5 \pm 0 . 5 }$ </td><td> $8 7 . 1 { \pm } 0 . 4 $ </td><td> $8 5 . 7 \pm 0 . 7$ </td><td> $8 5 . 8 { \pm } 0 . 9$ </td><td> $8 5 . 7 { \pm } 0 . 6 $ </td><td> $8 4 . 9 { \pm } 0 . 5 $ </td><td> $8 6 . 4 \pm 0 . 9$ </td><td> $8 4 . 9 2 0 . 7 $ </td></tr><tr><td>CoLA</td><td> $8 2 . 2 \pm 1 . 1$ </td><td> $\mathbf { 8 2 . 6 } \pm \mathbf { 1 . 3 }$ </td><td> $8 0 . 9 { \pm } 2 . 6 $ </td><td> $8 0 . 2 \pm 0 . 7$ </td><td> $8 0 . 1 \pm 1 . 2 $ </td><td> $7 5 . 7 \pm 1 . 5$ </td><td> $8 2 . 5 { \pm } 2 . 3 $ </td><td> $7 4 . 5 { \pm } 1 . 1 $ </td></tr><tr><td>RTE</td><td> ${ \bf 8 5 . 9 2 1 . 6 }$ </td><td> $8 4 . 1 \pm 1 . 2 $ </td><td> $8 1 . 3 { \pm } 0 . 7 \ $ </td><td> $8 2 . 4 \pm 0 . 7$ </td><td> $8 4 . 0 { \pm } 1 . 1 $ </td><td> $8 2 . 2 { \pm } 0 . 2 $ </td><td> $8 5 . 0 { \pm } 0 . 9 \ $ </td><td> $8 2 . 6 { \pm } 0 . 9 \ $ </td></tr><tr><td>MRPC</td><td> ${ \bf 8 7 . 5 \pm 0 . 3 }$ </td><td> $8 5 . 6 { \pm } 0 . 6 $ </td><td> $8 5 . 0 { \pm } 0 . 9 \ $ </td><td> $8 4 . 2 { \pm } 0 . 9$ </td><td> $8 5 . 3 { \pm } 0 . 9 \ $ </td><td> $8 4 . 2 { \pm } 0 . 3 $ </td><td> $8 7 . 3 { \pm } 1 . 1 $ </td><td> $8 2 . 8 { \pm } 0 . 4 $ </td></tr><tr><td>ARC-C</td><td> ${ \bf 7 7 . 7 \pm 1 . 6 }$ </td><td> $7 4 . 4 \pm 1 . 2$ </td><td> $7 6 . 5 { \pm } 1 . 6 $ </td><td> $7 5 . 8 \pm 1 . 6$ </td><td> $7 6 . 6 \pm 1 . 3$ </td><td> $7 7 . 4 \pm 0 . 7$ </td><td> $7 6 . 0 { \pm } 1 . 1 $ </td><td>76.8±0.9</td></tr><tr><td>OBQA</td><td> ${ \bf 8 2 . 1 \pm 0 . 4 }$ </td><td> $7 8 . 9 { \pm } 1 . 3 $ </td><td> $7 9 . 8 \pm 2 . 0$ </td><td> $8 1 . 1 { \pm } 0 . 3 $ </td><td> $8 0 . 3 { \pm } 0 . 6 $ </td><td> $8 1 . 9 { \pm } 0 . 5 $ </td><td> $8 2 . 0 { \pm } 0 . 3 $ </td><td> $7 9 . 9 2 0 . 2 $ </td></tr><tr><td>HellaSwag</td><td> ${ \bf 6 8 . 1 \pm 0 . 5 }$ </td><td> $6 5 . 1 { \pm } 0 . 3 $ </td><td> $6 4 . 9 2 1 . 5$ </td><td> $6 5 . 9 2 0 . 6 $ </td><td> $6 5 . 8 { \pm } 0 . 8 $ </td><td> $6 5 . 7 \pm 1 . 2$ </td><td> $6 7 . 1 { \pm } 0 . 5 $ </td><td> $6 7 . 1 { \pm } 0 . 2 $ </td></tr><tr><td>WinoG</td><td> ${ \bf 6 6 . 2 \pm 0 . 3 }$ </td><td> $6 1 . 5 { \pm } 1 . 5 $ </td><td> $6 3 . 3 { \pm } 1 . 2 $ </td><td> $6 3 . 8 { \pm } 0 . 7$ </td><td> $6 4 . 9 { \pm } 1 . 1 $ </td><td> $6 4 . 1 \pm 1 . 4$ </td><td> $6 4 . 6 { \pm } 1 . 8 $ </td><td> $6 2 . 5 { \pm } 0 . 8 $ </td></tr><tr><td>Mean</td><td>77.2</td><td>75.3</td><td>75.2</td><td>74.9</td><td>74.8</td><td>74.8</td><td>76.7</td><td>73.8</td></tr></table>

Table 11: Full matched-budget results on Qwen2.5-1.5B-Instruct. Mean±std over three seeds (×100); the Mean row is recomputed from the 11 displayed task means.

## G.4 Rank and adapted modules

ordering under fixed non-whitening hyperparameters, strengthening the causal interpretation. The evidence supports geometry-aware normalization and two-sided whitening as a robust default, not a guarantee that adding the second side must improve finite-sample validation accuracy.

## G.2 Calibration size and basis stability

Tables 18 and 19 separate downstream accuracy from convergence of the selected subspaces.

As Tables 18 and 19 show, accuracy is nearly flat from 64 to 512 examples even though the selected subspaces continue to converge. On SVAMP, FCCA spans 78.8–80.1 and remains 5.9–7.3 points above RawGrad at every calibration size; ARC-C spans only 0.4 point. The principal-angle cosines exceed .93 on both sides at 128 examples. These results show that the leading useful directions are identifiable with modest calibration data in the tested tasks, while not implying that all layers or domains will have the same sample requirement.

Main-text Table 7 separates the SVAMP rankby-module grid from the dedicated all-projection sweep in Table 6. The full sweep favors $r = 1 6$ on SVAMP and GSM8K and $r = 8$ on ARC-C; all three tasks deteriorate at $r = 3 2$ and $r = 6 4$ . The sweep cannot isolate estimation noise, optimization difficulty, and overcapacity. Because the two tables come from separate controlled campaigns, their $r = 8 , q / k / v / o$ cells are not duplicate measurements.

## G.3 Mismatched calibration

## G.5 QR coordinates and dense K-FAC

Main-text Table 6 reports the full three-seed QR and whitening-estimator controls. Without QR, raw inverse-whitened bases differ in scale by orders of magnitude: SVAMP and GSM8K diverge to infinite loss and zero exact match, while ARC-C falls to 22.9. QR adds no expressive directions, but it makes a common core optimizer usable in these runs.

Table 20 shows that same-domain GSM8K calibration loses only 1.7 points, whereas science QA and sentiment calibration lose 5.1 and 7.1. The most distant mismatch (72.6) is 0.8 point below the nowhitening SVAMP control (73.4), so mismatched calibration degrades gracefully here without guaranteeing an advantage over RawGrad. The ordering supports the task-aware interpretation: representative calibration data are what make the whitened cross moment especially useful.

Retuned no-QR control. We reran the no-QR variant at learning rates $\{ 5 \times 1 0 ^ { - 3 } , 2 \times$ $1 0 ^ { - 3 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } \}$ , holding all other settings fixed. Table 21 reports the best observed score over the grid. SVAMP and GSM8K remain at 0.0 for every rate because the loss diverges or becomes NaN and generation produces no extractable answer. ARC-C peaks at 22.7, close to four-choice chance (25%) and effectively unchanged from the 22.9 shared-learning-rate control.

Reducing the learning rate by an order of mag-

<table><tr><td>Task</td><td>FCCA</td><td>RawGrad</td><td> $_ { \mathrm { F i L o R A - c } }$ </td><td> $\mathbf { M i L o R A - c }$ </td><td>LoRA-XS-c</td><td>VeRA-c</td><td> $\mathrm { E V A – c }$ </td><td>Random</td></tr><tr><td>SVAMP</td><td>84.4±0.4</td><td> $8 3 . 8 { \pm } 0 . 3 $ </td><td> $8 1 . 6 { \pm } 0 . 6 $ </td><td> $7 6 . 6 { \pm } 0 . 2 $ </td><td> $7 8 . 7 \pm 1 . 7$ </td><td> $8 0 . 1 \pm 2 . 0$ </td><td> $8 1 . 7 { \pm } 0 . 5 $ </td><td> $7 5 . 7 { \pm } 1 . 1 $ </td></tr><tr><td>GSM8K</td><td> $7 8 . 1 \pm 0 . 6$ </td><td> $7 7 . 2 { \pm } 0 . 4 $ </td><td> $7 5 . 9 2 0 . 6 $ </td><td> $7 3 . 0 { \pm } 1 . 1 $ </td><td> $7 2 . 2 \pm 1 . 7$ </td><td> $\mathbf { 7 8 . 2 \pm 1 . 0 }$ </td><td> $7 7 . 3 { \pm } 0 . 5 $ </td><td> $7 2 . 0 { \pm } 0 . 7 $ </td></tr><tr><td>SST-2</td><td> ${ \bf 9 6 . 4 \pm 0 . 0 }$ </td><td> $9 5 . 9 2 0 . 2 $ </td><td>94.9±0.8</td><td>95.1±0.2</td><td> $9 5 . 1 \pm 0 . 2 $ </td><td> $9 4 . 9 { \pm } 0 . 1 $ </td><td> $9 6 . 2 \pm 0 . 2 $ </td><td> $9 4 . 8 { \pm } 0 . 3 $ </td></tr><tr><td>QNLI</td><td> $8 9 . 4 \pm 0 . 9$ </td><td> $8 7 . 6 { \pm } 0 . 2 $ </td><td> $8 9 . 2 { \pm } 0 . 6 $ </td><td> $8 8 . 1 \pm 0 . 7 $ </td><td> $8 8 . 4 \pm 0 . 3 $ </td><td> $8 8 . 3 { \pm } 0 . 7 \ $ </td><td> ${ \bf 9 0 . 2 \pm 0 . 7 }$ </td><td> $8 7 . 3 { \pm } 0 . 6 $ </td></tr><tr><td>CoLA</td><td> $8 6 . 8 { \pm } 0 . 3 $ </td><td> $8 6 . 8 { \pm } 1 . 6 $ </td><td> $8 5 . 9 { \pm } 1 . 0 \ $ </td><td> $8 5 . 9 { \pm } 0 . 2 \ $ </td><td> $8 6 . 1 \pm 0 . 2 $ </td><td> $8 5 . 3 { \pm } 0 . 4 $ </td><td> ${ \bf 8 7 . 4 \pm 1 . 0 }$ </td><td> $8 3 . 3 { \pm } 1 . 0 $ </td></tr><tr><td>RTE</td><td> ${ \bf 9 1 . 5 \pm 0 . 5 }$ </td><td> $9 0 . 1 { \pm } 0 . 6 $ </td><td> $9 0 . 1 { \pm } 0 . 7 \ $ </td><td> $9 0 . 3 { \pm } 0 . 9 \ \qquad $ </td><td> $9 0 . 6 { \pm } 0 . 8 $ </td><td> $9 1 . 0 { \pm } 0 . 8 $ </td><td> $9 0 . 6 { \pm } 1 . 3 $ </td><td> $8 9 . 2 { \pm } 0 . 3 $ </td></tr><tr><td>MRPC</td><td> $\mathbf { 8 8 . 5 \pm 0 . 4 }$ </td><td> $8 6 . 6 \pm 1 . 5$ </td><td> $8 7 . 7 { \pm } 0 . 3 $ </td><td> $8 4 . 7 { \pm } 0 . 6 $ </td><td> $8 7 . 4 \pm 0 . 5$ </td><td> $8 6 . 4 \pm 0 . 8$ </td><td> $8 8 . 1 \pm 0 . 4 $ </td><td> $8 3 . 3 { \pm } 0 . 9 \ \qquad $ </td></tr><tr><td>ARC-C</td><td> ${ \bf 9 1 . 2 \pm 0 . 4 }$ </td><td> $8 9 . 7 \pm 0 . 5$ </td><td> $8 8 . 5 { \pm } 0 . 7 \ $ </td><td> ${ \bf 9 1 . 2 \pm 0 . 3 }$ </td><td> $8 8 . 9 { \pm } 0 . 2 \ $ </td><td> $9 0 . 6 { \pm } 0 . 2 \ $ </td><td> $9 1 . 0 { \pm } 0 . 3 $ </td><td> $8 9 . 7 { \pm } 0 . 3 $ </td></tr><tr><td>OBQA</td><td> ${ \bf 9 2 . 5 } \pm { \bf 0 . 7 }$ </td><td> $8 9 . 8 \pm 0 . 7$ </td><td> $9 0 . 1 { \pm } 0 . 5 $ </td><td> $8 9 . 3 { \pm } 0 . 1 $ </td><td> $8 9 . 5 { \pm } 0 . 8 $ </td><td> $9 0 . 3 { \pm } 0 . 4 \ $ </td><td> $9 0 . 3 { \pm } 0 . 9 $ </td><td> $8 7 . 4 \pm 0 . 3 $ </td></tr><tr><td>HellaSwag</td><td>82.7±0.5</td><td> $8 0 . 1 \pm 0 . 1$ </td><td> $8 0 . 1 { \pm } 1 . 1$ </td><td> $7 9 . 1 \pm 0 . 9$ </td><td> $8 1 . 2 { \pm } 1 . 2 $ </td><td> $8 1 . 7 { \pm } 1 . 0 $ </td><td> $8 1 . 8 { \pm } 0 . 3$ </td><td> $7 9 . 4 \pm 0 . 6$ </td></tr><tr><td>WinoG</td><td>80.9±0.8</td><td> $7 8 . 1 \pm 1 . 2$ </td><td> $7 6 . 3 { \pm } 1 . 1 $ </td><td> $7 3 . 9 2 0 . 8 $ </td><td> $7 5 . 8 { \pm } 0 . 4 $ </td><td> $7 5 . 5 { \pm } 0 . 3 $ </td><td> $7 9 . 2 \pm 0 . 8$ </td><td> $6 9 . 3 { \pm } 1 . 1 $ </td></tr><tr><td>Mean</td><td>87.5</td><td>86.0</td><td>85.5</td><td>84.3</td><td>84.9</td><td>85.7</td><td>86.7</td><td>82.9</td></tr></table>

Table 12: Full matched-budget results on Qwen2.5-7B-Instruct. Mean±std over three seeds (×100); the Mean row is recomputed from the 11 displayed task means.

<table><tr><td>Task</td><td>FCCA</td><td>RawGrad</td><td> $_ { \mathrm { F i L o R A - c } }$ </td><td> $\mathbf { M i L o R A - c }$ </td><td> $_ { \mathrm { L o R A - X S - C } }$ </td><td> $\mathrm { V e R A - c }$ </td><td> $\mathrm { E V A – c }$ </td><td>Random</td></tr><tr><td>SVAMP</td><td> ${ \bf 6 4 . 9 2 1 . 5 }$ </td><td> $6 1 . 7 { \pm } 0 . 3 $ </td><td> ${ \bf 6 4 . 9 { \pm 1 . 0 } }$ </td><td> $5 5 . 9 { \pm } 0 . 3 $ </td><td> $5 8 . 3 { \pm } 1 . 2 $ </td><td></td><td>63.2±0.2 62.2±0.7</td><td> $5 5 . 3 { \pm } 2 . 2 $ </td></tr><tr><td>GSM8K</td><td> $3 8 . 6 \pm 1 . 7$ </td><td>38.8±0.3</td><td> $3 6 . 0 { \pm } 0 . 9 \ $ </td><td> $3 7 . 0 { \pm } 0 . 5 $ </td><td> $3 5 . 1 \pm 1 . 8$ </td><td> $3 7 . 7 \pm 2 . 2$ </td><td> $3 6 . 1 \pm 0 . 7$ </td><td> $3 4 . 7 { \pm } 1 . 4 $ </td></tr><tr><td>SST-2</td><td> $8 7 . 5 { \pm } 4 . 4 $ </td><td> $8 6 . 5 { \pm } 7 . 7 $ </td><td> $9 0 . 6 \pm 2 . 4 $ </td><td> $9 1 . 0 { \pm } 1 . 2 \ $ </td><td>92.3±0.3</td><td>92.5±0.5</td><td></td><td>93.3±1.4 91.0±0.7</td></tr><tr><td>QNLI</td><td> ${ \bf 8 5 . 1 \pm 1 . 2 }$ </td><td> $8 1 . 3 { \pm } 2 . 0 $ </td><td> $8 3 . 8 { \pm } 1 . 4 $ </td><td> $7 7 . 2 \pm 1 . 4$ </td><td> $8 1 . 4 \pm 2 . 5$ </td><td> $7 7 . 1 \pm 1 . 9$ </td><td> $8 4 . 3 { \pm } 1 . 6 $ </td><td> $7 5 . 1 { \pm } 0 . 8 $ </td></tr><tr><td>CoLA</td><td> $7 3 . 9 \pm 3 . 7$ </td><td> $7 4 . 5 { \pm } 3 . 2 $ </td><td> $7 2 . 1 \pm 1 . 5$ </td><td> $7 0 . 0 { \pm } 0 . 0 \ $ </td><td> $7 0 . 3 { \pm } 0 . 4 $ </td><td>70.0±0.0</td><td> $7 3 . 7 \pm 1 . 9$ </td><td>70.0±0.0</td></tr><tr><td>RTE</td><td> $7 8 . 7 { \pm } 1 . 4 $ </td><td> $7 5 . 1 \pm 1 . 5$ </td><td> $\mathbf { 7 9 . 4 \pm 0 . 0 }$ </td><td> $7 6 . 3 { \pm } 1 . 8 $ </td><td> $7 6 . 7 { \pm } 1 . 4 $ </td><td> $7 5 . 0 { \pm } 1 . 2 $ </td><td> $7 8 . 2 \pm 0 . 9$ </td><td> $7 0 . 5 { \pm } 0 . 9 $ </td></tr><tr><td>MRPC</td><td> $8 0 . 4 \pm 0 . 7$ </td><td> $7 9 . 5 { \pm } 3 . 6 $ </td><td> $8 0 . 3 { \pm } 0 . 6 $ </td><td> $7 4 . 6 { \pm } 0 . 8 $ </td><td> $7 9 . 1 \pm 1 . 1$ </td><td> $7 5 . 3 \pm 1 . 3$ </td><td> ${ \bf 8 0 . 7 \pm 2 . 8 }$ </td><td> $7 0 . 8 \pm 1 . 8$ </td></tr><tr><td> $\mathrm { A R C - C }$ </td><td> $5 7 . 4 { \pm } 0 . 6 $ </td><td> $5 8 . 3 { \pm } 0 . 7 $ </td><td> $5 7 . 4 \pm 0 . 7 $ </td><td> $5 7 . 7 { \pm } 0 . 9 $ </td><td> $6 0 . 7 \pm 1 . 2$ </td><td> ${ \bf 6 1 . 5 } \pm { \bf 0 . 2 }$ </td><td> $6 0 . 9 { \pm } 1 . 3 $ </td><td> $5 8 . 1 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { O B Q A }$ </td><td> $6 8 . 3 { \pm } 1 . 0 \ $ </td><td> $6 7 . 0 { \pm } 1 . 0 \ $ </td><td> $6 7 . 9 { \pm } 0 . 4 $ </td><td> $6 6 . 1 \pm 0 . 5$ </td><td> $6 7 . 8 { \pm } 0 . 9$ </td><td> ${ \bf 6 9 . 6 { \pm 0 . 3 } }$ </td><td> $6 9 . 5 { \pm } 0 . 9 \ $ </td><td> $6 5 . 3 { \pm } 1 . 0 $ </td></tr><tr><td> $_ { \mathrm { H e l l a S w a g } }$ </td><td> $5 7 . 0 { \pm } 1 . 0 $ </td><td> $5 4 . 5 { \pm } 2 . 6 $ </td><td> $5 6 . 5 { \pm } 0 . 7 $ </td><td> $5 3 . 9 2 0 . 6 $ </td><td> $5 5 . 5 \pm 1 . 7$ </td><td> $5 7 . 1 \pm 1 . 5$ </td><td> ${ \bf 5 9 . 1 \pm 1 . 0 }$ </td><td> $5 3 . 0 { \pm } 1 . 8 $ </td></tr><tr><td>WinoG</td><td> ${ \bf 5 6 . 9 } \pm 2 . 0$ </td><td> $5 2 . 6 \pm 3 . 1$ </td><td> $5 3 . 3 { \pm } 4 . 0 $ </td><td> $5 2 . 0 { \pm } 1 . 6 $ </td><td> $5 5 . 2 \pm 1 . 7$ </td><td> $5 1 . 2 { \pm } 1 . 3 $ </td><td> $5 2 . 1 \pm 2 . 9$ </td><td> $5 0 . 4 \pm 0 . 0 $ </td></tr><tr><td>Mean</td><td>68.1</td><td>66.3</td><td>67.5</td><td>64.7</td><td>66.6</td><td>66.4</td><td>68.2</td><td>63.1</td></tr></table>

Table 13: Full matched-budget results on Llama-3.2-1B-Instruct. Mean±std over three seeds (×100); the Mean row is recomputed from the 11 displayed task means.
<table><tr><td colspan="5">Task FCCA (36.9K) LoRA (7.37M) DoRA (7.54M) PiSSA (7.37M)</td></tr><tr><td>SVAMP</td><td> ${ \bf 7 9 . 7 \pm 1 . 5 }$ </td><td> $7 8 . 8 \pm 1 . 6$ </td><td> $7 9 . 2 \pm 0 . 3$ </td><td> $7 7 . 3 { \pm } 2 . 6 $ </td></tr><tr><td>GSM8K</td><td> ${ \bf 6 8 . 1 \pm 0 . 8 }$ </td><td>67.7±1.9</td><td> $6 7 . 2 \pm 0 . 7$ </td><td> $5 1 . 4 \pm 1 . 7$ </td></tr><tr><td>SST-2</td><td> ${ \bf 9 5 . 1 } { \bf \pm 0 . 2 }$ </td><td> $9 4 . 5 { \pm } 0 . 5 $ </td><td> $9 4 . 9 { \pm } 0 . 4 $ </td><td> $9 4 . 4 \pm 0 . 7 $ </td></tr><tr><td>QNLI</td><td> ${ \bf 8 8 . 9 2 1 . 2 }$ </td><td> $8 7 . 9 { \pm } 0 . 6 $ </td><td> $8 7 . 4 { \pm } 0 . 6 $ </td><td> $8 7 . 3 { \pm } 1 . 5 $ </td></tr><tr><td>CoLA</td><td> $8 5 . 3 { \pm } 1 . 3 $ </td><td> ${ \bf 8 5 . 9 2 0 . 7 }$ </td><td> ${ \bf 8 5 . 9 2 0 . 7 }$ </td><td> $8 5 . 0 { \pm } 1 . 0 \ $ </td></tr><tr><td>RTE</td><td> $8 9 . 3 { \pm } 0 . 9 \ \qquad $ </td><td> $\mathbf { 8 9 . 7 \pm 0 . 2 }$ </td><td> $8 9 . 4 \pm 0 . 7$ </td><td> $8 8 . 6 \pm 2 . 1$ </td></tr><tr><td>MRPC</td><td>89.4±1.3</td><td> ${ \bf8 9 . 5 \pm 0 . 4 }$ </td><td>89.4±0.3</td><td>89.1±0.5</td></tr><tr><td>ARC-C</td><td> $8 3 . 1 { \pm } 1 . 1 $ </td><td> $8 4 . 3 { \pm } 0 . 3 $ </td><td> ${ \bf 8 4 . 7 \pm 0 . 5 }$ </td><td> $8 2 . 5 { \pm } 0 . 8 $ </td></tr><tr><td>OBQA</td><td> $8 5 . 9 { \pm } 1 . 0 \ $ </td><td> $8 8 . 2 { \pm } 0 . 3 $ </td><td> $\mathbf { 8 8 . 3 \pm 0 . 7 }$ </td><td>84.6±1.2</td></tr><tr><td>HellaSwag</td><td> $7 5 . 6 \pm 1 . 0$ </td><td>76.9±1.1</td><td> $7 6 . 6 \pm 1 . 3$ </td><td> $7 3 . 0 { \pm } 0 . 2 \ $ </td></tr><tr><td>WinoG</td><td> $7 2 . 8 { \pm } 0 . 6 $ </td><td> ${ \bf 7 3 . 3 \pm 1 . 3 }$ </td><td> $7 2 . 7 { \pm } 0 . 9$ </td><td> $7 1 . 4 { \pm } 0 . 7 $ </td></tr><tr><td>Mean</td><td>83.0</td><td>83.3</td><td>83.2</td><td>80.4</td></tr></table>

<table><tr><td>Baseline</td><td>Mean ∆</td><td>Wins</td><td>Wilcox. p t-test p</td><td></td></tr><tr><td>RawGrad</td><td></td><td>+4.78 11/11</td><td>.001</td><td>.015</td></tr><tr><td>FiLoRA-core</td><td></td><td>+3.51 10/11</td><td>.002</td><td>.016</td></tr><tr><td>MiLoRA-core</td><td></td><td>+2.57 10/11</td><td>&lt; .01</td><td>.005</td></tr><tr><td>LoRA-XS-core</td><td></td><td>+2.31 10/11</td><td>&lt; .01</td><td>.017</td></tr><tr><td> $\mathrm { V e R A - c o r e }$ </td><td>+2.74</td><td>9/11</td><td>.014</td><td>.015</td></tr><tr><td> $_ \mathrm { E V A - c o r e }$ </td><td></td><td>+3.09 10/11</td><td>.007</td><td>.090</td></tr><tr><td>Random</td><td></td><td>+3.28 11/11</td><td>&lt; .01</td><td>&lt; .01</td></tr></table>

Table 15: FCCA minus each matched-budget baseline on Qwen2.5-3B, paired over 11 unrounded task means. “Wins” counts tasks on which FCCA is higher; rounded main-table cells can display a tie even when the unrounded comparison is nonzero.  
Table 14: Full Qwen2.5-3B comparison with trainablebasis rank-16 references. Mean±std over three seeds (×100).

## H Resource Accounting and Reproducibility

nitude does not rescue the raw inverse-whitened coordinates. Since thin QR preserves the representable update family, the sweep rules out simple learning-rate mistuning within the tested range and strengthens the coordinate-conditioning interpretation. Specialized preconditioners and layer-specific optimization remain untested.

## H.1 Parameter and storage formulas

For target matrices $\begin{array} { r } { W _ { \ell } \in \mathbb { R } ^ { m _ { \ell } \times n _ { \ell } } } \end{array}$ , the two trainable-state counts are

$$
N _ { \mathrm { c o r e } } = \sum _ { \ell } r ^ { 2 } ,\tag{10}
$$

$$
N _ { \mathrm { f a c t o r s } } = \sum _ { \ell } r ( m _ { \ell } + n _ { \ell } ) .\tag{11}
$$

The same main-text table compares diagonal moments with dense m × m and $n \times n$ K-FAC factors estimated from 128–256 examples. Diagonal whitening is higher by 8.7 points on SVAMP, 0.4 on ARC-C, and 1.7 on GSM8K. Finite-sample conditioning is a plausible explanation, but this comparison does not test stronger shrinkage, structured factors, or much larger calibration sets; it only justifies the diagonal default in this regime.

At rank 16 in every $q / k / v / o$ projection of Qwen2.5-3B, these are 36,864 and 7,372,800, respectively, exactly a factor of 200. DoRA adds magnitude parameters for a total of 7,538,688. Adam optimizer-state savings follow the trainablestate difference, aside from framework overhead.

<table><tr><td>Task</td><td>None: svd(G)</td><td>Input only</td><td>Output only</td><td>Both (FCCA)</td><td>Both-None</td></tr><tr><td>SVAMP</td><td>73.4</td><td>76.7</td><td>76.7</td><td>79.7</td><td>+6.3</td></tr><tr><td>ARC-C</td><td>74.7</td><td>83.2</td><td>80.7</td><td>83.1</td><td>+8.4</td></tr><tr><td>SST-2</td><td>92.4</td><td>95.1</td><td>94.8</td><td>95.1</td><td>+2.7</td></tr><tr><td>RTE</td><td>71.7</td><td>89.3</td><td>88.4</td><td>88.2</td><td>+16.5</td></tr><tr><td>MRPC</td><td>80.4</td><td>88.4</td><td>87.3</td><td>88.4</td><td>+8.0</td></tr><tr><td>WinoGrande</td><td>55.6</td><td>70.6</td><td>68.1</td><td>72.8</td><td>+17.2</td></tr></table>

Table 16: Whitening-side ablation on Qwen2.5-3B (three-seed means, ×100). Input/output divide by ${ \sqrt { A } } / { \sqrt { D } }$ Both is FCCA. The study reports means where per-seed standard deviations were not retained.
<table><tr><td>Configuration</td><td>SVAMP</td></tr><tr><td>None: svd(G)</td><td> $7 3 . 4 { \pm } 2 . 1 $ </td></tr><tr><td>Input only:  $G A ^ { - 1 / 2 }$ </td><td> $7 6 . 7 { \pm } 1 . 4 $ </td></tr><tr><td>Output only:  $D ^ { - 1 / 2 } G$ </td><td> $7 6 . 7 { \pm } 2 . 9$ </td></tr><tr><td>Both:  $D ^ { - 1 / 2 } G A ^ { - 1 / 2 }$ </td><td> ${ \bf 7 9 . 7 \pm 1 . 5 }$ </td></tr></table>

Table 17: Exact-pipeline whitening-side ablation on SVAMP. Only the whitening gates change.

<table><tr><td>Method</td><td>Task</td><td>64</td><td>128</td><td>256</td><td>512</td></tr><tr><td>FCCA</td><td>SVAMP</td><td>79.0±1.2</td><td>79.7±1.5</td><td>78.8±0.6</td><td>80.1±1.1</td></tr><tr><td>RawGrad</td><td>SVAMP</td><td>71.7±2.1</td><td>73.6±1.6</td><td>71.8±2.7</td><td>74.2±1.6</td></tr><tr><td>FCCA</td><td>ARC-C</td><td>83.3±0.8</td><td>83.1±1.1</td><td>82.9±1.0</td><td>83.0±0.5</td></tr></table>

Table 18: Accuracy sensitivity to calibration size (three seeds, ×100).

The complete unmerged FCCA artifact is larger than the core count: task-specific P and Q contain the same 7,372,800 basis elements as a pair of LoRA factors, plus the 36,864-value core. Those bases are frozen, so they do not require gradients or Adam states. If the update is merged, the separate adapter disappears at inference, but deployment then keeps a task-specific copy of the base weights. This distinction is why the paper describes FCCA as optimizer- and trainable-state efficient, not as a universally 36.9K stored adapter.

## H.2 Measured phase profile

Table 22 reports measured phase costs, and Table 23 converts them into projected totals for approximately 262 optimizer steps. The one-time 35-second calibration is offset by FCCA’s lower per-step time, producing a projected total close to LoRA, below DoRA, and roughly half PiSSA. Peak GPU memory differs by only 0.27–0.35 GB because the frozen model dominates; the larger saving is in gradient and optimizer state. Calibration becomes a smaller fraction of longer runs or reused bases, although host memory for cross moments can still matter at larger widths.

<table><tr><td>Calib. examples</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td><td>512</td></tr><tr><td>cos(P, P512)</td><td>.716</td><td>.789</td><td>.890</td><td>.932</td><td>.976</td><td>1.000</td></tr><tr><td>cos(Q, Q512)</td><td>.817</td><td>.863</td><td>.917</td><td>.943</td><td>.981</td><td>1.000</td></tr></table>

Table 19: Mean principal-angle cosine between the selected bases and the 512-example reference.

<table><tr><td>Calibration source</td><td>SVAMP</td><td>∆ from matched</td></tr><tr><td>SVAMP (matched)</td><td>79.7±1.5</td><td>0.0 (ref.)</td></tr><tr><td>GSM8K</td><td>78.0±0.5</td><td>-1.7</td></tr><tr><td>ARC-C</td><td> $7 4 . 6 { \pm } 1 . 6 $ </td><td>-5.1</td></tr><tr><td>SST-2</td><td>72.6±2.3</td><td>-7.1</td></tr></table>

Table 20: Training on SVAMP with bases calibrated on another dataset.

<table><tr><td>Task</td><td>Best no-QR</td><td>Sweep outcome</td></tr><tr><td>SVAMP</td><td>0.0</td><td>all four LRs diverged/NaN</td></tr><tr><td>ARC-C</td><td>22.7</td><td>near chance (25%)</td></tr><tr><td>GSM8K</td><td>0.0</td><td>all four LRs diverged/NaN</td></tr></table>

Table 21: No-QR learning-rate sweep on Qwen2.5-3B $( r = 1 6 ,$ all $q / k / v / o$ projections). The grid is {5 × $1 0 ^ { - 3 } , 2 \times 1 0 ^ { - 3 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } \}$ ; the table reports the best score observed for each task.

## H.3 Software and implementation checklist

Measurements use a single NVIDIA H20, CUDA 12.8, PyTorch 2.7.1, Transformers 4.54, PEFT 0.16, and bf16. The reproducibility-critical settings are:

• Models: Qwen2.5-1.5B/3B/7B-Instruct and Llama-3.2-1B-Instruct.

• Seeds: 42, 43, 44; batch size 1; gradient accumulation 8.

• Targets: every $q / k / v / o$ projection; default rank 16; zero core initialization.

• FCCA: learning rate $5 \times 1 0 ^ { - 3 }$ ; warmup 0.03 or 0.1; calibration 128 or 256.

• Matched baselines: learning rate $2 \times 1 0 ^ { - 3 }$ or 5 $\times 1 0 ^ { - 3 } ;$ warmup 0.1.

• Calibration: calib\_group\_size=999; CPU
<table><tr><td>Method</td><td>Init./calib.</td><td>Step time</td><td>Peak GPU</td><td>Trainable</td></tr><tr><td>FCCA</td><td>35 s</td><td>0.81 s</td><td>6.92 GB</td><td>36,864</td></tr><tr><td>LoRA</td><td>none</td><td>0.92 s</td><td>7.19 GB</td><td>7,372,800</td></tr><tr><td>DoRA</td><td>none</td><td>1.31 s</td><td>7.27 GB</td><td>7,538,688</td></tr><tr><td>PiSSA</td><td>41 s</td><td>1.76 s</td><td>7.20 GB</td><td>7,372,800</td></tr></table>

Table 22: Direct phase measurements on Qwen2.5-3B, SVAMP, rank 16, all $q / k / v / o$ projections, bf16, and one NVIDIA H20. The first column reports FCCA calibration and PiSSA initialization; PiSSA uses a one-time weight-SVD. “None” means no separate initialization or calibration phase.

<table><tr><td>Method</td><td>Init./calib.</td><td>Training</td><td>Total Init. share</td></tr><tr><td>FCCA</td><td>35 s</td><td>3.5 min</td><td>4.1 min 14%</td></tr><tr><td>LoRA</td><td>none</td><td>4.0 min</td><td>4.0 min n/a</td></tr><tr><td>DoRA</td><td>none</td><td>5.7 min 5.7 min</td><td>n/a</td></tr><tr><td>PiSSA</td><td>41 s</td><td>7.7 min 8.4 min</td><td>8%</td></tr></table>

Table 23: Projected complete SVAMP run from measured phase times and approximately 262 optimizer steps. FCCA’s first phase is calibration; PiSSA’s is weight-SVD initialization. “n/a” means no separate first phase. Totals are projections, not independent end-toend stopwatch measurements.

statistics; diagonal moments by default.

• Selection: best held-out validation configuration, followed by common evaluation on three seeds.

• Reporting: mean ± sample standard deviation; task-level Wilcoxon on unrounded means.

## I Use of Language Models

A language model was used to polish the writing and assist with organizing the manuscript structure. The authors reviewed and finalized all text.