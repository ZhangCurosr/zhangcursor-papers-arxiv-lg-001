# Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures

Liam Cooper<sup>1</sup>, Shinnung Jeong<sup>1</sup>, Hyeran Jeon<sup>2</sup>, Jefrey Young<sup>1</sup>, Hyesoon Kim<sup>1</sup>

<sup>1</sup>Georgia Institute of Technology, <sup>2</sup>University of California, Merced

## Abstract

Large language model (LLM) outputs are expected to be reproducible under greedy decoding, yet in practice the same model, prompt, and software stack produce diferent outputs on diferent GPUs. The root cause is floating-point non-associativity combined with hardware-dependent kernel selection. Inference frameworks select diferent matrix-multiplication kernels on each architecture, with diferent parallel reduction orders and unspecified tensor-core arithmetic, and the resulting rounding diferences can flip output tokens. Existing solutions have imperfect cross-architecture reproducibility and incur a significant performance penalty. We present a solution employing a set of fixed-configuration fused-upcast GEMM kernels that load 16-bit weights from memory, upcast them to FP32 in registers, and accumulate with IEEE-754 arithmetic in a reduction order that is a pure function ofthe problem shape and is therefore independent of the device, its SM count, or kernel scheduling. By fixing the floating-point reduction order as a function of problem shape alone, every GPU runs the same operation sequence, so cross-architecture reproducibility of the linear layers reduces to correct IEEE-754 arithmetic rather than to rounding diferences staying below a tie-flip threshold. We confirm our solution’s linear-layer outputs are bitwise identical across NVIDIA Ampere, Ada, and Hopper GPUs, while running 1.17 to 3.1× faster end-to-end than the state-of-the-art solution and cutting weight-memory trafic in half.

## Introduction

Reproducibility is a basic expectation of deployed software, and large language models are increasingly deployed inside systems that assume it. Continuous-integration pipelines need identical regression outputs to distinguish real behavioral changes from numerical noise. Regulated industries such as finance, healthcare, and law must be able to replay a model’s exact response for auditing. Agentic workloads rely on deterministic replay for undo/redo and debugging (Khatchadourian 2026; Gundersen and Kjensmo 2018). Greedy decoding (temperature 0, always selecting the highest-probability token, and fixing the seed) would ordinarily be expected to yield deterministic outputs.

Even with identical model weights, prompts, software versions, and greedy decoding, LLM outputs can diverge across GPU architectures, GPU counts, and batch sizes due to floating-point non-associativity and architecture-dependent tensor-core accumulations (Yuan et al. 2025; Atil et al. 2025; He and Thinking Machines Lab 2025; IEEE 2019; Muller et al. 2018). Low-precision formats such as BF16 further amplify these rounding diferences, which can flip the argmax when the top candidate tokens are nearly tied, causing the generated sequences to diverge (Fasi et al. 2021; Sun et al. 2023; Valpey et al. 2025).

The state-of-the-art mitigation for cross-GPU divergence is (Yuan et al. 2025), which stores weights in BF16 but performs all computation in FP32: every linear layer upcasts its weight matrix to a temporary FP32 copy and calls the vendor GEMM. This shrinks per-operation rounding error to the FP32 level, at which near-tie argmax flips become rare. But ’s design has two limitations. First, it enforces reproducibility at the cost of performance: LLM decoding is memory-bandwidth-bound, and because the transient FP32 copy is what the GEMM actually reads, ’s weight trafic, the quantity that sets decode latency, is fully FP32. The BF16 storage saves resident memory while leaving memory bandwidth demand unchanged. Second, its reproducibility is statistical rather than by construction: the vendor GEMM still selects diferent kernels with diferent reduction orders on each architecture, so diferent GPUs that we tested (A100, L40S, and H100) still compute diferent logits, and the tokens merely agree most of the time because the FP32-level diferences are usually too small to flip a tie.

We present , which delivers a stronger reproducibility property than at a significantly lower performance overhead. For accelerating reproducibility, replaces the cast-then-GEMM pattern with fused-upcast GEMM kernels: BF16 weights are loaded from HBM and upcast to FP32 in registers inside the matrixmultiplication kernel, with all accumulation performed by IEEE-754 fused multiply-add (FMA) instructions whose results are bit-specified for given operands on every architecture. Because the upcast happens entirely in registers, weight memory trafic returns to BF16 levels, halving the memory trafic during decode while eliminating the need for transient FP32 weight allocation. To enforce a stronger reproducibility, we pin every choice that vendor libraries vary per device: kernel configurations are compile-time constants selected by problem shape alone (no autotuning), no tensor-core paths are used for the FP32 accumulation, and small-batch decode shapes use a deterministic split-K scheme whose partial sums are combined in a fixed ascending order with no atomics. The complete floating-point reduction order is therefore a pure function of the problem shape.

This design converts cross-architecture reproducibility from an empirical tendency into a verifiable kernel-level property. We show that ’s linear-layer outputs are bitwise identical across three GPU architectures on every probe shape we test, spanning decode and prefill regimes, BF16 and FP32 weights, and ragged dimensions. As a corollary of the same design, is also batch-invariant within its decode bucket: a given request’s outputs are bitwise independent of how many other requests share its batch, a property identified as the key to run-to-run determinism in serving systems (He and Thinking Machines Lab 2025) and one that ’s vendor GEMMs do not have.

In summary, this paper makes the following contributions:

• An analysis of why FP32-compute nondeterminism mitigation pipelines remain both slow and only statistically reproducible, locating the residual divergence channel in the GEMM reduction order.

• , a linear layer built from fixed-configuration fused-upcast Triton GEMM kernels with IEEE-754 FMA accumulation and deterministic split-K, whose reduction order is a pure function of problem shape.

• A cross-architecture bitwise validation methodology and results: ’s linear-layer outputs are bit-identical across Ampere, Ada, and Hopper GPUs, a property we verify with seeded probes and independent per-node hashing.

• An end-to-end evaluation in vLLM showing improvement in reproducibility, throughput (1.17–3.1×), and memory (up to 1.4 GiB weight savings, converted to KV-cache capacity), over the state-of-the-art.

## Background and Related Work

## Floating-Point Non-Associativity and GPU Kernels

IEEE-754 floating-point addition rounds after every operation, so (a+b)+c ̸= a+(b+c) in general (IEEE 2019; Muller et al. 2018). A dot product of length K can therefore yield diferent results depending on how its partial sums are associated, with worst-case error growing with the accumulation depth and the condition number of the sum (Shanmugavelu et al. 2024). GPU GEMM kernels exploit this freedom aggressively: tile sizes, the number of accumulators, split-K factors, and atomic reduction schemes all can change the association order, and high-performance libraries such as cuBLAS select among many such kernels using architectureand shape-dependent heuristics. Run-to-run determinism on a single device is typically guaranteed, but nothing constrains two diferent architectures to select the same kernel.

Tensor cores add a second, deeper source of architecture dependence. The matrix-multiply-accumulate (MMA) units on NVIDIA GPUs accumulate internally with truncated (round-toward-zero) significand alignment, carry out of the standard, and per-generation diferences in intermediate width and normalization. These behaviors are oficially undocumented, but are observable to cause diferences across Volta, Ampere, Ada, and Hopper (Fasi et al. 2021; Sun et al. 2023; Markidis et al. 2018; Valpey et al. 2025; Li et al. 2024; Xie et al. 2025). Notably, “FP32” GEMMs on modern NVIDIA GPUs are typically executed on tensor cores in the TF32 format, which rounds inputs to a 10-bit significand (Stosic and Micikevicius 2021), so even a nominally FP32 pipeline can silently inherit tensor-core arithmetic unless TF32 is explicitly disabled.<sup>1</sup> In contrast, the scalar fused multiply-add (FMA) instructions on CUDA cores implement IEEE-754 arithmetic exactly: for given operands, an FMA produces the same bits on every architecture. This asymmetry, bit-specified scalar FMA versus unspecified MMA internals, is the foundation builds on.

## Nondeterminism in LLM Inference

Yuan et al. (2025) provide the first systematic study of LLM inference divergence across GPU types, GPU counts, and batch sizes, showing that greedy decoding is not reproducible in practice: under BF16, accuracy on AIME’24 (Zhang and Math-AI 2024) varies by up to 9% and output lengths by up to 9,000 tokens across 12 runtime configurations, with reasoning models diverging on essentially 100% of problems within the first ∼100 tokens. Their analysis attributes the divergence to reduction-order-sensitive rounding interacting with near-tie argmax decisions: at observed divergence points the top-1/top-2 probability gap is a fraction of a percent, well within reach of BF16 rounding noise. Their mitigation, , loads weights in FP32, stores linear-layer weights in BF16, and upcasts each weight matrix back to FP32 justin-time for its matmul, reducing divergence to below 3.4% of problems at 34% less memory than full FP32.

He and Thinking Machines Lab (2025) attribute serving nondeterminism to the lack of batch invariance: varying batch sizes change GEMM tilings and reduction orders, producing diferent outputs despite deterministic kernels. Their batch-invariant BF16 kernels eliminate withindevice nondeterminism at a reported ∼20% matmul slowdown but remain architecture-dependent. Like their approach, uses batch-invariant kernels, but additionally guarantees bit-identical results across GPU architectures by restricting all arithmetic to IEEE-compliant FMAs.

Two recent systems address determinism at diferent layers of the stack. LLM-42 (Gond et al. 2026) argues that batchinvariant kernels are overly restrictive because they preclude shape-adaptive optimizations. Instead, it restores determinism through scheduler-level verification: fast nondeterministic decoding is periodically checked against a fixed-shape deterministic replay, with rollback on mismatch, so overhead applies only to requests requiring determinism. LLM-42 only addresses reproducibility for LLMs on the same GPU architecture, unlike ’s goal of fast cross-architecture determinism. Hawkeye (Badash et al. 2026) reverse-engineers the architecture-specific internals of NVIDIA tensor-core MMA instructions to emulate them on CPUs, demonstrating that Ampere, Ada, and Hopper implement diferent MMA arithmetic. In contrast, makes the dominant kernels themselves bit-identical across GPU architectures, eliminating the need for slower replay on CPU. aims to reduce the verification burden by accelerating reproducibility whereas Hawkeye reports an order of magnitude slower for bitwise reproducibility.

Fusing a storage-format conversion into the GEMM has been explored by prior works. Weight-only quantization systems dequantize INT4 weights to FP16 in registers inside the matmul for bandwidth (Lin et al. 2024; Frantar et al. 2025). takes the essence of this technique, but with the opposite numerical goal: those kernels autotune per device and accumulate on tensor cores to maximize throughput, surrendering any cross-architecture bit-for-bit agreement, whereas pins every such degree of freedom to keep the reduction order identical across devices.

Broader literature characterizes numerical variability in HPC and deep learning: compiler-induced variability (Bentley et al. 2019), randomized testing of floating-point behavior across platforms (Laguna 2020), cross-vendor accelerator diferences (Zahid, Laguna, and Le 2024; Li et al. 2024), and the impact of tooling randomness on training (Zhuang et al. 2022). Our focus is complementary: we target the inferencetime, cross-architecture channel and the operations that dominate LLM FLOPs.

## Design

reimplements the linear layers of a hybridprecision (BF16-storage, FP32-compute) inference pipeline as fixed-configuration Triton GEMM kernels. Figure 1 summarizes the diference from . computes each linear layer as

$$
y = x W _ { \mathrm { f p 3 2 } } ^ { \top } + b , \qquad W _ { \mathrm { f p 3 2 } } = \mathrm { c a s t } ( W _ { \mathrm { b f 1 6 } } ) ,\tag{1}
$$

where the cast materializes a full FP32 copy of W in HBM on every forward pass: one extra kernel launch, a transient allocation of size 2|W|, and, because the GEMM reads the FP32 copy, weight trafic at FP32 width. computes the same quantity with the upcast fused into the GEMM:

$$
y = x \operatorname* { u p } ( W _ { \mathrm { b f 1 6 } } ) ^ { \top } + b ,\tag{2}
$$

where up(·) denotes a per-tile register upcast: each BF16 weight tile is loaded from HBM, widened to FP32 in registers, multiplied against the FP32 activation tile, and accumulated in FP32. Weights cross the memory bus at BF16 width and nothing FP32-sized is ever materialized in HBM. Since decode-phase GEMMs are memory-bandwidth-bound and weights dominate the bytes, halving weight trafic yields lower decode latency.

The performance improvement comes without numerical compromise. Reproducibility is supported by four design rules that constrain the principal sources of device-dependent variation:

![](images/4e88707e85fd01b30552f6df4007b269575e3f7ac013a3cffcf127146f74dd8f.jpg)  
Figure 1: Dataflow in the GPU for and . (a) materializes an FP32 copy of every weight matrix in HBM on every forward pass and hands it to an architecture-tuned vendor GEMM: each architecture is susceptible to reduce in a diferent order. (b) loads BF16 weights directly and upcasts in registers inside a fixedconfiguration IEEE-FMA kernel, reducing weight trafic and fixing the reduction order.

R1: IEEE FMA only. Every dot product is executed with IEEE-754 FMA on CUDA cores (input\_precision=“ieee” on each Triton tl.dot). Triton’s default for FP32 operands on Ampere and newer is TF32 tensor cores, whose internal accumulation is unspecified and architecture-dependent (Fasi et al. 2021; Stosic and Micikevicius 2021), causing cross-architecture disagreements (Xie et al. 2025). Framework-level TF32 switches do not propagate into Triton-generated kernels, so the constraint must be applied per operation. IEEE FMA is bit-specified for given operands on every architecture, making the elementary arithmetic outputs identical.

R2: No autotuning. Autotuners benchmark candidate configurations on the local device and keep the fastest, which makes the compiled kernel a function of the machine it was tuned on. In our ablation testing, running the same search over the same candidate space on A100, L40S and H100 selects the same configuration for only 3 of 9 representative GEMM shapes, and repeating the identical search on one device changes its own answer on 1 of 9 shapes, because near-equal candidates trade places under measurement noise. The split factor is a standard tuned knob that fully determines the reduction tree, and the K-block size sets the segment boundaries feeding it. ’s configurations are instead compile-time constants selected by a pure function of the problem shape (M, N, K): one for decode-shaped GEMMs (M ≤ 64) and one for prefill-shaped GEMMs, tuned once ofline and pinned for every architecture.

R3: Deterministic split-K. Decode-shaped GEMMs (M small, N × K large) underutilize the GPU without splitting the reduction dimension, but classic split-K combines partial sums with atomic additions whose order is a scheduling race (Figure 2a). splits K into S contiguous segments, where S is a pure function of K; each segment is accumulated sequentially by exactly one program into a partials workspace, and a second kernel reduces the S partials in fixed ascending segment order (Figure 2b). No atomics appear anywhere, so the complete floating-point reduction tree is a deterministic function of shape, independent of the device, its SM count, and grid scheduling.

![](images/f2066d5052e3d5174637d2f97f4497b9acc834cd7f093c30621e57bf1110a0e1.jpg)  
Figure 2: Split-K reduction of one output element. (a) Classic split-K merges partial sums with atomicAdd, so the reduction order is whatever order the segments’ programs happen to finish in, diferent per run and per device. (b) fixes the segmentation as a pure function of K and sums the partials in ascending segment order in a second kernel, making the floating-point reduction tree a deterministic function of the problem shape.

R4: Batch invariance by construction. Within the decode bucket, the configuration and the K-segmentation depend only on K, never on the batch dimension M. A given row of the batch therefore executes the same FMA sequence at any batch size from 1 to 64: per-request outputs are bitwise independent of co-scheduled load. This is the property He and Thinking Machines Lab (2025) obtain with dedicated batch-invariant kernels.

Together, R1–R4 supports reproducibility by construction: for a given input, weight, and shape, each FLOP and its position in the reduction tree are determined by the problem shape. Cross-architecture reproducibility then no longer depends on FP32-level rounding diferences staying below a tie-flip threshold; it reduces to a single premise, that each GPU implements IEEE-754 FP32 arithmetic correctly. That premise is documented for CUDA core FP32 and confirmed by our work on A100, L40S, and H100 GPUs where the linear layer outputs are bit-identical on every shape we probe. This is a diferent basis for reproducibility than ’s, whose reduction order stays device-dependent and which can only bound the magnitude of the resulting diferences. An ablation study of R1–R4 can be found in the technical supplement (Section C).

We additionally found that ’s published implementation stores the output projection (lm\_head) in FP32: its weight-conversion filter matches only attention and MLP projection layers, silently exempting the vocabulary projection and contradicting the stated bf16-storage policy at a cost of vocab×hidden×2 bytes (≈1 GiB for an 8B model with a 128k vocabulary). We remedy this by storing the lm\_head in BF16 for models with untied embeddings.

covers the linear layers, which account for the majority of FLOPs and the divergence channel identified by prior analysis. Attention, rotary embeddings, normalization, and the sampler remain the engine’s FP32 implementations and may in principle select architecture-specific code paths. The end-to-end experiments below measure whether any residual channel is observable in practice.

## Evaluation

## Experimental Setup

Hardware and software. We evaluate on three NVIDIA architectures: Ampere (A100), Ada Lovelace (L40S), and Hopper (H100). All experiments run on a single GPU. All nodes run an identical software environment: PyTorch 2.6.0 with CUDA 12.4, and Triton 3.2.0. Following , vLLM 0.8 and the xFormers attention backend is used.

Models and benchmarks. We evaluate three instructiontuned models spanning tied and untied embeddings and a range of sizes: meta-llama/Llama-3.2-3B-Instruct (Meta 2024), Qwen/Qwen3-4B-Instruct-2507 (Qwen Team 2025), and deepseek-ai/DeepSeek-R1-Distill-Llama-8B (DeepSeek-AI 2025). Qwen/Qwen3-14B is added for memory profiling. Benchmarks are GSM8K (Cobbe et al. 2021), MATH500 (Lightman et al. 2024), AIME24 (Zhang and Math-AI 2024), and GPQA-Diamond (Rein et al. 2023). We selected these datasets to align with ’s evaluation and all are evaluated with greedy decoding and batch size 32.

The Llama-3.2 template normally sets a date string which gets embedded in the rendered system prompt. Evaluation sweeps spanning a calendar day boundary therefore present diferent token sequences to nominally identical configurations. This artifact can masquerade as hardware nondeterminism, manifesting as both cross-GPU and cross-seed divergence, afecting the baseline and the mitigated system alike. We observed up to a 37.60% diference in cross-GPU prompt divergence which would otherwise have no divergence on sweeps spanning over two calendar days. We therefore pin the date string to a fixed value in all reported experiments. The Qwen3-4B and DeepSeek-R1-Distill templates contain no date field and are unafected.

Methods. We compare three inference configurations. The unmitigated model default (BF16 storage and compute) is the baseline whose nondeterminism motivates the problem. replaces ’s linear layers with our kernels. This comparison serves as our principal ablation: replacing ’s fixed-order kernels with architecturetuned vendor GEMMs while holding the storage precision, compute precision, model, and surrounding inference stack fixed isolates the efect of deterministic kernel design. The unmitigated BF16 configuration provides a complementary end-to-end ablation of the reproducibility mitigation stack. Each (model, benchmark, GPU, method) cell is run under three diferent seeds.

## Cross-Architecture Determinism

We first validate the kernel-level claim directly. A seeded probe generates fixed inputs and weights for seven GEMM cases spanning both shape buckets (decode and prefill), BF16 and FP32 weights, and ragged shapes. The inputs are seeded on the CPU so every GPU sees identical input bits. Across A100, L40S, and H100, all seven cases are bitwise identical under : every tensor compares equal element-forelement and every digest matches.

For the same probe through ’s cast-thencuBLAS path, no shape produces identical bits on all three architectures. A100 and L40S disagree on all seven shapes, A100 and H100 on four, and H100 and L40S on five; in each divergent case 88–99% of output elements difer, with maximum diferences of $2 . 7 \times 1 0 ^ { - 7 } \mathrm { t o } 4 . 9 \times 1 0 ^ { - 6 }$ of the output scale. The scattered pairwise agreements are cases where two architectures happened to select the same cuBLAS kernel for that shape: agreement by coincidence of undocumented kernel-selection heuristics, constrained by no specification and stable across neither shapes nor architecture pairs. These diferences sit four orders of magnitude below BF16 rounding noise, which is why ’s tokens usually (but are not guaranteed to) agree.

The probe also corroborates batch invariance in . On all three architectures, row 0 of a batch-M GEMM is bit-identical to the batch-size-1 result for every $M \leq 6 4$ tested. Under , row 0 changes bits at every tested batch size $M \geq 2$ on A100 and H100 (and at all but the smallest sizes on L40S), so inherits batchsize sensitivity even at FP32 precision, while ’s per-request outputs are independent of co-scheduled load.

## End-to-End Determinism

Table 1 reports cross-GPU and cross-seed token divergence for end-to-end inference using all three methods. Three observations follow.

First, unmitigated inference is not remotely reproducible. Under BF16, 31–100% ofproblems (median ≈85%) produce a diferent greedy token stream on at least one pair of GPU architectures, and 6–25% ofstreams difer across seeds on the same GPU purely from batch-composition variation. This is the reproducibility crisis set out to fix, quantified here across three architectures.

Second, FP32 compute alone ( ) shrinks divergence to well under 1% on every configuration. It does not reach zero: a residual persists on five of twelve cells, up to 0.51%. ’s guarantee is statistical: the FP32-level diferences are usually too small to flip a greedy decision, but nothing forces them to zero.

Third, the controlled kernel ablation shows that eliminates the linear-layer channel entirely. is no worse than in every one of the twelve cells. On DeepSeek it is zero across all benchmarks, and for Qwen on all but GSM8K. On Llama-3.2-3B it is zero on all four benchmarks. The single cell where is nonzero, Qwen3- 4B on GSM8K (0.08%, one problem versus ’s 0.23%), is a single token that sits at an exact FP32 tie on H100 while A100 and L40S resolve it with a $3 . 8 \times 1 0 ^ { - 6 } ,$ -nat margin. The residual is caused by the unpinned attention kernel in layer 0 on H100 rather than to the linear layers. The final answer is unchanged on all three GPUs. pins the reduction order of the linear layers, but attention, normalization, and the sampler remain vendor kernels whose reduction order stays architecture-dependent. The pinned QKV projection is bitwise identical on all three GPUs with the first divergence being the output of the layer-0 attention, where the H100 departs from A100 and L40S by about $6 \times 1 0 ^ { - 8 }$ That perturbation then propagates and compounds through the residual stream, and a greedy decision flips only where it grows to exceed the local decision margin. Such exposed positions are rare: across the 381,426 decoded GSM8K positions in this benchmark only one flips. At that position the tie is resolved by argmax, which deterministically returns the lower token id. ’s guarantee is thus exact for the linear layers, which carry the dominant FLOPs and the divergence channel prior analysis identifies, but does not extend to the remaining non-GEMM kernels. This confirms the Design section scope caveat and isolates fixed-order attention as the outstanding source of nondeterminism. Across every method and cell, ’s cross-seed divergence is zero.

![](images/a1e3f0723419e426f7c46289bbde743f6de33259c1b2f588d2e10882e4dfdb81.jpg)  
Figure 3: The two quantities that decide whether a token flips, plotted on one axis (nats). (a) Cumulative distribution of the greedy decision margin $^ { g , }$ the top-1/top-2 logit gap, over 2.6M decoded positions. Dotted verticals mark each method’s median cross-GPU shift, so the height at which a vertical meets a model’s curve is that method’s per-token flip probability on that model. (b) Cumulative distribution of the cross-GPU shift |∆| of that same margin, split by architecture pair. The left-hand intercept is the fraction of positions on which the two GPUs agree bitwise.

## Decision Margins and Logit Noise

Table 1 leaves an obvious question: why some solutions leave some model and benchmark cells reproducible and others not? A greedy token flips between two GPUs only when the hardware perturbs the top-1/top-2 logit diference by more than that diference itself, so the answer is a comparison of two quantities measured on the same scale. We denote the decision margin at position t as $g ( t ) = \ell _ { 1 } ( t ) - \ell _ { 2 } ( t )$ , the logit gap between the best and second-best token. The cross-GPU shift is $\Delta ( t ) = g _ { A } ( t ) - g _ { B } ( t )$ for two architectures A and B at the same position. Both are read directly from the stored top-5 log-probabilities: the log-sum-exp normalizer is token-independent and cancels in the diference, so a logprobability gap is also a logit gap. We measure ∆ only on the prefix over which the two runs still emit identical tokens, because past the first flip they are decoding diferent sequences and their positions are no longer comparable. A position is exposed exactly when $g < \Delta$

<table><tr><td rowspan="2"></td><td rowspan="2">Task</td><td colspan="3">Cross-GPU divergence (% of problems)</td><td colspan="3">Cross-seed divergence (% of pairs)</td></tr><tr><td>Unmitigated</td><td>LayerCast</td><td>ReproFuse</td><td>Unmitigated</td><td>LayerCast</td><td>ReproFuse</td></tr><tr><td rowspan="4">DeepSeek-R1-Distill-8B</td><td>GSM8K</td><td>88.86</td><td>0.15</td><td>0</td><td>19.06</td><td>0</td><td>0</td></tr><tr><td>MATH500</td><td>93.00</td><td>0.40</td><td>0</td><td>25.40</td><td>0</td><td>0</td></tr><tr><td>AIME24</td><td>100.00</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GPQA-Diamond</td><td>100.00</td><td>0.51</td><td>0</td><td>12.74</td><td>0</td><td>0</td></tr><tr><td rowspan="4">Llama-3.2-3B</td><td>GSM8K</td><td>66.64</td><td>0</td><td>0</td><td>17.34</td><td>0</td><td>0</td></tr><tr><td>MATH500</td><td>79.60</td><td>0</td><td>0</td><td>13.60</td><td>0</td><td>0</td></tr><tr><td>AIME24</td><td>86.67</td><td>0</td><td>0</td><td>17.78</td><td>0</td><td>0</td></tr><tr><td>GPQA-Diamond</td><td>30.81</td><td>0</td><td>0</td><td>12.07</td><td>0</td><td>0</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>GSM8K</td><td>70.81</td><td>0.23</td><td>0.08</td><td>5.96</td><td>0</td><td>0</td></tr><tr><td>MATH500</td><td>83.60</td><td>0</td><td>0</td><td>18.33</td><td>0</td><td>0</td></tr><tr><td>AIME24</td><td>100.00</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GPQA-Diamond</td><td>99.49</td><td>0.51</td><td>0</td><td>20.09</td><td>0</td><td>0</td></tr></table>

Table 1: End-to-end determinism across A100, L40S, and H100. Cross-GPU divergence: percentage of problems whose greedy token stream difers across at least one pair of GPU architectures at matched seed. Cross-seed divergence: percentage of (problem, seed-pair) comparisons that difer on the same GPU. Unmitigated BF16 diverges on 30.81–100% of problems across GPUs. greatly reduces this but leaves a residual on five of twelve configurations, up to 0.51%. is ≤ in every cell and is exactly 0 wherever the divergence was driven by the linear layers. Its one nonzero cell, Qwen3-4B GSM8K at 0.08%, is a single tied token localized to the unpinned attention kernel on H100 rather than the linear layers, and still improve on ’s 0.23%. The divergence metric used is further discussed in technical supplement Section B.

Figure 3(a) shows that the models are confident at the typical position (median margin 4 to 19 nats) but that the low tail is heavy enough to matter: about 1% of positions sit within 0.1 nats of a tie and roughly 1 in $1 0 ^ { 4 }$ within $1 0 ^ { - 3 }$ Below $1 0 ^ { - 1 }$ the tail is close to linear, $P ( g \leq x ) \approx \rho x$ with $\rho$ between 0.06 and 0.17 per nat, which is the reason there is no safe precision short of exactness: the exposed fraction is proportional to the noise, so every order of magnitude of extra precision buys exactly one order of magnitude fewer flips and no threshold below which flips stop.

Figure 3(b) places the three methods’ cross-GPU shifts on that same axis. Unmitigated BF16 shifts the margin by a median of $6 . 3 \times 1 0 ^ { - 2 }$ nats, squarely inside the exposed tail, which is why 31 to 100% of problems diverge. ’s FP32 compute moves the shift down by three and a half orders of magnitude, to a median of $1 . { \overset { \cdot } { 8 } } \times 1 0 ^ { - 5 }$ nats between A100 and L40S and $3 . 6 \times 1 0 ^ { - 5 }$ against H100, but it almost never removes it: only 4.0% and 2.3% of positions respectively come out bitwise equal, because cuBLAS still reduces in a diferent order on each architecture. inverts that ratio. Between A100 and L40S its logits are bitwise identical at 98.4% of 3.4M positions, and at 100.000% on both Llama-3.2-3B and Qwen3-4B, with the surviving mass another order of magnitude smaller than ’s. Against H100 the shift is not eliminated (median $\mathrm { 7 . 9 \times 1 0 ^ { - 6 } }$ nats, 9.6% bitwise equal): the linear layers are pinned, but the attention, normalization, and sampler kernels vLLM selects on sm90 are not. Llama-3.2-3B adds no residual of its own.

Two checks confirm that the margin-versus-shift picture is the operative mechanism. First, flips should occur where the margin is below the noise: at the first divergent token of every cross-GPU flip event, the median margin is 0.125 nats under BF16 and $1 . 1 \times 1 0 ^ { - 5 }$ to $7 . 4 \times 1 0 ^ { - 5 }$ nats under , tracking each method’s own shift across four decades. Second, the picture predicts rates, not just orderings. Treating margin and shift as independent draws gives a per-token flip probability $\begin{array} { r } { p = \frac { 1 } { 2 } \operatorname { E } _ { \Delta } [ \bar { P ( g < | \Delta | ) } ] } \end{array}$ ], where the factor of one half is the single sign of the shift that moves the decision, and a per-problem divergence of $1 - ( 1 - p ) ^ { L }$ over L decoded tokens. Across the 18 (model, benchmark, method) cells with nonzero measured divergence, spanning 0.08% to 100%, the median ratio of predicted to measured divergence is 1.5. The same analysis applied within a device across seeds isolates batch invariance at the logit level: ’s margins are bitwise identical at 100.000% of 10.0M positions for all three models while also reaches 100% on all three, and unmitigated BF16 on 95 to 99%.

## End-to-End Performance

Figure 4 reports full-pipeline end-to-end wall-clock times. is faster than in all 36 configurations, by 1.17–1.43× on A100 and H100 and by 1.6– 3.1× on L40S. The architecture-dependence is explained by a roofline model: ’s decode reads FP32-width weights while reads BF16-width, so the achievable improvement grows with how bandwidth-bound the GPU is. The A100’s low FP32-compute-to-bandwidth ratio (12.5 FLOP/byte) leaves decode partially FP32-computebound. The L40S sits at 106 FLOP/byte, where its decode is dominated by FP32 weight reads, so reaches and, for the decode-heavy DeepSeek reasoning model, well exceeds the 2× weight-trafic bound. The bound applies to GEMM trafic alone. End-to-end gains can exceed it because also eliminates ’s per-layer cast kernels and transient allocations. Measured speedups increase monotonically with the hardware ratio, exactly as the model predicts. These are conservative, whole-workload numbers: they include prefill (where ’s IEEE-FMA Triton GEMM reaches ∼0.85× cuBLAS throughput) and CPU-side scoring. Further analysis of the performance discrepancy between GPU architectures can be found in the technical supplement (Section A).

![](images/32e02316477bdee0a33dfd1939b6b60b293d696c5641c31cce01dbc850bdc2a5.jpg)  
Figure 4: End-to-end evaluation wall-clock time (mean of 3 seeds with error bars showing the standard deviation) for , , and unmitigated BF16 across three GPU architectures, three models, and four benchmarks. is faster than in every configuration, with the largest gains on the bandwidth-lean L40S, while retaining reproducibility that BF16 lacks.

<table><tr><td colspan="2">Weights (GiB)</td><td>Peak spike (GiB)</td><td>KV tokens</td></tr><tr><td colspan="2">DeepSeek-R1-Distill-Llama-8B</td><td></td><td></td></tr><tr><td>LayerCast</td><td>16.92</td><td>0.445</td><td>70,832</td></tr><tr><td>ReproFuse</td><td>15.94</td><td>0.051</td><td>74,912</td></tr><tr><td colspan="2">Qwen3-14B</td><td></td><td></td></tr><tr><td>LayerCast</td><td>30.41</td><td>0.673</td><td>128,288</td></tr><tr><td>ReproFuse</td><td>28.96</td><td>0.060</td><td>133,536</td></tr></table>

Table 2: Memory profile configured with vLLM’s gpu\_memory\_utilization = 0.9 on models with untied embeddings. “Weights”: resident parameter bytes. “Peak spike”: transient allocation above steady state during a forward pass. “KV tokens”: KV-cache capacity granted by vLLM under the same memory budget, which ’s savings convert directly into serving capacity.

## Memory

Since total GPU memory usage is uninformative as vLLM allocates a fixed fraction for KV cache, Table 2 isolates the three memory components that difer between methods. On untied-embedding models, ’s completed BF16 storage policy saves about 1–1.4 GiB of resident weights.

’s transient FP32 weight copies produce a per-step allocation spike the size of the largest projection matrix (0.4– 0.7 GiB on these models). ’s spike is smaller. Under a fixed memory budget, vLLM converts both savings into additional KV-cache blocks, i.e., 4–7% more KV capacity.

## Accuracy

changes only the order of FP32 additions relative to . Accuracy diferences between the methods are numerical-noise-level. On both Qwen3-4B tasks the scores are identical (93.93% GSM8K, 87.2% MATH500). On Llama-3.2-3B the methods difer by 1–2 problems per benchmark, a diference within the numerical noise that motivates this work. Eliminating that noise, not shifting the mean, is the contribution. Numerical error against an FP64 reference is at the same 10<sup>−7</sup>-relative level for both methods, and ’s split-K actually shortens accumulation chains slightly.

## Conclusion

’s argument covers any hardware with correctly implemented IEEE-754 FMA, which in principle could mitigate nondeterminism on various ML accelerators. Validating bitwise portability across diferent vendors (where software, not just hardware, difer) is an open problem.

LLM inference diverges across GPU architectures because vendor kernels choose architecture-specific reduction orders and unspecified tensor-core arithmetic, and because low-precision rounding places many greedy decisions within flipping distance. showed that FP32 compute can make token flips more rare, but paid FP32 bandwidth for BF16 storage and left the reduction-order channel open. closes that channel: fused-upcast GEMM kernels with IEEE-754 FMA arithmetic, shape-pure configurations, and deterministic split-K make linear-layer outputs bitwise identical across Ampere, Ada, and Hopper, while running 1.17–3.1× faster than end-to-end and freeing weight memory for KV cache. Reproducibility and eficiency are not in tension: the same discipline that fixes the reduction order also eliminates redundant memory trafic.

## References

Atil, B.; Aykent, S.; Chittams, A.; Fu, L.; Passonneau, R. J.; Radclife, E.; Rajagopal, G. R.; Sloan, A.; Tudrej, T.; Ture, F.; Wu, Z.; Xu, L.; and Baldwin, B. 2025. Non-Determinism of "Deterministic" LLM Settings. arXiv:2408.04667.

Badash, E.; Boneh, D.; Komargodski, I.; and Srivastava, M. 2026. Hawkeye: Reproducing GPU-Level Non-Determinism. In Proceedings ofthe 9th Conference on Machine Learning and Systems (MLSys).

Bentley, M.; Briggs, I.; Gopalakrishnan, G.; Ahn, D. H.; Laguna, I.; Lee, G. L.; and Jones, H. E. 2019. Multi-Level Analysis of Compiler-Induced Variability and Performance Tradeofs. In Proceedings of the 28th International Symposium on High-Performance Parallel and Distributed Computing, HPDC ’19, 61–72. New York, NY, USA: Association for Computing Machinery. ISBN 9781450366700.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Fasi, M.; Higham, N. J.; Mikaitis, M.; and Pranesh, S. 2021. Numerical behavior of NVIDIA tensor cores. PeerJ Computer Science, 7: e330.

Frantar, E.; Castro, R. L.; Chen, J.; Hoefler, T.; and Alistarh, D. 2025. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. In Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice ofParallel Programming, PPoPP ’25, 239–251. New York, NY, USA: Association for Computing Machinery. ISBN 9798400714436.

Gond, R.; Kamath, A. K.; Ramjee, R.; and Panwar, A. 2026. LLM-42: Enabling Determinism in LLM Inference with Verified Speculation. arXiv preprint arXiv:2601.17768.

Gundersen, O. E.; and Kjensmo, S. 2018. State of the art: Reproducibility in artificial intelligence. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

He, H.; and Thinking Machines Lab. 2025. Defeating Nondeterminism in LLM Inference. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/defeatingnondeterminism-in-llm-inference/.

IEEE. 2019. IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019 (Revision ofIEEE 754-2008), 1–84.

Khatchadourian, R. 2026. Replayable Financial Agents: A Determinism-Faithfulness Assurance Harness for Tool-Using LLM Agents. arXiv:2601.15322.

Laguna, I. 2020. Varity: Quantifying Floating-Point Variations in HPC Systems Through Randomized Testing. In 2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 622–633.

Li, X.; Li, A.; Fang, B.; Swirydowicz, K.; Laguna, I.; and Gopalakrishnan, G. 2024. FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA & AMD Matrix Accelerators. In 2024 IEEE 24th International Symposium on Cluster, Cloud and Internet Computing (CCGrid), 39–46.

Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2024. Let's Verify Step by Step. In International Conference on Learning Representations.

Lin, J.; Tang, J.; Tang, H.; Yang, S.; Chen, W.-M.; Wang, W.- C.; Xiao, G.; Dang, X.; Gan, C.; and Han, S. 2024. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. In Gibbons, P.; Pekhimenko, G.; and Sa, C. D., eds., Proceedings of Machine Learning and Systems, volume 6, 87–100.

Markidis, S.; Chien, S. W. D.; Laure, E.; Peng, I. B.; and Vetter, J. S. 2018. NVIDIA Tensor Core Programmability, Performance & Precision. In 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 522–531.

Meta. 2024. Llama 3.2 3B Instruct. https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.

Muller, J.-M.; Brunie, N.; de Dinechin, F.; Jeannerod, C.- P.; Joldes, M.; Lefèvre, V.; Melquiond, G.; Revol, N.; and Torres, S. 2018. Handbook of Floating-Point Arithmetic. Springer International Publishing. ISBN 9783319765266.

Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2023. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv:2311.12022.

Shanmugavelu, S.; Taillefumier, M.; Culver, C.; Hernandez, O.; Coletti, M.; and Sedova, A. 2024. Impacts of floatingpoint non-associativity on reproducibility for HPC and deep learning applications. arXiv:2408.05148.

Stosic, D.; and Micikevicius, P. 2021. Accelerating AI Training with NVIDIA TF32 Tensor Cores. https://developer.nvidia.com/blog/accelerating-aitraining-with-tf32-tensor-cores/. NVIDIA Blog.

Sun, W.; Li, A.; Geng, T.; Stuijk, S.; and Corporaal, H. 2023. Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors. IEEE Transactions on Parallel and Distributed Systems, 34(1): 246–261.

Valpey, B.; Li, X.; Pai, S.; and Gopalakrishnan, G. 2025. An SMT Formalization of Mixed-Precision Matrix Multiplication: Modeling Three Generations of Tensor Cores. arXiv:2502.15999.

Xie, P.; Gao, Y.; Wang, Y.; and Xue, J. 2025. Revealing Floating-Point Accumulation Orders in Software/Hardware Implementations. arXiv:2411.00442.

Yuan, J.; Li, H.; Ding, X.; Xie, W.; Li, Y.-J.; Zhao, W.; Wan, K.; Shi, J.; Hu, X.; and Liu, Z. 2025. Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Zahid, A. H.; Laguna, I.; and Le, W. 2024. Testing GPU Numerics: Finding Numerical Diferences Between NVIDIA and AMD GPUs. In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, 547–557.

Zhang, Y.; and Math-AI, T. 2024. American Invitational Mathematics Examination (AIME) 2024.

Zhuang, D.; Zhang, X.; Song, S.; and Hooker, S. 2022. Randomness in neural network training: Characterizing the impact of tooling. Proceedings ofMachine Learning and Systems, 4: 316–336.

## Technical Supplement

## A. Why the L40S Results Outperform the H100 Results under

On our decode-dominated workloads runs faster on the NVIDIA L40S than on the NVIDIA H100 for most layers. This is not an anomaly. It follows directly from the reproducibility mechanism of combined with the arithmetic profile of autoregressive decoding. We explain it in three steps: the compute path that is restricted to, the arithmetic intensity of decode, and the resulting utilization gap between the two GPUs.

## is accelerated by FP32 vector units, not tensor cores

To produce bitwise-identical results across GPU architectures, performs every linear-layer accumulation with IEEE-754 single-precision fused multiply-add on the CUDA (vector) cores. It never dispatches to the tensor cores, whose accumulation order and internal rounding are unspecified and vary across GPU generations. The hardware peak relevant to is therefore the FP32 vector throughput, not the tensor-core throughput that dominates vendor comparisons. Table 3 lists this peak for the three GPUs we study, together with the memory bandwidth and the resulting roofline ridge point, the arithmetic intensity above which a kernel becomes compute-bound. Two facts stand out. First, the L40S has the highest FP32 vector throughput of the three, because each of its Ada streaming multiprocessors exposes 128 FP32 lanes at a high clock, whereas the A100 exposes 64. Excluding the tensor cores therefore removes the H100’s principal advantage. Second, the ridge points span nearly an order of magnitude: a kernel must reach 106 FLOP/byte to be compute-bound on the L40S, but only 25.6 on the H100 and 12.5 on the A100.

## Decode runs at low arithmetic intensity

Consider a linear layer that multiplies a batch of M token vectors by a weight matrix of shape $N \times K$ stored in BF16. The layer performs 2MNK floating-point operations and reads 2NK bytes of weights, which dominate the memory trafic when M is small. Its arithmetic intensity is therefore

$$
\mathrm { A I } \ = \ { \frac { 2 M N K } { 2 N K } } \ = \ M \mathrm { F L O P / b y t e } .
$$

During decode, M equals the number of sequences generated concurrently, which is small. Our measurements use $M =$ 32. At this intensity the layer lies below the ridge point of every GPU in Table 3. In the ideal roofline limit each GPU would then be bound by weight-read bandwidth or by peak FP32, whichever is smaller. That ideal limit favors the H100, whose bandwidth is more than twice that of the L40S. The measured ordering is the reverse, which shows that neither GPU operates near its roofline and that peak numbers alone do not determine the outcome.

## Workload Size

The binding constraint is occupancy. A decode GEMM at M = 32 produces few output tiles, and with the tensor cores disabled each tile is evaluated as a stream of scalar FP32 fused multiply-adds. The L40S has fewer streaming multiprocessors but high per-SM FP32 throughput, so it keeps its execution units busy on these small problems and sustains a rate close to what the problem allows. The H100 has a larger SM array and more bandwidth, but a small GEMM does not expose enough parallel work to fill $\mathbf { i t } ,$ so much of the device stays idle and its efective throughput falls well below its peak. Table 4 reports per-layer decode latency and achieved FP32 throughput for a single Llama-3.2-3B forward at $M = 3 2$ . The L40S matches or exceeds the H100 on four of the five projections, and both GPUs realize 11 to 18 TFLOP/s. The one exception is the language-model head, whose output dimension $\bar { N } = 1 2 8 2 5 6$ is large enough to generate the tiles needed to fill the H100 SM array; there the H100 is faster. This exception supports the explanation rather than contradicting it: the H100 wins on the single layer wide enough to use it.

![](images/f5366ec865e749b4feb8b01a16d4baf24c9d5e9ec0339994721bd09fd85b9f10.jpg)  
FP32 compute / memory bandwidth (FLOP/byte)  
Figure 5: End-to-end speedup over against each GPU’s FP32-compute-to-bandwidth ratio (peak FP32 TFLOPS ÷ memory bandwidth). Small points are the four (model, benchmark) configurations and large points are means.

## Scope of the efect

The inversion is specific to the low-intensity, low-occupancy regime of small-batch decode. As the batch grows, the arithmetic intensity M rises, the number of output tiles increases, and the H100 SM array fills. The same holds during prefill, where M spans the full prompt length. In those regimes the H100 recovers its expected advantage. The behavior described here should therefore be read as a property of memory- and occupancy-limited decode under a tensor-corefree reproducibility constraint, not as a general ordering of the two GPUs.

## B. Divergence Metrics and Div\_Index Results

The determinism table in the main paper reports two percentages per (model, benchmark, method) cell: a cross-GPU divergence rate and a cross-seed divergence rate. This section states precisely how those percentages are computed, and then reports the same experiments under the Div\_Index metric of (Yuan et al. 2025), which records when a divergence first occurs rather than whether it occurs.

<table><tr><td>GPU</td><td>FP32 vector (TFLOP/s)</td><td>FP32 lanes/SM</td><td>SMs</td><td>HBM BW (GB/s)</td><td>Ridge (FLOP/byte)</td></tr><tr><td>A100</td><td>19.5</td><td>64</td><td>108</td><td>1555</td><td>12.5</td></tr><tr><td>L40S</td><td>91.6</td><td>128</td><td>142</td><td>864</td><td>106.0</td></tr><tr><td>H100</td><td>51.2</td><td>128</td><td>114</td><td>2000</td><td>25.6</td></tr></table>

Table 3: FP32 vector (non-tensor-core) compute peak, memory bandwidth, and roofline ridge point for the three GPUs. Because never uses the tensor cores, the FP32 vector peak is the relevant compute bound, and the L40S has the highest of the three.
<table><tr><td>Projection  $( N \times K )$ </td><td>L40S  $( \mu \mathrm { s } )$ </td><td> $\operatorname { H 1 0 0 } \left( \mu \mathbf { s } \right)$ </td><td>L40S (TFLOP/s)</td><td>H100 (TFLOP/s)</td></tr><tr><td> ${ \mathrm { q k v } } \left( 5 1 2 0 \times 3 0 7 2 \right)$ </td><td>57.2</td><td>70.6</td><td>17.6</td><td>14.3</td></tr><tr><td> $\hat { \textbf { o } } ( 3 0 7 2 \times 3 0 7 2 )$ </td><td>56.7</td><td>57.2</td><td>10.7</td><td>10.6</td></tr><tr><td> $\mathrm { g a t e / u p ( 1 6 3 8 4 \times 3 0 7 2 ) }$ </td><td>191.4</td><td>205.1</td><td>16.8</td><td>15.7</td></tr><tr><td>down  $\overline { { ( 3 0 7 2 \times 8 1 9 2 ) } }$ </td><td>88.8</td><td>132.1</td><td>18.1</td><td>12.2</td></tr><tr><td>lm head  $( 1 2 8 2 5 6 \times 3 0 7 2 )$ </td><td>1701.5</td><td>1532.7</td><td>14.8</td><td>16.5</td></tr></table>

Table 4: Per-layer decode latency and achieved FP32 throughput for a single Llama-3.2-3B forward at batch $M = 3 2$ , measured per GPU under . The L40S is faster on four of five projections. The H100 leads only on the language-model head, the one layer wide enough to fill its SM array.

## How the divergence percentages are computed

Every metric is built from one primitive: a comparison of two greedy token streams. Fix a model, a benchmark, and a method (unmitigated BF16, , or ). For a problem p decoded on GPU A we write the emitted greedy token stream as $y ^ { A } ( p ) = ( y _ { 0 } ^ { A } , y _ { 1 } ^ { A } , \dots )$ , taken from column 0 of the stored top-5 token dump. For an ordered pair of runs on GPUs A and B at the same seed, the first-divergence index is

$$
d ( p ; A , B ) = \operatorname* { m i n } \{ t : y _ { t } ^ { A } ( p ) \neq y _ { t } ^ { B } ( p ) \} ,
$$

with $d = \infty$ when the two streams are identical over their common length and have equal length. Because a single flipped token permanently forks the two sequences, d is the only event that matters: everything after it is a comparison between two already-diferent generations.

Cross-GPU divergence (the Div\_Percent analog). Hold the seed fixed and vary the GPU. A problem is cross-GPU divergent if $d ( p ; A , B ) < \infty$ for at least one architecture pair $\{ A , \bar { B } \} \subseteq \{ \bar { \mathrm { A } } 1 0 0 , \mathrm { L } 4 0 \mathrm { S } , \mathrm { H } 1 0 0 \}$ at any matched seed. The cross-GPU divergence rate is

$$
\displaystyle \mathrm { D i v } \% = \frac { \# \{ p : p \mathrm { i s } \mathrm { c r o s s . } \mathrm { G P U } \mathrm { d i v e r g e n t } \} } { \# \{ p \} } ,
$$

the fraction of problems whose greedy output is not identical on every architecture. This is the per-problem quantity reported in the main paper, and it is the direct counterpart of the Div\_Percent metric of , which reports the percentage of problems whose outputs difer across its 12 runtime configurations. We aggregate at the problem level rather than the pair level because a problem that forks on any pair is not reproducible for a user, independent of how many of the pairs reproduce it. A pair-level rate is also available in our released summary and is uniformly smaller.

Cross-seed divergence. Hold the GPU fixed and vary the seed. The cross-seed rate is the fraction of same-GPU, sameproblem seed-pair comparisons for which $d < \infty$ . Greedy decoding is seed-independent by definition, so this rate is 0 in the ideal case, and any nonzero value is pure run-to-run nondeterminism from batch composition and scheduling rather than from the numerics of a single forward pass. is batch-invariant by construction (rule R4 in the main paper), so its cross-seed rate is exactly 0 in every cell.

## Div\_Index

summarizes the same forking process from the other side. Its Div\_Index is the token index at which responses that were identical up to that point first disagree, averaged over the divergent examples of a dataset, with the convention that a higher Div\_Index means divergence is pushed later and the outputs are therefore more consistent ( writes −1 for a dataset in which nothing diverges). We report the direct analog: the mean first-divergence index

$$
\mathrm { D i v \_ I n d e x \ = \atop \{ p , A , B \} : \atop d ( p ; A , B ) < \infty } d ( p ; A , B ) ,
$$

averaged over every divergent cross-GPU comparison at matched seed. The two metrics answer complementary questions on the same event: Div% counts how often a fork happens, and Div\_Index measures how deep into the generation it happens.

Table 5 pairs the two for all twelve cells. Three patterns hold. First, unmitigated BF16 forks early, with a first divergence at a mean ofonly 65 to 208 tokens: on greedy reasoning traces thousands of tokens long, the architectures part ways almost at the start. Second, FP32 compute both lowers Div% and pushes Div\_Index far later, to as much as 3551 tokens for on DeepSeek GPQA-Diamond, which is the same statement as its low Div% seen through the timing lens. The surviving forks are not only rarer but also deferred. Third, removes the linear-layer fork outright: in every cell whose divergence originates in the GEMMs there is no divergent cross-GPU pair at all, so Div\_Index is undefined (shown as a dash), the strongest possible reading of the metric. The only finite entry is Qwen3-4B on GSM8K (671 tokens), a single problem whose residual originates in the unpinned attention kernel rather than the linear layers.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Benchmark</td><td colspan="2">Unmitigated BF16</td><td colspan="2">LayerCast</td><td colspan="2">ReproFuse</td></tr><tr><td>Div%</td><td>Div_Index</td><td>Div%</td><td>Div_Index</td><td>Div%</td><td>Div_Index</td></tr><tr><td rowspan="4">Llama-3.2-3B</td><td>GSM8K</td><td>66.6</td><td>65</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>MATH500</td><td>79.6</td><td>106</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>AIME24</td><td>86.7</td><td>153</td><td>0</td><td>一</td><td>0</td><td></td></tr><tr><td>GPQA-Diamond</td><td>30.8</td><td>114</td><td>0</td><td>一</td><td>0</td><td></td></tr><tr><td rowspan="4">Qwen3-4B</td><td>GSM8K</td><td>71.3</td><td>99</td><td>0.23</td><td>436</td><td>0.08</td><td>671</td></tr><tr><td>MATH500</td><td>83.4</td><td>204</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>AIME24</td><td>100</td><td>204</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>GPQA-Diamond</td><td>99.5</td><td>131</td><td>0.51</td><td>493</td><td>0</td><td></td></tr><tr><td rowspan="4">DeepSeek-R1-Distill-8B</td><td>GSM8K</td><td>88.9</td><td>104</td><td>0.15</td><td>187</td><td>0</td><td></td></tr><tr><td>MATH500</td><td>93.0</td><td>189</td><td>0.40</td><td>534</td><td>0</td><td></td></tr><tr><td>AIME24</td><td>100</td><td>208</td><td>0</td><td></td><td>0</td><td></td></tr><tr><td>GPQA-Diamond</td><td>100</td><td>145</td><td>0.51</td><td>3551</td><td>0</td><td></td></tr></table>

Table 5: Divergence percentage (Div%, the fraction of problems that fork across at least one GPU pair) alongside Div\_Index (the mean first-divergence token index over divergent cross-GPU pairs, higher is better) for all three methods. A dash marks a cell with no divergent cross-GPU pair, which is the ideal outcome. Div% falls and Div\_Index rises together as numeric improve, from left to right. For the sub-1% cells the Div\_Index mean rests on few divergent pairs (as few as one distinct problem on the GPQA-Diamond cells), so those entries are small-sample estimates.

## C. Ablation of the Design Rules R1–R4

The four design rules ofthe main paper each forbid something that a performance-first GEMM implementation would do. This section measures, one rule at a time, both what the rule buys and what it costs.

Method. We use a copy of the kernel in which each rule is a parameter rather than a constant, verified bitwise identical to the shipped kernel when all four rules are on. Exactly one rule is switched of per experiment with everything else held fixed. The test set is nine GEMM cases: the QKV, down, output and lm\_head projections of the evaluated models at the decode batch size of 32, two prefill projections at a chunk of 2048 rows, and one deliberately awkward 100×1000×1000 shape that exercises the masked code paths. Inputs are drawn from a CPU generator with a fixed seed and copied to the device, so every GPU is fed identical input bits and any later diference was introduced by the hardware or by the ablated rule. The ragged shape is not one any of the models serves, so it is reported separately in the timing summaries.

## R1: IEEE FMA only

Table 6 varies only the input\_precision argument of every tl.dot.

Under R1 every case is bitwise identical on all three architectures. Under TF32 tensor cores no case is. The failure pattern is instructive: A100 and L40S agree with each other on all nine cases, and both disagree with H100 on all nine.

<table><tr><td>Setting</td><td>Bitwise all GPUs</td><td>Worst cross-GPU</td><td>Error vs. FP64</td><td>Time vs. R1</td></tr><tr><td>ieee (R1)</td><td>9/9</td><td>一</td><td> $4 . 8 \times 1 0 ^ { - 6 }$ </td><td>1.00</td></tr><tr><td>tf32</td><td>0/9</td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td><td> $7 . 6 \times 1 0 ^ { - 4 }$ </td><td>0.42</td></tr><tr><td>tf32x3</td><td>0/9</td><td> $2 . 4 \times 1 0 ^ { - 7 }$ </td><td> $9 . 3 \times 1 0 ^ { - 7 }$ </td><td>1.45</td></tr><tr><td>omitted</td><td>0/9</td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td><td> $7 . 6 \times 1 0 ^ { - 4 }$ </td><td>0.42</td></tr></table>

Table 6: R1 ablation over nine GEMM cases on A100, L40S and H100. “Bitwise across $\mathrm { G P U s } ^ { \mathbf { \ ' } }$ counts cases whose output is identical on all three architectures. Diferences are scaled by the output magnitude. The last column is the median decode-shape time relative to ieee, so values below 1 mean the ablated setting is faster.

Ampere and Ada issue the same TF32 MMA instruction, while Hopper issues a diferent one, so the agreement between two of the three GPUs is a consequence of two architectures happening to share an instruction rather than of any guarantee. This is the same coincidental-agreement pattern the main paper reports for cuBLAS kernel selection, and it is exactly what a specification-level argument is meant to replace.

The accuracy column shows that the ablation also gives up what the FP32 pipeline was for. TF32 carries ten mantissa bits, and its error against an FP64 reference is 4.0 to $7 . 6 \times 1 0 ^ { - 4 }$ of the output scale, three orders of magnitude above IEEE FMA’s $4 . { \overset { \cdot } { 8 } } \times 1 0 ^ { - 6 }$ and of the same order as BF16 rounding. A hybrid-precision pipeline that computes in TF32 is therefore not computing in FP32 in any sense that matters. The three-pass tf32x3 variant recovers the accuracy but not the reproducibility, since it is built from the same architecture-dependent MMA.

R1 is not free at the kernel level: IEEE FMA runs on the CUDA cores, so the GEMMs are 2.4× slower than TF32 at the median decode shape and 5.6× slower at the median prefill shape. That cost is measured against an alternative that fails both goals of the pipeline, and it is not a cost against our baseline: ’s cuBLAS SGEMM path also executes on the FP32 units, so the end-to-end comparison in the main paper is between two FP32 CUDA-core implementations, and is the faster of the two.

<table><tr><td rowspan="2">Shape</td><td rowspan="2"> $( M , N , K )$ </td><td colspan="3">Configuration selected by autotuning  $( { \cal B } _ { M } , { \cal B } _ { N } , { \cal B } _ { K } ) , S$ </td><td rowspan="2">Bitwise across GPUs pinned / autotuned</td></tr><tr><td>A100</td><td>L40S</td><td>H100</td></tr><tr><td>Decode QKV, 8B</td><td>(32, 6144, 4096)</td><td>(32, 128, 64), 8</td><td>(32, 128, 64), 8</td><td>(32, 256, 32), 8</td><td>yes / yes</td></tr><tr><td>Decode down, 8B</td><td>(32, 4096, 14336)</td><td>(32, 128, 64), 8</td><td>(32, 128, 64), 8</td><td>(32, 128, 64), 8</td><td>yes / yes</td></tr><tr><td>Decode QKV, Qwen3-4B</td><td>(32, 6144, 2560)</td><td>(32, 128, 64), 8</td><td>(32, 128, 64), 8</td><td>(32, 256, 32), 8</td><td>yes / yes</td></tr><tr><td>Decode down, Qwen3-4B</td><td>(32, 2560, 9728)</td><td>(32, 64, 64), 8</td><td>(32, 64, 64), 8</td><td>(32, 128, 64), 8</td><td>yes / yes</td></tr><tr><td>Decode output proj.</td><td>(32, 4096, 4096)</td><td>(32, 128, 32), 8</td><td>(32, 128, 64), 4</td><td>(32, 128, 32), 8</td><td>yes / no</td></tr><tr><td>Decode 1m_head, 3B</td><td>(32, 128256, 3072)</td><td>(32, 128, 64), 2</td><td>(32, 128, 64), 2</td><td>(32, 128, 64), 1</td><td>yes / no</td></tr><tr><td>Prefill QKV, 8B</td><td>(2048, 6144, 4096)</td><td>(128, 256, 32), 1</td><td>(128, 256, 32), 1</td><td>(128, 256, 32), 1</td><td>yes / yes</td></tr><tr><td>Prefill down, 8B</td><td>(2048, 4096, 14336)</td><td>(128, 128, 32), 1</td><td>(128, 256, 32), 1</td><td>(128, 256, 32), 1</td><td>yes / yes</td></tr><tr><td>Ragged probe</td><td>(100, 1000, 1000)</td><td>(16, 64, 32), 1</td><td>(16, 64, 32), 1</td><td>(16, 64, 32), 1</td><td>yes / yes</td></tr></table>

Table 7: R2 ablation. Each architecture runs the identical search over the identical candidate space. The last column compares the output of the pinned configuration across the three GPUs, and then the output of each GPU’s own autotuned choice across the three GPUs.

Finally, the last row substantiates the claim that framework-level switches do not reach Triton. Every node in this experiment ran with torch.backends.cuda.matmul.allow\_tf32 = False. Simply omitting the input\_precision argument reproduced the TF32 result bit for bit on 9/9 cases on all three GPUs, and never matched the IEEE result. The constraint has to be applied per operation.

## R2: No autotuning

We replace the pinned configuration with a real autotuner: the same candidate space of 18 tile configurations crossed with split factors $S \in \overline { { \{ 1 , 2 , 4 , 8 \} } }$ , benchmarked on the local device, run identically on each architecture. we replace the pinned configuration with a benchmark-and-select search of the kind triton.autotune implements. The candidate space is filtered by one shared-memory bound for all three GPUs, so a diferent winner reflects a diferent choice and not a diferent menu. Table 7 lists what each device picks.

The same configuration wins on all three architectures for only 3 of the 9 shapes. Autotuning is not a function of just the device. Repeating the identical search three times on one node changed the winner on 1 of 9 shapes on the A100, because two near-equal candidates trade places under measurement noise.

That these choices are numerical choices, and not merely scheduling choices, is visible on a single device. Within the decode candidate space the 72 candidates produce 4 distinct outputs, and on 4 of the 27 (shape, GPU) cells the autotuner’s pick already disagrees bitwise with the pinned configuration on that same GPU. Composed across architectures, the pinned configuration is bitwise identical on $9 / 9$ shapes while the autotuned one is identical on $7 / 9$

One nuance is worth stating precisely, because it narrows where R2 does its work. With R1 and R3 in force, the accumulation over the reduction dimension is strictly ascending in $k ,$ so $B _ { M } , B _ { N }$ , the warp count and the pipeline depth are numerically inert: they change the schedule but not the reduction tree. The numerical content of the tuner’s choice therefore flows through the split factor, and through $B _ { K }$ where it changes the segmentation. R2 remains necessary for three reasons. The split factor is a tuned knob in every practical Triton GEMM and is precisely the one that moves the bits; the inertness of the remaining knobs is an observed property of one Triton version’s lowering rather than a specification, and pinning is what makes reproducibility independent of it. An autotuner optimizes time, so given the choice it removes R1 as well, selecting TF32 on 26 of the 27 (shape, GPU) cells.

Pinning does not incur significant cost over the entire LLM pipeline. Over the eight served shapes the pinned configuration takes 1.4% longer than the best found configuration at the median and 21.9% longer at worst, the worst case being the lm\_head shape on the L40S. The synthetic ragged probe is the one outlier, at 29 to 70×: a 100 × 1000 × 1000 problem gives the pinned 128 × 128 prefill tile only eight output tiles to spread over more than a hundred SMs. No served prefill chunk is that small, but the case does mark the boundary of where a single pinned prefill configuration is appropriate.

## R3: Deterministic split-K

R3 makes two commitments, and we ablate them separately.

Atomics. Table 9 repeats each variant 50 times on one device. Classic split-K returned 50 diferent results in 50 runs on 15 of the 18 (shape, GPU) cells, with run-to-run spreads up to $2 . 2 \times 1 0 ^ { - 7 }$ of the output scale, the same order as the cross-architecture diferences the main paper measures for and therefore large enough to flip a token at a narrow decision margin. The three exceptions are the lm\_head shape, where the merge order happened to be stable on all three GPUs across all 50 runs, which is the point rather than a counterexample: the order is a race, and whether it bites is a property of the schedule and not of the program. Ordered split-K and no split each returned exactly one result in every run on every GPU, and both are bitwise identical across all three architectures on all six shapes, while atomic split-K is identical across architectures on one.

Table 8: Hardware specifications of the evaluation machines.
<table><tr><td>Component</td><td>L40S</td><td>H100</td><td>A100</td></tr><tr><td>GPU</td><td></td><td></td><td></td></tr><tr><td>Model</td><td>NVIDIA L40S PCIe (SM89)</td><td>NVIDIA H100 PCIe (SM90)</td><td>NVIDIA A100 PCIe 40GB (SM80)</td></tr><tr><td>Count</td><td>1×</td><td>1×</td><td>1×</td></tr><tr><td>VRAM</td><td>45 GiB</td><td>80 GiB</td><td>40 GiB</td></tr><tr><td>CPU</td><td></td><td></td><td></td></tr><tr><td>Model</td><td>Intel(R) Xeon(R) CPU Max 9468 2</td><td>Intel(R) Xeon(R) Gold 6454S</td><td>Intel(R) Xeon(R) Gold 6454S</td></tr><tr><td>Sockets</td><td></td><td>2</td><td>2</td></tr><tr><td>Cores / Threads</td><td>96 / 192</td><td>64 / 128</td><td>64 / 64</td></tr><tr><td>Max. Freq.</td><td>2.1 GHz</td><td>3.4 GHz</td><td>3.4 GHz</td></tr><tr><td>Memory</td><td></td><td></td><td></td></tr><tr><td>System RAM</td><td>503 GiB</td><td>503 GiB</td><td>503 GiB</td></tr></table>

<table><tr><td rowspan="2">Decode shape</td><td colspan="3">Distinct outputs, 50 runs</td><td rowspan="2">Ordered</td></tr><tr><td>none</td><td>ordered</td><td>atomic</td></tr><tr><td>QKV, 8B</td><td>1</td><td>1</td><td>50</td><td>1.9-4.1×</td></tr><tr><td>Down, 8B</td><td>1</td><td>1</td><td>50</td><td>2.8-4.4×</td></tr><tr><td>QKV, Qwen3-4B</td><td>1</td><td>1</td><td>50</td><td>1.9-2.6×</td></tr><tr><td>Down, Qwen3-4B</td><td>1</td><td>1</td><td>50</td><td>3.7-5.6×</td></tr><tr><td>Output proj.</td><td>1</td><td>1</td><td>50</td><td>2.1-3.1×</td></tr><tr><td>1m_head,3B</td><td>1</td><td>1</td><td>1</td><td>0.8-1.0×</td></tr></table>

Table 9: R3 ablation on the six decode shapes, run-to-run on a fixed device. Counts are identical on A100, L40S and H100. The last column is the speedup of ordered split-K over no split, ranged over the three GPUs.

Split-K is not optional for the decode shapes: the ordered version is 1.9 to 5.6× faster than no split on the five projection shapes. Only lm\_head, whose N is already large enough to fill the machine, prefers no split. Ordering the reduction is free: the ordered variant is 4.2% faster than the atomic one at the median, since writing S partials and summing them in a second pass costs no more than contending on atomics.

The split factor must be shape-pure. Sweeping S ∈ {1, 2, 4, 8, 16} produces five distinct outputs on all six shapes, so the split factor fully determines the result. A performance-first rule that sizes the split by wave quantization, which is how CUTLASS-style GEMMs choose their slices, selects S = 2, 8 and 16 on A100, L40S and H100 for the same decode QKV shape, because the three devices have 108, 142 and 114 SMs. Its output consequently difers across architectures on four of the six shapes. Making S a function of K alone is what removes the SM count from the numerics.

## R4: Batch invariance by construction

We compute row 0 of the same input at batch sizes $M =$ 1 to 64 and compare its bits against the M = 1 result, under four policies: , a variant whose split factor is occupancy-sized, a variant whose configuration and split are re-searched by measurement at every batch size as a Triton autotuner keyed on M would do, and cuBLAS.

<table><tr><td rowspan="2">GPU</td><td colspan="4">Shapes with row 0 independent of M (of 6)</td></tr><tr><td>ReproFuse</td><td>occupancy</td><td>per-M tuned</td><td>cuBLAS</td></tr><tr><td>A100</td><td>6</td><td>1</td><td>2</td><td>0</td></tr><tr><td>L40S</td><td>6</td><td>1</td><td>1</td><td>0</td></tr><tr><td>H100</td><td>6</td><td>2</td><td>1</td><td>0</td></tr></table>

Table 10: R4 ablation over twelve batch sizes in $1 \leq M \leq 6 4 .$ A shape counts only if row 0 is bitwise identical at every one of the twelve. Under cuBLAS, row 0 difers from the M = 1 result at eleven of the twelve batch sizes on every shape and every GPU.

is batch-invariant on all 18 (shape, GPU) cells. The occupancy-sized and per-M tuned variants are invariant on 4 of 18 each, and cuBLAS on none, difering from the single-request result at eleven of the twelve batch sizes everywhere. The two ablated policies also lose cross-architecture agreement: comparing row 0 at M = 32 across the three GPUs, matches on all six shapes while the ablated policies match on two or fewer. This is the same mechanism seen in R3, since a batch-dependent tile count feeds a device-dependent split factor.

R4 is cheap but not free. Re-tuning at every batch size is 5 to 25% faster than the shape-pure policy at $M \leq 1 6$ on the A100 and H100, 2 to 9% faster at $M = 3 2 ,$ and within noise on the L40S. Buying per-request independence from co-scheduled load therefore costs a few percent of GEMM time at the batch size used throughout the paper, which the end-to-end results absorb.

R4’s claim is stated for the decode bucket, $M \leq 6 4 .$ . row 0 changes at M = 65 under every policy including , because that crosses the documented shape-bucket boundary into the prefill configuration. Immediately above that boundary the pinned prefill configuration is poorly matched to the shape, and a tuned kernel is roughly 3× faster at M = 65 to 128.