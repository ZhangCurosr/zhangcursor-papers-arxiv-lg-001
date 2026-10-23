# Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon

Kyrylo Yemets<sup>1,2</sup> Andrii Yaroshevych<sup>1</sup> Victor Muryn<sup>1</sup> Ostap Khomenko<sup>1</sup> Pavlo Kryven<sup>1</sup> Maksym Kmet<sup>1</sup> Maksym Shamrai<sup>1,</sup> <sup>3</sup> Mariya Hirna<sup>1</sup>

<sup>1</sup>MacPaw AI&Research, Kyiv, Ukraine

<sup>2</sup>Department of Artificial Intelligence, Lviv Polytechnic National University, Lviv, Ukraine <sup>3</sup> Institute of Mathematics of NAS of Ukraine, Kyiv, Ukraine

{ruiite, andrii.yaroshevych, victormuryn, ostap.khomenko, pavlo.kryven, maxkmet, mshamrai, maryhirna}@macpaw.com

## Abstract

Apple Silicon is among the most widely used platforms for local LLM inference, yet nearly all inference optimizations are designed and validated on server-class NVIDIA GPUs under batched serving workloads. Which optimizations retain their benefits when moved to Apple Silicon, and what determines their applicability? We evaluate four optimization families: KV-cache pruning, speculative decoding, ported GPU kernels, and post-training quantization. Experiments span four Apple machines under a common single-request decode workload, with matched RTX 4090 comparisons where possible. KV-cache pruning can reach up to 1.75× end-to-end throughput on Apple but can reduce throughput on desktop NVIDIA GPUs. Under the common benchmark, all evaluated speculative decoding methods reduce throughput on Apple, despite reported 2–6× gains on server GPUs. CUDA kernel designs transfer only partially: public Metal lacks asynchronous copy, and batch-one decode makes several serving-oriented mechanisms inapplicable, although transferable components yield up to 1.20× decode throughput. Quantization gains vary by machine. Across four families, a reported speedup is not a portable property of a technique alone but a joint function of technique, runtime, hardware contract, and workload regime.

## 1 Introduction

Local LLM inference increasingly happens on Apple machines: llama.cpp treats M-series chips as a first-class target [1], Apple ships MLX for unified-memory inference [2], and recent studies describe M-series unified memory as the practical consumer path to large open-weight models [3]. The optimizations that would make this inference fast, however, carry no evidence for it: none of the methods we study — KV-cache management [4], speculative decoding [5], the FlashAttention/FlashInfer, Marlin, and MegaBlocks kernels [6; 7; 8; 9], and post-training quantization [10] — reports a measurement on Apple hardware in its original publication.

An optimization that reduces memory traffic or computation may still fail to improve end-to-end throughput if it targets a secondary cost, adds runtime overhead, or relies on hardware features or workload conditions that are absent from local inference. This motivates our central question: Which LLM inference optimizations remain effective on Apple Silicon, and why?

Our experiments focus on the single-request, unified-memory regime that exists at consumer scale, and we evaluate the bundle users actually deploy: hardware, runtime, and workload together. We cover four Apple devices across three GPU generations, dense and hybrid architectures, and matched RTX 4090 controls where configurations can be aligned.

Prior Apple-Silicon work has mainly evaluated models or complete software stacks. MLX provides a framework for unified-memory inference [2], OpenELM reports local generation with MLX [11], and Rajesh et al. [12] compare several runtimes on an M2 Ultra. That study establishes performance trade offs among Apple-Silicon devices. In contrast, we isolate individual optimizations, port representative CUDA mechanisms to Metal/MLX, and measure their kernel-level and end-to-end effects.

Contributions. (1) A controlled comparison of four optimization families for local inference on Apple Silicon. (2) Metal/MLX ports of representative CUDA attention, quantized-matmul, and MoE mechanisms, evaluated at both kernel and end-to-end levels. (3) A cross-platform analysis explaining when gains persist, disappear, or reverse because of hardware, runtime, and workload differences.

## 2 Methods and Results

Table 1 summarizes the four optimization families evaluated in this paper: the techniques tested within each, the Apple Silicon generations used, and the key finding. The subsections that follow detail the experimental setup, results, and causal analysis for each area.

Table 1: Overview of the four optimization families evaluated in this paper. NVIDIA RTX 4090 is used as a matched control where noted.
<table><tr><td>Optimization</td><td>Methods</td><td>Devices</td><td>Key Finding</td></tr><tr><td>KV Cache</td><td>Prefix reuse [13]; KVPress [14] pruning (ExpectedAttention [15], FastKVzip [16], and 4 others)</td><td>M2 Pro, M3 Ultra, M4 Pro; RTX 4090</td><td>Prefix reuse: —57% wall time at turn 9. Pruning: speedup of up to 1.75× on Apple Silicon vs up to 1.02× on RTX 4090.</td></tr><tr><td>Speculative Decoding</td><td>Classic [5], N-gram, Cascade-n-gram [17], EAGLE-3 [18], Ultra DFlash [19], DDTree [20]</td><td>M2 Pro, M3</td><td>No method achieved a net speedup. Reported 2–6× gains on NVIDIA GPUs do not generally transfer to</td></tr><tr><td></td><td>Metal Kernels Marlin [8] (W4A16 GEMM), FlashInfer [7] (split-KV attention), MoE Gather [9]</td><td>M2, M2 Pro, M3 Ultra, M4 Pro; RTX 4090</td><td>different Apple GPUs. MoE Gather: +3–12% universally. FlashInfer: +20% on M4 Pro, -3% on M2. Marlin: effective at batch &gt;1, negligible at M=1.</td></tr><tr><td>Quantization</td><td>GGUF/llama.cpp; MLX [2] (affine, DWQ); RTN, QuaRot+RTN [21], QuaRot+GPTQ [10]</td><td>M2 Pro, M3</td><td>MLX native affine/DWQ is the best Ultra; RTX 4090 combined trade-off. QuaRot does not pay off on Apple Silicon without low-precision activation kernels.</td></tr></table>

## 2.1 KV Cache

KV caching reduces autoregressive decoding cost by retaining the keys and values computed for preceding tokens. This avoids repeated computation but introduces memory-footprint and bandwidth costs that increase with context length [22]. We characterize cache growth and evaluate the latency and memory effects of persistent prefix reuse and post-prefill pruning. Evaluating pruning-induced changes in output quality is outside our scope; these effects are expected to be comparable across platforms using identical pruning algorithms and configurations.

## 2.1.1 Experimental setup

Cache footprints are derived from the released configurations of mlx-community/Qwen3-4B-4bit, mlx-community/gemma-3-4b-it-4bit, mlx-community/granite-4.0-h-micro-4bit, and LiquidAI/LFM2.5-2.6B, together with the MLX-LM cache layout [13]. Cached tensors are assumed to use bfloat16, independently of checkpoint weight precision. Pruning experiments use Qwen3-4B-Instruct-2507 and LFM2.5-2.6B in bfloat16 through Transformers and KVPress [23; 14]. Qwen is evaluated on M2 Pro, M3 Ultra, M4 Pro, and NVIDIA RTX 4090; LFM is evaluated on the three Apple devices. The matched workloads use context lengths of 512 and 4,096 tokens and output lengths of 1, 16, 128, and 256 tokens at batch size one.

Timings are medians of three measured runs after one warm-up run. We define speedup as $S =$ $T _ { \mathrm { u n p r u n e d } } / T _ { \mathrm { p r u n e d } }$ , using the baseline for the same device and workload; values above one indicate lower end-to-end latency. These ratios only compare the effectiveness of pruning and are not indicative of absolute throughput. Peak memory is reported on M4 Pro, whose latency trends are broadly representative of the Apple measurements. Positive memory savings indicate a lower process-level peak than the unpruned baseline.

The experiments include SnapKV, KNorm, StreamingLLM, and TOVA methods at a compression ratio of 0.5 and ExpectedAttention at 0.75 [24; 25; 26; 27; 15]. The ratio denotes the fraction of cached entries removed. FastKVzip at 0.75 is included for Qwen but requires pretrained modelspecific gates unavailable for LFM2.5-2.6B [16]. LFM2.5-8B-A1B is excluded because the KVPress integration does not support its Lfm2MoeForCausalLM architecture; its bfloat16 weights also exceed the available memory on M2 Pro, precluding a matched three-device comparison [28].

## 2.1.2 Cache growth and persistent prefix reuse

The persistent cache footprint depends on attention architecture and retention policy rather than parameter count alone. At batch size one, it can be expressed as

$$
M _ { \mathrm { c a c h e } } ( n ) = 2 b \sum _ { \ell \in \cal { A } } h _ { \ell } ^ { \mathrm { K V } } d _ { \ell } \operatorname* { m i n } ( n , w _ { \ell } ) + M _ { \mathrm { f i x e d } } ,\tag{1}
$$

where n is context length, b is the cache scalar width in bytes, A is the set of attention layers, and $h _ { \ell } ^ { \mathrm { K V } }$ and $d _ { \ell }$ are the number and dimensionality of KV heads in layer ℓ. The retention window w is finite for sliding-window attention and infinite for global attention. The factor of two accounts for separate key and value tensors, while $M _ { \mathrm { f i x e d } }$ includes sequence-length-independent recurrent or convolutional state.

Figure 1(a) applies Equation 1 to the attention-only models. Qwen3-4B contains 36 global-attention blocks with eight 128-dimensional KV heads, giving 144 MiB at 1,024 tokens and 4,608 MiB at 32,768 tokens [29]. Gemma-3-4B contains 34 attention blocks with four 256-dimensional KV heads, but only five use global attention; the other 29 retain at most 1,024 tokens under the evaluated MLX cache policy. Its footprint therefore increases from 136 to 756 MiB over the same range [30; 13].

The hybrid models retain less token-indexed attention state (Figure 1(b)). Granite-4.0-H-Micro combines four global-attention blocks with 36 Mamba2 blocks, whereas LFM2.5-2.6B combines eight global-attention blocks with 22 short-convolution blocks [31; 32]. Their token-indexed state grows by 8 and 16 KiB per token, respectively; fixed recurrent and convolutional states contribute 36.9 and 0.17 MiB. At 32,768 tokens, their footprints are consequently 292.9 and 512.2 MiB. Fewer attention blocks reduce cache growth and the state eligible for pruning.

(a) Attention-only models  
![](images/9e500c501c4a3be75fc6472df04e5296b3cb5525bea139d22aeb3ff6c243fc5a.jpg)  
<sup>Context</sup> <sup>length</sup> <sup>(</sup>×<sup>1024</sup> <sup>tokens)</sup>

(b) Hybrid models  
![](images/ef3f670754dc557d7929b5abcde75f24d6d60b2222ecd566bac8eec209c646f6.jpg)  
<sup>Context</sup> <sup>length</sup> <sup>(</sup>×<sup>1024</sup> <sup>tokens)</sup>

(c) Prefix reuse  
![](images/dc3d8c6fe12b0b3090284ee94ad3dd58dd6e0e4e03a5adb179827890a43ae646.jpg)  
Figure 1: Cache growth and persistent prefix reuse. Panels (a) and (b) report architecture-derived persistent cache state under the bfloat16 MLX-LM layout. Gemma-3-4B uses 1,024-token localattention caches; hybrid-model values include fixed recurrent or convolutional state. Panel (c) reports measured Qwen3-4B wall time with non-persistent and persistent conversational caches.

Persistent caching extends reuse beyond a single response by retaining the longest unchanged prefix across requests, avoiding repeated prefill [13; 33]. In the MLX-LM conversation, wall time at turns 3, 5, and 9 falls by 33%, 42%, and 57%, respectively (Figure 1(c)). These savings reflect avoided prefill rather than increased steady-state decoding throughput.

## 2.1.3 Dense-model pruning across devices

Figure 2 compares three representative pruning methods at a 4,096-token context. SnapKV represents KNorm, StreamingLLM, and TOVA, which differ in retention criteria but share post-prefill token selection and cache compaction and exhibit similar measured trends. ExpectedAttention shows the strongest hardware dependence, whereas FastKVzip provides an overhead-dominated counterexample. At 128–256 generated tokens, SnapKV and ExpectedAttention achieve 1.26–1.75× speedup on Apple Silicon, versus $0 . 4 5 \mathrm { - } 1 . 0 1 \times \mathrm { ~ o n ~ }$ the RTX 4090; FastKVzip remains below baseline on all devices $( 0 . 4 9 \mathrm { - } 0 . 7 4 \times )$ . At 1–16 generated tokens, M2 Pro attains the largest speedups for SnapKV and ExpectedAttention, consistent with a larger relative contribution of the eliminated attention work. This separation diminishes with output length, and the ratios are not ordered by chip generation.

(a) SnapKV (0.5)  
![](images/bc5672b42c0aa91e6ea6c58e0b8156feb1248ea2eef1f674b4ec54a9a2a7ecf5.jpg)

(b) ExpectedAttention (0.75)  
![](images/f8e13f7128e6ab97c89d9fc0b7d9e4492f27a8e23d0f1fdacca6e95f57640167.jpg)

(c) FastKVzip (0.75)  
![](images/b10c7eb3bab0bac0ce152fd0b813cde3d599ce20efe8e834feba0682547d1075.jpg)  
M2 Pro M3 Ultra M4 Pro RTX 4090  
Figure 2: Device-normalized KV-pruning speedup for Qwen3-4B-Instruct-2507 in bfloat16 with a 4,096-token context. Each curve uses the unpruned baseline on the same device. Values in parentheses denote compression ratios.

The complete matched grid shows how this hardware dependence varies with context length (Figure 3). At a 4,096-token context and 256-token output, ExpectedAttention attains 1.75×, 1.61×, and 1.72× on M2 Pro, M3 Ultra, and M4 Pro, respectively, but 0.98× on the RTX 4090. SnapKV, KNorm, StreamingLLM, and TOVA yield 1.36–1.50× across the Apple devices and 0.99–1.02× on the RTX 4090 in this workload. At a 512-token context, the maximum Apple speedup is only 1.24×. FastKVzip remains below baseline throughout the grid, at 0.28–0.96×. These results are consistent with reduced attention work more effectively amortizing pruning overhead in longer Apple workloads; compression ratio alone does not determine the resulting speedup.

## 2.1.4 Hybrid-model pruning and peak memory

The smaller amount of prunable attention state in LFM2.5-2.6B is accompanied by smaller latency gains (Figure 4). At a 512-token context, every method remains between 0.92× and 1.03× across the three Apple devices. At a 4,096-token context and 256-token output, speedups range from 1.17–1.25× on M2 Pro, 1.09–1.13× on M3 Ultra, and 1.11–1.18× on M4 Pro, with ExpectedAttention highest on each device. The maximum is thus 1.25×, compared with 1.75× for Qwen3-4B-Instruct-2507. The output-length dependence is qualitatively similar to Qwen, but attention-only pruning leaves LFM’s 22 short-convolution blocks and their state unchanged, limiting the inference cost it can remove.

Process-level peak memory also depends on temporary allocations rather than retained-cache size alone (Figure 5). On M4 Pro at a 4,096-token context, Qwen peak-memory savings range from 20.4% to 25.6% across the reported methods and outputs. LFM2.5-2.6B shows no material reduction for outputs of 1 or 16 tokens, followed by savings of 8.2–8.3% at 128 tokens and 21.1% at 256 tokens. At a 512-token context, pruning instead increases peak memory by up to 155.5% for Qwen and 48.5% for LFM. These increases are consistent with transient scoring and compaction allocations outweighing the smaller persistent cache. Retained KV state, process-level peak memory, and end-to-end latency therefore measure distinct effects of pruning.

Persistent prefix reuse yields consistent latency reductions by eliminating redundant prefill without token-selection overhead. Pruning gains depend on context and output length, device, and the fraction of inference cost attributable to attention KV state. In the matched Qwen workloads, the largest gains occur on Apple Silicon, while the RTX 4090 remains close to baseline or slows down; FastKVzip slows execution on both platforms. Neither compression ratio nor processor generation alone predicts end-to-end benefit, and reduced KV state does not ensure lower process-level peak memory.

![](images/68dcc73640aed74938f33571bdb3490b103c3a16e79533ab6399394eada13132.jpg)  
Figure 3: Complete matched end-to-end speedup grid for Qwen3-4B-Instruct-2507 in bfloat16. Columns vary context and output length; each cell uses the unpruned baseline for the same device and workload. Values above one indicate a latency reduction.

![](images/b80d63490c1c1a2e89c9d3f11f68b132e0c93a1d82a163b1e69642feac652145.jpg)  
Figure 4: Device-normalized end-to-end speedup for LFM2.5-2.6B in bfloat16. The matched grid contains five pruning methods evaluated on all three Apple devices.

![](images/c1c9d34e057e675d00fb1feeff0cf74c01cf0a178c3738367f9e4ef1a5ef3903.jpg)  
Figure 5: Process-level peak-memory saving on M4 Pro relative to the corresponding unpruned workload. Positive values indicate a lower peak; negative values indicate an increase. FastKVzip is unavailable for LFM2.5-2.6B.

## 2.2 Speculative Decoding

Speculative decoding accelerates autoregressive generation by separating token proposal from verification [5]. A cheaper drafting procedure proposes a sequence of candidate tokens, which the target model evaluates in a single forward pass. Verification retains the accepted prefix and produces the next target token, allowing several output tokens to be generated per target-model invocation. Research on speculative decoding has developed this principle by reducing drafting cost, improving the agreement between draft and target predictions, and increasing the parallelism of token proposal.

The classical approach, represented by Classic in our evaluation, uses a separate, smaller autoregressive language model to generate the draft sequence. The target then verifies these proposals together, without requiring retraining. The original method reports 2–3× acceleration for T5-XXL relative to standard T5X inference [5]; a later greedy-decoding evaluation reports 1.93× on MT-Bench with Vicuna-13B [18]. Although the auxiliary model is smaller, its sequential drafting steps still contribute to generation latency.

One way to reduce this cost is to replace neural drafting with inexpensive statistical lookup. Ngram drafting retrieves stored continuations of a matching token suffix, avoiding a neural forward pass. Such a procedure can also assist a neural drafter: Cascade Speculative Drafting recursively accelerates each drafter with a cheaper one, terminating at the statistical Max-Gram model [17]. Our Cascade-n-gram evaluates this vertical cascade using N-gram proposals for a small neural drafter. The full published method additionally assigns cheaper drafters to later positions and achieves 1.70× speedup for Vicuna-7B on GSM8K (56.72 versus 33.34 tokens/s) [17].

A complementary direction improves draft quality by using features already computed by the target model. The EAGLE family introduces lightweight learned drafting modules conditioned on these features. EAGLE-3 predicts tokens from fused low-, middle-, and high-layer representations, with a training-time test that feeds back the drafter’s own outputs to improve successive predictions; inference can use a dynamic candidate tree. Under greedy decoding, it reports 5.58× and 4.40× on MT-Bench with Vicuna-13B and LLaMA-3.1-8B-Instruct, respectively, and 6.47× on HumanEval with Vicuna-13B [18].

Feature conditioning improves proposal quality, but autoregressive drafting still requires sequential forward passes. More recent work therefore explores generating draft tokens in parallel. DFlash combines target-feature conditioning with a block diffusion drafter that predicts masked positions together in one pass, reporting 2.85× on MT-Bench and 6.09× on MATH-500 with Qwen3-4B [19]. DDTree extends this approach by constructing a candidate tree from the same per-position distributions under a node budget, maximizing a surrogate for expected acceptance length before joint verification. In its separate evaluation, DDTree improves DFlash’s MT-Bench speedup from 2.64× to 4.18×, and its MATH-500 speedup from 5.54× to 7.50× [20]. These published gains motivate our evaluation, although their models, workloads, and runtimes differ from ours; the learned and block-drafting studies primarily use server-class NVIDIA GPUs and 16-bit targets.

## 2.2.1 Matched Apple Silicon results

We evaluate Classic, N-gram, Cascade-n-gram, EAGLE-3, DFlash, and DDTree to test whether the benefits of these drafting strategies persist in the single-request MLX regime motivated in the introduction. The principal comparison uses M2 Pro and M3 Ultra devices with a 4-bit Qwen3-4B target, representative of local inference, and a matched BF16 target to examine precision sensitivity. All six methods use the same 100 conversational prompts, batch size and concurrency of one, and greedy generation, with identical method-specific settings across the two devices. Prompts average 1,776 tokens, whereas baseline completions average approximately 54–62 tokens. This distribution is chosen to represent local chat and tool use with small models, where system prompts and conversational context produce relatively long inputs followed by short responses. Each MLX adaptation is normalized to its own autoregressive baseline; absolute throughput across implementations is therefore not a controlled cross-method comparison. Appendix A.1 gives the checkpoints, draft budgets, and measurement protocol.

<table><tr><td rowspan=1 colspan=3>(a)4-bitDec. E2E</td><td rowspan=1 colspan=1>BFDec.</td><td rowspan=1 colspan=1>16E2E</td></tr><tr><td rowspan=1 colspan=1>Classic</td><td rowspan=1 colspan=1>-43.8</td><td rowspan=1 colspan=1>-21.5</td><td rowspan=1 colspan=1>+1.1</td><td rowspan=1 colspan=1>-13.8</td></tr><tr><td rowspan=1 colspan=1>N-gram</td><td rowspan=1 colspan=1>-25.2</td><td rowspan=1 colspan=1>-5.5</td><td rowspan=1 colspan=1>-21.0</td><td rowspan=1 colspan=1>-10.8</td></tr><tr><td rowspan=1 colspan=1>Cascade</td><td rowspan=1 colspan=1>-32.9</td><td rowspan=1 colspan=1>-19.5</td><td rowspan=1 colspan=1>-12.4</td><td rowspan=1 colspan=1>-19.3</td></tr><tr><td rowspan=1 colspan=1>EAGLE-3</td><td rowspan=1 colspan=1>-27.3</td><td rowspan=1 colspan=1>-14.4</td><td rowspan=1 colspan=1>-44.0</td><td rowspan=1 colspan=1>-28.5</td></tr><tr><td rowspan=1 colspan=1>DFlash</td><td rowspan=1 colspan=1>-69.8</td><td rowspan=1 colspan=1>-32.5</td><td rowspan=1 colspan=1>-8.3</td><td rowspan=1 colspan=1>-8.4</td></tr><tr><td rowspan=1 colspan=1>DDTree</td><td rowspan=1 colspan=1>-63.7</td><td rowspan=1 colspan=1>-28.0</td><td rowspan=1 colspan=1>-1.8</td><td rowspan=1 colspan=1>-5.7</td></tr></table>

(b)
<table><tr><td colspan="2">(b) 4-bit Dec. E2E</td><td colspan="2">BF16 Dec. E2E</td><td colspan="2">(c) DT</td><td colspan="2">AL</td></tr><tr><td>-24.0</td><td>-22.0</td><td>-4.4</td><td>-19.9</td><td></td><td>4</td><td>2.41-2.51</td></tr><tr><td>-8.9</td><td>-3.3</td><td>-16.7</td><td>-8.8</td><td></td><td>2</td><td>0.61-0.65</td></tr><tr><td>-22.1</td><td>-18.5</td><td>-12.6</td><td>-14.9</td><td></td><td>2</td><td>1.52-1.61</td></tr><tr><td>-16.8</td><td>-13.0</td><td>-42.7</td><td>-27.9</td><td></td><td>1</td><td>0.44-0.46</td></tr><tr><td>-61.7</td><td>-33.6</td><td>-23.5</td><td>-15.2</td><td></td><td>16</td><td>2.57-2.67</td></tr><tr><td>-60.8</td><td>-32.8</td><td>-19.3</td><td>-13.0</td><td></td><td>24</td><td>3.45-3.60</td></tr></table>

Figure 6: Decode (Dec.) and standardized end-to-end (E2E) throughput changes (%) relative to each implementation’s autoregressive baseline for Qwen3-4B on (a) M2 Pro and (b) M3 Ultra. Panel (c) gives the configured draft budget (DT) and mean accepted draft length (AL), shown as a range across the two devices for the 4-bit target. For DFlash, DT is the 16-position block including the target-produced root; for DDTree, DT is the budget of 24 non-root candidate nodes. AL excludes the root for both methods.

No configuration in this matched comparison improves E2E throughput (Figure 6). With the 4-bit target, all six methods also reduce decode throughput: losses range from 25.2% to 69.8% on M2 Pro and from 8.9% to 61.7% on M3 Ultra. N-gram is closest to E2E break-even, at −5.5% and −3.3%, respectively, despite its approximately near-zero lookup cost. A slower target baseline alone does not make speculation beneficial: even N-gram’s decode penalty is larger on M2 Pro, indicating that the complete draft–verify path scales differently from an autoregressive step.

BF16 likewise yields no E2E gain. Classic on M2 Pro is the only decode-positive configuration (+1.1%), but its E2E throughput falls by 13.8%. DFlash and DDTree show the strongest precision sensitivity: their decode penalties fall from 69.8% and 63.7% to 8.3% and 1.8% on M2 Pro, and from 61.7% and 60.8% to 23.5% and 19.3% on M3 Ultra. On M2 Pro, this mainly reflects a slower baseline: moving to BF16 reduces autoregressive decode throughput from approximately 56.7 to 21.0 tokens/s, while DFlash and DDTree remain near 19–21 tokens/s. EAGLE-3 instead becomes slower relative to its baseline on both devices, so target precision does not affect all methods uniformly.

Taken together, these results show that neither inexpensive token proposals nor a more costly target step consistently produces a net gain. To examine why the methods remain slower, we relate the number of tokens accepted in each round to the total cost of drafting and verification.

Acceptance and decoding cost. In the 4-bit M3 Ultra setting, DFlash accepts a mean of 2.57 draft tokens per round and DDTree 3.45, excluding the root (equivalently, 3.57 and 4.45 with the root). Nevertheless, DFlash reduces decode throughput from 156.7 to 60.0 tokens/s, and DDTree reaches only 61.4 tokens/s. EAGLE-3 and Cascade-n-gram similarly fall from 261.2 to 217.3 and from 162.0 to 126.3 tokens/s, respectively. Classic accepts 2.41–2.51 draft tokens per round across the two devices, yet on M2 Pro falls from 56.65 to 31.86 tokens/s. These within-implementation comparisons show that accepted work must be evaluated against the cost of the entire speculative round.

An approximate decode speedup is

$$
S _ { \mathrm { d e c } } \approx \frac { \tau t _ { \mathrm { A R } } } { t _ { \mathrm { d r a f t } } + t _ { \mathrm { t r e e } } + t _ { \mathrm { v e r i f y } } ( B ) + t _ { \mathrm { c o m m i t } } } ,\tag{2}
$$

where τ is the mean number of newly committed output tokens per round, including the targetproduced correction or bonus token as applicable; $t _ { \mathrm { A R } }$ is the latency of one autoregressive target step; and the denominator includes drafting, tree construction, verification at width $B ,$ , and output/KV-cache commitment. For methods without a tree, $t _ { \mathrm { t r e e } } = 0$ . Thus τ is distinct from the accepted-draft length AL in Figure 6. Decoding accelerates only when the total round cost is smaller than $\tau t _ { \mathrm { A R } } ;$ increasing acceptance or the draft budget alone does not ensure this inequality. This trade-off motivates the following analysis of how workload predictability, target cost, and verification overhead affect decode performance, and how prefill limits the resulting E2E benefit.

## 2.2.2 Factors limiting speedup

Workload predictability. The published block-drafting results show substantially less amortizable work on chat than on mathematical reasoning. For Qwen3-4B, DFlash’s reported acceptance length, including the target-produced token, falls from 7.84 on MATH-500 to 4.35 on MT-Bench, while speedup falls from 6.09× to 2.85× [19]. DDTree similarly falls from 10.71 to 6.70 tokens and from 7.50× to 4.18× [20]. Our conversational workload is consistent with this less predictable regime. These cross-workload observations support predictability as a limiting factor, without isolating it from differences in checkpoints, precision, and implementation.

Small and quantized targets. A 4-bit 4B target usual for the local on-device inference already has inexpensive autoregressive steps, reducing the work that a speculative round can replace. Draft-side costs need not shrink proportionally: DFlash and DDTree retain a BF16 drafter [34], and the EAGLE-3 head remains 4-bit even with the BF16 target. The BF16 comparison is therefore a sensitivity test of target precision within each method’s pipeline. The supplementary 8B-4bit experiment also improves relative E2E performance for most method–device pairs, but none reaches break-even; N-gram comes closest at −1.4% on M3 Ultra (Appendix A.2). More expensive target steps improve the trade-off for several methods, but neither target size nor quantization alone explains the negative results.

Backend-dependent verification. The wider target pass is only part of a speculative round. Block or tree preparation, custom attention masks, verifier traversal, KV-cache updates, and synchronization can add costs that do not scale with single-token target throughput. DFlash and DDTree’s smaller BF16 decode deficits on M2 Pro than on M3 Ultra illustrate this mismatch. A separate A100 DDTree reproduction achieves 3.20× decode speedup despite using PyTorch scaled dot-product attention (SDPA), rather than FlashAttention, for the ancestor-only mask. Consequently, the observed gap concerns the complete MLX and PyTorch draft–verify paths. Without kernel-level profiling or matched backend ablations, these results do not establish a hardware-only cause or a specific Metal-kernel limitation.

Non-accelerated prefill. The long prompts and short completions make prefill a substantial part of E2E latency. Speculation mainly accelerates decode and can add draft-model prefill, feature extraction, and cache initialization. With the BF16 target on M3 Ultra, Classic and Cascade-n-gram reduce prefill throughput by 29.6% and 16.8%, respectively, compared with 0.7% for N-gram. If baseline prefill occupies a fraction $f _ { \mathrm { p r e } }$ of total latency, unchanged prefill imposes the idealized ceiling

$$
S _ { \mathrm { e 2 e } } \leq \frac { 1 } { f _ { \mathrm { p r e } } } ,\tag{3}
$$

even with cost-free decode; additional prefill overhead lowers it. In a separate M3 Pro DDTree run, baseline prefill and decode take 2.60 and 3.39 seconds per request, giving a ceiling near 2.3×, while speculative prefill rises to 2.77 seconds. A matched A100 experiment using the same dataset and target family reports mean time to first token of 0.109 seconds versus 2.66 seconds on M3 Pro, consistent with a larger prefill limitation in the tested Apple stack; the different runtimes preclude hardware-only attribution.

Shorter prompts and longer outputs would reduce prefill’s share, but would not by themselves reverse the measured decode and E2E losses. In the principal comparison, only M2 Pro BF16 Classic has a positive decode change, and that gain is 1.1%. Other length regimes were not directly evaluated and may also change acceptance and phase throughput. The negative result therefore applies to the measured configurations, while the cost model explains the conditions needed for a gain.

## 2.2.3 Configuration-specific positive cases

Supplementary experiments identify configurations with positive decode or E2E changes (Table 2). The M2 Pro BF16 Classic row repeats the decode-only exception in the principal comparison; the remaining rows extend the device coverage. These observations use separate implementations and are not a controlled ranking of methods or processors.

Table 2: Apple Silicon configurations with an observed positive change in decode or standardized E2E throughput relative to the paired autoregressive baseline.
<table><tr><td>Method</td><td>Device</td><td>Target</td><td>E2E</td><td>Decode</td></tr><tr><td>Classic</td><td>M2 Pro</td><td>Qwen3-4B-BF16</td><td>-13.8%</td><td>+1.1%</td></tr><tr><td>Classic</td><td>M4 Pro</td><td>Qwen3-4B-4bit</td><td>-5.0%</td><td>+10.0%</td></tr><tr><td>N-gram</td><td>M3Max</td><td>Qwen3-4B-4bit</td><td>+2.0%</td><td>+13.0%</td></tr><tr><td>Cascade-n-gram</td><td>M3Max</td><td>Qwen3-4B-4bit</td><td>-11.6%</td><td>+7.6%</td></tr><tr><td>DFlash</td><td>M3 Pro</td><td>Qwen3-4B-BF16</td><td>+9.2%</td><td>+25.0%</td></tr><tr><td>DDTree</td><td>M3 Pro</td><td>Qwen3-4B-BF16</td><td>+15.5%</td><td>+40.4%</td></tr></table>

N-gram illustrates a device-dependent break-even point. With the same two-token lookup setting and 4-bit target, decode throughput changes by −8.9% on M3 Ultra, whose paired baseline reaches 162 tokens/s, but by +13.0% on M3 Max, whose baseline reaches 92 tokens/s. The −25.2% result on the still slower M2 Pro confirms that baseline throughput alone is not predictive. Likewise, Classic on M4 Pro and Cascade-n-gram on M3 Max improve decode while remaining slower E2E, consistent with prefill offsetting decode-side savings.

DFlash and DDTree on M3 Pro produce the largest observed E2E gains, 9.2% and 15.5%, respectively. Their paired BF16 baseline decodes at only 15.89 tokens/s, suggesting greater scope to amortize weight reads. A memory-bandwidth limitation is a plausible explanation: M2 Pro’s higher nominal bandwidth (200 versus 150 GB/s) coincides with a BF16 baseline near 20 tokens/s and a 1.8% DDTree decode loss. This comparison supports the hypothesis but does not isolate bandwidth from other processor and runtime differences. Gains also fail to persist across other tests: DFlash is 15% slower E2E on M4 Pro, and EAGLE-3 reduces M4 Pro decode throughput by 16%.

Conclusions. Speculative decoding can accelerate particular Apple Silicon configurations, but the published server-GPU gains do not transfer robustly to the principal local chat benchmark. Its benefit depends jointly on accepted work, target-step cost, the complete draft–verify path, and prefill’s share of latency. The positive cases fit this cost trade-off and reinforce the paper’s broader conclusion that speedup belongs to a method–runtime–hardware–workload combination.

## 2.3 Metal Kernels

CUDA kernels exploit asynchronous copies, matrix units and cooperative grids [35]; MLX on Apple Silicon exposes a different contract [2]. We test which mechanisms from Marlin [8], FlashInfer [7] and grouped MoE GEMM [9] survive in single-request decode. Each mechanism is classified with the taxonomy of §2.3.1, and matched RTX 4090 controls are used where possible (§2.3.6). A one-model megakernel case study is reported separately in §2.3.5.

## 2.3.1 The transfer taxonomy

A mechanism that fails to port does so for one of four reasons. Conflating them is what makes porting work feel arbitrary; separating them makes the outcome predictable. Every mechanism below carries exactly one label:

TRANSFERS The idea and its mechanism both exist on Metal. Port it.

HW-GAP The idea is sound but the hardware or language offers no equivalent instruction. Nothing you write closes this. Example: cp.async.

RUNTIME The overhead the technique removes has already been removed by the runtime, so the ceiling is near zero before you start. Example: kernel-launch amortisation under MLX’s lazy graph.

N/A The technique’s precondition does not hold in our regime. Not a platform limitation at all — the same code on the same hardware would need it if the regime changed. Example: sorting tokens by expert when the batch is one token.

The labels differ in who can act on them. HW-GAP holds until the vendor ships an instruction. RUNTIME is an instruction to measure the runtime before writing a kernel. N/A marks a technique waiting on a workload — batching, speculative decoding, serving — and so is deferred rather than refuted. For each ported technique, the distribution of verdicts in its transfer table — not the speed-up the port produced — is the result we argue from.

## 2.3.2 Marlin: weight-only 4-bit GEMM

W4A16 stores weights as 4-bit codes and activations in fp16 [10]. Before multiplication, each code must be reconstructed as a floating-point value. Materialising these values in memory would erase the bandwidth saving (Listing 2.1); Marlin instead reconstructs each weight in registers, consumes it immediately, and never writes the expanded representation to memory [8].

```c
// Strawman W4A16: correct, and it throws away the entire point.
dequantize<<<...>>>(w_q4, scales, biases, w_fp16); // writes N*K fp16 to DRAM
gemm(x_fp16, w_fp16, out); // reads N*K fp16 from DRAM
```

Listing 2.1: The naive path. The GEMM reads 16 bit per weight, so every byte saved by quantising is spent again — the accuracy cost of 4-bit is paid for no bandwidth gain.

## Mechanisms

M1: cheap unpack. Marlin exploits the fp16 bit layout: an fp16 with the exponent of 1024.0 and a 4-bit integer q in its low mantissa bits represents 1024 + q. A mask and or construct this pattern, and NVIDIA’s lop3.b32 performs the three-input Boolean operation in one instruction [36]. Eight nibbles fall from roughly 32 operations to about eight (Listing 2.2). Shifted nibbles are not shifted down; their power-of-two factor is folded into arithmetic that already occurs.

```rust
// Two nibbles at a time. 0x6400 is the fp16 bit pattern for 1024.0.
d = lop3(q & 0x000F000F, 0x64006400, LUT_AND_OR); // -> {1024+q0, 1024+q1}
r = __hsub2(d, {1024.0h, 1024.0h}); // -> {q0, q1}, no cvt
```

Listing 2.2: Marlin’s unpack. Eight nibbles go from roughly 32 operations (shift, mask, convert, scale, per value) to about eight.

M2: group factoring. For one scale and bias per group, the affine transform can be moved outside the per-weight work:

$$
\sum _ { i } ( s q _ { i } + b ) x _ { i } = s \sum _ { i } q _ { i } x _ { i } + b \sum _ { i } x _ { i } .\tag{4}
$$

The scale is then applied once per group and no dequantised weight is formed. Grouping itself is a checkpoint-format accuracy choice; this identity is what makes that format cheap to consume.

M3: asynchronous prefetch. cp.async and Hopper TMA copy the next tile from global to shared memory while the current tile is multiplied [35], overlapping the memory bus and matrix unit.

M4: offline repacking. Weights and metadata are shuffled at load time into inner-loop order, producing coalesced loads without runtime rearrangement.

What transfers Register-only reconstruction, group factoring and repacking transfer; unpacking needs different arithmetic. The matrix unit is N/A at M = 1, and public MSL makes asynchronous prefetch a HW-GAP (Table 3).

The asynchronous-prefetch verdict is narrower than it looks. The Apple GPU can perform deviceto-threadgroup asynchronous copies: private air.simdgroup\_async\_copy\_1d/2d intrinsics exist and

<table><tr><td>Mechanism</td><td>CUDA</td><td>Metal</td><td>Verdict</td></tr><tr><td>Never materialise fp16</td><td>registers</td><td>registers</td><td>TRANSFERS</td></tr><tr><td>M2 group factoring</td><td>algebra</td><td>algebra</td><td>TRANSFERS</td></tr><tr><td>M4 ofline repacking</td><td>load time</td><td>prepare_model</td><td>TRANSFERS</td></tr><tr><td>M1 cheap unpack</td><td>1op3</td><td>prescale (Eq. 5)</td><td>TRANSFERS†</td></tr><tr><td>Matrix unit</td><td>mma</td><td>simdgroup_matrix</td><td> $\mathrm { \Delta N / A ^ { \ddagger } }$ </td></tr><tr><td>M3 async prefetch</td><td>cp.async</td><td></td><td>HW-GAP</td></tr></table>

Table 3: Marlin transfer analysis. <sup>†</sup> The idea transfers; the instruction does not. Metal has no three-input logic op, so we reach the same end by different arithmetic. <sup>‡</sup> Metal has a matrix unit, but at $\bar { M } = 1$ there is no matrix to multiply — this is a regime limitation, not a platform one, and it reverses at $M \ge 2 ( \ S 2 . 3 )$ .

Apple’s MPS uses them. They are undocumented and inaccessible from public MSL [37], making this an API gap rather than evidence of missing silicon.

Metal implementation Because MSL has no lop3 equivalent [37], the port reaches M1 by exact power-of-two prescaling. For nibble j, Metal leaves the value shifted in place and prescales its activation:

$$
\sum _ { j } x _ { j } q _ { j } \ = \ \sum _ { j } \bigl ( x _ { j } 1 6 ^ { - j } \bigr ) \bigl ( q _ { j } \ll 4 j \bigr ) .\tag{5}
$$

The factors are exact powers of two, preserving the result while removing inner-loop shifts. The Metal path folds $1 6 ^ { - j }$ into each activation rather than folding the opposite factor into a scale; because activations are loaded once per block of output rows, this amortises the adjustment. The final inner loop combines group factoring, interleaved scale/bias metadata and activation reuse, with $\Sigma { } x$ hoisted outside the row loop (Listing 2.3).

// marlin\_metal/csrc/marlin\_qmv.metal   
for (int i = 0; i < VPT; i += 4) { // activations: loaded ONCE per block   
sum += x0 + x1 + x2 + x3; // Σx for the group-factored bias   
xt[i] = x0;   
xt[i + 1] = x1 \* (1.0f / 16.0f); // fold 16<sup>−j</sup> into x   
xt[i + 2] = x2 \* (1.0f / 256.0f);   
xt[i + 3] = x3 \* (1.0f / 4096.0f);   
}   
for (int r = 0; r < NR; r++) { // NR output rows share that one load   
float accum = 0.0f;   
for (int i = 0; i < VPT / 4; i++) {   
accum += xt[4\*i ] \* (wr[i] & 0x000f) // masked in place: still   
+ xt[4\*i + 1] \* (wr[i] & 0x00f0) // shifted, and the prescale   
+ xt[4\*i + 2] \* (wr[i] & 0x0f00) // cancels the shift exactly   
+ xt[4\*i + 3] \* (wr[i] & 0xf000);   
}   
const vec<T,2> s\_b = sb\_v[r \* gpr + g]; // M4: one load, interleaved   
acc[r] = fma(s\_b[0], accum, fma(s\_b[1], sum, acc[r]));   
}

Listing 2.3: The Metal inner loop. Three of Marlin’s four mechanisms are visible: group factoring in the final fma, interleaved metadata in the single vec<T,2> load, and activation reuse across NR rows. Note that Σx is hoisted out of the row loop — the bias term of Eq. (4) costs one add per element amortised over all NR rows.

Metal shaders are monomorphic, so the implementation instantiates variants for three dtypes and six shape buckets: marlin\_qmv\_{f16,bf16,f32}\_p{1,2,4}\_r{4,8}. Making NR a template parameter rather than a runtime bound is worth 10 %, because it allows the compiler to unroll the row loop and allocate registers accordingly.

Results At M = 1 the kernel measures $0 . 9 5 { \times } { - 1 . 0 0 } { \times }$ against MLX qmv and 0.993× end to end. MLX already reaches $4 4 0 \mathrm { G B s ^ { - 1 } }$ , about 80 % of DRAM peak, leaving little room for cheaper unpacking; a 6-bit lm\_head matvec likewise measured parity, because MLX runs that shape at the same rate at 4, 6 and 8 bits (App. B.0.1). The picture changes at $M = 2 \ – 8$ . Stock MLX’s tiled

GEMM reaches roughly 5× the $M = 1$ time at $M = 8$ despite identical weight traffic. Dequantising each block once into registers and reusing it across activation rows (Listing 2.4) changes $t ( \bar { 8 } ) / t ( 1 )$ to 4.1–4.5, versus MLX’s 4.9–5.1, and yields 1.09×–1.20× on all tested Qwen3-4B and Qwen3-8B transformer-layer shapes [38].

// Fully dequantise this block into registers ONCE, folding s and b in, so the   
// per-m loop below is pure load+FMA and the dequant amortises over all M rows.   
float wf[NR][VPT];   
for (int r = 0; r < NR; r++) {   
const vec<T,2> s\_b = sb\_v[r \* gpr + g];   
const float s = float(s\_b[0]), b = float(s\_b[1]);   
for (int i = 0; i < 4; i++) {   
const uint16\_t wv = w\_row[i];   
wf[r][4\*i ] = fma(s, float(wv & 0x000f), b);   
wf[r][4\*i + 1] = fma(s \* (1.0f/16.0f), float(wv & 0x00f0), b);   
wf[r][4\*i + 2] = fma(s \* (1.0f/256.0f), float(wv & 0x0f00), b);   
wf[r][4\*i + 3] = fma(s \* (1.0f/4096.0f),float(wv & 0xf000), b);   
}   
}   
for (int m = 0; m < NM; m++) { /\* pure load + FMA against wf[][] \*/ }  
Listing 2.4: Small-M path. Here the weights are widened into registers, because they will be reused M times — the opposite trade from Listing 2.3, and correct for the same reason.

CUDA’s stock 4-bit kernel already serves $M = 8$ for only 5 % to 12 % more time than $M = 1$ (§2.3.6), so the Apple gain closes an MLX runtime gap rather than recovering an instruction-level mechanism.

Speculative decoding The verify pass of speculative decoding naturally produces $M \in \left[ 2 , 8 \right] \left[ 5 \right]$ making the small-M path directly useful. Table 4 reports Qwen3-8B-4bit with a Qwen3-0.6B-4bit draft.
<table><tr><td>k</td><td>verify M</td><td>stock (tok/s)</td><td>marlin (tok/s)</td><td>ratio</td></tr><tr><td>2</td><td>3</td><td>78.5</td><td>79.9</td><td>1.02×</td></tr><tr><td>4</td><td>5</td><td>68.6</td><td>72.9</td><td>1.06×</td></tr><tr><td>7</td><td>8</td><td>51.9</td><td>59.0</td><td>1.14×</td></tr><tr><td>8</td><td>9</td><td>45.1</td><td>45.1</td><td>1.00×</td></tr></table>

Table 4: Speculative decoding, 256 tokens, median of 2. Plain decode is 52.4 tok/s. At $k = 7$ speculation is break-even with stock kernels (0.99× against plain decode) and becomes 1.13× with ours. $\mathrm { A t } k = 8$ the verify is $M = 9 ;$ our kernel disengages and both paths measure 1.00×.

## 2.3.3 FlashInfer: decode attention

FlashInfer combines online softmax [39; 6] with split-KV, which partitions the cache and merges partials to restore batch-1 parallelism [40].

Mechanisms and constraints Decode attention has two awkward properties. First, its softmax cannot be materialised cheaply: stabilising it requires the maximum score, but storing all N scores would write and reread a vector per head. Online softmax (M1) instead maintains $( m , \ell , o )$ and rebases the accumulated history when m changes:

$$
m ^ { \prime } = \operatorname * { m a x } ( m , s ) , \qquad \alpha = e ^ { m - m ^ { \prime } }\tag{6}
$$

$$
\ell  \alpha \ell + e ^ { s - m ^ { \prime } } , \qquad o  \alpha o + e ^ { s - m ^ { \prime } } v .\tag{7}
$$

This gives one pass and memory constant in N.

Second, decode removes the positions dimension from available parallelism. A single request with 32 heads exposes only 32 large serial units. Split-KV (M2) partitions the cache into C chunks, turning these into 32C independent units; each returns a local $( o _ { c } , \ell _ { c } , m _ { c } )$ merged by

$$
m = \operatorname* { m a x } _ { c } m _ { c } , \qquad o = \sum _ { c } o _ { c } e ^ { m _ { c } - m } , \qquad \ell = \sum _ { c } \ell _ { c } e ^ { m _ { c } - m } , \qquad \mathrm { o u t } = o / \ell .\tag{8}
$$

The same rescaling identity makes the loop both streamable and composable. FlashInfer adds two serving-oriented mechanisms. A paged, block-sparse KV cache (M3) lets one template address paged caches [4], ragged batches and shared prefixes. A host-side plan (M4) distributes chunks from unequal sequence lengths so no CTA idles. Listing 2.5 shows the resulting decode loop.

plan = scheduler(seq\_lens, num\_heads); // host: balance ragged batches   
for (tile in my\_chunk) { // each CTA owns one (head, chunk)   
cp.async(smem[nxt], paged\_kv[tile + 1]);   
s = qk\_mma(q, smem[cur]); // matrix unit   
m2 = max(m, rowmax(s)); // online max update   
o = o \* exp(m - m2) + pv\_mma(exp(s - m2), v);   
l = l \* exp(m - m2) + rowsum(exp(s - m2));   
m = m2;   
}   
write\_partial(o, m, l); // consumed by the merge kernel  
Listing 2.5: FlashInfer’s decode loop, abridged. Three of its four mechanisms are visible in the first two lines and the last.

What transfers Online softmax, split-KV and zero-copy cache views transfer. Paging and loadbalanced planning require concurrent serving and are N/A; asynchronous prefetch is a HW-GAP (Table 5).

<table><tr><td>Mechanism</td><td>CUDA</td><td>Metal</td><td>Verdict</td></tr><tr><td>M1 online softmax</td><td>running triple</td><td>identical</td><td>TRANSFERS</td></tr><tr><td>M2 split-KV</td><td>2 kernels + workspace</td><td>2 dispatches</td><td>TRANSFERS</td></tr><tr><td>Zero-copy cache view</td><td>pointer arithmetic</td><td>strides at eval time</td><td>TRANSFERS</td></tr><tr><td>M3 paged / block-sparse</td><td>page table</td><td></td><td>N/A</td></tr><tr><td>M4 load-balanced plan</td><td>host scheduler</td><td>lookup table</td><td>N/A</td></tr><tr><td>QKT on matrix unit</td><td>mma</td><td>simd_sum</td><td>N/A</td></tr><tr><td>Async prefetch</td><td>cp.async</td><td></td><td>HW-GAP</td></tr></table>

Table 5: FlashInfer transfer analysis. Unlike Marlin (Table 3), most of what does not transfer is N/A rather than HW-GAP: MLX’s KV cache is a single growing contiguous slice, so there are no pages to index, and at batch 1 there is nothing to load-balance.

Metal and runtime implementation The online-softmax equations transcribe unchanged (Listing 2.6). Metal uses simd\_sum, a cross-lane shuffle touching no memory, instead of a matrix unit for QK<sup>⊤</sup>. MLX presents its growing cache as a strided view; materialising that view per call would dominate attention, so the primitive reads strides at evaluation time (ensure\_row\_contiguous=false). This is not a page table, but supplies the required property here: addressing the cache without copying it.

// cider\_mlx/csrc/cider\_sdpa\_vector.metal   
for (int i = simd\_gid; i < N; i += BN) { // simdgroups stride the cache   
U score = 0;   
for (int j = 0; j < qk\_per\_thread; j++) score += q\_reg[j] \* k\_ptr[j];   
score = simd\_sum(score); // no matrix unit: 32-lane reduce   
U new\_max = max(max\_score, score); // online max update   
U factor = fast::exp(max\_score - new\_max);   
U exp\_score = fast::exp(score - new\_max);   
max\_score = new\_max; // online accumulator update   
sum\_exp\_score = sum\_exp\_score \* factor + exp\_score;   
for (int j = 0; j < v\_per\_thread; j++)   
o\_reg[j] = o\_reg[j] \* factor + exp\_score \* U(v\_ptr[j]);   
k\_ptr += inner\_k\_stride; // strides read at eval time:   
v\_ptr += inner\_v\_stride; // the cache slice stays zero-copy   
}

![](images/cf5674a611e3c8ecb233a0f22d961af36c8139a88281e07fba3ecf2261f4c13d.jpg)  
Figure 7: Decode-throughput speed-up across context lengths. FlashInfer reaches 1.20× on M4 Pro but 0.97× on M2 (a,b), whereas MoE gather provides consistent gains across all four Apple GPUs (c). Exact values are reported in Tables 6 and 9.

Listing 2.6: The Metal kernel. The online-softmax update transcribes unchanged. The dot product is reduced with simd\_sum, a cross-lane shuffle that touches no memory, in place of the matrix unit. Runtime integration matters independently of shader quality. A Python wrapper around the identica shader measured 0.63× against stock at N = 256, 0.73× at $N = \dot { 1 } 0 2 4$ , and became favourable only at N = 16384 (1.34×). As a native MLX primitive the same source measures 1.06×, 1.06× and 1.38×. The kernel was not responsible for the short-context loss; per-call wrapper overhead was.

Results End to end, the gain reaches 1.20× on M4 Pro but falls to 0.97× on M2 (Fig. 7a,b; Table 6).
<table><tr><td rowspan="3">Model</td><td colspan="4">speed-up vs. stock, by context length</td></tr><tr><td>1k</td><td>5k</td><td>10k</td><td>16k</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>M4 Pro Qwen3-4B (dense)</td><td>1.02×</td><td>1.08×</td><td>1.15×</td><td>1.20×</td></tr><tr><td>Qwen3-8B (dense)</td><td>1.02×</td><td>1.05×</td><td>1.10×</td><td>1.15×</td></tr><tr><td>LFM2.5-1.2B (hybrid)</td><td>1.05×</td><td>1.03×</td><td>1.07×</td><td>1.09×</td></tr><tr><td>LFM2.5-8B-AiB (hybrid)</td><td>1.05×</td><td>1.02×</td><td>1.04×</td><td>1.06×</td></tr><tr><td>M2</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-4B (dense)</td><td>1.021×</td><td>0.991×</td><td>0.982×</td><td>0.970×</td></tr><tr><td>Qwen3-8B (dense)</td><td>1.015×</td><td>0.998×</td><td>0.987×</td><td>0.980×</td></tr><tr><td>LFM2.5-1.2B (hybrid)</td><td>0.997×</td><td>0.991×</td><td>0.986×</td><td>0.978×</td></tr><tr><td>LFM2.5-8B-A1B (hybrid)</td><td>1.005×</td><td>0.994×</td><td>0.989×</td><td>0.984×</td></tr></table>

Table 6: End-to-end decode, generation length 256 — the per-cell figures behind Fig. 7. M2 is reported to three decimals because its whole range is within 3 % of parity; at two decimals LFM2.5- 1.2B at 1k rounds to 1.00× and its 0.3 % loss disappears.

The end-to-end ordering follows model structure. Dense Qwen3-4B moves most; Qwen3-8B moves less because larger MLP and projection matmuls dilute attention; hybrid LFM2.5-8B-A1B caps at 1.06× because only 6 of 24 layers use attention. The fraction of token time touched by a kernel bounds its effect in either direction.

In isolation the attention kernel changes sign across Apple GPU families (Table 7).

On M2 and M2 Pro the split-KV loss grows with context, the mirror image of the large-GPU curve. Prefill is unchanged on M2 (1.00× across models and contexts, TTFT within 0.5 %), locating the effect in decode. The cause is split-KV itself: a second dispatch and a partials round trip have fixed cost, while extra parallelism has value only when one query does not already occupy the GPU. RTX 4090/FlashInfer and Metal/MLX also trend oppositely with context, but different baselines prevent hardware-only attribution (§2.3.6).

Split-KV is therefore enabled per architecture from a measured table, and an unmeasured GPU keeps stock MLX routing; the flat $N \geq 1 0 2 4$ threshold of the reference port cost 1–3% on M2 (App. B.0.2).

<table><tr><td>Machine</td><td>ctx 1k</td><td>ctx 16k</td><td>dependent-chain bubble</td></tr><tr><td>M2</td><td>+0.4%</td><td>-1.7%</td><td>7.0%</td></tr><tr><td>M2 Pro</td><td>-0.5%</td><td>-5.7%</td><td>24.0%</td></tr><tr><td>M4 Pro</td><td>+2.3%</td><td>+6.4%</td><td>32.7%</td></tr><tr><td>M3 Ultra</td><td>+1.4%</td><td>+2.2%</td><td>39.9%</td></tr></table>

Table 7: Attention kernel alone on LFM2.5-8B-A1B, generation length 512. The serialisation bubble is the shortfall between dependent and independent chains of 24 matvecs: bandwidth lost while one dispatch drains and the next ramps.

## 2.3.4 MoE gather kernel

An MoE router selects the top-k experts for each token, after which the gather applies a different weight matrix to each selected slot. NVIDIA grouped-GEMM and block-sparse kernels organize many tokens per expert [41; 9; 4], unlike batch-1 decode.

## Mechanisms and their preconditions

M1: sort by expert. Permute token rows so that all rows routed to one expert are contiguous; each expert’s slice then becomes an ordinary dense GEMM. The irregularity is paid once instead of inside the inner loop.

M2: grouped GEMM. One launch performs E matmuls with distinct shapes and base pointers. A problem visitor walks the $( M _ { e } , N , K , \bar { { \cal A } } _ { e } , B _ { e } )$ tuples and distributes tiles across all groups, preventing an expert with few tokens from leaving a multiprocessor idle.

M3: block-sparse rather than padded. MegaBlocks avoids rounding every expert’s token count to a tile multiple by expressing the layer as one block-sparse matmul with variable block sizes. This avoids both wasted padded compute and quality loss from dropping overflow.

M4: fuse the block. Gate, up, the SwiGLU nonlinearity [42], down and weighted reduction are combined in as few launches as possible, keeping large intermediates on chip.

What transfers All four mechanisms assume many rows per expert. At batch 1 there is nothing to sort, no group larger than one row, no tile quantisation to avoid and no load-balancing problem. Sorting, grouped scheduling, block sparsity and block fusion are therefore N/A; asynchronous prefetch is a HW-GAP (Table 8).

<table><tr><td>Mechanism</td><td>Precondition</td><td>At batch 1</td><td>Verdict</td></tr><tr><td>M1 sort by expert</td><td>many rows per expert</td><td>one row</td><td>N/A</td></tr><tr><td>M2 grouped GEMM</td><td>groups worth scheduling</td><td> $M _ { e } = 1$ </td><td>N/A</td></tr><tr><td>M3 block-sparse</td><td>tile-size quantisation</td><td>no tiles</td><td>N/A</td></tr><tr><td>M4 fuse the block</td><td>large intermediates</td><td>one vector</td><td> $\mathrm { \Delta N / A } ^ { \dagger }$ </td></tr><tr><td>Matrix unit</td><td> $\bar { M \geq }$  tile height</td><td> $M = 1$ </td><td>N/A</td></tr><tr><td>Async prefetch</td><td>cp.async</td><td></td><td>HW-GAP</td></tr></table>

Table 8: MoE grouped GEMM transfer analysis. Every mechanism is N/A at batch 1 — an unusually clean case. <sup>†</sup> We tested M4 anyway; see $\ S 2 . 7$

Metal implementation The implementation therefore reduces to the same quantized inner loop with one expert-weight indirection. For router indices $\iota _ { j }$ , the batch-1 operation is

$$
\mathrm { o u t } [ j ] = x \cdot \mathrm { d e q u a n t } \big ( { \cal W } [ \iota _ { j } ] \big ) ^ { \top } , \qquad j = 1 \ldots k .\tag{9}
$$

It differs from the dense matvec only in the weight base address.

Three details distinguish the MoE gather from the dense matvec. mlx-lm’s SwitchGLU uses broadcast inputs for gate and up (one activation shared across the token’s k slots) and per-slot inputs for down (one vector per expert); one flag selects the address calculation. LFM2.5’s expert width $K = 1 7 9 2$ divides no convenient tiling, so a wide main pass covers the largest tiled prefix and a guarded single-word loop handles the tail. Finally, the offline scale/bias interleaving of §2.3.2 transfers unchanged. The resulting two pointer offsets appear in Listing 2.7.

```c
// moe_metal/csrc/gather_qmv.metal -- the entire MoE-specific part
const int slot = int(tid.y); // which of the token's top-k slots
const int expert = int(indices[slot]); // expert gather, one load
const int m = per_slot ? slot : slot / topk;
const device T* x_ptr = x + size_t(m) * K;
const device uint16_t* ws = (const device uint16_t*)
(w + (size_t(expert) * N + row0) * K_w32); // offset the weights
const device vec<T,2>* sb_v = (const device vec<T,2>*)sb
+ (size_t(expert) * N + row0) * gpr; // and the metadata
// from here it is byte-for-byte the dense matvec chassis
gather_qmv_body<T, BITS, ROWS>(x_ptr, ws, sb_v, K, gpr, gs_shift, ...);
```  
Listing 2.7: Two pointer offsets. The expert identity enters the kernel as arithmetic on a base address and nothing else — no sort, no permutation, no scatter, and no divergent branch, since slot is uniform across the threadgroup.

Results and composition with attention Against mx.gather\_qmm, the kernel reaches 1.05× on gate/up and 1.23× on down; patching 66 SwitchLinear layers yields 1.03×–1.12× end to end across machines (Fig. 7c):
<table><tr><td>Machine</td><td>Speed-up range</td></tr><tr><td>M2</td><td>1.032×-1.045×</td></tr><tr><td>M2 Pro</td><td>1.086×-1.115×</td></tr><tr><td>M4 Pro</td><td>1.051×-1.104×</td></tr><tr><td>M3 Ultra</td><td>1.095×-1.122×</td></tr></table>

The gain comes from removing machinery batch-1 decode does not need, not from recreating grouped GEMM. Expert-weight traffic is independent of KV-cache length, whereas attention traffic grows with it. The two kernels therefore affect disjoint operations, the MoE contribution stays roughly flat with context, and the gains compose almost multiplicatively (Table 9).

<table><tr><td>ctx</td><td>stock (tok s−¹)</td><td>attention</td><td>MoE</td><td>both</td><td>product</td></tr><tr><td>1000</td><td>167.3</td><td>1.012×</td><td>1.068×</td><td>1.075×</td><td>1.081×</td></tr><tr><td>5000</td><td>158.5</td><td>1.024×</td><td>1.060×</td><td>1.090×</td><td>1.085×</td></tr><tr><td>10000</td><td>149.4</td><td>1.042×</td><td>1.058×</td><td>1.106×</td><td>1.102×</td></tr><tr><td>12000</td><td>145.6</td><td>1.051×</td><td>1.058×</td><td>1.113×</td><td>1.112×</td></tr><tr><td>16000</td><td>140.0</td><td>1.064×</td><td>1.052×</td><td>1.114×</td><td>1.119×</td></tr></table>

Table 9: Composition on M4 Pro. The final column is the product of the two single-kernel columns. Disjoint kernels multiply to within measurement noise.

## Follow-up experiments

Down projection plus routing reduction (kept). Combining the down matvec, router-score multiply and top-k sum removes a genuine intermediate tensor and adds 1.2–3.0% on top of the gather.

Gate, up and SwiGLU fusion (not enabled). This measured 0.986×–0.999× on LFM2.5’s coarse experts. It reads the same bytes as the separate kernels while adding register pressure.

Explicit repeated-expert deduplication (negative). Although two slots selecting the same expert appear to read its rows twice, explicit staging measured only 0.76×–0.93×. Apple’s system-level cache already serviced the repeat, leaving the staged version to pay ownership imbalance and register pressure for a saving already made.

Grouped GEMM in prefill (open negative). The grouped-GEMM precondition does hold during prefill. Our bit-exact implementation nevertheless measures 0.47× against stock on M4 Pro. This is an open implementation result, distinct from the N/A verdict for batch-1 decode.

## 2.3.5 Exploratory megakernel case study

A megakernel fuses an entire block or forward pass into one persistent launch: the grid remains resident, phases exchange data through memory, and later loads can be pipelined behind earlier computation. This can remove launch boundaries and inter-kernel bubbles, and has produced strong batch-1 results on NVIDIA systems [43; 44].

We treat the following experiment as an exploratory case study, not as one of the primary porting results. It covers one LFM2.5-8B-A1B MoE block on one M4 Pro under MLX, and the implementation was not evaluated broadly enough to support a claim about every model, Apple GPU or possible megakernel design. Its value is diagnostic: it exposes several conditions that a persistent, grid-synchronised design would have to overcome on public Metal.

Our hypothesis is that this class of megakernel is poorly matched to the current Metal/MLX contract for three reasons. MLX already amortises most dispatch and CPU encoding overhead; public Metal provides no cooperative-launch, grid-barrier or occupancy-query contract; and one persistent grid forces heterogeneous phases to share geometry and carry their combined resource footprint. The first two are properties of the exposed runtime/API contract. The size of the geometry penalty is specific to the model and kernel tested here.

The prototype fused eleven operations — router, fp32 softmax, top-k, gated experts, down projection and weighted reduction — into one bit-faithful dispatch. With the barrier disabled to isolate compute, it measured 174 µs against 96 µs for the stock chain, a 1.8× slowdown. This observation does not establish that megakernels cannot work on Metal; it motivates the precondition tests and interpretation below.

## P1: what a dispatch boundary actually costs

<table><tr><td>GPU-side dispatch boundary CPU graph build and encode per token</td><td>3.04μs 0.54ms</td></tr></table>

The last row is the one that matters: a pipelined token takes 6.04 ms against 6.53 ms for a synchronous eval, so the 0.54 ms of graph building is already hidden. MLX encodes a token’s whole graph into a few command buffers with consecutive dispatches sharing an encoder, and mlx-lm overlaps token t+1’s CPU work with token t’s GPU work; on NVIDIA, by contrast, a 3 µs to 10 µs launch against a kernel of a few microseconds puts boundaries at up to half of wall-clock. Summed over the MoE block the boundaries total 15 µs — the entire upside of fusing it.

Cooperation, occupancy and geometry Three further probes, detailed in App. B.0.3, show why the fused grid cannot be trusted or made fast. Public Metal offers no cooperative launch: a hand-rolled grid barrier over relaxed atomics works, but plain device stores were 96 % stale across threadgroups, and a register-heavier variant of the same kernel stopped converging with no occupancy query to predict it (App. B.0.3–B.2). Forcing all phases onto one grid also cost 78 µs on this block, about 5× the 15 µs that removing every boundary can recover (App. B.2).

Where the dependency bubble actually goes The dependency bubble a megakernel targets is real: 24 dependent matvecs run at $7 7 \mathrm { G B s ^ { - 1 } }$ where the same 24 run independently at $1 1 4 \mathrm { G B s ^ { - 1 } }$ , a 32.7 % serialisation loss on M4 Pro, and token accounting leaves 2 ms of a 6.04 ms token unexplained by summed operation times. The pool exists; the question is what reaches it (Table 10).

<table><tr><td>Experiment</td><td>Mechanism</td><td>End-to-end</td></tr><tr><td>Fused add + RMSNorm</td><td>fewer bytes (one pass)</td><td> $+ 1 \mathrm { t o } + 1 . 8 \%$ </td></tr><tr><td>MoE gather,  $M \geq 2$ </td><td>better rate</td><td> $1 . 2 5 \times \mathrm { a t } M { = } 8$ </td></tr><tr><td>Conv mixer, 6 ops to 1</td><td>2.2× rate, 1 dispatch</td><td> $1 . 0 1 \times - 1 . 0 7 \times ,$  short ctx only</td></tr><tr><td>4-bit KV cache</td><td>fewer KV bytes</td><td> $1 . 1 7 { \times } { - } 1 . 3 7 { \times }$  below 2k</td></tr><tr><td>Explicit expert dedup</td><td>remove repeated reads</td><td>negative</td></tr><tr><td>6-bit lm_head</td><td>cheaper unpack</td><td>parity (rate-capped shape)</td></tr><tr><td>Whole MoE block fused</td><td>remove all dispatches</td><td>negative</td></tr></table>

Table 10: Seven attempts on the same bubble. Every positive result moved fewer bytes or moved them faster; attempts aimed only at removing dispatches were neutral or negative.

In this Apple/MLX configuration, dispatch boundaries are nearly free and byte traffic dominates. Decode runs at roughly $1 7 0 \mathrm { G B s ^ { - 1 } }$ effective against per-operation rates of $2 5 0 \mathrm { \dot { G } B s ^ { - 1 } } \mathrm { - } 4 4 0 \mathrm { G B s ^ { - 1 } } ;$

these experiments reached the gap by shortening dependent chains — speculative decoding or batching — or by shrinking the byte budget, not by replacing dispatches with a persistent grid.

Interpretation and limitations In our view, persistent multi-phase megakernels are unlikely to be a robust optimisation under the current public Metal/MLX contract when they require grid-wide barriers. The removable boundary cost was only 15 µs in this case, whereas the fixed-grid penalty was 78 µs; correctness also depended on co-residency that Metal does not guarantee or expose through an occupancy query. These conditions make performance and forward progress sensitive to compiler resource allocation. An interpreter-style design [44] would retain both pressures (App. B.2).

This is a scoped interpretation, not an impossibility result. A narrower fused kernel without gridwide cooperation, a different model geometry, a future Metal API, or a runtime with larger visible dispatch costs could change the trade. The transferable recommendation is to measure boundary cost, cooperation guarantees, occupancy sensitivity and per-phase geometry before committing to a persistent implementation.

## 2.3.6 RTX 4090 comparison

The RTX 4090 controls below measure two platform-level quantities used by the kernel analysis: the gap between dependent and independent matvec chains, and the cost of a dispatch boundary. The matched runs use PyTorch 2.13, CUDA 13.2, the same chain length, shape, quantisation group and interstitial operation as the Apple experiment. They provide comparative evidence for these two mechanisms, not a direct counterpart for every kernel verdict.

Method and pitfalls Kernel timings are medians of 50 CUDA-event-timed repetitions after warmup; the dispatch experiment uses 20 repetitions. Three details changed the numbers by more than the effects being measured, and are recorded because a reader reproducing this will hit them:

• L2 residency. The 4090 holds 75 MB of L2, which swallows a whole layer’s weights (49.8 MB at K=2560, N=9728 in fp16) and a 16k KV cache (67 MB) alike. Timing one buffer repeatedly therefore reports cache bandwidth: our first attempt produced 1277 $\mathrm { G B } \mathrm { s } ^ { - 1 }$ and 1337 $\mathrm { G B s ^ { \frac { - 1 } { \hbar } } }$ , both above the card’s $1 0 0 8 \mathrm { G B s ^ { - 1 } }$ DRAM peak. Every bandwidth measurement below rotates over enough distinct buffers to exceed L2 by 3×.

• GQA expansion. Materialising the grouped-query KV cache with repeat\_interleave makes the baseline read $h _ { q } / h _ { k v } = 4$ times more bytes than the kernel under test, inflating the ratio fourfold. SDPA is therefore called with enable\_gqa=True.

• Implicit batching. llm.generate(prompts) runs the whole prompt set concurrently; the speculative sweep uses max\_num\_seqs=1 so that it measures single-request decode rather than batch-32 throughput, where speculation loses for unrelated reasons.

The matched contract experiment and the reported PyTorch int4 measurements use group size 64, as on Apple. PyTorch’s int4 primitive requires BF16 activations, where the Apple path uses FP16. The withheld vLLM Marlin attempt used group size 128.

<table><tr><td>Shape</td><td>Weights</td><td>dependent</td><td>independent</td><td>penalty</td><td>of peak</td></tr><tr><td>K=N=2048</td><td>4-bit, group 64</td><td> $9 2 . 2 \mathrm { G B s ^ { - 1 } }$ </td><td> $2 1 6 . 4 \mathrm { G B s ^ { - 1 } }$ </td><td>57.4%</td><td>21.5%</td></tr></table>

Table 11: Matched RTX 4090 chain with BF16 activations. The penalty exceeds the Apple range of 7.0 %–39.9 %, while the independent chain reaches 21.5 % of advertised DRAM bandwidth.

The platform contract The matched result reverses the earlier layer-shape measurement: the RTX 4090 loses 57.4 %, more than any tested Apple GPU. This quantity includes the interstitial elementwise dispatches in the dependent path, exactly as in the Apple experiment, so we interpret it as a dependency-chain penalty rather than a pure matvec roofline effect. Issuing the independent matvecs on 24 CUDA streams falls further to $\mathrm { 6 5 . 3 G B s ^ { - 1 } }$ (0.302× against one stream), so naive concurrency does not recover the gap.

What a dispatch boundary costs The matched chain of 512 dependent FP16 multiplies over 2048 elements takes 5.07 ms, or 9.91 µs per launch — 3.26× MLX’s 3.04 µs (§2.3.5). Captured as a CUDA graph the same chain takes 0.45 ms (0.88 µs per op), recovering 91.2 % of the boundary cost. This reverses the usual framing. NVIDIA has more boundary cost to reclaim than Apple does and a supported mechanism for reclaiming most of it without a persistent kernel; MLX has already amortised its own boundaries through command encoding and offers no equivalent capture. A megakernel on NVIDIA competes with graph capture for the same 91 %; on Apple it competes for 15 µs per block that MLX has largely taken already.

![](images/6ae72cf2be726d76cdc01f27aca530292178f432aa5e36f04805540763dff0e6.jpg)  
dependent chain, % of independent-chain bandwidth

Figure 8: Dependent-chain bandwidth as a share of the independent chain. All runs use 24 square 2048 × 2048 4-bit matvecs with group size 64; the dependent path applies the same interstitial multiply on both platforms.  
![](images/56e25241d699daa57ed0e077dfa02e25a6525d91d02d06dbfd30d1b019005331.jpg)  
Figure 9: Per-boundary cost. CUDA graph capture is hatched because it is a different mechanism, not a peer measurement: it removes boundaries rather than paying for them.

![](images/30787f9062ca48d4b760124848f199fd44fa6973295946b7628ecabb948d1ad3.jpg)  
Figure 10: Output rows delivered for the time one row costs, $8 t ( 1 ) / t ( 8 )$ , for a 4-bit linear at MLP width: eight means a batch of eight is free, one means no weight reuse at all. Apple ranges span the transformer layer shapes of $\ S 2 . 3 \bar { ; }$ the 4090 range spans the two reported Qwen3-4B and Qwen3-8B widths. The 4090’s figure is if anything flattered by its own overhead: its int4 kernel sustains only $5 0 7 \mathrm { G B s ^ { - 1 } }$ at $M { = } 1$ where $\tt f p 1 6$ on the same shape reaches $8 0 0 \mathrm { G B s ^ { - 1 } }$ , so roughly 10 µs of the 24.6 µs baseline is fixed cost, which inflates t(1) and shrinks the ratio.

W4A16 by batch size The consequence for §2.3 is a narrower claim than “Marlin’s premise holds again at $\dot { M } \geq 2 ^ { , , }$ . NVIDIA’s stock 4-bit kernel already serves $M \leq 8$ for almost nothing, so our

<table><tr><td>Shape / kernel</td><td> $M { = } 1$ </td><td> $M { = } 2$ </td><td> $M { = } 4$ </td><td> $M { = } 8$ </td><td> $t ( 8 ) / t ( 1 )$ </td></tr><tr><td> $K { = } 2 5 6 0 , N { = } 9 7 2 8 , \mathbf { f } \mathbf { p } 1 6$ </td><td>74.8μs</td><td>77.8 μs</td><td>75.9 μs</td><td> $7 6 . 8 \mu \mathrm { s }$ </td><td>1.03×</td></tr><tr><td> $K { = } 2 5 6 0 , N { = } 9 7 2 8 , \mathrm { i } \bar { \mathrm { n } } \mathrm { t } 4$ </td><td>24.6μs</td><td>24.6μs</td><td>24.6μs</td><td>27.6μs</td><td>1.12×</td></tr><tr><td> $K { = } 4 0 9 6 , N { = } 1 2 2 8 8 , { \bf f p } 1 6$ </td><td>125.9μs</td><td>131.1μs</td><td>131.1μs</td><td>131.1μs</td><td>1.04×</td></tr><tr><td> $K { = } 4 0 9 6 , N { = } 1 2 2 8 8 , \mathrm { i n t } 4$ </td><td>42.0μs</td><td>42.0μs</td><td>43.0μs</td><td>44.0μs</td><td>1.05×</td></tr></table>

Table 12: RTX 4090, PyTorch’s native int4 kernel as the stock 4-bit path. A batch of eight costs 5 % to 12 % more than a batch of one, against 4.9–5.1× for MLX’s tiled GEMM and $4 . 1 \mathrm { - } 4 . 5 \times$ for ours (§2.3).

small-M path closes a gap specific to MLX’s quantised GEMM rather than recovering an instructionlevel mechanism. The speed-up is real and the mechanism is weight reuse; what it demonstrates is a runtime deficiency, not a vindication of the ported machinery.
<table><tr><td rowspan="2">ctx</td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>SDPA</td><td>FlashInfer</td><td>ratio</td><td>SDPA</td><td>FlashInfer</td><td>ratio</td></tr><tr><td>1k</td><td>73.7μs</td><td>47.6μs</td><td>1.55×</td><td>73.7μs</td><td>48.1 μs</td><td>1.53×</td></tr><tr><td>5k</td><td>77.8μs</td><td>53.2μs</td><td>1.46×</td><td>80.9μs</td><td>53.2μs</td><td>1.52×</td></tr><tr><td>10k</td><td>100.4μs</td><td>74.8μs</td><td>1.34×</td><td>104.4μs</td><td>76.8μs</td><td>1.36×</td></tr><tr><td>16k</td><td>126.0μs</td><td>101.4μs</td><td>1.24×</td><td>124.9μs</td><td>102.4μs</td><td>1.22×</td></tr></table>

Table 13: FlashInfer’s own decode kernel against PyTorch SDPA on the RTX 4090, one query per head, KV caches rotated past L2. The gainfalls with context where the Metal-port/MLX ratio rises on M4 Pro (Fig. 7); different baselines make this a stack-level rather than hardware-only comparison. At 1k both kernels are far below the bandwidth roof $- 5 7 \mathrm { G B s ^ { - 1 } }$ and 88 $\mathrm { G B s ^ { - 1 } }$ against 1008 $\mathrm { G B s ^ { - 1 } }$ peak — so fixed costs and limited occupancy dominate rather than DRAM bandwidth.

## Decode attention

<table><tr><td>Configuration</td><td>tok/s</td><td>vs. autoregressive</td></tr><tr><td>autoregressive</td><td>150.6</td><td></td></tr><tr><td>k = 2</td><td>117.2</td><td>0.778×</td></tr><tr><td>k = 4</td><td>108.1</td><td>0.717×</td></tr><tr><td>k = 7</td><td>105.9</td><td>0.703×</td></tr><tr><td>k = 8</td><td>86.0</td><td>0.571×</td></tr></table>

Table 14: RTX 4090, vLLM, 4-bit Qwen3-8B target with a 4-bit Qwen3-0.6B draft, 32 prompts of 256 generated tokens at max\_num\_seqs=1. $\mathrm { A t } \ k = 7$ this path reaches 0.703×, versus 0.99× with stock speculation on Apple and 1.13× after the small-M Metal kernel accelerates verification (Table 4). The nominally matched configuration therefore changes sign across the tested stacks; differing runtimes and verifier kernels prevent attributing that change to hardware alone.

## Speculative decoding

What we did not measure Two comparisons were attempted and are withheld rather than reported. Marlin itself. vLLM’s marlin\_gemm returned an identical 57.3 µs at every M and at both layer widths — impossible for a bandwidth-bound kernel reading 12.45 MB and 3.3 MB respectively — which indicates a layout mismatch between the packing helper and that entry point. Our harness now checks the kernel’s output against x $W _ { \mathrm { r e f } }$ and refuses to report a timing that does not match, so the NVIDIA side of Table 12 is the stock int4 path only. The comparison against MLX is stock-to-stock, which is the appropriate pairing.

The MoE gather. At batch 1 our NVIDIA comparison issues one kernel per (row, expert) pair, so at $M { = } 8$ it carries 32 launches ≈ 258 µs of the 790 µs measured — both the gather and the grouped-GEMM shape run at $2 0 0 \mathrm { G B s ^ { - 1 } }$ to 260 $\mathrm { G B } \mathrm { s } ^ { - 1 }$ against $9 0 0 \mathrm { G B s ^ { - 1 } }$ achievable, i.e. both are launch-bound and the comparison between them is uninformative. A fused-MoE comparison is future work; the N/A verdict of §2.3.4 rests on the Apple-side prefill result instead.

## 2.3.7 Conclusions

Portability depends on whether the bottleneck and workload survive the move. Marlin’s weight reuse helps at $M > 1$ , making it useful for batching and speculative verification, but not for bandwidthbound M = 1. FlashInfer’s split-KV is more relevant on newer Apple GPUs that can repay its extra dispatch and partials traffic. MoE gather is positive on every tested machine and context because it removes machinery unused at batch 1. Public Metal’s missing asynchronous copy remains a common gap, and the RTX controls show that CUDA outcomes do not reliably predict results on Apple Silicon.

## 2.4 Quantization

Post-training quantization reduces the model’s memory footprint by lowering weight precision after training, trading a small loss in quality for faster inference and lower hardware requirements. We evaluate weight-only post-training quantization of LFM2.5-8B-A1B [28; 45; 46] across three independent toolchains: GGUF/llama.cpp, MLX, and a custom llm-compressor [47] pipeline covering RTN, QuaRot [21]+RTN, and QuaRot+GPTQ [10], all served on MLX kernels. We benchmark latency and throughput (TTFT, prefill, decode, E2E) for all three toolchains on M2 Pro and M3 Ultra. For cross-platform comparison, we additionally run the GGUF configurations on an NVIDIA RTX 4090. Peak memory and downstream quality (MMLU [48; 49], IFBench [50]) are reported once per configuration, including the BF16 baselines.

## 2.4.1 Latency and throughput

Table 15 reports the long-scenario latency and throughput on M2 Pro, and Table 16 on M3 Ultra. Short-scenario results (512-token prompt / 256-token generation) for both chips are in Appendix C. Throughout this section,family refers to a group of quantization methods (GGUF, MLX-native, and RTN/QuaRot), while runtime refers to the engine that executes them — llama.cpp for GGUF and MLX for both MLX-native and RTN/QuaRot configurations. Bolding in Tables 15 and 16 is per runtime.
<table><tr><td>Method</td><td></td><td></td><td>TTFT (ms) ↓ Prefill (tok/s) ↑ Decode (tok/s) ↑ E2E (tok/s) ↑</td><td></td></tr><tr><td>GGUF-Q8_0</td><td> $9 1 4 . 6 \pm 2 . 1$ </td><td> $^ { 1 , 1 2 0 \pm 3 }$ </td><td> $7 9 . 0 \pm 0 . 8$ </td><td> $6 9 . 3 \pm 0 . 6$ </td></tr><tr><td>GGUF-Q6_K</td><td> $1 , 0 2 3 . 4 \pm 1 . 8$ </td><td> $1 { , } 0 0 1 \pm 2$ </td><td> $9 3 . 4 \pm 1 . 0$ </td><td> $7 8 . 8 \pm 0 . 7$ </td></tr><tr><td>GGUF-Q5_K_M</td><td> $1 , 1 0 5 . 4 \pm 2 . 0$ </td><td> $9 2 6 \pm 2$ </td><td> $8 8 . 4 \pm 1 . 0$ </td><td> $7 4 . 3 \pm 0 . 8$ </td></tr><tr><td>GGUF-Q4_K_M</td><td> $9 8 9 . 7 \pm 1 . 8 $ </td><td> $1 , 0 3 5 \pm 2$ </td><td> $1 0 9 . 9 \pm 1 . 5$ </td><td> $9 0 . 8 \pm 1 . 0$ </td></tr><tr><td>GGUF-Q4_0</td><td> ${ \bf 8 6 8 . 1 \pm 1 . 1 }$ </td><td> ${ \bf 1 , 1 8 0 \pm 2 }$ </td><td> ${ \bf 1 2 9 . 3 \pm 1 . 2 }$ </td><td> ${ \bf 1 0 6 . 2 \pm 0 . 8 }$ </td></tr><tr><td>MLX-W8</td><td> $1 , 9 0 1 . 3 \pm 1 3 . 3 $ </td><td> $5 3 9 \pm 2$ </td><td> $8 4 . 0 \pm 0 . 2$ </td><td> $6 3 . 8 \pm 0 . 4$ </td></tr><tr><td>MLX-W4</td><td> $1 , 8 6 8 . 1 \pm 6 . 1 $ </td><td> $5 4 8 \pm 1$ </td><td> $1 1 5 . 9 \pm 0 . 5$ </td><td> $7 9 . 9 \pm 3 . 0$ </td></tr><tr><td>MLX-W4-DWQ</td><td> $1 , 8 6 3 . 0 \pm 3 . 6 $ </td><td> $5 5 0 \pm 1$ </td><td> ${ \bf 1 1 9 . 9 \pm 0 . 5 }$ </td><td> ${ \bf 8 3 . 5 \pm 0 . 3 }$ </td></tr><tr><td>MLX-W4-Affine</td><td> $1 , 8 6 2 . 1 \pm 5 . 7$ </td><td> $5 5 0 \pm 1$ </td><td> $1 1 9 . 8 \pm 0 . 6$ </td><td> $8 2 . 7 \pm 1 . 5$ </td></tr><tr><td>MLX-MXFP4</td><td> $1 , 7 7 7 . 2 \pm 6 . 4$ </td><td> $5 7 6 \pm 1$ </td><td> $1 1 5 . 9 \pm 0 . 5$ </td><td> $8 2 . 7 \pm 0 . 3$ </td></tr><tr><td>MLX-NVFP4</td><td> $1 , 8 1 5 . 7 \pm 4 . 6 $ </td><td> $5 6 4 \pm 1$ </td><td> $1 1 3 . 7 \pm 0 . 3$ </td><td> $8 1 . 1 \pm 0 . 2$ </td></tr><tr><td>RTN-Asym</td><td> ${ \bf 1 } , { \bf 7 1 4 } \pm 4 . 6$ </td><td> ${ \bf 5 9 7 \pm 4 }$ </td><td> $1 1 3 . 4 \pm 0 . 7$ </td><td> $8 2 . 3 \pm 0 . 4$ </td></tr><tr><td>QR+RTN-Asym</td><td> $1 , 7 3 0 . 6 \pm 1 2 . 5$ </td><td> $5 9 2 \pm 4$ </td><td> $1 1 0 . 4 \pm 0 . 8$ </td><td> $7 9 . 2 \pm 3 . 1$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - A s y m }$ </td><td> $1 , 7 3 3 . 9 \pm 9 . 0$ </td><td> $5 9 1 \pm 3$ </td><td> $1 1 0 . 3 \pm 0 . 8$ </td><td> $8 0 . 2 \pm 0 . 6$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - S y m }$ </td><td> $1 , 7 2 8 . 6 \pm 6 . 3$ </td><td> $5 9 2 \pm 2$ </td><td> $1 1 0 . 4 \pm 0 . 7$ </td><td> $8 0 . 1 \pm 1 . 4$ </td></tr><tr><td>QR+GPTQ-Mixed</td><td> $1 , 7 2 8 . 2 \pm 6 . 3 $ </td><td> $5 9 3 \pm 2$ </td><td> $1 1 0 . 3 \pm 0 . 8$ </td><td> $8 0 . 5 \pm 0 . 4$ </td></tr></table>

Table 15: M2 Pro, long scenario (1,024-token prompt / 512-token generation): latency and throughput for LFM2.5-8B-A1B. Bold marks the per-column best within each runtime: llama.cpp for the GGUF rows, MLX for both the MLX and RTN/QuaRot (QR) rows (↓ lower is better, ↑ higher is better). The BF16 checkpoint does not fit in M2 Pro memory, so no baseline is shown. Values are the mean ± standard deviation over repeated runs, with the deviation in grey.

<table><tr><td>Method</td><td>TTFT (ms) ↓</td><td>Prefill  $\mathrm { ( t o k / s ) \uparrow }$ </td><td>Decode (tok/s) ↑</td><td>E2E  $\mathrm { ( t o k / s ) \uparrow }$ </td></tr><tr><td>BF16 (baseline, GGUF)</td><td> $2 8 1 . 7 \pm 0 . 7$ </td><td> ${ 3 , 6 3 6 \pm 9 }$ </td><td> $1 4 0 . 0 \pm 1 . 4$ </td><td> $1 3 0 . 2 \pm 1 . 3$ </td></tr><tr><td>GGUF-Q8_0</td><td> $2 7 8 . 3 \pm 0 . 7$ </td><td> $3 , 6 8 0 \pm 1 0$ </td><td> $1 9 2 . 9 \pm 2 . 7$ </td><td> $1 7 4 . 9 \pm 2 . 2$ </td></tr><tr><td>GGUF-Q6_K</td><td> $2 9 2 . 7 \pm 0 . 5$ </td><td> ${ 3 , 4 9 9 \pm 6 }$ </td><td> $1 9 8 . 1 \pm 2 . 5$ </td><td> $1 7 8 . 2 \pm 2 . 1 $ </td></tr><tr><td>GGUF-Q5_K_M</td><td> $3 0 8 . 6 \pm 0 . 3$ </td><td> $^ { 3 , 3 1 8 \pm 4 }$ </td><td> $1 9 1 . 9 \pm 2 . 5$ </td><td> $1 7 2 . 3 \pm 2 . 0$ </td></tr><tr><td>GGUF-Q4_K_M</td><td> $2 8 2 . 4 \pm 0 . 7$ </td><td> ${ 3 , 6 2 6 \pm 8 }$ </td><td> $2 2 0 . 7 \pm 3 . 2$ </td><td> $1 9 7 . 1 \pm 2 . 6$ </td></tr><tr><td>GGUF-Q4_0</td><td> ${ \bf 2 6 8 . 0 \pm 0 . 3 }$ </td><td> ${ \bf 3 } , { \bf 8 2 0 \pm 5 }$ </td><td> ${ \bf 2 3 2 . 2 \pm } 3 . 3$ </td><td> ${ \bf 2 0 7 . 3 \pm 2 . 7 }$ </td></tr><tr><td>BF16 (baseline, MLX)</td><td> $3 4 4 . 7 \pm 3 . 8$ </td><td> $2 , 9 7 1 \pm 2 0$ </td><td> $1 4 1 . 2 \pm 0 . 2 $ </td><td> $1 2 9 . 2 \pm 0 . 2$ </td></tr><tr><td>MLX-W8</td><td> $3 1 8 . 0 \pm 2 . 2$ </td><td> $3 , 2 2 0 \pm 1 3$ </td><td> $1 9 1 . 4 \pm 0 . 3$ </td><td> $1 6 9 . 7 \pm 2 . 3$ </td></tr><tr><td>MLX-W4</td><td> $3 0 3 . 0 \pm 3 . 6$ </td><td> $^ { 3 , 3 8 0 \pm 2 4 }$ </td><td> $2 4 0 . 4 \pm 0 . 5$ </td><td> $2 0 8 . 4 \pm 4 . 2$ </td></tr><tr><td>MLX-W4-DWQ</td><td> $3 0 0 . 7 \pm 4 . 0$ </td><td> $3 , 4 0 6 \pm 2 7$ </td><td> $2 4 6 . 2 \pm 1 . 5$ </td><td> $2 1 4 . 8 \pm 1 . 3$ </td></tr><tr><td> $\mathrm { M L X - W 4 - A f f u n e }$ </td><td> $2 9 9 . 8 \pm 3 . 1 $ </td><td> $3 , 4 1 6 \pm 2 1$ </td><td> $\mathbf { 2 4 7 . 0 \Sigma } \pm 0 . 7$ </td><td> $2 1 6 . 1 \pm 0 . 5$ </td></tr><tr><td>MLX-MXFP4</td><td> $3 0 5 . 8 \pm 3 . 7$ </td><td> $^ { 3 , 3 4 9 } \pm 2 5$ </td><td> $2 4 5 . 4 \pm 1 . 1$ </td><td> $2 1 4 . 3 \pm 0 . 9$ </td></tr><tr><td>MLX-NVFP4</td><td> $3 0 4 . 0 \pm 3 . 8$ </td><td> $^ { 3 , 3 6 9 } \pm 2 5$ </td><td> $2 4 1 . 1 \pm 1 . 1$ </td><td> $2 1 1 . 2 \pm 0 . 9$ </td></tr><tr><td>RTN-Asym</td><td> ${ \bf 2 6 7 . 5 \pm 3 . 9 }$ </td><td> ${ \bf 3 } , { \bf 8 2 9 \pm 5 5 }$ </td><td> $2 4 6 . 8 \pm 0 . 8$ </td><td> ${ \bf 2 1 7 . 7 \pm 3 . 5 }$ </td></tr><tr><td> $\mathrm { Q R + R T N - A s y m }$ </td><td> $2 7 0 . 6 \pm 4 . 7$ </td><td> $3 , 7 8 6 \pm 6 4$ </td><td> $2 3 8 . 3 \pm 1 . 1$ </td><td> $2 0 8 . 9 \pm 8 . 1$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - A s y m }$ </td><td> $2 6 9 . 7 \pm 3 . 1$ </td><td> $3 , 7 9 7 \pm 4 3$ </td><td> $2 3 8 . 9 \pm 1 . 1 $ </td><td> $2 1 2 . 1 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - S y m }$ </td><td> $2 7 0 . 1 \pm 4 . 0$ </td><td> $3 , 7 9 2 \pm 5 5$ </td><td> $2 3 6 . 6 \pm 1 . 4$ </td><td> $2 0 9 . 2 \pm 3 . 8$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - M i x e d }$ </td><td> $2 6 9 . 5 \pm 3 . 4$ </td><td> $3 , 8 0 0 \pm 4 8$ </td><td> $2 3 6 . 5 \pm 1 . 4$ </td><td> $2 1 0 . 7 \pm 0 . 9$ </td></tr></table>

Table 16: M3 Ultra, long scenario (1,024-token prompt / 512-token generation): latency and throughput for LFM2.5-8B-A1B. Conventions as in Table 15; each runtime has its own BF16 baseline.

![](images/d708a4020dca3e647b9f62cbace7846e3938bdc11990199766af2c9724bfe74f.jpg)

![](images/f08f625408addbab0b9ed3c6b24ab7a45a472ecaaebf109b7f8f504ed1c183cc.jpg)  
Figure 11: M2 Pro: peak memory footprint versus prefill (left) and decode (right) throughput for LFM2.5-8B-A1B, long scenario. The x-axis is reversed, so points further right use less memory and the upper-right corner is best. Full data in Table 15.

Decode gains depend on bandwidth headroom. Figure 11 shows the memory–throughput tradeoff on M2 Pro. On both chips, decode throughput rises as bit-width drops, but the extent depends on how memory-bound the chip already is. On M2 Pro, GGUF decode improves from 79.0 tok/s at Q8\_0 to 129.3 tok/s at Q4\_0 (+63.7%), and MLX’s 4-bit variants reach 113.7–119.9 tok/s. M2 Pro cannot hold the BF16 checkpoint at all, so here quantization is a precondition for running the model rather than an optimization. On M3 Ultra, GGUF-Q4\_0 and MLX-W4-Affine improve decode by +65.9% and +74.9% over their BF16 baselines. Relative to 8-bit, however, M3 Ultra gains only 20.4%, compared with 63.7% on the more bandwidth-constrained M2 Pro.

Prefill gains less from quantization than decode. Prefill benefits far less from lower bit-width than decode (Figure 11, left, for M2 Pro). Moving from 8- to 4-bit raises MLX-native prefill by only 2–7% on M2 Pro (539 to 548–576 tok/s) and 4–6% on M3 Ultra (3,220 to 3,349–3,416 tok/s), against 20–64% for decode, consistent with prefill being compute-bound rather than bandwidthbound. GGUF prefill, in contrast, depends on the quantization format rather than on the number of bits. All three k-quant formats are slower than Q8\_0 on both chips and in both scenarios: on M2 Pro, Q6\_K, Q5\_K\_M, and Q4\_K\_M reach 1,001, 926, and 1,035 tok/s against 1,120 tok/s for Q8\_0 (up to −17%), and on M3 Ultra 3,499, 3,318, and 3,626 against 3,680 tok/s (up to −10%). Only the Q4\_0 format exceeds Q8\_0 (+5% on M2 Pro, +4% on M3 Ultra). Section 2.4.2 shows that this k-quant slowdown is largely absent on the RTX 4090.

The best runtime depends on the chip. On M2 Pro, GGUF-Q4\_0 is the clear choice. It has the lowest TTFT, the highest prefill, decode, and E2E throughput, and the smallest memory footprint of all configurations. Its prefill is about 2× that of the fastest MLX-runtime configuration, RTN-Asym (1,180 vs. 597 tok/s), and its decode is 8% higher than that of the best MLX-native variant, MLX-W4-DWQ (129.3 vs. 119.9 tok/s). On M3 Ultra, the two runtimes are close. GGUF-Q4\_0 leads the MLX-native formats on TTFT and prefill (3,820 vs. 3,416 tok/s for MLX-W4-Affine, +12%), while MLX-W4-Affine leads on decode (247.0 vs. 232.2 tok/s, +6%). Our RTN-Asym checkpoint, also served on MLX, matches GGUF-Q4\_0 on TTFT and prefill within measurement noise (267.5 vs. 268.0 ms and 3,829 vs. 3,820 tok/s) and has the highest E2E throughput on the chip (217.7 tok/s).

Rankings within each family. GGUF-Q4\_0 leads the GGUF family on every metric, on both chips and in both scenarios. The MLX-native 4-bit variants are nearly tied. In the long scenario, MLX-MXFP4 leads TTFT and prefill on M2 Pro and MLX-W4-DWQ leads decode and E2E, while on M3 Ultra MLX-W4-Affine leads all four. In the short scenario (Appendix Tables 20 and 21), the ranking on M2 Pro is largely unchanged: MLX-MXFP4 remains best on TTFT and prefill, while MLX-W4-DWQ and MLX-W4-Affine tie on decode (121.5 tok/s) and differ by 0.1 tok/s on E2E. On M3 Ultra, the best variant differs per metric, and the gaps between them fall within one standard deviation, so their ranking is not robust. Within the RTN/QuaRot family, plain RTN-Asym is fastest on every metric on both chips and in both scenarios. In our weight-only setting, QuaRot’s rotations are pure overhead: because RTN and GPTQ differ only during calibration, rotation is the primary runtime variable, and it costs 2.4–4.2% of decode and about 1% of long-scenario prefill relative to RTN-Asym. This is expected, since Hadamard transforms yield speedups only when paired with low-precision activation kernels, which this setting does not use.

## 2.4.2 Cross-platform comparison

To test which of these trends are specific to Apple Silicon, we run the GGUF configurations on an NVIDIA RTX 4090 and compare them with both Apple chips (Figure 12). RTX 4090-only results are in Appendix Figure 14.

The RTX 4090 is fastest at every quantization level. The ordering RTX 4090 > M3 Ultra > M2 Pro holds at every level for both prefill and decode (Figure 12). At Q4\_0, the 4090’s decode (469.1 tok/s) is 2.0× that of M3 Ultra (232.2 tok/s) and 3.6× that of M2 Pro (129.3 tok/s). The gap is far wider for prefill: at Q8\_0, the 4090 reaches 13,308 tok/s, 3.6× M3 Ultra and 11.9× M2 Pro.

Relative decode gains differ across devices. Normalized to each device’s own Q8\_0 (Figure 13), Q4\_0 improves decode by +63.7% on M2 Pro, +38.2% on the RTX 4090, and +20.4% on M3 Ultra. M2 Pro, the most bandwidth-constrained device, benefits the most. M3 Ultra gains the least, even less than the RTX 4090, although both have high memory bandwidth. This suggests that a larger share of its decode time goes to costs that do not depend on bit-width, such as per-token scheduling overheads in llama.cpp’s Metal backend.

![](images/aff6334a118e2414055237720a047b4cf482803389cf65c43858555b09d93acd.jpg)  
(a) Prefill throughput

![](images/32c2064f034c08c5f4f0e9b27d3db78b604a61d52da053c1d72a6bf1f3825366.jpg)  
(b) Decode throughput  
Figure 12: GGUF prefill and decode throughput versus peak memory across M2 Pro, M3 Ultra, and RTX 4090. M2 Pro has no BF16 point, as the checkpoint does not fit in its memory. Error bars show standard deviation over 10 repeated runs.

![](images/0685a62ed3c4561e348abf2fe62032ce49f7960c88d956decfea3cf5c52871d4.jpg)  
(a) Prefill throughput

![](images/4eaab15c76ca193831a1ef7da68868f27c644f583515e2fd2df76fbacbf25d11.jpg)  
(b) Decode throughput  
Figure 13: Same data as Figure 12, each device’s own Q8\_0 throughput normalized to 1.0. Q8\_0 is used as the common reference rather than BF16 because M2 Pro has no BF16 measurement; this normalization isolates the effect of moving from 8-bit to lower bit-widths on each device, independent of each device’s absolute throughput.

The k-quant prefill slowdown is specific to Apple Silicon. On the RTX 4090, Q5\_K\_M prefill is essentially flat relative to Q8\_0 (+0.9%), whereas it falls by 17.3% on M2 Pro and 9.8% on M3 Ultra (§2.4.1). This suggests that the slowdown reflects how llama.cpp’s Metal backend executes k-quant formats rather than a property of the formats themselves.

The RTX 4090 shows a prefill jump at 8 bits. Its prefill throughput rises 1.75× from BF16 to Q8\_0, from 7,602 to 13,308 tok/s (Appendix Figure 14a). On M3 Ultra, the same step barely changes prefill (3,636 to 3,680 tok/s, 1.01×). This jump is consistent with Ada Lovelace’s INT8 tensor-core paths engaging only once weights are quantized to 8 bits, a mechanism with no direct analog on Apple’s GPU architecture.

## 2.4.3 Memory footprint

<table><tr><td>Method</td><td>Bits</td><td>Model memory (MB) ↓</td><td>Peak (MB) ↓</td></tr><tr><td>BF16 (baseline, GGUF)</td><td>16</td><td>17,463</td><td>17,551</td></tr><tr><td>GGUF Q8_0</td><td>8</td><td>9,280 (-46.9%)</td><td>9,368 (-46.6%)</td></tr><tr><td>GGUF Q6_K</td><td>6</td><td>7,167 (-59.0%)</td><td>7,254 (-58.7%)</td></tr><tr><td>GGUF Q5_K_M</td><td>5</td><td>6,237 (-64.3%)</td><td>6,325 (-64.0%)</td></tr><tr><td>GGUF Q4_K_M</td><td>4</td><td>5,362 (-69.3%)</td><td>5,450 (-68.9%)</td></tr><tr><td>GGUF Q4_0</td><td>4</td><td>5,051 (-71.1%)</td><td>5,139 (-70.7%)</td></tr><tr><td>BF16 (baseline, MLX)</td><td>16</td><td>16,816</td><td>18,141</td></tr><tr><td>MLX-W8</td><td>8</td><td>9,079 (-46.0%)</td><td>10,336 (-43.0%)</td></tr><tr><td>MLX-W4</td><td>4</td><td>4,815 (-71.4%)</td><td>6,302 (-65.3%)</td></tr><tr><td>MLX-W4-DWQ</td><td>4</td><td>4,911 (−70.8%)</td><td>6,192 (-65.9%)</td></tr><tr><td>MLX-W4-affine</td><td>4</td><td>4,916 (−70.8%)</td><td>6,195 (-65.8%)</td></tr><tr><td>MLX-MXFP4</td><td>4</td><td>4,633 (−72.4%)</td><td>5,930 (-67.3%)</td></tr><tr><td>MLX-NVFP4</td><td>4</td><td>4,821 (-71.3%)</td><td>6,103 (-66.4%)</td></tr><tr><td>RTN-W4-asym</td><td>4</td><td>4,600 (−72.6%)</td><td>6,630 (-63.5%)</td></tr><tr><td>QuaRot+RTN-W4-asym</td><td>4</td><td>4,610 (−72.6%)</td><td>6,630 (−63.5%)</td></tr><tr><td>QuaRot+GPTQ-W4-asym</td><td>4</td><td>4,610 (−72.6%)</td><td>6,630 (-63.5%)</td></tr><tr><td>QuaRot+GPTQ-W4-sym</td><td>4</td><td>4,610 (−72.6%)</td><td>6,630 (–63.5%)</td></tr><tr><td>QuaRot+GPTQ-W4-mixed</td><td>4</td><td>4,610 (-72.6%)</td><td>6,640 (-63.4%)</td></tr></table>

Table 17: Memory footprint across all evaluated quantization methods for LFM2.5-8B-A1B. Bold marks the per-column best within each runtime; percentages are relative to that runtime’s BF16 baseline. GGUF’s KV cache is omitted because it is effectively constant across quantization levels (12.6 MB at 512 tokens, 22.0 MB at 1,024 tokens).

Table 17 reports model and peak memory for every configuration. Memory footprint decreases roughly in proportion to bit-width across all methods (GGUF-Q4\_0: −70.7%, MLX-MXFP4: −67.3%). Within each runtime’s native family, the method with the lowest peak memory also has the smallest weights (GGUF-Q4\_0 and MLX-MXFP4). RTN/QuaRot’s peak memory (6,630–6,640 MB) is notably higher than that of any MLX 4-bit variant (5,930–6,302 MB), despite comparable or smaller model-memory footprints (4,600–4,610 MB vs. 4,633–4,916 MB). The gap between model memory and peak memory is roughly 2,020–2,030 MB for RTN/QuaRot, compared with 1,279–1,487 MB for MLX’s 4-bit variants. This points to a larger transient memory overhead in our RTN/QuaRot pipeline, beyond the quantized weights themselves. Because plain RTN-Asym, which uses neither rotation nor calibration, shows the same gap, the overhead likely arises from how the pipeline loads or converts checkpoints for MLX rather than from QuaRot itself.

## 2.4.4 Downstream quality

Table 18 reports MMLU and IFBench scores for every configuration. All MMLU scores carry a 95% confidence interval of ±0.8 points, and all IFBench scores a 95% confidence interval of ±5.6 points, reflecting sampling uncertainty from the fixed evaluation-set size rather than run-to-run measurement variance. These intervals are constant across methods and are therefore omitted from the table.
<table><tr><td>Method</td><td>MMLU↑</td><td>IFBench ↑</td><td>Average ↑</td></tr><tr><td>BF16 (baseline, GGUF)</td><td>55.9</td><td>55.7</td><td> $5 5 . 8 \ ( + 0 . 0 \% )$ </td></tr><tr><td>GGUF Q8_0</td><td> $5 5 . 7 \ : ( - 0 . 3 \% )$ </td><td> $5 8 . 3 \ ( + 4 . 8 \% )$ </td><td> $5 7 . 0 \ ( + 2 . 2 \% )$ </td></tr><tr><td>GGUF Q6_K</td><td> ${ \bf 5 6 . 0 _ { \mathrm { ~ ( + 0 . 1 \% ) } } }$ </td><td> ${ \bf 6 0 . 7 _ { \mathrm { ~ ( + 9 . 0 \% ) ~ } } }$ </td><td> $5 8 . 4 \ ( + 4 . 6 \% )$ </td></tr><tr><td> $\mathrm { G G U F Q 5 \_ K \_ M }$ </td><td> $5 5 . 6 \ : ( - 0 . 5 \% )$ </td><td> $5 5 . 0 \ : ( - 1 . 3 \% )$ </td><td> $5 5 . 3 \ ( - 0 . 9 \% )$ </td></tr><tr><td>GGUF Q4_K_M</td><td> $5 4 . 2 \ ( - 3 . 0 \% )$ </td><td> $5 5 . 0 \ : ( - 1 . 3 \% )$ </td><td> $5 4 . 6 ~ ( - 2 . 2 \% )$ </td></tr><tr><td>GGUF Q4_0</td><td> $5 4 . 0 \ : ( - 3 . 4 \% )$ </td><td> $5 3 . 3 \ ( - 4 . 3 \% )$ </td><td> $5 3 . 7 \ ( - 3 . 9 \% )$ </td></tr><tr><td>BF16 (baseline, MLX)</td><td>55.68</td><td>51.67</td><td> $5 3 . 7 \ ( + 0 . 0 \% )$ </td></tr><tr><td>MLX-W8</td><td> ${ \bf 5 5 . 7 1 _ { \mathrm { ~ ( + 0 . 1 \% ) } } }$ </td><td> $5 2 . 6 7 \ ( + 1 . 9 \% )$ </td><td> $5 4 . 2 \ ( + 1 . 0 \% )$ </td></tr><tr><td>MLX-W4</td><td> $5 3 . 4 8 ~ ( - 4 . 0 \% )$ </td><td> ${ \bf 5 7 . 3 3 _ { \mathrm { \ ( + 1 1 . 0 \% ) } } }$ </td><td> ${ \bf 5 5 . 4 } _ { \mathrm { ~ ( + 3 . 2 \% ) ~ } }$ </td></tr><tr><td> $\mathrm { M L X - W 4 - D W Q }$ </td><td> $5 4 . 5 1 \ ( - 2 . 1 \% )$ </td><td> $5 2 . 3 3 \ : ( + 1 . 3 \% )$ </td><td> $5 3 . 4 \ ( - 0 . 5 \% )$ </td></tr><tr><td> $\mathrm { M L X - W 4 - a f f u n e }$ </td><td> $5 3 . 2 0 \ ( - 4 . 5 \% )$ </td><td> $5 3 . 3 3 \ ( + 3 . 2 \% )$ </td><td> $5 3 . 3 \ ( - 0 . 8 \% )$ </td></tr><tr><td> $\mathbf { M L X - M X F P 4 }$ </td><td> $5 2 . 3 5 \ : ( - 6 . 0 \% )$ </td><td> $5 2 . 6 7 \ ( + 1 . 9 \% )$ </td><td> $5 2 . 5 \ ( - 2 . 2 \% )$ </td></tr><tr><td> $\mathrm { M L X - N V F P 4 }$ </td><td> $5 2 . 5 4 \ ( - 5 . 6 \% )$ </td><td> $5 5 . 6 7 \ ( + 7 . 7 \% )$ </td><td> $5 4 . 1 \ ( + 0 . 8 \% )$ </td></tr><tr><td>RTN-W4-asym</td><td> $5 2 . 9 6 ~ ( - 4 . 9 \% )$ </td><td> $4 9 . 0 0 \ ( - 5 . 2 \% )$ </td><td> $5 1 . 0 \ ( - 5 . 0 \% )$ </td></tr><tr><td> $\mathrm { Q u a R o t + R T N - W 4 - a s y m }$ </td><td> $5 1 . 8 8 ( - 6 . 8 \% )$ </td><td> $4 2 . 0 0 ( - 1 8 . 7 \% )$ </td><td> $4 6 . 9 \ ( - 1 2 . 5 \% )$ </td></tr><tr><td> $\mathrm { Q u a R o t + G P T Q - W 4 - a s y m }$ </td><td> $5 3 . 0 9 \ ( - 4 . 7 \% )$ </td><td> $4 1 . 0 0 ( - 2 0 . 7 \% )$ </td><td> $4 7 . 0 \ : ( - 1 2 . 4 \% )$ </td></tr><tr><td> $\mathrm { Q u a R o t + G P T Q - W 4 - s y m }$ </td><td> $5 2 . 9 4 \ ( - 4 . 9 \% )$ </td><td> $3 9 . 6 0 \ ( - 2 3 . 4 \% )$ </td><td> $4 6 . 3 \ : ( - 1 3 . 8 \% )$ </td></tr><tr><td> $\mathrm { Q u a R o t + G P T Q - W 4 \mathrm { - } m i x e d }$ </td><td> $5 3 . 5 0 ( - 3 . 9 \% )$ </td><td> $4 0 . 3 0 ( - 2 2 . 0 \% )$ </td><td> $4 6 . 9 \ ( - 1 2 . 6 \% )$ </td></tr></table>

Table 18: Downstream quality (MMLU, IFBench) across all evaluated quantization methods for LFM2.5-8B-A1B (↑ higher is better). Bold indicates the best value within each runtime. Percentages are relative to the BF16 runtime baseline for each method (MLX for both MLX-native and RTN/QuaRot). Average is the unweighted mean of MMLU and IFBench.

GGUF-Q6\_K has the highest average quality in either runtime (58.4), and GGUF-Q8\_0 also scores above its BF16 baseline on average. Their gains, however, fall within the confidence intervals (+0.1 MMLU and +5.0 IFBench points for Q6\_K), so these configurations match rather than exceed BF16. On the MLX side, MLX-W4 achieves the best average (+3.2%), driven by an IFBench gain of 5.7 points, at the edge of the confidence interval, that outweighs its MMLU loss of 2.2 points. Across both runtimes, the GGUF and MLX-native 4-bit variants lose 1.2–3.3 MMLU points relative to their BF16 baseline. The RTN/QuaRot family behaves differently on the two benchmarks. On MMLU, its scores (51.9–53.5) are comparable to those of the MLX-native 4-bit variants (52.4–54.5). On IFBench, every RTN/QuaRot variant scores below every GGUF and MLX-native configuration. Plain RTN-W4-asym loses 2.7 points against the MLX BF16 baseline, within the confidence interval, whereas the four QuaRot variants lose 9.7–12.1 points. Although IFBench carries a much wider confidence interval than MMLU (±5.6 vs. ±0.8 points), the QuaRot drop exceeds it and is consistent across all four variants, which argues against the effect being noise.

## 2.5 Conclusions

As with the kernel results in §2.3, a reported quantization benefit does not automatically carry over across platforms. The same Q8\_0-to-Q4\_0 step speeds up GGUF decode by 1.64× on M2 Pro but only 1.20× on M3 Ultra, and the k-quant prefill slowdown on both Apple chips is largely absent on the RTX 4090. Without low-precision activation kernels, QuaRot’s rotations are pure overhead, and in our pipeline they also reduce IFBench quality. The best choice is chip-dependent: GGUF-Q4\_0 dominates on M2 Pro, while on M3 Ultra it leads on prefill and MLX’s affine/DWQ variants lead on decode.

## 3 Discussion and Conclusion

Across four optimization families, no single verdict holds uniformly. Each technique’s fate on Apple Silicon depends on a specific, identifiable mismatch between its assumptions and our batch-1, unified-memory regime.

KV-cache pruning helps only when the selection cost is low relative to the attention it removes. Profit grows with output length, context, and architecture, and reverses on NVIDIA GPUs. Speculative decoding produced no standardized E2E gain in the principal long-prompt chat benchmark. Draft–verify overhead and prefill penalties offset its decode-side benefits, although separate configurations showed positive results. Metal kernel ports succeed only where the workload matches their design. Marlin’s weight reuse pays off only once more than one token decodes at a time, useful for batching and draft verification but not single-token chat; FlashInfer’s split-KV attention earns back its overhead only on newer Apple GPUs; and the MoE gather helps everywhere, since it removes scheduling work batch-one decoding never needed. Quantization gains reflect how memory-bound the chip already is, rather than bit-width itself. Neither GGUF nor MLX is better in all scenarios. Generally, GGUF is faster on prefill, but MLX has better decode speed, so the best choice depends on which phase matters more for the workload.

The unifying result is that a reported speedup is not a portable property of a technique alone, but a joint function of technique, runtime, hardware contract, and workload regime. Practitioners porting CUDA-validated techniques to Apple Silicon should identify which bottleneck a mechanism actually removes before expecting a number.

Limitations. Our KV-pruning lacks task-quality evaluation, and the megakernel study covers a single model on a single chip. Extending this analysis to multi-request serving, where batching gives several currently idle mechanisms something to act on, is left to future work.

## References

[1] Georgi Gerganov et al. llama.cpp: LLM inference in C/C++. Software repository, https: //github.com/ggml-org/llama.cpp. Accessed 2026-09-04.

[2] Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. MLX: Efficient and flexible machine learning on apple silicon. https://github.com/ml-explore/mlx, 2023.

[3] Abdurrahman Javat and Allan Kazakov. Silicon showdown: Performance, efficiency, and ecosystem barriers in consumer-grade LLM inference. arXiv preprint arXiv:2605.00519, 2026. URL https://arxiv.org/abs/2605.00519.

[4] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023.

[5] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 19274–19286. PMLR, 2023. URL https://proceedings.mlr.press/v202/leviathan23a.html.

[6] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[7] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. FlashInfer: Efficient and customizable attention engine for LLM inference serving. In Proceedings of Machine Learning and Systems (MLSys), 2025.

[8] Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models, August 2024. URL http://arxiv.org/abs/2408.11743. arXiv:2408.11743 [cs.LG].

[9] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. MegaBlocks: Efficient sparse training with mixture-of-experts. In Proceedings of Machine Learning and Systems (MLSys), 2023.

[10] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, March 2023. URL http: //arxiv.org/abs/2210.17323. arXiv:2210.17323 [cs.LG].

[11] Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, and Mohammad Rastegari. OpenELM: An efficient language model family with open training and inference framework. arXiv preprint arXiv:2404.14619, 2024. doi: 10.48550/arXiv.2404.14619. URL https://arxiv.org/abs/2404.14619.

[12] Varun Rajesh, Om Jodhpurkar, Pooja Anbuselvan, Mantinder Singh, Ashok Jallepali, Shantanu Godbole, Pradeep Kumar Sharma, and Hritvik Shrivastava. Production-grade local LLM inference on apple silicon: A comparative study of MLX, MLC-LLM, ollama, llama.cpp, and PyTorch MPS. arXiv preprint arXiv:2511.05502, 2025. doi: 10.48550/arXiv.2511.05502. URL https://arxiv.org/abs/2511.05502.

[13] MLX Team. MLX-LM: Generate text with large language models and MLX. Software repository, 2026. URL https://github.com/ml-explore/mlx-lm. Accessed 2026-08-19.

[14] NVIDIA. KVPress: LLM KV-cache compression made easy. Software repository, 2026. URL https://github.com/NVIDIA/kvpress. Accessed 2026-08-19.

[15] Alessio Devoto, Maximilian Jeblick, and Simon Jégou. Expected attention: KV cache compression by estimating attention from future queries distribution. arXiv preprint arXiv:2510.00636, 2025. URL https://arxiv.org/abs/2510.00636.

[16] Jang-Hyun Kim, Dongyoon Han, and Sangdoo Yun. Fast KVzip: Efficient and accurate LLM inference with gated KV eviction. arXiv preprint arXiv:2601.17668, 2026. URL https: //arxiv.org/abs/2601.17668.

[17] Ziyi Chen, Xiaocong Yang, Jiacheng Lin, Chenkai Sun, Kevin Chen-Chuan Chang, and Jie Huang. Cascade speculative drafting for even faster LLM inference. In Advances in Neural Information Processing Systems, volume 37, pages 86226–86242, 2024. doi: 10.52202/ 079017-2738.

[18] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Advances in Neural Information Processing Systems, volume 38, pages 151568–151587, 2025. doi: 10.52202/085713-4562.

[19] Jian Chen, Yesheng Liang, and Zhijian Liu. DFlash: Block diffusion for flash speculative decoding. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research. PMLR, 2026.

[20] Liran Ringel and Yaniv Romano. Accelerating speculative decoding with block diffusion draft trees. arXiv preprint arXiv:2604.12989, 2026. URL https://arxiv.org/abs/2604.12989.

[21] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4- Bit Inference in Rotated LLMs, October 2024. URL http://arxiv.org/abs/2404.00456. arXiv:2404.00456 [cs.LG].

[22] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626. Association for Computing Machinery, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613165.

[23] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-ofthe-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, 2020. URL https://aclanthology.org/2020.emnlp-demos.6/.

[24] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024. URL https://arxiv.org/abs/ 2404.14469.

[25] Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. A simple and effective $L _ { 2 }$ norm-based strategy for KV cache compression. arXiv preprint arXiv:2406.11430, 2024. URL https://arxiv.org/abs/2406.11430.

[26] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=NG7sS51zVF.

[27] Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi, and Roy Schwartz. Transformers are multi-state RNNs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18724–18741. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.1043. URL https://aclanthology.org/2024. emnlp-main.1043/.

[28] Liquid AI. Lfm2.5-8b-a1b: Personal assistant on your laptop. Liquid AI Blog, 2026. www.liquid.ai/blog/lfm2-5-8b-a1b.

[29] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https: //arxiv.org/abs/2505.09388.

[30] Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. URL https://arxiv.org/abs/2503.19786.

[31] IBM Granite Team. Granite-4.0-h-micro model card. Hugging Face model documentation, 2025. URL https://huggingface.co/ibm-granite/granite-4.0-h-micro. Accessed 2026-08-19.

[32] Liquid AI Team. LFM2 technical report. arXiv preprint arXiv:2511.23404, 2025. URL https://arxiv.org/abs/2511.23404.

[33] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html.

[34] Z Lab. Qwen3-4B-DFlash-b16. Hugging Face model repository, 2026. URL https:// huggingface.co/z-lab/Qwen3-4B-DFlash-b16. Accessed 2026-08-19.

[35] NVIDIA. CUDA C++ programming guide. https://docs.nvidia.com/cuda/ cuda-c-programming-guide/, 2025.

[36] NVIDIA. Parallel thread execution ISA. https://docs.nvidia.com/cuda/ parallel-thread-execution/, 2025.

[37] Apple Inc. Metal shading language specification. https://developer.apple.com/metal/ Metal-Shading-Language-Specification.pdf, 2025.

[38] Qwen Team. Qwen3 technical report, 2025.

[39] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

[40] Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov. Flash-decoding for long-context inference, 2023. URL https://crfm.stanford.edu/2023/10/12/flashdecoding.html. Stanford CRFM blog.

[41] NVIDIA. CUTLASS: CUDA templates for linear algebra subroutines. https://github. com/NVIDIA/cutlass, 2024.

[42] Noam Shazeer. GLU variants improve transformer, 2020.

[43] Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B. URL https: //hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles.

[44] Mengdi Wu, Xinhao Cheng, Shengyu Liu, Chunan Shi, Jianan Ji, Kit Ao, Praveen Velliengiri, Xupeng Miao, Oded Padon, and Zhihao Jia. Mirage: A multi-level superoptimizer for tensor programs, 2024. The Mirage Persistent Kernel (MPK) compiles an LLM forward pass into a single megakernel.

[45] Liquid AI. Lfm2 technical report. arXiv preprint arXiv:2511.23404, 2025.

[46] Jimmy T.H. Smith, Tarek Dakhran, Alberto Cabrera, Simon S. Lee, Paul Pak, Aditya Tadimeti, Tim Seyde, Maxime Labonne, Alexander Amini, and Mathias Lechner. In-place tokenizer expansion for pre-trained llms. arXiv preprint arXiv:2607.15232, 2026.

[47] Red Hat AI and vLLM Project. LLM Compressor, 8 2024. URL https://github.com/ vllm-project/llm-compressor.

[48] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR), 2021.

[49] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. Proceedings ofthe International Conference on Learning Representations (ICLR), 2021.

[50] Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. Generalizing verifiable instruction following, 2025.

## A Additional Speculative-Decoding Results

This appendix records the implementation settings and supplementary 8B sensitivity experiment. The method review, acceptance/cost analysis, limiting factors, and configuration-specific positive cases appear in Section 2.2.

## A.1 Implementation and benchmark configuration

The principal comparison uses MLX with batch size one and concurrency one on M2 Pro and M3 Ultra. Targets are mlx-community/Qwen3-4B-4bit and BF16 Qwen/Qwen3-4B. The 100-prompt conversational dataset contains 46 tool-calling and 54 regular-chat samples, with a mean prefill length of 1,776 tokens. Baseline completions average 61.9 tokens with the 4-bit target and 54.2–55.0 tokens with BF16. Generation is greedy (T = 0), with a 512-token limit, one warmup, and three measured repetitions per prompt and method. Method-specific settings are held fixed across the two devices and target precisions unless stated below.

Classic. The auxiliary autoregressive drafter is mlx-community/Qwen3-0.6B-4bit for both target precisions. The draft width is four tokens, the fastest setting in our tested configurations.

N-gram and Cascade-n-gram. N-gram selects the longest matching stored suffix from a precomputed lookup table, with match lengths from three to eight tokens, and proposes two tokens per round. Cascade-n-gram uses the same table to speculate for the 0.6B-4bit model; its verified predictions then serve as drafts for the target. The target-level draft width remains two tokens. This implementation evaluates a vertical N-gram cascade, rather than the full vertical and horizontal Max-Gram cascade of Chen et al. [17].

EAGLE-3. The community head AngelSlim/Qwen3-4B\_eagle3 is converted to MLX and quantized to four bits, including for the BF16-target runs. It takes hidden states from target layers 2, 18, and 33. One token is drafted per step, the fastest tested setting; this local configuration differs from the wider dynamic-tree settings used to obtain the published speedups.

DFlash and DDTree. Both use the BF16 checkpoint z-lab/Qwen3-4B-DFlash-b16 [34]. DFlash’s 16-position block contains the target-produced root and up to 15 draft candidates. DDTree builds a tree with a budget of 24 non-root nodes from the same block drafter, the best fixed budget in the implementation’s local sweep. Both use greedy verification.

Baseline and acceptance conventions. Each adaptation is compared with its own autoregressive baseline. Standardized E2E throughput uses the corresponding baseline token workload, so speculative output-length changes do not redefine the workload used for this metric. Percentage changes are computed relative to these paired baselines; absolute rates across different implementations are not a controlled cross-method comparison. In Figure 6(c), AL counts accepted draft tokens and excludes the target-produced root for DFlash and DDTree. DT denotes the configured budget: four, two, two, and one draft tokens for Classic, N-gram, Cascade-n-gram, and EAGLE-3; a 16-position block including the root for DFlash; and 24 non-root tree nodes for DDTree. These differing budget definitions must be respected when interpreting AL. The committed-token count τ in Equation 2 includes target-produced progress as applicable and is not the same metric as AL.

## A.2 Additional results for Qwen3-8B-4bit

Table 19: Standardized end-to-end (E2E), prefill, and decode throughput changes relative to each implementation’s autoregressive baseline for Qwen3-8B-4bit on M2 Pro and M3 Ultra.
<table><tr><td></td><td colspan="3">M2 Pro</td><td colspan="3">M3 Ultra</td></tr><tr><td>Method</td><td>E2E</td><td>Prefill</td><td>Decode</td><td>E2E</td><td>Prefill</td><td>Decode</td></tr><tr><td>Classic</td><td>-14.4%</td><td>-9.1%</td><td>-38.6%</td><td>-16.9%</td><td>-15.1%</td><td>-22.2%</td></tr><tr><td>N-gram</td><td>-4.6%</td><td>-0.4%</td><td>-25.6%</td><td>-1.4%</td><td>-0.4%</td><td>-4.4%</td></tr><tr><td>Cascade-n-gram -12.6%</td><td></td><td>-9.8%</td><td>-27.2%-10.2%</td><td></td><td>-10.4%</td><td>-9.6%</td></tr><tr><td>EAGLE-3</td><td>-14.6%</td><td>-10.5%</td><td>-34.3%</td><td>6-12.5%</td><td>-9.7%</td><td>-20.4%</td></tr><tr><td>DFlash</td><td>-28.4% -10.0% -69.4%</td><td></td><td></td><td>6-29.5%</td><td>-8.7%</td><td>-59.5%</td></tr><tr><td>DDTree</td><td>-23.8% -10.0% -62.0% -26.6%</td><td></td><td></td><td></td><td></td><td>-8.7%-55.1%</td></tr></table>

To test whether a more expensive target step makes speculation more favorable, we repeat the M2 Pro and M3 Ultra comparison with Qwen/Qwen3-8B-MLX-4bit. The learned checkpoints are replaced by AngelSlim/Qwen3-8B\_eagle3 and z-lab/Qwen3-8B-DFlash-b16; the Classic drafter and other method parameters are unchanged. The workload, decoding protocol, and paired-baseline normalization follow the principal comparison.

Increasing the target from 4B to 8B improves relative E2E performance for most configurations, but no method achieves an E2E gain. N-gram comes closest on M3 Ultra (−1.4%). Decode changes are method-dependent: EAGLE-3 becomes slower relative to its baseline on both devices, whereas DFlash and DDTree modestly reduce their penalties. The larger target therefore provides additional amortization headroom without eliminating the drafting, verification, and prefill costs identified in the main text.

## B Appendix Metal

## B.0.1 Marlin: 6-bit lm\_head

A 6-bit matvec for the tied lm\_head, using a 4+2 bit-plane layout, measured parity with stock MLX (835 µs versus 844 µs). MLX sustained 252 GB s<sup>−1</sup>–264 GB s<sup>−1</sup> on that shape at 4, 6 and 8 bits alike, showing that unpack cost was not the limiting resource: the shape is rate-capped, so a cheaper unpack has nothing to recover (§2.3).

## B.0.2 FlashInfer: split-KV routing calibration

FlashInfer-style implementations gate split-KV on a context threshold [7]. The reference port used N ≥ 1024 on every GPU, forcing a two-pass path where stock MLX remained single-pass and under-splitting by 2× at 5–7k. This cost 1–3% on M2. Our router (Listing B.1) therefore uses a measured per-architecture table; an unmeasured machine inherits stock behaviour rather than a heuristic tuned on different hardware.

```lisp
// cider_mlx/sdpa.py -- the routing decision, resolved once at import
bool use_2pass(int N, int gqa) {
if (!use_mlx_routing) return N >= 1024; // the reference heuristic
if (device_class == 's' || device_class == 'd') return N >= 1024;
return gqa > 1 && N >= 4096; // MLX's own gate
}
// Architectures measured to prefer MLX's routing over the reference:
// applegpu_g14g (M2) -- the flat threshold forced split-KV where stock
// stays 1-pass, and under-split 2x at 5-7k: -1..-3%.
```

Listing B.1: Split-KV is enabled by measurement, not by default. An architecture that nobody has benchmarked inherits stock behaviour rather than a heuristic tuned elsewhere.

## B.0.3 Megakernel: cooperation, occupancy and geometry

The probes below complete the exploratory case study of §2.3.5. As there, they cover one LFM2.5- 8B-A1B MoE block on one M4 Pro under MLX.

P2: probing each cooperation guarantee Each clause CUDA guarantees in writing, probed on an M4 Pro:
<table><tr><td>CUDA guarantee</td><td>Public Metal</td><td>Measured</td></tr><tr><td>Cooperative launch</td><td>none</td><td>1024 toy TGs resident; 2048 silently fail</td></tr><tr><td>grid.sync() Fenced plain stores</td><td>hand-rolled</td><td>works: 0.38 μs at 256-thread TGs 96 % stale across TGs</td></tr><tr><td>Scoped atomics</td><td>none</td><td>coherent: 0 stale in 41.3 × 10⁶ reads</td></tr><tr><td>cp.async</td><td>relaxed only none</td><td>no prefetch across a barrier</td></tr></table>

Rows two and four are why a barrier can be built at all (Listing B.2); rows one and three are why it cannot be trusted. In this probe, ordinary device writes were not reliably visible across threadgroups and volatile did not fix them, so each tested cross-phase handoff had to round-trip through relaxed atomics.

```c
// The only grid barrier Metal permits: monotonic counter, relaxed atomics,
// bounded spin. Correct only if EVERY threadgroup is already resident.
threadgroup_barrier(mem_flags::mem_threadgroup); // 1. sync within my group
if (thread_index_in_threadgroup == 0) { // 2. one vote per group
atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
while (atomic_load_explicit(counter, memory_order_relaxed) < NUM_TGS) {
/* 3. spin -- deadlocks if any threadgroup has not been scheduled */
}
}
threadgroup_barrier(mem_flags::mem_threadgroup); // 4. release my group
```  
Listing B.2: Step 3 is the whole problem. If the grid exceeds what the GPU holds, the resident threadgroups spin forever waiting for threadgroups that cannot start until a resident one finishes. There is no API to determine that in advance.

P3: the same kernel, one register-footprint apart
<table><tr><td>Variant</td><td>Difference</td><td>Result</td></tr><tr><td>v1</td><td>lighter phase B</td><td>barrier converged, 183 μs</td></tr><tr><td>v2</td><td>register-heavier phase B, otherwise identical</td><td>non-convergent, silent 30 ms spin timeouts</td></tr></table>

A toy barrier held 1024 threadgroups; the real megakernel, carrying quantised-matmul register pressure and threadgroup-memory tiles, could not hold 64. One variable changed between v1 and v2: the per-thread register footprint, which the compiler chooses and never reports. There is no occupancy query with which to predict the outcome, no cooperative launch that would fail loudly instead, and no diagnostic distinguishing v2’s hang from a deadlocked barrier.

Geometry: fused versus shape-optimal grids With the barrier disabled, measuring pure compute:
<table><tr><td>Configuration</td><td>Time</td><td>Effective rate</td></tr><tr><td>Stock 11-op chain (each op its own grid)</td><td>96 μs</td><td> $2 2 5 \mathrm { G B s ^ { - 1 } }$  over 22 MB</td></tr><tr><td>One-dispatch megakernel, compute only</td><td>174μs</td><td> $1 2 4 \mathrm { G B s ^ { - 1 } }$ </td></tr></table>

Phase B’s row partition wants 64 threadgroups; at 64 threadgroups phase A streams 14.7 MB of gate and up weights at a third of the rate the same code achieves in its own dispatch. Shape-optimal grids are therefore worth 78 µs on this block, against the 15 µs that removing every boundary recovers: the two effects differ by 5× and point in opposite directions.

Would an interpreter design escape this? Mirage-style designs [44] replace fixed phases with an instruction interpreter, which might seem to sidestep the geometry constraint. For this workload it would retain both observed pressures. An interpreter contains the code for every instruction type, so its static register allocation reflects their combined worst case; and without a public cp.async-class mechanism it cannot prefetch the next instruction’s weights behind the current one’s compute.

## C Full Quantization Benchmark Results

This section reports the measurements underlying Section 2.4 that are not shown in the main text: latency and throughput for the short scenario (512-token prompt / 256-token generation) on M2 Pro and M3 Ultra (Tables 20 and 21), and additional GGUF results on the RTX 4090 (§C.2). All results are for LFM2.5-8B-A1B. Family, runtime, and bolding conventions follow Section 2.4.1.

## C.1 Latency and throughput

Table 20: M2 Pro, short scenario (512-token prompt / 256-token generation): latency and throughput for LFM2.5-8B-A1B. Conventions as in Table 15.
<table><tr><td>Method</td><td>TTFT (ms) ↓</td><td>Prefill  $\mathrm { ( t o k / s ) \uparrow }$ </td><td>Decode (tok/s) ↑</td><td>E2E (tok/s) ↑</td></tr><tr><td>GGUF-Q8_0</td><td> $5 0 5 . 7 \pm 1 . 2$ </td><td> $^ { 1 , 0 1 2 \pm 2 }$ </td><td> $7 9 . 1 \pm 0 . 7$ </td><td> $6 8 . 6 \pm 0 . 6$ </td></tr><tr><td>GGUF-Q6_K</td><td> $5 5 2 . 5 \pm 2 . 1$ </td><td> $9 2 7 \pm 3$ </td><td> $9 4 . 2 \pm 1 . 1$ </td><td> $7 8 . 5 \pm 0 . 8$ </td></tr><tr><td>GGUF-Q5_K_M</td><td> $5 9 7 . 1 \pm 1 . 0$ </td><td> $8 5 7 \pm 1$ </td><td> $8 9 . 2 \pm 1 . 0$ </td><td> $7 4 . 0 \pm 0 . 7$ </td></tr><tr><td>GGUF-Q4_K_M</td><td> $5 2 9 . 3 \pm 1 . 6$ </td><td> $9 6 7 \pm 3$ </td><td> $1 1 1 . 1 \pm 0 . 9$ </td><td> $9 0 . 6 \pm 0 . 7$ </td></tr><tr><td>GGUF-Q4_0</td><td> ${ \bf 4 5 9 . 0 \pm 1 . 7 }$ </td><td> ${ \bf 1 , 1 1 5 \pm 4 }$ </td><td> $\mathbf { 1 3 0 . 6 \ : \pm : } 1 . 2$ </td><td> ${ \bf 1 0 6 . 1 \pm 0 . 8 }$ </td></tr><tr><td>MLX-W8</td><td> $1 , 0 2 1 . 1 \pm 5 . 7$ </td><td> $5 0 1 \pm 2$ </td><td> $8 5 . 2 \pm 0 . 3$ </td><td> $6 3 . 7 \pm 0 . 2$ </td></tr><tr><td>MLX-W4</td><td> $1 , 0 0 2 . 2 \pm 5 . 1 $ </td><td> $5 1 1 \pm 2$ </td><td> $1 1 7 . 7 \pm 0 . 5$ </td><td> $8 0 . 7 \pm 0 . 2$ </td></tr><tr><td>MLX-W4-DWQ</td><td> $9 9 7 . 4 \pm 4 . 3$ </td><td> $5 1 3 \pm 1$ </td><td> ${ \bf 1 2 1 . 5 \pm 0 . 5 }$ </td><td> $8 2 . 6 \pm 0 . 2$ </td></tr><tr><td>MLX-W4-Affine</td><td> $9 9 3 . 1 \pm 3 . 6 $ </td><td> $5 1 6 \pm 1$ </td><td> ${ \bf 1 2 1 . 5 \pm 0 . 6 }$ </td><td> ${ \bf 8 2 . 7 \pm 0 . 2 }$ </td></tr><tr><td>MLX-MXFP4</td><td> $\mathbf { 9 5 1 . 9 \ : \pm 6 . 1 }$ </td><td> ${ \bf 5 3 8 \pm } 2 $ </td><td> $1 1 7 . 8 \pm 0 . 6$ </td><td> $8 2 . 0 \pm 0 . 2$ </td></tr><tr><td>MLX-NVFP4</td><td> $9 7 3 . 9 \pm 4 . 5$ </td><td> $5 2 6 \pm 2$ </td><td> $1 1 5 . 4 \pm 0 . 4$ </td><td> $8 0 . 3 \pm 0 . 2$ </td></tr><tr><td></td><td> $9 6 7 . 8 \pm 9 . 2$ </td><td></td><td></td><td></td></tr><tr><td>RTN-Asym  $\mathrm { Q R + R T N - A s y m }$ </td><td></td><td> $5 2 9 \pm 5$ </td><td> $1 1 4 . 9 \pm 1 . 1$ </td><td> $8 0 . 3 \pm 0 . 6$ </td></tr><tr><td></td><td> $9 7 7 . 8 \pm 9 . 8$ </td><td> $5 2 4 \pm 5$ </td><td> $1 1 1 . 9 \pm 1 . 0$ </td><td> $7 8 . 6 \pm 0 . 4$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - A s y m }$ </td><td> $9 7 4 . 0 \pm 5 . 1$ </td><td> $5 2 6 \pm 3$ </td><td> $1 1 2 . 1 \pm 0 . 7$ </td><td> $7 8 . 8 \pm 0 . 3$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - S y m }$ </td><td> $9 7 4 . 1 \pm 7 . 9$ </td><td> $5 2 6 \pm 4$ </td><td> $1 1 2 . 1 \pm 0 . 9$ </td><td> $7 8 . 8 \pm 0 . 5$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - M i x e d }$ </td><td> $9 7 7 . 2 \pm 1 0 . 0$ </td><td> $5 2 4 \pm 5$ </td><td> $1 1 1 . 7 \pm 0 . 9$ </td><td> $7 8 . 5 \pm 0 . 4$ </td></tr></table>

## C.2 Cross-platform comparison

Figure 14 shows the RTX 4090 peak memory footprint versus throughput for GGUF quantization of LFM2.5-8B-A1B.

Table 21: M3 Ultra, short scenario (512-token prompt / 256-token generation): latency and throughput for LFM2.5-8B-A1B, quantized with GGUF, MLX, and llm-compressor (RTN/QuaRot). Bolding and other conventions follow Table 15; each runtime has its own BF16 baseline.
<table><tr><td>Method</td><td>TTFT (ms) ↓</td><td>Prefill  $\mathrm { ( t o k / s ) \uparrow }$ </td><td>Decode  $\mathrm { ( t o k / s ) \uparrow }$ </td><td>E2E (tok/s) ↑</td></tr><tr><td>BF16 (baseline, GGUF)</td><td> $1 6 2 . 1 \pm 0 . 5$ </td><td> $3 , 1 5 9 \pm 1 0$ </td><td> $1 4 0 . 8 \pm 1 . 4$ </td><td> $1 2 9 . 7 \pm 1 . 2$ </td></tr><tr><td>GGUF-Q8_0</td><td> $1 5 6 . 1 \pm 0 . 4$ </td><td> $^ { 3 , 2 8 0 \pm 8 }$ </td><td> $1 9 3 . 6 \pm 2 . 7$ </td><td> $1 7 3 . 7 \pm 2 . 2$ </td></tr><tr><td>GGUF-Q6_K</td><td> $1 6 3 . 8 \pm 0 . 6$ </td><td> $^ { 3 , 1 2 6 \pm 1 1 }$ </td><td> $1 9 9 . 2 \pm 2 . 9$ </td><td> $1 7 7 . 2 \pm 2 . 4$ </td></tr><tr><td>GGUF-Q5_K_M</td><td> $1 7 2 . 4 \pm 0 . 6$ </td><td> $^ { 2 , 9 6 9 \pm 1 1 }$ </td><td> $1 9 2 . 6 \pm 2 . 6$ </td><td> $1 7 1 . 0 \pm 2 . 1 $ </td></tr><tr><td>GGUF-Q4_K_M</td><td> $1 5 6 . 8 \pm 0 . 3$ </td><td> $^ { 3 , 2 6 6 \pm 7 }$ </td><td> $2 2 1 . 2 \pm 3 . 2$ </td><td> $1 9 5 . 4 \pm 2 . 5$ </td></tr><tr><td>GGUF-Q4_0</td><td> $\mathbf { 1 4 7 . 8 \pm 0 . 5 }$ </td><td> ${ \bf 3 , 4 6 5 \pm 1 3 }$ </td><td> ${ \bf 2 3 2 . 8 \pm 3 . 4 }$ </td><td> $\mathbf { 2 0 5 . 8 \ : \pm 2 . 7 }$ </td></tr><tr><td>BF16 (baseline, MLX)</td><td> $2 1 5 . 5 \pm 3 . 2$ </td><td> $2 , 3 7 7 \pm 2 2$ </td><td> $1 4 2 . 0 \pm 0 . 2$ </td><td> $1 2 7 . 2 \pm 0 . 2$ </td></tr><tr><td>MLX-W8</td><td> $1 9 6 . 7 \pm 4 . 6 $ </td><td> $2 , 6 0 5 \pm 3 7$ </td><td> $1 9 1 . 9 \pm 0 . 3$ </td><td> $1 6 7 . 8 \pm 0 . 3$ </td></tr><tr><td>MLX-W4</td><td> $1 8 7 . 9 \pm 5 . 7$ </td><td> $2 , 7 2 7 \pm 4 9$ </td><td> $2 3 8 . 8 \pm 2 . 7$ </td><td> $2 0 3 . 7 \pm 2 . 0$ </td></tr><tr><td>MLX-W4-DWQ</td><td> $1 8 3 . 9 \pm 3 . 3$ </td><td> $2 , 7 8 5 \pm 3 0$ </td><td> $2 4 7 . 0 \pm 2 . 6$ </td><td> $2 1 0 . 3 \pm 2 . 0$ </td></tr><tr><td>MLX-W4-Affine</td><td> $1 8 4 . 0 \pm 3 . 9$ </td><td> $2 , 7 8 3 \pm 3 6$ </td><td> $2 4 8 . 0 \pm 0 . 6$ </td><td> $2 1 1 . 1 \pm 0 . 4$ </td></tr><tr><td>MLX-MXFP4</td><td> $1 8 6 . 2 \pm 3 . 3 $ </td><td> $2 , 7 5 1 \pm 3 0$ </td><td> $2 4 8 . 6 \pm 2 . 6$ </td><td> $2 1 0 . 8 \pm 2 . 0$ </td></tr><tr><td>MLX-NVFP4</td><td> $1 8 6 . 4 \pm 4 . 7$ </td><td> $2 , 7 4 8 \pm 4 2$ </td><td> $2 4 3 . 4 \pm 2 . 8$ </td><td> $2 0 7 . 3 \pm 2 . 2$ </td></tr><tr><td>RTN-Asym</td><td> $\mathbf { 1 7 3 . 3 \pm 4 . 0 }$ </td><td> $\mathbf { 2 , 9 5 7 \pm 6 6 }$ </td><td> ${ \bf 2 4 9 . 0 \pm 1 . 0 }$ </td><td> ${ \bf 2 1 3 . 8 \pm 0 . 4 }$ </td></tr><tr><td> $\mathrm { Q R + R T N - A s y m }$ </td><td> $1 7 7 . 1 \pm 4 . 8$ </td><td> $2 , 8 9 3 \pm 7 8$ </td><td> $2 4 1 . 0 \pm 1 . 2$ </td><td> $2 0 7 . 2 \pm 0 . 8$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - A s y m }$ </td><td> $1 7 7 . 6 \pm 6 . 1$ </td><td> $2 , 8 8 5 \pm 9 6$ </td><td> $2 4 0 . 7 \pm 1 . 7$ </td><td> $2 0 6 . 9 \pm 1 . 2$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - S y m }$ </td><td> $1 7 8 . 3 \pm 6 . 6$ </td><td> $2 , 8 7 5 \pm 1 0 5$ </td><td> $2 4 1 . 0 \pm 1 . 4$ </td><td> $2 0 7 . 0 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { Q R + G P T Q - M i x e d }$ </td><td> $1 7 9 . 1 \pm 6 . 7$ </td><td> $2 , 8 6 2 \pm 1 0 4$ </td><td> $2 3 9 . 9 \pm 1 . 3$ </td><td> $2 0 6 . 1 \pm 1 . 1$ </td></tr></table>

![](images/b6314db5df71fc2c8410bed63dc03c013a1228eec93928baeff51454a7b300aa.jpg)  
(a) Prefill throughput

![](images/24759818f300fd4e5afee74ac3d1939e1e037485af6b3903c8a4e0933e220aa3.jpg)  
(b) Decode throughput  
Figure 14: RTX 4090: peak memory footprint versus throughput for GGUF quantization of LFM2.5- 8B-A1B, same conventions as Figure 11.