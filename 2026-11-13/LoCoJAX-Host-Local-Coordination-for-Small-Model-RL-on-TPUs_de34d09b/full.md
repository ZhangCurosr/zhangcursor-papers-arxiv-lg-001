# LoCoJAX: Host-Local Coordination for Small-Model RL on TPUs

Loki Chen<sup>1,∗</sup>, Gregory B. Rehm<sup>1,∗</sup>, Yang Song<sup>1</sup>, Hongye Xie<sup>1</sup>, Ming Lei<sup>1</sup>, Chongyang Bai<sup>1</sup>, Bob Kamma<sup>1</sup>, Mingfu Liang<sup>1</sup>, Kavosh Asadi<sup>1</sup>, Yufei Li<sup>1</sup>, Xi Liu<sup>1</sup>, Jay Xu<sup>1</sup>, Rui Liu<sup>1</sup>, Maziar Sanjabi<sup>1</sup>, Luke Simon<sup>1</sup>, Hamed Firooz<sup>1</sup>

<sup>1</sup>Meta

<sup>∗</sup>Equal contribution

Reinforcement learning (RL) increasingly targets small language models, and executing these workloads eficiently across multiple accelerators requires efective coordination. On TPUs, Pathways-based RL uses a single controller to coordinate computation across hosts. This execution model is efective for large models, but its coordination computational costs are harder to amortize when small models repeatedly invoke short computations. To characterize these computational costs, we sweep dense models from 0.6B to 405B parameters on TPU v7x under a decode-heavy workload and measure how host and network computational overhead vary with model size. At 0.6B, host orchestration occupies 36% of step time and network communication 2%, whereas for large models, estimated network computational costs dominate the non-compute time. Motivated by these findings, we present Local Coordination JAX (LoCoJAX), which implements local rollout and reward processing on each TPU host with synchronous global training. In controlled comparisons against a tuned and optimized Pathways baseline on Qwen3-0.6B across three workloads and 8–64 TPU v7x chips, its colocated mode (LoCoJAX-CO) is 1.3–3.6× faster per step; on GSM8K, it completes a step faster on 8 chips than Pathways does on 64. Finally, a Ray-managed disaggregated mode (LoCoJAX-DA) overlaps rollout and training on separate hosts, and placement experiments establish when this overlap compensates for dividing devices and transferring weights: only when rollout dominates the step and receives most of the hosts. On NuminaMath-CoT at 256 chips, giving rollout 75% of hosts cuts step time by about 15% compared with LoCoJAX-CO.

Correspondence: Loki Chen at dengcchi@meta.com

∞Meta

## 1 Introduction

Reinforcement learning (RL) is now a standard stage of large language model (LLM) post-training (Christiano et al., 2017; Ouyang et al., 2022), particularly for reasoning tasks with verifiable rewards (Shao et al., 2024; Lambert et al., 2024; DeepSeek-AI et al., 2025). As a result, extensive efort has been devoted to characterizing and optimizing RL at diferent stages of its lifecycle, including rollouts (Fu et al., 2025; Zhou et al., 2025) and trainer-side improvements (Zhong et al., 2025b; Wan et al., 2026). Most of this research has focused on models over 7B in size on GPU clusters, where generating responses, training, and redistributing weights are all computationally expensive (Mei et al., 2025; Sheng et al., 2025; Zhong et al., 2025b; Fu et al., 2025; Zhong et al., 2025a; Yu et al., 2026). However, to the best of our knowledge, comparatively little work has addressed small-model RL, where optimization challenges can difer from those of larger models.

Today, RL increasingly targets smaller models: sub-billion and few-billion-parameter reasoners (Dang and Ngo, 2025; Wang et al., 2026; Zhuang et al., 2026; Zeng et al., 2025). These models often serve as fast reasoning agents in latency-sensitive applications (Belcak et al., 2025) or on embedded devices (Liu et al., 2024). Shrinking a model shrinks its compute but not the incidentals around it. Prior work finds framework dispatch (Fernandez et al., 2023), kernel boundaries (Spector et al., 2025), launch latency (Shen et al., 2024), request scheduling (vLLM Team, 2024; Zhou et al., 2025), and single-controller data paths (Zhou et al., 2025) can all become bottlenecks on small models. Taken together, these works show that small-model RL steps are often bound by incidentals outside of LLM computations.

![](images/31209689c806144b7ddf182be5b1876b0296686805a56f8e864aa0c949662da3.jpg)  
Figure 1 Where the Group Relative Policy Optimization (GRPO) step goes as model size grows. For each dense model, we report the host-orchestration and network shares of step time at its fastest tensor-parallel configuration on 128 TPU v7x chips under a decode-heavy workload with 8,192 forced output tokens (Table 2). Host orchestration is the largest non-compute component for small models and network communication for large ones.

Prior work on small-model RL bottlenecks was all run on GPUs, and to our knowledge, TPUs remain unstudied in this context. TPUs difer from GPUs in hardware, software and control. Chips connect over a dedicated inter-chip interconnect (ICI) in 3D-torus slices (Jouppi et al., 2023). The software stack is JAX and XLA (Bradbury et al., 2018), with kernels in Pallas (JAX Authors, 2026b; Jiang et al., 2026), training in MaxText/Tunix (Google, 2023; Bao et al., 2025), and rollout in vLLM’s TPU backend (vLLM Project, 2025). These diferences reach the RL loop: a TPU device can be opened by only one user-space process, so trainer and engine cannot be colocated in separate processes out of the box (Basile et al., 2026). Multi-host programs run either under multi-controller JAX, with one process per host issuing the same computations in the same order (JAX Authors, 2026a), or under Pathways, whose single controller drives every host (Barham et al., 2022); Google’s documented multi-host TPU RL recipes use Pathways (Google Cloud, 2026d,a,b). Prior work still finds large improvements in TPU LLM serving, from vLLM’s TPU backend to its Pallas attention kernel (Lee et al., 2025; vLLM Team, 2025; Jiang et al., 2026), and reports single-controller dispatch and communication computational overheads that grow with scale (Kumar et al., 2021; Barham et al., 2022; Google Cloud, 2026c). Together, these diferences make TPUs an optimization landscape distinct from GPUs, and one that small-model RL has not yet explored.

Motivated by this gap, we first measured where small-model RL spends its time on TPUs. We profile decode-heavy GRPO under a Pathways recipe on 128 TPU v7x chips, across dense models from 0.6B to 405B parameters, and split each step into compute, network communication and host orchestration (Figure 1, Table 2). The breakdown of step time varies with model size. For Qwen3-0.6B, host orchestration takes 36% of the GRPO step (rollout plus training) and network communication 2%. The two shares are equal at Qwen3-8B (18.3% and 18.5%), and beyond that network computational overhead becomes the dominant factor in latency. Thus, in the evaluated Pathways stack, host orchestration is the largest non-compute component for small models and our design therefore focuses on reducing this computational overhead.

Using these findings, we design Local Coordination JAX (LoCoJAX), which moves rollout control from a single controller to each TPU host while training remains globally sharded (Figure 2). Each prompt and its full GRPO response group go to one host, whose engine generates the responses and whose process computes rewards and group advantages without cross-host communication. All hosts then train jointly and load the updated weights into their local engines before the next rollout. This colocated mode (LoCoJAX-CO) changes execution, not the algorithm: training stays synchronous and on-policy, as in the Pathways recipe. Against our Pathways baseline built from public MaxText, Tunix and vLLM TPU components, optimized LoCoJAX-CO is 1.3–3.6× faster per step on Qwen3-0.6B across three workloads and 8–64 TPU v7x chips; on GSM8K, it completes a step faster on 8 chips than Pathways does on 64.

![](images/ba07bf6295008fa905421875e3adc0d097bf93b5525c9bb7e5d43404413c02c7.jpg)  
Figure 2 System architectures of Pathways, LoCoJAX-CO, and LoCoJAX-DA. (a) Our evaluated baseline follows Pathways documented single-controller SPMD execution model: one CPU-head controller drives a globally distributed rollout engine and sharded training across TPU hosts. (b) LoCoJAX-CO adopts synchronous, colocated execution with independent host-local JAX controllers and rollout engines. (c) LoCoJAX-DA uses Ray-managed disaggregation to separate rollout and training hosts, enabling asynchronous execution and overlap between generation and training.

Profiling both systems on GSM8K at 64 chips localizes most of the measured step-time gap to rollout (Section 4.3). Pathways’ rollout runtime incurs two kinds of recurring computational overhead: its head admits every request serially each step, a computational cost that does not shrink with more chips; and each decode iteration takes more than twice as long as on LoCoJAX’s host-local engines (Figure 7). We also contribute a disaggregated mode (LoCoJAX-DA) that uses Ray (Moritz et al., 2018) to run rollout and training on separate hosts (Section 4.4). LoCoJAX-DA is modestly superior when rollout dominates the step and receives most of the hosts, but on our balanced and prefill-heavy datasets colocation usually wins. As a result, we recommend LoCoJAX-CO as a strong default for small-model RL on TPUs.

## In summary, this paper makes the following contributions:

1. We characterize decode-heavy GRPO on TPU v7x across dense models from 0.6B to 405B parameters. Host orchestration is the largest component for the smallest models, whereas network communication dominates the non-compute time of the largest.

2. We design and implement LoCoJAX, an end-to-end TPU RL system designed for small models. Compared with a tuned and optimized Pathways baseline on Qwen3-0.6B, LoCoJAX reduces step time by a factor of 1.3–3.6. It outperforms Google’s public recipes on Qwen3-1.7B (5.3×) and Gemma4-26B MoE (2.1×).

3. We identify two rollout-side computational costs—serial request admission and higher per-iteration decode latency—that account for most of the measured performance gap between Pathways and LoCoJAX on GSM8K at 64 chips.

4. We compare colocated and disaggregated LoCoJAX execution at 8–256 chips and three trainer/rollout splits. Colocation is the better default, but disaggregation can improve step time for rollout-heavy workloads at suficient scale.

## 2 LoCoJAX Design

In this section, we introduce LoCoJAX, a system designed for RL training of small models on TPUs. It builds on JAX’s standard multi-controller mode, typically one process per host (JAX Authors, 2026a). LoCoJAX combines local request admission and GRPO grouping with globally sharded training and weight publication to local rollout engines. Figure 2 compares this execution model with the Pathways baseline. We support two

Algorithm 1 One colocated LoCoJAX step for Qwen3-0.6B, run on each host h among H hosts.   
Require: global prompt batch B of P prompts with G responses each; global policy θ with shard $\theta _ { h }$ on host h   
(FSDP); local engine $E _ { h } \ ( \mathrm { D P 8 } \times \mathrm { T P 1 } )$   
1: $B _ { h } \gets \mathrm { L o c a L S H A R D } ( B , h )$ ▷ $P / H$ prompts; balanced by prompt length when optimized   
2: $R _ { h } \gets E _ { h } . \mathrm { G E N E R A T E } ( B _ { h } )$ ▷ host admits only its $P G / H$ requests   
3: $r _ { h } \gets \mathrm { R E W A R D } ( R _ { h } ) ; A _ { h } \gets \mathrm { G R O U P A D V A N T A G E } ( r _ { h } )$ ▷ all G samples of a prompt are local   
4: $X _ { h } \gets \operatorname { P a c K } ( R _ { h } , A _ { h } )$ ▷ fixed-length rows; hosts agree on padded rows per device K   
5: $\theta \gets \mathrm { T R A I N S T E P } ( \theta , X _ { h } )$ ▷ one forward/backward pass and update; FSDP collectives   
6: $E _ { h } . \mathrm { L O A D W E I G H T S } ( \mathrm { R E S H A R D T O E N G I N E } ( \theta , h ) )$ ▷ gather FSDP shards into local engine replicas before   
next rollout  
Table 1 Stack-comparison workloads (Qwen3-0.6B): prompts × generations per step, prompt and response caps (tokens), and Qwen3 thinking mode.

<table><tr><td>Dataset</td><td>Character</td><td>Prompts × generations</td><td>Prompt cap</td><td>Response cap</td><td>Thinking</td></tr><tr><td>GSM8K</td><td>balanced prefill and decode</td><td> $1 2 8 \times 1 6$ </td><td>512</td><td>2,048</td><td>off</td></tr><tr><td>NuminaMath-CoT</td><td>decode-heavy</td><td> $2 5 6 \times 8$ </td><td>4,096</td><td>8,192</td><td>on</td></tr><tr><td>SWE-agent trajectories</td><td>prefill-heavy</td><td> $2 5 6 \times 8$ </td><td>15,360</td><td>1,024</td><td>off</td></tr></table>

modes: colocated execution (LoCoJAX-CO) and disaggregated execution managed by Ray (LoCoJAX-DA).   
Unless marked otherwise, LoCoJAX refers to LoCoJAX-CO.

Colocated mode (LoCoJAX-CO). Each host runs one process of a trainer whose state is sharded across all devices using fully sharded data parallelism (FSDP), alongside a local vLLM rollout engine (Figure 2b). For Qwen3-0.6B, the engine runs eight data-parallel (DP) replicas per host, one per TPU TensorCore, each using tensor parallelism (TP) of degree one (DP8 × TP1). Rollout TP is configurable; larger models can use TP2 or higher.

Each step follows five phases: request admission, generation, reward, training, and weight publication (Appendix C). Each host admits its share of the prompts and generates G responses per prompt, where G is the GRPO group size. Because each group stays on one host, that host can evaluate rewards and compute group advantages without cross-host communication. All hosts then train jointly on the batch. The updated weight shards are all-gathered over the TPU ICI, and each host loads the weights into its local engine before the next rollout. This keeps learning synchronous and on-policy.

Relative to the Pathways baseline, LoCoJAX changes where requests are admitted and where rollout ranks synchronize:

• Per-host request admission. The evaluated Pathways baseline admits all 2,048 requests serially in its head process. LoCoJAX distributes this work: each of H hosts admits 2,048/H requests in parallel with the others (128 requests per host at 64 chips).

• Rollout synchronization. The Pathways recipe runs all rollout ranks in lockstep as one SPMD program. During generation, LoCoJAX synchronizes only the devices within each host. Cross-host communication remains necessary for packed-shape agreement, FSDP training collectives, and weight publication.

Section 4.3 measures the two recurring controller computational costs: serial request admission once per step and the decode premium once per iteration.

Disaggregated mode (LoCoJAX-DA). LoCoJAX-DA uses Ray to launch and coordinate separate trainer and rollout hosts (Moritz et al., 2018). Ray (Moritz et al., 2018) is a general-purpose distributed actor framework that GPU RL systems such as HybridFlow (Sheng et al., 2025) use to place stages on separate workers; here it provides an alternative to both colocation and Pathways’ single controller for TPU RL. Generation uses the previously published policy while the trainer computes the next update, introducing one-step policy staleness. After each update, trainer hosts publish weights to Ray’s object store, and rollout hosts fetch a full copy over the data-center network (DCN). Rollout data sent to the trainer include the generating policy’s log-probabilities for importance weighting. Section 4.4 gives the tested splits and timing tradeofs.

![](images/b7c2bccb3cd6976e5280c52b92a813ed3839f4dbf8d8df0146220bedc78a16a4.jpg)  
(A) Step-time breakdown.

![](images/baaaa97c8a079ddea9581c3949461b72f10380ee37e91c3744cc22578c6789ef.jpg)  
(B) Tensor-parallel scaling.  
Figure 3 Scaling study on 128 TPU v7x chips, Pathways, decode-heavy workload (8,192 forced output tokens); medians over runs. (A) Share of step time by component at each model’s fastest TP (Table 2). (B) Step time at each rollout TP relative to the model’s fastest TP. Filled: each model’s fastest TP (Table 2). 14B has no TP16 (its 40 attention heads do not divide by 16), ∼100B ran at TP16 only, and 0.6B–70B were not run at TP32. Values: Table H1.

LoCoJAX adds four optional optimizations (O1–O4; Table B1): (O1) exporting metrics outside the host barrier, (O2) freezing Python garbage collection, (O3) balancing prompts by length, and (O4) publishing all weights to the local engine in one call. Section 4.2 compares Pathways with LoCoJAX base, which uses the compatible optimizations shared with the baseline, and LoCoJAX optimized, which also enables O1–O4.

## 3 Experimental Setup

All experiments run on Google TPU v7x. Each chip has two TensorCores, which JAX exposes as separate devices with about 95 GiB of usable HBM each, and each host has four chips. The stack comparison uses single slices ranging from 8 chips (two hosts, 2×2×2) to 64 chips (16 hosts, 4×4×4). The scale study uses 128-chip slices, and the colocated-vs-disaggregated comparison (Section 4.4) extends to 128 and 256 chips. Pathways jobs additionally use a CPU node to host the single controller.

Pathways recipe (baseline). Pathways (Barham et al., 2022) is the runtime used by Google’s documented multihost RL recipes on Cloud TPU: Tunix and MaxText run multi-host training on Pathways on Cloud (Google Cloud, $2 0 2 6 \mathrm { d } , \mathrm { a } , \mathrm { b } )$ . Our baseline is a GRPO recipe for Pathways that we assembled from public components: a MaxText and Tunix trainer (Google, 2023; Bao et al., 2025) and a vLLM TPU sampler (Kwon et al., 2023; vLLM Project, 2025). Both are built from pinned public main-branch commits (MaxText 3258182f with PR #5289 (Appendix D), Tunix d6893d5e, tpu-inference 518d1aaa, vLLM 51da0ca6; JAX 0.11.0 (Bradbury et al., 2018)). The trainer and sampler are colocated on the same devices, use bf16 trainer weights as in MaxText’s default RL configuration, and are driven by one head process as a single SPMD program. Rollout runs at TP1 with data parallelism over all devices $( \mathrm { D P = 2 \times \mathrm { c h i p s } ) }$ , and training uses FSDP. We froze the recipe before our experiments. To match work, we use two passes over the same batches, matched sampling, and extra sequence-length logging. The Pathways arm of Table 3 also runs every optimization compatible with both stacks.

## 3.1 Workloads

Scale study. To measure where time goes as model size changes, every cell runs GRPO on OpenMathInstruct-2 (Toshniwal et al., 2024) prompts with 256 prompts × 8 generations = 2,048 completions per step, each forced to 8,192 tokens, using the Pathways scheduler. We chose a very decode-heavy workload because RL for reasoning generates long chains of thought, so most rollout time goes to decode; forcing every completion to 8,192 tokens emulates those long traces, and with prompts of about 150 tokens on average, prefill is negligible. Fixing the output length makes the work per step identical across models. Models are Qwen3-0.6B to 32B (Yang et al., 2025), Llama-3.1-70B, two synthetic dense models with Llama-3.1-405B width (∼100B with 31 layers and ∼200B with 62), and Llama-3.1-405B (Grattafiori et al., 2024); the Llama-3.1 models contribute architecture configurations only. Each model runs on Pathways at 128 chips at every TP from the smallest that fits up to TP8, plus TP16 (not possible for 14B, whose attention heads do not divide by 16); ∼100B runs only at TP16, and ∼200B and 405B run at TP16 and TP32. Weights are randomly initialized.

Table 2 Step time and its breakdown at each model’s fastest TP, 128 TPU v7x chips, decode-heavy workload (8,192 forced output tokens). Computational overhead is the share of the step outside compute; host and network are its two largest parts. ≈ marks a tie within run-to-run spread; values are for the first-named TP. Medians over 3 runs (∼100B: 2; 405B: 1, a workaround extension); network shares are upper-end estimates.
<table><tr><td>Model</td><td>Fastest TP Step (s)</td><td>Computational Overhead</td><td></td><td>Host</td><td>Network</td></tr><tr><td>Qwen3-0.6B</td><td>TP1 ≈ TP2</td><td>68.6</td><td>40.6%</td><td>36.4%</td><td>2.2%</td></tr><tr><td>Qwen3-1.7B</td><td>TP1 ≈ TP2</td><td>79.6</td><td>29.8%</td><td>24.3%</td><td>3.5%</td></tr><tr><td>Qwen3-4B</td><td>TP2</td><td>108.5</td><td>32.2%</td><td>17.6%</td><td>12.2%</td></tr><tr><td>Qwen3-8B</td><td>TP4 ≈ TP2</td><td>134.1</td><td>39.6%</td><td>18.3%</td><td>18.5%</td></tr><tr><td>Qwen3-14B</td><td>TP4</td><td>168.9</td><td>34.2%</td><td>14.8%</td><td>17.4%</td></tr><tr><td>Qwen3-32B</td><td>TP4 ≈ TP8</td><td>301.3</td><td>29.2%</td><td>8.3%</td><td>18.6%</td></tr><tr><td>Llama-3.1-70B</td><td>TP8</td><td>425.6</td><td>28.3%</td><td>7.0%</td><td>20.0%</td></tr><tr><td>Synthetic ～100B</td><td>TP16 only</td><td>382.9</td><td>33.6%</td><td>5.5%</td><td>26.4%</td></tr><tr><td>Synthetic ~200B</td><td>TP16</td><td>729.3</td><td>31.8%</td><td>4.4%</td><td>25.9%</td></tr><tr><td>Llama-3.1-405B</td><td>TP16</td><td>3,334.0</td><td>40.3%</td><td>1.4%</td><td>38.5%</td></tr></table>

Stack comparison. Pathways and LoCoJAX are compared on Qwen3-0.6B over three public datasets, GSM8K (Cobbe et al., 2021), NuminaMath-CoT (Li et al., 2024) and SWE-agent trajectories (Nebius, 2024; Yang et al., 2024), chosen to vary where work falls (Table 1). Each run covers the same 20 batches twice (40 steps) and scores the second pass, so both stacks see the same prompts in the same order; we verify this at run time from both logs. On GSM8K we use bf16 trainer weights, which is also MaxText’s default RL configuration. On NuminaMath and SWE-agent we keep our stack’s default fp32 trainer weights, since both Pathways and LoCoJAX already produce matching output lengths there.

## 3.2 Measurement

Step time. Step time is the median over steady-state steps, read from each stack’s own clock. Compile time and end-to-end wall-clock time are not part of the comparison.

Time breakdown. For the scale study we split one profiled step into compute, network and host orchestration. The network includes TP collectives, other collectives, host-ofload copies, the DP all-reduce, and time spent waiting at the DP check and for TP partners. Host orchestration is host-side work outside device programs: controller round trips, scheduling and the host side of weight sync. Computational overhead is the share of the step outside compute (1 − compute/step).

Device-level TP comparison. Pathways step time at small model sizes moves by several seconds with the placement of its head node, which is as large as most TP efects there. We therefore compare TP degrees at the device level, in a separate 64-chip TP sweep on the same workload, using decode device time per step from xprof traces of one host, which is insensitive to head placement.

## 3.3 Comparison Protocol

Appendix A gives the full measurement protocol and verdict rules.

Equal work. A pair counts only if both runs process the same prompts (per-step mean prompt length within 0.5 tokens on all 40 steps), their mean and longest output lengths agree within a factor of [0.8, 1.25], and their truncation shares difer by at most 10 points. Output length matters because rollout time is set by the longest response.

Statistics. We report the speedup $S = T _ { \mathrm { P a t h w a y s } } / T _ { \mathrm { o u r s } } ,$ and the same ratio per processed token. Confidence intervals are 95% percentile bootstraps over steady steps (10,000 resamples). We call our stack faster only if the lower bound of both ratios exceeds 1.111, a margin that covers the roughly 10% shift the Pathways head node can cause on its own. GSM8K Pathways cells at 16–64 chips therefore use heads from one reference pool of CPU head hosts (the 8-chip head ran elsewhere).

## 4 Results

## 4.1 Scaling Study

We ran GRPO on Pathways for dense models from 0.6B to 405B on 128 TPU v7x chips, at each model’s fastest TP (Table 2, Figure 3A). What changes with model size is where that time goes. Host orchestration takes 36% of the step at 0.6B and falls steadily to 1% at 405B. Network does the opposite, rising from 2% at 0.6B to 26% at ∼200B and 39% at 405B.

Larger models need more tensor parallelism, and more TP means more collective trafic, so the fastest TP grows from TP1 at 0.6B to TP16 from ∼100B up (Figure 3B). Small models do not improve from going wider: in a separate 64-chip sweep, moving 0.6B from TP1 to TP2 saves 0.21 ms of weight streaming per decode iteration but adds 0.7–0.9 ms of collectives, while at 4B the saving grows to 1.16 ms and TP2 wins. Small models therefore stay at TP1, where they lose almost nothing for the network, and their computational overhead is almost all host.

The two ends of the model range need diferent fixes. Large models need high TP, so their levers are the choice of TP and computationally cheaper collectives. For small models the computational cost is host orchestration: controller round trips, request admission and scheduling, and the host side of weight sync. LoCoJAX targets that computational cost. It keeps TP1 rollout and takes the single controller out of the loop, giving each host its own controller and rollout engine. Section 4.2 compares it with Pathways on Qwen3-0.6B.

## 4.2 LoCoJAX vs. Pathways

LoCoJAX takes less time per step than Pathways on every dataset and chip count we ran (Table 3, Figure 4). LoCoJAX base is 1.95–2.97× faster on short-output GSM8K, 1.23–1.86× on decode-heavy NuminaMath and 1.64–2.05× on prefill-heavy SWE. LoCoJAX optimized adds O1–O4 and is 2.13–3.56×, 1.25–1.89× and 1.72–2.32× faster. The gap is largest on GSM8K, where each step does the least work, which fits Section 4.1; there LoCoJAX on 8 chips takes less time per step than Pathways on 64, even though the smaller run generates each batch in two waves. Our trainer does not accumulate gradients, so its training memory grows as chips are removed: it does not fit SWE at 8 chips (it needs 157.6 GiB per device), where Pathways still runs.

Model size. LoCoJAX’s advantage broadly shrinks as models grow, consistent with the falling host share in Figure 1 (Tables E1 and E2). On GSM8K at 64 chips, Pathways takes 3.25× the step time of LoCoJAX at 1.7B, 1.72× at 8B, 1.88× at 14B and 1.58× at 32B; on NuminaMath at 128 chips, it takes 1.88× at 1.7B, 1.51× at 4B and 1.30× at 8B.

LoCoJAX optimizations. As part of the LoCoJAX protocol, we found four additional host-side optimizations (Table B1). We call LoCoJAX with these optimizations LoCoJAX optimized and describe their operation in Appendix B.

## 4.3 What Coordination Explains

Figure 5 isolates what the Pathways controller adds to our jobs. Before any engine runs, the Pathways head tokenizes the batch and adds all 2,048 requests one at a time, about 1.0 s per step (timers of). This is serial work in one process; it stays at 0.88–0.98 s from 8 to 64 chips (Figure 5a) while the rest of the step shrinks, and its share of the Pathways step nearly doubles, from 8% to 14% (Figure 5b). What it scales with is the batch, so scaling out with a larger batch makes it worse, not better. LoCoJAX splits the same work across hosts: at 64 chips, each of the 16 hosts admits its own 128 requests, in parallel, in under 0.06 s (Appendix C, tradeof 2). Meanwhile, Pathways spends the equivalent of 48% of a whole LoCoJAX step before its first engine starts.

The head interval is not all of the controller computational cost. The controller impacts Pathways twice (Figure 7): once per step, when the head admits the batch (0.94 s with its other work before the engine call), and once per decode iteration. Each Pathways decode iteration takes 2.15 ms against 0.90 ms on LoCoJAX, and over the roughly 1,800 iterations of a step that premium adds 2.25 s, the largest term of the 64-chip gap. This matches the Pathways structure in Figure 2: Pathways runs every iteration as one lockstep program across all chips, while each LoCoJAX host steps its own engine.

![](images/01bffc2eea92143f872c479e2902fed79ab614bd5075c479655eb735050b5229.jpg)  
Figure 4 Step time vs TPU v7x chips (log–log), Qwen3-0.6B: Pathways, LoCoJAX base, and LoCoJAX optimized, which adds optimizations O1–O4 (Table B1). Values and speedups: Table 3. Bold: ratio of Pathways to each LoCoJAX line at 64 chips.

Table 3 Steady-state step time, Pathways vs colocated LoCoJAX base and optimized, Qwen3-0.6B; speedup over Pathways in parentheses. LoCoJAX base runs the shared optimizations of the Pathways arm; LoCoJAX optimized adds O1–O4 (Table B1). At NuminaMath 16 chips, optimized is 7% slower than base in same-window pairs too. LoCoJAX does not fit SWE at 8 chips (needs 157.6 of 94.7 GiB per device).
<table><tr><td>Dataset</td><td>Chips</td><td>Pathways (s)</td><td>LoCoJAX base (s)</td><td>LoCoJAX optimized (s)</td></tr><tr><td>GSM8K (short outputs)</td><td>8</td><td>10.82</td><td>5.56 (1.95x)</td><td>5.07 (2.13x)</td></tr><tr><td></td><td>16</td><td>8.56</td><td>4.08 (2.10x)</td><td>3.32 (2.58x)</td></tr><tr><td></td><td>32</td><td>8.41</td><td>2.83 (2.97x)</td><td>2.36 (3.56x)</td></tr><tr><td></td><td>64</td><td>6.75</td><td>2.69 (2.51x)</td><td>2.03 (3.32x)</td></tr><tr><td>NuminaMath (decode-heavy)</td><td>8</td><td>190.3</td><td>154.7 (1.23x)</td><td>152.2 (1.25x)</td></tr><tr><td></td><td>16</td><td>117.3</td><td>84.1 (1.39x)</td><td>89.4 (1.31x)</td></tr><tr><td></td><td>32</td><td>70.3</td><td>45.5 (1.55x)</td><td>44.6 (1.58x)</td></tr><tr><td></td><td>64</td><td>52.8</td><td>28.4 (1.86x)</td><td>28.0 (1.89x)</td></tr><tr><td>SWE (prefill-heavy)</td><td>8</td><td>163.1</td><td>N/A (no fit)</td><td>N/A (no fit)</td></tr><tr><td></td><td>16</td><td>105.0</td><td>64.2 (1.64x)</td><td>61.1 (1.72x)</td></tr><tr><td></td><td>32</td><td>65.2</td><td>34.7 (1.88x)</td><td>31.6 (2.06x)</td></tr><tr><td></td><td>64</td><td>39.4</td><td>19.2 (2.05x)</td><td>17.0 (2.32x)</td></tr></table>

## 4.4 Colocated vs. Disaggregated

We compare LoCoJAX-CO with LoCoJAX-DA (Section 2; Figure 6; Table F1). For every cell we run LoCoJAX-DA at three static splits that give the trainer 50%, 25% and 12.5% of the hosts, three launches each, and report all of them; a split is missing only where the host count, our Ray launcher or trainer memory rules it out (Appendix F). LoCoJAX-DA wins only where one phase dominates the step, and the three workloads show three ways this plays out.

GSM8K: rollout still scales at small sizes. At 8 and 16 chips LoCoJAX-DA is slower (1.61× and 1.13×) because rollout still speeds up with every host it gets, so moving hosts to the trainer requires more than overlapping training saves (2.9 vs 1.1 s at 8 chips). At 128 chips this reverses (0.90×): LoCoJAX-CO stops scaling (2.02 to 1.73 s from 64 to 128 chips, no improvement at 256) as the slowest rollout host falls further behind the average (38% to 55% of the step), and the one-step-stale schedule lets LoCoJAX-DA’s rollout hosts skip that wait.

NuminaMath: decode-heavy, so rollout needs the hosts. Long outputs make rollout most of the step. LoCoJAX-

![](images/d4323e70dbdcc90aaa657f93a6cc04f4854a4b61552f4a8a5ebad5cc84936ad8.jpg)  
Figure 5 Pathways head interval vs step time, GSM8K, Qwen3-0.6B, 2,048 requests, timers of. (a) Step time of Pathways and of LoCoJAX optimized (Table 3), and the Pathways head interval: time in the head process from the start of the global step to the opening of the generation progress bar, which contains the open-source request admission. (b) Head interval as a share of the Pathways step. Head intervals are means of two launch medians over steps 3–18, read from the Pathways launches behind Figure 4. Pathways heads ran in the reference node group at 16–64 chips and in a diferent node group at 8 chips. LoCoJAX generates in two waves at 8 chips.

DA loses at 32 chips when half the hosts generate (1.26×), and wins at 64 and 256, where three quarters do (0.95×, narrow, and 0.85×). At 128 chips, with an even split, it is unresolved; a rollout-heavier split tried afterwards reads 0.87× (provisional; Appendix F).

SWE: prefill-heavy, so the trainer is the bottleneck. Long prompts and short outputs make training on the long packed sequences the larger phase. LoCoJAX-DA gives the trainer only half the hosts, so that phase takes longer, and there is little rollout to overlap it with. LoCoJAX-CO wins at every size (1.14–1.30×; 32 chips added post hoc), and every split that moved hosts from training to rollout was slower still (Appendix F).

LoCoJAX-DA was not run on NuminaMath at 16 chips, where LoCoJAX-CO runs, and a 2-host SWE trainer runs out of memory: with fewer hosts, the trainer holds more model and optimizer state per chip, and LoCoJAX’s trainer does not accumulate gradients, so the whole packed batch must fit at once. These are timing results: on GSM8K, training reward with LoCoJAX-DA stays within our pre-registered 0.005 band of LoCoJAX-CO at 64 and 128 chips, and in an earlier, longer setup both LoCoJAX-DA splits at 32 chips failed a long-run quality gate. Colocation is the better default; disaggregation is beneficial when rollout dominates the step and one step of staleness is acceptable.

## 5 Related Work

RL post-training optimization. Most RL systems work targets larger LLMs on GPUs, where each stage is computationally expensive and the question is how to place and overlap them. Approaches include reallocating parallelism across stages (Mei et al., 2025; Sheng et al., 2025), fusing stages to remove pipeline bubbles (Zhong et al., 2025b), disaggregating generation from training (Zhong et al., 2025a), removing per-layer synchronization barriers under sequence-length imbalance (Wan et al., 2026), and choosing placement per workload (Yu et al., 2026). Closest to ours, RL in the Wild finds that serving-derived rollout engines add CPU scheduling computational overhead and that single-controller data paths bottleneck training (Zhou et al., 2025). All of these study GPUs at 1.5B parameters and above. We study small models on TPUs, where the documented recipe places a single controller on the rollout path, and find that removing it yields substantial improvements.

Small-model RL. Small models are a common post-training target, trained with RL as computationally cheap reasoners (Dang and Ngo, 2025; Wang et al., 2026; Zhuang et al., 2026) or examined at the low end of scaling studies (Zeng et al., 2025). Some use on-policy distillation as students (Zhong et al., 2026). These studies locate the bottlenecks in learning: optimization instability and length limits in long training (Dang and Ngo, 2025), limited capacity at 0.5B (Zhuang et al., 2026), rigid format rewards and mismatched data dificulty (Zeng et al., 2025), among others. Systems work on small computations finds execution computational overheads instead (framework dispatch (Fernandez et al., 2023), CPU-side serving and scheduling (vLLM Team, 2024), kernel boundaries (Spector et al., 2025)), but in inference, not RL. For small models, RAPID is the closest work: on a single 4-GPU server with 0.5–1.5B models, it cuts RL running time by 11–34% by sampling large inference batches and taking several importance-weighted of-policy updates on each, reducing switches between inference and training (Huang et al., 2025). LoCoJAX-CO instead keeps GRPO synchronous and on-policy and removes multi-host control computational overheads. The two optimizations target diferent computational costs and could be combined.

![](images/993c7c3fc813d73bdfb6f0a8c93b42323dee3532fa1ac8ad2880268207f71e32.jpg)  
Figure 6 Step time vs TPU v7x chips (log–log), LoCoJAX-CO vs LoCoJAX-DA at three trainer/rollout host splits, Qwen3-0.6B; medians over launches. Hollow markers: provisional cells. Missing LoCoJAX-DA splits are ruled out by host count, our Ray launcher or trainer memory (Appendix F). The LoCoJAX-CO point at NuminaMath 16 chips comes from Figure 4. Median step times per split: Table F1.

TPU and accelerator-native RL. In classical deep RL, SEED RL centralizes inference on accelerators (Espeholt et al., 2020), and Podracer either compiles the whole loop into one XLA program or splits TPU cores between actors and learners (Hessel et al., 2021). Both treat host-device coordination as the main computational cost, but for small networks without autoregressive decoding, KV caches or weight publication. TPU LLM RL is recent. Google’s public recipes run GRPO through MaxText or Tunix on Pathways (Google Cloud, 2026d; Google, 2026c,a; Barham et al., 2022); Tunix’s agentic stack adds asynchronous rollouts to keep TPUs busy during environment calls (Gao et al., 2026). Concurrently, Tunix added a distributed RL runtime with separate trainer and rollout workers (Google, 2026b), which we did not evaluate. Our work was inspired by Tunix RFC #706, which proposed multi-controller JAX for post-training but was not implemented (Wang, 2025). LoCoJAX applies it to the full RL loop by keeping variable-length generation local. To our knowledge, no published work studies sub-billion LLM RL on TPU or compares single- and multi-controller execution for it.

## 6 Discussion

Small-model RL on TPU is bound less by compute than by what surrounds it. Our characterization shows that at TP1, 41% of the Qwen3-0.6B Pathways step is outside compute: 36% host orchestration and 2% network (Section 4.1). LoCoJAX optimizes the host side; LoCoJAX optimized is 2.1–3.6× faster than Pathways on GSM8K, 1.3–1.9× on NuminaMath and 1.7–2.3× on SWE (Section 4.2). These results show the advantage of host-local rollouts over a centralized Pathways controller for small-model RL.

Our work draws inspiration from an unimplemented Tunix RFC (Wang, 2025) calling for multi-controller JAX for Tunix. This paper implements the RFC for RL and adds evidence about when it matters and how it speeds up small models. We show that the Pathways scheduler spends excessive time on host orchestration with small models. A single controller adds time to a small-model RL step in two places: once per step when it admits the batch and once per decode iteration (Figure 7). The Pathways head spends about 1 s per step admitting the batch, a computational cost that stays flat from 8 to 64 chips while the rest of the step time shrinks, so its share of the Pathways step nearly doubles, from 8% to 14% (Figure 5). Every result rests on public models, datasets and frameworks, so others can replicate our results.

We note that our Pathways baseline is a tuned one. This was because we found no peer-reviewed results for small-model RL on TPUs. The only other TPU baselines are Google’s public reference recipes, which we run with only the deviations listed in Appendix G, without these optimizations. Against them, on identical workloads, colocated LoCoJAX is 5.3× faster per step on Qwen3-1.7B with Tunix (Google, 2026c) and 2.1× on Gemma4-26B MoE with MaxText (Google, 2026a) (Appendix G).

Our contributions also highlight optimization choices for future RL workflows. Three choices mattered. (i) Keep per-request work of a single controller. Work that one process does for every request grows with the batch and does not shrink with chips; give each host its own controller and rollout engine (Section 4.3). (ii) Choose TP by model size, not by the smallest TP that fits. At 0.6B, TP2’s weight-bandwidth saving cannot cover its collective penalty, so TP1 wins; but from 4B up, higher TP wins (Section 4.1, Figure 3B). (iii) Colocate by default. Disaggregate only when one phase dominates the step and the split gives that phase enough hosts (Section 4.4).

Our disaggregation recommendation gives further color to existing literature on how RL work should be placed. GPU systems for large models increasingly disaggregate generation from training and run them asynchronously (Zhong et al., 2025a; Fu et al., 2025) or place stages across heterogeneous devices (He et al., 2026); colocated designs instead time-share the same devices across phases (Sheng et al., 2025; Mei et al., 2025; Zhong et al., 2025b), and RLinf finds that the best placement depends on the workload (Yu et al., 2026). To our knowledge, we give the first such comparison for small-model RL on TPU, and it yields concrete evidence that disaggregation is beneficial only when one phase dominates the step and the split gives that phase enough hosts. At small scale rollout still speeds up with every host it gets, so colocation wins (LoCoJAX-DA takes 1.61× the colocated step on GSM8K at 8 chips). At large scale the workload decides, and so does the split. On decode-heavy NuminaMath, rollout dominates. With three quarters of the hosts on rollout, LoCoJAX-DA beats colocation at 64 and 256 chips and is faster without a verdict at 32 and, post hoc, 128, while an even split never does. On prefill-heavy SWE, training dominates and colocation wins at every size: an even split already leaves the trainer half the hosts, and moving more of them to rollout makes LoCoJAX-DA 1.9–4.1× slower. Thus, for small models, colocation is the right default; disaggregation is beneficial on decode-heavy work, but only with a split that gives rollout most of the hosts.

LoCoJAX shows that scheduler defaults built for large SPMD programs can be a poor fit for small-model RL. Our scaling study shows that for small-model RL on TPUs, reducing host-orchestration computational cost is important. In fact, Pathways’ per-step excess over LoCoJAX on GSM8K stays at about 4.1–5.6 s from 8 to 64 chips (Table 3; the head’s serial admission is about 1 s of it) and does not fall systematically as chips are added, so LoCoJAX’s speedup over Pathways on GSM8K grows from 8 to 32 chips as its own step shrinks. Thus, by removing Pathways, we were able to scale our RL jobs more efectively. In summary, LoCoJAX shows why scheduler-side bottlenecks deserve direct attention when training small models.

## 6.1 Limitations

The Pathways runtime is closed source and we were unable to tune it the way we tuned LoCoJAX. The characterization study covers a decode-heavy workload only; behavior may difer on prefill-heavy datasets. The characterization and the stack comparison use diferent setups and are not directly comparable. On GSM8K we use bf16 weights, and on NuminaMath and SWE-agent we keep our stack’s default fp32 trainer weights, since both Pathways and LoCoJAX already produce matching output lengths there. However, some discrepancy in speedup factor may be attributable to the precision diference. LoCoJAX was not run on NuminaMath or SWE-agent configurations where we estimated it would not fit, because it does not accumulate gradients. Finally, LoCoJAX is rigorously evaluated on only one model (Qwen3-0.6B). We did run others (Table E1), but tuning them as heavily as Qwen3-0.6B was infeasible.

## References

T. Bao, J. Carpenter, L. Chai, H. Gao, Y. Jiang, S. Noghabi, et al. Tunix (Tune-in-JAX). Open-source software, 2025. https://github.com/google/tunix. Accessed 2026-10.

P. Barham, A. Chowdhery, J. Dean, S. Ghemawat, S. Hand, D. Hurt, et al. Pathways: Asynchronous Distributed Dataflow for ML. In MLSys, 2022. arXiv:2203.12533.

C. Basile, M. Voznesensky, G. Park, B. Wilson, P. Holloway, Z. Wan, et al. TorchTPU – A Hybrid Eager/Compiled Architecture for Native PyTorch Execution on TPU. arXiv:2610.24161, 2026.

P. Belcak, G. Heinrich, S. Diao, Y. Fu, X. Dong, S. Muralidharan, Y. C. Lin, and P. Molchanov. Small Language Models are the Future of Agentic AI. arXiv:2506.02153, 2025.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: Composable transformations of Python+NumPy programs. https://github.com/jax-ml/jax, 2018.

P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep Reinforcement Learning from Human Preferences. In NeurIPS, 2017. arXiv:1706.03741.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.

Q.-A. Dang and C. Ngo. Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn’t. arXiv:2503.16219, 2025.

DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.

L. Espeholt, R. Marinier, P. Stanczyk, K. Wang, and M. Michalski. SEED RL: Scalable and Eficient Deep-RL with Accelerated Central Inference. In ICLR, 2020. arXiv:1910.06591.

Jared Fernandez, Jacob Kahn, Clara Na, Yonatan Bisk, and Emma Strubell. The Framework Tax: Disparities Between Inference Eficiency in NLP Research and Deployment. In EMNLP, 2023.

W. Fu, J. Gao, X. Shen, C. Zhu, Z. Mei, C. He, et al. AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning. In NeurIPS, 2025. arXiv:2505.24298.

Haoyu Gao, Lance Wang, Shadi Noghabi, Tianshu Bao, and Weiren Yu. Scaling Agentic RL: High-Throughput Agentic Training with Tunix. Google Developers Blog, July 2026. https://developers.googleblog.com/scaling-agentic-rl high-throughput-agentic-training-with-tunix/. Accessed 2026-10.

Gemma Team. Gemma 4 Technical Report. arXiv:2607.02770, 2026.

Google. MaxText: A Simple, Performant, and Scalable JAX LLM. Open-source software, 2023. https://github.com/AI-Hypercomputer/maxtext. Accessed 2026-10.

Google. MaxText GRPO recipe for Gemma 4 26B (run\_gemma4\_26b\_rl.sh). MaxText repository, 2026a. https: //github.com/AI-Hypercomputer/maxtext/blob/3258182fb3570822bb118823621918c677ba9f50/src/maxtext/tr ainers/post\_train/rl/scripts/run\_gemma4\_26b\_rl.sh. Accessed 2026-10.

Google. Tunix distributed RL runtime (dist\_rl). Tunix repository, commits ec932ea1d and ba17165f0, 2026b. https://github.com/google/tunix/tree/ba17165f00fde2b25aaf304850afa4d80e161fe2/tunix/dist\_rl. Added October 22–23, 2026; accessed 2026-10.

Google. Tunix GRPO recipe for Qwen3 (run\_qwen3.sh). Tunix repository, 2026c. https://github.com/google/tunix/ blob/d6893d5ee97a91be6e0027efa048b9c0baf2844f/examples/rl/grpo/gsm8k/run\_qwen3.sh. Accessed 2026-10.

Google Cloud. Building production AI on Cloud TPUs with JAX. Cloud TPU documentation, 2026a. https: //docs.cloud.google.com/tpu/docs/jax-ai-stack. Accessed 2026-10.

Google Cloud. Introduction to Pathways on Cloud. AI Hypercomputer documentation, 2026b. https://docs.cloud.goo gle.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro. Accessed 2026-10.

Google Cloud. Port JAX workloads to Pathways. AI Hypercomputer documentation, 2026c. https://docs.cloud.googl e.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/porting-jax-workloads.

Google Cloud. Run multi-host RL training for Qwen3-30b-a3b on TPU v6e. AI Hypercomputer documentation, 2026d. https://docs.cloud.google.com/ai-hypercomputer/docs/tutorials/tpu/run-qwen3-30b-a3b-rl-maxtext. Accessed 2026-10.

Aaron Grattafiori et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024.

Y. He, S. Zhang, J. Gai, X. Zhang, B. Han, B. Wang, H. Rangwala, and G. Karypis. HetRL: Eficient Reinforcement Learning for LLMs in Heterogeneous Environments. In MLSys, 2026. arXiv:2512.12476.

M. Hessel, M. Kroiss, A. Clark, I. Kemaev, J. Quan, T. Keck, et al. Podracer Architectures for Scalable Reinforcement Learning. arXiv:2104.06272, 2021.

L. Huang, S. Anupam, I. Lee, S. Li, and O. Bastani. RAPID: An Eficient Reinforcement Learning Algorithm fo Small Language Models. arXiv:2510.03515, 2025.

JAX Authors. Introduction to Multi-Controller JAX (aka Multi-Process/Multi-Host JAX). JAX documentation, 2026a. https://docs.jax.dev/en/latest/multi\_process.html. Accessed 2026-10.

JAX Authors. Pallas: a JAX kernel language. JAX documentation, 2026b. https://docs.jax.dev/en/latest/pallas/in dex.html.

J. Jiang, Y. Chen, B. A. Hechtman, F. Zhang, and Y. Mu. Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU. arXiv:2604.15464, 2026.

Norman P. Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, Clif Young, Xiang Zhou, Zongwei Zhou, and David Patterson. TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings. In ISCA, 2023.

Sameer Kumar, James Bradbury, Clif Young, Yu Emma Wang, Anselm Levskaya, Blake Hechtman, Dehao Chen, HyoukJoong Lee, Mehmet Deveci, Naveen Kumar, Pankaj Kanwar, Shibo Wang, Skye Wanderman-Milne, Steve Lacy, Tao Wang, Tayo Oguntebi, Yazhou Zu, Yuanzhong Xu, and Andy Swing. Exploring the Limits of Concurrency in ML Training on Google TPUs. In MLSys, 2021.

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, et al. Eficient Memory Management for Large Language Model Serving with PagedAttention. In SOSP, 2023. arXiv:2309.06180.

N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.

Mark Lee et al. AXLearn: Modular Large Model Training on Heterogeneous Infrastructure. arXiv:2507.05411, 2025.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. NuminaMath. Hugging Face dataset AI-MO/NuminaMath-CoT, 2024. https://huggingface.co/datasets/AI MO/NuminaMath-CoT.

Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, Y. Xiong, E. Chang, Y. Shi, R. Krishnamoorthi, L. Lai, and V. Chandra. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. In ICML, 2024. arXiv:2402.14905.

Z. Mei, W. Fu, K. Li, G. Wang, H. Zhang, and Y. Wu. ReaL: Eficient RLHF Training of Large Language Models with Parameter Reallocation. In MLSys, 2025. arXiv:2406.14088.

P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, et al. Ray: A Distributed Framework for Emerging AI Applications. In OSDI, 2018. arXiv:1712.05889.

Nebius. SWE-agent-trajectories. Hugging Face dataset nebius/SWE-agent-trajectories, 2024. https://huggingface.co /datasets/nebius/SWE-agent-trajectories.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, et al. Training Language Models to Follow Instructions with Human Feedback. In NeurIPS, 2022. arXiv:2203.02155.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.

Gerald Shen, Zhilin Wang, Olivier Delalleau, Jiaqi Zeng, Yi Dong, Daniel Egert, et al. NeMo-Aligner: Scalable Toolkit for Eficient Model Alignment. In COLM, 2024.

G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, et al. HybridFlow: A Flexible and Eficient RLHF Framework. In EuroSys, 2025. arXiv:2409.19256.

Benjamin Spector, Jordan Juravsky, Stuart Sul, Owen Dugan, Dylan Lim, Dan Fu, Simran Arora, and Christopher Ré. Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B. Hazy Research Blog, 2025. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data. arXiv:2410.01560, 2024.

vLLM Project. tpu-inference: TPU Backend for vLLM. Open-source software, 2025. https://github.com/vllmproject/tpu-inference. Accessed 2026-10.

vLLM Team. vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction. vLLM Blog, 2024. https: //vllm.ai/blog/2024-09-05-perf-update.

vLLM Team. vLLM TPU: A New Unified Backend Supporting PyTorch and JAX on TPU. vLLM Blog, 2025.https://vllm.ai/blog/2025-10-16-vllm-tpu.

X. Wan, P. Qi, G. Huang, C. Ruan, M. Lin, and J. Li. Revisiting Parameter Server in LLM Post-Training. In ICLR, 2026. arXiv:2601.19362.

L. Wang. RFC: Multi-Controller JAX (McJAX) Support for Tunix. GitHub discussion #706, google/tunix, 2025. https://github.com/google/tunix/discussions/706. Accessed 2026-10.

S. Wang, J. Asilis, Ö. F. Akgül, E. B. Bilgin, O. Liu, and W. Neiswanger. Tina: Tiny Reasoning Models via LoRA. In ICLR, 2026. arXiv:2504.15777.

An Yang et al. Qwen3 Technical Report. arXiv:2505.09388, 2025.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

C. Yu, Y. Wang, Z. Guo, H. Lin, S. Xu, H. Zang, et al. RLinf: Flexible and Eficient Large-Scale Reinforcement Learning via Macro-to-Micro Flow Transformation. In OSDI, pages 829–846, 2026. arXiv:2509.15965.

W. Zeng, Y. Huang, Q. Liu, W. Liu, K. He, Z. Ma, et al. SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild. In COLM, 2025. arXiv:2503.18892.

Q. Zhong, M. Zheng, M. Song, X. Lin, J. Sun, H. Jiang, et al. SOD: Step-wise On-policy Distillation for Small Language Model Agents. arXiv:2605.07725, 2026.

Y. Zhong, Z. Zhang, X. Song, H. Hu, C. Jin, B. Wu, et al. StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation. arXiv:2504.15930, 2025a.

Y. Zhong, Z. Zhang, B. Wu, S. Liu, Y. Chen, C. Wan, et al. Optimizing RLHF Training for Large Language Models with Stage Fusion. In NSDI, 2025b. arXiv:2409.13221.

J. Zhou, Q. Hu, Y. Jin, Z. Wang, P. Sun, Y. Gu, W. Zhang, M. Zhai, X. Zhang, and W. Zhang. RL in the Wild: Characterizing RLVR Training in LLM Deployment. arXiv:2509.25279, 2025.

X. Zhuang, P. Ma, Z. Jia, Z. Cao, and S. Liu. Efective Learning for Small Reasoning Models: An Empirical Study on 0.5B Reasoning LLMs. In CPAL, 2026. arXiv:2506.13404.

## AI Use Statement

The investigators designed the technical approach, the research methodology, and the experimental plan.   
Claude Opus 5.5 assisted with writing the paper text, implementing code, and writing scripts to collect results.   
All results reported in this paper were validated by the investigators independently of AI assistance.

## Appendix

## A Measurement Protocol and Statistics

Step time. A launch’s step time is its median over steady steps: the second pass (steps 23–38) of a 40-step run that covers the same 20 batches twice. Start-up and compilation are excluded. Where scored steps recompile, we also report the ratio with compile time charged back. A cell averages its launches; ratios are geometric means over launches, or medians of the per-launch comparisons where a table says so.

Launches. Each arm has three launches unless a trial is counted as provisional, in which case less than 3 were run.

Pathways vs LoCoJAX. For each LoCoJAX launch we bootstrap the step-time ratio against the Pathways arm over steady steps (95% percentile interval, 10,000 resamples), per step and per token. “LoCoJAX faster” requires both lower limits above 1.111 in every comparison, equal work (Section 3.3) and three launches per arm.

Within LoCoJAX. LoCoJAX-DA against LoCoJAX-CO uses the 95% Welch interval on log launch values, and on NuminaMath and SWE also the paired-step bootstrap, which must fall on the same side of 1. Ablation rungs instead require disjoint launch ranges and a ratio that difers from 1 by more than either arm’s relative spread.

## B LoCoJAX Optimizations

LoCoJAX optimized as evaluated in Table 3 adds 4 optional optimizations to LoCoJAX base, which already runs the shared optimizations (Table B1). Three act on LoCoJAX’s own per-host learner and the fourth on weight publication.

## C Critical-Path Accounting

This appendix gives our critical-path accounting: the equations and the four tradeofs it explains. Coordination timings follow Section 4.3. Section 4 reports measured wall times; nothing here is used to predict them.

We model Section 4’s steady-state step time as its longest dependent chain:

$$
T _ { \mathrm { s t e p } } ( x ) = \operatorname* { m a x } _ { \pi } \sum _ { v \in \pi } d _ { v } ( x )\tag{1}
$$

where $\pi$ is a path through configuration x’s acyclic one-step graph, with edges for data dependencies and shared resources (colocated devices, the Pathways head). Durations $d _ { v }$ exclude waiting (slack, not computational cost). Decompositions use means; medians do not add.

Decode iteration (per TensorCore, TP degree t):

$$
d _ { \mathrm { i t e r } } ( t ) = \kappa ( t ) + ( Q _ { w } / t + Q _ { \mathrm { K V } } ) / \beta _ { \mathrm { e f f } } + C _ { \mathrm { T P } } ( t )\tag{2}
$$

Admission (n requests, one process of stack s):

$$
\begin{array} { l } { { d _ { \mathrm { a d m } } = \alpha _ { 0 } ^ { s } + \alpha _ { 1 } ^ { s } n , ~ \mathrm { o r } ~ \operatorname* { m a x } \bigl ( \alpha _ { 0 } ^ { s } + \alpha _ { 1 } ^ { s } n _ { h } \bigr ) } } \\ { { \mathrm { w h e n } ~ \mathrm { e a c h } ~ \mathrm { h o s t } ~ h ~ \mathrm { a d m i t s } ~ \mathrm { i t s } ~ \mathrm { o w n } ~ n _ h } } \end{array}\tag{3}
$$

Here κ is the fixed per-kernel time, $Q _ { w }$ the TP1 weight bytes, $Q _ { \mathrm { K V } }$ the per-core KV bytes (size unmeasured; inferred unchanged as t doubles), $\beta _ { \mathrm { e f f } }$ the measured marginal HBM rate, $C _ { \mathrm { T P } }$ the Section 4.1 TP penalty, and $\alpha _ { 0 } ^ { s } , \alpha _ { 1 } ^ { s }$ stack s’s fixed and per-request computational costs. Other operations use measured durations.

Compute is omitted (0.02–0.24 ms per iteration at peak FLOP rate, against 0.70–5.27 ms of TP1 weight time).

Disaggregation (one-step-stale rollouts; H<sub>r</sub> rollout, $H _ { t }$ trainer hosts) and colocation (all H hosts) have steady-state periods (step times)

$$
\begin{array} { l } { { \lambda _ { \mathrm { D A } } = \operatorname* { m a x } \{ D _ { R } ( H _ { r } ) + d _ { L } , ~ D _ { T } ( H _ { t } ) + d _ { \mathrm { p u b } } \} } } \\ { { - A + \tau , } } \\ { { \lambda _ { \mathrm { c o } } = D _ { R } ( H ) + D _ { T } ( H ) + d _ { \mathrm { p u b } } } } \end{array}\tag{4}
$$

where $D _ { R } , D _ { T }$ are rollout and training time, $d _ { \mathrm { p u b } }$ weight publication, $d _ { L }$ rollout weight load, $A \geq 0$ the straggler wait that staleness absorbs, and τ the residual Ray computational overhead (including the unmeasured batch transfer). With rollout always the longer branch, disaggregation is beneficial only if $O + A > F + d _ { L } + \tau .$ with fragmentation $F = D _ { R } ( H _ { r } ) - D _ { R } ( H )$ and hidden training $O = D _ { T } ( H ) + d _ { \mathrm { p u b } }$

## The four tradeoffs:

1. Weight streaming vs TP computational overhead (Eq. 2, Section 4.1, first decode window). TP1 → TP2 saves 0.21 ms per iteration at 0.6B and 1.16 ms at 4B (bytes term at $\beta _ { \mathrm { e f f } } = 3 . 3 2 ~ \mathrm { T B / s } \mathrm { : } ~ 0 . 1 8 , 1 . 2 0 ~ \mathrm { m s } )$ against penalties of 0.68 and 0.77–0.85 ms: TP2 wins at 4B, not 0.6B.

2. Centralized vs host-local admission (Eq. 3, Section 4.3). At 64 chips the Pathways head admits all 2,048 requests serially in about 1.0 s. Each of the 16 LoCoJAX hosts admits its own 128 requests in parallel with the others, in under 0.06 s.

3. Global vs local rollout control $( \tt E q . 1 ,$ , Section 4.3). Pathways runs every decode iteration as one lockstep program across all chips, while each LoCoJAX host steps its own engine. On GSM8K at 64 chips, Pathways takes 2.15 ms per decode iteration against LoCoJAX’s 0.90 ms; the 1.25 ms premium adds about 2.25 s over roughly 1,800 iterations.

4. Colocated vs disaggregated (Eq. 4, Section 4.4). On GSM8K at 8 chips F = 2.9 s exceeds O = 1.1 s and Ray takes 1.6× the colocated step time (one launch per arm, provisional); at 128 chips O = 0.32 s and A = 0.58 s outweigh F = 0.23 s, d<sub>L</sub> = 0.49 s and $\tau = 0 . 0 3 \mathrm { ~ s ~ }$ , and Ray is about 10% faster.

Scope. This is an accounting of measured step times, not a validated predictor: F and O are derived, A simulated, $d _ { L }$ a prior and τ a residual.

Unification. One accounting covers both kinds of change. LoCoJAX versus Pathways changes the graph of Eq. 1: which process serializes per-request work (the Pathways head for all n requests, or each host for its own $n _ { h } , \mathrm { E q . ~ 3 ) }$ and where hosts join (every engine step or only phase boundaries); colocation versus disaggregation changes a resource edge (Eq. 4). The upstream changes mostly keep who serializes and where hosts join. They change node terms instead: the per-iteration terms of Eq. 2, the work a node processes, how many serial calls it issues, local edges and lifecycle nodes within a phase, and which configurations count. Admission and weight-publication slicing (#2232) share the form $\alpha _ { 1 } ^ { s } n$ on one serializing process, each with its own α. Each change is booked once as its change in $T _ { \mathrm { s t e p } } ;$ changes that move the same term are not added. Like Eqs. 1–4, this explains measured steps and predicts none.

Table B1 LoCoJAX optimization levers. Outside this list, both stacks also used LoCoJAX’s sampler, which does not compute the top-k and top-p masks (unused here); LoCoJAX keeps it with its levers of.
<table><tr><td>Lever</td><td>What it changes</td><td>Code it acts on</td><td>Applies to</td></tr><tr><td>01</td><td>Exports metrics off the per-host barrier</td><td>LoCoJAX per-host learner</td><td>LoCoJAX only</td></tr><tr><td>O2</td><td>Freezes Python garbage collection after warmup</td><td>LoCoJAX per-host learner</td><td>LoCoJAX only</td></tr><tr><td>O3</td><td>Balances prompts across hosts by length</td><td>LoCoJAX per-host learner</td><td>LoCoJAX only</td></tr><tr><td>O4</td><td>Publishes all weights to the local engine in one call</td><td>LoCoJAX weight publish</td><td>LoCoJAX only</td></tr></table>

![](images/bbe160269aa3ce14bc6903a87ade55b3a03a6ac58869030d4f5cb8d9ba3733a1.jpg)

![](images/f984fcdb06cb52d626b34fa28d21ecba0388e541ab8c1063a2506079b56960a9.jpg)  
Figure 7 The two recurring controller computational costs of Section 4.3, GSM8K, Qwen3-0.6B, 64 chips. (a) Request admission once per step: the Pathways head admits all 2,048 requests in about 1.0 s, while each LoCoJAX host admits its own 128 requests in under 0.06 s. (b) Ten of about 1,800 decode iterations: each Pathways iteration takes 2.15 ms against 0.90 ms for LoCoJAX, adding about 2.25 s per step.

## D Upstream Contributions

Method. We profiled an end-to-end RL step, found the critical-path node (Appendix C) whose exposed time was large or growing, traced it to a code path and made a targeted change.

The work falls into three kinds:

• Measured or reported benefits: families A–H.

• Integration that makes a benefit reachable from MaxText: #3395, #5236, #5255 and #5289. Their numbers are their family’s benefit, not an extra one.

• Correctness and observability fixes: family I and the packing crash fix #2309. These are not speedups.

Efects are per mechanism, in native units, and not additive. Pairs that move the same term are reported once and never summed: #2231 with #5289, #2086 with #5236, #3102 with #3395, and #2233 with its detokenization tail

What the evaluated runs contain. For each image we checked whether the merge is present, whether its setting is reachable and set, and whether a log line shows the path ran. The frozen recipe has log evidence for families C, E, F and G, and only configuration or code evidence for A, B, D and H.

Acknowledgments note. We thank the tpu-inference, Tunix and MaxText maintainers for reviewing and merging these changes and for the upstream implementations they build on or that covered ours; Loki Chen authored tpu-inference #3101, #3102 and #3174, and Tianyu Gu authored the other 18 merged PRs in Table D2.

## E Supporting Details for Section 4.2

Model size. Tables E1 and E2 compare the frozen Pathways recipe with LoCoJAX optimized, each in its own configuration (on NuminaMath and SWE, also a bf16 against an fp32 trainer), so they are descriptive. LoCoJAX’s advantage shrinks as models grow on every workload, from 1.8–3.3× at 1.7B to 1.0–1.6× at 32B: Pathways adds a roughly fixed time per step, which weighs less as the step grows (Appendix E.1).

8 chips, one wave. At 8 chips LoCoJAX generates in two waves (Table 3). Rerun post hoc in one wave, with two engine settings changed, LoCoJAX optimized is 1.93× and base 1.84× faster than Table 3’s Pathways arm, which ran in a diferent window (descriptive).

## E.1 A Simple Model of the Pathways/LoCoJAX Ratio

Split each step into work that both stacks do, C, and stack-specific time outside it, $O _ { s } \colon T _ { \mathrm { L o C o } } = C + O _ { \mathrm { L o C o } }$ and $T _ { \mathrm { P W } } = a C + O _ { \mathrm { P W } }$ . Eliminating C gives, with $\Delta = O _ { \mathrm { P W } } - a O _ { \mathrm { L o C o } }$

$$
T _ { \mathrm { P W } } = a T _ { \mathrm { L o C o } } + \Delta , \qquad R = a + \frac { \Delta } { T _ { \mathrm { L o C o } } } .\tag{5}
$$

With $a = 1$ , Pathways takes a fixed ∆ longer per step, so the ratio falls as the step grows and rises as LoCoJAX’s step shrinks. We fit ∆ per block of cells and check each cell against the ∆ fitted on the others (Table E3, Figure 8). This accounts for measured step times; it does not predict them.

Where it holds. On GSM8K (Qwen3-0.6B, Pathways vs LoCoJAX base), ∆ stays at 4.0–4.8 s from 4 to 128 chips while LoCoJAX’s step falls from 10.06 to 2.54 s, so the ratio rises from 1.44 to 2.58 and flattens above 32 chips only because LoCoJAX’s step stops shrinking. A free fit gives a = 1.03 [0.89, 1.17]. On NuminaMath at 128 chips, ∆ is 20.5–24.4 s from 1.7B to 14B while LoCoJAX’s step grows $2 . 9 \times ; \Delta$ falls slightly with size, so a = 1 is approximate (free fit a = 0.92 [0.83, 1.00]).

Where it fails. At 32B the measured ratio is 1.03 against 1.15 predicted, but this cell runs rollout at TP8 and only LoCoJAX ofloads activations, so it does not show the gap closing with model size alone. On SWE, ∆ is 13.3 s at 1.7B but 7.5 s at 4B and 8B, so no single ∆ fits; with outputs of about 140 tokens, SWE’s $\Delta$ cannot come mainly from per-iteration decode computational cost. On the GSM8K size ladder, the model is within 0.15 of the measured ratios from 4B to 32B but over-predicts 1.7B (3.53 vs 3.25).

## F Colocated vs. Disaggregated: Details

This appendix gives the median step time of every LoCoJAX-CO cell and LoCoJAX-DA split (Table F1), and some detail behind that section.

NuminaMath and SWE. Colocated LoCoJAX (LoCoJAX-CO) runs a trainer shard and a rollout engine on every host and updates synchronously; Ray (LoCoJAX-DA) splits the slice into trainer and rollout hosts that run concurrently.

No Ray point exists for NuminaMath at 8–16 chips or SWE at 8–16 chips: colocated LoCoJAX does not fit SWE at 8 chips (Table 3), and Ray was not run on NuminaMath at 16 chips (projected trainer memory about 118 GB). At SWE 32 chips the 4/4 split fits (a memory probe peaked at 91.2 of 94.7 GiB per device), while a 2-host trainer (2/6) runs out of memory (157.6 vs 94.7 GiB). That SWE 32-chip cell was added after the main results, in a later window, and is never pooled across windows: with three launches per arm, colocated takes 31.1 s and the 4/4 split 40.4 s per step, a Ray/colocated ratio of 1.299x [1.296, 1.302; 1.268, 1.318] (median basis; Welch; bootstrap), colocated faster (post hoc).

## G Comparison with Published TPU Recipes

Baselines and matching. We found no peer-reviewed results for RL post-training of small LLMs on TPUs, so we compare against the closest public baselines: Google’s reference GRPO recipes for Qwen3-1.7B (Yang et al., 2025) with Tunix (Google, 2026c) and for Gemma4-26B MoE (Gemma Team, 2026) with MaxText (Google, 2026a). We changed none of the recipes’ workload settings. Each LoCoJAX arm reproduces its recipe’s workload: the same checkpoint, tokenizer and prompt template (rendered prompts are byte-identical), the same dataset and per-step prompt order, rewards, sampling (temperature, top-k, top-p, stop tokens), generation cap, prompts × generations per step, and the same GRPO loss with its KL term and estimator. The adapter also matches (LoRA rank 64 for Qwen3, full fine-tuning for Gemma), as do the optimizer and learning-rate schedule. Only the system difers: the controller, placement, rollout engine, weight synchronization, gradientaccumulation layout (one optimizer update per step on both sides), and LoCoJAX’s optimizations. Both sides of each pair run the same JAX, libtpu, vLLM and tpu-inference versions. For Gemma4-26B they also share one container image and MaxText build. For Qwen3-1.7B the recipe keeps its own pinned Tunix commit.

Reference Recipe Results On both of Google’s published TPU RL recipes we ran, LoCoJAX takes less time per training step than the recipe on the same chips. On one host (4 chips), against Tunix’s Qwen3-1.7B recipe with its default sampler, where no multi-host controller is involved, colocated LoCoJAX takes 5.34× less time per step and disaggregated LoCoJAX 3.48× less. On 64 chips, against MaxText’s Gemma4-26B MoE recipe on Pathways over steady steps inside learning-rate warmup, the ratios are 2.14× (colocated) and 1.10× (disaggregated) (Table G1).

Table D1 Upstream changes by mechanism. Efects come from the PR on its author’s workload [PR], our logs [Ours] or earlier measurements not re-verified [Hist.], and are not additive. Active in: Table 3 images in which the change runs. Per-PR detail: Table D2.
<table><tr><td>Family</td><td>PRs</td><td>Bottleneck</td><td>Effect</td><td>Active in</td></tr><tr><td>A. Attention block #3102; #3395 sizes</td><td></td><td>Block-size settings ignored on the MaxText path</td><td>#3395: step 267.9 → 218.6 s [PR]</td><td>Pathways; LoCoJAX uses own tiles</td></tr><tr><td>B. EOS-check control</td><td>#3174,#3509</td><td>Cross-DP EOS all-reduce in every decode iteration, even with the check iteration [PR] off</td><td>#3509: 216 µs of a 5.4 ms</td><td>Both</td></tr><tr><td>C. Reward parallelism</td><td>#1904;#5255，#5235</td><td>Serial host scoring of 2,048 completions</td><td>#1904: reward 29.6 → 3.5 s [PR]; no improvement on our</td><td>Pathways</td></tr><tr><td>D. GC lifecycle</td><td>#2086;#5236</td><td>Full gc.collect() after each weight sync</td><td>workload [Ours] ~1.2 s per step [PR]</td><td>Both</td></tr><tr><td>E. Sequence packing</td><td>#5100, #2309; covered by #2076</td><td>76% of training slots were padding [Hist.]</td><td>#5100: training 15.2 → 6.7 s [PR]</td><td>Pathways; LoCoJAX uses</td></tr><tr><td>F. KV-cache lifecycle</td><td>#2231;#5289</td><td>KV pool freed and reallocated at</td><td>#2231: sync 6.7 → 4.7 s [PR]; our A/B: sync 3.53 → 0.55 s,</td><td>own packer Pathways; LoCoJAX uses</td></tr><tr><td>G. Weight-sync dispatches</td><td>#2232</td><td>One dispatch per scanned layer (308 per sync), ~15 ms each through the</td><td>n=1 [Ours] sync 4.6 → 0.4 s [PR]</td><td>own lever Pathways</td></tr><tr><td>H. Post-processing #2233</td><td></td><td>Pathways proxy Detokenization and log-probs ran after the last request</td><td>~2 s per step [PR]; ours</td><td>Both</td></tr><tr><td>overlap I. Correctness, observability</td><td>#3101,#3374,#3466,</td><td>State leak, compile failure, wrong logits from over-token-budget padding, profiler crashes</td><td>inconclusive None claimed</td><td>#3466 in both</td></tr></table>

Table D2 Per-PR inventory. All merged except #3511, closed in favor of #3490, and #2087, superseded by upstream #2076. #5270 (CI repair) is excluded from all claims.
<table><tr><td>PR</td><td>Short title</td></tr><tr><td>tpu-inference #3101</td><td>Fix leaking mutable default metadata</td></tr><tr><td>tpu-inference #3102</td><td>Env overrides for attention block sizes</td></tr><tr><td>tpu-inference #3174</td><td>EOS-check interval from env</td></tr><tr><td>tpu-inference #3374</td><td>Continue decode on flax nnx; keep compile options</td></tr><tr><td>tpu-inference#3395</td><td>Forward block-size overrides on MaxText path</td></tr><tr><td>tpu-inference #3466</td><td>Never pad past the token budget</td></tr><tr><td>tpu-inference #3509</td><td>Skip EOS reduction when the check is off</td></tr><tr><td>tpu-inference #3510</td><td>Cap request IDs in trace annotation</td></tr><tr><td>tpu-inference #3511</td><td>Opt-in vocab-sharded sampling</td></tr><tr><td>Tunix #1904</td><td>Optional process-pool reward evaluation</td></tr><tr><td>Tunix #2086</td><td>Configurable per-step gc.collect</td></tr><tr><td>Tunix #2087</td><td>Build packed chunks in numpy</td></tr><tr><td>Tunix #2088</td><td>Disable profiler on empty window</td></tr><tr><td>Tunix #2231</td><td>Optionally keep KV cache across weight sync</td></tr><tr><td>Tunix #2232</td><td>One jitted unroll per scanned parameter</td></tr><tr><td>Tunix #2233</td><td>Overlap detokenization and log-probs with decode</td></tr><tr><td>Tunix #2309</td><td>Fix packed-row conversion on TPU</td></tr><tr><td>MaxText #5100</td><td>RL packing and vLLM block-size options</td></tr><tr><td>MaxText #5235</td><td>Linear-time answer extraction</td></tr><tr><td>MaxText #5236</td><td>Forward Tunix GC option</td></tr><tr><td>MaxText #5255</td><td>Picklable reward wrapper and worker count</td></tr><tr><td>MaxText #5270</td><td>Restore SparseCore MoE layout flag</td></tr><tr><td>MaxText #5289</td><td>Forward KV-free flag to Tunix</td></tr></table>

Table E1 Model size at 128 chips: frozen Pathways recipe vs LoCoJAX optimized (descriptive). Times: geometric means of launch median step times; TP: rollout TP of both stacks. Ratio: Pathways/LoCoJAX with Welch 95% interval; at 32B, the per-launch interval, which includes 1. <sup>a</sup> Value of record; times from a confirmatory rerun (1.91). <sup>†</sup> Post-result. <sup>‡</sup> LoCoJAX adopts Pathways’ activation ofload, which slows LoCoJAX only. <sup>b</sup> TP2 and TP4 not measured. <sup>c</sup> Pathways times on a basis that favours Pathways.
<table><tr><td>Qwen3</td><td>TP</td><td>Pathways (s)</td><td>LoCoJAX (s)</td><td>Ratio [95% CI]</td></tr><tr><td>NuminaMath</td><td></td><td></td><td></td><td></td></tr><tr><td>1.7B</td><td>1</td><td>51.1</td><td>26.7</td><td>1.88 [1.86, 1.90]a</td></tr><tr><td>4B</td><td>1</td><td>70.3</td><td>46.5</td><td>1.51 [1.49, 1.53]</td></tr><tr><td>8B</td><td>1</td><td>89.4</td><td>68.9</td><td>1.30 [1.28, 1.32]</td></tr><tr><td>14B</td><td>2</td><td>99.2</td><td>78.5</td><td>1.26 [1.22, 1.30]†</td></tr><tr><td>32B</td><td>8b</td><td>154.1</td><td>149.7</td><td>1.03 [0.95, 1.10]†‡</td></tr><tr><td>SWEc</td><td></td><td></td><td></td><td></td></tr><tr><td>1.7B</td><td>1</td><td>26.1</td><td>12.8</td><td>2.04 [1.68, 2.46]†</td></tr><tr><td>4B</td><td>1</td><td>35.3</td><td>27.8</td><td>1.27 [1.26, 1.28]†</td></tr><tr><td>8B</td><td>1</td><td>43.0</td><td></td><td>35.5 1.21 [1.07, 1.37]†‡</td></tr><tr><td>14B</td><td>2-8</td><td>LoCoJAX does not fit (4 settings)</td><td></td><td></td></tr></table>

Table E2 GSM8K size ladder at 64 chips: frozen Pathways recipe vs LoCoJAX optimized (descriptive). Times: means of launch mean step times; TP: rollout TP of both stacks. Ratio: geometric mean over launches with Welch 95% interval. <sup>†</sup> Post-result, pooled across windows. <sup>a</sup> On launch medians: 2.47 [2.18, 2.80].
<table><tr><td>Qwen3</td><td>TP</td><td>Pathways (s)</td><td>LoCoJAX (s)</td><td>Ratio [95% CI]</td></tr><tr><td>1.7B</td><td>1</td><td>11.73</td><td>3.60</td><td>3.25 [3.16, 3.36]</td></tr><tr><td>4B</td><td>1</td><td>14.79</td><td>6.48</td><td>2.28 [2.17, 2.40]†</td></tr><tr><td>8B</td><td>1</td><td>18.24</td><td>10.57</td><td>1.72 [1.71, 1.73]</td></tr><tr><td>14B</td><td>2</td><td>21.34</td><td>11.32</td><td>1.88 [1.66, 2.14]</td></tr><tr><td>32B</td><td>4</td><td>31.10</td><td>19.67</td><td>1.58 [1.49, 1.67]</td></tr></table>

Where the improvement comes from. On Qwen3-1.7B, nearly all of the improvement is in generation. LoCoJAX decodes a step’s 32 rows in one continuous batch instead of two 16-row loops (about 2.5 s of the improvement), and its engine’s per-iteration computational cost is lower (1.8 vs 8.1 ms; about 4.8 s); together they cut generation from 8.5 to about 1.0 s per step. The trainer improves by 0.15 s, and weight synchronization adds 0.25 s colocated and 0.83 s disaggregated. On Gemma4-26B, colocation gives rollout all 64 chips at the recipe’s per-chip rollout eficiency, and the in-place weight publish takes 0.66 s instead of the recipe’s 12.2 s reshard. Disaggregated LoCoJAX keeps the recipe’s 32 + 32 split. Its rollout is faster than the recipe’s on the same 32 chips (179 vs 238 s), but moving the weights through host memory to the rollout hosts requires 55 s per step, so the net improvement is small.

Limitations. (i) Google publishes the Qwen3 recipe for TPU v6e-8. We ran it on 4 v7x chips (8 JAX devices, the same mesh). (ii) The Pathways server version pinned by the Gemma recipe is incompatible with the client in our environment, so Google’s side uses a newer Pathways release. We also made 15 attention block-size settings explicit at their resolved values, because under the RL configuration they otherwise resolve to empty values and the run fails at start-up. (iii) We report steady-state step time. It excludes start-up and compilation. It also excludes the 13 of the 16 baseline Gemma launches that did not reach the measured window; we did not establish why they stopped, and the cause may lie in our environment or configuration. Google’s three Gemma runs take 293.7, 298.1 and 350.3 s per step at equal work. We quote the median; the mean, 314.0 s, would raise every Gemma speedup by about 5%. (iv) The three runs of each LoCoJAX arm share one sampling seed, and Google’s three Qwen3 runs are bit-identical, so for these arms the intervals cover timing noise, not seed variance.

![](images/275d20e4aabf214600500787674368757c37afc03c215988f0db605001fd6d12.jpg)

![](images/86026dde66b93d2ef8d9577f8db2a6889d414cd0af759a5751b704ce2019b35c.jpg)  
Figure 8 Pathways/LoCoJAX step-time ratio and the model $R = 1 + \Delta / T _ { \mathrm { L o C o } } .$ . (a) Against model size: NuminaMath and SWE at 128 chips and the GSM8K size ladder at 64 chips (frozen Pathways recipe vs LoCoJAX optimized). Lines: the model at each block’s mean $\Delta ;$ shaded: range of $\Delta ;$ hollow: outside the fit (32B, SWE 8B). (b) The same ratios against LoCoJAX step time, plus GSM8K Qwen3-0.6B at 4–128 chips (Pathways vs LoCoJAX base). Dotted blue: NuminaMath fit with a free slope (a = 0.92).

Table E3 Inputs and held-out checks of the model (Appendix E.1). Times: geometric means over three launches per arm (s). Measured: $T _ { \mathrm { P W } } / T _ { \mathrm { L o C o } }$ with Welch 95% interval (not computed for GSM8K). Model: $1 + \Delta / T _ { \mathrm { L o C o } }$ with $\Delta$ fitted on the block’s other cells; <sup>∗</sup> outside the fit. NuminaMath and SWE: frozen Pathways recipe vs LoCoJAX optimized; NuminaMath 1.7B from the confirmatory rerun. GSM8K: an earlier reading; the reading of record difers most at 16 chips (Pathways 8.25 s). <sup>‡</sup> Two generation waves. <sup>§</sup> Additional cell, same head group.
<table><tr><td>Cell</td><td>TPW</td><td> $T _ { \mathrm { L o C o } }$ </td><td></td><td>∆ Measured</td><td>Model</td></tr><tr><td>NuminaMath, 128 chips</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1.7B</td><td>51.12</td><td>26.71</td><td>24.42</td><td>1.91 [1.83, 2.01]</td><td>1.81</td></tr><tr><td>4B</td><td>70.34</td><td>46.53</td><td>23.81</td><td>1.51 [1.49, 1.53]</td><td>1.47</td></tr><tr><td>8B</td><td>89.43</td><td>68.95</td><td>20.48</td><td>1.30 [1.28, 1.32]</td><td>1.33</td></tr><tr><td>14B (TP2)</td><td>99.20</td><td>78.54</td><td>20.66</td><td>1.26 [1.22, 1.30]</td><td>1.29</td></tr><tr><td>32B (TP8) *</td><td>154.08</td><td>149.67</td><td>4.42</td><td>1.03 [1.02, 1.04]</td><td>1.15</td></tr><tr><td colspan="6">SWE, 128 chips</td></tr><tr><td>1.7B</td><td>26.10</td><td>12.82</td><td>13.28</td><td>2.04 [1.68, 2.46]</td><td>1.59</td></tr><tr><td>4B</td><td>35.26</td><td>27.75</td><td>7.51</td><td>1.27 [1.26, 1.28]</td><td>1.48</td></tr><tr><td>8B*</td><td>43.02</td><td>35.55</td><td>7.47</td><td>1.21 [1.07, 1.37]</td><td>1.29</td></tr><tr><td>GSM8K 0.6B, Pathways vs</td><td></td><td></td><td>LoCoJAX base</td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>4 chips</td><td>14.53</td><td>10.06</td><td>4.47</td><td>71.44</td><td>1.42</td></tr><tr><td>8 chips</td><td>9.64</td><td>5.54</td><td>4.10</td><td>1.74</td><td>1.77</td></tr><tr><td>16 chips</td><td>8.08</td><td>3.98</td><td>4.10</td><td>2.03</td><td>2.07</td></tr><tr><td>32 chips</td><td>7.65</td><td>2.88</td><td>4.77</td><td>2.66</td><td>2.44</td></tr><tr><td>64 chips</td><td>6.54</td><td>2.55</td><td>3.99</td><td>2.56</td><td>2.68</td></tr><tr><td>128 chips§</td><td>6.55</td><td>2.54</td><td>4.02</td><td>2.58</td><td>2.69</td></tr></table>

Table F1 Median step time (s), LoCoJAX-CO vs LoCoJAX-DA at three trainer-host shares, Qwen3-0.6B. Fastest per row in bold; –: not run; <sup>†</sup>provisional; <sup>‡</sup>added post hoc, in a later window; <sup>∗</sup>split chosen after the cell’s main result.
<table><tr><td></td><td></td><td colspan="3">LoCoJAX-DA</td></tr><tr><td>Dataset</td><td>Chips LoCoJAX-CO</td><td>50%</td><td></td><td>25%12.5%</td></tr><tr><td>GSM8K</td><td></td><td>5.08 8.20</td><td></td><td></td></tr><tr><td>16</td><td></td><td>4.32</td><td>3.75</td><td>一</td></tr><tr><td>32</td><td></td><td>2.87</td><td>2.63</td><td></td></tr><tr><td>64</td><td></td><td>2.02 1.98</td><td>1.94</td><td>2.56</td></tr><tr><td>128</td><td>1.73</td><td>1.56</td><td>1.88</td><td>2.90</td></tr><tr><td>256</td><td></td><td>1.77 1.62</td><td>1.80</td><td>2.22</td></tr><tr><td>NuminaMath 16</td><td>89.4</td><td></td><td></td><td></td></tr><tr><td>32</td><td></td><td>44.5 56.1</td><td>42.0</td><td>一</td></tr><tr><td>64</td><td></td><td>27.9 34.1</td><td>26.5</td><td>25.1</td></tr><tr><td>128</td><td></td><td>19.6 21.4</td><td>17.0†*</td><td>16.8</td></tr><tr><td>256</td><td></td><td>14.3 14.8*</td><td>12.1</td><td>11.8*</td></tr><tr><td>SWE</td><td>32‡</td><td>31.1</td><td>40.4</td><td></td></tr><tr><td>64</td><td></td><td>16.8 20.9</td><td>40.2</td><td></td></tr><tr><td>128</td><td></td><td>10.0 11.8</td><td>21.0*</td><td>40.6</td></tr><tr><td>256</td><td></td><td>6.3†</td><td>7.2 11.8*</td><td>21.5*</td></tr></table>

## H Full Scaling Study Results

Table H1 gives every cell of the scaling study of Section 4.1, of which Table 2 shows only each model’s fastest TP. All cells use the setup of Sections 3.1 and 3.2: GRPO on Pathways on 128 TPU v7x chips, 256 prompts × 8 generations = 2,048 completions per step, each forced to 8,192 output tokens (decode-heavy), with randomly initialized weights. The sampler holds 1,024 sequences at a time, so each of its DP replicas holds 1,024/DP sequences and rollout takes two decode waves; at 405B, a replica holds 16 sequences at TP16 (what fits in HBM) and 32 at TP32, so rollout takes 8 waves.

## I Computational Cost to Learn

The step-time comparison of Section 4.2 measures time per step, not training outcome. This appendix asks whether the faster step also reaches a quality target sooner. Separate GRPO runs train Qwen3-0.6B on GSM8K (three seeds per cell, 4–128 chips) to a frozen held-out target, and we charge each stack the TPU v7x chip-hours it needs. The main pairing is LoCoJAX base against Pathways, both with the same shared optimizations; “the frozen Pathways recipe” runs without them, and LoCoJAX optimized adds O1–O4 (Table B1). Both stacks train an identical recipe, and the target was fixed before any run.

## I.1 Protocol

Table I3 gives the recipe, which difers from the step-time study’s (Table 1). Both stacks keep fp32 trainer weights and a bf16 sampler, and use the Dr. GRPO advantage, no KL term and one update per batch, so the importance ratio is 1. The target is 85% GSM8K test accuracy at temperature 0.7.

A cell is charged at the first of two consecutive scored checkpoints whose three-seed score clears the target, and its computational cost is chips × wall time from job start to that checkpoint, summed over seeds. Start-up is included; queueing, evaluation and the Pathways CPU head are not, which favours Pathways. Intervals are 95% seed bootstraps with the charged checkpoints held fixed.

Table G1 Google’s published TPU RL recipes vs LoCoJAX (GRPO, TPU v7x, same chips; steady-state seconds per step, three runs per arm). Qwen3-1.7B: steps 4–150, mean of run medians. Gemma4-26B: median of run medians over the steady steps both systems share, within steps 4–19. Speedups per generated token (Qwen3): 5.06× and 3.49×.
<table><tr><td colspan="3"></td><td colspan="2">LoCoJAX</td></tr><tr><td>Model, chips Qwen3-1.7B, 4</td><td>s/step</td><td>Google 9.14</td><td>CO</td><td>DA</td></tr><tr><td>(Tunix)</td><td>speedup</td><td>1.00×</td><td>1.71 5.34×</td><td>2.62 3.48×</td></tr><tr><td>Gemma4-26B, 64 s/step (MaxText)</td><td>speedup</td><td>298.1 1.00×</td><td>139.1 2.14×</td><td>269.2ª 1.10×</td></tr></table>

<sup>a</sup>Compared on the eight steady steps DA shares with Google’s runs, where Google takes 295.8 s.

## I.2 Results

On GSM8K, Pathways needs 1.27–2.77× the chip-hours of LoCoJAX base to reach the target at 4–128 chips (Table I1). The ratio holds when both stacks are charged at the same checkpoint and over the steady segment from update 25 to 275 (1.23–3.01×; Table I2); the 4-chip cell sits on the bar.

The ratios do not rise monotonically with chips, and the one exception traces to start-up. GSM8K at 64 chips (2.33×) sits below 32 chips (2.49×) with overlapping intervals, and the steady segment reverses the order (2.58 against 2.39×). In one cell the rule charges the stacks at diferent checkpoints, in Pathways’ favour: at 16 chips LoCoJAX base misses update 300 by 3 of 3,957 answers (same-checkpoint ratio in Table I1). Two more seeds per cell give 1.89× [1.76, 2.03] and 2.21× [2.05, 2.38] at 16 and 64 chips (n=5).

On GSM8K, start-up (to update 25) holds 24–36% of Pathways’ extra chip-hours at 4, 8, 32 and 128 chips, but with Pathways’ resource-manager CPU limit raised fourfold, a separate pairing finds no start-up diference (0.97–1.41× from job creation to the first update at 4, 16 and 64 chips; every interval includes 1). The steady segment carries the robust part of the gap. O1, O2 and O4 contribute little: LoCoJAX base needs 1.01–1.11× the chip-hours of LoCoJAX optimized on GSM8K.

Held-out quality matches within evaluation noise: Pathways is 0.65 pp higher on GSM8K, below the noise floor of a single checkpoint (Table I6).

## I.3 Limitations

The comparison has two limits. (i) It covers one model and one dataset, Qwen3-0.6B on GSM8K, with three seeds per cell and at most 348 updates. (ii) Pathways kept the template resource-manager CPU limit, which slowed its first weight transfer. Replacing that transfer still gives 1.26–2.54×, but at 4 chips the improvement sits on the 1.25 bar and two sensitivities move a cell below it (Table I4).

Table H1 Full accounting of the scaling study (Section 4.1, Table 2): every model of the study at every TP measured with the study recipe on 128 TPU v7x chips (controls and recipe deviations are listed separately), decode-heavy workload (8,192 forced output tokens), random initialization. Each block starts at the smallest TP that fits (∼100B: TP16 only). Bold: the model’s fastest TP (or its only TP), the row shown in Table 2; ≈: tied with it within run-to-run spread by Table 2’s rule (the runner-up’s and the fastest cell’s run ranges overlap, or their median gap is below the larger range). TP, DP: the sampler’s tensor and data parallelism $( \mathrm { T P } \times \mathrm { D P } = 2 5 6$ TensorCores); m: sequences per replica $( m \times \mathrm { D P } = 1 , 0 2 4$ , so the 2,048 completions take 2 decode waves; 405B: see its block). n: valid runs. Step: median over runs of each run’s steady-step median (2 runs: mean); range: fastest and slowest run, given when $n \geq 2$ Shares of the profiled step, each the median over runs (2 runs: mean): computational overhead = share outside compute; host = host orchestration; network = upper-end estimate, including waits for TP partners and at the DP check; TP coll. = TP collectives, part of network; other = computational overhead − host − network (unassigned decode residual and launch-skew idle, not host orchestration); compute = 100 − computational overhead. Rounded parts may difer from their sum by 0.1.
<table><tr><td colspan="9">m</td><td colspan="5">Share of the step (%)</td><td></td></tr><tr><td>DP</td><td></td><td>Waves</td><td></td><td>n Step (s)</td><td></td><td>Range (s)</td><td>Computational Overhead</td><td>Host</td><td>Network</td><td>TP coll.</td><td>Other</td><td>Compute</td><td>Fastest</td></tr><tr><td colspan="10">Qwen3-0.6B</td><td rowspan="3"></td><td></td><td></td><td></td><td>fastest</td></tr><tr><td>1</td><td>256</td><td></td><td>2 3 3</td><td>68.6</td><td>67.8-69.3</td><td>40.6</td><td>36.4</td><td>2.2</td><td>0.0</td><td>2.0 2.4</td><td>59.4</td><td>≈ (tie)</td></tr><tr><td>2</td><td>128</td><td></td><td>2</td><td>71.4</td><td>67.4-72.2</td><td></td><td>45.8 28.7 22.6</td><td>14.8</td><td></td><td>5.3</td><td></td><td>54.2</td></tr><tr><td>4</td><td>64</td><td></td><td>2 3 1</td><td>74.9</td><td>73.9-77.0</td><td></td><td>44.0</td><td></td><td>17.7</td><td>12.1</td><td>3.7</td><td>56.0</td><td rowspan="3"></td></tr><tr><td>8</td><td>32</td><td>32</td><td>2</td><td>100.4</td><td></td><td></td><td>45.6</td><td>18.6</td><td>24.8</td><td>21.6</td><td>2.2</td><td>54.4</td></tr><tr><td>16</td><td>16</td><td>64</td><td>2</td><td>174.1</td><td>172.5-178.9</td><td></td><td>50.8</td><td>8.3</td><td>38.7</td><td>33.8</td><td>3.7</td><td>49.2</td></tr><tr><td colspan="10">Qwen3-1.7B</td><td colspan="5"></td></tr><tr><td>1</td><td>256</td><td></td><td>2</td><td>79.6</td><td>79.4-84.6</td><td></td><td>29.8</td><td>24.3</td><td>3.5</td><td>0.0</td><td>2.0</td><td>70.2</td><td rowspan="4">fastest ≈ (tie)</td></tr><tr><td>2</td><td>128</td><td></td><td>2 3 1</td><td>80.1</td><td>75.3-80.1</td><td></td><td>43.4 26.5</td><td>14.6</td><td>4.8</td><td>2.4</td><td></td><td>56.6</td></tr><tr><td>4</td><td>64</td><td></td><td>2</td><td>87.3</td><td></td><td></td><td>48.0 21.8</td><td>23.2</td><td>15.5</td><td>3.1</td><td></td><td>52.0</td></tr><tr><td>8</td><td>32</td><td></td><td>2 2</td><td>103.6 182.6</td><td>178.6-182.8</td><td></td><td>44.5 15.0 52.7 10.2</td><td>27.2 39.2</td><td>23.8</td><td>2.3</td><td></td><td>55.5 47.3</td></tr><tr><td colspan="10">16 16</td><td colspan="4">31.7 3.2</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>21.9</td><td></td><td></td><td></td><td></td><td rowspan="4"></td></tr><tr><td>1</td><td>256</td><td></td><td>2 3 3</td><td>123.6 108.5</td><td>118.7-124.2</td><td></td><td>25.4</td><td></td><td>2.2</td><td>0.0</td><td>1.2</td><td>74.6</td></tr><tr><td>2 4</td><td>128</td><td>8</td><td>2</td><td></td><td>106.5-109.2</td><td></td><td>32.2</td><td>17.6</td><td>12.2</td><td>5.4</td><td>2.4</td><td>67.8</td></tr><tr><td>8</td><td>64</td><td>16 32</td><td>2</td><td>118.5 135.3</td><td></td><td>115.2-118.8</td><td>41.6 40.0</td><td>17.7</td><td>20.9</td><td>15.0 23.3</td><td>3.1 2.2</td><td>58.4 60.0</td></tr><tr><td>16</td><td>32 16</td><td>64</td><td>2 2</td><td>236.3</td><td></td><td>232.7–240.4</td><td>50.7</td><td>12.0 8.8</td><td>25.9 38.8</td><td>32.3</td><td>3.2</td><td>49.3</td><td></td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td>3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">1</td><td>0.0</td><td>0.8</td><td></td><td>78.2</td><td></td></tr><tr><td>2</td><td>256 128</td><td></td><td>2 1 2 3</td><td>167.2 137.3</td><td>134.7-140.2</td><td></td><td>21.8 28.5</td><td></td><td>2.7 10.0</td><td>4.3</td><td>2.0</td><td>71.5</td><td rowspan="3">≈ (tie) fastest</td></tr><tr><td>4</td><td>64</td><td></td><td>2</td><td>134.1</td><td>134.0-134.4</td><td></td><td>39.6</td><td></td><td></td><td>13.5</td><td></td><td>60.4</td></tr><tr><td>8</td><td>32</td><td></td><td>2</td><td>151.2</td><td>146.6-152.4</td><td></td><td></td><td></td><td>18.5</td><td></td><td>2.7</td><td>59.2</td></tr><tr><td>16</td><td>16</td><td></td><td>3 2</td><td>247.2</td><td>240.3-248.2</td><td></td><td>40.8 50.2</td><td>15.3 9.4</td><td>23.8 37.9</td><td>21.6 32.9</td><td>1.7 2.9</td><td>49.8</td></tr><tr><td colspan="10">64 3</td><td colspan="5"></td></tr><tr><td>Qwen3-14B</td><td></td><td></td><td></td><td>190.0</td><td></td><td>no TP16 (its 40 attention heads do not divide by 16); TP1 only with non-default attention tiles, listed separately</td><td></td><td>13.7</td><td>9.0</td><td>3.5</td><td>1.5</td><td></td><td rowspan="3"></td></tr><tr><td>2</td><td>128</td><td>8</td><td>2</td><td></td><td></td><td></td><td>24.1</td><td></td><td></td><td></td><td></td><td>75.9</td></tr><tr><td>4 8</td><td>64</td><td>16 32</td><td>2 2</td><td>3 3</td><td>168.9 178.6</td><td>163.7-169.6 178.0-179.4</td><td>34.2</td><td>14.8 13.2</td><td>17.4 22.9</td><td>12.2</td><td>2.0 1.5</td><td>65.8 62.4</td></tr><tr><td colspan="10">32</td><td colspan="5">20.4</td></tr><tr><td>Qwen3-32B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="3"></td></tr><tr><td>2</td><td>128</td><td></td><td>2</td><td>350.6</td><td></td><td></td><td>17.9</td><td>7.9</td><td>8.7</td><td>3.0</td><td>1.3</td><td>82.1</td></tr><tr><td>4 8</td><td>64</td><td>16</td><td>2</td><td></td><td>301.3 303.8</td><td>298.4-302.5</td><td>29.2</td><td>8.3</td><td>18.6 24.5</td><td>10.6 18.5</td><td>2.3</td><td>70.8 67.2</td></tr><tr><td>16</td><td>32 16</td><td></td><td>2 2</td></table>

Table H2 Controls and diagnostics of the scaling study, kept out of Table H1 and never pooled with it; columns and statistics as there. (a) Wave control: Synthetic ∼200B at TP16 with 16 sequences per replica, so 8 decode waves as at 405B TP16 (grid row: m = 64, 2 waves). Workaround controls: the 405B memory workarounds applied to Synthetic ∼200B (TP16: the compiler flag and freeing the sampler’s weight copy and the KV cache, without remat; TP32: q/k/v remat), compared with steps 1–3 of the grid runs of the same TP (728.4 s at TP16, 963.8 s at TP32). <sup>∗</sup> Four-step control run, no steady readout by design: median over steps 1–3 (after the warm-up step 0) of the step plus the following weight sync (the readout’s basis); not profiled, so no breakdown. Weight pinning (with freeing): both attempts failed in the first weight sync, refused on Pathways. (b) Llama-3.1-8B architecture, randomly initialized, compared with Qwen3-8B at TP16 in Table H1. (c) Qwen3-14B at TP1 with a diferent attention-tile setting (a recipe deviation); with the default tiles 14B runs from TP2. (d) 16 replicas of TP8 on 64 chips; chip count and per-replica work difer from the 128-chip grid, so they are not comparable with Table H1. GB200 runs (another platform, qualitative only) are not shown.
<table><tr><td rowspan="2"></td><td colspan="7"></td><td colspan="6">Share of the step (%)</td><td></td></tr><tr><td></td><td>TP DP</td><td>m Waves</td><td></td><td>n Step (s)</td><td>Range (s)</td><td>Computational Overhead</td><td></td><td>Host Network TP coll. Other Compute Note</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Run set</td><td colspan="7"></td></tr><tr><td>(a) Synthetic ∼200B controls for the Wave control (m = 16)</td><td></td><td>405B 16</td><td>rows 16</td><td>(128 chips) 8 1</td><td>1,668.8</td><td></td><td></td><td>40.9</td><td>1.7</td><td>39.1</td><td>32.8</td><td>0.1</td><td></td><td>59.1 8 waves, as 405B TP16</td></tr><tr><td>TP16 package without remat</td><td>16 16</td><td>16</td><td>64</td><td>2 1*</td><td>742.6*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>+1.9% vs grid, same steps</td></tr><tr><td>TP32 q/k/v remat</td><td>32</td><td>8</td><td>128</td><td>2 1*</td><td>962.3*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-0.2% vs grid, same steps</td></tr><tr><td>TP16 weight pinning (PIN)</td><td>16</td><td>16</td><td>64</td><td>20/2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>refused on Pathways</td></tr><tr><td colspan="10">(b) Model-family control at TP16, 128 chips (never pooled with Qwen3-8B)</td><td colspan="7"></td></tr><tr><td colspan="10">Llama-3.1-8B shape 16 16 64 2 3</td><td colspan="7">9.4</td></tr><tr><td colspan="10"></td><td colspan="7">38.3 32.8</td></tr><tr><td colspan="10">(c) Deviation, 128 chips (never pooled)</td><td colspan="7">1.2 0.0</td></tr><tr><td>Qwen3-14B, other attention tiles</td><td>1 256</td><td>4</td><td></td><td>2 1</td><td>246.7</td><td></td><td colspan="7"></td><td>0.8</td><td>85.9 default tiles: TP2 up</td></tr><tr><td colspan="10">(d) Legacy TP8 × DP16 rows on 64 chips</td><td colspan="7"></td></tr><tr><td>Qwen3-0.6B</td><td>8</td><td>16 64</td><td></td><td>1</td><td>146.8</td><td></td><td colspan="7">(chip count differs from the 128-chip grid: confounded)</td><td>4.0</td><td>62.6</td></tr><tr><td>Qwen3-1.7B</td><td>8</td><td>64</td><td></td><td>1</td><td>150.2</td><td></td><td></td><td>36.7</td><td>12.9</td><td>19.8</td><td>17.7</td><td>4.0</td><td></td><td>63.3</td></tr><tr><td>Qwen3-4B</td><td>8</td><td>16 16 64</td><td></td><td>2 2 1</td><td>194.1</td><td></td><td></td><td>31.2</td><td>7.8</td><td>19.5</td><td>17.1</td><td>3.8</td><td></td><td>68.8</td></tr><tr><td>Qwen3-8B</td><td>8</td><td>16</td><td>64</td><td>2 1</td><td></td><td>208.8</td><td></td><td>30.3</td><td>8.9</td><td>17.8</td><td>15.9</td><td>3.6</td><td></td><td>69.7</td></tr><tr><td>Qwen3-14B</td><td>8</td><td>16</td><td>64</td><td>2</td><td></td><td>250.4</td><td></td><td>28.2</td><td>7.3</td><td>17.6</td><td>15.6</td><td></td><td>3.4</td><td>71.8</td></tr><tr><td>Qwen3-32B</td><td>8</td><td>16</td><td>64</td><td>2</td><td>1 1</td><td>420.5</td><td></td><td>23.2</td><td>4.3</td><td>15.1</td><td></td><td>14.2</td><td>3.7</td><td>76.8</td></tr><tr><td>Llama-3.1-70B</td><td>8</td><td>16</td><td>64</td><td>2</td><td>1</td><td>673.8</td><td></td><td>29.6</td><td>3.2</td><td>23.5</td><td></td><td>20.5</td><td>2.8</td><td>70.4</td></tr></table>

Table I1 TPU v7x chip-hours to the frozen held-out target, summed over 3 seeds, LoCoJAX base and Pathways with the same shared optimizations. Under the dataset: target and ∆, held-out quality of Pathways minus LoCoJAX (Table I6). Ratio: Pathways over LoCoJAX base [95% seed bootstrap]; bar 1.25 (pre-declared). <sup>m</sup>On the bar (interval 1.245–1.289). <sup>a</sup>LoCoJAX charged at update 325, Pathways at 275 (same checkpoint: 1.84).
<table><tr><td>Dataset</td><td></td><td>Chips LoCoJAX Pathways Ratio [interval]</td><td></td><td></td><td></td></tr><tr><td>GSM8K</td><td>4</td><td>23.3</td><td></td><td>29.6 1.27m</td><td></td></tr><tr><td>85% (T0.7)</td><td>8</td><td>28.3</td><td></td><td></td><td>43.0 1.52 [1.42, 1.63]</td></tr><tr><td>∆ +0.65 pp</td><td>16</td><td>48.0a</td><td></td><td></td><td>77.6 1.62 [1.49, 1.75]</td></tr><tr><td></td><td>32</td><td>64.7</td><td></td><td></td><td>160.7 2.49 [2.30, 2.64]</td></tr><tr><td></td><td>64</td><td>121.0</td><td></td><td></td><td>282.1 2.33 [2.14, 2.48]</td></tr><tr><td></td><td>128</td><td>228.4</td><td></td><td></td><td>633.0 2.77 [2.49, 3.00]</td></tr></table>

Table I2 Full grid, LoCoJAX base against Pathways with the same shared optimizations. Chip-hours to the target (3 seeds summed) with the charged checkpoint pair; ratio Pathways over LoCoJAX base [95% seed bootstrap]. Same checkpoint: both stacks charged at each update of the earlier stack’s pair. Steady: chip-hours from update 25 to 275. <sup>i</sup>Interval 1.201–1.263. Status: against the pre-declared 1.25 bar.
<table><tr><td>Dataset</td><td>Chips</td><td>LoCoJAX base (pair)</td><td>Pathways (pair)</td><td></td><td>Ratio [interval]</td><td></td><td>Same checkpoint</td><td>Steady</td><td>Status</td></tr><tr><td rowspan="6">GSM8K</td><td>4</td><td>23.326 (275/300)</td><td>29.550 (275/300)</td><td>1.267</td><td>[1.245, 1.289]</td><td>1.267</td><td>1.264</td><td>1.232i</td><td>on the bar</td></tr><tr><td>8</td><td>28.300 (275/300)</td><td>42.967 (275/300)</td><td>1.518</td><td>[1.424, 1.629]</td><td></td><td>1.518 1.509</td><td>1.493</td><td>supported</td></tr><tr><td>16</td><td>48.013 (325/348)</td><td>77.591 (275/300)</td><td>1.616</td><td>[1.492, 1.754]</td><td>1.838</td><td>1.841</td><td>1.757</td><td>supported</td></tr><tr><td>32</td><td>64.667 (275/300)</td><td>160.711 (275/300)</td><td>2.485</td><td>[2.301, 2.640]</td><td>2.485</td><td>2.499</td><td>2.394</td><td>supported</td></tr><tr><td>64</td><td>120.996 (275/300)</td><td>282.116 (275/300)</td><td>2.332</td><td>[2.138, 2.477]</td><td>2.332</td><td>2.368</td><td>2.578</td><td>supported</td></tr><tr><td>128</td><td>228.373 (275/300)</td><td>632.996 (275/300)</td><td>2.772</td><td>[2.495, 2.999]</td><td>2.772</td><td>2.811</td><td>3.014</td><td>supported</td></tr></table>

Table I3 Computational cost-to-learn recipe (Qwen3-0.6B, GRPO, 3 seeds per cell), identical on both stacks. $\mathrm { ~ P ~ } \times \mathrm { ~ G } \colon$ prompts × generations per update; caps in tokens. Scored checkpoints: updates 275, 300, 325 and 348.
<table><tr><td></td><td>Dataset Training data (revision)</td><td>P × G</td><td></td><td>Caps Updates Evaluation</td><td>Target</td></tr><tr><td></td><td>GSM8K openai/gsm8k @740312ad: first 7,424 training rows, 6 epochs</td><td>128 × 16 256 / 7,936</td><td></td><td>348 test split (1,319 rows), T 0.7</td><td>3-seed sum ≥ 3,364 of 3,957 (85%) at updates 275–348</td></tr></table>

Table I4 Sensitivities of the grid, hypothetical on measured inputs. Values are Pathways over LoCoJAX base; chip counts in parentheses. The template resource-manager CPU limit slowed Pathways’ first weight transfer to about 8–11 min in the 16- and 32-chip GSM8K seeds.
<table><tr><td>Sensitivity</td><td>Effect</td><td>Verdict change</td></tr><tr><td>Slow first weight transfer replaced (raised limit) or removed</td><td>GSM8K 1.265 / 1.263 (4), 1.508 (8), 1.516 (16; same checkpoint 1.724), 2.239 (32), 2.260–2.308 (64), 2.541 (128)</td><td>none</td></tr><tr><td>Limit&#x27;s effect on steady steps (1.031–1.062; 1.062 [1.000, 1.155] at n=3, 1.031 [0.980, 1.090] at</td><td>GSM8K (4) 1.21–1.24; every other cell ≥ 1.39 with all Pathways chip-hours divided by 1.062</td><td>GSM8K (4) below the bar</td></tr><tr><td>n=5, inconclusive) One-checkpoint counterfactuals at</td><td>GSM8K 1.838 (16), 2.342 (32), 2.214 (64)</td><td></td></tr><tr><td>near-threshold margins Head placement: off-reference-group Pathways seeds slow head nodes 2.530)</td><td>GSM8K 1.218 (4), 1.399 (8), 2.384 (32), 2.156 (64), 2.566 (128;</td><td>none GSM8K (4), whose seeds show no</td></tr></table>

Table I5 Other pairings of the same runs: chip-hour ratios to the target per cell (3 seeds summed); only the first column is a fair comparison. Fair: Pathways over LoCoJAX base. Own best: Pathways over LoCoJAX optimized. Frozen / Pathways: what the shared optimizations save Pathways. Base / optimized: what O1, O2 and O4 save LoCoJAX. Frozen / optimized: the product of the Fair, Frozen / Pathways and Base / optimized columns.
<table><tr><td>Dataset</td><td>Chips</td><td>Fair</td><td>Own best</td><td>Frozen / Pathways</td><td>Base / optimized</td><td>Frozen / optimized (not fair)</td></tr><tr><td rowspan="6">GSM8K</td><td>4</td><td>1.267</td><td>1.281</td><td>1.385</td><td>1.011</td><td>1.774</td></tr><tr><td>8</td><td>1.518</td><td>1.554</td><td>1.325</td><td>1.023</td><td>2.058</td></tr><tr><td>16</td><td>1.616</td><td>1.711</td><td>1.233</td><td>1.059</td><td>2.110</td></tr><tr><td>32</td><td>2.485</td><td>2.618</td><td>1.162</td><td>1.053</td><td>3.043</td></tr><tr><td>64</td><td>2.332</td><td>2.590</td><td>1.133</td><td>1.111</td><td>2.934</td></tr><tr><td>128</td><td>2.772</td><td>3.064</td><td>1.028</td><td>1.106</td><td>3.150</td></tr></table>

Table I6 Held-out quality, Pathways minus LoCoJAX (Welch 95% CI; runs per stack). Pooled over chip counts, which share seeds and data order, so the interval is optimistic. Floor: evaluation noise of one checkpoint.
<table><tr><td>Dataset (score)</td><td>Pathways - LoCoJAX</td><td>Floor</td></tr><tr><td>and 300)</td><td>GSM8K (T 0.7, mean of updates 275 +0.65 pp [+0.28, +1.02] (18 v 18)</td><td>~1 pp</td></tr></table>