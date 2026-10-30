# DWISE: DENOISING WORKLOAD-INFORMED SCHEDULING ENGINE

Lie Li <sup>1</sup> Wen Li <sup>1</sup> Junxiao Shen <sup>1</sup> Jindong Gu <sup>2</sup> Guosheng Hu <sup>1</sup>

## ABSTRACT

Unlike autoregressive large language models (AR-LLMs), diffusion LLMs (dLLMs) generate text by iteratively denoising many tokens in parallel, offering a promising path to efficient inference. Existing serving frameworks, however, schedule dLLM requests with policies designed for AR models, namely first-come-first-served (FCFS) admission and prefill priority. We find that both policies are ill-suited to dLLMs, wasting much of their speed. FCFS ignores that a dLLM request’s decoding cost is highly heterogeneous and unknown at arrival, causing severe head-of-line blocking. Prefill priority, widely used in AR-LLM serving to keep time-to-first-token low, sharply cuts dLLM efficiency, as rounds that run only prompts leave most of the batch idle. We present DWISE, a Denoising Workload-Informed Scheduling Engine for efficient dLLM serving. DWISE schedules requests by their remaining work, measured in a block-level unit shared by prefill and denoising, so both phases share a round. To predict the unknown denoising workload, DWISE combines a lightweight prompt-based predictor at admission with a single refinement from the first denoising round’s hidden states, which capture model-specific information absent from the prompt and require no additional model execution. Implemented in SGLang and evaluated on five models from three dLLM families under bursty workloads, DWISE reduces mean end-to-end latency over SGLang by 5.4× on average and up to 14.9× while preserving task accuracy. The code will be released publicly.

## 1 INTRODUCTION

Recently, diffusion large language models (dLLMs) have emerged rapidly as an alternative to autoregressive (AR) LLMs (Brown et al., 2020). Since 2025, they have grown from 8B research models (Nie et al., 2025) to 100Bparameter releases (Bie et al., 2025; 2026b), and Google has released the open-weight DiffusionGemma, fine-tuned from Gemma 4 (DiffusionGemma Team et al., 2026). Instead of emitting one token at a time, a dLLM iteratively denoises a fully masked sequence, resolving many tokens in parallel per round (Austin et al., 2021a; Sahoo et al., 2024; Shi et al., 2025). At comparable model scale, dLLMs are now competitive with AR models on language, mathematical reasoning, and coding tasks (Ye et al., 2025), while parallel decoding removes the token-by-token bottleneck for much faster generation (Inception Labs et al., 2025).

This speed matters most in deployment, where a model serves large and variable streams of requests, making inference efficiency a major systems concern: already in 2021, inference took 70% of Meta’s AI infrastructure power capacity, 3.5× that of training (Wu et al., 2022). dLLMs thus offer a promising path to more efficient inference than AR LLMs, and the serving system must turn faster generation into high throughput and low latency. Serving systems have begun to adapt: frameworks such as SGLang (Zheng et al., 2024) now support diffusion decoding, and dedicated dLLM systems target memory (Fan et al., 2026), recurring prefills from KV refresh (Kedia et al., 2026), and SLO attainment (Chang et al., 2026). Yet request scheduling in these frameworks has not adapted: under policies inherited from AR serving, prefill takes priority over decoding, in rounds of its own, leaving most of the batch empty while running requests wait. Long-context agentic workloads (Bie et al., 2026a) make this worse: longer prompts take more such rounds. In serving, dLLMs thus lose much of their speed, and recovering it requires scheduling built for how they generate text, which remains largely unexplored.

These inherited policies decide which requests run, wait, or retain accelerator resources per iteration (Yu et al., 2022), and thus affect serving throughput and request latency. Two of them make requests wait: first-come-first-served (FCFS) admission (Kleinrock, 1975), which ignores execution cost, and prefill priority (Agrawal et al., 2024), under which processing new prompts (prefill) precedes generating output (decoding). FCFS is attractive because it requires no prior knowledge of request cost, but under heterogeneous workloads it allows long-running requests to occupy scarce batch capacity while shorter requests queue behind them, and the resulting head-of-line (HOL) blocking inflates latency (Wu et al., 2024). Multi-level feedback queues (MLFQ) mitigate this blocking without such knowledge by demoting requests as they run, so short requests finish first (Corbato´ et al., 1962). The wait under prefill priority, in contrast, is a deliberate trade-off that pays off for AR models: running requests pause for one step, but the new request gets its first token in that same step.

![](images/6e0c99f9656b0ed017ab22a938b8d09961a5aad63945700b7db575a96f9092b5.jpg)  
Figure 1. The first denoising round reveals remaining work. Spearman ρ (Spearman, 1904) between true denoising workload and estimates from prompt features, the prompt hidden state, and the first-round hidden state on LLaDA2.1-mini (five-fold CV).

In practice, a dLLM denoises its output one block at a time, not all at once (Arriola et al., 2025). A block is a fixed-size span of consecutive tokens (e.g., 32). Each output block starts fully masked, and every denoising round unmasks, in parallel, the tokens the model is confident about (Wu et al., 2025), so a block needs a variable number of rounds. During serving, prompts are also split into blocks, and each round is one batched forward pass over one block per selected request: a prompt block (prefill) or its current output block (decoding). Hence, dLLMs break two properties AR scheduling relies on: that output length measures work, and that processing a prompt yields the first token.

Losing these properties raises two scheduling challenges: (1) Accurate workload prediction, which efficient scheduling requires, is hard for dLLMs. Prefill work is determined by the prompt length, but decoding time depends on the number of denoising rounds, which is unknown at arrival and highly heterogeneous, ranging from a few to hundreds of rounds across requests. Length-based LLM schedulers instead predict output length from the prompt (Jin et al., 2023; Zheng et al., 2023; Fu et al., 2024) or during generation (Shahout et al., 2024). A size-based scheduler therefore has no reliable estimate to act on. (2) For dLLMs, prefill priority keeps the cost but loses the benefit: the first output block still needs several rounds, which later prompts keep deferring. Admitting a new request stalls running requests for as many rounds as its prompt has blocks. Yet prompt and output blocks can share a round, so both challenges concern one decision: which blocks run each round.

To address these challenges, we present DWISE, a Denoising Workload-Informed Scheduling Engine that predicts denoising workload from the model’s own hidden states and schedules prefill and decoding as the same block-level work. For challenge (1), we observe that a prompt-based predictor sees only the input tokens and misses how the model will process them. The model’s hidden states capture both, and those after the first denoising round predict the remaining rounds significantly better than any prompt-side signal (Figure 1). Since every request runs this round anyway, DWISE estimates denoising workload from prompt features at admission and refines the estimate once from the first-round hidden states, without extra model execution. For challenge (2), DWISE drops prefill priority: it measures a request’s remaining work in blocks, as its remaining prefill blocks plus its estimated denoising rounds, and fills each round with the requests that have the least remaining work, whether they are in prefill or decoding. New arrivals thus no longer stall running requests, and short requests finish first.

In summary, we make the following contributions.

• Based on the intrinsic difference between block-wise (dLLM) and token-wise (AR) text generation, we design DWISE, to our knowledge the first dLLM scheduler that orders requests by their remaining work measured in denoising rounds, not output tokens. Instead of prioritizing prefill as AR serving does, DWISE counts prefill blocks in the same unit and serves the requests with the least remaining work first.

• We show that a dLLM’s first denoising round reveals what the prompt cannot. Its hidden state raises the rank correlation with remaining work from 0.74 to 0.87 (Figure 1), and from 0.16 to 0.64 within a request category, where prompt features barely order requests (Table 7). DWISE refines each estimate once, and the gain holds for unseen categories.

• We implement DWISE in SGLang and evaluate it on five models from three dLLM families against FCFS and five prior LLM schedulers. It achieves the lowest mean latency in every setting, reducing it over SGLang by 5.4× on average (up to 14.9×) and raising throughput by up to 2.2×, with accuracy within one point.

## 2 BACKGROUND AND RELATED WORK

## 2.1 dLLM Serving

Diffusion Language Models. Diffusion models (Ho et al., 2020; Sohl-Dickstein et al., 2015), first popularized in computer vision, generate through iterative denoising, and discrete diffusion formulations (Austin et al., 2021a) extend them to language tokens. Recent dLLMs, such as the LLaDA family (Nie et al., 2025), scale masked diffusion to large language models and denoise multiple token positions in parallel. To retain this parallelism while supporting causal, variable-length generation, block-diffusion models (Arriola et al., 2025) factorize the output into a sequence of blocks: blocks are generated causally, while tokens within each block are denoised in parallel.

Serving Execution. To sustain high GPU utilization under dynamic request arrivals, LLM serving systems use continuous batching, as in vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024). In autoregressive models, requests first process their prompts during prefill and then generate one token per decoding iteration. Prefill and decoding differ in execution characteristics (Patel et al., 2024; Zhong et al., 2024), motivating chunked prefill in Sarathi-Serve (Agrawal et al., 2024) to reduce interference. dLLMs, however, generate through iterative denoising (Wu et al., 2025; Liu et al., 2026). In SGLang’s dLLM path, each serving round is one batched forward pass, and each selected request processes one block, prompt or output. Prefill advances through a known number of blocks, whereas decoding an output block takes a variable number of denoising rounds. We refer to the rounds a request still needs as its denoising workload.

## 2.2 Scheduling Optimization for LLM Serving

Request Scheduling in Shared Systems. When job sizes are known, shortest-job-first (SJF) (Smith, 1956) prioritizes shorter jobs, while shortest-remaining-processing-time (SRPT) (Schrage, 1968) extends this principle through preemption and minimizes mean response time under standard assumptions. In practice, however, exact job sizes are often unavailable before execution. Non-clairvoyant policies such as multilevel feedback queues (MLFQ) (Corbato et al.´ , 1962) and other age-based policies (Scully et al., 2018) infer job characteristics from observed execution progress, while prediction-based policies (Mitzenmacher, 2019) replace unknown service times with estimated ones.

Request Scheduling for LLM Serving. LLM serving faces the same problem, since generation cost is unknown at arrival. FastServe (Wu et al., 2024) infers priority from attained service with a preemptive skip-join MLFQ. Prediction-based approaches estimate size from the prompt: $S ^ { 3 }$ (Jin et al., 2023) and response-length-aware scheduling (Zheng et al., 2023) predict output length, and LTR (Fu et al., 2024) predicts relative order. TRAIL (Shahout et al., 2024) refines the estimate during execution from intermediate representations. All of them use remaining output tokens as a proxy for remaining work. This proxy does not hold for dLLMs, whose output blocks each take a variable number of denoising rounds. dLLM serving systems have so far targeted memory (Fan et al., 2026), recurring prefills from KV refresh (Kedia et al., 2026), or SLO attainment (Chang et al., 2026), not ordering by remaining work.

![](images/df8655bb73fcd882847a9f1027553102cf8b8a573ad223f2c2e065f917474893.jpg)  
Figure 2. Denoising workload varies substantially across requests (LLaDA2.1-mini). Boxes ${ \mathsf { p 2 5 - p 7 5 } } ,$ line median, whiskers p5–p95.

## 3 MOTIVATION

For a dLLM request, the work that matters to scheduling is not how many tokens it will produce but how many denoising rounds it still needs. Policies inherited from AR serving overlook three properties of this workload: first, it varies by more than 100× across requests (Section 3.1); second, the first denoising round already predicts it far better than the prompt does (Section 3.2); and third, its blocks occupy the same kind of slot in a round as prefill blocks (Section 3.3).

## 3.1 Denoising Workload Is Highly Heterogeneous

Denoising workload varies by up to two orders of magnitude across requests in the models and workloads we study. In the LLaDA2.1-mini workload of Figure 2, a quarter of requests finish within 14 denoising rounds, the median needs 46, and the 90th percentile needs 149. Task type explains part of this spread: instruction-following and chat requests dominate the long tail, yet requests of the same task type still span a wide range of rounds. This heterogeneity is the condition under which size-based scheduling pays off: under arrival-order scheduling, a request that needs 14 rounds can wait behind one that needs 149. Exploiting it requires the right measure of size. Two requests that emit the same number of blocks can consume very different amounts of GPU time, because each block is resolved over its own number of denoising rounds. Generation-side work in dLLMs should therefore be measured in denoising rounds rather than output tokens.

## 3.2 First Denoising Round Reveals Remaining Work

A request’s remaining denoising workload is unknown at arrival, but one round of execution reveals much of it. At admission, a scheduler sees only the prompt, which is what existing length-aware schedulers predict from. The prompt, however, carries only the input tokens, not how the model will process them. Under the threshold-based parallel decoding used by LLaDA2.0 (Bie et al., 2025), a block finishes once every position’s predicted token clears the confidence threshold, so the number of rounds it takes depends on how quickly the model’s confidence rises during denoising, a trajectory that becomes visible only once denoising begins.

![](images/4b7f255d9b59e0dce9749be4d66c24a2bdccc16b7f09f39d384a8a0293191be5.jpg)  
Figure 3. Prefill-first execution can stall denoising, while both phases share the same block-level scheduling granularity.

The first denoising round exposes this trajectory. On LLaDA2.1-mini (Figure 1), the hidden state after this round reaches a Spearman correlation of 0.868 with true denoising workload, against 0.740 for prompt features and 0.789 for the prompt hidden state, and the same holds across all five models we study (Appendix A). The signal is also free: the first round is a forward pass the request executes anyway. A dLLM scheduler can therefore rank a request from its prompt at admission and correct that ranking once, after the first denoising round. This correction comes early enough to matter. After one round, the median request in Figure 2 still has 45 of its 46 rounds ahead, so a request whose promptbased estimate was wrong can still be reordered before most of its work is done.

## 3.3 Prefill and Denoising Blocks Can Share a Round

A request’s remaining service also includes prompt prefill, which, unlike denoising, is known exactly from the number of unprocessed prompt blocks. SGLang’s dLLM path inherits the prefill-first policy of autoregressive serving, where admitting new prompts quickly protects first-token latency (Gao et al., 2024; Sun et al., 2024). Each serving round is either a prefill round or a denoising round, and whenever prompts are waiting, the round goes to prefill and advances no denoising requests. Under a backlog, prompts keep coming, so denoising is interrupted round after round. A short request that remaining-work ordering would finish first instead waits behind new prompts. Prefill rounds also waste capacity: a waiting prompt enters the pool only as a running request finishes, and each prompt advances one block per round, so a prefill round carries one or two blocks and leaves its other slots idle (Figure 3). Section 6.3 shows that these nearly empty rounds dominate SGLang’s runtime.

Block diffusion removes the reason for this separation. AR serving separates the two phases because they differ in cost: a prefill step processes a whole prompt, a decode step only one token, so mixing them slows decoding. In block diffusion, a prefill step and a denoising step each process one block of tokens against the KV cache of earlier blocks, so they cost the same and can share a round. The idle slots of a prefill round can then carry denoising blocks, and a scheduler can fill each round by how much work each request has left, whichever phase it is in.

![](images/96bbb840dd18789cfcf68d59abeab6361367243597d169ed46174657a647be46.jpg)  
Figure 4. Overview of DWISE. Toy example illustrating blocklevel scheduling under SGLang, MLFQ, and DWISE.

## 4 DWISE: KEY IDEAS

DWISE is a runtime scheduler for block-diffusion LLM serving. Its key idea is to treat a prefill block and a denoising block as the same unit of work, since the two cost the same. A request’s remaining work is its unprocessed prefill blocks plus its remaining denoising rounds, and both phases compete for the same block slots of each forward pass. Since denoising rounds are unknown at arrival, DWISE first estimates them from the prompt. Before each round, it ranks requests by remaining work and fills the slots from the top, pausing any request that drops out, with its KV cache kept. After the first denoising round, a diffusion-involved predictor, a small network on the model’s final layer trained offline in under three minutes, refines this estimate from the hidden states. Figure 4 shows this loop on four requests: SGLang and MLFQ leave short requests C and D behind longer requests and prompt rounds, whereas DWISE finishes them first and lowers average completion time from 8.5 rounds to 6.0. Since DWISE only decides which blocks enter each round, it composes with decoding-level accelerations such as KV caching and parallel decoding (Wu et al., 2025); all evaluated models already run with both. Figure 5 shows how the loop splits across CPU and GPU.

![](images/f8e872d9a146063c670062e4b9ae3ea2628797611732dfd59d92ab148323f658.jpg)  
Figure 5. DWISE per-round scheduling loop.

## 4.1 Round-Based Workload Accounting

DWISE expresses request demand in the unit the runtime schedules. A remaining prefill block and a remaining denoising round each occupy one slot of one future round, so DWISE counts them equally. The remaining work of request i is

$$
\begin{array} { r } { R _ { i } = P _ { i } + \hat { r } _ { i } , } \end{array}\tag{1}
$$

where $P _ { i }$ is the exact number of unprocessed prefill blocks and $\hat { r } _ { i }$ is the estimated number of remaining denoising rounds. Size-based policies such as SRPT presuppose that all jobs’ remaining service is measured on one scale (Harchol-Balter, 2013). Eq. (1) supplies that scale, so DWISE can compare a request still in prefill with one already denoising. Which term dominates depends on the block size, so both are needed.

## 4.2 In-Path Workload Estimation

DWISE estimates $\hat { r } _ { i }$ twice, once from each signal in Section 3.2. At admission, the admission predictor on the CPU estimates it from prompt features. After the first denoising round, the diffusion-involved predictor refines it from the final-layer hidden states. It runs asynchronously on the GPU, and its output is collected at the existing end-of-round synchronization, so refinement adds neither model execution nor a synchronization point. One refinement suffices: the first round already exposes how the request will denoise, and refining more often only adds preemptions (Section 6.3).

Prompt-based Predictor. Our admission predictor is a LightGBM regressor (Ke et al., 2017) (100 trees, at most 15 leaves) trained over three prompt features: word count, newline count, and token count after the chat template. It predicts total denoising rounds in log space, and its inference takes 29 $\mu \mathrm { s }$ per request on the CPU. Among the regressors we compared, it ranks best at the lowest cost (Appendix B).

Diffusion-involved Predictor. The diffusion-involved predictor takes the final-layer hidden rows of the first denoising round, the model’s parallel draft of the whole block, meanpooled into one vector and concatenated with the normalized prompt length and output budget. It is a discrete-time survival model (Gensheimer & Narasimhan, 2019): a two-layer MLP (width 256, GELU, dropout 0.2) that outputs one hazard logit per future round up to K, the largest round count in training (K=279 on LLaDA2.1-mini), and the refined estimate is the mean of the resulting distribution over remaining rounds. An isotonic mapping (Barlow & Brunk, 1972), fitted on the training split, maps this output to actual remaining rounds, the scale the admission predictor targets, so requests holding admission and refined estimates are ranked consistently in Eq. (1). The predictor averages three such MLPs trained with different seeds. Together they total 1.79M parameters (0.01% of the model) and take about 0.8 ms of GPU time per round, under 1% of the 95 ms median round, so refinement is effectively free.

## 4.3 Preemptive Mixed-Phase Batching

Each serving round runs one batched forward pass over B block slots. DWISE selects the B resident requests with the smallest R and packs the next block of each into the batch, prompt or denoising. Prefill and denoising thus share every forward pass, and new prompts no longer stall generation. DWISE recomputes the ranking every round, so a running request is displaced as soon as it is outranked, either by a newly arrived shorter request or by a refinement that raises its own estimate. A displaced request is paused with its KV cache and partially denoised block kept on the GPU rather than swapped out (Sun et al., 2024), and resumes without recomputation. The policy is preemptive SRPT (Schrage, 1968) with the serving round as its quantum.

Preemption pays off only with mixed-phase batching: under prefill-first, each prefill round stalls all residents. It is safe because residency is controlled separately from execution. A waiting request becomes resident only when it ranks among the top B and the worst-case KV footprint of all residents still fits in the pool, with at most kB residents for a residentset multiplier k. Every resident request is thus guaranteed the memory to finish, so preemption never overcommits or deadlocks the pool.

## 5 SYSTEM IMPLEMENTATION

We implement DWISE in SGLang’s dLLM serving path as a runtime patch, without modifying SGLang’s source. The implementation splits along SGLang’s existing boundary between scheduling and execution and has two parts: a CPU-side scheduler that takes over admission and batch formation, and a GPU-side refinement path that evaluates the diffusion-involved predictor inside the forward pass. All other components, such as the model, denoising kernels, KV memory manager, and CUDA-graph path, stay unchanged.

Algorithm 1 DWISE per-round scheduling.   
Require: waiting set W, resident set S, block slots B,   
resident-set multiplier k   
1: for each serving round do   
2: sort $\mathcal { W } \cup \mathcal { S }$ by $R _ { i } = P _ { i } + \hat { r } _ { i }$   
3: for $i \in \mathcal { W }$ in sorted order do   
4: if i ranks in the top B and $| S | < k B$ and its KV   
reservation fits then   
5: move i from W to S   
6: end if   
7: end for   
8: $\mathcal { E }  B$ requests in S with the smallest $R _ { i }$   
9: pause $s \backslash \varepsilon$   
10: run one forward pass over the next blocks of E   
11: for $i \in \mathcal { E }$ do   
12: if i processed a prompt block then   
13: $P _ { i } \gets P _ { i } - 1$   
14: else if i finished its first denoising round then   
15: $\hat { r } _ { i } \gets$ refined estimate   
16: else   
17: $\hat { r } _ { i } \gets \operatorname* { m a x } ( \hat { r } _ { i } - 1 , 0 )$   
18: end if   
19: end for   
20: drop finished requests from S and free their KV   
21: end for

## 5.1 Scheduler Integration

The scheduler keeps three fields per request: the unprocessed prompt blocks $P _ { i }$ the remaining-rounds estimate ${ \hat { r } } _ { i } .$ and a residency state (waiting, running, or paused). Algorithm 1 shows the resulting per-round loop. DWISE intervenes at four points of each serving round. At admission, it applies the global remaining-work order together with the KV reservation policy of Section 4.3. At batch formation, it picks the next block of the highest-priority resident requests, which removes SGLang’s prefill-first phase barrier. At result processing, SGLang already walks the batch to commit outputs, and DWISE updates the three fields in that same pass. After the first denoising round, it records each request’s refined estimate once.

None of these steps touches memory management. The token pool, the radix prefix cache, and the pause/resume path are reused, so reordering execution needs neither a separate memory manager nor reconstruction of request state. Keeping the control loop on the CPU is deliberate. Its per-round work is many small, irregular operations, and an earlier GPU-resident planner was slower because kernellaunch cost dominated at that granularity.

Table 1. Serving and predictor-training requests.
<table><tr><td>Workload</td><td>Serving</td><td>Train</td></tr><tr><td>Open-ended</td><td>1,973</td><td>986</td></tr><tr><td>ShareGPT (ShareGPT Team, 2023)</td><td>1,085</td><td>542</td></tr><tr><td>MBPP (Austin et al., 2021b)</td><td>505</td><td>250</td></tr><tr><td>HumanEval (Chen et al., 2021)</td><td>87</td><td>46</td></tr><tr><td>IFEval (Zhou et al., 2023)</td><td>296</td><td>148</td></tr><tr><td>Single-answer</td><td>1,973</td><td>987</td></tr><tr><td>MATH (Level 5) (Hendrycks et al., 2021)</td><td>421</td><td>203</td></tr><tr><td>GSM8K (Cobbe et al., 2021)</td><td>345</td><td>180</td></tr><tr><td>MGSM (Shi et al., 2022)</td><td>258</td><td>129</td></tr><tr><td>MMLU-Pro (Wang et al., 2024)</td><td>637</td><td>319</td></tr><tr><td>ARC-C (Clark et al., 2018)</td><td>312</td><td>156</td></tr><tr><td>Long-context: Multi-needle (Kamradt, 2023)</td><td>150</td><td>75</td></tr><tr><td>Total</td><td>4,096</td><td>2,048</td></tr></table>

## 5.2 GPU Refinement Path

The diffusion-involved predictor is the one component that runs on the GPU, since its input is the hidden state of the first denoising round. The round’s final-layer hidden rows are pooled with tensor operations and passed through the predictor in one batched GEMM for every request refined in that round. The predictor’s scalar inputs, normalized prompt length and output budget, arrive through pinned host memory without blocking, and the predictor is enqueued on the forward pass’s stream. Because the hidden-state operations and refinement hooks are registered before CUDA-graph capture, SGLang’s graph-replay path stays enabled.

## 6 EVALUATION

## 6.1 Experiment Settings

Hardware. Each serving instance runs on one NVIDIA GH200 Grace Hopper Superchip, which pairs a Hopper GPU (132 SMs, 96 GB of HBM3) with a 72-core Arm Neoverse-V2 Grace CPU over NVLink-C2C.

Software and configuration. All policies run on SGLang v0.5.20’s dLLM path. Within each model, policies use the same block slots per round (B = 128; B = 16 for DiffusionGemma) and the same KV pool (50.8, 65.9, and 13.1 GiB for LLaDA2.x, SDAR-8B, and DiffusionGemma). DWISE uses k = 8 (at most kB resident requests), two blocks of reservation slack per resident request, and a 90% KV reservation limit. We report the mean of three runs.

Workloads. The workload mixes open-ended, singleanswer, and long-context requests (Table 1): 4,096 requests for evaluation and a prompt-disjoint set of 2,048 with the same composition, used only to train DWISE and the learned baselines. Long-context requests are synthetic multi-needle retrieval prompts (Kamradt, 2023; Hsieh et al., 2024) of about 10K tokens, each embedding 2–4 target facts among distractor sentences and querying all of them. They are prefill-heavy and automatically checkable.

Table 2. End-to-end serving performance and task accuracy across models and arrival patterns. B: burst; P: burst+Poisson.
<table><tr><td></td><td colspan="2">LLaDA2.0-mini</td><td colspan="2">LLaDA2.1-mini</td><td colspan="2">LLaDA2.2-mini</td><td colspan="2">SDAR-8B</td><td colspan="2">DiffusionGemma</td></tr><tr><td></td><td>B</td><td>P</td><td>B</td><td>P</td><td>B</td><td>P</td><td>B</td><td>P</td><td>B</td><td>P</td></tr><tr><td colspan="9">Mean latency (s) ↓</td><td></td></tr><tr><td>SGLang</td><td>603.1</td><td>597.4</td><td>545.5</td><td>535.9</td><td>576.6</td><td>571.7</td><td>9053</td><td>8973</td><td>325.9</td><td>262.7</td></tr><tr><td>MLFQ</td><td>716.0</td><td>721.8</td><td>625.9</td><td>616.9</td><td>651.0</td><td>654.4</td><td>11382</td><td>12141</td><td>346.2</td><td>282.1</td></tr><tr><td>SSJF</td><td>739.3</td><td>728.4</td><td>602.7</td><td>584.6</td><td>647.4</td><td>611.7</td><td>11912</td><td>10936</td><td>323.5</td><td>264.2</td></tr><tr><td>LTR</td><td>702.7</td><td>679.6</td><td>645.3</td><td>613.9</td><td>664.2</td><td>597.3</td><td>12217</td><td>11577</td><td>326.9</td><td>266.1</td></tr><tr><td>S3</td><td>469.2</td><td>470.3</td><td>412.1</td><td>408.9</td><td>417.3</td><td>410.4</td><td>13557</td><td>12026</td><td>321.1</td><td>257.4</td></tr><tr><td>TRAIL</td><td>763.3</td><td>750.3</td><td>661.1</td><td>661.3</td><td>679.5</td><td>683.4</td><td>15004</td><td>14304</td><td>328.4</td><td>264.6</td></tr><tr><td>DWISE</td><td>128.5</td><td>96.9</td><td>76.9</td><td>48.5</td><td>109.4</td><td>79.8</td><td>609</td><td>668</td><td>292.4</td><td>234.8</td></tr><tr><td colspan="9">Throughput (req/s) ↑</td><td></td><td></td></tr><tr><td>SGLang</td><td>3.48</td><td>3.34</td><td>3.90</td><td>3.76</td><td>3.65</td><td>3.49</td><td>0.24</td><td>0.23</td><td>6.22</td><td>6.14</td></tr><tr><td>MLFQ</td><td>3.59</td><td>3.43</td><td>4.13</td><td>4.00</td><td>3.87</td><td>3.70</td><td>0.22</td><td>0.20</td><td>5.88</td><td>5.85</td></tr><tr><td>SSJF</td><td>3.31</td><td>3.26</td><td>3.96</td><td>3.91</td><td>3.73</td><td>3.77</td><td>0.22</td><td>0.22</td><td>5.81</td><td>5.82</td></tr><tr><td>LTR</td><td>3.45</td><td>3.44</td><td>3.95</td><td>3.97</td><td>3.83</td><td>4.01</td><td>0.22</td><td>0.22</td><td>5.79</td><td>5.83</td></tr><tr><td>S3</td><td>3.41</td><td>3.23</td><td>3.86</td><td>3.67</td><td>3.66</td><td>3.50</td><td>0.21</td><td>0.22</td><td>5.83</td><td>5.92</td></tr><tr><td>TRAIL</td><td>3.21</td><td>3.14</td><td>3.74</td><td>3.59</td><td>3.47</td><td>3.33</td><td>0.19</td><td>0.19</td><td>6.11</td><td>6.04</td></tr><tr><td>DWISE</td><td>6.39</td><td>6.47</td><td>8.02</td><td>8.21</td><td>6.57</td><td>6.65</td><td>0.25</td><td>0.25</td><td>6.07</td><td>6.03</td></tr><tr><td colspan="9">Task accuracy (%) ↑</td><td></td><td></td></tr><tr><td>SGLang</td><td>45.4</td><td>45.4</td><td>44.2</td><td>44.1</td><td>46.5</td><td>46.5</td><td>55.5</td><td>55.2</td><td>51.3</td><td>51.3</td></tr><tr><td>DWISE</td><td>46.2</td><td>46.2</td><td>44.3</td><td>44.4</td><td>46.9</td><td>46.9</td><td>55.4</td><td>55.4</td><td>51.2</td><td>51.8</td></tr></table>

Models. We evaluate five block-diffusion LLMs: three LLaDA2.x-mini models (2.0, 2.1, and 2.2) (Bie et al., 2025; 2026b;a) (16B MoE, 32-token blocks), SDAR-8B-Chat (Cheng et al., 2025) (dense, 4-token blocks), and DiffusionGemma-26B-A4B (DiffusionGemma Team et al., 2026) (MoE, 256-token blocks, prompt encoded in a separate pass). DWISE runs on all five with the same policy. All models use their official decoding settings, a 256-token output budget, greedy decoding, and bfloat16 precision.

Arrival patterns. In burst, all 4,096 requests arrive at t = 0, modeling a backlogged server. In burst+Poisson, half arrive at t = 0 and the rest follow a Poisson process (Zhang et al., 2025) with a rate above SGLang’s service rate, so new requests keep arriving while the backlog is served. The rates are λ = 16, 0.85, and 8 req/s for LLaDA, SDAR, and DiffusionGemma, giving Poisson phases of about 128 s, 2,410 s, and 256 s. Request order and arrival times are generated with fixed seeds, so all policies of a model see the identical arrival trace. Appendix C shows that DWISE’s speedup is stable across seeds.

Baselines. We compare against SGLang’s FCFS scheduler with prefill priority (Zheng et al., 2024), MLFQ following FastServe (Wu et al., 2024), and four learned schedulers retrained on DWISE’s 2,048 training requests with labels from FCFS profiling. SSJF (Qiu et al., 2024), S<sup>3</sup> (Jin et al., 2023), and LTR (Fu et al., 2024) predict output length (LTR: its rank) from the prompt at admission, and TRAIL (Shahout et al., 2024) refines its remaining-output-length estimate from runtime embeddings every round. All four thus rank requests by output tokens, which do not order dLLM requests: code and math emit more tokens than chat yet need fewer denoising rounds (Appendix D). All baselines run on SGLang’s prefill-first batch former, as originally designed.

Metrics. Following common LLM serving practice (Kwon et al., 2023), we report mean, p50, and p99 end-to-end request latency and throughput. Because DWISE ranks requests by predicted workload, we evaluate workload prediction by Spearman rank correlation (ρ) (Spearman, 1904) between predicted and true workload. Rank-based LLM scheduling likewise depends on relative request order rather than exact cost (Fu et al., 2024). We also report predictor and scheduler overhead, the predicted-to-actual ratio of remaining work, and task accuracy on single-answer and long-context requests.

## 6.2 End-to-End Performance

Mean latency and throughput. On the LLaDA2.x models (Table 2), DWISE reduces mean latency over SGLang by 4.7–7.1× under burst arrivals and 6.2–11.1× under burst+Poisson arrivals, and raises throughput from 3.34– 3.90 to 6.39–8.21 req/s. On SDAR-8B, running long-context requests last lets DWISE reduce mean latency by 13.4– 14.9×; the geometric mean over the ten settings is 5.4×. On DiffusionGemma, where the two phases cannot share a round, ordering alone gains 1.1×, against at most 1.02× for any baseline. Prior LLM schedulers do not transfer to dLLMs: MLFQ, SSJF, LTR, and TRAIL are slower than SGLang on LLaDA2.x and SDAR-8B (0.60–0.96×), and $S ^ { 3 }$ gains at most 1.4× (Appendix D and Section 6.3). On multiple GPUs (TP=2/4, TP=4+EP=4), DWISE’s LLaDA2.1- mini speedup stays at 6.7–12.2×, and with a 2,048-token output budget at 4.4–5.2× (Appendix E).

![](images/9b953633fd9be695a8a30ad1a60423ea600e2b27341cba0efcf6e1efaefef79e.jpg)  
Figure 6. Normalized p50 latency relative to SGLang; lower is better, and annotations show speedup. DGemma: DiffusionGemma.

Median latency. Figure 6 reports p50 latency. On the three LLaDA2.x models, DWISE improves p50 over SGLang by 6.2–11× under burst arrivals and by 14–42× under burst+Poisson arrivals. On SDAR-8B, the improvements are 43× and 73×. These gains exceed those in mean latency because DWISE completes requests with little remaining work early, whereas the mean is dominated by a few long requests. On DiffusionGemma, p50 improves by 1.2× and 1.6×. SGLang’s prefill-first policy is meant to protect time to first token (TTFT), the time to a request’s first output block, yet it fails even there: DWISE lowers mean TTFT on LLaDA2.1-mini by 7.8–13.0× (Appendix F).

Tail latency. DWISE also shortens the tail (Figure 7), cutting p99 latency by 1.9–2.3× on the LLaDA2.x models and by 1.3× on SDAR-8B, whose throughput barely changes. SRPT serves the longest requests last, yet they still finish sooner, because mixed-phase batching keeps new prompts from stalling them and nearly doubles LLaDA2.x throughput (Table 2), so the backlog drains sooner. Appendix G confirms this on LLaDA2.1-mini: p99 and maximum latency drop in every category. On DiffusionGemma, where prompts are encoded in a separate pass and mixed-phase batching does not apply, DWISE can only reorder requests, so the longest ones wait longer: $\mathsf { p } 9 9$ is unchanged under burst arrivals and rises under burst+Poisson arrivals.

![](images/5a09b9f3d32a666fdc72d03783aed8832f8424dad81a55601b0f2ca0b9dd527d.jpg)

Figure 7. Normalized p99 latency relative to SGLang; lower is better, and annotations show speedup.  
![](images/dd630f1ed97ac8391269423829b52d86648abf0e46179a0cb62e8fbe877133bd.jpg)  
Figure 8. Cumulative speedup over SGLang as DWISE’s components are added (LLaDA2.1-mini, burst+Poisson). Rank by P<sub>i</sub>: remaining prefill blocks only $( \hat { r } _ { i } { = } 0 )$ . Oracle: ranking by true denoising rounds.

## 6.3 Ablation and Sensitivity Analysis

Component breakdown. Figure 8 enables the components of DWISE one at a time on LLaDA2.1-mini under burst+Poisson arrivals. Mixed-phase batching alone reduces mean latency by 2.35× over SGLang. Ranking by remaining prefill blocks alone $( \hat { r } _ { i } { = } 0 )$ raises the speedup to 8.15×, as long-context prompts no longer run ahead of short requests. Mixed-phase batching thus creates scheduling flexibility across phases, and remaining-work ranking decides how to use it. Adding the admission-time estimate of denoising rounds raises the speedup to 10.33×, and first-round refinement to 11.06× by correcting inaccurate admission estimates. An oracle ranking by true denoising rounds reaches $1 3 . 1 0 \times ;$ the predictors close 70% of the gap to $\mathbf { i t } ,$ and DWISE reaches 84% of its speedup.

Round composition. Mixed-phase batching pays off because prefill-first rounds leave most of the batch empty (Table 3). On LLaDA2.1-mini, 95% of SGLang’s rounds contain only prefill blocks. Such a round usually carries a single prompt block (1.1 of 128 slots on average), so 99% of its slots sit idle, yet these rounds take 78% of wallclock time. DWISE fills the idle slots with denoising blocks. It uses 113 slots per round, leaves 12% of slots idle, and spends only 11% of its time in prefill-only rounds. It therefore finishes the workload in 2.8K rounds instead of 64K, which is why its throughput roughly doubles in Table 2.

Table 3. Serving-round composition on LLaDA2.1-mini under burst+Poisson arrivals (B = 128). Prefill-only: rounds with only prefill blocks, as a share of all rounds and of wall-clock time.
<table><tr><td colspan="3"></td><td colspan="2">Prefill-only (%) Idle</td><td rowspan="2">Filled</td></tr><tr><td>Policy</td><td>Rounds</td><td>s rounds</td><td>time</td><td>slots (%) slots/round</td></tr><tr><td>SGLang</td><td>63,926</td><td>94.6</td><td>78.2</td><td>96.1</td><td>5.0</td></tr><tr><td>DWISE</td><td>2,819</td><td>3.0</td><td>11.2</td><td>11.6</td><td>113.2</td></tr></table>

Batch former versus ranking signal. To separate the two sources of gain, Table 4 runs FCFS and the learned baselines on DWISE’s mixed-phase batch former, where every baseline becomes 2.1–4.1× faster than in Table 2. With the batch former held fixed, DWISE remains 2.5× faster than the best baseline under burst arrivals $( \mathbf { S } ^ { 3 } )$ and 3.3× faster under burst+Poisson arrivals (TRAIL), a gap due to the ranking alone. The batch former matters most for preemptive policies. On the prefill-first batch former, the two preemptive baselines, MLFQ and TRAIL, are slower than SGLang $( 0 . 8 1 { - } 0 . 8 7 { \times } )$ . Their preemption keeps up to $\mathbf { k B } = 1 , 0 2 4$ requests resident instead of 128, yet a prefill-only round still carries only 1.6–1.7 rows, and every such round stalls all of them, so a resident request spends 5.4–5.8× more rounds running no block than under SGLang. Mixed-phase batching removes this coupling.

Table 4. Mean latency (s) on LLaDA2.1-mini with all ranking policies on DWISE’s mixed-phase batch former. FCFS is SGLang’s default order. B: burst; P: burst+Poisson.
<table><tr><td></td><td>FCFS</td><td>SSJF</td><td> ${ \mathrm { S } } ^ { 3 }$ </td><td>LTR</td><td>TRAIL</td><td>DWISE</td></tr><tr><td>B</td><td>251.7</td><td>282.6</td><td>188.4</td><td>271.1</td><td>198.5</td><td>76.9</td></tr><tr><td>P</td><td>228.0</td><td>256.9</td><td>165.1</td><td>248.3</td><td>160.7</td><td>48.5</td></tr></table>

Load sweep. Figure $9$ sweeps the arrival rate on LLaDA2.1- mini under Poisson arrivals; the five other baselines track SGLang at every rate. Below SGLang’s throughput $( \lambda \leq$ 3.8 req/s), DWISE reduces mean latency by 1.9–4.5× and p99 latency by 1.6–4.2×. Few requests wait at this load, so the gain comes from mixed-phase batching: new prompts no longer stall denoising. Once SGLang is overloaded $( \lambda = 7 . 5$ and 15 req/s), DWISE reduces mean latency by 25–31× and p99 latency by 2.5–11×, because requests queue long enough for remaining-work ranking to matter.

![](images/e362663cbbd6efc9ef3e682e6a8dd7f8ee3c8b8f9df3eb4efb22d91fc40d65ba.jpg)

![](images/4385a2cc09137baa543b93704c99e2646f3913ddd91a0e9b2653c0d0af7f8ab8.jpg)  
Figure 9. Latency on LLaDA2.1-mini under pure Poisson arrivals without an initial backlog. Dotted line: SGLang’s throughput; grey band: the five other baselines.

Refinement calibration. Refined and admission-time estimates must share one scale, since DWISE ranks them together (Table 5). The diffusion-involved predictor reports the mean of a right-skewed distribution while the admission predictor regresses in log space, so its raw output ranks requests better $( \rho { = } 0 . 8 9 0 { - } 0 . 8 9 1 )$ ) but overestimates remaining rounds: its median predicted-to-actual ratio is 1.16, against 1.03 at admission, so refined requests look longer than comparable unrefined ones. The isotonic mapping brings the ratio to 1.00 without changing $\rho ,$ and lowers mean latency from 80.2 to $7 6 . 9 \mathrm { s }$ under burst arrivals and from 50.9 to 48.5 s under burst+Poisson.

Table 5. Effect of refinement calibration on LLaDA2.1-mini, measured on the evaluation set. Ratio denotes the median predicted-toactual remaining rounds. B: burst; P: burst+Poisson.
<table><tr><td></td><td>Arrival Predictor</td><td>Ratio</td><td> $\rho$ </td><td>Mean (s)</td><td>p50 (s)</td></tr><tr><td rowspan="3">B</td><td>Admission only</td><td>1.03</td><td>0.714</td><td>80.8</td><td>58.4</td></tr><tr><td>Raw refinement</td><td>1.16</td><td>0.890</td><td>80.2</td><td>55.2</td></tr><tr><td>Calibrated</td><td>1.00</td><td>0.890</td><td>76.9</td><td>51.9</td></tr><tr><td rowspan="3">P</td><td>Admission only</td><td>1.03</td><td>0.716</td><td>51.9</td><td>17.2</td></tr><tr><td>Raw refinement</td><td>1.16</td><td>0.891</td><td>50.9</td><td>15.4</td></tr><tr><td>Calibrated</td><td>1.00</td><td>0.891</td><td>48.5</td><td>13.7</td></tr></table>

Refinement frequency. Table 6 varies how often the diffusion-involved predictor re-estimates remaining work on LLaDA2.1-mini under burst+Poisson arrivals. Refining once, after the first denoising round, gives the lowest mean and p50 latency. Refining every block, every four rounds, or every round raises mean latency to 49.5–51.0 s. More frequent re-estimates keep reordering running requests, and preemptions per request rise from 0.34 to 1.14. Between refinements, $\boldsymbol { { \hat { r } } } _ { i }$ decreases by one per completed round and is clamped at zero, so an underestimated request moves to the front of the queue and runs to completion. With one refinement, 37% of requests reach zero at least once, against at most 5.5% otherwise. The predictor costs about 0.8 ms of GPU time per round in every setting.

Table 6. Effect of refinement frequency on LLaDA2.1-mini under burst+Poisson arrivals. Predictor: GPU time per serving round.
<table><tr><td>Refine</td><td></td><td></td><td>Once Per block Per 4 rounds Per round</td><td></td></tr><tr><td>Mean (s)</td><td>48.5</td><td>49.5</td><td>51.0</td><td>51.0</td></tr><tr><td>p50 (s)</td><td>13.7</td><td>13.9</td><td>15.0</td><td>14.7</td></tr><tr><td>Preemptions / request</td><td>0.34</td><td>0.75</td><td>0.90</td><td>1.14</td></tr><tr><td>Reach  $\bar { r } _ { i } = 0 ( \% )$ </td><td>37.0</td><td>5.5</td><td>4.9</td><td>0.0</td></tr><tr><td>Predictor  $( \mu \mathrm { s } )$ </td><td>795</td><td>809</td><td>828</td><td>777</td></tr></table>

Table 7. Generalization to unseen request categories on LLaDA2.1- mini. Within-category $\rho$ is averaged over seven categories, each held out in turn (long context excluded; see Appendix H).
<table><tr><td></td><td>Admission  $\rho$ </td><td>Refined ρ</td></tr><tr><td>Within category, seen in training Within category, held out</td><td>0.16 -0.06</td><td>0.64 0.54</td></tr><tr><td>All requests, MCQ seen</td><td>0.72</td><td>0.89</td></tr><tr><td>All requests, MCQ held out</td><td>0.36</td><td>0.89</td></tr></table>

Generalization to unseen categories. Table 7 holds out one request category at a time and retrains both predictors on the rest. Prompt size features, the only input of DWISE’s admission predictor, order requests within a category only weakly, even when the category is seen in training $( \rho = 0 . 1 6$ on average) and not at all once the category is held out $( \rho = - 0 . 0 6 )$ . With MCQ held out, the admission $\rho$ over all requests drops from 0.72 to 0.36. The refined estimate stays useful on held-out categories $( \rho = 0 . 5 4 )$ and keeps $\rho = 0 . 8 9$ over all requests, because the first-round hidden state reflects how the model denoises a request rather than what its prompt looks like. End-to-end, retraining without MCQ raises DWISE’s mean latency from 47.3 to 50.9 s in one run, a 7.6% slowdown that still leaves DWISE 10.5× faster than SGLang (536.8 s in the same runs).

Resident-set sensitivity. Figure 10 varies the resident-set multiplier k, which caps KV-resident requests at kB. The cap trades ranking flexibility against residency: a larger resident set gives DWISE more requests to choose from in each round, but every resident request holds a KV reservation. A small cap keeps high-priority waiting requests out even when KV capacity is available. $\quad \mathrm { A t ~ } k = 2 ,$ the cap blocks admission in over 60% of serving rounds and raises mean latency by about 10% under burst arrivals and 20% under burst+Poisson arrivals. Raising k to 4 removes most of this blocking, and at $k = 8$ the cap no longer blocks admission in either workload. Beyond this point, admission is governed by the KV budget rather than by the resident count, so $k = 1 6$ brings no consistent improvement. We use $k = 8$ as the default.

Overhead. Figure 11 breaks down DWISE’s runtime cost on LLaDA2.1-mini under burst+Poisson arrivals. Figure 11(a) shows the cost per serving round. Ranking and admission take 0.96 ms, reordering resident requests 0.20 ms, and the diffusion-involved predictor 0.80 ms, plus 0.04 ms to read its output. Together they amount to 2.1% of the 95 ms median round. The predictor runs on the forward pass’s stream and is read at the existing end-of-round synchronization, so it adds no synchronization point. Figure 11(b) shows the cost of admission-time prediction per request. DWISE’s LightGBM predictor takes 0.42 ms from raw prompt text, 9–18× less than the text encoders that $S ^ { 3 }$ , SSJF, and LTR run on the prompt.

![](images/240756ad150633420cbef4c32c8381cb5b95f6cac11e45b35f49b2dcc2782e70.jpg)

![](images/87a5e1cd9d05a30d1c50154474f7f6f72e941e66a92eeb19c838ef2a6297c2d7.jpg)

Figure 10. Effect of resident-set multiplier k on LLaDA2.1-mini.  
![](images/7eb8f1f859bc495c5d4494c052310b60566215d71e58b4c5bb053ac23d3a3fa0.jpg)  
Figure 11. DWISE overhead on LLaDA2.1-mini. (a) Cost per serving round, as a share of the 95 ms median round. (b) Admission prediction cost per request from raw prompt text (log scale).

## 7 CONCLUSION

We presented DWISE, a scheduler for block-diffusion LLM serving that replaces FCFS admission and prefill priority. Its design rests on two properties that set dLLMs apart from AR models: prefill and denoising advance at the same block granularity, so neither phase needs priority over the other, and a request’s remaining work becomes far more predictable after a single denoising round, whose hidden states capture model-specific information absent from the prompt. Built on these properties, DWISE lowers mean latency over SGLang’s default scheduler by 5.4× on average and up to 14.9× and raises throughput by up to 2.2× across five models from three dLLM families without affecting task accuracy. Our results suggest that dLLM serving should be scheduled around how these models generate, block by block, rather than around policies inherited from AR serving.

## REFERENCES

Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve, 2024. URL https://arxiv.org/ abs/2403.02310.

Arriola, M., Gokaslan, A., Chiu, J. T., Yang, Z., Qi, Z., Han, J., Sahoo, S. S., and Kuleshov, V. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025. URL https://arxiv.org/ abs/2503.09573.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. Structured denoising diffusion models in discrete state-spaces. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 17981–17993. Curran Associates, Inc., 2021a. URL https://proceedings.neurips. cc/paper\_files/paper/2021/file/ 958c530554f78bcd8e97125b70e6973d-Paper pdf.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. Program synthesis with large language models, 2021b. URL https://arxiv.org/abs/ 2108.07732.

Barlow, R. E. and Brunk, H. D. The isotonic regression problem and its dual. Journal of the American Statistical Association, 67(337):140–147, 1972. ISSN 01621459, 1537274X. URL http://www.jstor. org/stable/2284712.

Bie, T., Cao, M., Chen, K., Du, L., Gong, M., Gong, Z., Gu, Y., Hu, J., Huang, Z., Lan, Z., Li, C., Li, C., Li, J., Li, Z., Liu, H., Liu, L., Lu, G., Lu, X., Ma, Y., Tan, J., Wei, L., Wen, J.-R., Xing, Y., Zhang, X., Zhao, J., Zheng, D., Zhou, J., Zhou, J., Zhou, Z., Zhu, L., and Zhuang, Y. LLaDA2.0: Scaling up diffusion language models to 100B, 2025. URL https://arxiv.org/ abs/2512.15745.

Bie, T., Cai, J., Cao, M., Cao, X., Chen, B., Chen, F., Chen, K., Ding, Y., Du, L., Gong, Z., Gu, Y., Guan, K., Guo, X., Han, K., Huang, Z., Huang, Z., Lan, Z., Li, C., Li, J., Li, Z., Liang, R., Lin, X., Liu, H., Liu, L., Liu, X., Liu, X., Lu, G., Lu, Y., Ma, Y., Mao, C., Qi, X., Qiu, K., Shi, X., Tian, Y., Wang, H., Wang, Q., Wang, R., Wang, S., Wei, L., Wu, T., Xiao, S., Xing, Y., Xu, C., Yan, S., Yan, Y., Yang, J., Yang, Y., Zha, L., Zhang, T., Zhang, Y., Zhao, J., Zhao, W., Zheng, D., Zhong, J., Zhou, J., Zhou, J., Zhou, T., Zhu, L., and Zhuang, Y. LLaDA2.2: Enabling agentic diffusion language models via Levenshtein editing.

Technical report, Inclusion AI, 2026a. URL https:// github.com/inclusionAI/LLaDA2.X/blob/ main/LLaDA2\_2\_tech\_report.pdf. LLaDA2.2- mini model card: https://huggingface.co/ inclusionAI/LLaDA2.2-mini.

Bie, T., Cao, M., Cao, X., Chen, B., Chen, F., Chen, K., Du, L., Feng, D., Feng, H., Gong, M., Gong, Z., Gu, Y., Guan, J., Guan, K., He, H., Huang, Z., Jiang, J., Jiang, Z., Lan, Z., Li, C., Li, J., Li, Z., Liu, H., Liu, L., Lu, G., Lu, Y., Ma, Y., Mou, X., Pan, Z., Qiu, K., Ren, Y., Tan, J., Tian, Y., Wang, Z., Wei, L., Wu, T., Xing, Y., Ye, W., Zha, L., Zhang, T., Zhang, X., Zhao, J., Zheng, D., Zhong, H., Zhong, W., Zhou, J., Zhou, J., Zhu, L., Zhu, M., and Zhuang, Y. LLaDA2.1: Speeding up text diffusion via token editing, 2026b. URL https: //arxiv.org/abs/2602.08676.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper pdf.

Chang, T.-T., Hong, B. Y., Pham, K., and Venkataraman, S. TeDiServe: High SLO attainment serving for diffusion language models, 2026. URL https://arxiv.org/ abs/2606.29094.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021.

Cheng, S., Bian, Y., Liu, D., Zhang, L., Yao, Q., Tian,

Z., Wang, W., Guo, Q., Chen, K., Qi, B., and Zhou, B. SDAR: A synergistic diffusion-autoregression paradigm for scalable sequence generation, 2025. URL https: //arxiv.org/abs/2510.06303.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/abs/ 1803.05457.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.

Corbato, F. J., Merwin-Daggett, M., and Daley, R. C.´ An experimental time-sharing system. In Proceedings of the May 1-3, 1962, Spring Joint Computer Conference, AIEE-IRE ’62 (Spring), pp. 335–344, New York, NY, USA, 1962. Association for Computing Machinery. ISBN 9781450378758. doi: 10.1145/ 1460833.1460871. URL https://doi.org/10. 1145/1460833.1460871.

DiffusionGemma Team, Ta¨ıga, A. A., Assiene, J., Calandriello, D., Chaabouni, R., Gante, J., von Glehn, T., Keating, N., Knutsen, C., Kukla, M., Liu, T., Lobov, I., Nabati, O., Oliveira, J. G., Perez-Nieves, N., Prutianova, N., Shahriari, B., Tarbouriech, J., Tyletski, P., C¸ aglar˘ Unl<sup>¨</sup> u, Wu, C., Cameron, G., Connor, J., Gir-¨ gin, S., Grootendorst, M., Levkovitch, A., Nachmani, E., Sanseviero, O., Stanczyk, P., Berthet, Q., Campbell, A., Crepy, C., Bortoli, V. D., Doucet, A., Elie, R., Galashov, A., Greff, K., Jacq, A., Ruhe, D., Wu, Y.- H., Flennerhag, S., O’Donoghue, B., Scrivener, G., and Thakoor, S. DiffusionGemma technical report, 2026. URL https://arxiv.org/abs/2608.00146.

Fan, J., Zhang, Y., Li, X., and Nikolopoulos, D. S. Taming the memory footprint crisis: System design for production diffusion LLM serving, 2026. URL https: //arxiv.org/abs/2512.17077.

Fu, Y., Zhu, S., Su, R., Qiao, A., Stoica, I., and Zhang, H. Efficient LLM scheduling by learning to rank, 2024. URL https://arxiv.org/abs/2408.15792.

Gao, B., He, Z., Sharma, P., Kang, Q., Jevdjic, D., Deng, J., Yang, X., Yu, Z., and Zuo, P. Cost-efficient large language model serving for multi-turn conversations with cachedattention, 2024. URL https://arxiv.org/ abs/2403.19708.

Gensheimer, M. F. and Narasimhan, B. A scalable discretetime survival model for neural networks. PeerJ, 7:

e6257, January 2019. ISSN 2167-8359. doi: 10.7717/ peerj.6257. URL http://dx.doi.org/10.7717/ peerj.6257.

Harchol-Balter, M. Performance Modeling and Design of Computer Systems: Queueing Theory in Action. Cambridge University Press, Cambridge, 2013. ISBN 978-1- 107-02750-3.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models, 2020. URL https://arxiv.org/ abs/2006.11239.

Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. RULER: What’s the real context size of your long-context language models?, 2024. URL https://arxiv.org/abs/2404. 06654.

Inception Labs, Khanna, S., Kharbanda, S., Li, S., Varma, H., Wang, E., Birnbaum, S., Luo, Z., Miraoui, Y., Palrecha, A., Ermon, S., Grover, A., and Kuleshov, V. Mercury: Ultra-fast language models based on diffusion, 2025. URL https://arxiv.org/abs/2506. 17298.

Jin, Y., Wu, C.-F., Brooks, D., and Wei, G.-Y. S<sup>3</sup>: Increasing GPU utilization during generative inference for higher throughput, 2023. URL https://arxiv.org/abs/ 2306.06000.

Kamradt, G. Needle In A Haystack – pressure testing LLMs, 2023. URL https://github.com/gkamradt/ LLMTest\_NeedleInAHaystack. GitHub repository.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. LightGBM: A highly efficient gradient boosting decision tree. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper\_files/paper/2017/file/ 6449f44a102fde848669bdd9eb6b76fa-Paper pdf.

Kedia, N., Agarwal, S., Lee, M., and Akella, A. Sangam: Efficiently serving diffusion LLMs with the AR stack, 2026. URL https://arxiv.org/abs/2607.04206.

Kleinrock, L. Queueing Systems, Volume I. Wiley, 1975. ISBN 9780471491101. URL https://books. google.co.uk/books?id=sJSc0QEACAAJ.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Liu, Z., Yang, Y., Zhang, Y., Chen, J., Zou, C., Wei, Q., Wang, S., Zhu, Y., and Zhang, L. dLLM-Cache: Accelerating diffusion large language models with adaptive caching, 2026. URL https://arxiv.org/abs/ 2506.06295.

Mitzenmacher, M. Scheduling with predictions and the price of misprediction, 2019. URL https://arxiv. org/abs/1902.00732.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models, 2025. URL https://arxiv.org/ abs/2502.09992.

Patel, P., Choukse, E., Zhang, C., Shah, A., <sup>´</sup>Inigo Goiri,˜ Maleki, S., and Bianchini, R. Splitwise: Efficient generative LLM inference using phase splitting, 2024. URL https://arxiv.org/abs/2311.18677.

Qiu, H., Mao, W., Patke, A., Cui, S., Jha, S., Wang, C., Franke, H., Kalbarczyk, Z. T., Bas¸ar, T., and Iyer, R. K. Efficient interactive LLM serving with proxy model-based sequence length prediction, 2024. URL https://arxiv.org/abs/2404.08509.

Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., and Kuleshov, V. Simple and effective masked diffusion language models, 2024. URL https://arxiv.org/abs/2406.07524.

Schrage, L. A proof of the optimality of the shortest remaining processing time discipline. Operations Research, 16 (3):687–690, 1968. ISSN 0030364X, 15265463. URL http://www.jstor.org/stable/168596.

Scully, Z., Harchol-Balter, M., and Scheller-Wolf, A. SOAP: One clean analysis of all age-based scheduling policies, 2018. URL https://arxiv.org/abs/1712. 00790.

Shahout, R., Malach, E., Liu, C., Jiang, W., Yu, M., and Mitzenmacher, M. Don’t stop me now: Embedding based scheduling for LLMs, 2024. URL https://arxiv. org/abs/2410.01035.

ShareGPT Team. ShareGPT: Share your wildest ChatGPT conversations with one click. https:// sharegpt.com/; data: https://huggingface. co/datasets/anon8231489123/ShareGPT\_ Vicuna\_unfiltered, 2023.

Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H. W., Tay, Y., Ruder, S., Zhou, D., Das, D., and Wei, J. Language models are multilingual chain-of-thought reasoners, 2022. URL https: //arxiv.org/abs/2210.03057.

Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K. Simplified and generalized masked diffusion for discrete data, 2025. URL https://arxiv.org/abs/2406. 04329.

Smith, W. E. Various optimizers for single-stage production. Naval Research Logistics Quarterly, 3(1-2):59–66, 1956. doi: https://doi.org/10.1002/nav.3800030106. URL https://onlinelibrary.wiley.com/doi/ abs/10.1002/nav.3800030106.

Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics, 2015. URL https: //arxiv.org/abs/1503.03585.

Spearman, C. The proof and measurement of association between two things. The American Journal ofPsychology, 15(1):72–101, 1904. ISSN 00029556. URL http:// www.jstor.org/stable/1412159.

Sun, B., Huang, Z., Zhao, H., Xiao, W., Zhang, X., Li, Y., and Lin, W. Llumnix: Dynamic scheduling for large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 173–191, Santa Clara, CA, July 2024. USENIX Association. ISBN 978-1-939133-40-3. URL https://www.usenix.org/conference/ osdi24/presentation/sun-biao.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W. MMLU-Pro: A more robust and challenging multitask language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574.

Wu, B., Zhong, Y., Zhang, Z., Liu, S., Liu, F., Sun, Y., Huang, G., Liu, X., and Jin, X. Fast distributed inference serving for large language models, 2024. URL https: //arxiv.org/abs/2305.05920.

Wu, C., Zhang, H., Xue, S., Liu, Z., Diao, S., Zhu, L., Luo, P., Han, S., and Xie, E. Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding, 2025. URL https://arxiv. org/abs/2505.22618.

Wu, C.-J., Raghavendra, R., Gupta, U., Acun, B., Ardalani, N., Maeng, K., Chang, G., Behram, F. A., Huang, J., Bai, C., Gschwind, M., Gupta, A., Ott, M., Melnikov, A., Candido, S., Brooks, D., Chauhan, G., Lee, B., Lee, H.-H. S.,

Akyildiz, B., Balandat, M., Spisak, J., Jain, R., Rabbat, M., and Hazelwood, K. Sustainable ai: Environmental implications, challenges and opportunities, 2022. URL https://arxiv.org/abs/2111.00364.

Ye, J., Xie, Z., Zheng, L., Gao, J., Wu, Z., Jiang, X., Li, Z., and Kong, L. Dream 7B: Diffusion large language models, 2025. URL https://arxiv.org/abs/2508. 15487.

Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.- G. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pp. 521–538, Carlsbad, CA, July 2022. USENIX Association. ISBN 978-1-939133-28-1. URL https://www.usenix.org/conference/ osdi22/presentation/yu.

Zhang, W., Wu, Z., Mu, Y., Ning, R., Liu, B., Sarda, N., Lee, M., and Lai, F. Jitserve: Slo-aware LLM serving with imprecise request information, 2025. URL https: //arxiv.org/abs/2504.20068.

Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. SGLang: Efficient execution of structured language model programs, 2024. URL https://arxiv.org/abs/2312.07104.

Zheng, Z., Ren, X., Xue, F., Luo, Y., Jiang, X., and You, Y. Response length perception and sequence scheduling: An LLM-empowered LLM inference pipeline, 2023. URL https://arxiv.org/abs/2305.13144.

Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving, 2024. URL https://arxiv.org/abs/2401. 09670.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. Instruction-following evaluation for large language models, 2023. URL https: //arxiv.org/abs/2311.07911.

## A WORKLOAD SPREAD AND PREDICTOR SIGNAL ACROSS MODELS

Table 8 extends Figures 1 and 2 to all five models. On every model, the first-round hidden state orders requests better than prompt features, by 0.11 to 0.16 in $\rho .$ Denoising rounds span two orders of magnitude on the LLaDA2.x models and SDAR-8B. DiffusionGemma, whose single 256-token output block is denoised in ten rounds at the median, spans only 24×; even there, the first-round hidden state raises $\rho$ by 0.11.

Table 8. Denoising-round spread and predictor rank correlation across models. Rounds: denoising rounds per request over all 2,048 training requests; the minimum is 2 on every model. $\rho \colon$ Spearman correlation with true denoising workload under five-fold cross-validation on the requests used to fit the predictors, with the estimators of Figure 1: LightGBM on prompt features and a ridge probe on the mean-pooled first-round hidden state.
<table><tr><td rowspan="2">Model</td><td colspan="3">Rounds per request</td><td colspan="2">ρ</td></tr><tr><td>p50</td><td> $\mathsf { p } 9 0$ </td><td>max</td><td>Prompt</td><td>1st round</td></tr><tr><td>LLaDA2.0-mini</td><td>110</td><td>236</td><td>292</td><td>0.657</td><td>0.809</td></tr><tr><td>LLaDA2.1-mini</td><td>46</td><td>149</td><td>279</td><td>0.740</td><td>0.868</td></tr><tr><td>LLaDA2.2-mini</td><td>53</td><td>189</td><td>296</td><td>0.723</td><td>0.863</td></tr><tr><td>SDAR-8B</td><td>97.5</td><td>252</td><td>317</td><td>0.697</td><td>0.853</td></tr><tr><td>DiffusionGemma</td><td>10</td><td>18</td><td>48</td><td>0.697</td><td>0.809</td></tr></table>

## B ADMISSION PREDICTOR SELECTION

Table 9 compares regressors for the admission-time predictor on the same three prompt features. Because the predictor runs at every admission and only needs to order requests, we compare rank correlation, measured by five-fold crossvalidation on the training split, and per-request time, measured from raw prompt text on one Grace CPU thread. The small LightGBM model ranks best $( \rho = 0 . 7 4 0 )$ ) at the lowest cost (0.42 ms). Larger LightGBM and XGBoost models rank no better, the MLP and ridge regression rank lower, and ExtraTrees takes 8× longer for a lower ρ. About half of the per-request time is tokenization; the LightGBM inference itself takes $2 9 \mu \mathrm { s }$

Table 9. Model selection for the admission-time workload predictor. Time is the per-request median from raw prompt text, including tokenization and feature extraction.
<table><tr><td>Predictor</td><td>ρ Time (ms, single request)</td></tr><tr><td>Ridge 0.535</td><td>0.42</td></tr><tr><td>MLP (256–256)</td><td>0.701 0.44</td></tr><tr><td>XGBoost (100, depth 6) 0.706</td><td>0.65</td></tr><tr><td>LightGBM (500, 31 leaves) 0.709</td><td>0.46</td></tr><tr><td>LightGBM (1,000, 63 leaves)</td><td>0.689 0.54</td></tr><tr><td>ExtraTrees (100) 0.722</td><td>3.57</td></tr><tr><td>LightGBM (100, 15 leaves)</td><td>0.740 0.42</td></tr></table>

## C SEED ROBUSTNESS

The main results average three runs of one fixed trace per model. To check that the speedup does not depend on this trace, we generate three traces for LLaDA2.1-mini with different seeds. Each trace changes the request order and, under burst+Poisson arrivals, the Poisson arrival times. We run SGLang and DWISE once on each trace. Table 10 reports the mean and standard deviation across the three traces. DWISE reduces mean latency by $7 . 1 0 \pm 0 . 1 4 \times$ under burst arrivals and $1 0 . 7 2 \pm 0 . 1 4 \times$ under burst+Poisson arrivals relative to SGLang, consistent with the single-trace results in Table 2.

Table 10. Seed robustness on LLaDA2.1-mini: mean ± standard deviation over three traces generated with different seeds, one run each.
<table><tr><td>Arrival Policy</td><td></td><td>Mean latency (s)</td><td>Speedup</td></tr><tr><td rowspan="2">B</td><td>SGLang</td><td> $5 2 8 . 0 \pm 7 . 8$ </td><td>1.00×</td></tr><tr><td>DWISE</td><td> ${ \bf 7 4 . 3 \pm 0 . 4 }$ </td><td> ${ \bf 7 . 1 0 \pm 0 . 1 4 \times }$ </td></tr><tr><td rowspan="2">P</td><td>SGLang</td><td> $5 2 1 . 8 \pm 6 . 8$ </td><td>1.00×</td></tr><tr><td>DWISE</td><td> ${ \bf 4 8 . 7 \pm 1 . 1 }$ </td><td> ${ \bf 1 0 . 7 2 \pm 0 . 1 4 \times }$ </td></tr></table>

A request’s number of denoising rounds also varies slightly from run to run, because it is batched with different requests in each run. Table 11 compares per-request round counts across runs on LLaDA2.1-mini. Under DWISE, 98% of requests need exactly the same number of rounds when the same trace is replayed, and 92% across different traces; under SGLang, 37% do across seeds. Rank correlation stays at 0.97 or above in every case, so the per-request workload that DWISE predicts is stable enough to rank requests by. The same variation is why the oracle in Figure 8, which uses round counts from a profiling run, is an approximate upper bound.

Table 11. Stability of per-request denoising round counts across runs on LLaDA2.1-mini. Identical: share of requests with the same round count in both runs.
<table><tr><td>Policy</td><td>Runs compared</td><td>Identical (%)</td><td>ρ</td></tr><tr><td>DWISE</td><td>same trace</td><td>98</td><td>0.998</td></tr><tr><td>DWISE</td><td>different traces</td><td>92</td><td>0.994</td></tr><tr><td>SGLang</td><td>different traces</td><td>37</td><td>0.972</td></tr></table>

## D OUTPUT LENGTH VERSUS REMAINING WORK

DWISE measures work in prefill blocks and denoising rounds rather than output tokens. On LLaDA2.1-mini, the two diverge sharply. The 57% of requests that use the full 256-token budget have the same output length but work ranging from 38 to 184 (10th to 90th percentile), so output length cannot order them. Across categories, the two even move in opposite directions: code and math requests produce more tokens than chat and instruction requests (242 vs.

215 on average) but need 2.6× fewer denoising rounds (45 vs. 115).

Output-length rankers therefore misorder the requests that carry most of the work (Table 12). SSJF and LTR correlate with work overall $( \rho = 0 . 5 1$ and 0.48) only because they rank MCQ and Hard MCQ requests first, which are 23% of requests but 3.1% of the work. On the other 96.9% of the work, their scores are uncorrelated with it $( \rho = - 0 . 0 2 $ and −0.07). Because mean latency weighs a misordering by the work of the request placed first, this costs more than it saves: the mean latency of code and math requests rises from 524 s under SGLang to 842 s (SSJF) and 925 s (LTR). DWISE’s estimates, in contrast, stay correlated with work on these requests $( \rho = 0 . 5 4$ at admission and 0.84 after refinement). Even its admission-time round estimate alone, without the known prefill blocks, reaches $\rho = 0 . 4 6$ , so a 3-feature predictor of rounds outperforms the BERT- and OPT-based predictors of output length on these requests.

Prefill blocks can dominate the work. SDAR-8B uses 4- token blocks, so a long-context request needs 2,684 prefill blocks on average but produces only 44 output tokens. These requests are 3.7% of requests yet 40.7% of the work. Output-length rankers see only the short output and rank them ahead of most requests (median at the 31st–38th percentile); DWISE counts their prefill blocks and ranks them near the end (98th percentile).

Table 12. Output-length rankers on LLaDA2.1-mini under burst arrivals (∆ averages three runs; $\rho$ uses the first). Work: prefill blocks plus denoising rounds. $\rho \colon$ Spearman correlation between a scheduler’s ranking score and work within the group; both include the exactly known prefill blocks, so ρ is higher than for denoising rounds alone. “first”: ranked ahead of nearly all other requests. DWISE adm./ref.: its score at admission and after refinement. ∆: the group’s contribution to the change in mean latency relative to SGLang; omitted for DWISE, whose change also includes the gain from mixed-phase batching.
<table><tr><td colspan="2">(Hard) MCQ</td><td>Other</td><td>All</td></tr><tr><td colspan="2">Requests</td><td>23%</td><td>77%</td><td>100%</td></tr><tr><td colspan="2">Work</td><td>3.1%</td><td>96.9%</td><td>100%</td></tr><tr><td colspan="2"> $\rho ,$  SSJF</td><td>first</td><td>-0.02</td><td>0.51</td></tr><tr><td colspan="2"> $\rho ,$  LTR</td><td>first</td><td>-0.07</td><td>0.48</td></tr><tr><td colspan="2"> $\rho ,$  DWISE adm.</td><td></td><td>0.54</td><td>0.76</td></tr><tr><td colspan="2"> $\rho ,$  DWISE ref.</td><td></td><td>0.84</td><td>0.92</td></tr><tr><td colspan="2"> $\Delta$  SSJF (s)</td><td>-113</td><td>+170</td><td>+57</td></tr><tr><td colspan="2">∆ LTR (s)</td><td>-114</td><td>+214</td><td>+100</td></tr></table>

## E MULTI-GPU DEPLOYMENT AND OUTPUT BUDGET

The main experiments serve each model on one GPU. To test whether DWISE carries over to multi-GPU serving, we deploy LLaDA2.1-mini with tensor parallelism over two and four GPUs (TP=2, TP=4) and with expert parallelism over four GPUs (TP=4+EP=4), where attention stays tensor-parallel and each GPU holds a quarter of the 256 experts. SGLang’s dLLM path does not support pipeline parallelism. DWISE runs unchanged. Every rank runs the same scheduler on the same request state, and the diffusioninvolved predictor reads the final-layer hidden state, which is replicated across ranks after the tensor-parallel all-reduce. On shorter runs, we checked that all ranks make identical scheduling decisions in every round. Both predictors are the ones trained for the single-GPU runs. All runs in Table 13, including the single-GPU row, come from one batch separate from Table 2, with three runs per configuration.

DWISE keeps its speedup on every configuration: 10.5– 12.2× under burst+Poisson arrivals and 6.7–7.1× under burst arrivals, against 11.8× and 7.4× on one GPU. For comparison, moving SGLang from one to four GPUs raises its throughput by only 1.2×. Ranking and admission take 0.76– 1.01 ms and the diffusion-involved predictor 0.72–0.83 ms per round, each about 1% of the median round, as on one GPU. Task accuracy stays between 43.8% and 44.5%, and the rank correlation of refined estimates differs from the single-GPU value by at most 0.004.

Table 13. Mean latency (s) of LLaDA2.1-mini under multi-GPU deployment, averaged over three runs. B: burst; P: burst+Poisson; Sp.: speedup (×) of DWISE over SGLang.
<table><tr><td></td><td colspan="2">B</td><td colspan="2">P</td></tr><tr><td>Config</td><td>SGLang</td><td>DWISE  $\mathrm { { S p . } }$ </td><td>SGLang</td><td>DWISE</td><td> $\mathrm { { S p . } }$ </td></tr><tr><td>1 GPU</td><td>543.0</td><td>73.07.4</td><td>540.2</td><td>45.9</td><td>11.8</td></tr><tr><td>TP=2</td><td>498.1</td><td>74.56.7</td><td>483.5</td><td>46.1</td><td>10.5</td></tr><tr><td>TP=4</td><td>457.4</td><td>64.77.1</td><td>439.4</td><td>36.0</td><td>12.2</td></tr><tr><td> $\scriptstyle \mathrm { T P = 4 + E P = 4 }$ </td><td>454.2</td><td>65.4 6.9</td><td>444.7</td><td>38.3</td><td>11.6</td></tr></table>

The main experiments also cap generation at 256 tokens. Table 14 repeats the single-GPU LLaDA2.1-mini experiment with budgets of 512, 1,024, and 2,048 tokens, one run per setting. For each budget we retrain both predictors with the unchanged pipeline on a training run at that budget; all other settings match the main experiments. Raising the budget from 256 to 2,048 tokens raises the mean output from 176 to 536 tokens and the mean number of denoising rounds from 62 to 180. At 2,048 tokens, DWISE is still 4.45× (burst) and 5.18× (burst+Poisson) faster than SGLang in mean latency and 1.74× faster at the 99th percentile, and its refined estimate keeps $\rho = 0 . 8 8 5  – 0 . 8 9 4$ at every budget. Tripling the denoising rounds raises SGLang’s mean latency by only 1.3–1.5×, because its time goes to prefillonly rounds, whose number is set by the prompts rather than the output length; DWISE’s latency instead grows with the denoising work (2.4–3.2×).

Table 14. Output-budget sweep on LLaDA2.1-mini. Tok. and Rnd.: mean output tokens and denoising rounds per request in the SGLang run. SGLang and DWISE: mean latency (s). Sp.: speedup (×) of DWISE over SGLang in mean latency, with the p99 ratio in parentheses, from unrounded latencies. ρ: Spearman correlation between DWISE’s refined estimate and the denoising rounds left after the first round. The 256-token row is rerun in the same batch. B: burst; P: burst+Poisson.
<table><tr><td>Budget</td><td>Tok.</td><td>Rnd.</td><td>SGLang</td><td>DWISE</td><td>Sp.</td><td>ρ</td></tr><tr><td rowspan="3">B</td><td>256</td><td>176</td><td>62</td><td>526.2</td><td>73.0</td><td>7.21 (2.12)</td><td>0.894</td></tr><tr><td>512</td><td>295</td><td>100</td><td>607.3</td><td>104.9</td><td>5.79 (1.99)</td><td>0.889</td></tr><tr><td>1,024 2,048</td><td>425 536</td><td>144 180</td><td>688.7 782.1</td><td>145.3 175.8</td><td>4.74 (1.78) 4.45 (1.74)</td><td>0.892 0.886</td></tr><tr><td rowspan="3">P</td><td>256</td><td>177</td><td>62</td><td>568.7</td><td>45.4</td><td>12.54 (2.45)</td><td>0.894</td></tr><tr><td>512</td><td>294</td><td>100</td><td>610.0</td><td>76.6</td><td>7.97 (2.01)</td><td>0.890</td></tr><tr><td>1,024</td><td>426</td><td>145</td><td>664.1</td><td>113.4</td><td>5.86 (1.75)</td><td>0.891</td></tr><tr><td></td><td>2,048</td><td>537</td><td>181</td><td>764.5</td><td>147.5</td><td>5.18 (1.74)</td><td>0.885</td></tr></table>

## F TIME TO FIRST TOKEN

Section 3.3 notes that SGLang inherits prefill-first execution from AR serving, where it protects first-token latency. For a dLLM, we measure time to first token (TTFT) as the time from a request’s arrival until its first output block is committed. That block still needs its denoising rounds, so prefill-first execution delays it whenever prompts are waiting. Table 15 reports TTFT on LLaDA2.1-mini, from the same batch of runs as the single-GPU row of Table 13.

Under SGLang, mean TTFT is close to mean end-to-end latency: requests wait almost as long for their first block as for their last. DWISE lowers mean TTFT by 13.0× under burst+Poisson arrivals and 7.8× under burst arrivals, and p99 TTFT by 2.2–2.4×. The median request receives its first block after 6.8 s instead of 570.5 s under burst+Poisson arrivals, because short requests are admitted and denoised first. The reduction holds under multi-GPU deployment: 11.6–13.7× under burst+Poisson arrivals and 7.0–7.4× under burst arrivals.

Table 15. Time to first token (s) on LLaDA2.1-mini, from arrival until the first output block is committed, averaged over three runs. B: burst; P: burst+Poisson.
<table><tr><td>Arrival</td><td>Policy</td><td>Mean</td><td>p50</td><td>p99</td></tr><tr><td rowspan="2">B</td><td>SGLang</td><td>525.2</td><td>541.1</td><td>1035.4</td></tr><tr><td>DWISE</td><td>67.1</td><td>45.2</td><td>472.3</td></tr><tr><td rowspan="2">P</td><td>SGLang</td><td>521.5</td><td>570.5</td><td>983.8</td></tr><tr><td>DWISE</td><td>40.0</td><td>6.8</td><td>403.3</td></tr></table>

## G PER-CATEGORY TAIL LATENCY

Table 16 breaks down p99 and maximum latency by request category on LLaDA2.1-mini. DWISE improves both p99 and maximum latency in every category under both arrival patterns. Long-context requests bear the cost of

SRPT: their prompts span over 300 prefill blocks, so DWISE runs them after shorter requests, and 44% (burst) and 39% (burst+Poisson) of them individually finish later than under SGLang. Their $\mathsf { p } 9 9$ and maximum latency still drop by about half, because SGLang delays them as well behind the prefill-first backlog.

Table 16. Per-category p99 and maximum latency (s) on LLaDA2.1-mini, SGLang → DWISE (per-run values averaged over three runs). B: burst; P: burst+Poisson.
<table><tr><td rowspan="2">Category</td><td colspan="2">B</td><td colspan="2">P</td></tr><tr><td> $\mathsf { p } 9 9$ </td><td>Max</td><td>p99</td><td>Max</td></tr><tr><td>Chat</td><td>1048→215 1050→244</td><td></td><td>1009→196</td><td>1016→224</td></tr><tr><td>Instruction</td><td>1045→246 1049→260</td><td></td><td>960→219</td><td>962→233</td></tr><tr><td>Code</td><td>1038→81</td><td>1041→130</td><td>950→44</td><td>952→116</td></tr><tr><td>Math</td><td>1040→83</td><td>1043→133</td><td>950→52</td><td>954→114</td></tr><tr><td>Multiling. math 1038→121</td><td></td><td>1045→125</td><td>953→86</td><td>958→105</td></tr><tr><td>MCQ</td><td>1036→17</td><td>1039→68</td><td>947→10</td><td>948→38</td></tr><tr><td>Hard MCQ</td><td>1035→6</td><td>1039→16</td><td>945→2</td><td>948→4</td></tr><tr><td>Long context</td><td>1039→510 1042→511</td><td></td><td>954→498</td><td>956→498</td></tr></table>

## H PER-CATEGORY GENERALIZATION

Table 17 gives the per-category results behind Table 7. For each of the eight request categories, we remove it from the 2,048 training requests, retrain both predictors on the remaining categories, and measure ρ among that category’s evaluation requests. Each cell reports ρ with the category in training → held out.

The admission predictor transfers poorly. Even in distribution, prompt size features order requests within a category only weakly $( \rho = 0 . 0 4 \ – 0 . 3 4 )$ . Once a category is held out, ρ turns negative in five of the seven defined cases, stays near zero for chat, and is positive only for multilingual math. On long context, the held-out predictor assigns every request the same value, since these prompts are far longer than any seen in training, so $\rho$ is undefined (n/a).

The refined estimate keeps a positive ρ on every held-out category. Its drop ranges from 0.02 on hard MCQ to 0.17 on long context, and the held-out ρ stays between 0.40 and 0.62 except on long context. The first-round hidden state thus carries information about denoising workload that transfers across request types, whereas prompt size features do not.

Table 17. Leave-one-category-out ρ on LLaDA2.1-mini.
<table><tr><td>Held-out category</td><td>Admission ρ</td><td>Refined  $\rho$ </td></tr><tr><td>Hard MCQ</td><td>0.200 → −0.154</td><td>0.481 → 0.463</td></tr><tr><td>MCQ</td><td>0.237 → −0.185</td><td>0.686 → 0.602</td></tr><tr><td>Code</td><td>0.251 → −0.202</td><td>0.657 → 0.539</td></tr><tr><td>Math</td><td>0.259 → −0.069</td><td>0.540 → 0.399</td></tr><tr><td>Multilingual math</td><td>0.087 → 0.163</td><td>0.755 → 0.623</td></tr><tr><td>Long context</td><td>0.338 → n/a</td><td>0.352 → 0.178</td></tr><tr><td>Chat</td><td>0.039 → 0.037</td><td>0.682 → 0.577</td></tr><tr><td>Instruction</td><td>0.053 → −0.033</td><td>0.658 → 0.561</td></tr></table>