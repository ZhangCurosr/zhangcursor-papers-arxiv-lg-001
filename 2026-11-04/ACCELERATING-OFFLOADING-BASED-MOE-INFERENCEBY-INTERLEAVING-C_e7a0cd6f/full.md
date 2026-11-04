# ACCELERATING OFFLOADING-BASED MOE INFERENCEBY INTERLEAVING CONCURRENT REQUESTS

Hongseung Yu <sup>1</sup> In Gim <sup>2</sup> Kyunghan Lee <sup>1</sup> Lin Zhong <sup>2</sup>

## ABSTRACT

Expert offloading enables large mixture-of-experts (MoE) language models to run locally, but conventional batching stalls the GPU while missing experts are fetched. A popular remedy, known as prefetching, is to predict what experts will be needed soon and fetch them using idle link bandwidth. Imprecise prediction, however, wastes bandwidth and GPU memory on fetching unnecessary experts. Our key finding is that prefetching can be harmful when there is an expert cache in GPU memory. This is because as the cache includes more necessary experts, expert prediction becomes less precise. Our key insight is that proper request scheduling beats prefetching. Building on this insight, we present Bevy, an MoE inference system based on runahead inference for concurrent requests. A request whose experts are available can run ahead to later layers while other requests wait for fetches, revealing the experts it needs next. Bevy can then fetch the missing experts so identified while computing ready requests, overlapping computation and fetching without predicting expert selections. Running requests separately, however, repeats reads of shared weights in GPU memory and increases compute time. To limit this cost, Bevy’s self-balancing scheduler batches ready requests at each layer and prioritizes lagging requests, adapting their grouping as expert availability changes. Across two local platforms, Bevy delivers up to 3.46× the decode throughput of llama.cpp and up to 1.55× that of the fastest prefetching baseline.

## 1 INTRODUCTION

Large language models (LLMs) are increasingly deployed on local systems such as personal desktops and AI workstations (Apple, 2026; NVIDIA, 2026). Although such local deployments typically serve a single user, applications that use these models often perform multiple tasks concurrently, generating several requests to the underlying model. Personal agents may execute subtasks in parallel, while chat applications can process multiple queries on the same model simultaneously (OpenClaw, 2026; LM Studio Team, 2026). The ability to serve concurrent requests efficiently is therefore becoming more important for local inference.

Local inference must make efficient use of limited compute resources. Mixture-of-experts (MoE) models are well suited to this setting because they reduce computation by activating only a few experts per token. Their full set of expert weights, however, can still exceed the memory available for inference on a personal device. Expert offloading systems keep a subset of the experts in GPU memory as an expert cache, managed by a replacement policy such as LRU or LFU. Missing experts are fetched from secondary storage over a link such as PCIe (ggml-org, 2023; Liang et al., 2026; Xue et al., 2024). As shown in Figure 1(a), after each layer’s gate network selects the experts, the system fetches those absent from the cache before computing the MoE. This sequence repeats at every layer, leaving the GPU idle during fetches and the link idle during computation.

Prefetching addresses this GPU idle time by predicting experts needed at later layers and fetching them while the GPU computes (Figure 1(b)) (Yu et al., 2026; Eliseev & Mazur, 2023; Song et al., 2024). However, we find that the benefit of prefetching is questionable when combined with an expert cache (§2.4). The cache already holds many of the experts needed next and tends to retain those that are easier to predict. The prefetcher must therefore choose among the remaining experts, of which needed experts are less common and harder to predict. Prefetching can keep the link busy, but much of that bandwidth is wasted fetching experts that requests do not need.

Our key insight is that concurrent requests can use otherwise idle link bandwidth to fetch experts known to be needed, without prediction. That is, proper request scheduling beats prefetching. Conventional batching computes requests together at each layer after the experts needed by the entire batch have arrived. Allowing a request to proceed as soon as its own experts are available lets it reach the next layer earlier and identify the experts needed there, including those missing from the cache. The system can then fetch these newly identified missing experts while the GPU computes other requests whose experts have arrived (Figure 1(c)). We call this execution method runahead inference.

![](images/cad6f702ffe85fbd530ce25b355ed992bc147be8607a2eec099e9479828af938.jpg)  
Figure 1. Timeline of two concurrent requests, R1 and R2. (a) With batching, the GPU idles during fetches and the link idles during computation. (b) Same as (a), but predicted experts are prefetched during computation. Because prediction is never perfect, prefetching unnecessary experts wastes both bandwidth and GPU memory. (c) Each request runs without waiting for the other’s fetches, so fetches overlap with computation.

Building on this idea, we present Bevy, an MoE inference system for concurrent requests. A key challenge is that overlapping computation and expert fetching does not by itself guarantee a shorter inference step. Running requests separately repeats reads of shared weights in GPU memory, increasing compute time and potentially offsetting the benefit of overlap. We find that statically grouping requests, e.g., based on profiling as in NanoFlow (Zhu et al., 2025a), is inadequate. This is because the number of expert cache misses, and hence the fetch time, varies substantially from step to step (Figure 4). Static grouping can therefore leave compute time unnecessarily high in some steps and keep ready requests waiting in others.

To address this problem, Bevy groups requests layer by layer dynamically, at runtime, according to request readiness and progress. At each scheduling decision, it batches the ready requests at the selected layer. Bevy gives priority to requests at earlier layers, both for computation and for expert fetching. This helps them catch up with other requests and creates more opportunities to batch requests at the same layer. With more expert misses, fewer requests tend to be ready together, so smaller batches run ahead while experts are fetched for other requests. With fewer misses, more requests tend to become ready together, allowing larger batches to reduce compute time. Together, these policies make execution selfbalancing: grouping adapts across layers within a step as well as across steps, without profiling or a fixed partition.

We implement Bevy on llama.cpp (ggml-org, 2023) with Metal and CUDA support and evaluate it on two local inference platforms using two MoE models and chat and agentic coding workloads. Across experiments varying concurrency and expert cache size, Bevy delivers 1.26–3.46× the decode throughput of llama.cpp and 1.01–1.55× that of the fastest prefetching baseline. Bevy achieves these gains without predicting expert selections, by exposing demand early and adapting request grouping during execution.

## 2 BACKGROUND & MOTIVATION

## 2.1 Concurrent Requests in Local LLM Inference

As LLMs spread to more applications, serving several requests at the same time is becoming common. LM Studio, a chat application, handles multiple queries from a user on the same model simultaneously (LM Studio Team, 2026). Personal agents such as OpenClaw execute code review and research sessions concurrently while the user continues an interactive session (OpenClaw, 2026). These agents also often split a single task, such as checking a mailbox, into four to eight subtasks and run them in parallel (Delgado, 2026). For such tasks, concurrent request throughput directly affects how long a single user waits for the overall result. Local inference thus increasingly needs to handle such concurrent requests efficiently.

## 2.2 Offloading-Based MoE Inference

A mixture-of-experts (MoE) LLM activates its experts sparsely: in each layer, the gate network after the attention picks a few experts for each token, so each inference step uses only a fraction of the model’s weights, which reduces computation. The set of experts needed in each step varies and is not known in advance, so the model still needs all of its experts for inference. When the experts do not fit in GPU memory, expert offloading systems store all of them in secondary storage (host memory or SSD) and fetch the experts needed for computation into GPU memory over a link such as PCIe (Yang et al., 2026; ggml-org, 2023; Liang et al., 2026; Xue et al., 2024).

The fetching strategy depends on how many tokens a step processes. Prefill of long input prompts and decode with large batches of concurrent requests process many tokens per step. Since the gate network picks a different set of experts for each token, these steps often need most of the experts in the model. The systems then fetch every expert of the next layer from secondary storage into GPU memory over the link ahead of time, in the same layerwise pipeline used to offload dense models (Sheng et al., 2023). In contrast, decode with a handful of concurrent requests (§2.1) processes few tokens per step and uses only a fraction of the experts. Since fetching whole layers is then inefficient, the systems instead keep a subset of the experts in GPU memory as an expert cache managed by a replacement policy such as LRU or LFU. After the gate network of a layer picks its experts, they fetch only the missing experts, those not in the expert cache, on demand over the link (Yu et al., 2026; Yang et al., 2026; Xue et al., 2024; Liang et al., 2026).

![](images/f5791934d759edec83bf4bbcb439f4518f505478fb2ff3b377969e583a72b719.jpg)  
(a) ProMoE (b) Mixtral-Offloading (c) FineMoE  
Figure 2. Prefetch precision under caching at 80% coverage of nonresident target-layer demand. To test whether the smaller fraction of needed candidates alone explains the decline, we compare the actual cache (solid) with a randomized cache (dashed). For each cache snapshot, the randomized cache holds the same numbers of needed and unneeded experts as the actual cache, selected randomly within each group. The dotted line is the no-cache reference. Region (1) shows the precision drop from this reference to the randomized cache; region (2) shows the additional drop to the actual cache.

## 2.3 Prefetching to Use Idle Link Bandwidth

On-demand fetches delay the step, since the computation cannot proceed until all the missing experts arrive (Figure 1(a)). To address this problem, many expert offloading systems prefetch experts. A prefetcher predicts the experts needed by a later layer that the GPU has not yet computed and fetches them into the expert cache in advance. As shown in Figure 1(b), the prefetcher fetches the experts while the GPU computes, so it also utilizes the otherwise idle link bandwidth. Prefetchers predict the experts in various ways. For instance, some feed the current hidden state into the gate network of the later layer (Eliseev & Mazur, 2023; Tang et al., 2026), some use past activation history and correlations (Xue et al., 2024; Yu et al., 2026; Fang et al., 2025), and others train a prediction model (Song et al., 2024; He et al., 2024; Shen et al., 2025).

## 2.4 Caching Reduces Prefetch Precision

However, we find that prefetching uses this otherwise idle bandwidth inefficiently when combined with an expert cache.

We evaluate ProMoE (Song et al., 2024), Mixtral-Offloading (Eliseev & Mazur, 2023), and FineMoE (Yu et al., 2026), one prefetcher for each prediction approach above. We run them in a trace-driven cache simulator on Qwen3.6-35B-A3B (Qwen Team, 2026) decode traces of eight concurrent requests from ShareGPT (ShareGPT, 2023). All three prefetchers use the same LFU cache policy. We vary the cache size and compare precision—the fraction of selected nonresident experts needed by the target layer— at a fixed coverage of nonresident demand (Figure 2).

![](images/2d9b77c4de28ce4bb5cfc65666d70274dc298cbb7b868243787efc78706c092e.jpg)  
Figure 3. Example timeline of runahead inference with three requests that reach layer l together. (a) Each request runs without waiting for the other requests’ fetches. (b) Same as (a), but R2 and R3 are grouped. (c) Same as (a), but the experts of R3 are fetched before those of R2.

Precision decreases as the cache grows for all three prefetchers. The cache already holds a large share of upcoming demand: a cache holding 30% of all experts contains 54.1% of the experts needed by the target layer on average. Needed experts therefore make up a smaller fraction of the nonresident candidates. To see whether this alone explains the decline, we randomize which experts are cached while keeping the fraction of needed candidates unchanged. The randomized cache reproduces part of the decline (region 1), while the actual cache shows an additional drop (region 2). This suggests that the cache tends to hold experts the prefetchers can predict accurately, so the remaining candidates are harder to predict.

Prefetching can thus keep the link busy, but much of what it fetches is not needed by the target layer. Filling idle periods with predictions therefore does not ensure that the bandwidth serves actual demand. Can we reduce idle link time while fetching only experts known to be needed? The next section presents a request scheduling method designed to achieve this goal.

## 3 RUNAHEAD INFERENCE

In this section, we present runahead inference, a novel scheduling method for offloading-based MoE inference. The key idea is to let ready requests run ahead to later layers and reveal their expert needs early, without waiting for other requests’ fetches. This allows the link to stay busy fetching only experts known to be needed, without relying on predictions.

![](images/273c8cd16084f66e2f3a2ba3dbcd974581cf7ea5514cdd0c9306cf662c58b992.jpg)

![](images/4e6af92413b9138298cab3d632970ad764c98b1742a0de761652401e2f1c860b.jpg)  
(b)  
Figure 4. Compute and fetch time per step for six concurrent requests (Gemma4-26B-A4B, Agentic Coding, 30% expert cache, Apple M6). (a) Compute time when the requests run in 1, 2, 3, or 6 groups. (b) Fetch time of each step over 256 steps.

## 3.1 Running Requests at Different Layers

Figure 3(a) shows in detail how runahead inference runs three requests (R1, R2, and R3) that reach layer l together. ❶ Only R1 finds all of its experts for layer l in the expert cache, so it runs the MoE right away instead of waiting for the fetches of R2 and R3. ❷ R1 then runs the attention and the gate of layer l+1, as their weights remain in GPU memory. The gate picks its experts while the link is still busy, so the fetches of its missing experts queue behind those of R2 and R3. ❸ Meanwhile, the GPU runs R2 once its fetches finish, ❹ followed by R3. ❺ By the time R3 finishes, the fetches of R1 for layer l+1 are done, so the GPU moves on to R1 right away. As the requests proceed through the remaining layers in this way, the link bandwidth is filled only with fetches of missing experts.

## 3.2 Challenges in Grouping and Ordering

To understand when runahead inference works well, we first define the compute time and fetch time of a step. The compute time is the total time the GPU is busy computing during the step. For the three requests in Figure 3(a), it is the sum of the time the GPU spends on R1, R2, and R3. Similarly, the fetch time is the total time the link is busy fetching the step’s missing experts. A step must complete both computation and fetching, so it cannot be shorter than the longer of these two times. Its actual duration also depends on how much computation and fetching overlap across requests. Effective scheduling must therefore aim to (i) keep both compute time and fetch time short and (ii) overlap computation and fetching across requests as much as possible. However, two challenges make this difficult.

C1: Grouping Granularity. Running each request separately increases compute time. With a small number of concurrent requests, GPU computation is memory-bound, so the GPU reads shared weights from GPU memory again for each request. In Figure 4(a), computing six requests as one batch takes 56.5 ms per step, whereas computing them one at a time takes 126.8 ms in total, above the mean fetch time (dashed line). However much computation and fetching overlap, the step still cannot be shorter than this compute time.

![](images/b75ee86aa1b648b425f79ddd539f5fe5179e4bd5240b1f7376a9ebe67aca20a9.jpg)  
Figure 5. Accumulated extra misses over the mean for four concurrent requests ranked by their misses in layers 0–3 (Qwen3.6-35B-A3B, ShareGPT, 30% expert cache).

One possible solution is static grouping, which splits the requests into groups in advance and runs each group as one batch for the whole step. For example, forming two groups of three requests or three groups of two requests reduces the compute time from 126.8 ms to 74.6 ms or 90.1 ms, respectively, both below the mean fetch time (Figure 4(a)). Static grouping, however, has two limitations. First, it restricts when requests can run ahead. In Figure 3(b), grouping R2 and R3 reduces the number of scheduling units from three to two. R2 must wait for R3’s experts instead of computing while they are fetched. After R1’s fetches for layer l+1 finish, the rest of the group’s computation runs without any overlapping fetches. Grouping can therefore reduce compute time at the cost of fewer opportunities to overlap computation and fetching.

Second, even under the same grouping, the fetch time and the compute time change from step to step. Because each step misses a different number of experts, the fetch time ranges from 55.7 to 160.1 ms over 256 steps (Figure 4(b)). The compute time also grows as the context of each request gets longer. The compute time of a grouping can thus be shorter than the fetch time in one step but longer in another, so no grouping fixed in advance suits every step. In §4.3, we present a mechanism that groups requests at each layer at runtime.

C2: Scheduling Order. When multiple requests are ready for computation or have experts to fetch, the scheduling order affects how much computation and fetching can overlap.

![](images/2619e0735e226976104975bd701638f20a3fb8fc5a0fdd2327f7bd77c9c0a4ff.jpg)  
Figure 6. System overview of Bevy.

Figure 3(c) shows the same three requests as Figure 3(a), but the link fetches the experts of R3, which misses more, before those of R2. When R1 finishes, neither R2 nor R3 is ready, so the GPU sits idle while the link continues fetching. By contrast, in Figure 3(a), the link serves R2 first, so R2 is ready when R1 finishes and its computation overlaps with the remaining fetches. A natural approach is therefore to fetch first for the request with the fewest remaining missing experts, allowing its computation to start sooner and overlap with fetches for other requests.

However, always prioritizing requests with fewer misses can leave other requests behind. Figure 5 ranks four requests by their misses in layers 0–3 and tracks their accumulated extra misses over the mean in later layers. Requests that miss more early tend to keep missing more. Repeatedly favoring requests with fewer misses can therefore cause a request with more misses to accumulate delays and fall behind as a straggler. The step cannot end until the straggler finishes its remaining fetches and computation alone, without other requests’ work to overlap with. The scheduling order must therefore consider both the overlap available now and the progress of all requests through the step.

## 4 BEVY SYSTEM DESIGN

In this section, we present Bevy, an MoE inference system based on runahead inference for concurrent requests.

## 4.1 Overview

Figure 6 presents an overview of Bevy, which consists of three modules: the Compute Scheduler, the Job Issuer, and the Expert Fetcher. Bevy divides the computation of each request into a sequence of jobs. Job l runs the MoE of layer l, followed by the attention and the gate of layer l+1 (Figure 6 labels it $l {  } l { + } 1 )$ . Bevy keeps a Progress Table that records the current job and state of each request. Using this perrequest information, Bevy schedules requests independently, allowing ready requests to run ahead while others wait for

Algorithm 1 Bevy’s Scheduling Policy   
$J _ { B }$ Blocked jobs with remaining missing experts   
$J _ { R }$ Ready jobs   
$l ( j )$ Layer of job j   
$m ( j )$ Number of remaining missing experts of job j   
function SELECTFETCHJOB(J<sub>B</sub>) ▷ Fetcher   
if $J _ { B } = \emptyset$ then return none   
return arg $\mathrm { m i n } _ { j \in J _ { B } } \left( l ( j ) , m ( j ) \right)$ ▷ lowest-layer-first,   
then SRTF   
end function   
function SELECTCOMPUTEGROUP(J ) ▷ Scheduler   
if $J _ { R } = \emptyset$ then return $\varnothing$   
$l _ { \mathrm { m i n } } \gets \operatorname* { m i n } _ { j \in J _ { R } } l ( j )$   
return $\{ j \in J _ { R } | l ( j ) = l _ { \operatorname* { m i n } } \}$ ▷ one batch (§4.3)   
end function

## experts.

At the start of each step, all requests run the attention and the gate of the first layer together, then move on to their first jobs. For each job, the Job Issuer checks whether its experts are in the expert cache. If all are present, it issues the job to the compute queue. Otherwise, it blocks the job and puts its missing experts into the fetch queue. The Expert Fetcher fetches these experts from secondary storage into the expert cache over the link, and the Job Issuer issues the job once all its missing experts have arrived. The Compute Scheduler commits jobs from the compute queue to keep the GPU busy. When a job ends, the next job of the same request is passed to the Job Issuer, and this process repeats. The last job runs only the MoE of the final layer. Once all requests finish their last jobs, they run the LM head together to complete the step.

The rest of this section presents the techniques that Bevy builds on this structure to address the two challenges in §3.2. §4.2 describes how the Compute Scheduler and the Expert Fetcher determine the order of computation and fetching (C2). §4.3 describes how Bevy groups ready jobs at the same layer and runs them as one batch (C1). Finally, §4.4 describes how Bevy manages the expert cache.

## 4.2 Lowest-Layer-First Scheduling

To address C2 (§3.2), Bevy prioritizes both computation and expert fetches by request progress (Algorithm 1). It gives priority to the job at the lowest layer, that is, to the request with the least progress. This request has the most jobs left, and the step ends only when the last request finishes. Serving it first follows critical-path scheduling (Hu, 1961) and cuts the extra waiting time of the requests that lag behind. The Scheduler selects the ready job at the lowest layer, and ready jobs at other layers run only while the lowest-layer job is blocked. The Fetcher likewise fetches the missing experts of the lowest-layer job first. This order also helps lagging requests catch up, creating batching opportunities, which we detail in §4.3.

![](images/1bf8bf1e2eb3e6620f0e99b241b353e4363e2fa718a399c2339381af3270db7e.jpg)  
Figure 7. Operation of opportunistic layerwise grouping. (a) With many misses, only R2 is ready when R1 finishes, so R2 runs alone. (b) With few misses, R2, R3, and R4 are all ready and run together as one batch.

Among requests at the same layer, the Fetcher first serves the one with the fewest remaining missing experts, as in shortest-remaining-time-first (SRTF) scheduling. That request becomes ready first, so its computation can overlap with the fetches of the others. Because SRTF orders only the requests at the same layer, the new fetches of a leading request never go before the pending fetches of the requests that lag behind.

## 4.3 Opportunistic Layerwise Grouping

To address C1 (§3.2), Bevy groups all ready jobs at the layer selected by the Scheduler and commits them as one batch (Algorithm 1). The group lasts only for this layer, rather than the whole step. All requests selected for a step still start and finish it together. This preserves the step-level interface used by iteration-level scheduling (Yu et al., 2022), allowing the request set to change between steps.

Figure 7 illustrates how the batches depend on expert misses. In both rows, R1 runs job l while the link fetches the missing experts of R2, R3, and R4 in that order (§4.2). With many misses (Figure 7(a)), only R2 is ready when R1 finishes, so R2 runs alone. With few misses (Figure 7(b)), R2, R3, and R4 are all ready, so the Scheduler batches their jobs and runs them together.

This gives layerwise grouping a self-balancing property that addresses the two limitations of static grouping in C1 (§3.2). With few misses, more jobs tend to become ready together, so batching reduces compute time when fetch time is short. With many misses, fewer jobs are ready together, so requests can run ahead in smaller batches and overlap with the remaining fetches. The grouping adapts not only between steps but also across layers within a step, without profiling. The priorities in §4.2 help create batching opportunities: lowest-layer-first helps lagging requests catch up, while SRTF can make additional jobs at the same layer ready before the next commit. We compare this adaptive grouping with static groupings in §5.4.1.

Waiting for more requests to become ready could yield a larger batch and further reduce compute time. For example, in Figure 7(a), the Scheduler could wait for R3 to become ready and batch it with R2. However, waiting also delays their next gate computation and the fetches it would trigger, potentially reducing opportunities to overlap those fetches with other requests’ computation. Alternatively, if a higherlayer job is ready, the Scheduler could defer R2 and run that job first. This gives R3 time to become ready but moves the leading request further ahead, potentially losing batching opportunities at later layers. In either case, the benefit of waiting depends on how the remaining jobs proceed and cannot be guaranteed at the time of scheduling. Bevy therefore batches only the jobs already ready at each commit, without delaying the selected job for additional requests.

## 4.4 Expert Cache Management

In conventional batched execution, the system fetches the union of the missing experts of all requests at a layer. With runahead inference, jobs request experts independently and can need the same expert at different times. To avoid duplicate reads, Bevy tracks each expert’s pending fetch. If an expert is already queued for fetching or being fetched for another job, the new job shares that fetch rather than issuing another.

As jobs progress independently, a fetch for one job must not evict experts still needed by another. At lookup, Bevy immediately pins the cache entry of a resident expert. For a new fetch, it first reserves and pins a cache entry, evicting an unpinned expert if necessary. Each entry’s reference count tracks all jobs that need it, including those sharing a pending fetch. Each job releases its reference only after completing on the GPU, and only entries with zero references can be evicted. To prevent deadlock, Bevy enables runahead inference only when the cache has enough entries to hold every concurrent request’s selected experts without relying on expert sharing. For example, 128 cache entries are sufficient for 16 concurrent requests selecting eight experts each. For Qwen3.6-35B-A3B, this is only 1.25% of all experts.

## 5 EVALUATION

## 5.1 Experimental Setup

Implementation. Bevy is implemented on llama.cpp (ggmlorg, 2023) with support for Metal and CUDA. To support opportunistic layerwise grouping at runtime, we modify its graph builder to generate ggml computation graphs for individual jobs and retain intermediate states across jobs. The jobs in a group run as one batch in a single graph. We reuse compatible job graphs and captured CUDA graphs to reduce graph construction and launch overhead. Bevy uses LFU eviction rather than a recency-based policy, since recency alone can be a poor guide to expert reuse across layers (Xue et al., 2024). Bevy stores the experts on SSD in the expert-pack layout of SSD-LLaMA (Liang et al., 2026) and fetches them with eight reader threads.

Testbed. We evaluate Bevy on two unified-memory platforms for local LLM inference: an Apple Mac mini with the M6 chip, 24 GB of unified memory, and a 512 GB SSD, and an NVIDIA Jetson AGX Thor Developer Kit with 128 GB of unified memory and a Samsung 9100 PRO 2 TB SSD (PCIe 5.0 ×4).

Models and workloads. We use two MoE models, Qwen3.6-35B-A3B (Qwen Team, 2026) and Gemma4-26B-A4B (Gemma Team, 2026), both quantized to Q4 0. We run two workloads, Chat and Agentic Coding, with requests drawn from ShareGPT (ShareGPT, 2023) and SWE-agent trajectories (Nebius, 2024), respectively.

Baselines. We compare Bevy with three baselines, all implemented in the same engine. llama.cpp (ggml-org, 2023) uses its built-in GPU expert cache with LRU eviction and does not prefetch. SpecMD (Hoang et al., 2026) evicts experts with its Least-Stale policy. Like Mixtral-Offloading (Eliseev & Mazur, 2023), it applies the next layer’s gate to the current layer’s gate input to predict which experts to prefetch. We set it to prefetch each token’s top-k experts with an overfetch factor of 1.5 (12 experts per token). FineMoE (Yu et al., 2026) prefetches the experts predicted by the most similar of the expert maps recorded from past tokens, and evicts the expert with the lowest product of usage frequency and predicted probability. We port FineMoE’s open-source implementation as is, with an expert map store of 4096 maps, the largest the implementation allows, and a prefetch distance of 3 layers. All baselines fetch missing experts from SSD in the same expert-pack layout and with the same eight reader threads as Bevy. Because prefetches share these readers with the fetches of missing experts, FineMoE and SpecMD use at most two of them for prefetching at a time, so that prefetches delay those fetches as little as possible.

Measurements. We report the decode throughput (tok/s) as the main evaluation metric. Each run decodes 512 steps, and we exclude the first 256 steps, which warm up the expert cache. We also exclude steps with observed SSD-read slowdowns from all systems in each comparison. On Thor, we fix the CPU and GPU clocks at their maximum with jetson clocks. On M6, the GPU clock drops whenever the GPU idles in the baselines, slowing them down considerably. To exclude this clock effect from the comparison, the baselines run a dummy kernel whenever the GPU would otherwise idle, which keeps the clock from dropping.

## 5.2 End-to-End Performance

We first compare the decode throughput of Bevy and the baselines as the number of concurrent requests and the expert cache size vary.

## 5.2.1 Number of Concurrent Requests

Figure 8 shows the decode throughput as the number of concurrent requests grows from 2 to 16, with 30% of the experts cached. We measure five concurrency levels for both workloads, both models, and both devices. Bevy achieves the highest throughput in all 40 configurations. It is 1.01– 1.38× faster than the fastest baseline and 1.29–3.46× faster than llama.cpp. The prefetching baselines are faster than llama.cpp, which does not prefetch, in almost all configurations, and the faster of the two is 1.13–2.52× faster than llama.cpp. With four or more concurrent requests, Bevy’s speedup over the fastest baseline is 1.15–1.38× on both devices, for both models and both workloads. The speedup is smallest with two requests in most configurations, down to 1.01× for Gemma4-26B-A4B on Thor, since only one other request can run ahead while a request waits for its experts. Even without prefetching, Bevy is faster than every baseline at every concurrency level.

## 5.2.2 Expert Cache Size

Figure 9 shows the throughput as the expert cache grows from 10% to 70% of all experts, with eight concurrent Chat requests. Bevy achieves the highest throughput at every cache size, 1.07–1.55× faster than the fastest baseline and 1.26–2.83× faster than llama.cpp. As the cache grows, the prefetching baselines approach llama.cpp. FineMoE’s speedup over llama.cpp reaches 1.43–2.06× at 20–30% of the experts and drops to 1.08–1.18× at 70%, as prefetch precision decreases with a larger cache (Figure 2). In contrast, while Bevy is bound by the link, fewer missing experts shorten its link time, and its throughput increases accordingly. In this regime, Bevy’s step time exceeds its link busy time by only 4–9 ms on Thor, up to 40% of the experts for Gemma4-26B-A4B and 60% for Qwen3.6-35B-A3B. Bevy’s speedup over the fastest baseline reaches 1.40– 1.55× at 50–60% of the experts on M6 with both models and on Thor with Qwen3.6-35B-A3B.

![](images/aba8b4334c62bc8f64d5b922ce549b74320a845c7c61c847776195f316213336.jpg)  
Figure 8. Throughput as the number of concurrent requests grows. The numbers above the bars give Bevy’s speedup over the fastest baseline. Even without prefetching, Bevy achieves the highest throughput in every configuration.

![](images/0876309291bbb40f6dab50ec487571089b190dc59e78536abdf3557b2ac7f234.jpg)  
Figure 9. Throughput across expert cache sizes. As the cache grows, prefetch precision drops and the prefetching baselines approach llama.cpp, while Bevy achieves the highest throughput at every cache size.

## 5.3 Where the Speedup Comes From

To understand where Bevy’s speedup comes from, we break down the step time and the expert fetches of each system for Gemma4-26B-A4B with eight Agentic Coding requests and 30% of the experts cached on M6. To exclude the effect of the eviction policy, we replace the eviction policy of each baseline with LFU, which Bevy uses, in this experiment.

![](images/bf17e71db0e8819f82d2e6b9033557e2ce5e4e602a0592fbded2adbdf9c1242c.jpg)  
Figure 10. (a) Step time split by whether the link and the GPU are busy or idle. (b) Expert fetches per step: missing experts fetched on demand, and prefetched experts that the step needed or wasted. Bevy keeps the link and the GPU busy together for most of the step while fetching only the experts that the requests need.

Figure 10(a) shows how much computation and fetching overlap. In llama.cpp + LFU, the link and the GPU are never busy at the same time, and the GPU is idle for 60% of each step while the link fetches. FineMoE and SpecMD overlap fetching with computation for 36–40% of the step, but the GPU still waits for the link for 57–59% of the step. Bevy keeps the link and the GPU busy together for 77% of the step, and the GPU waits for the link for only 10%. Although Bevy’s GPU busy time is 1.6–1.7× that of the baselines, Bevy overlaps 86% of it with fetching, so its step time is

![](images/9d761b38fda40d966c5d545bc9fe92626c463697a027ae219868abcb73b3f590.jpg)  
Figure 11. Mean step time against the number of missing experts per step, for static groupings and Bevy’s opportunistic layerwise grouping. Which static grouping is fastest depends on the number of missing experts, while Bevy achieves the lowest step time at every point by adapting groups to the missing experts at each layer.

## shorter.

Figure 10(b) shows that the prefetching baselines fetch many experts that are not needed. FineMoE and SpecMD fetch 1.4–1.5× as many experts per step as llama.cpp + LFU, and 51–64% of the experts they prefetch go unused. This agrees with the simulation in §2.4 under the same LFU policy. Bevy fetches 235 experts per step, almost the same as llama.cpp + LFU. Together, these results answer the question raised in §2.4: Bevy reduces idle link time through request scheduling while fetching only experts known to be needed.

## 5.4 Ablation Study

We next evaluate Bevy’s design components individually.

## 5.4.1 Opportunistic Layerwise Grouping

Using the same setting as Figure 4 with six concurrent requests, we compare Bevy’s opportunistic layerwise grouping with static groupings, which split the requests into fixed groups in advance and schedule each group as one unit: 3:3, 2:2:2, 2:2:1:1, 2:1:1:1:1, and 1:1:1:1:1:1. Figure 11 shows that which static grouping is fastest depends on the number of missing experts. Bevy achieves the lowest step time at every point, 0.9–4.3 ms below the fastest static grouping at that point, because it dynamically adapts the grouping granularity to the missing experts at each layer. This result shows that the self-balancing property described in §4.3 works effectively.

## 5.4.2 Lowest-Layer-First Scheduling

To evaluate the lowest-layer-first compute and fetch orders of §4.2, we replace each of them in Bevy with an order that ignores request progress, while keeping the grouping rule of §4.3. For computation, FIFO over ready jobs selects the layer of the oldest ready job instead of the lowest layer. For fetching, global SRTF fetches the missing experts of the job with the fewest remaining missing experts, regardless of its layer. We also replace both orders. We use the same setting as Figure 10.

![](images/1dd7dd7495d5fe55fd475b17b289651795e604f112d2f473903fec07b47d8e67.jpg)  
Figure 12. (a) Change in GPU busy time, GPU idle time, and step time relative to Bevy when the fetch order, the compute order, or both are changed. (b) Share of groups by group size, the number of jobs that run together as one batch. Lowest-layer-first compute ordering creates batching opportunities: it forms larger groups, and the GPU busy time it saves outweighs the increase in GPU idle time.

Figure 12 shows that the compute order has the larger effect. With FIFO, the step is 12.3% longer because the jobs run in smaller groups, which repeats reads of shared weights and increases compute time (§3.2). The share of groups with a single job rises from 56% to 67%, and the share of groups with three or more jobs falls from 23% to 12%. The GPU busy time grows by 19.2 ms, while the GPU idle time shrinks by 2.0 ms. The fetch order has a smaller effect: with global SRTF, the step is 1.7% longer, as the share of groups with a single job rises to 60% and the GPU busy time grows by 3.7 ms. Replacing both orders makes the step 13.6% longer. In all three cases, the GPU idle time does not increase, so the steps are longer because of additional computation rather than waiting. This confirms that lowest-layer-first effectively creates batching opportunities, as described in §4.2.

## 5.5 Combining Bevy with Prefetching

Bevy fetches experts only on demand, and prefetching can also be applied on top of it. To evaluate the effect of combining Bevy with prefetching, we apply the prediction method of Mixtral-Offloading (Eliseev & Mazur, 2023) to Bevy (Bevy + Prefetch). While job l of a request runs on the GPU and the link is idle, Bevy fetches the request’s predicted experts of layer l+1 that are not in the cache. The prefetch mechanism is the same as in the baselines (§5.1). We use Qwen3.6-35B-A3B with Agentic Coding requests and 30% of the experts cached on M6, and vary the number of concurrent requests from 2 to 16.

![](images/a0269bd66394455e7bc8bf6f7ef0d9689a97e0ed53f458679f15aa1eb351297f.jpg)

![](images/65b46e034d8f44375285d00f78ebbfad91e55655761404752b2691b0d40e8f88.jpg)  
Figure 13. (a) Throughput of Bevy and Bevy + Prefetch as the number of concurrent requests grows. (b) Expert prefetches per step, split into needed and wasted. Prefetching does not improve the throughput of Bevy: with few requests, many prefetches are wasted, and with more requests, the link has no idle time left for prefetching.

Figure 13(a) shows that prefetching does not improve the throughput of Bevy. With two and four requests, Bevy + Prefetch has 4.2% and 5.4% lower throughput. There are two reasons. First, even when a prefetched expert is needed, it is almost always still being fetched when the next job needs it, so it saves little time. Second, as Figure 13(b) shows, 46% and 39% of the prefetched experts are wasted, and these wasted fetches only delay the fetches of missing experts. From eight requests on, missing experts wait to be fetched for almost the whole step, so the link has no idle time left for prefetching. As a result, Bevy + Prefetch rarely prefetches and performs the same as Bevy. These results show that adding prefetching with the prediction method of Mixtral-Offloading does not improve the throughput of Bevy.

## 6 RELATED WORK

Overlapping computation with expert fetches. Expert offloading systems such as Klotski (Fang et al., 2025) and ExpertFlow (Shen et al., 2025) compute the experts already in GPU memory first while the missing experts are fetched. However, all requests in a batch still move to the next layer together, only after all the experts the batch needs at the current layer are fetched. To remove this barrier, KTransformers (Chen et al., 2025) uses Expert Deferral, which defers the computation of some experts and proceeds to the next layer. This, however, is an approximation that changes the model output. Bevy instead lets different requests run at different layers to overlap computation with expert fetches, and does not change the model output.

Request batching for expert sharing. To reduce the expert fetches of a batch, other systems choose which requests to batch together so that the requests in a batch activate many of the same experts (He et al., 2024; Zhuang et al., 2026).

Bevy addresses a different question, how to schedule those requests within a single step to overlap computation with expert fetches. Since the two approaches are orthogonal, Bevy can schedule the requests that these systems batch together.

Expert parallelism in large-scale MoE serving. In largescale server inference, expert parallelism distributes the experts across many GPUs that exchange tokens with each other. To hide this communication behind computation, prior systems split a batch into micro-batches and pipeline GPU computation with communication (Zhu et al., 2025b; DeepSeek-AI, 2024). Bevy differs from them in both its goal and its method. Within the limited memory of a local device, its goal is to keep the GPU from stalling on misses while making full use of the link bandwidth. Rather than splitting a batch into micro-batches, Bevy groups jobs dynamically according to the progress of each request (§4.3).

CPU–GPU hybrid systems. In local MoE inference, where not all experts fit in GPU memory, many systems keep the experts in CPU memory and use the compute resources of the CPU alongside the GPU (Chen et al., 2025; Yang et al., 2026; Wang et al., 2026; Kamahori et al., 2025). These systems, however, cannot be applied to devices with a unified memory architecture, such as Apple silicon and other systems-on-chip (SoCs) with integrated GPUs, which are common for running LLMs locally. Bevy is implemented on unified memory, but it also applies to discrete GPUs by offloading the experts to CPU memory instead of SSD. Extending Bevy to use CPU computation as well is possible, and we leave it for future work.

## 7 CONCLUSION

Our analysis showed that expert caching reduces prefetch precision in local MoE inference, wasting bandwidth on unneeded experts. Bevy uses runahead inference to let ready requests reach later layers and reveal actual expert demand early. It fetches the resulting missing experts while computing other requests, without predicting expert selections. Its self-balancing scheduler batches ready requests by layer and prioritizes lagging ones, preserving shared-weight reuse. Across two local platforms, Bevy achieved 1.26–3.46× the decode throughput of llama.cpp and 1.01–1.55× that of the fastest prefetching baseline.

## REFERENCES

Apple. Apple unveils a more powerful Mac mini featuring the all-new M6 and M5 Pro, 2026. URL https:// www.apple.com/newsroom/2026/08/apple -unveils-a-more-powerful-mac-mini-fea turing-the-all-new-m6-and-m5-pro/.

Chen, H., Xie, W., Zhang, B., Tang, J., Wang, J., Dong, J., Chen, S., Yuan, Z., Lin, C., Qiu, C., Zhu, Y., Ou, Q., Liao, J., Chen, X., Ai, Z., Wu, Y., and Zhang, M. KTransformers: Unleashing the full potential of CPU/GPU hybrid inference for MoE models. In Proceedings ofthe ACM SIGOPS 31st Symposium on Operating Systems Principles, pp. 1014–1029, 2025.

DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Delgado, G. Sparks fly: NVIDIA accelerates local AI at IFA 2026, 2026. URL https://blogs.nvidia.com /blog/local-ai-ifa-next-gen-agents-n v-pair-rtx-spark/.

Eliseev, A. and Mazur, D. Fast inference of mixture-ofexperts language models with offloading. arXiv preprint arXiv:2312.17238, 2023.

Fang, Z., Huang, Y., Hong, Z., Lyu, Y., Chen, W., Yu, Y., Yu, F., and Zheng, Z. Klotski: Efficient mixture-of-expert inference via expert-aware multi-batch pipeline. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 574–588, 2025.

Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

ggml-org. llama.cpp: LLM inference in C/C++, 2023. URL https://github.com/ggml-org/llama.c pp.

He, X., Zhang, S., Tang, K., Shi, S., Wang, Y., Zeng, Z., Tang, Z., Chu, X., Yin, H., Tsang, I. W., and Ong, Y. S. ExpertFlow: Efficient mixture-of-experts inference via predictive expert caching and token scheduling. arXiv preprint arXiv:2410.17954, 2024.

Hoang, D. N. M., Samragh, M., Jaiswal, A. K., and Cho, M. SpecMD: A comprehensive study on speculative expert prefetching. In Proceedings of the 43rd International Conference on Machine Learning, pp. 43336– 43350, 2026.

Hu, T. C. Parallel sequencing and assembly line problems. Operations Research, 9:841–848, 1961.

Kamahori, K., Tang, T., Gu, Y., Zhu, K., and Kasikci, B. Fiddler: CPU-GPU orchestration for fast inference of mixture-of-experts models. In International Conference on Learning Representations, pp. 56099–56115, 2025.

Liang, F., Shen, Y., Hu, J., Xu, J., Gao, H., Xu, M., and Meng, Z. SSD-LLaMA: SSD-native inference for trillionparameter MoE at 1+ token/s on a consumer PC. arXiv preprint arXiv:2609.18110, 2026.

LM Studio Team. Introducing LM Studio 0.4.0, 2026. URL https://lmstudio.ai/blog/0.4.0.

Nebius. SWE-agent-trajectories, 2024. URL https:// huggingface.co/datasets/nebius/SWE-a gent-trajectories.

NVIDIA. NVIDIA RTX Spark, 2026. URL https://ww w.nvidia.com/en-us/products/rtx-spark /.

OpenClaw. Sub-agents, 2026. URL https://docs.o penclaw.ai/tools/subagents.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, 2026. URL https://qwen.ai/bl og?id=qwen3.6-35b-a3b.

ShareGPT. ShareGPT Vicuna unfiltered, 2023. URL ht tps://huggingface.co/datasets/anon82 31489123/ShareGPT\_Vicuna\_unfiltered.

Shen, Z., Chu, K., Zhang, Y., Xiang, D., Wu, R., and Zhang, W. ExpertFlow: Adaptive expert scheduling and memory coordination for efficient MoE inference. arXiv preprint arXiv:2510.26730, 2025.

Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B., Liang, P., Re, C., Stoica, I., and Zhang, C. FlexGen:´ High-throughput generative inference of large language models with a single GPU. In Proceedings ofthe 40th International Conference on Machine Learning, pp. 31094– 31116, 2023.

Song, X., Zhong, Z., Chen, R., and Chen, H. ProMoE: Fast MoE-based LLM serving using proactive caching. arXiv preprint arXiv:2410.22134, 2024.

Tang, P., Liu, J., Hou, X., Pu, Y., Wang, J., Heng, P.-A., Li, C., and Guo, M. MoE-APEX: An efficient MoE inference system with adaptive precision expert offloading. In Proceedings ofthe 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 1185–1200, 2026.

Wang, W., Hou, Y., Ji, Y., Qu, P., and Zhang, Y. Achieving cloud-grade SLOs for local mixture-of-experts inference through CPU–GPU hybrid design. In Proceedings ofthe 20th USENIX Symposium on Operating Systems Design and Implementation, pp. 1089–1106, 2026.

Xue, L., Fu, Y., Lu, Z., Mai, L., and Marina, M. MoE-Infinity: Efficient MoE inference on personal machines with sparsity-aware expert cache. arXiv preprint arXiv:2401.14361, 2024.

Yang, S., Fan, X., Pan, M., Xi, H., Wang, Z., Sun, S., Keutzer, K., Han, S., Zaharia, M., Xu, C., and Stoica, I. FreeToken: Efficient edge-native MoE serving with bandwidth-adaptive execution. arXiv preprint arXiv:2608.16157, 2026.

Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.- G. Orca: A distributed serving system for Transformerbased generative models. In Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation, pp. 521–538, 2022.

Yu, H., Cui, X., Zhang, H., Wang, H., and Wang, H. Taming latency-memory trade-off in MoE-based LLM serving via fine-grained expert offloading. In Proceedings of the 21st European Conference on Computer Systems, pp. 176–191, 2026.

Zhu, K., Gao, Y., Zhao, Y., Zhao, L., Zuo, G., Gu, Y., Xie, D., Tang, T., Xu, Q., Ye, Z., Kamahori, K., Lin, C.-Y., Wang, Z., Wang, S., Krishnamurthy, A., and Kasikci, B. NanoFlow: Towards optimal large language model serving throughput. In Proceedings of the 19th USENIX Symposium on Operating Systems Design and Implementation, pp. 749–765, 2025a.

Zhu, R., Jiang, Z., Jin, C., Wu, P., Stuardo, C. A., Wang, D., Zhang, X., Zhou, H., Wei, H., Cheng, Y., Xiao, J., Zhang, X., Liu, L., Lin, H., Chang, L.-W., Ye, J., Yu, X., Liu, X., Jin, X., and Liu, X. MegaScale-Infer: Efficient mixture-of-experts model serving with disaggregated expert parallelism. In Proceedings ofthe ACM SIGCOMM 2025 Conference, pp. 592–608, 2025b.

Zhuang, Y., Gong, C., Zheng, Z., Lei, J., Huang, X., Wu, J., Wu, F., and Chen, G. Sparsity-aware scheduling for efficient MoE-based LLM serving on edge servers. In Proceedings ofthe 32nd Annual International Conference on Mobile Computing and Networking, pp. 867–884, 2026.