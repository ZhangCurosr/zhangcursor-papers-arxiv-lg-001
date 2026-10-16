# Architecting for Hardware Diversity: Meta’s End-to-End Co-Design for High-Performance AI Training on AMD GPUs

Haoyu Zhang<sup>∗</sup>, Arshad Rahman<sup>†</sup>, Adnan Aziz<sup>∗</sup>, Ajith Prabhudev<sup>∗</sup>, Amin Farmahini<sup>∗</sup>, Ben Rhoads<sup>∗</sup>, Benson Ma<sup>∗</sup>, Brian Shi<sup>†</sup>, Cen Zhao<sup>∗</sup>,   
CK Luk<sup>∗</sup>, Dingming Wu<sup>∗</sup>, Emad El-Haraty<sup>∗</sup>, Eric Chen<sup>∗</sup>, Frank Li<sup>∗</sup>, Gary Huang<sup>∗</sup>, Gefei Zuo<sup>∗</sup>, Haoran Zhang<sup>∗</sup>, Hiwot Kassa<sup>∗</sup>, Hongyi Zeng<sup>∗</sup>,   
Jack Taylor<sup>†</sup>, Jeremy Hadidjojo<sup>∗</sup>, John Donati<sup>∗</sup>, Jonathan Sheffield<sup>∗</sup>, Julio Machado Silva<sup>†</sup>, Karthik Ganesan<sup>∗</sup>, Kent Russell<sup>†</sup>, KR Kishore<sup>∗</sup>, Lei Tian<sup>∗</sup>, Liu Ke<sup>∗</sup>, Muhammad Osama<sup>†</sup>, Peng Chen<sup>∗</sup>, Rakesh Hemnani<sup>∗</sup>, Ryan Swann<sup>†</sup>, Saleel Kudchadker<sup>†</sup>, Shankar Yanamandram<sup>∗</sup>, Shobhit Kanaujia<sup>∗</sup>, Shucai Xiao<sup>†</sup>, Srinath Bayareddy<sup>∗</sup>, Srinivas Vaidyanathan<sup>∗</sup>, Sudharssun Subramanian<sup>∗</sup>, Supadchaya Puangpontip<sup>∗</sup>,   
Taylor Robie<sup>∗</sup>, Tianhao Huang<sup>∗</sup>, Tyler Hart<sup>∗</sup>, Viswesh Sankaran<sup>∗</sup>, Weijun Jiang<sup>†</sup>, Wenkai Du<sup>†</sup>, Xiaodong Wang<sup>∗</sup>, Xiaohu Guo<sup>†</sup>, Yao Rong<sup>∗</sup>, Yinbin Ma<sup>∗</sup>, Yue Dong<sup>∗</sup>, Yuzhen Huang<sup>∗</sup>, Yvonne Liu<sup>∗</sup>, Zhan Shu<sup>∗</sup>, Dan Rabinovitsj<sup>∗</sup>, Ajit Mathews<sup>∗</sup>, Chunqiang Tang<sup>∗</sup>

<sup>∗</sup>Meta <sup>†</sup>AMD

Abstract—Meta deploys a heterogeneous fleet comprising NVIDIA GPUs, AMD GPUs, and proprietary MTIA accelerators to reduce costs and mitigate vendor lock-in. However, the extensive history of NVIDIA-specific optimizations hinders performance portability to other hardware. This paper presents our methods for optimizing alternative accelerators, using AMD Instinct™ MI350X versus NVIDIA B200 for recommendation model training as a case study. We propose three reusable principles. First, hardware-model co-design is essential: rather than directly porting models from NVIDIA GPUs, revising model structures and software recipes allows full exploitation of accelerator-specific strengths (e.g., MI350X’s 1.6× HBM capacity relative to B200). Second, our analytical performance model enables zero-warmup optimal kernel selection across diverse datatypes and matrix shapes in production models, drastically simplifying tuning. Third, increasingly mature agentic workflows have lowered the barrier to porting from the CUDA ecosystem; we extend these workflows for microarchitecture-aware kernel optimizations and custom shim generation. Compared to the directly ported baseline, these methods deliver up to 1.5× improvement in end-to-end training throughput. To our knowledge, this is the first hyperscaler report on productionizing AMD GPUs for large-scale training. Our contributions have been upstreamed to the open-source PyTorch, Triton, and ROCm ecosystems.

Index Terms—Accelerators, DLRM, GPU, kernels, machine learning, recommendation systems

## I. INTRODUCTION

The immense potential of AI applications has catalyzed an unprecedented expansion in the accelerator landscape. While NVIDIA remains the incumbent platform, the AMD Instinct™ [1, 2] GPU has emerged as a viable alternative. Meanwhile, hyperscalers have also developed custom silicon, including Meta MTIA [3–6], Google TPU [7], Amazon Trainium [8], Microsoft Maia [9], Huawei Ascend [10], Alibaba Hanguang [11], and Baidu Kunlun [12].

Capitalizing on this hardware proliferation, Meta deploys a heterogeneous fleet comprising NVIDIA GPUs, AMD GPUs, and proprietary MTIA accelerators at scale. This strategic diversification delivers several benefits: (1) enabling cost optimization by preserving the flexibility to adopt the most economical solutions as the market evolves; (2) strengthening supply-chain resilience; and (3) matching model characteristics to the most suitable hardware.

However, the extensive history of NVIDIA-specific optimizations hinders performance portability to other hardware.

For example, fundamental differences in memory hierarchies and execution units (e.g., 32-thread warps vs. 64-thread wavefronts) thwart direct software migration between NVIDIA and AMD GPUs. These challenges are most acute in deep learning recommendation models (DLRMs) [13, 14]. Unlike the relatively uniform large language models (LLMs), DLRMs are characterized by diverse model architectures, fragmented compute kernel sets, irregular matrix dimensions, and complex memory access patterns.

While industry adoption of AMD GPUs has largely focused on inference—evidenced by Meta’s deployment of hundreds of thousands of AMD GPUs—distributed training presents a far greater technical challenge. This paper delivers the first hyperscaler report of enabling AMD GPUs for productionscale DLRM training. We demonstrate that through systematic optimization, AMD MI350X achieves performance comparable to the incumbent NVIDIA B200 GPU across a diverse suite of production DLRMs. We distill our production experience into three reusable principles for new hardware integration:

Hardware-model co-design. Different accelerator microarchitectures offer distinct advantages and trade-offs. Rather than directly porting models from NVIDIA GPUs, hardware-model co-design—revising the model structure and software recipe—is essential to leverage an accelerator’s specific strengths. Specifically, we exploit MI350X’s 1.6× HBM capacity relative to B200. To maximize training efficiency, we expand per-GPU batch sizes and reduce activation checkpointing while simultaneously optimizing data or model parallelism and implementing efficient embedding re-sharding. Beyond training throughput, we utilize the additional HBM to enhance model quality by enabling longer input sequences, larger hidden dimensions, and expanded embedding tables. This holistic approach delivers a 20% training speedup and a >0.1% improvement in normalized entropy (NE) [15], demonstrating that architectural synergies can simultaneously drive improvements in both hardware efficiency and model quality.

Analytical performance model. Traditionally, selecting high performance general matrix multiplication (GEMM) kernels relies on offline tuning—a cumbersome process that generalizes poorly across the disparate matrix dimensions common in DLRMs. We propose Origami, an analytical performance model that predicts optimal tile configurations in microseconds, achieving 93% accuracy relative to exhaustively autotuned kernels across diverse matrix shapes. While analytical models are not new, Origami extends prior approaches to multi-chiplet architectures, with accuracy validated at production scale.

Agentic workflows for kernel optimization. To reduce engineering burden and migration friction, we codify complex kernel-tuning principles into agentic workflows: automating microarchitecture-aware configuration selection, swizzling process IDs, incorporating occupancy hints, and leveraging specialized hardware instructions. Together, they achieve 1.6×– 2× speedups across both hand-written and compiler-generated kernels. Furthermore, we leverage agentic workflows to automate the application of solutions that address hardware-specific bottlenecks; e.g., to mitigate the performance impact of atomics on MI350X, we apply a tree-reduction algorithm that utilizes shared memory to deliver up to 4× speedup in custom kernels.

Our methods have been validated at scale, supporting produc tion training on tens of thousands of MI350X GPUs at Meta. Across a representative set of production models, MI350X achieves a geometric-mean throughput of 101% relative to our established B200 baseline—a platform highly optimized through extensive deployment. This performance parity validates the viability of our multi-vendor diversification strategy. Limitations. While end-to-end performance is comparable, MI350X’s GEMM performance remains an area for improvement, due to both the overhead of emulated TF32 GEMMs in the absence of native hardware support and the relative maturity of NVIDIA cuBLAS over AMD hipBLAS in handling irregular matrix shapes.

Although we use DLRMs to drive our optimizations, the underlying principles are generalizable to a broader spectrum of ML workloads, including LLMs. Indeed, we have successfully optimized LLM training on clusters of more than 2,000 MI350X GPUs; a comprehensive analysis of LLM-specific performance is reserved for future work.

To our knowledge, this is the first hyperscaler report on production-scale training using AMD GPUs. Our purpose is not to rank accelerators, but to share generalizable optimization methods for emerging hardware by synthesizing insights from our large-scale deployment of AMD, NVIDIA, and MTIA platforms. Our contributions have been upstreamed to the open source PyTorch, Triton, and ROCm ecosystems.

## II. BACKGROUND

## A. DLRM Evolution

Deep learning recommendation models (DLRMs) [13] have become the canonical architecture for industrial recommendation and personalization systems across major technology companies, such as Meta [16], ByteDance [17], Google [18], and Alibaba [19]. DLRMs often comprise (1) a sparse component that uses embedding tables to map high-cardinality categorical signals—such as user IDs, item IDs, geographic regions, and language preferences—into dense vectors, and (2) a dense component that learns feature interactions to predict user engagement (e.g., click-through or conversion rates). The importance of this model family is reflected both externally—through the MLPerf recommendation benchmark evolution and the adoption of DLRM [20], DLRM-v2 [16], and DLRM-v3 [14, 21]—and internally through the scale of Meta’s production deployment and infrastructure investment dedicated to improving DLRM efficiency.

![](images/b818950a7d2f5a7fafa7b1f8c890ec030bb90941ad2c3da90a84a39addcc50ba.jpg)  
Fig. 1. Total training compute for deep learning models.

Sparse evolution. Embedding tables grow rapidly in both vocabulary size and dimensionality, forcing distributed sharding and collective-heavy embedding exchange. As training clusters expanded, two practical directions emerged. First, multi-stage pipelining became essential [22]: embedding lookup, all-to-all exchange, and optimizer steps must be organized to reduce exposed latency and avoid synchronization bubbles. Second, sparse parallelism itself evolved beyond single dimension. Recent work on two-dimensional sparse parallelism [23] introduces a hierarchical strategy that adds data-parallel structure on top of model-parallel embedding sharding, reducing allto-all scope, lowering embedding activation memory growth, and mitigating imbalance/straggler behavior. Together, these trends reflect a broader shift: the sparse subsystem is no longer “just embeddings,” but a distributed system whose parallelism strategy, pipeline structure, and optimizer semantics jointly determine scalability.

Dense evolution. Classical DLRMs pair large embedding tables with relatively shallow dense components (e.g., a few fully connected layers for feature interaction). Recent evolution of DLRMs (Fig. 1) shows that scaling the dense subsystem can yield predictable quality gains analogous to LLM scaling laws [24, 25]. Wukong [26] demonstrates this by stacking factorization-machine layers with a synergistic upscaling strategy, achieving smooth quality improvements over orders of magnitude in dense-side FLOPs. A complementary trend is the adoption of Transformer-like modules to model long user-interaction histories. Meta’s HSTU [21] exemplifies this sequence-centric direction, replacing shallow pooling with attention over hundreds of past events. Together, these advances open a second axis for improving recommendation quality: in addition to scaling embedding tables, practitioners can now scale dense computation (FLOPs) to capture richer feature interactions and temporal patterns.

<table><tr><td>Specification</td><td>MI350X</td><td>MI355X</td><td>B200</td></tr><tr><td>Architecture</td><td>CDNA 4</td><td>CDNA 4</td><td>Blackwell</td></tr><tr><td>Compute Units / Streaming Multiprocessors</td><td>256 (8 XCD × 32)</td><td>256 (8 XCD × 32)</td><td>148 (2 chiplets)</td></tr><tr><td>Peak FP16/BF16 Dense Performance</td><td>2300 TFLOPS</td><td>2500 TFLOPS</td><td>2250 TFLOPS</td></tr><tr><td>HBM Type / Capacity</td><td>HBM3e / 288 GB</td><td>HBM3e / 288 GB</td><td>HBM3e / 180 GB</td></tr><tr><td>HBM Bandwidth</td><td>Up to 8 TB/s</td><td>Up to 8 TB/s</td><td>Up to 8 TB/s</td></tr><tr><td>LDS per CU or SM</td><td>160 KB</td><td>160 KB</td><td>256 KB</td></tr><tr><td>Scale Up BW per GPU (uni-dir)</td><td>538 GB/s</td><td>538 GB/s</td><td>900 GB/s</td></tr><tr><td>Cooling</td><td>Air-cooled</td><td>Direct Liquid Cooled</td><td>Air-cooled</td></tr><tr><td>Deployed Power Limit</td><td>1000 W</td><td>1400 W</td><td>750 W</td></tr><tr><td>Deployment</td><td>Meta On-Prem</td><td>Public Cloud</td><td>Meta On-Prem</td></tr></table>

Table I. Comparison of AMD Instinct MI350X, MI355X, and NVIDIA Blackwell B200 GPUs. MI355X is a 1400 W design using direct liquid cooling, while Meta uses the 1000 W air-cooled MI350X. Meta’s deployment of B200 operates at 750 W, which is different from the general B200 HGX solution that operates at 1000 W. Meta operates both GPUs below their vendor-specified TDPs on the same air-cooled platform. B200 is capped at a lower power limit to enable air cooling because its power density is higher than that of MI350X.

![](images/4caeb45ee2d558b35b2da8c711c3e1437345ddbd5f2fd61f409f41a18fe417ff.jpg)  
Fig. 2. Hipification process to enable ML jobs on AMD GPUs.

## B. AMD MI350X vs NVIDIA B200

Table I compares the GPU specs. AMD MI350X is built on the CDNA™ 4 architecture and is the first AMD GPU used for production-scale training at Meta. MI350X features eight accelerator complex dies (XCDs) with 32 compute units (CUs) each (a total of 256 CUs), a 288 GB HBM3e memory pool with up to 8 TB/s bandwidth, and 538 GB/s of scale-up bandwidth per GPU. B200 offers 180 GB of HBM capacity, with 900 GB/s of scale-up bandwidth per GPU.

B200 and MI350X are same-generation flagships with closely matching theoretical bounds: their near-identical peak compute (2250 vs. 2300 BF16 TFLOPS) and HBM bandwidth (8 TB/s) yield largely coincident roofline curves [27], and thus a similar performance envelope for ML workloads. Their strengths are complementary: B200 leads on per-GPU scale-up bandwidth (900 vs. 538 GB/s), an all-to-all NVSwitch fabric (versus AMD’s point-to-point xGMI mesh), shared memory per SM (256 vs. 160 KB), and native TF32 support, whereas MI350X’s principal advantages are its 1.6× HBM capacity (288 vs. 180 GB) and higher deployed power.

Hipification. Hipification is AMD’s CUDA-to-HIP (Heterogeneous-compute Interface for Portability) translation process that converts NVIDIA CUDA code into HIP equivalents; see Fig. 2. There are two primary approaches: (1) compile-time hipification using AMD’s open-source hipify-clang compiler, which translates CUDA constructs to HIP at build time, and (2) rule-based source-code translation using hipify\_torch, which traverses and rewrites source files before compilation. While hipification provides broad functional compatibility and enables most CUDA workloads to run on AMD with minimal manual effort, performance portability remains a challenge. For instance, CUDA uses a warp size of 32, whereas an AMD GPU uses a wavefront size of 64. Kernels that naively hardcode warp sizes will likely perform suboptimally without tuning.

The emergence of agentic AI tools has fundamentally lowered the cost of switching from NVIDIA to AMD. At Meta, we leverage AI agents to address issues in the hipification process.

These agents iteratively compile code, interpret error messages, and apply fixes until the build succeeds. When a niche CUDA library lacks a ROCm equivalent, AI agents can even generate custom shims to mimic the missing functionality. This makes AMD GPUs accessible to a broad range of ML researchers, who can now deploy high-performance code without mastering low-level hardware nuances.

Meta’s Usage of AMD GPUs. Meta started using the AMD MI300X [28] for inference in 2024. Today, models compatible with AMD GPUs make up over 60% of production model types, weighted by resource footprint. Training DLRMs, on the other hand, presents a greater challenge than inference:

• Synchronization and network pressure: Training involves numerous GPUs in tight lockstep with frequent synchronization, heavily straining network efficiency.

• Kernel and optimizer: Training requires comprehensive backward kernels and advanced optimizer support.

• Generalization: While inference can be optimized per workload, training must rely on generalized optimizations for out-of-the-box performance—orders of magnitude more training trials run before a model is finalized, making per-job optimization infeasible.

Addressing these challenges requires systematic optimizations across the software stack, from low-level libraries (Section III) and kernel tuning (Section IV) to high-level model co-design (Section V).

## III. ROOFLINE-DRIVEN OPTIMIZATION OF ROCM LIBRARIES

The ROCm software platform provides optimized libraries, runtimes, and firmware for AMD GPUs. Grounded in roofline models [27] and hardware constraints, we optimize core GEMM kernels and selection algorithms to approach theoretical limits. To optimize RCCL, the collective communication library for distributed training, we derive theoretical bounds from hardware specifications as optimization targets. These roofline-driven principles enable low-level libraries to deliver robust, generalizable performance across diverse AI workloads.

## A. GEMM Kernels for AMD MI350X

DLRM workloads present a broad range of matrix shapes and sizes, including irregular dimensions that stress standard GEMM kernel selection strategies. To deliver strong performance across this diversity, we integrate performance enhancements into hipBLASLt [29] GEMM kernels through its assembly backend, TensileLite [30]. These enhancements leverage new CDNA 4 [31] hardware capabilities alongside algorithmic refinements that reduce the solution space and enable finergrained runtime parameter selection via Origami (Section III-B). Beyond exploiting new ISA features, we employ the following algorithmic techniques.

Stream-K partitions GEMM computations across CUs at finer granularity than traditional data-parallel tiling, improving load balance for tensor sizes that do not evenly tile the device. We also introduce new reduction strategies and runtime dispatch logic, improving robustness across a broad range of computeand memory-bound tensor sizes. Notably, a single GEMM kernel can now achieve performance previously requiring multiple specialized kernels.

Workgroup mapping via space-filling curves. MI350X GPUs use a multi-XCD architecture in which workgroups are distributed across XCDs, each with a private L2 cache. Scheduling workgroups operating on adjacent or overlapping tiles across different XCDs degrades cache locality and efficiency. To address this, hipBLASLt introduces a mapping strategy based on space-filling curves that preferentially co-locates workgroups processing nearby tiles on the same XCD, improving spatial locality, cache utilization, and power efficiency. Multiple mapping strategies are supported within the GEMM assembly kernels, with runtime-selectable parameters integrated with Origami to select the optimal strategy for a given tensor size, kernel configuration, and target hardware.

## B. Origami: Analytical Kernel Selection

Prior kernel selection approaches in hipBLASLt relied on offline autotuning: enumerating kernel configurations on repre sentative tensor shapes and storing the best selections in lookup tables. This approach fails at scale: recommendation models execute thousands of unique GEMM shapes per model, and maintaining exhaustive databases for every shape is impractical.

Origami replaces offline autotuning with an analytical performance model that predicts optimal tile configurations in microseconds. The model can be viewed as inferring the critical path of a GEMM problem across the memory and compute hierarchy. It captures the full tiling hierarchy—from matrix instruction dimensions through local data share (LDS) tiles to implicit cache tiles formed by workgroups sharing the L2 and last level cache (LLC)—and computes cycle counts geometrically at each level.

Performance Model. We consider a GEMM operation $C =$ $A \times B ,$ where $A \in \mathbb { R } ^ { M \times K } , B \in \mathbb { R } ^ { K \times N }$ , and $C \in \mathbb { R } ^ { M \times N }$ . To efficiently execute this operation on the GPU, the output matrix C is partitioned into tiles, each computed by a workgroup. A workgroup iterates along the K dimension, loading tiles of A and B, performing multiply-accumulate operations, and writing the result back to global memory. The goal of the performance model is to select the tile configuration $( M _ { T } , N _ { T } , K _ { T } )$ that minimizes total execution latency.

For each candidate tile configuration with LDS tile size $( M _ { T } , N _ { T } , K _ { T } )$ , Origami computes cycle counts for compute and memory operations, then selects the configuration minimizing total latency. The compute latency for one tile is:

$$
L _ { \mathrm { c o m p u t e } } = N _ { \mathrm { M I } } \cdot L _ { \mathrm { M I } }\tag{1}
$$

where $N _ { \mathrm { M I } } = \lceil M _ { T } / M _ { I } \rceil \cdot \lceil N _ { T } / N _ { I } \rceil \cdot \lceil K _ { T } / K _ { I } \rceil$ is the number of matrix instructions per tile (macro tile dimensions divided by instruction dimensions) and $L _ { \mathrm { M I } }$ is the architecture-specific instruction latency.

Memory latency is computed by cascading traffic through the cache hierarchy. The bytes loaded per tile are:

$$
\mathrm { B y t e s } _ { \mathrm { t i l e } } = M _ { T } \cdot K _ { T } \cdot b _ { A } + N _ { T } \cdot K _ { T } \cdot b _ { B }\tag{2}
$$

where $b _ { A } , b _ { B }$ are bytes per element. Origami predicts cache hit rates by modeling workgroup sharing within each memory scope. For $N _ { \mathrm { C U } }$ CUs sharing a cache, arranged as an $M _ { C } \times N _ { C }$ grid of output tiles, the predicted hit rate is:

$$
H = 1 - \frac { M _ { C } \cdot M _ { T } + N _ { C } \cdot N _ { T } } { M _ { C } \cdot N _ { C } \cdot ( M _ { T } + N _ { T } ) }\tag{3}
$$

This formula applies to L2 and LLC, with $N _ { \mathrm { C U } }$ set to the number of CUs sharing that cache scope. It captures data reuse: A tiles are shared across $N _ { C }$ workgroups computing different output columns, while B tiles are shared across $M _ { C }$ workgroups computing different rows. Tiles are rounded to the nearest cache line width in the contiguous dimension of the allocation (determined by transpose). The model constrains cache tile dimensions when working sets exceed cache capacity.

Traffic reaching each memory level determines hierarchy latency. Bytes cascade through miss rates: $\mathrm { \Delta B y t e s } _ { \mathrm { L L C } } = ( 1 -$ $H _ { \mathrm { L } 2 } ) \cdot \mathrm { B y t e s } _ { \mathrm { t o t a l } }$ and $\mathrm { B y t e s } _ { \mathrm { H B M } } = ( 1 - H _ { \mathrm { L L C } } ) \cdot \mathrm { B y t e s } _ { \mathrm { L L C } } ,$ , where each H is computed using the CU count for that scope. Memory latency in cycles is:

$$
L _ { \mathrm { m e m } } = \operatorname* { m a x } \left( \frac { \mathrm { B y t e s } _ { \mathrm { L 2 } } } { R _ { \mathrm { L 2 } } } , \frac { \mathrm { B y t e s } _ { \mathrm { L L C } } } { R _ { \mathrm { L L C } } } , \frac { \mathrm { B y t e s } _ { \mathrm { H B M } } } { R _ { \mathrm { H B M } } } \right)\tag{4}
$$

where R is bytes per compute cycle at each level. Expressing memory bandwidth in bytes per compute cycle keeps the memory and compute latencies in the same unit, so the model need not reconcile the GPU’s separate memory and compute clock frequencies.

Issue Rate Bound. Effective memory throughput can be limited by the rate at which CUs issue loads and stores, not just bandwidth. This creates another parallelism-locality tradeoff on what are usually considered bandwidth-bound problems: smaller tiles increase the number of vector lanes issuing load/store requests but reduce data reuse, while larger tiles improve locality at the cost of issue rate. Origami models achievable throughput as a function of occupied vector lanes to navigate this tradeoff. Latency Model. Fig. 3 illustrates the pipeline structure: the prologue issues the initial load, the steady-state K-loop overlaps loads with compute, and the epilogue completes the final compute and store. Per-tile latency combines these phases:

$$
L _ { \mathrm { t i l e } } = L _ { \mathrm { p r e f e t c h } } + \left\lceil K / K _ { T } \right\rceil \cdot \operatorname* { m a x } ( L _ { \mathrm { c o m p u t e } } , L _ { \mathrm { m e m } } ) + L _ { \mathrm { s t o r e } }\tag{5}
$$

![](images/a94c8c4cada509bee2a014d19654174371df24dafcce92cfeca7d6ca32d49cf6.jpg)  
Fig. 3. Pipelined GEMM execution. Loads/stores use vector units; compute uses matrix units. The K-loop overlaps load and compute.

![](images/b4773528bfbf90df96388253394409d0fc6621ab63391d45495155e4763beaab.jpg)  
Fig. 4. Wave quantization tail effect. A 4×5 output (20 tiles) on 8 CUs requires 3 timesteps with 20 mod 8 = 4 underutilized CUs in the final timestep. Origami addresses this via two strategies: tile selection to change the number of output tiles, and Stream-K to partition work along the K dimension.

The number of output tiles $T _ { \mathrm { o u t } } ~ = ~ \lceil M / M _ { T } \rceil ~ \cdot ~ \lceil N / N _ { T } \rceil$ determines timestep count $\omega = \lceil T _ { \mathrm { o u t } } / N _ { \mathrm { C U } } \rceil$ . Total latency is $L _ { \mathrm { t o t a l } } = \omega \cdot L _ { \mathrm { t i l e } }$ . Fig. 4 illustrates the tail effect: when tiles do not evenly divide across CUs, the final timestep underutilizes the GPU.

This static analysis is enabled by the regularity of GEMM workloads: uniform tile shapes and predictable access patterns. Origami evaluates all candidate configurations and selects the one minimizing $L _ { \mathrm { t o t a l } }$ . Each candidate configuration evaluates in nanoseconds; the full search across all candidates completes in microseconds.

Grid Size Selection. Beyond tile configuration, Origami selects the optimal Stream-K grid size. Stream-K partitions work in the K dimension across CUs, and Origami balances three considerations: parallelism to maximize CU utilization, locality to enable data reuse across CUs iterating over distinct K slices, and scheduling to load balance across XCDs and minimize the wave quantization problem at each level of the chiplet hierarchy.

Chiplet-Aware Mapping. MI350X comprises 8 XCDs, each with private L2 cache. Origami extends hierarchical tiling to the chiplet level: a device-scope tile is composed of 8 XCD-scope tiles, each containing workgroups that share L2. XCD-aware workgroup ID swizzling maps spatially adjacent output tiles to the same XCD, improving L2 hit rates. Origami also selects the L2 tile shape dynamically based on the number of CUs per chiplet, arranging the concurrent workgroups into an $M _ { C } \times N _ { C }$ grid to maximize the cache hit rate predicted by Equation 3.

Results. Fig. 5 shows Origami performance on production GEMM shapes from recommendation models. The distribution exhibits classical roofline behavior: memory-bound at low arithmetic intensity and compute-bound at high intensity. Compared to exhaustively autotuned GEMM performance, Origami achieves 93% average selection efficiency across datatypes and transpose configurations. Origami is also integrated into tritonBLAS [32], enabling zero-warmup kernel selection for arbitrary matrix shapes at runtime, eliminating the need for offline autotuning entirely.

![](images/49baba8ecca6e88dc03ca277da029b49c92ceed0763b4affaac2d59c57b67c2b.jpg)  
Fig. 5. GEMM performance on MI350X for production shapes, shown as percentage of peak achievable FLOPS.

![](images/9d6b7ece098644fea01029286095f32026e65c0fd82f36b1f8db9767aa936fbb.jpg)  
Fig. 6. GEMM performance (M, N, K dimensions = 8192) with GPU power excursion across kernel durations and duty ratios on MI350X.

Generality. Although calibrated for MI350X, Origami’s formulation is general: because GEMM has regular tiling and predictable data movement, its cost reduces to a hardwareparameterized roofline. Retargeting another accelerator is therefore largely a matter of substituting a handful of hardware constants (cache sizes, achievable bandwidths, matrix-unit throughput, and chiplet topology) rather than re-deriving the model from scratch.

## C. GPU Power Excursion Boost

The Open Compute Project (OCP) is a collaborative initiative between hyperscale datacenter operators and technology vendors to standardize interfaces and specifications across infrastructure domains, including power delivery. The MI350X accelerator is implemented as a 1000 W open accelerator module (OAM) [33]. While the OAM must comply with the nominal total board power (TBP) over a 1s window, the specification permits transient power excursions over shorter intervals as defined by the OAM’s excursion design power (EDP). These excursions range from 1.1× TBP over 20 ms up to 2.0× TBP over 20 $\mu \mathrm { s }$

Although the OCP specification defines minimum electrical, thermal, and power-delivery capabilities, system designers may provision additional margin. Meta’s AI hardware platform, for example, supports average power levels of up to 1.36× TBP over a 50 ms window for a 1000 W OAM configuration. By dynamically managing transient power excursions, we leverage the thermal headroom of the MI350X to exceed steady-state power constraints. This allows compute-heavy kernels to surpass the nominal roofline limit, a strategy particularly effective for the hybrid execution profiles of DLRMs, where compute-intensive kernels are interleaved with lower-power operations.

<table><tr><td rowspan=12 colspan=1>1 MiB2 MiB4MiBMese ssize8 MiB16 MiB32 MiB64 MiB128 MiB256 MiB512 MiB1 GiB</td><td rowspan=1 colspan=5>8 GPU   16 GPU   32 GPU   64 GPU  128 GPU</td></tr><tr><td rowspan=1 colspan=1>32.2%</td><td rowspan=1 colspan=1>27.6%</td><td rowspan=1 colspan=1>45.5%</td><td rowspan=1 colspan=1>54.0%</td><td rowspan=1 colspan=1>46.2%</td></tr><tr><td rowspan=1 colspan=1>44.9%</td><td rowspan=1 colspan=1>31.1%</td><td rowspan=1 colspan=1>49.5%</td><td rowspan=1 colspan=1>62.9%</td><td rowspan=1 colspan=1>55.1%</td></tr><tr><td rowspan=1 colspan=1>52.5%</td><td rowspan=1 colspan=1>41.8%</td><td rowspan=1 colspan=1>47.5%</td><td rowspan=1 colspan=1>68.6%</td><td rowspan=1 colspan=1>69.5%</td></tr><tr><td rowspan=1 colspan=1>64.3%</td><td rowspan=1 colspan=1>53.9%</td><td rowspan=1 colspan=1>57.4%</td><td rowspan=1 colspan=1>65.2%</td><td rowspan=1 colspan=1>75.1%</td></tr><tr><td rowspan=1 colspan=1>71.0%</td><td rowspan=1 colspan=1>66.4%</td><td rowspan=1 colspan=1>65.6%</td><td rowspan=1 colspan=1>71.6%</td><td rowspan=1 colspan=1>67.0%</td></tr><tr><td rowspan=1 colspan=1>76.4%</td><td rowspan=1 colspan=1>74.1%</td><td rowspan=1 colspan=1>74.4%</td><td rowspan=1 colspan=1>78.5%</td><td rowspan=1 colspan=1>73.8%</td></tr><tr><td rowspan=1 colspan=1>85.6%</td><td rowspan=1 colspan=1>80.4%</td><td rowspan=1 colspan=1>78.4%</td><td rowspan=1 colspan=1>81.5%</td><td rowspan=1 colspan=1>80.8%</td></tr><tr><td rowspan=1 colspan=1>92.0%</td><td rowspan=1 colspan=1>84.3%</td><td rowspan=1 colspan=1>81.9%</td><td rowspan=1 colspan=1>81.2%</td><td rowspan=1 colspan=1>81.9%</td></tr><tr><td rowspan=1 colspan=1>96.4%</td><td rowspan=1 colspan=1>86.8%</td><td rowspan=1 colspan=1>77.5%</td><td rowspan=1 colspan=1>85.6%</td><td rowspan=1 colspan=1>80.0%</td></tr><tr><td rowspan=1 colspan=1>98.5%</td><td rowspan=1 colspan=1>86.8%</td><td rowspan=1 colspan=1>83.2%</td><td rowspan=1 colspan=1>90.2%</td><td rowspan=1 colspan=1>88.6%</td></tr><tr><td rowspan=1 colspan=1>99.6%</td><td rowspan=1 colspan=1>87.3%</td><td rowspan=1 colspan=1>87.8%</td><td rowspan=1 colspan=1>90.8%</td><td rowspan=1 colspan=1>93.7%</td></tr></table>

Fig. 7. Measured AllToAll bandwidth with different group and message sizes relative to the theoretical upper bounds.

To optimize excursion response under these extended capabilities, we developed a synthetic microbenchmark emulating heterogeneous execution phases. Fig. 6 measures performance boost by varying the GEMM duty ratio (percentage of highpower GEMM kernels interleaved with lower-power operations such as collectives or elementwise kernels), relative to steadystate operation at 1000 W with 100% back-to-back GEMMs. Depending on per-kernel duration, we achieve up to 1.33× performance gain. This improvement results from tuning the control-loop response to shape the boost profile, maximizing performance over the 50 ms window across varying phase periods and duty cycles while strictly respecting EDP limits.

In end-to-end applications, the benefit from higher excursion limits depends on the frequency and duration of compute-intensive phases, intervening low-power operations, and sensitivity to clock frequency. In Meta’s DLRM production training, this enhancement resulted in end-to-end performance gains of 3–6%.

## D. Collective Communication with RCCL

RCCL provides collective communication primitives for AMD GPUs, supporting PCIe, Infinity Fabric Interconnect [34], and network fabrics (InfiniBand, TCP/IP) across single and multi-node deployments.

We developed a theoretical performance projection model derived from hardware-level simulations of the memory subsystem, scale-up fabric, and NIC data paths<sup>1</sup>. Guided by these projections, we co-designed a suite of RCCL optimizations across the RDMA transport, GPU kernel, and protocol layers for the MI350X. These refinements narrowed the communication gap from 2× slower than NCCL to within 10–15% of NCCL performance, enabling production-scale DLRM training. Fig. 7 shows measured AllToAll bandwidth of different message sizes against performance upper bound.

Transport and protocol refinements. To resolve a performance conflict between collective types, we adopted collective-specific

RDMA queue pair (QP) scaling: AllToAll utilizes a single QP for large-chunk efficiency and 30% higher bus bandwidth, while ring-based collectives (e.g., AllGather) retain four QPs to saturate network bandwidth. We replaced the LL128 lowlatency protocol—whose 128-byte stores split across AMD’s 64-byte cache lines—with LL64, a protocol that packs 56-byte payload with 8-byte flags into one cache-line-atomic 64-byte packet, delivering 2× AllGather/ReduceScatter and 1.2×– 1.4× AllReduce speedups.

Algorithmic and routing enhancements. We introduced Direct AllGather to replace multi-hop ring transfers with simultaneous point-to-point sends, yielding a 1.4× speedup for small messages. Our P2P batching design reduced AllToAll latency by 40% at a 128-GPU scale through the aggregation of concurrent communications.

GPU resource and occupancy optimization. A key challenge in production DLRM training is maximizing computecommunication overlap while minimizing contention in shared compute and memory resources. To mitigate CU contention, we transitioned from a thread-block-per-GPU to a warp-per-GPU communication model, capped active channels at 48 (down from 64). By increasing the thread block size from 256 to 512, we achieved a 46% VGPR (vector general-purpose register) reduction and doubled occupancy. These optimizations delivered gains of 63% for ReduceScatter and 48% for AllReduce, while reducing the performance penalty when operating with restricted CU allocations.

## E. Shampoo Optimizer and rocSOLVER

Shampoo [35, 36] is a second-order optimizer widely deployed due to its faster convergence compared to first-order methods like Adam [37]. Shampoo maintains Kroneckerfactored preconditioner matrices, requiring periodic computation of matrix fourth roots to update the preconditioners. This matrix root computation relies on eigen decomposition provided by the rocSOLVER library. For large models, these steps dominate Shampoo’s overhead, making rocSOLVER’s performance critical to overall training throughput.

The de facto algorithm to compute the eigen decomposition of a symmetric input matrix A has three steps: (1) reduce A into tridiagonal form, yielding an orthogonal U and a tridiagonal T such that $A \stackrel { \cdot } { = } U T U ^ { \bar { T } }$ ; (2) compute eigenvalues Λ and eigenvectors V of T; and (3) backtransform eigenvectors $Q \ = \ U V$ so that $A = Q \Lambda Q ^ { T }$ . In rocSOLVER, reduction to tridiagonal form is accomplished with block Householder transformations [38] operating on panels of the input matrix; updates to the rest of the matrix are applied once per panel as a pair of GEMM operations. Step (2), based on Cuppen’s Divide-and-Conquer (D.C.) [39], is the most complex step and the focus of our optimizations.

Cuppen’s D.C. splits T into sub-matrices that are processed independently and then merged in pairs. Typically, D.C. operates on unreduced sub-blocks of T (i.e., blocks separated by zero off-diagonal elements), but this can lead to load unbalancing: sub-optimal CU allocation often follows because the size of subblocks is input-dependent. Our implementation in rocSOLVER lifts this restriction, yielding precise control over CU allocation for improved performance. During a merge step, D.C. identifies deflated eigenvalues (which will not be modified by the merge) and updates all others by solving the secular equation. To improve data locality and reduce data movement, our implementation ensures that eigenvalues (and associated eigenvectors) are properly ordered. Eigenvectors belonging to identical eigenvalues require (possibly costly) extra preprocessing; rocSOLVER includes specialized code to efficiently handle matrices exhibiting several identical eigenvalues in merge steps. Lastly, newly and previously computed eigenvectors are merged using batched GEMMs for optimal performance.

![](images/8d301e7eb0addaa70dd028480ac14192b9a8adb327430f287379b22cb8af6653.jpg)  
Fig. 8. Eigen decomposition performance.

Fig. 8 demonstrates that rocSOLVER achieves comparable eigen decomposition performance on AMD MI350X relative to NVIDIA B200. Distributed Shampoo partitions large preconditioner matrices into fixed-size blocks (typically 4096 × 4096 or 8192 × 8192) to bound per-GPU computation and enable parallel eigen decomposition across devices. These block sizes represent the practical sweet spot: large enough to preserve preconditioning quality, yet small enough for tractable eigen decomposition. rocSOLVER narrows the gap with cuSOLVER at these critical 4K and 8K dimensions, enabling competitive Shampoo-based training on AMD hardware.

## IV. KERNEL TUNING AND AGENTIC AI WORKFLOWS

In this section, we first identify kernel optimization principles through case studies of key production kernels, and then encode them as reusable skills in an agentic AI workflow that scale to the long tail of model-specific operators.

## A. Triton Kernel Optimizations

1) Attention Kernels in DLRM: We focus on two important Triton kernels in Meta’s production DLRM workloads, GDPA and HSTU, and describe the AMD-specific optimization techniques that yield substantial speedups on MI350X.

Generalized dot-product attention (GDPA). GDPA [40] extends standard attention [41, 42] by fusing GeLU-activated attention and a feed-forward network into a single operator. Central to Meta’s Kunlun [43] and Interformer [44] models, the fused kernel substantially increases the compute and memory intensity, positioning the kernel as a critical optimization target. Hierarchical sequential transduction unit (HSTU). HSTU [21] is a novel sequential attention architecture designed for large-scale recommendation systems, where the input consists of variable-length user interaction histories with high sparsity. Unlike standard attention, HSTU uses variable-length batching and processes sequences with sparsity ratios as high as 0.95, meaning the effective computation is concentrated in a small fraction of the sequence.

2) Optimization Techniques: We apply five optimization techniques to the Triton kernels, each addressing a different performance bottleneck on the MI350X architecture. While we use GDPA and HSTU as examples, these optimization techniques are generally applicable to a broad set of Triton kernels for ML models.

PID swizzle. MI350X features a multi-XCD (chiplet) architecture where eight XCDs share an L2 cache. The default linear mapping of Triton program IDs to tiles can cause XCDs to access disjoint L2 cache regions, resulting in poor cache utilization. PID swizzle uses tl.swizzle2d to remap the twodimensional program ID grid so that spatially adjacent tiles are scheduled on the same XCD, improving L2 cache locality.

Occupancy hints. The waves\_per\_eu hint controls the maximum number of wavefronts resident on each execution unit, which directly affects register allocation. By setting waves\_per\_eu=3 (GDPA) or exposing it as an autotunable parameter with values 0–4 (HSTU), the compiler reduces perwavefront register usage, allowing more concurrent wavefronts to hide memory latency. This trades per-wavefront performance for higher occupancy, which is beneficial when the kernel is not fully compute-bound.

MFMA instruction configuration. AMD CDNA architectures provide matrix fused multiply-add (MFMA) instructions in configurable tile sizes, controlled by the matrix\_instr\_nonkdim parameter. By setting matrix\_instr\_nonkdim=16, the system selects 16 × 16 MFMA tiles, which on CDNA 4 can achieve higher instruction throughput than the default 32 × 32 tiles for certain matrix shapes.

Triton configuration autotuning. Triton kernel configurations (BLOCK\_SIZE, num\_stages, num\_warps, etc.) are critical to performance, and optimal values differ between NVIDIA and AMD GPUs due to architectural differences. AMD’s 64-thread wavefronts, compared to NVIDIA’s 32-thread warps, favor larger num\_warps values to achieve sufficient occupancy and hide memory latency. NVIDIA’s Tensor Memory Accelerator (TMA) on H100/B200 enables deep software pipelining (e.g., num\_stages ∈ [4, 7]) by efficiently managing asynchronous memory transfers. AMD MI350X lacks a TMA equivalent; high stage counts increase LDS pressure without commensurate latency hiding benefits. Empirically, AMD kernels achieve better performance with shallower pipelines (e.g., num\_stages ≈ 2). GeLU fast math. The default fast\_gelu implementation in Triton uses a tanh-based approximation, which on AMD GPUs decomposes into tanh(x) = 2σ(2x) − 1, requiring a sigmoid evaluation (scaled by two) and additional arithmetic. We replace this with a direct sigmoid-based GeLU approximation using the compiler intrinsics fast\_expf and fast\_dividef, which map to single hardware instructions with better performance.

3) Performance Results: We evaluate these tuning techniques by incrementally applying them to the GDPA and HSTU forward and backward kernels.

![](images/7704a7fb112d6d9f051fe0b3df6fe30aef7521dc56d18b02459967c80326a762.jpg)  
(a) Forward latency (ms).

![](images/bb13330994738dd15a5febdd2969d990af7f310bb80cfc57b954612cefb5e998.jpg)  
(b) Forward throughput (TFLOP/s).

![](images/7ffeca81071c6926c4f575acbc8efde1d6780afcdf6476e009765b44097458c7.jpg)  
(c) Backward latency (ms).

![](images/8617fe855b29e6d547f47750ea17e460917603cf33e8dab4f3e181e1ceb601d9.jpg)  
(d) Bwd throughput (TFLOP/s).

Fig. 9. GDPA kernel performance with incremental optimization techniques applied on AMD MI350X, compared to optimal configuration on B200. Lower latency and higher throughput are better.  
![](images/26b53f4c4df7499b527817d3dcf6c9fe9fc3b50fa81acbbe391f96f1b3f66e38.jpg)  
(a) Forward latency (ms).

![](images/78e6232402026cf44df1c2737db5d9cd407333d6e8d519baf61d084414b74d57.jpg)  
(b) Forward throughput (TFLOP/s).

![](images/8c45a46fce38384b6e5dd0c5fce5427615340fbfc8696ba1c3c95259a1550f96.jpg)  
(c) Backward latency (ms).

![](images/a9da870ddfbaca141f682601da4c78189406c6bfe0af7ce2fa607316501712cd.jpg)  
(d) Bwd throughput (TFLOP/s).  
Fig. 10. HSTU kernel performance with incremental optimization techniques applied on AMD MI350X, compared to optimal configuration on B200. Lower latency and higher throughput are better.

GDPA. Fig. 9a and 9b show the incremental forward kernel improvements based on a representative configuration<sup>2</sup> in GEM models [45]. Starting from the baseline of 1.47 ms latency (58.4 TFLOP/s), the cumulative application of GeLU fast math, occupancy hints, MFMA instruction configuration, and PID swizzle reduces latency to 0.59 ms (145.9 TFLOP/s), a total 60.0% reduction in latency and 2.5× in throughput. Fig. 9c and 9d show the backward kernel results. The same techniques (except matrix\_instr\_nonkdim, which regresses the backward pass) reduce latency from 3.67 ms to 2.17 ms, a 40.9% reduction. Compared to B200, the optimized GDPA kernels on MI350X achieve better performance.

HSTU. Fig. 10a and 10b show the forward kernel results for a representative configuration<sup>3</sup> in production models. Block size tuning, occupancy hints, and PID swizzle together yield a 3.22× speedup, with peak throughput improving from 45.0 to 144.8 TFLOP/s. Fig. 10c and 10d show the backward kernel results. Out-of-the-box performance is already stronger on MI350X here, leaving less optimization headroom: MFMA parameter tuning delivers a 1.21× speedup, with peak through put improving from 60.2 to 72.6 TFLOP/s. Averaged across sequence lengths, the optimized MI350X kernels are 1.22× faster than B200 on the forward pass and 1.80× faster on the backward pass, for a combined 1.69× advantage that widens at longer sequences.

## B. CUDA Kernel Optimizations

GPU kernels written in CUDA can be compiled for AMD GPUs via HIP translation (Section II-B). While it enables code portability, certain CUDA idioms, particularly those relying on atomic operations, perform poorly on AMD’s multi-chiplet architecture. This section describes the architectural differences that cause these performance gaps and our algorithmic strategies to address them.

1) Atomic Operations on Multi-Chiplet GPUs: Unlike NVIDIA GPUs where atomic operations complete in a monolithic L2 cache, MI350X atomics may require cross-chiplet cache coherency (note 8 XCD chiplets in MI350X), incurring round-trip latencies across the Infinity Fabric. When thousands of threads contend on a single address (as is common in reduction-style kernels), these coherency messages serialize execution, leading to poor performance.

The fundamental issue is that per-thread global atomics scale poorly with thread count on multi-chiplet architectures. On a monolithic GPU, the L2 cache can coalesce nearby atomic operations; on a multi-chiplet GPU, each chiplet must coordinate with the others, and a single contended address becomes a serialization bottleneck. Our measurements show that MI350X can be over 10× slower than B200 on atomic-heavy kernels at high contention rates, while matching or exceeding B200 on atomic-free, bandwidth-bound kernels.

Algorithmic strategies: hierarchical reduction and caching. We employ two complementary strategies depending on the contention pattern. When many threads contend on a single shared address (e.g., a global counter), we apply hierarchical reduction: threads accumulate results in fast on-chip memory (registers or shared memory), a parallel tree reduction sums them within each block, and a single representative thread issues one global atomic. This converts N global atomics into ⌈N/block\_size⌉, a reduction factor of up to 1024×. When threads target many distinct addresses with temporal locality (e.g., gradient scatter to embedding rows), we apply caching: each thread maintains a small register cache of recently written addresses, merging consecutive updates to the same address locally before issuing a single atomic add on eviction. The tradeoff is additional shared memory usage. For kernels where contention is rare, the overhead may slightly increase latency, but the benefit grows superlinearly with contention rate, making this transformation highly effective for production workloads with moderate-to-high atomic traffic.

![](images/3605df1537003164f3998399dfa1a8e5cc81e485ce6cea8eef5d6093aa3ba4c5.jpg)  
Fig. 11. Bounds check forward kernel latency vs. violation rate. Tree reduction eliminates atomic contention, enabling MI350X to outperform B200 at high violation rates.

2) Example: Bounds Check Kernel: The kernel validates that all embedding table indices fall within valid ranges before performing lookups. When operating in WARNING mode, the kernel (1) accumulates a per-thread count of out-of-bounds violations, (2) corrects each invalid index, and (3) increments the global out-of-bounds violation counter via atomic add operations. On MI350X, atomic operations become severe bottlenecks at high violation rates.

Original implementation. After each thread accumulates its violation count, threads with non-zero counts issue gpuAtomicAdds to global memory. With 1024 threads per block and thousands of blocks processing millions of indices, a typical workload can produce millions of threads that contend on a single address.

Optimized implementation. In order to reduce atomic operations, per-thread global adds are replaced with shared-memory parallel tree reduction followed by a single atomic add. Rather than writing to a global counter directly, each thread writes its local violation count to a shared memory buffer allocated at the block level. A 10-step binary reduction tree then sums the per-thread contributions entirely within shared memory, avoiding any cross-XCD traffic. Finally, thread 0 issues a single gpuAtomicAdd with the block-wide sum to the global counter, reducing global atomics from ${ \sim } 1 0 ^ { 6 }$ to ${ \sim } 1 0 ^ { 3 }$ per batch for a typical production workload.

Results. Fig. 11 shows the latency on a production embedding configuration (30 tables, 13.8M indices) as violation rate varies from 0% to 20%. With per-thread atomic adds, MI350X forward latency degrades 8.5× from 0% to 20% violations while B200 degrades only 3.3×. At 10% violations, tree reduction delivers 4.26× speedup (2.1 ms vs 9.0 ms). At 20% violations, the speedup reaches 4.34×, and MI350X with tree reduction is 1.75× faster than B200.

3) Example: Grouped Index Select (GIS) Kernel: A core operation in recommendation model training, GIS gathers rows across multiple embedding tables in a single fused kernel. The forward pass is a bandwidth-bound simple gather, while the backward pass scatters gradients back via atomic add, with contention arising when multiple samples select the same row. Forward optimization. AMD’s native 64-thread wavefronts underutilize hardware when embedding dimensions are small. Thus, we introduce 32-thread emulated warps that enable multiple rows to be packed into the same wavefront, eliminating idle lanes and improving memory bandwidth utilization. This optimization delivers 1.16×–1.24× speedup across production configurations (Fig. 12a).

![](images/e587f023fb50f4e4a312804eb3145faeeb31c90c700925e6ad68fcbc3232d84a.jpg)  
(a) Forward latency (ms).

![](images/9549a248e0a5e46df10b758e15cac09b4a37e418d53ca0a8472873025356fa3f.jpg)  
(b) Backward latency (ms).

Fig. 12. GIS kernel performance. For the bandwidth-bound forward kernel, MI350X exceeds B200. For backward, multi-row packing helps on mixed dimension configs but cannot fully overcome atomic contention on uniform, high-collision workloads.
<table><tr><td>B T L</td><td colspan="3">D Prec α FWD BWD</td><td colspan="3">MI350X FWD BWD</td></tr><tr><td>65536</td><td>1.0</td><td>(ms) 1.75</td><td>(ms) 14.40</td><td>(ms) 1.48</td><td>Before 23.6</td><td>After 7.72</td></tr><tr><td>5 65536</td><td>55 192 fp16</td><td>0.84</td><td>4.61</td><td>(0.85×) 0.78</td><td>(1.64×) 9.92</td><td>(0.54×) 3.97</td></tr><tr><td></td><td>55 192 fp16 1.15</td><td></td><td></td><td>(0.93×) 1.84</td><td>(2.15×) 24.4</td><td>(0.86×) 8.06</td></tr><tr><td>65536</td><td>10 26 256 fp16 1.0</td><td>1.86</td><td>15.79</td><td>(0.99×) 0.91</td><td>(1.55×) 11.4</td><td>(0.51×) 4.36</td></tr><tr><td>65536</td><td>10 26 256 fp16 1.15</td><td>0.87</td><td>4.85</td><td>(1.05×) 1.50</td><td>(2.35×) 10.6</td><td>(0.90×) 5.60</td></tr><tr><td>65536 3 55 192 fp32</td><td>1.0</td><td>1.36</td><td>7.56</td><td>(1.10×) 0.53</td><td>(1.40×) 7.45</td><td>(0.74×) 2.69</td></tr><tr><td>65536</td><td>3 55 192 fp32 1.15</td><td>0.53</td><td>2.67</td><td>(1.00×) 2.03</td><td>(2.79×) 10.9</td><td>(1.01×) 5.80</td></tr><tr><td>65536 26 256 fp32</td><td>1.0</td><td>2.29</td><td>8.23</td><td>(0.89×) 0.66</td><td>(1.32×) 8.95</td><td>(0.70×) 3.60</td></tr><tr><td>65536</td><td>6 26 256 fp32 1.15</td><td>0.59</td><td>2.98</td><td>(1.12×) 0.85</td><td>(3.00×) 10.9</td><td>(1.21×) 3.69</td></tr><tr><td>131072 3 20 256 fp16</td><td>1.0</td><td>0.84</td><td>6.66</td><td>(1.01×) 0.46</td><td>(1.64×) 8.79</td><td>(0.55×) 2.65</td></tr><tr><td>131072 3</td><td>20 256 fp16 1.15</td><td>0.43</td><td>2.39</td><td>(1.07×) 0.99</td><td>(3.68×) 10.5</td><td>(1.11×) 4.10</td></tr><tr><td>524288</td><td>20 256 fp16 1.0</td><td>1.11</td><td>7.02</td><td>(0.89×)</td><td>(1.50×)</td><td>(0.58×)</td></tr><tr><td>524288 1 20 256 fp16 1.15</td><td></td><td>0.54</td><td>2.94</td><td>0.60 (1.11×)</td><td>8.26 (2.81×)</td><td>2.83 (0.96×)</td></tr></table>

Table II. TBE training performance: MI350X forward and backward before/after optimization vs. B200 with representative production configs: B (batch size), T (number of tables), L (lookups per table), D (embedding dimension), Prec (precision), α (Zipf skew factor). Parenthetical values show latency relative to B200 (<1.0× means MI350X is faster).

Backward optimization. In the backward pass, we exploit temporal locality within each thread’s iteration stream. Each thread maintains a 2-slot register cache that maps row indices to accumulated gradient values. When consecutive iterations target the same row (a cache hit), gradients are summed locally in registers with zero global memory traffic. On a miss with both slots occupied, the oldest entry is evicted via a single atomicAdd and replaced. This converts N consecutive updates to the same row into one atomic, with the coalescing factor determined by the locality of the index stream, delivering 1.04×–1.11× backward speedup (Fig. 12b).

4) Table Batched Embeddings (TBE): Table Batched Embeddings (TBE) [46] is a core primitive in Meta’s recommendation workloads, responsible for looking up and aggregating embeddings from hundreds of tables with billions of rows. TBE fuses multiple embedding lookups into a single kernel launch, amortizing kernel overhead and enabling memory access coalescing across tables. The forward pass gathers embeddings based on sparse indices; the backward pass scatters gradients back to the embedding tables, requiring atomic accumulation for indices that appear multiple times.

Backward optimization. We focus on the optimization of the custom backward kernel for MI350X’s wavefront-64 architecture. Key techniques include warp-per-row processing (each wavefront handles one embedding row), hand-tuned vectorized load/store for common embedding dimensions (64, 128, 192, 256), and wavefront-level accumulation that minimizes crosslane communication.

Results. Table II summarizes TBE training performance across 12 production configurations spanning different batch sizes, table counts, sequence lengths, embedding dimensions, precisions, and index distributions. The backward optimization delivers 1.87–3.32× speedup on MI350X, with larger gains on FP16 (2.5–3.3×) than FP32 (1.9–2.8×) due to better vectorization. Compared to B200, MI350X with the optimized backward kernel is 1.35–1.96× faster on uniform distributions (α=1.0), where memory bandwidth dominates. On Zipf distributions (α=1.15), where skewed access patterns cause cache thrashing, MI350X and B200 are comparable (0.83–1.16×).

## C. Agentic Kernel Optimization Workflow

While GDPA and HSTU are widely shared across models, Meta’s DLRM ecosystem contains a long tail of custom kernels tailored to individual models. Manually tuning each kernel is labor-intensive and does not scale. To address this, we developed an agentic AI workflow that encodes our tuning principles as executable skills, enabling automated kernel performance analysis and optimization. This workflow builds upon Meta’s KernelEvolve framework [47], which provides the underlying agent infrastructure and skill abstraction mechanisms. While KernelEvolve focuses on the general systems architecture for agentic code optimization, our contribution lies in encoding AMD-specific tuning principles as production-ready skills and validating their effectiveness at scale on Meta’s production DLRM training workloads.

The agentic workflow operates in three phases:

1) Benchmark & Profile: Capture representative inputs from production workloads, reproduce the performance in a standalone benchmark, and leverage profiler tools (e.g., AMD rocprofv3 [48], Triton Proton [49]) to identify the main bottlenecks such as low occupancy, poor cache utilization, or instruction stalls.

2) Match & Fix: Map observed bottlenecks to a library of optimization skills, each encoding a tuning principle (e.g., PID swizzling, occupancy hints, atomic avoidance) with code transformation templates and applicability conditions.

3) Generate & Validate: Produce candidate kernel modifications, benchmark them on representative inputs, and verify numerical equivalence (within tolerance, automatically rejecting failures) and performance improvements on a set of production workloads.

Case Study: Jagged Flash Attention. We illustrate the workflow with the jagged flash attention (JFA) kernel, which represents a challenging optimization target due to its irregular memory access patterns. The JFA kernel, used in our ranking models, exhibited a 2× slowdown on MI350X compared to B200 in its backward pass. Profiling revealed that tl.atomic\_add operations dominate execution time, a known bottleneck on AMD’s multi-XCD architecture (see Section IV-B for details). The agent recognized this pattern, noting that the grouped query attention (GQA) mode causes multiple Q heads to race on shared dK/dV gradients. Following the skill’s guidance, the agent rewrote the kernel to accumulate dK/dV in registers per KV-head group and issue a single tl.store at the end, a tree-reduction pattern that eliminates atomic operations. The optimized kernel achieved a 2.3× speedup (32 ms→14 ms), transforming MI350X from 2× slower than B200 to 15–20% faster, and delivering a 25% improvement in end-to-end model training throughput.

Applying to dozens of production Triton kernels within Meta’s DLRM ecosystem yields a success rate exceeding 80%. Agent-generated optimizations achieve near-expert performance while reducing development time by an order of magnitude. For a comprehensive evaluation of the effectiveness across diverse kernel types, we refer readers to [47]. By codifying expert knowledge into reusable skills with clear triggering conditions and transformation patterns, the workflow enables systematic exploration of the optimization space while ensuring that anti-patterns for AMD GPUs are automatically detected and addressed.

Generality. Our workflow builds on Meta’s KernelEvolve [47], which encodes tuning expertise as reusable skills across NVIDIA, AMD, and MTIA. The skill abstraction transfers across ISAs. The atomics-to-tree-reduction skill above has a direct analog on Meta’s MTIA accelerators [5, 6], whose PE array also handles contended atomics inefficiently: the same global-atomic reduction is rewritten as a per-PE local reduction followed by a global reduction over the on-chip reduction-engine network, yielding ∼10× (up to 65×) speedups on reduce-style kernels. Because the workflow needs only a portable tile-level kernel DSL and a compiler, which Google TPU (XLA/Pallas) [50] and AWS Trainium (Neuron/NKI) [8] also provide, these skills extend to a new accelerator as an incremental port rather than a from-scratch rebuild.

## D. PyTorch Inductor Kernel Optimizations

PyTorch Inductor is the default backend for compiler kernel fusion: transforming multiple eager operations into a single generated kernel launch to eliminate intermediate memory traffic, reducing bandwidth requirements by 2×–5× compared to unfused eager execution.

Inductor generates four main categories of kernels: pointwise (triton\_poi) for elementwise operations (accounts for 40– 60% of the total kernel count), reduction (triton\_red) for aggregations, persistent reduction (triton\_per) for reductions fitting in registers or shared memory, and for-each (triton\_for) for batched tensor operations.

![](images/3f8748f992c8ec73aa5466ce32e6225ad4f46969ff6c2b20fbe3c062c20f396c.jpg)  
(a) ResNet-50.

![](images/3cff02e375cdc91165bdb93801c28a82f62093379006baf89a7f10207749cc71.jpg)  
(b) DenseNet-121.

![](images/dc2f70876e5a573955d90b756e96a068f31179eb1c193051e78b41a351f0cd82.jpg)  
(c) NanoGPT.  
Fig. 13. Triton Inductor kernel latencies by category for three representative models, showing MI350X before/after AMD optimizations compared to B200. Lower latency is better.

Inductor’s performance benefits are largely rooted in NVIDIA-specific optimizations refined over years of development and iterative tuning. Porting to alternative accelerators, such as AMD GPUs, necessitates more than mechanical adaptation; it requires re-evaluating core heuristics, expanding autotuning search spaces, and developing bespoke microarchitecture aware optimizations.

1) Optimization Techniques: We apply AMD-specific optimizations leveraging architectural insights from Section IV-A: Software pipelining. We replace Python range() with Triton’s tl.range() using num\_stages=2, enabling better memorycompute overlap. This reduces latency by 11.7% for ResNet-50 reduction kernels.

Native Triton intrinsics are used to replace library function calls where available: tl.rsqrt() instead of libdevice.rsqrt(), and tl.minimum / tl.maximum instead of helper functions.

Register configuration. With 256 vector general purpose registers (VGPRs) per thread, we raise Inductor’s register spill threshold from 16 to 32 for larger tiles and better instructionlevel parallelism.

Expanded autotuning. We extend Inductor’s autotuning config urations with AMD-specific tile sizes and occupancy hints. For example, we add non-square tiles (64 × 32, 128 × 16, 128 × 32, 32 × 512) for 2D pointwise kernels.

2) Model-Level Results: We evaluate these optimizations on standard vision and language models. Fig. 13 shows percategory kernel latencies for three models trained on MI350X before/after optimization versus B200. For ResNet-50 [51], reduction kernels improve 11.7%, reaching 1.16× of B200 latency; other Triton inductor kernels outperform B200. For DenseNet-121 [52], pointwise kernels improve 9.2%; reduction improves 6.9%, reaching B200 parity; persistent reduction improves 3.5%. Overall kernel time is reduced by 4.2%. For NanoGPT [53], persistent reduction improves 4.8%; pointwise improves 3.8%. Overall kernel time is reduced by 3.4%.

In summary, AMD-specific optimizations deliver 6.9–11.7% improvement for reduction, 3.5–4.8% for persistent reduction, and up to 9.2% for pointwise. While MI350X matches or exceeds B200 across several categories for CNNs, it is still 30% slower on transformer Inductor kernels, indicating optimization headroom remains.

## V. MODEL-SYSTEM CO-DESIGN

AMD MI350X’s large HBM capacity creates opportunities for model-system co-design, where training configurations and model architectures are jointly optimized to exploit the additional memory resources. We explore two complementary directions: improving training throughput (examples/second) and improving model quality.

## A. Strategies for Training Efficiency

Increased batch size. Larger HBM capacity enables training with larger batch sizes, improving GPU utilization by amortizing kernel launch overhead and better saturating compute units. For example, a friend recommendation model achieves 26% speedup at 2× batch size, 46% at 4×, and 51% at 6× (Fig. 15a). For a social ranking model migrated to MI350X, doubling the batch size yields 12–15% throughput improvement over B200 baseline with the same batch size (B200 encounters out-ofmemory errors at doubled batch size).

Removal of activation checkpointing. Activation checkpointing (AC) trades memory for computation by recomputing activations during the backward pass instead of storing them. With larger HBM, AC can be disabled for memory constrained layers, eliminating redundant recomputation. The activation memory budget controls how aggressively PyTorch checkpoints activations: a budget of 1.0 saves all activations (maximum memory, no recomputation), while lower values trigger selective recomputation to reduce memory at the cost of throughput. Fig. 14 shows the tradeoff between activation memory budget and training throughput on a 1.3B-parameter GPT model. MI350X’s larger HBM enables it to run without AC entirely, achieving 96.4 examples/second. In contrast, B200 must use aggressive AC (budget ≤ 0.35) just to fit, achieving 71.5 examples/second, a 1.35× throughput disadvantage solely attributable to memory constraints.

Dense parallelism strategy tuning. Additional HBM allows for relaxed sharding strategies, mitigating communication overhead. For FSDP-based training [54], moving from ZeRO-3 to ZeRO-2 or ZeRO-1 [55] reduces AllGather and ReduceScatter communication volume. Switching to HSDP (Hybrid Sharded Data Parallelism) further hides communication latency by overlapping with computation. For a large Ads foundation model at 1K GPU scale, HSDP provides 6% speedup at the cost of 750 MB additional memory per GPU, a tradeoff feasible only with HBM capacity headroom.

![](images/c05921284804270190392252b37126f4fb33cb4601557b6ef8598ce368e4e225.jpg)  
Fig. 14. Training throughput vs. HBM tradeoff on a GPT-2 1.3B model with activation checkpointing.

Sparse embedding placement optimization. Recommendation models use large embedding tables that often reside in host memory due to GPU memory constraints. With larger HBM, embedding tables can be moved from host to HBM, eliminating PCIe transfer bottlenecks. Additionally, row-wise sharding can be replaced with table-wise or column-wise sharding, reducing AlltoAll communication and CPU/GPU synchronization overhead. Experiments on a feed ranking model show >10% speedup from embedding placement optimization, with the expectations for larger gains when combined with 2D embedding sharding strategies.

Combined results. Fig. 15b shows the combined effect of co-design strategies on a feed ranking model. After porting to MI350X, we increased the batch size, removed AC, and optimized embedding placement, resulting in a 1.26× speedup. In conjunction with kernel optimizations, the model realizes a total speedup of 1.5× compared to the initial baseline. These results demonstrate that combining multiple strategies delivers substantial efficiency improvements.

## B. Strategies for Model Quality

Model architecture upscaling. The MI350X’s expanded HBM capacity accommodates larger model architectures that would otherwise require additional GPU nodes. Increased memory footprint directly translates into superior model quality through several key avenues: (1) Longer input sequences: transformerbased recommendation models can leverage longer sequences to capture richer user context. (2) Expanded embedding tables: larger hash sizes reduce collision rates, thereby preserving higher-fidelity information within the learned representations. (3) Increased dense complexity: support for more dense layers and larger hidden dimensions enables the extraction of more expressive features.

Results. For a feed ranking model, increasing the number of attention layers from 8 to 16 yields 0.08% normalized entropy (NE) [15] improvement; further increasing to 20 layers yields 0.11% NE improvement.<sup>4</sup> These architectural changes fit within AMD MI350X’s memory capacity, whereas B200 would require cluster expansion.

![](images/bfa47759bee96086fc63efbd0d55c52cdcdf863968614d63503e512ad13bb2e0.jpg)  
(a) A model with increased batch sizes.

![](images/f9c8181ced5df6548a2751af4f2e93eced160f789d9ad0ee54f99f4ac5dffe1e.jpg)  
(b) A model with co-design techniques.  
Fig. 15. (a) Batch size scaling yields up to 51% throughput improvement. (b) Combined co-design strategies (batch size increase, AC removal, embedding placement optimization) yield 26% speedup.

Generality. This co-design principle, matching the model to the accelerator’s memory hierarchy, is not AMD-specific. It transfers directly to Meta’s MTIA training accelerator (MTIA-300 [5]), whose larger HBM relative to the incumbent GPU is likewise exploited for larger batches and fewer trainers. The principle is direction-agnostic: on the SRAM-bound prior inference generation (MTIA-200 [6]) it instead drives the model to shrink its activation footprint to fit on chip, e.g., decomposing batched matmuls to remove redundant in-batch broadcasts cuts activation memory by ∼25%, admitting 10–50× larger embedding tables.

## VI. EVALUATION

The previous sections presented the efficacy of specific ML stack optimizations. This section assesses their collective impact on production recommendation models, illustrating how the cross-layer optimizations jointly maximize end-to-end systemlevel performance.

## A. Production Results

Table III summarizes the job scale and measured throughput of ten representative production workloads from multiple product groups on MI350X and B200, using matched batch sizes. GPU specifications follow the details in Section II-B; readers should note the differences between our deployment and standard public offerings. Evaluated models range from single-node (8 GPUs) to multi-node (128 GPUs), representing the diversity of production training scales.<sup>5</sup> Here, Params is the total parameter count (sparse embeddings plus dense layers), and Seq. len is the longest input sequence a model ingests. The geometric mean performance ratio of 101.2% demonstrates that MI350X matches B200 throughput at equal GPU count and batch size across the model suite.

We use an equal number of B200 GPUs to give a straightforward comparison between same-generation hardware from both vendors (Section II-B). As a widely adopted platform, Blackwell serves as a familiar anchor point for readers to evaluate our results. The B200 baseline is not a strawman: as the long-established incumbent, it has received equal or greater optimization than the MI350X: years of model adoption, CUDA library tuning, and kernel development. The B200 keeps an edge on performance-per-watt, but power is a small share of capital dominated total cost of ownership (TCO), so the MI350X’s lower TCO drives provisioning.

<table><tr><td>Model</td><td>GPUs</td><td>Seq. len</td><td>Params</td><td></td><td>B200 ex/s MI350X ex/s MI350X/B200</td><td></td></tr><tr><td>A</td><td>8</td><td>512</td><td>~100M</td><td>921,714</td><td>754,936</td><td>82%</td></tr><tr><td>B</td><td>8</td><td>256</td><td>~100B</td><td>1,592,926</td><td>1,810,036</td><td>114%</td></tr><tr><td>C</td><td>64</td><td>200</td><td>~200B</td><td>728,001</td><td>884,216</td><td>121%</td></tr><tr><td>D</td><td>64</td><td>1000</td><td>~300B</td><td>774,324</td><td>755,221</td><td>98%</td></tr><tr><td>E</td><td>64</td><td>200</td><td>~300B</td><td>283,641</td><td>267,635</td><td>94%</td></tr><tr><td>F</td><td>128</td><td>1000</td><td>~3.6T</td><td>330,481</td><td>304,781</td><td>92%</td></tr><tr><td>G</td><td>64</td><td>200</td><td>~30B</td><td>544,537</td><td>499,066</td><td>92%</td></tr><tr><td>H</td><td>64</td><td>24,600</td><td>~500B</td><td>120,223</td><td>115,301</td><td>96%</td></tr><tr><td>I</td><td></td><td>80 4096–8192</td><td>~1.9T</td><td>546,072</td><td>558,127</td><td>102%</td></tr><tr><td>J</td><td>24</td><td>700</td><td>~20B</td><td>185,728</td><td>242,325</td><td>130%</td></tr><tr><td>Geomean</td><td></td><td></td><td></td><td></td><td></td><td>101.2%</td></tr></table>

Table III. End-to-end training throughput (examples/second) on MI350X vs B200 across ten production models. Higher is better.

![](images/ab2a3c41ce9f2c0d4322a170518f730570b3a7ca3d4d4bc78c33201803f475c3.jpg)  
Fig. 16. Kernel latency ratio (MI350X / B200) by category. For each model, the sum of kernel latencies on B200 is normalized to 100%. This analysis does not consider multi-stream overlap.

The production models also achieve higher per-GPU training throughput by leveraging larger HBM and batch sizes on MI350X, with gains in Model-D (98% to 102%), Model-E (94% to 110%), and Model-F (92% to 95%). For the kernel latency analysis in the next section, we intentionally use only the same batch sizes. While this underutilizes the MI350X, it provides consistent shapes needed to compare kernel-level performance between platforms.

## B. Kernel Category Analysis

To understand the sources of performance differences, we analyze GPU traces collected during training to categorize kernel latency by type: GEMM, Triton Inductor, CUDA embedding, elementwise, collectives, and hand-written Triton kernels. Fig. 16 shows the per-model breakdown of kernel latency ratio for each category.

Where MI350X excels. Hand-written Triton kernels show consistent MI350X advantages, delivering a 6% to 52% reduction in execution latency through the optimizations discussed in Section IV-A. Model-H benefits particularly from faster HSTU kernels, achieving 2.1× speedup for the Triton category. Embed ding operations show mixed results but generally favor MI350X (0.61×–1.04×), leveraging the CUDA kernel optimizations detailed in Section IV-B.

Overall, collective performance is comparable between the two GPU types, with MI350X outperforming B200 in multihost, small-message AllToAll collectives, specifically benefiting Models C and J. Note that overlapped kernel execution presents a challenge for latency analysis: due to the multistream nature on GPUs, concurrent compute and communication kernels complicate the measurement of their individual contributions to end-to-end latency. For instance, collectives in Models E and F take longer to execute but are masked by compute kernels in the B200 traces, resulting in no net impact on throughput. Consequently, total kernel latency may overstate the true impact of collectives, leading to discrepancies between Fig. 16 and measured throughput in Table III.

Where B200 leads. GEMM operations consistently favor B200, with MI350X showing 1.3×–1.7× higher latency. MI350X lacks native hardware support for TF32, instead emulating them with three BF16 GEMMs at a performance penalty. Model-A’s overall 18% performance deficit is largely attributable to TF32 GEMM operations. While this is a notable limitation, modern models are increasingly moving toward BF16 or lower precision. We expect continued model modernization and improvements to emulated TF32 performance to further narrow the overall gap. Additionally, cuBLAS outperforms hipBLAS due to its relative maturity, especially in the case of irregular matrix dimensions. Triton Inductor kernels show model-dependent results: some models achieve parity (0.99×– 1.08×) while others show MI350X incurring 1.22× higher latency than B200, reflecting ongoing efforts to refine Inductor and eliminate heuristics specifically tailored for NVIDIA GPUs (Section IV-D).

## C. Operational Considerations

Integrating a second GPU vendor into a previously NVIDIAonly fleet raised operational challenges in power, reliability, and observability. In addition to performance, operational maturity is just as critical to adopting a new vendor, and much of the tooling and fleet-management machinery we built generalizes to future accelerators.

Reliability. We track MI350X on the same metrics as our NVIDIA fleet (training-failure rate, unplanned unavailability, and hardware-swap rate), and it is steadily converging toward NVIDIA-fleet levels: the training-failure rate has more than halved from its peak as we resolved new at-scale failure modes, namely HBM RAS, multi-chiplet memory-access faults, and silent data corruption (SDC). SDC, the bit-flips that corrupt results without crashing, grows with cluster size and is the hardest to tame, so we address it end-to-end: a model-level checksumand-replay check flags non-deterministic steps (distinguishing on-chip corruption from communication non-determinism); an all-pairs vote across a host’s GPUs pinpoints the culprit device (the one that disagrees with all its peers) and auto-quarantines the host; and the same check runs during host qualification and burn-in to screen SDC-prone hosts before they enter the training pool, catching over 2× the SDC of the vendor’s field-health tool. These mitigations preserve correctness: across the tens of models we train on both vendors, results reach numerical parity (training-loss difference within 0.3% and NE within $\sim 1 0 ^ { - 4 }$ at multi-thousand-GPU scale).

Observability. AMD telemetry initially lagged NVIDIA’s mature stack, so we closed the gap: we integrated the efficiency metrics our NVIDIA fleet relies on (TFLOPS, SM/CU and memory utilization, throughput) and a hardware-agnostic trace-analysis interface (compute/communication breakdown, imbalance, overlap, and timeline analyses) that also enables side-by-side cross-vendor trace comparison (the basis for the apples-to-apples measurements here) and extends to MTIA.

Power management. Operating the MI350X against its excursion envelope rather than its steady-state limit is a fleet-level capability: it depends on platform provisioning beyond the OCP minimum and on tuning the control-loop response to production phase patterns. Section III-C describes the mechanism and the resulting 3–6% end-to-end gain.

## VII. RELATED WORK

Hardware Accelerators. Custom hardware accelerators exploit the structure of neural network computations. Meta’s MTIA [3,4,56] an in-house family of custom AI chips, designed to provide superior cost efficiency over commercial GPUs for the company’s large-scale DLRM and GenAI workloads. Google’s TPU [50] employs systolic arrays for efficient matrix multiplication. Recent accelerators target emerging workloads: spiking neural networks [57], heterogeneous precision [58], and processing-in-memory architectures that reduce the memory bandwidth bottleneck [59].

Distributed Training Infrastructure. The LLM pretraining study on AMD MI300X GPUs [60] presents the transformer sharding strategy and the practical guidance for fault-tolerant distributed training. Scaling DLRM training requires purposebuilt infrastructure. Zion [61] is a scale-up DLRM training platform that co-locates CPUs and accelerators to handle the memory capacity demands of embedding tables while leveraging GPU compute for dense operations. TorchRec [22] provides a PyTorch domain library with optimized distributed primitives, including automatic sharding planners that partition embedding tables across devices.

Embedding Table Sharding and Placement. DLRMs present unique systems challenges due to their massive embedding tables, which can exceed hundreds of gigabytes and dominate both memory capacity and bandwidth requirements. Intelligent partitioning of embedding tables across heterogeneous memory hierarchies is critical for training throughput. RecShard [62] formulates embedding placement as a mixed-integer linear program, leveraging per-feature access statistics—including categorical value frequency distributions and pooling factors—to achieve over 2.5× throughput improvement compared to sizebased sharding. The key insight is that not all embedding rows are accessed equally; exploiting this skew enables fine-grained placement across GPU HBM, CPU DRAM, and SSDs. 2D sparse parallelism [23] combines data and model parallelism for embeddings, enabling training to scale to thousands of GPUs.

Kernel-Level Optimizations. Efficient GPU kernels are essential for embedding operations. FBGEMM [46] provides high-performance implementations of table batched embedding lookups, fused optimizers, and quantized operations optimized for recommendation workloads. These kernels exploit memory access coalescing, warp-level parallelism, and mixed-precision computation to maximize GPU utilization. For inference, mixedprecision embeddings [63] achieve 200× compression by learning per-feature precision assignments, grouping features by frequency to reduce the search space. KernelEvolve [47] provides an agentic framework for kernel optimization; we build specialized workflows that incorporates expert-tuning heuristics and architectural insights tailored for training workloads, sim plifying cross-platform GPU kernel porting and optimization.

## VIII. CONCLUSION

We present the first comprehensive study demonstrating that AMD MI350X GPUs achieve performance parity with NVIDIA B200 for production DLRM training through crossstack optimizations spanning core ROCm libraries, kernel tuning, and model co-design. Our work addresses architectural differences between vendors—memory capacity, atomic performance, instruction sets—through principled algorithmic adaptations, hardware-aware autotuning, and HBM-conscious model co-design, providing a transferable framework for hardware-agnostic training at scale. Our optimizations have been upstreamed and open-sourced, establishing a foundation for multi-vendor ML infrastructure.

## REFERENCES

[1] Alan Smith and Vamsi Krishna Alla, “AMD Instinct MI300X: A Generative AI Accelerator and Platform Architecture,” in IEEE/ACM MICRO, May 2025.

[2] AMD. (Accessed: 2026-03-31) AMD Instinct MI350X GPUs. [Online]. Available: https://www.amd.com/en/products/accelerators/instinct/mi35 0/mi350x.html

[3] Yee Jiun Song, Andrew Tulloch, Harikrishna Reddy, Chunqiang Tang, and Vijay Thakkar. (Accessed: 2026-03-31) Four MTIA Chips in Two Years: Scaling AI Experiences for Billions. [Online]. Available: https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/

[4] Amin Firoozshahian, Joel Coburn, Roman Levenstein, Rakesh Nattoji, Ashwin Kamath, Olivia Wu, Gurdeepak Grewal, Harish Aepala, Bhasker Jakka, Bob Dreyer, Adam Hutchin, Utku Diril, Krishnakumar Nair, Ehsan K. Aredestani, Martin Schatz, Yuchen Hao, Rakesh Komuravelli, Kunming Ho, Sameer Abu Asal, Joe Shajrawi, Kevin Quinn, Nagesh Sreedhara, Pankaj Kansal, Willie Wei, Dheepak Jayaraman, Linda Cheng, Pritam Chopda, Eric Wang, Ajay Bikumandla, Arun Karthik Sengottuvel, Krishna Thottempudi, Ashwin Narasimha, Brian Dodds, Cao Gao, Jiyuan Zhang, Mohammed Al-Sanabani, Ana Zehtabioskuie, Jordan Fix, Hangchen Yu, Richard Li, Kaustubh Gondkar, Jack Montgomery, Mike Tsai, Saritha Dwarakapuram, Sanjay Desai, Nili Avidan, Poorvaja Ramani, Karthik Narayanan, Ajit Mathews, Sethu Gopal, Maxim Naumov, Vijay Rao, Krishna Noru, Harikrishna Reddy, Prahlad Venkatapuram, and Alexis Bjorlin, “MTIA: First Generation Silicon Targeting Meta’s Recommendation Systems,” in ACM/IEEE ISCA, 2023.

[5] MTIA Team, “MTIA 300: Meta’s First Training Chip Featuring Built-in NICs and Collective Offloading Engines,” in ACM/IEEE ISCA, 2026.

[6] Joel Coburn, Chunqiang Tang, Sameer Abu Asal, Neeraj Agrawal, Raviteja Chinta, Harish Dixit, Brian Dodds, Saritha Dwarakapuram, Amin Firoozshahian, Cao Gao, Kaustubh Gondkar, Tyler Graf, Junhan Hu, Jian Huang, Sterling Hughes, Adam Hutchin, Bhasker Jakka, Guoqiang Jerry Chen, Indu Kalyanaraman, Ashwin Kamath, Pankaj Kansal, Erum Kazi, Roman Levenstein, Mahesh Maddury, Alex Mastro, Siji Medaiyese, Pritesh Modi, Jack Montgomery, Satish Nadathur, Amit Nagpal, Ashwin Narasimha, Maxim Naumov, Eleanor Ozer, Jongsoo Park, Poorvaja Ramani, Harikrishna Reddy, David Reiss, Deboleena Roy, Sathish Sekar, Arushi Sharma, Pavan Shetty, Aravind Sukumaran-Rajam, Eran Tal, Mike Tsai, Shreya Varshini, Richard Wareing, Olivia Wu, Xiaolong Xie, Jinghan Yang, Hangchen Yu, Tanmay Zargar, Zitong Zeng, Feixiong Zhang, Ajit

Matthews, Xun Jiao, Jiyuan Zhang, Emmanuel Menage, Truls Edvard Stokke, and Mohammed Sourouri, “Meta’s Second Generation AI Chip: Model-Chip Co-Design and Productionization Experiences,” in ACM/IEEE ISCA, 2025.

[7] Norman P Jouppi, Doe Hyun Yoon, Matthew Ashcraft, Mark Gottscho, Thomas B Jablin, George Kurian, James Laudon, Sheng Li, Peter Ma, Xiaoyu Ma, Thomas Norrie, Nishant Patil, Sushma Prasad, and Cliff Young, “Ten Lessons From Three Generations Shaped Google’s TPUv4i,” in ACM/IEEE ISCA, 2021.

[8] Xinwei Fu, Zhen Zhang, Haozheng Fan, Guangtai Huang, Mohammad El-Shabani, Randy Huang, Rahul Solanki, Fei Wu, Ron Diamant, and Yida Wang, “Distributed Training of Large Language Models on AWS Trainium,” in ACM SoCC, 2024.

[9] Rani Borkar, Andrew Wall, Prasanth Pulavarthi, and Yuan Yu. (Accessed: 2026-03-31) Azure Maia for the era of AI: From silicon to software to systems. [Online]. Available: https://azure.microsoft.com/en-us/blog/azu re-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/

[10] Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu, “Ascend: A Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing,” in IEEE HPCA, 2021.

[11] Yang Jiao, Liang Han, and Xin Long, “Hanguang 800 NPU—The Ultimate AI Inference Solution for Data Centers,” in IEEE HotChips, 2020.

[12] Jian Ouyang, Mijung Noh, Yong Wang, Wei Qi, Yin Ma, Canghai Gu, SoonGon Kim, Ki-il Hong, Wang-Keun Bae, Zhibiao Zhao, Jing Wang, Peng Wu, Xiaozhang Gong, Jiaxin Shi, Hefei Zhu, and Xueliang Du, “Baidu Kunlun: An AI Processor for Diversified Workloads,” in IEEE HotChips, 2020.

[13] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy, “Deep Learning Recommendation Model for Personalization and Recommendation Systems,” arXiv, vol. abs/1906.00091, 2019. [Online]. Available: https://arxiv.org/abs/1906.00091

[14] MLCommons. (Accessed: 2026-03-31) DLRMv3: Generative recommendation benchmark in MLPerf Inference. [Online]. Available: https://mlcommons.org/2026/02/dlrmv3-inference-meta/

[15] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Akkaya, Yoshihisa Moon, Lei Saleski, Jianyang Yang, Wen Zheng, and Joaquin Quiñonero Candela, “Practical Lessons from Predicting Clicks on Ads at Facebook,” in Proceedings of the Eighth International Workshop on Data Mining for Online Advertising, 2014.

[16] Dheevatsa Mudigere, Yuchen Hao, Jianyu Huang, Zhihao Jia, Andrew Tulloch, Srinivas Sridharan, Xing Liu, Mustafa Ozdal, Jade Nie, Jongsoo Park, Liang Luo, Jie (Amy) Yang, Leon Gao, Dmytro Ivchenko, Aarti Basant, Yuxi Hu, Jiyan Yang, Ehsan K. Ardestani, Xiaodong Wang, Rakesh Komuravelli, Ching-Hsiang Chu, Serhat Yilmaz, Huayu Li, Jiyuan Qian, Zhuobo Feng, Yinbin Ma, Junjie Yang, Ellie Wen, Hong Li, Lin Yang, Chonglin Sun, Whitney Zhao, Dimitry Melts, Krishna Dhulipala, KR Kishore, Tyler Graf, Assaf Eisenman, Kiran Kumar Matam, Adi Gangidi, Guoqiang Jerry Chen, Manoj Krishnan, Avinash Nayak, Krishnakumar Nair, Bharath Muthiah, Mahmoud khorashadi, Pallab Bhattacharya, Petr Lapukhov, Maxim Naumov, Ajit Mathews, Lin Qiao, Mikhail Smelyanskiy, Bill Jia, and Vijay Rao, “Software-Hardware Co-Design for Fast and Scalable Training of Deep Learning Recommendation Models,” in ACM/IEEE ISCA, 2022.

[17] Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijun Zhu, Peng Wu, Ke Wang, and Youlong Cheng, “Monolith: Real Time Recommendation System With Collisionless Embedding Table,” in ACM RecSys, 2022.

[18] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah, “Wide & Deep Learning for Recommender Systems,” in Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS), 2016.

[19] Guorui Zhou, Xiaoqiang Zhu, Chengru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai, “Deep Interest Network for Click-Through Rate Prediction,” in ACM KDD, 2018.

[20] Udit Gupta, Carole-Jean Wu, Xiaodong Wang, Maxim Naumov, Brandon Reagen, David Brooks, Bradford Cottel, Kim Hazelwood, Mark Hempstead, Bill Jia, Hsien-Hsin S. Lee, Andrey Malevich, Dheevatsa Mudigere,

Mikhail Smelyanskiy, Liang Xiong, and Xuan Zhang, “The Architectural Implications of Facebook’s DNN-Based Personalized Recommendation,” in IEEE HPCA, 2020.

[21] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi, “Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,” in ICML, 2024.

[22] Dmytro Ivchenko, Dennis Van Der Staay, Colin Taylor, Xing Liu, Will Feng, Rahul Kindi, Anirudh Sudarshan, and Shahin Sefati, “TorchRec: A PyTorch Domain Library for Recommendation Systems,” in ACM RecSys, 2022.

[23] Xin Zhang, Quanyu Zhu, Liangbei Xu, Zain Huda, Wang Zhou, Jin Fang, Dennis van der Staay, Yuxi Hu, Jade Nie, Jiyan Yang, and Chunzhi Yang, “Two-dimensional Sparse Parallelism for Large Scale Deep Learning Recommendation Model Training,” arXiv, vol. abs/2508.03854, 2025. [Online]. Available: https://arxiv.org/abs/2508.03854

[24] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei, “Scaling Laws for Neural Language Models,” arXiv, vol. abs/2001.08361, 2020. [Online]. Available: https://arxiv.org/abs/2001.08361

[25] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre, “Training Compute-Optimal Large Language Models,” arXiv, vol. abs/2203.15556, 2022. [Online]. Available: https://arxiv.org/abs/2203.15556

[26] Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Shen Li, Yanli Zhao, Yuchen Hao, Yantao Yao, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, and Wenlin Chen, “Wukong: Towards a Scaling Law for Large-Scale Recommendation,” in ICML, 2024.

[27] Samuel Williams, Andrew Waterman, and David Patterson, “Roofline: An Insightful Visual Performance Model for Multicore Architectures,” Communications of the ACM, vol. 52, no. 4, pp. 65–76, 2009.

[28] Dan Rabinovitsj and Omar Baldonado. (Accessed: 2026-03-31) Meta’s open AI hardware vision. [Online]. Available: https://engineering.fb.com /2024/10/15/data-infrastructure/metas-open-ai-hardware-vision

[29] AMD. (Accessed: 2026-03-31) hipBLASLt: General Matrix Multiplication Library for AMD GPUs. [Online]. Available: https://rocm.docs.amd.com/projects/hipBLASLt/

[30] David E. Tanner, “Tensile: Auto-Tuning GEMM GPU Assembly for All Problem Sizes,” in 2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2018.

[31] AMD, “Introducing AMD CDNA™ 4 Architecture,” Advanced Micro Devices, Inc., White Paper, June 2025. [Online]. Available: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-doc s/white-papers/amd-cdna-4-architecture-whitepaper.pdf

[32] Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, and Alex Underwood, “tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection,” arXiv, vol. abs/2512.04226, 2025. [Online]. Available: https://arxiv.org/abs/2512.04226

[33] Open Compute Project. (Accessed: 2026-03-31) OAI OAM Base Specification, Revision 2.0. [Online]. Available: https://www.opencomput e.org/documents/oai-oam-base-specification-r2-0-v1-0-20230919-pdf

[34] Jayacharan Kolla, Pedram Alizadeh, and Gilbert Lee. (Accessed: 2026-03-31) Understanding RCCL Bandwidth and xGMI Performance on AMD Instinct™ MI300X. [Online]. Available: https://rocm.blogs.amd .com/software-tools-optimization/mi300x-rccl-xgmi/README.html

[35] Vineet Gupta, Tomer Koren, and Yoram Singer, “Shampoo: Preconditioned Stochastic Tensor Optimization,” in ICML, 2018.

[36] Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat, “A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale,” arXiv, vol. abs/2309.06497, 2023. [Online]. Available: https://arxiv.org/abs/2309.06497

[37] Diederik P. Kingma and Jimmy Ba, “Adam: A Method for Stochastic Optimization,” in ICLR, 2015.

[38] Roger S. Martin, Christian Reinsch, and James H. Wilkinson, “Householder’s Tridiagonalization of a Symmetric Matrix,” Numerische Mathe matik, vol. 11, no. 3, pp. 181–195, Mar. 1968.

[39] Jan J. M. Cuppen, “A Divide and Conquer Method for the Symmetric Eigenproblem,” Numerische Mathematik, vol. 36, pp. 177–195, June 1980.

[40] Jackie (Jiaqi) Xu, Chao Chen, Shuqi Yang, Markus Hoehnerbach, Xiaoyi (Leo) Liu, Ted Zadouri, Dev (Devashish) Shankar, Jacky Zhou, Hongtao Yu, Manman Ren, Han Xu, Chunzhi Yang, Jade Nie, Haoyu Zhang, Huayu Li, Michael Shu, Musharaf Sultan, Max Leung, John Bocharov, and Tri Dao. (Accessed: 2026-03-31) Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels. [Online]. Available: https://pytorch.org/blog/generalized-dot-product-att ention-tackling-real-world-challenges-in-gpu-training-kernels/

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, “Attention Is All You Need,” in NeurIPS, 2017.

[42] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” in NeurIPS, 2022.

[43] Bojian Hou, Xiaolong Liu, Xiaoyi Liu, Jiaqi Xu, Yasmine Badr, Mengyue Hang, Sudhanshu Chanpuriya, Junqing Zhou, Yuhang Yang, Han Xu, Qiuling Suo, Laming Chen, Yuxi Hu, Jiasheng Zhang, Huaqing Xiong, Yuzhen Huang, Chao Chen, Yue Dong, Yi Yang, Shuo Chang, Xiaorui Gan, Wenlin Chen, Santanu Kolay, Darren Liu, Jade Nie, Chunzhi Yang, Ellie Wen, Jiyan Yang, and Huayu Li, “Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design,” arXiv, vol. abs/2602.10016, 2026. [Online]. Available: https://arxiv.org/abs/2602.10016

[44] Zhichen Zeng, Xiaolong Liu, Mengyue Hang, Xiaoyi Liu, Qinghai Zhou, Chaofei Yang, Yiqun Liu, Yichen Ruan, Laming Chen, Yuxin Chen, Yujia Hao, Jiaqi Xu, Jade Nie, Xi Liu, Buyun Zhang, Wei Wen, Siyang Yuan, Hang Yin, Xin Zhang, Kai Wang, Wen-Yen Chen, Yiping Han, Huayu Li, Chunzhi Yang, Bo Long, Philip S. Yu, Hanghang Tong, and Jiyan Yang, “InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction,” arXiv, vol. abs/2411.09852, 2025. [Online]. Available: https://arxiv.org/abs/2411.09852

[45] Meta Engineering. (Accessed: 2026-03-31) Meta’s Generative Ads Model (GEM): The Central Brain Accelerating Ads Recommendation AI Innovation. [Online]. Available: https://engineering.fb.com/2025/11/10/ ml-applications/metas-generative-ads-model-gem-the-central-brain-acc elerating-ads-recommendation-ai-innovation/

[46] Daya Khudia, Jianyu Huang, Protonu Basu, Summer Deng, Haixin Liu, Jongsoo Park, and Mikhail Smelyanskiy, “FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference,” arXiv, vol. abs/2101.05615, 2021. [Online]. Available: https://arxiv.org/abs/2101.056 15

[47] Gang Liao, Hongsen Qin, Ying Wang, Alicia Golden, Michael Kuchnik, Yavuz Yetim, Jia Jiunn Ang, Chunli Fu, Yihan He, Samuel Hsia, Zewei Jiang, Dianshi Li, Uladzimir Pashkevich, Varna Puvvada, Feng Shi, Matt Steiner, Ruichao Xiao, Nathan Yan, Xiayu Yu, Zhou Fang, Roman Levenstein, Kunming Ho, Haishan Zhu, Alec Hammond, Richard Li, Ajit Mathews, Kaustubh Gondkar, Abdul Zainul-Abedin, Ketan Singh, Hongtao Yu, Wenyuan Chi, Barney Huang, Sean Zhang, Noah Weller, Zach Marine, Wyatt Cook, Carole-Jean Wu, and Gaoxiang Liu, “KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta,” in ACM/IEEE ISCA, 2026.

[48] AMD. (Accessed: 2026-03-31) ROCprofiler-SDK Documentation. [Online]. Available: https://rocm.docs.amd.com/projects/rocprofiler-sdk

[49] Triton Developers. (2024) Proton: A Lightweight Profiler for Triton. Part of the Triton language and compiler project. [Online]. Available: https://github.com/triton-lang/triton/tree/main/third\_party/proton

[50] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Harshit Khaitan, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin,

Gordon MacKean, Adriana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon, “In-Datacenter Performance Analysis of a Tensor Processing Unit,” in ACM/IEEE ISCA, 2017.

[51] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep Residual Learning for Image Recognition,” in CVPR, 2016.

[52] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger, “Densely Connected Convolutional Networks,” in CVPR, 2017.

[53] Andrej Karpathy. (Accessed: 2026-03-31) nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. [Online]. Available: https://github.com/karpathy/nanoGPT

[54] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li, “PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel,” Proceedings of the VLDB Endowment, vol. 16, no. 12, pp. 3848–3860, Aug. 2023.

[55] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He, “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,” in SC, 2020.

[56] Meta Platforms, Inc., “Meta AI Acceleration in the Next-Gen Meta MTIA for Recommendation Inference,” in IEEE HotChips, 2024.

[57] Boxun Xu, Richard Boone, and Peng Li, “SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks,” arXiv, vol. abs/2505.12292, 2026. [Online]. Available: https://arxiv.org/abs/2505.12292

[58] Cyrus Zhou, Pedro Savarese, Vaughn Richard, Zack Hassman, Xin Yuan, Michael Maire, Michael DiBrino, and Yanjing Li, “SySMOL: Co-designing Algorithms and Hardware for Neural Networks with Heterogeneous Precisions,” arXiv, vol. abs/2311.14114, 2024. [Online]. Available: https://arxiv.org/abs/2311.14114

[59] Hyoukjun Kwon, Prasanth Chatarasi, Michael Pellauer, Angshuman Parashar, Vivek Sarber, and Tushar Krishna, “Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach,” arXiv, vol. abs/1805.02566, 2019. [Online]. Available: https://arxiv.org/abs/1805.02566

[60] Quentin Anthony, Yury Tokpanov, Skyler Szot, Srivatsan Rajagopal, Praneeth Medepalli, Anna Golubeva, Vasu Shyam, Robert Washbourne, Rishi Iyer, Ansh Chaurasia, Tomas Figliolia, Xiao Yang, Abhinav Sarje, Drew Thorstensen, Amartey Pearson, Zack Grossbart, Jason van Patten, Emad Barsoum, Zhenyu Gu, Yao Fu, and Beren Millidge, “Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design,” arXiv, vol. abs/2511.17127, 2025. [Online]. Available: https://arxiv.org/abs/2511.17127

[61] Maxim Naumov, John Kim, Dheevatsa Mudigere, Srinivas Sridharan, Xiaodong Wang, Whitney Zhao, Serhat Yilmaz, Changkyu Kim, Hector Yuen, Mustafa Ozdal, Krishnakumar Nair, Isabel Gao, Bor-Yiing Su, Jiyan Yang, and Mikhail Smelyanskiy, “Deep Learning Training in Facebook Data Centers: Design of Scale-up and Scale-out Systems,” arXiv, vol. abs/2003.09518, 2020. [Online]. Available: https://arxiv.org/abs/2003.09518

[62] Geet Sethi, Bilge Acun, Niket Agrawal, Christos Kober, Zhengxing Zhang, Dheevatsa Mudigere, Krishnakumar Nair, Maxim Smelyanskiy, Carole Jean Wu, and Kim Hazelwood, “RecShard: Statistical Feature-Based Memory Optimization for Industry-Scale Neural Recommendation,” in MLSys, 2022.

[63] Shiwei Li, Zhuoqi Hu, Xing Tang, Haozhao Wang, Shijie Xu, Weihong Luo, Yuhua Li, Xiuqiang He, and Ruixuan Li, “Mixed-Precision Embeddings for Large-Scale Recommendation Models,” arXiv, vol. abs/2409.20305, 2024. [Online]. Available: https://arxiv.org/abs/2409.203 05