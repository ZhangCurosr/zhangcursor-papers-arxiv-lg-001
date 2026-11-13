# Infrastructure for Verifiable AI: Where Proof Costs Go and How Hardware Can Help

Yihan Xia, Wenxin Xu, and Taotao Wang

Abstract—Users of artificial intelligence services, including those powered by large language models and other transformerbased models, need assurance that inference and training follow the declared computation without exposing private models or data. Zero-knowledge proofs can provide this assurance, but proof generation remains expensive. Hardware acceleration research often reports gains for individual cryptographic operations, leaving their effect on the full cost of proving neuralnetwork workloads, particularly large transformer models, unclear. We introduce a four-layer framework linking verifiable AI workloads, proof systems, cryptographic kernels and hardware platforms, use it to review the literature, and add original measurements of six provers and seven kernels to identify which costs matter in practice. Our measurements show that the dominant cost changes with the proof system and input length. In low-rank fine-tuning, the cost of opening commitments to fixed model weights remains nearly constant across input lengths from 64 to 2048, while the work associated with intermediate values grows. A 13-billion-parameter fine-tuning configuration exceeds the capacity of a 40-gigabyte graphics processor but completes on an 80-gigabyte device, showing why capacity must be assessed alongside execution time. Our kernel profiling further identifies instruction issue and synchronization limits that a bandwidthonly model misses. We use these findings to assess graphics processors, reconfigurable logic, custom chips and existing artificial intelligence accelerators. The resulting research agenda connects proof-friendly numeric formats, reuse of fixed-weight commitments, heterogeneous execution and shared benchmarks to the measured bottlenecks. Effective acceleration requires selecting hardware and proof-system optimizations around the cost profile of the target workload.

Index Terms—Verifiable AI, zero-knowledge proofs, hardware acceleration, large language models, trustworthy machine learning.

## I. INTRODUCTION

RTIFICIAL intelligence, including large language models and other transformer-based models, is increasingly consumed as a service. Users send prompts to an inference interface, upload private data to a fine-tuning service, or rely on a provider’s claim that training followed a declared procedure. In each case the computation happens on someone else’s hardware, and the user sees only the result. This leaves room for quiet deviations that are hard to detect from the outside: an inference provider may serve a smaller or more aggressively quantized model than the one it advertises [1], [2]; a fine-tuning service may skip part of the user’s data or use different hyper-parameters [3], [4]; a training pipeline

may omit declared preprocessing, optimization, or privacypreserving steps [5], [6]. Detecting such deviations has become an active research topic, with approaches ranging from activation fingerprints [1], [2] to trusted hardware [3], [7] and replicated execution [8].

Zero-knowledge proofs provide cryptographic assurance about a precisely specified computation. As illustrated in Fig. 1, a provider attaches a short proof to its answer, and anyone can check the proof in milliseconds to seconds. A valid proof certifies that the encoded relation holds under the proof system’s security assumptions, while hiding the designated private inputs. When that relation binds the model, input and computation, it certifies the requested inference or training task. Unlike trusted hardware, the guarantee rests on cryptographic assumptions rather than on trust in a chip vendor, and the proof can be verified publicly. Systems now exist for proving inference of large language models [9]– [12], fine-tuning steps [4], [13], and differentially private training [5]. The obstacle is cost on the prover side. Proving a single forward pass of a 13-billion-parameter model takes close to 15 minutes [9]; proving one fine-tuning step with lowrank adaptation for models of 3 to 13 billion parameters takes two to four minutes on a data-center graphics processing unit (GPU), about three orders of magnitude longer than the step itself [13]; and a state-of-the-art prover for whole generated sequences reports 86–174 tokens per minute on a single machine [11], illustrating the cost of certifying generation. Verification is cheap; proving is the bottleneck. Reducing this cost requires hardware acceleration together with proof-system and implementation choices.

There is a substantial body of work on accelerating zero-knowledge provers on GPUs, field-programmable gate arrays (FPGAs) and application-specific integrated circuits (ASICs) [14]. Most of it, however, targets individual cryptographic kernels—above all multi-scalar multiplication (MSM) and the number-theoretic transform (NTT)—and reports speed-ups on generic benchmarks [15]–[19]. A kernel speedup alone does not establish an end-to-end gain: its value depends on the kernel’s share of total proving time and on the costs of feeding it data. Proofs of neural networks combine interactive summation checks (the sumcheck protocol) for matrix multiplications, lookup arguments for non-linear functions, commitments to and openings of billions of fixed weights, and recomputation of every intermediate activation over a finite field [9], [11], [13]. Evidence from GPU studies points the same way: once one kernel is accelerated, the bottleneck moves to the next one—from MSM to NTT [20], or from sumcheck to the polynomial commitment [21]. Moreover, hardware acceleration work has rarely validated its designs on AI workloads: only three of the twenty representative accelerators reviewed in this article report an evaluation on a machine learning model or circuit [22]–[24], all on convolutional networks.

![](images/6bdfb231457a3adc49dd84343001d76cb7bf1f7b75967fe917098bf540c1c1d8.jpg)  
Fig. 1. Verifiable AI as a service. The provider attaches a zero-knowledge proof to each result; verification is cheap, while proof generation dominates the cost.

This article treats proof generation as a layer of AI infrastructure alongside training clusters, inference serving and data pipelines. We organize the literature with a workloadoriented framework and add original measurements of six provers and of the seven cryptographic kernels they rely on. Our measurements show that (i) no single cost dominates, as the largest share shifts with the proof system and the input length (the number of tokens in the context window); (ii) the cost of opening fixed weights does not grow with the input length; and (iii) memory capacity limits what can be proved before data movement limits how fast. The contributions of this article are as follows.

Four-layer framework (Section III): We decompose verifiable AI into four layers, from verifiable AI workloads (Layer 1) through proof systems (Layer 2) and cryptographic kernels (Layer 3) to hardware platforms (Layer 4), identify the connections between components of adjacent layers, analyze where the proving cost arises at these connections, and highlight four structural properties of AI workloads that hardware can exploit to reduce it.

• Cross-system measurements (Section IV): We measure cost breakdowns for inference, low-rank fine-tuning and small-scale differentially private training, supplemented by block-level comparisons and CPU/GPU kernel microbenchmarks, and show how the dominant cost changes across implementations and scales and how memory capacity constrains feasible workloads.

• Workload-aware hardware review (Section V): We review GPU, FPGA, ASIC and AI-accelerator designs, assess the fit between each kernel and each platform, and identify where accelerator evaluations fall short of AI workloads.

• Open problems and roadmap (Section VI): We set out seven open problems for making verifiable AI practical, from proof-friendly numeric formats to commitment on the network data path, and order them by time horizon.   
Existing surveys provide complementary views of this de-

sign space. A recent survey of zk-SNARK hardware accelerators [14] covers generic proof systems without an AI workload perspective or end-to-end measurements. A survey of zero-knowledge proof applications [25] compares CPU, GPU, FPGA and ASIC platforms only briefly. A broad survey of zero-knowledge verifiable machine learning and a focused perspective on zero-knowledge federated learning [26], [27] organize protocols, systems and application roles. This article complements that coverage with cross-system cost measurements and a workload-oriented assessment of hardware acceleration. Systematizations of private inference [28], [29] address confidentiality through homomorphic encryption and multi-party computation, not verifiability. A systematization of hash-based polynomial commitments [30] studies their soundness, whereas we discuss the choice of commitment scheme from the hardware and performance side.

## II. VERIFIABLE AI: WHAT IS PROVED AND WHY IT IS EXPENSIVE

## A. Three Classes of Verifiable AI Tasks

Table I summarizes three classes of existing zero-knowledge systems for AI, grouped by what the proof certifies. Verifiable inference proves that an output was produced by a committed model on a given input. Systems such as zkLLM and zkGPT prove one forward pass [9], [10]. Recent systems prove an entire generated sequence at once, avoiding repeated proofs of the growing context for every new token [11], [12].

Verifiable training and fine-tuning proves that each update step—forward pass, backward pass and parameter update— was executed on the committed data and weights. Parameterefficient fine-tuning is the natural starting point because only a small adapter changes while the large base model stays fixed [4], [13]. Per-step proofs can be folded into a single certificate for a whole training run [4].

Training audit and privacy proofs certify a property of training rather than every arithmetic operation. A proof of differentially private stochastic gradient descent shows, step by step, that per-sample gradients were clipped and the declared noise was added [5]. A proof of training can instead certify that a committed checkpoint reaches a claimed objective value on inputs chosen by an auditor, at a cost independent of the number of training steps [31].

## B. Zero-Knowledge Proofs in a Nutshell

To prove a computation, it is first written as a set of algebraic constraints over a finite field—for example, integers modulo a prime. A matrix multiplication becomes many multiply-and-add constraints; a softmax becomes a combination of arithmetic constraints and table lookups. The prover then convinces the verifier that it knows values for all intermediate variables (the witness) that satisfy every constraint, without revealing them. Fig. 2 illustrates three building blocks that recur in these proofs.

Commitment. A commitment locks a large piece of data— a weight matrix, a layer’s activations—into a short fingerprint. Afterwards the prover cannot change the data, yet it can later open the commitment at chosen positions and prove that the revealed values are consistent with the fingerprint. Hash-based commitments build a Merkle tree over an encoded version of the data [33]–[35]; elliptic-curve-based commitments compute the fingerprint as a weighted sum of millions of curve points, an operation called multi-scalar multiplication (MSM) [36], [37].

TABLE I  
THREE CLASSES OF VERIFIABLE-AI TASKS AND REPRESENTATIVE ZERO-KNOWLEDGE SYSTEMS. ALL COSTS ARE AS REPORTED BY THE AUTHORS, ON THE STATED HARDWARE, AT DIFFERENT SECURITY LEVELS, FIELDS, QUANTIZATION SETTINGS AND INPUT LENGTHS, AND ARE not DIRECTLY COMPARABLE; VALUES IN PARENTHESES ARE NORMALIZED PER TOKEN OR PER STEP (NOTES BELOW). SECTION IV REPORTS MEASURED COST SHARES UNDER A COMMON TAXONOMY.
<table><tr><td>Task</td><td>What the proof certifies</td><td>System</td><td>Model / scale</td><td>Reported cost (normalized)</td><td>Platform</td></tr><tr><td>Verifiable inference</td><td>The output was produced by the committed model on the given input (one forward pass or a whole</td><td>zkLLM [9]</td><td>up to 13B</td><td>&lt;15 min per 2048-token forward pass (≈0.39 s/tokena); proof &lt;200kB</td><td>1×A100 (40 GB)</td></tr><tr><td></td><td></td><td>zkGPT [10]</td><td>GPT-2</td><td>21.8 s per inference (≈0.68 s/tokenb)</td><td>CPU server (Xeon 6126)</td></tr><tr><td></td><td></td><td>DeepProve [11]</td><td>GPT-2 124M. Gemma 3270M</td><td>174 / 86 tokens per minute, whole sequence (0.34–0.47  / 0.69–0.94 s/tokenc); verify 1–3.7 s</td><td>CPU servers (EPYC 9254; Ryzen 9)</td></tr><tr><td></td><td></td><td>zkAgent [12]</td><td>GPT-2, 512 tokens</td><td>≈0.30 s per generated tokend</td><td>CPU server (Xeon 6982P-C)</td></tr><tr><td></td><td></td><td>Attestable*</td><td>[32] Gemma 431B</td><td>proof 4.4–7.9 MiB</td><td>53 tokens/s, batch 1 (0.019 s/token); 1×H100</td></tr><tr><td>Verifiable training / fine-tuning</td><td>Each update step (forward, backward, update) ran on the committed data and</td><td>VeriLoRA [13]</td><td>3B-13B (LoRA)</td><td>122–249 s per step (≈103 × plain step); commitment 156–554 s</td><td>1×A100 (80 GB)</td></tr><tr><td></td><td>weights</td><td>VFT [4]</td><td>n/r (LoRA r=8/16, 2048 tokens)</td><td>16.8–31.2 s per step; verify &lt;200ms</td><td>2–4×80 GB GPUs</td></tr><tr><td>Training audit and DP proofs</td><td>Each DP-SGD step clipped per-sample gradients and added the declared noise</td><td>VeriDP [5]</td><td>up to 11.2M params</td><td>0.34–8.5 s per sample; memory is the main limit</td><td>1 CPU core</td></tr><tr><td></td><td>A committed checkpoint attains the claimed objective on auditor-chosen inputs</td><td>zkLLMPoT [31]</td><td>1.1B-13B</td><td>covered operators of one 512-token forward audit: 41–59 s (1.1–1.5B), 131 s (13B) (0.08–0.26 s/token); verify &lt;0.5 s</td><td>1×A100 (40 GB)</td></tr></table>

<sup>∗</sup>Vendor blog post, not peer-reviewed. n/r: not reported. <sup>a</sup>803 s per 2048-token prompt pass of the 13B model. <sup>b</sup>Input length inferred from matrix dimensions (32×768), not stated. <sup>c</sup>Peak tokens per minute (TPM) of GPT-2 / Gemma 3 on the faster of two reported machines; per token = 60/TPM over both machines. <sup>d</sup>v3 abstract; the earlier v2 paper reports 0.40 s/token for the same setting.

![](images/558b35d4afdd122115c2de5be666888c5956f25108926c5bef420bf2a514b120.jpg)  
Fig. 2. The three building blocks that dominate proofs of neural networks.

Sumcheck. Many claims about a computation take the form “this sum over a huge number of terms equals v”. An entry of a matrix product is exactly such a sum. The sumcheck protocol [38] reduces a sum over 2<sup>n</sup> terms to n short rounds; in each round the verifier sends a random challenge and the prover answers with a low-degree polynomial, so a cheating prover is caught with overwhelming probability. For the prover, each round is a pass over all remaining terms that halves the problem size—simple arithmetic, but over a lot of data. GKR-style systems chain sumchecks layer by layer through a whole network [39].

Lookup. Non-linear functions such as GELU, softmax exponentials or layer-norm square roots are expensive to express as arithmetic. Instead, all valid input–output pairs are placed in a pre-committed table, and a lookup argument proves that every pair used in the computation appears in the table [9], [40]. The cost grows with the number of lookups and the size of the table. Two further kernels appear depending on the proof system. The number-theoretic transform (NTT) is the finite-field analog of the fast Fourier transform and converts between polynomial representations in univariate systems such as Plonk [41]. Hashing—with arithmetic-friendly functions such as Poseidon2 [42] or bit-oriented ones such as BLAKE3 [43]—builds the Merkle trees used by hash-based commitments.

## C. Why Proving AI Is Expensive

Four factors combine to make the prover several orders of magnitude slower than the computation it proves.

Finite-field arithmetic instead of low-precision floating point. AI hardware is built for 16-bit and 8-bit floatingpoint or integer matrix multiplication. Proofs operate on field elements that are either about 256 bits wide (for elliptic-curvebased systems) or 31 to 64 bits wide (for recent small-field systems [44]). A single 256-bit modular multiplication requires multiple lower-width integer operations, and GPU provers run almost entirely on the integer pipeline [20]; the tensor cores that power AI were left unused until very recently [45]. In the measured provers, models are quantized to integer or fixedpoint representations before their computations are expressed over a field.

The witness is much larger than the model. A proof must account for every intermediate value: all activations, every lookup query and its multiplicity, and, for training, all gradients. The witness of a fine-tuning step therefore exceeds the model size by a large factor: proving one low-rank finetuning step of a 7-billion-parameter model allocates about 2.9 TB of GPU memory in total across layers, which fits on a single 80 GB GPU only because layers are processed one at a time [13].

Commitments touch all of this data. Every committed tensor must be encoded and hashed, or multiplied into an MSM. For hash-based schemes, committing to and proving a single polynomial with $2 ^ { 2 8 }$ coefficients already involves about 1.1 GB of data and takes 463 s on a high-end server CPU [46].

Data movement. At larger model sizes or input lengths, the witness can exceed accelerator memory. It is generated, moved between host and device, committed, and later reloaded to answer opening queries, so memory capacity, and with it the transfers needed to work around it, become first-order constraints alongside arithmetic [22], [47].

## D. Other Routes to Verifiability

Zero-knowledge proofs are not the only way to check an AI service. Trusted execution environments (TEE) [3], [7] are fast but require trusting the hardware vendor and the enclave implementation. Replicated execution with refereed disputes [8], made reproducible by removing hardware nondeterminism [48], needs only one honest provider among several but is not publicly verifiable by a single party. Activation fingerprints [2], learned proxy tasks [1] and checkpoint replay in proof-of-learning [6] are cheap but give statistical rather than cryptographic guarantees. This article focuses on cryptographic proofs because they combine public verifiability with independence from hardware trust—and because their cost is the problem that hardware can most directly address.

## III. A FOUR-LAYER FRAMEWORK FROM VERIFIABLE AI WORKLOADS TO HARDWARE

Hardware choices for verifiable AI are hard to reason about because the cost of a proof is determined by a chain of decisions: which workload is proved, which proof system expresses it, which kernels that proof system spends its time in, and which hardware runs those kernels well. We organize these decisions into four layers and examine the connections between adjacent layers, each of which introduces a bottleneck. Fig. 3 shows the four layers, the connections between them and the bottleneck introduced at each connection.

## A. The Four Layers

Layer 1: verifiable AI workloads. A workload is the computation that the prover must process to certify a verifiable AI task of Section II-A. We distinguish four kinds: inference (a single prefill or a whole generated sequence), fine-tuning (forward, backward and update), training audit, and differential-privacy proofs. They differ in size, in how much of the computation repeats, and in which parts of the model stay fixed.

Layer 2: proof systems. We distinguish four families. GKR/sumcheck-based systems prove layered computations through chained sumchecks and are paired with a separate polynomial commitment such as Hyrax or a KZG variant [36], [37], [39]; most large-model provers belong here [9], [11], [13]. Plonkish systems compile the computation into a circuit with custom gates; the original design relies on univariate polynomials and the NTT [41], while its multilinear successor replaces the NTT with sumcheck [49]. Hash-based systems use error-correcting codes and Merkle trees for commitments, avoid elliptic curves, and are plausibly post-quantum [33]– [35], [44]. Lattice-based systems are an emerging postquantum alternative [50]. The interactive part of a proof and the commitment scheme are largely independent choices, which is why the same sumcheck-based prover can be paired with very different commitments.

Layer 3: cryptographic kernels. Seven kernels account for most prover time: MSM, NTT, sumcheck, lookup, hashing with Merkle-tree construction, encoding with linear or Reed– Solomon codes, and the underlying finite-field multiplication.

Layer 4: hardware platforms. CPUs serve as the baseline. GPUs are the workhorse of current provers. FPGAs offer customizable streaming pipelines, ASICs specialized datapaths, AI accelerators such as TPUs massive integer matrix throughput, and smart network interface cards (SmartNICs) computation on the data path.

## B. Connections Between Layers and Their Bottlenecks

From workloads to proof systems (B1). The structure of neural networks steers the choice of proof system. Matrix multiplication, which dominates both inference and training, is a sum of products and can be proved directly with sumcheck, at a prover cost linear in the matrix size [9]. Non-linear operations are proved with lookup arguments. The price is the witness of Section II-C, recomputed over the field and later committed, so the first bottleneck is witness blow-up and field recomputation, which exists before any cryptographic kernel runs.

![](images/15b0dc3acdf8868cd1928ad2eec2abe3a12144d38def370c39ddc45eef9bf562.jpg)  
Fig. 3. Four-layer framework. Each verifiable AI workload (Layer 1) is expressed in a proof system (Layer 2), whose cost is concentrated in a few cryptographic kernels (Layer 3) that run with different efficiency on hardware platforms (Layer 4). Kernel colors give the limiting resource as measured on a GPU (Section IV-B); B1–B4 mark the bottleneck introduced at each connection between layers; S1–S4 mark where structure specific to AI workloads can be exploited.

From proof systems to kernels (B2). Each proof system concentrates its cost in a few kernels. Elliptic-curve commitments spend their time in MSM; hash-based commitments in encoding and hashing; Plonk-style provers additionally in NTT; sumcheck and lookup arguments in passes over large tables. For AI workloads the commitment side can dominate end-to-end time, because the witness is large and the weights must be opened again and again, and once sumcheck is accelerated on a GPU the commitment becomes the next bottleneck [21]. Whether it dominates in practice depends on the proof system and the input length, as Section IV-C shows. The second bottleneck is commitment and opening.

From kernels to hardware (B3, B4). Kernels differ in the resource that limits them. MSM consists of many independent elliptic-curve additions with high arithmetic intensity and is compute-bound; it parallelizes well on GPUs and ASICs [16], [19]. Sumcheck, lookup and encoding stream over large tables with little data reuse, which the literature treats as bandwidthbound; whether sumcheck or NTT is the better primitive depends on polynomial degree and on-chip memory [51]. Hashing consists of permutation rounds and bitwise operations and suits deep streaming pipelines. Our micro-benchmarks (Section IV-B) refine this picture: on a GPU, encoding and element-wise field multiplication are indeed memory-bound, but low-degree sumcheck reaches the integer-instruction issue limit, while lookup is limited by synchronization and instruction issue. Table II summarizes the resulting fit, with the limiting resource taken from those measurements. The third bottleneck is a mismatch: most accelerators were designed for MSM and NTT, while AI provers also incur substantial lookup, sumcheck and witness-generation costs, as measured in Section IV-C. The fourth is the platform itself: memory capacity and data movement, because model parameters, input length and repeated weight reloading together exceed what a single device holds.

## C. Structure Specific to AI Workloads

AI workloads expose regularities beyond those represented by isolated circuit benchmarks. Fig. 3 summarizes four such properties (S1–S4), which provide opportunities for reuse, batching and coordinated memory management.

S1: weights are fixed. The base model does not change across inference requests, and during parameter-efficient finetuning it does not change across steps. Its commitment can be computed once and reused—existing provers already separate a one-off weight commitment from per-query proving [9], [11]—and the many openings that each request or step requires can be merged into fewer, larger ones [61], [62]. This shifts the cost from repeated commitment to efficient opening.

TABLE II  
FIT BETWEEN CRYPTOGRAPHIC KERNELS AND HARDWARE PLATFORMS. RATINGS ARE QUALITATIVE AND CITE REPRESENTATIVE WORKS. THE Bound by COLUMN GIVES THE RESOURCE THAT LIMITS EACH KERNEL ON A GPU AS MEASURED IN SECTION IV-B (“INT. ISSUE”: INTEGER-INSTRUCTION ISSUE); FIG. 4 SHOWS MEASURED THROUGHPUT FOR THE CPU AND GPU COLUMNS.
<table><tr><td>Kernel</td><td>Bound by</td><td>CPU</td><td>GPU</td><td>FPGA</td><td>ASIC</td><td>AI acc.</td><td>SmartNIC</td><td>Main reason</td></tr><tr><td>MSM</td><td>compute</td><td>O</td><td>[16], [17], [52]  [18], [24], [53]●[15], [19] [54]</td><td></td><td></td><td></td><td></td><td>Many independent elliptic-curve additions; high arithmetic intensity,</td></tr><tr><td>NTT</td><td>memory</td><td>O</td><td>● [16], [20]</td><td>[24]</td><td>[15], [19] [54]</td><td></td><td></td><td>parallelizes well. Butterfly stages with strided access; limited by memory-issue queues and occupancy rather than DRAM bandwidth or integer</td></tr><tr><td>Sumcheck</td><td>int. issue</td><td></td><td>[21][21], [22], [45]</td><td>[55]</td><td>[55]-[58]</td><td></td><td></td><td>arithmetic. Each round streams over all evaluations with little reuse; low-degree rounds saturate integer issue before bandwidth, and the best primitive depends</td></tr><tr><td>Lookup</td><td>sync., issue</td><td>0</td><td>● [9]</td><td></td><td></td><td></td><td></td><td>memory [51]. Random table accesses plus batched field inversions (LogUp [40]); barriers in the inversion, not DRAM, cap the GPU gain.</td></tr><tr><td>Hash &amp; Merkle bit-level</td><td></td><td>0</td><td>[22]</td><td> [46]</td><td>[58], [59]</td><td>O</td><td>open</td><td>Permutation rounds and bitwise operations, limited by instruction issue rather than DRAM; tree reduction suits</td></tr><tr><td>Encoding</td><td>memory</td><td>0</td><td>[22], [45]</td><td> [46]</td><td>[59]</td><td>[54]</td><td></td><td>streaming pipelines. Sparse linear codes or Reed–Solomon (via NTT); off-chip bandwidth limits the gain [46].</td></tr><tr><td>Field mult.</td><td>memory</td><td>0</td><td></td><td></td><td></td><td>0 [54]</td><td>O</td><td>256-bit fields need multi-limb arithmetic, small fields (31/64-bit) map to native integer units; element-wise, it is memory-bound.</td></tr></table>

strong medium weak – no published evidence. Tensor cores inside GPUs [45] are counted under GPU; “AI acc.” means stand-alone AI accelerators such as G# #<sub>TPUs [54]. “open”: SmartNICs have been proposed for observing training traffic [60], but no published design hashes or commits tensors on the data path as discussed in</sub> Section VI.

S2: layers repeat. A transformer applies the same matrix shapes in every layer and for every request. Proofs of same-shape operations can be batched and pipelined to raise hardware utilization, an approach that already yields large throughput gains on GPUs [22].

S3: numeric formats are a design choice. If a model is quantized to integers or fixed-point numbers that the proof system can absorb, for example because every intermediate result, including accumulated sums of products, stays below the field modulus, the witness can be produced by the same integer matrix units that run the model, rather than by slow field arithmetic. Tensor cores [45] and TPU matrix units [54] have already been repurposed for finite-field kernels, and an industrial prover reports proving 8-bit integer matrix multiplication at large scale [32].

S4: scale sets memory pressure. Parameter count and input length determine the witness size and therefore whether a proof fits on one device. Because both are known before proving starts, the memory a workload needs can be predicted and the hardware provisioned accordingly. Proving a whole generated sequence at once [11], [12] trades redundant pertoken work for a larger single proof, which makes memory capacity and bandwidth even more decisive.

Used this way, the framework turns “which hardware should accelerate verifiable AI?” into a sequence of answerable questions: for a given workload, which proof system is used, which kernels dominate, which platforms suit them, and which AI-specific structure can reduce the work before any hardware is involved. Section IV follows this procedure with measurements on four workloads, and Section V uses it to assess existing accelerators.

## IV. WHERE THE COST GOES: MEASUREMENTS

The framework of Section III predicts where proving time should go for AI workloads. This section tests those predictions. We measure a set of workloads across publicly available provers and use their cost shares and scaling trends to identify recurring bottlenecks. Implementation differences remain part of the comparison and are accounted for when interpreting the results.

## A. Methodology

Questions. The measurements address four questions derived from the framework:

Q1 Do commitment and opening dominate proving time across proof systems (B2)?

Q2 What share of the time goes to generating the witness over the finite field (B1)?

Q3 How do the cost components scale with model size and input length, and is the cost of opening fixed weights independent of input length (S1, S4)?

Q4 When do memory capacity and data movement become the binding constraint (B4)?

Workloads. (W1) Inference over a prompt: a 7-billionparameter LLaMA-2 model with prompts of 1024 and 2048 tokens on zkLLM (shorter prompts are structurally unsupported by its code), and a 0.27-billion-parameter Gemma 3 model with prompts of 64–1024 tokens on DeepProve, which did not run the 7-billion-parameter model in any configuration we tried. (W2) Proving a whole generated sequence, on DeepProve only. (W3) One fine-tuning step with lowrank adaptation for 7- and 13-billion-parameter models at input lengths 64–2048. (W4) One step of differentially private stochastic gradient descent, at the largest scale its prover supports, far below that of W1–W3.

Provers. Table III lists the provers: public code, end-toend support for the workload, and as wide a coverage of Layer 2 as possible. Most open-source provers that reach the scale of large language models are built on sumcheck and GKR [9], [11], [13]; we found no hash-based prover that runs one end to end. We therefore add two narrower points. A GKR prover with interchangeable commitment schemes [63] runs the same transformer block with an elliptic-curve commitment (Hyrax or KZG) and a hash-based one (Orion), and a compiler built on KZG blocks and proof accumulation [64] provides a point outside the sumcheck family. All provers are used at fixed commits (Table III), with timing code added where a prover does not report its own breakdown and no change to what is proved. DeepProve [11] is run unmodified: its built-in timers already report the per-stage times that we use for the breakdown.

Cost taxonomy. To compare provers that report time in different ways, we map every timed region to one of eight categories aligned with the framework: witness generation (recomputing the model over the field, including quantization; B1), sumcheck, lookup, commitment, opening (proving evaluations of committed data; B2), hashing and encoding, data movement (host–device and disk transfers; B4), and other. Regions that cannot be split without modifying the code are reported as mixed rather than apportioned by guesswork. One-off setup—key generation and the commitment to fixed weights—is reported separately from per-query cost, because amortizing it is exactly the opportunity S1.

Metrics and kernels. For each cell we record prover time and its breakdown, peak GPU and host memory, proof size, verifier time, and the slowdown relative to the same computation in 16-bit floating point with PyTorch on the same GPU. Because the provers differ in field, security level and quantization, we compare cost shares and scaling trends, not absolute times. To connect the breakdowns to hardware, we also benchmark the seven kernels of Layer 3 in isolation, using GPU libraries [52], [65] or reference implementations, and place each on a roofline built from measured rather than vendor-specified roofs.

TABLE III  
PROVERS USED IN THE MEASUREMENTS.
<table><tr><td>Prover</td><td>Proof system / commitment</td><td>Workload</td><td>Role</td><td>Commit</td></tr><tr><td>zkLLM [9]</td><td>sumcheck, lookup; Hyrax-style (BLS12-381)</td><td>W1</td><td>full scale</td><td>993311e</td></tr><tr><td>DeepProve [11]</td><td>GKR, lookup; HyperKZG (BN254)</td><td>W1, W2</td><td>small model</td><td>9d1a53e</td></tr><tr><td>VeriLoRA [13]</td><td>sumcheck, lookup; Hyrax-style</td><td>W3</td><td>full scale</td><td>7cc0977</td></tr><tr><td>VeriDP [5]</td><td>(BLS12-381) GKR; incremental verification</td><td>W4</td><td>small scale</td><td>cf86a46</td></tr><tr><td>Expander [63]</td><td>GKR; Orion / Hyrax / one block KZG</td><td></td><td>commitment comparison</td><td>096581e</td></tr><tr><td>ZKTorch [64]</td><td>KZG blocks, accumulation (BN254; CPU)</td><td></td><td>MLP block non-sumcheck 63b9c68 point</td><td></td></tr></table>

Kernel libraries: ICICLE v2.8.0 (GPU) and v4.0.0 (CPU), sppark 69440fd.

Platforms and protocol. A shared server (five NVIDIA A100-PCIE-40GB cards; dual Xeon Gold 6248R, 96 logical cores) ran the kernel micro-benchmarks, the zkLLM breakdowns and the CPU-only provers; an exclusively used server (one A100-PCIE-80GB card of the same architecture; 192-core host) ran VeriLoRA, DeepProve, the commitment comparison and the engineering waterfall. Absolute times are not compared across the two. Each configuration is run once to warm up and three times for measurement, and we report the median; on the shared server every repetition is gated on low system load. A few long configurations of DeepProve (up to 22 minutes each) have a single repetition, as Table IV states. FPGA and ASIC results are taken from the literature and are not mixed with our measurements.

Limitations. Full-scale coverage is mostly restricted to sumcheck-based provers; the non-sumcheck point is a CPUonly multilayer-perceptron block, and the commitment comparison is made on one transformer block. The two inference provers differ in commitment scheme and implementation, so differences between them have no single cause, and the second one reaches only 0.27 billion parameters; its author-reported rates are CPU-only, so most of its cells give a breakdown rather than a rate comparison. W4 runs on a single-threaded CPU prover with small networks and indicates trends only. Capacity (Q4) is observed at two points, 40 and 80 GB, not as a continuous comparison.

## B. Kernel Throughput Across Platforms

Fig. 4 plots the throughput of the seven kernels, and Fig. 5 places them on a roofline built from the measured roofs of the 40 GB A100. MSM and NTT use version 2.8 of ICICLE [52], cross-checked on MSM against a second implementation [65]. CPU MSM and Poseidon2 are single-threaded reference paths, so their GPU/CPU ratios indicate direction only.

![](images/f68ddd796202a974dae1508c41ed6b48ecf61549c24996329b96fd406ee2a122.jpg)  
Fig. 4. Throughput of the seven Layer-3 kernels (log scale) at $2 ^ { 1 6 } - 2 ^ { 2 6 }$ elements on the A100 GPU and the host CPU. Solid lines: GPU execution time with data resident on device; dotted: including host–device copies; dashed: CPU. Color gives the field (or hash); markers separate sumcheck degrees (d), Reed–Solomon rates (r), the two MSM libraries and hashing from Merkle-tree construction. Hollow markers denote runs whose repetition spread exceeded 10%. No GPU NTT is shown for Goldilocks because the publicly usable library release lacks that field.

Compute and bandwidth limits. MSM is compute-bound in the measured implementation: its dominant GPU kernels reach only 15–23% of the copy roof and stall on dependencies between large-integer operations. Reed–Solomon encoding is bandwidth-bound, at 74–76% of the copy roof.

Instruction issue and synchronization. Three kernel classifications that the literature suggests do not hold on this hardware, and Table II uses the measured ones. Sumcheck (degree 2) is not memory-bound: it saturates the integer-issue roof (100–111% of the single-pipe roof) while using 31– 59% of the copy roof. Lookup is synchronization- and issuebound: DRAM use is 0.8–5% of the copy roof, integer issue reaches 77–78%, and warps stall on barriers inside the batched field inversion. Field multiplication, measured element-wise, is memory-bound in both a 31-bit field (at the copy roof) and a 254-bit field (88%), so its “compute” label reflects arithmetic complexity, not measured behavior.

Memory access and instruction issue. NTT stays memorybound, but it reaches only 26–39% of the copy roof and 9–17% of the integer roof; profiling attributes the gap to memory-issue queues (strided access, bank conflicts) and low occupancy, not to the 32-bit integer pipeline that a previous characterization blamed [20]. The streaming NTT stage inside our Reed–Solomon encoder reaches 74–76% of the copy roof on the same card, so data layout decides how close NTT gets to DRAM bandwidth. Hashing is issue-bound in two ways: Poseidon2 on the integer pipes (63% of the dual-pipe roof), BLAKE3 on load issue (12% of the copy roof).

![](images/799f333db24c37eb44eb4a2cf74430bf52ad40dfba6cd7592f80b53eb15a407a.jpg)  
Fig. 5. Roofline of the kernels at $2 ^ { 2 4 }$ elements on the A100: thread-level integer-instruction rate, summed over the ALU and FMA pipelines, against integer instructions per DRAM byte. The diagonal is the measured copy bandwidth (≈1.36 TB/s); the solid and dashed lines are the measured singlepipe (≈9.66×10<sup>12</sup> inst/s) and dual-pipe (≈1 $. 9 1 \times 1 0 ^ { 1 3 } )$ issue limits, so points may lie above the solid line. Left of the ridge point (≈7 inst/byte) bandwidth can limit a kernel, right of it instruction issue; a point far below both is limited by dependencies, synchronization or occupancy. Labels: sc sumcheck, lk lookup, fmul field mult., rs Reed–Solomon, p2 Poseidon2, b3 BLAKE3; bb BabyBear, gl Goldilocks, bn BN254; i2 ICICLE v2, sp sppark.

GPU versus CPU, and the cost of copies. The GPU/CPU throughput ratio spans three orders of magnitude, from 1.7× for large-field lookup (capped by the barriers above) to about 2500× for Poseidon2. Counting host–device copies costs the streaming kernels most—field multiplication 210–741×, sumcheck 113–412×, NTT 15–18×—while MSM loses only 3.5–7×: the kernels that move the most data are the ones for which offloading costs the most, an early sign of the B4 bottleneck.

## C. End-to-End Cost Breakdowns

Fig. 6 shows the cost shares for the reported configurations, and Table IV summarizes the overheads of a representative subset. DeepProve’s timers cannot separate its requantization and activation lookups from the sumcheck inside them, so 45–63% of its GPU time is mixed. Its cells run the CUDA backend, whereas its author-reported rates are CPU-only, so they serve as a breakdown rather than a rate comparison; one CPU cell at the author-reported setting provides the like-forlike ratio in Section IV-E.

Q1 — commitment and opening dominate only sometimes. On VeriLoRA, opening is the largest category at short input lengths (46.7% at input length 64 for 7B, 42.7% for 13B) but falls to 26.1% (24.5%) at 2048, while lookup stays at 27–39%. On zkLLM, opening leads at input length 1024 (35.9%) but is overtaken by witness generation at 2048 (25.5% versus 44.0%). On DeepProve’s GPU cells, commitment takes 18–25% and opening only 5–9%. On the Expander block, commitment and opening together are 1.6–5.5% under all three schemes and the chained sumchecks take 80–85%; on ZKTorch, commitment is the largest category (56.3%); on VeriDP it is near zero. What dominates is a function of proof system, workload and scale.

Q2 — the witness share varies with workload and scale. The witness share runs from effectively zero (Expander block, VeriDP) through 3–22% (LoRA, DeepProve on GPU) and 14% (ZKTorch) to 32–44% (zkLLM). Where input length varies, it mostly grows with it: by 12 points from 1024 to 2048 on zkLLM, and from 3.6% to 16.0% (7B) and 4.4% to 18.8% (13B) between 64 and 2048 on VeriLoRA, consistent with the quadratic attention witness (B1) outgrowing the weight-related work. DeepProve is the exception at W1, where the share stays at 3–6% between 64 and 1024 while its mixed category dominates; at W2 it grows with the number of generated tokens, from 6% to 18% between 32 and 512 at a prompt of 64.

Q3 — fixed-weight openings remain stable in the LoRA measurements. The clearest evidence is VeriLoRA: across input lengths 64–2048, opening the weights takes approximately 11.5 s (7B) or 12.4 s (13B), varying by less than 1%, while opening the intermediate tensors grows from 6.2 to 54.6 s and from 6.7 to 67.7 s. zkLLM shows a related shift in cost shares: doubling the input length raises proving time by 1.42× (33.6 to 47.8 s) yet lowers the opening share by 10 percentage points. The commitment scheme, in contrast, shows up mainly in proof size and verification: on the same block, moving from elliptic-curve commitments to the hash-based one grows the proof about 7× (0.90–0.96 to 6.56 MB) and cuts verification by about 60% (5.8–6.1 to 2.4 s), while prover time falls by a third (13.5–14.2 to 9.3 s). The two options also run on different fields (BN254 versus M31×16), so this compares EC-commitment-on-BN254 with hash-commitmenton-M31, not the schemes alone.

Q4 — memory binds as a capacity wall before it binds as a time share. Where it is measured, data movement is at most 8% of proof time. Capacity is different. The 13B LoRA step at input length 2048 needs 45.0 GB by the prover’s own account (56.2 GB sampled externally), so this configuration exceeds the 40 GB card’s capacity but completes in 327 s on the 80 GB device. The 7B step at the same length peaks at 39.9 GB on the 80 GB device, close to the smaller card’s nominal capacity. DeepProve aborted during model commitment on the 40 GB card, driven by vocabulary-embedding size rather than input length, but completes on the 80 GB card with GPU peaks of 13–66 GB; its 7B model failed on every configuration (bufferallocation failures, or a shape-inference defect that also fails on CPU). Host memory is a wall too: DeepProve’s host peak grows from 60 GB at input length 64 to 260 GB at 1024, and ZKTorch’s largest verifiable model needs about 48 GB on a CPU-only run.

One-off and verification costs are not negligible. Verification is 14–31% of proving time on VeriLoRA and 26–43% on the Expander block, a few seconds on DeepProve, and near zero on the interactive zkLLM. One-off setup is large where it exists: 295–460 s on DeepProve, longer than the proof of a short context (270 s at input length 64), and about a fifth of wall-clock on ZKTorch—the amortization opportunity S1 names.

Answers. No single category dominates across systems. Three generalizations hold in our data: each cell’s cost concentrates in two or three categories; witness work grows with input length while weight-related opening does not (S1, S4); and memory binds first as capacity, on device and host, rather than as a share of time (B4). These trends guide the hardware assessment in Section V.

## D. How Much Can Engineering Remove?

We evaluate how implementation choices change the cost profile before considering hardware acceleration. Starting from a re-run baseline, we apply four changes to zkLLM in a fixed order: a resident prover service (no per-stage process start-up or CUDA initialization), Pippenger MSM on the opening path, integer-resident witness generation, and keeping intermediates on device instead of swapping to disk. Every version must pass a correctness gate before it is timed: an unmodified run verifies, and truncated inputs, corrupted activations or truncated weight files are rejected. (In this refactored code base the weight values are not bound to the proved claim, so flipping weights alone still verifies—a limitation of the code, not of the gate.)

Fig. 7 shows a nearly unchanged aggregate proving time, from 29.1 s to 29.0 s, across this optimization sequence. The

![](images/1fa69205462b317591e467564203c85f4bede338431a5e6438c216ac9b954969.jpg)  
Fig. 6. Share of proof time by cost category for each measured (workload, prover, scale) cell; bar tops give the median proof time and the lower strip shows one-off setup and verification on a logarithmic scale. Labels give the prover, the model and the input length in tokens (P+G: P prompt plus G generated tokens), the platform (40G: shared A100-40GB; 80G: exclusive A100-80GB; CPU) and the number of repetitions n.

29.1 s baseline is a re-run on the exclusive 80 GB server and therefore differs from the 33.6 s measured on the shared 40 GB server in Table IV. Individual steps have opposing effects, and the last two plotted medians pool repetitions from both servers, as stated in the caption. The resident service saves 2.3 s of proving time and 18% of wall-clock (34.8 to 28.5 s); removing the disk swap saves 1.3 s. Pippenger MSM adds 1.3 s on both servers, even after tuning: a first version with 8192 bucket groups per window starved the GPU, and 1024 groups, the setting used in the timed runs, recovered most of that loss. Against the re-run baseline, however, the three large openings are unchanged (2.30 to 2.26–2.29 s) while the three small ones become 33% slower (1.27 to 1.69 s), so the fixed cost of the bucket method outweighs its gain on small MSMs. Perwindow parallelism, not the algorithm label, decides whether an MSM kernel helps. Integer-resident witness generation adds 2.2 s on the exclusive server, whereas on the shared server the same steps saved 4.8 s, consistent with, but not sufficient to isolate, an effect of disk contention; built with a newer CUDA toolchain it even failed its correctness check. The observed effects depend on the platform, configuration and toolchain. This sequence identifies changes that need further evaluation;

it does not establish a bound on software optimization or on the gains attainable from hardware.

## E. Comparison with Published Systems

Table I lists the author-reported costs of the published systems, normalized per token or per step. Within its caveats, per-token proving costs span about 0.02–0.9 s across very different designs, verification stays between milliseconds and seconds, and the two systems that report a plaintext baseline put the slowdown at $1 0 ^ { \overset { . } { 1 } } \ – 1 0 ^ { 3 } \times$ (DeepProve 24× to 10<sup>3</sup>× [11], VeriLoRA about $1 0 ^ { 3 } \times [ 1 3 ] )$ —consistent with the 249–341× we measure for one proved layer on W1 and the 92–143× on the small W4 networks.

Two reference comparisons illustrate the effects of configuration and implementation differences. Our zkLLM run—the refactored public release, interactive and without Fiat–Shamir, on synthetic fixed-point weights and a PCIe rather than SXM4 card—takes 47.8 s per layer at input length 2048, or about 25.5 minutes for 32 layers, roughly 1.9× the author-reported 803 s for a 13B model. For DeepProve one like-for-like ratio is available: the author-reported 64 tokens per minute use a 24-core CPU server with the HyperKZG commitment at input length 512, and the same setting on our CPU host, pinned to 24 threads, proves the same 0.27-billion-parameter model at 57.1 tokens per minute (median of three repetitions, 0.5% spread), 0.89× the reported rate. Because model sizes, versions and hardware differ, the zkLLM extrapolation provides context rather than a matched replication. A lookup-cost study across two other provers [66] and vendor-reported figures [32], [67], [68] complement this picture; we deliberately do not rank systems, since the purpose of this section is to show where the cost goes, not which prover is fastest.

TABLE IV  
MEASURED PROVER OVERHEAD PER WORKLOAD, SYSTEM, AND SCALE. “PROVE” IS THE MEDIAN SUM OF THE EIGHT TAXONOMY CATEGORIES WITH ONE-OFF SETUP AND VERIFICATION EXCLUDED. FIG. 6 SHOWS THE DECOMPOSITION. ×PLAIN IS THE RATIO OVER THE UNMODIFIED PYTORCH STEP MEASURED ON THE SAME MACHINE; N/A MARKS CELLS WITH NO COMPATIBLE BASELINE. A REPRESENTATIVE SUBSET OF CELLS IS SHOWN HERE; FIG. 6 SHOWS THE FULL SET. THE reps COLUMN GIVES THE NUMBER OF REPETITIONS n (CPU: HOST-CPU-ONLY RUNS). †DEEPPROVE TIMES TRACE GENERATION (EXECUTING THE QUANTIZED MODEL TO BUILD THE WITNESS) SEPARATELY, BEFORE ITS PROVING TIMER; TO KEEP THIS COLUMN COMPARABLE ACROSS PROVERS WE REPORT END-TO-END PROVER TIME, AND THE SUPERSCRIPT GIVES THE SHARE OF IT SPENT IN TRACE GENERATION.
<table><tr><td>Workload</td><td>System / scale</td><td>reps</td><td>prove (s)</td><td>verify (s)</td><td>proof</td><td>GPU pk (GB)</td><td>host pk (GB)</td><td>×plain</td></tr><tr><td>W1</td><td>zkLLM 7B, 1024 tok</td><td>n=1</td><td>33.6</td><td>1.8e-05</td><td>n/a</td><td>21.2</td><td>0.84</td><td>341×</td></tr><tr><td>W1</td><td>zkLLM 7B, 2048 tok</td><td>n=2</td><td>47.8</td><td>1.5e-05</td><td>n/a</td><td>29.3</td><td>0.84</td><td>249×</td></tr><tr><td>W1</td><td>DeepProve 270M, 64 tok</td><td>n=3†3%</td><td>270</td><td>2.8</td><td>12.66 MB</td><td>21.7</td><td>60.2</td><td>n/a</td></tr><tr><td>W1</td><td>DeepProve 270M, 256 tok</td><td> $\mathrm { n } { = } 2 ^ { \dagger } 6 \%$ </td><td>509</td><td>3.28</td><td>15.29 MB</td><td>65.6</td><td>95.4</td><td>n/a</td></tr><tr><td>W1</td><td>DeepProve 270M, 1024 tok</td><td>n=1†4%</td><td>1278</td><td>5.12</td><td>18.13 MB</td><td>24.4</td><td>260.4</td><td>n/a</td></tr><tr><td>W1</td><td>DeepProve 270M, 512 tok</td><td>n=3, CPU†67%</td><td>1615</td><td>3.8</td><td>16.66 MB</td><td>一</td><td>146.4</td><td>n/a</td></tr><tr><td>W3</td><td>VeriLoRA 7B, 64 tok</td><td>n=3</td><td>38.0</td><td>11.6</td><td>n/a</td><td>13.6</td><td>0.14</td><td>5.0k×</td></tr><tr><td>W3</td><td>VeriLoRA 7B, 512 tok</td><td>n=3</td><td>76.1</td><td>17.7</td><td>n/a</td><td>18.5</td><td>0.16</td><td>9.0k×</td></tr><tr><td>W3</td><td>VeriLoRA 7B, 2048 tok</td><td>n=3</td><td>253</td><td>39.7</td><td>n/a</td><td>39.9</td><td>0.52</td><td>11.1k×</td></tr><tr><td>W3</td><td>VeriLoRA 13B, 64 tok</td><td>n=3</td><td>44.7</td><td>12.8</td><td>n/a</td><td>20.6</td><td>0.14</td><td>n/a</td></tr><tr><td>W3</td><td>VeriLoRA 13B, 512 tok</td><td>n=3</td><td>94.8</td><td>20.2</td><td>n/a</td><td>27.2</td><td>0.16</td><td>n/a</td></tr><tr><td>W3</td><td>VeriLoRA 13B, 2048 tok</td><td>n=3</td><td>327</td><td>46.9</td><td>n/a</td><td>56.2</td><td>0.52</td><td>n/a</td></tr><tr><td>W4</td><td>VeriDP lin-17p</td><td>n=3, CPU</td><td>0.107</td><td>1.5e-04</td><td>3.5 kB</td><td></td><td>0.15</td><td>92×</td></tr><tr><td>W4</td><td>VeriDP log-513p</td><td>n=3, CPU</td><td>0.158</td><td>2.0e-04</td><td>3.7 kB</td><td></td><td>0.15</td><td>143×</td></tr><tr><td>W4</td><td>VeriDP mlp-102k</td><td>n=5, CPU</td><td>10.6</td><td>1.5e-04</td><td>3.9 kB</td><td>一</td><td>0.38</td><td>n/a</td></tr><tr><td>block</td><td>Expander Hyrax</td><td>n=3, CPU</td><td>13.5</td><td>5.76</td><td>939.0 kB</td><td>一</td><td>5.80</td><td>n/a</td></tr><tr><td>block</td><td>Expander KZG</td><td>n=3, CPU</td><td>14.2</td><td>6.1</td><td>876.8 kB</td><td></td><td>5.80</td><td>n/a</td></tr><tr><td>block</td><td>Expander Orion</td><td>n=3, CPU</td><td>9.26</td><td>2.4</td><td>6.25 MB</td><td>一</td><td>5.92</td><td>n/a</td></tr><tr><td>MLP</td><td>ZKTorch 2x8192</td><td>n=5, CPU</td><td>400</td><td>0.0693</td><td>14.9 kB</td><td>一</td><td>47.8</td><td>n/a</td></tr><tr><td>W2</td><td>DeepProve 270M, 64+32 tok</td><td> $\mathrm { n } { = } 3 ^ { \dagger } 6 \%$ </td><td>352</td><td>3.06</td><td>13.95 MB</td><td>14.7</td><td>71.9</td><td>n/a</td></tr><tr><td>W2</td><td>DeepProve 270M, 64+128 tok</td><td>n=3†12%</td><td>519</td><td>3.31</td><td>15.25 MB</td><td>14.7</td><td>95.5</td><td>n/a</td></tr><tr><td>W2</td><td>DeepProve 270M, 64+512 tok</td><td> $\mathrm { \Delta } \mathrm { n } { = } 1 ^ { \dag } 1 8 \%$ </td><td>1319</td><td>3.9</td><td>18.14 MB</td><td>15.0</td><td>254.0</td><td>n/a</td></tr><tr><td>W2</td><td>DeepProve 270M, 256+512 tok</td><td> $\mathrm { n } { = } 2 ^ { \dagger } 1 9 \%$ </td><td>1465</td><td>4.53</td><td>18.13 MB</td><td>22.9</td><td>254.1</td><td>n/a</td></tr></table>

Baseline denominators are the median of the unmodified PyTorch step repetitions matched on workload, model, input length or batch, and machine. CPU rows ran on the host CPU only; “–” marks quantities the prover does not report.

## V. STATE OF HARDWARE ACCELERATION

This section reviews acceleration on GPUs, FPGAs, ASICs and AI accelerators, and then assesses the body of work through the framework. Table V summarizes twenty representative designs. Their speed-ups are measured against different baselines and are not comparable across rows; what matters for our purpose is what each design accelerates.

## A. GPUs: The Workhorse

GPUs run almost every prover that reaches the scale of large language models today [9], [11], [13], and AI providers already own them. Early work such as GZKP [16] and cuZK [17] parallelized MSM and NTT for pairing-based

SNARKs and reduced host–device traffic, and open libraries now package these kernels [52], [65]. A systematic characterization of GPU provers exposes the limit of this kernelby-kernel approach: once MSM is optimized, NTT dominates proving latency [20]; its attribution of NTT to the 32-bit integer pipeline, however, is not borne out by our measurements (Section IV-B).

More recent systems widen the scope. BatchZK [22] pipelines sumcheck, Merkle-tree and encoder modules so that many proofs are generated concurrently, overlapping transfers with computation. ZKPoG [47] brings witness generation onto the GPU with the rest of a Plonkish prover and manages memory so that large circuits fit on consumer cards. Chimera [21] keeps the sumcheck state resident on the GPU across rounds, but its end-to-end gain stays far below its sumcheck speed up because the polynomial commitment becomes the next bottleneck. TensorZKP [45] uses the tensor cores that power AI, building finite-field arithmetic from low-precision matrix multiplication and recasting sumcheck and code encoding as matrix products.

GPUs are the default platform for verifiable AI. Their limits are the ones identified as B4: memory capacity, which decides whether a witness fits on one device, and the cost of moving data to the device, which falls hardest on streaming kernels such as encoding, sumcheck and field arithmetic (Section IV-B). Tensor cores open a path to exploit the numericformat structure S3, but only for provers designed around them.

TABLE V  
REPRESENTATIVE HARDWARE ACCELERATION OF ZERO-KNOWLEDGE PROVERS. SPEED-UPS ARE AS REPORTED BY THE AUTHORS AGAINST THE BASELINES NAMED IN PARENTHESES AND ARE NOT COMPARABLE ACROSS ROWS. THE LAST COLUMN MARKS WHETHER A NEURAL-NETWORK MODEL OR CIRCUIT IS PART OF THE EVALUATION.
<table><tr><td>Work</td><td>Platform</td><td>Target proof system</td><td>Accelerated kernels</td><td>Reported speed-up (baseline)</td><td>AI</td></tr><tr><td>GZKP [16]</td><td>GPU</td><td>pairing-based SNARK</td><td>MSM, NTT</td><td>up to 48.1× (MINA, V100); up to 17.6× (bellperson, Zcash, four V100)</td><td></td></tr><tr><td>cuZK [17]</td><td>GPU</td><td>pairing-based SNARK</td><td>MSM</td><td>MSM up to 2.94×, overall up to 4.86× (prior GPU systems)</td><td></td></tr><tr><td>BatchZK [22]</td><td>GPU</td><td>sumcheck + linear-code PCS</td><td>sumcheck, Merkle tree, encoder</td><td>&gt;259.5× throughput (GPU systems); 9.52 proofs/s for verifiable ML</td><td>√</td></tr><tr><td>ZKPoG [47]</td><td>GPU</td><td>Plonkish, incl. witness generation</td><td>end-to-end</td><td>22.8× (CPU), 12.7× (GPU)</td><td></td></tr><tr><td>Chimera [21]</td><td>GPU+CPU</td><td>Spartan</td><td>sumcheck</td><td>7.96× sumcheck region; 1.27–1.64× end-to-end</td><td></td></tr><tr><td>TensorZKP [45]</td><td>GPU tensor cores</td><td>HyperPlonk + Brakedown</td><td>sumcheck, encoder, field arithmetic</td><td>955× (CPU), 36.2× (SIMT GPU)</td><td></td></tr><tr><td>if-ZKP [53]</td><td>FPGA (Intel)</td><td>pairing-based SNARK</td><td>MSM</td><td>110–150× (software library)</td><td></td></tr><tr><td>OPTIMSM [18] Hirner et al. [46]</td><td>FPGA (U55C) FPGA (U280, HBM)</td><td>any EC-based Brakedown / Orion PCS</td><td>MSM encoder, Merkle tree</td><td>up to 12.77× (prior FPGA) commitment up to 264×, proving up</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>to 65× (Orion software, CPU); 3.85× end-to-end with Virgo</td><td></td></tr><tr><td>LegoZK [24] ZK-Flex [23]</td><td>reconfig. ASIC (simulated)</td><td>multiple multiple (POLY + EC)</td><td>MSM, NTT (shared field-arithmetic units) NTT, MSM,</td><td>avg. 31.96× (PipeZK), 11.30× (GZKP, V100) 5–11× (state of the art)</td><td>√</td></tr><tr><td>PipeZK [15]</td><td>reconfig. accel.</td><td></td><td>multi-precision mult. NTT, MSM</td><td>10× on standard benchmarks, 5× on</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Zcash (CPU); modules up to 197.5× (NTT), 77.7× (MSM) avg. 97× (CPU), 46× (GPU)</td><td></td></tr><tr><td>UniZK [69]</td><td>ASIC</td><td>hash-based (Plonky2)</td><td>NTT, hashing, polynomial ops (unified)</td><td>gmean 586× (32-core CPU); 41×</td><td></td></tr><tr><td>NoCap [59]</td><td>ASIC</td><td>hash-based (Orion-type)</td><td>full prover</td><td>(PipeZK, iso-resource) &gt;400× (CPU), 12× (GPU), 3×</td><td></td></tr><tr><td>SZKP [19]</td><td>ASIC</td><td>pairing-based SNARK</td><td>NTT, MSM (full proof)</td><td>(ASIC)</td><td></td></tr><tr><td>zkSpeed [56] zkPHIRE [57]</td><td>ASIC ASIC</td><td>HyperPlonk HyperPlonk, high-degree</td><td>sumcheck, MSM sumcheck, MSM</td><td>801 × geomean (CPU) 1486× (CPU), 11.87× iso-area</td><td></td></tr><tr><td>MTU [58]</td><td>ASIC unit</td><td>gates sumcheck / Merkle-based</td><td>sumcheck, Merkle tree</td><td>(SOTA) up to 1478× (bandwidth-matched</td><td></td></tr><tr><td>PipeSC [55]</td><td>ASIC</td><td>multilinear SNARKs</td><td>sumcheck</td><td>CPU) up to 2756× (Arkworks CPU),</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>5.02× (BatchZK GPU) up to 10× NTT throughput, MSM on</td><td></td></tr></table>

Baselines are given in parentheses. AI: ✓ = a neural-network model or circuit is part of the evaluation (a model for BatchZK; circuits for ZK-Flex and for LegoZK, which reports LeNet, AlexNet and VGG16 circuits); – = none reported.

## B. FPGAs: Streaming and Efficiency

FPGA work has concentrated on MSM: if-ZKP [53] scales Pippenger’s algorithm across Intel FPGAs, and OP-TIMSM [18] decouples the number of buckets from the window size to save on-chip memory. Reconfigurable architectures aim at the whole prover: LegoZK [24] builds on the observation that every proving stage reduces to the same basic finite-field operations, and ZK-Flex [23] combines a multiprecision multiplier core with a flexible on-chip network to keep utilization high as the workload shifts between polynomial and elliptic-curve stages.

The most instructive FPGA result for AI workloads concerns hash-based commitments. Hirner et al. [46] show that for the commitment of Section II-C it is random off-chip memory accesses, not arithmetic, that limit the encoder; by restructuring the code graph to remove random writes, generating it on the fly, and pipelining Merkle-tree hashing on an HBMequipped FPGA, they accelerate the full commitment proof by about two orders of magnitude. This example illustrates how FPGAs offer energy efficiency and deep streaming pipelines. They are a poor fit for kernels whose speed is set by memory bandwidth, unless the dataflow is redesigned as in the example above, but they are well suited to hashing, to bit-level arithmetic over binary fields [44], and to processing data as it streams through the system.

![](images/3109e29117c1b6ff6747c6f64be038fd765c4e9863874637c6eb32666d402ebe.jpg)  
Fig. 7. Engineering-optimization waterfall for zkLLM on W1 (7B, one layer, input length 1024, A100-80GB, exclusive server). Each middle bar shows the per-category change against the previous bar; red-edged segments are regressions. Labels give the median proof time (one-off setup and verification excluded) and the slowdown over the same step in unmodified PyTorch on the same machine (×plain). Medians of items 4 and 5 pool repetitions from the shared and the exclusive server (item 4 alone on the exclusive server: 30.8 s).

## C. ASICs and AI Accelerators

ASIC designs specialize the datapath and memory hierarchy for proving workloads. PipeZK [15] pipelined NTT and MSM for pairing-based SNARKs, and SZKP [19] targets full-proof acceleration. NoCap [59] and UniZK [69] target hash-based provers, the latter mapping their diverse kernels onto one array of vector-systolic units. A newer generation targets multilinear proof systems in which sumcheck dominates: zk-Speed [56] accelerates HyperPlonk end to end, zkPHIRE [57] adds a programmable sumcheck unit for high-degree gates, MTU [58] exploits the binary-tree pattern shared by sumcheck and Merkle trees, and PipeSC [55] pipelines sumcheck alone. Under equal memory budgets there is no universal winner between NTT- and sumcheck-based primitives [51].

AI accelerators are an emerging alternative that providers already deploy at scale. MORPH [54] converts high-precision modular arithmetic into dense low-precision matrix multiplications on TPU matrix units, exceeding a GPU baseline on NTT and matching it on MSM. Such reuse is attractive because proof systems evolve rapidly: within a few years the field has moved from univariate to multilinear constructions, from elliptic-curve to hash-based commitments, and from 256- bit to small fields [35], [44], [49]. A chip designed for one generation risks obsolescence before it ships, which favors programmable designs [23], [24], [57] and reuse of existing AI silicon [45], [54].

## D. A Workload-Aware Assessment

Viewed through the framework, the surveyed designs share four gaps with respect to AI workloads.

Kernel focus lags behind AI proof systems. Nine of the twenty designs in Table V take MSM or NTT as their main accelerated kernel, the kernels of pairing-based and univariate systems. Large-model provers, by contrast, are built on sumcheck and lookups (Section III). Sumcheck accelerators have appeared only recently [22], [55]–[58], and none of the surveyed designs is built specifically around lookup arguments for non-linear layers.

AI workloads are almost absent from evaluations. Only three designs report an evaluation on a machine learning model or circuit: BatchZK [22], ZK-Flex [23] and LegoZK [24], all on convolutional networks. LegoZK’s results for LeNet, AlexNet and VGG16 come from simulation of a 28 nm design rather than from fabricated hardware. The others report generic benchmarks, with problem sizes of up to $2 ^ { 2 4 }$ MSM points or $2 ^ { 3 0 }$ constraints [18], [57]. A forward pass of a 7-billionparameter model over 2048 tokens already involves on the order of $1 0 ^ { 1 3 }$ multiply–accumulate operations, so the memory behavior of AI-scale witnesses is largely untested.

AI-specific structure is unexploited. None of the surveyed accelerators is designed to reuse commitments to fixed weights across queries or to merge their openings (S1); batching of independent proofs [22] is the closest instance of exploiting repetition (S2). Only one of them accelerates witness generation [47], although field-based witness generation accounts for up to 44% of proving time in our zkLLM measurements (B1).

End-to-end gains trail kernel gains. When one kernel is accelerated, the bottleneck moves to the next (Section V-A), and our own waterfall shows that implementationlevel changes to a prover can cancel each other out (Section IV-D). The end-to-end benefit of a kernel speed-up depends on its original cost share and on any new transfer or synchronization costs. This motivates joint design of witness generation, commitment, opening and data movement.

These gaps are the hardware counterpart of the gap between kernel speed-ups and end-to-end gains noted in the introduction. Section IV quantifies them for concrete workloads, and Section VI turns them into open problems.

## VI. OPEN PROBLEMS AND ROADMAP

The gaps identified in Sections IV and V point to seven open problems. For each we note which bottleneck of the framework it addresses.

1) Proof-friendly numeric formats (B1, S3). Today a model is trained and served in floating point, then quantized and recomputed over a finite field for the proof. If the model itself ran in an integer or fixed-point format that the proof system can absorb, the witness could be produced as a byproduct of ordinary execution on integer tensor units, removing most of the first bottleneck. Encouraging signs exist: integeronly quantization at 12 bits preserves the output of small language models closely [11], an industrial prover works directly on 8-bit integer matrix multiplication [32], and tensor cores and TPU matrix units can already carry finite-field arithmetic [45], [54]. Open questions are whether such formats preserve accuracy for training as well as inference, and how to keep execution bit-exact across hardware so that the proved computation is the one actually run [48], [70].

2) Commit first, prove later (S1, B2). Proving every inference or training step is expensive; committing to the computation is much cheaper. A provider could commit to intermediate states while it computes and produce proofs only on demand or for randomly sampled steps. A system that commits to block-boundary tensors during fine-tuning and later re-executes sampled blocks in trusted hardware reports 11–36% training overhead across its evaluated models and stores 179–304 GB per 2048 samples for its 14-billionparameter model [3], and a research agenda for verifying frontier training proposes real-time Merkle commitments as a core component [60]. Replacing the trusted re-execution with zero-knowledge proofs raises three questions: how cheap the always-on commitment can be made, what to store versus recompute, and how sampling rates translate into detection guarantees.

3) Commitment on the data path (B4). In distributed training and sharded inference, tensors already flow over the network between devices. Hashing or committing to them as they pass through SmartNICs or FPGAs would take commitment off the GPU’s critical path and turn data movement from a cost into an opportunity. Hash-based commitments suit such streaming hardware well (Section V-B). A recent blueprint for verifying frontier training already relies on observing internode traffic through a physical tap or an attested SmartNIC and lists line-rate network taps among its open problems [60]; we are not aware of a published design that commits to tensors on the data path.

4) Heterogeneous scheduling (B3, B4). Table II summarizes the different resource demands of the kernels. Our GPU measurements distinguish compute-bound MSM, issuebound low-degree sumcheck, synchronization- and issuebound lookup, and bandwidth-bound encoding. Splitting a prover across devices—as already done between CPU and GPU for small sumcheck rounds [21]—could raise utilization, but every split adds transfers of large tables. A cost model that accounts for both kernel fit and data movement, and schedulers that use it, are missing.

5) Co-selecting the commitment scheme and the hardware (B2, B3). Elliptic-curve commitments concentrate cost in MSM, which suits GPUs and ASICs; hash-based commitments concentrate it in encoding and hashing over small fields, which suits FPGAs and bit-oriented logic [44], [46]. Because the interactive part of a proof and its commitment are largely independent (Section III-A), the commitment scheme can be chosen to match the available hardware. The blocklevel comparison in Section IV-C provides a starting point. Extending it to full AI workloads while controlling field and security parameters would help separate scheme effects from implementation effects.

6) Security and performance are coupled. Faster proofs must not weaken what is proved. A provider can keep a model’s architecture and parameter count while choosing weights that collapse the computation into a much smaller model, and the proof still verifies [71]: a proof of correct inference is not a proof that a large model was executed. Every acceleration that changes the computation—lower-precision formats, sampling, merged openings—should come with an argument that the semantic guarantee the user cares about is preserved. Hardware cannot fix this alone, but hardware– protocol co-design must keep it in view.

![](images/b63b62afeed2fdecd76d159e9fb1ff6634ee2b53ff52df9f1f718f08604c126c.jpg)  
Fig. 8. Roadmap toward practical verifiable AI. Numbers refer to the open problems in Section VI.

7) Benchmarks for AI-scale verifiable computation. Accelerators are evaluated on generic circuits, and AI provers on their own models and hardware (Section V-D). A shared benchmark suite—fixed models, input lengths and training steps, a common cost taxonomy, and reference implementations—would let hardware designers optimize for the workloads that matter. The workloads and taxonomy of Section IV are offered as a starting point.

Roadmap. Fig. 8 presents these problems by time horizon. In the near term, existing GPUs and shared benchmarks support evaluations of software optimizations, merged openings of fixed weights, batching of repeated layers and field arithmetic on tensor cores. The mixed effects in Section IV-D make end-to-end validation on the target platform essential. In the medium term, proof-friendly numeric formats, commit-firstprove-later training and heterogeneous scheduling change what has to be proved and where. In the long term, commitment on the data path and accelerators co-designed with AI silicon could make verifiability a built-in property of AI infrastructure rather than an expensive add-on.

## VII. CONCLUSION

This article connects the cost of verifiable AI to hardware design through a four-layer framework spanning verifiable AI workloads, proof systems, cryptographic kernels and platforms. Cross-system measurements show that acceleration priorities change with the prover and input length: witness generation, commitment and opening, lookup and sumcheck each dominate in different configurations. The LoRA measurements identify nearly constant fixed-weight opening costs alongside growing work on intermediate tensors, while the 13- billion-parameter configuration exposes a capacity boundary between 40 and 80 GB devices. Kernel profiling further shows that instruction issue and synchronization can limit kernels commonly described as bandwidth-bound. Together, these findings make workload-specific cost profiles and memory requirements central to accelerator selection.

The hardware review identifies opportunities to connect recent kernel advances to AI-scale evaluations, fixed-weight reuse and witness generation. Software optimization remains part of this agenda: the zkLLM case study shows opposing effects across a particular sequence of changes and motivates validation on the intended platform. Proof-friendly numeric formats, coordinated commitment and opening, heterogeneous execution and shared benchmarks provide a path toward integrating proof generation into AI infrastructure.

## REFERENCES

[1] Y. Sun, Y. Li, Y. Zhang, Y. Jin, and H. Zhang, “SVIP: Towards Verifiable Inference of Open-source Large Language Models,” arXiv:2410.22307, 2024.

[2] J. M. Ong, M. D. Ferrante, A. Pazdera, R. Garner, S. Jaghouar, M. Basra, M. Ryabinin, and J. Hagemann, “TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference,” arXiv:2501.16007, 2025.

[3] H. Jin, C. Zhang, H. Yu, S. Shi, N. Zhang, Y. T. Hou, and W. Lou, “Trusting What You Cannot See: Auditable Fine-Tuning and Inference for Proprietary AI,” arXiv:2603.07466, 2026.

[4] H. Akgul, D. Borg, A. Berisha, A. Rahimova, A. Novak, and M. Petrov, “Verifiable Fine-Tuning for LLMs: Zero-Knowledge Training Proofs Bound to Data Provenance and Policy,” arXiv:2510.16830, 2025.

[5] B. Abdolmaleki, A. R. Asadi, V. R. Asadi, S. Köpsell, B. Mohee, N. Roustaeifar, and M. Zarezadeh, “VeriDP: Verifiable Differentially Private Training,” Proceedings on Privacy Enhancing Technologies, vol. 2026, no. 3, pp. 48–64, 2026.

[6] H. Jia, M. Yaghini, C. A. Choquette-Choo, N. Dullerud, A. Thudi, V. Chandrasekaran, and N. Papernot, “Proof-of-Learning: Definitions and Practice,” in Proc. IEEE Symposium on Security and Privacy (SP), 2021, pp. 1039–1056.

[7] Z. Chen, M. Wu, M. Heinrich, J. Zeng, H. Lan, T. Zhang, and R. Tan, “Communication-Efficient Verifiable Attention for LLM Inference,” arXiv:2606.16352, 2026.

[8] A. Arun, A. S. Arnaud, A. Titov, B. Wilcox, V. Kolobaric, M. Brinkmann, O. Ersoy, B. Fielding, and J. Bonneau, “Verde: Verification via Refereed Delegation for Machine Learning Programs,” arXiv:2502.19405, 2025.

[9] H. Sun, J. Li, and H. Zhang, “zkLLM: Zero Knowledge Proofs for Large Language Models,” in Proc. ACM Conference on Computer and Communications Security (CCS), 2024, pp. 4405–4419.

[10] W. Qu, Y. Sun, X. Liu, T. Lu, Y. Guo, K. Chen, and J. Zhang, “zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference,” in Proc. 34th USENIX Security Symposium, 2025, pp. 2045– 2063.

[11] N. Gailly, I. Hishon-Rezaizadeh, T. Liu, N. Mainardi, D. Papadopoulos, C. Papamanthou, C. Pappas, S. Srinivasan, Z. Youell, and Y. Zhang, “DeepProve: Verifiable End-to-End Large Language Model Inference,” in Proc. ACM Conference on Computer and Communications Security (CCS), 2026, To appear.

[12] L. Wang, H. Lou, C. Li, Y. Yu, and Y. Hu, “zkAgent: Verifiable LLM Agent Execution via One-Shot Transcript Proofs,” in Proc. ACM Conference on Computer and Communications Security (CCS), 2026, To appear.

[13] G. Liao, T. Wang, S. Zhang, J. Zhang, L. Shi, and D. Tao, “VeriLoRA: Fine-Tuning Large Language Models with Verifiable Security via Zero-Knowledge Proofs,” in Proc. Network and Distributed System Security Symposium (NDSS), 2026.

[14] Q. Hu, C. Peng, Y. Dai, M. Luo, and D. He, “A Survey on the Implementation of zk-SNARK Hardware Accelerators,” Chinese Journal of Electronics, vol. 35, no. 3, pp. 1180–1194, 2026.

[15] Y. Zhang, S. Wang, X. Zhang, J. Dong, X. Mao, F. Long, C. Wang, D. Zhou, M. Gao, and G. Sun, “PipeZK: Accelerating Zero-Knowledge Proof with a Pipelined Architecture,” in Proc. ACM/IEEE International Symposium on Computer Architecture (ISCA), 2021, pp. 416–428.

[16] W. Ma, Q. Xiong, X. Shi, X. Ma, H. Jin, H. Kuang, M. Gao, Y. Zhang, H. Shen, and W. Hu, “GZKP: A GPU Accelerated Zero-Knowledge Proof System,” in Proc. ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems (ASPLOS), 2023, pp. 340–353.

[17] T. Lu, C. Wei, R. Yu, C. Chen, W. Fang, L. Wang, Z. Wang, and W. Chen, “cuZK: Accelerating Zero-Knowledge Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs,” IACR Transactions on Cryptographic Hardware and Embedded Systems, pp. 194– 220, 2023.

[18] X. Pottier, T. D. Ruijter, J. Bertels, W. Legiest, M. V. Beirendonck, and I. Verbauwhede, “OPTIMSM: FPGA hardware accelerator for Zero-Knowledge MSM,” IACR Transactions on Cryptographic Hardware and Embedded Systems, vol. 2025, no. 2, pp. 489–510, 2025.

[19] A. Daftardar, B. Reagen, and S. Garg, “SZKP: A Scalable Accelerator Architecture for Zero-Knowledge Proofs,” in Proc. International Conference on Parallel Architectures and Compilation Techniques (PACT), 2024, pp. 271–283.

[20] T. Verma, Y. Yuan, N. Talati, and T. Austin, “ZKProphet: Understanding Performance of Zero-Knowledge Proofs on GPUs,” in Proc. IEEE International Symposium on Workload Characterization (IISWC), 2025, pp. 142–154.

[21] K. Farheen and N. G. Tsoutsos, “Chimera: A Hybrid GPU Backend for Sumcheck Acceleration in Zero-Knowledge Provers,” Cryptology ePrint Archive, Paper 2026/1388, 2026.

[22] T. Lu, Y. Chen, Z. Wang, X. Wang, W. Chen, and J. Zhang, “BatchZK: A Fully Pipelined GPU-Accelerated System for Batch Generation of Zero-Knowledge Proofs,” in Proc. ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025, pp. 100–115.

[23] A. Putra, C. M. Duong, A. Q. Pham, and J.-Y. Kim, “ZK-Flex: A Flexible and Scalable Framework for Accelerating Zero-Knowledge Proofs,” in Proc. ACM/IEEE Design Automation Conference (DAC), 2026, pp. 1–7.

[24] Z. Yang, L. Zhao, P. Li, H. Liu, K. Li, B. Zhao, D. Meng, and R. Hou, “LegoZK: A Dynamically Reconfigurable Accelerator for Zero-Knowledge Proof,” in Proc. IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2025, pp. 113–126.

[25] R. Lavin, X. Liu, H. Mohanty, L. Norman, G. Zaarour, and B. Krishnamachari, “A Survey on the Applications of Zero-Knowledge Proofs,” arXiv:2408.00243, 2024.

[26] Z. Peng, C. Zhao, T. Wang, G. Liao, Z. Lin, Y. Liu, B. Cao, L. Shi, Q. Yang, and S. Zhang, “A survey of zero-knowledge proof based verifiable machine learning,” Artificial Intelligence Review, vol. 59, no. 7, 2026.

[27] T. Wang, Y. Jin, Q. Yang, Y. Xia, L. Shi, and S. Zhang, “Zero-Knowledge Federated Learning: A New Trustworthy and Privacy-Preserving Distributed Learning Paradigm,” IEEE Communications Magazine, vol. 64, no. 5, pp. 172–178, 2026.

[28] A. Y. L. Kei and S. S. M. Chow, “SoK: Private Transformer Inference Across Systems, Models, and Cryptography,” Cryptology ePrint Archive, Paper 2026/2005, 2026.

[29] A. A. Badawi, A. Alexandru, Y. Polyakov, and V. Vaikuntanathan, “SoK: Private LLM Inference using Approximate Homomorphic Encryption,” Cryptology ePrint Archive, Paper 2026/935, 2026.

[30] C. Skatharoudis, “SoK: Hash-Based Polynomial Commitments and Low-Degree Tests: From FRI to Basefold, STIR, and WHIR,” Cryptology ePrint Archive, Paper 2026/1367, 2026.

[31] J. Liang, Z. Guo, P. Wu, Q. Shen, J. Zhang, Z. Wu, H. Xue, and S. Zhai, “zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models,” arXiv:2610.08258, 2026.

[32] Attestable, “Proving LLMs at scale,” https://attestable.com/blog/ proving-llms-scale, 2026.

[33] E. Ben-Sasson, I. Bentov, Y. Horesh, and M. Riabzev, “Fast Reed-Solomon interactive oracle proofs of proximity,” in Proc. 45th International Colloquium on Automata, Languages, and Programming (ICALP), ser. LIPIcs, vol. 107, 2018, pp. 14:1–14:17.

[34] A. Golovnev, J. Lee, S. Setty, J. Thaler, and R. S. Wahby, “Brakedown: Linear-Time and Field-Agnostic SNARKs for R1CS,” in Advances in Cryptology (CRYPTO), 2023, pp. 193–226.

[35] H. Zeilberger, B. Chen, and B. Fisch, “BaseFold: Efficient Field-Agnostic Polynomial Commitment Schemes from Foldable Codes,” in Advances in Cryptology (CRYPTO), 2024, pp. 138–169.

[36] A. Kate, G. M. Zaverucha, and I. Goldberg, “Constant-Size Commitments to Polynomials and Their Applications,” in Advances in Cryptology (ASIACRYPT), 2010, pp. 177–194.

[37] R. S. Wahby, I. Tzialla, A. Shelat, J. Thaler, and M. Walfish, “Doubly-Efficient zkSNARKs Without Trusted Setup,” in Proc. IEEE Symposium on Security and Privacy (SP), 2018, pp. 926–943.

[38] C. Lund, L. Fortnow, H. Karloff, and N. Nisan, “Algebraic methods for interactive proof systems,” Journal of the ACM, vol. 39, no. 4, pp. 859–868, 1992.

[39] S. Goldwasser, Y. T. Kalai, and G. N. Rothblum, “Delegating Computation: Interactive Proofs for Muggles,” Journal of the ACM, vol. 62, no. 4, pp. 1–64, 2015.

[40] U. Haböck, “Multivariate lookups based on logarithmic derivatives,” Cryptology ePrint Archive, Paper 2022/1530, 2022.

[41] A. Gabizon, Z. J. Williamson, and O. Ciobotaru, “PLONK: Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge,” Cryptology ePrint Archive, Paper 2019/953, 2019.

[42] L. Grassi, D. Khovratovich, and M. Schofnegger, “Poseidon2: A Faster Version of the Poseidon Hash Function,” in Progress in Cryptology (AFRICACRYPT), 2023, pp. 177–203.

[43] J. O’Connor, J.-P. Aumasson, S. Neves, and Z. Wilcox-O’Hearn, “BLAKE3: One function, fast everywhere,” https://github.com/BLAKE3-team/BLAKE3-specs, 2020.

[44] B. E. Diamond and J. Posen, “Succinct Arguments over Towers of Binary Fields,” in Advances in Cryptology (EUROCRYPT), 2025, pp. 93– 122.

[45] T. Lu, J. Zhang, Y. Guo, X. Liu, W. Qu, Z. Wang, W. Chen, and J. Zhang, “TensorZKP: Repurposing GPU tensor cores for High-Performance zero-knowledge proofs,” in Proc. 35th USENIX Security Symposium, 2026, pp. 5849–5867.

[46] F. Hirner, F. Krieger, C. Piber, and S. S. Roy, “Accelerating Hash-Based Polynomial Commitment Schemes with Linear Prover Time,” IACR Transactions on Cryptographic Hardware and Embedded Systems, vol. 2025, no. 4, pp. 341–385, 2025.

[47] M. Li, Y. Yu, B. Wang, X. Fan, and S. Deng, “ZKPoG: Accelerating WitGen-Incorporated End-to-End Zero-Knowledge Proof on GPU,” Cryptology ePrint Archive, Paper 2025/765, 2025.

[48] M. Srivastava, S. Arora, and D. Boneh, “Optimistic Verifiable Training by Controlling Hardware Nondeterminism,” in Advances in Neural Information Processing Systems 37 (NeurIPS), 2024, pp. 95 639–95 661.

[49] B. Chen, B. Bünz, D. Boneh, and Z. Zhang, “HyperPlonk: Plonk with Linear-Time Prover and High-Degree Custom Gates,” in Advances in Cryptology (EUROCRYPT), 2023, pp. 499–530.

[50] W. Beullens and G. Seiler, “LaBRADOR: Compact Proofs for R1CS from Module-SIS,” in Advances in Cryptology (CRYPTO), 2023, pp. 518–548.

[51] J. Mo, A. Daftardar, B. GaneshKumar, K. Guo, H. Wang, B. Bunz, S. Garg, and B. Reagen, “When Proofs Meet Hardware: Comparing NTT and SumCheck in Zero-Knowledge Systems,” arXiv:2606.16146, 2026.

[52] K. Inbasekar, Y. Shekel, and M. Asa, “ICICLE v2: Polynomial API for Coding ZK Provers to Run on Specialized Hardware,” Cryptology ePrint Archive, Paper 2024/973, 2024.

[53] S. A. Butt, B. Reynolds, V. Ramamurthy, X. Xiao, P. Chu, S. Sharifian, S. Gribok, and B. Pasca, “if-ZKP: Intel FPGA-Based Acceleration of Zero Knowledge Proofs,” in Proc. IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2024, pp. 212–212.

[54] J. Tong, J. Dang, S. Langowski, T. Huang, A. Ali, J. Kun, J. Jiang, S. Devadas, and T. Krishna, “Enabling AI ASICs for Zero Knowledge Proof,” in Proc. ACM/IEEE Design Automation Conference (DAC), 2026, pp. 1–7.

[55] K. Wang, Y. Yanggong, X. Yang, C. Baixiao, and L. Wang, “PipeSC: A Resource-efficient and Pipelined Hardware Accelerator for Sumcheck Protocol,” Cryptology ePrint Archive, Paper 2026/691, 2026.

[56] A. Daftardar, J. Mo, J. Ah-kiow, B. Bünz, R. Karri, S. Garg, and B. Reagen, “Need for zkSpeed: Accelerating HyperPlonk for Zero-Knowledge Proofs,” in Proc. ACM/IEEE International Symposium on Computer Architecture (ISCA), 2025, pp. 1986–2001.

[57] A. Daftardar, J. Mo, J. Ah-kiow, B. Bünz, S. Garg, and B. Reagen, “zkPHIRE: A Programmable Accelerator for ZKPs over HIgh-degRee, Expressive Gates,” in Proc. IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2026, pp. 1–15.

[58] J. Mo, A. Daftardar, J. Ah-Kiow, K. Guo, B. Bünz, S. Garg, and B. Reagen, “MTU: The Multifunction Tree Unit for Accelerating Zero-Knowledge Proofs,” in Proc. International Workshop on Hardware and Architectural Support for Security and Privacy (HASP), 2025, pp. 19– 27.

[59] N. Samardzic, S. Langowski, S. Devadas, and D. Sanchez, “Accelerating Zero-Knowledge Proofs Through Hardware-Algorithm Co-Design,” in Proc. IEEE/ACM International Symposium on Microarchitecture (MI-CRO), 2024, pp. 366–379.

[60] P. Peigné, K. Nguyen, and P. Wang, “Proposals and challenges for zeroknowledge verification of frontier AI training,” arXiv:2606.05433, 2026.

[61] D. Boneh, J. Drake, B. Fisch, and A. Gabizon, “Efficient polynomial commitment schemes for multiple points and polynomials,” Cryptology ePrint Archive, Paper 2020/081, 2020.

[62] Z. Wang, P. Ma, Z. Xue, Y. Zhou, Q. Zhang, X. Zhang, and S. Wang, “ZK-Value: A Practical Zero-Knowledge System for Verifiable Data Valuation,” arXiv:2605.03581, 2026.

[63] Polyhedra Network, “Expander: An open-source GKR prover,” https: //github.com/PolyhedraZK/Expander, 2026.

[64] B.-J. Chen, L. Tang, and D. Kang, “ZKTorch: Compiling ML Inference to Zero-Knowledge Proofs via Parallel Proof Accumulation,” arXiv:2507.07031, 2025.

[65] Supranational, “sppark: Zero-knowledge template library,” https://github. com/supranational/sppark, 2026.

[66] M. J. Jo, “A Separation Principle for Lookup-Based zkML: Activation-Function Structure Cannot Reduce Per-Lookup Proving Cost,” Cryptology ePrint Archive, Paper 2026/1390, 2026.

[67] T. Xie, T. Lu, Z. Fang, S. Wang, Z. Zhang, Y. Jia, D. Song, and J. Zhang, “zkPyTorch: A Hierarchical Optimized Compiler for Zero-Knowledge Machine Learning,” Cryptology ePrint Archive, Paper 2025/535, 2025.

[68] W. Benno, A. Centelles, A. Douchet, and K. Gibran, “Jolt Atlas: Verifiable Inference via Lookup Arguments in Zero Knowledge,” arXiv:2602.17452, 2026.

[69] C. Wang and M. Gao, “UniZK: Accelerating Zero-Knowledge Proof with Unified Hardware and Flexible Kernel Mapping,” in Proc. ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025, pp. 1101–1117.

[70] N. Cankaya, “Bit-Exact AI Inference Verification Without Performance Tradeoffs,” arXiv:2606.00279, 2026.

[71] C. Gong, B. Liu, and M. Li, “Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference,” in Proc. IEEE Symposium on Security and Privacy (SP), 2026, pp. 4339– 4355.