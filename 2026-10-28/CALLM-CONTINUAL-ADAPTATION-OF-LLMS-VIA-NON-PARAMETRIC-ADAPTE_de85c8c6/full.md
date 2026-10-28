# CALLM: CONTINUAL ADAPTATION OF LLMS VIA NON-PARAMETRIC ADAPTER ROUTING

Bilge Celik   
Eindhoven University of Technology   
b.celik@tue.nl

Joaquin Vanschoren Eindhoven University of Technology j.vanschoren@tue.nl

Anna Vettoruzzo Eindhoven University of Technology a.vettoruzzo@tue.nl

## ABSTRACT

Continual adaptation of large language models (LLMs) to evolving task distributions while mitigating catastrophic forgetting remains an open problem. Existing strategies often struggle to scale or lack a robust mechanism for zero-shot task-to-expert assignment. We introduce CaLLM, a framework that resolves these tensions by combining metric-based meta-learning with modular parameter isolation via lightweight adapters. First, CaLLM projects each input batch into a task-agnostic embedding space to find its nearest prototype. It then routes the batch to the specific adapter specialized for that data distribution. This formulation provides (i) controllable cross-task interference by design, (ii) constant computational complexity regardless of task count, (iii) seamless integration of novel distributions through prototype expansion, and (iv) regime agnosticism as the local routing decision enables a unified approach to both online and offline settings. Our experiments demonstrate that this routing mechanism discovers semantically coherent task groupings while facilitating constructive interference between similar tasks, and keeping overhead at a minimum. Across diverse and fine grained continual learning benchmarks, CaLLM consistently outperforms traditional isolation and replay methods, establishing prototype-based modular routing as a principled and scalable approach to lifelong adaptation of LLMs.

## 1 INTRODUCTION

Large language models (LLMs) have become remarkable generalists, but real-world deployment often requires them to acquire task-specific expertise without sacrificing their broader capabilities. The same underlying model needs to be continually adapted to different use cases ranging from closely related to radically different, without retraining from scratch. However, the standard practice of sequential fine-tuning typically triggers catastrophic forgetting (Kirkpatrick et al., 2017), where adaptation to a new domain erodes performance on previously learned ones. A recent survey (Shi et al., 2025) formalizes this challenge along two axes: vertical continuity, moving from broad pretraining to task-specific fine-tuning, and horizontal continuity, adapting to shifting tasks and data distributions over time. Without careful management, adaptation along either of these dimensions typically results in severe performance degradation.

This problem is particularly pronounced for aligned, instruction-tuned LLMs. Wang et al. (2023b) demonstrate that even parameter-efficient methods such as LoRA (Hu et al., 2021) can severely degrade an aligned model’s general abilities and instruction-following quality after only a few rounds of sequential adaptation. Classical continual learning (CL) strategies such as regularization, experience replay, and orthogonal subspace isolation address specific aspects of this problem but they entail fundamental trade-offs. Regularization methods collapse when the model must transition between radically different domains (e.g., from natural language to code). Replay requires growing memory buffers that become unscalable as the task stream lengthens. Meanwhile, full retraining after each new task is computationally prohibitive, and real-world data often arrives as a continuous, non-stationary stream rather than the neatly delineated task blocks assumed in most CL benchmarks. Ideally, CL methods should adapt efficiently to new tasks in both online and offline settings.

We present CaLLM (Continual Adaptation of LLMs), a modular CL method that sidesteps these limitations through two interlocking components. First, CaLLM maintains a pool of modular experts (e.g., lightweight adapters implemented as LoRAs), each learning to specialize to a distinct region of the observed input distribution, ensuring that new knowledge is internalized without interfering with prior knowledge. Second, a lightweight non-parametric router, inspired by

Prototypical Networks (Snell et al., 2017), maps each incoming batch to its most relevant adapter by comparing its (frozen-backbone) embeddings against stored prototypes. This mechanism dynamically and efficiently manages the adapter pool: it spawns new adapters when the data is sufficiently novel and re-purposes the least-used modules when the pool reaches full capacity. While Prototypical Networks use class-mean embeddings for few-shot classification, CaLLM uses distributional prototypes for task-level routing. This approach inherits the non-parametric simplicity of metric learning, where no gradient-based router optimization is required, and the model absorbs novel distributions simply by initializing new prototypes. A core architectural advantage of CaLLM is its constant computational overhead: only one adapter is active per forward pass, while the remaining adapters reside on disk.

We evaluate CaLLM on two complementary benchmarks: TRACE-8, which tests robustness to radical cross-domain shifts, and CITB-19<sup>1</sup>, which tests scalability on a long horizon dialogue stream, under offline (task-incremental) and online protocols. CaLLM outperforms CL baselines on the majority of tasks in both benchmarks, with the largest gains observed on specialized domains where the pretrained backbone is weakest. Furthermore, a systematic analysis of pool capacity reveals that moderate pooling outperforms both full isolation and single-adapter training in online settings. In this regime, we observe that interleaved gradients from semantically grouped tasks function as implicit replay, fostering constructive interference that enhances generalization without the overhead of a physical replay buffer.

In summary, our primary contribution is the complete CaLLM system, which unifies non-parametric routing with dynamic adapter isolation. Specifically:

• We design and validate an end-to-end modular framework that achieves robust performance while maintaining a constant computational overhead by activating only one adapter per forward pass.

• We introduce a training-free routing mechanism that scales dynamically to multiple tasks without requiring gradient-based router optimization.

• We provide a comprehensive evaluation under both task-incremental and long-horizon online streaming scenarios, uncovering how constrained pool capacities foster constructive cross-task transfer.

The remainder of this paper is organized as follows. Section §2 contextualizes our work in the CL literature, before section §3 details the CaLLM method. Section §4 presents our empirical analysis, including router representations, task-incremental and online results, and pool capacity effects. Section §5 summarizes our findings and future directions.

## 2 RELATED WORK

Continual learning foundations. CL aims to accumulate knowledge from non-stationary streams without catastrophic forgetting. In this paradigm, each newly introduced task represents an inherently out-of-distribution shift relative to the historical training data seen by the model so far. Existing methods generally utilize replay (Rolnick et al., 2019; Lopez-Paz & Ranzato, 2017), regularization (Farajtabar et al., 2020), or parameter isolation (Rusu et al., 2016; Kang et al., 2023). Beyond architectural design, evaluation is a critical aspect. Recent studies advocate for per-iteration metrics to detect the “stability gap”—a transient but severe forgetting phase during training (De Lange et al., 2023)—and transferability-aware embeddings to leverage inter-task relationships (Wu et al., 2025). Online Continual Learning (OCL) (Cai et al., 2021) further emphasizes learning efficacy via a single-pass “test-then-train” protocol. This regime requires optimizing for immediate adaptation rather than long-term convergence, tracking performance through temporal moving averages instead of static end-of-task scalars.

Continual learning for LLMs. Adapting LLMs involves navigating vertical (pretrain-to-fine-tune) and horizontal (temporal shift) continuity. We focus on Continual Fine-Tuning (CFT), evaluated via benchmarks like TRACE (cross domain robustness) and CITB (long-horizon instruction streams). LoRA is the primary vehicle for CFT, with methods managing interference via subspace isolation, enforcing orthogonality through geometric constraints (Wang et al., 2023a; Liang & Li, 2024) or gradient projection (Qiao et al., 2024). TreeLoRA (Qian et al., 2025) organises per-layer LoRAs in a gradient-similarity tree, exploring task relationships via lower-confidence-bound bandits. However, these often sum branches with equal weight at inference, failing to select the most relevant parameters. While GainLoRA (Liang & Li, 2025) and SwitchCIT (Wu et al., 2024) introduce gating or switch networks, they require specialized optimization. Conversely, replay-based methods (Scialom et al., 2022; Wang et al., 2024; He et al., 2024) maintain performance but add memory overhead. Notably, recent work suggests that much degradation in LLMs stems from disrupted task alignment rather than true knowledge loss (Ma et al., 2025).

Routing and mixture-of-experts approaches. Sparse Mixture-of-Experts (MoE) architectures decouple model capacity from per-input computation by routing each token to a small subset of parallel experts (Fedus et al., 2022; Jiang et al., 2024). Learned routers discover interpretable data partitions aligned with latent clusters (Dikkala et al., 2023). In parameter-efficient fine-tuning (PEFT), LoRAMoE (Dou et al., 2024) and MoRAL (Yang et al., 2024) use balancing losses to preserve world knowledge, while MoA (Feng et al., 2024), SEE (Wang et al., 2025), and MLLM-CL (Zhao et al., 2025) assigns specific LoRA modules to different tasks or modalities, routing them at inference time to minimize interference. Despite their modularity, these methods generally rely on gradient-based gating and specialized training objectives and they are highly susceptible to routing drift and catastrophic forgetting within the router itself. Furthermore, they require complex balancing and regularization losses to prevent expert collapse. CaLLM overcomes this by using a non-parametric metric-based router, effectively decoupling router scaling from gradient optimization.

Prototypical networks and metric-based routing. CaLLM’s router is grounded in metric-based meta-learning (Vettoruzzo et al., 2024), drawing inspiration from Prototypical Networks. By computing distances to centroids in a learned embedding space, this non-parametric bias enables robust adaptation without inner-loop optimization. While prototypes have traditionally been used for low-level class-incremental classification (De Lange & Tuytelaars, 2021; Zhu et al., 2021), CaLLM transposes this to high-level system routing. Rather than using embeddings to represent a single class, CaLLM uses embedding prototypes to capture the geometric boundaries of entire multi-task streams. This allows the model to map task-level representation to isolated parameter spaces that are assigned to specific adapters. This yields a training-free router that inherits the modularity of MoE but avoids the overhead of learned gating. By combining parameter isolation with an online, no-replay constraint, CaLLM provides a scalable alternative that handles novel tasks via prototype expansion and bounds memory through an repurposing policy.

## 3 CONTINUAL ADAPTATION OF LLMS (CALLM)

CaLLM addresses catastrophic forgetting through a prototype-based routing mechanism coupled with dynamic, taskspecific parameter-efficient adapters. The central idea is to maintain a pool of lightweight adapter modules, each specializing in a region of the task embedding space. As illustrated in Figure 1, the system pipeline integrates four primary processes: (i) a non-parametric router that extracts embeddings from incoming data and maps data batches to adapters; (ii) an adapter pool that manages a collection of $P$ parameter-efficient adapters and their associated prototypes; (iii) a model trainer that performs localized SFT on selected adapters; and (iv) an evaluation pipeline that assesses performance across task streams to quantify learning progress and monitor for catastrophic forgetting. By routing data based on semantic similarity or spawning new adapters for sufficiently novel inputs, CaLLM decouples task capacity, allowing selective reuse or isolation as the task distribution evolves.

CaLLM operates in two modes: Batched (CaLLM-B), where tasks arrive sequentially, and Online (CaLLM-O), where small, randomly interleaved chunks from multiple tasks arrive in a non-stationary stream. Both share routing, training, and evaluation components but differ in data presentation order, evaluation protocol, and CL metric computation.

## 3.1 ADAPTER POOL

CaLLM maintains a pool $\mathcal { P } = \{ ( \ell _ { i } , \mathbf { p } _ { i } , c _ { i } ) \} _ { i = 1 } ^ { | \mathcal { P } | }$ of parameter-efficient adapters, with maximum capacity P. Each adapter $\ell _ { i }$ is paired with a prototype embedding $\mathbf { p } _ { i } \in \mathbb { R } ^ { d }$ that summarizes the data distribution it was trained on and a usage counter $c _ { i } .$ . The proposed system is both backbone- and adapter-agnostic, supporting any Transformer-based LLM and any PEFT method that can be independently instantiated, saved, and loaded (e.g., LoRA (Hu et al., 2021), (IA)<sup>3</sup> (Liu et al., 2022), or adapter modules (Houlsby et al., 2019)). We use the terms expert and adapter interchangeably here, since each expert in the pool is realized as a parameter-efficient adapter. In our experiments we use LoRA as the default adapter, where each $\ell _ { i }$ consists of low-rank decompositions $\Delta \dot { W } = B A$ injected into all linear projections in each Transformer block, parameterized with rank r and scaling factor $\alpha = 2 r$ . The modified forward pass for a projection W becomes:

$$
\mathbf { h } ^ { \prime } = W \mathbf { x } + \frac { \alpha } { r } B _ { i } A _ { i } \mathbf { x } ,\tag{1}
$$

where $A _ { i } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times r }$ and $B _ { i } \in \mathbb { R } ^ { r \times d _ { \mathrm { o u t } } }$ are the trainable low-rank factors of the currently loaded adapter $\ell _ { i }$ . All backbone parameters remain frozen throughout training and only the active adapter’s parameters are updated. This ensures that (i) the base model’s representations remain stable, and (ii) adapters from different tasks do not interfere with one another at the parameter level. Because only one adapter is active per forward pass while other reside in secondary storage, CaLLM ensures fixed-cost modularity: training FLOPs and GPU memory remain constant regardless of pool size |P|. Notably, routing does not compromise this property. Its additional pass through the frozen backbone is encoder-only, reuses the already-loaded model, and is invoked once per task (or once per chunk in the online setting) rather than once per query. Consequently, routing contributes only about 0.5% of training FLOPs and less than 0.1% of per-query inference cost (see Appendix B.2 for quantitative results).

![](images/8d1d55692311a3a20196a4488b116da454a7599d34c327fab8c08437747d1874.jpg)  
Figure 1: CaLLM system overview (training + evaluation). Each new data batch is embedded with a frozen base model (using mean-pooling). The non-parametric router manages a dynamic pool of specialized adapters, each mapped to a prototype. Based on semantic similarity between the new embedding and existing prototypes, the router triggers the reuse, creation, or repurposing of an adapter. The selected adapter is then trained or finetuned and the pool is updated. The diagram further details how this routing mechanism integrates with the distinct training and evaluation loops.

## 3.2 NON-PARAMETRIC ROUTER

The router maps incoming data to the most appropriate adapter in the pool via two stages: embedding extraction and prototype matching.

Embedding extraction. Given a batch of n input prompts, the router extracts a distributional representation e by performing a forward pass through the frozen backbone with all adapters disabled. Let $\mathbf { h } _ { j t } ^ { ( L ) }$ denote the hidden state at position t of sequence j in the final transformer layer L. We compute a single embedding by mean-pooling over valid (non-padding) tokens and averaging across the batch:

$$
\mathbf { e } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \frac { \sum _ { t = 1 } ^ { s _ { j } } m _ { j t } \cdot \mathbf { h } _ { j t } ^ { ( L ) } } { \sum _ { t = 1 } ^ { s _ { j } } m _ { j t } } ,\tag{2}
$$

where $m _ { j t } \in \{ 0 , 1 \}$ is the attention mask and $s _ { j }$ is the sequence length. Input sequences are tokenized with truncation at a configurable maximum length. By using the frozen backbone without adapters, the embeddings reflect the base model’s representation of the input distribution, providing a stable reference frame that works from t = 0 and does not drift as adapters are trained.

Prototype matching. The router computes the cosine distance between the new embedding e and each stored prototype p<sub>i</sub>:

$$
d _ { i } = 1 - { \frac { \mathbf { e } \cdot \mathbf { p } _ { i } } { \| \mathbf { e } \| \| \mathbf { p } _ { i } \| } } , \quad i = 1 , \ldots , | { \mathcal { P } } | .\tag{3}
$$

The adapter selection then follows a three-way decision rule governed by a distance threshold $\eta \colon$

1. Reuse $( d _ { \operatorname* { m i n } } \leq \eta ) \colon$ Load the closest adapter $\ell _ { \mathrm { a r g m i n } _ { i } d _ { i } }$ , update its prototype via an exponential moving average with equal weighting $\mathbf { p } _ { i }  \textstyle { \frac { 1 } { 2 } } ( \mathbf { p } _ { i } + \mathbf { e } )$ , and increment its usage counter $c _ { i }$

2. Create $( d _ { \operatorname* { m i n } } > \eta$ and $| \mathcal { P } | < P )$ : Initialize a new adapter from the base weights, add it to the pool with prototype $\mathbf { p } _ { \mathrm { n e w } } = \mathbf { e }$ and usage counter ${ c _ { \mathrm { n e w } } = 1 }$

3. Repurpose $( d _ { \operatorname* { m i n } } > \eta$ and $| { \mathcal { P } } | = P ) $ : Repurpose the least-used adapter slot (by usage count $c _ { i } )$ , reassign its prototype to e, reset its usage counter, and overwrite its weights during subsequent fine-tuning. Although repurposing can overwrite a rare adapter, our capacity analysis shows the router consistently groups similar tasks together, preventing such loss in practice.

During evaluation, the router operates in read-only mode. It finds the closest prototype and returns the associated adapter without creating or modifying prototypes. The threshold $\eta$ controls the granularity of the pool, lower values encourage more adapters (finer specialization), while higher values promote sharing across related distributions. We analyze the effect of pool capacity in §4.5.

When no fixed threshold is provided, an adaptive threshold is computed as half the minimum pairwise cosine distance between the closest prototype and all others:

$$
\eta _ { \mathrm { a d a p t } } = \frac { 1 } { 2 } \operatorname* { m i n } _ { j \neq i ^ { * } } \left( 1 - \frac { \mathbf { p } _ { i ^ { * } } \cdot \mathbf { p } _ { j } } { \| \mathbf { p } _ { i ^ { * } } \| \| \mathbf { p } _ { j } \| } \right) ,\tag{4}
$$

where $i ^ { * } = \arg \operatorname* { m i n } _ { i } d _ { i }$ is the closest prototype to the incoming embedding. This ensures the threshold tightens as the embedding space becomes more densely populated. When only one prototype exists, a default threshold of 0.1 is used.

## 3.3 TRAINING AND EVALUATION PROTOCOL

Each training step applies SFT on the selected adapter using the standard causal language modeling cross-entropy loss:

$$
\mathcal { L } = - \sum _ { t = 1 } ^ { | \mathbf { y } | } \log P _ { \theta _ { \mathrm { a d a p t } } } ( y _ { t } | y _ { < t } , \mathbf { x } ) ,\tag{5}
$$

where $\mathbf { x }$ is the input prompt, $\mathbf { y } = ( y _ { 1 } , \dots , y _ { | \mathbf { y } | } )$ is the target sequence, and $\theta _ { \mathrm { a d a p t } }$ denotes the adapter parameters. Optimization hyperparameters (e.g. optimizer, learning rate schedule, gradient clipping, numerical precision) as well as input formatting are all configurable and decoupled from the core method. Specific settings used in our experiments are reported in $\ S 4$

CaLLM supports two CL regimes that share the training objective above but differ in data presentation, evaluation timing, and metric computation. Full pseudocode for both protocols is provided in Appendix B.1.

Batched task-incremental protocol. In Batched mode, tasks arrive sequentially as complete datasets $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } , \ldots , \mathcal { D } _ { T }$ each with train, evaluation, and test splits (Algorithm 1). For each task with dataset $\mathcal { D } _ { t } ,$ the system first extracts an embedding from a sample of the training split and queries the router (with prototype updates enabled) to select or create an adapter. The selected adapter is then loaded onto the base model and fine-tuned on the full training split using the SFT objective (Eq. 5), with the evaluation split used for early stopping and checkpoint selection. After training, the adapter is saved to disk.

Evaluation proceeds in two phases. First, the trained adapter is evaluated on the current task’s test set to measure adaptation quality. Second, to assess backward transfer, all previously seen tasks $j < t$ are re-evaluated: for each prior task, the router operates in read-only mode (no prototype updates) to find the best-matching adapter under the current prototype set, and that adapter is loaded and evaluated on task $j ^ { \dagger } \mathbf { s }$ test set. Backward transfer is then computed as the average normalized performance change across prior tasks.

Online continual learning protocol. In Online mode, data arrives as a stream of small chunks randomly interleaved across multiple tasks in a non-stationary distribution. (Algorithm 2). Each chunk contains a fixed number of example (chunk size) from a single task.

Following the test-then-train paradigm, the protocol begins with the router extracting a chunk’s embedding $e _ { i }$ to select, create, or repurpose an adapter according to the logic described above. Once the parameter subspace is determined, the chunk is immediately evaluated using the selected adapter, or the frozen backbone if a new adapter is being initialized, prior to any gradient updates. This pre-update score provides an unbiased measure of the system’s current knowledge and is used to update the continual learning metrics defined below. After evaluation, the selected adapter is fine-tuned on the input chunk via SFT (Eq. 5). This single-pass procedure ensures structural consistency and strictly avoids post-update evaluation to prevent information leakage, allowing CaLLM to satisfy the core assumptions of online learning while tracking knowledge accumulation in real-time.

Continual learning metrics. We report the following measures to quantify plasticity and stability across the task stream:

• Average Performance: In the batched setting, after learning task t, we compute the normalized average over all tasks seen so far (each evaluated with its best-matching adapter via read-only routing): $\operatorname { A v g } _ { t } =$ $\textstyle { \frac { 1 } { t } } \sum _ { j = 1 } ^ { t } R _ { j , t } / s _ { j }$ , where $R _ { j , t }$ is the performance on task j after learning task t and $s _ { j }$ is the metric scale. This captures overall knowledge retention and adaptation quality at each point in the task sequence.

In the online setting, the equivalent is the Online Moving Average (OMA), a running normalized average updated at each step with the latest pre-update evaluation: OM $\begin{array} { r } { \mathsf { I A } _ { t } = \frac { 1 } { t } \sum _ { k = 1 } ^ { t } R _ { k } ^ { \mathrm { p r e } } / s _ { k } } \end{array}$ , where $s _ { k }$ is the task specific metric scale at step k. As a moving average over the stream, OMA tracks how the system’s overall performance evolves in real-time, reflecting its ability to generalize to new distributions.

• Backward Transfer (BWT): Measures the impact of learning new tasks on previously learned ones. In the batched setting, after learning task t: $\begin{array} { r } { \mathbf { B W T } _ { t } = \frac { 1 } { t - 1 } \sum _ { j = 1 } ^ { t - 1 } ( R _ { j , t } - R _ { j , j } ) / s _ { j } } \end{array}$ , where $R _ { j , j }$ is the performance immediately after learning task j and $s _ { j }$ is the metric scale.

In the online setting, BWT is computed as: $\mathbf { B W T } _ { j , t } = ( R _ { i } ^ { \mathrm { p r e } , t } - R _ { i } ^ { \mathrm { p r e , p r e v } } ) / s _ { j }$ , where $R _ { j } ^ { \mathrm { p r e , p r e v } }$ is the pre-update score from the most recent prior observation of task j. Unlike the batched variant above, online BWT measures the change between consecutive observations of the same task rather than from the initial post-training score.

## 4 EXPERIMENTS & RESULTS

We organize our experimental analysis around a progression from defining setup to understanding mechanism, followed by performance evaluation and stress-testing.

## 4.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate CaLLM on two complementary benchmarks that test different axes of CL difficulty:

1. TRACE-8 (Wang et al., 2023b) (Diversity & Shift) comprises 8 heterogeneous tasks spanning domainspecific QA (ScienceQA, FOMC, MeetingBank), multilingual processing (C-STANCE, 20Minuten), code completion (Py150), and math reasoning (NumGLUE-cm/ds) testing robustness to radical distribution shifts. We utilize the variant with 5000 training examples per task.

2. CITB-19 (Zhang et al., 2023) (Scale & Granularity) consists of 19 dialogue-centric tasks derived from Super-NaturalInstructions (Wang et al., 2022), categorized into Dialogue State Tracking (4), Dialogue Generation (11), and Intent Identification (4). This suite is used to test long-horizon streams and fine-grained intra-domain discrimination. To ensure training stability, we utilize a refined version of this benchmark augmented with 3000 high-quality synthetic examples per task (see Appendix A.5 for the generation and filtering protocol). This is the dataset version used in all our experiments.

Task performance is measured using standard metrics appropriate to each task type: accuracy for classification and multiple-choice tasks, ROUGE-L for summarization and all CITB-19 tasks, SARI for text simplification, and EDIM (edit-distance fuzzy match ratio) for code completion. In label-free stream settings, metric assignment is handled automatically via LLM-based task classification (Appendix A.6).

All input data is formatted using an Alpaca-style instruction template (Instruction / Response / <eos>), ensuring a uniform interface across diverse task types. Following the protocols defined in §3.3, the Batched setting processes tasks as complete sequential blocks, while the Online setting uses a non-stationary stream of max chunks steps. Each step consists of chunk size examples, with chunks randomly interleaved across tasks using a fixed seed to ensure reproducibility.

Baselines. We compare CaLLM against five representative CL strategies: Sequential Adapter Fine-tuning (Seq-FT), a single LoRA adapter fine-tuned sequentially on the task stream; Inference-only, the zero-shot performance of the frozen backbone; Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), a regularization-based method penalizing shifts in important weights; Experience Replay (ER) (Rolnick et al., 2019), a replay-based method with a fixed memory buffer; Orthogonal LoRA (oLoRA) (Wang et al., 2023a) a parameter-efficient CL baseline that learns tasks in orthogonal low-rank subspaces. TreeLoRA (Qian et al., 2025): layer-wise LoRAs organised by a hierarchical gradient-similarity tree with bandit-based task-relationship exploration.

All experiments use gemma-3-12b-it (4-bit quantized) as the backbone, selected for its superior zero-shot performance across all tasks (see Appendix A.1). Appendix A.2 shows implementation details and hyperparameter configuration. The code will be openly released upon paper acceptance.

## 4.2 ROUTER REPRESENTATIONS: WHY MODULAR ROUTING WORKS

CaLLM’s ability to mitigate interference in task-agnostic settings depends on the router’s capacity to recognize distributional boundaries from input features alone. Figure 2 visualizes 2D projections of the TRACE-8 benchmark. The router produces well-separated clusters, with radical domain shifts (e.g., code vs. multilingual text) pushed far apart in embedding space. This geometric separation is expected given the large distributional gaps between TRACE tasks, but it confirms that the router’s lightweight architecture and a basemodel representation is sufficient to capture these boundaries without additional finetuning.

Similarly, on CITB-19 (Figure 9(b) in Appendix D.2), the router discriminates between 19 fine-grained dialogue sub-tasks, such as dialogue state tracking and intent identification, despite their shared conversational format. This intrinsic geometric separation allows for reliable, task-agnostic routing. When pool capacity P is sufficient, each cluster maps to a dedicated adapter, achieving full parameter isolation. The consequences of restricting P and forcing cluster merges are analyzed in §4.5.

![](images/317a6bf2ab291d9335de8384cfd44cfaa344d5c0260d68a174c4bc6713964b56.jpg)  
Figure 2: Router Embedding Space (2D Projection) for TRACE-8. t-SNE projections of the router’s embeddings, colored by task identity.

## 4.3 TASK-INCREMENTAL RESULTS

We first evaluate CaLLM’s end-to-end performance in the standard batched task-incremental setting, where tasks arrive one at a time and the system trains on each complete dataset sequentially.

Final performance by individual task for CALLM-B and other baselines for TRACE benchmark is shown in Table 1. CaLLM-B achieves the highest score among all baseline methods on 4 of 8 tasks. The most significant gains over the inference-only baseline appear on specialized domains where the pre-trained backbone is weakest: code completion (Py150, +2.33 in EDIM), science reasoning (ScienceQA, 0.35 → 0.81 in accuracy), and math (NumGLUE-ds, 6× improvement in accuracy). TreeLoRA’s wins concentrate on the two arithmetic splits, where its hierarchical gradient-similarity tree consolidates closely related skills (mean accuracy 0.60 across NumGLUE-cm/ds, versus 0.28 for CaLLM-B), but it collapses on Py150 (5.52 vs. 17.24 in EDIM), where the code-generation gradient signature is dissimilar from the natural language tasks and the tree potentially mis-routes the adapter, a failure mode that CaLLM-B’s prototype router avoids by treating each task as a first-class cluster. Conversely, on C-STANCE and 20Minuten, performance is largely inherited from the backbone; on the former, fine-tuning slightly degrades pre-existing competence, while on the latter, all methods fall within a narrow 2-point SARI band. Because 20Minuten is the final task in the sequence, no method suffers from forgetting, narrowing the performance gap between CaLLM-B and baselines. Notably, CL baselines underperform Inference-Only on several tasks (e.g., C-STANCE, FOMC, MeetingBank, Py150). We attribute this to Inference-Only’s strong large-scale backbone—whose pre-trained competence makes it a strong baseline rather than a trivial lower bound—and to its avoidance of the catastrophic forgetting that erodes CL baselines as the task stream grows (Appendix A.3 shows this gap persists even with per-baseline hyperparameter tuning). CaLLM-B avoids this trade-off by routing task-specific updates to isolated experts, preserving the backbone’s zero-shot competence while enabling specialization.

Figure 3 shows the evolution of average normalized per-task performance across the sequential task stream both for TRACE-8 and CITB-19 datasets, while a detailed per-task performance breakdown can be seen in Tables 7 and 8 in Appendix C.2.1. On TRACE-8, baselines struggle with radical distribution shifts. EWC’s performance declines significantly upon encountering tasks that require more complex reasoning (Py150 at Task-4), while ER and oLoRA exhibits a steady downward trend in later tasks. In contrast, CaLLM-B demonstrates robust stability throughout the entire sequence, by routing incompatible distributions (e.g., code vs. math) to separate experts, as shown in Figure 2(a). CaLLM-B finishes the 8-task sequence with an average normalized score of 0.44, substantially outperforming the Inference-Only baseline (0.31) and ER (0.31). We additionally verified that using Gemma 3’s native chat template for the Inference-Only baseline does not yield a stronger competitor, making our reported Inf-Only a conservative comparison (Appendix C.3). The narrow variance bands (±1 SD) confirm CaLLM-B’s robustness to training stochasticity. On CITB-19 (Fig. 3b), the challenge shifts to sequence length. While Seq-FT degrades rapidly due to interference within a single adapter, CaLLM-B maintains a stable trajectory across all 19 tasks. By instantiating new experts as needed (P ≈ 19), the system prevents parameter overloading. The tight variance bands across all 19 steps confirm that this stability is not degrading even when sequence task length increases. CaLLM-B achieves the highest score among CL methods on 12 of 19 tasks, demonstrating broad competence across fine-grained CITB dialogue subtypes (see Table 9, Appendix C.2.2). However, fine-tuning on CITB-19 does not improve over Inference-Only for any method, likely because dialogue tasks are within the natural competence of a 12B instruction-tuned model (see Appendix A.4).

Table 1: Per-Task Final Performance on TRACE-8 (Batched). Final scores after training on all tasks sequentially.
<table><tr><td>Task</td><td>Metric</td><td>Inf-Only</td><td>Seq-FT</td><td>EWC</td><td>ER</td><td>oLoRA</td><td>TreeLoRA 2</td><td>CaLLM-B</td></tr><tr><td>C-STANCE</td><td>acc (0–1)</td><td>0.62</td><td>0.03</td><td>0.28</td><td>0.04</td><td>0.09</td><td>0.41</td><td>0.43</td></tr><tr><td>FOMC</td><td>acc (0–1)</td><td>0.64</td><td>0.30</td><td>0.31</td><td>0.14</td><td>0.06</td><td>0.51</td><td>0.68</td></tr><tr><td>MeetingBank</td><td>ROUGE-L (0–1)</td><td>0.21</td><td>0.06</td><td>0.10</td><td>0.10</td><td>0.16</td><td>0.11</td><td>0.23</td></tr><tr><td>Py150</td><td>EDIM (0–100)</td><td>14.91</td><td>11.96</td><td>10.55</td><td>12.98</td><td>9.81</td><td>5.52</td><td>17.24</td></tr><tr><td>ScienceQA</td><td>acc (0–1)</td><td>0.35</td><td>0.13</td><td>0.61</td><td>0.19</td><td>0.00</td><td>0.72</td><td>0.81</td></tr><tr><td>NumGLUE-cm</td><td>acc (0–1)</td><td>0.05</td><td>0.14</td><td>0.12</td><td>0.06</td><td>0.04</td><td>0.67</td><td>0.15</td></tr><tr><td>NumGLUE-ds</td><td>acc (0–1)</td><td>0.07</td><td>0.28</td><td>0.15</td><td>0.13</td><td>0.05</td><td>0.54</td><td>0.42</td></tr><tr><td>20Minuten</td><td>SARI (0–100)</td><td>39.22</td><td>38.06</td><td>40.06</td><td>39.63</td><td>39.91</td><td>41.05</td><td>39.49</td></tr><tr><td>Avg. perf.</td><td>normalized (0–1)</td><td>0.29</td><td>0.16</td><td>0.26</td><td>0.15</td><td>0.15</td><td>0.39</td><td>0.41</td></tr><tr><td>BWT</td><td>(higher is better)</td><td>一</td><td>-0.21</td><td>-0.10</td><td>-0.18</td><td>-0.11</td><td>1</td><td>0.01</td></tr></table>

![](images/8969732c7801263bf9e48557b18a2cb832d6db6c03b979c5cd713acf2725ed12.jpg)

![](images/35c40023cc79397a4b58b02ebd132a8a429cb36de9970ba44748baad5345be45.jpg)  
Figure 3: Task-Incremental Learning Performance (mean ±1 std). Average normalized score as tasks are added sequentially, for CaLLM-B and all baselines. (a) TRACE-8. (b) CITB-19.

We also analyzed the BWT dynamics across both benchmarks in Figure 7 (Appendix C.2.1) in the batched scenario. Across both TRACE-8 and CITB-19, CaLLM-B is the only method that consistently maintains a positive, or close to 0, BWT, with a very low variance across different random seeds. Conversely, all competing baselines exhibit a steady degradation in BWT as more tasks are introduced to the stream. This stark contrast demonstrates that CaLLM successfully learn new tasks without forgetting the previously acquired knowledge.

These gains come at negligible additional compute.Because only one adapter is active per forward pass, CaLLM consumes the same 7339 TFLOPs of training compute as single-adapter Seq-FT. The router adds only a small increment: Using the standard 2N FLOPs-per-token estimate with $\bar { N } \approx \bar { 1 } 2 \times 1 0 ^ { 9 }$ for gemma-3-12b-it, a routing call over n ≈ 20 prompts of ∼ 10 tokens costs ≈ 4.8 TFLOPs, so across TRACE-8’s eight tasks the router consumes ≈ 38.4 TFLOPs (only ≈ 0.5% of training). At inference this amortizes over all evaluation queries within a task, giving a per-query overhead ≪ 0.1% (see Appendix B.2 for further details).

## 4.4 ONLINE STREAMING DYNAMICS

Having established CaLLM’s task-incremental performance, we now examine the fine-grained dynamics of online learning: How does the router behave over time as the data stream unfolds, and how does expert specialization emerge without supervision? Unlike the batched setting, we do not report multi-seed variance here expecting the OMA metric averaging over 500 evaluation steps to provide inherent statistical stability.

Figure 4 summarises CaLLM-O’s behavior over the TRACE online stream. The top panel (Figure 4a) tracks the OMA performance whereas the bottom panel (Figure 4b) zooms into a representative 100-chunk window (chunks 200–300) of the task × expert assignment heatmap, where each row is a task, each column a chunk, and color encodes the assigned expert. The full 500-chunk heatmap is provided in Appendix D.3 showing this behavior persists for the whole stream.

Knowledge Accumulation. The OMA curve (Figure 4a) rises steadily from ∼0.29 to ∼0.40 over the 500-chunk stream, confirming that CaLLM-O accumulates knowledge faster than it forgets, even in the absence of explicit task boundaries. After brief initial volatility during the first ∼50 chunks, where the router is still forming its expert assignments and initialized experts building up knowledge over tasks, the trajectory stabilizes into a monotonic upward trend, indicating that new tasks are absorbed without disrupting previously learned ones.

Expert Specialization. The zoomed heatmap (Figure 4b) reveals consistent task–expert matching within the displayed window. The top strip shows the aggregate adapter timeline with distinct color band showing how frequently the adapter changes through the stream. The per-task rows confirm that each task is reliably routed to the same expert when it reappears in the stream, indicating that the router functions as a distribution-sensitive switch rather than randomly cycling through experts.

![](images/c2221cff13a1d8292fff06ef0f9f1ad49e48e365a54b4de94d57fe175e76ce34.jpg)

![](images/4f620ebfaa69c2a302e631b94218fbff945d7538fa96b3b1b9558aca6f193d6d.jpg)  
Figure 4: CaLLM Online Learning Dynamics (TRACE Stream, P=8). (a) OMA. (b) Task × expert assignment heatmap; aggregate adapter timeline on top and per-task routing below. See Appendix D.3 for the full 500-chunk view.

## 4.5 THE ROLE OF POOL CAPACITY

The router embedding analysis (§4.2) showed that the router is capable of separating task clusters both in diverse and fine-grained task distributions. When pool capacity is unconstrained, each cluster maps to a unique expert, achieving full isolation. But in practice, memory budgets are limited and the number of tasks to be seen is not always known in advance. We now investigate what happens as pool size P is progressively reduced, revealing a spectrum of routing behaviors from perfect isolation to system collapse.

## 4.5.1 EXPERT ASSIGNMENT DYNAMICS IN TASK-INCREMENTAL LEARNING

Figure 5 visualizes the evolution of expert assignments and the corresponding BWT curves on TRACE-8 as pool size varies from $P = 8$ to $P = 2$ (see Appendix D.2 for extended plots).

• Ideal Isolation $( P = 8 ) \colon$ The router assigns unique experts to each task, resulting in zero BWT loss. This confirms that the router’s embedding clusters (Figure 2(a)) translate directly into correct expert assignment.

• Constructive Clustering $( P = 6 ) \colon$ Semantically related tasks (e.g., NumGLUE-cm/ds) are co-routed to Expert 5, while other tasks retain dedicated experts. Because these merged tasks share similar gradient directions, parameter sharing is beneficial rather than destructive. The BWT curve remains relatively stable with a slight dip at Step 7 when the merge occurs.

• Destructive Overload $( P = 4 ) \colon \mathrm { A }$ “super-expert” absorbs disparate tasks (QA, Stance, Math), leading to conflicting updates and a sharp BWT drop to −0.19. Notably, the router still identifies $P y I 5 0$ (code) as maximally distinct, preserving its isolation in a separate expert even under severe constraints.

• System Collapse $( P \ = \ 2 ) \colon$ All tasks merge into one expert and the BWT curve declines to −0.25 as catastrophic forgetting becomes unavoidable. This represents the lower bound of the capacity spectrum and illustrates the fundamental trade-off between memory efficiency and interference prevention.

![](images/1a6369854f2a7e2c7675f763bb5efd9ab211d5e3ecdd3db0fdbbabfeac841489.jpg)  
Figure 5: Router behavior under pool capacity constraints for TRACE-8 in Batched scenario.

## 4.5.2 POOL CAPACITY CONSTRAINTS IN THE ONLINE SCENARIO

The batched analysis reveals a clear capacity-stability trade-off which we investigate here for the online scenario. We evaluate CaLLM-O over 500-chunk streams with varying pool sizes on TRACE $( P \in \{ 8 , 6 , 4 \} )$ and CITB $( P \in \{ 1 9 , 1 0 , 3 \} )$ ), against SeqFT $( P { = } 1 )$ , revealing a non-monotonic relation between performance and capacity.

Stability over capacity. On TRACE, P=4 achieves the highest final OMA (0.42) and the best BWT, outperforming both P=8 (0.39) and $P { = } 6 ~ ( 0 . 3 8 )$ (see Appendix C.1 for full plots). Routing stability—the consistency of taskto-expert assignment—outweighs raw capacity (see Appendix D.3.1 Fig. 12). At P=4, the router converges on four semantically coherent groups: (1) C-STANCE; (2) quantitative tasks (NumGLUE-ds, NumGLUE-cm, Py150); (3) FOMC; (4) long-form understanding (ScienceQA, MeetingBank, 20Minuten). Interleaved chunks within each group provide implicit replay via alternating gradient signals, fostering constructive interference. Conversely, P=6 suffers from task displacement: 20Minuten is scattered across multiple experts, preventing specialization. P=8 achieves perfect isolation but forgoes the co-assignment regularization that benefits the streaming regime.

Similarity and forgetting. In more homogeneous streams like CITB-19, performance is less sensitive to capacity; since similar tasks tolerate compression well, though smaller pools still cause deeper forgetting spikes. Task similarity ultimately determines how sensitive a stream is to pool size, diverse streams need larger, more carefully tuned pools.

## 5 CONCLUSION

We presented CaLLM, a modular continual learning method for LLMs that pairs a bounded pool of lightweight adapters with a non-parametric, prototype-based router operating entirely in the frozen backbone’s embedding space, requiring no gradient-based training or task labels at inference. Incoming data are matched to the nearest expert via cosine distance, with new experts spawned or repurposed automatically when the data is sufficiently novel. Our experiments on TRACE-8 and CITB-19 demonstrate that CaLLM is effective across both batched and online CL settings, where it reliably discovers distributional boundaries from input features alone, and translates this geometric separation into accurate, task-agnostic expert assignments. Although CITB-19 shows that the router can separate fine-grained tasks that share a language and format, we do not test whether it can keep tasks together when they overlap conceptually but differ lexically (e.g. the same task across languages), which we leave to future evaluation. Notably, we observe that limiting the adapter pool size can boost performance by forming semantically coherent groups and using interleaved gradients as implicit replay, without requiring memory buffers.

Future directions include integrating online AutoML to adapt currently fixed hyperparameters as the stream unfolds, explicitly merging related adapters to consolidate shared knowledge, and extending prototype-based routing to multimodal streams spanning vision, language, and other modalities simultaneously.

## REFERENCES

Zhipeng Cai, Ozan Sener, and Vladlen Koltun. Online continual learning with natural distribution shifts: An empirical study with visual data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8281–8290, 2021.

Matthias De Lange and Tinne Tuytelaars. Continual prototype evolution: Learning online from non-stationary data streams. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8250–8259, 2021.

Matthias De Lange, Gido M. van de Ven, and Tinne Tuytelaars. Continual evaluation for lifelong learning: Identifying the stability gap. In International Conference on Learning Representations (ICLR), 2023.

Nishanth Dikkala, Nikhil Ghosh, Raghu Meka, Rina Panigrahy, Nikhil Vyas, and Xin Wang. On the benefits of learning to route in mixture-of-experts models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 9376–9396, 2023.

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. LoRAMoE: Alleviating world knowledge forgetting in large language models via MoE-style plugin. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1932–1945, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. In Proceedings ofthe 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–40, 2022.

Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. Mixture-of-LoRAs: An efficient multitask tuning for large language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pp. 11371–11380, 2024.

Daniel Han, Michael Han, and Unsloth team. Unsloth. https://github.com/unslothai/unsloth, 2023.

Jinghan He, Haiyun Guo, Kuan Zhu, Zihan Zhao, Ming Tang, and Jinqiao Wang. SEEKR: Selective attention-guided knowledge retention for continual learning of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3254–3266, Miami, Florida, USA, November 2024. Association for Computational Linguistics.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. arXiv preprint arXiv:2403.01244, 2024.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Haeyong Kang, Jaehong Yoon, Sultan Rizky Madjid, Sung Ju Hwang, and Chang D. Yoo. Forget-free continual learning with soft-winning subnetworks. arXiv preprint arXiv:2303.14962, 2023.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy ofSciences, 114(13):3521–3526, 2017.

Yan-Shuo Liang and Wu-Jun Li. InfLoRA: Interference-free low-rank adaptation for continual learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23638–23647, 2024.

Yan-Shuo Liang and Wu-Jun Li. Gated integration of low-rank adaptation for continual learning of large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Phang, Lingling Tian, Colin Raffel, and Mohit Bansal. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Xinbei Ma, Zhuo Tang, Zhuowei Chen, Cheng Peng, and Lei Clifton. Spurious forgetting in continual learning of language models. In International Conference on Learning Representations (ICLR), 2025.

Yu-Yang Qian, Yuan-Ze Xu, Zhen-Yu Zhang, Peng Zhao, and Zhi-Hua Zhou. TreeLoRA: Efficient continual learning via layer-wise LoRAs guided by a hierarchical gradient-similarity tree. arXiv preprint arXiv:2506.10355, 2025.

Fuli Qiao, Guizhong Liu, et al. Learn more, but bother less: Parameter efficient continual learning. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Gregory Wayne. Experience replay for continual learning. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

Thomas Scialom, Tuhin Chakrabarty, and Smaranda Muresan. Fine-tuned language models are continual learners. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. Continual learning of large language models: A comprehensive survey. ACM Computing Surveys, 2025.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. Detecting pretraining data from large language models. arXiv preprint arXiv:2310.16789, 2023.

Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS), pp. 4077–4087, 2017.

Anna Vettoruzzo, Mohamed-Rafik Bouguelia, Joaquin Vanschoren, Thorsteinn Rognvaldsson, and K. C. Santosh.¨ Advances and challenges in meta-learning: A technical review. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(7):4763–4779, 2024.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. Orthogonal subspace learning for language model continual learning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pp. 10658–10671, Singapore, December 2023a. Association for Computational Linguistics.

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, and Xuanjing Huang. TRACE: A comprehensive benchmark for continual learning in large language models. arXiv preprint arXiv:2310.06762, 2023b.

Yifan Wang, Yafei Liu, Chufan Shi, Haoling Li, Chen Chen, Haonan Lu, and Yujiu Yang. InsCL: A data-efficient continual learning paradigm for fine-tuning large language models with instructions. arXiv preprint arXiv:2403.14555, 2024.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, and Daniel Khashabi. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. arXiv preprint arXiv:2204.07705, 2022.

Zhilin Wang, Yafu Li, Xiaoye Qu, and Yu Cheng. SEE: Continual fine-tuning with sequential ensemble of experts. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 7418–7432. Association for Computational Linguistics, 2025.

Xinbo Wu, Max Hartman, Vidhata Arjun Jayaraman, and Lav R. Varshney. SwitchCIT: Switching for continual instruction tuning of large language models. arXiv preprint arXiv:2407.11780, 2024.

Yanru Wu, Jianning Wang, Xiangyu Chen, Enming Zhang, Hanbing Liu, Yang Tan, and Yang Li. Exploiting task relationships in continual learning via transferability-aware task embeddings. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, and Di Wang. MoRAL: MoE augmented LoRA for LLMs’ lifelong learning. arXiv preprint arXiv:2402.11260, 2024.

Zihan Zhang, Meng Fang, Ling Chen, and Mohammad-Reza Namazi-Rad. CITB: A benchmark for continual instruction tuning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 9443–9455, Singapore, December 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023. findings-emnlp.633/.

Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, and Zhaoxiang Zhang. MLLM-CL: Continual learning for multimodal large language models. arXiv preprint arXiv:2506.05453, 2025.

Kai Zhu, Yang Cao, Wei Zhai, Jie Cheng, and Zheng-Jun Zha. Self-promoted prototype refinement for few-shot classincremental learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6801–6810, 2021.

## A APPENDIX

## A.1 BACKBONE SELECTION

Table 2 compares zero-shot performance across three candidate backbones – Llama-3.1-8B-Instruct, gemma-3-12b-it, and Qwen3-8B– on the TRACE-8 benchmark. While all models struggle on the NumGLUE (math tasks), confirming the necessity of domain-specific adaptation, gemma-3-12b-it consistently exhibits the strongest generalist prior. It achieves the highest zero-shot accuracy across all eight tasks, with marked advantages on C-STANCE (0.62 vs. 0.20), FOMC (0.64 vs. 0.14), and Py150 (14.91 vs. 6.42). Selecting gemma-3-12b-it as our primary backbone ensures that our results evaluate the efficacy of the CaLLM framework rather than compensating for a weak base-model initialization.

Table 2: Inference-only (Zero-Shot) Performance on TRACE-8. gemma-3-12b-it achieves the highest score on all 8 tasks. All models show low performance on math, motivating the need for continual adaptation.
<table><tr><td>Task</td><td>Metric</td><td>Llama-3.1-8B-Instruct</td><td>gemma-3-12b-it</td><td>Qwen3-8B</td></tr><tr><td>C-STANCE</td><td>acc (0-1)</td><td>0.20</td><td>0.62</td><td>0.20</td></tr><tr><td>FOMC</td><td>acc (0–1)</td><td>0.14</td><td>0.64</td><td>0.04</td></tr><tr><td>MeetingBank</td><td>ROUGE-L (0–1)</td><td>0.05</td><td>0.21</td><td>0.04</td></tr><tr><td>Py150</td><td>EDIM (0–100)</td><td>6.42</td><td>14.91</td><td>0.00</td></tr><tr><td>ScienceQA</td><td>acc (0–1)</td><td>0.09</td><td>0.35</td><td>0.07</td></tr><tr><td>NumGLUE-cm</td><td>acc (0–1)</td><td>0.01</td><td>0.05</td><td>0.00</td></tr><tr><td>NumGLUE-ds</td><td>acc (0-1)</td><td>0.02</td><td>0.07</td><td>0.00</td></tr><tr><td>20Minuten</td><td>SARI (0–100)</td><td>36.2</td><td>39.22</td><td>37.0</td></tr></table>

## A.2 IMPLEMENTATION DETAILS AND HYPERPARAMETERS

As detailed in Appendix A.1, all experiments use gemma-3-12b-it (4-bit quantized) as the primary backbone. We maintain this fixed model for all comparisons, though CaLLM is backbone-agnostic and compatible with any Transformer based LLM. The router utilizes the same frozen backbone as a feature extractor, where the last hidden states are mean-pooled over valid tokens to obtain a sample embedding and then averaged across the input batch to produce a single representation for routing.

LoRA adapters are trained using the HuggingFace SFTTrainer with an 8-bit AdamW optimizer, gradient clipping at norm 1.0, and bfloat16 mixed precision without sequence packing. Additional training constants include a maximum sequence length of 1024, temperature of 1.0, LoRA dropout of 0.05, and a warmup ratio of 0.05. In the online setting, we use max $\AA . c h u n k s = 5 0 0$ and chunk size = 40, with chunks generated via random resampling. All input follows the Alpaca prompt style detailed in $\ S 4 . 1$ , and the router uses the adaptive threshold defined in Eq. 4.

Batched experiments are reported as the mean and standard deviation across three random seeds with optimal configurations selected based on final mean validation performance. Online experiments are instead performed as a single run due to the significant computational cost of 500-chunk streams. We rely on the OMA metric, which averages over 500 evaluation steps, to provide inherent statistical stability. Table 3 summarizes the specific hyperparameters selected via grid search over learning rate $\in \{ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ , LoRA rank $\in \{ 4 , 8 , 1 6 \}$ , and epochs $\in \{ 3 , 5 \}$ . To mitigate the substantial computational overhead of tuning long-horizon streams, hyperparameter optimization for the online setting was restricted to a truncated proxy window of 200 evaluation steps under a standard pre-update (test-than-train) evaluation protocol. All experiments were conducted on a single NVIDIA RTX A6000 GPU with 40GB of VRAM using Unsloth (Han et al., 2023) and the trl, peft and transformers libraries from HuggingFace.

Table 3: CaLLM hyperparameters per scenario. All runs use the HuggingFace SFTTrainer with 8-bit AdamW, gradient clipping at norm 1.0, and bfloat16 mixed precision. LR = learning rate, $r / \alpha = \mathrm { L o R A }$ rank / scaling factor, Batch = batch size, GAS = gradient accumulation steps, LR Sched. = learning rate schedule.
<table><tr><td>Setting</td><td>LR</td><td>r</td><td>α</td><td>Epochs</td><td>Batch</td><td>GAS</td><td>LR Sched.</td></tr><tr><td>CaLLM-B, TRACE</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>16</td><td>32</td><td>3</td><td>16</td><td>4</td><td>cosine</td></tr><tr><td>CaLLM-B, CITB</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>8</td><td>16</td><td>5</td><td>25</td><td>4</td><td>cosine</td></tr><tr><td>CaLLM-O, TRACE</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td>4</td><td>8</td><td>1</td><td>10</td><td>2</td><td>linear</td></tr><tr><td>CaLLM-O, CITB</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>8</td><td>8</td><td>1</td><td>20</td><td>1</td><td>cosine</td></tr></table>

Table 4: Baseline hyperparameter ablation on TRACE-8 (Batched). Original rows use the configuration from the main comparison; tuned rows are selected from the same learning-rate/epoch grid used for CaLLM while keeping LoRA rank, scaling factor, and dropout fixed.
<table><tr><td>Method</td><td>Config.</td><td>LR</td><td>Epochs</td><td>C-STANCE</td><td>FOMC</td><td>MeetingBank</td><td>Py150</td><td>ScienceQA</td><td>NumGLUE-cm</td><td>NumGLUE-ds</td><td>20Minuten</td></tr><tr><td>ER</td><td>Original</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>3</td><td>0.04</td><td>0.14</td><td>0.10</td><td>12.98</td><td>0.19</td><td>0.06</td><td>0.13</td><td>39.63</td></tr><tr><td>ER</td><td>Tuned</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>5</td><td>0.11</td><td>0.19</td><td>0.12</td><td>13.04</td><td>0.22</td><td>0.09</td><td>0.15</td><td>39.58</td></tr><tr><td>EWC</td><td>Original</td><td>1 × 10−4</td><td>3</td><td>0.28</td><td>0.31</td><td>0.10</td><td>10.55</td><td>0.61</td><td>0.12</td><td>0.15</td><td>40.06</td></tr><tr><td>EWC</td><td>Tuned</td><td>5 × 10−4</td><td>3</td><td>0.17</td><td>0.35</td><td>0.09</td><td>10.61</td><td>0.55</td><td>0.21</td><td>0.11</td><td>39.01</td></tr></table>

## A.3 DETAILS ABOUT THE BASELINES

To evaluate CaLLM, we compare it against several representative strategies from the CL literature, adapted for the PEFT regime. To ensure a fair and rigorous comparison, all baselines share the same core architectural parameters as CaLLM and the same LoRA configuration (rank, alpha, and dropout). We then utilize the optimal hyperparameters (such as regularization strength or buffer size) specific to each method as reported in their respective original papers, unless stated otherwise

To understand the potential impact of this design choice on the fairness of baseline comparisons, we conducted an ablation study of hyperparameter search on selected continual learning baselines, namely ER and EWC. The search follows the CaLLM search space over learning rate and number of training epochs, using a grid search over learning rate $\in \{ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ and epochs ∈ {3, 5}. The ablation focuses on the TRACE benchmark, due to its higher variation across tasks and smaller number of tasks. Results of this search are presented in Table 4, including the original hyperparameter values and performance results. As shown in the table, the search selects a hyperparameter combination different from the originally chosen values for each baseline (where the per-task performance does not always improve compared with the original setting). However, the magnitude of the performance difference is small, and may also be explained by the inherent randomness of LLM generations. Importantly, this does not change the ordering of baseline–method comparisons for any task in the analysis, supporting the original design choice.

• Inference only is a simple baseline where the base model (e.g. gemma-3-12b-it) is used during inference in a continual/online way, every time a new task/chunk is presented to the model.

• Sequential Adapter Fine-Tuning (Seq-FT): A single LoRA adapter is fine-tuned sequentially on the task stream. This represents the lower bound for stability, as it is highly susceptible to catastrophic forgetting.

• Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017): A regularization-based approach that penalizes changes to parameters that were important for previous tasks. Importance is estimated via the diagonal of the Fisher Information Matrix. We apply EWC to the LoRA weights specifically.

• Experience Replay (ER) (Rolnick et al., 2019): A memory buffer stores a subset of samples from previous tasks. During the adaptation to a new task, each training batch is augmented with replayed samples from the buffer to preserve past knowledge. In our experiments, we use a fixed buffer size of 1000 samples.

• Orthogonal LoRA (OLoRA) (Wang et al., 2023a): A subspace isolation method that constrains the LoRA weight updates to be orthogonal to the pre-trained weights or previous task directions, aiming to prevent gradient-based interference. To ensure compatibility with the Gemma backbone and avoid out-of-memory (OOM) errors, we modified the original OLoRA implementation as follows: (i) reduced the evaluation batch size to 4, instead of using the batch size, and (ii) offloaded the orthogonal loss calculation to the CPU. For the TRACE-8 online stream, we utilized batch size = 8, r = 4, α = 8. For CITB, we used batch size = 2, r = 4, α = 8, and gradient accumulation step of 8.

• K-D Tree of Low-Rank Adapters (TreeLoRA) (Qian et al., 2025): Introduces layer-wise adapters to construct hierarchical tree based on the criteria of gradient similarity, exploring task relationships via lower confidence-bound bandits.

## A.4 GEMMA-3-12B-IT TRAINING DATA AND CITB OVERLAP

Gemma 3 models are well-suited for text-generation tasks such as question answering, summarization, and reasoning. The 12B model used in our work was trained on 12T tokens consisting of web documents (in over 140 languages), code, mathematics, and images. The pre-training data is not released openly and there is limited information about the data sources. Additionally, the IT variant has been instruction-tuned using knowledge distillation from larger teacher models (likely Gemini models).

CITB is built on top of the Super-NaturalInstructions dataset (Wang et al., 2022). In this work, we focus on the InstrDialog task stream, which consists of 19 dialogue-related tasks spanning three categories: dialogue state tracking (4 tasks), dialogue generation (11 tasks), and intent identification (4 tasks). All tasks are evaluated using the ROUGE-L metric.

While there is no clear information about the data used by Gemma 3 for pre-training or fine-tuning, it is plausible that the Super-NaturalInstructions dataset (on which CITB is based) was included in the training data of either Gemma 3 or the Gemini teacher models. This would explain why LoRA fine-tuning on CITB does not improve over the inference-only baseline. Despite this, all other baselines suffer from sequential fine-tuning, suggesting that CaLLM maintains the original performance even when the data has already been seen.

To investigate potential data contamination, we applied the method of Shi et al. (2023), which computes per-token probabilities under the hypothesis that “seen” examples have fewer low-probability outliers with respect to “unseen” words. We compared 500 CITB samples against 309 “old” Wikipedia pages (likely within Gemma 3’s training data) and 97 Wikipedia pages created in the following 60 days (unlikely to be in the training data). The results are as follows:

• Member mean: 0.005722

• Non-member mean: 0.003475

• CITB mean: 0.002848

From these results we can conclude that CITB has an average probability closer to the non-member samples than the (assumed) member samples, suggesting that CITB is likely not in the training data of Gemma 3.

This appears to contradict our earlier hypothesis that Gemma 3 has already seen CITB data. Indeed, Figure 3b shows that fine-tuning with CITB data does not improve over the inference-only baseline. However, it is important to note that the method of Shi et al. (2023) is only an empirical heuristic, and the candidate member and non-member sets are based on assumptions about Wikipedia page creation dates rather than ground-truth training data membership.

## A.5 SYNTHETIC DATA GENERATION FOR CITB

The original CITB benchmark, defined as InstrDialog in the original paper (Zhang et al., 2023), consists of 19 tasks derived from Super-NaturalInstructions, detailed in Table 5.

Table 5: The 19 dialogue tasks in the CITB InstrDialog stream.
<table><tr><td>Category</td><td>Task Type</td><td>Count</td></tr><tr><td>Dialogue State Tracking</td><td>Slot Filling, State Update</td><td>4</td></tr><tr><td>Dialogue Generation</td><td>Response Gen, Clarification</td><td>11</td></tr><tr><td>Intent Identification</td><td>Classification, Intent Detection</td><td>4</td></tr><tr><td>Total</td><td></td><td>19</td></tr></table>

Each task contains only 500 training instances, which is insufficient for effectively training LoRA adapters and creates an imbalance when compared to the TRACE-8 benchmark (5000 instances per task). To ensure a fair comparison and improve adaptation stability, we generated synthetic training data for each CITB task. Inspired by self-synthesized rehearsal strategies (Huang et al., 2024), we utilized the frozen gemma-3-12b-it model as a data generator. We employed a few-shot prompting strategy, providing the model with two randomly sampled real instances in-context and tasking it to generate a novel example following the same Alpaca-style format. By iterating this process with randomized in-context examples, we produced an initial pool of 6000 synthetic instances per task.To ensure data quality and distributional consistency, we applied a two-stage refinement pipeline:

1. Deduplication: We removed all exact string matches and near-duplicates to ensure diversity within the synthetic pool.

2. Centroid-based filtering: To retain only high-quality, representative samples, we converted the synthetic instances into embeddings using the frozen base model. We then applied K-means clustering to the embeddings of each task. Finally, we calculated the Euclidean distance between each instance and its respective cluster centroid, selecting the 3000 instances closest to the centroid for the final training set.

This procedure resulted in a refined CITB dataset containing 3000 examples per task. This refined version maintains the semantic integrity of the original tasks while providing sufficient signal for parameter-efficient fine-tuning, and it serves as the standard CITB data stream used for all experiments and baseline comparisons presented in this paper.

## A.6 AUTOMATIC METRIC SELECTION

When ground-truth metric labels are unavailable (i.e., in label-free stream configurations), CaLLM dispatches an automatic metric selection procedure at the start of each task’s evaluation phase. This adds only a negligible time overhead from a single short inference call and no memory overhead, as it reuses the already-loaded backbone. The procedure works as follows.

Model. The same base LLM used for task inference (gemma-3-12b-it, 4-bit quantized) serves as the zero-shot classifier for metric selection. This avoid the overhead of loading an auxiliary model and the system simply executes a single generation call before the model training and evaluation on each task.

Meta-prompt template. A single representative prompt is sampled from the task’s test split, truncated to 700 characters to manage context, and embedded into a structured meta-prompt. The prompt instructs the LLM to categorize the task based on its linguistic and structural features (e.g., presence of code-specific keywords, multiple-choice indicators, or summarization requests). The model is constrained to a closed set of responses: ACCURACY, ROUGE, SARI, or EDIM.

Analyze the following task prompt and select the best evaluation metric.   
RULES:   
1. Code completion tasks (contains ‘code’, ‘python’, ‘complete the code’,   
etc.) → Use EDIM   
2. Summarization tasks (contains ‘summarize’, ‘summary’, ‘write a   
summary’, etc.) → Use ROUGE   
3. Multiple choice questions (contains ‘A)’, ‘B)’, ‘C)’, ‘choose’, etc.)   
→ Use ACCURACY   
4. Text simplification tasks (contains ‘simplify’, ‘simplified version’,   
etc.) → Use SARI   
5. Default for all other tasks → Use ACCURACY   
Task prompt: {prompt}   
Respond with ONLY the metric name. Choose one: ACCURACY, ROUGE, SARI,   
or EDIM   
### Response:

Parsing and fallback. To ensure robustness, the model’s output is parsed for the first occurrence of a valid metric keyword. If the model fails to produce a recognized metric name within its response, the system defaults to ACCURACY as a conservative baseline. Once a metric is selected, it is cached for the duration of the task’s residency in the stream to prevent redundant computation.

## B ALGORITHMS AND EFFICIENCY

## B.1 ALGORITHMS (PSEUDOCODE)

Algorithm 1: CaLLM-Batched   
Input: Task sequence $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { T } ;$ threshold θ (fixed or adaptive, Eq. 4); pool capacity $P$   
Output: Performance table M   
Initialize pool $\mathcal { P } = \mathcal { D }$   
for $t = 1$ to T do   
// Route (training mode: prototype updates enabled)   
e ← EMBED(sample $( \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } ) )$   
$\begin{array} { r } { i ^ { * } , d ^ { * } \gets \arg \operatorname* { m i n } _ { i : \ell _ { i } \in \mathcal { P } } d _ { i } ^ { * } ( \mathrm { E q . ~ } 3 ; \ d ^ { * } = \infty \ \mathrm { i f } \ \mathcal { P } = \varnothing ) } \end{array}$   
if $\dot { d } ^ { * } \leq \theta$ then   
$a _ { t }  \ell _ { i ^ { * } } ; \mathbf { p } _ { i ^ { * } }  { \frac { 1 } { 2 } } ( \mathbf { p } _ { i ^ { * } } + \mathbf { e } ) ; c _ { i ^ { * } }  c _ { i ^ { * } } + 1$ // reuse   
else if $| \mathcal { P } | < P$ then   
a<sub>t</sub> ← NEWADAPTER(); $\mathcal { P }  \mathcal { P } \cup \{ ( a _ { t } , \mathbf { e } , 1 ) \}$ // create   
else   
i<sub>min</sub> ← arg min<sub>i</sub> c<sub>i</sub>; $a _ { t } \gets \ell _ { i _ { \operatorname* { m i n } } } ; \mathbf { p } _ { i _ { \operatorname* { m i n } } } \gets \mathbf { e } ; c _ { i _ { \operatorname* { m i n } } } \gets 1$ // repurpose   
end if   
// Train (overwrites adapter weights in the repurposing case)   
$\mathrm { S F T } ( \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } , \mathcal { D } _ { t } ^ { \mathrm { e v a l } }$ , adapter=a<sub>t</sub>)   
// Evaluate current task   
$\mathrm { L O A D A D A P T E R } ( a _ { t } ) ; \ \mathbf { M } [ t , t ] \gets \mathrm { E v A L U A T E } ( \mathcal { D } _ { t } ^ { \mathrm { t e s t } } )$   
// Backward transfer on all previous tasks (read-only routing)   
i $\ i > 1$ then   
for $j = 1 \mathbf { t o } t - 1$ do   
${ \mathbf { e } } _ { j } \gets$ EMBED(sample $( \mathcal { D } _ { j } ^ { \mathrm { t e s t } } ) )$   
$a _ { j }  \ell _ { \mathrm { a r g m i n } _ { i } d _ { i } ( \mathbf { e } _ { j } ) }$ // read-only: no prototype updates   
LOADADAPTE $: ( a _ { j } )$ ; M[j, t] ← EVALUATE(D<sup>test</sup><sub>j</sub> )   
end for   
$\begin{array} { r } { \mathbf { M } [ \mathrm { B W T } , t ] \longleftarrow \frac { 1 } { t - 1 } \sum _ { j = 1 } ^ { t - 1 } \left( \mathbf { M } [ j , t ] - \mathbf { M } [ j , j ] \right) / s _ { j } } \end{array}$   
else $\mathbf { M } [ \mathbf { B W T } , t ]  0$   
end if   
$\begin{array} { r } { \mathbf { M } [ \mathrm { A v g } , t ] \longleftarrow \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \mathbf { M } [ i , t ] / s _ { i } } \end{array}$   
end for

## Algorithm 2: CaLLM-Online

Input: Chunk stream $( \tau _ { k } , \mathcal { D } _ { k } )$ for $k = 1 , \ldots , N ;$ threshold θ (fixed or adaptive); pool capacity $P$   
Output: Performance table M with per-step Forgetting, BWT, and OMA   
Initialize: pool ${ \mathcal { P } } = { \mathcal { O } } ,$ last $= \{ \} , \ S _ { \mathrm { r u n } } = 0 , \ c = 0$   
for each step $k = 1 , \ldots , N ,$ receiving $( \tau , \mathcal { D } )$ do   
// Route (single pass: prototype updates enabled)   
e ← EMBED(sample(D))   
$i ^ { * } , d ^ { * } \gets$ arg min<sub>i:ℓ ∈P</sub> d<sub>i</sub> (d<sup>∗</sup> = ∞ if $\mathcal { P } = \emptyset )$   
if $\dot { d } ^ { * } \leq \theta$ then   
$a  \overline { { { \ell _ { i } } } } { * } ; \mathbf { p } _ { i ^ { * } }  \textstyle { \frac { 1 } { 2 } } ( \mathbf { p } _ { i ^ { * } } + \mathbf { e } ) ; c _ { i ^ { * } }  c _ { i ^ { * } } + 1$ // reuse   
else if $| \mathcal { P } | < P$ then   
$a \gets \dot { \mathrm { N E W A D A P T E R } } ( ) ; \mathcal { P } \gets \mathcal { P } \cup \{ ( a , \mathbf { e } , 1 ) \}$ // create   
else   
i<sub>min</sub> ← arg min<sub>i</sub> c<sub>i</sub>; $a \gets \ell _ { i _ { \mathrm { m i n } } } ; ~ { \bf p } _ { i _ { \mathrm { m i n } } } \gets { \bf e } ; ~ c _ { i _ { \mathrm { m i n } } } \gets 1$ // repurpose   
end if   
// Pre-update evaluation (test-then-train)   
LOADADAPTER(a); pre ← EVALUATE(D); M[τ (pre), k] ← pre

// BWT (defined only when task has a prior observation)   
if τ ∈ last then   
M[BWT, k] ← (pre − last[τ]) / s<sub>τ</sub>   
end if   
last[τ ] ← pre   
S<sub>run</sub> ← S<sub>run</sub> + pre / s<sub>τ</sub>; c ← c + 1; M[OMA, k] ← S<sub>run</sub> / c   
// Train on same adapter, same chunk   
SFT(D, adapter=a)   
end for

## B.2 COMPUTATIONAL AND MEMORY EFFICIENCY

A practical CL system must not only preserve performance but also remain efficient as the expert pool grows. Table 6 reports training FLOPs, throughput, and peak GPU memory for CaLLM at varying pool sizes alongside baselines, measured on the full TRACE-8 batched pipeline (gemma-3-12b-it, single NVIDIA RTX A6000 48 GB GPU, CUDA 12.4).

Table 6: Compute and Memory Overhead on TRACE-8 (Batched). CaLLM consumes the same training FLOPs as single-adapter Seq-FT regardless of pool size, with constant GPU memory. ER doubles wall-clock time due to replay, and oLoRA requires 2.1× more GPU memory for orthogonal projections. <sup>†</sup>EWC, ER, and oLoRA use custom training loops that do not report total flos; EWC’s FLOPs are comparable to CaLLM (same data/rank), ER’s are higher due to replay augmentation, and oLoRA’s are lower due to smaller rank (r=4).
<table><tr><td>Method</td><td>Pool P</td><td>Train TFLOPs</td><td>Samples/s</td><td>Peak GPU (GB)</td><td>Wall (h)</td><td># Adapters</td></tr><tr><td>CaLLM-B</td><td>2</td><td>7,339</td><td>3.81</td><td>12.15</td><td>60.0</td><td>2</td></tr><tr><td>CaLLM-B</td><td>4</td><td>7,339</td><td>3.81</td><td>12.15</td><td>59.3</td><td>4</td></tr><tr><td>CaLLM-B</td><td>6</td><td>7,339</td><td>3.67</td><td>12.15</td><td>59.7</td><td>6</td></tr><tr><td>CaLLM-B</td><td>8</td><td>7,339</td><td>3.87</td><td>12.15</td><td>59.5</td><td>8</td></tr><tr><td>Seq-FT</td><td>1</td><td>7,339</td><td>3.53</td><td>12.15</td><td>68.2</td><td>1</td></tr><tr><td>EWC†</td><td>1</td><td>一</td><td>一</td><td>12.48</td><td>76.9</td><td>1</td></tr><tr><td>ER†</td><td>1</td><td>一</td><td>1</td><td>12.10</td><td>125.8</td><td>1</td></tr><tr><td>oLoRA†</td><td>8</td><td>一</td><td>一</td><td>25.20</td><td>75.1</td><td>8</td></tr></table>

Training compute. The central result is that CaLLM’s training cost is independent ofpool size: all four configurations $( P = 2 , 4 , 6 , 8 )$ consume exactly 7,339 TFLOPs. This is a direct consequence of the modular architecture—only one adapter is active per forward pass, so the computational graph is identical regardless of how many adapters exist in the pool. Training throughput is also stable across pool sizes (3.67–3.87 samples/s), confirming that adapter switching introduces negligible latency.

CaLLM and Seq-FT share identical training FLOPs (7,339 TFLOPs), as expected—both perform the same number of gradient updates on the same data with the same adapter rank. The difference lies entirely in where gradients are routed, not in how many are computed. CaLLM achieves slightly higher throughput (3.81–3.87 vs. 3.53 samples/s) and lower wall-clock time (59–60 h vs. 68 h), likely because isolated adapters encounter simpler optimization landscapes than a shared adapter accumulating all tasks

Routing overhead. Unlike the baselines, CaLLM performs an additional forward pass through the frozen backbone for prototype matching (§ 3.2). Because this routing pass is encoder-only, i.e., no decoding, over a short input and reuses the already-loaded backbone, its marginal cost is small relative to the per-task training and inference compute. Using the standard transformer estimate of approximately 2N FLOPs per forward token, where $N \approx 1 2 \times 1 0 ^ { 5 }$ for gemma-3-12b-it, and noting that each routing call processes n ≈ 20 prompts of approximately 10 tokens each, one routing call costs 2 · $( 1 2 \times 1 0 ^ { 9 } ) \cdot 2 0 \cdot 1 0 \approx 4 . 8 \stackrel { . } { \times } 1 0 ^ { 1 2 }$ FLOPs, i.e., 4.8 TFLOPs. Over the full TRACE-8 batched run, with one routing call per task across 8 tasks, the router consumes 8 · 4.8 ≈ 38.4 TFLOPs, or roughly $( 3 8 . 4 / 7 3 3 9 ) \times 1 0 0 \approx 0 . 5 \%$ of the 7339 TFLOPs reported above for training. This is consistent with the observation that CaLLM and Seq-FT report identical training TFLOPs in Table 6: the router’s contribution falls below the reporting precision. The same logic applies at inference time, where each routing call is amortized over all subsequent evaluation queries within a task, giving per-query overhead ≪ 0.1%, and during the online continual-learning phase, where routing is invoked once per chunk rather than per query. The routing module is, moreover, a tunable component: a smaller encoder or cached prompt embeddings could reduce this already-small overhead further, which we leave to future work.

Baseline comparison. Among the baselines that explicitly address forgetting, each pays a distinct overhead. EWC trains on the same data with the same adapter rank, so its per-step FLOPs are expected to be comparable to CaLLM; the additional wall-clock cost (76.9 h) comes from Fisher information computation after each task. ER augments each training batch with replay buffer samples, which increases the effective data budget and thus is expected to incur higher total training FLOPs—this is reflected in its 2.1× wall-clock overhead (125.8 h). oLoRA uses a smaller rank (r=4 vs. r=16), which should reduce per-step FLOPs, but must maintain orthogonal projection matrices across all subspaces simultaneously, making it the most memory-intensive baseline at 25.2 GB (2.1× CaLLM).

These results confirm that CaLLM achieves its anti-forgetting benefits at zero additional training compute relative to na¨ıve sequential fine-tuning, while the baselines that attempt to mitigate forgetting through regularization, replay, or orthogonality constraints each pay a substantial cost in time, memory, or both.

Memory management. CaLLM maintains a constant 12.15 GB GPU memory footprint regardless of pool size, because exactly one adapter is loaded on GPU at any time. Inactive adapters are offloaded to CPU memory and swapped in on demand, so the GPU overhead does not grow with the number of adapters in the pool. The maximum pool capacity $P _ { \mathrm { m a x } }$ can be set explicitly or computed automatically based on available system memory according to Eq. 6:

$$
P = \left\lfloor { \frac { M _ { \mathrm { a v a i l } } } { S _ { \mathrm { a d a p t e r } } } } \right\rfloor - 1 ,\tag{6}
$$

where $M _ { \mathrm { a v a i l } }$ is the available system (CPU) memory and $S _ { \mathrm { a d a p t e r } }$ is the on-disk size of a single adapter (both in GB); the −1 reserves one slot for the adapter currently resident on GPU. In the online regime, peak GPU memory remains flat at 12.01 GB—virtually identical to the batched setting—confirming that this constant-memory property holds across both training protocols.

Online regime. The online streaming configuration uses lighter adapters (r=4, α=8) and single-epoch chunk updates, so its total FLOPs are not directly comparable to the batched runs above. A notable characteristic of the online setting is that evaluation dominates the runtime: out of 19.3 h total wall-clock time over 500 chunks, only 3.6 h (18.4%) is spent on training while 13.9 h (72.3%) is consumed by the per-chunk test-then-train evaluation loop. This suggests that evaluation scheduling—rather than training efficiency—is the primary lever for reducing online deployment cost, and motivates future work on sparse or amortized evaluation strategies.

## C EXTENDED QUANTITATIVE RESULTS

## C.1 ONLINE POOL CONSTRAINTS: OMA AND BWT CURVES

![](images/0a9f6827da5ab50133721192e0636ef4f269e91103c976ea661c717e6b80f120.jpg)  
(a) TRACE OMA $( P \in \{ 4 , 6 , 8 \} )$

![](images/7b28391be9eef82f1d67faa885037b3cdf38b8b0756f42f648e8b8679cf175c3.jpg)  
(b) TRACE BWT $( P \in \{ 4 , 6 , 8 \} )$

![](images/23b2a85207ab8c0a1d653263e63b63b9068cf1039fe238c9cdee8fba1983a0c3.jpg)  
(c) CITB OMA $( P \in \{ 3 , 1 0 , 1 9 \} )$ )

![](images/2dd5d6e3bb0211b161f17582fa913b45faf32d1ac53782719687203b83ac4ed5.jpg)  
(d) CITB BWT $( P \in \{ 3 , 1 0 , 1 9 \} )$  
Figure 6: Online Pool Constraints (500-chunk stream). OMA and BWT for CaLLM-O and the SeqFT baseline. Top row (TRACE-8): (a) OMA and (b) BWT for pool sizes $P \in \{ 4 , 6 , 8 \}$ . Bottom row (CITB-19): (c) OMA and (d) BWT for pool sizes $P \in \{ 3 , 1 0 , 1 9 \}$ . Knowledge accumulation (OMA) is successful across all configurations; performance is non-monotonic with respect to pool size on TRACE, while CITB shows comparable cumulative scores but diverging forgetting profiles.

## C.2 EXTENDED BATCHED PERFORMANCE

## C.2.1 PER-TASK BATCHED PERFORMANCE

Table 7 and Table 8 show the per-task performance on TRACE-8 and CITB-19 for the CaLLM-B baseline as each task is sequentially presented to the model, capturing the immediate performance right after all training data for that specific task has been processed. These tables are complementary to the results illustrated in Figure 3, demonstrating that performance on individual tasks is steadily maintained as the stream length increases. This stability is further supported by Figure 7, which shows that CaLLM consistently achieves a BWT score of nearly zero, confirming minimal catastrophic forgetting across the streaming timeline.

## C.2.2 CITB EXTENDED BATCHED PERFORMANCE

Table 9 disaggregates the final CITB-19 results by individual task, analogous to Table 1 for TRACE-8. All 19 tasks use ROUGE-L (0–1) as the evaluation metric. Scores are averaged across 3 seeds for all methods except EWC (2 seeds available). CaLLM-B achieves the highest score among all continual learning methods on 13 of 19 tasks. On the remaining 6, oLoRA leads on individual tasks but trails CaLLM-B in mean performance (0.24 vs. 0.31), indicating that its per-task wins come at the cost of broader forgetting elsewhere. The high Inference-Only scores on tasks such

Table 7: Per-Task Performance Table on TRACE-8 for CaLLM-B. Results are reported for a single random seed. Metrics used are accuracy (C-STANCE, FOMC, ScienceQA, NumGLUE-cm/ds), ROUGE-L (MeetingBank), EDIM (Py150), and SARI (20Minuten).
<table><tr><td>Task</td><td>C-STANCE</td><td>FOMC</td><td>MeetingBank</td><td>Py150</td><td>ScienceQA</td><td>NumGLUE-cm</td><td>NumGLUE-ds</td><td>20Minuten</td></tr><tr><td>C-STANCE</td><td>0.42</td><td>0.42</td><td>0.42</td><td>0.42</td><td>0.41</td><td>0.42</td><td>0.42</td><td>0.42</td></tr><tr><td>FOMC</td><td></td><td>0.56</td><td>0.55</td><td>0.56</td><td>0.56</td><td>0.56</td><td>0.56</td><td>0.56</td></tr><tr><td>MeetingBank</td><td></td><td></td><td>0.22</td><td>0.23</td><td>0.23</td><td>0.23</td><td>0.23</td><td>0.23</td></tr><tr><td>Py150</td><td></td><td></td><td></td><td>16.94</td><td>16.95</td><td>16.85</td><td>17.15</td><td>17.12</td></tr><tr><td>ScienceQA</td><td></td><td></td><td></td><td></td><td>0.56</td><td>0.57</td><td>0.56</td><td>0.56</td></tr><tr><td>NumGLUE-cm</td><td></td><td></td><td></td><td></td><td></td><td>0.19</td><td>0.19</td><td>0.19</td></tr><tr><td>NumGLUE-ds</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.30</td><td>0.30</td></tr><tr><td>20Minuten</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>39.59</td></tr></table>

Table 8: Per-Task Performance Table on CITB-19 for CaLLM-B. Results are reported for a single random seed. The metric used is ROUGE-L for all tasks.
<table><tr><td>Task</td><td>tsk590</td><td>tas713</td><td>tak565</td><td>tas730</td><td>task573</td><td>tk294</td><td>t1sk50</td><td>tas 603</td><td>task576</td><td>task639</td><td>tas714</td><td>task501</td><td>tas1729</td><td>tsk574</td><td>task848</td><td>task61</td><td>task6</td><td>t2sk384</td><td>2sk 60</td></tr><tr><td>task1590</td><td>0.41</td><td>0.50</td><td>0.55</td><td>0.44</td><td>0.53</td><td>0.45</td><td>0.52</td><td>0.44</td><td>0.46</td><td>0.46</td><td>0.40</td><td>0.43</td><td>0.47</td><td>0.38</td><td>0.56</td><td>0.55</td><td>0.44</td><td>0.46</td><td>0.53</td></tr><tr><td>task1713</td><td></td><td>0.60</td><td>0.52</td><td>0.56</td><td>0.53</td><td>0.59</td><td>0.59</td><td>0.51</td><td>0.56</td><td>0.53</td><td>0.55</td><td>0.64</td><td>0.52</td><td>0.53</td><td>0.52</td><td>0.49</td><td>0.56</td><td>0.49</td><td>0.54</td></tr><tr><td>task565</td><td></td><td></td><td>0.37</td><td>0.41</td><td>0.38</td><td>0.39</td><td>0.33</td><td>0.31</td><td>0.37</td><td>0.33</td><td>0.29</td><td>0.29</td><td>0.21</td><td>0.36</td><td>0.37</td><td>0.29</td><td>0.45</td><td>0.29</td><td>0.33</td></tr><tr><td>task1730</td><td></td><td></td><td></td><td>0.24</td><td>0.15</td><td>0.21</td><td>0.18</td><td>0.14</td><td>0.18</td><td>0.18</td><td>0.20</td><td>0.21</td><td>0.18</td><td>0.21</td><td>0.19</td><td>0.13</td><td>0.20</td><td>0.14</td><td>0.18</td></tr><tr><td>task573</td><td></td><td></td><td></td><td></td><td>0.45</td><td>0.43</td><td>0.47</td><td>0.49</td><td>0.48</td><td>0.44</td><td>0.55</td><td>0.56</td><td>0.45</td><td>0.44</td><td>0.34</td><td>0.50</td><td>0.56</td><td>0.60</td><td>0.51</td></tr><tr><td>task294</td><td></td><td></td><td></td><td></td><td></td><td>0.11</td><td>0.11</td><td>0.12</td><td>0.09</td><td>0.10</td><td>0.12</td><td>0.13</td><td>0.15</td><td>0.13</td><td>0.13</td><td>0.11</td><td>0.16</td><td>0.09</td><td>0.14</td></tr><tr><td>task1500</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.09</td><td>0.07</td><td>0.13</td><td>0.10</td><td>0.13</td><td>0.06</td><td>0.14</td><td>0.09</td><td>0.16</td><td>0.05</td><td>0.13</td><td>0.11</td><td>0.14</td></tr><tr><td>task1603</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.70</td><td>0.80</td><td>0.76</td><td>0.62</td><td>0.70</td><td>0.74</td><td>0.76</td><td>0.61</td><td>0.66</td><td>0.65</td><td>0.59</td><td>0.67</td></tr><tr><td>task576</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.08</td><td>0.10</td><td>0.09</td><td>0.07</td><td>0.07</td><td>0.10</td><td>0.07</td><td>0.07</td><td>0.07</td><td>0.07</td><td>0.08</td></tr><tr><td>task639</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.25</td><td>0.29</td><td>0.20</td><td>0.25</td><td>0.22</td><td>0.23</td><td>0.24</td><td>0.18</td><td>0.23</td><td>0.23</td></tr><tr><td>task1714</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.09</td><td>0.10</td><td>0.06</td><td>0.04</td><td>0.07</td><td>0.05</td><td>0.09</td><td>0.08</td><td>0.05</td></tr><tr><td>task1501</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>一</td><td>一</td><td></td><td></td><td>0.72</td><td>0.80</td><td>0.84</td><td>0.72</td><td>0.72</td><td>0.84</td><td>0.92</td><td>0.76</td></tr><tr><td>task1729</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.08</td><td>0.08</td><td>0.20</td><td>0.15</td><td>0.08</td><td>0.12</td><td>0.04</td></tr><tr><td>task574</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.09</td><td>0.04</td><td>0.08</td><td>0.08</td><td>0.09</td><td>0.09</td></tr><tr><td>task848</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.27</td><td>0.27</td><td>0.23</td><td>0.28</td><td>0.29</td></tr><tr><td>task611</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.31</td><td>0.28</td><td>0.23</td><td>0.21</td></tr><tr><td>task766</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.18</td><td>0.21</td><td>0.25</td></tr><tr><td>task1384</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.04</td><td>0.04</td></tr><tr><td>task1600</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.30</td></tr></table>

![](images/3505bf334f58db6b2366eeaadf6377bbd32882950cdf0324572737d29a6883b1.jpg)  
(a) TRACE-8 dataset

![](images/be3471e7901a4e004f99c6c2a09aa5e1dbf39de885b3d47b4fe0dfe2744323de.jpg)  
(b) CITB-19 dataset  
Figure 7: Task-Incremental BWT Comparison Across Baselines (mean ± std). CaLLM-B maintains a BWT score close to zero throughout the stream on both benchmarks.

as Diplomacy (T1590), P.Chat Gen. (T1729), and Mutual Dial. (T611) reflect strong backbone priors that sequential fine-tuning erodes for all methods  
Table 9: Per-Task Final Performance on CITB-19 (Batched). Scores measured after all 19 tasks have been trained sequentially; all tasks use ROUGE-L (0–1) averaged over 3 seeds.
<table><tr><td>Task</td><td>Inf-Only</td><td>Seq-FT</td><td>EWC</td><td>ER</td><td>oLoRA</td><td>CaLLM-B</td></tr><tr><td>Diplomacy (T1590)</td><td>0.84</td><td>0.14</td><td>0.05</td><td>0.20</td><td>0.10</td><td>0.31</td></tr><tr><td>ConvAI3-1713</td><td>0.36</td><td>0.05</td><td>0.07</td><td>0.14</td><td>0.18</td><td>0.28</td></tr><tr><td>CIRCA (T565)</td><td>0.08</td><td>0.10</td><td>0.04</td><td>0.09</td><td>0.28</td><td>0.40</td></tr><tr><td>P.Chat Choose (T1730)</td><td>0.02</td><td>0.24</td><td>0.02</td><td>0.09</td><td>0.39</td><td>0.45</td></tr><tr><td>AirDial. Class. (T573)</td><td>0.92</td><td>0.05</td><td>0.07</td><td>0.08</td><td>0.27</td><td>0.36</td></tr><tr><td>Story Motiv. (T294)</td><td>0.34</td><td>0.09</td><td>0.07</td><td>0.09</td><td>0.49</td><td>0.25</td></tr><tr><td>DSTC3 Class. (T1500)</td><td>0.12</td><td>0.05</td><td>0.00</td><td>0.03</td><td>0.12</td><td>0.44</td></tr><tr><td>SMCalFlow-1603</td><td>0.68</td><td>0.12</td><td>0.06</td><td>0.12</td><td>0.11</td><td>0.47</td></tr><tr><td>Curiosity Dial. (T576)</td><td>0.25</td><td>0.28</td><td>0.03</td><td>0.04</td><td>0.29</td><td>0.28</td></tr><tr><td>MultiWOZ (T639)</td><td>0.14</td><td>0.07</td><td>0.10</td><td>0.08</td><td>0.32</td><td>0.15</td></tr><tr><td>ConvAI3-1714</td><td>0.20</td><td>0.11</td><td>0.02</td><td>0.23</td><td>0.12</td><td>0.25</td></tr><tr><td>DSTC3 Ans. (T1501)</td><td>0.06</td><td>0.05</td><td>0.03</td><td>0.07</td><td>0.31</td><td>0.41</td></tr><tr><td>P.Chat Gen. (T1729)</td><td>0.82</td><td>0.16</td><td>0.08</td><td>0.07</td><td>0.18</td><td>0.09</td></tr><tr><td>AirDial. Gen. (T574)</td><td>0.22</td><td>0.11</td><td>0.05</td><td>0.09</td><td>0.16</td><td>0.20</td></tr><tr><td>PubMedQA (T848)</td><td>0.08</td><td>0.17</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.21</td></tr><tr><td>Mutual Dial. (T611)</td><td>0.84</td><td>0.05</td><td>0.12</td><td>0.19</td><td>0.36</td><td>0.26</td></tr><tr><td>Craigslist (T766)</td><td>0.21</td><td>0.13</td><td>0.05</td><td>0.12</td><td>0.11</td><td>0.36</td></tr><tr><td>DealOrNo (T1384)</td><td>0.07</td><td>0.12</td><td>0.10</td><td>0.10</td><td>0.26</td><td>0.38</td></tr><tr><td>SMCalFlow-1600</td><td>0.26</td><td>0.36</td><td>0.19</td><td>0.28</td><td>0.50</td><td>0.35</td></tr></table>

## C.3 INFERENCE-ONLY BASELINE: PROMPT-TEMPLATE SENSITIVITY

To test the robustness of the performance to the prompt template, we repeated the Inference-Only baseline on both TRACE-8 and CITB-19 using Gemma 3’s native chat format (<start of turn>user ...<end of turn><start of turn>model), keeping every other hyperpa rameter identical to the Alpaca-format run reported in the main paper. We present the per-task comparison in Figure 8. On TRACE-8, the Alpaca template is strictly stronger or equal on every task (mean normalized score 0.31 vs. 0.21), with the largest drops on structured-output tasks (Py150, ScienceQA, NumGLUE-cm/ds). On CITB-19 the two templates are tied in aggregate (mean ROUGE-L 0.34 vs. 0.35) but redistribute mass between generation-style tasks (which lose 0.4–0.8), and classification-/choice-style tasks, which gain a comparable amount. We therefore retain the Alpaca-format Inf-Only as the headline baseline as it is the more conservative comparison on TRACE-8 and equivalent on CITB-19.

![](images/0ce5fd8c5bcbab6841c6ee19b1774a686efdf5bc56b0895f1751bf9b93588fcd.jpg)  
(a) TRACE-8

![](images/16092aca39ae242432c5d31eb80b28700fe0bee6de28ce87c0d4246d0a44f96c.jpg)  
(b) CITB-19  
Figure 8: Prompt-template sensitivity of the Inference-Only baseline. Per-task final performance under the Alpaca template (blue, used in the main paper) and Gemma 3’s native chat template (red). (a) TRACE-8 (Batched): normalized scores (EDIM, SARI divided by 100); Alpaca dominates on every task (mean 0.31 vs. 0.21). (b) CITB-19 (Batched): ROUGE-L; the two templates are tied in aggregate (mean 0.34 vs. 0.35) but redistribute mass across generation- and classification-style tasks. Tasks are shown in original training/streaming order.

## D EXTENDED ROUTER ANALYSIS

This section provides a deep dive into the router’s internal logic, visualizing how it represents tasks and how capacity constraints (P) dictate expert assignment.

## D.1 TASK SEPARABILITY IN THE EMBEDDING SPACE

Figure 9 illustrates the inherent separability of the task distributions within the router’s embedding space. Using t-SNE projections of the frozen backbone’s mean-pooled features, we observe that diverse tasks (both for TRACE-8 and CITB-19) form well-defined, distant clusters.

![](images/568d28ec67f1ba05e39241e5b67a2900f74a97f980994ac4f8acbdcffc265354.jpg)

![](images/9a209dd6fb9a341fff86b8be31d20d6e92a8245ea12fd3aeb38bc33ce44d3d2e.jpg)  
Figure 9: Router Embedding Space (2D Projection). t-SNE projections of the router’s embeddings, coloured by task identity. (a) TRACE-8: 8 tasks spanning diverse domains. (b) CITB-19: 19 dialogue sub-tasks.

## D.2 PROTOTYPE DYNAMICS AND CAPACITY PROGRESSION

Figure 10 demonstrate how the router dynamically partitions the embedding space under varying capacity constrains, studied in §4.5.1, for TRACE-8 . Each panel overlays P prototypes (yellow stars) on the task-cluster structure; grey lines connect each data point to its nearest prototype. The prototype positions determine which tasks share an expert during training.

At P=8 (Figure 10a), every prototype settles at the centroid of one task cluster, giving zero-interference isolation. At P=6 (Figure 10b), the two NumGLUE math tasks and the C-STANCE dataset are pulled toward a shared prototype, a constructive merge that incurs negligible BWT. At P=4 (Figure 10c), one prototype absorbs five semantically diverse tasks on the right side of the space, producing the conflicting gradients that drive destructive overload. Even here, Py150 (code) retains its own isolated prototype (in red), confirming the router’s sensitivity to the most extreme distributional gap. At P=2 (Figure 10d), only 20Minuten keeps a dedicated prototype; all other tasks collapse onto a single expert.

![](images/23f97418a8b74d5a6b72b473b3f823436e95b765725473e5fe7f224c2d2b23be.jpg)  
(a) P=8: perfect isolation, one prototype per task cluster.

![](images/0095815f9f51bd36d3f1a8e896239ac7a2d3133c9b8cb8e548b9aafe99c6b0ff.jpg)  
(b) P=6: NumGLUE-cm/ds share a prototype—constructive merge, negligible BWT.

![](images/9fd5ad93e5c3b36201457c94d43c60bba503dc5c47153deda562a68a741eb044.jpg)  
(c) P=4: one prototype absorbs 5 diverse tasks—destructive overload, BWT = −0.19.

![](images/21f265151e106285f48d3e5bd8f3d9a3b0dc5e85d55e8d72f44329f7ab225cb8.jpg)  
(d) P=2: system collapse, 7 of 8 tasks share one prototype, BWT = −0.25.

Figure 10: Router Embedding Space with Prototypes (TRACE-8, t-SNE). Yellow stars are prototype positions; grey lines show nearest-prototype assignment. The transition from P=8 to P=2 maps directly onto the isolation → constructive clustering → destructive overload → collapse progression described in §4.5.1.

## D.3 ONLINE STREAMING HEATMAP

While the previous plots show the router assignment in the Batched scenario, Figure 11 shows the complete 500-chunk task×expert assignment heatmap for the TRACE online stream. Each row corresponds to a task, each column to a chunk, and colour encodes the expert ID assigned by the router. The “null” entries (dark blue) indicate chunks in which the corresponding task did not appear in the interleaved stream. Despite the sparsity, the per-task colour patterns are stable throughout the full stream, confirming the persistent task—expert binding reported in §4.4. The zoomed view (chunks 200–300) used in the main text (Figure 4(b)) was selected as a representative window that minimises null-cell density and maximises readability.

![](images/57bd3a478914b90355cc1e22b3e2149eab8dfbb2445cf1845b78b9ee11ca77ea.jpg)  
Figure 11: Full Expert-Assignment Heatmap (TRACE, 500 chunks, P=8). Each row is a task; each column a chunk; colour encodes the assigned expert ID. Dark blue (“null”) cells indicate chunks where that task was absent from the stream. The per-task colour patterns remain stable across the full 500-chunk stream, confirming persistent task–expert binding.

## D.3.1 AGGREGATE ONLINE ROUTING MATRICES

The temporal stability observed in the full stream is summarized numerically in Figure 12. These matrices for TRACE $( P \in \{ \hat { 4 , 6 , 8 } \} )$ show the fraction of chunks routed to each expert over a 500-chunk stream. As analyzed in §4.5.2, these matrices reveal that at $P { = } 4 ,$ the router settles into highly stable, semantically aligned groups that outperform the $P { = } 6$ configuration due to reduced task displacement.

![](images/f1d35eca6c6d266731b543face738d0b7abdabec2b443349e1eb02e765e742f9.jpg)  
Figure 12: Router Assignment Matrices on TRACE (Online). Pool sizes $P \in \{ 4 , 6 , 8 \}$ . Each cell shows the fraction of chunks from that task routed to each expert.