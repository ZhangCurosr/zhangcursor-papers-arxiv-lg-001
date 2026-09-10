# Positional task conditioning for scalable defect detection across product families in large product catalogs

Soham Satyadharma<sup>1</sup>, Gabriel Roccabruna<sup>1</sup>, Suleiman A. Khan<sup>1</sup>

<sup>1</sup> Amazon Catalog AI

ssatyadh@amazon.com, groccabr@amazon.es, suleimkh@amazon.com

## Abstract

Product families in large product catalogs suf fer from inconsistencies such as duplicates and unit mismatches that degrade customer experience. Detecting these requires reasoning over multiple error types across lengthy product listings, where LLM classification quality degrades due to long-context limitations. We address this by decomposing detection into focused sub-tasks that reduce context and isolate error types, improving F1 from 52% to 87%. For scalable deployment, we introduce Positional Task Conditioning (PTC), which distills this capability into a single smaller model by reinforcing task identity at structural prompt boundaries. PTC outperforms rationale-based distillation across five models and two architecture families, achieving within 1.79% F1 of the frontier at upto 98% lower cost. Our system is deployed across multiple countries processing 10+ million product families.

## 1 Introduction

Large product catalogs organize products into logical groupings that differ by attributes like size, color, or style. They suffer inconsistencies from heterogeneous seller inputs and automated ingestion manifesting as duplicate variations, unit conflicts, attribute overstuffing, and theme defects (Schmidts et al., 2020). These inconsistencies directly degrade customer experience: duplicates (e.g., XL vs. Extra Large) increase cognitive load, unit conflicts (e.g., Medium vs. 30 inches) force manual conversions, and overstuffed attributes conflate orthogonal concepts. Poor product information has endemically increased return rates and support costs while reducing conversions (Xu et al., 2015), making automated detection critical for large scale operational efficiency.

Prior work addresses single-product attribute correctness (Wang et al., 2020), entity matching (Li et al., 2020), or attribute generation (Nikolakopoulos et al., 2023), leaving intra-family inconsistency detection underexplored. Classical rule-based approaches lack generalization (Ilyas and Chu, 2019), and manual auditing is intractable at scale, necessitating automated solutions.

![](images/54102abfbdb634030a7d41a551fbb313a291c65ed22188ea1812e458ae181793.jpg)  
Figure 1: The figure depicts our divide and conquer approach. We take a product family as input and decompose each of its variating attributes (§3.1) into multiple classification tasks. Each task has its own prompts; the results are aggregated for the family.

A natural approach is providing complete family data and guidelines to a Large Language Model (LLM) to identify all errors simultaneously. However, families with numerous products create lengthy contexts where LLMs struggle to utilize the correct information (Liu et al., 2024). Multilabel classification requiring simultaneous reasoning over orthogonal error types challenges stateof-the-art models (Kou et al., 2025; Ortego et al., 2025; Tabatabaei et al., 2025). Inspired by work showing task decomposition improves LLM reasoning (Khot et al., 2022; Zhou et al., 2022), we adopt a divide-and-conquer strategy: we structure detection as nested attribute-level and task level classification with dedicated prompts, enabling focused reasoning and reduced context, to generate final predictions of egregious, non-egregious, or no error.

While our decomposition achieves productiongrade accuracy, frontier LLMs at catalog scale incur prohibitive costs. Knowledge distillation enables transferring capabilities to efficient student models (Hinton et al., 2015; Gou et al., 2021), but distilling separate models for each task introduces deployment complexity. Rationale-based distillation (RBD) (Hsieh et al., 2023) has demonstrated multi-task capabilities by training on teacher reasoning chains, but a gap between student and teacher performance persists. We propose Positional Task Conditioning (PTC), a task conditioning strategy for distillation, that bridges this gap by conditioning on task identifiers at structural prompt boundaries, leveraging context enrichment to improve generalization (Caruana, 1997).

Our contributions are: (1) a categorization of variation family inconsistencies spanning four tasks with different severity levels; (2) a divide-andconquer framework decomposing multi-label detection into three-class sub-tasks, improving precision by 54% and recall by 13% over monolithic prompting; (3) Positional Task Conditioning (PTC), a multi-task distillation methodology enabling a single model to handle all sub-tasks, outperforming rationale-based distillation while approaching frontier performance at upto 98% reduced cost; and (4) empirical validation on both human-annotated and synthetic data demonstrating PTC generalizes across models, yielding consistent improvements across five LLMs from two architecture families.

## 2 Related work

Multi-label classification and LLMs. The paradigm for multi-label text classification has shifted from discriminative methods (Devlin et al., 2019; Liu et al., 2019b) to generative LLMs, which often violate output space constraints (Niraula et al., 2024) and fail to capture conditional dependencies in flat generation settings (Ma et al., 2025; Zhou et al., 2024). We address this by decomposing the monolithic output space into four modular threeclass classification sub-tasks.

Multi-task learning. Multi-task learning (MTL) improves generalization by leveraging shared structure across related tasks (Ruder, 2017), traditionally via a shared encoder with task-specific heads (Liu et al., 2019a). PTC departs from this paradigm: instead of task-specific heads, it conditions a single LLM across all sub-tasks by inserting task identifiers at structural prompt boundaries, enabling one student model to discriminate between structurally similar prompts without any architectural changes.

Distillation in LLMs. While frontier LLMs exhibit emergent reasoning via CoT prompting (Wei et al., 2022; Kojima et al., 2022), their deployment is constrained by prohibitive latency and cost. Knowledge Distillation (Hinton et al., 2015) addresses this by transferring capabilities to a student model, with recent approaches training students on intermediate reasoning traces (Wang et al., 2023; Magister et al., 2023; Hsieh et al., 2023; Mukherjee et al., 2023; Trabelsi et al., 2025).

We extend this paradigm by introducing positional task conditioning that inserts task identifiers at multiple structural boundaries within the prompt. Unlike T5-style single-prefix conditioning (Raffel et al., 2020), which prepends a task token only at sequence start, PTC reinforces task identity at semantically meaningful positions. Related work on prompt optimization across foundation models (Opsahl-Ong et al., 2024; Gao et al., 2025; Venkataraman et al., 2026) similarly emphasizes that prompt structure materially affects model behavior, though we focus on positional task conditioning for distillation rather than prompt translation. Our ablations over seven configurations show that multi-position placement significantly outperforms the single-prefix baseline.

LLMs in product catalogs. LLMs have been deployed across large product catalogs for recommendation (Maragheh et al., 2023), search (Rokon et al., 2026), product discovery (Wang et al., 2024), knowledge graph completion (Chen et al., 2023), product matching (Herrero-Vidal et al., 2024), and categorization (Cheng et al., 2024; Kathiriya et al., 2023). Recent efforts extend to attribute extraction (Baumann et al., 2024; Zhang et al., 2025; Mansoori et al., 2026) and single-product quality assessment (Satyadharma et al., 2025), but to our knowledge no prior work addresses product family quality detection at large scale. A related challenge is obtaining high-quality annotations at volume; we adopt the approach of generating synthetic labels by injecting errors into correct inputs (Negri et al., 2025), enabling scalable evaluation data generation without manual labeling.

## 3 Method

## 3.1 Problem formulation

We formalize the task of detecting inconsistencies in product families as a multi-label classification problem. Let F denote the space of all families in a product catalog. A family $F \in { \mathcal { F } }$ consists of a set of n products $F = \{ P _ { 1 } , P _ { 2 } , . . . , P _ { n } \}$ , where each product $P _ { i }$ is represented as a set of attribute-value pairs: $P _ { i } ~ = ~ \{ ( a _ { 1 } , v _ { 1 } ) , ( a _ { 2 } , v _ { 2 } ) , \ldots , ( a _ { m } , v _ { m } ) \}$ Here, $a _ { j } \in { \mathcal { A } }$ denotes an attribute from the catalog’s attribute schema (e.g., size, color, style), and $v _ { j }$ represents the corresponding string value for attribute $a _ { j }$ . We define the variating attributes of a family as the set of attributes along which family members differ: $V ( F ) ~ = ~ \{ a _ { j } ~ \in ~ { \mathcal { A } } ~ :$ $| \{ v _ { j } ^ { ( i ) } : P _ { i } \in F \} | > 1 \}$ . For instance, a family of t-shirts varying in size and color would have $V ( F ) = \{ { \mathsf { s i z e } } , { \mathsf { c o l o r } } \}$

To judge the quality of a product family, we define a set of error types $\mathcal { E }$ , where each error type $e _ { i }$ is associated with natural language instructions $I _ { i }$ specifying detection criteria, including definitions, edge cases, and few-shot examples. The complete input for a family F is $X = ( F , V ( F ) , \mathcal { Z } )$ where $\mathcal { T } = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { | \varepsilon | } \}$ . The detection task requires learning a function:

$$
f : \mathcal { X } \to 2 ^ { \mathcal { E } \times \mathcal { A } }\tag{1}
$$

that maps input to a subset of (error type, attribute) pairs. The input comprising instructions for all error types and multiple product records, leads to long context, challenging LLM accuracy.

## 3.2 Divide and conquer (D&C) approach

Based on semantic equivalence of error types in $\mathcal { E } ,$ we decompose f (Eqn. 1) into a set of $| \tau |$ focused sub tasks, each with its own labels. Let the combined label set be Y. Each task groups semantically related error types through its labels: we define a labeling map $\psi : \mathcal { T } \times \mathcal { V } \to \mathcal { E } \cup \{ \emptyset \}$ , where $\psi ( t _ { k } , y )$ is the error type when task $t _ { k }$ emits label $y ,$ , and ∅ when $y$ denotes no error. This enables us to reformulate detection as $| \tau |$ independent classification problems rather than a single $\vert \mathcal { T } \vert \times \vert \mathcal { V } \vert$ multi-label problem. For each task $t _ { k } \in \mathcal { T }$ and variating attribute $a \in V ( F )$ , we define a sub-task classifier $g _ { k } : ( F , a , I _ { k } ^ { \prime } )  \mathcal { V }$ , where $I _ { k } ^ { \prime }$ is the instruction set for $t _ { k }$ . Each sub-task operates on reduced context: only the attribute values $\{ v _ { a } ^ { ( i ) } : P _ { i } \in F \}$ and instructions for a single task.

Each sub-task $g _ { k }$ is instantiated via a tailored prompt $x ^ { ( k ) }$ comprising four components: (1) Introduction: Role definition establishing the model as a catalog quality expert. (2) Instructions $I _ { k } ^ { \prime }$ : Detailed criteria for task $t _ { k }$ and attribute $^ { a , }$ including definitions, boundary cases, and few-shot examples. (3) Family input: The set of values $\{ v _ { a } ^ { ( i ) } : P _ { i } \in F \}$ for attribute a. (4) Output schema: Structured output response

We assemble the family level prediction as $\begin{array} { r l r } { f ( X ) } & { { } = } & { \bigcup _ { k = 1 } ^ { | { \mathcal T } | } \bigcup _ { a \in V ( F ) } \phi ( g _ { k } ( F , a , I _ { k } ^ { \prime } ) , t _ { k } , a ) } \end{array}$ where $\phi : \mathcal { V } \times \mathcal { T } \times \mathcal { A }  2 ^ { \mathcal { E } \times \mathcal { A } }$ maps a sub-task output to the corresponding (error type, attribute) pair via ψ where $\phi ( y , t _ { k } , a ) = \{ ( \psi ( t _ { k } , y ) , a ) \}$ for defect labels and ∅ otherwise. This decomposition requires at most $| \mathcal { T } | \times | V ( F ) |$ LLM invocations per family, but with reduced context and a simpler decision boundary per call.

## 3.3 Positional task conditioning

While D&C achieves strong performance with frontier LLMs, scalable deployment incurs prohibitive costs. We address this through distillation into a single small model. We first formalize a Rationale-Based Distillation baseline (RBD), then present our Positional Task Conditioning (PTC) approach, which extends it with explicit task conditioning at structural prompt boundaries.

Let $M _ { \theta }$ denote a student LLM parameterized by θ. For each task $t _ { k } ~ \in ~ \tau$ , we construct training dataset $\boldsymbol { \mathcal { D } } _ { k } \ = \ \{ ( \boldsymbol { x } _ { i } ^ { ( k ) } , \boldsymbol { y } _ { i } ^ { ( k ) } ) \} _ { i = 1 } ^ { N _ { k } }$ from teacher model outputs on sub-task $g _ { k }$ . Each input $x _ { i } ^ { ( k ) } ~ = ~ ( F _ { i } , a _ { i } , I _ { k } ^ { \prime } )$ comprises a family, attribute, and task specific instructions, with label $y _ { i } ^ { ( k ) } \in \mathcal { y }$ . Each input has an inherent structure $x _ { i } ^ { ( k ) } = [ s _ { \mathrm { i n t r o } } , s _ { \mathrm { i n s t } } , s _ { \mathrm { i n p u t } } , s _ { \mathrm { o u t p u t } } ]$ corresponding to the prompt components defined in Sec. 3.2.

Following research on distilling reasoning traces from larger models (Ho et al., 2023; Fu et al., 2023), we supervise the student with teachergenerated reasoning chains. For each sub-task $g _ { k }$ , the teacher model $M ^ { \prime }$ produces a reasoning chain and label $( r _ { i } ^ { ( k ) } , y _ { i } ^ { ( k ) } )$ and we train on these sampled outputs. RBD minimizes the negative log-likelihood over all sub-tasks, $\theta _ { \mathrm { r b d } }$ arg min<sub>θ k=1 i=1</sub> P<sup>|T</sup> <sup>|</sup> P<sup>Nk</sup> − log $P _ { \theta } ( r _ { i } ^ { ( k ) } , y _ { i } ^ { ( k ) } \mid x _ { i } ^ { ( k ) } )$ However, RBD lacks explicit task awareness: the model must implicitly infer the task from the instructions, leading to confusion when prompts share a similar structure across tasks.

PTC (Fig 2) introduces explicit task conditioning by associating each task $t _ { k }$ with dedicated tokens $\tau _ { k }$ and inserting $\tau _ { k }$ at structurally meaningful boundary positions: before the introduction, before the input data, and after the output schema. The positionally-conditioned input is:

$$
\begin{array} { r } { \tilde { x } _ { i } ^ { ( k ) } = \tau _ { k } \oplus s _ { \mathrm { i n t r o } } \oplus s _ { \mathrm { i n s t } } \oplus \tau _ { k } \oplus s _ { \mathrm { i n p u t } } \oplus s _ { \mathrm { o u t p u t } } \oplus \tau _ { k } } \\ { ( \lambda ) } \end{array}\tag{2}
$$

![](images/8cbed502eab71f402928dd9e887beb7eadb0dbfe590969fda0439724c705a7c2.jpg)  
Figure 2: Positional Task Conditioning. Task tokens $( \tau _ { 1 } , \dots , \tau _ { k } )$ are inserted at structural boundaries within each prompt: before the introduction, before the input data, and at the end. The teacher LLM data is generated using these task tokens and used to distill the student LLM.

where ⊕ denotes concatenation. PTC optimizes

$$
\theta _ { \mathrm { p t c } } = \underset { \theta } { \arg \operatorname* { m i n } } \sum _ { k = 1 } ^ { | \mathcal { T } | } \sum _ { i = 1 } ^ { N _ { k } } - \log P _ { \theta } ( r _ { i } ^ { ( k ) } , y _ { i } ^ { ( k ) } | \tilde { x } _ { i } ^ { ( k ) } )\tag{3}
$$

We hypothesize that each placement plays a complementary role: the initial $\tau _ { k }$ primes the task before the prompt is read, the mid-prompt $\tau _ { k }$ reanchors task identity at the transition from instructions to data, and the final $\tau _ { k }$ marks the transition into the generation phase. We test this hypothesis empirically in §4 across 7 positional configurations.

## 3.4 Synthetic data pipeline

To generate evaluation data at scale, we design a synthetic pipeline that produces families with controlled error injections. Given a family F (§ 3.1), each product $P _ { i }$ has an associated textual description $d _ { i } .$ . For each attribute $a \in V ( F )$ , we first generate a clean list of values by prompting an LLM with a sanitized description (brand-anonymized, attribute-references removed) and attribute-specific formatting instructions: $v _ { \mathrm { g e n } }$ We then inject errors via type-specific functions Inj for each task $t _ { k } \in \tau$ , producing $v _ { \mathrm { d e f e c t } } ~ = ~ \mathrm { I n j } _ { k } ( v _ { \mathrm { g e n } } )$ . Injection strategies are prompt-based (e.g., generating semantic duplicates) or deterministic (e.g., mixing unit systems). To avoid cross-interference when injecting multiple error types into the same family, we create an injection planner $\begin{array} { r l } { a c _ { k } } & { { } = } \end{array}$ (conditions, effects, $\operatorname { I n j } _ { k } )$ to find a valid injection order respecting inter-error dependencies.

## 4 Experimental setup

We address four research questions: RQ1: Does divide-and-conquer outperform monolithic prompting? RQ2: Does PTC outperform prompting and RBD? RQ3: Does PTC generalize across models? RQ4: How does task token placement affect

![](images/115244e177986903f4d8464fca8c2e9b5c312a547b0c5e7f9c9394fd17f2d140.jpg)  
Figure 3: Divide-and-conquer on a t-shirt family F: we extract the variating attributes $V ( F )$ (dropping the constant material) and classify each (task, attribute) pair independently as no error (ϕ), non-egregious (NE), or egregious (E). Two egregious defects are detected in this example: Overstuffing on color and Duplicates on size.

## performance?

Tasks. To comprehensively model the quality of a product family, we decompose eight error types E into four tasks, where each task is a pair of semantically related egregious (E) and non-egregious (NE) error variant. Hence, our D&C approach becomes four independent classification problems, each with three labels: egregious error, non-egregious error, or no error. Our tasks are: overstuffing $( t _ { 1 } ) \colon$ extraneous concepts injected into an attribute field (E: inconsistently, NE: consistently); unit systems (t<sub>2</sub>): incompatible measurement types (E: mixed systems, NE: mixed units within one system); duplicates (t<sub>3</sub>): semantically identical variations such as XL vs. Extra Large (E) or formatting inconsistencies (NE); and theme (t ): attribute values belonging to a different attribute entirely (E: inconsistently, NE: consistently). Egregious errors directly impair customer decisions; NE errors are quality concerns that do not mislead customer. Each task token $\tau _ { k }$ is a short natural-language marker inserted verbatim into the prompt at the positions defined in §3.3; e.g., the duplicates task uses <TASK=DUPLICATES>. These markers are present in the student’s training inputs, so the student learns to condition on them.

Baselines and methods. For RQ1, we compare monolithic prompting (all eight error types simultaneously) against D&C (four separate three-class sub-tasks) using Claude Sonnet 4.5.

For RQ2, we compare three configurations: (1)

a non-finetuned baseline on student models using D&C prompts without parameter updates; (2) RBD, which finetunes on reasoning chains without explicit task conditioning; and (3) PTC, which inserts task identifiers at structural prompt boundaries.

For RQ3, we evaluate across five models spanning two architecture families: Qwen3-8B, Qwen3- 14B, Qwen3-32B (Yang et al., 2025), Mistral-Nemo 12B (Mistral AI and NVIDIA, 2024), and Mistral-Small 24B (Mistral AI, 2024). RBD and PTC share identical training data and hyperparameters; for the positional ablation, only task token placement varies.

For RQ4, we perform ablations on task token placement on Qwen3-8B, exhaustively evaluating all 7 non-empty subsets of three structurally meaningful positions (start of prompt, before input, end of prompt) defined in §3.3 on the human labeled evaluation set detailed below.

Metrics and datasets. Our primary metric is detection of error (either E or NE) evaluated via precision, recall, and F1. We additionally analyze per task trends in the results and report full per task metrics in the appendix.

For training, we compile 73,723 family-attribute pairs from 45,000 product families sampled across four English-language countries (approx 14% defect rate). Training labels are generated via distillation from Claude Sonnet 4.5 using divide-andconquer prompting (Sec. 3.2). We run the teacher thrice with temperature 1 and retain only unanimous labels (98.13% of samples), discarding ambiguous cases. We use an 85/15 train/validation split stratified by error type.

For evaluation, our primary test set is a human labeled dataset of 2,296 family-attribute pairs (390 with any error; 257 overstuffing, 57 unit systems, 50 theme, 106 duplicates) with labels annotated by SMEs, verified through a 10% blind audit achieving ≥98% accuracy. <sup>1</sup> Since high-quality human annotation is costly and certain error types are underrepresented in real product families, we additionally construct a synthetic test set of 11,130 family-attribute pairs (4,562 with any error; 3,181 overstuffing, 825 unit systems, 1,372 theme, 1,158 duplicates) via the pipeline in Sec. 3.4, enabling controlled ablation studies with balanced task representation.

<table><tr><td>Model</td><td>Method</td><td>Precision</td><td>Recall</td><td>F1 score</td></tr><tr><td></td><td></td><td></td><td></td><td>Claude 4.5 Monolithic 0.4338 (±0.0498) 0.6535 (±0.0391) 0.5214 (±0.0489)</td></tr><tr><td>Sonnet</td><td>D&amp;C</td><td></td><td></td><td>0.9722 (±0.0175) 0.7974 (±0.0402) 0.8762 (±0.0264)</td></tr></table>

Table 1: Comparison of monolithic and divide-andconquer prompting strategies on Claude 4.5 Sonnet. Decomposing the multi-label classification problem into focused sub-tasks improves F1 from 52.14% to 87.62%.

Implementation details. We finetune using Axolotl (Axolotl maintainers and contributors, 2023) with QLoRA (rank 64, alpha 128, 4-bit NF4 quantization, LoRA dropout 0.05) (Dettmers et al., 2023), training for 3 epochs with early stopping on heldout data. All experiments run on 8x NVIDIA H100 GPUs (112-384 GPU hours per run depending on model size). The teacher model is Claude Sonnet 4.5 with D&C prompting (§ 3.2). We use fused AdamW (learning rate 2e-4, cosine schedule, 100 warmup steps), micro batch size 2 with gradient accumulation over 16 steps (effective batch size 32), and sequence length 8192 tokens. We use the same training data and hyperparameters for RBD and TCF. Inference is performed using vLLM (Kwon et al., 2023).

## 5 Results

Divide-and-conquer effectiveness. Table 1 compares monolithic and D&C prompting on Claude Sonnet 4.5 (RQ1). D&C achieves 87.62% F1, outperforming the monolithic baseline (52.14% F1). The 54% precision improvement is most pronounced, indicating the monolithic approach suffers high false positive rates when reasoning over multiple error types simultaneously. Decomposition into focused sub-tasks with reduced context transforms production-unviable precision (43.38%) into deployment-suitable performance (97.22%).

PTC vs baselines. Table 2 compares nonfinetuned, RBD, and PTC models across five architectures (RQ2, RQ3). PTC consistently achieves the highest F1, with improvements over RBD ranging from 1.53% (Mistral Small 24B) to 6.30% (Qwen3 8B). PTC improvements are consistent across both Qwen and Mistral families, and the 8B model (84.23% F1) matches the 14B variant (84.62%), suggesting finetuning normalizes capacity differences. Both finetuning approaches substantially outperform non-finetuned baselines.

Results on synthetic data (Table 3) confirm the same trends with higher absolute scores: PTC achieves 96.62% F1 on Qwen3 8B versus 94.59% for RBD. The absolute difference between real and synthetic performance reflects the inherent complexity of real product data. Real product data may contain cases where visually distinct products share near-identical names (e.g., Red1 and Red) but represent genuinely different color variants within the same family. As the synthetic pipeline is text-only, it does not generate these ambiguous cases, yielding cleaner decision boundaries and higher recall.

<table><tr><td>Model Method</td><td></td><td>Precision</td><td>Recall F1 score</td></tr><tr><td>Qwen3 8B</td><td>RBD PTC</td><td>Non FT 0.7732 (±0.0543) 0.4536 (±0.0490) 0.5718 (±0.0452) 0.9228 (±0.0311) 0.6744 (±0.0465) 0.7793 (±0.0346)</td><td>0.9344 (±0.0271) 0.7667 (±0.0418) 0.8423 (±0.0290)</td></tr><tr><td>Nemo 12B</td><td>RBD PTC</td><td>Mistral Non FT 0.5599 (±0.0451) 0.6104 (±0.0484) 0.5840 (±0.0393) 0.9364 (±0.0285) 0.6795 (±0.0463) 0.7875 (±0.0340) 0.9066 (±0.0315) 0.7718 (±0.0418) 0.8338 (±0.0295)</td><td></td></tr><tr><td>Qwen3 14B</td><td>RBD PTC</td><td>Non FT 0.8607 (±0.0409) 0.7197 (±0.0446) 0.7839 (±0.0341) 0.9386 (±0.0303) 0.7244 (±0.0422) 0.8176 (±0.0297)</td><td>0.9112 (±0.0303) 0.7897 (±0.0408) 0.8462 (±0.0282)</td></tr><tr><td>Small 24B</td><td>RBD PTC</td><td>Mistral Non FT 0.8424 (±0.0489) 0.5463 (±0.0495) 0.6628 (±0.0424) 0.9192 (±0.0347) 0.7030 (±0.0444) 0.7967 (±0.0321)</td><td>0.9135 (±0.0310) 0.7308 (±0.0440) 0.8120 (±0.0314)</td></tr><tr><td>Qwen3 32B</td><td>RBD PTC</td><td></td><td>Non FT 0.8897 (±0.0441) 0.5558 (±0.0495) 0.6842 (±0.0415) 0.9343 (±0.0291) 0.7434 (±0.0394) 0.8280 (±0.0277) 0.9157 (±0.0294) 0.8077 (±0.0392) 0.8583 (±0.0271)</td></tr></table>

Table 2: Performance comparison of non-finetuned , RBD, and PTC across five LLMs from the Qwen and Mistral families. PTC achieves the highest F1 score across all architectures.

<table><tr><td>Model Method</td><td></td><td>Precision</td><td>Recall F1 score</td></tr><tr><td>Qwen3 8B</td><td>RBD PTC</td><td>Non FT 0.9869 (±0.0041) 0.6434 (±0.0140) 0.7789 (±0.0104) 0.9899 (±0.0030) 0.9055 (±0.0084) 0.9459 (±0.0048)</td><td>0.9847 (±0.0036) 0.9483 (±0.0064) 0.9662 (±0.0038)</td></tr><tr><td>Nemo 12B</td><td>RBD PTC</td><td>Mistral Non FT 0.9254 (±0.0102) 0.5112 (±0.0146) 0.6586 (±0.0128) 0.9931 (±0.0032) 0.5708 (±0.0142) 0.7249 (±0.0115) 0.9862 (±0.0035) 0.9531 (±0.0061) 0.9693 (±0.0036)</td><td></td></tr><tr><td>Qwen3 14B</td><td>RBD PTC</td><td>Non FT 0.9842 (±0.0043) 0.6955 (±0.0135) 0.8151 (±0.0095) 0.9694 (±0.0051) 0.9290 (±0.0075) 0.9487 (±0.0047) 0.9836 (±0.0038) 0.9340 (±0.0072) 0.9582 (±0.0043)</td><td></td></tr><tr><td>Small</td><td>RBD</td><td>0.9760 (±0.0045) 0.9524 (±0.0061) 0.9640 (±0.0038)</td><td>Mistral Non FT 0.9841 (±0.0042) 0.7470 (±0.0127) 0.8493 (±0.0084)</td></tr></table>

Table 3: Performance comparison of non-finetuned, RBD, and PTC on synthetic data across four LLMs from the Qwen and Mistral families. PTC achieves the highest F1 score for every model.

Per task analysis. Table 4 shows PTC’s gains are largest on the most challenging tasks. On duplicates, where the baseline fails (20.01% F1), PTC achieves 77.84% (a 17% improvement over RBD). PTC also improves theme defects (5% over RBD) and overstuffing (4%). For unit systems, where RBD already performs well, gains are more modest (3%). This pattern suggests PTC particularly benefits tasks requiring nuanced contextual reasoning. Full per task results across all models are given in Appendix C.

<table><tr><td>Method</td><td>Overstuffing</td><td>Unit sys.</td><td>Theme</td><td>Duplicates</td></tr><tr><td>Non FT</td><td>0.6403</td><td>0.4348</td><td>0.4724</td><td>0.2001</td></tr><tr><td>RBD</td><td>0.7852</td><td>0.7619</td><td>0.7500</td><td>0.6065</td></tr><tr><td>PTC</td><td>0.8347</td><td>0.7925</td><td>0.8041</td><td>0.7784</td></tr></table>

Table 4: Per task F1 on Qwen3 8B. 95% bootstrap MoE range from ±0.03 to ±0.13. Full results with CI across all models in § C.

![](images/0801c4ccde5200cd75b394564fdda84c77ff120b1c9205bb6f6481e0c65a58b3.jpg)  
Figure 4: F1 scores across 7 positional configurations on Qwen3 8B with 95% bootstrap CI. S = start of prompt, I = before input, E = end of prompt. The optimal configuration (SIE) places task tokens at the start, input, and after the end of the prompt, achieving 84.2% F1. All PTC configurations outperform RBD (77.9%).

Positional ablation. We evaluate 7 positional configurations on Qwen3 8B (Fig. 4, RQ4). All PTC configurations outperform RBD, confirming that explicit task conditioning is beneficial regardless of placement. The best configuration is SIE (84.2% F1), placing task tokens at the three functional transition points: before the introduction (start), before the input data, and after the output (end). Among single positions, placement before the introduction (S: 83.0%) outperforms others (I: 81.9%, E: 81.6%), aligning with the intuition that early conditioning establishes task-appropriate representations. Most multi-position configurations do not improve over the best single position; SIE is the exception, combining early priming (S) with reinforcement at the two boundaries where the model shifts processing mode, from instructions to data and from context to generation.

Qualitative analysis: PTC vs RBD. Table 5 illustrates the mechanism behind PTC’s recall improvement over RBD. In this example, both models encounter a 27-value color list containing Dark Gray and Dark Grey, a clear spelling duplicate. RBD’s reasoning explicitly identifies the pair but rationalizes it away, concluding they represent standardized variants, not duplicates. PTC flags the pair as meeting the duplicate criteria for spelling differences. We hypothesize that the task token placed immediately before the input re-anchors attention at the point where the model transitions from instructions to data, PTC retains the focus on task, unlike RBD which defers to the majority class. This may explain PTC’s systematic recall advantage over RBD (Table 2): both models detect the same evidence, but PTC maintains sufficient task conditioning to act on it.

<table><tr><td>Category Attr.</td><td></td><td>Attribute values</td><td>RBD</td><td>RBD reason</td><td>PTC</td><td>PTC reason</td></tr><tr><td>Bed</td><td>color</td><td>Beige, Black, Dark Gray, Dark Grey, correct All values are genuinely different color egregiousDark Gray&#x27; and &#x27;Dark Grey&#x27; are Light Gray, Light Grey, Navy, Olive Green, Pink, Sage, White, . .. (27 values total)</td><td></td><td>variations. Similar spellings like &#x27;Dark error Gray&#x27; and &#x27;Dark Grey&#x27; represent stan- dardized variants, not duplicates.</td><td></td><td>present, differing only by the spelling of &#x27;Gray&#x27;/Grey&#x27;, meeting the duplicate cri-</td></tr></table>

Table 5: Qualitative duplicates example on Qwen3 8B where RBD labels the variation family correct but PTC detects the duplicate. The duplicate differs only by spelling (Dark Gray/Dark Grey). RBD treats these as intentional variants, while PTC’s explicit task conditioning flags them as duplicates.

Cost analysis. Although divide-and-conquer issues up to four LLM calls per attribute, its total token cost is comparable to a single monolithic call because each sub-task uses a much shorter, taskfocused prompt. On the human-annotated test set, monolithic prompting costs approximately \$59 at public pricing (mean input 7000 tokens, output 900 tokens), versus \$63 for the D&C solution (mean input 2500 tokens, output 150 tokens per call). Thus the accuracy gains of decomposition come at negligible additional teacher cost. For deployment, distillation yields the primary savings: the same D&C workload on Mistral Nemo 12B with PTC completes in approximately 2 minutes on 8× H100 GPUs, costing approximately \$1.17 based on public instance pricing (\$35/hr) versus \$63 for Claude Sonnet 4.5, a 98% cost reduction. The public pricing costs are at the time of writing.

Statistical significance. We performed a statistical evaluation of the results, showing that the performance gains are statistically significant in the vast majority of cases. We assessed the significance using a paired subsampled bootstrap test with 5,000 class-balanced 80% draws without replacement, applying a plus-one correction. The p values are shown in §E.

## 6 Conclusion

Our divide-and-conquer strategy decomposes monolithic detection into focused sub-tasks, improving F1 from 52.14% to 87.62% when using frontier LLM. The Positional Task Conditioning approach is able to distill this capability into smaller LLMs through explicit task conditioning, outperforming Rationale-Based Distillation across five evaluated models (statistically significant on four out of five) achieving within 1.79% F1 of the frontier model at upto 98% lower inference cost. Future work should explore PTC in other product catalog quality tasks such as attribute extraction or entity matching to improve classifier quality.

## 7 Limitations

First, our evaluation is conducted on a single domain. While consistent improvements across model families and scales suggest generalizability, we have not validated PTC on other multi-label classification tasks or alternative domains.

Second, PTC successfully consolidates four subtasks into a single model, but the scalability of task conditioning to larger numbers of tasks remains unexplored. Whether a single model can maintain task discrimination as the number of tasks grows is an open question we leave to future work.

Third, we do not evaluate the impact of training data size through ablation studies. Although absolute performance likely varies with dataset scale, our comparative analysis remains robust because all methods were evaluated using identical data and hyperparameters. Consequently, whether PTC enhances sample efficiency relative to RBD is a question we leave to future research.

Finally, a notable limitation is the proprietary nature of our dataset. To mitigate this concern, we provide a thorough description of the quality assessment tasks alongside a detailed statistical breakdown of the dataset’s composition.

## References

Axolotl maintainers and contributors. 2023. Axolotl: Open source llm post-training.

Nick Baumann, Alexander Brinkmann, and Christian Bizer. 2024. Using llms for the extraction and normalization of product attribute values. In Proceedings ofthe 28th European Conference on Advances in Databases and Information Systems (ADBIS 2024).

Rich Caruana. 1997. Multitask learning. Machine learning, 28(1):41–75.

Jiao Chen, Luyi Ma, Xiaohan Li, Nikhil Thakurdesai, Jianpeng Xu, Jason HD Cho, Kaushiki Nag, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2023. Knowledge graph completion models are few-shot learners: An empirical study of relation labeling in e-commerce with llms. arXiv preprint arXiv:2305.09858.

Zhu Cheng, Wen Zhang, Chih-Chi Chou, You-Yi Jau, Archita Pathak, Peng Gao, and Umit Batur. 2024. E-commerce product categorization with llm-based dual-expert classification paradigm. In Proceedings ofthe 1st Workshop on Customizable NLP: Progress and Challenges in Customizing NLPfor a Domain, Application, Group, or Individual (CustomNLP4U), pages 294–304.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning, pages 10421–10430. PMLR.

Shuzheng Gao, Chaozheng Wang, Cuiyun Gao, Xiaoqian Jiao, Chun Yong Chong, Shan Gao, and Michael Lyu. 2025. The prompt alchemist: Automated llmtailored prompt optimization for test case generation. arXiv preprint arXiv:2501.01329.

Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. 2021. Knowledge distillation: A survey. International journal of computer vision, 129(6):1789–1819.

Pedro Herrero-Vidal, You-Lin Chen, Cris Liu, Prithviraj Sen, and Lichao Wang. 2024. Learning variant product relationship and variation attributes from e-commerce website structures. arXiv preprint arXiv:2410.02779.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023. Large language models are reasoning teachers. In Proceedings ofthe 61st annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 14852–14882.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017.

Ihab F Ilyas and Xu Chu. 2019. Data cleaning. Morgan & Claypool.

Satish Kathiriya, Mahidhar Mullapudi, and Rajath Karangara. 2023. Optimizing ecommerce listing: Llm based description and keyword generation from multimodal data. Int. J. Sci. Res.(IJSR), 12:2123– 2130.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2022. Decomposed prompting: A modular approach for solving complex tasks. arXiv preprint arXiv:2210.02406.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Zhiqiang Kou, Junyang Chen, Xin-Qiang Cai, Ming-Kun Xie, Biao Liu, Changwei Wang, Lei Feng, Yuheng Jia, Gang Niu, Masashi Sugiyama, and 1 others. 2025. Rethinking toxicity evaluation in large language models: A multi-label perspective. arXiv preprint arXiv:2510.15007.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan. 2020. Deep entity matching with pre-trained language models. arXiv preprint arXiv:2004.00584.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. 2019a. Multi-task deep neural networks for natural language understanding. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 4487–4496.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019b. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Marcus Ma, Georgios Chochlakis, Niyantha Maruthu Pandiyan, Jesse Thomason, and Shrikanth Narayanan. 2025. Large language models do multi-label classification differently. arXiv preprint arXiv:2505.17510.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. 2023. Teaching small language models to reason. In Proceedings ofthe 61st annual meeting ofthe association for computational linguistics (volume 2: short papers), pages 1773–1781.

Fatemeh Mansoori, Andrea Scarinci, Aditya Aggarwal, Suleiman A Khan, and Ashwin Chandramouli. 2026. Improving cascade routing for structured attribute generation with heterogeneous confidence. In AdaptFM: Resource-Adaptive Foundation Model Inference.

Reza Yousefi Maragheh, Lalitesh Morishetti, Ramin Giahi, Kaushiki Nag, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2023. Llm-based aspect augmentations for recommendation systems.

Mistral AI. 2024. Mistral small: Optimized portability and performance. https://mistral.ai/news/ mistral-small-3/. 24B Parameter Model.

Mistral AI and NVIDIA. 2024. Mistral nemo. https: //mistral.ai/news/mistral-nemo/. Released July 2024.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707.

Virginia Negri, Víctor Martínez Gómez, Sergio A. Balanya, and Subburam Rajaram. 2025. Attributeaware controlled product generation with llms for e-commerce. Preprint, arXiv:2601.04200.

Athanasios N Nikolakopoulos, Swati Kaul, Siva Karthik Gade, Bella Dubrov, Umit Batur, and Suleiman Ali Khan. 2023. Sage: Structured attribute value generation for billion-scale product catalogs. arXiv preprint arXiv:2309.05920.

Nobal Niraula, Samet Ayhan, Balaguruna Chidambaram, and Daniel Whyatt. 2024. Multi-label classification with generative large language models. In 2024 AIAA DATC/IEEE 43rd Digital Avionics Systems Conference (DASC), pages 1–7. IEEE.

Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. 2024. Optimizing instructions and demonstrations for multi-stage language model programs. arXiv preprint arXiv:2406.11695.

Diego Ortego, Marlon Rodríguez, Mario Almagro, Kunal Dahiya, David Jiménez, and Juan C. SanMiguel. 2025. Large language models meet extreme multilabel classification: Scaling and multi-modal framework. Preprint, arXiv:2511.13189.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.

Md Omar Faruk Rokon, Andrei Simion, Weizhi Du, Musen Wen, Hong Yao, and Kuang chih Lee. 2026. Enhancement of e-commerce sponsored search relevancy with llm. Preprint, arXiv:2607.03886.

Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098.

Soham Satyadharma, Fatemeh Sheikholeslami, Swati Kaul, Aziz Umit Batur, and Suleiman A Khan. 2025. Auto prompting without training labels: An llm cascade for product quality assessment in e-commerce catalogs. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 937–953.

Oliver Schmidts, Bodo Kraft, Marvin Winkens, and Albert Zündorf. 2020. Catalog integration of lowquality product data by attribute label ranking. In DATA, pages 90–101.

Seyed Amin Tabatabaei, Sarah Fancher, Michael Parsons, and Arian Askari. 2025. Can large language models serve as effective classifiers for hierarchical multi-label classification of scientific documents at industrial scale? In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 163–174.

Ameni Trabelsi, Maria Zontak, Yiming Qian, Brian Jackson, Suleiman Khan, and Umit Batur. 2025. What matters when building vision language models for product image analysis? In Proceedings of the Winter Conference on Applications of Computer Vision, pages 1372–1381.

Abhinav Shankaranarayanan Venkataraman, Athanasios Nikolakopoulos, Vishwanath Kumaraswamy, Tao Zhang, Sarath Chander, Rohit Saboo, and Suleiman A. Khan. 2026. A framework for prompt optimization and translation across foundation models. In ICLR 2026 Workshop on AI with Recursive Self-Improvement.

Gaike Wang, Xin Ni, Qi Shen, and Mingxuan Yang. 2024. Leveraging large language models for contextaware product discovery in e-commerce search systems. Journal of Knowledge Learning and Science Technology ISSN: 2959-6386 (online), 3(4):300–312.

Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao, Bing Yin, and Xiang Ren. 2023. Scott: Self-consistent chain-of-thought distillation. arXiv preprint arXiv:2305.01879.

Yaqing Wang, Yifan Ethan Xu, Xian Li, Xin Luna Dong, and Jing Gao. 2020. Automatic validation of textual attribute values in e-commerce catalog by learning with limited labeled data. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2533–2541.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Pei Xu, Liang Chen, and Radhika Santhanam. 2015. Will video be the next generation of e-commerce product reviews? presentation format and the role of product type. Decision Support Systems, 73:85–96.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Bryan Zhang, Suleiman A Khan, and Stephan Walter. 2025. Leveraging product catalog patterns for multilingual e-commerce product attribute prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 267–275.

Chuang Zhou, Junnan Dong, Xiao Huang, Zirui Liu, Kaixiong Zhou, and Zhaozhuo Xu. 2024. Quest: Efficient extreme multi-label text classification with large language models on commodity hardware. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3929–3940.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and 1 others. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625.

## A Prompt templates

This section presents the prompt template structure used for PTC. Figure 5 shows an example template for detecting overstuffing in the size attribute. All prompts follow a consistent structure across tasks, with task-specific content in each section.

The task identifier (e.g., <TASK=OVERSTUFFING>) is inserted at three structural boundaries as described in Sec. 3.3: (1) before the introduction, establishing taskappropriate attention patterns from the first layer of processing; (2) before the input data, re-anchoring task context at the transition from instructions to data; and (3) after the output specification, conditioning the autoregressive generation phase.

The <introduction> section defines the classification objective and provides context about the attribute under evaluation. The <rules> and <examples> sections collectively constitute the instructions defined in Sec. 3.2: <rules> contains detailed criteria for each severity level (no error, non-egregious, and egregious), capturing domainspecific conventions and edge cases refined with subject matter experts, while <examples> provides 6-8 few-shot demonstrations per class to calibrate model outputs. The <input> section contains the attribute values to be evaluated, and the <output> section specifies the required response format, including fields for the identified pattern, rationale, and final prediction. This structured format ensures consistent, parseable outputs suitable for both distillation and production deployment.

## B Error categorization

We have eight error types organized into four tasks, each comprising an egregious (E) and nonegregious (NE) variant. Egregious errors directly impair customer decisions—for instance, duplicate variations force customers to determine whether XL and Extra Large are truly different options, while mixed unit systems (Large vs. 10) prevent meaningful comparison. Non-egregious errors are quality concerns that do not immediately mislead but degrade the browsing experience over time.

The distinction between egregious and nonegregious within each task captures a key operational insight: consistency mitigates severity. Overstuffing that applies uniformly across all values (every color includes a pack count) is less harmful than inconsistent overstuffing (only some values include extraneous information), because consistent patterns remain interpretable even if suboptimal. Similarly, theme errors where all values belong to a different attribute (all size values are actually scents) indicate a systematic misconfiguration, whereas inconsistent theme errors (a mix of colors and quantities) suggest data corruption that actively confuses customers.

## C Per task breakdown

## C.1 Human-annotated data

Table 6 reports F1 scores for each task across all models and methods on the human-annotated test set. Performance varies substantially across tasks, revealing that some defect categories are inherently more challenging for LLMs to detect. Overstuffing proves the most accessible task, with baselines achieving 53.12%-73.93% F1 and finetuned models reaching 77.43%-83.47%. Unit systems and Theme represent intermediate-difficulty tasks where baselines are inconsistent (31.22%- 74.14%) but finetuning yields reliable detection (71.58%-83.05%). Duplicates is the most challenging category: baselines catastrophically fail on most models (Qwen3 8B: 20.01%, Mistral Nemo 12B: 9.93%, Qwen3 32B: 12.28%), suggesting that duplicate detection requires reasoning capabilities that emerge only through targeted finetuning. Both RBD and PTC dramatically improve duplicates detection, with PTC achieving 77.84% on Qwen3 8B and 75.00% on Mistral Small 24B, representing 289% and 266% relative improvements over their respective baselines.

PTC’s advantage over RBD is most pronounced on the tasks where RBD leaves the largest gap. For duplicates, PTC outperforms RBD by 28% on Qwen3 8B (77.84% vs. 60.65%) and 10% on Mistral Small 24B (75.00% vs. 68.06%). A similar pattern emerges for theme defect detection, where PTC surpasses RBD by 7% on Qwen3 8B (80.41% vs. 75.00%) and 7% on Mistral Small 24B (82.69% vs. 77.55%). In contrast, for overstuffing, where RBD already achieves strong performance, PTC provides more modest gains. This pattern suggests that positional task conditioning particularly benefits error types requiring multi-step contextual reasoning (identifying semantic duplicates across products, detecting thematic inconsistencies), where sustained task identity throughout the prompt is critical for the model to maintain focus on the specific defect pattern.

<TASK=OVERSTUFFING>   
<introduction>   
Classify the given list of product size attribute values into one of three   
categories: NOT OVERSTUFFED, CONSISTENTLY OVERSTUFFED, or INCONSISTENTLY   
OVERSTUFFED.   
The list contains values that appear in the size attribute field for product   
families.   
You must analyze whether these values contain only size information or include   
additional attributes that don’t belong in a size field (overstuffing).   
</introduction>   
<rules>   
This section defines the specific criteria for each classification category:   
NOT OVERSTUFFED:   
### Rules for NOT OVERSTUFFED go here.   
CONSISTENTLY OVERSTUFFED   
### Rules for CONSISTENTLY OVERSTUFFED go here.   
INCONSISTENTLY OVERSTUFFED   
### Rules for INCONSISTENTLY OVERSTUFFED go here.   
</rules>   
<examples>   
### 6-8 examples of each class go here.   
</examples>   
<TASK=OVERSTUFFING>   
<input>   
### Input data goes here.   
</input>   
<output>   
Output the results in the following output format.   
### Output format here.   
</output>   
<TASK=OVERSTUFFING>  
Figure 5: Example prompt template for PTC. The task identifier <TASK=OVERSTUFFING> is prepended to certain areas of the prompt, providing explicit task conditioning. The template comprises five sections: introduction (task definition), rules (classification criteria for each severity level), input (family attribute values), examples (few-shot examples), and output (response format specification).

## C.2 Synthetic data

Table 7 reports per task F1 on the synthetic test set across four models. Qwen3 32B was excluded from synthetic evaluation due to prohibitive computational cost. The same difficulty ordering observed in real data holds: duplicates remains the hardest task for baselines (32.39%-45.99%), while overstuffing is more accessible. However, absolute scores are substantially higher across the board, reflecting the cleaner decision boundaries of synthetic data (see Sec. 5).

<table><tr><td>Model</td><td colspan="5">Method Overstuffing Unit systems Theme Duplicates</td></tr><tr><td rowspan="5">Qwen3 8B</td><td rowspan="2">Non FT</td><td>0.6403</td><td>0.4348</td><td>0.4724</td><td>0.2001</td></tr><tr><td>±0.0503</td><td>±0.1294</td><td>±0.1073</td><td>±0.0326</td></tr><tr><td rowspan="2">RBD</td><td>0.7852</td><td>0.7619</td><td>0.7500</td><td>0.6065</td></tr><tr><td>±0.0417</td><td>±0.0924</td><td>±0.0976</td><td>±0.0911</td></tr><tr><td rowspan="2">PTC</td><td>0.8347</td><td>0.7925</td><td>0.8041</td><td>0.7784</td></tr><tr><td>±0.0365</td><td>±0.0853</td><td>±0.0878</td><td>±0.0671</td></tr><tr><td rowspan="6">Mistral Nemo 12B</td><td rowspan="2">Non FT</td><td>0.5312</td><td>0.4737</td><td>0.3122</td><td>0.0993</td></tr><tr><td>±0.0469</td><td>±0.0987</td><td>±0.0772</td><td>±0.0684</td></tr><tr><td rowspan="2">RBD</td><td>0.7743</td><td>0.8000</td><td>0.7158</td><td>0.6627</td></tr><tr><td>±0.0434</td><td>±0.0849</td><td>±0.1037</td><td>±0.0833</td></tr><tr><td rowspan="2">PTC</td><td>0.8106</td><td>0.8034</td><td>0.7473</td><td>0.7234</td></tr><tr><td>±0.0380</td><td>±0.0800</td><td>±0.1016</td><td>±0.0728</td></tr><tr><td rowspan="6">Qwen3 14B</td><td rowspan="2">Non FT</td><td>0.7393</td><td>0.7414</td><td>0.5152</td><td>0.6354</td></tr><tr><td>±0.0428</td><td>±0.0900</td><td>±0.1050</td><td>±0.0803</td></tr><tr><td rowspan="2">RBD</td><td>0.8085</td><td>0.7652</td><td>0.7273</td><td>0.7647</td></tr><tr><td>±0.0394</td><td>±0.0866</td><td>±0.0946</td><td>±0.0650</td></tr><tr><td rowspan="2">PTC</td><td>0.8159</td><td>0.8305</td><td>0.7600</td><td>0.7615</td></tr><tr><td>±0.0384</td><td>±0.0746</td><td>±0.0933</td><td>±0.0628</td></tr><tr><td rowspan="6">Mistral Small 24B</td><td rowspan="2">Non FT</td><td>0.6816</td><td>0.6804</td><td>0.5229</td><td>0.2051</td></tr><tr><td>±0.0477</td><td>±0.1074</td><td>±0.0967</td><td>±0.0984</td></tr><tr><td rowspan="2">RBD</td><td>0.8065</td><td>0.8246</td><td>0.7755</td><td>0.6806</td></tr><tr><td>±0.0383</td><td>±0.0768</td><td>±0.0927</td><td>±0.0757</td></tr><tr><td rowspan="2">PTC</td><td>0.7881</td><td>0.8155</td><td>0.8269</td><td>0.7500</td></tr><tr><td>±0.0413</td><td>±0.0827</td><td>±0.0804</td><td>±0.0682</td></tr><tr><td rowspan="6">Qwen3 32B</td><td rowspan="2">Non FT</td><td>0.7381</td><td>0.6731</td><td>0.6260</td><td>0.1228</td></tr><tr><td>±0.0438</td><td>±0.1046</td><td>±0.0972</td><td>±0.0839</td></tr><tr><td rowspan="2">RBD</td><td>0.8168</td><td>0.8000</td><td>0.7800</td><td>0.8000</td></tr><tr><td>±0.0381</td><td>±0.0834</td><td>±0.0911</td><td>±0.0605</td></tr><tr><td rowspan="2">PTC</td><td>0.8293</td><td>0.8174</td><td>0.7863</td><td>0.8186</td></tr><tr><td>±0.0362</td><td>±0.0781</td><td>±0.0827</td><td>±0.0563</td></tr></table>

Table 6: Per task F1 (with 95% bootstrap margin of error) for non-finetuned, RBD, and PTC across five LLMs on human labeled data.

PTC achieves the highest F1 on nearly every task and model combination. The most striking result is on duplicates, where PTC reaches 94.04% on Qwen3 8B versus 85.04% for RBD—a 11% improvement—and 93.78% on Mistral Nemo 12B versus 88.93% for RBD (5% improvement). PTC’s consistent improvements across all tasks and models on synthetic data confirm that positional task conditioning provides robust task discrimination independent of data complexity.

## D Qualitative examples

Table 8 reveals that the gap between RBD and PTC on duplicate detection is not one of perception but of execution. In both examples, RBD’s reasoning explicitly surfaces the offending pair. It names Dark Gray/Dark Grey and Nimbus Cloud/Crimson/Nimbuscloud/Crimson as candidates, so the relevant variants are clearly within the model’s attention. Yet RBD then rationalizes them away, recasting a spelling difference as “different standardized spellings. . . commonly accepted as distinct color variations” and a spacing difference as something that “could represent intentional product distinctions.” Having identified the duplicate, RBD talks itself out of acting on the task. The error is therefore a failure to apply the task criterion, not a failure to notice the evidence.

<table><tr><td>Model</td><td colspan="4">Method Overstuffing Unit systems Theme Duplicates</td></tr><tr><td rowspan="6">Qwen3 8B</td><td rowspan="2">Non FT</td><td>0.8498</td><td>0.6676</td><td>0.8618</td><td>0.3918</td></tr><tr><td>±0.0100</td><td>±0.0278</td><td>±0.0141</td><td>±0.0317</td></tr><tr><td rowspan="2">RBD</td><td>0.9649</td><td>0.8589</td><td>0.9568</td><td>0.8504</td></tr><tr><td>±0.0046</td><td>±0.0169</td><td>±0.0081</td><td>±0.0157</td></tr><tr><td rowspan="2">PTC</td><td>0.9680</td><td>0.8671</td><td>0.9582</td><td>0.9404</td></tr><tr><td>±0.0044</td><td>±0.0163</td><td>±0.0077</td><td>±0.0090</td></tr><tr><td rowspan="6">Mistral Nemo 12B</td><td>Non FT</td><td>0.6307</td><td>0.5212</td><td>0.5403</td><td>0.3239</td></tr><tr><td rowspan="2"></td><td>±0.0163</td><td>±0.0297</td><td>±0.0271</td><td>±0.0290</td></tr><tr><td>0.4544</td><td>0.8422</td><td>0.6194</td><td>0.8893</td></tr><tr><td rowspan="2">RBD</td><td>±0.0188</td><td>±0.0180</td><td>±0.0234</td><td>±0.0131</td></tr><tr><td>0.9694</td><td>0.8789</td><td>0.9606</td><td>0.9378</td></tr><tr><td rowspan="2">PTC</td><td>±0.0044</td><td>±0.0159</td><td>±0.0074</td><td>±0.0093</td></tr><tr><td>0.8745</td><td>0.8300</td><td>0.8836</td><td>0.4521</td></tr><tr><td rowspan="6">Qwen3 14B</td><td>Non FT</td><td>±0.0090</td><td>±0.0187</td><td>±0.0118</td><td>±0.0292</td></tr><tr><td rowspan="2">RBD</td><td>0.9569</td><td>0.8507</td><td>0.9417</td><td>0.9031</td></tr><tr><td>±0.0050</td><td>±0.0174</td><td></td><td></td></tr><tr><td rowspan="2">PTC</td><td></td><td>0.8544</td><td>±0.0091</td><td>±0.0109</td></tr><tr><td>0.9704 ±0.0042</td><td>±0.0175</td><td>0.9620 ±0.0073</td><td>0.9080 ±0.0114</td></tr><tr><td rowspan="2">Non FT</td><td></td><td></td><td></td><td></td></tr><tr><td>0.9199</td><td>0.8580</td><td>0.8875</td><td>0.4599</td></tr><tr><td rowspan="4">Mistral Small 24B</td><td></td><td>±0.0071</td><td>±0.0179</td><td>±0.0120</td><td>±0.0291</td></tr><tr><td>RBD</td><td>0.9644</td><td>0.8613</td><td>0.9411</td><td>0.9443</td></tr><tr><td rowspan="2"></td><td>±0.0046</td><td>±0.0165</td><td>±0.0088</td><td>±0.0081</td></tr><tr><td>0.9637</td><td>0.8626</td><td>0.9537</td><td>0.9463</td></tr><tr><td></td><td>PTC</td><td>±0.0047</td><td>±0.0167</td><td>±0.0080</td><td>±0.0083</td></tr></table>

Table 7: Per task F1 (with 95% bootstrap margin of error) for non-finetuned, RBD, and PTC on synthetic data across four LLMs. Bold marks the best F1 per task within each model.

PTC encounters the same lists and flags the same pairs, but converts the observation into the correct label. We hypothesize this stems from the task token placed immediately before the input: by reanchoring the duplicate-detection objective at the point where the model transitions from instructions to data, PTC keeps the task criterion active while the candidate values are being read, rather than letting it be diluted across the preceding rule text. The effect is most visible in the 57-value SHOES list, where the lone Nimbuscloud/Crimson collision is buried among genuinely distinct multi-color combinations: PTC isolates the single offending pair and commits to the duplicate label, while RBD defers to the non-duplicate classification whenever any doubt remains. This pattern, repeated across the 29 duplicate cases that PTC recovers over RBD on Qwen3 8B, is consistent with our broader hypothesis that mid-prompt task reinforcement mitigates the attenuation of the initial task signal in long contexts.

<table><tr><td>Category</td><td>Attr.</td><td>Attribute values</td><td>RBD la- RBD reason bel</td><td>bel</td><td>PTC la- PTC reason</td><td></td></tr><tr><td>Bed</td><td>color</td><td>Beige, Black, Black &amp; Light Grey, Dark &amp; Light Grey, Dark Gray, Dark Grey, Dark Grey &amp; Light Grey, Green &amp; Ivory, Green &amp; Light Grey, Grey- ish Blue, Greyish Purple &amp; Light Grey, Light Gray, Light Grey, Navy, Navy Blue &amp; Grey, Olive Green, Orange &amp; Light Grey, Pink, Pink &amp; Ivory, Purple Red &amp; Black, Royal Blue &amp; Light Grey,</td><td>correct</td><td>All color values in the list are gen- uinely different color variations. While there are similar base colors (e.g., &#x27;Dark Gray&#x27; and &#x27;Dark Grey&#x27;, &#x27;Light Gray&#x27; and &#x27;Light Grey&#x27;), these repre- sent different standardized spellings of gray shades rather than duplicates, as they are commonly accepted as dis- tinct color variations in product listings.</td><td>egregious error</td><td>&#x27;Dark Gray&#x27; and &#x27;Dark Grey&#x27; appear in the list. These two values represent the same color with only spelling differ- ences between &#x27;Gray&#x27; and &#x27;Grey&#x27;. Both variations are present and are otherwise completely identical with no additional descriptive words. This meets the du- plicate criteria for spelling differences. All other values represent genuinely</td></tr><tr><td>Shoes</td><td>color</td><td>Angora/Light Mango, Beige/Black, correct Beige/Turtledove/Incense, Black Neo Flame, Black Nero, Black/Black, Black/Magnet, Black/Magnet/Gum, Black/Moonbeam, Black/Shadow Grey/White, ..., Night Tide As- pen, Night Tide/Aspen, Nimbus Cloud/Crimson, Nimbus Cloud/Night- watch Green, Nimbus Cloud/White, Nimbuscloud/Crimson, ..., White/-</td><td></td><td>All color values are distinct multi-color combinations with different descriptive elements. While &#x27;Night Tide Aspen&#x27; and &#x27;Night Tide/Aspen&#x27; appear simi- lar, they have spacing differences that could represent different color arrange- ments. Similarly, &#x27;Nimbus Cloud/Crim- son&#x27;and &#x27;Nimbuscloud/Crimson&#x27; have</td><td>egregious error</td><td>After analyzing all 57 color values, I found one duplicate pair: &#x27;Nim- bus Cloud/Crimson&#x27; and &#x27;Nimbus- cloud/Crimson&#x27;. These differ only in spacing (space vs no space between &#x27;Nimbus&#x27; and &#x27;Cloud&#x27;), making them identical except for formatting. All other values represent genuinely differ-</td></tr></table>

Table 8: Qualitative duplicates examples on Qwen3 8B where RBD labels the variation family correct but PTC detects the duplicate. In both cases the duplicate differs only by casing/spelling (Dark Gray/Dark Grey) or spacing (Nimbus Cloud/Nimbuscloud); RBD treats these as intentional variants, while PTC’s explicit task conditioning flags them as duplicates.

<table><tr><td>Model</td><td>PTC vs. method</td><td>p value</td></tr><tr><td>Qwen3 8B</td><td>Non FT RBD</td><td>&lt; 0.0002 &lt; 0.0002</td></tr><tr><td>Mistral Nemo 12B</td><td>Non FT RBD</td><td>&lt; 0.0002 0.0024</td></tr><tr><td>Qwen3 14B</td><td>Non FT RBD</td><td>&lt; 0.0002 0.036</td></tr><tr><td>Mistral Small 24B</td><td>Non FT RBD</td><td>&lt; 0.0002 0.0686</td></tr><tr><td>Qwen3 32B</td><td>Non FT RBD</td><td>&lt; 0.0002 0.0134</td></tr></table>

where the improvement is directionally consistent but does not reach significance.  
Table 9: Statistical significance of $\mathrm { P T C } \mathbf { \vec { s } }$ performance gains over the non-finetuned baseline and RBD, measured via a paired subsampled bootstrap test (5,000 class-balanced 80% draws without replacement, plusone correction) on the human-annotated test set. PTC is significant $( p < 0 . 0 5 )$ over RBD on four of five models.

## E Statistical significance

Table 9 reports the p-values from the paired subsampled bootstrap test described in §5, comparing PTC against both the non-finetuned baseline (Non FT) and rationale-based distillation (RBD) on the human-annotated test set. PTC’s improvement over the non-finetuned baseline is highly significant $( p < 0 . 0 0 0 2 )$ across all five models. Against the stronger RBD baseline, PTC’s gains are statistically significant $( p < 0 . 0 5 )$ on four of five models. The exception is Mistral Small 24B $( p = 0 . 0 6 8 6 )$