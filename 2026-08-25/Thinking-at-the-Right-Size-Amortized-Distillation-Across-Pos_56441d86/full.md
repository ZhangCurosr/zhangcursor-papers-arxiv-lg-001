# Thinking at the Right Size: Amortized Distillation Across Post-Trained LLMs

Yan Zhou<sup>1</sup>, Sara Kangaslahti<sup>1</sup>\*, Jonathan Geuter<sup>1,2</sup>\*, Nihal V. Nayak<sup>1</sup>, Marco Fumero<sup>3</sup>, Francesco Locatello<sup>3</sup>, David Alvarez-Melis<sup>1,2</sup>

<sup>1</sup>Harvard University, <sup>2</sup>Kempner Institute, <sup>3</sup>IST Austria Correspondence: terryzhou@fas.harvard.edu

## Abstract

Practical deployment of large language mod els (LLMs) requires families of post-trained variants—instruction-tuned, reasoning-tuned, and chat-style models—each at multiple sizes to meet diverse latency and memory budgets. Producing each (variant, size) pair independently is prohibitive, so model families typically span only a handful of coarse-grained sizes per post-trained variant. Boomerang distillation (Kangaslahti et al., 2026a) reduces this cost along the size axis for base models. Through model size interpolation, it constructs models of intermediate sizes from a single teacher-student pair without additional training. However, it still treats each post-trained variant as a separate object of optimization. We introduce ADAPT—Amortized Distillation Across Post-Trained LLMs—a framework for amortizing distillation across both axes of a model family: size and post-training variant, producing L K models for L interpolated sizes across K post-trained variants with a single distillation run. ADAPT combines two components. First, a two-phase distillation procedure constructs post-trained students through pre-training alignment and supervised fine-tuning distillation, enabling smooth size–performance interpolation on generation and reasoning tasks. Second, weight-delta initialization approximates this construction across post-trained variants by transferring the distillation-induced weight change from the base model to students initial ized from different post-trained variants. The resulting continuum of interpolated models also enables adaptive model-size selection at inference time, improving the compute–accuracy trade-off for long-form reasoning tasks.<sup>1</sup>

## 1 Introduction

Deployed LLM applications rely overwhelmingly on post-trained models—instruction-tuned, reasoning-tuned, and chat-style variants—rather than base pre-trained models. They also span a wide range of compute environments, from mobile devices to personal laptops to on-premise servers with multi-GPU clusters, with correspondingly diverse latency and memory constraints (Huyen, 2022; Narayan et al., 2025). Practical use thus requires families of models that vary along two axes: size, to meet deployment compute constraints, and post-trained variants, to support different downstream capabilities. Training each (variant, size) combination from scratch is computationally infeasible, so model families are typically released only at a few coarse-grained sizes per variant, and only at a few variants per base model (Qwen Team, 2026).

Existing approaches address one of these axes but not both. Task vectors (Ilharco et al., 2023) transfer fine-tuned capabilities across models without additional training, but are constrained to only models of the same size. Boomerang distillation (Kangaslahti et al., 2026a) transfers teacher capability to a fine-grained family of smaller models with a single distillation run. It thus efficiently covers the size axis, though only in the pre-trained setting and leaves the cross-variant axis unaddressed. Whether this strategy extends to post-trained models, and whether size interpolation can be applied zero-shot across multiple post-trained variants of the same base model, are the key questions this paper investigates.

A natural first approach is to apply boomerang distillation (BD) directly to a post-trained teacher: initialize a student from the teacher, distill on pre-training data, and patch the layers back in to create intermediate-sized models. Another approach to address the cross-variant axis is to cross-patch post-trained variants onto the boomerang-distilled base model, hoping that post-trained model capabilities can be transferred through patching alone. We find that neither approach works: generation and reasoning performance collapses across small to medium sizes on in-domain and out-of-domain tasks for both setups, despite the post-trained teacher itself achieving strong performance on these tasks (Figure 2). This asymmetry suggests that instruction-following depends on properties of the distillation process that the original BD recipe does not provide.

![](images/9815dab3cb6a653864c530fa8ac99b810ebc57981d2ef5139745ea2f2e6e59da.jpg)  
Figure 1: ADAPT: Amortized Distillation Across Post-Trained LLMs. We perform two-phase distillation (pre-training + SFT) on a base student once, yielding L interpolated model sizes from a single teacher–student pair. We then compute the distillation-induced weight delta and add it to the initialized students from multiple post-trained teachers. This amortizes distillation for post-trained LLMs: one distillation run can now produce L K models across K post-trained variants.

In response, we introduce ADAPT—Amortized Distillation Across Post-Trained LLMs—a framework for constructing post-trained model families across both model size and post-training variant axes. ADAPT combines two-phase distillation with weight-delta initialization. First, two-phase distillation distills a student for each post-trained variant, using both pre-training alignment and supervised fine-tuning distillation for instruction-following and reasoning, enabling smooth model size interpolation without any additional training. We then provide weight-delta initialization as an approximate construction that avoids separate distillation runs by transferring the distillation-induced weight change from the base model to students initialized from different post-trained variants of the same base model. This allows a single base-model distillation run to produce L K models: L interpolated model sizes for each of K post-trained variants.

Our experiments show that ADAPT recovers smooth size-performance interpolation on generation and reasoning benchmarks and outperforms standard boomerang distillation and layer pruning baselines at equal compute. The resulting family of intermediate-sized models supports adaptive inference: routing easier inputs to smaller models traces an empirical compute–accuracy Pareto frontier that dominates any single fixed model. To understand why a single base-model distillation transfers to multiple post-trained variants, we examine the underlying weight-space geometry. Empirically, base and post-trained models—and their corresponding distilled students—are connected by low-loss linear interpolation paths, a form of smooth weight interpolation that is preserved through distillation. The distillation update thus operates within a connected region of weight space shared by base and post-trained models, providing a geometric explanation of why weight-arithmetic transfer succeeds in this setting.

In summary, our contributions are:

• We introduce ADAPT, a framework for amortizing distillation runs to create interpolated posttrained model families across sizes and posttraining variants.

• We show that ADAPT outperforms strong baselines, including boomerang distillation and layer pruning approaches, while enabling improved performance–compute trade-offs through adaptive model-size selection.

• We provide an empirical analysis of size-agnostic smooth weight interpolation, showing that the weight-space structure connecting base and posttrained models is preserved after distillation and helps explain the success of weight-delta initialization.

## 2 Related Work

Knowledge distillation. Knowledge distillation is an effective technique for training a smaller model using a larger teacher model (Hinton et al., 2015; Sanh et al., 2020). Several LLM families use knowledge distillation to train or post-train smaller models, including Qwen (Yang et al., 2025), DeepSeek (Guo et al., 2025), and Gemma (Team et al., 2025). These smaller LLMs show improved performance when compared to naive training, but require individual distillation runs. In this work, we introduce a two-phase distillation pipeline and a weight-arithmetic procedure that produces a family of post-trained models using a single distillation run.

Post-training LLMs. Post-training is a critical step in the language model training pipeline, in which pre-trained LLMs are trained to follow instructions (Olmo et al., 2025; Guo et al., 2025; Bercovich et al., 2025). Because post-training is expensive, recent work aims to reduce its compute cost by training on smaller but effective datasets (Ye et al., 2025; Nayak et al., 2026), but it still requires training models at every size. Here, we distill a single post-trained student LLM and then create interpolated post-trained LLMs of varying sizes without training additional models, thereby dramatically reducing compute cost.

Model arithmetic. Model arithmetic enables practitioners to combine the skills of multiple models into a single unified model by simply adding or subtracting model weights (Ilharco et al., 2023). Prior work has used model arithmetic for instruction-following (Ilharco et al., 2023), creating multitask models (Yadav et al., 2025), and machine unlearning (Liu et al., 2025). In this work, we use model arithmetic to amortize student distillation by transferring the distillation-induced weight delta from a base student to students initialized from multiple post-trained variants.

Adaptive compute. Adapting LLM compute to input difficulty significantly reduces latency requirements during inference (Taghibakhshi et al., 2026). While recent work has shown promising results for test-time scaling (Muennighoff et al., 2025), many of these approaches require training fine-grained models from scratch (Kusupati et al., 2022) or rely on heuristics such as appending thinking tokens (Muennighoff et al., 2025). In this work, we dynamically adjust inference compute by adapting the model size based on the input difficulty for reasoning and generation tasks.

Layer pruning. Layer pruning removes transformer layers from a trained LLM under an importance criterion, sometimes followed by a healing step (Men et al., 2025; Chen et al., 2025). While effective at preserving classification accuracy, recent analyses show that pruning collapses openended generation (Shrestha et al., 2026). ADAPT instead recovers generation capability through a targeted two-phase distillation step, and substantially outperforms layer pruning approaches such as ShortGPT and LLM-Streamline (Appendix H).

## 3 ADAPT: Amortized Distillation Across Post-Trained LLMs

We begin by summarizing boomerang distillation (Kangaslahti et al., 2026a), a recently proposed method for zero-shot model size interpolation. We then introduce the two versions of ADAPT: ADAPT (distilled), a two-phase distillation framework for constructing interpolated post-trained models, and ADAPT (weight-delta), a weight-delta initialization method that approximates the two-phase distillation process across model variants and constructs multiple post-trained model families from a single distillation run.

## 3.1 Boomerang Distillation

Boomerang distillation (Kangaslahti et al., 2026a) distills a small student from a teacher in such a way that intermediate models can be constructed by combining student and teacher layers without additional training. We now discuss its three stages.

Student initialization. Let T be the teacher with N layers denoted by $\pmb { \theta } _ { T }$ , and S with parameters $\pmb { \theta } _ { S }$ and M layers be the student, where $M < N$ . The student is initialized by partitioning the N teacher layers into M blocks of consecutive layers. From each of these blocks, the first layer is used to initialize the corresponding student layer. The student’s embedding and LM head are initialized from T.

Knowledge distillation. The initialized student is trained on a pre-training corpus such as the Pile (Gao et al., 2020) on a combined loss involving a logit-level knowledge distillation loss, such as KL, and a block-wise alignment loss, such as cosine distance, that pushes each student layer’s output toward the outputs of the corresponding block of teacher layers. Specifically, given a training corpus  and a sample $x = ( x _ { 1 } , . . . , x _ { L } ) \in \mathcal { D }$ , an index j, and denoting the next-token distributions by $p ( \cdot \mid x _ { < j } )$ , the per-token loss $\mathcal { L } _ { \theta _ { S } } ( x _ { j } )$ is:

$$
\begin{array} { l } { { \displaystyle { \mathcal L } _ { \theta _ { S } } ( x _ { j } ) = \mathrm { C E } ( x _ { j } \mid x _ { < j } ; \theta _ { S } ) } } \\ { { \displaystyle ~ + \lambda _ { \mathrm { K L } } \mathrm { K L } \big ( p _ { T } ( \cdot \mid x _ { < j } ) \big \| p _ { S } ( \cdot \mid x _ { < j } ) \big ) } } \\ { { \displaystyle ~ + \lambda _ { \mathrm { c o s } } \sum _ { i = 1 } ^ { M } { \mathcal L } _ { \mathrm { c o s } } ^ { ( i ) } ( x _ { < j } ; \theta _ { T } , \theta _ { S } ) } . } \end{array}\tag{1}
$$

Here, CE denotes the next-token cross-entropy loss and $\mathcal { L } _ { \mathrm { c o s } } ^ { ( i ) }$ is the cosine distance between the outputs of the ith student layer and the ith teacher block. $\lambda _ { \mathrm { K L } } > 0$ and $\lambda _ { \mathrm { c o s } } > 0$ are hyperparameters that weight the KL and cosine terms relative to crossentropy.

Student patching. After the distillation stage, boomerang distillation constructs intermediate models by patching—i.e., replacing student layers with their corresponding sets of teacher layers. We treat student patching as an important but complementary design choice and use simple front-to-back and back-to-front patching in the main experiments, selecting the better-performing of the two orders for each model (see Appendix J for details).

## 3.2 ADAPT

## 3.2.1 ADAPT (Distilled): Two-Phase Post-Trained Distillation

We propose a unified two-phase distillation procedure for model size interpolation in post-trained models. Replacing the original distillation stage in boomerang distillation, the student first undergoes a pre-training phase, followed by a supervised finetuning phase. The pre-training phase ensures that we recover the general language modeling capability in the initialized student model before restoring the instruction-following and reasoning capabilities. In Section 4.2, we find that the two-phase distillation prevents overfitting and shows better out-of-domain performance.

Pre-training phase. We follow the same boomerang distillation setup over a pre-training corpus $\mathcal { D } _ { \mathrm { p r e } }$ Given a training sequence $x \quad =$ $( x _ { 1 } , \ldots , x _ { L } ) \in \mathcal { D } _ { \mathrm { p r e } }$ , the loss from Equation 1 is summed over all positions $x _ { 2 } , . . . , x _ { L }$ . Training on pre-training data not only mitigates catastrophic forgetting, but has even been shown to improve downstream performance (Kotha and Liang, 2026).

Supervised fine-tuning phase. In this phase, we transition to instruction-following data ${ \mathcal { D } } _ { \mathrm { S F T } }$ where each sequence $( x , y ) \in \mathcal { D } _ { \mathrm { S F T } }$ consists of a prompt $\boldsymbol { x } = ( x _ { 1 } , . . . , x _ { L } )$ and a response $y =$ $\left( y _ { 1 } , . . . , y _ { L ^ { \prime } } \right)$ . We use the same distillation objective, but loss (Equation 1) is computed only over all the response tokens. This stage allows the student to adopt instruction-following capabilities and adapt to downstream generation tasks, while preserving alignment to the teacher.

We dedicate half of training to pre-training and half to supervised fine-tuning (see Appendix E.1). We also use a continuous LR schedule for both phases as we found in initial experiments that two separate schedulers can hurt final performance.

## 3.2.2 ADAPT (Weight-Delta): Weight-Delta Initialization

Although the two-phase distillation produces strong interpolated model families for generation and reasoning tasks (Figure 2), applying it independently to each post-trained variant still requires a separate distillation run. We propose weight-delta initialization, a weight-arithmetic procedure that transfers a base-model distillation run to a posttrained variant without additional training, approximately constructing distilled post-trained student models and amortizing the cost of constructing multiple post-trained model families.

Let $\pmb { \theta } _ { S , \mathrm { i n i t } } ^ { \mathrm { b a s e } }$ denote the student initialized from the base teacher, and let $\theta _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { b a s e } }$ denote the corresponding student after two-phase distillation. We define the distillation delta as:

$$
\Delta _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { b a s e } } = \pmb { \theta } _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { b a s e } } - \pmb { \theta } _ { S , \mathrm { i n i t } } ^ { \mathrm { b a s e } } .
$$

Given a student $\theta _ { S , \mathrm { i n i t } } ^ { \mathrm { P T } }$ initialized from a posttrained teacher using the same layer-selection rule and parameter shapes as $\theta _ { S , \mathrm { i n i t } } ^ { \mathrm { b a s e } } ,$ we construct the weight-delta-initialized student $\theta _ { S , \Delta } ^ { \mathrm { P T } }$ as:

$$
\pmb { \theta } _ { S , \Delta } ^ { \mathrm { P T } } = \pmb { \theta } _ { S , \mathrm { i n i t } } ^ { \mathrm { P T } } + \Delta _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { b a s e } } .
$$

The resulting model $\theta _ { S , \Delta } ^ { \mathrm { P T } }$ is intended to approximate the directly distilled post-trained student $\theta _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { { P T } } }$ in downstream performance, while avoiding the cost of an additional distillation run.

## 4 Experiments

Here we study the creation of post-trained model families using ADAPT. We show the benefits of ADAPT over strong baselines (Sections 4.2 and

![](images/0a3077083bedc7b635b1c74d6a93fdc56eb3a168f3c165866793645e5fa4baac.jpg)  
Figure 2: ADAPT improves generation performance over boomerang distillation. For generation tasks that are both in-domain (math reasoning) and out-of-domain (instruction-following and general knowledge) relative to our SFT dataset, SFT training is necessary to recover generation performance, especially for smaller models. For OOD tasks, two-phase distillation stabilizes performance compared to SFT-only training. We observe similar results for Qwen3-14B, Olmo-3-7B-Instruct, and Llama-3.1-8B-Instruct in Appendix M.

4.3). Then, we show ADAPT offers a better performance–compute trade-off than any single model (Section 4.4). Finally, we analyze the loss landscape to understand weight-delta initialization (Section 4.5).

## 4.1 Setup

In our experiments, we primarily use Qwen3-4B-Instruct-2507 (Yang et al., 2025) as the teacher model. We also study Qwen3-4B-Thinking-2507 and Qwen3-4B in Section 4.3. We initialize the student models with 2.7B inference-time parameters by removing every other layer from the teacher (excluding the final layer). We train on a total budget of 1B tokens. For more training details, see Appendices A and B. We also perform the same experiments with Qwen3-14B, Olmo (Olmo et al., 2025), and Llama (Grattafiori et al., 2024) families in Appendix M. We train on the deduplicated Pile (Gao et al., 2020) during the pre-training stage, and the math split of the Llama Nemotron Post-Training Dataset (Bercovich et al., 2025) during the SFT stage. We primarily evaluate our models on a set of five generation benchmarks measuring instructionfollowing and mathematical reasoning. We also apply our method to coding tasks and report the results in Appendix M.4. For more implementation details, see Appendix C. We include error bars of one standard deviation in the figures.

## 4.2 Two-Phase Distillation for Reasoning

We begin by comparing ADAPT (distilled) to boomerang distillation (BD), showing that the twophase distillation pipeline is key to model size interpolation for generation and reasoning tasks, yielding substantially stronger interpolation performance.

Setup. We compare five setups under an equal training budget, all evaluated on three in-domain (ID) math reasoning tasks and two out-of-domain (OOD) tasks testing instruction-following and general understanding. (1) ADAPT (distilled): Our proposed two-phase distillation setup. (2) BD (pretrain): Directly applying BD to the post-trained teacher. (3) BD (SFT): BD on post-trained model but with SFT instead of pre-training data in distillation stage. (4) Cross-patch (pre-train): Patching post-trained teacher directly onto boomerangdistilled base student to study whether post-trained teacher capabilities can be transferred through patching alone. (5) Base model BD: BD on base model as a reference.

Results. Figure 2 shows that ADAPT (distilled) achieves the strongest interpolation performance on both ID and OOD tasks by combining pre-training alignment with SFT capability recovery. It also achieves the best teacher–student alignment, as measured via last-layer activation cosine similarity (Appendix D.1). We observe similar trends for Qwen3-14B, Olmo, and Llama in Appendix M and report additional ablation results in Appendix E.

BD (pre-train) remains weak at small and medium model sizes, improving only as model size increases. BD (SFT) does not generalize as well to OOD and classification tasks (Appendix D.2), despite comparable performance to ADAPT (distilled) at small model sizes on ID tasks. Base model BD collapses completely, indicating that base model interpolation does not directly transfer to generation and reasoning tasks.

![](images/97503ee6000acb1711f3078fd18fd1254e0af56fffe85d445a3d62067dbaa542.jpg)  
Figure 3: ADAPT (weight-delta) enables reusable base model distillation across post-trained variants. ADAPT (weight-delta) matches ADAPT (distilled) on ID tasks and remains competitive on OOD tasks, showing that one base model distillation run can transfer to multiple post-trained variants without additional training. ADAPT (weight-delta) outperforms baseline cross-patch (2-phase) across all variants. Degradation in thinking-style settings is mainly due to malformed <think> formatting, rather than broad loss of generation quality.

Interestingly, despite performing significantly worse than other setups, cross-patch (pre-train) recovers nontrivial performance as more post-trained teacher layers are inserted, outperforming base model BD. This suggests that base and post-trained models remain partially compatible, motivating our weight-delta transfer approach in Section 4.3 and smooth weight interpolation experiments in Section 4.5.

## 4.3 Weight-Delta Initialization Enables Reusable Base Distillation

We demonstrate that ADAPT (weight-delta) can reuse a single base-model distillation run across multiple post-trained variants, recovering much of the interpolation behavior of directly distilled posttrained students without additional distillation.

Setup. We construct the interpolated models for Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, and Qwen3-4B using ADAPT (weight-delta). Specifically, we first perform two-phase distillation on the Qwen3-4B-Base student to obtain the base distillation delta, then add this delta to students initialized from each post-trained variant. We compare ADAPT (weight-delta) against two setups: (1) ADAPT (distilled), the higher-compute upper bound that runs a separate two-phase distillation for each post-trained student. (2) Cross-patch (2- phase), which patches post-trained teacher layers directly into the two-phase-distilled base student without applying the weight delta. Cross-patch (2-phase) serves as a natural, compute-matched baseline: like ADAPT (weight-delta), it produces a family of interpolated post-trained models from a single base-model distillation run. All methods use the same student architecture, layer-selection rule, patching order, and evaluation protocol.

Results. Figure 3 shows that ADAPT (weightdelta) achieves interpolation performance comparable to ADAPT (distilled) on ID tasks across the evaluated Qwen variants, while requiring no additional distillation on the post-trained teachers. On OOD tasks (IFEval and MMLU-Redux), ADAPT (weight-delta) remains competitive with ADAPT (distilled), with slight degradation concentrated in thinking-style models at smaller and intermediate sizes. Inspecting outputs shows this is largely a parsing artifact rather than a loss of generation quality: weight-delta models are initialized from a base distillation run without thinkingstyle <think> tags and can produce malformed delimiters, which IFEval penalizes since it parses only the response after </think>. Performance remains strong on MMLU-Redux, which does not depend on </think> parsing, confirming the issue is specific to strict thinking-tag parsing rather than a broad degradation in capability. ADAPT (weightdelta) also substantially outperforms cross-patch (2-phase) across post-trained variants, suggesting that the base model distillation delta transfers useful instruction-following capability and enables stronger interpolation than patching alone.

![](images/8d895c3d9b1ede1e03421c865253317e8d7398450c42a09cf44d9197c0b6a753.jpg)  
Figure 4: ADAPT enables adaptive model-size selection based on input difficulty for more compute-efficient inference. By producing a continuum of interpolated models with smoothly varying performance, ADAPT allows for more fine-grained matching between model capacity and task difficulty. This approach achieves a Pareto-optimal trade-off between average and worst-case accuracy and computational efficiency.

Results for Olmo and Llama models (Appendix M) suggest that ADAPT (weight-delta) generally falls between ADAPT (distilled) and crosspatch (2-phase), and is comparable to ADAPT (distilled) in many cases. It also achieves better interpolation performance than cross-patch (2-phase) in all ID settings, making it a cheaper but still strongperforming alternative to separately distilling each post-trained variant. We provide preliminary explanation for when weight-delta initialization works best in Appendix G.

We also report results for single-phase weightdelta initializations in Appendix F.1, where the transferred distillation delta is computed from a base student distilled on the pre-training phase or SFT phase only. This setting still demonstrates some cross-model distillation transfer, but generally underperforms the two-phase weight-delta initialization used in ADAPT. We experiment with continued training from the weight-delta initialized model, $\theta _ { S , \Delta } ^ { \mathrm { P T } }$ , but find that it does not improve interpolation performance (Appendix F.2). We include further ablation results on post-trained model deltas and SFT phase loss components in Appendices F.3 and F.4. Finally, we compare both variants of ADAPT to popular layer pruning methods in Appendix H.

## 4.4 Adaptive Model-Size Selection

In this section, we show that ADAPT achieves a Pareto-optimal trade-off between downstream performance and computational efficiency by dynamically choosing from the suite of interpolated models at inference time based on input difficulty. This improves compute utilization while preserving performance, satisfying a practical requirement for deployment.

Setup. We use the MATH dataset (Hendrycks et al., 2021b), which contains competition-style problems with varying difficulty levels. We estimate each problem’s difficulty using a single forward pass with the teacher model, Qwen3-4B-Instruct-2507 (Appendix I.1). For each interpolated model, we calculate its accuracy on each of the difficulty levels on a training split. Given a target accuracy threshold, we route questions of each difficulty level to the smallest model with accuracy above the threshold.

We consider both average accuracy and minimum accuracy. Average accuracy measures a model’s example-weighted mean accuracy across all the difficulty levels, allowing it to undershoot the threshold on harder levels. Minimum accuracy requires every difficulty level to individually exceed the threshold, yielding a more conservative policy. Sweeping over the threshold produces routing policies with different accuracy–compute tradeoffs, which we compare against fixed single-model baselines.

Results. Figure 4 shows that the adaptive modelsize selection approach outperforms the singlemodel baseline, tracing out a superior Pareto frontier across both average and minimum accuracy metrics for both ADAPT (distilled) and

![](images/b44df68b4f1f3c36cd18848bc3014bdf68432a5a724087a33f0af9ceb53afbf0.jpg)  
Figure 5: Smooth weight interpolation is preserved after distillation. Across post-trained variants, both teacher and distilled student models exhibit smooth interpolation paths between base and post-trained endpoints, suggesting that the relevant weight-space structure is preserved after distillation.

ADAPT (weight-delta). This improvement arises from a more fine-grained matching between model capability and question difficulty, enabled by the suite of models with smoothly increasing performance. Therefore, by assigning smaller models to easier questions and reserving larger models for the harder ones, ADAPT offers a superior performance– compute trade-off over using a single model.

We report the results for Olmo and Llama models in Appendix I.2. We also notice that smaller models tend to generate more tokens overall. We provide further discussion of this in Appendix I.3.

## 4.5 Size-Agnostic Smooth Weight Interpolation

The weight-delta initialization results in Section 4.3 suggest that base model distillation remains compatible with its post-trained variants. To better understand this, we study whether interpolation between base and post-trained models is preserved after distillation. We call this property size-agnostic smooth weight interpolation: the low-loss linear path between a base model and its post-trained variant is preserved in the corresponding student models after distillation. For models $\bar { \pmb { \theta } } ^ { ( a ) }$ and $\pmb \theta ^ { ( b ) }$ , we evaluate models $\pmb { \theta } ( \alpha ) = ( 1 - \alpha ) \pmb { \theta } ^ { ( a ) } + \alpha \pmb { \theta } ^ { ( b ) }$ where $\alpha \in [ 0 , 1 ]$

Figure 5 shows that interpolation paths between base and post-trained models do not exhibit sharp loss barriers for either teachers or students. The loss increases toward the post-trained endpoints because they are evaluated on the Pile, which is better matched to base models than post-trained variants. Importantly, students retain similar interpolation behavior after distillation, suggesting that distillation preserves much of the weightspace structure connecting base and post-trained models. We observe similar trends for generation accuracy and SFT loss in Appendix K. We further examine this geometry using a two-dimensional generation-accuracy AUC landscape over the plane spanned by the base, instruct, and weight-delta students. Here, AUC denotes the area under the generation-accuracy curve, measured between the smaller student and the corresponding teacher model. Figure 6 shows that the weight-delta initialized student lies in a high-performing region connected to both the base and distilled instruct students. These results suggest that distillation preserves smooth weight interpolation across model sizes, suggesting a possible connection to linear-mode connectivity (Garipov et al., 2018).

![](images/63bbfdf19cf3e9dd6d7f382871cd5b0327bcde4404f15e8b7795679ebbe87bd2.jpg)  
Figure 6: Weight-delta initialization produces a student in a connected region of high generation performance. We report AUC for generation accuracy between student and teacher models on the plane intersecting base, instruct, and weight-delta students. The weight-delta student has improved AUC compared to the base student and lies in a high-performance region connected to both the base and instruct students.

## 5 Conclusion

We introduce ADAPT, a framework for amortizing distillation across both the size and post-training-variant axes of a model family. ADAPT substantially outperforms boomerang distillation baselines on reasoning and instructionfollowing benchmarks under matched compute, and the resulting continuum of interpolated models enables adaptive model-size selection at inference time. Crucially, we find that smooth linear weight interpolation between base and post-trained models is preserved across model sizes after distillation, providing a plausible mechanism for the success of weight-arithmetic transfer in our setting. This connection between size interpolation and linear-mode connectivity points to a broader principle—that distillation can be reused across related models, not just sizes—and we view its further investigation as a promising direction.

## Limitations

ADAPT is effective across the model families and benchmarks we study, but several aspects warrant further investigation. We highlight open questions about the underlying mechanism, assumptions implicit in our procedure, and the empirical scope of our evaluation.

Interpolation dips at specific sizes. As also observed by Kangaslahti et al. (2026a), some interpolation curves show sharp drops at particular intermediate sizes. A plausible explanation is that the inserted teacher layers are locally misaligned with the surrounding student layers at those configurations, but the precise mechanism remains unclear. Characterizing and predicting these failure modes remain open directions. We also observe that the weight-delta transfer is not uniformly reliable across Olmo and Llama (Figures 35 and 37), especially in some OOD settings and intermediate sizes. This suggests that weight-delta effectiveness depends on model-family compatibility and post-training details, and should be validated beyond the settings studied here.

Model and data scope. Our experiments cover three model families (Qwen, Olmo, Llama) at the 4B-14B scale and use the math and coding splits of the Llama Nemotron Post-Training Dataset for the SFT phase. It remains to be verified whether ADAPT retains its size-amortization benefits at larger model scales and across broader instructiontuning mixtures (e.g., multilingual).

Weight-delta assumptions. Weight-delta initialization requires access to the base model used to produce each post-trained variant. For some openweight releases (and most closed-weight releases) the corresponding base checkpoint is unavailable or not exactly identifiable, restricting which model families ADAPT (weight-delta) can target without additional approximation.

Adaptive inference depends on a difficulty signal. The compute–accuracy gains from adaptive model-size selection (Section 4.4) rely on a perinput difficulty estimate. Our MATH experiments use teacher-predicted difficulty, which is cheap but task-specific; we have not characterized how well this approach generalizes to settings where difficulty is not well-defined or where a separate difficulty predictor would be expensive to obtain.

## Ethical Considerations

Our experiments use publicly available datasets and evaluation benchmarks, primarily focused on mathematical reasoning, instruction following, and general model evaluation. We also use publicly available model checkpoints as teachers and base models. We do not collect new human-subject data. However, because pre-training and post-training datasets may contain web-derived text, they may inherit biases, harmful content, or privacy concerns from their source corpora. Similarly, the models used in this work may inherit biases or unsafe behaviors from their original training pipelines or the datasets we use. Our work does not attempt to remove these risks, and downstream users should evaluate generated outputs for safety, bias, and reliability before deployment.

## Reproducibility Statement

We implement all our experiments using Py-Torch (Paszke et al., 2019) and HuggingFace transformers (Wolf et al., 2020) packages. We also experiment with public models available on HuggingFace Hub. We provide our code and models at https://github.com/dcml-lab/ADAPT.

## Acknowledgments

David Alvarez-Melis, Sara Kangaslahti, Jonathan Geuter, and Nihal V. Nayak acknowledge support from the National Science Foundation Graduate Research Fellowship (Grant No. DGE 2140743), the Kempner Institute, FAS Dean’s Competitive Fund for Promising Scholarship, Aramont Fellowship Fund, and the NSF AI-SDM Institute (Grant No. IIS-2229881). Francesco Locatello’s contribution to this research was funded in part by the Austrian Science Fund (FWF) 10.55776/COE12.

## References

AI-MO. 2025. Aimo validation aime dataset. https://huggingface.co/datasets/AI-MO/ aimo-validation-aime. Accessed: 2026-03-29.

Art of Problem Solving. n.d. Aops wiki: Competition ratings. https://artofproblemsolving. com/wiki/index.php/AoPS\_Wiki:Competition\_ ratings. Accessed: 2026-03-23.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models. Preprint, arXiv:2108.07732.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, and 114 others. 2025. Llamanemotron: Efficient reasoning models. Preprint, arXiv:2505.00949.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code.

Xiaodong Chen, Yuxuan Hu, Jing Zhang, Yanling Wang, Cuiping Li, and Hong Chen. 2025. Streamlining redundant layers to compress large language models. Preprint, arXiv:2403.19135.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. Preprint, arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling. Preprint, arXiv:2101.00027.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and

5 others. 2023. A framework for few-shot language model evaluation.

Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, and Andrew Gordon Wilson. 2018. Loss surfaces, mode connectivity, and fast ensembling of dnns. Preprint, arXiv:1802.10026.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile van Krieken, and Pasquale Minervini. 2025. Are we done with mmlu? Preprint, arXiv:2406.04127.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Preprint, arXiv:1503.02531.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, and 3 others. 2022. Training compute-optimal large language models. Preprint, arXiv:2203.15556.

Chip Huyen. 2022. Designing machine learning systems. O’Reilly Media, Inc.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic.

In The Eleventh International Conference on Learning Representations.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations.

Sara Kangaslahti, Nihal V. Nayak, Jonathan Geuter, Marco Fumero, Francesco Locatello, and David Alvarez-Melis. 2026a. Boomerang distillation enables zero-shot model size interpolation. In The Fourteenth International Conference on Learning Representations.

Sara Kangaslahti, Jonathan Geuter, Nihal V. Nayak, Marco Fumero, Francesco Locatello, and David Alvarez-Melis. 2026b. Understanding layer patching in model size interpolation. Preprint, arXiv:2607.08170.

Yeganeh Kordi, Nihal V. Nayak, Max Zuo, Ilana Nguyen, and Stephen H. Bach. 2025. Revisiting generalization across difficulty levels: It’s not so easy. Preprint, arXiv:2511.21692.

Suhas Kotha and Percy Liang. 2026. Replaying pre-training data improves fine-tuning. Preprint, arXiv:2603.04964.

Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. 2022. Matryoshka representation learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785– 794, Copenhagen, Denmark. Association for Computational Linguistics.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chat-GPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems.

Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, and 1 others. 2025. Rethinking machine unlearning for large language models. Nature Machine Intelligence, 7(2):181–194.

Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2025. ShortGPT: Layers in large language models are more redundant than you expect. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 20192–20204, Vienna, Austria. Association for Computational Linguistics.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. Preprint, arXiv:2501.19393.

Avanika Narayan, Dan Biderman, Sabri Eyuboglu, Avner May, Scott Linderman, James Zou, and Christopher Re. 2025. Cost-efficient collaboration between on-device and cloud language models. In Forty-second International Conference on Machine Learning.

Nihal V. Nayak, Paula Rodriguez-Diaz, Neha Hulkund, Sara Beery, and David Alvarez-Melis. 2026. A critical look at targeted instruction selection: Disentangling what matters (and what doesn’t). In Forty-third International Conference on Machine Learning.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, and 49 others. 2025. Olmo 3. Preprint, arXiv:2512.13961.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, and 2 others. 2019. Pytorch: An imperative style, high-performance deep learning library. Preprint, arXiv:1912.01703.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Adir Rahamim, Asaf Yehudai, Boaz Carmeli, Leshem Choshen, Yosi Mass, and Yonatan Belinkov. 2026. Will it merge? on the causes of model mergeability. Preprint, arXiv:2601.06672.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2020. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. Preprint, arXiv:1910.01108.

Safal Shrestha, Anubhav Shrestha, Aadim Nepal, Minwu Kim, and Keith Ross. 2026. On the limits of layer pruning for generative reasoning in llms. Preprint, arXiv:2602.01997.

Ali Taghibakhshi, Ruisi Cai, Saurav Muralidharan, Sharath Turuvekere Sreenivas, Ameya Sunil Mahabaleshwarkar, Marcin Chochowski, Akhiad Bercovich, Ran Zilberstein, Ran El-Yaniv, Yonatan Geifman, Daniel Korzekwa, Yoshi Suhara, Oluwatobi Olabiyi, Ashwath Aithal, Nima Tajbakhsh, and Pavlo Molchanov. 2026. Star elastic: Many-in-one reasoning LLMs with efficient budget control. In Forty-third International Conference on Machine Learning.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Abdul Waheed, Chancharik Mitra, Laurie Z. Wang, Deva Ramanan, and Bhiksha Raj. 2025. Less is more tokens: Efficient math reasoning via difficultyaware chain-of-thought distillation. Preprint, arXiv:2509.05226.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Huggingface’s transformers: State-of-the-art natural language processing. Preprint, arXiv:1910.03771.

Prateek Yadav, Tu Vu, Jonathan Lai, Alexandra Chronopoulou, Manaal Faruqui, Mohit Bansal, and Tsendsuren Munkhdalai. 2025. What matters for model merging at scale? Transactions on Machine Learning Research.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. Limo: Less is more for reasoning. Preprint, arXiv:2502.03387.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arXiv:2311.07911.

## A Training Implementation Details

We use a mixture of non-thinking and thinking models in this paper. Non-thinking models include Qwen3-4B-Instruct-2507, Qwen3-4B, Olmo-3-7B-Instruct, Llama-3.1-8B-Instruct, and Qwen3-14B. Thinking models include Qwen3-4B-Thinking-2507, Qwen3-4B, Olmo-3-7B-Think, and Qwen3- 14B. Note that Qwen3-4B and Qwen3-14B can toggle between thinking and non-thinking modes.

For model training, we use a total token budget of 1B for the Qwen 4B-sized models. For the larger models, we scale up the total budget proportionally, to 2B and 4B tokens respectively (Hoffmann et al., 2022). Across different configurations of the same model, we choose hyperparameters such that the number of tokens per update remains approximately constant between the pre-training phase and SFT phase. The hyperparameters reported in Table 3 correspond to using the deduplicated Pile (Gao et al., 2020) for pre-training and the Llama Nemotron Post-Training Dataset (Bercovich et al., 2025) for the SFT phase. For Nemotron, we primarily use the math split for all models (see Appendix M.4 for coding tasks). For non-thinking models, we use examples with non-thinking ground truths (i.e., responses not enclosed in <think> and </think>), while for thinking models we use examples with thinking ground truths (version 1.1).

Each 4B model training run takes approximately 12 hours on 4 NVIDIA H100 GPUs. The Olmo and Llama models are trained using 4 NVIDIA H200 GPUs, each taking approximately 24 hours. Training Qwen3-14B takes approximately 48 hours on 16 H200 GPUs. We will release the distilled models under an Apache 2.0 license.

## B Hyperparameters

We choose the majority of the hyperparameters (Table 1) following Kangaslahti et al. (2026a). We determine the KL and cosine loss weights such that cross entropy, KL, and cosine loss are approximately equal in magnitude early in training. M denotes the number of student layers. When doing two-phase distillation, we use a single LR scheduler across the two phases.

## C Evaluation Implementation Details

For generation, we use a custom evaluation pipeline for more fine-grained control over chat template and system prompt settings. Table 4 shows the system prompts used for each benchmark. We report the average accuracy on 5 benchmarks: AIME (AI-MO, 2025), GSM8K (Cobbe et al., 2021), IFEval (Zhou et al., 2023), MATH500 (Hendrycks et al., 2021b), MMLU-Redux (Gema et al., 2025). We consider AIME, GSM8K, and MATH500 to be in-domain (ID) math reasoning tasks and IFEval and MMLU-Redux to be out-of-domain (OOD) tasks. We use a separate set of ID benchmarks described in Appendix M.4 for coding tasks. For Section 4.4, we use the MATH dataset (Hendrycks et al., 2021b), which contains ground truth question difficulty labels. We elaborate on this further in Appendix I.1.

<table><tr><td>Hyperparameters</td><td>Values</td></tr><tr><td>Learning rate</td><td>3e-4</td></tr><tr><td>LR scheduler</td><td>cosine 0.01</td></tr><tr><td>Warmup ratio Optimizer</td><td>AdamW</td></tr><tr><td>Adam betas</td><td>(0.9, 0.95)</td></tr><tr><td>Adam epsilon Weight decay</td><td>1e-8 0.1</td></tr><tr><td>Max gradient norm</td><td></td></tr><tr><td></td><td>1.0</td></tr><tr><td>Mixed precision</td><td>bf16</td></tr><tr><td></td><td></td></tr><tr><td>KL weight  $\lambda _ { K L }$  Cosine weight  $\lambda _ { c o s }$ </td><td>0.1 10.0 / (M+1)</td></tr></table>

Table 1: Hyperparameters for distillation training.

For classification tasks, we use lm-evaluation-harness (Gao et al., 2023) and report the average accuracy on 10 benchmarks: ARC-easy and ARC-challenge (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021a), OpenBookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), RACE (Lai et al., 2017), RTE (Wang et al., 2019), and WinoGrande (Sakaguchi et al., 2020).

Table 2 shows the sampling parameters used for non-thinking and thinking models, which follow the recommended values by Yang et al. (2025).
<table><tr><td>Models</td><td>Non-thinking</td><td>Thinking</td></tr><tr><td>Max output len.</td><td>16384</td><td>32768</td></tr><tr><td>Temperature</td><td>0.7</td><td>0.6</td></tr><tr><td>Top p</td><td>0.8</td><td>0.95</td></tr><tr><td>Top k</td><td>20</td><td>20</td></tr><tr><td>Presence penalty</td><td>0.5</td><td>0.5</td></tr><tr><td>Min. tokens</td><td>32</td><td>32</td></tr></table>

Table 2: Sampling parameters for non-thinking and thinking models.

<table><tr><td rowspan="2">Model Type</td><td rowspan="2">Setup</td><td colspan="3">Pre-training Phase</td><td colspan="3">SFT Phase</td></tr><tr><td>Steps</td><td>Effective BS</td><td>Seq. len.</td><td>Steps</td><td>Effective BS</td><td>Seq. len.</td></tr><tr><td rowspan="3">4B (Non-think)</td><td>Pile only</td><td>480</td><td>2048</td><td>1024</td><td>一</td><td></td><td>一</td></tr><tr><td>Nemotron only</td><td>一</td><td></td><td></td><td>585</td><td>4096</td><td>1024</td></tr><tr><td>Pile+Nemotron</td><td>240</td><td>2048</td><td>1024</td><td>293</td><td>4096</td><td>1024</td></tr><tr><td rowspan="3">4B (Think)</td><td>Pile only</td><td>500</td><td>512</td><td>4096</td><td></td><td></td><td></td></tr><tr><td>Nemotron only</td><td>一</td><td>一</td><td></td><td>400</td><td>1024</td><td>4096</td></tr><tr><td>Pile+Nemotron</td><td>250</td><td>512</td><td>4096</td><td>200</td><td>1024</td><td>4096</td></tr><tr><td rowspan="3">7B/8B (Non-think)</td><td>Pile only</td><td>960</td><td>2048</td><td>1024</td><td></td><td></td><td></td></tr><tr><td>Nemotron only</td><td>一</td><td></td><td></td><td>1170</td><td>4096</td><td>1024</td></tr><tr><td>Pile+Nemotron</td><td>480</td><td>2048</td><td>1024</td><td>585</td><td>4096</td><td>1024</td></tr><tr><td rowspan="3">7B/8B (Think)</td><td>Pile only</td><td>1000</td><td>512</td><td>4096</td><td>一</td><td></td><td>1</td></tr><tr><td>Nemotron only</td><td>一</td><td></td><td></td><td>800</td><td>1024</td><td>4096</td></tr><tr><td>Pile+Nemotron</td><td>500</td><td>512</td><td>4096</td><td>400</td><td>1024</td><td>4096</td></tr><tr><td rowspan="3">14B</td><td>Pile only</td><td>875</td><td>1024</td><td>4096</td><td></td><td></td><td>一</td></tr><tr><td>Nemotron only</td><td>一</td><td></td><td></td><td>1400</td><td>1024</td><td>4096</td></tr><tr><td>Pile+Nemotron</td><td>438</td><td>1024</td><td>4096</td><td>700</td><td>1024</td><td>4096</td></tr></table>

Table 3: Training hyperparameters across model families. We report the number of steps, effective batch size, and sequence length for the pre-training and SFT phases across all distillation setups.
<table><tr><td>Benchmarks</td><td>System Prompts</td></tr><tr><td></td><td>AIME, GSM8K, MATH500, MATH Please reason step by step, and put your final answer within \boxed{}.</td></tr><tr><td>IFEval</td><td>None</td></tr><tr><td>MMLU-Redux</td><td>Please reason step by step, and select the answer from the given choices A, B, C, or D. Respond only with the letter of the correct answer. Put the index within \boxed{}.</td></tr></table>

Table 4: System prompts for generation tasks.

## D Additional Evaluation Results for ADAPT (Distilled)

We provide additional results for ADAPT (distilled). We first study final-layer cosine similarity to the teacher in Appendix D.1, then show results on classification tasks in Appendix D.2.

## D.1 Cosine Similarity Analysis

In this section, we report the final-layer activation cosine similarity between each interpolated model and the corresponding teacher for ADAPT (distilled), BD (pre-train), BD (SFT), and cross-patch (pre-train). We compute activations by averaging over all downstream tasks.

As shown in Figure 7, ADAPT (distilled) achieves the highest cosine similarity across nearly the entire interpolation curve. BD (SFT) also maintains strong alignment, but is consistently below ADAPT (distilled). Cross-patch (pre-train) has the weakest alignment, especially at smaller model sizes, indicating that patching alone does not sufficiently align the student with the post-trained teacher. These results provide additional motivation for the two-phase distillation approach.

![](images/684cb60f894ec8b045ce6bd0402337f6791cfb821225c332958e625154698260.jpg)  
Figure 7: Final layer activation cosine similarity between student and teacher models. The two-phase distillation setup yields the best student-teacher alignment, further motivating our approach.

![](images/2b530ca307d01e4fc09a4a5faf61337e82e4e4b1a87cba3251e1ac73d713ac7c.jpg)  
Figure 8: Classification accuracy is relatively insensitive to the distillation strategy. Across Qwen, Olmo, and Llama, most setups exhibit similar interpolation behavior on classification benchmarks. The main exception is BD (SFT), which underperforms across much of the interpolation curve, suggesting that task-specific distillation alone can degrade broader alignment with the teacher.

![](images/7236582bc2f8af0ea13343ca0089224eaf23f50153f5f5c75c7bc1b026947b52.jpg)  
Figure 9: Distilling with equal pre-training and SFT phase offers an appropriate tradeoff between ID and OOD performance. Increasing the SFT proportion improves ID accuracy but degrades OOD accuracy, while increasing the pre-training proportion has the opposite effect; the 50:50 split balances the two. The single-phase endpoints are consistently dominated by the two-phase settings.

## D.2 Classification Performance

Figure 8 shows that classification benchmarks are relatively insensitive to the distillation strategy. Most variants achieve similar interpolation performance, indicating that classification tasks do not fully expose the alignment challenges that arise in post-trained generation. The main exception is the BD (SFT) setup, which underperforms across much of the interpolation curve for all three model families, suggesting that task-specific distillation alone can degrade broader alignment with the teacher.

## E Additional Ablations for ADAPT (Distilled)

## E.1 Pre-training vs. SFT Phase Ratios

Here we study how sensitive ADAPT (distilled) is to varying training budget between pre-training and SFT phase. All settings use the same total budget.

In Figure 9, we observe a consistent trade-off across the two-phase settings: increasing the SFT proportion improves ID accuracy but degrades OOD accuracy, while increasing the pre-training proportion has the opposite effect. The 50:50 split sits between these extremes, achieving strong ID accuracy without sacrificing OOD performance. We note that the three two-phase ratios perform comparably at larger model sizes, indicating that ADAPT is not overly sensitive to the exact split.

Finally, both single-phase endpoints are dominated by the two-phase settings, consistent with our finding in Section 4.2.

## E.2 Distillation with Cosine Loss Only

In Figure 10, we show that distilling with cosine loss alone can yield performance comparable to the full objective combining cross-entropy, KL, and cosine losses. This suggests that intermediate representation alignment may be a key driver of interpolation behavior, and points to an orthogonal direction for improving distillation efficiency by simplifying the training objective.

![](images/429d6eba9ed4ff927a4b8b3c6ee08fe4b94cd58618fb04ba4977f0ad6407620b.jpg)  
Figure 10: Distilling with cosine loss only yields interpolation performance comparable to the full objective. We compare training with cosine alignment loss only to training with all of the loss terms in Equation 1 and find that cosine only training has similar performance, indicating that the efficiency of interpolation training can potentially be improved by using only alignment loss.

![](images/4cbb1e7b2a33853e6e0653024a57c74eda9283a98837095ef4593e3e3f00fa6e.jpg)  
Figure 11: Pre-training weight-delta initialization for post-trained Qwen models.

## F Additional Ablations for ADAPT (Weight-Delta)

## F.1 Weight-Delta Transfer from Pre-training-only and SFT-only Deltas

We evaluate whether weight-delta initialization works when we calculate the weight delta from base model distilled with pre-training or SFT phase only. This setting tests whether the transferable component of distillation can be obtained without the two-phase distillation setup.

Setup. We test whether weight-delta transfer works with single-phase deltas, i.e., deltas computed from a base student distilled on only one type of data (pre-training or SFT). We then add the respective deltas to initialized post-trained students across model families. For comparison, we evaluate these single-phase weight-delta models against two references: the directly distilled BD (pre-train) and BD (SFT) students (which use no delta transfer), and the corresponding cross-patch baselines. We report the results for Qwen (Figures 11 and 12), Olmo (Figure 13), and Llama (Figure 14) families.

Results. Across model families, pre-training weight-delta initialization performs comparably to, and in some cases better than, BD (pre-train), indicating that the alignment learned during basemodel pre-training distillation transfers across posttrained variants. The SFT-only delta transfers less smoothly. Weight-delta initialization tracks

![](images/ba7d25ee181bd523218b4577b808f7da4ef01335455ad22f6001ee806c64fb67.jpg)

Figure 12: SFT weight-delta initialization for post-trained Qwen models.  
![](images/f186d568806c6f8607f3432656d04f9e4a8d454def7ab38c4112f8e1dd959f27.jpg)  
Figure 13: Pre-training weight-delta initialization performance for post-trained Olmo models.

![](images/7ec037a8a45e812995e0df3317441322732076fd1e5d74124481f69e0e5eda44.jpg)  
Figure 14: Pre-training weight-delta initialization performance for post-trained Llama models.

BD (SFT) closely for the instruct variant, but meaningfully underperforms on the other three variants. The weight-delta setup even performs worse than the cross-patch (SFT) baseline for Qwen3-4B, indicating that SFT-only distillation on the base model is not enough to facilitate effective weight-delta transfer across post-trained models.

Furthermore, single-phase weight-delta initialization setups remain weaker than the two-phase methods. Both ADAPT (weight-delta) and ADAPT (distilled) achieve stronger interpolation performance, reflecting that both phases are necessary to fully recover downstream generation and reasoning capabilities.

![](images/4fb34956407dc8b9a4a52085c3fc6889c42c1b82badb421a4bb7bee9f4e8dace.jpg)  
Figure 15: Continued training from weight-delta initialization does not improve interpolation performance. We initialize the instruct student using a pre-training, SFT, or two-phase weight delta, computed from base distillation on pre-training data, SFT data, or both respectively. We then continue training for 0.5B or 1B tokens. Continued training does not meaningfully improve over weight-delta initialization.

## F.2 Continued Training from Weight-Delta Initialization

We investigate whether weight-delta initialization can serve as a stronger initialization for additional distillation. Specifically, we ask whether continued training from a weight-delta-initialized student can outperform ADAPT (weight-delta).

Setup. We initialize the student by adding one of three weight deltas to it, each obtained by distilling the base model on different data: pre-training data only, SFT data only, or both phases (two-phase). Starting from each initialization, we continue training for either 0.5B or 1B tokens using the same distillation setup described in Appendix B. We also sweep the learning rates and find that 3e-4, the value used in Appendix B, performs the best.

Results. Figure 15 shows that continued training from weight-delta initialization does not meaningfully improve interpolation performance beyond ADAPT (weight-delta). Training for 0.5B or 1B tokens converges to similar interpolation curves, regardless of whether the initialization comes from pre-training (left panel), SFT (middle panel), or two-phase (right panel) weight delta. This suggests that continued distillation largely washes out the difference between the different initializations rather than improving the final interpolated family.

Overall, these results indicate that the main benefit of weight-delta initialization comes from its zero-training transferability, not from providing a better starting point for further training.

## F.3 Weight-Delta Initialization with Post-trained Model Deltas

In this section, we test whether distillation done directly on post-trained students (instead of the base student) can be transferred to other post-trained variants via weight-delta initialization.

Setup. We obtain the post-trained model weight deltas by directly performing two-phase distillation on Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507, and subtracting their corresponding initialized (pre-distillation) students. We then initialize the other post-trained models using the weight deltas from the instruct and thinking models separately following Section 3.2.2.

Results. Figures 16 and 17 show that distillation effects can be transferred onto other posttrained variants using post-trained weight deltas. ADAPT (weight-delta) using instruct and thinking model weight deltas both substantially outperform the cross-patch (2-phase) baseline. Compared against the higher-compute upper bound ADAPT (distilled), the instruct model delta performs competitively on ID tasks, while suffering slight degradation on OOD tasks similar to the base model delta. The thinking model delta matches the performance of ADAPT (distilled) on both ID and OOD tasks. Overall, these results indicate that the transferable distillation delta is not specific to the base model, deltas computed from post-trained students transfer to other variants equally effectively.

![](images/d29f0d8aeec34ec801378ad14d00795b46091a2ead801d4fa0527d09df5fa628.jpg)

Figure 16: Results for ADAPT (weight-delta) using delta from Qwen3-4B-Instruct-2507.  
![](images/d6bf2e743d8f824df5a48ad7c29736bd1d8fadbdbb1b76a1c55636d1615ade0a.jpg)  
Figure 17: Results for ADAPT (weight-delta) using delta from Qwen3-4B-Thinking-2507.

## F.4 Ablating SFT Phase Loss Components

While the combination of cross entropy, KL, and cosine loss terms is optimal for base model distillation under the boomerang distillation setup (Kangaslahti et al., 2026a), aligning the student too closely with the less capable base teacher on reasoning tasks during the SFT phase may hinder its learning. In this section, we explore whether relaxing the base student’s alignment with the teacher can improve the weight-delta transfer performance.

Setup. We keep the pre-training phase of base model distillation unchanged and ablate the loss terms used in the SFT phase, always retaining cross-entropy while removing the KL term, the cosine term, or both. For each ablated objective, we compute the resulting base model weight delta and use it to initialize the four Qwen post-trained variants following Section 3.2.2. We compare each ablation against the higher-compute upper bound ADAPT (distilled) and the cross-patch (2-phase) baseline.

Results. Removing the KL term leaves interpolation behavior largely unchanged (Figure 18), indicating that it contributes little during the SFT phase. Removing the cosine term has a substantially larger effect (Figure 19): ID performance improves across all four variants, matching ADAPT (distilled), while OOD performance degrades overall. This split is size-dependent — the ID gains come almost entirely from smaller interpolated models, where the full objective left the weight-delta students near total collapse until roughly 3.3B parameters, whereas OOD performance falls off at intermediate and larger sizes. Removing both terms behaves similarly but with even weaker OOD performance (Figure 20). Therefore, the cosine term in SFT phase trades ID performance for OOD generalization, and the choice of whether to retain it depends on the specific use cases.

Qwen3-4B (Non-Thinking)  
![](images/1391b2163f84f678109faf08d55e2d155fbd1d837b0880d665ded81141e2288d.jpg)

![](images/73aab6c988243f10be0c87149c82c97cd7ae57f37b6f1e40561ef5aecbb43d16.jpg)

![](images/fc520ebcc9c3f29cb97d492fa151083d5e705c1fa37028b4cef452867ac1ddd5.jpg)

![](images/81a7ed4cc75117416cc540498a906e430800da07857a4317ee079ced16cab32b.jpg)

![](images/00eca5e8e43daf345bef731e39089a4ccd11cdc5525439df8456e9e44c52c6bc.jpg)

![](images/28da717c9dfc72dff9d6ce632510d48224768faa10b9bc6c01fe6de91e866e1c.jpg)

![](images/c045981020c175a62f119ceed47d8a22032522750f320eacde774378025ed628.jpg)  
No. of Parameters (Billions)

![](images/a617ca570dd44ee18f0a08d43456a5e9f9a252d18d1cbc611e0b5642b09cc215.jpg)

<table><tr><td></td><td>Post-trained student</td><td></td><td>Base student</td><td></td><td>Weight-delta student</td><td></td><td>Interpolated models</td></tr><tr><td>★</td><td>Teacher</td><td></td><td>ADAPT (distilled)</td><td></td><td>ADAPT (weight-delta)</td><td></td><td>Cross-patch (2-phase)</td></tr></table>

Figure 18: Results for ablating KL loss term in SFT phase.

![](images/85c87a5494c4c4479b4ca744e3eb1b4aed53395174068834565a893fa13e80dc.jpg)

![](images/094ca9dd3e92b89d0b5503425664658257f310b268b355194c3aef52c07d21a9.jpg)

![](images/f5ddc64b24da9de65de648f211af7d53050d0de07ed5519c34b57837fb4904f5.jpg)

![](images/562712f18ac116edb3fdaaff6ce36636f7f8313b8d1f0e57df0417521cd90d85.jpg)

![](images/00c98f4b16c1ad4894b49394d5b6537e534300728e0f755013960ca0d694891f.jpg)

![](images/9e0073982a04f7700a3c2b3f98e857f5039c148fd3d6333a7bdca2099a4bce29.jpg)

![](images/aa9434968029652006fa85dc84fcb9cf865475757a40f66418834c6b5fba7436.jpg)  
No. of Parameters (Billions)

![](images/55dc824408b15ded23bce2332a64bf83922cc2d350ce6153ffe386a4f779430f.jpg)

<table><tr><td></td><td>Post-trained student</td><td></td><td>Base student</td><td></td><td>Weight-delta student</td><td></td><td>Interpolated models</td></tr><tr><td>★</td><td>Teacher</td><td></td><td>ADAPT (distilled)</td><td></td><td>ADAPT (weight-delta)</td><td></td><td>Cross-patch (2-phase)</td></tr></table>

Figure 19: Results for ablating cosine loss term in SFT phase.

![](images/29f2a09a6ac43b9a80ac95f0ecbf7b02c9348a06d423b1669a28bc0626d7a40d.jpg)

![](images/3e4e274617caa6e6f6aa19d267a862aff0c6e3590da628bbea8cd8d7f2aa4594.jpg)

![](images/2157caf56ba22f4d3ba6c5972dee6025c3b8eced9e2997ad6708a2964c32ae5c.jpg)

![](images/0e7f416d44307278d872472c809be49fe264e6c1a6591b9bea6ba89fc892344c.jpg)

![](images/9de8ae7978c8fc4d56afacb09b34df637dea996905f1b41ccc3e0acd03ef043c.jpg)

![](images/54cae8049bdf5a7615189837a86878d556943e13fc1c148beae885512bb240d5.jpg)

![](images/1ff12356255f172f44506b6434cea99478cf5075fbcfb1673a9f358c7c60afd5.jpg)  
No. of Parameters (Billions)

![](images/9cc538a4316cc8d75ddd0e26c6ebc316e4fe95ed4680411560326bf29e1e4440.jpg)

<table><tr><td>★</td><td>Post-trained student Teacher</td><td></td><td>Base student ADAPT (distilled)</td><td></td><td>Weight-delta student ADAPT (weight-delta)</td><td></td><td>Interpolated models Cross-patch (2-phase)</td></tr></table>

Figure 20: Results for ablating both KL and cosine loss terms in SFT phase (cross-entropy loss only).

## G Weight-Delta Alignment Experiments

Results in the preceding appendices and Appendix M show that ADAPT (weight-delta) is more successful for some models and setups than others, matching the ADAPT (distilled) performance at its best. However, understanding when exactly weight-delta transfer succeeds remains an important open question. Notably, this is a recognized and challenging problem even in the closely related setting of model merging and task vectors, where identifying the factors that govern whether weight-arithmetic combination succeeds is an active area of research (Rahamim et al., 2026). Here, we present preliminary results that yield some useful signals.

The task-vector view (Ilharco et al., 2023) treats a capability acquired through finetuning as a shift in weight space—in the language of this paper, a weight delta. A natural predictor of transfer success is therefore the alignment between the base distillation delta $( \pmb { \theta } _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { b a s e } } - \pmb { \theta } _ { S , \mathrm { i n i t } } ^ { \mathrm { b a s e } } )$ and each variant’s own distillation delta $( \pmb { \theta } _ { S , \mathrm { d i s t i l l } } ^ { \mathrm { P T } } - \pmb { \theta } _ { S , \mathrm { i n i t } } ^ { \mathrm { P T } } )$ . Table 5 reports the cosine similarity and norm ratio between these deltas.

Alignment tracks transfer reliability. Instruct and hybrid variants align most closely (cosine of 0.73-0.75 for Qwen and Olmo), thinking variants less so (0.57-0.64), and Llama least of all (0.30), mirroring the order in which we observe transfer to be reliable, partially reliable, and unreliable. Norm ratios stay near 1, so the deltas differ in direction rather than magnitude. Although this is a correlational analysis, it suggests directional alignment as a candidate diagnostic for when weight-delta transfer will hold. We leave further investigation to future work.

<table><tr><td>Variant</td><td>Cosine</td><td>Norm ratio</td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td>0.73</td><td>1.00</td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>0.57</td><td>0.92</td></tr><tr><td>Qwen3-4B</td><td>0.75</td><td>1.00</td></tr><tr><td>Qwen3-14B</td><td>0.75</td><td>1.00</td></tr><tr><td>Olmo-3-7B-Instruct</td><td>0.73</td><td>1.02</td></tr><tr><td>Olmo-3-7B-Think</td><td>0.64</td><td>0.94</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.30</td><td>0.95</td></tr></table>

Table 5: Cosine similarity and norm ratio between base and post-trained deltas for each variant.

![](images/3af674edec3a8a3cce56c5b8a610d6cbbe3b88294d6cfa42cf5594571cb466e3.jpg)  
Figure 21: ADAPT significantly outperforms layer pruning baselines on generation tasks. We compare ADAPT (distilled) and ADAPT (weight-delta) to two popular layer pruning methods for generation tasks, ShortGPT (Men et al., 2025) and LLM-Streamline (Chen et al., 2025), and show that our method performs substantially better across all sizes.

## H Comparison to Layer Pruning Methods

We compare ADAPT against layer pruning methods on generation benchmarks to show that it produces substantially stronger intermediate models.

Setup. We consider two popular layer pruning methods: ShortGPT (Men et al., 2025) and LLM-Streamline (Chen et al., 2025). ShortGPT prunes the least influential layers, as determined by the block influence score (see Appendix H.1). LLM-Streamline identifies the block of layers with the lowest block influence score and replaces it with a lightweight network (see Appendix H.2).

Results. Figure 21 shows that across all model sizes, ADAPT (distilled) and ADAPT (weightdelta) models consistently outperform ShortGPT and LLM-Streamline, which struggle to recover meaningful generation capability, especially at smaller sizes.

## H.1 ShortGPT

For ShortGPT (Men et al., 2025), we first calculate the Block Influence score (BI), which is the cosine distance between the input and output activations. A higher BI score indicates higher importance of a specific layer. The BI score of the $\ell ^ { \mathrm { { t h } } }$ layer can be calculated as follows:

$$
B I _ { \ell } = 1 - \mathbb { E } _ { X , t } \left[ \frac { \pmb { x } _ { t } ^ { ( \ell ) } \cdot \pmb { x } _ { t } ^ { ( \ell + 1 ) } } { \| \pmb { x } _ { t } ^ { ( \ell ) } \| \| \pmb { x } _ { t } ^ { ( \ell + 1 ) } \| } \right]
$$

where $\mathbf { \Delta } _ { \mathbf { \boldsymbol { x } } _ { t } ^ { ( \ell ) } }$ is the $t ^ { \mathrm { { t h } } }$ row of hidden states of the $\ell ^ { \mathrm { { t h } } }$ layer. When pruning, we sequentially remove layers with the lowest BI score. We compute BI using a held-out set of 128 calibration samples from Nemotron.

## H.2 LLM-Streamline

For LLM-Streamline (Chen et al., 2025), we identify the block of n layers $( \ell ^ { * } , \ell ^ { * } + n )$ with the lowest BI score (using the same calibration set from Nemotron as ShortGPT) and replace it with a single lightweight network. We then train the replacement network to imitate the pruned block using mean squared error (MSE):

$$
\operatorname* { m i n } _ { h } \mathbb { E } _ { ( \boldsymbol { x } ^ { ( \ell ^ { * } ) } , \boldsymbol { x } ^ { ( \ell ^ { * } + n ) } ) \in \mathcal { D } } \mathrm { M S E } \left( h ( \boldsymbol { x } _ { i } ^ { ( \ell ^ { * } ) } ) , \boldsymbol { x } _ { i } ^ { ( \ell ^ { * } + n ) } \right)
$$

where h denotes the lightweight network, denotes the recorded hidden states of samples, and $\pmb { x } ^ { ( i ) }$ is the hidden state of the $i ^ { \mathrm { { t h } } }$ layer.

Following Chen et al. (2025), we further posttrain the replaced layer on language modeling loss in order to have a fairer comparison with our method.

In our implementation, we use a transformer block from the original Qwen3-4B-Instruct-2507 model as the lightweight layer and train it on the same data and token budget as ADAPT (distilled). We allocate 80% of the total training budget to lightweight network training and the remaining 20% to post-training. The MSE stage runs for 192 Pile steps and 234 Nemotron steps, while the LMloss stage runs for 48 Pile steps and 59 Nemotron steps. All stages use sequence length of 1024, with effective batch size of 2048 for Pile and 4096 for Nemotron. We use the training hyperparameters shown in Table 6. We perform evaluation following the same sampling strategy as discussed in $\mathsf { A p } \cdot$ pendix C, using the non-thinking sampling parameters.

<table><tr><td>Hyperparameters</td><td>Values</td></tr><tr><td>LR (MSE)</td><td>2e-4</td></tr><tr><td>LR (LM-loss)</td><td>5e-5</td></tr><tr><td>LR scheduler</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>0.01</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Adam betas</td><td>(0.9, 0.999)</td></tr><tr><td>Adam epsilon</td><td>1e-8</td></tr><tr><td>Weight decay</td><td>1e-3</td></tr><tr><td>Max gradient norm</td><td>5.0</td></tr><tr><td>Mixed precision</td><td>bf16</td></tr></table>

Table 6: Hyperparameters for LLM-Streamline training.

## I Additional Results for Adaptive Model-Size Selection

## I.1 Input Difficulty Classification

As described in Section 4.4, we use the MATH dataset (Hendrycks et al., 2021b), which contains competition-style math problems with ground truth difficulty labels according to the Art of Problem Solving (AoPS) competition ratings (Art of Problem Solving, n.d.). For our task, we sample 2 separate subsets of 1050 problems from the original dataset, each with 210 problems for each difficulty level. We use one as the training set to determine the optimal model routing policy, and the other as a test set to measure the efficiency gains of our method.

To create the model routing policy, we require a lightweight and reliable method for estimating input difficulty. We test both the student and teacher models as classifiers by prompting them to assign a difficulty rating to each question. Specifically, each input question is paired with a prompt describing the AoPS difficulty scale (Figure 22), and the model is asked to output a rating from 1 to 10. We merge all levels above 5 into level 5, since we know a priori that the MATH dataset contains problems of at most level 5 difficulty. To obtain predictions efficiently, we perform a single forward pass and extract the predicted class directly from the output logits.

To evaluate classification performance, we use

![](images/42dafd0a11e83ac20c0aac6d2f8762ec7cce07d90385c24bd8dcb38d2350bdf3.jpg)  
Figure 22: Prompt for difficulty classification. Adapted from Waheed et al. (2025).

![](images/8e0a805f0598846fb045babf1d1e65f04406af8470ff0a9af4bf9409bb308536.jpg)  
Figure 23: Interpolation performance of Qwen3-4B-Instruct-2507 for different difficulty levels. We find that the teacher-defined difficulty score produces similar interpolation curves to the ground truth difficulty scores, but student-defined difficulty is unreliable. This shows that the teacher-defined labels can act as a reasonable approximation for groundtruth difficulties.

<table><tr><td>Levels</td><td>Teacher</td><td>Student</td></tr><tr><td>1</td><td>184</td><td>1035</td></tr><tr><td>2</td><td>282</td><td>9</td></tr><tr><td>3</td><td>149</td><td>2</td></tr><tr><td>4</td><td>195</td><td>0</td></tr><tr><td>5</td><td>240</td><td>4</td></tr></table>

Table 7: Breakdown of input-difficulty classification results from teacher and student models. The student is unreliable for difficulty classification, as it chooses the lowest difficulty level for the vast majority of examples.

the mean absolute error (MAE):

$$
\mathrm { M A E } _ { \mathcal { D } _ { \mathrm { t r a i n } } } ( \hat { \pmb { y } } , \pmb { y } ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { t r a i n } } | } \sum _ { i \in \mathcal { D } _ { \mathrm { t r a i n } } } | \hat { y } _ { i } - y _ { i } |
$$

where $\mathcal { D } _ { \mathrm { t r a i n } }$ denotes the training set, $\hat { y } _ { i }$ is the predicted difficulty for the $i ^ { \mathrm { t h } }$ problem, and $y _ { i }$ is the corresponding ground-truth difficulty.

On our training set, the teacher achieves an MAE of 0.970, while the student has an MAE of 1.995. This indicates that the teacher’s predictions are typically within one difficulty level of the ground truth, whereas the student’s predictions deviate by approximately two levels on average. This suggests that the teacher provides substantially more accurate estimates for adaptive model selection. The class breakdown of the teacher and student predictions is shown in Table 7.

While the MATH dataset provides humanannotated difficulty labels, prior work has shown that human difficulty does not always align with model-perceived difficulty (Kordi et al., 2025). Figure 23 verifies that teacher-predicted labels produce well-separated and consistently ordered performance curves, closely matching the trends from ground-truth labels, whereas student-predicted labels are noisy and do not yield clear separation. This further supports using the teacher for difficulty estimation in adaptive model selection.

We omit the cost of difficulty classification from the results in Section 4.4, as it requires only a single teacher forward pass and is negligible relative to the average response length of 3770.5 tokens.

## I.2 Adaptive Model-Size Selection Results for Olmo and Llama

In Figures 24 and 25, we report the results for the adaptive model size selection experiments for Olmo-3-7B-Instruct and Llama-3.1-8B-Instruct. Note that we use Qwen3-4B-Instruct-2507 to provide the difficulty labels (same as in Section 4.4) as it is smaller than both the teacher models of Olmo and Llama, and therefore cheaper.

## I.3 Discussion of Total Token Count

While smaller models offer more favorable FLOPs per token performance, we empirically observe that they often generate more tokens overall for the same set of questions. This is because smaller models are typically less capable, leaving more problems unsolved. On these questions, the model keeps generating tokens until it hits the max\_tokens limit, whereas on questions it solves correctly, it completes them using far fewer tokens. We believe that with further training, this issue can be mitigated, but this is beyond the scope of this paper and we leave it for future work.

![](images/cf01b7f98519128e0cf1cf907dc65a91cfbe0091b5a4d21936bb3c365f767807.jpg)  
Figure 24: Adaptive model size selection results for Olmo-3-7B-Instruct.

![](images/9362aa6dcb08a6e16503a1d52b179060b0c89db164936df8ef121f94b3306d0c.jpg)  
Figure 25: Adaptive model size selection results for Llama-3.1-8B-Instruct.

![](images/f4b0879dc235e709f0fb10e46dcfed384b01c421d0da1aef2e07ba760fa3248f.jpg)  
Figure 26: Comparison of interpolation performance for different patching orders. We test patching from the front, patching from the back, and KLPatch (Kangaslahti et al., 2026b). We find that KLPatch can further improve performance, indicating that more sophisticated patching strategies are orthogonal to our approach and can be used in conjunction with our student models.

## J Patching Order Experiment

In Figure 26, we show the full interpolation curves for all three model families when testing front-toback and back-to-front patching orders. We also test KLPatch, which is a recently proposed iterative greedy KL divergence-based algorithm for selecting the patching order (Kangaslahti et al., 2026b). At each size, KLPatch patches each student layer individually and selects the layer with minimum KL divergence to the teacher. We use a calibration set of 64 examples from the Nemotron non-thinking set in order to compute the KL divergence. These plots provide a more detailed view of how patching order affects interpolation behavior across model sizes.

We observe that the effect of patching order varies substantially across models. For Qwen3- 4B-Instruct-2507 and Olmo-3-7B-Instruct models, KLPatch improves performance over patching from the front and patching from the back. However, for Llama-3.1-8B-Instruct, patching from the front yields the best performance. This suggests that the optimal patching depends on how different layers contribute to downstream generation performance in each model and architecture. In our experiments, we use back to front for Qwen3-4B-Instruct-2507 and front to back for Olmo-3-7B-Instruct and Llama-3.1-8B-Instruct. As KLPatch improves performance for small Qwen and Olmo models, this indicates that more sophisticated patching methods can be used in conjunction with ADAPT to improve the performance of post-trained model families.

![](images/15b8c3643e01c10797a65b2e111428544dbd3c87db39aeec98de6bb01aed09b9.jpg)  
Figure 27: Smooth weight interpolation results for generation accuracy. Base and post-trained students display smooth interpolation behavior in downstream generation accuracy.

## K Additional Smooth Weight Interpolation Results

In this section, we report the smooth weight interpolation results for generation accuracy (Figure 27) and loss on SFT data (Figures 28 and 29). We observe the same smooth weight interpolation behavior between the base and post-trained students across all three setups. The difference between Figures 28, 29, and Figure 5 from Section 4.5 can be attributed to the calibration data we used to calculate the loss terms, as different models (base, instruct, thinking) naturally have lower loss on different types of calibration data. Therefore, the results provide additional evidence that distilled students mirror teacher-level connectivity in the weight space.

## L Training Dynamics

In Figure 30, we study how the interpolation performance behaves during the two-phase training. During the pre-training phase, the model converges to a lower interpolation performance curve very similar to the BD (pre-train) curve in Figure 2. The SFT phase on Nemotron data initially produces improved student model performance but has a dip in interpolation performance at higher model sizes, then converges to a solution with both improved student model performance and smooth interpolation behavior. Taken together, this indicates that both phases converge to distinct solutions with general purpose versus in-domain alignment to the teacher model.

![](images/96e81c97305a2174c2e532022945144c04931857938e0c7e9c30b462dbcdfafc.jpg)  
Figure 28: Smooth weight interpolation results for loss on non-thinking Nemotron data. Base and posttrained students display smooth interpolation behavior in loss evaluated on non-thinking Nemotron data.

![](images/10460c28754f317974b08a55a0ba90ad8b3f867813e63e285cbcae9d0f309a5b.jpg)  
Figure 29: Smooth weight interpolation results for loss on thinking Nemotron data. Base and post-trained students display smooth interpolation behavior in loss evaluated on thinking Nemotron data.

![](images/4818cccb561ed85bf2fd66455ebe4dd6e4ee9bf6e5716d7ce59fe046550eda9b.jpg)  
Figure 30: Training dynamics study of two-phase distillation. We find that two-phase distillation converges to two distinct interpolation curves during the pre-training and SFT phases, which allows the model to retain general-purpose and in-domain performance.

## M ADAPT for Additional Models and Tasks

In this section, we show the additional results for Qwen, Olmo and Llama models. We also show the performance of ADAPT on coding tasks.

## M.1 Results for Qwen3-14B

In Figures 31, 32, and 33, we show the twophase distillation and weight-delta initialization results for Qwen3-14B with both thinking and nonthinking modes.

## M.2 Results for Olmo

In Figures 34 and 35, we show the two-phase distillation and weight-delta initialization results for Olmo-3-7B-Instruct and Olmo-3-7B-Think.

## M.3 Results for Llama

In Figures 36 and 37, we show the two-phase distillation and weight-delta initialization results for Llama-3.1-8B-Instruct.

## M.4 Results for Coding Tasks

In this section, we report the performance of ADAPT for coding tasks and show that it works beyond mathematical reasoning. We replace the math data with coding data for the SFT phase, sampled from the coding split of Llama Nemotron Post-Training Dataset (Bercovich et al., 2025). For ID tasks, we use MBPP+ (Austin et al., 2021), HumanEval+ (Chen et al., 2021), and Live-CodeBench (Jain et al., 2025). MBPP+ and HumanEval+ are augmented versions created with EvalPlus (Liu et al., 2023). We use the same benchmarks for OOD tasks (IFEval and MMLU-Redux).

In Figures 38 and 39, we show the two-phase distillation and weight-delta initialization results for Qwen3-4B under thinking mode. The results show that, similar to math tasks, two-phase distillation continues to outperform BD (pre-train) and BD (SFT) setups in both ID and OOD tasks. ADAPT (weight-delta) remains as a tradeoff between the higher-compute upper bound ADAPT (distilled) and compute-matched baseline cross-patch (2-phase). It is cheaper than ADAPT (distilled) while performing better than cross-patch (2-phase).

![](images/f07c2de39c9353485d23bdc800a1ed888559ff7ff63d6d0ce85b0f7efdeea38d.jpg)  
Figure 31: ADAPT (distilled) results for Qwen3-14B (Thinking). Two-phase distillation produces the highest and most stable performance over both ID and OOD tasks.

![](images/8fc38fd799eca17330794c31430022aad2d5215a44bcda776519605fedbbc587.jpg)  
Figure 32: ADAPT (distilled) results for Qwen3-14B (Non-Thinking). Two-phase distillation produces the highest and most stable performance over both ID and OOD tasks.

![](images/e3e09e387b2812018b9a99358f9fa171288a155231730dc18eac13ba62a3b7c6.jpg)  
Figure 33: ADAPT (weight-delta) results for Qwen3-14B with both thinking and non-thinking modes. ADAPT (weight-delta) remains competitive with ADAPT (distilled) across both ID and OOD settings, while consistently outperforming cross-patch (2-phase).

![](images/5da0430d682aa2e5a08df5efd4417e1357ac96aa3816430bbd8ef33fb345f949.jpg)  
Figure 34: ADAPT (distilled) results for Olmo models. Two-phase distillation produces the highest and most stable performance over both ID and OOD tasks.

![](images/09a57c7084c023a5843d6d45896271fb2fb6d0f8b30003a30cffc507f9106470.jpg)  
Figure 35: ADAPT (weight-delta) results for Olmo models. ADAPT (weight-delta) remains competitive with ADAPT (distilled) across both ID and OOD settings, while consistently outperforming cross-patch (2-phase) (with the exception of OOD performance for Olmo-3-7B-Think).

![](images/3d57a4f804aa1d963b05e641e4896f1e72dab3f0d5bfb82d75815d0be736086d.jpg)  
Figure 36: ADAPT (distilled) results for Llama models. Two-phase distillation produces the highest and most stable performance over both ID and OOD tasks.

![](images/d63b8f8cdf1a3d4aee57e1638d19542eeaae855261b8603fe9ba5560b0159f32.jpg)

Figure 37: ADAPT (weight-delta) results for Llama models. ADAPT (weight-delta) remains comparable with ADAPT (distilled) across both ID and OOD settings.  
![](images/86a0294bcb2ebbe204c192e7875f6b45d95ae509b8acfac44588b9834670e73f.jpg)  
Figure 38: ADAPT (distilled) results for Qwen3-4B on coding tasks. Similar to the math setup, two-phase distillation produces the highest and most stable performance over both ID and OOD tasks. The evaluation is done on thinking mode only.

![](images/ef867a6930df10fe279e07c760e78c1f4184b857bbdb50d66c0ff876f8d0c091.jpg)  
Figure 39: ADAPT (weight-delta) results for Qwen3-4B on coding tasks. ADAPT (weight-delta) maintains a trade-off between the higher-compute upper bound ADAPT (distilled) and compute-matched baseline cross-patch (2- phase). It is cheaper than ADAPT (distilled) while performing better than cross-patch (2-phase).

## N Individual Benchmark Results

## N.1 Two-phase Post-trained Distillation Results

In Figure 40, we show the interpolation performance of Qwen3-4B-Instruct-2507 on individual benchmarks.

## N.2 Weight-delta Initialization Results

In Figure 41, we show the results from the weightdelta initialization experiments for Qwen model family on individual benchmarks.

## O Use of Large Language Models

We utilized generative AI tools for code completion, debugging, and minor grammatical corrections in the manuscript. The authors carried out all the substantive research contributions, analyses, and interpretations.

![](images/728033d22096e4732ac5b14d6ac87da7785adab759d424ac814c850c7a6c5d5b.jpg)  
Figure 40: Interpolation results for Qwen3-4B-Instruct-2507 on individual benchmarks.

![](images/a7f974e5a3dcc35723ea816a2ce2e6a86f06d875bb2be185245c871cee9318b0.jpg)  
Figure 41: Weight-delta initialization experiment results for Qwen 4B model family on individual benchmarks.