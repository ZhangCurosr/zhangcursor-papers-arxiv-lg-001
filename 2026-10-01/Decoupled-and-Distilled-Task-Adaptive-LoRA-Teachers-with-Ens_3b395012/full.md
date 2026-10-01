# Decoupled and Distilled: Task-Adaptive LoRA-Teachers with Ensemble Knowledge Transfer for Few-Shot Class-Incremental Learning

Hongwei Zhao<sup>a</sup>, Rui Liu<sup>a</sup>, Yansong Liu<sup>a</sup>, Zhiyuan Zou<sup>a</sup>, Yong Chen<sup>b,∗</sup>

<sup>a</sup>School of Computer Science and Engineering, Beihang University, Beijing, China <sup>b</sup>School of Computer Science, Beijing University of Posts and Telecommunications, Beijing, China

## Abstract

Few-Shot Class-Incremental Learning (FSCIL) addresses the fundamental challenge of learning new classes from very limited samples while retaining knowledge of previously learned ones. Although recent parameter-eficient fine-tuning (PEFT) techniques with pre-trained models have shown promise in class-incremental learning, they remain constrained in few-shot settings. In particular, constrained PEFT methods typically impose strict gradient-based constraints to mitigate forgetting; however, maintaining plasticity under such constraints requires abundant training data—a condition unavailable in FS-CIL—thereby intensifying the plasticity–stability trade-of. Moreover, multiexpert paradigms that address this plasticity bottleneck introduce substantial inference-time costs through dynamic module selection or runtime module generation, limiting practical deployment. To address these issues, we propose TALON (Task-Adaptive LORA-Teachers with ENsemble Knowledge Transfer), a novel framework that achieves both high accuracy and inference eficiency. TALON introduces task-adaptive LoRA-Teacher modules that are dynamically allocated for each incremental task, enabling efective

task-specific representation learning under severe data constraints. To reduce inference overhead, we design an Ensemble Knowledge Transfer (EKT) mechanism that distills knowledge from multiple frozen LoRA-Teachers into a unified LoRA-Student. Furthermore, a semantic-guided distillation strategy adaptively weights teacher contributions based on feature-space similarity, mitigating both catastrophic forgetting and overfitting. Across three class-order runs, TALON achieves comparable or better mean average accuracy across all four FSCIL benchmarks, obtaining 86.68±1.22% on CUB200, 90.39±0.27% on CIFAR100, 78.38±0.94% on ImageNet-R, and 96.34±0.33% on miniImageNet. TALON maintains clear mean improvements on CI-FAR100, ImageNet-R, and miniImageNet, while obtaining performance comparable to SEC-prompt on the more class-order-sensitive CUB200 benchmark. TALON also uses up to 33× fewer deployment parameters and reduces average inference time per task to 26.7 s, corresponding to a 41.70% reduction relative to ASP (45.8 s). Code is available at: https://github. com/hongwei-zhao/NEUCOM-TALON-main.

Keywords: Few-Shot Class-Incremental Learning, Task-Adaptive LoRA, Ensemble Knowledge Distillation, Semantic-Guided Similarity

## 1. Introduction

In open-world environments, data arrives in a streaming fashion with continually emerging new classes: a scenario known as Class-Incremental Learning (CIL). Conventional machine learning models trained sequentially on such data sufer from catastrophic forgetting [1, 2], where learning new knowledge disrupts previously acquired representations and leads to severe performance degradation.

Real-world applications such as autonomous driving, medical imaging, and robotics [3] often encounter this setting, where novel object categories emerge with only a handful of labeled examples. This gives rise to Few-Shot Class-Incremental Learning (FSCIL) [4], a particularly challenging paradigm in which models are first trained on base classes with abundant supervision, and then must incrementally adapt to novel classes from severely limited exemplars—typically under an N-way K-shot protocol [5]. This setting exacerbates both catastrophic forgetting of old classes and overfitting to the few new samples, resulting in a pronounced plasticity–stability dilemma [6].

Pre-trained models (PTMs), with their strong generalization abilities [7], learned from large-scale datasets [8], provide a strong foundation for incremental learning. However, fully fine-tuning PTMs risks compromising their generalization. Recent advances in CIL address this issue by freezing the PTM backbone and introducing parameter-eficient fine-tuning (PEFT) modules [9], such as prompts [10, 11] or low-rank adaptation (LoRA) [12, 13, 14, 15]. These methods enable task-specific adaptation with minimal trainable parameters, thereby reducing forgetting while preserving generalization.

Despite these advances, existing FSCIL methods still face two fundamental limitations:

Challenge I: Insuficient feature learning under data scarcity. Constrained PEFT-based methods for CIL [10, 13, 14, 5] impose gradientbased constraints (e.g., orthogonal subspaces, shared gradient directions) that require suficient data to estimate reliably. Under FSCIL’s extreme data scarcity, these constraints become unreliable, leading to underfitting due to insuficient plasticity or overfitting due to poor generalization, depending on the method and dataset (e.g., InfLoRA in Fig. 1; see the supplementary material for detailed analysis).

Challenge II: Inference ineficiency of multi-expert continual learning paradigms. To overcome the plasticity bottleneck (Challenge I), a powerful strategy is to use Multi-Expert Learning (e.g., mixture-of-experts or per-sample adapters). However, this introduces substantial inference-time costs:

• Dynamic module selection: Query-based methods [10, 11] require expensive similarity computations across module pools, creating scalability bottlenecks as tasks accumulate.

• Runtime module generation: Input-conditional approaches [5, 16] synthesize task-specific modules for each input, incurring per-sample computational overhead.

These challenges highlight a central research question: How can we jointly optimize the plasticity–stability trade-of under extreme data scarcity while maintaining inference-time eficiency?

To address this, we propose TALON (Task-Adaptive LORA-Teachers with ENsemble Knowledge Transfer), a novel inference-eficient framework for FSCIL that tackles both challenges through a principled multi-teacher distillation strategy. TALON allocates a dedicated LoRA-Teacher module for each incremental task. These teachers are integrated into Transformer layers, enabling eficient representation learning from limited data while maintaining full parameter isolation across tasks—thereby achieving plasticity without sacrificing stability. To reduce inference overhead, we introduce Ensemble Knowledge Transfer (EKT), which distills collective knowledge from all frozen LoRA-Teachers into a unified LoRA-Student. This design consolidates ensemble-level knowledge into a single model, eliminating dynamic module selection or runtime module generation. Furthermore, we propose a semantic-guided distillation strategy that adaptively weights teacher contributions using feature-space similarity, ensuring coherent and efective knowledge transfer while mitigating both forgetting and overfitting.

PEFT-Based Inference Time  
![](images/8d7098d666a4acd481384d2cbe45a0c853e1cc20cde12a64592a436f3c2c1977.jpg)

![](images/01c2007588260f39181bea1d451afefb1bda757bd07439d7aebb725752512612.jpg)  
Figure 1: Performance and inference time comparison among multi-expert-based and LoRA-based methods. TALON consolidates multi-expert knowledge into a single model via distillation, eliminating runtime module selection/generation overhead while achieving superior accuracy (inference time averaged per task on CIFAR100).

Our main contributions are listed as follows:

• We identify that the bottleneck of the existing PEFT-FSCIL lies in the ‘shared-parameter’ assumption. We propose a Decoupled Learning paradigm, which dynamically allocates an independent LoRA-Teacher for each incremental task to achieve efective task-specific adaptation under few-shot constraints. Only the current teacher is trained, while previous ones remain frozen, ensuring strong plasticity and stability through full parameter isolation.

• We propose Ensemble Knowledge Transfer (EKT), a multi-teacher distillation mechanism that consolidates knowledge from all LoRA-Teachers into a single student, removing the inference-time overhead of dynamic module selection or runtime module generation while retaining ensemble-level performance.

• We design a semantic-guided distillation strategy that optimally integrates knowledge from the teacher ensemble based on feature-space similarity, mitigating catastrophic forgetting and overfitting.

• Extensive experiments on multiple FSCIL benchmarks show that TALON achieves state-of-the-art performance with superior eficiency. We also demonstrate architectural flexibility through TALON-MLP, a variant exploring alternative teacher integration strategies while maintaining competitive results.

## 2. Related Work

## 2.1. KD-based CIL

Class-Incremental Learning (CIL) aims to continuously acquire knowledge of new classes while retaining previously learned representations [17, 18, 19, 20]. Knowledge Distillation (KD) [21] is widely used in this context, typically adopting a self-distillation framework where the model trained on prior tasks acts as a teacher, transferring knowledge to the current student model to mitigate forgetting. Following prior work [22], 3 main KD paradigms have been explored in CIL:

1. KD as regularization [17, 23, 24]: LwF [17] pioneered this direction by training the current model to mimic the outputs of the previous model on new data. PRD [24] further refined this by continually distilling feature representations using the KL-divergence [25]. Probability dampening and cascaded classifier design have also been combined with KD to balance retention and adaptation [26].

2. KD with data replay [27, 28, 29]: iCaRL [27] introduced exemplar replay alongside KD to alleviate forgetting, while PODNet [30] extended this with spatially aware feature distillation. K3D jointly optimizes synthetic exemplars and knowledge fusion distillation to preserve class decision boundaries [31].

3. KD with feature replay [32, 33, 34]: For rehearsal-free scenarios, methods such as PASS [32] use class prototypes augmented with Gaussian noise to prevent bias, while Fusion [34] addresses prototype drift by modeling features with Gaussian or variational distributions.

Recent exemplar-free work also combines logit- and feature-level distillation with prototype-based classifier calibration to address forgetting and incremental class bias [35].

## 2.2. PEFT-based CIL

Recent progress in Parameter-Eficient Fine-Tuning (PEFT) [36] has shown strong potential for CIL by enabling adaptation with minimal parameters while preserving the generalization of pre-trained models (PTMs). Several representative works include:

1. Prompt-based methods: L2P [10] introduced a dynamic prompt pool to guide PTMs in sequential task learning. DualPrompt [37] separated general prompts (task-invariant) from expert prompts (task-specific). S-Prompts [38] further enhanced flexibility by storing domain-specific prompts and selecting them at inference via K-Nearest Neighbors (KNN). CODA [11] proposed decomposed attention-based prompting for rehearsal-free continual learning.

2. LoRA-based methods: LAE [13] employed a learning-accumulationintegration strategy to maintain knowledge across tasks. InfLoRA [14] reduced task interference by defining task-specific subspaces with injected lowrank parameters. SD-LoRA [15] decoupled gradient direction and magnitude, preserving early-task directions while adapting to new ones, albeit at the expense of reduced plasticity.

## 2.3. PEFT-based FSCIL

Few-Shot Class-Incremental Learning (FSCIL) focuses on learning novel classes from extremely limited examples while retaining past knowledge [4,

39]. Early approaches [40, 41] relied on shallow networks and full backbone fine-tuning, which often caused overfitting and poor generalization [42]. More recent work leverages PTMs with PEFT:

1. ASP [5] freezes the PTM backbone and uses a prompt encoder to generate task-specific and task-invariant prompts.

2. SEC-prompt [16] designs hierarchical attention queries to generate discriminative and non-discriminative prompts, improving few-shot representation learning.

3. CPE-CLIP [3] integrates learnable prompts into vision–language encoders, combining prompt regularization with CLIP’s cross-modal alignment to mitigate forgetting and enhance transferability.

Recent continual-learning studies have investigated prompting and analytic learning, slow–fast collaborative learning, prototype-enhanced composition, uncertainty-guided expert selection, and domain-specific subspace or prototypical learning for SAR target recognition [43, 44, 45, 46, 47, 48]. These directions provide complementary perspectives on preserving and organizing knowledge across incremental tasks, while TALON focuses on few-shot image class expansion through task-specific LoRA adaptation and consolidation into a single Student.

## 2.4. Our Approach

Building on recent PEFT methods [14, 15, 5, 16], we adopt a PTM backbone with LoRA adaptation [12]. Unlike existing FSCIL approaches that rely on prompt encoders, we propose stage-wise dynamic LoRA allocation, assigning a dedicated LoRA-Teacher to each incremental task for more effective few-shot learning. To consolidate knowledge, we introduce Ensemble Knowledge Transfer (EKT), which distills information from all frozen LoRA-Teachers into a unified student model. A semantic-guided weighting strategy further ensures coherent integration by adapting teacher contributions based on feature-space similarity. This design enables eficient knowledge transfer, robust generalization, and the elimination of inference-time prompt generation.

## 3. Preliminaries

## 3.1. Problem Formulation

In Few-Shot Class-Incremental Learning (FSCIL), we consider a sequence of tasks denoted by $\mathcal { D } = \{ \mathcal { D } _ { 0 } , \cdot \cdot \cdot , \mathcal { D } _ { T } \}$ , where the t-th task $\mathcal { D } _ { t } = \{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ contains $n _ { t }$ instances. Here, $\mathbf { x } _ { i } \in \mathcal { X } _ { t }$ represents an input sample from domain $\mathcal { X } _ { t }$ , and $\mathbf { y } _ { i } \in \mathcal { V } _ { t }$ is corresponding label from the label space $\mathcal { V } _ { t }$ . Crucially, the label spaces are disjoint across tasks $( \mathcal { V } _ { t } \cap \mathcal { V } _ { t ^ { \prime } } = \emptyset \ \mathrm { f o r } \ t \neq t ^ { \prime } )$ . The first task $\mathcal { D } _ { 0 }$ provides ample training data and is referred to as the base task. In contrast, each subsequent incremental task $\mathcal { D } _ { t } ~ ( t \geq 1 )$ comprises a limited number of labeled samples, typically formulated as N-way K-shot classification tasks, where N denotes the number of novel classes introduced and K represents the number of examples per class.

Following the rehearsal-free setting [10, 11], the model only has access to the current task’s data during training. Model performance is evaluated on all previously seen classes $\mathcal { Y } _ { \leq t } = \mathcal { Y } _ { 0 } \cup \cdot \cdot \cdot \cup \mathcal { Y } _ { t }$ after learning each incremental task. Formally, our objective is to learn a model $f _ { \Theta } ( \mathbf { x } ) = \mathbf { W } _ { \mathrm { c l s } } ^ { \top } \phi ( \mathbf { x } )$ that minimizes the empirical risk over the current training set:

$$
\operatorname* { m i n } _ { \Theta } \mathbb { E } _ { ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \in \mathcal { D } _ { t } } \left[ \mathrm { L } \left( \mathrm { f } _ { \Theta } ( \mathbf { x } _ { i } ) , \mathbf { y } _ { i } \right) \right] ,\tag{1}
$$

where $\phi ( \cdot ) : \mathbb { R } ^ { D }  \mathbb { R } ^ { d }$ denotes the feature extractor, $\mathbf { W } _ { \mathrm { c l s } }$ denotes a generic linear classifier in the FSCIL problem formulation, and $\operatorname { L } ( \cdot , \cdot )$ denotes a loss function measuring the discrepancy between the prediction and the groundtruth label. TALON’s training-only shared linear head $\mathbf { W } _ { \mathrm { h e a d } }$ and the deployed prototype bank $\mathcal { P } _ { t }$ used for classification are introduced separately below.

## 3.2. Low-Rank Adaptation

LoRA [12] was initially introduced for fine-tuning pre-trained models. It achieves fine-tuning of models by adding low-rank matrices $\Delta \mathbf { W }$ to the weight matrices $\mathbf { W } \in \mathbf { R } ^ { d _ { i n } \times d _ { o u t } }$ of the pre-trained model. The update rule of the weight matrix W can be formulated as:

$$
\mathbf { W } + \Delta \mathbf { W } = \mathbf { W } + \mathbf { U } \mathbf { V } ,\tag{2}
$$

where $\mathbf { U } \in \mathbf { \Delta } \mathbf { R } ^ { d _ { i n } \times r } , \mathbf { V } \in \mathbf { \Delta } \mathbf { R } ^ { r \times d _ { o u t } }$ and the rank $r ~ \ll$ min $( d _ { i n } , d _ { o u t } )$ . LoRA significantly reduces the number of trainable parameters, thus improving eficiency and reducing costs in the fine-tuning phase.

## 3.3. Knowledge Distillation in Incremental Learning

Knowledge Distillation (KD) is a widely adopted technique that transfers knowledge from a well-trained teacher model to a student model following the teacher-student paradigm. In the context of incremental learning, KD serves as a crucial regularization mechanism to mitigate catastrophic forgetting by preserving knowledge acquired from previous tasks [22].

In incremental learning scenarios, KD typically employs a self-distillation framework where the teacher and student models share identical architectures. Specifically, the model trained on previous tasks acts as the teacher, transferring its learned representations to the current model (student), thereby maintaining the memory of previously encountered tasks. The knowledge distillation loss $L _ { \mathrm { K D } }$ in incremental learning is formulated as:

$$
L _ { \mathrm { K D } } = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { t } } \left[ \mathrm { K D } \left( \phi _ { t - 1 } ( \mathbf { x } ) \parallel \phi _ { t } ( \mathbf { x } ) \right) \right] ,\tag{3}
$$

where $\phi _ { t - 1 }$ and $\phi _ { t }$ represent the feature extractors from the old task model and the current task model, respectively, and KD(· ∥ ·) denotes the distillation loss function. This regularization term ensures that the current model $\phi _ { t }$ maintains compatibility with previously learned representations while effectively adapting to new tasks.

## 4. The Proposed Method

## 4.1. Overall Framework

The overall framework of TALON is illustrated in Fig. 2. It operates in a two-stage paradigm for each incremental task to efectively learn from limited samples while ensuring high inference eficiency.

Stage 1: Task-Adaptive LoRA-Teacher. Given the limited training samples in few-shot scenarios and strong generalization of pre-trained models, we keep the pre-trained weights W fixed throughout incremental training.

To ensure model plasticity and maintain within-task prediction performance at each incremental stage [49], we adopt a task-isolated LoRA training where a new LoRA-Teacher is dynamically introduced for each task while previous teachers remain frozen. Specifically, a new LoRA-Teacher is dynamically introduced for each incremental task, while the parameters of previously added LoRA-Teachers remain frozen, shown in Fig. 2(a). By combining the current task’s LoRA-Teacher with pre-trained weights, fine-tuning can be achieved with fewer parameters, enabling eficient capture of task-specific features.

Stage 2: Ensemble Knowledge Transfer (EKT). To reduce inference overhead, we propose an Ensemble Knowledge Transfer (EKT) framework that distills the collective knowledge from multiple LoRA-Teachers into a unified LoRA-Student model, shown in Fig. 2(b). This approach enables efective knowledge integration without requiring dynamic module selection or runtime module generation, while maintaining single-model inference efficiency. Unlike existing methods that necessitate complex routing mechanisms [10] or runtime module generation [5, 16] during inference, our EKT framework consolidates all teacher knowledge into a single student model, thereby ensuring computational eficiency without compromising learning effectiveness.

![](images/65a8639fe26d7c5e6efb78a2e8ca13466c21fe97d81cb577228b3deef782a452.jpg)  
Figure 2: Overview of the TALON framework. (a) Task-Adaptive LoRA-Teacher: For each incremental task t, a new LoRA-Teacher $\mathbf { T } _ { t }$ is trained to capture task-specific features while keeping previous LoRA-Teachers frozen. (b) Ensemble Knowledge Transfer: Multiple LoRA-Teachers are distilled into a unified LoRA-Student $\mathbf { S } _ { t }$ using semantic-guided adaptive coeficients. (c) Semantic-Guided Distillation Coeficients $\alpha _ { k } \colon$ prototypes are extracted from the k-th LoRA-Teacher for task t data using Eq. (14), and the final coeficient $\alpha _ { k }$ is obtained by normalizing raw task-relevance scores in the shared feature space of LoRA-Teacher $\mathbf { T } _ { k }$ according to Eq. (13).

During Stage 2 EKT, sequential evaluation of the accumulated frozen LoRA-Teachers incurs O(t) Teacher-side forward computation and a linearly growing retained Teacher parameter state. Both costs are training-only: before deployment, all LoRA-Teachers and their Teacher-side prototypes are discarded, leaving only the consolidated LoRA-Student and the accumulated Student prototype bank for inference.

To alleviate catastrophic forgetting and overfitting during distillation, we introduce the Semantic-Guided Distillation Coeficient, shown in Fig. 2(c). This strategy adaptively computes coeficients according to the similarity between current and previous category prototypes within the shared feature space. In this way, the student model emphasizes knowledge distilled from teachers most relevant to the current task, thereby improving learning eficiency while reducing conflicting supervision.

For prototype-based inference, new-class prototypes are extracted with the current LoRA-Student while historical prototypes remain fixed; the resulting prototype–feature drift and its decision-level efect are analyzed in Section 5.7.

## 4.2. Task-Adaptive LoRA-Teacher

To incorporate task-specific information into LoRA modules while mitigating forgetting and minimizing inference overhead, prior studies [14, 15] have explored gradient-based strategies such as subspace orthogonality and shared gradient directions. However, these approaches rely on suficient training data to extract meaningful gradient information, which is often infeasible in few-shot scenarios. In contrast, TALON introduces Task-Adaptive LoRA-Teachers that learn directly from limited samples without additional constraints, enabling more efective task-specific adaptation.

We integrate the LoRA-Teacher into the Vision Transformer (ViT) architecture [50]. In ViT, an input image is first partitioned into fixed-size patches, which are linearly projected and combined with positional embeddings before being processed by the Transformer encoder. The encoder consists of multihead self-attention (MHA) layers and multilayer perceptron (MLP) blocks.

To adaptively capture task-specific features in incremental learning, we dynamically introduce a LoRA-Teacher module at each incremental stage. This module can be attached as a parallel branch to the MLP block (MLP-Teacher), or to the query $\left( \mathbf { W } _ { q } \right)$ , key $\left( \mathbf { W } _ { k } \right)$ , and value $\left( \mathbf { W } _ { v } \right)$ projections within the MHA mechanism (QKV-Teacher). When applied jointly to $\mathbf { W } _ { q }$ and $\mathbf { W } _ { v } ,$ we refer to the configuration as the QV-Teacher.

The modified forward computations for the MLP and MHA layers are formulated as follows:

$$
{ \bf h } ^ { \prime } = { \bf e } + \mathrm { M L P } ( { \bf e } ) + { \bf T } _ { t } ^ { M L P } \left( { \bf e } \right) ,\tag{4}
$$

$$
{ \bf h } ^ { \prime } = \mathrm { A t t n } \left( { \bf h } _ { Q } + { \bf T } _ { t } ^ { Q } \left( { \bf e } \right) , { \bf h } _ { K } + { \bf T } _ { t } ^ { K } \left( { \bf e } \right) , { \bf h } _ { V } + { \bf T } _ { t } ^ { V } \left( { \bf e } \right) \right) ,\tag{5}
$$

where e and h are the input and output of the original module, and $\mathbf { T } _ { t } ^ { Q }$ is the Q-Teacher for task t, and Attn is defined as:

$$
{ \mathrm { A t t n } } \left( \mathbf { Q } , \mathbf { K } , \mathbf { V } \right) = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } ,\tag{6}
$$

and the multi-head mechanism is omitted for conciseness. All LoRA-Teachers have the same architecture but with diferent weights. The LoRA-Teacher consists of a low-rank matrix $\mathbf { U } \in \mathbf { R } ^ { d _ { i n } \times r }$ and $\mathbf { V } \in \mathbf { R } ^ { r \times d _ { o u t } }$ , which is formulated as:

$$
\mathbf { T } _ { t } \left( \mathbf { e } \right) = \mathbf { e U } _ { t } \mathbf { V } _ { t } .\tag{7}
$$

Following LoRA [12], we initialize V as a zero matrix and U using Kaiming initialization [51]. Each introduced LoRA-Teacher is initialized in this manner. As illustrated in Fig. 2(a), for the t-th incremental task, a new LoRA-Teacher is learned, while all previously learned LoRA-Teachers remain frozen.

## 4.3. Ensemble Knowledge Transfer

After learning each incremental stage’s LoRA-Teacher, we can follow previous strategies to complete inference, such as using routing mechanisms [49] to select the most suitable LoRA-Teacher or obtaining the final inference model through Exponential Moving Average (EMA) [13]. However, both strategies have their respective drawbacks: routing mechanisms increase inference overhead while classification performance is also afected by routing accuracy [49]. The EMA approach requires manually setting hyperparameters to control the EMA decay rate, which is not suitable for FSCIL scenarios.

Model ensemble is a powerful technique to improve model accuracy [52], yet few works explore its potential in FSCIL. In TALON, we propose an Ensemble Knowledge Transfer (EKT) mechanism that consolidates knowledge from multiple LoRA-Teachers into a unified LoRA-Student through complementary soft logits distillation and hard label distillation strategies.

Soft Logits Distillation. As described in Section 2.1, three main KD paradigms exist in CIL: KD as regularization, KD with data replay, and KD with feature replay. Given FSCIL’s limited sample constraints, data replay and feature replay become impractical due to insuficient historical data availability. Soft logits convey the subtle diference between two samples and, therefore, can help the student model generalize better than directly learning from hard labels. In this approach, we distill the logits of the student model by minimizing the Kullback-Leibler (KL) divergence [25] between the student and teacher logits distributions.

Specifically, for a given input x, the soft logits distillation loss is formulated as:

$$
L _ { \mathrm { S D } } = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { t } } \left[ \operatorname { L } _ { \mathrm { K L } } \left( \sum _ { i = 0 } ^ { t } \alpha _ { i } T _ { i } \parallel S _ { t } \right) \right] ,\tag{8}
$$

where $\mathrm { L } _ { \mathrm { K L } }$ denotes the Kullback-Leibler divergence loss, $\alpha _ { i }$ is the semanticguided distillation coeficient described in Section 4.4, and $T _ { i } , \ S _ { t }$ denote the logits of the i-th LoRA-Teacher and t-stage LoRA-Student, respectively:

$$
T _ { i } = \mathbf { W } _ { \mathrm { h e a d } } ^ { \top } \phi \left( \mathbf { x } ; \mathrm { T } _ { i } \right) , \quad S _ { t } = \mathbf { W } _ { \mathrm { h e a d } } ^ { \top } \phi \left( \mathbf { x } ; \mathrm { S } _ { t } \right) ,\tag{9}
$$

where $\phi ( \cdot ; \mathrm { T } _ { i } )$ and $\phi ( \cdot ; \mathrm { S } _ { t } )$ denote the feature extractors of the i-th LoRA-Teacher and the stage-t LoRA-Student, respectively. At session t, the shared linear head $\mathbf { W } _ { \mathrm { h e a d } } \in \mathbb { R } ^ { d \times | \mathcal { V } _ { \leq t } | }$ contains one column for each observed class. It maps all Teacher and Student features into the same logit coordinates during EKT. The weighted combination $\textstyle \sum _ { i = 0 } ^ { t } \alpha _ { i } T _ { i }$ is computed in logit space before applying softmax. During EKT, $\mathbf { W } _ { \mathrm { h e a d } }$ and all accumulated Teachers are frozen; only the low-rank parameters of $\mathbf { S } _ { t }$ are optimized. The shared linear head is used only during training and is not used for final inference. At session $t ,$ the LoRA-Student is initialized from $\mathbf { S } _ { t - 1 }$ , which carries forward the representation consolidated over previous sessions. The relevance-weighted Teacher logits over $\mathcal { \mathrm { { y } } } _ { \leq t }$ then regularize the low-rank Student update on $\mathcal { D } _ { t }$ while the hard-label term incorporates the current classes.

Hard Label Distillation. To ensure the student model maintains strong discriminative capability on the primary classification task, we employ hard label distillation as complementary supervision. This strategy provides explicit ground-truth guidance that balances the implicit knowledge transfer from soft features, preventing the student from over-relying on teacher representations at the expense of task-specific performance. The hard label distillation loss is formulated as:

$$
{ \cal L } _ { \mathrm { H D } } = \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { t } } \left[ \mathrm { L } _ { \mathrm { C E } } \left( \mathbf { W } _ { \mathrm { h e a d } } ^ { \top } \boldsymbol { \phi } ( \mathbf { x } ; \mathrm { S } _ { t } ) , \mathbf { y } \right) \right] ,\tag{10}
$$

where $\operatorname { L _ { C E } } ( \cdot , \cdot )$ denotes the cross-entropy loss between the student’s predictions and ground-truth labels.

Unified Knowledge Transfer. The total EKT loss combines both distillation strategies with adaptive weighting:

$$
{ \cal L } _ { \mathrm { E K T } } = { \cal L } _ { \mathrm { S D } } + { \cal L } _ { \mathrm { H D } } .\tag{11}
$$

This unified approach allows the student model to harness nuanced feature representations from the teacher ensemble while preserving prior knowledge and adapting to new tasks in FSCIL.

## 4.4. Semantic-Guided Distillation Coeficient

Since diferent LoRA-Teachers exhibit varying relevance to the current task, their contributions to the student model are not uniformly beneficial. In fact, an inefective teacher may even hinder the student’s learning. To selectively transfer useful knowledge, inspired by EASE [53], we introduce a semantic-guided distillation coeficient that adaptively weights each teacher according to its task relevance. This mechanism enables the student to emphasize the most pertinent knowledge while suppressing less informative or conflicting signals. Concretely, for each teacher $\mathrm { T } _ { k } .$ , we compute prototypes of its own classes and those of the current task within $\mathrm { T } _ { k } \mathrm { \ ' } _ { \mathrm { s } }$ feature space, and measure their raw semantic relevance score:

$$
r _ { k } = \sum _ { i = 1 } ^ { | \mathcal { V } _ { k } | } \sum _ { j = 1 } ^ { | \mathcal { V } _ { t } | } \left( \frac { \mathbf { P } _ { k , k } [ i ] } { \| \mathbf { P } _ { k , k } [ i ] \| _ { 2 } } \cdot \frac { \mathbf { P } _ { t , k } [ j ] ^ { \top } } { \| \mathbf { P } _ { t , k } [ j ] \| _ { 2 } } \right) ,\tag{12}
$$

The final semantic-guided distillation coeficient is obtained by applying a temperature-scaled softmax over all accumulated teachers:

$$
\alpha _ { k } = \frac { \exp ( r _ { k } / \tau ) } { \sum _ { m = 0 } ^ { t } \exp ( r _ { m } / \tau ) } , \quad k = 0 , \ldots , t ,\tag{13}
$$

where $| { \mathcal { D } } _ { k } |$ is the number of categories in the k-th task, the former subscript in $\mathbf { P } _ { i , j }$ , stands for the task index, and the latter for the subspace, the $\mathbf { P } _ { t , k } [ j ]$ is the prototype of the j-th class in the t-th task extracted by the k-th teacher. Specifically, the prototype is computed as:

$$
\mathbf { P } _ { t , k } [ j ] = \frac { 1 } { N } \sum _ { i = 1 } ^ { | \mathcal { D } _ { t } | } \mathbb { I } ( y _ { i } = j ) \phi ( \mathbf { x } ; \mathrm { T } _ { k } ) ,\tag{14}
$$

where I denotes the indicator function, N is the number of instances in class $j , \left| \mathcal { D } _ { t } \right|$ is the number of examples in the current task, and $\tau$ is the temperature coeficient. The raw similarity scores $\{ r _ { k } \} _ { k = 0 } ^ { t }$ are normalized via softmax to obtain the final distillation coeficients $\{ \alpha _ { k } \} _ { k = 0 } ^ { t }$ used in Eq. (8), ensuring that each teacher’s contribution is proportional to its semantic relevance to the current task and that $\textstyle \sum _ { k = 0 } ^ { t } \alpha _ { k } = 1$

The semantic coeficient $\alpha _ { k }$ is a task-level quantity computed once per incremental stage from the current-task class prototypes. Current-task Teacher features are cached during coeficient construction, and the resulting coeficients are reused for all EKT mini-batches and epochs. It introduces no learnable parameters, optimizer state, or backward propagation. Given feature dimension d, the prototype-level similarity computation requires $\begin{array} { r } { \mathcal { O } ( d \vert \mathcal { V } _ { t } \vert \sum _ { k = 0 } ^ { t } \vert \mathcal { V } _ { k } \vert ) } \end{array}$ scalar operations. This training-only computation does not afect deployment storage or inference computation.

## 4.5. Optimization Objective and Training Procedure

TALON employs a two-stage training paradigm that sequentially learns task-specific LoRA-Teacher modules and then consolidates their collective knowledge into a unified LoRA-Student through ensemble knowledge transfer. Algorithm 1 provides the complete training procedure.

Stage 1: Task-Adaptive LoRA-Teacher. For each incremental task t, we train a dedicated LoRA-Teacher $\mathbf { T } _ { t }$ while maintaining the pre-trained backbone in a frozen state. This stage optimizes the cross-entropy loss to ensure efective task-specific adaptation:

$$
\begin{array} { r } { \underset { \mathbf { T } _ { t } , \mathbf { W } _ { \mathrm { h e a d } } [ : , \mathcal { V } _ { t } ] } { \operatorname* { m i n } } \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { t } } \left[ \mathrm { L } _ { \mathrm { C E } } \left( \mathrm { f } ( \mathbf { x } ; \mathbf { T } _ { t } ) , \mathbf { y } \right) \right] , } \end{array}\tag{15}
$$

where the pre-trained backbone remains frozen. At session t, the columns of $\mathbf { W } _ { \mathrm { h e a d } }$ corresponding to the current classes $\mathcal { V } _ { t }$ provide task-local supervision for the current LoRA-Teacher. Historical-class logits are masked when computing the current-task cross-entropy loss. During this stage, the current LoRA-Teacher and the current-class columns of $\mathbf { W } _ { \mathrm { h e a d } }$ are optimized. During the subsequent EKT stage, the complete shared head is frozen.

Algorithm 1: TALON for FSCIL   
Input: Pre-trained model $\phi ( \cdot )$ , incremental datasets $\{ \mathcal { D } _ { 0 } , \hdots , \mathcal { D } _ { T } \}$   
LoRA rank r   
Output: LoRA-Student $\mathbf { S } _ { T }$   
1 Initialize Teacher storage list: $\tau  \emptyset ;$   
2 for $t = 0$ to $T$ do   
// Stage 1: Task-Adaptive LoRA-Teacher.   
3 Initialize LoRA-Teacher $\mathbf T _ { t } \colon \mathbf V _ { t } = \mathbf 0 , \mathbf U _ { t } \sim$ Kaiming;   
4 Train $\mathbf { T } _ { t }$ and the current-class columns of $\mathbf { W } _ { \mathrm { h e a d } }$ on $\mathcal { D } _ { t }$ using   
Eq. (15); retain the historical columns unchanged;   
5 Freeze $\mathbf { T } _ { t }$ parameters;   
6 Append $\mathbf { T } _ { t }$ to Teacher storage: $\mathcal { T }  \mathcal { T } \cup \{ \mathbf { T } _ { t } \}$ ;   
7 Extract prototypes $\{ \mathbf { P } _ { k , k } [ j ] \}$ using Eq. (14) where $k = t ;$   
// Stage 2: Ensemble Knowledge Transfer.   
8 Freeze the complete shared linear head $\mathbf { W } _ { \mathrm { h e a d } }$ during EKT;   
9 if $t = 0$ then   
10 Initialize LoRA-Student $\mathbf { S } _ { 0 } = \mathbf { T } _ { 0 } ;$   
11 end   
12 else   
13 Update LoRA-Student $\mathbf { S } _ { t } = \mathbf { S } _ { t } .$ <sub>−1</sub>;   
14 end   
15 if $t > 0$ then   
16 Compute semantic coeficients $\{ \alpha _ { k } \} _ { k = 0 } ^ { t }$ over all Teachers in T   
via Eq. (13);   
17 for mini-batch $( \mathbf { x } , \mathbf { y } ) \in \mathcal { D } _ { t }$ do   
18 Compute EKT loss $L _ { \mathrm { E K T } }$ using Eq. (11);   
19 Update S<sub>t</sub> using ∇L<sub>EKT</sub>;   
20 end   
21 end   
22 Extract the prototypes $\mathrm { { \bf P } _ { \mathrm { S t u } } } [ j ]$ for classes $j \in \mathcal { V } _ { t }$ via Eq. (16);   
23 Append $\mathbf { P } _ { \mathrm { S t u } }$ to the prototype classifier; prototypes of $\scriptstyle { \mathcal { P } } _ { < t }$   
remain frozen;   
24 end   
25 return the LoRA-Student $\mathbf { S } _ { T } ;$

Stage 2: Ensemble Knowledge Transfer. Given the collection of trained LoRA-Teacher modules $\{ \mathbf { T } _ { 0 } , \ldots , \mathbf { T } _ { t } \}$ , we optimize a LoRA-Student $\mathbf { S } _ { t }$ that consolidates their collective knowledge through the ensemble knowledge transfer mechanism described in Eq. (11). This stage enables eficient single-model inference while preserving the benefits of multi-teacher expertise.

Inference Protocol. After EKT at session t, the current LoRA-Student $\mathbf { S } _ { t }$ serves as the single deployed feature extractor. For each newly introduced class $c \in \mathcal { V } _ { t }$ , its prototype is computed as

$$
\mathbf { P } _ { \mathrm { S t u } } ^ { ( t ) } [ c ] = \frac { 1 } { | \mathcal { D } _ { t } ^ { c } | } \sum _ { ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } _ { t } ^ { c } } \phi ( \mathbf { x } _ { i } ; \mathbf { S } _ { t } ) , \qquad \mathcal { D } _ { t } ^ { c } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } _ { t } : y _ { i } = c \} .\tag{16}
$$

The accumulated prototype bank is the class-indexed collection

$$
\mathcal { P } _ { t } = \left\{ \mathbf { P } _ { \mathrm { S t u } } ^ { ( j _ { c } ) } [ c ] : c \in \mathcal { V } _ { \leq t } \right\} ,\tag{17}
$$

where $j _ { c }$ denotes the session in which class c was introduced. At each session, prototypes of the newly introduced classes are appended to $\mathcal { P } _ { t } ,$ whereas historical entries remain unchanged because TALON follows an exemplar-free protocol and does not retain historical samples. Consequently, a historical class c introduced at session $j _ { c }$ is represented by $\mathbf { P } _ { \mathrm { S t u } } ^ { ( j _ { c } ) } [ c ]$ , while a test sample at a later session $t > j _ { c }$ is represented using the current Student $\mathbf { S } _ { t }$ Prediction is performed by

$$
\boldsymbol { \hat { y } } = \arg \operatorname* { m a x } _ { c \in \mathcal { V } _ { \leq t } } \sin \left( \mathbf { P } _ { \mathrm { S t u } } ^ { ( j _ { c } ) } [ c ] , \boldsymbol { \phi } ( \mathbf { x } ; \mathbf { S } _ { t } ) \right) ,\tag{18}
$$

where sim $( \cdot , \cdot )$ denotes cosine similarity. Because $\mathbf { S } _ { t }$ continues to change while historical entries of $\mathcal { P } _ { t }$ remain frozen, this protocol may introduce prototype– feature inconsistency across incremental stages. We quantify this efect in Section 5.7.

TALON follows a strict exemplar-free protocol: no raw samples or minibatches from previous tasks are stored or replayed. Before deployment, all LoRA-Teachers, their training-time prototype banks, and the shared linear head $\mathbf { W } _ { \mathrm { h e a d } }$ are discarded; inference requires only the current LoRA-Student $\mathbf { S } _ { t }$ and the accumulated Student prototype bank $\mathcal { P } _ { t }$

## 5. Experiments

This section provides a comprehensive evaluation of TALON on multiple benchmarks.

## 5.1. Implementation Details

Datasets: We follow [5, 16, 54] to evaluate the performance on four benchmark datasets: CUB200 [55], CIFAR100 [56], ImageNet-R [57], and miniImageNet [58]. CUB200 and ImageNet-R contain 200 classes, whereas CIFAR100 and miniImageNet each include 100 classes. For all datasets, we adopt the split configuration proposed in [5], as summarized in Table 1.

Table 1: Setups for the four datasets
<table><tr><td>Task</td><td>CUB200</td><td>CIFAR100</td><td>ImageNet-R</td><td>miniImageNet</td></tr><tr><td>Base</td><td>100</td><td>60</td><td>100</td><td>60</td></tr><tr><td></td><td></td><td></td><td>Incremental 10-way 5-shot 5-way 5-shot 10-way 5-shot</td><td>5-way 5-shot</td></tr><tr><td># of tasks</td><td>1+10</td><td>1+8</td><td>1+10</td><td>1+8</td></tr></table>

Comparison methods: We evaluate TALON against state-of-the-art approaches across two categories: PEFT-based CIL methods, including L2P [10], CODA-Prompt [11], LAE [13], InfLoRA [14], and SD-LoRA [15]; and PEFTbased FSCIL methods, including ASP [5] and SEC-prompt [16]. For broader comparison, we additionally consider SimpleFSCIL [59], which employs a prototype-based classifier on a frozen pre-trained backbone, as well as standard full fine-tuning. All methods are evaluated under identical conditions with the same pre-trained backbone (ViT-B/16-IN1K [50]) and dataset splits.

Evaluation metrics: We evaluate model performance using five established metrics [42, 60]: $\mathcal { A } _ { \mathrm { B a s e } } , \mathcal { A } _ { L } , \bar { \mathcal { A } }$ , performance drop (PD, in pp) [5], and forward transfer (FWT) [61]. Specifically, $\mathcal { A } _ { \mathrm { B a s e } }$ denotes the accuracy on the base task, $\boldsymbol { \mathcal { A } } _ { L }$ denotes the accuracy on the final incremental task, and $\begin{array} { r } { \bar { \mathcal { A } } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \mathcal { A C C } _ { i } } \end{array}$ measures the average accuracy across all T incremental stages, where $\begin{array} { r } { \mathcal { A C C } _ { i } = \frac { 1 } { i } \sum _ { j = 1 } ^ { i } a _ { i , j } } \end{array}$ , with $a _ { i , j }$ representing the accuracy on the j-th task after training on the i-th task [14].

The performance drop (PD) quantifies knowledge forgetting, i.e., the absolute accuracy decline (in percentage points) from the base to the final session: $\mathrm { P D } = { \mathcal { A } } _ { \mathrm { B a s e } } - { \mathcal { A } } _ { L }$ , where lower values indicate stronger retention of learned knowledge.

Forward transfer (FWT) evaluates the influence of learning task t on the performance of a future task $k > t$ , capturing the model’s ability to generalize to unseen tasks in a zero-shot manner [61]: $\begin{array} { r } { \mathrm { F W T } = \frac { 1 } { T - 1 } \sum _ { i = 2 } ^ { T } \left( a _ { i - 1 , i } - a _ { \mathrm { r a n d } , i } \right) } \end{array}$ 2 where $a _ { i - 1 , i }$ represents the accuracy on task i after training on task $i - 1$ ， and $a _ { \mathrm { r a n d } , i }$ is the accuracy on task i using a randomly initialized model with prototype-based classification computed from the incremental training data. Positive FWT indicates beneficial knowledge transfer across tasks. When comparing models with similar ${ \bar { \mathcal { A } } } ,$ the one with higher FWT is preferred. Note that FWT is not computed for the final task as there are no subsequent tasks for evaluation.

Training details: We employ ViT-B/16-IN1K [50] as the backbone architecture. For optimization, we utilize SGD with learning rates of 0.01 for MLP-Teachers and 0.02 for QV-Teachers, employing cosine annealing learning rate decay. LoRA-Teachers are trained for 20 epochs on the base task and 10 epochs for each incremental task, with batch sizes of 48 and 16, respectively. We set the LoRA rank to $r = 8$ . For EKT, we use 5 distillation epochs with batch size 16, semantic temperature $\tau = 1$ , and fixed equal weights for $L _ { \mathrm { S D } }$ and ${ \cal L } _ { \mathrm { H D } }$ . The supplementary analysis shows limited sensitivity to τ , with the four-dataset macro-average accuracy varying by only 0.12 percentage points over $\tau \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ . Unless otherwise specified, experiments use seed 1993; multi-seed experiments use seeds 42, 1993, and 2025.

TALON Variants: We evaluate two variants: TALON-QV tunes Query and Value projections, consistent with existing LoRA-based methods (InfLoRA [14], SD-LoRA [15]), enabling fair comparison; TALON-MLP tunes MLP layers, achieving comparable performance with $2 \times$ fewer parameters (0.15M vs. 0.29M). This demonstrates that our framework is agnostic to the specific PEFT location.

## 5.2. Benchmark Comparison

We conduct a comprehensive evaluation of TALON against state-of-theart methods on four benchmark datasets: CUB200, CIFAR100, ImageNet-R, and miniImageNet. Results are reported in Table 2 and Table 3. Performance trajectories across incremental tasks are further illustrated in Fig. 3.

Accuracy: As shown in Table 2, TALON obtains comparable or better mean last-session and average accuracy across all four benchmarks. On CI-FAR100, ImageNet-R, and miniImageNet, the best TALON variant exceeds SEC-prompt in mean average accuracy by 0.86, 0.81, and 0.88 percentage points, respectively. On CUB200, TALON-MLP and SEC-prompt obtain comparable mean average accuracies of $8 6 . 6 8 \pm 1 . 2 2 \%$ and $8 6 . 4 9 \pm 2 . 4 3 \%$ respectively, and their relative ranking varies across class orders. These results highlight TALON’s ability to efectively balance stability and plasticity in FSCIL scenarios.

Table 2: Comparison with state-of-the-art FSCIL methods. Values are mean±sample standard deviation over three class-order runs generated using seeds 42, 1993, and 2025. All methods use the same ViT-B/16-IN1K backbone and dataset splits. The best mean is highlighted in bold, and the second-best mean is underlined.
<table><tr><td rowspan="2">Method</td><td colspan="3">CUB200 (T = 11)</td><td colspan="3">CIFAR100 (T = 9)</td><td colspan="3">ImageNet-R (T = 11)</td><td colspan="3">miniImageNet (T = 9)</td></tr><tr><td> $A _ { \mathrm { B a s e } }$ </td><td> $A _ { L }$ </td><td>A</td><td> $A _ { \mathrm { B a s e } }$ </td><td>AL</td><td>A</td><td> $A _ { \mathrm { B a s e } }$ </td><td>AL</td><td>A</td><td> $A _ { \mathrm { B a s e } }$ </td><td> $A _ { L }$ </td><td>A</td></tr><tr><td>Full Finetune</td><td>89.36±0.44 11.28±3.09 22.19±6.46 92.08±0.54 40.86±7.14 66.17±4.02 82.66±0.76 13.23±2.93 27.99±7.29 95.60±0.52 69.04±4.67 80.88±1.67</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimpleFSCIL</td><td>88.57±2.01</td><td>74.67±1.41 79.71±0.90 78.13±1.51</td><td></td><td></td><td>63.85±0.55</td><td></td><td>69.75±0.69 63.73±1.1050.32±0.79 55.46±0.64</td><td></td><td></td><td>94.74±0.66</td><td>87.55±1.07</td><td>90.65±0.47</td></tr><tr><td>L2P</td><td>90.41±1.89 48.33±1.06 65.13±1.26 92.43±0.72 55.43±0.27</td><td></td><td></td><td></td><td></td><td></td><td>71.25±0.38 79.79±0.96 42.37±2.03 57.51±1.67</td><td></td><td></td><td>97.01±0.30 62.09±0.52</td><td></td><td>77.05±0.67</td></tr><tr><td>CODA-Prompt 91.34±1.92</td><td></td><td>50.38±0.88 66.75±0.68 93.50±0.26</td><td></td><td></td><td>56.25±0.13 72.12±0.14 81.70±0.36 45.67±1.45 60.01±1.76</td><td></td><td></td><td></td><td></td><td>97.65±0.14 63.46±0.91</td><td></td><td>77.68±0.47</td></tr><tr><td>LAE</td><td>90.91±2.37</td><td>50.53±3.99 66.67±2.85 92.19±1.44 55.97±2.14</td><td></td><td></td><td></td><td>71.51±1.79 76.89±2.8942.79±5.41</td><td></td><td></td><td>56.44±4.59</td><td>97.09±0.40 64.34±5.56</td><td></td><td>78.29±3.08</td></tr><tr><td>InfLoRA</td><td>92.24±1.0245.36±0.5864.73±0.31 94.43±0.4656.47±0.22 72.49±0.2485.05±0.5943.37±2.1060.22±1.92</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>97.84±0.09</td><td>58.69±0.12</td><td>75.41±0.09</td></tr><tr><td>SD-LoRA</td><td>91.51±1.91</td><td>60.05±6.01 70.73±3.78 94.16±0.84 74.37±0.80 81.71±0.64 84.17±0.81 51.65±1.15 57.91±1.23 97.88±0.22 80.56±4.39</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>84.79±1.65</td></tr><tr><td>ASP</td><td>90.36±2.69</td><td>82.78±0.15 85.73±2.23</td><td></td><td></td><td></td><td></td><td>89.89±5.04 78.79±12.39 83.72±9.40 81.82±1.10 70.75±3.28</td><td></td><td></td><td>75.40±2.32 96.85±0.22 94.38±0.49</td><td></td><td>95.33±0.51</td></tr><tr><td>SEC-prompt</td><td>90.85±2.84 83.56±0.24 86.49±2.43 92.42±0.74 87.02±0.13 89.53±0.27 83.19±0.35 73.64±1.36</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>77.57±0.98</td><td>96.72±0.28 94.66±0.14</td><td></td><td>95.46±0.25</td></tr><tr><td>TALON-MLP</td><td>91.41±1.89 83.94±1.15 86.68±1.22 93.68±0.68 87.56±0.63 90.39±0.27 84.85±0.59 73.57±1.22 78.19±1.01 97.61±0.12 95.30±0.32</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>96.16±0.33</td></tr><tr><td>TALON-QV</td><td>91.45±2.16 83.42±0.78 86.39±1.55 93.69±0.60 87.41±0.38 90.29±0.25 85.03±0.55 73.71±1.18 78.38±0.94 97.63±0.23 95.56±0.35 96.34±0.33</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Comparison with state-of-the-art FSCIL methods under seed 1993. We report performance drop (PD, in pp) and forward transfer (FWT) using the same ViT-B/16- IN1K backbone.
<table><tr><td rowspan="2">Method</td><td colspan="2">CUB200 (T=11)</td><td colspan="2">CIFAR100 (T=9)</td><td colspan="2">ImageNet-R (T=11)</td><td colspan="2">miniImageNet (T=9)</td></tr><tr><td>PD↓</td><td>FWT↑</td><td>PD↓</td><td>FWT↑</td><td>PD↓</td><td>FWT↑</td><td>PD↓</td><td>FWT↑</td></tr><tr><td>Full Finetune</td><td>79.49</td><td>-17.13</td><td>51.52</td><td>34.80</td><td>66.60</td><td>15.69</td><td>31.92</td><td>9.02</td></tr><tr><td>SimpleFSCIL</td><td>9.96</td><td>13.67</td><td>12.74</td><td>33.80</td><td>11.56</td><td>35.89</td><td>7.32</td><td>-0.47</td></tr><tr><td>L2P</td><td>41.52</td><td>11.67</td><td>36.97</td><td>40.80</td><td>37.37</td><td>38.89</td><td>35.08</td><td>15.52</td></tr><tr><td>CODA-Prompt</td><td>39.79</td><td>17.78</td><td>37.29</td><td>44.30</td><td>36.00</td><td>45.29</td><td>33.51</td><td>13.02</td></tr><tr><td>LAE</td><td>42.16</td><td>9.07</td><td>36.83</td><td>45.30</td><td>36.34</td><td>35.49</td><td>38.56</td><td>-30.98</td></tr><tr><td>InfLoRA</td><td>45.66</td><td>14.87</td><td>37.75</td><td>37.30</td><td>42.44</td><td>44.89</td><td>39.18</td><td>17.02</td></tr><tr><td>SD-LoRA</td><td>23.13</td><td>23.60</td><td>19.21</td><td>47.50</td><td>32.17</td><td>35.01</td><td>15.28</td><td>13.92</td></tr><tr><td>ASP</td><td>4.33</td><td>9.47</td><td>19.63</td><td>12.30</td><td>13.86</td><td>17.69</td><td>2.60</td><td>6.52</td></tr><tr><td>SEC-prompt</td><td>4.33</td><td>9.47</td><td>4.75</td><td>30.80</td><td>10.79</td><td>26.09</td><td>2.36</td><td>8.52</td></tr><tr><td>TALON-MLP</td><td>4.00</td><td>13.27</td><td>5.12</td><td>39.30</td><td>12.30</td><td>39.49</td><td>2.46</td><td>16.02</td></tr><tr><td>TALON-QV</td><td>4.76</td><td>16.67</td><td>5.44</td><td>47.80</td><td>12.51</td><td>41.49</td><td>2.36</td><td>16.52</td></tr></table>

Forgetting: Table 3 shows that conventional PEFT-based methods such as InfLoRA and SD-LoRA sufer from severe catastrophic forgetting. The limited training samples in FSCIL provide insuficient gradient signals, leading to both overfitting to new classes and erosion of previously learned knowledge. By contrast, TALON maintains relatively low PD values across all four datasets. For example, on CUB200, TALON-MLP obtains the lowest PD of

Table 4: Detailed Top-1 accuracy (%) at each FSCIL session on CUB200 under seed 1993. We report per-session accuracy $A _ { t } ,$ average accuracy ${ \bar { \mathcal { A } } } ,$ and performance drop (PD, in pp).
<table><tr><td>Method</td><td> $A _ { 0 }$   $A _ { 1 }$ </td><td> $A _ { 2 }$   $A _ { 3 }$ </td><td> $A _ { 4 }$   $A _ { 5 }$   $A _ { 6 }$ </td><td> $A _ { 7 }$   $A _ { 8 }$ </td><td></td><td> $A _ { 9 }$   $A _ { 1 0 }$ </td><td>À PD↓</td></tr><tr><td>Full Finetune</td><td>88.90 2.28 3.30</td><td></td><td>7.91 6.18 9.83 10.49 13.00 11.88 7.70 9.41 15.53 79.49</td><td></td><td></td><td></td><td></td></tr><tr><td>SimpleFSCIL</td><td></td><td></td><td></td><td>86.25 83.23 81.69 80.07 79.33 77.70 77.30 77.26 76.48 76.38 76.2979.279.96</td><td></td><td></td><td></td></tr><tr><td>L2P</td><td></td><td></td><td></td><td>88.81 82.91 75.74 70.03 64.6561.23 57.40 53.4650.69 48.41 47.2963.69 41.52</td><td></td><td></td><td></td></tr><tr><td></td><td>CODA-Prompt 89.5884.80 77.89 72.09 67.6563.69 59.9856.22 53.05 50.92 49.79 65.97 39.79</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LAE InfLoRA</td><td>88.56 82.2875.09 69.50 65.0261.01 57.45 53.61 50.54 48.50 46.40 63.45 42.16</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SD-LoRA</td><td></td><td></td><td></td><td>91.12 85.4377.96 72.09 66.2461.75 57.88 53.8250.64 47.92 45.4664.57 45.66</td><td>89.50 82.9979.11 75.42 74.62 71.41 70.68 70.4369.3868.01 66.37 74.36 23.13</td><td></td><td></td></tr><tr><td>ASP</td><td>87.2885.91 84.4283.06 82.75 81.76 81.66 82.23 81.92 82.3782.95 83.304.33</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SEC-prompt</td><td>87.6286.5485.0783.6583.4382.4582.1482.8382.4482.7783.2983.844.33</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TALON-MLP</td><td></td><td></td><td></td><td></td><td>89.2487.80 86.1585.2585.0283.82 84.19 84.9984.4884.8385.2485.554.00</td><td></td><td></td></tr></table>

4.00%, substantially lower than InfLoRA (45.66%) and SD-LoRA (23.13%). Across CIFAR100, ImageNet-R, and miniImageNet, TALON maintains competitive knowledge retention, while SEC-prompt obtains lower or equal PD values.

Forward Transfer: TALON also exhibits positive forward transfer (FWT) across all datasets (Table 3). Compared with methods achieving similar accuracy $( \mathrm { e . g . } )$ , SEC-prompt and ASP), TALON demonstrates a stronger ability to leverage prior knowledge for learning novel tasks. For instance, on CUB200, TALON-QV achieves an FWT of 16.67%, significantly higher than SEC-prompt’s 9.47%. On CIFAR100, TALON-QV obtains the best FWT at 47.80%, markedly surpassing SEC-prompt’s 30.80%. On miniImageNet, TALON-QV records 16.52%, outperforming SEC-prompt’s 8.52%. This indicates TALON’s capacity for positive knowledge transfer, which is particularly valuable in few-shot incremental settings.

Following common FSCIL reporting practice, we additionally provide detailed per-session Top-1 accuracies on CUB200 in Table 4, which verifies that TALON maintains consistently strong performance throughout the incremental sequence rather than only at the final session.

Finally, the performance curves in Fig. 3 confirm that TALON maintains consistently high accuracy throughout the entire incremental learning process, underscoring its robustness and stability.

Longer task sequences (T=21) and a larger backbone (ViT-L/16) are further evaluated in the supplementary material, where TALON’s advantages remain consistent.

![](images/a8547c3d8921e87dd5b26819877d07e522c1cf204b793a1aa7e46a3080a6db42.jpg)  
(a) CUB200 (T =11)

![](images/46022ea324e779cfb3f7c4dc0eabfc6b05dee1f1e291ac24577f0b4ce9666ffe.jpg)  
(b) CIFAR100 (T =9)

![](images/f18640cbd3e7e82b5348f8ff8bd96044badbbd9b0d438f73dc052f1fffe6b60c.jpg)  
(c) ImageNet-R (T =11)

![](images/388fbede080bc24e3218bb1894080222297894170bd823ee6b3bbf516519d8d1.jpg)  
(d) miniImageNet (T =9)  
Figure 3: Performance curves on diferent datasets. All methods are based on the same pre-trained model (ViT-B/16-IN1K).

## 5.3. Eficiency Analysis

This subsection evaluates computational eficiency by explicitly separating training-phase and deployment-phase costs. Deployment eficiency is measured using inference latency and the model state retained for inference. Training eficiency is measured using training-state parameters, peak GPU memory, per-epoch training time, cumulative Teacher-training time, cumulative EKT time, and end-to-end wall-clock time. We first compare TALON with baselines under the standard FSCIL sequences and then evaluate TALON-QV under the longer T = 21 setting.

Inference Eficiency: Many existing methods incur substantial computational overhead during inference due to reliance on dynamic module selection or runtime module generation. Specifically, prompt-based continual learning approaches such as L2P and CODA-Prompt require additional inference time for prompt selection on each input sample. LAE introduces extra forward passes for ensemble operations, while ASP and SEC-prompt employ computationally intensive encoder networks to generate task-specific modules conditioned on input features.

In contrast, TALON eliminates these bottlenecks via its Ensemble Knowledge Transfer (EKT) mechanism, which consolidates knowledge from all LoRA-Teachers into a single LoRA-Student during training. This design enables eficient single-model inference without runtime module selection or additional ensemble computations. Fig. 4 presents a comprehensive comparison of inference times across methods, demonstrating that TALON achieves the lowest latency for all evaluated tasks. For example, TALON-QV completes inference for a single task in 26.7 s, approximately 46.3% faster than SEC-prompt (49.7 s) and 41.7% faster than ASP (45.8 s). This substantial reduction in inference time highlights TALON’s suitability for real-time deployment in resource-constrained scenarios.

![](images/3fb01ba84148ea9142d48c105aef6220da1892eabff03ac37450d915667e68e0.jpg)  
Figure 4: Inference time per task (seconds) for diferent methods. Here, “per task” denotes one FSCIL session, and each value measures the wall-clock evaluation time at task t on all classes seen so far. Experiments are conducted on an NVIDIA A800 GPU, with batch size 48 for the base task and 16 for incremental tasks.

Expanded Parameters Analysis: An important distinction in TALON is between training-phase and deployment-phase expanded parameters. During training, TALON retains all accumulated LoRA-Teachers, the LoRA-Student, the auxiliary classifier, and the Teacher and Student prototype banks. Before deployment, the LoRA-Teachers, Teacher prototype banks, and auxiliary classifier are discarded; inference retains only the LoRA-Student and the Student prototype classifier. Accordingly, Fig. 5 compares deploymentphase expanded parameters and accuracy on CUB200, CIFAR100, and miniImageNet, where expanded parameters refer to the additional learned model parameters retained for inference beyond the frozen pre-trained backbone.

![](images/0a1121deb1e22eccdc1fd7b14a0ad0fd1faa0c2e9013f344f9077d6d0ae9b030.jpg)  
(a) CUB200 (T=11)

![](images/c83c802b25e2c1f8f381b17f510eecb1d0be629bc998439e3db6c036ac4b035d.jpg)  
(b) CIFAR100 (T=9)

![](images/299b9984d89a5338962a258b921d55d2f7a52133a57d22143ec316ca522b5754.jpg)  
(c) miniImageNet (T=9)  
Figure 5: The number of expanded parameters and accuracy of diferent methods.

The parameter composition of each method is as follows: L2P and CODA-Prompt add parameters via learnable prompts and their keys; LAE incorporates LoRA modules alongside ensemble components; ASP and SECprompt employ both task-specific and task-agnostic prompts. In contrast, TALON’s expanded parameters are solely attributed to the LoRA-Student model, maintaining a parameter count comparable to InfLoRA and SD-LoRA while delivering superior accuracy.

Experimental results demonstrate that TALON establishes a new benchmark in the parameter-performance trade-of. Both TALON-MLP and TALON-QV achieve superior accuracy with substantially fewer parameters than existing methods. For example, on CIFAR100, TALON-MLP attains 90.26% accuracy with only 0.15M parameters, outperforming SEC-prompt (89.22% with 2.50M parameters) while using over 16× fewer parameters. On CUB200, TALON-MLP achieves 85.55%, surpassing SEC-prompt (83.84%) with approximately 33× fewer parameters (0.15M vs. 4.99M).

Notably, TALON maintains consistent parameter eficiency across datasets, unlike methods such as SEC-prompt, whose parameter count varies with dataset size. When compared to other LoRA-based approaches with similar budgets, e.g., InfLoRA (0.37M) and SD-LoRA (0.37M), TALON’s advantages become even more pronounced. On miniImageNet, TALON-QV (0.29M parameters) achieves 96.44%, substantially higher than InfLoRA (75.34%) and SD-LoRA (85.20%). These results highlight TALON’s exceptional parameter-performance trade-of, demonstrating its ability to achieve high accuracy while remaining computationally eficient—particularly valuable in resource-constrained continual learning settings.

Training Eficiency: At session t, TALON first trains a new taskspecific LoRA-Teacher and then distills all t + 1 accumulated frozen Teachers into the LoRA-Student. Because the Teachers require forward computation but no gradient computation during EKT, the Teacher-side cost of each EKT epoch grows as O(t).

Excluding the shared frozen pre-trained model, the retained training state at session t is

$$
M _ { \mathrm { t r a i n } } ( t ) = ( t + 1 ) P _ { \mathrm { T } } + P _ { \mathrm { S } } + P _ { \mathrm { a u x } } + M _ { \mathrm { p r o t o } } ( t ) ,\tag{19}
$$

where $P _ { \mathrm { T } } , \ P _ { \mathrm { S } }$ , and $P _ { \mathrm { a u x } }$ denote one LoRA-Teacher, the LoRA-Student, and the auxiliary classifier, respectively, while $M _ { \mathrm { p r o t o } } ( t )$ denotes the retained Teacher and Student prototype banks.

After training, the accumulated Teachers, Teacher prototype banks, and auxiliary classifier are not required for inference and can be discarded. The deployment state is therefore

$$
M _ { \mathrm { d e p l o y } } = P _ { \mathrm { S } } + C _ { \leq T } d ,\tag{20}
$$

where $C _ { \leq T }$ is the number of observed classes and d is the feature dimension. Thus, the number of accumulated Teachers afects training-phase computation and retained training state, while deployment retains one LoRA-Student and one Student prototype classifier.

Long-Sequence Training Scalability: Using TALON-QV throughout, we compare $T = 1 1$ with $T = 2 1$ on CUB200 and ImageNet-R, and $T = 9$ with $T = 2 1$ on CIFAR100 and miniImageNet, where $T$ denotes the total number of sessions including the base session. Accuracy is evaluated using the protocol adopted in the main benchmark comparison. Eficiency statistics are measured on the same NVIDIA A800 and reported as mean±sample standard deviation over three complete timing repetitions. Teacher-training time is accumulated over all task-specific Teacher updates, cumulative EKT includes semantic-score computation and Student distillation, and total wallclock time is measured end to end by the incremental training pipeline.

Table 5 summarizes TALON-QV performance and eficiency at $T =$ $1 1 / T = 9$ and $T = 2 1$ . Extending the task sequence changes average accuracy by $+ 0 . 2 9 , + 0 . 0 7 , - 0 . 1 6 .$ , and +0.03 percentage points on CUB200, CI-FAR100, ImageNet-R, and miniImageNet, respectively. At $T = 2 1$ , TALON-

Table 5: Performance and eficiency of TALON-QV under the dataset-specific $T = 1 1 / T =$ 9 settings and the extended $T = 2 1$ setting. Training times are mean±sample standard deviation over three repetitions (seeds 42, 1993, and 2025) on the same NVIDIA A800; peak GPU memory and state sizes are deterministic. Retained training state excludes the frozen PTM and optimizer state, while deployment retains only the LoRA-Student and Student prototype classifier.
<table><tr><td colspan="6">(a) Accuracy and cumulative time</td></tr><tr><td></td><td></td><td></td><td>Teacher train (s)</td><td>EKT time (s)</td><td>Total time (s)</td></tr><tr><td>Dataset</td><td>T</td><td>À (%)</td><td> $6 5 0 . 3 \pm 2 . 2$ </td><td> $1 0 0 . 2 \pm 0 . 3$ </td><td></td></tr><tr><td>CUB200 CUB200</td><td> $T = 1 1$   $T = 2 1$ </td><td>84.95 85.24</td><td> $6 8 9 . 8 \pm 4 . 1$ </td><td> $1 9 9 . 5 \pm 4 . 4$ </td><td> $7 8 7 . 8 \pm 1 . 6$   $9 3 2 . 4 \pm 8 . 4$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CIFAR100 CIFAR100</td><td> $T = 9$   $T = 2 1$ </td><td>90.03</td><td> $3 5 5 3 . 0 \pm 6 3 . 3$   $3 6 1 3 . 0 \pm 8 5 . 4$ </td><td> $6 3 . 6 \pm 2 1 . 2$   $1 5 9 . 9 \pm 5 7 . 5$ </td><td> $3 8 1 7 . 2 \pm 9 9 . 3$   $3 9 8 1 . 6 \pm 1 5 9 . 3$ </td></tr><tr><td></td><td></td><td>90.10</td><td></td><td></td><td></td></tr><tr><td>ImageNet-R ImageNet-R</td><td> $T = 1 1$ </td><td>77.99</td><td> $1 6 1 3 . 4 \pm 2 5 . 0$   $1 7 0 5 . 6 \pm 1 7 . 1$ </td><td> $1 1 8 . 9 \pm 2 4 . 1$   $2 7 6 . 7 \pm 5 7 . 6$ </td><td> $1 8 2 6 . 2 \pm 4 6 . 6$ </td></tr><tr><td></td><td> $T = 2 1$ </td><td>77.83</td><td></td><td></td><td> $2 0 9 4 . 2 \pm 8 3 . 5$ </td></tr><tr><td>miniImageNet</td><td>T = 9</td><td>96.44</td><td> $4 2 4 3 . 2 \pm 1 1 2 6 . 1$ </td><td> $6 3 . 9 \pm 1 5 . 9$ </td><td> $4 5 9 8 . 8 \pm 1 2 5 5 . 0$ </td></tr><tr><td>miniImageNet</td><td> $T = 2 1$ </td><td>96.47</td><td> $4 6 5 7 . 0 \pm 9 6 9 . 9$ </td><td> $1 3 8 . 1 \pm 3 6 . 9$ </td><td> $5 0 9 6 . 6 \pm 1 0 5 6 . 7$ </td></tr></table>

<table><tr><td colspan="5">(b) Memory and retained states</td></tr><tr><td>Dataset</td><td>T</td><td>Peak GPU (MiB)</td><td>Training state (MiB)</td><td>Deploy. state (MiB)</td></tr><tr><td>CUB200</td><td> $T = 1 1$ </td><td>4962.3</td><td>15.26</td><td>1.71</td></tr><tr><td>CUB200</td><td> $T = 2 1$ </td><td>4962.3</td><td>26.51</td><td>1.71</td></tr><tr><td>CIFAR100</td><td> $T = 9$ </td><td>4961.0</td><td>12.13</td><td>1.42</td></tr><tr><td>CIFAR100</td><td> $T = 2 1$ </td><td>4961.0</td><td>25.63</td><td>1.42</td></tr><tr><td>ImageNet-R</td><td> $T = 1 1$ </td><td>4962.3</td><td>15.26</td><td>1.71</td></tr><tr><td>ImageNet-R</td><td> $T = 2 1$ </td><td>4962.3</td><td>26.51</td><td>1.71</td></tr><tr><td>miniImageNet</td><td></td><td>4961.0</td><td>12.13</td><td>1.42</td></tr><tr><td>miniImageNet</td><td> $T = 9$ </td><td>4961.0</td><td>25.63</td><td>1.42</td></tr><tr><td></td><td> $T = 2 1$ </td><td></td><td></td><td></td></tr></table>

QV achieves average accuracies of 85.24%, 90.10%, 77.83%, and $9 6 . 4 7 \%$ , exceeding SEC-prompt by 0.78, 1.29, 1.41, and 0.57 points, respectively. The complete comparison with other methods is reported in Table C.4 of the supplementary material.

From $T = 1 1 / T = 9$ to $T = 2 1$ , cumulative EKT time grows by 1.99×, 2.52×, 2.33×, and 2.16× on CUB200, CIFAR100, ImageNet-R, and miniImageNet, respectively, while end-to-end wall-clock time increases by 18.4%, 4.3%, 14.7%,

Table 6: Training-phase comparison with baselines at T = 11 on CUB200 and ImageNet-R and at $T = 9$ on CIFAR100 and miniImageNet. We report training-state expanded parameters, peak allocated GPU memory, and training time per epoch.
<table><tr><td>Dataset</td><td>Method</td><td>Training params (M)</td><td>Peak GPU (MiB)</td><td>Time/epoch (s)</td></tr><tr><td>CUB200</td><td></td><td></td><td></td><td></td></tr><tr><td>(T = 11)</td><td>ASP</td><td>2.00</td><td>5213.16</td><td>15.94</td></tr><tr><td></td><td>SEC-prompt</td><td>4.03</td><td>4892.90</td><td>3.63</td></tr><tr><td></td><td>TALON-MLP</td><td>3.10</td><td>4372.37</td><td>23.96</td></tr><tr><td></td><td>TALON-QV</td><td>6.19</td><td>4961.68</td><td>24.36</td></tr><tr><td>CIFAR100</td><td></td><td></td><td></td><td></td></tr><tr><td>(T = 9)</td><td>ASP</td><td>2.00</td><td>9738.25</td><td>80.00</td></tr><tr><td></td><td>SEC-prompt</td><td>1.99</td><td>4747.56</td><td>18.86</td></tr><tr><td></td><td>TALON-MLP</td><td>2.51</td><td>4371.35</td><td>28.15</td></tr><tr><td></td><td>TALON-QV</td><td>5.01</td><td>4960.67</td><td>28.34</td></tr><tr><td>ImageNet-R</td><td></td><td></td><td></td><td></td></tr><tr><td>(T = 11)</td><td>ASP</td><td>4.83</td><td>5692.49</td><td>34.45</td></tr><tr><td></td><td>SEC-prompt</td><td>5.64</td><td>6474.20</td><td>7.42</td></tr><tr><td></td><td>TALON-MLP</td><td>3.10</td><td>4372.37</td><td>29.39</td></tr><tr><td></td><td>TALON-QV</td><td>6.19</td><td>4961.68</td><td>24.99</td></tr><tr><td>miniImageNet</td><td></td><td></td><td></td><td></td></tr><tr><td>(T = 9)</td><td>ASP</td><td>2.00</td><td>9738.00</td><td>90.89</td></tr><tr><td></td><td>SEC-prompt</td><td>1.99</td><td>4747.56</td><td>28.66</td></tr><tr><td></td><td>TALON-MLP</td><td>2.51</td><td>4371.35</td><td>28.84</td></tr><tr><td></td><td>TALON-QV</td><td>5.01</td><td>4960.67</td><td>29.26</td></tr></table>

and 10.8%. Retained training state increases to 25.63–26.51 MiB because additional Teachers and their prototypes are retained. Peak GPU memory remains unchanged because the frozen Teachers are evaluated sequentially without retaining gradients. Deployment state also remains unchanged because the accumulated Teachers and training-only components are not required for inference.

As shown in Table 6, TALON-MLP maintains the lowest peak GPU memory usage (∼ 4.3 GB) across all datasets, corresponding to reductions of 7.9%–32.5% relative to SEC-prompt and 16.1%–55.1% relative to ASP. The reported training time per epoch, approximately 24–29 s, is higher than SEC-prompt because EKT performs sequential forward passes through the frozen Teacher ensemble, but remains lower than ASP on three of the four benchmarks. Moreover, EKT requires only 5 epochs, compared with 10–20 epochs for Teacher training, which limits its absolute cost at the corresponding $T = 1 1 / T = 9$ task horizons.

Table 7 shows that extending TALON-QV to T = 21 changes average accuracy by at most 0.29 percentage points. Cumulative EKT time grows by $1 . 9 9 \times - 2 . 5 2 \times$ , while end-to-end wall-clock time increases by 4.3%–18.4% and retained training state grows by 1.74×–2.11×. TALON-QV continues to outperform SEC-prompt by 0.57–1.41 points at T = 21. Peak GPU memory and deployment storage remain unchanged in every comparison.

Table 7: Changes produced by extending TALON-QV from $T = 1 1 / T = 9$ to T = 21. EKT and retained-state columns report multiplicative growth.
<table><tr><td>Dataset</td><td>Tasks</td><td>∆Ã</td><td>Wall increase</td><td>EKT growth</td><td>Retained growth</td><td>Gain over SEC-prompt</td></tr><tr><td>CUB200</td><td>11 → 21</td><td>+0.29</td><td>+18.4%</td><td>1.99×</td><td>1.74×</td><td>+0.78</td></tr><tr><td>CIFAR100</td><td>9 → 21</td><td>+0.07</td><td>+4.3%</td><td>2.52×</td><td>2.11×</td><td>+1.29</td></tr><tr><td>ImageNet-R</td><td>11 → 21</td><td>-0.16</td><td>+14.7%</td><td>2.33×</td><td>1.74×</td><td>+1.41</td></tr><tr><td>miniImageNet</td><td>9 → 21</td><td>+0.03</td><td>+10.8%</td><td>2.16×</td><td>2.11×</td><td>+0.57</td></tr></table>

Semantic-Score Computation: In addition to the repeated Teacher forwards used by EKT, semantic coeficient construction is performed once per incremental stage using cached current-task Teacher features. The complete computation includes feature extraction, prototype construction, and similarity evaluation but requires neither learnable parameters nor backward propagation. On an NVIDIA A800, its cumulative time over a complete incremental sequence is 27.06 s on CUB200, 11.36 s on CIFAR100, 29.81 s on ImageNet-R, and 14.33 s on miniImageNet. These values correspond to 4.08%, 0.33%, 1.84%, and 0.31% of task-specific Teacher-training time, respectively. Thus, semantic weighting adds a limited training-only cost and does not change TALON’s trainable parameter count, deployment state, or inference complexity.

Detailed per-method expanded parameter calculations and per-session EKT timing breakdowns are provided in the supplementary material.

## 5.4. Ablation Studies

This subsection presents comprehensive ablation studies to validate the efectiveness of individual components in TALON. We systematically investigate the contribution of each module and analyze the impact of architectural choices on overall performance.

Diferent Components: We first evaluate the contribution of TALON’s core components, with results reported in Table 8.

Table 8: Ablation studies of diferent components. For each metric, the left/right values represent performance with MLP-Teacher and QV-Teacher fine-tuning, respectively
<table><tr><td rowspan="2">Ablated Components</td><td colspan="2">CUB200 (T=11)</td><td colspan="2">CIFAR100 (T=9)</td><td rowspan="2"></td><td colspan="2">miniImageNet (T=9)</td></tr><tr><td> $\mathcal { A } _ { L }$ </td><td>A</td><td> $\boldsymbol { \mathcal { A } } _ { L }$ </td><td>A</td><td> $A _ { L }$ </td><td>A</td></tr><tr><td>w/o LoRA-Teacher KD</td><td>29.13 71.76</td><td>45.61 77.50</td><td>74.49 85.27</td><td>79.16 88.08</td><td>87.70</td><td>93.60</td><td>93.71 95.20</td></tr><tr><td>w/o Semantic Similarity</td><td>83.14 83.56</td><td>84.07 84.02</td><td>82.12 86.37</td><td>88.52 89.19</td><td>94.12</td><td>94.43</td><td>95.18 95.42</td></tr><tr><td>TALON-MLP /QV</td><td>85.24 84.31</td><td>85.55 84.95</td><td>87.95 87.66</td><td>90.26 90.03</td><td>95.22</td><td>95.49</td><td>96.22 96.44</td></tr></table>

• w/o LoRA-Teacher KD: This variant removes the ensemble knowledge distillation mechanism and instead relies on a single LoRA-Student continuously updated across incremental tasks. This design yields inferior performance, as it fails to balance the stability–plasticity trade-of. In contrast, assigning a dedicated LoRA-Teacher per task preserves past knowledge more efectively while supporting adaptation to new tasks, confirming the benefit of task-specific teachers in PEFT-based continual learning.

• w/o Semantic Similarity: This variant eliminates the adaptive distillation coeficients and treats all teachers equally during knowledge transfer. As shown in Table 8, this consistently degrades performance across datasets. For instance, on CIFAR100, the average accuracy (A<sup>¯</sup>) of TALON-MLP drops from 90.26% to 88.52% (-1.74%). Without semantic guidance, the student is unable to prioritize knowledge from relevant teachers, making it more vulnerable to conflicting supervision signals.

Together, these results highlight that both the ensemble distillation mechanism and the semantic-aware coeficients are indispensable for TALON’s superior performance in Few-Shot Class-Incremental Learning.

Comparison of Teacher-Weighting Strategies: To isolate the efect of the coeficient formulation, we keep the TALON-QV architecture, frozen Teachers, pre-EKT Student checkpoints, auxiliary head, prototype banks, data splits, class orders, optimization settings, and EKT objective fixed, and change only the computation of the Teacher coeficients. We compare summed cosine with T0-KD, Uniform-KD, maximum similarity, mean pairwise cosine, centroid cosine, normalized negative Euclidean similarity, and learnable task-level weighting.

Table 9 shows that summed pairwise cosine achieves the highest fourdataset macro-average accuracy of 87.35%. All alternative weighting rules reduce the macro average by 0.54–1.10 percentage points. TALON-QV ranks first on CUB200, CIFAR100, and ImageNet-R, while remaining within 0.03 points of the best result on miniImageNet. The class-count-normalized mean score obtains 86.46%, corresponding to a decrease of 0.89 points, and T0-KD obtains 86.65%, corresponding to a decrease of 0.70 points. These results support summed cosine as the most efective and balanced weighting formulation across the four benchmarks.

Table 9: Comparison of TALON’s summed semantic-guided coeficient with alternative task-level Teacher-weighting strategies. Values are average accuracy A<sup>¯</sup> (%). “Macro Avg.” is the unweighted mean over the four datasets, and ∆ denotes the change relative to summed cosine. Macro averages and diferences are calculated before rounding.
<table><tr><td>Weighting strategy</td><td>CUB200</td><td>CIFAR100</td><td>ImageNet-R</td><td>miniImageNet</td><td>Macro Avg.</td><td>∆</td></tr><tr><td>Summed cosine (TALON-QV)</td><td>84.95</td><td>90.03</td><td>77.99</td><td>96.44</td><td>87.35</td><td></td></tr><tr><td>T0-KD</td><td>84.23</td><td>89.11</td><td>76.82</td><td>96.44</td><td>86.65</td><td>-0.70</td></tr><tr><td>Uniform-KD</td><td>84.16</td><td>89.27</td><td>76.16</td><td>96.46</td><td>86.51</td><td>-0.84</td></tr><tr><td>Maximum similarity</td><td>84.81</td><td>89.09</td><td>76.89</td><td>96.47</td><td>86.82</td><td>-0.54</td></tr><tr><td>Mean pairwise cosine</td><td>84.53</td><td>89.08</td><td>75.80</td><td>96.43</td><td>86.46</td><td>-0.89</td></tr><tr><td>Centroid cosine</td><td>84.74</td><td>89.09</td><td>75.77</td><td>96.43</td><td>86.51</td><td>-0.85</td></tr><tr><td>Negative Euclidean</td><td>83.68</td><td>89.09</td><td>75.81</td><td>96.43</td><td>86.25</td><td>-1.10</td></tr><tr><td>Learnable task-level weighting</td><td>82.59</td><td>90.02</td><td>76.21</td><td>96.36</td><td>86.30</td><td>-1.06</td></tr></table>

The learnable task-level weighting control obtains a four-dataset macroaverage accuracy of 86.30%, which is 1.06 percentage points below summed cosine. Its average accuracies are lower by 2.36 and 1.78 points on CUB200 and ImageNet-R, respectively, while the results remain close on CIFAR100 and miniImageNet, with diferences of 0.01 and 0.08 points. Under the same checkpoints and EKT settings, the evaluated learnable parameterization does not improve average accuracy on any of the four benchmarks. These results support retaining summed cosine as the default weighting rule without additional trainable Teacher-weight parameters.

Knowledge Consolidation Strategies: A central design question in TALON is how to consolidate knowledge from multiple task-specific LoRA-Teachers. We compare TALON’s EKT (output-level knowledge distillation) against three representative alternatives, directly addressing whether (i) using the teacher ensemble at inference can replace the distilled student, and (ii) simple deterministic merge baselines sufice. Results are summarized in Table 10.

• Ensemble-Logits (Teacher Ensemble at Inference): This variant retains all LoRA-Teachers and ensembles their logit outputs at test time without distilling a student. It degrades sharply on later incremental tasks: $\boldsymbol { \mathcal { A } } _ { L }$ drops to 64.90% on CIFAR100 and 50.43% on ImageNet-R, trailing TALON by 12.45% and 13.15% in ${ \bar { A } } .$ Without knowledge consolidation, conflicting teacher signals accumulate, and inference cost scales linearly with the number of tasks.

Table 10: Comparison of diferent knowledge consolidation strategies on FSCIL. We report the base, last, and average accuracy. All methods are based on the same pretrained model (ViT-B/16-IN1K).
<table><tr><td rowspan="2">Method</td><td colspan="3">CUB200 (T=11)</td><td colspan="3">CIFAR100 (T=9)</td><td colspan="3">ImageNet-R (T=11) miniImageNet (T=9)</td><td colspan="3"></td></tr><tr><td> $\mathcal { A } _ { \mathrm { B a s e } }$  ←</td><td> $\boldsymbol { \mathcal { A } } _ { L } \uparrow$ </td><td>A↑</td><td> $A _ { \mathrm { B a s e } }$  ←</td><td> $\boldsymbol { \mathcal { A } } _ { L }$  ←</td><td> $\bar { A } \uparrow$ </td><td> $A _ { \mathrm { B a s e } }$  ←</td><td> $\cdot \mathcal { A } _ { L } \cdot \mathcal $  ←</td><td>A↑</td><td> $\mathcal { A } _ { \mathrm { B a s e } }$  ↑</td><td> $\boldsymbol { \mathcal { A } } _ { L } \uparrow$ </td><td>A↑</td></tr><tr><td>Routers</td><td>88.56</td><td>79.13</td><td>81.62</td><td>93.17</td><td>57.29</td><td>72.30</td><td>84.47</td><td>55.50</td><td>64.22</td><td>97.68</td><td>93.44</td><td>95.40</td></tr><tr><td>Ensemble-Weights</td><td>88.73</td><td>6.91</td><td>48.48</td><td>91.28</td><td>85.57</td><td>87.94</td><td>82.86</td><td>13.53</td><td>48.77</td><td>97.52</td><td>95.14</td><td>96.03</td></tr><tr><td>Ensemble-Logits</td><td>90.35</td><td>67.51</td><td>76.10</td><td>94.32</td><td>64.90</td><td>77.81</td><td>87.23</td><td>50.43</td><td>64.84</td><td>97.93</td><td>85.44</td><td>91.04</td></tr><tr><td>TALON-MLP</td><td>89.24</td><td></td><td>85.2485.55</td><td>93.07</td><td>87.95 90.26</td><td></td><td>84.53</td><td>72.23</td><td>77.54</td><td>97.68</td><td>95.22</td><td>96.22</td></tr><tr><td>TALON-QV</td><td>89.07</td><td>84.31</td><td>84.95</td><td>93.10</td><td>87.66</td><td>90.03</td><td>85.06</td><td>72.55 77.99</td><td></td><td>97.85</td><td>95.49</td><td>96.44</td></tr></table>

• Ensemble-Weights (LoRA Weight Averaging): This baseline averages the LoRA weight matrices across all teachers: $\Delta \mathbf { W } _ { \mathrm { m e r g e d } } ~ =$ $\begin{array} { r } { \frac { 1 } { t + 1 } \sum _ { k = 0 } ^ { t } \Delta \mathbf { W } _ { k } } \end{array}$ . On CUB200 and ImageNet-R, this produces catastrophic results $( \mathcal { A } _ { L }$ of 6.91% and 13.53%), because linear parameter averaging destroys the non-linear representations learned by individual teachers. On CIFAR100 and miniImageNet, the damage is less severe, but A<sup>¯</sup> still lags behind TALON by 2.32% and 0.41%, respectively.

• Learnable Routers (MoE-style): A layer-wise learnable router computes softmax gating weights over all LoRA-Teachers. For each Transformer layer $l ,$ the router produces per-expert weights via a two-layer MLP: $\mathbf { g } ^ { ( l ) } = \mathrm { S o f t m a x } \Big ( \mathbf { W } _ { 2 } ^ { ( l ) }$ $\mathrm { R e L U } ( \mathbf { W } _ { 1 } ^ { ( l ) } \mathbf { h } ^ { ( l ) } ) \Big )$ , and the expert output is $\begin{array} { r } { \mathbf { h } ^ { \prime ( l ) } = \mathbf { h } ^ { ( l ) } + \sum _ { k } g _ { k } ^ { ( l ) } \cdot \mathbf { T } _ { k } ^ { ( l ) } ( \mathbf { h } ^ { ( l ) } ) } \end{array}$ . Despite being standard practice in MoE literature, this approach underperforms TALON by 3.93–17.96% in ${ \bar { \mathcal { A } } } ,$ because the limited 5 samples per class are insuficient to train reliable gating networks without severe overfitting.

These results support TALON’s use of output-level knowledge distillation: by operating on soft logits rather than directly merging parameters, EKT consolidates the task-specific Teacher knowledge into a single compact Student for eficient inference. The task-level semantic-guided coeficients emphasize Teachers relevant to the current task at each incremental stage; the same coeficient vector is shared by all samples and EKT epochs in that stage.

Table 11: Results of diferent methods on CIFAR100 (T=9) using supervised pre-training model ViT-B/16-IN21K (denoted as Sup-21K) and self-supervised model DINO
<table><tr><td rowspan=1 colspan=1>PTM</td><td rowspan=1 colspan=4>Method</td><td rowspan=1 colspan=1> $A _ { \mathrm { B a s e } }$     $\boldsymbol { \mathcal { A } } _ { L }$      A</td></tr><tr><td rowspan=12 colspan=1>Sup-21K</td><td rowspan=1 colspan=4>Full Finetune</td><td rowspan=1 colspan=1>90.97 40.83 63.89</td></tr><tr><td rowspan=1 colspan=4>SimpleFSCILL2P</td><td rowspan=1 colspan=1>83.00 71.53 76.7191.92 55.35 71.09</td></tr><tr><td rowspan=3 colspan=4>CODA-Prompt</td><td rowspan=3 colspan=1>93.2355.89</td></tr><tr><td rowspan=2 colspan=1>71</td></tr><tr><td rowspan=1 colspan=1>71.87</td></tr><tr><td rowspan=1 colspan=4>InfLoRA</td><td rowspan=1 colspan=1>94.1056.18 72.29</td></tr><tr><td rowspan=2 colspan=2>SD-LoR</td><td rowspan=1 colspan=2>D</td><td rowspan=2 colspan=1>A</td><td rowspan=2 colspan=1>93.58 78.56 84.98</td></tr><tr><td rowspan=1 colspan=2>010</td></tr><tr><td rowspan=1 colspan=4>ASP</td><td rowspan=2 colspan=1>92.3387.8389.7393.5284.57 89.02</td></tr><tr><td rowspan=1 colspan=4>SEC-prompt</td></tr><tr><td rowspan=1 colspan=4>TALON-MLP</td><td rowspan=1 colspan=1>93.5287.62 90.10</td></tr><tr><td rowspan=1 colspan=4>TALON-QV</td><td rowspan=1 colspan=1>93.43 87.4290.24</td></tr><tr><td rowspan=10 colspan=1>DINO</td><td rowspan=1 colspan=4>Full Finetune</td><td rowspan=1 colspan=1>88.27 41.36 55.61</td></tr><tr><td rowspan=1 colspan=4>SimpleFSCIL</td><td rowspan=1 colspan=1>58.82 42.49 49.46</td></tr><tr><td rowspan=1 colspan=4>L2P</td><td rowspan=1 colspan=1>85.65 48.68 64.23</td></tr><tr><td rowspan=1 colspan=4>CODA-Prompt</td><td rowspan=1 colspan=1>89.5053.73 68.86</td></tr><tr><td rowspan=1 colspan=4>InfLoRA</td><td rowspan=1 colspan=1>88.52 52.87 68.11</td></tr><tr><td rowspan=1 colspan=4>SD-LoRA</td><td rowspan=1 colspan=1>88.57 30.12 27.75</td></tr><tr><td rowspan=1 colspan=4>ASP</td><td rowspan=1 colspan=1>68.23 48.69 56.98</td></tr><tr><td rowspan=1 colspan=4>SEC-prompt</td><td rowspan=1 colspan=1>85.40 69.47 76.99</td></tr><tr><td rowspan=1 colspan=4>TALON-MLP</td><td rowspan=1 colspan=1>87.13 74.85 80.12</td></tr><tr><td rowspan=1 colspan=4>TALON-QV</td><td rowspan=1 colspan=1>87.3275.8780.76</td></tr></table>

Robustness across Pre-training Paradigms: Table 11 provides a comprehensive comparison across supervised (ViT-B/16-IN21K) and selfsupervised (DINO) pre-training. Key findings:

• Supervised (IN21K): TALON-QV achieves $\bar { A } = 9 0 . 2 4 \%$ , outperforming ASP (89.73%) and SEC-prompt (89.02%)

• Self-supervised (DINO): TALON-QV achieves $\bar { A } = 8 0 . 7 6 \%$ , significantly outperforming SEC-prompt (76.99%) and ASP (56.98%)

Notably, prompt-based methods (ASP, SEC-prompt) are more sensitive to pre-training paradigm shifts, while TALON maintains robust performance across diferent feature spaces.

![](images/d4cdc7510275271b3e2b6d83caa58daae30de32ff703f2ec991590af829829bb.jpg)  
(a) First stage  
(b) Second stage  
Figure 6: Visualization of the decision boundary on miniImageNet between two incremental tasks. Dots represent old classes, and triangles stand for new classes. Decision boundaries are shown with the shadow region.

Diferent PTMs and diferent LoRA-Teachers are further analyzed in the supplementary material.

## 5.5. Visualization of Incremental Sessions

To validate TALON’s efectiveness in the few-shot regime, we visualize feature representations from the first and second incremental stages (following the base stage) on miniImageNet using t-SNE [62], as shown in Fig. 6. Two key observations emerge: (i) TALON efectively separates instances into their corresponding classes with well-defined decision boundaries, and (ii) when extending from the first to the second stage, TALON maintains clear separation for both old classes (dots) and new classes (triangles), demonstrating efective knowledge consolidation without catastrophic forgetting. These visualizations confirm that TALON learns genuinely discriminative features rather than relying solely on increased model capacity.

## 5.6. Further Analysis

Parameter Sensitivity Analysis: TALON involves two key hyperparameters: (1) the rank r of the LoRA-Teachers, and (2) their insertion positions within the Vision Transformer (ViT).

![](images/2a224baaa279e0af9ac5f4f78e99abc3dbbe8e941e0d9e1104d0da614efb0c91.jpg)  
Figure 7: Sensitivity of hyperparameters (the rank r, the insertion positions) on CUB200 (T=11).

To examine the sensitivity of these hyperparameters, we conducted experiments on the CUB200 (T=11) benchmark. Specifically, the rank r was varied over {1, 2, 4, 8, 10, 16}, while the insertion positions were configured as {0-2, 0-4, 0-8, 0-12}, where “0-2” indicates insertion into the first two Transformer layers. The results, illustrated in Fig. 7, show that TALON delivers stable performance across a wide range of settings, indicating robustness to variations in both rank and insertion position. Although reported here on CUB200, similar trends are observed on other datasets. Based on this analysis, we adopt r = 8 and insert LoRA-Teachers into all Transformer layers (0-12) as the default configuration.

The “Diferent KD Losses” section of the supplementary material compares Logit-KL, Logit-L2, Feature-L2, Feature-CS, and feature-space-consistent Prototype Relation for both TALON-MLP and TALON-QV. Logit-KL achieves the highest combined four-dataset macro-average of 87.37%. Although several alternatives are close in individual settings, none provides a consistent improvement across datasets and variants. These results support Logit-KL as TALON’s most balanced EKT objective.

Table 12: Prototype-drift analysis and comparison of historical-prototype update strategies. $D _ { \mathrm { d r i f t } }$ is the final-session mean cumulative L2 drift; $\rho _ { \Delta }$ is the pooled drift-direction agreement; and $R _ { \mathrm { c e n t } }$ is the pooled centroid-distance reduction. $\Delta _ { \mathrm { S D C } }$ and $\Delta _ { \mathrm { R e c o m p } }$ are mean paired changes in final accuracy relative to Frozen, in percentage points. $D _ { \mathrm { d r i f t } }$ and Frozen $\boldsymbol { \mathcal { A } } _ { L }$ are reported as mean±standard deviation over class-order seeds 42, 1993, and 2025.
<table><tr><td></td><td></td><td></td><td></td><td>Frozen</td><td></td><td></td></tr><tr><td>Dataset</td><td>Variant</td><td> $D _ { \mathrm { d r i f t } }$ </td><td> $\rho _ { \Delta }$   $R _ { \mathrm { c e n t } }$ </td><td> $A _ { L } ~ ( \% )$ </td><td> $\Delta _ { \mathrm { S D C } }$ </td><td> $\Delta _ { \mathrm { R e c o m p } }$ </td></tr><tr><td>CUB200</td><td>TALON-MLP</td><td> $0 . 1 1 0 5 \pm 0 . 0 1 8 8$ </td><td>0.860 24.0%</td><td> $8 3 . 9 0 \pm 1 . 0 8 + 0 . 0 4 2$ </td><td></td><td>+0.014</td></tr><tr><td></td><td>TALON-QV</td><td> $0 . 0 6 0 5 \pm 0 . 0 2 3 7$ </td><td>0.810 18.7%</td><td> $8 3 . 8 0 \pm 0 . 8 5 - 0 . 0 4 2$ </td><td></td><td>-0.028</td></tr><tr><td>CIFAR100</td><td>TALON-MLP</td><td> $0 . 0 5 0 3 \pm 0 . 0 0 5 7$ </td><td>0.852 25.9%</td><td> $8 7 . 4 2 \pm 0 . 6 6 - 0 . 0 0 3 ~ + 0 . 0 0 3$ </td><td></td><td></td></tr><tr><td></td><td>TALON-QV</td><td> $0 . 0 1 3 0 \pm 0 . 0 0 3 2$ </td><td>0.650 5.9%</td><td> $8 7 . 4 8 \pm 0 . 2 3 + 0 . 0 3 0 \ \mathrm { ~ + 0 . 0 2 0 }$ </td><td></td><td></td></tr><tr><td>ImageNet-R</td><td>TALON-MLP</td><td> $0 . 1 3 0 5 \pm 0 . 0 2 7 9$ </td><td>0.874 25.7%</td><td></td><td> $7 3 . 5 9 \pm 0 . 8 0 + 0 . 0 7 8$ </td><td>-0.044</td></tr><tr><td></td><td>TALON-QV</td><td> $0 . 0 6 6 6 \pm 0 . 0 0 4 0$ </td><td>0.766 11.4%</td><td> $7 3 . 6 6 \pm 1 . 2 1 - 0 . 0 1 7 \ - 0 . 0 2 2$ </td><td></td><td></td></tr><tr><td></td><td>miniImageNet TALON-MLP</td><td> $0 . 0 1 4 3 \pm 0 . 0 0 8 4$ </td><td>0.841 24.6%</td><td> $9 5 . 3 0 \pm 0 . 2 3 - 0 . 0 1 0 \ \mathrm { ~ + 0 . 0 1 3 }$ </td><td></td><td></td></tr><tr><td></td><td>TALON-QV</td><td> $0 . 0 0 2 1 5 \pm 0 . 0 0 0 6 0 0 . 6 2 7$ </td><td>3.5%</td><td> $9 5 . 4 4 \pm 0 . 3 6 + 0 . 0 0 7 \ + 0 . 0 1 0$ </td><td></td><td></td></tr></table>

## 5.7. Prototype–Feature Consistency Across Incremental Stages

For a class c introduced at session $j ,$ , let $\mathbf { P } _ { \mathrm { S t u } } ^ { ( j ) } [ c ]$ denote its stored prototype and $\widetilde { \mathbf { P } } _ { \mathrm { S t u } } ^ { ( t ) } [ c ]$ the centroid recomputed from the same historical samples using the current Student $\mathbf { S } _ { t }$ . Historical samples are accessed only for post-hoc drift measurement and prototype recomputing; they are never used by TALON or SDC.

Following SDC [63], we estimate historical-class displacement using currenttask features before and after the Student update:

$$
\begin{array} { r } { \mathbf { z } _ { i , t } ^ { - } = \mathrm { n o r m } ( \phi ( \mathbf { x } _ { i } ; \mathbf { S } _ { t - 1 } ) ) , \qquad \mathbf { z } _ { i , t } ^ { + } = \mathrm { n o r m } ( \phi ( \mathbf { x } _ { i } ; \mathbf { S } _ { t } ) ) . } \end{array}\tag{21}
$$

The historical prototype is recursively updated as

$$
\begin{array} { r l } & { w _ { i , c } ^ { ( t ) } = \exp \left( - \frac { \left\| \mathbf { z } _ { i , t } ^ { - } - \mathbf { P } _ { \mathrm { S D C } } ^ { ( t - 1 ) } [ c ] \right\| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , } \\ & { \mathbf { P } _ { \mathrm { S D C } } ^ { ( t ) } [ c ] = \mathbf { P } _ { \mathrm { S D C } } ^ { ( t - 1 ) } [ c ] + \frac { \sum _ { ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } _ { t } } w _ { i , c } ^ { ( t ) } \left( \mathbf { z } _ { i , t } ^ { + } - \mathbf { z } _ { i , t } ^ { - } \right) } { \sum _ { ( \mathbf { x } _ { i } , y _ { i } ) \in \mathcal { D } _ { t } } w _ { i , c } ^ { ( t ) } } . } \end{array}\tag{22}
$$

Here, $\sigma = 0 . 2 0$ and $\mathbf { P } _ { \mathrm { S D C } } ^ { ( j ) } [ c ] = \mathbf { P } _ { \mathrm { S t u } } ^ { ( j ) } [ c ]$

The true and SDC-estimated cumulative drift vectors are

$$
\Delta _ { c } ^ { j  t } = \widetilde { \bf P } _ { \mathrm { S t u } } ^ { ( t ) } [ c ] - { \bf P } _ { \mathrm { S t u } } ^ { ( j ) } [ c ] , \qquad \widehat { \Delta } _ { c } ^ { j  t } = { \bf P } _ { \mathrm { S D C } } ^ { ( t ) } [ c ] - { \bf P } _ { \mathrm { S t u } } ^ { ( j ) } [ c ] .\tag{23}
$$

We report

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { d r i f t } } = \frac { 1 } { | \Omega _ { L } | } \sum _ { ( s , c , j , L ) \in \Omega _ { L } } \| \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } _ { s , c } ^ { j  L } \| _ { 2 } } , } \\ { { \displaystyle \rho _ { \Delta } = \frac { 1 } { | \Omega | } \sum _ { ( s , c , j , t ) \in \Omega } \frac {  \widehat { \mathbf { \Delta } } \widehat { \mathbf { \Delta } } _ { s , c } ^ { j  t } , \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { j  t }  } { \| \widehat { \mathbf { \Delta } } \widehat { \mathbf { \Delta } } _ { s , c } ^ { j  t } \| _ { 2 } \| \mathbf { \Delta } \mathbf { \Delta } _ { s , c } ^ { j  t } \| _ { 2 } + \varepsilon } } . } \end{array}\tag{24}
$$

Here, s indexes the class-order seed, Ω contains all valid historical-class/session/seed records, and $\Omega _ { L }$ denotes its final-session subset.

The relative reduction in prototype-to-current-centroid distance is

$$
R _ { \mathrm { c e n t } } = 1 - \frac { \sum _ { ( s , c , j , t ) \in \Omega } \left\| \mathbf { P } _ { \mathrm { S D C } , s } ^ { ( t ) } [ c ] - \widetilde { \mathbf { P } } _ { \mathrm { S t u } , s } ^ { ( t ) } [ c ] \right\| _ { 2 } } { \sum _ { ( s , c , j , t ) \in \Omega } \left\| \mathbf { P } _ { \mathrm { S t u } , s } ^ { ( j ) } [ c ] - \widetilde { \mathbf { P } } _ { \mathrm { S t u } , s } ^ { ( t ) } [ c ] \right\| _ { 2 } } .\tag{25}
$$

Within each run, we fix the Student checkpoint, test features, class order, and cosine nearest-prototype classifier, and change only the historical prototype entries. We compare Frozen, SDC, and Historical-Data Prototype Recomputing. The last strategy recomputes each historical prototype from its original class samples using the current Student and is included only as a post-hoc diagnostic.

Table 12 shows that SDC consistently tracks and reduces the measured displacement: $\rho _ { \Delta }$ ranges from 0.627 to 0.874, and $R _ { \mathrm { c e n t } }$ ranges from 3.5% to 25.9%. Crucially, TALON’s predictions remain stable despite this geometric drift. SDC changes final accuracy by only −0.042 to +0.078 percentage points, while Historical-Data Prototype Recomputing changes it by −0.044 to +0.020 points. Across all 24 runs, neither strategy produces a significant final-session McNemar result.

These findings distinguish measurable feature-space displacement from decision-level degradation. Although historical prototypes drift relative to the current Student space, TALON’s cosine nearest-prototype classifier remains robust. We therefore retain Frozen, which preserves the strict exemplarfree protocol without sacrificing meaningful classification performance.

## 5.8. Historical-Knowledge Preservation During EKT

To directly quantify historical-knowledge preservation during EKT, we record historical-task accuracy immediately before and after each EKT update. At stage t, we define

$$
\Delta _ { \mathrm { o l d } } ^ { ( t ) } = \frac { 1 } { t } \sum _ { j = 0 } ^ { t - 1 } \left( a _ { t , j } ^ { \mathrm { a f t e r } } - a _ { t , j } ^ { \mathrm { b e f o r e } } \right) ,\tag{26}
$$

The mean historical-task changes after Logit-KL EKT are +0.05, −0.00, and −0.12 percentage points on CUB200, CIFAR100, and ImageNet-R, respectively, showing that historical-task performance remains essentially unchanged on average. Comparisons with alternative EKT targets are provided in the “Diferent KD Losses” section of the supplementary material.

## 6. Conclusion

In this work, we introduced TALON, a novel framework addressing the core challenges of Few-Shot Class-Incremental Learning (FSCIL) through task-adaptive LoRA-Teacher modules and ensemble knowledge transfer. TALON efectively balances the stability-plasticity trade-of by dynamically allocating a dedicated LoRA-Teacher for each incremental task while consolidating their collective knowledge into a unified LoRA-Student model for eficient inference.

The key contributions of TALON are:

• Dynamic multi-expert paradigm: Eliminates the need for predefined module counts and achieves superior plasticity through complete parameter isolation;

• Distills knowledge from multiple teachers into a single student model, removing computational overhead during inference;

• Semantic-guided distillation: Enables optimal integration of teacher knowledge while mitigating catastrophic forgetting and overfitting.

Extensive experiments over three class-order runs show that TALON achieves comparable or better mean average accuracy across four benchmark datasets. TALON maintains clear mean improvements on CIFAR100, ImageNet-R, and miniImageNet, while obtaining performance comparable to SEC-prompt on the more class-order-sensitive CUB200 benchmark. In addition, TALON retains its parameter-eficient design and fast single-model inference. The expanded ablation studies further show limited sensitivity to the semantic temperature, soft-distillation weighting, and EKT duration.

Looking ahead, TALON’s dynamic adaptation and eficient knowledge consolidation provide promising directions for future incremental learning research, especially in scenarios demanding both sample eficiency and computational eficiency.

## Acknowledgments

This work is supported in part by the National Natural Science Foundation of China (Grant No. 62372054, 62006005) and National Key Research and Development Program of China (Grant No. 2022YFC3302200).

## References

[1] R. M. French, Catastrophic forgetting in connectionist networks, Trends in Cognitive Sciences 3 (4) (1999) 128–135. doi:10.1016/ S1364-6613(99)01294-2.

[2] R. M. French, A. Ferrara, Modeling time perception in rats: Evidence for catastrophic interference in animal learning, in: Proceedings of the Twenty-first Annual Conference of the Cognitive Science Society, 2020, pp. 173–178. doi:10.4324/9781410603494-35.

[3] M. D’Alessandro, A. Alonso, E. Calabr´es, M. Galar, Multimodal parameter-eficient few-shot class incremental learning, in: 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2023, pp. 3385–3395. doi:10.1109/ICCVW60793.2023. 00364.

[4] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, Y. Gong, Few-shot class-incremental learning, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 12180–12189. doi:10.1109/CVPR42600.2020.01220.

[5] C. Liu, Z. Wang, T. Xiong, R. Chen, Y. Wu, J. Guo, H. Huang, Few-shot class incremental learning with attention-aware self-adaptive

prompt, in: Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXI, Springer Nature Switzerland, Cham, 2024, pp. 1–18. doi: 10.1007/978-3-031-73004-7\_1.

[6] S. T. Grossberg, Studies of mind and brain: Neural principles of learning, perception, development, cognition, and motor control 70 (2012). doi:10.1007/978-94-009-7758-7.

[7] C. Zhou, Q. Li, C. Li, J. Yu, Y. Liu, G. Wang, K. Zhang, C. Ji, Q. Yan, L. He, et al., A comprehensive survey on pretrained foundation models: A history from bert to chatgpt, International Journal of Machine Learning and Cybernetics (2024). doi:10.1007/s13042-024-02443-6.

[8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, Imagenet: A large-scale hierarchical image database, in: 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255. doi: 10.1109/CVPR.2009.5206848.

[9] Y. Xin, S. Luo, H. Zhou, J. Du, X. Liu, Y. Fan, Q. Li, Y. Du, Parametereficient fine-tuning for pre-trained vision models: A survey, CoRR abs/2402.02242 (2024). doi:10.48550/ARXIV.2402.02242.

[10] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, T. Pfister, Learning to prompt for continual learning, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 139–149. doi:10.1109/CVPR52688. 2022.00024.

[11] J. S. Smith, L. Karlinsky, V. Gutta, P. Cascante-Bonilla, D. Kim, A. Arbelle, R. Panda, R. Feris, Z. Kira, Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning, in: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 11909–11919. doi:10.1109/CVPR52729.2023. 01146.

[12] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, Lora: Low-rank adaptation of large language models, in: The Tenth International Conference on Learning Representations, ICLR

2022, Virtual Event, April 25-29, 2022, 2022. doi:10.48550/arXiv. 2106.09685.

[13] Q. Gao, C. Zhao, Y. Sun, T. Xi, G. Zhang, B. Ghanem, J. Zhang, A unified continual learning framework with general parameter-eficient tuning, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 11449–11459. doi:10.1109/ICCV51070.2023.01055.

[14] Y.-S. Liang, W.-J. Li, Inflora: Interference-free low-rank adaptation for continual learning, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 23638–23647. doi:10. 1109/CVPR52733.2024.02231.

[15] Y. Wu, H. Piao, L. Huang, R. Wang, W. Li, H. Pfister, D. Meng, K. Ma, Y. Wei, Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning, in: The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, 2025. doi:10.48550/arXiv.2501.13198.

[16] Y. Liu, M. Yang, Sec-prompt:semantic complementary prompting for few-shot class-incremental learning, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 25643– 25656. doi:10.1109/CVPR52734.2025.02388.

[17] Z. Li, D. Hoiem, Learning without forgetting, IEEE Transactions on Pattern Analysis and Machine Intelligence 40 (12) (2018) 2935–2947. doi:10.1109/TPAMI.2017.2773081.

[18] L. Wang, X. Zhang, H. Su, J. Zhu, A comprehensive survey of continual learning: Theory, method and application, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (8) (2024) 5362–5383. doi:10.1109/TPAMI.2024.3367329.

[19] Z. Qiu, L. Xu, Z. Wang, Q. Wu, F. Meng, H. Li, ISM-Net: Mining incremental semantics for class incremental learning, Neurocomputing 523 (2023) 130–143. doi:10.1016/j.neucom.2022.12.029.

[20] X. Xi, G. Cao, W. Cao, Y. Liu, Y. Li, H. Wang, H. Ren, Potential knowledge extraction network for class-incremental learning, Neurocomputing 616 (2025) 128923. doi:10.1016/j.neucom.2024.128923.

[21] G. E. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network, CoRR abs/1503.02531 (2015). doi:10.48550/arXiv.1503. 02531.

[22] S. Li, T. Su, X.-Y. Zhang, Z. Wang, Continual learning with knowledge distillation: A survey, IEEE Transactions on Neural Networks and Learning Systems 36 (6) (2025) 9798–9818. doi:10.1109/TNNLS.2024. 3476068.

[23] P. Dhar, R. V. Singh, K.-C. Peng, Z. Wu, R. Chellappa, Learning without memorizing, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5133–5141. doi:10.1109/CVPR.2019.00528.

[24] N. Asadi, M. Davari, S. P. Mudur, R. Aljundi, E. Belilovsky, Prototypesample relation distillation: Towards replay-free continual learning, in: International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, 2023. doi:10.48550/arXiv.2303.14771.

[25] S. Kullback, R. A. Leibler, On information and suficiency, The Annals of Mathematical Statistics 22 (1) (1951) 79–86. doi:10.1214/aoms/ 1177729694.

[26] J. Pomponi, A. Devoto, S. Scardapane, Class incremental learning with probability dampening and cascaded gated classifier, Neurocomputing 640 (2025) 130295. doi:10.1016/j.neucom.2025.130295.

[27] S.-A. Rebufi, A. Kolesnikov, G. Sperl, C. H. Lampert, icarl: Incremental classifier and representation learning, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5533– 5542. doi:10.1109/CVPR.2017.587.

[28] S. Hou, X. Pan, C. C. Loy, Z. Wang, D. Lin, Lifelong learning via progressive distillation and retrospection, in: Proceedings of the European Conference on Computer Vision (ECCV), Springer-Verlag, Berlin, Heidelberg, 2018, p. 452–467. doi:10.1007/978-3-030-01219-9\_27.

[29] K. Lee, K. Lee, J. Shin, H. Lee, Overcoming catastrophic forgetting with unlabeled data in the wild, in: 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 312–321. doi:10.1109/ICCV.2019.00040.

[30] A. Douillard, M. Cord, C. Ollion, T. Robert, E. Valle, Podnet: Pooled outputs distillation for small-tasks incremental learning, in: Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX, Springer-Verlag, Berlin, Heidelberg, 2020, p. 86–102. doi:10.1007/978-3-030-58565-5\_6.

[31] L. Xiong, X. Guan, H. Xiong, K. Zhu, F. Zhang, Knowledge fusion distillation and gradient-based data distillation for class-incremental learning, Neurocomputing 622 (2025) 129286. doi:10.1016/j.neucom. 2024.129286.

[32] F. Zhu, X.-Y. Zhang, C. Wang, F. Yin, C.-L. Liu, Prototype augmentation and self-supervision for incremental learning, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 5867–5876. doi:10.1109/CVPR46437.2021.00581.

[33] W. Shi, M. Ye, Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 1772–1781. doi:10.1109/ICCV51070.2023.00170.

[34] M. Toldo, M. Ozay, Bring evanescent representations to life in lifelong class incremental learning, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16711–16720. doi:10.1109/CVPR52688.2022.01623.

[35] X. Chen, Z. Wang, Uni-DTR: A unified framework for balancing knowledge preservation and adaptation in exemplar-free class-incremental learning, Neurocomputing 686 (2026) 133722. doi:10.1016/j.neucom. 2026.133722.

[36] J. He, C. Zhou, X. Ma, T. Berg-Kirkpatrick, G. Neubig, Towards a unified view of parameter-eficient transfer learning, in: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022. doi:10.48550/arXiv.2110.04366.

[37] Z. Wang, Z. Zhang, S. Ebrahimi, R. Sun, H. Zhang, C.-Y. Lee, X. Ren, G. Su, V. Perot, J. Dy, T. Pfister, Dualprompt: Complementary prompting for rehearsal-free continual learning, in: Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022,

Proceedings, Part XXVI, Springer-Verlag, Berlin, Heidelberg, 2022, p. 631–648. doi:10.1007/978-3-031-19809-0\_36.

[38] Y. Wang, Z. Huang, X. Hong, S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning, in: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. doi:10.48550/arXiv.2207.12819.

[39] Y. Li, H. Zhu, J. Ma, C. Xiang, P. Vadakkepat, Incremental few-shot learning via implanting and consolidating, Neurocomputing 559 (2023) 126800. doi:10.1016/j.neucom.2023.126800.

[40] D. Kim, D. Han, J. Seo, J. Moon, Warping the space: Weight space rotation for class-incremental few-shot learning, in: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023.

[41] C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, Y. Xu, Few-shot incremental learning with continually evolved classifiers, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 12450–12459. doi:10.1109/CVPR46437.2021.01227.

[42] K.-H. Park, K. Song, G.-M. Park, Pre-trained vision and language transformers are few-shot incremental learners, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 23881–23890. doi:10.1109/CVPR52733.2024.02254.

[43] X. Yue, Y. Chen, X. Zhang, X. Gao, M. Feng, M. Lao, H. Zhuang, H. Li, PAL: Prompting analytic learning with missing modality for multi-modal class-incremental learning, Pattern Recognition 179 (2026) 113467. doi:10.1016/j.patcog.2026.113467.

[44] X. Zhang, C. Zhang, Z. Li, X. Wang, S. Cai, M. Lao, Y. Guo, H. Zhuang, Rep deep & machine learning: Exemplar-free continual video action recognition via slow-fast collaborative learning, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 36075– 36083. doi:10.1609/aaai.v40i42.40924.

[45] Z. Li, X. Zhang, Y. Guo, S. Cai, M. Lao, PENCIL: Prototype-enhanced compositional learning for class-incremental hand gesture recognition, IEEE Transactions on Consumer Electronics (2025). doi:10.1109/TCE. 2025.3569912.

[46] X. Zhang, P. Zhu, J. Sui, X. Yang, J. Tian, M. Lao, S. Cai, Y. Guo, J. Tang, Choose your expert: Uncertainty-guided expert selection for continual deepfake detection, in: Proceedings of the 33rd ACM International Conference on Multimedia, ACM, 2025, pp. 11502–11511. doi:10.1145/3746027.3755398.

[47] Y. Zhao, L. Zhao, S. Zhang, K. Ji, G. Kuang, L. Liu, Azimuthaware subspace classifier for few-shot class-incremental SAR ATR, IEEE Transactions on Geoscience and Remote Sensing 62 (2024) 1–20. doi: 10.1109/TGRS.2024.3354800.

[48] Y. Xu, H. Sun, C. Liu, K. Ji, G. Kuang, Physical attributes embedded prototypical network for incremental SAR automatic target recognition, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 19 (2026) 4319–4336. doi:10.1109/JSTARS.2025. 3650513.

[49] L. Wang, J. Xie, X. Zhang, M. Huang, H. Su, J. Zhu, Hierarchical decomposition of prompt-based continual learning: Rethinking obscured suboptimality, in: A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine (Eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. doi:10.52202/075280-3022.

[50] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021. doi:10.48550/arXiv.2010.11929.

[51] K. He, X. Zhang, S. Ren, J. Sun, Delving deep into rectifiers: Surpassing human-level performance on imagenet classification, in: 2015

IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026–1034. doi:10.1109/ICCV.2015.123.

[52] J. Zhu, J. Liu, W. Li, J. Lai, X. He, L. Chen, Z. Zheng, Ensembled ctr prediction via knowledge distillation, in: Proceedings of the 29th ACM International Conference on Information & Knowledge Management, CIKM ’20, Association for Computing Machinery, New York, NY, USA, 2020, p. 2941–2958. doi:10.1145/3340531.3412704.

[53] D.-W. Zhou, H.-L. Sun, H.-J. Ye, D.-C. Zhan, Expandable subspace ensemble for pre-trained model-based class-incremental learning, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 23554–23564. doi:10.1109/CVPR52733.2024. 02223.

[54] X. Wang, Z. Ji, Y. Yu, Y. Pang, J. Han, Model attention expansion for few-shot class-incremental learning, IEEE Transactions on Image Processing 33 (2024) 4419–4431. doi:10.1109/TIP.2024.3434475.

[55] C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie, The caltechucsd birds-200-2011 dataset (2011).

[56] A. Krizhevsky, G. Hinton, Learning multiple layers of features from tiny images (2009).

[57] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, et al., The many faces of robustness: A critical analysis of out-of-distribution generalization, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8340–8349.

[58] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. S. Bernstein, A. C. Berg, L. Fei-Fei, Imagenet large scale visual recognition challenge, Int. J. Comput. Vis. 115 (3) (2015) 211–252. doi:10.1007/S11263-015-0816-Y.

[59] D.-W. Zhou, Z.-W. Cai, H.-J. Ye, D.-C. Zhan, Z. Liu, Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need, International Journal of Computer Vision 133 (3) (2024) 1012–1032. doi:10.1007/s11263-024-02218-0.

[60] S. Tian, L. Li, W. Li, H. Ran, X. Ning, P. Tiwari, A survey on fewshot class-incremental learning, Neural Netw. 169 (C) (2024) 307–324. doi:10.1016/j.neunet.2023.10.039.

[61] D. Lopez-Paz, M. Ranzato, Gradient episodic memory for continual learning, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 6467–6476. doi:10.48550/ arXiv.1706.08840.

[62] L. Van der Maaten, G. Hinton, Visualizing data using t-sne., Journal of machine learning research 9 (11) (2008).

[63] L. Yu, B. Twardowski, X. Liu, L. Herranz, K. Wang, Y. Cheng, S. Jui, J. van de Weijer, Semantic drift compensation for class-incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 6982–6991. doi:10.1109/ CVPR42600.2020.00701.