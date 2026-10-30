# Class-aware Latent Attention Replay (CLAR): Class Incremental Hazard Diagnosis for Resilient AI in the Manufacturing Industrial Internet

Avranshu Mahanty<sup>a</sup>, Ran Jin<sup>b</sup>, Yingyan Zeng<sup>a,\*</sup>

<sup>a</sup>Department of Mechanical, Materials, and Industrial Engineering, University of Cincinnati, Cincinnati, OH <sup>b</sup>Department of Industrial and Systems Engineering, Virginia Tech, Blacksburg, VA

## Abstract

Artificial Intelligence (AI) systems provide critical computation services in the Manufacturing Industrial Internet (MII), and their resilience depends on timely diagnosis of hazards across the data, AI pipeline, and cyber-physical layers. During operation, new hazard types keep emerging with only a few samples each, hazards co-occur, and the diagnosis model runs on memory-constrained computation nodes. Replay-based class-incremental learning (CIL) can learn new hazard classes while revisiting a small memory of past samples, but existing methods assume single-label, balanced streams, store high-dimensional raw samples, and select them independently of the classifier. We propose Class-aware Latent Attention Replay (CLAR), which combines a dual bufer that stores mostly compact latent representations with a few raw samples, an Attention Leverage Score (ALS) that upper-bounds the classifier logit and guides class-balanced reservoir sampling, and an exponential-moving-average teacher that slows latentspace drift. For an idealized bufer model, we prove that the bufer is class-balanced and follows an ALS-weighted class distribution, and that below a critical per-class capacity, ALS-guided replay has a lower replay-gradient error than class-balanced uniform replay. In the case study, On hazard diagnosis data from an MII testbed, CLAR achieves the highest mean average weighted F1 score among the evaluated replay methods with the same bufer size.

## 1 Introduction

The Manufacturing Industrial Internet (MII) connects manufacturing equipment, physical processes, systems, and networks through ubiquitous sensors, actuators, and computing units [1, 2]. The data collected via the MII enable AI models that deliver computation services such as quality modeling, process variation analysis, and fault diagnosis [3]. In this context, an AI system consists of three interconnected layers [2, 4]: (i) the data layer, i.e., the manufacturing data collected via the MII; (ii) the AI pipeline layer, where the data are processed by deployed AI pipelines, such as data augmentation, standardization, and deep neural network (DNN) classifiers [5]; and (iii) the cyber-physical layer, i.e., the Fog–Cloud computing infrastructure that supports communication and computation. Each execution of an AI pipeline on a batch of data at a computation node constitutes a computation task, e.g., predicting the quality of the products produced by one machine.

Manufacturing decisions, such as accepting a printed part or adjusting a process setting, rely on the outputs of these computation tasks, so the AI system has to deliver reliable computation services continuously throughout production. Hazards in any of the three layers threaten this reliability. Sensor contamination and low signal-to-noise ratio corrupt the data layer, model singularity destabilizes the AI pipeline layer, and fog node or communication failures disrupt the cyber-physical layer [2]. The resulting performance degradation leads to incorrect quality predictions, faulty process decisions, and economic losses. AI resilience refers to the ability of the AI system to withstand, respond to, and recover from such hazards [2, 6]. Recovery requires a mitigation action that matches the root cause, so accurate root-cause diagnosis is the prerequisite for resilience. Diagnosis can use information that the MII already collects after each computation task: the supervised learning performance of the deployed pipelines and the runtime metrics of the computation node, such as CPU utilization and memory consumption [2]. Hazard diagnosis therefore becomes a multimodal, multi-label classification problem over these two modalities.

Diagnosis models of this kind are trained ofline on a fixed set of hazard types. In operation, the MII keeps changing: new machines, process settings, and software updates introduce hazard types that were absent from the training data, and known hazard types reappear later. A resilient AI system thus needs a diagnosis model that keeps learning from the stream of computation tasks, which raises five challenges. (i) New hazard classes emerge over time and previously seen classes recur, and retraining from scratch on all historical data is computationally prohibitive. (ii) Normal operation dominates the stream, so each hazard class has only a few samples. (iii) Several hazards can afect the same computation task, making the problem multi-label. (iv) The runtime metrics are high-dimensional multivariate time series, which, together with the tabular performance metrics, form multimodal samples. (v) The diagnosis model runs on resourceconstrained computation nodes, which strictly limits its memory.

Class-Incremental Learning (CIL) addresses the first challenge by updating a model with new classes as they appear [7,8]. Its central dificulty is catastrophic forgetting: training on new classes overwrites the knowledge of earlier classes and degrades their accuracy [9]. Replay-based methods counter forgetting by storing a small memory of past samples and revisiting them while learning new classes [10, 11]. Three limitations restrict their use for hazard diagnosis in the MII. First, most methods assume single-label streams with balanced and suficient samples per class, while hazard data are multi-label, highly imbalanced, and scarce. Second, they store past samples in raw form; a raw sample in our case study contains about $2 . 5 \times 1 0 ^ { 6 }$ values, so a fixed memory budget holds only a few exemplars per hazard class. Third, they select samples to store by random, class-count, gradient, or class-mean criteria, which are independent of the classspecific parameters that the classifier uses to make its decisions.

To address these limitations, we propose Class-aware Latent Attention Replay (CLAR), a class-incremental hazard diagnosis framework for AI systems in the MII. The key idea of CLAR is to keep the memory of past hazards predominantly in the compact latent space of the diagnosis model and to let the class-specific attention parameters of the classifier decide which samples are worth remembering. The contributions of this work are as follows.

• We formulate class-incremental hazard diagnosis for AI systems in the MII as a multi-label, multimodal CIL problem over a sequential stream of tasks with recurring classes, severe class imbalance, and a strict memory budget.

• We propose a latent-dominant dual replay bufer that stores many compact latent representations and a few raw samples per hazard class, which multiplies the number of bufer slots per class within the same memory budget.

• We propose the Attention Leverage Score (ALS), an attention-induced Mahalanobis-type norm that upperbounds the magnitude of the classifier logit, and use it to tilt class-balanced reservoir sampling toward samples that strongly drive the classifier under imbalanced, multi-label streams.

• We show that ALS-guided sampling keeps a classbalanced bufer whose content follows an ALS-weighted class distribution, decompose its replay-gradient error into sampling variance and selection bias, and characterize, for an idealized model, the per-class capacities below which it has a lower replay-gradient error and a smaller upper bound on the expected squared margin deviation than class-balanced uniform replay.

• We evaluate CLAR on hazard data from an MII testbed, where it achieves the highest mean average weighted F1 score among the evaluated methods with

a smaller configured bufer.

The remainder of this paper is organized as follows. Section 2 reviews CIL, replay-based CIL, and few-shot CIL. Section 3 formulates the problem and presents CLAR. Section 4 analyzes ALS-guided sampling theoretically. Section 5 reports the case study, and Section 6 concludes the paper.

## 2 Related Work

## 2.1 Class-Incremental Learning

Continual learning trains a model on a sequence of tasks, where each task provides a new set of training data and the data of earlier tasks become unavailable [7]. CIL is the setting of continual learning in which new tasks introduce new classes and the model has to classify a test sample among all classes seen so far, without being told which task the sample comes from [8, 12]. This setting is the most challenging one, because the model has to separate classes that were never trained together, in addition to learning each task. It matches applications where new categories emerge during deployment, such as new defect types in visual inspection or new fault types in machinery diagnosis [13].

Two related phenomena make CIL dificult. Catastrophic forgetting occurs when gradient updates for new classes change the parameters that encode earlier classes, so the accuracy on earlier classes drops sharply [9]. Avoiding it requires stability, whereas learning new classes requires plasticity, and the trade-of between the two is known as the stability–plasticity dilemma [14]. At the output layer, forgetting appears as task-recency bias: since the training data of each task are dominated by the newest classes, the classifier assigns higher scores to recent classes and misclassifies samples of earlier classes as recent ones [15, 16]. Practical streams add further dificulties. In online CIL, each sample is observed once in a single pass [17]. In blurry CIL, classes recur across tasks [18]; in manufacturing, for example, a sensor fault can reappear after maintenance or when a process setting is reused. In multi-label CIL, samples carry several labels and the label distribution is long-tailed [19].

CIL methods fall into three families [7]. Architecturebased methods expand the network for each new task and freeze or isolate the parameters of earlier tasks [20, 21]. Since the parameters that encode earlier classes are left untouched by later updates, forgetting is avoided by construction. The model size, however, grows with the number of tasks, which becomes prohibitive on memoryconstrained computation nodes. Regularization-based methods add a penalty to the training loss that discourages changes to the knowledge of earlier tasks. The penalty can act on parameters, e.g., a squared distance between the current and previous parameters weighted by their estimated importance for earlier tasks [22, 23], or on outputs, e.g., a distillation loss between the predictions of the current model and of a frozen copy, acting as a teacher, on the new data [24, 25]. Parameter penalties accumulate over tasks and progressively restrict the capacity to learn new classes, and output penalties are only as informative as the new data are about earlier classes, so their efect weakens under large distribution shifts between tasks. Both variants train on new-class data alone, which leaves task-recency bias unaddressed [8,15]. Replay-based methods, the third family, address this gap by revisiting samples of earlier classes; as they are the basis of this work, we review them separately.

![](images/b33da977ad6db37f0e815311a9d535886a2995d9385ee035a635020d12deeae1.jpg)  
Figure 1: Overview of the incremental diagnosis model, MII testbed, and the CLAR framework.

## 2.2 Replay-Based Class-Incremental Learning

Replay-based methods maintain a memory bufer, a fixedcapacity store of past samples. When the model learns a new task, samples drawn from the bufer are trained together with the new data, so the training objective approximates the objective over all classes seen so far and the classifier keeps observing earlier classes [10, 11]. The design of a replay method answers two questions: which samples to store (memory update) and which stored samples to replay (memory retrieval) [17].

For memory update, Experience Replay (ER) [10] uses reservoir sampling [26], which keeps a uniform random subset of the stream by admitting the n-th sample with probability proportional to $1 / n$ . Under an imbalanced stream, a uniform subset inherits the imbalance and minority classes nearly vanish from the bufer. Class-Balancing Reservoir Sampling (CBRS) [27] therefore applies reservoir sampling within each class and reserves equal space for every class, and partitioning reservoir sampling extends this idea to multi-label streams [19]. These strategies choose samples at random within a class. Informativeness-driven strategies select samples by gradient diversity [28], by diversity and uncertainty [18], or by proximity to the class mean in the feature space, as in the herding selection [29] of iCaRL [11]. Feature-space criteria become unreliable as the feature extractor is fine-tuned on new classes, since the embedding of stored samples drifts. For memory retrieval, Maximally Interfered Retrieval (MIR) [30] replays the stored samples whose loss would increase most under the upcoming update, and ASER [31] ranks stored samples by adversarial Shapley values.

Most replay methods store raw inputs. Latent replay stores intermediate representations of a network, which are much smaller and therefore fit more exemplars into the same memory [32, 33]. To keep the stored representations valid, these methods freeze or slowly update the layers below the replay layer, which limits the adaptation of the feature extractor to new classes. In summary, existing replay methods select samples independently of the classspecific parameters of the classifier, mostly assume singlelabel streams, and trade memory eficiency against feature adaptation.

## 2.3 Few-Shot Class-Incremental Learning

When new classes arrive with only a handful of labeled samples, CIL becomes Few-Shot Class-Incremental Learning (FS-CIL) [34]. Industrial faults are a typical example, since hazards and defects occur rarely compared with normal operation. Prototype-based approaches compute a center, or prototype, for each class from its few samples and classify new samples by their distance to the prototypes [35, 36]. Feature-space-based approaches reserve room in the embedding space for future classes with virtual prototypes [37], which provides forward compatibility, i.e., new class prototypes can be added with little disturbance to the existing representations. For industrial fault diagnosis, Forward-Back Compatible Representation (FBCR) [13] combines such feature-space reservation with replay. All these approaches depend on a stable feature extractor: once the representation space drifts during incremental updates, the prototypes and reserved regions become outdated and the diagnosis accuracy degrades.

## 3 Methodology

This section formalizes class-incremental hazard diagnosis for an AI system in the MII (Section 3.1), states the key idea of CLAR (Section 3.2), describes the diagnosis model that CLAR updates over time (Section 3.3), and presents the components of CLAR and its training procedure (Sections 3.4–3.7).

## 3.1 Problem Setting

Samples and labels. For the AI system defined in Section 1, each completed computation task yields one diagnosis sample with two modalities [2]. The supervised learning performance of the deployed pipelines on the processed batch, such as accuracy, precision, and F1 score, is reported to the Cloud as a tabular vector $X _ { i , t } ^ { ( P ) } \in \mathbb { R } ^ { \dot { P } }$ . The runtime metrics recorded by the computation node during execution, such as CPU utilization, memory consumption, and data transmission volume, form a multivariate time series $X _ { i , t } ^ { ( R ) } \in \mathbb { R } ^ { N \times R \times Q }$ with N time stamps, R runtime metrics, and $Q$ channels. The raw sample with index i in task t is

$$
X _ { i , t } = \{ X _ { i , t } ^ { ( P ) } , X _ { i , t } ^ { ( R ) } \} .\tag{3.1}
$$

These modalities are complementary: hazards in the data and AI pipeline layers mainly alter the supervised learning performance, whereas resource-related hazards mainly alter the runtime behavior. Each hazard class $c \in \{ 1 , \ldots , C \}$ corresponds to one root cause factor of AI system failure [2], such as sensor contamination or pipeline singularity. Since hazards can co-occur, the label of $X _ { i , t }$ is a binary vector $y _ { i } = ( y _ { i , 1 } , \ldots , y _ { i , C } ) \in \{ 0 , 1 \} ^ { C }$ where $y _ { i , c } = 1$ indicates the presence of hazard class c and $y _ { i } = \mathbf { 0 }$ indicates normal operation.

Tasks and incremental classes. The samples arrive as a sequential data stream $\mathcal { D } = \{ \mathcal { D } _ { 0 } , \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { T } \}$ , where each task $\mathcal { D } _ { t }$ collects the samples of one operational phase, $\mathcal { D } _ { 0 }$ is the base task, and $\mathcal { D } _ { t }$ with $t > 0$ is an incremental task. Let $\mathcal { Y } _ { t } = \{ c : y _ { i , c } = 1$ for some $X _ { i , t } \in \mathcal { D } _ { t } \big \}$ be the hazard classes present in task $\textstyle t , { \mathcal { C } } _ { t } = \bigcup _ { k = 0 } ^ { t } { \mathcal { Y } } _ { k }$ the classes seen up to task t, and $\mathcal { C } _ { \mathrm { n e w } } = \mathcal { C } _ { t } \setminus \mathcal { C } _ { t - 1 }$ the new classes introduced by task t. A task can introduce zero or more new classes, and classes can recur, so in general $\mathcal { V } _ { k } \cap \mathcal { V } _ { t } \neq \emptyset$ for $k \neq t .$

Each task is partitioned into disjoint training and test sets, $\mathcal { D } _ { t } = \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } \cup \mathcal { D } _ { t } ^ { \mathrm { t e s t } }$ with $\mathcal { D } _ { t } ^ { \mathrm { t r a i n } } \cap \mathcal { D } _ { t } ^ { \mathrm { t e s t } } \stackrel {  } { = } \varnothing$ . While learning task $t ,$ the model can access $\mathcal { D } _ { t } ^ { \mathrm { t r a i n } }$ and a memory of fixed capacity, and the training data of earlier tasks are unavailable. The objective is that the model obtained after task t diagnoses all classes in $\mathcal { C } _ { t }$ accurately on the cumulative test set $\begin{array} { r } { \mathcal { D } _ { t } ^ { \mathrm { e v a l } } = \bigcup _ { k = 0 } ^ { t } \mathcal { D } _ { k } ^ { \mathrm { t e s t } } } \end{array}$ . The model must therefore acquire the classes in C<sub>new</sub> (plasticity) while retaining the classes in $\mathcal { C } _ { t - 1 }$ (stability).

Assumptions. Following the challenges in Section 1, we assume that (i) the stream is heavily imbalanced toward normal operation, with only a few samples per hazard class; (ii) samples are high-dimensional and multimodal; (iii) hazards may co-occur, i.e., the problem is multi-label; (iv) tasks can introduce zero or more new classes and earlier classes can recur; and (v) the memory available for storing past samples is strictly limited.

## 3.2 Overview of CLAR

CLAR is a replay-based class-incremental framework. Its key idea is to keep the memory of past hazards predominantly in the compact latent space of the diagnosis model, and to let the class-specific geometry learned by the classifier decide which samples are worth remembering. Three components realize this idea, each addressing the assumptions in Section 3.1, as shown in Fig. 2: (i) a latentdominant dual replay bufer (Section 3.4) stores many compact latent representations and a few raw samples, which addresses the memory budget for high-dimensional samples (assumptions (ii) and (v)); (ii) an ALS-guided class-balanced sampling strategy (Section 3.5) keeps the bufer balanced across hazard classes under an imbalanced, multi-label stream with recurring classes, and favors samples with a high ALS, i.e., samples that strongly drive the classifier response (assumptions (i), (iii), and (iv)); Section 4 analyzes when this selection improves replay; (iii) a teacher-anchored latent space (Section 3.6) constrains the encoders through distillation from a slowly updated teacher, which encourages stored latent representations to stay compatible with the evolving model.

The three components depend on each other. The latent bufer stays useful only while the latent space remains close to the space in which its representations were stored, which is the role of the teacher, and the sampling strategy reads the parameters of the classifier, so the quality of memory selection improves as the classifier learns. The diagnosis model itself follows the generic architecture of modality-specific encoders and class-specific attention used for root cause diagnosis in [2]. CLAR applies to any encoder–classifier model with class-wise parameters.

## 3.3 Incremental Hazard Diagnosis Model

The incremental hazard diagnosis model has two stages: a multimodal feature extractor, which maps a raw sample to a low-dimensional latent representation, and an attentionbased classifier, which maps the latent representation to hazard class scores. This design follows the common strategy in multimodal learning of encoding each modality with a modality-specific encoder and fusing the results in a shared latent space [2, 38, 39].

CLAR components (proposed): dual replay bufer M, ALS-CBRS, teacher  
AI system in MII [2]  
![](images/5865dea019ca9c5b2e4003bde808b7359ac3500ea51d6c505c3c03663d142896.jpg)  
Figure 2: Overview of CLAR. After each computation task, the AI system in the MII yields the supervised learning performance $X _ { i , t } ^ { ( \stackrel { \triangledown } { \partial } ) }$ and the runtime metrics $X _ { i , t } ^ { ( R ) }$ . The diagnosis model encodes both modalities into a latent representation $z _ { i }$ and scores each hazard class with class-specific attention parameters $\left( \mathbf { A } _ { c } , \mathbf { v } _ { c } \right)$ . The CLAR components (shaded) are the dual replay bufer, the ALS-guided class-balanced sampling, which uses $\left( \mathbf { A } _ { c } , \mathbf { v } _ { c } \right)$ to admit samples, and the EMA teacher, which anchors the latent space through ${ \mathcal { L } } _ { \mathrm { K D } }$ . Dashed arrows denote replay.

Multimodal feature extractor. Encoding into a lowdimensional space serves two purposes. First, a raw sample is high-dimensional, and a classifier trained directly on raw samples from a few hazardous samples per class overfits easily. Second, compact representations are cheap to store, which enables the latent bufer of Section 3.4. Modality-specific encoders map each modality to a latent representation,

$$
z _ { i } ^ { R } = \mathrm { E n c o d e r } _ { R } ( X _ { i , t } ^ { ( R ) } ) , \quad z _ { i } ^ { R } \in \mathbb { R } ^ { d _ { R } } ,\tag{3.2}
$$

$$
z _ { i } ^ { P } = \mathrm { E n c o d e r } _ { P } ( X _ { i , t } ^ { ( P ) } ) , \quad z _ { i } ^ { P } \in \mathbb { R } ^ { d _ { P } } ,\tag{3.3}
$$

which are concatenated into the latent representation of dimension $d = d _ { R } + d _ { P }$ 7

$$
z _ { i } = \mathrm { c o n c a t } ( z _ { i } ^ { R } , z _ { i } ^ { P } ) \in \mathbb { R } ^ { d } .\tag{3.4}
$$

Any modality-specific encoder can be used. Following [2], we use variational autoencoders (VAEs) [40]: an LSTMbased VAE for the time-series modality, whose step-wise outputs are aggregated over time, and a standard VAE for the tabular modality. For modality $m \in \{ R , P \}$ the encoder outputs a Gaussian approximate posterior $q _ { \phi ^ { ( m ) } } ( z _ { i } ^ { m } \ \mid \ X _ { i , t } ^ { ( m ) } )$ with parameters $\phi ^ { ( m ) }$ , whose mean is used as $z _ { i } ^ { m } ;$ the decoders are used only for training (Section 3.6).

Attention-based classifier. For each seen class $c \in { \mathcal { C } } _ { t }$ the classifier holds a trainable attention vector $\mathbf { v } _ { c } \in \mathbb { R } ^ { d }$ which weights the latent dimensions relevant to class c, and a trainable attention matrix $\mathbf { A } _ { c } \in \mathbb { R } ^ { d \times d }$ , which models the interactions between them. The logit of class c is

$$
s _ { c } ( z _ { i } ) = ( \mathbf { A } _ { c } z _ { i } ) ^ { \top } \big ( \mathrm { d i a g } ( \mathbf { v } _ { c } ) z _ { i } \big ) ,\tag{3.5}
$$

where diag $( \mathbf { v } _ { c } ) \in \mathbb { R } ^ { d \times d }$ is the diagonal matrix formed by $\mathbf { v } _ { c } ,$ and the predicted probability of class c is $\sigma ( s _ { c } ( z _ { i } ) )$ with the sigmoid function $\sigma ( \cdot )$ . Each class is scored with its own parameters, which suits the multi-label setting and mirrors the root-cause-specific attention heads of [2]; a new class is accommodated by adding a pair $\left( \mathbf { A } _ { c } , \mathbf { v } _ { c } \right)$ when it first appears. We refer to $\{ ( \mathbf { A } _ { c } , \mathbf { v } _ { c } ) \} _ { c \in \mathcal { C } _ { t } }$ as the learned attention geometry, since it determines how the classifier responds to each region of the latent space; CLAR reuses it to select samples for the bufer (Section 3.5).

## 3.4 Latent-Dominant Dual Replay Bufer

In replay-based CIL, a memory bufer is a fixed-capacity store of past training samples that are trained together with the data of each new task (Section 2). Under a strict memory budget, storing raw samples alone leaves very few exemplars per hazard class: in our case study, one raw sample is about 260 times larger than the 150 latent representations of the latent bufer combined. Storing latent representations alone is insuficient as well, because the encoders keep being updated and need raw inputs of earlier classes to preserve their representation.

CLAR therefore maintains a dual replay bufer $\mathcal { M } =$ $\{ B _ { x } , B _ { z } \}$ with two complementary roles. The latent bufer $B _ { z }$ stores many latent representations $( z _ { i } , y _ { i } )$ and gives the classifier broad coverage of each earlier class at low cost. The raw bufer $B _ { x }$ stores a few raw samples $( X _ { i , t } , y _ { i } )$ , which are re-encoded by the current encoders during training and keep the encoders exposed to earlier classes. To keep the bufer class-balanced, both $B _ { x }$ and $B _ { z }$ are partitioned by hazard class into $\lvert \boldsymbol { B } _ { x } [ c ]$ and $\boldsymbol { B _ { z } } [ c ]$ , with per-class quotas $q _ { c , \mathrm { r a w } } = \lfloor | \mathcal { B } _ { x } | _ { \operatorname* { m a x } } / | \mathcal { C } _ { t } | \rfloor$ and $q _ { c , \mathrm { l a t } } = \lfloor | \mathcal { B } _ { z } | _ { \mathrm { m a x } } / | \mathcal { C } _ { t } | \rfloor$ 2 where $| \cdot | _ { \mathrm { m a x } }$ denotes the capacity. The quotas are recomputed whenever new classes appear; the floor can leave up to $| \mathcal { C } _ { t } | - 1$ slots of each bufer unused, and positive quotas require $| \mathcal { C } _ { t } |$ to stay below the capacity. We call the bufer latent-dominant because latent representations make up most of its slots; in bytes, the few raw samples still dominate. Only hazardous samples are stored, since normal samples are abundant in every task, and a multilabel sample can belong to several partitions at the same time.

## 3.5 ALS-Guided Class-Balanced Sampling

The sampling strategy, referred to as ALS-CBRS, decides which samples enter each partition of the bufer. While a partition is below its quota, every new sample of that class is inserted. Once the partition is full, a new sample of class c replaces a randomly chosen stored sample with probability

$$
p = \operatorname* { m i n } \{ 1 , r _ { c } \} \cdot \tilde { \sigma } \big ( \mathrm { A L S } _ { c } ( z _ { i } ) \big ) , \qquad r _ { c } = \frac { q _ { c } } { N _ { c } } ,\tag{3.6}
$$

where $q _ { c }$ is the quota of the corresponding bufer and $N _ { c }$ is the cumulative number of class-c samples observed so far.

The reservoir factor $r _ { c }$ is the admission probability of reservoir sampling [26] applied within each class, as in CBRS [27]. With $\tilde { \sigma } \equiv 1$ , it keeps each partition a uniform random sample of the class-c samples observed so far. The second factor scales this probability by the response of the classifier to the sample, measured by the Attention Leverage Score

$$
\mathrm { A L S } _ { c } ( z _ { i } ) = \sqrt { \big ( \mathrm { d i a g } ( { \bf v } _ { c } ) z _ { i } \big ) ^ { \top } P _ { c } \big ( \mathrm { d i a g } ( { \bf v } _ { c } ) z _ { i } \big ) } ,\tag{3.7}
$$

where $P _ { c } = \mathbf { A } _ { c } \mathbf { A } _ { c } ^ { \top } + \varepsilon \mathbf { I }$ , and $\varepsilon > 0$ makes $P _ { c }$ positive definite. The term $\mathrm { d i a g } ( \mathbf { v } _ { c } ) z _ { i }$ keeps the latent dimensions that class c attends to, and $P _ { c }$ weights them by the interactions learned in $\mathbf { A } _ { c } ,$ so the ALS is a Mahalanobistype norm in the metric induced by the attention geometry of class c.

The ALS is directly tied to the classifier. Rewriting Eq. (3.5) as $s _ { c } ( z _ { i } ) = z _ { i } ^ { \top } \mathbf { A } _ { c } ^ { \top }$ diag $\left( \mathbf { v } _ { c } \right) z _ { i }$ and applying the Cauchy–Schwarz inequality gives

$$
\begin{array} { r } { | s _ { c } ( \boldsymbol { z } _ { i } ) | \leq \| \boldsymbol { z } _ { i } \| _ { 2 } \left\| \mathbf { A } _ { c } ^ { \top } \mathrm { d i a g } ( \mathbf { v } _ { c } ) \boldsymbol { z } _ { i } \right\| _ { 2 } \leq \| \boldsymbol { z } _ { i } \| _ { 2 } \mathrm { A L S } _ { c } ( \boldsymbol { z } _ { i } ) , } \end{array}\tag{3.8}
$$

where the second inequality holds because $\mathrm { A L S } _ { c } ( z _ { i } ) ^ { 2 } =$ $\| \mathbf { A } _ { c } ^ { \top } \operatorname { d i a g } ( \mathbf { v } _ { c } ) z _ { i } \| _ { 2 } ^ { 2 } + \varepsilon \| \operatorname { d i a g } ( \mathbf { v } _ { c } ) z _ { i } \| _ { 2 } ^ { 2 }$ . The bound holds for every sample, every class, and every $\varepsilon > 0$ , with nearequality when $z _ { i }$ is aligned with $\mathbf { A } _ { c } ^ { \top }$ diag $( { \bf v } _ { c } ) z _ { i }$ and ε is small. For a given sample norm, a large logit magnitude for class c thus requires a large ALS, so the ALS measures how strongly a sample can drive the class-c output in the classifier’s own geometry.

```tcl
Algorithm 1 ALS-CBRS bufer update after task t
Require: $\mathcal { M } = \{ \mathcal { B } _ { x } , \mathcal { B } _ { z } \} , \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } , \mathcal { C } _ { t - 1 } , \mathcal { C } _ { \mathrm { n e w } } , \{ ( \mathbf { A } _ { c } , \mathbf { v } _ { c } ) \}$
1: if $\mathcal { C } _ { \mathrm { n e w } } \neq \emptyset$ then
2: $\mathcal { C } _ { t }  \mathcal { C } _ { t - 1 } \cup \mathcal { C } _ { \mathrm { n e w } } ;$ recompute q<sub>c,raw</sub>, q<sub>c,lat</sub>
3: Evict random items from partitions over their
quota
4: end if
5: for all hazardous $( X _ { i , t } , y _ { i } ) \in \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } \setminus \mathcal { D } _ { t } ^ { \mathrm { r e t r i e v e } }$ do
6: z<sub>i</sub> ← concat(Encoder ${ } _ { R } ( X _ { i , t } ^ { ( R ) } )$ , Encoder<sub>P</sub> $( X _ { i , t } ^ { ( P ) } ) )$
7: for all $c \in { \mathcal { C } } _ { t }$ with $y _ { i , c } = 1$ do
8: $N _ { c } \gets N _ { c } + 1$
9: for all $B \in \{ B _ { x } , B _ { z } \}$ do
10: $( e , q _ { c } ) \ \gets \ ( X _ { i , t } , q _ { c , \mathrm { r a w } } )$ if $\begin{array} { r } { B \ = \ B _ { x } } \end{array}$ , else
$\left( z _ { i } , q _ { c , \mathrm { l a t } } \right)$
11: Draw u ∼ Uniform(0, 1)
12: if $| B [ c ] | < q _ { c }$ then
13: Insert $( e , y _ { i } )$ into $B [ c ]$
14: else if $u \ < \ \mathrm { m i n } \{ 1 , q _ { c } / \tilde { N } _ { c } \} \ \cdot \tilde { \sigma } ( \mathrm { A L S } _ { c } ( z _ { i } ) )$
then
15: Replace a random item of B[c] with
(e, y<sub>i</sub>)
16: end if
17: end for
18: end for
19: end for
```

Finally, the scaled and shifted sigmoid $\tilde { \sigma } ( u ) = 2 / ( 1 +$ $e ^ { - u } ) - 1$ maps the non-negative ALS to [0, 1). The standard sigmoid maps non-negative inputs to [0.5, 1), whereas σ˜ lets weakly responding samples have an admission probability close to zero. The two factors play distinct roles. The per-class quotas keep the bufer balanced across hazard classes for any admission factor (Proposition 1), and the ALS factor tilts the content of each partition toward samples with a high ALS. Section 4 characterizes the resulting bufer and derives when this tilt improves replay over class-balanced uniform replay. It is applied to $B _ { x }$ and $B _ { z }$ with their respective quotas at the end of each task, using the updated encoders and classifier (Algorithm 1).

## 3.6 Teacher-Anchored Training Objective

When the encoders are updated on new hazard classes, the latent space shifts. This representation drift has two consequences: the classifier loses the representation of earlier classes, and the latent representations in $B _ { z }$ which were produced by earlier encoders, drift away from the representations that the current encoders produce. CLAR counters the drift with a teacher model whose parameters $\theta _ { \mathrm { t e a c h e r } }$ are an exponential moving average (EMA) of the parameters $\theta _ { \mathrm { s t u d e n t } }$ of the diagnosis model (the student) [41]. After training on task t, the teacher is

updated as

$$
\theta _ { \mathrm { t e a c h e r } } ^ { ( t ) } = \alpha \theta _ { \mathrm { t e a c h e r } } ^ { ( t - 1 ) } + \left( 1 - \alpha \right) \theta _ { \mathrm { s t u d e n t } } ^ { ( t ) } ,\tag{3.9}
$$

with momentum coeficient $\alpha ~ \in ~ [ 0 , 1 ]$ , so the teacher changes slowly and provides a stable reference for the latent space. The student is trained on the training pool $\mathcal { D } _ { t } ^ { \mathrm { t r a i n } } \cup \mathcal { D } _ { t } ^ { \mathrm { r e t r i e v e } }$ , where $\mathcal { D } _ { t } ^ { \mathrm { r e t r i e v e } }$ is a class-balanced set of raw samples retrieved from $B _ { x } ,$ , with three loss terms.

Weighted multi-label classification loss. The classifier is trained on the classification pool $ { S _ { \mathrm { c l s } } }$ , which contains the latent representations $z _ { i } ^ { \mathrm { t r a i n } }$ of the current mini-batch, encoded by the current encoders, and an equal number of latent representations $z _ { j } ^ { \mathrm { r } }$ <sup>retrieved</sup> retrieved from B<sub>z</sub>. To $B _ { z }$ counter class imbalance, a weighted binary cross-entropy is used,

$$
\mathcal { L } _ { \mathrm { c l f } } = - \frac { 1 } { \left| \mathcal { S } _ { \mathrm { c l s } } \right| } \sum _ { ( z _ { i } , y _ { i } ) \in \mathcal { S } _ { \mathrm { c l s } } } \sum _ { c \in \mathcal { C } _ { t } } \ell _ { i , c } ,\tag{3.10}
$$

$$
\ell _ { i , c } = w _ { c } y _ { i , c } \log \sigma ( \hat { y } _ { i , c } ) + \left( 1 - y _ { i , c } \right) \log \left( 1 - \sigma ( \hat { y } _ { i , c } ) \right) ,\tag{3.11}
$$

where $\hat { y } _ { i , c } = s _ { c } ( z _ { i } )$ , and the positive weight $w _ { c }$ is the ratio of negative to positive samples of class c in the training pool of task t.

VAE loss. The encoders and decoders are trained with the negative evidence lower bound of each modality, which consists of a reconstruction loss and a Kullback–Leibler (KL) divergence toward the standard normal prior [2, 40],

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { V A E } } = \displaystyle \sum _ { m \in \{ R , P \} } \bigg ( \mathcal { L } _ { \mathrm { r e c o n } } \big ( X _ { i , t } ^ { ( m ) } , \hat { X } _ { i , t } ^ { ( m ) } \big ) } & { } \\ { + \mathrm { K L } \big ( q _ { \phi ^ { ( m ) } } \big ( z _ { i } ^ { m } \mid X _ { i , t } ^ { ( m ) } \big ) \big \| \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \big ) \bigg ) , } \end{array}\tag{3.12}
$$

averaged over the training pool, where $\hat { X } _ { i , t } ^ { ( m ) }$ is the reconstruction by the decoder of modality m. This term keeps the latent space informative about both modalities and, through the retrieved raw samples, about earlier hazard classes.

Latent distillation loss. Knowledge distillation [24, 25] is applied in the latent space of each encoder by penalizing the divergence between the posteriors of the student and the teacher,

$$
\mathcal { L } _ { \mathrm { K D } } = \sum _ { m \in \{ R , P \} } \mathrm { K L } \Big ( q _ { \phi _ { \mathrm { s t u d e n t } } ^ { ( m ) } } ( z _ { i } ^ { m } \mid X _ { i , t } ^ { ( m ) } )\tag{3.13}
$$

averaged over the training pool.

Overall objective. The student minimizes

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { c l f } } + \lambda _ { \mathrm { V A E } } { \mathcal { L } } _ { \mathrm { V A E } } + \lambda _ { \mathrm { K D } } { \mathcal { L } } _ { \mathrm { K D } } ,\tag{3.14}
$$

where λ<sub>VAE</sub> $\geq 0$ and λ<sub>KD</sub> $\geq 0$ balance representation learning and stability against diagnosis accuracy.

Algorithm 2 Incremental training of CLAR on task   
$t > 0$   
1: Retrieve class-balanced $\mathcal { D } _ { t } ^ { \mathrm { r e t r i e v e } }$ from $\begin{array} { r } { B _ { x } ; ~ { \mathcal { D } } _ { t } ^ { \mathrm { t r a i n } } \gets } \end{array}$   
D<sup>train</sup><sub>t</sub> ∪ D<sup>retrieve</sup><sub>t</sub>   
2: Add $\left( \mathbf { A } _ { c } , \mathbf { v } _ { c } \right)$ for each $c \in \mathcal { C } _ { \mathrm { n e w } }$   
3: repeat for each mini-batch of size b from $\mathcal { D } _ { t } ^ { \mathrm { t } 1 }$ ain   
4: Encode the mini-batch into $z _ { i } ^ { \mathrm { t r a i n } }$ by $\mathrm { E q . ~ ( 3 . 4 ) }$   
5: $S _ { \mathrm { c l s } } ~  ~ \{ z _ { i } ^ { \mathrm { t r a i n } } \} \cup$ class-balanced $\bar { \{ z _ { j } ^ { \mathrm { r e t r i e v e d } } \} } _ { j = 1 } ^ { b }$   
from $B _ { z }$   
6: Update $\theta _ { \mathrm { s t u d e n t } }$ by a gradient step on $\mathcal { L } _ { \mathrm { t o t a l } }$ in   
Eq. (3.14)   
7: until early stopping   
8: Update the teacher by Eq. (3.9); update M by Algo   
rithm 1

## 3.7 Incremental Training Procedure

On the base task $\mathcal { D } _ { 0 }$ , the diagnosis model is trained with $\mathcal { L } _ { \mathrm { c l f } } + \lambda _ { \mathrm { V A E } } \mathcal { L } _ { \mathrm { V A E } }$ , the teacher is initialized as a copy of the trained model, and the bufer is filled with Algorithm 1. Each incremental task then follows Algorithm 2. Raw samples are retrieved once per task and appended to the training set, so they contribute to all three loss terms; latent representations are retrieved for every mini-batch and contribute to $\mathcal { L } _ { \mathrm { c l f } }$

## 4 Theoretical Analysis of ALS-CBRS

This section analyzes, for an idealized model of the bufer update and of training, what ALS-CBRS stores and when its selection benefits learning. The results describe a mechanism that can explain the role of ALS-guided replay, which provide guarantees for this model. The full adaptive training procedure of CLAR is assessed empirically in Section 5. We consider one partition $B [ c ]$ , which can be $\lvert \boldsymbol { B } _ { x } [ c ]$ or $\boldsymbol { B _ { z } } [ c ]$ , with quota $q _ { c } ,$ and compare ALS-CBRS with its uniform counterpart, i.e., class-balanced reservoir sampling with the same quota, obtained by setting $\tilde { \sigma } \equiv 1$ in Eq. (3.6) and referred to as CBRS below [27]. Proofs are given in the Appendix.

## 4.1 Class Balance

Proposition 1 (Class balance). Under Algorithm 1, after every bufer update, |B[c]| = min $\{ q _ { c } , N _ { c } \}$ for every $c \in { \mathcal { C } } _ { t }$ and for both $B _ { x }$ and $B _ { z }$

Every class observed at least $q _ { c }$ times thus occupies exactly $q _ { c }$ slots, however frequent it is in the stream, and a rarer class keeps all of its $N _ { c }$ samples. Class balance therefore follows from the per-class quotas and holds for any admission factor, including $\tilde { \sigma } ( \mathrm { A L S } _ { c } )$ . The ALS factor determines which samples occupy the slots. For multilabel data, balance refers to the partition sizes: a sample in $B [ c ]$ can carry further labels, so the label counts over the whole bufer still reflect co-occurrence.

## 4.2 Replay Law of ALS-CBRS

Fix class c, a quota $q _ { c } \geq 1$ , and the parameters used by Algorithm 1, i.e., the encoders and $\left( \mathbf { A } _ { c } , \mathbf { v } _ { c } \right)$ . We treat these parameters, and hence the gate below, as fixed and independent of the arrivals analyzed; in Algorithm 1 the gate is learned partly from the same task’s samples that it scores, so this independence is an idealization (Section 4- D). Let the class-c samples arrive as independent and identically distributed (i.i.d.) pairs $( z , y ) \sim \pi _ { c } ,$ where z is the latent representation and y the label vector, and let $N _ { c }$ count the arrivals. Define the ALS gate and its mean as

$$
\begin{array} { r } { h _ { c } ( z ) = \tilde { \sigma } \big ( \mathrm { A L S } _ { c } ( z ) \big ) \in [ 0 , 1 ) , \qquad \bar { h } _ { c } = \mathbb { E } _ { \pi _ { c } } h _ { c } ( z ) > 0 . } \end{array}\tag{4.1}
$$

By Eq. (3.6), the first $q _ { c }$ arrivals fill the partition, and afterward the $N _ { c } { \mathrm { - t h } }$ arrival replaces a uniformly chosen entry with probability $( q _ { c } / N _ { c } ) h _ { c } ( z )$ . The ALS-weighted class distribution $\tilde { \pi } _ { c }$ is defined by

$$
d \tilde { \pi } _ { c } = \frac { h _ { c } } { \bar { h } _ { c } } d \pi _ { c } .\tag{4.2}
$$

Lemma 1 (Replay law). Every sample admitted $a f t e r$ the partition is full has law $\tilde { \pi } _ { c }$ . Each entry of the initial fill survives the first $N _ { c }$ arrivals with probability

$$
\kappa _ { c } ( N _ { c } ) = \prod _ { j = q _ { c } + 1 } ^ { N _ { c } } \Big ( 1 - \frac { \bar { h } _ { c } } { j } \Big ) ,\tag{4.3}
$$

and the total variation distance between the law of $B [ c ]$ and $\tilde { \pi } _ { c } ^ { \otimes q _ { c } }$ is at most min $\{ 1 , q _ { c } \kappa _ { c } ( N _ { c } ) \}$ . In particular, $B [ c ]$ converges in distribution to ${ { q } _ { c } } \mathrm { ~ } i . i . d .$ . draws from $\tilde { \pi } _ { c }$ as $N _ { c }  \infty$

Lemma 1 states precisely what favoring high-ALS samples means: in the long run, each partition is a sample from the class distribution reweighted by $\tilde { \sigma } ( \mathrm { A L S } _ { c } )$ whereas CBRS $( h _ { c } \equiv 1 )$ keeps a uniform sample of $\pi _ { c }$ For finite streams, entries of the initial fill, which enter without ALS, persist with probability $\kappa _ { c } ( N _ { c } )$ . CBRS gives $\kappa _ { c } ( N _ { c } ) = q _ { c } / N _ { c }$ , and ALS-CBRS gives $\kappa _ { c } ( N _ { c } )$ ≈ $( q _ { c } / N _ { c } ) ^ { \bar { h } _ { c } }$ for large $q _ { c } ,$ so a gate with a small mean $\bar { h } _ { c }$ refreshes the partition more slowly. With only a few samples per hazard class, $N _ { c } / q _ { c }$ is small and part of each partition originates from the initial fill; the results below hold at every finite $N _ { c }$ and account for this efect.

## 4.3 Replay-Gradient Error and Margin Stability

To relate the stored samples to learning, consider an output class $c ^ { \prime } \in \mathcal { C } _ { t }$ , for instance $c ^ { \prime } = c ,$ and hold $\mathbf { A } _ { c ^ { \prime } }$ and the encoders fixed. The logit in Eq. (3.5) is then linear in the attention vector,

$$
s _ { c ^ { \prime } } ( z ) = \mathbf { v } ^ { \top } \psi _ { c ^ { \prime } } ( z ) , \qquad \psi _ { c ^ { \prime } } ( z ) = z \odot ( \mathbf { A } _ { c ^ { \prime } } z ) ,\tag{4.4}
$$

where v denotes the attention vector $\mathbf { v } _ { c ^ { \prime } }$ treated as the optimization variable. Let $- \ell _ { c ^ { \prime } } ( \mathbf { v } ; z , y )$ be the weighted binary cross-entropy of Eq. (3.11) for class $c ^ { \prime }$ with logit $\mathbf { v } ^ { \top } \psi _ { c ^ { \prime } } ( z )$ and a fixed weight $w _ { c ^ { \prime } }$ . The population and replay objectives are

$$
\begin{array} { l l } { \displaystyle \mathcal { I } ( \mathbf { v } ) = \mathbb { E } _ { \pi _ { c } } \big [ - \ell _ { c ^ { \prime } } ( \mathbf { v } ; z , y ) \big ] + \Omega ( \mathbf { v } ) , } \\ { \displaystyle \hat { \mathcal { I } } ( \mathbf { v } ) = \frac { 1 } { q _ { c } } \sum _ { ( z , y ) \in B [ c ] } \big [ - \ell _ { c ^ { \prime } } ( \mathbf { v } ; z , y ) \big ] + \Omega ( \mathbf { v } ) , } \end{array}\tag{4.5}
$$

where Ω is a twice diferentiable convex term that collects the remaining training loss, such as the loss on the current task, held fixed and common to both sampling rules. The replay term contributes the Hessian $\begin{array} { r } { q _ { c } ^ { - 1 } \sum _ { \left( z , y \right) \in \mathcal { B } \left[ c \right] } a \sigma ( 1 - \sigma ) \psi _ { c ^ { \prime } } ( z ) \psi _ { c ^ { \prime } } ( z ) ^ { \top } } \end{array}$ with $a =$ $w _ { c ^ { \prime } } y _ { c ^ { \prime } } + 1 - y _ { c ^ { \prime } }$ and $\boldsymbol { \sigma } = \sigma ( \mathbf { v } ^ { \top } \boldsymbol { \psi } _ { c ^ { \prime } } ( z ) )$ , whose rank is at most $q _ { c } ;$ for $q _ { c } < d ,$ the curvature condition of Theorem 1 therefore relies on $\Omega$ , and we keep it as an assumption of the analysis. Let $\mathbf { v } ^ { \star }$ and vˆ minimize $\mathcal { I }$ and ${ \hat { \mathcal { I } } } ,$ respectively. The replay gradient at $\mathbf { v } ^ { \star }$ is governed by

$$
\mathbf { g } ( z , y ) = \left( w _ { c ^ { \prime } } y _ { c ^ { \prime } } + 1 - y _ { c ^ { \prime } } \right) \left( \sigma ( \mathbf { v } ^ { \star \top } \psi _ { c ^ { \prime } } ( z ) ) - y _ { c ^ { \prime } } \right) \psi _ { c ^ { \prime } } ( z ) ,\tag{4.6}
$$

with the statistics

$$
\begin{array} { r l r } & { \displaystyle V _ { c } = \mathrm { t r } \mathrm { C o v } _ { \pi _ { c } } ( \mathbf { g } ) , \qquad \tilde { V } _ { c } = \mathrm { t r } \mathrm { C o v } _ { \tilde { \pi } _ { c } } ( \mathbf { g } ) , } & \\ & { \displaystyle \pmb { \delta } _ { c } = \mathbb { E } _ { \tilde { \pi } _ { c } } \mathbf { g } - \mathbb { E } _ { \pi _ { c } } \mathbf { g } = \mathrm { C o v } _ { \pi _ { c } } ( \mathbf { g } , h _ { c } ) / \bar { h } _ { c } . } & \end{array}\tag{4.7}
$$

Here, $V _ { c }$ and $\tilde { V } _ { c }$ are the gradient variances under uniform and ALS-weighted selection, and $\delta _ { c }$ is the selection bias.

Theorem 1 (Replay-gradient error and margin stability). Suppose $\mathbb { E } _ { \pi _ { c } } \lVert z \rVert _ { 2 } ^ { 4 } < \infty , \mathbf { v } ^ { \star }$ and vˆ are interior minimizers, and $\nabla ^ { 2 } \hat { \mathcal { I } } \succeq \mu \mathbf { I }$ on the segment between $\mathbf { v } ^ { \star }$ and vˆ almost surely, with a deterministic $\mu > 0$ common to both sampling rules. Let $\begin{array} { r } { \bar { \mathbf { g } } = q _ { c } ^ { - 1 } \sum _ { ( z , y ) \in B [ c ] } \mathbf { g } ( z , y ) } \end{array}$ , and let $\displaystyle \mathcal { E } _ { c } ( N _ { c } ) = \mathbb { E } \| \bar { \mathbf { g } } - \mathbb { E } _ { \pi _ { c } } \mathbf { g } \| _ { 2 } ^ { 2 }$ be the mean-squared error of the replay gradient of ALS-CBRS after $N _ { c }$ arrivals and $\mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ that of CBRS. Then

$$
\operatorname* { l i m } _ { N _ { c } \to \infty } \mathscr { E } _ { c } ( N _ { c } ) = \frac { \tilde { V } _ { c } } { q _ { c } } { + } \| \delta _ { c } \| _ { 2 } ^ { 2 } , \qquad \mathscr { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } ) = \frac { V _ { c } } { q _ { c } } \ \forall N _ { c } \geq q _ { c } .\tag{4.8}
$$

Moreover, for a test sample $( z _ { * } , y _ { * } )$ and a decision threshold $\tau ,$ the margin $\gamma ( \mathbf { v } ) \ = \ ( 2 y _ { * , c ^ { \prime } } - 1 ) \big ( \mathbf { v } ^ { \top } \psi _ { c ^ { \prime } } ( z _ { * } ) - \tau \big )$ satisfies

$$
\mathbb { E } \big ( \gamma ( \hat { \mathbf { v } } ) - \gamma ( \mathbf { v } ^ { \star } ) \big ) ^ { 2 } \leq \frac { \| \psi _ { c ^ { \prime } } ( z _ { * } ) \| _ { 2 } ^ { 2 } } { \mu ^ { 2 } } \mathcal { E } _ { c } ( N _ { c } ) ,\tag{4.9}
$$

and the same bound with $\mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ holds for CBRS.

Corollary 1 (Critical capacity). Under Theorem 1, suppose $\delta _ { c } \neq \mathbf { 0 }$ and define

$$
q _ { c } ^ { \star } = \frac { V _ { c } - \tilde { V } _ { c } } { \| \delta _ { c } \| _ { 2 } ^ { 2 } } = 1 - \frac { \mathrm { C o v } _ { \pi _ { c } } \left( h _ { c } , \| \mathbf { g } - \mathbb { E } _ { \pi _ { c } } \mathbf { g } \| _ { 2 } ^ { 2 } \right) } { \bar { h } _ { c } \| \delta _ { c } \| _ { 2 } ^ { 2 } } .\tag{4.10}
$$

$H f 1 \le q _ { c } < q _ { c } ^ { \star }$ , then $\mathcal { E } _ { c } ( N _ { c } ) \ : < \ : \mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ for every $N _ { c } > q _ { c } ,$ , and the margin bound in $E q .$ (4.9) is smaller for ALS-CBRS whenever $\psi _ { c ^ { \prime } } ( z _ { * } ) \neq \mathbf { 0 }$ . If $q _ { c } > q _ { c } ^ { \star }$ , then $\mathsf { \bar { E } } _ { c } ( N _ { c } ) > \mathsf { \mathcal { E } } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ for all suficiently large $\begin{array} { l l } { { N _ { c } . } } & { { { \ I f \ } } } \end{array}$ $\delta _ { c } = \mathbf { 0 } _ { : }$ then $\mathcal { E } _ { c } ( N _ { c } ) < \mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ for every $N _ { c } > q _ { c }$ exactly when $\tilde { V } _ { c } < V _ { c }$

## 4.4 Implications for CLAR

Theorem 1 splits the replay-gradient error into a sampling variance $\tilde { V } _ { c } / q _ { c }$ and a squared selection bias $\lVert \pmb { \delta } _ { c } \rVert _ { 2 } ^ { 2 }$ . CBRS is unbiased, and ALS-CBRS trades a bias for a change in variance. By Corollary 1, in the limit $N _ { c }  \infty$ , a capacity $q _ { c } \geq 1$ with an advantage over CBRS exists exactly when $q _ { c } ^ { \star } > 1$ (the exact finite-stream criterion is given in Appendix D), i.e., when $\begin{array} { r } { \mathrm { C o v } _ { \pi _ { c } } ( h _ { c } , \| \mathbf { g } - \mathbb { E } _ { \pi _ { c } } \mathbf { g } \| _ { 2 } ^ { 2 } ) < 0 \mathrm { : } } \end{array}$ the ALS gate has to favor samples whose replay gradients lie close to the class-average gradient. This is the precise sense in which ALS-selected samples are informative for replay: they represent the gradient of their class with less variance. Lemma 1, Theorem 1, and Corollary 1 hold for any fixed gate with values in [0, 1] and a positive mean. The logit bound in Eq. (3.8) motivates the ALS gate and leaves the sign of the covariance open, so the analysis characterizes when gated replay helps, and whether the ALS gate meets the condition is a property of the data and the learned attention geometry that can be estimated at training checkpoints. Through Eq. (4.9), a smaller replay-gradient error gives a smaller upper bound on the expected squared deviation of the classification margin of each test sample, which relates the sampling rule to the stability of the decision boundaries of earlier hazard classes.

Per-class capacity and the dual bufer. Holding the gradient statistics fixed, the limiting advantage over CBRS, $( V _ { c } - \tilde { V } _ { c } ) / q _ { c } - \| \delta _ { c } \| _ { 2 } ^ { 2 }$ , grows as q<sub>c</sub> decreases. In CLAR, the quotas $q _ { c , \mathrm { r a w } } = \lfloor | \mathcal { B } _ { x } | _ { \operatorname* { m a x } } / | \mathcal { C } _ { t } | \rfloor$ and $q _ { c , \mathrm { l a t } } = \lfloor | \mathcal { B } _ { z } | _ { \mathrm { m a x } } / | \mathcal { C } _ { t } | \rfloor$ shrink as hazard classes accumulate, so ALS selection matters most when many classes share a small bufer. The two bufers play complementary roles: the latentdominant design enlarges $^ { q _ { c , \mathrm { l a t } } }$ and thereby reduces the variance term under any sampling rule, while ALS selection gains most in partitions with a small quota, such as those of the raw bufer.

Scope. The analysis holds the encoders, $\mathbf { A } _ { c ^ { \prime } }$ , and the ALS gate fixed and independent of the analyzed arrivals, assumes i.i.d. arrivals within a class, a fixed quota $q _ { c } \geq 1$ and class weight, and a curvature bound common to both rules, optimizes one attention vector with the remaining loss held fixed, and compares ALS-CBRS with CBRS at the same quota. CLAR departs from this model in four ways: the gate is learned partly from the samples it scores; the parameters change between tasks, a drift that the teacher-anchored latent space (Section 3.6) is designed to slow; the whole network is trained jointly with several interacting partitions, and multi-label samples couple the partitions of co-occurring classes; and quotas change when classes appear, where random eviction keeps a uniformly chosen subset and thus preserves the composition of a partition in expectation. The analysis concerns the replay gradient and the margin deviation, and the diagnosis performance along the incremental sequence is assessed empirically in Section 5.

## 5 Experiments

## 5.1 MII Testbed and AI System

We evaluate CLAR on hazard diagnosis data from the MII testbed of [2], whose AI system provides product quality modeling for an Aerosol Jet Printing (AJP) process as its computation service. The manufacturing data come from a real AJP dataset of 95 samples, each with six in situ process variables in time-series format (atomizer gas flow, sheath gas flow, current, nozzle X-coordinate, nozzle Y-coordinate, and nozzle vibration) and a binary quality label (conforming or nonconforming). Four synthetic time-series variables with negligible predictive power are added, giving ten input variables. Five AJP machines are simulated, each with its own ground-truth DNN model that labels samples generated by augmenting the 95 original samples (time warping, pooling, convolution, and others), which yields 500 samples per machine. A multivariate time-series DNN pipeline [5] is trained for each machine, and the top three pipelines of each machine are deployed on five fog nodes (Raspberry Pi computers) coordinated by a Cloud server. A computation task is the quality prediction for a batch of data from one machine, executed on one computation node.

Hazards are injected by varying the levels of the root cause factors of [2] in the data and AI pipeline layers, as summarized in Table I. Hazards in the cyber-physical layer, i.e., failed fog nodes and communication channels, prevent the computation task from completing and leave the supervised learning performance missing, so they are trivially identifiable [2]; we therefore consider the five root cause factors whose computation tasks complete successfully.

## 5.2 Diagnosis Dataset

The diagnosis dataset consists of 4,705 successfully executed computation tasks, each forming one sample $X _ { i , t } = \{ X _ { i , t } ^ { ( \bar { P } ) } , X _ { i , t } ^ { ( R ) } \}$ as defined in Section 3.1. The supervised learning performance $X _ { i , t } ^ { ( P ) }$ is a 92-dimensional vector that concatenates 9 core performance metrics (accuracy, precision, and F1 score of the top three pipelines) with 83 additional features. The runtime metrics $X _ { i , t } ^ { ( R ) }$ form a tensor of shape 500 × 500 × 10, which records 10 hardware variables, including CPU temperature, memory consumption, and data transmission volume, over 500 time stamps for the 500 data samples of the processed batch. Each sample is labeled with the hazard classes in Table I, and several hazards can co-occur in the same computation task.

Table I: Hazard classes and the corresponding root cause factors [2].
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=1>Root cause factor andsimulation</td></tr><tr><td rowspan=1 colspan=1>Class 0</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Sensor contamination orfailure: % of contaminated orfailed sensors among the teninputs</td></tr><tr><td rowspan=1 colspan=1>Class 1</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Low signal-to-noise ratio(SNR): random noise injectedat controlled scales</td></tr><tr><td rowspan=1 colspan=1>Class 2</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Distribution change of inputvariables: Gaussian mixturedata with controlled KLdivergence</td></tr><tr><td rowspan=1 colspan=1>Class 3</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Class imbalance:nonconforming-to-conformingratio within a data batch</td></tr><tr><td rowspan=1 colspan=1>Class 4</td><td rowspan=1 colspan=1>AI pipeline</td><td rowspan=1 colspan=1>Pipeline singularity: % ofpipelines forced to produceidentical predictions</td></tr></table>

## 5.3 Class-Incremental Protocol

The dataset is highly imbalanced and sparse: only 273 of the 4,705 samples contain at least one hazard, and the remaining samples represent normal operation. The base task is built from the most frequent label configuration to initialize the feature extractor and the classifier. Among all three-label subsets, {Class 0, Class 3, Class 4} has the largest support, and the base task contains all hazardous samples of these classes. The remaining classes enter the incremental tasks in the order given by the class schedule $S _ { \mathrm { s c h e d } } = [ 2 , 1 ]$ , i.e., Class 2 appears before Class 1.

The task-allocation hyperparameter $K = 4 0$ sets the number of hazardous samples per incremental task, which yields five incremental tasks and six tasks in total. Since only two new classes appear after the base task, most incremental tasks introduce zero new classes and revisit classes that have already been seen, which reflects the sparse and irregular emergence of new hazards in the MII (assumption (iv)). Normal samples are added to each task according to the imbalance ratio, i.e., the number of normal samples per hazardous sample, which is $\rho _ { \mathrm { b a s e } } = 1 0$ for the base task and $\rho _ { \mathrm { i n c } } \in \{ 1 0 , 1 5 \}$ for the incremental tasks.

## 5.4 Implementation Details

CLAR follows Algorithm 2 with mini-batch size $b = 8 .$ which is small owing to the size of the raw samples. The loss weights in Eq. (3.14) are $\lambda _ { \mathrm { V A E } } = 0 . 1$ and $\lambda _ { \mathrm { K D } } = 1 . 0$ and the EMA momentum in Eq. (3.9) is $\alpha = 0 . 6$ . Training on each task stops early when the relative change in $\mathcal { L } _ { \mathrm { c l f } }$ falls below 0.005 for 5 epochs without improvement.

All replay methods are configured with comparable memory budgets. The baselines have a capacity of 50 raw samples and retrieve 40 of them per task. CLAR has a capacity of 49 raw samples in $B _ { x }$ and 150 latent representations of dimension $d = 6 4$ in $B _ { z }$ ; with five classes, $q _ { c , \mathrm { r a w } } = 9$ and $q _ { c , \mathrm { l a t } } = 3 0$ , so at most 45 raw slots are occupied. One raw sample contains $5 0 0 \times 5 0 0 \times 1 0 + 9 2 \approx 2 . 5 0 \times 1 0 ^ { 6 }$ values, whereas the whole latent bufer contains $1 5 0 \times 6 4 = 9 { , } 6 0 0$ values, i.e., 0.38% of one raw sample. At equal numerical precision, the configured capacity of CLAR is therefore about 2.0% smaller than that of the baselines (about 490 MB compared with 500 MB in 32-bit floating point) and provides about four times as many slots. Since a multi-label sample can occupy several partitions, the number of distinct stored samples can be smaller than the number of slots. Latent representations are retrieved for every mini-batch with $b = 8 .$ , which avoids re-encoding large raw tensors at each step.

## 5.5 Baselines

All baselines use the diagnosis model of Section 3.3, i.e., the same multimodal feature extractor and attentionbased classifier, so that the comparison isolates the replay design of Sections 3.4 and 3.5. No Replay trains on each task sequentially without a bufer and serves as the reference for forgetting. Experience Replay (ER) [10] updates a raw bufer by reservoir sampling and retrieves raw samples for every mini-batch. Class-Balancing Reservoir Sampling (CBRS) [27] applies reservoir sampling within each class, and can be viewed as ALS-CBRS with $\tilde { \sigma } ( \mathrm { A L S } _ { c } ) \equiv 1$ and a raw bufer only. iCaRL [11] selects exemplars by herding in the feature space and appends them to the training set of each task, and FBCR [13] combines virtual prototypes with class-center-based exemplar selection. The baselines thus difer from CLAR in how samples are selected (random, class-balanced random, or by feature-space centers, compared with ALS-guided selection in CLAR) and in how they are stored (raw samples, compared with the latent-dominant bufer of CLAR).

## 5.6 Evaluation Metrics

After training on task t, the model is evaluated on the test sets of all tasks seen so far. Since normal samples dominate the data, accuracy is uninformative, and we use the weighted F1 score. Let $S _ { k } ^ { ( c ) }$ be the number of occurrences of class c in the test set of task $k ,$ and let $F 1 _ { t , k } ^ { ( c ) }$ be the F1 score of class c for the model trained up to task t and evaluated on task $k ,$ with class-wise predictions obtained by thresholding $\sigma ( s _ { c } ( z _ { i } ) )$ under the same protocol for all methods. The weighted F1 score is

$$
M _ { t , k } = \frac { 1 } { \sum _ { c \in \mathcal { C } _ { t } } S _ { k } ^ { ( c ) } } \sum _ { c \in \mathcal { C } _ { t } } S _ { k } ^ { ( c ) } F 1 _ { t , k } ^ { ( c ) } .\tag{5.1}
$$

We report the average weighted F1 (AvgF1), forgetting (F), and backward transfer (BWT) after the final task

Table II: Performance with $\rho _ { \mathrm { b a s e } } = 1 0$ and $\rho _ { \mathrm { i n c } } \in \{ 1 0 , 1 5 \}$ (mean ± std over 5 runs). Bufer capacity: 50 raw samples for the baselines; 49 raw samples (B<sub>x</sub>) and 150 latent representations (B<sub>z</sub>) for CLAR. The best value in each column is in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3"> $\rho _ { \mathrm { i n c } } = 1 0$ </td><td colspan="3"> $\overline { { \rho _ { \mathrm { i n c } } = 1 5 } }$ </td></tr><tr><td> $\overline { { \mathbf { A v g F 1 } \mathrm { \uparrow } } }$ </td><td> $\overline { { \mathbf { F o r g e t t i n g } \downarrow } }$ </td><td>BWT</td><td>AvgF1 ↑</td><td> $\overline { { \mathbf { F o r g e t t i n g } \downarrow } }$ </td><td>BWT</td></tr><tr><td>No Replay</td><td> $\overline { { 0 . 8 9 3 2 \pm 0 . 0 1 9 4 } }$ </td><td> $\overline { { 0 . 0 0 7 1 \pm 0 . 0 0 3 8 } }$ </td><td> $\overline { { 0 . 0 4 6 8 \pm 0 . 0 1 1 8 } }$ </td><td> $\overline { { 0 . 8 8 5 5 \pm 0 . 0 1 8 3 } }$ </td><td> $\overline { { 0 . 0 1 3 1 \pm 0 . 0 1 2 9 } }$ </td><td> $\overline { { 0 . 0 3 8 2 \pm 0 . 0 1 8 6 } }$ </td></tr><tr><td>iCaRL</td><td> $\overline { { 0 . 8 8 7 1 \pm 0 . 0 1 4 4 } }$ </td><td> $\mathbf { \overline { { 0 . 0 0 5 8 \pm 0 . 0 0 2 6 } } }$ </td><td> $\overline { { 0 . 0 4 2 8 \pm 0 . 0 0 6 2 } }$ </td><td> $\overline { { 0 . 8 8 9 1 \pm 0 . 0 0 6 0 } }$ </td><td> $\mathbf { \overline { { 0 . 0 0 6 2 \pm 0 . 0 0 4 2 } } }$ </td><td> $\overline { { 0 . 0 4 2 6 \pm 0 . 0 0 9 6 } }$ </td></tr><tr><td>ER</td><td> $0 . 9 0 6 9 \pm 0 . 0 0 9 0$ </td><td> $0 . 0 1 9 1 \pm 0 . 0 1 2 5$ </td><td> $0 . 0 3 4 9 \pm 0 . 0 1 5 9$ </td><td> $0 . 8 9 3 7 \pm 0 . 0 2 0 0$ </td><td> $0 . 0 3 0 6 \pm 0 . 0 1 0 2$ </td><td> $0 . 0 2 5 4 \pm 0 . 0 1 5 3$ </td></tr><tr><td>CBRS</td><td> $0 . 8 8 3 1 \pm 0 . 0 3 1 6$ </td><td> $0 . 0 3 9 4 \pm 0 . 0 3 5 0$ </td><td> $0 . 0 1 4 5 \pm 0 . 0 3 1 8$ </td><td> $0 . 8 9 1 8 \pm 0 . 0 1 9 0$ </td><td> $0 . 0 2 6 9 \pm 0 . 0 0 9 6$ </td><td> $0 . 0 3 3 0 \pm 0 . 0 1 9 0$ </td></tr><tr><td>FBCR</td><td> $0 . 9 0 0 7 \pm 0 . 0 2 1 4$ </td><td> $0 . 0 1 5 2 \pm 0 . 0 1 2 9$ </td><td> $0 . 0 4 9 0 \pm 0 . 0 2 2 6$ </td><td> $0 . 8 9 2 6 \pm 0 . 0 2 3 0$ </td><td> $0 . 0 1 9 9 \pm 0 . 0 1 2 5$ </td><td> $0 . 0 3 9 4 \pm 0 . 0 2 2 7$ </td></tr><tr><td>CLAR</td><td> $\mathbf { 0 . 9 2 4 8 \pm 0 . 0 1 7 4 }$ </td><td> $0 . 0 1 2 0 \pm 0 . 0 0 9 6$ </td><td> $0 . 0 0 7 3 \pm 0 . 0 1 1 1$ </td><td> $\mathbf { 0 . 9 3 3 8 \pm 0 . 0 1 3 0 }$ </td><td> $0 . 0 0 9 2 \pm 0 . 0 0 7 0$ </td><td> $0 . 0 1 6 1 \pm 0 . 0 0 9 3$ </td></tr></table>

![](images/a4ac2813a5d27a2bcd25fa42616e8718e75b3f86b40c8ee66050b983f731b764.jpg)

![](images/48f1f20576617f264317d9e29ec4bf5c67a76d41a784b90bab99a0853e9742ec.jpg)  
Figure 3: Weighted F1 score per task for $\rho _ { \mathrm { i n c } } = 1 0$ (left) and $\rho _ { \mathrm { i n c } } = 1 5$ (right).

T [42, 43],

$$
\mathrm { A v g F 1 } = \frac { 1 } { T + 1 } \sum _ { k = 0 } ^ { T } M _ { T , k } ,\tag{5.2}
$$

$$
F = \frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } \Big ( \operatorname* { m a x } _ { l \in \{ k , \ldots , T - 1 \} } M _ { l , k } - M _ { T , k } \Big ) ,\tag{5.3}
$$

$$
\mathrm { B W T } = \frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } \left( M _ { T , k } - M _ { k , k } \right) .\tag{5.4}
$$

AvgF1 measures the final diagnosis performance over all tasks, F measures the largest drop from the best past performance, and BWT measures the net change of earlier tasks between learning them and the end of the stream. All results are the mean and standard deviation over 5 independent runs.

## 5.7 Results

Table II and Fig. 3 summarize the results.

Diagnosis performance. CLAR achieves the highest mean AvgF1 under both imbalance levels. For $\rho _ { \mathrm { i n c } } = 1 0$ CLAR reaches 0.9248±0.0174, which exceeds the strongest baseline, ER (0.9069 ± 0.0090), by 0.0179. For $\rho _ { \mathrm { i n c } } = 1 5 $ CLAR reaches 0.9338±0.0130, which exceeds the strongest baseline, again ER (0.8937 ± 0.0200), by 0.0401. The advantage of CLAR thus grows with the imbalance of the incremental tasks. From $\rho _ { \mathrm { i n c } } = 1 0$ to 15, the AvgF1 of CLAR changes by +0.0090, while the changes of the baselines range from −0.0132 (ER) to +0.0087 (CBRS), so

CLAR remains stable over the tested range of imbalance. Since the base-task imbalance is fixed and only two levels are tested, this result supports robustness within the tested conditions. The raw-only replay baselines remain close to No Replay (0.8932 and 0.8855), and iCaRL and CBRS fall slightly below it for $\rho _ { \mathrm { i n c } } = 1 0$ , which indicates that a few raw exemplars per class bring limited benefit when hazard classes are scarce and multi-label; the latentdominant bufer of CLAR addresses exactly this regime.

Forgetting. iCaRL attains the lowest forgetting in both settings (0.0058 and 0.0062), and No Replay also forgets little for $\rho _ { \mathrm { i n c } } = 1 0 \ ( 0 . 0 0 7 1 )$ Both, however, have low $\mathrm { A v g F 1 }$ , so their low forgetting reflects limited learning of the incremental classes. CLAR has the lowest forgetting among the methods with AvgF1 above 0.90 (0.0120 and 0.0092), and its forgetting decreases as the imbalance grows, whereas the forgetting of ER increases from 0.0191 to 0.0306. Among the methods with AvgF1 above 0.90, CLAR thus combines the highest AvgF1 with the lowest forgetting.

Backward transfer. All methods, including No Replay, show positive BWT, i.e., the performance on earlier tasks improves on average after later tasks. This follows from the protocol, in which most incremental tasks revisit classes that have already been seen and thus provide additional training samples for earlier classes. CLAR shows the smallest BWT in both settings (0.0073 and 0.0161). Combined with its highest AvgF1, this is consistent with CLAR reaching high performance on each task when it is first learned and changing it little afterward, whereas the baselines start lower on each task and gain more from revisited classes. BWT is therefore read together with AvgF1 and forgetting.

Overall assessment. CLAR provides the highest mean diagnosis performance under both imbalance levels with competitive forgetting and the smallest net backward change, with a smaller configured bufer than the rawreplay baselines. iCaRL achieves the lowest forgetting at a lower AvgF1, which shows that the preferred method depends on the metric of interest. These results support the CLAR framework as a whole under the evaluated memory and imbalance constraints; the contribution of the ALS gate alone is isolated by an ablation with $\tilde { \sigma } \equiv 1$ as suggested by Corollary 1.

## 5.8 Discussions

The empirical evaluation of the proposed framework provides deep insights into its continual learning capabilities, parameter dynamics, and representational stability under severe class imbalance. Figure 4 illustrates the per-class

![](images/9ac64d53a01e8632fd7fafe1aba119f09d88db8ee4050a60176499ba4abf9c54.jpg)  
Figure 4: Class 1 per-class F1 Score Comparison across 5 benchmark runs, showing the mean ± standard deviation for imbalance ratio of 10. Class 1 is introduced as a new class at task index $j = 3$

F1 score comparison for Class 1, which is a new class introduced in the incremental learning process, specifically at Task 3, across five benchmark runs under an imbalance ratio of 10. As observed, CLAR significantly outperforms other continual learning baselines, including ER, iCARL, CBRS, and FBCR, achieving a much higher F1 score and maintaining robust performance. CLAR’s dual-replay mechanism and attention formulation enable rapid and stable acquisition of Class 1 without degrading prior knowledge.

To understand the internal mechanisms of the attention based classifier, Figure 5 visualizes the learned class-specific attention parameters after the final task across hazard classes 0 through 4. The trainable perclass attention vectors $\mathbf { v } _ { c }$ (top row) isolate critical feature dimensions, while the per-class interaction matrices $\mathbf { A } _ { c }$ (bottom row) capture complex cross-dimensional weightings and feature geometries. This dual formulation allows the model to compute class-aware logits via $s _ { c } ( z _ { i } ) = ( \mathbf { A } _ { c } z _ { i } ) ^ { \top } ( \mathrm { d i a g } ( \mathbf { v } _ { c } ) z _ { i } )$ , tailoring the decision space specifically to the unique feature correlations of each hazard class.

Further diagnostic validation is provided in Figure $6 ,$ which plots the Attention Logit Score (ALS) against the final logit values for all classes after the final task. The clear separation between hazard-free samples (gray points) and hazardous samples (red points) relative to the decision threshold $( \mathrm { l o g i t } = 0 $ , corresponding to a 0.5 probability) confirms that the attention-based scoring function reliably discriminates hazardous instances.

Finally, Figure 7 examines the representational consistency during incremental training across all hazard classes. By visualizing current streaming samples (blue circles), batch-wise retrieved raw samples from $B _ { x }$ (green diamonds), and retrieved latent representations from $B _ { z }$ (orange triangles), we observe a strong spatial overlap across modalities. This alignment highlights the efectiveness of the dual-replay memory bufer in preserving feature distribution consistency, preventing representation drift, and ensuring stable continual learning throughout the entire training lifecycle.

## 6 Conclusion

This work proposed Class-aware Latent Attention Replay (CLAR), a class-incremental hazard diagnosis framework that supports resilient AI systems in the MII. CLAR addresses newly emerging and recurring hazard classes, high-dimensional multimodal samples, severe class imbalance, multi-label co-occurrence, and limited memory through three coupled components: a latent-dominant dual replay bufer, ALS-guided class-balanced sampling, and a teacher-anchored latent space. The ALS links sample selection to the classifier through an upper bound on the logit magnitude. For an idealized model of the bufer update, the analysis showed that ALS-CBRS keeps a classbalanced bufer that follows an ALS-weighted class distribution, and that it has a lower replay-gradient error and a smaller upper bound on the expected squared margin deviation than class-balanced uniform replay whenever the per-class capacity lies below a critical value determined by the gate and the gradient statistics of the class.

On hazard data from an MII testbed with $\rho _ { \mathrm { b a s e } } = 1 0$ and $\rho _ { \mathrm { i n c } } \in \{ 1 0 , 1 5 \} , \mathrm { C L A R }$ achieved the highest mean AvgF1 (0.9248 and 0.9338) among all methods with a configured bufer about 2% smaller than those of the raw-replay baselines. Its forgetting (0.0120 and 0.0092) was the lowest among methods with comparable diagnosis performance, while iCaRL forgot less at a lower AvgF1. All methods showed positive BWT, and CLAR showed the smallest one, reflecting high initial performance on each task with little subsequent change. These results highlight the value of evaluating incremental diagnosis with complementary metrics. Future work will estimate these gradient statistics at training checkpoints to verify the critical capacity empirically, examine the separation of class clusters in the latent space, and extend CLAR to cyber-physical layer hazards and to the diagnosis of hazard severity levels.

![](images/dde6a7237a640bf1ae99059e7b72214fea49fa7985c52735e0b96c3edf0cf096.jpg)  
Figure 5: Plot of parameters of the classifier illustrating the learned class-specific attention parameters for after the final task across classes 0 through 4. The top row displays the trainable per-class attention vectors $\mathbf { v } _ { c } \in \mathbb { R } ^ { d }$ (visualized as bar plots across feature dimensions), which scale feature activations independently. The bottom row shows heatmaps of the per-class attention matrices $\mathbf { A } _ { c } \in \mathbb { R } ^ { d \times d }$ , capturing complex feature interaction geometries and cross-dimensional weightings for each hazard class.

![](images/00bdfaf97f866006a2a372e62d7536c5d2769015603769a990e6bcf5600362fc.jpg)

![](images/84e2ddc50b1e04652d634b4b0f349376d9b58d7c7db779944ee6553c4f8247e8.jpg)

![](images/ccdda0235dbfd30a6358565caec085de17cde125e04f4954f40df4af48462c4a.jpg)

![](images/747ea35c23d10270e880f309138634538114c95d00941fca22bd6ae2032822da.jpg)

![](images/fe3ce88a9c55adea578931de29d2a18df02dfefdf63b375f4e22111ce0580d54.jpg)  
Figure 6: Plot of ALS vs. Logit diagnostics. Gray points represent hazard-free samples, while red points denote hazardous samples for class c. The decision boundary is established at logit = 0 (corresponding to a classification threshold of 0.5)

![](images/c079cf4c94253e01e67c855818c6024016ca109a188af68bcfd5dcbd0d2622b3.jpg)

![](images/00338eacec641e924cc23cf961d5cadafd6cd75d259a487351855efb0e1c5af3.jpg)

![](images/699e0bf76a1aba271c7a56c57ce703550a3564c1b7159bc8eae5bc7cbb5a4483.jpg)

![](images/a2a468e40aeaed1c3841c3995840ba380df5cfc0f24182048f102c3864c702fd.jpg)

![](images/ec7968fc4f16a8703a5dee3bc6f99cf189fb883054ad58bbed9c6cb7901dff46.jpg)  
Figure 7: Plot of Training samples used to train the incremental hazard diagnosis model. Illustrating representational consistency during incremental training (Algorithm 2). Current streaming training samples are marked in blue circles (Stream), batch-wise retrieved raw samples from $\boldsymbol { B } _ { x }$ are shown as green diamonds, and retrieved latent representations from $B _ { z }$ are shown as orange triangles. The spatial overlap between the modalities highlights the consistency and alignment maintained by the dual-replay memory bufer across classes.

## References

[1] Xiaoyu Chen, Lening Wang, Canran Wang, and Ran Jin. Predictive ofloading in mobile-fog-cloud enabled cyber-manufacturing systems. In 2018 IEEE Industrial Cyber-Physical Systems (ICPS), pages 167–172, 2018.

[2] Yingyan Zeng, Ismini Lourentzou, Xinwei Deng, and Ran Jin. Fair: Facilitating artificial intelligence resilience in manufacturing industrial internet. arXiv preprint arXiv:2503.01086, 2025.

[3] Jorge F Arinez, Qing Chang, Robert X Gao, Chengying Xu, and Jianjing Zhang. Artificial intelligence in advanced manufacturing: Current status and future outlook. Journal of Manufacturing Science and Engineering, 142(11):110804, 2020.

[4] Xiaoyu Chen and Ran Jin. AdaPipe: A recommender system for adaptive computation pipelines in cybermanufacturing computation services. IEEE Transactions on Industrial Informatics, 17(9):6221–6229, 2020.

[5] Parshin Shojaee, Yingyan Zeng, Xiaoyu Chen, Ran Jin, Xinwei Deng, and Chuck Zhang. Deep neural network pipelines for multivariate time series classification in smart manufacturing. In 2021 4th IEEE International Conference on Industrial Cyber-Physical Systems (ICPS), pages 98–103, 2021.

[6] Craig Poulin and Michael B Kane. Infrastructure resilience curves: Performance measures and summary metrics. Reliability Engineering & System Safety, 216:107926, 2021.

[7] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366–3385, 2021.

[8] Marc Masana, Xialei Liu, Bartłomiej Twardowski, Mikel Menta, Andrew D Bagdanov, and Joost van de Weijer. Class-incremental learning: Survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5513–5533, 2023.

[9] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pages 109–165. 1989.

[10] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elho seiny, Thalaiyasingam Ajanthan, P Dokania, P Torr, and M Ranzato. Continual learning with tiny episodic memories. In Workshop on Multi-Task and Lifelong Reinforcement Learning, 2019.

[11] Sylvestre-Alvise Rebufi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[12] Gido M van de Ven, Tinne Tuytelaars, and Andreas S Tolias. Three types of incremental learning. Nature Machine Intelligence, 4(12):1185–1197, 2022.

[13] Shen Yan, Haidong Shao, Xudong Wang, and Jie Wang. Few-shot class-incremental learning for system-level fault diagnosis of wind turbine. IEEE/ASME Transactions on Mechatronics, 2024.

[14] Martial Mermillod, Aurélia Bugaiska, and Patrick Bonin. The stability-plasticity dilemma: Investigating the continuum from catastrophic forgetting to age-limited learning efects. Frontiers in Psychology, 4:504, 2013.

[15] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 374–382, 2019.

[16] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin. Learning a unified classifier incrementally via rebalancing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 831–839, 2019.

[17] Zheda Mai, Ruiwen Li, Jihwan Jeong, David Quispe, Hyunwoo Kim, and Scott Sanner. Online continual learning in image classification: An empirical survey. Neurocomputing, 469:28–51, 2022.

[18] Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, and Jonghyun Choi. Rainbow memory: Continual learning with a memory of diverse samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8218–8227, 2021.

[19] Chris Dongjoo Kim, Jinseo Jeong, and Gunhee Kim. Imbalanced continual learning with partitioning reservoir sampling. In European Conference on Computer Vision, pages 411–428, 2020.

[20] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

[21] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017.

[22] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

[23] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International Conference on Machine Learning, pages 3987–3995, 2017.

[24] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

[25] Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[26] Jefrey S Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software (TOMS), 11(1):37–57, 1985.

[27] Aristotelis Chrysakis and Marie-Francine Moens. Online continual learning from imbalanced data. In International Conference on Machine Learning, pages 1952–1961. PMLR, 2020.

[28] Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. In Advances in Neural Information Processing Systems, volume 32, 2019.

[29] Max Welling. Herding dynamical weights to learn. In Proceedings of the 26th annual international conference on machine learning, pages 1121–1128, 2009.

[30] Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. Online continual learning with maximal interfered retrieval. In Advances in Neural Information Processing Systems, volume 32, 2019.

[31] Dongsub Shim, Zheda Mai, Jihwan Jeong, Scott Sanner, Hyunwoo Kim, and Jongseong Jang. Online class-incremental continual learning with adversarial Shapley value. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 9630–9638, 2021.

[32] Lorenzo Pellegrini, Gabriele Grafieti, Vincenzo Lomonaco, and Davide Maltoni. Latent replay for real-time continual learning. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10203–10209, 2020.

[33] Tyler L Hayes, Kushal Kafle, Robik Shrestha, Manoj Acharya, and Christopher Kanan. REMIND your neural network to prevent catastrophic forgetting. In European Conference on Computer Vision, pages 466–483, 2020.

[34] Songsong Tian, Lusi Li, Weijun Li, Hang Ran, Xin Ning, and Prayag Tiwari. A survey on few-shot classincremental learning. Neural Networks, 169:307–324, 2024.

[35] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017.

[36] Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. arXiv preprint arXiv:1807.05960, 2018.

[37] Da-Wei Zhou, Han-Jia Ye, Liang Ma, Di Xie, Shiliang Pu, and De-Chuan Zhan. Few-shot class-incremental learning by sampling multi-phase tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):12816–12831, 2022.

[38] Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and Andrew Y Ng. Multimodal deep learning. In International Conference on Machine Learning, pages 689–696, 2011.

[39] Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2):423– 443, 2019.

[40] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.

[41] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, volume 30, 2017.

[42] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems, volume 30, 2017.

[43] Arslan Chaudhry, Puneet K Dokania, Thalaiyasingam Ajanthan, and Philip HS Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In European Conference on Computer Vision, pages 532–547, 2018.

[44] John von Neumann. Various techniques used in connection with random digits. In Monte Carlo Method, volume 12 of National Bureau of Standards Applied Mathematics Series, pages 36–38. 1951.

## Appendix

## A Proof of Proposition 1

The quotas never increase, because $| \mathcal { C } _ { t } |$ is nondecreasing in t. A partition of a new class starts empty with $N _ { c } = 0$ An arrival of class c increases $| B [ c ] |$ by one if $| B [ c ] | <$ $q _ { c }$ and leaves it unchanged otherwise, so with a fixed quota, $| B [ c ] | = \operatorname* { m i n } \{ q _ { c } , N _ { c } \}$ by induction over arrivals. When the quota decreases from $q _ { c }$ to $q _ { c } ^ { \prime } < q _ { c } .$ , eviction leaves min $\{ q _ { c } ^ { \prime } , \mathrm { m i n } \{ q _ { c } , N _ { c } \} \} = \mathrm { m i n } \{ q _ { c } ^ { \prime } , \bar { N } _ { c } \}$ entries, and the induction continues with $q _ { c } ^ { \prime }$ □

## B Proof of Lemma 1 and Exact Replay Error

For a measurable set A and $N _ { c } > q _ { c }$ , conditioning on admission gives

$$
\operatorname* { P r } \left( ( z , y ) \in { \cal A } \mid \mathrm { a d m i s s i o n } \right) = \frac { \int _ { \cal A } ( q _ { c } / N _ { c } ) h _ { c } d \pi _ { c } } { q _ { c } \bar { h } _ { c } / N _ { c } } = \tilde { \pi } _ { c } ( { \cal A } ) ,
$$

which is the acceptance–rejection identity [44]. The process can therefore be represented by independent replacement events with probabilities $q _ { c } \bar { h } _ { c } / N _ { c }$ , independent uniform slot choices, and independent replacement samples with law $\tilde { \pi } _ { c }$ . A fixed slot is replaced at arrival j with probability $\bar { h } _ { c } / j$ , which gives Eq. (4.3); hence the number $\nu _ { c }$ of surviving initial entries satisfies $\mathbb { E } \nu _ { c } = q _ { c } \kappa _ { c } ( N _ { c } )$ . Couple $B [ c ]$ with a partition initialized with i.i.d. entries from $\tilde { \pi } _ { c }$ and subjected to the same replacements. The coupled partition keeps the law $\tilde { \pi } _ { c } ^ { \otimes q _ { c } }$ , and the two partitions coincide once every initial entry has been replaced, so the total variation distance is at most $\mathrm { P r } ( \nu _ { c } \geq 1 ) \leq \mathbb { E } \nu _ { c } = q _ { c } \kappa _ { c } ( N _ { c } )$ Since $\sum _ { j } \bar { h } _ { c } / j$ diverges, $\kappa _ { c } ( N _ { c } )  0$ , which proves convergence. □

Lemma 2 (Exact replay error). Let f be a statistic with $\mathbb { E } _ { \pi _ { c } } \| \mathbf { f } \| _ { 2 } ^ { 2 } < \infty$ , with means m and m˜ and covariance traces V<sub>f</sub> and V<sup>˜</sup><sub>f</sub> under $\pi _ { c }$ and $\tilde { \pi } _ { c } ,$ and let $\delta _ { \mathbf { f } } = \tilde { \mathbf { m } } - \mathbf { m }$ and $\begin{array} { r } { \overline { { \bf f } } = q _ { c } ^ { - 1 } \sum _ { B [ c ] } { \bf f } . } \end{array}$ . Then

$$
\mathbb { E } \| \bar { \mathbf { f } } - \mathbf { m } \| _ { 2 } ^ { 2 } = \frac { ( 1 - \kappa _ { c } ) \tilde { V } _ { \mathbf { f } } + \kappa _ { c } V _ { \mathbf { f } } } { q _ { c } } + \beta _ { c } \| \pmb { \delta } _ { \mathbf { f } } \| _ { 2 } ^ { 2 } ,\tag{B.1}
$$

where $\kappa _ { c } = \kappa _ { c } ( N _ { c } ) , \ \beta _ { c } = \beta _ { c } ( N _ { c } ) = \mathbb { E } ( 1 - \nu _ { c } / q _ { c } ) ^ { 2 } , \ ( 1 -$ $\kappa _ { c } ) / q _ { c } \leq \beta _ { c } \leq 1 - \kappa _ { c } ,$ and $\beta _ { c }  1$ as $N _ { c }  \infty$

Proof. For slot $j ,$ let $\mathbf { F } _ { j } = \mathbf { f } ( \mathrm { s l o t } ~ j ) - \tilde { \mathbf { m } }$ . Conditional on the replacement schedule, the slots are independent, with law $\pi _ { c }$ for surviving initial entries and $\tilde { \pi } _ { c }$ otherwise. Hence $\mathbb { E } \| \mathbf { F } _ { j } \| _ { 2 } ^ { 2 } = \kappa _ { c } ( V _ { \mathbf { f } } + \| \pmb { \delta } _ { \mathbf { f } } \| _ { 2 } ^ { 2 } ) + ( 1 - \kappa _ { c } ) \tilde { V } _ { \mathbf { f } } ,$ . For $q _ { c } \geq 2$ two distinct initial entries both survive with probability $\begin{array} { r } { \kappa _ { c } ^ { ( 2 ) } = \prod _ { i = q _ { c } + 1 } ^ { N _ { c } } ( 1 - 2 \bar { h } _ { c } / j ) } \end{array}$ , so $\mathbb { E } \mathbf { F } _ { j } ^ { \top } \mathbf { F } _ { k } = \kappa _ { c } ^ { ( 2 ) } \vert \vert \delta _ { \mathbf { f } } \vert \vert _ { 2 } ^ { 2 }$ for $j \neq k$ , and $\bar { \mathbb { E } } \mathbf { \breve { F } } _ { j } = - \kappa _ { c } \delta _ { \mathbf { f } }$ . Expanding the second moment of $q _ { c } ^ { - 1 } \sum _ { j } \mathbf { F } _ { j } \dot { + } \delta _ { \mathbf { f } }$ gives Eq. (B.1) with $\beta _ { c } = 1 - 2 \kappa _ { c } +$ $\kappa _ { c } / q _ { c } + ( q _ { c } - 1 ) \kappa _ { c } ^ { ( 2 ) } / q _ { c }$ , where the last term vanishes for $q _ { c } = 1$ . The identities $\mathbb { E } \nu _ { c } = q _ { c } \kappa _ { c }$ and $\mathbb { E } \nu _ { c } ^ { 2 } = q _ { c } \kappa _ { c } + q _ { c } ( q _ { c } -$ $1 ) \kappa _ { c } ^ { ( 2 ) }$ identify $\beta _ { c }$ with $\mathbb { E } ( 1 - \nu _ { c } / q _ { c } ) ^ { 2 }$ . Since $1 - \nu _ { c } / q _ { c } \in$ $\{ 0 , 1 / q _ { c } , \ldots , 1 \}$ , its square lies between $( 1 - \nu _ { c } / q _ { c } ) / q _ { c }$ and $1 - \nu _ { c } / q _ { c } ,$ , which gives the bounds. Finally, $\kappa _ { c }  0$ implies $\nu _ { c } / q _ { c } \to 0$ in probability, and boundedness gives $\beta _ { c }  1$ □

## C Proof of Theorem 1

In Eq. (4.6), the residual factor is bounded by max $\{ w _ { c ^ { \prime } } , 1 \}$ and $\| \dot { \psi _ { c ^ { \prime } } } ( z ) \| _ { 2 } ^ { 2 } \leq \| z \| _ { 2 } ^ { 2 } \| \mathbf { A } _ { c ^ { \prime } } z \| _ { 2 } ^ { 2 } \leq \| \mathbf { A } _ { c ^ { \prime } } \| _ { \mathrm { o p } } ^ { 2 } \| z \| _ { 2 } ^ { 4 }$ . Hence $\mathbf { g }$ is square integrable under $\pi _ { c }$ and, since $h _ { c } \leq 1$ , under $\tilde { \pi } _ { c } .$ Lemma 2 with $\mathbf { f } = \mathbf { g } , \kappa _ { c }  0$ , and $\beta _ { c }  1$ gives the limit in Eq. (4.8). For CBRS, $\tilde { \pi } _ { c } = \pi _ { c }$ and $\delta _ { c } = \mathbf { 0 }$ , so Eq. (B.1) equals $V _ { c } / q _ { c }$ for every $N _ { c } \geq q _ { c }$

For the margin, let $\Delta = \hat { \mathbf { v } } - \mathbf { v } ^ { \star }$ . The curvature condition and $\nabla \hat { \mathcal { I } } ( \hat { \mathbf { v } } ) = \mathbf { 0 }$ give

$$
\mu \| \boldsymbol { \Delta } \| _ { 2 } ^ { 2 } \leq \boldsymbol { \Delta } ^ { \top } \big ( \nabla \hat { \mathcal { I } } ( \hat { \mathbf { v } } ) - \nabla \hat { \mathcal { I } } ( \mathbf { v } ^ { \star } ) \big ) \leq \| \boldsymbol { \Delta } \| _ { 2 } \| \nabla \hat { \mathcal { I } } ( \mathbf { v } ^ { \star } ) \| _ { 2 } .
$$

Since $\nabla \mathcal { I } ( \mathbf { v } ^ { \star } ) = \mathbf { 0 }$ , the common term Ω cancels and $\nabla \hat { \mathcal { I } } ( \mathbf { v } ^ { \star } ) = \bar { \mathbf { g } } - \mathbb { E } _ { \pi _ { c } } \mathbf { g }$ , so $\mathbb { E } \| \Delta \| _ { 2 } ^ { 2 } \le { \mathscr { E } _ { c } ( N _ { c } ) / \mu ^ { 2 } }$ . Finally, $\gamma ( \hat { \mathbf { v } } ) - \gamma ( \mathbf { v } ^ { \star } ) = ( 2 y _ { * , c ^ { \prime } } - 1 ) \psi _ { c ^ { \prime } } ( z _ { * } ) ^ { \top } \Delta$ , and the Cauchy– Schwarz inequality gives Eq. (4.9). The same argument applies to CBRS. □

## D Proof of Corollary 1

Lemma 2 with $\mathbf { f } = \mathbf { g }$ gives

$$
\mathcal { E } _ { c } ( N _ { c } ) - \mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } ) = \frac { 1 - \kappa _ { c } } { q _ { c } } \big ( \tilde { V } _ { c } - V _ { c } \big ) + \beta _ { c } \| \pmb { \delta } _ { c } \| _ { 2 } ^ { 2 } .\tag{D.1}
$$

If $q _ { c } < q _ { c } ^ { \star }$ , then $( V _ { c } - \tilde { V } _ { c } ) / q _ { c } > \| \delta _ { c } \| _ { 2 } ^ { 2 }$ , and $\beta _ { c } \le 1 - \kappa _ { c }$ yields

$$
\mathcal { E } _ { c } ( N _ { c } ) - \mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } ) \leq ( 1 - \kappa _ { c } ) \Big ( \| \delta _ { c } \| _ { 2 } ^ { 2 } - \frac { V _ { c } - \tilde { V } _ { c } } { q _ { c } } \Big ) < 0
$$

for $N _ { c } > q _ { c } ,$ , since $\kappa _ { c } < 1$ there. The margin bounds inherit this ordering whenever $\psi _ { c ^ { \prime } } ( z _ { * } ) \neq \mathbf { 0 }$ . As $N _ { c }  \infty ,$ the diference tends to $\lVert \delta _ { c } \rVert _ { 2 } ^ { 2 } - ( V _ { c } - \tilde { V } _ { c } ) / q _ { c } ,$ which is positive for $q _ { c } > q _ { c } ^ { \star }$ . For the second form of $q _ { c } ^ { \star } .$ , a change of measure gives

$$
\begin{array} { r l r } & { } & { \tilde { V } _ { c } = \mathbb { E } _ { \tilde { \pi } _ { c } } \| { \bf g } - \mathbb { E } _ { \pi _ { c } } { \bf g } \| _ { 2 } ^ { 2 } - \| \pmb { \delta } _ { c } \| _ { 2 } ^ { 2 } } \\ & { } & { = V _ { c } + \frac { \mathrm { C o v } _ { \pi _ { c } } \left( h _ { c } , \| { \bf g } - \mathbb { E } _ { \pi _ { c } } { \bf g } \| _ { 2 } ^ { 2 } \right) } { \bar { h } _ { c } } - \| \pmb { \delta } _ { c } \| _ { 2 } ^ { 2 } , } \end{array}
$$

and substitution into Eq. (4.10) completes the proof. At a finite $N _ { c } ~ > ~ q _ { c }$ , Eq. (D.1) gives the exact criterion $\mathcal { E } _ { c } ( N _ { c } ) < \mathcal { E } _ { c } ^ { \mathrm { C B R S } } ( N _ { c } )$ if and only if $( 1 - \kappa _ { c } ) ( V _ { c } - \tilde { V } _ { c } ) >$ $q _ { c } \beta _ { c } \| \pmb { \delta } _ { c } \| _ { 2 } ^ { 2 }$ □