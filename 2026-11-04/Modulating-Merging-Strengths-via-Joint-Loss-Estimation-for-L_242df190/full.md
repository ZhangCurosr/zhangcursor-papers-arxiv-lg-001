# Modulating Merging Strengths via Joint Loss Estimation for LoRA-based Continual Learning

Kun Gu<sup>1</sup> De Cheng<sup>1,B</sup> Zhipeng Xu<sup>1</sup> Lingfeng He<sup>1</sup> Di Xu<sup>2</sup> Nannan Wang<sup>1</sup>

<sup>1</sup>Xidian University <sup>2</sup>Huawei Technologies Co., Ltd.

{kung@stu., dcheng@, xu\_zhipeng@stu., lfhe@stu., nnwang@}xidian.edu.cn xudi21@huawei.com

## Abstract

Continual learning (CL) aims to adapt models to new tasks while preserving previously learned knowledge. Recently, Low-Rank Adaptation (LoRA), a representative Parameter-Efficient Fine-Tuning (PEFT) method, has gained increasing attention in CL due to its efficiency and scalability. Several LoRA-based merging methods maintain a single inference model by scaling the newly learned LoRA update with one coefficient and merging it into the previous weights after each task. However, they usually control the whole update with a shared scalar factor, although different update elements may affect previous and current tasks in different ways. To address this issue, we propose Merging Strength Modulation via Joint Loss Estimation (M<sup>2</sup>LE). It consists of Hessian Information-guided Adaptive Merging (HIM) and Task Simulation Perturbation for Merging (TSP). Instead of searching for a global merge coefficient, HIM directly solves for the target merged update under a joint objective over previous and current tasks. Solving this objective gives an analytic solution that naturally takes the form of element-wise modulation over the current task LoRA update. To improve the stability of the learned update under subsequent merging, we introduce TSP which simulates different tasks interference. Experiments on multiple benchmarks show that M<sup>2</sup>LE consistently outperforms existing PEFT-based continual learning methods. Code is available at https://github.com/bbdeeer/M2LE\_NeurIPS2026.

## 1 Introduction

Continual learning (CL) aims to enable models to learn continuously from a stream of tasks without retraining from scratch or accessing data from previous tasks [13, 25], which is essential for evolving open-world scenarios. A central challenge in CL is catastrophic forgetting: models must preserve previously learned capabilities during adapting to new tasks [17, 26]. In this work, we consider the challenging rehearsal-free Continual Learning setting, where the model must continually learn new tasks without storing past data and predict test samples over all seen tasks without task identities.

Recently, with the advancement of Pre-Trained Models (PTMs) [4], Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a promising solution for CL [31, 35, 43]. By updating a small number of paramters, PEFT-based methods enable efficient adaptation to new tasks while preserving general pre-trained knowledge, making it particularly suitable for rehearsal-free CL.

In the regime of PEFT-based CL, Low-Rank Adaptation (LoRA) [11] methods have gained increasing attention due to their efficiency and scalability. To maintain a single inference model, a common strategy is to merge the current task LoRA update into the accumulated model after each task [28], rather than storing a separate LoRA module per task. This controlled merging reduces interference with prior knowledge while enabling adaptation to the current task. Early studies mainly adopt EMAstyle interpolation [20] or weight averaging [6], while recent approaches estimate the interpolation coefficient from joint objectives [28] or Bayesian formulations [16]. Despite these different designs, most existing methods still use a shared scalar factor to control the whole LoRA update. This naturally raises a question: should all LoRA parameters share the same merge strength, or should different positions in the LoRA update be treated differently during merging? To answer this question, we conduct a preliminary analysis in Fig. 1, where the LoRA update is split into two groups according to different normalized magnitudes [12]. We then assign an independent merge coefficient to each group. The results show that this simple grouped merge already outperforms the best shared factor merge, suggesting that different parts of the LoRA update may prefer different merge strengths.

Building on this observation, we propose a Merging Strength Modulation via Joint Loss Estimation $( \mathrm { M } ^ { \mathrm { 2 } } \mathrm { L E } )$ framework for LoRA-based Continual Learning, consists of Hessian Information-guided Adaptive Merging (HIM) and Task Simulation Perturbation for Merging (TSP). Instead of searching for an interpolation coefficient, HIM directly formulates the optimization of the target merged update ∆W<sup>∗</sup> under a joint objective that accounts for losses on all seen tasks. Through a local quadratic approximation to the task-wise losses, we obtain an analytic solution, which naturally takes the form of an element-wise modulation of the current task LoRA update. Specifically, each element is adaptively scaled by the associated Hessian information, allowing the merging process to account for the varying sensitivity of individual parameter elements across tasks, thereby achieving a more fine-grained balance between old and new task performance. However, once the current task update is merged into the inference model, it becomes part of the fixed parameters and will be affected by subsequent task updates. If the merged update lies in a sharp local region, later merging steps can easily increase the loss on old tasks, resulting in forgetting. Therefore, besides determining how to merge the update, it is also important to shape the update itself before merging. To this end, we introduce TSP, which simulates cross-task interference during current task training. TSP induces a local parameter change with one subset of the current mini-batch and evaluates the resulting loss variation on another subset, thereby encouraging the learned update to remain stable during subsequent merging.

Our main contributions can be summarized as follows:

• We propose Hessian Information-guided Adaptive Merging (HIM), a parameter-adaptive merging algorithm derived from an approximated joint loss objective, enabling more finegrained stability-plasticity balancing beyond a shared scalar merging coefficient.

• We propose Task Simulation Perturbation (TSP), a training strategy that simulates future cross-task interference through local parameter perturbations induced from label-disjoint subset, thereby encouraging flat solutions and improving LoRA merging stability.

• Extensive experiments demonstrate the effectiveness of our proposed method, achieving superior performances over state-of-the-art PEFT-based CL methods on multiple benchmarks.

## 2 Related Work

PEFT-based and LoRA-based continual learning. PEFT-based CL [50] typically keeps a pretrained model frozen and introduces lightweight learnable modules for continual learning. L2P [43] retrieves instance specific prompts from a prompt pool, and SECA [8] uses textual semantic priors to support knowledge transfer and prototype refinement in CLIP-based continual learning. EWC [13] preserves previous knowledge by penalizing changes to important parameters, and EKPC [38] mitigates forgetting through elastic knowledge preservation and compensation. Among PEFT-based method [1–3, 39, 46–48, 52], LoRA has become a representative choice for continual learning due to its parameter efficiency and compatibility with frozen pre-trained backbones [11]. InfLoRA [18] constrains new LoRA updates to interference-free subspaces through orthogonality, PLAN [41] proactively allocates low-rank subspaces and optimizes them against worst-case perturbations, Gain LoRA [19] expands a new LoRA for each task and introduces gating modules to adaptively weight the old and new branches, and MACIL [44] mitigates forgetting through semantic drift calibration. LoDA [7] continually decomposes LoRA updates into general and isolated subspaces for better stability-plasticity trade-off.

Model Merging. Model merging aims to integrate weights trained on different tasks into a single network [33]. RobustMerge [49] improves parameter-efficient merging by enhancing the robustness of low-rank update directions, while OPCM [36] enables sequential merging through orthogonal projection and adaptive scaling. This idea has also been used in continual learning, where the current model or update is merged into the inference model after each task [12]. ACMap [6] averages the adapters learned from different tasks into a shared adapter, BECAME [16] derives an adaptive coefficient from a Bayesian formulation, and LoRA-P&M [28] combines the previous and current models using a shared merging factor computed in closed form. However, these methods mainly rely on a shared factor to interpolate updates, while our method derives the merged update from the loss changes of old and new tasks under a joint objective.

## 3 Preliminary Analysis

Research Question. Should LoRA parameters at different positions be treated differently during merging, rather than being controlled by a sharedfactor?

Experiment Setting. As a case study, we consider the 10 task setting on ImageNet-R and focus on the first merge step. Specifically, we take the model after learning task 0 and the update learned on task 1 for analysis. Let $\mathbf { W } _ { 0 }$ denote the inference parameter after learning task 0, and let $\Delta { \bf W } _ { 1 }$ denote the LoRA update learned on task 1. To this end, we build a simple merging setup. Specifically, we compute the normalized magnitude map of $\Delta \mathbf { W } _ { 1 }$ as $\mathbf { R } = | \Delta \mathbf { W } _ { 1 } | / \operatorname* { m a x } _ { i , j } | \Delta \mathbf { W } _ { 1 } ^ { ( i , j ) } |$ [12], and divide the update into two groups: elements with normalized values below 0.5, and elements with normalized values at least 0.5. We then assign two independent coefficients, $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , to the two groups, and define the merged model as: $\mathbf { W } ( \bar { \alpha } _ { 1 } , \alpha _ { 2 } ) = \bar { \mathbf { W } } _ { 0 } + \alpha _ { 1 } ( \mathbf { M } _ { 1 } \odot \Delta \mathbf { W } _ { 1 } ) + \alpha _ { 2 } ( \mathbf { M } _ { 2 } \odot \Delta \mathbf { W } _ { 1 } )$ where ${ { \bf { M } } _ { 1 } }$ and $\mathbf { M } _ { 2 }$ denote the corresponding masks, and ⊙ denotes element-wise multiplication. By varying $\alpha _ { 1 }$ and $\alpha _ { 2 }$ , we evaluate the joint loss on task 0 and task 1.

Observation. Fig. 1 shows that the low loss region is not centered on the diagonal $\alpha _ { 1 } = \alpha _ { 2 }$ , but is shifted toward an off-diagonal region. Consistently, the green diamond marks the point with the best average accuracy on task0 and task1, which is located at (0.40, 0.80), while the purple diamond marks the best scalar merge on the diagonal, located at (0.60, 0.60). Compared with the scalar merge, the green-diamond point improves the average accuracy from 91.04% to 91.84%.

Insight. These results suggest that a shared scalar coefficient is overly restrictive: different parameter groups do not, in general, share the same preferred merge strength, and should therefore be treated differently during merging.

![](images/101c9b62618f9d08b6b42c84597ebaaa00749b3b06725a9c8b7191f32ed20f37.jpg)  
Figure 1: Joint loss landscape of the two group merge on 10S-ImageNet-R.

## 4 Methodology

## 4.1 Preliminaries

In $\mathrm { C L } , { \mathrm { a } }$ model learns from a sequence of tasks $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T }$ . The dataset of task t is denoted by $\mathcal { D } _ { t } = \{ ( x _ { t } ^ { i } , y _ { t } ^ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ , where $x _ { t } ^ { i } \in \mathbb { R } ^ { d }$ is the input and $y _ { t } ^ { i } \in { \mathcal { C } } _ { t }$ is the corresponding label. Each task introduces a set of new classes, and the label spaces of different tasks are mutually disjoint, i.e., $\mathcal { C } _ { i } \cap \mathcal { C } _ { j } = \mathcal { O }$ for any $i \neq j$ . Accordingly, the cumulative label space observed up to task t is $\textstyle { \mathcal { C } } _ { 1 : t } = \bigcup _ { i = 1 } ^ { t } { \mathcal { C } } _ { i }$ . Under the standard rehearsal-free CL setting, only the current task data $\mathcal { D } _ { t }$ is accessible when learning task t. The goal is to learn tasks sequentially while maintaining strong performance over all tasks observed so far.

## 4.2 Overall framework

Let $\mathbf { W } _ { t - 1 } \in \mathbb { R } ^ { d _ { o } \times d _ { i } }$ denote the inference parameter after learning task $t - 1$ , where $d _ { i }$ and $d _ { o }$ are the input and output dimensions. At t-th task, for each ViT layers, we freeze the backbone weight $\mathbf { W } _ { t - 1 }$ and attach a LoRA module. The LoRA update is parameterized by two low-rank matrices:

$$
\Delta \mathbf { W } _ { t } = \mathbf { B } _ { t } \mathbf { A } _ { t } , \quad \mathbf { A } _ { t } \in \mathbb { R } ^ { r \times d _ { i } } , \quad \mathbf { B } _ { t } \in \mathbb { R } ^ { d _ { o } \times r } ,\tag{1}
$$

where $r$ is the LoRA rank. A common LoRA merging strategy directly updates the inference model by scaling the current task update with a shared coefficient $\begin{array} { r } { \dot { \mathbf { W } _ { t } } = \mathbf { W } _ { t - 1 } + \alpha _ { t } \Delta \mathbf { W } _ { t } } \end{array}$ , where the scalar $\alpha _ { t }$ controls the merge strength of the whole update. However, this scalar merge applies the same merge strength to all elements in $\Delta \mathbf { W } _ { t } .$ , although different elements may have different effects on previous and current tasks.

To address this, $\mathbf { M } ^ { 2 } \mathbf { I }$ LE consists of two components: Hessian Information-guided Adaptive Merging (HIM) and Task Simulation Perturbation for Merging (TSP). During training, TSP regularizes the LoRA update by simulating cross task perturbations, encouraging the learned update $\Delta \bar { \mathbf { W } _ { t } }$ <sub>t</sub> to improve the stability during subsequent merging. After training, HIM derives a target merged update $\Delta \mathbf { W } _ { t } ^ { * }$ under a joint loss impact objective over previous and current tasks, instead of searching for a global merge coefficient. The inference parameter is then updated by $\mathbf { W } _ { t } = \mathbf { W } _ { t - 1 } + \Delta \mathbf { W } _ { t } ^ { * }$ . An overview of the framework and the complete training procedure are provided in Appendix A (Figure 5 and Algorithm 1).

## 4.3 Hessian Information-guided Adaptive Merging

Existing LoRA-based merging methods for CL always merge the current task LoRA update into the inference model using a shared scalar factor. This forces all parameter elements to be merged with the same strength, although they may affect previous and current tasks differently. This motivates us to account for the different preferred merge strengths of update parameter during merging.

Specifically, we directly solve for the target merged update $\Delta \mathbf { W } _ { t } ^ { * } \in \mathbb { R } ^ { d _ { o } \times d _ { i } }$ under a joint objective over previous and current tasks. Intuitively, we want this update to incorporate new knowledge as effectively as possible while reducing catastrophic forgetting on previous tasks. To this end, we first define $\mathcal { I } _ { \mathrm { o l d } }$ and $\mathcal { I } _ { \mathrm { n e w } }$ to measure how an ideal merged update $\Delta \hat { \mathbf W } _ { t } \in \mathbb R ^ { d _ { o } \times d _ { i } }$ affects the losses on previous and current tasks:

$$
\mathcal { I } _ { \mathrm { o l d } } ( \Delta { \mathbf { \hat { W } } _ { t } } ) = \sum _ { i = 1 } ^ { t - 1 } \left( \mathcal { L } _ { i } ( \mathbf { W } _ { t - 1 } + \Delta { \mathbf { \hat { W } } _ { t } } ) - \mathcal { L } _ { i } ( \mathbf { W } _ { t - 1 } ) \right) ,\tag{2}
$$

$$
\mathcal { I } _ { \mathrm { n e w } } \big ( \Delta \mathbf { \hat { W } } _ { t } \big ) = \mathcal { L } _ { t } \big ( \mathbf { W } _ { t - 1 } + \Delta \mathbf { \hat { W } } _ { t } \big ) - \mathcal { L } _ { t } \big ( \mathbf { W } _ { t - 1 } + \Delta \mathbf { W } _ { t } \big ) .\tag{3}
$$

Here, $\Delta \mathbf { W } _ { t } \in \mathbb { R } ^ { d _ { o } \times d _ { i } }$ denotes the update learned for task $t , { \mathcal { L } } _ { i }$ denotes the loss of task i. Therefore, we seek a merged update that jointly minimizes $\mathcal { I } _ { \mathrm { o l d } }$ on previous tasks and $\mathcal { I } _ { \mathrm { n e w } }$ on the current task, with a hyper-parameter λ controlling the strength of the current task term. The target merged update $\Delta \mathbf { W } _ { t } ^ { * }$ is obtained by solving:

$$
\Delta \mathbf { W } _ { t } ^ { * } = \arg \operatorname* { m i n } _ { \Delta \mathbf { \hat { W } } _ { t } } \mathcal { I } _ { \mathrm { o l d } } ( \Delta \mathbf { \hat { W } } _ { t } ) + \lambda \mathcal { I } _ { \mathrm { n e w } } ( \Delta \mathbf { \hat { W } } _ { t } ) .\tag{4}
$$

In the rehearsal-free setting, Eq. (4) cannot be optimized directly, since the old-task term $\mathcal { I } _ { \mathrm { o l d } }$ depends on previous task data that are no longer accessible when learning task t. To obtain a tractable approximation, we approximate Eq. (2) and Eq. (3) with local second-order Taylor expansions. For notational simplicity, we vectorize $\Delta \hat { \mathbf W }$ <sub>t</sub> into $\Delta \hat { \mathbf { w } } _ { t } = \mathrm { v e c } ( \Delta \hat { \mathbf { W } } _ { t } ) \in \mathbb { R } ^ { d _ { o } d _ { i } \times 1 }$ . For the old task term $\mathcal { I } _ { \mathrm { o l d } } ( \Delta \mathbf { \hat { w } } _ { t } )$ , expanding around $\mathbf { w } _ { t - 1 }$ gives:

$$
\mathcal { I } _ { \mathrm { o l d } } ( \Delta { \mathbf { \hat { w } } } _ { t } ) \approx \sum _ { i = 1 } ^ { t - 1 } \left( \mathbf { g } _ { i } ^ { \top } \hat { \Delta \mathbf { w } } _ { t } + \frac { 1 } { 2 } \hat { \Delta \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { i } \hat { \Delta \mathbf { w } } _ { t } \right) ,\tag{5}
$$

where $\mathbf { g } _ { i } \in \mathbb { R } ^ { d _ { o } d _ { i } \times 1 }$ and $\mathbf { H } _ { i } \in \mathbb { R } ^ { d _ { o } d _ { i } \times d _ { o } d _ { i } }$ denote the first and second terms of the i-th old task loss around $\mathbf { w } _ { t - 1 }$ , respectively. Similarly, for the current task, expanding around $\mathbf { w } _ { t - 1 } + \Delta \mathbf { w } _ { t }$ yields:

$$
\mathcal { I } _ { \mathrm { n e w } } ( \Delta \mathbf { \hat { w } } _ { t } ) \approx \mathbf { g } _ { t } ^ { \top } ( \Delta \mathbf { \hat { w } } _ { t } - \Delta \mathbf { w } _ { t } ) + \frac { 1 } { 2 } ( \Delta \mathbf { \hat { w } } _ { t } - \Delta \mathbf { w } _ { t } ) ^ { \top } \mathbf { H } _ { t } ( \Delta \mathbf { \hat { w } } _ { t } - \Delta \mathbf { w } _ { t } ) ,\tag{6}
$$

where $\mathbf { g } _ { t }$ and $\mathbf { H } _ { t }$ denote the first- and second-order terms of the current task loss around $\mathbf { w } _ { t - 1 } + \Delta \mathbf { w } _ { t }$ respectively. Since the previous model $\mathbf { w } _ { t - 1 }$ is obtained by recursively merging past updates and $\mathbf { w } _ { t - 1 } + \Delta \mathbf { w } _ { t }$ is the learned current-task solution. Under a local approximation, we treat the first-order terms in Eq. (5) and (6) as approximately zero. Substituting the remaining quadratic terms into Eq. (4), we obtain:

$$
\begin{array} { r l } { \Delta \mathbf { W } _ { t } ^ { * } = \mathop { \arg \operatorname* { m i n } } _ { \Delta \mathbf { \hat { w } } _ { t } } } & { \overset { t - 1 } { \sum } \frac { 1 } { 2 } \Delta \mathbf { \hat { w } } _ { t } ^ { \top } \mathbf { H } _ { i } \Delta \mathbf { \hat { w } } _ { t } } \\ & { + \frac { \lambda } { 2 } ( \Delta \mathbf { \hat { w } } _ { t } - \Delta \mathbf { w } _ { t } ) ^ { \top } \mathbf { H } _ { t } \big ( \Delta \mathbf { \hat { w } } _ { t } - \Delta \mathbf { w } _ { t } \big ) . } \end{array}\tag{7}
$$

Taking the derivative with respect to $\Delta \hat { \mathbf w } _ { t }$ and setting it to zero gives:

$$
\left( \sum _ { i = 1 } ^ { t - 1 } \mathbf { H } _ { i } + \lambda \mathbf { H } _ { t } \right) \Delta \mathbf { w } _ { t } ^ { * } = \lambda \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } .\tag{8}
$$

A detailed derivation is provided in Appendix B, and Appendix C.4 further analyzes the effectiveness of accumulated historical Fisher statistics. Following EWC [13], we approximate each Hessian in Eq. (8) with the diagonal empirical Fisher, reusing their historical running sum for the old task term. For task t, the i-th diagonal element of $\mathbf { F } _ { t }$ at the learned parameters $\mathbf { W } _ { t - 1 } + \Delta \mathbf { W } _ { t }$ is defined as

$$
\begin{array} { r } { \mathbf { F } _ { t } ^ { i , i } = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { t } } \left\lceil \left( \frac { \partial \mathcal { L } ( f _ { \mathbf { W } } ( x ) , y ) } { \partial w _ { i } } \Big | _ { \mathbf { W } = \mathbf { W } _ { t - 1 } + \Delta \mathbf { W } _ { t } } \right) ^ { 2 } \right\rceil } \end{array}
$$

$$
\mathbf { F } _ { t } \in \mathbb { R } ^ { d _ { o } \times d _ { i } }
$$

Fisher of the trainable update, reshaped to match $\Delta \bar { \mathbf { W } _ { t } }$ . Under this diagonal approximation, Eq. (8) gives the following element-wise merged update:

$$
\Delta \mathbf { W } _ { t } ^ { * } = \frac { \lambda \mathbf { F } _ { t } } { \mathbf { F } _ { 1 : t - 1 } + \lambda \mathbf { F } _ { t } + \varepsilon } \odot \Delta \mathbf { W } _ { t } ,\tag{9}
$$

where ε is a small damping constant for numerical stability and after merging task $t ,$ we update the accumulated Fisher by $\mathbf { F } _ { 1 : t } = \mathbf { F } _ { 1 : t - 1 } + \mathbf { F } _ { t }$ , which is used in the next merging step.

## 4.4 Task Simulation Perturbation for Merging

According to Eq. (9), HIM obtains $\Delta \mathbf { W } _ { t } ^ { * }$ by element-wise reweighting $\Delta \mathbf { W } _ { t }$ and merge it into the inference model, where it becomes part of the fixed parameters for later tasks. If the merged update lies in a sharp region, subsequent new task merging can easily increase its loss and lead to forgetting. This naturally connects to the flat-minimum principle, making SAM [5] a straightforward way to improve the local stability of the current task update. However, SAM computes the perturbation from the overall current task loss and minimizes the perturbed loss on the same task, so its flatness constraint remains limited to the current task. Such a design makes it difficult to approximate the case in CL where the update from one task affects another task. To approximating cross-task interference when old task data are unavailable. We randomly split each mini-batch into two label-disjoint subsets $( S _ { 1 } , S _ { 2 } )$ , using one subset to induce a local parameter change and the other to evaluate the resulting loss change.

For the $S _ { 1 }  S _ { 2 }$ direction, we first compute the gradient on the source subset $\begin{array} { r } { S _ { 1 } \mathrm { { : } } } \end{array}$

$$
\mathbf { g } _ { 1 } = \nabla \mathcal { L } ( S _ { 1 } ) .\tag{10}
$$

We then construct a probing perturbation along the induced descent direction:

$$
\epsilon _ { 1 } = - \rho \frac { \mathbf { g } _ { 1 } } { \| \mathbf { g } _ { 1 } \| _ { 2 } ^ { 2 } + \varepsilon } ,\tag{11}
$$

where $\rho$ controls the perturbation scale and ε is a small constant for numerical stability. The target subset $S _ { 2 }$ is evaluated under both the perturbed and original parameters:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p e r t } } ^ { 1  2 } = \mathcal { L } ( S _ { 2 } ; \mathbf { B } _ { t } \mathbf { A } _ { t } + \epsilon _ { 1 } ) , \qquad \mathcal { L } _ { \mathrm { c l e a n } } ^ { 1  2 } = \mathcal { L } ( S _ { 2 } ; \mathbf { B } _ { t } \mathbf { A } _ { t } ) . } \end{array}\tag{12}
$$

The perturbed loss evaluates whether the direction induced by $S _ { 1 }$ remains effective on the disjoint subset $S _ { 2 }$ under a local displacement. To prevent the perturbation from pulling training too far away from the normal update and degrading the clean loss on $S _ { 2 }$ , we retain the clean loss as an anchor and define the TSP objective as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T S P } } ^ { 1  2 } = \beta \mathcal { L } _ { \mathrm { p e r t } } ^ { 1  2 } + ( 1 - \beta ) \mathcal { L } _ { \mathrm { c l e a n } } ^ { 1  2 } , } \end{array}\tag{13}
$$

where $\beta$ balances the perturbed and clean terms. We first update the parameters for the $S _ { 1 }  S _ { 2 }$ step by:

$$
\mathbf { B } _ { t } ^ { \prime } \mathbf { A } _ { t } ^ { \prime } = \mathbf { B } _ { t } \mathbf { A } _ { t } - \eta \nabla \mathcal { L } _ { \mathrm { T S P } } ^ { 1  2 } ( \mathbf { B } _ { t } \mathbf { A } _ { t } ) ,\tag{14}
$$

Table 1: Experimental results for 10 incremental tasks on ImageNet-R, ImageNet-A, CIFAR100, and CUB. We report the averaged results over 3 random seeds. Within each group, the highest results are in bold, and the second highest results are underlined. "FR" denotes the use of feature replay.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">FR</td><td colspan="2">10S-ImageNet-R</td><td colspan="2">10S-ImageNet-A</td><td colspan="2">10S-CIFAR100</td><td colspan="2">10S-CUB</td></tr><tr><td> $A _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $A _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td><td> $\boldsymbol { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td></tr><tr><td>L2P [43]</td><td>CVPR&#x27;22</td><td>x</td><td> $7 2 . 3 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $7 7 . 3 6 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $4 4 . 0 4 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $5 1 . 2 4 { \scriptstyle \pm 2 . 2 6 }$ </td><td> $8 4 . 0 6 _ { \pm 0 . 8 8 }$ </td><td> $8 8 . 2 6 _ { \pm 1 . 3 4 }$ </td><td> $6 7 . 0 2 { \scriptstyle \pm 1 . 9 0 }$ </td><td> $7 0 . 6 2 { \scriptstyle \pm 1 . 6 0 }$ </td></tr><tr><td>CODAPrompt [31]</td><td>CVPR&#x27;23</td><td>x</td><td> $7 3 . 3 1 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 8 . 4 7 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $5 2 . 0 8 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $6 3 . 9 2 _ { \pm 0 . 1 2 }$ </td><td> $8 3 . 2 1 { \scriptstyle \pm 3 . 3 9 }$ </td><td> $8 7 . 7 1 { \scriptstyle \pm 3 . 1 7 }$ </td><td> $7 7 . 2 3 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $8 1 . 9 0 { \scriptstyle \pm 0 . 8 5 }$ </td></tr><tr><td>InfLoRA [18]</td><td>CVPR&#x27;24</td><td>x</td><td> $7 4 . 7 5 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $8 0 . 6 7 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $4 9 . 2 0 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $6 0 . 9 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $8 6 . 7 5 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $9 1 . 7 2 _ { \pm 0 . 1 5 }$ </td><td> $7 0 . 8 2 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 1 . 3 9 { \scriptstyle \pm 0 . 1 4 }$ </td></tr><tr><td>PLAN [41]</td><td>ICCV&#x27;25</td><td>x</td><td> $7 5 . 2 5 _ { \pm 0 . 4 2 }$ </td><td> $8 0 . 4 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td></td><td></td><td> $8 7 . 5 4 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $9 2 . 2 1 _ { \pm 0 . 3 5 }$ </td><td>一</td><td></td></tr><tr><td>SEMA [40]</td><td>CVPR&#x27;25</td><td>x</td><td>78.00</td><td>83.56</td><td> $5 3 . 3 2 $ </td><td> $6 4 . 5 3 $ </td><td>86.98</td><td>91.37</td><td></td><td></td></tr><tr><td>SD-LoRA [45]</td><td>ICLR&#x27;25</td><td>x</td><td> $7 7 . 3 4 _ { \pm 0 . 3 5 }$ </td><td> $8 2 . 0 4 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $5 5 . 9 6 _ { \pm 0 . 7 3 }$ </td><td> $6 4 . 9 5 _ { \pm 1 . 6 3 }$ </td><td> $8 8 . 0 1 _ { \pm 0 . 3 1 }$ </td><td> $9 2 . 5 4 _ { \pm 0 . 1 8 }$ </td><td> $7 7 . 4 8 _ { \pm 0 . 2 0 }$ </td><td> $\underline { { 8 5 . 5 9 } } { \scriptstyle \pm 0 . 4 4 }$ </td></tr><tr><td>BiLoRA [57]</td><td>CVPR&#x27;25</td><td>x</td><td> $7 7 . 9 5 _ { \pm 0 . 1 4 }$ </td><td> $8 1 . 5 2 _ { \pm 0 . 2 6 }$ </td><td></td><td></td><td> $8 7 . 4 6 _ { \pm 0 . 7 6 }$ </td><td> $9 2 . 5 0 _ { \pm 0 . 6 2 }$ </td><td>一</td><td></td></tr><tr><td>LoRA-P&amp;M [28]</td><td>NIPS&#x27;25</td><td>x</td><td> $\underline { { 7 9 . 9 5 } } { \scriptstyle \pm 0 . 1 8 }$ </td><td> $\underline { { 8 5 . 2 9 } } _ { \pm 0 . 9 3 }$ </td><td> $5 6 . 5 7 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $6 5 . 3 5 { \scriptstyle \pm 1 . 8 1 }$ </td><td> $\underline { { 8 8 . 4 5 } } _ { \pm 0 . 3 5 }$ </td><td> $\underline { { 9 2 . 8 9 } } _ { \pm 1 . 1 3 }$ </td><td> $7 8 . 2 9 _ { \pm 0 . 5 0 }$ </td><td> $8 3 . 3 9 _ { \pm 0 . 6 1 }$ </td></tr><tr><td>EWC-LoRA [53]</td><td>ICLR&#x27;26</td><td>x</td><td> $7 2 . 8 6 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $7 8 . 9 5 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $5 9 . 8 9 _ { \pm 0 . 2 6 }$ </td><td> $\underline { { 6 8 . 3 3 } } { \scriptstyle \pm 0 . 6 7 }$ </td><td> $8 7 . 9 1 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $9 2 . 2 7 { \scriptstyle \pm 0 . 3 9 }$ </td><td></td><td></td></tr><tr><td>M2LE (Ours)</td><td>一</td><td>x</td><td> ${ \bf 8 1 . 3 4 { _ { \pm 0 . 1 6 } } }$ </td><td> $\mathbf { 8 6 . 5 3 _ { \pm 0 . 5 1 } }$ </td><td> ${ \bf 6 2 . 4 5 _ { \pm 1 . 3 4 } }$ </td><td> ${ \bf 7 1 . 0 2 _ { \pm 1 . 8 2 } }$ </td><td> $\mathbf { 8 9 . 9 7 _ { \pm 0 . 0 4 } }$ </td><td> $\mathbf { 9 3 . 3 1 _ { \pm 0 . 9 5 } }$ </td><td> $\mathbf { 8 1 . 4 0 _ { \pm 0 . 2 1 } }$ </td><td> $\mathbf { 8 9 . 2 3 _ { \pm 0 . 4 8 } }$ </td></tr><tr><td>SLCA [51]</td><td>ICCV&#x27;23</td><td>√</td><td> $7 9 . 3 5 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $8 3 . 2 9 _ { \pm 0 . 4 6 }$ </td><td> $6 1 . 0 5 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $6 8 . 8 8 _ { \pm 2 . 3 1 }$ </td><td> $9 1 . 2 6 _ { \pm 0 . 3 7 }$ </td><td> $9 4 . 2 9 _ { \pm 0 . 9 2 }$ </td><td> $8 4 . 6 8 _ { \pm 0 . 0 9 }$ </td><td> $9 0 . 7 7 { \scriptstyle \pm 0 . 7 9 }$ </td></tr><tr><td>SSIAT [35]</td><td>CVPR&#x27;24</td><td></td><td> $7 9 . 3 8 _ { \pm 0 . 5 9 }$ </td><td> $8 3 . 6 3 _ { \pm 0 . 4 3 }$ </td><td> $6 2 . 4 3 _ { \pm 1 . 6 3 }$ </td><td> $7 0 . 8 3 _ { \pm 1 . 6 3 }$ </td><td> $9 1 . 3 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $9 4 . 3 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $8 8 . 7 5 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 3 . 0 0 { \scriptstyle \pm 0 . 9 0 }$ </td></tr><tr><td>MOS [34]</td><td>AAAI&#x27;25</td><td>V</td><td> $7 7 . 6 8 _ { \pm 0 . 4 1 }$ </td><td> $8 2 . 0 6 _ { \pm 0 . 5 3 }$ </td><td> $5 4 . 7 5 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $_ { \textrm { -- } } ^ { 6 3 . 3 2 _ { \pm 2 . 3 8 } }$ </td><td> $9 1 . 5 3 _ { \pm 0 . 3 5 } ^ { - }$ </td><td> $9 4 . 2 1 _ { \pm 0 . 9 1 } ^ { - }$ </td><td> $8 9 . 9 7 _ { \pm 0 . 3 2 }$ </td><td> $9 3 . 4 3 _ { \pm 0 . 6 0 } ^ { - }$ </td></tr><tr><td>MACIL [44]</td><td>ICML&#x27;25</td><td>√</td><td> $\underline { { 8 1 . 8 8 } } _ { \pm 0 . 0 7 }$ </td><td> $\frac { 8 5 . 9 5 } { 8 4 5 . 4 } { \pm 0 . 2 7 }$ </td><td> $6 4 . 1 4 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $7 1 . 4 5 _ { \pm 1 . 3 5 }$ </td><td> $\mathbf { 9 1 . 9 4 _ { \pm 0 . 1 7 } }$ </td><td> $9 4 . 4 3 _ { \pm 0 . 7 9 }$ </td><td> ${ \bf 9 0 . 5 2 _ { \pm 0 . 1 3 } }$ </td><td> ${ \bf 9 3 . 9 3 _ { \pm 0 . 4 7 } }$ </td></tr><tr><td>SAEF [21]</td><td>CVPR&#x27;26</td><td>√</td><td>80.02</td><td>84.54</td><td>64.91</td><td>73.39</td><td>90.60</td><td>94.53</td><td>一</td><td></td></tr><tr><td> $\mathbf { M } ^ { 2 } \mathbf { L E } \left( \mathbf { O u r s } \right) + \mathbf { C A }$ </td><td>一</td><td>√</td><td> $\mathbf { 8 2 . 6 8 _ { \pm 0 . 3 0 } }$ </td><td> $\mathbf { 8 7 . 0 5 _ { \pm 0 . 5 1 } }$ </td><td> ${ \bf 6 5 . 9 6 _ { \pm 0 . 9 0 } }$ </td><td> ${ \bf 7 4 . 0 2 _ { \pm 0 . 8 5 } }$ </td><td> $\underline { { 9 1 . 8 6 } } _ { \pm 0 . 2 1 }$ </td><td> $\mathbf { 9 4 . 6 2 _ { \pm 0 . 9 0 } }$ </td><td> $9 0 . 3 5 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $9 3 . 9 2 _ { \pm 0 . 3 8 }$ </td></tr></table>

Table 2: Additional results under different task settings, including 5 and 20 tasks on ImageNet-R, 20 tasks on ImageNet-A, and 10 tasks on DomainNet. Within each group, the highest results are in bold, and the second highest results are underlined. $\mathbf { \ddot { \mu } } _ { \mathrm { F R } } , \mathbf { \vec { \mu } }$ denotes the use of feature replay.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td rowspan="2">FR</td><td colspan="2">5S-ImageNet-R</td><td colspan="2">20S-ImageNet-R</td><td colspan="2">20S-ImageNet-A</td><td colspan="2">10S-DomainNet</td></tr><tr><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $\boldsymbol { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td></tr><tr><td>L2P [43]</td><td>CVPR&#x27;22</td><td>x</td><td> $7 0 . 8 3 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $7 8 . 3 4 _ { \pm 0 . 4 7 }$ </td><td> $6 9 . 6 4 _ { \pm 0 . 4 2 }$ </td><td> $7 5 . 2 8 _ { \pm 0 . 5 7 }$ </td><td> $4 0 . 4 8 _ { \pm 1 . 7 8 }$ </td><td> $4 9 . 6 2 _ { \pm 1 . 4 6 }$ </td><td> $8 1 . 1 7 _ { \pm 0 . 8 3 }$ </td><td> $8 7 . 4 3 _ { \pm 0 . 9 5 }$ </td></tr><tr><td>CODAPrompt [31]</td><td> $\mathrm { C V P R } ^ { \prime } 2 3$ </td><td>×</td><td> $7 4 . 9 1 _ { \pm 0 . 3 3 }$ </td><td> $7 9 . 2 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 9 . 9 6 _ { \pm 0 . 5 0 }$ </td><td> $7 5 . 3 4 _ { \pm 0 . 8 5 }$ </td><td> $4 4 . 6 2 _ { \pm 1 . 9 2 }$ </td><td> $5 4 . 8 6 _ { \pm 0 . 5 0 }$ </td><td> $8 0 . 0 4 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $8 6 . 2 7 _ { \pm 0 . 8 2 }$ </td></tr><tr><td>InfLoRA [18]</td><td>CVPR&#x27;24</td><td>×</td><td> $7 7 . 5 2 _ { \pm 0 . 3 7 }$ </td><td> $8 2 . 0 1 _ { \pm 0 . 1 2 }$ </td><td> $7 1 . 0 1 _ { \pm 0 . 4 5 }$ </td><td> $7 7 . 2 8 _ { \pm 0 . 4 5 }$ </td><td>46.81±2.30</td><td> $5 7 . 1 3 _ { \pm 0 . 9 5 }$ </td><td> $8 1 . 4 5 _ { \pm 0 . 6 8 }$ </td><td> $8 7 . 5 5 { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td>SEMA [40]</td><td>CVPR&#x27;25</td><td>×</td><td>79.78</td><td>84.75</td><td>74.53</td><td>81.75</td><td></td><td> $^ -$ </td><td></td><td></td></tr><tr><td>SD-LoRA [45]</td><td>ICLR&#x27;25</td><td>×</td><td> $7 9 . 1 5 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $8 3 . 0 1 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $7 5 . 2 6 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $8 0 . 2 2 \substack { \pm 0 . 7 2 }$ </td><td> $4 9 . 1 2 _ { \pm 1 . 7 8 }$ </td><td> $5 9 . 6 3 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $8 1 . 0 1 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $8 6 . 8 5 { \scriptstyle \pm 0 . 4 0 }$ </td></tr><tr><td>BiLoRA [57]</td><td>CVPR&#x27;25</td><td>×</td><td> $7 8 . 3 2 _ { \pm 0 . 3 7 }$ </td><td> $8 3 . 3 1 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $7 2 . 4 1 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $7 9 . 2 8 { \scriptstyle \pm 0 . 7 6 }$ </td><td></td><td></td><td> $7 6 . 5 6 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 3 . 2 0 { \scriptstyle \pm 0 . 3 5 }$ </td></tr><tr><td>LoRA-P&amp;M [28]</td><td>NIPS&#x27;25</td><td>×</td><td> $8 1 . 4 7 _ { \pm 0 . 5 6 }$ </td><td> $8 5 . 9 6 _ { \pm 0 . 5 2 }$ </td><td> $\underline { { 7 6 . 3 7 } } { \scriptstyle \pm 0 . 0 9 }$ </td><td> $8 2 . 7 7 _ { \pm 0 . 7 1 }$ </td><td> $5 2 . 2 \overline { { 7 } } _ { \pm 1 . 8 4 }$ </td><td> $6 2 . 1 5 _ { \pm 1 . 3 3 }$ </td><td> $8 4 . 3 5 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $\underline { { 8 9 . 4 3 } } _ { \pm 0 . 2 7 }$ </td></tr><tr><td>EWC-LoRA [53]</td><td>ICLR&#x27;26</td><td>x</td><td> $7 6 . 3 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $8 1 . 4 3 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $7 0 . 1 8 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $7 7 . 0 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td></td><td>一</td><td></td><td></td></tr><tr><td>M2LE (Ours)</td><td>一</td><td>x</td><td> $\mathbf { 8 2 . 6 7 { \scriptstyle \pm 0 . 1 8 } }$ </td><td> $\mathbf { 8 7 . 3 1 { \scriptstyle \pm 0 . 3 5 } }$ </td><td> ${ \bf 7 8 . 6 2 { \scriptstyle \pm 0 . 4 3 } }$ </td><td> $\mathbf { 8 4 . 6 2 _ { \pm 0 . 2 6 } }$ </td><td> ${ \bf 5 3 . 8 7 \pm 1 . 4 5 }$ </td><td> ${ \bf 6 3 . 6 9 } _ { \pm 1 . 8 7 }$ </td><td> $\mathbf { 8 4 . 7 8 { \scriptstyle \pm 0 . 1 3 } }$ </td><td> $\mathbf { 8 9 . 7 4 { \scriptstyle \pm 0 . 4 4 } }$ </td></tr><tr><td>SLCA [51]</td><td>ICCV&#x27;23</td><td>√</td><td> $8 1 . 0 1 _ { \pm 0 . 1 1 }$ </td><td> $8 4 . 1 8 _ { \pm 0 . 2 8 }$ </td><td> $7 4 . 6 3 _ { \pm 1 . 5 5 }$ </td><td> $7 9 . 9 2 _ { \pm 1 . 2 9 }$ </td><td> $3 6 . 6 9 _ { \pm 2 1 . 3 1 }$ </td><td> $5 6 . 3 5 { \scriptstyle \pm 7 . 0 9 }$ </td><td> $8 4 . 3 5 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $8 8 . 5 6 _ { \pm 0 . 2 1 }$ </td></tr><tr><td>SSIAT [35]</td><td>CVPR&#x27;24</td><td>√</td><td> $8 0 . 5 2 _ { \pm 0 . 0 7 }$ </td><td> $8 4 . 2 5 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $7 5 . 6 7 _ { \pm 0 . 1 4 }$ </td><td> $8 2 . 3 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $5 9 . 1 6 _ { \pm 1 . 0 3 }$ </td><td> $\underline { { 6 8 . 4 5 } } _ { \pm 1 . 9 2 }$ </td><td> $8 5 . 1 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $8 9 . 8 0 _ { \pm 0 . 3 4 }$ </td></tr><tr><td>MOS [34]</td><td> $\mathbf { A A A I ^ { \prime } } 2 5$ </td><td>√</td><td> $7 8 . 7 6 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $8 2 . 3 7 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $7 5 . 1 6 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $8 0 . 5 3 { \scriptstyle \pm 0 . 7 0 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>MACIL [44]</td><td>ICML&#x27;25</td><td>√</td><td> ${ \bf 8 3 . 3 7 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $8 7 . 0 0 _ { \pm 0 . 3 5 }$ </td><td> $7 9 . 4 3 _ { \pm 0 . 3 4 }$ </td><td> $\underline { { 8 4 . 3 4 } } _ { \pm 0 . 3 2 }$ </td><td> $5 9 . 3 6 _ { \pm 1 . 2 7 }$ </td><td> $6 8 . 0 3 { \scriptstyle \pm 2 . 0 0 }$ </td><td> $8 6 . 6 2 _ { \pm 0 . 3 5 }$ </td><td> $9 0 . 8 8 _ { \pm 0 . 2 0 }$ </td></tr><tr><td> $\mathbf { M } ^ { 2 } \mathbf { L E } \left( \mathbf { O u r s } \right) + \mathbf { C A }$ </td><td>一</td><td>√</td><td> $8 2 . 7 4 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $\mathbf { 8 7 . 2 5 { \scriptstyle \pm 0 . 2 8 } }$ </td><td> ${ \bf 8 1 . 1 2 { \scriptstyle \pm 0 . 1 4 } }$ </td><td> $\mathbf { 8 6 . 1 2 { \scriptstyle \pm 0 . 3 3 } }$ </td><td> ${ \bf 6 4 . 2 5 { \scriptstyle \pm 0 . 5 6 } }$ </td><td> ${ \bf 7 2 . 5 4 { \scriptstyle \pm 0 . 9 2 } }$ </td><td> $\mathbf { 8 7 . 6 1 { \scriptstyle \pm 0 . 1 3 } }$ </td><td> ${ \bf 9 1 . 5 1 { \bf _ { \pm 0 . 1 8 } } }$ </td></tr></table>

where $\eta$ is the learning rate. We then exchange the roles of the two subsets and perform the reverse update:

$$
\mathbf { B } _ { t } ^ { \prime \prime } \mathbf { A } _ { t } ^ { \prime \prime } = \mathbf { B } _ { t } ^ { \prime } \mathbf { A } _ { t } ^ { \prime } - \eta \nabla \mathcal { L } _ { \mathrm { T S P } } ^ { 2  1 } ( \mathbf { B } _ { t } ^ { \prime } \mathbf { A } _ { t } ^ { \prime } ) ,\tag{15}
$$

where $ { \mathcal { L } } _ { \mathrm { T S P } } ^ { 2 \to 1 }$ is defined in the same way, with the roles of $S _ { 1 }$ and $S _ { 2 }$ exchanged. Thus, within each mini-batch, the optimization proceeds sequentially as $\begin{array} { r l } { \mathbf { B } _ { t } \mathbf { A } _ { t } } & { { } \xrightarrow { S _ { 1 }  S _ { 2 } } \mathbf { B } _ { t } ^ { \prime } \mathbf { A } _ { t } ^ { \prime } \xrightarrow { S _ { 2 }  S _ { 1 } } \mathbf { B } _ { t } ^ { \prime \prime } \mathbf { A } _ { t } ^ { \prime \prime } } \end{array}$ This dual training framework learns the current task update under task simulating perturbations, so that the subsequent merge better preserves previously learned knowledge.

## 5 Experiment

Datasets. We evaluate our method on five standard continual learning benchmarks: ImageNet-R [9], ImageNet-A [10], CIFAR100 [15], CUB200 [37], and DomainNet [27]. ImageNet-R contains 30,000 images from 200 classes with diverse styles, while ImageNet-A consists of 7,500 natural adversarial examples spanning 200 classes. CIFAR100 is a standard benchmark with 60,000 images from 100 classes, and CUB200 is a fine-grained bird classification dataset with 200 categories. For DomainNet, following prior works [7], we use the commonly adopted 200-class evaluation protocol.

Evaluation Metrics. Following prior PEFT-based continual learning works [35, 44], we adopt two standard evaluation metrics: (1) $\mathbf { \mathcal { A } } _ { L a s t } , \mathbf { i . e . }$ , the final accuracy over all classes after the last task, and (2) $\boldsymbol { \mathcal { A } } _ { A v g } ,$ i.e., the average accuracy across all incremental tasks.

Implementation Details. We use ViT-B/16 [4] pre-trained on ImageNet-21K [30] as the backbone. All experiments are conducted on a single NVIDIA RTX 3090 GPU. The model is optimized with SGD and a Cosine Annealing scheduler following [44]. We train each task for 20 epochs with a batch size of 48. The initial learning rate is set to 0.02 on ImageNet-R and 0.01 on the other datasets. For the HIM, the hyper-parameter λ in Eq. (9) is set to 5 for all datasets. For the TSP, $\rho$ in Eq. (11) is set to 0.3 on ImageNet-A and 0.5 on the other datasets, while $\beta$ in Eq. (13) is set to 0.6 on ImageNet-A and 0.8 on the other datasets.

Table 3: Ablation studies on ImageNet-R and ImageNet-A under 10 and 20 incremental tasks.
<table><tr><td rowspan="2"></td><td rowspan="2">Idx HIM TSP</td><td rowspan="2"></td><td colspan="2"> $\mathrm { 1 0 S \mathrm { - } I m a g e N e t \mathrm { - } R }$ </td><td colspan="2">10S-ImageNet-A</td><td colspan="2"> $2 0 \mathrm { S - I m a g e N e t - R }$ </td><td colspan="2"> $2 0 \mathrm { S - I m a g e N e t - A }$ </td></tr><tr><td> $\boldsymbol { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td><td> $\boldsymbol { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \ \hat { 1 }$  1</td><td> $\boldsymbol { A } _ { A v g } \uparrow$ </td></tr><tr><td>1234</td><td></td><td>--v√</td><td> $7 3 . 0 5 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $8 0 . 4 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $5 2 . 5 6 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $6 3 . 3 6 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $6 6 . 1 4 _ { \pm 0 . 2 0 }$ </td><td> $7 3 . 7 2 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $4 6 . 0 6 { \scriptstyle \pm 2 . 1 4 }$ </td><td> $5 9 . 4 3 _ { \pm 1 . 7 7 }$ </td></tr><tr><td></td><td></td><td></td><td> $8 0 . 3 6 _ { \pm 0 . 4 0 }$ </td><td> $8 5 . 7 8 _ { \pm 0 . 4 4 }$ </td><td> $6 1 . 3 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $7 0 . 2 3 _ { \pm 2 . 0 4 }$ </td><td> $7 8 . 0 4 _ { \pm 0 . 3 5 }$ </td><td> $8 4 . 3 1 _ { \pm 0 . 5 4 }$ </td><td> $5 2 . 8 0 _ { \pm 1 . 4 6 }$ </td><td> $6 2 . 6 4 _ { \pm 2 . 1 4 }$ </td></tr><tr><td></td><td></td><td></td><td> $7 4 . 2 0 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $8 0 . 6 7 _ { \pm 0 . 4 7 }$ </td><td> $5 6 . 5 7 { \scriptstyle \pm 2 . 3 1 }$ </td><td> $6 6 . 2 6 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $6 9 . 1 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $7 7 . 1 6 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $4 9 . 5 0 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $6 0 . 4 5 _ { \pm 1 . 5 2 }$ </td></tr><tr><td></td><td>-√-√</td><td></td><td> $\mathbf { 8 1 . 3 4 { \scriptstyle \pm 0 . 1 6 } }$ </td><td> $\mathbf { 8 6 . 5 3 { \scriptstyle \pm 0 . 5 1 } }$ </td><td> $\mathbf { 6 2 . 4 5 _ { \pm 1 . 3 4 } }$ </td><td> $\mathbf { 7 1 . 0 2 _ { \pm 1 . 8 2 } }$ </td><td> $\mathbf { 7 8 . 6 2 _ { \pm 0 . 4 3 } }$ </td><td> $\mathbf { 8 4 . 6 2 _ { \pm 0 . 2 6 } }$ </td><td> $\mathbf { 5 3 . 8 7 _ { \pm 1 . 4 5 } }$ </td><td> ${ \bf 6 3 . 6 9 _ { \pm 1 . 8 7 } }$ </td></tr></table>

Table 4: Comparison of different merge strategies on 10S-ImageNet-R and 10S-ImageNet-A.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">10S-ImageNet-R</td><td colspan="2">10S-ImageNet-A</td></tr><tr><td> $A _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $A _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td></tr><tr><td>Seq-FT</td><td> $7 3 . 0 5 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $8 0 . 4 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $5 2 . 5 6 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $6 3 . 3 6 { \scriptstyle \pm 1 . 3 2 }$ </td></tr><tr><td> $\begin{array} { r } { \mathrm { E M A - s t y l e } \left( \theta = \frac { t - 1 } { t } \theta _ { t - 1 } + \frac { 1 } { t } \theta _ { t } \right) \left[ 2 0 \right] } \end{array}$ </td><td> $7 9 . 1 4 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $8 5 . 0 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 9 . 6 4 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $6 9 . 5 2 { \scriptstyle \pm 1 . 8 2 }$ </td></tr><tr><td>LoRA-P&amp;M [28]</td><td> $7 8 . 8 9 _ { \pm 0 . 4 4 }$ </td><td> $8 4 . 9 9 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $5 9 . 9 3 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $6 9 . 8 7 { \scriptstyle \pm 2 . 0 5 }$ </td></tr><tr><td>HIM-last  $( \mathcal { J } _ { o l d } = \mathcal { L } _ { t - 1 } ( \mathbf { W } _ { t - 1 } + \Delta \mathbf { \hat { W } } _ { t } ) - \mathcal { L } _ { t - 1 } ( \mathbf { W } _ { t - 1 } ) )$ </td><td> $7 6 . 2 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $8 3 . 8 1 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $5 8 . 1 7 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $6 8 . 8 4 { \scriptstyle \pm 1 . 9 3 }$ </td></tr><tr><td>HIM (Ours)</td><td> $\mathbf { 8 0 . 3 6 _ { \pm 0 . 4 0 } }$ </td><td> $\mathbf { 8 5 . 7 8 { \scriptstyle \pm 0 . 4 4 } }$ </td><td> $\mathbf { 6 1 . 3 5 { \scriptstyle \pm 1 . 0 6 } }$ </td><td> $\mathbf { 7 0 . 2 3 _ { \pm 2 . 0 4 } }$ </td></tr></table>

Variant with classifier alignment. For fair comparison with methods that use feature replay or classifier alignment (CA), we also evaluate an enhanced variant, denoted as $\mathbf { M } ^ { 2 } \mathbf { L } \mathbf { E } { + } \mathbf { C } \mathbf { A }$ . M<sup>2</sup>LE follows the end-to-end continual learning protocol without feature replay or post-hoc classifier adjustment. M<sup>2</sup>LE+CA further applies the widely used classifier alignment strategy after each task, following SLCA [51]. We report these two variants separately to distinguish the effect of our proposed method from the additional classifier alignment stage.

## 5.1 Comparison with the State-of-the-Arts

We compare our method with representative state-of-the-art continual learning approaches from three categories, including prompt-based methods [31, 43], adapter-based methods [21, 34, 35, 40] and LoRA-based methods [18, 28, 41, 44, 45, 53, 57]. For completeness, methods that rely on feature replay or classifier alignment are reported separately from strictly rehearsal-free methods.

As shown in Tables 1 and 2, our method demonstrates clear advantages over existing PEFT-based continual learning methods. Without classifier alignment, compared with the LoRA-P&M, M<sup>2</sup>LE consistently improves $A _ { L a s t }$ across all settings, with gains of 1.39%–5.88% on the 10 task benchmarks and 0.43%–2.25% under the additional task settings. With classifier alignment, M<sup>2</sup>LE+CA achieves highly competitive performance against feature-replay-based methods, outperforming MACIL in $A _ { L a s t }$ by 0.80%–1.82% on the 10 task ImageNet-R/ImageNet-A settings and by 1.69%–4.89% on their 20 task settings. The margins are relatively smaller on CIFAR100 and CUB, where strong pre-trained representations already provide discriminative features and thus reduce the need for further backbone adaptation.

## 5.2 Ablation Studies

We conduct ablation studies on the main components of our training pipeline, including the sequential fine-tuning baseline, the proposed HIM, and TSP. As shown in Tab. 3, HIM (Idx 2) brings the main improvements over the baseline (Idx 1), increasing $A _ { L a s t }$ by 6.74%–11.90% across all four settings. This shows that reweighting the current task update under the joint loss impact objective in Eq. (4) is more effective than directly adding it to the previous inference model. TSP (Idx 3) also consistently improves the baseline, with $\mathcal { A } _ { L a s t }$ gains of 1.15%–4.01%. The improvement is particularly clear on the more challenging ImageNet-A benchmark, suggesting that TSP helps the current task update remain stable under task simulating perturbations, making it less sensitive to subsequent merge-induced changes. Notably, combining HIM and TSP (Idx 4) further improves $A _ { L a s t }$ by 0.58%–1.10% over HIM alone and achieves the best overall performance across all settings, indicating that the two components play complementary roles.

Table 5: Comparison of different perturbation strategies on 10S-ImageNet-R and 10S-ImageNet-A.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">10S-ImageNet-R</td><td colspan="2">10S-ImageNet-A</td></tr><tr><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td></tr><tr><td>w/o TSP</td><td> $8 0 . 3 6 _ { \pm 0 . 4 0 }$ </td><td> $8 5 . 7 8 _ { \pm 0 . 4 4 }$ </td><td> $6 1 . 3 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $7 0 . 2 3 _ { \pm 2 . 0 4 }$ </td></tr><tr><td>SAM [5]</td><td> $7 9 . 5 5 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $8 4 . 9 2 _ { \pm 0 . 4 4 }$ </td><td> $6 0 . 4 6 _ { \pm 1 . 5 1 }$ </td><td> $6 9 . 0 0 { \scriptstyle \pm 1 . 9 8 } $ </td></tr><tr><td>TSP (Ours)</td><td> ${ \bf 8 1 . 3 4 { _ { \pm 0 . 1 6 } } }$ </td><td> $\mathbf { 8 6 . 5 3 { \scriptstyle \pm 0 . 5 1 } }$ </td><td> ${ \bf 6 2 . 4 5 _ { \pm 1 . 3 4 } }$ </td><td> $\mathbf { 7 1 . 0 2 _ { \pm 1 . 8 2 } }$ </td></tr></table>

Table 6: Task-by-task results (%) on ImageNet-R and ImageNet-A under the 10 task setting. We compare the results with and without TSP. The forgetting metric is defined in Appendix C.1
<table><tr><td>Dataset</td><td>Metrics</td><td>|Methods |</td><td>Task 0</td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Task 4</td><td>Task 5</td><td>Task 6</td><td>Task 7</td><td>Task 8</td><td>Task 9</td></tr><tr><td rowspan="4">ImageNet-R</td><td rowspan="2">Accuracy ↑</td><td>w/o TSP w/ TSP</td><td> $9 4 . 9 6 _ { \pm 1 . 2 3 }$  95.14±1.07</td><td> $9 1 . 3 1 { \scriptstyle \pm 1 . 2 2 }$  92.14±0.86</td><td>89.48±0.20 89.93±0.85</td><td> $8 6 . 9 1 _ { \pm 0 . 6 7 }$  87.58±0.56</td><td>84.73±0.64 85.86±0.91</td><td> $8 3 . 6 8 { \scriptstyle \pm 0 . 2 0 }$  84.49±0.61</td><td> $8 3 . 0 0 { \scriptstyle \pm 0 . 6 1 }$  83.67±0.42</td><td> $8 2 . 1 3 { \scriptstyle \pm 0 . 5 3 }$  82.98±0.40</td><td>81.24±0.56 82.14±0.54</td><td> $8 0 . 3 6 _ { \pm 0 . 4 0 }$  81.34±0.16</td></tr><tr><td>Δ</td><td>+0.18</td><td> $+ 0 . 8 3$ </td><td> $+ 0 . 4 5$ </td><td> $+ 0 . { \dot { 6 } } 7 \qquad $ </td><td>+1.13</td><td>+0.81</td><td> $+ 0 . { \dot { 6 } } { \dot { 7 } }$ </td><td> $+ 0 . { \dot { 8 } } { \ddot { 5 } }$ </td><td>+0.90</td><td>+0.98</td></tr><tr><td></td><td>w/o TSP</td><td></td><td></td><td></td><td></td><td></td><td>7.34±0.20</td><td></td><td></td><td></td><td></td></tr><tr><td>Forgetting ↓</td><td>w/TSP</td><td>一 -</td><td> $4 . 9 8 _ { \pm 0 . 9 0 }$  3.17±0.60</td><td> $^ { 4 . 8 7 \pm 0 . 6 0 } _ { - }$  3.79±0.86</td><td>6.16±0.58 4.40±0.58</td><td> $7 . 0 0 { \scriptstyle \pm 0 . 5 9 }$  4.89±0.78</td><td>5.41±0.24</td><td> $7 . 2 6 _ { \pm 0 . 6 7 }$  5.46±0.62</td><td> $7 . 3 4 { \scriptstyle \pm 0 . 4 9 }$  5.49±0.88</td><td>7.43±0.44 5.52±0.82</td><td> $\stackrel { 7 . 7 6 _ { \pm 0 . 3 8 } } { - - - - \infty }$  5.77±0.64</td></tr><tr><td rowspan="6"></td><td></td><td>Δ</td><td>-</td><td>-1.81</td><td>-1.08</td><td>-1.76</td><td>-2.11</td><td>-1.93</td><td>-1.80</td><td>-1.85</td><td>-1.91</td><td>-1.99</td></tr><tr><td></td><td>w/o TSP</td><td>86.56±1.34</td><td> $8 0 . 0 6 _ { \pm 1 . 7 0 }$ </td><td>74.96±1.79</td><td>71.95±3.44</td><td></td><td></td><td></td><td></td><td></td><td>61.35±1.06</td></tr><tr><td>Accuracy ↑</td><td>w/ TSP</td><td>86.70±0.30</td><td>81.60±0.54</td><td>75.84±2.67</td><td>72.28±2.89</td><td>69.65±3.18 70.06±2.83</td><td>67.65±2.97</td><td>64.79±2.13</td><td>63.60±1.98</td><td>61.75±1.88</td><td>62.45±1.34</td></tr><tr><td></td><td>Δ</td><td>+0.14</td><td>+1.54</td><td>+0.88</td><td>+0.33</td><td>+0.41</td><td>68.60±2.70 +0.95</td><td>65.83±2.63 +1.04</td><td>64.07±1.58 +0.47</td><td>62.79±1.64 +1.04</td><td>+1.10</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>w/o TSP</td><td>一</td><td>13.54±6.63</td><td>11.43±1.91</td><td>12.49±1.06</td><td>12.71±0.85</td><td>12.27±0.15</td><td>13.88±0.72</td><td>13.47±0.44</td><td>14.19±1.52</td><td>14.01±0.90</td></tr><tr><td></td><td>Forgetting ↓</td><td>w/ TSP</td><td></td><td>10.56±0.60</td><td></td><td></td><td></td><td></td><td></td><td>10.77±0.69</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>9.19±2.12</td><td>9.98±1.49</td><td></td><td>9.24±0.91</td><td>10.52±1.26</td><td></td><td>10.49±0.86</td><td>10.04±1.29</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>9.95±1.04</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Δ</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>-2.98</td><td>-2.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>-2.51</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-2.76</td><td>-3.03</td><td>-3.36</td><td>-2.70</td><td>-3.70</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-3.97</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5.3 In-Depth Analysis

Comparison of different merge strategies. We further compare different merge strategies on 10S-ImageNet-R and 10S-ImageNet-A to analyze how the choice of merge rule affects continual learning performance. Seq-FT is used as a without merge baseline that updates the inference model by directly adding the current task LoRA update without any merge rule. As shown in Tab. 4, EMA-style and LoRA-P&M both outperform Seq-FT, confirming the benefit of merging in rehearsal-free CL.

However, these methods still rely on a scalar factor shared by all parameters, matching the limitation discussed in Sec. 3. In contrast, HIM directly solves the target merged update under Eq. (4) and obtains the element-wise solution in Eq. (9). It achieves the best results, improving $A _ { L a s t }$ over EMAstyle averaging by 1.22%/1.71% and over LoRA-P&M by 1.47%/1.42% on ImageNet-R/ImageNet-A, showing the assigning merge strengths at the element level is more effective than using a shared scalar factor. We further compare HIM with a HIM variant (HIM-last) that only uses task t − 1 in Eq. (2); without explicit constraints from earlier tasks, this variant suffers from more severe forgetting, indicating that accumulated old task information in $\mathbf { F } _ { 1 : t - 1 }$ is important for merging. More HIM variants are reported in Appendix C.3.

Analysis of perturbation training under cross-task merging. We further compare TSP with SAM [5] and without TSP to examine the effect of different perturbation training strategies during subsequent merging. As shown in Tab. 5, SAM performs worse than HIM alone. Compared with HIM alone, TSP improves $A _ { L a s t }$ by 0.98%/1.10% on ImageNet-R/ImageNet-A. To better understand this result, we visualize its behavior on both the current task and the old tasks after merging. Fig. 2 shows that SAM can still obtain a low loss solution on task 2, but the merged solution falls into a less favorable region for old tasks 0 and 1. This indicates that optimizing current task sharpness cannot well capture the cross-task interference caused by sequential LoRA updates. In contrast, TSP constructs the perturbation from one label-disjoint subset and evaluates it on the other subset. As a result, our method achieves the best performance, improving $A _ { L a s t }$ over SAM by 1.79%/1.99% on ImageNet-R/ImageNet-A.

![](images/e0d89df7cee4eec57225a958be6d19acf7222da5fe2dcf8aa1a1a671b8e9fd65.jpg)

![](images/253f581abb9966e4b1eee47ec92fbc1c4e7bb370d315c6e008dc9a9be6c041af.jpg)

![](images/abfb173e24bb6bdff5c853a698b4cdb5de8d849dceb9b76760ee011dc1bcc561.jpg)

![](images/f99b974c631f0f5481a6ce212184057c75c187cf968dc9beb43628ac218af85e.jpg)  
Figure 2: Analysis of the effect of SAM on 10S-ImageNet-R. Panels (a,b) visualize the task 2 loss landscapes for updates learned without/with SAM, while panels (c,d) show the total old task loss after merging. The dashed line denotes scalar interpolation, and the red star marks the merged solution.

Table 7: Computational and memory overhead on 10S-ImageNet-A.
<table><tr><td>Methods</td><td>GPU Memory (MiB)</td><td>Time/Batch (s)</td><td>Extra Memory (MB)</td><td> $A _ { L a s t } \mathrm { ~ \cdot ~ }$  个</td></tr><tr><td>Seq-FT</td><td>4560</td><td>0.12</td><td>0.0</td><td> $5 2 . 5 6 { \scriptstyle \pm 2 . 0 8 }$ </td></tr><tr><td>TSP</td><td>2881</td><td>0.34</td><td>0.0</td><td> $5 6 . 5 7 { \scriptstyle \pm 2 . 3 1 }$ </td></tr><tr><td>HIM</td><td>4560</td><td>0.12</td><td>54.0</td><td> $6 1 . 3 5 { \scriptstyle \pm 1 . 0 6 }$ </td></tr><tr><td>HIM w/ SAM</td><td>4568</td><td>0.26</td><td>54.0</td><td> $6 0 . 4 6 _ { \pm 1 . 5 1 }$ </td></tr><tr><td>HIM w/ TSP</td><td>2881</td><td>0.34</td><td>54.0</td><td> ${ \bf 6 2 . 4 5 { \scriptstyle \pm 1 . 3 4 } }$ </td></tr></table>

Effect of TSP on forgetting. TSP aims to reduce the sensitivity of the current task update to the parameter change introduced by the subsequent merge. As shown in Tab. 6, adding TSP consistently improves total accuracy and reduces forgetting across incremental stages on both ImageNet-R and ImageNet-A. The final forgetting decreases from 7.76% to 5.77% on ImageNet-R and from 14.01% to 10.04% on ImageNet-A. This supports our motivation that stabilizing the current task update can alleviate forgetting during continual merging. More hyper-parameter analyses and results with different backbones are provided in Appendix C.5 and Appendix C.6.

Computational and Memory Overhead. Table 7 compares the overhead of HIM and TSP on 10S-ImageNet-A. HIM requires 54.0 MB of additional Fisher storage, independent of the number of tasks, and 3.8 s per task for Fisher estimation. TSP increases the training time per batch from 0.12 s to 0.34 s, but reduces GPU memory usage from 4560 MiB to 2881 MiB. These costs are limited to training, with no additional inference overhead. Further results on the effect of LoRA on overhead and accuracy are provided in Appendix C.2.

## 5.4 Further Analysis

![](images/4c9e303494e8444531efec7e3c3bcd28cae1fd6acd55d33a42f1988af7886df5.jpg)

![](images/c99cef23deab4fa6eae3b480b4f6b67162910f8403384c5cd53e965133b2d675.jpg)

![](images/35be78ec2b5b358f11363e71978afd76ab67846e968d6c567d804d747d45d399.jpg)

![](images/0de1b7fd4af1a58911d501f5b745d7472e8a166eee3059f058233683fd1770df.jpg)  
Figure 3: Analysis of scalar merge and HIM without TSP on 10S-ImageNet-R. Panel (a) compares task 0/1 accuracy under different merge choices, and panels (b–d) show the task 0, task 1, and joint loss landscapes. The dashed line denotes scalar interpolation; the green diamond and red star mark the best scalar merge and selected HIM solution, respectively.

![](images/797dcaf2c92e5de61ab8cc58762f06b8fd9e499b94223d1f4dd384eb89229df4.jpg)

![](images/b9048b715bf6fec63273c65829fd30689780f8aee8ca1974f66e7e7f3204e1d9.jpg)

![](images/84033bd243af50c45b3fb5a59fc96897d4f263298657a6a20e81c0763fed95d8.jpg)

![](images/7b05be4501b63c4ffab076a2a733867536a9c137f5547a0d12a9db95106c7a4e.jpg)

![](images/697e6376ab16059748ed3249dca6a69a08cb28fc25e0fd1d830c951a8ed47193.jpg)  
Figure 4: Analysis of the effect of TSP on 10S-ImageNet-R. Panels (a–c) show the task 0, task 1, and joint loss landscapes after introducing TSP, corresponding to Fig. 3(b–d). Panels (d,e) isolate TSP at the task 2 merge step: from the same task 1 model, we learn and merge the task 2 update without/with TSP and visualize the joint loss over old tasks 0 and 1. The dashed line denotes scalar interpolation, and the red star marks the merged solution.

Further analysis of HIM and TSP. We further analyze the behaviors of HIM and TSP using the task 1 accuracy against task 0 accuracy and the corresponding loss landscape visualizations. For the loss landscapes, we construct a 2D plane to compare the scalar interpolation path with our merged solution. Let $\mathbf { W } _ { 1 }$ denote the model obtained by directly adding the task 1 update to $\mathbf { W } _ { 0 } ,$ , and define the scalar interpolation direction as $\mathbf { u } = \mathbf { W } _ { 1 } - \mathbf { W } _ { 0 } . \operatorname { L e t } \mathbf { d } = \mathbf { \bar { W } } _ { \mathrm { o u r s } } - \mathbf { W } _ { 0 }$ denote the displacement from $\mathbf { W } _ { 0 }$ to our selected merged solution. We then remove from d its component along u and obtain the residual direction $\begin{array} { r } { { \bf r } = { \bf d } - \frac { \langle { \bf d } , { \bf u } \rangle } { \langle { \bf u } , { \bf u } \rangle } } \end{array}$ u and $\mathbf { v } = \mathbf { r } / \lVert \mathbf { r } \rVert _ { 2 }$ , where v is orthogonal to the scalar interpolation direction. The loss is visualized on the plane $\mathbf { W } ( x , y ) = \mathbf { W } _ { 0 } + x \mathbf { u } + y \mathbf { v }$

Observation 1: HIM reaches a better solution beyond scalar interpolation. Fig. 3(a) shows that scalar merge is restricted to a interpolation path, while varying λ in our method produces solutions outside this path. This indicates that HIM provides a more flexible search space than a shared scalar factor. More importantly, the joint loss landscape in Fig. 3(d) shows that the red curve produced by varying λ mostly remains in the low loss region, rather than being restricted to the scalar path. This suggests that the merged update under the joint objective can provide more flexible balances between previous task and current task.

Observation 2: TSP improves the local stability of the merged solution. Compared with Fig. 3(b)– (d), Fig. 4(a)–(c) show that introducing TSP makes the low loss regions around the merged solution visibly wider for task 0, task 1, and the joint loss. This suggests that TSP makes the learned update less sensitive to local parameter changes before merging. To further isolate the effect of TSP in the next merge step, Fig. 4(d) and Fig. 4(e) start from the same model before learning task 2 and visualize the joint loss over old tasks 0 and 1 after merging the task 2 update without and with TSP, respectively. With TSP, the old task low loss region becomes wider, and the merged solution is located within more stable region, indicating better robustness to the parameter change introduced by the task 2 merge.

## 6 Conclusion

In this work, we show that different parameters in LoRA update may prefer different merge strengths. Motivated by this, we propose $\mathrm { { \bf M } ^ { 2 } L E }$ , which directly solves for the target merged update, yielding an analytic element-wise modulation of the current LoRA update. M<sup>2</sup>LE first derives Hessian Information-guided Adaptive Merging (HIM), an analytic element-wise modulation rule under a joint loss objective over old and new tasks. It further introduces Task Simulation Perturbation (TSP) to simulate cross-task interference during training and improve the stability during subsequent merging. Overall, M<sup>2</sup>LE offers a new view for continual learning with LoRA, showing that adapting merge strengths for different LoRA elements provides an effective way to balance stability and plasticity.

## Acknowledgments and Disclosure of Funding

This work was supported in part by the National Key R&D Program of China under Grant No.2023YFA1008600, in part by the National Natural Science Foundation of China under Grants 62576262, in part by the Natural Science Foundation Basic Research Program of Shaanxi Province under grant 2026JC-YXQN-202, in part by Scientific and Technological Innovation Teams in Shaanxi Province under Grant 2025RS-CXTD-011, in part by the Shaanxi Province Core Technology Research and Development Project under Grant 2024QY2-GJHX-11, in part by the Natural Science Basic Research Program of Shaanxi under Grant 2026SYS-SYSZD-105, in part by the Key Research and Development Program of Shaanxi under Grant 2026JC-ZDGG-42, in part by the Establishment Fund of the State Key Laboratory of Wakefulness/Sleep and Cognition (Chongqing Institute for Brain and Intelligence), in part by the Fundamental Research Funds for the Central Universities under Grant QTZX25083 and QTZX26138.

## References

[1] De Cheng, Mingyue Zeng, Zhipeng Xu, Di Xu, Nannan Wang, and Xinbo Gao. Interferenceisolated elastic weight consolidation and knowledge calibration for incremental object detection. In The Fourteenth International Conference on Learning Representations.

[2] De Cheng, Zhipeng Xu, Xinyang Jiang, Nannan Wang, Dongsheng Li, and Xinbo Gao. Disentangled prompt representation for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23595–23604, 2024.

[3] De Cheng, Zhipeng Xu, Xinyang Jiang, Dongsheng Li, Nannan Wang, and Xinbo Gao. Prompt disentanglement via language guidance and representation alignment for domain generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[5] Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. arXiv preprint arXiv:2010.01412, 2020.

[6] Takuma Fukuda, Hiroshi Kera, and Kazuhiko Kawamoto. Adapter merging with centroid prototype mapping for scalable class-incremental learning. In Proceedings of the computer vision and pattern recognition conference, pages 4884–4893, 2025.

[7] Lingfeng He, De Cheng, Huaijie Wang, Xiaofeng Zhu, Xi Yang, Nannan Wang, and Xinbo Gao. Task-driven subspace decomposition for knowledge sharing and isolation in lora-based continual learning. In Forty-third International Conference on Machine Learning, 2026.

[8] Lingfeng He, De Cheng, Di Xu, Huaijie Wang, and Nannan Wang. Harnessing textual semantic priors for knowledge transfer and refinement in clip-driven continual learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 21645–21653, 2026.

[9] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021.

[10] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262–15271, 2021.

[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.

[12] Linlan Huang, Xusheng Cao, Haori Lu, and Xialei Liu. Class-incremental learning with clip: Adaptive representation adjustment and parameter fusion. In European Conference on Computer Vision, pages 214–231. Springer, 2024.

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

[14] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013.

[15] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[16] Mei Li, Yuxiang Lu, Qinyan Dai, Suizhi Huang, Yue Ding, and Hongtao Lu. Became: Bayesian continual learning with adaptive model merging. arXiv preprint arXiv:2504.02666, 2025.

[17] Yan Li, Yuzhu Shi, Kan Zhou, Shu Zhang, Diqi He, Dingwen Zhang, and Junwei Han. Few-shot hybrid incremental learning: Continually learning under data scarcity and task uncertainty. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32334–32344, 2026.

[18] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23638–23647, 2024.

[19] Yan-Shuo Liang and Wu-Jun Li. Gated integration of low-rank adaptation for continual learning of language models. 2025.

[20] Guoliang Lin, Hanlu Chu, and Hanjiang Lai. Towards better plasticity-stability trade-off in incremental learning: A simple linear connector. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 89–98, 2022.

[21] Ruiqi Liu, Boyu Diao, Hangda Liu, Zhulin An, Fei Wang, and Yongjun Xu. From isolation to integration: Building an adaptive expert forest for pre-trained model-based class-incremental learning. arXiv preprint arXiv:2602.20911, 2026.

[22] Yue Lu, Shizhou Zhang, De Cheng, Yinghui Xing, Nannan Wang, Peng Wang, and Yanning Zhang. Visual Prompt Tuning in Null Space for Continual Learning. In Conference on Neural Information Processing Systems, 2024.

[23] Yue Lu, Shizhou Zhang, De Cheng, Guoqiang Liang, Yinghui Xing, Nannan Wang, and Yanning Zhang. Training consistent mixture-of-experts-based prompt generator for continual learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 19152–19160, 2025.

[24] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Finegrained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[25] Marc Masana, Xialei Liu, Bartłomiej Twardowski, Mikel Menta, Andrew D Bagdanov, and Joost Van De Weijer. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 5513–5533, 2022.

[26] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pages 109–165. Elsevier, 1989.

[27] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 1406–1415, 2019.

[28] Haomiao Qiu, Miao Zhang, Ziyue Qiao, and Liqiang Nie. Train with perturbation, infer after merging: A two-stage framework for continual learning. arXiv preprint arXiv:2505.22389, 2025.

[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[30] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115(3):211–252, 2015.

[31] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11909–11919, 2023.

[32] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

[33] George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. Model merging with svd to tie the knots. arXiv preprint arXiv:2410.19735, 2024.

[34] Hai-Long Sun, Da-Wei Zhou, Hanbin Zhao, Le Gan, De-Chuan Zhan, and Han-Jia Ye. Mos: Model surgery for pre-trained model-based class-incremental learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 20699–20707, 2025.

[35] Yuwen Tan, Qinhao Zhou, Xiang Xiang, Ke Wang, Yuchuan Wu, and Yongbin Li. Semanticallyshifted incremental adapter-tuning is a continual vitransformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23252–23262, 2024.

[36] Anke Tang, Enneng Yang, Li Shen, Yong Luo, Han Hu, Lefei Zhang, Bo Du, and Dacheng Tao. Merging on the fly without retraining: A sequential approach to scalable continual model merging. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[37] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

[38] Huaijie Wang, De Cheng, Lingfeng He, Yan Li, Jie Li, Nannan Wang, and Xinbo Gao. Ekpc: Elastic knowledge preservation and compensation for class-incremental learning: H. wang et al. International Journal ofComputer Vision, 134(5):238, 2026.

[39] Huaijie Wang, De Cheng, Guozhang Li, Zhipeng Xu, Lingfeng He, Jie Li, Nannan Wang, and Xinbo Gao. Stpr: Spatiotemporal preservation and routing for exemplar-free video classincremental learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=VAn2YVMuZC.

[40] Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. Self-expansion of pre-trained models with mixture of adapters for continual learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10087–10098, 2025.

[41] Xiequn Wang, Zhan Zhuang, and Yu Zhang. Plan: Proactive low-rank allocation for continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2909–2918, 2025.

[42] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European conference on computer vision, pages 631–648. Springer, 2022.

[43] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 139–149, 2022.

[44] Fangwen Wu, Lechao Cheng, Shengeng Tang, Xiaofeng Zhu, Chaowei Fang, Dingwen Zhang, and Meng Wang. Navigating semantic drift in task-agnostic class-incremental learning. arXiv preprint arXiv:2502.07560, 2025.

[45] Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng, Kede Ma, and Ying Wei. Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning. arXiv preprint arXiv:2501.13198, 2025.

[46] Zhipeng Xu, De Cheng, Xinyang Jiang, Nannan Wang, Dongsheng Li, and Xinbo Gao. Adversarial domain prompt tuning and generation for single domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18584–18595, 2025.

[47] Zhipeng Xu, De Cheng, Xinyang Jiang, Lingfeng He, Huaijie Wang, Dongsheng Li, Nannan Wang, and Xinbo Gao. Papt++: Risk-aware adversarial tuning and generation for single domain generalization. arXiv preprint arXiv:2609.04837, 2026.

[48] Zhipeng Xu, Zilong Wang, Xinyang Jiang, Dongsheng Li, De Cheng, and Nannan Wang. Reasoning-driven multimodal llm for domain generalization. arXiv preprint arXiv:2602.23777, 2026.

[49] Fanhu Zeng, Haiyang Guo, Fei Zhu, Li Shen, and Hao Tang. Robustmerge: Parameter-efficient model merging for mllms with direction robustness. arXiv preprint arXiv:2502.17159, 2025.

[50] Mingyue Zeng, De Cheng, Zhipeng Xu, Huaijie Wang, Nannan Wang, and Xinbo Gao. Symbiosis-inspired knowledge distillation for incremental object detection. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

[51] Gengwei Zhang, Liyuan Wang, Guoliang Kang, Ling Chen, and Yunchao Wei. Slca: Slow learner with classifier alignment for continual learning on a pre-trained model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19148–19158, 2023.

[52] Xiangqian Zhao, Xinyang Jiang, Zhipeng Xu, Lingfeng He, Zilong Wang, Dongsheng Li, De Cheng, and Nannan Wang. Dual-branch cross-projection debiasing through diffusion-based disentanglement. arXiv preprint arXiv:2606.24161, 2026.

[53] Yaoyue Zheng, Yin Zhang, Joost van de Weijer, Gido M van de Ven, Shaoyi Du, Xuetao Zhang, and Zhiqiang Tian. Revisiting weight regularization for low-rank continual learning. arXiv preprint arXiv:2602.17559, 2026.

[54] Da-Wei Zhou, Zi-Wen Cai, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Revisiting classincremental learning with pre-trained models: Generalizability and adaptivity are all you need. International Journal ofComputer Vision, pages 1–21, 2024.

[55] Da-Wei Zhou, Yuanhan Zhang, Yan Wang, Jingyi Ning, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Learning without forgetting for vision-language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[56] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal ofComputer Vision, 130(9):2337–2348, 2022.

[57] Hao Zhu, Yifei Zhang, Junhao Dong, and Piotr Koniusz. Bilora: almost-orthogonal parameter spaces for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25613–25622, 2025.

## A Training Pipeline

![](images/15c5b9c2a0d06d76300efc44c2004d283954fb4178b4b094f37701647cd46eb3.jpg)  
Figure 5: Overall framework of $\mathrm { M } ^ { 2 } \mathrm { L E } .$ (B) During training, TSP simulates cross task interference between label-disjoint subsets. (A) After training, HIM adaptively merges the LoRA update using historical and current-task Fisher information.

Figure 5 illustrates the overall framework of $\mathrm { M ^ { 2 } L E } ,$ and the training pipeline of $\mathbf { M } ^ { 2 } \mathrm { L E }$ is outlined in Algorithm 1. At task $t ,$ we start from the inference parameter $\mathbf { W } _ { t - 1 }$ and adapt the frozen backbone on the current data $\mathcal { D } _ { t }$ with LoRA, where the current update is $\Delta \mathbf { W } _ { t } = \mathbf { B } _ { t } \dot { \mathbf { A } } _ { t }$ . During current task training, TSP simulates cross-task interference with label-disjoint subsets, encouraging the learned update to remain stable for subsequent merging. After convergence, HIM estimates the current task Fisher $\mathbf { F } _ { t }$ and obtains the target merged update $\Delta \mathbf { W } _ { t } ^ { * }$ by Eq. (9). The merged update is then integrated into the inference model, and the Fisher statistics are accumulated for later tasks.

Algorithm 1 Training pipeline of the proposed Merging Strength Modulation via Joint Loss   
Estimation $( \mathbf { M } ^ { 2 } \mathbf { L E } )$   
Input: Task sequence $\{ \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$ , where $\mathcal { D } _ { t } = \{ ( x _ { t } ^ { i } , y _ { t } ^ { i } ) \} _ { i = 1 } ^ { n _ { t } } ;$ ViT model $f ( \cdot )$ with weight   
$\mathbf { W } _ { 0 } ;$ LoRA $\Delta \bar { \mathbf { W } } _ { t } = \bar { \mathbf { B } _ { t } } \bar { \mathbf { A } } _ { t }$ in each layer; accumulated Fisher statistics $\mathbf { F } _ { 1 : t - 1 }$ in each layer.   
Output: The adapted inference parameters $\mathbf { W } _ { t }$ for each layer in the ViT model.   
1: Initialization: $\mathbf { F } _ { 1 : t - 1 } \gets \mathbf { 0 } ^ { d _ { o } \times d _ { i } }$   
2: for task $t = 1$ to T do   
3: Freeze the inference weight $\mathbf { W } _ { t - 1 }$ and attach current-task LoRA modules.   
4: for each LoRA layer do   
5: Initialize current-task LoRA parameters ${ \bf A } _ { t }$ and $\mathbf { B } _ { t }$   
6: // During Training: Task Simulation Perturbation for Merging   
7: repeat   
8: Sample a mini-batch S from $\mathcal { D } _ { t }$ and split it into two label-disjoint subsets $S _ { 1 }$ and $S _ { 2 } ;$   
9: Update trainable parameters with the dual TSP objective in Eq. (14) and Eq. (15).   
10: until Convergence on $\mathcal { D } _ { t }$   
11: // After Training: Hessian Information-guided Adaptive Merging   
12: for each LoRA layer do   
13: Given the trained update $\Delta \mathbf { W } _ { t } ,$ estimate $\mathbf { F } _ { t }$ on $\mathcal { D } _ { t }$ and compute the HIM merged update   
$\Delta \mathbf { W } _ { t } ^ { * }$ by Eq. (9).   
14: Update the inference model weight $\mathbf { W } _ { t } \gets \mathbf { W } _ { t - 1 } + \Delta \mathbf { W } _ { t } ^ { * }$   
15: Update accumulated Fisher $\mathbf { F } _ { 1 : t } \gets \mathbf { F } _ { 1 : t - 1 } + \mathbf { F } _ { t } . ;$

## B Theoretical Analysis

Starting from Eq. (7), we solve the following quadratic problem over the candidate merged update $\Delta \hat { \mathbf { w } } _ { t } .$

$$
\begin{array} { r l } { \Delta \mathbf { w } _ { t } ^ { * } = \arg \operatorname* { m i n } _ { \Delta \hat { \mathbf { w } } _ { t } } } & { \displaystyle \sum _ { i = 1 } ^ { t - 1 } \frac { 1 } { 2 } \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { i } \Delta \hat { \mathbf { w } } _ { t } } \\ & { + \left. \frac { \lambda } { 2 } ( \Delta \hat { \mathbf { w } } _ { t } - \Delta \mathbf { w } _ { t } ) ^ { \top } \mathbf { H } _ { t } ( \Delta \hat { \mathbf { w } } _ { t } - \Delta \mathbf { w } _ { t } ) . \right. } \end{array}\tag{B.0.1}
$$

We first expand the current task term and separate the parts that depend on $\Delta \hat { \mathbf w } _ { t }$ :

$$
\begin{array} { r l } & { \quad \displaystyle \frac { \lambda } { 2 } \big ( \Delta \hat { \mathbf { w } } _ { t } - \Delta \mathbf { w } _ { t } \big ) ^ { \top } \mathbf { H } _ { t } \big ( \Delta \hat { \mathbf { w } } _ { t } - \Delta \mathbf { w } _ { t } \big ) } \\ & { = \displaystyle \frac { \lambda } { 2 } \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { t } \Delta \hat { \mathbf { w } } _ { t } - \lambda \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } + \frac { \lambda } { 2 } \Delta \mathbf { w } _ { t } ^ { \top } \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } . } \end{array}\tag{B.0.2}
$$

The last term in Eq. (B.0.2) is constant with respect to $\Delta \hat { \mathbf w } _ { t }$ and can be ignored when solving the minimization problem. Thus, the derivative of Eq. (B.0.1) with respect to $\Delta \hat { \mathbf w } _ { t }$ is:

$$
\begin{array} { r l } & { \displaystyle \frac { \partial } { \partial \Delta \hat { \mathbf { w } } _ { t } } \left[ \sum _ { i = 1 } ^ { t - 1 } \frac { 1 } { 2 } \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { i } \Delta \hat { \mathbf { w } } _ { t } + \frac { \lambda } { 2 } \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { t } \Delta \hat { \mathbf { w } } _ { t } - \lambda \Delta \hat { \mathbf { w } } _ { t } ^ { \top } \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } \right] } \\ & { \quad \quad \quad = \displaystyle \sum _ { i = 1 } ^ { t - 1 } \mathbf { H } _ { i } \Delta \hat { \mathbf { w } } _ { t } + \lambda \mathbf { H } _ { t } \Delta \hat { \mathbf { w } } _ { t } - \lambda \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } . } \end{array}\tag{B.0.3}
$$

By setting Eq. (B.0.3) to zero and denoting the solution by $\Delta \mathbf { w } _ { t } ^ { * }$ , we obtain:

$$
\sum _ { i = 1 } ^ { t - 1 } { \bf H } _ { i } \Delta { \bf w } _ { t } ^ { * } + \lambda { \bf H } _ { t } \Delta { \bf w } _ { t } ^ { * } - \lambda { \bf H } _ { t } \Delta { \bf w } _ { t } = { \bf 0 } .\tag{B.0.4}
$$

Rearranging the terms gives the linear system, which corresponds to Eq. (8):

$$
\left( \sum _ { i = 1 } ^ { t - 1 } \mathbf { H } _ { i } + \lambda \mathbf { H } _ { t } \right) \Delta \mathbf { w } _ { t } ^ { * } = \lambda \mathbf { H } _ { t } \Delta \mathbf { w } _ { t } .\tag{B.0.5}
$$

## C More Detailed Experimental Results

## C.1 Forgetting metric

We measure forgetting after task $t \geq 1$ as Forgetting<sub>t</sub> $\begin{array} { r } { \underline { { = } } \ . \frac { 1 } { t } \sum _ { i = 0 } ^ { t - 1 } \left( \operatorname* { m a x } _ { j \in \{ i , \ldots , t - 1 \} } a _ { j , i } - a _ { t , i } \right) } \end{array}$ where $a _ { j , i }$ denotes the accuracy on task i after learning task j.

## C.2 Effect of Different LoRA Ranks on M<sup>2</sup>LE Performance

We study the effect of rank in $\mathbf { M } ^ { 2 } \mathrm { L E }$ on 20S-ImageNet-R in Tab. 8. We vary the LoRA rank $r \in \{ 4 , \dot { 8 } , 1 6 , 3 2 , 6 4 \}$ and denote each variant as $\scriptstyle \mathbf { \dot { M } } ^ { 2 } \mathbf { L E - r } = r$ . Across all ranks, $\mathbf { M } ^ { 2 } \mathrm { L E }$ achieves competitive performance with only a small number of trainable parameters, ranging from 0.15M to 2.36M. It also keeps the inference overhead unchanged, with 0.00M extra parameters and 17.60 GFLOPs, since the learned components are merged into the backbone after training. For each LoRA layer, the maintained Fisher statistic $\mathbf { F } _ { 1 : t }$ has size $d _ { o } \times d _ { i }$ , so the extra storage remains 54.0MB for all ranks once the layer dimensions are fixed.

As r increases from 4 to $3 2 , A _ { L a s t }$ improves from 76.92 to 78.62, while $\mathcal { A } _ { A v g }$ increases from 83.28 to 84.62. Further increasing the rank to 64 brings only marginal improvement in $\mathcal { A } _ { A v g }$ (84.66) but slightly decreases $A _ { L a s t }$ to 78.20, together with higher training cost. Thus, we adopt $r = 3 2$ as a practical trade-off between accuracy and efficiency in the remaining experiments.

Table 8: Analysis of the effect of rank on additional parameters, storage and computational overhead, and performance on 20S-ImageNet-R. We compare our $\mathbf { M } ^ { 2 } \mathbf { I }$ E with different ranks against representative CL methods.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Venue</td><td colspan="3">Training Stage (Per Sample)</td><td colspan="2">Inference Stage (Per Sample)</td><td colspan="2">20S-ImageNet-R</td></tr><tr><td>Extra Memory</td><td>Trainable Params</td><td>Training GFLOPs</td><td>Extra Params</td><td>Inference GFLOPs</td><td> $\boldsymbol { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { A } _ { A v g }$ </td></tr><tr><td colspan="9">Methods based on the ViT-21K model (86M).</td></tr><tr><td>CODA [31]</td><td>CVPR-23</td><td>15.4MB</td><td>3.84M (4.46%)</td><td>35.17</td><td>3.84M (4.46%)</td><td>35.17</td><td> $6 9 . 9 6 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 5 . 3 4 { \scriptstyle \pm 0 . 8 5 }$ </td></tr><tr><td>SLCA [51]</td><td>ICCV-23</td><td>450.6MB</td><td>86M (100%)</td><td>17.60</td><td>0.00M (0.00%)</td><td>17.60</td><td> $7 4 . 6 3 { \scriptstyle \pm 1 . 5 5 }$ </td><td> $7 9 . 9 2 _ { \pm 1 . 2 9 }$ </td></tr><tr><td>SSIAT [35]</td><td>CVPR-24</td><td>450.6MB</td><td>1.19M (1.38%)</td><td>17.81</td><td>1.19M (1.38%)</td><td>17.81</td><td> $7 5 . 6 7 _ { \pm 0 . 2 4 }$ </td><td> $8 2 . 3 0 { \scriptstyle \pm 0 . 3 6 }$ </td></tr><tr><td>InfLoRA [18]</td><td>CVPR-24</td><td>11.3MB</td><td>0.37M (0.43%)</td><td>17.67</td><td>0.00M (0.00%)</td><td>17.67</td><td> $7 1 . 0 1 _ { \pm 0 . 4 5 }$ </td><td> $7 7 . 2 8 _ { \pm 0 . 4 5 }$ </td></tr><tr><td>VPT-NSP [22]</td><td>NIPS-24</td><td>28.4MB</td><td>0.09M (0.10%)</td><td>17.72</td><td>0.09M (0.10%)</td><td>17.72</td><td> $7 5 . 6 9 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $8 1 . 8 7 _ { \pm 0 . 5 9 }$ </td></tr><tr><td>SD-LoRA [45]</td><td>ICLR-25</td><td>14.1MB</td><td>0.37M (0.43%)</td><td>20.51</td><td>0.00M (0.00%)</td><td>17.72</td><td> $7 5 . 2 6 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $8 0 . 2 2 { \scriptstyle \pm 0 . 7 2 }$ </td></tr><tr><td>VPT-CPG [23]</td><td>AAAI-25</td><td>28.4MB</td><td>1.66M (1.93%)</td><td>17.72</td><td>1.66M (1.93%)</td><td>17.72</td><td> $7 5 . 3 3 _ { \pm 0 . 3 7 }$ </td><td> $8 0 . 2 2 _ { \pm 0 . 7 2 }$ </td></tr><tr><td>M2LE-r=4 (Ours)</td><td></td><td>54.0MB</td><td>0.15M (0.17%)</td><td>17.59</td><td>0.00M (0.00%)</td><td>17.60</td><td> $\mathbf { 7 6 . 9 2 { \scriptstyle \pm 0 . 2 0 } }$ </td><td> $\mathbf { 8 3 . 2 8 { \scriptstyle \pm 0 . 2 5 } }$ </td></tr><tr><td>M2LE-r=8 (Ours)</td><td></td><td>54.0MB</td><td>0.29M (0.34%)</td><td>17.62</td><td>0.00M (0.00%)</td><td>17.60</td><td> ${ \bf 7 7 . 3 2 _ { \pm 0 . 5 5 } }$ </td><td> $\mathbf { 8 3 . 6 8 _ { \pm 0 . 5 1 } }$ </td></tr><tr><td> $\scriptstyle \mathbf { M } ^ { 2 } \mathbf { L E - r = 1 6 } ( \mathbf { O u r s } )$ </td><td></td><td>54.0MB</td><td>0.59M (0.69%)</td><td>17.68</td><td>0.00M (0.00%)</td><td>17.60</td><td> ${ \bf 7 8 . 0 1 _ { \pm 0 . 1 3 } }$ </td><td> $\mathbf { 8 4 . 0 7 _ { \pm 0 . 4 3 } }$ </td></tr><tr><td> $\mathbf { M } ^ { 2 } \mathbf { L E - r = } 3 2 \ ( \mathbf { O u r s } )$ </td><td></td><td>54.0MB</td><td>1.18M (1.37%)</td><td>17.80</td><td>0.00M (0.00%)</td><td>17.60</td><td> $\mathbf { 7 8 . 6 2 _ { \pm 0 . 4 3 } }$ </td><td> $\mathbf { 8 4 . 6 2 _ { \pm 0 . 2 6 } }$ </td></tr><tr><td> $\mathbf { M ^ { 2 } L E { - } r { = } 6 4 \ ( O u r s ) }$ </td><td></td><td>54.0MB</td><td>2.36M (2.74%)</td><td>18.03</td><td>0.00M (0.00%)</td><td>17.60</td><td> ${ \bf 7 8 . 2 0 _ { \pm 0 . 3 0 } }$ </td><td> $\mathbf { 8 4 . 6 6 _ { \pm 0 . 7 0 } }$ </td></tr></table>

Table 9: Comparison of different HIM variants on 10S-ImageNet-R and 10S-ImageNet-A.
<table><tr><td rowspan="2">Variant</td><td colspan="2">10S-ImageNet-R</td><td colspan="2">10S-ImageNet-A</td></tr><tr><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td><td> $\mathcal { A } _ { L a s t } \uparrow$ </td><td> $\boldsymbol { \mathcal { A } _ { A v g } } \uparrow$ </td></tr><tr><td>HIM-last (k = 1)</td><td> $7 6 . 2 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $8 3 . 8 1 _ { \pm 0 . 6 1 }$ </td><td> $5 8 . 1 7 { \scriptstyle \pm 2 . 3 8 }$ </td><td> $6 8 . 8 4 { \scriptstyle \pm 1 . 9 3 }$ </td></tr><tr><td> $\mathrm { H I M } - k = 3$ </td><td> $7 8 . 8 8 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $8 5 . 3 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $6 0 . 8 1 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $6 9 . 9 7 _ { \pm 2 . 2 8 }$ </td></tr><tr><td> $\mathrm { H I M } - k = 5$ </td><td> $8 0 . 0 4 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $8 5 . 7 3 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $6 0 . 9 2 _ { \pm 0 . 9 7 }$ </td><td> $7 0 . 1 4 { \scriptstyle \pm 2 . 4 8 }$ </td></tr><tr><td> $\mathrm { H I M } - k = 7$ </td><td> $8 0 . 2 1 _ { \pm 0 . 4 5 }$ </td><td> $8 5 . 7 7 _ { \pm 0 . 4 5 }$ </td><td> $6 1 . 1 6 _ { \pm 1 . 1 5 }$ </td><td> $7 0 . 1 7 _ { \pm 2 . 4 2 }$ </td></tr><tr><td>HIM-full (Ours)</td><td> $\mathbf { 8 0 . 3 6 _ { \pm 0 . 4 0 } }$ </td><td> $\mathbf { 8 5 . 7 8 { \scriptstyle \pm 0 . 4 4 } }$ </td><td> $\mathbf { 6 1 . 3 5 { \scriptstyle \pm 1 . 0 6 } }$ </td><td> $\mathbf { 7 0 . 2 3 _ { \pm 2 . 0 4 } }$ </td></tr></table>

## C.3 Analysis of HIM Variants

We further analyze several HIM variants that retain different numbers of past tasks in the old task term. For a more explicit comparison, we define a truncated HIM variant by retaining only the most recent k past tasks in the old task loss impact:

$$
\mathcal { I } _ { \mathrm { o l d } } ^ { ( k ) } ( \Delta { \mathbf { \hat { W } } _ { t } } ) = \sum _ { i = \operatorname* { m a x } ( 1 , t - k ) } ^ { t - 1 } \left( \mathcal { L } _ { i } ( \mathbf { W } _ { t - 1 } + \Delta { \mathbf { \hat { W } } _ { t } } ) - \mathcal { L } _ { i } ( \mathbf { W } _ { t - 1 } ) \right) .\tag{C.3.1}
$$

Here, $k \in \{ 1 , 3 , 5 , 7 \}$ controls the number of past tasks used in the old task term. Correspondingly, this variant replaces the accumulated Fisher statistic in Eq. (9) with $\begin{array} { r } { \mathbf { F } _ { t - k : t - 1 } = \sum _ { i = \operatorname* { m a x } ( 1 , t - k ) } ^ { t - 1 } \mathbf { F } _ { i } } \end{array}$ while full HIM uses $\begin{array} { r } { \mathbf { F } _ { 1 : t - 1 } = \sum _ { i = 1 } ^ { t - 1 } \mathbf { F } _ { i } } \end{array}$ . When $k = 1$ , the variant uses only task t − 1, corresponding to the HIM-last setting discussed in the main text.

As shown in Tab. 9, HIM-last gives much lower performance, i.e., 76.23/83.81 on 10S-ImageNet-R and 58.17/68.84 on 10S-ImageNet-A. When more past tasks are retained, the performance consistently improves, e.g., HIM-k = 5 reaches 80.04/85.73 and 60.92/70.14 on the two datasets, respectively. Full HIM achieves the best results of 80.36/85.78 and 61.35/70.23, indicating that using a broader set of past tasks provides more complete constraints for preserving previously learned knowledge during merging. These results also validate the use of $\mathbf { F } _ { 1 : t - 1 }$ in HIM, which aggregates information from all past tasks rather than relying only on recent ones.

## C.4 Analysis of Historical Fisher Reuse

We further analyze the accumulated Fisher statistics in HIM on 10S-ImageNet-R. HIM estimates each task’s Fisher at its pre-merge solution while the data are available, and retains only the accumulated statistic $F _ { 1 : t - 1 }$ . For comparison, Recomputed Fisher revisits previous task data before each merge and re-estimates their accumulated Fisher at its current merged model. The recomputed statistic is used for merging and subsequent learning, so the two methods follow their respective model trajectories. We measure their Fisher similarity by $\begin{array} { r } { S _ { t } = \frac { 1 } { | \mathcal { L } | } \sum _ { \ell \in \mathcal { L } } \cos ( F _ { 1 : t - 1 } ^ { \ell } , \widetilde { F } _ { 1 : t - 1 } ^ { \ell } ) } \end{array}$ , where L denotes the set of layers with Fisher statistics, and $F _ { 1 : t - 1 } ^ { \ell }$ and $\widetilde { F } _ { 1 : t - 1 } ^ { \ell }$ are the accumulated statistics from HIM and Recomputed Fisher, respectively.

Table 10: Task-by-task comparison of accumulated historical Fisher and recomputed Fisher on 10S-ImageNet-R. Results are obtained with a single random seed.
<table><tr><td>Metric</td><td>Task 0</td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Task 4</td><td>Task 5</td><td>Task 6</td><td>Task 7</td><td>Task 8</td><td>Task 9</td></tr><tr><td>St ↑</td><td></td><td>1.000</td><td>0.988</td><td>0.986</td><td>0.986</td><td>0.987</td><td>0.988</td><td>0.987</td><td>0.987</td><td>0.986</td></tr><tr><td>HIM  $\mathcal { A } _ { L a s t } \uparrow$ </td><td>96.66</td><td>92.81</td><td>89.41</td><td>87.39</td><td>84.69</td><td>83.84</td><td>83.18</td><td>82.62</td><td>81.68</td><td>80.67</td></tr><tr><td>Recomputed Fisher  $A _ { L a s t } \uparrow$ </td><td>96.66</td><td>92.81</td><td>89.36</td><td>87.11</td><td>84.76</td><td>83.70</td><td>83.15</td><td>82.70</td><td>81.68</td><td>80.67</td></tr></table>

As shown in Tab. 10, $S _ { t }$ remains at least 0.986 across the evaluated stages, indicating similar directions of the accumulated Fisher statistics. Recomputed Fisher achieves similar accuracy to HIM, with a maximum difference of 0.28 performance and the same final $A _ { L a s t }$ of 80.67%. These results support using the accumulated historical Fisher in HIM, avoiding repeated access to previous task data with little impact on performance over the evaluated task sequence.

## C.5 Hyper-Parameter Analysis

![](images/332bdc52999bd8e5a1c9993c7cacfe8882ce0c079361508190c9dc1003a5c9d5.jpg)

![](images/adcd987f231f5b685be0e884e498090cff9e53e2466324eeedc444793b551ae1.jpg)

![](images/4922b57eb7b9a292e9ecaab831b3d76b6f9cd77b1c13eea1204f046af0c3f09d.jpg)  
Figure 6: Hyper-parameter analysis on 10S-ImageNet-R. From left to right, λ, $\rho ,$ and $\beta$ are varied while the other two are fixed. All results are averaged over three random seeds, and error bars denote the standard deviation.

![](images/797f6c87bde3f12c8bd03e23205fa860928614aeae4505f91d5d886975efe93f.jpg)

![](images/cdc67b344363297ce0515917cf11ed6cdb301dc0a0bc0f77f653c537991719ec.jpg)

![](images/0f3dfd6233186feb51739d12ba606d8f8f460c447b654d1b7402ec383ff6e394.jpg)  
Figure 7: Same as Fig. 6, but on 10S-ImageNet-A.

We conduct experiments to investigate how the key hyper-parameters $\lambda , \rho ,$ and $\beta$ affect performance on 10S-ImageNet-R and 10S-ImageNet-A. Here, λ is the trade-off coefficient in the merge objective in Eq. (9), ρ controls the perturbation magnitude in Eq. (11), and β balances the perturbed and clean losses in Eq. (13). We use the default settings $\lambda = 5 , \rho = 0 . 5$ , and $\beta = 0 . 8$ on ImageNet-R, and $\lambda = 5 , \rho = 0 . 3 .$ , and $\beta = 0 . 6$ on ImageNet-A. For each analysis, we vary only one hyper-parameter while fixing the other two at their default values. The results are shown in Fig. 6 and Fig. 7. All results are averaged over three random seeds, and we report the mean with error bars.

Impact of λ. The hyper-parameter λ controls the relative weight of the current task term in the merge objective, and thus adjusts the balance between old task retention and current task adaptation. A too-small λ ${ \mathfrak { i } } ( { \mathfrak { e } } . { \mathfrak { g } } . , \lambda = 1 )$ underweights the current task term and limits new task adaptation, whereas an overly large $\lambda \left( \mathrm { e . g . , } \lambda \geq 1 0 \right)$ makes the merged update overly biased toward the current task and weakens old task retention. Overall, the results remain stable for $1 \leq \lambda \leq 1 0$ , with the best trade-off achieved around $\lambda = 3 – 5$ . Based on this observation, we set $\lambda = 5$

Impact of $\rho _ { \bullet }$ The hyper-parameter $\rho$ controls the perturbation magnitude in the TSP. A too-small $\rho \left( \mathbf { e . g . } , \rho = 0 . 1 \right)$ leads to a weak cross-subset probing effect, whereas a too-large $\rho \left( \mathbf { e . g . } , \rho \geq 0 . 8 \right)$ over-amplifies the perturbation. Overall, the results remain stable for $0 . 1 \leq \rho \leq \mathrm { 1 . 0 }$ , with the best performance achieved at moderate values. Accordingly, we set $\rho = 0 . 5$ on ImageNet-R and $\rho = 0 . 3$ on ImageNet-A.

Impact of $\beta .$ The hyper-parameter $\beta$ controls the relative weight of the perturbed loss and the clean loss in the TSP. A too-small $\beta ( { \bf e . g . } , \beta = 0 . 2 )$ underweights the perturbation signal, whereas a too-large $\beta ( { \bf e . g . } , \beta = 1 . 0 )$ makes the perturbation effect overly strong and is less favorable overall. Overall, the results remain stable for $\bar { 0 } . 2 \leq \beta \leq 1 . 0$ , and the best results are obtained at moderate values. Based on the results, we set $\beta = 0 . 8$ on ImageNet-R and $\beta = 0 . 6$ on ImageNet-A.

## C.6 Experimental Results Using the CLIP Model as the Backbone

Table 11: Experimental results on ImageNet-R, CIFAR100 and CUB datasets using CLIP as the backbone. All methods are initialized with the CLIP pre-trained on LAION-400M without experience replay for fair comparisons. PROOF<sup>†</sup> represents our reproduced results of PROOF [55] without experience replay. The best results are in bold and the second highest results are underlined.
<table><tr><td rowspan="3">Method</td><td colspan="4">ImageNet-R</td><td colspan="4">CIFAR100</td><td colspan="4">CUB</td></tr><tr><td colspan="2">B0 Inc10</td><td colspan="2">B50 Inc10</td><td colspan="2">B0 Inc10</td><td colspan="2">B50 Inc10</td><td colspan="2">B0 Inc20</td><td colspan="2">B100 Inc20</td></tr><tr><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td></tr><tr><td>CoOp [56]</td><td>37.52</td><td>60.73</td><td>39.77</td><td>54.20</td><td>24.24</td><td>47.00</td><td>24.12</td><td>41.23</td><td>8.57</td><td>27.61</td><td>10.14</td><td>24.03</td></tr><tr><td>ZS-CLIP [29]</td><td>77.17</td><td>83.37</td><td>77.17</td><td>79.57</td><td>71.38</td><td>81.81</td><td>71.38</td><td>76.49</td><td>63.06</td><td>74.38</td><td>63.06</td><td>67.96</td></tr><tr><td>L2P [43]</td><td>66.52</td><td>75.97</td><td>66.77</td><td>72.82</td><td>73.03</td><td>82.74</td><td>73.61</td><td>81.14</td><td>57.93</td><td>70.87</td><td>66.12</td><td>75.64</td></tr><tr><td>DualPrompt [42]</td><td>66.65</td><td>76.21</td><td>67.58</td><td>73.22</td><td>72.44</td><td>81.63</td><td>72.57</td><td>80.12</td><td>57.46</td><td>69.89</td><td>64.84</td><td>71.40</td></tr><tr><td>CODA-Prompt [31]</td><td>68.95</td><td>77.69</td><td>68.05</td><td>73.71</td><td>73.43</td><td>82.43</td><td>71.58</td><td>78.69</td><td>62.98</td><td>73.12</td><td>62.21</td><td>73.95</td></tr><tr><td>SimpleCIL [54]</td><td>74.48</td><td>81.06</td><td>74.48</td><td>76.84</td><td>76.63</td><td>84.15</td><td>76.63</td><td>80.20</td><td>77.52</td><td>83.81</td><td>77.52</td><td>79.75</td></tr><tr><td>RAPF [12]</td><td>79.62</td><td>86.28</td><td>79.21</td><td>84.03</td><td>79.04</td><td>86.19</td><td>78.25</td><td>82.35</td><td>62.77</td><td>79.09</td><td>62.93</td><td>72.82</td></tr><tr><td>PROOF† [55]</td><td>77.25</td><td>82.69</td><td>77.03</td><td>80.56</td><td>76.29</td><td>84.88</td><td>75.93</td><td>81.85</td><td>72.60</td><td>79.94</td><td>71.80</td><td>76.43</td></tr><tr><td>M2LE (Ours)</td><td>82.05</td><td>87.16</td><td>82.65</td><td>85.24</td><td>80.52</td><td>87.88</td><td>83.12</td><td>87.80</td><td>80.03</td><td>86.92</td><td>80.41</td><td>84.25</td></tr></table>

Table 12: Experimental results on Aircraft, Cars and UCF datasets using CLIP as the backbone. All methods are initialized with the CLIP pre-trained on LAION-400M without experience replay for fair comparisons. $\mathrm { P R O O F ^ { \dagger } }$ represents our reproduced results of PROOF [55] without experience replay. The best results are in bold and the second highest results are underlined.
<table><tr><td rowspan="3">Method</td><td colspan="4">Aircraft</td><td colspan="4"> $\mathbf { C a r s }$ </td><td colspan="4">UCF</td></tr><tr><td colspan="2">B0 Inc10</td><td colspan="2">B50 Inc10</td><td colspan="2">B0 Inc10</td><td colspan="2">B50 Inc10</td><td colspan="2">B0 Inc20</td><td colspan="2">B100 Inc20</td></tr><tr><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td><td> $A _ { L a s t }$ </td><td> $\mathcal { A } _ { A v g }$ </td></tr><tr><td>CoOp [56]</td><td>7.14</td><td>14.54</td><td>7.77</td><td>13.05</td><td>21.65</td><td>36.46</td><td>20.87</td><td>37.40</td><td>33.46</td><td>47.85</td><td>24.74</td><td>42.02</td></tr><tr><td>ZS-LIP [29]</td><td>17.22</td><td>26.66</td><td>17.22</td><td>21.70</td><td>76.37</td><td>82.60</td><td>76.37</td><td>78.32</td><td>67.64</td><td>75.50</td><td>67.64</td><td>71.44</td></tr><tr><td>L2P [43]</td><td>28.29</td><td>47.19</td><td>32.13</td><td>44.07</td><td>61.82</td><td>76.63</td><td>65.64</td><td>76.37</td><td>76.43</td><td>86.34</td><td>76.62</td><td>83.95</td></tr><tr><td>DualPrompt [42]</td><td>25.83</td><td>44.30</td><td>33.57</td><td>46.07</td><td>62.94</td><td>76.26</td><td>67.55</td><td>76.88</td><td>75.82</td><td>85.21</td><td>76.35</td><td>84.31</td></tr><tr><td>CODA-Prompt [31]</td><td>27.69</td><td>45.98</td><td>32.28</td><td>45.14</td><td>66.47</td><td>80.21</td><td>64.19</td><td>75.06</td><td>80.14</td><td>87.76</td><td>75.03</td><td>83.04</td></tr><tr><td>SimpleCIL [54]</td><td>48.09</td><td>59.24</td><td>48.09</td><td>53.05</td><td>86.85</td><td>92.04</td><td>86.85</td><td>88.96</td><td>85.68</td><td>90.44</td><td>85.68</td><td>88.12</td></tr><tr><td>RAF [12]</td><td>23.61</td><td>50.38</td><td>25.44</td><td>40.47</td><td>62.85</td><td>82.89</td><td>63.19</td><td>75.87</td><td>80.33</td><td>92.28</td><td>81.55</td><td>90.31</td></tr><tr><td>PROOF† [55]</td><td>39.03</td><td>47.33</td><td>45.36</td><td>54.35</td><td>84.06</td><td>86.87</td><td>82.66</td><td>88.68</td><td>81.47</td><td>87.04</td><td>81.85</td><td>86.15</td></tr><tr><td>M2LE (Ours)</td><td>60.43</td><td>70.63</td><td>63.85</td><td>69.50</td><td>90.57</td><td>94.39</td><td>90.91</td><td>92.40</td><td>90.53</td><td>94.38</td><td>91.28</td><td>94.81</td></tr></table>

To examine whether $\mathbf { M } ^ { 2 } \mathrm { L E }$ can be transferred beyond a single visual backbone, we further conduct experiments with CLIP as the pre-trained model. The results on six benchmarks are summarized in Tab. 11 and Tab. 12.

Datasets and Evaluation Metrics. Following existing CLIP-based continual learning studies [12, 55], we evaluate $\mathbf { M } ^ { 2 } \mathrm { L E }$ on ImageNet-R [9], CIFAR100 [15], CUB [37], Aircraft [24], Cars [14], and UCF [32]. ImageNet-R contains 200 ImageNet categories with different artistic renditions, such as sketches, cartoons, and paintings. CIFAR100 includes 100 object categories with 60,000 natural images. CUB, Aircraft, and Cars are fine-grained recognition datasets, covering bird species, aircraft model variants, and car categories, respectively. UCF101 is an action recognition benchmark with 101 human action classes. For Aircraft, Cars, and UCF, we follow prior protocols and sample 100 classes for continual learning evaluation. The split notation $\mathrm { \bf \ddot { \tau } B - X \ I n c - y ^ { , } }$ denotes that the first stage contains x classes and each later stage introduces $y$ new classes. When $x = 0$ , all tasks contain the same number of classes.

Implementation Details of $\mathbf { M } ^ { 2 } \mathbf { L E }$ based on CLIP. We build $\mathbf { M } ^ { 2 } \mathbf { I }$ LE upon SimpleCIL [54], a strong training-free baseline that freezes the pre-trained model and constructs a prototypical classifier from downstream features. Specifically, we keep one frozen CLIP branch $f _ { \mathrm { p t } } ( \cdot )$ together with its prototype classifier $\mathbf { W } _ { \mathrm { p t } }$ , and sequentially adapt another CLIP branch $f _ { \mathrm { f t } } ( \cdot )$ using $\mathbf { M } ^ { 2 } \mathrm { L E }$ . The adapted branch is trained with the CLIP-style image–text matching objective [55], where the text classifier is denoted as $\mathbf { W } _ { \mathrm { t e x t } }$ . At inference time, we combine the predictions from the frozen branch and the adapted branch:

$$
\mathbf { p } = \alpha \mathbf { W } _ { \mathrm { p t } } ^ { \top } f _ { \mathrm { p t } } ( \mathbf { x } ) + ( 1 - \alpha ) \mathbf { W } _ { \mathrm { t e x t } } ^ { \top } f _ { \mathrm { f t } } ( \mathbf { x } ) .\tag{C.6.1}
$$

Comparison Methods. We compare $\mathbf { M } ^ { 2 } \mathbf { I }$ LE with a wide range of CLIP-based continual learning baselines, including prompt-based methods CoOp [56], L2P [43], DualPrompt [42], and CODA-Prompt [31]; CLIP-oriented methods RAPF [12] and PROOF [55]; the training-free SimpleCIL [54]; and the zero-shot CLIP baseline [29]. For fair comparison, all methods use the CLIP backbone pre-trained on LAION-400M, and no experience replay is used.

Results and Analysis. As shown in Tab. 11 and Tab. 12, $\mathbf { M } ^ { 2 } \mathrm { L E }$ achieves the best performance across all CLIP-based settings, showing that it can generalize well to vision-language pre-trained models. Some adaptation baselines even perform worse than ZS-CLIP or SimpleCIL, indicating that directly updating CLIP may disturb its pre-trained knowledge. In contrast, $\scriptstyle \mathrm { { \bf { \dot { M } } } ^ { 2 } L E }$ enables effective layer-wise continual adaptation through adaptive LoRA merging: HIM assigns element-wise merge strengths to better balance old and new knowledge, while TSP improves the stability of the learned update for later merging. Moreover, $\mathbf { M } ^ { 2 } \mathrm { L E }$ is replay-free and does not require class-wise feature statistics, making it efficient and scalable as the label space grows.

## D Limitations

Although $\mathbf { M } ^ { 2 } \mathrm { L E }$ shows strong performance across diverse continual learning benchmarks, its effectiveness still depends on the quality and adaptability of the underlying pre-trained backbone. When the backbone provides less transferable representations or is not well suited to low-rank updates, the benefit of LoRA-based continual adaptation may be reduced.

## E Potential Impact

$\mathbf { M } ^ { 2 } ]$ LE achieves efficient continual adaptation by updating only low-rank trainable parameters, which makes it suitable for long task sequences. However, this efficiency-oriented design may not always be optimal for accuracy-critical applications, such as medical imaging or other safety-sensitive scenarios. In such cases, practitioners should carefully evaluate whether the resulting performance meets the precision requirements of the target domain.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the research objectives, methods, and contributions, all of which are consistently elaborated and supported throughout the paper and supplementary materials.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please see Appendix D.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: The paper provides clearly stated assumptions and complete derivations for the theoretical results. The corresponding formulas and proofs are presented with proper references, and additional details are included in Appendix when necessary.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Our method and experimental setup are thoroughly described in Sec.5 to ensure reproducibility. Code for reproducing the main results is provided in the supplementary material.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide the code in the supplementary material.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Please see Datasets, Evaluation Metrics and Implementation Details in Sec.5.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our experiments use three random seeds, except for the analysis in Table 10, which uses a single seed. We report mean performance for the three seed experiments, and we report the average performance with standard deviation.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the information of our compute resources in Appendix.C.2.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: Our research complies the NeurIPS Code of Ethics in all aspects.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Please see Appendix E.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: Our paper poses no such risks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The existing assets covered in the paper are subject to their license and terms of use, and are explicitly cited and described.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We provide the code in the supplementary material.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: No, our paper does not involve the use of large language models (LLMs) as a component of the core methodology.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.